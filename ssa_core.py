"""
══════════════════════════════════════════════════════════════
  SSA CORE LIBRARY v5.1
══════════════════════════════════════════════════════════════
"""
import numpy as np
import pandas as pd
from scipy.stats import shapiro, jarque_bera
from scipy.signal import periodogram
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
from statsmodels.stats.diagnostic import acorr_ljungbox
import warnings
warnings.filterwarnings('ignore')


class SSA:
    def __init__(self, time_series, window_length='auto', name='Time Series'):
        self.original = np.array(time_series, dtype=float).flatten()
        self.N = len(self.original)
        self.name = name
        self.L = self.N // 2 if window_length == 'auto' else int(window_length)
        self.K = self.N - self.L + 1
        assert 2 <= self.L <= self.N // 2 + 1
        self._embed(); self._decompose()

    def _embed(self):
        self.trajectory_matrix = np.column_stack(
            [self.original[i:i+self.L] for i in range(self.K)])

    def _decompose(self):
        S = self.trajectory_matrix @ self.trajectory_matrix.T
        eigenvalues, eigenvectors = np.linalg.eigh(S)
        idx = np.argsort(eigenvalues)[::-1]
        self.eigenvalues = np.maximum(eigenvalues[idx], 0)
        self.singular_values = np.sqrt(self.eigenvalues)
        self.U = eigenvectors[:, idx]
        self.d = int(np.sum(self.singular_values > 1e-10))
        self.V = np.zeros((self.K, self.d))
        self.elementary_matrices = []
        for i in range(self.d):
            vi = self.trajectory_matrix.T @ self.U[:, i] / self.singular_values[i]
            self.V[:, i] = vi
            self.elementary_matrices.append(
                self.singular_values[i] * np.outer(self.U[:, i], vi))
        total_var = np.sum(self.eigenvalues[:self.d])
        self.contribution = self.eigenvalues[:self.d] / total_var * 100
        self.cumulative_contribution = np.cumsum(self.contribution)

    def _diagonal_averaging(self, matrix):
        L, K = matrix.shape; N = L + K - 1
        result = np.zeros(N); counts = np.zeros(N)
        for i in range(L):
            for j in range(K):
                result[i+j] += matrix[i, j]; counts[i+j] += 1
        return result / counts

    def reconstruct_component(self, index):
        return self._diagonal_averaging(self.elementary_matrices[index])

    # ── W-Correlation ────────────────────────────────────────
    def w_correlation(self, num_components=None):
        if num_components is None: num_components = min(self.d, 20)
        num_components = min(num_components, self.d)
        components = [self.reconstruct_component(i) for i in range(num_components)]
        weights = np.zeros(self.N)
        Ls = min(self.L, self.K); Ks = max(self.L, self.K)
        for i in range(self.N):
            if i < Ls-1:   weights[i] = i+1
            elif i < Ks:   weights[i] = Ls
            else:          weights[i] = self.N-i
        wc = np.zeros((num_components, num_components))
        for i in range(num_components):
            for j in range(num_components):
                wi = np.sqrt(np.sum(weights * components[i]**2))
                wj = np.sqrt(np.sum(weights * components[j]**2))
                if wi > 0 and wj > 0:
                    wc[i,j] = np.sum(weights*components[i]*components[j])/(wi*wj)
        self.wcorr_matrix = wc
        return wc

    # ── Auto Grouping: Hierarchical ─────────────────────────
    def auto_group_wcorr(self, num_components=None, n_signal_groups=2,
                         linkage_method='average'):
        """
        n_signal_groups: jumlah grup SINYAL yang diinginkan.
        Contoh:
          n_signal_groups=1 → 1 grup sinyal + Noise  (cluster=1)
          n_signal_groups=2 → 2 grup sinyal + Noise  (cluster=2)
        Total cluster = n_signal_groups, lalu sisa komponen → Noise.
        """
        if num_components is None: num_components = min(self.d, 20)
        num_components = min(num_components, self.d)
        wcorr = self.w_correlation(num_components)
        dist = 1 - np.abs(wcorr)
        np.fill_diagonal(dist, 0); dist = (dist+dist.T)/2
        condensed = squareform(dist, checks=False)
        Z = linkage(condensed, method=linkage_method)
        # Cluster hanya komponen yang di-input
        n_clust = max(1, n_signal_groups)
        labels = fcluster(Z, t=n_clust, criterion='maxclust')
        self._hc_linkage = Z; self._hc_labels = labels

        # Buat grup sinyal berdasarkan cluster
        raw_groups = {}
        for cl in sorted(set(labels)):
            raw_groups[cl] = [i for i,lb in enumerate(labels) if lb==cl]

        # Beri nama berdasarkan frekuensi dominan
        renamed = {}; trend_found = False; seas_count = 0
        for cl, members in raw_groups.items():
            rc = self.reconstruct_component(members[0])
            freqs, psd = periodogram(rc, fs=1.0)
            dom = freqs[np.argmax(psd[1:])+1] if len(psd)>1 else 0
            if dom < 0.02 and not trend_found:
                renamed['Trend'] = members; trend_found = True
            elif dom < 0.02:
                renamed[f'Low_Freq_{cl}'] = members
            else:
                seas_count += 1
                T = 1/dom if dom > 0 else np.inf
                renamed[f'Seasonal_{seas_count} (T≈{T:.1f})'] = members

        # Sisa komponen → Noise
        used = set(i for v in renamed.values() for i in v)
        noise_idx = [i for i in range(self.d) if i not in used and i >= num_components]
        # Juga tambah komponen yg tidak di-cluster sebagai noise
        noise_idx += [i for i in range(num_components, min(self.d, self.L)) if i not in used]
        noise_idx = sorted(set(noise_idx))
        if noise_idx:
            renamed['Noise'] = noise_idx
        return renamed

    # ── Auto Grouping: Periodogram ───────────────────────────
    def auto_group_periodogram(self, num_components=None,
                               freq_threshold=0.02, pair_tolerance=0.01):
        if num_components is None:
            num_components = max(2, int(np.sum(self.contribution > 0.5)))
            num_components = min(num_components, self.d, 30)
        dom_freqs = []
        for i in range(num_components):
            rc = self.reconstruct_component(i)
            freqs, psd = periodogram(rc, fs=1.0)
            dom = freqs[np.argmax(psd[1:])+1] if len(psd)>1 else 0
            dom_freqs.append(dom)
        trend_idx = [i for i,f in enumerate(dom_freqs) if f < freq_threshold]
        seasonal_idx = [i for i,f in enumerate(dom_freqs) if f >= freq_threshold]
        assigned = set(); seasonal_groups = []
        for i in seasonal_idx:
            if i in assigned: continue
            grp = [i]; assigned.add(i)
            for j in seasonal_idx:
                if j not in assigned and abs(dom_freqs[i]-dom_freqs[j]) < pair_tolerance:
                    grp.append(j); assigned.add(j)
            seasonal_groups.append(grp)
        groups = {}
        if trend_idx: groups['Trend'] = trend_idx
        for k, grp in enumerate(seasonal_groups, 1):
            T = 1/dom_freqs[grp[0]] if dom_freqs[grp[0]] > 0 else np.inf
            groups[f'Seasonal_{k} (T≈{T:.1f})'] = grp
        noise_idx = list(range(num_components, min(self.d, self.L)))
        if noise_idx: groups['Noise'] = noise_idx
        return groups

    # ── Reconstruct ──────────────────────────────────────────
    def reconstruct(self, groups):
        self.groups = groups; self.reconstructed = {}
        for name, indices in groups.items():
            mat = sum(self.elementary_matrices[i] for i in indices if i < self.d)
            self.reconstructed[name] = self._diagonal_averaging(mat)
        all_idx = sorted(set(i for v in groups.values() for i in v if i < self.d))
        total_mat = sum(self.elementary_matrices[i] for i in all_idx)
        self.reconstructed['_Total'] = self._diagonal_averaging(total_mat)
        self.reconstructed['_Residual'] = self.original - self.reconstructed['_Total']
        return self.reconstructed

    # ── Forecast Recurrent ───────────────────────────────────
    def forecast_recurrent(self, groups, steps=10, use_indices=None):
        indices = self._resolve_indices(groups, use_indices)
        signal_mat = sum(self.elementary_matrices[i] for i in indices if i < self.d)
        signal = self._diagonal_averaging(signal_mat)
        U_sel = self.U[:, indices]; pi = U_sel[-1, :]
        nu2 = np.sum(pi**2); nu2c = min(nu2, 0.9999)
        R = np.sum(pi[np.newaxis,:]*U_sel[:-1,:], axis=1)/(1-nu2c)
        self.lrr_coefficients = R[::-1].copy()
        self.lrr_nu_squared = nu2
        self.lrr_info = dict(nu_squared=nu2, num_coefficients=len(R),
            num_eigentriples_used=len(indices), eigentriple_indices=indices,
            formula=f"x(n) = Σ a_j * x(n-j), j=1..{len(R)}")
        y = np.concatenate([signal, np.zeros(steps)])
        for t in range(self.N, self.N+steps):
            y[t] = np.dot(R, y[t-self.L+1:t][::-1][:self.L-1])
        self.forecast_r = y; self.forecast_r_steps = steps
        return y

    # ── Forecast Vector ──────────────────────────────────────
    def forecast_vector(self, groups, steps=10, use_indices=None):
        indices = self._resolve_indices(groups, use_indices)
        U_sel = self.U[:, indices]; pi = U_sel[-1, :]
        nu2 = np.sum(pi**2); nu2c = min(nu2, 0.9999)
        P_pi = U_sel[:-1,:] @ pi / (1-nu2c)
        self.vforecast_coefficients = P_pi.copy()
        self.vforecast_nu_squared = nu2
        self.vforecast_info = dict(nu_squared=nu2, num_coefficients=len(P_pi),
            num_eigentriples_used=len(indices), eigentriple_indices=indices)
        signal_mat = sum(self.elementary_matrices[i] for i in indices if i < self.d)
        signal = self._diagonal_averaging(signal_mat)
        Q = np.column_stack([signal[i:i+self.L] for i in range(self.K)])
        for _ in range(steps):
            last = Q[:,-1]; nl = last[1:]
            Q = np.column_stack([Q, np.append(nl, np.dot(P_pi, nl))])
        Le,Ke = Q.shape; Ne = Le+Ke-1
        res = np.zeros(Ne); cnt = np.zeros(Ne)
        for i in range(Le):
            for j in range(Ke):
                res[i+j] += Q[i,j]; cnt[i+j] += 1
        y = (res/cnt)[:self.N+steps]
        self.forecast_v = y; self.forecast_v_steps = steps
        return y

    def _resolve_indices(self, groups, use_indices):
        if use_indices is not None: return sorted(use_indices)
        if isinstance(groups, dict): return sorted(set(i for v in groups.values() for i in v))
        return sorted(groups)

    # ── Bootstrap ────────────────────────────────────────────
    def bootstrap_intervals(self, groups, steps=10, method='recurrent',
                            n_bootstrap=500, confidence=0.95):
        indices = self._resolve_indices(groups, None)
        signal_mat = sum(self.elementary_matrices[i] for i in indices if i < self.d)
        signal = self._diagonal_averaging(signal_mat)
        residual = self.original - signal
        residual_std = np.std(residual)
        forecasts = np.zeros((n_bootstrap, steps))
        for b in range(n_bootstrap):
            boot_resid = np.random.choice(residual, size=self.N, replace=True)
            boot_ts = signal + boot_resid
            try:
                ssa_b = SSA(boot_ts, window_length=self.L, name='boot')
                if method == 'vector':
                    fc = ssa_b.forecast_vector(groups, steps=steps)
                else:
                    fc = ssa_b.forecast_recurrent(groups, steps=steps)
                forecasts[b,:] = fc[self.N:self.N+steps]
            except:
                forecasts[b,:] = np.nan
        valid = ~np.any(np.isnan(forecasts), axis=1)
        fv = forecasts[valid]
        if len(fv) < 10: return None
        alpha = 1-confidence
        fc_mean = np.mean(fv, axis=0)
        ci_lower = np.percentile(fv, alpha/2*100, axis=0)
        ci_upper = np.percentile(fv, (1-alpha/2)*100, axis=0)
        pi_lower = ci_lower - 1.96*residual_std
        pi_upper = ci_upper + 1.96*residual_std
        self.bootstrap_result = dict(
            forecast_mean=fc_mean, ci_lower=ci_lower, ci_upper=ci_upper,
            pi_lower=pi_lower, pi_upper=pi_upper, all_forecasts=fv,
            confidence=confidence, method=method, residual_std=residual_std)
        return self.bootstrap_result

    @staticmethod
    def evaluate_intervals(actual, lower, upper, confidence=0.95):
        actual=np.array(actual); lower=np.array(lower); upper=np.array(upper)
        n=len(actual); alpha=1-confidence
        inside=((actual>=lower)&(actual<=upper)).astype(float)
        picp=np.mean(inside); widths=upper-lower; mw=np.mean(widths)
        dr=np.max(actual)-np.min(actual)
        pinaw=mw/dr if dr>0 else np.inf
        ace=picp-confidence
        eta=50; mu=1 if picp<confidence else 0
        cwc=pinaw*(1+mu*np.exp(-eta*(picp-confidence)))
        scores=np.zeros(n)
        for i in range(n):
            w=widths[i]
            if actual[i]<lower[i]: scores[i]=w+(2/alpha)*(lower[i]-actual[i])
            elif actual[i]>upper[i]: scores[i]=w+(2/alpha)*(actual[i]-upper[i])
            else: scores[i]=w
        return dict(PICP=picp,PINAW=pinaw,ACE=ace,CWC=cwc,
            Winkler_Score=np.mean(scores),Mean_Width=mw,
            Nominal_Coverage=confidence,N=n)

    # ── Monte Carlo ──────────────────────────────────────────
    def monte_carlo_test(self, num_surrogates=1000, confidence=0.95):
        ts=self.original-np.mean(self.original)
        lag1=np.corrcoef(ts[:-1],ts[1:])[0,1]
        nvar=np.var(ts)*(1-lag1**2); n_eig=min(self.L,self.K)
        surr_eig=np.zeros((num_surrogates,n_eig))
        for s in range(num_surrogates):
            surr=np.zeros(self.N); surr[0]=np.random.normal(0,np.sqrt(np.var(ts)))
            for t in range(1,self.N):
                surr[t]=lag1*surr[t-1]+np.random.normal(0,np.sqrt(max(nvar,1e-12)))
            traj=np.column_stack([surr[i:i+self.L] for i in range(self.N-self.L+1)])
            eig=np.linalg.eigvalsh(traj@traj.T)[::-1]
            surr_eig[s,:len(eig)]=eig[:n_eig]
        n_test=min(20,self.d)
        lower=np.percentile(surr_eig,(1-confidence)/2*100,axis=0)[:n_test]
        upper=np.percentile(surr_eig,(1+confidence)/2*100,axis=0)[:n_test]
        median_s=np.median(surr_eig,axis=0)[:n_test]
        ev=self.eigenvalues[:n_test]
        self.mc_results=dict(eigenvalues=ev,surrogate_lower=lower,surrogate_upper=upper,
            surrogate_median=median_s,significant=ev>upper,confidence=confidence)
        return self.mc_results

    @staticmethod
    def evaluate(actual, predicted):
        actual,predicted=np.array(actual),np.array(predicted)
        err=actual-predicted; ae=np.abs(err)
        rmse=np.sqrt(np.mean(err**2)); mae=np.mean(ae)
        nz=np.abs(actual)>1e-10
        mape=np.mean(ae[nz]/np.abs(actual[nz]))*100 if nz.any() else np.inf
        denom=np.abs(actual)+np.abs(predicted); nzs=denom>1e-10
        smape=np.mean(2*ae[nzs]/denom[nzs])*100 if nzs.any() else np.inf
        ss_res=np.sum(err**2); ss_tot=np.sum((actual-np.mean(actual))**2)
        r2=1-ss_res/ss_tot if ss_tot>0 else np.nan
        rng=np.max(actual)-np.min(actual)
        return dict(N=len(actual),RMSE=rmse,MAE=mae,MAPE_pct=mape,
            sMAPE_pct=smape,R2=r2,NRMSE=rmse/rng if rng>0 else np.inf,
            MaxAE=np.max(ae),MedAE=np.median(ae))

    def residual_analysis(self, residuals=None):
        if residuals is None:
            residuals=self.reconstructed.get('_Residual',None)
        if residuals is None: return None
        r=np.array(residuals)
        info=dict(mean=np.mean(r),std=np.std(r),min_val=np.min(r),max_val=np.max(r),
            skewness=float(pd.Series(r).skew()),kurtosis=float(pd.Series(r).kurtosis()))
        if len(r)<=5000:
            sw_s,sw_p=shapiro(r); info['shapiro_stat']=sw_s; info['shapiro_p']=sw_p
        jb_s,jb_p=jarque_bera(r); info['jarque_bera_stat']=jb_s; info['jarque_bera_p']=jb_p
        n_lags=min(20,len(r)//5)
        if n_lags>=1:
            lb=acorr_ljungbox(r,lags=n_lags,return_df=True)
            info['ljung_box_stat']=float(lb['lb_stat'].iloc[-1])
            info['ljung_box_p']=float(lb['lb_pvalue'].iloc[-1])
        self.residuals=r; self.residual_info=info
        return info

    def save_results(self, filename='SSA_Results.xlsx'):
        with pd.ExcelWriter(filename, engine='openpyxl') as w:
            pd.DataFrame({'Original':self.original}).to_excel(w,sheet_name='Data',index_label='t')
            pd.DataFrame({
                'Komponen':range(1,self.d+1),'Singular_Value':self.singular_values[:self.d],
                'Eigenvalue':self.eigenvalues[:self.d],'Kontribusi_pct':self.contribution,
                'Kumulatif_pct':self.cumulative_contribution
            }).to_excel(w,sheet_name='Eigenvalues',index=False)
            if hasattr(self,'reconstructed'):
                rc={'Original':self.original}; rc.update(self.reconstructed)
                pd.DataFrame(rc).to_excel(w,sheet_name='Rekonstruksi',index_label='t')
            if hasattr(self,'lrr_coefficients'):
                nc=len(self.lrr_coefficients)
                pd.DataFrame({'j':range(1,nc+1),'a_j':self.lrr_coefficients}).to_excel(
                    w,sheet_name='R_Coefficients',index=False)
            if hasattr(self,'vforecast_coefficients'):
                nc=len(self.vforecast_coefficients)
                pd.DataFrame({'j':range(1,nc+1),'P_pi':self.vforecast_coefficients}).to_excel(
                    w,sheet_name='V_Coefficients',index=False)
            if hasattr(self,'forecast_r') or hasattr(self,'forecast_v'):
                fc={}
                if hasattr(self,'forecast_r'): fc['R_Forecast']=self.forecast_r
                if hasattr(self,'forecast_v'): fc['V_Forecast']=self.forecast_v
                if len(fc)>1:
                    mx=max(len(v) for v in fc.values())
                    fc={k:np.pad(v,(0,mx-len(v)),constant_values=np.nan) for k,v in fc.items()}
                pd.DataFrame(fc).to_excel(w,sheet_name='Forecast',index_label='t')
            if hasattr(self,'wcorr_matrix'):
                n=self.wcorr_matrix.shape[0]; lbl=[f'F{i+1}' for i in range(n)]
                pd.DataFrame(self.wcorr_matrix,index=lbl,columns=lbl).to_excel(w,sheet_name='W-Correlation')
            if hasattr(self,'bootstrap_result') and self.bootstrap_result:
                br=self.bootstrap_result
                pd.DataFrame({'Forecast_Mean':br['forecast_mean'],
                    'CI_Lower':br['ci_lower'],'CI_Upper':br['ci_upper'],
                    'PI_Lower':br['pi_lower'],'PI_Upper':br['pi_upper']
                }).to_excel(w,sheet_name='Bootstrap_Intervals',index_label='h')


def find_optimal_L(time_series, L_min=None, L_max=None, L_step=None):
    ts=np.array(time_series,dtype=float); N=len(ts)
    if L_min is None: L_min=max(2,N//4)
    if L_max is None: L_max=N//2
    L_min=max(2,int(L_min)); L_max=min(N//2,int(L_max))
    if L_step is None: L_step=max(1,(L_max-L_min)//40)
    L_step=max(1,int(L_step))
    results=[]
    for L in range(L_min,L_max+1,L_step):
        try:
            ssa=SSA(ts,window_length=L)
            n_sig=max(1,int(np.sum(ssa.contribution>1.0)))
            ssa.reconstruct({'Signal':list(range(n_sig))})
            res=ssa.reconstructed['_Residual']
            results.append({'L':L,'RMSE':np.sqrt(np.mean(res**2))})
        except: continue
    if not results: return None
    df=pd.DataFrame(results); best=df.loc[df['RMSE'].idxmin()]
    return {'best_L':int(best['L']),'best_RMSE':float(best['RMSE']),'all_results':df}
