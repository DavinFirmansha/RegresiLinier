"""
══════════════════════════════════════════════════════════════════
  SSA CORE LIBRARY — Singular Spectrum Analysis (1D Time Series)
  Untuk dipakai oleh Streamlit App

  Fitur:
  ──────
  1.  Embedding (Trajectory Matrix)
  2.  SVD Decomposition
  3.  W-Correlation Matrix
  4.  Grouping (Manual & Auto)
  5.  Diagonal Averaging / Rekonstruksi
  6.  Recurrent Forecasting (R-forecast / LRF)
  7.  Vector Forecasting (V-forecast)
  8.  Monte Carlo SSA Significance Test
  9.  Evaluasi (RMSE, MAE, MAPE, sMAPE, R², NRMSE, dll)
  10. Residual Analysis (Shapiro-Wilk, Jarque-Bera, Ljung-Box)
  11. Optimasi Window Length
══════════════════════════════════════════════════════════════════
"""

import numpy as np
import pandas as pd
from scipy.stats import shapiro, jarque_bera
from scipy.signal import periodogram
from statsmodels.stats.diagnostic import acorr_ljungbox
import warnings, io
from contextlib import redirect_stdout

warnings.filterwarnings('ignore')


class SSA:
    """
    Singular Spectrum Analysis untuk time series 1 dimensi.

    Parameters
    ----------
    time_series : array-like
        Data time series 1D.
    window_length : int atau 'auto'
        Panjang window (L). 'auto' → L = N//2.
    name : str
        Label series (untuk judul plot).
    """

    def __init__(self, time_series, window_length='auto', name='Time Series'):
        self.original = np.array(time_series, dtype=float).flatten()
        self.N = len(self.original)
        self.name = name

        if window_length == 'auto':
            self.L = self.N // 2
        else:
            self.L = int(window_length)

        self.K = self.N - self.L + 1

        assert 2 <= self.L <= self.N // 2 + 1, \
            f"L harus antara 2 dan N/2+1. L={self.L}, N={self.N}"

        self._embed()
        self._decompose()

    # ── Step 1: Embedding ────────────────────────────────────────────
    def _embed(self):
        self.trajectory_matrix = np.column_stack(
            [self.original[i:i + self.L] for i in range(self.K)]
        )

    # ── Step 2: SVD ──────────────────────────────────────────────────
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
            Xi = self.singular_values[i] * np.outer(self.U[:, i], vi)
            self.elementary_matrices.append(Xi)

        total_var = np.sum(self.eigenvalues[:self.d])
        self.contribution = self.eigenvalues[:self.d] / total_var * 100
        self.cumulative_contribution = np.cumsum(self.contribution)

    # ── Diagonal Averaging ───────────────────────────────────────────
    def _diagonal_averaging(self, matrix):
        L, K = matrix.shape
        N = L + K - 1
        result = np.zeros(N)
        counts = np.zeros(N)
        for i in range(L):
            for j in range(K):
                result[i + j] += matrix[i, j]
                counts[i + j] += 1
        return result / counts

    # ── Rekonstruksi satu komponen ───────────────────────────────────
    def reconstruct_component(self, index):
        return self._diagonal_averaging(self.elementary_matrices[index])

    # ── W-Correlation ────────────────────────────────────────────────
    def w_correlation(self, num_components=None):
        if num_components is None:
            num_components = min(self.d, 20)
        components = [self.reconstruct_component(i) for i in range(num_components)]
        weights = np.zeros(self.N)
        Ls = min(self.L, self.K)
        Ks = max(self.L, self.K)
        for i in range(self.N):
            if i < Ls - 1:
                weights[i] = i + 1
            elif i < Ks:
                weights[i] = Ls
            else:
                weights[i] = self.N - i
        self.wcorr_matrix = np.zeros((num_components, num_components))
        for i in range(num_components):
            for j in range(num_components):
                wi = np.sqrt(np.sum(weights * components[i] ** 2))
                wj = np.sqrt(np.sum(weights * components[j] ** 2))
                if wi > 0 and wj > 0:
                    self.wcorr_matrix[i, j] = np.sum(weights * components[i] * components[j]) / (wi * wj)
        return self.wcorr_matrix

    # ── Grouping & Reconstruction ────────────────────────────────────
    def reconstruct(self, groups):
        self.groups = groups
        self.reconstructed = {}
        for name, indices in groups.items():
            mat = sum(self.elementary_matrices[i] for i in indices)
            self.reconstructed[name] = self._diagonal_averaging(mat)
        all_idx = [i for v in groups.values() for i in v]
        total_mat = sum(self.elementary_matrices[i] for i in all_idx)
        self.reconstructed['_Total'] = self._diagonal_averaging(total_mat)
        self.reconstructed['_Residual'] = self.original - self.reconstructed['_Total']
        return self.reconstructed

    # ── Auto Grouping ────────────────────────────────────────────────
    def auto_group(self, num_components=None, threshold=0.5):
        if num_components is None:
            num_components = max(2, int(np.sum(self.contribution > 0.5)))
            num_components = min(num_components, self.d, 20)
        wcorr = self.w_correlation(num_components)
        groups_list, assigned = [], set()
        for i in range(num_components):
            if i in assigned:
                continue
            group = [i]
            assigned.add(i)
            for j in range(i + 1, num_components):
                if j not in assigned and abs(wcorr[i, j]) > threshold:
                    group.append(j)
                    assigned.add(j)
            groups_list.append(group)
        groups = {}
        trend_found = False
        seasonal_count = 0
        for idx, grp in enumerate(groups_list):
            rc = self.reconstruct_component(grp[0])
            freqs, psd = periodogram(rc, fs=1.0)
            dominant_freq = freqs[np.argmax(psd[1:]) + 1] if len(psd) > 1 else 0
            if dominant_freq < 0.02 and not trend_found:
                groups['Trend'] = grp
                trend_found = True
            elif dominant_freq < 0.02:
                groups[f'Trend_{idx}'] = grp
            else:
                seasonal_count += 1
                period = 1 / dominant_freq if dominant_freq > 0 else float('inf')
                groups[f'Seasonal_{seasonal_count} (T≈{period:.1f})'] = grp
        remaining = sorted(set(range(num_components)) - assigned)
        noise_all = remaining + list(range(num_components, min(self.d, self.L)))
        if noise_all:
            groups['Noise'] = noise_all
        return groups

    # ── Recurrent Forecasting (R-forecast / LRF) ────────────────────
    def forecast_recurrent(self, groups, steps=10, use_indices=None):
        indices = self._resolve_indices(groups, use_indices)
        signal_mat = sum(self.elementary_matrices[i] for i in indices if i < self.d)
        signal = self._diagonal_averaging(signal_mat)
        U_sel = self.U[:, indices]
        pi = U_sel[-1, :]
        nu2 = np.sum(pi ** 2)
        if nu2 >= 1:
            nu2 = 0.9999
        R = np.sum(pi[np.newaxis, :] * U_sel[:-1, :], axis=1) / (1 - nu2)
        y = np.concatenate([signal, np.zeros(steps)])
        for t in range(self.N, self.N + steps):
            y[t] = np.dot(R, y[t - self.L + 1:t][::-1][:self.L - 1])
        self.forecast_r = y
        self.forecast_r_steps = steps
        return y

    # ── Vector Forecasting (V-forecast) ──────────────────────────────
    def forecast_vector(self, groups, steps=10, use_indices=None):
        indices = self._resolve_indices(groups, use_indices)
        U_sel = self.U[:, indices]
        pi = U_sel[-1, :]
        nu2 = np.sum(pi ** 2)
        if nu2 >= 1:
            nu2 = 0.9999
        U_del = U_sel[:-1, :]
        P_pi = U_del @ pi / (1 - nu2)
        signal_mat = sum(self.elementary_matrices[i] for i in indices if i < self.d)
        signal = self._diagonal_averaging(signal_mat)
        Q = np.column_stack([signal[i:i + self.L] for i in range(self.K)])
        for _ in range(steps):
            last = Q[:, -1]
            new_last = last[1:]
            new_elem = np.dot(P_pi, new_last)
            Q = np.column_stack([Q, np.append(new_last, new_elem)])
        L_ext, K_ext = Q.shape
        N_ext = L_ext + K_ext - 1
        res = np.zeros(N_ext)
        cnt = np.zeros(N_ext)
        for i in range(L_ext):
            for j in range(K_ext):
                res[i + j] += Q[i, j]
                cnt[i + j] += 1
        y = (res / cnt)[:self.N + steps]
        self.forecast_v = y
        self.forecast_v_steps = steps
        return y

    def _resolve_indices(self, groups, use_indices):
        if use_indices is not None:
            return sorted(use_indices)
        if isinstance(groups, dict):
            return sorted(set(i for v in groups.values() for i in v))
        return sorted(groups)

    # ── Monte Carlo SSA Significance Test ────────────────────────────
    def monte_carlo_test(self, num_surrogates=1000, confidence=0.95):
        ts = self.original - np.mean(self.original)
        lag1 = np.corrcoef(ts[:-1], ts[1:])[0, 1]
        nvar = np.var(ts) * (1 - lag1 ** 2)
        n_eig = min(self.L, self.K)
        surr_eig = np.zeros((num_surrogates, n_eig))
        for s in range(num_surrogates):
            surr = np.zeros(self.N)
            surr[0] = np.random.normal(0, np.sqrt(np.var(ts)))
            for t in range(1, self.N):
                surr[t] = lag1 * surr[t - 1] + np.random.normal(0, np.sqrt(max(nvar, 1e-12)))
            traj = np.column_stack([surr[i:i + self.L] for i in range(self.N - self.L + 1)])
            eig = np.linalg.eigvalsh(traj @ traj.T)[::-1]
            surr_eig[s, :len(eig)] = eig[:n_eig]
        n_test = min(20, self.d)
        lower = np.percentile(surr_eig, (1 - confidence) / 2 * 100, axis=0)[:n_test]
        upper = np.percentile(surr_eig, (1 + confidence) / 2 * 100, axis=0)[:n_test]
        median_s = np.median(surr_eig, axis=0)[:n_test]
        ev = self.eigenvalues[:n_test]
        self.mc_results = dict(
            eigenvalues=ev, surrogate_lower=lower, surrogate_upper=upper,
            surrogate_median=median_s, significant=ev > upper, confidence=confidence
        )
        return self.mc_results

    # ── Evaluasi ─────────────────────────────────────────────────────
    @staticmethod
    def evaluate(actual, predicted):
        actual, predicted = np.array(actual), np.array(predicted)
        err = actual - predicted
        ae = np.abs(err)
        rmse = np.sqrt(np.mean(err ** 2))
        mae = np.mean(ae)
        nz = np.abs(actual) > 1e-10
        mape = np.mean(ae[nz] / np.abs(actual[nz])) * 100 if nz.any() else np.inf
        denom = np.abs(actual) + np.abs(predicted)
        nzs = denom > 1e-10
        smape = np.mean(2 * ae[nzs] / denom[nzs]) * 100 if nzs.any() else np.inf
        ss_res = np.sum(err ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        rng = np.max(actual) - np.min(actual)
        nrmse = rmse / rng if rng > 0 else np.inf
        return dict(N=len(actual), RMSE=rmse, MAE=mae, MAPE_pct=mape,
                    sMAPE_pct=smape, R2=r2, NRMSE=nrmse,
                    MaxAE=np.max(ae), MedAE=np.median(ae))

    def evaluate_split(self, train_size, forecast_values):
        if isinstance(train_size, float) and 0 < train_size < 1:
            tn = int(self.N * train_size)
        else:
            tn = int(train_size)
        pred = forecast_values[:self.N]
        return dict(
            train=self.evaluate(self.original[:tn], pred[:tn]),
            test=self.evaluate(self.original[tn:], pred[tn:]),
            overall=self.evaluate(self.original, pred),
            train_size=tn, test_size=self.N - tn
        )

    # ── Residual Analysis ────────────────────────────────────────────
    def residual_analysis(self, residuals=None):
        if residuals is None:
            residuals = self.reconstructed.get('_Residual', None)
        if residuals is None:
            return None
        r = np.array(residuals)
        info = dict(
            mean=np.mean(r), std=np.std(r), min_val=np.min(r), max_val=np.max(r),
            skewness=float(pd.Series(r).skew()), kurtosis=float(pd.Series(r).kurtosis())
        )
        if len(r) <= 5000:
            sw_stat, sw_p = shapiro(r)
            info['shapiro_stat'] = sw_stat
            info['shapiro_p'] = sw_p
        jb_stat, jb_p = jarque_bera(r)
        info['jarque_bera_stat'] = jb_stat
        info['jarque_bera_p'] = jb_p
        n_lags = min(20, len(r) // 5)
        if n_lags >= 1:
            lb = acorr_ljungbox(r, lags=n_lags, return_df=True)
            info['ljung_box_stat'] = float(lb['lb_stat'].iloc[-1])
            info['ljung_box_p'] = float(lb['lb_pvalue'].iloc[-1])
        self.residuals = r
        self.residual_info = info
        return info

    # ── Save to Excel ────────────────────────────────────────────────
    def save_results(self, filename='SSA_Results.xlsx'):
        with pd.ExcelWriter(filename, engine='openpyxl') as w:
            pd.DataFrame({'Original': self.original}).to_excel(w, sheet_name='Data', index_label='t')
            pd.DataFrame({
                'Komponen': range(1, self.d + 1),
                'Singular_Value': self.singular_values[:self.d],
                'Eigenvalue': self.eigenvalues[:self.d],
                'Kontribusi_pct': self.contribution,
                'Kumulatif_pct': self.cumulative_contribution
            }).to_excel(w, sheet_name='Eigenvalues', index=False)
            if hasattr(self, 'reconstructed'):
                rc = {'Original': self.original}
                rc.update(self.reconstructed)
                pd.DataFrame(rc).to_excel(w, sheet_name='Rekonstruksi', index_label='t')
            if hasattr(self, 'forecast_r'):
                fc = {'R_Forecast': self.forecast_r}
                if hasattr(self, 'forecast_v'):
                    mx = max(len(self.forecast_r), len(self.forecast_v))
                    fc = {
                        'R_Forecast': np.pad(self.forecast_r, (0, mx - len(self.forecast_r)), constant_values=np.nan),
                        'V_Forecast': np.pad(self.forecast_v, (0, mx - len(self.forecast_v)), constant_values=np.nan),
                    }
                pd.DataFrame(fc).to_excel(w, sheet_name='Forecast', index_label='t')
            if hasattr(self, 'wcorr_matrix'):
                n = self.wcorr_matrix.shape[0]
                lbl = [f'F{i+1}' for i in range(n)]
                pd.DataFrame(self.wcorr_matrix, index=lbl, columns=lbl).to_excel(w, sheet_name='W-Correlation')


def find_optimal_L(time_series, L_range=None):
    ts = np.array(time_series, dtype=float)
    N = len(ts)
    if L_range is None:
        step = max(1, N // 40)
        L_range = range(3, N // 2 + 1, step)
    results = []
    for L in L_range:
        try:
            ssa = SSA(ts, window_length=L)
            n_sig = max(1, int(np.sum(ssa.contribution > 1.0)))
            ssa.reconstruct({'Signal': list(range(n_sig))})
            res = ssa.reconstructed['_Residual']
            results.append({'L': L, 'RMSE': np.sqrt(np.mean(res ** 2))})
        except Exception:
            continue
    if not results:
        return None
    df = pd.DataFrame(results)
    best = df.loc[df['RMSE'].idxmin()]
    return {'best_L': int(best['L']), 'best_RMSE': float(best['RMSE']), 'all_results': df}
