import streamlit as st
import pandas as pd
import numpy as np
import math
from scipy import stats
from scipy.optimize import minimize
from scipy.special import gammaln, gammaincc, gammainc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Analisis Data Uji Hidup", page_icon="⏱️", layout="wide")
st.title("⏱️ Analisis Data Uji Hidup (Survival Analysis)")
st.markdown("""
Aplikasi lengkap untuk **analisis ketahanan hidup / survival analysis** dengan dukungan
**data tersensor Tipe I, Tipe II, dan Random (Tipe III)**.  
Model parametrik: **15 distribusi** termasuk varian 2-parameter dan 3-parameter.
""")

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def kaplan_meier(durations, events):
    df = pd.DataFrame({'t': durations, 'e': events}).sort_values('t').reset_index(drop=True)
    unique_times = np.sort(df['t'].unique())
    n_total = len(df); rows = []; survived = 1.0; var_sum = 0.0; n_at_risk = n_total
    for t in unique_times:
        d = int(df[(df['t'] == t) & (df['e'] == 1)].shape[0])
        c = int(df[(df['t'] == t) & (df['e'] == 0)].shape[0])
        if n_at_risk > 0 and d > 0:
            survived *= (1 - d / n_at_risk)
            var_sum += d / (n_at_risk * (n_at_risk - d)) if (n_at_risk - d) > 0 else 0
        se = survived * np.sqrt(var_sum) if var_sum >= 0 else 0
        ci_lo = max(0, survived - 1.96 * se); ci_up = min(1, survived + 1.96 * se)
        rows.append({'Time': t, 'n_risk': n_at_risk, 'n_event': d, 'n_censor': c,
                     'S(t)': round(survived, 6), 'SE': round(se, 6),
                     'CI_lower': round(ci_lo, 6), 'CI_upper': round(ci_up, 6)})
        n_at_risk -= (d + c)
    return pd.DataFrame(rows)

def nelson_aalen(durations, events):
    df = pd.DataFrame({'t': durations, 'e': events}).sort_values('t').reset_index(drop=True)
    unique_times = np.sort(df['t'].unique())
    n_total = len(df); rows = []; H = 0.0; var_H = 0.0; n_at_risk = n_total
    for t in unique_times:
        d = int(df[(df['t'] == t) & (df['e'] == 1)].shape[0])
        c = int(df[(df['t'] == t) & (df['e'] == 0)].shape[0])
        if n_at_risk > 0:
            H += d / n_at_risk; var_H += d / (n_at_risk**2)
        se = np.sqrt(var_H); S_na = np.exp(-H)
        rows.append({'Time': t, 'n_risk': n_at_risk, 'n_event': d, 'n_censor': c,
                     'H(t)': round(H, 6), 'SE_H': round(se, 6), 'S_NA(t)': round(S_na, 6)})
        n_at_risk -= (d + c)
    return pd.DataFrame(rows)

def log_rank_test(durations1, events1, durations2, events2):
    all_times = np.sort(np.unique(np.concatenate([durations1, durations2])))
    df1 = pd.DataFrame({'t': durations1, 'e': events1})
    df2 = pd.DataFrame({'t': durations2, 'e': events2})
    O1, E1, var_sum = 0.0, 0.0, 0.0
    for t in all_times:
        d1 = int(df1[(df1['t'] == t) & (df1['e'] == 1)].shape[0])
        d2 = int(df2[(df2['t'] == t) & (df2['e'] == 1)].shape[0])
        r1 = int((df1['t'] >= t).sum()); r2 = int((df2['t'] >= t).sum())
        d = d1 + d2; r = r1 + r2
        if r > 0:
            e1 = r1 * d / r; O1 += d1; E1 += e1
            if r > 1: var_sum += (r1 * r2 * d * (r - d)) / (r**2 * (r - 1))
    chi2 = (O1 - E1)**2 / var_sum if var_sum > 0 else 0
    return chi2, 1 - stats.chi2.cdf(chi2, 1), O1, E1

def wilcoxon_gehan_test(d1, e1, d2, e2):
    all_times = np.sort(np.unique(np.concatenate([d1, d2])))
    df1 = pd.DataFrame({'t': d1, 'e': e1}); df2 = pd.DataFrame({'t': d2, 'e': e2})
    U, V = 0.0, 0.0
    for t in all_times:
        dd1 = int(df1[(df1['t']==t)&(df1['e']==1)].shape[0])
        dd2 = int(df2[(df2['t']==t)&(df2['e']==1)].shape[0])
        r1 = int((df1['t']>=t).sum()); r2 = int((df2['t']>=t).sum())
        d = dd1+dd2; r = r1+r2
        if r > 0:
            ee1 = r1*d/r; w = r; U += w*(dd1-ee1)
            if r > 1: V += w**2*(r1*r2*d*(r-d))/(r**2*(r-1))
    chi2 = U**2/V if V > 0 else 0
    return chi2, 1-stats.chi2.cdf(chi2, 1)

def tarone_ware_test(d1, e1, d2, e2):
    all_times = np.sort(np.unique(np.concatenate([d1, d2])))
    df1 = pd.DataFrame({'t': d1, 'e': e1}); df2 = pd.DataFrame({'t': d2, 'e': e2})
    U, V = 0.0, 0.0
    for t in all_times:
        dd1 = int(df1[(df1['t']==t)&(df1['e']==1)].shape[0])
        dd2 = int(df2[(df2['t']==t)&(df2['e']==1)].shape[0])
        r1 = int((df1['t']>=t).sum()); r2 = int((df2['t']>=t).sum())
        d = dd1+dd2; r = r1+r2
        if r > 0:
            ee1 = r1*d/r; w = np.sqrt(r); U += w*(dd1-ee1)
            if r > 1: V += w**2*(r1*r2*d*(r-d))/(r**2*(r-1))
    chi2 = U**2/V if V > 0 else 0
    return chi2, 1-stats.chi2.cdf(chi2, 1)

def peto_peto_test(d1, e1, d2, e2):
    km_all = kaplan_meier(np.concatenate([d1,d2]), np.concatenate([e1,e2]))
    all_times = np.sort(np.unique(np.concatenate([d1, d2])))
    df1 = pd.DataFrame({'t': d1, 'e': e1}); df2 = pd.DataFrame({'t': d2, 'e': e2})
    U, V = 0.0, 0.0
    for t in all_times:
        dd1 = int(df1[(df1['t']==t)&(df1['e']==1)].shape[0])
        dd2 = int(df2[(df2['t']==t)&(df2['e']==1)].shape[0])
        r1 = int((df1['t']>=t).sum()); r2 = int((df2['t']>=t).sum())
        d = dd1+dd2; r = r1+r2
        if r > 0:
            ee1 = r1*d/r; km_row = km_all[km_all['Time']<=t]
            w = km_row['S(t)'].iloc[-1] if len(km_row) > 0 else 1.0
            U += w*(dd1-ee1)
            if r > 1: V += w**2*(r1*r2*d*(r-d))/(r**2*(r-1))
    chi2 = U**2/V if V > 0 else 0
    return chi2, 1-stats.chi2.cdf(chi2, 1)

def cox_ph_simple(durations, events, X):
    n = len(durations); p = X.shape[1] if X.ndim > 1 else 1
    if X.ndim == 1: X = X.reshape(-1, 1)
    order = np.argsort(durations)
    t_s = durations[order]; e_s = events[order]; X_s = X[order]
    beta = np.zeros(p)
    for _ in range(50):
        grad = np.zeros(p); hess = np.zeros((p, p))
        for i in range(n):
            if e_s[i] == 0: continue
            rs = np.where(t_s >= t_s[i])[0]
            if len(rs) == 0: continue
            exb = np.exp(X_s[rs] @ beta); den = exb.sum()
            wx = (X_s[rs].T * exb).T; mx = wx.sum(0)/den
            grad += X_s[i] - mx
            for j in rs:
                diff = X_s[j] - mx
                hess -= np.outer(diff, diff) * exb[list(rs).index(j)] / den
        try:
            step = np.linalg.solve(hess, grad); beta -= step
            if np.max(np.abs(step)) < 1e-8: break
        except: break
    se = np.sqrt(np.diag(np.linalg.inv(-hess + np.eye(p)*1e-10)))
    z = beta / se; pv = 2*(1-stats.norm.cdf(np.abs(z)))
    return beta, se, z, pv, np.exp(beta)

def median_survival(km_table):
    below = km_table[km_table['S(t)'] <= 0.5]
    return below.iloc[0]['Time'] if len(below) > 0 else np.nan

# ============================================================
# CENSORING TYPE DETECTION & ANALYSIS
# ============================================================
def detect_censoring_type(durations, events):
    """Auto-detect censoring type from data patterns."""
    cens_times = durations[events == 0]
    event_times = durations[events == 1]
    n_cens = len(cens_times); n_evt = len(event_times)
    if n_cens == 0:
        return "Tidak ada data tersensor", {}
    # Type I: semua censored punya waktu yang sama (fixed time)
    unique_cens = np.unique(cens_times)
    if len(unique_cens) == 1:
        return "Tipe I (Fixed Time)", {
            'T_censor': unique_cens[0],
            'desc': f'Semua {n_cens} observasi tersensor pada waktu tetap T = {unique_cens[0]:.2f}'
        }
    # Type I approx: sebagian besar censored di satu titik
    most_common_cens = pd.Series(cens_times).value_counts()
    if most_common_cens.iloc[0] / n_cens > 0.8:
        tc = most_common_cens.index[0]
        return "Tipe I (Fixed Time) — perkiraan", {
            'T_censor': tc,
            'desc': f'{most_common_cens.iloc[0]}/{n_cens} ({most_common_cens.iloc[0]/n_cens:.0%}) tersensor di T ≈ {tc:.2f}'
        }
    # Type II: censored times == max event time, dan semua censored > semua uncensored
    max_evt = event_times.max() if n_evt > 0 else 0
    if n_cens > 0 and np.all(np.isclose(cens_times, max_evt, atol=0.01)):
        return "Tipe II (Fixed Number)", {
            'r': n_evt, 'n': n_evt + n_cens,
            'desc': f'Eksperimen dihentikan setelah r = {n_evt} event dari n = {n_evt+n_cens}. '
                    f'Semua censored di T = {max_evt:.2f}'
        }
    # Random censoring
    return "Random / Tipe III", {
        'desc': f'Waktu sensor bervariasi ({len(unique_cens)} nilai unik). '
                f'Range censored: [{cens_times.min():.2f}, {cens_times.max():.2f}]'
    }

def censoring_analysis_info(durations, events, cens_type, cens_info):
    """Return markdown info about censoring."""
    n = len(durations); d = events.sum(); c = n - d
    lines = []
    lines.append(f"**Tipe Penyensoran Terdeteksi:** `{cens_type}`")
    if 'desc' in cens_info:
        lines.append(f"- {cens_info['desc']}")
    if 'Tipe I' in cens_type:
        Tc = cens_info.get('T_censor', durations[events==0].max())
        lines.append(f"- **Waktu sensor tetap (T_c):** {Tc:.2f}")
        lines.append(f"- Jumlah gagal sebelum T_c: **{int(d)}** (random)")
        lines.append(f"- Likelihood: L = ∏ f(tᵢ)^δᵢ · [S(T_c)]^(n-d)")
        lines.append("")
        lines.append("📌 **Catatan Tipe I:** Jumlah event bersifat **acak**, waktu sensor **tetap**. "
                      "Semua unit yang belum gagal sampai T_c dicatat sebagai censored.")
    elif 'Tipe II' in cens_type:
        r = cens_info.get('r', int(d))
        lines.append(f"- **Jumlah event tetap (r):** {r}")
        lines.append(f"- **Ukuran sampel (n):** {n}")
        lines.append(f"- Waktu henti = waktu event ke-{r}: **{durations[events==1].max():.2f}**")
        lines.append(f"- Likelihood: L = [n!/(n-r)!] · ∏ᵢ₌₁ʳ f(t₍ᵢ₎) · [S(t₍ᵣ₎)]^(n-r)")
        lines.append("")
        lines.append("📌 **Catatan Tipe II:** Jumlah event bersifat **tetap**, waktu henti eksperimen **acak**. "
                      "Unit yang masih hidup setelah event ke-r semuanya di-censor.")
    else:
        lines.append("- Setiap observasi memiliki waktu sensor independen")
        lines.append(f"- Likelihood standar: L = ∏ f(tᵢ)^δᵢ · S(tᵢ)^(1-δᵢ)")
        lines.append("")
        lines.append("📌 **Catatan Tipe III/Random:** Waktu sensor dan waktu gagal saling independen. "
                      "Ini adalah asumsi paling umum di analisis survival.")
    return "\n".join(lines)

# ============================================================
# 15 PARAMETRIC MODELS
# ============================================================
def _safe_log(x): return np.log(np.maximum(x, 1e-300))

# --- EXPONENTIAL ---
def fit_exp1p(t, e):
    """Exponential 1P: S(t)=exp(-λt)"""
    d = e.sum(); T = t.sum()
    lam = d/T if T > 0 else 0.001
    ll = d*np.log(lam+1e-15) - lam*T
    return {'λ': lam}, ll, 1

def fit_exp2p(t, e):
    """Exponential 2P (shifted): S(t)=exp(-λ(t-γ)), t>γ"""
    def neg_ll(params):
        lam, gamma = params
        if lam <= 0 or gamma < 0: return 1e15
        tt = t - gamma
        if np.any(tt <= 0): return 1e15
        ll = e.sum()*np.log(lam+1e-15) - lam*np.sum(tt)
        return -ll
    gamma0 = max(t.min()*0.5, 0)
    res = minimize(neg_ll, [e.sum()/t.sum(), gamma0], method='Nelder-Mead', options={'maxiter':5000})
    lam, gamma = max(abs(res.x[0]), 1e-6), max(res.x[1], 0)
    return {'λ': lam, 'γ': gamma}, -res.fun, 2

# --- WEIBULL ---
def fit_weibull2p(t, e):
    """Weibull 2P: S(t)=exp(-(t/η)^β)"""
    def neg_ll(params):
        beta, eta = params
        if beta <= 0 or eta <= 0: return 1e15
        z = t / eta
        ll = np.sum(e * (np.log(beta+1e-15) - np.log(eta+1e-15) + (beta-1)*_safe_log(z)) - z**beta)
        return -ll
    res = minimize(neg_ll, [1.0, np.median(t)], method='Nelder-Mead', options={'maxiter':5000})
    beta, eta = max(abs(res.x[0]),0.01), max(abs(res.x[1]),0.01)
    return {'β (shape)': beta, 'η (scale)': eta}, -res.fun, 2

def fit_weibull3p(t, e):
    """Weibull 3P (shifted): S(t)=exp(-((t-γ)/η)^β), t>γ"""
    def neg_ll(params):
        beta, eta, gamma = params
        if beta <= 0 or eta <= 0 or gamma < 0: return 1e15
        tt = t - gamma
        if np.any(tt <= 0): return 1e15
        z = tt / eta
        ll = np.sum(e * (np.log(beta+1e-15) - np.log(eta+1e-15) + (beta-1)*_safe_log(z)) - z**beta)
        return -ll
    gamma0 = max(t.min()*0.5, 0)
    res = minimize(neg_ll, [1.0, np.median(t), gamma0], method='Nelder-Mead', options={'maxiter':8000})
    beta, eta, gamma = max(abs(res.x[0]),0.01), max(abs(res.x[1]),0.01), max(res.x[2], 0)
    return {'β (shape)': beta, 'η (scale)': eta, 'γ (location)': gamma}, -res.fun, 3

# --- LOG-NORMAL ---
def fit_lognormal2p(t, e):
    """Log-Normal 2P: ln(T) ~ N(μ, σ²)"""
    def neg_ll(params):
        mu, sigma = params
        if sigma <= 0: return 1e15
        z = (np.log(t) - mu) / sigma
        ll = np.sum(e * (stats.norm.logpdf(z) - np.log(sigma) - np.log(t+1e-15)) + (1-e)*stats.norm.logsf(z))
        return -ll
    t_obs = t[e==1]
    mu0 = np.mean(np.log(t_obs)) if len(t_obs)>0 else 0
    s0 = np.std(np.log(t_obs), ddof=1) if len(t_obs)>1 else 1
    res = minimize(neg_ll, [mu0, max(s0,0.1)], method='Nelder-Mead', options={'maxiter':5000})
    return {'μ': res.x[0], 'σ': max(abs(res.x[1]),0.01)}, -res.fun, 2

def fit_lognormal3p(t, e):
    """Log-Normal 3P (shifted): ln(T-γ) ~ N(μ, σ²)"""
    def neg_ll(params):
        mu, sigma, gamma = params
        if sigma <= 0 or gamma < 0: return 1e15
        tt = t - gamma
        if np.any(tt <= 0): return 1e15
        z = (np.log(tt) - mu) / sigma
        ll = np.sum(e*(stats.norm.logpdf(z) - np.log(sigma) - np.log(tt+1e-15)) + (1-e)*stats.norm.logsf(z))
        return -ll
    t_obs = t[e==1]; gamma0 = max(t.min()*0.3, 0)
    mu0 = np.mean(np.log(np.maximum(t_obs-gamma0, 0.01))) if len(t_obs)>0 else 0
    s0 = max(np.std(np.log(np.maximum(t_obs-gamma0, 0.01)), ddof=1), 0.1) if len(t_obs)>1 else 1
    res = minimize(neg_ll, [mu0, s0, gamma0], method='Nelder-Mead', options={'maxiter':8000})
    return {'μ': res.x[0], 'σ': max(abs(res.x[1]),0.01), 'γ': max(res.x[2],0)}, -res.fun, 3

# --- LOG-LOGISTIC ---
def fit_loglogistic2p(t, e):
    """Log-Logistic 2P: S(t)=1/(1+(t/α)^β)"""
    def neg_ll(params):
        alpha, beta = params
        if alpha <= 0 or beta <= 0: return 1e15
        z = (t/alpha)**beta
        ll = np.sum(e*(np.log(beta+1e-15)-np.log(alpha+1e-15)+(beta-1)*_safe_log(t/alpha)-2*np.log(1+z)) + (1-e)*(-np.log(1+z)))
        return -ll
    res = minimize(neg_ll, [np.median(t), 1.0], method='Nelder-Mead', options={'maxiter':5000})
    return {'α (scale)': max(abs(res.x[0]),0.01), 'β (shape)': max(abs(res.x[1]),0.01)}, -res.fun, 2

def fit_loglogistic3p(t, e):
    """Log-Logistic 3P (shifted): S(t)=1/(1+((t-γ)/α)^β)"""
    def neg_ll(params):
        alpha, beta, gamma = params
        if alpha <= 0 or beta <= 0 or gamma < 0: return 1e15
        tt = t - gamma
        if np.any(tt <= 0): return 1e15
        z = (tt/alpha)**beta
        ll = np.sum(e*(np.log(beta+1e-15)-np.log(alpha+1e-15)+(beta-1)*_safe_log(tt/alpha)-2*np.log(1+z)) + (1-e)*(-np.log(1+z)))
        return -ll
    gamma0 = max(t.min()*0.3, 0)
    res = minimize(neg_ll, [np.median(t), 1.0, gamma0], method='Nelder-Mead', options={'maxiter':8000})
    return {'α (scale)': max(abs(res.x[0]),0.01), 'β (shape)': max(abs(res.x[1]),0.01), 'γ': max(res.x[2],0)}, -res.fun, 3

# --- GAMMA ---
def fit_gamma2p(t, e):
    """Gamma 2P: f(t) = β^α/Γ(α) · t^(α-1) · exp(-βt)"""
    def neg_ll(params):
        a, b = params
        if a <= 0 or b <= 0: return 1e15
        ll = np.sum(e*(a*np.log(b+1e-15)-gammaln(a)+(a-1)*_safe_log(t)-b*t) + (1-e)*_safe_log(gammaincc(a, b*t)+1e-15))
        return -ll
    t_obs = t[e==1]; m = np.mean(t_obs) if len(t_obs)>0 else np.mean(t)
    v = np.var(t_obs) if len(t_obs)>1 else m**2
    res = minimize(neg_ll, [max(m**2/(v+1e-15),0.5), max(m/(v+1e-15),0.01)], method='Nelder-Mead', options={'maxiter':5000})
    return {'α (shape)': max(abs(res.x[0]),0.01), 'β (rate)': max(abs(res.x[1]),0.001)}, -res.fun, 2

def fit_gamma3p(t, e):
    """Gamma 3P (shifted): same but with location γ"""
    def neg_ll(params):
        a, b, gamma = params
        if a <= 0 or b <= 0 or gamma < 0: return 1e15
        tt = t - gamma
        if np.any(tt <= 0): return 1e15
        ll = np.sum(e*(a*np.log(b+1e-15)-gammaln(a)+(a-1)*_safe_log(tt)-b*tt) + (1-e)*_safe_log(gammaincc(a, b*tt)+1e-15))
        return -ll
    t_obs = t[e==1]; gamma0 = max(t.min()*0.3, 0)
    tt0 = np.maximum(t_obs-gamma0, 0.01); m = np.mean(tt0); v = np.var(tt0)+1e-15
    res = minimize(neg_ll, [max(m**2/v,0.5), max(m/v,0.01), gamma0], method='Nelder-Mead', options={'maxiter':8000})
    return {'α (shape)': max(abs(res.x[0]),0.01), 'β (rate)': max(abs(res.x[1]),0.001), 'γ': max(res.x[2],0)}, -res.fun, 3

# --- GOMPERTZ ---
def fit_gompertz2p(t, e):
    """Gompertz 2P: h(t)=b·exp(a·t), S(t)=exp(-(b/a)(exp(at)-1))"""
    def neg_ll(params):
        a, b = params
        if b <= 0: return 1e15
        if abs(a) < 1e-10: a = 1e-10
        H = (b/a)*(np.exp(a*t)-1)
        if np.any(H < 0): return 1e15
        h = b * np.exp(a*t)
        ll = np.sum(e*_safe_log(h) - H)
        return -ll
    res = minimize(neg_ll, [0.01, 0.01], method='Nelder-Mead', options={'maxiter':5000})
    return {'a (shape)': res.x[0], 'b (scale)': max(abs(res.x[1]),1e-6)}, -res.fun, 2

# --- GENERALIZED GAMMA (Stacy) ---
def fit_gengamma3p(t, e):
    """Generalized Gamma 3P: params (μ, σ, Q)"""
    def neg_ll(params):
        mu, sigma, Q = params
        if sigma <= 0: return 1e15
        if abs(Q) < 1e-10: Q = 1e-10
        gv = abs(Q)**(-2)
        w = (np.log(t+1e-15) - mu) / sigma
        ll = 0
        for i in range(len(t)):
            wi = w[i]
            if Q > 0:
                u = gv * np.exp(abs(Q)*wi)
                if e[i] == 1:
                    ll += np.log(abs(Q)+1e-15)+gv*np.log(gv)-np.log(sigma)-np.log(t[i]+1e-15)-gammaln(gv)+gv*abs(Q)*wi-u
                else:
                    ll += _safe_log(gammaincc(gv, u)+1e-15)
            else:
                u = gv * np.exp(-abs(Q)*wi)
                if e[i] == 1:
                    ll += np.log(abs(Q)+1e-15)+gv*np.log(gv)-np.log(sigma)-np.log(t[i]+1e-15)-gammaln(gv)+gv*(-abs(Q))*wi-u
                else:
                    ll += _safe_log(1-gammaincc(gv, u)+1e-15)
            if not np.isfinite(ll): return 1e15
        return -ll
    t_obs = t[e==1]
    mu0 = np.mean(np.log(t_obs)) if len(t_obs)>0 else np.mean(np.log(t))
    s0 = np.std(np.log(t_obs), ddof=1) if len(t_obs)>1 else 1
    res = minimize(neg_ll, [mu0, max(s0,0.1), 1.0], method='Nelder-Mead', options={'maxiter':10000})
    return {'μ': res.x[0], 'σ': max(abs(res.x[1]),0.01), 'Q': res.x[2]}, -res.fun, 3

# --- INVERSE GAUSSIAN ---
def fit_invgauss2p(t, e):
    """Inverse Gaussian 2P: f(t) = sqrt(λ/(2πt³)) exp(-λ(t-μ)²/(2μ²t))"""
    def neg_ll(params):
        mu, lam = params
        if mu <= 0 or lam <= 0: return 1e15
        ll = 0
        for i in range(len(t)):
            if e[i] == 1:
                ll += 0.5*(np.log(lam+1e-15)-np.log(2*np.pi)-3*np.log(t[i]+1e-15))-lam*(t[i]-mu)**2/(2*mu**2*t[i]+1e-15)
            else:
                S = 1 - stats.invgauss.cdf(t[i], mu=mu/lam, scale=lam)
                ll += _safe_log(S+1e-15)
        return -ll
    t_obs = t[e==1]; mu0 = np.mean(t_obs) if len(t_obs)>0 else np.mean(t)
    lam0 = mu0**3/(np.var(t_obs)+1e-15) if len(t_obs)>1 else mu0
    res = minimize(neg_ll, [max(mu0,0.1), max(lam0,0.1)], method='Nelder-Mead', options={'maxiter':5000})
    return {'μ': max(abs(res.x[0]),0.001), 'λ': max(abs(res.x[1]),0.001)}, -res.fun, 2

# --- RAYLEIGH (Weibull with β=2) ---
def fit_rayleigh1p(t, e):
    """Rayleigh 1P: S(t)=exp(-t²/(2σ²))  [Weibull β=2, η=σ√2]"""
    def neg_ll(params):
        sigma = params[0]
        if sigma <= 0: return 1e15
        ll = np.sum(e*(np.log(t+1e-15)-2*np.log(sigma+1e-15)) - t**2/(2*sigma**2))
        return -ll
    res = minimize(neg_ll, [np.std(t)], method='Nelder-Mead', options={'maxiter':3000})
    return {'σ': max(abs(res.x[0]),0.01)}, -res.fun, 1

# --- BIRNBAUM-SAUNDERS ---
def fit_birnsaun2p(t, e):
    """Birnbaum-Saunders 2P: params (α, β)"""
    def neg_ll(params):
        alpha, beta = params
        if alpha <= 0 or beta <= 0: return 1e15
        ll = 0
        for i in range(len(t)):
            a_t = (1/alpha)*(np.sqrt(t[i]/beta) - np.sqrt(beta/t[i]))
            da_t = (1/(2*alpha))*(1/np.sqrt(beta*t[i]) + np.sqrt(beta)/t[i]**1.5) if t[i] > 0 else 0
            if e[i] == 1:
                ll += stats.norm.logpdf(a_t) + np.log(da_t+1e-15)
            else:
                ll += stats.norm.logsf(a_t)
        return -ll
    t_obs = t[e==1]
    beta0 = np.median(t_obs) if len(t_obs)>0 else np.median(t)
    res = minimize(neg_ll, [0.5, max(beta0,0.1)], method='Nelder-Mead', options={'maxiter':5000})
    return {'α (shape)': max(abs(res.x[0]),0.01), 'β (scale)': max(abs(res.x[1]),0.01)}, -res.fun, 2

# ============================================================
# SURVIVAL & HAZARD EVALUATORS
# ============================================================
def surv_exp1p(t_grid, p):
    return np.exp(-p['λ']*t_grid)
def haz_exp1p(t_grid, p):
    return np.ones_like(t_grid)*p['λ']
def surv_exp2p(t_grid, p):
    tt = np.maximum(t_grid - p['γ'], 1e-15)
    return np.where(t_grid > p['γ'], np.exp(-p['λ']*tt), 1.0)
def haz_exp2p(t_grid, p):
    return np.where(t_grid > p['γ'], p['λ'], 0.0)

def surv_weibull2p(t_grid, p):
    return np.exp(-(t_grid/p['η (scale)'])**p['β (shape)'])
def haz_weibull2p(t_grid, p):
    b, eta = p['β (shape)'], p['η (scale)']
    return (b/eta)*(t_grid/eta)**(b-1)
def surv_weibull3p(t_grid, p):
    tt = np.maximum(t_grid - p['γ (location)'], 1e-15)
    S = np.exp(-(tt/p['η (scale)'])**p['β (shape)'])
    return np.where(t_grid > p['γ (location)'], S, 1.0)
def haz_weibull3p(t_grid, p):
    tt = np.maximum(t_grid - p['γ (location)'], 1e-15)
    b, eta = p['β (shape)'], p['η (scale)']
    return np.where(t_grid > p['γ (location)'], (b/eta)*(tt/eta)**(b-1), 0.0)

def surv_ln2p(t_grid, p):
    return 1 - stats.norm.cdf((np.log(t_grid)-p['μ'])/p['σ'])
def haz_ln2p(t_grid, p):
    z = (np.log(t_grid)-p['μ'])/p['σ']; S = surv_ln2p(t_grid, p)
    f = stats.norm.pdf(z)/(p['σ']*t_grid+1e-15)
    return f/(S+1e-15)
def surv_ln3p(t_grid, p):
    tt = np.maximum(t_grid - p['γ'], 1e-15)
    return np.where(t_grid > p['γ'], 1 - stats.norm.cdf((np.log(tt)-p['μ'])/p['σ']), 1.0)
def haz_ln3p(t_grid, p):
    dt = t_grid[1]-t_grid[0]; S = surv_ln3p(t_grid, p)
    h = np.gradient(-_safe_log(S+1e-15), dt); return np.clip(h, 0, None)

def surv_ll2p(t_grid, p):
    return 1/(1+(t_grid/p['α (scale)'])**p['β (shape)'])
def haz_ll2p(t_grid, p):
    a, b = p['α (scale)'], p['β (shape)']
    return ((b/a)*(t_grid/a)**(b-1))/(1+(t_grid/a)**b)
def surv_ll3p(t_grid, p):
    tt = np.maximum(t_grid - p['γ'], 1e-15)
    return np.where(t_grid > p['γ'], 1/(1+(tt/p['α (scale)'])**p['β (shape)']), 1.0)
def haz_ll3p(t_grid, p):
    dt = t_grid[1]-t_grid[0]; S = surv_ll3p(t_grid, p)
    h = np.gradient(-_safe_log(S+1e-15), dt); return np.clip(h, 0, None)

def surv_gamma2p(t_grid, p):
    return np.array([gammaincc(p['α (shape)'], p['β (rate)']*ti) for ti in t_grid])
def haz_gamma2p(t_grid, p):
    a, b = p['α (shape)'], p['β (rate)']
    S = surv_gamma2p(t_grid, p)
    f = np.array([np.exp(a*np.log(b+1e-15)-gammaln(a)+(a-1)*np.log(ti+1e-15)-b*ti) for ti in t_grid])
    return f/(S+1e-15)
def surv_gamma3p(t_grid, p):
    tt = np.maximum(t_grid - p['γ'], 1e-15)
    S = np.array([gammaincc(p['α (shape)'], p['β (rate)']*ti) for ti in tt])
    return np.where(t_grid > p['γ'], S, 1.0)
def haz_gamma3p(t_grid, p):
    dt = t_grid[1]-t_grid[0]; S = surv_gamma3p(t_grid, p)
    h = np.gradient(-_safe_log(S+1e-15), dt); return np.clip(h, 0, None)

def surv_gompertz2p(t_grid, p):
    a, b = p['a (shape)'], p['b (scale)']
    return np.array([np.exp(-(b/a)*(np.exp(a*ti)-1)) if abs(a)>1e-10 else np.exp(-b*ti) for ti in t_grid])
def haz_gompertz2p(t_grid, p):
    return p['b (scale)'] * np.exp(p['a (shape)']*t_grid)

def surv_gengamma3p(t_grid, p):
    mu, sigma, Q = p['μ'], p['σ'], p['Q']
    if abs(Q) < 1e-10: Q = 1e-10
    gv = abs(Q)**(-2)
    def _s(ti):
        w = (np.log(ti+1e-15)-mu)/sigma
        if Q > 0: return gammaincc(gv, gv*np.exp(abs(Q)*w))
        else: return 1 - gammaincc(gv, gv*np.exp(-abs(Q)*w))
    return np.array([_s(ti) for ti in t_grid])
def haz_gengamma3p(t_grid, p):
    dt = t_grid[1]-t_grid[0]; S = surv_gengamma3p(t_grid, p)
    h = np.gradient(-_safe_log(S+1e-15), dt); return np.clip(h, 0, None)

def surv_invgauss2p(t_grid, p):
    mu, lam = p['μ'], p['λ']
    return np.array([1 - stats.invgauss.cdf(ti, mu=mu/lam, scale=lam) for ti in t_grid])
def haz_invgauss2p(t_grid, p):
    mu, lam = p['μ'], p['λ']; S = surv_invgauss2p(t_grid, p)
    f = np.array([np.sqrt(lam/(2*np.pi*ti**3+1e-15))*np.exp(-lam*(ti-mu)**2/(2*mu**2*ti+1e-15)) for ti in t_grid])
    return f/(S+1e-15)

def surv_rayleigh1p(t_grid, p):
    return np.exp(-t_grid**2/(2*p['σ']**2))
def haz_rayleigh1p(t_grid, p):
    return t_grid/p['σ']**2

def surv_birnsaun2p(t_grid, p):
    a, b = p['α (shape)'], p['β (scale)']
    return np.array([1 - stats.norm.cdf((1/a)*(np.sqrt(ti/b)-np.sqrt(b/ti))) for ti in t_grid])
def haz_birnsaun2p(t_grid, p):
    dt = t_grid[1]-t_grid[0]; S = surv_birnsaun2p(t_grid, p)
    h = np.gradient(-_safe_log(S+1e-15), dt); return np.clip(h, 0, None)

# Model registry
ALL_MODELS = {
    'Exponential 1P': {'fit': fit_exp1p, 'surv': surv_exp1p, 'haz': haz_exp1p,
        'formula': 'S(t) = exp(-λt)', 'hz_type': 'Constant'},
    'Exponential 2P': {'fit': fit_exp2p, 'surv': surv_exp2p, 'haz': haz_exp2p,
        'formula': 'S(t) = exp(-λ(t−γ)), t>γ', 'hz_type': 'Constant (shifted)'},
    'Weibull 2P': {'fit': fit_weibull2p, 'surv': surv_weibull2p, 'haz': haz_weibull2p,
        'formula': 'S(t) = exp(−(t/η)^β)', 'hz_type': 'Monoton'},
    'Weibull 3P': {'fit': fit_weibull3p, 'surv': surv_weibull3p, 'haz': haz_weibull3p,
        'formula': 'S(t) = exp(−((t−γ)/η)^β)', 'hz_type': 'Monoton (shifted)'},
    'Log-Normal 2P': {'fit': fit_lognormal2p, 'surv': surv_ln2p, 'haz': haz_ln2p,
        'formula': 'ln(T) ~ N(μ, σ²)', 'hz_type': 'Non-monoton (↑↓)'},
    'Log-Normal 3P': {'fit': fit_lognormal3p, 'surv': surv_ln3p, 'haz': haz_ln3p,
        'formula': 'ln(T−γ) ~ N(μ, σ²)', 'hz_type': 'Non-monoton (shifted)'},
    'Log-Logistic 2P': {'fit': fit_loglogistic2p, 'surv': surv_ll2p, 'haz': haz_ll2p,
        'formula': 'S(t) = 1/(1+(t/α)^β)', 'hz_type': 'Non-monoton / ↓'},
    'Log-Logistic 3P': {'fit': fit_loglogistic3p, 'surv': surv_ll3p, 'haz': haz_ll3p,
        'formula': 'S(t) = 1/(1+((t−γ)/α)^β)', 'hz_type': 'Non-monoton (shifted)'},
    'Gamma 2P': {'fit': fit_gamma2p, 'surv': surv_gamma2p, 'haz': haz_gamma2p,
        'formula': 'f = β^α/Γ(α)·t^(α−1)·e^(−βt)', 'hz_type': 'Monoton'},
    'Gamma 3P': {'fit': fit_gamma3p, 'surv': surv_gamma3p, 'haz': haz_gamma3p,
        'formula': 'f(t−γ), shifted Gamma', 'hz_type': 'Monoton (shifted)'},
    'Gompertz 2P': {'fit': fit_gompertz2p, 'surv': surv_gompertz2p, 'haz': haz_gompertz2p,
        'formula': 'h(t) = b·exp(a·t)', 'hz_type': 'Eksponensial ↑'},
    'Gen. Gamma 3P': {'fit': fit_gengamma3p, 'surv': surv_gengamma3p, 'haz': haz_gengamma3p,
        'formula': 'Stacy GenGamma(μ,σ,Q)', 'hz_type': 'Sangat fleksibel'},
    'Inverse Gaussian 2P': {'fit': fit_invgauss2p, 'surv': surv_invgauss2p, 'haz': haz_invgauss2p,
        'formula': 'f = √(λ/2πt³)·exp(−λ(t−μ)²/2μ²t)', 'hz_type': 'Non-monoton (↑↓)'},
    'Rayleigh 1P': {'fit': fit_rayleigh1p, 'surv': surv_rayleigh1p, 'haz': haz_rayleigh1p,
        'formula': 'S(t) = exp(−t²/2σ²)', 'hz_type': 'Linear ↑'},
    'Birnbaum-Saunders 2P': {'fit': fit_birnsaun2p, 'surv': surv_birnsaun2p, 'haz': haz_birnsaun2p,
        'formula': 'Φ((1/α)(√(t/β)−√(β/t)))', 'hz_type': 'Non-monoton'},
}

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.header("📁 Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV / Excel", type=["csv", "xlsx", "xls"])
use_demo = st.sidebar.checkbox("Gunakan data demo", value=True if uploaded_file is None else False)

if uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'): df = pd.read_csv(uploaded_file)
    else: df = pd.read_excel(uploaded_file)
    use_demo = False
elif use_demo:
    np.random.seed(42); n = 200
    group = np.random.choice(['Treatment','Control'], n, p=[0.5,0.5])
    age = np.random.normal(55, 12, n).round(0).astype(int)
    stage = np.random.choice([1,2,3,4], n, p=[0.2,0.3,0.3,0.2])
    base_time_t = np.random.weibull(1.5, n)*30
    base_time_c = np.random.weibull(1.2, n)*20
    time = np.where(group=='Treatment', base_time_t, base_time_c)
    time = np.clip(time, 0.5, 100).round(1)
    censor_time = np.random.uniform(10, 80, n)
    observed_time = np.minimum(time, censor_time).round(1)
    event = (time <= censor_time).astype(int)
    df = pd.DataFrame({'Patient_ID': range(1,n+1),'Time': observed_time,'Event': event,
                        'Group': group,'Age': age,'Stage': stage})
    st.sidebar.success(f"Data demo: {n} pasien, {event.sum()} events, {n-event.sum()} censored")
else:
    st.warning("Silakan upload data atau gunakan data demo."); st.stop()

# ============================================================
# 1. EKSPLORASI
# ============================================================
st.header("1. Eksplorasi Data")
c1, c2, c3, c4 = st.columns(4)
c1.metric("N Observasi", df.shape[0]); c2.metric("Variabel", df.shape[1])
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(include=['object','category']).columns.tolist()
tab_d1, tab_d2 = st.tabs(["Data","Deskriptif"])
with tab_d1: st.dataframe(df.head(30), use_container_width=True)
with tab_d2: st.dataframe(df.describe(include='all').T.round(4), use_container_width=True)

# ============================================================
# 2. SETUP VARIABEL
# ============================================================
st.header("2. Setup Variabel Survival")
col_s1, col_s2, col_s3 = st.columns(3)
with col_s1:
    time_col = st.selectbox("Variabel Waktu (T)", numeric_cols,
                            index=numeric_cols.index('Time') if 'Time' in numeric_cols else 0)
with col_s2:
    event_col = st.selectbox("Variabel Event/Status (1=event, 0=censored)", numeric_cols,
                             index=numeric_cols.index('Event') if 'Event' in numeric_cols else 0)
with col_s3:
    group_options = ['(Tidak ada)'] + cat_cols + [c for c in numeric_cols if df[c].nunique() <= 10 and c != time_col and c != event_col]
    group_col = st.selectbox("Variabel Grup (opsional)", group_options, index=0)

alpha = st.selectbox("α (signifikansi)", [0.01, 0.05, 0.10], index=1)
panel = df[[time_col, event_col] + ([group_col] if group_col != '(Tidak ada)' else [])].dropna().copy()
durations = panel[time_col].values.astype(float)
events = panel[event_col].values.astype(int)
N = len(durations); n_events = int(events.sum()); n_censor = N - n_events
c3.metric("Events", n_events); c4.metric("Censored", f"{n_censor} ({n_censor/N:.1%})")
st.info(f"**N = {N}** | Events: {n_events} ({n_events/N:.1%}) | Censored: {n_censor} ({n_censor/N:.1%}) | "
        f"Median time: {np.median(durations):.2f} | Range: [{durations.min():.2f}, {durations.max():.2f}]")

# ============================================================
# 3. ANALISIS TIPE PENYENSORAN
# ============================================================
st.header("3. Analisis Tipe Penyensoran")

cens_type_options = ["Auto-Detect", "Tipe I (Fixed Time)", "Tipe II (Fixed Number)", "Random / Tipe III"]
cens_type_choice = st.radio("Pilih atau deteksi otomatis tipe penyensoran:", cens_type_options, horizontal=True)

auto_cens_type, auto_cens_info = detect_censoring_type(durations, events)

if cens_type_choice == "Auto-Detect":
    cens_type_used = auto_cens_type
    cens_info_used = auto_cens_info
else:
    cens_type_used = cens_type_choice
    if 'Tipe I' in cens_type_choice:
        cens_times_arr = durations[events == 0]
        Tc = cens_times_arr.max() if len(cens_times_arr) > 0 else durations.max()
        cens_info_used = {'T_censor': Tc, 'desc': f'Manual: Tipe I dengan T_c = {Tc:.2f}'}
    elif 'Tipe II' in cens_type_choice:
        cens_info_used = {'r': n_events, 'n': N,
                          'desc': f'Manual: Tipe II dengan r = {n_events} events dari n = {N}'}
    else:
        cens_info_used = {'desc': 'Manual: Random/Independent censoring'}

cens_md = censoring_analysis_info(durations, events, cens_type_used, cens_info_used)
st.markdown(cens_md)

# Censoring diagnostics
col_cd1, col_cd2 = st.columns(2)
with col_cd1:
    cens_times_plot = durations[events == 0]
    evt_times_plot = durations[events == 1]
    fig_cens = go.Figure()
    fig_cens.add_trace(go.Histogram(x=evt_times_plot, nbinsx=20, name='Event', marker_color='salmon', opacity=0.7))
    fig_cens.add_trace(go.Histogram(x=cens_times_plot, nbinsx=20, name='Censored', marker_color='steelblue', opacity=0.7))
    fig_cens.update_layout(barmode='overlay', title="Distribusi Waktu: Event vs Censored",
                           xaxis_title="Time", yaxis_title="Count", height=380)
    st.plotly_chart(fig_cens, use_container_width=True)
with col_cd2:
    sorted_dur = np.sort(durations)
    sorted_evt = events[np.argsort(durations)]
    fig_swim = go.Figure()
    for i in range(min(60, N)):
        color = 'red' if sorted_evt[i] == 1 else 'steelblue'
        sym = 'x' if sorted_evt[i] == 1 else 'circle-open'
        fig_swim.add_trace(go.Scatter(x=[0, sorted_dur[i]], y=[i,i], mode='lines',
                                      line=dict(color=color, width=1.2), showlegend=False))
        fig_swim.add_trace(go.Scatter(x=[sorted_dur[i]], y=[i], mode='markers',
                                      marker=dict(symbol=sym, size=6, color=color), showlegend=False))
    if 'Tipe I' in cens_type_used and 'T_censor' in cens_info_used:
        fig_swim.add_vline(x=cens_info_used['T_censor'], line_dash="dash", line_color="green",
                           annotation_text=f"T_c={cens_info_used['T_censor']:.1f}")
    if 'Tipe II' in cens_type_used and n_events > 0:
        t_r = np.sort(durations[events==1])[-1]
        fig_swim.add_vline(x=t_r, line_dash="dash", line_color="orange",
                           annotation_text=f"t_(r)={t_r:.1f}")
    fig_swim.update_layout(title="Swimmer Plot (sorted)", xaxis_title="Time",
                           yaxis_title="Subject", height=380)
    st.plotly_chart(fig_swim, use_container_width=True)

# Type-specific summary
if 'Tipe I' in cens_type_used:
    Tc = cens_info_used.get('T_censor', durations[events==0].max() if n_censor > 0 else durations.max())
    st.markdown(f"""
    ### Ringkasan Tipe I
    | Statistik | Nilai |
    |-----------|-------|
    | Waktu sensor tetap (T_c) | **{Tc:.2f}** |
    | N total | {N} |
    | Events (d) | {n_events} |
    | Censored (n−d) | {n_censor} |
    | % Censored | {n_censor/N:.1%} |
    | Event rate | {n_events/Tc:.4f} per unit waktu |
    """)
elif 'Tipe II' in cens_type_used:
    t_r = np.sort(durations[events==1])[-1] if n_events > 0 else 0
    st.markdown(f"""
    ### Ringkasan Tipe II
    | Statistik | Nilai |
    |-----------|-------|
    | Jumlah event tetap (r) | **{n_events}** |
    | N total | {N} |
    | Waktu henti t_(r) | **{t_r:.2f}** |
    | Censored di t_(r) | {n_censor} |
    | Fraction failed | {n_events/N:.1%} |
    """)

# ============================================================
# 4. KAPLAN-MEIER
# ============================================================
st.header("4. Estimasi Kaplan-Meier")
if group_col == '(Tidak ada)':
    km = kaplan_meier(durations, events)
    med_surv = median_survival(km)
    st.markdown(f"**Median Survival Time:** {med_surv if not np.isnan(med_surv) else 'Belum tercapai'}")
    with st.expander("Tabel Life Table Kaplan-Meier"):
        st.dataframe(km, use_container_width=True, hide_index=True)
    fig_km = go.Figure()
    fig_km.add_trace(go.Scatter(x=km['Time'], y=km['S(t)'], mode='lines', name='S(t)',
                                line=dict(color='blue', width=2, shape='hv')))
    fig_km.add_trace(go.Scatter(x=km['Time'], y=km['CI_upper'], mode='lines',
                                line=dict(color='lightblue', width=1, dash='dash', shape='hv'), showlegend=False))
    fig_km.add_trace(go.Scatter(x=km['Time'], y=km['CI_lower'], mode='lines',
                                line=dict(color='lightblue', width=1, dash='dash', shape='hv'),
                                fill='tonexty', fillcolor='rgba(173,216,230,0.3)', showlegend=False))
    cm = (events == 0); ct = durations[cm]
    kac = [km[km['Time']<=c]['S(t)'].iloc[-1] if len(km[km['Time']<=c])>0 else 1.0 for c in ct]
    fig_km.add_trace(go.Scatter(x=ct, y=kac, mode='markers', name='Censored',
                                marker=dict(symbol='cross', size=6, color='red')))
    fig_km.add_hline(y=0.5, line_dash="dot", line_color="gray", annotation_text="S=0.5")
    fig_km.update_layout(title="Kaplan-Meier Survival Curve", xaxis_title="Time",
                         yaxis_title="S(t)", height=500, yaxis=dict(range=[0,1.05]))
    st.plotly_chart(fig_km, use_container_width=True)
else:
    groups = panel[group_col].unique()
    km_by_group = {}; colors = px.colors.qualitative.Plotly; fig_km = go.Figure(); med_rows = []
    for i, g in enumerate(sorted(groups)):
        mask = panel[group_col]==g; dur_g = durations[mask]; evt_g = events[mask]
        km_g = kaplan_meier(dur_g, evt_g); km_by_group[g] = km_g
        med_rows.append({'Group': g, 'N': int(mask.sum()), 'Events': int(evt_g.sum()),
                         'Censored': int((evt_g==0).sum()), 'Median Survival': median_survival(km_g)})
        col = colors[i % len(colors)]
        fig_km.add_trace(go.Scatter(x=km_g['Time'], y=km_g['S(t)'], mode='lines',
                                    name=str(g), line=dict(color=col, width=2, shape='hv')))
        fig_km.add_trace(go.Scatter(x=km_g['Time'], y=km_g['CI_upper'], mode='lines',
                                    line=dict(color=col, width=0.5, dash='dash', shape='hv'), showlegend=False))
        fig_km.add_trace(go.Scatter(x=km_g['Time'], y=km_g['CI_lower'], mode='lines',
                                    line=dict(color=col, width=0.5, dash='dash', shape='hv'),
                                    fill='tonexty', fillcolor='rgba(150,150,150,0.1)', showlegend=False))
    fig_km.add_hline(y=0.5, line_dash="dot", line_color="gray")
    fig_km.update_layout(title="Kaplan-Meier by Group", xaxis_title="Time",
                         yaxis_title="S(t)", height=550, yaxis=dict(range=[0,1.05]))
    st.plotly_chart(fig_km, use_container_width=True)
    st.dataframe(pd.DataFrame(med_rows), use_container_width=True, hide_index=True)

    # LOG-RANK & VARIANTS
    st.header("5. Uji Perbandingan Kurva Survival")
    if len(groups) == 2:
        g1, g2 = sorted(groups)
        m1 = panel[group_col]==g1; m2 = panel[group_col]==g2
        dd1, ee1 = durations[m1], events[m1]; dd2, ee2 = durations[m2], events[m2]
        lr_chi2, lr_p, *_ = log_rank_test(dd1, ee1, dd2, ee2)
        gw_chi2, gw_p = wilcoxon_gehan_test(dd1, ee1, dd2, ee2)
        tw_chi2, tw_p = tarone_ware_test(dd1, ee1, dd2, ee2)
        pp_chi2, pp_p = peto_peto_test(dd1, ee1, dd2, ee2)
        test_df = pd.DataFrame({
            'Uji': ['Log-Rank','Gehan-Wilcoxon','Tarone-Ware','Peto-Peto'],
            'Bobot': ['1','nⱼ','√nⱼ','S_KM(t)'],
            'χ²': [round(lr_chi2,4),round(gw_chi2,4),round(tw_chi2,4),round(pp_chi2,4)],
            'p-value': [round(lr_p,6),round(gw_p,6),round(tw_p,6),round(pp_p,6)],
            f'Keputusan (α={alpha})': ['Berbeda' if p < alpha else 'Tidak Berbeda'
                                        for p in [lr_p, gw_p, tw_p, pp_p]]
        })
        st.dataframe(test_df, use_container_width=True, hide_index=True)
    else:
        st.info("Uji perbandingan saat ini mendukung 2 grup.")

# ============================================================
# NELSON-AALEN
# ============================================================
sec_na = "6" if group_col != '(Tidak ada)' else "5"
st.header(f"{sec_na}. Estimasi Nelson-Aalen (Cumulative Hazard)")
na = nelson_aalen(durations, events)
with st.expander("Tabel Nelson-Aalen"):
    st.dataframe(na, use_container_width=True, hide_index=True)
fig_na = go.Figure()
if group_col == '(Tidak ada)':
    fig_na.add_trace(go.Scatter(x=na['Time'], y=na['H(t)'], mode='lines', name='H(t)', line=dict(width=2, shape='hv')))
else:
    for i, g in enumerate(sorted(groups)):
        mask = panel[group_col]==g; na_g = nelson_aalen(durations[mask], events[mask])
        fig_na.add_trace(go.Scatter(x=na_g['Time'], y=na_g['H(t)'], mode='lines', name=str(g),
                                    line=dict(width=2, shape='hv', color=colors[i%len(colors)])))
fig_na.update_layout(title="Nelson-Aalen Cumulative Hazard", xaxis_title="Time", yaxis_title="H(t)", height=450)
st.plotly_chart(fig_na, use_container_width=True)

# HAZARD FUNCTION
sec_hz = str(int(sec_na)+1)
st.header(f"{sec_hz}. Estimasi Fungsi Hazard (Kernel Smoothed)")
bandwidth = st.slider("Bandwidth", 1.0, 20.0, 5.0, 0.5)
tg_hz = np.linspace(durations.min(), durations.max(), 200)
et = durations[events==1]; hz_est = np.zeros(len(tg_hz))
for e_t in et: hz_est += stats.norm.pdf((tg_hz-e_t)/bandwidth)/bandwidth
nr_g = np.array([np.sum(durations>=t) for t in tg_hz])
hz_est = hz_est / np.maximum(nr_g, 1)
fig_hz = go.Figure()
fig_hz.add_trace(go.Scatter(x=tg_hz, y=hz_est, mode='lines', name='h(t) smoothed'))
fig_hz.update_layout(title="Smoothed Hazard Function", xaxis_title="Time", yaxis_title="h(t)", height=400)
st.plotly_chart(fig_hz, use_container_width=True)

# ============================================================
# COX PH
# ============================================================
sec_cox = str(int(sec_hz)+1)
st.header(f"{sec_cox}. Cox Proportional Hazards Model")
cov_opts = [c for c in numeric_cols if c != time_col and c != event_col]
covariates = st.multiselect("Pilih Kovariat", cov_opts, default=cov_opts[:min(3, len(cov_opts))])
if len(covariates) >= 1:
    pc = df[[time_col, event_col]+covariates].dropna()
    dc = pc[time_col].values.astype(float); ec = pc[event_col].values.astype(int)
    Xc = pc[covariates].values.astype(float)
    Xm = Xc.mean(0); Xs = Xc.std(0); Xs[Xs==0]=1; Xsc = (Xc-Xm)/Xs
    beta, se_b, z_b, pv_b, hr_b = cox_ph_simple(dc, ec, Xsc)
    bo = beta/Xs; so = se_b/Xs; zo = bo/so
    po = 2*(1-stats.norm.cdf(np.abs(zo))); ho = np.exp(bo)
    cox_df = pd.DataFrame({
        'Kovariat': covariates, 'β': bo.round(6), 'SE': so.round(6), 'z': zo.round(4),
        'p-value': po.round(6), 'HR (eᵝ)': ho.round(4),
        'HR CI Lo': np.exp(bo-1.96*so).round(4), 'HR CI Hi': np.exp(bo+1.96*so).round(4),
        f'Sig (α={alpha})': ['Ya' if p < alpha else 'Tidak' for p in po]
    })
    st.dataframe(cox_df, use_container_width=True, hide_index=True)
    rs = Xc @ bo; conc = 0; disc = 0; tied = 0
    eidx = np.where(ec==1)[0]
    for i in eidx[:min(500, len(eidx))]:
        for j in range(len(dc)):
            if dc[j] > dc[i]:
                if rs[i] > rs[j]: conc += 1
                elif rs[i] < rs[j]: disc += 1
                else: tied += 1
    ci = (conc+0.5*tied)/(conc+disc+tied) if (conc+disc+tied)>0 else 0.5
    st.markdown(f"**C-index:** {ci:.4f} ({'Baik' if ci>=0.7 else 'Moderat' if ci>=0.6 else 'Lemah'})")

    st.subheader("Forest Plot: Hazard Ratio")
    fig_hr = go.Figure()
    for _, row in cox_df.iterrows():
        color = '#d62728' if row['HR (eᵝ)'] > 1 else '#2ca02c'
        fig_hr.add_trace(go.Scatter(x=[row['HR CI Lo'], row['HR (eᵝ)'], row['HR CI Hi']],
                                    y=[row['Kovariat']]*3, mode='lines+markers',
                                    marker=dict(size=[8,14,8], color=color),
                                    line=dict(color=color, width=3), showlegend=False))
    fig_hr.add_vline(x=1, line_dash="dash", line_color="gray")
    fig_hr.update_layout(title="Forest Plot: HR ± 95% CI", xaxis_title="Hazard Ratio",
                         height=max(300, 60*len(covariates)))
    st.plotly_chart(fig_hr, use_container_width=True)

    if group_col != '(Tidak ada)':
        st.subheader("Diagnostik Asumsi PH")
        fig_lnln = go.Figure()
        for i, g in enumerate(sorted(groups)):
            km_g = km_by_group[g]; valid = km_g['S(t)']>0; km_v = km_g[valid]
            if len(km_v) > 0:
                fig_lnln.add_trace(go.Scatter(x=np.log(km_v['Time'].values+1e-10),
                                              y=np.log(-np.log(km_v['S(t)'].values+1e-10)),
                                              mode='lines+markers', name=str(g),
                                              line=dict(color=colors[i%len(colors)])))
        fig_lnln.update_layout(title="Log-Log Plot: Cek Asumsi PH",
                               xaxis_title="log(t)", yaxis_title="log(-log(S(t)))", height=450)
        st.plotly_chart(fig_lnln, use_container_width=True)

# ============================================================
# PARAMETRIC MODELS — 15 DISTRIBUSI
# ============================================================
sec_par = str(int(sec_cox)+1) if len(covariates)>=1 else str(int(sec_hz)+1)
st.header(f"{sec_par}. Model Parametrik (15 Distribusi)")
st.markdown("""
Fit **15 distribusi parametrik** ke data survival — termasuk varian **2-parameter** dan **3-parameter (shifted)**.
Setiap model menampilkan estimasi parameter, Log-Likelihood, AIC, BIC, dan ranking.
""")

dur_pos = durations.copy(); dur_pos[dur_pos <= 0] = 0.001

available_models = list(ALL_MODELS.keys())
selected_models = st.multiselect("Pilih Model (atau biarkan semua):", available_models, default=available_models)
if len(selected_models) == 0:
    selected_models = available_models

t_grid = np.linspace(0.01, durations.max()*1.1, 300)

# Fit all selected models
results = []
fitted = {}
with st.spinner("Fitting parametric models..."):
    for name in selected_models:
        info = ALL_MODELS[name]
        try:
            params, ll_val, n_p = info['fit'](dur_pos, events)
            aic = 2*n_p - 2*ll_val
            bic = n_p*np.log(N) - 2*ll_val
            fitted[name] = {'params': params, 'll': ll_val, 'np': n_p, 'aic': aic, 'bic': bic}
            param_str = ', '.join([f'{k}={v:.5f}' for k, v in params.items()])
            results.append({
                'Model': name, 'Formula': info['formula'], '#Param': n_p,
                'Parameters': param_str,
                'Log-Lik': round(ll_val, 2), 'AIC': round(aic, 2), 'BIC': round(bic, 2),
                'Hazard Type': info['hz_type']
            })
        except Exception as ex:
            results.append({
                'Model': name, 'Formula': info['formula'], '#Param': '-',
                'Parameters': f'Error: {str(ex)[:40]}',
                'Log-Lik': '-', 'AIC': '-', 'BIC': '-', 'Hazard Type': info['hz_type']
            })

res_df = pd.DataFrame(results)
# Rank by AIC
valid_aic = res_df[res_df['AIC'] != '-'].copy()
if len(valid_aic) > 0:
    valid_aic['Rank'] = valid_aic['AIC'].astype(float).rank().astype(int)
    res_df = res_df.merge(valid_aic[['Model','Rank']], on='Model', how='left')
    res_df['Rank'] = res_df['Rank'].fillna('-')
    res_df = res_df.sort_values('Rank', key=lambda x: pd.to_numeric(x, errors='coerce')).reset_index(drop=True)
else:
    res_df['Rank'] = '-'

st.dataframe(res_df, use_container_width=True, hide_index=True)

if len(fitted) > 0:
    best_name = min(fitted, key=lambda k: fitted[k]['aic'])
    best = fitted[best_name]
    st.success(f"🏆 **Model terbaik (AIC):** {best_name} — AIC = {best['aic']:.2f}, BIC = {best['bic']:.2f}")

    # Detail parameters of best model
    st.subheader(f"Detail Parameter: {best_name}")
    pdetail = []
    for k, v in best['params'].items():
        pdetail.append({'Parameter': k, 'Estimasi': f'{v:.6f}'})
    st.dataframe(pd.DataFrame(pdetail), use_container_width=True, hide_index=True)

# ---- PARAMETRIC PLOTS ----
tab_p1, tab_p2, tab_p3, tab_p4 = st.tabs(["Survival Fit", "Hazard Fit", "QQ-Plot", "Parameter Detail (Semua)"])
km_overall = kaplan_meier(durations, events)
plot_colors_all = (px.colors.qualitative.Plotly + px.colors.qualitative.Set2
                   + px.colors.qualitative.Pastel + px.colors.qualitative.Dark2)
line_styles = ['dash','dot','dashdot','longdash','longdashdot','solid']

with tab_p1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=km_overall['Time'], y=km_overall['S(t)'], mode='lines',
                             name='Kaplan-Meier', line=dict(color='black', width=2.5, shape='hv')))
    for idx, name in enumerate(fitted):
        info = ALL_MODELS[name]
        try:
            S = info['surv'](t_grid, fitted[name]['params'])
            fig.add_trace(go.Scatter(x=t_grid, y=S, mode='lines', name=name,
                                     line=dict(dash=line_styles[idx%len(line_styles)],
                                               color=plot_colors_all[idx%len(plot_colors_all)], width=1.5)))
        except: pass
    fig.update_layout(title=f"Parametric Survival vs Kaplan-Meier ({len(fitted)} Models)", height=580,
                      xaxis_title="Time", yaxis_title="S(t)", yaxis=dict(range=[0,1.05]))
    st.plotly_chart(fig, use_container_width=True)

with tab_p2:
    fig = go.Figure()
    for idx, name in enumerate(fitted):
        info = ALL_MODELS[name]
        try:
            h = info['haz'](t_grid, fitted[name]['params'])
            h = np.copy(h); h[~np.isfinite(h)] = 0
            p99 = np.percentile(h[h>0], 99) if np.any(h>0) else 1
            h = np.clip(h, 0, p99*2)
            fig.add_trace(go.Scatter(x=t_grid, y=h, mode='lines', name=name,
                                     line=dict(dash=line_styles[idx%len(line_styles)],
                                               color=plot_colors_all[idx%len(plot_colors_all)], width=1.5)))
        except: pass
    fig.update_layout(title=f"Hazard Functions ({len(fitted)} Models)", height=520,
                      xaxis_title="Time", yaxis_title="h(t)")
    st.plotly_chart(fig, use_container_width=True)

with tab_p3:
    st.markdown("**Weibull QQ-Plot:** log(t) vs log(-log(S_KM(t)))")
    km_qq = km_overall[km_overall['S(t)']>0].copy()
    if len(km_qq) > 2:
        log_t_qq = np.log(km_qq['Time'].values)
        lnln_s_qq = np.log(-np.log(km_qq['S(t)'].values + 1e-15))
        valid_mask = np.isfinite(log_t_qq) & np.isfinite(lnln_s_qq)
        fig = px.scatter(x=log_t_qq, y=lnln_s_qq, labels={'x':'log(t)','y':'log(-log(S(t)))'},
                         title="Weibull QQ-Plot")
        if valid_mask.sum() > 1:
            slope, intercept = np.polyfit(log_t_qq[valid_mask], lnln_s_qq[valid_mask], 1)
            fig.add_trace(go.Scatter(x=log_t_qq, y=slope*log_t_qq+intercept, mode='lines',
                                     name=f'Fit (shape≈{slope:.2f})', line=dict(color='red')))
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

with tab_p4:
    st.markdown("### Estimasi Parameter Semua Model")
    all_param_rows = []
    for name in fitted:
        for k, v in fitted[name]['params'].items():
            all_param_rows.append({
                'Model': name, 'Parameter': k, 'Estimasi': round(v, 6),
                'Log-Lik': round(fitted[name]['ll'], 2),
                'AIC': round(fitted[name]['aic'], 2)
            })
    if all_param_rows:
        st.dataframe(pd.DataFrame(all_param_rows), use_container_width=True, hide_index=True)

# ============================================================
# ADDITIONAL VISUALIZATIONS
# ============================================================
sec_vis = str(int(sec_par)+1)
st.header(f"{sec_vis}. Visualisasi Tambahan")
tab_v1, tab_v2, tab_v3 = st.tabs(["Distribusi Waktu", "Risk Table", "Censoring Pattern"])

with tab_v1:
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Events","Censored"))
    fig.add_trace(go.Histogram(x=durations[events==1], nbinsx=20, marker_color='salmon', name='Event'), row=1, col=1)
    fig.add_trace(go.Histogram(x=durations[events==0], nbinsx=20, marker_color='steelblue', name='Censored'), row=1, col=2)
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
with tab_v2:
    tps = np.linspace(0, durations.max(), 8).round(1)
    rr = [{'Time': tp, 'N at Risk': int(np.sum(durations>=tp)),
           '% Remaining': f'{np.sum(durations>=tp)/N:.1%}'} for tp in tps]
    st.dataframe(pd.DataFrame(rr), use_container_width=True, hide_index=True)
with tab_v3:
    fig = px.histogram(panel, x=time_col, color=event_col, barmode='stack',
                       title="Censoring Pattern", nbins=25,
                       color_discrete_map={0:'steelblue', 1:'salmon'})
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# EXPORT
# ============================================================
sec_exp = str(int(sec_vis)+1)
st.header(f"{sec_exp}. Ekspor Hasil")
col_e1, col_e2, col_e3 = st.columns(3)
with col_e1:
    lines = [
        "="*70, "ANALISIS DATA UJI HIDUP (SURVIVAL ANALYSIS)", "="*70,
        f"N            : {N}", f"Events       : {n_events} ({n_events/N:.1%})",
        f"Censored     : {n_censor} ({n_censor/N:.1%})",
        f"Tipe Sensor  : {cens_type_used}",
        f"Median Time  : {np.median(durations):.2f}", f"Alpha        : {alpha}", "",
        "="*70, f"PARAMETRIC FITS ({len(fitted)} MODELS)", "="*70]
    for name in fitted:
        pstr = ', '.join([f'{k}={v:.5f}' for k,v in fitted[name]['params'].items()])
        lines.append(f"  {name:25s} | {pstr} | AIC={fitted[name]['aic']:.2f}")
    if len(fitted) > 0:
        lines.append(f"\nBest Model (AIC): {best_name} (AIC={best['aic']:.2f})")
    if len(covariates) >= 1:
        lines += ["", "="*70, "COX PH REGRESSION", "="*70, f"C-index: {ci:.4f}"]
        for _, row in cox_df.iterrows():
            lines.append(f"  {row['Kovariat']:15s} | β={row['β']:>10.6f} | HR={row['HR (eᵝ)']:>8.4f} | p={row['p-value']:>10.6f}")
    st.download_button("📥 Summary (TXT)", data="\n".join(lines),
                       file_name="survival_analysis_summary.txt", mime="text/plain")
with col_e2:
    st.download_button("📥 KM Life Table (CSV)", data=kaplan_meier(durations, events).to_csv(index=False),
                       file_name="kaplan_meier_table.csv", mime="text/csv")
with col_e3:
    st.download_button("📥 Data (CSV)", data=df.to_csv(index=False),
                       file_name="survival_data.csv", mime="text/csv")

# FOOTER
st.markdown("---")
st.markdown("""
**Referensi Metodologis:**
- **Kaplan & Meier (1958):** Estimasi nonparametrik S(t) dengan data censored.
- **Nelson-Aalen:** Estimasi H(t), lebih baik untuk sampel kecil.
- **Log-Rank / Gehan-Wilcoxon / Tarone-Ware / Peto-Peto:** Uji perbandingan kurva survival.
- **Cox PH (1972):** Model semi-parametrik h(t|X) = h₀(t)exp(βX).
- **Tipe Penyensoran:** Tipe I (waktu tetap), Tipe II (jumlah event tetap), Random/Tipe III.
- **Exponential 1P/2P:** Hazard konstan; 2P menambahkan parameter lokasi γ.
- **Weibull 2P/3P:** β>1 (↑ hazard), β<1 (↓); 3P menambahkan threshold γ.
- **Log-Normal 2P/3P:** Hazard non-monoton (naik lalu turun).
- **Log-Logistic 2P/3P:** Hazard non-monoton jika β>1; ekor lebih berat dari log-normal.
- **Gamma 2P/3P:** Fleksibel; α>1 (increasing), α<1 (decreasing).
- **Gompertz 2P:** Hazard naik eksponensial — cocok untuk mortalitas/penuaan.
- **Generalized Gamma 3P (Stacy):** Mencakup Weibull, Log-Normal, Gamma sebagai kasus khusus.
- **Inverse Gaussian 2P:** Model degradasi / first-passage time.
- **Rayleigh 1P:** Kasus khusus Weibull (β=2), hazard linear naik.
- **Birnbaum-Saunders 2P:** Model fatigue/kerusakan kumulatif.

Dibangun dengan **Streamlit** + **SciPy** + **NumPy** + **Plotly** | Python
""")
