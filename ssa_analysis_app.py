"""
══════════════════════════════════════════════════════════════
  APLIKASI SSA (Singular Spectrum Analysis) — Streamlit
  Untuk Skripsi & Artikel Ilmiah
══════════════════════════════════════════════════════════════
"""

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import norm, probplot
from scipy.signal import periodogram
from statsmodels.graphics.tsaplots import plot_acf
import warnings, os, tempfile

warnings.filterwarnings('ignore')

from ssa_core import SSA, find_optimal_L

# ── Konfigurasi halaman ──────────────────────────────────────
st.set_page_config(page_title="SSA Time Series 1D", layout="wide", page_icon="📈")

plt.rcParams.update({
    'figure.dpi': 150, 'font.size': 10, 'font.family': 'serif',
    'axes.grid': True, 'grid.alpha': 0.3, 'lines.linewidth': 1.3,
    'figure.autolayout': True,
})

# ══════════════════════════════════════════════════════════════
# SIDEBAR — Semua pengaturan
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("⚙️ Pengaturan SSA")

    st.header("1️⃣ Data")
    data_source = st.radio("Sumber data:", ["Upload CSV/Excel", "Data Demo Sintetik"])
    uploaded_file = None
    if data_source == "Upload CSV/Excel":
        uploaded_file = st.file_uploader("Upload file", type=["csv", "xlsx", "xls"])

    st.header("2️⃣ Window Length (L)")
    L_mode = st.radio("Mode:", ["Auto (N/2)", "Manual", "Optimasi L (lambat)"])
    L_manual = 48
    if L_mode == "Manual":
        L_manual = st.number_input("Nilai L:", min_value=2, value=48, step=1)

    st.header("3️⃣ Grouping")
    grouping_mode = st.radio("Mode grouping:", ["Auto Grouping", "Manual Grouping"])
    wcorr_threshold = st.slider("W-Corr threshold (auto)", 0.1, 0.9, 0.5, 0.05)
    num_comp_wcorr = st.slider("Jml komponen W-Corr", 4, 30, 12)

    st.header("4️⃣ Forecasting")
    steps_ahead = st.number_input("Langkah forecast:", min_value=1, value=24, step=1)
    do_r_forecast = st.checkbox("Recurrent (R-forecast)", value=True)
    do_v_forecast = st.checkbox("Vector (V-forecast)", value=True)

    st.header("5️⃣ Evaluasi Train/Test")
    train_pct = st.slider("Training %", 50, 95, 80, 5)

    st.header("6️⃣ Monte Carlo SSA")
    do_mc = st.checkbox("Jalankan Monte Carlo SSA", value=False)
    mc_surr = st.number_input("Surrogate:", 100, 2000, 500, 100)
    mc_conf = st.slider("Confidence", 0.90, 0.99, 0.95, 0.01)

# ══════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════
st.title("📈 Singular Spectrum Analysis (SSA)")
st.caption("Analisis lengkap: Dekomposisi → W-Correlation → Grouping → Rekonstruksi → "
           "Forecasting (R & V) → Evaluasi Train/Test/Overall → Residual → Monte Carlo")

ts = None
series_name = "Time Series"

if data_source == "Data Demo Sintetik":
    np.random.seed(42)
    N_demo = 200
    t_demo = np.arange(N_demo)
    ts = (0.02 * t_demo + 5
          + 3 * np.sin(2 * np.pi * t_demo / 12)
          + 1.5 * np.sin(2 * np.pi * t_demo / 6)
          + np.random.normal(0, 0.5, N_demo))
    series_name = "Demo (Trend + Seasonal12 + Seasonal6 + Noise)"
    st.info("🔹 Menggunakan data demo sintetik (N=200): Trend + Seasonal(T=12) + Seasonal(T=6) + Noise")
else:
    if uploaded_file is None:
        st.warning("⬆️ Silakan upload file CSV/Excel di sidebar, atau pilih Data Demo.")
        st.stop()
    if uploaded_file.name.endswith('.csv'):
        df_raw = pd.read_csv(uploaded_file)
    else:
        df_raw = pd.read_excel(uploaded_file)
    st.subheader("Preview Data")
    st.dataframe(df_raw.head(10), use_container_width=True)
    num_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        st.error("Tidak ada kolom numerik di file.")
        st.stop()
    target_col = st.selectbox("Pilih kolom target:", num_cols)
    ts = df_raw[target_col].dropna().values.astype(float)
    series_name = target_col

N = len(ts)
train_n = int(N * train_pct / 100)

# ══════════════════════════════════════════════════════════════
# TENTUKAN L
# ══════════════════════════════════════════════════════════════
if L_mode == "Optimasi L (lambat)":
    with st.spinner("🔍 Mencari L optimal (ini agak lama)..."):
        opt = find_optimal_L(ts)
    if opt:
        L_val = opt['best_L']
        st.success(f"✅ L optimal = **{L_val}** (RMSE = {opt['best_RMSE']:.6f})")
    else:
        L_val = N // 2
elif L_mode == "Manual":
    L_val = min(int(L_manual), N // 2)
else:
    L_val = N // 2

# ══════════════════════════════════════════════════════════════
# INISIALISASI SSA
# ══════════════════════════════════════════════════════════════
with st.spinner("⏳ Menjalankan SSA Decomposition..."):
    ssa = SSA(ts, window_length=L_val, name=series_name)

col_info1, col_info2, col_info3, col_info4 = st.columns(4)
col_info1.metric("N (panjang data)", N)
col_info2.metric("L (window length)", ssa.L)
col_info3.metric("K (kolom trajectory)", ssa.K)
col_info4.metric("d (rank efektif)", ssa.d)

# ══════════════════════════════════════════════════════════════
# 1. DATA ORIGINAL
# ══════════════════════════════════════════════════════════════
st.header("1️⃣ Data Original")
fig, ax = plt.subplots(figsize=(12, 3.5))
ax.plot(ts, 'b-', linewidth=1)
ax.axvline(train_n, color='orange', ls='--', lw=1.5, label=f'Train/Test split (n={train_n})')
ax.set_title(f'Data Original: {series_name}', fontweight='bold')
ax.set_xlabel('Waktu'); ax.set_ylabel('Nilai'); ax.legend()
st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════════
# 2. SCREE PLOT
# ══════════════════════════════════════════════════════════════
st.header("2️⃣ Scree Plot & Kontribusi Varians")
n_scree = min(20, ssa.d)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
x = np.arange(1, n_scree + 1)
ax1.bar(x, ssa.singular_values[:n_scree], color='steelblue', alpha=0.7, edgecolor='navy')
ax1.plot(x, ssa.singular_values[:n_scree], 'ro-', ms=4)
ax1.set_title('Singular Values', fontweight='bold'); ax1.set_xlabel('Komponen'); ax1.set_ylabel('σ')
ax1.set_xticks(x)
ax2.bar(x, ssa.contribution[:n_scree], color='coral', alpha=0.7, edgecolor='darkred', label='Individual')
ax2.plot(x, ssa.cumulative_contribution[:n_scree], 'ko-', ms=4, label='Kumulatif')
ax2.axhline(95, color='r', ls='--', alpha=.5, label='95%')
ax2.axhline(99, color='g', ls='--', alpha=.5, label='99%')
ax2.set_title('Kontribusi Varians (%)', fontweight='bold'); ax2.set_xlabel('Komponen')
ax2.set_ylabel('%'); ax2.set_xticks(x); ax2.legend(fontsize=8)
st.pyplot(fig); plt.close()

# Tabel eigenvalue
with st.expander("📋 Tabel Eigenvalue & Kontribusi"):
    ev_df = pd.DataFrame({
        'No': range(1, n_scree + 1),
        'Singular Value': ssa.singular_values[:n_scree],
        'Eigenvalue': ssa.eigenvalues[:n_scree],
        'Kontribusi (%)': ssa.contribution[:n_scree],
        'Kumulatif (%)': ssa.cumulative_contribution[:n_scree],
    })
    st.dataframe(ev_df.style.format({
        'Singular Value': '{:.4f}', 'Eigenvalue': '{:.4f}',
        'Kontribusi (%)': '{:.4f}', 'Kumulatif (%)': '{:.4f}'
    }), use_container_width=True)

# ══════════════════════════════════════════════════════════════
# 3. EIGENVECTORS
# ══════════════════════════════════════════════════════════════
st.header("3️⃣ Eigenvectors")
n_ev = min(8, ssa.d)
fig, axes = plt.subplots(2, 4, figsize=(14, 5))
for i, ax in enumerate(axes.flatten()):
    if i < n_ev:
        ax.plot(ssa.U[:, i], 'b-', lw=0.8)
        ax.set_title(f'EV{i+1} (σ={ssa.singular_values[i]:.1f}, {ssa.contribution[i]:.1f}%)', fontsize=8)
        ax.axhline(0, color='k', lw=0.3)
    else:
        ax.set_visible(False)
plt.suptitle('Left Singular Vectors (Eigenvectors)', fontweight='bold')
plt.tight_layout(); st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════════
# 4. PAIRED EIGENVECTORS
# ══════════════════════════════════════════════════════════════
st.header("4️⃣ Paired Eigenvectors (🔵 Lingkaran = Periodik)")
n_pairs = min(6, ssa.d - 1)
fig, axes = plt.subplots(1, n_pairs, figsize=(3.5 * n_pairs, 3.5))
if n_pairs == 1: axes = [axes]
for idx in range(n_pairs):
    i, j = idx, idx + 1
    axes[idx].scatter(ssa.U[:, i], ssa.U[:, j], s=5, alpha=.6, c='steelblue')
    axes[idx].set_xlabel(f'EV{i+1}'); axes[idx].set_ylabel(f'EV{j+1}')
    axes[idx].set_aspect('equal'); axes[idx].axhline(0, color='k', lw=.3); axes[idx].axvline(0, color='k', lw=.3)
plt.suptitle('Paired Eigenvectors — Pola lingkaran menandakan komponen periodik', fontweight='bold', fontsize=10)
plt.tight_layout(); st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════════
# 5. PERIODOGRAM PER KOMPONEN
# ══════════════════════════════════════════════════════════════
st.header("5️⃣ Periodogram per Komponen")
n_pg = min(8, ssa.d)
fig, axes = plt.subplots(n_pg, 1, figsize=(12, 2.2 * n_pg), sharex=True)
if n_pg == 1: axes = [axes]
for i in range(n_pg):
    rc = ssa.reconstruct_component(i)
    freqs, psd = periodogram(rc, fs=1.0)
    axes[i].plot(freqs, psd, 'b-', lw=.8); axes[i].fill_between(freqs, psd, alpha=.3)
    axes[i].set_ylabel(f'F{i+1}', fontweight='bold')
    if len(freqs) > 1:
        pk = freqs[np.argmax(psd[1:]) + 1]
        T = 1 / pk if pk > 0 else np.inf
        axes[i].text(0.97, 0.75, f'f={pk:.4f}  T≈{T:.1f}', transform=axes[i].transAxes,
                     fontsize=8, ha='right', bbox=dict(boxstyle='round', fc='lightyellow', alpha=.8))
axes[-1].set_xlabel('Frequency')
plt.suptitle('Periodogram — Identifikasi Frekuensi Dominan', fontweight='bold', y=1.01)
plt.tight_layout(); st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════════
# 6. W-CORRELATION & GROUPING
# ══════════════════════════════════════════════════════════════
st.header("6️⃣ W-Correlation Matrix & Grouping")

nc = min(num_comp_wcorr, ssa.d)
with st.spinner("Menghitung W-Correlation..."):
    wcorr = ssa.w_correlation(nc)

fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(np.abs(wcorr), cmap='RdBu_r', vmin=0, vmax=1, interpolation='nearest')
ax.set_xticks(range(nc)); ax.set_yticks(range(nc))
ax.set_xticklabels([f'F{i+1}' for i in range(nc)], fontsize=7)
ax.set_yticklabels([f'F{i+1}' for i in range(nc)], fontsize=7)
ax.set_title('W-Correlation Matrix (|ρʷ|)', fontweight='bold')
plt.colorbar(im, ax=ax, shrink=.8, label='|W-Correlation|')
if nc <= 15:
    for i in range(nc):
        for j in range(nc):
            v = abs(wcorr[i, j])
            ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=6,
                    color='white' if v > .5 else 'black')
st.pyplot(fig); plt.close()

st.markdown("""
> **Cara baca:** Nilai mendekati **0** → komponen terpisah baik (*well-separated*). 
> Nilai mendekati **1** → harus di-grup bersama.
""")

# ── Grouping ─────────────────────────────────────────────────
if grouping_mode == "Auto Grouping":
    groups = ssa.auto_group(num_components=nc, threshold=wcorr_threshold)
else:
    st.subheader("Manual Grouping")
    st.markdown("Tentukan indeks komponen untuk tiap grup (pisah koma). Indeks mulai dari 0.")
    group_names_input = st.text_input("Nama grup (pisah koma):", "Trend, Seasonal_1, Seasonal_2, Noise")
    group_indices_input = st.text_input(
        "Indeks per grup (pisah titik koma ';'):",
        "0,1 ; 2,3 ; 4,5 ; 6,7,8,9,10,11"
    )
    gnames = [g.strip() for g in group_names_input.split(',')]
    gidxs = group_indices_input.split(';')
    groups = {}
    for gn, gi in zip(gnames, gidxs):
        try:
            groups[gn] = [int(x.strip()) for x in gi.split(',') if x.strip()]
        except:
            st.error(f"Format indeks grup '{gn}' salah."); st.stop()

st.subheader("Hasil Grouping")
group_df = pd.DataFrame([
    {'Grup': k, 'Komponen': str(v),
     'Kontribusi (%)': sum(ssa.contribution[i] for i in v if i < len(ssa.contribution))}
    for k, v in groups.items()
])
st.dataframe(group_df, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# 7. REKONSTRUKSI
# ══════════════════════════════════════════════════════════════
st.header("7️⃣ Rekonstruksi Komponen")
with st.spinner("Merekonstruksi..."):
    ssa.reconstruct(groups)

user_groups = {k: v for k, v in ssa.reconstructed.items() if not k.startswith('_')}
n_grp = len(user_groups)
fig, axes = plt.subplots(n_grp + 1, 1, figsize=(13, 2.8 * (n_grp + 1)), sharex=True)
axes[0].plot(ts, 'b-', alpha=.5, lw=.8, label='Original')
axes[0].plot(ssa.reconstructed['_Total'], 'r-', lw=1, label='Rekonstruksi Total')
axes[0].set_title('Original vs Rekonstruksi Total', fontweight='bold'); axes[0].legend()
colors = plt.cm.Set1(np.linspace(0, 1, n_grp))
for idx, (gname, gvals) in enumerate(user_groups.items()):
    axes[idx + 1].plot(gvals, color=colors[idx], lw=.8)
    axes[idx + 1].set_ylabel(gname, fontweight='bold')
    axes[idx + 1].axhline(0, color='k', lw=.3)
    share = sum(ssa.contribution[i] for i in ssa.groups[gname] if i < len(ssa.contribution))
    axes[idx + 1].text(0.98, .8, f'{share:.1f}%', transform=axes[idx + 1].transAxes,
                       fontsize=10, ha='right', fontweight='bold',
                       bbox=dict(boxstyle='round', fc='lightyellow', alpha=.8))
axes[-1].set_xlabel('Waktu')
plt.suptitle('SSA Reconstruction', fontweight='bold', fontsize=13, y=1.01)
plt.tight_layout(); st.pyplot(fig); plt.close()

rmse_recon = np.sqrt(np.mean(ssa.reconstructed['_Residual'] ** 2))
st.metric("RMSE Rekonstruksi Total", f"{rmse_recon:.6f}")

# ══════════════════════════════════════════════════════════════
# 8. MONTE CARLO SSA (opsional)
# ══════════════════════════════════════════════════════════════
if do_mc:
    st.header("8️⃣ Monte Carlo SSA Significance Test")
    with st.spinner(f"Monte Carlo ({mc_surr} surrogates)... bisa 1-3 menit"):
        mc = ssa.monte_carlo_test(num_surrogates=int(mc_surr), confidence=float(mc_conf))
    n_mc = len(mc['eigenvalues'])
    x_mc = np.arange(1, n_mc + 1)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.semilogy(x_mc, mc['eigenvalues'], 'ro-', ms=7, label='Data', zorder=5)
    ax.fill_between(x_mc, mc['surrogate_lower'], mc['surrogate_upper'],
                    alpha=.3, color='blue', label=f'{mc["confidence"]*100:.0f}% CI Red Noise')
    ax.semilogy(x_mc, mc['surrogate_median'], 'b--', lw=1, label='Median Red Noise')
    sig = mc['significant']
    ax.semilogy(x_mc[sig], mc['eigenvalues'][sig], 'r*', ms=14, label='Signifikan', zorder=6)
    ax.semilogy(x_mc[~sig], mc['eigenvalues'][~sig], 'kx', ms=9, label='Tidak Signifikan', zorder=6)
    ax.set_title('Monte Carlo SSA Significance Test', fontweight='bold')
    ax.set_xlabel('Komponen'); ax.set_ylabel('Eigenvalue (log)'); ax.set_xticks(x_mc); ax.legend()
    st.pyplot(fig); plt.close()
    st.success(f"Komponen signifikan: **{int(np.sum(sig))}** dari {n_mc} → indeks: {list(np.where(sig)[0])}")

# ══════════════════════════════════════════════════════════════
# 9. FORECASTING & EVALUASI
# ══════════════════════════════════════════════════════════════
st.header("9️⃣ Forecasting & Evaluasi")

signal_groups = {k: v for k, v in groups.items() if 'noise' not in k.lower()}
if not signal_groups:
    signal_groups = groups

fc_r, fc_v = None, None
if do_r_forecast:
    with st.spinner("R-forecasting..."):
        fc_r = ssa.forecast_recurrent(signal_groups, steps=int(steps_ahead))
if do_v_forecast:
    with st.spinner("V-forecasting..."):
        fc_v = ssa.forecast_vector(signal_groups, steps=int(steps_ahead))

# Plot forecast
fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(range(N), ts, 'b-', lw=1, label='Actual', alpha=.7)
if fc_r is not None:
    ax.plot(range(len(fc_r)), fc_r, 'r--', lw=1.2, label='R-Forecast')
if fc_v is not None:
    ax.plot(range(len(fc_v)), fc_v, 'g--', lw=1.2, label='V-Forecast')
ax.axvline(train_n, color='orange', ls='--', lw=1.5, label=f'Split (train={train_n})')
ax.axvline(N - 1, color='gray', ls=':', lw=1, label=f'End data (N={N})')
ax.set_title('SSA Forecasting', fontweight='bold')
ax.set_xlabel('Waktu'); ax.set_ylabel('Nilai'); ax.legend()
st.pyplot(fig); plt.close()

# Evaluasi
def show_eval(label, eval_dict):
    st.markdown(f"**{label}**")
    rows = []
    for split in ['train', 'test', 'overall']:
        d = eval_dict[split]
        rows.append({
            'Split': split.upper(),
            'N': d['N'], 'RMSE': d['RMSE'], 'MAE': d['MAE'],
            'MAPE (%)': d['MAPE_pct'], 'sMAPE (%)': d['sMAPE_pct'],
            'R²': d['R2'], 'NRMSE': d['NRMSE']
        })
    st.dataframe(pd.DataFrame(rows).style.format({
        'RMSE': '{:.4f}', 'MAE': '{:.4f}', 'MAPE (%)': '{:.4f}',
        'sMAPE (%)': '{:.4f}', 'R²': '{:.6f}', 'NRMSE': '{:.6f}'
    }), use_container_width=True)

if fc_r is not None:
    eval_r = ssa.evaluate_split(train_pct / 100, fc_r)
    show_eval("📊 Evaluasi Recurrent (R-forecast)", eval_r)
if fc_v is not None:
    eval_v = ssa.evaluate_split(train_pct / 100, fc_v)
    show_eval("📊 Evaluasi Vector (V-forecast)", eval_v)

# Detail error plot
if fc_r is not None or fc_v is not None:
    fc_show = fc_r if fc_r is not None else fc_v
    lbl_show = "R-forecast" if fc_r is not None else "V-forecast"
    errors = ts - fc_show[:N]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].bar(range(train_n), errors[:train_n], color='green', alpha=.5, width=1, label='Train Error')
    axes[0].bar(range(train_n, N), errors[train_n:], color='orange', alpha=.5, width=1, label='Test Error')
    axes[0].axhline(0, color='k', lw=.5); axes[0].set_title(f'Forecast Errors ({lbl_show})', fontweight='bold')
    axes[0].set_xlabel('Waktu'); axes[0].set_ylabel('Error'); axes[0].legend()
    axes[1].scatter(fc_show[:N], ts, s=8, alpha=.5, c='steelblue')
    mn, mx = min(ts.min(), fc_show[:N].min()), max(ts.max(), fc_show[:N].max())
    axes[1].plot([mn, mx], [mn, mx], 'r--', lw=1)
    axes[1].set_title('Actual vs Predicted', fontweight='bold')
    axes[1].set_xlabel('Predicted'); axes[1].set_ylabel('Actual'); axes[1].set_aspect('equal')
    plt.tight_layout(); st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════════
# 10. RESIDUAL ANALYSIS
# ══════════════════════════════════════════════════════════════
st.header("🔟 Analisis Residual")
res_info = ssa.residual_analysis()
residuals = ssa.residuals

if res_info:
    col1, col2, col3 = st.columns(3)
    col1.metric("Mean", f"{res_info['mean']:.6f}")
    col2.metric("Std Dev", f"{res_info['std']:.6f}")
    col3.metric("Skewness", f"{res_info['skewness']:.4f}")

    st.markdown("**Uji Normalitas & White Noise**")
    test_rows = []
    if 'shapiro_stat' in res_info:
        p = res_info['shapiro_p']
        test_rows.append({'Test': 'Shapiro-Wilk', 'Statistic': res_info['shapiro_stat'],
                          'p-value': p, 'Kesimpulan': 'Normal ✅' if p > 0.05 else 'Tidak Normal ❌'})
    p_jb = res_info['jarque_bera_p']
    test_rows.append({'Test': 'Jarque-Bera', 'Statistic': res_info['jarque_bera_stat'],
                      'p-value': p_jb, 'Kesimpulan': 'Normal ✅' if p_jb > 0.05 else 'Tidak Normal ❌'})
    if 'ljung_box_p' in res_info:
        p_lb = res_info['ljung_box_p']
        test_rows.append({'Test': 'Ljung-Box', 'Statistic': res_info['ljung_box_stat'],
                          'p-value': p_lb, 'Kesimpulan': 'White Noise ✅' if p_lb > 0.05 else 'Bukan WN ❌'})
    st.dataframe(pd.DataFrame(test_rows), use_container_width=True)

    # Plot diagnostik residual
    fig = plt.figure(figsize=(13, 9))
    gs = gridspec.GridSpec(2, 2, hspace=.35, wspace=.3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(residuals, 'b-', lw=.5); ax1.axhline(0, color='r', lw=.8)
    ax1.set_title('Residual Time Plot', fontweight='bold'); ax1.set_xlabel('Waktu')
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(residuals, bins='auto', density=True, alpha=.7, color='steelblue', edgecolor='navy')
    xr = np.linspace(residuals.min(), residuals.max(), 100)
    ax2.plot(xr, norm.pdf(xr, np.mean(residuals), np.std(residuals)), 'r-', lw=2, label='Normal fit')
    ax2.set_title('Histogram Residual', fontweight='bold'); ax2.legend()
    ax3 = fig.add_subplot(gs[1, 0])
    plot_acf(residuals, ax=ax3, lags=min(40, len(residuals) // 3), alpha=.05)
    ax3.set_title('ACF Residual', fontweight='bold')
    ax4 = fig.add_subplot(gs[1, 1])
    probplot(residuals, dist='norm', plot=ax4)
    ax4.set_title('Q-Q Plot (Normal)', fontweight='bold')
    plt.suptitle('Diagnostik Residual', fontsize=13, fontweight='bold', y=1.02)
    st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════════
# 11. DOWNLOAD EXCEL
# ══════════════════════════════════════════════════════════════
st.header("1️⃣1️⃣ Download Hasil (Excel)")
tmp_path = os.path.join(tempfile.gettempdir(), 'SSA_Results.xlsx')
with st.spinner("Menyiapkan file Excel..."):
    ssa.save_results(tmp_path)
with open(tmp_path, 'rb') as f:
    excel_bytes = f.read()
os.remove(tmp_path)
st.download_button("📥 Download SSA_Results.xlsx", excel_bytes,
                   file_name="SSA_Results.xlsx",
                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.success("✅ Analisis SSA selesai! Ubah pengaturan di sidebar untuk eksplorasi lebih lanjut.")
st.caption("SSA Analysis App v3.0 — Cocok untuk Skripsi & Artikel Ilmiah")
