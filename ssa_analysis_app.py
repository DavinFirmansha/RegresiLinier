"""
══════════════════════════════════════════════════════════════
  APLIKASI SSA — Streamlit v5.6
  - Grouping: method selectbox OUTSIDE form, settings INSIDE form
  - Optimasi L: button + full table
══════════════════════════════════════════════════════════════
"""
import numpy as np, pandas as pd, streamlit as st
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.gridspec as gridspec
from scipy.stats import norm, probplot
from scipy.signal import periodogram
from scipy.cluster.hierarchy import dendrogram
from statsmodels.graphics.tsaplots import plot_acf
import warnings, os, tempfile
warnings.filterwarnings('ignore')
from ssa_core import SSA, find_optimal_L

st.set_page_config(page_title="SSA Time Series", layout="wide", page_icon="📈")
plt.rcParams.update({'figure.dpi': 150, 'font.size': 10, 'font.family': 'serif',
    'axes.grid': True, 'grid.alpha': 0.3, 'lines.linewidth': 1.3, 'figure.autolayout': True})

def to1(idx): return [i+1 for i in idx]
def to0(idx): return [i-1 for i in idx]
def fmt(v, d): return f'{v:.{d}f}'

# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("⚙️ Pengaturan SSA")

    st.header("1️⃣ Data")
    data_source = st.radio("Sumber:", ["Upload CSV/Excel", "Data Demo"])
    uploaded_file = None
    if data_source == "Upload CSV/Excel":
        uploaded_file = st.file_uploader("Upload", type=["csv", "xlsx", "xls"])

    st.header("2️⃣ Window Length (L)")
    L_mode = st.radio("Mode:", ["Auto (N/2)", "Manual", "Optimasi L"])
    L_manual = 48
    if L_mode == "Manual":
        L_manual = st.number_input("L:", min_value=2, value=48, step=1)
    if L_mode == "Optimasi L":
        opt_L_from = st.text_input("L dari:", "N/4")
        opt_L_to = st.text_input("L sampai:", "N/2")
        opt_L_step = st.number_input("Step:", min_value=1, value=1, step=1)

    st.header("3️⃣ Train / Test")
    train_pct = st.slider("Training %", 50, 95, 80, 5)

    st.header("4️⃣ Desimal")
    nd = st.slider("Angka di belakang koma", 2, 10, 4, 1)

# ══════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════
st.title("📈 SSA v5.6")
ts = None; sname = "Time Series"

if data_source == "Data Demo":
    np.random.seed(42); ND = 300; td = np.arange(ND)
    trend = 0.015*td + 5 + 0.00005*td**2
    seas1 = 3.0 * np.sin(2*np.pi*td/12)
    seas2 = 1.5 * np.sin(2*np.pi*td/6)
    seas3 = 0.8 * np.sin(2*np.pi*td/4 + 0.5)
    seas4 = 0.5 * np.cos(2*np.pi*td/2.4)
    noise = np.random.normal(0, 0.6, ND)
    ts = trend + seas1 + seas2 + seas3 + seas4 + noise
    sname = "Demo (Trend + 4 Seasonal + Noise)"
else:
    if uploaded_file is None:
        st.warning("⬆️ Upload file CSV atau Excel."); st.stop()
    if uploaded_file.name.endswith('.csv'):
        df_raw = pd.read_csv(uploaded_file)
    else:
        df_raw = pd.read_excel(uploaded_file)
    st.subheader("📄 Preview Data")
    st.dataframe(df_raw.head(10), use_container_width=True)
    st.caption(f"Shape: {df_raw.shape[0]} baris × {df_raw.shape[1]} kolom")
    all_cols = df_raw.columns.tolist()
    num_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        st.error("❌ Tidak ada kolom numerik."); st.stop()
    st.markdown(f"**Kolom tersedia**: {', '.join(all_cols)}")
    st.markdown(f"**Kolom numerik**: {', '.join(num_cols)}")
    target_col = st.selectbox("Pilih kolom untuk analisis SSA:", num_cols)
    ts = df_raw[target_col].dropna().values.astype(float)
    sname = target_col

N = len(ts); train_n = int(N * train_pct / 100); test_n = N - train_n

def parse_L(expr, N):
    expr = expr.strip().upper().replace(' ', '')
    if '/' in expr:
        p = expr.split('/')
        num = N if p[0] == 'N' else int(p[0])
        return max(2, num // int(p[1]))
    try: return max(2, int(expr))
    except: return N // 2

# ══════════════════════════════════════════════════════════════
# OPTIMASI L — button-driven + full table
# ══════════════════════════════════════════════════════════════
if L_mode == "Optimasi L":
    Lmin = parse_L(opt_L_from, N); Lmax = parse_L(opt_L_to, N)
    st.subheader("🔍 Optimasi Window Length (L)")
    st.markdown(f"Range: L = {Lmin} .. {Lmax}, step = {int(opt_L_step)}")

    if st.button("▶️ Jalankan Optimasi L", type='primary', key='opt_L_btn'):
        progress = st.progress(0, text="Optimasi L...")
        L_range = list(range(Lmin, Lmax+1, max(1, int(opt_L_step))))
        results = []
        for idx_l, L_try in enumerate(L_range):
            try:
                ssa_try = SSA(ts, window_length=L_try)
                # Auto-group & reconstruct
                n_sig = max(1, int(np.sum(ssa_try.contribution > 1.0)))
                n_sig = min(n_sig, ssa_try.d)
                grp_try = {'Signal': list(range(n_sig))}
                ssa_try.reconstruct(grp_try)
                res_try = ssa_try.reconstructed['_Residual']
                rmse = np.sqrt(np.mean(res_try**2))
                mae = np.mean(np.abs(res_try))
                r2 = 1 - np.sum(res_try**2)/np.sum((ts - np.mean(ts))**2)
                results.append({
                    'L': L_try, 'K': ssa_try.K, 'd': ssa_try.d,
                    'n_signal': n_sig,
                    'Top_5_Cum%': round(ssa_try.cumulative_contribution[min(4, ssa_try.d-1)], 2),
                    'RMSE': round(rmse, 6), 'MAE': round(mae, 6), 'R²': round(r2, 6)
                })
            except:
                results.append({'L': L_try, 'K': N-L_try+1, 'd': 0,
                    'n_signal': 0, 'Top_5_Cum%': 0,
                    'RMSE': np.nan, 'MAE': np.nan, 'R²': np.nan})
            progress.progress((idx_l+1)/len(L_range), text=f"L={L_try} ({idx_l+1}/{len(L_range)})")
        progress.empty()
        df_opt = pd.DataFrame(results)
        st.session_state['opt_results'] = df_opt

    df_opt = st.session_state.get('opt_results')
    if df_opt is not None and len(df_opt) > 0:
        valid = df_opt.dropna(subset=['RMSE'])
        if len(valid) > 0:
            best = valid.loc[valid['RMSE'].idxmin()]
            st.success(f"✅ **L optimal = {int(best['L'])}** — RMSE = {best['RMSE']:.6f}, R² = {best['R²']:.6f}")
            st.session_state['opt_best_L'] = int(best['L'])

            # Plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))
            ax1.plot(valid['L'], valid['RMSE'], 'b.-', lw=1, ms=4)
            ax1.axvline(best['L'], color='red', ls='--', lw=1.5, label=f"Best L={int(best['L'])}")
            ax1.set_title('RMSE vs L', fontweight='bold'); ax1.set_xlabel('L'); ax1.set_ylabel('RMSE')
            ax1.legend()
            ax2.plot(valid['L'], valid['R²'], 'g.-', lw=1, ms=4)
            ax2.axvline(best['L'], color='red', ls='--', lw=1.5, label=f"Best L={int(best['L'])}")
            ax2.set_title('R² vs L', fontweight='bold'); ax2.set_xlabel('L'); ax2.set_ylabel('R²')
            ax2.legend()
            plt.tight_layout(); st.pyplot(fig); plt.close()

            # Full table
            with st.expander("📋 Tabel Lengkap Optimasi L", expanded=True):
                fmts_o = {'RMSE': f'{{:.{nd}f}}', 'MAE': f'{{:.{nd}f}}', 'R²': f'{{:.{nd}f}}',
                           'Top_5_Cum%': '{:.2f}'}
                styled = df_opt.style.format(fmts_o, na_rep='—')
                # Highlight best row
                def highlight_best(row):
                    if row['L'] == best['L']:
                        return ['background-color: #d4edda'] * len(row)
                    return [''] * len(row)
                styled = df_opt.style.format(fmts_o, na_rep='—').apply(highlight_best, axis=1)
                st.dataframe(styled, use_container_width=True, height=400)
        else:
            st.error("Semua L gagal.")

    L_val = st.session_state.get('opt_best_L', N // 2)

elif L_mode == "Manual":
    L_val = min(int(L_manual), N // 2)
else:
    L_val = N // 2

ssa = SSA(ts, window_length=L_val, name=sname)
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("N", N); c2.metric("L", ssa.L); c3.metric("K", ssa.K)
c4.metric("d", ssa.d); c5.metric("Train", train_n); c6.metric("Test", test_n)

# ══════════════════════════════════════════════════════════════
tabs = st.tabs(["📊 Data", "📉 Scree", "🔢 Eigenvectors", "🔁 Paired EV",
    "📶 Komponen", "🎵 Periodogram", "🔲 W-Corr & Group",
    "🧩 Rekonstruksi", "🔮 Forecast Test", "🔮 Forecast Future",
    "📐 Koefisien", "📏 Bootstrap Test", "📏 Bootstrap Future",
    "🧪 Residual", "🎲 Monte Carlo", "📥 Download"])

# ── TAB 0: DATA ──
with tabs[0]:
    st.subheader("Data Original")
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.plot(range(1, N+1), ts, 'b-', lw=1)
    ax.axvline(train_n, color='orange', ls='--', lw=1.5, label=f'Split (t={train_n})')
    ax.set_title(sname, fontweight='bold'); ax.set_xlabel('t'); ax.legend()
    st.pyplot(fig); plt.close()

# ── TAB 1: SCREE ──
with tabs[1]:
    st.subheader("Scree Plot")
    ns = st.slider("Komponen:", 2, min(100, ssa.d), min(20, ssa.d), key='s1')
    ns = min(ns, ssa.d); x = np.arange(1, ns+1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 4.5))
    a1.bar(x, ssa.singular_values[:ns], color='steelblue', alpha=.7, edgecolor='navy')
    a1.plot(x, ssa.singular_values[:ns], 'ro-', ms=3)
    a1.set_title('Singular Values', fontweight='bold'); a1.set_xlabel('Komponen')
    if ns <= 50: a1.set_xticks(x)
    a2.bar(x, ssa.contribution[:ns], color='coral', alpha=.7, edgecolor='darkred', label='Individual')
    a2.plot(x, ssa.cumulative_contribution[:ns], 'ko-', ms=3, label='Kumulatif')
    a2.axhline(95, color='r', ls='--', alpha=.5, label='95%')
    a2.axhline(99, color='g', ls='--', alpha=.5, label='99%')
    a2.set_title('Kontribusi (%)', fontweight='bold'); a2.set_xlabel('Komponen')
    if ns <= 50: a2.set_xticks(x)
    a2.legend(fontsize=8); st.pyplot(fig); plt.close()
    with st.expander("📋 Tabel"):
        tbl = pd.DataFrame({'Komponen': x, 'σ': ssa.singular_values[:ns],
            'λ': ssa.eigenvalues[:ns], '%': ssa.contribution[:ns],
            'Cum%': ssa.cumulative_contribution[:ns]}).set_index('Komponen')
        st.dataframe(tbl.style.format({c: f'{{:.{nd}f}}' for c in ['σ','λ','%','Cum%']}),
                      use_container_width=True)

# ── TAB 2: EIGENVECTORS ──
with tabs[2]:
    st.subheader("Eigenvectors")
    ne = st.slider("Komponen:", 2, min(50, ssa.d), min(20, ssa.d), key='s2')
    ne = min(ne, ssa.d); nc_e = 4; nr_e = max(1, (ne+nc_e-1)//nc_e)
    fig, axes = plt.subplots(nr_e, nc_e, figsize=(14, 3*nr_e))
    af = np.array(axes).flatten()
    for i in range(len(af)):
        if i < ne:
            af[i].plot(ssa.U[:, i], 'b-', lw=.8)
            af[i].set_title(f'EV {i+1} ({ssa.contribution[i]:.1f}%)', fontsize=8)
            af[i].axhline(0, color='k', lw=.3)
        else: af[i].set_visible(False)
    plt.tight_layout(); st.pyplot(fig); plt.close()

# ── TAB 3: PAIRED ──
with tabs[3]:
    st.subheader("Paired Eigenvectors")
    ca, cb = st.columns(2)
    np_ = ca.slider("Pairs:", 1, min(50, ssa.d-1), min(20, ssa.d-1), key='s3a')
    ps = cb.radio("Gaya:", ["Titik", "Garis+Titik", "Garis"], horizontal=True, key='s3b')
    np_ = min(np_, ssa.d-1); ncp = min(4, np_); nrp = max(1, (np_+ncp-1)//ncp)
    fig, axes = plt.subplots(nrp, ncp, figsize=(3.8*ncp, 3.8*nrp))
    af = np.array(axes).flatten() if np_ > 1 else [axes]
    for idx in range(len(af)):
        if idx < np_:
            xi, yi = ssa.U[:, idx], ssa.U[:, idx+1]
            if "+" in ps:
                af[idx].plot(xi, yi, '-', color='gray', lw=.5, alpha=.5)
                af[idx].scatter(xi, yi, s=10, c=np.arange(len(xi)), cmap='viridis', edgecolors='none', alpha=.8)
            elif "Garis" in ps:
                af[idx].plot(xi, yi, '-', color='steelblue', lw=.7)
            else:
                af[idx].scatter(xi, yi, s=8, alpha=.6, c='steelblue')
            af[idx].set_xlabel(f'EV {idx+1}'); af[idx].set_ylabel(f'EV {idx+2}')
            af[idx].set_aspect('equal')
            af[idx].axhline(0, color='k', lw=.3); af[idx].axvline(0, color='k', lw=.3)
        else: af[idx].set_visible(False)
    plt.tight_layout(); st.pyplot(fig); plt.close()

# ── TAB 4: KOMPONEN ──
with tabs[4]:
    st.subheader("Komponen Individual")
    nr = st.slider("Komponen:", 1, min(50, ssa.d), min(20, ssa.d), key='s4')
    nr = min(nr, ssa.d)
    fig, axes = plt.subplots(nr, 1, figsize=(13, 2.2*nr), sharex=True)
    if nr == 1: axes = [axes]
    for i in range(nr):
        rc = ssa.reconstruct_component(i); axes[i].plot(rc, 'b-', lw=.8)
        axes[i].set_ylabel(f'F{i+1}', fontweight='bold'); axes[i].axhline(0, color='k', lw=.3)
        axes[i].text(0.98, .8, f'σ={ssa.singular_values[i]:.2f} ({ssa.contribution[i]:.1f}%)',
            transform=axes[i].transAxes, fontsize=8, ha='right',
            bbox=dict(boxstyle='round', fc='wheat', alpha=.5))
    axes[-1].set_xlabel('t'); plt.tight_layout(); st.pyplot(fig); plt.close()

# ── TAB 5: PERIODOGRAM ──
with tabs[5]:
    st.subheader("Periodogram")
    npg = st.slider("Komponen:", 1, min(50, ssa.d), min(20, ssa.d), key='s5')
    npg = min(npg, ssa.d)
    fig, axes = plt.subplots(npg, 1, figsize=(12, 2.2*npg), sharex=True)
    if npg == 1: axes = [axes]
    for i in range(npg):
        rc = ssa.reconstruct_component(i); freqs, psd = periodogram(rc, fs=1.0)
        axes[i].plot(freqs, psd, 'b-', lw=.8); axes[i].fill_between(freqs, psd, alpha=.3)
        axes[i].set_ylabel(f'F{i+1}', fontweight='bold')
        if len(freqs) > 1:
            pk = freqs[np.argmax(psd[1:])+1]; T = 1/pk if pk > 0 else np.inf
            axes[i].text(0.97, .75, f'f={pk:.4f} T≈{T:.1f}', transform=axes[i].transAxes,
                fontsize=8, ha='right', bbox=dict(boxstyle='round', fc='lightyellow', alpha=.8))
    axes[-1].set_xlabel('Freq'); plt.tight_layout(); st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════════
# TAB 6: W-CORR & GROUP
# Method selectbox OUTSIDE form → refreshes UI for different options
# Settings INSIDE form → no refresh until button clicked
# ══════════════════════════════════════════════════════════════
with tabs[6]:
    st.subheader("W-Correlation & Grouping")

    nwc = st.slider("Komponen W-Corr:", 4, min(100, ssa.d), min(20, ssa.d), key='s6wc')
    nc = min(nwc, ssa.d); wcorr = ssa.w_correlation(nc)
    fig, ax = plt.subplots(figsize=(max(5, nc*.45), max(4, nc*.4)))
    im = ax.imshow(np.abs(wcorr), cmap='hot_r', vmin=0, vmax=1, interpolation='nearest', origin='lower')
    ax.set_xticks(range(nc)); ax.set_yticks(range(nc))
    fs_t = max(4, 8-nc//8)
    ax.set_xticklabels([f'F{i+1}' for i in range(nc)], fontsize=fs_t, rotation=90)
    ax.set_yticklabels([f'F{i+1}' for i in range(nc)], fontsize=fs_t)
    ax.set_title('|ρʷ|', fontweight='bold'); plt.colorbar(im, ax=ax, shrink=.8)
    if nc <= 20:
        for i in range(nc):
            for j in range(nc):
                v = np.abs(wcorr[i, j])
                ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                        fontsize=max(3, 6-nc//6), color='white' if v > .6 else 'black')
    st.pyplot(fig); plt.close()

    st.markdown("---")
    st.subheader("🔧 Grouping")

    # Method selectbox OUTSIDE form — switches UI immediately
    gmode = st.selectbox("Metode Grouping:", ["Auto: Hierarchical", "Auto: Periodogram", "Manual"], key='s6gm')

    # Settings INSIDE form — no refresh until submit
    if "Hierarchical" in gmode:
        with st.form("form_grp_hc"):
            st.markdown("**Hierarchical Clustering**")
            gc1, gc2, gc3 = st.columns(3)
            hc_lk = gc1.selectbox("Linkage:", ['average','single','complete','centroid','ward','weighted','median'], key='s6lk')
            n_sig_grp = gc2.number_input("Grup sinyal:", 1, 20, 2, 1, key='s6ng')
            hc_comp = gc3.slider("Komp clustering:", 4, min(100, ssa.d), min(20, ssa.d), key='s6hc')
            sub_grp = st.form_submit_button("▶️ Terapkan Grouping", type='primary')
        if sub_grp:
            groups = ssa.auto_group_wcorr(min(hc_comp, ssa.d), n_sig_grp, hc_lk)
            st.session_state['groups'] = groups
            st.session_state['grp_mode'] = 'hc'
            st.session_state['hc_comp'] = min(hc_comp, ssa.d)
            st.session_state['hc_lk'] = hc_lk
            st.session_state['n_sig_grp'] = n_sig_grp
            st.success("✅ Grouping diterapkan!")

    elif "Periodogram" in gmode:
        with st.form("form_grp_pg"):
            st.markdown("**Auto Periodogram**")
            gc1, gc2, gc3 = st.columns(3)
            pg_comp = gc1.slider("Komp:", 4, min(100, ssa.d), min(20, ssa.d), key='s6pc')
            pg_ft = gc2.number_input("Freq thresh:", 0.001, 0.2, 0.02, 0.005, format="%.3f", key='s6ft')
            pg_pt = gc3.number_input("Pair tol:", 0.001, 0.2, 0.01, 0.005, format="%.3f", key='s6pt')
            sub_grp = st.form_submit_button("▶️ Terapkan Grouping", type='primary')
        if sub_grp:
            groups = ssa.auto_group_periodogram(min(pg_comp, ssa.d), pg_ft, pg_pt)
            st.session_state['groups'] = groups
            st.session_state['grp_mode'] = 'pg'
            st.success("✅ Grouping diterapkan!")

    else:  # Manual
        with st.form("form_grp_manual"):
            st.markdown(f"""**Manual Grouping** — Indeks **1-based**, pisah grup dgn `;`

Komponen di luar grup sinyal → otomatis **Others (Noise)**. Default total = L = {ssa.L}.""")
            max_comp = st.number_input("Komponen total:", 2, ssa.d, min(ssa.L, ssa.d), 1, key='s6mc')
            gn_in = st.text_input("Nama grup (koma):", "Trend, Seasonal_1, Seasonal_2", key='s6gn')
            gi_in = st.text_input("Indeks per grup (`;`):", "1,2 ; 3,4 ; 5,6", key='s6gi')
            sub_grp = st.form_submit_button("▶️ Terapkan Grouping", type='primary')
        if sub_grp:
            gnames = [g.strip() for g in gn_in.split(',')]
            gidxs = gi_in.split(';')
            groups = {}; used_idx = set()
            valid = True
            for gn, gi in zip(gnames, gidxs):
                try:
                    idxs = to0([int(x.strip()) for x in gi.split(',') if x.strip()])
                    groups[gn] = idxs; used_idx.update(idxs)
                except:
                    st.error(f"Format salah: '{gn}'"); valid = False
            if valid:
                others = sorted(set(range(max_comp)) - used_idx)
                if others:
                    groups['Others (Noise)'] = others
                st.session_state['groups'] = groups
                st.session_state['grp_mode'] = 'manual'
                st.success("✅ Grouping diterapkan!")

    # ── Show results ──
    groups = st.session_state.get('groups')
    if groups:
        if st.session_state.get('grp_mode') == 'hc':
            hcc = st.session_state.get('hc_comp', 12)
            ssa.auto_group_wcorr(hcc, st.session_state.get('n_sig_grp', 2),
                                 st.session_state.get('hc_lk', 'average'))
            if hasattr(ssa, '_hc_linkage'):
                fig, ax = plt.subplots(figsize=(max(10, hcc*.5), 4))
                dendrogram(ssa._hc_linkage, labels=[f'F{i+1}' for i in range(hcc)],
                           leaf_rotation=90, leaf_font_size=max(5, 9-hcc//8), ax=ax)
                ax.set_title('Dendrogram', fontweight='bold')
                st.pyplot(fig); plt.close()

        st.subheader("Hasil Grouping")
        rows_g = []
        for k, v in groups.items():
            ctr = sum(ssa.contribution[i] for i in v if i < len(ssa.contribution))
            rows_g.append({'Grup': k, 'Komponen (1-based)': str(to1(v)), 'Kontribusi (%)': round(ctr, nd)})
        st.dataframe(pd.DataFrame(rows_g), use_container_width=True)

        if st.session_state.get('grp_mode') == 'manual':
            others = [v for k, v in groups.items() if 'others' in k.lower() or 'noise' in k.lower()]
            if others:
                st.info(f"🔹 **Others/Noise**: komponen {to1(others[0])} — total {len(others[0])} komponen")
    else:
        st.info("Pilih metode, atur parameter, lalu klik **▶️ Terapkan Grouping**.")

# ── TAB 7: REKONSTRUKSI ──
with tabs[7]:
    st.subheader("Rekonstruksi")
    groups = st.session_state.get('groups')
    if not groups: st.info("⬅️ Grouping dulu."); st.stop()
    ssa.reconstruct(groups)
    ug = {k: v for k, v in ssa.reconstructed.items() if not k.startswith('_')}; ng = len(ug)
    fig, axes = plt.subplots(ng+1, 1, figsize=(13, 2.8*(ng+1)), sharex=True)
    t_ax = np.arange(1, N+1)
    axes[0].plot(t_ax, ts, 'b-', alpha=.5, lw=.8, label='Original')
    axes[0].plot(t_ax, ssa.reconstructed['_Total'], 'r-', lw=1, label='Total')
    axes[0].legend(); axes[0].set_title('Original vs Rekonstruksi', fontweight='bold')
    colors = plt.cm.Set1(np.linspace(0, 1, max(ng, 1)))
    for idx, (gn, gv) in enumerate(ug.items()):
        axes[idx+1].plot(t_ax, gv, color=colors[idx], lw=.8)
        axes[idx+1].set_ylabel(gn, fontweight='bold'); axes[idx+1].axhline(0, color='k', lw=.3)
        sh = sum(ssa.contribution[i] for i in ssa.groups[gn] if i < len(ssa.contribution))
        axes[idx+1].text(0.98, .8, f'{sh:.1f}%', transform=axes[idx+1].transAxes,
            fontsize=10, ha='right', fontweight='bold',
            bbox=dict(boxstyle='round', fc='lightyellow', alpha=.8))
    axes[-1].set_xlabel('t'); plt.tight_layout(); st.pyplot(fig); plt.close()
    st.metric("RMSE Rekonstruksi", fmt(np.sqrt(np.mean(ssa.reconstructed['_Residual']**2)), nd))

# ══════════════════════════════════════════════════════════════
# TAB 8: FORECAST TEST
# ══════════════════════════════════════════════════════════════
with tabs[8]:
    st.subheader("Forecast Test")
    groups = st.session_state.get('groups')
    if not groups: st.info("⬅️ Grouping dulu."); st.stop()

    with st.form("form_fc_test"):
        st.markdown(f"SSA fit pada **training** (t=1..{train_n}), forecast **{test_n}** langkah.")
        sig_grp = {k: v for k, v in groups.items() if 'noise' not in k.lower() and 'others' not in k.lower()}
        if not sig_grp: sig_grp = groups
        cf1, cf2, cf3 = st.columns(3)
        do_r = cf1.checkbox("R-forecast", True, key='t8r')
        do_v = cf2.checkbox("V-forecast", True, key='t8v')
        fc_pm = cf3.radio("Tampilan:", ["Gabung", "Pisah"], horizontal=True, key='t8pm')
        submitted = st.form_submit_button("▶️ Jalankan Forecast Test", type='primary')

    if submitted:
        L_train = min(L_val, train_n // 2)
        ssa_train = SSA(ts[:train_n], window_length=L_train, name='train')
        ssa_train.reconstruct(sig_grp)
        fc_r = ssa_train.forecast_recurrent(sig_grp, steps=test_n) if do_r else None
        fc_v = ssa_train.forecast_vector(sig_grp, steps=test_n) if do_v else None
        st.session_state['ssa_train'] = ssa_train
        st.session_state['sig_grp'] = sig_grp
        st.session_state['fc_r_test'] = fc_r
        st.session_state['fc_v_test'] = fc_v
        st.session_state['fc_pm'] = fc_pm
        st.success("✅ Forecast Test selesai!")

    fc_r = st.session_state.get('fc_r_test')
    fc_v = st.session_state.get('fc_v_test')
    fc_pm_s = st.session_state.get('fc_pm', 'Gabung')

    if fc_r is not None or fc_v is not None:
        t_data = np.arange(1, N+1)
        if fc_pm_s == "Gabung":
            fig, ax = plt.subplots(figsize=(13, 5))
            ax.plot(t_data, ts, 'b-', lw=1, label='Actual', alpha=.7)
            if fc_r is not None:
                ax.plot(np.arange(1, len(fc_r)+1), fc_r, 'r--', lw=1.2, label='R-Forecast')
            if fc_v is not None:
                ax.plot(np.arange(1, len(fc_v)+1), fc_v, 'g--', lw=1.2, label='V-Forecast')
            ax.axvline(train_n, color='orange', ls='--', lw=1.5, label=f'Split (t={train_n})')
            ax.set_title('Forecast Test', fontweight='bold'); ax.set_xlabel('t'); ax.legend(fontsize=8)
            st.pyplot(fig); plt.close()
        else:
            for fc, lbl, clr in [(fc_r, 'R', 'red'), (fc_v, 'V', 'green')]:
                if fc is not None:
                    fig, ax = plt.subplots(figsize=(13, 4))
                    ax.plot(t_data, ts, 'b-', lw=1, label='Actual', alpha=.7)
                    ax.plot(np.arange(1, len(fc)+1), fc, '--', color=clr, lw=1.2, label=f'{lbl}-Forecast')
                    ax.axvline(train_n, color='orange', ls='--', lw=1.5)
                    ax.set_title(f'{lbl}-Forecast', fontweight='bold'); ax.legend()
                    st.pyplot(fig); plt.close()

        st.subheader(f"Evaluasi (n_test={test_n})")
        actual_test = ts[train_n:]
        def show_ev(label, fc):
            rows = []
            for sp, act, pred in [('TRAIN', ts[:train_n], fc[:train_n]),
                                   ('TEST', actual_test, fc[train_n:train_n+test_n]),
                                   ('OVERALL', ts, fc[:N])]:
                d = SSA.evaluate(act, pred)
                rows.append({'Split': sp, 'N': d['N'], 'RMSE': d['RMSE'], 'MAE': d['MAE'],
                    'MAPE%': d['MAPE_pct'], 'sMAPE%': d['sMAPE_pct'], 'R²': d['R2'], 'NRMSE': d['NRMSE']})
            fmts = {c: f'{{:.{nd}f}}' for c in ['RMSE','MAE','MAPE%','sMAPE%','R²','NRMSE']}
            st.markdown(f"**{label}**")
            st.dataframe(pd.DataFrame(rows).style.format(fmts), use_container_width=True)
        if fc_r is not None: show_ev("📊 R-forecast", fc_r)
        if fc_v is not None: show_ev("📊 V-forecast", fc_v)

        for fc, lbl in [(fc_r, 'R'), (fc_v, 'V')]:
            if fc is not None:
                err_tr = ts[:train_n] - fc[:train_n]
                err_te = actual_test - fc[train_n:train_n+test_n]
                fig, axes = plt.subplots(1, 2, figsize=(13, 4))
                axes[0].bar(range(1, train_n+1), err_tr, color='green', alpha=.5, width=1, label='Train')
                axes[0].bar(range(train_n+1, N+1), err_te, color='orange', alpha=.5, width=1, label='Test')
                axes[0].axhline(0, color='k', lw=.5)
                axes[0].set_title(f'Error ({lbl})', fontweight='bold'); axes[0].legend()
                pred_all = fc[:N]
                axes[1].scatter(pred_all, ts, s=8, alpha=.5, c='steelblue')
                mn_, mx_ = min(ts.min(), pred_all.min()), max(ts.max(), pred_all.max())
                axes[1].plot([mn_, mx_], [mn_, mx_], 'r--', lw=1)
                axes[1].set_title(f'Actual vs Pred ({lbl})', fontweight='bold')
                axes[1].set_xlabel('Predicted'); axes[1].set_ylabel('Actual')
                plt.tight_layout(); st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════════
# TAB 9: FORECAST FUTURE
# ══════════════════════════════════════════════════════════════
with tabs[9]:
    st.subheader("Forecast Future")
    groups = st.session_state.get('groups')
    if not groups: st.info("⬅️ Grouping dulu."); st.stop()

    with st.form("form_fc_future"):
        st.markdown(f"SSA fit pada **seluruh data** (t=1..{N}). Pure forecast ke depan.")
        sig_grp_f = {k: v for k, v in groups.items() if 'noise' not in k.lower() and 'others' not in k.lower()}
        if not sig_grp_f: sig_grp_f = groups
        cf1, cf2, cf3 = st.columns(3)
        h_future = cf1.number_input("Langkah (h):", 1, 500, 24, 1, key='t9h')
        do_r_f = cf2.checkbox("R-forecast", True, key='t9r')
        do_v_f = cf3.checkbox("V-forecast", True, key='t9v')
        submitted_f = st.form_submit_button("▶️ Jalankan Forecast Future", type='primary')

    if submitted_f:
        ssa_full = SSA(ts, window_length=L_val, name=sname)
        ssa_full.reconstruct(sig_grp_f)
        fc_r_f = ssa_full.forecast_recurrent(sig_grp_f, steps=h_future) if do_r_f else None
        fc_v_f = ssa_full.forecast_vector(sig_grp_f, steps=h_future) if do_v_f else None
        st.session_state['ssa_full'] = ssa_full
        st.session_state['sig_grp_f'] = sig_grp_f
        st.session_state['h_future'] = h_future
        st.session_state['fc_r_future'] = fc_r_f
        st.session_state['fc_v_future'] = fc_v_f
        st.success("✅ Forecast Future selesai!")

    fc_r_f = st.session_state.get('fc_r_future')
    fc_v_f = st.session_state.get('fc_v_future')
    h_fut = st.session_state.get('h_future', 24)

    if fc_r_f is not None or fc_v_f is not None:
        fig, ax = plt.subplots(figsize=(13, 5))
        ax.plot(range(1, N+1), ts, 'b-', lw=1, label='Data', alpha=.7)
        if fc_r_f is not None:
            ax.plot(np.arange(1, len(fc_r_f)+1), fc_r_f, 'r--', lw=1.2, label='R')
        if fc_v_f is not None:
            ax.plot(np.arange(1, len(fc_v_f)+1), fc_v_f, 'g--', lw=1.2, label='V')
        ax.axvline(N, color='gray', ls=':', lw=1.5, label=f't={N}')
        ax.axvspan(N, N+h_fut, alpha=.1, color='green', label=f'Future ({h_fut})')
        ax.set_title(f'Future Forecast (h={h_fut})', fontweight='bold')
        ax.set_xlabel('t'); ax.legend(fontsize=8)
        st.pyplot(fig); plt.close()
        fut_rows = []
        for h in range(h_fut):
            row = {'t': N+h+1}
            if fc_r_f is not None and N+h < len(fc_r_f): row['R'] = fc_r_f[N+h]
            if fc_v_f is not None and N+h < len(fc_v_f): row['V'] = fc_v_f[N+h]
            fut_rows.append(row)
        fmts_ft = {c: f'{{:.{nd}f}}' for c in ['R','V'] if c in fut_rows[0]}
        st.dataframe(pd.DataFrame(fut_rows).set_index('t').style.format(fmts_ft), use_container_width=True)

# ── TAB 10: KOEFISIEN ──
with tabs[10]:
    st.subheader("Parameter & Koefisien")
    ssa_tr = st.session_state.get('ssa_train')
    ssa_fu = st.session_state.get('ssa_full')
    src = st.radio("Dari:", ["SSA Train", "SSA Full"], horizontal=True, key='t10src')
    obj = ssa_tr if 'Train' in src else ssa_fu
    if obj is None:
        st.info("⬅️ Jalankan forecast dulu."); st.stop()
    if hasattr(obj, 'lrr_info'):
        info = obj.lrr_info; et = to1(info['eigentriple_indices'])
        st.markdown(f"""### 🔴 R-Forecast — LRR
| Parameter | Nilai |
|-----------|-------|
| Koefisien (L−1) | **{info['num_coefficients']}** |
| Eigentriple | **{info['num_eigentriples_used']}**: {et} |
| ν² | **{fmt(info['nu_squared'],nd)}** {'✅' if info['nu_squared']<1 else '❌'} |""")
        coef = obj.lrr_coefficients; nsc = min(30, len(coef))
        st.dataframe(pd.DataFrame({'j': range(1, nsc+1), 'a_j': coef[:nsc]}).style.format(
            {'a_j': f'{{:.{nd}f}}'}), use_container_width=True)
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.bar(range(1, len(coef)+1), coef, color='steelblue', alpha=.7, width=1)
        ax.axhline(0, color='k', lw=.5); ax.set_title('LRR', fontweight='bold')
        st.pyplot(fig); plt.close()
    if hasattr(obj, 'vforecast_info'):
        info_v = obj.vforecast_info; et_v = to1(info_v['eigentriple_indices'])
        st.markdown(f"""### 🟢 V-Forecast — P_π
| Parameter | Nilai |
|-----------|-------|
| Koefisien | **{info_v['num_coefficients']}** |
| Eigentriple | **{info_v['num_eigentriples_used']}**: {et_v} |
| ν² | **{fmt(info_v['nu_squared'],nd)}** |""")
        cv = obj.vforecast_coefficients; nsv = min(30, len(cv))
        st.dataframe(pd.DataFrame({'j': range(1, nsv+1), 'P_π': cv[:nsv]}).style.format(
            {'P_π': f'{{:.{nd}f}}'}), use_container_width=True)
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.bar(range(1, len(cv)+1), cv, color='forestgreen', alpha=.7, width=1)
        ax.axhline(0, color='k', lw=.5); ax.set_title('P_π', fontweight='bold')
        st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════════
# TAB 11: BOOTSTRAP TEST
# ══════════════════════════════════════════════════════════════
with tabs[11]:
    st.subheader("Bootstrap CI & PI — Test")
    ssa_tr = st.session_state.get('ssa_train')
    sig_grp = st.session_state.get('sig_grp')
    if ssa_tr is None or sig_grp is None:
        st.info("⬅️ Jalankan **Forecast Test** dulu.")
    else:
        with st.form("form_boot_test"):
            st.markdown(f"Bootstrap **{test_n}** langkah, koefisien identik dengan Forecast Test.")
            cb1, cb2, cb3 = st.columns(3)
            bn = cb1.number_input("n_bootstrap:", 100, 2000, 300, 50, key='t11bn')
            bc = cb2.slider("Confidence:", 0.80, 0.99, 0.95, 0.01, key='t11bc')
            bm = cb3.selectbox("Metode:", ["recurrent", "vector"], key='t11bm')
            sub_bt = st.form_submit_button("▶️ Jalankan Bootstrap Test", type='primary')
        if sub_bt:
            with st.spinner(f"Bootstrap ({bn}x, {test_n} steps)..."):
                br = ssa_tr.bootstrap_intervals(sig_grp, test_n, bm, int(bn), float(bc))
            if br is None: st.error("❌ Bootstrap gagal.")
            else:
                st.session_state['boot_test'] = br
                st.success(f"✅ {br['n_success']} bootstrap berhasil")
        br = st.session_state.get('boot_test')
        if br is not None:
            h = np.arange(train_n+1, train_n+1+len(br['forecast_mean']))
            fig, ax = plt.subplots(figsize=(13, 5))
            ax.plot(range(1, N+1), ts, 'b-', lw=1, label='Actual', alpha=.7)
            ax.plot(h, br['forecast_mean'], 'k-', lw=1.5, label='Point Forecast')
            ax.fill_between(h, br['ci_lower'], br['ci_upper'], alpha=.35,
                            color='dodgerblue', label=f'{br["confidence"]*100:.0f}% CI')
            ax.fill_between(h, br['pi_lower'], br['pi_upper'], alpha=.15,
                            color='orange', label=f'{br["confidence"]*100:.0f}% PI')
            ax.axvline(train_n, color='orange', ls='--', lw=1.5)
            ax.set_title('Bootstrap Test', fontweight='bold')
            ax.set_xlabel('t'); ax.legend(fontsize=8)
            st.pyplot(fig); plt.close()
            st.subheader("Evaluasi Interval")
            actual_test = ts[train_n:]
            for iv, lo, up in [("CI", br['ci_lower'][:test_n], br['ci_upper'][:test_n]),
                                ("PI", br['pi_lower'][:test_n], br['pi_upper'][:test_n])]:
                m = SSA.evaluate_intervals(actual_test, lo, up, float(br['confidence']))
                st.markdown(f"**{iv}** (n={test_n})")
                fiv = {c: f'{{:.{nd}f}}' for c in ['PICP','PINAW','ACE','CWC','Winkler_Score','Mean_Width']}
                st.dataframe(pd.DataFrame([{k: v for k, v in m.items()
                    if k not in ('N','Nominal_Coverage')}]).style.format(fiv), use_container_width=True)

# ══════════════════════════════════════════════════════════════
# TAB 12: BOOTSTRAP FUTURE
# ══════════════════════════════════════════════════════════════
with tabs[12]:
    st.subheader("Bootstrap CI & PI — Future")
    ssa_fu = st.session_state.get('ssa_full')
    sig_grp_f = st.session_state.get('sig_grp_f')
    h_fut = st.session_state.get('h_future', 24)
    if ssa_fu is None or sig_grp_f is None:
        st.info("⬅️ Jalankan **Forecast Future** dulu.")
    else:
        with st.form("form_boot_future"):
            st.markdown(f"Bootstrap dari SSA full data, **{h_fut}** langkah future.")
            cb1, cb2, cb3 = st.columns(3)
            bn_f = cb1.number_input("n_bootstrap:", 100, 2000, 300, 50, key='t12bn')
            bc_f = cb2.slider("Confidence:", 0.80, 0.99, 0.95, 0.01, key='t12bc')
            bm_f = cb3.selectbox("Metode:", ["recurrent", "vector"], key='t12bm')
            sub_bf = st.form_submit_button("▶️ Jalankan Bootstrap Future", type='primary')
        if sub_bf:
            with st.spinner(f"Bootstrap ({bn_f}x, {h_fut} steps)..."):
                br_f = ssa_fu.bootstrap_intervals(sig_grp_f, h_fut, bm_f, int(bn_f), float(bc_f))
            if br_f is None: st.error("❌ Bootstrap gagal.")
            else:
                st.session_state['boot_future'] = br_f
                st.success(f"✅ {br_f['n_success']} bootstrap berhasil")
        br_f = st.session_state.get('boot_future')
        if br_f is not None:
            h = np.arange(N+1, N+1+len(br_f['forecast_mean']))
            fig, ax = plt.subplots(figsize=(13, 5))
            ax.plot(range(1, N+1), ts, 'b-', lw=1, label='Data', alpha=.7)
            ax.plot(h, br_f['forecast_mean'], 'k-', lw=1.5, label='Point Forecast')
            ax.fill_between(h, br_f['ci_lower'], br_f['ci_upper'], alpha=.35,
                            color='dodgerblue', label=f'{br_f["confidence"]*100:.0f}% CI')
            ax.fill_between(h, br_f['pi_lower'], br_f['pi_upper'], alpha=.15,
                            color='orange', label=f'{br_f["confidence"]*100:.0f}% PI')
            ax.axvline(N, color='gray', ls=':', lw=1.5)
            ax.set_title(f'Bootstrap Future (h={h_fut})', fontweight='bold')
            ax.set_xlabel('t'); ax.legend(fontsize=8)
            st.pyplot(fig); plt.close()
            with st.expander("📋 Tabel"):
                bi = pd.DataFrame({'t': h, 'Mean': br_f['forecast_mean'],
                    'CI_Lo': br_f['ci_lower'], 'CI_Up': br_f['ci_upper'],
                    'PI_Lo': br_f['pi_lower'], 'PI_Up': br_f['pi_upper']}).set_index('t')
                st.dataframe(bi.style.format({c: f'{{:.{nd}f}}' for c in bi.columns}), use_container_width=True)

# ── TAB 13: RESIDUAL ──
with tabs[13]:
    st.subheader("Analisis Residual")
    if not hasattr(ssa, 'reconstructed') or '_Residual' not in ssa.reconstructed:
        groups = st.session_state.get('groups')
        if groups: ssa.reconstruct(groups)
        else: st.info("⬅️ Grouping dulu."); st.stop()
    ri = ssa.residual_analysis(); res = ssa.residuals
    if ri:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mean", fmt(ri['mean'], nd)); c2.metric("Std", fmt(ri['std'], nd))
        c3.metric("Skew", fmt(ri['skewness'], nd)); c4.metric("Kurt", fmt(ri['kurtosis'], nd))
        rows = []
        if 'shapiro_stat' in ri:
            p = ri['shapiro_p']
            rows.append({'Test': 'Shapiro-Wilk', 'Stat': fmt(ri['shapiro_stat'], nd),
                'p': fmt(p, nd), 'Ket': 'Normal ✅' if p > .05 else 'Tdk Normal ❌'})
        pjb = ri['jarque_bera_p']
        rows.append({'Test': 'Jarque-Bera', 'Stat': fmt(ri['jarque_bera_stat'], nd),
            'p': fmt(pjb, nd), 'Ket': 'Normal ✅' if pjb > .05 else 'Tdk Normal ❌'})
        if 'ljung_box_p' in ri:
            plb = ri['ljung_box_p']
            rows.append({'Test': 'Ljung-Box', 'Stat': fmt(ri['ljung_box_stat'], nd),
                'p': fmt(plb, nd), 'Ket': 'WN ✅' if plb > .05 else 'Bukan WN ❌'})
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        fig = plt.figure(figsize=(13, 9)); gs = gridspec.GridSpec(2, 2, hspace=.35, wspace=.3)
        a1 = fig.add_subplot(gs[0, 0]); a1.plot(range(1, N+1), res, 'b-', lw=.5)
        a1.axhline(0, color='r', lw=.8); a1.set_title('Residual', fontweight='bold')
        a2 = fig.add_subplot(gs[0, 1])
        a2.hist(res, bins='auto', density=True, alpha=.7, color='steelblue', edgecolor='navy')
        xr = np.linspace(res.min(), res.max(), 100)
        a2.plot(xr, norm.pdf(xr, np.mean(res), np.std(res)), 'r-', lw=2)
        a2.set_title('Histogram', fontweight='bold')
        a3 = fig.add_subplot(gs[1, 0])
        plot_acf(res, ax=a3, lags=min(40, len(res)//3), alpha=.05)
        a3.set_title('ACF', fontweight='bold')
        a4 = fig.add_subplot(gs[1, 1]); probplot(res, dist='norm', plot=a4)
        a4.set_title('Q-Q', fontweight='bold')
        st.pyplot(fig); plt.close()

# ── TAB 14: MONTE CARLO ──
with tabs[14]:
    st.subheader("Monte Carlo SSA")
    with st.form("form_mc"):
        cm1, cm2 = st.columns(2)
        mc_s = cm1.number_input("Surrogates:", 100, 2000, 500, 100, key='t14s')
        mc_c = cm2.slider("Confidence:", 0.90, 0.99, 0.95, 0.01, key='t14c')
        sub_mc = st.form_submit_button("▶️ Jalankan Monte Carlo", type='primary')
    if sub_mc:
        with st.spinner(f"MC ({mc_s})..."):
            mc = ssa.monte_carlo_test(int(mc_s), float(mc_c))
        st.session_state['mc'] = mc
        st.success("✅ Selesai")
    mc = st.session_state.get('mc')
    if mc is not None:
        nmc = len(mc['eigenvalues']); xmc = np.arange(1, nmc+1)
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.semilogy(xmc, mc['eigenvalues'], 'ro-', ms=7, label='Data', zorder=5)
        ax.fill_between(xmc, mc['surrogate_lower'], mc['surrogate_upper'],
                        alpha=.3, color='blue', label=f'{mc["confidence"]*100:.0f}% CI')
        ax.semilogy(xmc, mc['surrogate_median'], 'b--', lw=1, label='Median')
        sig = mc['significant']
        ax.semilogy(xmc[sig], mc['eigenvalues'][sig], 'r*', ms=14, label='Sig', zorder=6)
        ax.semilogy(xmc[~sig], mc['eigenvalues'][~sig], 'kx', ms=9, label='Not Sig', zorder=6)
        ax.set_title('Monte Carlo SSA', fontweight='bold'); ax.set_xlabel('Komponen')
        ax.set_xticks(xmc); ax.legend()
        st.pyplot(fig); plt.close()
        st.success(f"Signifikan: **{int(np.sum(sig))}** dari {nmc}")

# ── TAB 15: DOWNLOAD ──
with tabs[15]:
    st.subheader("Download Excel")
    if not hasattr(ssa, 'reconstructed'):
        groups = st.session_state.get('groups')
        if groups: ssa.reconstruct(groups)
    tmp = os.path.join(tempfile.gettempdir(), 'SSA_Results.xlsx')
    ssa.save_results(tmp)
    with open(tmp, 'rb') as f: xb = f.read()
    os.remove(tmp)
    st.download_button("📥 SSA_Results.xlsx", xb, file_name="SSA_Results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    st.caption("SSA App v5.6")
