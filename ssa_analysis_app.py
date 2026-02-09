"""
══════════════════════════════════════════════════════════════
  APLIKASI SSA — Streamlit v4.0 (Revisi Lengkap)
══════════════════════════════════════════════════════════════
"""
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import norm, probplot
from scipy.signal import periodogram
from scipy.cluster.hierarchy import dendrogram
from statsmodels.graphics.tsaplots import plot_acf
import warnings, os, tempfile
warnings.filterwarnings('ignore')
from ssa_core import SSA, find_optimal_L

st.set_page_config(page_title="SSA Time Series 1D", layout="wide", page_icon="📈")
plt.rcParams.update({
    'figure.dpi':150,'font.size':10,'font.family':'serif',
    'axes.grid':True,'grid.alpha':0.3,'lines.linewidth':1.3,'figure.autolayout':True,
})

# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("⚙️ Pengaturan SSA")

    # ── Data ──
    st.header("1️⃣ Data")
    data_source = st.radio("Sumber data:", ["Upload CSV/Excel","Data Demo Sintetik"])
    uploaded_file = None
    if data_source == "Upload CSV/Excel":
        uploaded_file = st.file_uploader("Upload file", type=["csv","xlsx","xls"])

    # ── Window Length ──
    st.header("2️⃣ Window Length (L)")
    L_mode = st.radio("Mode:", ["Auto (N/2)","Manual","Optimasi L (lambat)"])
    L_manual = 48
    if L_mode == "Manual":
        L_manual = st.number_input("Nilai L:", min_value=2, value=48, step=1)

    # ── Plot Controls ──
    st.header("3️⃣ Jumlah Komponen Plot")
    n_comp_scree = st.slider("Scree plot", 4, 50, 20, key='scree')
    n_comp_ev = st.slider("Eigenvectors", 2, 20, 8, key='ev')
    n_comp_pair = st.slider("Paired eigenvectors", 1, 20, 6, key='pair')
    n_comp_pgram = st.slider("Periodogram", 2, 20, 8, key='pgram')
    n_comp_wcorr = st.slider("W-Correlation", 4, 50, 12, key='wcorr')
    n_comp_recon = st.slider("Komponen individual", 2, 20, 8, key='recon')

    # ── Grouping ──
    st.header("4️⃣ Grouping")
    grouping_mode = st.selectbox("Metode grouping:", [
        "Auto: Hierarchical Clustering (W-Corr)",
        "Auto: Periodogram-based",
        "Manual Grouping"
    ])
    if "Hierarchical" in grouping_mode:
        hc_linkage = st.selectbox("Linkage method:", [
            'average','single','complete','centroid','ward','weighted','median'
        ])
        hc_n_clusters = st.number_input("Jumlah cluster:", 2, 20, 3, 1)
        hc_n_comp = st.slider("Komponen untuk clustering:", 4, 50, 12, key='hc_comp')
    elif "Periodogram" in grouping_mode:
        pg_n_comp = st.slider("Komponen untuk pgram grouping:", 4, 50, 12, key='pg_comp')
        pg_freq_thresh = st.number_input("Freq threshold (trend < x):", 0.001, 0.1, 0.02, 0.005, format="%.3f")
        pg_pair_tol = st.number_input("Pair tolerance (freq diff):", 0.001, 0.1, 0.01, 0.005, format="%.3f")

    # ── Forecasting ──
    st.header("5️⃣ Forecasting")
    steps_ahead = st.number_input("Langkah forecast:", 1, 200, 24, 1)
    do_r_forecast = st.checkbox("Recurrent (R-forecast)", True)
    do_v_forecast = st.checkbox("Vector (V-forecast)", True)
    forecast_plot_mode = st.radio("Tampilan plot forecast:", [
        "Gabung (semua dalam 1 plot)",
        "Pisah (tiap metode plot sendiri)"
    ])

    # ── Train/Test ──
    st.header("6️⃣ Evaluasi Train/Test")
    train_pct = st.slider("Training %", 50, 95, 80, 5)

    # ── Bootstrap ──
    st.header("7️⃣ Bootstrap Intervals")
    do_bootstrap = st.checkbox("Hitung Bootstrap CI & PI", False)
    if do_bootstrap:
        boot_n = st.number_input("Jumlah bootstrap:", 100, 2000, 300, 50)
        boot_conf = st.slider("Confidence level (bootstrap):", 0.80, 0.99, 0.95, 0.01)
        boot_method = st.selectbox("Metode forecast bootstrap:", ["recurrent","vector"])

    # ── Monte Carlo ──
    st.header("8️⃣ Monte Carlo SSA")
    do_mc = st.checkbox("Jalankan Monte Carlo SSA", False)
    if do_mc:
        mc_surr = st.number_input("Surrogate:", 100, 2000, 500, 100)
        mc_conf = st.slider("Confidence (MC):", 0.90, 0.99, 0.95, 0.01)

# ══════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════
st.title("📈 Singular Spectrum Analysis (SSA) — v4.0")
st.caption("Dekomposisi · W-Correlation · Clustering/Periodogram Grouping · Rekonstruksi · "
           "R & V Forecast · Bootstrap CI/PI · Evaluasi · Residual · Monte Carlo")

ts = None; series_name = "Time Series"
if data_source == "Data Demo Sintetik":
    np.random.seed(42); N_d=200; t_d=np.arange(N_d)
    ts = 0.02*t_d+5 + 3*np.sin(2*np.pi*t_d/12) + 1.5*np.sin(2*np.pi*t_d/6) + np.random.normal(0,0.5,N_d)
    series_name = "Demo (Trend+Seasonal12+Seasonal6+Noise)"
    st.info("🔹 Data demo sintetik (N=200)")
else:
    if uploaded_file is None:
        st.warning("⬆️ Upload file CSV/Excel di sidebar, atau pilih Data Demo."); st.stop()
    df_raw = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
    st.subheader("Preview Data"); st.dataframe(df_raw.head(10), use_container_width=True)
    num_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols: st.error("Tidak ada kolom numerik."); st.stop()
    target_col = st.selectbox("Pilih kolom target:", num_cols)
    ts = df_raw[target_col].dropna().values.astype(float); series_name = target_col

N = len(ts); train_n = int(N * train_pct / 100)

# ── Tentukan L ──
if L_mode == "Optimasi L (lambat)":
    with st.spinner("🔍 Mencari L optimal..."): opt = find_optimal_L(ts)
    if opt: L_val=opt['best_L']; st.success(f"✅ L optimal = **{L_val}** (RMSE={opt['best_RMSE']:.6f})")
    else: L_val = N//2
elif L_mode == "Manual": L_val = min(int(L_manual), N//2)
else: L_val = N//2

# ── Init SSA ──
with st.spinner("⏳ SSA Decomposition..."):
    ssa = SSA(ts, window_length=L_val, name=series_name)

c1,c2,c3,c4 = st.columns(4)
c1.metric("N",N); c2.metric("L",ssa.L); c3.metric("K",ssa.K); c4.metric("d (rank)",ssa.d)

# ══════════════════════════════════════════════════════════════
# 1. ORIGINAL
# ══════════════════════════════════════════════════════════════
st.header("1️⃣ Data Original")
fig,ax = plt.subplots(figsize=(12,3.5))
ax.plot(ts,'b-',lw=1); ax.axvline(train_n,color='orange',ls='--',lw=1.5,label=f'Split (n={train_n})')
ax.set_title(f'Data: {series_name}',fontweight='bold'); ax.set_xlabel('Waktu'); ax.set_ylabel('Nilai'); ax.legend()
st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════════
# 2. SCREE PLOT
# ══════════════════════════════════════════════════════════════
st.header("2️⃣ Scree Plot & Kontribusi Varians")
ns = min(n_comp_scree, ssa.d)
fig,(ax1,ax2) = plt.subplots(1,2,figsize=(13,4.5)); x=np.arange(1,ns+1)
ax1.bar(x,ssa.singular_values[:ns],color='steelblue',alpha=.7,edgecolor='navy')
ax1.plot(x,ssa.singular_values[:ns],'ro-',ms=4)
ax1.set_title('Singular Values',fontweight='bold'); ax1.set_xlabel('Komponen'); ax1.set_ylabel('σ'); ax1.set_xticks(x)
ax2.bar(x,ssa.contribution[:ns],color='coral',alpha=.7,edgecolor='darkred',label='Individual')
ax2.plot(x,ssa.cumulative_contribution[:ns],'ko-',ms=4,label='Kumulatif')
ax2.axhline(95,color='r',ls='--',alpha=.5,label='95%'); ax2.axhline(99,color='g',ls='--',alpha=.5,label='99%')
ax2.set_title('Kontribusi Varians (%)',fontweight='bold'); ax2.set_xlabel('Komponen')
ax2.set_ylabel('%'); ax2.set_xticks(x); ax2.legend(fontsize=8)
st.pyplot(fig); plt.close()
with st.expander("📋 Tabel Eigenvalue"):
    ev_df = pd.DataFrame({'No':range(1,ns+1),'Singular Value':ssa.singular_values[:ns],
        'Eigenvalue':ssa.eigenvalues[:ns],'Kontribusi (%)':ssa.contribution[:ns],
        'Kumulatif (%)':ssa.cumulative_contribution[:ns]})
    st.dataframe(ev_df.style.format({'Singular Value':'{:.4f}','Eigenvalue':'{:.4f}',
        'Kontribusi (%)':'{:.4f}','Kumulatif (%)':'{:.4f}'}),use_container_width=True)

# ══════════════════════════════════════════════════════════════
# 3. EIGENVECTORS
# ══════════════════════════════════════════════════════════════
st.header("3️⃣ Eigenvectors")
ne = min(n_comp_ev, ssa.d)
ncols_ev = 4; nrows_ev = (ne+ncols_ev-1)//ncols_ev
fig,axes = plt.subplots(nrows_ev,ncols_ev,figsize=(14,3*nrows_ev))
axes_flat = axes.flatten() if ne>1 else [axes]
for i in range(len(axes_flat)):
    if i < ne:
        axes_flat[i].plot(ssa.U[:,i],'b-',lw=.8)
        axes_flat[i].set_title(f'EV{i+1} (σ={ssa.singular_values[i]:.1f}, {ssa.contribution[i]:.1f}%)',fontsize=8)
        axes_flat[i].axhline(0,color='k',lw=.3)
    else: axes_flat[i].set_visible(False)
plt.suptitle('Left Singular Vectors',fontweight='bold'); plt.tight_layout(); st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════════
# 4. PAIRED EIGENVECTORS
# ══════════════════════════════════════════════════════════════
st.header("4️⃣ Paired Eigenvectors")
np_ = min(n_comp_pair, ssa.d-1)
ncols_p = min(4, np_); nrows_p = (np_+ncols_p-1)//ncols_p
fig,axes = plt.subplots(nrows_p,ncols_p,figsize=(3.5*ncols_p,3.5*nrows_p))
axes_flat = np.array(axes).flatten() if np_>1 else [axes]
for idx in range(len(axes_flat)):
    if idx < np_:
        i,j = idx,idx+1
        axes_flat[idx].scatter(ssa.U[:,i],ssa.U[:,j],s=5,alpha=.6,c='steelblue')
        axes_flat[idx].set_xlabel(f'EV{i+1}'); axes_flat[idx].set_ylabel(f'EV{j+1}')
        axes_flat[idx].set_aspect('equal')
        axes_flat[idx].axhline(0,color='k',lw=.3); axes_flat[idx].axvline(0,color='k',lw=.3)
    else: axes_flat[idx].set_visible(False)
plt.suptitle('Paired EV — Lingkaran = Periodik',fontweight='bold',fontsize=10)
plt.tight_layout(); st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════════
# 5. KOMPONEN INDIVIDUAL
# ══════════════════════════════════════════════════════════════
st.header("5️⃣ Komponen Individual (Reconstructed)")
nr = min(n_comp_recon, ssa.d)
fig,axes = plt.subplots(nr,1,figsize=(13,2.2*nr),sharex=True)
if nr==1: axes=[axes]
for i in range(nr):
    rc = ssa.reconstruct_component(i)
    axes[i].plot(rc,'b-',lw=.8); axes[i].set_ylabel(f'F{i+1}',fontweight='bold')
    axes[i].axhline(0,color='k',lw=.3)
    axes[i].text(0.98,.8,f'σ={ssa.singular_values[i]:.2f} ({ssa.contribution[i]:.1f}%)',
                 transform=axes[i].transAxes,fontsize=8,ha='right',
                 bbox=dict(boxstyle='round',fc='wheat',alpha=.5))
axes[-1].set_xlabel('Waktu')
plt.suptitle('Komponen SSA',fontweight='bold',y=1.01); plt.tight_layout(); st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════════
# 6. PERIODOGRAM PER KOMPONEN
# ══════════════════════════════════════════════════════════════
st.header("6️⃣ Periodogram per Komponen")
npg = min(n_comp_pgram, ssa.d)
fig,axes = plt.subplots(npg,1,figsize=(12,2.2*npg),sharex=True)
if npg==1: axes=[axes]
for i in range(npg):
    rc = ssa.reconstruct_component(i)
    freqs,psd = periodogram(rc,fs=1.0)
    axes[i].plot(freqs,psd,'b-',lw=.8); axes[i].fill_between(freqs,psd,alpha=.3)
    axes[i].set_ylabel(f'F{i+1}',fontweight='bold')
    if len(freqs)>1:
        pk=freqs[np.argmax(psd[1:])+1]; T=1/pk if pk>0 else np.inf
        axes[i].text(0.97,.75,f'f={pk:.4f} T≈{T:.1f}',transform=axes[i].transAxes,fontsize=8,ha='right',
                     bbox=dict(boxstyle='round',fc='lightyellow',alpha=.8))
axes[-1].set_xlabel('Frequency')
plt.suptitle('Periodogram',fontweight='bold',y=1.01); plt.tight_layout(); st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════════
# 7. W-CORRELATION & GROUPING
# ══════════════════════════════════════════════════════════════
st.header("7️⃣ W-Correlation & Grouping")

nc = min(n_comp_wcorr, ssa.d)
wcorr = ssa.w_correlation(nc)

fig,ax = plt.subplots(figsize=(max(6,nc*0.5),max(5,nc*0.45)))
im = ax.imshow(np.abs(wcorr),cmap='RdBu_r',vmin=0,vmax=1,interpolation='nearest')
ax.set_xticks(range(nc)); ax.set_yticks(range(nc))
ax.set_xticklabels([f'F{i+1}' for i in range(nc)],fontsize=max(5,8-nc//10))
ax.set_yticklabels([f'F{i+1}' for i in range(nc)],fontsize=max(5,8-nc//10))
ax.set_title('W-Correlation Matrix |ρʷ|',fontweight='bold')
plt.colorbar(im,ax=ax,shrink=.8,label='|W-Correlation|')
if nc <= 20:
    for i in range(nc):
        for j in range(nc):
            v=abs(wcorr[i,j])
            ax.text(j,i,f'{v:.2f}',ha='center',va='center',fontsize=max(4,7-nc//8),
                    color='white' if v>.5 else 'black')
st.pyplot(fig); plt.close()
st.markdown("> **0** = well-separated, **1** = harus di-grup bersama")

# ── GROUPING ─────────────────────────────────────────────────
if "Hierarchical" in grouping_mode:
    hc_nc = min(hc_n_comp, ssa.d)
    groups = ssa.auto_group_wcorr(num_components=hc_nc, n_clusters=hc_n_clusters,
                                  linkage_method=hc_linkage)
    # Dendrogram
    if hasattr(ssa,'_hc_linkage'):
        st.subheader("Dendrogram Hierarchical Clustering")
        fig,ax = plt.subplots(figsize=(12,4))
        dendrogram(ssa._hc_linkage, labels=[f'F{i+1}' for i in range(hc_nc)],
                   leaf_rotation=90, leaf_font_size=8, ax=ax)
        ax.set_title(f'Dendrogram (linkage={hc_linkage}, clusters={hc_n_clusters})',fontweight='bold')
        ax.set_ylabel('Distance (1-|ρʷ|)')
        st.pyplot(fig); plt.close()

elif "Periodogram" in grouping_mode:
    groups = ssa.auto_group_periodogram(
        num_components=min(pg_n_comp, ssa.d),
        freq_threshold=pg_freq_thresh,
        pair_tolerance=pg_pair_tol
    )
else:
    st.subheader("Manual Grouping")
    st.markdown("Tentukan indeks komponen tiap grup (mulai dari 0). Pisah grup dengan `;`.")
    gn_input = st.text_input("Nama grup (koma):", "Trend, Seasonal_1, Seasonal_2, Noise")
    gi_input = st.text_input("Indeks per grup (`;`):", "0,1 ; 2,3 ; 4,5 ; 6,7,8,9,10,11")
    gnames = [g.strip() for g in gn_input.split(',')]
    gidxs = gi_input.split(';')
    groups = {}
    for gn,gi in zip(gnames,gidxs):
        try: groups[gn] = [int(x.strip()) for x in gi.split(',') if x.strip()]
        except: st.error(f"Format salah untuk '{gn}'"); st.stop()

st.subheader("Hasil Grouping")
gdf = pd.DataFrame([
    {'Grup':k,'Komponen':str(v),
     'Kontribusi (%)':sum(ssa.contribution[i] for i in v if i<len(ssa.contribution))}
    for k,v in groups.items()])
st.dataframe(gdf,use_container_width=True)

# ══════════════════════════════════════════════════════════════
# 8. REKONSTRUKSI
# ══════════════════════════════════════════════════════════════
st.header("8️⃣ Rekonstruksi Komponen")
ssa.reconstruct(groups)
ug = {k:v for k,v in ssa.reconstructed.items() if not k.startswith('_')}
ng = len(ug)
fig,axes = plt.subplots(ng+1,1,figsize=(13,2.8*(ng+1)),sharex=True)
axes[0].plot(ts,'b-',alpha=.5,lw=.8,label='Original')
axes[0].plot(ssa.reconstructed['_Total'],'r-',lw=1,label='Total Rekonstruksi')
axes[0].set_title('Original vs Rekonstruksi',fontweight='bold'); axes[0].legend()
colors = plt.cm.Set1(np.linspace(0,1,ng))
for idx,(gn,gv) in enumerate(ug.items()):
    axes[idx+1].plot(gv,color=colors[idx],lw=.8); axes[idx+1].set_ylabel(gn,fontweight='bold')
    axes[idx+1].axhline(0,color='k',lw=.3)
    sh = sum(ssa.contribution[i] for i in ssa.groups[gn] if i<len(ssa.contribution))
    axes[idx+1].text(0.98,.8,f'{sh:.1f}%',transform=axes[idx+1].transAxes,fontsize=10,ha='right',
                     fontweight='bold',bbox=dict(boxstyle='round',fc='lightyellow',alpha=.8))
axes[-1].set_xlabel('Waktu')
plt.suptitle('SSA Reconstruction',fontweight='bold',fontsize=13,y=1.01)
plt.tight_layout(); st.pyplot(fig); plt.close()
st.metric("RMSE Rekonstruksi Total",f"{np.sqrt(np.mean(ssa.reconstructed['_Residual']**2)):.6f}")

# ══════════════════════════════════════════════════════════════
# 9. MONTE CARLO SSA (opsional)
# ══════════════════════════════════════════════════════════════
if do_mc:
    st.header("9️⃣ Monte Carlo SSA Significance Test")
    with st.spinner(f"Monte Carlo ({mc_surr} surrogates)..."):
        mc = ssa.monte_carlo_test(int(mc_surr),float(mc_conf))
    nmc=len(mc['eigenvalues']); xmc=np.arange(1,nmc+1)
    fig,ax = plt.subplots(figsize=(11,5))
    ax.semilogy(xmc,mc['eigenvalues'],'ro-',ms=7,label='Data',zorder=5)
    ax.fill_between(xmc,mc['surrogate_lower'],mc['surrogate_upper'],alpha=.3,color='blue',
                    label=f'{mc["confidence"]*100:.0f}% CI Red Noise')
    ax.semilogy(xmc,mc['surrogate_median'],'b--',lw=1,label='Median Red Noise')
    sig=mc['significant']
    ax.semilogy(xmc[sig],mc['eigenvalues'][sig],'r*',ms=14,label='Signifikan',zorder=6)
    ax.semilogy(xmc[~sig],mc['eigenvalues'][~sig],'kx',ms=9,label='Tidak Signifikan',zorder=6)
    ax.set_title('Monte Carlo SSA',fontweight='bold'); ax.set_xlabel('Komponen')
    ax.set_ylabel('Eigenvalue (log)'); ax.set_xticks(xmc); ax.legend()
    st.pyplot(fig); plt.close()
    st.success(f"Signifikan: **{int(np.sum(sig))}** dari {nmc} → {list(np.where(sig)[0])}")

# ══════════════════════════════════════════════════════════════
# 10. FORECASTING & EVALUASI
# ══════════════════════════════════════════════════════════════
st.header("🔟 Forecasting & Evaluasi")
signal_groups = {k:v for k,v in groups.items() if 'noise' not in k.lower()}
if not signal_groups: signal_groups = groups

fc_r, fc_v = None, None
if do_r_forecast:
    with st.spinner("R-forecasting..."): fc_r = ssa.forecast_recurrent(signal_groups, int(steps_ahead))
if do_v_forecast:
    with st.spinner("V-forecasting..."): fc_v = ssa.forecast_vector(signal_groups, int(steps_ahead))

# ── PLOT FORECAST ──
if forecast_plot_mode.startswith("Gabung"):
    fig,ax = plt.subplots(figsize=(13,5))
    ax.plot(range(N),ts,'b-',lw=1,label='Actual',alpha=.7)
    if fc_r is not None: ax.plot(range(len(fc_r)),fc_r,'r--',lw=1.2,label='R-Forecast')
    if fc_v is not None: ax.plot(range(len(fc_v)),fc_v,'g--',lw=1.2,label='V-Forecast')
    ax.axvline(train_n,color='orange',ls='--',lw=1.5,label=f'Split ({train_n})')
    ax.axvline(N-1,color='gray',ls=':',lw=1,label=f'End (N={N})')
    ax.set_title('SSA Forecasting',fontweight='bold'); ax.set_xlabel('Waktu')
    ax.set_ylabel('Nilai'); ax.legend()
    st.pyplot(fig); plt.close()
else:
    for fc, lbl, clr in [(fc_r,'R-Forecast','red'),(fc_v,'V-Forecast','green')]:
        if fc is not None:
            fig,ax = plt.subplots(figsize=(13,4.5))
            ax.plot(range(N),ts,'b-',lw=1,label='Actual',alpha=.7)
            ax.plot(range(len(fc)),fc,'--',color=clr,lw=1.2,label=lbl)
            ax.axvline(train_n,color='orange',ls='--',lw=1.5,label=f'Split ({train_n})')
            ax.axvline(N-1,color='gray',ls=':',lw=1)
            ax.set_title(f'SSA {lbl}',fontweight='bold'); ax.set_xlabel('Waktu')
            ax.set_ylabel('Nilai'); ax.legend()
            st.pyplot(fig); plt.close()

# ── Evaluasi ──
def show_eval(label, ev):
    st.markdown(f"**{label}**")
    rows=[]
    for sp in ['train','test','overall']:
        d=ev[sp]
        rows.append({'Split':sp.upper(),'N':d['N'],'RMSE':d['RMSE'],'MAE':d['MAE'],
                     'MAPE (%)':d['MAPE_pct'],'sMAPE (%)':d['sMAPE_pct'],'R²':d['R2'],'NRMSE':d['NRMSE']})
    st.dataframe(pd.DataFrame(rows).style.format({
        'RMSE':'{:.4f}','MAE':'{:.4f}','MAPE (%)':'{:.4f}','sMAPE (%)':'{:.4f}',
        'R²':'{:.6f}','NRMSE':'{:.6f}'}),use_container_width=True)

if fc_r is not None: show_eval("📊 Evaluasi R-forecast",ssa.evaluate_split(train_pct/100,fc_r))
if fc_v is not None: show_eval("📊 Evaluasi V-forecast",ssa.evaluate_split(train_pct/100,fc_v))

# Actual vs Predicted & Error
for fc,lbl,clr in [(fc_r,'R-forecast','red'),(fc_v,'V-forecast','green')]:
    if fc is not None:
        errors = ts - fc[:N]
        fig,axes = plt.subplots(1,2,figsize=(13,4))
        axes[0].bar(range(train_n),errors[:train_n],color='green',alpha=.5,width=1,label='Train')
        axes[0].bar(range(train_n,N),errors[train_n:],color='orange',alpha=.5,width=1,label='Test')
        axes[0].axhline(0,color='k',lw=.5); axes[0].set_title(f'Error ({lbl})',fontweight='bold')
        axes[0].set_xlabel('Waktu'); axes[0].set_ylabel('Error'); axes[0].legend()
        axes[1].scatter(fc[:N],ts,s=8,alpha=.5,c='steelblue')
        mn_,mx_=min(ts.min(),fc[:N].min()),max(ts.max(),fc[:N].max())
        axes[1].plot([mn_,mx_],[mn_,mx_],'r--',lw=1)
        axes[1].set_title(f'Actual vs Predicted ({lbl})',fontweight='bold')
        axes[1].set_xlabel('Predicted'); axes[1].set_ylabel('Actual')
        plt.tight_layout(); st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════════
# 11. BOOTSTRAP CONFIDENCE & PREDICTION INTERVALS
# ══════════════════════════════════════════════════════════════
if do_bootstrap:
    st.header("1️⃣1️⃣ Bootstrap Confidence & Prediction Intervals")
    with st.spinner(f"Bootstrap ({boot_n} iterasi)... bisa 2-5 menit"):
        br = ssa.bootstrap_intervals(signal_groups, steps=int(steps_ahead),
                                     method=boot_method, n_bootstrap=int(boot_n),
                                     confidence=float(boot_conf))
    if br is None:
        st.error("Bootstrap gagal (terlalu banyak error). Coba kurangi L atau ganti metode.")
    else:
        h = np.arange(N, N+int(steps_ahead))
        fig,ax = plt.subplots(figsize=(13,5))
        ax.plot(range(N),ts,'b-',lw=1,label='Actual',alpha=.7)
        # In-sample
        if boot_method=='recurrent' and fc_r is not None:
            ax.plot(range(len(fc_r)),fc_r,'r--',lw=1,alpha=.5)
        elif boot_method=='vector' and fc_v is not None:
            ax.plot(range(len(fc_v)),fc_v,'g--',lw=1,alpha=.5)
        ax.plot(h,br['forecast_mean'],'k-',lw=1.5,label='Bootstrap Mean Forecast')
        ax.fill_between(h,br['ci_lower'],br['ci_upper'],alpha=.35,color='dodgerblue',
                        label=f'{br["confidence"]*100:.0f}% Confidence Interval')
        ax.fill_between(h,br['pi_lower'],br['pi_upper'],alpha=.15,color='orange',
                        label=f'{br["confidence"]*100:.0f}% Prediction Interval')
        ax.axvline(N-1,color='gray',ls=':',lw=1)
        ax.set_title(f'Bootstrap CI & PI ({boot_method}, {boot_n} samples)',fontweight='bold')
        ax.set_xlabel('Waktu'); ax.set_ylabel('Nilai'); ax.legend()
        st.pyplot(fig); plt.close()

        with st.expander("📋 Tabel Bootstrap Intervals"):
            bi_df = pd.DataFrame({
                'h':range(1,int(steps_ahead)+1),
                'Mean':br['forecast_mean'],
                f'CI Lower ({br["confidence"]*100:.0f}%)':br['ci_lower'],
                f'CI Upper ({br["confidence"]*100:.0f}%)':br['ci_upper'],
                f'PI Lower ({br["confidence"]*100:.0f}%)':br['pi_lower'],
                f'PI Upper ({br["confidence"]*100:.0f}%)':br['pi_upper'],
            })
            st.dataframe(bi_df.style.format({c:'{:.4f}' for c in bi_df.columns if c!='h'}),
                         use_container_width=True)

# ══════════════════════════════════════════════════════════════
# 12. RESIDUAL ANALYSIS
# ══════════════════════════════════════════════════════════════
st.header("1️⃣2️⃣ Analisis Residual")
res_info = ssa.residual_analysis()
residuals = ssa.residuals
if res_info:
    c1,c2,c3 = st.columns(3)
    c1.metric("Mean",f"{res_info['mean']:.6f}"); c2.metric("Std",f"{res_info['std']:.6f}")
    c3.metric("Skew",f"{res_info['skewness']:.4f}")
    st.markdown("**Uji Normalitas & White Noise**")
    rows=[]
    if 'shapiro_stat' in res_info:
        p=res_info['shapiro_p']
        rows.append({'Test':'Shapiro-Wilk','Statistic':f"{res_info['shapiro_stat']:.6f}",
                     'p-value':f"{p:.6f}",'Kesimpulan':'Normal ✅' if p>.05 else 'Tidak Normal ❌'})
    p_jb=res_info['jarque_bera_p']
    rows.append({'Test':'Jarque-Bera','Statistic':f"{res_info['jarque_bera_stat']:.6f}",
                 'p-value':f"{p_jb:.6f}",'Kesimpulan':'Normal ✅' if p_jb>.05 else 'Tidak Normal ❌'})
    if 'ljung_box_p' in res_info:
        p_lb=res_info['ljung_box_p']
        rows.append({'Test':'Ljung-Box','Statistic':f"{res_info['ljung_box_stat']:.6f}",
                     'p-value':f"{p_lb:.6f}",'Kesimpulan':'White Noise ✅' if p_lb>.05 else 'Bukan WN ❌'})
    st.dataframe(pd.DataFrame(rows),use_container_width=True)
    fig = plt.figure(figsize=(13,9))
    gs = gridspec.GridSpec(2,2,hspace=.35,wspace=.3)
    ax1=fig.add_subplot(gs[0,0]); ax1.plot(residuals,'b-',lw=.5); ax1.axhline(0,color='r',lw=.8)
    ax1.set_title('Residual Time Plot',fontweight='bold'); ax1.set_xlabel('Waktu')
    ax2=fig.add_subplot(gs[0,1])
    ax2.hist(residuals,bins='auto',density=True,alpha=.7,color='steelblue',edgecolor='navy')
    xr=np.linspace(residuals.min(),residuals.max(),100)
    ax2.plot(xr,norm.pdf(xr,np.mean(residuals),np.std(residuals)),'r-',lw=2,label='Normal fit')
    ax2.set_title('Histogram',fontweight='bold'); ax2.legend()
    ax3=fig.add_subplot(gs[1,0])
    plot_acf(residuals,ax=ax3,lags=min(40,len(residuals)//3),alpha=.05)
    ax3.set_title('ACF Residual',fontweight='bold')
    ax4=fig.add_subplot(gs[1,1]); probplot(residuals,dist='norm',plot=ax4)
    ax4.set_title('Q-Q Plot',fontweight='bold')
    plt.suptitle('Diagnostik Residual',fontsize=13,fontweight='bold',y=1.02)
    st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════════
# 13. DOWNLOAD
# ══════════════════════════════════════════════════════════════
st.header("1️⃣3️⃣ Download Hasil (Excel)")
tmp=os.path.join(tempfile.gettempdir(),'SSA_Results.xlsx')
with st.spinner("Menyiapkan Excel..."): ssa.save_results(tmp)
with open(tmp,'rb') as f: xb=f.read()
os.remove(tmp)
st.download_button("📥 Download SSA_Results.xlsx",xb,file_name="SSA_Results.xlsx",
                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.success("✅ Analisis SSA selesai! Ubah parameter di sidebar untuk eksplorasi.")
st.caption("SSA Analysis App v4.0 — Skripsi & Artikel Ilmiah")
