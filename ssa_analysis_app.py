"""
══════════════════════════════════════════════════════════════
  APLIKASI SSA — Streamlit v4.1 (Revisi Lengkap)
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

st.set_page_config(page_title="SSA Time Series 1D", layout="wide", page_icon="📈")
plt.rcParams.update({'figure.dpi':150,'font.size':10,'font.family':'serif',
    'axes.grid':True,'grid.alpha':0.3,'lines.linewidth':1.3,'figure.autolayout':True})

# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("⚙️ Pengaturan SSA")

    st.header("1️⃣ Data")
    data_source = st.radio("Sumber data:", ["Upload CSV/Excel","Data Demo Sintetik"])
    uploaded_file = None
    if data_source == "Upload CSV/Excel":
        uploaded_file = st.file_uploader("Upload file", type=["csv","xlsx","xls"])

    st.header("2️⃣ Window Length (L)")
    L_mode = st.radio("Mode:", ["Auto (N/2)","Manual","Optimasi L"])
    L_manual = 48
    if L_mode == "Manual":
        L_manual = st.number_input("Nilai L:", min_value=2, value=48, step=1)
    if L_mode == "Optimasi L":
        st.markdown("**Range pencarian L** (untuk separabilitas)")
        opt_L_from = st.text_input("L dari (misal N/4, N/3, atau angka):", "N/4")
        opt_L_to   = st.text_input("L sampai (misal N/2 atau angka):", "N/2")
        opt_L_step = st.number_input("Step:", min_value=1, value=1, step=1)

    st.header("3️⃣ Jumlah Komponen Plot")
    n_scree  = st.slider("Scree plot",       2, 100, 20, key='scree')
    n_ev     = st.slider("Eigenvectors",      2, 30,  8, key='ev')
    n_pair   = st.slider("Paired EV",         1, 30,  6, key='pair')
    n_pgram  = st.slider("Periodogram",       2, 30,  8, key='pgram')
    n_wcorr  = st.slider("W-Correlation",     4, 100, 12, key='wcorr')
    n_recon  = st.slider("Komponen individual",2, 30,  8, key='recon')

    st.header("4️⃣ Grouping")
    grouping_mode = st.selectbox("Metode:", [
        "Auto: Hierarchical Clustering (W-Corr)",
        "Auto: Periodogram-based",
        "Manual Grouping"])
    if "Hierarchical" in grouping_mode:
        hc_link = st.selectbox("Linkage:", ['average','single','complete','centroid','ward','weighted','median'])
        hc_nc   = st.number_input("Jumlah cluster:", 2, 20, 3, 1)
        hc_comp = st.slider("Komponen clustering:", 4, 100, 12, key='hcc')
    elif "Periodogram" in grouping_mode:
        pg_comp = st.slider("Komponen pgram:", 4, 100, 12, key='pgc')
        pg_ft   = st.number_input("Freq threshold:", 0.001, 0.2, 0.02, 0.005, format="%.3f")
        pg_pt   = st.number_input("Pair tolerance:", 0.001, 0.2, 0.01, 0.005, format="%.3f")

    st.header("5️⃣ Forecasting")
    steps_ahead = st.number_input("Langkah forecast:", 1, 500, 24, 1)
    do_r = st.checkbox("Recurrent (R-forecast)", True)
    do_v = st.checkbox("Vector (V-forecast)", True)
    fc_plot_mode = st.radio("Tampilan forecast:", ["Gabung","Pisah (per metode)"])

    st.header("6️⃣ Train/Test Split")
    train_pct = st.slider("Training %", 50, 95, 80, 5)

    st.header("7️⃣ Bootstrap CI & PI")
    do_boot = st.checkbox("Hitung Bootstrap CI & PI", False)
    if do_boot:
        boot_n    = st.number_input("Jumlah bootstrap:", 100, 2000, 300, 50)
        boot_conf = st.slider("Confidence:", 0.80, 0.99, 0.95, 0.01, key='bconf')
        boot_meth = st.selectbox("Metode bootstrap:", ["recurrent","vector"])

    st.header("8️⃣ Monte Carlo SSA")
    do_mc = st.checkbox("Jalankan Monte Carlo SSA", False)
    if do_mc:
        mc_surr = st.number_input("Surrogate:", 100, 2000, 500, 100)
        mc_conf = st.slider("Confidence (MC):", 0.90, 0.99, 0.95, 0.01, key='mcconf')

# ══════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════
st.title("📈 Singular Spectrum Analysis (SSA) — v4.1")
st.caption("Dekomposisi · W-Corr Clustering · Periodogram Grouping · Rekonstruksi · "
           "R & V Forecast · Bootstrap CI/PI + Metrik Interval · Evaluasi · Residual · Monte Carlo")

ts = None; series_name = "Time Series"
if data_source == "Data Demo Sintetik":
    np.random.seed(42); ND=200; td=np.arange(ND)
    ts = 0.02*td+5+3*np.sin(2*np.pi*td/12)+1.5*np.sin(2*np.pi*td/6)+np.random.normal(0,.5,ND)
    series_name = "Demo (Trend+Seasonal12+Seasonal6+Noise)"
    st.info("🔹 Data demo sintetik (N=200)")
else:
    if uploaded_file is None:
        st.warning("⬆️ Upload file di sidebar atau pilih Data Demo."); st.stop()
    df_raw = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
    st.subheader("Preview Data"); st.dataframe(df_raw.head(10), use_container_width=True)
    num_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols: st.error("Tidak ada kolom numerik."); st.stop()
    target_col = st.selectbox("Pilih kolom target:", num_cols)
    ts = df_raw[target_col].dropna().values.astype(float); series_name = target_col

N = len(ts); train_n = int(N*train_pct/100)

# ── Parse L ──
def parse_L_expr(expr, N):
    expr = expr.strip().upper().replace(' ','')
    if '/' in expr:
        parts = expr.split('/')
        num = N if parts[0]=='N' else int(parts[0])
        den = int(parts[1])
        return max(2, num//den)
    try: return max(2, int(expr))
    except: return N//2

if L_mode == "Optimasi L":
    Lmin = parse_L_expr(opt_L_from, N); Lmax = parse_L_expr(opt_L_to, N)
    with st.spinner(f"🔍 Mencari L optimal (range {Lmin}–{Lmax}, step={opt_L_step})..."):
        opt = find_optimal_L(ts, L_min=Lmin, L_max=Lmax, L_step=int(opt_L_step))
    if opt:
        L_val = opt['best_L']
        st.success(f"✅ L optimal = **{L_val}** (RMSE={opt['best_RMSE']:.6f})")
        with st.expander("📋 Semua hasil optimasi L"):
            st.dataframe(opt['all_results'], use_container_width=True)
    else:
        L_val = N//2; st.warning("Optimasi gagal, fallback L=N/2")
elif L_mode == "Manual":
    L_val = min(int(L_manual), N//2)
else:
    L_val = N//2

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
ax.plot(ts,'b-',lw=1); ax.axvline(train_n,color='orange',ls='--',lw=1.5,label=f'Split ({train_n})')
ax.set_title(f'{series_name}',fontweight='bold'); ax.set_xlabel('Waktu'); ax.set_ylabel('Nilai'); ax.legend()
st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════════
# 2. SCREE PLOT
# ══════════════════════════════════════════════════════════════
st.header("2️⃣ Scree Plot & Kontribusi Varians")
ns = min(n_scree, ssa.d); x = np.arange(1,ns+1)
fig,(ax1,ax2) = plt.subplots(1,2,figsize=(14,4.5))
ax1.bar(x,ssa.singular_values[:ns],color='steelblue',alpha=.7,edgecolor='navy')
ax1.plot(x,ssa.singular_values[:ns],'ro-',ms=3); ax1.set_title('Singular Values',fontweight='bold')
ax1.set_xlabel('Komponen'); ax1.set_ylabel('σ')
if ns <= 50: ax1.set_xticks(x)
ax2.bar(x,ssa.contribution[:ns],color='coral',alpha=.7,edgecolor='darkred',label='Individual')
ax2.plot(x,ssa.cumulative_contribution[:ns],'ko-',ms=3,label='Kumulatif')
ax2.axhline(95,color='r',ls='--',alpha=.5,label='95%'); ax2.axhline(99,color='g',ls='--',alpha=.5,label='99%')
ax2.set_title('Kontribusi Varians (%)',fontweight='bold'); ax2.set_xlabel('Komponen'); ax2.set_ylabel('%')
if ns <= 50: ax2.set_xticks(x)
ax2.legend(fontsize=8); st.pyplot(fig); plt.close()
with st.expander("📋 Tabel Eigenvalue"):
    st.dataframe(pd.DataFrame({'No':x,'σ':ssa.singular_values[:ns],'λ':ssa.eigenvalues[:ns],
        '%':ssa.contribution[:ns],'Cum%':ssa.cumulative_contribution[:ns]}).style.format(
        {'σ':'{:.4f}','λ':'{:.4f}','%':'{:.4f}','Cum%':'{:.4f}'}),use_container_width=True)

# ══════════════════════════════════════════════════════════════
# 3. EIGENVECTORS
# ══════════════════════════════════════════════════════════════
st.header("3️⃣ Eigenvectors")
ne = min(n_ev, ssa.d); ncols_e = 4; nrows_e = max(1,(ne+ncols_e-1)//ncols_e)
fig,axes = plt.subplots(nrows_e,ncols_e,figsize=(14,3*nrows_e))
af = np.array(axes).flatten()
for i in range(len(af)):
    if i < ne:
        af[i].plot(ssa.U[:,i],'b-',lw=.8)
        af[i].set_title(f'EV{i+1} (σ={ssa.singular_values[i]:.1f}, {ssa.contribution[i]:.1f}%)',fontsize=8)
        af[i].axhline(0,color='k',lw=.3)
    else: af[i].set_visible(False)
plt.suptitle('Left Singular Vectors',fontweight='bold'); plt.tight_layout(); st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════════
# 4. PAIRED EIGENVECTORS
# ══════════════════════════════════════════════════════════════
st.header("4️⃣ Paired Eigenvectors")
np_ = min(n_pair, ssa.d-1); ncols_p = min(4,np_); nrows_p = max(1,(np_+ncols_p-1)//ncols_p)
fig,axes = plt.subplots(nrows_p,ncols_p,figsize=(3.5*ncols_p,3.5*nrows_p))
af = np.array(axes).flatten() if np_>1 else [axes]
for idx in range(len(af)):
    if idx < np_:
        i,j = idx,idx+1
        af[idx].scatter(ssa.U[:,i],ssa.U[:,j],s=5,alpha=.6,c='steelblue')
        af[idx].set_xlabel(f'EV{i+1}'); af[idx].set_ylabel(f'EV{j+1}')
        af[idx].set_aspect('equal'); af[idx].axhline(0,color='k',lw=.3); af[idx].axvline(0,color='k',lw=.3)
    else: af[idx].set_visible(False)
plt.suptitle('Paired EV — Lingkaran = Periodik',fontweight='bold',fontsize=10)
plt.tight_layout(); st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════════
# 5. KOMPONEN INDIVIDUAL
# ══════════════════════════════════════════════════════════════
st.header("5️⃣ Komponen Individual")
nr = min(n_recon, ssa.d)
fig,axes = plt.subplots(nr,1,figsize=(13,2.2*nr),sharex=True)
if nr==1: axes=[axes]
for i in range(nr):
    rc = ssa.reconstruct_component(i); axes[i].plot(rc,'b-',lw=.8)
    axes[i].set_ylabel(f'F{i+1}',fontweight='bold'); axes[i].axhline(0,color='k',lw=.3)
    axes[i].text(0.98,.8,f'σ={ssa.singular_values[i]:.2f} ({ssa.contribution[i]:.1f}%)',
        transform=axes[i].transAxes,fontsize=8,ha='right',bbox=dict(boxstyle='round',fc='wheat',alpha=.5))
axes[-1].set_xlabel('Waktu')
plt.suptitle('Komponen SSA',fontweight='bold',y=1.01); plt.tight_layout(); st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════════
# 6. PERIODOGRAM
# ══════════════════════════════════════════════════════════════
st.header("6️⃣ Periodogram per Komponen")
npg = min(n_pgram, ssa.d)
fig,axes = plt.subplots(npg,1,figsize=(12,2.2*npg),sharex=True)
if npg==1: axes=[axes]
for i in range(npg):
    rc=ssa.reconstruct_component(i); freqs,psd=periodogram(rc,fs=1.0)
    axes[i].plot(freqs,psd,'b-',lw=.8); axes[i].fill_between(freqs,psd,alpha=.3)
    axes[i].set_ylabel(f'F{i+1}',fontweight='bold')
    if len(freqs)>1:
        pk=freqs[np.argmax(psd[1:])+1]; T=1/pk if pk>0 else np.inf
        axes[i].text(0.97,.75,f'f={pk:.4f} T≈{T:.1f}',transform=axes[i].transAxes,fontsize=8,
            ha='right',bbox=dict(boxstyle='round',fc='lightyellow',alpha=.8))
axes[-1].set_xlabel('Frequency')
plt.suptitle('Periodogram',fontweight='bold',y=1.01); plt.tight_layout(); st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════════
# 7. W-CORRELATION & GROUPING
# ══════════════════════════════════════════════════════════════
st.header("7️⃣ W-Correlation & Grouping")
nc = min(n_wcorr, ssa.d); wcorr = ssa.w_correlation(nc)

# ── PLOT W-CORR: origin='lower' → kiri bawah ke kanan atas ──
fig,ax = plt.subplots(figsize=(max(6,nc*.5+1),max(5,nc*.45+1)))
wcorr_abs = np.abs(wcorr)
im = ax.imshow(wcorr_abs, cmap='hot_r', vmin=0, vmax=1, interpolation='nearest',
               origin='lower')
ax.set_xticks(range(nc)); ax.set_yticks(range(nc))
fs_tick = max(5, 9 - nc//8)
ax.set_xticklabels([f'F{i+1}' for i in range(nc)], fontsize=fs_tick)
ax.set_yticklabels([f'F{i+1}' for i in range(nc)], fontsize=fs_tick)
ax.set_title('W-Correlation Matrix |ρʷ|', fontweight='bold')
plt.colorbar(im, ax=ax, shrink=.8, label='|W-Correlation|')
if nc <= 25:
    for i in range(nc):
        for j in range(nc):
            v = wcorr_abs[i,j]
            ax.text(j,i,f'{v:.2f}',ha='center',va='center',
                    fontsize=max(4,7-nc//6), color='white' if v>.6 else 'black')
st.pyplot(fig); plt.close()
st.markdown("> **Baca:** Sumbu dimulai dari kiri-bawah (F1). Warna gelap = korelasi tinggi → harus digrup.")

# ── GROUPING ──
if "Hierarchical" in grouping_mode:
    hc_nc_ = min(hc_comp, ssa.d)
    groups = ssa.auto_group_wcorr(num_components=hc_nc_, n_clusters=hc_nc, linkage_method=hc_link)
    if hasattr(ssa,'_hc_linkage'):
        st.subheader("Dendrogram")
        fig,ax = plt.subplots(figsize=(max(10,hc_nc_*.5),4))
        dendrogram(ssa._hc_linkage, labels=[f'F{i+1}' for i in range(hc_nc_)],
                   leaf_rotation=90, leaf_font_size=max(5,9-hc_nc_//8), ax=ax)
        ax.set_title(f'Dendrogram ({hc_link}, k={hc_nc})',fontweight='bold')
        ax.set_ylabel('Distance (1−|ρʷ|)'); st.pyplot(fig); plt.close()
elif "Periodogram" in grouping_mode:
    groups = ssa.auto_group_periodogram(min(pg_comp,ssa.d), pg_ft, pg_pt)
else:
    st.subheader("Manual Grouping")
    st.markdown("Indeks mulai 0. Pisah grup dengan `;`")
    gn_in = st.text_input("Nama grup (koma):", "Trend, Seasonal_1, Seasonal_2, Noise")
    gi_in = st.text_input("Indeks per grup (`;`):", "0,1 ; 2,3 ; 4,5 ; 6,7,8,9,10,11")
    gnames = [g.strip() for g in gn_in.split(',')]; gidxs = gi_in.split(';')
    groups = {}
    for gn,gi in zip(gnames,gidxs):
        try: groups[gn]=[int(x.strip()) for x in gi.split(',') if x.strip()]
        except: st.error(f"Format salah '{gn}'"); st.stop()

st.subheader("Hasil Grouping")
st.dataframe(pd.DataFrame([{'Grup':k,'Komponen':str(v),
    'Kontribusi (%)':sum(ssa.contribution[i] for i in v if i<len(ssa.contribution))}
    for k,v in groups.items()]),use_container_width=True)

# ══════════════════════════════════════════════════════════════
# 8. REKONSTRUKSI
# ══════════════════════════════════════════════════════════════
st.header("8️⃣ Rekonstruksi")
ssa.reconstruct(groups)
ug = {k:v for k,v in ssa.reconstructed.items() if not k.startswith('_')}; ng=len(ug)
fig,axes = plt.subplots(ng+1,1,figsize=(13,2.8*(ng+1)),sharex=True)
axes[0].plot(ts,'b-',alpha=.5,lw=.8,label='Original')
axes[0].plot(ssa.reconstructed['_Total'],'r-',lw=1,label='Total'); axes[0].legend()
axes[0].set_title('Original vs Rekonstruksi',fontweight='bold')
colors = plt.cm.Set1(np.linspace(0,1,max(ng,1)))
for idx,(gn,gv) in enumerate(ug.items()):
    axes[idx+1].plot(gv,color=colors[idx],lw=.8); axes[idx+1].set_ylabel(gn,fontweight='bold')
    axes[idx+1].axhline(0,color='k',lw=.3)
    sh=sum(ssa.contribution[i] for i in ssa.groups[gn] if i<len(ssa.contribution))
    axes[idx+1].text(0.98,.8,f'{sh:.1f}%',transform=axes[idx+1].transAxes,fontsize=10,
        ha='right',fontweight='bold',bbox=dict(boxstyle='round',fc='lightyellow',alpha=.8))
axes[-1].set_xlabel('Waktu')
plt.suptitle('SSA Reconstruction',fontweight='bold',fontsize=13,y=1.01)
plt.tight_layout(); st.pyplot(fig); plt.close()
st.metric("RMSE Rekonstruksi",f"{np.sqrt(np.mean(ssa.reconstructed['_Residual']**2)):.6f}")

# ══════════════════════════════════════════════════════════════
# 9. MONTE CARLO
# ══════════════════════════════════════════════════════════════
if do_mc:
    st.header("9️⃣ Monte Carlo SSA")
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
sig_grp = {k:v for k,v in groups.items() if 'noise' not in k.lower()}
if not sig_grp: sig_grp = groups

fc_r,fc_v = None,None
if do_r:
    with st.spinner("R-forecasting..."): fc_r = ssa.forecast_recurrent(sig_grp, int(steps_ahead))
if do_v:
    with st.spinner("V-forecasting..."): fc_v = ssa.forecast_vector(sig_grp, int(steps_ahead))

if fc_plot_mode == "Gabung":
    fig,ax = plt.subplots(figsize=(13,5))
    ax.plot(range(N),ts,'b-',lw=1,label='Actual',alpha=.7)
    if fc_r is not None: ax.plot(range(len(fc_r)),fc_r,'r--',lw=1.2,label='R-Forecast')
    if fc_v is not None: ax.plot(range(len(fc_v)),fc_v,'g--',lw=1.2,label='V-Forecast')
    ax.axvline(train_n,color='orange',ls='--',lw=1.5,label=f'Split ({train_n})')
    ax.axvline(N-1,color='gray',ls=':',lw=1); ax.set_title('Forecasting',fontweight='bold')
    ax.set_xlabel('Waktu'); ax.set_ylabel('Nilai'); ax.legend(); st.pyplot(fig); plt.close()
else:
    for fc,lbl,clr in [(fc_r,'R-Forecast','red'),(fc_v,'V-Forecast','green')]:
        if fc is not None:
            fig,ax = plt.subplots(figsize=(13,4.5))
            ax.plot(range(N),ts,'b-',lw=1,label='Actual',alpha=.7)
            ax.plot(range(len(fc)),fc,'--',color=clr,lw=1.2,label=lbl)
            ax.axvline(train_n,color='orange',ls='--',lw=1.5); ax.axvline(N-1,color='gray',ls=':',lw=1)
            ax.set_title(f'{lbl}',fontweight='bold'); ax.set_xlabel('Waktu')
            ax.set_ylabel('Nilai'); ax.legend(); st.pyplot(fig); plt.close()

# ── Evaluasi ──
def show_eval(label, ev):
    st.markdown(f"**{label}**")
    rows=[]
    for sp in ['train','test','overall']:
        d=ev[sp]
        rows.append({'Split':sp.upper(),'N':d['N'],'RMSE':d['RMSE'],'MAE':d['MAE'],
            'MAPE%':d['MAPE_pct'],'sMAPE%':d['sMAPE_pct'],'R²':d['R2'],'NRMSE':d['NRMSE']})
    st.dataframe(pd.DataFrame(rows).style.format({
        'RMSE':'{:.4f}','MAE':'{:.4f}','MAPE%':'{:.4f}','sMAPE%':'{:.4f}',
        'R²':'{:.6f}','NRMSE':'{:.6f}'}),use_container_width=True)

if fc_r is not None: show_eval("📊 R-forecast",ssa.evaluate_split(train_pct/100,fc_r))
if fc_v is not None: show_eval("📊 V-forecast",ssa.evaluate_split(train_pct/100,fc_v))

for fc,lbl in [(fc_r,'R-forecast'),(fc_v,'V-forecast')]:
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
# 11. BOOTSTRAP CI & PI + METRIK INTERVAL
# ══════════════════════════════════════════════════════════════
if do_boot:
    st.header("1️⃣1️⃣ Bootstrap CI & PI")
    with st.spinner(f"Bootstrap ({boot_n}x)..."):
        br = ssa.bootstrap_intervals(sig_grp, int(steps_ahead), boot_meth, int(boot_n), float(boot_conf))
    if br is None:
        st.error("Bootstrap gagal. Coba kurangi L atau ganti metode.")
    else:
        h = np.arange(N, N+int(steps_ahead))
        fig,ax = plt.subplots(figsize=(13,5))
        ax.plot(range(N),ts,'b-',lw=1,label='Actual',alpha=.7)
        if boot_meth=='recurrent' and fc_r is not None:
            ax.plot(range(len(fc_r)),fc_r,'r--',lw=1,alpha=.5)
        elif boot_meth=='vector' and fc_v is not None:
            ax.plot(range(len(fc_v)),fc_v,'g--',lw=1,alpha=.5)
        ax.plot(h,br['forecast_mean'],'k-',lw=1.5,label='Bootstrap Mean')
        ax.fill_between(h,br['ci_lower'],br['ci_upper'],alpha=.35,color='dodgerblue',
            label=f'{br["confidence"]*100:.0f}% CI')
        ax.fill_between(h,br['pi_lower'],br['pi_upper'],alpha=.15,color='orange',
            label=f'{br["confidence"]*100:.0f}% PI')
        ax.axvline(N-1,color='gray',ls=':',lw=1)
        ax.set_title(f'Bootstrap CI & PI ({boot_meth}, n={boot_n})',fontweight='bold')
        ax.set_xlabel('Waktu'); ax.set_ylabel('Nilai'); ax.legend(); st.pyplot(fig); plt.close()

        # ── Metrik evaluasi interval (jika ada data aktual di horizon) ──
        st.subheader("Metrik Evaluasi Prediction Interval")
        st.markdown("""
        | Metrik | Deskripsi |
        |--------|-----------|
        | **PICP** | *Coverage Probability* — % data aktual di dalam interval. Idealnya ≥ nominal coverage |
        | **PINAW** | *Normalized Average Width* — lebar interval relatif. Semakin kecil semakin baik |
        | **ACE** | *Average Coverage Error* = PICP − nominal. Idealnya mendekati 0 |
        | **CWC** | *Coverage Width Criterion* — menghukum interval lebar jika coverage kurang |
        | **Winkler** | *Winkler/Interval Score* — penalti jika aktual di luar interval. Semakin kecil semakin baik |
        """)

        # Evaluasi pada in-sample (data yang diketahui)
        # Gunakan bootstrap pada in-sample untuk demonstrasi
        # Jika steps <= test_size, bisa evaluasi langsung
        test_n = N - train_n
        if int(steps_ahead) <= test_n and test_n > 0:
            # Re-bootstrap hanya untuk horizon = test period
            with st.spinner("Evaluasi interval pada data testing..."):
                br_eval = ssa.bootstrap_intervals(sig_grp, steps=test_n,
                    method=boot_meth, n_bootstrap=int(boot_n), confidence=float(boot_conf))
            if br_eval is not None:
                actual_test = ts[train_n:]
                for iv_name, lo, up in [
                    ("Confidence Interval", br_eval['ci_lower'][:test_n], br_eval['ci_upper'][:test_n]),
                    ("Prediction Interval", br_eval['pi_lower'][:test_n], br_eval['pi_upper'][:test_n])]:
                    metrics = SSA.evaluate_intervals(actual_test, lo, up, float(boot_conf))
                    st.markdown(f"**{iv_name}** (pada {test_n} data testing)")
                    mdf = pd.DataFrame([{
                        'PICP':metrics['PICP'], 'PINAW':metrics['PINAW'],
                        'ACE':metrics['ACE'], 'CWC':metrics['CWC'],
                        'Winkler Score':metrics['Winkler_Score'],
                        'Mean Width':metrics['Mean_Width'],
                        'Nominal':metrics['Nominal_Coverage']}])
                    st.dataframe(mdf.style.format({
                        'PICP':'{:.4f}','PINAW':'{:.4f}','ACE':'{:.4f}',
                        'CWC':'{:.4f}','Winkler Score':'{:.4f}',
                        'Mean Width':'{:.4f}','Nominal':'{:.2f}'}),use_container_width=True)
        else:
            st.info(f"Metrik interval dihitung jika langkah forecast ≤ jumlah data test ({test_n}). "
                    f"Saat ini steps={steps_ahead}.")

        with st.expander("📋 Tabel Bootstrap Intervals"):
            bi_df = pd.DataFrame({'h':range(1,int(steps_ahead)+1),
                'Mean':br['forecast_mean'],
                f'CI_Lo':br['ci_lower'],f'CI_Up':br['ci_upper'],
                f'PI_Lo':br['pi_lower'],f'PI_Up':br['pi_upper']})
            st.dataframe(bi_df.style.format({c:'{:.4f}' for c in bi_df.columns if c!='h'}),
                use_container_width=True)

# ══════════════════════════════════════════════════════════════
# 12. RESIDUAL ANALYSIS
# ══════════════════════════════════════════════════════════════
st.header("1️⃣2️⃣ Analisis Residual")
res_info = ssa.residual_analysis(); residuals = ssa.residuals
if res_info:
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Mean",f"{res_info['mean']:.6f}"); c2.metric("Std",f"{res_info['std']:.6f}")
    c3.metric("Skew",f"{res_info['skewness']:.4f}"); c4.metric("Kurtosis",f"{res_info['kurtosis']:.4f}")
    rows=[]
    if 'shapiro_stat' in res_info:
        p=res_info['shapiro_p']
        rows.append({'Test':'Shapiro-Wilk','Statistic':f"{res_info['shapiro_stat']:.6f}",
            'p-value':f"{p:.6f}",'Kesimpulan':'Normal ✅' if p>.05 else 'Tidak Normal ❌'})
    pjb=res_info['jarque_bera_p']
    rows.append({'Test':'Jarque-Bera','Statistic':f"{res_info['jarque_bera_stat']:.6f}",
        'p-value':f"{pjb:.6f}",'Kesimpulan':'Normal ✅' if pjb>.05 else 'Tidak Normal ❌'})
    if 'ljung_box_p' in res_info:
        plb=res_info['ljung_box_p']
        rows.append({'Test':'Ljung-Box','Statistic':f"{res_info['ljung_box_stat']:.6f}",
            'p-value':f"{plb:.6f}",'Kesimpulan':'White Noise ✅' if plb>.05 else 'Bukan WN ❌'})
    st.dataframe(pd.DataFrame(rows),use_container_width=True)
    fig = plt.figure(figsize=(13,9)); gs=gridspec.GridSpec(2,2,hspace=.35,wspace=.3)
    ax1=fig.add_subplot(gs[0,0]); ax1.plot(residuals,'b-',lw=.5); ax1.axhline(0,color='r',lw=.8)
    ax1.set_title('Residual Time Plot',fontweight='bold'); ax1.set_xlabel('Waktu')
    ax2=fig.add_subplot(gs[0,1])
    ax2.hist(residuals,bins='auto',density=True,alpha=.7,color='steelblue',edgecolor='navy')
    xr=np.linspace(residuals.min(),residuals.max(),100)
    ax2.plot(xr,norm.pdf(xr,np.mean(residuals),np.std(residuals)),'r-',lw=2,label='Normal')
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
st.header("1️⃣3️⃣ Download Hasil")
tmp=os.path.join(tempfile.gettempdir(),'SSA_Results.xlsx')
with st.spinner("Excel..."): ssa.save_results(tmp)
with open(tmp,'rb') as f: xb=f.read()
os.remove(tmp)
st.download_button("📥 Download SSA_Results.xlsx",xb,file_name="SSA_Results.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
st.success("✅ Analisis SSA selesai!"); st.caption("SSA App v4.1 — Skripsi & Artikel Ilmiah")
