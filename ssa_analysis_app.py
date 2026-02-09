"""
══════════════════════════════════════════════════════════════
  APLIKASI SSA — Streamlit v4.3 (TABS + CACHE)
  Navigasi via TABS horizontal, bukan scroll ke bawah.
  SSA decomposition di-cache → ganti parameter tab tidak
  re-run decomposition.
══════════════════════════════════════════════════════════════
"""
import numpy as np, pandas as pd, streamlit as st, hashlib
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
plt.rcParams.update({'figure.dpi':150,'font.size':10,'font.family':'serif',
    'axes.grid':True,'grid.alpha':0.3,'lines.linewidth':1.3,'figure.autolayout':True})

# ── Cache SSA ────────────────────────────────────────────────
@st.cache_resource(show_spinner="⏳ Menjalankan SSA Decomposition...")
def run_ssa(ts_bytes, L_val, name):
    ts = np.frombuffer(ts_bytes, dtype=float)
    return SSA(ts, window_length=L_val, name=name)

@st.cache_data(show_spinner="🔍 Optimasi L...")
def run_optimal_L(ts_bytes, Lmin, Lmax, Lstep):
    ts = np.frombuffer(ts_bytes, dtype=float)
    return find_optimal_L(ts, L_min=Lmin, L_max=Lmax, L_step=Lstep)

# ══════════════════════════════════════════════════════════════
# SIDEBAR — Data + L (hanya parameter inti yg trigger re-decompose)
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
        opt_L_from = st.text_input("L dari:", "N/4")
        opt_L_to   = st.text_input("L sampai:", "N/2")
        opt_L_step = st.number_input("Step:", min_value=1, value=1, step=1)

    st.header("3️⃣ Train/Test")
    train_pct = st.slider("Training %", 50, 95, 80, 5)

# ══════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════
st.title("📈 Singular Spectrum Analysis (SSA) — v4.3")

ts = None; series_name = "Time Series"
if data_source == "Data Demo Sintetik":
    np.random.seed(42); ND=200; td=np.arange(ND)
    ts = 0.02*td+5+3*np.sin(2*np.pi*td/12)+1.5*np.sin(2*np.pi*td/6)+np.random.normal(0,.5,ND)
    series_name = "Demo (Trend+Seasonal12+Seasonal6+Noise)"
else:
    if uploaded_file is None:
        st.warning("⬆️ Upload file atau pilih Data Demo."); st.stop()
    df_raw = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
    st.dataframe(df_raw.head(5), use_container_width=True)
    num_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols: st.error("Tidak ada kolom numerik."); st.stop()
    target_col = st.selectbox("Kolom target:", num_cols)
    ts = df_raw[target_col].dropna().values.astype(float); series_name = target_col

N = len(ts); train_n = int(N*train_pct/100)
ts_bytes = ts.tobytes()

def parse_L_expr(expr, N):
    expr=expr.strip().upper().replace(' ','')
    if '/' in expr:
        p=expr.split('/'); num=N if p[0]=='N' else int(p[0]); return max(2,num//int(p[1]))
    try: return max(2,int(expr))
    except: return N//2

if L_mode == "Optimasi L":
    Lmin=parse_L_expr(opt_L_from,N); Lmax=parse_L_expr(opt_L_to,N)
    opt=run_optimal_L(ts_bytes, Lmin, Lmax, int(opt_L_step))
    if opt:
        L_val=opt['best_L']; st.success(f"✅ L optimal = **{L_val}** (RMSE={opt['best_RMSE']:.6f})")
    else: L_val=N//2
elif L_mode == "Manual": L_val=min(int(L_manual),N//2)
else: L_val=N//2

ssa = run_ssa(ts_bytes, L_val, series_name)

c1,c2,c3,c4 = st.columns(4)
c1.metric("N",N); c2.metric("L",ssa.L); c3.metric("K",ssa.K); c4.metric("d (rank)",ssa.d)

# ══════════════════════════════════════════════════════════════
# TABS — Navigasi horizontal
# ══════════════════════════════════════════════════════════════
tab_names = [
    "📊 Data",
    "📉 Scree",
    "🔢 Eigenvectors",
    "🔁 Paired EV",
    "📶 Komponen",
    "🎵 Periodogram",
    "🔲 W-Corr & Group",
    "🧩 Rekonstruksi",
    "🔮 Forecast",
    "📐 Koefisien",
    "📏 Bootstrap",
    "🧪 Residual",
    "🎲 Monte Carlo",
    "📥 Download",
]
tabs = st.tabs(tab_names)

# ──────────────────────────────────────────────────────────────
# TAB 0: DATA
# ──────────────────────────────────────────────────────────────
with tabs[0]:
    st.subheader("Data Original")
    fig,ax = plt.subplots(figsize=(12,3.5))
    ax.plot(ts,'b-',lw=1); ax.axvline(train_n,color='orange',ls='--',lw=1.5,label=f'Split ({train_n})')
    ax.set_title(series_name,fontweight='bold'); ax.set_xlabel('Waktu'); ax.set_ylabel('Nilai'); ax.legend()
    st.pyplot(fig); plt.close()
    st.dataframe(pd.DataFrame({'t':range(N),'Nilai':ts}).set_index('t').head(20),use_container_width=True)

# ──────────────────────────────────────────────────────────────
# TAB 1: SCREE
# ──────────────────────────────────────────────────────────────
with tabs[1]:
    st.subheader("Scree Plot & Kontribusi Varians")
    n_scree = st.slider("Komponen:",2,min(100,ssa.d),min(20,ssa.d),key='t1_scree')
    ns=min(n_scree,ssa.d); x=np.arange(1,ns+1)
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(14,4.5))
    ax1.bar(x,ssa.singular_values[:ns],color='steelblue',alpha=.7,edgecolor='navy')
    ax1.plot(x,ssa.singular_values[:ns],'ro-',ms=3); ax1.set_title('Singular Values',fontweight='bold')
    ax1.set_xlabel('Komponen'); ax1.set_ylabel('σ')
    if ns<=50: ax1.set_xticks(x)
    ax2.bar(x,ssa.contribution[:ns],color='coral',alpha=.7,edgecolor='darkred',label='Individual')
    ax2.plot(x,ssa.cumulative_contribution[:ns],'ko-',ms=3,label='Kumulatif')
    ax2.axhline(95,color='r',ls='--',alpha=.5,label='95%'); ax2.axhline(99,color='g',ls='--',alpha=.5,label='99%')
    ax2.set_title('Kontribusi (%)',fontweight='bold'); ax2.set_xlabel('Komponen'); ax2.set_ylabel('%')
    if ns<=50: ax2.set_xticks(x)
    ax2.legend(fontsize=8); st.pyplot(fig); plt.close()
    with st.expander("📋 Tabel"):
        st.dataframe(pd.DataFrame({'No':x,'σ':ssa.singular_values[:ns],'λ':ssa.eigenvalues[:ns],
            '%':ssa.contribution[:ns],'Cum%':ssa.cumulative_contribution[:ns]}).style.format(
            {'σ':'{:.4f}','λ':'{:.4f}','%':'{:.4f}','Cum%':'{:.4f}'}),use_container_width=True)

# ──────────────────────────────────────────────────────────────
# TAB 2: EIGENVECTORS
# ──────────────────────────────────────────────────────────────
with tabs[2]:
    st.subheader("Eigenvectors")
    n_ev = st.slider("Komponen:",2,min(30,ssa.d),min(8,ssa.d),key='t2_ev')
    ne=min(n_ev,ssa.d); ncols_e=4; nrows_e=max(1,(ne+ncols_e-1)//ncols_e)
    fig,axes=plt.subplots(nrows_e,ncols_e,figsize=(14,3*nrows_e))
    af=np.array(axes).flatten()
    for i in range(len(af)):
        if i<ne:
            af[i].plot(ssa.U[:,i],'b-',lw=.8)
            af[i].set_title(f'EV{i+1} ({ssa.contribution[i]:.1f}%)',fontsize=8)
            af[i].axhline(0,color='k',lw=.3)
        else: af[i].set_visible(False)
    plt.suptitle('Left Singular Vectors',fontweight='bold'); plt.tight_layout(); st.pyplot(fig); plt.close()

# ──────────────────────────────────────────────────────────────
# TAB 3: PAIRED EV
# ──────────────────────────────────────────────────────────────
with tabs[3]:
    st.subheader("Paired Eigenvectors")
    col_a,col_b = st.columns(2)
    n_pair = col_a.slider("Jumlah pair:",1,min(30,ssa.d-1),min(6,ssa.d-1),key='t3_pair')
    pair_style = col_b.radio("Gaya:",["Titik (scatter)","Garis+Titik (R-style)","Garis saja"],key='t3_style',horizontal=True)
    np_=min(n_pair,ssa.d-1); ncols_p=min(4,np_); nrows_p=max(1,(np_+ncols_p-1)//ncols_p)
    fig,axes=plt.subplots(nrows_p,ncols_p,figsize=(3.8*ncols_p,3.8*nrows_p))
    af=np.array(axes).flatten() if np_>1 else [axes]
    for idx in range(len(af)):
        if idx<np_:
            i,j=idx,idx+1; xi,yi=ssa.U[:,i],ssa.U[:,j]
            if "scatter" in pair_style.lower():
                af[idx].scatter(xi,yi,s=8,alpha=.6,c='steelblue',zorder=3)
            elif "R-style" in pair_style:
                af[idx].plot(xi,yi,'-',color='gray',lw=.5,alpha=.5,zorder=1)
                af[idx].scatter(xi,yi,s=10,c=np.arange(len(xi)),cmap='viridis',zorder=3,edgecolors='none',alpha=.8)
            else:
                af[idx].plot(xi,yi,'-',color='steelblue',lw=.7,alpha=.7)
            af[idx].set_xlabel(f'EV{i+1}'); af[idx].set_ylabel(f'EV{j+1}')
            af[idx].set_aspect('equal'); af[idx].axhline(0,color='k',lw=.3); af[idx].axvline(0,color='k',lw=.3)
        else: af[idx].set_visible(False)
    plt.suptitle('Paired EV — Lingkaran = Periodik',fontweight='bold',fontsize=10)
    plt.tight_layout(); st.pyplot(fig); plt.close()

# ──────────────────────────────────────────────────────────────
# TAB 4: KOMPONEN INDIVIDUAL
# ──────────────────────────────────────────────────────────────
with tabs[4]:
    st.subheader("Komponen Individual (Rekonstruksi per ET)")
    n_recon = st.slider("Komponen:",2,min(30,ssa.d),min(8,ssa.d),key='t4_recon')
    nr=min(n_recon,ssa.d)
    fig,axes=plt.subplots(nr,1,figsize=(13,2.2*nr),sharex=True)
    if nr==1: axes=[axes]
    for i in range(nr):
        rc=ssa.reconstruct_component(i); axes[i].plot(rc,'b-',lw=.8)
        axes[i].set_ylabel(f'F{i+1}',fontweight='bold'); axes[i].axhline(0,color='k',lw=.3)
        axes[i].text(0.98,.8,f'σ={ssa.singular_values[i]:.2f} ({ssa.contribution[i]:.1f}%)',
            transform=axes[i].transAxes,fontsize=8,ha='right',bbox=dict(boxstyle='round',fc='wheat',alpha=.5))
    axes[-1].set_xlabel('Waktu'); plt.tight_layout(); st.pyplot(fig); plt.close()

# ──────────────────────────────────────────────────────────────
# TAB 5: PERIODOGRAM
# ──────────────────────────────────────────────────────────────
with tabs[5]:
    st.subheader("Periodogram per Komponen")
    n_pgram = st.slider("Komponen:",2,min(30,ssa.d),min(8,ssa.d),key='t5_pg')
    npg=min(n_pgram,ssa.d)
    fig,axes=plt.subplots(npg,1,figsize=(12,2.2*npg),sharex=True)
    if npg==1: axes=[axes]
    for i in range(npg):
        rc=ssa.reconstruct_component(i); freqs,psd=periodogram(rc,fs=1.0)
        axes[i].plot(freqs,psd,'b-',lw=.8); axes[i].fill_between(freqs,psd,alpha=.3)
        axes[i].set_ylabel(f'F{i+1}',fontweight='bold')
        if len(freqs)>1:
            pk=freqs[np.argmax(psd[1:])+1]; T=1/pk if pk>0 else np.inf
            axes[i].text(0.97,.75,f'f={pk:.4f} T≈{T:.1f}',transform=axes[i].transAxes,fontsize=8,
                ha='right',bbox=dict(boxstyle='round',fc='lightyellow',alpha=.8))
    axes[-1].set_xlabel('Frequency'); plt.tight_layout(); st.pyplot(fig); plt.close()

# ──────────────────────────────────────────────────────────────
# TAB 6: W-CORR & GROUPING
# ──────────────────────────────────────────────────────────────
with tabs[6]:
    st.subheader("W-Correlation & Grouping")
    col_w1,col_w2 = st.columns([1,2])
    with col_w1:
        n_wcorr = st.slider("Komponen W-Corr:",4,min(100,ssa.d),min(12,ssa.d),key='t6_wc')
        grouping_mode = st.selectbox("Metode grouping:", [
            "Auto: Hierarchical Clustering (W-Corr)",
            "Auto: Periodogram-based","Manual Grouping"],key='t6_gm')
        if "Hierarchical" in grouping_mode:
            hc_link=st.selectbox("Linkage:",['average','single','complete','centroid','ward','weighted','median'],key='t6_lnk')
            hc_nc=st.number_input("Clusters:",2,20,3,1,key='t6_nc')
            hc_comp=st.slider("Komp clustering:",4,min(100,ssa.d),min(12,ssa.d),key='t6_hcc')
        elif "Periodogram" in grouping_mode:
            pg_comp=st.slider("Komp pgram:",4,min(100,ssa.d),min(12,ssa.d),key='t6_pgc')
            pg_ft=st.number_input("Freq thresh:",0.001,0.2,0.02,0.005,format="%.3f",key='t6_ft')
            pg_pt=st.number_input("Pair tol:",0.001,0.2,0.01,0.005,format="%.3f",key='t6_pt')
    with col_w2:
        nc=min(n_wcorr,ssa.d); wcorr=ssa.w_correlation(nc)
        fig,ax=plt.subplots(figsize=(max(5,nc*.45),max(4,nc*.4)))
        im=ax.imshow(np.abs(wcorr),cmap='hot_r',vmin=0,vmax=1,interpolation='nearest',origin='lower')
        ax.set_xticks(range(nc)); ax.set_yticks(range(nc))
        fs_t=max(4,8-nc//8)
        ax.set_xticklabels([f'F{i+1}' for i in range(nc)],fontsize=fs_t,rotation=90)
        ax.set_yticklabels([f'F{i+1}' for i in range(nc)],fontsize=fs_t)
        ax.set_title('|ρʷ|',fontweight='bold')
        plt.colorbar(im,ax=ax,shrink=.8)
        if nc<=20:
            for i in range(nc):
                for j in range(nc):
                    v=np.abs(wcorr[i,j])
                    ax.text(j,i,f'{v:.2f}',ha='center',va='center',fontsize=max(3,6-nc//6),
                            color='white' if v>.6 else 'black')
        st.pyplot(fig); plt.close()

    # Grouping
    if "Hierarchical" in grouping_mode:
        groups=ssa.auto_group_wcorr(min(hc_comp,ssa.d),hc_nc,hc_link)
        if hasattr(ssa,'_hc_linkage'):
            fig,ax=plt.subplots(figsize=(max(10,hc_comp*.5),4))
            dendrogram(ssa._hc_linkage,labels=[f'F{i+1}' for i in range(min(hc_comp,ssa.d))],
                       leaf_rotation=90,leaf_font_size=max(5,9-hc_comp//8),ax=ax)
            ax.set_title(f'Dendrogram ({hc_link}, k={hc_nc})',fontweight='bold')
            ax.set_ylabel('Dist'); st.pyplot(fig); plt.close()
    elif "Periodogram" in grouping_mode:
        groups=ssa.auto_group_periodogram(min(pg_comp,ssa.d),pg_ft,pg_pt)
    else:
        gn_in=st.text_input("Nama (koma):","Trend, Seasonal_1, Seasonal_2, Noise",key='t6_gn')
        gi_in=st.text_input("Indeks (`;`):","0,1 ; 2,3 ; 4,5 ; 6,7,8,9,10,11",key='t6_gi')
        gnames=[g.strip() for g in gn_in.split(',')]; gidxs=gi_in.split(';')
        groups={}
        for gn,gi in zip(gnames,gidxs):
            try: groups[gn]=[int(x.strip()) for x in gi.split(',') if x.strip()]
            except: st.error(f"Format salah '{gn}'"); st.stop()

    st.dataframe(pd.DataFrame([{'Grup':k,'Komponen':str(v),
        '%':sum(ssa.contribution[i] for i in v if i<len(ssa.contribution))}
        for k,v in groups.items()]),use_container_width=True)

    # Simpan groups ke session_state agar dipakai tab lain
    st.session_state['groups'] = groups

# ──────────────────────────────────────────────────────────────
# TAB 7: REKONSTRUKSI
# ──────────────────────────────────────────────────────────────
with tabs[7]:
    st.subheader("Rekonstruksi Komponen")
    groups = st.session_state.get('groups', None)
    if groups is None:
        st.info("⬅️ Tentukan grouping di tab **W-Corr & Group** dulu.")
    else:
        ssa.reconstruct(groups)
        ug={k:v for k,v in ssa.reconstructed.items() if not k.startswith('_')}; ng=len(ug)
        fig,axes=plt.subplots(ng+1,1,figsize=(13,2.8*(ng+1)),sharex=True)
        axes[0].plot(ts,'b-',alpha=.5,lw=.8,label='Original')
        axes[0].plot(ssa.reconstructed['_Total'],'r-',lw=1,label='Total'); axes[0].legend()
        axes[0].set_title('Original vs Rekonstruksi',fontweight='bold')
        colors=plt.cm.Set1(np.linspace(0,1,max(ng,1)))
        for idx,(gn,gv) in enumerate(ug.items()):
            axes[idx+1].plot(gv,color=colors[idx],lw=.8); axes[idx+1].set_ylabel(gn,fontweight='bold')
            axes[idx+1].axhline(0,color='k',lw=.3)
            sh=sum(ssa.contribution[i] for i in ssa.groups[gn] if i<len(ssa.contribution))
            axes[idx+1].text(0.98,.8,f'{sh:.1f}%',transform=axes[idx+1].transAxes,fontsize=10,
                ha='right',fontweight='bold',bbox=dict(boxstyle='round',fc='lightyellow',alpha=.8))
        axes[-1].set_xlabel('Waktu'); plt.tight_layout(); st.pyplot(fig); plt.close()
        st.metric("RMSE Rekonstruksi",f"{np.sqrt(np.mean(ssa.reconstructed['_Residual']**2)):.6f}")

# ──────────────────────────────────────────────────────────────
# TAB 8: FORECAST
# ──────────────────────────────────────────────────────────────
with tabs[8]:
    st.subheader("Forecasting & Evaluasi")
    groups = st.session_state.get('groups', None)
    if groups is None:
        st.info("⬅️ Tentukan grouping dulu."); st.stop()

    if not hasattr(ssa,'reconstructed') or '_Total' not in ssa.reconstructed:
        ssa.reconstruct(groups)

    col_f1,col_f2 = st.columns(2)
    steps_ahead = col_f1.number_input("Langkah:",1,500,24,1,key='t8_steps')
    fc_plot_mode = col_f2.radio("Tampilan:",["Gabung","Pisah"],horizontal=True,key='t8_pm')
    col_f3,col_f4 = st.columns(2)
    do_r = col_f3.checkbox("R-forecast",True,key='t8_r')
    do_v = col_f4.checkbox("V-forecast",True,key='t8_v')

    sig_grp={k:v for k,v in groups.items() if 'noise' not in k.lower()}
    if not sig_grp: sig_grp=groups

    fc_r,fc_v = None,None
    if do_r: fc_r=ssa.forecast_recurrent(sig_grp,int(steps_ahead))
    if do_v: fc_v=ssa.forecast_vector(sig_grp,int(steps_ahead))
    st.session_state['fc_r']=fc_r; st.session_state['fc_v']=fc_v
    st.session_state['sig_grp']=sig_grp; st.session_state['steps']=int(steps_ahead)

    if fc_plot_mode=="Gabung":
        fig,ax=plt.subplots(figsize=(13,5))
        ax.plot(range(N),ts,'b-',lw=1,label='Actual',alpha=.7)
        if fc_r is not None: ax.plot(range(len(fc_r)),fc_r,'r--',lw=1.2,label='R-Forecast')
        if fc_v is not None: ax.plot(range(len(fc_v)),fc_v,'g--',lw=1.2,label='V-Forecast')
        ax.axvline(train_n,color='orange',ls='--',lw=1.5,label=f'Split ({train_n})')
        ax.axvline(N-1,color='gray',ls=':',lw=1); ax.set_title('Forecasting',fontweight='bold')
        ax.set_xlabel('Waktu'); ax.set_ylabel('Nilai'); ax.legend(); st.pyplot(fig); plt.close()
    else:
        for fc,lbl,clr in [(fc_r,'R-Forecast','red'),(fc_v,'V-Forecast','green')]:
            if fc is not None:
                fig,ax=plt.subplots(figsize=(13,4))
                ax.plot(range(N),ts,'b-',lw=1,label='Actual',alpha=.7)
                ax.plot(range(len(fc)),fc,'--',color=clr,lw=1.2,label=lbl)
                ax.axvline(train_n,color='orange',ls='--',lw=1.5); ax.axvline(N-1,color='gray',ls=':',lw=1)
                ax.set_title(lbl,fontweight='bold'); ax.legend(); st.pyplot(fig); plt.close()

    # Evaluasi
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

    for fc,lbl in [(fc_r,'R'),(fc_v,'V')]:
        if fc is not None:
            errors=ts-fc[:N]
            fig,axes=plt.subplots(1,2,figsize=(13,4))
            axes[0].bar(range(train_n),errors[:train_n],color='green',alpha=.5,width=1,label='Train')
            axes[0].bar(range(train_n,N),errors[train_n:],color='orange',alpha=.5,width=1,label='Test')
            axes[0].axhline(0,color='k',lw=.5); axes[0].set_title(f'Error ({lbl})',fontweight='bold'); axes[0].legend()
            axes[1].scatter(fc[:N],ts,s=8,alpha=.5,c='steelblue')
            mn_,mx_=min(ts.min(),fc[:N].min()),max(ts.max(),fc[:N].max())
            axes[1].plot([mn_,mx_],[mn_,mx_],'r--',lw=1); axes[1].set_title(f'Actual vs Pred ({lbl})',fontweight='bold')
            axes[1].set_xlabel('Predicted'); axes[1].set_ylabel('Actual')
            plt.tight_layout(); st.pyplot(fig); plt.close()

# ──────────────────────────────────────────────────────────────
# TAB 9: KOEFISIEN FORECAST
# ──────────────────────────────────────────────────────────────
with tabs[9]:
    st.subheader("Parameter & Koefisien Forecasting")
    fc_r=st.session_state.get('fc_r'); fc_v=st.session_state.get('fc_v')

    if fc_r is not None and hasattr(ssa,'lrr_info'):
        info=ssa.lrr_info
        st.markdown(f"""
### 🔴 R-Forecast — Linear Recurrence Relation
**Formula:** $x_n = \\sum_{{j=1}}^{{L-1}} a_j \\cdot x_{{n-j}}$

| Parameter | Nilai |
|-----------|-------|
| Koefisien (L−1) | **{info['num_coefficients']}** |
| Eigentriple | **{info['num_eigentriples_used']}**: {info['eigentriple_indices']} |
| ν² | **{info['nu_squared']:.6f}** {'✅ valid' if info['nu_squared']<1 else '❌ tidak stabil'} |
        """)
        coef=ssa.lrr_coefficients; n_s=min(30,len(coef))
        st.dataframe(pd.DataFrame({'j':range(1,n_s+1),'a_j':coef[:n_s]}).style.format({'a_j':'{:.8f}'}),use_container_width=True)
        fig,ax=plt.subplots(figsize=(12,3))
        ax.bar(range(1,len(coef)+1),coef,color='steelblue',alpha=.7,width=1)
        ax.axhline(0,color='k',lw=.5); ax.set_title('LRR Coefficients',fontweight='bold')
        ax.set_xlabel('j'); ax.set_ylabel('a_j'); st.pyplot(fig); plt.close()

    if fc_v is not None and hasattr(ssa,'vforecast_info'):
        info_v=ssa.vforecast_info
        st.markdown(f"""
### 🟢 V-Forecast — Vector Coefficients
**Formula:** $\\hat{{z}}_L = P_\\pi^T \\cdot z[1:L-1]$

| Parameter | Nilai |
|-----------|-------|
| Koefisien (L−1) | **{info_v['num_coefficients']}** |
| Eigentriple | **{info_v['num_eigentriples_used']}**: {info_v['eigentriple_indices']} |
| ν² | **{info_v['nu_squared']:.6f}** |
        """)
        coef_v=ssa.vforecast_coefficients; n_sv=min(30,len(coef_v))
        st.dataframe(pd.DataFrame({'j':range(1,n_sv+1),'P_π':coef_v[:n_sv]}).style.format({'P_π':'{:.8f}'}),use_container_width=True)
        fig,ax=plt.subplots(figsize=(12,3))
        ax.bar(range(1,len(coef_v)+1),coef_v,color='forestgreen',alpha=.7,width=1)
        ax.axhline(0,color='k',lw=.5); ax.set_title('V-Forecast Coefficients P_π',fontweight='bold')
        ax.set_xlabel('j'); ax.set_ylabel('P_π'); st.pyplot(fig); plt.close()

    if fc_r is None and fc_v is None:
        st.info("⬅️ Jalankan forecast di tab **Forecast** dulu.")

# ──────────────────────────────────────────────────────────────
# TAB 10: BOOTSTRAP
# ──────────────────────────────────────────────────────────────
with tabs[10]:
    st.subheader("Bootstrap Confidence & Prediction Interval")
    sig_grp=st.session_state.get('sig_grp')
    if sig_grp is None:
        st.info("⬅️ Jalankan forecast dulu.")
    else:
        col_b1,col_b2,col_b3 = st.columns(3)
        boot_n=col_b1.number_input("Bootstrap:",100,2000,300,50,key='t10_bn')
        boot_conf=col_b2.slider("Confidence:",0.80,0.99,0.95,0.01,key='t10_bc')
        boot_meth=col_b3.selectbox("Metode:",["recurrent","vector"],key='t10_bm')
        steps=st.session_state.get('steps',24)

        if st.button("▶️ Jalankan Bootstrap",key='t10_run'):
            with st.spinner(f"Bootstrap ({boot_n}x)..."):
                br=ssa.bootstrap_intervals(sig_grp,steps,boot_meth,int(boot_n),float(boot_conf))
            if br is None:
                st.error("Bootstrap gagal.")
            else:
                st.session_state['bootstrap']=br
                st.success("Bootstrap selesai!")

        br=st.session_state.get('bootstrap')
        if br is not None:
            h=np.arange(N,N+steps)
            fig,ax=plt.subplots(figsize=(13,5))
            ax.plot(range(N),ts,'b-',lw=1,label='Actual',alpha=.7)
            ax.plot(h,br['forecast_mean'],'k-',lw=1.5,label='Mean')
            ax.fill_between(h,br['ci_lower'],br['ci_upper'],alpha=.35,color='dodgerblue',
                label=f'{br["confidence"]*100:.0f}% CI')
            ax.fill_between(h,br['pi_lower'],br['pi_upper'],alpha=.15,color='orange',
                label=f'{br["confidence"]*100:.0f}% PI')
            ax.axvline(N-1,color='gray',ls=':',lw=1)
            ax.set_title('Bootstrap CI & PI',fontweight='bold'); ax.legend(); st.pyplot(fig); plt.close()

            # Metrik
            test_n=N-train_n
            if steps<=test_n and test_n>0:
                st.markdown("**Metrik Evaluasi Interval (pada data test)**")
                with st.spinner("Evaluasi interval..."):
                    br_eval=ssa.bootstrap_intervals(sig_grp,test_n,boot_meth,int(boot_n),float(boot_conf))
                if br_eval is not None:
                    actual_test=ts[train_n:]
                    for iv,lo,up in [("CI",br_eval['ci_lower'][:test_n],br_eval['ci_upper'][:test_n]),
                                     ("PI",br_eval['pi_lower'][:test_n],br_eval['pi_upper'][:test_n])]:
                        m=SSA.evaluate_intervals(actual_test,lo,up,float(boot_conf))
                        st.markdown(f"**{iv}** (n={test_n})")
                        st.dataframe(pd.DataFrame([{k:v for k,v in m.items() if k!='N'}]).style.format(
                            {c:'{:.4f}' for c in ['PICP','PINAW','ACE','CWC','Winkler_Score','Mean_Width']}),
                            use_container_width=True)

            with st.expander("📋 Tabel"):
                bi_df=pd.DataFrame({'h':range(1,steps+1),'Mean':br['forecast_mean'],
                    'CI_Lo':br['ci_lower'],'CI_Up':br['ci_upper'],
                    'PI_Lo':br['pi_lower'],'PI_Up':br['pi_upper']})
                st.dataframe(bi_df.style.format({c:'{:.4f}' for c in bi_df.columns if c!='h'}),use_container_width=True)

# ──────────────────────────────────────────────────────────────
# TAB 11: RESIDUAL
# ──────────────────────────────────────────────────────────────
with tabs[11]:
    st.subheader("Analisis Residual")
    if not hasattr(ssa,'reconstructed') or '_Residual' not in ssa.reconstructed:
        groups=st.session_state.get('groups')
        if groups: ssa.reconstruct(groups)
        else: st.info("⬅️ Tentukan grouping dulu."); st.stop()

    res_info=ssa.residual_analysis(); residuals=ssa.residuals
    if res_info:
        c1,c2,c3,c4=st.columns(4)
        c1.metric("Mean",f"{res_info['mean']:.6f}"); c2.metric("Std",f"{res_info['std']:.6f}")
        c3.metric("Skew",f"{res_info['skewness']:.4f}"); c4.metric("Kurt",f"{res_info['kurtosis']:.4f}")
        rows=[]
        if 'shapiro_stat' in res_info:
            p=res_info['shapiro_p']
            rows.append({'Test':'Shapiro-Wilk','Stat':f"{res_info['shapiro_stat']:.6f}",
                'p':f"{p:.6f}",'Ket':'Normal ✅' if p>.05 else 'Tdk Normal ❌'})
        pjb=res_info['jarque_bera_p']
        rows.append({'Test':'Jarque-Bera','Stat':f"{res_info['jarque_bera_stat']:.6f}",
            'p':f"{pjb:.6f}",'Ket':'Normal ✅' if pjb>.05 else 'Tdk Normal ❌'})
        if 'ljung_box_p' in res_info:
            plb=res_info['ljung_box_p']
            rows.append({'Test':'Ljung-Box','Stat':f"{res_info['ljung_box_stat']:.6f}",
                'p':f"{plb:.6f}",'Ket':'WN ✅' if plb>.05 else 'Bukan WN ❌'})
        st.dataframe(pd.DataFrame(rows),use_container_width=True)
        fig=plt.figure(figsize=(13,9)); gs=gridspec.GridSpec(2,2,hspace=.35,wspace=.3)
        ax1=fig.add_subplot(gs[0,0]); ax1.plot(residuals,'b-',lw=.5); ax1.axhline(0,color='r',lw=.8)
        ax1.set_title('Residual',fontweight='bold')
        ax2=fig.add_subplot(gs[0,1])
        ax2.hist(residuals,bins='auto',density=True,alpha=.7,color='steelblue',edgecolor='navy')
        xr=np.linspace(residuals.min(),residuals.max(),100)
        ax2.plot(xr,norm.pdf(xr,np.mean(residuals),np.std(residuals)),'r-',lw=2); ax2.set_title('Histogram',fontweight='bold')
        ax3=fig.add_subplot(gs[1,0])
        plot_acf(residuals,ax=ax3,lags=min(40,len(residuals)//3),alpha=.05); ax3.set_title('ACF',fontweight='bold')
        ax4=fig.add_subplot(gs[1,1]); probplot(residuals,dist='norm',plot=ax4); ax4.set_title('Q-Q',fontweight='bold')
        st.pyplot(fig); plt.close()

# ──────────────────────────────────────────────────────────────
# TAB 12: MONTE CARLO
# ──────────────────────────────────────────────────────────────
with tabs[12]:
    st.subheader("Monte Carlo SSA Significance Test")
    col_m1,col_m2 = st.columns(2)
    mc_surr=col_m1.number_input("Surrogates:",100,2000,500,100,key='t12_ms')
    mc_conf=col_m2.slider("Confidence:",0.90,0.99,0.95,0.01,key='t12_mc')
    if st.button("▶️ Jalankan Monte Carlo",key='t12_run'):
        with st.spinner(f"Monte Carlo ({mc_surr})..."):
            mc=ssa.monte_carlo_test(int(mc_surr),float(mc_conf))
        st.session_state['mc']=mc
    mc=st.session_state.get('mc')
    if mc is not None:
        nmc=len(mc['eigenvalues']); xmc=np.arange(1,nmc+1)
        fig,ax=plt.subplots(figsize=(11,5))
        ax.semilogy(xmc,mc['eigenvalues'],'ro-',ms=7,label='Data',zorder=5)
        ax.fill_between(xmc,mc['surrogate_lower'],mc['surrogate_upper'],alpha=.3,color='blue',
            label=f'{mc["confidence"]*100:.0f}% CI')
        ax.semilogy(xmc,mc['surrogate_median'],'b--',lw=1,label='Median')
        sig=mc['significant']
        ax.semilogy(xmc[sig],mc['eigenvalues'][sig],'r*',ms=14,label='Sig',zorder=6)
        ax.semilogy(xmc[~sig],mc['eigenvalues'][~sig],'kx',ms=9,label='Not Sig',zorder=6)
        ax.set_title('Monte Carlo SSA',fontweight='bold'); ax.set_xticks(xmc); ax.legend()
        st.pyplot(fig); plt.close()
        st.success(f"Signifikan: **{int(np.sum(sig))}** dari {nmc}")

# ──────────────────────────────────────────────────────────────
# TAB 13: DOWNLOAD
# ──────────────────────────────────────────────────────────────
with tabs[13]:
    st.subheader("Download Hasil (Excel)")
    if not hasattr(ssa,'reconstructed'):
        groups=st.session_state.get('groups')
        if groups: ssa.reconstruct(groups)
    tmp=os.path.join(tempfile.gettempdir(),'SSA_Results.xlsx')
    ssa.save_results(tmp)
    with open(tmp,'rb') as f: xb=f.read()
    os.remove(tmp)
    st.download_button("📥 SSA_Results.xlsx",xb,file_name="SSA_Results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    st.success("✅ File siap diunduh. Semua koefisien, forecast, dan interval tersimpan di Excel.")
