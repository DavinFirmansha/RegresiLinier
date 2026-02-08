# ============================================================
# APLIKASI ANALISIS GWPR (Geographically Weighted Poisson Regression)
# File: gwpr_app.py
# Jalankan: streamlit run gwpr_app.py
# ============================================================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import io
import json
from datetime import datetime
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Poisson as smPoisson

warnings.filterwarnings('ignore')

try:
    import folium
    from streamlit_folium import st_folium
    import branca.colormap as cm
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False

try:
    import geopandas as gpd
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False

st.set_page_config(page_title="GWPR Analysis Suite", page_icon="\U0001f9a0", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 700; color: #1B4F72 !important; text-align: center; padding: 1rem 0; border-bottom: 3px solid #8E44AD; margin-bottom: 1rem; }
    .sub-header { font-size: 1.2rem; color: #5D6D7E !important; text-align: center; margin-bottom: 2rem; }
    .metric-card { background: linear-gradient(135deg, #8E44AD 0%, #3498DB 100%); padding: 1rem; border-radius: 10px; color: white !important; text-align: center; margin: 0.5rem 0; }
    .success-box { background-color: #D5F5E3; border-left: 5px solid #27AE60; padding: 1rem; border-radius: 5px; margin: 0.5rem 0; color: #1a5c2e !important; }
    .success-box b, .success-box strong, .success-box span { color: #1a5c2e !important; }
    .warning-box { background-color: #FEF9E7; border-left: 5px solid #F39C12; padding: 1rem; border-radius: 5px; margin: 0.5rem 0; color: #7d5a00 !important; }
    .warning-box b, .warning-box strong, .warning-box span { color: #7d5a00 !important; }
    .error-box { background-color: #FADBD8; border-left: 5px solid #E74C3C; padding: 1rem; border-radius: 5px; margin: 0.5rem 0; color: #922b21 !important; }
    .error-box b, .error-box strong, .error-box span { color: #922b21 !important; }
    .info-box { background-color: #D6EAF8; border-left: 5px solid #2E86C1; padding: 1rem; border-radius: 5px; margin: 0.5rem 0; color: #1a4971 !important; }
    .info-box b, .info-box strong, .info-box span { color: #1a4971 !important; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { padding: 10px 20px; border-radius: 5px 5px 0 0; }
</style>
""", unsafe_allow_html=True)

defaults = {
    'data': None, 'poisson_results': None, 'gwpr_results': None,
    'coords': None, 'y': None, 'X': None, 'var_names': None,
    'dep_var': None, 'indep_vars': [], 'lon_col': None, 'lat_col': None,
    'gwpr_bw': None, 'gwpr_selector': None, 'offset_col': None,
    'poisson_aicc': None, 'poisson_deviance': None
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

def create_colormap_safe(series, cmap_name='RdYlBu', n_bins=7):
    vmin, vmax = float(series.min()), float(series.max())
    if vmin == vmax:
        vmin -= 1; vmax += 1
    rgba = plt.get_cmap(cmap_name)(np.linspace(0, 1, n_bins))
    hex_colors = [mcolors.to_hex(c) for c in rgba]
    return cm.LinearColormap(colors=hex_colors, vmin=vmin, vmax=vmax)

def compute_poisson_deviance(y, mu):
    y = np.asarray(y, dtype=float).flatten()
    mu = np.maximum(np.asarray(mu, dtype=float).flatten(), 1e-10)
    dev = np.zeros_like(y)
    mask = y > 0
    dev[mask] = y[mask] * np.log(y[mask] / mu[mask])
    return 2 * np.sum(dev - (y - mu))

def compute_poisson_pseudo_r2(y, mu):
    from scipy.special import gammaln
    y = np.asarray(y, dtype=float).flatten()
    mu = np.maximum(np.asarray(mu, dtype=float).flatten(), 1e-10)
    y_safe = np.maximum(y, 0)
    ll_model = np.sum(y_safe * np.log(mu) - mu - gammaln(y_safe + 1))
    mu_null = max(np.mean(y_safe), 1e-10)
    ll_null = np.sum(y_safe * np.log(mu_null) - mu_null - gammaln(y_safe + 1))
    return 1 - ll_model / ll_null if ll_null != 0 else 0.0

def generate_demo_data(scenario="default"):
    np.random.seed(42)
    if scenario == "Kasus DBD (Dengue Fever)":
        n = 150
        lon = np.random.uniform(110.3, 114.6, n); lat = np.random.uniform(-8.2, -6.8, n)
        kepadatan = np.random.uniform(500, 20000, n); puskesmas = np.random.uniform(0.5, 10, n)
        sanitasi = np.random.uniform(20, 95, n)
        b0 = 2.5 + 0.5*np.sin((lon-112)*2) + np.random.normal(0,0.1,n)
        b1 = 0.00005 + 0.00003*(lat+7.5) + np.random.normal(0,0.00001,n)
        b2 = -0.08 + 0.03*np.cos(lon*2) + np.random.normal(0,0.01,n)
        b3 = -0.005 + 0.003*(lon-112) + np.random.normal(0,0.001,n)
        log_mu = np.clip(b0 + b1*kepadatan + b2*puskesmas + b3*sanitasi, 0, 6)
        y = np.random.poisson(np.exp(log_mu))
        df = pd.DataFrame({'ID':[f'Kec_{i+1}' for i in range(n)],'Longitude':np.round(lon,4),'Latitude':np.round(lat,4),'Kasus_DBD':y,'Kepadatan_km2':np.round(kepadatan,0),'Rasio_Puskesmas':np.round(puskesmas,2),'Pct_Sanitasi':np.round(sanitasi,1)})
        return df,"Kasus_DBD",["Kepadatan_km2","Rasio_Puskesmas","Pct_Sanitasi"],"Longitude","Latitude"
    elif scenario == "Kematian Bayi (Infant Mortality)":
        n = 120
        lon = np.random.uniform(106.6,107.1,n); lat = np.random.uniform(-6.9,-6.1,n)
        bidan = np.random.uniform(1,20,n); miskin = np.random.uniform(5,40,n); imun = np.random.uniform(50,99,n)
        b0 = 1.5+0.3*(lon-106.85)*5+np.random.normal(0,0.1,n)
        b1 = -0.05+0.02*(lat+6.5)+np.random.normal(0,0.005,n)
        b2 = 0.02+0.01*np.sin(lon*30)+np.random.normal(0,0.003,n)
        b3 = -0.01+0.005*(lon-106.85)+np.random.normal(0,0.002,n)
        log_mu = np.clip(b0+b1*bidan+b2*miskin+b3*imun,0,5)
        y = np.random.poisson(np.exp(log_mu))
        df = pd.DataFrame({'ID':[f'Kel_{i+1}' for i in range(n)],'Longitude':np.round(lon,6),'Latitude':np.round(lat,6),'Kematian_Bayi':y,'Jumlah_Bidan':np.round(bidan,1),'Pct_Miskin':np.round(miskin,2),'Pct_Imunisasi':np.round(imun,1)})
        return df,"Kematian_Bayi",["Jumlah_Bidan","Pct_Miskin","Pct_Imunisasi"],"Longitude","Latitude"
    elif scenario == "Kecelakaan Lalu Lintas":
        n = 180
        lon = np.random.uniform(106.6,107.0,n); lat = np.random.uniform(-6.95,-6.1,n)
        pj = np.random.uniform(5,200,n); kk = np.random.uniform(100,5000,n); tl = np.random.uniform(0,30,n)
        b0 = 2.0+0.4*(lat+6.5)+np.random.normal(0,0.1,n)
        b1 = 0.005+0.002*np.sin(lon*30)+np.random.normal(0,0.001,n)
        b2 = 0.0002+0.0001*(lon-106.8)*10+np.random.normal(0,0.00005,n)
        b3 = -0.03+0.01*(lat+6.5)+np.random.normal(0,0.005,n)
        log_mu = np.clip(b0+b1*pj+b2*kk+b3*tl,0,6)
        y = np.random.poisson(np.exp(log_mu))
        df = pd.DataFrame({'ID':[f'Ruas_{i+1}' for i in range(n)],'Longitude':np.round(lon,6),'Latitude':np.round(lat,6),'Jml_Kecelakaan':y,'Panjang_Jalan_km':np.round(pj,1),'Kepadatan_Kend':np.round(kk,0),'Jml_Traffic_Light':np.round(tl,0)})
        return df,"Jml_Kecelakaan",["Panjang_Jalan_km","Kepadatan_Kend","Jml_Traffic_Light"],"Longitude","Latitude"
    else:
        n = 100
        lon = np.random.uniform(0,10,n); lat = np.random.uniform(0,10,n)
        x1 = np.random.uniform(1,50,n); x2 = np.random.uniform(0,30,n)
        b0 = 1.5+0.2*lon+np.random.normal(0,0.1,n)
        b1 = 0.02+0.01*lat+np.random.normal(0,0.005,n)
        b2 = -0.01+0.005*lon+np.random.normal(0,0.003,n)
        log_mu = np.clip(b0+b1*x1+b2*x2,0,6)
        y = np.random.poisson(np.exp(log_mu))
        df = pd.DataFrame({'ID':[f'Loc_{i+1}' for i in range(n)],'Longitude':np.round(lon,4),'Latitude':np.round(lat,4),'Y_Count':y,'X1':np.round(x1,2),'X2':np.round(x2,2)})
        return df,"Y_Count",["X1","X2"],"Longitude","Latitude"

# ============================================================
# HEADER & SIDEBAR
# ============================================================
st.markdown('<div class="main-header">\U0001f9a0 GWPR Analysis Suite</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Geographically Weighted Poisson Regression \u2014 Analisis Data Count Spasial</div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## \u2699\ufe0f Konfigurasi")
    data_source = st.radio("\U0001f4c2 Sumber Data:", ["Upload CSV/Excel", "Data Demo"], key="data_src")
    if data_source == "Data Demo":
        scenario = st.selectbox("Pilih Skenario Demo:", ["Kasus DBD (Dengue Fever)","Kematian Bayi (Infant Mortality)","Kecelakaan Lalu Lintas","Default (Simulated)"])
        demo_df, demo_dep, demo_indeps, demo_lon, demo_lat = generate_demo_data(scenario)
        st.session_state.data = demo_df
        st.success(f"\u2705 Data demo: {len(demo_df)} observasi")
    else:
        uploaded = st.file_uploader("Upload CSV/Excel:", type=['csv','xlsx','xls'])
        if uploaded:
            try:
                if uploaded.name.endswith('.csv'):
                    sep = st.selectbox("Separator:", [",",";","\\t","|"])
                    if sep == "\\t": sep = "\t"
                    df_up = pd.read_csv(uploaded, sep=sep)
                else:
                    df_up = pd.read_excel(uploaded)
                st.session_state.data = df_up
                st.success(f"\u2705 {len(df_up)} baris dimuat")
            except Exception as e:
                st.error(f"Error: {e}")

    df = st.session_state.data
    if df is not None:
        st.markdown("---")
        st.markdown("### \U0001f3af Pilih Variabel")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if data_source == "Data Demo":
            d_dep, d_indeps, d_lon, d_lat = demo_dep, demo_indeps, demo_lon, demo_lat
        else:
            d_dep, d_indeps, d_lon, d_lat = None, [], None, None

        dep_var = st.selectbox("Variabel Dependen (Y count):", numeric_cols, index=numeric_cols.index(d_dep) if d_dep in numeric_cols else 0)
        st.session_state.dep_var = dep_var
        if dep_var:
            yv = df[dep_var].dropna()
            if np.all(yv == yv.astype(int)) and np.all(yv >= 0):
                st.success(f"\u2705 Count data (range: {int(yv.min())}\u2013{int(yv.max())})")
            else:
                st.warning("\u26a0\ufe0f Y sebaiknya bilangan bulat \u2265 0")

        remaining = [c for c in numeric_cols if c != dep_var]
        d_idx = [remaining.index(v) for v in d_indeps if v in remaining] if d_indeps else []
        indep_vars = st.multiselect("Variabel Independen (X):", remaining, default=[remaining[i] for i in d_idx])
        st.session_state.indep_vars = indep_vars
        lon_col = st.selectbox("Longitude:", numeric_cols, index=numeric_cols.index(d_lon) if d_lon in numeric_cols else 0)
        lat_col = st.selectbox("Latitude:", numeric_cols, index=numeric_cols.index(d_lat) if d_lat in numeric_cols else 0)
        st.session_state.lon_col = lon_col; st.session_state.lat_col = lat_col

        offset_opts = ["Tidak ada (None)"] + numeric_cols
        offset_sel = st.selectbox("Offset / Exposure (opsional):", offset_opts, help="log(populasi) sebagai offset")
        st.session_state.offset_col = None if offset_sel == "Tidak ada (None)" else offset_sel

        st.markdown("---")
        st.markdown("### \U0001f527 Pengaturan GWPR")
        kernel_type = st.selectbox("Kernel:", ["bisquare","gaussian","exponential"], key="kernel")
        bw_fixed = st.checkbox("Fixed bandwidth", value=False, key="bwfixed")
        criterion = st.selectbox("Criterion:", ["AICc","AIC","BIC","CV"], key="criterion")
        search_method = st.selectbox("Search:", ["golden_section","interval"], key="search")
        st.session_state.kernel_type = kernel_type; st.session_state.bw_fixed = bw_fixed
        st.session_state.criterion = criterion; st.session_state.search_method = search_method

if st.session_state.data is None:
    st.markdown("""<div class="info-box"><b>\U0001f9a0 Selamat datang di GWPR Analysis Suite!</b><br><br>
    Aplikasi ini melakukan analisis <b>Geographically Weighted Poisson Regression (GWPR)</b>
    untuk data count/diskrit dengan variasi spasial.<br><br>
    <b>Fitur:</b> Data Explorer, Uji Asumsi Poisson, Poisson GLM, GWPR, Perbandingan Model, Peta, Laporan & Export.
    </div>""", unsafe_allow_html=True)
    st.info("\U0001f448 Mulai dengan memilih sumber data di sidebar!")
    st.stop()

df = st.session_state.data
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["\U0001f4ca Data Explorer","\U0001f52c Uji Asumsi","\U0001f4c8 Poisson GLM","\U0001f30d GWPR Model","\u2696\ufe0f Perbandingan","\U0001f5fa\ufe0f Peta","\U0001f4cb Laporan"])

# ============================================================
# TAB 1: DATA EXPLORER
# ============================================================
with tab1:
    st.markdown("## \U0001f4ca Data Explorer")
    c1,c2,c3,c4 = st.columns(4)
    with c1: st.metric("Observasi", len(df))
    with c2: st.metric("Variabel", len(df.columns))
    with c3: st.metric("Missing", df.isnull().sum().sum())
    with c4: st.metric("Duplikat", df.duplicated().sum())

    st.markdown("### Preview Data")
    st.dataframe(df.head(20), use_container_width=True)
    st.markdown("### Statistik Deskriptif")
    st.dataframe(df.describe().round(4), use_container_width=True)

    if st.session_state.dep_var:
        dep = st.session_state.dep_var
        st.markdown(f"### \U0001f522 Validasi Count: `{dep}`")
        yv = df[dep].dropna()
        c1,c2,c3,c4,c5 = st.columns(5)
        with c1: st.metric("Mean", f"{yv.mean():.2f}")
        with c2: st.metric("Variance", f"{yv.var():.2f}")
        with c3: st.metric("Min", f"{int(yv.min())}")
        with c4: st.metric("Max", f"{int(yv.max())}")
        disp_ratio = yv.var() / yv.mean() if yv.mean() > 0 else 0
        with c5: st.metric("Var/Mean", f"{disp_ratio:.2f}")

        c1,c2 = st.columns(2)
        with c1:
            fig_h = px.histogram(df, x=dep, nbins=30, title=f"Distribusi {dep}", marginal="box", color_discrete_sequence=["#8E44AD"])
            st.plotly_chart(fig_h, use_container_width=True)
        with c2:
            from scipy.stats import poisson
            xr = np.arange(0, int(yv.max())+1)
            pmf = poisson.pmf(xr, yv.mean())
            of = yv.value_counts().sort_index(); ofn = of / of.sum()
            fig_c = go.Figure()
            fig_c.add_trace(go.Bar(x=ofn.index, y=ofn.values, name="Observed", marker_color="#8E44AD", opacity=0.7))
            fig_c.add_trace(go.Scatter(x=xr, y=pmf, mode='lines+markers', name="Poisson", line=dict(color='red',width=2)))
            fig_c.update_layout(title=f"Observed vs Poisson (\u03bb={yv.mean():.2f})", xaxis_title=dep, yaxis_title="Probability")
            st.plotly_chart(fig_c, use_container_width=True)

        if disp_ratio > 1.5:
            st.markdown(f'<div class="warning-box">\u26a0\ufe0f <b>Overdispersi!</b> Var/Mean = {disp_ratio:.2f} > 1</div>', unsafe_allow_html=True)
        elif disp_ratio < 0.5:
            st.markdown(f'<div class="info-box">\u2139\ufe0f <b>Underdispersi.</b> Var/Mean = {disp_ratio:.2f}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="success-box">\u2705 <b>Dispersi wajar.</b> Var/Mean = {disp_ratio:.2f}</div>', unsafe_allow_html=True)

    if st.session_state.lon_col and st.session_state.lat_col:
        st.markdown("### \U0001f5fa\ufe0f Peta Lokasi")
        fig_m = px.scatter_mapbox(df, lon=st.session_state.lon_col, lat=st.session_state.lat_col, color=st.session_state.dep_var, size_max=15, zoom=6, mapbox_style="open-street-map", color_continuous_scale="Viridis")
        fig_m.update_layout(height=500)
        st.plotly_chart(fig_m, use_container_width=True)

    st.markdown("### \U0001f517 Korelasi")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cv = [st.session_state.dep_var]+st.session_state.indep_vars if st.session_state.dep_var else numeric_cols
    cv = [v for v in cv if v in df.columns]
    cm_fig = px.imshow(df[cv].corr(), text_auto=".3f", color_continuous_scale="RdBu_r", aspect="auto", zmin=-1, zmax=1)
    cm_fig.update_layout(height=500)
    st.plotly_chart(cm_fig, use_container_width=True)

# ============================================================
# TAB 2: UJI ASUMSI POISSON
# ============================================================
with tab2:
    st.markdown("## \U0001f52c Uji Asumsi Poisson")
    st.markdown('''<div class="info-box"><b>Asumsi regresi Poisson:</b> (1) Y = count \u2265 0, (2) Mean = Var (equidispersion), (3) Independen, (4) log(\u03bc) = X\u03b2. GWPR merelaksasi stasioneritas spasial.</div>''', unsafe_allow_html=True)

    if st.session_state.dep_var and st.session_state.indep_vars:
        dep = st.session_state.dep_var; indeps = st.session_state.indep_vars
        y_data = df[dep].values; X_data = df[indeps].values

        st.markdown("### 1\ufe0f\u20e3 Uji Distribusi Poisson")
        from scipy.stats import chisquare, poisson, kstest
        of = pd.Series(y_data).value_counts().sort_index()
        lam = y_data.mean()
        ef = np.maximum(np.array([poisson.pmf(k, lam)*len(y_data) for k in of.index]), 1e-10)

        c1,c2,c3 = st.columns(3)
        with c1:
            st.markdown("**Chi-Square GoF**")
            try:
                chi2_stat, chi2_p = chisquare(of.values, f_exp=ef)
                st.write(f"Stat: `{chi2_stat:.4f}`, P: `{chi2_p:.6f}`")
                if chi2_p > 0.05: st.markdown('<div class="success-box">\u2705 Sesuai Poisson</div>', unsafe_allow_html=True)
                else: st.markdown('<div class="warning-box">\u26a0\ufe0f Tidak sesuai Poisson</div>', unsafe_allow_html=True)
            except Exception as e: st.warning(f"Gagal: {e}")
        with c2:
            st.markdown("**KS Test**")
            try:
                ks_s, ks_p = kstest(y_data, 'poisson', args=(lam,))
                st.write(f"Stat: `{ks_s:.6f}`, P: `{ks_p:.6f}`")
                if ks_p > 0.05: st.markdown('<div class="success-box">\u2705 Sesuai</div>', unsafe_allow_html=True)
                else: st.markdown('<div class="warning-box">\u26a0\ufe0f Tidak sesuai</div>', unsafe_allow_html=True)
            except Exception as e: st.warning(f"Gagal: {e}")
        with c3:
            st.markdown("**Zero-Inflation**")
            nz = np.sum(y_data==0); pz = 100*nz/len(y_data); ez = 100*poisson.pmf(0,lam)
            st.write(f"Zeros: `{nz}` ({pz:.1f}%), Expected: `{ez:.1f}%`")
            if pz > ez*2: st.markdown('<div class="warning-box">\u26a0\ufe0f Zero-inflated</div>', unsafe_allow_html=True)
            else: st.markdown('<div class="success-box">\u2705 Wajar</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 2\ufe0f\u20e3 Uji Overdispersi")
        st.markdown('''<div class="info-box">Overdispersi: Var > Mean. Jika dilanggar, SE terlalu kecil dan inferensi tidak valid.</div>''', unsafe_allow_html=True)
        X_const = sm.add_constant(X_data)
        try:
            pg = smPoisson(y_data, X_const).fit(disp=0)
            mu_hat = pg.predict(X_const)
            pr = (y_data - mu_hat) / np.sqrt(mu_hat)
            pchi2 = np.sum(pr**2); dp = pchi2 / (len(y_data) - len(pg.params))
            c1,c2 = st.columns(2)
            with c1:
                st.markdown("**Pearson Dispersion**")
                st.write(f"Chi\u00b2: `{pchi2:.4f}`, df: `{len(y_data)-len(pg.params)}`, \u03c6: `{dp:.4f}`")
                if dp < 1.5: st.markdown('<div class="success-box">\u2705 Equidispersion</div>', unsafe_allow_html=True)
                elif dp < 3: st.markdown('<div class="warning-box">\u26a0\ufe0f Mild overdispersion</div>', unsafe_allow_html=True)
                else: st.markdown('<div class="error-box">\u274c Severe overdispersion</div>', unsafe_allow_html=True)
            with c2:
                st.markdown("**Dean\'s Test**")
                try:
                    ay = ((y_data-mu_hat)**2-y_data)/mu_hat
                    am = sm.OLS(ay, sm.add_constant(mu_hat)).fit()
                    st.write(f"t: `{am.tvalues[1]:.4f}`, P: `{am.pvalues[1]:.6f}`")
                    if am.pvalues[1] > 0.05: st.markdown('<div class="success-box">\u2705 OK</div>', unsafe_allow_html=True)
                    else: st.markdown('<div class="warning-box">\u26a0\ufe0f Overdispersi signifikan</div>', unsafe_allow_html=True)
                except Exception as e: st.warning(f"Dean gagal: {e}")
        except Exception as e: st.error(f"Gagal fit: {e}")

        st.markdown("---")
        st.markdown("### 3\ufe0f\u20e3 VIF")
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        vif_df = pd.DataFrame({'Variabel': indeps})
        vif_df['VIF'] = [variance_inflation_factor(X_const, i+1) for i in range(len(indeps))]
        vif_df['Status'] = vif_df['VIF'].apply(lambda x: '\u2705 <5' if x<5 else ('\u26a0\ufe0f 5-10' if x<10 else '\u274c >10'))
        st.dataframe(vif_df, use_container_width=True)
        fv = px.bar(vif_df, x='Variabel', y='VIF', color='VIF', color_continuous_scale='RdYlGn_r', text='VIF')
        fv.add_hline(y=5, line_dash="dash", line_color="orange"); fv.add_hline(y=10, line_dash="dash", line_color="red")
        fv.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        st.plotly_chart(fv, use_container_width=True)

        st.markdown("---")
        st.markdown("### 4\ufe0f\u20e3 Moran\'s I")
        try:
            from libpysal.weights import KNN; from esda.moran import Moran
            ca = np.array(list(zip(df[st.session_state.lon_col].values, df[st.session_state.lat_col].values)))
            w = KNN.from_array(ca, k=8); w.transform = 'r'
            mi = Moran(pr, w)
            st.write(f"Moran\'s I: `{mi.I:.6f}`, Z: `{mi.z_norm:.4f}`, P: `{mi.p_norm:.6f}`")
            if mi.p_norm < 0.05:
                st.markdown('''<div class="warning-box">\u26a0\ufe0f <b>Autokorelasi spasial!</b> GWPR diperlukan.</div>''', unsafe_allow_html=True)
            else:
                st.markdown('<div class="success-box">\u2705 Tidak ada autokorelasi spasial</div>', unsafe_allow_html=True)
        except ImportError: st.warning("Install `libpysal` dan `esda`.")
        except Exception as e: st.warning(f"Moran gagal: {e}")
    else:
        st.warning("\u26a0\ufe0f Pilih variabel terlebih dahulu!")

# ============================================================
# TAB 3: POISSON GLOBAL (GLM)
# ============================================================
with tab3:
    st.markdown("## \U0001f4c8 Regresi Poisson Global (GLM)")
    if st.session_state.dep_var and st.session_state.indep_vars:
        dep = st.session_state.dep_var; indeps = st.session_state.indep_vars
        y_pois = df[dep].values; X_pois = df[indeps].values
        st.markdown('''<div class="info-box"><b>Poisson GLM:</b> log(\u03bc) = \u03b2\u2080 + \u03b2\u2081X\u2081 + ... + \u03b2\u2096X\u2096. Ini baseline sebelum GWPR.</div>''', unsafe_allow_html=True)
        X_const = sm.add_constant(X_pois)
        offset_arr = None
        if st.session_state.offset_col and st.session_state.offset_col in df.columns:
            offset_arr = np.log(np.maximum(df[st.session_state.offset_col].values, 1e-10))
            st.info(f"Offset: log({st.session_state.offset_col})")
        try:
            pm = smPoisson(y_pois, X_const, offset=offset_arr).fit(disp=0) if offset_arr is not None else smPoisson(y_pois, X_const).fit(disp=0)
            st.session_state.poisson_results = pm
            mu_h = pm.predict(X_const)
            pr2 = compute_poisson_pseudo_r2(y_pois, mu_h)
            dev = compute_poisson_deviance(y_pois, mu_h)
            pchi = np.sum((y_pois-mu_h)**2 / np.maximum(mu_h,1e-10))
            n = len(y_pois); k = len(pm.params)
            aicc = pm.aic + (2*k*(k+1)) / max(n-k-1, 1)
            disp = pchi / (n - k)
            st.session_state.poisson_aicc = aicc

            st.markdown("### Model Summary")
            c1,c2,c3,c4 = st.columns(4)
            with c1: st.metric("Pseudo R\u00b2", f"{pr2:.6f}")
            with c2: st.metric("AIC", f"{pm.aic:.4f}")
            with c3: st.metric("AICc", f"{aicc:.4f}")
            with c4: st.metric("BIC", f"{pm.bic:.4f}")
            c1,c2,c3,c4 = st.columns(4)
            with c1: st.metric("Deviance", f"{dev:.4f}")
            with c2: st.metric("Pearson Chi\u00b2", f"{pchi:.4f}")
            with c3: st.metric("Log-Lik", f"{pm.llf:.4f}")
            with c4: st.metric("Dispersion", f"{disp:.4f}")

            st.markdown("### Tabel Koefisien")
            ci = pm.conf_int()
            ci_lo = ci.iloc[:,0].values if hasattr(ci,'iloc') else ci[:,0]
            ci_hi = ci.iloc[:,1].values if hasattr(ci,'iloc') else ci[:,1]
            cdf = pd.DataFrame({'Variable':['Intercept']+indeps, 'Beta':pm.params.flatten(), 'IRR':np.exp(pm.params.flatten()), 'SE':pm.bse.flatten(), 'z':pm.tvalues.flatten(), 'P':pm.pvalues.flatten(), 'CI_lo':ci_lo, 'CI_hi':ci_hi, 'Sig':['\u2705' if p<0.05 else '\u274c' for p in pm.pvalues.flatten()]})
            st.dataframe(cdf.round(6), use_container_width=True)

            st.markdown("### IRR Forest Plot")
            idf = cdf[cdf['Variable']!='Intercept'].copy()
            idf['CI_lo_IRR'] = np.exp(idf['CI_lo']); idf['CI_hi_IRR'] = np.exp(idf['CI_hi'])
            fi = go.Figure()
            fi.add_trace(go.Scatter(x=idf['IRR'], y=idf['Variable'], mode='markers', marker=dict(size=12, color='#8E44AD'), error_x=dict(type='data', symmetric=False, array=idf['CI_hi_IRR']-idf['IRR'], arrayminus=idf['IRR']-idf['CI_lo_IRR']), name='IRR'))
            fi.add_vline(x=1, line_dash="dash", line_color="red")
            fi.update_layout(title="IRR + 95% CI", xaxis_title="IRR", height=400)
            st.plotly_chart(fi, use_container_width=True)

            st.markdown("### Persamaan Model")
            _p = pm.params.flatten()
            eq = f"log(\u03bc) = {_p[0]:.4f}"
            for i,v in enumerate(indeps):
                s = "+" if _p[i+1]>=0 else ""
                eq += f" {s}{_p[i+1]:.4f}\u00d7{v}"
            st.markdown(f"**{eq}**")

            with st.expander("Full Summary"):
                st.text(pm.summary().as_text())

            st.markdown("### Diagnostics")
            c1,c2 = st.columns(2)
            with c1:
                ff = px.scatter(x=mu_h, y=y_pois, title="Actual vs Predicted", labels={'x':'Predicted','y':'Actual'}, color_discrete_sequence=["#8E44AD"])
                ff.add_trace(go.Scatter(x=[min(mu_h.min(),y_pois.min()),max(mu_h.max(),y_pois.max())], y=[min(mu_h.min(),y_pois.min()),max(mu_h.max(),y_pois.max())], mode='lines', line=dict(color='red',dash='dash'), name='Perfect'))
                st.plotly_chart(ff, use_container_width=True)
            with c2:
                fd = px.scatter(x=mu_h, y=pm.resid_deviance, title="Deviance Residuals", labels={'x':'Predicted','y':'Dev Resid'}, color_discrete_sequence=["#E74C3C"])
                fd.add_hline(y=0, line_dash="dash"); fd.add_hline(y=2, line_dash="dot", line_color="orange"); fd.add_hline(y=-2, line_dash="dot", line_color="orange")
                st.plotly_chart(fd, use_container_width=True)
            c1,c2 = st.columns(2)
            with c1:
                fp = px.histogram(x=pm.resid_pearson, nbins=30, title="Pearson Residuals", marginal="box", color_discrete_sequence=["#27AE60"])
                st.plotly_chart(fp, use_container_width=True)
            with c2:
                from scipy.stats import poisson as pdist
                cr = np.arange(0, min(int(y_pois.max())+1, 50))
                oc = np.array([np.sum(y_pois==c) for c in cr])
                ec = np.array([pdist.pmf(c, mu_h.mean())*len(y_pois) for c in cr])
                fr = go.Figure()
                fr.add_trace(go.Bar(x=cr, y=oc, name='Observed', marker_color='#8E44AD', opacity=0.7))
                fr.add_trace(go.Scatter(x=cr, y=ec, name='Expected', mode='lines+markers', line=dict(color='red')))
                fr.update_layout(title="Obs vs Expected Counts")
                st.plotly_chart(fr, use_container_width=True)
        except Exception as e:
            st.error(f"Error: {e}")
            import traceback; st.code(traceback.format_exc())
    else:
        st.warning("Pilih variabel terlebih dahulu!")

# ============================================================
# TAB 4: GWPR MODEL
# ============================================================
with tab4:
    st.markdown("## \U0001f30d GWPR Model")
    if st.session_state.dep_var and st.session_state.indep_vars:
        dep = st.session_state.dep_var; indeps = st.session_state.indep_vars
        y_gw = df[dep].values.reshape(-1,1); X_gw = df[indeps].values
        st.markdown('''<div class="info-box"><b>GWPR:</b> log(\u03bc\u1d62) = \u03b2\u2080(u\u1d62,v\u1d62) + \u2211 \u03b2\u2096(u\u1d62,v\u1d62)\u00b7X\u2096\u1d62. Koefisien bervariasi spasial.</div>''', unsafe_allow_html=True)
        offset_gw = None
        if st.session_state.offset_col and st.session_state.offset_col in df.columns:
            offset_gw = np.log(np.maximum(df[st.session_state.offset_col].values.reshape(-1,1), 1e-10))

        if st.button("\U0001f680 Jalankan GWPR", use_container_width=True, type="primary"):
            with st.spinner("Mencari bandwidth & estimasi GWPR..."):
                try:
                    from mgwr.gwr import GWR; from mgwr.sel_bw import Sel_BW; from spglm.family import Poisson
                    ca = np.array(list(zip(df[st.session_state.lon_col].values, df[st.session_state.lat_col].values)))
                    sel = Sel_BW(ca, y_gw, X_gw, family=Poisson(), offset=offset_gw, kernel=st.session_state.kernel_type, fixed=st.session_state.bw_fixed)
                    bw = sel.search(criterion=st.session_state.criterion, search_method=st.session_state.search_method)
                    st.session_state.gwpr_bw = bw; st.session_state.gwpr_selector = sel
                    gm = GWR(ca, y_gw, X_gw, bw=bw, family=Poisson(), offset=offset_gw, kernel=st.session_state.kernel_type, fixed=st.session_state.bw_fixed)
                    gr = gm.fit()
                    st.session_state.gwpr_results = gr; st.session_state.gwpr_coords = ca; st.session_state.y = y_gw
                    st.success(f"\u2705 GWPR OK! Bandwidth: {bw}")
                except Exception as e:
                    st.error(f"Error: {e}"); import traceback; st.code(traceback.format_exc())

        if st.session_state.gwpr_results is not None:
            gr = st.session_state.gwpr_results; bw = st.session_state.gwpr_bw
            ya = df[dep].values; mu_g = np.maximum(gr.predy.flatten(), 1e-10)
            pr2g = compute_poisson_pseudo_r2(ya, mu_g); devg = compute_poisson_deviance(ya, mu_g)
            pchg = np.sum((ya-mu_g)**2/mu_g)
            enp = gr.ENP if hasattr(gr,'ENP') else len(indeps)+1

            st.markdown("### Diagnostik Global")
            c1,c2,c3,c4 = st.columns(4)
            with c1: st.metric("Bandwidth", f"{bw:.2f}" if isinstance(bw,float) else str(bw))
            with c2: st.metric("Pseudo R\u00b2", f"{pr2g:.6f}")
            with c3: st.metric("AICc", f"{gr.aicc:.4f}")
            with c4: st.metric("AIC", f"{gr.aic:.4f}")
            c1,c2,c3,c4 = st.columns(4)
            with c1: st.metric("Deviance", f"{devg:.4f}")
            with c2: st.metric("Pearson Chi\u00b2", f"{pchg:.4f}")
            with c3: st.metric("ENP", f"{enp:.2f}" if isinstance(enp,(int,float,np.floating)) else str(enp))
            enp_v = enp if isinstance(enp,(int,float,np.floating)) else len(indeps)+1
            with c4: st.metric("Dispersion", f"{pchg/max(len(ya)-enp_v,1):.4f}")

            vn = ['Intercept'] + indeps; pa = gr.params; tv = gr.tvalues; se = gr.bse

            st.markdown("### Koefisien Lokal")
            cs = pd.DataFrame()
            for i,v in enumerate(vn):
                cs = pd.concat([cs, pd.DataFrame([{'Variable':v,'Mean':pa[:,i].mean(),'Std':pa[:,i].std(),'Min':pa[:,i].min(),'Q1':np.percentile(pa[:,i],25),'Median':np.median(pa[:,i]),'Q3':np.percentile(pa[:,i],75),'Max':pa[:,i].max()}])], ignore_index=True)
            st.dataframe(cs.round(6), use_container_width=True)

            st.markdown("### IRR Lokal")
            irr_p = np.exp(pa); isum = pd.DataFrame()
            for i,v in enumerate(vn):
                isum = pd.concat([isum, pd.DataFrame([{'Variable':v,'Mean IRR':irr_p[:,i].mean(),'Min':irr_p[:,i].min(),'Median':np.median(irr_p[:,i]),'Max':irr_p[:,i].max(),'%>1':f"{100*np.sum(irr_p[:,i]>1)/len(irr_p):.1f}%",'%<1':f"{100*np.sum(irr_p[:,i]<1)/len(irr_p):.1f}%"}])], ignore_index=True)
            st.dataframe(isum.round(6), use_container_width=True)

            st.markdown("### t-values Lokal")
            tsum = pd.DataFrame()
            for i,v in enumerate(vn):
                ns = np.sum(np.abs(tv[:,i])>1.96)
                tsum = pd.concat([tsum, pd.DataFrame([{'Variable':v,'Mean |t|':np.abs(tv[:,i]).mean(),'Min t':tv[:,i].min(),'Max t':tv[:,i].max(),'Sig (|t|>1.96)':f"{ns} ({100*ns/len(tv):.1f}%)"}])], ignore_index=True)
            st.dataframe(tsum, use_container_width=True)

            st.markdown("### Distribusi Koefisien Lokal")
            nv = len(vn); cpr = min(3,nv); rn = (nv+cpr-1)//cpr
            fc = make_subplots(rows=rn, cols=cpr, subplot_titles=vn)
            for i,v in enumerate(vn):
                r = i//cpr+1; c = i%cpr+1
                fc.add_trace(go.Histogram(x=pa[:,i], nbinsx=25, name=v, marker_color=px.colors.qualitative.Set2[i%8]), row=r, col=c)
                if st.session_state.poisson_results is not None:
                    fc.add_vline(x=st.session_state.poisson_results.params[i], line_dash="dash", line_color="red", row=r, col=c)
            fc.update_layout(height=300*rn, showlegend=False, title_text="Koefisien Lokal (merah=global)")
            st.plotly_chart(fc, use_container_width=True)

            st.markdown("### Distribusi IRR Lokal")
            fi2 = make_subplots(rows=rn, cols=cpr, subplot_titles=[f"IRR: {v}" for v in vn])
            for i,v in enumerate(vn):
                r = i//cpr+1; c = i%cpr+1
                fi2.add_trace(go.Histogram(x=irr_p[:,i], nbinsx=25, name=v, marker_color=px.colors.qualitative.Pastel[i%8]), row=r, col=c)
                fi2.add_vline(x=1, line_dash="dash", line_color="red", row=r, col=c)
            fi2.update_layout(height=300*rn, showlegend=False, title_text="IRR Lokal (merah=1)")
            st.plotly_chart(fi2, use_container_width=True)

            st.markdown("### Diagnostics GWPR")
            c1,c2 = st.columns(2)
            with c1:
                fg = px.scatter(x=mu_g, y=ya, title="Actual vs Predicted GWPR", labels={'x':'Predicted','y':'Actual'}, color_discrete_sequence=["#27AE60"])
                fg.add_trace(go.Scatter(x=[min(mu_g.min(),ya.min()),max(mu_g.max(),ya.max())],y=[min(mu_g.min(),ya.min()),max(mu_g.max(),ya.max())],mode='lines',line=dict(color='red',dash='dash'),name='Perfect'))
                st.plotly_chart(fg, use_container_width=True)
            with c2:
                sr = (ya-mu_g)/np.sqrt(mu_g)
                fsr = px.scatter(x=mu_g, y=sr, title="Std Pearson Resid", labels={'x':'Predicted','y':'Resid'}, color_discrete_sequence=["#E74C3C"])
                fsr.add_hline(y=0, line_dash="dash"); fsr.add_hline(y=2, line_dash="dot", line_color="orange"); fsr.add_hline(y=-2, line_dash="dot", line_color="orange")
                st.plotly_chart(fsr, use_container_width=True)

            st.markdown("### Export Parameter Lokal")
            edf = df.copy()
            for i,v in enumerate(vn):
                edf[f'Beta_{v}']=pa[:,i]; edf[f'IRR_{v}']=np.exp(pa[:,i]); edf[f'SE_{v}']=se[:,i]; edf[f'tval_{v}']=tv[:,i]; edf[f'Sig_{v}']=np.where(np.abs(tv[:,i])>1.96,'Yes','No')
            edf['Predicted']=mu_g; edf['Residual']=ya-mu_g; edf['Pearson_Resid']=(ya-mu_g)/np.sqrt(mu_g)
            buf = io.StringIO(); edf.to_csv(buf, index=False)
            st.download_button("\U0001f4e5 Download GWPR Params (CSV)", buf.getvalue(), f"gwpr_params_{datetime.now().strftime(\'%Y%m%d_%H%M%S\')}.csv", "text/csv", use_container_width=True)
    else:
        st.warning("Pilih variabel terlebih dahulu!")

# ============================================================
# TAB 5: PERBANDINGAN MODEL
# ============================================================
with tab5:
    st.markdown("## Perbandingan Model")
    hp = st.session_state.poisson_results is not None
    hg = st.session_state.gwpr_results is not None
    if not (hp or hg):
        st.warning("Jalankan minimal satu model terlebih dahulu.")
    else:
        dep = st.session_state.dep_var; indeps = st.session_state.indep_vars
        ya = df[dep].values
        st.markdown("### Tabel Perbandingan")
        cd = []
        if hp:
            pois = st.session_state.poisson_results
            mu_p = pois.predict(sm.add_constant(df[indeps].values))
            pr2p = compute_poisson_pseudo_r2(ya, mu_p)
            dev_p = compute_poisson_deviance(ya, mu_p)
            pchp = np.sum((ya-mu_p)**2/np.maximum(mu_p,1e-10))
            cd.append({'Model':'Poisson Global','Pseudo R2':f"{pr2p:.6f}",'AIC':f"{pois.aic:.4f}",'AICc':f"{st.session_state.get('poisson_aicc',pois.aic):.4f}",'BIC':f"{pois.bic:.4f}",'Deviance':f"{dev_p:.4f}",'Pearson Chi2':f"{pchp:.4f}",'LogLik':f"{pois.llf:.4f}",'Params':f"{len(pois.params)}"})
        if hg:
            gw = st.session_state.gwpr_results
            mu_gw = np.maximum(gw.predy.flatten(), 1e-10)
            pr2g = compute_poisson_pseudo_r2(ya, mu_gw)
            dev_g = compute_poisson_deviance(ya, mu_gw)
            pchg = np.sum((ya-mu_gw)**2/mu_gw)
            enp = gw.ENP if hasattr(gw,'ENP') else len(indeps)+1
            cd.append({'Model':'GWPR','Pseudo R2':f"{pr2g:.6f}",'AIC':f"{gw.aic:.4f}",'AICc':f"{gw.aicc:.4f}",'Deviance':f"{dev_g:.4f}",'Pearson Chi2':f"{pchg:.4f}",'ENP':f"{enp:.2f}" if isinstance(enp,(int,float,np.floating)) else str(enp)})
        st.dataframe(pd.DataFrame(cd), use_container_width=True)

        if hp and hg:
            st.markdown("### Rekomendasi Model")
            ap = st.session_state.get('poisson_aicc', pois.aic)
            ag = gw.aicc; da = ap - ag
            c1,c2,c3 = st.columns(3)
            with c1: st.metric("Delta AIC", f"{pois.aic-gw.aic:.4f}")
            with c2: st.metric("Delta AICc", f"{da:.4f}")
            with c3:
                di = ((dev_p-dev_g)/dev_p*100) if dev_p>0 else 0
                st.metric("Dev Improvement", f"{di:.2f}%")
            if da > 2:
                st.markdown(f'<div class="success-box">\u2705 <b>GWPR lebih baik!</b> Delta AICc = {da:.2f} > 2.</div>', unsafe_allow_html=True)
            elif da > 0:
                st.markdown(f'<div class="warning-box">\u26a0\ufe0f <b>GWPR sedikit lebih baik.</b> Delta AICc = {da:.2f}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="info-box">\u2139\ufe0f <b>Global cukup.</b> Delta AICc = {da:.2f}</div>', unsafe_allow_html=True)

        st.markdown("### Predicted vs Actual")
        fp = go.Figure()
        fp.add_trace(go.Scatter(x=[ya.min(),ya.max()],y=[ya.min(),ya.max()],mode='lines',line=dict(color='black',dash='dash'),name='Perfect'))
        if hp: fp.add_trace(go.Scatter(x=mu_p,y=ya,mode='markers',name='Global',marker=dict(color='blue',size=5,opacity=0.5)))
        if hg: fp.add_trace(go.Scatter(x=mu_gw,y=ya,mode='markers',name='GWPR',marker=dict(color='green',size=5,opacity=0.5)))
        fp.update_layout(title="Predicted vs Actual", xaxis_title="Predicted", yaxis_title="Actual", height=500)
        st.plotly_chart(fp, use_container_width=True)

        st.markdown("### Distribusi Residual")
        rd = []
        if hp:
            for r in (ya-mu_p)/np.sqrt(np.maximum(mu_p,1e-10)): rd.append({'Model':'Global','Pearson Residual':r})
        if hg:
            for r in (ya-mu_gw)/np.sqrt(mu_gw): rd.append({'Model':'GWPR','Pearson Residual':r})
        if rd:
            st.plotly_chart(px.box(pd.DataFrame(rd), x='Model', y='Pearson Residual', color='Model', title="Pearson Residual"), use_container_width=True)

        st.markdown("### Error Metrics")
        md = []
        if hp: md.append({'Model':'Global','RMSE':f"{np.sqrt(np.mean((ya-mu_p)**2)):.4f}",'MAE':f"{np.mean(np.abs(ya-mu_p)):.4f}",'MAPE':f"{np.mean(np.abs((ya-mu_p)/np.maximum(ya,1)))*100:.2f}%"})
        if hg: md.append({'Model':'GWPR','RMSE':f"{np.sqrt(np.mean((ya-mu_gw)**2)):.4f}",'MAE':f"{np.mean(np.abs(ya-mu_gw)):.4f}",'MAPE':f"{np.mean(np.abs((ya-mu_gw)/np.maximum(ya,1)))*100:.2f}%"})
        if md: st.dataframe(pd.DataFrame(md), use_container_width=True)

        if hp and hg and st.session_state.lon_col and st.session_state.lat_col:
            st.markdown("### Peta Residual")
            c1,c2 = st.columns(2)
            with c1:
                st.plotly_chart(px.scatter_mapbox(df, lon=st.session_state.lon_col, lat=st.session_state.lat_col, color=ya-mu_p, size=np.abs(ya-mu_p), size_max=15, zoom=6, mapbox_style="open-street-map", title="Residual Global", color_continuous_scale="RdBu_r", color_continuous_midpoint=0).update_layout(height=400), use_container_width=True)
            with c2:
                st.plotly_chart(px.scatter_mapbox(df, lon=st.session_state.lon_col, lat=st.session_state.lat_col, color=ya-mu_gw, size=np.abs(ya-mu_gw), size_max=15, zoom=6, mapbox_style="open-street-map", title="Residual GWPR", color_continuous_scale="RdBu_r", color_continuous_midpoint=0).update_layout(height=400), use_container_width=True)

# ============================================================
# TAB 6: VISUALISASI PETA
# ============================================================
with tab6:
    st.markdown("## Visualisasi Peta")
    if not FOLIUM_AVAILABLE:
        st.error("Install folium: `pip install folium streamlit-folium branca`")
    elif st.session_state.dep_var and st.session_state.indep_vars:
        dep = st.session_state.dep_var; indeps = st.session_state.indep_vars
        lc = st.session_state.lon_col; ac = st.session_state.lat_col
        mo = []
        if st.session_state.poisson_results is not None: mo.append("Poisson Global")
        if st.session_state.gwpr_results is not None: mo.append("GWPR")
        if not mo:
            st.warning("Belum ada model!")
        else:
            mm = st.selectbox("Model:", mo)
            vn = ['Intercept'] + indeps; gdf = df.copy()
            if mm == "Poisson Global":
                pois = st.session_state.poisson_results
                mu = pois.predict(sm.add_constant(df[indeps].values))
                gdf['Predicted']=mu; gdf['Residual']=df[dep].values-mu; gdf['PR']=gdf['Residual']/np.sqrt(np.maximum(mu,1e-10))
            else:
                gw = st.session_state.gwpr_results
                for i,v in enumerate(vn):
                    gdf[f'B_{v}']=gw.params[:,i]; gdf[f'IRR_{v}']=np.exp(gw.params[:,i]); gdf[f't_{v}']=gw.tvalues[:,i]; gdf[f'Sig_{v}']=np.where(np.abs(gw.tvalues[:,i])>1.96,'Sig','No')
                mu = gw.predy.flatten()
                gdf['Predicted']=mu; gdf['Residual']=df[dep].values-mu; gdf['PR']=gdf['Residual']/np.sqrt(np.maximum(mu,1e-10))

            lo = ["Predicted vs Observed","Residual (Pearson)"]
            if mm == "GWPR":
                for v in vn: lo += [f"Beta: {v}", f"IRR: {v}", f"t-stat: {v}"]
            lt = st.selectbox("Layer:", lo)

            m = folium.Map(location=[df[ac].mean(), df[lc].mean()], zoom_start=8, tiles='CartoDB positron')
            if lt == "Predicted vs Observed":
                mx = max(np.abs(gdf['Residual']).max(), 1e-10)
                for _,row in gdf.iterrows():
                    rs = row['Residual']; cl = 'green' if rs>=0 else 'red'; rd = 4+8*abs(rs)/mx
                    folium.CircleMarker([row[ac],row[lc]], radius=rd, color=cl, fill=True, fill_color=cl, fill_opacity=0.7, popup=folium.Popup(f"<b>{dep}</b>:{row[dep]}<br>Pred:{row['Predicted']:.2f}<br>Res:{rs:.2f}", max_width=250)).add_to(m)
            elif lt == "Residual (Pearson)":
                ser = gdf['PR']; cmp = create_colormap_safe(ser, 'RdBu_r')
                for _,row in gdf.iterrows():
                    val = row['PR']
                    try: cl = cmp(float(val))
                    except: cl = '#808080'
                    folium.CircleMarker([row[ac],row[lc]], radius=6, color=cl, fill=True, fill_color=cl, fill_opacity=0.8, popup=folium.Popup(f"PR:{val:.3f}", max_width=200)).add_to(m)
                cmp.caption="Pearson Residual"; m.add_child(cmp)
            elif lt.startswith("Beta:"):
                var = lt.replace("Beta: ",""); fld = f'B_{var}'
                if fld in gdf.columns:
                    ser = gdf[fld]; cmp = create_colormap_safe(ser, 'RdYlBu')
                    for _,row in gdf.iterrows():
                        val = row[fld]
                        try: cl = cmp(float(val))
                        except: cl = '#808080'
                        folium.CircleMarker([row[ac],row[lc]], radius=6, color=cl, fill=True, fill_color=cl, fill_opacity=0.8, popup=folium.Popup(f"<b>{var}</b><br>B={val:.4f}<br>IRR={np.exp(val):.4f}<br>{row.get(f'Sig_{var}','-')}", max_width=250)).add_to(m)
                    cmp.caption=f"Beta({var})"; m.add_child(cmp)
            elif lt.startswith("IRR:"):
                var = lt.replace("IRR: ",""); fld = f'IRR_{var}'
                if fld in gdf.columns:
                    ser = gdf[fld]; cmp = create_colormap_safe(ser, 'RdYlGn')
                    for _,row in gdf.iterrows():
                        val = row[fld]
                        try: cl = cmp(float(val))
                        except: cl = '#808080'
                        folium.CircleMarker([row[ac],row[lc]], radius=6, color=cl, fill=True, fill_color=cl, fill_opacity=0.8, popup=folium.Popup(f"IRR {var}={val:.4f}", max_width=200)).add_to(m)
                    cmp.caption=f"IRR({var})"; m.add_child(cmp)
            elif lt.startswith("t-stat:"):
                var = lt.replace("t-stat: ",""); fld = f't_{var}'
                if fld in gdf.columns:
                    ser = gdf[fld]; cmp = create_colormap_safe(ser, 'PiYG')
                    for _,row in gdf.iterrows():
                        val = row[fld]
                        try: cl = cmp(float(val))
                        except: cl = '#808080'
                        sig = "Yes" if abs(val)>1.96 else "No"
                        folium.CircleMarker([row[ac],row[lc]], radius=6, color=cl, fill=True, fill_color=cl, fill_opacity=0.8, popup=folium.Popup(f"t({var})={val:.4f}<br>Sig:{sig}", max_width=200)).add_to(m)
                    cmp.caption=f"t({var})"; m.add_child(cmp)
            folium.LayerControl().add_to(m)
            st_folium(m, width=1200, height=600)
    else:
        st.warning("Pilih variabel terlebih dahulu!")

# ============================================================
# TAB 7: LAPORAN & EXPORT
# ============================================================
with tab7:
    st.markdown("## Laporan & Export")
    if st.session_state.dep_var and st.session_state.indep_vars:
        dep = st.session_state.dep_var; indeps = st.session_state.indep_vars
        rl = []
        rl.append("="*60); rl.append("LAPORAN ANALISIS GWPR")
        rl.append(f"Tanggal: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        rl.append("="*60); rl.append("")
        rl.append("1. DATA")
        rl.append(f"   N: {len(df)}, Y: {dep}, X: {', '.join(indeps)}")
        rl.append(f"   Koordinat: {st.session_state.lon_col}, {st.session_state.lat_col}")
        if st.session_state.offset_col: rl.append(f"   Offset: log({st.session_state.offset_col})")
        rl.append("")
        yv = df[dep].values
        rl.append("2. DESKRIPSI Y")
        rl.append(f"   Mean={yv.mean():.4f}, Var={yv.var():.4f}, Var/Mean={yv.var()/max(yv.mean(),1e-10):.4f}")
        rl.append(f"   Range=({yv.min()},{yv.max()}), Zeros={np.sum(yv==0)} ({100*np.sum(yv==0)/len(yv):.1f}%)")
        rl.append("")

        if st.session_state.poisson_results is not None:
            pois = st.session_state.poisson_results
            mu_p = pois.predict(sm.add_constant(df[indeps].values))
            rl.append("3. POISSON GLOBAL")
            rl.append(f"   AIC={pois.aic:.4f}, BIC={pois.bic:.4f}, LogLik={pois.llf:.4f}")
            rl.append(f"   Deviance={compute_poisson_deviance(yv,mu_p):.4f}, PseudoR2={compute_poisson_pseudo_r2(yv,mu_p):.6f}")
            rl.append("   Koefisien:")
            vnl = ['Intercept']+indeps
            for i,v in enumerate(vnl):
                sig = "*" if pois.pvalues[i]<0.05 else ""
                rl.append(f"   {v:25s} B={pois.params[i]:.6f} IRR={np.exp(pois.params[i]):.4f} SE={pois.bse[i]:.6f} z={pois.tvalues[i]:.4f} p={pois.pvalues[i]:.4f} {sig}")
            rl.append("")

        if st.session_state.gwpr_results is not None:
            gw = st.session_state.gwpr_results
            mu_gw = np.maximum(gw.predy.flatten(), 1e-10)
            rl.append("4. GWPR")
            rl.append(f"   BW={st.session_state.gwpr_bw}, AICc={gw.aicc:.4f}, AIC={gw.aic:.4f}")
            rl.append(f"   Deviance={compute_poisson_deviance(yv,mu_gw):.4f}, PseudoR2={compute_poisson_pseudo_r2(yv,mu_gw):.6f}")
            enp = gw.ENP if hasattr(gw,'ENP') else '-'
            rl.append(f"   ENP={enp}")
            rl.append("   Koefisien Lokal:")
            vnl = ['Intercept']+indeps
            rl.append(f"   {'Var':20s} {'Mean':>10s} {'Std':>10s} {'Min':>10s} {'Med':>10s} {'Max':>10s} {'%Sig':>7s}")
            for i,v in enumerate(vnl):
                ns = np.sum(np.abs(gw.tvalues[:,i])>1.96); ps = 100*ns/len(gw.tvalues)
                rl.append(f"   {v:20s} {gw.params[:,i].mean():>10.4f} {gw.params[:,i].std():>10.4f} {gw.params[:,i].min():>10.4f} {np.median(gw.params[:,i]):>10.4f} {gw.params[:,i].max():>10.4f} {ps:>6.1f}%")
            rl.append("")

        if st.session_state.poisson_results is not None and st.session_state.gwpr_results is not None:
            rl.append("5. PERBANDINGAN")
            ap = st.session_state.get('poisson_aicc', pois.aic); ag = gw.aicc
            rl.append(f"   AICc Global={ap:.4f}, AICc GWPR={ag:.4f}, Delta={ap-ag:.4f}")
            rl.append(f"   Model terbaik: {'GWPR' if ag<ap else 'Global'}")
            rl.append("")

        rl.append("CATATAN INTERPRETASI")
        rl.append("- IRR=exp(B): IRR>1=meningkatkan, IRR<1=menurunkan count")
        rl.append("- |t|>1.96 = signifikan pada alpha=5%")
        rl.append("- BW kecil=sangat lokal, BW besar=mendekati global")

        rt = "\n".join(rl)
        st.text_area("Preview", rt, height=400)
        st.download_button("Download Laporan (.txt)", rt, f"laporan_GWPR_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", "text/plain", use_container_width=True)

        st.markdown("### Export Konfigurasi")
        cfg = {"analysis":"GWPR","timestamp":datetime.now().isoformat(),"data":{"n":len(df),"dep":dep,"indep":indeps,"lon":st.session_state.lon_col,"lat":st.session_state.lat_col,"offset":st.session_state.offset_col},"settings":{"kernel":st.session_state.get('kernel_type','bisquare'),"fixed":st.session_state.get('bw_fixed',False),"criterion":st.session_state.get('criterion','AICc')}}
        if st.session_state.poisson_results is not None:
            p = st.session_state.poisson_results
            cfg["global"] = {"aic":float(p.aic),"bic":float(p.bic),"ll":float(p.llf),"params":{v:float(p.params[i]) for i,v in enumerate(['Intercept']+indeps)},"irr":{v:float(np.exp(p.params[i])) for i,v in enumerate(['Intercept']+indeps)}}
        if st.session_state.gwpr_results is not None:
            g = st.session_state.gwpr_results
            cfg["gwpr"] = {"bw":float(st.session_state.gwpr_bw) if isinstance(st.session_state.gwpr_bw,(int,float,np.floating)) else str(st.session_state.gwpr_bw),"aicc":float(g.aicc),"aic":float(g.aic)}
        cj = json.dumps(cfg, indent=2, default=str)
        st.download_button("Download Config (.json)", cj, f"gwpr_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "application/json", use_container_width=True)

        st.markdown("### Export Data Lengkap")
        if st.session_state.gwpr_results is not None:
            gw = st.session_state.gwpr_results; edf = df.copy()
            vnl = ['Intercept']+indeps
            for i,v in enumerate(vnl):
                edf[f'Beta_{v}']=gw.params[:,i]; edf[f'IRR_{v}']=np.exp(gw.params[:,i])
                edf[f'SE_{v}']=gw.bse[:,i]; edf[f'tval_{v}']=gw.tvalues[:,i]
                edf[f'Sig_{v}']=np.where(np.abs(gw.tvalues[:,i])>1.96,'Yes','No')
            mu = gw.predy.flatten()
            edf['Predicted']=mu; edf['Residual']=df[dep].values-mu
            edf['Pearson_Resid']=(df[dep].values-mu)/np.sqrt(np.maximum(mu,1e-10))
            buf = io.StringIO(); edf.to_csv(buf, index=False)
            st.download_button("Download Full Data+GWPR (CSV)", buf.getvalue(), f"gwpr_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", "text/csv", use_container_width=True)
        else:
            st.info("Jalankan GWPR terlebih dahulu.")
    else:
        st.warning("Pilih variabel terlebih dahulu!")

# FOOTER
st.markdown("---")
st.markdown(f'<div style="text-align:center;color:#888;font-size:0.85rem;">GWPR Analysis Suite | Powered by mgwr, statsmodels, streamlit | {datetime.now().strftime("%Y-%m-%d %H:%M")}</div>', unsafe_allow_html=True)
