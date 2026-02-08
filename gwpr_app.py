# ============================================================
# APLIKASI ANALISIS GWPR (Geographically Weighted Poisson Regression)
# File: gwpr_app.py
# Jalankan: streamlit run gwpr_app.py
# ============================================================
# INSTALL:
# pip install streamlit pandas numpy scipy matplotlib seaborn
# pip install statsmodels scikit-learn mgwr libpysal geopandas
# pip install folium streamlit-folium plotly branca esda
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

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="GWPR Analysis Suite",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1B4F72 !important;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 3px solid #8E44AD;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #5D6D7E !important;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #8E44AD 0%, #3498DB 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white !important;
        text-align: center;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #D5F5E3;
        border-left: 5px solid #27AE60;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        color: #1a5c2e !important;
    }
    .success-box b, .success-box strong, .success-box span {
        color: #1a5c2e !important;
    }
    .warning-box {
        background-color: #FEF9E7;
        border-left: 5px solid #F39C12;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        color: #7d5a00 !important;
    }
    .warning-box b, .warning-box strong, .warning-box span {
        color: #7d5a00 !important;
    }
    .error-box {
        background-color: #FADBD8;
        border-left: 5px solid #E74C3C;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        color: #922b21 !important;
    }
    .error-box b, .error-box strong, .error-box span {
        color: #922b21 !important;
    }
    .info-box {
        background-color: #D6EAF8;
        border-left: 5px solid #2E86C1;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        color: #1a4971 !important;
    }
    .info-box b, .info-box strong, .info-box span {
        color: #1a4971 !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 5px 5px 0 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================
defaults = {
    'data': None, 'poisson_results': None, 'gwpr_results': None,
    'ols_results': None, 'coords': None, 'y': None, 'X': None,
    'var_names': None, 'dep_var': None, 'indep_vars': [],
    'lon_col': None, 'lat_col': None, 'gwpr_bw': None,
    'gwpr_selector': None, 'offset_col': None
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def safe_get_localR2(model_results, y=None):
    """Safely get localR2, with fallback for Poisson models."""
    try:
        return model_results.localR2.flatten()
    except (NotImplementedError, AttributeError):
        if y is not None and hasattr(model_results, 'predy'):
            y_flat = y.flatten()
            pred_flat = model_results.predy.flatten()
            ss_tot = np.sum((y_flat - y_flat.mean()) ** 2)
            if ss_tot == 0:
                return np.zeros_like(y_flat)
            ss_res = np.sum((y_flat - pred_flat) ** 2)
            global_r2 = 1 - ss_res / ss_tot
            return np.full_like(y_flat, global_r2, dtype=float)
        return None

def create_colormap_safe(series, cmap_name='RdYlBu', n_bins=7):
    """Create branca colormap with hex colors."""
    vmin, vmax = float(series.min()), float(series.max())
    if vmin == vmax:
        vmin -= 1
        vmax += 1
    rgba = plt.get_cmap(cmap_name)(np.linspace(0, 1, n_bins))
    hex_colors = [mcolors.to_hex(c) for c in rgba]
    return cm.LinearColormap(colors=hex_colors, vmin=vmin, vmax=vmax)

def compute_poisson_deviance(y, mu):
    """Compute Poisson deviance: 2 * sum(y*log(y/mu) - (y - mu))."""
    y = np.asarray(y, dtype=float).flatten()
    mu = np.asarray(mu, dtype=float).flatten()
    mu = np.maximum(mu, 1e-10)
    dev = np.zeros_like(y)
    mask = y > 0
    dev[mask] = y[mask] * np.log(y[mask] / mu[mask])
    dev = 2 * np.sum(dev - (y - mu))
    return dev

def compute_poisson_pseudo_r2(y, mu):
    """McFadden pseudo R2 for Poisson: 1 - loglik(model)/loglik(null)."""
    from scipy.special import gammaln
    y = np.asarray(y, dtype=float).flatten()
    mu = np.asarray(mu, dtype=float).flatten()
    mu = np.maximum(mu, 1e-10)
    y_safe = np.maximum(y, 0)
    # Log-likelihood of fitted model
    ll_model = np.sum(y_safe * np.log(mu) - mu - gammaln(y_safe + 1))
    # Log-likelihood of null model (intercept only = mean)
    mu_null = np.mean(y_safe)
    mu_null = max(mu_null, 1e-10)
    ll_null = np.sum(y_safe * np.log(mu_null) - mu_null - gammaln(y_safe + 1))
    if ll_null == 0:
        return 0.0
    return 1 - ll_model / ll_null

def compute_poisson_aic(y, mu, k):
    """AIC = -2*loglik + 2*k."""
    from scipy.special import gammaln
    y = np.asarray(y, dtype=float).flatten()
    mu = np.asarray(mu, dtype=float).flatten()
    mu = np.maximum(mu, 1e-10)
    y_safe = np.maximum(y, 0)
    ll = np.sum(y_safe * np.log(mu) - mu - gammaln(y_safe + 1))
    return -2 * ll + 2 * k

# ============================================================
# DEMO DATA GENERATOR
# ============================================================
def generate_demo_data(scenario="default"):
    """Generate demo count data for GWPR analysis."""
    np.random.seed(42)

    if scenario == "Kasus DBD (Dengue Fever)":
        n = 150
        lon = np.random.uniform(110.3, 114.6, n)
        lat = np.random.uniform(-8.2, -6.8, n)
        kepadatan = np.random.uniform(500, 20000, n)
        puskesmas = np.random.uniform(0.5, 10, n)
        sanitasi = np.random.uniform(20, 95, n)
        # Spatially varying log-linear
        beta0 = 2.5 + 0.5 * np.sin((lon - 112) * 2) + np.random.normal(0, 0.1, n)
        beta1 = 0.00005 + 0.00003 * (lat + 7.5) + np.random.normal(0, 0.00001, n)
        beta2 = -0.08 + 0.03 * np.cos(lon * 2) + np.random.normal(0, 0.01, n)
        beta3 = -0.005 + 0.003 * (lon - 112) + np.random.normal(0, 0.001, n)
        log_mu = beta0 + beta1 * kepadatan + beta2 * puskesmas + beta3 * sanitasi
        log_mu = np.clip(log_mu, 0, 6)
        mu = np.exp(log_mu)
        y = np.random.poisson(mu)
        df = pd.DataFrame({
            'ID': [f'Kecamatan_{i+1}' for i in range(n)],
            'Longitude': np.round(lon, 4),
            'Latitude': np.round(lat, 4),
            'Kasus_DBD': y,
            'Kepadatan_per_km2': np.round(kepadatan, 0),
            'Rasio_Puskesmas': np.round(puskesmas, 2),
            'Pct_Sanitasi_Baik': np.round(sanitasi, 1)
        })
        return df, "Kasus_DBD", ["Kepadatan_per_km2", "Rasio_Puskesmas", "Pct_Sanitasi_Baik"], "Longitude", "Latitude"

    elif scenario == "Kematian Bayi (Infant Mortality)":
        n = 120
        lon = np.random.uniform(106.6, 107.1, n)
        lat = np.random.uniform(-6.9, -6.1, n)
        bidan = np.random.uniform(1, 20, n)
        kemiskinan = np.random.uniform(5, 40, n)
        imunisasi = np.random.uniform(50, 99, n)
        penduduk = np.random.uniform(5000, 100000, n)
        beta0 = 1.5 + 0.3 * (lon - 106.85) * 5 + np.random.normal(0, 0.1, n)
        beta1 = -0.05 + 0.02 * (lat + 6.5) + np.random.normal(0, 0.005, n)
        beta2 = 0.02 + 0.01 * np.sin(lon * 30) + np.random.normal(0, 0.003, n)
        beta3 = -0.01 + 0.005 * (lon - 106.85) + np.random.normal(0, 0.002, n)
        log_mu = beta0 + beta1 * bidan + beta2 * kemiskinan + beta3 * imunisasi
        log_mu = np.clip(log_mu, 0, 5)
        mu = np.exp(log_mu)
        y = np.random.poisson(mu)
        df = pd.DataFrame({
            'ID': [f'Kelurahan_{i+1}' for i in range(n)],
            'Longitude': np.round(lon, 6),
            'Latitude': np.round(lat, 6),
            'Kematian_Bayi': y,
            'Jumlah_Bidan': np.round(bidan, 1),
            'Pct_Miskin': np.round(kemiskinan, 2),
            'Pct_Imunisasi': np.round(imunisasi, 1),
            'Jumlah_Penduduk': np.round(penduduk, 0)
        })
        return df, "Kematian_Bayi", ["Jumlah_Bidan", "Pct_Miskin", "Pct_Imunisasi"], "Longitude", "Latitude"

    elif scenario == "Kecelakaan Lalu Lintas (Traffic Accidents)":
        n = 180
        lon = np.random.uniform(106.6, 107.0, n)
        lat = np.random.uniform(-6.95, -6.1, n)
        panjang_jalan = np.random.uniform(5, 200, n)
        kepadatan_kend = np.random.uniform(100, 5000, n)
        traffic_light = np.random.uniform(0, 30, n)
        beta0 = 2.0 + 0.4 * (lat + 6.5) + np.random.normal(0, 0.1, n)
        beta1 = 0.005 + 0.002 * np.sin(lon * 30) + np.random.normal(0, 0.001, n)
        beta2 = 0.0002 + 0.0001 * (lon - 106.8) * 10 + np.random.normal(0, 0.00005, n)
        beta3 = -0.03 + 0.01 * (lat + 6.5) + np.random.normal(0, 0.005, n)
        log_mu = beta0 + beta1 * panjang_jalan + beta2 * kepadatan_kend + beta3 * traffic_light
        log_mu = np.clip(log_mu, 0, 6)
        mu = np.exp(log_mu)
        y = np.random.poisson(mu)
        df = pd.DataFrame({
            'ID': [f'Ruas_{i+1}' for i in range(n)],
            'Longitude': np.round(lon, 6),
            'Latitude': np.round(lat, 6),
            'Jumlah_Kecelakaan': y,
            'Panjang_Jalan_km': np.round(panjang_jalan, 1),
            'Kepadatan_Kendaraan': np.round(kepadatan_kend, 0),
            'Jumlah_Traffic_Light': np.round(traffic_light, 0)
        })
        return df, "Jumlah_Kecelakaan", ["Panjang_Jalan_km", "Kepadatan_Kendaraan", "Jumlah_Traffic_Light"], "Longitude", "Latitude"

    else:  # Default
        n = 100
        lon = np.random.uniform(0, 10, n)
        lat = np.random.uniform(0, 10, n)
        x1 = np.random.uniform(1, 50, n)
        x2 = np.random.uniform(0, 30, n)
        beta0 = 1.5 + 0.2 * lon + np.random.normal(0, 0.1, n)
        beta1 = 0.02 + 0.01 * lat + np.random.normal(0, 0.005, n)
        beta2 = -0.01 + 0.005 * lon + np.random.normal(0, 0.003, n)
        log_mu = beta0 + beta1 * x1 + beta2 * x2
        log_mu = np.clip(log_mu, 0, 6)
        mu = np.exp(log_mu)
        y = np.random.poisson(mu)
        df = pd.DataFrame({
            'ID': [f'Loc_{i+1}' for i in range(n)],
            'Longitude': np.round(lon, 4),
            'Latitude': np.round(lat, 4),
            'Y_Count': y,
            'X1': np.round(x1, 2),
            'X2': np.round(x2, 2)
        })
        return df, "Y_Count", ["X1", "X2"], "Longitude", "Latitude"

# ============================================================
# HEADER
# ============================================================
st.markdown('<div class="main-header">🦠 GWPR Analysis Suite</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Geographically Weighted Poisson Regression — Analisis Data Count Spasial</div>', unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## ⚙️ Konfigurasi")

    data_source = st.radio("📂 Sumber Data:", ["Upload CSV/Excel", "Data Demo"], key="data_src")

    if data_source == "Data Demo":
        scenario = st.selectbox("Pilih Skenario Demo:", [
            "Kasus DBD (Dengue Fever)",
            "Kematian Bayi (Infant Mortality)",
            "Kecelakaan Lalu Lintas (Traffic Accidents)",
            "Default (Simulated)"
        ])
        demo_df, demo_dep, demo_indeps, demo_lon, demo_lat = generate_demo_data(scenario)
        st.session_state.data = demo_df
        st.success(f"✅ Data demo dimuat: {len(demo_df)} observasi")

    else:
        uploaded = st.file_uploader("Upload file CSV/Excel:", type=['csv', 'xlsx', 'xls'])
        if uploaded:
            try:
                if uploaded.name.endswith('.csv'):
                    sep = st.selectbox("Separator:", [",", ";", "\\t", "|"])
                    if sep == "\\t":
                        sep = "\t"
                    df_up = pd.read_csv(uploaded, sep=sep)
                else:
                    df_up = pd.read_excel(uploaded)
                st.session_state.data = df_up
                st.success(f"✅ {len(df_up)} baris × {len(df_up.columns)} kolom dimuat")
            except Exception as e:
                st.error(f"Error: {e}")

    df = st.session_state.data

    if df is not None:
        st.markdown("---")
        st.markdown("### 🎯 Pilih Variabel")

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if data_source == "Data Demo":
            default_dep = demo_dep
            default_indeps = demo_indeps
            default_lon = demo_lon
            default_lat = demo_lat
        else:
            default_dep = None
            default_indeps = []
            default_lon = None
            default_lat = None

        dep_var = st.selectbox("Variabel Dependen (Y count):",
                               numeric_cols,
                               index=numeric_cols.index(default_dep) if default_dep in numeric_cols else 0)
        st.session_state.dep_var = dep_var

        # Check if Y is count data
        if dep_var:
            y_vals = df[dep_var].dropna()
            is_integer = np.all(y_vals == y_vals.astype(int))
            is_nonneg = np.all(y_vals >= 0)
            if not is_integer or not is_nonneg:
                st.warning("⚠️ Variabel Y sebaiknya berupa data count (bilangan bulat ≥ 0) untuk regresi Poisson.")
            else:
                st.success(f"✅ Y adalah count data (range: {int(y_vals.min())}–{int(y_vals.max())})")

        remaining = [c for c in numeric_cols if c != dep_var]
        default_idx = [remaining.index(v) for v in default_indeps if v in remaining] if default_indeps else []
        indep_vars = st.multiselect("Variabel Independen (X):", remaining, default=[remaining[i] for i in default_idx])
        st.session_state.indep_vars = indep_vars

        lon_col = st.selectbox("Kolom Longitude:",
                               numeric_cols,
                               index=numeric_cols.index(default_lon) if default_lon in numeric_cols else 0)
        lat_col = st.selectbox("Kolom Latitude:",
                               numeric_cols,
                               index=numeric_cols.index(default_lat) if default_lat in numeric_cols else 0)
        st.session_state.lon_col = lon_col
        st.session_state.lat_col = lat_col

        # Offset (optional, for Poisson with exposure)
        offset_options = ["Tidak ada (None)"] + numeric_cols
        offset_sel = st.selectbox("Offset / Exposure (opsional):", offset_options,
                                  help="Gunakan log(populasi) atau log(exposure) sebagai offset untuk Poisson rate model.")
        st.session_state.offset_col = None if offset_sel == "Tidak ada (None)" else offset_sel

        st.markdown("---")
        st.markdown("### 🔧 Pengaturan GWPR")
        kernel_type = st.selectbox("Kernel:", ["bisquare", "gaussian", "exponential"], key="kernel")
        bw_fixed = st.checkbox("Fixed bandwidth", value=False, key="bwfixed")
        criterion = st.selectbox("Criterion bandwidth:", ["AICc", "AIC", "BIC", "CV"], key="criterion")
        search_method = st.selectbox("Search method:", ["golden_section", "interval"], key="search")
        st.session_state.kernel_type = kernel_type
        st.session_state.bw_fixed = bw_fixed
        st.session_state.criterion = criterion
        st.session_state.search_method = search_method

# ============================================================
# LANDING PAGE
# ============================================================
if st.session_state.data is None:
    st.markdown("""
    <div class="info-box">
    <b>🦠 Selamat datang di GWPR Analysis Suite!</b><br><br>
    Aplikasi ini melakukan analisis <b>Geographically Weighted Poisson Regression (GWPR)</b>
    untuk data count/diskrit yang memiliki variasi spasial.<br><br>
    <b>Fitur:</b>
    <ul>
        <li>Data Explorer & validasi data count</li>
        <li>Uji Overdispersi & Asumsi Poisson</li>
        <li>Regresi Poisson Global (GLM Baseline)</li>
        <li>Model GWPR (Poisson Lokal)</li>
        <li>Perbandingan Model Global vs Lokal</li>
        <li>Visualisasi Peta Koefisien Lokal</li>
        <li>Laporan & Export Lengkap</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    st.info("👈 Mulai dengan memilih sumber data di sidebar!")
    st.stop()

# ============================================================
# DATA LOADED → SHOW TABS
# ============================================================
df = st.session_state.data
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Data Explorer",
    "🔬 Uji Asumsi Poisson",
    "📈 Poisson Global (GLM)",
    "🌍 GWPR Model",
    "⚖️ Perbandingan Model",
    "🗺️ Visualisasi Peta",
    "📋 Laporan & Export"
])

# ============================================================
# TAB 1: DATA EXPLORER
# ============================================================
with tab1:
    st.markdown("## 📊 Data Explorer")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Jumlah Observasi", len(df))
    with col2:
        st.metric("Jumlah Variabel", len(df.columns))
    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())
    with col4:
        st.metric("Duplikat", df.duplicated().sum())

    st.markdown("### 📋 Preview Data")
    st.dataframe(df.head(20), use_container_width=True)

    st.markdown("### 📈 Statistik Deskriptif")
    desc = df.describe().round(4)
    st.dataframe(desc, use_container_width=True)

    # Count data validation
    if st.session_state.dep_var:
        dep = st.session_state.dep_var
        st.markdown(f"### 🔢 Validasi Data Count: `{dep}`")

        y_vals = df[dep].dropna()
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Mean", f"{y_vals.mean():.2f}")
        with col2:
            st.metric("Variance", f"{y_vals.var():.2f}")
        with col3:
            st.metric("Min", f"{int(y_vals.min())}")
        with col4:
            st.metric("Max", f"{int(y_vals.max())}")
        with col5:
            disp_ratio = y_vals.var() / y_vals.mean() if y_vals.mean() > 0 else 0
            st.metric("Var/Mean", f"{disp_ratio:.2f}")

        col1, col2 = st.columns(2)
        with col1:
            fig_hist = px.histogram(df, x=dep, nbins=30, title=f"Distribusi {dep}",
                                    marginal="box", color_discrete_sequence=["#8E44AD"])
            st.plotly_chart(fig_hist, use_container_width=True)

        with col2:
            # Compare with Poisson distribution
            from scipy.stats import poisson
            x_range = np.arange(0, int(y_vals.max()) + 1)
            pmf_vals = poisson.pmf(x_range, y_vals.mean())
            obs_freq = y_vals.value_counts().sort_index()
            obs_freq_norm = obs_freq / obs_freq.sum()

            fig_comp = go.Figure()
            fig_comp.add_trace(go.Bar(x=obs_freq_norm.index, y=obs_freq_norm.values,
                                      name="Observed", marker_color="#8E44AD", opacity=0.7))
            fig_comp.add_trace(go.Scatter(x=x_range, y=pmf_vals,
                                          mode='lines+markers', name="Poisson (theoretical)",
                                          line=dict(color='red', width=2)))
            fig_comp.update_layout(title=f"Observed vs Poisson Distribution (λ={y_vals.mean():.2f})",
                                   xaxis_title=dep, yaxis_title="Probability")
            st.plotly_chart(fig_comp, use_container_width=True)

        if disp_ratio > 1.5:
            st.markdown(f'''<div class="warning-box">⚠️ <b>Overdispersi terdeteksi!</b> Var/Mean = {disp_ratio:.2f} > 1.
            Data menunjukkan variasi lebih besar dari yang diharapkan model Poisson.
            Pertimbangkan model Quasi-Poisson atau Negative Binomial sebagai alternatif.</div>''', unsafe_allow_html=True)
        elif disp_ratio < 0.5:
            st.markdown(f'''<div class="info-box">ℹ️ <b>Underdispersi terdeteksi.</b> Var/Mean = {disp_ratio:.2f} < 1.
            Data mungkin lebih cocok dengan distribusi Binomial.</div>''', unsafe_allow_html=True)
        else:
            st.markdown(f'''<div class="success-box">✅ <b>Dispersi wajar.</b> Var/Mean = {disp_ratio:.2f} ≈ 1.
            Data cukup sesuai dengan asumsi Poisson.</div>''', unsafe_allow_html=True)

    # Spatial map
    if st.session_state.lon_col and st.session_state.lat_col:
        st.markdown("### 🗺️ Peta Lokasi Observasi")
        fig_map = px.scatter_mapbox(
            df, lon=st.session_state.lon_col, lat=st.session_state.lat_col,
            color=st.session_state.dep_var if st.session_state.dep_var else None,
            size_max=15, zoom=6, mapbox_style="open-street-map",
            title="Distribusi Spasial Observasi",
            color_continuous_scale="Viridis",
            hover_data=df.columns.tolist()[:8]
        )
        fig_map.update_layout(height=500)
        st.plotly_chart(fig_map, use_container_width=True)

    # Correlation
    st.markdown("### 🔗 Korelasi Antar Variabel")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    corr_vars = [st.session_state.dep_var] + st.session_state.indep_vars if st.session_state.dep_var else numeric_cols
    corr_vars = [v for v in corr_vars if v in df.columns]
    corr_matrix = df[corr_vars].corr()
    fig_corr = px.imshow(corr_matrix, text_auto=".3f", color_continuous_scale="RdBu_r",
                         title="Correlation Matrix", aspect="auto", zmin=-1, zmax=1)
    fig_corr.update_layout(height=500)
    st.plotly_chart(fig_corr, use_container_width=True)

# ============================================================
# TAB 2: UJI ASUMSI POISSON
# ============================================================
with tab2:
    st.markdown("## 🔬 Uji Asumsi & Kesesuaian Model Poisson")

    st.markdown('''<div class="info-box">
    <b>Asumsi utama regresi Poisson:</b>
    <ul>
        <li>Variabel dependen berupa data count (bilangan bulat ≥ 0)</li>
        <li>Mean = Variance (equidispersion)</li>
        <li>Observasi independen satu sama lain</li>
        <li>Log(μ) = Xβ (log-linear relationship)</li>
    </ul>
    GWPR merelaksasi asumsi stasioneritas spasial tetapi asumsi distribusi Poisson tetap penting.
    </div>''', unsafe_allow_html=True)

    if st.session_state.dep_var and st.session_state.indep_vars:
        dep = st.session_state.dep_var
        indeps = st.session_state.indep_vars
        y_data = df[dep].values
        X_data = df[indeps].values

        # ---- 1. Uji Distribusi ----
        st.markdown("### 1️⃣ Uji Distribusi Poisson")

        col1, col2, col3 = st.columns(3)

        # Chi-square goodness of fit
        from scipy.stats import chisquare, poisson, kstest
        obs_freq = pd.Series(y_data).value_counts().sort_index()
        lam = y_data.mean()
        exp_freq = np.array([poisson.pmf(k, lam) * len(y_data) for k in obs_freq.index])
        exp_freq = np.maximum(exp_freq, 1e-10)

        with col1:
            st.markdown("**Chi-Square Goodness of Fit**")
            try:
                chi2_stat, chi2_p = chisquare(obs_freq.values, f_exp=exp_freq)
                st.write(f"Statistic: `{chi2_stat:.4f}`")
                st.write(f"P-value: `{chi2_p:.6f}`")
                if chi2_p > 0.05:
                    st.markdown('<div class="success-box">✅ Data sesuai distribusi Poisson (p > 0.05)</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="warning-box">⚠️ Data tidak sesuai Poisson (p ≤ 0.05)</div>', unsafe_allow_html=True)
            except Exception as e:
                st.warning(f"Chi-square gagal: {e}")

        # KS Test
        with col2:
            st.markdown("**Kolmogorov-Smirnov Test**")
            try:
                # Compare with Poisson CDF
                ks_stat, ks_p = kstest(y_data, 'poisson', args=(lam,))
                st.write(f"Statistic: `{ks_stat:.6f}`")
                st.write(f"P-value: `{ks_p:.6f}`")
                if ks_p > 0.05:
                    st.markdown('<div class="success-box">✅ Sesuai Poisson (p > 0.05)</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="warning-box">⚠️ Tidak sesuai Poisson (p ≤ 0.05)</div>', unsafe_allow_html=True)
            except Exception as e:
                st.warning(f"KS test gagal: {e}")

        # Zero proportion
        with col3:
            st.markdown("**Zero-Inflation Check**")
            n_zeros = np.sum(y_data == 0)
            pct_zeros = 100 * n_zeros / len(y_data)
            expected_zeros = 100 * poisson.pmf(0, lam)
            st.write(f"Observed zeros: `{n_zeros}` ({pct_zeros:.1f}%)")
            st.write(f"Expected zeros (Poisson): `{expected_zeros:.1f}%`")
            if pct_zeros > expected_zeros * 2:
                st.markdown('<div class="warning-box">⚠️ Kemungkinan zero-inflated. Pertimbangkan ZIP model.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="success-box">✅ Proporsi zero wajar.</div>', unsafe_allow_html=True)

        st.markdown("---")

        # ---- 2. Uji Overdispersi ----
        st.markdown("### 2️⃣ Uji Overdispersi")

        st.markdown('''<div class="info-box">
        Overdispersi terjadi ketika variansi data lebih besar dari mean (Var > Mean).
        Dalam regresi Poisson, diasumsikan E(Y) = Var(Y). Jika dilanggar, standard error
        menjadi terlalu kecil dan inferensi tidak valid.
        </div>''', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        # Fit global Poisson first for dispersion test
        import statsmodels.api as sm
        from statsmodels.discrete.discrete_model import Poisson as smPoisson

        X_const = sm.add_constant(X_data)
        try:
            poisson_global = smPoisson(y_data, X_const).fit(disp=0)
            mu_hat = poisson_global.predict(X_const)

            # Cameron-Trivedi test for overdispersion
            pearson_resid = (y_data - mu_hat) / np.sqrt(mu_hat)
            pearson_chi2 = np.sum(pearson_resid ** 2)
            disp_param = pearson_chi2 / (len(y_data) - len(poisson_global.params))

            with col1:
                st.markdown("**Dispersion Parameter (Pearson Chi²/df)**")
                st.write(f"Pearson Chi²: `{pearson_chi2:.4f}`")
                st.write(f"df: `{len(y_data) - len(poisson_global.params)}`")
                st.write(f"Dispersion: `{disp_param:.4f}`")
                if disp_param < 1.5:
                    st.markdown('<div class="success-box">✅ Equidispersion (Dispersi ≈ 1)</div>', unsafe_allow_html=True)
                elif disp_param < 3:
                    st.markdown('<div class="warning-box">⚠️ Mild overdispersion (1.5-3). GWPR masih applicable.</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="error-box">❌ Severe overdispersion (>3). Pertimbangkan Negative Binomial atau QuasiPoisson.</div>', unsafe_allow_html=True)

            # Dean's test
            with col2:
                st.markdown("**Dean's Test for Overdispersion**")
                try:
                    aux_y = ((y_data - mu_hat) ** 2 - y_data) / mu_hat
                    aux_x = mu_hat
                    aux_model = sm.OLS(aux_y, sm.add_constant(aux_x)).fit()
                    dean_stat = aux_model.tvalues[1]
                    dean_p = aux_model.pvalues[1]
                    st.write(f"t-statistic: `{dean_stat:.4f}`")
                    st.write(f"P-value: `{dean_p:.6f}`")
                    if dean_p > 0.05:
                        st.markdown('<div class="success-box">✅ Tidak ada overdispersi signifikan (p > 0.05)</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="warning-box">⚠️ Overdispersi signifikan (p ≤ 0.05)</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.warning(f"Dean test gagal: {e}")

        except Exception as e:
            st.error(f"Gagal fit Poisson global: {e}")

        st.markdown("---")

        # ---- 3. Multikolinearitas ----
        st.markdown("### 3️⃣ Uji Multikolinearitas (VIF)")

        from statsmodels.stats.outliers_influence import variance_inflation_factor
        vif_data = pd.DataFrame()
        vif_data['Variabel'] = indeps
        vif_data['VIF'] = [variance_inflation_factor(X_const, i+1) for i in range(len(indeps))]
        vif_data['Tolerance'] = 1 / vif_data['VIF']
        vif_data['Status'] = vif_data['VIF'].apply(
            lambda x: '✅ Baik (< 5)' if x < 5 else ('⚠️ Moderat (5-10)' if x < 10 else '❌ Tinggi (> 10)'))
        st.dataframe(vif_data, use_container_width=True)

        fig_vif = px.bar(vif_data, x='Variabel', y='VIF', title="Variance Inflation Factor (VIF)",
                         color='VIF', color_continuous_scale='RdYlGn_r', text='VIF')
        fig_vif.add_hline(y=5, line_dash="dash", line_color="orange", annotation_text="Threshold = 5")
        fig_vif.add_hline(y=10, line_dash="dash", line_color="red", annotation_text="Threshold = 10")
        fig_vif.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        st.plotly_chart(fig_vif, use_container_width=True)

        st.markdown("---")

        # ---- 4. Spatial Autocorrelation ----
        st.markdown("### 4️⃣ Uji Autokorelasi Spasial (Moran's I)")
        st.markdown("**Moran's I pada Pearson Residual Poisson Global**")

        try:
            from libpysal.weights import KNN
            from esda.moran import Moran

            coords_arr = list(zip(df[st.session_state.lon_col].values, df[st.session_state.lat_col].values))
            w = KNN.from_array(np.array(coords_arr), k=8)
            w.transform = 'r'

            moran_obj = Moran(pearson_resid, w)
            st.write(f"Moran's I: `{moran_obj.I:.6f}`")
            st.write(f"Expected I: `{moran_obj.EI:.6f}`")
            st.write(f"Z-score: `{moran_obj.z_norm:.6f}`")
            st.write(f"P-value: `{moran_obj.p_norm:.6f}`")

            if moran_obj.p_norm < 0.05:
                st.markdown('''<div class="warning-box">
                <b>⚠️ Autokorelasi spasial terdeteksi!</b><br>
                Residual Poisson global menunjukkan pola spasial signifikan.
                Ini mengindikasikan bahwa GWPR sangat diperlukan untuk menangkap variasi spasial.
                </div>''', unsafe_allow_html=True)
            else:
                st.markdown('<div class="success-box">✅ Tidak ada autokorelasi spasial signifikan (p > 0.05)</div>', unsafe_allow_html=True)

        except ImportError:
            st.warning("Library `libpysal` dan `esda` diperlukan untuk Moran's I.")
        except Exception as e:
            st.warning(f"Moran's I gagal: {e}")

        st.markdown("---")

        # ---- 5. Ringkasan Asumsi ----
        st.markdown("### 📋 Ringkasan Uji Asumsi")
        try:
            assumption_summary = pd.DataFrame({
                'Uji': ['Distribusi Poisson (Chi²)', 'Overdispersi (Pearson)', 'Multikolinearitas (VIF max)',
                         "Autokorelasi Spasial (Moran's I)"],
                'Statistik': [f'{chi2_stat:.4f}', f'{disp_param:.4f}', f'{vif_data["VIF"].max():.4f}',
                              f'{moran_obj.I:.4f}' if 'moran_obj' in locals() else '-'],
                'P-value': [f'{chi2_p:.4f}', '-', '-',
                            f'{moran_obj.p_norm:.4f}' if 'moran_obj' in locals() else '-'],
                'Keputusan': [
                    'Sesuai Poisson' if chi2_p > 0.05 else 'Tidak Sesuai',
                    'Equidispersion' if disp_param < 1.5 else ('Mild Overdispersion' if disp_param < 3 else 'Severe Overdispersion'),
                    'Tidak Multikolinear' if vif_data['VIF'].max() < 10 else 'Multikolinear',
                    ('Autokorelasi' if moran_obj.p_norm < 0.05 else 'Tidak Autokorelasi') if 'moran_obj' in locals() else '-'
                ]
            })
            st.dataframe(assumption_summary, use_container_width=True)
        except Exception:
            st.info("Ringkasan tidak dapat ditampilkan. Jalankan semua uji terlebih dahulu.")

    else:
        st.warning("⚠️ Pilih variabel dependen dan independen terlebih dahulu di sidebar!")

# ============================================================
# TAB 3: POISSON GLOBAL (GLM BASELINE)
# ============================================================
with tab3:
    st.markdown("## 📈 Regresi Poisson Global (GLM Baseline)")

    if st.session_state.dep_var and st.session_state.indep_vars:
        dep = st.session_state.dep_var
        indeps = st.session_state.indep_vars

        y_pois = df[dep].values
        X_pois = df[indeps].values

        st.markdown('''<div class="info-box">
        <b>Regresi Poisson Global (GLM)</b> mengasumsikan hubungan log-linear:<br>
        <b>log(μ) = β₀ + β₁X₁ + β₂X₂ + ... + βₖXₖ</b><br><br>
        Model ini menjadi baseline sebelum GWPR. Jika koefisien bervariasi secara spasial,
        model global tidak cukup dan GWPR diperlukan.
        </div>''', unsafe_allow_html=True)

        import statsmodels.api as sm
        from statsmodels.discrete.discrete_model import Poisson as smPoisson

        X_const = sm.add_constant(X_pois)

        # Handle offset
        offset_arr = None
        if st.session_state.offset_col and st.session_state.offset_col in df.columns:
            offset_vals = df[st.session_state.offset_col].values
            offset_arr = np.log(np.maximum(offset_vals, 1e-10))
            st.info(f"📌 Menggunakan offset: log({st.session_state.offset_col})")

        try:
            if offset_arr is not None:
                poisson_model = smPoisson(y_pois, X_const, offset=offset_arr).fit(disp=0)
            else:
                poisson_model = smPoisson(y_pois, X_const).fit(disp=0)

            st.session_state.poisson_results = poisson_model
            mu_hat = poisson_model.predict(X_const)

            # ---- Model Summary ----
            st.markdown("### 📊 Model Summary")

            pseudo_r2 = compute_poisson_pseudo_r2(y_pois, mu_hat)
            deviance = compute_poisson_deviance(y_pois, mu_hat)
            pearson_chi2 = np.sum((y_pois - mu_hat) ** 2 / np.maximum(mu_hat, 1e-10))
            n_obs = len(y_pois)
            k_params = len(poisson_model.params)
            aic_val = poisson_model.aic
            bic_val = poisson_model.bic
            ll_val = poisson_model.llf
            aicc_val = aic_val + (2 * k_params * (k_params + 1)) / max(n_obs - k_params - 1, 1)
            disp_param = pearson_chi2 / (n_obs - k_params)

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Pseudo R² (McFadden)", f"{pseudo_r2:.6f}")
            with col2:
                st.metric("AIC", f"{aic_val:.4f}")
            with col3:
                st.metric("AICc", f"{aicc_val:.4f}")
            with col4:
                st.metric("BIC", f"{bic_val:.4f}")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Deviance", f"{deviance:.4f}")
            with col2:
                st.metric("Pearson Chi²", f"{pearson_chi2:.4f}")
            with col3:
                st.metric("Log-Likelihood", f"{ll_val:.4f}")
            with col4:
                st.metric("Dispersion (φ)", f"{disp_param:.4f}")

            st.session_state.poisson_aicc = aicc_val
            st.session_state.poisson_deviance = deviance

            # ---- Coefficient Table ----
            st.markdown("### 📊 Tabel Koefisien")
            _ci = poisson_model.conf_int()
            coef_df = pd.DataFrame({
                'Variable': ['Intercept'] + indeps,
                'Coefficient (β)': poisson_model.params.flatten(),
                'IRR (exp(β))': np.exp(poisson_model.params.flatten()),
                'Std. Error': poisson_model.bse.flatten(),
                'z-value': poisson_model.tvalues.flatten(),
                'P-value': poisson_model.pvalues.flatten(),
                'CI Lower (95%)': _ci.iloc[:, 0].values if hasattr(_ci, 'iloc') else _ci[:, 0],
                'CI Upper (95%)': _ci.iloc[:, 1].values if hasattr(_ci, 'iloc') else _ci[:, 1],
                'Significant': ['✅ Yes' if p < 0.05 else '❌ No' for p in poisson_model.pvalues.flatten()]
            })
            st.dataframe(coef_df.round(6), use_container_width=True)

            st.markdown('''<div class="info-box">
            <b>IRR (Incidence Rate Ratio) = exp(β)</b>: Jika IRR = 1.05, maka kenaikan 1 unit X
            meningkatkan expected count sebesar 5%. IRR < 1 berarti penurunan, IRR > 1 berarti peningkatan.
            </div>''', unsafe_allow_html=True)

            # ---- IRR Visualization ----
            st.markdown("### 📊 Incidence Rate Ratio (IRR)")
            irr_df = coef_df[coef_df['Variable'] != 'Intercept'].copy()
            irr_df['CI_Lower_IRR'] = np.exp(irr_df['CI Lower (95%)'])
            irr_df['CI_Upper_IRR'] = np.exp(irr_df['CI Upper (95%)'])

            fig_irr = go.Figure()
            fig_irr.add_trace(go.Scatter(
                x=irr_df['IRR (exp(β))'], y=irr_df['Variable'],
                mode='markers', marker=dict(size=12, color='#8E44AD'),
                error_x=dict(
                    type='data',
                    symmetric=False,
                    array=irr_df['CI_Upper_IRR'] - irr_df['IRR (exp(β))'],
                    arrayminus=irr_df['IRR (exp(β))'] - irr_df['CI_Lower_IRR']
                ),
                name='IRR'
            ))
            fig_irr.add_vline(x=1, line_dash="dash", line_color="red",
                              annotation_text="IRR = 1 (no effect)")
            fig_irr.update_layout(title="Incidence Rate Ratios dengan 95% CI",
                                  xaxis_title="IRR (exp(β))", yaxis_title="Variable", height=400)
            st.plotly_chart(fig_irr, use_container_width=True)

            # ---- Model Equation ----
            st.markdown("### ✏️ Persamaan Model Poisson Global")
            _p = poisson_model.params.flatten()
            equation = f"**log(μ)** = {_p[0]:.4f}"
            for i, var in enumerate(indeps):
                coef = _p[i + 1]
                sign = "+" if coef >= 0 else ""
                equation += f" {sign} {coef:.4f} × **{var}**"
            st.markdown(equation)

            try:
                latex_str = f"\\log(\\mu) = {_p[0]:.4f}"
                for _i, _v in enumerate(indeps):
                    _cv = _p[_i+1]
                    _s = "+" if _cv >= 0 else "-"
                    latex_str += f" {_s} {abs(_cv):.4f} \\cdot \\text{{{_v}}}"
                st.latex(latex_str)
            except Exception:
                pass

            # ---- Full Summary ----
            with st.expander("📄 Full Poisson GLM Summary"):
                st.text(poisson_model.summary().as_text())

            # ---- Diagnostic Plots ----
            st.markdown("### 📉 Diagnostic Plots")

            deviance_resid = poisson_model.resid_deviance
            pearson_resid = poisson_model.resid_pearson

            col1, col2 = st.columns(2)
            with col1:
                fig_fit = px.scatter(x=mu_hat, y=y_pois,
                                     title="Actual vs Predicted (μ̂)",
                                     labels={'x': 'Predicted (μ̂)', 'y': f'Actual ({dep})'},
                                     color_discrete_sequence=["#8E44AD"])
                min_v = min(mu_hat.min(), y_pois.min())
                max_v = max(mu_hat.max(), y_pois.max())
                fig_fit.add_trace(go.Scatter(x=[min_v, max_v], y=[min_v, max_v],
                                             mode='lines', line=dict(color='red', dash='dash'),
                                             name='Perfect Fit'))
                st.plotly_chart(fig_fit, use_container_width=True)

            with col2:
                fig_dev_resid = px.scatter(x=mu_hat, y=deviance_resid,
                                           title="Deviance Residuals vs Predicted",
                                           labels={'x': 'Predicted (μ̂)', 'y': 'Deviance Residuals'},
                                           color_discrete_sequence=["#E74C3C"])
                fig_dev_resid.add_hline(y=0, line_dash="dash", line_color="black")
                fig_dev_resid.add_hline(y=2, line_dash="dot", line_color="orange")
                fig_dev_resid.add_hline(y=-2, line_dash="dot", line_color="orange")
                st.plotly_chart(fig_dev_resid, use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                fig_pearson = px.histogram(x=pearson_resid, nbins=30,
                                           title="Distribusi Pearson Residuals",
                                           labels={'x': 'Pearson Residuals'}, marginal="box",
                                           color_discrete_sequence=["#27AE60"])
                st.plotly_chart(fig_pearson, use_container_width=True)

            with col2:
                from scipy.stats import poisson as poisson_dist
                max_count = int(y_pois.max())
                counts_range = np.arange(0, min(max_count + 1, 50))
                obs_counts = np.array([np.sum(y_pois == c) for c in counts_range])
                exp_counts = np.array([poisson_dist.pmf(c, mu_hat.mean()) * len(y_pois) for c in counts_range])

                fig_root = go.Figure()
                fig_root.add_trace(go.Bar(x=counts_range, y=obs_counts, name='Observed',
                                          marker_color='#8E44AD', opacity=0.7))
                fig_root.add_trace(go.Scatter(x=counts_range, y=exp_counts, name='Expected (Poisson)',
                                              mode='lines+markers', line=dict(color='red', width=2)))
                fig_root.update_layout(title="Observed vs Expected Counts",
                                       xaxis_title="Count", yaxis_title="Frequency")
                st.plotly_chart(fig_root, use_container_width=True)

        except Exception as e:
            st.error(f"Error fitting Poisson GLM: {e}")
            import traceback
            st.code(traceback.format_exc())

    else:
        st.warning("⚠️ Pilih variabel dependen dan independen terlebih dahulu!")

# ============================================================
# TAB 4: GWPR MODEL
# ============================================================
with tab4:
    st.markdown("## 🌍 Geographically Weighted Poisson Regression (GWPR)")

    if st.session_state.dep_var and st.session_state.indep_vars:
        dep = st.session_state.dep_var
        indeps = st.session_state.indep_vars

        y_gwpr = df[dep].values.reshape(-1, 1)
        X_gwpr = df[indeps].values

        st.markdown('''<div class="info-box">
        <b>GWPR</b> memperluas regresi Poisson dengan mengizinkan koefisien bervariasi secara spasial:<br>
        <b>log(μᵢ) = β₀(uᵢ,vᵢ) + β₁(uᵢ,vᵢ)X₁ᵢ + ... + βₖ(uᵢ,vᵢ)Xₖᵢ</b><br><br>
        Setiap lokasi (uᵢ,vᵢ) memiliki set koefisien lokal sendiri berdasarkan
        observasi terdekat yang diberi bobot kernel.
        </div>''', unsafe_allow_html=True)

        # Handle offset
        offset_arr = None
        if st.session_state.offset_col and st.session_state.offset_col in df.columns:
            offset_vals = df[st.session_state.offset_col].values.reshape(-1, 1)
            offset_arr = np.log(np.maximum(offset_vals, 1e-10))
            st.info(f"📌 Offset: log({st.session_state.offset_col})")

        run_gwpr = st.button("🚀 Jalankan GWPR Model", use_container_width=True, type="primary")

        if run_gwpr:
            with st.spinner("Mencari bandwidth optimal dan estimasi GWPR (Poisson)..."):
                try:
                    from mgwr.gwr import GWR
                    from mgwr.sel_bw import Sel_BW
                    from spglm.family import Poisson

                    coords_arr = np.array(list(zip(
                        df[st.session_state.lon_col].values,
                        df[st.session_state.lat_col].values
                    )))

                    kernel = st.session_state.kernel_type
                    fixed = st.session_state.bw_fixed
                    criterion_map = {'AICc': 'AICc', 'AIC': 'AIC', 'BIC': 'BIC', 'CV': 'CV'}
                    crit = criterion_map[st.session_state.criterion]

                    # Bandwidth selection for Poisson
                    gwpr_selector = Sel_BW(
                        coords_arr, y_gwpr, X_gwpr,
                        family=Poisson(),
                        offset=offset_arr,
                        kernel=kernel, fixed=fixed
                    )
                    gwpr_bw = gwpr_selector.search(criterion=crit,
                                                   search_method=st.session_state.search_method)

                    st.session_state.gwpr_bw = gwpr_bw
                    st.session_state.gwpr_selector = gwpr_selector

                    # Fit GWPR
                    gwpr_model = GWR(
                        coords_arr, y_gwpr, X_gwpr,
                        bw=gwpr_bw,
                        family=Poisson(),
                        offset=offset_arr,
                        kernel=kernel, fixed=fixed
                    )
                    gwpr_results = gwpr_model.fit()

                    st.session_state.gwpr_results = gwpr_results
                    st.session_state.gwpr_coords = coords_arr
                    st.session_state.y = y_gwpr

                    st.success(f"✅ GWPR berhasil diestimasi! Bandwidth optimal: {gwpr_bw}")

                except Exception as e:
                    st.error(f"Error: {e}")
                    import traceback
                    st.code(traceback.format_exc())
                    st.info("Pastikan library `mgwr` terinstall: `pip install mgwr`")

        # ---- Display results if available ----
        if st.session_state.gwpr_results is not None:
            gwpr_results = st.session_state.gwpr_results
            gwpr_bw = st.session_state.gwpr_bw

            # ---- Global Diagnostics ----
            st.markdown("### 📊 Diagnostik Global GWPR")

            # Compute diagnostics
            y_actual = df[dep].values
            mu_gwpr = gwpr_results.predy.flatten()
            mu_gwpr = np.maximum(mu_gwpr, 1e-10)
            pseudo_r2_gwpr = compute_poisson_pseudo_r2(y_actual, mu_gwpr)
            deviance_gwpr = compute_poisson_deviance(y_actual, mu_gwpr)
            pearson_chi2_gwpr = np.sum((y_actual - mu_gwpr) ** 2 / mu_gwpr)
            n_obs = len(y_actual)

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                bw_display = f"{gwpr_bw:.2f}" if isinstance(gwpr_bw, float) else f"{gwpr_bw}"
                st.metric("Bandwidth", bw_display)
            with col2:
                st.metric("Pseudo R² (McFadden)", f"{pseudo_r2_gwpr:.6f}")
            with col3:
                st.metric("AICc", f"{gwpr_results.aicc:.4f}")
            with col4:
                st.metric("AIC", f"{gwpr_results.aic:.4f}")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Deviance", f"{deviance_gwpr:.4f}")
            with col2:
                st.metric("Pearson Chi²", f"{pearson_chi2_gwpr:.4f}")
            with col3:
                enp = gwpr_results.ENP if hasattr(gwpr_results, 'ENP') else '-'
                st.metric("Eff. Parameters (ENP)", f"{enp:.4f}" if isinstance(enp, (int, float, np.floating)) else str(enp))
            with col4:
                enp_val = enp if isinstance(enp, (int, float, np.floating)) else len(indeps) + 1
                disp_gwpr = pearson_chi2_gwpr / max(n_obs - enp_val, 1)
                st.metric("Dispersion (φ)", f"{disp_gwpr:.4f}")

            # ---- Local Coefficient Summary ----
            st.markdown("### 📊 Ringkasan Koefisien Lokal")

            var_names = ['Intercept'] + indeps
            params = gwpr_results.params
            tvalues = gwpr_results.tvalues
            std_errs = gwpr_results.bse

            coef_summary = pd.DataFrame()
            for i, vn in enumerate(var_names):
                row = {
                    'Variable': vn,
                    'Mean': params[:, i].mean(),
                    'Std': params[:, i].std(),
                    'Min': params[:, i].min(),
                    'Q1 (25%)': np.percentile(params[:, i], 25),
                    'Median': np.median(params[:, i]),
                    'Q3 (75%)': np.percentile(params[:, i], 75),
                    'Max': params[:, i].max(),
                    'IQR': np.percentile(params[:, i], 75) - np.percentile(params[:, i], 25),
                    'Range': params[:, i].max() - params[:, i].min()
                }
                coef_summary = pd.concat([coef_summary, pd.DataFrame([row])], ignore_index=True)
            st.dataframe(coef_summary.round(6), use_container_width=True)

            # ---- IRR Local Summary ----
            st.markdown("### 📊 Ringkasan IRR Lokal (exp(β))")

            irr_params = np.exp(params)
            irr_summary = pd.DataFrame()
            for i, vn in enumerate(var_names):
                row = {
                    'Variable': vn,
                    'Mean IRR': irr_params[:, i].mean(),
                    'Min IRR': irr_params[:, i].min(),
                    'Median IRR': np.median(irr_params[:, i]),
                    'Max IRR': irr_params[:, i].max(),
                    'Std IRR': irr_params[:, i].std(),
                    '% IRR > 1': f"{100 * np.sum(irr_params[:, i] > 1) / len(irr_params):.1f}%",
                    '% IRR < 1': f"{100 * np.sum(irr_params[:, i] < 1) / len(irr_params):.1f}%"
                }
                irr_summary = pd.concat([irr_summary, pd.DataFrame([row])], ignore_index=True)
            st.dataframe(irr_summary.round(6), use_container_width=True)

            # ---- t-values Summary ----
            st.markdown("### 📊 Ringkasan t-values Lokal")

            tval_summary = pd.DataFrame()
            for i, vn in enumerate(var_names):
                n_sig = np.sum(np.abs(tvalues[:, i]) > 1.96)
                row = {
                    'Variable': vn,
                    'Mean |t|': np.abs(tvalues[:, i]).mean(),
                    'Min t': tvalues[:, i].min(),
                    'Max t': tvalues[:, i].max(),
                    'Significant (|t|>1.96)': f"{n_sig} ({100*n_sig/len(tvalues):.1f}%)",
                    'Positive Coef': f"{100*np.sum(params[:, i] > 0)/len(params):.1f}%",
                    'Negative Coef': f"{100*np.sum(params[:, i] < 0)/len(params):.1f}%"
                }
                tval_summary = pd.concat([tval_summary, pd.DataFrame([row])], ignore_index=True)
            st.dataframe(tval_summary, use_container_width=True)

            # ---- Coefficient Distribution Plots ----
            st.markdown("### 📊 Distribusi Koefisien Lokal")

            n_vars = len(var_names)
            cols_per_row = min(3, n_vars)
            rows_needed = (n_vars + cols_per_row - 1) // cols_per_row
            fig_coef = make_subplots(rows=rows_needed, cols=cols_per_row,
                                     subplot_titles=var_names)
            for i, vn in enumerate(var_names):
                r = i // cols_per_row + 1
                c = i % cols_per_row + 1
                fig_coef.add_trace(
                    go.Histogram(x=params[:, i], nbinsx=25, name=vn,
                                 marker_color=px.colors.qualitative.Set2[i % 8]),
                    row=r, col=c
                )
                # Add global Poisson coefficient line if available
                if st.session_state.poisson_results is not None:
                    global_coef = st.session_state.poisson_results.params[i]
                    fig_coef.add_vline(x=global_coef, line_dash="dash", line_color="red",
                                       row=r, col=c)

            fig_coef.update_layout(height=300 * rows_needed, showlegend=False,
                                   title_text="Distribusi Koefisien Lokal (garis merah = Poisson global)")
            st.plotly_chart(fig_coef, use_container_width=True)

            # ---- IRR Distribution ----
            st.markdown("### 📊 Distribusi IRR Lokal")

            fig_irr_local = make_subplots(rows=rows_needed, cols=cols_per_row,
                                          subplot_titles=[f"IRR: {vn}" for vn in var_names])
            for i, vn in enumerate(var_names):
                r = i // cols_per_row + 1
                c = i % cols_per_row + 1
                fig_irr_local.add_trace(
                    go.Histogram(x=irr_params[:, i], nbinsx=25, name=vn,
                                 marker_color=px.colors.qualitative.Pastel[i % 8]),
                    row=r, col=c
                )
                fig_irr_local.add_vline(x=1, line_dash="dash", line_color="red",
                                         row=r, col=c)
            fig_irr_local.update_layout(height=300 * rows_needed, showlegend=False,
                                        title_text="Distribusi IRR Lokal (garis merah = IRR=1 / no effect)")
            st.plotly_chart(fig_irr_local, use_container_width=True)

            # ---- Persamaan Model GWPR ----
            st.markdown("### ✏️ Persamaan Model GWPR")
            st.markdown("**Form umum GWPR:**")
            try:
                _gl = "\\log(\\mu_i) = \\beta_0(u_i, v_i)"
                for _j in range(len(indeps)):
                    _gl += f" + \\beta_{{{_j+1}}}(u_i, v_i) \\cdot X_{{{_j+1}i}}"
                st.latex(_gl)
            except Exception:
                st.markdown("**log(μᵢ) = β₀(uᵢ,vᵢ) + Σ βₖ(uᵢ,vᵢ)·Xₖᵢ**")

            st.markdown("**Koefisien rata-rata:**")
            try:
                _parts = [f"{params[:, 0].mean():.4f}"]
                for _j, _vn in enumerate(indeps):
                    _cm = params[:, _j+1].mean()
                    _sg = "+" if _cm >= 0 else "-"
                    _parts.append(f"{_sg} {abs(_cm):.4f} × {_vn}")
                st.markdown(f"**log(μ̂) = {' '.join(_parts)}**")
            except Exception:
                pass

            # ---- Diagnostic Plots ----
            st.markdown("### 📉 Diagnostic Plots GWPR")

            col1, col2 = st.columns(2)
            with col1:
                fig_gwpr_fit = px.scatter(x=mu_gwpr, y=y_actual,
                                          title="Actual vs Predicted GWPR",
                                          labels={'x': 'Predicted (μ̂)', 'y': f'Actual ({dep})'},
                                          color_discrete_sequence=["#27AE60"])
                min_v = min(mu_gwpr.min(), y_actual.min())
                max_v = max(mu_gwpr.max(), y_actual.max())
                fig_gwpr_fit.add_trace(go.Scatter(x=[min_v, max_v], y=[min_v, max_v],
                                                   mode='lines', line=dict(color='red', dash='dash'),
                                                   name='Perfect Fit'))
                st.plotly_chart(fig_gwpr_fit, use_container_width=True)

            with col2:
                resid_gwpr = y_actual - mu_gwpr
                std_resid_gwpr = resid_gwpr / np.sqrt(np.maximum(mu_gwpr, 1e-10))
                fig_gwpr_resid = px.scatter(x=mu_gwpr, y=std_resid_gwpr,
                                            title="Std Pearson Residuals vs Predicted",
                                            labels={'x': 'Predicted (μ̂)', 'y': 'Std Pearson Residuals'},
                                            color_discrete_sequence=["#E74C3C"])
                fig_gwpr_resid.add_hline(y=0, line_dash="dash", line_color="black")
                fig_gwpr_resid.add_hline(y=2, line_dash="dot", line_color="orange")
                fig_gwpr_resid.add_hline(y=-2, line_dash="dot", line_color="orange")
                st.plotly_chart(fig_gwpr_resid, use_container_width=True)

            # ---- Full Summary ----
            with st.expander("📄 Full GWPR Summary"):
                try:
                    import sys as _sys
                    _buf = io.StringIO()
                    _old = _sys.stdout
                    _sys.stdout = _buf
                    gwpr_results.summary()
                    _sys.stdout = _old
                    _t = _buf.getvalue()
                    st.text(_t if _t.strip() else "Summary tidak tersedia dalam format teks.")
                except Exception as _e:
                    try:
                        _sys.stdout = _sys.__stdout__
                    except Exception:
                        pass
                    st.warning(f"Summary error: {_e}")

            # ---- Export Local Parameters ----
            st.markdown("### 💾 Export Parameter Lokal")

            export_df = df.copy()
            for i, vn in enumerate(var_names):
                export_df[f'Beta_{vn}'] = params[:, i]
                export_df[f'IRR_{vn}'] = np.exp(params[:, i])
                export_df[f'SE_{vn}'] = std_errs[:, i]
                export_df[f'tval_{vn}'] = tvalues[:, i]
                export_df[f'Sig_{vn}'] = np.where(np.abs(tvalues[:, i]) > 1.96, 'Yes', 'No')

            export_df['Predicted_mu'] = mu_gwpr
            export_df['Residual'] = y_actual - mu_gwpr
            export_df['Pearson_Residual'] = (y_actual - mu_gwpr) / np.sqrt(np.maximum(mu_gwpr, 1e-10))

            csv_buffer = io.StringIO()
            export_df.to_csv(csv_buffer, index=False)
            st.download_button(
                label="📥 Download Parameter Lokal GWPR (CSV)",
                data=csv_buffer.getvalue(),
                file_name=f"gwpr_local_parameters_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )

    else:
        st.warning("⚠️ Pilih variabel terlebih dahulu!")

# ============================================================
# TAB 5: PERBANDINGAN MODEL
# ============================================================
with tab5:
    st.markdown("## ⚖️ Perbandingan Model: Poisson Global vs GWPR")

    has_poisson = st.session_state.poisson_results is not None
    has_gwpr = st.session_state.gwpr_results is not None

    if not (has_poisson or has_gwpr):
        st.warning("⚠️ Jalankan minimal satu model terlebih dahulu.")
    else:
        dep = st.session_state.dep_var
        indeps = st.session_state.indep_vars
        y_actual = df[dep].values

        # ---- Comparison Table ----
        st.markdown("### 📊 Tabel Perbandingan Model")

        comparison_data = []
        if has_poisson:
            pois = st.session_state.poisson_results
            mu_pois = pois.predict(sm.add_constant(df[indeps].values))
            pr2_pois = compute_poisson_pseudo_r2(y_actual, mu_pois)
            dev_pois = compute_poisson_deviance(y_actual, mu_pois)
            pchi2_pois = np.sum((y_actual - mu_pois) ** 2 / np.maximum(mu_pois, 1e-10))
            comparison_data.append({
                'Model': 'Poisson Global (GLM)',
                'Pseudo R² (McFadden)': f"{pr2_pois:.6f}",
                'AIC': f"{pois.aic:.4f}",
                'AICc': f"{st.session_state.get('poisson_aicc', pois.aic):.4f}",
                'BIC': f"{pois.bic:.4f}",
                'Deviance': f"{dev_pois:.4f}",
                'Pearson Chi²': f"{pchi2_pois:.4f}",
                'Log-Likelihood': f"{pois.llf:.4f}",
                'Parameters': f"{len(pois.params)}"
            })

        if has_gwpr:
            gwpr = st.session_state.gwpr_results
            mu_gwpr = gwpr.predy.flatten()
            mu_gwpr_safe = np.maximum(mu_gwpr, 1e-10)
            pr2_gwpr = compute_poisson_pseudo_r2(y_actual, mu_gwpr_safe)
            dev_gwpr = compute_poisson_deviance(y_actual, mu_gwpr_safe)
            pchi2_gwpr = np.sum((y_actual - mu_gwpr_safe) ** 2 / mu_gwpr_safe)
            enp_gwpr = gwpr.ENP if hasattr(gwpr, 'ENP') else len(indeps) + 1
            comparison_data.append({
                'Model': 'GWPR (Local)',
                'Pseudo R² (McFadden)': f"{pr2_gwpr:.6f}",
                'AIC': f"{gwpr.aic:.4f}",
                'AICc': f"{gwpr.aicc:.4f}",
                'BIC': f"{getattr(gwpr, 'bic', '-')}",
                'Deviance': f"{dev_gwpr:.4f}",
                'Pearson Chi²': f"{pchi2_gwpr:.4f}",
                'Log-Likelihood': '-',
                'Parameters (ENP)': f"{enp_gwpr:.2f}" if isinstance(enp_gwpr, (int, float, np.floating)) else str(enp_gwpr)
            })

        comp_df = pd.DataFrame(comparison_data)
        st.dataframe(comp_df, use_container_width=True)

        # ---- Model Selection Recommendation ----
        if has_poisson and has_gwpr:
            st.markdown("### 🏆 Rekomendasi Model")

            aicc_pois = st.session_state.get('poisson_aicc', pois.aic)
            aicc_gwpr = gwpr.aicc
            delta_aicc = aicc_pois - aicc_gwpr

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("ΔAIC (Global - GWPR)", f"{pois.aic - gwpr.aic:.4f}",
                          delta_color="normal")
            with col2:
                st.metric("ΔAICc (Global - GWPR)", f"{delta_aicc:.4f}",
                          delta_color="normal")
            with col3:
                dev_improvement = ((dev_pois - dev_gwpr) / dev_pois * 100) if dev_pois > 0 else 0
                st.metric("Deviance Improvement", f"{dev_improvement:.2f}%")

            if delta_aicc > 2:
                st.markdown(f'''<div class="success-box">
                ✅ <b>GWPR lebih baik dari Poisson Global!</b><br>
                ΔAICc = {delta_aicc:.2f} > 2, menunjukkan bukti kuat bahwa koefisien
                bervariasi secara spasial. GWPR direkomendasikan.
                </div>''', unsafe_allow_html=True)
            elif delta_aicc > 0:
                st.markdown(f'''<div class="warning-box">
                ⚠️ <b>GWPR sedikit lebih baik.</b><br>
                ΔAICc = {delta_aicc:.2f} (0–2), perbedaan marginal.
                Kedua model dapat dipertimbangkan.
                </div>''', unsafe_allow_html=True)
            else:
                st.markdown(f'''<div class="info-box">
                ℹ️ <b>Poisson Global cukup memadai.</b><br>
                ΔAICc = {delta_aicc:.2f} < 0, GWPR tidak memberikan peningkatan signifikan.
                Model global lebih parsimoni.
                </div>''', unsafe_allow_html=True)

        # ---- Predicted vs Actual Comparison ----
        st.markdown("### 📊 Predicted vs Actual — Semua Model")
        fig_pred = go.Figure()
        min_val = y_actual.min()
        max_val = y_actual.max()

        fig_pred.add_trace(go.Scatter(
            x=[min_val, max_val], y=[min_val, max_val],
            mode='lines', line=dict(color='black', dash='dash'), name='Perfect Fit'
        ))

        if has_poisson:
            fig_pred.add_trace(go.Scatter(
                x=mu_pois, y=y_actual, mode='markers', name='Poisson Global',
                marker=dict(color='blue', size=5, opacity=0.5)
            ))
        if has_gwpr:
            fig_pred.add_trace(go.Scatter(
                x=mu_gwpr, y=y_actual, mode='markers', name='GWPR',
                marker=dict(color='green', size=5, opacity=0.5)
            ))
        fig_pred.update_layout(title="Predicted vs Actual (All Models)",
                               xaxis_title="Predicted (μ̂)", yaxis_title=f"Actual ({dep})", height=500)
        st.plotly_chart(fig_pred, use_container_width=True)

        # ---- Residual Comparison ----
        st.markdown("### 📊 Distribusi Residual per Model")

        residual_data = []
        if has_poisson:
            pois_resid = (y_actual - mu_pois) / np.sqrt(np.maximum(mu_pois, 1e-10))
            for r in pois_resid:
                residual_data.append({'Model': 'Poisson Global', 'Pearson Residual': r})
        if has_gwpr:
            gwpr_resid = (y_actual - mu_gwpr) / np.sqrt(np.maximum(mu_gwpr, 1e-10))
            for r in gwpr_resid:
                residual_data.append({'Model': 'GWPR', 'Pearson Residual': r})

        if residual_data:
            resid_df = pd.DataFrame(residual_data)
            fig_resid_comp = px.box(resid_df, x='Model', y='Pearson Residual',
                                    title="Distribusi Pearson Residual per Model", color='Model')
            st.plotly_chart(fig_resid_comp, use_container_width=True)

        # ---- RMSE / MAE ----
        st.markdown("### 📊 Error Metrics")

        metrics_data = []
        if has_poisson:
            rmse_pois = np.sqrt(np.mean((y_actual - mu_pois) ** 2))
            mae_pois = np.mean(np.abs(y_actual - mu_pois))
            mape_pois = np.mean(np.abs((y_actual - mu_pois) / np.maximum(y_actual, 1))) * 100
            metrics_data.append({
                'Model': 'Poisson Global', 'RMSE': f"{rmse_pois:.4f}",
                'MAE': f"{mae_pois:.4f}", 'MAPE (%)': f"{mape_pois:.2f}"
            })
        if has_gwpr:
            rmse_gwpr = np.sqrt(np.mean((y_actual - mu_gwpr) ** 2))
            mae_gwpr = np.mean(np.abs(y_actual - mu_gwpr))
            mape_gwpr = np.mean(np.abs((y_actual - mu_gwpr) / np.maximum(y_actual, 1))) * 100
            metrics_data.append({
                'Model': 'GWPR', 'RMSE': f"{rmse_gwpr:.4f}",
                'MAE': f"{mae_gwpr:.4f}", 'MAPE (%)': f"{mape_gwpr:.2f}"
            })

        if metrics_data:
            metrics_df = pd.DataFrame(metrics_data)
            st.dataframe(metrics_df, use_container_width=True)

        # ---- Spatial Residual Comparison ----
        if has_poisson and has_gwpr and st.session_state.lon_col and st.session_state.lat_col:
            st.markdown("### 🗺️ Peta Residual — Global vs GWPR")

            col1, col2 = st.columns(2)
            with col1:
                resid_pois_map = y_actual - mu_pois
                fig_map_pois = px.scatter_mapbox(
                    df, lon=st.session_state.lon_col, lat=st.session_state.lat_col,
                    color=resid_pois_map, size=np.abs(resid_pois_map),
                    size_max=15, zoom=6, mapbox_style="open-street-map",
                    title="Residual Poisson Global",
                    color_continuous_scale="RdBu_r", color_continuous_midpoint=0
                )
                fig_map_pois.update_layout(height=400)
                st.plotly_chart(fig_map_pois, use_container_width=True)

            with col2:
                resid_gwpr_map = y_actual - mu_gwpr
                fig_map_gwpr = px.scatter_mapbox(
                    df, lon=st.session_state.lon_col, lat=st.session_state.lat_col,
                    color=resid_gwpr_map, size=np.abs(resid_gwpr_map),
                    size_max=15, zoom=6, mapbox_style="open-street-map",
                    title="Residual GWPR",
                    color_continuous_scale="RdBu_r", color_continuous_midpoint=0
                )
                fig_map_gwpr.update_layout(height=400)
                st.plotly_chart(fig_map_gwpr, use_container_width=True)

# ============================================================
# TAB 6: VISUALISASI PETA
# ============================================================
with tab6:
    st.markdown("## 🗺️ Visualisasi Peta Koefisien & Diagnostics")

    if not FOLIUM_AVAILABLE:
        st.error("Library `folium` dan `streamlit-folium` belum terinstall.")
        st.code("pip install folium streamlit-folium branca geopandas", language="bash")
    elif st.session_state.dep_var and st.session_state.indep_vars:
        dep = st.session_state.dep_var
        indeps = st.session_state.indep_vars
        lon_col = st.session_state.lon_col
        lat_col = st.session_state.lat_col

        model_options = []
        if st.session_state.poisson_results is not None:
            model_options.append("Poisson Global")
        if st.session_state.gwpr_results is not None:
            model_options.append("GWPR")

        if not model_options:
            st.warning("⚠️ Belum ada model yang dijalankan!")
        else:
            map_model = st.selectbox("Pilih model untuk divisualisasikan:", model_options)

            var_names = ['Intercept'] + indeps

            # Build map dataframe
            gdf = df.copy()

            if map_model == "Poisson Global":
                pois = st.session_state.poisson_results
                mu_pois = pois.predict(sm.add_constant(df[indeps].values))
                gdf['Predicted'] = mu_pois
                gdf['Residual'] = df[dep].values - mu_pois
                gdf['Pearson_Residual'] = gdf['Residual'] / np.sqrt(np.maximum(mu_pois, 1e-10))

            else:  # GWPR
                gwpr = st.session_state.gwpr_results
                params_g = gwpr.params
                tvals_g = gwpr.tvalues
                mu_gwpr_map = gwpr.predy.flatten()

                for i, vn in enumerate(var_names):
                    gdf[f'GWPR_Beta_{vn}'] = params_g[:, i]
                    gdf[f'GWPR_IRR_{vn}'] = np.exp(params_g[:, i])
                    gdf[f'GWPR_t_{vn}'] = tvals_g[:, i]
                    gdf[f'GWPR_Sig_{vn}'] = np.where(np.abs(tvals_g[:, i]) > 1.96, 'Signifikan', 'Tidak')

                gdf['Predicted'] = mu_gwpr_map
                gdf['Residual'] = df[dep].values - mu_gwpr_map
                gdf['Pearson_Residual'] = gdf['Residual'] / np.sqrt(np.maximum(mu_gwpr_map, 1e-10))

            # Layer selection
            layer_options = ["Predicted vs Observed", "Residual (Pearson)"]
            if map_model == "GWPR":
                for v in var_names:
                    layer_options.append(f"Koefisien β: {v}")
                    layer_options.append(f"IRR: {v}")
                    layer_options.append(f"t-stat: {v}")

            layer_type = st.selectbox("Pilih layer:", layer_options)

            # Create map
            center_lat = df[lat_col].mean()
            center_lon = df[lon_col].mean()
            m = folium.Map(location=[center_lat, center_lon], zoom_start=8,
                           tiles='CartoDB positron')

            if layer_type == "Predicted vs Observed":
                max_abs_res = max(np.abs(gdf['Residual']).max(), 1e-10)
                for _, row in gdf.iterrows():
                    res = row['Residual']
                    color = 'green' if res >= 0 else 'red'
                    radius = 4 + 8 * abs(res) / max_abs_res
                    popup = folium.Popup(
                        f"<b>{dep}</b>: {row[dep]}<br>"
                        f"<b>Predicted</b>: {row['Predicted']:.2f}<br>"
                        f"<b>Residual</b>: {res:.2f}",
                        max_width=250
                    )
                    folium.CircleMarker(
                        location=[row[lat_col], row[lon_col]],
                        radius=radius, color=color, fill=True,
                        fill_color=color, fill_opacity=0.7, popup=popup
                    ).add_to(m)

            elif layer_type == "Residual (Pearson)":
                series = gdf['Pearson_Residual']
                cmap = create_colormap_safe(series, cmap_name='RdBu_r')
                for _, row in gdf.iterrows():
                    val = row['Pearson_Residual']
                    try:
                        color = cmap(float(val)) if pd.notna(val) else '#808080'
                    except Exception:
                        color = '#808080'
                    popup = folium.Popup(
                        f"<b>{dep}</b>: {row[dep]}<br>"
                        f"<b>Pearson Residual</b>: {val:.3f}",
                        max_width=250
                    )
                    folium.CircleMarker(
                        location=[row[lat_col], row[lon_col]],
                        radius=6, color=color, fill=True,
                        fill_color=color, fill_opacity=0.8, popup=popup
                    ).add_to(m)
                cmap.caption = "Pearson Residual"
                m.add_child(cmap)

            elif layer_type.startswith("Koefisien"):
                var = layer_type.replace("Koefisien β: ", "")
                field = f'GWPR_Beta_{var}'
                if field in gdf.columns:
                    series = gdf[field]
                    cmap = create_colormap_safe(series, cmap_name='RdYlBu')
                    for _, row in gdf.iterrows():
                        val = row[field]
                        try:
                            color = cmap(float(val)) if pd.notna(val) else '#808080'
                        except Exception:
                            color = '#808080'
                        sig_status = row.get(f'GWPR_Sig_{var}', '-')
                        popup = folium.Popup(
                            f"<b>{var}</b><br>"
                            f"β = {val:.4f}<br>"
                            f"IRR = {np.exp(val):.4f}<br>"
                            f"Signifikan: {sig_status}",
                            max_width=250
                        )
                        folium.CircleMarker(
                            location=[row[lat_col], row[lon_col]],
                            radius=6, color=color, fill=True,
                            fill_color=color, fill_opacity=0.8, popup=popup
                        ).add_to(m)
                    cmap.caption = f"β ({var})"
                    m.add_child(cmap)

            elif layer_type.startswith("IRR"):
                var = layer_type.replace("IRR: ", "")
                field = f'GWPR_IRR_{var}'
                if field in gdf.columns:
                    series = gdf[field]
                    cmap = create_colormap_safe(series, cmap_name='RdYlGn')
                    for _, row in gdf.iterrows():
                        val = row[field]
                        try:
                            color = cmap(float(val)) if pd.notna(val) else '#808080'
                        except Exception:
                            color = '#808080'
                        popup = folium.Popup(
                            f"<b>IRR {var}</b><br>"
                            f"IRR = {val:.4f}<br>"
                            f"Efek: {'Meningkatkan' if val > 1 else 'Menurunkan'} count",
                            max_width=250
                        )
                        folium.CircleMarker(
                            location=[row[lat_col], row[lon_col]],
                            radius=6, color=color, fill=True,
                            fill_color=color, fill_opacity=0.8, popup=popup
                        ).add_to(m)
                    cmap.caption = f"IRR ({var})"
                    m.add_child(cmap)

            elif layer_type.startswith("t-stat"):
                var = layer_type.replace("t-stat: ", "")
                field = f'GWPR_t_{var}'
                if field in gdf.columns:
                    series = gdf[field]
                    cmap = create_colormap_safe(series, cmap_name='PiYG')
                    for _, row in gdf.iterrows():
                        val = row[field]
                        try:
                            color = cmap(float(val)) if pd.notna(val) else '#808080'
                        except Exception:
                            color = '#808080'
                        sig = "✅ Ya" if abs(val) > 1.96 else "❌ Tidak"
                        popup = folium.Popup(
                            f"<b>t-stat {var}</b><br>"
                            f"t = {val:.4f}<br>"
                            f"Signifikan (α=5%): {sig}",
                            max_width=250
                        )
                        folium.CircleMarker(
                            location=[row[lat_col], row[lon_col]],
                            radius=6, color=color, fill=True,
                            fill_color=color, fill_opacity=0.8, popup=popup
                        ).add_to(m)
                    cmap.caption = f"t-stat ({var})"
                    m.add_child(cmap)

            folium.LayerControl().add_to(m)
            st_folium(m, width=1200, height=600)

    else:
        st.warning("⚠️ Pilih variabel terlebih dahulu!")

# ============================================================
# TAB 7: LAPORAN & EXPORT
# ============================================================
with tab7:
    st.markdown("## 📋 Laporan & Export")

    if st.session_state.dep_var and st.session_state.indep_vars:
        dep = st.session_state.dep_var
        indeps = st.session_state.indep_vars

        st.markdown("### 📝 Generate Laporan Otomatis")

        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("LAPORAN ANALISIS GWPR")
        report_lines.append(f"Tanggal: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("=" * 60)
        report_lines.append("")
        report_lines.append("1. DATA")
        report_lines.append(f"   Jumlah observasi: {len(df)}")
        report_lines.append(f"   Variabel dependen (Y): {dep}")
        report_lines.append(f"   Variabel independen (X): {', '.join(indeps)}")
        report_lines.append(f"   Koordinat: {st.session_state.lon_col}, {st.session_state.lat_col}")
        if st.session_state.offset_col:
            report_lines.append(f"   Offset: log({st.session_state.offset_col})")
        report_lines.append("")

        y_vals = df[dep].values
        report_lines.append("2. DESKRIPSI VARIABEL DEPENDEN")
        report_lines.append(f"   Mean      : {y_vals.mean():.4f}")
        report_lines.append(f"   Variance  : {y_vals.var():.4f}")
        report_lines.append(f"   Var/Mean  : {y_vals.var()/max(y_vals.mean(),1e-10):.4f}")
        report_lines.append(f"   Min, Max  : ({y_vals.min()}, {y_vals.max()})")
        report_lines.append(f"   Zeros     : {np.sum(y_vals == 0)} ({100*np.sum(y_vals==0)/len(y_vals):.1f}%)")
        report_lines.append("")

        # Poisson Global
        if st.session_state.poisson_results is not None:
            pois = st.session_state.poisson_results
            mu_p = pois.predict(sm.add_constant(df[indeps].values))
            report_lines.append("3. MODEL POISSON GLOBAL (GLM)")
            report_lines.append(f"   AIC       : {pois.aic:.4f}")
            report_lines.append(f"   BIC       : {pois.bic:.4f}")
            report_lines.append(f"   AICc      : {st.session_state.get('poisson_aicc', pois.aic):.4f}")
            report_lines.append(f"   Deviance  : {compute_poisson_deviance(y_vals, mu_p):.4f}")
            report_lines.append(f"   Pseudo R² : {compute_poisson_pseudo_r2(y_vals, mu_p):.6f}")
            report_lines.append(f"   Log-Lik   : {pois.llf:.4f}")
            report_lines.append("")
            report_lines.append("   Koefisien Global:")
            var_names_g = ['Intercept'] + indeps
            for i, vn in enumerate(var_names_g):
                p_val = pois.pvalues[i]
                sig = "*" if p_val < 0.05 else ""
                report_lines.append(
                    f"   {vn:25s} β={pois.params[i]:.6f}  IRR={np.exp(pois.params[i]):.4f}"
                    f"  SE={pois.bse[i]:.6f}  z={pois.tvalues[i]:.4f}  p={p_val:.4f} {sig}"
                )
            report_lines.append("")

        # GWPR
        if st.session_state.gwpr_results is not None:
            gwpr = st.session_state.gwpr_results
            mu_gw = gwpr.predy.flatten()
            mu_gw_safe = np.maximum(mu_gw, 1e-10)
            report_lines.append("4. MODEL GWPR (Poisson Lokal)")
            report_lines.append(f"   Bandwidth : {st.session_state.gwpr_bw}")
            report_lines.append(f"   AICc      : {gwpr.aicc:.4f}")
            report_lines.append(f"   AIC       : {gwpr.aic:.4f}")
            report_lines.append(f"   Deviance  : {compute_poisson_deviance(y_vals, mu_gw_safe):.4f}")
            report_lines.append(f"   Pseudo R² : {compute_poisson_pseudo_r2(y_vals, mu_gw_safe):.6f}")
            enp = gwpr.ENP if hasattr(gwpr, 'ENP') else '-'
            report_lines.append(f"   ENP       : {enp}")
            report_lines.append("")

            var_names_l = ['Intercept'] + indeps
            params_l = gwpr.params
            tvals_l = gwpr.tvalues

            report_lines.append("   Ringkasan Koefisien Lokal:")
            report_lines.append(f"   {'Variable':20s} {'Mean':>10s} {'Std':>10s} {'Min':>10s} {'Median':>10s} {'Max':>10s} {'%Sig':>8s}")
            for i, vn in enumerate(var_names_l):
                n_sig = np.sum(np.abs(tvals_l[:, i]) > 1.96)
                pct_sig = 100 * n_sig / len(tvals_l)
                report_lines.append(
                    f"   {vn:20s} {params_l[:,i].mean():>10.4f} {params_l[:,i].std():>10.4f}"
                    f" {params_l[:,i].min():>10.4f} {np.median(params_l[:,i]):>10.4f}"
                    f" {params_l[:,i].max():>10.4f} {pct_sig:>7.1f}%"
                )
            report_lines.append("")

            report_lines.append("   Ringkasan IRR Lokal:")
            report_lines.append(f"   {'Variable':20s} {'Mean IRR':>10s} {'Min IRR':>10s} {'Max IRR':>10s} {'%IRR>1':>8s}")
            irr_l = np.exp(params_l)
            for i, vn in enumerate(var_names_l):
                pct_gt1 = 100 * np.sum(irr_l[:, i] > 1) / len(irr_l)
                report_lines.append(
                    f"   {vn:20s} {irr_l[:,i].mean():>10.4f} {irr_l[:,i].min():>10.4f}"
                    f" {irr_l[:,i].max():>10.4f} {pct_gt1:>7.1f}%"
                )
            report_lines.append("")

        # Perbandingan
        if st.session_state.poisson_results is not None and st.session_state.gwpr_results is not None:
            report_lines.append("5. PERBANDINGAN MODEL")
            aicc_p = st.session_state.get('poisson_aicc', pois.aic)
            aicc_g = gwpr.aicc
            report_lines.append(f"   AICc Global  : {aicc_p:.4f}")
            report_lines.append(f"   AICc GWPR    : {aicc_g:.4f}")
            report_lines.append(f"   Delta AICc   : {aicc_p - aicc_g:.4f}")
            winner = "GWPR" if aicc_g < aicc_p else "Poisson Global"
            report_lines.append(f"   Model terbaik: {winner}")
            report_lines.append("")

        # Interpretasi
        report_lines.append("CATATAN INTERPRETASI")
        report_lines.append("- GWPR menggunakan family=Poisson dengan link function log.")
        report_lines.append("- Koefisien β interpretasi: kenaikan 1 unit X mengubah log(μ) sebesar β.")
        report_lines.append("- IRR = exp(β): kenaikan 1 unit X mengalikan expected count sebesar IRR.")
        report_lines.append("- IRR > 1 = meningkatkan count, IRR < 1 = menurunkan count.")
        report_lines.append("- |t| > 1.96 → koefisien signifikan pada α ≈ 5%.")
        report_lines.append("- Bandwidth kecil → hubungan sangat lokal, bandwidth besar → mendekati global.")
        report_lines.append("- Overdispersi (Var/Mean >> 1) dapat menyebabkan underestimasi SE.")

        report_text = "\n".join(report_lines)

        st.text_area("Preview Laporan", report_text, height=400)

        st.download_button(
            label="📥 Download Laporan (.txt)",
            data=report_text,
            file_name=f"laporan_GWPR_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True
        )

        # ---- Export Configuration JSON ----
        st.markdown("### ⚙️ Export Konfigurasi Model")
        config = {
            "analysis": "GWPR",
            "timestamp": datetime.now().isoformat(),
            "data": {
                "
