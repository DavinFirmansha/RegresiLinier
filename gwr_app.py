# ============================================================
# FILE 1: gwr_app_part1.py
# ============================================================


# ============================================================
# APLIKASI ANALISIS GWR LENGKAP - BAGIAN 1/3
# File: gwr_app.py
# Jalankan: streamlit run gwr_app.py
# ============================================================
# INSTALL DULU:
# pip install streamlit pandas numpy scipy matplotlib seaborn
# pip install statsmodels scikit-learn mgwr libpysal geopandas
# pip install folium streamlit-folium plotly branca esda spreg
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
    page_title="GWR Analysis Suite",
    page_icon="🌍",
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
        border-bottom: 3px solid #2E86C1;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #5D6D7E !important;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
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
if 'data' not in st.session_state:
    st.session_state.data = None
if 'gwr_results' not in st.session_state:
    st.session_state.gwr_results = None
if 'mgwr_results' not in st.session_state:
    st.session_state.mgwr_results = None
if 'ols_results' not in st.session_state:
    st.session_state.ols_results = None
if 'coords' not in st.session_state:
    st.session_state.coords = None
if 'y' not in st.session_state:
    st.session_state.y = None
if 'X' not in st.session_state:
    st.session_state.X = None
if 'var_names' not in st.session_state:
    st.session_state.var_names = None
if 'dep_var' not in st.session_state:
    st.session_state.dep_var = None
if 'gwr_bw' not in st.session_state:
    st.session_state.gwr_bw = None
if 'gwr_selector' not in st.session_state:
    st.session_state.gwr_selector = None
if 'indep_vars' not in st.session_state:
    st.session_state.indep_vars = []
if 'lon_col' not in st.session_state:
    st.session_state.lon_col = None
if 'lat_col' not in st.session_state:
    st.session_state.lat_col = None

# ============================================================
# DEMO DATA GENERATOR
# ============================================================
def generate_demo_data(scenario="default"):
    """Generate demo spatial data for GWR analysis"""
    np.random.seed(42)

    if scenario == "Georgia (Classic GWR Dataset)":
        # Simulated Georgia-like county data
        n = 159
        # Generate coordinates resembling Georgia state
        lon = np.random.uniform(-85.5, -80.8, n)
        lat = np.random.uniform(30.4, 35.0, n)

        # Spatially varying coefficients
        beta0 = 10 + 2 * (lat - 32) + np.random.normal(0, 0.5, n)
        beta1 = 0.5 + 0.3 * np.sin((lon + 83) * 2) + np.random.normal(0, 0.1, n)
        beta2 = -0.3 + 0.2 * (lat - 32) / 4 + np.random.normal(0, 0.1, n)
        beta3 = -0.1 - 0.15 * np.cos((lon + 83) * 1.5) + np.random.normal(0, 0.05, n)

        PctFB = np.random.uniform(0.5, 15, n)      # % Foreign Born
        PctBlack = np.random.uniform(5, 70, n)       # % African American
        PctRural = np.random.uniform(0, 100, n)      # % Rural

        PctBach = beta0 + beta1 * PctFB + beta2 * PctBlack + beta3 * PctRural
        PctBach = np.clip(PctBach + np.random.normal(0, 1.5, n), 3, 45)

        df = pd.DataFrame({
            'ID': [f'County_{i+1}' for i in range(n)],
            'Longitude': np.round(lon, 4),
            'Latitude': np.round(lat, 4),
            'PctBach': np.round(PctBach, 2),
            'PctFB': np.round(PctFB, 2),
            'PctBlack': np.round(PctBlack, 2),
            'PctRural': np.round(PctRural, 2)
        })
        return df, "PctBach", ["PctFB", "PctBlack", "PctRural"], "Longitude", "Latitude"

    elif scenario == "Harga Rumah (Real Estate)":
        n = 200
        lon = np.random.uniform(106.6, 107.0, n)  # Surabaya-like
        lat = np.random.uniform(-7.4, -7.1, n)

        dist_center = np.sqrt((lon - 106.8)**2 + (lat + 7.25)**2)
        beta0 = 500 - 800 * dist_center + np.random.normal(0, 20, n)
        beta1 = 2.5 + 1.5 * np.sin(lon * 50) + np.random.normal(0, 0.3, n)
        beta2 = 0.8 + 0.5 * (lat + 7.25) * 10 + np.random.normal(0, 0.1, n)
        beta3 = 15 - 10 * dist_center + np.random.normal(0, 2, n)

        luas = np.random.uniform(30, 300, n)
        kamar = np.random.randint(1, 6, n).astype(float)
        jarak_sekolah = np.random.uniform(0.1, 5, n)

        harga = beta0 + beta1 * luas + beta2 * kamar + beta3 * jarak_sekolah
        harga = np.clip(harga + np.random.normal(0, 30, n), 100, 2000)

        df = pd.DataFrame({
            'ID': [f'Property_{i+1}' for i in range(n)],
            'Longitude': np.round(lon, 6),
            'Latitude': np.round(lat, 6),
            'Harga_Juta': np.round(harga, 1),
            'Luas_m2': np.round(luas, 1),
            'Jumlah_Kamar': kamar,
            'Jarak_Sekolah_km': np.round(jarak_sekolah, 2)
        })
        return df, "Harga_Juta", ["Luas_m2", "Jumlah_Kamar", "Jarak_Sekolah_km"], "Longitude", "Latitude"

    elif scenario == "Kemiskinan (Poverty Analysis)":
        n = 150
        lon = np.random.uniform(110.3, 114.6, n)  # Jawa Timur
        lat = np.random.uniform(-8.2, -6.8, n)

        beta0 = 25 + 5 * (lon - 112) + np.random.normal(0, 1, n)
        beta1 = -0.3 + 0.1 * np.sin(lat * 10) + np.random.normal(0, 0.05, n)
        beta2 = -0.15 + 0.08 * (lon - 112) + np.random.normal(0, 0.03, n)
        beta3 = 0.2 - 0.1 * (lat + 7.5) + np.random.normal(0, 0.04, n)

        pendidikan = np.random.uniform(4, 13, n)
        pengeluaran = np.random.uniform(500, 3000, n)
        pengangguran = np.random.uniform(2, 15, n)

        kemiskinan = beta0 + beta1 * pendidikan + beta2 * pengeluaran/100 + beta3 * pengangguran
        kemiskinan = np.clip(kemiskinan + np.random.normal(0, 2, n), 3, 40)

        df = pd.DataFrame({
            'ID': [f'Kecamatan_{i+1}' for i in range(n)],
            'Longitude': np.round(lon, 4),
            'Latitude': np.round(lat, 4),
            'Pct_Miskin': np.round(kemiskinan, 2),
            'Rata_Pendidikan_Thn': np.round(pendidikan, 1),
            'Pengeluaran_Ribu': np.round(pengeluaran, 0),
            'Pct_Pengangguran': np.round(pengangguran, 2)
        })
        return df, "Pct_Miskin", ["Rata_Pendidikan_Thn", "Pengeluaran_Ribu", "Pct_Pengangguran"], "Longitude", "Latitude"

    elif scenario == "Kesehatan (Health Epidemiology)":
        n = 180
        lon = np.random.uniform(106.6, 107.1, n)
        lat = np.random.uniform(-6.9, -6.1, n)

        beta0 = 50 + 20 * np.sin(lon * 30) + np.random.normal(0, 3, n)
        beta1 = 0.5 + 0.3 * (lat + 6.5) + np.random.normal(0, 0.1, n)
        beta2 = -2.0 + 1.0 * np.cos(lon * 20) + np.random.normal(0, 0.3, n)
        beta3 = 0.8 - 0.4 * (lon - 106.85) * 10 + np.random.normal(0, 0.1, n)

        kepadatan = np.random.uniform(500, 20000, n)
        puskesmas = np.random.uniform(0.5, 10, n)
        sanitasi = np.random.uniform(20, 95, n)

        kasus_dbd = beta0 + beta1 * kepadatan/1000 + beta2 * puskesmas + beta3 * sanitasi/10
        kasus_dbd = np.clip(kasus_dbd + np.random.normal(0, 5, n), 5, 200)

        df = pd.DataFrame({
            'ID': [f'Kelurahan_{i+1}' for i in range(n)],
            'Longitude': np.round(lon, 6),
            'Latitude': np.round(lat, 6),
            'Kasus_DBD': np.round(kasus_dbd, 0),
            'Kepadatan_per_km2': np.round(kepadatan, 0),
            'Rasio_Puskesmas': np.round(puskesmas, 2),
            'Pct_Sanitasi_Baik': np.round(sanitasi, 1)
        })
        return df, "Kasus_DBD", ["Kepadatan_per_km2", "Rasio_Puskesmas", "Pct_Sanitasi_Baik"], "Longitude", "Latitude"

    else:  # Custom/Default
        n = 120
        lon = np.random.uniform(0, 10, n)
        lat = np.random.uniform(0, 10, n)
        x1 = np.random.normal(50, 10, n)
        x2 = np.random.normal(30, 5, n)
        beta0 = 10 + 2 * lon + np.random.normal(0, 1, n)
        beta1 = 0.5 + 0.1 * lat + np.random.normal(0, 0.05, n)
        beta2 = -0.3 + 0.05 * lon + np.random.normal(0, 0.05, n)
        y_val = beta0 + beta1 * x1 + beta2 * x2 + np.random.normal(0, 3, n)

        df = pd.DataFrame({
            'ID': [f'Loc_{i+1}' for i in range(n)],
            'Longitude': np.round(lon, 4),
            'Latitude': np.round(lat, 4),
            'Y': np.round(y_val, 2),
            'X1': np.round(x1, 2),
            'X2': np.round(x2, 2)
        })
        return df, "Y", ["X1", "X2"], "Longitude", "Latitude"


# ============================================================
# HEADER
# ============================================================
st.markdown('<div class="main-header">🌍 GWR Analysis Suite</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Geographically Weighted Regression — Comprehensive Spatial Analysis Platform</div>', unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/globe-earth.png", width=80)
    st.markdown("## ⚙️ Pengaturan")

    st.markdown("---")
    st.markdown("### 📊 Sumber Data")
    data_source = st.radio(
        "Pilih sumber data:",
        ["📦 Data Demo", "📁 Upload CSV", "📁 Upload Excel"],
        index=0
    )

    if data_source == "📦 Data Demo":
        demo_scenario = st.selectbox(
            "Pilih skenario demo:",
            [
                "Georgia (Classic GWR Dataset)",
                "Harga Rumah (Real Estate)",
                "Kemiskinan (Poverty Analysis)",
                "Kesehatan (Health Epidemiology)",
                "Simple Default"
            ]
        )
        if st.button("🔄 Load Demo Data", use_container_width=True):
            df, dep, indep, lon_col, lat_col = generate_demo_data(demo_scenario)
            st.session_state.data = df
            st.session_state.dep_var = dep
            st.session_state.indep_vars = indep
            st.session_state.lon_col = lon_col
            st.session_state.lat_col = lat_col
            st.success(f"✅ Data demo '{demo_scenario}' berhasil dimuat! ({len(df)} observasi)")

    elif data_source == "📁 Upload CSV":
        uploaded_file = st.file_uploader("Upload file CSV", type=['csv'])
        if uploaded_file is not None:
            sep = st.selectbox("Separator:", [",", ";", "\t", "|"])
            if sep == "\t":
                sep = "\t"
            df = pd.read_csv(uploaded_file, sep=sep)
            st.session_state.data = df
            st.success(f"✅ Data berhasil dimuat! ({len(df)} baris, {len(df.columns)} kolom)")

    elif data_source == "📁 Upload Excel":
        uploaded_file = st.file_uploader("Upload file Excel", type=['xlsx', 'xls'])
        if uploaded_file is not None:
            sheet = st.text_input("Nama Sheet (kosongkan untuk default):", "")
            if sheet:
                df = pd.read_excel(uploaded_file, sheet_name=sheet)
            else:
                df = pd.read_excel(uploaded_file)
            st.session_state.data = df
            st.success(f"✅ Data berhasil dimuat! ({len(df)} baris, {len(df.columns)} kolom)")

    st.markdown("---")

    # Variable selection (when data is loaded)
    if st.session_state.data is not None:
        df = st.session_state.data
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        st.markdown("### 🎯 Pemilihan Variabel")

        if data_source != "📦 Data Demo" or 'dep_var' not in st.session_state or st.session_state.dep_var is None:
            dep_var = st.selectbox("Variabel Dependen (Y):", numeric_cols)
            st.session_state.dep_var = dep_var

            remaining = [c for c in numeric_cols if c != dep_var]
            indep_vars = st.multiselect("Variabel Independen (X):", remaining, default=remaining[:3] if len(remaining) >= 3 else remaining)
            st.session_state.indep_vars = indep_vars

            lon_col = st.selectbox("Kolom Longitude:", [c for c in numeric_cols if any(k in c.lower() for k in ['lon', 'x', 'long', 'bujur'])] or numeric_cols)
            lat_col = st.selectbox("Kolom Latitude:", [c for c in numeric_cols if any(k in c.lower() for k in ['lat', 'y', 'lintang'])] or numeric_cols)
            st.session_state.lon_col = lon_col
            st.session_state.lat_col = lat_col
        else:
            st.info(f"**Y:** {st.session_state.dep_var}")
            st.info(f"**X:** {', '.join(st.session_state.indep_vars)}")
            st.info(f"**Coords:** ({st.session_state.lon_col}, {st.session_state.lat_col})")
            if st.button("🔧 Ubah Variabel"):
                st.session_state.dep_var = None
                st.rerun()

        st.markdown("---")
        st.markdown("### 🔬 Parameter Model")

        kernel_type = st.selectbox("Kernel Function:", ["bisquare", "gaussian", "exponential"], index=0)
        fixed_or_adaptive = st.radio("Bandwidth Type:", ["Adaptive", "Fixed"])
        bw_fixed = True if fixed_or_adaptive == "Fixed" else False

        search_method = st.selectbox("Bandwidth Search:", ["golden_section", "interval"], index=0)
        criterion = st.selectbox("Selection Criterion:", ["AICc", "AIC", "BIC", "CV"], index=0)

        st.session_state.kernel_type = kernel_type
        st.session_state.bw_fixed = bw_fixed
        st.session_state.search_method = search_method
        st.session_state.criterion = criterion

        st.markdown("---")
        st.markdown("### 🚀 Advanced Options")

        with st.expander("⚙️ Advanced Settings"):
            alpha_level = st.slider("Significance Level (α):", 0.01, 0.10, 0.05, 0.01)
            st.session_state.alpha_level = alpha_level

            n_monte_carlo = st.number_input("Monte Carlo Iterations:", 50, 1000, 100, 50)
            st.session_state.n_monte_carlo = int(n_monte_carlo)

            run_mgwr = st.checkbox("Juga estimasi MGWR", value=True)
            st.session_state.run_mgwr = run_mgwr

            standardize_mgwr = st.checkbox("Standardisasi variabel (MGWR)", value=True)
            st.session_state.standardize_mgwr = standardize_mgwr

            compare_ols = st.checkbox("Bandingkan dengan OLS", value=True)
            st.session_state.compare_ols = compare_ols

    st.markdown("---")
    st.markdown("### 📝 Info")
    st.markdown("""
    **GWR Analysis Suite v2.0**
    Powered by `mgwr`, `PySAL`
    © 2026 Spatial Analytics
    """)


# ============================================================
# MAIN CONTENT AREA
# ============================================================
if st.session_state.data is None:
    # Landing page
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        ### 📊 Data Management
        - Upload CSV/Excel atau gunakan data demo
        - 4 skenario demo built-in
        - Eksplorasi data interaktif
        """)
    with col2:
        st.markdown("""
        ### 🔬 Analisis Lengkap
        - OLS Baseline → GWR → MGWR
        - Uji asumsi klasik lengkap
        - Parameter & diagnostik detail
        """)
    with col3:
        st.markdown("""
        ### 🗺️ Visualisasi
        - Peta koefisien interaktif
        - Local R², residual maps
        - Perbandingan model visual
        """)

    st.info("👈 Mulai dengan memilih sumber data di sidebar!")
    st.stop()

# ============================================================
# DATA LOADED → SHOW TABS
# ============================================================
df = st.session_state.data

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "📋 Data Explorer",
    "📐 Uji Asumsi",
    "📊 OLS Baseline",
    "🌍 GWR Model",
    "🔬 MGWR Model",
    "📈 Perbandingan Model",
    "🗺️ Visualisasi Peta",
    "📄 Laporan & Export"
])

# ============================================================
# TAB 1: DATA EXPLORER
# ============================================================
with tab1:
    st.markdown("## 📋 Data Explorer")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Jumlah Observasi", len(df))
    with col2:
        st.metric("Jumlah Variabel", len(df.columns))
    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())
    with col4:
        st.metric("Duplikat", df.duplicated().sum())

    st.markdown("### 🔎 Preview Data")
    st.dataframe(df.head(20), use_container_width=True)

    st.markdown("### 📊 Statistik Deskriptif")
    desc = df.describe().round(4)
    st.dataframe(desc, use_container_width=True)

    st.markdown("### 📍 Peta Lokasi Observasi")
    if st.session_state.lon_col and st.session_state.lat_col:
        fig_map = px.scatter_mapbox(
            df,
            lon=st.session_state.lon_col,
            lat=st.session_state.lat_col,
            color=st.session_state.dep_var if st.session_state.dep_var else None,
            size_max=15,
            zoom=6,
            mapbox_style="open-street-map",
            title="Distribusi Spasial Observasi",
            color_continuous_scale="Viridis",
            hover_data=df.columns.tolist()[:8]
        )
        fig_map.update_layout(height=500)
        st.plotly_chart(fig_map, use_container_width=True)

    st.markdown("### 📉 Distribusi Variabel")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    selected_var = st.selectbox("Pilih variabel:", numeric_cols, key="dist_var")

    col1, col2 = st.columns(2)
    with col1:
        fig_hist = px.histogram(df, x=selected_var, nbins=30, title=f"Histogram: {selected_var}",
                                marginal="box", color_discrete_sequence=["#2E86C1"])
        st.plotly_chart(fig_hist, use_container_width=True)
    with col2:
        fig_qq = go.Figure()
        from scipy import stats
        sorted_data = np.sort(df[selected_var].dropna())
        theoretical_q = stats.norm.ppf(np.linspace(0.01, 0.99, len(sorted_data)))
        fig_qq.add_trace(go.Scatter(x=theoretical_q, y=sorted_data, mode='markers',
                                    name='Data', marker=dict(color='#2E86C1', size=4)))
        fig_qq.add_trace(go.Scatter(x=[theoretical_q.min(), theoretical_q.max()],
                                    y=[sorted_data.min(), sorted_data.max()],
                                    mode='lines', name='Reference', line=dict(color='red', dash='dash')))
        fig_qq.update_layout(title=f"Q-Q Plot: {selected_var}", xaxis_title="Theoretical Quantiles",
                            yaxis_title="Sample Quantiles")
        st.plotly_chart(fig_qq, use_container_width=True)

    st.markdown("### 🔗 Korelasi Antar Variabel")
    corr_vars = [st.session_state.dep_var] + st.session_state.indep_vars if st.session_state.dep_var else numeric_cols
    corr_vars = [v for v in corr_vars if v in df.columns]
    corr_matrix = df[corr_vars].corr()

    fig_corr = px.imshow(corr_matrix, text_auto=".3f", color_continuous_scale="RdBu_r",
                         title="Correlation Matrix", aspect="auto",
                         zmin=-1, zmax=1)
    fig_corr.update_layout(height=500)
    st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown("### 📊 Scatter Matrix")
    if len(corr_vars) <= 6:
        fig_scatter = px.scatter_matrix(df[corr_vars], dimensions=corr_vars,
                                        color=st.session_state.dep_var if st.session_state.dep_var in corr_vars else None,
                                        title="Scatter Matrix")
        fig_scatter.update_layout(height=700)
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.warning("Terlalu banyak variabel untuk scatter matrix. Pilih ≤ 6 variabel.")


# ============================================================
# TAB 2: UJI ASUMSI (ASSUMPTION TESTING)
# ============================================================
with tab2:
    st.markdown("## 📐 Uji Asumsi Regresi")
    st.markdown("""
    <div class="info-box">
    Uji asumsi ini dilakukan pada model OLS global sebagai baseline sebelum melanjutkan ke GWR.
    GWR sendiri merelaksasi asumsi stasioneritas spasial, namun asumsi lain tetap penting.
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.dep_var and st.session_state.indep_vars:
        dep = st.session_state.dep_var
        indeps = st.session_state.indep_vars

        y_data = df[dep].values
        X_data = df[indeps].values
        X_const = np.column_stack([np.ones(len(y_data)), X_data])

        import statsmodels.api as sm
        from scipy import stats as scipy_stats
        from scipy.stats import shapiro, kstest, jarque_bera
        from statsmodels.stats.diagnostic import het_breuschpagan, het_white
        from statsmodels.stats.stattools import durbin_watson

        # Fit OLS for assumption testing
        ols_model = sm.OLS(y_data, sm.add_constant(X_data)).fit()
        residuals = ols_model.resid
        fitted = ols_model.fittedvalues

        st.markdown("### 1️⃣ Uji Normalitas Residual")
        col1, col2, col3 = st.columns(3)

        # Shapiro-Wilk
        if len(residuals) <= 5000:
            sw_stat, sw_p = shapiro(residuals)
        else:
            sw_stat, sw_p = shapiro(np.random.choice(residuals, 5000, replace=False))

        with col1:
            st.markdown("**Shapiro-Wilk Test**")
            st.write(f"Statistic: `{sw_stat:.6f}`")
            st.write(f"P-value: `{sw_p:.6f}`")
            if sw_p > 0.05:
                st.markdown('<div class="success-box">✅ Residual berdistribusi normal (p > 0.05)</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="warning-box">⚠️ Residual TIDAK normal (p ≤ 0.05)</div>', unsafe_allow_html=True)

        # Jarque-Bera
        jb_result = jarque_bera(residuals)
        try:
            jb_stat, jb_p = float(jb_result.statistic), float(jb_result.pvalue)
        except AttributeError:
            jb_stat, jb_p = float(jb_result[0]), float(jb_result[1])
        skew_val = float(pd.Series(residuals).skew())
        kurt_val = float(pd.Series(residuals).kurtosis())
        with col2:
            st.markdown("**Jarque-Bera Test**")
            st.write(f"Statistic: `{jb_stat:.6f}`")
            st.write(f"P-value: `{jb_p:.6f}`")
            st.write(f"Skewness: `{skew_val:.4f}`")
            st.write(f"Kurtosis: `{kurt_val:.4f}`")
            if jb_p > 0.05:
                st.markdown('<div class="success-box">✅ Normal (p > 0.05)</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="warning-box">⚠️ Tidak Normal (p ≤ 0.05)</div>', unsafe_allow_html=True)

        # KS Test
        ks_stat, ks_p = kstest(residuals, 'norm', args=(residuals.mean(), residuals.std()))
        with col3:
            st.markdown("**Kolmogorov-Smirnov Test**")
            st.write(f"Statistic: `{ks_stat:.6f}`")
            st.write(f"P-value: `{ks_p:.6f}`")
            if ks_p > 0.05:
                st.markdown('<div class="success-box">✅ Normal (p > 0.05)</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="warning-box">⚠️ Tidak Normal (p ≤ 0.05)</div>', unsafe_allow_html=True)

        # Residual plots
        col1, col2 = st.columns(2)
        with col1:
            fig_resid_hist = px.histogram(x=residuals, nbins=30, title="Distribusi Residual",
                                          labels={'x': 'Residual'}, marginal="box",
                                          color_discrete_sequence=["#E74C3C"])
            st.plotly_chart(fig_resid_hist, use_container_width=True)

        with col2:
            sorted_res = np.sort(residuals)
            theoretical_q = scipy_stats.norm.ppf(np.linspace(0.01, 0.99, len(sorted_res)))
            fig_qq_res = go.Figure()
            fig_qq_res.add_trace(go.Scatter(x=theoretical_q, y=sorted_res, mode='markers',
                                            marker=dict(color='#E74C3C', size=4), name='Residuals'))
            min_val = min(theoretical_q.min(), sorted_res.min())
            max_val = max(theoretical_q.max(), sorted_res.max())
            fig_qq_res.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                                            mode='lines', line=dict(color='blue', dash='dash'), name='45° Line'))
            fig_qq_res.update_layout(title="Q-Q Plot Residual", xaxis_title="Theoretical Quantiles",
                                    yaxis_title="Sample Quantiles")
            st.plotly_chart(fig_qq_res, use_container_width=True)

        st.markdown("---")
        st.markdown("### 2️⃣ Uji Heteroskedastisitas")
        col1, col2 = st.columns(2)

        # Breusch-Pagan
        bp_stat, bp_p, bp_f, bp_fp = het_breuschpagan(residuals, sm.add_constant(X_data))
        with col1:
            st.markdown("**Breusch-Pagan Test**")
            st.write(f"LM Statistic: `{bp_stat:.6f}`")
            st.write(f"LM P-value: `{bp_p:.6f}`")
            st.write(f"F-Statistic: `{bp_f:.6f}`")
            st.write(f"F P-value: `{bp_fp:.6f}`")
            if bp_p > 0.05:
                st.markdown('<div class="success-box">✅ Homoskedastik (p > 0.05)</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="warning-box">⚠️ Heteroskedastik (p ≤ 0.05)</div>', unsafe_allow_html=True)

        # White Test
        try:
            w_stat, w_p, w_f, w_fp = het_white(residuals, sm.add_constant(X_data))
            with col2:
                st.markdown("**White Test**")
                st.write(f"LM Statistic: `{w_stat:.6f}`")
                st.write(f"LM P-value: `{w_p:.6f}`")
                st.write(f"F-Statistic: `{w_f:.6f}`")
                st.write(f"F P-value: `{w_fp:.6f}`")
                if w_p > 0.05:
                    st.markdown('<div class="success-box">✅ Homoskedastik (p > 0.05)</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="warning-box">⚠️ Heteroskedastik (p ≤ 0.05)</div>', unsafe_allow_html=True)
        except Exception as e:
            with col2:
                st.warning(f"White Test gagal: {e}")

        # Residual vs Fitted plot
        fig_rvf = px.scatter(x=fitted, y=residuals, title="Residual vs Fitted Values",
                             labels={'x': 'Fitted Values', 'y': 'Residuals'},
                             color_discrete_sequence=["#8E44AD"])
        fig_rvf.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig_rvf, use_container_width=True)

        st.markdown("---")
        st.markdown("### 3️⃣ Uji Multikolinearitas (VIF)")
        from statsmodels.stats.outliers_influence import variance_inflation_factor

        vif_data = pd.DataFrame()
        vif_data['Variabel'] = indeps
        vif_data['VIF'] = [variance_inflation_factor(X_const, i+1) for i in range(len(indeps))]
        vif_data['Tolerance'] = 1 / vif_data['VIF']
        vif_data['Status'] = vif_data['VIF'].apply(
            lambda x: '✅ Baik (< 5)' if x < 5 else ('⚠️ Moderat (5-10)' if x < 10 else '❌ Tinggi (> 10)')
        )
        st.dataframe(vif_data, use_container_width=True)

        fig_vif = px.bar(vif_data, x='Variabel', y='VIF', title="Variance Inflation Factor (VIF)",
                         color='VIF', color_continuous_scale='RdYlGn_r',
                         text='VIF')
        fig_vif.add_hline(y=5, line_dash="dash", line_color="orange", annotation_text="Threshold = 5")
        fig_vif.add_hline(y=10, line_dash="dash", line_color="red", annotation_text="Threshold = 10")
        fig_vif.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        st.plotly_chart(fig_vif, use_container_width=True)

        st.markdown("---")
        st.markdown("### 4️⃣ Uji Autokorelasi")
        dw_stat = durbin_watson(residuals)
        st.write(f"**Durbin-Watson Statistic:** `{dw_stat:.6f}`")
        if 1.5 < dw_stat < 2.5:
            st.markdown('<div class="success-box">✅ Tidak ada autokorelasi signifikan (1.5 < DW < 2.5)</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="warning-box">⚠️ Kemungkinan ada autokorelasi (DW di luar 1.5-2.5)</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 5️⃣ Uji Linearitas")
        st.markdown("**RESET Test (Ramsey)**")
        try:
            from statsmodels.stats.diagnostic import linear_reset
            reset_result = linear_reset(ols_model, power=3, use_f=True)
            st.write(f"F-Statistic: `{reset_result.fvalue:.6f}`")
            st.write(f"P-value: `{reset_result.pvalue:.6f}`")
            if reset_result.pvalue > 0.05:
                st.markdown('<div class="success-box">✅ Model linear memadai (p > 0.05)</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="warning-box">⚠️ Hubungan mungkin non-linear (p ≤ 0.05)</div>', unsafe_allow_html=True)
        except Exception as e:
            st.warning(f"RESET Test gagal: {e}")

        st.markdown("---")
        st.markdown("### 6️⃣ Uji Autokorelasi Spasial (Moran's I)")
        st.markdown("**Moran's I pada Residual OLS**")
        try:
            from libpysal.weights import KNN
            from esda.moran import Moran

            coords_arr = list(zip(df[st.session_state.lon_col].values, df[st.session_state.lat_col].values))
            w = KNN.from_array(np.array(coords_arr), k=8)
            w.transform = 'r'

            moran_obj = Moran(residuals, w)
            st.write(f"**Moran's I:** `{moran_obj.I:.6f}`")
            st.write(f"**Expected I:** `{moran_obj.EI:.6f}`")
            st.write(f"**Z-score:** `{moran_obj.z_norm:.6f}`")
            st.write(f"**P-value:** `{moran_obj.p_norm:.6f}`")

            if moran_obj.p_norm < 0.05:
                st.markdown("""
                <div class="warning-box">
                ⚠️ <b>Autokorelasi spasial terdeteksi!</b><br>
                Ini mengindikasikan bahwa model OLS global tidak cukup menangkap variasi spasial.
                GWR sangat direkomendasikan untuk mengatasi masalah ini.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown('<div class="success-box">✅ Tidak ada autokorelasi spasial signifikan (p > 0.05)</div>', unsafe_allow_html=True)
        except ImportError:
            st.warning("Library `libpysal` dan `esda` diperlukan untuk Moran's I test.")
        except Exception as e:
            st.warning(f"Moran's I gagal: {e}")

        st.markdown("---")
        st.markdown("### 📝 Ringkasan Uji Asumsi")
        assumption_summary = pd.DataFrame({
            'Uji': ['Normalitas (Shapiro-Wilk)', 'Normalitas (Jarque-Bera)',
                     'Heteroskedastisitas (Breusch-Pagan)', 'Multikolinearitas (VIF max)',
                     'Autokorelasi (Durbin-Watson)'],
            'Statistik': [f'{sw_stat:.4f}', f'{jb_stat:.4f}', f'{bp_stat:.4f}',
                          f'{vif_data["VIF"].max():.4f}', f'{dw_stat:.4f}'],
            'P-value': [f'{sw_p:.4f}', f'{jb_p:.4f}', f'{bp_p:.4f}', '-', '-'],
            'Keputusan': [
                '✅ Normal' if sw_p > 0.05 else '⚠️ Tidak Normal',
                '✅ Normal' if jb_p > 0.05 else '⚠️ Tidak Normal',
                '✅ Homoskedastik' if bp_p > 0.05 else '⚠️ Heteroskedastik',
                '✅ Tidak Multikolinear' if vif_data['VIF'].max() < 10 else '⚠️ Multikolinear',
                '✅ Tidak Autokorelasi' if 1.5 < dw_stat < 2.5 else '⚠️ Autokorelasi'
            ]
        })
        st.dataframe(assumption_summary, use_container_width=True)

    else:
        st.warning("⚠️ Pilih variabel dependen dan independen terlebih dahulu di sidebar!")


# ════════════════════════════════════════════════════════════
# END OF: gwr_app_part1.py
# START OF: gwr_app_part2.py
# ════════════════════════════════════════════════════════════

# ============================================================
# FILE 2: gwr_app_part2.py
# ============================================================


# ============================================================
# APLIKASI ANALISIS GWR LENGKAP - BAGIAN 2/3
# TAMBAHKAN KODE INI SETELAH BAGIAN 1 (sebelum closing)
# ============================================================

# ============================================================
# TAB 3: OLS BASELINE
# ============================================================
with tab3:
    st.markdown("## 📊 OLS Global Regression (Baseline)")

    if st.session_state.dep_var and st.session_state.indep_vars:

        dep = st.session_state.dep_var
        indeps = st.session_state.indep_vars

        y_ols = df[dep].values
        X_ols = df[indeps].values
        X_ols_const = sm.add_constant(X_ols)

        ols_model = sm.OLS(y_ols, X_ols_const).fit()
        st.session_state.ols_results = ols_model

        # ---- Model Summary ----
        st.markdown("### 📋 Model Summary")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("R²", f"{ols_model.rsquared:.6f}")
        with col2:
            st.metric("Adj. R²", f"{ols_model.rsquared_adj:.6f}")
        with col3:
            st.metric("AIC", f"{ols_model.aic:.4f}")
        with col4:
            st.metric("BIC", f"{ols_model.bic:.4f}")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("F-statistic", f"{ols_model.fvalue:.4f}")
        with col2:
            st.metric("Prob (F-stat)", f"{ols_model.f_pvalue:.6f}")
        with col3:
            st.metric("Log-Likelihood", f"{ols_model.llf:.4f}")
        with col4:
            n_obs = len(y_ols)
            k_params = len(indeps) + 1
            aicc_ols = ols_model.aic + (2 * k_params * (k_params + 1)) / (n_obs - k_params - 1)
            st.metric("AICc", f"{aicc_ols:.4f}")

        st.session_state.ols_aicc = aicc_ols

        # ---- Coefficient Table ----
        st.markdown("### 📊 Tabel Koefisien")
        _ci = ols_model.conf_int()
        coef_df = pd.DataFrame({
            'Variable': ['Intercept'] + indeps,
            'Coefficient': ols_model.params.flatten(),
            'Std. Error': ols_model.bse.flatten(),
            't-value': ols_model.tvalues.flatten(),
            'P-value': ols_model.pvalues.flatten(),
            'CI Lower (95%)': _ci.iloc[:, 0].values if hasattr(_ci, 'iloc') else _ci[:, 0],
            'CI Upper (95%)': _ci.iloc[:, 1].values if hasattr(_ci, 'iloc') else _ci[:, 1],
            'Significant': ['✅ Yes' if p < 0.05 else '❌ No' for p in ols_model.pvalues.flatten()]
        })
        st.dataframe(coef_df.round(6), use_container_width=True)

        # ---- Model Equation ----
        st.markdown("### ✏️ Persamaan Model OLS")
        _p = ols_model.params.flatten()
        equation = f"**{dep}** = {_p[0]:.4f}"
        for i, var in enumerate(indeps):
            coef = _p[i + 1]
            sign = "+" if coef >= 0 else ""
            equation += f" {sign} {coef:.4f} × **{var}**"
        st.markdown(equation)
        try:
            _p = ols_model.params.flatten()
            latex_str = f"{dep} = {_p[0]:.4f}"
            for _i, _v in enumerate(indeps):
                _cv = _p[_i+1]
                _s = "+" if _cv >= 0 else "-"
                latex_str += f" {_s} {abs(_cv):.4f} \\cdot \\text{{{_v}}}"
            st.latex(latex_str)
        except Exception:
            pass

        # ---- Full OLS Summary ----
        with st.expander("📄 Full OLS Summary (statsmodels)"):
            st.text(ols_model.summary().as_text())

        # ---- Diagnostic Plots ----
        st.markdown("### 📈 Diagnostic Plots")
        col1, col2 = st.columns(2)

        with col1:
            fig_fit = px.scatter(x=ols_model.fittedvalues.flatten(), y=y_ols.flatten(),
                                 title="Actual vs Predicted",
                                 labels={'x': 'Predicted', 'y': 'Actual'},
                                 color_discrete_sequence=["#2E86C1"])
            min_val = min(ols_model.fittedvalues.min(), y_ols.min())
            max_val = max(ols_model.fittedvalues.max(), y_ols.max())
            fig_fit.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                                         mode='lines', line=dict(color='red', dash='dash'),
                                         name='Perfect Fit'))
            st.plotly_chart(fig_fit, use_container_width=True)

        with col2:
            fig_resid = px.scatter(x=ols_model.fittedvalues.flatten(), y=ols_model.resid.flatten(),
                                   title="Residuals vs Fitted",
                                   labels={'x': 'Fitted Values', 'y': 'Residuals'},
                                   color_discrete_sequence=["#E74C3C"])
            fig_resid.add_hline(y=0, line_dash="dash", line_color="black")
            st.plotly_chart(fig_resid, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            fig_scale = px.scatter(x=ols_model.fittedvalues.flatten(),
                                   y=np.sqrt(np.abs(ols_model.get_influence().resid_studentized_internal)),
                                   title="Scale-Location Plot",
                                   labels={'x': 'Fitted Values', 'y': '√|Standardized Residuals|'},
                                   color_discrete_sequence=["#27AE60"])
            st.plotly_chart(fig_scale, use_container_width=True)

        with col2:
            leverage = ols_model.get_influence().hat_matrix_diag
            fig_lev = px.scatter(x=leverage, y=ols_model.get_influence().resid_studentized_internal,
                                 title="Residuals vs Leverage",
                                 labels={'x': 'Leverage', 'y': 'Studentized Residuals'},
                                 color_discrete_sequence=["#8E44AD"])
            fig_lev.add_hline(y=0, line_dash="dash", line_color="black")
            st.plotly_chart(fig_lev, use_container_width=True)

    else:
        st.warning("⚠️ Pilih variabel dependen dan independen terlebih dahulu!")

# ============================================================
# TAB 4: GWR MODEL
# ============================================================
with tab4:
    st.markdown("## 🌍 Geographically Weighted Regression (GWR)")

    if st.session_state.dep_var and st.session_state.indep_vars:
        dep = st.session_state.dep_var
        indeps = st.session_state.indep_vars

        y_gwr = df[dep].values.reshape(-1, 1)
        X_gwr = df[indeps].values
        coords_gwr = list(zip(df[st.session_state.lon_col].values, df[st.session_state.lat_col].values))

        st.markdown("""
        <div class="info-box">
        <b>GWR</b> memungkinkan koefisien regresi bervariasi secara spasial.
        Setiap lokasi memiliki set koefisien lokal berdasarkan observasi terdekat
        yang diberi bobot menggunakan fungsi kernel.
        </div>
        """, unsafe_allow_html=True)

        run_gwr = st.button("🚀 Jalankan GWR Model", use_container_width=True, type="primary")

        if run_gwr:
            with st.spinner("⏳ Mencari bandwidth optimal dan estimasi GWR..."):
                try:
                    from mgwr.gwr import GWR
                    from mgwr.sel_bw import Sel_BW

                    coords_arr = np.array(coords_gwr)

                    kernel = st.session_state.kernel_type
                    fixed = st.session_state.bw_fixed
                    criterion_map = {'AICc': 'AICc', 'AIC': 'AIC', 'BIC': 'BIC', 'CV': 'CV'}
                    crit = criterion_map[st.session_state.criterion]

                    # Bandwidth selection
                    gwr_selector = Sel_BW(coords_arr, y_gwr, X_gwr, kernel=kernel, fixed=fixed)
                    gwr_bw = gwr_selector.search(criterion=crit, search_method=st.session_state.search_method)

                    st.session_state.gwr_bw = gwr_bw
                    st.session_state.gwr_selector = gwr_selector

                    # Fit GWR
                    gwr_model = GWR(coords_arr, y_gwr, X_gwr, bw=gwr_bw, kernel=kernel, fixed=fixed)
                    gwr_results = gwr_model.fit()

                    st.session_state.gwr_results = gwr_results
                    st.session_state.gwr_coords = coords_arr

                    st.success(f"✅ GWR berhasil diestimasi! Bandwidth optimal = {gwr_bw}")
                except Exception as e:
                    st.error(f"❌ Error: {e}")
                    st.info("Pastikan library `mgwr` terinstall: `pip install mgwr`")

        # Display results if available
        if st.session_state.gwr_results is not None:
            gwr_results = st.session_state.gwr_results
            gwr_bw = st.session_state.gwr_bw

            # ---- Global Diagnostics ----
            st.markdown("### 📊 Diagnostik Global GWR")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Bandwidth", f"{gwr_bw:.2f}" if isinstance(gwr_bw, float) else f"{gwr_bw}")
            with col2:
                st.metric("R²", f"{gwr_results.R2:.6f}")
            with col3:
                st.metric("Adj. R²", f"{gwr_results.adj_R2:.6f}")
            with col4:
                st.metric("AICc", f"{gwr_results.aicc:.4f}")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("AIC", f"{gwr_results.aic:.4f}")
            with col2:
                st.metric("BIC", f"{gwr_results.bic:.4f}")
            with col3:
                st.metric("Eff. Parameters (ENP)", f"{gwr_results.ENP:.4f}")
            with col4:
                st.metric("Sigma²", f"{gwr_results.sigma2:.6f}")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Residual SS", f"{gwr_results.resid_ss:.4f}")
            with col2:
                st.metric("df Model", f"{gwr_results.df_model:.4f}")
            with col3:
                st.metric("tr(S)", f"{gwr_results.tr_S:.4f}")

            # ---- Bandwidth CI ----
            st.markdown("### 🔍 Bandwidth Confidence Interval")
            try:
                bw_ci = gwr_results.get_bws_intervals(st.session_state.gwr_selector)
                st.write(f"**Bandwidth CI:** {bw_ci}")
            except Exception as e:
                st.warning(f"CI tidak tersedia: {e}")

            # ---- Local Coefficient Summary ----
            st.markdown("### 📋 Ringkasan Koefisien Lokal")
            var_names = ['Intercept'] + indeps
            params = gwr_results.params
            tvalues = gwr_results.tvalues
            std_errs = gwr_results.bse

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

            # ---- Local t-values Summary ----
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
                    'Positive Coef (%)': f"{100*np.sum(params[:, i] > 0)/len(params):.1f}%",
                    'Negative Coef (%)': f"{100*np.sum(params[:, i] < 0)/len(params):.1f}%"
                }
                tval_summary = pd.concat([tval_summary, pd.DataFrame([row])], ignore_index=True)

            st.dataframe(tval_summary, use_container_width=True)

            # ---- Standard Errors Summary ----
            st.markdown("### 📊 Ringkasan Standard Errors Lokal")
            se_summary = pd.DataFrame()
            for i, vn in enumerate(var_names):
                row = {
                    'Variable': vn,
                    'Mean SE': std_errs[:, i].mean(),
                    'Min SE': std_errs[:, i].min(),
                    'Median SE': np.median(std_errs[:, i]),
                    'Max SE': std_errs[:, i].max(),
                    'Std of SE': std_errs[:, i].std()
                }
                se_summary = pd.concat([se_summary, pd.DataFrame([row])], ignore_index=True)
            st.dataframe(se_summary.round(6), use_container_width=True)

            # ---- Local R² Summary ----
            st.markdown("### 📊 Local R² Summary")
            localR2 = gwr_results.localR2
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Mean Local R²", f"{localR2.mean():.6f}")
            with col2:
                st.metric("Min Local R²", f"{localR2.min():.6f}")
            with col3:
                st.metric("Median Local R²", f"{np.median(localR2):.6f}")
            with col4:
                st.metric("Max Local R²", f"{localR2.max():.6f}")
            with col5:
                st.metric("Std Local R²", f"{localR2.std():.6f}")

            fig_lr2 = px.histogram(x=localR2.flatten(), nbins=30,
                                    title="Distribusi Local R²",
                                    labels={'x': 'Local R²'},
                                    color_discrete_sequence=["#2E86C1"],
                                    marginal="box")
            st.plotly_chart(fig_lr2, use_container_width=True)

            # ---- Spatial Variability Test (Monte Carlo) ----
            st.markdown("### 🎲 Monte Carlo Test of Spatial Variability")
            st.markdown("""
            <div class="info-box">
            Test ini menguji apakah koefisien benar-benar bervariasi secara spasial
            atau cukup dimodelkan secara global (konstan).
            H₀: Koefisien konstan secara spasial | H₁: Koefisien bervariasi secara spasial
            </div>
            """, unsafe_allow_html=True)

            n_mc = st.session_state.get('n_monte_carlo', 100)
            run_mc = st.button(f"🎲 Jalankan Monte Carlo Test ({n_mc} iterasi)", key="mc_gwr")

            if run_mc:
                with st.spinner(f"⏳ Monte Carlo test sedang berjalan ({n_mc} iterasi)..."):
                    try:
                        try:
                            mc_results = gwr_results.spatial_variability(st.session_state.gwr_selector, n_mc)
                        except TypeError:
                            mc_results = gwr_results.spatial_variability(st.session_state.gwr_selector, niter=n_mc)
                        st.session_state.mc_results = mc_results

                        mc_df = pd.DataFrame({
                            'Variable': var_names,
                            'P-value': mc_results,
                            'Significant (p<0.05)': ['✅ Yes - Spatially Varying' if p < 0.05
                                                      else '❌ No - Spatially Constant' for p in mc_results]
                        })
                        st.dataframe(mc_df, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Monte Carlo test gagal: {e}")

            # ---- Coefficient Distribution Plots ----
            st.markdown("### 📈 Distribusi Koefisien Lokal")
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
                # Add global OLS coefficient line if available
                if st.session_state.ols_results is not None:
                    ols_coef = st.session_state.ols_results.params[i]
                    fig_coef.add_vline(x=ols_coef, line_dash="dash", line_color="red",
                                       row=r, col=c)
            fig_coef.update_layout(height=300 * rows_needed, showlegend=False,
                                    title_text="Distribusi Koefisien Lokal (garis merah = OLS global)")
            st.plotly_chart(fig_coef, use_container_width=True)

            # ---- Persamaan Model GWR ----
            st.markdown("### ✏️ Persamaan Model GWR")
            st.markdown("""
            **Form umum GWR:**
            """)
            try:
                _gl = "y_i = \\beta_0(u_i, v_i)"
                for _j in range(len(indeps)):
                    _gl += f" + \\beta_{{{_j+1}}}(u_i, v_i) \\cdot X_{{{_j+1}i}}"
                _gl += " + \\varepsilon_i"
                st.latex(_gl)
            except Exception:
                st.markdown("**y = β₀(u,v) + Σ βₖ(u,v)·Xₖ + ε**")

            st.markdown("**Dimana koefisien rata-rata:**")
            try:
                _parts = [f"{params[:, 0].mean():.4f}"]
                for _j, _vn in enumerate(indeps):
                    _cm = params[:, _j+1].mean()
                    _sg = "+" if _cm >= 0 else "-"
                    _parts.append(f"{_sg} {abs(_cm):.4f} × {_vn}")
                st.markdown(f"**ŷ = {' '.join(_parts)}**")
            except Exception:
                pass

            # ---- Cook's Distance ----
            st.markdown("### 🍳 Cook's Distance (Influential Observations)")
            try:
                cooks_d = gwr_results.CooksD
                if cooks_d is not None:
                    mean_cooks = cooks_d.mean(axis=1)
                    fig_cooks = px.bar(x=list(range(len(mean_cooks))), y=mean_cooks,
                                       title="Mean Cook's Distance per Observation",
                                       labels={'x': 'Observation Index', 'y': "Mean Cook's D"})
                    threshold = 4 / len(mean_cooks)
                    fig_cooks.add_hline(y=threshold, line_dash="dash", line_color="red",
                                        annotation_text=f"Threshold = {threshold:.4f}")
                    st.plotly_chart(fig_cooks, use_container_width=True)

                    n_influential = np.sum(mean_cooks > threshold)
                    st.write(f"**Observasi berpengaruh (Cook's D > {threshold:.4f}):** {n_influential} dari {len(mean_cooks)}")
            except Exception as e:
                st.info(f"Cook's Distance: {e}")

            # ---- Multicollinearity Diagnostics ----
            st.markdown("### 📊 Local Multicollinearity Diagnostics")
            try:
                local_CN, local_VDP, local_VIF = gwr_results.local_collinearity()
                st.write(f"**Local Condition Number** — Mean: {local_CN.mean():.4f}, Max: {local_CN.max():.4f}")

                fig_cn = px.histogram(x=local_CN.flatten(), nbins=30,
                                       title="Distribusi Local Condition Number",
                                       labels={'x': 'Condition Number'},
                                       color_discrete_sequence=["#E67E22"],
                                       marginal="box")
                fig_cn.add_vline(x=30, line_dash="dash", line_color="red",
                                 annotation_text="Threshold = 30")
                st.plotly_chart(fig_cn, use_container_width=True)

                n_high_cn = np.sum(local_CN > 30)
                st.write(f"**Lokasi dengan CN > 30:** {n_high_cn} ({100*n_high_cn/len(local_CN):.1f}%)")

                if local_VIF is not None:
                    st.markdown("**Local VIF per Variable:**")
                    local_vif_df = pd.DataFrame()
                    for i, vn in enumerate(var_names):
                        if i < local_VIF.shape[1]:
                            row = {
                                'Variable': vn,
                                'Mean Local VIF': local_VIF[:, i].mean(),
                                'Max Local VIF': local_VIF[:, i].max(),
                                'Pct > 10': f"{100*np.sum(local_VIF[:, i] > 10)/len(local_VIF):.1f}%"
                            }
                            local_vif_df = pd.concat([local_vif_df, pd.DataFrame([row])], ignore_index=True)
                    st.dataframe(local_vif_df.round(4), use_container_width=True)

            except Exception as e:
                st.info(f"Local collinearity diagnostics: {e}")

            # ---- Full GWR Summary ----
            with st.expander("📄 Full GWR Summary"):
                try:
                    import sys as _sys
                    _buf = io.StringIO()
                    _old = _sys.stdout
                    _sys.stdout = _buf
                    gwr_results.summary()
                    _sys.stdout = _old
                    _t = _buf.getvalue()
                    st.text(_t if _t.strip() else "Summary tidak tersedia.")
                except Exception as _e:
                    try: _sys.stdout = _sys.__stdout__
                    except: pass
                    st.warning(f"Summary error: {_e}")

            # ---- Export Local Parameters ----
            st.markdown("### 💾 Export Parameter Lokal")
            export_df = df.copy()
            for i, vn in enumerate(var_names):
                export_df[f'Beta_{vn}'] = params[:, i]
                export_df[f'SE_{vn}'] = std_errs[:, i]
                export_df[f'tval_{vn}'] = tvalues[:, i]
            export_df['Local_R2'] = localR2.flatten()
            export_df['Predicted_Y'] = gwr_results.predy.flatten()
            export_df['Residual'] = gwr_results.resid_response.flatten()
            export_df['Std_Residual'] = gwr_results.std_res.flatten()
            export_df['Influence'] = gwr_results.influ.flatten()

            csv_buffer = io.StringIO()
            export_df.to_csv(csv_buffer, index=False)
            st.download_button(
                label="📥 Download Parameter Lokal (CSV)",
                data=csv_buffer.getvalue(),
                file_name=f"gwr_local_parameters_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )

    else:
        st.warning("⚠️ Pilih variabel terlebih dahulu!")


# ============================================================
# TAB 5: MGWR MODEL
# ============================================================
with tab5:
    st.markdown("## 🔬 Multiscale GWR (MGWR)")

    if st.session_state.dep_var and st.session_state.indep_vars:
        dep = st.session_state.dep_var
        indeps = st.session_state.indep_vars

        st.markdown("""
        <div class="info-box">
        <b>MGWR</b> memperluas GWR dengan memungkinkan setiap variabel memiliki bandwidth sendiri.
        Ini menangkap proses spasial pada skala yang berbeda — beberapa variabel mungkin
        beroperasi secara lokal sementara yang lain beroperasi secara regional/global.
        </div>
        """, unsafe_allow_html=True)

        run_mgwr = st.button("🚀 Jalankan MGWR Model", use_container_width=True, type="primary")

        if run_mgwr and st.session_state.get('run_mgwr', True):
            with st.spinner("⏳ Estimasi MGWR (proses backfitting, mungkin memakan waktu)..."):
                try:
                    from mgwr.gwr import MGWR
                    from mgwr.sel_bw import Sel_BW

                    coords_arr = np.array(list(zip(
                        df[st.session_state.lon_col].values,
                        df[st.session_state.lat_col].values
                    )))

                    y_mgwr = df[dep].values.reshape(-1, 1)
                    X_mgwr = df[indeps].values

                    kernel = st.session_state.kernel_type
                    fixed = st.session_state.bw_fixed

                    # Standardize for MGWR
                    if st.session_state.get('standardize_mgwr', True):
                        Zy = (y_mgwr - y_mgwr.mean(axis=0)) / y_mgwr.std(axis=0)
                        ZX = (X_mgwr - X_mgwr.mean(axis=0)) / X_mgwr.std(axis=0)
                        st.info("ℹ️ Variabel telah di-standardisasi untuk MGWR")
                    else:
                        Zy = y_mgwr
                        ZX = X_mgwr

                    # Bandwidth selection
                    mgwr_selector = Sel_BW(coords_arr, Zy, ZX, multi=True, kernel=kernel, fixed=fixed)
                    mgwr_bw = mgwr_selector.search(multi_bw_min=[2], multi_bw_max=[len(df)])

                    # Fit MGWR
                    mgwr_model = MGWR(coords_arr, Zy, ZX, mgwr_selector)
                    mgwr_results = mgwr_model.fit()

                    st.session_state.mgwr_results = mgwr_results
                    st.session_state.mgwr_bw = mgwr_bw
                    st.session_state.mgwr_selector = mgwr_selector
                    st.session_state.mgwr_standardized = st.session_state.get('standardize_mgwr', True)

                    st.success(f"✅ MGWR berhasil diestimasi!")
                    st.write(f"**Bandwidth per variabel:** {mgwr_bw}")

                except Exception as e:
                    st.error(f"❌ Error: {e}")
                    import traceback
                    st.text(traceback.format_exc())

        # Display MGWR results
        if st.session_state.mgwr_results is not None:
            mgwr_results = st.session_state.mgwr_results
            mgwr_bw = st.session_state.mgwr_bw
            var_names = ['Intercept'] + indeps

            # ---- Global Diagnostics ----
            st.markdown("### 📊 Diagnostik Global MGWR")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("R²", f"{mgwr_results.R2:.6f}")
            with col2:
                st.metric("Adj. R²", f"{mgwr_results.adj_R2:.6f}")
            with col3:
                st.metric("AICc", f"{mgwr_results.aicc:.4f}")
            with col4:
                st.metric("Sigma²", f"{mgwr_results.sigma2:.6f}")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("AIC", f"{mgwr_results.aic:.4f}")
            with col2:
                st.metric("BIC", f"{mgwr_results.bic:.4f}")
            with col3:
                st.metric("ENP (Total)", f"{mgwr_results.ENP:.4f}")

            # ---- Bandwidth per variable ----
            st.markdown("### 🎯 Bandwidth per Variabel")
            bw_df = pd.DataFrame({
                'Variable': var_names,
                'Bandwidth': mgwr_bw,
                'Scale Interpretation': [
                    '🌐 Global' if bw >= len(df) * 0.8
                    else ('🏘️ Regional' if bw >= len(df) * 0.3
                          else '📍 Local')
                    for bw in mgwr_bw
                ]
            })
            st.dataframe(bw_df, use_container_width=True)

            fig_bw = px.bar(bw_df, x='Variable', y='Bandwidth',
                            title="Bandwidth MGWR per Variabel",
                            color='Bandwidth',
                            color_continuous_scale='Viridis',
                            text='Bandwidth')
            fig_bw.add_hline(y=len(df), line_dash="dash", line_color="red",
                             annotation_text=f"n = {len(df)}")
            fig_bw.update_traces(texttemplate='%{text:.0f}', textposition='outside')
            st.plotly_chart(fig_bw, use_container_width=True)

            # ---- Bandwidth CI ----
            st.markdown("### 🔍 Bandwidth Confidence Intervals (MGWR)")
            try:
                mgwr_bw_ci = mgwr_results.get_bws_intervals(st.session_state.mgwr_selector)
                ci_df = pd.DataFrame(mgwr_bw_ci, columns=['Lower', 'Optimal', 'Upper'])
                ci_df.insert(0, 'Variable', var_names)
                st.dataframe(ci_df, use_container_width=True)

                fig_ci = go.Figure()
                for idx, row in ci_df.iterrows():
                    fig_ci.add_trace(go.Scatter(
                        x=[row['Lower'], row['Optimal'], row['Upper']],
                        y=[row['Variable']] * 3,
                        mode='markers+lines',
                        marker=dict(size=[8, 15, 8]),
                        name=row['Variable'],
                        showlegend=True
                    ))
                fig_ci.update_layout(title="Bandwidth Confidence Intervals",
                                     xaxis_title="Bandwidth", yaxis_title="Variable",
                                     height=400)
                st.plotly_chart(fig_ci, use_container_width=True)

            except Exception as e:
                st.warning(f"CI tidak tersedia: {e}")

            # ---- MGWR Coefficient Summary ----
            st.markdown("### 📋 Ringkasan Koefisien MGWR")
            params_mgwr = mgwr_results.params
            tvalues_mgwr = mgwr_results.tvalues
            std_errs_mgwr = mgwr_results.bse

            coef_summary_mgwr = pd.DataFrame()
            for i, vn in enumerate(var_names):
                row = {
                    'Variable': vn,
                    'Bandwidth': mgwr_bw[i],
                    'Mean': params_mgwr[:, i].mean(),
                    'Std': params_mgwr[:, i].std(),
                    'Min': params_mgwr[:, i].min(),
                    'Median': np.median(params_mgwr[:, i]),
                    'Max': params_mgwr[:, i].max(),
                    'Pct Significant': f"{100*np.sum(np.abs(tvalues_mgwr[:, i]) > 1.96)/len(tvalues_mgwr):.1f}%"
                }
                coef_summary_mgwr = pd.concat([coef_summary_mgwr, pd.DataFrame([row])], ignore_index=True)
            st.dataframe(coef_summary_mgwr.round(6), use_container_width=True)

            # ---- MGWR Coefficient Distribution ----
            st.markdown("### 📈 Distribusi Koefisien MGWR")
            n_vars = len(var_names)
            cols_per_row = min(3, n_vars)
            rows_needed = (n_vars + cols_per_row - 1) // cols_per_row

            fig_mgwr_coef = make_subplots(rows=rows_needed, cols=cols_per_row,
                                           subplot_titles=[f"{vn} (BW={mgwr_bw[i]:.0f})" for i, vn in enumerate(var_names)])
            for i, vn in enumerate(var_names):
                r = i // cols_per_row + 1
                c = i % cols_per_row + 1
                fig_mgwr_coef.add_trace(
                    go.Histogram(x=params_mgwr[:, i], nbinsx=25, name=vn,
                                 marker_color=px.colors.qualitative.Set2[i % 8]),
                    row=r, col=c
                )
            fig_mgwr_coef.update_layout(height=300 * rows_needed, showlegend=False,
                                         title_text="Distribusi Koefisien MGWR")
            st.plotly_chart(fig_mgwr_coef, use_container_width=True)

            # ---- Full MGWR Summary ----
            with st.expander("📄 Full MGWR Summary"):
                try:
                    import sys as _sys
                    _buf = io.StringIO()
                    _old = _sys.stdout
                    _sys.stdout = _buf
                    mgwr_results.summary()
                    _sys.stdout = _old
                    _t = _buf.getvalue()
                    st.text(_t if _t.strip() else "Summary tidak tersedia.")
                except Exception as _e:
                    try: _sys.stdout = _sys.__stdout__
                    except: pass
                    st.warning(f"Summary error: {_e}")

            # ---- Export MGWR ----
            st.markdown("### 💾 Export Parameter MGWR")
            export_mgwr = df.copy()
            for i, vn in enumerate(var_names):
                export_mgwr[f'MGWR_Beta_{vn}'] = params_mgwr[:, i]
                export_mgwr[f'MGWR_SE_{vn}'] = std_errs_mgwr[:, i]
                export_mgwr[f'MGWR_tval_{vn}'] = tvalues_mgwr[:, i]
            export_mgwr['MGWR_LocalR2'] = mgwr_results.localR2.flatten()
            export_mgwr['MGWR_Predicted'] = mgwr_results.predy.flatten()
            export_mgwr['MGWR_Residual'] = mgwr_results.resid_response.flatten()

            csv_buf = io.StringIO()
            export_mgwr.to_csv(csv_buf, index=False)
            st.download_button(
                label="📥 Download Parameter MGWR (CSV)",
                data=csv_buf.getvalue(),
                file_name=f"mgwr_local_parameters_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    else:
        st.warning("⚠️ Pilih variabel terlebih dahulu!")


# ============================================================
# TAB 6: MODEL COMPARISON
# ============================================================
with tab6:
    st.markdown("## 📈 Perbandingan Model: OLS vs GWR vs MGWR")

    has_ols = st.session_state.ols_results is not None
    has_gwr = st.session_state.gwr_results is not None
    has_mgwr = st.session_state.mgwr_results is not None

    if not (has_ols or has_gwr or has_mgwr):
        st.warning("⚠️ Jalankan minimal satu model terlebih dahulu.")

    # ---- Comparison Table ----
    st.markdown("### 📊 Tabel Perbandingan Model")
    comparison_data = []

    if has_ols:
        ols_r = st.session_state.ols_results
        comparison_data.append({
            'Model': 'OLS (Global)',
            'R²': f"{ols_r.rsquared:.6f}",
            'Adj. R²': f"{ols_r.rsquared_adj:.6f}",
            'AIC': f"{ols_r.aic:.4f}",
            'AICc': f"{st.session_state.get('ols_aicc', ols_r.aic):.4f}",
            'BIC': f"{ols_r.bic:.4f}",
            'RSS': f"{np.sum(ols_r.resid**2):.4f}",
            'Sigma²': f"{ols_r.mse_resid:.6f}",
            'Parameters': f"{len(ols_r.params)}"
        })

    if has_gwr:
        gwr_r = st.session_state.gwr_results
        comparison_data.append({
            'Model': 'GWR (Local)',
            'R²': f"{gwr_r.R2:.6f}",
            'Adj. R²': f"{gwr_r.adj_R2:.6f}",
            'AIC': f"{gwr_r.aic:.4f}",
            'AICc': f"{gwr_r.aicc:.4f}",
            'BIC': f"{gwr_r.bic:.4f}",
            'RSS': f"{gwr_r.resid_ss:.4f}",
            'Sigma²': f"{gwr_r.sigma2:.6f}",
            'Parameters (ENP)': f"{gwr_r.ENP:.2f}"
        })

    if has_mgwr:
        mgwr_r = st.session_state.mgwr_results
        comparison_data.append({
            'Model': 'MGWR (Multiscale)',
            'R²': f"{mgwr_r.R2:.6f}",
            'Adj. R²': f"{mgwr_r.adj_R2:.6f}",
            'AIC': f"{mgwr_r.aic:.4f}",
            'AICc': f"{mgwr_r.aicc:.4f}",
            'BIC': f"{mgwr_r.bic:.4f}",
            'RSS': f"{mgwr_r.resid_ss:.4f}",
            'Sigma²': f"{mgwr_r.sigma2:.6f}",
            'Parameters (ENP)': f"{mgwr_r.ENP:.2f}"
        })

    comp_df = pd.DataFrame(comparison_data)
    st.dataframe(comp_df, use_container_width=True)

    # ---- AICc Comparison ----
    st.markdown("### 📊 AICc Comparison")
    st.markdown("""
    <div class="info-box">
    Jika perbedaan AICc > 3, model dengan AICc lebih rendah dianggap lebih baik secara signifikan.
    </div>
    """, unsafe_allow_html=True)

    aicc_values = {}
    if has_ols:
        aicc_values['OLS'] = st.session_state.get('ols_aicc', st.session_state.ols_results.aic)
    if has_gwr:
        aicc_values['GWR'] = st.session_state.gwr_results.aicc
    if has_mgwr:
        aicc_values['MGWR'] = st.session_state.mgwr_results.aicc

    if len(aicc_values) > 1:
        fig_aicc = px.bar(x=list(aicc_values.keys()), y=list(aicc_values.values()),
                          title="Perbandingan AICc",
                          labels={'x': 'Model', 'y': 'AICc'},
                          color=list(aicc_values.keys()),
                          text=[f"{v:.2f}" for v in aicc_values.values()])
        fig_aicc.update_traces(textposition='outside')
        st.plotly_chart(fig_aicc, use_container_width=True)

        best_model = min(aicc_values, key=aicc_values.get)
        diff = max(aicc_values.values()) - min(aicc_values.values())
        st.markdown(f"""
        <div class="success-box">
        🏆 <b>Model terbaik berdasarkan AICc: {best_model}</b> (AICc = {aicc_values[best_model]:.4f})<br>
        Selisih AICc maksimum: {diff:.4f} {"— Signifikan!" if diff > 3 else "— Tidak signifikan (< 3)"}
        </div>
        """, unsafe_allow_html=True)

    # ---- R² Comparison ----
    st.markdown("### 📊 R² Comparison")
    r2_values = {}
    adj_r2_values = {}
    if has_ols:
        r2_values['OLS'] = st.session_state.ols_results.rsquared
        adj_r2_values['OLS'] = st.session_state.ols_results.rsquared_adj
    if has_gwr:
        r2_values['GWR'] = st.session_state.gwr_results.R2
        adj_r2_values['GWR'] = st.session_state.gwr_results.adj_R2
    if has_mgwr:
        r2_values['MGWR'] = st.session_state.mgwr_results.R2
        adj_r2_values['MGWR'] = st.session_state.mgwr_results.adj_R2

    if len(r2_values) > 1:
        fig_r2 = go.Figure()
        fig_r2.add_trace(go.Bar(x=list(r2_values.keys()), y=list(r2_values.values()),
                                name='R²', text=[f"{v:.4f}" for v in r2_values.values()],
                                textposition='outside'))
        fig_r2.add_trace(go.Bar(x=list(adj_r2_values.keys()), y=list(adj_r2_values.values()),
                                name='Adj. R²', text=[f"{v:.4f}" for v in adj_r2_values.values()],
                                textposition='outside'))
        fig_r2.update_layout(title="Perbandingan R² dan Adjusted R²", barmode='group',
                             yaxis_title="Value")
        st.plotly_chart(fig_r2, use_container_width=True)

    # ---- Residual Comparison ----
    st.markdown("### 📊 Perbandingan Residual")
    residual_data = []

    if has_ols:
        ols_resid = st.session_state.ols_results.resid.flatten()
        for r in ols_resid:
            residual_data.append({'Model': 'OLS', 'Residual': r})

    if has_gwr:
        gwr_resid = st.session_state.gwr_results.resid_response.flatten()
        for r in gwr_resid:
            residual_data.append({'Model': 'GWR', 'Residual': r})

    if has_mgwr:
        mgwr_resid = st.session_state.mgwr_results.resid_response.flatten()
        for r in mgwr_resid:
            residual_data.append({'Model': 'MGWR', 'Residual': r})

    if residual_data:
        resid_df = pd.DataFrame(residual_data)
        fig_resid_comp = px.box(resid_df, x='Model', y='Residual',
                                 title="Distribusi Residual per Model",
                                 color='Model')
        st.plotly_chart(fig_resid_comp, use_container_width=True)

    # ---- Predicted vs Actual Comparison ----
    st.markdown("### 📊 Predicted vs Actual")
    y_actual = df[st.session_state.dep_var].values

    fig_pred = go.Figure()
    min_val = y_actual.min()
    max_val = y_actual.max()
    fig_pred.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                                   mode='lines', line=dict(color='black', dash='dash'),
                                   name='Perfect Fit'))

    if has_ols:
        fig_pred.add_trace(go.Scatter(x=st.session_state.ols_results.fittedvalues.flatten(),
                                       y=y_actual, mode='markers', name='OLS',
                                       marker=dict(color='blue', size=5, opacity=0.5)))
    if has_gwr:
        fig_pred.add_trace(go.Scatter(x=st.session_state.gwr_results.predy.flatten(),
                                       y=y_actual, mode='markers', name='GWR',
                                       marker=dict(color='green', size=5, opacity=0.5)))
    if has_mgwr:
        fig_pred.add_trace(go.Scatter(x=st.session_state.mgwr_results.predy.flatten(),
                                       y=y_actual, mode='markers', name='MGWR',
                                       marker=dict(color='red', size=5, opacity=0.5)))

    fig_pred.update_layout(title="Predicted vs Actual — All Models",
                           xaxis_title="Predicted", yaxis_title="Actual", height=500)
    st.plotly_chart(fig_pred, use_container_width=True)

    # ---- ANOVA-like F-test GWR vs OLS ----
    if has_ols and has_gwr:
        st.markdown("### 📐 F-Test: GWR vs OLS")
        ols_rss = np.sum(st.session_state.ols_results.resid ** 2)
        gwr_rss = st.session_state.gwr_results.resid_ss
        n = len(y_actual)
        k_ols = len(st.session_state.ols_results.params)
        enp_gwr = st.session_state.gwr_results.ENP

        # Leung et al. (2000) approximate F-test
        numerator = (ols_rss - gwr_rss) / (enp_gwr - k_ols)
        denominator = gwr_rss / (n - enp_gwr)

        if denominator > 0 and numerator > 0:
            f_stat = numerator / denominator
            df1 = enp_gwr - k_ols
            df2 = n - enp_gwr
            from scipy.stats import f as f_dist
            f_pval = 1 - f_dist.cdf(f_stat, df1, df2)

            st.write(f"**F-statistic:** `{f_stat:.4f}`")
            st.write(f"**df1 (ENP_GWR - k_OLS):** `{df1:.2f}`")
            st.write(f"**df2 (n - ENP_GWR):** `{df2:.2f}`")
            st.write(f"**P-value:** `{f_pval:.6f}`")

            if f_pval < 0.05:
                st.markdown("""
                <div class="success-box">
                ✅ <b>GWR signifikan lebih baik dari OLS</b> (p < 0.05).
                Terdapat variasi spasial signifikan dalam hubungan variabel.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="warning-box">
                ⚠️ GWR tidak signifikan lebih baik dari OLS (p ≥ 0.05).
                Hubungan variabel mungkin cukup stasioner secara spasial.
                </div>
                """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# END OF: gwr_app_part2.py
# START OF: gwr_app_part3.py
# ════════════════════════════════════════════════════════════

# ============================================================
# FILE 3: gwr_app_part3.py
# ============================================================


# ============================================================
# APLIKASI ANALISIS GWR LENGKAP - BAGIAN 3/3
# VISUALISASI PETA + LAPORAN OTOMATIS
# ============================================================

# (folium/geopandas imports at top)

# ============================================================
# TAB 7: VISUALISASI PETA
# ============================================================
with tab7:
    st.markdown("## 🗺️ Visualisasi Peta Koefisien & Diagnostics")

    if not FOLIUM_AVAILABLE:
        st.error("Library `folium` dan `streamlit-folium` belum terinstall.")
        st.code("pip install folium streamlit-folium branca geopandas", language="bash")
    elif not GEOPANDAS_AVAILABLE:
        st.error("Library `geopandas` belum terinstall.")
        st.code("pip install geopandas", language="bash")
    elif st.session_state.dep_var and st.session_state.indep_vars:
        dep = st.session_state.dep_var
        indeps = st.session_state.indep_vars

        lon_col = st.session_state.lon_col
        lat_col = st.session_state.lat_col

        st.markdown("""
        <div class="info-box">
        Peta ini menggunakan geometri titik (point) langsung dari koordinat.
        Untuk shapefile polygon, Anda bisa meng-join hasil CSV ke shapefile di GIS (QGIS/ArcGIS)
        dan membuat choropleth di sana.
        </div>
        """, unsafe_allow_html=True)

        # Base GeoDataFrame from points
        gdf = gpd.GeoDataFrame(df.copy(), geometry=gpd.points_from_xy(df[lon_col], df[lat_col]), crs="EPSG:4326")

        # Helper: create colormap
        def create_colormap(series, cmap_name='RdYlBu', n_bins=7):
            vmin, vmax = float(series.min()), float(series.max())
            if vmin == vmax:
                vmin -= 1
                vmax += 1
            rgba = plt.get_cmap(cmap_name)(np.linspace(0, 1, n_bins))
            hex_colors = [mcolors.to_hex(c) for c in rgba]
            return cm.LinearColormap(colors=hex_colors, vmin=vmin, vmax=vmax)

        # Select source (OLS / GWR / MGWR)
        st.markdown("### 🎯 Pilih Model & Layer")
        model_options = []
        if st.session_state.ols_results is not None:
            model_options.append("OLS")
        if st.session_state.gwr_results is not None:
            model_options.append("GWR")
        if st.session_state.mgwr_results is not None:
            model_options.append("MGWR")

        if not model_options:
            st.warning("⚠️ Belum ada model yang dijalankan!")
            st.stop()

        map_model = st.selectbox("Pilih model untuk divisualisasikan:", model_options)

        # Build dataframe with fields to map
        map_df = gdf.copy()

        if map_model == "OLS":
            ols = st.session_state.ols_results
            map_df['Predicted'] = ols.fittedvalues.flatten()
            map_df['Residual'] = ols.resid.flatten()
        elif map_model == "GWR":
            gwr = st.session_state.gwr_results
            params = gwr.params
            tvals = gwr.tvalues
            localR2 = gwr.localR2

            var_names = ['Intercept'] + indeps
            for i, vn in enumerate(var_names):
                map_df[f'GWR_Beta_{vn}'] = params[:, i]
                map_df[f'GWR_t_{vn}'] = tvals[:, i]
            map_df['Local_R2'] = localR2.flatten()
            map_df['Predicted'] = gwr.predy.flatten()
            map_df['Residual'] = gwr.resid_response.flatten()
            map_df['Std_Residual'] = gwr.std_res.flatten()
        else:  # MGWR
            mgwr = st.session_state.mgwr_results
            params_m = mgwr.params
            tvals_m = mgwr.tvalues
            localR2_m = mgwr.localR2

            var_names = ['Intercept'] + indeps
            for i, vn in enumerate(var_names):
                map_df[f'MGWR_Beta_{vn}'] = params_m[:, i]
                map_df[f'MGWR_t_{vn}'] = tvals_m[:, i]
            map_df['Local_R2'] = localR2_m.flatten()
            map_df['Predicted'] = mgwr.predy.flatten()
            map_df['Residual'] = mgwr.resid_response.flatten()

        # Layer type selection
        layer_type = st.selectbox("Pilih layer yang ingin dipetakan:", [
            "Predicted vs Observed",
            "Residual & Std Residual",
            "Local R²",
        ] + [f"Koefisien: {v}" for v in (['Intercept'] + indeps)]
          + [f"t-stat: {v}" for v in (['Intercept'] + indeps)])

        # Center map
        center_lat = map_df[lat_col].mean()
        center_lon = map_df[lon_col].mean()

        # Build Folium map
        m = folium.Map(location=[center_lat, center_lon], zoom_start=7, tiles='CartoDB positron')

        if layer_type == "Predicted vs Observed":
            # Color by residual sign, size by |residual|
            max_abs_res = np.abs(map_df['Residual']).max()
            for _, row in map_df.iterrows():
                res = row['Residual']
                color = 'green' if res >= 0 else 'red'
                radius = 4 + 8 * (abs(res) / max_abs_res) if max_abs_res > 0 else 4
                popup = folium.Popup(
                    f"<b>{dep}</b>: {row[dep]}<br>"
                    f"<b>Predicted</b>: {row['Predicted']:.3f}<br>"
                    f"<b>Residual</b>: {res:.3f}", max_width=250
                )
                folium.CircleMarker(
                    location=[row[lat_col], row[lon_col]],
                    radius=radius,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.7,
                    popup=popup
                ).add_to(m)

            folium.LayerControl().add_to(m)
            st_folium(m, width=1200, height=600)

        elif layer_type == "Residual & Std Residual":
            col_option = st.radio("Tampilkan:", ["Residual", "Std_Residual" if 'Std_Residual' in map_df.columns else "Residual"],
                                  horizontal=True)
            series = map_df[col_option]
            cmap = create_colormap(series, cmap_name='RdBu_r')

            for _, row in map_df.iterrows():
                val = row[col_option]
                color = cmap(val)
                popup = folium.Popup(
                    f"<b>{dep}</b>: {row[dep]}<br>"
                    f"<b>{col_option}</b>: {val:.3f}", max_width=250
                )
                folium.CircleMarker(
                    location=[row[lat_col], row[lon_col]],
                    radius=6,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.8,
                    popup=popup
                ).add_to(m)

            cmap.caption = f"{col_option}"
            m.add_child(cmap)
            folium.LayerControl().add_to(m)
            st_folium(m, width=1200, height=600)

        elif layer_type == "Local R²":
            if 'Local_R2' not in map_df.columns:
                st.warning("Local R² hanya tersedia untuk GWR/MGWR.")
            else:
                series = map_df['Local_R2']
                cmap = create_colormap(series, cmap_name='YlGnBu')

                for _, row in map_df.iterrows():
                    val = row['Local_R2']
                    try:
                        color = cmap(float(val)) if pd.notna(val) else '#808080'
                    except Exception:
                        color = '#808080'
                    popup = folium.Popup(
                        f"<b>Local R²</b>: {val:.3f}<br>"
                        f"<b>{dep}</b>: {row[dep]}", max_width=250
                    )
                    folium.CircleMarker(
                        location=[row[lat_col], row[lon_col]],
                        radius=6,
                        color=color,
                        fill=True,
                        fill_color=color,
                        fill_opacity=0.8,
                        popup=popup
                    ).add_to(m)

                cmap.caption = "Local R²"
                m.add_child(cmap)
                folium.LayerControl().add_to(m)
                st_folium(m, width=1200, height=600)

        else:
            # Coef or t-stat layer
            if layer_type.startswith("Koefisien:"):
                target_var = layer_type.replace("Koefisien: ", "")
                if map_model == "GWR":
                    field = f'GWR_Beta_{target_var}'
                elif map_model == "MGWR":
                    field = f'MGWR_Beta_{target_var}'
                else:
                    st.warning("Koefisien lokal hanya tersedia untuk GWR/MGWR.")
            else:
                target_var = layer_type.replace("t-stat: ", "")
                if map_model == "GWR":
                    field = f'GWR_t_{target_var}'
                elif map_model == "MGWR":
                    field = f'MGWR_t_{target_var}'
                else:
                    st.warning("t-stat lokal hanya tersedia untuk GWR/MGWR.")

            if field not in map_df.columns:
                st.warning(f"Field {field} tidak ditemukan.")
            else:
                series = map_df[field]
                cmap = create_colormap(series, cmap_name='RdBu_r')

                sig_mask = None
                if layer_type.startswith("t-stat:"):
                    sig_mask = np.abs(series) > 1.96

                for _, row in map_df.iterrows():
                    val = row[field]
                    try:
                        color = cmap(float(val)) if pd.notna(val) else '#808080'
                    except Exception:
                        color = '#808080'
                    popup = folium.Popup(
                        f"<b>{field}</b>: {val:.3f}<br>"
                        f"<b>{dep}</b>: {row[dep]}", max_width=250
                    )
                    # Halo hijau untuk signifikan (mirip ArcGIS) [web:7][web:37]
                    if sig_mask is not None and val is not None and abs(val) > 1.96:
                        folium.CircleMarker(
                            location=[row[lat_col], row[lon_col]],
                            radius=7,
                            color='lime',
                            fill=False,
                            opacity=0.9,
                        ).add_to(m)

                    folium.CircleMarker(
                        location=[row[lat_col], row[lon_col]],
                        radius=5,
                        color=color,
                        fill=True,
                        fill_color=color,
                        fill_opacity=0.8,
                        popup=popup
                    ).add_to(m)

                cmap.caption = field
                m.add_child(cmap)
                folium.LayerControl().add_to(m)
                st_folium(m, width=1200, height=600)
    else:
        st.warning("⚠️ Pilih variabel dependen dan independen terlebih dahulu!")


# ============================================================
# TAB 8: LAPORAN & EXPORT
# ============================================================
with tab8:
    st.markdown("## 📄 Laporan Otomatis & Export Konfigurasi")

    dep = st.session_state.dep_var
    indeps = st.session_state.indep_vars

    st.markdown("### 🧾 Ringkasan Naratif (Auto Report)")

    # Build narrative summary
    report_lines = []
    report_lines.append(f"LAPORAN ANALISIS GWR / MGWR")
    report_lines.append("" )
    report_lines.append(f"Tanggal analisis   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Jumlah observasi   : {len(df)}")
    report_lines.append(f"Variabel dependen  : {dep}")
    report_lines.append(f"Variabel independen: {', '.join(indeps)}")
    report_lines.append(f"Koordinat          : {st.session_state.lon_col} (lon), {st.session_state.lat_col} (lat)")
    report_lines.append("" )

    # Assumption summary from Tab 2 (recompute quick version)
    y_rep = df[dep].values
    X_rep = df[indeps].values
    ols_rep = sm.OLS(y_rep, sm.add_constant(X_rep)).fit()

    from statsmodels.stats.diagnostic import het_breuschpagan
    from scipy.stats import shapiro

    resid_rep = ols_rep.resid

    # Normality (Shapiro)
    if len(resid_rep) <= 5000:
        sw_stat, sw_p = shapiro(resid_rep)
    else:
        sw_stat, sw_p = shapiro(np.random.choice(resid_rep, 5000, replace=False))

    # Heteroskedasticity
    bp_stat, bp_p, _, _ = het_breuschpagan(resid_rep, sm.add_constant(X_rep))

    # Autocorrelation
    dw_stat = durbin_watson(resid_rep)

    # VIF
    X_const_rep = sm.add_constant(X_rep)
    vifs = [variance_inflation_factor(X_const_rep, i+1) for i in range(len(indeps))]
    max_vif = max(vifs)

    report_lines.append("HASIL UJI ASUMSI (MODEL OLS GLOBAL)")
    report_lines.append(f"- Normalitas residual (Shapiro-Wilk): stat = {sw_stat:.4f}, p-value = {sw_p:.4f}")
    report_lines.append(f"- Heteroskedastisitas (Breusch-Pagan): stat = {bp_stat:.4f}, p-value = {bp_p:.4f}")
    report_lines.append(f"- Autokorelasi (Durbin-Watson): DW = {dw_stat:.4f}")
    report_lines.append(f"- Multikolinearitas: VIF maksimum = {max_vif:.4f}")
    report_lines.append("" )

    # Model summaries
    if st.session_state.ols_results is not None:
        ols = st.session_state.ols_results
        report_lines.append("RINGKASAN MODEL OLS (GLOBAL)")
        report_lines.append(f"- R²           : {ols.rsquared:.4f}")
        report_lines.append(f"- Adj. R²      : {ols.rsquared_adj:.4f}")
        report_lines.append(f"- AIC          : {ols.aic:.4f}")
        report_lines.append(f"- AICc         : {st.session_state.get('ols_aicc', ols.aic):.4f}")
        report_lines.append(f"- BIC          : {ols.bic:.4f}")
        report_lines.append(f"- F-statistic  : {ols.fvalue:.4f} (p = {ols.f_pvalue:.6f})")
        report_lines.append("" )

    if st.session_state.gwr_results is not None:
        gwr = st.session_state.gwr_results
        report_lines.append("RINGKASAN MODEL GWR")
        report_lines.append(f"- Bandwidth    : {st.session_state.gwr_bw}")
        report_lines.append(f"- R²           : {gwr.R2:.4f}")
        report_lines.append(f"- Adj. R²      : {gwr.adj_R2:.4f}")
        report_lines.append(f"- AIC          : {gwr.aic:.4f}")
        report_lines.append(f"- AICc         : {gwr.aicc:.4f}")
        report_lines.append(f"- BIC          : {gwr.bic:.4f}")
        report_lines.append(f"- ENP          : {gwr.ENP:.4f}")
        report_lines.append("" )

        # Local R2 summary
        localR2 = gwr.localR2
        report_lines.append("Ringkasan Local R² GWR:")
        report_lines.append(f"- Mean         : {localR2.mean():.4f}")
        report_lines.append(f"- Median       : {np.median(localR2):.4f}")
        report_lines.append(f"- Min, Max     : ({localR2.min():.4f}, {localR2.max():.4f})")
        report_lines.append("" )

    if st.session_state.mgwr_results is not None:
        mgwr = st.session_state.mgwr_results
        report_lines.append("RINGKASAN MODEL MGWR")
        report_lines.append(f"- R²           : {mgwr.R2:.4f}")
        report_lines.append(f"- Adj. R²      : {mgwr.adj_R2:.4f}")
        report_lines.append(f"- AIC          : {mgwr.aic:.4f}")
        report_lines.append(f"- AICc         : {mgwr.aicc:.4f}")
        report_lines.append(f"- BIC          : {mgwr.bic:.4f}")
        report_lines.append(f"- ENP          : {mgwr.ENP:.4f}")
        report_lines.append("" )

        # Bandwidth interpretation
        bw = st.session_state.mgwr_bw
        var_names = ['Intercept'] + indeps
        report_lines.append("Bandwidth per variabel (MGWR):")
        for name, b in zip(var_names, bw):
            if b >= len(df) * 0.8:
                scale = "Global"
            elif b >= len(df) * 0.3:
                scale = "Regional"
            else:
                scale = "Lokal"
            report_lines.append(f"- {name}: BW = {b:.1f} → skala {scale}")
        report_lines.append("" )

    # Simple interpretation hints inspired by ArcGIS docs [web:35][web:37]
    report_lines.append("CATATAN INTERPRETASI (RINGKAS)")
    report_lines.append("- Bandwidth besar → hubungan lebih stabil (global), bandwidth kecil → hubungan sangat lokal.")
    report_lines.append("- Local R² tinggi → model menjelaskan variasi dengan baik di lokasi tersebut.")
    report_lines.append("- Koefisien positif besar → peningkatan variabel X meningkatkan Y secara kuat di lokasi tersebut.")
    report_lines.append("- Koefisien negatif besar → peningkatan variabel X menurunkan Y secara kuat di lokasi tersebut.")
    report_lines.append("- t-statistik |t| > 1.96 → koefisien signifikan pada α ≈ 5%. Lokasi ini penting untuk variabel tersebut.")
    report_lines.append("- Pola klaster residual → kemungkinan ada variabel penting yang belum dimasukkan ke model.")

    report_text = "\n".join(report_lines)

    st.text_area("Preview Laporan (Markdown/Teks)", report_text, height=400)

    st.download_button(
        label="📥 Download Laporan (.txt)",
        data=report_text,
        file_name=f"laporan_GWR_MGWR_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain",
        use_container_width=True
    )

    # ========================================================
    # EXPORT KONFIGURASI MODEL (JSON)
    # ========================================================
    st.markdown("### ⚙️ Export Konfigurasi Model (JSON)")

    config = {
        'datetime': datetime.now().isoformat(),
        'n_obs': len(df),
        'dependent_var': dep,
        'independent_vars': indeps,
        'lon_col': st.session_state.lon_col,
        'lat_col': st.session_state.lat_col,
        'kernel_type': st.session_state.get('kernel_type', None),
        'bw_fixed': st.session_state.get('bw_fixed', None),
        'search_method': st.session_state.get('search_method', None),
        'criterion': st.session_state.get('criterion', None),
        'alpha_level': st.session_state.get('alpha_level', 0.05),
        'n_monte_carlo': st.session_state.get('n_monte_carlo', 100),
        'run_mgwr': st.session_state.get('run_mgwr', True),
        'standardize_mgwr': st.session_state.get('standardize_mgwr', True),
        'compare_ols': st.session_state.get('compare_ols', True),
        'ols': {
            'has_model': st.session_state.ols_results is not None,
            'rsq': float(st.session_state.ols_results.rsquared) if st.session_state.ols_results is not None else None,
            'aic': float(st.session_state.ols_results.aic) if st.session_state.ols_results is not None else None,
            'aicc': float(st.session_state.get('ols_aicc', np.nan)),
        },
        'gwr': {
            'has_model': st.session_state.gwr_results is not None,
            'bw': float(st.session_state.gwr_bw) if st.session_state.gwr_results is not None else None,
            'rsq': float(st.session_state.gwr_results.R2) if st.session_state.gwr_results is not None else None,
            'aicc': float(st.session_state.gwr_results.aicc) if st.session_state.gwr_results is not None else None,
        },
        'mgwr': {
            'has_model': st.session_state.mgwr_results is not None,
            'rsq': float(st.session_state.mgwr_results.R2) if st.session_state.mgwr_results is not None else None,
            'aicc': float(st.session_state.mgwr_results.aicc) if st.session_state.mgwr_results is not None else None,
        }
    }

    config_json = json.dumps(config, indent=4)

    st.code(config_json, language='json')

    st.download_button(
        label="📥 Download Konfigurasi Model (JSON)",
        data=config_json,
        file_name=f"konfigurasi_GWR_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
        use_container_width=True
    )

