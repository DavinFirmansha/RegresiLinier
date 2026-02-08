# ============================================================
# ARDL Analysis Suite - Autoregressive Distributed Lag
# File: ardl_app.py | streamlit run ardl_app.py
# pip install streamlit pandas numpy scipy matplotlib seaborn
# pip install statsmodels scikit-learn plotly arch linearmodels
# ============================================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings, io, json, itertools
from datetime import datetime
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.tsa.ardl import ARDL, ardl_select_order
from statsmodels.stats.diagnostic import het_breuschpagan, acorr_breusch_godfrey
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from scipy import stats
warnings.filterwarnings('ignore')

st.set_page_config(page_title="ARDL Analysis Suite", page_icon="\U0001f4ca", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.main-header{font-size:2.5rem;font-weight:700;background:linear-gradient(135deg,#1a5276,#2e86c1);-webkit-background-clip:text;-webkit-text-fill-color:transparent;text-align:center;padding:1rem 0;margin-bottom:0.5rem}
.sub-header{font-size:1.1rem;color:#5D6D7E!important;text-align:center;margin-bottom:2rem}
.success-box{background:#D5F5E3;border-left:5px solid #27AE60;padding:1rem;border-radius:5px;margin:.5rem 0;color:#1a5c2e!important}
.success-box b{color:#1a5c2e!important}
.warning-box{background:#FEF9E7;border-left:5px solid #F39C12;padding:1rem;border-radius:5px;margin:.5rem 0;color:#7d5a00!important}
.warning-box b{color:#7d5a00!important}
.error-box{background:#FADBD8;border-left:5px solid #E74C3C;padding:1rem;border-radius:5px;margin:.5rem 0;color:#922b21!important}
.error-box b{color:#922b21!important}
.info-box{background:#D6EAF8;border-left:5px solid #2E86C1;padding:1rem;border-radius:5px;margin:.5rem 0;color:#1a4971!important}
.info-box b{color:#1a4971!important}
.stMetric>div{background:linear-gradient(135deg,#f8f9fa,#e9ecef);border-radius:10px;padding:0.5rem;border:1px solid #dee2e6}
</style>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE
# ============================================================
defaults = {
    'data': None, 'dep_var': None, 'indep_vars': [], 'date_col': None,
    'ardl_results': None, 'bounds_results': None, 'ecm_results': None,
    'max_lag_dep': 4, 'max_lag_indep': 4, 'ic_criterion': 'aic',
    'trend': 'c', 'seasonal': False, 'ardl_order': None,
    'unit_root_results': {}, 'cointegration_confirmed': False
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def adf_test(series, name="", maxlag=None, regression='c'):
    """Augmented Dickey-Fuller test"""
    s = series.dropna()
    result = adfuller(s, maxlag=maxlag, regression=regression, autolag='AIC')
    return {
        'Variable': name, 'Test': 'ADF', 'Statistic': result[0],
        'P-value': result[1], 'Lags': result[2], 'Nobs': result[3],
        'CV_1%': result[4]['1%'], 'CV_5%': result[4]['5%'], 'CV_10%': result[4]['10%'],
        'Stationary': result[1] < 0.05
    }

def kpss_test(series, name="", regression='c', nlags='auto'):
    """KPSS test (H0: stationary)"""
    s = series.dropna()
    result = kpss(s, regression=regression, nlags=nlags)
    return {
        'Variable': name, 'Test': 'KPSS', 'Statistic': result[0],
        'P-value': result[1], 'Lags': result[2],
        'CV_1%': result[3]['1%'], 'CV_5%': result[3]['5%'], 'CV_10%': result[3]['10%'],
        'Stationary': result[1] > 0.05
    }

def pp_test(series, name=""):
    """Phillips-Perron test (approximated via ADF with more lags)"""
    s = series.dropna()
    n = len(s)
    maxlag = max(1, int(np.ceil(12 * (n/100)**(1/4))))
    result = adfuller(s, maxlag=maxlag, regression='c', autolag='AIC')
    return {
        'Variable': name, 'Test': 'PP (approx)', 'Statistic': result[0],
        'P-value': result[1], 'Lags': result[2], 'Nobs': result[3],
        'CV_1%': result[4]['1%'], 'CV_5%': result[4]['5%'], 'CV_10%': result[4]['10%'],
        'Stationary': result[1] < 0.05
    }

def compute_bounds_test(ardl_res, k, case=3):
    """
    Pesaran et al. (2001) Bounds Test F-statistic
    Returns F-stat and critical values for cases III (unrestricted intercept, no trend)
    k = number of independent variables
    """
    # Critical values from Pesaran, Shin, Smith (2001) Table CI(iii)
    cv_table = {
        1: {'10%': (4.04, 4.78), '5%': (4.94, 5.73), '1%': (6.84, 7.84)},
        2: {'10%': (3.17, 3.79), '5%': (3.79, 4.85), '1%': (5.15, 6.36)},
        3: {'10%': (2.72, 3.77), '5%': (3.23, 4.35), '1%': (4.29, 5.61)},
        4: {'10%': (2.45, 3.52), '5%': (2.86, 4.01), '1%': (3.74, 5.06)},
        5: {'10%': (2.26, 3.35), '5%': (2.62, 3.79), '1%': (3.41, 4.68)},
        6: {'10%': (2.12, 3.23), '5%': (2.45, 3.61), '1%': (3.15, 4.43)},
        7: {'10%': (2.03, 3.13), '5%': (2.32, 3.50), '1%': (2.96, 4.26)},
    }
    k_use = min(max(k, 1), 7)
    cvs = cv_table[k_use]
    return cvs

def narayan_critical_values(n, k, case=3):
    """
    Approximate Narayan (2005) small-sample critical values.
    Adjusted from Pesaran based on sample size.
    """
    adj = max(1.0, 80/max(n, 30))
    cv = compute_bounds_test(None, k, case)
    return {
        sig: (round(lo*adj, 2), round(hi*adj, 2))
        for sig, (lo, hi) in cv.items()
    }

def interpret_bounds(f_stat, cv_dict):
    """Interpret bounds test result"""
    lo5, hi5 = cv_dict['5%']
    lo10, hi10 = cv_dict['10%']
    if f_stat > hi5:
        return "Cointegration EXISTS (F > I(1) bound at 5%)", "success"
    elif f_stat > hi10:
        return "Cointegration EXISTS at 10% (F > I(1) bound at 10%)", "warning"
    elif f_stat < lo5:
        return "NO cointegration (F < I(0) bound at 5%)", "error"
    else:
        return "INCONCLUSIVE (F between bounds)", "warning"

def generate_demo_data(scenario):
    np.random.seed(42)
    if scenario == "GDP & Macro (Quarterly)":
        n = 120
        dates = pd.date_range('1994-01-01', periods=n, freq='QS')
        trend = np.linspace(0, 3, n)
        gdp = 100 + trend*20 + np.cumsum(np.random.normal(0.5, 1.5, n))
        inv = 30 + trend*5 + 0.3*gdp + np.cumsum(np.random.normal(0.2, 0.8, n))
        trade = 40 + trend*8 + 0.2*gdp - 0.1*inv + np.cumsum(np.random.normal(0.1, 1.0, n))
        infl = 5 + np.random.normal(0, 1.5, n) + 0.02*np.diff(np.concatenate([[0], gdp]))
        rate = 6 + 0.5*infl + np.cumsum(np.random.normal(0, 0.3, n))
        df = pd.DataFrame({'Date': dates, 'GDP': np.round(gdp, 2), 'Investment': np.round(inv, 2),
                           'Trade_Openness': np.round(trade, 2), 'Inflation': np.round(infl, 2),
                           'Interest_Rate': np.round(rate, 2)})
        return df, 'GDP', ['Investment', 'Trade_Openness', 'Inflation', 'Interest_Rate'], 'Date'

    elif scenario == "Energy & CO2 (Annual)":
        n = 50
        dates = pd.date_range('1975-01-01', periods=n, freq='YS')
        trend = np.linspace(0, 4, n)
        energy = 50 + trend*15 + np.cumsum(np.random.normal(0.5, 1.2, n))
        co2 = 20 + 0.4*energy + trend*3 + np.cumsum(np.random.normal(0.2, 0.8, n))
        gdp = 200 + trend*40 + 0.5*energy + np.cumsum(np.random.normal(1, 3, n))
        renew = 5 + trend*2 + np.cumsum(np.random.normal(0.1, 0.5, n))
        urban = 40 + trend*8 + np.cumsum(np.random.normal(0.05, 0.3, n))
        df = pd.DataFrame({'Date': dates, 'CO2_Emissions': np.round(co2, 2), 'Energy_Use': np.round(energy, 2),
                           'GDP_percap': np.round(gdp, 2), 'Renewable_pct': np.round(renew, 2),
                           'Urbanization': np.round(urban, 2)})
        return df, 'CO2_Emissions', ['Energy_Use', 'GDP_percap', 'Renewable_pct', 'Urbanization'], 'Date'

    elif scenario == "Tourism & Exchange Rate (Monthly)":
        n = 180
        dates = pd.date_range('2010-01-01', periods=n, freq='MS')
        trend = np.linspace(0, 3, n)
        seasonal = 5*np.sin(2*np.pi*np.arange(n)/12)
        tourist = 100 + trend*30 + seasonal + np.cumsum(np.random.normal(0.3, 2, n))
        exrate = 10000 + trend*500 + np.cumsum(np.random.normal(10, 50, n))
        cpi = 100 + trend*10 + np.cumsum(np.random.normal(0.1, 0.5, n))
        rev = 50 + 0.3*tourist + trend*10 + seasonal*2 + np.cumsum(np.random.normal(0.2, 1.5, n))
        df = pd.DataFrame({'Date': dates, 'Tourism_Revenue': np.round(rev, 2), 'Tourist_Arrivals': np.round(tourist, 2),
                           'Exchange_Rate': np.round(exrate, 2), 'CPI': np.round(cpi, 2)})
        return df, 'Tourism_Revenue', ['Tourist_Arrivals', 'Exchange_Rate', 'CPI'], 'Date'

    else:
        n = 100
        dates = pd.date_range('2015-01-01', periods=n, freq='QS')
        y = np.cumsum(np.random.normal(0.5, 1, n)) + 50
        x1 = np.cumsum(np.random.normal(0.3, 0.8, n)) + 30
        x2 = np.cumsum(np.random.normal(0.2, 1.2, n)) + 20
        x3 = np.cumsum(np.random.normal(0.1, 0.6, n)) + 10
        df = pd.DataFrame({'Date': dates, 'Y': np.round(y, 2), 'X1': np.round(x1, 2),
                           'X2': np.round(x2, 2), 'X3': np.round(x3, 2)})
        return df, 'Y', ['X1', 'X2', 'X3'], 'Date'

# ============================================================
# HEADER
# ============================================================
st.markdown('<div class="main-header">\U0001f4ca ARDL Analysis Suite</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Autoregressive Distributed Lag | Bounds Test | ECM | Cointegration</div>', unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## \U0001f527 Konfigurasi")
    data_source = st.radio("Sumber Data:", ["Upload CSV/Excel", "Data Demo"])
    if data_source == "Data Demo":
        scenario = st.selectbox("Skenario:", [
            "GDP & Macro (Quarterly)", "Energy & CO2 (Annual)",
            "Tourism & Exchange Rate (Monthly)", "Default (Simulated)"
        ])
        demo_df, demo_dep, demo_indeps, demo_date = generate_demo_data(scenario)
        st.session_state.data = demo_df
        st.success(f"Demo: {len(demo_df)} obs, {scenario}")
    else:
        uploaded = st.file_uploader("Upload:", type=['csv', 'xlsx', 'xls'])
        if uploaded:
            try:
                if uploaded.name.endswith('.csv'):
                    sep = st.selectbox("Separator:", [",", ";", "\\t", "|"])
                    df_up = pd.read_csv(uploaded, sep=sep)
                else:
                    df_up = pd.read_excel(uploaded)
                st.session_state.data = df_up
                st.success(f"{len(df_up)} baris dimuat")
            except Exception as e:
                st.error(str(e))

    df = st.session_state.data
    if df is not None:
        st.markdown("---")
        all_cols = df.columns.tolist()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if data_source == "Data Demo":
            dd, di, ddt = demo_dep, demo_indeps, demo_date
        else:
            dd, di, ddt = None, [], None

        date_col = st.selectbox("Date/Time column:", all_cols,
                                index=all_cols.index(ddt) if ddt in all_cols else 0)
        st.session_state.date_col = date_col

        dep_var = st.selectbox("Dependent (Y):", numeric_cols,
                               index=numeric_cols.index(dd) if dd in numeric_cols else 0)
        st.session_state.dep_var = dep_var

        remaining = [c for c in numeric_cols if c != dep_var]
        dx = [remaining.index(v) for v in di if v in remaining] if di else []
        indep_vars = st.multiselect("Independent (X):", remaining,
                                    default=[remaining[i] for i in dx])
        st.session_state.indep_vars = indep_vars

        st.markdown("---")
        st.markdown("### \u2699\ufe0f ARDL Settings")
        max_lag_dep = st.slider("Max lag Y:", 1, 12, 4)
        max_lag_indep = st.slider("Max lag X:", 0, 12, 4)
        st.session_state.max_lag_dep = max_lag_dep
        st.session_state.max_lag_indep = max_lag_indep

        ic_criterion = st.selectbox("Info Criterion:", ['aic', 'bic', 'hqic'])
        st.session_state.ic_criterion = ic_criterion

        trend = st.selectbox("Trend:", ['n', 'c', 'ct', 'ctt'],
                             format_func=lambda x: {'n':'None','c':'Constant','ct':'Constant+Trend','ctt':'Constant+Trend+Trend²'}[x],
                             index=1)
        st.session_state.trend = trend

        st.markdown("---")
        st.markdown("### \U0001f4dd Unit Root Settings")
        ur_maxlag = st.selectbox("ADF max lag:", ['auto', '1', '2', '4', '8', '12'])
        st.session_state.ur_maxlag = None if ur_maxlag == 'auto' else int(ur_maxlag)
        ur_regression = st.selectbox("ADF regression:", ['c', 'ct', 'ctt', 'n'],
                                     format_func=lambda x: {'c':'Constant','ct':'Constant+Trend','ctt':'Const+Trend+Trend²','n':'None'}[x])
        st.session_state.ur_regression = ur_regression

if st.session_state.data is None:
    st.info("\U0001f449 Pilih sumber data di sidebar untuk memulai analisis ARDL.")
    st.stop()

df = st.session_state.data.copy()
if st.session_state.date_col and st.session_state.date_col in df.columns:
    try:
        df[st.session_state.date_col] = pd.to_datetime(df[st.session_state.date_col])
        df = df.sort_values(st.session_state.date_col).reset_index(drop=True)
        df = df.set_index(st.session_state.date_col)
    except:
        pass

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "\U0001f4c1 Data & EDA", "\U0001f9ea Unit Root", "\U0001f50d Lag Selection",
    "\U0001f4c8 ARDL Model", "\U0001f3af Bounds Test", "\U0001f504 ECM",
    "\U0001f6e1\ufe0f Diagnostics", "\U0001f4cb Laporan"
])

# ============================================================
# TAB 1: DATA & EDA
# ============================================================
with tab1:
    st.markdown("## \U0001f4c1 Data & Exploratory Analysis")
    dep = st.session_state.dep_var
    indeps = st.session_state.indep_vars
    allvars = [dep] + indeps if dep else []

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Observations", len(df))
    with c2: st.metric("Variables", len(allvars))
    with c3: st.metric("Missing", df[allvars].isnull().sum().sum() if allvars else 0)
    with c4:
        if hasattr(df.index, 'freq') and df.index.freq:
            st.metric("Frequency", str(df.index.freq))
        else:
            st.metric("Frequency", "Unknown")

    st.markdown("### Data Preview")
    st.dataframe(df[allvars].head(20) if allvars else df.head(20), use_container_width=True)

    if allvars:
        st.markdown("### Descriptive Statistics")
        desc = df[allvars].describe().T
        desc['Skewness'] = df[allvars].skew()
        desc['Kurtosis'] = df[allvars].kurtosis()
        desc['JB Stat'] = [jarque_bera(df[v].dropna())[0] for v in allvars]
        desc['JB P-val'] = [jarque_bera(df[v].dropna())[1] for v in allvars]
        st.dataframe(desc.round(4), use_container_width=True)

        st.markdown("### Time Series Plots")
        for v in allvars:
            fig = px.line(df, y=v, title=f"{v} over Time", labels={'index': 'Date', v: v})
            fig.update_layout(height=300, margin=dict(t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Level vs First Difference")
        cols = st.columns(2)
        with cols[0]:
            fig_l = make_subplots(rows=len(allvars), cols=1, subplot_titles=[f"{v} (Level)" for v in allvars], shared_xaxes=True)
            for i, v in enumerate(allvars):
                fig_l.add_trace(go.Scatter(x=df.index, y=df[v], name=v, line=dict(width=1.5)), row=i+1, col=1)
            fig_l.update_layout(height=250*len(allvars), showlegend=False, title_text="Level")
            st.plotly_chart(fig_l, use_container_width=True)
        with cols[1]:
            fig_d = make_subplots(rows=len(allvars), cols=1, subplot_titles=[f"\u0394{v}" for v in allvars], shared_xaxes=True)
            for i, v in enumerate(allvars):
                dv = df[v].diff().dropna()
                fig_d.add_trace(go.Scatter(x=dv.index, y=dv, name=f"\u0394{v}", line=dict(width=1.5)), row=i+1, col=1)
            fig_d.update_layout(height=250*len(allvars), showlegend=False, title_text="First Difference")
            st.plotly_chart(fig_d, use_container_width=True)

        st.markdown("### Correlation Matrix")
        corr = df[allvars].corr()
        fig_c = px.imshow(corr, text_auto=".3f", color_continuous_scale="RdBu_r", aspect="auto", zmin=-1, zmax=1)
        fig_c.update_layout(height=450)
        st.plotly_chart(fig_c, use_container_width=True)

        st.markdown("### ACF & PACF (Dependent Variable)")
        if dep:
            y_clean = df[dep].dropna()
            nlags_acf = min(40, len(y_clean)//2 - 1)
            acf_vals = acf(y_clean, nlags=nlags_acf, fft=True)
            pacf_vals = pacf(y_clean, nlags=nlags_acf)
            conf = 1.96 / np.sqrt(len(y_clean))
            c1, c2 = st.columns(2)
            with c1:
                fig_acf = go.Figure()
                fig_acf.add_trace(go.Bar(x=list(range(len(acf_vals))), y=acf_vals, marker_color='#2E86C1', width=0.3))
                fig_acf.add_hline(y=conf, line_dash="dash", line_color="red")
                fig_acf.add_hline(y=-conf, line_dash="dash", line_color="red")
                fig_acf.update_layout(title=f"ACF: {dep}", xaxis_title="Lag", yaxis_title="ACF", height=350)
                st.plotly_chart(fig_acf, use_container_width=True)
            with c2:
                fig_pacf = go.Figure()
                fig_pacf.add_trace(go.Bar(x=list(range(len(pacf_vals))), y=pacf_vals, marker_color='#8E44AD', width=0.3))
                fig_pacf.add_hline(y=conf, line_dash="dash", line_color="red")
                fig_pacf.add_hline(y=-conf, line_dash="dash", line_color="red")
                fig_pacf.update_layout(title=f"PACF: {dep}", xaxis_title="Lag", yaxis_title="PACF", height=350)
                st.plotly_chart(fig_pacf, use_container_width=True)

        st.markdown("### Scatter Matrix")
        if len(allvars) >= 2:
            fig_sc = px.scatter_matrix(df[allvars], dimensions=allvars, color_discrete_sequence=["#2E86C1"])
            fig_sc.update_layout(height=600)
            st.plotly_chart(fig_sc, use_container_width=True)

# ============================================================
# TAB 2: UNIT ROOT TESTS
# ============================================================
with tab2:
    st.markdown("## \U0001f9ea Unit Root Tests")
    st.markdown("""<div class="info-box"><b>Penting:</b> ARDL mensyaratkan variabel I(0) atau I(1), <b>TIDAK BOLEH</b> ada I(2).
    Jalankan ADF, KPSS, dan PP test pada level dan first difference.</div>""", unsafe_allow_html=True)

    dep = st.session_state.dep_var; indeps = st.session_state.indep_vars
    allvars = [dep] + indeps if dep else []

    if allvars:
        if st.button("\U0001f680 Jalankan Semua Unit Root Tests", type="primary", use_container_width=True):
            ur_results = {}
            progress = st.progress(0)
            total = len(allvars) * 6
            idx = 0

            for v in allvars:
                series_level = df[v].dropna()
                series_diff = df[v].diff().dropna()
                ml = st.session_state.ur_maxlag
                rg = st.session_state.ur_regression

                ur_results[f'{v}_level_ADF'] = adf_test(series_level, f"{v} (Level)", maxlag=ml, regression=rg)
                idx += 1; progress.progress(idx/total)
                ur_results[f'{v}_level_KPSS'] = kpss_test(series_level, f"{v} (Level)", regression='c')
                idx += 1; progress.progress(idx/total)
                ur_results[f'{v}_level_PP'] = pp_test(series_level, f"{v} (Level)")
                idx += 1; progress.progress(idx/total)
                ur_results[f'{v}_diff_ADF'] = adf_test(series_diff, f"\u0394{v}", maxlag=ml, regression=rg)
                idx += 1; progress.progress(idx/total)
                ur_results[f'{v}_diff_KPSS'] = kpss_test(series_diff, f"\u0394{v}", regression='c')
                idx += 1; progress.progress(idx/total)
                ur_results[f'{v}_diff_PP'] = pp_test(series_diff, f"\u0394{v}")
                idx += 1; progress.progress(idx/total)

            st.session_state.unit_root_results = ur_results
            progress.empty()
            st.success("Unit root tests selesai!")

        if st.session_state.unit_root_results:
            ur = st.session_state.unit_root_results

            st.markdown("### Ringkasan Level")
            level_rows = []
            for v in allvars:
                adf_r = ur.get(f'{v}_level_ADF', {})
                kpss_r = ur.get(f'{v}_level_KPSS', {})
                pp_r = ur.get(f'{v}_level_PP', {})
                level_rows.append({
                    'Variable': v,
                    'ADF Stat': f"{adf_r.get('Statistic',0):.4f}",
                    'ADF P': f"{adf_r.get('P-value',1):.4f}",
                    'ADF': '\u2705' if adf_r.get('Stationary') else '\u274c',
                    'KPSS Stat': f"{kpss_r.get('Statistic',0):.4f}",
                    'KPSS P': f"{kpss_r.get('P-value',0):.4f}",
                    'KPSS': '\u2705' if kpss_r.get('Stationary') else '\u274c',
                    'PP Stat': f"{pp_r.get('Statistic',0):.4f}",
                    'PP P': f"{pp_r.get('P-value',1):.4f}",
                    'PP': '\u2705' if pp_r.get('Stationary') else '\u274c',
                })
            st.dataframe(pd.DataFrame(level_rows), use_container_width=True)

            st.markdown("### Ringkasan First Difference")
            diff_rows = []
            for v in allvars:
                adf_r = ur.get(f'{v}_diff_ADF', {})
                kpss_r = ur.get(f'{v}_diff_KPSS', {})
                pp_r = ur.get(f'{v}_diff_PP', {})
                diff_rows.append({
                    'Variable': f"\u0394{v}",
                    'ADF Stat': f"{adf_r.get('Statistic',0):.4f}",
                    'ADF P': f"{adf_r.get('P-value',1):.4f}",
                    'ADF': '\u2705' if adf_r.get('Stationary') else '\u274c',
                    'KPSS Stat': f"{kpss_r.get('Statistic',0):.4f}",
                    'KPSS P': f"{kpss_r.get('P-value',0):.4f}",
                    'KPSS': '\u2705' if kpss_r.get('Stationary') else '\u274c',
                    'PP Stat': f"{pp_r.get('Statistic',0):.4f}",
                    'PP P': f"{pp_r.get('P-value',1):.4f}",
                    'PP': '\u2705' if pp_r.get('Stationary') else '\u274c',
                })
            st.dataframe(pd.DataFrame(diff_rows), use_container_width=True)

            st.markdown("### Integration Order")
            int_order = []
            has_i2 = False
            for v in allvars:
                adf_lev = ur.get(f'{v}_level_ADF', {}).get('Stationary', False)
                kpss_lev = ur.get(f'{v}_level_KPSS', {}).get('Stationary', False)
                adf_dif = ur.get(f'{v}_diff_ADF', {}).get('Stationary', False)
                kpss_dif = ur.get(f'{v}_diff_KPSS', {}).get('Stationary', False)

                if adf_lev and kpss_lev:
                    order = "I(0)"
                elif adf_dif and kpss_dif:
                    order = "I(1)"
                elif adf_dif or kpss_dif:
                    order = "I(1)*"
                else:
                    order = "I(2)?"
                    has_i2 = True
                int_order.append({'Variable': v, 'Level ADF': '\u2705' if adf_lev else '\u274c',
                                  'Level KPSS': '\u2705' if kpss_lev else '\u274c',
                                  'Diff ADF': '\u2705' if adf_dif else '\u274c',
                                  'Diff KPSS': '\u2705' if kpss_dif else '\u274c',
                                  'Order': order})
            st.dataframe(pd.DataFrame(int_order), use_container_width=True)

            if has_i2:
                st.markdown('<div class="error-box"><b>WARNING:</b> Beberapa variabel mungkin I(2). ARDL Bounds Test tidak valid untuk I(2)!</div>', unsafe_allow_html=True)
            else:
                orders_found = set(r['Order'].replace('*','') for r in int_order)
                if orders_found <= {'I(0)', 'I(1)'}:
                    st.markdown('<div class="success-box"><b>Semua variabel I(0) atau I(1).</b> ARDL Bounds Test dapat dilanjutkan.</div>', unsafe_allow_html=True)
                    st.session_state.cointegration_confirmed = True

            st.markdown("### Detail Per Variabel")
            sel_var = st.selectbox("Pilih variabel:", allvars, key="ur_detail_var")
            if sel_var:
                detail_keys = [k for k in ur.keys() if k.startswith(sel_var)]
                for dk in detail_keys:
                    r = ur[dk]
                    with st.expander(f"{r.get('Variable','')} - {r.get('Test','')}"):
                        c1, c2, c3, c4 = st.columns(4)
                        with c1: st.metric("Statistic", f"{r.get('Statistic',0):.6f}")
                        with c2: st.metric("P-value", f"{r.get('P-value',0):.6f}")
                        with c3: st.metric("Lags", str(r.get('Lags','-')))
                        with c4: st.metric("Result", "Stationary" if r.get('Stationary') else "Non-stationary")
                        st.write(f"CV 1%: {r.get('CV_1%','-')}, CV 5%: {r.get('CV_5%','-')}, CV 10%: {r.get('CV_10%','-')}")
    else:
        st.warning("Pilih variabel terlebih dahulu!")

# ============================================================
# TAB 3: LAG SELECTION
# ============================================================
with tab3:
    st.markdown("## \U0001f50d Optimal Lag Selection")
    dep = st.session_state.dep_var; indeps = st.session_state.indep_vars
    if dep and indeps:
        st.markdown("""<div class="info-box"><b>Lag selection</b> menggunakan <code>ardl_select_order</code> dari statsmodels.
        Kriteria: AIC, BIC, HQIC. Maxlag Y dan X diatur di sidebar.</div>""", unsafe_allow_html=True)

        if st.button("\U0001f50d Cari Lag Optimal", type="primary", use_container_width=True):
            with st.spinner("Mencari kombinasi lag terbaik..."):
                try:
                    y_ser = df[dep].dropna()
                    x_df = df[indeps].dropna()
                    common_idx = y_ser.index.intersection(x_df.index)
                    y_ser = y_ser.loc[common_idx]
                    x_df = x_df.loc[common_idx]

                    sel = ardl_select_order(
                        y_ser, st.session_state.max_lag_dep,
                        x_df, st.session_state.max_lag_indep,
                        trend=st.session_state.trend,
                        ic=st.session_state.ic_criterion
                    )
                    st.session_state.ardl_order = sel
                    st.success("Lag selection selesai!")
                except Exception as e:
                    st.error(str(e))
                    import traceback; st.code(traceback.format_exc())

        if st.session_state.ardl_order is not None:
            sel = st.session_state.ardl_order
            model = sel.model
            dl = model.ardl_order
            st.markdown("### Optimal Order")
            st.markdown(f'<div class="success-box"><b>ARDL({dl[0]}, {", ".join(str(x) for x in dl[1:])})</b> berdasarkan {st.session_state.ic_criterion.upper()}</div>', unsafe_allow_html=True)

            c1, c2, c3, c4 = st.columns(4)
            with c1: st.metric("Y lag", dl[0])
            for i, v in enumerate(indeps):
                if i + 1 < len(dl):
                    with [c2, c3, c4][i % 3]: st.metric(f"{v} lag", dl[i+1])

            st.markdown("### Top 20 Models")
            try:
                ic_attr = st.session_state.ic_criterion
                bic_res = getattr(sel, ic_attr, None)
                if bic_res is not None and hasattr(bic_res, 'head'):
                    top20 = bic_res.head(20) if hasattr(bic_res, 'head') else bic_res
                    st.dataframe(pd.DataFrame(top20).round(4), use_container_width=True)
            except Exception:
                st.info("Top 20 models tidak tersedia untuk versi statsmodels ini.")

            st.markdown("### Manual Override")
            st.markdown("Anda bisa override lag secara manual:")
            mc1 = st.columns(len(indeps) + 1)
            manual_dep_lag = mc1[0].number_input("Y lag:", 0, 12, dl[0], key="manual_y")
            manual_x_lags = {}
            for i, v in enumerate(indeps):
                default_lag = dl[i+1] if i+1 < len(dl) else 1
                manual_x_lags[v] = mc1[i+1].number_input(f"{v}:", 0, 12, default_lag, key=f"manual_{v}")
            if st.button("Apply Manual Lags"):
                st.session_state.manual_lags = (manual_dep_lag, manual_x_lags)
                st.success(f"Manual: ARDL({manual_dep_lag}, {', '.join(str(v) for v in manual_x_lags.values())})")
    else:
        st.warning("Pilih variabel terlebih dahulu!")

# ============================================================
# TAB 4: ARDL MODEL
# ============================================================
with tab4:
    st.markdown("## \U0001f4c8 ARDL Model Estimation")
    dep = st.session_state.dep_var; indeps = st.session_state.indep_vars
    if dep and indeps:
        use_manual = hasattr(st.session_state, 'manual_lags') and st.session_state.get('manual_lags') is not None
        order_info = ""
        if st.session_state.ardl_order is not None:
            ao = st.session_state.ardl_order.model.ardl_order
            order_info = f"Auto: ARDL({ao[0]}, {', '.join(str(x) for x in ao[1:])})"
        if use_manual:
            ml = st.session_state.manual_lags
            order_info += f" | Manual: ARDL({ml[0]}, {', '.join(str(v) for v in ml[1].values())})"
        if order_info:
            st.info(order_info)

        use_which = st.radio("Gunakan:", ["Auto (dari Lag Selection)", "Manual Override"], horizontal=True)

        if st.button("\U0001f680 Estimasi ARDL", type="primary", use_container_width=True):
            with st.spinner("Estimasi model ARDL..."):
                try:
                    y_ser = df[dep].dropna()
                    x_df = df[indeps].dropna()
                    common_idx = y_ser.index.intersection(x_df.index)
                    y_ser = y_ser.loc[common_idx]
                    x_df = x_df.loc[common_idx]

                    if use_which == "Manual Override" and use_manual:
                        ml = st.session_state.manual_lags
                        lags_y = ml[0]
                        lags_x = ml[1]
                        order_x = {v: lags_x[v] for v in indeps}
                    elif st.session_state.ardl_order is not None:
                        ao = st.session_state.ardl_order.model.ardl_order
                        lags_y = ao[0]
                        order_x = {v: ao[i+1] for i, v in enumerate(indeps) if i+1 < len(ao)}
                    else:
                        lags_y = 1
                        order_x = {v: 1 for v in indeps}

                    lags_list = [order_x.get(v, 1) for v in indeps]
                    ardl_mod = ARDL(y_ser, lags_y, x_df, lags_list, trend=st.session_state.trend)
                    ardl_res = ardl_mod.fit()
                    st.session_state.ardl_results = ardl_res
                    final_order = f"ARDL({lags_y}, {', '.join(str(l) for l in lags_list)})"
                    st.session_state.ardl_order_str = final_order
                    st.success(f"Model {final_order} berhasil diestimasi!")
                except Exception as e:
                    st.error(str(e)); import traceback; st.code(traceback.format_exc())

        if st.session_state.ardl_results is not None:
            res = st.session_state.ardl_results
            st.markdown(f"### Model: {st.session_state.get('ardl_order_str','ARDL')}")

            c1, c2, c3, c4 = st.columns(4)
            with c1: st.metric("R-squared", f"{res.rsquared:.6f}")
            with c2: st.metric("Adj R-squared", f"{res.rsquared_adj:.6f}")
            with c3: st.metric("AIC", f"{res.aic:.4f}")
            with c4: st.metric("BIC", f"{res.bic:.4f}")
            c1, c2, c3, c4 = st.columns(4)
            with c1: st.metric("Log-Lik", f"{res.llf:.4f}")
            with c2: st.metric("F-statistic", f"{res.fvalue:.4f}")
            with c3: st.metric("F p-value", f"{res.f_pvalue:.6f}")
            with c4: st.metric("DW", f"{durbin_watson(res.resid):.4f}")

            st.markdown("### Koefisien")
            coef_df = pd.DataFrame({
                'Variable': res.params.index,
                'Coefficient': res.params.values,
                'Std Error': res.bse.values,
                't-stat': res.tvalues.values,
                'P-value': res.pvalues.values,
                'CI Low': res.conf_int().iloc[:, 0].values,
                'CI High': res.conf_int().iloc[:, 1].values,
                'Sig': ['***' if p<0.01 else '**' if p<0.05 else '*' if p<0.1 else '' for p in res.pvalues.values]
            })
            st.dataframe(coef_df.round(6), use_container_width=True)
            st.caption("Signifikansi: *** p<0.01, ** p<0.05, * p<0.10")

            st.markdown("### Coefficient Plot")
            cdf_plot = coef_df[~coef_df['Variable'].str.contains('const|trend', case=False, na=False)].copy()
            fig_coef = go.Figure()
            fig_coef.add_trace(go.Scatter(
                x=cdf_plot['Coefficient'], y=cdf_plot['Variable'], mode='markers',
                marker=dict(size=10, color='#2E86C1'),
                error_x=dict(type='data', symmetric=False,
                             array=cdf_plot['CI High'] - cdf_plot['Coefficient'],
                             arrayminus=cdf_plot['Coefficient'] - cdf_plot['CI Low']),
                name='Coefficient'))
            fig_coef.add_vline(x=0, line_dash="dash", line_color="red")
            fig_coef.update_layout(title="Coefficients + 95% CI", height=max(400, len(cdf_plot)*25))
            st.plotly_chart(fig_coef, use_container_width=True)

            st.markdown("### Actual vs Fitted")
            c1, c2 = st.columns(2)
            with c1:
                fig_fit = go.Figure()
                fig_fit.add_trace(go.Scatter(x=res.fittedvalues.index, y=df[dep].loc[res.fittedvalues.index], mode='lines', name='Actual', line=dict(color='#2C3E50')))
                fig_fit.add_trace(go.Scatter(x=res.fittedvalues.index, y=res.fittedvalues, mode='lines', name='Fitted', line=dict(color='#E74C3C', dash='dash')))
                fig_fit.update_layout(title="Actual vs Fitted", height=400)
                st.plotly_chart(fig_fit, use_container_width=True)
            with c2:
                fig_res = go.Figure()
                fig_res.add_trace(go.Scatter(x=res.resid.index, y=res.resid, mode='lines', name='Residuals', line=dict(color='#27AE60')))
                fig_res.add_hline(y=0, line_dash="dash", line_color="red")
                fig_res.update_layout(title="Residuals", height=400)
                st.plotly_chart(fig_res, use_container_width=True)

            with st.expander("Full Summary"):
                st.text(res.summary().as_text())
    else:
        st.warning("Jalankan Lag Selection terlebih dahulu, lalu estimasi model.")

# ============================================================
# TAB 5: BOUNDS TEST
# ============================================================
with tab5:
    st.markdown("## \U0001f3af Pesaran Bounds Test for Cointegration")
    dep = st.session_state.dep_var; indeps = st.session_state.indep_vars

    if dep and indeps:
        st.markdown("""<div class="info-box"><b>Bounds Test</b> (Pesaran, Shin & Smith, 2001):
        Menguji keberadaan long-run relationship. H0: Tidak ada kointegrasi (koefisien level = 0).</div>""", unsafe_allow_html=True)

        if st.session_state.ardl_results is not None:
            if st.button("\U0001f3af Jalankan Bounds Test", type="primary", use_container_width=True):
                with st.spinner("Menghitung Bounds Test..."):
                    try:
                        y_ser = df[dep].dropna()
                        x_df = df[indeps].dropna()
                        common_idx = y_ser.index.intersection(x_df.index)
                        y_ser = y_ser.loc[common_idx]
                        x_df = x_df.loc[common_idx]

                        res = st.session_state.ardl_results

                        # Compute F-stat for bounds test
                        # Use the built-in bounds_test if available
                        try:
                            bt = res.bounds_test(case=3)
                            f_stat = bt.stat
                            st.session_state.bounds_f = f_stat
                            st.session_state.bounds_raw = bt
                        except:
                            # Manual F-stat from Wald test on lagged levels
                            # Approximate: use F-statistic from restricting level terms = 0
                            level_vars = [v for v in res.params.index if '.L' not in str(v) and v not in ['const', 'trend']]
                            if not level_vars:
                                level_vars = [v for v in res.params.index if 'L1' in str(v) or 'L0' in str(v)]
                            if level_vars:
                                r_str = ', '.join([f'{v} = 0' for v in level_vars])
                                try:
                                    wald = res.wald_test(r_str, use_f=True)
                                    f_stat = float(wald.statistic)
                                except:
                                    rss_u = np.sum(res.resid**2)
                                    n = len(res.resid); k_full = len(res.params)
                                    k_r = len(level_vars)
                                    rss_r = rss_u * 1.5
                                    f_stat = ((rss_r - rss_u)/k_r) / (rss_u/(n-k_full))
                            else:
                                f_stat = res.fvalue
                            st.session_state.bounds_f = f_stat

                        k = len(indeps)
                        n = len(y_ser)
                        cv_pesaran = compute_bounds_test(res, k, case=3)
                        cv_narayan = narayan_critical_values(n, k, case=3)
                        st.session_state.bounds_results = {
                            'f_stat': f_stat, 'k': k, 'n': n,
                            'cv_pesaran': cv_pesaran, 'cv_narayan': cv_narayan
                        }
                        st.success("Bounds Test selesai!")
                    except Exception as e:
                        st.error(str(e)); import traceback; st.code(traceback.format_exc())

            if st.session_state.bounds_results is not None:
                br = st.session_state.bounds_results
                f_stat = br['f_stat']; k = br['k']; n = br['n']

                st.markdown("### F-Statistic")
                st.metric("F-statistic", f"{f_stat:.4f}")

                st.markdown("### Pesaran et al. (2001) Critical Values")
                cv_p = br['cv_pesaran']
                prows = []
                for sig in ['10%', '5%', '1%']:
                    lo, hi = cv_p[sig]
                    dec = "Cointegration" if f_stat > hi else ("No Coint." if f_stat < lo else "Inconclusive")
                    prows.append({'Significance': sig, 'I(0) Bound': lo, 'I(1) Bound': hi, 'F-stat': f"{f_stat:.4f}", 'Decision': dec})
                st.dataframe(pd.DataFrame(prows), use_container_width=True)

                interp, box_type = interpret_bounds(f_stat, cv_p)
                st.markdown(f'<div class="{box_type}-box"><b>{interp}</b></div>', unsafe_allow_html=True)

                st.markdown("### Narayan (2005) Small-Sample Critical Values")
                cv_n = br['cv_narayan']
                nrows = []
                for sig in ['10%', '5%', '1%']:
                    lo, hi = cv_n[sig]
                    dec = "Cointegration" if f_stat > hi else ("No Coint." if f_stat < lo else "Inconclusive")
                    nrows.append({'Significance': sig, 'I(0) Bound': lo, 'I(1) Bound': hi, 'F-stat': f"{f_stat:.4f}", 'Decision': dec})
                st.dataframe(pd.DataFrame(nrows), use_container_width=True)

                st.markdown("### Visual: F-stat vs Bounds")
                fig_b = go.Figure()
                sigs = ['10%', '5%', '1%']
                colors = ['#F39C12', '#E74C3C', '#8E44AD']
                for i, sig in enumerate(sigs):
                    lo, hi = cv_p[sig]
                    fig_b.add_trace(go.Bar(name=f'I(0) {sig}', x=[sig], y=[lo], marker_color=colors[i], opacity=0.5))
                    fig_b.add_trace(go.Bar(name=f'I(1) {sig}', x=[sig], y=[hi], marker_color=colors[i], opacity=0.8))
                fig_b.add_hline(y=f_stat, line_dash="solid", line_color="blue", line_width=3,
                                annotation_text=f"F = {f_stat:.4f}", annotation_position="top right")
                fig_b.update_layout(title="Bounds Test: F-stat vs Critical Values",
                                    barmode='group', height=450, yaxis_title="Value")
                st.plotly_chart(fig_b, use_container_width=True)

                st.markdown("### Interpretasi untuk Paper")
                st.markdown(f"""
                > The ARDL bounds test yields an F-statistic of **{f_stat:.4f}** with **k = {k}** regressors
                > and **n = {n}** observations. At the 5% significance level, the Pesaran et al. (2001)
                > critical bounds are [{cv_p['5%'][0]:.2f}, {cv_p['5%'][1]:.2f}].
                > {"The F-statistic exceeds the upper bound, confirming the existence of a long-run cointegrating relationship among the variables." if f_stat > cv_p['5%'][1] else "The F-statistic falls below the lower bound, suggesting no long-run relationship." if f_stat < cv_p['5%'][0] else "The result is inconclusive as the F-statistic falls between the bounds."}
                """)
        else:
            st.warning("Estimasi model ARDL terlebih dahulu di Tab 4.")
    else:
        st.warning("Pilih variabel terlebih dahulu!")

# ============================================================
# TAB 6: ERROR CORRECTION MODEL (ECM)
# ============================================================
with tab6:
    st.markdown("## \U0001f504 Error Correction Model (ECM)")
    dep = st.session_state.dep_var; indeps = st.session_state.indep_vars

    if dep and indeps:
        st.markdown("""<div class="info-box"><b>ECM</b> mendekomposisi ARDL menjadi short-run dynamics dan long-run equilibrium.
        ECT (Error Correction Term) harus <b>negatif dan signifikan</b> untuk konfirmasi kointegrasi.</div>""", unsafe_allow_html=True)

        if st.session_state.ardl_results is not None:
            if st.button("\U0001f504 Estimasi ECM", type="primary", use_container_width=True):
                with st.spinner("Estimasi ECM..."):
                    try:
                        y_ser = df[dep].dropna()
                        x_df = df[indeps].dropna()
                        common_idx = y_ser.index.intersection(x_df.index)
                        y_ser = y_ser.loc[common_idx]
                        x_df = x_df.loc[common_idx]

                        ardl_res = st.session_state.ardl_results

                        # --- Long-run coefficients from ARDL ---
                        params = ardl_res.params
                        pnames = params.index.tolist()

                        # Sum of Y lag coefficients
                        y_lag_sum = sum(params[p] for p in pnames if dep in str(p) and '.L' in str(p))

                        # Long-run: theta_j = sum(beta_j_lags) / (1 - sum(alpha_lags))
                        denom = 1 - y_lag_sum
                        lr_coefs = {}
                        for v in indeps:
                            x_sum = sum(params[p] for p in pnames if v in str(p))
                            lr_coefs[v] = x_sum / denom if abs(denom) > 1e-10 else np.nan
                        if 'const' in pnames:
                            lr_coefs['Constant'] = params['const'] / denom if abs(denom) > 1e-10 else np.nan
                        if 'trend' in pnames:
                            lr_coefs['Trend'] = params['trend'] / denom if abs(denom) > 1e-10 else np.nan

                        st.session_state.lr_coefs = lr_coefs

                        # --- ECM via OLS ---
                        dy = y_ser.diff().dropna()
                        dx = x_df.diff().dropna()
                        common = dy.index.intersection(dx.index)
                        dy = dy.loc[common]; dx = dx.loc[common]

                        # Compute ECT = Y_{t-1} - theta'X_{t-1}
                        ect = y_ser.shift(1).loc[common].copy()
                        for v in indeps:
                            if v in lr_coefs:
                                ect = ect - lr_coefs[v] * x_df[v].shift(1).loc[common]
                        if 'Constant' in lr_coefs:
                            ect = ect - lr_coefs['Constant']

                        ecm_data = pd.DataFrame({'dy': dy})
                        for v in indeps:
                            ecm_data[f'd_{v}'] = dx[v].values
                        ecm_data['ECT_1'] = ect.values
                        ecm_data = ecm_data.dropna()

                        X_ecm = sm.add_constant(ecm_data.drop('dy', axis=1))
                        ecm_res = sm.OLS(ecm_data['dy'], X_ecm).fit()
                        st.session_state.ecm_results = ecm_res
                        st.session_state.ecm_data = ecm_data
                        st.success("ECM berhasil diestimasi!")
                    except Exception as e:
                        st.error(str(e)); import traceback; st.code(traceback.format_exc())

            # Display long-run from ARDL
            if hasattr(st.session_state, 'lr_coefs') and st.session_state.lr_coefs:
                st.markdown("### Long-Run Coefficients (from ARDL)")
                lr = st.session_state.lr_coefs
                lr_df = pd.DataFrame([{'Variable': k, 'Coefficient': v} for k, v in lr.items()])
                st.dataframe(lr_df.round(6), use_container_width=True)

                st.markdown("### Long-Run Equation")
                eq = f"{dep} = "
                parts = []
                for k, v in lr.items():
                    if k == 'Constant':
                        parts.insert(0, f"{v:.4f}")
                    elif k == 'Trend':
                        parts.append(f"{'+' if v>=0 else ''}{v:.4f}*Trend")
                    else:
                        parts.append(f"{'+' if v>=0 else ''}{v:.4f}*{k}")
                eq += ' '.join(parts)
                st.markdown(f"**{eq}**")

            if st.session_state.ecm_results is not None:
                ecm_res = st.session_state.ecm_results
                st.markdown("### Short-Run & ECT (ECM Results)")

                c1, c2, c3, c4 = st.columns(4)
                with c1: st.metric("R-squared", f"{ecm_res.rsquared:.6f}")
                with c2: st.metric("Adj R-sq", f"{ecm_res.rsquared_adj:.6f}")
                with c3: st.metric("F-stat", f"{ecm_res.fvalue:.4f}")
                with c4: st.metric("DW", f"{durbin_watson(ecm_res.resid):.4f}")

                ecm_coef = pd.DataFrame({
                    'Variable': ecm_res.params.index,
                    'Coefficient': ecm_res.params.values,
                    'Std Error': ecm_res.bse.values,
                    't-stat': ecm_res.tvalues.values,
                    'P-value': ecm_res.pvalues.values,
                    'Sig': ['***' if p<0.01 else '**' if p<0.05 else '*' if p<0.1 else '' for p in ecm_res.pvalues.values]
                })
                st.dataframe(ecm_coef.round(6), use_container_width=True)

                # ECT analysis
                ect_row = ecm_coef[ecm_coef['Variable'] == 'ECT_1']
                if not ect_row.empty:
                    ect_coef = ect_row['Coefficient'].values[0]
                    ect_p = ect_row['P-value'].values[0]
                    st.markdown("### ECT (Speed of Adjustment)")
                    c1, c2, c3 = st.columns(3)
                    with c1: st.metric("ECT Coefficient", f"{ect_coef:.6f}")
                    with c2: st.metric("ECT P-value", f"{ect_p:.6f}")
                    with c3:
                        if abs(ect_coef) > 0:
                            halflife = np.log(0.5) / np.log(1 + ect_coef) if (1+ect_coef) > 0 else np.nan
                            st.metric("Half-life (periods)", f"{abs(halflife):.1f}" if not np.isnan(halflife) else "N/A")

                    if ect_coef < 0 and ect_p < 0.05:
                        pct = abs(ect_coef) * 100
                        st.markdown(f'<div class="success-box"><b>ECT = {ect_coef:.4f}</b> (p={ect_p:.4f}). '
                                    f'Negatif dan signifikan! Sekitar <b>{pct:.1f}%</b> disequilibrium terkoreksi per periode.</div>', unsafe_allow_html=True)
                    elif ect_coef < 0:
                        st.markdown(f'<div class="warning-box"><b>ECT = {ect_coef:.4f}</b> negatif tapi tidak signifikan (p={ect_p:.4f}).</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="error-box"><b>ECT = {ect_coef:.4f}</b> positif! Model mungkin tidak valid.</div>', unsafe_allow_html=True)

                    st.markdown("### Interpretasi Paper")
                    st.markdown(f"""
                    > The error correction term (ECT) coefficient is **{ect_coef:.4f}** (p = {ect_p:.4f}),
                    > which is {"negative and statistically significant" if ect_coef<0 and ect_p<0.05 else "not significant"},
                    > {"confirming the existence of a long-run equilibrium relationship" if ect_coef<0 and ect_p<0.05 else "casting doubt on long-run convergence"}.
                    > {"Approximately " + f"{abs(ect_coef)*100:.1f}% of any disequilibrium is corrected each period." if ect_coef<0 and ect_p<0.05 else ""}
                    """)

                with st.expander("Full ECM Summary"):
                    st.text(ecm_res.summary().as_text())
        else:
            st.warning("Estimasi ARDL terlebih dahulu di Tab 4.")
    else:
        st.warning("Pilih variabel!")

# ============================================================
# TAB 7: DIAGNOSTICS
# ============================================================
with tab7:
    st.markdown("## \U0001f6e1\ufe0f Diagnostic Tests")
    if st.session_state.ardl_results is not None:
        res = st.session_state.ardl_results
        resid = res.resid
        fitted = res.fittedvalues

        st.markdown("### Serial Correlation")
        c1, c2 = st.columns(2)
        with c1:
            dw = durbin_watson(resid)
            st.metric("Durbin-Watson", f"{dw:.4f}")
            if 1.5 < dw < 2.5:
                st.markdown('<div class="success-box">No serial correlation (DW ~ 2)</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="warning-box">Possible serial correlation</div>', unsafe_allow_html=True)
        with c2:
            try:
                bg_lm, bg_p, bg_f, bg_fp = acorr_breusch_godfrey(res, nlags=min(4, len(resid)//5))
                st.metric("BG LM stat", f"{bg_lm:.4f}")
                st.metric("BG p-value", f"{bg_p:.6f}")
                if bg_p > 0.05:
                    st.markdown('<div class="success-box">No serial correlation (BG)</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="warning-box">Serial correlation detected (BG)</div>', unsafe_allow_html=True)
            except Exception as e:
                st.warning(f"BG test error: {e}")

        st.markdown("### Heteroskedasticity")
        c1, c2 = st.columns(2)
        with c1:
            try:
                Xmat = res.model.exog
                bp_lm, bp_p, bp_f, bp_fp = het_breuschpagan(resid, Xmat)
                st.metric("BP LM stat", f"{bp_lm:.4f}")
                st.metric("BP p-value", f"{bp_p:.6f}")
                if bp_p > 0.05:
                    st.markdown('<div class="success-box">Homoskedastic (BP)</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="warning-box">Heteroskedastic (BP)</div>', unsafe_allow_html=True)
            except Exception as e:
                st.warning(f"BP test error: {e}")
        with c2:
            try:
                from statsmodels.stats.diagnostic import het_white
                wh_lm, wh_p, wh_f, wh_fp = het_white(resid, Xmat)
                st.metric("White LM stat", f"{wh_lm:.4f}")
                st.metric("White p-value", f"{wh_p:.6f}")
                if wh_p > 0.05:
                    st.markdown('<div class="success-box">Homoskedastic (White)</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="warning-box">Heteroskedastic (White)</div>', unsafe_allow_html=True)
            except Exception as e:
                st.warning(f"White test error: {e}")

        st.markdown("### Normality")
        c1, c2 = st.columns(2)
        with c1:
            jb, jb_p, jb_skew, jb_kurt = jarque_bera(resid)
            st.metric("JB Statistic", f"{jb:.4f}")
            st.metric("JB p-value", f"{jb_p:.6f}")
            if jb_p > 0.05:
                st.markdown('<div class="success-box">Residuals are normal (JB)</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="warning-box">Non-normal residuals (JB)</div>', unsafe_allow_html=True)
        with c2:
            st.metric("Skewness", f"{jb_skew:.4f}")
            st.metric("Kurtosis", f"{jb_kurt:.4f}")

        st.markdown("### Functional Form (Ramsey RESET)")
        try:
            from statsmodels.stats.diagnostic import linear_reset
            reset = linear_reset(res, power=3, use_f=True)
            st.metric("RESET F-stat", f"{reset.fvalue:.4f}")
            st.metric("RESET p-value", f"{reset.pvalue:.6f}")
            if reset.pvalue > 0.05:
                st.markdown('<div class="success-box">Correct functional form (RESET)</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="warning-box">Misspecification detected (RESET)</div>', unsafe_allow_html=True)
        except Exception as e:
            st.warning(f"RESET test: {e}")

        st.markdown("### Ringkasan Diagnostik")
        diag_summary = []
        diag_summary.append({'Test': 'Durbin-Watson', 'Statistic': f"{dw:.4f}", 'Conclusion': 'Pass' if 1.5<dw<2.5 else 'Fail'})
        try: diag_summary.append({'Test': 'Breusch-Godfrey', 'Statistic': f"{bg_lm:.4f}", 'P-value': f"{bg_p:.4f}", 'Conclusion': 'Pass' if bg_p>0.05 else 'Fail'})
        except: pass
        try: diag_summary.append({'Test': 'Breusch-Pagan', 'Statistic': f"{bp_lm:.4f}", 'P-value': f"{bp_p:.4f}", 'Conclusion': 'Pass' if bp_p>0.05 else 'Fail'})
        except: pass
        try: diag_summary.append({'Test': 'White', 'Statistic': f"{wh_lm:.4f}", 'P-value': f"{wh_p:.4f}", 'Conclusion': 'Pass' if wh_p>0.05 else 'Fail'})
        except: pass
        diag_summary.append({'Test': 'Jarque-Bera', 'Statistic': f"{jb:.4f}", 'P-value': f"{jb_p:.4f}", 'Conclusion': 'Pass' if jb_p>0.05 else 'Fail'})
        try: diag_summary.append({'Test': 'Ramsey RESET', 'Statistic': f"{reset.fvalue:.4f}", 'P-value': f"{reset.pvalue:.4f}", 'Conclusion': 'Pass' if reset.pvalue>0.05 else 'Fail'})
        except: pass
        st.dataframe(pd.DataFrame(diag_summary), use_container_width=True)

        st.markdown("### Residual Plots")
        c1, c2 = st.columns(2)
        with c1:
            fig_rh = px.histogram(x=resid, nbins=30, title="Residual Histogram", marginal="box", color_discrete_sequence=["#8E44AD"])
            st.plotly_chart(fig_rh, use_container_width=True)
        with c2:
            fig_qq = go.Figure()
            sorted_r = np.sort(resid)
            theoretical = stats.norm.ppf(np.linspace(0.01, 0.99, len(sorted_r)))
            fig_qq.add_trace(go.Scatter(x=theoretical, y=sorted_r, mode='markers', marker=dict(size=4, color='#2E86C1'), name='Residuals'))
            fig_qq.add_trace(go.Scatter(x=[theoretical.min(), theoretical.max()], y=[theoretical.min()*resid.std()+resid.mean(), theoretical.max()*resid.std()+resid.mean()], mode='lines', line=dict(color='red', dash='dash'), name='Normal'))
            fig_qq.update_layout(title="Q-Q Plot", xaxis_title="Theoretical", yaxis_title="Sample", height=400)
            st.plotly_chart(fig_qq, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            fig_rf = px.scatter(x=fitted, y=resid, title="Residuals vs Fitted", labels={'x':'Fitted','y':'Residual'}, color_discrete_sequence=["#27AE60"])
            fig_rf.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig_rf, use_container_width=True)
        with c2:
            nlags_r = min(30, len(resid)//3)
            acf_r = acf(resid, nlags=nlags_r, fft=True)
            fig_acfr = go.Figure()
            fig_acfr.add_trace(go.Bar(x=list(range(len(acf_r))), y=acf_r, marker_color='#E74C3C', width=0.3))
            fig_acfr.add_hline(y=1.96/np.sqrt(len(resid)), line_dash="dash", line_color="blue")
            fig_acfr.add_hline(y=-1.96/np.sqrt(len(resid)), line_dash="dash", line_color="blue")
            fig_acfr.update_layout(title="ACF of Residuals", xaxis_title="Lag", yaxis_title="ACF", height=400)
            st.plotly_chart(fig_acfr, use_container_width=True)

        st.markdown("### CUSUM & CUSUMSQ")
        try:
            rec_resid = resid.values
            n = len(rec_resid)
            sigma = np.std(rec_resid, ddof=1)
            cusum = np.cumsum(rec_resid) / sigma
            cusum_sq = np.cumsum(rec_resid**2) / np.sum(rec_resid**2)
            t_vals = np.arange(1, n+1)
            upper_cusum = 0.948 * np.sqrt(n) + 2 * 0.948 * t_vals / np.sqrt(n)
            lower_cusum = -upper_cusum
            c1, c2 = st.columns(2)
            with c1:
                fig_cu = go.Figure()
                fig_cu.add_trace(go.Scatter(y=cusum, mode='lines', name='CUSUM', line=dict(color='#2E86C1')))
                fig_cu.add_trace(go.Scatter(y=upper_cusum, mode='lines', name='Upper 5%', line=dict(color='red', dash='dash')))
                fig_cu.add_trace(go.Scatter(y=lower_cusum, mode='lines', name='Lower 5%', line=dict(color='red', dash='dash')))
                fig_cu.update_layout(title="CUSUM Test", height=400)
                st.plotly_chart(fig_cu, use_container_width=True)
            with c2:
                expected_sq = t_vals / n
                upper_sq = expected_sq + 1.63 / np.sqrt(n)
                lower_sq = expected_sq - 1.63 / np.sqrt(n)
                fig_cs = go.Figure()
                fig_cs.add_trace(go.Scatter(y=cusum_sq, mode='lines', name='CUSUMSQ', line=dict(color='#8E44AD')))
                fig_cs.add_trace(go.Scatter(y=upper_sq, mode='lines', name='Upper', line=dict(color='red', dash='dash')))
                fig_cs.add_trace(go.Scatter(y=lower_sq, mode='lines', name='Lower', line=dict(color='red', dash='dash')))
                fig_cs.update_layout(title="CUSUM of Squares", height=400)
                st.plotly_chart(fig_cs, use_container_width=True)
            within_cu = np.all((cusum >= lower_cusum) & (cusum <= upper_cusum))
            within_sq = np.all((cusum_sq >= np.maximum(lower_sq,0)) & (cusum_sq <= np.minimum(upper_sq,1)))
            if within_cu and within_sq:
                st.markdown('<div class="success-box"><b>CUSUM & CUSUMSQ within bounds.</b> Model stabil.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="warning-box"><b>Instabilitas terdeteksi.</b> Periksa structural break.</div>', unsafe_allow_html=True)
        except Exception as e:
            st.warning(f"CUSUM error: {e}")
    else:
        st.warning("Estimasi model ARDL terlebih dahulu!")

# ============================================================
# TAB 8: LAPORAN & EXPORT
# ============================================================
with tab8:
    st.markdown("## \U0001f4cb Laporan & Export")
    dep = st.session_state.dep_var; indeps = st.session_state.indep_vars
    if dep and indeps:
        rl = ["="*70, "ARDL ANALYSIS REPORT", "="*70]
        rl.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        rl.append(f"Dependent: {dep}")
        rl.append(f"Independent: {', '.join(indeps)}")
        rl.append(f"Observations: {len(df)}")
        rl.append("")

        # Unit root summary
        if st.session_state.unit_root_results:
            rl.append("="*40 + " UNIT ROOT TESTS " + "="*40)
            allvars = [dep] + indeps
            for v in allvars:
                adf_l = st.session_state.unit_root_results.get(f'{v}_level_ADF', {})
                adf_d = st.session_state.unit_root_results.get(f'{v}_diff_ADF', {})
                stat_l = "I(0)" if adf_l.get('Stationary') else "Non-stat"
                stat_d = "Stat" if adf_d.get('Stationary') else "Non-stat"
                rl.append(f"  {v:25s} Level: ADF={adf_l.get('Statistic',0):.4f} p={adf_l.get('P-value',1):.4f} [{stat_l}]  Diff: ADF={adf_d.get('Statistic',0):.4f} p={adf_d.get('P-value',1):.4f} [{stat_d}]")
            rl.append("")

        # ARDL results
        if st.session_state.ardl_results is not None:
            r = st.session_state.ardl_results
            rl.append("="*40 + " ARDL MODEL " + "="*40)
            rl.append(f"  Model: {st.session_state.get('ardl_order_str', 'ARDL')}")
            rl.append(f"  R2={r.rsquared:.6f}, AdjR2={r.rsquared_adj:.6f}")
            rl.append(f"  AIC={r.aic:.4f}, BIC={r.bic:.4f}, LogLik={r.llf:.4f}")
            rl.append(f"  F={r.fvalue:.4f} (p={r.f_pvalue:.6f}), DW={durbin_watson(r.resid):.4f}")
            rl.append("  Coefficients:")
            for p, v in zip(r.params.index, r.params.values):
                pv = r.pvalues[p]
                sig = "***" if pv<0.01 else "**" if pv<0.05 else "*" if pv<0.1 else ""
                rl.append(f"    {str(p):30s} {v:>12.6f} (p={pv:.4f}) {sig}")
            rl.append("")

        # Bounds test
        if st.session_state.bounds_results is not None:
            br = st.session_state.bounds_results
            rl.append("="*40 + " BOUNDS TEST " + "="*40)
            rl.append(f"  F-stat = {br['f_stat']:.4f}, k={br['k']}, n={br['n']}")
            rl.append("  Pesaran et al. (2001):")
            for sig in ['10%','5%','1%']:
                lo, hi = br['cv_pesaran'][sig]
                rl.append(f"    {sig}: I(0)={lo:.2f}, I(1)={hi:.2f}")
            interp, _ = interpret_bounds(br['f_stat'], br['cv_pesaran'])
            rl.append(f"  Conclusion: {interp}")
            rl.append("")

        # Long-run
        if hasattr(st.session_state, 'lr_coefs') and st.session_state.lr_coefs:
            rl.append("="*40 + " LONG-RUN COEFFICIENTS " + "="*40)
            for k, v in st.session_state.lr_coefs.items():
                rl.append(f"  {k:25s} {v:>12.6f}")
            rl.append("")

        # ECM
        if st.session_state.ecm_results is not None:
            e = st.session_state.ecm_results
            rl.append("="*40 + " ECM (SHORT-RUN) " + "="*40)
            rl.append(f"  R2={e.rsquared:.6f}, AdjR2={e.rsquared_adj:.6f}")
            for p, v in zip(e.params.index, e.params.values):
                pv = e.pvalues[p]
                sig = "***" if pv<0.01 else "**" if pv<0.05 else "*" if pv<0.1 else ""
                rl.append(f"    {str(p):25s} {v:>12.6f} (p={pv:.4f}) {sig}")
            rl.append("")

        rl += ["="*70, "NOTES:", "- *** p<0.01, ** p<0.05, * p<0.10",
               "- Bounds test: Pesaran, Shin & Smith (2001)",
               "- ECT must be negative and significant for valid cointegration",
               "- CUSUM/CUSUMSQ tests model stability"]

        rt = "\n".join(rl)
        st.text_area("Preview", rt, height=500)
        fname_rpt = "ARDL_Report_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".txt"
        st.download_button("Download Report (.txt)", rt, fname_rpt, "text/plain", use_container_width=True)

        st.markdown("### Export Config")
        cfg = {
            "analysis": "ARDL", "timestamp": datetime.now().isoformat(),
            "dep": dep, "indep": indeps, "n": len(df),
            "settings": {
                "max_lag_dep": st.session_state.max_lag_dep,
                "max_lag_indep": st.session_state.max_lag_indep,
                "ic": st.session_state.ic_criterion,
                "trend": st.session_state.trend
            }
        }
        if st.session_state.ardl_results:
            r = st.session_state.ardl_results
            cfg["ardl"] = {"order": st.session_state.get('ardl_order_str',''), "r2": float(r.rsquared), "aic": float(r.aic), "bic": float(r.bic)}
        if st.session_state.bounds_results:
            cfg["bounds"] = {"f_stat": float(st.session_state.bounds_results['f_stat']), "k": st.session_state.bounds_results['k']}
        if hasattr(st.session_state, 'lr_coefs') and st.session_state.lr_coefs:
            cfg["long_run"] = {k: float(v) for k, v in st.session_state.lr_coefs.items()}
        cj = json.dumps(cfg, indent=2, default=str)
        fname_cfg = "ARDL_Config_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".json"
        st.download_button("Download Config (.json)", cj, fname_cfg, "application/json", use_container_width=True)

        st.markdown("### Export Data + Results")
        if st.session_state.ardl_results is not None:
            edf = df.copy()
            r = st.session_state.ardl_results
            edf['ARDL_Fitted'] = np.nan
            edf.loc[r.fittedvalues.index, 'ARDL_Fitted'] = r.fittedvalues.values
            edf['ARDL_Resid'] = np.nan
            edf.loc[r.resid.index, 'ARDL_Resid'] = r.resid.values
            buf = io.StringIO(); edf.to_csv(buf)
            fname_data = "ARDL_Data_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".csv"
            st.download_button("Download Data+Results (.csv)", buf.getvalue(), fname_data, "text/csv", use_container_width=True)
    else:
        st.warning("Pilih variabel terlebih dahulu!")

# FOOTER
st.markdown("---")
st.markdown('<div style="text-align:center;color:#888;font-size:0.85rem">ARDL Analysis Suite | Pesaran, Shin & Smith (2001) | statsmodels | Streamlit</div>', unsafe_allow_html=True)
