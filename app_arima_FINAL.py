import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import boxcox, inv_boxcox
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings, io, itertools
warnings.filterwarnings('ignore')

try:
    from arch import arch_model
    HAS_ARCH = True
except ImportError:
    HAS_ARCH = False

st.set_page_config(page_title="ARIMA Pro — Time Series Analysis", layout="wide", page_icon="📈")
st.title("📈 ARIMA Pro — Analisis Time Series Lengkap")
st.caption("EDA · Decomposition · Stationarity · ARIMA/SARIMA · ARIMAX · GARCH · Forecast · Evaluation")

# ============================================================
# DEMO DATA
# ============================================================
@st.cache_data
def load_demo_airline():
    np.random.seed(42)
    n = 144
    t = np.arange(n)
    trend = 100 + 2.5 * t
    seasonal = 30 * np.sin(2 * np.pi * t / 12)
    noise = np.random.normal(0, 10, n)
    y = trend + seasonal + noise
    dates = pd.date_range('1949-01-01', periods=n, freq='MS')
    return pd.DataFrame({'Date': dates, 'Passengers': np.round(y, 1)})

@st.cache_data
def load_demo_stock():
    np.random.seed(123)
    n = 500
    returns = np.random.normal(0.0005, 0.02, n)
    # Add GARCH-like volatility clustering
    vol = np.zeros(n)
    vol[0] = 0.02
    for i in range(1, n):
        vol[i] = np.sqrt(0.00001 + 0.1 * returns[i-1]**2 + 0.85 * vol[i-1]**2)
        returns[i] = np.random.normal(0.0005, vol[i])
    price = 100 * np.exp(np.cumsum(returns))
    dates = pd.date_range('2023-01-02', periods=n, freq='B')
    return pd.DataFrame({'Date': dates, 'Price': np.round(price, 2), 'Volume': np.random.randint(1000000, 5000000, n)})

@st.cache_data
def load_demo_sales():
    np.random.seed(99)
    n = 120
    t = np.arange(n)
    trend = 500 + 3 * t
    seasonal = 80 * np.sin(2 * np.pi * t / 12) + 40 * np.cos(2 * np.pi * t / 6)
    promo = np.zeros(n)
    promo_idx = np.random.choice(n, 20, replace=False)
    promo[promo_idx] = np.random.uniform(50, 150, 20)
    noise = np.random.normal(0, 20, n)
    y = trend + seasonal + promo + noise
    dates = pd.date_range('2014-01-01', periods=n, freq='MS')
    return pd.DataFrame({'Date': dates, 'Sales': np.round(y, 1), 'Promo': np.round(promo, 1),
                         'Temperature': np.round(25 + 10*np.sin(2*np.pi*t/12) + np.random.normal(0,2,n), 1)})

@st.cache_data
def load_demo_multiplicative():
    np.random.seed(55)
    n = 144
    t = np.arange(n)
    trend = 100 + 1.5 * t
    seasonal_mult = 1 + 0.3 * np.sin(2 * np.pi * t / 12)
    noise = np.random.normal(1, 0.05, n)
    y = trend * seasonal_mult * noise
    dates = pd.date_range('1949-01-01', periods=n, freq='MS')
    return pd.DataFrame({'Date': dates, 'Value': np.round(y, 2)})

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def adf_test(series):
    result = adfuller(series.dropna(), autolag='AIC')
    return {'Test Statistic': result[0], 'p-value': result[1], 'Lags Used': result[2],
            'Observations': result[3], 'Critical 1%': result[4]['1%'],
            'Critical 5%': result[4]['5%'], 'Critical 10%': result[4]['10%'],
            'Stationary': 'Yes' if result[1] < 0.05 else 'No'}

def kpss_test(series, regression='c'):
    try:
        result = kpss(series.dropna(), regression=regression, nlags='auto')
        return {'Test Statistic': result[0], 'p-value': result[1], 'Lags Used': result[2],
                'Critical 1%': result[3]['1%'], 'Critical 5%': result[3]['5%'],
                'Critical 10%': result[3]['10%'],
                'Stationary': 'Yes' if result[1] > 0.05 else 'No'}
    except Exception as e:
        return {'Test Statistic': None, 'p-value': None, 'Error': str(e), 'Stationary': 'Unknown'}

def pp_test(series):
    """Phillips-Perron approximation via ADF with maxlag"""
    try:
        result = adfuller(series.dropna(), autolag='AIC', maxlag=int(len(series)**0.25))
        return {'Test Statistic': result[0], 'p-value': result[1], 'Stationary': 'Yes' if result[1] < 0.05 else 'No'}
    except:
        return {'Test Statistic': None, 'p-value': None, 'Stationary': 'Unknown'}

def calc_metrics(actual, predicted):
    actual = np.array(actual); predicted = np.array(predicted)
    mask = ~(np.isnan(actual) | np.isnan(predicted))
    actual = actual[mask]; predicted = predicted[mask]
    if len(actual) == 0: return {}
    e = actual - predicted
    ae = np.abs(e)
    se = e**2
    mape_vals = np.abs(e / actual) * 100
    mape_vals = mape_vals[np.isfinite(mape_vals)]
    return {
        'ME': np.mean(e), 'MAE': np.mean(ae), 'MSE': np.mean(se),
        'RMSE': np.sqrt(np.mean(se)), 'MAPE': np.mean(mape_vals) if len(mape_vals) > 0 else None,
        'MdAPE': np.median(mape_vals) if len(mape_vals) > 0 else None,
        'MaxError': np.max(ae), 'R2': 1 - np.sum(se) / np.sum((actual - actual.mean())**2) if np.var(actual) > 0 else None
    }

def plot_acf_pacf(series, lags=40, title=""):
    acf_vals = acf(series.dropna(), nlags=lags, fft=True)
    pacf_vals = pacf(series.dropna(), nlags=min(lags, len(series)//2 - 1))
    n = len(series.dropna())
    ci = 1.96 / np.sqrt(n)
    fig = make_subplots(rows=1, cols=2, subplot_titles=(f"ACF {title}", f"PACF {title}"))
    for i, vals in enumerate([acf_vals, pacf_vals]):
        col = i + 1
        for j in range(len(vals)):
            color = 'red' if abs(vals[j]) > ci and j > 0 else 'steelblue'
            fig.add_trace(go.Bar(x=[j], y=[vals[j]], marker_color=color, showlegend=False, width=0.3), row=1, col=col)
        fig.add_hline(y=ci, line=dict(color='blue', dash='dash', width=1), row=1, col=col)
        fig.add_hline(y=-ci, line=dict(color='blue', dash='dash', width=1), row=1, col=col)
        fig.add_hline(y=0, line=dict(color='black', width=0.5), row=1, col=col)
    fig.update_layout(height=350, template='plotly_white', showlegend=False)
    return fig

def apply_transformation(series, method):
    if method == 'None': return series, None
    elif method == 'Log': return np.log(series), None
    elif method == 'Sqrt': return np.sqrt(series), None
    elif method == 'Box-Cox':
        from scipy.stats import boxcox as bc
        transformed, lam = bc(series)
        return pd.Series(transformed, index=series.index), lam
    elif method == 'Diff(1)': return series.diff().dropna(), None
    elif method == 'Diff(2)': return series.diff().diff().dropna(), None
    elif method == 'Seasonal Diff': return series.diff(12).dropna(), None
    elif method == 'Log + Diff(1)': return np.log(series).diff().dropna(), None
    return series, None

def inverse_transform(values, method, original_series, lam=None):
    if method == 'None': return values
    elif method == 'Log': return np.exp(values)
    elif method == 'Sqrt': return values**2
    elif method == 'Box-Cox' and lam is not None:
        from scipy.special import inv_boxcox as ibc
        return ibc(values, lam)
    return values

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.header("Menu")
module = st.sidebar.selectbox("Modul:", [
    'eda', 'decomposition', 'stationarity', 'acf_pacf',
    'arima', 'sarima', 'arimax', 'garch',
    'auto_arima', 'comparison', 'forecast'
], format_func=lambda x: {
    'eda': '1. EDA & Visualization',
    'decomposition': '2. Decomposition',
    'stationarity': '3. Stationarity Tests',
    'acf_pacf': '4. ACF & PACF',
    'arima': '5. ARIMA Modeling',
    'sarima': '6. SARIMA Modeling',
    'arimax': '7. ARIMAX Modeling',
    'garch': '8. ARIMA-GARCH',
    'auto_arima': '9. Auto ARIMA (Grid Search)',
    'comparison': '10. Model Comparison',
    'forecast': '11. Forecasting & CI/PI'
}[x])

# ============================================================
# DATA LOADING (shared)
# ============================================================
st.sidebar.markdown("---")
st.sidebar.subheader("Data")
data_src = st.sidebar.selectbox("Sumber Data:", ['Demo: Airline Passengers', 'Demo: Stock Price (GARCH)',
                                                  'Demo: Sales + Exog (ARIMAX)', 'Demo: Multiplicative', 'Upload CSV'])

if data_src == 'Demo: Airline Passengers':
    raw_df = load_demo_airline(); date_col = 'Date'; target_col = 'Passengers'
elif data_src == 'Demo: Stock Price (GARCH)':
    raw_df = load_demo_stock(); date_col = 'Date'; target_col = 'Price'
elif data_src == 'Demo: Sales + Exog (ARIMAX)':
    raw_df = load_demo_sales(); date_col = 'Date'; target_col = 'Sales'
elif data_src == 'Demo: Multiplicative':
    raw_df = load_demo_multiplicative(); date_col = 'Date'; target_col = 'Value'
else:
    up = st.sidebar.file_uploader("Upload CSV:", type=['csv'])
    if up:
        raw_df = pd.read_csv(up)
        date_col = st.sidebar.selectbox("Date column:", raw_df.columns)
        target_col = st.sidebar.selectbox("Target column:", [c for c in raw_df.columns if c != date_col])
    else:
        st.info("Upload CSV dari sidebar."); st.stop()

raw_df[date_col] = pd.to_datetime(raw_df[date_col])
raw_df = raw_df.sort_values(date_col).reset_index(drop=True)
raw_df = raw_df.set_index(date_col)
y_full = raw_df[target_col].dropna()

st.sidebar.markdown(f"**n = {len(y_full)}** | {y_full.index[0].strftime('%Y-%m')} → {y_full.index[-1].strftime('%Y-%m')}")

# ============================================================
# 1. EDA
# ============================================================
if module == 'eda':
    st.header("1. Exploratory Data Analysis")

    # Time series plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_full.index, y=y_full.values, mode='lines', line=dict(color='steelblue', width=1.5), name=target_col))
    fig.update_layout(title=f"Time Series: {target_col}", height=400, template='plotly_white', xaxis_title='Date', yaxis_title=target_col)
    st.plotly_chart(fig, use_container_width=True)

    # Descriptive stats
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("n", str(len(y_full))); c2.metric("Mean", f"{y_full.mean():.2f}")
    c3.metric("Std", f"{y_full.std():.2f}"); c4.metric("CV%", f"{y_full.std()/y_full.mean()*100:.2f}%")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Min", f"{y_full.min():.2f}"); c2.metric("Max", f"{y_full.max():.2f}")
    c3.metric("Skewness", f"{y_full.skew():.4f}"); c4.metric("Kurtosis", f"{y_full.kurtosis():.4f}")

    # Distribution
    fig2 = make_subplots(rows=1, cols=3, subplot_titles=("Histogram", "Box Plot", "Rolling Stats"))
    fig2.add_trace(go.Histogram(x=y_full, nbinsx=30, marker_color='steelblue', opacity=0.7), row=1, col=1)
    fig2.add_trace(go.Box(y=y_full, marker_color='steelblue', name=target_col), row=1, col=2)
    window = max(6, len(y_full) // 20)
    rm = y_full.rolling(window).mean(); rs = y_full.rolling(window).std()
    fig2.add_trace(go.Scatter(x=rm.index, y=rm.values, mode='lines', name='Rolling Mean', line=dict(color='crimson')), row=1, col=3)
    fig2.add_trace(go.Scatter(x=rs.index, y=rs.values, mode='lines', name='Rolling Std', line=dict(color='green')), row=1, col=3)
    fig2.update_layout(height=350, template='plotly_white'); st.plotly_chart(fig2, use_container_width=True)

    # Monthly / seasonal patterns
    if hasattr(y_full.index, 'month'):
        monthly = y_full.groupby(y_full.index.month).mean()
        fig3 = go.Figure(data=[go.Bar(x=monthly.index, y=monthly.values, marker_color='steelblue')])
        fig3.update_layout(title="Average by Month", height=320, template='plotly_white', xaxis_title='Month')
        st.plotly_chart(fig3, use_container_width=True)

    # Data table
    with st.expander("Raw Data"):
        st.dataframe(raw_df.round(4), use_container_width=True)

# ============================================================
# 2. DECOMPOSITION
# ============================================================
elif module == 'decomposition':
    st.header("2. Time Series Decomposition")
    c1, c2, c3 = st.columns(3)
    method = c1.selectbox("Metode:", ['Classical', 'STL'])
    model_type = c2.selectbox("Model:", ['additive', 'multiplicative'])
    period = c3.number_input("Period:", 2, 365, 12)

    if method == 'Classical':
        try:
            decomp = seasonal_decompose(y_full, model=model_type, period=period)
        except Exception as e:
            st.error(f"Error: {e}"); st.stop()
    else:
        try:
            decomp = STL(y_full, period=period).fit()
        except Exception as e:
            st.error(f"Error: {e}"); st.stop()

    fig = make_subplots(rows=4, cols=1, subplot_titles=("Observed", "Trend", "Seasonal", "Residual"),
                        shared_xaxes=True, vertical_spacing=0.05)
    fig.add_trace(go.Scatter(x=y_full.index, y=y_full.values, mode='lines', line=dict(width=1), name='Observed'), row=1, col=1)
    fig.add_trace(go.Scatter(x=y_full.index, y=decomp.trend, mode='lines', line=dict(color='crimson', width=1.5), name='Trend'), row=2, col=1)
    fig.add_trace(go.Scatter(x=y_full.index, y=decomp.seasonal, mode='lines', line=dict(color='green', width=1), name='Seasonal'), row=3, col=1)
    fig.add_trace(go.Scatter(x=y_full.index, y=decomp.resid, mode='markers', marker=dict(size=3, color='gray'), name='Residual'), row=4, col=1)
    fig.update_layout(height=700, template='plotly_white', showlegend=False); st.plotly_chart(fig, use_container_width=True)

    resid = pd.Series(decomp.resid).dropna()
    c1,c2,c3 = st.columns(3)
    c1.metric("Trend range", f"{np.nanmin(decomp.trend):.2f} - {np.nanmax(decomp.trend):.2f}")
    c2.metric("Seasonal amp", f"{np.nanmax(decomp.seasonal) - np.nanmin(decomp.seasonal):.2f}")
    c3.metric("Resid std", f"{resid.std():.4f}")

    # Strength of trend & seasonality
    var_resid = resid.var()
    trend_clean = pd.Series(decomp.trend).dropna()
    var_trend_resid = (trend_clean - trend_clean.mean()).var() + var_resid if len(trend_clean) > 0 else 1
    Ft = max(0, 1 - var_resid / var_trend_resid)
    seasonal_clean = pd.Series(decomp.seasonal).dropna()
    var_seas_resid = seasonal_clean.var() + var_resid
    Fs = max(0, 1 - var_resid / var_seas_resid) if var_seas_resid > 0 else 0
    st.markdown(f"**Strength of Trend:** {Ft:.4f} | **Strength of Seasonality:** {Fs:.4f}")

# ============================================================
# 3. STATIONARITY TESTS
# ============================================================
elif module == 'stationarity':
    st.header("3. Stationarity Testing")
    st.markdown("Uji stasioneritas: **ADF** (H0: non-stationary), **KPSS** (H0: stationary), **PP**.")

    # Transformation
    transform = st.selectbox("Transformasi:", ['None', 'Log', 'Sqrt', 'Box-Cox', 'Diff(1)', 'Diff(2)', 'Seasonal Diff', 'Log + Diff(1)'])
    if transform != 'None' and (transform in ['Log', 'Box-Cox', 'Sqrt']) and (y_full <= 0).any():
        st.warning("Data mengandung nilai <= 0. Transformasi Log/Sqrt/Box-Cox tidak bisa digunakan.")
        y_test = y_full
        bc_lambda = None
    else:
        y_test, bc_lambda = apply_transformation(y_full, transform)

    c1, c2 = st.columns(2)
    # Before vs After
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Original", f"After: {transform}"))
    fig.add_trace(go.Scatter(x=y_full.index, y=y_full.values, mode='lines', line=dict(width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=y_test.index, y=y_test.values, mode='lines', line=dict(width=1, color='crimson')), row=1, col=2)
    fig.update_layout(height=320, template='plotly_white', showlegend=False); st.plotly_chart(fig, use_container_width=True)

    # ADF
    st.markdown("### ADF Test (H₀: Unit Root / Non-Stationary)")
    adf_res = adf_test(y_test)
    st.dataframe(pd.DataFrame([adf_res]), use_container_width=True, hide_index=True)
    if adf_res['Stationary'] == 'Yes': st.success("ADF: Stationary (reject H0)")
    else: st.warning("ADF: Non-Stationary (fail to reject H0)")

    # KPSS
    st.markdown("### KPSS Test (H₀: Stationary)")
    kpss_res = kpss_test(y_test)
    st.dataframe(pd.DataFrame([kpss_res]), use_container_width=True, hide_index=True)
    if kpss_res.get('Stationary') == 'Yes': st.success("KPSS: Stationary (fail to reject H0)")
    else: st.warning("KPSS: Non-Stationary (reject H0)")

    # PP
    st.markdown("### Phillips-Perron Test")
    pp_res = pp_test(y_test)
    st.dataframe(pd.DataFrame([pp_res]), use_container_width=True, hide_index=True)

    # Summary
    st.markdown("### Ringkasan")
    adf_ok = adf_res['Stationary'] == 'Yes'
    kpss_ok = kpss_res.get('Stationary') == 'Yes'
    if adf_ok and kpss_ok: st.success("Kedua test setuju: **STATIONARY**. Siap untuk modeling.")
    elif adf_ok and not kpss_ok: st.info("ADF: stationary, KPSS: non-stationary → kemungkinan **trend-stationary**.")
    elif not adf_ok and kpss_ok: st.info("ADF: non-stationary, KPSS: stationary → kemungkinan **difference-stationary**.")
    else: st.error("Kedua test: **NON-STATIONARY**. Perlu differencing/transformasi.")

    # Differencing suggestion
    if not adf_ok:
        st.markdown("### Auto-Differencing")
        d = 0; temp = y_full.copy()
        for dd in range(1, 4):
            temp = temp.diff().dropna()
            res = adfuller(temp, autolag='AIC')
            if res[1] < 0.05: d = dd; break
        if d > 0: st.info(f"Suggested d = **{d}** (stationary after {d}x differencing)")
        else: st.warning("Masih non-stationary setelah 3x differencing. Coba transformasi lain.")

    if bc_lambda is not None: st.info(f"Box-Cox lambda = **{bc_lambda:.4f}**")

# ============================================================
# 4. ACF & PACF
# ============================================================
elif module == 'acf_pacf':
    st.header("4. ACF & PACF Analysis")
    st.markdown("Identifikasi orde **p** (dari PACF) dan **q** (dari ACF) untuk ARIMA.")

    transform = st.selectbox("Transformasi:", ['None', 'Log', 'Diff(1)', 'Diff(2)', 'Seasonal Diff', 'Log + Diff(1)'])
    if transform != 'None' and transform in ['Log'] and (y_full <= 0).any():
        y_plot = y_full
    else:
        y_plot, _ = apply_transformation(y_full, transform)

    max_lags = st.slider("Max lags:", 10, min(100, len(y_plot)//2 - 1), min(40, len(y_plot)//3))

    # ACF & PACF
    fig = plot_acf_pacf(y_plot, lags=max_lags, title=f"({transform})")
    st.plotly_chart(fig, use_container_width=True)

    # Time series
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=y_plot.index, y=y_plot.values, mode='lines', line=dict(width=1)))
    fig2.update_layout(title=f"Series ({transform})", height=300, template='plotly_white')
    st.plotly_chart(fig2, use_container_width=True)

    # Interpretation guide
    with st.expander("Panduan Identifikasi Model"):
        st.markdown("""
| ACF Pattern | PACF Pattern | Model Suggested |
|---|---|---|
| Tails off (decays) | Cuts off after lag p | **AR(p)** |
| Cuts off after lag q | Tails off (decays) | **MA(q)** |
| Tails off | Tails off | **ARMA(p,q)** |
| Significant at lag s, 2s, 3s... | - | **Seasonal component (P,Q)** |
| All within bounds | All within bounds | **White noise / No model needed** |
""")

    # Seasonal ACF hint
    st.markdown("### Seasonal Lags")
    acf_vals = acf(y_plot.dropna(), nlags=max_lags, fft=True)
    seasonal_lags = [i for i in range(6, len(acf_vals)) if abs(acf_vals[i]) > 1.96/np.sqrt(len(y_plot)) and i % 6 == 0]
    if seasonal_lags:
        st.info(f"Significant seasonal lags at: {seasonal_lags} → consider seasonal period = {seasonal_lags[0]}")
    else:
        st.info("No clear seasonal pattern detected in ACF.")

# ============================================================
# 5. ARIMA MODELING
# ============================================================
elif module == 'arima':
    st.header("5. ARIMA(p,d,q) Modeling")
    st.markdown("Autoregressive Integrated Moving Average.")

    # Transform
    transform = st.selectbox("Transformasi sebelum modeling:", ['None', 'Log', 'Sqrt', 'Box-Cox'])
    if transform != 'None' and (y_full <= 0).any():
        st.warning("Data <= 0, transform tidak diterapkan."); y_model = y_full; bc_lam = None
    else:
        y_model, bc_lam = apply_transformation(y_full, transform)

    # Train-Test Split
    st.markdown("### Train-Test Split")
    split_pct = st.slider("Train %:", 50, 95, 80)
    n_train = int(len(y_model) * split_pct / 100)
    y_train = y_model.iloc[:n_train]; y_test = y_model.iloc[n_train:]
    st.info(f"Train: {len(y_train)} obs ({y_train.index[0].strftime('%Y-%m')} - {y_train.index[-1].strftime('%Y-%m')}) | Test: {len(y_test)} obs")

    # Parameters
    st.markdown("### Parameters")
    c1,c2,c3 = st.columns(3)
    p = c1.number_input("p (AR):", 0, 10, 1); d = c2.number_input("d (Diff):", 0, 3, 1); q = c3.number_input("q (MA):", 0, 10, 1)
    trend_opt = st.selectbox("Trend:", ['n (none)', 'c (constant)', 't (linear)', 'ct (constant+trend)'])
    trend_map = {'n (none)':'n', 'c (constant)':'c', 't (linear)':'t', 'ct (constant+trend)':'ct'}

    if st.button("Fit ARIMA", type="primary"):
        with st.spinner("Fitting ARIMA..."):
            try:
                model = ARIMA(y_train, order=(p,d,q), trend=trend_map[trend_opt])
                result = model.fit()

                st.markdown("### Model Summary")
                summary_str = str(result.summary())
                st.text(summary_str)

                # Metrics
                st.markdown("### Information Criteria")
                c1,c2,c3,c4 = st.columns(4)
                c1.metric("AIC", f"{result.aic:.2f}"); c2.metric("BIC", f"{result.bic:.2f}")
                c3.metric("HQIC", f"{result.hqic:.2f}"); c4.metric("Log-Lik", f"{result.llf:.2f}")

                # Fitted vs Actual (train)
                fitted = result.fittedvalues
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=y_train.index, y=y_train.values, mode='lines', name='Actual', line=dict(color='steelblue')))
                fig.add_trace(go.Scatter(x=fitted.index, y=fitted.values, mode='lines', name='Fitted', line=dict(color='crimson', dash='dash')))
                fig.update_layout(title="Fitted vs Actual (Train)", height=380, template='plotly_white')
                st.plotly_chart(fig, use_container_width=True)

                # Forecast on test set
                forecast_obj = result.get_forecast(steps=len(y_test))
                forecast_mean = forecast_obj.predicted_mean
                forecast_ci = forecast_obj.conf_int(alpha=0.05)
                forecast_pi = forecast_obj.conf_int(alpha=0.05)  # prediction interval approx

                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=y_train.index, y=y_train.values, mode='lines', name='Train', line=dict(color='steelblue')))
                fig2.add_trace(go.Scatter(x=y_test.index, y=y_test.values, mode='lines', name='Test (Actual)', line=dict(color='green', width=2)))
                fig2.add_trace(go.Scatter(x=forecast_mean.index, y=forecast_mean.values, mode='lines', name='Forecast', line=dict(color='crimson', width=2)))
                # CI
                ci_lower = forecast_ci.iloc[:, 0]; ci_upper = forecast_ci.iloc[:, 1]
                fig2.add_trace(go.Scatter(x=ci_lower.index.tolist()+ci_upper.index.tolist()[::-1],
                    y=ci_lower.values.tolist()+ci_upper.values.tolist()[::-1],
                    fill='toself', fillcolor='rgba(255,0,0,0.1)', line=dict(color='rgba(255,0,0,0)'), name='95% CI'))
                fig2.update_layout(title="Forecast vs Actual (Test)", height=420, template='plotly_white')
                st.plotly_chart(fig2, use_container_width=True)

                # Evaluation
                st.markdown("### Model Evaluation")
                train_metrics = calc_metrics(y_train.values[d:], fitted.values[d:] if len(fitted) > d else fitted.values)
                test_metrics = calc_metrics(y_test.values, forecast_mean.values)
                mdf = pd.DataFrame([
                    {**{'Set': 'Train'}, **{k: f"{v:.4f}" if v is not None else '-' for k,v in train_metrics.items()}},
                    {**{'Set': 'Test'}, **{k: f"{v:.4f}" if v is not None else '-' for k,v in test_metrics.items()}}
                ])
                st.dataframe(mdf, use_container_width=True, hide_index=True)

                # Residual Diagnostics
                st.markdown("### Residual Diagnostics")
                resid = result.resid.dropna()

                fig3 = make_subplots(rows=2, cols=2, subplot_titles=("Residuals", "Histogram", "ACF Residuals", "Q-Q Plot"))
                fig3.add_trace(go.Scatter(x=resid.index, y=resid.values, mode='lines', line=dict(width=0.8, color='gray')), row=1, col=1)
                fig3.add_hline(y=0, line=dict(color='red', dash='dash'), row=1, col=1)
                fig3.add_trace(go.Histogram(x=resid, nbinsx=30, marker_color='steelblue', opacity=0.7), row=1, col=2)
                # ACF of residuals
                acf_r = acf(resid, nlags=min(30, len(resid)//2-1), fft=True)
                ci_r = 1.96 / np.sqrt(len(resid))
                for j in range(len(acf_r)):
                    col_r = 'red' if abs(acf_r[j]) > ci_r and j > 0 else 'steelblue'
                    fig3.add_trace(go.Bar(x=[j], y=[acf_r[j]], marker_color=col_r, showlegend=False, width=0.3), row=2, col=1)
                fig3.add_hline(y=ci_r, line=dict(color='blue', dash='dash'), row=2, col=1)
                fig3.add_hline(y=-ci_r, line=dict(color='blue', dash='dash'), row=2, col=1)
                # QQ
                osm = stats.norm.ppf(np.linspace(1/(len(resid)+1), len(resid)/(len(resid)+1), len(resid)))
                osr = np.sort(resid.values)
                fig3.add_trace(go.Scatter(x=osm, y=osr, mode='markers', marker=dict(size=3, color='steelblue', opacity=0.6)), row=2, col=2)
                fig3.add_trace(go.Scatter(x=[osm.min(),osm.max()], y=[resid.mean()+resid.std()*osm.min(), resid.mean()+resid.std()*osm.max()],
                    mode='lines', line=dict(color='red', dash='dash')), row=2, col=2)
                fig3.update_layout(height=600, template='plotly_white', showlegend=False)
                st.plotly_chart(fig3, use_container_width=True)

                # Statistical tests on residuals
                st.markdown("### Assumption Tests")
                # Ljung-Box
                lb = acorr_ljungbox(resid, lags=[10, 20], return_df=True)
                st.markdown("**Ljung-Box (H₀: No autocorrelation)**")
                st.dataframe(lb.round(4), use_container_width=True)
                if (lb['lb_pvalue'] > 0.05).all(): st.success("No significant autocorrelation in residuals.")
                else: st.warning("Residuals show autocorrelation!")

                # Normality
                jb_stat, jb_p = stats.jarque_bera(resid)
                sw_stat, sw_p = stats.shapiro(resid[:5000]) if len(resid) <= 5000 else (None, None)
                st.markdown("**Normality (H₀: Normal)**")
                norm_tests = [{"Test":"Jarque-Bera","Statistic":f"{jb_stat:.4f}","p-value":f"{jb_p:.6f}","Normal":"Yes" if jb_p>0.05 else "No"}]
                if sw_stat: norm_tests.append({"Test":"Shapiro-Wilk","Statistic":f"{sw_stat:.4f}","p-value":f"{sw_p:.6f}","Normal":"Yes" if sw_p>0.05 else "No"})
                st.dataframe(pd.DataFrame(norm_tests), use_container_width=True, hide_index=True)

                # Heteroscedasticity (ARCH effect)
                st.markdown("**ARCH Effect (H₀: No ARCH effect)**")
                try:
                    arch_res = het_arch(resid, nlags=min(10, len(resid)//5))
                    st.dataframe(pd.DataFrame([{"LM Statistic":f"{arch_res[0]:.4f}","p-value":f"{arch_res[1]:.6f}",
                        "ARCH Effect":"No" if arch_res[1]>0.05 else "Yes - Consider GARCH!"}]), use_container_width=True, hide_index=True)
                except: st.info("ARCH test skipped.")

                # Store in session
                st.session_state['arima_result'] = result
                st.session_state['arima_order'] = (p,d,q)

            except Exception as e:
                st.error(f"Error fitting ARIMA: {e}")

# ============================================================
# 6. SARIMA
# ============================================================
elif module == 'sarima':
    st.header("6. SARIMA(p,d,q)(P,D,Q,s) Modeling")
    st.markdown("Seasonal ARIMA — untuk data dengan pola musiman.")

    transform = st.selectbox("Transformasi:", ['None', 'Log', 'Box-Cox'])
    if transform != 'None' and (y_full <= 0).any():
        y_model = y_full; bc_lam = None
    else:
        y_model, bc_lam = apply_transformation(y_full, transform)

    split_pct = st.slider("Train %:", 50, 95, 80, key='sarima_split')
    n_train = int(len(y_model) * split_pct / 100)
    y_train = y_model.iloc[:n_train]; y_test = y_model.iloc[n_train:]
    st.info(f"Train: {len(y_train)} | Test: {len(y_test)}")

    st.markdown("### Parameters")
    c1,c2,c3 = st.columns(3)
    p = c1.number_input("p:", 0, 5, 1, key='sp'); d = c2.number_input("d:", 0, 2, 1, key='sd'); q = c3.number_input("q:", 0, 5, 1, key='sq')
    c1,c2,c3,c4 = st.columns(4)
    P = c1.number_input("P:", 0, 3, 1, key='sP'); D = c2.number_input("D:", 0, 2, 1, key='sD')
    Q = c3.number_input("Q:", 0, 3, 1, key='sQ'); s = c4.number_input("s (period):", 1, 365, 12, key='ss')
    trend_opt = st.selectbox("Trend:", ['n', 'c', 't', 'ct'], key='sarima_trend')

    if st.button("Fit SARIMA", type="primary"):
        with st.spinner("Fitting SARIMA..."):
            try:
                model = SARIMAX(y_train, order=(p,d,q), seasonal_order=(P,D,Q,s), trend=trend_opt,
                                enforce_stationarity=False, enforce_invertibility=False)
                result = model.fit(disp=False, maxiter=500)

                st.text(str(result.summary()))

                c1,c2,c3 = st.columns(3)
                c1.metric("AIC", f"{result.aic:.2f}"); c2.metric("BIC", f"{result.bic:.2f}"); c3.metric("HQIC", f"{result.hqic:.2f}")

                # Forecast
                forecast_obj = result.get_forecast(steps=len(y_test))
                fc_mean = forecast_obj.predicted_mean
                fc_ci = forecast_obj.conf_int(alpha=0.05)

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=y_train.index, y=y_train.values, mode='lines', name='Train'))
                fig.add_trace(go.Scatter(x=y_test.index, y=y_test.values, mode='lines', name='Actual', line=dict(color='green', width=2)))
                fig.add_trace(go.Scatter(x=fc_mean.index, y=fc_mean.values, mode='lines', name='Forecast', line=dict(color='crimson', width=2)))
                ci_l = fc_ci.iloc[:,0]; ci_u = fc_ci.iloc[:,1]
                fig.add_trace(go.Scatter(x=ci_l.index.tolist()+ci_u.index.tolist()[::-1],
                    y=ci_l.values.tolist()+ci_u.values.tolist()[::-1],
                    fill='toself', fillcolor='rgba(255,0,0,0.1)', line=dict(color='rgba(0,0,0,0)'), name='95% CI'))
                fig.update_layout(title=f"SARIMA({p},{d},{q})({P},{D},{Q},{s})", height=420, template='plotly_white')
                st.plotly_chart(fig, use_container_width=True)

                # Metrics
                test_m = calc_metrics(y_test.values, fc_mean.values)
                st.dataframe(pd.DataFrame([{k: f"{v:.4f}" if v is not None else '-' for k,v in test_m.items()}]), use_container_width=True, hide_index=True)

                # Residual diagnostics
                resid = result.resid.dropna()
                fig2 = make_subplots(rows=1, cols=3, subplot_titles=("Residuals", "ACF Resid", "Q-Q"))
                fig2.add_trace(go.Scatter(y=resid.values, mode='lines', line=dict(width=0.8, color='gray')), row=1, col=1)
                acf_r = acf(resid, nlags=min(30, len(resid)//2-1), fft=True)
                ci_val = 1.96/np.sqrt(len(resid))
                for j in range(len(acf_r)):
                    c_r = 'red' if abs(acf_r[j])>ci_val and j>0 else 'steelblue'
                    fig2.add_trace(go.Bar(x=[j], y=[acf_r[j]], marker_color=c_r, showlegend=False, width=0.3), row=1, col=2)
                fig2.add_hline(y=ci_val, line=dict(color='blue', dash='dash'), row=1, col=2)
                fig2.add_hline(y=-ci_val, line=dict(color='blue', dash='dash'), row=1, col=2)
                osm = stats.norm.ppf(np.linspace(1/(len(resid)+1), len(resid)/(len(resid)+1), len(resid)))
                fig2.add_trace(go.Scatter(x=osm, y=np.sort(resid.values), mode='markers', marker=dict(size=3, opacity=0.6)), row=1, col=3)
                fig2.update_layout(height=350, template='plotly_white', showlegend=False)
                st.plotly_chart(fig2, use_container_width=True)

                lb = acorr_ljungbox(resid, lags=[10,20], return_df=True)
                jb_s, jb_p = stats.jarque_bera(resid)
                st.markdown(f"**Ljung-Box p(10):** {lb['lb_pvalue'].iloc[0]:.4f} | **Ljung-Box p(20):** {lb['lb_pvalue'].iloc[1]:.4f} | **JB p:** {jb_p:.4f}")

                st.session_state['sarima_result'] = result

            except Exception as e:
                st.error(f"Error: {e}")

# ============================================================
# 7. ARIMAX
# ============================================================
elif module == 'arimax':
    st.header("7. ARIMAX — ARIMA with Exogenous Variables")
    st.markdown("SARIMAX dengan variabel eksogen. Gunakan **Demo: Sales + Exog** untuk contoh.")

    exog_cols = [c for c in raw_df.columns if c != target_col]
    if not exog_cols:
        st.warning("Tidak ada variabel eksogen. Pilih dataset 'Sales + Exog'."); st.stop()

    sel_exog = st.multiselect("Variabel Exog:", exog_cols, default=exog_cols[:2])
    if not sel_exog: st.warning("Pilih min 1 variabel."); st.stop()

    X = raw_df[sel_exog].dropna()
    y_model = y_full.loc[X.index]

    split_pct = st.slider("Train %:", 50, 95, 80, key='arimax_split')
    n_train = int(len(y_model) * split_pct / 100)
    y_train = y_model.iloc[:n_train]; y_test = y_model.iloc[n_train:]
    X_train = X.iloc[:n_train]; X_test = X.iloc[n_train:]
    st.info(f"Train: {len(y_train)} | Test: {len(y_test)} | Exog: {sel_exog}")

    c1,c2,c3 = st.columns(3)
    p = c1.number_input("p:", 0, 5, 1, key='axp'); d = c2.number_input("d:", 0, 2, 1, key='axd'); q = c3.number_input("q:", 0, 5, 1, key='axq')
    c1,c2,c3,c4 = st.columns(4)
    P = c1.number_input("P:", 0, 3, 0, key='axP'); D = c2.number_input("D:", 0, 2, 0, key='axD')
    Q = c3.number_input("Q:", 0, 3, 0, key='axQ'); s = c4.number_input("s:", 1, 365, 12, key='axs')

    if st.button("Fit ARIMAX", type="primary"):
        with st.spinner("Fitting ARIMAX..."):
            try:
                model = SARIMAX(y_train, exog=X_train, order=(p,d,q), seasonal_order=(P,D,Q,s),
                                enforce_stationarity=False, enforce_invertibility=False)
                result = model.fit(disp=False, maxiter=500)
                st.text(str(result.summary()))

                fc = result.get_forecast(steps=len(y_test), exog=X_test)
                fc_mean = fc.predicted_mean; fc_ci = fc.conf_int(alpha=0.05)

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=y_train.index, y=y_train.values, mode='lines', name='Train'))
                fig.add_trace(go.Scatter(x=y_test.index, y=y_test.values, mode='lines', name='Actual', line=dict(color='green', width=2)))
                fig.add_trace(go.Scatter(x=fc_mean.index, y=fc_mean.values, mode='lines', name='Forecast', line=dict(color='crimson', width=2)))
                ci_l = fc_ci.iloc[:,0]; ci_u = fc_ci.iloc[:,1]
                fig.add_trace(go.Scatter(x=ci_l.index.tolist()+ci_u.index.tolist()[::-1],
                    y=ci_l.values.tolist()+ci_u.values.tolist()[::-1],
                    fill='toself', fillcolor='rgba(255,0,0,0.1)', line=dict(color='rgba(0,0,0,0)'), name='95% CI'))
                fig.update_layout(title="ARIMAX Forecast", height=420, template='plotly_white')
                st.plotly_chart(fig, use_container_width=True)

                test_m = calc_metrics(y_test.values, fc_mean.values)
                st.dataframe(pd.DataFrame([{k: f"{v:.4f}" if v is not None else '-' for k,v in test_m.items()}]), use_container_width=True, hide_index=True)

            except Exception as e:
                st.error(f"Error: {e}")

# ============================================================
# 8. ARIMA-GARCH
# ============================================================
elif module == 'garch':
    st.header("8. ARIMA-GARCH Modeling")
    st.markdown("Untuk data dengan **volatility clustering** (heteroscedasticity). Gunakan **Demo: Stock Price**.")

    if not HAS_ARCH:
        st.error("Library `arch` belum terinstall. Jalankan: `pip install arch`"); st.stop()

    # Returns
    use_returns = st.checkbox("Gunakan returns (diff log)?", True)
    if use_returns:
        y_model = np.log(y_full).diff().dropna() * 100  # percentage returns
        st.info("Using log-returns (x100)")
    else:
        y_model = y_full

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_model.index, y=y_model.values, mode='lines', line=dict(width=0.8)))
    fig.update_layout(title="Series for GARCH", height=300, template='plotly_white')
    st.plotly_chart(fig, use_container_width=True)

    split_pct = st.slider("Train %:", 50, 95, 80, key='garch_split')
    n_train = int(len(y_model) * split_pct / 100)
    y_train = y_model.iloc[:n_train]; y_test = y_model.iloc[n_train:]

    st.markdown("### ARIMA for Mean")
    c1,c2,c3 = st.columns(3)
    ar = c1.number_input("AR:", 0, 5, 1, key='gar'); d_g = c2.number_input("d:", 0, 2, 0, key='gd'); ma = c3.number_input("MA:", 0, 5, 1, key='gma')

    st.markdown("### GARCH for Variance")
    c1,c2 = st.columns(2)
    gp = c1.number_input("GARCH p:", 0, 5, 1, key='gp'); gq = c2.number_input("GARCH q:", 0, 5, 1, key='gq')
    vol_model = st.selectbox("Volatility Model:", ['GARCH', 'EGARCH', 'GJR-GARCH'])
    dist = st.selectbox("Error Distribution:", ['normal', 't', 'skewt', 'ged'])

    if st.button("Fit ARIMA-GARCH", type="primary"):
        with st.spinner("Fitting..."):
            try:
                vol_map = {'GARCH':'Garch', 'EGARCH':'EGARCH', 'GJR-GARCH':'GARCH'}
                o_map = {'GARCH':0, 'EGARCH':0, 'GJR-GARCH':gp}
                am = arch_model(y_train, mean='ARX', lags=ar, vol=vol_map[vol_model],
                               p=gp, q=gq, o=o_map[vol_model], dist=dist)
                res = am.fit(disp='off')
                st.text(str(res.summary()))

                # Conditional volatility
                fig = make_subplots(rows=2, cols=1, subplot_titles=("Returns + Cond. Volatility", "Standardized Residuals"), shared_xaxes=True)
                fig.add_trace(go.Scatter(x=y_train.index, y=y_train.values, mode='lines', line=dict(width=0.5, color='gray'), name='Returns'), row=1, col=1)
                fig.add_trace(go.Scatter(x=y_train.index, y=res.conditional_volatility, mode='lines', line=dict(color='crimson', width=1.5), name='Cond Vol'), row=1, col=1)
                fig.add_trace(go.Scatter(x=y_train.index, y=-res.conditional_volatility, mode='lines', line=dict(color='crimson', width=1.5), showlegend=False), row=1, col=1)
                std_resid = res.resid / res.conditional_volatility
                fig.add_trace(go.Scatter(x=y_train.index, y=std_resid, mode='lines', line=dict(width=0.5, color='steelblue')), row=2, col=1)
                fig.update_layout(height=500, template='plotly_white'); st.plotly_chart(fig, use_container_width=True)

                # Forecast
                fc = res.forecast(horizon=len(y_test))
                fc_mean = fc.mean.iloc[-1].values
                fc_var = fc.variance.iloc[-1].values
                fc_vol = np.sqrt(fc_var)

                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=y_test.index, y=y_test.values, mode='lines', name='Actual', line=dict(color='green')))
                fig2.add_trace(go.Scatter(x=y_test.index, y=fc_mean, mode='lines', name='Mean Forecast', line=dict(color='crimson')))
                fig2.add_trace(go.Scatter(x=y_test.index.tolist()+y_test.index.tolist()[::-1],
                    y=(fc_mean+1.96*fc_vol).tolist()+(fc_mean-1.96*fc_vol).tolist()[::-1],
                    fill='toself', fillcolor='rgba(255,0,0,0.1)', line=dict(color='rgba(0,0,0,0)'), name='95% CI'))
                fig2.update_layout(title="GARCH Forecast", height=400, template='plotly_white')
                st.plotly_chart(fig2, use_container_width=True)

                # Volatility forecast
                fig3 = go.Figure()
                fig3.add_trace(go.Scatter(x=y_test.index, y=fc_vol, mode='lines', line=dict(color='crimson', width=2)))
                fig3.update_layout(title="Forecasted Volatility", height=300, template='plotly_white')
                st.plotly_chart(fig3, use_container_width=True)

                test_m = calc_metrics(y_test.values[:len(fc_mean)], fc_mean[:len(y_test)])
                st.dataframe(pd.DataFrame([{k: f"{v:.4f}" if v is not None else '-' for k,v in test_m.items()}]), use_container_width=True, hide_index=True)

            except Exception as e:
                st.error(f"Error: {e}")

# ============================================================
# 9. AUTO ARIMA (GRID SEARCH)
# ============================================================
elif module == 'auto_arima':
    st.header("9. Auto ARIMA — Grid Search")
    st.markdown("Cari kombinasi (p,d,q)(P,D,Q,s) terbaik secara otomatis berdasarkan AIC/BIC.")

    transform = st.selectbox("Transformasi:", ['None', 'Log'], key='auto_tr')
    if transform == 'Log' and (y_full <= 0).any():
        y_model = y_full
    elif transform == 'Log':
        y_model = np.log(y_full)
    else:
        y_model = y_full

    split_pct = st.slider("Train %:", 50, 95, 80, key='auto_split')
    n_train = int(len(y_model) * split_pct / 100)
    y_train = y_model.iloc[:n_train]; y_test = y_model.iloc[n_train:]

    st.markdown("### Search Range")
    c1,c2,c3 = st.columns(3)
    max_p = c1.number_input("Max p:", 0, 5, 3); max_d = c2.number_input("Max d:", 0, 2, 2); max_q = c3.number_input("Max q:", 0, 5, 3)
    seasonal = st.checkbox("Include Seasonal?", True)
    if seasonal:
        c1,c2,c3,c4 = st.columns(4)
        max_P = c1.number_input("Max P:", 0, 2, 1); max_D = c2.number_input("Max D:", 0, 1, 1)
        max_Q = c3.number_input("Max Q:", 0, 2, 1); s = c4.number_input("s:", 1, 365, 12, key='auto_s')
    criterion = st.selectbox("Kriteria:", ['AIC', 'BIC'])

    if st.button("Run Grid Search", type="primary"):
        with st.spinner("Searching... This may take a while."):
            results = []
            p_range = range(0, max_p+1); d_range = range(0, max_d+1); q_range = range(0, max_q+1)
            if seasonal:
                P_range = range(0, max_P+1); D_range = range(0, max_D+1); Q_range = range(0, max_Q+1)
                combos = list(itertools.product(p_range, d_range, q_range, P_range, D_range, Q_range))
            else:
                combos = list(itertools.product(p_range, d_range, q_range))

            progress = st.progress(0)
            total = len(combos)
            for idx, combo in enumerate(combos):
                progress.progress((idx+1)/total)
                try:
                    if seasonal:
                        pp,dd,qq,PP,DD,QQ = combo
                        model = SARIMAX(y_train, order=(pp,dd,qq), seasonal_order=(PP,DD,QQ,s),
                                        enforce_stationarity=False, enforce_invertibility=False)
                    else:
                        pp,dd,qq = combo
                        model = ARIMA(y_train, order=(pp,dd,qq))
                    res = model.fit(disp=False, maxiter=200)
                    fc = res.get_forecast(steps=len(y_test))
                    fc_mean = fc.predicted_mean
                    test_m = calc_metrics(y_test.values, fc_mean.values)
                    row = {
                        'Order': f"({pp},{dd},{qq})",
                        'AIC': round(res.aic, 2), 'BIC': round(res.bic, 2),
                        'RMSE_test': round(test_m.get('RMSE', 999), 4),
                        'MAPE_test': round(test_m.get('MAPE', 999), 2) if test_m.get('MAPE') else '-',
                        'MAE_test': round(test_m.get('MAE', 999), 4)
                    }
                    if seasonal: row['Seasonal'] = f"({PP},{DD},{QQ},{s})"
                    results.append(row)
                except:
                    pass
            progress.empty()

            if results:
                rdf = pd.DataFrame(results)
                sort_col = criterion
                rdf = rdf.sort_values(sort_col)
                st.markdown(f"### Results (sorted by {criterion}) — Top 20")
                st.dataframe(rdf.head(20), use_container_width=True, hide_index=True)

                best = rdf.iloc[0]
                st.success(f"Best: **{best['Order']}** {best.get('Seasonal','')} | AIC={best['AIC']} | BIC={best['BIC']} | RMSE={best['RMSE_test']}")

                # Fit best model
                best_order = eval(best['Order'])
                if seasonal:
                    best_seasonal = eval(best['Seasonal'].replace(f',{s}', '') + f',{s})')  if 'Seasonal' in best else (0,0,0,s)
                    bm = SARIMAX(y_train, order=best_order, seasonal_order=best_seasonal,
                                 enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
                else:
                    bm = ARIMA(y_train, order=best_order).fit()

                fc = bm.get_forecast(steps=len(y_test))
                fc_mean = fc.predicted_mean; fc_ci = fc.conf_int(alpha=0.05)

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=y_train.index, y=y_train.values, mode='lines', name='Train'))
                fig.add_trace(go.Scatter(x=y_test.index, y=y_test.values, mode='lines', name='Actual', line=dict(color='green', width=2)))
                fig.add_trace(go.Scatter(x=fc_mean.index, y=fc_mean.values, mode='lines', name='Forecast', line=dict(color='crimson', width=2)))
                ci_l = fc_ci.iloc[:,0]; ci_u = fc_ci.iloc[:,1]
                fig.add_trace(go.Scatter(x=ci_l.index.tolist()+ci_u.index.tolist()[::-1],
                    y=ci_l.values.tolist()+ci_u.values.tolist()[::-1],
                    fill='toself', fillcolor='rgba(255,0,0,0.1)', line=dict(color='rgba(0,0,0,0)'), name='95% CI'))
                fig.update_layout(title=f"Best: {best['Order']} {best.get('Seasonal','')}", height=420, template='plotly_white')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("No valid models found.")

# ============================================================
# 10. MODEL COMPARISON
# ============================================================
elif module == 'comparison':
    st.header("10. Model Comparison")
    st.markdown("Bandingkan beberapa model secara side-by-side.")

    transform = st.selectbox("Transformasi:", ['None', 'Log'], key='comp_tr')
    if transform == 'Log' and (y_full > 0).all():
        y_model = np.log(y_full)
    else:
        y_model = y_full

    split_pct = st.slider("Train %:", 50, 95, 80, key='comp_split')
    n_train = int(len(y_model) * split_pct / 100)
    y_train = y_model.iloc[:n_train]; y_test = y_model.iloc[n_train:]

    models_to_compare = st.multiselect("Models:", [
        'ARIMA(1,1,1)', 'ARIMA(2,1,2)', 'ARIMA(1,1,0)', 'ARIMA(0,1,1)',
        'SARIMA(1,1,1)(1,1,1,12)', 'SARIMA(0,1,1)(0,1,1,12)', 'SARIMA(1,0,1)(1,1,0,12)',
        'Holt-Winters (Add)', 'Holt-Winters (Mul)', 'Naive', 'Drift'
    ], default=['ARIMA(1,1,1)', 'SARIMA(1,1,1)(1,1,1,12)', 'Holt-Winters (Add)', 'Naive'])

    if st.button("Compare Models", type="primary") and models_to_compare:
        results = []; forecasts = {}
        for mname in models_to_compare:
            try:
                if mname.startswith('ARIMA'):
                    order = eval(mname.replace('ARIMA',''))
                    m = ARIMA(y_train, order=order).fit()
                    fc = m.get_forecast(len(y_test)).predicted_mean
                    aic = m.aic; bic = m.bic
                elif mname.startswith('SARIMA'):
                    parts = mname.replace('SARIMA','').split(')(')
                    order = eval(parts[0]+')')
                    seasonal = eval('('+parts[1])
                    m = SARIMAX(y_train, order=order, seasonal_order=seasonal,
                                enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
                    fc = m.get_forecast(len(y_test)).predicted_mean
                    aic = m.aic; bic = m.bic
                elif 'Holt' in mname:
                    seasonal_type = 'add' if 'Add' in mname else 'mul'
                    try:
                        m = ExponentialSmoothing(y_train, trend='add', seasonal=seasonal_type, seasonal_periods=12).fit()
                        fc = m.forecast(len(y_test))
                        aic = m.aic; bic = m.bic
                    except:
                        m = ExponentialSmoothing(y_train, trend='add', seasonal='add', seasonal_periods=12).fit()
                        fc = m.forecast(len(y_test))
                        aic = m.aic; bic = m.bic
                elif mname == 'Naive':
                    fc = pd.Series(y_train.iloc[-1], index=y_test.index)
                    aic = None; bic = None
                elif mname == 'Drift':
                    drift = (y_train.iloc[-1] - y_train.iloc[0]) / (len(y_train) - 1)
                    fc = pd.Series([y_train.iloc[-1] + drift*(i+1) for i in range(len(y_test))], index=y_test.index)
                    aic = None; bic = None
                else:
                    continue

                metrics = calc_metrics(y_test.values, fc.values[:len(y_test)])
                results.append({
                    'Model': mname, 'AIC': f"{aic:.2f}" if aic else '-', 'BIC': f"{bic:.2f}" if bic else '-',
                    'RMSE': f"{metrics.get('RMSE',0):.4f}", 'MAE': f"{metrics.get('MAE',0):.4f}",
                    'MAPE': f"{metrics.get('MAPE',0):.2f}%" if metrics.get('MAPE') else '-',
                    'R2': f"{metrics.get('R2',0):.4f}" if metrics.get('R2') is not None else '-'
                })
                forecasts[mname] = fc
            except Exception as e:
                results.append({'Model': mname, 'AIC':'-','BIC':'-','RMSE':f'Error: {str(e)[:30]}','MAE':'-','MAPE':'-','R2':'-'})

        st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)

        # Plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_train.index, y=y_train.values, mode='lines', name='Train', line=dict(color='gray', width=1)))
        fig.add_trace(go.Scatter(x=y_test.index, y=y_test.values, mode='lines', name='Actual', line=dict(color='black', width=2)))
        colors = px.colors.qualitative.Set1
        for i, (mname, fc) in enumerate(forecasts.items()):
            fig.add_trace(go.Scatter(x=fc.index, y=fc.values, mode='lines', name=mname,
                                     line=dict(color=colors[i % len(colors)], width=1.5)))
        fig.update_layout(title="Model Comparison", height=450, template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# 11. FORECASTING & CI/PI
# ============================================================
elif module == 'forecast':
    st.header("11. Forecasting with Confidence & Prediction Intervals")
    st.markdown("Forecast ke depan dengan **CI** (confidence interval) dan **PI** (prediction interval).")

    transform = st.selectbox("Transformasi:", ['None', 'Log'], key='fc_tr')
    if transform == 'Log' and (y_full > 0).all():
        y_model = np.log(y_full)
    else:
        y_model = y_full

    st.markdown("### Model Configuration")
    model_type = st.selectbox("Model:", ['ARIMA', 'SARIMA'])
    c1,c2,c3 = st.columns(3)
    p = c1.number_input("p:", 0, 5, 1, key='fcp'); d = c2.number_input("d:", 0, 2, 1, key='fcd'); q = c3.number_input("q:", 0, 5, 1, key='fcq')
    if model_type == 'SARIMA':
        c1,c2,c3,c4 = st.columns(4)
        P = c1.number_input("P:", 0, 3, 1, key='fcP'); D = c2.number_input("D:", 0, 2, 1, key='fcD')
        Q = c3.number_input("Q:", 0, 3, 1, key='fcQ'); s = c4.number_input("s:", 1, 365, 12, key='fcs')

    h = st.slider("Forecast horizon:", 1, 120, 24)
    alpha = st.slider("Significance level (alpha):", 0.01, 0.20, 0.05, 0.01)
    ci_level = f"{(1-alpha)*100:.0f}%"

    if st.button("Generate Forecast", type="primary"):
        with st.spinner("Fitting full model & forecasting..."):
            try:
                if model_type == 'ARIMA':
                    model = ARIMA(y_model, order=(p,d,q)).fit()
                else:
                    model = SARIMAX(y_model, order=(p,d,q), seasonal_order=(P,D,Q,s),
                                    enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)

                # Summary
                c1,c2,c3 = st.columns(3)
                c1.metric("AIC", f"{model.aic:.2f}"); c2.metric("BIC", f"{model.bic:.2f}"); c3.metric("Log-Lik", f"{model.llf:.2f}")

                # Forecast
                fc_obj = model.get_forecast(steps=h)
                fc_mean = fc_obj.predicted_mean
                fc_ci = fc_obj.conf_int(alpha=alpha)

                # PI (wider): approximate as CI * sqrt adjustment
                resid_std = model.resid.std()
                fc_se = np.array([resid_std * np.sqrt(1 + i * 0.01) for i in range(h)])  # growing uncertainty
                z = stats.norm.ppf(1 - alpha/2)
                pi_lower = fc_mean - z * np.sqrt((fc_ci.iloc[:,1] - fc_mean)**2 + resid_std**2)
                pi_upper = fc_mean + z * np.sqrt((fc_ci.iloc[:,1] - fc_mean)**2 + resid_std**2)

                # Inverse transform
                if transform == 'Log':
                    fc_mean_orig = np.exp(fc_mean)
                    ci_l_orig = np.exp(fc_ci.iloc[:,0]); ci_u_orig = np.exp(fc_ci.iloc[:,1])
                    pi_l_orig = np.exp(pi_lower); pi_u_orig = np.exp(pi_upper)
                    y_plot = np.exp(y_model)
                else:
                    fc_mean_orig = fc_mean
                    ci_l_orig = fc_ci.iloc[:,0]; ci_u_orig = fc_ci.iloc[:,1]
                    pi_l_orig = pi_lower; pi_u_orig = pi_upper
                    y_plot = y_model

                # Plot
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=y_plot.index, y=y_plot.values, mode='lines', name='Historical', line=dict(color='steelblue', width=1.5)))
                fig.add_trace(go.Scatter(x=fc_mean_orig.index, y=fc_mean_orig.values, mode='lines', name='Forecast', line=dict(color='crimson', width=2)))

                # CI
                fig.add_trace(go.Scatter(
                    x=ci_l_orig.index.tolist() + ci_u_orig.index.tolist()[::-1],
                    y=ci_l_orig.values.tolist() + ci_u_orig.values.tolist()[::-1],
                    fill='toself', fillcolor='rgba(255,0,0,0.15)', line=dict(color='rgba(0,0,0,0)'),
                    name=f'{ci_level} CI'))

                # PI
                fig.add_trace(go.Scatter(
                    x=pi_l_orig.index.tolist() + pi_u_orig.index.tolist()[::-1],
                    y=pi_l_orig.values.tolist() + pi_u_orig.values.tolist()[::-1],
                    fill='toself', fillcolor='rgba(255,165,0,0.08)', line=dict(color='rgba(0,0,0,0)'),
                    name=f'{ci_level} PI'))

                fig.update_layout(title=f"Forecast — {h} periods ahead ({ci_level} CI & PI)",
                                  height=500, template='plotly_white')
                st.plotly_chart(fig, use_container_width=True)

                # Forecast table
                fc_df = pd.DataFrame({
                    'Date': fc_mean_orig.index,
                    'Forecast': fc_mean_orig.values.round(4),
                    f'CI Lower ({ci_level})': ci_l_orig.values.round(4),
                    f'CI Upper ({ci_level})': ci_u_orig.values.round(4),
                    f'PI Lower ({ci_level})': pi_l_orig.values.round(4),
                    f'PI Upper ({ci_level})': pi_u_orig.values.round(4),
                })
                st.dataframe(fc_df, use_container_width=True, hide_index=True)

                # Download
                csv = fc_df.to_csv(index=False)
                st.download_button("Download Forecast CSV", csv, "forecast.csv", "text/csv")

                # Model diagnostics
                st.markdown("### Final Model Diagnostics")
                resid = model.resid.dropna()
                lb = acorr_ljungbox(resid, lags=[10,20], return_df=True)
                jb_s, jb_p = stats.jarque_bera(resid)

                fig2 = make_subplots(rows=1, cols=3, subplot_titles=("Residuals", "ACF Resid", "Q-Q"))
                fig2.add_trace(go.Scatter(y=resid.values, mode='lines', line=dict(width=0.8, color='gray')), row=1, col=1)
                fig2.add_hline(y=0, line=dict(color='red', dash='dash'), row=1, col=1)
                acf_r = acf(resid, nlags=min(30, len(resid)//2-1), fft=True)
                ci_v = 1.96/np.sqrt(len(resid))
                for j in range(len(acf_r)):
                    cr = 'red' if abs(acf_r[j])>ci_v and j>0 else 'steelblue'
                    fig2.add_trace(go.Bar(x=[j], y=[acf_r[j]], marker_color=cr, showlegend=False, width=0.3), row=1, col=2)
                fig2.add_hline(y=ci_v, line=dict(color='blue', dash='dash'), row=1, col=2)
                fig2.add_hline(y=-ci_v, line=dict(color='blue', dash='dash'), row=1, col=2)
                osm = stats.norm.ppf(np.linspace(1/(len(resid)+1), len(resid)/(len(resid)+1), len(resid)))
                fig2.add_trace(go.Scatter(x=osm, y=np.sort(resid.values), mode='markers', marker=dict(size=3, opacity=0.6)), row=1, col=3)
                fig2.update_layout(height=350, template='plotly_white', showlegend=False)
                st.plotly_chart(fig2, use_container_width=True)

                st.markdown(f"**Ljung-Box p(10):** {lb['lb_pvalue'].iloc[0]:.4f} | **JB p:** {jb_p:.4f}")
                if lb['lb_pvalue'].iloc[0] > 0.05 and jb_p > 0.05:
                    st.success("Residuals: No autocorrelation + Normal. Model assumptions satisfied.")
                elif lb['lb_pvalue'].iloc[0] > 0.05:
                    st.warning("Residuals: No autocorrelation, but NOT normal. Consider robust CI.")
                else:
                    st.error("Residuals show autocorrelation. Consider adjusting model order.")

            except Exception as e:
                st.error(f"Error: {e}")

# ============================================================
# FOOTER
# ============================================================
st.divider()
st.markdown("<div style='text-align:center;color:gray;font-size:0.85rem'><b>ARIMA Pro — Time Series Analysis</b><br>11 Modul | ARIMA · SARIMA · ARIMAX · GARCH · Auto Search · Forecast | 2026</div>", unsafe_allow_html=True)
