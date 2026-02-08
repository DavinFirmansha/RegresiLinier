import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings, io, itertools
warnings.filterwarnings('ignore')

try:
    from arch import arch_model
    HAS_ARCH = True
except ImportError:
    HAS_ARCH = False

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="ARIMA Pro", layout="wide", page_icon="📈")
st.title("📈 ARIMA Pro — Analisis Time Series Lengkap")
st.caption("EDA · Decomposition · Stationarity · ARIMA/SARIMA · ARIMAX · GARCH · Forecast · Evaluation")

# ============================================================
# DEMO DATA
# ============================================================
@st.cache_data
def load_demo_airline():
    np.random.seed(42)
    n = 144; t = np.arange(n)
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
    vol = np.zeros(n); vol[0] = 0.02
    for i in range(1, n):
        vol[i] = np.sqrt(0.00001 + 0.1 * returns[i-1]**2 + 0.85 * vol[i-1]**2)
        returns[i] = np.random.normal(0.0005, vol[i])
    price = 100 * np.exp(np.cumsum(returns))
    dates = pd.date_range('2023-01-02', periods=n, freq='B')
    return pd.DataFrame({'Date': dates, 'Price': np.round(price, 2), 'Volume': np.random.randint(1000000, 5000000, n)})

@st.cache_data
def load_demo_sales():
    np.random.seed(99)
    n = 120; t = np.arange(n)
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
    n = 144; t = np.arange(n)
    trend = 100 + 1.5 * t
    seasonal_mult = 1 + 0.3 * np.sin(2 * np.pi * t / 12)
    noise = np.random.normal(1, 0.05, n)
    y = trend * seasonal_mult * noise
    dates = pd.date_range('1949-01-01', periods=n, freq='MS')
    return pd.DataFrame({'Date': dates, 'Value': np.round(y, 2)})

# ============================================================
# HELPERS
# ============================================================
def adf_test(series):
    result = adfuller(series.dropna(), autolag='AIC')
    return {'Test Statistic': round(result[0], 4), 'p-value': round(result[1], 6),
            'Lags Used': result[2], 'Observations': result[3],
            'Critical 1%': round(result[4]['1%'], 4), 'Critical 5%': round(result[4]['5%'], 4),
            'Stationary': 'Yes' if result[1] < 0.05 else 'No'}

def kpss_test(series, regression='c'):
    try:
        result = kpss(series.dropna(), regression=regression, nlags='auto')
        return {'Test Statistic': round(result[0], 4), 'p-value': round(result[1], 6),
                'Lags Used': result[2], 'Critical 5%': round(result[3]['5%'], 4),
                'Stationary': 'Yes' if result[1] > 0.05 else 'No'}
    except Exception as e:
        return {'Test Statistic': None, 'p-value': None, 'Error': str(e), 'Stationary': 'Unknown'}

def calc_metrics(actual, predicted):
    actual = np.array(actual, dtype=float)
    predicted = np.array(predicted, dtype=float)
    min_len = min(len(actual), len(predicted))
    actual = actual[:min_len]; predicted = predicted[:min_len]
    mask = np.isfinite(actual) & np.isfinite(predicted)
    actual = actual[mask]; predicted = predicted[mask]
    if len(actual) == 0:
        return {'ME':0,'MAE':0,'MSE':0,'RMSE':0,'MAPE':None,'R2':None}
    e = actual - predicted; ae = np.abs(e); se = e**2
    nonzero = actual != 0
    mape_vals = np.abs(e[nonzero] / actual[nonzero]) * 100 if nonzero.sum() > 0 else np.array([])
    var_a = np.var(actual)
    return {
        'ME': np.mean(e), 'MAE': np.mean(ae), 'MSE': np.mean(se),
        'RMSE': np.sqrt(np.mean(se)),
        'MAPE': np.mean(mape_vals) if len(mape_vals) > 0 else None,
        'R2': 1 - np.sum(se) / np.sum((actual - actual.mean())**2) if var_a > 0 else None
    }

def plot_acf_pacf(series, lags=40, title=""):
    series_clean = series.dropna()
    max_lag = min(lags, len(series_clean) // 2 - 2)
    if max_lag < 1:
        max_lag = 1
    acf_vals = acf(series_clean, nlags=max_lag, fft=True)
    pacf_vals = pacf(series_clean, nlags=max_lag)
    n = len(series_clean)
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
    if method == 'None':
        return series.copy(), None
    elif method == 'Log':
        s = series.copy()
        s[s <= 0] = np.nan
        return np.log(s).dropna(), None
    elif method == 'Sqrt':
        s = series.copy()
        s[s < 0] = np.nan
        return np.sqrt(s).dropna(), None
    elif method == 'Box-Cox':
        from scipy.stats import boxcox as bc
        s = series.copy()
        s = s[s > 0]
        transformed, lam = bc(s.values)
        return pd.Series(transformed, index=s.index), lam
    elif method == 'Diff(1)':
        return series.diff().dropna(), None
    elif method == 'Diff(2)':
        return series.diff().diff().dropna(), None
    elif method == 'Seasonal Diff':
        return series.diff(12).dropna(), None
    elif method == 'Log + Diff(1)':
        s = series.copy()
        s[s <= 0] = np.nan
        return np.log(s).diff().dropna(), None
    return series.copy(), None

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

# DATA
st.sidebar.markdown("---")
st.sidebar.subheader("Data")
data_src = st.sidebar.selectbox("Sumber Data:", [
    'Demo: Airline Passengers', 'Demo: Stock Price (GARCH)',
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
y_full = raw_df[target_col].dropna().astype(float)

st.sidebar.markdown(f"**n = {len(y_full)}** | {y_full.index[0].strftime('%Y-%m')} → {y_full.index[-1].strftime('%Y-%m')}")

# ============================================================
# 1. EDA
# ============================================================
if module == 'eda':
    st.header("1. Exploratory Data Analysis")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_full.index, y=y_full.values, mode='lines', line=dict(color='steelblue', width=1.5)))
    fig.update_layout(title=f"Time Series: {target_col}", height=400, template='plotly_white')
    st.plotly_chart(fig, use_container_width=True)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("n", str(len(y_full))); c2.metric("Mean", f"{y_full.mean():.2f}")
    c3.metric("Std", f"{y_full.std():.2f}"); c4.metric("CV%", f"{y_full.std()/y_full.mean()*100:.2f}%")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Min", f"{y_full.min():.2f}"); c2.metric("Max", f"{y_full.max():.2f}")
    c3.metric("Skewness", f"{y_full.skew():.4f}"); c4.metric("Kurtosis", f"{y_full.kurtosis():.4f}")

    fig2 = make_subplots(rows=1, cols=3, subplot_titles=("Histogram", "Box Plot", "Rolling Stats"))
    fig2.add_trace(go.Histogram(x=y_full, nbinsx=30, marker_color='steelblue', opacity=0.7), row=1, col=1)
    fig2.add_trace(go.Box(y=y_full, marker_color='steelblue'), row=1, col=2)
    window = max(6, len(y_full) // 20)
    rm = y_full.rolling(window).mean(); rs = y_full.rolling(window).std()
    fig2.add_trace(go.Scatter(x=rm.index, y=rm.values, mode='lines', name='Rolling Mean', line=dict(color='crimson')), row=1, col=3)
    fig2.add_trace(go.Scatter(x=rs.index, y=rs.values, mode='lines', name='Rolling Std', line=dict(color='green')), row=1, col=3)
    fig2.update_layout(height=350, template='plotly_white'); st.plotly_chart(fig2, use_container_width=True)

    if hasattr(y_full.index, 'month'):
        monthly = y_full.groupby(y_full.index.month).mean()
        fig3 = go.Figure(data=[go.Bar(x=monthly.index, y=monthly.values, marker_color='steelblue')])
        fig3.update_layout(title="Average by Month", height=320, template='plotly_white')
        st.plotly_chart(fig3, use_container_width=True)

    with st.expander("Raw Data"):
        st.dataframe(raw_df.round(4), use_container_width=True)

# ============================================================
# 2. DECOMPOSITION
# ============================================================
elif module == 'decomposition':
    st.header("2. Time Series Decomposition")
    c1,c2,c3 = st.columns(3)
    method = c1.selectbox("Metode:", ['Classical', 'STL'])
    model_type = c2.selectbox("Model:", ['additive', 'multiplicative'])
    period = c3.number_input("Period:", 2, 365, 12)

    if method == 'Classical':
        try:
            decomp = seasonal_decompose(y_full, model=model_type, period=period)
        except Exception as e:
            st.error(f"Error: {e}"); st.stop()
    else:
        if model_type == 'multiplicative':
            st.info("STL hanya mendukung additive. Untuk multiplicative, data di-log dulu lalu di-STL.")
            try:
                y_log = np.log(y_full)
                decomp = STL(y_log, period=period).fit()
            except Exception as e:
                st.error(f"Error: {e}"); st.stop()
        else:
            try:
                decomp = STL(y_full, period=period).fit()
            except Exception as e:
                st.error(f"Error: {e}"); st.stop()

    fig = make_subplots(rows=4, cols=1, subplot_titles=("Observed", "Trend", "Seasonal", "Residual"),
                        shared_xaxes=True, vertical_spacing=0.05)
    obs_vals = np.log(y_full) if (method == 'STL' and model_type == 'multiplicative') else y_full
    fig.add_trace(go.Scatter(x=obs_vals.index, y=obs_vals.values, mode='lines', line=dict(width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=y_full.index, y=decomp.trend, mode='lines', line=dict(color='crimson', width=1.5)), row=2, col=1)
    fig.add_trace(go.Scatter(x=y_full.index, y=decomp.seasonal, mode='lines', line=dict(color='green', width=1)), row=3, col=1)
    fig.add_trace(go.Scatter(x=y_full.index, y=decomp.resid, mode='markers', marker=dict(size=3, color='gray')), row=4, col=1)
    fig.update_layout(height=700, template='plotly_white', showlegend=False); st.plotly_chart(fig, use_container_width=True)

    resid = pd.Series(decomp.resid).dropna()
    c1,c2,c3 = st.columns(3)
    c1.metric("Trend range", f"{np.nanmin(decomp.trend):.2f} - {np.nanmax(decomp.trend):.2f}")
    c2.metric("Seasonal amp", f"{np.nanmax(decomp.seasonal) - np.nanmin(decomp.seasonal):.2f}")
    c3.metric("Resid std", f"{resid.std():.4f}")

    var_resid = resid.var()
    trend_s = pd.Series(decomp.trend).dropna()
    Ft = max(0, 1 - var_resid / (trend_s.var() + var_resid)) if (trend_s.var() + var_resid) > 0 else 0
    seas_s = pd.Series(decomp.seasonal).dropna()
    Fs = max(0, 1 - var_resid / (seas_s.var() + var_resid)) if (seas_s.var() + var_resid) > 0 else 0
    st.markdown(f"**Strength of Trend:** {Ft:.4f} | **Strength of Seasonality:** {Fs:.4f}")

# ============================================================
# 3. STATIONARITY
# ============================================================
elif module == 'stationarity':
    st.header("3. Stationarity Testing")
    transform = st.selectbox("Transformasi:", ['None', 'Log', 'Sqrt', 'Box-Cox', 'Diff(1)', 'Diff(2)', 'Seasonal Diff', 'Log + Diff(1)'])
    y_test, bc_lambda = apply_transformation(y_full, transform)
    if len(y_test) < 10:
        st.error("Terlalu sedikit data setelah transformasi."); st.stop()

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Original", f"After: {transform}"))
    fig.add_trace(go.Scatter(x=y_full.index, y=y_full.values, mode='lines', line=dict(width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=y_test.index, y=y_test.values, mode='lines', line=dict(width=1, color='crimson')), row=1, col=2)
    fig.update_layout(height=320, template='plotly_white', showlegend=False); st.plotly_chart(fig, use_container_width=True)

    st.markdown("### ADF Test (H₀: Unit Root / Non-Stationary)")
    adf_res = adf_test(y_test)
    st.dataframe(pd.DataFrame([adf_res]), use_container_width=True, hide_index=True)
    if adf_res['Stationary'] == 'Yes':
        st.success("ADF: Stationary")
    else:
        st.warning("ADF: Non-Stationary")

    st.markdown("### KPSS Test (H₀: Stationary)")
    kpss_res = kpss_test(y_test)
    st.dataframe(pd.DataFrame([kpss_res]), use_container_width=True, hide_index=True)
    if kpss_res.get('Stationary') == 'Yes':
        st.success("KPSS: Stationary")
    else:
        st.warning("KPSS: Non-Stationary")

    st.markdown("### Ringkasan")
    adf_ok = adf_res['Stationary'] == 'Yes'
    kpss_ok = kpss_res.get('Stationary') == 'Yes'
    if adf_ok and kpss_ok:
        st.success("Kedua test: **STATIONARY**.")
    elif adf_ok and not kpss_ok:
        st.info("ADF: stationary, KPSS: non-stationary → **trend-stationary**.")
    elif not adf_ok and kpss_ok:
        st.info("ADF: non-stationary, KPSS: stationary → **difference-stationary**.")
    else:
        st.error("Kedua test: **NON-STATIONARY**. Perlu differencing/transformasi.")

    if not adf_ok:
        st.markdown("### Auto-Differencing")
        d = 0; temp = y_full.copy()
        for dd in range(1, 4):
            temp = temp.diff().dropna()
            if len(temp) < 10:
                break
            r = adfuller(temp, autolag='AIC')
            if r[1] < 0.05:
                d = dd; break
        if d > 0:
            st.info(f"Suggested d = **{d}**")
        else:
            st.warning("Masih non-stationary setelah 3x differencing.")

    if bc_lambda is not None:
        st.info(f"Box-Cox lambda = **{bc_lambda:.4f}**")

# ============================================================
# 4. ACF & PACF
# ============================================================
elif module == 'acf_pacf':
    st.header("4. ACF & PACF Analysis")
    transform = st.selectbox("Transformasi:", ['None', 'Log', 'Diff(1)', 'Diff(2)', 'Seasonal Diff', 'Log + Diff(1)'])
    y_plot, _ = apply_transformation(y_full, transform)
    if len(y_plot) < 10:
        st.error("Data terlalu sedikit."); st.stop()
    max_lags = st.slider("Max lags:", 5, min(80, len(y_plot)//2 - 2), min(40, len(y_plot)//3))
    fig = plot_acf_pacf(y_plot, lags=max_lags, title=f"({transform})")
    st.plotly_chart(fig, use_container_width=True)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=y_plot.index, y=y_plot.values, mode='lines', line=dict(width=1)))
    fig2.update_layout(title=f"Series ({transform})", height=300, template='plotly_white')
    st.plotly_chart(fig2, use_container_width=True)

    with st.expander("Panduan Identifikasi Model"):
        st.markdown("""
| ACF Pattern | PACF Pattern | Model Suggested |
|---|---|---|
| Tails off (decays) | Cuts off after lag p | **AR(p)** |
| Cuts off after lag q | Tails off (decays) | **MA(q)** |
| Tails off | Tails off | **ARMA(p,q)** |
| Significant at lag s, 2s, 3s... | - | **Seasonal** |
""")

# ============================================================
# 5. ARIMA
# ============================================================
elif module == 'arima':
    st.header("5. ARIMA(p,d,q) Modeling")
    transform = st.selectbox("Transformasi:", ['None', 'Log', 'Sqrt', 'Box-Cox'], key='ar_tr')
    y_model, bc_lam = apply_transformation(y_full, transform)
    if len(y_model) < 10:
        st.error("Data terlalu sedikit."); st.stop()

    split_pct = st.slider("Train %:", 50, 95, 80)
    n_train = int(len(y_model) * split_pct / 100)
    y_train = y_model.iloc[:n_train]; y_test = y_model.iloc[n_train:]
    st.info(f"Train: {len(y_train)} | Test: {len(y_test)}")

    c1,c2,c3 = st.columns(3)
    p = c1.number_input("p (AR):", 0, 10, 1); d = c2.number_input("d (Diff):", 0, 3, 1); q = c3.number_input("q (MA):", 0, 10, 1)
    trend_opt = st.selectbox("Trend:", ['n', 'c', 't', 'ct'])

    if st.button("Fit ARIMA", type="primary"):
        with st.spinner("Fitting..."):
            try:
                model = ARIMA(y_train, order=(p,d,q), trend=trend_opt)
                result = model.fit()
                st.text(str(result.summary()))

                c1,c2,c3,c4 = st.columns(4)
                c1.metric("AIC", f"{result.aic:.2f}"); c2.metric("BIC", f"{result.bic:.2f}")
                c3.metric("HQIC", f"{result.hqic:.2f}"); c4.metric("Log-Lik", f"{result.llf:.2f}")

                fitted = result.fittedvalues
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=y_train.index, y=y_train.values, mode='lines', name='Actual'))
                fig.add_trace(go.Scatter(x=fitted.index, y=fitted.values, mode='lines', name='Fitted', line=dict(dash='dash', color='crimson')))
                fig.update_layout(title="Fitted vs Actual (Train)", height=380, template='plotly_white')
                st.plotly_chart(fig, use_container_width=True)

                fc_obj = result.get_forecast(steps=len(y_test))
                fc_mean = fc_obj.predicted_mean
                fc_ci = fc_obj.conf_int(alpha=0.05)

                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=y_train.index, y=y_train.values, mode='lines', name='Train'))
                fig2.add_trace(go.Scatter(x=y_test.index, y=y_test.values, mode='lines', name='Actual', line=dict(color='green', width=2)))
                fig2.add_trace(go.Scatter(x=fc_mean.index, y=fc_mean.values, mode='lines', name='Forecast', line=dict(color='crimson', width=2)))
                ci_cols = fc_ci.columns.tolist()
                ci_l = fc_ci[ci_cols[0]]; ci_u = fc_ci[ci_cols[1]]
                fig2.add_trace(go.Scatter(
                    x=ci_l.index.tolist() + ci_u.index.tolist()[::-1],
                    y=ci_l.values.tolist() + ci_u.values.tolist()[::-1],
                    fill='toself', fillcolor='rgba(255,0,0,0.1)', line=dict(color='rgba(0,0,0,0)'), name='95% CI'))
                fig2.update_layout(title="Forecast vs Actual", height=420, template='plotly_white')
                st.plotly_chart(fig2, use_container_width=True)

                st.markdown("### Evaluation")
                train_m = calc_metrics(y_train.values, fitted.values)
                test_m = calc_metrics(y_test.values, fc_mean.values)
                mdf = pd.DataFrame([
                    {**{'Set':'Train'}, **{k: f"{v:.4f}" if v is not None else '-' for k,v in train_m.items()}},
                    {**{'Set':'Test'}, **{k: f"{v:.4f}" if v is not None else '-' for k,v in test_m.items()}}
                ])
                st.dataframe(mdf, use_container_width=True, hide_index=True)

                st.markdown("### Residual Diagnostics")
                resid = result.resid.dropna()
                fig3 = make_subplots(rows=2, cols=2, subplot_titles=("Residuals", "Histogram", "ACF Residuals", "Q-Q Plot"))
                fig3.add_trace(go.Scatter(x=resid.index, y=resid.values, mode='lines', line=dict(width=0.8, color='gray')), row=1, col=1)
                fig3.add_hline(y=0, line=dict(color='red', dash='dash'), row=1, col=1)
                fig3.add_trace(go.Histogram(x=resid, nbinsx=30, marker_color='steelblue', opacity=0.7), row=1, col=2)
                max_lag_r = min(30, len(resid)//2 - 2)
                if max_lag_r > 1:
                    acf_r = acf(resid, nlags=max_lag_r, fft=True)
                    ci_r = 1.96 / np.sqrt(len(resid))
                    for j in range(len(acf_r)):
                        cr = 'red' if abs(acf_r[j]) > ci_r and j > 0 else 'steelblue'
                        fig3.add_trace(go.Bar(x=[j], y=[acf_r[j]], marker_color=cr, showlegend=False, width=0.3), row=2, col=1)
                    fig3.add_hline(y=ci_r, line=dict(color='blue', dash='dash'), row=2, col=1)
                    fig3.add_hline(y=-ci_r, line=dict(color='blue', dash='dash'), row=2, col=1)
                nn = len(resid)
                osm = stats.norm.ppf(np.linspace(1/(nn+1), nn/(nn+1), nn)); osr = np.sort(resid.values)
                fig3.add_trace(go.Scatter(x=osm, y=osr, mode='markers', marker=dict(size=3, opacity=0.6)), row=2, col=2)
                fig3.add_trace(go.Scatter(x=[osm.min(),osm.max()], y=[resid.mean()+resid.std()*osm.min(), resid.mean()+resid.std()*osm.max()],
                    mode='lines', line=dict(color='red', dash='dash')), row=2, col=2)
                fig3.update_layout(height=600, template='plotly_white', showlegend=False)
                st.plotly_chart(fig3, use_container_width=True)

                st.markdown("### Assumption Tests")
                lb = acorr_ljungbox(resid, lags=[10, 20], return_df=True)
                st.markdown("**Ljung-Box**")
                st.dataframe(lb.round(4), use_container_width=True)
                if (lb['lb_pvalue'] > 0.05).all():
                    st.success("No autocorrelation in residuals.")
                else:
                    st.warning("Autocorrelation detected!")

                jb_stat, jb_p = stats.jarque_bera(resid)
                st.markdown(f"**Jarque-Bera:** stat={jb_stat:.4f}, p={jb_p:.6f} → {'Normal' if jb_p>0.05 else 'Non-Normal'}")

                try:
                    arch_res = het_arch(resid, nlags=min(10, len(resid)//5))
                    st.markdown(f"**ARCH Effect:** LM={arch_res[0]:.4f}, p={arch_res[1]:.6f} → {'No ARCH' if arch_res[1]>0.05 else 'ARCH detected → Consider GARCH!'}")
                except:
                    pass

                st.session_state['arima_result'] = result
            except Exception as e:
                st.error(f"Error: {e}")

# ============================================================
# 6. SARIMA
# ============================================================
elif module == 'sarima':
    st.header("6. SARIMA(p,d,q)(P,D,Q,s)")
    transform = st.selectbox("Transformasi:", ['None', 'Log'], key='sar_tr')
    y_model, _ = apply_transformation(y_full, transform)
    if len(y_model) < 10:
        st.error("Data terlalu sedikit."); st.stop()

    split_pct = st.slider("Train %:", 50, 95, 80, key='sar_sp')
    n_train = int(len(y_model) * split_pct / 100)
    y_train = y_model.iloc[:n_train]; y_test = y_model.iloc[n_train:]
    st.info(f"Train: {len(y_train)} | Test: {len(y_test)}")

    c1,c2,c3 = st.columns(3)
    p = c1.number_input("p:", 0, 5, 1, key='sp'); d = c2.number_input("d:", 0, 2, 1, key='sd'); q = c3.number_input("q:", 0, 5, 1, key='sq')
    c1,c2,c3,c4 = st.columns(4)
    P = c1.number_input("P:", 0, 3, 1, key='sP'); D = c2.number_input("D:", 0, 2, 1, key='sD')
    Q = c3.number_input("Q:", 0, 3, 1, key='sQ'); s = c4.number_input("s:", 1, 365, 12, key='ss')
    trend_opt = st.selectbox("Trend:", ['n', 'c', 't', 'ct'], key='sar_trend')

    if st.button("Fit SARIMA", type="primary"):
        with st.spinner("Fitting..."):
            try:
                model = SARIMAX(y_train, order=(p,d,q), seasonal_order=(P,D,Q,s), trend=trend_opt,
                                enforce_stationarity=False, enforce_invertibility=False)
                result = model.fit(disp=False, maxiter=500)
                st.text(str(result.summary()))

                c1,c2,c3 = st.columns(3)
                c1.metric("AIC", f"{result.aic:.2f}"); c2.metric("BIC", f"{result.bic:.2f}"); c3.metric("HQIC", f"{result.hqic:.2f}")

                fc_obj = result.get_forecast(steps=len(y_test))
                fc_mean = fc_obj.predicted_mean
                fc_ci = fc_obj.conf_int(alpha=0.05)

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=y_train.index, y=y_train.values, mode='lines', name='Train'))
                fig.add_trace(go.Scatter(x=y_test.index, y=y_test.values, mode='lines', name='Actual', line=dict(color='green', width=2)))
                fig.add_trace(go.Scatter(x=fc_mean.index, y=fc_mean.values, mode='lines', name='Forecast', line=dict(color='crimson', width=2)))
                ci_cols = fc_ci.columns.tolist()
                ci_l = fc_ci[ci_cols[0]]; ci_u = fc_ci[ci_cols[1]]
                fig.add_trace(go.Scatter(
                    x=ci_l.index.tolist()+ci_u.index.tolist()[::-1],
                    y=ci_l.values.tolist()+ci_u.values.tolist()[::-1],
                    fill='toself', fillcolor='rgba(255,0,0,0.1)', line=dict(color='rgba(0,0,0,0)'), name='95% CI'))
                fig.update_layout(title=f"SARIMA({p},{d},{q})({P},{D},{Q},{s})", height=420, template='plotly_white')
                st.plotly_chart(fig, use_container_width=True)

                test_m = calc_metrics(y_test.values, fc_mean.values)
                st.dataframe(pd.DataFrame([{k: f"{v:.4f}" if v is not None else '-' for k,v in test_m.items()}]),
                             use_container_width=True, hide_index=True)

                resid = result.resid.dropna()
                fig2 = make_subplots(rows=1, cols=3, subplot_titles=("Residuals", "ACF Resid", "Q-Q"))
                fig2.add_trace(go.Scatter(y=resid.values, mode='lines', line=dict(width=0.8, color='gray')), row=1, col=1)
                max_lag_r = min(30, len(resid)//2 - 2)
                if max_lag_r > 1:
                    acf_r = acf(resid, nlags=max_lag_r, fft=True)
                    ci_v = 1.96/np.sqrt(len(resid))
                    for j in range(len(acf_r)):
                        cr = 'red' if abs(acf_r[j])>ci_v and j>0 else 'steelblue'
                        fig2.add_trace(go.Bar(x=[j], y=[acf_r[j]], marker_color=cr, showlegend=False, width=0.3), row=1, col=2)
                    fig2.add_hline(y=ci_v, line=dict(color='blue', dash='dash'), row=1, col=2)
                    fig2.add_hline(y=-ci_v, line=dict(color='blue', dash='dash'), row=1, col=2)
                nn = len(resid)
                osm = stats.norm.ppf(np.linspace(1/(nn+1), nn/(nn+1), nn))
                fig2.add_trace(go.Scatter(x=osm, y=np.sort(resid.values), mode='markers', marker=dict(size=3, opacity=0.6)), row=1, col=3)
                fig2.update_layout(height=350, template='plotly_white', showlegend=False)
                st.plotly_chart(fig2, use_container_width=True)

                lb = acorr_ljungbox(resid, lags=[10,20], return_df=True)
                jb_s, jb_p = stats.jarque_bera(resid)
                st.markdown(f"**Ljung-Box p(10):** {lb['lb_pvalue'].iloc[0]:.4f} | **JB p:** {jb_p:.4f}")
                st.session_state['sarima_result'] = result
            except Exception as e:
                st.error(f"Error: {e}")

# ============================================================
# 7. ARIMAX
# ============================================================
elif module == 'arimax':
    st.header("7. ARIMAX — ARIMA with Exogenous Variables")
    exog_cols = [c for c in raw_df.columns if c != target_col]
    if not exog_cols:
        st.warning("Tidak ada variabel eksogen. Pilih dataset 'Sales + Exog'."); st.stop()

    sel_exog = st.multiselect("Variabel Exog:", exog_cols, default=exog_cols[:min(2, len(exog_cols))])
    if not sel_exog:
        st.warning("Pilih min 1 variabel."); st.stop()

    X = raw_df[sel_exog].dropna()
    common_idx = y_full.index.intersection(X.index)
    y_model = y_full.loc[common_idx]
    X = X.loc[common_idx]

    split_pct = st.slider("Train %:", 50, 95, 80, key='ax_sp')
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
        with st.spinner("Fitting..."):
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
                ci_cols = fc_ci.columns.tolist()
                ci_l = fc_ci[ci_cols[0]]; ci_u = fc_ci[ci_cols[1]]
                fig.add_trace(go.Scatter(
                    x=ci_l.index.tolist()+ci_u.index.tolist()[::-1],
                    y=ci_l.values.tolist()+ci_u.values.tolist()[::-1],
                    fill='toself', fillcolor='rgba(255,0,0,0.1)', line=dict(color='rgba(0,0,0,0)'), name='95% CI'))
                fig.update_layout(title="ARIMAX Forecast", height=420, template='plotly_white')
                st.plotly_chart(fig, use_container_width=True)

                test_m = calc_metrics(y_test.values, fc_mean.values)
                st.dataframe(pd.DataFrame([{k: f"{v:.4f}" if v is not None else '-' for k,v in test_m.items()}]),
                             use_container_width=True, hide_index=True)
            except Exception as e:
                st.error(f"Error: {e}")

# ============================================================
# 8. ARIMA-GARCH
# ============================================================
elif module == 'garch':
    st.header("8. ARIMA-GARCH Modeling")
    if not HAS_ARCH:
        st.error("Library `arch` belum terinstall. Jalankan: `pip install arch`"); st.stop()

    use_returns = st.checkbox("Gunakan log-returns?", True)
    if use_returns:
        y_pos = y_full[y_full > 0]
        y_model = (np.log(y_pos).diff().dropna() * 100)
        st.info("Using log-returns (x100)")
    else:
        y_model = y_full.copy()

    if len(y_model) < 30:
        st.error("Data terlalu sedikit."); st.stop()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_model.index, y=y_model.values, mode='lines', line=dict(width=0.8)))
    fig.update_layout(title="Series for GARCH", height=300, template='plotly_white')
    st.plotly_chart(fig, use_container_width=True)

    split_pct = st.slider("Train %:", 50, 95, 80, key='g_sp')
    n_train = int(len(y_model) * split_pct / 100)
    y_train = y_model.iloc[:n_train]; y_test = y_model.iloc[n_train:]

    c1,c2,c3 = st.columns(3)
    ar = c1.number_input("AR lags:", 0, 5, 1, key='gar'); ma = c3.number_input("MA lags:", 0, 5, 1, key='gma')
    c1,c2 = st.columns(2)
    gp = c1.number_input("GARCH p:", 1, 5, 1, key='gp'); gq = c2.number_input("GARCH q:", 1, 5, 1, key='gq')
    vol_model = st.selectbox("Volatility Model:", ['GARCH', 'EGARCH', 'GJR-GARCH'])
    dist = st.selectbox("Distribution:", ['normal', 't', 'skewt', 'ged'])

    if st.button("Fit GARCH", type="primary"):
        with st.spinner("Fitting..."):
            try:
                vol_map = {'GARCH': 'Garch', 'EGARCH': 'EGARCH', 'GJR-GARCH': 'GARCH'}
                o_val = gp if vol_model == 'GJR-GARCH' else 0
                am = arch_model(y_train, mean='ARX', lags=ar, vol=vol_map[vol_model],
                                p=gp, q=gq, o=o_val, dist=dist)
                res = am.fit(disp='off')
                st.text(str(res.summary()))

                fig = make_subplots(rows=2, cols=1, subplot_titles=("Returns + Cond. Volatility", "Std Residuals"), shared_xaxes=True)
                fig.add_trace(go.Scatter(x=y_train.index, y=y_train.values, mode='lines', line=dict(width=0.5, color='gray'), name='Returns'), row=1, col=1)
                fig.add_trace(go.Scatter(x=y_train.index, y=res.conditional_volatility, mode='lines', line=dict(color='crimson', width=1.5), name='Cond Vol'), row=1, col=1)
                fig.add_trace(go.Scatter(x=y_train.index, y=-res.conditional_volatility, mode='lines', line=dict(color='crimson', width=1.5), showlegend=False), row=1, col=1)
                std_resid = res.resid / res.conditional_volatility
                fig.add_trace(go.Scatter(x=y_train.index, y=std_resid, mode='lines', line=dict(width=0.5, color='steelblue')), row=2, col=1)
                fig.update_layout(height=500, template='plotly_white'); st.plotly_chart(fig, use_container_width=True)

                fc = res.forecast(horizon=min(len(y_test), 100))
                fc_mean = fc.mean.iloc[-1].values
                fc_var = fc.variance.iloc[-1].values
                fc_vol = np.sqrt(fc_var)
                h = min(len(y_test), len(fc_mean))

                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=y_test.index[:h], y=y_test.values[:h], mode='lines', name='Actual'))
                fig2.add_trace(go.Scatter(x=y_test.index[:h], y=fc_mean[:h], mode='lines', name='Forecast', line=dict(color='crimson')))
                fig2.add_trace(go.Scatter(
                    x=y_test.index[:h].tolist()+y_test.index[:h].tolist()[::-1],
                    y=(fc_mean[:h]+1.96*fc_vol[:h]).tolist()+(fc_mean[:h]-1.96*fc_vol[:h]).tolist()[::-1],
                    fill='toself', fillcolor='rgba(255,0,0,0.1)', line=dict(color='rgba(0,0,0,0)'), name='95% CI'))
                fig2.update_layout(title="GARCH Forecast", height=400, template='plotly_white')
                st.plotly_chart(fig2, use_container_width=True)

                test_m = calc_metrics(y_test.values[:h], fc_mean[:h])
                st.dataframe(pd.DataFrame([{k: f"{v:.4f}" if v is not None else '-' for k,v in test_m.items()}]),
                             use_container_width=True, hide_index=True)
            except Exception as e:
                st.error(f"Error: {e}")

# ============================================================
# 9. AUTO ARIMA (GRID SEARCH) — FIXED
# ============================================================
elif module == 'auto_arima':
    st.header("9. Auto ARIMA — Grid Search")

    transform = st.selectbox("Transformasi:", ['None', 'Log'], key='auto_tr')
    if transform == 'Log' and (y_full > 0).all():
        y_model = np.log(y_full)
    else:
        y_model = y_full.copy()

    split_pct = st.slider("Train %:", 50, 95, 80, key='auto_sp')
    n_train = int(len(y_model) * split_pct / 100)
    y_train = y_model.iloc[:n_train]; y_test = y_model.iloc[n_train:]
    st.info(f"Train: {len(y_train)} | Test: {len(y_test)}")

    st.markdown("### Search Range")
    c1,c2,c3 = st.columns(3)
    max_p = c1.number_input("Max p:", 0, 5, 2, key='ap')
    max_d = c2.number_input("Max d:", 0, 2, 1, key='ad')
    max_q = c3.number_input("Max q:", 0, 5, 2, key='aq')
    seasonal = st.checkbox("Include Seasonal?", True, key='auto_seas')
    s_period = 12
    max_P = 0; max_D = 0; max_Q = 0
    if seasonal:
        c1,c2,c3,c4 = st.columns(4)
        max_P = c1.number_input("Max P:", 0, 2, 1, key='aP')
        max_D = c2.number_input("Max D:", 0, 1, 1, key='aD')
        max_Q = c3.number_input("Max Q:", 0, 2, 1, key='aQ')
        s_period = c4.number_input("s:", 1, 365, 12, key='as_')
    criterion = st.selectbox("Kriteria:", ['AIC', 'BIC'], key='auto_crit')

    if st.button("Run Grid Search", type="primary"):
        with st.spinner("Searching..."):
            results = []
            # Build combos
            if seasonal:
                combos = list(itertools.product(
                    range(max_p+1), range(max_d+1), range(max_q+1),
                    range(max_P+1), range(max_D+1), range(max_Q+1)
                ))
            else:
                combos = list(itertools.product(
                    range(max_p+1), range(max_d+1), range(max_q+1)
                ))

            total = len(combos)
            st.write(f"Total kombinasi: {total}")
            progress = st.progress(0)

            for idx, combo in enumerate(combos):
                progress.progress(min((idx+1)/total, 1.0))
                try:
                    if seasonal:
                        pp, dd, qq, PP, DD, QQ = combo
                        if pp == 0 and qq == 0 and PP == 0 and QQ == 0:
                            continue
                        model = SARIMAX(y_train, order=(pp,dd,qq),
                                        seasonal_order=(PP,DD,QQ,s_period),
                                        enforce_stationarity=False,
                                        enforce_invertibility=False)
                    else:
                        pp, dd, qq = combo
                        if pp == 0 and qq == 0:
                            continue
                        model = ARIMA(y_train, order=(pp,dd,qq))

                    res = model.fit(disp=False, maxiter=200)
                    fc = res.get_forecast(steps=len(y_test))
                    fc_mean = fc.predicted_mean
                    test_m = calc_metrics(y_test.values, fc_mean.values)

                    row = {
                        'p': pp, 'd': dd, 'q': qq,
                        'AIC': round(res.aic, 2),
                        'BIC': round(res.bic, 2),
                        'RMSE': round(test_m.get('RMSE', 999), 4),
                        'MAE': round(test_m.get('MAE', 999), 4),
                        'MAPE': round(test_m['MAPE'], 2) if test_m.get('MAPE') is not None else None,
                    }
                    if seasonal:
                        row['P'] = PP; row['D'] = DD; row['Q'] = QQ; row['s'] = s_period
                    results.append(row)
                except:
                    pass

            progress.empty()

            if results:
                rdf = pd.DataFrame(results)
                rdf = rdf.sort_values(criterion).reset_index(drop=True)
                st.markdown(f"### Top 20 (sorted by {criterion})")
                st.dataframe(rdf.head(20), use_container_width=True, hide_index=True)

                # Fit best model
                best = rdf.iloc[0]
                best_p = int(best['p']); best_d = int(best['d']); best_q = int(best['q'])
                st.markdown("### Best Model Forecast")

                try:
                    if seasonal:
                        best_P = int(best['P']); best_D = int(best['D']); best_Q = int(best['Q'])
                        best_s = int(best['s'])
                        bm = SARIMAX(y_train, order=(best_p, best_d, best_q),
                                     seasonal_order=(best_P, best_D, best_Q, best_s),
                                     enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
                        label = f"SARIMA({best_p},{best_d},{best_q})({best_P},{best_D},{best_Q},{best_s})"
                    else:
                        bm = ARIMA(y_train, order=(best_p, best_d, best_q)).fit()
                        label = f"ARIMA({best_p},{best_d},{best_q})"

                    st.success(f"**{label}** | AIC={best['AIC']} | BIC={best['BIC']} | RMSE={best['RMSE']}")

                    fc = bm.get_forecast(steps=len(y_test))
                    fc_mean = fc.predicted_mean; fc_ci = fc.conf_int(alpha=0.05)

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=y_train.index, y=y_train.values, mode='lines', name='Train'))
                    fig.add_trace(go.Scatter(x=y_test.index, y=y_test.values, mode='lines', name='Actual', line=dict(color='green', width=2)))
                    fig.add_trace(go.Scatter(x=fc_mean.index, y=fc_mean.values, mode='lines', name='Forecast', line=dict(color='crimson', width=2)))
                    ci_cols = fc_ci.columns.tolist()
                    ci_l = fc_ci[ci_cols[0]]; ci_u = fc_ci[ci_cols[1]]
                    fig.add_trace(go.Scatter(
                        x=ci_l.index.tolist()+ci_u.index.tolist()[::-1],
                        y=ci_l.values.tolist()+ci_u.values.tolist()[::-1],
                        fill='toself', fillcolor='rgba(255,0,0,0.1)', line=dict(color='rgba(0,0,0,0)'), name='95% CI'))
                    fig.update_layout(title=f"Best: {label}", height=420, template='plotly_white')
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error fitting best model: {e}")
            else:
                st.error("Tidak ada model valid ditemukan.")

# ============================================================
# 10. MODEL COMPARISON
# ============================================================
elif module == 'comparison':
    st.header("10. Model Comparison")
    transform = st.selectbox("Transformasi:", ['None', 'Log'], key='comp_tr')
    if transform == 'Log' and (y_full > 0).all():
        y_model = np.log(y_full)
    else:
        y_model = y_full.copy()

    split_pct = st.slider("Train %:", 50, 95, 80, key='comp_sp')
    n_train = int(len(y_model) * split_pct / 100)
    y_train = y_model.iloc[:n_train]; y_test = y_model.iloc[n_train:]

    models_to_compare = st.multiselect("Models:", [
        'ARIMA(1,1,1)', 'ARIMA(2,1,2)', 'ARIMA(1,1,0)', 'ARIMA(0,1,1)',
        'SARIMA(1,1,1)(1,1,1,12)', 'SARIMA(0,1,1)(0,1,1,12)', 'SARIMA(1,0,1)(1,1,0,12)',
        'Holt-Winters (Add)', 'Holt-Winters (Mul)', 'Naive', 'Drift'
    ], default=['ARIMA(1,1,1)', 'SARIMA(1,1,1)(1,1,1,12)', 'Naive'])

    if st.button("Compare", type="primary") and models_to_compare:
        results = []; forecasts = {}
        for mname in models_to_compare:
            try:
                fc = None; aic = None; bic = None
                if mname.startswith('ARIMA('):
                    order_str = mname.replace('ARIMA', '')
                    order = tuple(int(x) for x in order_str.strip('()').split(','))
                    m = ARIMA(y_train, order=order).fit()
                    fc = m.get_forecast(len(y_test)).predicted_mean; aic = m.aic; bic = m.bic
                elif mname.startswith('SARIMA('):
                    # Parse SARIMA(p,d,q)(P,D,Q,s)
                    inner = mname.replace('SARIMA', '')
                    # Split by ')(' to get two groups
                    parts = inner.split(')(')
                    order_part = parts[0].strip('(')
                    seasonal_part = parts[1].strip(')')
                    order = tuple(int(x) for x in order_part.split(','))
                    seasonal = tuple(int(x) for x in seasonal_part.split(','))
                    m = SARIMAX(y_train, order=order, seasonal_order=seasonal,
                                enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
                    fc = m.get_forecast(len(y_test)).predicted_mean; aic = m.aic; bic = m.bic
                elif 'Holt' in mname:
                    st_type = 'add' if 'Add' in mname else 'mul'
                    try:
                        m = ExponentialSmoothing(y_train, trend='add', seasonal=st_type, seasonal_periods=12).fit()
                    except:
                        m = ExponentialSmoothing(y_train, trend='add', seasonal='add', seasonal_periods=12).fit()
                    fc = m.forecast(len(y_test)); aic = m.aic; bic = m.bic
                elif mname == 'Naive':
                    fc = pd.Series(np.full(len(y_test), y_train.iloc[-1]), index=y_test.index)
                elif mname == 'Drift':
                    n_tr = len(y_train)
                    drift_val = (y_train.iloc[-1] - y_train.iloc[0]) / (n_tr - 1) if n_tr > 1 else 0
                    fc = pd.Series([y_train.iloc[-1] + drift_val*(i+1) for i in range(len(y_test))], index=y_test.index)

                if fc is not None:
                    metrics = calc_metrics(y_test.values, fc.values)
                    results.append({
                        'Model': mname,
                        'AIC': f"{aic:.2f}" if aic else '-',
                        'BIC': f"{bic:.2f}" if bic else '-',
                        'RMSE': f"{metrics.get('RMSE',0):.4f}",
                        'MAE': f"{metrics.get('MAE',0):.4f}",
                        'MAPE': f"{metrics['MAPE']:.2f}%" if metrics.get('MAPE') is not None else '-',
                        'R2': f"{metrics['R2']:.4f}" if metrics.get('R2') is not None else '-'
                    })
                    forecasts[mname] = fc
            except Exception as e:
                results.append({'Model': mname, 'AIC':'-','BIC':'-','RMSE':str(e)[:40],'MAE':'-','MAPE':'-','R2':'-'})

        if results:
            st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)

        if forecasts:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=y_train.index, y=y_train.values, mode='lines', name='Train', line=dict(color='gray', width=1)))
            fig.add_trace(go.Scatter(x=y_test.index, y=y_test.values, mode='lines', name='Actual', line=dict(color='black', width=2)))
            colors = px.colors.qualitative.Set1
            for i, (mn, fc) in enumerate(forecasts.items()):
                fig.add_trace(go.Scatter(x=fc.index, y=fc.values, mode='lines', name=mn,
                                         line=dict(color=colors[i % len(colors)], width=1.5)))
            fig.update_layout(title="Model Comparison", height=450, template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)

# ============================================================
# 11. FORECAST & CI/PI
# ============================================================
elif module == 'forecast':
    st.header("11. Forecasting with CI & PI")
    transform = st.selectbox("Transformasi:", ['None', 'Log'], key='fc_tr')
    if transform == 'Log' and (y_full > 0).all():
        y_model = np.log(y_full)
    else:
        y_model = y_full.copy()

    model_type = st.selectbox("Model:", ['ARIMA', 'SARIMA'])
    c1,c2,c3 = st.columns(3)
    p = c1.number_input("p:", 0, 5, 1, key='fcp'); d = c2.number_input("d:", 0, 2, 1, key='fcd'); q = c3.number_input("q:", 0, 5, 1, key='fcq')
    if model_type == 'SARIMA':
        c1,c2,c3,c4 = st.columns(4)
        P = c1.number_input("P:", 0, 3, 1, key='fcP'); D = c2.number_input("D:", 0, 2, 1, key='fcD')
        Q = c3.number_input("Q:", 0, 3, 1, key='fcQ'); s = c4.number_input("s:", 1, 365, 12, key='fcs')

    h = st.slider("Forecast horizon:", 1, 120, 24, key='fc_h')
    alpha = st.slider("Alpha:", 0.01, 0.20, 0.05, 0.01, key='fc_alpha')
    ci_label = f"{(1-alpha)*100:.0f}%"

    if st.button("Generate Forecast", type="primary"):
        with st.spinner("Fitting & forecasting..."):
            try:
                if model_type == 'ARIMA':
                    model = ARIMA(y_model, order=(p,d,q)).fit()
                else:
                    model = SARIMAX(y_model, order=(p,d,q), seasonal_order=(P,D,Q,s),
                                    enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)

                c1,c2,c3 = st.columns(3)
                c1.metric("AIC", f"{model.aic:.2f}"); c2.metric("BIC", f"{model.bic:.2f}"); c3.metric("Log-Lik", f"{model.llf:.2f}")

                fc_obj = model.get_forecast(steps=h)
                fc_mean = fc_obj.predicted_mean
                fc_ci = fc_obj.conf_int(alpha=alpha)
                ci_cols = fc_ci.columns.tolist()
                ci_l = fc_ci[ci_cols[0]]; ci_u = fc_ci[ci_cols[1]]

                # PI (wider than CI)
                resid_std = model.resid.std()
                z = stats.norm.ppf(1 - alpha/2)
                ci_half_width = (ci_u.values - fc_mean.values)
                pi_half_width = np.sqrt(ci_half_width**2 + resid_std**2)
                pi_l = fc_mean.values - z * np.sqrt(ci_half_width**2 / z**2 + resid_std**2)
                pi_u = fc_mean.values + z * np.sqrt(ci_half_width**2 / z**2 + resid_std**2)

                # Inverse transform
                if transform == 'Log':
                    fc_mean_plot = np.exp(fc_mean)
                    ci_l_plot = np.exp(ci_l); ci_u_plot = np.exp(ci_u)
                    pi_l_plot = np.exp(pi_l); pi_u_plot = np.exp(pi_u)
                    y_plot = np.exp(y_model)
                else:
                    fc_mean_plot = fc_mean
                    ci_l_plot = ci_l; ci_u_plot = ci_u
                    pi_l_plot = pd.Series(pi_l, index=fc_mean.index)
                    pi_u_plot = pd.Series(pi_u, index=fc_mean.index)
                    y_plot = y_model

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=y_plot.index, y=y_plot.values, mode='lines', name='Historical', line=dict(color='steelblue', width=1.5)))
                fig.add_trace(go.Scatter(x=fc_mean_plot.index, y=fc_mean_plot.values, mode='lines', name='Forecast', line=dict(color='crimson', width=2)))

                # PI band (outer)
                pi_l_vals = pi_l_plot.values if hasattr(pi_l_plot, 'values') else pi_l_plot
                pi_u_vals = pi_u_plot.values if hasattr(pi_u_plot, 'values') else pi_u_plot
                fig.add_trace(go.Scatter(
                    x=fc_mean_plot.index.tolist() + fc_mean_plot.index.tolist()[::-1],
                    y=list(pi_l_vals) + list(pi_u_vals)[::-1],
                    fill='toself', fillcolor='rgba(255,165,0,0.08)', line=dict(color='rgba(0,0,0,0)'),
                    name=f'{ci_label} PI'))

                # CI band (inner)
                ci_l_vals = ci_l_plot.values if hasattr(ci_l_plot, 'values') else ci_l_plot
                ci_u_vals = ci_u_plot.values if hasattr(ci_u_plot, 'values') else ci_u_plot
                fig.add_trace(go.Scatter(
                    x=fc_mean_plot.index.tolist() + fc_mean_plot.index.tolist()[::-1],
                    y=list(ci_l_vals) + list(ci_u_vals)[::-1],
                    fill='toself', fillcolor='rgba(255,0,0,0.15)', line=dict(color='rgba(0,0,0,0)'),
                    name=f'{ci_label} CI'))

                fig.update_layout(title=f"Forecast — {h} periods ({ci_label} CI & PI)", height=500, template='plotly_white')
                st.plotly_chart(fig, use_container_width=True)

                # Table
                fc_df = pd.DataFrame({
                    'Date': fc_mean_plot.index,
                    'Forecast': np.round(fc_mean_plot.values if hasattr(fc_mean_plot,'values') else fc_mean_plot, 4),
                    f'CI Lower': np.round(ci_l_vals, 4),
                    f'CI Upper': np.round(ci_u_vals, 4),
                    f'PI Lower': np.round(pi_l_vals, 4),
                    f'PI Upper': np.round(pi_u_vals, 4),
                })
                st.dataframe(fc_df, use_container_width=True, hide_index=True)
                csv = fc_df.to_csv(index=False)
                st.download_button("Download Forecast CSV", csv, "forecast.csv", "text/csv")

                # Diagnostics
                st.markdown("### Model Diagnostics")
                resid = model.resid.dropna()
                lb = acorr_ljungbox(resid, lags=[10,20], return_df=True)
                jb_s, jb_p = stats.jarque_bera(resid)
                st.markdown(f"**Ljung-Box p(10):** {lb['lb_pvalue'].iloc[0]:.4f} | **Ljung-Box p(20):** {lb['lb_pvalue'].iloc[1]:.4f} | **JB p:** {jb_p:.4f}")
                if lb['lb_pvalue'].iloc[0] > 0.05 and jb_p > 0.05:
                    st.success("Residuals OK: No autocorrelation + Normal.")
                elif lb['lb_pvalue'].iloc[0] > 0.05:
                    st.warning("No autocorrelation, but Non-Normal residuals.")
                else:
                    st.error("Autocorrelation in residuals — consider adjusting model.")
            except Exception as e:
                st.error(f"Error: {e}")

# FOOTER
st.divider()
st.markdown("<div style='text-align:center;color:gray;font-size:0.85rem'><b>ARIMA Pro</b> | 11 Modul | ARIMA · SARIMA · ARIMAX · GARCH | 2026</div>", unsafe_allow_html=True)
