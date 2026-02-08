import streamlit as st
st.set_page_config(page_title="ARIMA Pro", layout="wide", page_icon="📈")

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings, io, itertools, os, importlib
warnings.filterwarnings('ignore')

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================
# AUTO-INSTALL dari requirements-arima.txt
# ============================================================
def _auto_install():
    """Baca requirements-arima.txt, install package yang belum ada."""
    req_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "requirements-arima.txt")
    if not os.path.exists(req_file):
        return
    with open(req_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            pkg_name = line.split(">=")[0].split("==")[0].split("<")[0].strip()
            # Mapping: nama package di pip != nama import
            import_map = {"scikit-learn":"sklearn","Pillow":"PIL","opencv-python":"cv2"}
            mod_name = import_map.get(pkg_name, pkg_name)
            try:
                importlib.import_module(mod_name)
            except ImportError:
                os.system(f'pip install "{line}" -q')

_auto_install()

HAS_ARCH = False
try:
    from arch import arch_model
    HAS_ARCH = True
except ImportError:
    pass

st.title("📈 ARIMA Pro — Analisis Time Series Lengkap")
st.caption("EDA · Decomposition · Stationarity · ARIMA/SARIMA · ARIMAX · GARCH · Forecast · Evaluation")

# ============================================================
# DEMO DATA
# ============================================================
@st.cache_data
def load_demo_airline():
    np.random.seed(42); n=144; t=np.arange(n)
    y=100+2.5*t+30*np.sin(2*np.pi*t/12)+np.random.normal(0,10,n)
    return pd.DataFrame({'Date':pd.date_range('1949-01-01',periods=n,freq='MS'),'Passengers':np.round(y,1)})

@st.cache_data
def load_demo_stock():
    np.random.seed(123); n=500
    ret=np.random.normal(0.0005,0.02,n); vol=np.zeros(n); vol[0]=0.02
    for i in range(1,n):
        vol[i]=np.sqrt(0.00001+0.1*ret[i-1]**2+0.85*vol[i-1]**2)
        ret[i]=np.random.normal(0.0005,vol[i])
    price=100*np.exp(np.cumsum(ret))
    return pd.DataFrame({'Date':pd.date_range('2023-01-02',periods=n,freq='B'),
                         'Price':np.round(price,2),'Volume':np.random.randint(1000000,5000000,n)})

@st.cache_data
def load_demo_sales():
    np.random.seed(99); n=120; t=np.arange(n)
    promo=np.zeros(n); promo[np.random.choice(n,20,replace=False)]=np.random.uniform(50,150,20)
    y=500+3*t+80*np.sin(2*np.pi*t/12)+40*np.cos(2*np.pi*t/6)+promo+np.random.normal(0,20,n)
    return pd.DataFrame({'Date':pd.date_range('2014-01-01',periods=n,freq='MS'),
        'Sales':np.round(y,1),'Promo':np.round(promo,1),
        'Temperature':np.round(25+10*np.sin(2*np.pi*t/12)+np.random.normal(0,2,n),1)})

@st.cache_data
def load_demo_mult():
    np.random.seed(55); n=144; t=np.arange(n)
    y=(100+1.5*t)*(1+0.3*np.sin(2*np.pi*t/12))*np.random.normal(1,0.05,n)
    return pd.DataFrame({'Date':pd.date_range('1949-01-01',periods=n,freq='MS'),'Value':np.round(y,2)})

# ============================================================
# HELPERS
# ============================================================
def adf_test(series):
    r=adfuller(series.dropna(),autolag='AIC')
    return {'Statistic':round(r[0],4),'p-value':round(r[1],6),'Lags':r[2],
            'CV 1%':round(r[4]['1%'],4),'CV 5%':round(r[4]['5%'],4),
            'Stationary':'Yes' if r[1]<0.05 else 'No'}

def kpss_test(series):
    try:
        r=kpss(series.dropna(),regression='c',nlags='auto')
        return {'Statistic':round(r[0],4),'p-value':round(r[1],6),
                'CV 5%':round(r[3]['5%'],4),'Stationary':'Yes' if r[1]>0.05 else 'No'}
    except: return {'Statistic':None,'p-value':None,'Stationary':'Unknown'}

def calc_metrics(actual, predicted):
    a=np.array(actual,dtype=float); p=np.array(predicted,dtype=float)
    ml=min(len(a),len(p)); a=a[:ml]; p=p[:ml]
    mask=np.isfinite(a)&np.isfinite(p); a=a[mask]; p=p[mask]
    if len(a)==0: return {k:None for k in ['ME','MAE','MSE','RMSE','MAPE','MdAPE','MASE','R2','TheilU']}
    e=a-p; ae=np.abs(e); se=e**2
    nz=a!=0; ape=np.abs(e[nz]/a[nz])*100 if nz.sum()>0 else np.array([])
    naive_e=np.abs(np.diff(a))
    mase=np.mean(ae)/np.mean(naive_e) if len(naive_e)>0 and np.mean(naive_e)>0 else None
    num=np.sqrt(np.mean(se)); den=np.sqrt(np.mean(a**2))+np.sqrt(np.mean(p**2))
    theil_u=num/den if den>0 else None; var_a=np.var(a)
    return {'ME':np.mean(e),'MAE':np.mean(ae),'MSE':np.mean(se),'RMSE':np.sqrt(np.mean(se)),
        'MAPE':np.mean(ape) if len(ape)>0 else None,'MdAPE':np.median(ape) if len(ape)>0 else None,
        'MASE':mase,'R2':1-np.sum(se)/np.sum((a-a.mean())**2) if var_a>0 else None,'TheilU':theil_u}

def display_three_metrics(y_train, fitted_train, y_test, forecast_test):
    in_m=calc_metrics(y_train,fitted_train)
    out_m=calc_metrics(y_test,forecast_test)
    all_a=np.concatenate([np.array(y_train,dtype=float),np.array(y_test,dtype=float)])
    all_p=np.concatenate([np.array(fitted_train,dtype=float),np.array(forecast_test,dtype=float)])
    ov_m=calc_metrics(all_a,all_p)
    def fmt(v):
        if v is None: return '-'
        return f"{v:.4f}"
    rows=[]
    for key in ['ME','MAE','MSE','RMSE','MAPE','MdAPE','MASE','R2','TheilU']:
        label=key
        if key in ('MAPE','MdAPE'): label+=' (%)'
        rows.append({'Metric':label,'In-Sample (Train)':fmt(in_m.get(key)),
                     'Out-Sample (Test)':fmt(out_m.get(key)),'Overall':fmt(ov_m.get(key))})
    st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)

def write_model_notation(model_type, order, seasonal_order=None, exog_names=None, transform=None):
    p,d,q=order
    tf={'Log':'\\ln','Sqrt':'\\sqrt{}','Box-Cox':'\\text{BoxCox}'}.get(transform,'') if transform and transform!='None' else ''
    lhs=f"{tf}(Y_t)" if tf else "Y_t"
    diff_str={0:'',1:'(1-B)',2:'(1-B)^2'}.get(d,f"(1-B)^{d}")
    seas_diff_str=""
    if seasonal_order:
        P,D,Q,s=seasonal_order
        if D==1: seas_diff_str=f"(1-B^{{{s}}})"
        elif D>1: seas_diff_str=f"(1-B^{{{s}}})^{D}"
    ar_str=""
    if p>0:
        ar_terms=" - ".join([f"\\phi_{{{i}}}B^{{{i}}}" if i>1 else "\\phi_1 B" for i in range(1,p+1)])
        ar_str=f"(1 - {ar_terms})"
    ma_str=""
    if q>0:
        ma_terms=" + ".join([f"\\theta_{{{i}}}B^{{{i}}}" if i>1 else "\\theta_1 B" for i in range(1,q+1)])
        ma_str=f"(1 + {ma_terms})"
    sar_str=""
    if seasonal_order and seasonal_order[0]>0:
        Pv=seasonal_order[0]; sv=seasonal_order[3]
        sar_terms=" - ".join([f"\\Phi_{{{i}}}B^{{{i*sv}}}" for i in range(1,Pv+1)])
        sar_str=f"(1 - {sar_terms})"
    sma_str=""
    if seasonal_order and seasonal_order[2]>0:
        Qv=seasonal_order[2]; sv=seasonal_order[3]
        sma_terms=" + ".join([f"\\Theta_{{{i}}}B^{{{i*sv}}}" for i in range(1,Qv+1)])
        sma_str=f"(1 + {sma_terms})"
    exog_str=""
    if exog_names:
        exog_str=" + "+" + ".join([f"\\beta_{{{i+1}}} X_{{{i+1},t}}" for i in range(len(exog_names))])
    lhs_full=f"{ar_str}{sar_str}{diff_str}{seas_diff_str} {lhs}"
    rhs_full=f"{ma_str}{sma_str} \\varepsilon_t{exog_str}"
    equation=f"{lhs_full} = {rhs_full}"
    if seasonal_order:
        label=f"SARIMA({p},{d},{q})({seasonal_order[0]},{seasonal_order[1]},{seasonal_order[2]}){{{seasonal_order[3]}}}"
    else: label=f"ARIMA({p},{d},{q})"
    if exog_names: label+=f" + Exog({', '.join(exog_names)})"
    if transform and transform!='None': label=f"{transform} + {label}"
    return label, equation

def show_model_notation(label, equation):
    st.markdown(f"### Model: **{label}**")
    st.latex(equation)

def show_param_table(result):
    params=result.params; se=result.bse; zv=result.zvalues; pv=result.pvalues
    ci=result.conf_int(alpha=0.05); rows=[]
    for idx_num,name in enumerate(params.index):
        rows.append({'Parameter':name,'Estimate':f"{params[name]:.6f}",
            'Std Error':f"{se[name]:.6f}" if name in se.index else '-',
            'z-value':f"{zv[name]:.4f}" if name in zv.index else '-',
            'p-value':f"{pv[name]:.6f}" if name in pv.index else '-',
            'CI Lower':f"{ci.iloc[idx_num,0]:.6f}",'CI Upper':f"{ci.iloc[idx_num,1]:.6f}",
            'Signif':'***' if pv.get(name,1)<0.001 else ('**' if pv.get(name,1)<0.01 else ('*' if pv.get(name,1)<0.05 else ''))})
    st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)

def plot_acf_pacf(series, lags=40, title=""):
    sc=series.dropna(); ml=min(lags,len(sc)//2-2)
    if ml<1: ml=1
    a_v=acf(sc,nlags=ml,fft=True); p_v=pacf(sc,nlags=ml); ci=1.96/np.sqrt(len(sc))
    fig=make_subplots(rows=1,cols=2,subplot_titles=(f"ACF {title}",f"PACF {title}"))
    for i,vals in enumerate([a_v,p_v]):
        for j in range(len(vals)):
            c='red' if abs(vals[j])>ci and j>0 else 'steelblue'
            fig.add_trace(go.Bar(x=[j],y=[vals[j]],marker_color=c,showlegend=False,width=0.3),row=1,col=i+1)
        fig.add_hline(y=ci,line=dict(color='blue',dash='dash',width=1),row=1,col=i+1)
        fig.add_hline(y=-ci,line=dict(color='blue',dash='dash',width=1),row=1,col=i+1)
    fig.update_layout(height=350,template='plotly_white',showlegend=False); return fig

def apply_transformation(series, method):
    if method=='None': return series.copy(),None
    elif method=='Log':
        s=series.copy(); s[s<=0]=np.nan; return np.log(s).dropna(),None
    elif method=='Sqrt':
        s=series.copy(); s[s<0]=np.nan; return np.sqrt(s).dropna(),None
    elif method=='Box-Cox':
        from scipy.stats import boxcox as bc
        s=series[series>0]; t,lam=bc(s.values); return pd.Series(t,index=s.index),lam
    elif method=='Diff(1)': return series.diff().dropna(),None
    elif method=='Diff(2)': return series.diff().diff().dropna(),None
    elif method=='Seasonal Diff': return series.diff(12).dropna(),None
    elif method=='Log + Diff(1)':
        s=series.copy(); s[s<=0]=np.nan; return np.log(s).diff().dropna(),None
    return series.copy(),None

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.header("Menu")
module=st.sidebar.selectbox("Modul:",[
    'eda','decomposition','stationarity','acf_pacf',
    'arima','sarima','arimax','garch',
    'auto_arima','comparison','forecast'
],format_func=lambda x:{
    'eda':'1. EDA','decomposition':'2. Decomposition','stationarity':'3. Stationarity Tests',
    'acf_pacf':'4. ACF & PACF','arima':'5. ARIMA','sarima':'6. SARIMA',
    'arimax':'7. ARIMAX','garch':'8. ARIMA-GARCH','auto_arima':'9. Auto ARIMA',
    'comparison':'10. Model Comparison','forecast':'11. Forecast & CI/PI'}[x])

st.sidebar.markdown("---")
st.sidebar.subheader("Data")
data_src=st.sidebar.selectbox("Sumber:",['Demo: Airline','Demo: Stock (GARCH)',
    'Demo: Sales + Exog','Demo: Multiplicative','Upload CSV'])
if data_src=='Demo: Airline':
    raw_df=load_demo_airline(); date_col='Date'; target_col='Passengers'
elif data_src=='Demo: Stock (GARCH)':
    raw_df=load_demo_stock(); date_col='Date'; target_col='Price'
elif data_src=='Demo: Sales + Exog':
    raw_df=load_demo_sales(); date_col='Date'; target_col='Sales'
elif data_src=='Demo: Multiplicative':
    raw_df=load_demo_mult(); date_col='Date'; target_col='Value'
else:
    up=st.sidebar.file_uploader("CSV:",type=['csv'])
    if up:
        raw_df=pd.read_csv(up)
        date_col=st.sidebar.selectbox("Date col:",raw_df.columns)
        target_col=st.sidebar.selectbox("Target col:",[c for c in raw_df.columns if c!=date_col])
    else: st.info("Upload CSV dari sidebar."); st.stop()

raw_df[date_col]=pd.to_datetime(raw_df[date_col])
raw_df=raw_df.sort_values(date_col).reset_index(drop=True).set_index(date_col)
y_full=raw_df[target_col].dropna().astype(float)
st.sidebar.markdown(f"**n={len(y_full)}** | {y_full.index[0].strftime('%Y-%m')} → {y_full.index[-1].strftime('%Y-%m')}")

# ============================================================
# 1. EDA
# ============================================================
if module=='eda':
    st.header("1. Exploratory Data Analysis")
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=y_full.index,y=y_full.values,mode='lines',line=dict(color='steelblue',width=1.5)))
    fig.update_layout(title=f"Time Series: {target_col}",height=400,template='plotly_white')
    st.plotly_chart(fig,use_container_width=True)
    c1,c2,c3,c4=st.columns(4)
    c1.metric("n",str(len(y_full))); c2.metric("Mean",f"{y_full.mean():.2f}")
    c3.metric("Std",f"{y_full.std():.2f}"); c4.metric("CV%",f"{y_full.std()/y_full.mean()*100:.2f}%")
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Min",f"{y_full.min():.2f}"); c2.metric("Max",f"{y_full.max():.2f}")
    c3.metric("Skew",f"{y_full.skew():.4f}"); c4.metric("Kurtosis",f"{y_full.kurtosis():.4f}")
    fig2=make_subplots(rows=1,cols=3,subplot_titles=("Histogram","Box Plot","Rolling Stats"))
    fig2.add_trace(go.Histogram(x=y_full,nbinsx=30,marker_color='steelblue',opacity=0.7),row=1,col=1)
    fig2.add_trace(go.Box(y=y_full,marker_color='steelblue'),row=1,col=2)
    w=max(6,len(y_full)//20); rm=y_full.rolling(w).mean(); rs=y_full.rolling(w).std()
    fig2.add_trace(go.Scatter(x=rm.index,y=rm.values,mode='lines',name='Rolling Mean',line=dict(color='crimson')),row=1,col=3)
    fig2.add_trace(go.Scatter(x=rs.index,y=rs.values,mode='lines',name='Rolling Std',line=dict(color='green')),row=1,col=3)
    fig2.update_layout(height=350,template='plotly_white'); st.plotly_chart(fig2,use_container_width=True)
    if hasattr(y_full.index,'month'):
        mo=y_full.groupby(y_full.index.month).mean()
        fig3=go.Figure(data=[go.Bar(x=mo.index,y=mo.values,marker_color='steelblue')])
        fig3.update_layout(title="Average by Month",height=320,template='plotly_white')
        st.plotly_chart(fig3,use_container_width=True)
    with st.expander("Raw Data"): st.dataframe(raw_df.round(4),use_container_width=True)

# ============================================================
# 2. DECOMPOSITION
# ============================================================
elif module=='decomposition':
    st.header("2. Time Series Decomposition")
    c1,c2,c3=st.columns(3)
    method=c1.selectbox("Metode:",['Classical','STL'])
    model_type=c2.selectbox("Model:",['additive','multiplicative'])
    period=c3.number_input("Period:",2,365,12)
    if method=='Classical':
        try: decomp=seasonal_decompose(y_full,model=model_type,period=period)
        except Exception as e: st.error(str(e)); st.stop()
    else:
        if model_type=='multiplicative':
            st.info("STL additive only. Data di-log untuk approximate multiplicative.")
            try: decomp=STL(np.log(y_full),period=period).fit()
            except Exception as e: st.error(str(e)); st.stop()
        else:
            try: decomp=STL(y_full,period=period).fit()
            except Exception as e: st.error(str(e)); st.stop()
    fig=make_subplots(rows=4,cols=1,subplot_titles=("Observed","Trend","Seasonal","Residual"),
                      shared_xaxes=True,vertical_spacing=0.05)
    obs=np.log(y_full) if (method=='STL' and model_type=='multiplicative') else y_full
    fig.add_trace(go.Scatter(x=obs.index,y=obs.values,mode='lines',line=dict(width=1)),row=1,col=1)
    fig.add_trace(go.Scatter(x=y_full.index,y=decomp.trend if hasattr(decomp.trend,'__iter__') else [decomp.trend],
                             mode='lines',line=dict(color='crimson',width=1.5)),row=2,col=1)
    fig.add_trace(go.Scatter(x=y_full.index,y=decomp.seasonal,mode='lines',line=dict(color='green',width=1)),row=3,col=1)
    fig.add_trace(go.Scatter(x=y_full.index,y=decomp.resid,mode='markers',marker=dict(size=3,color='gray')),row=4,col=1)
    fig.update_layout(height=700,template='plotly_white',showlegend=False); st.plotly_chart(fig,use_container_width=True)
    resid=pd.Series(decomp.resid).dropna(); vr=resid.var()
    ts=pd.Series(decomp.trend).dropna(); ss=pd.Series(decomp.seasonal).dropna()
    Ft=max(0,1-vr/(ts.var()+vr)) if (ts.var()+vr)>0 else 0
    Fs=max(0,1-vr/(ss.var()+vr)) if (ss.var()+vr)>0 else 0
    st.markdown(f"**Strength of Trend:** {Ft:.4f} | **Strength of Seasonality:** {Fs:.4f}")

# ============================================================
# 3. STATIONARITY
# ============================================================
elif module=='stationarity':
    st.header("3. Stationarity Testing")
    transform=st.selectbox("Transformasi:",['None','Log','Sqrt','Box-Cox','Diff(1)','Diff(2)','Seasonal Diff','Log + Diff(1)'])
    yt,bc_lam=apply_transformation(y_full,transform)
    if len(yt)<10: st.error("Data terlalu sedikit."); st.stop()
    fig=make_subplots(rows=1,cols=2,subplot_titles=("Original",f"After: {transform}"))
    fig.add_trace(go.Scatter(x=y_full.index,y=y_full.values,mode='lines',line=dict(width=1)),row=1,col=1)
    fig.add_trace(go.Scatter(x=yt.index,y=yt.values,mode='lines',line=dict(width=1,color='crimson')),row=1,col=2)
    fig.update_layout(height=320,template='plotly_white',showlegend=False); st.plotly_chart(fig,use_container_width=True)
    st.markdown("### ADF (H₀: Non-Stationary)")
    adf_r=adf_test(yt); st.dataframe(pd.DataFrame([adf_r]),use_container_width=True,hide_index=True)
    if adf_r['Stationary']=='Yes': st.success("ADF: Stationary ✓")
    else: st.warning("ADF: Non-Stationary ✗")
    st.markdown("### KPSS (H₀: Stationary)")
    kpss_r=kpss_test(yt); st.dataframe(pd.DataFrame([kpss_r]),use_container_width=True,hide_index=True)
    if kpss_r.get('Stationary')=='Yes': st.success("KPSS: Stationary ✓")
    else: st.warning("KPSS: Non-Stationary ✗")
    adf_ok=adf_r['Stationary']=='Yes'; kpss_ok=kpss_r.get('Stationary')=='Yes'
    st.markdown("### Ringkasan")
    if adf_ok and kpss_ok: st.success("**STATIONARY**")
    elif adf_ok: st.info("Trend-stationary")
    elif kpss_ok: st.info("Difference-stationary")
    else: st.error("**NON-STATIONARY**")
    if not adf_ok:
        d=0; tmp=y_full.copy()
        for dd in range(1,4):
            tmp=tmp.diff().dropna()
            if len(tmp)<10: break
            if adfuller(tmp,autolag='AIC')[1]<0.05: d=dd; break
        if d>0: st.info(f"Suggested **d = {d}**")
    if bc_lam is not None: st.info(f"Box-Cox λ = **{bc_lam:.4f}**")

# ============================================================
# 4. ACF & PACF
# ============================================================
elif module=='acf_pacf':
    st.header("4. ACF & PACF Analysis")
    transform=st.selectbox("Transformasi:",['None','Log','Diff(1)','Diff(2)','Seasonal Diff','Log + Diff(1)'])
    yp,_=apply_transformation(y_full,transform)
    if len(yp)<10: st.error("Data terlalu sedikit."); st.stop()
    ml=st.slider("Max lags:",5,min(80,len(yp)//2-2),min(40,len(yp)//3))
    st.plotly_chart(plot_acf_pacf(yp,lags=ml,title=f"({transform})"),use_container_width=True)
    fig2=go.Figure(); fig2.add_trace(go.Scatter(x=yp.index,y=yp.values,mode='lines',line=dict(width=1)))
    fig2.update_layout(title=f"Series ({transform})",height=300,template='plotly_white')
    st.plotly_chart(fig2,use_container_width=True)
    with st.expander("Panduan Identifikasi"):
        st.markdown("""
| ACF | PACF | Model |
|---|---|---|
| Decays | Cuts off lag p | AR(p) |
| Cuts off lag q | Decays | MA(q) |
| Decays | Decays | ARMA(p,q) |
| Signif at s, 2s... | — | Seasonal |""")

# ============================================================
# 5. ARIMA
# ============================================================
elif module=='arima':
    st.header("5. ARIMA(p,d,q) Modeling")
    transform=st.selectbox("Transformasi:",['None','Log','Sqrt','Box-Cox'],key='ar_tr')
    ym,bc_lam=apply_transformation(y_full,transform)
    if len(ym)<10: st.error("Data terlalu sedikit."); st.stop()
    split_pct=st.slider("Train %:",50,95,80)
    nt=int(len(ym)*split_pct/100); y_tr=ym.iloc[:nt]; y_te=ym.iloc[nt:]
    st.info(f"Train: {len(y_tr)} | Test: {len(y_te)}")
    c1,c2,c3=st.columns(3)
    p=c1.number_input("p:",0,10,1); d=c2.number_input("d:",0,3,1); q=c3.number_input("q:",0,10,1)
    trend_opt=st.selectbox("Trend:",['n','c','t','ct'])
    if st.button("Fit ARIMA",type="primary"):
        with st.spinner("Fitting..."):
            try:
                model=ARIMA(y_tr,order=(p,d,q),trend=trend_opt); result=model.fit()
                label,eq=write_model_notation('ARIMA',(p,d,q),transform=transform)
                show_model_notation(label,eq)
                st.text(str(result.summary()))
                st.markdown("### Parameter Estimates"); show_param_table(result)
                c1,c2,c3,c4=st.columns(4)
                c1.metric("AIC",f"{result.aic:.2f}"); c2.metric("BIC",f"{result.bic:.2f}")
                c3.metric("HQIC",f"{result.hqic:.2f}"); c4.metric("Log-Lik",f"{result.llf:.2f}")
                fitted=result.fittedvalues
                fig=go.Figure()
                fig.add_trace(go.Scatter(x=y_tr.index,y=y_tr.values,mode='lines',name='Actual'))
                fig.add_trace(go.Scatter(x=fitted.index,y=fitted.values,mode='lines',name='Fitted',line=dict(dash='dash',color='crimson')))
                fig.update_layout(title="Fitted vs Actual (Train)",height=380,template='plotly_white')
                st.plotly_chart(fig,use_container_width=True)
                fc_obj=result.get_forecast(steps=len(y_te)); fc_mean=fc_obj.predicted_mean
                fc_ci=fc_obj.conf_int(alpha=0.05); cc=fc_ci.columns.tolist()
                fig2=go.Figure()
                fig2.add_trace(go.Scatter(x=y_tr.index,y=y_tr.values,mode='lines',name='Train'))
                fig2.add_trace(go.Scatter(x=y_te.index,y=y_te.values,mode='lines',name='Actual',line=dict(color='green',width=2)))
                fig2.add_trace(go.Scatter(x=fc_mean.index,y=fc_mean.values,mode='lines',name='Forecast',line=dict(color='crimson',width=2)))
                cl=fc_ci[cc[0]]; cu=fc_ci[cc[1]]
                fig2.add_trace(go.Scatter(x=cl.index.tolist()+cu.index.tolist()[::-1],
                    y=cl.values.tolist()+cu.values.tolist()[::-1],
                    fill='toself',fillcolor='rgba(255,0,0,0.1)',line=dict(color='rgba(0,0,0,0)'),name='95% CI'))
                fig2.update_layout(title="Forecast vs Actual",height=420,template='plotly_white')
                st.plotly_chart(fig2,use_container_width=True)
                st.markdown("### Evaluation (In-Sample / Out-Sample / Overall)")
                display_three_metrics(y_tr.values,fitted.values,y_te.values,fc_mean.values)
                st.markdown("### Residual Diagnostics")
                resid=result.resid.dropna()
                fig3=make_subplots(rows=2,cols=2,subplot_titles=("Residuals","Histogram","ACF Residuals","Q-Q Plot"))
                fig3.add_trace(go.Scatter(x=resid.index,y=resid.values,mode='lines',line=dict(width=0.8,color='gray')),row=1,col=1)
                fig3.add_hline(y=0,line=dict(color='red',dash='dash'),row=1,col=1)
                fig3.add_trace(go.Histogram(x=resid,nbinsx=30,marker_color='steelblue',opacity=0.7),row=1,col=2)
                mlr=min(30,len(resid)//2-2)
                if mlr>1:
                    acf_r=acf(resid,nlags=mlr,fft=True); ci_r=1.96/np.sqrt(len(resid))
                    for j in range(len(acf_r)):
                        cr='red' if abs(acf_r[j])>ci_r and j>0 else 'steelblue'
                        fig3.add_trace(go.Bar(x=[j],y=[acf_r[j]],marker_color=cr,showlegend=False,width=0.3),row=2,col=1)
                    fig3.add_hline(y=ci_r,line=dict(color='blue',dash='dash'),row=2,col=1)
                    fig3.add_hline(y=-ci_r,line=dict(color='blue',dash='dash'),row=2,col=1)
                nn=len(resid); osm=stats.norm.ppf(np.linspace(1/(nn+1),nn/(nn+1),nn)); osr=np.sort(resid.values)
                fig3.add_trace(go.Scatter(x=osm,y=osr,mode='markers',marker=dict(size=3,opacity=0.6)),row=2,col=2)
                fig3.add_trace(go.Scatter(x=[osm.min(),osm.max()],
                    y=[resid.mean()+resid.std()*osm.min(),resid.mean()+resid.std()*osm.max()],
                    mode='lines',line=dict(color='red',dash='dash')),row=2,col=2)
                fig3.update_layout(height=600,template='plotly_white',showlegend=False)
                st.plotly_chart(fig3,use_container_width=True)
                st.markdown("### Assumption Tests")
                lb=acorr_ljungbox(resid,lags=[10,20],return_df=True)
                st.markdown("**Ljung-Box (H₀: No Autocorrelation)**")
                st.dataframe(lb.round(4),use_container_width=True)
                if (lb['lb_pvalue']>0.05).all(): st.success("No autocorrelation ✓")
                else: st.warning("Autocorrelation detected ✗")
                jb_s,jb_p=stats.jarque_bera(resid)
                st.markdown(f"**Jarque-Bera:** stat={jb_s:.4f}, p={jb_p:.6f} → {'Normal ✓' if jb_p>0.05 else 'Non-Normal ✗'}")
                try:
                    ar_r=het_arch(resid,nlags=min(10,len(resid)//5))
                    st.markdown(f"**ARCH Effect:** LM={ar_r[0]:.4f}, p={ar_r[1]:.6f} → {'No ARCH ✓' if ar_r[1]>0.05 else 'ARCH → Try GARCH! ✗'}")
                except: pass
            except Exception as e: st.error(f"Error: {e}")

# ============================================================
# 6. SARIMA
# ============================================================
elif module=='sarima':
    st.header("6. SARIMA(p,d,q)(P,D,Q,s)")
    transform=st.selectbox("Transformasi:",['None','Log'],key='sar_tr')
    ym,_=apply_transformation(y_full,transform)
    if len(ym)<10: st.stop()
    sp=st.slider("Train %:",50,95,80,key='sar_sp')
    nt=int(len(ym)*sp/100); y_tr=ym.iloc[:nt]; y_te=ym.iloc[nt:]
    st.info(f"Train: {len(y_tr)} | Test: {len(y_te)}")
    c1,c2,c3=st.columns(3)
    p=c1.number_input("p:",0,5,1,key='sp'); d=c2.number_input("d:",0,2,1,key='sd'); q=c3.number_input("q:",0,5,1,key='sq')
    c1,c2,c3,c4=st.columns(4)
    P=c1.number_input("P:",0,3,1,key='sP'); D=c2.number_input("D:",0,2,1,key='sD')
    Q=c3.number_input("Q:",0,3,1,key='sQ'); s=c4.number_input("s:",1,365,12,key='ss')
    trend_opt=st.selectbox("Trend:",['n','c','t','ct'],key='sar_trend')
    if st.button("Fit SARIMA",type="primary"):
        with st.spinner("Fitting..."):
            try:
                model=SARIMAX(y_tr,order=(p,d,q),seasonal_order=(P,D,Q,s),trend=trend_opt,
                              enforce_stationarity=False,enforce_invertibility=False)
                result=model.fit(disp=False,maxiter=500)
                label,eq=write_model_notation('SARIMA',(p,d,q),seasonal_order=(P,D,Q,s),transform=transform)
                show_model_notation(label,eq)
                st.text(str(result.summary()))
                st.markdown("### Parameter Estimates"); show_param_table(result)
                c1,c2,c3=st.columns(3)
                c1.metric("AIC",f"{result.aic:.2f}"); c2.metric("BIC",f"{result.bic:.2f}"); c3.metric("HQIC",f"{result.hqic:.2f}")
                fitted=result.fittedvalues
                fc_obj=result.get_forecast(steps=len(y_te)); fc_mean=fc_obj.predicted_mean
                fc_ci=fc_obj.conf_int(alpha=0.05); cc=fc_ci.columns.tolist()
                fig=go.Figure()
                fig.add_trace(go.Scatter(x=y_tr.index,y=y_tr.values,mode='lines',name='Train'))
                fig.add_trace(go.Scatter(x=y_te.index,y=y_te.values,mode='lines',name='Actual',line=dict(color='green',width=2)))
                fig.add_trace(go.Scatter(x=fc_mean.index,y=fc_mean.values,mode='lines',name='Forecast',line=dict(color='crimson',width=2)))
                cl=fc_ci[cc[0]]; cu=fc_ci[cc[1]]
                fig.add_trace(go.Scatter(x=cl.index.tolist()+cu.index.tolist()[::-1],
                    y=cl.values.tolist()+cu.values.tolist()[::-1],
                    fill='toself',fillcolor='rgba(255,0,0,0.1)',line=dict(color='rgba(0,0,0,0)'),name='95% CI'))
                fig.update_layout(title=f"SARIMA({p},{d},{q})({P},{D},{Q},{s})",height=420,template='plotly_white')
                st.plotly_chart(fig,use_container_width=True)
                st.markdown("### Evaluation (In-Sample / Out-Sample / Overall)")
                display_three_metrics(y_tr.values,fitted.values,y_te.values,fc_mean.values)
                resid=result.resid.dropna()
                lb=acorr_ljungbox(resid,lags=[10,20],return_df=True)
                jb_s,jb_p=stats.jarque_bera(resid)
                st.markdown(f"**Ljung-Box p(10):** {lb['lb_pvalue'].iloc[0]:.4f} | **JB p:** {jb_p:.4f}")
            except Exception as e: st.error(f"Error: {e}")

# ============================================================
# 7. ARIMAX
# ============================================================
elif module=='arimax':
    st.header("7. ARIMAX — ARIMA + Exogenous")
    exog_cols=[c for c in raw_df.columns if c!=target_col]
    if not exog_cols: st.warning("No exog vars. Use 'Sales + Exog' demo."); st.stop()
    sel_exog=st.multiselect("Exog vars:",exog_cols,default=exog_cols[:min(2,len(exog_cols))])
    if not sel_exog: st.warning("Pilih min 1."); st.stop()
    X=raw_df[sel_exog].dropna(); ci_idx=y_full.index.intersection(X.index)
    ym=y_full.loc[ci_idx]; X=X.loc[ci_idx]
    sp=st.slider("Train %:",50,95,80,key='ax_sp')
    nt=int(len(ym)*sp/100)
    y_tr=ym.iloc[:nt]; y_te=ym.iloc[nt:]; X_tr=X.iloc[:nt]; X_te=X.iloc[nt:]
    st.info(f"Train: {len(y_tr)} | Test: {len(y_te)} | Exog: {sel_exog}")
    c1,c2,c3=st.columns(3)
    p=c1.number_input("p:",0,5,1,key='axp'); d=c2.number_input("d:",0,2,1,key='axd'); q=c3.number_input("q:",0,5,1,key='axq')
    c1,c2,c3,c4=st.columns(4)
    P=c1.number_input("P:",0,3,0,key='axP'); D=c2.number_input("D:",0,2,0,key='axD')
    Q=c3.number_input("Q:",0,3,0,key='axQ'); s=c4.number_input("s:",1,365,12,key='axs')
    if st.button("Fit ARIMAX",type="primary"):
        with st.spinner("Fitting..."):
            try:
                model=SARIMAX(y_tr,exog=X_tr,order=(p,d,q),seasonal_order=(P,D,Q,s),
                              enforce_stationarity=False,enforce_invertibility=False)
                result=model.fit(disp=False,maxiter=500)
                label,eq=write_model_notation('ARIMAX',(p,d,q),seasonal_order=(P,D,Q,s),exog_names=sel_exog)
                show_model_notation(label,eq)
                st.text(str(result.summary()))
                st.markdown("### Parameter Estimates"); show_param_table(result)
                fitted=result.fittedvalues
                fc=result.get_forecast(steps=len(y_te),exog=X_te)
                fc_mean=fc.predicted_mean; fc_ci=fc.conf_int(alpha=0.05); cc=fc_ci.columns.tolist()
                fig=go.Figure()
                fig.add_trace(go.Scatter(x=y_tr.index,y=y_tr.values,mode='lines',name='Train'))
                fig.add_trace(go.Scatter(x=y_te.index,y=y_te.values,mode='lines',name='Actual',line=dict(color='green',width=2)))
                fig.add_trace(go.Scatter(x=fc_mean.index,y=fc_mean.values,mode='lines',name='Forecast',line=dict(color='crimson',width=2)))
                cl=fc_ci[cc[0]]; cu=fc_ci[cc[1]]
                fig.add_trace(go.Scatter(x=cl.index.tolist()+cu.index.tolist()[::-1],
                    y=cl.values.tolist()+cu.values.tolist()[::-1],
                    fill='toself',fillcolor='rgba(255,0,0,0.1)',line=dict(color='rgba(0,0,0,0)'),name='95% CI'))
                fig.update_layout(title="ARIMAX Forecast",height=420,template='plotly_white')
                st.plotly_chart(fig,use_container_width=True)
                st.markdown("### Evaluation (In-Sample / Out-Sample / Overall)")
                display_three_metrics(y_tr.values,fitted.values,y_te.values,fc_mean.values)
            except Exception as e: st.error(f"Error: {e}")

# ============================================================
# 8. ARIMA-GARCH
# ============================================================
elif module=='garch':
    st.header("8. ARIMA-GARCH Modeling")
    if not HAS_ARCH:
        st.error("⚠️ Package `arch` belum terinstall!")
        st.markdown("""
### Cara Install
**Streamlit Cloud:** Tambahkan `arch` di file `requirements-arima.txt`, lalu redeploy.

**Lokal:**
```bash
pip install arch
```
Setelah install, **refresh** halaman ini.
""")
        st.stop()
    use_returns=st.checkbox("Gunakan log-returns?",True)
    if use_returns:
        yp=y_full[y_full>0]; ym=(np.log(yp).diff().dropna()*100)
        st.info("Using log-returns (×100)")
    else: ym=y_full.copy()
    if len(ym)<30: st.error("Data terlalu sedikit."); st.stop()
    fig=go.Figure(); fig.add_trace(go.Scatter(x=ym.index,y=ym.values,mode='lines',line=dict(width=0.8)))
    fig.update_layout(title="Series for GARCH",height=300,template='plotly_white'); st.plotly_chart(fig,use_container_width=True)
    sp=st.slider("Train %:",50,95,80,key='g_sp')
    nt=int(len(ym)*sp/100); y_tr=ym.iloc[:nt]; y_te=ym.iloc[nt:]
    st.info(f"Train: {len(y_tr)} | Test: {len(y_te)}")
    c1,c2=st.columns(2)
    ar=c1.number_input("AR lags:",0,5,1,key='gar'); ma=c2.number_input("MA lags:",0,5,1,key='gma')
    c1,c2=st.columns(2)
    gp=c1.number_input("GARCH p:",1,5,1,key='gp'); gq=c2.number_input("GARCH q:",1,5,1,key='gq')
    vol_model=st.selectbox("Volatility:",['GARCH','EGARCH','GJR-GARCH'])
    dist=st.selectbox("Distribution:",['normal','t','skewt','ged'])
    if st.button("Fit GARCH",type="primary"):
        with st.spinner("Fitting..."):
            try:
                vol_map={'GARCH':'Garch','EGARCH':'EGARCH','GJR-GARCH':'GARCH'}
                o_val=gp if vol_model=='GJR-GARCH' else 0
                am=arch_model(y_tr,mean='ARX',lags=ar,vol=vol_map[vol_model],p=gp,q=gq,o=o_val,dist=dist)
                res=am.fit(disp='off')
                st.markdown(f"### Model: **AR({ar})-{vol_model}({gp},{gq})** [{dist}]")
                mean_eq=f"r_t = \\mu"
                if ar>0: mean_eq+=" + "+" + ".join([f"\\phi_{{{i}}} r_{{t-{i}}}" for i in range(1,ar+1)])
                mean_eq+=" + \\varepsilon_t, \\quad \\varepsilon_t = \\sigma_t z_t"
                if vol_model=='GARCH':
                    vol_eq=f"\\sigma_t^2 = \\omega + "+" + ".join([f"\\alpha_{{{i}}} \\varepsilon_{{t-{i}}}^2" for i in range(1,gp+1)])+" + "+" + ".join([f"\\beta_{{{i}}} \\sigma_{{t-{i}}}^2" for i in range(1,gq+1)])
                elif vol_model=='EGARCH':
                    vol_eq=f"\\ln(\\sigma_t^2) = \\omega + "+" + ".join([f"\\alpha_{{{i}}} |z_{{t-{i}}}|" for i in range(1,gp+1)])+" + "+" + ".join([f"\\beta_{{{i}}} \\ln(\\sigma_{{t-{i}}}^2)" for i in range(1,gq+1)])
                else:
                    vol_eq=f"\\sigma_t^2 = \\omega + "+" + ".join([f"(\\alpha_{{{i}}}+\\gamma_{{{i}}} I_{{t-{i}}})\\varepsilon_{{t-{i}}}^2" for i in range(1,gp+1)])+" + "+" + ".join([f"\\beta_{{{i}}} \\sigma_{{t-{i}}}^2" for i in range(1,gq+1)])
                st.latex(mean_eq); st.latex(vol_eq)
                st.text(str(res.summary()))
                st.markdown("### Parameter Estimates")
                gp_=res.params; gs_=res.std_err; gpv=res.pvalues
                rows=[]
                for nm in gp_.index:
                    rows.append({'Parameter':nm,'Estimate':f"{gp_[nm]:.6f}",
                        'Std Error':f"{gs_[nm]:.6f}" if nm in gs_.index else '-',
                        'p-value':f"{gpv[nm]:.6f}" if nm in gpv.index else '-',
                        'Signif':'***' if gpv.get(nm,1)<0.001 else ('**' if gpv.get(nm,1)<0.01 else ('*' if gpv.get(nm,1)<0.05 else ''))})
                st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
                fig=make_subplots(rows=2,cols=1,subplot_titles=("Returns + Cond. Volatility","Std Residuals"),shared_xaxes=True)
                fig.add_trace(go.Scatter(x=y_tr.index,y=y_tr.values,mode='lines',line=dict(width=0.5,color='gray'),name='Returns'),row=1,col=1)
                fig.add_trace(go.Scatter(x=y_tr.index,y=res.conditional_volatility,mode='lines',line=dict(color='crimson',width=1.5),name='Cond Vol'),row=1,col=1)
                fig.add_trace(go.Scatter(x=y_tr.index,y=-res.conditional_volatility,mode='lines',line=dict(color='crimson',width=1.5),showlegend=False),row=1,col=1)
                std_r=res.resid/res.conditional_volatility
                fig.add_trace(go.Scatter(x=y_tr.index,y=std_r,mode='lines',line=dict(width=0.5,color='steelblue')),row=2,col=1)
                fig.update_layout(height=500,template='plotly_white'); st.plotly_chart(fig,use_container_width=True)
                fc=res.forecast(horizon=min(len(y_te),100))
                fc_m=fc.mean.iloc[-1].values; fc_v=fc.variance.iloc[-1].values; fc_vol=np.sqrt(fc_v)
                h=min(len(y_te),len(fc_m))
                fig2=go.Figure()
                fig2.add_trace(go.Scatter(x=y_te.index[:h],y=y_te.values[:h],mode='lines',name='Actual'))
                fig2.add_trace(go.Scatter(x=y_te.index[:h],y=fc_m[:h],mode='lines',name='Forecast',line=dict(color='crimson')))
                fig2.add_trace(go.Scatter(x=y_te.index[:h].tolist()+y_te.index[:h].tolist()[::-1],
                    y=(fc_m[:h]+1.96*fc_vol[:h]).tolist()+(fc_m[:h]-1.96*fc_vol[:h]).tolist()[::-1],
                    fill='toself',fillcolor='rgba(255,0,0,0.1)',line=dict(color='rgba(0,0,0,0)'),name='95% CI'))
                fig2.update_layout(title="GARCH Forecast",height=400,template='plotly_white')
                st.plotly_chart(fig2,use_container_width=True)
                in_fitted=y_tr.values-res.resid.values
                st.markdown("### Evaluation (In-Sample / Out-Sample / Overall)")
                display_three_metrics(y_tr.values,in_fitted,y_te.values[:h],fc_m[:h])
            except Exception as e: st.error(f"Error: {e}")

# ============================================================
# 9. AUTO ARIMA (GRID SEARCH)
# ============================================================
elif module=='auto_arima':
    st.header("9. Auto ARIMA — Grid Search")
    transform=st.selectbox("Transformasi:",['None','Log'],key='auto_tr')
    if transform=='Log' and (y_full>0).all(): ym=np.log(y_full)
    else: ym=y_full.copy()
    sp=st.slider("Train %:",50,95,80,key='auto_sp')
    nt=int(len(ym)*sp/100); y_tr=ym.iloc[:nt]; y_te=ym.iloc[nt:]
    st.info(f"Train: {len(y_tr)} | Test: {len(y_te)}")
    st.markdown("### Search Range")
    c1,c2,c3=st.columns(3)
    max_p=c1.number_input("Max p:",0,5,2,key='ap')
    max_d=c2.number_input("Max d:",0,2,1,key='ad')
    max_q=c3.number_input("Max q:",0,5,2,key='aq')
    seasonal=st.checkbox("Seasonal?",True,key='auto_seas')
    s_period=12; max_P=0; max_D=0; max_Q=0
    if seasonal:
        c1,c2,c3,c4=st.columns(4)
        max_P=c1.number_input("Max P:",0,2,1,key='aP')
        max_D=c2.number_input("Max D:",0,1,1,key='aD')
        max_Q=c3.number_input("Max Q:",0,2,1,key='aQ')
        s_period=c4.number_input("s:",1,365,12,key='as_')
    criterion=st.selectbox("Kriteria:",['AIC','BIC'],key='auto_crit')
    if st.button("Run Grid Search",type="primary"):
        with st.spinner("Searching..."):
            results=[]
            if seasonal:
                combos=list(itertools.product(range(max_p+1),range(max_d+1),range(max_q+1),
                    range(max_P+1),range(max_D+1),range(max_Q+1)))
            else:
                combos=list(itertools.product(range(max_p+1),range(max_d+1),range(max_q+1)))
            total=len(combos); st.write(f"Total: {total} kombinasi")
            progress=st.progress(0)
            for idx,combo in enumerate(combos):
                progress.progress(min((idx+1)/total,1.0))
                try:
                    if seasonal:
                        pp,dd,qq,PP,DD,QQ=combo
                        if pp==0 and qq==0 and PP==0 and QQ==0: continue
                        m=SARIMAX(y_tr,order=(pp,dd,qq),seasonal_order=(PP,DD,QQ,s_period),
                                  enforce_stationarity=False,enforce_invertibility=False)
                    else:
                        pp,dd,qq=combo
                        if pp==0 and qq==0: continue
                        m=ARIMA(y_tr,order=(pp,dd,qq))
                    res=m.fit(disp=False,maxiter=200)
                    fv=res.fittedvalues; fc=res.get_forecast(steps=len(y_te)); fcm=fc.predicted_mean
                    im=calc_metrics(y_tr.values,fv.values); om=calc_metrics(y_te.values,fcm.values)
                    row={'p':pp,'d':dd,'q':qq,'AIC':round(res.aic,2),'BIC':round(res.bic,2),
                        'RMSE_train':round(im.get('RMSE',999),4),'RMSE_test':round(om.get('RMSE',999),4),
                        'MAE_test':round(om.get('MAE',999),4),
                        'MAPE_test':round(om['MAPE'],2) if om.get('MAPE') is not None else None}
                    if seasonal: row['P']=PP; row['D']=DD; row['Q']=QQ; row['s']=s_period
                    results.append(row)
                except: pass
            progress.empty()
            if results:
                rdf=pd.DataFrame(results).sort_values(criterion).reset_index(drop=True)
                st.markdown(f"### Top 20 (by {criterion})")
                st.dataframe(rdf.head(20),use_container_width=True,hide_index=True)
                best=rdf.iloc[0]; bp=int(best['p']); bd=int(best['d']); bq=int(best['q'])
                st.markdown("### Best Model")
                try:
                    if seasonal:
                        bP=int(best['P']); bD=int(best['D']); bQ=int(best['Q']); bs=int(best['s'])
                        bm=SARIMAX(y_tr,order=(bp,bd,bq),seasonal_order=(bP,bD,bQ,bs),
                                   enforce_stationarity=False,enforce_invertibility=False).fit(disp=False)
                        label,eq=write_model_notation('SARIMA',(bp,bd,bq),seasonal_order=(bP,bD,bQ,bs),transform=transform)
                    else:
                        bm=ARIMA(y_tr,order=(bp,bd,bq)).fit()
                        label,eq=write_model_notation('ARIMA',(bp,bd,bq),transform=transform)
                    show_model_notation(label,eq)
                    st.success(f"**{label}** | AIC={best['AIC']} | BIC={best['BIC']} | RMSE_test={best['RMSE_test']}")
                    st.markdown("### Parameter Estimates"); show_param_table(bm)
                    fv=bm.fittedvalues; fc=bm.get_forecast(steps=len(y_te))
                    fcm=fc.predicted_mean; fcci=fc.conf_int(alpha=0.05); cc=fcci.columns.tolist()
                    fig=go.Figure()
                    fig.add_trace(go.Scatter(x=y_tr.index,y=y_tr.values,mode='lines',name='Train'))
                    fig.add_trace(go.Scatter(x=y_te.index,y=y_te.values,mode='lines',name='Actual',line=dict(color='green',width=2)))
                    fig.add_trace(go.Scatter(x=fcm.index,y=fcm.values,mode='lines',name='Forecast',line=dict(color='crimson',width=2)))
                    cl=fcci[cc[0]]; cu=fcci[cc[1]]
                    fig.add_trace(go.Scatter(x=cl.index.tolist()+cu.index.tolist()[::-1],
                        y=cl.values.tolist()+cu.values.tolist()[::-1],
                        fill='toself',fillcolor='rgba(255,0,0,0.1)',line=dict(color='rgba(0,0,0,0)'),name='95% CI'))
                    fig.update_layout(title=f"Best: {label}",height=420,template='plotly_white')
                    st.plotly_chart(fig,use_container_width=True)
                    st.markdown("### Evaluation (In-Sample / Out-Sample / Overall)")
                    display_three_metrics(y_tr.values,fv.values,y_te.values,fcm.values)
                except Exception as e: st.error(f"Error: {e}")
            else: st.error("No valid models found.")

# ============================================================
# 10. MODEL COMPARISON
# ============================================================
elif module=='comparison':
    st.header("10. Model Comparison")
    transform=st.selectbox("Transformasi:",['None','Log'],key='comp_tr')
    if transform=='Log' and (y_full>0).all(): ym=np.log(y_full)
    else: ym=y_full.copy()
    sp=st.slider("Train %:",50,95,80,key='comp_sp')
    nt=int(len(ym)*sp/100); y_tr=ym.iloc[:nt]; y_te=ym.iloc[nt:]
    models_sel=st.multiselect("Models:",['ARIMA(1,1,1)','ARIMA(2,1,2)','ARIMA(1,1,0)','ARIMA(0,1,1)',
        'SARIMA(1,1,1)(1,1,1,12)','SARIMA(0,1,1)(0,1,1,12)',
        'Holt-Winters (Add)','Holt-Winters (Mul)','Naive','Drift'],
        default=['ARIMA(1,1,1)','SARIMA(1,1,1)(1,1,1,12)','Naive'])
    if st.button("Compare",type="primary") and models_sel:
        all_res=[]; forecasts={}
        for mname in models_sel:
            try:
                fc=None; aic=None; bic=None; fv=None
                if mname.startswith('ARIMA('):
                    order=tuple(int(x) for x in mname.replace('ARIMA','').strip('()').split(','))
                    m=ARIMA(y_tr,order=order).fit()
                    fc=m.get_forecast(len(y_te)).predicted_mean; fv=m.fittedvalues.values; aic=m.aic; bic=m.bic
                elif mname.startswith('SARIMA('):
                    inner=mname.replace('SARIMA',''); parts=inner.split(')(')
                    order=tuple(int(x) for x in parts[0].strip('(').split(','))
                    seasonal=tuple(int(x) for x in parts[1].strip(')').split(','))
                    m=SARIMAX(y_tr,order=order,seasonal_order=seasonal,
                              enforce_stationarity=False,enforce_invertibility=False).fit(disp=False)
                    fc=m.get_forecast(len(y_te)).predicted_mean; fv=m.fittedvalues.values; aic=m.aic; bic=m.bic
                elif 'Holt' in mname:
                    st_=('add' if 'Add' in mname else 'mul')
                    try: m=ExponentialSmoothing(y_tr,trend='add',seasonal=st_,seasonal_periods=12).fit()
                    except: m=ExponentialSmoothing(y_tr,trend='add',seasonal='add',seasonal_periods=12).fit()
                    fc=m.forecast(len(y_te)); fv=m.fittedvalues.values; aic=m.aic; bic=m.bic
                elif mname=='Naive':
                    fc=pd.Series(np.full(len(y_te),y_tr.iloc[-1]),index=y_te.index)
                    fv=np.full(len(y_tr),y_tr.iloc[-1])
                elif mname=='Drift':
                    ntr=len(y_tr); dr=(y_tr.iloc[-1]-y_tr.iloc[0])/(ntr-1) if ntr>1 else 0
                    fc=pd.Series([y_tr.iloc[-1]+dr*(i+1) for i in range(len(y_te))],index=y_te.index)
                    fv=np.array([y_tr.iloc[0]+dr*i for i in range(ntr)])
                if fc is not None:
                    im=calc_metrics(y_tr.values,fv) if fv is not None else {k:None for k in ['RMSE','MAE','MAPE']}
                    om=calc_metrics(y_te.values,fc.values)
                    all_res.append({'Model':mname,
                        'AIC':f"{aic:.2f}" if aic else '-','BIC':f"{bic:.2f}" if bic else '-',
                        'RMSE (Train)':f"{im.get('RMSE',0):.4f}" if im.get('RMSE') is not None else '-',
                        'RMSE (Test)':f"{om.get('RMSE',0):.4f}",
                        'MAE (Test)':f"{om.get('MAE',0):.4f}",
                        'MAPE (Test)':f"{om['MAPE']:.2f}%" if om.get('MAPE') is not None else '-',
                        'R2 (Test)':f"{om['R2']:.4f}" if om.get('R2') is not None else '-'})
                    forecasts[mname]=fc
            except Exception as e:
                all_res.append({'Model':mname,'AIC':'-','BIC':'-','RMSE (Train)':'-',
                    'RMSE (Test)':str(e)[:30],'MAE (Test)':'-','MAPE (Test)':'-','R2 (Test)':'-'})
        if all_res: st.dataframe(pd.DataFrame(all_res),use_container_width=True,hide_index=True)
        if forecasts:
            fig=go.Figure()
            fig.add_trace(go.Scatter(x=y_tr.index,y=y_tr.values,mode='lines',name='Train',line=dict(color='gray',width=1)))
            fig.add_trace(go.Scatter(x=y_te.index,y=y_te.values,mode='lines',name='Actual',line=dict(color='black',width=2)))
            colors=px.colors.qualitative.Set1
            for i,(mn,fc) in enumerate(forecasts.items()):
                fig.add_trace(go.Scatter(x=fc.index,y=fc.values,mode='lines',name=mn,line=dict(color=colors[i%len(colors)],width=1.5)))
            fig.update_layout(title="Model Comparison",height=450,template='plotly_white')
            st.plotly_chart(fig,use_container_width=True)

# ============================================================
# 11. FORECAST & CI/PI
# ============================================================
elif module=='forecast':
    st.header("11. Forecasting with CI & PI")
    transform=st.selectbox("Transformasi:",['None','Log'],key='fc_tr')
    if transform=='Log' and (y_full>0).all(): ym=np.log(y_full)
    else: ym=y_full.copy()
    model_type=st.selectbox("Model:",['ARIMA','SARIMA'])
    c1,c2,c3=st.columns(3)
    p=c1.number_input("p:",0,5,1,key='fcp'); d=c2.number_input("d:",0,2,1,key='fcd'); q=c3.number_input("q:",0,5,1,key='fcq')
    if model_type=='SARIMA':
        c1,c2,c3,c4=st.columns(4)
        P=c1.number_input("P:",0,3,1,key='fcP'); D=c2.number_input("D:",0,2,1,key='fcD')
        Q=c3.number_input("Q:",0,3,1,key='fcQ'); s=c4.number_input("s:",1,365,12,key='fcs')
    h=st.slider("Horizon:",1,120,24,key='fc_h')
    alpha=st.slider("Alpha:",0.01,0.20,0.05,0.01,key='fc_alpha')
    ci_lb=f"{(1-alpha)*100:.0f}%"
    if st.button("Forecast",type="primary"):
        with st.spinner("Forecasting..."):
            try:
                if model_type=='ARIMA':
                    model=ARIMA(ym,order=(p,d,q)).fit()
                    label,eq=write_model_notation('ARIMA',(p,d,q),transform=transform)
                else:
                    model=SARIMAX(ym,order=(p,d,q),seasonal_order=(P,D,Q,s),
                                  enforce_stationarity=False,enforce_invertibility=False).fit(disp=False)
                    label,eq=write_model_notation('SARIMA',(p,d,q),seasonal_order=(P,D,Q,s),transform=transform)
                show_model_notation(label,eq)
                st.markdown("### Parameter Estimates"); show_param_table(model)
                c1,c2,c3=st.columns(3)
                c1.metric("AIC",f"{model.aic:.2f}"); c2.metric("BIC",f"{model.bic:.2f}"); c3.metric("Log-Lik",f"{model.llf:.2f}")
                fc_obj=model.get_forecast(steps=h); fc_mean=fc_obj.predicted_mean
                fc_ci=fc_obj.conf_int(alpha=alpha); cc=fc_ci.columns.tolist()
                ci_l=fc_ci[cc[0]]; ci_u=fc_ci[cc[1]]
                resid_std=model.resid.std()
                ci_hw=(ci_u.values-fc_mean.values)
                pi_l=fc_mean.values-np.sqrt(ci_hw**2+resid_std**2)
                pi_u=fc_mean.values+np.sqrt(ci_hw**2+resid_std**2)
                if transform=='Log':
                    fcp=np.exp(fc_mean); cll=np.exp(ci_l); cuu=np.exp(ci_u)
                    pll=np.exp(pi_l); puu=np.exp(pi_u); yp=np.exp(ym)
                else:
                    fcp=fc_mean; cll=ci_l; cuu=ci_u; pll=pi_l; puu=pi_u; yp=ym
                fig=go.Figure()
                fig.add_trace(go.Scatter(x=yp.index,y=yp.values,mode='lines',name='Historical',line=dict(color='steelblue',width=1.5)))
                fig.add_trace(go.Scatter(x=fcp.index,y=fcp.values if hasattr(fcp,'values') else fcp,mode='lines',name='Forecast',line=dict(color='crimson',width=2)))
                pll_v=pll if isinstance(pll,np.ndarray) else pll.values
                puu_v=puu if isinstance(puu,np.ndarray) else puu.values
                fig.add_trace(go.Scatter(x=fcp.index.tolist()+fcp.index.tolist()[::-1],
                    y=list(pll_v)+list(puu_v)[::-1],fill='toself',fillcolor='rgba(255,165,0,0.08)',
                    line=dict(color='rgba(0,0,0,0)'),name=f'{ci_lb} PI'))
                cll_v=cll.values if hasattr(cll,'values') else cll; cuu_v=cuu.values if hasattr(cuu,'values') else cuu
                fig.add_trace(go.Scatter(x=fcp.index.tolist()+fcp.index.tolist()[::-1],
                    y=list(cll_v)+list(cuu_v)[::-1],fill='toself',fillcolor='rgba(255,0,0,0.15)',
                    line=dict(color='rgba(0,0,0,0)'),name=f'{ci_lb} CI'))
                fig.update_layout(title=f"Forecast {h} periods ({ci_lb} CI & PI)",height=500,template='plotly_white')
                st.plotly_chart(fig,use_container_width=True)
                fcp_v=fcp.values if hasattr(fcp,'values') else fcp
                fc_df=pd.DataFrame({'Date':fc_mean.index,'Forecast':np.round(fcp_v,4),
                    'CI Lower':np.round(cll_v,4),'CI Upper':np.round(cuu_v,4),
                    'PI Lower':np.round(pll_v,4),'PI Upper':np.round(puu_v,4)})
                st.dataframe(fc_df,use_container_width=True,hide_index=True)
                st.download_button("Download CSV",fc_df.to_csv(index=False),"forecast.csv","text/csv")
                resid=model.resid.dropna()
                lb=acorr_ljungbox(resid,lags=[10,20],return_df=True)
                jb_s,jb_p=stats.jarque_bera(resid)
                st.markdown(f"**Ljung-Box p(10):** {lb['lb_pvalue'].iloc[0]:.4f} | **JB p:** {jb_p:.4f}")
            except Exception as e: st.error(f"Error: {e}")

# FOOTER
st.divider()
st.markdown("<div style='text-align:center;color:gray;font-size:0.85rem'><b>ARIMA Pro</b> | 11 Modul | 2026</div>",unsafe_allow_html=True)
