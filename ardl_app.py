import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings, io, json
from datetime import datetime
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.tsa.ardl import ARDL, ardl_select_order
from statsmodels.stats.diagnostic import het_breuschpagan, acorr_breusch_godfrey
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from scipy import stats
warnings.filterwarnings("ignore")
st.set_page_config(page_title="ARDL Analysis Suite",page_icon="\U0001f4ca",layout="wide",initial_sidebar_state="expanded")
st.markdown("""<style>
.main-header{font-size:2.5rem;font-weight:700;background:linear-gradient(135deg,#1a5276,#2e86c1);-webkit-background-clip:text;-webkit-text-fill-color:transparent;text-align:center;padding:1rem 0}
.sub-header{font-size:1.1rem;color:#5D6D7E;text-align:center;margin-bottom:2rem}
.success-box{background-color:#D5F5E3;border-left:5px solid #27AE60;padding:1rem;border-radius:5px;margin:.5rem 0}.success-box,.success-box b,.success-box *{color:#145a32!important}
.warning-box{background-color:#FEF9E7;border-left:5px solid #F39C12;padding:1rem;border-radius:5px;margin:.5rem 0}.warning-box,.warning-box b,.warning-box *{color:#7d6608!important}
.error-box{background-color:#FADBD8;border-left:5px solid #E74C3C;padding:1rem;border-radius:5px;margin:.5rem 0}.error-box,.error-box b,.error-box *{color:#78281f!important}
.info-box{background-color:#D6EAF8;border-left:5px solid #2E86C1;padding:1rem;border-radius:5px;margin:.5rem 0}.info-box,.info-box b,.info-box *{color:#1a4971!important}
</style>""",unsafe_allow_html=True)
_d={"data":None,"dep_var":None,"indep_vars":[],"date_col":None,"ardl_results":None,"bounds_results":None,"ecm_results":None,"max_lag_dep":4,"max_lag_indep":4,"ic_criterion":"aic","trend":"c","ardl_order":None,"ardl_order_str":"","unit_root_results":{},"lr_coefs":{},"manual_lags":None,"ecm_data":None}
for k,v in _d.items():
    if k not in st.session_state: st.session_state[k]=v
def adf_test(series,name="",maxlag=None,regression="c"):
    s=series.dropna()
    if len(s)<10: return {"Variable":name,"Test":"ADF","Statistic":float("nan"),"P-value":float("nan"),"Lags":0,"Nobs":len(s),"CV1":float("nan"),"CV5":float("nan"),"CV10":float("nan"),"Stationary":False}
    r=adfuller(s,maxlag=maxlag,regression=regression,autolag="AIC")
    return {"Variable":name,"Test":"ADF","Statistic":r[0],"P-value":r[1],"Lags":r[2],"Nobs":r[3],"CV1":r[4]["1%"],"CV5":r[4]["5%"],"CV10":r[4]["10%"],"Stationary":r[1]<0.05}
def kpss_test(series,name="",regression="c"):
    s=series.dropna()
    if len(s)<10: return {"Variable":name,"Test":"KPSS","Statistic":float("nan"),"P-value":float("nan"),"Lags":0,"CV1":float("nan"),"CV5":float("nan"),"CV10":float("nan"),"Stationary":False}
    r=kpss(s,regression=regression,nlags="auto")
    return {"Variable":name,"Test":"KPSS","Statistic":r[0],"P-value":r[1],"Lags":r[2],"CV1":r[3]["1%"],"CV5":r[3]["5%"],"CV10":r[3]["10%"],"Stationary":r[1]>0.05}
def pp_test(series,name=""):
    s=series.dropna()
    if len(s)<10: return {"Variable":name,"Test":"PP","Statistic":float("nan"),"P-value":float("nan"),"Lags":0,"Nobs":len(s),"CV1":float("nan"),"CV5":float("nan"),"CV10":float("nan"),"Stationary":False}
    ml=max(1,int(np.ceil(12*(len(s)/100)**(1/4))))
    r=adfuller(s,maxlag=ml,regression="c",autolag="AIC")
    return {"Variable":name,"Test":"PP","Statistic":r[0],"P-value":r[1],"Lags":r[2],"Nobs":r[3],"CV1":r[4]["1%"],"CV5":r[4]["5%"],"CV10":r[4]["10%"],"Stationary":r[1]<0.05}
def get_pesaran_cv(k):
    t={1:{"10%":(4.04,4.78),"5%":(4.94,5.73),"1%":(6.84,7.84)},2:{"10%":(3.17,3.79),"5%":(3.79,4.85),"1%":(5.15,6.36)},3:{"10%":(2.72,3.77),"5%":(3.23,4.35),"1%":(4.29,5.61)},4:{"10%":(2.45,3.52),"5%":(2.86,4.01),"1%":(3.74,5.06)},5:{"10%":(2.26,3.35),"5%":(2.62,3.79),"1%":(3.41,4.68)},6:{"10%":(2.12,3.23),"5%":(2.45,3.61),"1%":(3.15,4.43)},7:{"10%":(2.03,3.13),"5%":(2.32,3.50),"1%":(2.96,4.26)}}
    return t.get(min(max(k,1),7),t[7])
def narayan_cv(n,k):
    adj=max(1.0,80/max(n,30))
    return {s:(round(l*adj,2),round(h*adj,2)) for s,(l,h) in get_pesaran_cv(k).items()}
def interpret_bounds(f,cv):
    l5,h5=cv["5%"];l10,h10=cv["10%"]
    if f>h5: return "Cointegration EXISTS (F > upper 5%)","success"
    elif f>h10: return "Cointegration at 10%","warning"
    elif f<l5: return "NO cointegration","error"
    else: return "INCONCLUSIVE","warning"
def generate_demo_data(scenario):
    np.random.seed(42)
    if scenario=="GDP & Macro (Quarterly)":
        n=120;dates=pd.date_range("1994-01-01",periods=n,freq="QS");t=np.linspace(0,3,n);gdp=100+t*20+np.cumsum(np.random.normal(0.5,1.5,n));inv=30+t*5+0.3*gdp+np.cumsum(np.random.normal(0.2,0.8,n));trd=40+t*8+0.2*gdp-0.1*inv+np.cumsum(np.random.normal(0.1,1.0,n));inf=5+np.random.normal(0,1.5,n)+0.02*np.diff(np.concatenate([[0],gdp]));rat=6+0.5*inf+np.cumsum(np.random.normal(0,0.3,n))
        return pd.DataFrame({"Date":dates,"GDP":np.round(gdp,2),"Investment":np.round(inv,2),"Trade":np.round(trd,2),"Inflation":np.round(inf,2),"IntRate":np.round(rat,2)}),"GDP",["Investment","Trade","Inflation","IntRate"],"Date"
    elif scenario=="Energy & CO2 (Annual)":
        n=50;dates=pd.date_range("1975-01-01",periods=n,freq="YS");t=np.linspace(0,4,n);en=50+t*15+np.cumsum(np.random.normal(0.5,1.2,n));co=20+0.4*en+t*3+np.cumsum(np.random.normal(0.2,0.8,n));gd=200+t*40+0.5*en+np.cumsum(np.random.normal(1,3,n));rn=5+t*2+np.cumsum(np.random.normal(0.1,0.5,n));ur=40+t*8+np.cumsum(np.random.normal(0.05,0.3,n))
        return pd.DataFrame({"Date":dates,"CO2":np.round(co,2),"Energy":np.round(en,2),"GDP_pc":np.round(gd,2),"Renewable":np.round(rn,2),"Urban":np.round(ur,2)}),"CO2",["Energy","GDP_pc","Renewable","Urban"],"Date"
    elif scenario=="Tourism (Monthly)":
        n=180;dates=pd.date_range("2010-01-01",periods=n,freq="MS");t=np.linspace(0,3,n);ss=5*np.sin(2*np.pi*np.arange(n)/12);to=100+t*30+ss+np.cumsum(np.random.normal(0.3,2,n));ex=10000+t*500+np.cumsum(np.random.normal(10,50,n));cp=100+t*10+np.cumsum(np.random.normal(0.1,0.5,n));rv=50+0.3*to+t*10+ss*2+np.cumsum(np.random.normal(0.2,1.5,n))
        return pd.DataFrame({"Date":dates,"Revenue":np.round(rv,2),"Arrivals":np.round(to,2),"ExRate":np.round(ex,2),"CPI":np.round(cp,2)}),"Revenue",["Arrivals","ExRate","CPI"],"Date"
    else:
        n=100;dates=pd.date_range("2015-01-01",periods=n,freq="QS");y=np.cumsum(np.random.normal(0.5,1,n))+50;x1=np.cumsum(np.random.normal(0.3,0.8,n))+30;x2=np.cumsum(np.random.normal(0.2,1.2,n))+20;x3=np.cumsum(np.random.normal(0.1,0.6,n))+10
        return pd.DataFrame({"Date":dates,"Y":np.round(y,2),"X1":np.round(x1,2),"X2":np.round(x2,2),"X3":np.round(x3,2)}),"Y",["X1","X2","X3"],"Date"
st.markdown('<div class="main-header">ARDL Analysis Suite</div>',unsafe_allow_html=True)
st.markdown('<div class="sub-header">Autoregressive Distributed Lag | Bounds Test | ECM</div>',unsafe_allow_html=True)
with st.sidebar:
    st.markdown("## Konfigurasi")
    data_source=st.radio("Sumber Data:",["Upload CSV/Excel","Data Demo"])
    if data_source=="Data Demo":
        scenario=st.selectbox("Skenario:",["GDP & Macro (Quarterly)","Energy & CO2 (Annual)","Tourism (Monthly)","Default (Simulated)"])
        demo_df,demo_dep,demo_indeps,demo_date=generate_demo_data(scenario);st.session_state.data=demo_df;st.success(f"Demo: {len(demo_df)} obs")
    else:
        uploaded=st.file_uploader("Upload:",type=["csv","xlsx","xls"])
        if uploaded:
            try: df_up=pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded);st.session_state.data=df_up;st.success(f"{len(df_up)} rows")
            except Exception as e: st.error(str(e))
    df=st.session_state.data
    if df is not None:
        st.markdown("---");all_cols=df.columns.tolist();num_cols=df.select_dtypes(include=[np.number]).columns.tolist()
        dd,di,ddt=(demo_dep,demo_indeps,demo_date) if data_source=="Data Demo" else (None,[],None)
        date_col=st.selectbox("Date column:",all_cols,index=all_cols.index(ddt) if ddt in all_cols else 0);st.session_state.date_col=date_col
        dep_var=st.selectbox("Y (Dependent):",num_cols,index=num_cols.index(dd) if dd in num_cols else 0);st.session_state.dep_var=dep_var
        rem=[c for c in num_cols if c!=dep_var];dx=[rem.index(v) for v in di if v in rem] if di else []
        indep_vars=st.multiselect("X (Independent):",rem,default=[rem[i] for i in dx]);st.session_state.indep_vars=indep_vars
        st.markdown("---");st.markdown("### ARDL Settings")
        st.session_state.max_lag_dep=st.slider("Max lag Y:",1,12,4);st.session_state.max_lag_indep=st.slider("Max lag X:",0,12,4)
        st.session_state.ic_criterion=st.selectbox("IC:",["aic","bic","hqic"])
        st.session_state.trend=st.selectbox("Trend:",["n","c","ct","ctt"],format_func=lambda x:{"n":"None","c":"Constant","ct":"C+Trend","ctt":"C+T+T2"}[x],index=1)
        st.markdown("---");st.markdown("### Unit Root")
        ur_ml=st.selectbox("ADF max lag:",["auto","1","2","4","8","12"]);st.session_state.ur_maxlag=None if ur_ml=="auto" else int(ur_ml)
        st.session_state.ur_regression=st.selectbox("ADF type:",["c","ct","ctt","n"],format_func=lambda x:{"c":"Constant","ct":"C+Trend","ctt":"C+T+T2","n":"None"}[x])
if st.session_state.data is None: st.info("Pilih sumber data di sidebar.");st.stop()
df=st.session_state.data.copy()
if st.session_state.date_col and st.session_state.date_col in df.columns:
    try: df[st.session_state.date_col]=pd.to_datetime(df[st.session_state.date_col]);df=df.sort_values(st.session_state.date_col).reset_index(drop=True).set_index(st.session_state.date_col)
    except Exception: pass
tab1,tab2,tab3,tab4,tab5,tab6,tab7,tab8=st.tabs(["Data & EDA","Unit Root","Lag Selection","ARDL Model","Bounds Test","ECM","Diagnostics","Laporan"])
with tab1:
    st.markdown("## Data & Exploratory Analysis")
    dep=st.session_state.dep_var;indeps=st.session_state.indep_vars;allv=[dep]+indeps if dep else []
    c1,c2,c3,c4=st.columns(4)
    with c1: st.metric("Obs",len(df))
    with c2: st.metric("Vars",len(allv))
    with c3: st.metric("Missing",int(df[allv].isnull().sum().sum()) if allv else 0)
    with c4: st.metric("Freq",str(df.index.freq) if hasattr(df.index,"freq") and df.index.freq else "Unknown")
    st.dataframe(df[allv].head(20) if allv else df.head(20),use_container_width=True)
    if allv:
        desc=df[allv].describe().T;desc["Skew"]=df[allv].skew();desc["Kurt"]=df[allv].kurtosis()
        desc["JB"]=[jarque_bera(df[v].dropna())[0] for v in allv];desc["JB_p"]=[jarque_bera(df[v].dropna())[1] for v in allv]
        st.dataframe(desc.round(4),use_container_width=True)
        for v in allv: st.plotly_chart(px.line(df,y=v,title=v).update_layout(height=280,margin=dict(t=40,b=20)),use_container_width=True)
        st.plotly_chart(px.imshow(df[allv].corr(),text_auto=".3f",color_continuous_scale="RdBu_r",aspect="auto",zmin=-1,zmax=1).update_layout(height=450),use_container_width=True)
        if dep:
            yc=df[dep].dropna();nla=min(40,len(yc)//2-1)
            if nla>1:
                av=acf(yc,nlags=nla,fft=True);pv=pacf(yc,nlags=nla);cf=1.96/np.sqrt(len(yc))
                co2=st.columns(2)
                with co2[0]:
                    fa=go.Figure();fa.add_trace(go.Bar(x=list(range(len(av))),y=av,marker_color="#2E86C1",width=0.3));fa.add_hline(y=cf,line_dash="dash",line_color="red");fa.add_hline(y=-cf,line_dash="dash",line_color="red");fa.update_layout(title=f"ACF: {dep}",height=350);st.plotly_chart(fa,use_container_width=True)
                with co2[1]:
                    fp=go.Figure();fp.add_trace(go.Bar(x=list(range(len(pv))),y=pv,marker_color="#8E44AD",width=0.3));fp.add_hline(y=cf,line_dash="dash",line_color="red");fp.add_hline(y=-cf,line_dash="dash",line_color="red");fp.update_layout(title=f"PACF: {dep}",height=350);st.plotly_chart(fp,use_container_width=True)
with tab2:
    st.markdown("## Unit Root Tests")
    st.markdown('<div class="info-box"><b>ARDL: variabel harus I(0) atau I(1), TIDAK BOLEH I(2).</b></div>',unsafe_allow_html=True)
    dep=st.session_state.dep_var;indeps=st.session_state.indep_vars;allv=[dep]+indeps if dep else []
    if allv:
        if st.button("Jalankan Unit Root Tests",type="primary",use_container_width=True):
            ur={};prog=st.progress(0);tot=len(allv)*6;idx=0;ml=st.session_state.ur_maxlag;rg=st.session_state.ur_regression
            for v in allv:
                sl=df[v].dropna();sd=df[v].diff().dropna()
                ur[f"{v}_lv_ADF"]=adf_test(sl,f"{v}(Lv)",ml,rg);idx+=1;prog.progress(idx/tot)
                ur[f"{v}_lv_KPSS"]=kpss_test(sl,f"{v}(Lv)");idx+=1;prog.progress(idx/tot)
                ur[f"{v}_lv_PP"]=pp_test(sl,f"{v}(Lv)");idx+=1;prog.progress(idx/tot)
                ur[f"{v}_d1_ADF"]=adf_test(sd,f"d({v})",ml,rg);idx+=1;prog.progress(idx/tot)
                ur[f"{v}_d1_KPSS"]=kpss_test(sd,f"d({v})");idx+=1;prog.progress(idx/tot)
                ur[f"{v}_d1_PP"]=pp_test(sd,f"d({v})");idx+=1;prog.progress(idx/tot)
            st.session_state.unit_root_results=ur;prog.empty();st.success("Done!")
        if st.session_state.unit_root_results:
            ur=st.session_state.unit_root_results;st.markdown("### Level");lr=[]
            for v in allv:
                a=ur.get(f"{v}_lv_ADF",{});k=ur.get(f"{v}_lv_KPSS",{});p=ur.get(f"{v}_lv_PP",{})
                lr.append({"Var":v,"ADF":round(a.get("Statistic",0),4),"ADF_p":round(a.get("P-value",1),4),"ADF_R":"S" if a.get("Stationary") else "NS","KPSS":round(k.get("Statistic",0),4),"KPSS_p":round(k.get("P-value",0),4),"KPSS_R":"S" if k.get("Stationary") else "NS","PP":round(p.get("Statistic",0),4),"PP_p":round(p.get("P-value",1),4),"PP_R":"S" if p.get("Stationary") else "NS"})
            st.dataframe(pd.DataFrame(lr),use_container_width=True);st.markdown("### First Difference");dr=[]
            for v in allv:
                a=ur.get(f"{v}_d1_ADF",{});k=ur.get(f"{v}_d1_KPSS",{});p=ur.get(f"{v}_d1_PP",{})
                dr.append({"Var":f"d({v})","ADF":round(a.get("Statistic",0),4),"ADF_p":round(a.get("P-value",1),4),"ADF_R":"S" if a.get("Stationary") else "NS","KPSS":round(k.get("Statistic",0),4),"KPSS_p":round(k.get("P-value",0),4),"KPSS_R":"S" if k.get("Stationary") else "NS","PP":round(p.get("Statistic",0),4),"PP_p":round(p.get("P-value",1),4),"PP_R":"S" if p.get("Stationary") else "NS"})
            st.dataframe(pd.DataFrame(dr),use_container_width=True);st.markdown("### Integration Order");io_rows=[];has_i2=False
            for v in allv:
                al=ur.get(f"{v}_lv_ADF",{}).get("Stationary",False);kl=ur.get(f"{v}_lv_KPSS",{}).get("Stationary",False);ad=ur.get(f"{v}_d1_ADF",{}).get("Stationary",False);kd=ur.get(f"{v}_d1_KPSS",{}).get("Stationary",False)
                if al and kl: o="I(0)"
                elif ad and kd: o="I(1)"
                elif ad or kd: o="I(1)*"
                else: o="I(2)?";has_i2=True
                io_rows.append({"Var":v,"Order":o})
            st.dataframe(pd.DataFrame(io_rows),use_container_width=True)
            if has_i2: st.markdown('<div class="error-box"><b>WARNING:</b> I(2) terdeteksi!</div>',unsafe_allow_html=True)
            else: st.markdown('<div class="success-box"><b>Semua I(0)/I(1). ARDL OK.</b></div>',unsafe_allow_html=True)
    else: st.warning("Pilih variabel!")
with tab3:
    st.markdown("## Optimal Lag Selection")
    dep=st.session_state.dep_var;indeps=st.session_state.indep_vars
    if dep and indeps:
        if st.button("Cari Lag Optimal",type="primary",use_container_width=True):
            with st.spinner("Searching..."):
                try:
                    ys=df[dep].dropna();xd=df[indeps].dropna();ci=ys.index.intersection(xd.index);ys=ys.loc[ci];xd=xd.loc[ci]
                    sel=ardl_select_order(ys,st.session_state.max_lag_dep,xd,st.session_state.max_lag_indep,trend=st.session_state.trend,ic=st.session_state.ic_criterion)
                    st.session_state.ardl_order=sel;st.success("Done!")
                except Exception as e: st.error(str(e));import traceback;st.code(traceback.format_exc())
        if st.session_state.ardl_order is not None:
            sel=st.session_state.ardl_order;ao=sel.model.ardl_order;lbl=[dep]+indeps
            st.markdown(f'<div class="success-box"><b>ARDL({", ".join(str(x) for x in ao)})</b> by {st.session_state.ic_criterion.upper()}</div>',unsafe_allow_html=True)
            cl=st.columns(min(len(ao),6))
            for i,v in enumerate(lbl):
                if i<len(ao) and i<len(cl):
                    with cl[i]: st.metric(f"{v} lag",ao[i])
            st.markdown("### Manual Override");mc=st.columns(len(indeps)+1)
            my=mc[0].number_input("Y lag:",0,12,int(ao[0]),key="my");mx={}
            for i,v in enumerate(indeps): dl=int(ao[i+1]) if i+1<len(ao) else 1;mx[v]=mc[i+1].number_input(f"{v}:",0,12,dl,key=f"m_{v}")
            if st.button("Apply Manual"): st.session_state.manual_lags=(my,mx);st.success("Applied")
    else: st.warning("Pilih variabel!")
with tab4:
    st.markdown("## ARDL Model Estimation")
    dep=st.session_state.dep_var;indeps=st.session_state.indep_vars
    if dep and indeps:
        has_m=st.session_state.get("manual_lags") is not None;has_a=st.session_state.ardl_order is not None
        uw="Auto" if not has_m else st.radio("Use:",["Auto","Manual"],horizontal=True)
        if st.button("Estimasi ARDL",type="primary",use_container_width=True):
            with st.spinner("Estimating..."):
                try:
                    ys=df[dep].dropna();xd=df[indeps].dropna();ci=ys.index.intersection(xd.index);ys=ys.loc[ci];xd=xd.loc[ci]
                    if "Manual" in str(uw) and has_m:
                        ml=st.session_state.manual_lags;ly=int(ml[0]);od={v:int(ml[1][v]) for v in indeps}
                    elif has_a:
                        ao=st.session_state.ardl_order.model.ardl_order;ly=int(ao[0]);od={v:int(ao[i+1]) if i+1<len(ao) else 1 for i,v in enumerate(indeps)}
                    else: ly=1;od={v:1 for v in indeps}
                    am=ARDL(ys,ly,xd,od,trend=st.session_state.trend);ar=am.fit()
                    st.session_state.ardl_results=ar;fo=f"ARDL({ly}, {", ".join(str(od[v]) for v in indeps)})";st.session_state.ardl_order_str=fo;st.success(f"Model {fo} estimated!")
                except Exception as e: st.error(str(e));import traceback;st.code(traceback.format_exc())
        if st.session_state.ardl_results is not None:
            res=st.session_state.ardl_results
            st.markdown(f"### {st.session_state.get('ardl_order_str','ARDL')}")
            c1,c2,c3,c4=st.columns(4)
            with c1: st.metric("R2",f"{res.rsquared:.6f}")
            with c2: st.metric("Adj R2",f"{res.rsquared_adj:.6f}")
            with c3: st.metric("AIC",f"{res.aic:.4f}")
            with c4: st.metric("BIC",f"{res.bic:.4f}")
            c1,c2,c3,c4=st.columns(4)
            with c1: st.metric("LogLik",f"{res.llf:.4f}")
            with c2: st.metric("F",f"{res.fvalue:.4f}")
            with c3: st.metric("F_p",f"{res.f_pvalue:.6f}")
            with c4: st.metric("DW",f"{durbin_watson(res.resid):.4f}")
            cdf=pd.DataFrame({"Var":res.params.index,"Coef":res.params.values,"SE":res.bse.values,"t":res.tvalues.values,"P":res.pvalues.values,"Sig":["***" if p<0.01 else "**" if p<0.05 else "*" if p<0.1 else "" for p in res.pvalues.values]})
            st.dataframe(cdf.round(6),use_container_width=True);st.caption("*** p<0.01, ** p<0.05, * p<0.10")
            co=st.columns(2)
            with co[0]:
                ff=go.Figure();ff.add_trace(go.Scatter(x=res.fittedvalues.index,y=df[dep].loc[res.fittedvalues.index],mode="lines",name="Actual",line=dict(color="#2C3E50")));ff.add_trace(go.Scatter(x=res.fittedvalues.index,y=res.fittedvalues,mode="lines",name="Fitted",line=dict(color="#E74C3C",dash="dash")));ff.update_layout(title="Actual vs Fitted",height=400);st.plotly_chart(ff,use_container_width=True)
            with co[1]:
                fr=go.Figure();fr.add_trace(go.Scatter(x=res.resid.index,y=res.resid,mode="lines",name="Resid",line=dict(color="#27AE60")));fr.add_hline(y=0,line_dash="dash",line_color="red");fr.update_layout(title="Residuals",height=400);st.plotly_chart(fr,use_container_width=True)
            with st.expander("Full Summary"): st.text(res.summary().as_text())
    else: st.warning("Pilih variabel dan Lag Selection dulu.")
with tab5:
    st.markdown("## Pesaran Bounds Test")
    dep=st.session_state.dep_var;indeps=st.session_state.indep_vars
    if dep and indeps and st.session_state.ardl_results is not None:
        if st.button("Jalankan Bounds Test",type="primary",use_container_width=True):
            with st.spinner("Menghitung..."):
                try:
                    res=st.session_state.ardl_results;k=len(indeps);ys=df[dep].dropna();xd=df[indeps].dropna();ci=ys.index.intersection(xd.index);n=len(ci)
                    f_stat=None
                    try: bt=res.bounds_test(case=3);f_stat=float(bt.stat)
                    except Exception: pass
                    if f_stat is None: f_stat=float(res.fvalue)
                    st.session_state.bounds_results={"f_stat":f_stat,"k":k,"n":n,"cv_pesaran":get_pesaran_cv(k),"cv_narayan":narayan_cv(n,k)};st.success("Done!")
                except Exception as e: st.error(str(e))
        if st.session_state.bounds_results is not None:
            br=st.session_state.bounds_results;f_stat=br["f_stat"];st.metric("F-statistic",f"{f_stat:.4f}")
            cv_p=br["cv_pesaran"];prows=[]
            for sig in ["10%","5%","1%"]:
                lo,hi=cv_p[sig];dec="Coint." if f_stat>hi else ("No" if f_stat<lo else "Inconclusive");prows.append({"Sig":sig,"I(0)":lo,"I(1)":hi,"F":round(f_stat,4),"Decision":dec})
            st.dataframe(pd.DataFrame(prows),use_container_width=True)
            interp,btype=interpret_bounds(f_stat,cv_p);st.markdown(f'<div class="{btype}-box"><b>{interp}</b></div>',unsafe_allow_html=True)
            cv_n=br["cv_narayan"];nrows=[]
            for sig in ["10%","5%","1%"]:
                lo,hi=cv_n[sig];dec="Coint." if f_stat>hi else ("No" if f_stat<lo else "Inconclusive");nrows.append({"Sig":sig,"I(0)":lo,"I(1)":hi,"F":round(f_stat,4),"Decision":dec})
            st.markdown("### Narayan (2005)");st.dataframe(pd.DataFrame(nrows),use_container_width=True)
            fig_b=go.Figure();sigs=["10%","5%","1%"];colors=["#F39C12","#E74C3C","#8E44AD"]
            for i,sig in enumerate(sigs):
                lo,hi=cv_p[sig];fig_b.add_trace(go.Bar(name=f"I(0) {sig}",x=[sig],y=[lo],marker_color=colors[i],opacity=0.5));fig_b.add_trace(go.Bar(name=f"I(1) {sig}",x=[sig],y=[hi],marker_color=colors[i],opacity=0.9))
            fig_b.add_hline(y=f_stat,line_dash="solid",line_color="blue",line_width=3,annotation_text=f"F={f_stat:.4f}");fig_b.update_layout(barmode="group",height=450);st.plotly_chart(fig_b,use_container_width=True)
    elif dep and indeps: st.warning("Estimasi ARDL dulu di Tab 4.")
    else: st.warning("Pilih variabel!")
with tab6:
    st.markdown("## Error Correction Model")
    dep=st.session_state.dep_var;indeps=st.session_state.indep_vars
    if dep and indeps and st.session_state.ardl_results is not None:
        if st.button("Estimasi ECM",type="primary",use_container_width=True):
            with st.spinner("ECM..."):
                try:
                    ys=df[dep].dropna();xd=df[indeps].dropna();ci=ys.index.intersection(xd.index);ys=ys.loc[ci];xd=xd.loc[ci]
                    params=st.session_state.ardl_results.params;pnames=params.index.tolist()
                    y_lag_sum=sum(params[p] for p in pnames if str(p).startswith(f"{dep}.L") or str(p).startswith(f"{dep}.l"))
                    denom=1.0-y_lag_sum;lr_c={}
                    for v in indeps:
                        x_sum=sum(params[p] for p in pnames if v in str(p) and str(p)!="const" and str(p)!="trend")
                        lr_c[v]=x_sum/denom if abs(denom)>1e-10 else np.nan
                    if "const" in pnames: lr_c["Constant"]=params["const"]/denom if abs(denom)>1e-10 else np.nan
                    st.session_state.lr_coefs=lr_c
                    dy=ys.diff().dropna();dx=xd.diff().dropna();common=dy.index.intersection(dx.index);dy=dy.loc[common];dx=dx.loc[common]
                    ect=ys.shift(1).loc[common].copy()
                    for v in indeps:
                        if v in lr_c and not np.isnan(lr_c[v]): ect=ect-lr_c[v]*xd[v].shift(1).loc[common]
                    if "Constant" in lr_c and not np.isnan(lr_c["Constant"]): ect=ect-lr_c["Constant"]
                    ecm_df=pd.DataFrame({"dy":dy.values},index=common)
                    for v in indeps: ecm_df[f"d_{v}"]=dx[v].values
                    ecm_df["ECT_1"]=ect.values;ecm_df=ecm_df.dropna()
                    ecm_res=sm.OLS(ecm_df["dy"],sm.add_constant(ecm_df.drop("dy",axis=1))).fit()
                    st.session_state.ecm_results=ecm_res;st.session_state.ecm_data=ecm_df;st.success("ECM OK!")
                except Exception as e: st.error(str(e));import traceback;st.code(traceback.format_exc())
        if st.session_state.lr_coefs:
            st.markdown("### Long-Run Coefficients");st.dataframe(pd.DataFrame([{"Variable":k,"Coefficient":round(v,6)} for k,v in st.session_state.lr_coefs.items()]),use_container_width=True)
        if st.session_state.ecm_results is not None:
            ecm_res=st.session_state.ecm_results;st.markdown("### Short-Run & ECT")
            c1,c2,c3,c4=st.columns(4)
            with c1: st.metric("R2",f"{ecm_res.rsquared:.6f}")
            with c2: st.metric("Adj R2",f"{ecm_res.rsquared_adj:.6f}")
            with c3: st.metric("F",f"{ecm_res.fvalue:.4f}")
            with c4: st.metric("DW",f"{durbin_watson(ecm_res.resid):.4f}")
            ecm_cdf=pd.DataFrame({"Var":ecm_res.params.index,"Coef":ecm_res.params.values,"SE":ecm_res.bse.values,"t":ecm_res.tvalues.values,"P":ecm_res.pvalues.values,"Sig":["***" if p<0.01 else "**" if p<0.05 else "*" if p<0.1 else "" for p in ecm_res.pvalues.values]})
            st.dataframe(ecm_cdf.round(6),use_container_width=True)
            ect_row=ecm_cdf[ecm_cdf["Var"]=="ECT_1"]
            if not ect_row.empty:
                ec=float(ect_row["Coef"].iloc[0]);ep=float(ect_row["P"].iloc[0])
                c1,c2,c3=st.columns(3)
                with c1: st.metric("ECT",f"{ec:.6f}")
                with c2: st.metric("P-val",f"{ep:.6f}")
                with c3:
                    hl=abs(np.log(0.5)/np.log(1+ec)) if ec<0 and (1+ec)>0 else float("nan");st.metric("Half-life",f"{hl:.1f}" if not np.isnan(hl) else "N/A")
                if ec<0 and ep<0.05: st.markdown(f'<div class="success-box"><b>ECT={ec:.4f}</b> (p={ep:.4f}). Negatif &amp; signifikan! ~{abs(ec)*100:.1f}% koreksi/periode.</div>',unsafe_allow_html=True)
                elif ec<0: st.markdown(f'<div class="warning-box"><b>ECT={ec:.4f}</b> negatif tapi tidak signifikan.</div>',unsafe_allow_html=True)
                else: st.markdown(f'<div class="error-box"><b>ECT={ec:.4f}</b> positif.</div>',unsafe_allow_html=True)
            with st.expander("Full ECM Summary"): st.text(ecm_res.summary().as_text())
    elif dep and indeps: st.warning("Estimasi ARDL dulu.")
    else: st.warning("Pilih variabel!")
with tab7:
    st.markdown("## Diagnostic Tests")
    if st.session_state.ardl_results is not None:
        res=st.session_state.ardl_results;resid=res.resid;fitted=res.fittedvalues
        st.markdown("### Serial Correlation")
        c1,c2=st.columns(2)
        with c1:
            dw=durbin_watson(resid);st.metric("Durbin-Watson",f"{dw:.4f}")
            if 1.5<dw<2.5: st.markdown('<div class="success-box">No serial correlation (DW~2)</div>',unsafe_allow_html=True)
            else: st.markdown('<div class="warning-box">Possible serial correlation</div>',unsafe_allow_html=True)
        with c2:
            try:
                bg_lm,bg_p,bg_f,bg_fp=acorr_breusch_godfrey(res,nlags=min(4,len(resid)//5));st.metric("BG LM",f"{bg_lm:.4f}");st.metric("BG p",f"{bg_p:.6f}")
                if bg_p>0.05: st.markdown('<div class="success-box">No serial corr (BG)</div>',unsafe_allow_html=True)
                else: st.markdown('<div class="warning-box">Serial corr detected</div>',unsafe_allow_html=True)
            except Exception as e: st.warning(f"BG: {e}")
        st.markdown("### Heteroskedasticity")
        c1,c2=st.columns(2)
        with c1:
            try:
                bp_lm,bp_p,bp_f,bp_fp=het_breuschpagan(resid,res.model.exog);st.metric("BP LM",f"{bp_lm:.4f}");st.metric("BP p",f"{bp_p:.6f}")
                if bp_p>0.05: st.markdown('<div class="success-box">Homoskedastic (BP)</div>',unsafe_allow_html=True)
                else: st.markdown('<div class="warning-box">Heteroskedastic (BP)</div>',unsafe_allow_html=True)
            except Exception as e: st.warning(f"BP: {e}")
        with c2:
            try:
                from statsmodels.stats.diagnostic import het_white;wh_lm,wh_p,wh_f,wh_fp=het_white(resid,res.model.exog);st.metric("White",f"{wh_lm:.4f}");st.metric("White p",f"{wh_p:.6f}")
                if wh_p>0.05: st.markdown('<div class="success-box">Homoskedastic (White)</div>',unsafe_allow_html=True)
                else: st.markdown('<div class="warning-box">Heteroskedastic (White)</div>',unsafe_allow_html=True)
            except Exception as e: st.warning(f"White: {e}")
        st.markdown("### Normality")
        jb,jb_p,jb_sk,jb_ku=jarque_bera(resid)
        c1,c2=st.columns(2)
        with c1: st.metric("JB",f"{jb:.4f}");st.metric("JB p",f"{jb_p:.6f}")
        with c2: st.metric("Skewness",f"{jb_sk:.4f}");st.metric("Kurtosis",f"{jb_ku:.4f}")
        if jb_p>0.05: st.markdown('<div class="success-box">Normal residuals (JB)</div>',unsafe_allow_html=True)
        else: st.markdown('<div class="warning-box">Non-normal residuals</div>',unsafe_allow_html=True)
        st.markdown("### Ramsey RESET")
        try:
            from statsmodels.stats.diagnostic import linear_reset;reset=linear_reset(res,power=3,use_f=True)
            st.metric("RESET F",f"{reset.fvalue:.4f}");st.metric("RESET p",f"{reset.pvalue:.6f}")
            if reset.pvalue>0.05: st.markdown('<div class="success-box">Correct form (RESET)</div>',unsafe_allow_html=True)
            else: st.markdown('<div class="warning-box">Misspecification (RESET)</div>',unsafe_allow_html=True)
        except Exception as e: st.warning(f"RESET: {e}")
        st.markdown("### Residual Plots")
        c1,c2=st.columns(2)
        with c1: st.plotly_chart(px.histogram(x=resid,nbins=30,title="Residual Histogram",color_discrete_sequence=["#8E44AD"]).update_layout(height=400),use_container_width=True)
        with c2:
            sr=np.sort(resid);th=stats.norm.ppf(np.linspace(0.01,0.99,len(sr)));fig_qq=go.Figure()
            fig_qq.add_trace(go.Scatter(x=th,y=sr,mode="markers",marker=dict(size=4,color="#2E86C1"),name="Resid"))
            fig_qq.add_trace(go.Scatter(x=[th.min(),th.max()],y=[th.min()*resid.std()+resid.mean(),th.max()*resid.std()+resid.mean()],mode="lines",line=dict(color="red",dash="dash"),name="Normal"))
            fig_qq.update_layout(title="Q-Q Plot",height=400);st.plotly_chart(fig_qq,use_container_width=True)
        c1,c2=st.columns(2)
        with c1:
            fig_rf=px.scatter(x=fitted,y=resid,title="Residuals vs Fitted",labels={"x":"Fitted","y":"Residual"},color_discrete_sequence=["#27AE60"])
            fig_rf.add_hline(y=0,line_dash="dash",line_color="red");st.plotly_chart(fig_rf,use_container_width=True)
        with c2:
            nlr=min(30,len(resid)//3)
            if nlr>1:
                acf_r=acf(resid,nlags=nlr,fft=True);fig_ar=go.Figure();fig_ar.add_trace(go.Bar(x=list(range(len(acf_r))),y=acf_r,marker_color="#E74C3C",width=0.3))
                fig_ar.add_hline(y=1.96/np.sqrt(len(resid)),line_dash="dash",line_color="blue");fig_ar.add_hline(y=-1.96/np.sqrt(len(resid)),line_dash="dash",line_color="blue")
                fig_ar.update_layout(title="ACF Residuals",height=400);st.plotly_chart(fig_ar,use_container_width=True)
        st.markdown("### CUSUM & CUSUMSQ")
        try:
            rv=resid.values;ncu=len(rv);sig_cu=np.std(rv,ddof=1);cusum=np.cumsum(rv)/sig_cu;cusum_sq=np.cumsum(rv**2)/np.sum(rv**2)
            t_v=np.arange(1,ncu+1);up_cu=0.948*np.sqrt(ncu)+2*0.948*t_v/np.sqrt(ncu);lo_cu=-up_cu
            exp_sq=t_v/ncu;up_sq=exp_sq+1.63/np.sqrt(ncu);lo_sq=exp_sq-1.63/np.sqrt(ncu)
            c1,c2=st.columns(2)
            with c1:
                fig_cu=go.Figure();fig_cu.add_trace(go.Scatter(y=cusum,mode="lines",name="CUSUM",line=dict(color="#2E86C1")));fig_cu.add_trace(go.Scatter(y=up_cu,mode="lines",name="Upper",line=dict(color="red",dash="dash")));fig_cu.add_trace(go.Scatter(y=lo_cu,mode="lines",name="Lower",line=dict(color="red",dash="dash")))
                fig_cu.update_layout(title="CUSUM",height=400);st.plotly_chart(fig_cu,use_container_width=True)
            with c2:
                fig_cs=go.Figure();fig_cs.add_trace(go.Scatter(y=cusum_sq,mode="lines",name="CUSUMSQ",line=dict(color="#8E44AD")));fig_cs.add_trace(go.Scatter(y=up_sq,mode="lines",name="Upper",line=dict(color="red",dash="dash")));fig_cs.add_trace(go.Scatter(y=lo_sq,mode="lines",name="Lower",line=dict(color="red",dash="dash")))
                fig_cs.update_layout(title="CUSUMSQ",height=400);st.plotly_chart(fig_cs,use_container_width=True)
            w_cu=bool(np.all((cusum>=lo_cu)&(cusum<=up_cu)));w_sq=bool(np.all((cusum_sq>=np.maximum(lo_sq,0))&(cusum_sq<=np.minimum(up_sq,1))))
            if w_cu and w_sq: st.markdown('<div class="success-box"><b>CUSUM &amp; CUSUMSQ within bounds. Model stabil.</b></div>',unsafe_allow_html=True)
            else: st.markdown('<div class="warning-box"><b>Instabilitas terdeteksi.</b></div>',unsafe_allow_html=True)
        except Exception as e: st.warning(f"CUSUM: {e}")
    else: st.warning("Estimasi ARDL terlebih dahulu!")
with tab8:
    st.markdown("## Laporan & Export")
    dep=st.session_state.dep_var;indeps=st.session_state.indep_vars
    if dep and indeps:
        rl=["="*60,"ARDL ANALYSIS REPORT","="*60,f"Generated: {datetime.now():%Y-%m-%d %H:%M}",f"Dependent: {dep}",f"Independent: {", ".join(indeps)}",f"N: {len(df)}",""]
        if st.session_state.ardl_results is not None:
            r=st.session_state.ardl_results;rl+=["ARDL MODEL",f"  Order: {st.session_state.get('ardl_order_str','')}", f"  R2={r.rsquared:.6f}  AdjR2={r.rsquared_adj:.6f}",f"  AIC={r.aic:.4f}  BIC={r.bic:.4f}",f"  F={r.fvalue:.4f}  DW={durbin_watson(r.resid):.4f}","  Coefficients:"]
            for pn,pv in zip(r.params.index,r.params.values): rl.append(f"    {str(pn):25s} {pv:>12.6f}  p={r.pvalues[pn]:.4f}")
            rl.append("")
        if st.session_state.bounds_results is not None:
            br=st.session_state.bounds_results;rl+=["BOUNDS TEST",f"  F={br['f_stat']:.4f}  k={br['k']}  n={br['n']}"]
            for sig in ["10%","5%","1%"]: lo,hi=br["cv_pesaran"][sig];rl.append(f"    {sig}: I(0)={lo:.2f}  I(1)={hi:.2f}")
            rl.append("")
        if st.session_state.lr_coefs:
            rl.append("LONG-RUN COEFFICIENTS")
            for k,v in st.session_state.lr_coefs.items(): rl.append(f"  {k:25s} {v:>12.6f}")
            rl.append("")
        if st.session_state.ecm_results is not None:
            e=st.session_state.ecm_results;rl+=["ECM (SHORT-RUN)",f"  R2={e.rsquared:.6f}"]
            for pn,pv in zip(e.params.index,e.params.values): rl.append(f"    {str(pn):25s} {pv:>12.6f}  p={e.pvalues[pn]:.4f}")
            rl.append("")
        rl+=["="*60,"*** p<0.01  ** p<0.05  * p<0.10"]
        rt="\n".join(rl);st.text_area("Preview",rt,height=400)
        st.download_button("Download Report (.txt)",rt,f"ARDL_Report_{datetime.now():%Y%m%d_%H%M}.txt","text/plain",use_container_width=True)
        cfg={"analysis":"ARDL","timestamp":datetime.now().isoformat(),"dep":dep,"indep":indeps,"n":len(df)}
        if st.session_state.ardl_results: r=st.session_state.ardl_results;cfg["ardl"]={"order":st.session_state.get("ardl_order_str",""),"r2":float(r.rsquared),"aic":float(r.aic),"bic":float(r.bic)}
        if st.session_state.bounds_results: cfg["bounds"]={"f_stat":float(st.session_state.bounds_results["f_stat"]),"k":st.session_state.bounds_results["k"]}
        if st.session_state.lr_coefs: cfg["long_run"]={k:float(v) for k,v in st.session_state.lr_coefs.items() if not np.isnan(v)}
        st.download_button("Download Config (.json)",json.dumps(cfg,indent=2,default=str),f"ARDL_Config_{datetime.now():%Y%m%d_%H%M}.json","application/json",use_container_width=True)
        if st.session_state.ardl_results is not None:
            edf=df.copy();r=st.session_state.ardl_results;edf["ARDL_Fitted"]=np.nan;edf.loc[r.fittedvalues.index,"ARDL_Fitted"]=r.fittedvalues.values
            edf["ARDL_Resid"]=np.nan;edf.loc[r.resid.index,"ARDL_Resid"]=r.resid.values
            buf=io.StringIO();edf.to_csv(buf)
            st.download_button("Download Data (.csv)",buf.getvalue(),f"ARDL_Data_{datetime.now():%Y%m%d_%H%M}.csv","text/csv",use_container_width=True)
    else: st.warning("Pilih variabel!")
st.markdown("---")
st.markdown('<div style="text-align:center;color:#888;font-size:0.85rem">ARDL Analysis Suite | Pesaran, Shin &amp; Smith (2001) | statsmodels</div>',unsafe_allow_html=True)