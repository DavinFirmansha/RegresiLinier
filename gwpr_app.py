# ============================================================
# GWPR Analysis Suite - Geographically Weighted Poisson Regression
# File: gwpr_app.py | Jalankan: streamlit run gwpr_app.py
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
import warnings, io, json
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

st.set_page_config(page_title="GWPR Analysis Suite", page_icon="\U0001f9a0", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.main-header{font-size:2.5rem;font-weight:700;color:#1B4F72!important;text-align:center;padding:1rem 0;border-bottom:3px solid #8E44AD;margin-bottom:1rem}
.sub-header{font-size:1.2rem;color:#5D6D7E!important;text-align:center;margin-bottom:2rem}
.success-box{background:#D5F5E3;border-left:5px solid #27AE60;padding:1rem;border-radius:5px;margin:.5rem 0;color:#1a5c2e!important}
.success-box b{color:#1a5c2e!important}
.warning-box{background:#FEF9E7;border-left:5px solid #F39C12;padding:1rem;border-radius:5px;margin:.5rem 0;color:#7d5a00!important}
.warning-box b{color:#7d5a00!important}
.error-box{background:#FADBD8;border-left:5px solid #E74C3C;padding:1rem;border-radius:5px;margin:.5rem 0;color:#922b21!important}
.error-box b{color:#922b21!important}
.info-box{background:#D6EAF8;border-left:5px solid #2E86C1;padding:1rem;border-radius:5px;margin:.5rem 0;color:#1a4971!important}
.info-box b{color:#1a4971!important}
</style>
""", unsafe_allow_html=True)

for k, v in {'data':None,'poisson_results':None,'gwpr_results':None,'dep_var':None,'indep_vars':[],'lon_col':None,'lat_col':None,'gwpr_bw':None,'offset_col':None,'poisson_aicc':None}.items():
    if k not in st.session_state:
        st.session_state[k] = v

def create_colormap_safe(series, cmap_name='RdYlBu', n_bins=7):
    vmin, vmax = float(series.min()), float(series.max())
    if vmin == vmax: vmin -= 1; vmax += 1
    rgba = plt.get_cmap(cmap_name)(np.linspace(0, 1, n_bins))
    return cm.LinearColormap([mcolors.to_hex(c) for c in rgba], vmin=vmin, vmax=vmax)

def compute_poisson_deviance(y, mu):
    y = np.asarray(y, dtype=float).flatten()
    mu = np.maximum(np.asarray(mu, dtype=float).flatten(), 1e-10)
    d = np.zeros_like(y); mask = y > 0
    d[mask] = y[mask] * np.log(y[mask] / mu[mask])
    return 2 * np.sum(d - (y - mu))

def compute_poisson_pseudo_r2(y, mu):
    from scipy.special import gammaln
    y = np.asarray(y, dtype=float).flatten()
    mu = np.maximum(np.asarray(mu, dtype=float).flatten(), 1e-10)
    ys = np.maximum(y, 0)
    ll_m = np.sum(ys * np.log(mu) - mu - gammaln(ys + 1))
    mn = max(np.mean(ys), 1e-10)
    ll_n = np.sum(ys * np.log(mn) - mn - gammaln(ys + 1))
    return 1 - ll_m / ll_n if ll_n != 0 else 0.0

def calc_deviance_residuals(y, mu):
    """Manual deviance residuals: sign(y-mu)*sqrt(2*(y*log(y/mu)-(y-mu)))"""
    y = np.asarray(y, dtype=float).flatten()
    mu = np.maximum(np.asarray(mu, dtype=float).flatten(), 1e-10)
    term = np.where(y > 0, y * np.log(y / mu), 0.0) - (y - mu)
    return np.sign(y - mu) * np.sqrt(np.maximum(2 * term, 0))

def calc_pearson_residuals(y, mu):
    """Manual Pearson residuals: (y-mu)/sqrt(mu)"""
    y = np.asarray(y, dtype=float).flatten()
    mu = np.maximum(np.asarray(mu, dtype=float).flatten(), 1e-10)
    return (y - mu) / np.sqrt(mu)

def generate_demo_data(scenario):
    np.random.seed(42)
    if scenario == "Kasus DBD (Dengue Fever)":
        n=150; lon=np.random.uniform(110.3,114.6,n); lat=np.random.uniform(-8.2,-6.8,n)
        kp=np.random.uniform(500,20000,n); pk=np.random.uniform(0.5,10,n); sn=np.random.uniform(20,95,n)
        b0=2.5+0.5*np.sin((lon-112)*2)+np.random.normal(0,0.1,n)
        b1=0.00005+0.00003*(lat+7.5)+np.random.normal(0,0.00001,n)
        b2=-0.08+0.03*np.cos(lon*2)+np.random.normal(0,0.01,n)
        b3=-0.005+0.003*(lon-112)+np.random.normal(0,0.001,n)
        lm=np.clip(b0+b1*kp+b2*pk+b3*sn,0,6); y=np.random.poisson(np.exp(lm))
        df=pd.DataFrame({'ID':[f'Kec_{i+1}' for i in range(n)],'Longitude':np.round(lon,4),'Latitude':np.round(lat,4),'Kasus_DBD':y,'Kepadatan_km2':np.round(kp,0),'Rasio_Puskesmas':np.round(pk,2),'Pct_Sanitasi':np.round(sn,1)})
        return df,"Kasus_DBD",["Kepadatan_km2","Rasio_Puskesmas","Pct_Sanitasi"],"Longitude","Latitude"
    elif scenario == "Kematian Bayi (Infant Mortality)":
        n=120; lon=np.random.uniform(106.6,107.1,n); lat=np.random.uniform(-6.9,-6.1,n)
        bd=np.random.uniform(1,20,n); ms=np.random.uniform(5,40,n); im=np.random.uniform(50,99,n)
        b0=1.5+0.3*(lon-106.85)*5+np.random.normal(0,0.1,n)
        b1=-0.05+0.02*(lat+6.5)+np.random.normal(0,0.005,n)
        b2=0.02+0.01*np.sin(lon*30)+np.random.normal(0,0.003,n)
        b3=-0.01+0.005*(lon-106.85)+np.random.normal(0,0.002,n)
        lm=np.clip(b0+b1*bd+b2*ms+b3*im,0,5); y=np.random.poisson(np.exp(lm))
        df=pd.DataFrame({'ID':[f'Kel_{i+1}' for i in range(n)],'Longitude':np.round(lon,6),'Latitude':np.round(lat,6),'Kematian_Bayi':y,'Jumlah_Bidan':np.round(bd,1),'Pct_Miskin':np.round(ms,2),'Pct_Imunisasi':np.round(im,1)})
        return df,"Kematian_Bayi",["Jumlah_Bidan","Pct_Miskin","Pct_Imunisasi"],"Longitude","Latitude"
    elif scenario == "Kecelakaan Lalu Lintas":
        n=180; lon=np.random.uniform(106.6,107.0,n); lat=np.random.uniform(-6.95,-6.1,n)
        pj=np.random.uniform(5,200,n); kk=np.random.uniform(100,5000,n); tl=np.random.uniform(0,30,n)
        b0=2.0+0.4*(lat+6.5)+np.random.normal(0,0.1,n)
        b1=0.005+0.002*np.sin(lon*30)+np.random.normal(0,0.001,n)
        b2=0.0002+0.0001*(lon-106.8)*10+np.random.normal(0,0.00005,n)
        b3=-0.03+0.01*(lat+6.5)+np.random.normal(0,0.005,n)
        lm=np.clip(b0+b1*pj+b2*kk+b3*tl,0,6); y=np.random.poisson(np.exp(lm))
        df=pd.DataFrame({'ID':[f'Ruas_{i+1}' for i in range(n)],'Longitude':np.round(lon,6),'Latitude':np.round(lat,6),'Jml_Kecelakaan':y,'Panjang_Jalan_km':np.round(pj,1),'Kepadatan_Kend':np.round(kk,0),'Jml_Traffic_Light':np.round(tl,0)})
        return df,"Jml_Kecelakaan",["Panjang_Jalan_km","Kepadatan_Kend","Jml_Traffic_Light"],"Longitude","Latitude"
    else:
        n=100; lon=np.random.uniform(0,10,n); lat=np.random.uniform(0,10,n)
        x1=np.random.uniform(1,50,n); x2=np.random.uniform(0,30,n)
        b0=1.5+0.2*lon+np.random.normal(0,0.1,n); b1=0.02+0.01*lat+np.random.normal(0,0.005,n)
        b2=-0.01+0.005*lon+np.random.normal(0,0.003,n)
        lm=np.clip(b0+b1*x1+b2*x2,0,6); y=np.random.poisson(np.exp(lm))
        df=pd.DataFrame({'ID':[f'Loc_{i+1}' for i in range(n)],'Longitude':np.round(lon,4),'Latitude':np.round(lat,4),'Y_Count':y,'X1':np.round(x1,2),'X2':np.round(x2,2)})
        return df,"Y_Count",["X1","X2"],"Longitude","Latitude"

st.markdown('<div class="main-header">\U0001f9a0 GWPR Analysis Suite</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Geographically Weighted Poisson Regression</div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## Konfigurasi")
    data_source = st.radio("Sumber Data:", ["Upload CSV/Excel", "Data Demo"])
    if data_source == "Data Demo":
        scenario = st.selectbox("Skenario:", ["Kasus DBD (Dengue Fever)","Kematian Bayi (Infant Mortality)","Kecelakaan Lalu Lintas","Default (Simulated)"])
        demo_df, demo_dep, demo_indeps, demo_lon, demo_lat = generate_demo_data(scenario)
        st.session_state.data = demo_df
        st.success("Data demo dimuat: " + str(len(demo_df)) + " obs")
    else:
        uploaded = st.file_uploader("Upload CSV/Excel:", type=['csv','xlsx','xls'])
        if uploaded:
            try:
                if uploaded.name.endswith('.csv'):
                    sep = st.selectbox("Separator:", [",",";","\\t","|"])
                    df_up = pd.read_csv(uploaded, sep=sep)
                else:
                    df_up = pd.read_excel(uploaded)
                st.session_state.data = df_up
                st.success(str(len(df_up)) + " baris dimuat")
            except Exception as e:
                st.error(str(e))
    df = st.session_state.data
    if df is not None:
        st.markdown("---")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if data_source == "Data Demo":
            dd, di, dl, da = demo_dep, demo_indeps, demo_lon, demo_lat
        else:
            dd, di, dl, da = None, [], None, None
        dep_var = st.selectbox("Y (count):", numeric_cols, index=numeric_cols.index(dd) if dd in numeric_cols else 0)
        st.session_state.dep_var = dep_var
        remaining = [c for c in numeric_cols if c != dep_var]
        dx = [remaining.index(v) for v in di if v in remaining] if di else []
        indep_vars = st.multiselect("X:", remaining, default=[remaining[i] for i in dx])
        st.session_state.indep_vars = indep_vars
        lon_col = st.selectbox("Longitude:", numeric_cols, index=numeric_cols.index(dl) if dl in numeric_cols else 0)
        lat_col = st.selectbox("Latitude:", numeric_cols, index=numeric_cols.index(da) if da in numeric_cols else 0)
        st.session_state.lon_col = lon_col; st.session_state.lat_col = lat_col
        offset_opts = ["None"] + numeric_cols
        offset_sel = st.selectbox("Offset:", offset_opts)
        st.session_state.offset_col = None if offset_sel == "None" else offset_sel
        st.markdown("---")
        kernel_type = st.selectbox("Kernel:", ["bisquare","gaussian","exponential"])
        bw_fixed = st.checkbox("Fixed bandwidth", False)
        criterion = st.selectbox("Criterion:", ["AICc","AIC","BIC","CV"])
        search_method = st.selectbox("Search:", ["golden_section","interval"])
        st.session_state.kernel_type = kernel_type; st.session_state.bw_fixed = bw_fixed
        st.session_state.criterion = criterion; st.session_state.search_method = search_method

if st.session_state.data is None:
    st.info("Pilih sumber data di sidebar untuk memulai.")
    st.stop()

df = st.session_state.data
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Data Explorer","Uji Asumsi","Poisson GLM","GWPR Model","Perbandingan","Peta","Laporan"])

# TAB 1
with tab1:
    st.markdown("## Data Explorer")
    c1,c2,c3,c4 = st.columns(4)
    with c1: st.metric("Observasi", len(df))
    with c2: st.metric("Variabel", len(df.columns))
    with c3: st.metric("Missing", df.isnull().sum().sum())
    with c4: st.metric("Duplikat", df.duplicated().sum())
    st.dataframe(df.head(20), use_container_width=True)
    st.dataframe(df.describe().round(4), use_container_width=True)
    if st.session_state.dep_var:
        dep = st.session_state.dep_var; yv = df[dep].dropna()
        c1,c2,c3,c4,c5 = st.columns(5)
        dr = yv.var()/yv.mean() if yv.mean()>0 else 0
        with c1: st.metric("Mean",f"{yv.mean():.2f}")
        with c2: st.metric("Var",f"{yv.var():.2f}")
        with c3: st.metric("Min",int(yv.min()))
        with c4: st.metric("Max",int(yv.max()))
        with c5: st.metric("Var/Mean",f"{dr:.2f}")
        c1,c2 = st.columns(2)
        with c1: st.plotly_chart(px.histogram(df,x=dep,nbins=30,title=f"Distribusi {dep}",marginal="box",color_discrete_sequence=["#8E44AD"]),use_container_width=True)
        with c2:
            from scipy.stats import poisson
            xr=np.arange(0,int(yv.max())+1); pmf=poisson.pmf(xr,yv.mean())
            of=yv.value_counts().sort_index(); ofn=of/of.sum()
            fc=go.Figure()
            fc.add_trace(go.Bar(x=ofn.index,y=ofn.values,name="Observed",marker_color="#8E44AD",opacity=0.7))
            fc.add_trace(go.Scatter(x=xr,y=pmf,mode='lines+markers',name="Poisson",line=dict(color='red',width=2)))
            fc.update_layout(title="Observed vs Poisson",xaxis_title=dep,yaxis_title="Prob")
            st.plotly_chart(fc,use_container_width=True)
        if dr>1.5: st.markdown(f'<div class="warning-box">Overdispersi: Var/Mean={dr:.2f}</div>',unsafe_allow_html=True)
        elif dr<0.5: st.markdown(f'<div class="info-box">Underdispersi: Var/Mean={dr:.2f}</div>',unsafe_allow_html=True)
        else: st.markdown(f'<div class="success-box">Dispersi wajar: Var/Mean={dr:.2f}</div>',unsafe_allow_html=True)
    if st.session_state.lon_col and st.session_state.lat_col:
        fm=px.scatter_mapbox(df,lon=st.session_state.lon_col,lat=st.session_state.lat_col,color=st.session_state.dep_var,size_max=15,zoom=6,mapbox_style="open-street-map",color_continuous_scale="Viridis")
        fm.update_layout(height=500); st.plotly_chart(fm,use_container_width=True)
    nc=df.select_dtypes(include=[np.number]).columns.tolist()
    cv=[st.session_state.dep_var]+st.session_state.indep_vars if st.session_state.dep_var else nc
    cv=[v for v in cv if v in df.columns]
    st.plotly_chart(px.imshow(df[cv].corr(),text_auto=".3f",color_continuous_scale="RdBu_r",aspect="auto",zmin=-1,zmax=1).update_layout(height=500),use_container_width=True)

# TAB 2
with tab2:
    st.markdown("## Uji Asumsi Poisson")
    if st.session_state.dep_var and st.session_state.indep_vars:
        dep=st.session_state.dep_var; indeps=st.session_state.indep_vars
        yd=df[dep].values; Xd=df[indeps].values
        from scipy.stats import chisquare, poisson, kstest
        of=pd.Series(yd).value_counts().sort_index(); lam=yd.mean()
        ef=np.maximum(np.array([poisson.pmf(k,lam)*len(yd) for k in of.index]),1e-10)
        c1,c2,c3=st.columns(3)
        with c1:
            st.markdown("**Chi-Square GoF**")
            try:
                cs,cp=chisquare(of.values,f_exp=ef)
                st.write(f"Stat={cs:.4f}, P={cp:.6f}")
                if cp>0.05: st.markdown('<div class="success-box">Sesuai Poisson</div>',unsafe_allow_html=True)
                else: st.markdown('<div class="warning-box">Tidak sesuai</div>',unsafe_allow_html=True)
            except Exception as e: st.warning(str(e))
        with c2:
            st.markdown("**KS Test**")
            try:
                ks,kp=kstest(yd,'poisson',args=(lam,))
                st.write(f"Stat={ks:.6f}, P={kp:.6f}")
                if kp>0.05: st.markdown('<div class="success-box">Sesuai</div>',unsafe_allow_html=True)
                else: st.markdown('<div class="warning-box">Tidak sesuai</div>',unsafe_allow_html=True)
            except Exception as e: st.warning(str(e))
        with c3:
            st.markdown("**Zero-Inflation**")
            nz=np.sum(yd==0); pz=100*nz/len(yd); ez=100*poisson.pmf(0,lam)
            st.write(f"Zeros={nz} ({pz:.1f}%), Expected={ez:.1f}%")
            if pz>ez*2: st.markdown('<div class="warning-box">Zero-inflated</div>',unsafe_allow_html=True)
            else: st.markdown('<div class="success-box">Wajar</div>',unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("### Overdispersi")
        Xc=sm.add_constant(Xd)
        try:
            pg=smPoisson(yd,Xc).fit(disp=0); mh=pg.predict(Xc)
            pr=calc_pearson_residuals(yd,mh); pchi=np.sum(pr**2); dp=pchi/(len(yd)-len(pg.params))
            c1,c2=st.columns(2)
            with c1:
                st.write(f"Pearson Chi2={pchi:.4f}, df={len(yd)-len(pg.params)}, Disp={dp:.4f}")
                if dp<1.5: st.markdown('<div class="success-box">Equidispersion</div>',unsafe_allow_html=True)
                elif dp<3: st.markdown('<div class="warning-box">Mild overdispersion</div>',unsafe_allow_html=True)
                else: st.markdown('<div class="error-box">Severe overdispersion</div>',unsafe_allow_html=True)
            with c2:
                try:
                    ay=((yd-mh)**2-yd)/mh; am=sm.OLS(ay,sm.add_constant(mh)).fit()
                    st.write(f"Dean t={am.tvalues[1]:.4f}, P={am.pvalues[1]:.6f}")
                    if am.pvalues[1]>0.05: st.markdown('<div class="success-box">OK</div>',unsafe_allow_html=True)
                    else: st.markdown('<div class="warning-box">Overdispersi signifikan</div>',unsafe_allow_html=True)
                except Exception as e: st.warning(str(e))
        except Exception as e: st.error(str(e))
        st.markdown("---")
        st.markdown("### VIF")
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        Xc=sm.add_constant(Xd)
        vdf=pd.DataFrame({'Variabel':indeps,'VIF':[variance_inflation_factor(Xc,i+1) for i in range(len(indeps))]})
        vdf['Status']=vdf['VIF'].apply(lambda x:'OK' if x<5 else ('Moderat' if x<10 else 'Tinggi'))
        st.dataframe(vdf,use_container_width=True)
        fv=px.bar(vdf,x='Variabel',y='VIF',color='VIF',color_continuous_scale='RdYlGn_r',text='VIF')
        fv.add_hline(y=5,line_dash="dash",line_color="orange"); fv.add_hline(y=10,line_dash="dash",line_color="red")
        fv.update_traces(texttemplate='%{text:.2f}',textposition='outside')
        st.plotly_chart(fv,use_container_width=True)
        st.markdown("---")
        st.markdown("### Moran's I")
        try:
            from libpysal.weights import KNN; from esda.moran import Moran
            ca=np.array(list(zip(df[st.session_state.lon_col].values,df[st.session_state.lat_col].values)))
            w=KNN.from_array(ca,k=8); w.transform='r'
            mi=Moran(pr,w)
            st.write(f"I={mi.I:.6f}, Z={mi.z_norm:.4f}, P={mi.p_norm:.6f}")
            if mi.p_norm<0.05: st.markdown('<div class="warning-box">Autokorelasi spasial! GWPR diperlukan.</div>',unsafe_allow_html=True)
            else: st.markdown('<div class="success-box">Tidak ada autokorelasi spasial</div>',unsafe_allow_html=True)
        except ImportError: st.warning("Install libpysal dan esda")
        except Exception as e: st.warning(str(e))
    else: st.warning("Pilih variabel terlebih dahulu!")

# TAB 3
with tab3:
    st.markdown("## Poisson Global (GLM)")
    if st.session_state.dep_var and st.session_state.indep_vars:
        dep=st.session_state.dep_var; indeps=st.session_state.indep_vars
        yp=df[dep].values; Xp=df[indeps].values; Xc=sm.add_constant(Xp)
        oa=None
        if st.session_state.offset_col and st.session_state.offset_col in df.columns:
            oa=np.log(np.maximum(df[st.session_state.offset_col].values,1e-10))
        try:
            if oa is not None: pm=smPoisson(yp,Xc,offset=oa).fit(disp=0)
            else: pm=smPoisson(yp,Xc).fit(disp=0)
            st.session_state.poisson_results=pm; mh=pm.predict(Xc)
            pr2=compute_poisson_pseudo_r2(yp,mh); dev=compute_poisson_deviance(yp,mh)
            pchi=np.sum((yp-mh)**2/np.maximum(mh,1e-10))
            n=len(yp); k=len(pm.params); aicc=pm.aic+(2*k*(k+1))/max(n-k-1,1); disp=pchi/(n-k)
            st.session_state.poisson_aicc=aicc
            c1,c2,c3,c4=st.columns(4)
            with c1: st.metric("Pseudo R2",f"{pr2:.6f}")
            with c2: st.metric("AIC",f"{pm.aic:.4f}")
            with c3: st.metric("AICc",f"{aicc:.4f}")
            with c4: st.metric("BIC",f"{pm.bic:.4f}")
            c1,c2,c3,c4=st.columns(4)
            with c1: st.metric("Deviance",f"{dev:.4f}")
            with c2: st.metric("Pearson Chi2",f"{pchi:.4f}")
            with c3: st.metric("Log-Lik",f"{pm.llf:.4f}")
            with c4: st.metric("Dispersion",f"{disp:.4f}")
            st.markdown("### Koefisien")
            ci=pm.conf_int()
            cilo=ci.iloc[:,0].values if hasattr(ci,'iloc') else ci[:,0]
            cihi=ci.iloc[:,1].values if hasattr(ci,'iloc') else ci[:,1]
            cdf=pd.DataFrame({'Variable':['Intercept']+indeps,'Beta':pm.params.flatten(),'IRR':np.exp(pm.params.flatten()),'SE':pm.bse.flatten(),'z':pm.tvalues.flatten(),'P':pm.pvalues.flatten(),'CI_lo':cilo,'CI_hi':cihi,'Sig':['Yes' if p<0.05 else 'No' for p in pm.pvalues.flatten()]})
            st.dataframe(cdf.round(6),use_container_width=True)
            st.markdown("### IRR Forest Plot")
            idf=cdf[cdf['Variable']!='Intercept'].copy()
            idf['IRR_lo']=np.exp(idf['CI_lo']); idf['IRR_hi']=np.exp(idf['CI_hi'])
            fi=go.Figure()
            fi.add_trace(go.Scatter(x=idf['IRR'],y=idf['Variable'],mode='markers',marker=dict(size=12,color='#8E44AD'),error_x=dict(type='data',symmetric=False,array=idf['IRR_hi']-idf['IRR'],arrayminus=idf['IRR']-idf['IRR_lo']),name='IRR'))
            fi.add_vline(x=1,line_dash="dash",line_color="red")
            fi.update_layout(title="IRR + 95% CI",xaxis_title="IRR",height=400)
            st.plotly_chart(fi,use_container_width=True)
            st.markdown("### Persamaan")
            pp=pm.params.flatten()
            eq="log(mu) = " + f"{pp[0]:.4f}"
            for i,v in enumerate(indeps):
                s="+" if pp[i+1]>=0 else ""
                eq+=f" {s}{pp[i+1]:.4f}*{v}"
            st.markdown(f"**{eq}**")
            with st.expander("Full Summary"): st.text(pm.summary().as_text())
            st.markdown("### Diagnostics")
            dev_resid = calc_deviance_residuals(yp, mh)
            pear_resid = calc_pearson_residuals(yp, mh)
            c1,c2=st.columns(2)
            with c1:
                ff=px.scatter(x=mh,y=yp,title="Actual vs Predicted",labels={'x':'Predicted','y':'Actual'},color_discrete_sequence=["#8E44AD"])
                ff.add_trace(go.Scatter(x=[min(mh.min(),yp.min()),max(mh.max(),yp.max())],y=[min(mh.min(),yp.min()),max(mh.max(),yp.max())],mode='lines',line=dict(color='red',dash='dash'),name='Perfect'))
                st.plotly_chart(ff,use_container_width=True)
            with c2:
                fd=px.scatter(x=mh,y=dev_resid,title="Deviance Resid",labels={'x':'Predicted','y':'Dev Resid'},color_discrete_sequence=["#E74C3C"])
                fd.add_hline(y=0,line_dash="dash"); fd.add_hline(y=2,line_dash="dot",line_color="orange"); fd.add_hline(y=-2,line_dash="dot",line_color="orange")
                st.plotly_chart(fd,use_container_width=True)
            c1,c2=st.columns(2)
            with c1: st.plotly_chart(px.histogram(x=pear_resid,nbins=30,title="Pearson Resid",marginal="box",color_discrete_sequence=["#27AE60"]),use_container_width=True)
            with c2:
                from scipy.stats import poisson as pdist
                cr=np.arange(0,min(int(yp.max())+1,50))
                oc=np.array([np.sum(yp==c) for c in cr]); ec=np.array([pdist.pmf(c,mh.mean())*len(yp) for c in cr])
                fr=go.Figure()
                fr.add_trace(go.Bar(x=cr,y=oc,name='Obs',marker_color='#8E44AD',opacity=0.7))
                fr.add_trace(go.Scatter(x=cr,y=ec,name='Expected',mode='lines+markers',line=dict(color='red')))
                fr.update_layout(title="Obs vs Expected")
                st.plotly_chart(fr,use_container_width=True)
        except Exception as e:
            st.error(str(e)); import traceback; st.code(traceback.format_exc())
    else: st.warning("Pilih variabel!")

# TAB 4
with tab4:
    st.markdown("## GWPR Model")
    if st.session_state.dep_var and st.session_state.indep_vars:
        dep=st.session_state.dep_var; indeps=st.session_state.indep_vars
        ygw=df[dep].values.reshape(-1,1); Xgw=df[indeps].values
        ogw=None
        if st.session_state.offset_col and st.session_state.offset_col in df.columns:
            ogw=np.log(np.maximum(df[st.session_state.offset_col].values.reshape(-1,1),1e-10))
        if st.button("Jalankan GWPR",use_container_width=True,type="primary"):
            with st.spinner("Estimasi GWPR..."):
                try:
                    from mgwr.gwr import GWR; from mgwr.sel_bw import Sel_BW; from spglm.family import Poisson
                    ca=np.array(list(zip(df[st.session_state.lon_col].values,df[st.session_state.lat_col].values)))
                    sel=Sel_BW(ca,ygw,Xgw,family=Poisson(),offset=ogw,kernel=st.session_state.kernel_type,fixed=st.session_state.bw_fixed)
                    bw=sel.search(criterion=st.session_state.criterion,search_method=st.session_state.search_method)
                    st.session_state.gwpr_bw=bw
                    gm=GWR(ca,ygw,Xgw,bw=bw,family=Poisson(),offset=ogw,kernel=st.session_state.kernel_type,fixed=st.session_state.bw_fixed)
                    gr=gm.fit(); st.session_state.gwpr_results=gr
                    st.success("GWPR OK! Bandwidth=" + str(bw))
                except Exception as e:
                    st.error(str(e)); import traceback; st.code(traceback.format_exc())
        if st.session_state.gwpr_results is not None:
            gr=st.session_state.gwpr_results; bw=st.session_state.gwpr_bw
            ya=df[dep].values; mg=np.maximum(gr.predy.flatten(),1e-10)
            pr2g=compute_poisson_pseudo_r2(ya,mg); dg=compute_poisson_deviance(ya,mg)
            pcg=np.sum((ya-mg)**2/mg)
            enp=gr.ENP if hasattr(gr,'ENP') else len(indeps)+1
            c1,c2,c3,c4=st.columns(4)
            with c1: st.metric("Bandwidth",f"{bw:.2f}" if isinstance(bw,float) else str(bw))
            with c2: st.metric("Pseudo R2",f"{pr2g:.6f}")
            with c3: st.metric("AICc",f"{gr.aicc:.4f}")
            with c4: st.metric("AIC",f"{gr.aic:.4f}")
            c1,c2,c3,c4=st.columns(4)
            with c1: st.metric("Deviance",f"{dg:.4f}")
            with c2: st.metric("Pearson Chi2",f"{pcg:.4f}")
            enpv=enp if isinstance(enp,(int,float,np.floating)) else len(indeps)+1
            with c3: st.metric("ENP",f"{enpv:.2f}")
            with c4: st.metric("Disp",f"{pcg/max(len(ya)-enpv,1):.4f}")
            vn=['Intercept']+indeps; pa=gr.params; tv=gr.tvalues; se=gr.bse
            st.markdown("### Koefisien Lokal")
            cs=[]
            for i,v in enumerate(vn):
                cs.append({'Variable':v,'Mean':pa[:,i].mean(),'Std':pa[:,i].std(),'Min':pa[:,i].min(),'Q1':np.percentile(pa[:,i],25),'Median':np.median(pa[:,i]),'Q3':np.percentile(pa[:,i],75),'Max':pa[:,i].max()})
            st.dataframe(pd.DataFrame(cs).round(6),use_container_width=True)
            st.markdown("### IRR Lokal")
            ip=np.exp(pa); il=[]
            for i,v in enumerate(vn):
                il.append({'Variable':v,'Mean IRR':ip[:,i].mean(),'Min':ip[:,i].min(),'Med':np.median(ip[:,i]),'Max':ip[:,i].max(),'%>1':f"{100*np.sum(ip[:,i]>1)/len(ip):.1f}%"})
            st.dataframe(pd.DataFrame(il).round(6),use_container_width=True)
            st.markdown("### t-values")
            tl=[]
            for i,v in enumerate(vn):
                ns=np.sum(np.abs(tv[:,i])>1.96)
                tl.append({'Variable':v,'Mean|t|':np.abs(tv[:,i]).mean(),'Min':tv[:,i].min(),'Max':tv[:,i].max(),'Sig':f"{ns} ({100*ns/len(tv):.1f}%)"})
            st.dataframe(pd.DataFrame(tl),use_container_width=True)
            st.markdown("### Distribusi Koefisien")
            nv=len(vn); cpr=min(3,nv); rn=(nv+cpr-1)//cpr
            fc=make_subplots(rows=rn,cols=cpr,subplot_titles=vn)
            for i,v in enumerate(vn):
                r=i//cpr+1; c=i%cpr+1
                fc.add_trace(go.Histogram(x=pa[:,i],nbinsx=25,name=v,marker_color=px.colors.qualitative.Set2[i%8]),row=r,col=c)
                if st.session_state.poisson_results is not None:
                    fc.add_vline(x=st.session_state.poisson_results.params[i],line_dash="dash",line_color="red",row=r,col=c)
            fc.update_layout(height=300*rn,showlegend=False,title_text="Koefisien Lokal (merah=global)")
            st.plotly_chart(fc,use_container_width=True)
            st.markdown("### Distribusi IRR")
            fi=make_subplots(rows=rn,cols=cpr,subplot_titles=["IRR: "+v for v in vn])
            for i,v in enumerate(vn):
                r=i//cpr+1; c=i%cpr+1
                fi.add_trace(go.Histogram(x=ip[:,i],nbinsx=25,name=v,marker_color=px.colors.qualitative.Pastel[i%8]),row=r,col=c)
                fi.add_vline(x=1,line_dash="dash",line_color="red",row=r,col=c)
            fi.update_layout(height=300*rn,showlegend=False,title_text="IRR Lokal (merah=1)")
            st.plotly_chart(fi,use_container_width=True)
            st.markdown("### Diagnostics")
            c1,c2=st.columns(2)
            with c1:
                fg=px.scatter(x=mg,y=ya,title="Actual vs Predicted",labels={'x':'Predicted','y':'Actual'},color_discrete_sequence=["#27AE60"])
                fg.add_trace(go.Scatter(x=[min(mg.min(),ya.min()),max(mg.max(),ya.max())],y=[min(mg.min(),ya.min()),max(mg.max(),ya.max())],mode='lines',line=dict(color='red',dash='dash'),name='Perfect'))
                st.plotly_chart(fg,use_container_width=True)
            with c2:
                sr=calc_pearson_residuals(ya,mg)
                fs=px.scatter(x=mg,y=sr,title="Std Pearson Resid",labels={'x':'Predicted','y':'Resid'},color_discrete_sequence=["#E74C3C"])
                fs.add_hline(y=0,line_dash="dash"); fs.add_hline(y=2,line_dash="dot",line_color="orange"); fs.add_hline(y=-2,line_dash="dot",line_color="orange")
                st.plotly_chart(fs,use_container_width=True)
            st.markdown("### Export")
            edf=df.copy()
            for i,v in enumerate(vn):
                edf[f'Beta_{v}']=pa[:,i]; edf[f'IRR_{v}']=np.exp(pa[:,i]); edf[f'SE_{v}']=se[:,i]; edf[f'tval_{v}']=tv[:,i]; edf[f'Sig_{v}']=np.where(np.abs(tv[:,i])>1.96,'Yes','No')
            edf['Predicted']=mg; edf['Residual']=ya-mg; edf['Pearson_Resid']=calc_pearson_residuals(ya,mg)
            buf=io.StringIO(); edf.to_csv(buf,index=False)
            fname_gwpr = "gwpr_params_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".csv"
            st.download_button("Download GWPR Params (CSV)",buf.getvalue(),fname_gwpr,"text/csv",use_container_width=True)
    else: st.warning("Pilih variabel!")

# TAB 5
with tab5:
    st.markdown("## Perbandingan Model")
    hp=st.session_state.poisson_results is not None; hg=st.session_state.gwpr_results is not None
    if not(hp or hg): st.warning("Jalankan model terlebih dahulu.")
    else:
        dep=st.session_state.dep_var; indeps=st.session_state.indep_vars; ya=df[dep].values
        cd=[]
        if hp:
            pois=st.session_state.poisson_results; mp=pois.predict(sm.add_constant(df[indeps].values))
            dp_v=compute_poisson_deviance(ya,mp); pp_v=compute_poisson_pseudo_r2(ya,mp)
            cd.append({'Model':'Global','PseudoR2':f"{pp_v:.6f}",'AIC':f"{pois.aic:.4f}",'AICc':f"{st.session_state.get('poisson_aicc',pois.aic):.4f}",'BIC':f"{pois.bic:.4f}",'Deviance':f"{dp_v:.4f}",'LogLik':f"{pois.llf:.4f}"})
        if hg:
            gw=st.session_state.gwpr_results; mgw=np.maximum(gw.predy.flatten(),1e-10)
            dg_v=compute_poisson_deviance(ya,mgw); pg_v=compute_poisson_pseudo_r2(ya,mgw)
            cd.append({'Model':'GWPR','PseudoR2':f"{pg_v:.6f}",'AIC':f"{gw.aic:.4f}",'AICc':f"{gw.aicc:.4f}",'Deviance':f"{dg_v:.4f}"})
        st.dataframe(pd.DataFrame(cd),use_container_width=True)
        if hp and hg:
            ap=st.session_state.get('poisson_aicc',pois.aic); ag=gw.aicc; da=ap-ag
            c1,c2,c3=st.columns(3)
            with c1: st.metric("Delta AIC",f"{pois.aic-gw.aic:.4f}")
            with c2: st.metric("Delta AICc",f"{da:.4f}")
            with c3: st.metric("Dev Improve",f"{((dp_v-dg_v)/dp_v*100) if dp_v>0 else 0:.2f}%")
            if da>2: st.markdown(f'<div class="success-box"><b>GWPR lebih baik!</b> Delta AICc={da:.2f}>2</div>',unsafe_allow_html=True)
            elif da>0: st.markdown(f'<div class="warning-box"><b>GWPR sedikit lebih baik.</b> Delta={da:.2f}</div>',unsafe_allow_html=True)
            else: st.markdown(f'<div class="info-box"><b>Global cukup.</b> Delta={da:.2f}</div>',unsafe_allow_html=True)
        fp=go.Figure()
        fp.add_trace(go.Scatter(x=[ya.min(),ya.max()],y=[ya.min(),ya.max()],mode='lines',line=dict(color='black',dash='dash'),name='Perfect'))
        if hp: fp.add_trace(go.Scatter(x=mp,y=ya,mode='markers',name='Global',marker=dict(color='blue',size=5,opacity=0.5)))
        if hg: fp.add_trace(go.Scatter(x=mgw,y=ya,mode='markers',name='GWPR',marker=dict(color='green',size=5,opacity=0.5)))
        fp.update_layout(title="Predicted vs Actual",xaxis_title="Predicted",yaxis_title="Actual",height=500)
        st.plotly_chart(fp,use_container_width=True)
        rd=[]
        if hp:
            rp=calc_pearson_residuals(ya,mp)
            for r in rp: rd.append({'Model':'Global','PR':r})
        if hg:
            rg=calc_pearson_residuals(ya,mgw)
            for r in rg: rd.append({'Model':'GWPR','PR':r})
        if rd: st.plotly_chart(px.box(pd.DataFrame(rd),x='Model',y='PR',color='Model',title="Pearson Residual"),use_container_width=True)
        md=[]
        if hp: md.append({'Model':'Global','RMSE':f"{np.sqrt(np.mean((ya-mp)**2)):.4f}",'MAE':f"{np.mean(np.abs(ya-mp)):.4f}"})
        if hg: md.append({'Model':'GWPR','RMSE':f"{np.sqrt(np.mean((ya-mgw)**2)):.4f}",'MAE':f"{np.mean(np.abs(ya-mgw)):.4f}"})
        if md: st.dataframe(pd.DataFrame(md),use_container_width=True)
        if hp and hg and st.session_state.lon_col:
            c1,c2=st.columns(2)
            with c1: st.plotly_chart(px.scatter_mapbox(df,lon=st.session_state.lon_col,lat=st.session_state.lat_col,color=ya-mp,size=np.abs(ya-mp),size_max=15,zoom=6,mapbox_style="open-street-map",title="Resid Global",color_continuous_scale="RdBu_r",color_continuous_midpoint=0).update_layout(height=400),use_container_width=True)
            with c2: st.plotly_chart(px.scatter_mapbox(df,lon=st.session_state.lon_col,lat=st.session_state.lat_col,color=ya-mgw,size=np.abs(ya-mgw),size_max=15,zoom=6,mapbox_style="open-street-map",title="Resid GWPR",color_continuous_scale="RdBu_r",color_continuous_midpoint=0).update_layout(height=400),use_container_width=True)

# TAB 6
with tab6:
    st.markdown("## Visualisasi Peta")
    if not FOLIUM_AVAILABLE:
        st.error("Install: pip install folium streamlit-folium branca")
    elif st.session_state.dep_var and st.session_state.indep_vars:
        dep=st.session_state.dep_var; indeps=st.session_state.indep_vars
        lc=st.session_state.lon_col; ac=st.session_state.lat_col
        mo=[]
        if st.session_state.poisson_results is not None: mo.append("Global")
        if st.session_state.gwpr_results is not None: mo.append("GWPR")
        if not mo: st.warning("Belum ada model!")
        else:
            mm=st.selectbox("Model:",mo,key="map_model"); vn=['Intercept']+indeps; gdf=df.copy()
            if mm=="Global":
                mu=st.session_state.poisson_results.predict(sm.add_constant(df[indeps].values))
                gdf['Predicted']=mu; gdf['Residual']=df[dep].values-mu; gdf['PR']=calc_pearson_residuals(df[dep].values,mu)
            else:
                gw=st.session_state.gwpr_results
                for i,v in enumerate(vn):
                    gdf[f'B_{v}']=gw.params[:,i]; gdf[f'IRR_{v}']=np.exp(gw.params[:,i]); gdf[f't_{v}']=gw.tvalues[:,i]
                    gdf[f'Sig_{v}']=np.where(np.abs(gw.tvalues[:,i])>1.96,'Sig','No')
                mu=gw.predy.flatten()
                gdf['Predicted']=mu; gdf['Residual']=df[dep].values-mu; gdf['PR']=calc_pearson_residuals(df[dep].values,mu)
            lo=["Predicted vs Observed","Residual (Pearson)"]
            if mm=="GWPR":
                for v in vn: lo+=[f"Beta: {v}",f"IRR: {v}",f"t-stat: {v}"]
            lt=st.selectbox("Layer:",lo,key="map_layer")
            m=folium.Map(location=[df[ac].mean(),df[lc].mean()],zoom_start=8,tiles='CartoDB positron')
            if lt=="Predicted vs Observed":
                mx=max(np.abs(gdf['Residual']).max(),1e-10)
                for _,row in gdf.iterrows():
                    rs=row['Residual']; cl='green' if rs>=0 else 'red'; rd=4+8*abs(rs)/mx
                    folium.CircleMarker([row[ac],row[lc]],radius=rd,color=cl,fill=True,fill_color=cl,fill_opacity=0.7,popup=folium.Popup(f"<b>{dep}</b>:{row[dep]}<br>Pred:{row['Predicted']:.2f}<br>Res:{rs:.2f}",max_width=250)).add_to(m)
            elif lt=="Residual (Pearson)":
                ser=gdf['PR']; cmp=create_colormap_safe(ser,'RdBu_r')
                for _,row in gdf.iterrows():
                    val=row['PR']
                    try: cl=cmp(float(val))
                    except: cl='#808080'
                    folium.CircleMarker([row[ac],row[lc]],radius=6,color=cl,fill=True,fill_color=cl,fill_opacity=0.8,popup=folium.Popup(f"PR:{val:.3f}",max_width=200)).add_to(m)
                cmp.caption="Pearson Residual"; m.add_child(cmp)
            elif lt.startswith("Beta:"):
                var=lt.replace("Beta: ",""); fld=f'B_{var}'
                if fld in gdf.columns:
                    ser=gdf[fld]; cmp=create_colormap_safe(ser,'RdYlBu')
                    for _,row in gdf.iterrows():
                        val=row[fld]
                        try: cl=cmp(float(val))
                        except: cl='#808080'
                        folium.CircleMarker([row[ac],row[lc]],radius=6,color=cl,fill=True,fill_color=cl,fill_opacity=0.8,popup=folium.Popup(f"<b>{var}</b><br>B={val:.4f}<br>IRR={np.exp(val):.4f}<br>{row.get(f'Sig_{var}','-')}",max_width=250)).add_to(m)
                    cmp.caption=f"Beta({var})"; m.add_child(cmp)
            elif lt.startswith("IRR:"):
                var=lt.replace("IRR: ",""); fld=f'IRR_{var}'
                if fld in gdf.columns:
                    ser=gdf[fld]; cmp=create_colormap_safe(ser,'RdYlGn')
                    for _,row in gdf.iterrows():
                        val=row[fld]
                        try: cl=cmp(float(val))
                        except: cl='#808080'
                        folium.CircleMarker([row[ac],row[lc]],radius=6,color=cl,fill=True,fill_color=cl,fill_opacity=0.8,popup=folium.Popup(f"IRR {var}={val:.4f}",max_width=200)).add_to(m)
                    cmp.caption=f"IRR({var})"; m.add_child(cmp)
            elif lt.startswith("t-stat:"):
                var=lt.replace("t-stat: ",""); fld=f't_{var}'
                if fld in gdf.columns:
                    ser=gdf[fld]; cmp=create_colormap_safe(ser,'PiYG')
                    for _,row in gdf.iterrows():
                        val=row[fld]
                        try: cl=cmp(float(val))
                        except: cl='#808080'
                        sig="Yes" if abs(val)>1.96 else "No"
                        folium.CircleMarker([row[ac],row[lc]],radius=6,color=cl,fill=True,fill_color=cl,fill_opacity=0.8,popup=folium.Popup(f"t({var})={val:.4f}<br>Sig:{sig}",max_width=200)).add_to(m)
                    cmp.caption=f"t({var})"; m.add_child(cmp)
            folium.LayerControl().add_to(m)
            st_folium(m,width=1200,height=600)
    else: st.warning("Pilih variabel!")

# TAB 7
with tab7:
    st.markdown("## Laporan & Export")
    if st.session_state.dep_var and st.session_state.indep_vars:
        dep=st.session_state.dep_var; indeps=st.session_state.indep_vars
        rl=["="*60,"LAPORAN ANALISIS GWPR",datetime.now().strftime('%Y-%m-%d %H:%M:%S'),"="*60,""]
        rl+=["1. DATA",f"   N={len(df)}, Y={dep}, X={', '.join(indeps)}",f"   Coord: {st.session_state.lon_col}, {st.session_state.lat_col}",""]
        yv=df[dep].values
        rl+=["2. DESKRIPSI Y",f"   Mean={yv.mean():.4f}, Var={yv.var():.4f}, Var/Mean={yv.var()/max(yv.mean(),1e-10):.4f}",f"   Range=({yv.min()},{yv.max()}), Zeros={np.sum(yv==0)}",""]
        if st.session_state.poisson_results is not None:
            p=st.session_state.poisson_results; mp_r=p.predict(sm.add_constant(df[indeps].values))
            rl+=["3. POISSON GLOBAL",f"   AIC={p.aic:.4f}, BIC={p.bic:.4f}, LogLik={p.llf:.4f}",f"   Dev={compute_poisson_deviance(yv,mp_r):.4f}, R2={compute_poisson_pseudo_r2(yv,mp_r):.6f}","   Koefisien:"]
            for i,v in enumerate(['Intercept']+indeps):
                sig="*" if p.pvalues[i]<0.05 else ""
                rl.append(f"   {v:20s} B={p.params[i]:.6f} IRR={np.exp(p.params[i]):.4f} p={p.pvalues[i]:.4f}{sig}")
            rl.append("")
        if st.session_state.gwpr_results is not None:
            g=st.session_state.gwpr_results; mg_r=np.maximum(g.predy.flatten(),1e-10)
            rl+=["4. GWPR",f"   BW={st.session_state.gwpr_bw}, AICc={g.aicc:.4f}",f"   Dev={compute_poisson_deviance(yv,mg_r):.4f}, R2={compute_poisson_pseudo_r2(yv,mg_r):.6f}","   Koefisien Lokal:"]
            vnl=['Intercept']+indeps
            for i,v in enumerate(vnl):
                ns=np.sum(np.abs(g.tvalues[:,i])>1.96); ps=100*ns/len(g.tvalues)
                rl.append(f"   {v:20s} Mean={g.params[:,i].mean():.4f} Std={g.params[:,i].std():.4f} %Sig={ps:.1f}%")
            rl.append("")
        if st.session_state.poisson_results is not None and st.session_state.gwpr_results is not None:
            ap=st.session_state.get('poisson_aicc',p.aic); ag=g.aicc
            rl+=["5. PERBANDINGAN",f"   AICc Global={ap:.4f}, AICc GWPR={ag:.4f}, Delta={ap-ag:.4f}",f"   Best: {'GWPR' if ag<ap else 'Global'}",""]
        rl+=["INTERPRETASI","- IRR=exp(B): >1=meningkatkan, <1=menurunkan count","- |t|>1.96 = signifikan (alpha=5%)"]
        rt="\n".join(rl)
        st.text_area("Preview",rt,height=400)
        fname_rpt = "laporan_GWPR_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".txt"
        st.download_button("Download Laporan (.txt)",rt,fname_rpt,"text/plain",use_container_width=True)
        st.markdown("### Export Konfigurasi")
        cfg={"analysis":"GWPR","timestamp":datetime.now().isoformat(),"data":{"n":len(df),"dep":dep,"indep":indeps,"lon":st.session_state.lon_col,"lat":st.session_state.lat_col}}
        if st.session_state.poisson_results is not None:
            p=st.session_state.poisson_results
            cfg["global"]={"aic":float(p.aic),"bic":float(p.bic),"params":{v:float(p.params[i]) for i,v in enumerate(['Intercept']+indeps)}}
        if st.session_state.gwpr_results is not None:
            g=st.session_state.gwpr_results; bwv=st.session_state.gwpr_bw
            cfg["gwpr"]={"bw":float(bwv) if isinstance(bwv,(int,float,np.floating)) else str(bwv),"aicc":float(g.aicc),"aic":float(g.aic)}
        cj=json.dumps(cfg,indent=2,default=str)
        fname_cfg = "gwpr_config_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".json"
        st.download_button("Download Config (.json)",cj,fname_cfg,"application/json",use_container_width=True)
        st.markdown("### Export Data Lengkap")
        if st.session_state.gwpr_results is not None:
            g=st.session_state.gwpr_results; edf=df.copy(); vnl=['Intercept']+indeps
            for i,v in enumerate(vnl):
                edf[f'Beta_{v}']=g.params[:,i]; edf[f'IRR_{v}']=np.exp(g.params[:,i])
                edf[f'SE_{v}']=g.bse[:,i]; edf[f'tval_{v}']=g.tvalues[:,i]
                edf[f'Sig_{v}']=np.where(np.abs(g.tvalues[:,i])>1.96,'Yes','No')
            mu_e=g.predy.flatten()
            edf['Predicted']=mu_e; edf['Residual']=df[dep].values-mu_e
            edf['Pearson_Resid']=calc_pearson_residuals(df[dep].values,mu_e)
            buf=io.StringIO(); edf.to_csv(buf,index=False)
            fname_full = "gwpr_full_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".csv"
            st.download_button("Download Full (CSV)",buf.getvalue(),fname_full,"text/csv",use_container_width=True)
        else: st.info("Jalankan GWPR terlebih dahulu.")
    else: st.warning("Pilih variabel!")

st.markdown("---")
st.markdown('<div style="text-align:center;color:#888;font-size:0.85rem">GWPR Analysis Suite | Powered by mgwr, statsmodels, streamlit</div>',unsafe_allow_html=True)
