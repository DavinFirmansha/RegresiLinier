import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram, cophenet
from itertools import combinations
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io, warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Analisis Multivariat Pro", layout="wide", page_icon="📊")
st.title("📊 Aplikasi Analisis Statistik Multivariat — Versi Lengkap & Mendalam")

# ============================================================
#                    HELPER FUNCTIONS
# ============================================================
def standardize(X):
    mu=X.mean(axis=0); sd=X.std(axis=0,ddof=1); sd[sd==0]=1
    return (X-mu)/sd, mu, sd

def mardia_test(X):
    n,p=X.shape; mu=X.mean(0); S=np.cov(X.T,ddof=1)
    try: Si=np.linalg.inv(S)
    except: Si=np.linalg.pinv(S)
    Y=(X-mu)@Si@(X-mu).T
    b1=(Y**3).mean(); chi2=n*b1/6; df_s=p*(p+1)*(p+2)/6; p_s=1-stats.chi2.cdf(chi2,df_s)
    b2=np.trace(Y**2)/n; mu_k=p*(p+2); var_k=8*p*(p+2)/n
    z_k=(b2-mu_k)/np.sqrt(var_k); p_k=2*(1-stats.norm.cdf(abs(z_k)))
    return b1,chi2,p_s,b2,z_k,p_k

def hz_test(X):
    n,p=X.shape; S=np.cov(X.T,ddof=1)
    try: Si=np.linalg.inv(S)
    except: Si=np.linalg.pinv(S)
    dij=cdist(X,X,'mahalanobis',VI=Si)
    beta=1.0/(np.sqrt(2))*((2*p+1)*n/4.0)**(1.0/(p+4))
    eb=np.exp(-beta**2/2*dij**2)
    HZ=n*(np.mean(eb)-2*(1+beta**2)**(-p/2)*np.mean(np.exp(-beta**2/(2*(1+beta**2))*np.sum(((X-X.mean(0))@Si)**2,axis=1)))+(1+2*beta**2)**(-p/2))
    mu_hz=1-(1+2*beta**2)**(-p/2)*(1+p*beta**2/(1+2*beta**2)+(p*(p+2)*beta**4)/(2*(1+2*beta**2)**2))
    si2=2*(1+4*beta**2)**(-p/2)+2*(1+2*beta**2)**(-p)*(1+2*p*beta**4/(1+2*beta**2)**2)+p*(p+2)*beta**8*3/(4*(1+2*beta**2)**(p/2+2))-mu_hz**2
    si2=max(si2,1e-15); z=(HZ-mu_hz)/np.sqrt(si2); pval=1-stats.norm.cdf(z)
    return abs(HZ),z,max(min(pval,1),0)

def royston_test(X):
    n,p=X.shape; H=0
    for j in range(p):
        w,pw=stats.shapiro(X[:,j][:5000]); z=stats.norm.ppf(max(min(1-pw,0.9999),0.0001)); H+=z**2
    return H,1-stats.chi2.cdf(H,p)

def kmo_test(X):
    n,p=X.shape; R=np.corrcoef(X.T)
    try: Ri=np.linalg.inv(R)
    except: Ri=np.linalg.pinv(R)
    D=np.diag(1.0/np.sqrt(np.diag(Ri))); Q=D@Ri@D
    np.fill_diagonal(Q,0); np.fill_diagonal(R,0)
    sr=np.sum(R**2); sq=np.sum(Q**2)
    kmo_total=sr/(sr+sq) if (sr+sq)>0 else 0
    kmo_per=np.zeros(p)
    for j in range(p):
        sr_j=np.sum(R[j,:]**2); sq_j=np.sum(Q[j,:]**2)
        kmo_per[j]=sr_j/(sr_j+sq_j) if (sr_j+sq_j)>0 else 0
    return kmo_total,kmo_per

def bartlett_sphericity(X):
    n,p=X.shape; R=np.corrcoef(X.T); det_R=max(np.linalg.det(R),1e-300)
    chi2=-(n-1-(2*p+5)/6)*np.log(det_R); df=p*(p-1)/2; p_val=1-stats.chi2.cdf(chi2,df)
    return chi2,df,p_val

def parallel_analysis(X,n_iter=500):
    n,p=X.shape; eig_rand=np.zeros((n_iter,p))
    for i in range(n_iter):
        Xr=np.random.normal(size=(n,p)); eig_rand[i]=np.sort(np.linalg.eigvalsh(np.corrcoef(Xr.T)))[::-1]
    return eig_rand.mean(0),np.percentile(eig_rand,95,axis=0)

def pca_analysis(X):
    Xs,mu,sd=standardize(X); R=np.corrcoef(Xs.T)
    eigvals,eigvecs=np.linalg.eigh(R); idx=np.argsort(eigvals)[::-1]; eigvals=eigvals[idx]; eigvecs=eigvecs[:,idx]
    eigvals=np.maximum(eigvals,0); loadings=eigvecs*np.sqrt(eigvals)
    scores=Xs@eigvecs; prop_var=eigvals/eigvals.sum(); cum_var=np.cumsum(prop_var)
    return eigvals,eigvecs,loadings,scores,prop_var,cum_var,R

def _varimax(L,max_iter=100,tol=1e-6):
    p,k=L.shape; R=np.eye(k)
    for _ in range(max_iter):
        old=L@R; u2=old**2; cm=u2.mean(0); A=old.T@(old**3-old*cm)
        U,S,Vt=np.linalg.svd(A); Rnew=U@Vt
        if np.max(np.abs(Rnew-R))<tol: R=Rnew; break
        R=Rnew
    return L@R,R

def _promax(L,power=4):
    L_vm,_=_varimax(L); h2=np.sum(L_vm**2,axis=1,keepdims=True)
    L_norm=L_vm/np.sqrt(h2+1e-15); target=np.sign(L_norm)*np.abs(L_norm)**power
    coef=np.linalg.lstsq(L_vm,target,rcond=None)[0]; return L_vm@coef,coef

def factor_analysis(X,n_factors,rotation='varimax'):
    Xs,mu,sd=standardize(X); n,p=Xs.shape; R=np.corrcoef(Xs.T)
    eigvals,eigvecs=np.linalg.eigh(R); idx=np.argsort(eigvals)[::-1]; eigvals=eigvals[idx]; eigvecs=eigvecs[:,idx]
    L=eigvecs[:,:n_factors]*np.sqrt(np.maximum(eigvals[:n_factors],0)); R_rot=None
    if rotation=='varimax': L,R_rot=_varimax(L)
    elif rotation=='promax': L,R_rot=_promax(L)
    comm=np.sum(L**2,axis=1); uniq=1-comm
    reproduced=L@L.T; np.fill_diagonal(reproduced,1.0); residual=R-reproduced
    return L,comm,uniq,None,R,reproduced,residual,R_rot

def box_m_test(groups):
    k=len(groups); p=groups[0].shape[1]; ns=[len(g) for g in groups]; N=sum(ns)
    Sp=sum((n-1)*np.cov(g.T,ddof=1) for n,g in zip(ns,groups))/(N-k); M=0
    for i,g in enumerate(groups):
        Si=np.cov(g.T,ddof=1); M+=(ns[i]-1)*(np.log(max(np.linalg.det(Sp),1e-300))-np.log(max(np.linalg.det(Si),1e-300)))
    c1=(sum(1.0/(n-1) for n in ns)-1.0/(N-k))*(2*p**2+3*p-1)/(6*(p+1)*(k-1))
    df=p*(p+1)*(k-1)/2; chi2=M*(1-c1); pval=1-stats.chi2.cdf(chi2,df)
    return M,chi2,df,pval

def levene_multivariate(X,groups):
    unique_g=np.unique(groups); results=[]
    for j in range(X.shape[1]):
        gdata=[X[groups==g,j] for g in unique_g]; stat,pval=stats.levene(*gdata); results.append((stat,pval))
    return results

def hotelling_t2(X1,X2):
    n1,p=X1.shape; n2=X2.shape[0]; d=X1.mean(0)-X2.mean(0)
    Sp=((n1-1)*np.cov(X1.T,ddof=1)+(n2-1)*np.cov(X2.T,ddof=1))/(n1+n2-2)
    try: Spi=np.linalg.inv(Sp)
    except: Spi=np.linalg.pinv(Sp)
    T2=(n1*n2)/(n1+n2)*d@Spi@d; F=T2*(n1+n2-p-1)/(p*(n1+n2-2)); p_val=1-stats.f.cdf(F,p,n1+n2-p-1)
    return T2,F,p,n1+n2-p-1,p_val,Sp,d

def manova_test(X,groups):
    unique_g=np.unique(groups); k=len(unique_g); n,p=X.shape; gm=X.mean(0)
    B=np.zeros((p,p)); W=np.zeros((p,p))
    for g in unique_g:
        Xg=X[groups==g]; ng=len(Xg); mg=Xg.mean(0); B+=ng*np.outer(mg-gm,mg-gm); W+=(Xg-mg).T@(Xg-mg)
    T=B+W
    try: Wi=np.linalg.inv(W)
    except: Wi=np.linalg.pinv(W)
    eigvals=np.sort(np.real(np.linalg.eigvals(Wi@B)))[::-1]; eigvals=np.maximum(eigvals,0)
    s=min(p,k-1); wilks=np.prod(1/(1+eigvals[:s])); pillai=np.sum(eigvals[:s]/(1+eigvals[:s]))
    hotelling_l=np.sum(eigvals[:s]); roy=eigvals[0]; m=n-k; results=[]
    r=m-(p-k+2)/2; t=np.sqrt(max((p**2*(k-1)**2-4)/(p**2+(k-1)**2-5),1))
    lam_1t=max(wilks**(1/t),1e-15); df1_w=p*(k-1); df2_w=max(r*t-df1_w/2+1,1)
    F_w=((1-lam_1t)/lam_1t)*(df2_w/df1_w) if lam_1t>0 else 0; p_w=1-stats.f.cdf(max(F_w,0),df1_w,df2_w)
    results.append({'Statistic':"Wilks' Λ",'Value':round(wilks,4),'F':round(F_w,4),'df1':int(df1_w),'df2':int(df2_w),'p-value':round(p_w,6),'Sig?':'✅' if p_w<0.05 else '❌'})
    s_p=min(p,k-1); df1_p=s_p*p; df2_p=s_p*(m-p+s_p)
    F_p=(pillai/s_p)/((1-pillai/s_p))*(df2_p/df1_p) if (1-pillai/s_p)>0 and s_p>0 else 0; p_p=1-stats.f.cdf(max(F_p,0),max(df1_p,1),max(df2_p,1))
    results.append({'Statistic':"Pillai's Trace",'Value':round(pillai,4),'F':round(F_p,4),'df1':int(df1_p),'df2':int(df2_p),'p-value':round(p_p,6),'Sig?':'✅' if p_p<0.05 else '❌'})
    df1_h=s_p*p; df2_h=s_p*(m-p-1)
    F_h=(hotelling_l/s_p)*(max(df2_h,1)/max(df1_h,1)); p_h=1-stats.f.cdf(max(F_h,0),max(df1_h,1),max(df2_h,1))
    results.append({'Statistic':"Hotelling-Lawley",'Value':round(hotelling_l,4),'F':round(F_h,4),'df1':int(df1_h),'df2':int(df2_h),'p-value':round(p_h,6),'Sig?':'✅' if p_h<0.05 else '❌'})
    df1_r=max(p,k-1); df2_r=m-max(p,k-1)+k-1
    F_r=roy*max(df2_r,1)/max(df1_r,1); p_r=1-stats.f.cdf(max(F_r,0),max(df1_r,1),max(df2_r,1))
    results.append({'Statistic':"Roy's Largest Root",'Value':round(roy,4),'F':round(F_r,4),'df1':int(df1_r),'df2':int(df2_r),'p-value':round(p_r,6),'Sig?':'✅' if p_r<0.05 else '❌'})
    return pd.DataFrame(results),eigvals,B,W,T

def lda_fit(X,y):
    classes=np.unique(y); k=len(classes); n,p=X.shape; grand_mean=X.mean(0)
    Sw=np.zeros((p,p)); Sb=np.zeros((p,p)); priors={}; group_means={}
    for c in classes:
        Xc=X[y==c]; nc=len(Xc); mc=Xc.mean(0); priors[c]=nc/n; group_means[c]=mc
        Sw+=(Xc-mc).T@(Xc-mc); Sb+=nc*np.outer(mc-grand_mean,mc-grand_mean)
    try: Swi=np.linalg.inv(Sw)
    except: Swi=np.linalg.pinv(Sw)
    eigvals,eigvecs=np.linalg.eig(Swi@Sb); idx=np.argsort(np.real(eigvals))[::-1]
    eigvals=np.real(eigvals[idx]); eigvecs=np.real(eigvecs[:,idx])
    n_disc=min(k-1,p); eigvals=np.maximum(eigvals[:n_disc],0); Wm=eigvecs[:,:n_disc]
    scores=X@Wm; y_pred=np.empty_like(y)
    for i in range(n):
        best_c=None; best_d=np.inf
        for c in classes:
            sc=group_means[c]@Wm; d=np.sum((scores[i]-sc)**2)-2*np.log(priors[c])
            if d<best_d: best_d=d; best_c=c
        y_pred[i]=best_c
    return Wm,eigvals,scores,y_pred,np.mean(y==y_pred),priors,Sw,Sb,group_means

def kmeans_fit(X,k,n_init=10,max_iter=300):
    best_inertia=np.inf; best_labels=None; best_centers=None; n=len(X)
    for _ in range(n_init):
        idx=np.random.choice(n,k,replace=False); centers=X[idx].copy()
        for _ in range(max_iter):
            dists=cdist(X,centers,'euclidean'); labels=np.argmin(dists,axis=1)
            new_c=np.array([X[labels==j].mean(0) if np.sum(labels==j)>0 else centers[j] for j in range(k)])
            if np.allclose(new_c,centers): break
            centers=new_c
        inertia=sum(np.sum((X[labels==j]-centers[j])**2) for j in range(k))
        if inertia<best_inertia: best_inertia=inertia; best_labels=labels; best_centers=centers
    return best_labels,best_centers,best_inertia

def silhouette_score(X,labels):
    n=len(X); sil=np.zeros(n); unique_l=np.unique(labels)
    if len(unique_l)<2: return 0.0,sil
    D=squareform(pdist(X,'euclidean'))
    for i in range(n):
        ci=labels[i]; mask_ci=labels==ci
        a_i=np.mean(D[i,mask_ci&(np.arange(n)!=i)]) if np.sum(mask_ci)>1 else 0; b_i=np.inf
        for c in unique_l:
            if c==ci: continue
            b_c=np.mean(D[i,labels==c])
            if b_c<b_i: b_i=b_c
        sil[i]=(b_i-a_i)/max(a_i,b_i,1e-15)
    return np.mean(sil),sil

def dbscan_fit(X,eps,min_samples):
    n=len(X); labels=np.full(n,-1); D=squareform(pdist(X,'euclidean')); cluster_id=0
    for i in range(n):
        if labels[i]!=-1: continue
        neighbors=np.where(D[i]<=eps)[0]
        if len(neighbors)<min_samples: continue
        labels[i]=cluster_id; seed=list(neighbors)
        while seed:
            q=seed.pop(0)
            if labels[q]==-1 or labels[q]==-2: labels[q]=cluster_id
            else: continue
            q_nb=np.where(D[q]<=eps)[0]
            if len(q_nb)>=min_samples: seed.extend(q_nb)
        cluster_id+=1
    return labels

def cca_analysis(X,Y):
    n=X.shape[0]; Xs,_,_=standardize(X); Ys,_,_=standardize(Y)
    Rxx=np.corrcoef(Xs.T); Ryy=np.corrcoef(Ys.T); Rxy=(Xs.T@Ys)/(n-1)
    try: Rxxi=np.linalg.inv(Rxx); Ryyi=np.linalg.inv(Ryy)
    except: Rxxi=np.linalg.pinv(Rxx); Ryyi=np.linalg.pinv(Ryy)
    M=Rxxi@Rxy@Ryyi@Rxy.T; eigvals,eigvecs=np.linalg.eig(M)
    idx=np.argsort(np.real(eigvals))[::-1]; eigvals=np.maximum(np.real(eigvals[idx]),0)
    A=np.real(eigvecs[:,idx]); can_corr=np.sqrt(eigvals)
    M2=Ryyi@Rxy.T@Rxxi@Rxy; eigvals2,eigvecs2=np.linalg.eig(M2)
    idx2=np.argsort(np.real(eigvals2))[::-1]; B=np.real(eigvecs2[:,idx2])
    return can_corr,A,B,Xs@A,Ys@B

def ca_analysis(ct):
    ct=ct.astype(float); N=ct.sum(); P=ct/N; r=P.sum(axis=1); c=P.sum(axis=0)
    Dr=np.diag(1.0/np.sqrt(r+1e-15)); Dc=np.diag(1.0/np.sqrt(c+1e-15))
    S=Dr@(P-np.outer(r,c))@Dc; U,sv,Vt=np.linalg.svd(S,full_matrices=False)
    inertias=sv**2; total=inertias.sum(); prop=inertias/total if total>0 else inertias
    row_coords=Dr@U*sv; col_coords=Dc@Vt.T*sv
    return row_coords[:,1:],col_coords[:,1:],inertias[1:],total,prop[1:]

# ============================================================
#              DEMO DATA GENERATOR
# ============================================================
def generate_demo_data(name):
    np.random.seed(42)
    if name=='Iris (Fisher)':
        n_per=50
        setosa=np.column_stack([np.random.normal(5.0,0.35,n_per),np.random.normal(3.4,0.38,n_per),np.random.normal(1.5,0.17,n_per),np.random.normal(0.2,0.10,n_per)])
        versicolor=np.column_stack([np.random.normal(5.9,0.52,n_per),np.random.normal(2.8,0.31,n_per),np.random.normal(4.3,0.47,n_per),np.random.normal(1.3,0.20,n_per)])
        virginica=np.column_stack([np.random.normal(6.6,0.64,n_per),np.random.normal(3.0,0.32,n_per),np.random.normal(5.6,0.55,n_per),np.random.normal(2.0,0.27,n_per)])
        data=np.vstack([setosa,versicolor,virginica])
        species=['setosa']*n_per+['versicolor']*n_per+['virginica']*n_per
        df=pd.DataFrame(data,columns=['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width']); df['Species']=species
        return df
    elif name=='Wine Quality':
        n=178; cultivar=np.random.choice(['Cultivar_1','Cultivar_2','Cultivar_3'],n,p=[0.33,0.40,0.27])
        base={'Cultivar_1':[13.7,2.0,2.5,17,106,2.8,3.0,0.29,1.9,5.5,1.06,3.2,1100],'Cultivar_2':[12.3,1.9,2.2,20,95,2.2,2.0,0.36,1.6,3.1,1.06,2.8,520],'Cultivar_3':[13.2,3.3,2.4,22,100,1.7,1.0,0.45,1.5,7.4,0.68,2.0,630]}
        sds=[0.8,0.6,0.3,3,15,0.5,0.6,0.1,0.4,1.5,0.2,0.4,300]
        cols=['Alcohol','Malic_Acid','Ash','Alcalinity_Ash','Magnesium','Phenols','Flavanoids','Nonflavanoid_Phenols','Proanthocyanins','Color_Intensity','Hue','OD280_OD315','Proline']
        rows=[]
        for i in range(n):
            c=cultivar[i]; b=base[c]; rows.append([max(b[j]+np.random.normal(0,sds[j]),0.1) for j in range(13)])
        df=pd.DataFrame(rows,columns=cols); df['Cultivar']=cultivar; return df
    elif name=='Students Performance':
        n=200; gender=np.random.choice(['Male','Female'],n); program=np.random.choice(['Science','Arts','Business'],n)
        math=np.random.normal(70,12,n).clip(20,100); reading=(math*0.6+np.random.normal(25,8,n)).clip(20,100)
        writing=(reading*0.7+math*0.2+np.random.normal(5,6,n)).clip(20,100); science=(math*0.5+np.random.normal(30,10,n)).clip(20,100)
        study_hours=(math/10+np.random.normal(0,2,n)).clip(1,20); attendance=(70+math*0.2+np.random.normal(0,8,n)).clip(40,100)
        return pd.DataFrame({'Math':np.round(math,1),'Reading':np.round(reading,1),'Writing':np.round(writing,1),'Science':np.round(science,1),'Study_Hours':np.round(study_hours,1),'Attendance':np.round(attendance,1),'Gender':gender,'Program':program})
    elif name=='Body Measurements':
        n=150; gender=np.random.choice(['Male','Female'],n); rows=[]
        for i in range(n):
            if gender[i]=='Male': h=np.random.normal(175,7); w=h*0.45+np.random.normal(-5,8); chest=np.random.normal(100,6); waist=np.random.normal(85,8)
            else: h=np.random.normal(162,6); w=h*0.38+np.random.normal(-2,7); chest=np.random.normal(90,5); waist=np.random.normal(72,7)
            bmi=w/(h/100)**2; hip=waist+np.random.normal(12,4); rows.append([h,w,bmi,chest,waist,hip])
        df=pd.DataFrame(rows,columns=['Height_cm','Weight_kg','BMI','Chest_cm','Waist_cm','Hip_cm']).round(1)
        df['Gender']=gender; df['Age_Group']=np.random.choice(['20-30','31-40','41-50'],n,p=[0.4,0.35,0.25]); return df
    elif name=='Employee Survey':
        n=180; dept=np.random.choice(['Engineering','Marketing','Finance','HR'],n,p=[0.35,0.25,0.25,0.15])
        satisfaction=np.random.normal(3.5,0.8,n).clip(1,5); engagement=(satisfaction*0.5+np.random.normal(1.5,0.6,n)).clip(1,5)
        work_life=np.random.normal(3.2,0.9,n).clip(1,5); compensation=np.random.normal(3.0,1.0,n).clip(1,5)
        growth=(satisfaction*0.3+np.random.normal(2.0,0.7,n)).clip(1,5); teamwork=(engagement*0.4+np.random.normal(2.0,0.6,n)).clip(1,5)
        leadership=np.random.normal(3.3,0.85,n).clip(1,5); tenure=np.random.exponential(4,n).clip(0.5,25)
        perf=(satisfaction*0.3+engagement*0.3+np.random.normal(1.5,0.5,n)).clip(1,5)
        return pd.DataFrame({'Satisfaction':np.round(satisfaction,2),'Engagement':np.round(engagement,2),'Work_Life_Balance':np.round(work_life,2),'Compensation':np.round(compensation,2),'Growth_Opportunity':np.round(growth,2),'Teamwork':np.round(teamwork,2),'Leadership':np.round(leadership,2),'Tenure_Years':np.round(tenure,1),'Performance_Rating':np.round(perf,2),'Department':dept})

# ============================================================
#                 SIDEBAR & DATA
# ============================================================
st.sidebar.header("📁 Data")
demo_choice=st.sidebar.selectbox("📦 Data Demo:",[
    '(Upload sendiri)','Iris (Fisher)','Wine Quality','Students Performance','Body Measurements','Employee Survey'])
uploaded_file=st.sidebar.file_uploader("Atau upload CSV/Excel",type=['csv','xlsx','xls'])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'): df=pd.read_csv(uploaded_file)
        else: df=pd.read_excel(uploaded_file)
    except Exception as e: st.error(f"Error: {e}"); st.stop()
elif demo_choice!='(Upload sendiri)':
    df=generate_demo_data(demo_choice)
    st.sidebar.success(f"✅ '{demo_choice}' dimuat ({df.shape[0]}×{df.shape[1]})")
else:
    st.info("📌 Pilih **Data Demo** dari sidebar atau **upload file** untuk memulai.")
    st.markdown("""
### 📦 Data Demo Tersedia
| Dataset | n | Numerik | Kategorikal | Cocok Untuk |
|---|---|---|---|---|
| **Iris** | 150 | 4 | 1 (Species) | PCA, FA, LDA, K-Means, MANOVA |
| **Wine Quality** | 178 | 13 | 1 (Cultivar) | PCA, FA, LDA, MANOVA, CCA |
| **Students** | 200 | 6 | 2 (Gender, Program) | MANOVA, Hotelling, CCA, MDS |
| **Body Measurements** | 150 | 6 | 2 (Gender, Age) | PCA, Hotelling, DBSCAN, MDS |
| **Employee Survey** | 180 | 9 | 1 (Department) | FA, PCA, K-Means, Hierarchical |
    """)
    st.stop()

numeric_cols=df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols=df.select_dtypes(include=['object','category']).columns.tolist()
if len(numeric_cols)<2: st.error("Data harus memiliki minimal 2 kolom numerik."); st.stop()

method=st.sidebar.selectbox("🔬 Metode Analisis:",['pca','fa','manova','hotelling','lda','kmeans','hclust','dbscan','mds','cca','ca','mvnorm'],
    format_func=lambda x:{'pca':'PCA','fa':'Factor Analysis','manova':'MANOVA','hotelling':"Hotelling's T²",
    'lda':'LDA','kmeans':'K-Means','hclust':'Hierarchical Clustering','dbscan':'DBSCAN',
    'mds':'MDS','cca':'CCA','ca':'Correspondence Analysis','mvnorm':'Uji Normalitas MV'}[x])

# ============================================================
# 1. EKSPLORASI DATA
# ============================================================
st.header("1. Eksplorasi Data")
tab_info,tab_desc,tab_dist,tab_corr,tab_outlier=st.tabs(["📋 Info","📈 Deskriptif","📊 Distribusi","🔗 Korelasi","🔍 Outlier"])
with tab_info:
    c1,c2,c3,c4=st.columns(4); c1.metric("Observasi",df.shape[0]); c2.metric("Variabel",df.shape[1]); c3.metric("Numerik",len(numeric_cols)); c4.metric("Kategorikal",len(cat_cols))
    st.dataframe(df.head(20),use_container_width=True)
    miss=df.isnull().sum()
    if miss.sum()>0: st.warning(f"Missing: {miss.sum()}")
    else: st.success("✅ Tidak ada missing values.")
with tab_desc:
    desc=df[numeric_cols].describe().T; desc['CV(%)']=((desc['std']/desc['mean'])*100).round(2)
    desc['Skewness']=df[numeric_cols].skew().round(4); desc['Kurtosis']=df[numeric_cols].kurtosis().round(4)
    st.dataframe(desc.round(4),use_container_width=True)
with tab_dist:
    sel_var=st.selectbox("Variabel:",numeric_cols,key='dist_v'); data_v=df[sel_var].dropna()
    cd1,cd2=st.columns(2)
    with cd1:
        fig=px.histogram(data_v,nbins=30,title=f"Histogram: {sel_var}",marginal='box'); fig.update_layout(height=400); st.plotly_chart(fig,use_container_width=True)
    with cd2:
        sd=np.sort(data_v); th=stats.norm.ppf(np.linspace(0.01,0.99,len(sd)))
        fig=go.Figure(); fig.add_trace(go.Scatter(x=th,y=sd,mode='markers',marker=dict(size=4,opacity=0.5)))
        fig.add_trace(go.Scatter(x=[th.min(),th.max()],y=[sd.mean()+sd.std()*th.min(),sd.mean()+sd.std()*th.max()],mode='lines',line=dict(color='red',dash='dash')))
        fig.update_layout(title=f"Q-Q Plot: {sel_var}",height=400); st.plotly_chart(fig,use_container_width=True)
    sw,swp=stats.shapiro(data_v.values[:5000]); ks,ksp=stats.kstest(data_v,'norm',args=(data_v.mean(),data_v.std()))
    st.markdown(f"Shapiro-Wilk: W={sw:.4f}, p={swp:.6f} {'✅' if swp>0.05 else '❌'} | KS: D={ks:.4f}, p={ksp:.6f} {'✅' if ksp>0.05 else '❌'}")
with tab_corr:
    cm=st.selectbox("Metode:",['pearson','spearman','kendall'],key='cm')
    Rc=df[numeric_cols].corr(method=cm)
    fig=go.Figure(data=go.Heatmap(z=Rc.values,x=numeric_cols,y=numeric_cols,colorscale='RdBu_r',zmid=0,text=Rc.values.round(3),texttemplate='%{text}'))
    fig.update_layout(title=f"Korelasi ({cm.title()})",height=500); st.plotly_chart(fig,use_container_width=True)
with tab_outlier:
    sv_o=st.selectbox("Variabel:",numeric_cols,key='ov'); do=df[sv_o].dropna()
    Q1=do.quantile(0.25);Q3=do.quantile(0.75);IQR=Q3-Q1
    out_n=len(do[(do<Q1-1.5*IQR)|(do>Q3+1.5*IQR)]); st.markdown(f"IQR outliers: **{out_n}**")
    fig=px.box(do,title=f"Boxplot: {sv_o}"); fig.update_layout(height=350); st.plotly_chart(fig,use_container_width=True)

# ============================================================
# PCA
# ============================================================
if method=='pca':
    st.header("2. Principal Component Analysis (PCA)")
    vars_pca=st.multiselect("Variabel:",numeric_cols,default=numeric_cols)
    if len(vars_pca)<2: st.warning("Pilih minimal 2."); st.stop()
    X=df[vars_pca].dropna().values; n,p=X.shape
    st.subheader("2a. Uji Kelayakan")
    kmo_val,kmo_per=kmo_test(X); bart_chi2,bart_df,bart_p=bartlett_sphericity(X)
    kl=('Sangat Baik' if kmo_val>=0.9 else 'Baik' if kmo_val>=0.8 else 'Cukup' if kmo_val>=0.7 else 'Mediocre' if kmo_val>=0.6 else 'Buruk' if kmo_val>=0.5 else 'Tidak Layak')
    c1,c2,c3=st.columns(3); c1.metric("KMO",f"{kmo_val:.4f} ({kl})"); c2.metric("Bartlett χ²",f"{bart_chi2:.2f}"); c3.metric("Bartlett p",f"{bart_p:.2e}")
    with st.expander("KMO per Variabel"):
        st.dataframe(pd.DataFrame({'Variabel':vars_pca,'KMO':kmo_per.round(4),'Status':['✅' if k>=0.5 else '❌' for k in kmo_per]}),use_container_width=True,hide_index=True)
    eigenvalues,eigenvectors,loadings,scores,prop_var,cum_var,corr_matrix=pca_analysis(X)
    st.subheader("2b. Eigenvalues & Retensi")
    pa_mean,pa_95=parallel_analysis(X,n_iter=500)
    n_kaiser=int(np.sum(eigenvalues>=1)); n_pa=int(np.sum([eigenvalues[i]>pa_95[i] for i in range(p)]))
    eig_df=pd.DataFrame({'PC':[f'PC{i+1}' for i in range(p)],'Eigenvalue':eigenvalues.round(4),'%Var':(prop_var*100).round(2),'%Cum':(cum_var*100).round(2),'Kaiser':['✅' if e>1 else '❌' for e in eigenvalues],'PA':['✅' if eigenvalues[i]>pa_95[i] else '❌' for i in range(p)]})
    st.dataframe(eig_df,use_container_width=True,hide_index=True)
    fig=make_subplots(specs=[[{"secondary_y":True}]])
    fig.add_trace(go.Bar(x=eig_df['PC'],y=eig_df['Eigenvalue'],name='Eigenvalue',marker_color='steelblue',opacity=0.7),secondary_y=False)
    fig.add_trace(go.Scatter(x=eig_df['PC'],y=eig_df['%Cum'],name='%Cum',mode='lines+markers',line=dict(color='red')),secondary_y=True)
    fig.add_trace(go.Scatter(x=eig_df['PC'],y=pa_95,name='PA 95%',mode='lines+markers',line=dict(color='orange',dash='dash')),secondary_y=False)
    fig.add_hline(y=1,line_dash="dash",line_color="gray",secondary_y=False); fig.update_layout(title="Scree Plot",height=450)
    st.plotly_chart(fig,use_container_width=True)
    n_retain=st.slider("PC retain:",1,p,n_pa)
    st.subheader("2c. Loadings")
    ld=pd.DataFrame(loadings[:,:n_retain],columns=[f'PC{i+1}' for i in range(n_retain)],index=vars_pca).round(4)
    st.dataframe(ld,use_container_width=True)
    fig=go.Figure(data=go.Heatmap(z=ld.values,x=ld.columns,y=ld.index,colorscale='RdBu_r',zmid=0,text=ld.values.round(3),texttemplate='%{text}'))
    fig.update_layout(title="Loading Heatmap",height=400); st.plotly_chart(fig,use_container_width=True)
    st.subheader("2d. Score Coefficients")
    st.dataframe(pd.DataFrame(eigenvectors[:,:n_retain],columns=[f'PC{i+1}' for i in range(n_retain)],index=vars_pca).round(4),use_container_width=True)
    if n_retain>=2:
        st.subheader("2e. Biplot")
        fig=go.Figure(); fig.add_trace(go.Scatter(x=scores[:,0],y=scores[:,1],mode='markers',marker=dict(size=5,opacity=0.4,color='steelblue'),name='Obs'))
        scale=max(np.abs(scores[:,:2]).max()/np.abs(loadings[:,:2]).max()*0.8,1)
        for i,v in enumerate(vars_pca):
            fig.add_annotation(x=loadings[i,0]*scale,y=loadings[i,1]*scale,text=v,showarrow=True,arrowhead=2,arrowcolor='red',font=dict(color='red'),ax=0,ay=0)
        fig.update_layout(title=f"Biplot PC1({prop_var[0]*100:.1f}%) vs PC2({prop_var[1]*100:.1f}%)",height=550); st.plotly_chart(fig,use_container_width=True)
    st.subheader("2f. Communalities")
    comm=np.sum(loadings[:,:n_retain]**2,axis=1)
    st.dataframe(pd.DataFrame({'Variabel':vars_pca,'Communality':comm.round(4),'Uniqueness':(1-comm).round(4),'Status':['✅' if c>0.5 else '⚠️' for c in comm]}),use_container_width=True,hide_index=True)
    st.subheader("2g. Residual")
    reproduced=loadings[:,:n_retain]@loadings[:,:n_retain].T; np.fill_diagonal(reproduced,1.0); residual=corr_matrix-reproduced
    ut=residual[np.triu_indices_from(residual,k=1)]; nl=np.sum(np.abs(ut)>0.05); pl=nl/len(ut)*100
    st.markdown(f"Residual |r|>0.05: **{nl} ({pl:.1f}%)** — {'✅' if pl<50 else '⚠️'}")
    st.subheader("2h. Interpretasi")
    for j in range(n_retain):
        sig=[(vars_pca[i],loadings[i,j]) for i in range(p) if abs(loadings[i,j])>=0.4]; sig.sort(key=lambda x:abs(x[1]),reverse=True)
        pos=[f"**{v}**(+{l:.3f})" for v,l in sig if l>0]; neg=[f"**{v}**({l:.3f})" for v,l in sig if l<0]
        st.markdown(f"- **PC{j+1}** ({prop_var[j]*100:.1f}%): {' '.join(pos)} {' '.join(neg)}")
    se=df[vars_pca].dropna().copy()
    for j in range(n_retain): se[f'PC{j+1}']=scores[:,j]
    st.download_button("📥 Download PCA",se.to_csv(index=False),"pca.csv","text/csv")

# ============================================================
# FACTOR ANALYSIS
# ============================================================
elif method=='fa':
    st.header("2. Factor Analysis")
    vars_fa=st.multiselect("Variabel:",numeric_cols,default=numeric_cols)
    if len(vars_fa)<2: st.warning("Min 2."); st.stop()
    X=df[vars_fa].dropna().values; n,p=X.shape
    st.subheader("2a. Kelayakan")
    kmo_val,kmo_per=kmo_test(X); bart_chi2,bart_df,bart_p=bartlett_sphericity(X)
    kl=('Sangat Baik' if kmo_val>=0.9 else 'Baik' if kmo_val>=0.8 else 'Cukup' if kmo_val>=0.7 else 'Mediocre' if kmo_val>=0.6 else 'Buruk' if kmo_val>=0.5 else 'Tidak Layak')
    c1,c2,c3=st.columns(3); c1.metric("KMO",f"{kmo_val:.4f} ({kl})"); c2.metric("Bartlett χ²",f"{bart_chi2:.2f}"); c3.metric("p",f"{bart_p:.2e}")
    st.subheader("2b. Jumlah Faktor")
    eig_fa=np.sort(np.linalg.eigvalsh(np.corrcoef(X.T)))[::-1]; pa_m,pa_95=parallel_analysis(X,500)
    n_kaiser=int(np.sum(eig_fa>=1)); n_pa=int(np.sum([eig_fa[i]>pa_95[i] for i in range(p)]))
    fig=go.Figure(); fig.add_trace(go.Scatter(x=list(range(1,p+1)),y=eig_fa,mode='lines+markers',name='Eigenvalue'))
    fig.add_trace(go.Scatter(x=list(range(1,p+1)),y=pa_95,mode='lines+markers',name='PA 95%',line=dict(dash='dash',color='orange')))
    fig.add_hline(y=1,line_dash="dash",line_color="gray"); fig.update_layout(title="Scree+PA",height=400); st.plotly_chart(fig,use_container_width=True)
    n_factors=st.slider("Jumlah Faktor:",1,p,max(n_pa,1)); rotation=st.selectbox("Rotasi:",['varimax','promax','none'])
    rot=rotation if rotation!='none' else 'varimax'
    L,comm,uniq,_,corr_orig,reproduced,residual,_=factor_analysis(X,n_factors,rotation=rot)
    if rotation=='none':
        ev_f,evec_f=np.linalg.eigh(np.corrcoef(X.T)); ix_f=np.argsort(ev_f)[::-1]
        L=evec_f[:,:n_factors]*np.sqrt(np.maximum(ev_f[ix_f[:n_factors]],0)); comm=np.sum(L**2,axis=1); uniq=1-comm
        reproduced=L@L.T; np.fill_diagonal(reproduced,1.0); residual=np.corrcoef(X.T)-reproduced
    st.subheader("2c. Factor Loadings")
    st.dataframe(pd.DataFrame(L,columns=[f'F{i+1}' for i in range(n_factors)],index=vars_fa).round(4),use_container_width=True)
    for j in range(n_factors):
        sig=[(vars_fa[i],L[i,j]) for i in range(p) if abs(L[i,j])>=0.5]; sig.sort(key=lambda x:abs(x[1]),reverse=True)
        st.markdown(f"- **F{j+1}:** {', '.join([f'{v}({l:+.3f})' for v,l in sig]) if sig else 'Tidak ada loading dominan'}")
    if n_factors>=2:
        fig=go.Figure()
        for i,v in enumerate(vars_fa):
            fig.add_trace(go.Scatter(x=[0,L[i,0]],y=[0,L[i,1]],mode='lines+text',text=['',v],textposition='top center',line=dict(width=2),showlegend=False))
        fig.add_shape(type='circle',x0=-1,y0=-1,x1=1,y1=1,line=dict(dash='dash',color='gray'))
        fig.update_layout(title="Loading Plot",height=500,xaxis=dict(range=[-1.2,1.2]),yaxis=dict(range=[-1.2,1.2],scaleanchor='x')); st.plotly_chart(fig,use_container_width=True)
    st.subheader("2d. Communalities")
    st.dataframe(pd.DataFrame({'Variabel':vars_fa,'Extraction':comm.round(4),'Uniqueness':uniq.round(4),'Status':['✅' if c>=0.5 else '⚠️' for c in comm]}),use_container_width=True,hide_index=True)
    st.subheader("2e. Variance Explained")
    ve=np.sum(L**2,axis=0); vp=ve/p*100
    st.dataframe(pd.DataFrame({'Factor':[f'F{i+1}' for i in range(n_factors)],'SS Loadings':ve.round(4),'%Var':vp.round(2),'%Cum':np.cumsum(vp).round(2)}),use_container_width=True,hide_index=True)
    st.subheader("2f. Residual")
    ut=residual[np.triu_indices_from(residual,k=1)]; rmsr=np.sqrt(np.mean(ut**2)); nl=np.sum(np.abs(ut)>0.05); pl=nl/len(ut)*100
    st.markdown(f"RMSR: **{rmsr:.4f}** | Residual>0.05: **{nl} ({pl:.1f}%)**")
    Xs_fa,_,_=standardize(X); fs=Xs_fa@np.linalg.pinv(L.T)
    se=df[vars_fa].dropna().copy()
    for j in range(n_factors): se[f'Factor{j+1}']=fs[:,j]
    st.download_button("📥 Download FA",se.to_csv(index=False),"fa.csv","text/csv")

# ============================================================
# MANOVA
# ============================================================
elif method=='manova':
    st.header("2. MANOVA")
    resp_vars=st.multiselect("Variabel Respon (≥2):",numeric_cols,default=numeric_cols[:min(3,len(numeric_cols))])
    group_col=st.selectbox("Variabel Grup:",cat_cols+[c for c in numeric_cols if df[c].nunique()<=10 and c not in resp_vars])
    alpha=st.selectbox("α",[0.01,0.05,0.10],index=1)
    if len(resp_vars)<2: st.warning("Min 2 respon."); st.stop()
    panel=df[resp_vars+[group_col]].dropna(); X=panel[resp_vars].values; groups=panel[group_col].values
    unique_g=np.unique(groups); k_groups=len(unique_g)
    if k_groups<2: st.error("Min 2 grup."); st.stop()
    st.subheader("2a. Asumsi")
    tab_a1,tab_a2,tab_a3=st.tabs(["Normalitas","Box's M","Levene"])
    with tab_a1:
        for g in unique_g:
            Xg=panel[panel[group_col]==g][resp_vars].values
            if len(Xg)>len(resp_vars)+1:
                _,_,pg_s,_,_,pg_k=mardia_test(Xg)
                st.markdown(f"**{g}**(n={len(Xg)}): Skew p={pg_s:.4f}{'✅' if pg_s>alpha else '❌'} Kurt p={pg_k:.4f}{'✅' if pg_k>alpha else '❌'}")
    with tab_a2:
        gdata=[panel[panel[group_col]==g][resp_vars].values for g in unique_g]
        if all(len(g)>X.shape[1] for g in gdata):
            Mv,chi2b,dfb,pb=box_m_test(gdata)
            st.markdown(f"Box's M={Mv:.4f}, χ²={chi2b:.4f}, p={pb:.6f} {'✅ Homogen' if pb>alpha else '⚠️ Tidak homogen'}")
    with tab_a3:
        lev=levene_multivariate(X,groups)
        st.dataframe(pd.DataFrame([{'Var':resp_vars[j],'F':round(lev[j][0],4),'p':round(lev[j][1],6),'?':'✅' if lev[j][1]>alpha else '❌'} for j in range(len(resp_vars))]),use_container_width=True,hide_index=True)
    st.subheader("2b. Tabel MANOVA")
    manova_res,eigvals_m,B,W,T_m=manova_test(X,groups)
    st.dataframe(manova_res,use_container_width=True,hide_index=True)
    with st.expander("SSCP"):
        st.markdown("**Between:**"); st.dataframe(pd.DataFrame(B,index=resp_vars,columns=resp_vars).round(4),use_container_width=True)
        st.markdown("**Within:**"); st.dataframe(pd.DataFrame(W,index=resp_vars,columns=resp_vars).round(4),use_container_width=True)
    st.subheader("2c. Effect Size")
    for var in resp_vars:
        gd_v=[panel[panel[group_col]==g][var].values for g in unique_g]; fv,pv=stats.f_oneway(*gd_v); gm_v=panel[var].mean()
        ss_b=sum(len(gd)*(gd.mean()-gm_v)**2 for gd in gd_v); ss_w=sum(np.sum((gd-gd.mean())**2) for gd in gd_v)
        eta2=ss_b/(ss_b+ss_w) if (ss_b+ss_w)>0 else 0
        st.markdown(f"- **{var}**: F={fv:.4f}, p={pv:.6f}, η²={eta2:.4f}")
    st.subheader("2d. Mean Profile")
    means=panel.groupby(group_col)[resp_vars].mean(); fig=go.Figure()
    for g in unique_g: fig.add_trace(go.Scatter(x=resp_vars,y=means.loc[g].values,mode='lines+markers',name=str(g)))
    fig.update_layout(height=400); st.plotly_chart(fig,use_container_width=True)
    st.subheader("2e. Post-Hoc")
    pairs=list(combinations(unique_g,2)); ab=alpha/len(pairs); ph=[]
    for g1,g2 in pairs:
        X1=panel[panel[group_col]==g1][resp_vars].values; X2=panel[panel[group_col]==g2][resp_vars].values
        T2,Fp,d1,d2,pp,_,_=hotelling_t2(X1,X2)
        ph.append({'Pair':f'{g1} vs {g2}','T²':round(T2,4),'F':round(Fp,4),'p':round(pp,6),'Sig?':'✅' if pp<ab else '❌'})
    st.dataframe(pd.DataFrame(ph),use_container_width=True,hide_index=True)

# ============================================================
# HOTELLING T²
# ============================================================
elif method=='hotelling':
    st.header("2. Hotelling's T²")
    resp_vars=st.multiselect("Respon:",numeric_cols,default=numeric_cols[:min(3,len(numeric_cols))])
    group_col=st.selectbox("Grup (2 level):",cat_cols+[c for c in numeric_cols if df[c].nunique()==2])
    alpha=st.selectbox("α",[0.01,0.05,0.10],index=1)
    if len(resp_vars)<2: st.warning("Min 2."); st.stop()
    panel=df[resp_vars+[group_col]].dropna(); gl=panel[group_col].unique()
    if len(gl)!=2: st.error("Harus 2 grup."); st.stop()
    X1=panel[panel[group_col]==gl[0]][resp_vars].values; X2=panel[panel[group_col]==gl[1]][resp_vars].values; pv=len(resp_vars); n1,n2=len(X1),len(X2)
    st.subheader("2a. Asumsi")
    for g,Xg in [(gl[0],X1),(gl[1],X2)]:
        _,_,ps,_,_,pk=mardia_test(Xg)
        st.markdown(f"**{g}**(n={len(Xg)}): Skew p={ps:.4f}{'✅' if ps>alpha else '❌'} Kurt p={pk:.4f}{'✅' if pk>alpha else '❌'}")
    Mv,chi2b,dfb,pb=box_m_test([X1,X2]); st.markdown(f"Box's M={Mv:.4f}, p={pb:.6f} {'✅' if pb>alpha else '⚠️'}")
    st.subheader("2b. Hasil")
    T2,F_val,df1,df2,p_val,Sp,diff=hotelling_t2(X1,X2); Fc=stats.f.ppf(1-alpha,df1,df2)
    try: Spi=np.linalg.inv(Sp)
    except: Spi=np.linalg.pinv(Sp)
    md=np.sqrt(diff@Spi@diff)
    st.markdown(f"| | Nilai |\n|---|---|\n| T² | {T2:.4f} |\n| F | {F_val:.4f} (Fc={Fc:.4f}) |\n| df | ({df1},{df2}) |\n| p | {p_val:.6f} |\n| Mahalanobis D | {md:.4f} |\n| Keputusan | {'🔴 Tolak H₀' if p_val<alpha else '🟢 Gagal tolak'} |")
    st.subheader("2c. Mean Comparison")
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=resp_vars,y=X1.mean(0),mode='lines+markers',name=str(gl[0]),error_y=dict(type='data',array=X1.std(0,ddof=1)/np.sqrt(n1))))
    fig.add_trace(go.Scatter(x=resp_vars,y=X2.mean(0),mode='lines+markers',name=str(gl[1]),error_y=dict(type='data',array=X2.std(0,ddof=1)/np.sqrt(n2))))
    fig.update_layout(title="Mean ±SE",height=400); st.plotly_chart(fig,use_container_width=True)
    st.subheader("2d. Simultaneous CI")
    tb=stats.t.ppf(1-alpha/(2*pv),n1+n2-2); ci_rows=[]
    for j,var in enumerate(resp_vars):
        dj=X1[:,j].mean()-X2[:,j].mean(); sej=np.sqrt(Sp[j,j]*(1/n1+1/n2)); cl=dj-tb*sej; cu=dj+tb*sej
        psd=np.sqrt(((n1-1)*np.var(X1[:,j],ddof=1)+(n2-1)*np.var(X2[:,j],ddof=1))/(n1+n2-2)); cd=dj/psd if psd>0 else 0
        ci_rows.append({'Var':var,'Diff':round(dj,4),"Cohen's d":round(cd,4),'CI':f'[{cl:.4f},{cu:.4f}]','Sig?':'✅' if cl>0 or cu<0 else '❌'})
    st.dataframe(pd.DataFrame(ci_rows),use_container_width=True,hide_index=True)
    fig=go.Figure()
    for r in ci_rows:
        vals=[float(x) for x in r['CI'].strip('[]').split(',')]; color='red' if r['Sig?']=='✅' else 'gray'
        fig.add_trace(go.Scatter(x=vals,y=[r['Var']]*2,mode='lines',line=dict(color=color,width=3),showlegend=False))
        fig.add_trace(go.Scatter(x=[r['Diff']],y=[r['Var']],mode='markers',marker=dict(size=10,color=color,symbol='diamond'),showlegend=False))
    fig.add_vline(x=0,line_dash="dash"); fig.update_layout(title="Forest Plot",height=350); st.plotly_chart(fig,use_container_width=True)

# ============================================================
# LDA
# ============================================================
elif method=='lda':
    st.header("2. Linear Discriminant Analysis")
    pred_vars=st.multiselect("Prediktor:",numeric_cols,default=numeric_cols[:min(3,len(numeric_cols))])
    group_col=st.selectbox("Grup:",cat_cols+[c for c in numeric_cols if df[c].nunique()<=10 and c not in pred_vars])
    alpha=st.selectbox("α",[0.01,0.05,0.10],index=1)
    if len(pred_vars)<2: st.warning("Min 2."); st.stop()
    panel=df[pred_vars+[group_col]].dropna(); X=panel[pred_vars].values; y=panel[group_col].values
    classes=np.unique(y); k_cls=len(classes); n_total,p_vars=X.shape
    if k_cls<2: st.error("Min 2 grup."); st.stop()
    st.subheader("2a. Asumsi")
    for g in classes:
        Xg=panel[panel[group_col]==g][pred_vars].values
        if len(Xg)>p_vars+1:
            _,_,ps,_,_,pk=mardia_test(Xg); st.markdown(f"**{g}**(n={len(Xg)}): Skew p={ps:.4f}{'✅' if ps>alpha else '❌'} Kurt p={pk:.4f}{'✅' if pk>alpha else '❌'}")
    gdata_l=[panel[panel[group_col]==g][pred_vars].values for g in classes]
    if all(len(g)>p_vars for g in gdata_l): Mv,chi2b,dfb,pb=box_m_test(gdata_l); st.markdown(f"Box's M={Mv:.4f}, p={pb:.6f} {'✅' if pb>alpha else '⚠️'}")
    W_lda,eigvals_lda,scores_lda,y_pred,acc,priors,Sw,Sb,group_means=lda_fit(X,y); n_disc=W_lda.shape[1]
    st.subheader("2b. Discriminant Functions")
    ep=eigvals_lda/eigvals_lda.sum() if eigvals_lda.sum()>0 else eigvals_lda; ccl=np.sqrt(eigvals_lda/(1+eigvals_lda))
    st.dataframe(pd.DataFrame({'LD':[f'LD{i+1}' for i in range(n_disc)],'Eigenvalue':eigvals_lda.round(4),'%Var':(ep*100).round(2),'Can r':ccl.round(4)}),use_container_width=True,hide_index=True)
    for m in range(n_disc):
        wm=np.prod(1.0/(1.0+eigvals_lda[m:])); chi2m=-(n_total-1-(p_vars+k_cls)/2)*np.log(max(wm,1e-300))
        dfm=(p_vars-m)*(k_cls-1-m); pm=1-stats.chi2.cdf(chi2m,max(dfm,1))
        st.markdown(f"- LD≥{m+1}: Λ={wm:.4f}, χ²={chi2m:.4f}, p={pm:.6f} {'✅' if pm<alpha else '❌'}")
    tab_raw,tab_struct=st.tabs(["Raw","Structure"])
    with tab_raw: st.dataframe(pd.DataFrame(W_lda,index=pred_vars,columns=[f'LD{i+1}' for i in range(n_disc)]).round(4),use_container_width=True)
    with tab_struct:
        struct=np.zeros((p_vars,n_disc))
        for j in range(p_vars):
            for d in range(n_disc): struct[j,d]=np.corrcoef(X[:,j],scores_lda[:,d])[0,1]
        st.dataframe(pd.DataFrame(struct,index=pred_vars,columns=[f'LD{i+1}' for i in range(n_disc)]).round(4),use_container_width=True)
    st.subheader("2c. Centroids")
    centroids={c:scores_lda[y==c].mean(0) for c in classes}
    st.dataframe(pd.DataFrame(centroids,index=[f'LD{i+1}' for i in range(n_disc)]).T.round(4),use_container_width=True)
    st.subheader("2d. Classification")
    st.metric("Akurasi",f"{acc:.2%}")
    cm=np.zeros((k_cls,k_cls),dtype=int)
    for i,c1 in enumerate(classes):
        for j,c2 in enumerate(classes): cm[i,j]=np.sum((y==c1)&(y_pred==c2))
    st.dataframe(pd.DataFrame(cm,index=[f'Act:{c}' for c in classes],columns=[f'Pred:{c}' for c in classes]),use_container_width=True)
    with st.spinner("LOO-CV..."):
        y_cv=np.empty_like(y)
        for i in range(n_total):
            Xt=np.delete(X,i,0); yt=np.delete(y,i); Sw_cv=np.zeros((p_vars,p_vars)); mc_cv={}; pr_cv={}
            for c in np.unique(yt):
                Xc=Xt[yt==c]; mc_cv[c]=Xc.mean(0); pr_cv[c]=len(Xc)/len(yt); Sw_cv+=(Xc-Xc.mean(0)).T@(Xc-Xc.mean(0))
            try: Swi=np.linalg.inv(Sw_cv)
            except: Swi=np.linalg.pinv(Sw_cv)
            bc=None; bd=np.inf
            for c in np.unique(yt):
                d=(X[i]-mc_cv[c])@Swi@(X[i]-mc_cv[c])-2*np.log(pr_cv[c])
                if d<bd: bd=d; bc=c
            y_cv[i]=bc
    st.metric("LOO-CV",f"{np.mean(y_cv==y):.2%}")
    if n_disc>=2:
        fig=px.scatter(x=scores_lda[:,0],y=scores_lda[:,1],color=y.astype(str),labels={'x':'LD1','y':'LD2','color':'Group'},title="LDA Scores")
        for c in classes: fig.add_trace(go.Scatter(x=[centroids[c][0]],y=[centroids[c][1]],mode='markers',marker=dict(symbol='x',size=15,color='black'),name=f'C:{c}'))
        fig.update_layout(height=550); st.plotly_chart(fig,use_container_width=True)

# ============================================================
# K-MEANS
# ============================================================
elif method=='kmeans':
    st.header("2. K-Means Clustering")
    vars_cl=st.multiselect("Variabel:",numeric_cols,default=numeric_cols)
    if len(vars_cl)<2: st.warning("Min 2."); st.stop()
    X_raw=df[vars_cl].dropna().values; Xs,mu_c,sd_c=standardize(X_raw); n_obs=len(Xs)
    st.subheader("2a. Optimal k")
    max_k=min(10,n_obs-1); inertias=[]; sil_scores=[]; all_labels={}; all_centers={}
    with st.spinner("Evaluasi k..."):
        for ke in range(2,max_k+1):
            lk,ck,ik=kmeans_fit(Xs,ke); inertias.append(ik); all_labels[ke]=lk; all_centers[ke]=ck
            sk,_=silhouette_score(Xs,lk); sil_scores.append(sk)
    opt_sil=np.argmax(sil_scores)+2
    fig=make_subplots(rows=1,cols=2,subplot_titles=("Elbow","Silhouette"))
    fig.add_trace(go.Scatter(x=list(range(2,max_k+1)),y=inertias,mode='lines+markers'),row=1,col=1)
    fig.add_trace(go.Scatter(x=list(range(2,max_k+1)),y=sil_scores,mode='lines+markers',line=dict(color='green')),row=1,col=2)
    fig.update_layout(height=350,showlegend=False); st.plotly_chart(fig,use_container_width=True)
    k=st.slider("k:",2,max_k,opt_sil)
    labels=all_labels.get(k); centers=all_centers.get(k)
    if labels is None: labels,centers,_=kmeans_fit(Xs,k)
    sil_mean,sil_vals=silhouette_score(Xs,labels)
    c1,c2,c3=st.columns(3); c1.metric("k",k); c2.metric("Silhouette",f"{sil_mean:.4f}")
    c3.metric("Kualitas",'Kuat' if sil_mean>0.70 else 'Baik' if sil_mean>0.50 else 'Sedang' if sil_mean>0.25 else 'Lemah')
    df_res=df[vars_cl].dropna().copy(); df_res['Cluster']=labels.astype(str)
    st.subheader("2b. Profil")
    centers_orig=centers*sd_c+mu_c
    cd=pd.DataFrame(centers_orig,columns=vars_cl,index=[f'Cl{i}' for i in range(k)]).round(4); cd['N']=[int(np.sum(labels==i)) for i in range(k)]
    st.dataframe(cd,use_container_width=True)
    fig=go.Figure()
    for cl in range(k):
        vals=list(centers[cl])+[centers[cl][0]]; fig.add_trace(go.Scatterpolar(r=vals,theta=vars_cl+[vars_cl[0]],fill='toself',name=f'Cl{cl}'))
    fig.update_layout(height=500); st.plotly_chart(fig,use_container_width=True)
    st.subheader("2c. Visualisasi")
    if len(vars_cl)>=2:
        xv=st.selectbox("X:",vars_cl,index=0,key='kx'); yv=st.selectbox("Y:",vars_cl,index=min(1,len(vars_cl)-1),key='ky')
        fig=px.scatter(df_res,x=xv,y=yv,color='Cluster'); ix_k=vars_cl.index(xv); iy_k=vars_cl.index(yv)
        fig.add_trace(go.Scatter(x=centers_orig[:,ix_k],y=centers_orig[:,iy_k],mode='markers',marker=dict(symbol='x',size=15,color='black'),name='Centroids'))
        fig.update_layout(height=500); st.plotly_chart(fig,use_container_width=True)
    st.subheader("2d. Silhouette")
    sdf=pd.DataFrame({'Cluster':labels.astype(str),'Silhouette':sil_vals})
    fig=px.box(sdf,x='Cluster',y='Silhouette',color='Cluster'); fig.add_hline(y=sil_mean,line_dash="dash"); fig.add_hline(y=0,line_dash="dot",line_color="red")
    fig.update_layout(height=400); st.plotly_chart(fig,use_container_width=True)
    st.subheader("2e. ANOVA")
    for j,var in enumerate(vars_cl):
        gs=[X_raw[labels==cl,j] for cl in range(k)]; fv,pv=stats.f_oneway(*gs)
        st.markdown(f"- **{var}**: F={fv:.4f}, p={pv:.6f} {'✅' if pv<0.05 else '❌'}")
    st.download_button("📥 Download",df_res.to_csv(index=False),"kmeans.csv","text/csv")

# ============================================================
# HIERARCHICAL
# ============================================================
elif method=='hclust':
    st.header("2. Hierarchical Clustering")
    vars_cl=st.multiselect("Variabel:",numeric_cols,default=numeric_cols)
    if len(vars_cl)<2: st.warning("Min 2."); st.stop()
    X_raw=df[vars_cl].dropna().values; Xs,mu_h,sd_h=standardize(X_raw); n_obs=len(Xs)
    linkage_method=st.selectbox("Linkage:",['ward','complete','average','single'])
    dist_metric='euclidean' if linkage_method=='ward' else st.selectbox("Metrik:",['euclidean','cityblock','cosine'])
    Z=linkage(Xs,method=linkage_method,metric=dist_metric)
    st.subheader("2a. Dendrogram")
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    ct_h=st.slider("Cut height:",float(Z[:,2].min()),float(Z[:,2].max()),float(Z[-3,2]),step=0.1)
    fig_m,ax=plt.subplots(figsize=(14,5))
    dendrogram(Z,ax=ax,leaf_rotation=90,leaf_font_size=7,color_threshold=ct_h,above_threshold_color='gray')
    ax.axhline(y=ct_h,color='red',linestyle='--',alpha=0.7); ax.set_ylabel("Distance"); st.pyplot(fig_m); plt.close()
    st.subheader("2b. Optimal k")
    max_k_h=min(10,n_obs-1); sil_h=[]
    for kh in range(2,max_k_h+1): lh=fcluster(Z,kh,criterion='maxclust')-1; sh,_=silhouette_score(Xs,lh); sil_h.append(sh)
    opt_h=np.argmax(sil_h)+2
    fig=go.Figure(data=[go.Scatter(x=list(range(2,max_k_h+1)),y=sil_h,mode='lines+markers')]); fig.update_layout(title="Silhouette vs k",height=300); st.plotly_chart(fig,use_container_width=True)
    n_clust=st.slider("Clusters:",2,max_k_h,opt_h); labels_hc=fcluster(Z,n_clust,criterion='maxclust')-1
    sil_hc,_=silhouette_score(Xs,labels_hc); st.metric("Silhouette",f"{sil_hc:.4f}")
    df_hc=df[vars_cl].dropna().copy(); df_hc['Cluster']=labels_hc.astype(str)
    st.subheader("2c. Profil")
    ch=np.array([Xs[labels_hc==cl].mean(0) for cl in range(n_clust)]); fig=go.Figure()
    for cl in range(n_clust): vals=list(ch[cl])+[ch[cl][0]]; fig.add_trace(go.Scatterpolar(r=vals,theta=vars_cl+[vars_cl[0]],fill='toself',name=f'Cl{cl}'))
    fig.update_layout(height=500); st.plotly_chart(fig,use_container_width=True)
    if len(vars_cl)>=2: fig=px.scatter(df_hc,x=vars_cl[0],y=vars_cl[1],color='Cluster',title="Cluster Plot"); fig.update_layout(height=500); st.plotly_chart(fig,use_container_width=True)
    st.subheader("2d. Cophenetic")
    coph_r=cophenet(Z,pdist(Xs,dist_metric if dist_metric!='correlation' else 'euclidean'))[0]
    st.metric("Cophenetic r",f"{coph_r:.4f}"); st.markdown(f"{'✅' if coph_r>0.75 else '⚠️' if coph_r>0.50 else '❌'}")
    st.download_button("📥 Download",df_hc.to_csv(index=False),"hclust.csv","text/csv")

# ============================================================
# DBSCAN
# ============================================================
elif method=='dbscan':
    st.header("2. DBSCAN Clustering")
    vars_cl=st.multiselect("Variabel:",numeric_cols,default=numeric_cols)
    if len(vars_cl)<2: st.warning("Min 2."); st.stop()
    X_raw=df[vars_cl].dropna().values; Xs,mu_d,sd_d=standardize(X_raw); n_obs=len(Xs)
    st.subheader("2a. Parameter")
    p_dim=len(vars_cl); sm=max(2*p_dim,5); k_nn=st.slider("k:",2,30,sm)
    dists_all=squareform(pdist(Xs,'euclidean')); k_dists=np.sort(dists_all,axis=1)[:,k_nn]; k_dists_sorted=np.sort(k_dists)[::-1]
    sd2=np.diff(np.diff(k_dists_sorted)); se=k_dists_sorted[np.argmax(sd2)+1] if len(sd2)>0 else float(np.median(k_dists))
    fig=go.Figure(data=[go.Scatter(y=k_dists_sorted,mode='lines')]); fig.add_hline(y=se,line_dash="dash",line_color="red",annotation_text=f"eps≈{se:.3f}")
    fig.update_layout(title=f"{k_nn}-Distance",height=350); st.plotly_chart(fig,use_container_width=True)
    eps=st.slider("eps:",0.05,float(np.percentile(k_dists,99))*2,float(round(se,2)),0.01); min_samples=st.slider("min_samples:",2,30,k_nn)
    labels_db=dbscan_fit(Xs,eps,min_samples); n_clusters=len(set(labels_db)-{-1}); n_noise=np.sum(labels_db==-1)
    st.subheader("2b. Hasil")
    c1,c2,c3=st.columns(3); c1.metric("Clusters",n_clusters); c2.metric("Noise",f"{n_noise}({n_noise/n_obs:.0%})"); c3.metric("n",n_obs)
    if n_clusters>1: nm=labels_db!=-1; sdb,_=silhouette_score(Xs[nm],labels_db[nm]); st.metric("Silhouette",f"{sdb:.4f}")
    df_db=df[vars_cl].dropna().copy(); df_db['Cluster']=labels_db.astype(str)
    if len(vars_cl)>=2: fig=px.scatter(df_db,x=vars_cl[0],y=vars_cl[1],color='Cluster',title="DBSCAN"); fig.update_layout(height=500); st.plotly_chart(fig,use_container_width=True)
    st.download_button("📥 Download",df_db.to_csv(index=False),"dbscan.csv","text/csv")

# ============================================================
# MDS
# ============================================================
elif method=='mds':
    st.header("2. Multidimensional Scaling")
    vars_mds=st.multiselect("Variabel:",numeric_cols,default=numeric_cols)
    if len(vars_mds)<2: st.warning("Min 2."); st.stop()
    X_raw=df[vars_mds].dropna().values; Xs,mu_m,sd_m=standardize(X_raw); n_obs,p_dim=Xs.shape
    dist_m=st.selectbox("Jarak:",['euclidean','cityblock','cosine']); n_dim_mds=st.slider("Dimensi:",2,min(5,p_dim),2)
    D=squareform(pdist(Xs,dist_m))
    nn=D.shape[0]; H=np.eye(nn)-np.ones((nn,nn))/nn; Bm=-0.5*H@(D**2)@H
    evm,evcm=np.linalg.eigh(Bm); ixm=np.argsort(evm)[::-1]; evm=evm[ixm]; evcm=evcm[:,ixm]
    coords=evcm[:,:n_dim_mds]*np.sqrt(np.maximum(evm[:n_dim_mds],0))
    ti=np.sum(np.maximum(evm,0)); expl=np.sum(np.maximum(evm[:n_dim_mds],0))/ti*100 if ti>0 else 0
    Dm=squareform(pdist(coords,'euclidean')); Df=D[np.triu_indices_from(D,k=1)]; Dmf=Dm[np.triu_indices_from(Dm,k=1)]
    s1=np.sqrt(np.sum((Df-Dmf)**2)/np.sum(Df**2)) if np.sum(Df**2)>0 else 0
    sr=np.sum((Df-Dmf)**2); st2=np.sum((Df-np.mean(Df))**2); r2=1-sr/st2 if st2>0 else 0
    sl='Excellent' if s1<0.025 else 'Good' if s1<0.05 else 'Fair' if s1<0.10 else 'Poor'
    c1,c2,c3=st.columns(3); c1.metric("Stress-1",f"{s1:.4f}({sl})"); c2.metric("R²",f"{r2:.4f}"); c3.metric("Var Exp",f"{expl:.1f}%")
    st.subheader("2a. Shepard Diagram")
    fig=go.Figure(); fig.add_trace(go.Scatter(x=Df,y=Dmf,mode='markers',marker=dict(size=3,opacity=0.3,color='steelblue')))
    md=max(Df.max(),Dmf.max()); fig.add_trace(go.Scatter(x=[0,md],y=[0,md],mode='lines',line=dict(color='red',dash='dash')))
    fig.update_layout(title="Shepard",height=450); st.plotly_chart(fig,use_container_width=True)
    st.subheader("2b. MDS Map")
    if n_dim_mds>=2:
        mdf=pd.DataFrame({'Dim1':coords[:,0],'Dim2':coords[:,1]})
        if cat_cols:
            cc=st.selectbox("Warnai:",['(Tidak)']+cat_cols,key='mc')
            if cc!='(Tidak)': mdf['group']=df[cc].iloc[df[vars_mds].dropna().index].values; fig=px.scatter(mdf,x='Dim1',y='Dim2',color='group')
            else: fig=px.scatter(mdf,x='Dim1',y='Dim2')
        else: fig=px.scatter(mdf,x='Dim1',y='Dim2')
        fig.update_layout(height=550); st.plotly_chart(fig,use_container_width=True)
    st.download_button("📥 Download",pd.DataFrame(coords,columns=[f'Dim{i+1}' for i in range(n_dim_mds)]).to_csv(index=False),"mds.csv","text/csv")

# ============================================================
# CCA
# ============================================================
elif method=='cca':
    st.header("2. Canonical Correlation Analysis")
    vars_x=st.multiselect("Set X:",numeric_cols,default=numeric_cols[:min(3,len(numeric_cols))],key='cx')
    remaining=[c for c in numeric_cols if c not in vars_x]
    vars_y=st.multiselect("Set Y:",remaining,default=remaining[:min(3,len(remaining))],key='cy')
    alpha=st.selectbox("α",[0.01,0.05,0.10],index=1)
    if len(vars_x)<2 or len(vars_y)<2: st.warning("Min 2 per set."); st.stop()
    panel=df[vars_x+vars_y].dropna(); X=panel[vars_x].values; Y=panel[vars_y].values
    nc=len(panel); pxc=len(vars_x); pyc=len(vars_y); np_c=min(pxc,pyc)
    can_corr,A,B,Xsc,Ysc=cca_analysis(X,Y)
    st.subheader("2a. Significance")
    sig_rows=[]
    for m in range(np_c):
        wm=np.prod(1-can_corr[m:]**2); chi2m=-(nc-1-0.5*(pxc+pyc+1))*np.log(max(wm,1e-300))
        dfm=(pxc-m)*(pyc-m); pm=1-stats.chi2.cdf(chi2m,max(dfm,1))
        sig_rows.append({'CV':f'CV{m+1}','r':round(can_corr[m],4),'r²':round(can_corr[m]**2,4),'Wilks':round(wm,4),'p':round(pm,6),'Sig?':'✅' if pm<alpha else '❌'})
    st.dataframe(pd.DataFrame(sig_rows),use_container_width=True,hide_index=True)
    st.subheader("2b. Loadings")
    nrc=min(np_c,3)
    lx=np.zeros((pxc,nrc)); ly=np.zeros((pyc,nrc))
    for i in range(pxc):
        for j in range(nrc): lx[i,j]=np.corrcoef(X[:,i],Xsc[:,j])[0,1]
    for i in range(pyc):
        for j in range(nrc): ly[i,j]=np.corrcoef(Y[:,i],Ysc[:,j])[0,1]
    fig=make_subplots(rows=1,cols=2,subplot_titles=("X","Y"))
    fig.add_trace(go.Heatmap(z=lx,x=[f'CV{i+1}' for i in range(nrc)],y=vars_x,colorscale='RdBu_r',zmid=0,text=np.round(lx,3),texttemplate='%{text}',showscale=False),row=1,col=1)
    fig.add_trace(go.Heatmap(z=ly,x=[f'CV{i+1}' for i in range(nrc)],y=vars_y,colorscale='RdBu_r',zmid=0,text=np.round(ly,3),texttemplate='%{text}'),row=1,col=2)
    fig.update_layout(height=max(350,50*max(pxc,pyc))); st.plotly_chart(fig,use_container_width=True)
    st.subheader("2c. Scores")
    fig=go.Figure(); fig.add_trace(go.Scatter(x=Xsc[:,0],y=Ysc[:,0],mode='markers',marker=dict(size=5,opacity=0.5)))
    sl=np.polyfit(Xsc[:,0],Ysc[:,0],1); xr=np.linspace(Xsc[:,0].min(),Xsc[:,0].max(),100)
    fig.add_trace(go.Scatter(x=xr,y=sl[0]*xr+sl[1],mode='lines',line=dict(color='red',dash='dash'),name=f'r={can_corr[0]:.3f}'))
    fig.update_layout(title=f"CV1 (r={can_corr[0]:.4f})",height=450); st.plotly_chart(fig,use_container_width=True)

# ============================================================
# CORRESPONDENCE ANALYSIS
# ============================================================
elif method=='ca':
    st.header("2. Correspondence Analysis")
    if len(cat_cols)<2: st.warning("Butuh ≥2 kategorikal."); st.stop()
    row_var=st.selectbox("Baris:",cat_cols,index=0,key='cr'); col_var=st.selectbox("Kolom:",[c for c in cat_cols if c!=row_var],key='cc')
    ct=pd.crosstab(df[row_var],df[col_var]); nca=ct.values.sum()
    st.subheader("2a. Kontingensi")
    ctt=ct.copy(); ctt['Total']=ct.sum(1); ctt.loc['Total']=ctt.sum(); st.dataframe(ctt,use_container_width=True)
    O=ct.values.astype(float); rt=O.sum(1,keepdims=True); cc_v=O.sum(0,keepdims=True)
    E=rt*cc_v/nca; chi2ca=np.sum((O-E)**2/E); dfc=(O.shape[0]-1)*(O.shape[1]-1)
    pch=1-stats.chi2.cdf(chi2ca,dfc); cv=np.sqrt(chi2ca/(nca*(min(O.shape)-1)))
    st.markdown(f"χ²={chi2ca:.4f}, df={dfc}, p={pch:.6f} {'✅' if pch<0.05 else '❌'} | V={cv:.4f}")
    row_coords,col_coords,inertias_ca,total_inertia,prop_inertia=ca_analysis(ct.values); nd=len(inertias_ca)
    if nd>=2:
        st.subheader("2b. CA Biplot")
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=row_coords[:,0],y=row_coords[:,1],mode='markers+text',text=ct.index.astype(str),textposition='top center',marker=dict(size=10,color='blue',symbol='circle'),name=row_var))
        fig.add_trace(go.Scatter(x=col_coords[:,0],y=col_coords[:,1],mode='markers+text',text=ct.columns.astype(str),textposition='bottom center',marker=dict(size=10,color='red',symbol='diamond'),name=col_var))
        fig.add_hline(y=0,line_dash="dot"); fig.add_vline(x=0,line_dash="dot")
        fig.update_layout(title=f"CA Biplot (D1:{prop_inertia[0]*100:.1f}%,D2:{prop_inertia[1]*100:.1f}%)",height=600)
        st.plotly_chart(fig,use_container_width=True)

# ============================================================
# UJI NORMALITAS MV
# ============================================================
elif method=='mvnorm':
    st.header("2. Uji Normalitas Multivariat")
    vars_norm=st.multiselect("Variabel:",numeric_cols,default=numeric_cols)
    if len(vars_norm)<2: st.warning("Min 2."); st.stop()
    X=df[vars_norm].dropna().values; n,p=X.shape; alpha=st.selectbox("α",[0.01,0.05,0.10],index=1)
    st.subheader("2a. Uji Formal")
    b1,chi2_s,p_s,b2,z_k,p_k=mardia_test(X); hz_s,hz_z,hz_p=hz_test(X)
    try: Hr,pr=royston_test(X)
    except: Hr,pr=np.nan,np.nan
    st.markdown(f"""| Uji | Stat | p | ? |
|---|---|---|---|
| Mardia Skew | χ²={chi2_s:.4f} | {p_s:.6f} | {'✅' if p_s>alpha else '❌'} |
| Mardia Kurt | z={z_k:.4f} | {p_k:.6f} | {'✅' if p_k>alpha else '❌'} |
| Henze-Zirkler | HZ={hz_s:.4f} | {hz_p:.6f} | {'✅' if hz_p>alpha else '❌'} |
| Royston | H={Hr:.4f} | {pr:.6f} | {'✅' if pr>alpha else '❌'} |""")
    np_t=sum([p_s>alpha,p_k>alpha,hz_p>alpha,(pr>alpha if not np.isnan(pr) else False)])
    st.markdown(f"**{np_t}/4 normal** → {'✅ Terpenuhi' if np_t>=2 else '⚠️ Tidak'}")
    st.subheader("2b. Q-Q Plot")
    mX=X.mean(0)
    try: ci=np.linalg.inv(np.cov(X.T,ddof=1))
    except: ci=np.linalg.pinv(np.cov(X.T,ddof=1))
    msq=np.array([((x-mX)@ci@(x-mX)) for x in X]); ms=np.sort(msq); tq=stats.chi2.ppf(np.linspace(1/(n+1),n/(n+1),n),df=p)
    fig=go.Figure(); fig.add_trace(go.Scatter(x=tq,y=ms,mode='markers',marker=dict(size=4,opacity=0.5,color='steelblue')))
    fig.add_trace(go.Scatter(x=[0,max(tq)],y=[0,max(tq)],mode='lines',line=dict(color='red',dash='dash')))
    fig.update_layout(title=f"Q-Q (D² vs χ²({p}))",height=500); st.plotly_chart(fig,use_container_width=True)
    co=stats.chi2.ppf(0.975,p); on=np.sum(msq>co); st.markdown(f"**Outlier MV** (D²>χ²₉₇.₅={co:.4f}): **{on}** ({on/n:.1%})")

# ============================================================
# FOOTER
# ============================================================
st.divider()
st.markdown("""
<div style='text-align:center; color:gray; font-size:0.85rem;'>
<b>Aplikasi Analisis Statistik Multivariat — Versi Lengkap</b><br>
12 Metode · 5 Data Demo · Semua from scratch (NumPy/SciPy) · © 2026
</div>
""", unsafe_allow_html=True)
