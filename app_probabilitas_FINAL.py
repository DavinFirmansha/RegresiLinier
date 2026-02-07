import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import comb, factorial, gammaln, beta as beta_func
from scipy.integrate import quad
from itertools import combinations, permutations, product
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')
st.set_page_config(page_title="Analisis Probabilitas Pro",layout="wide",page_icon="🎲")
st.title("🎲 Aplikasi Analisis Probabilitas — Lengkap & Interaktif")
st.caption("Mata Kuliah Probabilitas · 10 Modul · 40+ Topik")

def fmt(x,d=6):
    if abs(x)<1e-10: return "0"
    if abs(x-1)<1e-10: return "1"
    return f"{x:.{d}f}".rstrip('0').rstrip('.')
def safe_comb(n,k):
    if k<0 or k>n: return 0
    return int(comb(n,k,exact=True))
def safe_perm(n,k):
    if k<0 or k>n: return 0
    return int(factorial(n,exact=True)//factorial(n-k,exact=True))
def plot_disc(xv,pmf,cdf,title,xl="x"):
    fig=make_subplots(rows=1,cols=2,subplot_titles=("PMF","CDF"))
    fig.add_trace(go.Bar(x=xv,y=pmf,marker_color='steelblue',opacity=0.8),row=1,col=1)
    fig.add_trace(go.Scatter(x=xv,y=cdf,mode='lines+markers',line=dict(color='crimson',width=2),marker=dict(size=4)),row=1,col=2)
    fig.update_layout(title=title,height=400,showlegend=False);return fig
def plot_cont(xv,pdf,cdf,title,xl="x",sx=None,sp=None,sl=None):
    fig=make_subplots(rows=1,cols=2,subplot_titles=("PDF","CDF"))
    fig.add_trace(go.Scatter(x=xv,y=pdf,fill='tozeroy',fillcolor='rgba(70,130,180,0.2)',line=dict(color='steelblue',width=2),name='f(x)'),row=1,col=1)
    if sx is not None and sp is not None:
        fig.add_trace(go.Scatter(x=sx,y=sp,fill='tozeroy',fillcolor='rgba(220,20,60,0.35)',line=dict(width=0),name=sl or'Area'),row=1,col=1)
    fig.add_trace(go.Scatter(x=xv,y=cdf,line=dict(color='crimson',width=2),name='F(x)'),row=1,col=2)
    fig.update_layout(title=title,height=400);return fig
def dsummary(mean,var,std,skew,kurt,mode=None,med=None,mgf=None):
    r=[("Mean",fmt(mean,4)),("Variance",fmt(var,4)),("Std Dev",fmt(std,4)),("Skewness",fmt(skew,4)),("Kurtosis",fmt(kurt,4))]
    if mode is not None: r.insert(1,("Mode",str(mode)))
    if med is not None: r.insert(2,("Median",fmt(med,4) if isinstance(med,(int,float)) else str(med)))
    if mgf: r.append(("MGF",mgf))
    return pd.DataFrame(r,columns=["Ukuran","Nilai"])

st.sidebar.header("🎲 Menu")
module=st.sidebar.selectbox("Modul:",['dasar','kombinatorik','diskrit','kontinu','joint','clt','markov','montecarlo','bayes','fitting'],
format_func=lambda x:{'dasar':'1. Dasar Probabilitas','kombinatorik':'2. Kombinatorik','diskrit':'3. Distribusi Diskrit','kontinu':'4. Distribusi Kontinu','joint':'5. Distribusi Gabungan','clt':'6. CLT','markov':'7. Rantai Markov','montecarlo':'8. Monte Carlo','bayes':'9. Teorema Bayes','fitting':'10. Distribution Fitting'}[x])

if module=='dasar':
    st.header("📐 Dasar Probabilitas")
    sub=st.selectbox("Topik:",['Aturan Probabilitas','Probabilitas Kondisional','Independensi','Hukum Total & Bayes','Kalkulator Himpunan'])
    if sub=='Aturan Probabilitas':
        st.markdown("| Aturan | Formula |\n|---|---|\n| Komplemen | P(A')=1-P(A) |\n| Penjumlahan | P(AuB)=P(A)+P(B)-P(AnB) |\n| Perkalian | P(AnB)=P(A)P(B given A) |")
        c1,c2,c3=st.columns(3);pa=c1.number_input("P(A)",0.0,1.0,0.5,0.01);pb=c2.number_input("P(B)",0.0,1.0,0.4,0.01);pab=c3.number_input("P(AnB)",0.0,1.0,0.2,0.01)
        if pab<=min(pa,pb):
            pu=pa+pb-pab;pba=pab/pa if pa>0 else 0;pab2=pab/pb if pb>0 else 0
            st.dataframe(pd.DataFrame([("P(AuB)",fmt(pu,4)),("P(A')",fmt(1-pa,4)),("P(B|A)",fmt(pba,4)),("P(A|B)",fmt(pab2,4)),("Independen?","Ya" if abs(pab-pa*pb)<1e-9 else "Tidak")],columns=["","Nilai"]),use_container_width=True,hide_index=True)
            fig=go.Figure();fig.add_shape(type="circle",x0=0,y0=0,x1=2,y1=2,line=dict(color="steelblue",width=3),fillcolor="rgba(70,130,180,0.2)")
            fig.add_shape(type="circle",x0=1,y0=0,x1=3,y1=2,line=dict(color="crimson",width=3),fillcolor="rgba(220,20,60,0.2)")
            fig.add_annotation(x=0.6,y=1,text=f"A<br>{fmt(pa-pab,3)}",showarrow=False);fig.add_annotation(x=1.5,y=1,text=f"AnB<br>{fmt(pab,3)}",showarrow=False)
            fig.add_annotation(x=2.4,y=1,text=f"B<br>{fmt(pb-pab,3)}",showarrow=False);fig.update_layout(height=300,xaxis=dict(visible=False,range=[-0.5,3.5]),yaxis=dict(visible=False,range=[-0.5,2.5],scaleanchor='x'))
            st.plotly_chart(fig,use_container_width=True)
    elif sub=='Probabilitas Kondisional':
        st.latex(r"P(A|B)=\frac{P(A\cap B)}{P(B)}")
        c1,c2=st.columns(2);pab_c=c1.number_input("P(AnB)",0.0,1.0,0.15,0.01,key='ca');pb_c=c2.number_input("P(B)",0.001,1.0,0.30,0.01,key='cb')
        st.success(f"P(A|B) = {pab_c/pb_c:.6f}")
        st.markdown("### Tabel Kontingensi");c1,c2=st.columns(2)
        n11=c1.number_input("n(AnB)",0,10000,30,key='n11');n12=c2.number_input("n(AnB')",0,10000,20,key='n12');n21=c1.number_input("n(A'nB)",0,10000,10,key='n21');n22=c2.number_input("n(A'nB')",0,10000,40,key='n22')
        N=n11+n12+n21+n22
        if N>0: st.dataframe(pd.DataFrame([[n11,n12,n11+n12],[n21,n22,n21+n22],[n11+n21,n12+n22,N]],index=['A',"A'",'Tot'],columns=['B',"B'",'Tot']),use_container_width=True)
    elif sub=='Independensi':
        c1,c2,c3=st.columns(3);pa=c1.number_input("P(A)",0.0,1.0,0.6,0.01,key='ia');pb=c2.number_input("P(B)",0.0,1.0,0.5,0.01,key='ib');pab=c3.number_input("P(AnB)",0.0,1.0,0.30,0.01,key='ic')
        ex=pa*pb;st.markdown(f"P(A)P(B)=**{ex:.6f}** vs P(AnB)=**{pab}**")
        if abs(pab-ex)<1e-9: st.success("✅ Independen")
        else: st.error(f"❌ Tidak independen (diff={abs(pab-ex):.6f})")
    elif sub=='Hukum Total & Bayes':
        st.latex(r"P(A_k|B)=\frac{P(B|A_k)P(A_k)}{\sum P(B|A_i)P(A_i)}")
        nh=st.slider("Partisi:",2,6,3);pr=[];lk=[];cols=st.columns(nh)
        for i in range(nh):
            with cols[i]: pr.append(st.number_input(f"P(A{i+1})",0.0,1.0,round(1/nh,2),0.01,key=f'pr{i}'));lk.append(st.number_input(f"P(B|A{i+1})",0.0,1.0,round(0.3+0.2*i,2),0.01,key=f'lk{i}'))
        tp=sum(pr);pr=[p/tp for p in pr] if abs(tp-1)>0.01 else pr
        pb=sum(l*p for l,p in zip(lk,pr));po=[(l*p/pb) if pb>0 else 0 for l,p in zip(lk,pr)]
        st.markdown(f"### P(B)={fmt(pb,6)}")
        st.dataframe(pd.DataFrame({'H':[f'A{i+1}' for i in range(nh)],'Prior':[fmt(p,4) for p in pr],'Likelihood':[fmt(l,4) for l in lk],'Posterior':[fmt(p,6) for p in po]}),use_container_width=True,hide_index=True)
        fig=make_subplots(rows=1,cols=2,subplot_titles=("Prior vs Posterior","Likelihood"));lb=[f'A{i+1}' for i in range(nh)]
        fig.add_trace(go.Bar(x=lb,y=pr,name='Prior'),row=1,col=1);fig.add_trace(go.Bar(x=lb,y=po,name='Posterior'),row=1,col=1)
        fig.add_trace(go.Bar(x=lb,y=lk,name='Likelihood',marker_color='green'),row=1,col=2);fig.update_layout(height=380,barmode='group');st.plotly_chart(fig,use_container_width=True)
    elif sub=='Kalkulator Himpunan':
        S_i=st.text_input("S:","1,2,3,4,5,6");A_i=st.text_input("A:","1,2,3");B_i=st.text_input("B:","2,4,6")
        S={x.strip() for x in S_i.split(',') if x.strip()};A={x.strip() for x in A_i.split(',') if x.strip()}&S;B={x.strip() for x in B_i.split(',') if x.strip()}&S;ns=len(S)
        if ns>0: st.markdown(f"|Op|n|P|\n|---|---|---|\n|A|{len(A)}|{len(A)/ns:.4f}|\n|B|{len(B)}|{len(B)/ns:.4f}|\n|AuB|{len(A|B)}|{len(A|B)/ns:.4f}|\n|AnB|{len(A&B)}|{len(A&B)/ns:.4f}|\n|A'|{len(S-A)}|{len(S-A)/ns:.4f}|")

elif module=='kombinatorik':
    st.header("🔢 Kombinatorik");sub=st.selectbox("Topik:",['Permutasi & Kombinasi','Multinomial','Prinsip Counting','Stars and Bars','Inclusion-Exclusion'])
    if sub=='Permutasi & Kombinasi':
        c1,c2=st.columns(2);n_c=c1.number_input("n",0,1000,10);r_c=c2.number_input("r",0,1000,3)
        if r_c<=n_c: st.markdown(f"|Formula|Nilai|\n|---|---|\n|P(n,r)|**{safe_perm(n_c,r_c):,}**|\n|C(n,r)|**{safe_comb(n_c,r_c):,}**|\n|n^r|**{n_c**r_c:,}**|\n|C(n+r-1,r)|**{safe_comb(n_c+r_c-1,r_c):,}**|")
    elif sub=='Multinomial':
        st.latex(r"\frac{n!}{k_1!\cdots k_m!}");n_m=st.number_input("n:",1,200,10);ki=st.text_input("k1,k2,...:","3,3,4")
        ks=[int(x.strip()) for x in ki.split(',') if x.strip().isdigit()]
        if ks and sum(ks)==n_m: st.success(f"**{np.exp(gammaln(n_m+1)-sum(gammaln(k+1) for k in ks)):,.0f}**")
    elif sub=='Prinsip Counting':
        ns=st.slider("Tahap:",2,8,3);w=[];cols=st.columns(ns)
        for i in range(ns):
            with cols[i]: w.append(st.number_input(f"Tahap{i+1}",1,10000,3+i,key=f'c{i}'))
        t=1
        for x in w: t*=x
        st.success(f"Total={t:,}")
    elif sub=='Stars and Bars':
        n_s=st.number_input("n:",1,200,10);k_s=st.number_input("k:",1,50,3)
        st.success(f"C({n_s+k_s-1},{k_s-1})=**{safe_comb(n_s+k_s-1,k_s-1):,}**")
    elif sub=='Inclusion-Exclusion':
        ns=st.slider("Sets:",2,4,3);sz=[];cols=st.columns(ns)
        for i in range(ns):
            with cols[i]: sz.append(st.number_input(f"|A{i+1}|",0,10000,30+i*10,key=f'ie{i}'))
        prs=list(combinations(range(ns),2));pi=[];cols2=st.columns(max(len(prs),1))
        for idx,(i,j) in enumerate(prs):
            with cols2[idx]: pi.append(st.number_input(f"|A{i+1}nA{j+1}|",0,10000,5,key=f'ip{idx}'))
        tr=st.number_input("|A1nA2nA3|",0,10000,2) if ns>=3 else 0
        st.success(f"|Union|=**{sum(sz)-sum(pi)+tr}**")

elif module=='diskrit':
    st.header("📊 Distribusi Diskrit");dist=st.selectbox("Distribusi:",['Bernoulli','Binomial','Poisson','Geometric','Negative Binomial','Hypergeometric','Uniform Diskrit','Multinomial Sim'])
    if dist=='Bernoulli':
        p=st.slider("p:",0.01,0.99,0.5,0.01);q=1-p
        st.dataframe(dsummary(p,p*q,np.sqrt(p*q),(1-2*p)/np.sqrt(p*q),(1-6*p*q)/(p*q),mode=1 if p>0.5 else 0),use_container_width=True,hide_index=True)
        fig=go.Figure(data=[go.Bar(x=['0','1'],y=[q,p],marker_color=['steelblue','crimson'])]);fig.update_layout(height=280);st.plotly_chart(fig,use_container_width=True)
    elif dist=='Binomial':
        st.latex(r"P(X=k)=\binom{n}{k}p^k(1-p)^{n-k}");c1,c2=st.columns(2);nb=c1.number_input("n:",1,1000,20);pb=c2.slider("p:",0.01,0.99,0.5,0.01)
        rv=stats.binom(nb,pb);x=np.arange(0,nb+1);st.plotly_chart(plot_disc(x,rv.pmf(x),rv.cdf(x),f"Bin(n={nb},p={pb})"),use_container_width=True)
        st.dataframe(dsummary(rv.mean(),rv.var(),rv.std(),float(rv.stats('s')[0]),float(rv.stats('k')[0]),mode=int((nb+1)*pb),med=rv.median(),mgf="(1-p+pe^t)^n"),use_container_width=True,hide_index=True)
        calc=st.selectbox("Hitung:",['P(X=k)','P(X<=k)','P(X>=k)','P(a<=X<=b)'])
        if calc=='P(X=k)':k=st.number_input("k:",0,nb,nb//2);st.success(f"**{rv.pmf(k):.8f}**")
        elif calc=='P(X<=k)':k=st.number_input("k:",0,nb,nb//2);st.success(f"**{rv.cdf(k):.8f}**")
        elif calc=='P(X>=k)':k=st.number_input("k:",0,nb,nb//2);st.success(f"**{rv.sf(k-1):.8f}**")
        else:c1,c2=st.columns(2);a=c1.number_input("a:",0,nb,0);b=c2.number_input("b:",0,nb,nb);st.success(f"**{rv.cdf(b)-rv.cdf(a-1):.8f}**")
    elif dist=='Poisson':
        st.latex(r"P(X=k)=\frac{\lambda^k e^{-\lambda}}{k!}");lam=st.slider("lam:",0.1,50.0,5.0,0.1);rv=stats.poisson(lam)
        xm=int(rv.ppf(0.9999))+1;x=np.arange(0,xm);st.plotly_chart(plot_disc(x,rv.pmf(x),rv.cdf(x),f"Poi(lam={lam})"),use_container_width=True)
        st.dataframe(dsummary(lam,lam,np.sqrt(lam),1/np.sqrt(lam),1/lam,mode=int(lam)),use_container_width=True,hide_index=True)
        k=st.number_input("k:",0,10000,int(lam));st.markdown(f"P(X={k})=**{rv.pmf(k):.8f}** | P(X<={k})=**{rv.cdf(k):.8f}**")
    elif dist=='Geometric':
        pg=st.slider("p:",0.01,0.99,0.3,0.01);rv=stats.geom(pg);qg=1-pg;xm=int(rv.ppf(0.999))+1;x=np.arange(1,xm+1)
        st.plotly_chart(plot_disc(x,rv.pmf(x),rv.cdf(x),f"Geom(p={pg})"),use_container_width=True)
        st.dataframe(dsummary(1/pg,qg/pg**2,np.sqrt(qg)/pg,(2-pg)/np.sqrt(qg),6+pg**2/qg,mode=1,med=rv.median()),use_container_width=True,hide_index=True)
    elif dist=='Negative Binomial':
        c1,c2=st.columns(2);rn=c1.number_input("r:",1,100,3);pn=c2.slider("p:",0.01,0.99,0.4,0.01,key='nbp');rv=stats.nbinom(rn,pn)
        xs=np.arange(0,int(rv.ppf(0.999))+1);st.plotly_chart(plot_disc(xs+rn,rv.pmf(xs),rv.cdf(xs),f"NB(r={rn},p={pn})"),use_container_width=True)
        mn=rn/pn;vn=rn*(1-pn)/pn**2;st.dataframe(dsummary(mn,vn,np.sqrt(vn),(2-pn)/np.sqrt(rn*(1-pn)),6/rn+pn**2/(rn*(1-pn))),use_container_width=True,hide_index=True)
    elif dist=='Hypergeometric':
        c1,c2,c3=st.columns(3);Nh=c1.number_input("N:",1,100000,50);Kh=c2.number_input("K:",0,Nh,20);nh=c3.number_input("n:",1,Nh,10)
        rv=stats.hypergeom(Nh,Kh,nh);lo=max(0,nh-(Nh-Kh));hi=min(nh,Kh);x=np.arange(lo,hi+1)
        st.plotly_chart(plot_disc(x,rv.pmf(x),rv.cdf(x),f"HG(N={Nh},K={Kh},n={nh})"),use_container_width=True)
        st.dataframe(dsummary(rv.mean(),rv.var(),rv.std(),float(rv.stats('s')[0]),float(rv.stats('k')[0]),med=rv.median()),use_container_width=True,hide_index=True)
    elif dist=='Uniform Diskrit':
        c1,c2=st.columns(2);au=c1.number_input("a:",value=1);bu=c2.number_input("b:",value=6);nu=bu-au+1;rv=stats.randint(au,bu+1);x=np.arange(au,bu+1)
        st.plotly_chart(plot_disc(x,rv.pmf(x),rv.cdf(x),f"Unif({au},{bu})"),use_container_width=True)
        mu=(au+bu)/2;vu=(nu**2-1)/12;st.dataframe(dsummary(mu,vu,np.sqrt(vu),0,-(6*(nu**2+1))/(5*(nu**2-1)),mode="All",med=mu),use_container_width=True,hide_index=True)
    elif dist=='Multinomial Sim':
        nm=st.number_input("n:",1,10000,100);km=st.slider("k:",2,8,3);pi=[];cols=st.columns(km)
        for i in range(km):
            with cols[i]: pi.append(st.number_input(f"p{i+1}",0.01,1.0,round(1/km,2),0.01,key=f'mp{i}'))
        sp=sum(pi);pr=[p/sp for p in pi];ns=st.slider("Sim:",100,50000,5000);samp=np.random.multinomial(nm,pr,size=ns)
        fig=make_subplots(rows=1,cols=km,subplot_titles=[f'X{i+1}' for i in range(km)])
        for i in range(km): fig.add_trace(go.Histogram(x=samp[:,i],nbinsx=30,marker_color='steelblue',opacity=0.7),row=1,col=i+1)
        fig.update_layout(height=320,showlegend=False);st.plotly_chart(fig,use_container_width=True)

elif module=='kontinu':
    st.header("📈 Distribusi Kontinu");dist=st.selectbox("Distribusi:",['Normal','Exponential','Uniform','Gamma','Beta','Chi-Square','t-Student','F','Weibull','Log-Normal','Cauchy','Pareto'])
    if dist=='Normal':
        st.latex(r"f(x)=\frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}")
        c1,c2=st.columns(2);mu=c1.number_input("mu:",-100.0,100.0,0.0,0.1);sig=c2.number_input("sigma:",0.01,50.0,1.0,0.1)
        rv=stats.norm(mu,sig);x=np.linspace(rv.ppf(0.0001),rv.ppf(0.9999),500)
        st.plotly_chart(plot_cont(x,rv.pdf(x),rv.cdf(x),f"N(mu={mu},sig={sig})"),use_container_width=True)
        st.dataframe(dsummary(mu,sig**2,sig,0,0,mode=mu,med=mu,mgf="exp(mut+sig^2t^2/2)"),use_container_width=True,hide_index=True)
        calc=st.selectbox("Hitung:",['P(X<=x)','P(X>=x)','P(a<=X<=b)','Quantile','Z-score'],key='nc')
        if calc=='P(X<=x)':
            xv=st.number_input("x:",value=mu);z=(xv-mu)/sig;prob=rv.cdf(xv);st.success(f"z={z:.4f}, P(X<={xv})=**{prob:.8f}**")
            sx=x[x<=xv];st.plotly_chart(plot_cont(x,rv.pdf(x),rv.cdf(x),f"P(X<={xv})={prob:.6f}",sx=sx,sp=rv.pdf(sx),sl=f'P={prob:.4f}'),use_container_width=True)
        elif calc=='P(X>=x)':xv=st.number_input("x:",value=mu);st.success(f"P(X>={xv})=**{rv.sf(xv):.8f}**")
        elif calc=='P(a<=X<=b)':
            c1,c2=st.columns(2);a=c1.number_input("a:",value=mu-sig);b=c2.number_input("b:",value=mu+sig);prob=rv.cdf(b)-rv.cdf(a)
            st.success(f"P({a}<=X<={b})=**{prob:.8f}**");m=(x>=a)&(x<=b);st.plotly_chart(plot_cont(x,rv.pdf(x),rv.cdf(x),f"P={prob:.6f}",sx=x[m],sp=rv.pdf(x)[m]),use_container_width=True)
        elif calc=='Quantile':pq=st.number_input("P:",0.001,0.999,0.975,0.001);st.success(f"x=**{rv.ppf(pq):.6f}**")
        else:xv=st.number_input("x:",value=mu+1.5);st.success(f"z=**{(xv-mu)/sig:.6f}**")
        with st.expander("Aturan Empiris"):
            for k,pct in [(1,68.27),(2,95.45),(3,99.73)]: st.markdown(f"mu+-{k}sig=[{mu-k*sig:.2f},{mu+k*sig:.2f}] -> **{pct}%**")
    elif dist=='Exponential':
        st.latex(r"f(x)=\lambda e^{-\lambda x}");lam=st.slider("lam:",0.01,10.0,1.0,0.01);rv=stats.expon(scale=1/lam)
        x=np.linspace(0,rv.ppf(0.999),500);st.plotly_chart(plot_cont(x,rv.pdf(x),rv.cdf(x),f"Exp(lam={lam})"),use_container_width=True)
        st.dataframe(dsummary(1/lam,1/lam**2,1/lam,2,6,mode=0,med=np.log(2)/lam,mgf="lam/(lam-t)"),use_container_width=True,hide_index=True)
        t=st.number_input("t:",0.0,1000.0,1.0);st.markdown(f"P(X<={t})=**{rv.cdf(t):.8f}** | P(X>{t})=**{rv.sf(t):.8f}**")
    elif dist=='Uniform':
        c1,c2=st.columns(2);au=c1.number_input("a:",-100.0,100.0,0.0,0.1);bu=c2.number_input("b:",au+0.01,200.0,1.0,0.1)
        rv=stats.uniform(au,bu-au);x=np.linspace(au-0.5,bu+0.5,500)
        st.plotly_chart(plot_cont(x,rv.pdf(x),rv.cdf(x),f"Unif({au},{bu})"),use_container_width=True)
        st.dataframe(dsummary((au+bu)/2,(bu-au)**2/12,np.sqrt((bu-au)**2/12),0,-6/5,mode="All in [a,b]",med=(au+bu)/2),use_container_width=True,hide_index=True)
    elif dist=='Gamma':
        c1,c2=st.columns(2);al=c1.number_input("alpha:",0.1,50.0,2.0,0.1);bt=c2.number_input("beta:",0.1,50.0,1.0,0.1)
        rv=stats.gamma(al,scale=1/bt);x=np.linspace(0,rv.ppf(0.999),500)
        st.plotly_chart(plot_cont(x,rv.pdf(x),rv.cdf(x),f"Gamma(a={al},b={bt})"),use_container_width=True)
        st.dataframe(dsummary(al/bt,al/bt**2,np.sqrt(al)/bt,2/np.sqrt(al),6/al,mode=(al-1)/bt if al>=1 else 0),use_container_width=True,hide_index=True)
    elif dist=='Beta':
        c1,c2=st.columns(2);ab=c1.number_input("alpha:",0.1,50.0,2.0,0.1,key='ba');bb=c2.number_input("beta:",0.1,50.0,5.0,0.1,key='bb')
        rv=stats.beta(ab,bb);x=np.linspace(0.001,0.999,500)
        st.plotly_chart(plot_cont(x,rv.pdf(x),rv.cdf(x),f"Beta(a={ab},b={bb})"),use_container_width=True)
        st.dataframe(dsummary(rv.mean(),rv.var(),rv.std(),float(rv.stats('s')[0]),float(rv.stats('k')[0]),mode=(ab-1)/(ab+bb-2) if ab>1 and bb>1 else None,med=rv.median()),use_container_width=True,hide_index=True)
    elif dist=='Chi-Square':
        k=st.slider("df:",1,50,5);rv=stats.chi2(k);x=np.linspace(0,rv.ppf(0.999),500)
        st.plotly_chart(plot_cont(x,rv.pdf(x),rv.cdf(x),f"Chi2(k={k})"),use_container_width=True)
        st.dataframe(dsummary(k,2*k,np.sqrt(2*k),np.sqrt(8/k),12/k,mode=max(k-2,0),med=rv.median()),use_container_width=True,hide_index=True)
        xv=st.number_input("x:",0.0,1000.0,float(k));st.markdown(f"P(X<={xv})=**{rv.cdf(xv):.8f}** | P(X>{xv})=**{rv.sf(xv):.8f}**")
    elif dist=='t-Student':
        nu=st.slider("df:",1,100,10);rv=stats.t(nu);x=np.linspace(rv.ppf(0.0001),rv.ppf(0.9999),500)
        fig=plot_cont(x,rv.pdf(x),rv.cdf(x),f"t(df={nu})");fig.add_trace(go.Scatter(x=x,y=stats.norm.pdf(x),mode='lines',line=dict(color='gray',dash='dash',width=1),name='N(0,1)'))
        st.plotly_chart(fig,use_container_width=True)
        vt=nu/(nu-2) if nu>2 else float('inf');kt=6/(nu-4) if nu>4 else float('inf')
        st.dataframe(dsummary(0,vt,np.sqrt(vt) if nu>2 else float('inf'),0,kt,mode=0,med=0),use_container_width=True,hide_index=True)
        calc=st.selectbox("Hitung:",['P(T<=t)','Critical value','CI'],key='tc')
        if calc=='P(T<=t)':tv=st.number_input("t:",value=1.96,step=0.01);st.success(f"P(T<={tv})=**{rv.cdf(tv):.8f}** | two-tail=**{2*rv.sf(abs(tv)):.8f}**")
        elif calc=='Critical value':
            al=st.number_input("alpha:",0.001,0.5,0.05,0.001);tw=st.checkbox("Two-tailed?",True)
            cv=rv.ppf(1-al/2) if tw else rv.ppf(1-al);st.success(f"t_crit=**{cv:.6f}**")
        else:
            xb=st.number_input("xbar:",value=0.0);se=st.number_input("SE:",0.01,100.0,1.0);al=st.number_input("alpha:",0.001,0.5,0.05,0.001)
            cv=rv.ppf(1-al/2);st.success(f"{(1-al)*100:.1f}% CI: [{xb-cv*se:.4f}, {xb+cv*se:.4f}]")
    elif dist=='F':
        c1,c2=st.columns(2);d1=c1.slider("d1:",1,100,5);d2=c2.slider("d2:",1,100,20);rv=stats.f(d1,d2)
        x=np.linspace(0,rv.ppf(0.999),500);st.plotly_chart(plot_cont(x,rv.pdf(x),rv.cdf(x),f"F({d1},{d2})"),use_container_width=True)
        mf=d2/(d2-2) if d2>2 else float('inf');vf=2*d2**2*(d1+d2-2)/(d1*(d2-2)**2*(d2-4)) if d2>4 else float('inf')
        st.dataframe(dsummary(mf,vf,np.sqrt(vf) if d2>4 else float('inf'),float(rv.stats('s')[0]) if d2>6 else float('inf'),float(rv.stats('k')[0]) if d2>8 else float('inf'),mode=(d1-2)/d1*d2/(d2+2) if d1>2 else 0),use_container_width=True,hide_index=True)
    elif dist=='Weibull':
        c1,c2=st.columns(2);kw=c1.number_input("k(shape):",0.1,20.0,1.5,0.1);lw=c2.number_input("lam(scale):",0.1,100.0,1.0,0.1)
        rv=stats.weibull_min(kw,scale=lw);x=np.linspace(0,rv.ppf(0.999),500)
        st.plotly_chart(plot_cont(x,rv.pdf(x),rv.cdf(x),f"Weibull(k={kw},lam={lw})"),use_container_width=True)
        st.dataframe(dsummary(rv.mean(),rv.var(),rv.std(),float(rv.stats('s')[0]),float(rv.stats('k')[0]),med=rv.median()),use_container_width=True,hide_index=True)
    elif dist=='Log-Normal':
        c1,c2=st.columns(2);ml=c1.number_input("mu:",-10.0,10.0,0.0,0.1);sl=c2.number_input("sigma:",0.01,5.0,1.0,0.1)
        rv=stats.lognorm(sl,scale=np.exp(ml));x=np.linspace(0.001,rv.ppf(0.999),500)
        st.plotly_chart(plot_cont(x,rv.pdf(x),rv.cdf(x),f"LogN(mu={ml},sig={sl})"),use_container_width=True)
        st.dataframe(dsummary(rv.mean(),rv.var(),rv.std(),float(rv.stats('s')[0]),float(rv.stats('k')[0]),mode=np.exp(ml-sl**2),med=np.exp(ml)),use_container_width=True,hide_index=True)
    elif dist=='Cauchy':
        c1,c2=st.columns(2);x0=c1.number_input("x0:",-100.0,100.0,0.0,0.1);gm=c2.number_input("gamma:",0.01,50.0,1.0,0.1)
        rv=stats.cauchy(x0,gm);x=np.linspace(rv.ppf(0.005),rv.ppf(0.995),500)
        st.plotly_chart(plot_cont(x,rv.pdf(x),rv.cdf(x),f"Cauchy(x0={x0},g={gm})"),use_container_width=True)
        st.warning("Mean, Variance, MGF undefined");st.markdown(f"Median=**{x0}** | Mode=**{x0}** | IQR=**{2*gm}**")
    elif dist=='Pareto':
        c1,c2=st.columns(2);ap=c1.number_input("alpha:",0.1,20.0,2.5,0.1);xm=c2.number_input("xm:",0.1,100.0,1.0,0.1)
        rv=stats.pareto(ap,scale=xm);x=np.linspace(xm,rv.ppf(0.99),500)
        st.plotly_chart(plot_cont(x,rv.pdf(x),rv.cdf(x),f"Pareto(a={ap},xm={xm})"),use_container_width=True)
        mp=ap*xm/(ap-1) if ap>1 else float('inf');vp=xm**2*ap/((ap-1)**2*(ap-2)) if ap>2 else float('inf')
        st.dataframe(dsummary(mp,vp,np.sqrt(vp) if ap>2 else float('inf'),float(rv.stats('s')[0]) if ap>3 else float('inf'),float(rv.stats('k')[0]) if ap>4 else float('inf'),mode=xm,med=xm*2**(1/ap)),use_container_width=True,hide_index=True)

elif module=='joint':
    st.header("🔗 Distribusi Gabungan");sub=st.selectbox("Topik:",['Joint Diskrit','Bivariate Normal','Covariance & Correlation','Transformasi Variabel'])
    if sub=='Joint Diskrit':
        st.subheader("Tabel Joint P(X,Y)")
        nr=st.number_input("Nilai X:",2,8,3,key='jr');nc=st.number_input("Nilai Y:",2,8,3,key='jc')
        xv=[str(i) for i in range(nr)];yv=[str(j) for j in range(nc)]
        jt=np.zeros((nr,nc))
        for i in range(nr):
            cols=st.columns(nc)
            for j in range(nc):
                with cols[j]: jt[i,j]=st.number_input(f"P(X={i},Y={j})",0.0,1.0,round(1/(nr*nc),3),0.001,key=f'j{i}{j}')
        mx=jt.sum(1);my=jt.sum(0);tot=jt.sum()
        if abs(tot-1)>0.01: st.warning(f"Sum={tot:.4f}")
        jdf=pd.DataFrame(jt,index=[f'X={v}' for v in xv],columns=[f'Y={v}' for v in yv]);jdf['P(X)']=mx
        jdf.loc['P(Y)']=list(my)+[tot];st.dataframe(jdf.round(4),use_container_width=True)
        xn=np.arange(nr,dtype=float);yn=np.arange(nc,dtype=float)
        ex=np.sum(xn*mx);ey=np.sum(yn*my);exy=np.sum(jt*np.outer(xn,yn));cov=exy-ex*ey
        vx=np.sum(xn**2*mx)-ex**2;vy=np.sum(yn**2*my)-ey**2;corr=cov/np.sqrt(vx*vy) if vx>0 and vy>0 else 0
        st.markdown(f"E[X]={ex:.4f} | E[Y]={ey:.4f} | Cov={cov:.4f} | rho={corr:.4f}")
        indep=np.allclose(jt,np.outer(mx,my),atol=0.001);st.markdown(f"Independen? {'✅' if indep else '❌'}")
    elif sub=='Bivariate Normal':
        c1,c2,c3=st.columns(3);m1=c1.number_input("mu1:",value=0.0);m2=c2.number_input("mu2:",value=0.0)
        s1=c1.number_input("sig1:",0.1,10.0,1.0);s2=c2.number_input("sig2:",0.1,10.0,1.0);rho=c3.slider("rho:",-0.99,0.99,0.5,0.01)
        x=np.linspace(m1-4*s1,m1+4*s1,80);y=np.linspace(m2-4*s2,m2+4*s2,80);X,Y=np.meshgrid(x,y)
        cv=[[s1**2,rho*s1*s2],[rho*s1*s2,s2**2]];rv=stats.multivariate_normal([m1,m2],cv);Z=rv.pdf(np.dstack((X,Y)))
        fig=make_subplots(rows=1,cols=2,subplot_titles=("Contour","3D"),specs=[[{"type":"xy"},{"type":"surface"}]])
        fig.add_trace(go.Contour(z=Z,x=x,y=y,colorscale='Viridis',ncontours=20),row=1,col=1)
        fig.add_trace(go.Surface(z=Z,x=x,y=y,colorscale='Viridis',showscale=False),row=1,col=2)
        fig.update_layout(height=480);st.plotly_chart(fig,use_container_width=True)
        nsim=st.slider("Sim:",500,20000,5000);samp=np.random.multivariate_normal([m1,m2],cv,nsim)
        fig=px.scatter(x=samp[:,0],y=samp[:,1],opacity=0.3,labels={'x':'X1','y':'X2'});fig.update_layout(height=420);st.plotly_chart(fig,use_container_width=True)
    elif sub=='Covariance & Correlation':
        st.latex(r"\rho=\frac{\text{Cov}(X,Y)}{\sigma_X\sigma_Y}")
        c1,c2=st.columns(2);exy=c1.number_input("E[XY]:",value=12.0);ex=c1.number_input("E[X]:",value=3.0);ey=c2.number_input("E[Y]:",value=4.0)
        sx=c1.number_input("sigX:",0.01,100.0,2.0);sy=c2.number_input("sigY:",0.01,100.0,3.0)
        cov=exy-ex*ey;corr=cov/(sx*sy);st.success(f"Cov={cov:.4f} | rho={corr:.4f}")
        a=st.number_input("a:",value=2.0);b=st.number_input("b:",value=3.0)
        st.markdown(f"Var(aX+bY)={a**2*sx**2+b**2*sy**2+2*a*b*cov:.4f}")
    elif sub=='Transformasi Variabel':
        st.markdown("### Y=aX+b")
        c1,c2=st.columns(2);mu=c1.number_input("E[X]:",value=5.0);sig=c2.number_input("sigX:",0.01,100.0,2.0)
        a=c1.number_input("a:",value=3.0);b=c2.number_input("b:",value=-1.0)
        st.success(f"E[Y]={a*mu+b:.4f} | Var(Y)={a**2*sig**2:.4f} | sigY={abs(a)*sig:.4f}")

elif module=='clt':
    st.header("📉 Central Limit Theorem")
    st.latex(r"\bar{X}_n\xrightarrow{d}N(\mu,\sigma^2/n)")
    src=st.selectbox("Sumber:",['Uniform','Exponential','Bernoulli','Poisson','Chi-Square'])
    ns=st.slider("n (sample size):",1,200,30);nsim=st.slider("Simulasi:",500,50000,10000)
    if src=='Uniform': pop=np.random.uniform(0,10,(nsim,ns));mu=5;sig=10/np.sqrt(12)
    elif src=='Exponential':lc=st.slider("lam:",0.1,5.0,1.0,0.1);pop=np.random.exponential(1/lc,(nsim,ns));mu=1/lc;sig=1/lc
    elif src=='Bernoulli':pc=st.slider("p:",0.05,0.95,0.3,0.05);pop=np.random.binomial(1,pc,(nsim,ns));mu=pc;sig=np.sqrt(pc*(1-pc))
    elif src=='Poisson':lc=st.slider("lam:",0.5,20.0,3.0,0.5);pop=np.random.poisson(lc,(nsim,ns));mu=lc;sig=np.sqrt(lc)
    else:kc=st.slider("df:",1,20,3);pop=np.random.chisquare(kc,(nsim,ns));mu=kc;sig=np.sqrt(2*kc)
    xbars=pop.mean(1);se=sig/np.sqrt(ns)
    fig=make_subplots(rows=1,cols=2,subplot_titles=(f"Populasi({src})",f"Dist Xbar(n={ns})"))
    fig.add_trace(go.Histogram(x=pop[0],nbinsx=40,marker_color='steelblue',opacity=0.7,histnorm='probability density'),row=1,col=1)
    fig.add_trace(go.Histogram(x=xbars,nbinsx=60,marker_color='crimson',opacity=0.7,histnorm='probability density'),row=1,col=2)
    xn=np.linspace(xbars.min(),xbars.max(),200);fig.add_trace(go.Scatter(x=xn,y=stats.norm.pdf(xn,mu,se),mode='lines',line=dict(color='black',dash='dash',width=2),name='Normal'),row=1,col=2)
    fig.update_layout(height=400);st.plotly_chart(fig,use_container_width=True)
    c1,c2,c3=st.columns(3);c1.metric("mu pop",f"{mu:.4f}");c2.metric("Mean Xbar",f"{xbars.mean():.4f}");c3.metric("sig/sqrt(n)",f"{se:.4f}")
    sw,swp=stats.shapiro(xbars[:5000]);st.markdown(f"Shapiro-Wilk: W={sw:.4f}, p={swp:.6f} {'✅' if swp>0.05 else '⚠️'}")

elif module=='markov':
    st.header("🔄 Rantai Markov");sub=st.selectbox("Topik:",['Matriks Transisi','Steady-State','Simulasi','Absorbing Chains','n-Step'])
    if sub=='Matriks Transisi':
        ns=st.slider("States:",2,6,3);sn=[st.text_input(f"State{i+1}:",f"S{i+1}",key=f's{i}') for i in range(ns)]
        P=np.zeros((ns,ns))
        for i in range(ns):
            cols=st.columns(ns)
            for j in range(ns):
                with cols[j]: P[i,j]=st.number_input(f"{sn[i]}->{sn[j]}",0.0,1.0,round(1/ns,2),0.01,key=f't{i}{j}')
        rs=P.sum(1);valid=np.allclose(rs,1,atol=0.01)
        if not valid:
            st.error(f"Rows not 1: {rs.round(3)}")
            if st.button("Normalize"): P=P/rs[:,None]
        st.dataframe(pd.DataFrame(P.round(4),index=sn,columns=sn),use_container_width=True)
        fig=go.Figure(data=go.Heatmap(z=P,x=sn,y=sn,colorscale='Blues',text=np.round(P,3),texttemplate='%{text}'));fig.update_layout(height=380);st.plotly_chart(fig,use_container_width=True)
    elif sub=='Steady-State':
        st.latex(r"\pi P=\pi,\sum\pi_i=1")
        pre=st.selectbox("Preset:",['Custom','Weather','Market'])
        if pre=='Weather': ns=2;sn=['Sunny','Rainy'];P=np.array([[0.8,0.2],[0.4,0.6]])
        elif pre=='Market': ns=3;sn=['Bull','Bear','Stag'];P=np.array([[0.6,0.2,0.2],[0.3,0.4,0.3],[0.2,0.3,0.5]])
        else:
            ns=st.slider("States:",2,6,3,key='ssn');sn=[f"S{i+1}" for i in range(ns)];P=np.full((ns,ns),1/ns)
            for i in range(ns):
                cols=st.columns(ns)
                for j in range(ns):
                    with cols[j]: P[i,j]=st.number_input(f"P{sn[i]}{sn[j]}",0.0,1.0,round(1/ns,2),0.01,key=f'ss{i}{j}')
            rs=P.sum(1);P=P/rs[:,None]
        A=(P.T-np.eye(len(sn)));A[-1]=1;b=np.zeros(len(sn));b[-1]=1
        try: pi=np.linalg.solve(A,b)
        except: pi=np.ones(len(sn))/len(sn)
        st.dataframe(pd.DataFrame({'State':sn,'pi':pi.round(6)}),use_container_width=True,hide_index=True)
        fig=go.Figure(data=[go.Bar(x=sn,y=pi,marker_color='steelblue')]);fig.update_layout(height=320);st.plotly_chart(fig,use_container_width=True)
        nstep=st.slider("Steps:",1,100,20);Pk=np.eye(len(sn));hist=[Pk[0].copy()]
        for _ in range(nstep): Pk=Pk@P;hist.append(Pk[0].copy())
        ha=np.array(hist);fig=go.Figure()
        for j in range(len(sn)): fig.add_trace(go.Scatter(y=ha[:,j],mode='lines+markers',name=sn[j],marker=dict(size=3)));fig.add_hline(y=pi[j],line_dash="dot",opacity=0.4)
        fig.update_layout(height=380,xaxis_title="n",yaxis_title="P");st.plotly_chart(fig,use_container_width=True)
    elif sub=='Simulasi':
        ns=st.slider("States:",2,5,3,key='simn');sn=[f"S{i+1}" for i in range(ns)]
        pre=st.selectbox("Preset:",['Custom','Gambler 3'],key='simp')
        if pre=='Gambler 3': ns=3;sn=['$0','$1','$2'];P=np.array([[1,0,0],[0.4,0,0.6],[0,0,1]])
        else:
            P=np.full((ns,ns),1/ns)
            for i in range(ns):
                cols=st.columns(ns)
                for j in range(ns):
                    with cols[j]: P[i,j]=st.number_input(f"{sn[i]}->{sn[j]}",0.0,1.0,round(1/ns,2),0.01,key=f'sm{i}{j}')
            rs=P.sum(1);P=P/rs[:,None]
        init=st.selectbox("Start:",sn);nst=st.slider("Steps:",10,1000,100);nw=st.slider("Walks:",1,50,5)
        idx=sn.index(init);fig=go.Figure()
        for w in range(nw):
            ch=[idx]
            for _ in range(nst): ch.append(np.random.choice(len(sn),p=P[ch[-1]]))
            fig.add_trace(go.Scatter(y=ch,mode='lines',opacity=0.6,line=dict(width=1),name=f'W{w+1}'))
        fig.update_layout(height=400,yaxis=dict(tickvals=list(range(len(sn))),ticktext=sn));st.plotly_chart(fig,use_container_width=True)
    elif sub=='Absorbing Chains':
        st.info("Gambler's Ruin");N=st.slider("$N:",2,20,5);pw=st.slider("P(win):",0.01,0.99,0.5,0.01)
        ns=N+1;sn=[f'${i}' for i in range(ns)];P=np.zeros((ns,ns));P[0,0]=1;P[-1,-1]=1
        for i in range(1,ns-1): P[i,i+1]=pw;P[i,i-1]=1-pw
        tr=list(range(1,ns-1));ab=[0,ns-1];nt=len(tr)
        if nt>0:
            Q=P[np.ix_(tr,tr)];Nf=np.linalg.inv(np.eye(nt)-Q);R=P[np.ix_(tr,ab)];B=Nf@R;ta=Nf.sum(1)
            adf=pd.DataFrame(B,index=[sn[i] for i in tr],columns=[sn[i] for i in ab]).round(4);adf['E[steps]']=ta.round(1)
            st.dataframe(adf,use_container_width=True)
            s=st.slider("Start $:",1,N-1,N//2);ix=s-1;st.success(f"P(ruin)={B[ix,0]:.4f} | P(win)={B[ix,1]:.4f} | E[steps]={ta[ix]:.1f}")
    elif sub=='n-Step':
        ns=st.slider("States:",2,5,3,key='nsn');sn=[f"S{i+1}" for i in range(ns)];P=np.full((ns,ns),1/ns)
        for i in range(ns):
            cols=st.columns(ns)
            for j in range(ns):
                with cols[j]: P[i,j]=st.number_input(f"P{sn[i]}{sn[j]}",0.0,1.0,round(1/ns,2),0.01,key=f'ns{i}{j}')
        rs=P.sum(1);P=P/rs[:,None];nst=st.number_input("n:",1,500,5);Pn=np.linalg.matrix_power(P,nst)
        st.dataframe(pd.DataFrame(Pn.round(6),index=sn,columns=sn),use_container_width=True)
        fig=go.Figure(data=go.Heatmap(z=Pn,x=sn,y=sn,colorscale='Blues',text=np.round(Pn,4),texttemplate='%{text}'));fig.update_layout(height=380);st.plotly_chart(fig,use_container_width=True)

elif module=='montecarlo':
    st.header("🎰 Monte Carlo");sub=st.selectbox("Topik:",['Estimasi pi','Integrasi MC','Law of Large Numbers','Random Walk','Birthday Problem','Monty Hall'])
    if sub=='Estimasi pi':
        st.latex(r"\pi\approx 4\times\frac{\text{inside}}{\text{total}}");np.random.seed(None)
        n=st.slider("Titik:",100,200000,10000);x=np.random.uniform(-1,1,n);y=np.random.uniform(-1,1,n);ins=x**2+y**2<=1
        pe=4*ins.sum()/n;c1,c2,c3=st.columns(3);c1.metric("pi_est",f"{pe:.6f}");c2.metric("pi",f"{np.pi:.6f}");c3.metric("Err",f"{abs(pe-np.pi):.6f}")
        ns=min(n,20000);fig=go.Figure()
        fig.add_trace(go.Scatter(x=x[:ns][ins[:ns]],y=y[:ns][ins[:ns]],mode='markers',marker=dict(size=2,color='blue',opacity=0.3),name='In'))
        fig.add_trace(go.Scatter(x=x[:ns][~ins[:ns]],y=y[:ns][~ins[:ns]],mode='markers',marker=dict(size=2,color='red',opacity=0.3),name='Out'))
        th=np.linspace(0,2*np.pi,100);fig.add_trace(go.Scatter(x=np.cos(th),y=np.sin(th),mode='lines',line=dict(color='black',width=2)))
        fig.update_layout(height=480,yaxis=dict(scaleanchor='x'));st.plotly_chart(fig,use_container_width=True)
        cp=np.unique(np.geomspace(10,n,100).astype(int));pr=[4*ins[:c].sum()/c for c in cp]
        fig=go.Figure();fig.add_trace(go.Scatter(x=cp,y=pr,mode='lines'));fig.add_hline(y=np.pi,line_dash="dash",line_color="red")
        fig.update_layout(height=320,xaxis_type="log",title="Convergence");st.plotly_chart(fig,use_container_width=True)
    elif sub=='Integrasi MC':
        st.latex(r"\int_a^b g(x)dx\approx(b-a)\frac{1}{N}\sum g(x_i)")
        fc=st.selectbox("f:",['sin(x)','x^2','exp(-x^2)','sqrt(x)','1/(1+x^2)'])
        c1,c2=st.columns(2);a=c1.number_input("a:",value=0.0);b=c2.number_input("b:",value=float(np.pi) if fc=='sin(x)' else 1.0);nm=st.slider("N:",100,500000,50000)
        fs={'sin(x)':np.sin,'x^2':lambda x:x**2,'exp(-x^2)':lambda x:np.exp(-x**2),'sqrt(x)':np.sqrt,'1/(1+x^2)':lambda x:1/(1+x**2)}
        g=fs[fc];xs=np.random.uniform(a,b,nm);gx=g(xs);est=(b-a)*gx.mean();se=(b-a)*gx.std()/np.sqrt(nm);ex,_=quad(g,a,b)
        c1,c2,c3=st.columns(3);c1.metric("MC",f"{est:.6f}");c2.metric("Exact",f"{ex:.6f}");c3.metric("Err",f"{abs(est-ex):.6f}")
        xp=np.linspace(a,b,300);fig=go.Figure()
        fig.add_trace(go.Scatter(x=xp,y=g(xp),fill='tozeroy',fillcolor='rgba(70,130,180,0.2)',line=dict(color='steelblue',width=2)))
        fig.update_layout(height=380);st.plotly_chart(fig,use_container_width=True)
    elif sub=='Law of Large Numbers':
        st.latex(r"\bar{X}_n\xrightarrow{P}\mu");dl=st.selectbox("Dist:",['Coin','Die','Exp(1)']);nm=st.slider("Max n:",100,100000,10000)
        if dl=='Coin': d=np.random.binomial(1,0.5,nm);mu=0.5
        elif dl=='Die': d=np.random.randint(1,7,nm);mu=3.5
        else: d=np.random.exponential(1,nm);mu=1.0
        ra=np.cumsum(d)/np.arange(1,nm+1);fig=go.Figure()
        fig.add_trace(go.Scatter(y=ra,mode='lines',line=dict(width=1)));fig.add_hline(y=mu,line_dash="dash",line_color="red")
        fig.update_layout(height=380,title=f"LLN: Xbar->mu={mu}");st.plotly_chart(fig,use_container_width=True)
    elif sub=='Random Walk':
        c1,c2=st.columns(2);dim=c1.selectbox("Dim:",['1D','2D']);nst=c2.slider("Steps:",10,10000,500);nw=st.slider("Walks:",1,20,5)
        if dim=='1D':
            fig=go.Figure()
            for w in range(nw): s=np.random.choice([-1,1],nst);wk=np.insert(np.cumsum(s),0,0);fig.add_trace(go.Scatter(y=wk,mode='lines',opacity=0.7,line=dict(width=1)))
            fig.add_hline(y=0,line_dash="dot");fig.update_layout(height=420);st.plotly_chart(fig,use_container_width=True)
        else:
            fig=go.Figure()
            for w in range(nw):
                dx=np.random.choice([-1,1],nst);dy=np.random.choice([-1,1],nst)
                xw=np.insert(np.cumsum(dx),0,0);yw=np.insert(np.cumsum(dy),0,0)
                fig.add_trace(go.Scatter(x=xw,y=yw,mode='lines',opacity=0.6,line=dict(width=1)))
            fig.add_trace(go.Scatter(x=[0],y=[0],mode='markers',marker=dict(size=10,color='red',symbol='circle'),name='Start'))
            fig.update_layout(height=500,yaxis=dict(scaleanchor='x'));st.plotly_chart(fig,use_container_width=True)
    elif sub=='Birthday Problem':
        mx=st.slider("Max orang:",10,100,60);nsim=st.slider("Sim:",1000,100000,10000)
        ex=[];sp=[]
        for np_ in range(1,mx+1):
            pe=1.0
            for i in range(np_): pe*=(365-i)/365
            ex.append(1-pe);sp.append(sum(1 for _ in range(nsim) if len(set(np.random.randint(1,366,np_)))<np_)/nsim)
        fig=go.Figure();fig.add_trace(go.Scatter(y=ex,mode='lines',name='Exact'));fig.add_trace(go.Scatter(y=sp,mode='markers',marker=dict(size=3,opacity=0.5),name='Sim'))
        fig.add_hline(y=0.5,line_dash="dash");fig.add_vline(x=22,line_dash="dash",line_color="green")
        fig.update_layout(height=380);st.plotly_chart(fig,use_container_width=True);st.success(f"n=23: P={ex[22]:.4f}")
    elif sub=='Monty Hall':
        nsim=st.slider("Sim:",1000,200000,50000);ws=0;ww=0
        for _ in range(nsim):
            p=np.random.randint(3);c=np.random.randint(3)
            if c==p: ws+=1
            else: ww+=1
        ps=ws/nsim;pw=ww/nsim;c1,c2=st.columns(2);c1.metric("Stay",f"{ps:.4f}");c2.metric("Switch",f"{pw:.4f}")
        fig=go.Figure(data=[go.Bar(x=['Stay','Switch'],y=[ps,pw],marker_color=['crimson','green'],text=[f'{ps:.4f}',f'{pw:.4f}'],textposition='auto')])
        fig.update_layout(height=340);st.plotly_chart(fig,use_container_width=True);st.success("Always switch! 2/3 vs 1/3")

elif module=='bayes':
    st.header("🧠 Teorema Bayes Lanjutan");sub=st.selectbox("Topik:",['Diagnostic Test','Sequential Updating','Bayesian Estimation'])
    if sub=='Diagnostic Test':
        st.latex(r"PPV=\frac{Sens\times Prev}{Sens\times Prev+(1-Spec)(1-Prev)}")
        c1,c2,c3=st.columns(3);prev=c1.number_input("Prevalence:",0.001,0.99,0.01,0.001);sens=c2.number_input("Sensitivity:",0.01,1.0,0.95,0.01);spec=c3.number_input("Specificity:",0.01,1.0,0.95,0.01)
        ppv=sens*prev/(sens*prev+(1-spec)*(1-prev));npv=spec*(1-prev)/(spec*(1-prev)+(1-sens)*prev)
        lrp=sens/(1-spec) if spec<1 else float('inf');lrn=(1-sens)/spec if spec>0 else float('inf')
        st.markdown(f"|Metrik|Nilai|\n|---|---|\n|PPV|**{ppv:.4f}** ({ppv*100:.2f}%)|\n|NPV|**{npv:.4f}**|\n|LR+|**{lrp:.2f}**|\n|LR-|**{lrn:.4f}**|")
        pop=100000;tp=int(pop*prev*sens);fn=int(pop*prev*(1-sens));fp=int(pop*(1-prev)*(1-spec));tn=int(pop*(1-prev)*spec)
        st.dataframe(pd.DataFrame([[tp,fp,tp+fp],[fn,tn,fn+tn],[tp+fn,fp+tn,pop]],index=['T+','T-','Tot'],columns=['D+','D-','Tot']),use_container_width=True)
        pvs=np.linspace(0.001,0.5,200);ppvs=sens*pvs/(sens*pvs+(1-spec)*(1-pvs))
        fig=go.Figure(data=[go.Scatter(x=pvs*100,y=ppvs*100,mode='lines',line=dict(color='crimson',width=2))]);fig.add_hline(y=50,line_dash="dash");fig.add_vline(x=prev*100,line_dash="dot")
        fig.update_layout(height=380,xaxis_title="Prevalence%",yaxis_title="PPV%");st.plotly_chart(fig,use_container_width=True)
    elif sub=='Sequential Updating':
        st.latex(r"Prior\xrightarrow{data}Posterior")
        pa=st.number_input("Prior a:",0.1,100.0,1.0,0.1);pb=st.number_input("Prior b:",0.1,100.0,1.0,0.1)
        di=st.text_input("Data (0/1):","1,1,0,1,1,1,0,1,0,1,1,1,0,1,1");data=[int(x.strip()) for x in di.split(',') if x.strip() in('0','1')]
        xp=np.linspace(0.001,0.999,300);a,b=pa,pb;fig=go.Figure()
        fig.add_trace(go.Scatter(x=xp,y=stats.beta.pdf(xp,a,b),mode='lines',name='Prior',line=dict(color='gray',dash='dash')))
        hist=[(a,b)]
        for d in data: a+=d;b+=(1-d);hist.append((a,b))
        for s in [len(data)//4,len(data)//2,len(data)]:
            as_,bs_=hist[s];fig.add_trace(go.Scatter(x=xp,y=stats.beta.pdf(xp,as_,bs_),mode='lines',name=f'n={s}'))
        fig.update_layout(height=420,xaxis_title="theta");st.plotly_chart(fig,use_container_width=True)
        ci_l=stats.beta.ppf(0.025,a,b);ci_h=stats.beta.ppf(0.975,a,b)
        st.success(f"Posterior: Beta({a:.1f},{b:.1f}) | E[theta]={a/(a+b):.4f} | 95%CI: [{ci_l:.4f},{ci_h:.4f}]")
    elif sub=='Bayesian Estimation':
        st.markdown("### Normal-Normal")
        c1,c2=st.columns(2);m0=c1.number_input("Prior mu0:",value=0.0);s0=c2.number_input("Prior sig0:",0.01,100.0,10.0,0.1)
        sig=st.number_input("Known sigma:",0.01,100.0,1.0);ds=st.text_input("Data:","2.1,1.8,2.5,1.9,2.3,2.0,2.2")
        data=[float(x.strip()) for x in ds.split(',') if x.strip()];nd=len(data);xb=np.mean(data) if data else 0
        pv=1/(1/s0**2+nd/sig**2);pm=pv*(m0/s0**2+nd*xb/sig**2);ps=np.sqrt(pv)
        xp=np.linspace(min(m0-3*s0,xb-3*sig/np.sqrt(max(nd,1))),max(m0+3*s0,xb+3*sig/np.sqrt(max(nd,1))),300)
        fig=go.Figure();fig.add_trace(go.Scatter(x=xp,y=stats.norm.pdf(xp,m0,s0),mode='lines',name='Prior',line=dict(dash='dash',color='gray')))
        if nd>0: fig.add_trace(go.Scatter(x=xp,y=stats.norm.pdf(xp,xb,sig/np.sqrt(nd)),mode='lines',name='Likelihood',line=dict(dash='dot',color='green')))
        fig.add_trace(go.Scatter(x=xp,y=stats.norm.pdf(xp,pm,ps),mode='lines',name='Posterior',line=dict(color='crimson',width=2)))
        fig.update_layout(height=420);st.plotly_chart(fig,use_container_width=True)
        st.success(f"Posterior: N({pm:.4f},{ps:.4f}^2) | 95%CI: [{pm-1.96*ps:.4f},{pm+1.96*ps:.4f}]")

elif module=='fitting':
    st.header("🔧 Distribution Fitting")
    ds=st.selectbox("Data:",['Generate','Manual','Upload CSV'])
    if ds=='Generate':
        gd=st.selectbox("From:",['Normal','Exponential','Gamma','Lognormal','Weibull','Mixed']);ng=st.slider("n:",50,10000,500)
        np.random.seed(42)
        if gd=='Normal': data=np.random.normal(5,2,ng)
        elif gd=='Exponential': data=np.random.exponential(3,ng)
        elif gd=='Gamma': data=np.random.gamma(2,3,ng)
        elif gd=='Lognormal': data=np.random.lognormal(1,0.5,ng)
        elif gd=='Weibull': data=np.random.weibull(2,ng)*5
        else: data=np.concatenate([np.random.normal(3,1,ng//2),np.random.normal(8,1.5,ng//2)])
    elif ds=='Manual':
        di=st.text_area("Data:","2.1,3.5,1.8,4.2,2.9,5.1,3.3,2.7,4.5,3.8,2.2,3.1,4.8,3.6,2.5")
        data=np.array([float(x.strip()) for x in di.replace('\n',',').split(',') if x.strip()])
    else:
        up=st.file_uploader("CSV:",type=['csv'])
        if up: df=pd.read_csv(up);col=st.selectbox("Col:",df.select_dtypes(include=[np.number]).columns);data=df[col].dropna().values
        else: st.info("Upload CSV");st.stop()
    if len(data)<5: st.warning("Too few data");st.stop()
    st.markdown(f"**n={len(data)}** | Mean={data.mean():.4f} | Std={data.std():.4f}")
    fig=px.histogram(data,nbins=40,histnorm='probability density',opacity=0.7);fig.update_layout(height=320);st.plotly_chart(fig,use_container_width=True)
    cands=['norm','expon','gamma','lognorm','weibull_min','beta','uniform','rayleigh','t']
    res=[];xf=np.linspace(data.min()-data.std(),data.max()+data.std(),300)
    fig=go.Figure();fig.add_trace(go.Histogram(x=data,nbinsx=40,histnorm='probability density',marker_color='steelblue',opacity=0.5,name='Data'))
    for dn in cands:
        try:
            do=getattr(stats,dn);p=do.fit(data);ks,ksp=stats.kstest(data,dn,args=p);ll=np.sum(do.logpdf(data,*p));kp=len(p)
            aic=2*kp-2*ll;bic=kp*np.log(len(data))-2*ll;res.append({'Dist':dn,'KS':round(ks,4),'p':round(ksp,6),'AIC':round(aic,2),'BIC':round(bic,2),'Params':str(tuple(round(x,3) for x in p))})
            fig.add_trace(go.Scatter(x=xf,y=do.pdf(xf,*p),mode='lines',name=dn,line=dict(width=1.5)))
        except: pass
    fig.update_layout(height=460,title="Fits");st.plotly_chart(fig,use_container_width=True)
    rdf=pd.DataFrame(res).sort_values('AIC');st.dataframe(rdf,use_container_width=True,hide_index=True)
    if len(rdf)>0:
        best=rdf.iloc[0];st.success(f"Best: **{best['Dist']}** (AIC={best['AIC']}, p={best['p']})")
        bd=getattr(stats,best['Dist']);bp=bd.fit(data);th=bd.ppf(np.linspace(0.01,0.99,len(data)),*bp);emp=np.sort(data)
        fig=go.Figure();fig.add_trace(go.Scatter(x=th,y=emp,mode='markers',marker=dict(size=4,opacity=0.5,color='steelblue')))
        mn=min(th.min(),emp.min());mx=max(th.max(),emp.max());fig.add_trace(go.Scatter(x=[mn,mx],y=[mn,mx],mode='lines',line=dict(color='red',dash='dash')))
        fig.update_layout(title=f"QQ: {best['Dist']}",height=420,xaxis_title="Theoretical",yaxis_title="Empirical");st.plotly_chart(fig,use_container_width=True)

st.divider()
st.markdown("<div style='text-align:center;color:gray;font-size:0.85rem'><b>Aplikasi Analisis Probabilitas Lengkap</b><br>10 Modul | 40+ Topik | Streamlit+SciPy+Plotly | 2026</div>",unsafe_allow_html=True)
