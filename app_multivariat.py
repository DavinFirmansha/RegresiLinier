import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.linalg import eigh, inv, det
from scipy.special import gammaln
from itertools import combinations
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Analisis Multivariat", page_icon="📐", layout="wide")
st.title("📐 Analisis Multivariat Komprehensif")
st.markdown("""
Aplikasi lengkap **12+ metode analisis multivariat** — dari reduksi dimensi, perbandingan grup,
analisis klaster, hingga analisis dependensi, dilengkapi berbagai **uji statistik & diagnostik**.
""")

# ============================================================
# HELPERS
# ============================================================
def standardize(X):
    m = X.mean(axis=0); s = X.std(axis=0, ddof=0); s[s==0]=1
    return (X - m) / s, m, s

def kmo_test(X):
    """Kaiser-Meyer-Olkin measure of sampling adequacy."""
    n, p = X.shape
    corr = np.corrcoef(X.T)
    try:
        corr_inv = np.linalg.inv(corr)
    except:
        corr_inv = np.linalg.pinv(corr)
    partial = np.zeros((p, p))
    for i in range(p):
        for j in range(p):
            if i != j:
                partial[i,j] = -corr_inv[i,j] / np.sqrt(corr_inv[i,i]*corr_inv[j,j])
    r2_sum = np.sum(corr**2) - p
    q2_sum = np.sum(partial**2) - p
    kmo_total = r2_sum / (r2_sum + q2_sum)
    kmo_per = []
    for j in range(p):
        r2 = np.sum(corr[j,:]**2) - 1
        q2 = np.sum(partial[j,:]**2) - 1
        kmo_per.append(r2 / (r2 + q2) if (r2+q2) > 0 else 0)
    return kmo_total, np.array(kmo_per)

def bartlett_sphericity(X):
    """Bartlett's test of sphericity."""
    n, p = X.shape
    corr = np.corrcoef(X.T)
    det_corr = max(np.linalg.det(corr), 1e-300)
    chi2 = -((n-1) - (2*p+5)/6) * np.log(det_corr)
    df = p*(p-1)/2
    p_val = 1 - stats.chi2.cdf(chi2, df)
    return chi2, df, p_val

def mardia_test(X):
    """Mardia's multivariate skewness and kurtosis test."""
    n, p = X.shape
    X_c = X - X.mean(axis=0)
    S = np.cov(X.T, ddof=1)
    try:
        S_inv = np.linalg.inv(S)
    except:
        S_inv = np.linalg.pinv(S)
    D = X_c @ S_inv @ X_c.T
    # Skewness
    b1p = np.sum(D**3) / n**2
    chi2_skew = n * b1p / 6
    df_skew = p*(p+1)*(p+2)/6
    p_skew = 1 - stats.chi2.cdf(chi2_skew, df_skew)
    # Kurtosis
    b2p = np.trace(D**2) / n
    z_kurt = (b2p - p*(p+2)) / np.sqrt(8*p*(p+2)/n)
    p_kurt = 2 * (1 - stats.norm.cdf(abs(z_kurt)))
    return b1p, chi2_skew, p_skew, b2p, z_kurt, p_kurt

def box_m_test(groups_data):
    """Box's M test for equality of covariance matrices."""
    k = len(groups_data)
    ns = [len(g) for g in groups_data]
    p = groups_data[0].shape[1]
    N = sum(ns)
    S_pooled = np.zeros((p, p))
    log_dets = []
    for i, g in enumerate(groups_data):
        Si = np.cov(g.T, ddof=1)
        S_pooled += (ns[i]-1) * Si
        d = np.linalg.det(Si)
        log_dets.append(np.log(max(d, 1e-300)))
    S_pooled /= (N - k)
    det_pool = np.linalg.det(S_pooled)
    M = (N-k)*np.log(max(det_pool,1e-300)) - sum((ns[i]-1)*log_dets[i] for i in range(k))
    c1 = (sum(1/(ns[i]-1) for i in range(k)) - 1/(N-k)) * (2*p**2+3*p-1)/(6*(p+1)*(k-1))
    chi2 = M * (1 - c1)
    df = p*(p+1)*(k-1)/2
    p_val = 1 - stats.chi2.cdf(chi2, df)
    return M, chi2, df, p_val

def hotelling_t2(X1, X2):
    """Hotelling's T² two-sample test."""
    n1, p = X1.shape; n2 = X2.shape[0]
    m1 = X1.mean(0); m2 = X2.mean(0); diff = m1 - m2
    S1 = np.cov(X1.T, ddof=1); S2 = np.cov(X2.T, ddof=1)
    Sp = ((n1-1)*S1 + (n2-1)*S2) / (n1+n2-2)
    try:
        Sp_inv = np.linalg.inv(Sp)
    except:
        Sp_inv = np.linalg.pinv(Sp)
    T2 = (n1*n2/(n1+n2)) * diff @ Sp_inv @ diff
    F_val = T2 * (n1+n2-p-1) / (p*(n1+n2-2))
    p_val = 1 - stats.f.cdf(F_val, p, n1+n2-p-1)
    return T2, F_val, p, n1+n2-p-1, p_val

def manova_test(X, groups):
    """MANOVA: Wilks, Pillai, Hotelling-Lawley, Roy."""
    unique_g = np.unique(groups); k = len(unique_g)
    n, p = X.shape
    gm = X.mean(0)
    # Between & Within
    B = np.zeros((p,p)); W = np.zeros((p,p))
    for g in unique_g:
        Xg = X[groups==g]
        ng = len(Xg); mg = Xg.mean(0)
        B += ng * np.outer(mg-gm, mg-gm)
        W += (Xg-mg).T @ (Xg-mg)
    try:
        W_inv = np.linalg.inv(W)
    except:
        W_inv = np.linalg.pinv(W)
    E = W_inv @ B
    eigenvalues = np.real(np.linalg.eigvals(E))
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    s = min(k-1, p)
    # Wilks' Lambda
    wilks = np.linalg.det(W) / np.linalg.det(W+B) if np.linalg.det(W+B)!=0 else 0
    # Pillai's Trace
    pillai = sum(ev/(1+ev) for ev in eigenvalues)
    # Hotelling-Lawley Trace
    hl_trace = sum(eigenvalues)
    # Roy's Largest Root
    roy = max(eigenvalues) if len(eigenvalues) > 0 else 0
    # Approx F for Wilks
    df_h = p*(k-1)
    df_e = n-k
    if wilks > 0:
        m = n - 1 - (p+k)/2
        ss = np.sqrt((p**2*(k-1)**2-4)/(p**2+(k-1)**2-5)) if (p**2+(k-1)**2-5)>0 else 1
        ss = max(ss, 1)
        lam_1s = max(wilks**(1/ss), 1e-15)
        df1 = p*(k-1); df2 = m*ss - p*(k-1)/2 + 1
        df2 = max(df2, 1)
        F_wilks = ((1-lam_1s)/lam_1s) * (df2/df1) if lam_1s > 0 else 0
        p_wilks = 1 - stats.f.cdf(F_wilks, df1, df2)
    else:
        F_wilks, p_wilks, df1, df2 = 0, 0, 1, 1
    # Approx F for Pillai
    ss_p = min(k-1, p)
    F_pillai = (pillai/(ss_p)) * ((n-k-p+ss_p)/(ss_p)) / ((1-pillai/ss_p)) if (1-pillai/ss_p)>0 else 0
    df1_p = p*(k-1); df2_p = ss_p*(n-k-p+ss_p)
    df2_p = max(df2_p, 1)
    p_pillai = 1 - stats.f.cdf(abs(F_pillai), df1_p, df2_p) if F_pillai > 0 else 1
    results = pd.DataFrame({
        'Statistik': ['Wilks\' Λ', 'Pillai\'s Trace', 'Hotelling-Lawley', 'Roy\'s Largest Root'],
        'Nilai': [round(wilks,6), round(pillai,6), round(hl_trace,6), round(roy,6)],
        'F-approx': [round(F_wilks,4), round(abs(F_pillai),4), '-', '-'],
        'p-value': [round(p_wilks,6), round(p_pillai,6), '-', '-'],
    })
    return results, eigenvalues, B, W

def pca_analysis(X, n_comp=None):
    """PCA via eigendecomposition of correlation matrix."""
    Xs, mu, sd = standardize(X)
    n, p = Xs.shape
    corr = np.corrcoef(Xs.T)
    eigenvalues, eigenvectors = np.linalg.eigh(corr)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]; eigenvectors = eigenvectors[:, idx]
    if n_comp is None: n_comp = p
    prop_var = eigenvalues / eigenvalues.sum()
    cum_var = np.cumsum(prop_var)
    loadings = eigenvectors[:, :n_comp] * np.sqrt(eigenvalues[:n_comp])
    scores = Xs @ eigenvectors[:, :n_comp]
    return eigenvalues, eigenvectors, loadings, scores, prop_var, cum_var

def factor_analysis(X, n_factors, rotation='varimax', max_iter=100):
    """Factor Analysis with optional varimax rotation."""
    Xs, mu, sd = standardize(X)
    n, p = Xs.shape
    corr = np.corrcoef(Xs.T)
    eigenvalues, eigenvectors = np.linalg.eigh(corr)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]; eigenvectors = eigenvectors[:, idx]
    L = eigenvectors[:,:n_factors] * np.sqrt(np.maximum(eigenvalues[:n_factors], 0))
    if rotation == 'varimax' and n_factors > 1:
        L = _varimax(L, max_iter=max_iter)
    communalities = np.sum(L**2, axis=1)
    uniqueness = 1 - communalities
    scores = Xs @ np.linalg.pinv(L.T)
    return L, communalities, uniqueness, scores

def _varimax(L, max_iter=100, tol=1e-6):
    """Varimax rotation."""
    p, k = L.shape
    R = np.eye(k)
    for _ in range(max_iter):
        Lrot = L @ R
        u, s, vt = np.linalg.svd(
            L.T @ (Lrot**3 - Lrot @ np.diag(np.sum(Lrot**2, axis=0)) / p))
        R_new = u @ vt
        if np.max(np.abs(R_new - R)) < tol: break
        R = R_new
    return L @ R

def lda_fit(X, y):
    """Linear Discriminant Analysis."""
    classes = np.unique(y); k = len(classes)
    n, p = X.shape; gm = X.mean(0)
    Sw = np.zeros((p,p)); Sb = np.zeros((p,p))
    priors = {}
    for c in classes:
        Xc = X[y==c]; nc = len(Xc); mc = Xc.mean(0)
        priors[c] = nc/n
        Sw += (Xc-mc).T @ (Xc-mc)
        Sb += nc * np.outer(mc-gm, mc-gm)
    try:
        Sw_inv = np.linalg.inv(Sw)
    except:
        Sw_inv = np.linalg.pinv(Sw)
    M = Sw_inv @ Sb
    eigenvalues, eigenvectors = np.linalg.eig(M)
    idx = np.argsort(np.real(eigenvalues))[::-1]
    eigenvalues = np.real(eigenvalues[idx])
    eigenvectors = np.real(eigenvectors[:, idx])
    n_disc = min(k-1, p)
    W = eigenvectors[:, :n_disc]
    scores = X @ W
    # Predict
    means_proj = {}
    for c in classes:
        means_proj[c] = X[y==c].mean(0)
    def predict(Xnew):
        preds = []
        for xi in Xnew:
            best_c = None; best_d = np.inf
            for c in classes:
                d = (xi - means_proj[c]) @ Sw_inv @ (xi - means_proj[c])
                d -= 2*np.log(priors[c])
                if d < best_d: best_d = d; best_c = c
            preds.append(best_c)
        return np.array(preds)
    y_pred = predict(X)
    acc = np.mean(y_pred == y)
    return W, eigenvalues[:n_disc], scores, y_pred, acc, priors

def kmeans_fit(X, k, max_iter=100, n_init=10):
    """K-Means clustering."""
    n, p = X.shape; best_inertia = np.inf; best_labels = None; best_centers = None
    for _ in range(n_init):
        idx = np.random.choice(n, k, replace=False)
        centers = X[idx].copy()
        for it in range(max_iter):
            dists = cdist(X, centers, 'euclidean')
            labels = np.argmin(dists, axis=1)
            new_centers = np.array([X[labels==j].mean(0) if np.sum(labels==j)>0 else centers[j] for j in range(k)])
            if np.allclose(new_centers, centers): break
            centers = new_centers
        inertia = sum(np.sum((X[labels==j]-centers[j])**2) for j in range(k))
        if inertia < best_inertia:
            best_inertia = inertia; best_labels = labels; best_centers = centers
    return best_labels, best_centers, best_inertia

def silhouette_score(X, labels):
    """Compute mean silhouette score."""
    n = len(X); sil = np.zeros(n)
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2: return 0, sil
    dist_matrix = squareform(pdist(X, 'euclidean'))
    for i in range(n):
        ci = labels[i]
        mask_in = labels == ci
        if mask_in.sum() <= 1:
            sil[i] = 0; continue
        a_i = np.mean(dist_matrix[i, mask_in & (np.arange(n)!=i)])
        b_i = np.inf
        for c in unique_labels:
            if c == ci: continue
            mask_c = labels == c
            if mask_c.sum() == 0: continue
            b_c = np.mean(dist_matrix[i, mask_c])
            b_i = min(b_i, b_c)
        sil[i] = (b_i - a_i) / max(a_i, b_i, 1e-15)
    return np.mean(sil), sil

def dbscan_fit(X, eps, min_samples):
    """DBSCAN clustering."""
    n = len(X); labels = np.full(n, -1); cluster_id = 0
    dist_m = squareform(pdist(X, 'euclidean'))
    visited = np.zeros(n, dtype=bool)
    for i in range(n):
        if visited[i]: continue
        visited[i] = True
        neighbors = np.where(dist_m[i] <= eps)[0]
        if len(neighbors) < min_samples: continue
        labels[i] = cluster_id
        seed_set = list(neighbors)
        j = 0
        while j < len(seed_set):
            q = seed_set[j]
            if not visited[q]:
                visited[q] = True
                q_neighbors = np.where(dist_m[q] <= eps)[0]
                if len(q_neighbors) >= min_samples:
                    seed_set.extend(q_neighbors.tolist())
            if labels[q] == -1:
                labels[q] = cluster_id
            j += 1
        cluster_id += 1
    return labels

def mds_classical(D, n_comp=2):
    """Classical Multidimensional Scaling."""
    n = D.shape[0]
    H = np.eye(n) - np.ones((n,n))/n
    B = -0.5 * H @ (D**2) @ H
    eigenvalues, eigenvectors = np.linalg.eigh(B)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx][:n_comp]
    eigenvectors = eigenvectors[:, idx][:, :n_comp]
    eigenvalues = np.maximum(eigenvalues, 0)
    coords = eigenvectors * np.sqrt(eigenvalues)
    return coords, eigenvalues

def canonical_corr(X, Y):
    """Canonical Correlation Analysis."""
    n = X.shape[0]; p = X.shape[1]; q = Y.shape[1]
    Xc = X - X.mean(0); Yc = Y - Y.mean(0)
    Sxx = Xc.T @ Xc / (n-1); Syy = Yc.T @ Yc / (n-1); Sxy = Xc.T @ Yc / (n-1)
    try:
        Sxx_inv_half = np.linalg.inv(np.linalg.cholesky(Sxx + np.eye(p)*1e-8))
        Syy_inv_half = np.linalg.inv(np.linalg.cholesky(Syy + np.eye(q)*1e-8))
    except:
        return np.array([]), np.array([[]]), np.array([[]])
    M = Sxx_inv_half @ Sxy @ Syy_inv_half.T @ Syy_inv_half @ Sxy.T @ Sxx_inv_half.T
    eigenvalues = np.real(np.linalg.eigvals(M))
    eigenvalues = np.sort(eigenvalues)[::-1][:min(p,q)]
    can_corr = np.sqrt(np.maximum(eigenvalues, 0))
    return can_corr

def correspondence_analysis(contingency):
    """Simple Correspondence Analysis."""
    P = contingency / contingency.sum()
    r = P.sum(axis=1); c = P.sum(axis=0)
    Dr_inv = np.diag(1/np.sqrt(r)); Dc_inv = np.diag(1/np.sqrt(c))
    S = Dr_inv @ (P - np.outer(r, c)) @ Dc_inv
    U, sigma, Vt = np.linalg.svd(S, full_matrices=False)
    n_dim = min(len(sigma), contingency.shape[0]-1, contingency.shape[1]-1)
    row_coords = Dr_inv @ U[:, :n_dim] * sigma[:n_dim]
    col_coords = Dc_inv @ Vt[:n_dim, :].T * sigma[:n_dim]
    inertia = sigma**2
    return row_coords, col_coords, inertia[:n_dim], sigma[:n_dim]

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.header("📁 Data")
uploaded = st.sidebar.file_uploader("Upload CSV / Excel", type=["csv","xlsx","xls"])
use_demo = st.sidebar.checkbox("Gunakan data demo", value=True if uploaded is None else False)

METHODS = {
    '🔻 PCA (Principal Component Analysis)': 'pca',
    '🔻 Factor Analysis (Analisis Faktor)': 'fa',
    '📊 MANOVA': 'manova',
    '📊 Hotelling T² Test': 'hotelling',
    '🎯 LDA (Discriminant Analysis)': 'lda',
    '🔵 K-Means Clustering': 'kmeans',
    '🌳 Hierarchical Clustering': 'hclust',
    '🔘 DBSCAN Clustering': 'dbscan',
    '📏 MDS (Multidimensional Scaling)': 'mds',
    '🔗 Canonical Correlation': 'cca',
    '📋 Correspondence Analysis': 'ca',
    '🧪 Multivariate Normality Test': 'mvnorm',
}
method_label = st.sidebar.selectbox("🔧 Metode Analisis", list(METHODS.keys()))
method = METHODS[method_label]

if uploaded is not None:
    if uploaded.name.endswith('.csv'): df = pd.read_csv(uploaded)
    else: df = pd.read_excel(uploaded)
    use_demo = False
elif use_demo:
    np.random.seed(42)
    if method in ['pca','fa','mds','mvnorm']:
        n = 150
        X1 = np.random.multivariate_normal([5,3,4,6], [[2,.8,.3,.1],[.8,1.5,.2,.05],[.3,.2,1,.1],[.1,.05,.1,.8]], n)
        X2 = np.random.normal(0,1,(n,2))
        df = pd.DataFrame(np.hstack([X1, X2]),
                          columns=['Tinggi_Tanaman','Jumlah_Daun','Berat_Buah','Diameter','Kelembaban','pH'])
    elif method in ['manova','hotelling','lda']:
        n = 120
        g1 = np.random.multivariate_normal([5,3,7], [[1,.3,.2],[.3,1,.1],[.2,.1,1.5]], 40)
        g2 = np.random.multivariate_normal([7,5,5], [[1,.2,.3],[.2,1.2,.1],[.3,.1,1]], 40)
        g3 = np.random.multivariate_normal([4,6,8], [[1.2,.1,.2],[.1,1,.15],[.2,.15,1.3]], 40)
        X = np.vstack([g1,g2,g3])
        grp = np.repeat(['Spesies_A','Spesies_B','Spesies_C'], 40)
        df = pd.DataFrame(X, columns=['Sepal_Length','Sepal_Width','Petal_Length'])
        df['Species'] = grp
    elif method in ['kmeans','hclust','dbscan']:
        c1 = np.random.normal([2,2], 0.8, (50,2))
        c2 = np.random.normal([7,7], 1.0, (50,2))
        c3 = np.random.normal([2,8], 0.6, (40,2))
        c4 = np.random.normal([8,2], 0.9, (35,2))
        X = np.vstack([c1,c2,c3,c4])
        df = pd.DataFrame(X, columns=['X1','X2'])
    elif method == 'cca':
        n = 100
        Z = np.random.multivariate_normal(np.zeros(6), np.eye(6)*0.5+0.5, n)
        df = pd.DataFrame(Z, columns=['Fisika','Kimia','Bio','Mtk','Bhs_Ind','Bhs_Ing'])
    elif method == 'ca':
        ct = np.array([[20,10,5],[15,25,10],[5,10,30],[10,5,15]])
        df = pd.DataFrame(ct, columns=['Produk_A','Produk_B','Produk_C'],
                          index=['Region_1','Region_2','Region_3','Region_4'])
        df.index.name = 'Region'
        df = df.reset_index()
    st.sidebar.success(f"Demo: {method_label}")
else:
    st.warning("Upload data atau gunakan data demo."); st.stop()

# ============================================================
# 1. EKSPLORASI
# ============================================================
st.header("1. Eksplorasi Data")
c1, c2 = st.columns(2)
c1.metric("N Observasi", df.shape[0]); c2.metric("Variabel", df.shape[1])
tab_d1, tab_d2, tab_d3 = st.tabs(["Data", "Deskriptif", "Korelasi"])
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(include=['object','category']).columns.tolist()
with tab_d1: st.dataframe(df.head(30), use_container_width=True)
with tab_d2: st.dataframe(df[numeric_cols].describe().T.round(4), use_container_width=True)
with tab_d3:
    if len(numeric_cols) >= 2:
        corr_m = df[numeric_cols].corr().round(4)
        fig = go.Figure(data=go.Heatmap(z=corr_m.values, x=corr_m.columns, y=corr_m.index,
                                         colorscale='RdBu_r', zmid=0, text=corr_m.values.round(2),
                                         texttemplate='%{text}'))
        fig.update_layout(title="Matriks Korelasi", height=500)
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# PCA
# ============================================================
if method == 'pca':
    st.header("2. Principal Component Analysis (PCA)")
    vars_pca = st.multiselect("Variabel Numerik:", numeric_cols, default=numeric_cols)
    if len(vars_pca) < 2:
        st.warning("Pilih minimal 2 variabel."); st.stop()
    X = df[vars_pca].dropna().values
    n, p = X.shape

    # Adequacy tests
    st.subheader("2a. Uji Kelayakan Data")
    kmo_val, kmo_per = kmo_test(X)
    bart_chi2, bart_df, bart_p = bartlett_sphericity(X)
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        kmo_label = ('Sangat Baik' if kmo_val>=0.9 else 'Baik' if kmo_val>=0.8 else
                     'Cukup' if kmo_val>=0.7 else 'Mediocre' if kmo_val>=0.6 else
                     'Buruk' if kmo_val>=0.5 else 'Tidak Layak')
        st.metric("KMO Overall", f"{kmo_val:.4f} ({kmo_label})")
        kmo_df = pd.DataFrame({'Variabel': vars_pca, 'KMO': kmo_per.round(4)})
        st.dataframe(kmo_df, use_container_width=True, hide_index=True)
    with col_t2:
        st.metric("Bartlett χ²", f"{bart_chi2:.4f}")
        st.markdown(f"**df =** {int(bart_df)}, **p-value =** {bart_p:.6f}")
        st.markdown(f"{'✅ Korelasi signifikan → PCA layak' if bart_p < 0.05 else '❌ Tidak signifikan'}")

    # PCA
    eigenvalues, eigenvectors, loadings, scores, prop_var, cum_var = pca_analysis(X)

    st.subheader("2b. Eigenvalues & Variance Explained")
    eig_df = pd.DataFrame({
        'PC': [f'PC{i+1}' for i in range(p)],
        'Eigenvalue': eigenvalues.round(4),
        '% Variance': (prop_var*100).round(2),
        '% Kumulatif': (cum_var*100).round(2),
    })
    st.dataframe(eig_df, use_container_width=True, hide_index=True)

    # Scree plot
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=eig_df['PC'], y=eig_df['Eigenvalue'], name='Eigenvalue',
                         marker_color='steelblue'), secondary_y=False)
    fig.add_trace(go.Scatter(x=eig_df['PC'], y=eig_df['% Kumulatif'], name='% Kumulatif',
                             mode='lines+markers', line=dict(color='red')), secondary_y=True)
    fig.add_hline(y=1, line_dash="dash", line_color="gray", annotation_text="Kaiser (λ=1)", secondary_y=False)
    fig.update_layout(title="Scree Plot", height=450)
    fig.update_yaxes(title_text="Eigenvalue", secondary_y=False)
    fig.update_yaxes(title_text="% Kumulatif", secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)

    n_retain = st.slider("Jumlah PC dipertahankan:", 1, p, int(np.sum(eigenvalues>=1)))

    st.subheader("2c. Loadings (Component Matrix)")
    load_df = pd.DataFrame(loadings[:, :n_retain],
                           columns=[f'PC{i+1}' for i in range(n_retain)], index=vars_pca).round(4)
    st.dataframe(load_df, use_container_width=True)

    # Biplot
    st.subheader("2d. Biplot (PC1 vs PC2)")
    if n_retain >= 2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=scores[:,0], y=scores[:,1], mode='markers',
                                 marker=dict(size=5, opacity=0.5), name='Scores'))
        for i, var in enumerate(vars_pca):
            fig.add_annotation(x=loadings[i,0]*3, y=loadings[i,1]*3, text=var,
                               showarrow=True, arrowhead=2, ax=0, ay=0)
        fig.update_layout(title="Biplot", xaxis_title=f"PC1 ({prop_var[0]*100:.1f}%)",
                          yaxis_title=f"PC2 ({prop_var[1]*100:.1f}%)", height=550)
        st.plotly_chart(fig, use_container_width=True)

    # Communalities
    st.subheader("2e. Communalities")
    comm = np.sum(loadings[:, :n_retain]**2, axis=1)
    comm_df = pd.DataFrame({'Variabel': vars_pca, 'Communality': comm.round(4),
                            'Uniqueness': (1-comm).round(4)})
    st.dataframe(comm_df, use_container_width=True, hide_index=True)

# ============================================================
# FACTOR ANALYSIS
# ============================================================
elif method == 'fa':
    st.header("2. Analisis Faktor (Factor Analysis)")
    vars_fa = st.multiselect("Variabel Numerik:", numeric_cols, default=numeric_cols)
    if len(vars_fa) < 2: st.warning("Pilih minimal 2 variabel."); st.stop()
    X = df[vars_fa].dropna().values; n, p = X.shape

    st.subheader("2a. Uji Kelayakan")
    kmo_val, kmo_per = kmo_test(X)
    bart_chi2, bart_df, bart_p = bartlett_sphericity(X)
    st.markdown(f"**KMO:** {kmo_val:.4f} | **Bartlett χ²:** {bart_chi2:.4f} (p={bart_p:.6f})")
    if kmo_val < 0.5: st.warning("KMO < 0.5 → Data mungkin tidak cocok untuk FA.")

    n_factors = st.slider("Jumlah Faktor:", 1, p, min(3, p))
    rotation = st.selectbox("Rotasi:", ['varimax','none'])
    rot = rotation if rotation != 'none' else None
    L, comm, uniq, fscores = factor_analysis(X, n_factors, rotation=rot if rot else 'varimax')
    if rot is None:
        eigenvalues, eigenvectors = np.linalg.eigh(np.corrcoef(X.T))
        idx = np.argsort(eigenvalues)[::-1]
        L = eigenvectors[:,:n_factors] * np.sqrt(np.maximum(eigenvalues[idx[:n_factors]], 0))
        comm = np.sum(L**2, axis=1); uniq = 1-comm

    st.subheader("2b. Factor Loadings")
    load_df = pd.DataFrame(L, columns=[f'Factor{i+1}' for i in range(n_factors)], index=vars_fa).round(4)
    st.dataframe(load_df, use_container_width=True)
    # Highlight
    st.markdown("*Loading > 0.5 dianggap signifikan.*")

    st.subheader("2c. Communalities")
    comm_df = pd.DataFrame({'Variabel': vars_fa, 'Communality': comm.round(4), 'Uniqueness': uniq.round(4)})
    st.dataframe(comm_df, use_container_width=True, hide_index=True)

    st.subheader("2d. Variance Explained")
    var_exp = np.sum(L**2, axis=0)
    var_pct = var_exp / p * 100
    ve_df = pd.DataFrame({
        'Factor': [f'Factor{i+1}' for i in range(n_factors)],
        'SS Loadings': var_exp.round(4),
        '% Variance': var_pct.round(2),
        '% Kumulatif': np.cumsum(var_pct).round(2),
    })
    st.dataframe(ve_df, use_container_width=True, hide_index=True)

    # Loading plot
    if n_factors >= 2:
        fig = go.Figure()
        for i, var in enumerate(vars_fa):
            fig.add_trace(go.Scatter(x=[0, L[i,0]], y=[0, L[i,1]], mode='lines+text',
                                     text=['', var], textposition='top center',
                                     line=dict(width=2), showlegend=False))
        fig.add_shape(type='circle', x0=-1, y0=-1, x1=1, y1=1, line=dict(dash='dash', color='gray'))
        fig.update_layout(title="Loading Plot", xaxis_title="Factor 1", yaxis_title="Factor 2",
                          height=500, xaxis=dict(range=[-1.2,1.2]), yaxis=dict(range=[-1.2,1.2]))
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# MANOVA
# ============================================================
elif method == 'manova':
    st.header("2. MANOVA (Multivariate ANOVA)")
    resp_vars = st.multiselect("Variabel Respon (≥2):", numeric_cols, default=numeric_cols[:min(3,len(numeric_cols))])
    group_col = st.selectbox("Variabel Grup:", cat_cols + [c for c in numeric_cols if df[c].nunique()<=10 and c not in resp_vars])
    alpha = st.selectbox("α", [0.01, 0.05, 0.10], index=1)
    if len(resp_vars) < 2: st.warning("Pilih minimal 2 variabel respon."); st.stop()
    panel = df[resp_vars + [group_col]].dropna()
    X = panel[resp_vars].values; groups = panel[group_col].values

    # Assumptions
    st.subheader("2a. Uji Asumsi")
    tab_as1, tab_as2 = st.tabs(["Normalitas Multivariat", "Homogenitas Kovariansi"])
    with tab_as1:
        b1, chi2_s, p_s, b2, z_k, p_k = mardia_test(X)
        st.markdown(f"""
        **Mardia's Test:**
        | | Statistik | p-value | Keputusan |
        |---|---|---|---|
        | Skewness (b₁,p) | {b1:.4f} (χ²={chi2_s:.4f}) | {p_s:.6f} | {'Normal' if p_s > alpha else 'Tidak Normal'} |
        | Kurtosis (b₂,p) | {b2:.4f} (z={z_k:.4f}) | {p_k:.6f} | {'Normal' if p_k > alpha else 'Tidak Normal'} |
        """)
    with tab_as2:
        gdata = [panel[panel[group_col]==g][resp_vars].values for g in np.unique(groups)]
        if all(len(g) > X.shape[1] for g in gdata):
            M, chi2_bm, df_bm, p_bm = box_m_test(gdata)
            st.markdown(f"**Box's M Test:** M = {M:.4f}, χ² = {chi2_bm:.4f}, df = {int(df_bm)}, p = {p_bm:.6f}")
            st.markdown(f"{'✅ Matriks kovariansi homogen' if p_bm > alpha else '⚠️ Matriks kovariansi TIDAK homogen'}")

    # MANOVA
    st.subheader("2b. Tabel MANOVA")
    manova_res, eigvals, B, W = manova_test(X, groups)
    st.dataframe(manova_res, use_container_width=True, hide_index=True)

    # Per-variable ANOVA
    st.subheader("2c. Univariate ANOVA per Variabel")
    uniq_g = np.unique(groups)
    uanova_rows = []
    for var in resp_vars:
        gdata_var = [panel[panel[group_col]==g][var].values for g in uniq_g]
        f_val, p_val = stats.f_oneway(*gdata_var)
        uanova_rows.append({'Variabel': var, 'F': round(f_val,4), 'p-value': round(p_val,6),
                            f'Sig (α={alpha})': 'Ya' if p_val < alpha else 'Tidak'})
    st.dataframe(pd.DataFrame(uanova_rows), use_container_width=True, hide_index=True)

    # Means
    st.subheader("2d. Mean per Grup")
    means_df = panel.groupby(group_col)[resp_vars].mean().round(4).reset_index()
    st.dataframe(means_df, use_container_width=True, hide_index=True)

# ============================================================
# HOTELLING T²
# ============================================================
elif method == 'hotelling':
    st.header("2. Hotelling's T² Test")
    resp_vars = st.multiselect("Variabel Respon (≥2):", numeric_cols, default=numeric_cols[:min(3,len(numeric_cols))])
    group_col = st.selectbox("Variabel Grup (2 level):", cat_cols + [c for c in numeric_cols if df[c].nunique()==2])
    alpha = st.selectbox("α", [0.01, 0.05, 0.10], index=1)
    if len(resp_vars) < 2: st.warning("Pilih minimal 2 variabel."); st.stop()
    panel = df[resp_vars + [group_col]].dropna()
    g_levels = panel[group_col].unique()
    if len(g_levels) != 2: st.error("Harus tepat 2 grup."); st.stop()
    X1 = panel[panel[group_col]==g_levels[0]][resp_vars].values
    X2 = panel[panel[group_col]==g_levels[1]][resp_vars].values
    T2, F_val, df1, df2, p_val = hotelling_t2(X1, X2)
    st.markdown(f"""
    | Statistik | Nilai |
    |---|---|
    | T² | {T2:.4f} |
    | F | {F_val:.4f} |
    | df | ({df1}, {df2}) |
    | p-value | {p_val:.6f} |
    | Keputusan (α={alpha}) | {'**Berbeda signifikan**' if p_val < alpha else 'Tidak berbeda'} |
    """)
    st.subheader("Mean per Grup")
    for g in g_levels:
        sub = panel[panel[group_col]==g][resp_vars]
        st.markdown(f"**{g}:** {', '.join([f'{v}={sub[v].mean():.4f}' for v in resp_vars])}")

# ============================================================
# LDA
# ============================================================
elif method == 'lda':
    st.header("2. Linear Discriminant Analysis (LDA)")
    pred_vars = st.multiselect("Variabel Prediktor:", numeric_cols, default=numeric_cols[:min(3,len(numeric_cols))])
    group_col = st.selectbox("Variabel Grup:", cat_cols + [c for c in numeric_cols if df[c].nunique()<=10 and c not in pred_vars])
    alpha = st.selectbox("α", [0.01, 0.05, 0.10], index=1)
    if len(pred_vars) < 2: st.warning("Pilih minimal 2 prediktor."); st.stop()
    panel = df[pred_vars + [group_col]].dropna()
    X = panel[pred_vars].values; y = panel[group_col].values

    # Assumptions
    st.subheader("2a. Uji Asumsi")
    gdata = [panel[panel[group_col]==g][pred_vars].values for g in np.unique(y)]
    if all(len(g) > X.shape[1] for g in gdata) and len(gdata)>1:
        M, chi2_bm, df_bm, p_bm = box_m_test(gdata)
        st.markdown(f"**Box's M:** χ²={chi2_bm:.4f}, p={p_bm:.6f} → "
                    f"{'Homogen' if p_bm > alpha else 'Tidak Homogen (pertimbangkan QDA)'}")

    # Fit
    W_lda, eigvals_lda, scores_lda, y_pred, acc, priors = lda_fit(X, y)

    st.subheader("2b. Discriminant Functions")
    coef_df = pd.DataFrame(W_lda, index=pred_vars,
                           columns=[f'LD{i+1}' for i in range(W_lda.shape[1])]).round(4)
    st.dataframe(coef_df, use_container_width=True)
    eig_prop = eigvals_lda / eigvals_lda.sum() if eigvals_lda.sum() > 0 else eigvals_lda
    eig_lda_df = pd.DataFrame({
        'LD': [f'LD{i+1}' for i in range(len(eigvals_lda))],
        'Eigenvalue': eigvals_lda.round(4),
        '% Variance': (eig_prop*100).round(2),
    })
    st.dataframe(eig_lda_df, use_container_width=True, hide_index=True)

    # Confusion Matrix
    st.subheader("2c. Klasifikasi (Resubstitution)")
    st.metric("Akurasi", f"{acc:.2%}")
    classes = np.unique(y)
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    for i, c1 in enumerate(classes):
        for j, c2 in enumerate(classes):
            cm[i,j] = np.sum((y==c1) & (y_pred==c2))
    cm_df = pd.DataFrame(cm, index=[f'Actual: {c}' for c in classes],
                         columns=[f'Pred: {c}' for c in classes])
    st.dataframe(cm_df, use_container_width=True)

    # Classification report
    report_rows = []
    for i, c in enumerate(classes):
        tp = cm[i,i]; fp = cm[:,i].sum()-tp; fn = cm[i,:].sum()-tp
        prec = tp/(tp+fp) if (tp+fp)>0 else 0
        rec = tp/(tp+fn) if (tp+fn)>0 else 0
        f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0
        report_rows.append({'Class': c, 'Precision': round(prec,4), 'Recall': round(rec,4),
                            'F1-Score': round(f1,4), 'N': int(cm[i,:].sum())})
    st.dataframe(pd.DataFrame(report_rows), use_container_width=True, hide_index=True)

    # Plot
    if scores_lda.shape[1] >= 2:
        fig = px.scatter(x=scores_lda[:,0], y=scores_lda[:,1], color=y,
                         labels={'x':'LD1','y':'LD2','color':'Group'},
                         title="LDA Scores Plot")
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    elif scores_lda.shape[1] == 1:
        fig = px.histogram(x=scores_lda[:,0], color=y, barmode='overlay', opacity=0.6,
                           labels={'x':'LD1','color':'Group'}, title="LDA Scores Distribution")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# K-MEANS
# ============================================================
elif method == 'kmeans':
    st.header("2. K-Means Clustering")
    vars_cl = st.multiselect("Variabel:", numeric_cols, default=numeric_cols)
    if len(vars_cl) < 2: st.warning("Pilih minimal 2 variabel."); st.stop()
    X_raw = df[vars_cl].dropna().values
    Xs, mu, sd = standardize(X_raw)

    # Elbow
    st.subheader("2a. Elbow Method & Silhouette")
    max_k = min(10, len(Xs)-1)
    inertias = []; sil_scores = []
    for k in range(2, max_k+1):
        labels_k, centers_k, inertia_k = kmeans_fit(Xs, k)
        inertias.append(inertia_k)
        sil_k, _ = silhouette_score(Xs, labels_k)
        sil_scores.append(sil_k)
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Elbow Plot","Silhouette Score"))
    fig.add_trace(go.Scatter(x=list(range(2,max_k+1)), y=inertias, mode='lines+markers'), row=1, col=1)
    fig.add_trace(go.Scatter(x=list(range(2,max_k+1)), y=sil_scores, mode='lines+markers',
                             line=dict(color='green')), row=1, col=2)
    fig.update_layout(height=380, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    optimal_k_sil = np.argmax(sil_scores) + 2
    st.info(f"Silhouette optimal: **k = {optimal_k_sil}** (score = {max(sil_scores):.4f})")

    k = st.slider("Jumlah Cluster (k):", 2, max_k, optimal_k_sil)
    labels, centers, inertia = kmeans_fit(Xs, k)
    sil_mean, sil_vals = silhouette_score(Xs, labels)

    st.subheader("2b. Hasil Clustering")
    col_r1, col_r2, col_r3 = st.columns(3)
    col_r1.metric("k", k); col_r2.metric("Inertia (WCSS)", f"{inertia:.2f}")
    col_r3.metric("Silhouette", f"{sil_mean:.4f}")

    df_result = df[vars_cl].dropna().copy()
    df_result['Cluster'] = labels.astype(str)

    # Cluster profile
    st.subheader("2c. Profil Cluster")
    profile = df_result.groupby('Cluster')[vars_cl].agg(['mean','std','count']).round(4)
    st.dataframe(profile, use_container_width=True)

    # Scatter
    if len(vars_cl) >= 2:
        fig = px.scatter(df_result, x=vars_cl[0], y=vars_cl[1], color='Cluster',
                         title="Cluster Scatter Plot")
        centers_orig = centers * sd + mu
        fig.add_trace(go.Scatter(x=centers_orig[:,0], y=centers_orig[:,1], mode='markers',
                                 marker=dict(symbol='x', size=15, color='black', line=dict(width=2)),
                                 name='Centroids'))
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

    # Silhouette per cluster
    sil_df = pd.DataFrame({'Cluster': labels.astype(str), 'Silhouette': sil_vals})
    fig = px.box(sil_df, x='Cluster', y='Silhouette', color='Cluster', title="Silhouette per Cluster")
    fig.add_hline(y=sil_mean, line_dash="dash", annotation_text=f"Mean={sil_mean:.3f}")
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# HIERARCHICAL CLUSTERING
# ============================================================
elif method == 'hclust':
    st.header("2. Hierarchical Clustering")
    vars_cl = st.multiselect("Variabel:", numeric_cols, default=numeric_cols)
    if len(vars_cl) < 2: st.warning("Pilih minimal 2 variabel."); st.stop()
    X_raw = df[vars_cl].dropna().values; Xs, mu, sd = standardize(X_raw)
    linkage_method = st.selectbox("Metode Linkage:", ['ward','complete','average','single'])
    dist_metric = 'euclidean' if linkage_method == 'ward' else st.selectbox("Metrik Jarak:", ['euclidean','cityblock','cosine'])
    Z = linkage(Xs, method=linkage_method, metric=dist_metric)

    st.subheader("2a. Dendrogram")
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig_mpl, ax = plt.subplots(figsize=(12, 5))
    dn = dendrogram(Z, ax=ax, truncate_mode='lastp', p=min(30, len(Xs)),
                    leaf_rotation=90, leaf_font_size=8)
    ax.set_title("Dendrogram"); ax.set_ylabel("Distance")
    st.pyplot(fig_mpl)

    n_clust = st.slider("Jumlah Cluster:", 2, min(10, len(Xs)-1), 3)
    labels_hc = fcluster(Z, n_clust, criterion='maxclust') - 1
    sil_mean, sil_vals = silhouette_score(Xs, labels_hc)
    st.metric("Silhouette Score", f"{sil_mean:.4f}")

    df_result = df[vars_cl].dropna().copy(); df_result['Cluster'] = labels_hc.astype(str)
    st.subheader("2b. Profil Cluster")
    st.dataframe(df_result.groupby('Cluster')[vars_cl].mean().round(4), use_container_width=True)

    if len(vars_cl) >= 2:
        fig = px.scatter(df_result, x=vars_cl[0], y=vars_cl[1], color='Cluster', title="Cluster Plot")
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

    # Agglomerative schedule
    st.subheader("2c. Agglomeration Schedule (last 15)")
    n_obs = len(Xs)
    sched = pd.DataFrame(Z[-15:], columns=['Cluster1','Cluster2','Distance','N_obs'])
    sched['Step'] = range(n_obs-15, n_obs)
    st.dataframe(sched.round(4), use_container_width=True, hide_index=True)

# ============================================================
# DBSCAN
# ============================================================
elif method == 'dbscan':
    st.header("2. DBSCAN Clustering")
    vars_cl = st.multiselect("Variabel:", numeric_cols, default=numeric_cols)
    if len(vars_cl) < 2: st.warning("Pilih minimal 2 variabel."); st.stop()
    X_raw = df[vars_cl].dropna().values; Xs, mu, sd = standardize(X_raw)

    # k-distance plot
    st.subheader("2a. k-Distance Plot (untuk menentukan eps)")
    k_nn = st.slider("k (MinPts):", 2, 20, 5)
    dists_all = squareform(pdist(Xs, 'euclidean'))
    k_dists = np.sort(dists_all, axis=1)[:, k_nn]
    k_dists_sorted = np.sort(k_dists)[::-1]
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=k_dists_sorted, mode='lines', name=f'{k_nn}-distance'))
    fig.update_layout(title=f"{k_nn}-Distance Plot (sorted desc)", xaxis_title="Points", yaxis_title="Distance", height=400)
    st.plotly_chart(fig, use_container_width=True)

    eps = st.slider("eps (radius):", 0.1, float(np.percentile(k_dists, 95))*2, float(np.median(k_dists)), 0.05)
    min_samples = st.slider("min_samples:", 2, 20, k_nn)
    labels_db = dbscan_fit(Xs, eps, min_samples)
    n_clusters = len(set(labels_db) - {-1})
    n_noise = np.sum(labels_db == -1)
    st.markdown(f"**Cluster ditemukan:** {n_clusters} | **Noise points:** {n_noise} ({n_noise/len(labels_db):.1%})")
    if n_clusters > 1:
        non_noise = labels_db != -1
        sil_mean, _ = silhouette_score(Xs[non_noise], labels_db[non_noise])
        st.metric("Silhouette (excl. noise)", f"{sil_mean:.4f}")

    df_result = df[vars_cl].dropna().copy(); df_result['Cluster'] = labels_db.astype(str)
    if len(vars_cl) >= 2:
        fig = px.scatter(df_result, x=vars_cl[0], y=vars_cl[1], color='Cluster',
                         title="DBSCAN Cluster Plot", color_discrete_map={'-1':'lightgray'})
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    st.subheader("2b. Profil Cluster")
    st.dataframe(df_result.groupby('Cluster')[vars_cl].agg(['mean','count']).round(4), use_container_width=True)

# ============================================================
# MDS
# ============================================================
elif method == 'mds':
    st.header("2. Multidimensional Scaling (MDS)")
    vars_mds = st.multiselect("Variabel:", numeric_cols, default=numeric_cols)
    if len(vars_mds) < 2: st.warning("Pilih minimal 2 variabel."); st.stop()
    X_raw = df[vars_mds].dropna().values; Xs, mu, sd = standardize(X_raw)
    dist_metric = st.selectbox("Metrik Jarak:", ['euclidean','cityblock','cosine','correlation'])
    D = squareform(pdist(Xs, dist_metric))
    n_comp = st.slider("Dimensi MDS:", 2, min(5, len(Xs)-1), 2)
    coords, eigvals_mds = mds_classical(D, n_comp)

    # Stress (Kruskal)
    D_mds = squareform(pdist(coords, 'euclidean'))
    stress = np.sqrt(np.sum((D - D_mds)**2) / np.sum(D**2))
    stress_label = ('Excellent' if stress < 0.025 else 'Good' if stress < 0.05 else
                    'Fair' if stress < 0.1 else 'Poor' if stress < 0.2 else 'Bad')
    st.metric("Stress (Kruskal)", f"{stress:.4f} ({stress_label})")

    fig = px.scatter(x=coords[:,0], y=coords[:,1],
                     labels={'x':'Dim 1','y':'Dim 2'}, title="MDS Configuration")
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    # Shepard diagram
    st.subheader("Shepard Diagram")
    D_orig_flat = D[np.triu_indices_from(D, k=1)]
    D_mds_flat = D_mds[np.triu_indices_from(D_mds, k=1)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=D_orig_flat, y=D_mds_flat, mode='markers', opacity=0.3, name='Pairs'))
    r_max = max(D_orig_flat.max(), D_mds_flat.max())
    fig.add_trace(go.Scatter(x=[0,r_max], y=[0,r_max], mode='lines', line=dict(dash='dash',color='red'), name='Perfect'))
    fig.update_layout(title="Shepard Diagram", xaxis_title="Original Distance",
                      yaxis_title="MDS Distance", height=450)
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# CANONICAL CORRELATION
# ============================================================
elif method == 'cca':
    st.header("2. Canonical Correlation Analysis (CCA)")
    st.markdown("Analisis hubungan antara **dua set variabel**.")
    set_x = st.multiselect("Set X (variabel 1):", numeric_cols, default=numeric_cols[:len(numeric_cols)//2])
    remaining = [c for c in numeric_cols if c not in set_x]
    set_y = st.multiselect("Set Y (variabel 2):", remaining, default=remaining)
    if len(set_x) < 1 or len(set_y) < 1: st.warning("Pilih min 1 variabel di tiap set."); st.stop()
    panel = df[set_x + set_y].dropna()
    Xc = panel[set_x].values; Yc = panel[set_y].values
    can_corr = canonical_corr(Xc, Yc)

    if len(can_corr) > 0:
        st.subheader("2a. Canonical Correlations")
        cc_df = pd.DataFrame({
            'Canonical Variate': [f'CV{i+1}' for i in range(len(can_corr))],
            'Canonical r': can_corr.round(4),
            'r²': (can_corr**2).round(4),
            'Eigenvalue (r²/(1-r²))': (can_corr**2/(1-can_corr**2+1e-15)).round(4),
        })
        st.dataframe(cc_df, use_container_width=True, hide_index=True)
        # Wilks test
        n = len(panel); p = len(set_x); q = len(set_y)
        st.subheader("2b. Significance Tests")
        sig_rows = []
        for m in range(len(can_corr)):
            wilks = np.prod(1 - can_corr[m:]**2)
            df_num = (p-m)*(q-m); df_den = n - 1 - (p+q+1)/2
            chi2 = -(n - 1 - (p+q+1)/2) * np.log(max(wilks, 1e-300))
            pv = 1 - stats.chi2.cdf(chi2, df_num)
            sig_rows.append({'Test (CV≥'+str(m+1)+')': f'Wilks Λ={wilks:.4f}',
                             'χ²': round(chi2,4), 'df': int(df_num),
                             'p-value': round(pv,6)})
        st.dataframe(pd.DataFrame(sig_rows), use_container_width=True, hide_index=True)

# ============================================================
# CORRESPONDENCE ANALYSIS
# ============================================================
elif method == 'ca':
    st.header("2. Correspondence Analysis (CA)")
    st.markdown("Untuk data **tabel kontingensi** (frekuensi).")
    row_var = st.selectbox("Variabel Baris:", df.columns)
    val_vars = st.multiselect("Variabel Kolom (frekuensi):", [c for c in numeric_cols if c != row_var],
                              default=[c for c in numeric_cols if c != row_var])
    if len(val_vars) < 2: st.warning("Pilih min 2 kolom frekuensi."); st.stop()
    ct = df.set_index(row_var)[val_vars].values.astype(float)
    row_labels = df[row_var].values.astype(str)
    col_labels = val_vars

    row_coords, col_coords, inertia, sing_vals = correspondence_analysis(ct)
    total_inertia = inertia.sum()

    st.subheader("2a. Inertia")
    in_df = pd.DataFrame({
        'Dimensi': [f'Dim{i+1}' for i in range(len(inertia))],
        'Singular Value': sing_vals.round(4),
        'Inertia': inertia.round(4),
        '% Inertia': (inertia/total_inertia*100).round(2),
        '% Kumulatif': (np.cumsum(inertia)/total_inertia*100).round(2),
    })
    st.dataframe(in_df, use_container_width=True, hide_index=True)
    st.markdown(f"**Total Inertia (χ²/n):** {total_inertia:.4f}")

    # CA Map
    if row_coords.shape[1] >= 2 and col_coords.shape[1] >= 2:
        st.subheader("2b. CA Biplot")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=row_coords[:,0], y=row_coords[:,1], mode='markers+text',
                                 text=row_labels, textposition='top center', name='Baris',
                                 marker=dict(size=10, color='blue')))
        fig.add_trace(go.Scatter(x=col_coords[:,0], y=col_coords[:,1], mode='markers+text',
                                 text=col_labels, textposition='bottom center', name='Kolom',
                                 marker=dict(size=12, symbol='diamond', color='red')))
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.add_vline(x=0, line_dash="dash", line_color="gray")
        fig.update_layout(title="CA Biplot", xaxis_title=f"Dim1 ({inertia[0]/total_inertia*100:.1f}%)",
                          yaxis_title=f"Dim2 ({inertia[1]/total_inertia*100:.1f}%)" if len(inertia)>1 else "Dim2",
                          height=550)
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# MULTIVARIATE NORMALITY
# ============================================================
elif method == 'mvnorm':
    st.header("2. Uji Normalitas Multivariat")
    vars_test = st.multiselect("Variabel:", numeric_cols, default=numeric_cols)
    if len(vars_test) < 2: st.warning("Pilih minimal 2 variabel."); st.stop()
    X = df[vars_test].dropna().values; n, p = X.shape
    alpha = st.selectbox("α", [0.01, 0.05, 0.10], index=1)

    st.subheader("2a. Mardia's Test")
    b1, chi2_s, p_s, b2, z_k, p_k = mardia_test(X)
    st.markdown(f"""
    | Komponen | Statistik | Test Stat | p-value | Keputusan (α={alpha}) |
    |----------|-----------|-----------|---------|----------|
    | **Skewness** (b₁,p) | {b1:.4f} | χ² = {chi2_s:.4f} | {p_s:.6f} | {'Normal' if p_s>alpha else '**Tidak Normal**'} |
    | **Kurtosis** (b₂,p) | {b2:.4f} | z = {z_k:.4f} | {p_k:.6f} | {'Normal' if p_k>alpha else '**Tidak Normal**'} |
    """)
    overall = "✅ **Normal Multivariat**" if (p_s > alpha and p_k > alpha) else "❌ **Tidak Normal Multivariat**"
    st.markdown(f"**Keputusan keseluruhan:** {overall}")

    st.subheader("2b. Univariate Normality (Shapiro-Wilk)")
    sw_rows = []
    for v in vars_test:
        vals = df[v].dropna().values
        if len(vals) >= 3:
            w, p_sw = stats.shapiro(vals)
            sw_rows.append({'Variabel': v, 'W': round(w,4), 'p-value': round(p_sw,6),
                            'Normal?': 'Ya' if p_sw > alpha else 'Tidak'})
    st.dataframe(pd.DataFrame(sw_rows), use_container_width=True, hide_index=True)

    st.subheader("2c. Chi-Square QQ Plot (Mahalanobis)")
    X_c = X - X.mean(0)
    S = np.cov(X.T, ddof=1)
    try: S_inv = np.linalg.inv(S)
    except: S_inv = np.linalg.pinv(S)
    mahal = np.array([x @ S_inv @ x for x in X_c])
    mahal_sorted = np.sort(mahal)
    chi2_quantiles = stats.chi2.ppf(np.linspace(1/(n+1), n/(n+1), n), p)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=chi2_quantiles, y=mahal_sorted, mode='markers', name='Data', opacity=0.5))
    qmax = max(chi2_quantiles.max(), mahal_sorted.max())
    fig.add_trace(go.Scatter(x=[0,qmax], y=[0,qmax], mode='lines', name='Normal Line',
                             line=dict(dash='dash', color='red')))
    fig.update_layout(title="Chi-Square QQ-Plot (Mahalanobis Distance)",
                      xaxis_title="χ² Theoretical Quantiles", yaxis_title="Mahalanobis D²",
                      height=500)
    st.plotly_chart(fig, use_container_width=True)
    pct_within = np.mean(mahal <= stats.chi2.ppf(0.5, p))
    st.markdown(f"**% observasi dengan D² ≤ χ²₀.₅:** {pct_within:.1%} (expected ≈ 50%)")

# ============================================================
# EXPORT
# ============================================================
st.header("📥 Ekspor")
st.download_button("📥 Data (CSV)", data=df.to_csv(index=False),
                   file_name="multivariate_data.csv", mime="text/csv")

# FOOTER
st.markdown("---")
st.markdown("""
**Referensi Metodologis:**
- **PCA:** Reduksi dimensi via dekomposisi matriks korelasi. Kaiser rule (λ≥1), Scree plot.
- **Factor Analysis:** Mencari faktor laten; rotasi Varimax untuk interpretabilitas.
- **KMO & Bartlett:** Uji kelayakan data untuk PCA/FA. KMO ≥ 0.6 direkomendasikan.
- **MANOVA:** Perbandingan vektor mean antar grup. 4 statistik: Wilks', Pillai, Hotelling-Lawley, Roy.
- **Hotelling T²:** Generalisasi multivariat dari uji t dua sampel.
- **Box's M:** Uji homogenitas matriks kovariansi antar grup.
- **LDA:** Mencari fungsi diskriminan linear untuk klasifikasi.
- **K-Means:** Partisi data ke k cluster; optimasi WCSS. Elbow & Silhouette untuk pemilihan k.
- **Hierarchical Clustering:** Aglomeratif (bottom-up). Metode: Ward, Complete, Average, Single.
- **DBSCAN:** Density-based clustering; deteksi outlier/noise otomatis.
- **MDS:** Representasi jarak antar-objek dalam dimensi rendah. Stress < 0.05 = baik.
- **CCA:** Korelasi kanonik antar dua set variabel.
- **Correspondence Analysis:** Visualisasi tabel kontingensi dalam ruang berdimensi rendah.
- **Mardia's Test:** Uji normalitas multivariat (skewness & kurtosis).

Dibangun dengan **Streamlit** + **SciPy** + **NumPy** + **Plotly** | Python
""")
