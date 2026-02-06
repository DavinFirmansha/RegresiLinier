import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Uji Nonparametrik K Sampel", page_icon="📊", layout="wide")
st.title("📊 Uji Nonparametrik — K Sampel (≥ 3 Kelompok)")
st.markdown("""
Aplikasi lengkap untuk **uji nonparametrik k-sampel** — **independen** maupun **berpasangan** —
dilengkapi uji omnibus, post-hoc pairwise, effect size, dan visualisasi interaktif.
""")

# ============================
# HELPERS
# ============================
def _arr(obj):
    return np.asarray(obj).flatten()

def effect_size_r(z, n):
    return abs(z) / np.sqrt(n) if n > 0 else 0

def interpret_effect(r):
    r = abs(r)
    if r < 0.1: return "Sangat Kecil"
    elif r < 0.3: return "Kecil"
    elif r < 0.5: return "Sedang"
    else: return "Besar"

def epsilon_squared(H, n):
    return float(H) / (n - 1) if n > 1 else 0

def kendall_w(chi2, n_subj, k):
    return float(chi2) / (n_subj * (k - 1)) if (n_subj * (k - 1)) > 0 else 0

def dunn_test(groups, group_names, alpha=0.05):
    """Dunn's post-hoc test with Bonferroni correction."""
    all_data = np.concatenate(groups)
    all_ranks = stats.rankdata(all_data)
    N = len(all_data)

    # Assign ranks back to groups
    idx = 0
    group_ranks = []
    for g in groups:
        n_g = len(g)
        group_ranks.append(all_ranks[idx:idx+n_g])
        idx += n_g

    mean_ranks = [np.mean(gr) for gr in group_ranks]
    ns = [len(g) for g in groups]

    # Tie correction
    unique, counts = np.unique(all_ranks, return_counts=True)
    tie_sum = np.sum(counts**3 - counts)
    tie_corr = 1 - tie_sum / (N**3 - N) if (N**3 - N) > 0 else 1

    results = []
    pairs = list(combinations(range(len(groups)), 2))
    n_comp = len(pairs)

    for i, j in pairs:
        diff = abs(mean_ranks[i] - mean_ranks[j])
        se = np.sqrt((N*(N+1)/12 * tie_corr) * (1/ns[i] + 1/ns[j]))
        z = diff / se if se > 0 else 0
        p = 2 * (1 - stats.norm.cdf(abs(z)))
        p_adj = min(p * n_comp, 1.0)  # Bonferroni
        results.append({
            'Perbandingan': f'{group_names[i]} vs {group_names[j]}',
            'Mean Rank 1': round(mean_ranks[i], 2),
            'Mean Rank 2': round(mean_ranks[j], 2),
            'Z-statistic': round(z, 4),
            'p-value (raw)': round(p, 6),
            'p-value (Bonferroni)': round(p_adj, 6),
            f'Signifikan (α={alpha})': 'Ya' if p_adj < alpha else 'Tidak'
        })
    return pd.DataFrame(results)

def conover_test(groups, group_names, H_stat, alpha=0.05):
    """Conover-Iman post-hoc test."""
    all_data = np.concatenate(groups)
    all_ranks = stats.rankdata(all_data)
    N = len(all_data)
    k = len(groups)

    idx = 0
    group_ranks = []
    for g in groups:
        n_g = len(g)
        group_ranks.append(all_ranks[idx:idx+n_g])
        idx += n_g

    mean_ranks = [np.mean(gr) for gr in group_ranks]
    ns = [len(g) for g in groups]

    S2 = (1/(N-1)) * (np.sum(all_ranks**2) - N*(N+1)**2/4)
    denom_factor = S2 * (1 - float(H_stat)/(N-1)) if (N-1) > 0 and float(H_stat) < (N-1) else 1

    results = []
    pairs = list(combinations(range(k), 2))
    n_comp = len(pairs)

    for i, j in pairs:
        diff = abs(mean_ranks[i] - mean_ranks[j])
        se = np.sqrt(denom_factor * (N-1-float(H_stat)) / (N-k) * (1/ns[i] + 1/ns[j])) if (N-k) > 0 else 1
        t_val = diff / se if se > 0 else 0
        df_val = N - k
        p = 2 * (1 - stats.t.cdf(abs(t_val), df_val)) if df_val > 0 else 1
        p_adj = min(p * n_comp, 1.0)
        results.append({
            'Perbandingan': f'{group_names[i]} vs {group_names[j]}',
            'Mean Rank 1': round(mean_ranks[i], 2),
            'Mean Rank 2': round(mean_ranks[j], 2),
            't-statistic': round(t_val, 4),
            'p-value (raw)': round(p, 6),
            'p-value (Bonferroni)': round(p_adj, 6),
            f'Signifikan (α={alpha})': 'Ya' if p_adj < alpha else 'Tidak'
        })
    return pd.DataFrame(results)

def pairwise_wilcoxon(groups, group_names, alpha=0.05):
    """Pairwise Wilcoxon tests for paired groups with Bonferroni correction."""
    pairs = list(combinations(range(len(groups)), 2))
    n_comp = len(pairs)
    results = []
    for i, j in pairs:
        min_n = min(len(groups[i]), len(groups[j]))
        d1, d2 = groups[i][:min_n], groups[j][:min_n]
        diff = d1 - d2
        diff_nz = diff[diff != 0]
        if len(diff_nz) >= 5:
            w_stat, w_p = stats.wilcoxon(diff_nz)
            p_adj = min(w_p * n_comp, 1.0)
            results.append({
                'Perbandingan': f'{group_names[i]} vs {group_names[j]}',
                'W-statistic': round(float(w_stat), 4),
                'p-value (raw)': round(float(w_p), 6),
                'p-value (Bonferroni)': round(p_adj, 6),
                f'Signifikan (α={alpha})': 'Ya' if p_adj < alpha else 'Tidak'
            })
        else:
            results.append({
                'Perbandingan': f'{group_names[i]} vs {group_names[j]}',
                'W-statistic': np.nan, 'p-value (raw)': np.nan,
                'p-value (Bonferroni)': np.nan,
                f'Signifikan (α={alpha})': 'N/A (n < 5)'
            })
    return pd.DataFrame(results)

def nemenyi_friedman(groups, group_names, alpha=0.05):
    """Nemenyi post-hoc for Friedman via pairwise sign differences."""
    k = len(groups)
    n_subj = min(len(g) for g in groups)
    data_matrix = np.column_stack([g[:n_subj] for g in groups])
    ranks = np.array([stats.rankdata(row) for row in data_matrix])
    mean_ranks = np.mean(ranks, axis=0)

    pairs = list(combinations(range(k), 2))
    n_comp = len(pairs)
    q_crit = stats.studentized_range.ppf(1-alpha, k, np.inf) / np.sqrt(2)
    se = np.sqrt(k*(k+1) / (6*n_subj))

    results = []
    for i, j in pairs:
        diff = abs(mean_ranks[i] - mean_ranks[j])
        q = diff / se if se > 0 else 0
        sig = 'Ya' if q > q_crit else 'Tidak'
        results.append({
            'Perbandingan': f'{group_names[i]} vs {group_names[j]}',
            'Mean Rank 1': round(mean_ranks[i], 4),
            'Mean Rank 2': round(mean_ranks[j], 4),
            '|Diff|': round(diff, 4),
            'q-statistic': round(q, 4),
            f'Signifikan (α={alpha})': sig
        })
    return pd.DataFrame(results), mean_ranks

# ============================
# SIDEBAR
# ============================
st.sidebar.header("📁 Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV / Excel", type=["csv", "xlsx", "xls"])
use_demo = st.sidebar.checkbox("Gunakan data demo", value=True if uploaded_file is None else False)

if uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    use_demo = False
elif use_demo:
    np.random.seed(42)
    n_per = 40
    df = pd.DataFrame({
        'Metode_A': np.random.normal(65, 12, n_per),
        'Metode_B': np.random.normal(72, 10, n_per),
        'Metode_C': np.random.normal(68, 15, n_per),
        'Metode_D': np.random.normal(78, 11, n_per),
    })
    # Paired demo (within-subject)
    base = np.random.normal(50, 10, n_per)
    df['Waktu_1'] = base + np.random.normal(0, 3, n_per)
    df['Waktu_2'] = base + np.random.normal(4, 3, n_per)
    df['Waktu_3'] = base + np.random.normal(8, 4, n_per)
    df['Waktu_4'] = base + np.random.normal(6, 3.5, n_per)
    st.sidebar.success("Data demo dimuat (40 obs, 8 variabel: 4 independen + 4 repeated)")
else:
    st.warning("Silakan upload data atau gunakan data demo.")
    st.stop()

# ============================
# 1. EKSPLORASI DATA
# ============================
st.header("1. Eksplorasi Data")
c1, c2, c3 = st.columns(3)
c1.metric("Jumlah Observasi", df.shape[0])
c2.metric("Jumlah Variabel", df.shape[1])
c3.metric("Missing Values", df.isnull().sum().sum())

tab_d1, tab_d2 = st.tabs(["Data", "Statistik Deskriptif"])
with tab_d1:
    st.dataframe(df.head(30), use_container_width=True)
with tab_d2:
    st.dataframe(df.describe().T.round(4), use_container_width=True)

# ============================
# 2. DESAIN & VARIABEL
# ============================
st.header("2. Desain Penelitian & Variabel")
design = st.radio("Pilih jenis desain:", [
    "Independen (K kelompok bebas)",
    "Berpasangan / Repeated Measures (K pengukuran pada subjek sama)"
], horizontal=True)
is_independent = "Independen" in design

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if len(numeric_cols) < 3:
    st.error("Minimal 3 kolom numerik untuk uji K sampel.")
    st.stop()

selected_vars = st.multiselect("Pilih variabel kelompok (min. 3)",
                                numeric_cols, default=numeric_cols[:4] if len(numeric_cols) >= 4 else numeric_cols[:3])
if len(selected_vars) < 3:
    st.warning("Pilih minimal 3 variabel.")
    st.stop()

k = len(selected_vars)
groups = [df[v].dropna().values.astype(float) for v in selected_vars]
group_names = selected_vars
ns = [len(g) for g in groups]
N_total = sum(ns)

if not is_independent:
    min_n = min(ns)
    groups = [g[:min_n] for g in groups]
    ns = [min_n] * k
    N_total = min_n * k

st.info(f"**Desain:** {'Independen' if is_independent else 'Berpasangan'} | "
        f"**K = {k} kelompok** | Ukuran sampel: {', '.join(f'{v}(n={n})' for v, n in zip(selected_vars, ns))}")

# ============================
# 3. STATISTIK DESKRIPTIF
# ============================
st.header("3. Statistik Deskriptif")
desc_rows = []
for v, g in zip(selected_vars, groups):
    desc_rows.append({
        'Kelompok': v, 'N': len(g), 'Mean': np.mean(g), 'Median': np.median(g),
        'Std. Dev': np.std(g, ddof=1), 'Min': np.min(g), 'Max': np.max(g),
        'Q1': np.percentile(g, 25), 'Q3': np.percentile(g, 75),
        'IQR': np.percentile(g, 75) - np.percentile(g, 25),
        'Skewness': stats.skew(g), 'Kurtosis': stats.kurtosis(g)
    })
desc_df = pd.DataFrame(desc_rows).round(4)
st.dataframe(desc_df, use_container_width=True, hide_index=True)

# ============================
# 4. VISUALISASI
# ============================
st.header("4. Visualisasi Data")

melted = pd.DataFrame({
    'Nilai': np.concatenate(groups),
    'Kelompok': np.concatenate([[v]*len(g) for v, g in zip(selected_vars, groups)])
})

tab_v1, tab_v2, tab_v3, tab_v4, tab_v5 = st.tabs([
    "Box Plot", "Violin Plot", "Histogram", "Strip / Swarm", "Mean + CI"])

with tab_v1:
    fig = px.box(melted, x='Kelompok', y='Nilai', color='Kelompok', points='all',
                  title="Box Plot K Sampel")
    st.plotly_chart(fig, use_container_width=True)

with tab_v2:
    fig = px.violin(melted, x='Kelompok', y='Nilai', color='Kelompok', box=True,
                     points='all', title="Violin Plot K Sampel")
    st.plotly_chart(fig, use_container_width=True)

with tab_v3:
    fig = px.histogram(melted, x='Nilai', color='Kelompok', barmode='overlay',
                        nbins=25, opacity=0.6, title="Histogram Overlay", marginal="rug")
    st.plotly_chart(fig, use_container_width=True)

with tab_v4:
    fig = px.strip(melted, x='Kelompok', y='Nilai', color='Kelompok',
                    title="Strip Plot K Sampel")
    st.plotly_chart(fig, use_container_width=True)

with tab_v5:
    means = [np.mean(g) for g in groups]
    sems = [stats.sem(g) for g in groups]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=selected_vars, y=means,
                          error_y=dict(type='data', array=[1.96*s for s in sems], visible=True),
                          marker_color=px.colors.qualitative.Set2[:k]))
    fig.update_layout(title="Mean ± 95% CI", yaxis_title="Nilai", height=450)
    st.plotly_chart(fig, use_container_width=True)

# Paired: line plot
if not is_independent:
    st.subheader("Paired / Repeated Measures Plot")
    fig_pair = go.Figure()
    for i in range(min(30, min(ns))):
        vals = [g[i] for g in groups]
        fig_pair.add_trace(go.Scatter(x=selected_vars, y=vals, mode='lines+markers',
                                       line=dict(width=1), marker=dict(size=4),
                                       opacity=0.3, showlegend=False))
    group_means = [np.mean(g) for g in groups]
    fig_pair.add_trace(go.Scatter(x=selected_vars, y=group_means, mode='lines+markers',
                                   line=dict(color='red', width=4), marker=dict(size=10),
                                   name='Mean'))
    fig_pair.update_layout(title="Profil Repeated Measures (individual + mean)",
                            yaxis_title="Nilai", height=500)
    st.plotly_chart(fig_pair, use_container_width=True)

# ============================
# 5. UJI NORMALITAS
# ============================
st.header("5. Uji Normalitas Setiap Kelompok")
col_a1, col_a2 = st.columns(2)
with col_a1:
    alpha = st.selectbox("Tingkat Signifikansi (α)", [0.01, 0.05, 0.10], index=1)
with col_a2:
    alt_label = st.selectbox("Info: Arah uji omnibus", ["Minimal 1 kelompok berbeda"], disabled=True)

norm_rows = []
for v, g in zip(selected_vars, groups):
    if len(g) <= 5000:
        sw_s, sw_p = stats.shapiro(g)
    else:
        sw_s, sw_p = np.nan, np.nan
    jb_s, jb_p = stats.jarque_bera(g)
    norm_rows.append({'Kelompok': v, 'Shapiro-Wilk Stat': round(sw_s, 6) if not np.isnan(sw_s) else np.nan,
                      'SW p-value': round(sw_p, 6) if not np.isnan(sw_p) else np.nan,
                      'SW Keputusan': ('Normal' if sw_p > alpha else 'Tidak Normal') if not np.isnan(sw_p) else 'N/A',
                      'Jarque-Bera Stat': round(float(jb_s), 6),
                      'JB p-value': round(float(jb_p), 6),
                      'JB Keputusan': 'Normal' if jb_p > alpha else 'Tidak Normal'})
norm_df = pd.DataFrame(norm_rows)
st.dataframe(norm_df, use_container_width=True, hide_index=True)

# QQ Plots
with st.expander("QQ-Plot Setiap Kelompok"):
    ncols = min(4, k)
    nrows = int(np.ceil(k / ncols))
    fig_qq = make_subplots(rows=nrows, cols=ncols,
                            subplot_titles=[v for v in selected_vars])
    for idx, (v, g) in enumerate(zip(selected_vars, groups)):
        r, c = divmod(idx, ncols)
        qq = stats.probplot(g, dist="norm")
        fig_qq.add_trace(go.Scatter(x=qq[0][0], y=qq[0][1], mode='markers',
                                     marker=dict(size=3), showlegend=False), row=r+1, col=c+1)
        fig_qq.add_trace(go.Scatter(x=qq[0][0], y=qq[1][0]*qq[0][0]+qq[1][1],
                                     mode='lines', line=dict(color='red'), showlegend=False), row=r+1, col=c+1)
    fig_qq.update_layout(height=300*nrows, title_text="QQ-Plots")
    st.plotly_chart(fig_qq, use_container_width=True)

# ============================
# 6. UJI HOMOGENITAS
# ============================
st.header("6. Uji Homogenitas Variansi")
lev_s, lev_p = stats.levene(*groups, center='median')
flig_s, flig_p = stats.fligner(*groups)
bart_s, bart_p = stats.bartlett(*groups)

homo_df = pd.DataFrame({
    'Uji': ['Levene (median)', 'Fligner-Killeen', 'Bartlett'],
    'Statistik': [round(lev_s,4), round(flig_s,4), round(bart_s,4)],
    'p-value': [round(lev_p,6), round(flig_p,6), round(bart_p,6)],
    f'Keputusan (α={alpha})': [
        'Homogen' if lev_p > alpha else 'Tidak Homogen',
        'Homogen' if flig_p > alpha else 'Tidak Homogen',
        'Homogen' if bart_p > alpha else 'Tidak Homogen']
})
st.dataframe(homo_df, use_container_width=True, hide_index=True)

# ============================
# COLLECTION OF ALL RESULTS
# ============================
summary_rows = []

if is_independent:
    # ============================
    # 7. KRUSKAL-WALLIS
    # ============================
    st.header("7. Kruskal-Wallis Test")
    st.markdown("Uji omnibus nonparametrik utama untuk **K sampel independen** — ekuivalen nonparametrik One-Way ANOVA.")

    kw_stat, kw_p = stats.kruskal(*groups)
    kw_stat = float(kw_stat); kw_p = float(kw_p)
    eps_sq = epsilon_squared(kw_stat, N_total)
    eta_sq = (kw_stat - k + 1) / (N_total - k) if (N_total - k) > 0 else 0

    # Mean ranks
    all_data = np.concatenate(groups)
    all_ranks = stats.rankdata(all_data)
    idx_r = 0
    mean_rank_list = []
    for g in groups:
        nr = len(g)
        mean_rank_list.append(np.mean(all_ranks[idx_r:idx_r+nr]))
        idx_r += nr

    kw_result = pd.DataFrame({
        'Metrik': ['H-statistic (χ²)', 'df', 'p-value', 'ε² (Epsilon-squared)',
                   'η² (Eta-squared approx)', 'Effect Size Interpretation',
                   f'Keputusan (α={alpha})'],
        'Nilai': [round(kw_stat,4), k-1, round(kw_p,6), round(eps_sq,4),
                  round(eta_sq,4), interpret_effect(eps_sq),
                  'Tolak H₀ (Ada perbedaan signifikan)' if kw_p < alpha else 'Gagal Tolak H₀']
    })
    st.dataframe(kw_result, use_container_width=True, hide_index=True)
    summary_rows.append({'Uji': 'Kruskal-Wallis', 'Statistik': f'H={kw_stat:.4f}',
                         'p-value': round(kw_p,6), 'Effect Size': f'ε²={eps_sq:.4f}',
                         'Keputusan': 'Tolak H₀' if kw_p < alpha else 'Gagal Tolak H₀'})

    # Mean rank bar
    fig_mr = go.Figure()
    fig_mr.add_trace(go.Bar(x=selected_vars, y=mean_rank_list,
                             marker_color=px.colors.qualitative.Set2[:k]))
    fig_mr.update_layout(title="Mean Rank Setiap Kelompok (Kruskal-Wallis)",
                          yaxis_title="Mean Rank", height=400)
    st.plotly_chart(fig_mr, use_container_width=True)

    # ============================
    # 8. MEDIAN TEST
    # ============================
    st.header("8. Mood's Median Test")
    st.markdown("Uji omnibus berdasarkan **tabel kontingensi** — berapa banyak observasi di atas/bawah grand median.")

    try:
        med_stat, med_p, med_med, med_table = stats.median_test(*groups)
        med_result = pd.DataFrame({
            'Metrik': ['Chi-Square', 'p-value', 'Grand Median', f'Keputusan (α={alpha})'],
            'Nilai': [round(float(med_stat),4), round(float(med_p),6), round(float(med_med),4),
                      'Median Berbeda' if med_p < alpha else 'Median Sama']
        })
        st.dataframe(med_result, use_container_width=True, hide_index=True)
        summary_rows.append({'Uji': "Mood's Median Test", 'Statistik': f'χ²={float(med_stat):.4f}',
                             'p-value': round(float(med_p),6), 'Effect Size': '-',
                             'Keputusan': 'Berbeda' if med_p < alpha else 'Sama'})

        cont_df = pd.DataFrame(med_table, columns=selected_vars,
                                index=['> Grand Median', '≤ Grand Median'])
        st.dataframe(cont_df, use_container_width=True)
    except Exception as e:
        st.warning(f"Mood's Median Test gagal: {e}")

    # ============================
    # 9. JONCKHEERE-TERPSTRA
    # ============================
    st.header("9. Jonckheere-Terpstra Trend Test")
    st.markdown("Menguji apakah ada **tren berurutan (ordered)** antar kelompok — lebih powerful jika ada ordering alami.")

    JT_count = 0
    for i in range(k):
        for j in range(i+1, k):
            for xi in groups[i]:
                for xj in groups[j]:
                    if xi < xj: JT_count += 1
                    elif xi == xj: JT_count += 0.5

    # Approximate normal
    prod_n = 1
    sum_n = sum(ns)
    sum_n2 = sum(n**2 for n in ns)
    sum_n3 = sum(n**3 for n in ns)

    E_JT = (N_total**2 - sum_n2) / 4
    var_JT = (N_total**2 * (2*N_total + 3) - sum(n**2 * (2*n + 3) for n in ns)) / 72
    z_jt = (JT_count - E_JT) / np.sqrt(var_JT) if var_JT > 0 else 0
    p_jt = 2 * (1 - stats.norm.cdf(abs(z_jt)))

    jt_result = pd.DataFrame({
        'Metrik': ['JT-statistic', 'E(JT)', 'Z-approx', 'p-value (two-sided)',
                   f'Keputusan (α={alpha})'],
        'Nilai': [round(JT_count, 2), round(E_JT, 2), round(z_jt, 4), round(p_jt, 6),
                  'Ada tren signifikan' if p_jt < alpha else 'Tidak ada tren']
    })
    st.dataframe(jt_result, use_container_width=True, hide_index=True)
    summary_rows.append({'Uji': 'Jonckheere-Terpstra', 'Statistik': f'JT={JT_count:.2f}',
                         'p-value': round(p_jt,6), 'Effect Size': f'Z={z_jt:.4f}',
                         'Keputusan': 'Ada tren' if p_jt < alpha else 'Tidak ada tren'})

    # ============================
    # 10. POST-HOC: DUNN'S TEST
    # ============================
    st.header("10. Post-Hoc: Dunn's Test (Bonferroni)")
    st.markdown("Membandingkan semua pasangan kelompok menggunakan **peringkat gabungan** dari Kruskal-Wallis.")

    dunn_df = dunn_test(groups, selected_vars, alpha)
    st.dataframe(dunn_df, use_container_width=True, hide_index=True)

    # Heatmap p-values
    pval_matrix = np.ones((k, k))
    for _, row in dunn_df.iterrows():
        names = row['Perbandingan'].split(' vs ')
        i_idx = selected_vars.index(names[0])
        j_idx = selected_vars.index(names[1])
        pval_matrix[i_idx, j_idx] = row['p-value (Bonferroni)']
        pval_matrix[j_idx, i_idx] = row['p-value (Bonferroni)']
    np.fill_diagonal(pval_matrix, 1.0)

    fig_hm = go.Figure(data=go.Heatmap(
        z=pval_matrix, x=selected_vars, y=selected_vars,
        colorscale='RdYlGn', zmin=0, zmax=0.1,
        text=np.round(pval_matrix, 4), texttemplate="%{text}"))
    fig_hm.update_layout(title="Dunn's Test: Heatmap p-value (Bonferroni)", height=450)
    st.plotly_chart(fig_hm, use_container_width=True)

    # ============================
    # 11. POST-HOC: CONOVER-IMAN
    # ============================
    st.header("11. Post-Hoc: Conover-Iman Test (Bonferroni)")
    st.markdown("Alternatif post-hoc yang lebih **powerful** dari Dunn's — menggunakan distribusi t.")

    conover_df = conover_test(groups, selected_vars, kw_stat, alpha)
    st.dataframe(conover_df, use_container_width=True, hide_index=True)

    # ============================
    # 12. POST-HOC: PAIRWISE MANN-WHITNEY
    # ============================
    st.header("12. Post-Hoc: Pairwise Mann-Whitney U (Bonferroni)")
    st.markdown("Membandingkan setiap pasangan dengan uji Mann-Whitney U, lalu dikoreksi Bonferroni.")

    pairs = list(combinations(range(k), 2))
    n_comp = len(pairs)
    mw_rows = []
    for i, j in pairs:
        u_s, u_p = stats.mannwhitneyu(groups[i], groups[j], alternative='two-sided')
        p_adj = min(float(u_p) * n_comp, 1.0)
        mw_rows.append({
            'Perbandingan': f'{selected_vars[i]} vs {selected_vars[j]}',
            'U-statistic': round(float(u_s), 2),
            'p-value (raw)': round(float(u_p), 6),
            'p-value (Bonferroni)': round(p_adj, 6),
            f'Signifikan (α={alpha})': 'Ya' if p_adj < alpha else 'Tidak'
        })
    mw_df = pd.DataFrame(mw_rows)
    st.dataframe(mw_df, use_container_width=True, hide_index=True)

    # ============================
    # 13. PERMUTATION TEST (K-SAMPLE)
    # ============================
    st.header("13. Permutation Test (K-Sample)")
    with st.expander("Jalankan Permutation Test", expanded=False):
        n_perm = st.number_input("Jumlah permutasi", min_value=500, max_value=50000, value=5000, step=500)
        if st.button("Run Permutation (Independent)", type="primary"):
            obs_H = kw_stat
            combined = np.concatenate(groups)
            perm_H = np.zeros(int(n_perm))
            for p_i in range(int(n_perm)):
                perm = np.random.permutation(combined)
                perm_groups = []
                idx_p = 0
                for n_g in ns:
                    perm_groups.append(perm[idx_p:idx_p+n_g])
                    idx_p += n_g
                perm_H[p_i], _ = stats.kruskal(*perm_groups)
            perm_p = np.mean(perm_H >= obs_H)
            st.metric("Permutation p-value", f"{perm_p:.6f}")
            st.markdown(f"**Keputusan:** {'Tolak H₀' if perm_p < alpha else 'Gagal Tolak H₀'}")
            summary_rows.append({'Uji': 'Permutation K-Sample', 'Statistik': f'H_obs={obs_H:.4f}',
                                 'p-value': round(perm_p,6), 'Effect Size': '-',
                                 'Keputusan': 'Tolak H₀' if perm_p < alpha else 'Gagal Tolak H₀'})

            fig_pm = go.Figure()
            fig_pm.add_trace(go.Histogram(x=perm_H, nbinsx=50, marker_color='steelblue'))
            fig_pm.add_vline(x=obs_H, line_color="red", annotation_text=f"Obs H={obs_H:.2f}")
            fig_pm.update_layout(title="Permutation Distribution of H", height=400,
                                  xaxis_title="H-statistic", yaxis_title="Frekuensi")
            st.plotly_chart(fig_pm, use_container_width=True)

# ============================
# PAIRED / REPEATED MEASURES
# ============================
else:
    n_subj = min(ns)

    # ============================
    # 7. FRIEDMAN TEST
    # ============================
    st.header("7. Friedman Test")
    st.markdown("Uji omnibus nonparametrik utama untuk **K sampel berpasangan / repeated measures** — ekuivalen nonparametrik Repeated Measures ANOVA.")

    fr_stat, fr_p = stats.friedmanchisquare(*groups)
    fr_stat = float(fr_stat); fr_p = float(fr_p)
    w_kendall = kendall_w(fr_stat, n_subj, k)

    # Ranks
    data_matrix = np.column_stack(groups)
    ranks_matrix = np.array([stats.rankdata(row) for row in data_matrix])
    mean_ranks_fr = np.mean(ranks_matrix, axis=0)

    fr_result = pd.DataFrame({
        'Metrik': ['χ² (Friedman)', 'df', 'p-value', "Kendall's W",
                   'Effect Size Interpretation', f'Keputusan (α={alpha})'],
        'Nilai': [round(fr_stat,4), k-1, round(fr_p,6), round(w_kendall,4),
                  interpret_effect(w_kendall),
                  'Tolak H₀ (Ada perbedaan signifikan)' if fr_p < alpha else 'Gagal Tolak H₀']
    })
    st.dataframe(fr_result, use_container_width=True, hide_index=True)
    summary_rows.append({'Uji': 'Friedman', 'Statistik': f'χ²={fr_stat:.4f}',
                         'p-value': round(fr_p,6), 'Effect Size': f"W={w_kendall:.4f}",
                         'Keputusan': 'Tolak H₀' if fr_p < alpha else 'Gagal Tolak H₀'})

    fig_mr = go.Figure()
    fig_mr.add_trace(go.Bar(x=selected_vars, y=mean_ranks_fr,
                             marker_color=px.colors.qualitative.Set2[:k]))
    fig_mr.update_layout(title="Mean Rank (Friedman)", yaxis_title="Mean Rank", height=400)
    st.plotly_chart(fig_mr, use_container_width=True)

    # ============================
    # 8. COCHRAN'S Q TEST (dichotomized)
    # ============================
    st.header("8. Cochran's Q Test (Dichotomized)")
    st.markdown("Versi **kategorikal** dari Friedman — menguji apakah proporsi 'sukses' berbeda antar kondisi.")

    grand_median = np.median(np.concatenate(groups))
    binary_matrix = (data_matrix > grand_median).astype(int)
    row_sums = binary_matrix.sum(axis=1)
    col_sums = binary_matrix.sum(axis=0)
    T_total = binary_matrix.sum()
    Q_num = (k-1) * (k * np.sum(col_sums**2) - T_total**2)
    Q_den = k * T_total - np.sum(row_sums**2)
    Q_stat = Q_num / Q_den if Q_den > 0 else 0
    Q_p = 1 - stats.chi2.cdf(Q_stat, k-1)

    q_result = pd.DataFrame({
        'Metrik': ["Cochran's Q", 'df', 'p-value', f'Keputusan (α={alpha})'],
        'Nilai': [round(Q_stat,4), k-1, round(Q_p,6),
                  'Proporsi Berbeda' if Q_p < alpha else 'Proporsi Sama']
    })
    st.dataframe(q_result, use_container_width=True, hide_index=True)
    summary_rows.append({'Uji': "Cochran's Q", 'Statistik': f'Q={Q_stat:.4f}',
                         'p-value': round(Q_p,6), 'Effect Size': '-',
                         'Keputusan': 'Berbeda' if Q_p < alpha else 'Sama'})

    # Proportions bar
    props = col_sums / n_subj
    fig_prop = go.Figure()
    fig_prop.add_trace(go.Bar(x=selected_vars, y=props,
                               marker_color=px.colors.qualitative.Set2[:k]))
    fig_prop.update_layout(title="Proporsi > Grand Median per Kelompok",
                            yaxis_title="Proporsi", height=400)
    st.plotly_chart(fig_prop, use_container_width=True)

    # ============================
    # 9. PAGE'S L TREND TEST
    # ============================
    st.header("9. Page's L Trend Test")
    st.markdown("Menguji apakah ada **tren monoton** dalam repeated measures (urutan kelompok bermakna).")

    L_stat = np.sum(mean_ranks_fr * np.arange(1, k+1))
    E_L = n_subj * k * (k+1)**2 / 4
    var_L = n_subj * k**2 * (k+1) * (k**2 - 1) / 144
    z_L = (L_stat - E_L) / np.sqrt(var_L) if var_L > 0 else 0
    p_L = 1 - stats.norm.cdf(z_L)  # one-sided (increasing trend)

    page_result = pd.DataFrame({
        'Metrik': ["Page's L", 'E(L)', 'Z-approx', 'p-value (one-sided, increasing)',
                   f'Keputusan (α={alpha})'],
        'Nilai': [round(L_stat,4), round(E_L,4), round(z_L,4), round(p_L,6),
                  'Ada tren meningkat' if p_L < alpha else 'Tidak ada tren']
    })
    st.dataframe(page_result, use_container_width=True, hide_index=True)
    summary_rows.append({'Uji': "Page's L", 'Statistik': f'L={L_stat:.4f}',
                         'p-value': round(p_L,6), 'Effect Size': f'Z={z_L:.4f}',
                         'Keputusan': 'Ada tren' if p_L < alpha else 'Tidak ada tren'})

    # ============================
    # 10. POST-HOC: NEMENYI
    # ============================
    st.header("10. Post-Hoc: Nemenyi Test")
    st.markdown("Post-hoc standar untuk Friedman — membandingkan semua pasangan berdasarkan **perbedaan mean rank**.")

    nemenyi_df, mean_ranks_nem = nemenyi_friedman(groups, selected_vars, alpha)
    st.dataframe(nemenyi_df, use_container_width=True, hide_index=True)

    # ============================
    # 11. POST-HOC: PAIRWISE WILCOXON
    # ============================
    st.header("11. Post-Hoc: Pairwise Wilcoxon Signed-Rank (Bonferroni)")
    st.markdown("Membandingkan setiap pasangan menggunakan Wilcoxon Signed-Rank, lalu dikoreksi Bonferroni.")

    pw_df = pairwise_wilcoxon(groups, selected_vars, alpha)
    st.dataframe(pw_df, use_container_width=True, hide_index=True)

    # Post-hoc heatmap
    pval_matrix = np.ones((k, k))
    for _, row in pw_df.iterrows():
        names = row['Perbandingan'].split(' vs ')
        i_idx = selected_vars.index(names[0])
        j_idx = selected_vars.index(names[1])
        pv = row['p-value (Bonferroni)']
        if not np.isnan(pv):
            pval_matrix[i_idx, j_idx] = pv
            pval_matrix[j_idx, i_idx] = pv
    np.fill_diagonal(pval_matrix, 1.0)

    fig_hm = go.Figure(data=go.Heatmap(
        z=pval_matrix, x=selected_vars, y=selected_vars,
        colorscale='RdYlGn', zmin=0, zmax=0.1,
        text=np.round(pval_matrix, 4), texttemplate="%{text}"))
    fig_hm.update_layout(title="Pairwise Wilcoxon: Heatmap p-value (Bonferroni)", height=450)
    st.plotly_chart(fig_hm, use_container_width=True)

    # ============================
    # 12. POST-HOC: SIGN TEST PAIRWISE
    # ============================
    st.header("12. Post-Hoc: Pairwise Sign Test (Bonferroni)")
    pairs = list(combinations(range(k), 2))
    n_comp = len(pairs)
    sign_rows = []
    for i, j in pairs:
        min_n_s = min(len(groups[i]), len(groups[j]))
        d = groups[i][:min_n_s] - groups[j][:min_n_s]
        n_p = np.sum(d > 0)
        n_neg = np.sum(d < 0)
        n_eff = n_p + n_neg
        if n_eff > 0:
            sp = 2 * stats.binom.cdf(min(n_p, n_neg), n_eff, 0.5)
            sp = min(sp, 1.0)
            sp_adj = min(sp * n_comp, 1.0)
        else:
            sp, sp_adj = 1.0, 1.0
        sign_rows.append({
            'Perbandingan': f'{selected_vars[i]} vs {selected_vars[j]}',
            'n+': n_p, 'n-': n_neg,
            'p-value (raw)': round(sp, 6),
            'p-value (Bonferroni)': round(sp_adj, 6),
            f'Signifikan (α={alpha})': 'Ya' if sp_adj < alpha else 'Tidak'
        })
    sign_df = pd.DataFrame(sign_rows)
    st.dataframe(sign_df, use_container_width=True, hide_index=True)

    # ============================
    # 13. PERMUTATION (PAIRED K-SAMPLE)
    # ============================
    st.header("13. Permutation Friedman Test")
    with st.expander("Jalankan Permutation Test", expanded=False):
        n_perm = st.number_input("Jumlah permutasi", min_value=500, max_value=50000, value=5000, step=500, key='perm_paired')
        if st.button("Run Permutation (Paired)", type="primary"):
            obs_F = fr_stat
            perm_F = np.zeros(int(n_perm))
            for p_i in range(int(n_perm)):
                perm_matrix = np.copy(data_matrix)
                for row_i in range(n_subj):
                    perm_matrix[row_i] = np.random.permutation(perm_matrix[row_i])
                perm_groups = [perm_matrix[:, col_i] for col_i in range(k)]
                perm_F[p_i], _ = stats.friedmanchisquare(*perm_groups)
            perm_p = np.mean(perm_F >= obs_F)
            st.metric("Permutation p-value", f"{perm_p:.6f}")
            summary_rows.append({'Uji': 'Permutation Friedman', 'Statistik': f'χ²_obs={obs_F:.4f}',
                                 'p-value': round(perm_p,6), 'Effect Size': '-',
                                 'Keputusan': 'Tolak H₀' if perm_p < alpha else 'Gagal Tolak H₀'})

            fig_pf = go.Figure()
            fig_pf.add_trace(go.Histogram(x=perm_F, nbinsx=50, marker_color='steelblue'))
            fig_pf.add_vline(x=obs_F, line_color="red", annotation_text=f"Obs={obs_F:.2f}")
            fig_pf.update_layout(title="Permutation Distribution of Friedman χ²", height=400,
                                  xaxis_title="χ²", yaxis_title="Frekuensi")
            st.plotly_chart(fig_pf, use_container_width=True)

# ============================
# 14. SUMMARY
# ============================
st.header("14. Ringkasan Seluruh Uji")
if len(summary_rows) > 0:
    summary_df = pd.DataFrame(summary_rows)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
else:
    st.info("Jalankan uji-uji di atas terlebih dahulu.")

# ============================
# 15. EXPORT
# ============================
st.header("15. Ekspor Hasil")
col_e1, col_e2, col_e3 = st.columns(3)

with col_e1:
    lines = [
        "=" * 70, "UJI NONPARAMETRIK - K SAMPEL", "=" * 70,
        f"Desain         : {'Independen' if is_independent else 'Berpasangan'}",
        f"K Kelompok     : {k}",
        f"Alpha          : {alpha}",
        f"Variabel       : {', '.join(selected_vars)}",
        f"Ukuran Sampel  : {', '.join(str(n) for n in ns)}",
        "", "=" * 70, "HASIL UJI OMNIBUS & POST-HOC", "=" * 70]
    if len(summary_rows) > 0:
        for row in summary_rows:
            lines.append(f"  {row['Uji']:30s} | {str(row['Statistik']):>15s} | p={row['p-value']:<10} | {row['Keputusan']}")
    st.download_button("Download Summary (TXT)", data="\n".join(lines),
                       file_name="nonparam_ksample_summary.txt", mime="text/plain")

with col_e2:
    st.download_button("Download Data (CSV)", data=df[selected_vars].to_csv(index=False),
                       file_name="nonparam_ksample_data.csv", mime="text/csv")

with col_e3:
    if len(summary_rows) > 0:
        st.download_button("Download Ringkasan (CSV)",
                           data=pd.DataFrame(summary_rows).to_csv(index=False),
                           file_name="nonparam_ksample_tests.csv", mime="text/csv")

# FOOTER
st.markdown("---")
st.markdown("""
**Catatan Metodologis:**

**Sampel Independen:**
- **Kruskal-Wallis**: omnibus rank-based, ekuivalen nonparametrik One-Way ANOVA.
- **Mood's Median Test**: berdasarkan tabel kontingensi 2×K.
- **Jonckheere-Terpstra**: khusus untuk mendeteksi tren berurutan antar kelompok.
- **Post-hoc**: Dunn's Test, Conover-Iman, Pairwise Mann-Whitney (semua Bonferroni-corrected).

**Sampel Berpasangan:**
- **Friedman**: omnibus rank-based, ekuivalen nonparametrik RM-ANOVA.
- **Cochran's Q**: versi dikotomisasi dari Friedman.
- **Page's L**: mendeteksi tren monoton dalam repeated measures.
- **Post-hoc**: Nemenyi, Pairwise Wilcoxon, Pairwise Sign Test (semua Bonferroni-corrected).

**Effect Size:** ε² (epsilon-squared) untuk Kruskal-Wallis, Kendall's W untuk Friedman.
""")
st.markdown("Dibangun dengan **Streamlit** + **SciPy** + **Plotly** | Python")
