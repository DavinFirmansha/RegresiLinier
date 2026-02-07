import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import f as f_dist, t as t_dist, shapiro, levene, bartlett
from scipy.stats import studentized_range
from itertools import combinations
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Analisis Rancangan Percobaan", page_icon="🔬", layout="wide")
st.title("🔬 Analisis Rancangan Percobaan (Experimental Design)")
st.markdown("""
Aplikasi lengkap untuk **analisis rancangan percobaan** — mendukung rancangan dasar hingga lanjutan,  
uji asumsi, **8 uji post-hoc**, visualisasi interaktif, dan ekspor hasil.
""")

# ============================================================
# HELPER: ANOVA COMPUTATIONS
# ============================================================
def anova_crd(df, resp_col, treat_col):
    """Completely Randomized Design (CRD / RAL)"""
    groups = df[treat_col].unique()
    k = len(groups)
    N = len(df)
    grand_mean = df[resp_col].mean()
    # SS
    ss_treat = sum(len(df[df[treat_col]==g]) * (df[df[treat_col]==g][resp_col].mean() - grand_mean)**2 for g in groups)
    ss_total = sum((df[resp_col] - grand_mean)**2)
    ss_error = ss_total - ss_treat
    df_treat = k - 1
    df_error = N - k
    df_total = N - 1
    ms_treat = ss_treat / df_treat if df_treat > 0 else 0
    ms_error = ss_error / df_error if df_error > 0 else 0.001
    f_val = ms_treat / ms_error if ms_error > 0 else 0
    p_val = 1 - f_dist.cdf(f_val, df_treat, df_error)
    kk = max(len(df[df[treat_col]==g]) for g in groups)
    cv = (np.sqrt(ms_error) / grand_mean * 100) if grand_mean != 0 else 0
    table = pd.DataFrame({
        'Source': ['Perlakuan', 'Galat', 'Total'],
        'df': [df_treat, df_error, df_total],
        'SS': [round(ss_treat,4), round(ss_error,4), round(ss_total,4)],
        'MS': [round(ms_treat,4), round(ms_error,4), '-'],
        'F-hitung': [round(f_val,4), '-', '-'],
        'p-value': [round(p_val,6), '-', '-'],
    })
    return table, ms_error, df_error, cv, grand_mean

def anova_rcbd(df, resp_col, treat_col, block_col):
    """Randomized Complete Block Design (RCBD / RAK)"""
    groups = df[treat_col].unique(); blocks = df[block_col].unique()
    k = len(groups); b = len(blocks); N = len(df)
    grand_mean = df[resp_col].mean()
    treat_means = {g: df[df[treat_col]==g][resp_col].mean() for g in groups}
    block_means = {bl: df[df[block_col]==bl][resp_col].mean() for bl in blocks}
    ss_treat = b * sum((treat_means[g] - grand_mean)**2 for g in groups)
    ss_block = k * sum((block_means[bl] - grand_mean)**2 for bl in blocks)
    ss_total = sum((df[resp_col] - grand_mean)**2)
    ss_error = ss_total - ss_treat - ss_block
    ss_error = max(ss_error, 0)
    df_treat = k - 1; df_block = b - 1; df_error = (k-1)*(b-1); df_total = N - 1
    ms_treat = ss_treat/df_treat if df_treat > 0 else 0
    ms_block = ss_block/df_block if df_block > 0 else 0
    ms_error = ss_error/df_error if df_error > 0 else 0.001
    f_treat = ms_treat/ms_error if ms_error > 0 else 0
    f_block = ms_block/ms_error if ms_error > 0 else 0
    p_treat = 1 - f_dist.cdf(f_treat, df_treat, df_error)
    p_block = 1 - f_dist.cdf(f_block, df_block, df_error)
    cv = (np.sqrt(ms_error)/grand_mean*100) if grand_mean != 0 else 0
    re = ((df_error*ms_block + (df_treat+1)*(ms_error)) /
          ((df_block+df_error)*ms_error)) if ms_error > 0 else 1
    table = pd.DataFrame({
        'Source': ['Blok', 'Perlakuan', 'Galat', 'Total'],
        'df': [df_block, df_treat, df_error, df_total],
        'SS': [round(ss_block,4), round(ss_treat,4), round(ss_error,4), round(ss_total,4)],
        'MS': [round(ms_block,4), round(ms_treat,4), round(ms_error,4), '-'],
        'F-hitung': [round(f_block,4), round(f_treat,4), '-', '-'],
        'p-value': [round(p_block,6), round(p_treat,6), '-', '-'],
    })
    return table, ms_error, df_error, cv, grand_mean, re

def anova_latin_square(df, resp_col, treat_col, row_col, col_col):
    """Latin Square Design (RBSL)"""
    treats = df[treat_col].unique(); rows = df[row_col].unique(); cols = df[col_col].unique()
    p = len(treats); N = len(df)
    gm = df[resp_col].mean()
    ss_treat = p * sum((df[df[treat_col]==t][resp_col].mean()-gm)**2 for t in treats)
    ss_row = p * sum((df[df[row_col]==r][resp_col].mean()-gm)**2 for r in rows)
    ss_col = p * sum((df[df[col_col]==c][resp_col].mean()-gm)**2 for c in cols)
    ss_total = sum((df[resp_col]-gm)**2)
    ss_error = ss_total - ss_treat - ss_row - ss_col
    ss_error = max(ss_error, 0)
    df_treat = p-1; df_row = p-1; df_col = p-1; df_error = (p-1)*(p-2); df_total = N-1
    ms_treat = ss_treat/df_treat if df_treat > 0 else 0
    ms_row = ss_row/df_row if df_row > 0 else 0
    ms_col = ss_col/df_col if df_col > 0 else 0
    ms_error = ss_error/df_error if df_error > 0 else 0.001
    f_t = ms_treat/ms_error; f_r = ms_row/ms_error; f_c = ms_col/ms_error
    p_t = 1-f_dist.cdf(f_t, df_treat, df_error)
    p_r = 1-f_dist.cdf(f_r, df_row, df_error)
    p_c = 1-f_dist.cdf(f_c, df_col, df_error)
    cv = (np.sqrt(ms_error)/gm*100) if gm != 0 else 0
    table = pd.DataFrame({
        'Source': ['Baris', 'Kolom', 'Perlakuan', 'Galat', 'Total'],
        'df': [df_row, df_col, df_treat, df_error, df_total],
        'SS': [round(ss_row,4), round(ss_col,4), round(ss_treat,4), round(ss_error,4), round(ss_total,4)],
        'MS': [round(ms_row,4), round(ms_col,4), round(ms_treat,4), round(ms_error,4), '-'],
        'F-hitung': [round(f_r,4), round(f_c,4), round(f_t,4), '-', '-'],
        'p-value': [round(p_r,6), round(p_c,6), round(p_t,6), '-', '-'],
    })
    return table, ms_error, df_error, cv, gm

def anova_factorial(df, resp_col, factor_a, factor_b):
    """Two-Factor Factorial Design"""
    a_levels = sorted(df[factor_a].unique()); b_levels = sorted(df[factor_b].unique())
    a = len(a_levels); b = len(b_levels); N = len(df)
    gm = df[resp_col].mean()
    cell_counts = df.groupby([factor_a, factor_b]).size()
    n_per_cell = cell_counts.min()
    # SS Factor A
    ss_a = sum(len(df[df[factor_a]==ai]) * (df[df[factor_a]==ai][resp_col].mean()-gm)**2 for ai in a_levels)
    # SS Factor B
    ss_b = sum(len(df[df[factor_b]==bi]) * (df[df[factor_b]==bi][resp_col].mean()-gm)**2 for bi in b_levels)
    # SS Cells
    ss_cells = sum(len(df[(df[factor_a]==ai)&(df[factor_b]==bi)]) *
                   (df[(df[factor_a]==ai)&(df[factor_b]==bi)][resp_col].mean()-gm)**2
                   for ai in a_levels for bi in b_levels)
    ss_ab = ss_cells - ss_a - ss_b
    ss_ab = max(ss_ab, 0)
    ss_total = sum((df[resp_col]-gm)**2)
    ss_error = ss_total - ss_cells
    ss_error = max(ss_error, 0)
    df_a = a-1; df_b = b-1; df_ab = (a-1)*(b-1); df_error = N - a*b; df_total = N-1
    ms_a = ss_a/df_a if df_a > 0 else 0
    ms_b = ss_b/df_b if df_b > 0 else 0
    ms_ab = ss_ab/df_ab if df_ab > 0 else 0
    ms_error = ss_error/df_error if df_error > 0 else 0.001
    f_a = ms_a/ms_error; f_b = ms_b/ms_error; f_ab = ms_ab/ms_error
    p_a = 1-f_dist.cdf(f_a, df_a, df_error)
    p_b = 1-f_dist.cdf(f_b, df_b, df_error)
    p_ab = 1-f_dist.cdf(f_ab, df_ab, df_error)
    cv = (np.sqrt(ms_error)/gm*100) if gm != 0 else 0
    table = pd.DataFrame({
        'Source': [f'Faktor A ({factor_a})', f'Faktor B ({factor_b})', 'Interaksi A×B', 'Galat', 'Total'],
        'df': [df_a, df_b, df_ab, df_error, df_total],
        'SS': [round(ss_a,4), round(ss_b,4), round(ss_ab,4), round(ss_error,4), round(ss_total,4)],
        'MS': [round(ms_a,4), round(ms_b,4), round(ms_ab,4), round(ms_error,4), '-'],
        'F-hitung': [round(f_a,4), round(f_b,4), round(f_ab,4), '-', '-'],
        'p-value': [round(p_a,6), round(p_b,6), round(p_ab,6), '-', '-'],
    })
    return table, ms_error, df_error, cv, gm

def anova_split_plot(df, resp_col, main_plot, sub_plot, block_col):
    """Split-Plot Design"""
    a_levels = sorted(df[main_plot].unique()); b_levels = sorted(df[sub_plot].unique())
    blocks = sorted(df[block_col].unique())
    a = len(a_levels); b = len(b_levels); r = len(blocks); N = len(df)
    gm = df[resp_col].mean()
    # Main-plot ANOVA
    ss_block = a*b*sum((df[df[block_col]==bl][resp_col].mean()-gm)**2 for bl in blocks)
    ss_a = r*b*sum((df[df[main_plot]==ai][resp_col].mean()-gm)**2 for ai in a_levels)
    # Whole-plot error = SS(block x A)
    ss_whole_cells = b*sum((df[(df[block_col]==bl)&(df[main_plot]==ai)][resp_col].mean()-gm)**2
                            for bl in blocks for ai in a_levels
                            if len(df[(df[block_col]==bl)&(df[main_plot]==ai)]) > 0)
    ss_ea = ss_whole_cells - ss_block - ss_a
    ss_ea = max(ss_ea, 0)
    # Sub-plot
    ss_b = r*a*sum((df[df[sub_plot]==bi][resp_col].mean()-gm)**2 for bi in b_levels)
    ss_all_cells = r*sum((df[(df[main_plot]==ai)&(df[sub_plot]==bi)][resp_col].mean()-gm)**2
                          for ai in a_levels for bi in b_levels
                          if len(df[(df[main_plot]==ai)&(df[sub_plot]==bi)]) > 0)
    ss_ab = ss_all_cells - ss_a - ss_b
    ss_ab = max(ss_ab, 0)
    ss_total = sum((df[resp_col]-gm)**2)
    ss_eb = ss_total - ss_block - ss_a - ss_ea - ss_b - ss_ab
    ss_eb = max(ss_eb, 0)
    df_block = r-1; df_a = a-1; df_ea = (r-1)*(a-1)
    df_b = b-1; df_ab = (a-1)*(b-1); df_eb = a*(r-1)*(b-1); df_total = N-1
    ms_a = ss_a/df_a if df_a > 0 else 0
    ms_ea = ss_ea/df_ea if df_ea > 0 else 0.001
    ms_b = ss_b/df_b if df_b > 0 else 0
    ms_ab = ss_ab/df_ab if df_ab > 0 else 0
    ms_eb = ss_eb/df_eb if df_eb > 0 else 0.001
    ms_block = ss_block/df_block if df_block > 0 else 0
    f_a = ms_a/ms_ea if ms_ea > 0 else 0
    f_b = ms_b/ms_eb if ms_eb > 0 else 0
    f_ab = ms_ab/ms_eb if ms_eb > 0 else 0
    p_a = 1-f_dist.cdf(f_a, df_a, df_ea)
    p_b = 1-f_dist.cdf(f_b, df_b, df_eb)
    p_ab = 1-f_dist.cdf(f_ab, df_ab, df_eb)
    cv_main = (np.sqrt(ms_ea)/gm*100) if gm != 0 else 0
    cv_sub = (np.sqrt(ms_eb)/gm*100) if gm != 0 else 0
    table = pd.DataFrame({
        'Source': ['Blok', f'Main-Plot ({main_plot})', 'Galat (a)', f'Sub-Plot ({sub_plot})',
                   f'Interaksi ({main_plot}×{sub_plot})', 'Galat (b)', 'Total'],
        'df': [df_block, df_a, df_ea, df_b, df_ab, df_eb, df_total],
        'SS': [round(ss_block,4), round(ss_a,4), round(ss_ea,4), round(ss_b,4),
               round(ss_ab,4), round(ss_eb,4), round(ss_total,4)],
        'MS': [round(ms_block,4), round(ms_a,4), round(ms_ea,4), round(ms_b,4),
               round(ms_ab,4), round(ms_eb,4), '-'],
        'F-hitung': ['-', round(f_a,4), '-', round(f_b,4), round(f_ab,4), '-', '-'],
        'p-value': ['-', round(p_a,6), '-', round(p_b,6), round(p_ab,6), '-', '-'],
    })
    return table, ms_ea, ms_eb, df_ea, df_eb, cv_main, cv_sub, gm

def anova_nested(df, resp_col, factor_a, factor_b_nested):
    """Nested Design: B nested in A"""
    a_levels = sorted(df[factor_a].unique())
    a = len(a_levels); N = len(df)
    gm = df[resp_col].mean()
    # SS A
    ss_a = sum(len(df[df[factor_a]==ai])*(df[df[factor_a]==ai][resp_col].mean()-gm)**2 for ai in a_levels)
    # SS B(A)
    ss_ba = 0; total_b_levels = 0
    for ai in a_levels:
        sub = df[df[factor_a]==ai]
        b_in_a = sub[factor_b_nested].unique()
        total_b_levels += len(b_in_a)
        mean_a = sub[resp_col].mean()
        for bi in b_in_a:
            sub_b = sub[sub[factor_b_nested]==bi]
            ss_ba += len(sub_b) * (sub_b[resp_col].mean()-mean_a)**2
    ss_total = sum((df[resp_col]-gm)**2)
    ss_error = ss_total - ss_a - ss_ba
    ss_error = max(ss_error, 0)
    df_a = a-1; df_ba = total_b_levels - a; df_error = N - total_b_levels; df_total = N-1
    ms_a = ss_a/df_a if df_a > 0 else 0
    ms_ba = ss_ba/df_ba if df_ba > 0 else 0.001
    ms_error = ss_error/df_error if df_error > 0 else 0.001
    f_a = ms_a/ms_ba if ms_ba > 0 else 0
    f_ba = ms_ba/ms_error if ms_error > 0 else 0
    p_a = 1-f_dist.cdf(f_a, df_a, df_ba)
    p_ba = 1-f_dist.cdf(f_ba, df_ba, df_error)
    cv = (np.sqrt(ms_error)/gm*100) if gm != 0 else 0
    table = pd.DataFrame({
        'Source': [f'Faktor A ({factor_a})', f'B({factor_b_nested}) nested in A', 'Galat', 'Total'],
        'df': [df_a, df_ba, df_error, df_total],
        'SS': [round(ss_a,4), round(ss_ba,4), round(ss_error,4), round(ss_total,4)],
        'MS': [round(ms_a,4), round(ms_ba,4), round(ms_error,4), '-'],
        'F-hitung': [round(f_a,4), round(f_ba,4), '-', '-'],
        'p-value': [round(p_a,6), round(p_ba,6), '-', '-'],
    })
    return table, ms_error, df_error, cv, gm

def anova_strip_plot(df, resp_col, horiz_factor, vert_factor, block_col):
    """Strip-Plot (Strip-Split) Design"""
    a_levels = sorted(df[horiz_factor].unique()); b_levels = sorted(df[vert_factor].unique())
    blocks = sorted(df[block_col].unique())
    a = len(a_levels); b = len(b_levels); r = len(blocks); N = len(df)
    gm = df[resp_col].mean()
    ss_block = a*b*sum((df[df[block_col]==bl][resp_col].mean()-gm)**2 for bl in blocks)
    ss_a = r*b*sum((df[df[horiz_factor]==ai][resp_col].mean()-gm)**2 for ai in a_levels)
    ss_b = r*a*sum((df[df[vert_factor]==bi][resp_col].mean()-gm)**2 for bi in b_levels)
    # Error for A: Block x A
    ss_ra = b*sum((df[(df[block_col]==bl)&(df[horiz_factor]==ai)][resp_col].mean()-gm)**2
                   for bl in blocks for ai in a_levels
                   if len(df[(df[block_col]==bl)&(df[horiz_factor]==ai)])>0)
    ss_ea = ss_ra - ss_block - ss_a; ss_ea = max(ss_ea, 0)
    # Error for B: Block x B
    ss_rb = a*sum((df[(df[block_col]==bl)&(df[vert_factor]==bi)][resp_col].mean()-gm)**2
                   for bl in blocks for bi in b_levels
                   if len(df[(df[block_col]==bl)&(df[vert_factor]==bi)])>0)
    ss_eb = ss_rb - ss_block - ss_b; ss_eb = max(ss_eb, 0)
    ss_all = sum((df[(df[horiz_factor]==ai)&(df[vert_factor]==bi)][resp_col].mean()-gm)**2 * r
                  for ai in a_levels for bi in b_levels
                  if len(df[(df[horiz_factor]==ai)&(df[vert_factor]==bi)])>0)
    ss_ab = ss_all - ss_a - ss_b; ss_ab = max(ss_ab, 0)
    ss_total = sum((df[resp_col]-gm)**2)
    ss_eab = ss_total - ss_block - ss_a - ss_ea - ss_b - ss_eb - ss_ab; ss_eab = max(ss_eab, 0)
    df_bl = r-1; df_a = a-1; df_ea = (r-1)*(a-1)
    df_b = b-1; df_eb = (r-1)*(b-1); df_ab = (a-1)*(b-1)
    df_eab = (a-1)*(b-1)*(r-1); df_total = N-1
    ms_a = ss_a/df_a if df_a>0 else 0; ms_ea = ss_ea/df_ea if df_ea>0 else 0.001
    ms_b = ss_b/df_b if df_b>0 else 0; ms_eb = ss_eb/df_eb if df_eb>0 else 0.001
    ms_ab = ss_ab/df_ab if df_ab>0 else 0; ms_eab = ss_eab/df_eab if df_eab>0 else 0.001
    f_a = ms_a/ms_ea; f_b = ms_b/ms_eb; f_ab = ms_ab/ms_eab
    p_a = 1-f_dist.cdf(f_a, df_a, df_ea)
    p_b = 1-f_dist.cdf(f_b, df_b, df_eb)
    p_ab = 1-f_dist.cdf(f_ab, df_ab, df_eab)
    table = pd.DataFrame({
        'Source': ['Blok', f'Horiz ({horiz_factor})', 'Galat (a)',
                   f'Vert ({vert_factor})', 'Galat (b)',
                   f'Interaksi ({horiz_factor}×{vert_factor})', 'Galat (ab)', 'Total'],
        'df': [df_bl, df_a, df_ea, df_b, df_eb, df_ab, df_eab, df_total],
        'SS': [round(ss_block,4), round(ss_a,4), round(ss_ea,4), round(ss_b,4),
               round(ss_eb,4), round(ss_ab,4), round(ss_eab,4), round(ss_total,4)],
        'MS': ['-', round(ms_a,4), round(ms_ea,4), round(ms_b,4),
               round(ms_eb,4), round(ms_ab,4), round(ms_eab,4), '-'],
        'F-hitung': ['-', round(f_a,4), '-', round(f_b,4), '-', round(f_ab,4), '-', '-'],
        'p-value': ['-', round(p_a,6), '-', round(p_b,6), '-', round(p_ab,6), '-', '-'],
    })
    cv = (np.sqrt(ms_eab)/gm*100) if gm!=0 else 0
    return table, ms_eab, df_eab, cv, gm

def anova_augmented(df, resp_col, treat_col, block_col, check_col):
    """Augmented Design: checks replicated, new treatments unreplicated"""
    checks = df[df[check_col]==1]
    news = df[df[check_col]==0]
    blocks = df[block_col].unique()
    b = len(blocks); N = len(df)
    gm = df[resp_col].mean()
    c_treats = checks[treat_col].unique(); nc = len(c_treats)
    n_treats = news[treat_col].unique(); nn = len(n_treats)
    # Block SS from checks
    ss_block = nc*sum((checks[checks[block_col]==bl][resp_col].mean()-checks[resp_col].mean())**2
                       for bl in blocks if len(checks[checks[block_col]==bl])>0)
    # Adjusted treatment SS
    all_treats = df[treat_col].unique(); k = len(all_treats)
    ss_treat_unadj = sum(len(df[df[treat_col]==t])*(df[df[treat_col]==t][resp_col].mean()-gm)**2 for t in all_treats)
    ss_total = sum((df[resp_col]-gm)**2)
    ss_error = ss_total - ss_block - ss_treat_unadj
    ss_error = max(ss_error, 1e-6)
    df_block = b-1; df_treat = k-1; df_error = N-k-b+1; df_total = N-1
    df_error = max(df_error, 1)
    ms_treat = ss_treat_unadj/df_treat if df_treat>0 else 0
    ms_error = ss_error/df_error if df_error>0 else 0.001
    f_treat = ms_treat/ms_error
    p_treat = 1-f_dist.cdf(f_treat, df_treat, df_error)
    table = pd.DataFrame({
        'Source': ['Blok', 'Perlakuan (adj)', 'Galat', 'Total'],
        'df': [df_block, df_treat, df_error, df_total],
        'SS': [round(ss_block,4), round(ss_treat_unadj,4), round(ss_error,4), round(ss_total,4)],
        'MS': ['-', round(ms_treat,4), round(ms_error,4), '-'],
        'F-hitung': ['-', round(f_treat,4), '-', '-'],
        'p-value': ['-', round(p_treat,6), '-', '-'],
    })
    cv = (np.sqrt(ms_error)/gm*100) if gm!=0 else 0
    return table, ms_error, df_error, cv, gm

# ============================================================
# POST-HOC TESTS
# ============================================================
def posthoc_tukey(df, resp_col, treat_col, mse, dfe, alpha=0.05):
    groups = sorted(df[treat_col].unique()); k = len(groups)
    means = {g: df[df[treat_col]==g][resp_col].mean() for g in groups}
    ns = {g: len(df[df[treat_col]==g]) for g in groups}
    rows = []
    for g1, g2 in combinations(groups, 2):
        diff = means[g1] - means[g2]
        n_harm = 2 / (1/ns[g1] + 1/ns[g2])
        se = np.sqrt(mse / n_harm)
        q_val = abs(diff) / (np.sqrt(mse / n_harm)) * np.sqrt(2) / np.sqrt(2)
        q_val_raw = abs(diff) / se
        try:
            q_crit = studentized_range.ppf(1-alpha, k, dfe)
        except:
            q_crit = t_dist.ppf(1-alpha/(2*k*(k-1)/2), dfe) * np.sqrt(2)
        sig = 'Ya' if q_val_raw > q_crit else 'Tidak'
        rows.append({'Perbandingan': f'{g1} vs {g2}', 'Selisih': round(diff,4),
                     'SE': round(se,4), 'q-hitung': round(q_val_raw,4),
                     'q-tabel': round(q_crit,4), f'Signifikan (α={alpha})': sig})
    return pd.DataFrame(rows)

def posthoc_lsd(df, resp_col, treat_col, mse, dfe, alpha=0.05):
    groups = sorted(df[treat_col].unique())
    means = {g: df[df[treat_col]==g][resp_col].mean() for g in groups}
    ns = {g: len(df[df[treat_col]==g]) for g in groups}
    t_crit = t_dist.ppf(1-alpha/2, dfe)
    rows = []
    for g1, g2 in combinations(groups, 2):
        diff = means[g1] - means[g2]
        se = np.sqrt(mse * (1/ns[g1] + 1/ns[g2]))
        lsd_val = t_crit * se
        t_val = abs(diff) / se
        p_val = 2 * (1 - t_dist.cdf(t_val, dfe))
        sig = 'Ya' if abs(diff) > lsd_val else 'Tidak'
        rows.append({'Perbandingan': f'{g1} vs {g2}', 'Selisih': round(diff,4),
                     'SE': round(se,4), 't-hitung': round(t_val,4),
                     'LSD': round(lsd_val,4), 'p-value': round(p_val,6),
                     f'Signifikan (α={alpha})': sig})
    return pd.DataFrame(rows)

def posthoc_duncan(df, resp_col, treat_col, mse, dfe, alpha=0.05):
    groups = sorted(df[treat_col].unique()); k = len(groups)
    means = {g: df[df[treat_col]==g][resp_col].mean() for g in groups}
    ns = {g: len(df[df[treat_col]==g]) for g in groups}
    sorted_groups = sorted(groups, key=lambda g: means[g], reverse=True)
    rows = []
    for g1, g2 in combinations(sorted_groups, 2):
        diff = abs(means[g1] - means[g2])
        idx1 = sorted_groups.index(g1); idx2 = sorted_groups.index(g2)
        p_range = abs(idx2 - idx1) + 1
        n_harm = 2 / (1/ns[g1] + 1/ns[g2])
        se = np.sqrt(mse / n_harm)
        try:
            q_crit = studentized_range.ppf(1-alpha, p_range, dfe)
        except:
            q_crit = t_dist.ppf(1-alpha/2, dfe) * np.sqrt(2) * (1 + 0.05*(p_range-2))
        rp = q_crit * se
        sig = 'Ya' if diff > rp else 'Tidak'
        rows.append({'Perbandingan': f'{g1} vs {g2}', 'Range (p)': p_range,
                     '|Selisih|': round(diff,4), 'SE': round(se,4),
                     'Rp (Duncan)': round(rp,4), f'Signifikan (α={alpha})': sig})
    return pd.DataFrame(rows)

def posthoc_bonferroni(df, resp_col, treat_col, mse, dfe, alpha=0.05):
    groups = sorted(df[treat_col].unique()); k = len(groups)
    means = {g: df[df[treat_col]==g][resp_col].mean() for g in groups}
    ns = {g: len(df[df[treat_col]==g]) for g in groups}
    m = k*(k-1)/2
    alpha_adj = alpha / m
    t_crit = t_dist.ppf(1-alpha_adj/2, dfe)
    rows = []
    for g1, g2 in combinations(groups, 2):
        diff = means[g1] - means[g2]
        se = np.sqrt(mse*(1/ns[g1]+1/ns[g2]))
        t_val = abs(diff)/se
        p_val = min(2*(1-t_dist.cdf(t_val, dfe))*m, 1.0)
        sig = 'Ya' if p_val < alpha else 'Tidak'
        rows.append({'Perbandingan': f'{g1} vs {g2}', 'Selisih': round(diff,4),
                     'SE': round(se,4), 't-hitung': round(t_val,4),
                     'p-adj (Bonf)': round(p_val,6), f'Signifikan (α={alpha})': sig})
    return pd.DataFrame(rows)

def posthoc_scheffe(df, resp_col, treat_col, mse, dfe, alpha=0.05):
    groups = sorted(df[treat_col].unique()); k = len(groups)
    means = {g: df[df[treat_col]==g][resp_col].mean() for g in groups}
    ns = {g: len(df[df[treat_col]==g]) for g in groups}
    f_crit = f_dist.ppf(1-alpha, k-1, dfe)
    rows = []
    for g1, g2 in combinations(groups, 2):
        diff = means[g1] - means[g2]
        se = np.sqrt(mse*(1/ns[g1]+1/ns[g2]))
        f_val = diff**2 / (mse*(1/ns[g1]+1/ns[g2])*(k-1))
        scheffe_crit = np.sqrt((k-1)*f_crit) * se
        sig = 'Ya' if abs(diff) > scheffe_crit else 'Tidak'
        rows.append({'Perbandingan': f'{g1} vs {g2}', 'Selisih': round(diff,4),
                     'SE': round(se,4), 'Scheffé Crit': round(scheffe_crit,4),
                     f'Signifikan (α={alpha})': sig})
    return pd.DataFrame(rows)

def posthoc_dunnett(df, resp_col, treat_col, mse, dfe, control_group, alpha=0.05):
    groups = sorted(df[treat_col].unique())
    means = {g: df[df[treat_col]==g][resp_col].mean() for g in groups}
    ns = {g: len(df[df[treat_col]==g]) for g in groups}
    k = len(groups) - 1
    t_crit = t_dist.ppf(1-alpha/2, dfe) * (1 + 0.02*k)  # approx Dunnett
    rows = []
    for g in groups:
        if g == control_group: continue
        diff = means[g] - means[control_group]
        se = np.sqrt(mse*(1/ns[g]+1/ns[control_group]))
        t_val = abs(diff)/se
        p_val = 2*(1-t_dist.cdf(t_val, dfe))
        sig = 'Ya' if t_val > t_crit else 'Tidak'
        rows.append({'Perlakuan vs Kontrol': f'{g} vs {control_group}',
                     'Selisih': round(diff,4), 'SE': round(se,4),
                     't-hitung': round(t_val,4), 't-Dunnett(≈)': round(t_crit,4),
                     'p-value': round(p_val,6), f'Signifikan (α={alpha})': sig})
    return pd.DataFrame(rows)

def posthoc_snk(df, resp_col, treat_col, mse, dfe, alpha=0.05):
    """Student-Newman-Keuls (SNK) test"""
    groups = sorted(df[treat_col].unique()); k = len(groups)
    means = {g: df[df[treat_col]==g][resp_col].mean() for g in groups}
    ns = {g: len(df[df[treat_col]==g]) for g in groups}
    sorted_groups = sorted(groups, key=lambda g: means[g], reverse=True)
    rows = []
    for g1, g2 in combinations(sorted_groups, 2):
        diff = abs(means[g1] - means[g2])
        idx1 = sorted_groups.index(g1); idx2 = sorted_groups.index(g2)
        p_range = abs(idx2-idx1)+1
        n_harm = 2/(1/ns[g1]+1/ns[g2])
        se = np.sqrt(mse/n_harm)
        try:
            q_crit = studentized_range.ppf(1-alpha, p_range, dfe)
        except:
            q_crit = t_dist.ppf(1-alpha/2, dfe)*np.sqrt(2)*(1+0.03*(p_range-2))
        crit_val = q_crit * se
        sig = 'Ya' if diff > crit_val else 'Tidak'
        rows.append({'Perbandingan': f'{g1} vs {g2}', 'Range (p)': p_range,
                     '|Selisih|': round(diff,4), 'Crit Value': round(crit_val,4),
                     f'Signifikan (α={alpha})': sig})
    return pd.DataFrame(rows)

def posthoc_games_howell(df, resp_col, treat_col, alpha=0.05):
    """Games-Howell (no equal variance assumption)"""
    groups = sorted(df[treat_col].unique())
    means = {g: df[df[treat_col]==g][resp_col].mean() for g in groups}
    ns = {g: len(df[df[treat_col]==g]) for g in groups}
    vars_ = {g: df[df[treat_col]==g][resp_col].var(ddof=1) for g in groups}
    rows = []
    for g1, g2 in combinations(groups, 2):
        diff = means[g1] - means[g2]
        se = np.sqrt(vars_[g1]/ns[g1] + vars_[g2]/ns[g2])
        t_val = abs(diff)/se if se > 0 else 0
        # Welch df
        num = (vars_[g1]/ns[g1] + vars_[g2]/ns[g2])**2
        den = (vars_[g1]/ns[g1])**2/(ns[g1]-1) + (vars_[g2]/ns[g2])**2/(ns[g2]-1)
        df_w = num/den if den > 0 else 1
        p_val = 2*(1-t_dist.cdf(t_val, df_w))
        k = len(groups)
        p_adj = min(p_val * k*(k-1)/2, 1.0)
        sig = 'Ya' if p_adj < alpha else 'Tidak'
        rows.append({'Perbandingan': f'{g1} vs {g2}', 'Selisih': round(diff,4),
                     'SE': round(se,4), 't-hitung': round(t_val,4),
                     'df (Welch)': round(df_w,2), 'p-adj': round(p_adj,6),
                     f'Signifikan (α={alpha})': sig})
    return pd.DataFrame(rows)

def grouping_letters(df, resp_col, treat_col, posthoc_result, alpha=0.05):
    """Assign letter grouping from post-hoc results"""
    groups = sorted(df[treat_col].unique())
    means = {g: df[df[treat_col]==g][resp_col].mean() for g in groups}
    sorted_groups = sorted(groups, key=lambda g: means[g], reverse=True)
    sig_col = [c for c in posthoc_result.columns if 'Signifikan' in c]
    if not sig_col: return pd.DataFrame()
    sig_col = sig_col[0]
    not_sig_pairs = set()
    for _, row in posthoc_result.iterrows():
        comp = row.get('Perbandingan', '')
        if ' vs ' in comp:
            parts = comp.split(' vs ')
            if row[sig_col] == 'Tidak':
                not_sig_pairs.add((parts[0].strip(), parts[1].strip()))
                not_sig_pairs.add((parts[1].strip(), parts[0].strip()))
    letters = {}
    current_letter = ord('a')
    assigned = {g: [] for g in sorted_groups}
    for i, g1 in enumerate(sorted_groups):
        if not assigned[g1]:
            letter = chr(current_letter)
            assigned[g1].append(letter)
            for j in range(i+1, len(sorted_groups)):
                g2 = sorted_groups[j]
                all_nonsig = True
                for prev_g in sorted_groups[i:j]:
                    if assigned[prev_g] and letter in assigned[prev_g]:
                        if (prev_g, g2) not in not_sig_pairs:
                            all_nonsig = False; break
                if all_nonsig:
                    assigned[g2].append(letter)
            current_letter += 1
    result = []
    for g in sorted_groups:
        result.append({'Perlakuan': g, 'Mean': round(means[g],4),
                       'Grup': ''.join(assigned[g]) if assigned[g] else chr(current_letter)})
    return pd.DataFrame(result)

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.header("📁 Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV / Excel", type=["csv","xlsx","xls"])
use_demo = st.sidebar.checkbox("Gunakan data demo", value=True if uploaded_file is None else False)

DESIGNS = {
    'RAL / CRD (Rancangan Acak Lengkap)': 'crd',
    'RAK / RCBD (Rancangan Acak Kelompok)': 'rcbd',
    'RBSL / Latin Square': 'latin',
    'Faktorial (2 Faktor)': 'factorial',
    'Split-Plot': 'split_plot',
    'Nested (Tersarang)': 'nested',
    'Strip-Plot (Strip-Split)': 'strip_plot',
    'Augmented Design': 'augmented',
}
design_label = st.sidebar.selectbox("🔧 Rancangan Percobaan", list(DESIGNS.keys()))
design = DESIGNS[design_label]

if uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'): df = pd.read_csv(uploaded_file)
    else: df = pd.read_excel(uploaded_file)
    use_demo = False
elif use_demo:
    np.random.seed(42)
    if design == 'crd':
        treats = np.repeat(['A','B','C','D'], 6)
        resp = np.concatenate([np.random.normal(m, 2, 6) for m in [25, 28, 22, 30]]).round(2)
        df = pd.DataFrame({'Perlakuan': treats, 'Hasil': resp})
    elif design == 'rcbd':
        treats = np.tile(['A','B','C','D'], 5)
        blocks = np.repeat(['Blok1','Blok2','Blok3','Blok4','Blok5'], 4)
        base = np.array([25,28,22,30]*5) + np.repeat([0,2,-1,3,1], 4)
        resp = (base + np.random.normal(0, 1.5, 20)).round(2)
        df = pd.DataFrame({'Perlakuan': treats, 'Blok': blocks, 'Hasil': resp})
    elif design == 'latin':
        ts = ['A','B','C','D']
        rows_ls = ['R1']*4 + ['R2']*4 + ['R3']*4 + ['R4']*4
        cols_ls = ['C1','C2','C3','C4']*4
        t_assign = ['A','B','C','D','B','C','D','A','C','D','A','B','D','A','B','C']
        resp = (np.array([25,28,22,30,28,22,30,25,22,30,25,28,30,25,28,22]) + np.random.normal(0,1.5,16)).round(2)
        df = pd.DataFrame({'Baris': rows_ls, 'Kolom': cols_ls, 'Perlakuan': t_assign, 'Hasil': resp})
    elif design == 'factorial':
        fa = np.repeat(['Pupuk_A','Pupuk_B','Pupuk_C'], 12)
        fb = np.tile(np.repeat(['Dosis_1','Dosis_2'], 6), 3)
        base = {'Pupuk_A': {'Dosis_1': 20, 'Dosis_2': 25}, 'Pupuk_B': {'Dosis_1': 22, 'Dosis_2': 28},
                'Pupuk_C': {'Dosis_1': 18, 'Dosis_2': 32}}
        resp = np.array([base[a][b] + np.random.normal(0,2) for a,b in zip(fa,fb)]).round(2)
        df = pd.DataFrame({'Pupuk': fa, 'Dosis': fb, 'Hasil': resp})
    elif design == 'split_plot':
        blocks = np.repeat(['B1','B2','B3'], 8)
        mp = np.tile(np.repeat(['Irigasi_1','Irigasi_2'], 4), 3)
        sp = np.tile(['Var_A','Var_B','Var_C','Var_D'], 6)
        base = np.array([20,22,25,28]*6) + np.repeat([0,2,1],8) + np.tile(np.repeat([0,3],4),3)
        resp = (base + np.random.normal(0,1.5,24)).round(2)
        df = pd.DataFrame({'Blok': blocks, 'Irigasi': mp, 'Varietas': sp, 'Hasil': resp})
    elif design == 'nested':
        fa = np.repeat(['Lab_A','Lab_B','Lab_C'], 12)
        fb = np.repeat(['T1','T2','T3','T4','T1','T2','T3','T4','T1','T2','T3','T4'], 3)
        fb_full = []
        for i, a in enumerate(fa):
            lab = a[-1]
            fb_full.append(f"T{(i%4)+1}_{lab}")
        resp = (np.array([50]*36) + np.repeat([0,5,-3],12) + np.random.normal(0,2,36)).round(2)
        df = pd.DataFrame({'Lab': fa, 'Teknisi': fb_full, 'Hasil': resp})
    elif design == 'strip_plot':
        blocks = np.repeat(['B1','B2','B3'], 8)
        hf = np.tile(np.repeat(['Jarak_1','Jarak_2'], 4), 3)
        vf = np.tile(['Var_A','Var_B','Var_C','Var_D'], 6)
        base = np.array([20,22,25,28]*6) + np.repeat([0,2,1],8) + np.tile(np.repeat([0,3],4),3)
        resp = (base + np.random.normal(0,1.5,24)).round(2)
        df = pd.DataFrame({'Blok': blocks, 'Jarak_Tanam': hf, 'Varietas': vf, 'Hasil': resp})
    elif design == 'augmented':
        checks = np.tile(['Check1','Check2'], 6)
        blocks_c = np.repeat(['B1','B2','B3'], 4)
        new_treats = [f'New_{i}' for i in range(1,10)]
        blocks_n = np.random.choice(['B1','B2','B3'], 9)
        treats_all = np.concatenate([checks, new_treats])
        blocks_all = np.concatenate([blocks_c, blocks_n])
        is_check = np.array([1]*12 + [0]*9)
        resp = (np.concatenate([np.random.normal(30,2,12), np.random.normal(28,3,9)])).round(2)
        df = pd.DataFrame({'Perlakuan': treats_all, 'Blok': blocks_all, 'Check': is_check, 'Hasil': resp})
    st.sidebar.success(f"Demo: {design_label} — {len(df)} observasi")
else:
    st.warning("Upload data atau gunakan data demo."); st.stop()

# ============================================================
# 1. EKSPLORASI
# ============================================================
st.header("1. Eksplorasi Data")
c1, c2 = st.columns(2)
c1.metric("N Observasi", df.shape[0]); c2.metric("Variabel", df.shape[1])
tab_d1, tab_d2 = st.tabs(["Data", "Deskriptif"])
with tab_d1: st.dataframe(df, use_container_width=True)
with tab_d2: st.dataframe(df.describe(include='all').T.round(4), use_container_width=True)

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(include=['object','category']).columns.tolist()
all_cols = numeric_cols + cat_cols

# ============================================================
# 2. SETUP VARIABEL
# ============================================================
st.header(f"2. Setup: {design_label}")

resp_col = st.selectbox("📊 Variabel Respon (Y)", numeric_cols, index=len(numeric_cols)-1 if numeric_cols else 0)
alpha = st.selectbox("α (signifikansi)", [0.01, 0.05, 0.10], index=1)

if design == 'crd':
    treat_col = st.selectbox("Variabel Perlakuan", [c for c in all_cols if c != resp_col])
elif design == 'rcbd':
    col_a, col_b = st.columns(2)
    with col_a: treat_col = st.selectbox("Variabel Perlakuan", [c for c in all_cols if c != resp_col])
    with col_b: block_col = st.selectbox("Variabel Blok", [c for c in all_cols if c != resp_col and c != treat_col])
elif design == 'latin':
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a: treat_col = st.selectbox("Perlakuan", [c for c in all_cols if c != resp_col])
    with col_b: row_col = st.selectbox("Baris", [c for c in all_cols if c != resp_col and c != treat_col])
    with col_c: col_col = st.selectbox("Kolom", [c for c in all_cols if c != resp_col and c != treat_col])
elif design == 'factorial':
    col_a, col_b = st.columns(2)
    with col_a: factor_a = st.selectbox("Faktor A", [c for c in all_cols if c != resp_col])
    with col_b: factor_b = st.selectbox("Faktor B", [c for c in all_cols if c != resp_col and c != factor_a])
    treat_col = factor_a  # primary for post-hoc
elif design == 'split_plot':
    col_a, col_b, col_c = st.columns(3)
    with col_a: block_col = st.selectbox("Blok", [c for c in all_cols if c != resp_col])
    with col_b: main_plot = st.selectbox("Main-Plot (Petak Utama)", [c for c in all_cols if c != resp_col and c != block_col])
    with col_c: sub_plot = st.selectbox("Sub-Plot (Anak Petak)", [c for c in all_cols if c != resp_col and c != block_col and c != main_plot])
    treat_col = sub_plot
elif design == 'nested':
    col_a, col_b = st.columns(2)
    with col_a: factor_a = st.selectbox("Faktor A (utama)", [c for c in all_cols if c != resp_col])
    with col_b: factor_b_nested = st.selectbox("Faktor B (nested in A)", [c for c in all_cols if c != resp_col and c != factor_a])
    treat_col = factor_a
elif design == 'strip_plot':
    col_a, col_b, col_c = st.columns(3)
    with col_a: block_col = st.selectbox("Blok", [c for c in all_cols if c != resp_col])
    with col_b: horiz_factor = st.selectbox("Faktor Horizontal", [c for c in all_cols if c != resp_col and c != block_col])
    with col_c: vert_factor = st.selectbox("Faktor Vertikal", [c for c in all_cols if c != resp_col and c != block_col and c != horiz_factor])
    treat_col = horiz_factor
elif design == 'augmented':
    col_a, col_b, col_c = st.columns(3)
    with col_a: treat_col = st.selectbox("Perlakuan", [c for c in all_cols if c != resp_col])
    with col_b: block_col = st.selectbox("Blok", [c for c in all_cols if c != resp_col and c != treat_col])
    with col_c: check_col = st.selectbox("Indikator Check (1=check, 0=new)", [c for c in numeric_cols if c != resp_col])

# ============================================================
# 3. STATISTIK DESKRIPTIF
# ============================================================
st.header("3. Statistik Deskriptif per Perlakuan")
desc = df.groupby(treat_col)[resp_col].agg(['count','mean','std','min','median','max']).round(4)
desc.columns = ['N', 'Mean', 'Std', 'Min', 'Median', 'Max']
desc = desc.reset_index()
st.dataframe(desc, use_container_width=True, hide_index=True)

col_v1, col_v2 = st.columns(2)
with col_v1:
    fig = px.box(df, x=treat_col, y=resp_col, color=treat_col, title="Box Plot per Perlakuan",
                 points="all")
    fig.update_layout(height=420, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
with col_v2:
    means_bar = df.groupby(treat_col)[resp_col].agg(['mean','std']).reset_index()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=means_bar[treat_col], y=means_bar['mean'],
                         error_y=dict(type='data', array=means_bar['std']),
                         marker_color=px.colors.qualitative.Plotly[:len(means_bar)]))
    fig.update_layout(title="Mean ± SD per Perlakuan", xaxis_title=treat_col,
                      yaxis_title=resp_col, height=420)
    st.plotly_chart(fig, use_container_width=True)

# Interaction plot for factorial/split/strip
if design in ['factorial', 'split_plot', 'strip_plot']:
    st.subheader("Interaction Plot")
    if design == 'factorial':
        int_df = df.groupby([factor_a, factor_b])[resp_col].mean().reset_index()
        fig = px.line(int_df, x=factor_a, y=resp_col, color=factor_b, markers=True,
                      title=f"Interaksi {factor_a} × {factor_b}")
    elif design == 'split_plot':
        int_df = df.groupby([main_plot, sub_plot])[resp_col].mean().reset_index()
        fig = px.line(int_df, x=main_plot, y=resp_col, color=sub_plot, markers=True,
                      title=f"Interaksi {main_plot} × {sub_plot}")
    elif design == 'strip_plot':
        int_df = df.groupby([horiz_factor, vert_factor])[resp_col].mean().reset_index()
        fig = px.line(int_df, x=horiz_factor, y=resp_col, color=vert_factor, markers=True,
                      title=f"Interaksi {horiz_factor} × {vert_factor}")
    fig.update_layout(height=420)
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# 4. UJI ASUMSI
# ============================================================
st.header("4. Uji Asumsi ANOVA")

tab_a1, tab_a2, tab_a3 = st.tabs(["Normalitas", "Homogenitas Variansi", "Independensi (Residual)"])

# Compute residuals per group
residuals_all = []
for g in df[treat_col].unique():
    sub = df[df[treat_col]==g][resp_col]
    residuals_all.extend((sub - sub.mean()).tolist())
residuals_all = np.array(residuals_all)

with tab_a1:
    st.markdown("**Uji Normalitas Residual:**")
    # Shapiro-Wilk on residuals
    if len(residuals_all) >= 3 and len(residuals_all) <= 5000:
        sw_stat, sw_p = shapiro(residuals_all)
    else:
        sw_stat, sw_p = np.nan, np.nan
    # Per group
    norm_rows = []
    for g in sorted(df[treat_col].unique()):
        sub = df[df[treat_col]==g][resp_col].values
        if len(sub) >= 3:
            s, p = shapiro(sub)
            norm_rows.append({'Grup': g, 'N': len(sub), 'Shapiro-Wilk W': round(s,4),
                              'p-value': round(p,6), 'Normal?': 'Ya' if p > alpha else 'Tidak'})
    st.dataframe(pd.DataFrame(norm_rows), use_container_width=True, hide_index=True)
    st.markdown(f"**Shapiro-Wilk Residual Keseluruhan:** W = {sw_stat:.4f}, p = {sw_p:.6f} → "
                f"{'**Normal**' if sw_p > alpha else '**Tidak Normal**'}")
    # QQ Plot
    fig = go.Figure()
    sorted_res = np.sort(residuals_all)
    theoretical = stats.norm.ppf(np.linspace(0.01,0.99, len(sorted_res)))
    fig.add_trace(go.Scatter(x=theoretical, y=sorted_res, mode='markers', name='Residual'))
    r_min, r_max = min(theoretical.min(), sorted_res.min()), max(theoretical.max(), sorted_res.max())
    fig.add_trace(go.Scatter(x=[r_min, r_max], y=[r_min, r_max], mode='lines',
                             name='Normal Line', line=dict(color='red', dash='dash')))
    fig.update_layout(title="QQ-Plot Residual", xaxis_title="Theoretical Quantiles",
                      yaxis_title="Sample Quantiles", height=400)
    st.plotly_chart(fig, use_container_width=True)

with tab_a2:
    st.markdown("**Uji Homogenitas Variansi (Homoscedasticity):**")
    group_data = [df[df[treat_col]==g][resp_col].values for g in sorted(df[treat_col].unique()) if len(df[df[treat_col]==g]) >= 2]
    if len(group_data) >= 2:
        lev_stat, lev_p = levene(*group_data)
        bart_stat, bart_p = bartlett(*group_data)
        homo_df = pd.DataFrame({
            'Uji': ['Levene', 'Bartlett'],
            'Statistik': [round(lev_stat,4), round(bart_stat,4)],
            'p-value': [round(lev_p,6), round(bart_p,6)],
            'Keputusan': ['Homogen' if lev_p > alpha else 'Tidak Homogen',
                          'Homogen' if bart_p > alpha else 'Tidak Homogen'],
            'Catatan': ['Robust terhadap non-normalitas', 'Sensitif terhadap normalitas']
        })
        st.dataframe(homo_df, use_container_width=True, hide_index=True)
    # Variance per group
    var_df = df.groupby(treat_col)[resp_col].var(ddof=1).reset_index()
    var_df.columns = [treat_col, 'Variansi']
    fig = px.bar(var_df, x=treat_col, y='Variansi', title="Variansi per Perlakuan",
                 color=treat_col)
    fig.update_layout(height=350, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with tab_a3:
    st.markdown("**Plot Residual vs Fitted Values:**")
    fitted_vals = []; res_vals = []
    for g in df[treat_col].unique():
        sub = df[df[treat_col]==g][resp_col]
        m = sub.mean()
        fitted_vals.extend([m]*len(sub))
        res_vals.extend((sub-m).tolist())
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fitted_vals, y=res_vals, mode='markers', opacity=0.6))
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    fig.update_layout(title="Residual vs Fitted", xaxis_title="Fitted", yaxis_title="Residual", height=400)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("Pola acak menunjukkan asumsi independensi & homogenitas terpenuhi.")

# ============================================================
# 5. TABEL ANOVA
# ============================================================
st.header("5. Tabel ANOVA")
st.markdown(f"**Rancangan:** {design_label}")

if design == 'crd':
    anova_table, mse, dfe, cv, gm = anova_crd(df, resp_col, treat_col)
elif design == 'rcbd':
    anova_table, mse, dfe, cv, gm, re = anova_rcbd(df, resp_col, treat_col, block_col)
elif design == 'latin':
    anova_table, mse, dfe, cv, gm = anova_latin_square(df, resp_col, treat_col, row_col, col_col)
elif design == 'factorial':
    anova_table, mse, dfe, cv, gm = anova_factorial(df, resp_col, factor_a, factor_b)
elif design == 'split_plot':
    anova_table, mse_a, mse_b, dfe_a, dfe_b, cv_main, cv_sub, gm = anova_split_plot(
        df, resp_col, main_plot, sub_plot, block_col)
    mse = mse_b; dfe = dfe_b; cv = cv_sub
elif design == 'nested':
    anova_table, mse, dfe, cv, gm = anova_nested(df, resp_col, factor_a, factor_b_nested)
elif design == 'strip_plot':
    anova_table, mse, dfe, cv, gm = anova_strip_plot(df, resp_col, horiz_factor, vert_factor, block_col)
elif design == 'augmented':
    anova_table, mse, dfe, cv, gm = anova_augmented(df, resp_col, treat_col, block_col, check_col)

st.dataframe(anova_table, use_container_width=True, hide_index=True)

# Summary
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
col_m1.metric("Grand Mean", f"{gm:.4f}")
col_m2.metric("MSE", f"{mse:.4f}")
col_m3.metric("df Error", int(dfe))
col_m4.metric("CV (%)", f"{cv:.2f}")

if design == 'rcbd':
    st.info(f"**Relative Efficiency (RE):** {re:.2f} — "
            f"{'Blocking efektif (RE > 1)' if re > 1 else 'Blocking kurang efektif (RE ≤ 1)'}")
if design == 'split_plot':
    st.info(f"**CV Main-Plot:** {cv_main:.2f}% | **CV Sub-Plot:** {cv_sub:.2f}%")

# ANOVA Interpretation
pval_row = anova_table[anova_table['p-value'] != '-']
if len(pval_row) > 0:
    sig_effects = pval_row[pval_row['p-value'].astype(float) < alpha]
    if len(sig_effects) > 0:
        sig_names = ', '.join(sig_effects['Source'].tolist())
        st.warning(f"⚠️ Efek **signifikan** (α = {alpha}): **{sig_names}** → Lanjutkan ke uji post-hoc.")
    else:
        st.success(f"✅ Tidak ada efek signifikan pada α = {alpha}. Uji post-hoc tidak diperlukan.")

# ============================================================
# 6. UJI POST-HOC
# ============================================================
st.header("6. Uji Lanjut (Post-Hoc)")

posthoc_options = ['Tukey HSD', 'LSD (Fisher)', 'Duncan (DMRT)', 'Bonferroni',
                   'Scheffé', 'Dunnett', 'SNK (Newman-Keuls)', 'Games-Howell']

selected_posthoc = st.multiselect("Pilih Uji Post-Hoc:", posthoc_options,
                                  default=['Tukey HSD', 'Duncan (DMRT)'])

# For factorial/split/strip, let user choose which factor to compare
if design == 'factorial':
    ph_factor = st.radio("Post-hoc untuk faktor:", [factor_a, factor_b], horizontal=True)
    treat_col_ph = ph_factor
elif design == 'split_plot':
    ph_factor = st.radio("Post-hoc untuk faktor:", [main_plot, sub_plot], horizontal=True)
    treat_col_ph = ph_factor
    if ph_factor == main_plot: mse = mse_a; dfe = dfe_a
elif design == 'strip_plot':
    ph_factor = st.radio("Post-hoc untuk faktor:", [horiz_factor, vert_factor], horizontal=True)
    treat_col_ph = ph_factor
else:
    treat_col_ph = treat_col

# Dunnett needs control
control_group = None
if 'Dunnett' in selected_posthoc:
    groups_list = sorted(df[treat_col_ph].unique())
    control_group = st.selectbox("Grup Kontrol (untuk Dunnett):", groups_list, index=0)

for ph_name in selected_posthoc:
    st.subheader(f"📌 {ph_name}")
    try:
        if ph_name == 'Tukey HSD':
            ph_result = posthoc_tukey(df, resp_col, treat_col_ph, mse, dfe, alpha)
        elif ph_name == 'LSD (Fisher)':
            ph_result = posthoc_lsd(df, resp_col, treat_col_ph, mse, dfe, alpha)
        elif ph_name == 'Duncan (DMRT)':
            ph_result = posthoc_duncan(df, resp_col, treat_col_ph, mse, dfe, alpha)
        elif ph_name == 'Bonferroni':
            ph_result = posthoc_bonferroni(df, resp_col, treat_col_ph, mse, dfe, alpha)
        elif ph_name == 'Scheffé':
            ph_result = posthoc_scheffe(df, resp_col, treat_col_ph, mse, dfe, alpha)
        elif ph_name == 'Dunnett':
            ph_result = posthoc_dunnett(df, resp_col, treat_col_ph, mse, dfe, control_group, alpha)
        elif ph_name == 'SNK (Newman-Keuls)':
            ph_result = posthoc_snk(df, resp_col, treat_col_ph, mse, dfe, alpha)
        elif ph_name == 'Games-Howell':
            ph_result = posthoc_games_howell(df, resp_col, treat_col_ph, alpha)
        st.dataframe(ph_result, use_container_width=True, hide_index=True)
        # Letter grouping
        if ph_name != 'Dunnett' and ph_name != 'Games-Howell':
            grp_letters = grouping_letters(df, resp_col, treat_col_ph, ph_result, alpha)
            if len(grp_letters) > 0:
                st.markdown(f"**Pengelompokan Huruf ({ph_name}):**")
                st.dataframe(grp_letters, use_container_width=True, hide_index=True)
    except Exception as ex:
        st.error(f"Error pada {ph_name}: {str(ex)}")

# ============================================================
# 7. VISUALISASI POST-HOC
# ============================================================
st.header("7. Visualisasi Hasil")

tab_viz1, tab_viz2, tab_viz3 = st.tabs(["Mean Comparison", "Confidence Intervals", "Heatmap Pairwise"])

with tab_viz1:
    grp_means = df.groupby(treat_col_ph)[resp_col].agg(['mean','std','count']).reset_index()
    grp_means['se'] = grp_means['std'] / np.sqrt(grp_means['count'])
    grp_means['ci'] = grp_means['se'] * t_dist.ppf(1-alpha/2, dfe)
    grp_means = grp_means.sort_values('mean', ascending=False)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=grp_means[treat_col_ph], y=grp_means['mean'],
                         error_y=dict(type='data', array=grp_means['ci']),
                         marker_color=px.colors.qualitative.Set2[:len(grp_means)]))
    fig.update_layout(title="Mean ± 95% CI per Perlakuan", xaxis_title=treat_col_ph,
                      yaxis_title=resp_col, height=450)
    st.plotly_chart(fig, use_container_width=True)

with tab_viz2:
    fig = go.Figure()
    for i, row in grp_means.iterrows():
        fig.add_trace(go.Scatter(
            x=[row['mean']-row['ci'], row['mean'], row['mean']+row['ci']],
            y=[row[treat_col_ph]]*3, mode='lines+markers',
            marker=dict(size=[8,12,8], color=px.colors.qualitative.Plotly[i%10]),
            line=dict(width=3), showlegend=False))
    fig.add_vline(x=gm, line_dash="dash", line_color="gray", annotation_text="Grand Mean")
    fig.update_layout(title="Interval Kepercayaan per Perlakuan", xaxis_title=resp_col,
                      height=max(300, 50*len(grp_means)))
    st.plotly_chart(fig, use_container_width=True)

with tab_viz3:
    groups_sorted = sorted(df[treat_col_ph].unique())
    means_dict = {g: df[df[treat_col_ph]==g][resp_col].mean() for g in groups_sorted}
    k = len(groups_sorted)
    diff_matrix = np.zeros((k, k))
    for i, g1 in enumerate(groups_sorted):
        for j, g2 in enumerate(groups_sorted):
            diff_matrix[i][j] = means_dict[g1] - means_dict[g2]
    fig = go.Figure(data=go.Heatmap(z=diff_matrix, x=groups_sorted, y=groups_sorted,
                                     colorscale='RdBu_r', zmid=0,
                                     text=np.round(diff_matrix, 2), texttemplate='%{text}'))
    fig.update_layout(title="Heatmap Selisih Mean Antar Perlakuan", height=450)
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# 8. TRANSFORMASI DATA (OPSIONAL)
# ============================================================
st.header("8. Transformasi Data (Opsional)")
st.markdown("Jika asumsi tidak terpenuhi, coba transformasi data:")
transform = st.selectbox("Pilih transformasi:", ['Tidak ada', 'Log (ln)', 'Log10', 'Square Root (√)',
                                                   'Reciprocal (1/x)', 'Arcsin √p (proporsi)',
                                                   'Box-Cox'])
if transform != 'Tidak ada':
    df_tr = df.copy()
    y = df_tr[resp_col].values.astype(float)
    if transform == 'Log (ln)':
        y_tr = np.log(y[y > 0]); label = 'ln(Y)'
    elif transform == 'Log10':
        y_tr = np.log10(y[y > 0]); label = 'log10(Y)'
    elif transform == 'Square Root (√)':
        y_tr = np.sqrt(y[y >= 0]); label = '√Y'
    elif transform == 'Reciprocal (1/x)':
        y_tr = 1 / y[y > 0]; label = '1/Y'
    elif transform == 'Arcsin √p (proporsi)':
        yc = np.clip(y, 0, 1)
        y_tr = np.arcsin(np.sqrt(yc)); label = 'arcsin(√Y)'
    elif transform == 'Box-Cox':
        from scipy.stats import boxcox
        y_pos = y[y > 0]
        y_tr, lam = boxcox(y_pos)
        label = f'Box-Cox (λ={lam:.3f})'
    if len(y_tr) == len(df_tr):
        df_tr[f'{resp_col}_transformed'] = y_tr
        st.dataframe(df_tr.head(20), use_container_width=True)
        sw2, p2 = shapiro(y_tr) if len(y_tr) >= 3 else (np.nan, np.nan)
        st.markdown(f"**Shapiro-Wilk setelah transformasi {label}:** W = {sw2:.4f}, p = {p2:.6f} → "
                    f"{'Normal' if p2 > alpha else 'Masih Tidak Normal'}")
    else:
        st.warning("Beberapa nilai tidak bisa ditransformasi (negatif/nol). Gunakan data valid saja.")

# ============================================================
# 9. KRUSKAL-WALLIS (NON-PARAMETRIK)
# ============================================================
st.header("9. Alternatif Non-Parametrik")
st.markdown("Jika asumsi ANOVA tidak terpenuhi, gunakan uji non-parametrik sebagai alternatif.")

groups_np = [df[df[treat_col_ph]==g][resp_col].values for g in sorted(df[treat_col_ph].unique())]
kw_stat, kw_p = stats.kruskal(*groups_np)
st.markdown(f"**Kruskal-Wallis H Test:** H = {kw_stat:.4f}, p = {kw_p:.6f} → "
            f"{'**Signifikan**' if kw_p < alpha else '**Tidak Signifikan**'} pada α = {alpha}")

if len(sorted(df[treat_col_ph].unique())) == 2:
    u_stat, u_p = stats.mannwhitneyu(groups_np[0], groups_np[1], alternative='two-sided')
    st.markdown(f"**Mann-Whitney U Test:** U = {u_stat:.4f}, p = {u_p:.6f}")

# Post-hoc non-parametrik: Dunn's test (simplified)
if kw_p < alpha:
    st.markdown("**Post-hoc: Dunn's Test (approximation):**")
    groups_sorted_np = sorted(df[treat_col_ph].unique())
    N_all = len(df)
    ranks = stats.rankdata(df[resp_col].values)
    df_ranked = df.copy(); df_ranked['_rank'] = ranks
    dunn_rows = []
    for g1, g2 in combinations(groups_sorted_np, 2):
        r1 = df_ranked[df_ranked[treat_col_ph]==g1]['_rank']
        r2 = df_ranked[df_ranked[treat_col_ph]==g2]['_rank']
        n1, n2 = len(r1), len(r2)
        mean_diff = r1.mean() - r2.mean()
        se = np.sqrt((N_all*(N_all+1)/12) * (1/n1 + 1/n2))
        z = abs(mean_diff) / se
        p_dunn = 2 * (1 - stats.norm.cdf(z))
        m_comp = len(list(combinations(groups_sorted_np, 2)))
        p_adj = min(p_dunn * m_comp, 1.0)
        dunn_rows.append({'Perbandingan': f'{g1} vs {g2}', 'z': round(z,4),
                          'p-value': round(p_dunn,6), 'p-adj (Bonf)': round(p_adj,6),
                          f'Signifikan (α={alpha})': 'Ya' if p_adj < alpha else 'Tidak'})
    st.dataframe(pd.DataFrame(dunn_rows), use_container_width=True, hide_index=True)

# ============================================================
# 10. EKSPOR
# ============================================================
st.header("10. Ekspor Hasil")
col_e1, col_e2, col_e3 = st.columns(3)
with col_e1:
    lines = [
        "="*70, "ANALISIS RANCANGAN PERCOBAAN", "="*70,
        f"Rancangan    : {design_label}",
        f"N            : {len(df)}", f"Respon       : {resp_col}",
        f"Perlakuan    : {treat_col_ph}", f"Alpha        : {alpha}",
        f"Grand Mean   : {gm:.4f}", f"MSE          : {mse:.4f}",
        f"CV (%)       : {cv:.2f}", "",
        "="*70, "TABEL ANOVA", "="*70,
        anova_table.to_string(index=False), "",
        "="*70, "KRUSKAL-WALLIS", "="*70,
        f"H = {kw_stat:.4f}, p = {kw_p:.6f}"]
    st.download_button("📥 Summary (TXT)", data="\n".join(lines),
                       file_name="anova_summary.txt", mime="text/plain")
with col_e2:
    st.download_button("📥 ANOVA Table (CSV)", data=anova_table.to_csv(index=False),
                       file_name="anova_table.csv", mime="text/csv")
with col_e3:
    st.download_button("📥 Data (CSV)", data=df.to_csv(index=False),
                       file_name="experiment_data.csv", mime="text/csv")

# FOOTER
st.markdown("---")
st.markdown("""
**Referensi Metodologis:**
- **CRD/RAL:** Rancangan paling sederhana, unit eksperimen homogen.
- **RCBD/RAK:** Pengendalian 1 sumber variasi (blok). RE > 1 → blocking efektif.
- **Latin Square:** Pengendalian 2 sumber variasi (baris & kolom).
- **Faktorial:** Mengevaluasi efek utama dan interaksi 2+ faktor.
- **Split-Plot:** 2 faktor dengan ukuran petak berbeda; galat terpisah untuk main/sub-plot.
- **Nested:** Faktor B tersarang dalam A; level B unik untuk setiap level A.
- **Strip-Plot:** 2 faktor diterapkan dalam strip horizontal & vertikal; 3 galat.
- **Augmented:** Check diulang, perlakuan baru tidak; efisien untuk skrining awal.
- **Tukey HSD:** Kontrol family-wise error rate; cocok untuk semua pairwise.
- **LSD (Fisher):** Paling liberal; hanya jika ANOVA signifikan (protected LSD).
- **Duncan (DMRT):** Stepwise; critical value bervariasi menurut range.
- **Bonferroni:** Koreksi α/m; konservatif tapi simpel.
- **Scheffé:** Paling konservatif; berlaku untuk semua kontras.
- **Dunnett:** Khusus perbandingan terhadap kontrol.
- **SNK (Newman-Keuls):** Mirip Duncan, menggunakan studentized range.
- **Games-Howell:** Tidak asumsikan variansi homogen; gunakan jika Levene signifikan.

Dibangun dengan **Streamlit** + **SciPy** + **NumPy** + **Plotly** | Python
""")
