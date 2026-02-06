import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Logit, Probit
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.families import links as smlinks
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Regresi Logistik Biner", page_icon="🎯", layout="wide")
st.title("🎯 Regresi Logistik Biner — Extended Edition")
st.markdown("""
Aplikasi lengkap untuk **regresi logistik biner** dengan **6 link function**,
**Train-Test Split**, **Stratified K-Fold Cross Validation**, estimasi MLE,
uji asumsi, diagnostik, ROC, confusion matrix, odds ratio, Hosmer-Lemeshow, dan visualisasi.
""")

# ============================
# HELPERS
# ============================
def _arr(obj):
    return np.asarray(obj).flatten()

def hosmer_lemeshow(y, pred_prob, g=10):
    df_hl = pd.DataFrame({'y': y, 'prob': pred_prob})
    df_hl['decile'] = pd.qcut(df_hl['prob'], g, labels=False, duplicates='drop')
    obs = df_hl.groupby('decile')['y'].agg(['sum', 'count'])
    obs.columns = ['obs_1', 'n']
    obs['obs_0'] = obs['n'] - obs['obs_1']
    exp_prob = df_hl.groupby('decile')['prob'].mean()
    obs['exp_1'] = exp_prob * obs['n']
    obs['exp_0'] = (1 - exp_prob) * obs['n']
    hl_stat = np.sum((obs['obs_1'] - obs['exp_1'])**2 / (obs['exp_1'] + 1e-15) +
                      (obs['obs_0'] - obs['exp_0'])**2 / (obs['exp_0'] + 1e-15))
    df_dof = len(obs) - 2
    p_val = 1 - stats.chi2.cdf(hl_stat, df_dof)
    return hl_stat, p_val, df_dof, obs

def classification_table(y_true, y_pred):
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tp, tn, fp, fn

def calc_metrics(y_true, y_pred, y_prob):
    tp, tn, fp, fn = classification_table(y_true, y_pred)
    N = len(y_true)
    accuracy = (tp + tn) / N if N > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2*precision*sensitivity/(precision+sensitivity) if (precision+sensitivity) > 0 else 0
    fpr, tpr, _ = roc_curve_manual(y_true, y_prob)
    auc_val = auc_trapezoidal(fpr, tpr)
    return {'Accuracy': accuracy, 'Sensitivity': sensitivity, 'Specificity': specificity,
            'Precision': precision, 'F1': f1, 'AUC': auc_val, 'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn}

def roc_curve_manual(y_true, y_prob):
    thresholds = np.linspace(0, 1, 201)
    tpr_list, fpr_list = [], []
    for t in thresholds:
        pred = (y_prob >= t).astype(int)
        tp, tn, fp, fn = classification_table(y_true, pred)
        tpr_list.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        fpr_list.append(fp / (fp + tn) if (fp + tn) > 0 else 0)
    return np.array(fpr_list), np.array(tpr_list), thresholds

def auc_trapezoidal(fpr, tpr):
    idx = np.argsort(fpr)
    return np.abs(np.trapz(tpr[idx], fpr[idx]))

def precision_recall_manual(y_true, y_prob):
    thresholds = np.linspace(0.01, 1, 200)
    prec_list, rec_list = [], []
    for t in thresholds:
        pred = (y_prob >= t).astype(int)
        tp, tn, fp, fn = classification_table(y_true, pred)
        prec_list.append(tp / (tp + fp) if (tp + fp) > 0 else 0)
        rec_list.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
    return np.array(prec_list), np.array(rec_list), thresholds

def interpret_or(or_val):
    if or_val > 1: return f"Meningkatkan odds {or_val:.2f}×"
    elif or_val < 1: return f"Menurunkan odds {1/or_val:.2f}×"
    else: return "Tidak ada efek"

def interpret_corr(r):
    r = abs(r)
    if r < 0.1: return "Sangat Lemah"
    elif r < 0.3: return "Lemah"
    elif r < 0.5: return "Sedang"
    elif r < 0.7: return "Kuat"
    else: return "Sangat Kuat"

LINK_MAP = {
    'Logit': smlinks.Logit(),
    'Probit': smlinks.Probit(),
    'Complementary Log-Log (CLogLog)': smlinks.CLogLog(),
    'Log-Log': smlinks.LogLog(),
    'Cauchy (Cauchit)': smlinks.Cauchy(),
    'Log': smlinks.Log(),
}

LINK_DESC = {
    'Logit': 'g(μ) = log(μ/(1-μ)) — Standar, simetris, odds ratio interpretable',
    'Probit': 'g(μ) = Φ⁻¹(μ) — Berdasarkan distribusi normal standar, simetris',
    'Complementary Log-Log (CLogLog)': 'g(μ) = log(-log(1-μ)) — Asimetris, cocok untuk survival/rare events',
    'Log-Log': 'g(μ) = -log(-log(μ)) — Asimetris (kebalikan CLogLog)',
    'Cauchy (Cauchit)': 'g(μ) = tan(π(μ-0.5)) — Heavy-tailed, simetris',
    'Log': 'g(μ) = log(μ) — Risk ratio interpretable',
}

def fit_glm_binary(y, X, link_obj):
    family = Binomial(link=link_obj)
    model = sm.GLM(y, X, family=family).fit(maxiter=100)
    return model

def stratified_split(y, test_size=0.2, random_state=42):
    np.random.seed(random_state)
    idx_0 = np.where(y == 0)[0]
    idx_1 = np.where(y == 1)[0]
    np.random.shuffle(idx_0)
    np.random.shuffle(idx_1)
    n_test_0 = max(1, int(len(idx_0) * test_size))
    n_test_1 = max(1, int(len(idx_1) * test_size))
    test_idx = np.concatenate([idx_0[:n_test_0], idx_1[:n_test_1]])
    train_idx = np.concatenate([idx_0[n_test_0:], idx_1[n_test_1:]])
    np.random.shuffle(test_idx)
    np.random.shuffle(train_idx)
    return train_idx, test_idx

def stratified_kfold_indices(y, k=5, random_state=42):
    np.random.seed(random_state)
    idx_0 = np.where(y == 0)[0].copy()
    idx_1 = np.where(y == 1)[0].copy()
    np.random.shuffle(idx_0)
    np.random.shuffle(idx_1)
    folds_0 = np.array_split(idx_0, k)
    folds_1 = np.array_split(idx_1, k)
    folds = []
    for i in range(k):
        test = np.concatenate([folds_0[i], folds_1[i]])
        train_parts_0 = [folds_0[j] for j in range(k) if j != i]
        train_parts_1 = [folds_1[j] for j in range(k) if j != i]
        train = np.concatenate(train_parts_0 + train_parts_1)
        folds.append((train.astype(int), test.astype(int)))
    return folds

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
    n = 400
    x1 = np.random.normal(50, 15, n)
    x2 = np.random.normal(5, 2, n)
    x3 = np.random.choice([0, 1], n, p=[0.6, 0.4])
    x4 = np.random.normal(100, 25, n)
    x5 = np.random.exponential(3, n)
    logit_p = -6 + 0.08*x1 + 0.4*x2 + 1.2*x3 - 0.01*x4 + 0.15*x5
    prob = 1 / (1 + np.exp(-logit_p))
    y = np.random.binomial(1, prob)
    df = pd.DataFrame({
        'Status': y, 'Usia': x1.round(1), 'Skor_Risiko': x2.round(2),
        'Riwayat': x3.astype(int), 'Pendapatan': x4.round(1), 'Lama_Paparan': x5.round(2)
    })
    st.sidebar.success(f"Data demo: {n} obs, P(Y=1)={y.mean():.2%}")
else:
    st.warning("Silakan upload data atau gunakan data demo.")
    st.stop()

# ============================
# 1. EKSPLORASI
# ============================
st.header("1. Eksplorasi Data")
c1, c2, c3 = st.columns(3)
c1.metric("Jumlah Observasi", df.shape[0])
c2.metric("Jumlah Variabel", df.shape[1])
c3.metric("Missing Values", df.isnull().sum().sum())

tab_d1, tab_d2 = st.tabs(["Data", "Deskriptif"])
with tab_d1:
    st.dataframe(df.head(30), use_container_width=True)
with tab_d2:
    st.dataframe(df.describe(include='all').T.round(4), use_container_width=True)

# ============================
# 2. SETUP
# ============================
st.header("2. Setup Variabel & Model")
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

dep_var = st.selectbox("Variabel Dependen (Y — biner 0/1)", numeric_cols, index=0)
y_unique = df[dep_var].dropna().unique()
if len(y_unique) != 2:
    st.error(f"Variabel dependen harus biner. Saat ini: {len(y_unique)} nilai unik.")
    st.stop()

indep_options = [c for c in numeric_cols if c != dep_var]
indep_vars = st.multiselect("Variabel Independen (X₁, X₂, ...)", indep_options, default=indep_options)
if len(indep_vars) < 1:
    st.warning("Pilih minimal 1 variabel independen.")
    st.stop()

# Link function
st.subheader("2b. Link Function")
link_names = list(LINK_MAP.keys())
selected_links = st.multiselect("Pilih Link Function (bisa multi untuk perbandingan)",
                                 link_names, default=['Logit'])
if len(selected_links) < 1:
    st.warning("Pilih minimal 1 link function.")
    st.stop()

for lk in selected_links:
    st.caption(f"**{lk}:** {LINK_DESC[lk]}")

# Alpha & threshold
col_s1, col_s2 = st.columns(2)
with col_s1:
    alpha = st.selectbox("Tingkat Signifikansi (α)", [0.01, 0.05, 0.10], index=1)
with col_s2:
    threshold = st.slider("Threshold Klasifikasi", 0.0, 1.0, 0.5, 0.01)

# ============================
# 3. TRAIN-TEST SPLIT
# ============================
st.header("3. Train-Test Split & Cross Validation")

panel = df[[dep_var] + indep_vars].dropna().copy().reset_index(drop=True)
y_all = panel[dep_var].values.astype(float)
X_all = panel[indep_vars].values.astype(float)
X_all_const = sm.add_constant(X_all)
var_names = ['const'] + indep_vars
N_all = len(y_all)

col_tt1, col_tt2, col_tt3 = st.columns(3)
with col_tt1:
    test_size = st.slider("Test Size (%)", 10, 40, 20, 5) / 100
with col_tt2:
    k_folds = st.selectbox("K-Fold Cross Validation", [3, 5, 7, 10], index=1)
with col_tt3:
    random_state = st.number_input("Random Seed", 1, 9999, 42)

train_idx, test_idx = stratified_split(y_all, test_size=test_size, random_state=random_state)
y_train, y_test = y_all[train_idx], y_all[test_idx]
X_train, X_test = X_all_const[train_idx], X_all_const[test_idx]

n1_tr = int(y_train.sum()); n0_tr = len(y_train) - n1_tr
n1_te = int(y_test.sum()); n0_te = len(y_test) - n1_te

split_info = pd.DataFrame({
    'Set': ['Train', 'Test', 'Total'],
    'N': [len(y_train), len(y_test), N_all],
    'Y=0': [n0_tr, n0_te, n0_tr+n0_te],
    'Y=1': [n1_tr, n1_te, n1_tr+n1_te],
    'P(Y=1)': [f'{n1_tr/len(y_train):.2%}', f'{n1_te/len(y_test):.2%}', f'{y_all.mean():.2%}']
})
st.dataframe(split_info, use_container_width=True, hide_index=True)

# ============================
# 4. ESTIMASI PER LINK FUNCTION
# ============================
st.header("4. Estimasi Model per Link Function")

model_results = {}
for link_name in selected_links:
    st.subheader(f"4. {link_name}")
    link_obj = LINK_MAP[link_name]

    try:
        model = fit_glm_binary(y_train, X_train, link_obj)
    except Exception as e:
        st.error(f"Estimasi {link_name} gagal: {e}")
        continue

    params = _arr(model.params)
    bse = _arr(model.bse)
    zvals = params / bse
    pvals = 2 * (1 - stats.norm.cdf(np.abs(zvals)))
    conf = np.column_stack([params - 1.96*bse, params + 1.96*bse])

    # Koefisien
    coef_df = pd.DataFrame({
        'Variabel': var_names,
        'Koefisien (β)': params.round(6),
        'Std. Error': bse.round(6),
        'z-statistic': zvals.round(4),
        'p-value': pvals.round(6),
        f'Signifikan (α={alpha})': ['Ya' if p < alpha else 'Tidak' for p in pvals]
    })

    if link_name == 'Logit':
        coef_df['Odds Ratio (eᵝ)'] = np.exp(params).round(4)
        coef_df['OR Interpretasi'] = [interpret_or(np.exp(b)) for b in params]

    st.dataframe(coef_df, use_container_width=True, hide_index=True)

    # Train metrics
    pred_train = _arr(model.predict(X_train))
    pred_test = _arr(model.predict(X_test))

    train_m = calc_metrics(y_train, (pred_train >= threshold).astype(int), pred_train)
    test_m = calc_metrics(y_test, (pred_test >= threshold).astype(int), pred_test)

    perf_df = pd.DataFrame({
        'Metrik': ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'F1-Score', 'AUC'],
        'Train': [round(train_m[k], 4) for k in ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'F1', 'AUC']],
        'Test': [round(test_m[k], 4) for k in ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'F1', 'AUC']]
    })
    st.markdown(f"**Performance (threshold={threshold}):**")
    st.dataframe(perf_df, use_container_width=True, hide_index=True)

    # Goodness of fit
    ll_full = float(model.llf)
    aic = float(model.aic)
    bic = float(model.bic)
    deviance = float(model.deviance)
    pearson_chi2 = float(model.pearson_chi2)

    # Pseudo R2
    null_model = fit_glm_binary(y_train, sm.add_constant(np.ones(len(y_train)))[:, :1], link_obj)
    ll_null = float(null_model.llf)
    mcfadden = 1 - ll_full / ll_null if ll_null != 0 else 0

    gof_df = pd.DataFrame({
        'Metrik': ['Log-Likelihood', 'AIC', 'BIC', 'Deviance', 'Pearson χ²', 'McFadden R²'],
        'Nilai': [round(ll_full, 4), round(aic, 4), round(bic, 4),
                  round(deviance, 4), round(pearson_chi2, 4), round(mcfadden, 6)]
    })
    with st.expander(f"Goodness of Fit — {link_name}"):
        st.dataframe(gof_df, use_container_width=True, hide_index=True)

    model_results[link_name] = {
        'model': model, 'coef_df': coef_df, 'params': params, 'bse': bse,
        'pred_train': pred_train, 'pred_test': pred_test,
        'train_m': train_m, 'test_m': test_m,
        'aic': aic, 'bic': bic, 'mcfadden': mcfadden, 'll': ll_full, 'deviance': deviance
    }

if len(model_results) == 0:
    st.error("Tidak ada model yang berhasil diestimasi.")
    st.stop()

# ============================
# 5. PERBANDINGAN LINK FUNCTIONS
# ============================
if len(model_results) > 1:
    st.header("5. Perbandingan Link Functions")

    comp_rows = []
    for lk, res in model_results.items():
        comp_rows.append({
            'Link Function': lk,
            'AIC': round(res['aic'], 4),
            'BIC': round(res['bic'], 4),
            'McFadden R²': round(res['mcfadden'], 6),
            'Deviance': round(res['deviance'], 4),
            'Train AUC': round(res['train_m']['AUC'], 4),
            'Test AUC': round(res['test_m']['AUC'], 4),
            'Train Acc': round(res['train_m']['Accuracy'], 4),
            'Test Acc': round(res['test_m']['Accuracy'], 4),
        })
    comp_df = pd.DataFrame(comp_rows)
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    best_link = comp_df.loc[comp_df['AIC'].idxmin(), 'Link Function']
    st.success(f"**Link function terbaik berdasarkan AIC: {best_link}** (AIC = {comp_df['AIC'].min():.4f})")

    # ROC Comparison
    fig_roc_comp = go.Figure()
    colors = px.colors.qualitative.Plotly
    for i, (lk, res) in enumerate(model_results.items()):
        fpr, tpr, _ = roc_curve_manual(y_test, res['pred_test'])
        auc_v = auc_trapezoidal(fpr, tpr)
        fig_roc_comp.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                                           name=f'{lk} (AUC={auc_v:.4f})',
                                           line=dict(color=colors[i % len(colors)], width=2)))
    fig_roc_comp.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Random',
                                       line=dict(color='gray', dash='dash')))
    fig_roc_comp.update_layout(title="ROC Curve Comparison (Test Set)", height=500,
                                xaxis_title="FPR", yaxis_title="TPR")
    st.plotly_chart(fig_roc_comp, use_container_width=True)

    # Link function curves
    st.subheader("Kurva Link Functions")
    fig_link = go.Figure()
    eta = np.linspace(-5, 5, 500)
    for i, lk in enumerate(selected_links):
        link_obj = LINK_MAP[lk]
        try:
            mu = link_obj.inverse(eta)
            fig_link.add_trace(go.Scatter(x=eta, y=mu, mode='lines', name=lk,
                                           line=dict(color=colors[i % len(colors)], width=2)))
        except:
            pass
    fig_link.update_layout(title="Perbandingan Kurva Link Functions: η → P(Y=1)",
                            xaxis_title="η (linear predictor)", yaxis_title="P(Y=1)",
                            height=450)
    fig_link.add_hline(y=0.5, line_dash="dot", line_color="gray")
    st.plotly_chart(fig_link, use_container_width=True)

    # Coefficient comparison
    fig_coef = go.Figure()
    for i, (lk, res) in enumerate(model_results.items()):
        fig_coef.add_trace(go.Bar(x=indep_vars, y=res['params'][1:], name=lk))
    fig_coef.update_layout(title="Perbandingan Koefisien per Link Function", barmode='group', height=450)
    st.plotly_chart(fig_coef, use_container_width=True)

    primary_link = best_link
else:
    primary_link = selected_links[0]

# Use primary model from here
prim = model_results[primary_link]
model = prim['model']
params = prim['params']
pred_train = prim['pred_train']
pred_test = prim['pred_test']

# ============================
# 6. K-FOLD CROSS VALIDATION
# ============================
st.header("6. Stratified K-Fold Cross Validation")
st.markdown(f"**{k_folds}-Fold** Stratified CV pada seluruh data — mempertahankan proporsi kelas di tiap fold.")

link_obj_primary = LINK_MAP[primary_link]
folds = stratified_kfold_indices(y_all, k=k_folds, random_state=random_state)

cv_rows = []
fold_auc_list = []
fold_acc_list = []
fold_f1_list = []

for fold_i, (tr_idx, te_idx) in enumerate(folds):
    y_tr_f = y_all[tr_idx]; y_te_f = y_all[te_idx]
    X_tr_f = X_all_const[tr_idx]; X_te_f = X_all_const[te_idx]
    try:
        m_f = fit_glm_binary(y_tr_f, X_tr_f, link_obj_primary)
        pred_f = _arr(m_f.predict(X_te_f))
        m_f_metrics = calc_metrics(y_te_f, (pred_f >= threshold).astype(int), pred_f)
        cv_rows.append({
            'Fold': fold_i + 1, 'N_train': len(y_tr_f), 'N_test': len(y_te_f),
            'P(Y=1) test': f'{y_te_f.mean():.2%}',
            'Accuracy': round(m_f_metrics['Accuracy'], 4),
            'Sensitivity': round(m_f_metrics['Sensitivity'], 4),
            'Specificity': round(m_f_metrics['Specificity'], 4),
            'F1-Score': round(m_f_metrics['F1'], 4),
            'AUC': round(m_f_metrics['AUC'], 4)
        })
        fold_auc_list.append(m_f_metrics['AUC'])
        fold_acc_list.append(m_f_metrics['Accuracy'])
        fold_f1_list.append(m_f_metrics['F1'])
    except:
        cv_rows.append({'Fold': fold_i+1, 'N_train': len(y_tr_f), 'N_test': len(y_te_f),
                        'Accuracy': np.nan, 'AUC': np.nan})

cv_df = pd.DataFrame(cv_rows)
st.dataframe(cv_df, use_container_width=True, hide_index=True)

# Summary stats
if len(fold_auc_list) > 0:
    cv_summary = pd.DataFrame({
        'Metrik': ['Accuracy', 'F1-Score', 'AUC'],
        'Mean': [round(np.mean(fold_acc_list), 4), round(np.mean(fold_f1_list), 4), round(np.mean(fold_auc_list), 4)],
        'Std': [round(np.std(fold_acc_list), 4), round(np.std(fold_f1_list), 4), round(np.std(fold_auc_list), 4)],
        'Min': [round(np.min(fold_acc_list), 4), round(np.min(fold_f1_list), 4), round(np.min(fold_auc_list), 4)],
        'Max': [round(np.max(fold_acc_list), 4), round(np.max(fold_f1_list), 4), round(np.max(fold_auc_list), 4)],
    })
    st.dataframe(cv_summary, use_container_width=True, hide_index=True)

    fig_cv = go.Figure()
    fig_cv.add_trace(go.Bar(x=cv_df['Fold'].astype(str), y=cv_df['AUC'], name='AUC', marker_color='steelblue'))
    fig_cv.add_trace(go.Bar(x=cv_df['Fold'].astype(str), y=cv_df['Accuracy'], name='Accuracy', marker_color='salmon'))
    fig_cv.add_hline(y=np.mean(fold_auc_list), line_dash="dash", line_color="blue",
                      annotation_text=f"Mean AUC={np.mean(fold_auc_list):.4f}")
    fig_cv.update_layout(title=f"{k_folds}-Fold CV Performance ({primary_link})", barmode='group', height=400)
    st.plotly_chart(fig_cv, use_container_width=True)

# ============================
# 7. ODDS RATIO & FOREST PLOT (LOGIT ONLY)
# ============================
if 'Logit' in model_results:
    st.header("7. Odds Ratio & Forest Plot")
    logit_res = model_results['Logit']
    or_vals = np.exp(logit_res['params'])
    or_ci_low = np.exp(logit_res['params'] - 1.96 * logit_res['bse'])
    or_ci_up = np.exp(logit_res['params'] + 1.96 * logit_res['bse'])

    or_df = pd.DataFrame({
        'Variabel': var_names[1:],
        'β': logit_res['params'][1:].round(6),
        'Odds Ratio': or_vals[1:].round(4),
        'OR CI Lower': or_ci_low[1:].round(4),
        'OR CI Upper': or_ci_up[1:].round(4),
        'Interpretasi': [interpret_or(o) for o in or_vals[1:]]
    })
    st.dataframe(or_df, use_container_width=True, hide_index=True)

    fig_or = go.Figure()
    for _, row in or_df.iterrows():
        color = '#2ca02c' if row['Odds Ratio'] > 1 else '#d62728'
        fig_or.add_trace(go.Scatter(
            x=[row['OR CI Lower'], row['Odds Ratio'], row['OR CI Upper']],
            y=[row['Variabel']] * 3,
            mode='lines+markers', marker=dict(size=[8, 14, 8], color=color),
            line=dict(color=color, width=3), showlegend=False))
    fig_or.add_vline(x=1, line_dash="dash", line_color="gray")
    fig_or.update_layout(title="Forest Plot: Odds Ratio ± 95% CI",
                          xaxis_title="Odds Ratio", height=max(300, 60*len(indep_vars)))
    st.plotly_chart(fig_or, use_container_width=True)

# ============================
# 8. HOSMER-LEMESHOW
# ============================
st.header("8. Hosmer-Lemeshow & Calibration")
hl_stat, hl_p, hl_df_val, hl_table = hosmer_lemeshow(y_test, pred_test)
hl_result = pd.DataFrame({
    'Metrik': ['HL χ²', 'df', 'p-value', f'Keputusan (α={alpha})'],
    'Nilai': [round(hl_stat, 4), hl_df_val, round(hl_p, 6),
              'Model Fit Baik (Gagal Tolak H₀)' if hl_p > alpha else 'Model Tidak Fit']
})
st.dataframe(hl_result, use_container_width=True, hide_index=True)

fig_cal = go.Figure()
fig_cal.add_trace(go.Scatter(x=hl_table['exp_1']/hl_table['n'],
                              y=hl_table['obs_1']/hl_table['n'],
                              mode='markers+text', marker=dict(size=10),
                              text=list(range(1, len(hl_table)+1)),
                              textposition='top center', name='Desil'))
fig_cal.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines',
                              line=dict(color='red', dash='dash'), name='Perfect'))
fig_cal.update_layout(title="Calibration Plot (Test Set)", height=450,
                       xaxis_title="Expected P", yaxis_title="Observed P")
st.plotly_chart(fig_cal, use_container_width=True)

# ============================
# 9. ROC, PR, CONFUSION MATRIX (TEST SET)
# ============================
st.header("9. ROC, Precision-Recall & Confusion Matrix (Test Set)")
tab_roc, tab_pr, tab_cm = st.tabs(["ROC Curve", "Precision-Recall", "Confusion Matrix"])

fpr, tpr, roc_th = roc_curve_manual(y_test, pred_test)
auc_val = auc_trapezoidal(fpr, tpr)

with tab_roc:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'AUC={auc_val:.4f}',
                              line=dict(color='blue', width=2)))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Random',
                              line=dict(color='gray', dash='dash')))
    fig.update_layout(title=f"ROC Curve — Test Set (AUC={auc_val:.4f})", height=500,
                       xaxis_title="FPR", yaxis_title="TPR")
    st.plotly_chart(fig, use_container_width=True)
    j_idx = np.argmax(tpr - fpr)
    st.info(f"**Youden's J Optimal Threshold:** {roc_th[j_idx]:.4f} | "
            f"Sens={tpr[j_idx]:.4f} | Spec={1-fpr[j_idx]:.4f}")

with tab_pr:
    prec_a, rec_a, _ = precision_recall_manual(y_test, pred_test)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rec_a, y=prec_a, mode='lines', line=dict(width=2)))
    fig.add_hline(y=y_test.mean(), line_dash="dash", line_color="gray",
                   annotation_text=f"Baseline={y_test.mean():.3f}")
    fig.update_layout(title="Precision-Recall Curve (Test)", height=450,
                       xaxis_title="Recall", yaxis_title="Precision")
    st.plotly_chart(fig, use_container_width=True)

with tab_cm:
    y_pred_test = (pred_test >= threshold).astype(int)
    tp, tn, fp, fn = classification_table(y_test, y_pred_test)
    fig = go.Figure(data=go.Heatmap(
        z=[[tn, fp], [fn, tp]],
        x=['Pred=0', 'Pred=1'], y=['Actual=0', 'Actual=1'],
        colorscale='Blues', text=[[tn, fp], [fn, tp]], texttemplate="%{text}",
        textfont=dict(size=20)))
    fig.update_layout(title=f"Confusion Matrix — Test (threshold={threshold})", height=400)
    st.plotly_chart(fig, use_container_width=True)

    test_metrics = calc_metrics(y_test, y_pred_test, pred_test)
    tm_df = pd.DataFrame({
        'Metrik': list(test_metrics.keys()),
        'Nilai': [round(v, 4) if isinstance(v, float) else v for v in test_metrics.values()]
    })
    st.dataframe(tm_df, use_container_width=True, hide_index=True)

# ============================
# 10. UJI ASUMSI
# ============================
st.header("10. Uji Asumsi")

# VIF
st.subheader("10a. Multikolinieritas (VIF)")
from statsmodels.stats.outliers_influence import variance_inflation_factor
X_vif = sm.add_constant(X_all)
vif_data = pd.DataFrame({
    'Variabel': indep_vars,
    'VIF': [round(variance_inflation_factor(X_vif, i+1), 4) for i in range(len(indep_vars))],
})
vif_data['Keterangan'] = vif_data['VIF'].apply(
    lambda v: 'OK (< 5)' if v < 5 else ('Moderat (5-10)' if v < 10 else 'Tinggi ⚠️'))
st.dataframe(vif_data, use_container_width=True, hide_index=True)

# Box-Tidwell
st.subheader("10b. Linieritas dalam Logit (Box-Tidwell)")
bt_rows = []
for v in indep_vars:
    x_v = panel[v].values.astype(float)
    if np.all(x_v > 0) and len(np.unique(x_v)) > 2:
        x_log_int = x_v * np.log(x_v + 1e-15)
        X_bt = sm.add_constant(np.column_stack([x_v, x_log_int]))
        try:
            bt_model = Logit(y_all, X_bt).fit(disp=0, maxiter=50)
            bt_p = _arr(bt_model.pvalues)[2]
            bt_rows.append({'Variabel': v, 'X*ln(X) p-value': round(bt_p, 6),
                            'Keputusan': 'Linier' if bt_p > alpha else 'Tidak linier ⚠️'})
        except:
            bt_rows.append({'Variabel': v, 'X*ln(X) p-value': np.nan, 'Keputusan': 'N/A'})
    else:
        bt_rows.append({'Variabel': v, 'X*ln(X) p-value': np.nan, 'Keputusan': 'Biner/ordinal — skip'})
st.dataframe(pd.DataFrame(bt_rows), use_container_width=True, hide_index=True)

# Sample size
st.subheader("10c. Kecukupan Sampel (EPP)")
n1_all = int(y_all.sum()); n0_all = N_all - n1_all
epp = min(n0_all, n1_all) / len(indep_vars) if len(indep_vars) > 0 else 0
st.markdown(f"**EPP = {epp:.1f}** — {'Memadai ✅' if epp >= 10 else 'Kurang ⚠️'} (Events Per Predictor, ≥ 10 direkomendasikan)")

# ============================
# 11. DIAGNOSTIK RESIDUAL
# ============================
st.header("11. Diagnostik Residual")
resid_pear = _arr(model.resid_pearson)
resid_dev = _arr(model.resid_deviance)
fitted_train = pred_train

tab_r1, tab_r2, tab_r3 = st.tabs(["Pearson Residual", "Deviance Residual", "Cook's Distance"])
with tab_r1:
    fig = px.scatter(x=fitted_train, y=resid_pear, opacity=0.4,
                      labels={'x': 'Predicted P', 'y': 'Pearson Residual'})
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    fig.add_hline(y=2, line_dash="dot", line_color="orange")
    fig.add_hline(y=-2, line_dash="dot", line_color="orange")
    fig.update_layout(title="Pearson Residual vs Predicted (Train)", height=400)
    st.plotly_chart(fig, use_container_width=True)

with tab_r2:
    fig = px.scatter(x=fitted_train, y=resid_dev, opacity=0.4,
                      labels={'x': 'Predicted P', 'y': 'Deviance Residual'})
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    fig.update_layout(title="Deviance Residual vs Predicted (Train)", height=400)
    st.plotly_chart(fig, use_container_width=True)

with tab_r3:
    try:
        infl = model.get_influence()
        cooks = _arr(infl.cooks_distance[0])
        cook_th = 4 / len(y_train)
        fig = px.scatter(x=list(range(len(cooks))), y=cooks, opacity=0.5,
                          labels={'x': 'Obs', 'y': "Cook's D"})
        fig.add_hline(y=cook_th, line_dash="dash", line_color="red")
        fig.update_layout(title=f"Cook's Distance (threshold={cook_th:.4f})", height=400)
        st.plotly_chart(fig, use_container_width=True)
    except:
        st.info("Cook's distance tidak tersedia.")

# ============================
# 12. MARGINAL EFFECTS
# ============================
if primary_link == 'Logit':
    st.header("12. Marginal Effects (AME)")
    try:
        logit_model = Logit(y_train, X_train).fit(disp=0)
        mfx = logit_model.get_margeff(at='overall')
        mfx_params = _arr(mfx.margeff)
        mfx_se = _arr(mfx.margeff_se)
        mfx_z = mfx_params / mfx_se
        mfx_p = 2 * (1 - stats.norm.cdf(np.abs(mfx_z)))
        mfx_df = pd.DataFrame({
            'Variabel': indep_vars, 'dy/dx (AME)': mfx_params.round(6),
            'Std. Error': mfx_se.round(6), 'z': mfx_z.round(4), 'p-value': mfx_p.round(6)
        })
        st.dataframe(mfx_df, use_container_width=True, hide_index=True)
        fig_mfx = px.bar(mfx_df, x='Variabel', y='dy/dx (AME)', error_y=1.96*mfx_df['Std. Error'],
                          title="Average Marginal Effects ± 95% CI")
        fig_mfx.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig_mfx, use_container_width=True)
    except Exception as e:
        st.warning(f"AME gagal: {e}")

# ============================
# 13. THRESHOLD ANALYSIS
# ============================
st.header("13. Analisis Threshold (Test Set)")
thresholds_an = np.linspace(0.05, 0.95, 19)
thresh_rows = []
for t in thresholds_an:
    yp = (pred_test >= t).astype(int)
    m = calc_metrics(y_test, yp, pred_test)
    thresh_rows.append({'Threshold': round(t, 2), 'Accuracy': round(m['Accuracy'], 4),
                        'Sensitivity': round(m['Sensitivity'], 4), 'Specificity': round(m['Specificity'], 4),
                        'Precision': round(m['Precision'], 4), 'F1': round(m['F1'], 4)})
thresh_df = pd.DataFrame(thresh_rows)

fig_th = go.Figure()
for col in ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'F1']:
    fig_th.add_trace(go.Scatter(x=thresh_df['Threshold'], y=thresh_df[col], mode='lines+markers', name=col))
fig_th.update_layout(title="Metrik vs Threshold (Test Set)", height=450,
                      xaxis_title="Threshold", yaxis_title="Nilai")
st.plotly_chart(fig_th, use_container_width=True)

# ============================
# 14. PREDIKSI MANUAL
# ============================
st.header("14. Prediksi Manual")
pred_inputs = {}
cols_pred = st.columns(min(4, len(indep_vars)))
for i, v in enumerate(indep_vars):
    with cols_pred[i % len(cols_pred)]:
        pred_inputs[v] = st.number_input(v, value=float(panel[v].median()), format="%.4f", key=f'pred_{v}')

x_new = np.array([1.0] + [pred_inputs[v] for v in indep_vars])
link_obj_p = LINK_MAP[primary_link]
eta_new = np.dot(params, x_new)
p_new = float(link_obj_p.inverse(eta_new))

pred_r = pd.DataFrame({
    'Metrik': ['Linear Predictor (η)', 'P(Y=1)', f'Kelas (threshold={threshold})', 'Link Function'],
    'Nilai': [round(eta_new, 4), round(p_new, 4),
              f'Y=1' if p_new >= threshold else 'Y=0', primary_link]
})
st.dataframe(pred_r, use_container_width=True, hide_index=True)

# ============================
# 15. DISTRIBUSI PROBABILITAS
# ============================
st.header("15. Distribusi Probabilitas Prediksi (Test Set)")
fig_dist = go.Figure()
fig_dist.add_trace(go.Histogram(x=pred_test[y_test == 0], nbinsx=30, opacity=0.6,
                                  name='Y=0', marker_color='steelblue'))
fig_dist.add_trace(go.Histogram(x=pred_test[y_test == 1], nbinsx=30, opacity=0.6,
                                  name='Y=1', marker_color='salmon'))
fig_dist.add_vline(x=threshold, line_dash="dash", line_color="black",
                    annotation_text=f"Threshold={threshold}")
fig_dist.update_layout(title="Distribusi P(Y=1) per Kelas (Test Set)", barmode='overlay', height=450)
st.plotly_chart(fig_dist, use_container_width=True)

# ============================
# 16. EXPORT
# ============================
st.header("16. Ekspor Hasil")
col_e1, col_e2, col_e3 = st.columns(3)

with col_e1:
    lines = [
        "="*70, "REGRESI LOGISTIK BINER — EXTENDED", "="*70,
        f"Link Function  : {primary_link}", f"Dependen (Y)   : {dep_var}",
        f"Independen (X) : {', '.join(indep_vars)}",
        f"N total        : {N_all}", f"Train          : {len(y_train)} | Test: {len(y_test)}",
        f"K-Fold CV      : {k_folds}", f"Alpha          : {alpha}", f"Threshold      : {threshold}",
        "", "="*70, "TEST SET PERFORMANCE", "="*70,
        f"AUC            : {auc_val:.4f}",
        f"Accuracy       : {test_metrics['Accuracy']:.4f}",
        f"Sensitivity    : {test_metrics['Sensitivity']:.4f}",
        f"Specificity    : {test_metrics['Specificity']:.4f}",
        f"F1-Score       : {test_metrics['F1']:.4f}"]
    if len(fold_auc_list) > 0:
        lines += ["", f"CV Mean AUC    : {np.mean(fold_auc_list):.4f} ± {np.std(fold_auc_list):.4f}",
                   f"CV Mean Acc    : {np.mean(fold_acc_list):.4f} ± {np.std(fold_acc_list):.4f}"]
    if len(model_results) > 1:
        lines += ["", "="*70, "PERBANDINGAN LINK FUNCTIONS", "="*70]
        for lk, res in model_results.items():
            lines.append(f"  {lk:35s} | AIC={res['aic']:.4f} | Test AUC={res['test_m']['AUC']:.4f}")
    st.download_button("📥 Summary (TXT)", data="\n".join(lines),
                       file_name="logistic_extended_summary.txt", mime="text/plain")

with col_e2:
    export_df = panel.copy()
    export_df['Split'] = 'N/A'
    export_df.loc[train_idx, 'Split'] = 'Train'
    export_df.loc[test_idx, 'Split'] = 'Test'
    all_pred = _arr(model.predict(X_all_const))
    export_df['Pred_Prob'] = all_pred.round(6)
    export_df['Pred_Class'] = (all_pred >= threshold).astype(int)
    st.download_button("📥 Data + Prediksi (CSV)", data=export_df.to_csv(index=False),
                       file_name="logistic_data_predictions.csv", mime="text/csv")

with col_e3:
    cv_export = cv_df.copy()
    st.download_button("📥 CV Results (CSV)", data=cv_export.to_csv(index=False),
                       file_name="logistic_cv_results.csv", mime="text/csv")

# FOOTER
st.markdown("---")
st.markdown("""
**Catatan Metodologis:**

**Link Functions:**
| Link | Formula | Sifat |
|------|---------|-------|
| Logit | g(μ) = log(μ/(1-μ)) | Simetris, OR interpretable |
| Probit | g(μ) = Φ⁻¹(μ) | Simetris, berbasis normal |
| CLogLog | g(μ) = log(-log(1-μ)) | Asimetris, survival/rare events |
| Log-Log | g(μ) = -log(-log(μ)) | Asimetris (kebalikan CLogLog) |
| Cauchy | g(μ) = tan(π(μ-0.5)) | Heavy-tailed, simetris |
| Log | g(μ) = log(μ) | Risk ratio interpretable |

**Validasi:**
- **Train-Test Split:** Stratified — mempertahankan proporsi kelas.
- **K-Fold CV:** Stratified K-Fold — setiap fold memiliki proporsi Y yang sama.
- **Metrik:** AUC, Accuracy, Sensitivity, Specificity, Precision, F1 dihitung pada test set.

**Asumsi:** (1) Observasi independen, (2) Tidak multikolinieritas, (3) Linieritas dalam logit (Box-Tidwell), (4) Sampel cukup (EPP ≥ 10).
""")
st.markdown("Dibangun dengan **Streamlit** + **Statsmodels** + **SciPy** + **Plotly** | Python")
