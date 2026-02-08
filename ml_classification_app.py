import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings, io, json, time, inspect, math, pickle, sys
from datetime import datetime

from sklearn.preprocessing import (StandardScaler, MinMaxScaler, RobustScaler,
    MaxAbsScaler, LabelEncoder, OrdinalEncoder, PowerTransformer, QuantileTransformer,
    label_binarize, PolynomialFeatures, KBinsDiscretizer)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import (SelectKBest, f_classif, mutual_info_classif,
    chi2, RFE, VarianceThreshold, SelectFromModel)
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA_decomp
from sklearn.inspection import permutation_importance

from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier, Perceptron, PassiveAggressiveClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
    AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier,
    HistGradientBoostingClassifier, VotingClassifier, StackingClassifier)
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, ComplementNB
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

from sklearn.model_selection import (train_test_split, cross_val_score, cross_val_predict,
    StratifiedKFold, GridSearchCV, RandomizedSearchCV, learning_curve)
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve, auc,
    precision_recall_curve, average_precision_score, matthews_corrcoef,
    cohen_kappa_score, balanced_accuracy_score, log_loss)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.pipeline import Pipeline
from scipy.stats import binomtest, ks_2samp

try:
    from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, RandomOverSampler
    from imblearn.under_sampling import RandomUnderSampler, TomekLinks, NearMiss
    from imblearn.combine import SMOTETomek, SMOTEENN
    HAS_IMBLEARN=True
except ImportError: HAS_IMBLEARN=False
try:
    from xgboost import XGBClassifier; HAS_XGB=True
except ImportError: HAS_XGB=False
try:
    from lightgbm import LGBMClassifier; HAS_LGBM=True
except ImportError: HAS_LGBM=False
try:
    from catboost import CatBoostClassifier; HAS_CAT=True
except ImportError: HAS_CAT=False
try:
    import shap; HAS_SHAP=True
except ImportError: HAS_SHAP=False

warnings.filterwarnings("ignore")
st.set_page_config(page_title="ML Classification Suite",page_icon="\U0001f916",layout="wide")
st.markdown("""<style>
.main-header{font-size:2.5rem;font-weight:700;background:linear-gradient(135deg,#8E2DE2,#4A00E0);-webkit-background-clip:text;-webkit-text-fill-color:transparent;text-align:center;padding:1rem 0}
.sub-header{font-size:1.1rem;color:#5D6D7E;text-align:center;margin-bottom:2rem}
.success-box{background:#D5F5E3;border-left:5px solid #27AE60;padding:1rem;border-radius:5px;margin:.5rem 0}.success-box,.success-box *{color:#145a32!important}
.metric-card{background:linear-gradient(135deg,#667eea,#764ba2);padding:1.5rem;border-radius:10px;text-align:center;color:white!important;margin:.5rem 0}.metric-card *{color:white!important}
.info-box{background:#D6EAF8;border-left:5px solid #2E86C1;padding:1rem;border-radius:5px;margin:.5rem 0}.info-box,.info-box *{color:#1a4971!important}
.warn-box{background:#FEF9E7;border-left:5px solid #F39C12;padding:1rem;border-radius:5px;margin:.5rem 0}.warn-box,.warn-box *{color:#7d6608!important}
</style>""",unsafe_allow_html=True)

defaults={"data":None,"X":None,"y":None,"X_train":None,"X_test":None,"y_train":None,"y_test":None,
    "feature_names":[],"target_name":"","class_names":[],"preprocessing_done":False,
    "model_results":{},"comparison_df":None,"all_predictions":{},"hp_configs":{},"trained_models":{},
    "scaler_obj":None,"label_encoder_obj":None}
for k,v in defaults.items():
    if k not in st.session_state: st.session_state[k]=v

def safe_metric(y_true, y_pred, y_prob, avg="weighted", n_classes=2):
    m={}
    m["accuracy"]=accuracy_score(y_true,y_pred)
    m["balanced_accuracy"]=balanced_accuracy_score(y_true,y_pred)
    m["precision"]=precision_score(y_true,y_pred,average=avg,zero_division=0)
    m["recall"]=recall_score(y_true,y_pred,average=avg,zero_division=0)
    m["f1"]=f1_score(y_true,y_pred,average=avg,zero_division=0)
    m["mcc"]=matthews_corrcoef(y_true,y_pred)
    m["kappa"]=cohen_kappa_score(y_true,y_pred)
    if y_prob is not None:
        try:
            if n_classes==2: m["roc_auc"]=roc_auc_score(y_true,y_prob[:,1])
            else: m["roc_auc"]=roc_auc_score(y_true,y_prob,multi_class="ovr",average=avg)
        except: m["roc_auc"]=float("nan")
        try: m["log_loss"]=log_loss(y_true,y_prob)
        except: m["log_loss"]=float("nan")
    else: m["roc_auc"]=float("nan");m["log_loss"]=float("nan")
    for a2 in ["macro","micro"]:
        m[f"precision_{a2}"]=precision_score(y_true,y_pred,average=a2,zero_division=0)
        m[f"recall_{a2}"]=recall_score(y_true,y_pred,average=a2,zero_division=0)
        m[f"f1_{a2}"]=f1_score(y_true,y_pred,average=a2,zero_division=0)
    return m

def parse_hidden_layers(s):
    s=str(s).strip().replace("(","").replace(")","").replace(" ","")
    return tuple(int(x) for x in s.split(",") if x.strip())

def render_tuning_widget(hp_name, hp_cfg, model_name, col):
    key=f"tune_{model_name}_{hp_name}";t=hp_cfg["type"]
    with col:
        if t=="select":
            opts=hp_cfg["options"];dv=hp_cfg["default"];defaults=[dv] if dv in opts else [opts[0]]
            chosen=st.multiselect(f"{hp_name}",opts,default=defaults,key=key)
            if not chosen: chosen=[dv if dv in opts else opts[0]]
            if hp_name=="hidden_layer_sizes": return [parse_hidden_layers(o) for o in chosen]
            return chosen
        elif t=="int":
            st.write(f"**{hp_name}** ({hp_cfg['min']}–{hp_cfg['max']})")
            vals_str=st.text_input("Values:",value=str(hp_cfg["default"]),key=key)
            try: return sorted(set([int(x.strip()) for x in vals_str.split(",") if x.strip()]))
            except: return [hp_cfg["default"]]
        elif t=="int_none":
            st.write(f"**{hp_name}** ({hp_cfg['min']}–{hp_cfg['max']}, None ok)")
            vals_str=st.text_input("Values:",value="None",key=key)
            r=[]
            for x in vals_str.split(","):
                x=x.strip()
                if x.lower()=="none": r.append(None)
                else:
                    try: r.append(int(x))
                    except: pass
            return r if r else [None]
        elif t in ["float","float_log"]:
            st.write(f"**{hp_name}** ({hp_cfg['min']}–{hp_cfg['max']})")
            vals_str=st.text_input("Values:",value=str(round(hp_cfg["default"],6)),key=key)
            try: return sorted(set([float(x.strip()) for x in vals_str.split(",") if x.strip()]))
            except: return [hp_cfg["default"]]

def render_hp_widget(hp_name, hp_cfg, model_name, col):
    key=f"hp_{model_name}_{hp_name}";t=hp_cfg["type"]
    with col:
        if t=="int": return st.number_input(hp_name,min_value=hp_cfg["min"],max_value=hp_cfg["max"],value=hp_cfg["default"],step=1,key=key)
        elif t=="int_none":
            if st.checkbox(f"{hp_name}=None",value=hp_cfg["default"] is None,key=key+"_n"): return None
            return st.number_input(hp_name,min_value=hp_cfg["min"],max_value=hp_cfg["max"],value=hp_cfg.get("default",10) or 10,step=1,key=key)
        elif t=="float": return st.number_input(hp_name,min_value=float(hp_cfg["min"]),max_value=float(hp_cfg["max"]),value=float(hp_cfg["default"]),format="%.4f",key=key)
        elif t=="float_log":
            lo=math.log10(max(hp_cfg["min"],1e-15));hi=math.log10(max(hp_cfg["max"],1e-15));df=math.log10(max(hp_cfg["default"],1e-15))
            return 10**st.slider(hp_name,lo,hi,df,step=0.1,format="%.2f",key=key)
        elif t=="select":
            opts=hp_cfg["options"];dv=hp_cfg["default"];idx=opts.index(dv) if dv in opts else 0
            return st.selectbox(hp_name,opts,index=idx,key=key)

def calc_vif(X, names):
    from numpy.linalg import lstsq
    vifs=[]
    for i in range(X.shape[1]):
        y_v=X[:,i];X_v=np.delete(X,i,axis=1)
        X_v=np.column_stack([X_v,np.ones(X_v.shape[0])])
        coef,_,_,_=lstsq(X_v,y_v,rcond=None)
        pred=X_v@coef;ss_res=np.sum((y_v-pred)**2);ss_tot=np.sum((y_v-y_v.mean())**2)
        r2=1-ss_res/(ss_tot+1e-15);vifs.append(1/(1-r2+1e-15))
    return pd.DataFrame({"Feature":names,"VIF":vifs}).sort_values("VIF",ascending=False)

def generate_pipeline_code(preprocessing_info, model_name, best_params):
    code_lines=["import pandas as pd","import numpy as np","from sklearn.model_selection import train_test_split",""]
    code_lines.append("# Load data")
    code_lines.append('df = pd.read_csv("your_data.csv")')
    code_lines.append(f'target = "{preprocessing_info.get("target","target")}"')
    code_lines.append("X = df.drop(columns=[target])")
    code_lines.append("y = df[target]")
    code_lines.append("")
    code_lines.append("# Train-test split")
    code_lines.append("X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)")
    code_lines.append("")
    if preprocessing_info.get("scaler"):
        code_lines.append(f'from sklearn.preprocessing import {preprocessing_info["scaler"]}')
        code_lines.append(f'scaler = {preprocessing_info["scaler"]}()')
        code_lines.append("X_train = scaler.fit_transform(X_train)")
        code_lines.append("X_test = scaler.transform(X_test)")
        code_lines.append("")
    code_lines.append(f"# Model: {model_name}")
    bp_str=", ".join([f"{k}={repr(v)}" for k,v in best_params.items()]) if isinstance(best_params,dict) else ""
    code_lines.append(f"from sklearn.* import *  # adjust import")
    code_lines.append(f"model = {model_name}({bp_str})")
    code_lines.append("model.fit(X_train, y_train)")
    code_lines.append("y_pred = model.predict(X_test)")
    code_lines.append("")
    code_lines.append("from sklearn.metrics import accuracy_score, classification_report")
    code_lines.append('print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")')
    code_lines.append("print(classification_report(y_test, y_pred))")
    return "\n".join(code_lines)

def get_model_registry():
    R={}
    R["Logistic Regression"]={"class":LogisticRegression,"default":{"C":1.0,"max_iter":1000,"solver":"lbfgs","penalty":"l2","random_state":42},"hp":{"C":{"type":"float_log","min":0.001,"max":100.0,"default":1.0},"max_iter":{"type":"int","min":100,"max":5000,"default":1000},"solver":{"type":"select","options":["lbfgs","liblinear","newton-cg","saga"],"default":"lbfgs"},"penalty":{"type":"select","options":["l1","l2","elasticnet","none"],"default":"l2"}}}
    R["Decision Tree"]={"class":DecisionTreeClassifier,"default":{"max_depth":None,"min_samples_split":2,"criterion":"gini","random_state":42},"hp":{"max_depth":{"type":"int_none","min":1,"max":50,"default":None},"min_samples_split":{"type":"int","min":2,"max":50,"default":2},"min_samples_leaf":{"type":"int","min":1,"max":50,"default":1},"criterion":{"type":"select","options":["gini","entropy","log_loss"],"default":"gini"}}}
    R["Random Forest"]={"class":RandomForestClassifier,"default":{"n_estimators":100,"max_depth":None,"random_state":42},"hp":{"n_estimators":{"type":"int","min":10,"max":1000,"default":100},"max_depth":{"type":"int_none","min":1,"max":50,"default":None},"min_samples_split":{"type":"int","min":2,"max":50,"default":2},"criterion":{"type":"select","options":["gini","entropy","log_loss"],"default":"gini"}}}
    R["Extra Trees"]={"class":ExtraTreesClassifier,"default":{"n_estimators":100,"random_state":42},"hp":{"n_estimators":{"type":"int","min":10,"max":1000,"default":100},"max_depth":{"type":"int_none","min":1,"max":50,"default":None},"criterion":{"type":"select","options":["gini","entropy","log_loss"],"default":"gini"}}}
    R["Gradient Boosting"]={"class":GradientBoostingClassifier,"default":{"n_estimators":100,"learning_rate":0.1,"max_depth":3,"random_state":42},"hp":{"n_estimators":{"type":"int","min":10,"max":1000,"default":100},"learning_rate":{"type":"float_log","min":0.001,"max":1.0,"default":0.1},"max_depth":{"type":"int","min":1,"max":20,"default":3},"subsample":{"type":"float","min":0.5,"max":1.0,"default":1.0}}}
    R["Hist Gradient Boosting"]={"class":HistGradientBoostingClassifier,"default":{"max_iter":100,"learning_rate":0.1,"random_state":42},"hp":{"max_iter":{"type":"int","min":10,"max":1000,"default":100},"learning_rate":{"type":"float_log","min":0.001,"max":1.0,"default":0.1},"max_depth":{"type":"int_none","min":1,"max":50,"default":None},"min_samples_leaf":{"type":"int","min":1,"max":100,"default":20}}}
    R["AdaBoost"]={"class":AdaBoostClassifier,"default":{"n_estimators":50,"learning_rate":1.0,"random_state":42},"hp":{"n_estimators":{"type":"int","min":10,"max":500,"default":50},"learning_rate":{"type":"float_log","min":0.01,"max":2.0,"default":1.0}}}
    R["Bagging"]={"class":BaggingClassifier,"default":{"n_estimators":10,"random_state":42},"hp":{"n_estimators":{"type":"int","min":5,"max":200,"default":10},"max_samples":{"type":"float","min":0.1,"max":1.0,"default":1.0}}}
    R["SVM (RBF)"]={"class":SVC,"default":{"C":1.0,"kernel":"rbf","gamma":"scale","probability":True,"random_state":42},"hp":{"C":{"type":"float_log","min":0.001,"max":100.0,"default":1.0},"gamma":{"type":"select","options":["scale","auto"],"default":"scale"},"kernel":{"type":"select","options":["rbf","linear","poly","sigmoid"],"default":"rbf"}}}
    R["Linear SVM"]={"class":LinearSVC,"default":{"C":1.0,"max_iter":10000,"random_state":42},"hp":{"C":{"type":"float_log","min":0.001,"max":100.0,"default":1.0},"loss":{"type":"select","options":["hinge","squared_hinge"],"default":"squared_hinge"}}}
    R["Nu-SVM"]={"class":NuSVC,"default":{"nu":0.5,"kernel":"rbf","probability":True,"random_state":42},"hp":{"nu":{"type":"float","min":0.01,"max":0.99,"default":0.5},"kernel":{"type":"select","options":["rbf","linear","poly","sigmoid"],"default":"rbf"}}}
    R["KNN"]={"class":KNeighborsClassifier,"default":{"n_neighbors":5},"hp":{"n_neighbors":{"type":"int","min":1,"max":50,"default":5},"weights":{"type":"select","options":["uniform","distance"],"default":"uniform"},"metric":{"type":"select","options":["minkowski","euclidean","manhattan"],"default":"minkowski"}}}
    R["Gaussian NB"]={"class":GaussianNB,"default":{"var_smoothing":1e-9},"hp":{"var_smoothing":{"type":"float_log","min":1e-12,"max":1e-1,"default":1e-9}}}
    R["Bernoulli NB"]={"class":BernoulliNB,"default":{"alpha":1.0},"hp":{"alpha":{"type":"float_log","min":0.001,"max":100.0,"default":1.0}}}
    R["Complement NB"]={"class":ComplementNB,"default":{"alpha":1.0},"hp":{"alpha":{"type":"float_log","min":0.001,"max":100.0,"default":1.0}}}
    R["MLP Neural Net"]={"class":MLPClassifier,"default":{"hidden_layer_sizes":(100,),"max_iter":500,"random_state":42},"hp":{"hidden_layer_sizes":{"type":"select","options":["(50,)","(100,)","(100,50)","(128,64)","(128,64,32)","(256,128,64)"],"default":"(100,)"},"activation":{"type":"select","options":["relu","tanh","logistic"],"default":"relu"},"solver":{"type":"select","options":["adam","sgd","lbfgs"],"default":"adam"},"alpha":{"type":"float_log","min":1e-6,"max":1.0,"default":0.0001},"max_iter":{"type":"int","min":100,"max":5000,"default":500}}}
    R["LDA"]={"class":LinearDiscriminantAnalysis,"default":{"solver":"svd"},"hp":{"solver":{"type":"select","options":["svd","lsqr","eigen"],"default":"svd"}}}
    R["QDA"]={"class":QuadraticDiscriminantAnalysis,"default":{"reg_param":0.0},"hp":{"reg_param":{"type":"float","min":0.0,"max":1.0,"default":0.0}}}
    R["Ridge Classifier"]={"class":RidgeClassifier,"default":{"alpha":1.0,"random_state":42},"hp":{"alpha":{"type":"float_log","min":0.001,"max":100.0,"default":1.0}}}
    R["SGD Classifier"]={"class":SGDClassifier,"default":{"loss":"hinge","max_iter":1000,"random_state":42},"hp":{"loss":{"type":"select","options":["hinge","log_loss","modified_huber","perceptron"],"default":"hinge"},"alpha":{"type":"float_log","min":1e-6,"max":1.0,"default":0.0001},"penalty":{"type":"select","options":["l1","l2","elasticnet"],"default":"l2"}}}
    R["Perceptron"]={"class":Perceptron,"default":{"max_iter":1000,"random_state":42},"hp":{"alpha":{"type":"float_log","min":1e-6,"max":1.0,"default":0.0001},"max_iter":{"type":"int","min":100,"max":5000,"default":1000}}}
    R["Passive Aggressive"]={"class":PassiveAggressiveClassifier,"default":{"C":1.0,"max_iter":1000,"random_state":42},"hp":{"C":{"type":"float_log","min":0.001,"max":100.0,"default":1.0}}}
    if HAS_XGB: R["XGBoost"]={"class":XGBClassifier,"default":{"n_estimators":100,"learning_rate":0.1,"max_depth":6,"random_state":42,"use_label_encoder":False,"eval_metric":"logloss"},"hp":{"n_estimators":{"type":"int","min":10,"max":1000,"default":100},"learning_rate":{"type":"float_log","min":0.001,"max":1.0,"default":0.1},"max_depth":{"type":"int","min":1,"max":20,"default":6},"subsample":{"type":"float","min":0.5,"max":1.0,"default":1.0}}}
    if HAS_LGBM: R["LightGBM"]={"class":LGBMClassifier,"default":{"n_estimators":100,"learning_rate":0.1,"random_state":42,"verbose":-1},"hp":{"n_estimators":{"type":"int","min":10,"max":1000,"default":100},"learning_rate":{"type":"float_log","min":0.001,"max":1.0,"default":0.1},"max_depth":{"type":"int","min":-1,"max":50,"default":-1},"num_leaves":{"type":"int","min":10,"max":200,"default":31}}}
    if HAS_CAT: R["CatBoost"]={"class":CatBoostClassifier,"default":{"iterations":100,"learning_rate":0.1,"depth":6,"random_state":42,"verbose":0},"hp":{"iterations":{"type":"int","min":10,"max":1000,"default":100},"learning_rate":{"type":"float_log","min":0.001,"max":1.0,"default":0.1},"depth":{"type":"int","min":1,"max":16,"default":6}}}
    return R
MODEL_REGISTRY=get_model_registry()

st.markdown('<div class="main-header">\U0001f916 ML Classification Suite v3.0</div>',unsafe_allow_html=True)
st.markdown('<div class="sub-header">Complete ML Pipeline \u2022 25 Models \u2022 AutoTuning \u2022 SHAP \u2022 Prediction</div>',unsafe_allow_html=True)
with st.sidebar:
    st.markdown("## \U0001f4c2 Data Input")
    data_src=st.radio("Source:",["Upload CSV/Excel","Demo Dataset"])
    if data_src=="Demo Dataset":
        demo_name=st.selectbox("Dataset:",["Iris","Wine","Breast Cancer","Digits"])
        from sklearn.datasets import load_iris,load_wine,load_breast_cancer,load_digits
        d={"Iris":load_iris,"Wine":load_wine,"Breast Cancer":load_breast_cancer,"Digits":load_digits}[demo_name]()
        demo_df=pd.DataFrame(d.data,columns=d.feature_names);demo_df["target"]=d.target
        st.session_state.data=demo_df;st.session_state.class_names=[str(c) for c in d.target_names]
    else:
        up=st.file_uploader("Upload:",type=["csv","xlsx","xls"])
        if up:
            try: st.session_state.data=pd.read_csv(up) if up.name.endswith(".csv") else pd.read_excel(up)
            except Exception as e: st.error(str(e))
    if st.session_state.data is not None:
        dfx=st.session_state.data;all_cols=dfx.columns.tolist()
        st.markdown("---");st.markdown("## \U0001f3af Target")
        target_col=st.selectbox("Target:",all_cols,index=len(all_cols)-1);st.session_state.target_name=target_col
        feat_cols=[c for c in all_cols if c!=target_col]
        selected_feats=st.multiselect("Features:",feat_cols,default=feat_cols);st.session_state.feature_names=selected_feats
        st.markdown("---");st.markdown("## Split")
        test_size=st.slider("Test %:",0.1,0.5,0.2,0.05)
        random_state=st.number_input("Seed:",0,9999,42)
        stratify_split=st.checkbox("Stratified",value=True)
if st.session_state.data is None: st.info("Upload data or select demo.");st.stop()
df=st.session_state.data.copy()
tab1,tab2,tab3,tab4,tab5,tab6,tab7,tab8=st.tabs(["\U0001f4ca EDA","\U0001f527 Preprocess","\U0001f916 Training","\U0001f4c8 Results","\U0001f3c6 Compare","\U0001f50d Analysis","\U0001f52e Predict","\U0001f4be Export"])

# ===================== TAB 1: EDA + VIF =====================
with tab1:
    st.markdown("## Data Overview")
    tc=st.session_state.target_name;fs=st.session_state.feature_names
    c1,c2,c3,c4,c5=st.columns(5)
    with c1: st.metric("Samples",len(df))
    with c2: st.metric("Features",len(fs))
    with c3: st.metric("Classes",df[tc].nunique())
    with c4: st.metric("Missing",int(df[fs].isnull().sum().sum()))
    with c5: st.metric("Duplicates",int(df.duplicated().sum()))
    st.dataframe(df.head(20),use_container_width=True)
    nf=[c for c in fs if df[c].dtype in ["int64","float64","int32","float32"]];cf=[c for c in fs if c not in nf]
    if nf: st.markdown("### Statistics");st.dataframe(df[nf].describe().round(4),use_container_width=True)
    st.markdown("### Target Distribution")
    tvc=df[tc].value_counts().reset_index();tvc.columns=["Class","Count"]
    co=st.columns(2)
    with co[0]: st.plotly_chart(px.bar(tvc,x="Class",y="Count",color="Class").update_layout(height=400),use_container_width=True)
    with co[1]: st.plotly_chart(px.pie(tvc,names="Class",values="Count").update_layout(height=400),use_container_width=True)
    if len(nf)>=2:
        st.markdown("### Correlation Heatmap")
        st.plotly_chart(px.imshow(df[nf].corr(),text_auto=".2f",color_continuous_scale="RdBu_r",zmin=-1,zmax=1,aspect="auto").update_layout(height=500),use_container_width=True)
    # VIF
    if len(nf)>=2:
        st.markdown("### Multicollinearity (VIF)")
        if st.button("Compute VIF",key="vif_btn"):
            try:
                X_vif=df[nf].dropna().values;vdf=calc_vif(X_vif,nf)
                st.dataframe(vdf.round(2),use_container_width=True)
                high_vif=vdf[vdf["VIF"]>10]
                if len(high_vif)>0: st.markdown(f'<div class="warn-box">\u26a0\ufe0f {len(high_vif)} features with VIF>10 (high multicollinearity)</div>',unsafe_allow_html=True)
                else: st.markdown('<div class="success-box">\u2705 No severe multicollinearity detected</div>',unsafe_allow_html=True)
            except Exception as e: st.warning(str(e))
    if nf:
        st.markdown("### Feature Distributions")
        sel_f=st.multiselect("Plot:",nf,default=nf[:min(6,len(nf))])
        if sel_f:
            nc2=min(3,len(sel_f))
            for ri in range((len(sel_f)+nc2-1)//nc2):
                cols=st.columns(nc2)
                for ci in range(nc2):
                    idx=ri*nc2+ci
                    if idx<len(sel_f):
                        with cols[ci]: st.plotly_chart(px.histogram(df,x=sel_f[idx],color=tc,marginal="box",barmode="overlay",opacity=0.7,title=sel_f[idx]).update_layout(height=350),use_container_width=True)

# ===================== TAB 2: PREPROCESSING + AUTO FE =====================
with tab2:
    st.markdown("## Preprocessing Pipeline")
    tc=st.session_state.target_name;fs=st.session_state.feature_names
    nf=[c for c in fs if df[c].dtype in ["int64","float64","int32","float32"]];cf=[c for c in fs if c not in nf]
    dup_act=st.radio("1. Duplicates:",["Keep","Remove"],horizontal=True)
    miss_num=st.selectbox("2. Numeric missing:",["None","Mean","Median","Most frequent","Constant(0)","KNN Imputer"])
    miss_cat=st.selectbox("3. Cat missing:",["None","Most frequent","Constant(unknown)"])
    enc_m=st.selectbox("4. Encoding:",["None","Label Encoding","One-Hot","Ordinal"],index=0 if not cf else 1)
    out_m=st.selectbox("5. Outliers:",["None","IQR clip","Z-score clip"])
    scl_m=st.selectbox("6. Scaling:",["None","StandardScaler","MinMaxScaler","RobustScaler","MaxAbsScaler","PowerTransformer","QuantileTransformer"])
    st.markdown("### 7. Auto Feature Engineering")
    fe_poly=st.checkbox("Polynomial Features (degree=2)",value=False)
    fe_bins=st.checkbox("Binning (KBins, 5 bins)",value=False)
    fs_m=st.selectbox("8. Feature Selection:",["None","Variance Threshold","SelectKBest (ANOVA)","SelectKBest (MI)","SelectKBest (Chi2)","RFE (RF)","SelectFromModel (RF)"])
    fs_k=st.slider("N features:",1,max(1,len(fs)*3 if fe_poly else len(fs)),min(len(fs),50),key="fsk") if fs_m!="None" else 999
    dr_m=st.selectbox("9. Dim Reduction:",["None","PCA","Truncated SVD","LDA"])
    dr_n=st.slider("N components:",1,max(1,len(fs)),min(2,len(fs)),key="drn") if dr_m!="None" else 0
    st.markdown("### 10. Class Imbalance")
    if HAS_IMBLEARN: imb_m=st.selectbox("Resampling:",["None","SMOTE","ADASYN","BorderlineSMOTE","RandomOverSampler","RandomUnderSampler","TomekLinks","SMOTETomek","SMOTEENN"])
    else: imb_m="None";st.info("pip install imbalanced-learn")
    if st.button("\U0001f680 Apply",type="primary",use_container_width=True):
        with st.spinner("Processing..."):
            try:
                pdf=df.copy();log=[];pp_info={"target":tc,"scaler":scl_m if scl_m!="None" else None}
                if dup_act=="Remove": b=len(pdf);pdf=pdf.drop_duplicates();log.append(f"Removed {b-len(pdf)} dupes")
                y_raw=pdf[tc].copy()
                if y_raw.dtype==object or str(y_raw.dtype)=="category":
                    le=LabelEncoder();y_enc=le.fit_transform(y_raw);st.session_state.class_names=[str(c) for c in le.classes_];st.session_state.label_encoder_obj=le
                else: y_enc=y_raw.values;st.session_state.class_names=[str(c) for c in sorted(y_raw.unique())]
                X_df=pdf[fs].copy()
                if miss_num!="None" and nf:
                    nf_in=[c for c in nf if c in X_df.columns]
                    imp={"Mean":SimpleImputer(strategy="mean"),"Median":SimpleImputer(strategy="median"),"Most frequent":SimpleImputer(strategy="most_frequent"),"Constant(0)":SimpleImputer(strategy="constant",fill_value=0),"KNN Imputer":KNNImputer(n_neighbors=5)}[miss_num]
                    X_df[nf_in]=imp.fit_transform(X_df[nf_in]);log.append(f"Imputed: {miss_num}")
                if miss_cat!="None" and cf:
                    cf_in=[c for c in cf if c in X_df.columns]
                    imp_c=SimpleImputer(strategy="most_frequent") if "frequent" in miss_cat else SimpleImputer(strategy="constant",fill_value="unknown")
                    X_df[cf_in]=imp_c.fit_transform(X_df[cf_in])
                if enc_m!="None" and cf:
                    cf_in=[c for c in cf if c in X_df.columns]
                    if enc_m=="Label Encoding":
                        for c in cf_in: X_df[c]=LabelEncoder().fit_transform(X_df[c].astype(str))
                    elif enc_m=="One-Hot": X_df=pd.get_dummies(X_df,columns=cf_in,drop_first=True)
                    elif enc_m=="Ordinal": X_df[cf_in]=OrdinalEncoder(handle_unknown="use_encoded_value",unknown_value=-1).fit_transform(X_df[cf_in].astype(str))
                    log.append(f"Encoded: {enc_m}")
                nf_now=[c for c in X_df.columns if X_df[c].dtype in ["int64","float64","int32","float32"]]
                if out_m=="IQR clip" and nf_now:
                    for c in nf_now: q1,q3=X_df[c].quantile(0.25),X_df[c].quantile(0.75);iqr=q3-q1;X_df[c]=X_df[c].clip(q1-1.5*iqr,q3+1.5*iqr)
                    log.append("IQR clip")
                elif out_m=="Z-score clip" and nf_now:
                    for c in nf_now: mu,sig=X_df[c].mean(),X_df[c].std()+1e-15;X_df[c]=X_df[c].clip(mu-3*sig,mu+3*sig)
                    log.append("Z-score clip")
                X_arr=X_df.values.astype(float);fn_now=list(X_df.columns)
                if scl_m!="None":
                    sc={"StandardScaler":StandardScaler(),"MinMaxScaler":MinMaxScaler(),"RobustScaler":RobustScaler(),"MaxAbsScaler":MaxAbsScaler(),"PowerTransformer":PowerTransformer(),"QuantileTransformer":QuantileTransformer(output_distribution="normal")}[scl_m]
                    X_arr=sc.fit_transform(X_arr);st.session_state.scaler_obj=sc;log.append(f"Scaled: {scl_m}")
                # Auto FE
                if fe_poly:
                    pf=PolynomialFeatures(degree=2,interaction_only=False,include_bias=False)
                    X_arr=pf.fit_transform(X_arr);fn_now=pf.get_feature_names_out(fn_now).tolist();log.append(f"Poly features: {X_arr.shape[1]}")
                if fe_bins and X_arr.shape[1]<=50:
                    kb=KBinsDiscretizer(n_bins=5,encode="ordinal",strategy="quantile")
                    X_binned=kb.fit_transform(X_arr)
                    bin_names=[f"{fn_now[i]}_bin" for i in range(len(fn_now))]
                    X_arr=np.hstack([X_arr,X_binned]);fn_now=fn_now+bin_names;log.append(f"Binned features added: {X_arr.shape[1]}")
                if fs_m!="None":
                    ku=min(fs_k,X_arr.shape[1])
                    if fs_m=="Variance Threshold": sel=VarianceThreshold(threshold=0.01)
                    elif "ANOVA" in fs_m: sel=SelectKBest(f_classif,k=ku)
                    elif "MI" in fs_m: sel=SelectKBest(mutual_info_classif,k=ku)
                    elif "Chi2" in fs_m: X_arr=X_arr-X_arr.min(axis=0);sel=SelectKBest(chi2,k=ku)
                    elif "RFE" in fs_m: sel=RFE(RandomForestClassifier(n_estimators=50,random_state=42),n_features_to_select=ku)
                    else: sel=SelectFromModel(RandomForestClassifier(n_estimators=50,random_state=42),max_features=ku)
                    X_arr=sel.fit_transform(X_arr,y_enc);mask=sel.get_support();fn_now=[fn_now[i] for i in range(len(mask)) if mask[i]]
                    log.append(f"Selected: {X_arr.shape[1]}")
                if dr_m!="None":
                    nc3=min(dr_n,X_arr.shape[1],X_arr.shape[0]-1)
                    if dr_m=="PCA": red=PCA(n_components=nc3)
                    elif dr_m=="Truncated SVD": red=TruncatedSVD(n_components=nc3)
                    else: nc3=min(nc3,len(np.unique(y_enc))-1);red=LDA_decomp(n_components=max(nc3,1))
                    X_arr=red.fit_transform(X_arr,y_enc);fn_now=[f"Comp_{i+1}" for i in range(X_arr.shape[1])]
                    log.append(f"DimRed: {dr_m} -> {X_arr.shape[1]}")
                strat=y_enc if stratify_split else None
                X_train,X_test,y_train,y_test=train_test_split(X_arr,y_enc,test_size=test_size,random_state=random_state,stratify=strat)
                if imb_m!="None" and HAS_IMBLEARN:
                    samplers={"SMOTE":SMOTE(random_state=42),"ADASYN":ADASYN(random_state=42),"BorderlineSMOTE":BorderlineSMOTE(random_state=42),"RandomOverSampler":RandomOverSampler(random_state=42),"RandomUnderSampler":RandomUnderSampler(random_state=42),"TomekLinks":TomekLinks(),"SMOTETomek":SMOTETomek(random_state=42),"SMOTEENN":SMOTEENN(random_state=42)}
                    b4=len(y_train);X_train,y_train=samplers[imb_m].fit_resample(X_train,y_train);log.append(f"Resampled: {imb_m} ({b4}->{len(y_train)})")
                st.session_state.update({"X":X_arr,"y":y_enc,"X_train":X_train,"X_test":X_test,"y_train":y_train,"y_test":y_test,"feature_names":fn_now,"preprocessing_done":True})
                if "pp_info" not in st.session_state: st.session_state["pp_info"]={}
                st.session_state["pp_info"]=pp_info
                st.success("\u2705 Done!")
                for s in log: st.markdown(f"- {s}")
                c1,c2,c3,c4=st.columns(4)
                with c1: st.metric("Train",len(X_train))
                with c2: st.metric("Test",len(X_test))
                with c3: st.metric("Features",X_train.shape[1])
                with c4: st.metric("Classes",len(np.unique(y_enc)))
            except Exception as e: st.error(str(e));import traceback;st.code(traceback.format_exc())

# ===================== TAB 3: TRAINING =====================
with tab3:
    st.markdown("## Model Training")
    if not st.session_state.preprocessing_done: st.warning("Preprocess first.");st.stop()
    all_mn=list(MODEL_REGISTRY.keys())
    qs=st.radio("Quick:",["Custom","All","Tree-based","Linear","Boosting"],horizontal=True)
    if qs=="All": dm=all_mn
    elif qs=="Tree-based": dm=[m for m in all_mn if any(k in m for k in ["Tree","Forest","Extra","Bagging"])]
    elif qs=="Linear": dm=[m for m in all_mn if any(k in m for k in ["Logistic","Ridge","SGD","Perceptron","Passive","LDA","Linear SVM"])]
    elif qs=="Boosting": dm=[m for m in all_mn if any(k in m for k in ["Boost","XGB","LGBM","Cat","Ada"])]
    else: dm=["Logistic Regression","Random Forest","SVM (RBF)","KNN","Gradient Boosting"]
    selected=st.multiselect("Models:",all_mn,default=[m for m in dm if m in all_mn])
    if selected:
        st.markdown("### HP Tuning")
        use_tuning=st.checkbox("Enable HP Tuning",value=False)
        tune_method="none";tune_cv=5;tune_scoring="accuracy";tune_niter=20
        if use_tuning:
            st.markdown('<div class="info-box">\U0001f4a1 Select <b>multiple values</b> per param. Tuner tries all combos!</div>',unsafe_allow_html=True)
            tune_method=st.selectbox("Method:",["RandomizedSearchCV","GridSearchCV"])
            tune_cv=st.slider("CV folds:",2,10,5)
            tune_scoring=st.selectbox("Scoring:",["accuracy","f1_weighted","precision_weighted","recall_weighted","roc_auc_ovr_weighted"])
            if tune_method=="RandomizedSearchCV": tune_niter=st.slider("N iter:",5,200,20)
        tune_grids={}
        for mname in selected:
            with st.expander(f"\u2699\ufe0f {mname}",expanded=False):
                mreg=MODEL_REGISTRY[mname];hp_grid=mreg["hp"]
                hp_names=list(hp_grid.keys());ncols=min(3,len(hp_names));cols=st.columns(ncols)
                if use_tuning:
                    hp_gv={}
                    for i,hp in enumerate(hp_names): hp_gv[hp]=render_tuning_widget(hp,hp_grid[hp],mname,cols[i%ncols])
                    tune_grids[mname]=hp_gv
                    combos=1
                    for v in hp_gv.values(): combos*=max(1,len(v))
                    st.info(f"\U0001f4ca {combos} combos x {tune_cv} folds = {combos*tune_cv} fits")
                else:
                    hp_vals={}
                    for i,hp in enumerate(hp_names): hp_vals[hp]=render_hp_widget(hp,hp_grid[hp],mname,cols[i%ncols])
                    st.session_state.hp_configs[mname]=hp_vals
        eval_cv=st.slider("Eval CV:",2,20,5,key="ecv")
        if st.button("\U0001f680 Train All",type="primary",use_container_width=True):
            X_tr=st.session_state.X_train;X_te=st.session_state.X_test
            y_tr=st.session_state.y_train;y_te=st.session_state.y_test
            n_cls=len(np.unique(y_tr));avg="binary" if n_cls==2 else "weighted"
            results={};all_preds={};trained={};prog=st.progress(0);status=st.empty()
            for mi,mname in enumerate(selected):
                status.text(f"Training {mname} ({mi+1}/{len(selected)})...")
                try:
                    mreg=MODEL_REGISTRY[mname]
                    valid_p=set(inspect.signature(mreg["class"].__init__).parameters.keys())-{"self"}
                    t0=time.time()
                    if use_tuning and tune_method!="none" and mname in tune_grids:
                        pg={hk:hv for hk,hv in tune_grids[mname].items() if hk in valid_p and isinstance(hv,list) and len(hv)>0}
                        base_p={dk:dv for dk,dv in mreg["default"].items() if dk in valid_p}
                        base=mreg["class"](**base_p)
                        if tune_method=="RandomizedSearchCV":
                            tc2=1
                            for v in pg.values(): tc2*=len(v)
                            search=RandomizedSearchCV(base,pg,n_iter=min(tune_niter,tc2),cv=tune_cv,scoring=tune_scoring,random_state=42,n_jobs=-1,error_score=0)
                        else: search=GridSearchCV(base,pg,cv=tune_cv,scoring=tune_scoring,n_jobs=-1,error_score=0)
                        search.fit(X_tr,y_tr);model=search.best_estimator_;best_params=search.best_params_;best_score=search.best_score_
                    else:
                        hp=st.session_state.hp_configs.get(mname,{}).copy()
                        if "hidden_layer_sizes" in hp and isinstance(hp["hidden_layer_sizes"],str): hp["hidden_layer_sizes"]=parse_hidden_layers(hp["hidden_layer_sizes"])
                        hp_clean={k:v for k,v in hp.items() if k in valid_p}
                        for dk,dv in mreg["default"].items():
                            if dk not in hp_clean and dk in valid_p: hp_clean[dk]=dv
                        model=mreg["class"](**hp_clean);model.fit(X_tr,y_tr);best_params=hp_clean;best_score=None
                    train_time=time.time()-t0;y_pred=model.predict(X_te)
                    y_prob=None
                    if hasattr(model,"predict_proba"):
                        try: y_prob=model.predict_proba(X_te)
                        except: pass
                    elif hasattr(model,"decision_function"):
                        try: cal=CalibratedClassifierCV(model,cv=3);cal.fit(X_tr,y_tr);y_prob=cal.predict_proba(X_te)
                        except: pass
                    try: cv_sc=cross_val_score(model,X_tr,y_tr,cv=min(eval_cv,len(np.unique(y_tr))),scoring="accuracy")
                    except: cv_sc=np.array([0.0])
                    metrics=safe_metric(y_te,y_pred,y_prob,avg,n_cls)
                    n_params=sum(getattr(model,a).size for a in dir(model) if hasattr(getattr(model,a,None),"size") and not a.startswith("_") and isinstance(getattr(model,a),np.ndarray))
                    mem=sys.getsizeof(pickle.dumps(model))
                    metrics.update({"cv_mean":cv_sc.mean(),"cv_std":cv_sc.std(),"train_time":train_time,"model_name":mname,
                        "best_params":str(best_params),"tuning_cv_score":best_score if best_score else float("nan"),
                        "n_params":n_params,"model_size_kb":round(mem/1024,1)})
                    results[mname]=metrics;all_preds[mname]={"y_pred":y_pred,"y_prob":y_prob,"model":model,"best_params":best_params};trained[mname]=model
                except Exception as e: results[mname]={"model_name":mname,"accuracy":0,"error":str(e)};st.warning(f"{mname}: {e}")
                prog.progress((mi+1)/len(selected))
            st.session_state.model_results=results;st.session_state.all_predictions=all_preds;st.session_state.trained_models=trained
            comp=pd.DataFrame([v for v in results.values() if v.get("accuracy",0)>0])
            if len(comp)>0: comp=comp.sort_values("accuracy",ascending=False)
            st.session_state.comparison_df=comp;prog.empty();status.empty()
            st.success(f"\u2705 {len(comp)} models trained!")
            if len(comp)>0:
                best=comp.iloc[0]
                st.markdown(f'<div class="success-box"><b>\U0001f3c6 Best: {best["model_name"]}</b> | Acc={best["accuracy"]:.4f} | F1={best.get("f1",0):.4f}</div>',unsafe_allow_html=True)
                st.markdown("### Best Parameters Per Model")
                for mn in comp["model_name"].tolist():
                    bp=all_preds.get(mn,{}).get("best_params",{})
                    ts=results.get(mn,{}).get("tuning_cv_score",float("nan"))
                    icon="\U0001f947" if mn==best["model_name"] else "\U0001f4cb"
                    ts_str=f" | CV={ts:.4f}" if not (isinstance(ts,float) and ts!=ts) else ""
                    st.markdown(f"**{icon} {mn}** (Acc={results[mn]['accuracy']:.4f}){ts_str}")
                    if isinstance(bp,dict): st.json(bp)
                    else: st.code(str(bp))

# ===================== TAB 4: RESULTS + THRESHOLD =====================
with tab4:
    st.markdown("## Detailed Results")
    if not st.session_state.model_results: st.warning("Train models first.");st.stop()
    res=st.session_state.model_results;preds=st.session_state.all_predictions
    mnames=[k for k in res if res[k].get("accuracy",0)>0]
    if not mnames: st.error("No models.");st.stop()
    sel=st.selectbox("Model:",mnames);mr=res[sel];mp=preds.get(sel,{})
    y_te=st.session_state.y_test;cn=st.session_state.class_names
    c1,c2,c3,c4,c5=st.columns(5)
    with c1: st.markdown(f'<div class="metric-card"><h3>Accuracy</h3><h2>{mr["accuracy"]:.4f}</h2></div>',unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="metric-card"><h3>F1</h3><h2>{mr.get("f1",0):.4f}</h2></div>',unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="metric-card"><h3>Precision</h3><h2>{mr.get("precision",0):.4f}</h2></div>',unsafe_allow_html=True)
    with c4: st.markdown(f'<div class="metric-card"><h3>Recall</h3><h2>{mr.get("recall",0):.4f}</h2></div>',unsafe_allow_html=True)
    with c5: st.markdown(f'<div class="metric-card"><h3>MCC</h3><h2>{mr.get("mcc",0):.4f}</h2></div>',unsafe_allow_html=True)
    bp=mp.get("best_params",{})
    if bp:
        st.markdown("### Best Parameters")
        if isinstance(bp,dict): st.json(bp)
        else: st.code(str(bp))
    st.markdown("### All Metrics")
    mkeys=["accuracy","balanced_accuracy","precision","recall","f1","mcc","kappa","roc_auc","log_loss","cv_mean","cv_std","tuning_cv_score","n_params","model_size_kb","train_time"]
    st.dataframe(pd.DataFrame([{"Metric":k,"Value":round(mr.get(k,float("nan")),6)} for k in mkeys if k in mr]),use_container_width=True)
    if "y_pred" in mp:
        cm=confusion_matrix(y_te,mp["y_pred"]);clbl=cn if len(cn)==cm.shape[0] else [str(i) for i in range(cm.shape[0])]
        st.markdown("### Confusion Matrix")
        co=st.columns(2)
        with co[0]: st.plotly_chart(px.imshow(cm,text_auto=True,color_continuous_scale="Blues",x=clbl,y=clbl).update_layout(title="Raw",height=450),use_container_width=True)
        with co[1]:
            cmn=cm.astype(float)/(cm.sum(axis=1,keepdims=True)+1e-15)
            st.plotly_chart(px.imshow(cmn,text_auto=".2%",color_continuous_scale="Purples",x=clbl,y=clbl).update_layout(title="Normalized",height=450),use_container_width=True)
        st.markdown("### Classification Report")
        cr=classification_report(y_te,mp["y_pred"],target_names=clbl,output_dict=True);st.dataframe(pd.DataFrame(cr).T.round(4),use_container_width=True)
        # Threshold Tuning (binary)
        nc_t=len(np.unique(y_te))
        if nc_t==2 and mp.get("y_prob") is not None:
            st.markdown("### Threshold Tuning")
            thresh=st.slider("Decision Threshold:",0.0,1.0,0.5,0.01,key="thresh_sl")
            y_thr=(mp["y_prob"][:,1]>=thresh).astype(int)
            tc1,tc2,tc3,tc4=st.columns(4)
            with tc1: st.metric("Accuracy",f"{accuracy_score(y_te,y_thr):.4f}")
            with tc2: st.metric("Precision",f"{precision_score(y_te,y_thr,zero_division=0):.4f}")
            with tc3: st.metric("Recall",f"{recall_score(y_te,y_thr,zero_division=0):.4f}")
            with tc4: st.metric("F1",f"{f1_score(y_te,y_thr,zero_division=0):.4f}")
            # Threshold sweep chart
            thresholds=np.arange(0.05,1.0,0.05)
            th_data=[]
            for th in thresholds:
                yt=(mp["y_prob"][:,1]>=th).astype(int)
                th_data.append({"Threshold":th,"Precision":precision_score(y_te,yt,zero_division=0),"Recall":recall_score(y_te,yt,zero_division=0),"F1":f1_score(y_te,yt,zero_division=0)})
            thdf=pd.DataFrame(th_data)
            fig_th=go.Figure()
            for m in ["Precision","Recall","F1"]: fig_th.add_trace(go.Scatter(x=thdf["Threshold"],y=thdf[m],mode="lines+markers",name=m))
            fig_th.update_layout(title="Metrics vs Threshold",height=400);st.plotly_chart(fig_th,use_container_width=True)
        if mp.get("y_prob") is not None:
            yp=mp["y_prob"];nc=len(clbl)
            st.markdown("### ROC & PR Curves")
            co=st.columns(2)
            with co[0]:
                fig_r=go.Figure()
                if nc==2: fpr,tpr,_=roc_curve(y_te,yp[:,1]);fig_r.add_trace(go.Scatter(x=fpr,y=tpr,name=f"AUC={auc(fpr,tpr):.4f}"))
                else:
                    yb=label_binarize(y_te,classes=list(range(nc)));clrs=px.colors.qualitative.Set2
                    for i in range(nc): fpr,tpr,_=roc_curve(yb[:,i],yp[:,i]);fig_r.add_trace(go.Scatter(x=fpr,y=tpr,name=f"{clbl[i]} {auc(fpr,tpr):.3f}",line=dict(color=clrs[i%len(clrs)])))
                fig_r.add_trace(go.Scatter(x=[0,1],y=[0,1],line=dict(dash="dash",color="gray"),showlegend=False))
                fig_r.update_layout(title="ROC",height=450);st.plotly_chart(fig_r,use_container_width=True)
            with co[1]:
                fig_p=go.Figure()
                if nc==2: pr,rc,_=precision_recall_curve(y_te,yp[:,1]);fig_p.add_trace(go.Scatter(x=rc,y=pr,name=f"AP={average_precision_score(y_te,yp[:,1]):.4f}"))
                else:
                    yb=label_binarize(y_te,classes=list(range(nc)));clrs=px.colors.qualitative.Set2
                    for i in range(nc): pr,rc,_=precision_recall_curve(yb[:,i],yp[:,i]);fig_p.add_trace(go.Scatter(x=rc,y=pr,name=f"{clbl[i]}",line=dict(color=clrs[i%len(clrs)])))
                fig_p.update_layout(title="Precision-Recall",height=450);st.plotly_chart(fig_p,use_container_width=True)
        mdl=mp.get("model");fn=st.session_state.feature_names
        if mdl and hasattr(mdl,"feature_importances_") and len(mdl.feature_importances_)==len(fn):
            st.markdown("### Feature Importance")
            fi_df=pd.DataFrame({"Feature":fn,"Importance":mdl.feature_importances_}).sort_values("Importance",ascending=True).tail(20)
            st.plotly_chart(px.bar(fi_df,y="Feature",x="Importance",orientation="h",color="Importance",color_continuous_scale="Viridis").update_layout(height=500),use_container_width=True)
        if mdl:
            st.markdown("### Permutation Importance")
            if st.button("Compute",key="perm_imp"):
                with st.spinner("..."):
                    pi=permutation_importance(mdl,st.session_state.X_test,y_te,n_repeats=10,random_state=42,scoring="accuracy")
                    pi_df=pd.DataFrame({"Feature":fn,"Mean":pi.importances_mean,"Std":pi.importances_std}).sort_values("Mean",ascending=True).tail(20)
                    st.plotly_chart(px.bar(pi_df,y="Feature",x="Mean",error_x="Std",orientation="h",color="Mean",color_continuous_scale="Cividis").update_layout(height=500),use_container_width=True)

# ===================== TAB 5: COMPARISON + DRIFT + COMPLEXITY =====================
with tab5:
    st.markdown("## \U0001f3c6 Model Comparison")
    if st.session_state.comparison_df is None or len(st.session_state.comparison_df)==0: st.warning("Train first.");st.stop()
    comp=st.session_state.comparison_df.copy();preds=st.session_state.all_predictions;y_te=st.session_state.y_test;cn=st.session_state.class_names
    st.markdown("### Leaderboard")
    show_c=["model_name","accuracy","f1","mcc","roc_auc","cv_mean","cv_std","tuning_cv_score","n_params","model_size_kb","train_time"]
    st.dataframe(comp[[c for c in show_c if c in comp.columns]].round(4),use_container_width=True)
    b=comp.iloc[0]
    st.markdown(f'<div class="success-box"><b>\U0001f3c6 {b["model_name"]}</b> Acc={b["accuracy"]:.4f} F1={b.get("f1",0):.4f}</div>',unsafe_allow_html=True)
    st.markdown("### Metric Bars")
    m_opts=[m for m in ["accuracy","f1","mcc","roc_auc","cv_mean"] if m in comp.columns]
    sel_m=st.multiselect("Metrics:",m_opts,default=m_opts[:4],key="cmp_m")
    for met in sel_m:
        sdf=comp[["model_name",met]].dropna().sort_values(met,ascending=False)
        colors=["#27AE60" if i==0 else "#3498DB" for i in range(len(sdf))]
        fig=go.Figure(go.Bar(x=sdf["model_name"],y=sdf[met],marker_color=colors,text=sdf[met].round(4),textposition="outside"))
        fig.update_layout(title=met.title(),height=400);st.plotly_chart(fig,use_container_width=True)
    st.markdown("### Radar")
    rm=[m for m in ["accuracy","precision","recall","f1","mcc","balanced_accuracy"] if m in comp.columns]
    if len(rm)>=3 and len(comp)>=2:
        topn=st.slider("Top N:",2,min(10,len(comp)),min(5,len(comp)))
        fig_rd=go.Figure();clrs=px.colors.qualitative.Set2
        for i in range(topn):
            r=comp.iloc[i];vals=[r.get(m,0) for m in rm]+[r.get(rm[0],0)]
            fig_rd.add_trace(go.Scatterpolar(r=vals,theta=rm+[rm[0]],fill="toself",name=r["model_name"],line=dict(color=clrs[i%len(clrs)])))
        fig_rd.update_layout(polar=dict(radialaxis=dict(visible=True,range=[0,1])),height=550);st.plotly_chart(fig_rd,use_container_width=True)
    # Model Complexity
    if "n_params" in comp.columns and "train_time" in comp.columns:
        st.markdown("### Model Complexity")
        st.plotly_chart(px.scatter(comp,x="model_size_kb",y="accuracy",size="train_time",color="model_name",text="model_name",title="Accuracy vs Size (bubble=time)").update_traces(textposition="top center").update_layout(height=500,xaxis_title="Model Size (KB)",yaxis_title="Accuracy"),use_container_width=True)
    # Data Drift
    st.markdown("### Data Drift Detection")
    if st.button("Check Train vs Test Drift",key="drift_btn"):
        X_tr=st.session_state.X_train;X_te=st.session_state.X_test;fn=st.session_state.feature_names
        drift_data=[]
        for i,fname in enumerate(fn):
            if i>=X_tr.shape[1]: break
            stat,pv=ks_2samp(X_tr[:,i],X_te[:,i])
            drift_data.append({"Feature":fname,"KS Statistic":round(stat,4),"p-value":round(pv,4),"Drift":"Yes" if pv<0.05 else "No"})
        ddf=pd.DataFrame(drift_data).sort_values("KS Statistic",ascending=False)
        st.dataframe(ddf,use_container_width=True)
        nd=len(ddf[ddf["Drift"]=="Yes"])
        if nd>0: st.markdown(f'<div class="warn-box">\u26a0\ufe0f {nd} features show significant drift (p<0.05)</div>',unsafe_allow_html=True)
        else: st.markdown('<div class="success-box">\u2705 No significant drift detected</div>',unsafe_allow_html=True)
    st.markdown("### ROC Overlay")
    nc4=len(cn) if cn else 2;fig_ra=go.Figure();clrs4=px.colors.qualitative.Set2+px.colors.qualitative.Pastel
    for i,mn in enumerate(comp["model_name"].tolist()):
        yp2=preds.get(mn,{}).get("y_prob")
        if yp2 is not None:
            try:
                if nc4==2: fpr,tpr,_=roc_curve(y_te,yp2[:,1]);fig_ra.add_trace(go.Scatter(x=fpr,y=tpr,name=f"{mn} ({auc(fpr,tpr):.3f})",line=dict(color=clrs4[i%len(clrs4)])))
            except: pass
    fig_ra.add_trace(go.Scatter(x=[0,1],y=[0,1],line=dict(dash="dash",color="gray"),showlegend=False))
    fig_ra.update_layout(title="ROC All",height=500);st.plotly_chart(fig_ra,use_container_width=True)
    st.markdown("### Heatmap")
    hc=[c for c in ["accuracy","f1","mcc","roc_auc","cv_mean"] if c in comp.columns]
    if hc:
        hdf=comp.set_index("model_name")[hc].dropna(axis=1,how="all")
        st.plotly_chart(px.imshow(hdf,text_auto=".3f",color_continuous_scale="RdYlGn",aspect="auto").update_layout(height=max(300,len(comp)*40)),use_container_width=True)
    # Per-fold CV
    st.markdown("### Per-Fold CV Scores")
    cvf_model=st.selectbox("Model:",list(st.session_state.trained_models.keys()),key="cvf_m")
    if st.button("Show Folds",key="cvf_btn"):
        try:
            cv_scores=cross_val_score(st.session_state.trained_models[cvf_model],st.session_state.X_train,st.session_state.y_train,cv=5,scoring="accuracy")
            fold_df=pd.DataFrame({"Fold":[f"Fold {i+1}" for i in range(len(cv_scores))],"Accuracy":cv_scores})
            fig_cv=go.Figure(go.Bar(x=fold_df["Fold"],y=fold_df["Accuracy"],text=fold_df["Accuracy"].round(4),textposition="outside",marker_color=px.colors.qualitative.Set2[:len(cv_scores)]))
            fig_cv.add_hline(y=cv_scores.mean(),line_dash="dash",annotation_text=f"Mean={cv_scores.mean():.4f}")
            fig_cv.update_layout(height=400,title=f"Per-Fold: {cvf_model}");st.plotly_chart(fig_cv,use_container_width=True)
        except Exception as e: st.warning(str(e))

# ===================== TAB 6: ANALYSIS + SHAP =====================
with tab6:
    st.markdown("## \U0001f50d Deep Analysis")
    if not st.session_state.trained_models: st.warning("Train first.");st.stop()
    trained=st.session_state.trained_models;preds=st.session_state.all_predictions
    X_tr=st.session_state.X_train;X_te=st.session_state.X_test;y_tr=st.session_state.y_train;y_te=st.session_state.y_test
    cn=st.session_state.class_names;fn=st.session_state.feature_names;mnames=list(trained.keys())
    st.markdown("### Learning Curves")
    lc_model=st.selectbox("Model:",mnames,key="lc_m")
    if st.button("Plot",key="lc_btn"):
        with st.spinner("..."):
            try:
                tsz,tr_sc,te_sc=learning_curve(trained[lc_model],X_tr,y_tr,cv=5,n_jobs=-1,train_sizes=np.linspace(0.1,1.0,10),scoring="accuracy",random_state=42)
                fig_lc=go.Figure()
                fig_lc.add_trace(go.Scatter(x=tsz,y=tr_sc.mean(axis=1),mode="lines+markers",name="Train"))
                fig_lc.add_trace(go.Scatter(x=tsz,y=te_sc.mean(axis=1),mode="lines+markers",name="Val"))
                fig_lc.update_layout(title=f"Learning Curve: {lc_model}",height=450);st.plotly_chart(fig_lc,use_container_width=True)
            except Exception as e: st.warning(str(e))
    st.markdown("### Calibration (Binary)")
    nc5=len(np.unique(y_te))
    if nc5==2 and st.button("Plot Calibration",key="cal_btn"):
        fig_cal=go.Figure();clrs5=px.colors.qualitative.Set2
        for i,mn in enumerate(mnames):
            yp3=preds.get(mn,{}).get("y_prob")
            if yp3 is not None:
                try: fp,mp2=calibration_curve(y_te,yp3[:,1],n_bins=10);fig_cal.add_trace(go.Scatter(x=mp2,y=fp,mode="lines+markers",name=mn,line=dict(color=clrs5[i%len(clrs5)])))
                except: pass
        fig_cal.add_trace(go.Scatter(x=[0,1],y=[0,1],line=dict(dash="dash",color="gray"),name="Perfect"))
        fig_cal.update_layout(title="Calibration",height=500);st.plotly_chart(fig_cal,use_container_width=True)
    # SHAP
    st.markdown("### SHAP Explainability")
    if HAS_SHAP:
        shap_model=st.selectbox("Model:",mnames,key="shap_m")
        if st.button("Compute SHAP",key="shap_btn"):
            with st.spinner("Computing SHAP values (may take a moment)..."):
                try:
                    mdl_s=trained[shap_model]
                    X_sample=X_te[:min(200,len(X_te))]
                    if hasattr(mdl_s,"predict_proba"):
                        try: explainer=shap.TreeExplainer(mdl_s)
                        except: explainer=shap.KernelExplainer(mdl_s.predict_proba,shap.kmeans(X_tr,min(50,len(X_tr))))
                    else:
                        explainer=shap.KernelExplainer(mdl_s.predict,shap.kmeans(X_tr,min(50,len(X_tr))))
                    shap_values=explainer.shap_values(X_sample)
                    if isinstance(shap_values,list): sv=np.abs(shap_values[0]) if len(shap_values)>0 else np.abs(shap_values)
                    elif shap_values.ndim==3: sv=np.abs(shap_values).mean(axis=2)
                    else: sv=np.abs(shap_values)
                    shap_imp=sv.mean(axis=0)
                    if len(shap_imp)==len(fn):
                        shap_df=pd.DataFrame({"Feature":fn,"SHAP":shap_imp}).sort_values("SHAP",ascending=True).tail(20)
                        st.plotly_chart(px.bar(shap_df,y="Feature",x="SHAP",orientation="h",color="SHAP",color_continuous_scale="Reds",title=f"SHAP: {shap_model}").update_layout(height=500),use_container_width=True)
                    st.success("SHAP computed!")
                except Exception as e: st.warning(f"SHAP error: {e}")
    else: st.info("pip install shap for SHAP explainability")
    # CV Predictions
    st.markdown("### Cross-Val Predictions")
    cvp_model=st.selectbox("Model:",mnames,key="cvp_m")
    if st.button("CV Predict",key="cvp_btn"):
        with st.spinner("..."):
            try:
                cvp=cross_val_predict(trained[cvp_model],X_tr,y_tr,cv=5)
                cm_cv=confusion_matrix(y_tr,cvp);clbl3=cn if len(cn)==cm_cv.shape[0] else [str(i) for i in range(cm_cv.shape[0])]
                co2=st.columns(2)
                with co2[0]: st.plotly_chart(px.imshow(cm_cv,text_auto=True,color_continuous_scale="Blues",x=clbl3,y=clbl3).update_layout(title=f"CV: {cvp_model}",height=400),use_container_width=True)
                with co2[1]: st.dataframe(pd.DataFrame(classification_report(y_tr,cvp,target_names=clbl3,output_dict=True)).T.round(4),use_container_width=True)
            except Exception as e: st.warning(str(e))
    # Error Analysis
    st.markdown("### Error Analysis")
    err_model=st.selectbox("Model:",mnames,key="err_m")
    mp3=preds.get(err_model,{})
    if "y_pred" in mp3:
        y_pred3=mp3["y_pred"];wrong=y_pred3!=y_te;n_wrong=int(wrong.sum())
        st.metric("Misclassified",f"{n_wrong}/{len(y_te)} ({n_wrong/len(y_te)*100:.1f}%)")
        if n_wrong>0:
            err_idx=np.where(wrong)[0]
            err_df=pd.DataFrame(X_te[err_idx],columns=fn);err_df["True"]=y_te[err_idx];err_df["Predicted"]=y_pred3[err_idx]
            st.dataframe(err_df.head(50).round(4),use_container_width=True)
            err_cm=pd.crosstab(pd.Series(y_te[err_idx],name="True"),pd.Series(y_pred3[err_idx],name="Pred"))
            st.plotly_chart(px.imshow(err_cm,text_auto=True,color_continuous_scale="Reds",title="Error Pattern").update_layout(height=400),use_container_width=True)
    # Ensemble
    st.markdown("### Ensemble Builder")
    if len(mnames)>=2:
        ens_models=st.multiselect("Models:",mnames,default=mnames[:min(3,len(mnames))],key="ens_m")
        ens_type=st.radio("Type:",["Soft Voting","Hard Voting","Stacking"],horizontal=True)
        if st.button("Build",key="ens_btn") and len(ens_models)>=2:
            with st.spinner("..."):
                try:
                    estimators=[(mn,trained[mn]) for mn in ens_models]
                    prob_ok=all(hasattr(trained[mn],"predict_proba") for mn in ens_models)
                    if ens_type=="Stacking": ens=StackingClassifier(estimators=estimators,final_estimator=LogisticRegression(max_iter=1000),cv=5)
                    else: ens=VotingClassifier(estimators=estimators,voting="soft" if ens_type=="Soft Voting" and prob_ok else "hard")
                    ens.fit(X_tr,y_tr);yp_e=ens.predict(X_te)
                    ea=accuracy_score(y_te,yp_e);ef=f1_score(y_te,yp_e,average="weighted",zero_division=0)
                    st.markdown(f'<div class="success-box"><b>Ensemble: Acc={ea:.4f} F1={ef:.4f}</b></div>',unsafe_allow_html=True)
                    cm_e=confusion_matrix(y_te,yp_e);clbl_e=cn if len(cn)==cm_e.shape[0] else [str(i) for i in range(cm_e.shape[0])]
                    st.plotly_chart(px.imshow(cm_e,text_auto=True,color_continuous_scale="Greens",x=clbl_e,y=clbl_e).update_layout(title="Ensemble CM",height=400),use_container_width=True)
                except Exception as e: st.error(str(e))
    # McNemar
    st.markdown("### McNemar Test")
    if len(mnames)>=2:
        m1=st.selectbox("A:",mnames,index=0,key="sa");m2=st.selectbox("B:",mnames,index=min(1,len(mnames)-1),key="sb")
        if st.button("Test",key="stat_btn") and m1!=m2:
            pa=preds[m1]["y_pred"];pb=preds[m2]["y_pred"];ca=(pa==y_te);cb=(pb==y_te)
            bv=int(np.sum(ca&~cb));cv2=int(np.sum(~ca&cb))
            pv=binomtest(bv,bv+cv2,0.5).pvalue if bv+cv2>0 else 1.0
            st.metric("p-value",f"{pv:.6f}")
            if pv<0.05: st.markdown(f'<div class="success-box">Significant (p={pv:.4f})</div>',unsafe_allow_html=True)
            else: st.markdown(f'<div class="info-box">Not significant (p={pv:.4f})</div>',unsafe_allow_html=True)

# ===================== TAB 7: PREDICTION PLAYGROUND + BATCH =====================
with tab7:
    st.markdown("## \U0001f52e Prediction Playground")
    if not st.session_state.trained_models: st.warning("Train first.");st.stop()
    trained=st.session_state.trained_models;fn=st.session_state.feature_names;cn=st.session_state.class_names;mnames=list(trained.keys())
    pred_model=st.selectbox("Model:",mnames,key="pred_m")
    st.markdown("### Single Prediction")
    st.markdown("Enter feature values:")
    input_vals={}
    ncols_p=min(4,len(fn));cols_p=st.columns(ncols_p)
    X_tr=st.session_state.X_train
    for i,fname in enumerate(fn):
        with cols_p[i%ncols_p]:
            mn_v=float(X_tr[:,i].min()) if i<X_tr.shape[1] else 0.0
            mx_v=float(X_tr[:,i].max()) if i<X_tr.shape[1] else 1.0
            md_v=float(np.median(X_tr[:,i])) if i<X_tr.shape[1] else 0.5
            input_vals[fname]=st.number_input(fname,value=md_v,key=f"inp_{fname}")
    if st.button("\U0001f52e Predict",type="primary",key="pred_btn"):
        inp_arr=np.array([[input_vals[f] for f in fn]])
        mdl=trained[pred_model]
        pred_cls=mdl.predict(inp_arr)[0]
        cls_name=cn[int(pred_cls)] if int(pred_cls)<len(cn) else str(pred_cls)
        st.markdown(f'<div class="success-box" style="font-size:1.3rem;text-align:center"><b>Predicted Class: {cls_name}</b></div>',unsafe_allow_html=True)
        if hasattr(mdl,"predict_proba"):
            try:
                probs=mdl.predict_proba(inp_arr)[0]
                prob_df=pd.DataFrame({"Class":cn[:len(probs)],"Probability":probs}).sort_values("Probability",ascending=True)
                st.plotly_chart(px.bar(prob_df,y="Class",x="Probability",orientation="h",color="Probability",color_continuous_scale="Viridis",title="Class Probabilities").update_layout(height=350),use_container_width=True)
            except: pass
    # Batch Prediction
    st.markdown("---")
    st.markdown("### Batch Prediction")
    batch_file=st.file_uploader("Upload CSV for batch prediction:",type=["csv"],key="batch_up")
    if batch_file:
        try:
            bdf=pd.read_csv(batch_file)
            st.dataframe(bdf.head(),use_container_width=True)
            if st.button("Run Batch Prediction",key="batch_btn"):
                mdl=trained[pred_model]
                missing_cols=[f for f in fn if f not in bdf.columns]
                if missing_cols:
                    st.warning(f"Missing columns: {missing_cols}")
                else:
                    X_batch=bdf[fn].values.astype(float)
                    batch_preds=mdl.predict(X_batch)
                    bdf["Predicted"]=batch_preds
                    bdf["Predicted_Label"]=[cn[int(p)] if int(p)<len(cn) else str(p) for p in batch_preds]
                    if hasattr(mdl,"predict_proba"):
                        try:
                            bp=mdl.predict_proba(X_batch)
                            for i,c in enumerate(cn[:bp.shape[1]]): bdf[f"Prob_{c}"]=bp[:,i].round(4)
                        except: pass
                    st.dataframe(bdf,use_container_width=True)
                    buf=io.StringIO();bdf.to_csv(buf,index=False)
                    st.download_button("Download Predictions",buf.getvalue(),"batch_predictions.csv","text/csv",use_container_width=True)
        except Exception as e: st.error(str(e))

# ===================== TAB 8: EXPORT =====================
with tab8:
    st.markdown("## \U0001f4be Export & Code")
    if not st.session_state.trained_models: st.warning("Train first.");st.stop()
    trained=st.session_state.trained_models;preds=st.session_state.all_predictions;res=st.session_state.model_results;mnames=list(trained.keys())
    comp=st.session_state.comparison_df
    # Model Download
    st.markdown("### Download Trained Model")
    dl_model=st.selectbox("Model:",mnames,key="dl_m")
    if st.button("Prepare Download",key="dl_btn"):
        buf=io.BytesIO();pickle.dump(trained[dl_model],buf);buf.seek(0)
        st.download_button(f"Download {dl_model} (.pkl)",buf.getvalue(),f"{dl_model.replace(' ','_').lower()}.pkl","application/octet-stream",use_container_width=True,key="dl_pkl")
        st.code(f"""# Load model:\nimport pickle\nwith open("{dl_model.replace(' ','_').lower()}.pkl","rb") as f:\n    model = pickle.load(f)\ny_pred = model.predict(X_new)""")
    # Pipeline Code Gen
    st.markdown("### Pipeline Code Generator")
    code_model=st.selectbox("Model:",mnames,key="code_m")
    bp_code=preds.get(code_model,{}).get("best_params",{})
    pp_info=st.session_state.get("pp_info",{})
    gen_code=generate_pipeline_code(pp_info,code_model,bp_code if isinstance(bp_code,dict) else {})
    st.code(gen_code,language="python")
    st.download_button("Download Code (.py)",gen_code,f"pipeline_{code_model.replace(' ','_').lower()}.py","text/plain",use_container_width=True,key="dl_code")
    # CSV Export
    st.markdown("### Export Results")
    if comp is not None and len(comp)>0:
        buf=io.StringIO();comp.to_csv(buf,index=False)
        st.download_button("Download Comparison CSV",buf.getvalue(),"comparison.csv","text/csv",use_container_width=True,key="dl_csv")
    # Full Report
    st.markdown("### Full Report")
    if st.button("Generate Report",key="rpt_btn"):
        rpt=f"ML Classification Suite v3.0 Report\n{'='*60}\nDate: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        rpt+=f"Dataset: {len(st.session_state.y)} samples, {len(st.session_state.feature_names)} features, {len(st.session_state.class_names)} classes\n"
        rpt+=f"Train: {len(st.session_state.X_train)}, Test: {len(st.session_state.X_test)}\n\n"
        if comp is not None and len(comp)>0:
            rpt+=f"Best Model: {comp.iloc[0]['model_name']} (Acc={comp.iloc[0]['accuracy']:.4f})\n\n"
            rpt+=f"{'='*60}\nAll Models:\n{'='*60}\n"
            for _,r in comp.iterrows():
                rpt+=f"\n{r['model_name']}:\n"
                rpt+=f"  Accuracy:  {r['accuracy']:.4f}\n"
                rpt+=f"  F1:        {r.get('f1',0):.4f}\n"
                rpt+=f"  MCC:       {r.get('mcc',0):.4f}\n"
                rpt+=f"  ROC AUC:   {r.get('roc_auc',float('nan')):.4f}\n"
                rpt+=f"  CV Mean:   {r.get('cv_mean',0):.4f} +/- {r.get('cv_std',0):.4f}\n"
                rpt+=f"  Time:      {r.get('train_time',0):.2f}s\n"
                rpt+=f"  Size:      {r.get('model_size_kb',0):.1f} KB\n"
                bp3=preds.get(r['model_name'],{}).get("best_params",{})
                if bp3: rpt+=f"  Params:    {bp3}\n"
        st.text_area("Report Preview:",rpt,height=400)
        st.download_button("Download Report",rpt,"ml_report.txt","text/plain",use_container_width=True,key="dl_rpt")
    # Session Save/Load
    st.markdown("### Session Persistence")
    if st.button("Save Session (JSON)",key="save_sess"):
        try:
            sess={"feature_names":st.session_state.feature_names,"target_name":st.session_state.target_name,
                "class_names":st.session_state.class_names,"pp_info":st.session_state.get("pp_info",{})}
            if comp is not None: sess["comparison"]=comp[["model_name","accuracy","f1","mcc"]].to_dict(orient="records")
            for mn in mnames:
                bp=preds.get(mn,{}).get("best_params",{})
                if isinstance(bp,dict): sess[f"params_{mn}"]=bp
            sj=json.dumps(sess,indent=2,default=str)
            st.download_button("Download Session",sj,"ml_session.json","application/json",use_container_width=True,key="dl_sess")
        except Exception as e: st.warning(str(e))

st.markdown("---")
st.markdown('<div style="text-align:center;color:#888;font-size:0.85rem">ML Classification Suite v3.0 \u2022 25 Models \u2022 SHAP \u2022 AutoTuning</div>',unsafe_allow_html=True)
