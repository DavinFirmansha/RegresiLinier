import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import gammaln
from scipy.optimize import minimize_scalar
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings, io
warnings.filterwarnings('ignore')

st.set_page_config(page_title="SQC Pro — Statistical Quality Control",layout="wide",page_icon="🏭")
st.title("🏭 Statistical Quality Control — Analisis Lengkap")
st.caption("Control Charts · Process Capability · MSA · Acceptance Sampling · Pareto · Reliability · DOE")

# ============================================================
# CONSTANTS: SPC factors (A2, A3, D3, D4, B3, B4, d2, d3, c4)
# ============================================================
SPC_FACTORS = {
    2:  {'A2':1.880,'A3':2.659,'B3':0,'B4':3.267,'D3':0,'D4':3.267,'d2':1.128,'d3':0.853,'c4':0.7979},
    3:  {'A2':1.023,'A3':1.954,'B3':0,'B4':2.568,'D3':0,'D4':2.574,'d2':1.693,'d3':0.888,'c4':0.8862},
    4:  {'A2':0.729,'A3':1.628,'B3':0,'B4':2.266,'D3':0,'D4':2.282,'d2':2.059,'d3':0.880,'c4':0.9213},
    5:  {'A2':0.577,'A3':1.427,'B3':0,'B4':2.089,'D3':0,'D4':2.114,'d2':2.326,'d3':0.864,'c4':0.9400},
    6:  {'A2':0.483,'A3':1.287,'B3':0.030,'B4':1.970,'D3':0,'D4':2.004,'d2':2.534,'d3':0.848,'c4':0.9515},
    7:  {'A2':0.419,'A3':1.182,'B3':0.118,'B4':1.882,'D3':0.076,'D4':1.924,'d2':2.704,'d3':0.833,'c4':0.9594},
    8:  {'A2':0.373,'A3':1.099,'B3':0.185,'B4':1.815,'D3':0.136,'D4':1.864,'d2':2.847,'d3':0.820,'c4':0.9650},
    9:  {'A2':0.337,'A3':1.032,'B3':0.239,'B4':1.761,'D3':0.184,'D4':1.816,'d2':2.970,'d3':0.808,'c4':0.9693},
    10: {'A2':0.308,'A3':0.975,'B3':0.284,'B4':1.716,'D3':0.223,'D4':1.777,'d2':3.078,'d3':0.797,'c4':0.9727},
    15: {'A2':0.223,'A3':0.789,'B3':0.428,'B4':1.572,'D3':0.347,'D4':1.653,'d2':3.472,'d3':0.756,'c4':0.9823},
    20: {'A2':0.180,'A3':0.680,'B3':0.510,'B4':1.490,'D3':0.415,'D4':1.585,'d2':3.735,'d3':0.729,'c4':0.9869},
    25: {'A2':0.153,'A3':0.606,'B3':0.565,'B4':1.435,'D3':0.459,'D4':1.541,'d2':3.931,'d3':0.708,'c4':0.9896},
}

def get_factor(n, key):
    if n in SPC_FACTORS: return SPC_FACTORS[n][key]
    keys = sorted(SPC_FACTORS.keys())
    for i in range(len(keys)-1):
        if keys[i] <= n <= keys[i+1]:
            f = (n - keys[i]) / (keys[i+1] - keys[i])
            return SPC_FACTORS[keys[i]][key]*(1-f) + SPC_FACTORS[keys[i+1]][key]*f
    return SPC_FACTORS[keys[-1]][key]

# ============================================================
# DEMO DATA GENERATOR
# ============================================================
def gen_demo_variable(n_subgroups=25, n_size=5, mu=50.0, sigma=2.0, shift_at=None, shift_mag=0):
    np.random.seed(42)
    data = np.random.normal(mu, sigma, (n_subgroups, n_size))
    if shift_at and shift_at < n_subgroups:
        data[shift_at:] += shift_mag
    return data

def gen_demo_attribute_p(n_subgroups=30, n_inspected=100, p0=0.05, shift_at=None, shift_p=0):
    np.random.seed(42)
    ps = np.full(n_subgroups, p0)
    if shift_at and shift_at < n_subgroups: ps[shift_at:] += shift_p
    defects = np.array([np.random.binomial(n_inspected, p) for p in ps])
    return defects, n_inspected

def gen_demo_attribute_c(n_subgroups=30, c0=5.0, shift_at=None, shift_c=0):
    np.random.seed(42)
    cs = np.full(n_subgroups, c0)
    if shift_at and shift_at < n_subgroups: cs[shift_at:] += shift_c
    return np.array([np.random.poisson(c) for c in cs])

# ============================================================
# CONTROL CHART PLOTTING HELPER
# ============================================================
def plot_control_chart(x_vals, y_vals, cl, ucl, lcl, title, y_label, point_labels=None,
                       zones=True, violations=None, sigma_lines=True):
    fig = go.Figure()
    colors = ['steelblue'] * len(y_vals)
    sizes = [7] * len(y_vals)
    if violations:
        for idx in violations:
            if 0 <= idx < len(colors):
                colors[idx] = 'red'; sizes[idx] = 10
    fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines+markers',
        marker=dict(color=colors, size=sizes, line=dict(width=1, color='white')),
        line=dict(color='steelblue', width=1.5), name='Data',
        text=point_labels, hoverinfo='text+y' if point_labels else 'y'))
    fig.add_hline(y=cl, line=dict(color='green', width=2), annotation_text=f"CL={cl:.4f}", annotation_position="bottom right")
    fig.add_hline(y=ucl, line=dict(color='red', width=2, dash='dash'), annotation_text=f"UCL={ucl:.4f}", annotation_position="top right")
    fig.add_hline(y=lcl, line=dict(color='red', width=2, dash='dash'), annotation_text=f"LCL={lcl:.4f}", annotation_position="bottom right")
    if sigma_lines and zones:
        s1_u = cl + (ucl - cl) / 3; s2_u = cl + 2 * (ucl - cl) / 3
        s1_l = cl - (cl - lcl) / 3; s2_l = cl - 2 * (cl - lcl) / 3
        for yy, lbl in [(s1_u,'+1σ'),(s2_u,'+2σ'),(s1_l,'-1σ'),(s2_l,'-2σ')]:
            fig.add_hline(y=yy, line=dict(color='gray', width=0.5, dash='dot'))
        fig.add_vrect(x0=x_vals[0]-0.5, x1=x_vals[-1]+0.5, y0=s1_l, y1=s1_u, fillcolor="green", opacity=0.05, layer="below")
        fig.add_vrect(x0=x_vals[0]-0.5, x1=x_vals[-1]+0.5, y0=s1_u, y1=s2_u, fillcolor="yellow", opacity=0.05, layer="below")
        fig.add_vrect(x0=x_vals[0]-0.5, x1=x_vals[-1]+0.5, y0=s2_l, y1=s1_l, fillcolor="yellow", opacity=0.05, layer="below")
        fig.add_vrect(x0=x_vals[0]-0.5, x1=x_vals[-1]+0.5, y0=s2_u, y1=ucl, fillcolor="orange", opacity=0.05, layer="below")
        fig.add_vrect(x0=x_vals[0]-0.5, x1=x_vals[-1]+0.5, y0=lcl, y1=s2_l, fillcolor="orange", opacity=0.05, layer="below")
    fig.update_layout(title=title, height=420, xaxis_title="Subgroup", yaxis_title=y_label,
                      hovermode='x unified', template='plotly_white')
    return fig

# ============================================================
# WESTERN ELECTRIC / NELSON RULES
# ============================================================
def detect_violations(data, cl, ucl, lcl):
    violations = {}
    sigma = (ucl - cl) / 3
    n = len(data)
    s1u = cl + sigma; s2u = cl + 2*sigma
    s1l = cl - sigma; s2l = cl - 2*sigma
    # Rule 1: Point beyond 3σ
    r1 = [i for i in range(n) if data[i] > ucl or data[i] < lcl]
    if r1: violations['Rule 1: Beyond 3σ'] = r1
    # Rule 2: 9 points same side of CL
    r2 = []
    for i in range(8, n):
        seg = data[i-8:i+1]
        if all(s > cl for s in seg) or all(s < cl for s in seg): r2.append(i)
    if r2: violations['Rule 2: 9 same side'] = r2
    # Rule 3: 6 points trending
    r3 = []
    for i in range(5, n):
        seg = data[i-5:i+1]
        if all(seg[j] < seg[j+1] for j in range(5)) or all(seg[j] > seg[j+1] for j in range(5)): r3.append(i)
    if r3: violations['Rule 3: 6 trending'] = r3
    # Rule 4: 14 alternating
    r4 = []
    for i in range(13, n):
        seg = data[i-13:i+1]
        alt = all((seg[j]-seg[j+1])*(seg[j+1]-seg[j+2]) < 0 for j in range(12))
        if alt: r4.append(i)
    if r4: violations['Rule 4: 14 alternating'] = r4
    # Rule 5: 2/3 beyond 2σ
    r5 = []
    for i in range(2, n):
        seg = data[i-2:i+1]
        cnt = sum(1 for s in seg if s > s2u or s < s2l)
        if cnt >= 2: r5.append(i)
    if r5: violations['Rule 5: 2/3 beyond 2σ'] = r5
    # Rule 6: 4/5 beyond 1σ
    r6 = []
    for i in range(4, n):
        seg = data[i-4:i+1]
        cnt = sum(1 for s in seg if s > s1u or s < s1l)
        if cnt >= 4: r6.append(i)
    if r6: violations['Rule 6: 4/5 beyond 1σ'] = r6
    # Rule 7: 15 within 1σ (stratification)
    r7 = []
    for i in range(14, n):
        seg = data[i-14:i+1]
        if all(s1l < s < s1u for s in seg): r7.append(i)
    if r7: violations['Rule 7: 15 within 1σ'] = r7
    # Rule 8: 8 beyond 1σ both sides
    r8 = []
    for i in range(7, n):
        seg = data[i-7:i+1]
        if all(s > s1u or s < s1l for s in seg): r8.append(i)
    if r8: violations['Rule 8: 8 beyond 1σ both'] = r8
    return violations

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.header("🏭 Menu SQC")
module = st.sidebar.selectbox("Modul:", [
    'xbar_r','xbar_s','imr','p_chart','np_chart','c_chart','u_chart',
    'cusum','ewma','capability','normality','pareto','msa','acceptance','reliability','doe'
], format_func=lambda x: {
    'xbar_r':'1. X̄-R Chart','xbar_s':'2. X̄-S Chart','imr':'3. I-MR Chart',
    'p_chart':'4. p-Chart','np_chart':'5. np-Chart','c_chart':'6. c-Chart','u_chart':'7. u-Chart',
    'cusum':'8. CUSUM Chart','ewma':'9. EWMA Chart',
    'capability':'10. Process Capability','normality':'11. Normality Testing',
    'pareto':'12. Pareto Analysis','msa':'13. MSA (Gage R&R)',
    'acceptance':'14. Acceptance Sampling','reliability':'15. Reliability Analysis','doe':'16. DOE'
}[x])

# ============================================================
# 1. X-BAR & R CHART
# ============================================================
if module == 'xbar_r':
    st.header("📊 X̄-R Chart (Mean & Range)")
    st.markdown("Control chart untuk **rata-rata** dan **range** subgroup. Paling umum untuk data variabel dengan n=2–10.")
    src = st.radio("Data:", ['Demo', 'Manual', 'Upload CSV'], horizontal=True)
    if src == 'Demo':
        c1,c2 = st.columns(2)
        nsub = c1.slider("Subgroup:", 10, 50, 25)
        nsize = c2.slider("Sample size (n):", 2, 10, 5)
        mu = c1.number_input("μ:", value=50.0)
        sig = c2.number_input("σ:", 0.1, 50.0, 2.0)
        shift = st.checkbox("Tambah shift?")
        shift_at = st.slider("Shift mulai subgroup:", 1, nsub-1, nsub//2) if shift else None
        shift_mag = st.number_input("Magnitude shift:", value=3.0) if shift else 0
        data = gen_demo_variable(nsub, nsize, mu, sig, shift_at, shift_mag)
        df = pd.DataFrame(data, columns=[f'X{i+1}' for i in range(nsize)])
        df.index = [f'SG{i+1}' for i in range(nsub)]
        df.index.name = 'Subgroup'
    elif src == 'Manual':
        st.markdown("Masukkan data (baris=subgroup, kolom=sampel), pisah koma per baris:")
        txt = st.text_area("Data:", "50.2,49.8,50.5,51.0,49.7\n49.5,50.1,50.3,49.9,50.0\n51.2,50.8,49.6,50.4,50.1\n49.8,50.2,50.6,50.0,49.5\n50.5,50.3,49.8,50.1,50.7")
        rows = [r.strip() for r in txt.strip().split('\n') if r.strip()]
        data = np.array([[float(v) for v in r.split(',')] for r in rows])
        nsize = data.shape[1]; nsub = data.shape[0]
        df = pd.DataFrame(data, columns=[f'X{i+1}' for i in range(nsize)])
        df.index = [f'SG{i+1}' for i in range(nsub)]; df.index.name = 'Subgroup'
    else:
        up = st.file_uploader("Upload CSV (kolom=sampel, baris=subgroup):", type=['csv'])
        if up:
            df = pd.read_csv(up); data = df.select_dtypes(include=[np.number]).values
            nsize = data.shape[1]; nsub = data.shape[0]
        else: st.info("Upload CSV"); st.stop()

    with st.expander("📋 Data", expanded=False): st.dataframe(df.round(4), use_container_width=True)

    xbar = data.mean(axis=1); R = data.max(axis=1) - data.min(axis=1)
    xdbar = xbar.mean(); Rbar = R.mean()
    A2 = get_factor(nsize, 'A2'); D3 = get_factor(nsize, 'D3'); D4 = get_factor(nsize, 'D4'); d2 = get_factor(nsize, 'd2')
    UCL_x = xdbar + A2 * Rbar; LCL_x = xdbar - A2 * Rbar
    UCL_r = D4 * Rbar; LCL_r = D3 * Rbar
    sigma_est = Rbar / d2
    sg = np.arange(1, nsub + 1)

    viol_x = detect_violations(xbar, xdbar, UCL_x, LCL_x)
    viol_r = detect_violations(R, Rbar, UCL_r, LCL_r)
    all_viol_x = set(); [all_viol_x.update(v) for v in viol_x.values()]
    all_viol_r = set(); [all_viol_r.update(v) for v in viol_r.values()]

    fig_x = plot_control_chart(sg, xbar, xdbar, UCL_x, LCL_x, "X̄ Chart", "X̄", violations=list(all_viol_x))
    fig_r = plot_control_chart(sg, R, Rbar, UCL_r, LCL_r, "R Chart", "Range", violations=list(all_viol_r))
    st.plotly_chart(fig_x, use_container_width=True)
    st.plotly_chart(fig_r, use_container_width=True)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("X̿", f"{xdbar:.4f}"); c2.metric("R̄", f"{Rbar:.4f}")
    c3.metric("σ̂ (R̄/d₂)", f"{sigma_est:.4f}"); c4.metric("n", str(nsize))

    st.markdown("### SPC Factors & Control Limits")
    st.dataframe(pd.DataFrame([
        {"Chart":"X̄","CL":f"{xdbar:.4f}","UCL":f"{UCL_x:.4f}","LCL":f"{LCL_x:.4f}","Factor":f"A₂={A2:.3f}"},
        {"Chart":"R","CL":f"{Rbar:.4f}","UCL":f"{UCL_r:.4f}","LCL":f"{LCL_r:.4f}","Factor":f"D₃={D3:.3f}, D₄={D4:.3f}"}
    ]), use_container_width=True, hide_index=True)

    if viol_x or viol_r:
        st.markdown("### ⚠️ Violations Detected (Nelson/Western Electric Rules)")
        for rule, idxs in {**viol_x}.items():
            st.warning(f"**X̄ — {rule}:** Subgroup {[i+1 for i in idxs[:10]]}{'...' if len(idxs)>10 else ''}")
        for rule, idxs in {**viol_r}.items():
            st.warning(f"**R — {rule}:** Subgroup {[i+1 for i in idxs[:10]]}{'...' if len(idxs)>10 else ''}")
    else:
        st.success("✅ Proses dalam kendali — tidak ada pelanggaran aturan ditemukan.")

    with st.expander("📊 Detail per Subgroup"):
        detail = pd.DataFrame({'Subgroup': sg, 'X̄': xbar.round(4), 'R': R.round(4),
            'X̄ in control': ['✅' if LCL_x<=xbar[i]<=UCL_x else '❌' for i in range(nsub)],
            'R in control': ['✅' if LCL_r<=R[i]<=UCL_r else '❌' for i in range(nsub)]})
        st.dataframe(detail, use_container_width=True, hide_index=True)

# ============================================================
# 2. X-BAR & S CHART
# ============================================================
elif module == 'xbar_s':
    st.header("📊 X̄-S Chart (Mean & Std Dev)")
    st.markdown("Lebih akurat dari X̄-R untuk **n > 10**. Menggunakan standar deviasi subgroup.")
    src = st.radio("Data:", ['Demo', 'Upload CSV'], horizontal=True)
    if src == 'Demo':
        c1,c2 = st.columns(2); nsub = c1.slider("Subgroup:", 10, 50, 25); nsize = c2.slider("n:", 2, 25, 10)
        data = gen_demo_variable(nsub, nsize, 50.0, 2.0)
    else:
        up = st.file_uploader("CSV:", type=['csv'])
        if up: data = pd.read_csv(up).select_dtypes(include=[np.number]).values; nsub,nsize=data.shape
        else: st.stop()

    xbar = data.mean(1); S = data.std(1, ddof=1)
    xdbar = xbar.mean(); Sbar = S.mean()
    A3 = get_factor(nsize,'A3'); B3 = get_factor(nsize,'B3'); B4 = get_factor(nsize,'B4'); c4 = get_factor(nsize,'c4')
    UCL_x = xdbar + A3*Sbar; LCL_x = xdbar - A3*Sbar
    UCL_s = B4*Sbar; LCL_s = B3*Sbar
    sigma_est = Sbar / c4; sg = np.arange(1, nsub+1)

    vx = detect_violations(xbar, xdbar, UCL_x, LCL_x); avx = set(); [avx.update(v) for v in vx.values()]
    vs = detect_violations(S, Sbar, UCL_s, LCL_s); avs = set(); [avs.update(v) for v in vs.values()]
    st.plotly_chart(plot_control_chart(sg, xbar, xdbar, UCL_x, LCL_x, "X̄ Chart", "X̄", violations=list(avx)), use_container_width=True)
    st.plotly_chart(plot_control_chart(sg, S, Sbar, UCL_s, LCL_s, "S Chart", "Std Dev", violations=list(avs)), use_container_width=True)
    c1,c2,c3 = st.columns(3); c1.metric("X̿",f"{xdbar:.4f}"); c2.metric("S̄",f"{Sbar:.4f}"); c3.metric("σ̂ (S̄/c₄)",f"{sigma_est:.4f}")
    st.dataframe(pd.DataFrame([{"Chart":"X̄","CL":f"{xdbar:.4f}","UCL":f"{UCL_x:.4f}","LCL":f"{LCL_x:.4f}"},{"Chart":"S","CL":f"{Sbar:.4f}","UCL":f"{UCL_s:.4f}","LCL":f"{LCL_s:.4f}"}]),use_container_width=True,hide_index=True)
    if vx or vs:
        for r,idxs in vx.items(): st.warning(f"X̄ — {r}: SG {[i+1 for i in idxs[:10]]}")
        for r,idxs in vs.items(): st.warning(f"S — {r}: SG {[i+1 for i in idxs[:10]]}")
    else: st.success("✅ In control")

# ============================================================
# 3. I-MR (INDIVIDUALS & MOVING RANGE)
# ============================================================
elif module == 'imr':
    st.header("📊 I-MR Chart (Individual & Moving Range)")
    st.markdown("Untuk data **individual** (n=1 per subgroup). Cocok untuk batch kecil atau pengukuran mahal.")
    src = st.radio("Data:", ['Demo', 'Manual', 'Upload CSV'], horizontal=True)
    if src == 'Demo':
        n = st.slider("n:", 15, 100, 30)
        shift = st.checkbox("Shift?", key='imr_shift')
        shift_at = st.slider("Shift at:", 1, n-1, n//2, key='imr_sa') if shift else None
        shift_mag = st.number_input("Shift:", value=4.0, key='imr_sm') if shift else 0
        np.random.seed(42); data = np.random.normal(50, 2, n)
        if shift_at: data[shift_at:] += shift_mag
    elif src == 'Manual':
        txt = st.text_area("Data (1 per baris atau koma):", "50.2,49.8,50.5,51.0,49.7,49.5,50.1,50.3,49.9,50.0,51.2,50.8,49.6,50.4,50.1")
        data = np.array([float(v.strip()) for v in txt.replace('\n',',').split(',') if v.strip()])
        n = len(data)
    else:
        up = st.file_uploader("CSV (1 kolom):", type=['csv'])
        if up: df=pd.read_csv(up);col=st.selectbox("Kolom:",df.select_dtypes(include=[np.number]).columns);data=df[col].dropna().values;n=len(data)
        else: st.stop()

    MR = np.abs(np.diff(data)); Ibar = data.mean(); MRbar = MR.mean()
    d2 = 1.128; sigma_est = MRbar / d2
    UCL_i = Ibar + 3*sigma_est; LCL_i = Ibar - 3*sigma_est
    UCL_mr = 3.267 * MRbar; LCL_mr = 0
    sg = np.arange(1, n+1); sg_mr = np.arange(2, n+1)

    viol_i = detect_violations(data, Ibar, UCL_i, LCL_i); avi = set(); [avi.update(v) for v in viol_i.values()]
    viol_mr = detect_violations(MR, MRbar, UCL_mr, LCL_mr); avmr = set(); [avmr.update(v) for v in viol_mr.values()]
    st.plotly_chart(plot_control_chart(sg, data, Ibar, UCL_i, LCL_i, "Individual Chart", "X", violations=list(avi)), use_container_width=True)
    st.plotly_chart(plot_control_chart(sg_mr, MR, MRbar, UCL_mr, LCL_mr, "Moving Range Chart", "MR", violations=list(avmr)), use_container_width=True)

    c1,c2,c3 = st.columns(3); c1.metric("X̄",f"{Ibar:.4f}"); c2.metric("MR̄",f"{MRbar:.4f}"); c3.metric("σ̂",f"{sigma_est:.4f}")
    if viol_i or viol_mr:
        for r,idxs in viol_i.items(): st.warning(f"I — {r}: Obs {[i+1 for i in idxs[:10]]}")
        for r,idxs in viol_mr.items(): st.warning(f"MR — {r}: Obs {[i+1 for i in idxs[:10]]}")
    else: st.success("✅ In control")

# ============================================================
# 4. p-CHART
# ============================================================
elif module == 'p_chart':
    st.header("📊 p-Chart (Proportion Defective)")
    st.markdown("Untuk data atribut — **proporsi cacat** per subgroup. Ukuran sampel boleh bervariasi.")
    src = st.radio("Data:", ['Demo', 'Manual'], horizontal=True)
    if src == 'Demo':
        nsub = st.slider("Subgroup:", 10, 60, 30); ni = st.slider("n per subgroup:", 50, 500, 100)
        shift = st.checkbox("Shift?", key='pshift')
        sa = st.slider("Shift at:", 1, nsub-1, nsub//2, key='psa') if shift else None
        sp = st.number_input("Δp:", 0.0, 0.5, 0.05, key='psp') if shift else 0
        defects, n_insp = gen_demo_attribute_p(nsub, ni, 0.05, sa, sp)
        n_arr = np.full(nsub, n_insp)
    else:
        txt = st.text_area("Data (defects,n per baris):", "5,100\n3,100\n7,100\n4,100\n6,100\n2,100\n8,100\n5,100\n4,100\n6,100")
        rows = [r.strip().split(',') for r in txt.strip().split('\n') if r.strip()]
        defects = np.array([int(r[0]) for r in rows]); n_arr = np.array([int(r[1]) for r in rows])
        nsub = len(defects)

    p_vals = defects / n_arr; pbar = defects.sum() / n_arr.sum()
    ucl_p = pbar + 3*np.sqrt(pbar*(1-pbar)/n_arr); lcl_p = np.maximum(0, pbar - 3*np.sqrt(pbar*(1-pbar)/n_arr))
    sg = np.arange(1, nsub+1)

    fig = go.Figure()
    ooc = [i for i in range(nsub) if p_vals[i] > ucl_p[i] or p_vals[i] < lcl_p[i]]
    colors = ['red' if i in ooc else 'steelblue' for i in range(nsub)]
    fig.add_trace(go.Scatter(x=sg, y=p_vals, mode='lines+markers', marker=dict(color=colors, size=7), line=dict(color='steelblue')))
    fig.add_hline(y=pbar, line=dict(color='green', width=2), annotation_text=f"p̄={pbar:.4f}")
    fig.add_trace(go.Scatter(x=sg, y=ucl_p, mode='lines', line=dict(color='red', dash='dash'), name='UCL'))
    fig.add_trace(go.Scatter(x=sg, y=lcl_p, mode='lines', line=dict(color='red', dash='dash'), name='LCL'))
    fig.update_layout(title="p-Chart", height=420, xaxis_title="Subgroup", yaxis_title="Proportion", template='plotly_white')
    st.plotly_chart(fig, use_container_width=True)

    c1,c2 = st.columns(2); c1.metric("p̄", f"{pbar:.4f}"); c2.metric("OOC Points", str(len(ooc)))
    if ooc: st.warning(f"Out of control: SG {[i+1 for i in ooc]}")
    else: st.success("✅ In control")
    with st.expander("Data"):
        st.dataframe(pd.DataFrame({'SG':sg,'Defects':defects,'n':n_arr,'p':p_vals.round(4),'UCL':ucl_p.round(4),'LCL':lcl_p.round(4),
            'Status':['❌' if i in ooc else '✅' for i in range(nsub)]}), use_container_width=True, hide_index=True)

# ============================================================
# 5. np-CHART
# ============================================================
elif module == 'np_chart':
    st.header("📊 np-Chart (Number Defective)")
    st.markdown("Untuk **jumlah cacat** per subgroup — n harus **konstan**.")
    src = st.radio("Data:", ['Demo', 'Manual'], horizontal=True)
    if src == 'Demo':
        nsub = st.slider("Subgroup:", 10, 60, 30, key='np_ns'); ni = st.slider("n:", 50, 500, 100, key='np_ni')
        defects, _ = gen_demo_attribute_p(nsub, ni, 0.05); n_arr = ni
    else:
        ni = st.number_input("n (konstan):", 10, 10000, 100)
        txt = st.text_input("Defects per SG (koma):", "5,3,7,4,6,2,8,5,4,6,3,5,7,4,2,6,5,3,8,4")
        defects = np.array([int(v.strip()) for v in txt.split(',') if v.strip()]); nsub = len(defects); n_arr = ni

    pbar = defects.sum() / (nsub * n_arr); npbar = n_arr * pbar
    UCL = npbar + 3*np.sqrt(npbar*(1-pbar)); LCL = max(0, npbar - 3*np.sqrt(npbar*(1-pbar)))
    sg = np.arange(1, nsub+1)

    viol = detect_violations(defects.astype(float), npbar, UCL, LCL)
    av = set(); [av.update(v) for v in viol.values()]
    st.plotly_chart(plot_control_chart(sg, defects.astype(float), npbar, UCL, LCL, "np-Chart", "Defects Count", violations=list(av)), use_container_width=True)
    c1,c2,c3 = st.columns(3); c1.metric("np̄", f"{npbar:.2f}"); c2.metric("p̄", f"{pbar:.4f}"); c3.metric("n", str(n_arr))
    if viol:
        for r,idxs in viol.items(): st.warning(f"{r}: SG {[i+1 for i in idxs[:10]]}")
    else: st.success("✅ In control")

# ============================================================
# 6. c-CHART
# ============================================================
elif module == 'c_chart':
    st.header("📊 c-Chart (Count of Defects)")
    st.markdown("Jumlah **cacat (defects)** per unit inspeksi — area inspeksi konstan.")
    src = st.radio("Data:", ['Demo', 'Manual'], horizontal=True)
    if src == 'Demo':
        nsub = st.slider("Subgroup:", 10, 60, 30, key='c_ns')
        shift = st.checkbox("Shift?", key='cshift')
        sa = st.slider("Shift at:", 1, nsub-1, nsub//2, key='csa') if shift else None
        sc = st.number_input("Δc:", 0.0, 20.0, 5.0, key='csc') if shift else 0
        defects = gen_demo_attribute_c(nsub, 5.0, sa, sc)
    else:
        txt = st.text_input("Defects per unit (koma):", "3,5,4,7,2,6,5,4,8,3,5,7,4,6,2,5,3,8,4,6")
        defects = np.array([int(v.strip()) for v in txt.split(',') if v.strip()]); nsub = len(defects)

    cbar = defects.mean(); UCL = cbar + 3*np.sqrt(cbar); LCL = max(0, cbar - 3*np.sqrt(cbar))
    sg = np.arange(1, nsub+1)
    viol = detect_violations(defects.astype(float), cbar, UCL, LCL)
    av = set(); [av.update(v) for v in viol.values()]
    st.plotly_chart(plot_control_chart(sg, defects.astype(float), cbar, UCL, LCL, "c-Chart", "Defect Count", violations=list(av)), use_container_width=True)
    c1,c2 = st.columns(2); c1.metric("c̄", f"{cbar:.2f}"); c2.metric("OOC", str(len(av)))
    if viol:
        for r,idxs in viol.items(): st.warning(f"{r}: SG {[i+1 for i in idxs[:10]]}")
    else: st.success("✅ In control")

# ============================================================
# 7. u-CHART
# ============================================================
elif module == 'u_chart':
    st.header("📊 u-Chart (Defects per Unit)")
    st.markdown("Rata-rata **cacat per unit** — area inspeksi boleh bervariasi.")
    src = st.radio("Data:", ['Demo', 'Manual'], horizontal=True)
    if src == 'Demo':
        nsub = st.slider("Subgroup:", 10, 60, 30, key='u_ns')
        np.random.seed(42)
        n_units = np.random.randint(3, 10, nsub)
        defects = np.array([np.random.poisson(5*n) for n in n_units])
    else:
        txt = st.text_area("defects,n_units per baris:", "15,3\n20,4\n10,2\n25,5\n18,4\n12,3\n30,6\n14,3\n22,5\n16,3")
        rows = [r.strip().split(',') for r in txt.strip().split('\n') if r.strip()]
        defects = np.array([int(r[0]) for r in rows]); n_units = np.array([int(r[1]) for r in rows]); nsub = len(defects)

    u_vals = defects / n_units; ubar = defects.sum() / n_units.sum()
    ucl_u = ubar + 3*np.sqrt(ubar/n_units); lcl_u = np.maximum(0, ubar - 3*np.sqrt(ubar/n_units))
    sg = np.arange(1, nsub+1)

    fig = go.Figure()
    ooc = [i for i in range(nsub) if u_vals[i] > ucl_u[i] or u_vals[i] < lcl_u[i]]
    colors = ['red' if i in ooc else 'steelblue' for i in range(nsub)]
    fig.add_trace(go.Scatter(x=sg, y=u_vals, mode='lines+markers', marker=dict(color=colors, size=7), line=dict(color='steelblue')))
    fig.add_hline(y=ubar, line=dict(color='green', width=2), annotation_text=f"ū={ubar:.4f}")
    fig.add_trace(go.Scatter(x=sg, y=ucl_u, mode='lines', line=dict(color='red', dash='dash'), name='UCL'))
    fig.add_trace(go.Scatter(x=sg, y=lcl_u, mode='lines', line=dict(color='red', dash='dash'), name='LCL'))
    fig.update_layout(title="u-Chart", height=420, template='plotly_white', xaxis_title="Subgroup", yaxis_title="Defects/Unit")
    st.plotly_chart(fig, use_container_width=True)
    c1,c2 = st.columns(2); c1.metric("ū", f"{ubar:.4f}"); c2.metric("OOC", str(len(ooc)))
    if ooc: st.warning(f"OOC: SG {[i+1 for i in ooc]}")
    else: st.success("✅ In control")

# ============================================================
# 8. CUSUM CHART
# ============================================================
elif module == 'cusum':
    st.header("📊 CUSUM Chart (Cumulative Sum)")
    st.markdown("Sensitif untuk mendeteksi **small persistent shifts** pada mean proses.")
    src = st.radio("Data:", ['Demo', 'Manual'], horizontal=True)
    if src == 'Demo':
        n = st.slider("n:", 30, 200, 50, key='cu_n')
        shift = st.checkbox("Shift?", key='cu_shift')
        sa = st.slider("Shift at:", 1, n-1, n//2, key='cu_sa') if shift else None
        sm = st.number_input("Shift (σ):", 0.0, 5.0, 1.0, key='cu_sm') if shift else 0
        np.random.seed(42); data = np.random.normal(0, 1, n)
        if sa: data[sa:] += sm
    else:
        txt = st.text_area("Data:", "0.2,-0.3,0.5,0.1,-0.4,0.3,-0.1,0.6,-0.2,0.1,0.4,0.8,0.7,1.1,0.9,1.3,1.0,1.5,1.2,1.4")
        data = np.array([float(v.strip()) for v in txt.replace('\n',',').split(',') if v.strip()]); n = len(data)

    c1,c2 = st.columns(2)
    target = c1.number_input("Target μ₀:", value=float(data[:min(20,n)].mean()), step=0.1)
    K = c2.number_input("K (allowance, σ):", 0.1, 5.0, 0.5, 0.1)
    H = c1.number_input("H (decision interval, σ):", 1.0, 20.0, 5.0, 0.5)
    sigma = c2.number_input("σ (known/estimated):", 0.01, 100.0, float(data[:min(20,n)].std()), 0.01)

    zi = (data - target) / sigma
    Cp = np.zeros(n); Cm = np.zeros(n)
    signals_p = []; signals_m = []
    for i in range(n):
        Cp[i] = max(0, (Cp[i-1] if i>0 else 0) + zi[i] - K)
        Cm[i] = max(0, (Cm[i-1] if i>0 else 0) - zi[i] - K)
        if Cp[i] > H: signals_p.append(i)
        if Cm[i] > H: signals_m.append(i)

    sg = np.arange(1, n+1)
    fig = make_subplots(rows=2, cols=1, subplot_titles=("CUSUM C⁺ (upward shift)", "CUSUM C⁻ (downward shift)"), shared_xaxes=True)
    fig.add_trace(go.Scatter(x=sg, y=Cp, mode='lines+markers', marker=dict(size=5, color=['red' if i in signals_p else 'steelblue' for i in range(n)]), line=dict(color='steelblue'), name='C⁺'), row=1, col=1)
    fig.add_hline(y=H, line=dict(color='red', dash='dash'), annotation_text=f"H={H}", row=1, col=1)
    fig.add_trace(go.Scatter(x=sg, y=Cm, mode='lines+markers', marker=dict(size=5, color=['red' if i in signals_m else 'crimson' for i in range(n)]), line=dict(color='crimson'), name='C⁻'), row=2, col=1)
    fig.add_hline(y=H, line=dict(color='red', dash='dash'), annotation_text=f"H={H}", row=2, col=1)
    fig.update_layout(height=550, template='plotly_white'); st.plotly_chart(fig, use_container_width=True)

    if signals_p: st.warning(f"⬆️ Upward shift signal at obs: {[i+1 for i in signals_p[:10]]}")
    if signals_m: st.warning(f"⬇️ Downward shift signal at obs: {[i+1 for i in signals_m[:10]]}")
    if not signals_p and not signals_m: st.success("✅ No shift detected")

    c1,c2,c3 = st.columns(3); c1.metric("Target", f"{target:.4f}"); c2.metric("K", f"{K}"); c3.metric("H", f"{H}")

# ============================================================
# 9. EWMA CHART
# ============================================================
elif module == 'ewma':
    st.header("📊 EWMA Chart (Exponentially Weighted Moving Average)")
    st.markdown("Weighted average — sensitif untuk **small shifts**, smoothing adjustable via λ.")
    src = st.radio("Data:", ['Demo', 'Manual'], horizontal=True)
    if src == 'Demo':
        n = st.slider("n:", 30, 200, 50, key='ew_n')
        shift = st.checkbox("Shift?", key='ew_shift')
        sa = st.slider("Shift at:", 1, n-1, n//2, key='ew_sa') if shift else None
        sm = st.number_input("Shift:", 0.0, 5.0, 1.0, key='ew_sm') if shift else 0
        np.random.seed(42); data = np.random.normal(50, 2, n)
        if sa: data[sa:] += sm
    else:
        txt = st.text_area("Data:", "50.2,49.8,50.5,51.0,49.7,49.5,50.1,50.3,49.9,50.0")
        data = np.array([float(v.strip()) for v in txt.replace('\n',',').split(',') if v.strip()]); n = len(data)

    c1,c2 = st.columns(2)
    lam = c1.slider("λ (smoothing):", 0.05, 1.0, 0.2, 0.05)
    L = c2.number_input("L (sigma multiplier):", 1.0, 5.0, 3.0, 0.1)
    target = c1.number_input("Target:", value=float(data.mean()), step=0.1, key='ew_tgt')
    sigma = c2.number_input("σ:", 0.01, 100.0, float(data.std()), 0.01, key='ew_sig')

    ewma_vals = np.zeros(n); ewma_vals[0] = lam*data[0] + (1-lam)*target
    for i in range(1, n): ewma_vals[i] = lam*data[i] + (1-lam)*ewma_vals[i-1]

    ucl_ewma = np.array([target + L*sigma*np.sqrt(lam/(2-lam)*(1-(1-lam)**(2*(i+1)))) for i in range(n)])
    lcl_ewma = np.array([target - L*sigma*np.sqrt(lam/(2-lam)*(1-(1-lam)**(2*(i+1)))) for i in range(n)])
    sg = np.arange(1, n+1)

    ooc = [i for i in range(n) if ewma_vals[i] > ucl_ewma[i] or ewma_vals[i] < lcl_ewma[i]]
    fig = go.Figure()
    colors = ['red' if i in ooc else 'steelblue' for i in range(n)]
    fig.add_trace(go.Scatter(x=sg, y=ewma_vals, mode='lines+markers', marker=dict(color=colors, size=6), line=dict(color='steelblue'), name='EWMA'))
    fig.add_trace(go.Scatter(x=sg, y=data, mode='markers', marker=dict(size=3, color='gray', opacity=0.4), name='Raw'))
    fig.add_hline(y=target, line=dict(color='green', width=2), annotation_text=f"Target={target:.2f}")
    fig.add_trace(go.Scatter(x=sg, y=ucl_ewma, mode='lines', line=dict(color='red', dash='dash'), name='UCL'))
    fig.add_trace(go.Scatter(x=sg, y=lcl_ewma, mode='lines', line=dict(color='red', dash='dash'), name='LCL'))
    fig.update_layout(title=f"EWMA Chart (λ={lam}, L={L})", height=450, template='plotly_white')
    st.plotly_chart(fig, use_container_width=True)

    c1,c2,c3 = st.columns(3); c1.metric("λ", f"{lam}"); c2.metric("L", f"{L}"); c3.metric("OOC", str(len(ooc)))
    if ooc: st.warning(f"OOC: Obs {[i+1 for i in ooc[:15]]}")
    else: st.success("✅ In control")
    st.info(f"💡 Steady-state limits: UCL={target+L*sigma*np.sqrt(lam/(2-lam)):.4f}, LCL={target-L*sigma*np.sqrt(lam/(2-lam)):.4f}")

# ============================================================
# 10. PROCESS CAPABILITY
# ============================================================
elif module == 'capability':
    st.header("📏 Process Capability Analysis")
    st.markdown("Mengukur kemampuan proses memenuhi spesifikasi: **Cp, Cpk, Pp, Ppk, Cpm**.")
    src = st.radio("Data:", ['Demo', 'Manual'], horizontal=True)
    if src == 'Demo':
        np.random.seed(42); data = np.random.normal(50, 2, 200)
        LSL_d = 44.0; USL_d = 56.0
    else:
        txt = st.text_area("Data:", ",".join([f"{np.random.normal(50,2):.2f}" for _ in range(100)]))
        data = np.array([float(v.strip()) for v in txt.replace('\n',',').split(',') if v.strip()])
        LSL_d = 44.0; USL_d = 56.0
    c1,c2 = st.columns(2); LSL = c1.number_input("LSL:", value=LSL_d); USL = c2.number_input("USL:", value=USL_d)
    target = st.number_input("Target:", value=(LSL+USL)/2)

    xbar = data.mean(); s = data.std(ddof=1); s_pop = data.std(ddof=0); n = len(data)
    sigma_w = s  # within (approx — for subgroup use Rbar/d2)

    Cp = (USL - LSL) / (6 * sigma_w)
    Cpu = (USL - xbar) / (3 * sigma_w); Cpl = (xbar - LSL) / (3 * sigma_w)
    Cpk = min(Cpu, Cpl)
    Pp = (USL - LSL) / (6 * s_pop)
    Ppu = (USL - xbar) / (3 * s_pop); Ppl = (xbar - LSL) / (3 * s_pop)
    Ppk = min(Ppu, Ppl)
    Cpm = (USL - LSL) / (6 * np.sqrt(s**2 + (xbar - target)**2))
    ppm_above = stats.norm.sf((USL - xbar) / sigma_w) * 1e6
    ppm_below = stats.norm.cdf((LSL - xbar) / sigma_w) * 1e6
    ppm_total = ppm_above + ppm_below
    yield_pct = (1 - ppm_total / 1e6) * 100
    sigma_level = stats.norm.ppf(1 - ppm_total / 2e6)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Cp", f"{Cp:.4f}"); c2.metric("Cpk", f"{Cpk:.4f}"); c3.metric("Pp", f"{Pp:.4f}"); c4.metric("Ppk", f"{Ppk:.4f}")
    c1.metric("Cpm", f"{Cpm:.4f}"); c2.metric("PPM Total", f"{ppm_total:.1f}"); c3.metric("Yield", f"{yield_pct:.4f}%"); c4.metric("σ Level", f"{sigma_level:.2f}")

    st.markdown("### Interpretasi")
    def cap_color(v):
        if v >= 1.67: return "🟢 Excellent"
        elif v >= 1.33: return "🟡 Capable"
        elif v >= 1.0: return "🟠 Marginal"
        else: return "🔴 Incapable"
    st.dataframe(pd.DataFrame([
        {"Index":"Cp","Value":f"{Cp:.4f}","Rating":cap_color(Cp),"Meaning":"Potential capability (spread)"},
        {"Index":"Cpk","Value":f"{Cpk:.4f}","Rating":cap_color(Cpk),"Meaning":"Actual capability (centering)"},
        {"Index":"Pp","Value":f"{Pp:.4f}","Rating":cap_color(Pp),"Meaning":"Overall performance (spread)"},
        {"Index":"Ppk","Value":f"{Ppk:.4f}","Rating":cap_color(Ppk),"Meaning":"Overall performance (centering)"},
        {"Index":"Cpm","Value":f"{Cpm:.4f}","Rating":cap_color(Cpm),"Meaning":"Taguchi capability (target)"},
    ]), use_container_width=True, hide_index=True)

    # Histogram + spec
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=data, nbinsx=40, histnorm='probability density', marker_color='steelblue', opacity=0.7, name='Data'))
    x_fit = np.linspace(data.min()-s, data.max()+s, 300)
    fig.add_trace(go.Scatter(x=x_fit, y=stats.norm.pdf(x_fit, xbar, s), mode='lines', line=dict(color='navy', width=2), name='Normal fit'))
    fig.add_vline(x=LSL, line=dict(color='red', width=2), annotation_text="LSL"); fig.add_vline(x=USL, line=dict(color='red', width=2), annotation_text="USL")
    fig.add_vline(x=target, line=dict(color='green', width=1, dash='dot'), annotation_text="Target")
    fig.add_vline(x=xbar, line=dict(color='blue', width=1, dash='dash'), annotation_text="X̄")
    fig.update_layout(title="Process Capability Histogram", height=450, template='plotly_white')
    st.plotly_chart(fig, use_container_width=True)

    # Six Sigma table
    with st.expander("📊 Six Sigma Reference"):
        ss = pd.DataFrame([
            {"Sigma":f"{i}σ","Cpk":f"{i/3:.2f}","PPM":[697700,308537,66807,6210,233,3.4][i-1],"Yield":f"{[30.23,69.15,93.32,99.38,99.977,99.99966][i-1]}%"}
            for i in range(1,7)
        ]); st.dataframe(ss, use_container_width=True, hide_index=True)

# ============================================================
# 11. NORMALITY TESTING
# ============================================================
elif module == 'normality':
    st.header("📐 Normality Testing")
    st.markdown("Uji normalitas data sebelum menggunakan control chart berbasis distribusi normal.")
    src = st.radio("Data:", ['Demo Normal', 'Demo Skewed', 'Manual'], horizontal=True)
    if src == 'Demo Normal': np.random.seed(42); data = np.random.normal(50, 2, 200); label = "Normal(50,2)"
    elif src == 'Demo Skewed': np.random.seed(42); data = np.random.exponential(5, 200); label = "Exponential(5)"
    else:
        txt = st.text_area("Data:", ",".join([f"{np.random.normal(50,2):.2f}" for _ in range(100)]))
        data = np.array([float(v.strip()) for v in txt.replace('\n',',').split(',') if v.strip()]); label = "User"

    n = len(data); xbar = data.mean(); s = data.std(ddof=1); skw = stats.skew(data); krt = stats.kurtosis(data)
    c1,c2,c3,c4 = st.columns(4); c1.metric("n",str(n)); c2.metric("Mean",f"{xbar:.4f}"); c3.metric("Std",f"{s:.4f}"); c4.metric("Skewness",f"{skw:.4f}")

    st.markdown("### Tests")
    tests = []
    if n <= 5000:
        sw, swp = stats.shapiro(data); tests.append(("Shapiro-Wilk", f"{sw:.4f}", f"{swp:.6f}", "✅" if swp>0.05 else "❌"))
    ks, ksp = stats.kstest(data, 'norm', args=(xbar, s)); tests.append(("Kolmogorov-Smirnov", f"{ks:.4f}", f"{ksp:.6f}", "✅" if ksp>0.05 else "❌"))
    ad = stats.anderson(data, 'norm')
    ad_pass = "✅" if ad.statistic < ad.critical_values[2] else "❌"
    tests.append(("Anderson-Darling", f"{ad.statistic:.4f}", f"CV(5%)={ad.critical_values[2]:.4f}", ad_pass))
    dag, dagp = stats.normaltest(data); tests.append(("D'Agostino-Pearson", f"{dag:.4f}", f"{dagp:.6f}", "✅" if dagp>0.05 else "❌"))
    jb = n/6*(skw**2 + (krt)**2/4); jbp = 1 - stats.chi2.cdf(jb, 2); tests.append(("Jarque-Bera", f"{jb:.4f}", f"{jbp:.6f}", "✅" if jbp>0.05 else "❌"))
    st.dataframe(pd.DataFrame(tests, columns=["Test","Statistic","p / CV","Normal?"]), use_container_width=True, hide_index=True)

    # Histogram + QQ
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Histogram + Normal Fit", "Q-Q Plot"))
    fig.add_trace(go.Histogram(x=data, nbinsx=40, histnorm='probability density', marker_color='steelblue', opacity=0.7), row=1, col=1)
    xf = np.linspace(data.min()-s, data.max()+s, 200)
    fig.add_trace(go.Scatter(x=xf, y=stats.norm.pdf(xf, xbar, s), mode='lines', line=dict(color='red', width=2)), row=1, col=1)
    th = stats.norm.ppf(np.linspace(0.01, 0.99, n)); emp = np.sort(data)
    fig.add_trace(go.Scatter(x=th, y=emp, mode='markers', marker=dict(size=4, opacity=0.5, color='steelblue')), row=1, col=2)
    mn = min(th.min(), emp.min()); mx = max(th.max(), emp.max())
    fig.add_trace(go.Scatter(x=[mn,mx], y=[xbar+s*mn, xbar+s*mx], mode='lines', line=dict(color='red', dash='dash')), row=1, col=2)
    fig.update_layout(height=420, showlegend=False, template='plotly_white'); st.plotly_chart(fig, use_container_width=True)

# ============================================================
# 12. PARETO ANALYSIS
# ============================================================
elif module == 'pareto':
    st.header("📊 Pareto Analysis")
    st.markdown("Identifikasi **vital few** penyebab masalah (80/20 rule).")
    src = st.radio("Data:", ['Demo', 'Manual'], horizontal=True)
    if src == 'Demo':
        cats = ['Surface Scratch','Dimension Error','Color Defect','Crack','Contamination','Misalignment','Porosity','Other']
        counts = np.array([45, 38, 25, 18, 12, 8, 5, 3])
    else:
        txt = st.text_area("Category,Count:", "Surface Scratch,45\nDimension Error,38\nColor Defect,25\nCrack,18\nContamination,12\nMisalignment,8\nPorosity,5\nOther,3")
        rows = [r.strip().split(',') for r in txt.strip().split('\n') if r.strip()]
        cats = [r[0] for r in rows]; counts = np.array([int(r[1]) for r in rows])

    idx = np.argsort(-counts); cats = [cats[i] for i in idx]; counts = counts[idx]
    cum_pct = np.cumsum(counts) / counts.sum() * 100; pct = counts / counts.sum() * 100

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=cats, y=counts, marker_color='steelblue', name='Count', text=counts, textposition='auto'), secondary_y=False)
    fig.add_trace(go.Scatter(x=cats, y=cum_pct, mode='lines+markers', line=dict(color='crimson', width=2), marker=dict(size=7), name='Cumulative %'), secondary_y=True)
    fig.add_hline(y=80, line=dict(color='red', dash='dot'), secondary_y=True, annotation_text="80%")
    fig.update_layout(title="Pareto Chart", height=450, template='plotly_white')
    fig.update_yaxes(title_text="Count", secondary_y=False); fig.update_yaxes(title_text="Cumulative %", secondary_y=True, range=[0, 105])
    st.plotly_chart(fig, use_container_width=True)

    vital = [cats[i] for i in range(len(cats)) if cum_pct[i] <= 80 or (i > 0 and cum_pct[i-1] < 80)]
    st.success(f"**Vital Few** (≤80%): {', '.join(vital)} — fokus perbaikan di sini!")
    st.dataframe(pd.DataFrame({'Category':cats,'Count':counts,'%':pct.round(2),'Cum%':cum_pct.round(2)}), use_container_width=True, hide_index=True)

# ============================================================
# 13. MSA (GAGE R&R)
# ============================================================
elif module == 'msa':
    st.header("🔬 MSA — Gage R&R (Measurement System Analysis)")
    st.markdown("Menilai **repeatability** dan **reproducibility** sistem pengukuran.")
    src = st.radio("Data:", ['Demo', 'Manual'], horizontal=True)
    if src == 'Demo':
        np.random.seed(42); n_parts = 10; n_ops = 3; n_trials = 3
        part_effect = np.random.normal(50, 3, n_parts)
        op_effect = np.random.normal(0, 0.5, n_ops)
        rows = []
        for p in range(n_parts):
            for o in range(n_ops):
                for t in range(n_trials):
                    rows.append({'Part':p+1,'Operator':f'Op{o+1}','Trial':t+1,'Measurement':round(part_effect[p]+op_effect[o]+np.random.normal(0,0.3),3)})
        df = pd.DataFrame(rows)
    else:
        st.markdown("Kolom: Part, Operator, Trial, Measurement")
        txt = st.text_area("CSV:", "Part,Operator,Trial,Measurement\n1,Op1,1,50.2\n1,Op1,2,50.3\n1,Op2,1,50.1\n1,Op2,2,50.4")
        df = pd.read_csv(io.StringIO(txt))

    with st.expander("Data", expanded=False): st.dataframe(df, use_container_width=True, hide_index=True)

    parts = df['Part'].unique(); operators = df['Operator'].unique()
    n_parts = len(parts); n_ops = len(operators); n_trials = len(df) // (n_parts * n_ops)
    tol = st.number_input("Tolerance (USL-LSL):", 0.01, 1000.0, 12.0, 0.1)

    # ANOVA approach
    grand_mean = df['Measurement'].mean()
    part_means = df.groupby('Part')['Measurement'].mean()
    op_means = df.groupby('Operator')['Measurement'].mean()

    SS_part = n_ops * n_trials * np.sum((part_means - grand_mean)**2)
    SS_op = n_parts * n_trials * np.sum((op_means - grand_mean)**2)
    SS_total = np.sum((df['Measurement'] - grand_mean)**2)

    cell_means = df.groupby(['Part','Operator'])['Measurement'].mean()
    SS_interact = n_trials * np.sum((cell_means - part_means[df.set_index(['Part','Operator']).index.get_level_values(0)].values - op_means[df.set_index(['Part','Operator']).index.get_level_values(1)].values + grand_mean)**2) if n_trials > 1 else 0
    SS_equip = SS_total - SS_part - SS_op - SS_interact

    df_part = n_parts - 1; df_op = n_ops - 1; df_interact = df_part * df_op
    df_equip = n_parts * n_ops * (n_trials - 1) if n_trials > 1 else max(1, len(df) - n_parts - n_ops + 1)

    MS_equip = SS_equip / max(df_equip, 1)
    repeatability = np.sqrt(max(MS_equip, 0))
    MS_op = SS_op / max(df_op, 1)
    reproducibility = np.sqrt(max((MS_op - MS_equip) / (n_parts * n_trials), 0))
    grr = np.sqrt(repeatability**2 + reproducibility**2)
    part_var = np.sqrt(max((SS_part/df_part - MS_equip) / (n_ops * n_trials), 0))
    total_var = np.sqrt(grr**2 + part_var**2)

    pct_grr = grr / total_var * 100 if total_var > 0 else 0
    pct_repeat = repeatability / total_var * 100 if total_var > 0 else 0
    pct_reprod = reproducibility / total_var * 100 if total_var > 0 else 0
    pct_part = part_var / total_var * 100 if total_var > 0 else 0
    pt_ratio = 6 * grr / tol * 100
    ndc = int(1.41 * part_var / grr) if grr > 0 else 999

    def grr_rating(pct):
        if pct < 10: return "🟢 Acceptable"
        elif pct < 30: return "🟡 Marginal"
        else: return "🔴 Unacceptable"

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("GRR %Study", f"{pct_grr:.2f}%"); c2.metric("P/T Ratio", f"{pt_ratio:.2f}%")
    c3.metric("NDC", str(ndc)); c4.metric("Rating", grr_rating(pct_grr))

    st.dataframe(pd.DataFrame([
        {"Source":"Repeatability (EV)","StdDev":f"{repeatability:.4f}","% Study Var":f"{pct_repeat:.2f}%"},
        {"Source":"Reproducibility (AV)","StdDev":f"{reproducibility:.4f}","% Study Var":f"{pct_reprod:.2f}%"},
        {"Source":"GRR (R&R)","StdDev":f"{grr:.4f}","% Study Var":f"{pct_grr:.2f}%"},
        {"Source":"Part-to-Part (PV)","StdDev":f"{part_var:.4f}","% Study Var":f"{pct_part:.2f}%"},
        {"Source":"Total Variation (TV)","StdDev":f"{total_var:.4f}","% Study Var":"100%"},
    ]), use_container_width=True, hide_index=True)

    st.markdown("### Visualization")
    fig = make_subplots(rows=1, cols=2, subplot_titles=("By Operator", "By Part"))
    for op in operators:
        od = df[df['Operator']==op]; fig.add_trace(go.Box(y=od['Measurement'], name=op), row=1, col=1)
    for p in parts[:15]:
        pd_ = df[df['Part']==p]; fig.add_trace(go.Box(y=pd_['Measurement'], name=f'P{p}', showlegend=False), row=1, col=2)
    fig.update_layout(height=400, template='plotly_white'); st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"| Criteria | Threshold | Result |\n|---|---|---|\n| %GRR | <10% OK, 10-30% marginal | **{pct_grr:.2f}%** {grr_rating(pct_grr)} |\n| P/T Ratio | <10% OK | **{pt_ratio:.2f}%** |\n| NDC | ≥5 OK | **{ndc}** {'✅' if ndc>=5 else '❌'} |")

# ============================================================
# 14. ACCEPTANCE SAMPLING
# ============================================================
elif module == 'acceptance':
    st.header("📦 Acceptance Sampling")
    st.markdown("Menentukan rencana sampling untuk inspeksi lot — berdasarkan **AQL** dan **LTPD**.")
    plan = st.selectbox("Plan:", ['Single Sampling', 'Double Sampling', 'OC Curve Explorer'])
    if plan == 'Single Sampling':
        c1,c2 = st.columns(2); N = c1.number_input("Lot size (N):", 10, 1000000, 1000)
        n_s = c2.number_input("Sample size (n):", 1, N, 50); c_acc = st.number_input("Acceptance # (c):", 0, n_s, 2)

        p_range = np.linspace(0, 0.15, 200)
        Pa = np.array([stats.binom.cdf(c_acc, n_s, p) for p in p_range])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=p_range*100, y=Pa*100, mode='lines', line=dict(color='steelblue', width=2)))
        fig.add_hline(y=95, line_dash="dot", line_color="green", annotation_text="α=0.05")
        fig.add_hline(y=10, line_dash="dot", line_color="red", annotation_text="β=0.10")
        fig.update_layout(title=f"OC Curve: n={n_s}, c={c_acc}", height=420, xaxis_title="Defect Rate (%)", yaxis_title="P(Accept) %", template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)

        # Find AQL & LTPD
        aql_idx = np.argmin(np.abs(Pa - 0.95)); ltpd_idx = np.argmin(np.abs(Pa - 0.10))
        aql_val = p_range[aql_idx]*100; ltpd_val = p_range[ltpd_idx]*100
        AOQ = Pa * p_range * (N - n_s) / N
        AOQL = AOQ.max() * 100
        ATI = n_s * Pa + N * (1 - Pa)

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("AQL (Pa=95%)", f"{aql_val:.2f}%"); c2.metric("LTPD (Pa=10%)", f"{ltpd_val:.2f}%")
        c3.metric("AOQL", f"{AOQL:.3f}%"); c4.metric("Disc. Ratio", f"{ltpd_val/aql_val:.2f}" if aql_val > 0 else "∞")

        fig2 = make_subplots(rows=1, cols=2, subplot_titles=("AOQ Curve", "ATI Curve"))
        fig2.add_trace(go.Scatter(x=p_range*100, y=AOQ*100, mode='lines', line=dict(color='green', width=2)), row=1, col=1)
        fig2.add_trace(go.Scatter(x=p_range*100, y=ATI, mode='lines', line=dict(color='crimson', width=2)), row=1, col=2)
        fig2.update_layout(height=350, template='plotly_white'); st.plotly_chart(fig2, use_container_width=True)

    elif plan == 'Double Sampling':
        st.markdown("### Double Sampling Plan")
        c1,c2 = st.columns(2); n1 = c1.number_input("n₁:", 1, 1000, 50); n2 = c2.number_input("n₂:", 1, 1000, 50)
        c1_ = c1.number_input("c₁ (accept 1st):", 0, n1, 1); r1 = c2.number_input("r₁ (reject 1st):", c1_+1, n1, 4)
        c2_v = st.number_input("c₂ (accept 2nd):", c1_, n1+n2, 4)
        p_range = np.linspace(0, 0.15, 200); Pa = []
        for p in p_range:
            pa1 = stats.binom.cdf(c1_, n1, p)
            pr1 = 1 - stats.binom.cdf(r1-1, n1, p)
            pa2 = 0
            for d1 in range(c1_+1, r1):
                pd1 = stats.binom.pmf(d1, n1, p)
                pa2 += pd1 * stats.binom.cdf(c2_v - d1, n2, p)
            Pa.append(pa1 + pa2)
        Pa = np.array(Pa)
        fig = go.Figure(data=[go.Scatter(x=p_range*100, y=Pa*100, mode='lines', line=dict(width=2))])
        fig.update_layout(title="OC Curve (Double)", height=400, xaxis_title="Defect%", yaxis_title="Pa%", template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)

    elif plan == 'OC Curve Explorer':
        st.markdown("### Compare OC Curves")
        plans = st.text_area("Plans (n,c per baris):", "30,1\n50,2\n80,3\n125,5")
        p_range = np.linspace(0, 0.15, 200); fig = go.Figure()
        for line in plans.strip().split('\n'):
            n_p, c_p = [int(v.strip()) for v in line.split(',')]
            Pa = [stats.binom.cdf(c_p, n_p, p)*100 for p in p_range]
            fig.add_trace(go.Scatter(x=p_range*100, y=Pa, mode='lines', name=f'n={n_p},c={c_p}'))
        fig.add_hline(y=95, line_dash="dot", line_color="green"); fig.add_hline(y=10, line_dash="dot", line_color="red")
        fig.update_layout(title="OC Curves Comparison", height=450, xaxis_title="Defect%", yaxis_title="Pa%", template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# 15. RELIABILITY ANALYSIS
# ============================================================
elif module == 'reliability':
    st.header("⏱️ Reliability Analysis")
    st.markdown("Analisis keandalan produk — **Weibull**, **Exponential**, hazard, survival.")
    dist = st.selectbox("Model:", ['Weibull', 'Exponential', 'Lognormal'])
    if dist == 'Weibull':
        c1,c2 = st.columns(2); beta = c1.number_input("β (shape):", 0.1, 10.0, 2.0, 0.1); eta = c2.number_input("η (scale):", 0.1, 10000.0, 100.0, 1.0)
        t = np.linspace(0.01, eta*3, 500)
        R = np.exp(-(t/eta)**beta); f_t = beta/eta*(t/eta)**(beta-1)*np.exp(-(t/eta)**beta)
        h_t = beta/eta*(t/eta)**(beta-1)
        MTTF = eta * np.exp(gammaln(1+1/beta))
    elif dist == 'Exponential':
        lam = st.number_input("λ (failure rate):", 0.0001, 10.0, 0.01, 0.001)
        t = np.linspace(0, 5/lam, 500); R = np.exp(-lam*t); f_t = lam*np.exp(-lam*t); h_t = np.full_like(t, lam)
        MTTF = 1/lam; beta = 1; eta = 1/lam
    else:
        c1,c2 = st.columns(2); mu_ln = c1.number_input("μ (log-mean):", 0.1, 20.0, 4.0, 0.1); sig_ln = c2.number_input("σ (log-std):", 0.01, 5.0, 0.5, 0.1)
        rv = stats.lognorm(sig_ln, scale=np.exp(mu_ln)); t = np.linspace(0.01, rv.ppf(0.999), 500)
        R = rv.sf(t); f_t = rv.pdf(t); h_t = f_t / R; MTTF = rv.mean(); beta = None; eta = None

    fig = make_subplots(rows=2, cols=2, subplot_titles=("Reliability R(t)", "PDF f(t)", "Hazard h(t)", "Cumulative Hazard H(t)"))
    fig.add_trace(go.Scatter(x=t, y=R, mode='lines', line=dict(color='steelblue', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=f_t, mode='lines', line=dict(color='crimson', width=2)), row=1, col=2)
    fig.add_trace(go.Scatter(x=t, y=h_t, mode='lines', line=dict(color='green', width=2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=-np.log(np.maximum(R, 1e-10)), mode='lines', line=dict(color='purple', width=2)), row=2, col=2)
    fig.update_layout(height=600, showlegend=False, template='plotly_white'); st.plotly_chart(fig, use_container_width=True)

    c1,c2,c3 = st.columns(3); c1.metric("MTTF", f"{MTTF:.2f}")
    if beta: c2.metric("β", f"{beta}"); c3.metric("η", f"{eta}")
    t_q = st.number_input("t:", value=MTTF/2, step=1.0)
    idx_t = np.argmin(np.abs(t - t_q))
    st.success(f"R({t_q:.1f}) = **{R[idx_t]:.4f}** | F({t_q:.1f}) = **{1-R[idx_t]:.4f}** | h({t_q:.1f}) = **{h_t[idx_t]:.6f}**")

    if beta is not None:
        st.markdown("### Bathtub Curve Interpretation")
        if beta < 1: st.info("β < 1 → **Infant mortality** (decreasing failure rate)")
        elif abs(beta - 1) < 0.05: st.info("β ≈ 1 → **Random failures** (constant rate = exponential)")
        else: st.info("β > 1 → **Wear-out** (increasing failure rate)")

# ============================================================
# 16. DOE (DESIGN OF EXPERIMENTS)
# ============================================================
elif module == 'doe':
    st.header("🧪 Design of Experiments (DOE)")
    st.markdown("Analisis **2^k factorial** dan **fractional factorial** design.")
    sub = st.selectbox("Topik:", ['Full Factorial 2²', 'Full Factorial 2³', 'Custom Analysis'])
    if sub == 'Full Factorial 2²':
        st.subheader("2² Full Factorial")
        c1,c2 = st.columns(2)
        fa = c1.text_input("Factor A:", "Temperature"); fb = c2.text_input("Factor B:", "Pressure")
        st.markdown("Masukkan response (urutan: --, -+, +-, ++):")
        c1,c2,c3,c4 = st.columns(4)
        y1 = c1.number_input(f"({fa}-, {fb}-):", value=45.0); y2 = c2.number_input(f"({fa}-, {fb}+):", value=71.0)
        y3 = c3.number_input(f"({fa}+, {fb}-):", value=48.0); y4 = c4.number_input(f"({fa}+, {fb}+):", value=65.0)
        rep = st.number_input("Replikasi:", 1, 10, 1)

        A_eff = ((y3 + y4) - (y1 + y2)) / 2; B_eff = ((y2 + y4) - (y1 + y3)) / 2
        AB_eff = ((y1 + y4) - (y2 + y3)) / 2
        st.markdown(f"| Effect | Value |\n|---|---|\n| {fa} (A) | **{A_eff:.4f}** |\n| {fb} (B) | **{B_eff:.4f}** |\n| A×B | **{AB_eff:.4f}** |")

        fig = go.Figure(data=[go.Bar(x=[fa, fb, f'{fa}×{fb}'], y=[abs(A_eff), abs(B_eff), abs(AB_eff)],
            marker_color=['steelblue','crimson','green'], text=[f'{A_eff:.2f}',f'{B_eff:.2f}',f'{AB_eff:.2f}'], textposition='auto')])
        fig.update_layout(title="Main & Interaction Effects", height=380, template='plotly_white'); st.plotly_chart(fig, use_container_width=True)

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=['-','+'], y=[np.mean([y1,y2]), np.mean([y3,y4])], mode='lines+markers', name=fa, line=dict(width=2)))
        fig2.add_trace(go.Scatter(x=['-','+'], y=[np.mean([y1,y3]), np.mean([y2,y4])], mode='lines+markers', name=fb, line=dict(width=2)))
        fig2.update_layout(title="Main Effect Plots", height=350, template='plotly_white'); st.plotly_chart(fig2, use_container_width=True)

        st.markdown("### Interaction Plot")
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=['-','+'], y=[y1, y3], mode='lines+markers', name=f'{fb} = -', line=dict(width=2)))
        fig3.add_trace(go.Scatter(x=['-','+'], y=[y2, y4], mode='lines+markers', name=f'{fb} = +', line=dict(width=2, dash='dash')))
        fig3.update_layout(title=f"Interaction: {fa} × {fb}", height=350, xaxis_title=fa, yaxis_title="Response", template='plotly_white')
        st.plotly_chart(fig3, use_container_width=True)

    elif sub == 'Full Factorial 2³':
        st.subheader("2³ Full Factorial")
        c1,c2,c3 = st.columns(3)
        fa = c1.text_input("A:", "Temp"); fb = c2.text_input("B:", "Pressure"); fc = c3.text_input("C:", "Speed")
        st.markdown("Response (urutan: ---, --+, -+-, -++, +--, +-+, ++-, +++):")
        cols = st.columns(8)
        combos = ['---','--+','-+-','-++','+--','+-+','++-','+++']
        y = [cols[i].number_input(combos[i], value=40.0+i*5.0+np.random.normal(0,2), key=f'doe{i}') for i in range(8)]

        signs = np.array([[-1,-1,-1],[-1,-1,1],[-1,1,-1],[-1,1,1],[1,-1,-1],[1,-1,1],[1,1,-1],[1,1,1]])
        y_arr = np.array(y); effects = {}
        for i, name in enumerate([fa, fb, fc]):
            effects[name] = np.mean(y_arr[signs[:, i] == 1]) - np.mean(y_arr[signs[:, i] == -1])
        effects[f'{fa}×{fb}'] = np.mean(y_arr * signs[:,0] * signs[:,1]) * 2 / len(y_arr) * 2
        effects[f'{fa}×{fc}'] = np.mean(y_arr * signs[:,0] * signs[:,2]) * 2 / len(y_arr) * 2
        effects[f'{fb}×{fc}'] = np.mean(y_arr * signs[:,1] * signs[:,2]) * 2 / len(y_arr) * 2
        effects[f'{fa}×{fb}×{fc}'] = np.mean(y_arr * signs[:,0] * signs[:,1] * signs[:,2]) * 2 / len(y_arr) * 2

        eff_df = pd.DataFrame([(k, v) for k, v in effects.items()], columns=['Effect', 'Value']).sort_values('Value', key=abs, ascending=False)
        st.dataframe(eff_df.round(4), use_container_width=True, hide_index=True)

        fig = go.Figure(data=[go.Bar(x=eff_df['Effect'], y=eff_df['Value'].abs(), text=eff_df['Value'].round(2), textposition='auto', marker_color='steelblue')])
        fig.update_layout(title="Effect Magnitudes (Pareto of Effects)", height=400, template='plotly_white'); st.plotly_chart(fig, use_container_width=True)

    elif sub == 'Custom Analysis':
        st.subheader("Custom DOE Data Analysis")
        txt = st.text_area("CSV (Factor columns + Response):", "A,B,Response\n-1,-1,45\n-1,1,71\n1,-1,48\n1,1,65\n-1,-1,43\n-1,1,68\n1,-1,50\n1,1,63")
        df = pd.read_csv(io.StringIO(txt))
        st.dataframe(df, use_container_width=True, hide_index=True)
        resp_col = st.selectbox("Response:", df.columns)
        factor_cols = [c for c in df.columns if c != resp_col]
        st.markdown("### Effects")
        for fc in factor_cols:
            hi = df[df[fc] > 0][resp_col].mean(); lo = df[df[fc] <= 0][resp_col].mean()
            st.markdown(f"**{fc}**: Effect = {hi - lo:.4f} (High={hi:.2f}, Low={lo:.2f})")

# ============================================================
# FOOTER
# ============================================================
st.divider()
st.markdown("<div style='text-align:center;color:gray;font-size:0.85rem'><b>Statistical Quality Control Pro</b><br>16 Modul | Control Charts · Capability · MSA · Sampling · Reliability · DOE | 2026</div>", unsafe_allow_html=True)
