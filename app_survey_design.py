import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Perancangan Survey", page_icon="📋", layout="wide")
st.title("📋 Perancangan Survey — Complete Toolkit")
st.markdown("""
Aplikasi lengkap untuk **perancangan survey**: penentuan ukuran sampel (6 metode),
teknik sampling (7 metode), alokasi strata, design effect, margin of error,
power analysis, dan visualisasi interaktif.
""")

# ============================
# HELPERS
# ============================
Z_MAP = {90: 1.645, 95: 1.96, 99: 2.576}

def cochran_infinite(z, p, e):
    return (z**2 * p * (1 - p)) / e**2

def cochran_finite(n0, N):
    return n0 / (1 + (n0 - 1) / N) if N > 0 else n0

def slovin(N, e):
    return N / (1 + N * e**2) if (1 + N * e**2) > 0 else N

def yamane(N, e):
    return N / (1 + N * e**2)

def sample_mean(z, sigma, e):
    return (z * sigma / e)**2

def sample_mean_finite(z, sigma, e, N):
    n0 = (z * sigma / e)**2
    return n0 / (1 + (n0 - 1) / N)

def design_effect(rho, m):
    return 1 + rho * (m - 1) if m > 1 else 1

def effective_sample(n, deff):
    return n / deff if deff > 0 else n

# ============================
# SIDEBAR NAVIGATION
# ============================
st.sidebar.header("📑 Navigasi")
module = st.sidebar.radio("Pilih Modul:", [
    "1. Penentuan Ukuran Sampel",
    "2. Teknik Sampling & Alokasi",
    "3. Margin of Error & Power Analysis",
    "4. Design Effect & Cluster",
    "5. Kuesioner & Skala",
    "6. Simulasi Sampling",
    "7. Laporan & Ekspor"
])

# ================================================================
# MODULE 1: SAMPLE SIZE
# ================================================================
if "1." in module:
    st.header("1. Penentuan Ukuran Sampel")
    st.markdown("Hitung ukuran sampel minimum berdasarkan **6 metode** berbeda.")

    method = st.selectbox("Metode Perhitungan", [
        "Cochran (Proporsi — Populasi Tak Terhingga)",
        "Cochran (Proporsi — Koreksi Populasi Terhingga)",
        "Slovin",
        "Yamane",
        "Estimasi Mean (Populasi Tak Terhingga)",
        "Estimasi Mean (Populasi Terhingga)",
    ])

    col1, col2, col3 = st.columns(3)

    if "Cochran" in method and "Tak Terhingga" in method:
        with col1:
            conf = st.selectbox("Confidence Level (%)", [90, 95, 99], index=1, key='c1')
        with col2:
            p_est = st.slider("Estimasi Proporsi (p)", 0.01, 0.99, 0.50, 0.01, key='p1')
        with col3:
            moe = st.slider("Margin of Error (e)", 0.01, 0.20, 0.05, 0.01, key='e1')

        z = Z_MAP[conf]
        n0 = cochran_infinite(z, p_est, moe)
        n_final = int(np.ceil(n0))

        st.success(f"### Ukuran Sampel Minimum: **n = {n_final}**")

        formula = f"n₀ = Z² × p(1-p) / e² = {z}² × {p_est}×{1-p_est:.2f} / {moe}² = **{n_final}**"
        st.latex(r"n_0 = \frac{Z^2 \cdot p(1-p)}{e^2} = \frac{%.3f^2 \times %.2f \times %.2f}{%.3f^2} = %d" %
                 (z, p_est, 1-p_est, moe, n_final))

    elif "Cochran" in method and "Terhingga" in method:
        with col1:
            conf = st.selectbox("Confidence Level (%)", [90, 95, 99], index=1, key='c2')
        with col2:
            p_est = st.slider("Estimasi Proporsi (p)", 0.01, 0.99, 0.50, 0.01, key='p2')
        with col3:
            moe = st.slider("Margin of Error (e)", 0.01, 0.20, 0.05, 0.01, key='e2')

        N_pop = st.number_input("Ukuran Populasi (N)", 10, 10_000_000, 10000, 100, key='N2')
        z = Z_MAP[conf]
        n0 = cochran_infinite(z, p_est, moe)
        n_final = int(np.ceil(cochran_finite(n0, N_pop)))

        st.success(f"### Ukuran Sampel Minimum: **n = {n_final}** (dari populasi N={N_pop:,})")
        st.latex(r"n_0 = \frac{Z^2 \cdot p(1-p)}{e^2} = %d" % int(np.ceil(n0)))
        st.latex(r"n = \frac{n_0}{1 + \frac{n_0 - 1}{N}} = \frac{%d}{1 + \frac{%d}{%s}} = %d" %
                 (int(np.ceil(n0)), int(np.ceil(n0))-1, f"{N_pop:,}", n_final))

    elif "Slovin" in method:
        with col1:
            N_pop = st.number_input("Ukuran Populasi (N)", 10, 10_000_000, 10000, 100, key='Ns')
        with col2:
            moe = st.slider("Margin of Error (e)", 0.01, 0.20, 0.05, 0.01, key='es')
        n_final = int(np.ceil(slovin(N_pop, moe)))
        st.success(f"### Ukuran Sampel Minimum: **n = {n_final}**")
        st.latex(r"n = \frac{N}{1 + Ne^2} = \frac{%s}{1 + %s \times %.3f^2} = %d" %
                 (f"{N_pop:,}", f"{N_pop:,}", moe, n_final))

    elif "Yamane" in method:
        with col1:
            N_pop = st.number_input("Ukuran Populasi (N)", 10, 10_000_000, 10000, 100, key='Ny')
        with col2:
            moe = st.slider("Margin of Error (e)", 0.01, 0.20, 0.05, 0.01, key='ey')
        n_final = int(np.ceil(yamane(N_pop, moe)))
        st.success(f"### Ukuran Sampel Minimum: **n = {n_final}**")
        st.latex(r"n = \frac{N}{1 + Ne^2} = %d" % n_final)

    elif "Mean" in method and "Tak Terhingga" in method:
        with col1:
            conf = st.selectbox("Confidence Level (%)", [90, 95, 99], index=1, key='cm')
        with col2:
            sigma = st.number_input("Standar Deviasi Populasi (σ)", 0.1, 1000.0, 10.0, 0.1, key='sm')
        with col3:
            moe_abs = st.number_input("Margin of Error (satuan asli)", 0.01, 100.0, 2.0, 0.1, key='em')
        z = Z_MAP[conf]
        n_final = int(np.ceil(sample_mean(z, sigma, moe_abs)))
        st.success(f"### Ukuran Sampel Minimum: **n = {n_final}**")
        st.latex(r"n = \left(\frac{Z \cdot \sigma}{e}\right)^2 = \left(\frac{%.3f \times %.1f}{%.1f}\right)^2 = %d"
                 % (z, sigma, moe_abs, n_final))

    else:  # Mean finite
        with col1:
            conf = st.selectbox("Confidence Level (%)", [90, 95, 99], index=1, key='cmf')
        with col2:
            sigma = st.number_input("Standar Deviasi Populasi (σ)", 0.1, 1000.0, 10.0, 0.1, key='smf')
        with col3:
            moe_abs = st.number_input("Margin of Error (satuan asli)", 0.01, 100.0, 2.0, 0.1, key='emf')
        N_pop = st.number_input("Ukuran Populasi (N)", 10, 10_000_000, 10000, 100, key='Nmf')
        z = Z_MAP[conf]
        n_final = int(np.ceil(sample_mean_finite(z, sigma, moe_abs, N_pop)))
        st.success(f"### Ukuran Sampel Minimum: **n = {n_final}**")

    # Sensitivity table
    st.subheader("Tabel Sensitivitas Ukuran Sampel")
    st.markdown("Bagaimana **margin of error** dan **confidence level** mempengaruhi n.")

    sens_rows = []
    for cl in [90, 95, 99]:
        for e_val in [0.01, 0.02, 0.03, 0.05, 0.07, 0.10]:
            z_s = Z_MAP[cl]
            n_s = int(np.ceil(cochran_infinite(z_s, 0.5, e_val)))
            sens_rows.append({'Confidence (%)': cl, 'MoE': e_val, 'n (p=0.5, ∞)': n_s})
    sens_df = pd.DataFrame(sens_rows)
    pivot_sens = sens_df.pivot(index='MoE', columns='Confidence (%)', values='n (p=0.5, ∞)')
    st.dataframe(pivot_sens, use_container_width=True)

    fig_sens = px.line(sens_df, x='MoE', y='n (p=0.5, ∞)', color='Confidence (%)',
                        markers=True, title="Ukuran Sampel vs Margin of Error (Cochran, p=0.5)")
    fig_sens.update_layout(height=400, xaxis_title="Margin of Error", yaxis_title="Ukuran Sampel (n)")
    st.plotly_chart(fig_sens, use_container_width=True)

    # Proportion sensitivity
    st.subheader("Sensitivitas terhadap Proporsi (p)")
    p_range = np.arange(0.05, 1.0, 0.05)
    p_sens = [int(np.ceil(cochran_infinite(1.96, p, 0.05))) for p in p_range]
    fig_p = px.bar(x=p_range.round(2), y=p_sens, labels={'x': 'Proporsi (p)', 'y': 'n'},
                    title="Ukuran Sampel vs Proporsi (95% CL, MoE=5%)")
    st.plotly_chart(fig_p, use_container_width=True)

    # Finite population correction chart
    st.subheader("Efek Koreksi Populasi Terhingga (FPC)")
    N_range = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 50000, 100000]
    n0_base = cochran_infinite(1.96, 0.5, 0.05)
    fpc_data = [{'N': N_v, 'n (tanpa FPC)': int(np.ceil(n0_base)),
                 'n (dengan FPC)': int(np.ceil(cochran_finite(n0_base, N_v))),
                 'Sampling Fraction (%)': round(cochran_finite(n0_base, N_v)/N_v*100, 1)}
                for N_v in N_range]
    fpc_df = pd.DataFrame(fpc_data)
    st.dataframe(fpc_df, use_container_width=True, hide_index=True)

# ================================================================
# MODULE 2: SAMPLING TECHNIQUES & ALLOCATION
# ================================================================
elif "2." in module:
    st.header("2. Teknik Sampling & Alokasi Strata")

    sampling_method = st.selectbox("Teknik Sampling", [
        "Simple Random Sampling (SRS)",
        "Systematic Sampling",
        "Stratified Sampling — Proportional Allocation",
        "Stratified Sampling — Neyman (Optimal) Allocation",
        "Stratified Sampling — Equal Allocation",
        "Cluster Sampling",
        "Multistage Sampling"
    ])

    # ---- SRS ----
    if "Simple Random" in sampling_method:
        st.subheader("Simple Random Sampling (SRS)")
        st.markdown("Setiap anggota populasi memiliki probabilitas yang **sama** untuk terpilih.")

        col1, col2 = st.columns(2)
        with col1:
            N_pop = st.number_input("Ukuran Populasi (N)", 10, 1_000_000, 1000, key='srs_N')
        with col2:
            n_sample = st.number_input("Ukuran Sampel (n)", 1, 100000, 100, key='srs_n')

        prob = n_sample / N_pop
        st.markdown(f"**Probabilitas Seleksi:** {prob:.4f} ({prob*100:.2f}%)")

        if st.button("Generate Sampel SRS", type="primary"):
            np.random.seed(42)
            selected = np.sort(np.random.choice(N_pop, size=min(n_sample, N_pop), replace=False)) + 1
            st.dataframe(pd.DataFrame({'ID Terpilih': selected}), use_container_width=True, hide_index=True)
            fig = px.histogram(x=selected, nbins=30, title="Distribusi ID Sampel Terpilih",
                                labels={'x': 'ID', 'y': 'Frekuensi'})
            st.plotly_chart(fig, use_container_width=True)

    # ---- SYSTEMATIC ----
    elif "Systematic" in sampling_method:
        st.subheader("Systematic Sampling")
        st.markdown("Pilih setiap **k-th** elemen dari populasi setelah start acak.")

        col1, col2 = st.columns(2)
        with col1:
            N_pop = st.number_input("Ukuran Populasi (N)", 10, 1_000_000, 1000, key='sys_N')
        with col2:
            n_sample = st.number_input("Ukuran Sampel (n)", 1, 100000, 100, key='sys_n')

        k = max(1, int(N_pop / n_sample))
        st.markdown(f"**Interval Sampling (k):** {k}")

        start = st.number_input("Random Start (1 — k)", 1, k, 1)
        selected = np.arange(start, N_pop + 1, k)[:n_sample]

        result_df = pd.DataFrame({'Urutan': range(1, len(selected)+1), 'ID Terpilih': selected})
        st.dataframe(result_df, use_container_width=True, hide_index=True)

        fig = go.Figure()
        all_ids = np.arange(1, min(200, N_pop)+1)
        colors = ['red' if i in selected else 'lightgray' for i in all_ids]
        fig.add_trace(go.Bar(x=all_ids, y=[1]*len(all_ids), marker_color=colors))
        fig.update_layout(title=f"Systematic Sampling (k={k}, start={start}) — first 200",
                          xaxis_title="ID Populasi", showlegend=False, height=300)
        st.plotly_chart(fig, use_container_width=True)

    # ---- STRATIFIED ----
    elif "Stratified" in sampling_method:
        alloc_type = "Proportional" if "Proportional" in sampling_method else \
                     "Neyman" if "Neyman" in sampling_method else "Equal"

        st.subheader(f"Stratified Sampling — {alloc_type} Allocation")

        n_total = st.number_input("Total Sampel (n)", 10, 100000, 200, key='strat_n')
        n_strata = st.number_input("Jumlah Strata", 2, 20, 4, key='strat_k')

        st.markdown("**Input Data per Stratum:**")
        strata_data = []
        cols = st.columns(min(4, n_strata))
        for i in range(n_strata):
            with cols[i % len(cols)]:
                st.markdown(f"**Stratum {i+1}**")
                Ni = st.number_input(f"Nᵢ (populasi)", 10, 1000000, (i+1)*500, key=f'Ni_{i}')
                if alloc_type == "Neyman":
                    Si = st.number_input(f"σᵢ (std. dev)", 0.1, 1000.0, float(5+i*3), key=f'Si_{i}')
                else:
                    Si = 1.0
                strata_data.append({'Stratum': f'Stratum {i+1}', 'Nᵢ': Ni, 'σᵢ': Si})

        sdf = pd.DataFrame(strata_data)
        N_total = sdf['Nᵢ'].sum()

        if alloc_type == "Proportional":
            sdf['Wᵢ'] = sdf['Nᵢ'] / N_total
            sdf['nᵢ'] = (sdf['Wᵢ'] * n_total).round(0).astype(int)
        elif alloc_type == "Neyman":
            sdf['Nᵢσᵢ'] = sdf['Nᵢ'] * sdf['σᵢ']
            total_Ns = sdf['Nᵢσᵢ'].sum()
            sdf['Wᵢ'] = sdf['Nᵢσᵢ'] / total_Ns
            sdf['nᵢ'] = (sdf['Wᵢ'] * n_total).round(0).astype(int)
        else:  # Equal
            sdf['Wᵢ'] = 1 / n_strata
            sdf['nᵢ'] = (n_total / n_strata * np.ones(n_strata)).round(0).astype(int)

        # Adjust rounding
        diff = n_total - sdf['nᵢ'].sum()
        if diff != 0:
            sdf.loc[sdf['Nᵢ'].idxmax(), 'nᵢ'] += diff

        sdf['Sampling Fraction'] = (sdf['nᵢ'] / sdf['Nᵢ']).round(4)
        st.dataframe(sdf, use_container_width=True, hide_index=True)
        st.info(f"**Total Populasi:** {N_total:,} | **Total Sampel:** {int(sdf['nᵢ'].sum())} | "
                f"**Overall Sampling Fraction:** {sdf['nᵢ'].sum()/N_total:.4f}")

        fig_strat = make_subplots(rows=1, cols=2, subplot_titles=("Populasi per Stratum", "Sampel per Stratum"))
        fig_strat.add_trace(go.Bar(x=sdf['Stratum'], y=sdf['Nᵢ'], name='Populasi', marker_color='steelblue'),
                            row=1, col=1)
        fig_strat.add_trace(go.Bar(x=sdf['Stratum'], y=sdf['nᵢ'], name='Sampel', marker_color='salmon'),
                            row=1, col=2)
        fig_strat.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_strat, use_container_width=True)

        # Pie chart
        fig_pie = px.pie(sdf, values='nᵢ', names='Stratum', title="Alokasi Sampel per Stratum")
        st.plotly_chart(fig_pie, use_container_width=True)

    # ---- CLUSTER ----
    elif "Cluster" in sampling_method and "Multi" not in sampling_method:
        st.subheader("Cluster Sampling")
        st.markdown("Populasi dibagi menjadi cluster, lalu cluster dipilih secara acak. **Seluruh** anggota cluster terpilih disurvei.")

        col1, col2, col3 = st.columns(3)
        with col1:
            n_clusters_total = st.number_input("Jumlah Cluster di Populasi", 5, 10000, 50, key='cl_M')
        with col2:
            n_clusters_select = st.number_input("Jumlah Cluster Terpilih", 1, 1000, 10, key='cl_m')
        with col3:
            avg_cluster_size = st.number_input("Rata-rata Ukuran Cluster", 5, 10000, 30, key='cl_b')

        n_total_est = n_clusters_select * avg_cluster_size
        st.success(f"**Estimasi Total Sampel:** {n_total_est}")

        if st.button("Pilih Cluster Secara Acak", type="primary"):
            np.random.seed(42)
            selected_cl = np.sort(np.random.choice(n_clusters_total, n_clusters_select, replace=False)) + 1
            cl_sizes = np.random.poisson(avg_cluster_size, n_clusters_select)
            cl_sizes = np.maximum(cl_sizes, 1)
            cl_df = pd.DataFrame({'Cluster ID': selected_cl, 'Ukuran Cluster': cl_sizes})
            st.dataframe(cl_df, use_container_width=True, hide_index=True)
            st.markdown(f"**Total Sampel Aktual:** {cl_sizes.sum()}")

            fig = px.bar(cl_df, x='Cluster ID', y='Ukuran Cluster', color='Ukuran Cluster',
                          title="Ukuran per Cluster Terpilih", color_continuous_scale='Viridis')
            st.plotly_chart(fig, use_container_width=True)

    # ---- MULTISTAGE ----
    else:
        st.subheader("Multistage Sampling")
        st.markdown("Sampling bertahap: Tahap 1 pilih cluster (PSU), Tahap 2 pilih sub-unit (SSU).")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            total_psu = st.number_input("Total PSU", 10, 10000, 100, key='ms_psu')
        with col2:
            select_psu = st.number_input("PSU Terpilih", 2, 1000, 20, key='ms_psu_s')
        with col3:
            ssu_per_psu = st.number_input("SSU per PSU", 5, 10000, 50, key='ms_ssu')
        with col4:
            select_ssu = st.number_input("SSU Terpilih per PSU", 1, 1000, 15, key='ms_ssu_s')

        total_sample = select_psu * select_ssu
        st.success(f"**Total Sampel:** {select_psu} PSU × {select_ssu} SSU = **{total_sample}**")

        if st.button("Simulasi Multistage", type="primary"):
            np.random.seed(42)
            psu_ids = np.sort(np.random.choice(total_psu, select_psu, replace=False)) + 1
            ms_rows = []
            for psu in psu_ids:
                ssus = np.sort(np.random.choice(ssu_per_psu, select_ssu, replace=False)) + 1
                for ssu in ssus:
                    ms_rows.append({'PSU': psu, 'SSU': ssu})
            ms_df = pd.DataFrame(ms_rows)
            st.dataframe(ms_df.head(50), use_container_width=True, hide_index=True)

            fig = px.scatter(ms_df, x='PSU', y='SSU', title="Multistage Sampling Result",
                              color='PSU', opacity=0.7)
            st.plotly_chart(fig, use_container_width=True)

# ================================================================
# MODULE 3: MARGIN OF ERROR & POWER
# ================================================================
elif "3." in module:
    st.header("3. Margin of Error & Power Analysis")

    tab1, tab2, tab3 = st.tabs(["MoE Calculator", "Power Analysis", "Confidence Interval"])

    with tab1:
        st.subheader("Margin of Error dari Sampel yang Ada")
        col1, col2, col3 = st.columns(3)
        with col1:
            n_exist = st.number_input("Ukuran Sampel (n)", 10, 1000000, 384, key='moe_n')
        with col2:
            conf_moe = st.selectbox("Confidence Level (%)", [90, 95, 99], index=1, key='moe_c')
        with col3:
            p_moe = st.slider("Proporsi (p)", 0.01, 0.99, 0.50, 0.01, key='moe_p')

        z_moe = Z_MAP[conf_moe]
        moe_calc = z_moe * np.sqrt(p_moe * (1 - p_moe) / n_exist)
        st.success(f"### Margin of Error: **± {moe_calc:.4f}** ({moe_calc*100:.2f}%)")
        st.latex(r"e = Z \sqrt{\frac{p(1-p)}{n}} = %.3f \sqrt{\frac{%.2f \times %.2f}{%d}} = %.4f" %
                 (z_moe, p_moe, 1-p_moe, n_exist, moe_calc))

        # MoE vs n chart
        n_range = np.arange(20, 2001, 10)
        moe_range = z_moe * np.sqrt(0.5 * 0.5 / n_range)
        fig = px.line(x=n_range, y=moe_range * 100, labels={'x': 'Ukuran Sampel (n)', 'y': 'MoE (%)'},
                       title=f"Margin of Error vs Ukuran Sampel ({conf_moe}% CL, p=0.5)")
        fig.add_hline(y=5, line_dash="dash", line_color="red", annotation_text="MoE = 5%")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Power Analysis untuk Proporsi")
        st.markdown("Berapa probabilitas mendeteksi efek tertentu?")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            p0 = st.slider("H₀: p₀", 0.01, 0.99, 0.50, 0.01, key='pa_p0')
        with col2:
            p1 = st.slider("H₁: p₁ (true)", 0.01, 0.99, 0.60, 0.01, key='pa_p1')
        with col3:
            alpha_pa = st.selectbox("α", [0.01, 0.05, 0.10], index=1, key='pa_a')
        with col4:
            n_pa = st.number_input("n", 10, 100000, 200, key='pa_n')

        z_alpha = stats.norm.ppf(1 - alpha_pa / 2)
        se0 = np.sqrt(p0 * (1-p0) / n_pa)
        se1 = np.sqrt(p1 * (1-p1) / n_pa)
        z_power = (abs(p1 - p0) - z_alpha * se0) / se1
        power = stats.norm.cdf(z_power)

        st.success(f"### Power = **{power:.4f}** ({power*100:.1f}%)")
        st.markdown(f"{'✅ Cukup (power ≥ 80%)' if power >= 0.8 else '⚠️ Kurang (power < 80%)'}")

        # Power curve
        n_power_range = np.arange(20, 2001, 10)
        powers = []
        for n_i in n_power_range:
            se0_i = np.sqrt(p0*(1-p0)/n_i)
            se1_i = np.sqrt(p1*(1-p1)/n_i)
            z_i = (abs(p1-p0) - z_alpha*se0_i) / se1_i
            powers.append(stats.norm.cdf(z_i))
        fig = px.line(x=n_power_range, y=powers, labels={'x': 'n', 'y': 'Power'},
                       title=f"Power Curve (p₀={p0}, p₁={p1}, α={alpha_pa})")
        fig.add_hline(y=0.8, line_dash="dash", line_color="red", annotation_text="Power = 80%")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

        # Find minimum n for 80% power
        for n_min in range(10, 100001):
            se0_m = np.sqrt(p0*(1-p0)/n_min)
            se1_m = np.sqrt(p1*(1-p1)/n_min)
            z_m = (abs(p1-p0) - z_alpha*se0_m) / se1_m
            if stats.norm.cdf(z_m) >= 0.8:
                break
        st.info(f"**Sampel minimum untuk Power ≥ 80%:** n = {n_min}")

    with tab3:
        st.subheader("Confidence Interval Calculator")
        ci_type = st.radio("Tipe Data", ["Proporsi", "Mean"], key='ci_type')

        if ci_type == "Proporsi":
            col1, col2, col3 = st.columns(3)
            with col1:
                p_ci = st.slider("Proporsi sampel (p̂)", 0.01, 0.99, 0.45, 0.01, key='ci_p')
            with col2:
                n_ci = st.number_input("n", 10, 1000000, 300, key='ci_n')
            with col3:
                conf_ci = st.selectbox("CL (%)", [90, 95, 99], index=1, key='ci_c')

            z_ci = Z_MAP[conf_ci]
            se_ci = np.sqrt(p_ci * (1-p_ci) / n_ci)
            ci_low = p_ci - z_ci * se_ci
            ci_up = p_ci + z_ci * se_ci
            st.success(f"### CI: [{ci_low:.4f}, {ci_up:.4f}]")
            st.markdown(f"**MoE:** ± {z_ci*se_ci:.4f}")
        else:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                mean_ci = st.number_input("Mean sampel (x̄)", -1e6, 1e6, 50.0, key='ci_mean')
            with col2:
                sd_ci = st.number_input("Std. Dev sampel (s)", 0.01, 1e6, 10.0, key='ci_sd')
            with col3:
                n_ci = st.number_input("n", 2, 1000000, 100, key='ci_n2')
            with col4:
                conf_ci = st.selectbox("CL (%)", [90, 95, 99], index=1, key='ci_c2')

            z_ci = Z_MAP[conf_ci]
            se_ci = sd_ci / np.sqrt(n_ci)
            ci_low = mean_ci - z_ci * se_ci
            ci_up = mean_ci + z_ci * se_ci
            st.success(f"### CI: [{ci_low:.4f}, {ci_up:.4f}]")

# ================================================================
# MODULE 4: DESIGN EFFECT
# ================================================================
elif "4." in module:
    st.header("4. Design Effect (DEFF) & Cluster Adjustment")
    st.markdown("Mengukur **hilangnya efisiensi** akibat desain sampling yang kompleks (cluster) dibanding SRS.")

    col1, col2 = st.columns(2)
    with col1:
        rho = st.slider("Intraclass Correlation (ρ)", 0.00, 1.00, 0.05, 0.01,
                          help="ICC — kesamaan antar anggota dalam cluster. Makin tinggi → DEFF makin besar.")
    with col2:
        m_cluster = st.number_input("Rata-rata Ukuran Cluster (m)", 2, 1000, 25,
                                     help="Jumlah responden rata-rata per cluster.")

    deff = design_effect(rho, m_cluster)
    st.success(f"### Design Effect (DEFF) = **{deff:.4f}**")
    st.latex(r"DEFF = 1 + \rho(m - 1) = 1 + %.2f \times (%d - 1) = %.4f" % (rho, m_cluster, deff))

    st.markdown(f"**Artinya:** Anda membutuhkan sampel **{deff:.1f}× lebih besar** dari perhitungan SRS untuk presisi yang sama.")

    # Adjusted sample size
    st.subheader("Koreksi Ukuran Sampel")
    n_srs = st.number_input("Ukuran Sampel SRS (n₀)", 10, 100000, 384, key='deff_n')
    n_adj = int(np.ceil(n_srs * deff))
    n_eff = int(np.ceil(effective_sample(n_adj, deff)))
    st.markdown(f"**n (SRS):** {n_srs} → **n (adjusted):** {n_adj} → **Effective Sample Size:** {n_eff}")

    # DEFF sensitivity
    st.subheader("Sensitivitas DEFF")
    rho_range = np.arange(0, 0.51, 0.01)
    m_options = [5, 10, 25, 50, 100]
    fig_deff = go.Figure()
    for m in m_options:
        deffs = [design_effect(r, m) for r in rho_range]
        fig_deff.add_trace(go.Scatter(x=rho_range, y=deffs, mode='lines', name=f'm={m}'))
    fig_deff.update_layout(title="DEFF vs ICC (ρ) untuk berbagai ukuran cluster",
                            xaxis_title="ρ (ICC)", yaxis_title="DEFF", height=450)
    fig_deff.add_hline(y=1, line_dash="dash", line_color="gray")
    st.plotly_chart(fig_deff, use_container_width=True)

# ================================================================
# MODULE 5: KUESIONER & SKALA
# ================================================================
elif "5." in module:
    st.header("5. Perancangan Kuesioner & Skala")

    tab1, tab2, tab3 = st.tabs(["Panduan Skala", "Reliabilitas (Cronbach's α)", "Validitas (Korelasi Item-Total)"])

    with tab1:
        st.subheader("Panduan Pemilihan Skala")
        scale_guide = pd.DataFrame({
            'Skala': ['Likert 5-point', 'Likert 7-point', 'Semantic Differential',
                      'Guttman', 'Thurstone', 'Visual Analog Scale (VAS)', 'Numerik Rating (NRS)'],
            'Tipe': ['Ordinal', 'Ordinal', 'Interval-like', 'Ordinal', 'Interval-like', 'Kontinu', 'Ordinal'],
            'Range': ['1-5', '1-7', '1-7 (bipolar)', '0/1 kumulatif', '1-11', '0-100mm', '0-10'],
            'Kegunaan': ['Paling umum, sikap/persepsi', 'Lebih sensitif, riset akademik',
                         'Evaluasi produk/brand', 'Skalogram', 'Sikap (judge-based)',
                         'Nyeri, kepuasan', 'Nyeri, intensitas']
        })
        st.dataframe(scale_guide, use_container_width=True, hide_index=True)

    with tab2:
        st.subheader("Kalkulator Reliabilitas — Cronbach's Alpha")
        st.markdown("Upload data item kuesioner atau gunakan demo.")

        use_demo_q = st.checkbox("Gunakan data demo kuesioner", value=True, key='demo_q')
        if use_demo_q:
            np.random.seed(42)
            n_resp = 100
            n_items = 8
            base = np.random.normal(3.5, 0.8, n_resp)
            items = np.clip(np.column_stack([base + np.random.normal(0, 0.5, n_resp) for _ in range(n_items)]),
                            1, 5).round(0).astype(int)
            q_df = pd.DataFrame(items, columns=[f'Item_{i+1}' for i in range(n_items)])
        else:
            q_file = st.file_uploader("Upload data kuesioner (CSV)", type=['csv'], key='q_upload')
            if q_file:
                q_df = pd.read_csv(q_file)
            else:
                st.info("Upload file atau gunakan demo.")
                q_df = None

        if q_df is not None:
            st.dataframe(q_df.head(10), use_container_width=True)

            k = q_df.shape[1]
            item_vars = q_df.var(ddof=1)
            total_var = q_df.sum(axis=1).var(ddof=1)
            cronbach = (k / (k - 1)) * (1 - item_vars.sum() / total_var) if total_var > 0 else 0

            st.success(f"### Cronbach's Alpha = **{cronbach:.4f}**")

            if cronbach >= 0.9: interp = "Sangat Baik (Excellent)"
            elif cronbach >= 0.8: interp = "Baik (Good)"
            elif cronbach >= 0.7: interp = "Dapat Diterima (Acceptable)"
            elif cronbach >= 0.6: interp = "Dipertanyakan (Questionable)"
            elif cronbach >= 0.5: interp = "Buruk (Poor)"
            else: interp = "Tidak Dapat Diterima (Unacceptable)"
            st.markdown(f"**Interpretasi:** {interp}")

            # Alpha if item deleted
            st.markdown("**Alpha if Item Deleted:**")
            aid_rows = []
            for col in q_df.columns:
                q_reduced = q_df.drop(columns=[col])
                k_r = q_reduced.shape[1]
                iv_r = q_reduced.var(ddof=1).sum()
                tv_r = q_reduced.sum(axis=1).var(ddof=1)
                a_r = (k_r/(k_r-1))*(1 - iv_r/tv_r) if tv_r > 0 and k_r > 1 else 0
                aid_rows.append({'Item': col, 'α if Deleted': round(a_r, 4),
                                 'Rekomendasi': 'Hapus ↑' if a_r > cronbach + 0.01 else 'Pertahankan'})
            aid_df = pd.DataFrame(aid_rows)
            st.dataframe(aid_df, use_container_width=True, hide_index=True)

            fig = px.bar(aid_df, x='Item', y='α if Deleted', color='Rekomendasi',
                          title="Cronbach's Alpha if Item Deleted")
            fig.add_hline(y=cronbach, line_dash="dash", line_color="red",
                           annotation_text=f"α={cronbach:.4f}")
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Validitas — Korelasi Item-Total")
        if q_df is not None:
            total_score = q_df.sum(axis=1)
            val_rows = []
            for col in q_df.columns:
                corrected_total = total_score - q_df[col]
                r, p = stats.pearsonr(q_df[col], corrected_total)
                val_rows.append({'Item': col, 'Corrected Item-Total r': round(r, 4),
                                 'p-value': round(p, 6),
                                 'Valid': 'Ya (r ≥ 0.3)' if r >= 0.3 else 'Tidak (r < 0.3)'})
            val_df = pd.DataFrame(val_rows)
            st.dataframe(val_df, use_container_width=True, hide_index=True)

            fig = px.bar(val_df, x='Item', y='Corrected Item-Total r', color='Valid',
                          title="Corrected Item-Total Correlation")
            fig.add_hline(y=0.3, line_dash="dash", line_color="red", annotation_text="r = 0.3")
            st.plotly_chart(fig, use_container_width=True)

# ================================================================
# MODULE 6: SIMULASI
# ================================================================
elif "6." in module:
    st.header("6. Simulasi Sampling")
    st.markdown("Demonstrasi visual bagaimana sampling bekerja pada populasi buatan.")

    np.random.seed(42)
    N_sim = st.slider("Ukuran Populasi", 100, 5000, 1000, 100, key='sim_N')
    n_sim = st.slider("Ukuran Sampel", 10, min(1000, N_sim), 100, 10, key='sim_n')

    pop_x = np.random.normal(50, 15, N_sim)
    pop_y = np.random.normal(50, 15, N_sim)
    pop_group = np.random.choice(['A', 'B', 'C', 'D'], N_sim, p=[0.4, 0.3, 0.2, 0.1])

    pop_df = pd.DataFrame({'X': pop_x, 'Y': pop_y, 'Group': pop_group, 'ID': range(1, N_sim+1)})

    sim_method = st.selectbox("Metode Sampling Simulasi", [
        "Simple Random Sampling", "Systematic Sampling",
        "Stratified (Proportional)", "Cluster Sampling"
    ], key='sim_method')

    if sim_method == "Simple Random Sampling":
        sample_idx = np.random.choice(N_sim, n_sim, replace=False)
    elif sim_method == "Systematic Sampling":
        k = max(1, N_sim // n_sim)
        start = np.random.randint(0, k)
        sample_idx = np.arange(start, N_sim, k)[:n_sim]
    elif sim_method == "Stratified (Proportional)":
        sample_idx = []
        for g in pop_df['Group'].unique():
            g_idx = pop_df[pop_df['Group'] == g].index.values
            n_g = max(1, int(n_sim * len(g_idx) / N_sim))
            sample_idx.extend(np.random.choice(g_idx, min(n_g, len(g_idx)), replace=False))
        sample_idx = np.array(sample_idx)
    else:  # Cluster
        clusters = pop_df['Group'].unique()
        n_cl = max(1, len(clusters) // 2)
        sel_cl = np.random.choice(clusters, n_cl, replace=False)
        sample_idx = pop_df[pop_df['Group'].isin(sel_cl)].index.values
        if len(sample_idx) > n_sim:
            sample_idx = np.random.choice(sample_idx, n_sim, replace=False)

    pop_df['Selected'] = 'Populasi'
    pop_df.loc[sample_idx, 'Selected'] = 'Sampel'

    fig = px.scatter(pop_df, x='X', y='Y', color='Selected', symbol='Group',
                      color_discrete_map={'Populasi': 'lightgray', 'Sampel': 'red'},
                      opacity=0.6, title=f"{sim_method}: {len(sample_idx)} dari {N_sim}")
    fig.update_layout(height=550)
    st.plotly_chart(fig, use_container_width=True)

    # Compare distribution
    st.subheader("Perbandingan Distribusi: Populasi vs Sampel")
    fig_comp = make_subplots(rows=1, cols=2, subplot_titles=("Distribusi X", "Distribusi Y"))
    fig_comp.add_trace(go.Histogram(x=pop_df['X'], name='Populasi', opacity=0.5, marker_color='steelblue'), row=1, col=1)
    fig_comp.add_trace(go.Histogram(x=pop_df.loc[sample_idx, 'X'], name='Sampel', opacity=0.7, marker_color='salmon'), row=1, col=1)
    fig_comp.add_trace(go.Histogram(x=pop_df['Y'], name='Populasi', opacity=0.5, marker_color='steelblue', showlegend=False), row=1, col=2)
    fig_comp.add_trace(go.Histogram(x=pop_df.loc[sample_idx, 'Y'], name='Sampel', opacity=0.7, marker_color='salmon', showlegend=False), row=1, col=2)
    fig_comp.update_layout(barmode='overlay', height=400)
    st.plotly_chart(fig_comp, use_container_width=True)

    # Stats comparison
    comp_stats = pd.DataFrame({
        'Statistik': ['Mean X', 'Std X', 'Mean Y', 'Std Y'],
        'Populasi': [pop_df['X'].mean().round(3), pop_df['X'].std().round(3),
                     pop_df['Y'].mean().round(3), pop_df['Y'].std().round(3)],
        'Sampel': [pop_df.loc[sample_idx, 'X'].mean().round(3), pop_df.loc[sample_idx, 'X'].std().round(3),
                   pop_df.loc[sample_idx, 'Y'].mean().round(3), pop_df.loc[sample_idx, 'Y'].std().round(3)]
    })
    comp_stats['Selisih'] = (comp_stats['Sampel'] - comp_stats['Populasi']).round(3)
    st.dataframe(comp_stats, use_container_width=True, hide_index=True)

    # Sampling distribution of mean (CLT demo)
    st.subheader("Distribusi Sampling Mean (CLT Demonstration)")
    n_reps = st.slider("Jumlah Replikasi", 100, 5000, 1000, 100, key='clt_reps')
    sample_means = [np.random.choice(pop_x, n_sim, replace=False).mean() for _ in range(n_reps)]

    fig_clt = go.Figure()
    fig_clt.add_trace(go.Histogram(x=sample_means, nbinsx=40, marker_color='steelblue'))
    fig_clt.add_vline(x=pop_x.mean(), line_color="red", annotation_text=f"μ={pop_x.mean():.2f}")
    fig_clt.update_layout(title=f"Distribusi Sampling Mean X (n={n_sim}, {n_reps} replikasi)",
                           xaxis_title="Sample Mean", height=400)
    st.plotly_chart(fig_clt, use_container_width=True)

    se_theory = pop_x.std() / np.sqrt(n_sim)
    se_empiric = np.std(sample_means)
    st.markdown(f"**SE Teoritis:** {se_theory:.4f} | **SE Empiris:** {se_empiric:.4f}")

# ================================================================
# MODULE 7: REPORT
# ================================================================
elif "7." in module:
    st.header("7. Laporan & Ekspor")

    st.subheader("Template Metodologi Survey")
    st.markdown("Isi parameter di bawah, lalu download laporan otomatis.")

    col1, col2 = st.columns(2)
    with col1:
        research_title = st.text_input("Judul Penelitian", "Survei Kepuasan Pelanggan 2026")
        population_desc = st.text_area("Deskripsi Populasi", "Seluruh pelanggan aktif perusahaan XYZ.")
        N_report = st.number_input("Ukuran Populasi (N)", 100, 10_000_000, 5000, key='rep_N')
        conf_report = st.selectbox("Confidence Level", [90, 95, 99], index=1, key='rep_cl')
    with col2:
        moe_report = st.slider("Margin of Error", 0.01, 0.20, 0.05, 0.01, key='rep_moe')
        p_report = st.slider("Estimasi Proporsi (p)", 0.01, 0.99, 0.50, 0.01, key='rep_p')
        sampling_tech = st.selectbox("Teknik Sampling", [
            "Simple Random Sampling", "Systematic Sampling",
            "Stratified Proportional", "Cluster Sampling", "Multistage"
        ], key='rep_tech')
        data_collection = st.selectbox("Metode Pengumpulan Data", [
            "Online Questionnaire", "Face-to-Face Interview",
            "Telephone Survey", "Mail Survey", "Mixed Mode"
        ])

    z_rep = Z_MAP[conf_report]
    n0_rep = cochran_infinite(z_rep, p_report, moe_report)
    n_rep = int(np.ceil(cochran_finite(n0_rep, N_report)))

    st.success(f"### Ukuran Sampel: **n = {n_rep}**")

    nonresp_rate = st.slider("Antisipasi Non-Response Rate (%)", 0, 50, 20, 5, key='nr_rate')
    n_adjusted = int(np.ceil(n_rep / (1 - nonresp_rate / 100)))
    st.info(f"**Sampel setelah koreksi non-response ({nonresp_rate}%):** n' = {n_adjusted}")

    # Generate report
    report = f"""{'='*70}
LAPORAN METODOLOGI PERANCANGAN SURVEY
{'='*70}

JUDUL PENELITIAN:
{research_title}

POPULASI:
{population_desc}
Ukuran Populasi (N): {N_report:,}

PARAMETER SAMPLING:
- Confidence Level     : {conf_report}% (Z = {z_rep})
- Margin of Error      : ±{moe_report*100:.1f}%
- Estimasi Proporsi    : p = {p_report}
- Teknik Sampling      : {sampling_tech}
- Pengumpulan Data     : {data_collection}

PERHITUNGAN UKURAN SAMPEL:
Metode: Cochran (1977) dengan Finite Population Correction

    n₀ = Z² × p(1-p) / e²
    n₀ = {z_rep}² × {p_report} × {1-p_report:.2f} / {moe_report}²
    n₀ = {int(np.ceil(n0_rep))}

    n  = n₀ / (1 + (n₀-1)/N)
    n  = {int(np.ceil(n0_rep))} / (1 + ({int(np.ceil(n0_rep))-1}/{N_report:,}))
    n  = {n_rep}

KOREKSI NON-RESPONSE:
- Non-response rate    : {nonresp_rate}%
- Sampel akhir         : n' = {n_rep} / (1 - {nonresp_rate/100}) = {n_adjusted}
- Sampling fraction    : {n_adjusted/N_report*100:.2f}%

RINGKASAN:
- Minimum sampel yang harus diperoleh: {n_adjusted} responden
- Dari populasi: {N_report:,}
- Dengan presisi: ±{moe_report*100:.1f}% pada {conf_report}% confidence level

{'='*70}
Digenerate oleh: Survey Design Toolkit
{'='*70}
"""

    st.text_area("Preview Laporan", report, height=400)

    col_e1, col_e2 = st.columns(2)
    with col_e1:
        st.download_button("📥 Download Laporan (TXT)", data=report,
                           file_name="survey_methodology_report.txt", mime="text/plain")
    with col_e2:
        summary_csv = pd.DataFrame({
            'Parameter': ['Judul', 'Populasi (N)', 'Confidence Level', 'Margin of Error',
                          'Proporsi (p)', 'Teknik Sampling', 'n (Cochran)', 'n (non-response adj)',
                          'Sampling Fraction'],
            'Nilai': [research_title, N_report, f'{conf_report}%', f'{moe_report*100:.1f}%',
                      p_report, sampling_tech, n_rep, n_adjusted, f'{n_adjusted/N_report*100:.2f}%']
        })
        st.download_button("📥 Download Parameter (CSV)", data=summary_csv.to_csv(index=False),
                           file_name="survey_parameters.csv", mime="text/csv")

# ============================
# FOOTER
# ============================
st.markdown("---")
st.markdown("""
**Referensi Metodologis:**
- **Cochran (1977):** Sampling Techniques, 3rd Ed. Wiley.
- **Slovin (1960):** Formula untuk populasi terhingga tanpa informasi variansi.
- **Yamane (1967):** Statistics: An Introductory Analysis, 2nd Ed.
- **Kish (1965):** Survey Sampling. Design Effect = 1 + ρ(m-1).
- **Cronbach (1951):** Coefficient Alpha and the Internal Structure of Tests.

Dibangun dengan **Streamlit** + **SciPy** + **Plotly** | Python
""")
