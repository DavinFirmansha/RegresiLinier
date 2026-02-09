#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
  APLIKASI SINGULAR SPECTRUM ANALYSIS (SSA) LENGKAP
  Untuk Analisis Time Series 1 Dimensi
  ──────────────────────────────────────────────────
  Cocok untuk Skripsi & Artikel Ilmiah

  Fitur Utama:
  ─────────────
  1.  Input data CSV/Excel fleksibel
  2.  Pemilihan Window Length (L) otomatis & manual
  3.  Embedding (Trajectory Matrix Construction)
  4.  SVD Decomposition + Scree Plot
  5.  W-Correlation Matrix (Weighted Correlation)
  6.  Grouping komponen (Manual/Auto)
  7.  Rekonstruksi komponen (Trend, Seasonal, Noise)
  8.  Paired Correlation (Periodogram per komponen)
  9.  Recurrent Forecasting (R-forecasting)
  10. Vector Forecasting (V-forecasting)
  11. Split Train-Test dengan evaluasi (RMSE, MAE, MAPE, R², dll)
  12. Evaluasi: Training, Testing, dan Overall
  13. Residual Analysis (ACF, Normality Test, Ljung-Box)
  14. Monte Carlo SSA Significance Test
  15. Visualisasi lengkap dan siap publikasi

  Author : SSA Analysis Tool
  Version: 3.0 (2026)
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
from scipy.linalg import svd, hankel
from scipy.stats import shapiro, jarque_bera, norm
from scipy.signal import periodogram
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
import os
import sys
import traceback
from itertools import combinations

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════════
# KONFIGURASI VISUAL UNTUK PUBLIKASI ILMIAH
# ══════════════════════════════════════════════════════════════════════════════
plt.rcParams.update({
    'figure.figsize': (14, 6),
    'figure.dpi': 150,
    'font.size': 11,
    'font.family': 'serif',
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 1.5,
    'figure.autolayout': True,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


# ══════════════════════════════════════════════════════════════════════════════
# CLASS UTAMA: SSA (Singular Spectrum Analysis)
# ══════════════════════════════════════════════════════════════════════════════
class SSA:
    """
    Kelas utama untuk Singular Spectrum Analysis (SSA).

    Tahapan SSA:
    ────────────
    TAHAP 1 - DEKOMPOSISI:
      Step 1: Embedding     → Membentuk Trajectory Matrix (Hankel Matrix)
      Step 2: SVD           → Singular Value Decomposition

    TAHAP 2 - REKONSTRUKSI:
      Step 3: Grouping      → Pengelompokan komponen eigen-triple
      Step 4: Diagonal Averaging → Rekonstruksi time series dari grup

    TAHAP 3 - FORECASTING:
      - Recurrent Forecasting (R-forecasting / LRF)
      - Vector Forecasting (V-forecasting)

    Parameters
    ----------
    time_series : array-like
        Data time series 1 dimensi
    window_length : int or str
        Panjang window (L). Bisa integer atau 'auto'.
        Jika 'auto', akan dipilih L = N//2.
    name : str
        Nama series (untuk label plot)
    """

    def __init__(self, time_series, window_length='auto', name='Time Series'):
        self.original = np.array(time_series, dtype=float).flatten()
        self.N = len(self.original)
        self.name = name

        # Tentukan Window Length
        if window_length == 'auto':
            self.L = self.N // 2
            print(f"[INFO] Window Length otomatis: L = N/2 = {self.L}")
        else:
            self.L = int(window_length)

        self.K = self.N - self.L + 1  # Jumlah kolom trajectory matrix

        # Validasi
        assert 2 <= self.L <= self.N // 2 + 1, \
            f"Window Length L harus antara 2 dan N/2+1. Diberikan L={self.L}, N={self.N}"

        print(f"{'='*70}")
        print(f"  SINGULAR SPECTRUM ANALYSIS (SSA)")
        print(f"{'='*70}")
        print(f"  Nama Series    : {self.name}")
        print(f"  Panjang Series : N = {self.N}")
        print(f"  Window Length  : L = {self.L}")
        print(f"  Trajectory Dim : K = {self.K}")
        print(f"  Matrix Size    : {self.L} × {self.K}")
        print(f"{'='*70}\n")

        # Jalankan dekomposisi
        self._embed()
        self._decompose()

    # ══════════════════════════════════════════════════════════════════════
    # STEP 1: EMBEDDING (Pembentukan Trajectory Matrix)
    # ══════════════════════════════════════════════════════════════════════
    def _embed(self):
        """
        Membentuk Trajectory Matrix X dari time series.
        X adalah Hankel Matrix berukuran L × K.

        X = [X1, X2, ..., XK] dimana Xi = (y_i, y_{i+1}, ..., y_{i+L-1})^T
        """
        self.trajectory_matrix = np.column_stack(
            [self.original[i:i+self.L] for i in range(self.K)]
        )
        print("[✓] Step 1: Embedding (Trajectory Matrix) selesai.")
        print(f"    Trajectory Matrix shape: {self.trajectory_matrix.shape}\n")

    # ══════════════════════════════════════════════════════════════════════
    # STEP 2: SVD (Singular Value Decomposition)
    # ══════════════════════════════════════════════════════════════════════
    def _decompose(self):
        """
        Melakukan SVD pada Trajectory Matrix X.
        X = U * diag(sigma) * V^T

        Elementary matrix: Xi = sigma_i * u_i * v_i^T
        """
        S = self.trajectory_matrix @ self.trajectory_matrix.T

        # Eigen decomposition (lebih stabil untuk matriks simetris)
        eigenvalues, eigenvectors = np.linalg.eigh(S)

        # Urutkan dari terbesar
        idx = np.argsort(eigenvalues)[::-1]
        self.eigenvalues = eigenvalues[idx]
        self.eigenvalues = np.maximum(self.eigenvalues, 0)  # Pastikan non-negatif
        self.singular_values = np.sqrt(self.eigenvalues)
        self.U = eigenvectors[:, idx]

        # Hitung V dan elementary matrices
        self.d = np.sum(self.singular_values > 1e-10)  # Rank efektif
        self.V = np.zeros((self.K, self.d))
        self.elementary_matrices = []

        for i in range(self.d):
            vi = self.trajectory_matrix.T @ self.U[:, i] / self.singular_values[i]
            self.V[:, i] = vi
            Xi = self.singular_values[i] * np.outer(self.U[:, i], vi)
            self.elementary_matrices.append(Xi)

        # Kontribusi varians
        total_var = np.sum(self.eigenvalues[:self.d])
        self.contribution = self.eigenvalues[:self.d] / total_var * 100
        self.cumulative_contribution = np.cumsum(self.contribution)

        print(f"[✓] Step 2: SVD Decomposition selesai.")
        print(f"    Rank efektif trajectory matrix: d = {self.d}")
        print(f"    Top 10 Singular Values & Kontribusi:")
        print(f"    {'No':>4} {'Singular Value':>16} {'Share (%)':>12} {'Cumul (%)':>12}")
        print(f"    {'─'*48}")
        for i in range(min(10, self.d)):
            print(f"    {i+1:>4} {self.singular_values[i]:>16.4f} "
                  f"{self.contribution[i]:>11.4f}% {self.cumulative_contribution[i]:>11.4f}%")
        print()

    # ══════════════════════════════════════════════════════════════════════
    # W-CORRELATION MATRIX
    # ══════════════════════════════════════════════════════════════════════
    def w_correlation(self, num_components=None):
        """
        Menghitung W-Correlation Matrix antar komponen.

        W-correlation mengukur keterkaitan antar komponen SSA.
        - Nilai mendekati 0: komponen well-separated (bisa dipisahkan)
        - Nilai mendekati 1: komponen terkait erat (harus digrup bersama)

        Parameters
        ----------
        num_components : int
            Jumlah komponen yang dihitung. Default = min(d, 20)
        """
        if num_components is None:
            num_components = min(self.d, 20)

        # Rekonstruksi tiap komponen
        components = []
        for i in range(num_components):
            rc = self._diagonal_averaging(self.elementary_matrices[i])
            components.append(rc)

        # Hitung weights
        weights = np.zeros(self.N)
        Ls = min(self.L, self.K)
        Ks = max(self.L, self.K)
        for i in range(self.N):
            if i < Ls - 1:
                weights[i] = i + 1
            elif i < Ks:
                weights[i] = Ls
            else:
                weights[i] = self.N - i

        # W-correlation matrix
        self.wcorr_matrix = np.zeros((num_components, num_components))
        for i in range(num_components):
            for j in range(num_components):
                wi_norm = np.sqrt(np.sum(weights * components[i]**2))
                wj_norm = np.sqrt(np.sum(weights * components[j]**2))
                if wi_norm > 0 and wj_norm > 0:
                    self.wcorr_matrix[i, j] = (
                        np.sum(weights * components[i] * components[j]) / (wi_norm * wj_norm)
                    )

        print(f"[✓] W-Correlation Matrix ({num_components}×{num_components}) dihitung.\n")
        return self.wcorr_matrix

    # ══════════════════════════════════════════════════════════════════════
    # DIAGONAL AVERAGING (Anti-diagonal Averaging / Hankelization)
    # ══════════════════════════════════════════════════════════════════════
    def _diagonal_averaging(self, matrix):
        """
        Mengubah elementary matrix kembali menjadi time series
        melalui diagonal averaging.
        """
        L, K = matrix.shape
        N = L + K - 1
        result = np.zeros(N)
        counts = np.zeros(N)

        for i in range(L):
            for j in range(K):
                result[i + j] += matrix[i, j]
                counts[i + j] += 1

        return result / counts

    # ══════════════════════════════════════════════════════════════════════
    # STEP 3 & 4: GROUPING & RECONSTRUCTION
    # ══════════════════════════════════════════════════════════════════════
    def reconstruct(self, groups):
        """
        Rekonstruksi time series dari grup komponen.

        Parameters
        ----------
        groups : dict
            Dictionary dengan key = nama grup, value = list indeks komponen.
            Contoh: {'Trend': [0,1], 'Seasonal': [2,3,4,5], 'Noise': [6,7,8,9]}

        Returns
        -------
        dict : Dictionary berisi rekonstruksi tiap grup.
        """
        self.groups = groups
        self.reconstructed = {}

        print("[✓] Step 3 & 4: Grouping & Reconstruction")
        print(f"    Grup yang didefinisikan:")

        for name, indices in groups.items():
            group_matrix = sum(self.elementary_matrices[i] for i in indices)
            self.reconstructed[name] = self._diagonal_averaging(group_matrix)
            var_share = sum(self.contribution[i] for i in indices if i < len(self.contribution))
            print(f"    • {name:20s}: komponen {indices} "
                  f"(kontribusi: {var_share:.2f}%)")

        # Total rekonstruksi
        all_indices = [i for indices in groups.values() for i in indices]
        total_matrix = sum(self.elementary_matrices[i] for i in all_indices)
        self.reconstructed['_Total'] = self._diagonal_averaging(total_matrix)

        # Residual
        self.reconstructed['_Residual'] = self.original - self.reconstructed['_Total']

        rmse_total = np.sqrt(np.mean(self.reconstructed['_Residual']**2))
        print(f"\n    RMSE rekonstruksi total: {rmse_total:.6f}")
        print()

        return self.reconstructed

    # ══════════════════════════════════════════════════════════════════════
    # AUTO GROUPING
    # ══════════════════════════════════════════════════════════════════════
    def auto_group(self, num_components=None, threshold=0.5):
        """
        Grouping otomatis berdasarkan W-Correlation.
        Komponen dengan w-correlation > threshold akan digrup bersama.

        Parameters
        ----------
        num_components : int
            Jumlah komponen yang dipertimbangkan
        threshold : float
            Threshold korelasi untuk menggabungkan komponen (default: 0.5)

        Returns
        -------
        dict : Suggested grouping
        """
        if num_components is None:
            # Pilih komponen yang kontribusinya > 0.5%
            num_components = max(2, np.sum(self.contribution > 0.5))
            num_components = min(num_components, self.d, 20)

        wcorr = self.w_correlation(num_components)

        # Spectral-based grouping
        groups_list = []
        assigned = set()

        for i in range(num_components):
            if i in assigned:
                continue
            group = [i]
            assigned.add(i)
            for j in range(i + 1, num_components):
                if j not in assigned and abs(wcorr[i, j]) > threshold:
                    group.append(j)
                    assigned.add(j)
            groups_list.append(group)

        # Labeling berdasarkan periodogram
        groups = {}
        trend_found = False
        seasonal_count = 0

        for idx, grp in enumerate(groups_list):
            # Analisis periodogram dari komponen pertama di grup
            rc = self._diagonal_averaging(self.elementary_matrices[grp[0]])
            freqs, psd = periodogram(rc, fs=1.0)

            if len(freqs) > 1:
                dominant_freq = freqs[np.argmax(psd[1:]) + 1] if len(psd) > 1 else 0
            else:
                dominant_freq = 0

            if dominant_freq < 0.02 and not trend_found:
                groups[f'Trend'] = grp
                trend_found = True
            elif dominant_freq < 0.02:
                groups[f'Trend_{idx}'] = grp
            else:
                seasonal_count += 1
                period = 1.0 / dominant_freq if dominant_freq > 0 else float('inf')
                groups[f'Seasonal_{seasonal_count} (T≈{period:.1f})'] = grp

        # Sisanya jadi noise
        remaining = [i for i in range(num_components) if i not in assigned]
        noise_all = list(range(num_components, min(self.d, self.L)))
        if remaining:
            noise_all = remaining + noise_all
        if noise_all:
            groups['Noise'] = noise_all

        print(f"[✓] Auto Grouping (threshold={threshold}):")
        for name, indices in groups.items():
            var_share = sum(self.contribution[i] for i in indices if i < len(self.contribution))
            print(f"    • {name:30s}: {indices[:10]}{'...' if len(indices)>10 else ''} "
                  f"({var_share:.2f}%)")
        print()

        return groups

    # ══════════════════════════════════════════════════════════════════════
    # RECURRENT FORECASTING (R-Forecasting / LRF)
    # ══════════════════════════════════════════════════════════════════════
    def forecast_recurrent(self, groups, steps=10, use_indices=None):
        """
        Recurrent SSA Forecasting menggunakan Linear Recurrence Formula (LRF).

        Formula: y_{N+1} = sum_{j=1}^{L-1} a_j * y_{N+1-j}

        dimana koefisien a diperoleh dari eigenvectors terpilih.

        Parameters
        ----------
        groups : dict or list
            Jika dict: menggunakan semua indeks dari semua grup
            Jika list: langsung list indeks komponen
        steps : int
            Jumlah langkah forecasting ke depan
        use_indices : list, optional
            Override: gunakan indeks komponen spesifik

        Returns
        -------
        np.array : Forecast values (panjang = N + steps)
        """
        if use_indices is not None:
            indices = use_indices
        elif isinstance(groups, dict):
            indices = sorted(set(i for v in groups.values() for i in v))
        else:
            indices = sorted(groups)

        # Rekonstruksi signal dari komponen terpilih
        signal_matrix = sum(self.elementary_matrices[i] for i in indices if i < self.d)
        signal = self._diagonal_averaging(signal_matrix)

        # Hitung LRF coefficients
        r = len(indices)
        U_selected = self.U[:, indices]

        # Pisahkan baris terakhir
        pi = U_selected[-1, :]  # Baris ke-L (terakhir)

        nu_squared = np.sum(pi**2)

        if nu_squared >= 1:
            print("[!] WARNING: ν² ≥ 1. R-forecasting mungkin tidak stabil.")
            print(f"    ν² = {nu_squared:.6f}")
            nu_squared = 0.9999  # Clamp

        # Koefisien LRF
        # R = (1/(1-ν²)) * Σ πi * u_i(tanpa baris terakhir)
        R = np.sum(
            pi[np.newaxis, :] * U_selected[:-1, :],
            axis=1
        ) / (1 - nu_squared)

        self.lrf_coefficients = R

        # Forecasting
        y = np.concatenate([signal, np.zeros(steps)])

        for t in range(self.N, self.N + steps):
            y[t] = np.dot(R, y[t-self.L+1:t][::-1][:self.L-1])

        self.forecast_r = y
        self.forecast_r_steps = steps
        self.forecast_indices = indices

        print(f"[✓] Recurrent Forecasting (R-forecasting) selesai.")
        print(f"    Komponen digunakan: {indices[:15]}{'...' if len(indices)>15 else ''}")
        print(f"    ν² = {np.sum(self.U[-1, indices]**2):.6f}")
        print(f"    Jumlah langkah forecast: {steps}")
        print(f"    Forecast values: {y[self.N:self.N+min(5,steps)]}{'...' if steps>5 else ''}\n")

        return y

    # ══════════════════════════════════════════════════════════════════════
    # VECTOR FORECASTING (V-Forecasting)
    # ══════════════════════════════════════════════════════════════════════
    def forecast_vector(self, groups, steps=10, use_indices=None):
        """
        Vector SSA Forecasting (V-forecasting).

        Berbeda dari R-forecasting, V-forecasting melanjutkan subspace
        dari eigenvectors terpilih dan kemudian melakukan diagonal averaging.

        Parameters
        ----------
        groups : dict or list
            Grouping komponen
        steps : int
            Jumlah langkah forecast
        use_indices : list, optional
            Override indeks komponen

        Returns
        -------
        np.array : Forecast values (panjang = N + steps)
        """
        if use_indices is not None:
            indices = use_indices
        elif isinstance(groups, dict):
            indices = sorted(set(i for v in groups.values() for i in v))
        else:
            indices = sorted(groups)

        U_selected = self.U[:, indices]

        pi = U_selected[-1, :]
        nu_squared = np.sum(pi**2)

        if nu_squared >= 1:
            print("[!] WARNING: ν² ≥ 1. V-forecasting mungkin tidak stabil.")
            nu_squared = 0.9999

        # Proyektor
        U_del = U_selected[:-1, :]  # L-1 baris pertama

        P_pi = U_del @ pi / (1 - nu_squared)  # vektor (L-1)

        # Extended trajectory matrix
        signal_matrix = sum(self.elementary_matrices[i] for i in indices if i < self.d)
        signal = self._diagonal_averaging(signal_matrix)

        # Buat extended matrix
        Q = np.column_stack([signal[i:i+self.L] for i in range(self.K)])

        # Forecast step by step
        for h in range(steps):
            last_vector = Q[:, -1]  # Kolom terakhir
            new_last = last_vector[1:]  # Geser ke atas
            new_element = np.dot(P_pi, new_last)
            new_col = np.append(new_last, new_element)
            Q = np.column_stack([Q, new_col])

        # Diagonal averaging pada extended matrix
        L_ext, K_ext = Q.shape
        N_ext = L_ext + K_ext - 1
        result = np.zeros(N_ext)
        counts = np.zeros(N_ext)
        for i in range(L_ext):
            for j in range(K_ext):
                result[i+j] += Q[i, j]
                counts[i+j] += 1
        y = result / counts

        # Trim ke panjang yang benar
        y = y[:self.N + steps]

        self.forecast_v = y
        self.forecast_v_steps = steps

        print(f"[✓] Vector Forecasting (V-forecasting) selesai.")
        print(f"    Komponen digunakan: {indices[:15]}{'...' if len(indices)>15 else ''}")
        print(f"    Jumlah langkah forecast: {steps}")
        print(f"    Forecast values: {y[self.N:self.N+min(5,steps)]}{'...' if steps>5 else ''}\n")

        return y

    # ══════════════════════════════════════════════════════════════════════
    # MONTE CARLO SSA SIGNIFICANCE TEST
    # ══════════════════════════════════════════════════════════════════════
    def monte_carlo_test(self, num_surrogates=1000, confidence=0.95):
        """
        Monte Carlo SSA test untuk menentukan signifikansi singular values.

        Membandingkan eigenvalues dari data asli dengan eigenvalues 
        dari surrogate red-noise data.

        H0: Komponen ke-i berasal dari red noise
        H1: Komponen ke-i signifikan (bukan noise)

        Parameters
        ----------
        num_surrogates : int
            Jumlah surrogate data (default: 1000)
        confidence : float
            Tingkat kepercayaan (default: 0.95)

        Returns
        -------
        dict : Hasil test untuk setiap komponen
        """
        print(f"[⏳] Monte Carlo SSA Test ({num_surrogates} surrogates)...")

        # Estimasi parameter AR(1) dari data asli
        ts = self.original - np.mean(self.original)
        lag1_corr = np.corrcoef(ts[:-1], ts[1:])[0, 1]
        noise_var = np.var(ts) * (1 - lag1_corr**2)

        # Generate surrogate eigenvalues
        surrogate_eigenvalues = np.zeros((num_surrogates, min(self.L, self.K)))

        for s in range(num_surrogates):
            # Generate AR(1) surrogate
            surr = np.zeros(self.N)
            surr[0] = np.random.normal(0, np.sqrt(np.var(ts)))
            for t in range(1, self.N):
                surr[t] = lag1_corr * surr[t-1] + np.random.normal(0, np.sqrt(noise_var))

            # SSA of surrogate
            traj = np.column_stack([surr[i:i+self.L] for i in range(self.N - self.L + 1)])
            S_surr = traj @ traj.T
            eig_surr = np.linalg.eigvalsh(S_surr)[::-1]
            n_eig = min(len(eig_surr), surrogate_eigenvalues.shape[1])
            surrogate_eigenvalues[s, :n_eig] = eig_surr[:n_eig]

        # Confidence intervals
        lower = np.percentile(surrogate_eigenvalues, (1-confidence)/2*100, axis=0)
        upper = np.percentile(surrogate_eigenvalues, (1+confidence)/2*100, axis=0)
        median_surr = np.median(surrogate_eigenvalues, axis=0)

        # Test
        n_test = min(20, self.d)
        self.mc_results = {
            'eigenvalues': self.eigenvalues[:n_test],
            'surrogate_lower': lower[:n_test],
            'surrogate_upper': upper[:n_test],
            'surrogate_median': median_surr[:n_test],
            'significant': self.eigenvalues[:n_test] > upper[:n_test],
            'confidence': confidence
        }

        sig_count = np.sum(self.mc_results['significant'])
        print(f"[✓] Monte Carlo SSA Test selesai.")
        print(f"    Signifikansi level: {confidence*100:.0f}%")
        print(f"    Komponen signifikan: {sig_count} dari {n_test} yang diuji")
        sig_idx = np.where(self.mc_results['significant'])[0]
        print(f"    Indeks signifikan: {list(sig_idx)}\n")

        return self.mc_results

    # ══════════════════════════════════════════════════════════════════════
    # EVALUASI: TRAIN, TEST, OVERALL
    # ══════════════════════════════════════════════════════════════════════
    def evaluate(self, actual, predicted, label=""):
        """
        Menghitung metrik evaluasi komprehensif.

        Metrics:
        - RMSE  : Root Mean Square Error
        - MAE   : Mean Absolute Error
        - MAPE  : Mean Absolute Percentage Error
        - sMAPE : Symmetric MAPE
        - R²    : Coefficient of Determination
        - NRMSE : Normalized RMSE
        - MaxAE : Maximum Absolute Error
        - MedAE : Median Absolute Error
        """
        actual = np.array(actual)
        predicted = np.array(predicted)

        n = len(actual)
        errors = actual - predicted
        abs_errors = np.abs(errors)

        rmse = np.sqrt(np.mean(errors**2))
        mae = np.mean(abs_errors)

        # MAPE (hindari pembagian nol)
        nonzero = np.abs(actual) > 1e-10
        if np.any(nonzero):
            mape = np.mean(abs_errors[nonzero] / np.abs(actual[nonzero])) * 100
        else:
            mape = float('inf')

        # sMAPE
        denom = (np.abs(actual) + np.abs(predicted))
        nonzero_s = denom > 1e-10
        if np.any(nonzero_s):
            smape = np.mean(2 * abs_errors[nonzero_s] / denom[nonzero_s]) * 100
        else:
            smape = float('inf')

        # R²
        ss_res = np.sum(errors**2)
        ss_tot = np.sum((actual - np.mean(actual))**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')

        # NRMSE
        data_range = np.max(actual) - np.min(actual)
        nrmse = rmse / data_range if data_range > 0 else float('inf')

        max_ae = np.max(abs_errors)
        med_ae = np.median(abs_errors)

        results = {
            'N': n,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE (%)': mape,
            'sMAPE (%)': smape,
            'R²': r2,
            'NRMSE': nrmse,
            'MaxAE': max_ae,
            'MedAE': med_ae
        }

        if label:
            print(f"\n  ═══ Evaluasi: {label} ═══")
        print(f"  {'Metrik':<12} {'Nilai':>15}")
        print(f"  {'─'*30}")
        for k, v in results.items():
            if isinstance(v, int):
                print(f"  {k:<12} {v:>15d}")
            else:
                print(f"  {k:<12} {v:>15.6f}")
        print()

        return results

    def evaluate_split(self, train_size, forecast_values=None, method='recurrent'):
        """
        Evaluasi dengan split training-testing.

        Parameters
        ----------
        train_size : int or float
            Jika int: jumlah data training
            Jika float (0-1): proporsi data training
        forecast_values : np.array, optional
            Hasil forecast (jika sudah ada)
        method : str
            'recurrent' atau 'vector'

        Returns
        -------
        dict : Evaluasi training, testing, dan overall
        """
        if isinstance(train_size, float) and 0 < train_size < 1:
            train_n = int(self.N * train_size)
        else:
            train_n = int(train_size)

        test_n = self.N - train_n

        if forecast_values is None:
            if method == 'recurrent' and hasattr(self, 'forecast_r'):
                forecast_values = self.forecast_r
            elif method == 'vector' and hasattr(self, 'forecast_v'):
                forecast_values = self.forecast_v
            else:
                raise ValueError("Belum ada forecast. Jalankan forecast terlebih dahulu.")

        actual_train = self.original[:train_n]
        actual_test = self.original[train_n:]
        pred_train = forecast_values[:train_n]
        pred_test = forecast_values[train_n:self.N]

        print(f"{'═'*60}")
        print(f"  EVALUASI SPLIT: Train={train_n}, Test={test_n}")
        print(f"{'═'*60}")

        eval_train = self.evaluate(actual_train, pred_train, "TRAINING")
        eval_test = self.evaluate(actual_test, pred_test, "TESTING")
        eval_overall = self.evaluate(self.original, forecast_values[:self.N], "OVERALL")

        self.eval_results = {
            'train': eval_train,
            'test': eval_test,
            'overall': eval_overall,
            'train_size': train_n,
            'test_size': test_n
        }

        return self.eval_results

    # ══════════════════════════════════════════════════════════════════════
    # RESIDUAL ANALYSIS
    # ══════════════════════════════════════════════════════════════════════
    def residual_analysis(self, residuals=None):
        """
        Analisis residual lengkap:
        1. Statistik deskriptif residual
        2. Shapiro-Wilk test (normalitas)
        3. Jarque-Bera test (normalitas)
        4. Ljung-Box test (autokorelasi)
        """
        if residuals is None:
            if hasattr(self, 'reconstructed') and '_Residual' in self.reconstructed:
                residuals = self.reconstructed['_Residual']
            else:
                print("[!] Residual belum tersedia. Jalankan reconstruct() terlebih dahulu.")
                return

        residuals = np.array(residuals)

        print(f"\n{'═'*60}")
        print(f"  ANALISIS RESIDUAL")
        print(f"{'═'*60}")
        print(f"  Statistik Deskriptif:")
        print(f"    Mean     : {np.mean(residuals):.6f}")
        print(f"    Std Dev  : {np.std(residuals):.6f}")
        print(f"    Min      : {np.min(residuals):.6f}")
        print(f"    Max      : {np.max(residuals):.6f}")
        print(f"    Skewness : {pd.Series(residuals).skew():.6f}")
        print(f"    Kurtosis : {pd.Series(residuals).kurtosis():.6f}")

        # Shapiro-Wilk
        if len(residuals) <= 5000:
            stat_sw, p_sw = shapiro(residuals)
            print(f"\n  Shapiro-Wilk Test:")
            print(f"    Statistic: {stat_sw:.6f}")
            print(f"    p-value  : {p_sw:.6f}")
            print(f"    Normal?  : {'Ya (p > 0.05)' if p_sw > 0.05 else 'Tidak (p ≤ 0.05)'}")

        # Jarque-Bera
        stat_jb, p_jb = jarque_bera(residuals)
        print(f"\n  Jarque-Bera Test:")
        print(f"    Statistic: {stat_jb:.6f}")
        print(f"    p-value  : {p_jb:.6f}")
        print(f"    Normal?  : {'Ya (p > 0.05)' if p_jb > 0.05 else 'Tidak (p ≤ 0.05)'}")

        # Ljung-Box
        try:
            n_lags = min(20, len(residuals) // 5)
            if n_lags >= 1:
                lb = acorr_ljungbox(residuals, lags=n_lags, return_df=True)
                print(f"\n  Ljung-Box Test (lag={n_lags}):")
                print(f"    Statistic: {lb['lb_stat'].iloc[-1]:.6f}")
                print(f"    p-value  : {lb['lb_pvalue'].iloc[-1]:.6f}")
                sig = 'Ya (p > 0.05)' if lb['lb_pvalue'].iloc[-1] > 0.05 else 'Tidak (p ≤ 0.05)'
                print(f"    White Noise? : {sig}")
        except Exception as e:
            print(f"    Ljung-Box: Error - {e}")

        print()
        self.residuals = residuals
        return residuals

    # ══════════════════════════════════════════════════════════════════════
    #  VISUALISASI LENGKAP
    # ══════════════════════════════════════════════════════════════════════

    def plot_original(self, time_index=None, save=False):
        """Plot data asli."""
        fig, ax = plt.subplots(figsize=(14, 5))
        x = time_index if time_index is not None else np.arange(self.N)
        ax.plot(x, self.original, 'b-', linewidth=1.2, label=self.name)
        ax.set_title(f'Data Original: {self.name}', fontweight='bold')
        ax.set_xlabel('Waktu')
        ax.set_ylabel('Nilai')
        ax.legend()
        if save: plt.savefig('01_original_series.png')
        plt.show()

    def plot_trajectory_matrix(self, save=False):
        """Visualisasi Trajectory Matrix sebagai heatmap."""
        fig, ax = plt.subplots(figsize=(12, 8))
        im = ax.imshow(self.trajectory_matrix, aspect='auto', cmap='RdBu_r',
                       interpolation='nearest')
        ax.set_title('Trajectory Matrix (Hankel Matrix)', fontweight='bold')
        ax.set_xlabel(f'Kolom (K = {self.K})')
        ax.set_ylabel(f'Baris (L = {self.L})')
        plt.colorbar(im, ax=ax, shrink=0.8)
        if save: plt.savefig('02_trajectory_matrix.png')
        plt.show()

    def plot_scree(self, num_components=None, save=False):
        """
        Scree Plot: Eigenvalues dan kontribusi kumulatif.
        Membantu menentukan jumlah komponen signifikan.
        """
        if num_components is None:
            num_components = min(20, self.d)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        x = np.arange(1, num_components + 1)

        # Eigenvalue / Singular value plot
        ax1.bar(x, self.singular_values[:num_components], color='steelblue', alpha=0.7,
                edgecolor='navy', label='Singular Values')
        ax1.plot(x, self.singular_values[:num_components], 'ro-', markersize=5)
        ax1.set_title('Scree Plot (Singular Values)', fontweight='bold')
        ax1.set_xlabel('Komponen')
        ax1.set_ylabel('Singular Value (σ)')
        ax1.set_xticks(x)

        # Log scale eigenvalue
        ax1_twin = ax1.twinx()
        ax1_twin.plot(x, self.eigenvalues[:num_components], 's--', color='green',
                      markersize=4, alpha=0.7, label='Eigenvalues (λ)')
        ax1_twin.set_ylabel('Eigenvalue (λ)', color='green')
        ax1_twin.tick_params(axis='y', labelcolor='green')

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_twin.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

        # Kontribusi kumulatif
        ax2.bar(x, self.contribution[:num_components], color='coral', alpha=0.7,
                edgecolor='darkred', label='Individual (%)')
        ax2.plot(x, self.cumulative_contribution[:num_components], 'ko-',
                 markersize=5, label='Cumulative (%)')
        ax2.axhline(y=95, color='r', linestyle='--', alpha=0.5, label='95%')
        ax2.axhline(y=99, color='g', linestyle='--', alpha=0.5, label='99%')
        ax2.set_title('Kontribusi Varians', fontweight='bold')
        ax2.set_xlabel('Komponen')
        ax2.set_ylabel('Kontribusi (%)')
        ax2.set_xticks(x)
        ax2.legend()

        plt.suptitle(f'SSA Decomposition — {self.name}', fontsize=14, fontweight='bold', y=1.02)
        if save: plt.savefig('03_scree_plot.png')
        plt.show()

    def plot_eigenvectors(self, num_components=8, save=False):
        """Plot eigenvectors (left singular vectors)."""
        num_components = min(num_components, self.d)
        nrows = (num_components + 3) // 4
        fig, axes = plt.subplots(nrows, 4, figsize=(16, 3*nrows))
        axes = axes.flatten() if num_components > 4 else [axes] if num_components == 1 else axes.flatten()

        for i in range(num_components):
            axes[i].plot(self.U[:, i], 'b-', linewidth=0.8)
            axes[i].set_title(f'Eigenvector {i+1}\n(σ={self.singular_values[i]:.2f}, '
                            f'{self.contribution[i]:.1f}%)', fontsize=9)
            axes[i].set_xlabel('Index')
            axes[i].axhline(y=0, color='k', linewidth=0.5)

        for i in range(num_components, len(axes)):
            axes[i].set_visible(False)

        plt.suptitle('Eigenvectors (Left Singular Vectors)', fontweight='bold', fontsize=13)
        plt.tight_layout()
        if save: plt.savefig('04_eigenvectors.png')
        plt.show()

    def plot_paired_eigenvectors(self, pairs=None, save=False):
        """
        Scatter plot pasangan eigenvectors.
        Pola lingkaran → komponen periodik (harus dipasangkan).
        """
        if pairs is None:
            n = min(6, self.d - 1)
            pairs = [(i, i+1) for i in range(0, n, 1)]

        n_pairs = len(pairs)
        ncols = min(4, n_pairs)
        nrows = (n_pairs + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))
        if n_pairs == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for idx, (i, j) in enumerate(pairs):
            if i < self.d and j < self.d:
                axes[idx].scatter(self.U[:, i], self.U[:, j], s=5, alpha=0.6, c='steelblue')
                axes[idx].set_xlabel(f'Eigenvector {i+1}')
                axes[idx].set_ylabel(f'Eigenvector {j+1}')
                axes[idx].set_title(f'EV{i+1} vs EV{j+1}')
                axes[idx].set_aspect('equal')
                axes[idx].axhline(y=0, color='k', linewidth=0.3)
                axes[idx].axvline(x=0, color='k', linewidth=0.3)

        for idx in range(n_pairs, len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle('Paired Eigenvectors (Circular → Periodic)', fontweight='bold')
        plt.tight_layout()
        if save: plt.savefig('05_paired_eigenvectors.png')
        plt.show()

    def plot_wcorrelation(self, num_components=None, save=False):
        """
        Plot W-Correlation Matrix.
        """
        if not hasattr(self, 'wcorr_matrix'):
            self.w_correlation(num_components)

        n = self.wcorr_matrix.shape[0]

        fig, ax = plt.subplots(figsize=(10, 8))

        norm_obj = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
        im = ax.imshow(np.abs(self.wcorr_matrix), cmap='RdBu_r', 
                       vmin=0, vmax=1, interpolation='nearest')

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels([f'F{i+1}' for i in range(n)], fontsize=8)
        ax.set_yticklabels([f'F{i+1}' for i in range(n)], fontsize=8)
        ax.set_title('W-Correlation Matrix (|ρ^w|)', fontweight='bold')

        plt.colorbar(im, ax=ax, shrink=0.8, label='|W-Correlation|')

        # Tambah angka
        if n <= 15:
            for i in range(n):
                for j in range(n):
                    val = abs(self.wcorr_matrix[i, j])
                    color = 'white' if val > 0.5 else 'black'
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                           fontsize=7, color=color)

        if save: plt.savefig('06_wcorrelation.png')
        plt.show()

    def plot_components(self, num_components=8, time_index=None, save=False):
        """Plot komponen individual hasil SSA."""
        num_components = min(num_components, self.d)

        fig, axes = plt.subplots(num_components, 1, figsize=(14, 2.5*num_components),
                                 sharex=True)
        if num_components == 1:
            axes = [axes]

        x = time_index if time_index is not None else np.arange(self.N)

        for i in range(num_components):
            rc = self._diagonal_averaging(self.elementary_matrices[i])
            axes[i].plot(x, rc, 'b-', linewidth=0.8)
            axes[i].set_ylabel(f'F{i+1}', fontweight='bold')
            info = f'σ={self.singular_values[i]:.2f} ({self.contribution[i]:.1f}%)'
            axes[i].text(0.98, 0.85, info, transform=axes[i].transAxes,
                        fontsize=8, ha='right', 
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            axes[i].axhline(y=0, color='k', linewidth=0.3)

        axes[-1].set_xlabel('Waktu')
        plt.suptitle('Komponen SSA (Reconstructed Components)', 
                     fontweight='bold', fontsize=13, y=1.01)
        plt.tight_layout()
        if save: plt.savefig('07_ssa_components.png')
        plt.show()

    def plot_reconstruction(self, time_index=None, save=False):
        """Plot hasil rekonstruksi per grup."""
        if not hasattr(self, 'reconstructed'):
            print("[!] Belum ada rekonstruksi. Jalankan reconstruct() dulu.")
            return

        user_groups = {k: v for k, v in self.reconstructed.items() 
                      if not k.startswith('_')}
        n_groups = len(user_groups)

        fig, axes = plt.subplots(n_groups + 1, 1, 
                                figsize=(14, 3*(n_groups + 1)), sharex=True)

        x = time_index if time_index is not None else np.arange(self.N)

        # Original + Total Rekonstruksi
        axes[0].plot(x, self.original, 'b-', alpha=0.5, linewidth=0.8, label='Original')
        axes[0].plot(x, self.reconstructed['_Total'], 'r-', linewidth=1, 
                    label='Rekonstruksi Total')
        axes[0].set_ylabel('Nilai')
        axes[0].legend(loc='upper right')
        axes[0].set_title('Original vs Rekonstruksi Total', fontweight='bold')

        colors = plt.cm.Set1(np.linspace(0, 1, n_groups))
        for idx, (name, values) in enumerate(user_groups.items()):
            axes[idx+1].plot(x, values, color=colors[idx], linewidth=0.8)
            axes[idx+1].set_ylabel(name, fontweight='bold')
            axes[idx+1].axhline(y=0, color='k', linewidth=0.3)

            var_share = sum(self.contribution[i] for i in self.groups[name] 
                          if i < len(self.contribution))
            axes[idx+1].text(0.98, 0.85, f'{var_share:.1f}%',
                           transform=axes[idx+1].transAxes, fontsize=10,
                           ha='right', fontweight='bold',
                           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        axes[-1].set_xlabel('Waktu')
        plt.suptitle(f'SSA Reconstruction — {self.name}', fontweight='bold', 
                    fontsize=13, y=1.01)
        plt.tight_layout()
        if save: plt.savefig('08_reconstruction.png')
        plt.show()

    def plot_forecast(self, method='both', train_size=None, time_index=None, save=False):
        """
        Plot hasil forecasting.

        Parameters
        ----------
        method : str
            'recurrent', 'vector', atau 'both'
        train_size : int, optional
            Garis pemisah train/test
        time_index : array-like, optional
            Indeks waktu kustom
        """
        fig, ax = plt.subplots(figsize=(14, 6))

        ax.plot(range(self.N), self.original, 'b-', linewidth=1, label='Original', alpha=0.7)

        if method in ('recurrent', 'both') and hasattr(self, 'forecast_r'):
            n_total = len(self.forecast_r)
            ax.plot(range(n_total), self.forecast_r, 'r--', linewidth=1.2,
                   label='R-Forecast (Recurrent)', alpha=0.8)

        if method in ('vector', 'both') and hasattr(self, 'forecast_v'):
            n_total = len(self.forecast_v)
            ax.plot(range(n_total), self.forecast_v, 'g--', linewidth=1.2,
                   label='V-Forecast (Vector)', alpha=0.8)

        if train_size is not None:
            ax.axvline(x=train_size, color='orange', linestyle=':', linewidth=2,
                      label=f'Train/Test Split (n={train_size})')

        ax.axvline(x=self.N-1, color='gray', linestyle=':', linewidth=1,
                  label=f'End of Data (N={self.N})')

        ax.set_title(f'SSA Forecasting — {self.name}', fontweight='bold')
        ax.set_xlabel('Waktu')
        ax.set_ylabel('Nilai')
        ax.legend(loc='best')

        if save: plt.savefig('09_forecast.png')
        plt.show()

    def plot_forecast_detail(self, method='recurrent', train_size=None, save=False):
        """Plot detail forecast dengan area training/testing yang disorot."""
        if method == 'recurrent' and hasattr(self, 'forecast_r'):
            forecast = self.forecast_r
            method_name = 'Recurrent (R-forecast)'
        elif method == 'vector' and hasattr(self, 'forecast_v'):
            forecast = self.forecast_v
            method_name = 'Vector (V-forecast)'
        else:
            print("[!] Forecast belum tersedia.")
            return

        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        # Panel atas: Full view
        ax = axes[0]
        ax.plot(range(self.N), self.original, 'b-', linewidth=1, label='Actual', alpha=0.7)
        ax.plot(range(len(forecast)), forecast, 'r--', linewidth=1.2, label=f'{method_name}')

        if train_size:
            ax.axvspan(0, train_size, alpha=0.05, color='green', label='Training')
            ax.axvspan(train_size, self.N, alpha=0.05, color='orange', label='Testing')
            ax.axvspan(self.N, len(forecast), alpha=0.05, color='red', label='Forecast')

        ax.set_title(f'SSA {method_name} — Full View', fontweight='bold')
        ax.legend()
        ax.set_ylabel('Nilai')

        # Panel bawah: Error/Residual
        ax2 = axes[1]
        errors = self.original - forecast[:self.N]

        if train_size:
            ax2.bar(range(train_size), errors[:train_size], color='green', 
                   alpha=0.5, width=1, label='Train Error')
            ax2.bar(range(train_size, self.N), errors[train_size:], color='orange',
                   alpha=0.5, width=1, label='Test Error')
        else:
            ax2.bar(range(self.N), errors, color='steelblue', alpha=0.5, width=1)

        ax2.axhline(y=0, color='k', linewidth=0.5)
        ax2.set_title('Forecast Errors', fontweight='bold')
        ax2.set_xlabel('Waktu')
        ax2.set_ylabel('Error (Actual - Predicted)')
        ax2.legend()

        plt.tight_layout()
        if save: plt.savefig(f'10_forecast_detail_{method}.png')
        plt.show()

    def plot_monte_carlo(self, save=False):
        """Plot hasil Monte Carlo SSA test."""
        if not hasattr(self, 'mc_results'):
            print("[!] Jalankan monte_carlo_test() terlebih dahulu.")
            return

        mc = self.mc_results
        n = len(mc['eigenvalues'])
        x = np.arange(1, n + 1)

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.semilogy(x, mc['eigenvalues'], 'ro-', markersize=8, 
                    label='Data Eigenvalues', zorder=5)
        ax.fill_between(x, mc['surrogate_lower'], mc['surrogate_upper'],
                       alpha=0.3, color='blue', label=f'{mc["confidence"]*100:.0f}% CI (Red Noise)')
        ax.semilogy(x, mc['surrogate_median'], 'b--', linewidth=1, 
                   label='Median Red Noise')

        # Mark significant
        sig = mc['significant']
        ax.semilogy(x[sig], mc['eigenvalues'][sig], 'r*', markersize=15,
                   label='Signifikan', zorder=6)
        ax.semilogy(x[~sig], mc['eigenvalues'][~sig], 'kx', markersize=10,
                   label='Tidak Signifikan', zorder=6)

        ax.set_title('Monte Carlo SSA Significance Test', fontweight='bold')
        ax.set_xlabel('Komponen')
        ax.set_ylabel('Eigenvalue (log scale)')
        ax.set_xticks(x)
        ax.legend()

        if save: plt.savefig('11_monte_carlo_test.png')
        plt.show()

    def plot_periodogram_components(self, num_components=8, save=False):
        """Periodogram untuk tiap komponen (identifikasi frekuensi/periode)."""
        num_components = min(num_components, self.d)

        fig, axes = plt.subplots(num_components, 1, figsize=(14, 2.5*num_components),
                                sharex=True)
        if num_components == 1:
            axes = [axes]

        for i in range(num_components):
            rc = self._diagonal_averaging(self.elementary_matrices[i])
            freqs, psd = periodogram(rc, fs=1.0)

            axes[i].plot(freqs, psd, 'b-', linewidth=0.8)
            axes[i].fill_between(freqs, psd, alpha=0.3)
            axes[i].set_ylabel(f'F{i+1}')

            # Peak frequency
            if len(freqs) > 1:
                peak_idx = np.argmax(psd[1:]) + 1
                peak_freq = freqs[peak_idx]
                period = 1.0/peak_freq if peak_freq > 0 else float('inf')
                axes[i].text(0.98, 0.8, f'Peak: f={peak_freq:.4f}\nT≈{period:.1f}',
                           transform=axes[i].transAxes, fontsize=8, ha='right',
                           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        axes[-1].set_xlabel('Frequency')
        plt.suptitle('Periodogram per Komponen SSA', fontweight='bold', y=1.01)
        plt.tight_layout()
        if save: plt.savefig('12_periodogram.png')
        plt.show()

    def plot_residual_diagnostics(self, residuals=None, save=False):
        """Plot diagnostik residual lengkap (4 panel)."""
        if residuals is None:
            if hasattr(self, 'residuals'):
                residuals = self.residuals
            elif hasattr(self, 'reconstructed') and '_Residual' in self.reconstructed:
                residuals = self.reconstructed['_Residual']
            else:
                print("[!] Residual tidak tersedia.")
                return

        fig = plt.figure(figsize=(14, 10))
        gs = gridspec.GridSpec(2, 2, hspace=0.3, wspace=0.3)

        # 1. Time plot
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(residuals, 'b-', linewidth=0.5)
        ax1.axhline(y=0, color='r', linewidth=0.8)
        ax1.set_title('Residual Time Plot', fontweight='bold')
        ax1.set_xlabel('Waktu')
        ax1.set_ylabel('Residual')

        # 2. Histogram + Normal fit
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.hist(residuals, bins='auto', density=True, alpha=0.7, 
                color='steelblue', edgecolor='navy')
        xmin, xmax = ax2.get_xlim()
        x_norm = np.linspace(xmin, xmax, 100)
        ax2.plot(x_norm, norm.pdf(x_norm, np.mean(residuals), np.std(residuals)),
                'r-', linewidth=2, label='Normal fit')
        ax2.set_title('Histogram Residual', fontweight='bold')
        ax2.legend()

        # 3. ACF
        ax3 = fig.add_subplot(gs[1, 0])
        plot_acf(residuals, ax=ax3, lags=min(40, len(residuals)//3), alpha=0.05)
        ax3.set_title('ACF Residual', fontweight='bold')

        # 4. Q-Q Plot
        ax4 = fig.add_subplot(gs[1, 1])
        from scipy.stats import probplot
        probplot(residuals, dist="norm", plot=ax4)
        ax4.set_title('Q-Q Plot (Normal)', fontweight='bold')

        plt.suptitle('Diagnostik Residual', fontsize=14, fontweight='bold', y=1.02)
        if save: plt.savefig('13_residual_diagnostics.png')
        plt.show()

    def plot_all(self, time_index=None, save=False):
        """Generate semua plot utama sekaligus."""
        print("\n" + "="*70)
        print("  GENERATING ALL PLOTS...")
        print("="*70 + "\n")

        self.plot_original(time_index=time_index, save=save)
        self.plot_scree(save=save)
        self.plot_eigenvectors(save=save)
        self.plot_paired_eigenvectors(save=save)
        self.plot_periodogram_components(save=save)
        if hasattr(self, 'wcorr_matrix'):
            self.plot_wcorrelation(save=save)
        if hasattr(self, 'reconstructed'):
            self.plot_reconstruction(time_index=time_index, save=save)
        if hasattr(self, 'forecast_r') or hasattr(self, 'forecast_v'):
            self.plot_forecast(method='both', time_index=time_index, save=save)
        if hasattr(self, 'residuals') or (hasattr(self, 'reconstructed') and '_Residual' in self.reconstructed):
            self.plot_residual_diagnostics(save=save)
        if hasattr(self, 'mc_results'):
            self.plot_monte_carlo(save=save)

    # ══════════════════════════════════════════════════════════════════════
    # SAVE RESULTS TO EXCEL
    # ══════════════════════════════════════════════════════════════════════
    def save_results(self, filename='SSA_Results.xlsx'):
        """Simpan semua hasil analisis ke file Excel."""
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # 1. Original data
            pd.DataFrame({'Original': self.original}).to_excel(
                writer, sheet_name='Original Data', index_label='t')

            # 2. Eigenvalues & Contribution
            ev_df = pd.DataFrame({
                'Komponen': range(1, self.d + 1),
                'Singular Value': self.singular_values[:self.d],
                'Eigenvalue': self.eigenvalues[:self.d],
                'Contribution (%)': self.contribution,
                'Cumulative (%)': self.cumulative_contribution
            })
            ev_df.to_excel(writer, sheet_name='Eigenvalues', index=False)

            # 3. Reconstructed components
            if hasattr(self, 'reconstructed'):
                rc_data = {'Original': self.original}
                for name, values in self.reconstructed.items():
                    rc_data[name] = values
                pd.DataFrame(rc_data).to_excel(
                    writer, sheet_name='Reconstruction', index_label='t')

            # 4. Forecast
            if hasattr(self, 'forecast_r'):
                fc_data = {'R_Forecast': self.forecast_r}
                if hasattr(self, 'forecast_v'):
                    # Pad if different lengths
                    max_len = max(len(self.forecast_r), len(self.forecast_v))
                    r_padded = np.pad(self.forecast_r, (0, max_len - len(self.forecast_r)), 
                                     constant_values=np.nan)
                    v_padded = np.pad(self.forecast_v, (0, max_len - len(self.forecast_v)),
                                     constant_values=np.nan)
                    fc_data = {'R_Forecast': r_padded, 'V_Forecast': v_padded}
                pd.DataFrame(fc_data).to_excel(
                    writer, sheet_name='Forecast', index_label='t')

            # 5. Evaluation
            if hasattr(self, 'eval_results'):
                eval_data = {}
                for split_name, metrics in self.eval_results.items():
                    if isinstance(metrics, dict):
                        eval_data[split_name] = metrics
                pd.DataFrame(eval_data).to_excel(
                    writer, sheet_name='Evaluation')

            # 6. W-Correlation
            if hasattr(self, 'wcorr_matrix'):
                n = self.wcorr_matrix.shape[0]
                labels = [f'F{i+1}' for i in range(n)]
                pd.DataFrame(self.wcorr_matrix, index=labels, columns=labels).to_excel(
                    writer, sheet_name='W-Correlation')

        print(f"[✓] Hasil disimpan ke: {filename}\n")


# ══════════════════════════════════════════════════════════════════════════════
# FUNGSI HELPER: FIND OPTIMAL WINDOW LENGTH
# ══════════════════════════════════════════════════════════════════════════════
def find_optimal_L(time_series, L_range=None, groups_func=None, metric='RMSE'):
    """
    Mencari Window Length (L) optimal berdasarkan metrik evaluasi.

    Parameters
    ----------
    time_series : array-like
        Data time series
    L_range : list/range
        Range L yang akan dicoba. Default: range(3, N//2+1, step)
    groups_func : callable
        Fungsi yang menerima SSA object dan mengembalikan dict groups.
        Default: menggunakan komponen yang kontribusi > 1%
    metric : str
        Metrik optimasi ('RMSE', 'MAE', 'MAPE')

    Returns
    -------
    dict : Hasil pencarian L optimal
    """
    ts = np.array(time_series, dtype=float)
    N = len(ts)

    if L_range is None:
        step = max(1, N // 40)
        L_range = range(3, N // 2 + 1, step)

    results = []

    print(f"[⏳] Mencari L optimal ({len(list(L_range))} kandidat)...")

    import io
    from contextlib import redirect_stdout

    for L in L_range:
        try:
            f = io.StringIO()
            with redirect_stdout(f):
                ssa = SSA(ts, window_length=L)

                # Default grouping: semua komponen signifikan
                n_sig = max(1, np.sum(ssa.contribution > 1.0))
                groups = {'Signal': list(range(n_sig))}
                ssa.reconstruct(groups)

            residual = ssa.reconstructed['_Residual']
            rmse = np.sqrt(np.mean(residual**2))
            mae = np.mean(np.abs(residual))

            results.append({'L': L, 'RMSE': rmse, 'MAE': mae, 'n_components': n_sig})

        except Exception:
            continue

    if not results:
        print("[!] Tidak ada L yang valid.")
        return None

    df = pd.DataFrame(results)
    best_idx = df[metric.upper()].idxmin()
    best = df.iloc[best_idx]

    print(f"[✓] L optimal berdasarkan {metric}: L = {int(best['L'])}")
    print(f"    RMSE = {best['RMSE']:.6f}, MAE = {best['MAE']:.6f}")
    print(f"    Komponen signifikan: {int(best['n_components'])}\n")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df['L'], df['RMSE'], 'bo-', markersize=3, label='RMSE')
    ax.axvline(x=best['L'], color='r', linestyle='--', label=f'Optimal L={int(best["L"])}')
    ax.set_title('Window Length Optimization', fontweight='bold')
    ax.set_xlabel('Window Length (L)')
    ax.set_ylabel('RMSE')
    ax.legend()
    plt.show()

    return {'best_L': int(best['L']), 'results': df}


# ══════════════════════════════════════════════════════════════════════════════
# INTERACTIVE MENU / PANDUAN PENGGUNAAN
# ══════════════════════════════════════════════════════════════════════════════
def print_guide():
    """Cetak panduan penggunaan lengkap."""
    guide = """
╔══════════════════════════════════════════════════════════════════════════════╗
║           PANDUAN PENGGUNAAN APLIKASI SSA (Singular Spectrum Analysis)      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                            ║
║  LANGKAH-LANGKAH ANALISIS SSA:                                             ║
║  ═══════════════════════════════                                           ║
║                                                                            ║
║  1. PERSIAPAN DATA                                                         ║
║     • Siapkan data time series 1D dalam format CSV/Excel                   ║
║     • Data harus berupa satu kolom numerik                                 ║
║     • Contoh: df = pd.read_csv('data.csv')                                ║
║     • ts = df['kolom_nilai'].values                                        ║
║                                                                            ║
║  2. INISIALISASI SSA                                                       ║
║     • ssa = SSA(ts, window_length='auto', name='Nama Data')                ║
║     • Window length 'auto' → L = N/2 (rekomendasi umum)                   ║
║     • Untuk data periodik: L sebaiknya kelipatan periode                   ║
║     • Atau gunakan: find_optimal_L(ts) untuk mencari L terbaik             ║
║                                                                            ║
║  3. ANALISIS DEKOMPOSISI                                                   ║
║     a. Lihat Scree Plot:                                                   ║
║        ssa.plot_scree()                                                    ║
║     b. Lihat Eigenvectors:                                                 ║
║        ssa.plot_eigenvectors()                                             ║
║     c. Lihat Paired Eigenvectors (deteksi periodik):                       ║
║        ssa.plot_paired_eigenvectors()                                      ║
║     d. Lihat Periodogram komponen:                                         ║
║        ssa.plot_periodogram_components()                                   ║
║     e. W-Correlation Matrix:                                               ║
║        ssa.w_correlation(num_components=12)                                ║
║        ssa.plot_wcorrelation()                                             ║
║                                                                            ║
║  4. GROUPING & REKONSTRUKSI                                                ║
║     a. Manual Grouping:                                                    ║
║        groups = {                                                          ║
║            'Trend': [0, 1],                                                ║
║            'Seasonal_1': [2, 3],                                           ║
║            'Seasonal_2': [4, 5],                                           ║
║            'Noise': list(range(6, 20))                                     ║
║        }                                                                   ║
║        ssa.reconstruct(groups)                                             ║
║                                                                            ║
║     b. Auto Grouping:                                                      ║
║        groups = ssa.auto_group(threshold=0.5)                              ║
║        ssa.reconstruct(groups)                                             ║
║                                                                            ║
║     c. Visualisasi:                                                        ║
║        ssa.plot_reconstruction()                                           ║
║                                                                            ║
║  5. MONTE CARLO SIGNIFICANCE TEST                                          ║
║     mc = ssa.monte_carlo_test(num_surrogates=1000, confidence=0.95)        ║
║     ssa.plot_monte_carlo()                                                 ║
║                                                                            ║
║  6. FORECASTING                                                            ║
║     a. Recurrent (R-forecasting / LRF):                                    ║
║        fc_r = ssa.forecast_recurrent(groups, steps=24)                     ║
║                                                                            ║
║     b. Vector (V-forecasting):                                             ║
║        fc_v = ssa.forecast_vector(groups, steps=24)                        ║
║                                                                            ║
║     c. Visualisasi:                                                        ║
║        ssa.plot_forecast(method='both', train_size=100)                    ║
║        ssa.plot_forecast_detail(method='recurrent', train_size=100)        ║
║                                                                            ║
║  7. EVALUASI (Train/Test/Overall)                                          ║
║     eval_r = ssa.evaluate_split(train_size=0.8, method='recurrent')        ║
║     eval_v = ssa.evaluate_split(train_size=0.8, method='vector')           ║
║                                                                            ║
║  8. ANALISIS RESIDUAL                                                      ║
║     ssa.residual_analysis()                                                ║
║     ssa.plot_residual_diagnostics()                                        ║
║                                                                            ║
║  9. SIMPAN HASIL                                                           ║
║     ssa.save_results('Hasil_SSA.xlsx')                                     ║
║     ssa.plot_all(save=True)  # Simpan semua plot                           ║
║                                                                            ║
║  10. OPTIMASI WINDOW LENGTH                                                ║
║      result = find_optimal_L(ts)                                           ║
║      best_L = result['best_L']                                             ║
║                                                                            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  TIPS UNTUK SKRIPSI:                                                       ║
║  • Gunakan save=True pada setiap plot → menghasilkan PNG 300 DPI           ║
║  • Simpan hasil ke Excel → ssa.save_results()                              ║
║  • Bandingkan R-forecast vs V-forecast                                     ║
║  • Lakukan Monte Carlo test untuk justifikasi ilmiah                       ║
║  • Analisis residual wajib untuk validasi model                            ║
║  • Coba beberapa window length dan bandingkan                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    print(guide)


# ══════════════════════════════════════════════════════════════════════════════
# CONTOH PENGGUNAAN LENGKAP (DEMO)
# ══════════════════════════════════════════════════════════════════════════════
def run_demo():
    """
    Demo lengkap SSA dengan data sintetik.
    Cocok untuk memahami alur analisis sebelum menggunakan data sendiri.
    """
    print("\n" + "█"*70)
    print("  DEMO: ANALISIS SSA LENGKAP")
    print("█"*70 + "\n")

    # ─────────────────────────────────────────────
    # 1. GENERATE DATA SINTETIK
    # ─────────────────────────────────────────────
    np.random.seed(42)
    N = 200
    t = np.arange(N)

    trend = 0.02 * t + 5                          # Trend linear
    seasonal1 = 3 * np.sin(2 * np.pi * t / 12)    # Musiman periode 12
    seasonal2 = 1.5 * np.sin(2 * np.pi * t / 6)   # Musiman periode 6
    noise = np.random.normal(0, 0.5, N)            # Noise

    ts = trend + seasonal1 + seasonal2 + noise

    print(f"Data sintetik: N={N}")
    print(f"  Komponen: Trend + Seasonal(T=12) + Seasonal(T=6) + Noise\n")

    # ─────────────────────────────────────────────
    # 2. INISIALISASI SSA
    # ─────────────────────────────────────────────
    ssa = SSA(ts, window_length=48, name='Demo Sintetik')

    # ─────────────────────────────────────────────
    # 3. VISUALISASI AWAL
    # ─────────────────────────────────────────────
    ssa.plot_original()
    ssa.plot_scree(num_components=20)
    ssa.plot_eigenvectors(num_components=8)
    ssa.plot_paired_eigenvectors()
    ssa.plot_periodogram_components(num_components=8)

    # ─────────────────────────────────────────────
    # 4. W-CORRELATION
    # ─────────────────────────────────────────────
    ssa.w_correlation(num_components=12)
    ssa.plot_wcorrelation()

    # ─────────────────────────────────────────────
    # 5. MONTE CARLO TEST
    # ─────────────────────────────────────────────
    mc = ssa.monte_carlo_test(num_surrogates=500, confidence=0.95)
    ssa.plot_monte_carlo()

    # ─────────────────────────────────────────────
    # 6. GROUPING & REKONSTRUKSI
    # ─────────────────────────────────────────────
    # Manual grouping berdasarkan analisis di atas
    groups = {
        'Trend': [0],
        'Seasonal_1 (T≈12)': [1, 2],
        'Seasonal_2 (T≈6)': [3, 4],
        'Noise': list(range(5, 20))
    }
    ssa.reconstruct(groups)
    ssa.plot_reconstruction()

    # ─────────────────────────────────────────────
    # 7. FORECASTING
    # ─────────────────────────────────────────────
    train_size = 160  # 80% training

    # Gunakan komponen signal (tanpa noise) untuk forecasting
    signal_groups = {k: v for k, v in groups.items() if k != 'Noise'}

    fc_r = ssa.forecast_recurrent(signal_groups, steps=40)
    fc_v = ssa.forecast_vector(signal_groups, steps=40)

    ssa.plot_forecast(method='both', train_size=train_size)
    ssa.plot_forecast_detail(method='recurrent', train_size=train_size)
    ssa.plot_forecast_detail(method='vector', train_size=train_size)

    # ─────────────────────────────────────────────
    # 8. EVALUASI
    # ─────────────────────────────────────────────
    print("\n" + "═"*60)
    print("  EVALUASI R-FORECASTING")
    print("═"*60)
    eval_r = ssa.evaluate_split(train_size=train_size, method='recurrent')

    print("\n" + "═"*60)
    print("  EVALUASI V-FORECASTING")
    print("═"*60)
    eval_v = ssa.evaluate_split(train_size=train_size, method='vector')

    # ─────────────────────────────────────────────
    # 9. ANALISIS RESIDUAL
    # ─────────────────────────────────────────────
    ssa.residual_analysis()
    ssa.plot_residual_diagnostics()

    # ─────────────────────────────────────────────
    # 10. SIMPAN HASIL
    # ─────────────────────────────────────────────
    ssa.save_results('Demo_SSA_Results.xlsx')

    print("\n" + "█"*70)
    print("  DEMO SELESAI!")
    print("  Hasil disimpan di: Demo_SSA_Results.xlsx")
    print("  Untuk menyimpan plot, gunakan: ssa.plot_all(save=True)")
    print("█"*70 + "\n")

    return ssa


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print_guide()

    print("\nApakah Anda ingin menjalankan DEMO? (y/n): ", end='')
    try:
        choice = input().strip().lower()
        if choice in ('y', 'yes', 'ya'):
            ssa_demo = run_demo()
        else:
            print("\nSilakan gunakan kelas SSA sesuai panduan di atas.")
            print("Contoh cepat:")
            print("  import pandas as pd")
            print("  df = pd.read_csv('data_anda.csv')")
            print("  ts = df['kolom_nilai'].values")
            print("  ssa = SSA(ts, window_length='auto', name='Data Saya')")
            print("  ssa.plot_scree()")
    except EOFError:
        print("\nJalankan run_demo() untuk melihat contoh lengkap.")
