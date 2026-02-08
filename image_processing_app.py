import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw, ImageFont, ImageOps
import io, os, time, zipfile, json, math, tempfile, base64
from datetime import datetime
from collections import deque

# Optional imports
try:
    from skimage import exposure, feature, morphology, measure, segmentation, color, restoration, transform, util
    from skimage.filters import threshold_otsu, threshold_local, gaussian, median as sk_median, sobel, roberts, prewitt, scharr, frangi, hessian, unsharp_mask
    from skimage.morphology import disk, square, diamond, erosion, dilation, opening, closing, skeletonize
    from skimage.segmentation import watershed, slic, felzenszwalb, chan_vese
    from skimage.measure import regionprops, label as sk_label
    from skimage.color import rgb2gray, rgb2hsv, rgb2lab, lab2rgb, rgb2ycbcr
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

try:
    from rembg import remove as rembg_remove
    HAS_REMBG = True
except ImportError:
    HAS_REMBG = False

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================
# PAGE CONFIG & STYLE
# ============================================================
st.set_page_config(page_title="Image Processing Suite", page_icon="\U0001f5bc\ufe0f", layout="wide")
st.markdown("""<style>
.main-header{font-size:2.5rem;font-weight:700;background:linear-gradient(135deg,#f093fb,#f5576c);
-webkit-background-clip:text;-webkit-text-fill-color:transparent;text-align:center;padding:1rem 0}
.sub-header{font-size:1.1rem;color:#5D6D7E;text-align:center;margin-bottom:2rem}
.success-box{background:#D5F5E3;border-left:5px solid #27AE60;padding:1rem;border-radius:5px;margin:.5rem 0}
.success-box,.success-box *{color:#145a32!important}
.info-box{background:#D6EAF8;border-left:5px solid #2E86C1;padding:1rem;border-radius:5px;margin:.5rem 0}
.info-box,.info-box *{color:#1a4971!important}
.warn-box{background:#FEF9E7;border-left:5px solid #F39C12;padding:1rem;border-radius:5px;margin:.5rem 0}
.warn-box,.warn-box *{color:#7d6608!important}
.metric-card{background:linear-gradient(135deg,#667eea,#764ba2);padding:1.2rem;border-radius:10px;
text-align:center;color:white!important;margin:.3rem 0}.metric-card *{color:white!important}
.before-after{border:2px solid #ddd;border-radius:8px;padding:5px}
</style>""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE
# ============================================================
state_defaults = {
    "original_image": None,
    "current_image": None,
    "history": deque(maxlen=50),
    "redo_stack": deque(maxlen=50),
    "file_name": "",
    "processing_log": [],
}
for k, v in state_defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ============================================================
# UTILITY FUNCTIONS
# ============================================================
def pil_to_np(img):
    """PIL Image -> numpy array (RGB)"""
    return np.array(img.convert("RGB"))

def np_to_pil(arr):
    """numpy array -> PIL Image"""
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        return Image.fromarray(arr, mode="L")
    return Image.fromarray(arr, mode="RGB")

def push_history(img):
    """Save current state before modification"""
    if st.session_state.current_image is not None:
        st.session_state.history.append(st.session_state.current_image.copy())
        st.session_state.redo_stack.clear()
    st.session_state.current_image = img

def log_op(operation):
    st.session_state.processing_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] {operation}")

def get_image_info(img):
    """Return dict of image metadata"""
    arr = pil_to_np(img)
    info = {
        "Width": img.width, "Height": img.height,
        "Channels": arr.shape[2] if arr.ndim == 3 else 1,
        "Mode": img.mode,
        "Size (KB)": round(len(img.tobytes()) / 1024, 1),
        "Dtype": str(arr.dtype),
        "Min pixel": int(arr.min()), "Max pixel": int(arr.max()),
        "Mean pixel": round(float(arr.mean()), 2),
        "Std pixel": round(float(arr.std()), 2),
    }
    return info

def display_before_after(original, processed, label1="Original", label2="Processed"):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**{label1}**")
        st.image(original, use_container_width=True)
    with c2:
        st.markdown(f"**{label2}**")
        st.image(processed, use_container_width=True)

def get_download_buffer(img, fmt="PNG", quality=95):
    buf = io.BytesIO()
    if fmt.upper() == "JPEG":
        img.convert("RGB").save(buf, format="JPEG", quality=quality)
    elif fmt.upper() == "WEBP":
        img.save(buf, format="WEBP", quality=quality)
    elif fmt.upper() == "BMP":
        img.save(buf, format="BMP")
    elif fmt.upper() == "TIFF":
        img.save(buf, format="TIFF")
    else:
        img.save(buf, format="PNG")
    buf.seek(0)
    return buf

def apply_convolution(img_arr, kernel):
    """Apply custom convolution kernel"""
    if img_arr.ndim == 3:
        result = np.zeros_like(img_arr)
        for c in range(img_arr.shape[2]):
            result[:, :, c] = cv2.filter2D(img_arr[:, :, c], -1, kernel)
        return np.clip(result, 0, 255).astype(np.uint8)
    return np.clip(cv2.filter2D(img_arr, -1, kernel), 0, 255).astype(np.uint8)

# ============================================================
# HEADER
# ============================================================
st.markdown('<div class="main-header">\U0001f5bc\ufe0f Image Processing Suite</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Professional Image Processing \u2022 Filters \u2022 Segmentation \u2022 Enhancement \u2022 Analysis</div>', unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## \U0001f4f7 Image Input")
    input_src = st.radio("Source:", ["Upload Image", "Camera", "Sample Image"], key="src")

    if input_src == "Upload Image":
        uploaded = st.file_uploader("Upload:", type=["png", "jpg", "jpeg", "bmp", "tiff", "webp", "gif"], key="uploader")
        if uploaded:
            img = Image.open(uploaded).convert("RGB")
            if st.session_state.original_image is None or st.session_state.file_name != uploaded.name:
                st.session_state.original_image = img
                st.session_state.current_image = img.copy()
                st.session_state.file_name = uploaded.name
                st.session_state.history.clear()
                st.session_state.redo_stack.clear()
                st.session_state.processing_log = []

    elif input_src == "Camera":
        cam_img = st.camera_input("Take a photo")
        if cam_img:
            img = Image.open(cam_img).convert("RGB")
            st.session_state.original_image = img
            st.session_state.current_image = img.copy()
            st.session_state.file_name = "camera_capture.png"

    elif input_src == "Sample Image":
        sample = st.selectbox("Sample:", ["Gradient", "Checkerboard", "Noise", "Color Bars"])
        if sample == "Gradient":
            arr = np.tile(np.linspace(0, 255, 512, dtype=np.uint8), (512, 1))
            arr = np.stack([arr, arr, arr], axis=2)
        elif sample == "Checkerboard":
            arr = np.zeros((512, 512, 3), dtype=np.uint8)
            for i in range(0, 512, 64):
                for j in range(0, 512, 64):
                    if (i // 64 + j // 64) % 2 == 0:
                        arr[i:i+64, j:j+64] = 255
        elif sample == "Noise":
            arr = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
        else:
            arr = np.zeros((512, 512, 3), dtype=np.uint8)
            colors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255),(255,0,255),(255,255,255),(0,0,0)]
            bw = 512 // len(colors)
            for i, c in enumerate(colors):
                arr[:, i*bw:(i+1)*bw] = c
        img = np_to_pil(arr)
        st.session_state.original_image = img
        st.session_state.current_image = img.copy()
        st.session_state.file_name = f"{sample.lower()}.png"

    # Undo / Redo
    st.markdown("---")
    st.markdown("## \u21a9\ufe0f History")
    uc1, uc2, uc3 = st.columns(3)
    with uc1:
        if st.button("\u21a9 Undo", use_container_width=True, disabled=len(st.session_state.history) == 0):
            st.session_state.redo_stack.append(st.session_state.current_image.copy())
            st.session_state.current_image = st.session_state.history.pop()
            log_op("Undo")
    with uc2:
        if st.button("\u21aa Redo", use_container_width=True, disabled=len(st.session_state.redo_stack) == 0):
            st.session_state.history.append(st.session_state.current_image.copy())
            st.session_state.current_image = st.session_state.redo_stack.pop()
            log_op("Redo")
    with uc3:
        if st.button("\U0001f504 Reset", use_container_width=True):
            if st.session_state.original_image:
                st.session_state.current_image = st.session_state.original_image.copy()
                st.session_state.history.clear()
                st.session_state.redo_stack.clear()
                st.session_state.processing_log = []
                log_op("Reset to original")

    st.caption(f"History: {len(st.session_state.history)} | Redo: {len(st.session_state.redo_stack)}")

    # Image Info
    if st.session_state.current_image:
        st.markdown("---")
        st.markdown("## \u2139\ufe0f Image Info")
        info = get_image_info(st.session_state.current_image)
        for k, v in info.items():
            st.caption(f"**{k}:** {v}")

    # Quick Export
    if st.session_state.current_image:
        st.markdown("---")
        st.markdown("## \U0001f4be Quick Save")
        fmt = st.selectbox("Format:", ["PNG", "JPEG", "WEBP", "BMP", "TIFF"])
        q = st.slider("Quality:", 1, 100, 95) if fmt in ["JPEG", "WEBP"] else 95
        buf = get_download_buffer(st.session_state.current_image, fmt, q)
        fname = os.path.splitext(st.session_state.file_name)[0] + f".{fmt.lower()}"
        st.download_button(f"\U0001f4be Save {fmt}", buf.getvalue(), fname, use_container_width=True)

# ============================================================
# STOP IF NO IMAGE
# ============================================================
if st.session_state.current_image is None:
    st.info("\U0001f4f7 Upload an image, take a photo, or select a sample to get started.")
    st.stop()

cur_img = st.session_state.current_image
orig_img = st.session_state.original_image

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "\U0001f3a8 Adjust", "\U0001f300 Filters", "\u2702\ufe0f Transform",
    "\U0001f4ca Analyze", "\U0001f9e9 Segment", "\u2728 Enhance",
    "\U0001f3ad Advanced", "\U0001f4e6 Batch/Export"
])

# ===================== TAB 1: BASIC ADJUSTMENTS =====================
with tab1:
    st.markdown("## \U0001f3a8 Basic Adjustments")
    cur_arr = pil_to_np(cur_img)

    adj_mode = st.radio("Mode:", ["Brightness/Contrast", "Color/Saturation", "Levels & Curves", "White Balance", "Color Temperature"], horizontal=True)

    if adj_mode == "Brightness/Contrast":
        c1, c2 = st.columns(2)
        with c1:
            brightness = st.slider("Brightness:", -100, 100, 0, key="adj_bright")
            contrast = st.slider("Contrast:", -100, 100, 0, key="adj_contrast")
        with c2:
            gamma = st.slider("Gamma:", 0.1, 3.0, 1.0, 0.05, key="adj_gamma")
            exposure_val = st.slider("Exposure:", -2.0, 2.0, 0.0, 0.1, key="adj_expo")

        if st.button("Apply", key="adj_bc_btn", type="primary"):
            result = cur_arr.astype(float)
            # Brightness
            result = result + brightness
            # Contrast
            factor = (259 * (contrast + 255)) / (255 * (259 - contrast))
            result = factor * (result - 128) + 128
            # Gamma
            if gamma != 1.0:
                result = np.clip(result, 0, 255)
                result = 255 * (result / 255) ** (1.0 / gamma)
            # Exposure
            if exposure_val != 0:
                result = result * (2 ** exposure_val)
            result = np.clip(result, 0, 255).astype(np.uint8)
            push_history(np_to_pil(result))
            log_op(f"Brightness={brightness}, Contrast={contrast}, Gamma={gamma:.2f}, Exposure={exposure_val:.1f}")
            st.rerun()

    elif adj_mode == "Color/Saturation":
        c1, c2 = st.columns(2)
        with c1:
            saturation = st.slider("Saturation:", 0.0, 3.0, 1.0, 0.05, key="adj_sat")
            vibrance = st.slider("Vibrance:", -100, 100, 0, key="adj_vib")
        with c2:
            hue_shift = st.slider("Hue Shift:", -180, 180, 0, key="adj_hue")
            color_balance_r = st.slider("Red:", -50, 50, 0, key="adj_r")
            color_balance_g = st.slider("Green:", -50, 50, 0, key="adj_g")
            color_balance_b = st.slider("Blue:", -50, 50, 0, key="adj_b")

        if st.button("Apply", key="adj_cs_btn", type="primary"):
            result = cur_img.copy()
            # Saturation
            if saturation != 1.0:
                result = ImageEnhance.Color(result).enhance(saturation)
            # Hue shift via HSV
            if hue_shift != 0:
                hsv = cv2.cvtColor(pil_to_np(result), cv2.COLOR_RGB2HSV).astype(float)
                hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
                result = np_to_pil(cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB))
            # Color balance
            arr = pil_to_np(result).astype(float)
            arr[:, :, 0] = np.clip(arr[:, :, 0] + color_balance_r, 0, 255)
            arr[:, :, 1] = np.clip(arr[:, :, 1] + color_balance_g, 0, 255)
            arr[:, :, 2] = np.clip(arr[:, :, 2] + color_balance_b, 0, 255)
            result = np_to_pil(arr.astype(np.uint8))
            push_history(result)
            log_op(f"Saturation={saturation:.2f}, Hue={hue_shift}, RGB=({color_balance_r},{color_balance_g},{color_balance_b})")
            st.rerun()

    elif adj_mode == "Levels & Curves":
        st.markdown("Adjust input/output levels for tone mapping.")
        c1, c2 = st.columns(2)
        with c1:
            in_low = st.slider("Input Low:", 0, 254, 0, key="lv_il")
            in_high = st.slider("Input High:", 1, 255, 255, key="lv_ih")
        with c2:
            out_low = st.slider("Output Low:", 0, 254, 0, key="lv_ol")
            out_high = st.slider("Output High:", 1, 255, 255, key="lv_oh")
            midtone = st.slider("Midtone:", 0.1, 3.0, 1.0, 0.05, key="lv_mid")

        if st.button("Apply Levels", key="lv_btn", type="primary"):
            arr = cur_arr.astype(float)
            arr = (arr - in_low) / max(in_high - in_low, 1) * 255
            arr = np.clip(arr, 0, 255) / 255
            arr = arr ** (1.0 / midtone)
            arr = arr * (out_high - out_low) + out_low
            push_history(np_to_pil(np.clip(arr, 0, 255).astype(np.uint8)))
            log_op(f"Levels: in=({in_low},{in_high}), out=({out_low},{out_high}), mid={midtone:.2f}")
            st.rerun()

    elif adj_mode == "White Balance":
        wb_method = st.selectbox("Method:", ["Gray World", "Max White", "Manual"])
        if wb_method == "Manual":
            temp = st.slider("Temperature (K):", 2000, 12000, 6500, 100)
        if st.button("Apply WB", key="wb_btn", type="primary"):
            arr = cur_arr.astype(float)
            if wb_method == "Gray World":
                avg = arr.mean(axis=(0, 1))
                gray = avg.mean()
                arr[:, :, 0] *= gray / max(avg[0], 1)
                arr[:, :, 1] *= gray / max(avg[1], 1)
                arr[:, :, 2] *= gray / max(avg[2], 1)
            elif wb_method == "Max White":
                mx = arr.max(axis=(0, 1))
                arr[:, :, 0] *= 255.0 / max(mx[0], 1)
                arr[:, :, 1] *= 255.0 / max(mx[1], 1)
                arr[:, :, 2] *= 255.0 / max(mx[2], 1)
            else:
                # Manual temperature adjustment
                t = temp / 6500.0
                arr[:, :, 0] *= min(t * 1.2, 2.0)
                arr[:, :, 2] *= min(1.0 / max(t, 0.3), 2.0)
            push_history(np_to_pil(np.clip(arr, 0, 255).astype(np.uint8)))
            log_op(f"White Balance: {wb_method}")
            st.rerun()

    elif adj_mode == "Color Temperature":
        temp_k = st.slider("Temperature:", 1000, 15000, 6500, 100, key="ct_k")
        tint = st.slider("Tint:", -100, 100, 0, key="ct_tint")
        if st.button("Apply Temp", key="ct_btn", type="primary"):
            arr = cur_arr.astype(float)
            t_factor = temp_k / 6500.0
            arr[:, :, 0] *= np.clip(t_factor * 1.1, 0.5, 2.0)
            arr[:, :, 2] *= np.clip(1.0 / max(t_factor, 0.3) * 0.9, 0.5, 2.0)
            arr[:, :, 1] += tint * 0.5
            push_history(np_to_pil(np.clip(arr, 0, 255).astype(np.uint8)))
            log_op(f"Temp={temp_k}K, Tint={tint}")
            st.rerun()

    display_before_after(orig_img, cur_img)

# ===================== TAB 2: FILTERS & NOISE =====================
with tab2:
    st.markdown("## \U0001f300 Filters & Noise")
    cur_arr = pil_to_np(cur_img)

    filt_cat = st.radio("Category:", ["Blur/Smooth", "Sharpen", "Edge Detection", "Noise Add/Remove", "Artistic", "Custom Kernel"], horizontal=True)

    if filt_cat == "Blur/Smooth":
        blur_type = st.selectbox("Type:", ["Gaussian", "Box/Mean", "Median", "Bilateral", "Motion Blur", "Radial Blur"])
        if blur_type in ["Gaussian", "Box/Mean"]:
            ksize = st.slider("Kernel:", 1, 51, 5, 2, key="blur_k")
        elif blur_type == "Median":
            ksize = st.slider("Kernel:", 3, 51, 5, 2, key="med_k")
        elif blur_type == "Bilateral":
            d = st.slider("Diameter:", 1, 25, 9)
            sc = st.slider("Sigma Color:", 1, 200, 75)
            ss = st.slider("Sigma Space:", 1, 200, 75)
        elif blur_type == "Motion Blur":
            msize = st.slider("Size:", 3, 50, 15)
            mangle = st.slider("Angle:", 0, 180, 0)
        else:
            ramt = st.slider("Amount:", 1, 30, 5)

        if st.button("Apply Blur", key="blur_btn", type="primary"):
            if blur_type == "Gaussian":
                result = cv2.GaussianBlur(cur_arr, (ksize, ksize), 0)
            elif blur_type == "Box/Mean":
                result = cv2.blur(cur_arr, (ksize, ksize))
            elif blur_type == "Median":
                result = cv2.medianBlur(cur_arr, ksize)
            elif blur_type == "Bilateral":
                result = cv2.bilateralFilter(cur_arr, d, sc, ss)
            elif blur_type == "Motion Blur":
                kernel = np.zeros((msize, msize))
                cos_a, sin_a = math.cos(math.radians(mangle)), math.sin(math.radians(mangle))
                for i in range(msize):
                    x = int(msize // 2 + (i - msize // 2) * cos_a)
                    y = int(msize // 2 + (i - msize // 2) * sin_a)
                    if 0 <= x < msize and 0 <= y < msize:
                        kernel[y, x] = 1
                kernel /= max(kernel.sum(), 1)
                result = apply_convolution(cur_arr, kernel)
            else:
                # Radial blur approximation
                result = cur_arr.copy().astype(float)
                for i in range(1, ramt + 1):
                    scale = 1 + i * 0.01
                    M = cv2.getRotationMatrix2D((cur_arr.shape[1]//2, cur_arr.shape[0]//2), 0, scale)
                    rotated = cv2.warpAffine(cur_arr, M, (cur_arr.shape[1], cur_arr.shape[0]))
                    result += rotated.astype(float)
                result = (result / (ramt + 1)).astype(np.uint8)
            push_history(np_to_pil(result))
            log_op(f"Blur: {blur_type}")
            st.rerun()

    elif filt_cat == "Sharpen":
        sharp_type = st.selectbox("Type:", ["Unsharp Mask", "Laplacian", "High Pass", "Detail Enhance"])
        if sharp_type == "Unsharp Mask":
            radius = st.slider("Radius:", 1, 20, 3)
            amount = st.slider("Amount:", 0.1, 5.0, 1.5, 0.1)
        elif sharp_type == "Laplacian":
            lap_k = st.slider("Kernel:", 1, 31, 3, 2)
            lap_s = st.slider("Strength:", 0.1, 3.0, 1.0, 0.1)
        elif sharp_type == "High Pass":
            hp_r = st.slider("Radius:", 1, 30, 3)
            hp_s = st.slider("Strength:", 0.1, 5.0, 1.0, 0.1)

        if st.button("Apply Sharpen", key="sharp_btn", type="primary"):
            if sharp_type == "Unsharp Mask":
                blurred = cv2.GaussianBlur(cur_arr, (0, 0), radius)
                result = cv2.addWeighted(cur_arr, 1 + amount, blurred, -amount, 0)
            elif sharp_type == "Laplacian":
                gray = cv2.cvtColor(cur_arr, cv2.COLOR_RGB2GRAY)
                lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=lap_k)
                lap = np.stack([lap]*3, axis=2)
                result = np.clip(cur_arr.astype(float) + lap_s * lap, 0, 255).astype(np.uint8)
            elif sharp_type == "High Pass":
                blurred = cv2.GaussianBlur(cur_arr.astype(float), (0, 0), hp_r)
                hp = cur_arr.astype(float) - blurred
                result = np.clip(cur_arr.astype(float) + hp_s * hp, 0, 255).astype(np.uint8)
            else:
                result = cv2.detailEnhance(cur_arr, sigma_s=10, sigma_r=0.15)
            push_history(np_to_pil(result))
            log_op(f"Sharpen: {sharp_type}")
            st.rerun()

    elif filt_cat == "Edge Detection":
        edge_type = st.selectbox("Type:", ["Canny", "Sobel", "Laplacian", "Roberts", "Prewitt", "Scharr"])
        if edge_type == "Canny":
            lo = st.slider("Low:", 0, 255, 50, key="canny_lo")
            hi = st.slider("High:", 0, 255, 150, key="canny_hi")
        overlay = st.checkbox("Overlay on original", value=False)

        if st.button("Detect Edges", key="edge_btn", type="primary"):
            gray = cv2.cvtColor(cur_arr, cv2.COLOR_RGB2GRAY)
            if edge_type == "Canny":
                edges = cv2.Canny(gray, lo, hi)
            elif edge_type == "Sobel":
                sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                edges = np.clip(np.sqrt(sx**2 + sy**2), 0, 255).astype(np.uint8)
            elif edge_type == "Laplacian":
                edges = np.clip(np.abs(cv2.Laplacian(gray, cv2.CV_64F)), 0, 255).astype(np.uint8)
            elif edge_type == "Roberts" and HAS_SKIMAGE:
                edges = (roberts(gray / 255.0) * 255).astype(np.uint8)
            elif edge_type == "Prewitt" and HAS_SKIMAGE:
                edges = (prewitt(gray / 255.0) * 255).astype(np.uint8)
            elif edge_type == "Scharr" and HAS_SKIMAGE:
                edges = (scharr(gray / 255.0) * 255).astype(np.uint8)
            else:
                edges = cv2.Canny(gray, 50, 150)

            if overlay:
                edge_rgb = np.stack([edges]*3, axis=2)
                result = np.clip(cur_arr.astype(float) * 0.7 + edge_rgb.astype(float) * 0.3, 0, 255).astype(np.uint8)
            else:
                result = np.stack([edges]*3, axis=2)
            push_history(np_to_pil(result))
            log_op(f"Edge: {edge_type}")
            st.rerun()

    elif filt_cat == "Noise Add/Remove":
        noise_op = st.radio("Operation:", ["Add Noise", "Remove Noise"], horizontal=True)
        if noise_op == "Add Noise":
            ntype = st.selectbox("Type:", ["Gaussian", "Salt & Pepper", "Poisson", "Speckle"])
            namt = st.slider("Amount:", 1, 100, 25)
            if st.button("Add Noise", key="nadd_btn", type="primary"):
                arr = cur_arr.astype(float)
                if ntype == "Gaussian":
                    noise = np.random.normal(0, namt, arr.shape)
                    result = np.clip(arr + noise, 0, 255).astype(np.uint8)
                elif ntype == "Salt & Pepper":
                    result = arr.copy().astype(np.uint8)
                    prob = namt / 200
                    salt = np.random.random(arr.shape[:2]) < prob
                    pepper = np.random.random(arr.shape[:2]) < prob
                    result[salt] = 255
                    result[pepper] = 0
                elif ntype == "Poisson":
                    noisy = np.random.poisson(arr / 255.0 * (namt / 5)) / (namt / 5) * 255
                    result = np.clip(noisy, 0, 255).astype(np.uint8)
                else:
                    speckle = np.random.randn(*arr.shape) * (namt / 100)
                    result = np.clip(arr + arr * speckle, 0, 255).astype(np.uint8)
                push_history(np_to_pil(result))
                log_op(f"Add Noise: {ntype} ({namt})")
                st.rerun()
        else:
            dn_type = st.selectbox("Method:", ["Non-Local Means", "Bilateral", "Gaussian Blur", "Median"])
            dn_str = st.slider("Strength:", 1, 30, 10)
            if st.button("Denoise", key="dn_btn", type="primary"):
                if dn_type == "Non-Local Means":
                    result = cv2.fastNlMeansDenoisingColored(cur_arr, None, dn_str, dn_str, 7, 21)
                elif dn_type == "Bilateral":
                    result = cv2.bilateralFilter(cur_arr, 9, dn_str * 10, dn_str * 10)
                elif dn_type == "Gaussian Blur":
                    k = max(3, dn_str | 1)
                    result = cv2.GaussianBlur(cur_arr, (k, k), 0)
                else:
                    k = max(3, dn_str | 1)
                    result = cv2.medianBlur(cur_arr, k)
                push_history(np_to_pil(result))
                log_op(f"Denoise: {dn_type} (str={dn_str})")
                st.rerun()

    elif filt_cat == "Artistic":
        art_type = st.selectbox("Effect:", ["Oil Painting", "Pencil Sketch", "Watercolor", "Cartoon", "Emboss", "Posterize", "Solarize", "Sepia", "Vintage", "HDR Effect"])
        if st.button("Apply Effect", key="art_btn", type="primary"):
            if art_type == "Oil Painting":
                result = cv2.xphoto.oilPainting(cur_arr, 7, 1) if hasattr(cv2, "xphoto") else cv2.bilateralFilter(cur_arr, 9, 75, 75)
            elif art_type == "Pencil Sketch":
                gray = cv2.cvtColor(cur_arr, cv2.COLOR_RGB2GRAY)
                inv = 255 - gray
                blur = cv2.GaussianBlur(inv, (21, 21), 0)
                sketch = cv2.divide(gray, 255 - blur, scale=256)
                result = np.stack([sketch]*3, axis=2)
            elif art_type == "Watercolor":
                result = cv2.stylization(cur_arr, sigma_s=60, sigma_r=0.6)
            elif art_type == "Cartoon":
                gray = cv2.cvtColor(cur_arr, cv2.COLOR_RGB2GRAY)
                gray = cv2.medianBlur(gray, 5)
                edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
                color = cv2.bilateralFilter(cur_arr, 9, 300, 300)
                result = cv2.bitwise_and(color, color, mask=edges)
            elif art_type == "Emboss":
                kernel = np.array([[-2,-1,0],[-1,1,1],[0,1,2]])
                result = apply_convolution(cur_arr, kernel)
            elif art_type == "Posterize":
                n = 4
                result = (cur_arr // (256 // n)) * (256 // n)
            elif art_type == "Solarize":
                result = cur_arr.copy()
                result[result < 128] = 255 - result[result < 128]
            elif art_type == "Sepia":
                kernel = np.array([[0.272,0.534,0.131],[0.349,0.686,0.168],[0.393,0.769,0.189]])
                result = np.clip(cur_arr.astype(float) @ kernel.T, 0, 255).astype(np.uint8)
            elif art_type == "Vintage":
                sepia_k = np.array([[0.272,0.534,0.131],[0.349,0.686,0.168],[0.393,0.769,0.189]])
                result = np.clip(cur_arr.astype(float) @ sepia_k.T, 0, 255).astype(np.uint8)
                noise = np.random.normal(0, 15, result.shape)
                result = np.clip(result.astype(float) + noise, 0, 255).astype(np.uint8)
                # Vignette
                rows, cols = result.shape[:2]
                Y, X = np.ogrid[:rows, :cols]
                center = (rows/2, cols/2)
                dist = np.sqrt((X - center[1])**2 + (Y - center[0])**2)
                maxd = np.sqrt(center[0]**2 + center[1]**2)
                vignette = 1 - 0.5 * (dist / maxd) ** 2
                for c in range(3):
                    result[:,:,c] = np.clip(result[:,:,c].astype(float) * vignette, 0, 255)
                result = result.astype(np.uint8)
            else:  # HDR
                lab = cv2.cvtColor(cur_arr, cv2.COLOR_RGB2LAB).astype(float)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                lab[:,:,0] = clahe.apply(lab[:,:,0].astype(np.uint8)).astype(float)
                result = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
                result = cv2.detailEnhance(result, sigma_s=12, sigma_r=0.15)
            push_history(np_to_pil(result))
            log_op(f"Artistic: {art_type}")
            st.rerun()

    elif filt_cat == "Custom Kernel":
        st.markdown("Define a custom 3x3 convolution kernel:")
        cols = st.columns(3)
        k = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                with cols[j]:
                    k[i, j] = st.number_input(f"[{i},{j}]", value=0.0, step=0.1, key=f"ck_{i}{j}")
        normalize = st.checkbox("Normalize kernel", value=True)
        if normalize and k.sum() != 0:
            k = k / k.sum()
        st.markdown("**Kernel:**")
        st.dataframe(k, use_container_width=True)
        if st.button("Apply Kernel", key="ck_btn", type="primary"):
            result = apply_convolution(cur_arr, k.astype(np.float32))
            push_history(np_to_pil(result))
            log_op(f"Custom kernel applied")
            st.rerun()

    display_before_after(orig_img, cur_img)

# ===================== TAB 3: TRANSFORM =====================
with tab3:
    st.markdown("## \u2702\ufe0f Transform & Geometry")
    cur_arr = pil_to_np(cur_img)

    tr_mode = st.radio("Mode:", ["Resize", "Crop", "Rotate/Flip", "Perspective", "Distortion"], horizontal=True)

    if tr_mode == "Resize":
        c1, c2 = st.columns(2)
        with c1:
            rw = st.number_input("Width:", 1, 10000, cur_img.width, key="rw")
        with c2:
            rh = st.number_input("Height:", 1, 10000, cur_img.height, key="rh")
        keep_ratio = st.checkbox("Keep aspect ratio", value=True)
        interp = st.selectbox("Interpolation:", ["LANCZOS", "BILINEAR", "BICUBIC", "NEAREST"])
        if keep_ratio:
            ratio = min(rw / cur_img.width, rh / cur_img.height)
            rw = int(cur_img.width * ratio)
            rh = int(cur_img.height * ratio)
            st.caption(f"Actual size: {rw} x {rh}")
        if st.button("Resize", key="rsz_btn", type="primary"):
            interp_map = {"LANCZOS": Image.LANCZOS, "BILINEAR": Image.BILINEAR, "BICUBIC": Image.BICUBIC, "NEAREST": Image.NEAREST}
            result = cur_img.resize((rw, rh), interp_map[interp])
            push_history(result)
            log_op(f"Resize: {rw}x{rh}")
            st.rerun()

    elif tr_mode == "Crop":
        c1, c2 = st.columns(2)
        with c1:
            cx1 = st.number_input("Left:", 0, cur_img.width-1, 0, key="cx1")
            cy1 = st.number_input("Top:", 0, cur_img.height-1, 0, key="cy1")
        with c2:
            cx2 = st.number_input("Right:", 1, cur_img.width, cur_img.width, key="cx2")
            cy2 = st.number_input("Bottom:", 1, cur_img.height, cur_img.height, key="cy2")
        # Presets
        preset = st.selectbox("Preset:", ["Custom", "Center Square", "Rule of Thirds", "16:9", "4:3", "1:1"])
        if preset == "Center Square":
            s = min(cur_img.width, cur_img.height)
            cx1 = (cur_img.width - s) // 2; cy1 = (cur_img.height - s) // 2
            cx2 = cx1 + s; cy2 = cy1 + s
        elif preset == "16:9":
            nh = int(cur_img.width * 9 / 16)
            cy1 = max(0, (cur_img.height - nh) // 2); cy2 = cy1 + nh; cx1 = 0; cx2 = cur_img.width
        elif preset == "4:3":
            nh = int(cur_img.width * 3 / 4)
            cy1 = max(0, (cur_img.height - nh) // 2); cy2 = cy1 + nh; cx1 = 0; cx2 = cur_img.width
        elif preset == "1:1":
            s = min(cur_img.width, cur_img.height)
            cx1 = (cur_img.width - s) // 2; cy1 = (cur_img.height - s) // 2; cx2 = cx1 + s; cy2 = cy1 + s
        if st.button("Crop", key="crop_btn", type="primary"):
            result = cur_img.crop((cx1, cy1, cx2, cy2))
            push_history(result)
            log_op(f"Crop: ({cx1},{cy1})-({cx2},{cy2})")
            st.rerun()

    elif tr_mode == "Rotate/Flip":
        angle = st.slider("Rotation (degrees):", -180, 180, 0, key="rot_ang")
        expand = st.checkbox("Expand canvas", value=True)
        c1, c2, c3 = st.columns(3)
        with c1: flip_h = st.checkbox("Flip Horizontal")
        with c2: flip_v = st.checkbox("Flip Vertical")
        with c3: transpose = st.checkbox("Transpose")
        if st.button("Apply Transform", key="rot_btn", type="primary"):
            result = cur_img
            if angle != 0: result = result.rotate(-angle, expand=expand, resample=Image.BICUBIC, fillcolor=(0,0,0))
            if flip_h: result = result.transpose(Image.FLIP_LEFT_RIGHT)
            if flip_v: result = result.transpose(Image.FLIP_TOP_BOTTOM)
            if transpose: result = result.transpose(Image.TRANSPOSE)
            push_history(result)
            log_op(f"Rotate={angle}, FlipH={flip_h}, FlipV={flip_v}")
            st.rerun()

    elif tr_mode == "Perspective":
        st.markdown("Adjust perspective correction:")
        tilt_x = st.slider("Tilt X:", -30, 30, 0, key="px")
        tilt_y = st.slider("Tilt Y:", -30, 30, 0, key="py")
        if st.button("Apply Perspective", key="persp_btn", type="primary"):
            h, w = cur_arr.shape[:2]
            dx = tilt_x * w / 100
            dy = tilt_y * h / 100
            src = np.float32([[0,0],[w,0],[0,h],[w,h]])
            dst = np.float32([[dx,dy],[w-dx,dy],[0,h],[w,h]])
            M = cv2.getPerspectiveTransform(src, dst)
            result = cv2.warpPerspective(cur_arr, M, (w, h))
            push_history(np_to_pil(result))
            log_op(f"Perspective: tilt=({tilt_x},{tilt_y})")
            st.rerun()

    elif tr_mode == "Distortion":
        dist_type = st.selectbox("Type:", ["Barrel", "Pincushion", "Swirl", "Wave", "Spherize"])
        strength = st.slider("Strength:", -100, 100, 20, key="dist_str")
        if st.button("Apply Distortion", key="dist_btn", type="primary"):
            h, w = cur_arr.shape[:2]
            result = cur_arr.copy()
            cx, cy = w / 2, h / 2
            Y, X = np.mgrid[0:h, 0:w].astype(float)

            if dist_type in ["Barrel", "Pincushion"]:
                dx = X - cx; dy = Y - cy
                r = np.sqrt(dx**2 + dy**2) / max(cx, cy)
                k = strength / 1000 * (1 if dist_type == "Barrel" else -1)
                factor = 1 + k * r**2
                nx = (cx + dx * factor).astype(np.float32)
                ny = (cy + dy * factor).astype(np.float32)
            elif dist_type == "Swirl":
                dx = X - cx; dy = Y - cy
                r = np.sqrt(dx**2 + dy**2)
                theta = strength / 100 * r / max(cx, cy)
                nx = (cx + dx * np.cos(theta) - dy * np.sin(theta)).astype(np.float32)
                ny = (cy + dx * np.sin(theta) + dy * np.cos(theta)).astype(np.float32)
            elif dist_type == "Wave":
                nx = (X + strength * np.sin(2 * np.pi * Y / 60)).astype(np.float32)
                ny = (Y + strength * np.cos(2 * np.pi * X / 60)).astype(np.float32)
            else:  # Spherize
                dx = (X - cx) / cx; dy = (Y - cy) / cy
                r = np.sqrt(dx**2 + dy**2)
                r = np.clip(r, 0, 1)
                theta = np.arcsin(r) / max(np.pi/2, 1e-5) if strength > 0 else r
                factor = np.where(r > 0, theta / (r + 1e-10), 1) * (strength / 50)
                nx = (cx + dx * cx * factor).astype(np.float32)
                ny = (cy + dy * cy * factor).astype(np.float32)

            result = cv2.remap(cur_arr, nx, ny, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            push_history(np_to_pil(result))
            log_op(f"Distortion: {dist_type} (str={strength})")
            st.rerun()

    display_before_after(orig_img, cur_img)

# ===================== TAB 4: ANALYZE =====================
with tab4:
    st.markdown("## \U0001f4ca Image Analysis")
    cur_arr = pil_to_np(cur_img)

    ana_mode = st.radio("Mode:", ["Histogram", "Color Analysis", "Statistics", "Frequency Domain", "Quality Metrics"], horizontal=True)

    if ana_mode == "Histogram":
        hist_type = st.selectbox("Type:", ["RGB Combined", "Per Channel", "Luminance", "Cumulative"])
        if hist_type == "RGB Combined":
            fig = go.Figure()
            for i, (ch, clr) in enumerate(zip(["Red","Green","Blue"], ["red","green","blue"])):
                hist = np.histogram(cur_arr[:,:,i], bins=256, range=(0,256))[0]
                fig.add_trace(go.Scatter(x=list(range(256)), y=hist, name=ch, line=dict(color=clr), fill="tozeroy", opacity=0.4))
            fig.update_layout(title="RGB Histogram", height=450); st.plotly_chart(fig, use_container_width=True)
        elif hist_type == "Per Channel":
            cols = st.columns(3)
            for i, (ch, clr) in enumerate(zip(["Red","Green","Blue"], ["red","green","blue"])):
                with cols[i]:
                    hist = np.histogram(cur_arr[:,:,i], bins=256, range=(0,256))[0]
                    fig = go.Figure(go.Bar(x=list(range(256)), y=hist, marker_color=clr))
                    fig.update_layout(title=ch, height=300); st.plotly_chart(fig, use_container_width=True)
        elif hist_type == "Luminance":
            gray = cv2.cvtColor(cur_arr, cv2.COLOR_RGB2GRAY)
            hist = np.histogram(gray, bins=256, range=(0,256))[0]
            fig = go.Figure(go.Bar(x=list(range(256)), y=hist, marker_color="gray"))
            fig.update_layout(title="Luminance", height=450); st.plotly_chart(fig, use_container_width=True)
        else:
            fig = go.Figure()
            for i, (ch, clr) in enumerate(zip(["R","G","B"], ["red","green","blue"])):
                hist = np.histogram(cur_arr[:,:,i], bins=256, range=(0,256))[0]
                cum = np.cumsum(hist)
                fig.add_trace(go.Scatter(x=list(range(256)), y=cum, name=ch, line=dict(color=clr)))
            fig.update_layout(title="Cumulative", height=450); st.plotly_chart(fig, use_container_width=True)

    elif ana_mode == "Color Analysis":
        co1, co2 = st.columns(2)
        with co1:
            st.markdown("### Color Space Distribution")
            # 3D scatter of pixel colors (sampled)
            step = max(1, cur_arr.shape[0] * cur_arr.shape[1] // 5000)
            flat = cur_arr.reshape(-1, 3)[::step]
            fig = px.scatter_3d(x=flat[:,0], y=flat[:,1], z=flat[:,2], color=[f"rgb({r},{g},{b})" for r,g,b in flat[:1000]],
                labels={"x":"Red","y":"Green","z":"Blue"}, title="RGB Space (sampled)")
            fig.update_layout(height=500); st.plotly_chart(fig, use_container_width=True)
        with co2:
            st.markdown("### Dominant Colors")
            pixels = cur_arr.reshape(-1, 3).astype(np.float32)
            k_colors = st.slider("N colors:", 3, 15, 5)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
            _, labels, centers = cv2.kmeans(pixels[::max(1,len(pixels)//10000)], k_colors, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
            centers = centers.astype(int)
            unique, counts = np.unique(labels, return_counts=True)
            pct = counts / counts.sum() * 100
            palette = np.zeros((80, 400, 3), dtype=np.uint8)
            x = 0
            for idx in np.argsort(-counts):
                w = int(400 * counts[idx] / counts.sum())
                palette[:, x:x+w] = centers[idx]
                x += w
            st.image(palette, caption="Color Palette", use_container_width=True)
            for idx in np.argsort(-counts):
                c = centers[idx]
                st.markdown(f'<div style="display:inline-block;width:20px;height:20px;background:rgb({c[0]},{c[1]},{c[2]});border-radius:3px;margin-right:8px;vertical-align:middle"></div> RGB({c[0]},{c[1]},{c[2]}) — {pct[idx]:.1f}%', unsafe_allow_html=True)

    elif ana_mode == "Statistics":
        st.markdown("### Pixel Statistics")
        info = get_image_info(cur_img)
        cols = st.columns(5)
        for i, (k, v) in enumerate(info.items()):
            with cols[i % 5]:
                st.markdown(f'<div class="metric-card"><h4>{k}</h4><h3>{v}</h3></div>', unsafe_allow_html=True)
        # Per-channel stats
        st.markdown("### Per-Channel Statistics")
        stats = []
        for i, ch in enumerate(["Red", "Green", "Blue"]):
            c = cur_arr[:,:,i]
            stats.append({"Channel": ch, "Mean": round(c.mean(), 2), "Std": round(c.std(), 2),
                "Min": int(c.min()), "Max": int(c.max()), "Median": int(np.median(c)),
                "Skewness": round(float(((c.astype(float) - c.mean()) / max(c.std(), 1e-10)) ** 3).mean() if c.std() > 0 else 0, 3)})
        st.dataframe(pd.DataFrame(stats) if 'pd' in dir() else stats, use_container_width=True)
        import pandas as pd
        st.dataframe(pd.DataFrame(stats), use_container_width=True)

    elif ana_mode == "Frequency Domain":
        st.markdown("### FFT Analysis")
        gray = cv2.cvtColor(cur_arr, cv2.COLOR_RGB2GRAY).astype(float)
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude = 20 * np.log(np.abs(fshift) + 1)
        phase = np.angle(fshift)
        co = st.columns(2)
        with co[0]:
            st.markdown("**Magnitude Spectrum**")
            mag_norm = ((magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-10) * 255).astype(np.uint8)
            st.image(mag_norm, use_container_width=True, clamp=True)
        with co[1]:
            st.markdown("**Phase Spectrum**")
            ph_norm = ((phase - phase.min()) / (phase.max() - phase.min() + 1e-10) * 255).astype(np.uint8)
            st.image(ph_norm, use_container_width=True, clamp=True)

    elif ana_mode == "Quality Metrics":
        st.markdown("### Image Quality")
        gray = cv2.cvtColor(cur_arr, cv2.COLOR_RGB2GRAY)
        # Laplacian variance (blur detection)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        # Noise estimate
        noise_sigma = np.median(np.abs(gray.astype(float) - cv2.medianBlur(gray, 3).astype(float)))
        # Dynamic range
        dr = int(gray.max()) - int(gray.min())
        # Entropy
        hist = np.histogram(gray, bins=256, range=(0,256))[0].astype(float)
        hist = hist / hist.sum()
        entropy = -np.sum(hist[hist>0] * np.log2(hist[hist>0]))

        c1,c2,c3,c4 = st.columns(4)
        with c1: st.markdown(f'<div class="metric-card"><h4>Sharpness</h4><h3>{lap_var:.1f}</h3></div>', unsafe_allow_html=True)
        with c2: st.markdown(f'<div class="metric-card"><h4>Noise Est</h4><h3>{noise_sigma:.1f}</h3></div>', unsafe_allow_html=True)
        with c3: st.markdown(f'<div class="metric-card"><h4>Dynamic Range</h4><h3>{dr}</h3></div>', unsafe_allow_html=True)
        with c4: st.markdown(f'<div class="metric-card"><h4>Entropy</h4><h3>{entropy:.2f}</h3></div>', unsafe_allow_html=True)

        blur_status = "Sharp" if lap_var > 100 else "Moderate" if lap_var > 30 else "Blurry"
        st.markdown(f'<div class="info-box">Blur Detection: <b>{blur_status}</b> (Laplacian variance = {lap_var:.1f})</div>', unsafe_allow_html=True)

# ===================== TAB 5: SEGMENTATION =====================
with tab5:
    st.markdown("## \U0001f9e9 Segmentation & Morphology")
    cur_arr = pil_to_np(cur_img)

    seg_mode = st.radio("Mode:", ["Thresholding", "Morphology", "Contour Detection", "Watershed", "Color Segmentation"], horizontal=True)

    if seg_mode == "Thresholding":
        thr_type = st.selectbox("Method:", ["Global (Binary)", "Otsu", "Adaptive Mean", "Adaptive Gaussian", "Multi-level"])
        if thr_type == "Global (Binary)":
            tval = st.slider("Threshold:", 0, 255, 128)
            inv = st.checkbox("Invert")
        if st.button("Apply Threshold", key="thr_btn", type="primary"):
            gray = cv2.cvtColor(cur_arr, cv2.COLOR_RGB2GRAY)
            if thr_type == "Global (Binary)":
                flag = cv2.THRESH_BINARY_INV if inv else cv2.THRESH_BINARY
                _, result = cv2.threshold(gray, tval, 255, flag)
            elif thr_type == "Otsu":
                _, result = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            elif thr_type == "Adaptive Mean":
                result = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
            elif thr_type == "Adaptive Gaussian":
                result = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            else:
                levels = st.session_state.get("ml_levels", 3)
                step = 256 // levels
                result = (gray // step * step).astype(np.uint8)
            push_history(np_to_pil(np.stack([result]*3, axis=2)))
            log_op(f"Threshold: {thr_type}")
            st.rerun()
        if thr_type == "Multi-level":
            st.session_state["ml_levels"] = st.slider("Levels:", 2, 10, 4)

    elif seg_mode == "Morphology":
        morph_op = st.selectbox("Operation:", ["Erosion", "Dilation", "Opening", "Closing", "Gradient", "Top Hat", "Black Hat", "Skeletonize"])
        kshape = st.selectbox("Kernel Shape:", ["Rectangle", "Ellipse", "Cross"])
        ksize = st.slider("Kernel Size:", 1, 31, 3, 2, key="morph_k")
        iters = st.slider("Iterations:", 1, 10, 1)
        if st.button("Apply Morphology", key="morph_btn", type="primary"):
            shape_map = {"Rectangle": cv2.MORPH_RECT, "Ellipse": cv2.MORPH_ELLIPSE, "Cross": cv2.MORPH_CROSS}
            kern = cv2.getStructuringElement(shape_map[kshape], (ksize, ksize))
            if morph_op == "Erosion": result = cv2.erode(cur_arr, kern, iterations=iters)
            elif morph_op == "Dilation": result = cv2.dilate(cur_arr, kern, iterations=iters)
            elif morph_op == "Opening": result = cv2.morphologyEx(cur_arr, cv2.MORPH_OPEN, kern, iterations=iters)
            elif morph_op == "Closing": result = cv2.morphologyEx(cur_arr, cv2.MORPH_CLOSE, kern, iterations=iters)
            elif morph_op == "Gradient": result = cv2.morphologyEx(cur_arr, cv2.MORPH_GRADIENT, kern)
            elif morph_op == "Top Hat": result = cv2.morphologyEx(cur_arr, cv2.MORPH_TOPHAT, kern)
            elif morph_op == "Black Hat": result = cv2.morphologyEx(cur_arr, cv2.MORPH_BLACKHAT, kern)
            else:
                gray = cv2.cvtColor(cur_arr, cv2.COLOR_RGB2GRAY)
                _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
                skel = np.zeros_like(binary)
                element = kern.copy()
                while True:
                    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, element)
                    temp = cv2.subtract(binary, opened)
                    eroded = cv2.erode(binary, element)
                    skel = cv2.bitwise_or(skel, temp)
                    binary = eroded.copy()
                    if cv2.countNonZero(binary) == 0: break
                result = np.stack([skel]*3, axis=2)
            push_history(np_to_pil(result))
            log_op(f"Morphology: {morph_op} (k={ksize})")
            st.rerun()

    elif seg_mode == "Contour Detection":
        c_mode = st.selectbox("Retrieval:", ["External", "All", "Tree"])
        min_area = st.slider("Min area:", 0, 5000, 100)
        draw_bbox = st.checkbox("Draw bounding boxes", value=True)
        draw_contour = st.checkbox("Draw contours", value=True)
        if st.button("Find Contours", key="cnt_btn", type="primary"):
            gray = cv2.cvtColor(cur_arr, cv2.COLOR_RGB2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            mode_map = {"External": cv2.RETR_EXTERNAL, "All": cv2.RETR_LIST, "Tree": cv2.RETR_TREE}
            contours, _ = cv2.findContours(binary, mode_map[c_mode], cv2.CHAIN_APPROX_SIMPLE)
            contours = [c for c in contours if cv2.contourArea(c) >= min_area]
            result = cur_arr.copy()
            if draw_contour:
                cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
            if draw_bbox:
                for c in contours:
                    x, y, w, h = cv2.boundingRect(c)
                    cv2.rectangle(result, (x, y), (x+w, y+h), (255, 0, 0), 2)
            push_history(np_to_pil(result))
            log_op(f"Contours: {len(contours)} found (min_area={min_area})")
            st.markdown(f'<div class="success-box">Found <b>{len(contours)}</b> contours</div>', unsafe_allow_html=True)
            st.rerun()

    elif seg_mode == "Watershed":
        st.markdown("Automatic watershed segmentation.")
        if st.button("Run Watershed", key="ws_btn", type="primary"):
            gray = cv2.cvtColor(cur_arr, cv2.COLOR_RGB2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            kernel = np.ones((3,3), np.uint8)
            opening_r = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
            sure_bg = cv2.dilate(opening_r, kernel, iterations=3)
            dist_transform = cv2.distanceTransform(opening_r, cv2.DIST_L2, 5)
            _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
            sure_fg = sure_fg.astype(np.uint8)
            unknown = cv2.subtract(sure_bg, sure_fg)
            _, markers = cv2.connectedComponents(sure_fg)
            markers = markers + 1
            markers[unknown == 255] = 0
            img_bgr = cv2.cvtColor(cur_arr, cv2.COLOR_RGB2BGR)
            markers = cv2.watershed(img_bgr, markers)
            result = cur_arr.copy()
            result[markers == -1] = [255, 0, 0]
            # Color regions
            unique_labels = np.unique(markers)
            colored = np.zeros_like(cur_arr)
            np.random.seed(42)
            for lbl in unique_labels:
                if lbl <= 0: continue
                colored[markers == lbl] = np.random.randint(50, 255, 3)
            alpha = st.slider("Overlay alpha:", 0.0, 1.0, 0.5, key="ws_alpha")
            result = np.clip(cur_arr.astype(float) * (1-alpha) + colored.astype(float) * alpha, 0, 255).astype(np.uint8)
            push_history(np_to_pil(result))
            log_op(f"Watershed: {len(unique_labels)-1} regions")
            st.rerun()

    elif seg_mode == "Color Segmentation":
        st.markdown("Segment by color range (HSV).")
        c1, c2 = st.columns(2)
        with c1:
            h_lo = st.slider("Hue Low:", 0, 179, 0); s_lo = st.slider("Sat Low:", 0, 255, 50); v_lo = st.slider("Val Low:", 0, 255, 50)
        with c2:
            h_hi = st.slider("Hue High:", 0, 179, 30); s_hi = st.slider("Sat High:", 0, 255, 255); v_hi = st.slider("Val High:", 0, 255, 255)
        if st.button("Segment", key="cseg_btn", type="primary"):
            hsv = cv2.cvtColor(cur_arr, cv2.COLOR_RGB2HSV)
            mask = cv2.inRange(hsv, np.array([h_lo,s_lo,v_lo]), np.array([h_hi,s_hi,v_hi]))
            result = cv2.bitwise_and(cur_arr, cur_arr, mask=mask)
            push_history(np_to_pil(result))
            log_op(f"Color Segment: H({h_lo}-{h_hi}) S({s_lo}-{s_hi}) V({v_lo}-{v_hi})")
            st.rerun()

    display_before_after(orig_img, cur_img)

# ===================== TAB 6: ENHANCE =====================
with tab6:
    st.markdown("## \u2728 Enhancement")
    cur_arr = pil_to_np(cur_img)

    enh_mode = st.radio("Mode:", ["Histogram EQ", "CLAHE", "Auto Enhance", "Super Resolution", "Inpainting", "Background Remove"], horizontal=True)

    if enh_mode == "Histogram EQ":
        eq_type = st.selectbox("Type:", ["Global", "Per Channel", "YCbCr Luma Only"])
        if st.button("Equalize", key="eq_btn", type="primary"):
            if eq_type == "Global":
                gray = cv2.cvtColor(cur_arr, cv2.COLOR_RGB2GRAY)
                eq = cv2.equalizeHist(gray)
                result = np.stack([eq]*3, axis=2)
            elif eq_type == "Per Channel":
                result = cur_arr.copy()
                for i in range(3): result[:,:,i] = cv2.equalizeHist(result[:,:,i])
            else:
                ycrcb = cv2.cvtColor(cur_arr, cv2.COLOR_RGB2YCrCb)
                ycrcb[:,:,0] = cv2.equalizeHist(ycrcb[:,:,0])
                result = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
            push_history(np_to_pil(result))
            log_op(f"Histogram EQ: {eq_type}")
            st.rerun()

    elif enh_mode == "CLAHE":
        clip = st.slider("Clip Limit:", 1.0, 10.0, 2.0, 0.5)
        grid = st.slider("Grid Size:", 2, 32, 8)
        if st.button("Apply CLAHE", key="clahe_btn", type="primary"):
            clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
            lab = cv2.cvtColor(cur_arr, cv2.COLOR_RGB2LAB)
            lab[:,:,0] = clahe.apply(lab[:,:,0])
            result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            push_history(np_to_pil(result))
            log_op(f"CLAHE: clip={clip}, grid={grid}")
            st.rerun()

    elif enh_mode == "Auto Enhance":
        st.markdown("One-click automatic enhancement pipeline.")
        auto_bright = st.checkbox("Auto Brightness", value=True)
        auto_contrast = st.checkbox("Auto Contrast", value=True)
        auto_color = st.checkbox("Auto Color", value=True)
        auto_sharp = st.checkbox("Auto Sharpen", value=True)
        auto_denoise = st.checkbox("Auto Denoise", value=False)
        if st.button("Auto Enhance", key="auto_btn", type="primary"):
            result = cur_img.copy()
            if auto_contrast:
                result = ImageOps.autocontrast(result, cutoff=1)
            if auto_bright:
                enhancer = ImageEnhance.Brightness(result)
                arr_t = pil_to_np(result)
                mean_lum = cv2.cvtColor(arr_t, cv2.COLOR_RGB2GRAY).mean()
                factor = 128 / max(mean_lum, 1)
                factor = np.clip(factor, 0.7, 1.5)
                result = enhancer.enhance(factor)
            if auto_color:
                result = ImageEnhance.Color(result).enhance(1.15)
            if auto_sharp:
                result = ImageEnhance.Sharpness(result).enhance(1.3)
            if auto_denoise:
                arr_r = pil_to_np(result)
                arr_r = cv2.fastNlMeansDenoisingColored(arr_r, None, 5, 5, 7, 21)
                result = np_to_pil(arr_r)
            push_history(result)
            log_op("Auto Enhance")
            st.rerun()

    elif enh_mode == "Super Resolution":
        st.markdown("Upscale image using interpolation.")
        scale = st.selectbox("Scale:", [2, 3, 4])
        method = st.selectbox("Method:", ["INTER_CUBIC", "INTER_LANCZOS4", "INTER_LINEAR"])
        if st.button("Upscale", key="sr_btn", type="primary"):
            interp_map = {"INTER_CUBIC": cv2.INTER_CUBIC, "INTER_LANCZOS4": cv2.INTER_LANCZOS4, "INTER_LINEAR": cv2.INTER_LINEAR}
            h, w = cur_arr.shape[:2]
            result = cv2.resize(cur_arr, (w*scale, h*scale), interpolation=interp_map[method])
            push_history(np_to_pil(result))
            log_op(f"Upscale: {scale}x ({method})")
            st.rerun()

    elif enh_mode == "Inpainting":
        st.markdown("Paint over areas to repair (uses mask-based inpainting).")
        st.markdown('<div class="info-box">Draw white areas on a black mask to mark regions for inpainting.</div>', unsafe_allow_html=True)
        paint_radius = st.slider("Inpaint Radius:", 1, 20, 5)
        method_ip = st.selectbox("Method:", ["Telea", "Navier-Stokes"])
        # Simple auto-inpainting: repair bright/dark artifacts
        thresh_ip = st.slider("Auto-detect threshold:", 0, 50, 10)
        if st.button("Auto Inpaint", key="ip_btn", type="primary"):
            gray = cv2.cvtColor(cur_arr, cv2.COLOR_RGB2GRAY)
            mask = np.zeros_like(gray)
            mask[gray < thresh_ip] = 255
            mask[gray > 255 - thresh_ip] = 255
            flag = cv2.INPAINT_TELEA if method_ip == "Telea" else cv2.INPAINT_NS
            result = cv2.inpaint(cur_arr, mask, paint_radius, flag)
            push_history(np_to_pil(result))
            log_op(f"Inpaint: {method_ip} (r={paint_radius})")
            st.rerun()

    elif enh_mode == "Background Remove":
        if HAS_REMBG:
            if st.button("Remove Background", key="rembg_btn", type="primary"):
                with st.spinner("Removing background..."):
                    buf = io.BytesIO()
                    cur_img.save(buf, format="PNG"); buf.seek(0)
                    out = rembg_remove(buf.getvalue())
                    result = Image.open(io.BytesIO(out)).convert("RGB")
                    push_history(result)
                    log_op("Background removed")
                    st.rerun()
        else:
            st.info("pip install rembg for background removal")
            st.markdown("**Alternative: Chroma Key**")
            ck_color = st.color_picker("Key color:", "#00FF00")
            ck_thresh = st.slider("Tolerance:", 1, 100, 30)
            if st.button("Apply Chroma Key", key="ck_btn2", type="primary"):
                r, g, b = int(ck_color[1:3],16), int(ck_color[3:5],16), int(ck_color[5:7],16)
                diff = np.sqrt(np.sum((cur_arr.astype(float) - np.array([r,g,b]))**2, axis=2))
                mask = diff < ck_thresh
                result = cur_arr.copy()
                result[mask] = [255, 255, 255]
                push_history(np_to_pil(result))
                log_op(f"Chroma Key: {ck_color}")
                st.rerun()

    display_before_after(orig_img, cur_img)

# ===================== TAB 7: ADVANCED =====================
with tab7:
    st.markdown("## \U0001f3ad Advanced Operations")
    cur_arr = pil_to_np(cur_img)

    adv_mode = st.radio("Mode:", ["Color Spaces", "Blend/Composite", "Watermark", "HDR Tone Map", "Face/Feature Detect", "Drawing Tools"], horizontal=True)

    if adv_mode == "Color Spaces":
        cs = st.selectbox("Convert to:", ["Grayscale", "HSV", "LAB", "YCrCb", "HLS", "LUV", "Negative", "Binary"])
        show_channels = st.checkbox("Show individual channels", value=False)
        if st.button("Convert", key="cs_btn", type="primary"):
            if cs == "Grayscale":
                gray = cv2.cvtColor(cur_arr, cv2.COLOR_RGB2GRAY)
                result = np.stack([gray]*3, axis=2)
            elif cs == "Negative":
                result = 255 - cur_arr
            elif cs == "Binary":
                gray = cv2.cvtColor(cur_arr, cv2.COLOR_RGB2GRAY)
                _, bw = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
                result = np.stack([bw]*3, axis=2)
            else:
                conv_map = {"HSV": cv2.COLOR_RGB2HSV, "LAB": cv2.COLOR_RGB2LAB, "YCrCb": cv2.COLOR_RGB2YCrCb,
                    "HLS": cv2.COLOR_RGB2HLS, "LUV": cv2.COLOR_RGB2LUV}
                converted = cv2.cvtColor(cur_arr, conv_map[cs])
                if show_channels:
                    cols = st.columns(3)
                    ch_names = {"HSV":["H","S","V"],"LAB":["L","A","B"],"YCrCb":["Y","Cr","Cb"],"HLS":["H","L","S"],"LUV":["L","U","V"]}
                    for i, name in enumerate(ch_names.get(cs, ["Ch1","Ch2","Ch3"])):
                        with cols[i]: st.image(converted[:,:,i], caption=name, use_container_width=True, clamp=True)
                result = converted
            push_history(np_to_pil(result))
            log_op(f"Color Space: {cs}")
            st.rerun()

    elif adv_mode == "Blend/Composite":
        st.markdown("Blend current image with another image.")
        blend_file = st.file_uploader("Second image:", type=["png","jpg","jpeg","bmp"], key="blend_up")
        blend_mode = st.selectbox("Mode:", ["Normal", "Multiply", "Screen", "Overlay", "Soft Light", "Difference", "Add", "Subtract"])
        alpha = st.slider("Opacity:", 0.0, 1.0, 0.5, 0.05, key="blend_a")
        if blend_file and st.button("Blend", key="blend_btn", type="primary"):
            img2 = Image.open(blend_file).convert("RGB").resize((cur_img.width, cur_img.height))
            a = cur_arr.astype(float) / 255; b = pil_to_np(img2).astype(float) / 255
            if blend_mode == "Normal": result = a * (1-alpha) + b * alpha
            elif blend_mode == "Multiply": result = a * (1-alpha) + (a * b) * alpha
            elif blend_mode == "Screen": result = a * (1-alpha) + (1 - (1-a)*(1-b)) * alpha
            elif blend_mode == "Overlay":
                ov = np.where(a < 0.5, 2*a*b, 1 - 2*(1-a)*(1-b))
                result = a * (1-alpha) + ov * alpha
            elif blend_mode == "Soft Light":
                sl = np.where(b < 0.5, a*(2*b + a*(1-2*b)), a + (2*b-1)*(np.sqrt(a)-a))
                result = a * (1-alpha) + sl * alpha
            elif blend_mode == "Difference": result = a * (1-alpha) + np.abs(a - b) * alpha
            elif blend_mode == "Add": result = np.clip(a + b * alpha, 0, 1)
            else: result = np.clip(a - b * alpha, 0, 1)
            push_history(np_to_pil((np.clip(result, 0, 1) * 255).astype(np.uint8)))
            log_op(f"Blend: {blend_mode} (a={alpha:.2f})")
            st.rerun()

    elif adv_mode == "Watermark":
        wm_text = st.text_input("Text:", "© 2026")
        wm_pos = st.selectbox("Position:", ["Bottom Right", "Bottom Left", "Top Right", "Top Left", "Center"])
        wm_size = st.slider("Font size:", 10, 200, 40)
        wm_opacity = st.slider("Opacity:", 0, 255, 128)
        wm_color = st.color_picker("Color:", "#FFFFFF")
        if st.button("Add Watermark", key="wm_btn", type="primary"):
            result = cur_img.copy().convert("RGBA")
            overlay = Image.new("RGBA", result.size, (255,255,255,0))
            draw = ImageDraw.Draw(overlay)
            try: font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", wm_size)
            except: font = ImageFont.load_default()
            bbox = draw.textbbox((0,0), wm_text, font=font)
            tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
            pad = 20
            pos_map = {
                "Bottom Right": (result.width-tw-pad, result.height-th-pad),
                "Bottom Left": (pad, result.height-th-pad),
                "Top Right": (result.width-tw-pad, pad),
                "Top Left": (pad, pad),
                "Center": ((result.width-tw)//2, (result.height-th)//2),
            }
            pos = pos_map[wm_pos]
            r,g,b = int(wm_color[1:3],16), int(wm_color[3:5],16), int(wm_color[5:7],16)
            draw.text(pos, wm_text, fill=(r,g,b,wm_opacity), font=font)
            result = Image.alpha_composite(result, overlay).convert("RGB")
            push_history(result)
            log_op(f"Watermark: '{wm_text}'")
            st.rerun()

    elif adv_mode == "HDR Tone Map":
        method_hdr = st.selectbox("Method:", ["Drago", "Reinhard", "Mantiuk"])
        gamma_hdr = st.slider("Gamma:", 0.5, 3.0, 1.0, 0.1)
        saturation_hdr = st.slider("Saturation:", 0.0, 2.0, 1.0, 0.1)
        if st.button("Tone Map", key="hdr_btn", type="primary"):
            hdr = cur_arr.astype(np.float32) / 255.0
            if method_hdr == "Drago": tonemap = cv2.createTonemapDrago(gamma_hdr, saturation_hdr)
            elif method_hdr == "Reinhard": tonemap = cv2.createTonemapReinhard(gamma_hdr, 0, 0, 0)
            else: tonemap = cv2.createTonemapMantiuk(gamma_hdr, saturation_hdr, 0.85)
            result = tonemap.process(hdr)
            result = np.clip(result * 255, 0, 255).astype(np.uint8)
            push_history(np_to_pil(result))
            log_op(f"HDR Tone Map: {method_hdr}")
            st.rerun()

    elif adv_mode == "Face/Feature Detect":
        detect_type = st.selectbox("Detector:", ["Haar Face", "Haar Eyes", "Haar Full Body", "Corner Detection (Harris)", "Blob Detection"])
        if st.button("Detect", key="det_btn", type="primary"):
            result = cur_arr.copy()
            if "Haar" in detect_type:
                gray = cv2.cvtColor(cur_arr, cv2.COLOR_RGB2GRAY)
                cascade_map = {
                    "Haar Face": cv2.data.haarcascades + "haarcascade_frontalface_default.xml",
                    "Haar Eyes": cv2.data.haarcascades + "haarcascade_eye.xml",
                    "Haar Full Body": cv2.data.haarcascades + "haarcascade_fullbody.xml",
                }
                cascade = cv2.CascadeClassifier(cascade_map[detect_type])
                rects = cascade.detectMultiScale(gray, 1.1, 5)
                for (x,y,w,h) in rects:
                    cv2.rectangle(result, (x,y), (x+w,y+h), (0,255,0), 3)
                    cv2.putText(result, detect_type.split()[-1], (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                st.markdown(f'<div class="success-box">Detected <b>{len(rects)}</b> objects</div>', unsafe_allow_html=True)
            elif detect_type == "Corner Detection (Harris)":
                gray = cv2.cvtColor(cur_arr, cv2.COLOR_RGB2GRAY).astype(np.float32)
                corners = cv2.cornerHarris(gray, 2, 3, 0.04)
                result[corners > 0.01 * corners.max()] = [255, 0, 0]
            else:
                params = cv2.SimpleBlobDetector_Params()
                detector = cv2.SimpleBlobDetector_create(params)
                gray = cv2.cvtColor(cur_arr, cv2.COLOR_RGB2GRAY)
                kps = detector.detect(gray)
                for kp in kps:
                    x, y = int(kp.pt[0]), int(kp.pt[1])
                    r = int(kp.size / 2)
                    cv2.circle(result, (x, y), r, (0, 0, 255), 2)
                st.markdown(f'<div class="success-box">Detected <b>{len(kps)}</b> blobs</div>', unsafe_allow_html=True)
            push_history(np_to_pil(result))
            log_op(f"Detect: {detect_type}")
            st.rerun()

    elif adv_mode == "Drawing Tools":
        draw_type = st.selectbox("Tool:", ["Rectangle", "Circle", "Line", "Arrow", "Text"])
        draw_color = st.color_picker("Color:", "#FF0000", key="draw_clr")
        draw_thick = st.slider("Thickness:", 1, 20, 3)
        r,g,b = int(draw_color[1:3],16), int(draw_color[3:5],16), int(draw_color[5:7],16)

        if draw_type == "Rectangle":
            c1,c2 = st.columns(2)
            with c1: dx1=st.number_input("X1:",0,cur_img.width,50,key="dx1");dy1=st.number_input("Y1:",0,cur_img.height,50,key="dy1")
            with c2: dx2=st.number_input("X2:",0,cur_img.width,200,key="dx2");dy2=st.number_input("Y2:",0,cur_img.height,200,key="dy2")
            fill = st.checkbox("Fill", value=False)
            if st.button("Draw", key="draw_btn", type="primary"):
                result = cur_arr.copy()
                t = -1 if fill else draw_thick
                cv2.rectangle(result,(dx1,dy1),(dx2,dy2),(r,g,b),t)
                push_history(np_to_pil(result)); log_op("Draw Rectangle"); st.rerun()
        elif draw_type == "Circle":
            cx_d=st.number_input("Center X:",0,cur_img.width,cur_img.width//2,key="dcx")
            cy_d=st.number_input("Center Y:",0,cur_img.height,cur_img.height//2,key="dcy")
            radius_d=st.slider("Radius:",1,max(cur_img.width,cur_img.height)//2,50,key="drad")
            fill = st.checkbox("Fill",value=False)
            if st.button("Draw", key="draw_c_btn", type="primary"):
                result = cur_arr.copy()
                t=-1 if fill else draw_thick
                cv2.circle(result,(cx_d,cy_d),radius_d,(r,g,b),t)
                push_history(np_to_pil(result)); log_op("Draw Circle"); st.rerun()
        elif draw_type == "Line":
            c1,c2=st.columns(2)
            with c1: lx1=st.number_input("X1:",0,cur_img.width,0,key="lx1");ly1=st.number_input("Y1:",0,cur_img.height,0,key="ly1")
            with c2: lx2=st.number_input("X2:",0,cur_img.width,cur_img.width,key="lx2");ly2=st.number_input("Y2:",0,cur_img.height,cur_img.height,key="ly2")
            if st.button("Draw",key="draw_l_btn",type="primary"):
                result=cur_arr.copy();cv2.line(result,(lx1,ly1),(lx2,ly2),(r,g,b),draw_thick)
                push_history(np_to_pil(result));log_op("Draw Line");st.rerun()
        elif draw_type == "Arrow":
            c1,c2=st.columns(2)
            with c1: ax1=st.number_input("X1:",0,cur_img.width,50,key="ax1");ay1=st.number_input("Y1:",0,cur_img.height,50,key="ay1")
            with c2: ax2=st.number_input("X2:",0,cur_img.width,200,key="ax2");ay2=st.number_input("Y2:",0,cur_img.height,200,key="ay2")
            if st.button("Draw",key="draw_a_btn",type="primary"):
                result=cur_arr.copy();cv2.arrowedLine(result,(ax1,ay1),(ax2,ay2),(r,g,b),draw_thick)
                push_history(np_to_pil(result));log_op("Draw Arrow");st.rerun()
        elif draw_type == "Text":
            dt_text=st.text_input("Text:","Hello!",key="dt_txt")
            dt_x=st.number_input("X:",0,cur_img.width,50,key="dtx")
            dt_y=st.number_input("Y:",0,cur_img.height,50,key="dty")
            dt_scale=st.slider("Scale:",0.5,5.0,1.0,0.1,key="dts")
            if st.button("Draw",key="draw_t_btn",type="primary"):
                result=cur_arr.copy();cv2.putText(result,dt_text,(dt_x,dt_y),cv2.FONT_HERSHEY_SIMPLEX,dt_scale,(r,g,b),draw_thick)
                push_history(np_to_pil(result));log_op(f"Draw Text: {dt_text}");st.rerun()

    display_before_after(orig_img, cur_img)

# ===================== TAB 8: BATCH & EXPORT =====================
with tab8:
    st.markdown("## \U0001f4e6 Batch Processing & Export")

    batch_tab, export_tab, log_tab = st.tabs(["Batch Process", "Export Options", "Processing Log"])

    with batch_tab:
        st.markdown("### Batch Process Multiple Images")
        batch_files = st.file_uploader("Upload images:", type=["png","jpg","jpeg","bmp","webp"], accept_multiple_files=True, key="batch_files")
        if batch_files:
            st.markdown(f"**{len(batch_files)} images loaded**")
            batch_ops = st.multiselect("Operations to apply:", [
                "Resize (512x512)", "Resize (1024x1024)", "Grayscale", "Auto Enhance",
                "Sharpen", "Denoise", "Histogram EQ", "CLAHE", "Sepia", "Rotate 90°",
                "Flip Horizontal", "Gaussian Blur (3)", "Edge Detection"
            ])
            out_fmt = st.selectbox("Output format:", ["PNG", "JPEG", "WEBP"], key="batch_fmt")

            if st.button("Process Batch", key="batch_run", type="primary") and batch_ops:
                zip_buf = io.BytesIO()
                prog = st.progress(0)
                with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                    for idx, bf in enumerate(batch_files):
                        img = Image.open(bf).convert("RGB")
                        arr = pil_to_np(img)

                        for op in batch_ops:
                            if op == "Resize (512x512)": img = img.resize((512,512), Image.LANCZOS); arr = pil_to_np(img)
                            elif op == "Resize (1024x1024)": img = img.resize((1024,1024), Image.LANCZOS); arr = pil_to_np(img)
                            elif op == "Grayscale":
                                g = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY); arr = np.stack([g]*3, axis=2); img = np_to_pil(arr)
                            elif op == "Auto Enhance":
                                img = ImageOps.autocontrast(img, cutoff=1)
                                img = ImageEnhance.Sharpness(img).enhance(1.3); arr = pil_to_np(img)
                            elif op == "Sharpen":
                                blurred = cv2.GaussianBlur(arr, (0,0), 3)
                                arr = cv2.addWeighted(arr, 1.5, blurred, -0.5, 0); img = np_to_pil(arr)
                            elif op == "Denoise":
                                arr = cv2.fastNlMeansDenoisingColored(arr, None, 7, 7, 7, 21); img = np_to_pil(arr)
                            elif op == "Histogram EQ":
                                ycrcb = cv2.cvtColor(arr, cv2.COLOR_RGB2YCrCb)
                                ycrcb[:,:,0] = cv2.equalizeHist(ycrcb[:,:,0])
                                arr = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB); img = np_to_pil(arr)
                            elif op == "CLAHE":
                                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                                lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB); lab[:,:,0] = clahe.apply(lab[:,:,0])
                                arr = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB); img = np_to_pil(arr)
                            elif op == "Sepia":
                                k = np.array([[0.272,0.534,0.131],[0.349,0.686,0.168],[0.393,0.769,0.189]])
                                arr = np.clip(arr.astype(float) @ k.T, 0, 255).astype(np.uint8); img = np_to_pil(arr)
                            elif op == "Rotate 90°":
                                img = img.rotate(-90, expand=True); arr = pil_to_np(img)
                            elif op == "Flip Horizontal":
                                img = img.transpose(Image.FLIP_LEFT_RIGHT); arr = pil_to_np(img)
                            elif op == "Gaussian Blur (3)":
                                arr = cv2.GaussianBlur(arr, (3,3), 0); img = np_to_pil(arr)
                            elif op == "Edge Detection":
                                g = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
                                edges = cv2.Canny(g, 50, 150); arr = np.stack([edges]*3, axis=2); img = np_to_pil(arr)

                        img_buf = get_download_buffer(img, out_fmt)
                        fname = os.path.splitext(bf.name)[0] + f"_processed.{out_fmt.lower()}"
                        zf.writestr(fname, img_buf.getvalue())
                        prog.progress((idx + 1) / len(batch_files))

                prog.empty()
                st.success(f"Processed {len(batch_files)} images!")
                st.download_button("\U0001f4be Download ZIP", zip_buf.getvalue(), "batch_processed.zip", "application/zip", use_container_width=True)

    with export_tab:
        st.markdown("### Export Options")
        if st.session_state.current_image:
            img_exp = st.session_state.current_image

            st.markdown("#### Single Image Export")
            exp_fmt = st.selectbox("Format:", ["PNG", "JPEG", "WEBP", "BMP", "TIFF"], key="exp_fmt")
            exp_q = st.slider("Quality:", 1, 100, 95, key="exp_q") if exp_fmt in ["JPEG", "WEBP"] else 95

            # Resize before export
            exp_resize = st.checkbox("Resize before export")
            if exp_resize:
                erw = st.number_input("Width:", 1, 10000, img_exp.width, key="erw")
                erh = st.number_input("Height:", 1, 10000, img_exp.height, key="erh")
                img_exp = img_exp.resize((erw, erh), Image.LANCZOS)

            buf = get_download_buffer(img_exp, exp_fmt, exp_q)
            size_kb = len(buf.getvalue()) / 1024
            st.caption(f"File size: {size_kb:.1f} KB")
            fname = os.path.splitext(st.session_state.file_name)[0] + f"_edited.{exp_fmt.lower()}"
            st.download_button(f"\U0001f4be Download {exp_fmt}", buf.getvalue(), fname, use_container_width=True, key="exp_dl")

            st.markdown("#### Multi-Format Export")
            if st.button("Export All Formats", key="exp_all"):
                zip_buf = io.BytesIO()
                with zipfile.ZipFile(zip_buf, "w") as zf:
                    for fmt in ["PNG", "JPEG", "WEBP", "BMP"]:
                        b = get_download_buffer(img_exp, fmt)
                        fn = os.path.splitext(st.session_state.file_name)[0] + f".{fmt.lower()}"
                        zf.writestr(fn, b.getvalue())
                st.download_button("\U0001f4be Download All Formats (ZIP)", zip_buf.getvalue(), "all_formats.zip", "application/zip", use_container_width=True)

            st.markdown("#### Compare Export")
            if st.session_state.original_image:
                if st.button("Export Before/After Side by Side", key="exp_ba"):
                    orig = st.session_state.original_image
                    curr = st.session_state.current_image
                    # Resize to same height
                    h = min(orig.height, curr.height, 1080)
                    o_r = orig.resize((int(orig.width * h / orig.height), h))
                    c_r = curr.resize((int(curr.width * h / curr.height), h))
                    combined = Image.new("RGB", (o_r.width + c_r.width + 10, h), (40,40,40))
                    combined.paste(o_r, (0, 0))
                    combined.paste(c_r, (o_r.width + 10, 0))
                    buf = get_download_buffer(combined, "PNG")
                    st.download_button("\U0001f4be Download Comparison", buf.getvalue(), "before_after.png", use_container_width=True)

    with log_tab:
        st.markdown("### Processing Log")
        if st.session_state.processing_log:
            for entry in reversed(st.session_state.processing_log):
                st.markdown(f"- {entry}")
            log_text = "\n".join(st.session_state.processing_log)
            st.download_button("\U0001f4be Download Log", log_text, "processing_log.txt", "text/plain", use_container_width=True)
        else:
            st.info("No operations recorded yet.")

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown('<div style="text-align:center;color:#888;font-size:0.85rem">Image Processing Suite v1.0 \u2022 OpenCV \u2022 Pillow \u2022 scikit-image</div>', unsafe_allow_html=True)
