import streamlit as st
import numpy as np
import io, os, json, time, tempfile, zipfile, base64
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont, ImageOps

# ============================================================
# SAFE IMPORTS
# ============================================================
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.applications import (
        MobileNetV2, ResNet50, EfficientNetB0, InceptionV3, VGG16, DenseNet121
    )
    from tensorflow.keras.applications import (
        mobilenet_v2, resnet50, efficientnet, inception_v3, vgg16, densenet
    )
    from tensorflow.keras.models import load_model, Model
    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    HAS_TF = True
except ImportError:
    HAS_TF = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    import pandas as pd
    HAS_PD = True
except ImportError:
    HAS_PD = False

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="AI Image Classifier", page_icon="\U0001f9e0", layout="wide")
st.markdown("""<style>
.main-header{font-size:2.5rem;font-weight:700;background:linear-gradient(135deg,#667eea,#764ba2);
-webkit-background-clip:text;-webkit-text-fill-color:transparent;text-align:center;padding:1rem 0}
.sub-header{font-size:1.1rem;color:#5D6D7E;text-align:center;margin-bottom:2rem}
.result-card{background:linear-gradient(135deg,#11998e,#38ef7d);padding:1.5rem;border-radius:15px;
text-align:center;color:white!important;margin:1rem 0;box-shadow:0 4px 15px rgba(0,0,0,0.2)}
.result-card *{color:white!important}
.metric-card{background:linear-gradient(135deg,#667eea,#764ba2);padding:1rem;border-radius:10px;
text-align:center;color:white!important;margin:.3rem 0}.metric-card *{color:white!important}
.info-box{background:#D6EAF8;border-left:5px solid #2E86C1;padding:1rem;border-radius:5px;margin:.5rem 0}
.info-box,.info-box *{color:#1a4971!important}
.warn-box{background:#FEF9E7;border-left:5px solid #F39C12;padding:1rem;border-radius:5px;margin:.5rem 0}
.warn-box,.warn-box *{color:#7d6608!important}
.success-box{background:#D5F5E3;border-left:5px solid #27AE60;padding:1rem;border-radius:5px;margin:.5rem 0}
.success-box,.success-box *{color:#145a32!important}
</style>""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================
st.markdown('<div class="main-header">\U0001f9e0 AI Image Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Upload gambar \u2192 AI mengenali kategorinya secara otomatis</div>', unsafe_allow_html=True)

if not HAS_TF:
    st.error("\u274c **TensorFlow belum terinstall!** Tambahkan `tensorflow` ke `requirements.txt`")
    st.code("# requirements.txt\nstreamlit\ntensorflow\nPillow\nplotly\npandas\nopencv-python-headless", language="text")
    st.stop()

# ============================================================
# MODEL REGISTRY
# ============================================================
PRETRAINED_MODELS = {
    "MobileNetV2 (Ringan & Cepat)": {
        "class": MobileNetV2, "preprocess": mobilenet_v2.preprocess_input,
        "decode": mobilenet_v2.decode_predictions, "size": (224, 224),
        "desc": "Model ringan, cocok untuk mobile & deployment cepat"
    },
    "ResNet50 (Akurasi Tinggi)": {
        "class": ResNet50, "preprocess": resnet50.preprocess_input,
        "decode": resnet50.decode_predictions, "size": (224, 224),
        "desc": "Model klasik dengan akurasi tinggi untuk klasifikasi umum"
    },
    "EfficientNetB0 (Efisien)": {
        "class": EfficientNetB0, "preprocess": efficientnet.preprocess_input,
        "decode": efficientnet.decode_predictions, "size": (224, 224),
        "desc": "Keseimbangan terbaik antara akurasi dan kecepatan"
    },
    "InceptionV3 (Detail Tinggi)": {
        "class": InceptionV3, "preprocess": inception_v3.preprocess_input,
        "decode": inception_v3.decode_predictions, "size": (299, 299),
        "desc": "Bagus untuk gambar dengan detail dan pola kompleks"
    },
    "VGG16 (Klasik)": {
        "class": VGG16, "preprocess": vgg16.preprocess_input,
        "decode": vgg16.decode_predictions, "size": (224, 224),
        "desc": "Arsitektur klasik, mudah dipahami"
    },
    "DenseNet121 (Padat)": {
        "class": DenseNet121, "preprocess": densenet.preprocess_input,
        "decode": densenet.decode_predictions, "size": (224, 224),
        "desc": "Koneksi padat antar layer, sangat efisien"
    },
}

# ============================================================
# CLASSIFICATION PRESETS
# ============================================================
CLASSIFICATION_PRESETS = {
    "\U0001f338 Jenis Bunga": {
        "categories": ["Daisy", "Dandelion", "Rose", "Sunflower", "Tulip"],
        "description": "Klasifikasi 5 jenis bunga populer",
    },
    "\U0001f9b4 Postur Tulang Belakang": {
        "categories": ["Normal", "Kyphosis (Kifosis/Bungkuk)", "Lordosis (Lordosis)", "Scoliosis (Skoliosis)"],
        "description": "Deteksi kelainan tulang belakang: kifosis, lordosis, skoliosis",
    },
    "\U0001f3e5 Penyakit Paru (X-Ray)": {
        "categories": ["Normal", "Pneumonia", "COVID-19", "Tuberculosis"],
        "description": "Klasifikasi X-Ray paru-paru",
    },
    "\U0001f431 Hewan Peliharaan": {
        "categories": ["Dog", "Cat", "Bird", "Fish", "Hamster", "Rabbit"],
        "description": "Mengenali jenis hewan peliharaan",
    },
    "\U0001f34e Buah-buahan": {
        "categories": ["Apple", "Banana", "Orange", "Grape", "Mango", "Strawberry", "Watermelon"],
        "description": "Klasifikasi jenis buah",
    },
    "\U0001f697 Kendaraan": {
        "categories": ["Car", "Motorcycle", "Bus", "Truck", "Bicycle"],
        "description": "Mengenali jenis kendaraan",
    },
    "\u270d\ufe0f Tulisan Tangan (Digit)": {
        "categories": ["0","1","2","3","4","5","6","7","8","9"],
        "description": "Mengenali angka tulisan tangan (MNIST)",
    },
    "\U0001f3d7\ufe0f Custom (Buat Sendiri)": {
        "categories": [],
        "description": "Buat kategori klasifikasi sendiri",
    },
}

# ============================================================
# STATE
# ============================================================
defaults = {
    "model_loaded": None, "model_name": None,
    "custom_model": None, "custom_categories": [],
    "classification_history": [], "training_log": [],
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ============================================================
# HELPERS
# ============================================================
@st.cache_resource
def load_pretrained_model(model_name):
    info = PRETRAINED_MODELS[model_name]
    model = info["class"](weights="imagenet")
    return model

def preprocess_image(img_pil, target_size):
    img = img_pil.convert("RGB").resize(target_size)
    arr = np.array(img).astype(np.float32)
    arr = np.expand_dims(arr, axis=0)
    return arr

def classify_imagenet(img_pil, model_name, top_k=10):
    info = PRETRAINED_MODELS[model_name]
    model = load_pretrained_model(model_name)
    arr = preprocess_image(img_pil, info["size"])
    arr = info["preprocess"](arr)
    preds = model.predict(arr, verbose=0)
    decoded = info["decode"](preds, top=top_k)
    results = []
    for (class_id, label, score) in decoded[0]:
        results.append({"class_id": class_id, "label": label.replace("_", " ").title(), "confidence": float(score)})
    return results

def build_transfer_model(base_model_name, num_classes, freeze_base=True):
    info = PRETRAINED_MODELS[base_model_name]
    base = info["class"](weights="imagenet", include_top=False, input_shape=(*info["size"], 3))
    if freeze_base:
        for layer in base.layers:
            layer.trainable = False
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.2)(x)
    output = Dense(num_classes, activation="softmax")(x)
    model = Model(inputs=base.input, outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def generate_grad_cam(model, img_array, class_index, last_conv_layer_name=None):
    if last_conv_layer_name is None:
        for layer in reversed(model.layers):
            if len(layer.output_shape) == 4:
                last_conv_layer_name = layer.name
                break
    if last_conv_layer_name is None:
        return None
    grad_model = Model(inputs=model.input, outputs=[model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(img_array)
        loss = predictions[:, class_index]
    grads = tape.gradient(loss, conv_output)
    if grads is None:
        return None
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()

def overlay_heatmap(img_pil, heatmap, alpha=0.4):
    if heatmap is None:
        return img_pil
    img = np.array(img_pil.convert("RGB").resize((224, 224)))
    heatmap_resized = np.array(Image.fromarray((heatmap * 255).astype(np.uint8)).resize((224, 224)))
    if HAS_CV2:
        heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    else:
        heatmap_colored = np.stack([heatmap_resized, np.zeros_like(heatmap_resized), 255 - heatmap_resized], axis=2)
    blended = np.clip(img.astype(float) * (1 - alpha) + heatmap_colored.astype(float) * alpha, 0, 255).astype(np.uint8)
    return Image.fromarray(blended)

def get_download_buffer(img, fmt="PNG", quality=95):
    buf = io.BytesIO()
    if fmt.upper() == "JPEG":
        img.convert("RGB").save(buf, format="JPEG", quality=quality)
    else:
        img.save(buf, format=fmt.upper())
    buf.seek(0)
    return buf

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## \u2699\ufe0f Pengaturan")

    st.markdown("### \U0001f916 Pilih Model")
    model_choice = st.selectbox("Model AI:", list(PRETRAINED_MODELS.keys()), key="model_select")
    st.caption(PRETRAINED_MODELS[model_choice]["desc"])

    st.markdown("### \U0001f3af Mode Klasifikasi")
    class_mode = st.radio("Mode:", ["ImageNet (1000 Kategori)", "Preset Kategori", "Custom Model (.h5)"], key="class_mode")

    preset_choice = None
    custom_cats = []
    if class_mode == "Preset Kategori":
        preset_choice = st.selectbox("Pilih Preset:", list(CLASSIFICATION_PRESETS.keys()), key="preset_sel")
        preset = CLASSIFICATION_PRESETS[preset_choice]
        st.caption(preset["description"])
        if preset_choice == "\U0001f3d7\ufe0f Custom (Buat Sendiri)":
            cats_text = st.text_area("Kategori (satu per baris):", "Kucing\nAnjing\nBurung")
            custom_cats = [c.strip() for c in cats_text.strip().split("\n") if c.strip()]
            st.caption(f"{len(custom_cats)} kategori")
        else:
            custom_cats = preset["categories"]
            st.markdown("**Kategori:**")
            for c in custom_cats:
                st.markdown(f"- {c}")

    elif class_mode == "Custom Model (.h5)":
        model_file = st.file_uploader("Upload model (.h5/.keras):", type=["h5", "keras"], key="model_upload")
        labels_file = st.file_uploader("Upload labels (.txt/.json):", type=["txt", "json"], key="labels_upload")
        if model_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp:
                tmp.write(model_file.read())
                tmp_path = tmp.name
            try:
                st.session_state.custom_model = load_model(tmp_path)
                st.success("Model loaded!")
            except Exception as e:
                st.error(f"Error: {e}")
        if labels_file:
            content = labels_file.read().decode("utf-8")
            if labels_file.name.endswith(".json"):
                custom_cats = json.loads(content)
            else:
                custom_cats = [l.strip() for l in content.strip().split("\n") if l.strip()]
            st.session_state.custom_categories = custom_cats
            st.success(f"{len(custom_cats)} labels loaded")

    st.markdown("---")
    st.markdown("### \U0001f4ca Pengaturan Output")
    top_k = st.slider("Top-K Prediksi:", 1, 20, 5)
    show_gradcam = st.checkbox("Tampilkan Grad-CAM", value=True)
    confidence_threshold = st.slider("Confidence Min (%):", 0, 100, 5)

    st.markdown("---")
    st.markdown("### \U0001f4cb History")
    if st.session_state.classification_history:
        st.caption(f"{len(st.session_state.classification_history)} klasifikasi")
        if st.button("\U0001f5d1\ufe0f Clear History"):
            st.session_state.classification_history = []
            st.rerun()

# ============================================================
# MAIN TABS
# ============================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "\U0001f50d Klasifikasi", "\U0001f4da Batch", "\U0001f3eb Training",
    "\U0001f4c8 History", "\u2139\ufe0f Panduan"
])

# ===================== TAB 1: KLASIFIKASI =====================
with tab1:
    st.markdown("## \U0001f50d Klasifikasi Gambar")
    input_method = st.radio("Input:", ["Upload Gambar", "Kamera", "URL Gambar"], horizontal=True)

    img_input = None
    if input_method == "Upload Gambar":
        uploaded = st.file_uploader("Pilih gambar:", type=["png","jpg","jpeg","bmp","webp","tiff"], key="main_upload")
        if uploaded:
            img_input = Image.open(uploaded).convert("RGB")
    elif input_method == "Kamera":
        cam = st.camera_input("Ambil foto")
        if cam:
            img_input = Image.open(cam).convert("RGB")
    elif input_method == "URL Gambar":
        url = st.text_input("URL gambar:", placeholder="https://example.com/image.jpg")
        if url:
            try:
                import urllib.request
                with urllib.request.urlopen(url) as resp:
                    img_input = Image.open(io.BytesIO(resp.read())).convert("RGB")
            except Exception as e:
                st.error(f"Gagal load: {e}")

    if img_input:
        col_img, col_result = st.columns([1, 1])

        with col_img:
            st.markdown("### \U0001f5bc\ufe0f Gambar Input")
            st.image(img_input, width=None)
            st.caption(f"Ukuran: {img_input.width} x {img_input.height} px")

        with col_result:
            st.markdown("### \U0001f3af Hasil Klasifikasi")

            if st.button("\U0001f680 Klasifikasi Sekarang!", type="primary", use_container_width=True, key="classify_btn"):
                with st.spinner("Menganalisis gambar..."):
                    start_time = time.time()

                    if class_mode == "ImageNet (1000 Kategori)":
                        results = classify_imagenet(img_input, model_choice, top_k=top_k)
                        results = [r for r in results if r["confidence"] >= confidence_threshold / 100]

                    elif class_mode == "Preset Kategori":
                        imagenet_results = classify_imagenet(img_input, model_choice, top_k=50)
                        results = []
                        for cat in custom_cats:
                            max_conf = 0
                            cat_lower = cat.lower().split("(")[0].strip().split("/")[0].strip()
                            for r in imagenet_results:
                                if cat_lower in r["label"].lower() or r["label"].lower() in cat_lower:
                                    max_conf = max(max_conf, r["confidence"])
                            for r in imagenet_results:
                                words_r = set(r["label"].lower().split())
                                words_c = set(cat_lower.split())
                                if words_r & words_c:
                                    max_conf = max(max_conf, r["confidence"])
                            results.append({"label": cat, "confidence": max_conf, "class_id": ""})
                        total = sum(r["confidence"] for r in results)
                        if total > 0:
                            for r in results:
                                r["confidence"] /= total
                        results.sort(key=lambda x: x["confidence"], reverse=True)
                        results = results[:top_k]

                    elif class_mode == "Custom Model (.h5)":
                        if st.session_state.custom_model is not None:
                            cmodel = st.session_state.custom_model
                            inp_shape = cmodel.input_shape[1:3]
                            arr = preprocess_image(img_input, inp_shape)
                            arr = arr / 255.0
                            preds = cmodel.predict(arr, verbose=0)[0]
                            labels = st.session_state.custom_categories or [f"Class {i}" for i in range(len(preds))]
                            results = []
                            for i, conf in enumerate(preds):
                                lbl = labels[i] if i < len(labels) else f"Class {i}"
                                results.append({"label": lbl, "confidence": float(conf), "class_id": str(i)})
                            results.sort(key=lambda x: x["confidence"], reverse=True)
                            results = results[:top_k]
                        else:
                            st.warning("Upload model terlebih dahulu!")
                            results = []
                    else:
                        results = []

                    elapsed = time.time() - start_time

                if results:
                    best = results[0]
                    st.markdown(f"""
                    <div class="result-card">
                        <h2>\U0001f3af {best['label']}</h2>
                        <h3>{best['confidence']*100:.1f}% confidence</h3>
                        <p>Model: {model_choice.split('(')[0].strip()} \u2022 {elapsed:.2f}s</p>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("#### Top Prediksi:")
                    for i, r in enumerate(results):
                        pct = r["confidence"] * 100
                        emoji = ["\U0001f947", "\U0001f948", "\U0001f949"][i] if i < 3 else f"#{i+1}"
                        st.progress(min(r["confidence"], 1.0), text=f"{emoji} {r['label']} — {pct:.1f}%")

                    if show_gradcam:
                        st.markdown("#### \U0001f525 Grad-CAM (Area Fokus AI)")
                        try:
                            model = load_pretrained_model(model_choice)
                            info = PRETRAINED_MODELS[model_choice]
                            arr = preprocess_image(img_input, info["size"])
                            arr_p = info["preprocess"](arr.copy())
                            preds = model.predict(arr_p, verbose=0)
                            top_class = np.argmax(preds[0])
                            heatmap = generate_grad_cam(model, arr_p, top_class)
                            if heatmap is not None:
                                cam_img = overlay_heatmap(img_input, heatmap)
                                gc1, gc2 = st.columns(2)
                                with gc1:
                                    st.image(img_input, caption="Original", width=None)
                                with gc2:
                                    st.image(cam_img, caption="Grad-CAM", width=None)
                                st.caption("Merah/kuning = area paling diperhatikan AI")
                            else:
                                st.info("Grad-CAM tidak tersedia.")
                        except Exception as e:
                            st.warning(f"Grad-CAM error: {e}")

                    st.session_state.classification_history.append({
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "model": model_choice.split("(")[0].strip(),
                        "top_prediction": best["label"],
                        "confidence": best["confidence"],
                        "all_predictions": results,
                        "elapsed": elapsed,
                    })
                else:
                    st.warning("Tidak ada hasil.")

# ===================== TAB 2: BATCH =====================
with tab2:
    st.markdown("## \U0001f4da Batch Klasifikasi")
    st.markdown("Upload banyak gambar sekaligus.")

    batch_files = st.file_uploader("Upload gambar:", type=["png","jpg","jpeg","bmp","webp"],
                                    accept_multiple_files=True, key="batch_upload")
    if batch_files:
        st.markdown(f"**{len(batch_files)} gambar**")

        if st.button("\U0001f680 Klasifikasi Semua", type="primary", key="batch_btn"):
            all_results = []
            prog = st.progress(0)
            status = st.empty()
            for idx, f in enumerate(batch_files):
                status.text(f"Mengklasifikasi {f.name}... ({idx+1}/{len(batch_files)})")
                img = Image.open(f).convert("RGB")
                res = classify_imagenet(img, model_choice, top_k=top_k)
                if res:
                    all_results.append({
                        "filename": f.name,
                        "prediction": res[0]["label"],
                        "confidence": f"{res[0]['confidence']*100:.1f}%",
                        "top_3": " | ".join([f"{r['label']} ({r['confidence']*100:.1f}%)" for r in res[:3]]),
                    })
                prog.progress((idx + 1) / len(batch_files))
            prog.empty()
            status.empty()

            if all_results and HAS_PD:
                df = pd.DataFrame(all_results)
                st.dataframe(df, use_container_width=True)

                st.markdown("### \U0001f4ca Ringkasan")
                pred_counts = df["prediction"].value_counts()
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Distribusi Kategori:**")
                    for cat, cnt in pred_counts.items():
                        st.markdown(f"- **{cat}**: {cnt} gambar ({cnt/len(df)*100:.0f}%)")
                with c2:
                    if HAS_PLOTLY:
                        fig = px.pie(values=pred_counts.values, names=pred_counts.index, title="Distribusi")
                        st.plotly_chart(fig, use_container_width=True)

                csv_buf = io.StringIO()
                df.to_csv(csv_buf, index=False)
                st.download_button("\U0001f4be Download CSV", csv_buf.getvalue(), "batch_results.csv", "text/csv", use_container_width=True)

            st.markdown("### \U0001f5bc\ufe0f Gallery")
            cols = st.columns(4)
            for idx, f in enumerate(batch_files):
                with cols[idx % 4]:
                    img = Image.open(f).convert("RGB")
                    cap = f"{all_results[idx]['prediction']} ({all_results[idx]['confidence']})" if idx < len(all_results) else f.name
                    st.image(img, caption=cap, width=None)

# ===================== TAB 3: TRAINING =====================
with tab3:
    st.markdown("## \U0001f3eb Training Custom Model")
    st.markdown("Latih model sendiri dengan **transfer learning**.")

    st.markdown("""
    <div class="info-box">
    <b>Cara Pakai:</b><br>
    1. Siapkan folder ZIP berisi sub-folder per kategori (misal: <code>daisy/*.jpg</code>, <code>rose/*.jpg</code>)<br>
    2. Upload ZIP tersebut<br>
    3. Pilih parameter training<br>
    4. Klik Train!<br>
    5. Download model hasil training (.h5 + labels.txt)
    </div>
    """, unsafe_allow_html=True)

    train_zip = st.file_uploader("Upload dataset (ZIP):", type=["zip"], key="train_zip")

    if train_zip:
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = os.path.join(tmpdir, "dataset.zip")
            with open(zip_path, "wb") as fw:
                fw.write(train_zip.read())

            import zipfile as zf_mod
            with zf_mod.ZipFile(zip_path, "r") as z:
                z.extractall(tmpdir)

            data_root = tmpdir
            subdirs = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d)) and not d.startswith("__") and not d.startswith(".")]
            if len(subdirs) == 1:
                inner = os.path.join(data_root, subdirs[0])
                inner_subs = [d for d in os.listdir(inner) if os.path.isdir(os.path.join(inner, d)) and not d.startswith("__") and not d.startswith(".")]
                if len(inner_subs) > 1:
                    data_root = inner
                    subdirs = inner_subs

            categories = sorted([d for d in subdirs if not d.startswith(".")])
            img_counts = {}
            for cat in categories:
                cat_path = os.path.join(data_root, cat)
                imgs = [f for f in os.listdir(cat_path) if f.lower().endswith((".jpg",".jpeg",".png",".bmp",".webp"))]
                img_counts[cat] = len(imgs)

            st.success(f"Dataset: **{len(categories)}** kategori, **{sum(img_counts.values())}** gambar")

            if HAS_PD:
                df_cats = pd.DataFrame({"Kategori": categories, "Jumlah": [img_counts[c] for c in categories]})
                st.dataframe(df_cats, use_container_width=True)

            st.markdown("### \u2699\ufe0f Parameter")
            c1, c2, c3 = st.columns(3)
            with c1:
                base_model = st.selectbox("Base Model:", list(PRETRAINED_MODELS.keys()), key="train_base")
            with c2:
                epochs = st.slider("Epochs:", 1, 50, 10)
                batch_size = st.selectbox("Batch Size:", [8, 16, 32, 64], index=1)
            with c3:
                lr = st.select_slider("Learning Rate:", [0.0001, 0.0005, 0.001, 0.005, 0.01], value=0.001)
                freeze = st.checkbox("Freeze base", value=True)
            val_split = st.slider("Validation Split:", 0.1, 0.4, 0.2, 0.05)
            augment = st.checkbox("Data Augmentation", value=True)

            if st.button("\U0001f3cb\ufe0f Mulai Training", type="primary", key="train_btn"):
                info = PRETRAINED_MODELS[base_model]
                target_size = info["size"]

                st.markdown("### \U0001f3c3 Training...")
                prog_bar = st.progress(0)
                status_text = st.empty()

                status_text.text("Mempersiapkan data...")
                if augment:
                    train_gen = keras.preprocessing.image.ImageDataGenerator(
                        preprocessing_function=info["preprocess"],
                        validation_split=val_split,
                        rotation_range=30, width_shift_range=0.2, height_shift_range=0.2,
                        shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode="nearest"
                    )
                else:
                    train_gen = keras.preprocessing.image.ImageDataGenerator(
                        preprocessing_function=info["preprocess"], validation_split=val_split
                    )

                train_data = train_gen.flow_from_directory(
                    data_root, target_size=target_size, batch_size=batch_size,
                    class_mode="categorical", subset="training", shuffle=True
                )
                val_data = train_gen.flow_from_directory(
                    data_root, target_size=target_size, batch_size=batch_size,
                    class_mode="categorical", subset="validation", shuffle=False
                )

                num_classes = len(train_data.class_indices)
                class_names = list(train_data.class_indices.keys())

                status_text.text("Membangun model...")
                model = build_transfer_model(base_model, num_classes, freeze_base=freeze)
                model.compile(optimizer=Adam(learning_rate=lr), loss="categorical_crossentropy", metrics=["accuracy"])

                class StreamlitCB(keras.callbacks.Callback):
                    def on_epoch_end(self, epoch, logs=None):
                        prog_bar.progress((epoch + 1) / epochs)
                        acc = logs.get("accuracy", 0)
                        val_acc = logs.get("val_accuracy", 0)
                        status_text.text(f"Epoch {epoch+1}/{epochs} — acc: {acc:.4f} — val_acc: {val_acc:.4f}")
                        st.session_state.training_log.append({
                            "epoch": epoch+1, "accuracy": acc, "loss": logs.get("loss",0),
                            "val_accuracy": val_acc, "val_loss": logs.get("val_loss",0)
                        })

                status_text.text("Training dimulai...")
                early_stop = EarlyStopping(patience=5, restore_best_weights=True, monitor="val_loss")
                history = model.fit(
                    train_data, validation_data=val_data, epochs=epochs,
                    callbacks=[StreamlitCB(), early_stop], verbose=0
                )

                prog_bar.progress(1.0)
                status_text.text("Selesai!")

                final_acc = history.history["accuracy"][-1]
                final_val_acc = history.history.get("val_accuracy", [0])[-1]

                st.markdown(f"""
                <div class="result-card">
                    <h2>\u2705 Training Selesai!</h2>
                    <h3>Accuracy: {final_acc*100:.1f}% | Val: {final_val_acc*100:.1f}%</h3>
                </div>
                """, unsafe_allow_html=True)

                if HAS_PLOTLY:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(y=history.history["accuracy"], name="Train Acc", mode="lines+markers"))
                    fig.add_trace(go.Scatter(y=history.history.get("val_accuracy",[]), name="Val Acc", mode="lines+markers"))
                    fig.update_layout(title="Training Curves", yaxis_title="Accuracy", height=400)
                    st.plotly_chart(fig, use_container_width=True)

                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(y=history.history["loss"], name="Train Loss", mode="lines+markers"))
                    fig2.add_trace(go.Scatter(y=history.history.get("val_loss",[]), name="Val Loss", mode="lines+markers"))
                    fig2.update_layout(title="Loss Curves", yaxis_title="Loss", height=400)
                    st.plotly_chart(fig2, use_container_width=True)

                with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
                    model.save(tmp.name)
                    with open(tmp.name, "rb") as mf:
                        model_bytes = mf.read()

                labels_str = "\n".join(class_names)
                c1, c2 = st.columns(2)
                with c1:
                    st.download_button("\U0001f4be Download Model (.h5)", model_bytes, "trained_model.h5", use_container_width=True)
                with c2:
                    st.download_button("\U0001f4be Download Labels (.txt)", labels_str, "labels.txt", "text/plain", use_container_width=True)

                st.session_state.custom_model = model
                st.session_state.custom_categories = class_names
                st.markdown('<div class="success-box">Model tersimpan di session! Pilih "Custom Model" di sidebar untuk langsung menggunakannya.</div>', unsafe_allow_html=True)

# ===================== TAB 4: HISTORY =====================
with tab4:
    st.markdown("## \U0001f4c8 History & Analisis")

    if st.session_state.classification_history:
        history_data = st.session_state.classification_history

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f'<div class="metric-card"><h4>Total</h4><h3>{len(history_data)}</h3></div>', unsafe_allow_html=True)
        avg_conf = np.mean([h["confidence"] for h in history_data])
        with c2:
            st.markdown(f'<div class="metric-card"><h4>Avg Confidence</h4><h3>{avg_conf*100:.1f}%</h3></div>', unsafe_allow_html=True)
        avg_time = np.mean([h["elapsed"] for h in history_data])
        with c3:
            st.markdown(f'<div class="metric-card"><h4>Avg Time</h4><h3>{avg_time:.2f}s</h3></div>', unsafe_allow_html=True)
        unique_cats = len(set(h["top_prediction"] for h in history_data))
        with c4:
            st.markdown(f'<div class="metric-card"><h4>Categories</h4><h3>{unique_cats}</h3></div>', unsafe_allow_html=True)

        if HAS_PD:
            df_hist = pd.DataFrame([{
                "Waktu": h["timestamp"], "Model": h["model"],
                "Prediksi": h["top_prediction"], "Confidence": f"{h['confidence']*100:.1f}%",
                "Waktu Proses": f"{h['elapsed']:.2f}s"
            } for h in history_data])
            st.dataframe(df_hist, use_container_width=True)

        if HAS_PLOTLY:
            cats = [h["top_prediction"] for h in history_data]
            cat_counts = {}
            for c in cats:
                cat_counts[c] = cat_counts.get(c, 0) + 1
            c1, c2 = st.columns(2)
            with c1:
                fig = px.bar(x=list(cat_counts.keys()), y=list(cat_counts.values()),
                    title="Distribusi Prediksi", labels={"x":"Kategori","y":"Jumlah"})
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                confs = [h["confidence"]*100 for h in history_data]
                fig = px.histogram(x=confs, nbins=20, title="Distribusi Confidence", labels={"x":"Confidence (%)"})
                st.plotly_chart(fig, use_container_width=True)

        if HAS_PD:
            csv_buf = io.StringIO()
            df_hist.to_csv(csv_buf, index=False)
            st.download_button("\U0001f4be Download CSV", csv_buf.getvalue(), "history.csv", "text/csv", use_container_width=True)
    else:
        st.info("Belum ada history. Mulai klasifikasi gambar di tab pertama!")

# ===================== TAB 5: PANDUAN =====================
with tab5:
    st.markdown("## \u2139\ufe0f Panduan Penggunaan")
    st.markdown("""
    ### \U0001f680 Quick Start
    1. **Upload gambar** di tab Klasifikasi
    2. Klik **Klasifikasi Sekarang!**
    3. Lihat hasil prediksi + confidence score

    ### \U0001f916 Model yang Tersedia

    | Model | Ukuran | Kecepatan | Akurasi | Cocok Untuk |
    |-------|--------|-----------|---------|-------------|
    | MobileNetV2 | 14 MB | \u26a1\u26a1\u26a1 | \u2b50\u2b50\u2b50 | Mobile, real-time |
    | EfficientNetB0 | 29 MB | \u26a1\u26a1\u26a1 | \u2b50\u2b50\u2b50\u2b50 | Keseimbangan |
    | ResNet50 | 98 MB | \u26a1\u26a1 | \u2b50\u2b50\u2b50\u2b50 | Klasifikasi umum |
    | InceptionV3 | 92 MB | \u26a1\u26a1 | \u2b50\u2b50\u2b50\u2b50 | Detail tinggi |
    | DenseNet121 | 33 MB | \u26a1\u26a1 | \u2b50\u2b50\u2b50\u2b50 | Fitur padat |
    | VGG16 | 528 MB | \u26a1 | \u2b50\u2b50\u2b50 | Pembelajaran |

    ### \U0001f3af Mode Klasifikasi

    **1. ImageNet (1000 Kategori)** — langsung pakai, mengenali 1000 objek umum

    **2. Preset Kategori** — template siap pakai:
    - \U0001f338 Bunga: Daisy, Dandelion, Rose, Sunflower, Tulip
    - \U0001f9b4 Tulang Belakang: Normal, Kifosis, Lordosis, Skoliosis
    - \U0001f3e5 X-Ray Paru: Normal, Pneumonia, COVID-19, TB
    - \U0001f431 Hewan, \U0001f34e Buah, \U0001f697 Kendaraan, \u270d\ufe0f Digit

    **3. Custom Model** — upload model `.h5` sendiri atau latih model baru

    ### \U0001f3eb Training Custom Model
    1. Siapkan dataset dalam format folder:
    ```
    dataset/
    \u251c\u2500\u2500 kategori_1/ (img1.jpg, img2.jpg, ...)
    \u251c\u2500\u2500 kategori_2/ (img1.jpg, img2.jpg, ...)
    ```
    2. ZIP folder tersebut
    3. Upload di tab Training \u2192 pilih parameter \u2192 Train!
    4. Download model `.h5` + `labels.txt`

    ### \U0001f525 Grad-CAM
    Menunjukkan **area gambar** yang paling berpengaruh terhadap keputusan AI.
    - **Merah/Kuning** = area penting
    - **Biru/Hijau** = area kurang penting

    ### \U0001f4a1 Tips
    - Gunakan gambar resolusi cukup (min 224x224 px)
    - Objek harus terlihat jelas
    - Untuk training: semakin banyak data = semakin akurat
    - Aktifkan **Data Augmentation** untuk generalisasi lebih baik
    """)

st.markdown("---")
st.markdown('<div style="text-align:center;color:#888;font-size:0.85rem">\U0001f9e0 AI Image Classifier \u2022 TensorFlow/Keras \u2022 Transfer Learning \u2022 Grad-CAM</div>', unsafe_allow_html=True)
