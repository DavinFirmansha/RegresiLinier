import streamlit as st
import numpy as np
import io, os, json, time, tempfile, base64
from datetime import datetime
from PIL import Image, ImageOps

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
# PAGE CONFIG & STYLE
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
.success-box{background:#D5F5E3;border-left:5px solid #27AE60;padding:1rem;border-radius:5px;margin:.5rem 0}
.success-box,.success-box *{color:#145a32!important}
</style>""", unsafe_allow_html=True)

st.markdown('<div class="main-header">\U0001f9e0 AI Image Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Upload gambar \u2192 AI mengenali kategorinya secara otomatis</div>', unsafe_allow_html=True)

if not HAS_TF:
    st.error("\u274c **TensorFlow belum terinstall!** Tambahkan `tensorflow` ke `requirements.txt`")
    st.stop()

# ============================================================
# MODEL REGISTRY
# ============================================================
PRETRAINED_MODELS = {
    "MobileNetV2 (Ringan & Cepat)": {
        "class": MobileNetV2, "preprocess": mobilenet_v2.preprocess_input,
        "decode": mobilenet_v2.decode_predictions, "size": (224, 224),
        "desc": "Model ringan, cocok untuk deployment cepat"
    },
    "ResNet50 (Akurasi Tinggi)": {
        "class": ResNet50, "preprocess": resnet50.preprocess_input,
        "decode": resnet50.decode_predictions, "size": (224, 224),
        "desc": "Akurasi tinggi untuk klasifikasi umum"
    },
    "EfficientNetB0 (Efisien)": {
        "class": EfficientNetB0, "preprocess": efficientnet.preprocess_input,
        "decode": efficientnet.decode_predictions, "size": (224, 224),
        "desc": "Keseimbangan terbaik akurasi dan kecepatan"
    },
    "InceptionV3 (Detail Tinggi)": {
        "class": InceptionV3, "preprocess": inception_v3.preprocess_input,
        "decode": inception_v3.decode_predictions, "size": (299, 299),
        "desc": "Bagus untuk pola kompleks"
    },
    "VGG16 (Klasik)": {
        "class": VGG16, "preprocess": vgg16.preprocess_input,
        "decode": vgg16.decode_predictions, "size": (224, 224),
        "desc": "Arsitektur klasik"
    },
    "DenseNet121 (Padat)": {
        "class": DenseNet121, "preprocess": densenet.preprocess_input,
        "decode": densenet.decode_predictions, "size": (224, 224),
        "desc": "Koneksi padat antar layer"
    },
}

CLASSIFICATION_PRESETS = {
    "\U0001f338 Jenis Bunga": {
        "categories": ["Daisy", "Dandelion", "Rose", "Sunflower", "Tulip"],
        "description": "Klasifikasi 5 jenis bunga populer",
    },
    "\U0001f9b4 Postur Tulang Belakang": {
        "categories": ["Normal", "Kyphosis (Kifosis/Bungkuk)", "Lordosis (Lordosis)", "Scoliosis (Skoliosis)"],
        "description": "Deteksi kelainan tulang belakang",
    },
    "\U0001f3e5 Penyakit Paru (X-Ray)": {
        "categories": ["Normal", "Pneumonia", "COVID-19", "Tuberculosis"],
        "description": "Klasifikasi X-Ray paru-paru",
    },
    "\U0001f431 Hewan Peliharaan": {
        "categories": ["Dog", "Cat", "Bird", "Fish", "Hamster", "Rabbit"],
        "description": "Mengenali jenis hewan",
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
        "description": "Mengenali angka tulisan tangan",
    },
    "\U0001f3d7\ufe0f Custom (Buat Sendiri)": {
        "categories": [],
        "description": "Buat kategori sendiri",
    },
}

# ============================================================
# SESSION STATE
# ============================================================
defaults = {
    "custom_model": None, "custom_categories": [],
    "classification_history": [], "training_log": [],
    "trained_model_bytes": None, "trained_labels_str": None,
    "trained_history": None, "trained_acc": 0, "trained_val_acc": 0,
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
    return info["class"](weights="imagenet")

def preprocess_image(img_pil, target_size):
    img = img_pil.convert("RGB").resize(target_size)
    arr = np.array(img).astype(np.float32)
    return np.expand_dims(arr, axis=0)

def classify_imagenet(img_pil, model_name, top_k=10):
    info = PRETRAINED_MODELS[model_name]
    model = load_pretrained_model(model_name)
    arr = preprocess_image(img_pil, info["size"])
    arr = info["preprocess"](arr)
    preds = model.predict(arr, verbose=0)
    decoded = info["decode"](preds, top=top_k)
    return [{"class_id": cid, "label": lbl.replace("_"," ").title(), "confidence": float(sc)} for cid, lbl, sc in decoded[0]]

def build_transfer_model(base_model_name, num_classes, freeze_base=True):
    info = PRETRAINED_MODELS[base_model_name]
    base = info["class"](weights="imagenet", include_top=False, input_shape=(*info["size"], 3))
    if freeze_base:
        for layer in base.layers:
            layer.trainable = False
    x = GlobalAveragePooling2D()(base.output)
    x = Dropout(0.3)(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.2)(x)
    output = Dense(num_classes, activation="softmax")(x)
    m = Model(inputs=base.input, outputs=output)
    m.compile(optimizer=Adam(0.001), loss="categorical_crossentropy", metrics=["accuracy"])
    return m

def generate_grad_cam(model, img_array, class_index):
    last_conv = None
    for layer in reversed(model.layers):
        if len(layer.output_shape) == 4:
            last_conv = layer.name
            break
    if last_conv is None:
        return None
    grad_model = Model(inputs=model.input, outputs=[model.get_layer(last_conv).output, model.output])
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array)
        loss = preds[:, class_index]
    grads = tape.gradient(loss, conv_out)
    if grads is None:
        return None
    pooled = tf.reduce_mean(grads, axis=(0,1,2))
    heatmap = tf.squeeze(conv_out[0] @ pooled[..., tf.newaxis])
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()

def overlay_heatmap(img_pil, heatmap, alpha=0.4):
    if heatmap is None:
        return img_pil
    img = np.array(img_pil.convert("RGB").resize((224, 224)))
    hm = np.array(Image.fromarray((heatmap * 255).astype(np.uint8)).resize((224, 224)))
    if HAS_CV2:
        colored = cv2.cvtColor(cv2.applyColorMap(hm, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    else:
        colored = np.stack([hm, np.zeros_like(hm), 255 - hm], axis=2)
    return Image.fromarray(np.clip(img * (1 - alpha) + colored * alpha, 0, 255).astype(np.uint8))

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## \u2699\ufe0f Pengaturan")
    model_choice = st.selectbox("Model AI:", list(PRETRAINED_MODELS.keys()), key="model_select")
    st.caption(PRETRAINED_MODELS[model_choice]["desc"])

    st.markdown("### \U0001f3af Mode")
    class_mode = st.radio("Mode:", ["ImageNet (1000 Kategori)", "Preset Kategori", "Custom Model (.h5)"], key="class_mode")

    preset_choice = None
    custom_cats = []
    if class_mode == "Preset Kategori":
        preset_choice = st.selectbox("Preset:", list(CLASSIFICATION_PRESETS.keys()), key="preset_sel")
        preset = CLASSIFICATION_PRESETS[preset_choice]
        st.caption(preset["description"])
        if preset_choice == "\U0001f3d7\ufe0f Custom (Buat Sendiri)":
            cats_text = st.text_area("Kategori (per baris):", "Kucing\nAnjing\nBurung")
            custom_cats = [c.strip() for c in cats_text.strip().split("\n") if c.strip()]
        else:
            custom_cats = preset["categories"]
            for c in custom_cats:
                st.markdown(f"- {c}")
    elif class_mode == "Custom Model (.h5)":
        model_file = st.file_uploader("Model (.h5/.keras):", type=["h5","keras"], key="model_upload")
        labels_file = st.file_uploader("Labels (.txt/.json):", type=["txt","json"], key="labels_upload")
        if model_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp:
                tmp.write(model_file.read()); tmp_path = tmp.name
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
            st.success(f"{len(custom_cats)} labels")

    st.markdown("---")
    top_k = st.slider("Top-K Prediksi:", 1, 20, 5)
    show_gradcam = st.checkbox("Tampilkan Grad-CAM", value=True)
    confidence_threshold = st.slider("Confidence Min (%):", 0, 100, 5)

    st.markdown("---")
    if st.session_state.classification_history:
        st.caption(f"{len(st.session_state.classification_history)} klasifikasi")
        if st.button("\U0001f5d1\ufe0f Clear History"):
            st.session_state.classification_history = []
            st.rerun()

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "\U0001f50d Klasifikasi", "\U0001f4da Batch", "\U0001f3eb Training",
    "\U0001f4c8 History", "\u2139\ufe0f Panduan"
])

# ===================== TAB 1 =====================
with tab1:
    st.markdown("## \U0001f50d Klasifikasi Gambar")
    input_method = st.radio("Input:", ["Upload", "Kamera", "URL"], horizontal=True)

    img_input = None
    if input_method == "Upload":
        uploaded = st.file_uploader("Pilih gambar:", type=["png","jpg","jpeg","bmp","webp","tiff"], key="main_upload")
        if uploaded:
            img_input = Image.open(uploaded).convert("RGB")
    elif input_method == "Kamera":
        cam = st.camera_input("Ambil foto")
        if cam:
            img_input = Image.open(cam).convert("RGB")
    else:
        url = st.text_input("URL:", placeholder="https://example.com/image.jpg")
        if url:
            try:
                import urllib.request
                with urllib.request.urlopen(url) as resp:
                    img_input = Image.open(io.BytesIO(resp.read())).convert("RGB")
            except Exception as e:
                st.error(f"Gagal: {e}")

    if img_input:
        col_img, col_result = st.columns([1, 1])
        with col_img:
            st.markdown("### \U0001f5bc\ufe0f Gambar Input")
            st.image(img_input, width="stretch")
            st.caption(f"{img_input.width} x {img_input.height} px")

        with col_result:
            st.markdown("### \U0001f3af Hasil Klasifikasi")
            if st.button("\U0001f680 Klasifikasi Sekarang!", type="primary", use_container_width=True, key="cls_btn"):
                with st.spinner("Menganalisis..."):
                    t0 = time.time()
                    if class_mode == "ImageNet (1000 Kategori)":
                        results = classify_imagenet(img_input, model_choice, top_k=top_k)
                        results = [r for r in results if r["confidence"] >= confidence_threshold / 100]
                    elif class_mode == "Preset Kategori":
                        imagenet_res = classify_imagenet(img_input, model_choice, top_k=50)
                        results = []
                        for cat in custom_cats:
                            mx = 0
                            cat_lw = cat.lower().split("(")[0].strip()
                            for r in imagenet_res:
                                if cat_lw in r["label"].lower() or r["label"].lower() in cat_lw:
                                    mx = max(mx, r["confidence"])
                                if set(r["label"].lower().split()) & set(cat_lw.split()):
                                    mx = max(mx, r["confidence"])
                            results.append({"label": cat, "confidence": mx, "class_id": ""})
                        total = sum(r["confidence"] for r in results)
                        if total > 0:
                            for r in results:
                                r["confidence"] /= total
                        results.sort(key=lambda x: x["confidence"], reverse=True)
                        results = results[:top_k]
                    elif class_mode == "Custom Model (.h5)" and st.session_state.custom_model is not None:
                        cm = st.session_state.custom_model
                        arr = preprocess_image(img_input, cm.input_shape[1:3]) / 255.0
                        preds = cm.predict(arr, verbose=0)[0]
                        labels = st.session_state.custom_categories or [f"Class {i}" for i in range(len(preds))]
                        results = [{"label": labels[i] if i < len(labels) else f"Class {i}", "confidence": float(c), "class_id": str(i)} for i, c in enumerate(preds)]
                        results.sort(key=lambda x: x["confidence"], reverse=True)
                        results = results[:top_k]
                    else:
                        results = []
                    elapsed = time.time() - t0

                if results:
                    best = results[0]
                    st.markdown(f'<div class="result-card"><h2>\U0001f3af {best["label"]}</h2><h3>{best["confidence"]*100:.1f}% confidence</h3><p>{model_choice.split("(")[0].strip()} \u2022 {elapsed:.2f}s</p></div>', unsafe_allow_html=True)

                    st.markdown("#### Top Prediksi:")
                    for i, r in enumerate(results):
                        emoji = ["\U0001f947", "\U0001f948", "\U0001f949"][i] if i < 3 else f"#{i+1}"
                        st.progress(min(r["confidence"], 1.0), text=f"{emoji} {r['label']} \u2014 {r['confidence']*100:.1f}%")

                    if show_gradcam:
                        st.markdown("#### \U0001f525 Grad-CAM")
                        try:
                            mdl = load_pretrained_model(model_choice)
                            info = PRETRAINED_MODELS[model_choice]
                            arr = preprocess_image(img_input, info["size"])
                            arr_p = info["preprocess"](arr.copy())
                            top_cls = np.argmax(mdl.predict(arr_p, verbose=0)[0])
                            hm = generate_grad_cam(mdl, arr_p, top_cls)
                            if hm is not None:
                                cam_img = overlay_heatmap(img_input, hm)
                                gc1, gc2 = st.columns(2)
                                with gc1:
                                    st.image(img_input, caption="Original", width="stretch")
                                with gc2:
                                    st.image(cam_img, caption="Grad-CAM", width="stretch")
                                st.caption("Merah/kuning = area fokus AI")
                        except Exception as e:
                            st.warning(f"Grad-CAM error: {e}")

                    st.session_state.classification_history.append({
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "model": model_choice.split("(")[0].strip(),
                        "top_prediction": best["label"], "confidence": best["confidence"],
                        "all_predictions": results, "elapsed": elapsed,
                    })
                else:
                    st.warning("Tidak ada hasil.")

# ===================== TAB 2: BATCH =====================
with tab2:
    st.markdown("## \U0001f4da Batch Klasifikasi")
    batch_files = st.file_uploader("Upload gambar:", type=["png","jpg","jpeg","bmp","webp"],
                                    accept_multiple_files=True, key="batch_upload")
    if batch_files:
        st.markdown(f"**{len(batch_files)} gambar**")
        if st.button("\U0001f680 Klasifikasi Semua", type="primary", key="batch_btn"):
            all_results = []
            prog = st.progress(0)
            for idx, f in enumerate(batch_files):
                img = Image.open(f).convert("RGB")
                res = classify_imagenet(img, model_choice, top_k=top_k)
                if res:
                    all_results.append({"filename": f.name, "prediction": res[0]["label"],
                        "confidence": f"{res[0]['confidence']*100:.1f}%",
                        "top_3": " | ".join([f"{r['label']} ({r['confidence']*100:.1f}%)" for r in res[:3]])})
                prog.progress((idx + 1) / len(batch_files))
            prog.empty()

            if all_results and HAS_PD:
                df = pd.DataFrame(all_results)
                st.dataframe(df, use_container_width=True)
                pred_counts = df["prediction"].value_counts()
                c1, c2 = st.columns(2)
                with c1:
                    for cat, cnt in pred_counts.items():
                        st.markdown(f"- **{cat}**: {cnt} ({cnt/len(df)*100:.0f}%)")
                with c2:
                    if HAS_PLOTLY:
                        fig = px.pie(values=pred_counts.values, names=pred_counts.index, title="Distribusi")
                        st.plotly_chart(fig, use_container_width=True)
                csv_buf = io.StringIO(); df.to_csv(csv_buf, index=False)
                st.download_button("\U0001f4be Download CSV", csv_buf.getvalue(), "batch_results.csv", "text/csv", use_container_width=True)

            st.markdown("### Gallery")
            cols = st.columns(4)
            for idx, f in enumerate(batch_files):
                with cols[idx % 4]:
                    img = Image.open(f).convert("RGB")
                    cap = f"{all_results[idx]['prediction']} ({all_results[idx]['confidence']})" if idx < len(all_results) else f.name
                    st.image(img, caption=cap, width="stretch")

# ===================== TAB 3: TRAINING =====================
with tab3:
    st.markdown("## \U0001f3eb Training Custom Model")
    st.markdown("""<div class="info-box">
    <b>Cara:</b> Siapkan ZIP berisi sub-folder per kategori (misal: <code>daisy/*.jpg</code>, <code>rose/*.jpg</code>).
    Upload \u2192 pilih parameter \u2192 Train! \u2192 Download model.
    </div>""", unsafe_allow_html=True)

    train_zip = st.file_uploader("Dataset (ZIP):", type=["zip"], key="train_zip")

    if train_zip:
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = os.path.join(tmpdir, "dataset.zip")
            with open(zip_path, "wb") as fw:
                fw.write(train_zip.read())
            import zipfile as zf_mod
            with zf_mod.ZipFile(zip_path, "r") as z:
                z.extractall(tmpdir)

            data_root = tmpdir
            subdirs = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d)) and not d.startswith(("__","."))]
            if len(subdirs) == 1:
                inner = os.path.join(data_root, subdirs[0])
                inner_subs = [d for d in os.listdir(inner) if os.path.isdir(os.path.join(inner, d)) and not d.startswith(("__","."))]
                if len(inner_subs) > 1:
                    data_root = inner
                    subdirs = inner_subs

            categories = sorted(subdirs)
            img_counts = {}
            for cat in categories:
                cat_path = os.path.join(data_root, cat)
                img_counts[cat] = len([f for f in os.listdir(cat_path) if f.lower().endswith((".jpg",".jpeg",".png",".bmp",".webp"))])

            st.success(f"**{len(categories)}** kategori, **{sum(img_counts.values())}** gambar")
            if HAS_PD:
                st.dataframe(pd.DataFrame({"Kategori": categories, "Jumlah": [img_counts[c] for c in categories]}), use_container_width=True)

            c1, c2, c3 = st.columns(3)
            with c1:
                base_model = st.selectbox("Base Model:", list(PRETRAINED_MODELS.keys()), key="train_base")
            with c2:
                epochs = st.slider("Epochs:", 1, 50, 10)
                batch_size = st.selectbox("Batch:", [8, 16, 32], index=1)
            with c3:
                lr = st.select_slider("LR:", [0.0001, 0.0005, 0.001, 0.005], value=0.001)
                freeze = st.checkbox("Freeze base", value=True)
            val_split = st.slider("Val Split:", 0.1, 0.4, 0.2, 0.05)
            augment = st.checkbox("Augmentation", value=True)

            if st.button("\U0001f3cb\ufe0f Train!", type="primary", key="train_btn"):
                info = PRETRAINED_MODELS[base_model]
                target_size = info["size"]
                prog_bar = st.progress(0)
                status_text = st.empty()

                status_text.text("Mempersiapkan data...")
                if augment:
                    gen = keras.preprocessing.image.ImageDataGenerator(
                        preprocessing_function=info["preprocess"], validation_split=val_split,
                        rotation_range=30, width_shift_range=0.2, height_shift_range=0.2,
                        shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode="nearest")
                else:
                    gen = keras.preprocessing.image.ImageDataGenerator(
                        preprocessing_function=info["preprocess"], validation_split=val_split)

                train_data = gen.flow_from_directory(data_root, target_size=target_size, batch_size=batch_size,
                    class_mode="categorical", subset="training", shuffle=True)
                val_data = gen.flow_from_directory(data_root, target_size=target_size, batch_size=batch_size,
                    class_mode="categorical", subset="validation", shuffle=False)

                num_classes = len(train_data.class_indices)
                class_names = list(train_data.class_indices.keys())

                status_text.text("Building model...")
                model = build_transfer_model(base_model, num_classes, freeze)
                model.compile(optimizer=Adam(lr), loss="categorical_crossentropy", metrics=["accuracy"])

                class SLCB(keras.callbacks.Callback):
                    def on_epoch_end(self, epoch, logs=None):
                        prog_bar.progress((epoch + 1) / epochs)
                        status_text.text(f"Epoch {epoch+1}/{epochs} \u2014 acc: {logs.get('accuracy',0):.4f} \u2014 val_acc: {logs.get('val_accuracy',0):.4f}")

                history = model.fit(train_data, validation_data=val_data, epochs=epochs,
                    callbacks=[SLCB(), EarlyStopping(patience=5, restore_best_weights=True, monitor="val_loss")], verbose=0)

                prog_bar.progress(1.0)
                status_text.text("Selesai!")

                with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
                    model.save(tmp.name)
                    with open(tmp.name, "rb") as mf:
                        model_bytes = mf.read()

                labels_str = "\n".join(class_names)
                final_acc = history.history["accuracy"][-1]
                final_val = history.history.get("val_accuracy", [0])[-1]

                st.session_state.custom_model = model
                st.session_state.custom_categories = class_names
                st.session_state.trained_model_bytes = model_bytes
                st.session_state.trained_labels_str = labels_str
                st.session_state.trained_history = history.history
                st.session_state.trained_acc = final_acc
                st.session_state.trained_val_acc = final_val
                st.rerun()

    # PERSISTENT TRAINING RESULTS (survives download-button rerun)
    if st.session_state.trained_model_bytes is not None:
        st.markdown("---")
        st.markdown("### \u2705 Training Results")
        t_acc = st.session_state.trained_acc
        t_val = st.session_state.trained_val_acc
        st.markdown(f'<div class="result-card"><h2>\u2705 Selesai!</h2><h3>Acc: {t_acc*100:.1f}% | Val: {t_val*100:.1f}%</h3></div>', unsafe_allow_html=True)

        if HAS_PLOTLY and st.session_state.trained_history:
            th = st.session_state.trained_history
            c1, c2 = st.columns(2)
            with c1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=th.get("accuracy", []), name="Train Acc", mode="lines+markers"))
                fig.add_trace(go.Scatter(y=th.get("val_accuracy", []), name="Val Acc", mode="lines+markers"))
                fig.update_layout(title="Accuracy", height=350)
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(y=th.get("loss", []), name="Train Loss", mode="lines+markers"))
                fig2.add_trace(go.Scatter(y=th.get("val_loss", []), name="Val Loss", mode="lines+markers"))
                fig2.update_layout(title="Loss", height=350)
                st.plotly_chart(fig2, use_container_width=True)

        dl1, dl2, dl3 = st.columns(3)
        with dl1:
            st.download_button("\U0001f4be Download Model (.h5)", st.session_state.trained_model_bytes,
                "trained_model.h5", use_container_width=True, key="dl_model")
        with dl2:
            st.download_button("\U0001f4be Download Labels (.txt)", st.session_state.trained_labels_str,
                "labels.txt", "text/plain", use_container_width=True, key="dl_labels")
        with dl3:
            if st.button("\U0001f5d1\ufe0f Clear", key="clear_train"):
                st.session_state.trained_model_bytes = None
                st.session_state.trained_labels_str = None
                st.session_state.trained_history = None
                st.rerun()

        st.markdown('<div class="success-box">Model tersimpan! Pilih <b>Custom Model (.h5)</b> di sidebar untuk langsung pakai.</div>', unsafe_allow_html=True)

# ===================== TAB 4: HISTORY =====================
with tab4:
    st.markdown("## \U0001f4c8 History")
    if st.session_state.classification_history:
        hd = st.session_state.classification_history
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f'<div class="metric-card"><h4>Total</h4><h3>{len(hd)}</h3></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric-card"><h4>Avg Conf</h4><h3>{np.mean([h["confidence"] for h in hd])*100:.1f}%</h3></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="metric-card"><h4>Avg Time</h4><h3>{np.mean([h["elapsed"] for h in hd]):.2f}s</h3></div>', unsafe_allow_html=True)
        with c4:
            st.markdown(f'<div class="metric-card"><h4>Categories</h4><h3>{len(set(h["top_prediction"] for h in hd))}</h3></div>', unsafe_allow_html=True)

        if HAS_PD:
            df_h = pd.DataFrame([{"Waktu": h["timestamp"], "Model": h["model"], "Prediksi": h["top_prediction"],
                "Confidence": f"{h['confidence']*100:.1f}%", "Proses": f"{h['elapsed']:.2f}s"} for h in hd])
            st.dataframe(df_h, use_container_width=True)

        if HAS_PLOTLY:
            cats = [h["top_prediction"] for h in hd]
            cc = {}
            for c in cats:
                cc[c] = cc.get(c, 0) + 1
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(px.bar(x=list(cc.keys()), y=list(cc.values()), title="Distribusi", labels={"x":"Kategori","y":"Jumlah"}), use_container_width=True)
            with c2:
                st.plotly_chart(px.histogram(x=[h["confidence"]*100 for h in hd], nbins=20, title="Confidence", labels={"x":"Confidence (%)"}), use_container_width=True)

        if HAS_PD:
            csv_buf = io.StringIO(); df_h.to_csv(csv_buf, index=False)
            st.download_button("\U0001f4be CSV", csv_buf.getvalue(), "history.csv", "text/csv", use_container_width=True)
    else:
        st.info("Belum ada history.")

# ===================== TAB 5: PANDUAN =====================
with tab5:
    st.markdown("## \u2139\ufe0f Panduan")
    st.markdown("""
### \U0001f680 Quick Start
1. Upload gambar di tab **Klasifikasi**
2. Klik **Klasifikasi Sekarang!**
3. Lihat hasil prediksi + confidence

### \U0001f916 Model

| Model | Ukuran | Kecepatan | Akurasi | Cocok Untuk |
|-------|--------|-----------|---------|-------------|
| MobileNetV2 | 14 MB | \u26a1\u26a1\u26a1 | \u2b50\u2b50\u2b50 | Mobile, real-time |
| EfficientNetB0 | 29 MB | \u26a1\u26a1\u26a1 | \u2b50\u2b50\u2b50\u2b50 | Keseimbangan |
| ResNet50 | 98 MB | \u26a1\u26a1 | \u2b50\u2b50\u2b50\u2b50 | Klasifikasi umum |
| InceptionV3 | 92 MB | \u26a1\u26a1 | \u2b50\u2b50\u2b50\u2b50 | Detail tinggi |
| DenseNet121 | 33 MB | \u26a1\u26a1 | \u2b50\u2b50\u2b50\u2b50 | Fitur padat |
| VGG16 | 528 MB | \u26a1 | \u2b50\u2b50\u2b50 | Pembelajaran |

### \U0001f3af Mode Klasifikasi

**ImageNet** \u2014 1000 kategori, langsung pakai

**Preset** \u2014 Bunga, tulang belakang, X-Ray paru, hewan, buah, kendaraan, digit

**Custom Model** \u2014 Upload `.h5` atau latih di tab Training

### \U0001f3eb Training
1. ZIP folder dengan sub-folder per kategori
2. Upload \u2192 pilih parameter \u2192 Train
3. Download model `.h5` + `labels.txt`

### \U0001f525 Grad-CAM
Area **merah/kuning** = bagian gambar yang paling mempengaruhi keputusan AI
    """)

st.markdown("---")
st.markdown('<div style="text-align:center;color:#888;font-size:0.85rem">\U0001f9e0 AI Image Classifier \u2022 TensorFlow \u2022 Transfer Learning \u2022 Grad-CAM</div>', unsafe_allow_html=True)
