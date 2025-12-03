import streamlit as st
import numpy as np
from PIL import Image

# GUNAKAN tflite-runtime (lebih ringan, cocok untuk Streamlit Cloud)
from tflite_runtime.interpreter import Interpreter


# ============================
#  PAGE CONFIG
# ============================
st.set_page_config(
    page_title="AI Image Detector",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ============================
#  CUSTOM CSS
# ============================
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    div[data-testid="stFileUploader"] {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 20px;
        backdrop-filter: blur(10px);
    }
    .upload-text {
        color: white;
        text-align: center;
        font-size: 1.2em;
        margin-bottom: 10px;
    }
    .result-box {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 30px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        margin-top: 20px;
    }
    h1 {
        color: white;
        text-align: center;
        font-size: 3em;
        margin-bottom: 10px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .subtitle {
        color: rgba(255, 255, 255, 0.9);
        text-align: center;
        font-size: 1.2em;
        margin-bottom: 30px;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 15px;
        border-radius: 10px;
        font-size: 1.1em;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
    }
</style>
""", unsafe_allow_html=True)


# ============================
# LOAD TFLITE MODEL
# ============================
@st.cache_resource
def load_tflite_model(model_path):
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


# ============================
# IMAGE PREPROCESSING
# ============================
def preprocess_image(image, target_size=(224, 224)):
    img = image.resize(target_size)
    img = img.convert("RGB")

    img_array = np.array(img).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


# ============================
# TFLITE PREDICTION
# ============================
def predict(interpreter, image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Resize sesuai model input
    h, w = input_details[0]["shape"][1:3]

    processed = preprocess_image(image, (h, w))

    interpreter.set_tensor(input_details[0]["index"], processed)
    interpreter.invoke()

    result = interpreter.get_tensor(output_details[0]["index"])
    return result[0]


# ============================
#        UI
# ============================
st.markdown("<h1>🔍 AI Image Detector</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Deteksi apakah gambar dibuat oleh AI atau asli</p>", unsafe_allow_html=True)

uploaded = st.file_uploader("", type=["png", "jpg", "jpeg"])

model_path = "model_ai_vs_asli.tflite"  # Pastikan nama file benar


# ============================
#        LOGIC
# ============================
if uploaded:
    img = Image.open(uploaded)

    # Center image
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(img, use_container_width=True)

    if st.button("🚀 Analisis Gambar"):
        with st.spinner("Menganalisis..."):
            interpreter = load_tflite_model(model_path)
            pred = predict(interpreter, img)

            # Case: sigmoid (single neuron)
            if pred.shape == () or pred.shape[0] == 1:
                ai = float(pred) * 100
                real = 100 - ai
            else:
                ai = float(pred[0]) * 100
                real = float(pred[1]) * 100

            label = "🤖 AI-Generated" if ai > real else "📸 Real Photo"
            conf = max(ai, real)
            color = "#ff6b6b" if ai > real else "#51cf66"

            st.markdown(f"""
            <div class='result-box'>
                <h2 style='text-align:center; color:{color};'>{label}</h2>
                <h3 style='text-align:center;'>Confidence: {conf:.2f}%</h3>
            </div>
            """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div style='background: rgba(255,255,255,0.15); padding:25px; border-radius:15px; margin-top:20px;'>
    <h3 style='text-align:center; color:white;'>✨ Cara Penggunaan</h3>
    <p style='color:white; text-align:center;'>
        Upload gambar lalu klik "Analisis Gambar" untuk mendeteksi apakah gambar AI atau asli.
    </p>
    </div>
    """, unsafe_allow_html=True)
