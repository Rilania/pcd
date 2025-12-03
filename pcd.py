import streamlit as st
import numpy as np
from PIL import Image
import io

# Import TFLite Runtime (pengganti TensorFlow Lite)
from tflite_runtime.interpreter import Interpreter

# Konfigurasi halaman
st.set_page_config(
    page_title="AI Image Detector",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# CSS
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

# Load model TFLite
@st.cache_resource
def load_tflite_model(model_path):
    try:
        interpreter = Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Preprocessing
def preprocess_image(image, target_size=(224, 224)):
    img = image.resize(target_size)
    img_array = np.array(img).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Prediksi
def predict_image(interpreter, image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    processed = preprocess_image(image, tuple(input_details[0]['shape'][1:3]))
    interpreter.set_tensor(input_details[0]['index'], processed)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data[0]

# UI
st.markdown("<h1 style='color:white;text-align:center;'>🔍 AI Image Detector</h1>",
            unsafe_allow_html=True)

model_path = "model_ai_vs_asli.tflite"

uploaded_file = st.file_uploader("", type=['png', 'jpg', 'jpeg'])

if uploaded_file:
    image = Image.open(uploaded_file)
    if image.mode == "RGBA":
        image = image.convert("RGB")

    st.image(image, caption="Gambar yang diupload", use_container_width=True)

    if st.button("🚀 Analisis Gambar"):
        with st.spinner("Menganalisis..."):
            interpreter = load_tflite_model(model_path)

            if interpreter:
                pred = predict_image(interpreter, image)

                # Jika output 1 neuron (sigmoid)
                if len(pred.shape) == 0 or pred.shape[0] == 1:
                    ai_prob = float(pred) * 100
                    real_prob = 100 - ai_prob
                else:
                    ai_prob = float(pred[0]) * 100
                    real_prob = float(pred[1]) * 100

                if ai_prob > real_prob:
                    label = "🤖 AI-Generated"
                    confidence = ai_prob
                    color = "#ff6b6b"
                else:
                    label = "📸 Real Photo"
                    confidence = real_prob
                    color = "#51cf66"

                st.markdown(
                    f"<h2 style='text-align:center;color:{color}'>{label}</h2>",
                    unsafe_allow_html=True
                )

                st.markdown(
                    f"<h3 style='text-align:center;'>Confidence: {confidence:.2f}%</h3>",
                    unsafe_allow_html=True
                )

                st.progress(confidence / 100)

else:
    st.info("Upload gambar terlebih dahulu.")
