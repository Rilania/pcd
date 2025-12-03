import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Konfigurasi halaman
st.set_page_config(
    page_title="AI Image Detector",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS untuk styling
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
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
    .metric-container {
        display: flex;
        justify-content: space-around;
        margin-top: 20px;
    }
    .metric-box {
        text-align: center;
        padding: 15px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        color: white;
        min-width: 150px;
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

# Fungsi untuk load model TFLite
@st.cache_resource
def load_tflite_model(model_path):
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Fungsi preprocessing gambar
def preprocess_image(image, target_size=(224, 224)):
    """
    Preprocessing gambar untuk model
    Sesuaikan target_size dengan input model kamu
    """
    # Resize gambar
    img = image.resize(target_size)
    
    # Convert ke array
    img_array = np.array(img)
    
    # Normalisasi (0-1)
    img_array = img_array.astype('float32') / 255.0
    
    # Tambah batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# Fungsi prediksi
def predict_image(interpreter, image):
    """
    Melakukan prediksi menggunakan TFLite model
    """
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Preprocessing
    processed_img = preprocess_image(image, 
                                     tuple(input_details[0]['shape'][1:3]))
    
    # Set tensor input
    interpreter.set_tensor(input_details[0]['index'], processed_img)
    
    # Run inference
    interpreter.invoke()
    
    # Get output
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    return output_data[0]

# Header
st.markdown("<h1>🔍 AI Image Detector</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Deteksi apakah gambar dibuat oleh AI atau asli</p>", 
            unsafe_allow_html=True)

# Upload model TFLite
model_path = "model_ai_vs_asli.tflite"  # Ganti dengan path model kamu

# File uploader
st.markdown("<p class='upload-text'>📤 Upload gambar untuk dianalisis</p>", 
            unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "",
    type=['png', 'jpg', 'jpeg'],
    help="Upload gambar dalam format PNG, JPG, atau JPEG"
)

if uploaded_file is not None:
    # Load dan tampilkan gambar
    image = Image.open(uploaded_file)
    
    # Convert RGBA ke RGB jika perlu
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(image, caption="Gambar yang diupload", use_container_width=True)
    
    # Tombol analisis
    if st.button("🚀 Analisis Gambar"):
        with st.spinner("Menganalisis gambar..."):
            try:
                # Load model
                interpreter = load_tflite_model(model_path)
                
                if interpreter is not None:
                    # Prediksi
                    prediction = predict_image(interpreter, image)
                    
                    # Interpretasi hasil (sesuaikan dengan model kamu)
                    # Asumsi: output[0] = probabilitas AI-generated
                    # Asumsi: output[1] = probabilitas Real (jika binary classification)
                    
                    # Untuk single output (sigmoid)
                    if len(prediction.shape) == 0 or prediction.shape[0] == 1:
                        ai_prob = float(prediction) * 100
                        real_prob = 100 - ai_prob
                    # Untuk dual output (softmax)
                    else:
                        ai_prob = float(prediction[0]) * 100
                        real_prob = float(prediction[1]) * 100
                    
                    # Tentukan label
                    if ai_prob > real_prob:
                        label = "🤖 AI-Generated"
                        confidence = ai_prob
                        color = "#ff6b6b"
                    else:
                        label = "📸 Real Photo"
                        confidence = real_prob
                        color = "#51cf66"
                    
                    # Tampilkan hasil
                    st.markdown("<div class='result-box'>", unsafe_allow_html=True)
                    
                    st.markdown(f"<h2 style='text-align: center; color: {color};'>{label}</h2>", 
                                unsafe_allow_html=True)
                    
                    st.markdown(f"<h3 style='text-align: center;'>Confidence: {confidence:.2f}%</h3>", 
                                unsafe_allow_html=True)
                    
                    # Progress bar
                    st.progress(confidence / 100)
                    
                    # Detail probabilitas
                    st.markdown("### 📊 Detail Analisis")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"""
                        <div style='background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%); 
                                    padding: 20px; border-radius: 10px; text-align: center; color: white;'>
                            <h4>🤖 AI-Generated</h4>
                            <h2>{ai_prob:.2f}%</h2>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div style='background: linear-gradient(135deg, #51cf66 0%, #37b24d 100%); 
                                    padding: 20px; border-radius: 10px; text-align: center; color: white;'>
                            <h4>📸 Real Photo</h4>
                            <h2>{real_prob:.2f}%</h2>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Info tambahan
                    with st.expander("ℹ️ Informasi"):
                        st.write(f"**Ukuran gambar:** {image.size[0]} x {image.size[1]} pixels")
                        st.write(f"**Format:** {image.format}")
                        st.write(f"**Mode:** {image.mode}")
                
            except Exception as e:
                st.error(f"❌ Terjadi kesalahan: {str(e)}")
                st.info("Pastikan file model.tflite ada di direktori yang sama dengan app ini")

else:
    # Tampilan awal
    st.markdown("""
    <div style='background: rgba(255, 255, 255, 0.1); border-radius: 15px; 
                padding: 30px; margin-top: 30px; backdrop-filter: blur(10px);'>
        <h3 style='color: white; text-align: center;'>✨ Cara Penggunaan</h3>
        <ol style='color: rgba(255, 255, 255, 0.9); font-size: 1.1em;'>
            <li>Upload gambar menggunakan tombol di atas</li>
            <li>Klik tombol "Analisis Gambar"</li>
            <li>Tunggu hasil analisis muncul</li>
            <li>Lihat apakah gambar dibuat oleh AI atau asli</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; color: rgba(255, 255, 255, 0.7); margin-top: 50px;'>
    <p>Powered by TensorFlow Lite & Streamlit</p>
</div>
""", unsafe_allow_html=True)