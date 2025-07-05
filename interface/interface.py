import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import pickle

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess

# ---------- Background Styling ----------
def set_custom_style():
    st.markdown(f"""
        <style>
        body {{
            background-color: #e3f2fd;
        }}
        .upload-box {{
            background-color: rgba(255, 255, 255, 0.15);
            border-radius: 12px;
            padding: 30px;
            margin: 20px 0;
            border: 2px dashed #90caf9;
            text-align: center;
            transition: border 0.3s ease-in-out;
        }}
       
        
        .file-info {{
            font-size: 14px;
            color: #555;
            margin-top: 10px;
        }}
        .stButton>button {{
            background-color: #1976d2;
            color: white;
            font-size: 16px;
            padding: 10px 25px;
            border-radius: 10px;
            transition: all 0.3s ease-in-out;
            margin-top: 20px;
            width: 100%;
        }}
        .stButton>button:hover {{
            background-color: #0d47a1;
            transform: scale(1.02);
        }}
        .prediction {{
            font-size: 22px;
            color: #1565c0;
            font-weight: 600;
            background-color: rgba(255, 255, 255, 0.75);
            padding: 20px;
            border-radius: 10px;
            border-left: 6px solid #1565c0;
            margin-top: 20px;
        }}
        .stImage img {{
            border: 3px solid #fff;
            border-radius: 10px;
            box-shadow: 0 8px 20px rgba(0,0,0,0.2);
        }}
        .confidence-bar {{
            height: 20px;
            border-radius: 10px;
            background: linear-gradient(90deg, #42a5f5 {50}%, #cfd8dc {50}%);
        }}
        </style>
    """, unsafe_allow_html=True)

set_custom_style()

# ---------- Title & Subtitle ----------
st.markdown('<h1 style="text-align: center;">🧪 Skin Disease Classifier</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle" style="text-align: center; font-size:18px; color:#333;">Upload a skin image to detect <b>Acne</b>, <b>Scabies</b>, or <b>Healthy</b> condition.</p>', unsafe_allow_html=True)

# ---------- Sidebar Settings ----------
st.sidebar.markdown("## ⚙️ Settings")
model_choice = st.sidebar.selectbox(
    "Choose a Model",
    ("MobileNet", "EfficientNetB0"),
    help="Select between MobileNetV2 or EfficientNetB0 for classification"
)

# ---------- Load Model ----------
@st.cache_resource
def load_model(model_name):
    if model_name == "MobileNet":
        model = tf.keras.models.load_model("model/skin_disease_model_mobilenetv2.keras")
        label_encoder = pickle.load(open('model/label_encoder1.pkl', 'rb'))
        preprocess_function = mobilenet_preprocess
    else:
        model = tf.keras.models.load_model("model/efficentnetb0_skindisease_model.keras")
        label_encoder = pickle.load(open('model/label_encoder2.pkl', 'rb'))
        preprocess_function = efficientnet_preprocess
    return model, label_encoder, preprocess_function

model, label_encoder, preprocess_function = load_model(model_choice)

# ---------- Preprocess Image ----------
def preprocess_image(img):
    img = img.convert("RGB")
    img = img.resize((224, 224))
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_function(img)
    return img

# ---------- Upload Section ----------
st.markdown('<div class="upload-box">', unsafe_allow_html=True)
st.markdown('### 📤 Drag and drop your image here')
uploaded_file = st.file_uploader(" ", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
st.markdown('<div class="file-info">Max 200MB • Supported: JPG, PNG, JPEG</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ---------- Main Logic ----------
if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        col1, col2 = st.columns([1, 1.4])

        with col1:
            st.image(image, caption="🖼️ Your Uploaded Image", use_container_width=True)

        with col2:
            if st.button("🔍 Analyze Image"):
                with st.spinner("Analyzing with AI..."):
                    input_img = preprocess_image(image)
                    preds = model.predict(input_img)
                    class_idx = np.argmax(preds)
                    confidence = np.max(preds)
                    label = label_encoder.inverse_transform([class_idx])[0]

                    # Prediction display
                    st.markdown(
                        f'<div class="prediction">'
                        f'<b>Model:</b> {model_choice}<br>'
                        f'<b>Predicted Condition:</b> <span style="color:#1b5e20;">{label}</span><br>'
                        f'<b>Confidence:</b> {confidence*100:.2f}%'
                        '</div>', unsafe_allow_html=True
                    )

                    # Confidence bar
                    st.markdown(f"""
                        <div style="margin: 10px 0;">
                            <div class="confidence-bar" style="width: 100%; --progress: {confidence*100:.0f}%;">
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

                    # Diagnosis Info
                    if label == "Acne":
                        st.info("💡 *Acne Tip:* Wash your face twice daily with a mild cleanser that contains salicylic acid.")
                    elif label == "Scabies":
                        st.warning("⚠️ *Scabies Alert:* Please consult a dermatologist. Use prescribed permethrin creams as directed.")
                    elif label == "Healthy":
                        st.success("🌟 *Great News:* Your skin appears healthy. Maintain your current skincare routine!")

    except Exception as e:
        st.error(f"Something went wrong while processing the image: {str(e)}")
