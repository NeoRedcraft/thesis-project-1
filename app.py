import os
os.environ["KERAS_BACKEND"] = "torch"

import streamlit as st
import numpy as np
from PIL import Image
import keras

# Set page configuration with a premium icon and title
st.set_page_config(
    page_title="FoodScan - Meal Classifier",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling using CSS injection
st.markdown("""
<style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Elegant Dark Mode Background & Sidebar */
    .stApp {
        background: linear-gradient(135deg, #101216 0%, #1a1d24 100%);
        color: #e2e8f0;
    }
    
    section[data-testid="stSidebar"] {
        background-color: #111317 !important;
        border-right: 1px solid #2d3139;
    }

    /* Cards and Glassmorphism Containers */
    .glass-card {
        background: rgba(30, 34, 42, 0.6);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        margin-bottom: 20px;
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        border-color: rgba(255, 255, 255, 0.1);
        transform: translateY(-2px);
    }
    
    /* Headings and Titles */
    .main-title {
        background: linear-gradient(90deg, #ff7e5f 0%, #feb47b 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 700;
        margin-bottom: 5px;
    }
    
    .subtitle {
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 25px;
        font-weight: 300;
    }

    /* Glow Cards for Results */
    .glow-card-beverage {
        background: rgba(13, 148, 136, 0.15);
        border: 1px solid rgba(13, 148, 136, 0.3);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 0 20px rgba(13, 148, 136, 0.1);
    }
    
    .glow-card-snack {
        background: rgba(244, 63, 94, 0.15);
        border: 1px solid rgba(244, 63, 94, 0.3);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 0 20px rgba(244, 63, 94, 0.1);
    }
    
    .glow-card-staple {
        background: rgba(245, 158, 11, 0.15);
        border: 1px solid rgba(245, 158, 11, 0.3);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 0 20px rgba(245, 158, 11, 0.1);
    }
    
    .glow-title {
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: 1px;
        margin: 5px 0;
    }

    /* Custom Progress Bar Styling */
    .progress-container {
        margin-bottom: 15px;
    }
    
    .progress-label-row {
        display: flex;
        justify-content: space-between;
        margin-bottom: 5px;
        font-size: 0.95rem;
        font-weight: 500;
    }
    
    .progress-bar-bg {
        background-color: #2a2e37;
        border-radius: 8px;
        height: 12px;
        width: 100%;
        overflow: hidden;
    }
    
    .progress-bar-fill {
        height: 100%;
        border-radius: 8px;
        transition: width 0.8s ease-in-out;
    }
    
    /* Color mapping for HSL variables */
    .fill-beverage { background: linear-gradient(90deg, #0d9488, #2dd4bf); }
    .fill-snack { background: linear-gradient(90deg, #e11d48, #fb7185); }
    .fill-staple { background: linear-gradient(90deg, #d97706, #fbbf24); }

    /* File uploader custom borders */
    [data-testid="stFileUploader"] {
        background: rgba(30, 34, 42, 0.4) !important;
        border: 2px dashed rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        padding: 15px !important;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #ff7e5f !important;
    }

    /* Sidebar buttons styling */
    .stButton>button {
        background: linear-gradient(90deg, #ff7e5f 0%, #feb47b 100%) !important;
        color: white !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 8px 20px !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton>button:hover {
        transform: scale(1.02) !important;
        box-shadow: 0 4px 15px rgba(255, 126, 95, 0.4) !important;
    }
    
    /* Divider Customization */
    hr {
        border-color: rgba(255, 255, 255, 0.08) !important;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- Deep Learning Model Management -----------------

# Lazy import tensorflow only when needed to prevent loading delays
@st.cache_resource
def load_keras_model(model_path):
    """Loads the compiled Keras model using the PyTorch backend."""
    import keras
    try:
        model = keras.models.load_model(model_path)
        return model, None
    except Exception as e:
        return None, str(e)

# Helper function to save uploaded model file locally
def save_uploaded_file(uploaded_file):
    try:
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return uploaded_file.name
    except Exception as e:
        st.error(f"Error saving model: {e}")
        return None

# Class names exactly matching the training notebook
CLASS_NAMES = ['BEVERAGE', 'SNACK', 'STAPLE']

# ----------------- Sidebar Configuration -----------------

st.sidebar.markdown("<h3 style='color: #ff7e5f; font-weight: 700; margin-bottom: 0;'>⚙️ Model Settings</h3>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='color: #94a3b8; font-size: 0.85rem; margin-top: 0;'>Manage the deployed Keras classifier</p>", unsafe_allow_html=True)
st.sidebar.markdown("---")

default_model_name = "mobilenetv2_meal_classifier.keras"
model_source = st.sidebar.radio(
    "Choose Model Source:",
    ("Use Pre-trained Model (Default)", "Upload Custom .keras Model")
)

model_path = None
uploaded_model = None

if model_source == "Use Pre-trained Model (Default)":
    if os.path.exists(default_model_name):
        model_path = default_model_name
        st.sidebar.success(f"✔ Found default model: `{default_model_name}`")
    else:
        st.sidebar.error(f"❌ Default model `{default_model_name}` not found in the root directory.")
        st.sidebar.info("Please select the 'Upload Custom' option or place the model file in the workspace.")
else:
    uploaded_model = st.sidebar.file_uploader(
        "Upload your .keras file", 
        type=["keras"],
        help="Provide a compiled food classification model trained on MobileNetV2"
    )
    if uploaded_model is not None:
        model_path = save_uploaded_file(uploaded_model)
        st.sidebar.success(f"✔ Uploaded model saved as: `{model_path}`")
    else:
        st.sidebar.warning("Waiting for model file upload...")

st.sidebar.markdown("---")
st.sidebar.markdown("<h4 style='color: #feb47b;'>📊 Model Architecture</h4>", unsafe_allow_html=True)
st.sidebar.markdown("""
- **Backbone**: MobileNetV2 (ImageNet)
- **Input Dimensions**: 224 x 224 x 3
- **Output Nodes**: 3 Categories
- **Internal Preprocessing**: Built-in `preprocess_input` layer mapping pixels from [0, 255] to [-1, 1].
""")

st.sidebar.markdown("---")
st.sidebar.markdown("<p style='text-align: center; color: #64748b; font-size: 0.8rem;'>FoodScan Meal Classifier v1.0.0<br>© 2026 Felipe III</p>", unsafe_allow_html=True)

# ----------------- Main Interface -----------------

st.markdown("<h1 class='main-title'>🍽️ FoodScan: Meal Category Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Deploy your MobileNetV2 deep learning model in seconds to distinguish Staples, Snacks, and Beverages.</p>", unsafe_allow_html=True)

# Ensure keras and torch are imported in the main loop to verify installation
try:
    import keras
    import torch
except ImportError:
    st.error("### 🛠️ Dependency Setup In Progress...")
    st.info("Keras or PyTorch is currently installing in the background. Please wait a few moments and then refresh this page.")
    st.stop()

# Load the active model
model = None
if model_path is not None:
    with st.spinner("🧠 Loading neural network model..."):
        model, error_msg = load_keras_model(model_path)
    
    if error_msg:
        st.error(f"### ❌ Error Loading Model File")
        st.markdown(f"""
        The model failed to load due to a version or deserialization error:
        ```text
        {error_msg}
        ```
        **Troubleshooting:**
        1. Ensure your model was saved with a compatible Keras version.
        2. Double-check that all custom layers or custom activation layers are registered correctly.
        """)
        st.stop()
    else:
        st.toast("🧠 Neural network model loaded successfully!", icon="✅")
else:
    st.info("💡 **Getting Started:** Select your model in the left sidebar to proceed with classification.")
    st.stop()

# Setup side-by-side columns for a premium dashboard look
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("<h3 style='margin-top: 0; color: #ff7e5f;'>📸 Step 1: Input Food Image</h3>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8; font-size: 0.95rem; margin-top: -10px;'>Upload a photo of the dish or beverage you want to classify.</p>", unsafe_allow_html=True)
    
    uploaded_image = st.file_uploader(
        "Choose an image...", 
        type=["png", "jpg", "jpeg"],
        help="Supported file extensions: PNG, JPG, JPEG"
    )
    
    image = None
    if uploaded_image is not None:
        image = Image.open(uploaded_image).convert("RGB")
        st.markdown("<p style='font-size: 0.9rem; font-weight: 500; margin-bottom: 5px; color: #feb47b;'>Uploaded Image Preview:</p>", unsafe_allow_html=True)
        st.image(image, use_container_width=True, caption=f"Uploaded: {uploaded_image.name}")
    else:
        st.info("ℹ️ Upload an image of food to trigger real-time neural network predictions.")
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='glass-card' style='height: 100%;'>", unsafe_allow_html=True)
    st.markdown("<h3 style='margin-top: 0; color: #feb47b;'>📈 Step 2: Prediction Analysis</h3>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8; font-size: 0.95rem; margin-top: -10px;'>Classification breakdown and confidence intervals computed by Keras.</p>", unsafe_allow_html=True)
    
    if image is not None and model is not None:
        with st.spinner("⏳ Analyzing pixels and running inference..."):
            # Image Preprocessing matching the single-image inference pipeline in training
            img_resized = image.resize((224, 224))
            img_array = np.array(img_resized, dtype=np.float32)
            img_array = np.expand_dims(img_array, axis=0) # Shape becomes (1, 224, 224, 3)
            
            # Run prediction
            prediction = model.predict(img_array, verbose=0)[0]
            
            # Map values
            pred_index = np.argmax(prediction)
            pred_class = CLASS_NAMES[pred_index]
            confidence = float(prediction[pred_index])
            
        # Display pred class in a high-end styled card
        card_class = f"glow-card-{pred_class.lower()}"
        color_theme = ""
        if pred_class == "BEVERAGE":
            color_theme = "#2dd4bf" # Teal
        elif pred_class == "SNACK":
            color_theme = "#fb7185" # Coral/Red
        else:
            color_theme = "#fbbf24" # Gold/Amber
            
        st.markdown(f"""
        <div class='{card_class}'>
            <span style='font-size: 0.9rem; text-transform: uppercase; letter-spacing: 2px; color: #cbd5e1; font-weight: 500;'>Top Prediction</span>
            <div class='glow-title' style='color: {color_theme};'>{pred_class}</div>
            <span style='font-size: 1.4rem; font-weight: 600; color: #f8fafc;'>{confidence:.2%} Confidence</span>
        </div>
        <br>
        """, unsafe_allow_html=True)
        
        # Display HSL customized progress bars for all 3 classes
        st.markdown("<p style='font-weight: 600; font-size: 1.1rem; color: #e2e8f0; margin-bottom: 12px;'>Confidence Breakdown:</p>", unsafe_allow_html=True)
        
        for idx, class_name in enumerate(CLASS_NAMES):
            val = float(prediction[idx])
            fill_class = f"fill-{class_name.lower()}"
            st.markdown(f"""
            <div class='progress-container'>
                <div class='progress-label-row'>
                    <span>{class_name}</span>
                    <span>{val:.2%}</span>
                </div>
                <div class='progress-bar-bg'>
                    <div class='progress-bar-fill {fill_class}' style='width: {val*100:.1f}%;'></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<hr>", unsafe_allow_html=True)
        
        # Display dynamic dietary balancing advice
        st.markdown("<h4 style='color: #feb47b; font-weight: 600; margin-top: 20px;'>🌱 Balanced Nutrition Guide</h4>", unsafe_allow_html=True)
        
        if pred_class == "BEVERAGE":
            st.markdown("""
            > [!TIP]
            > **Hydration & Sugar Check**
            > - **Watch Hidden Sugars**: Smoothies, soda, and sports drinks often contain high amounts of refined sugar which lead to insulin spikes.
            > - **Opt for Whole Foods**: Instead of juice, eat whole fruits to preserve dietary fiber which slows digestion.
            > - **Protein Boost**: If this is a protein shake or matcha smoothie, ensure it has healthy fats (like chia seeds or nut butter) for sustained satiety.
            """)
        elif pred_class == "SNACK":
            st.markdown("""
            > [!TIP]
            > **Satiety & Portion Control**
            > - **Go Whole Food**: Try to choose snacks rich in protein and fiber (like almonds, greek yogurt, or apple slices) over ultra-processed crackers or chips.
            > - **Mindful Eating**: Avoid snacking directly from a large package. Portions can get away quickly! Pour a small serving into a bowl.
            > - **Timing**: A healthy snack can help maintain blood sugar levels between main meals to prevent over-eating at dinner.
            """)
        elif pred_class == "STAPLE":
            st.markdown("""
            > [!TIP]
            > **Plate Proportion Method**
            > - **Plate Balance**: Aim for **1/2 of your plate vegetables**, **1/4 lean protein** (chicken, fish, tofu), and **1/4 complex carbohydrates** (brown rice, quinoa, whole grains).
            > - **Fiber First**: Eating your veggies first followed by protein helps flatten blood sugar spikes and keeps you energetic.
            """)
            
    else:
        st.markdown("""
        <div style='text-align: center; padding: 50px 20px; color: #64748b;'>
            <span style='font-size: 4rem;'>📸</span>
            <p style='margin-top: 15px; font-size: 1rem;'>Waiting for an image to run classification...</p>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("</div>", unsafe_allow_html=True)
