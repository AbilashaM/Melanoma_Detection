import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
from google import generativeai as genai
import io
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.platypus import Image as RLImage
from reportlab.lib.utils import ImageReader
import gdown
import os
import pickle


api_key = st.secrets["api_keys"]["my_api_key"]
# 🔹 Configure Google Gemini AI
genai.configure(api_key=api_key)  
MODEL_URL = "https://drive.google.com/uc?id=1ambozQoAuebq8ZrOdk9KXc2fth5A0Mr7"
MODEL_PATH = "melanoma_detection_model_final.keras"

@st.cache_resource  # ✅ keeps it in memory across reruns
def load_keras_model(url: str, path: str):
    if not os.path.exists(path):
        with st.spinner("Downloading model (~1× only)…"):
            gdown.download(url, path, quiet=False)
    return tf.keras.models.load_model(path)  # works for .keras files :contentReference[oaicite:0]{index=0}

model1 = load_keras_model(MODEL_URL, MODEL_PATH)


#@st.cache_resource()
#def load_model():
#    return tf.keras.models.load_model("melanoma_detection_model_final.keras")

#smodel1 = load_model()

# 🔹 Class names
CLASS_NAMES = ["Actinic Keratosis", "Basal Cell Carcinoma", "Dermatofibroma", 
               "Melanoma", "Vascular Lesion"]

# 🔹 Explanation texts
CLASS_EXPLANATIONS = {
    "Actinic Keratosis": "Pre-cancerous scaly spots due to sun exposure.",
    "Basal Cell Carcinoma": "Most common skin cancer, slow-growing, and rarely spreads.",
    "Dermatofibroma": "Benign skin growth, often appearing as a firm bump.",
    "Melanoma": "Most dangerous skin cancer, can spread quickly if untreated.",
    "Vascular Lesion": "Includes angiomas and hemangiomas, usually harmless blood vessel growths."
}

# 🔹 Heatmap explanation texts
HEATMAP_EXPLANATIONS = heatmap_explanations = {
    "Actinic Keratosis": """
        - 🔴 **Red/yellow areas** indicate sun-damaged skin regions.  
        - The AI focuses on **rough, scaly patches**, which are typical of actinic keratosis.  
        - Early detection is important as it can develop into skin cancer.
    """,
    "Basal Cell Carcinoma": """
        - 🔴 **Red/yellow areas** highlight **pearly, waxy bumps** or sores that don’t heal.  
        - AI focuses on **raised edges** or **shiny skin areas**, which are common signs of BCC.  
        - This is the **most common** type of skin cancer but grows slowly.
    """,
    "Dermatofibroma": """
        - 🔴 **Red/yellow areas** are around **firm, dark bumps** on the skin.  
        - AI detects **well-defined, smooth borders**, which differentiate it from melanoma.  
        - This condition is **benign** and usually does not require treatment.
    """,
    "Melanoma": """
        - 🔴 **Red/yellow areas** highlight **irregular borders, multiple colors, or asymmetry**.  
        - AI focuses on **dark patches** or **uneven textures**, which are warning signs of melanoma.  
        - Early detection is critical as melanoma can **spread quickly**.
    """,
    "Vascular Lesion": """
        - 🔴 **Red/yellow areas** indicate **blood vessel growths like hemangiomas**.  
        - AI focuses on **irregular red or purple patches**, distinguishing them from skin cancers.  
        - These lesions are **harmless** and usually do not require treatment.
    """
}

# 🔹 Image Preprocessing
def preprocess_image(image):
    """Resizes, normalizes, and prepares the image for model prediction."""
    image = image.resize((180, 180))  # Resize to match model input
    image = img_to_array(image)  # Convert to NumPy array

    if image.shape[-1] == 4:  # Convert RGBA to RGB if needed
        image = image[:, :, :3]

    image = image / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    return image.astype("float32")

# Function to generate heatmap (alternative to Grad-CAM)
def generate_heatmap(image):
    """Creates a simple heatmap using OpenCV for visualization."""
    heatmap = cv2.applyColorMap(np.uint8(255 * image[0]), cv2.COLORMAP_JET)
    heatmap = cv2.addWeighted(heatmap, 0.5, np.uint8(255 * image[0]), 0.5, 0)
    return heatmap

# 🔹 Initialize Chat State
if "messages" not in st.session_state:
    st.session_state.messages = []

if "prediction_done" not in st.session_state:
    st.session_state.prediction_done = False  # Flag to track if detection ran
if "selected_page" not in st.session_state:
    st.session_state.selected_page = "Home"

# Sidebar layout
st.sidebar.title(" ")

# Function to render buttons and highlight the active one
def nav_button(label):
    is_active = st.session_state.selected_page == label

    if st.sidebar.button(label, key=label):
        st.session_state.selected_page = label

# Add nav buttons
nav_button("Home")
nav_button("About")
if st.session_state.selected_page == "Home":

    # 🔹 Streamlit UI
    st.title("🩺 AI-Powered Melanoma Detector & Chatbot")
    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["sender"]):
            if msg["type"] == "text":
                st.markdown(msg["text"])
            elif msg["type"] == "image":
                st.image(msg["image"], caption=msg["caption"], width=250)


    # 🔹 Image Upload for Detection (Runs Only Once)
    uploaded_file = st.file_uploader("📤 Upload a skin lesion image:", type=["jpg", "jpeg", "png"])
    if st.button("🔮 Predict"):
        st.session_state.prediction_done = False
        if uploaded_file and not st.session_state.prediction_done:
            image1 = load_img(uploaded_file)
            img_array = preprocess_image(image1)
            img = Image.open(uploaded_file).convert("RGB")  # Convert to RGB
                # Display uploaded image
            st.session_state.messages.append({"sender": "user", "type": "image", "image": img, "caption": "Uploaded Image"})
            with st.chat_message("user"):
                st.image(img, caption="Uploaded Image", width=250)

            # Convert to model-compatible format
            img = img.resize((180, 180))  # Resize to match model input
            img_array1 = img_to_array(img)  # Convert to NumPy array
            img_array1= np.expand_dims(img_array1, axis=0)  # Add batch dimension

            # Predict
            pred = model1.predict(img_array1)
            pred_index = np.argmax(pred)
            pred_class = CLASS_NAMES[pred_index]
            confidence = np.max(pred)*100
        
            # Display results
            if pred_class != "Melanoma":

                diagnosis_text = f"**🩺 It is not melanoma,it can be {pred_class}  \n📊 Confidence Level: {confidence:.2f}%  \nℹ️ Info : {CLASS_EXPLANATIONS[CLASS_NAMES[pred_index]]}**"
            else:
                diagnosis_text = f"**🩺 Prediction: {pred_class}  \n📊 Confidence Level: {confidence:.2f}%  \nℹ️ Info:{CLASS_EXPLANATIONS[CLASS_NAMES[pred_index]]}**"

            # Display Prediction
        
            st.session_state.messages.append({"sender": "bot", "type": "text", "text": diagnosis_text})

            with st.chat_message("bot"):
                st.markdown(diagnosis_text)
            
            # Display Heatmap Section Title
            heatmap_text = "📊 **Explanation using Heatmap**"
            st.session_state.messages.append({"sender": "bot", "type": "text", "text": heatmap_text})

            with st.chat_message("bot"):
                st.markdown(heatmap_text)

            # Generate Heatmap
            heatmap = generate_heatmap(img_array)

            # Display Heatmap and Explanation Side by Side
            col1, col2 = st.columns([1, 1])

            with col1:
                st.image(heatmap, caption="Heatmap Overlay", width=250)

            with col2:
                heatmap_expl = heatmap_explanations.get(pred_class, "No specific explanation available.")
                st.markdown(heatmap_expl)

            st.session_state.messages.append({"sender": "bot", "type": "image", "image": heatmap, "caption": "Heatmap Overlay"})
            st.session_state.messages.append({"sender": "bot", "type": "text", "text": heatmap_expl})

            # Explanation of Heatmap Guide (Below the Image and Text)
            heatmap_guide = """
            <div style="
                border: 2px solid #ccc; 
                padding: 10px; 
                border-radius: 10px; 
                box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
                text-align: center;">
                <b>🔍 Understanding the Heatmap</b>  
                <p>The heatmap highlights the regions that influenced the AI's decision.</p> 
                <div style="
                    width: 100%; 
                    height: 10px; 
                    background: linear-gradient(to right, red, yellow, green, blue);
                    border-radius: 5px;
                    margin-top: 10px;
                "></div>
                <p style="display: flex; justify-content: space-between;">
                    <span style="color: red;">High Importance</span> 
                    <span style="color: blue;">Low Importance</span>
                </p>
                <ul style="text-align: left;">
                    <li><b style="color:red;">🔴 Red:</b> Most important areas for diagnosis.</li>
                    <li><b style="color:gold;">🟡 Yellow:</b> Significant but slightly less important.</li>
                    <li><b style="color:green;">🟢 Green:</b> Less important in decision-making.</li>
                    <li><b style="color:blue;">🔵 Blue:</b> Least important areas.</li>
                </ul>
            </div>
            """

            with st.chat_message("bot"):
                st.markdown(heatmap_guide, unsafe_allow_html=True)
            from datetime import datetime

            if not st.session_state.prediction_done:
                buffer = BytesIO()
                c = canvas.Canvas(buffer, pagesize=letter)
                width, height = letter

                # Title
                c.setFont("Helvetica-Bold", 16)
                c.drawString(50, height - 50, "🩺 Melanoma Detection Report")

                # Date at top-right
                c.setFont("Helvetica", 10)
                date_str = datetime.now().strftime("%B %d, %Y")
                c.drawRightString(width - 50, height - 50, f"Date: {date_str}")

                # Uploaded Image
                c.setFont("Helvetica-Bold", 12)
                c.drawString(50, height - 100, "📷 Uploaded Image:")
                uploaded_img_pil = Image.open(uploaded_file).convert("RGB")
                uploaded_img_io = BytesIO()
                uploaded_img_pil.save(uploaded_img_io, format='PNG')
                uploaded_img_io.seek(0)
                img_reader = ImageReader(uploaded_img_io)
                y = height - 150
                c.drawImage(img_reader, 50, y - 350, width=200, preserveAspectRatio=True, mask='auto')

                # Prediction Info
                c.setFont("Helvetica-Bold", 12)
                c.drawString(50, height - 320, "🧠 Prediction:")
                c.setFont("Helvetica", 11)
                c.drawString(70, height - 340, f"Class: {pred_class}")
                c.drawString(70, height - 360, f"Confidence: {confidence:.2f}%")

                # Explanation Text
                explanation = CLASS_EXPLANATIONS[CLASS_NAMES[pred_index]]
                c.setFont("Helvetica-Bold", 12)
                c.drawString(50, height - 390, "ℹ️ Explanation:")
                c.setFont("Helvetica", 10)
                y = height - 410
                for line in explanation.split('\n'):
                    c.drawString(70, y, line.strip())
                    y -= 15

                # Heatmap Image
                c.setFont("Helvetica-Bold", 12)
                c.drawString(50, y - 10, "🔥 Heatmap:")
                heatmap_pil = Image.fromarray(heatmap)
                heatmap_io = BytesIO()
                heatmap_pil.save(heatmap_io, format='PNG')
                heatmap_io.seek(0)
                heatmap_reader = ImageReader(heatmap_io)
                c.drawImage(heatmap_reader, 50, y - 210, width=200, preserveAspectRatio=True, mask='auto')

                # Heatmap explanation
                c.setFont("Helvetica", 10)
                line_height = 15
                text_y = y - 230  # Starting Y position for heatmap explanation

                for line in heatmap_expl.strip().split('\n'):
                    c.drawString(50, text_y, line.strip())
                    text_y -= line_height

                # Disclaimer at bottom
                c.setFont("Helvetica-Bold", 9)
                c.drawString(50, 40, "* Disclaimer: This tool is for informational purposes only. Consult a dermatologist for a professional diagnosis.")

                # Finalize and download
                c.save()
                buffer.seek(0)
                st.download_button("📥 Click to Download Report", data=buffer, file_name="melanoma_report.pdf", mime="application/pdf")




            # 🔹 Set Prediction Done Flag to Avoid Repeating
            st.session_state.prediction_done = True





    # 🔹 Chatbot for Skin Cancer Queries
    user_input = st.chat_input("💬 Ask me anything about skin cancer...")

    if user_input:
        # Append user message
        st.session_state.messages.append({"sender": "user", "type": "text", "text": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # **🔹 Google Gemini AI for Chatbot**
        model = genai.GenerativeModel("gemini-2.0-flash")

        chatbot_prompt = (
            f"You are a medical assistant specialized in skin cancer. "
            f"Provide answers about melanoma, basal cell carcinoma, actinic keratosis, "
            f"dermatofibroma, and vascular lesions. "
            f"If the question is unrelated, say: "
            f"'I can only assist with skin cancer and related conditions.'\n\n"
            f"give only relavent and precise answers "
            f"User's question: {user_input}"
        )

        try:
            response = model.generate_content(chatbot_prompt)
            bot_reply = response.text if response.text else "I'm not sure. Please ask something else."
        except Exception:
            bot_reply = "⚠️ Sorry, something went wrong."

        # Append bot response
        st.session_state.messages.append({"sender": "bot", "type": "text", "text": bot_reply})
        with st.chat_message("bot"):
            st.markdown(bot_reply)

if st.session_state.selected_page == "About":
    st.markdown("## 🩺 About Melanoma Detection")
    st.markdown("""
    Our AI model detects **melanoma** and differentiates it from 5 other skin conditions:

    - **Actinic Keratosis** (Pre-cancerous sun damage)  
    - **Basal Cell Carcinoma** (Slow-growing skin cancer)  
    - **Dermatofibroma** (Benign skin growth)  
    - **Vascular Lesion** (Harmless blood vessel growth)  

    🔍 **How It Works:**  
    1️⃣ Upload a skin lesion image  
    2️⃣ CNN model analyzes & classifies the lesion  
    3️⃣ A **heatmap** highlights key areas  
    4️⃣ Get a detailed explanation & confidence score  

    🚨 **Disclaimer:** This tool is for **informational purposes only**. Consult a dermatologist for a professional diagnosis.
    """)

    st.markdown("---")

    st.markdown("## 💬 About Chatbot")
    st.markdown("""
    Our **AI chatbot** provides reliable information on:  
    ✅ Skin Cancer  
    ✅ Skin lesion symptoms & risks  
    ✅ Prevention & early detection tips  
    ✅ When to seek medical attention  

    🤖 **Powered by Google Gemini AI**, it ensures **accurate** & **medically relevant** responses.
    """)
