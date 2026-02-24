import os
import json
import hashlib
import streamlit as st
import numpy as np
import tensorflow as tf
import requests
from PIL import Image


OPENROUTER_API_KEY = "YOUR-API-KEY"
response = requests.post(
    url="https://openrouter.ai/api/v1/chat/completions",
    headers={
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    },
    json={
        "model": "openai/gpt-4o-mini",
        "messages": [
            {"role": "user", "content": "prompt"}
        ]
    }
)

working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(working_dir, "trained_model", "plant_disease_prediction_model.h5")
class_indices_path = os.path.join(working_dir, "class_indices.json")

# Load the pre-trained model
try:
    model = tf.keras.models.load_model(model_path)
except Exception as e:
    st.error(f"⚠ Error loading model: {str(e)}")
    model = None  # Prevent crashes

# Load class names
try:
    with open(class_indices_path, "r") as f:
        class_indices = json.load(f)
except Exception as e: 
    st.error(f"⚠ Error loading class indices: {str(e)}")
    class_indices = {}

# Caching setup
CACHE = {}

def generate_cache_key(image):
    """Generate a unique cache key based on image hash."""
    return hashlib.md5(image.tobytes()).hexdigest()

def load_and_preprocess_image(image, target_size=(128, 128)):
    """Preprocess an uploaded image for model prediction."""
    img = image.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0
    return img_array

def fetch_recommendations(disease_name):

    if disease_name in CACHE:
        return CACHE[disease_name]

    # ✅ DEFINE PROMPT FIRST
    if "healthy" in disease_name.lower():
        prompt = f"The plant is healthy ({disease_name}). Give maintenance tips."
    else:
        prompt = f"Suggest treatment and prevention for {disease_name} in plants."

    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "openai/gpt-4o-mini",
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            }
        )

        data = response.json()

        if "error" in data:
            return f"⚠ OpenRouter API Error: {data['error']['message']}"

        recommendation = data["choices"][0]["message"]["content"]

        CACHE[disease_name] = recommendation
        return recommendation

    except Exception as e:
        return f"⚠ OpenRouter Exception: {str(e)}"
    
def predict_image_class(model, image, class_indices):
    """Predicts the class of an uploaded image."""
    if model is None:
        return "⚠ Model not loaded."

    cache_key = generate_cache_key(image)
    if cache_key in CACHE:
        return CACHE[cache_key]
    
    preprocessed_img = load_and_preprocess_image(image)
    predictions = model.predict(preprocessed_img)
    
    if predictions.size == 0:
        return "⚠ No prediction available."

    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices.get(str(predicted_class_index), "⚠ Unknown class")
    
    CACHE[cache_key] = predicted_class_name  # Store in cache
    return predicted_class_name



# Custom CSS for better UI
st.markdown("""
    <style>
        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background: linear-gradient(135deg, #4CAF50, #2E8B57);
            color: white;
        }
        [data-testid="stSidebar"] h1 {
            color: white;
        }
        
        /* Main Heading */
        .main-heading {
            text-align: center;
            font-size: 40px;
            color: #2E8B57;
            font-weight: bold;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }
        
        /* Image Styling */
        .uploaded-img {
            border-radius: 10px;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.2);
        }

        /* Button Styling */
        div.stButton > button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 24px;
            font-size: 16px;
            border-radius: 8px;
            border: none;
            transition: all 0.3s ease;
        }
        div.stButton > button:hover {
            background-color: #2E8B57;
        }
        .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background: linear-gradient(90deg, #0077B6, #2E8B57);
        color: white;
        text-align: center;
        padding: 10px;
        font-size: 14px;
        z-index: 100;
        }

    </style>
    <div class="footer">
        © 2026 Snehal Jadhav | 🌱 Plant Disease Detection System 🌱 | All Rights Reserved
    </div>
""", unsafe_allow_html=True)
    
# Sidebar navigation
st.sidebar.title("🌿 Navigation")
page = st.sidebar.radio("Go to", ["Home", "Demo", "Developer Info"])

if page == "Home":
    st.markdown("<h1 class='main-heading'>🌱 Plant Disease Detection System 🌱</h1>", unsafe_allow_html=True)
    st.write("""
        ### 🌿 About Dataset
        The PlantVillage dataset contains over 50,000 expertly curated images of healthy and diseased plant leaves. 
        It is designed to help develop machine learning-based plant disease detection solutions.
        
        ### 🔍 Project Workflow
        1. **Collect Image Data**  
        2. **Processing of The collected Data**  
        3. **Split in Train and Test**  
        4. **Build a Streamlit Web App**  
        5. **Model Evaluation**  
        6. **CNN Training**  
    """)

    workflow_image_path = "C:/Users/USER/Downloads/plant-disease-prediction-cnn-deep-leanring-project-main-master/plant-disease-prediction-cnn-deep-leanring-project-main-master/app/{06441611-6856-4DD6-8748-C5EEDBCF04C3}.png"  # Replace with correct path
    st.image(workflow_image_path, caption="Project Workflow Diagram", use_container_width=True)

    st.write("""
        ### 🔍 Tech Stack
        - **Machine Learning & Deep Learning:** TensorFlow/Keras, Pre-trained CNN Model (PlantVillage dataset)  
        - **Data Processing:** Pillow (PIL), NumPy, JSON  
        - **Web Framework & Frontend:** Streamlit, Custom CSS  
        - **File & System Management:** OS Module  
    """)

    tech_image_path = "C:/Users/USER/Downloads/plant-disease-prediction-cnn-deep-leanring-project-main-master/plant-disease-prediction-cnn-deep-leanring-project-main-master/app/Untitled design.jpg"  
    st.image(tech_image_path, caption="Tech Stack Diagram", use_container_width=True)

elif page == "Demo":
    st.markdown("<h1 class='main-heading'>📷 Plant Disease Detection Demo 📷</h1>", unsafe_allow_html=True)
    st.write("Upload an image to test the model.")

    uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Image', width=300)

        if st.button('🔍 Classify'):
            prediction = predict_image_class(model, image, class_indices)
            st.success(f'✅ Prediction: 🌿 {str(prediction)} 🌿')
            
            with st.spinner("🌿 Getting AI recommendations..."):
                recommendations = fetch_recommendations(prediction)
            st.info(f'🌱 Recommended Care: {recommendations}')


elif page == "Developer Info":
    st.markdown("<h1 class='main-heading'>🚀 Developer Information 🚀</h1>", unsafe_allow_html=True)


    developer_image_path = "C:/Users/USER/Downloads/plant-disease-prediction-cnn-deep-leanring-project-main-master/plant-disease-prediction-cnn-deep-leanring-project-main-master/app/WhatsApp Image 2026-01-12 at 15.06.31.jpeg"  # Replace with your actual image path
    st.image(developer_image_path, caption="Snehal Jadhav", width=700)

 
    st.markdown("""
        ## 📬 Connect with Me
        - 📧 **Email:** [buisnesssnehal@example.com](buisnesssnehal@example.com)  
        - 🔗 **GitHub:** [github.com/snehaljadhav](https://github.com/snehal395u)  
        - 🔗 **LinkedIn:** [linkedin.com/in/snehaljadhav](https://www.linkedin.com/in/snehal-jadhav-1b1101305/)  

        <p style='text-align: center; font-size: 16px;'>
            <b>Feel free to reach out! Let's build something amazing together. 🚀</b>
        </p>

        <hr style="border: 1px solid #2E8B57;">
    """, unsafe_allow_html=True)

    st.markdown("""
        <h2 style='text-align: center; color: #2E8B57;'>👋 Hey there! I'm Snehal Jadhav</h2>
        <p style='text-align: center; font-size: 18px;'>
        A passionate <b>Software Developer,AI Engineer</b> and <b>Data Science Enthusiast</b>, currently in my <b>third year</b> of a BSc in Computer Science. 
        I love working with <b>Deep Learning, Backend Development, and Data Structures & Algorithms</b>.
        </p>
        <hr style="border: 1px solid #2E8B57;">
    """, unsafe_allow_html=True)

    st.markdown("""
        ## 🌱 About This Project  
        This is a **Plant Disease Detection System** that uses **Deep Learning (CNNs)** to classify plant diseases.  
        The model is trained on the **PlantVillage dataset** and deployed using **Streamlit** for an interactive user experience.

        ### 🛠️ Tech Stack Used:
        - **Machine Learning & Deep Learning:** TensorFlow, Keras, Pre-trained CNN Model  
        - **Data Processing:** Pillow (PIL), NumPy, JSON  
        - **Backend & Web App:** Streamlit, Python  
        - **System & File Management:** OS Module  
        
        <br>
    """, unsafe_allow_html=True)

    st.markdown("""
        ## 💡 My Technical Expertise
        - 🔹 **Software Development:** Experienced in **Java, Spring Boot / Spring MVC / WebFlux, React / Next.js, Design Patterns, Database Design, Web Development, and General Aptitude, Microservises , Hibernate**, and REST API development.  
        - 🔹 **Machine Learning & AI:** Strong foundation in **Deep Learning, TensorFlow, and PyTorch**.  
        - 🔹 **Data Structures & Algorithms:** Proficient in **Java,Python**, with a focus on problem-solving.  
        - 🔹 **Web Development:** Hands-on experience with **Streamlit, Flask, and Django**.  
        - 🔹 **Database Management:** Worked with **MongoDB, MySQL, Oracle and Firebase**.  
        
        <hr style="border: 1px solid #2E8B57;">
    """, unsafe_allow_html=True)
    
