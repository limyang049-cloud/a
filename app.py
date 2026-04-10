# -*- coding: utf-8 -*-
"""
Doctello - Complete Medical Website with Diabetes Prediction System
Merged Healthcare Website + ML Classification (KNN, SVM, ANN)
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import re
from datetime import datetime

warnings.filterwarnings('ignore')

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Doctello - Healthcare & Diabetes Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================
# CUSTOM CSS (Matching Medical Theme)
# ==========================================
def local_css():
    st.markdown("""
    <style>
    /* Main color scheme */
    :root {
        --primary: #0cb8b6;
        --primary-dark: #0a9a98;
        --secondary: #ffb737;
        --dark: #222222;
        --light: #f8f9fa;
    }
    
    /* Global styles */
    .stApp {
        background-color: white;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Raleway', sans-serif;
        font-weight: 600;
        color: #222222;
    }
    
    h1 {
        font-size: 35px;
        text-transform: uppercase;
    }
    
    h2 {
        font-size: 28px;
        font-weight: 700;
        text-transform: uppercase;
    }
    
    /* Hero section */
    .hero-section {
        background: linear-gradient(rgba(13, 70, 83, 0.78), rgba(13, 70, 83, 0.78)), 
                    url('https://images.unsplash.com/photo-1505751172876-fa5323b3ceb4?ixlib=rb-4.0.3&auto=format&fit=crop&w=1920&q=80');
        background-size: cover;
        background-attachment: fixed;
        min-height: 500px;
        padding: 100px 0;
        color: white;
        text-align: center;
    }
    
    .hero-section h1 {
        color: white;
        font-family: 'Candal', sans-serif;
        font-size: 42px;
        margin-bottom: 20px;
    }
    
    .hero-section p {
        font-size: 18px;
        margin-bottom: 30px;
    }
    
    .btn-system-checker {
        background-color: rgba(12, 184, 182, 0.91);
        padding: 12px 30px;
        border-radius: 3px;
        color: white;
        text-decoration: none;
        display: inline-block;
        font-weight: 500;
        transition: all 0.3s;
        border: none;
        cursor: pointer;
    }
    
    .btn-system-checker:hover {
        background-color: #0cb8b6;
        color: white;
    }
    
    /* Section styling */
    .section-padding {
        padding: 60px 0;
    }
    
    .botm-line {
        height: 3px;
        width: 60px;
        background: #ffb737;
        border: 0;
        margin: 20px 0 20px 0;
    }
    
    /* Service cards */
    .service-card {
        text-align: center;
        margin-bottom: 30px;
        padding: 20px;
    }
    
    .service-icon {
        font-size: 45px;
        color: #0cb8b6;
        margin-bottom: 20px;
    }
    
    /* CTA Section */
    .cta-section {
        background-color: #0cb8b6;
        padding: 40px 0;
        color: white;
    }
    
    /* Footer */
    .footer {
        background-color: #325C6A;
        color: white;
        padding: 40px 0 20px;
        text-align: center;
    }
    
    .social-icon {
        display: inline-block;
        width: 45px;
        height: 45px;
        border-radius: 50%;
        text-align: center;
        line-height: 45px;
        margin: 0 10px;
        transition: all 0.5s;
    }
    
    .social-icon:hover {
        transform: rotate(360deg) scale(1.1);
    }
    
    .social-linkedin { background-color: #0976b4; color: white; }
    .social-instagram { background-color: #cb2027; color: white; }
    .social-github { background-color: #4183c4; color: white; }
    
    /* Success/Error messages */
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    
    .text-primary-custom {
        color: #0cb8b6;
    }
    
    .lg-line {
        font-size: 28px;
        line-height: 1.4;
    }
    
    /* Feature box */
    .feature-box {
        margin-bottom: 30px;
    }
    
    .feature-icon {
        float: left;
        width: 40px;
        height: 40px;
        background: #0cb8b6;
        border-radius: 50%;
        text-align: center;
        padding-top: 10px;
        color: white;
        margin-right: 20px;
    }
    
    .feature-text {
        margin-left: 60px;
    }
    
    /* Metric cards */
    div[data-testid="metric-container"] {
        padding: 15px;
        border-radius: 8px;
        border: 1px solid rgba(128, 128, 128, 0.2);
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        background-color: transparent;
    }
    </style>
    
    <link href="https://fonts.googleapis.com/css2?family=Open+Sans:wght@300;400;600;700&family=Raleway:wght@400;500;600;700&family=Candal&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
    """, unsafe_allow_html=True)

# ==========================================
# DATA LOADING & PREPROCESSING
# ==========================================
@st.cache_data
def load_and_prep_data():
    columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
               'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    
    # Try to load from uploaded file first, then default
    if 'uploaded_data' in st.session_state and st.session_state.uploaded_data is not None:
        df = st.session_state.uploaded_data
        # Ensure columns match
        if list(df.columns) != columns:
            st.warning("Uploaded file has different columns. Using default dataset.")
            df = pd.read_csv("dataset.csv", names=columns, skiprows=38)
    else:
        try:
            df = pd.read_csv("dataset.csv", names=columns, skiprows=38)
        except FileNotFoundError:
            st.error("❌ 'dataset.csv' not found. Please upload a dataset using the file uploader.")
            return None, None, None, None, None
    
    X = df.drop(columns=['Outcome'])
    y = df['Outcome']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler, df, X.columns

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = "home"
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None

# ==========================================
# TRAIN MODELS
# ==========================================
@st.cache_resource
def train_base_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        "KNN": KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train),
        "SVM": SVC(kernel='rbf', probability=True, random_state=42).fit(X_train, y_train),
        "ANN": MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42).fit(X_train, y_train)
    }
    return models

# ==========================================
# HERO SECTION
# ==========================================
def hero_section():
    st.markdown("""
    <div class="hero-section">
        <div style="max-width: 800px; margin: 0 auto;">
            <h1>Healthcare at your desk!!</h1>
            <p>This website is specially designed to help different peoples to classify their<br>respective disease and their concern doctors.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# SERVICES SECTION
# ==========================================
def services_section():
    st.markdown('<div class="section-padding">', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 2])
    
    with col1:
        st.markdown("""
        <h2>Our Service</h2>
        <hr class="botm-line">
        <p>Medical care is something everyone needs. Good customer service is just as important in medicine as it is in any other field.</p>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="service-card">
            <div class="service-icon"><i class="fas fa-stethoscope"></i></div>
            <h4>24 Hour Support</h4>
            <p>Call our Medical Help Line 24/7 to speak with an experienced registered nurse.</p>
        </div>
        <div class="service-card">
            <div class="service-icon"><i class="fas fa-ambulance"></i></div>
            <h4>Emergency Services</h4>
            <p>24/7 patient transport vehicle available for specialized care.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="service-card">
            <div class="service-icon"><i class="fas fa-user-md"></i></div>
            <h4>Medical Counseling</h4>
            <p>Professional medical advice and assistance to improve patient health.</p>
        </div>
        <div class="service-card">
            <div class="service-icon"><i class="fas fa-medkit"></i></div>
            <h4>OPD Services</h4>
            <p>Clinical consultation including history taking, examination, diagnosis and prescription.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# CTA SECTION
# ==========================================
def cta_section():
    st.markdown("""
    <div class="cta-section">
        <div style="max-width: 1200px; margin: 0 auto; display: flex; flex-wrap: wrap;">
            <div style="flex: 1; padding: 20px;">
                <h3>Cancer Care</h3>
                <p>Holistic integrated care with experts in Surgical Oncology, Radiation Oncology, and Medical Oncology.</p>
            </div>
            <div style="flex: 1; padding: 20px;">
                <h3>Bone Marrow Transplant</h3>
                <p>Advanced transplant procedures for leukemia, multiple myeloma, and severe blood diseases.</p>
            </div>
            <div style="flex: 1; padding: 20px; background-color: #0a9a98;">
                <h3>Opening Hours</h3>
                <table style="width:100%; color:white;">
                    <tr><td>Monday - Friday</td><td>8:00 - 17:00</td></tr>
                    <tr><td>Saturday</td><td>9:30 - 17:30</td></tr>
                    <tr><td>Sunday</td><td>9:30 - 15:00</td></tr>
                </table>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# ABOUT SECTION
# ==========================================
def about_section():
    st.markdown('<div style="background-color: rgba(238, 238, 238, 0.15); padding: 60px 0;">', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("""
        <h2 class="lg-line">Our Mission</h2>
        <hr class="botm-line">
        <p>To promote awareness among functionaries involved in Health and Hospital Management. To improve the efficiency of Health Care delivery systems.</p>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-box">
            <div class="feature-icon"><i class="fas fa-angle-right"></i></div>
            <div class="feature-text">
                <h3>Objective 1:</h3>
                <p>Increase the range of services wherever there are opportunities to meet customer needs.</p>
            </div>
        </div>
        <div class="feature-box">
            <div class="feature-icon"><i class="fas fa-angle-right"></i></div>
            <div class="feature-text">
                <h3>Objective 2:</h3>
                <p>Provide a safe and therapeutic environment for patients, staff, and visitors.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# QUOTE SECTION
# ==========================================
def quote_section():
    st.markdown("""
    <div style="background-color: #29302E; padding: 60px 0; color: white;">
        <div style="max-width: 1000px; margin: 0 auto; text-align: center;">
            <div style="display: flex; justify-content: center; gap: 50px; flex-wrap: wrap;">
                <div>
                    <h2 class="lg-line">« A few words<br> about me »</h2>
                </div>
                <div style="max-width: 400px; text-align: left;">
                    <p>I am a technocrat in Computer Science. My career objective is to touch the zenith of career by converting innovative ideas into fruitful results.</p>
                    <p class="text-primary-custom" style="text-align: right; margin-top: 20px;"><i>— Amber Kakkar<br>B.tech CSE</i></p>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# CONTACT SECTION
# ==========================================
def contact_section():
    st.markdown('<div class="section-padding">', unsafe_allow_html=True)
    
    st.markdown('<h2 style="text-align: center;">Contact us</h2>', unsafe_allow_html=True)
    st.markdown('<hr class="botm-line" style="margin: 20px auto;">', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("""
        <h3>Contact Info</h3>
        <p><i class="fas fa-map-marker-alt"></i> Dehradun, Uttarakhand, India, 248001</p>
        <p><i class="fas fa-envelope"></i> info@doctello.com</p>
        <p><i class="fas fa-phone"></i> +1 600 123 1234</p>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown('<h3>Having Any Query? Shoot us a message.</h3>', unsafe_allow_html=True)
        
        with st.form("contact_form"):
            name = st.text_input("Your Name", placeholder="Your Name")
            email = st.text_input("Your Email", placeholder="Your Email")
            subject = st.text_input("Subject", placeholder="Subject")
            message = st.text_area("Message", placeholder="Message", height=120)
            
            submitted = st.form_submit_button("Send Message", use_container_width=True)
            
            if submitted:
                if len(name) >= 2 and email and len(message) >= 5:
                    st.success("✅ Your message has been sent! Thank you!")
                else:
                    st.error("❌ Please fill all fields correctly.")
    
    st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# FOOTER SECTION
# ==========================================
def footer_section():
    st.markdown("""
    <div class="footer">
        <h4>Connect With Me</h4>
        <div style="margin: 20px 0;">
            <a href="https://www.linkedin.com/company/ieeeditu" target="_blank" class="social-icon social-linkedin">
                <i class="fab fa-linkedin"></i>
            </a>
            <a href="https://www.instagram.com/ieeeditu/?hl=en" target="_blank" class="social-icon social-instagram">
                <i class="fab fa-instagram"></i>
            </a>
            <a href="https://github.com/ieeeditu" target="_blank" class="social-icon social-github">
                <i class="fab fa-github-alt"></i>
            </a>
        </div>
        <hr style="border-color: rgba(255,255,255,0.1); margin: 20px auto; width: 80%;">
        <p>© All Copyright Reserved. Doctello</p>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# FAQ PAGE
# ==========================================
def faq_page():
    st.markdown("""
    <div style="background: linear-gradient(rgba(13, 70, 83, 0.9), rgba(13, 70, 83, 0.9)),
                url('https://images.unsplash.com/photo-1579684385127-1ef15d508118?ixlib=rb-4.0.3&auto=format&fit=crop&w=1920&q=80');
                background-size: cover; padding: 80px 0;">
        <div style="max-width: 900px; margin: 0 auto; background: rgba(255,255,255,0.95); border-radius: 10px; padding: 40px;">
            <h1 style="text-align: center; color: #0cb8b6;">Frequently Asked Questions</h1>
            <hr class="botm-line" style="margin: 20px auto;">
            
            <h3>Q1. What are the conditions treated by general physicians?</h3>
            <p>Ans: General physicians treat conditions like headaches, flu, urinary infections, blood pressure, diabetes, common aches, etc.</p>
            
            <h3>Q2. Do general physicians offer home consultation?</h3>
            <p>Ans: Yes, most general physicians offer home consultation in case of emergencies.</p>
            
            <h3>Q3. Will the consultation and diagnosis be covered in insurance?</h3>
            <p>Ans: Most common conditions are not covered by insurance. Please check with your provider.</p>
            
            <h3>Q4. Do general physicians provide in-house medicines?</h3>
            <p>Ans: Yes, some general physicians provide in-house medicines.</p>
            
            <h3>Q5. What are the consultation charges of general physicians in Dehradun?</h3>
            <p>Ans: Consultation fees range from Rs. 250 to Rs. 500.</p>
            
            <h3>Q6. What are the side effects of medications?</h3>
            <p>Ans: Common side effects include headaches, skin rashes, and mouth blisters. Always consult your doctor.</p>
            
            <h3>Q7. What diseases can be treated by a general physician?</h3>
            <p>Ans: Pain, headaches, cough, influenza, urinary infections, blood pressure, diabetes, and more.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# SYSTEM CHECKER / PREDICTION PAGE
# ==========================================
def system_checker_page():
    st.markdown('<div style="padding: 40px 0 20px;">', unsafe_allow_html=True)
    st.title("🩺 Diabetes Risk Prediction System")
    st.markdown("Enter patient diagnostic measurements below. The system will evaluate using multiple AI models.")
    
    # File uploader for custom dataset
    st.markdown("---")
    st.subheader("📁 Dataset Management")
    
    uploaded_file = st.file_uploader("Upload your own dataset (CSV format)", type=['csv'])
    
    if uploaded_file is not None:
        try:
            columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                       'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
            df_uploaded = pd.read_csv(uploaded_file)
            
            # Check if columns match expected format
            if len(df_uploaded.columns) == 9:
                df_uploaded.columns = columns
                st.session_state.uploaded_data = df_uploaded
                st.success(f"✅ Dataset loaded successfully! {df_uploaded.shape[0]} rows, {df_uploaded.shape[1]} columns")
                with st.expander("Preview uploaded data"):
                    st.dataframe(df_uploaded.head(10))
            else:
                st.error("Uploaded file must have 9 columns matching the diabetes dataset format.")
        except Exception as e:
            st.error(f"Error loading file: {e}")
    else:
        if st.button("📊 Use Default Dataset"):
            st.session_state.uploaded_data = None
            st.rerun()
    
    # Load data
    X_scaled, y, scaler, df, feature_names = load_and_prep_data()
    
    if X_scaled is None:
        st.warning("⚠️ Please upload a dataset or ensure 'dataset.csv' is available.")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    # Train models
    with st.spinner("Training AI models..."):
        models = train_base_models(X_scaled, y)
        st.session_state.models_trained = True
    
    st.markdown("---")
    st.subheader("📊 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", df.shape[0])
    with col2:
        st.metric("Diabetic Cases", df['Outcome'].sum())
    with col3:
        st.metric("Healthy Cases", len(df) - df['Outcome'].sum())
    with col4:
        st.metric("Features", len(feature_names))
    
    # Prediction Form
    st.markdown("---")
    st.subheader("🧪 Patient Prediction")
    
    with st.container():
        st.markdown("#### 🧑‍⚕️ Patient Information")
        c1, c2, c3 = st.columns(3)
        age = c1.number_input("Age (Years)", min_value=1, max_value=120, value=30)
        bmi = c2.number_input("BMI", min_value=0.0, value=28.0, step=0.1)
        bp = c3.number_input("Blood Pressure (mm Hg)", min_value=0.0, value=70.0)
        
        st.markdown("#### 🩸 Lab Results")
        c4, c5, c6 = st.columns(3)
        glucose = c4.number_input("Glucose Level", min_value=0.0, value=120.0)
        insulin = c5.number_input("Insulin Level", min_value=0.0, value=80.0)
        skin = c6.number_input("Skin Thickness (mm)", min_value=0.0, value=20.0)
        
        st.markdown("#### 🧬 Medical History")
        c7, c8 = st.columns(2)
        pregnancies = c7.number_input("Number of Pregnancies", min_value=0, max_value=20, value=2)
        dpf = c8.number_input("Diabetes Pedigree Function", min_value=0.000, value=0.5, format="%.3f")
    
    st.markdown("---")
    
    col_model, col_btn = st.columns([2, 1])
    with col_model:
        selected_model = st.selectbox("Select Prediction Engine:", 
                                      ["Ensemble (All 3 Models)", "KNN", "SVM", "ANN"])
    with col_btn:
        st.write("")
        st.write("")
        submit_button = st.button("🔬 Analyze Patient", type="primary", use_container_width=True)
    
    if submit_button:
        user_input = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
        user_input_scaled = scaler.transform(user_input)
        
        st.markdown("### 📊 Diagnostic Results")
        
        if selected_model == "Ensemble (All 3 Models)":
            res_cols = st.columns(3)
            results = {}
            for i, (name, model) in enumerate(models.items()):
                pred = model.predict(user_input_scaled)[0]
                results[name] = pred
                with res_cols[i]:
                    if pred == 1:
                        st.error(f"### {name}\n**Result:** High Risk ⚠️")
                    else:
                        st.success(f"### {name}\n**Result:** Low Risk ✅")
            
            # Ensemble voting
            votes = list(results.values())
            final_pred = 1 if sum(votes) >= 2 else 0
            
            st.markdown("---")
            if final_pred == 1:
                st.error("## 🚨 FINAL VERDICT: HIGH RISK OF DIABETES")
                st.markdown("""
                **Recommendations:**
                - Consult a physician immediately
                - Get a fasting blood sugar test
                - Monitor your diet and exercise
                """)
            else:
                st.success("## ✅ FINAL VERDICT: LOW RISK OF DIABETES")
                st.markdown("""
                **Recommendations:**
                - Maintain healthy lifestyle
                - Regular check-ups recommended
                - Balanced diet and exercise
                """)
        else:
            pred = models[selected_model].predict(user_input_scaled)[0]
            proba = models[selected_model].predict_proba(user_input_scaled)[0]
            
            if pred == 1:
                st.error(f"#### The **{selected_model}** model indicates a **HIGH RISK ⚠️** of diabetes")
                st.progress(proba[1], text=f"Risk Probability: {proba[1]:.1%}")
            else:
                st.success(f"#### The **{selected_model}** model indicates a **LOW RISK ✅** of diabetes")
                st.progress(proba[0], text=f"Confidence: {proba[0]:.1%}")
    
    # Model Comparison Section
    st.markdown("---")
    st.subheader("📊 Model Performance Comparison")
    
    if st.button("🔄 Run Model Comparison"):
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        comp_models = {
            "KNN": KNeighborsClassifier(n_neighbors=5),
            "SVM": SVC(kernel='rbf', random_state=42),
            "ANN": MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
        }
        
        results = []
        for name, model in comp_models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            results.append({
                "Model": name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred, zero_division=0),
                "Recall": recall_score(y_test, y_pred, zero_division=0),
                "F1 Score": f1_score(y_test, y_pred, zero_division=0)
            })
        
        results_df = pd.DataFrame(results).set_index("Model")
        st.dataframe(results_df.style.format("{:.2%}").highlight_max(axis=0, color="#d4edda"), use_container_width=True)
        
        # Bar chart
        fig, ax = plt.subplots(figsize=(10, 5))
        results_df.plot(kind='bar', ax=ax, color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12'])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Score")
        ax.set_title("Model Performance Comparison")
        ax.legend(loc='lower right')
        ax.grid(axis='y', alpha=0.3)
        st.pyplot(fig)
        plt.close()
    
    st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# DOCTOR DIRECTORY PAGE
# ==========================================
def doctor_directory_page():
    st.markdown('<div class="section-padding">', unsafe_allow_html=True)
    st.title("👨‍⚕️ Find a Doctor")
    st.markdown("Browse our specialist directory and book an appointment.")
    
    doctors_data = {
        "Specialist": ["General Physician", "Cardiologist", "Neurologist", "Dermatologist", 
                      "Orthopedic", "Pediatrician", "Gynecologist", "ENT Specialist"],
        "Available Days": ["Mon-Sat 9AM-5PM", "Mon-Wed-Fri 10AM-2PM", "Tue-Thu 11AM-3PM", 
                          "Mon-Fri 9AM-4PM", "Mon-Sat 10AM-6PM", "Mon-Fri 9AM-1PM", 
                          "Mon-Thu 10AM-4PM", "Mon-Sat 11AM-3PM"],
        "Fee (₹)": ["400", "800", "750", "500", "600", "450", "650", "550"]
    }
    
    df_doctors = pd.DataFrame(doctors_data)
    st.dataframe(df_doctors, use_container_width=True, hide_index=True)
    
    st.markdown("""
    <div style="background-color: #f5f5f5; padding: 20px; border-radius: 8px; margin-top: 30px;">
        <h4 style="color: #0cb8b6;"><i class="fas fa-calendar-alt"></i> Book an Appointment</h4>
        <p>Call our helpline: <strong>+1 600 123 1234</strong> or email: <strong>appointments@doctello.com</strong></p>
        <p>Emergency services available 24/7. All major insurance plans accepted.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# HEALTH TIPS PAGE
# ==========================================
def health_tips_page():
    st.markdown('<div class="section-padding">', unsafe_allow_html=True)
    st.title("💚 Daily Health Tips")
    st.markdown("Simple lifestyle changes for better health.")
    
    tips = [
        {"title": "Stay Hydrated", "desc": "Drink at least 8 glasses of water daily.", "icon": "fa-tint"},
        {"title": "Regular Exercise", "desc": "30 minutes of moderate exercise daily.", "icon": "fa-heartbeat"},
        {"title": "Balanced Diet", "desc": "Include fruits, vegetables, and whole grains.", "icon": "fa-apple-alt"},
        {"title": "Adequate Sleep", "desc": "Adults need 7-9 hours of quality sleep.", "icon": "fa-bed"},
        {"title": "Stress Management", "desc": "Practice meditation or yoga.", "icon": "fa-peace"},
        {"title": "Regular Checkups", "desc": "Annual health screenings save lives.", "icon": "fa-stethoscope"}
    ]
    
    cols = st.columns(3)
    for idx, tip in enumerate(tips):
        with cols[idx % 3]:
            st.markdown(f"""
            <div style="text-align: center; padding: 20px; background: #f8f9fa; border-radius: 8px; margin-bottom: 20px;">
                <i class="fas {tip['icon']}" style="font-size: 40px; color: #0cb8b6;"></i>
                <h4 style="margin-top: 15px;">{tip['title']}</h4>
                <p style="color: #666;">{tip['desc']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# MAIN APP
# ==========================================
def main():
    # Apply custom CSS
    local_css()
    
    # Sidebar Navigation
    with st.sidebar:
        st.markdown("### 🏥 Doctello")
        st.markdown("---")
        
        nav_options = {
            "🏠 Home": "home",
            "🩺 Services": "services",
            "📋 About": "about",
            "❓ FAQ": "faq",
            "📞 Contact": "contact",
            "🔬 System Checker": "checker",
            "👨‍⚕️ Find Doctor": "doctor",
            "💚 Health Tips": "tips"
        }
        
        for label, page_key in nav_options.items():
            if st.button(label, use_container_width=True, key=f"nav_{page_key}"):
                st.session_state.page = page_key
                st.rerun()
        
        st.markdown("---")
        st.markdown("### Contact Info")
        st.markdown("📞 +1 600 123 1234")
        st.markdown("✉️ info@doctello.com")
        st.markdown("📍 Dehradun, India")
    
    # Page Routing
    page = st.session_state.page
    
    if page == "home":
        hero_section()
        services_section()
        cta_section()
        about_section()
        quote_section()
        contact_section()
        footer_section()
    
    elif page == "services":
        st.markdown('<div style="padding: 40px 0 20px;"><h1 style="text-align: center;">Our Services</h1><hr class="botm-line" style="margin: 20px auto;"></div>', unsafe_allow_html=True)
        services_section()
        cta_section()
        footer_section()
    
    elif page == "about":
        st.markdown('<div style="padding: 40px 0 20px;"><h1 style="text-align: center;">About Us</h1><hr class="botm-line" style="margin: 20px auto;"></div>', unsafe_allow_html=True)
        about_section()
        quote_section()
        footer_section()
    
    elif page == "faq":
        faq_page()
        footer_section()
    
    elif page == "contact":
        st.markdown('<div style="padding: 40px 0 20px;"><h1 style="text-align: center;">Contact Us</h1><hr class="botm-line" style="margin: 20px auto;"></div>', unsafe_allow_html=True)
        contact_section()
        footer_section()
    
    elif page == "checker":
        system_checker_page()
        footer_section()
    
    elif page == "doctor":
        doctor_directory_page()
        footer_section()
    
    elif page == "tips":
        health_tips_page()
        footer_section()

if __name__ == "__main__":
    main()
