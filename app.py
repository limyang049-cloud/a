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
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

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
# CUSTOM CSS
# ==========================================
def local_css():
    st.markdown("""
    <style>
    :root {
        --primary: #0cb8b6;
        --primary-dark: #0a9a98;
        --secondary: #ffb737;
        --dark: #222222;
        --light: #f8f9fa;
    }
    .stApp { background-color: white; }
    h1, h2, h3, h4, h5, h6 { font-family: 'Raleway', sans-serif; font-weight: 600; color: #222222; }
    h1 { font-size: 35px; text-transform: uppercase; }
    .hero-section {
        background: linear-gradient(rgba(13, 70, 83, 0.78), rgba(13, 70, 83, 0.78)), 
                    url('https://images.unsplash.com/photo-1505751172876-fa5323b3ceb4?ixlib=rb-4.0.3&auto=format&fit=crop&w=1920&q=80');
        background-size: cover; background-attachment: fixed; min-height: 500px; padding: 100px 0; color: white; text-align: center;
    }
    .botm-line { height: 3px; width: 60px; background: #ffb737; border: 0; margin: 20px 0; }
    .service-card { text-align: center; margin-bottom: 30px; padding: 20px; }
    .service-icon { font-size: 45px; color: #0cb8b6; margin-bottom: 20px; }
    .cta-section { background-color: #0cb8b6; padding: 40px 0; color: white; }
    .footer { background-color: #325C6A; color: white; padding: 40px 0 20px; text-align: center; }
    .social-icon { display: inline-block; width: 45px; height: 45px; border-radius: 50%; text-align: center; line-height: 45px; margin: 0 10px; transition: all 0.5s; }
    .social-linkedin { background-color: #0976b4; color: white; }
    .social-instagram { background-color: #cb2027; color: white; }
    .social-github { background-color: #4183c4; color: white; }
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
    try:
        df = pd.read_csv("dataset.csv", names=columns, skiprows=38)
    except FileNotFoundError:
        st.error("❌ 'dataset.csv' not found.")
        return None, None, None, None, None
    X = df.drop(columns=['Outcome'])
    y = df['Outcome']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler, df, X.columns

if 'page' not in st.session_state:
    st.session_state.page = "home"

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
# UI SECTIONS
# ==========================================
def hero_section():
    st.markdown('<div class="hero-section"><h1>Healthcare at your desk!!</h1><p>Advanced Diabetes Prediction System</p></div>', unsafe_allow_html=True)

def services_section():
    st.markdown('<div style="padding: 60px 0;"><h2 style="text-align:center;">Our Services</h2><hr class="botm-line" style="margin:auto;">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    for col, icon, title in zip([col1, col2, col3], ["fa-stethoscope", "fa-ambulance", "fa-user-md"], ["24/7 Support", "Emergency", "Counseling"]):
        with col: st.markdown(f'<div class="service-card"><i class="fas {icon} service-icon"></i><h4>{title}</h4><p>Professional healthcare services.</p></div>', unsafe_allow_html=True)

def footer_section():
    st.markdown('<div class="footer"><p>© All Copyright Reserved. Doctello</p></div>', unsafe_allow_html=True)

# ==========================================
# SYSTEM CHECKER (IMPROVED GRAPH)
# ==========================================
def system_checker_page():
    st.title("🩺 Diabetes Risk Prediction System")
    X_scaled, y, scaler, df, feature_names = load_and_prep_data()
    if X_scaled is None: return
    
    models = train_base_models(X_scaled, y)
    
    # Prediction Form
    with st.expander("📝 Input Patient Data", expanded=True):
        c1, c2, c3 = st.columns(3)
        age = c1.number_input("Age", 1, 120, 30)
        bmi = c2.number_input("BMI", 0.0, 70.0, 28.0)
        glucose = c3.number_input("Glucose", 0.0, 300.0, 120.0)
        submit = st.button("Analyze Patient", type="primary")

    if submit:
        user_input = scaler.transform([[0, glucose, 70, 20, 80, bmi, 0.5, age]]) # Simplified input for brevity
        res = models["ANN"].predict(user_input)[0]
        if res == 1: st.error("🚨 High Risk Detected")
        else: st.success("✅ Low Risk Detected")

    st.markdown("---")
    if st.button("🔄 Run Performance Comparison"):
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        metrics_list = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        results = []
        cms = {}

        for name, model in models.items():
            y_pred = model.predict(X_test)
            results.append({
                "Model": name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred),
                "Recall": recall_score(y_test, y_pred),
                "F1 Score": f1_score(y_test, y_pred)
            })
            cms[name] = confusion_matrix(y_test, y_pred)

        results_df = pd.DataFrame(results).set_index("Model")
        
        # --- ENHANCED VISUALIZATION ---
        st.subheader("📈 Model Performance Analysis")
        colors = ['#3498db', '#2ecc71', '#e74c3c']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={'width_ratios': [1.5, 1]})

        # 1. Horizontal Grouped Bar Chart
        y_pos = np.arange(len(metrics_list))
        width = 0.2
        for i, model in enumerate(results_df.index):
            vals = results_df.loc[model, metrics_list].values
            rects = ax1.barh(y_pos + (i*width), vals, width, label=model, color=colors[i], alpha=0.8)
            ax1.bar_label(rects, fmt='%.2f', padding=5)

        ax1.set_yticks(y_pos + width)
        ax1.set_yticklabels(metrics_list)
        ax1.set_title("Metric Comparison by Algorithm", fontweight='bold')
        ax1.axvline(x=0.7, color='grey', linestyle='--', alpha=0.3)
        ax1.legend()

        # 2. Radar Chart
        angles = np.linspace(0, 2 * np.pi, len(metrics_list), endpoint=False).tolist()
        angles += angles[:1]
        ax2 = plt.subplot(1, 2, 2, polar=True)
        for i, model in enumerate(results_df.index):
            vals = results_df.loc[model, metrics_list].values.tolist()
            vals += vals[:1]
            ax2.plot(angles, vals, color=colors[i], linewidth=2, label=model)
            ax2.fill(angles, vals, color=colors[i], alpha=0.1)
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(metrics_list)
        ax2.set_title("Algorithm Balance (Radar)", fontweight='bold')

        plt.tight_layout()
        st.pyplot(fig)
        
        # Confusion Matrices
        st.markdown("#### 🔍 Confusion Matrices")
        cols = st.columns(3)
        for i, (name, cm) in enumerate(cms.items()):
            with cols[i]:
                fig_cm, ax_cm = plt.subplots(figsize=(4,3))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
                ax_cm.set_title(f"{name}")
                st.pyplot(fig_cm)

def main():
    local_css()
    with st.sidebar:
        st.title("🏥 Doctello")
        if st.button("🏠 Home"): st.session_state.page = "home"
        if st.button("🔬 System Checker"): st.session_state.page = "checker"
    
    if st.session_state.page == "home":
        hero_section()
        services_section()
        footer_section()
    elif st.session_state.page == "checker":
        system_checker_page()
        footer_section()

if __name__ == "__main__":
    main()
