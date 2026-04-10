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
    
    try:
        df = pd.read_csv("dataset.csv", names=columns, skiprows=38)
    except FileNotFoundError:
        st.error("❌ 'dataset.csv' not found. Please ensure the dataset is in the correct directory.")
        return None, None, None, None, None
    
    X = df.drop(columns=['Outcome'])
    y = df['Outcome']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler, df, X.columns

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = "home"

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
                    <tr><td>Monday - Friday</td><td>8:00 - 17:00</td>
                    <tr>
                    <tr><td>Saturday</td><td>9:30 - 17:30</td>
                    </tr>
                    <tr><td>Sunday</td><td>9:30 - 15:00</td>
                    </tr>
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
# SYSTEM CHECKER / PREDICTION PAGE
# ==========================================
def system_checker_page():
    st.markdown('<div style="padding: 40px 0 20px;">', unsafe_allow_html=True)
    st.title("🩺 Diabetes Risk Prediction System")
    st.markdown("Enter patient diagnostic measurements below. The system will evaluate using multiple AI models.")
    
    # Load data
    X_scaled, y, scaler, df, feature_names = load_and_prep_data()
    
    if X_scaled is None:
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    # Train models
    with st.spinner("Training AI models..."):
        models = train_base_models(X_scaled, y)
    
    st.markdown("---")
    st.subheader("📊 Dataset Overview")
    
    # Dataset statistics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Records", df.shape[0])
    with col2:
        st.metric("Diabetic Cases", df['Outcome'].sum())
    with col3:
        st.metric("Healthy Cases", len(df) - df['Outcome'].sum())
    with col4:
        st.metric("Features", len(feature_names))
    with col5:
        st.metric("Diabetes Rate", f"{(df['Outcome'].sum()/len(df))*100:.1f}%")
    
    # Feature statistics expander
    with st.expander("📈 Feature Statistics Summary"):
        st.dataframe(df.describe(), use_container_width=True)
    
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
    st.markdown("Comprehensive evaluation of all three algorithms with detailed metrics and visualizations.")
    
    if st.button("🔄 Run Model Comparison", type="secondary"):
        with st.spinner("Training and evaluating models..."):
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
            
            comp_models = {
                "KNN": KNeighborsClassifier(n_neighbors=5),
                "SVM": SVC(kernel='rbf', random_state=42),
                "ANN": MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
            }
            
            results = []
            trained_models = {}
            confusion_matrices = {}
            
            for name, model in comp_models.items():
                model.fit(X_train, y_train)
                trained_models[name] = model
                y_pred = model.predict(X_test)
                confusion_matrices[name] = confusion_matrix(y_test, y_pred)
                
                results.append({
                    "Model": name,
                    "Accuracy": accuracy_score(y_test, y_pred),
                    "Precision": precision_score(y_test, y_pred, zero_division=0),
                    "Recall": recall_score(y_test, y_pred, zero_division=0),
                    "F1 Score": f1_score(y_test, y_pred, zero_division=0)
                })
            
            results_df = pd.DataFrame(results).set_index("Model")
            
            # Display metrics table
            st.markdown("#### 📋 Performance Metrics Table")
            st.dataframe(results_df.style.format("{:.2%}").highlight_max(axis=0, color="#d4edda"), use_container_width=True)
            
            # Comprehensive Metric Comparison - Dot Plot
            st.markdown("#### 📈 Comprehensive Metric Comparison (Dot Plot)")
            
            fig_dot, ax_dot = plt.subplots(figsize=(12, 6))
            
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
            colors = {'KNN': '#3498db', 'SVM': '#2ecc71', 'ANN': '#e74c3c'}
            markers = {'KNN': 'o', 'SVM': 's', 'ANN': '^'}
            
            y_pos = np.arange(len(metrics))
            
            for i, model in enumerate(results_df.index):
                values = [results_df.loc[model, 'Accuracy'], 
                         results_df.loc[model, 'Precision'],
                         results_df.loc[model, 'Recall'], 
                         results_df.loc[model, 'F1 Score']]
                
                # Add small horizontal offset for each model
                offset = (i - 1) * 0.2
                ax_dot.scatter(values, y_pos + offset, s=150, c=colors[model], 
                              marker=markers[model], label=model, alpha=0.8, zorder=3)
                
                # Add value labels
                for j, val in enumerate(values):
                    ax_dot.annotate(f'{val:.2f}', (val + 0.02, y_pos[j] + offset), 
                                   fontsize=8, ha='left', va='center')
            
            ax_dot.set_yticks(y_pos)
            ax_dot.set_yticklabels(metrics)
            ax_dot.set_xlabel('Score')
            ax_dot.set_title('Model Performance Comparison - Dot Plot', fontsize=14, fontweight='bold')
            ax_dot.set_xlim(0, 1.05)
            ax_dot.axvline(x=0.7, color='gray', linestyle='--', alpha=0.5, label='Good Threshold')
            ax_dot.legend(loc='lower right')
            ax_dot.grid(axis='x', alpha=0.3)
            
            st.pyplot(fig_dot)
            plt.close()
            
            # Confusion Matrices for Individual Algorithms
            st.markdown("#### 🔍 Confusion Matrices (Individual Algorithm Performance)")
            
            cm_cols = st.columns(3)
            cm_figs = []
            
            for idx, (name, cm) in enumerate(confusion_matrices.items()):
                fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
                
                # Custom heatmap with better styling
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                           xticklabels=['No Diabetes', 'Diabetes'],
                           yticklabels=['No Diabetes', 'Diabetes'],
                           ax=ax_cm, cbar=False, annot_kws={'size': 14, 'weight': 'bold'})
                
                ax_cm.set_title(f'{name} - Confusion Matrix', fontsize=12, fontweight='bold')
                ax_cm.set_xlabel('Predicted', fontsize=10)
                ax_cm.set_ylabel('Actual', fontsize=10)
                
                # Add accuracy annotation
                acc = results_df.loc[name, 'Accuracy']
                ax_cm.text(0.5, -0.15, f'Accuracy: {acc:.2%}', transform=ax_cm.transAxes,
                          ha='center', va='center', fontsize=10, fontweight='bold',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                
                with cm_cols[idx]:
                    st.pyplot(fig_cm)
                plt.close()
            
            # Performance comparison bar chart
            st.markdown("#### 📊 Performance Bar Chart")
            fig_bar, ax_bar = plt.subplots(figsize=(10, 5))
            results_df.plot(kind='bar', ax=ax_bar, color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12'])
            ax_bar.set_ylim(0, 1)
            ax_bar.set_ylabel("Score")
            ax_bar.set_title("Model Performance Comparison - Bar Chart", fontsize=14, fontweight='bold')
            ax_bar.legend(loc='lower right')
            ax_bar.grid(axis='y', alpha=0.3)
            ax_bar.set_xticklabels(ax_bar.get_xticklabels(), rotation=0)
            
            # Add value labels on bars
            for container in ax_bar.containers:
                ax_bar.bar_label(container, fmt='%.2f', fontsize=9, padding=2)
            
            st.pyplot(fig_bar)
            plt.close()
            
            # Summary statistics
            st.markdown("#### 📝 Summary Analysis")
            
            best_accuracy = results_df['Accuracy'].idxmax()
            best_f1 = results_df['F1 Score'].idxmax()
            best_precision = results_df['Precision'].idxmax()
            best_recall = results_df['Recall'].idxmax()
            
            col_s1, col_s2, col_s3, col_s4 = st.columns(4)
            with col_s1:
                st.info(f"🏆 **Best Accuracy**: {best_accuracy} ({results_df.loc[best_accuracy, 'Accuracy']:.2%})")
            with col_s2:
                st.info(f"🎯 **Best F1 Score**: {best_f1} ({results_df.loc[best_f1, 'F1 Score']:.2%})")
            with col_s3:
                st.success(f"📊 **Best Precision**: {best_precision} ({results_df.loc[best_precision, 'Precision']:.2%})")
            with col_s4:
                st.warning(f"🔄 **Best Recall**: {best_recall} ({results_df.loc[best_recall, 'Recall']:.2%})")
            
            # Algorithm characteristics table
            st.markdown("#### 🧬 Algorithm Characteristics")
            char_data = {
                "Algorithm": ["KNN", "SVM", "ANN"],
                "Strengths": [
                    "Simple, no training time, interpretable, good for small datasets",
                    "Effective in high dimensions, memory efficient, robust to overfitting",
                    "Captures complex patterns, highly flexible, learns non-linear relationships"
                ],
                "Weaknesses": [
                    "Slow prediction, sensitive to irrelevant features, needs feature scaling",
                    "Parameter tuning required, slower training on large datasets",
                    "Black-box nature, requires more data, computationally intensive"
                ],
                "Best For": [
                    "Small to medium datasets with clear clusters",
                    "High-dimensional data, binary classification",
                    "Complex patterns, large datasets, deep feature learning"
                ]
            }
            char_df = pd.DataFrame(char_data)
            st.dataframe(char_df, use_container_width=True, hide_index=True)
            
            # Recommendation
            st.markdown("---")
            st.markdown("#### 💡 Recommendation")
            
            if best_f1 == best_accuracy:
                recommended = best_accuracy
            else:
                # Choose based on F1 score (balance of precision and recall)
                recommended = best_f1
            
            st.success(f"""
            **{recommended}** is recommended for this diabetes prediction task based on:
            - Accuracy: {results_df.loc[recommended, 'Accuracy']:.2%}
            - Precision: {results_df.loc[recommended, 'Precision']:.2%}
            - Recall: {results_df.loc[recommended, 'Recall']:.2%}
            - F1 Score: {results_df.loc[recommended, 'F1 Score']:.2%}
            
            This algorithm provides the best balance between false positives and false negatives,
            which is crucial for medical diagnosis where both types of errors have significant consequences.
            """)
    
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
        "Fee (RM)": ["1200", "800", "750", "500", "600", "450", "650", "550"]
    }
    
    df_doctors = pd.DataFrame(doctors_data)
    st.dataframe(df_doctors, use_container_width=True, hide_index=True)
    
    st.markdown("""
    <div style="background-color: #f5f5f5; padding: 20px; border-radius: 8px; margin-top: 30px;">
        <h4 style="color: #0cb8b6;"><i class="fas fa-calendar-alt"></i> Book an Appointment</h4>
        <p>Call our helpline: <strong>+60 11 4455 1234</strong> or email: <strong>appointments@doctello.com</strong></p>
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
        st.markdown("📞 +60 11 4455 1234")
        st.markdown("✉️ info@doctello.com")
        st.markdown("📍 Maylaysia, Subang Jaya")
    
    # Page Routing
    page = st.session_state.page
    
    if page == "home":
        hero_section()
        services_section()
        cta_section()
        about_section()
        quote_section()
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
