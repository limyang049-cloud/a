import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Ignore warnings for cleaner output
warnings.filterwarnings('ignore')

# --- Page Configuration ---
st.set_page_config(page_title="Diabetes Prediction System", page_icon="🩺", layout="wide")

# --- Data Loading & Preprocessing ---
@st.cache_data
def load_and_prep_data():
    columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
               'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    # Skip the header text lines as per your original code
    df = pd.read_csv("dataset.csv", names=columns, skiprows=38)
    
    X = df.drop(columns=['Outcome'])
    y = df['Outcome']
    
    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler, df, X.columns

try:
    X_scaled, y, scaler, df, feature_names = load_and_prep_data()
except FileNotFoundError:
    st.error("❌ 'dataset.csv' not found. Please ensure it is in the same directory as this script.")
    st.stop()

# --- Pre-Train Models for Prediction Page ---
@st.cache_resource
def train_base_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        "KNN": KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train),
        "SVM": SVC(kernel='rbf', probability=True, random_state=42).fit(X_train, y_train),
        "ANN": MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42).fit(X_train, y_train)
    }
    return models

models = train_base_models(X_scaled, y)

# --- Sidebar Navigation ---
st.sidebar.title("Navigation 🧭")
page = st.sidebar.radio("Select a Page:", 
                        ["🏠 Home & Analytics", "🧪 Make a Prediction", "📊 Model Comparison"])

# ==========================================
# PAGE 1: HOME
# ==========================================
if page == "🏠 Home & Analytics":
    st.title("🩺 Diabetes Prediction System")
    st.write("Welcome to the Diabetes Prediction Portal. Use the sidebar to navigate to the prediction engine or view our algorithm performance metrics.")
    
    st.markdown("---")
    st.subheader("📊 Dataset Overview")
    st.dataframe(df.head(10), use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Dataset Shape:**", df.shape)
        st.write("**Diabetic Cases (Outcome=1):**", df['Outcome'].sum())
    with col2:
        st.write("**Non-Diabetic Cases (Outcome=0):**", len(df) - df['Outcome'].sum())
        st.write("**Total Features:**", len(feature_names))

# ==========================================
# PAGE 2: MAKE A PREDICTION
# ==========================================
elif page == "🧪 Make a Prediction":
    st.title("🧪 Patient Prediction Interface")
    st.write("Enter the patient's medical details below to predict the risk of diabetes using our trained models.")
    
    st.markdown("---")
    
    # Input Form
    col1, col2 = st.columns(2)
    
    with col1:
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=int(df['Pregnancies'].median()))
        glucose = st.number_input("Glucose Level", min_value=0.0, value=float(df['Glucose'].median()))
        bp = st
