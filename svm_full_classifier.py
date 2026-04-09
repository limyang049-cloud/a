import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="SVM Classifier", page_icon="🎯", layout="wide")

st.title("🎯 Support Vector Machine (SVM) Classification")
st.write("Supervised Machine Learning - Diabetes Prediction")

# --- 1. Data Pre-processing ---
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
           'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']

@st.cache_data
def load_data():
    df = pd.read_csv("dataset.csv", names=columns, skiprows=38)
    return df

try:
    df = load_data()
    
    st.subheader("📊 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Samples", len(df))
    with col2:
        st.metric("Diabetic Cases", df['Outcome'].sum())
    with col3:
        st.metric("Non-Diabetic", len(df) - df['Outcome'].sum())
    with col4:
        st.metric("Features", len(columns)-1)
    
    st.dataframe(df.head(10), use_container_width=True)
    
    # --- 2. Feature Engineering ---
    X = df.drop(columns=['Outcome'])
    y = df['Outcome']
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # --- 3. SVM Model Parameters ---
    st.sidebar.header("🎯 SVM Model Settings")
    
    test_size = st.sidebar.slider("Test Size (%)", 10, 50, 20, key="svm_test")
    
    st.sidebar.subheader("Kernel Configuration")
    kernel_type = st.sidebar.selectbox(
        "Kernel Function",
        ["rbf", "linear", "poly", "sigmoid"],
        index=0,
        help="RBF kernel works best for most cases"
    )
    
    # Kernel-specific parameters
    if kernel_type in ["rbf", "poly", "sigmoid"]:
        gamma = st.sidebar.selectbox(
            "Gamma",
            ["scale", "auto", "custom"],
            index=0
        )
        if gamma == "custom":
            gamma_value = st.sidebar.number_input("Gamma Value", 0.001, 1.0, 0.1, 0.001, format="%.3f")
        else:
            gamma_value = gamma
    
    if kernel_type == "poly":
        degree = st.sidebar.slider("Polynomial Degree", 2, 5, 3)
    else:
        degree = 3
    
    C_value = st.sidebar.slider(
        "Regularization (C)",
        0.1, 10.0, 1.0, 0.1,
        help="Lower C = smoother decision boundary, Higher C = more complex"
    )
    
    random_state = st.sidebar.number_input("Random State", 0, 100, 42)
    
    # Advanced options
    st.sidebar.subheader("Advanced Options")
    use_grid_search = st.sidebar.checkbox("Use Grid Search (Optimal Parameters)", value=False)
    use_pca = st.sidebar.checkbox("Apply PCA Visualization", value=True)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size/100, random_state=random_state, stratify=y
    )
    
    # --- 4. Train SVM Model ---
    st.subheader("🔄 Model Training")
    
    if st.button("🚀 Train SVM Model", type="primary"):
        with st.spinner("Training Support Vector Machine..."):
            
            if use_grid_search:
                st.info("🔍 Performing Grid Search for optimal parameters...")
                param_grid = {
                    'C': [0.1, 1, 10],
                    'gamma': [0.01, 0.1, 1, 'scale', 'auto'],
                    'kernel': [kernel_type]
                }
                
                svm_base = SVC(random_state=random_state, probability=True)
                grid_search = GridSearchCV(
                    svm_base, param_grid, cv=5, scoring='accuracy', n_jobs=-1
                )
                grid_search.fit(X_train, y_train)
                svm_model = grid_search.best_estimator_
                
                st.success(f"✅ Best parameters found: {grid_search.best_params_}")
                st.info(f"📊 Best cross-validation score: {grid_search.best_score_:.2%}")
            else:
                # Build SVM with selected parameters
                svm_params = {
                    'kernel': kernel_type,
                    'C': C_value,
                    'random_state': random_state,
                    'probability': True
                }
                
                if kernel_type in ["rbf", "poly", "sigmoid"]:
                    svm_params['gamma'] = gamma_value
                if kernel_type == "poly":
                    svm_params['degree'] = degree
                
                svm_model = SVC(**svm_params)
                svm_model.fit(X_train, y_train)
            
            y_pred = svm_model.predict(X_test)
            
            # Store in session state
            st.session_state['svm_model'] = svm_model
            st.session_state['svm_scaler'] = scaler
            st.session_state['svm_trained'] = True
            st.session_state['svm_kernel'] = kernel_type
            
            st.success(f"✅ SVM Model trained successfully with {kernel_type.upper()} kernel!")
            
            # --- 5. Evaluation Metrics ---
            st.subheader("📈 Model Evaluation")
            
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("🎯 Accuracy
