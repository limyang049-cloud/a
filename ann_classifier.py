import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="ANN Classifier", page_icon="🧠", layout="wide")

st.title("🧠 Artificial Neural Network (ANN) Classification")
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
    
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Display dataset statistics
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Dataset Shape:**", df.shape)
        st.write("**Diabetic Cases:**", df['Outcome'].sum())
    with col2:
        st.write("**Non-Diabetic Cases:**", len(df) - df['Outcome'].sum())
        st.write("**Features:**", len(columns)-1)
    
    # --- 2. Feature Engineering ---
    X = df.drop(columns=['Outcome'])
    y = df['Outcome']
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # --- 3. ANN Model Parameters ---
    st.sidebar.header("🧠 ANN Model Settings")
    test_size = st.sidebar.slider("Test Size (%)", 10, 50, 20, key="ann_test")
    
    st.sidebar.subheader("Network Architecture")
    hidden_layers = st.sidebar.selectbox(
        "Hidden Layers Configuration",
        ["(50,25)", "(100,50)", "(100,50,25)", "(200,100,50)", "(150,100,50,25)"],
        index=2
    )
    
    # Parse hidden layers
    hidden_layer_sizes = tuple(map(int, hidden_layers.strip("()").split(",")))
    
    activation = st.sidebar.selectbox(
        "Activation Function",
        ["relu", "tanh", "logistic"],
        index=0
    )
    
    learning_rate = st.sidebar.selectbox(
        "Learning Rate",
        ["constant", "adaptive"],
        index=1
    )
    
    max_iterations = st.sidebar.slider("Max Iterations", 100, 2000, 1000, 100)
    random_state = st.sidebar.number_input("Random State", 0, 100, 42)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size/100, random_state=random_state
    )
    
    # --- 4. Train ANN Model ---
    st.subheader("🔄 Model Training")
    
    if st.button("🚀 Train ANN Model", type="primary"):
        with st.spinner("Training Artificial Neural Network... This may take a moment."):
            # Create and train ANN
            ann_model = MLPClassifier(
                hidden_layer_sizes=hidden_layer_sizes,
                activation=activation,
                learning_rate=learning_rate,
                max_iter=max_iterations,
                random_state=random_state,
                early_stopping=True,
                validation_fraction=0.1,
                verbose=False
            )
            
            ann_model.fit(X_train, y_train)
            y_pred = ann_model.predict(X_test)
            
            # Store model in session state
            st.session_state['ann_model'] = ann_model
            st.session_state['ann_scaler'] = scaler
            st.session_state['ann_trained'] = True
            
            st.success(f"✅ Model trained successfully! Iterations completed: {ann_model.n_iter_}")
            
            # --- 5. Evaluation Metrics ---
            st.subheader("📈 Model Evaluation")
            
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("🎯 Accuracy", f"{acc:.2%}", delta=None)
            col2.metric("📊 Precision", f"{prec:.2%}")
            col3.metric("🔄 Recall", f"{rec:.2%}")
            col4.metric("📐 F1 Score", f"{f1:.2%}")
            
            # Loss curve
            st.subheader("📉 Training Loss Curve")
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(ann_model.loss_curve_, color='blue', linewidth=2)
            ax.set_xlabel("Iterations")
            ax.set_ylabel("Loss")
            ax.set_title("ANN Training Loss Over Iterations")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()
            
            # Confusion Matrix
            st.subheader("📊 Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['No Diabetes', 'Diabetes'],
                       yticklabels=['No Diabetes', 'Diabetes'])
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.title("ANN Confusion Matrix")
            st.pyplot(fig)
            plt.close()
            
            # Additional metrics
            st.subheader("📋 Detailed Classification Report")
            tn, fp, fn, tp = cm.ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Additional Metrics:**")
                st.write(f"- Sensitivity (Recall): {rec:.4f}")
                st.write(f"- Specificity: {specificity:.4f}")
            with col2:
                st.write("**Predictive Values:**")
                st.write(f"- Positive Predictive Value: {prec:.4f}")
                st.write(f"- Negative Predictive Value: {npv:.4f}")
    
    # --- 6. Manual Prediction Section ---
    st.markdown("---")
    st.subheader("🔮 Predict for New Patient")
    
    if 'ann_trained' not in st.session_state:
        st.warning("⚠️ Please train the model first using the button above!")
    else:
        st.write("Enter patient data for prediction:")
        
        # Create input fields in columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            preg = st.number_input("Pregnancies", 0, 20, 3)
            glucose = st.number_input("Glucose", 0, 250, 120)
            bp = st.number_input("Blood Pressure", 0, 150, 70)
        
        with col2:
            skin = st.number_input("Skin Thickness", 0, 100, 20)
            insulin = st.number_input("Insulin", 0, 900, 80)
            bmi = st.number_input("BMI", 0.0, 70.0, 28.0, step=0.1)
        
        with col3:
            dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5, step=0.001)
            age = st.number_input("Age", 1, 120, 35)
        
        if st.button("🔍 Predict Diabetes Risk", type="secondary"):
            input_array = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
            input_scaled = st.session_state['ann_scaler'].transform(input_array)
            
            prediction = st.session_state['ann_model'].predict(input_scaled)[0]
            probability = st.session_state['ann_model'].predict_proba(input_scaled)[0]
            
            st.markdown("---")
            st.subheader("📋 Prediction Result")
            
            col1, col2 = st.columns(2)
            with col1:
                if prediction == 1:
                    st.error(f"🔴 **Positive for Diabetes**")
                else:
                    st.success(f"🟢 **Negative for Diabetes**")
            
            with col2:
                st.write(f"**Confidence:** {probability[prediction]:.2%}")
            
            # Probability bars
            st.write("**Probability Distribution:**")
            prob_df = pd.DataFrame({
                'Outcome': ['No Diabetes', 'Diabetes'],
                'Probability': probability
            })
            st.bar_chart(prob_df.set_index('Outcome'))
            
            # Risk factors analysis
            st.subheader("🔍 Risk Factor Analysis")
            risk_factors = []
            
            if glucose >= 140:
                risk_factors.append(f"⚠️ High Glucose ({glucose} mg/dL)")
            elif glucose >= 100:
                risk_factors.append(f"⚡ Elevated Glucose ({glucose} mg/dL)")
            
            if bmi >= 30:
                risk_factors.append(f"⚠️ Obese BMI ({bmi:.1f})")
            elif bmi >= 25:
                risk_factors.append(f"⚡ Overweight BMI ({bmi:.1f})")
            
            if age >= 45:
                risk_factors.append(f"⚠️ Age factor ({age} years)")
            
            if dpf >= 0.7:
                risk_factors.append(f"⚠️ High genetic risk (DPF: {dpf:.3f})")
            
            if insulin >= 200:
                risk_factors.append(f"⚠️ Elevated Insulin ({insulin} μU/mL)")
            
            if risk_factors:
                for factor in risk_factors:
                    st.write(factor)
            else:
                st.success("✅ No significant risk factors detected")
    
    # --- 7. Model Architecture Info ---
    st.markdown("---")
    st.subheader("🧬 Model Architecture Information")
    st.write("""
    **Artificial Neural Network (ANN)** is a computational model inspired by biological neural networks. 
    This implementation uses a Multi-Layer Perceptron (MLP) with:
    - **Input Layer:** 8 neurons (features)
    - **Hidden Layers:** Configurable architecture
    - **Output Layer:** 1 neuron (binary classification)
    
    **Key Features:**
    - Backpropagation for training
    - Early stopping to prevent overfitting
    - Adaptive learning rate optimization
    - ReLU/Tanh activation functions
    """)

except FileNotFoundError:
    st.error("❌ Error: 'dataset.csv' not found. Please ensure it is in the correct directory.")
except Exception as e:
    st.error(f"❌ An unexpected error occurred: {e}")
