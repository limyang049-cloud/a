import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

# --- Page Configuration ---
st.set_page_config(page_title="Diabetes Prediction System", page_icon="🩺", layout="wide")

# --- Data Loading & Preprocessing (Cached for performance) ---
@st.cache_data
def load_and_prep_data():
    columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
               'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    df = pd.read_csv("dataset.csv", names=columns, skiprows=38)
    
    X = df.drop(columns=['Outcome'])
    y = df['Outcome']
    
    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler, df

X_scaled, y, scaler, df = load_and_prep_data()

# --- Train Models (Cached so they only train once) ---
@st.cache_resource
def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        "KNN": KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train),
        "SVM": SVC(kernel='rbf', probability=True, random_state=42).fit(X_train, y_train),
        "ANN": MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42).fit(X_train, y_train)
    }
    return models

models = train_models(X_scaled, y)

# --- Sidebar Navigation ---
st.sidebar.title("Navigation 🧭")
page = st.sidebar.radio("Select a Page:", 
                        ["🏠 Home & Analytics", "🧪 Make a Prediction", "📊 Model Comparison"])

# --- Page Routing ---
if page == "🏠 Home & Analytics":
    st.title("Diabetes Prediction System")
    st.write("Welcome to the Diabetes Prediction Portal. Use the sidebar to navigate to the prediction engine or view our algorithm performance metrics.")
    st.subheader("Dataset Overview")
    st.dataframe(df.head())

elif page == "🧪 Make a Prediction":
    st.title("Patient Prediction Interface")
    st.write("Enter the patient's medical details below to predict the risk of diabetes.")
    
    # Create a nice form layout
    col1, col2 = st.columns(2)
    
    with col1:
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0)
        glucose = st.number_input("Glucose Level", min_value=0.0, value=120.0)
        bp = st.number_input("Blood Pressure", min_value=0.0, value=70.0)
        skin = st.number_input("Skin Thickness", min_value=0.0, value=20.0)
        
    with col2:
        insulin = st.number_input("Insulin Level", min_value=0.0, value=79.0)
        bmi = st.number_input("BMI", min_value=0.0, value=25.0)
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.000, value=0.500, format="%.3f")
        age = st.number_input("Age", min_value=1, max_value=120, value=30)
        
    selected_model = st.selectbox("Select Prediction Algorithm", ["Ensemble (All 3)", "KNN", "SVM", "ANN"])
    
    if st.button("Generate Prediction", type="primary"):
        # Format input for the model
        user_input = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
        user_input_scaled = scaler.transform(user_input)
        
        st.markdown("---")
        st.subheader("Results")
        
        if selected_model == "Ensemble (All 3)":
            # Show results from all models side-by-side
            res_cols = st.columns(3)
            for i, (name, model) in enumerate(models.items()):
                pred = model.predict(user_input_scaled)[0]
                status = "High Risk ⚠️" if pred == 1 else "Low Risk ✅"
                res_cols[i].metric(label=f"{name} Prediction", value=status)
        else:
            # Show single model result
            pred = models[selected_model].predict(user_input_scaled)[0]
            if pred == 1:
                st.error(f"**Result:** The {selected_model} model indicates a **High Risk** of diabetes.")
            else:
                st.success(f"**Result:** The {selected_model} model indicates a **Low Risk** of diabetes.")

elif page == "📊 Model Comparison":
    st.title("Algorithm Performance")
    st.write("This section contains the logic from your `model_comparison.py` file.")
    # You can copy and paste the visualization logic from your model_comparison.py here
