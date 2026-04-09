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
        bp = st.number_input("Blood Pressure", min_value=0.0, value=float(df['BloodPressure'].median()))
        skin = st.number_input("Skin Thickness", min_value=0.0, value=float(df['SkinThickness'].median()))
        
    with col2:
        insulin = st.number_input("Insulin Level", min_value=0.0, value=float(df['Insulin'].median()))
        bmi = st.number_input("BMI", min_value=0.0, value=float(df['BMI'].median()))
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.000, value=float(df['DiabetesPedigreeFunction'].median()), format="%.3f")
        age = st.number_input("Age", min_value=1, max_value=120, value=int(df['Age'].median()))
        
    st.markdown("---")
    selected_model = st.selectbox("Select Prediction Algorithm", ["Ensemble (All 3 Models)", "KNN", "SVM", "ANN"])
    
    if st.button("Generate Prediction", type="primary"):
        # Format input for the model
        user_input = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
        user_input_scaled = scaler.transform(user_input)
        
        st.subheader("Prediction Results")
        
        if selected_model == "Ensemble (All 3 Models)":
            res_cols = st.columns(3)
            for i, (name, model) in enumerate(models.items()):
                pred = model.predict(user_input_scaled)[0]
                if pred == 1:
                    res_cols[i].error(f"**{name} Model**\n\nHigh Risk ⚠️")
                else:
                    res_cols[i].success(f"**{name} Model**\n\nLow Risk ✅")
        else:
            pred = models[selected_model].predict(user_input_scaled)[0]
            if pred == 1:
                st.error(f"The **{selected_model}** model indicates a **High Risk ⚠️** of diabetes.")
            else:
                st.success(f"The **{selected_model}** model indicates a **Low Risk ✅** of diabetes.")

# ==========================================
# PAGE 3: MODEL COMPARISON
# ==========================================
elif page == "📊 Model Comparison":
    st.title("📊 Algorithm Performance Comparison")
    st.write("Compare the performance of KNN, SVM, and ANN by adjusting the test parameters below.")

    # --- Experiment Settings ---
    st.markdown("### ⚙️ Experiment Settings")
    col_settings_1, col_settings_2 = st.columns(2)
    with col_settings_1:
        test_size = st.slider("Test Size (%)", 10, 50, 20)
    with col_settings_2:
        random_state = st.number_input("Random State", 0, 100, 42)

    if st.button("🚀 Run Comparison", type="primary"):
        # Re-split data based on user slider settings
        X_train_comp, X_test_comp, y_train_comp, y_test_comp = train_test_split(
            X_scaled, y, test_size=test_size/100.0, random_state=random_state
        )

        # Define fresh models for the comparison
        comp_models = {
            "KNN": KNeighborsClassifier(n_neighbors=5),
            "SVM": SVC(kernel='rbf', probability=True, random_state=random_state),
            "ANN": MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=random_state)
        }

        results = []
        
        # Train and evaluate models
        with st.spinner("Training models and calculating metrics..."):
            for name, model in comp_models.items():
                model.fit(X_train_comp, y_train_comp)
                y_pred = model.predict(X_test_comp)
                
                results.append({
                    "Model": name,
                    "Accuracy": accuracy_score(y_test_comp, y_pred),
                    "Precision": precision_score(y_test_comp, y_pred),
                    "Recall": recall_score(y_test_comp, y_pred),
                    "F1 Score": f1_score(y_test_comp, y_pred)
                })

        # Create and display DataFrame
        results_df = pd.DataFrame(results).set_index("Model")
        
        st.markdown("---")
        st.subheader("📈 Performance Metrics")
        
        # Highlight the best scores in green
        st.dataframe(
            results_df.style.format("{:.2%}")
            .highlight_max(axis=0, color="darkgreen"),
            use_container_width=True
        )

        # --- Visualizations ---
        st.subheader("📊 F1 Score Comparison")
        fig, ax = plt.subplots(figsize=(8, 4))
        
        bars = ax.bar(results_df.index, results_df['F1 Score'], color=['#3498db', '#2ecc71', '#e74c3c'])
        
        ax.set_ylabel('F1 Score')
        ax.set_title('Model Comparison by F1 Score')
        ax.set_ylim(0, 1.0)
        
        # Add value labels on top of the bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), 
                        textcoords="offset points",
                        ha='center', va='bottom')
        
        st.pyplot(fig)

        # --- Summary Analysis ---
        st.subheader("📝 Summary Analysis")
        st.write("""
        | Algorithm | Strengths | Weaknesses |
        |-----------|-----------|------------|
        | **KNN** | Simple, no training time, interpretable | Slow prediction, sensitive to irrelevant features |
        | **ANN** | Captures complex patterns, highly flexible | Requires more data, black-box nature |
        | **SVM** | Effective in high dimensions, memory efficient | Parameter tuning required, slower training |
        """)
        
        # --- Best Model Recommendation ---
        best_f1_model = results_df['F1 Score'].idxmax()
        best_accuracy_model = results_df['Accuracy'].idxmax()
        
        st.subheader("🏆 Recommended Model")
        
        if results_df.loc[best_f1_model, 'F1 Score'] == results_df.loc[best_accuracy_model, 'Accuracy']:
            recommended = best_accuracy_model
        else:
            recommended = best_f1_model
            
        st.success(f"""
        **{recommended}** is recommended for this diabetes prediction task based on the current settings:
        - Accuracy: **{results_df.loc[recommended, 'Accuracy']:.2%}**
        - Precision: **{results_df.loc[recommended, 'Precision']:.2%}**
        - Recall: **{results_df.loc[recommended, 'Recall']:.2%}**
        - F1 Score: **{results_df.loc[recommended, 'F1 Score']:.2%}**
        """)
    else:
        st.info("👆 Adjust your settings and click 'Run Comparison' to generate metrics and charts.")
