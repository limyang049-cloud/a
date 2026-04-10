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

# Ignore warnings for cleaner output
warnings.filterwarnings('ignore')

# --- Page Configuration ---
# Setting a wide layout and a custom theme color (handled slightly by Streamlit config, but we can structure it well)
st.set_page_config(page_title="Diabetes Prediction System", page_icon="🩸", layout="wide", initial_sidebar_state="expanded")

# --- Custom CSS for UI Polish ---
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        font-weight: bold;
    }
    .stMetric {
        background-color: white;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    </style>
""", unsafe_allow_html=True)

# --- Data Loading & Preprocessing ---
@st.cache_data
def load_and_prep_data():
    columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
               'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    # Adjust skiprows if your dataset header requires it
    df = pd.read_csv("dataset.csv", names=columns, skiprows=38)
    
    X = df.drop(columns=['Outcome'])
    y = df['Outcome']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler, df, X.columns

try:
    X_scaled, y, scaler, df, feature_names = load_and_prep_data()
except FileNotFoundError:
    st.error("❌ 'dataset.csv' not found. Please ensure it is in the same directory as this script.")
    st.stop()

# --- Pre-Train Models ---
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
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2966/2966327.png", width=100) # Optional placeholder logo
    st.title("Main Menu")
    st.markdown("Navigate through the application modules:")
    page = st.radio("", ["🏠 Home & Analytics", "🧪 Patient Prediction", "📊 Model Comparison"])
    st.markdown("---")
    st.caption("Developed for Machine Learning Coursework")

# ==========================================
# PAGE 1: HOME & ANALYTICS
# ==========================================
if page == "🏠 Home & Analytics":
    st.title("🩺 Diabetes Risk Prediction Dashboard")
    st.markdown("Welcome to the **Intelligent Clinical Decision Support System**. This portal leverages advanced Machine Learning algorithms to assist healthcare professionals in early diabetes risk detection.")
    
    st.markdown("### 📈 Database Overview")
    
    # Use metrics for a professional dashboard look
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Patient Records", df.shape[0])
    with col2:
        st.metric("Diabetic Cases ⚠️", df['Outcome'].sum())
    with col3:
        st.metric("Healthy Cases ✅", len(df) - df['Outcome'].sum())
    with col4:
        st.metric("Clinical Features", len(feature_names))

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Hide the giant dataframe inside an expander so it doesn't clutter the page
    with st.expander("🔍 View Raw Patient Dataset"):
        st.dataframe(df.style.highlight_max(axis=0, color="#ffcccc"), use_container_width=True)

# ==========================================
# PAGE 2: MAKE A PREDICTION
# ==========================================
elif page == "🧪 Patient Prediction":
    st.title("🧪 Patient Prediction Interface")
    st.markdown("Enter the patient's diagnostic measurements below. The system will evaluate the data against our trained predictive models.")
    
    # Create an organized form so inputs look grouped
    with st.container():
        st.markdown("#### 🧑‍⚕️ Vitals & Demographics")
        c1, c2, c3 = st.columns(3)
        age = c1.number_input("Age (Years)", min_value=1, max_value=120, value=int(df['Age'].median()))
        bmi = c2.number_input("Body Mass Index (BMI)", min_value=0.0, value=float(df['BMI'].median()))
        bp = c3.number_input("Blood Pressure (mm Hg)", min_value=0.0, value=float(df['BloodPressure'].median()))
        
        st.markdown("#### 🩸 Lab Results")
        c4, c5, c6 = st.columns(3)
        glucose = c4.number_input("Glucose Level", min_value=0.0, value=float(df['Glucose'].median()))
        insulin = c5.number_input("Insulin Level", min_value=0.0, value=float(df['Insulin'].median()))
        skin = c6.number_input("Skin Thickness (mm)", min_value=0.0, value=float(df['SkinThickness'].median()))
        
        st.markdown("#### 🧬 Medical History")
        c7, c8 = st.columns(2)
        pregnancies = c7.number_input("Number of Pregnancies", min_value=0, max_value=20, value=int(df['Pregnancies'].median()))
        dpf = c8.number_input("Diabetes Pedigree Function", min_value=0.000, value=float(df['DiabetesPedigreeFunction'].median()), format="%.3f")

    st.markdown("---")
    
    c9, c10 = st.columns([1, 2])
    with c9:
        selected_model = st.selectbox("Select Prediction Engine:", ["Ensemble (All 3 Models)", "KNN", "SVM", "ANN"])
    
    with c10:
        st.write("") # Spacing
        st.write("") # Spacing
        submit_button = st.button("🧬 Analyze Patient Data", type="primary")

    if submit_button:
        user_input = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
        user_input_scaled = scaler.transform(user_input)
        
        st.markdown("### 📊 Diagnostic Results")
        
        if selected_model == "Ensemble (All 3 Models)":
            res_cols = st.columns(3)
            for i, (name, model) in enumerate(models.items()):
                pred = model.predict(user_input_scaled)[0]
                with res_cols[i]:
                    if pred == 1:
                        st.error(f"### {name}\n**Result:** High Risk ⚠️")
                    else:
                        st.success(f"### {name}\n**Result:** Low Risk ✅")
        else:
            pred = models[selected_model].predict(user_input_scaled)[0]
            if pred == 1:
                st.error(f"#### The **{selected_model}** model indicates a **High Risk ⚠️** of diabetes for this patient.")
            else:
                st.success(f"#### The **{selected_model}** model indicates a **Low Risk ✅** of diabetes for this patient.")

# ==========================================
# PAGE 3: MODEL COMPARISON
# ==========================================
elif page == "📊 Model Comparison":
    st.title("📊 Algorithm Performance Comparison")
    st.markdown("Evaluate and compare the underlying machine learning architectures. Adjust parameters on the left to re-run simulations.")

    # Split the screen into controls (left) and results (right)
    col_settings, col_results = st.columns([1, 2.5])

    with col_settings:
        st.markdown("### ⚙️ Simulation Controls")
        st.info("Adjust the train/test split to see how data volume affects model accuracy.")
        test_size = st.slider("Test Data Allocation (%)", 10, 50, 20)
        random_state = st.number_input("Random Seed", 0, 100, 42)
        run_sim = st.button("🚀 Run Simulation", type="primary")

    with col_results:
        if run_sim or 'results_df' not in st.session_state:
            # Re-split data based on user slider settings
            X_train_comp, X_test_comp, y_train_comp, y_test_comp = train_test_split(
                X_scaled, y, test_size=test_size/100.0, random_state=random_state
            )

            comp_models = {
                "KNN": KNeighborsClassifier(n_neighbors=5),
                "SVM": SVC(kernel='rbf', probability=True, random_state=random_state),
                "ANN": MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=random_state)
            }

            results = []
            
            with st.spinner("Training neural networks and classifiers..."):
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

            results_df = pd.DataFrame(results).set_index("Model")
            
            # --- Display Results ---
            st.markdown("### 🏆 Leaderboard")
            st.dataframe(results_df.style.format("{:.2%}").highlight_max(axis=0, color="#d4edda"), use_container_width=True)
            
            # --- Graphical Visualizations ---
            tab1, tab2, tab3 = st.tabs(["F1 Score Bar Chart", "Metric Radar Plot", "Confusion Matrices"])
            
            with tab1:
                fig, ax = plt.subplots(figsize=(8, 4))
                # Removed chart borders for cleaner modern look
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                bars = ax.bar(results_df.index, results_df['F1 Score'], color=['#3498db', '#2ecc71', '#e74c3c'])
                ax.set_ylabel('F1 Score')
                ax.set_ylim(0, 1.0)
                for bar in bars:
                    height = bar.get_height()
                    ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
                st.pyplot(fig)
                
            with tab2:
                fig_line, ax_line = plt.subplots(figsize=(8, 4))
                ax_line.spines['top'].set_visible(False)
                ax_line.spines['right'].set_visible(False)
                colors = {'KNN': '#3498db', 'SVM': '#2ecc71', 'ANN': '#e74c3c'}
                metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
                for model in results_df.index:
                    ax_line.plot(metrics, results_df.loc[model, metrics], marker='o', markersize=8, linewidth=2, label=model, color=colors[model])
                ax_line.set_ylim(0, 1.1)
                ax_line.legend(loc='lower right')
                ax_line.grid(True, linestyle='--', alpha=0.3)
                st.pyplot(fig_line)
                
            with tab3:
                cm_cols = st.columns(3)
                for i, (name, model) in enumerate(comp_models.items()):
                    y_pred_cm = model.predict(X_test_comp)
                    cm = confusion_matrix(y_test_comp, y_pred_cm)
                    fig_cm, ax_cm = plt.subplots(figsize=(3, 3))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                                xticklabels=['Neg', 'Pos'], yticklabels=['Neg', 'Pos'], ax=ax_cm)
                    ax_cm.set_title(f'{name}')
                    with cm_cols[i]:
                        st.pyplot(fig_cm)
