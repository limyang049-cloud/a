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
st.set_page_config(
    page_title="Clinical Decision Support", 
    page_icon="🩺", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# --- Custom CSS for UI Polish ---
# This hides the default Streamlit top menu and footer to make it look like an independent app
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .stButton>button { width: 100%; border-radius: 5px; font-weight: bold; }
    
    div[data-testid="metric-container"] {
        padding: 15px; border-radius: 8px; border: 1px solid rgba(128, 128, 128, 0.2);
        box-shadow: 0 4px 6px rgba(0,0,0,0.05); background-color: transparent;
    }
    </style>
""", unsafe_allow_html=True)

# --- Data Loading & Preprocessing ---
@st.cache_data
def load_and_prep_data():
    columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
               'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    # Keeping your specific assignment requirement
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
    st.image("https://cdn-icons-png.flaticon.com/512/2966/2966327.png", width=80)
    st.title("Main Menu")
    page = st.radio("Navigate through the application:", ["🏠 Home & Analytics", "🧪 Patient Prediction", "📊 Model Comparison"])
    st.markdown("---")
    st.caption("Developed for Machine Learning Coursework")

# ==========================================
# PAGE 1: HOME & ANALYTICS
# ==========================================
if page == "🏠 Home & Analytics":
    st.title("🩺 Diabetes Risk Prediction Dashboard")
    st.markdown("Welcome to the **Intelligent Clinical Decision Support System**. This portal leverages advanced Machine Learning algorithms to assist healthcare professionals in early diabetes risk detection.")
    
    st.markdown("### 📈 Database Overview")
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Total Patient Records", df.shape[0])
    with col2: st.metric("Diabetic Cases ⚠️", df['Outcome'].sum())
    with col3: st.metric("Healthy Cases ✅", len(df) - df['Outcome'].sum())
    with col4: st.metric("Clinical Features", len(feature_names))

    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("🔍 View Raw Patient Dataset"):
        st.dataframe(df.style.highlight_max(axis=0, color="#ffcccc"), use_container_width=True)

# ==========================================
# PAGE 2: MAKE A PREDICTION
# ==========================================
elif page == "🧪 Patient Prediction":
    st.title("🧪 Patient Diagnostic Interface")
    st.markdown("Enter the patient's diagnostic measurements below. The system will evaluate the data against our trained predictive models.")
    
    with st.container():
        st.markdown("#### 🧑‍⚕️ Vitals & Demographics")
        c1, c2, c3 = st.columns(3)
        age = c1.number_input("Age (Years)", 1, 120, int(df['Age'].median()))
        bmi = c2.number_input("Body Mass Index (BMI)", 0.0, 70.0, float(df['BMI'].median()))
        bp = c3.number_input("Blood Pressure (mm Hg)", 0.0, 200.0, float(df['BloodPressure'].median()))
        
        st.markdown("#### 🩸 Lab Results")
        c4, c5, c6 = st.columns(3)
        glucose = c4.number_input("Glucose Level", 0.0, 300.0, float(df['Glucose'].median()))
        insulin = c5.number_input("Insulin Level", 0.0, 900.0, float(df['Insulin'].median()))
        skin = c6.number_input("Skin Thickness (mm)", 0.0, 100.0, float(df['SkinThickness'].median()))
        
        st.markdown("#### 🧬 Medical History")
        c7, c8 = st.columns(2)
        pregnancies = c7.number_input("Number of Pregnancies", 0, 20, int(df['Pregnancies'].median()))
        dpf = c8.number_input("Diabetes Pedigree Function", 0.0, 3.0, float(df['DiabetesPedigreeFunction'].median()), format="%.3f")

    st.markdown("---")
    c9, c10 = st.columns([1, 2])
    with c9:
        selected_model = st.selectbox("Select Prediction Engine:", ["Ensemble (All 3 Models)", "KNN", "SVM", "ANN"])
    with c10:
        st.write("") # Spacing
        st.write("") # Spacing
        submit_button = st.button("🧬 Analyze Patient Data", type="primary")
    
    if submit_button:
        user_input_scaled = scaler.transform(np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]]))
        st.markdown("### 📊 Diagnostic Results")
        
        if selected_model == "Ensemble (All 3 Models)":
            res_cols = st.columns(3)
            for i, (name, model) in enumerate(models.items()):
                pred = model.predict(user_input_scaled)[0]
                if pred == 1: res_cols[i].error(f"### {name}\n**Result:** High Risk ⚠️")
                else: res_cols[i].success(f"### {name}\n**Result:** Low Risk ✅")
        else:
            pred = models[selected_model].predict(user_input_scaled)[0]
            if pred == 1: st.error(f"#### The **{selected_model}** model indicates a **High Risk ⚠️**")
            else: st.success(f"#### The **{selected_model}** model indicates a **Low Risk ✅**")

# ==========================================
# PAGE 3: MODEL COMPARISON
# ==========================================
elif page == "📊 Model Comparison":
    st.title("📊 Algorithm Performance Comparison")
    col_settings, col_results = st.columns([1, 2.5])

    with col_settings:
        st.markdown("### ⚙️ Simulation Controls")
        test_size = st.slider("Test Data Allocation (%)", 10, 50, 20, help="20% is the industry standard.")
        random_state = st.number_input("Random Seed", 0, 100, 42, help="Locks mathematical randomness.")
        run_sim = st.button("🚀 Run Simulation", type="primary")

    with col_results:
        if run_sim or 'results_df' not in st.session_state:
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
                        "Model": name, "Accuracy": accuracy_score(y_test_comp, y_pred),
                        "Precision": precision_score(y_test_comp, y_pred),
                        "Recall": recall_score(y_test_comp, y_pred), "F1 Score": f1_score(y_test_comp, y_pred)
                    })
            results_df = pd.DataFrame(results).set_index("Model")
            
            st.markdown("### 🏆 Performance Leaderboard")
            st.dataframe(results_df.style.format("{:.2%}").highlight_max(axis=0, color="#d4edda"), use_container_width=True)
            
            # Advanced Visualizations
            t1, t2, t3 = st.tabs(["F1 Score Bar Chart", "Metric Comparison Plot", "Confusion Matrices"])
            with t1:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                bars = ax.bar(results_df.index, results_df['F1 Score'], color=['#3498db', '#2ecc71', '#e74c3c'])
                ax.set_ylabel('F1 Score')
                st.pyplot(fig)
            with t2:
                fig_line, ax_line = plt.subplots(figsize=(8, 4))
                for m in results_df.index:
                    ax_line.plot(['Accuracy', 'Precision', 'Recall', 'F1 Score'], results_df.loc[m], marker='o', label=m)
                ax_line.legend()
                ax_line.grid(True, linestyle='--', alpha=0.3)
                st.pyplot(fig_line)
            with t3:
                cm_cols = st.columns(3)
                for i, (name, model) in enumerate(comp_models.items()):
                    cm = confusion_matrix(y_test_comp, model.predict(X_test_comp))
                    fig_cm, ax_cm = plt.subplots(figsize=(3, 3))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm, cbar=False,
                                xticklabels=['Neg', 'Pos'], yticklabels=['Neg', 'Pos'])
                    ax_cm.set_title(name)
                    with cm_cols[i]: st.pyplot(fig_cm)

            # Summary Analysis
            st.markdown("---")
            st.subheader("📝 Summary Analysis")
            st.write("""
            | Algorithm | Strengths | Weaknesses |
            |-----------|-----------|------------|
            | **KNN** | Simple, no training time | Slow prediction, sensitive to irrelevant features |
            | **ANN** | Captures complex patterns | Black-box nature, requires more data |
            | **SVM** | Effective in high dimensions | Parameter tuning required, slower training |
            """)
            
            # Recommendation Engine
            best_model = results_df['F1 Score'].idxmax()
            st.success(f"🏆 **System Recommendation:** The **{best_model}** model is recommended for this clinical task based on achieving the highest F1 Score ({results_df.loc[best_model, 'F1 Score']:.2%}), representing the best balance between precision and recall.")
