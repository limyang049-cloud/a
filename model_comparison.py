# model_comparison.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Model Comparison", page_icon="📊", layout="wide")

st.title("📊 Algorithm Comparison: KNN vs ANN vs SVM")
st.write("Diabetes Prediction - Performance Analysis")

# --- Load Data ---
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
           'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']

@st.cache_data
def load_data():
    df = pd.read_csv("dataset.csv", names=columns, skiprows=38)
    return df

df = load_data()
X = df.drop(columns=['Outcome'])
y = df['Outcome']

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Sidebar settings
st.sidebar.header("⚙️ Experiment Settings")
test_size = st.sidebar.slider("Test Size (%)", 10, 50, 20)
random_state = st.sidebar.number_input("Random State", 0, 100, 42)
run_experiment = st.sidebar.button("🚀 Run Comparison", type="primary")

if run_experiment:
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size/100, random_state=random_state, stratify=y
    )
    
    st.subheader("📊 Dataset Split Information")
    col1, col2, col3 = st.columns(3)
    col1.metric("Training Samples", len(X_train))
    col2.metric("Testing Samples", len(X_test))
    col3.metric("Total Features", X.shape[1])
    
    # --- Define Models ---
    models = {
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
        "ANN (MLP)": MLPClassifier(hidden_layer_sizes=(100,50), max_iter=1000, random_state=random_state),
        "SVM (RBF)": SVC(kernel='rbf', C=1.0, random_state=random_state)
    }
    
    results = {}
    
    st.subheader("🔄 Training Models...")
    progress_bar = st.progress(0)
    
    for i, (name, model) in enumerate(models.items()):
        with st.spinner(f"Training {name}..."):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            results[name] = {
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred, zero_division=0),
                'Recall': recall_score(y_test, y_pred, zero_division=0),
                'F1 Score': f1_score(y_test, y_pred, zero_division=0)
            }
        progress_bar.progress((i + 1) / len(models))
    
    # --- Display Results ---
    st.subheader("📈 Performance Comparison")
    
    # Results Table
    results_df = pd.DataFrame(results).T
    results_df = results_df.round(4)
    
    st.dataframe(results_df, use_container_width=True)
    
    # Highlight Best Model
    best_accuracy_model = results_df['Accuracy'].idxmax()
    best_f1_model = results_df['F1 Score'].idxmax()
    
    col1, col2 = st.columns(2)
    with col1:
        st.success(f"🏆 **Best Accuracy:** {best_accuracy_model} ({results_df.loc[best_accuracy_model, 'Accuracy']:.2%})")
    with col2:
        st.success(f"🎯 **Best F1 Score:** {best_f1_model} ({results_df.loc[best_f1_model, 'F1 Score']:.2%})")
    
    # Bar Chart Comparison
    st.subheader("📊 Visual Comparison")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(results_df.columns))
    width = 0.25
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    for i, (model_name, metrics) in enumerate(results.items()):
        values = [metrics['Accuracy'], metrics['Precision'], metrics['Recall'], metrics['F1 Score']]
        ax.bar(x + i*width, values, width, label=model_name, color=colors[i])
    
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison - All Metrics')
    ax.set_xticks(x + width)
    ax.set_xticklabels(['Accuracy', 'Precision', 'Recall', 'F1 Score'])
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3)
    
    st.pyplot(fig)
    plt.close()
    
    # Individual Metric Comparison
    st.subheader("📉 Metric-by-Metric Analysis")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    
    for idx, (ax, metric) in enumerate(zip(axes.flat, metrics)):
        values = [results[model][metric] for model in models.keys()]
        bars = ax.bar(models.keys(), values, color=colors)
        ax.set_title(f'{metric} Comparison')
        ax.set_ylim(0, 1)
        ax.set_ylabel('Score')
        
        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    # Summary Analysis
    st.subheader("📝 Summary Analysis")
    
    st.write("""
    ### Model Characteristics:
    
    | Algorithm | Strengths | Weaknesses |
    |-----------|-----------|------------|
    | **KNN** | Simple, no training time, interpretable | Slow prediction, sensitive to irrelevant features |
    | **ANN** | Captures complex patterns, highly flexible | Requires more data, black-box nature |
    | **SVM** | Effective in high dimensions, memory efficient | Parameter tuning required, slower training |
    
    """)
    
    # Display best model details
    st.subheader("🏆 Recommended Model")
    
    if results_df.loc[best_f1_model, 'F1 Score'] == results_df.loc[best_accuracy_model, 'Accuracy']:
        recommended = best_accuracy_model
    else:
        recommended = best_f1_model
    
    st.info(f"""
    **{recommended}** is recommended for this diabetes prediction task based on:
    - Accuracy: {results_df.loc[recommended, 'Accuracy']:.2%}
    - Precision: {results_df.loc[recommended, 'Precision']:.2%}
    - Recall: {results_df.loc[recommended, 'Recall']:.2%}
    - F1 Score: {results_df.loc[recommended, 'F1 Score']:.2%}
    """)

else:
    st.info("👆 Click 'Run Comparison' in the sidebar to train and compare all three algorithms!")
    
    # Preview dataset
    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)
