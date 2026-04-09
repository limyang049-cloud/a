import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

st.title("SVM Diabetes Classification Results")

# --- 1. Data Pre-processing ---
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
           'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']

# Loading data and skipping the 38 lines of header text
try:
    df = pd.read_csv('dataset.csv', names=columns, skiprows=38)
    
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    # Split and Scale
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # --- 2. Model Training (SVM) ---
    svm_model = SVC(kernel='rbf', random_state=42)
    svm_model.fit(X_train, y_train)
    y_pred = svm_model.predict(X_test)

    # --- 3. Display Metrics in Streamlit ---
    st.subheader("Evaluation Metrics")
    
    # Using columns to make it look professional
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2%}")
    col2.metric("Precision", f"{precision_score(y_test, y_pred):.2f}")
    col3.metric("Recall", f"{recall_score(y_test, y_pred):.2f}")
    col4.metric("F1 Score", f"{f1_score(y_test, y_pred):.2f}")

    st.write("### Dataset Preview (Processed)")
    st.dataframe(df.head())

except FileNotFoundError:
    st.error("Make sure 'dataset.csv' is uploaded to your GitHub repository!")
