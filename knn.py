import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Title
st.title("KNN Classification App (Supervised Machine Learning)")

# --- 1. Data Pre-processing ---
# Define the column names based on the dataset description
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
           'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']

# Loading data and skipping the 38 lines of header text found in dataset.csv
try:
    df = pd.read_csv("dataset.csv", names=columns, skiprows=38)

    st.subheader("Dataset Preview")
    st.write(df.head())

    # --- 2. Feature Engineering ---
    # Target column is fixed as 'Outcome' for this specific dataset
    target_column = 'Outcome'
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- 3. Model Parameters ---
    st.sidebar.header("Model Settings")
    test_size = st.sidebar.slider("Test Size (%)", 10, 50, 20)
    k_value = st.sidebar.slider("Select K Value (Neighbors)", 1, 15, 5)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size/100, random_state=42
    )

    # --- 4. Train & Predict ---
    model = KNeighborsClassifier(n_neighbors=k_value)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # --- 5. Evaluation Metrics ---
    st.subheader("Model Evaluation")
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{acc:.2%}")
    col2.metric("Precision", f"{prec:.2%}")
    col3.metric("Recall", f"{rec:.2%}")
    col4.metric("F1 Score", f"{f1:.2%}")

    # --- 6. Visualizations ---
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig)

    # --- 7. Manual Prediction Section ---
    st.subheader("Predict for New Patient")
    input_data = []
    col_a, col_b = st.columns(2)
    
    for i, col in enumerate(X.columns):
        with col_a if i % 2 == 0 else col_b:
            val = st.number_input(f"Enter {col}", value=float(df[col].median()))
            input_data.append(val)

    if st.button("Run Prediction"):
        input_array = np.array(input_data).reshape(1, -1)
        input_array = scaler.transform(input_array)
        prediction = model.predict(input_array)
        
        if prediction[0] == 1:
            st.error("Result: Positive for Diabetes")
        else:
            st.success("Result: Negative for Diabetes")

except FileNotFoundError:
    st.error("Error: 'dataset.csv' not found. Please ensure it is in your GitHub repository.")
except Exception as e:
    st.error(f"An unexpected error occurred: {e}")
