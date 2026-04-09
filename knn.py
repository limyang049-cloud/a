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

# Upload dataset
# Note: Ensure "dataset.csv" is in the same folder as this script!
try:
    df = pd.read_csv("dataset.csv")

    st.subheader("Dataset Preview")
    st.write(df.head())

    # Select target column
    target_column = st.selectbox("Select Target Column", df.columns)

    # Split features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Encode categorical data
    le = LabelEncoder()
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = le.fit_transform(X[col])

    if y.dtype == 'object':
        y = le.fit_transform(y)

    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Train-test split
    test_size = st.slider("Test Size (%)", 10, 50, 20)
    k_value = st.slider("Select K Value", 1, 15, 5)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size/100, random_state=42
    )

    # Train model
    model = KNeighborsClassifier(n_neighbors=k_value)
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluation
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    st.subheader("Model Evaluation")
    col1, col2 = st.columns(2)
    col1.metric("Accuracy", f"{acc:.4f}")
    col1.metric("Precision", f"{precision:.4f}")
    col2.metric("Recall", f"{recall:.4f}")
    col2.metric("F1 Score", f"{f1:.4f}")

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    st.pyplot(fig)

    # Prediction section
    st.subheader("Make Prediction")
    input_data = []
    for col in df.drop(columns=[target_column]).columns:
        value = st.number_input(f"Enter {col}", value=0.0)
        input_data.append(value)

    if st.button("Predict"):
        input_array = np.array(input_data).reshape(1, -1)
        input_array = scaler.transform(input_array)
        prediction = model.predict(input_array)
        st.success(f"Predicted Class: {prediction[0]}")

except FileNotFoundError:
    st.error("Error: 'dataset.csv' not found. Please place the CSV file in the same folder as this script.")
