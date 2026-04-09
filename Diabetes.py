import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# ===============================
# Load Model & Files
# ===============================
model = load_model("diabetes_ann_model.h5", compile=False)
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("features.pkl")

st.title("🧠 Diabetes Prediction App (ANN)")
st.write("Enter patient details below:")

# ===============================
# User Inputs
# ===============================
Pregnancies = st.number_input("Pregnancies", 0, 20)
Glucose = st.number_input("Glucose", 0.0, 200.0)
BloodPressure = st.number_input("Blood Pressure", 0.0, 150.0)
SkinThickness = st.number_input("Skin Thickness", 0.0, 100.0)
Insulin = st.number_input("Insulin", 0.0, 900.0)
BMI = st.number_input("BMI", 0.0, 70.0)
DPF = st.number_input("Diabetes Pedigree Function", 0.0, 3.0)
Age = st.number_input("Age", 1, 120)

# ===============================
# Feature Engineering
# ===============================
def create_features():
    epsilon = 1e-5

    BMI_Age = BMI * Age
    Glucose_BMI = Glucose * BMI
    Insulin_Glucose_Ratio = Insulin / (Glucose + epsilon)
    Pregnancy_Age_Ratio = min(Pregnancies / (Age + 1), 0.5)
    Insulin_log = np.log1p(Insulin)

    BMI_Category = 0 if BMI < 18.5 else 1 if BMI < 25 else 2 if BMI < 30 else 3
    Age_Group = 0 if Age < 30 else 1 if Age < 40 else 2 if Age < 50 else 3 if Age < 60 else 4
    Glucose_Level = 0 if Glucose < 100 else 1 if Glucose < 125 else 2
    Pregnancy_Risk = 0 if Pregnancy_Age_Ratio < 0.1 else 1 if Pregnancy_Age_Ratio < 0.3 else 2

    return {
        'Pregnancies': Pregnancies,
        'Glucose': Glucose,
        'BloodPressure': BloodPressure,
        'SkinThickness': SkinThickness,
        'Insulin': Insulin,
        'BMI': BMI,
        'DiabetesPedigreeFunction': DPF,
        'Age': Age,
        'BMI_Category': BMI_Category,
        'Age_Group': Age_Group,
        'Glucose_Level': Glucose_Level,
        'BMI_Age': BMI_Age,
        'Glucose_BMI': Glucose_BMI,
        'Insulin_Glucose_Ratio': Insulin_Glucose_Ratio,
        'Pregnancy_Age_Ratio': Pregnancy_Age_Ratio,
        'Pregnancy_Risk': Pregnancy_Risk,
        'Insulin_log': Insulin_log
    }

# ===============================
# Explanation Function
# ===============================
def explain_prediction(input_data):
    explanation = []

    if input_data['Glucose'] > 140:
        explanation.append("High glucose level increases diabetes risk")
    elif input_data['Glucose'] > 110:
        explanation.append("Moderately elevated glucose detected")

    if input_data['BMI'] > 30:
        explanation.append("High BMI (obesity) contributes to risk")
    elif input_data['BMI'] > 25:
        explanation.append("Slightly elevated BMI")

    if input_data['Age'] > 45:
        explanation.append("Older age increases risk")

    if input_data['DiabetesPedigreeFunction'] > 0.7:
        explanation.append("Strong family history detected")

    if input_data['Pregnancies'] > 4:
        explanation.append("Higher pregnancy count may increase risk")

    return explanation

# ===============================
# Prediction
# ===============================
if st.button("Predict"):
    input_data = create_features()

    input_df = pd.DataFrame([input_data])
    input_df = input_df[feature_names]

    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0][0]

    # ===============================
    # Output
    # ===============================
    st.subheader("📊 Prediction Result")
    st.write(f"Prediction Score: {prediction:.4f}")

    # Risk Level
    if prediction < 0.25:
        st.success(f"✅ Low Risk ({prediction:.2f})")
        st.write("Risk is minimal. Maintain a healthy lifestyle.")
    elif prediction < 0.50:
        st.warning(f"⚠️ Medium Risk ({prediction:.2f})")
        st.write("Moderate risk detected. Consider medical check-up.")
    else:
        st.error(f"🚨 High Risk ({prediction:.2f})")
        st.write("High probability of diabetes. Seek medical advice.")

    # ===============================
    # Explanation Panel
    # ===============================
    st.subheader("🧾 Explanation")

    explanations = explain_prediction(input_data)

    if explanations:
        for exp in explanations:
            st.write(f"- {exp}")
    else:
        st.write("No strong risk factors detected.")

    # ===============================
    # Risk Visualization
    # ===============================
    st.subheader("📈 Risk Score")
    st.progress(float(prediction))

else:
    st.info("Enter patient details and click Predict")
