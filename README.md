# AI2

# 🩺 Diabetes Prediction System

An interactive Machine Learning web application built with Python and Streamlit. This system predicts the risk of diabetes in patients using three distinct classification algorithms: **K-Nearest Neighbors (KNN)**, **Support Vector Machine (SVM)**, and an **Artificial Neural Network (ANN)**.

---

## 🌟 Project Features

This application consolidates individual machine learning scripts into a single, cohesive web interface with three main sections:

1. **🏠 Home & Analytics:** Provides an exploratory overview of the dataset, including patient distribution and feature shapes.
2. **🧪 Make a Prediction:** A user-friendly interface to input new patient data. It features an **Ensemble Mode** that runs the patient's data through all three models simultaneously to provide a consensus prediction.
3. **📊 Model Comparison:** A dynamic testing environment where the user can adjust the **Test Size** and **Random Seed**. It generates real-time performance metrics (Accuracy, Precision, Recall, F1 Score) and visualizes the results using bar charts, comprehensive line plots, and side-by-side Confusion Matrices.

---

## 🛠️ Technologies Used

* **Language:** Python 3
* **Frontend/Framework:** Streamlit
* **Machine Learning:** Scikit-Learn (`KNeighborsClassifier`, `SVC`, `MLPClassifier`)
* **Data Manipulation:** Pandas, NumPy
* **Data Visualization:** Matplotlib, Seaborn

---

## 📂 Project Structure

Ensure the following files are in the same directory before running the application:

* `app.py`: The main Streamlit application containing the UI and consolidated model logic.
* `dataset.csv`: The diabetes dataset (NIDDK). *Note: The code automatically handles the 38 lines of header text in this specific file.*
* `requirements.txt`: List of dependencies required to run the project.

---

## 🚀 How to Run the Application

Follow these steps to run the project locally on your machine:

**Step 1: Open your terminal**
Navigate to the directory where the project files are saved.

**Step 2: Install required packages**
Ensure you have Python installed. Install the necessary libraries using the `requirements.txt` file:
```bash
pip install -r requirements.txt

streamlit run app.py
