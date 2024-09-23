import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Load the heart disease dataset
df = pd.read_csv(r"C:\Users\seshu\Desktop\ml00009\heart-disease.csv")  # Replace with your dataset file

# Feature Selection
features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
target = 'target'  # The target variable is whether heart disease is present (1) or not (0)

# Split data into features (X) and target (y)
X = df[features]
y = df[target]

# Normalize the feature data using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train models
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_scaled, y)

lr_model = LogisticRegression()
lr_model.fit(X_scaled, y)

# Streamlit app title
st.title("Heart Disease Prediction")

# Instructions for user input
st.subheader("Enter patient details to predict the presence of heart disease:")

# Create form to take user input for features
with st.form("prediction_form"):
    age = st.number_input("Age", min_value=1, step=1)
    sex = st.selectbox("Sex (1 = Male, 0 = Female)", [1, 0])
    cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure (in mm Hg)", min_value=0, step=1)
    chol = st.number_input("Serum Cholestrol (in mg/dl)", min_value=0, step=1)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False)", [1, 0])
    restecg = st.selectbox("Resting Electrocardiographic Results (0-2)", [0, 1, 2])
    thalach = st.number_input("Maximum Heart Rate Achieved", min_value=0, step=1)
    exang = st.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", [1, 0])
    oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, step=0.1)
    slope = st.selectbox("Slope of the Peak Exercise ST Segment (0-2)", [0, 1, 2])
    ca = st.number_input("Number of Major Vessels (0-3)", min_value=0, max_value=3, step=1)
    thal = st.selectbox("Thalassemia (0 = Normal, 1 = Fixed Defect, 2 = Reversible Defect)", [0, 1, 2])
    
    # Model selection
    model_choice = st.selectbox("Choose Classification Model", ("Random Forest", "Logistic Regression"))
    
    # Submit button
    submitted = st.form_submit_button("Predict")

# If the user submits the form, perform the prediction
if submitted:
    # Prepare the input data
    user_input = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    user_input_scaled = scaler.transform(user_input)
    
    # Predict based on chosen model
    if model_choice == "Random Forest":
        prediction = rf_model.predict(user_input_scaled)[0]
    elif model_choice == "Logistic Regression":
        prediction = lr_model.predict(user_input_scaled)[0]
    
    # Display the prediction result
    if prediction == 1:
        st.subheader("The model predicts the patient has heart disease.")
    else:
        st.subheader("The model predicts the patient does not have heart disease.")