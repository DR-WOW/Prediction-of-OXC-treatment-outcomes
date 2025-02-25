import streamlit as st
import joblib
import pandas as pd
import numpy as np
from PIL import Image
import shap
import matplotlib.pyplot as plt

# 允许加载高分辨率图片
Image.MAX_IMAGE_PIXELS = None

# Load the models
random_forest = joblib.load('Random Forest.pkl')
extra_trees = joblib.load('Extra Trees.pkl')

# Model dictionary
models = {
    'Random Forest (RF)': random_forest,
    'Extra Trees': extra_trees
}

# Title
st.title("Antiepileptic Drug (OXC) Treatment Outcome Prediction with SHAP Visualization")

# Description
st.write("""
This app predicts the likelihood of treatment outcomes based on input features.
Select one or more models, input feature values, and get predictions and SHAP explanations.
""")

# Sidebar for model selection with multi-select option
selected_models = st.sidebar.multiselect("Select models to use for prediction", list(models.keys()), default=list(models.keys()))

# Input fields for the features
st.sidebar.header("Enter the following feature values:")
AGE = st.sidebar.number_input("AGE", min_value=0.0, max_value=18.0, value=5.0)
WT = st.sidebar.number_input("Weight (WT)", min_value=0.0, max_value=200.0, value=20.0)
Daily_Dose = st.sidebar.number_input("Daily Dose (Daily_Dose)", min_value=0.0, max_value=4000.0, value=2000.0)
Single_Dose = st.sidebar.number_input("Single Dose (Single_Dose)", min_value=0.0, max_value=4000.0, value=450.0)
VPA = st.sidebar.selectbox("VPA (1 = Combined with VPA, 0 = Combined without VPA)", [0, 1])
Terms = st.sidebar.selectbox("Terms (1 = Outpatient, 0 = Hospitalized)", [0, 1])
Cmin = st.sidebar.number_input("Trough concentration (Cmin)", min_value=0.0, max_value=100.0, value=15.0)
DBIL = st.sidebar.number_input("Direct Bilirubin (DBIL)", min_value=0.0, max_value=1000.0, value=5.0)
TBIL = st.sidebar.number_input("Total Bilirubin (TBIL)", min_value=0.0, max_value=200.0, value=5.0)
ALT = st.sidebar.number_input("Alanine Aminotransferase (ALT)", min_value=0.0, max_value=200.0, value=20.0)
AST = st.sidebar.number_input("Aspartate Aminotransferase (AST)", min_value=0.0, max_value=500.0, value=20.0)
SCR = st.sidebar.number_input("Serum Creatinine (SCR)", min_value=0.0, max_value=200.0, value=35.0)
BUN = st.sidebar.number_input("Blood Urea Nitrogen (BUN)", min_value=0.0, max_value=200.0, value=5.0)
CLCR = st.sidebar.number_input("Creatinine Clearance Rate (CLCR)", min_value=0.0, max_value=500.0, value=90.0)
HGB = st.sidebar.number_input("Hemoglobin (HGB)", min_value=0.0, max_value=500.0, value=120.0)
HCT = st.sidebar.number_input("Hematocrit (HCT)", min_value=0.0, max_value=200.0, value=35.0)
MCH = st.sidebar.number_input("Mean Corpuscular Hemoglobin (MCH)", min_value=0.0, max_value=1000.0, value=30.0)
MCHC = st.sidebar.number_input("Mean Corpuscular Hemoglobin Concentration (MCHC)", min_value=0.0, max_value=500.0, value=345.0)

# 获取模型训练时的特征名称
feature_names = random_forest.feature_names_in_
input_data = pd.DataFrame({
    feature: [value] for feature, value in zip(feature_names, [AGE, WT, Daily_Dose, Single_Dose, VPA, Terms, Cmin, DBIL, TBIL, ALT, AST, SCR, BUN, CLCR, HGB, HCT, MCH, MCHC])
})

# Add a predict button
if st.sidebar.button("Predict"):
    # Display predictions and probabilities for selected models
    for model_name in selected_models:
        model = models[model_name]
        prediction = model.predict(input_data)[0]
        predicted_proba = model.predict_proba(input_data)[0]

        # Display the prediction and probabilities for each selected model
        st.write(f"## Model: {model_name}")
        st.write(f"**Prediction**: {'Good Responder' if prediction == 1 else 'Poor Responder'}")
        st.write("**Prediction Probabilities**")
        probability_good = predicted_proba[1] * 100
        probability_poor = predicted_proba[0] * 100
        st.write(f"Based on feature values, predicted possibility of Good Responder is {probability_good:.2f}%")
        st.write(f"Based on feature values, predicted possibility of Poor Responder is {probability_poor:.2f}%")

        # SHAP 瀑布图
        st.write("### SHAP Waterfall Plot")
        explainer = shap.Explainer(model)
        shap_values = explainer(input_data)
        shap.plots.waterfall(shap_values[0], max_display=10)
        st.pyplot()
