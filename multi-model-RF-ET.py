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
This app predicts the likelihood of heart disease based on input features.
Select one or more models, input feature values, and get predictions and probability estimates.
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
Terms = st.sidebar.selectbox("Terms (1 = Outpatient, 0 = Be hospitalized)", [0, 1])
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

# 确保 input_data 的列名与模型训练时的特征名称一致
expected_feature_names = ['AGE', 'WT', 'Daily_Dose', 'Single_Dose', 'VPA', 'Terms', 'Cmin', 'DBIL', 'TBIL', 'ALT', 'AST', 'SCR', 'BUN', 'CLCR', 'HGB', 'HCT', 'MCH', 'MCHC']
input_data = pd.DataFrame({
    feature: [value] for feature, value in zip(expected_feature_names, [AGE, WT, Daily_Dose, Single_Dose, VPA, Terms, Cmin, DBIL, TBIL, ALT, AST, SCR, BUN, CLCR, HGB, HCT, MCH, MCHC])
})

# Add a predict button
if st.sidebar.button("Predict"):
    # Display predictions and probabilities for selected models
    for model_name in selected_models:
        model = models[model_name]
        
        # 检查输入数据的列名
        st.write(f"Input Data Columns for {model_name}: {input_data.columns.tolist()}")  # Debug output
        
        # 检查列名是否匹配模型特征
        if set(input_data.columns) != set(expected_feature_names):
            st.error("The input feature names do not match the expected feature names.")
            continue  # 跳过这个模型的预测
        
        # Make sure that input data is in the correct format
        try:
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

            # 显示预测结果，使用 Matplotlib 渲染指定字体
            text = f"Based on feature values, predicted possibility of good responder is {probability_good:.2f}%"
            fig, ax = plt.subplots(figsize=(8, 1))
            ax.text(
                0.5, 0.5, text,
                fontsize=16,
                ha='center', va='center',
                fontname='Times New Roman',
                transform=ax.transAxes
            )
            ax.axis('off')
            plt.savefig("prediction_text.png", bbox_inches='tight', dpi=300)
            st.image("prediction_text.png")

            # 计算 SHAP 值
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_data)

            # 生成 SHAP 力图
            class_index = prediction  # 当前预测类别
            shap_fig = shap.force_plot(
                explainer.expected_value[class_index],
                shap_values[class_index],
                input_data,
                matplotlib=True,
            )
            
            # 显示 SHAP 力图
            st.subheader(f"SHAP Force Plot for {model_name}")
            plt.savefig(f"shap_force_plot_{model_name}.png", bbox_inches='tight', dpi=1200)
            st.image(f"shap_force_plot_{model_name}.png")
        
        except ValueError as e:
            st.error(f"Error predicting with {model_name}: {e}")
