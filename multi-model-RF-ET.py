import streamlit as st
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

# 允许加载高分辨率图片
Image.MAX_IMAGE_PIXELS = None

# 加载模型
random_forest = joblib.load('Random Forest.pkl')
extra_trees = joblib.load('Extra Trees.pkl')

# 模型字典
models = {
    'Random Forest (RF)': random_forest,
    'Extra Trees': extra_trees
}

# 标题
st.title("Antiepileptic Drug (OXC) Treatment Outcome Prediction with SHAP Visualization")

# 描述
st.write("""
This app predicts the likelihood of heart disease based on input features.
Select one or more models, input feature values, and get predictions and probability estimates.
""")

# 侧边栏模型选择，支持多选
selected_models = st.sidebar.multiselect("Select models to use for prediction", list(models.keys()), default=list(models.keys()))

# 输入字段，用于输入特征值
st.sidebar.header("Enter the following feature values:")
AGE = st.sidebar.number_input("AGE", min_value=0.0, max_value=18.0, value=5.0)
WT = st.sidebar.number_input("Weight (WT)", min_value=0.0, max_value=200.0, value=20.0)
Daily_Dose = st.sidebar.number_input("Daily Dose (Daily_Dose)", min_value=0.0, max_value=4000.0, value=2000.0)
Single_Dose = st.sidebar.number_input("Single Dose (Single_Dose)", min_value=0.0, max_value=4000.0, value=450.0)
VPA = st.sidebar.selectbox("VPA (1 = Combined with VPA, 0 = Combined without VPA)", [0, 1])
Terms = st.sidebar.selectbox("Terms(1 = Outpatient, 0 = Be hospitalized)", [0, 1])
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

# 转换输入为模型预测格式
input_data = pd.DataFrame({
    'AGE': [AGE],
    'WT': [WT],
    'Daily_Dose': [Daily_Dose],
    'Single_Dose': [Single_Dose],
    'VPA': [VPA],
    'Terms': [Terms],
    'Cmin': [Cmin],
    'DBIL': [DBIL],
    'TBIL': [TBIL],
    'ALT': [ALT],
    'AST': [AST],
    'SCR': [SCR],
    'BUN': [BUN],
    'CLCR': [CLCR],
    'HGB': [HGB],
    'HCT': [HCT],
    'MCH': [MCH],
    'MCHC': [MCHC]
})

# 添加预测按钮
if st.sidebar.button("Predict"):
    # 显示所选模型的预测和概率
    for model_name in selected_models:
        model = models[model_name]
        prediction = model.predict(input_data)[0]
        predicted_proba = model.predict_proba(input_data)[0]

        # 显示每个所选模型的预测和概率
        st.write(f"## Model: {model_name}")
        st.write(f"**Prediction**: {'Good Responder' if prediction ==1 else 'Poor Responder'}")
        st.write("**Prediction Probabilities**")
        probability_good = predicted_proba[1] * 100
        probability_poor = predicted_proba[0] * 100
        st.write(f"Based on feature values, predicted possibility of Good Responder is {probability_good:.2f}%")
        st.write(f"Based on feature values, predicted possibility of Poor Responder is {probability_poor:.2f}%")

        # 计算 SHAP 值
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_data)

        # 生成 SHAP 力图
        class_index = prediction  # 当前预测类别
        shap.force_plot(
            explainer.expected_value[class_index],
            shap_values[class_index][0],  # 选择第一个样本的 SHAP 值
            input_data.iloc[0],  # 选择第一个样本的特征值
            matplotlib=True,
        )
        # 保存并显示 SHAP 图
        plt.savefig(f"{model_name}_shap_force_plot.png", bbox_inches='tight', dpi=1200)
        st.image(f"{model_name}_shap_force_plot.png")
