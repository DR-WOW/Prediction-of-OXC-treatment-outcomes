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
# ... (其他特征输入)

# 转换输入为模型预测格式
input_data = pd.DataFrame({
    'AGE': [AGE],
    'WT': [WT],
    # ... (其他特征)
})

# 预测按钮
if st.sidebar.button("Predict"):
    # 显示所选模型的预测和概率
    for model_name in selected_models:
        model = models[model_name]
        prediction = model.predict(input_data)[0]
        predicted_proba = model.predict_proba(input_data)[0]

        # 显示每个所选模型的预测和概率
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
            shap_values[class_index][0],  # 选择第一个样本的 SHAP 值
            input_data.iloc[0],  # 选择第一个样本的特征值
            matplotlib=True,
        )
        # 保存并显示 SHAP 图
        plt.savefig(f"{model_name}_shap_force_plot.png", bbox_inches='tight', dpi=1200)
        st.image(f"{model_name}_shap_force_plot.png")
