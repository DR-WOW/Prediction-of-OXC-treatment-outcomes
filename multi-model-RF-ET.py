import streamlit as st
import joblib
import pandas as pd
import numpy as np
import shap
import streamlit.components.v1 as components

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
# ... (other feature inputs)

# Convert inputs to DataFrame for model prediction
input_data = pd.DataFrame({
    'AGE': [AGE],
    'WT': [WT],
    # ... (other features)
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

        # Calculate SHAP values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_data)

        # Generate SHAP force plots for multiple samples and save as HTML
        for i in range(input_data.shape[0]):
            shap.force_plot(
                explainer.expected_value[prediction],
                shap_values[prediction][i],
                input_data.iloc[i],
                show=False,
                save_html=f"{model_name}_shap_force_plot_sample_{i}.html"
            )
            components.html(f"{model_name}_shap_force_plot_sample_{i}.html", height=500)  # 显示保存的 HTML 文件
