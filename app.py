import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# ------------------------------
# Page Config
# ------------------------------
st.set_page_config(page_title="Farmer Income Prediction", layout="wide")
st.title("🌾 L&T Farmer Income Prediction App")

st.write("""
Upload your Excel file to get predictions, explore dataset statistics, 
and visualize your results.
""")

# ------------------------------
# Load Model
# ------------------------------
@st.cache_resource
def load_model():
    return joblib.load("lightgbm_model.joblib")

# ------------------------------
# Read Excel
# ------------------------------
@st.cache_data
def read_excel(uploaded):
    xls = pd.ExcelFile(uploaded)
    return {
        "TrainData": xls.parse("TrainData"),
        "TestData": xls.parse("TestData"),
        "Dictionary": xls.parse("Dictionary")
    }

# ------------------------------
# File Upload
# ------------------------------
uploaded_file = st.file_uploader("📂 Upload your Excel file", type=["xlsx"])

if uploaded_file:
    data = read_excel(uploaded_file)
    train_df = data["TrainData"]
    test_df = data["TestData"]
    dictionary_df = data["Dictionary"]

    st.subheader("🔍 Training Data Preview")
    st.write(train_df.head())

    # ------------------------------
    # EDA
    # ------------------------------
    st.subheader("📊 Target Variable Distribution (Train)")
    fig, ax = plt.subplots()
    ax.hist(train_df["Target_Variable/Total Income"].dropna(), bins=50)
    st.pyplot(fig)

    # ------------------------------
    # Preprocessing
    # ------------------------------
    cols_to_drop = [
        'FarmerID', 'Location', 'Address type', 'Ownership', 'Avg_Disbursement_Amount_Bureau',
        'Rabi Seasons  Season Irrigated area in 2022',
        'Rabi Seasons Cropping density in 2022',
        'KO22-Village score based on socio-economic parameters (0 to 100)',
        'Village score based on socio-economic parameters (Non normalised)',
        'Rabi Seasons Seasonal average groundwater thickness (cm) in 2022',
        'Kharif Seasons  Seasonal average groundwater thickness (cm) in 2022',
    ]
    for col in cols_to_drop:
        if col in train_df.columns:
            train_df = train_df.drop(columns=[col])
        if col in test_df.columns:
            test_df = test_df.drop(columns=[col])

    # Encode categoricals
    categorical_cols = train_df.select_dtypes(include=['object']).columns.tolist()
    train_df_num = pd.get_dummies(train_df, columns=categorical_cols)
    test_df_num = pd.get_dummies(test_df, columns=categorical_cols)

    # Align test with train
    test_df_num = test_df_num.reindex(columns=train_df_num.columns, fill_value=0)

    # Drop target variable
    if "Target_Variable/Total Income" in test_df_num.columns:
        test_df_num = test_df_num.drop(columns=["Target_Variable/Total Income"])
    if "Target_Variable/Total Income" in train_df_num.columns:
        X_train = train_df_num.drop(columns=["Target_Variable/Total Income"])
    else:
        X_train = train_df_num

    # ------------------------------
    # Prediction
    # ------------------------------
    st.subheader("🤖 Run Predictions")
    if st.button("Run Model and Predict"):
        model = load_model()
        pred_log = model.predict(test_df_num)
        predictions = np.expm1(pred_log)  # if model was trained on log1p
        test_ids = test_df["FarmerID"] if "FarmerID" in test_df else range(len(predictions))

        result_df = pd.DataFrame({
            "FarmerID": test_ids,
            "Predicted_Total_Income": predictions
        })

        st.write(result_df.head())

        csv = result_df.to_csv(index=False).encode()
        st.download_button("📥 Download Predictions", csv, "lt_fincome_predictions.csv")

    # ------------------------------
    # Feature Importance
    # ------------------------------
    st.subheader("📌 Feature Importance (SHAP)")
    if st.button("Show SHAP Summary Plot"):
        model = load_model()
        explainer = shap.TreeExplainer(model)
        shap_sample = X_train.sample(min(1000, X_train.shape[0]), random_state=42)
        shap_values = explainer.shap_values(shap_sample)

        fig2 = plt.figure()
        shap.summary_plot(shap_values, shap_sample, show=False)
        st.pyplot(fig2)

else:
    st.info("👆 Please upload an Excel (.xlsx) file to begin.")

st.caption("💡 Tip: Upload the sample Excel data used in training for best results.")
