# Hp.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(page_title="House Price Predictor", layout="wide")
st.title("🏠 House Price Prediction")

# ---------------------------
# Load Saved Objects
# ---------------------------
encoder = joblib.load("encoder.pkl")
model = joblib.load("model.pkl")
feature_order = joblib.load("feature_order.pkl")    # load column order from training

# SAME categorical columns as training
object_cols = ['MSZoning', 'LotConfig', 'BldgType', 'Exterior1st']

# ---------------------------
# Input UI
# ---------------------------
col1, col2, col3 = st.columns(3)

with col1:
    MSZoning = st.selectbox("MSZoning", ["RL","RM","FV","RH"])
    LotArea = st.number_input("LotArea (sq ft)", min_value=0, value=7000)
    LotConfig = st.selectbox("LotConfig", ["Inside","Corner","CulDSac","FR2","FR3"])

with col2:
    BldgType = st.selectbox("BldgType", ["1Fam","2fmCon","Duplex","Twnhs","TwnhsE"])
    OverallCond = st.slider("OverallCond", 1, 10, 5)
    YearBuilt = st.number_input("YearBuilt", 1800, 2025, 2000)

with col3:
    YearRemodAdd = st.number_input("YearRemodAdd", 1800, 2025, 2000)
    BsmtFinSF2 = st.number_input("BsmtFinSF2", 0, 2000, 0)
    TotalBsmtSF = st.number_input("TotalBsmtSF", 0, 3000, 800)
    Exterior1st = st.selectbox("Exterior1st", ["VinylSd","MetalSd","Wd Sdng","HdBoard","Plywood"])

st.write("---")

# ---------------------------
# Predict Button
# ---------------------------
if st.button("🔍 Predict"):
    
    input_dict = {
        "MSZoning": MSZoning,
        "LotArea": LotArea,
        "LotConfig": LotConfig,
        "BldgType": BldgType,
        "OverallCond": OverallCond,
        "YearBuilt": YearBuilt,
        "YearRemodAdd": YearRemodAdd,
        "BsmtFinSF2": BsmtFinSF2,
        "TotalBsmtSF": TotalBsmtSF,
        "Exterior1st": Exterior1st
    }

    df_input = pd.DataFrame([input_dict])

    # Encode categorical columns
    encoded = encoder.transform(df_input[object_cols])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out())

    # Combine numeric + encoded
    X_input = pd.concat([df_input.drop(object_cols, axis=1), encoded_df], axis=1)

    # Align feature order to match training
    X_input = X_input.reindex(columns=feature_order, fill_value=0)

    pred = model.predict(X_input)[0]
    st.success(f"💰 Predicted Sale Price: ${pred:,.2f}")

# ---------------------------
# File Upload (CSV & Excel)
# ---------------------------
st.write("### 📁 Upload CSV or Excel file for batch prediction")
uploaded = st.file_uploader("Upload File", type=["csv", "xlsx", "xls"])

if uploaded is not None:

    file_name = uploaded.name

    # Read file
    if file_name.endswith(".csv"):
        df_upload = pd.read_csv(uploaded)
    else:
        df_upload = pd.read_excel(uploaded)

    # Remove unwanted columns (Id, SalePrice)
    for bad_col in ["Id", "SalePrice"]:
        if bad_col in df_upload.columns:
            df_upload = df_upload.drop(bad_col, axis=1)

    # Encode categorical
    encoded = encoder.transform(df_upload[object_cols])
    enc_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(), index=df_upload.index)

    # Merge numeric + encoded
    X_upload = pd.concat([df_upload.drop(object_cols, axis=1), enc_df], axis=1)

    # Align features
    X_upload = X_upload.reindex(columns=feature_order, fill_value=0)

    # Predict
    preds = model.predict(X_upload)
    df_upload["PredictedPrice"] = preds

    st.dataframe(df_upload.head())

    # Download button
    st.download_button(
        "⬇ Download Predictions CSV",
        df_upload.to_csv(index=False).encode("utf-8"),
        "house_price_predictions.csv",
        "text/csv"
    )

