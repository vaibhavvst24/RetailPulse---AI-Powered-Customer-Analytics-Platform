import streamlit as st
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from pathlib import Path

def run():
    # ---------------------------------
    # CONFIG & PATHS
    # ---------------------------------
    # Assuming this file is in RetailPulse/app/pages/
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    MODEL_PATH = BASE_DIR / "models" / "churn_model.pkl"
    SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"

    st.title("Customer Churn Prediction")

    tab_pred, = st.tabs(["Prediction"])

    # -------------------------------------------------
    # PREDICTION SECTION
    # -------------------------------------------------
    try:
        if not MODEL_PATH.exists():
            st.warning("Model file not found")
            model = None
        else:
            model = joblib.load(MODEL_PATH)
    except Exception as e:
        st.warning(f"Error loading model: {e}")
        model = None

    with tab_pred:
        if model is None:
            st.warning("Prediction model is not available.")
            return

        st.header("Customer Churn Prediction")

        # ---------------- INPUTS ----------------
        col_inp1, col_inp2 = st.columns(2)
        with col_inp1:
            age = st.slider("Age", 18, 80, 30)
            tenure = st.slider("Tenure (months)", 1, 72, 12)
        with col_inp2:
            usage = st.slider("Usage Frequency", 1, 100, 50)
            total_spend = st.number_input("Total Spend", 100, 100000, 1000)

        gender = st.selectbox("Gender", ["Male", "Female"])
        subscription = st.selectbox("Subscription Type", ["Standard", "Premium"])
        contract = st.selectbox("Contract Length", ["Monthly", "Quarterly"])

        # ---------------- FEATURE SETUP ----------------
        if hasattr(model, 'feature_names_in_'):
            feature_names = model.feature_names_in_
            input_df = pd.DataFrame(columns=feature_names)
            input_df.loc[0] = 0

            # ---------------- RAW FEATURES ----------------
            if "Age" in feature_names: input_df.at[0, "Age"] = age
            if "Tenure" in feature_names: input_df.at[0, "Tenure"] = tenure
            if "Usage Frequency" in feature_names: input_df.at[0, "Usage Frequency"] = usage
            if "Total Spend" in feature_names: input_df.at[0, "Total Spend"] = total_spend
            if "Gender" in feature_names: input_df.at[0, "Gender"] = 1 if gender == "Male" else 0

            # ---------------- CATEGORICAL OHE ----------------
            if f"Subscription Type_{subscription}" in feature_names:
                input_df.at[0, f"Subscription Type_{subscription}"] = 1
            if f"Contract Length_{contract}" in feature_names:
                input_df.at[0, f"Contract Length_{contract}"] = 1

            # ---------------- TENURE GROUP ----------------
           # tenure group logic (MUST MATCH TRAINING)
            if tenure <= 12:
                input_df["tenure_group_1-2yr"] = 0
                input_df["tenure_group_2-4yr"] = 0
                input_df["tenure_group_4+yr"] = 0
            elif tenure <= 24:
                input_df["tenure_group_1-2yr"] = 1
            elif tenure <= 48:
                input_df["tenure_group_2-4yr"] = 1
            else:
                input_df["tenure_group_4+yr"] = 1

            # ---------------- ALIGN & SCALE ----------------
            input_df = input_df[feature_names]

            if SCALER_PATH.exists():
                try:
                    scaler = joblib.load(SCALER_PATH)
                    input_df_scaled = scaler.transform(input_df)
                except:
                    input_df_scaled = input_df
            else:
                input_df_scaled = input_df

            # ---------------- PREDICTION ----------------
            if st.button("Predict Churn Risk", key="predict_btn"):
                prediction = model.predict(input_df_scaled)[0]
                prob = model.predict_proba(input_df_scaled)[0][1]

                st.subheader("Prediction Results")
                st.metric("Churn Probability", f"{prob*100:.2f}%")
                st.progress(float(prob))

                if prob < 0.3:
                    risk, color, msg = "Low Risk", "green", "Customer is stable and unlikely to churn."
                    action = "Maintain engagement. No immediate action needed."
                elif prob < 0.7:
                    risk, color, msg = "Medium Risk", "orange", "Customer shows moderate churn signals."
                    action = "Offer discounts or personalized engagement."
                else:
                    risk, color, msg = "High Risk", "red", "Customer is very likely to churn."
                    action = "Immediate retention action required (offers, calls, support)."

                if risk == "High Risk":
                    st.error(f"⚠️ {risk}")
                elif risk == "Medium Risk":
                    st.warning(f"⚠️ {risk}")
                else:
                    st.success(f"✅ {risk}")

                st.write(f"**Insight:** {msg}")
                st.write(f"**Recommended Action:** {action}")
        else:
            st.error("Model features could not be identified.")

if __name__ == "__main__":
    run()