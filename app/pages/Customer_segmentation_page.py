import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import joblib

from pathlib import Path

def run():
    #----------CONFIG & PATHS------------
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    MODEL_PATH = BASE_DIR / "models" / "customer_sales_model.pkl"
    SCALER_PATH = BASE_DIR / "models" / "customer_sales_scaler.pkl"

    #----------PREDICTION ASSETS------------
    try:
        if not MODEL_PATH.exists() or not SCALER_PATH.exists():
            st.warning("Required model files missing.")
            model, scaler = None, None
        else:
            model = joblib.load(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
    except Exception as e:
        st.warning(f"Error loading model files: {e}")
        model, scaler = None, None

    st.markdown("""
        <style>
            .main {
                background-color: #0E1117;
            }
            .stMetric {
                background-color: #1c1f26;
                padding: 5px;
                border-radius: 10px;
            }
        </style>
    """, unsafe_allow_html=True)

    # ----------------------------
    # Page Config
    # ----------------------------
    st.set_page_config(page_title="Customer Segmentation", layout="wide")
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');

        .stApp {
            font-family: 'Poppins', sans-serif;
        }

        h1 {
            font-family: 'Poppins', sans-serif !important;
            font-weight: 600;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1 style='text-align: center;'>🧑🏻Customer Sales Segmentation📈</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>Enter customer details to predict segment</h3>", unsafe_allow_html=True)

    # ----------------------------
    # Input Fields
    # ----------------------------
    recency = st.number_input("Recency (days since last purchase)", min_value=0)
    frequency = st.number_input("Frequency (number of transactions)", min_value=0)
    monetary = st.number_input("Monetary (total spending)", min_value=0.0)

    avg_order_value = st.number_input("Average Order Value", min_value=0.0)
    purchase_frequency = st.number_input("Purchase Frequency", min_value=0.0)
    total_quantity = st.number_input("Total Quantity Purchased", min_value=0)
    customer_lifetime = st.number_input("Customer Lifetime (days)", min_value=0)

    # ----------------------------
    # Prediction Button
    # ----------------------------
    if st.button("Predict Segment"):
        if model is None or scaler is None:
            st.error("Model or Scaler not loaded. Cannot predict.")
            return

        # Create feature array
        features = np.array([[
            recency,
            frequency,
            monetary,
            avg_order_value,
            purchase_frequency,
            total_quantity,
            customer_lifetime
        ]])

        # Scale features
        features_scaled = scaler.transform(features)

        # Predict cluster
        cluster = model.predict(features_scaled)[0]

        # ----------------------------
        # Map Cluster to Segment
        # (based on your earlier analysis)
        # ----------------------------
        cluster_labels = {
            0: "🛒 Regular Customer",
            1: "⭐ VIP Customer",
            2: "⚠️ At-Risk Customer",
            3: "💰 Low-Value Customer"
        }

        segment = cluster_labels.get(cluster, "Unknown")

        # ----------------------------
        # Output
        # ----------------------------
        st.success(f"Predicted Cluster: {cluster}")
        st.success(f"Customer Segment: {segment}")

        # ----------------------------
        # Business Insight
        # ----------------------------
        if cluster == 1:
            st.info("💎 High-value customer → Offer loyalty rewards & premium services.")
        elif cluster == 2:
            st.warning("⚠️ Customer is at risk → Provide discounts or re-engagement offers.")
        elif cluster == 0:
            st.info("🛍️ Regular customer → Encourage more purchases with offers.")
        else:
            st.info("💰 Low-value customer → Focus on awareness & engagement strategies.")

if __name__ == "__main__":
    run()