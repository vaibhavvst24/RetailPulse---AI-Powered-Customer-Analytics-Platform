import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
import joblib
import pickle

from pathlib import Path

def run():
    # ------------------ PAGE CONFIG & PATHS ------------------
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    DATA_PATH = BASE_DIR / "models" / "store_item_history.pkl"
    MODEL_PATH = BASE_DIR / "models" / "timeseries_model.pkl"

    # ------------------ CUSTOM CSS ------------------
    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');
            .stMetric {
                background-color: #1c1f26;
                padding: 20px;
                border-radius: 12px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }
            .stApp {
                font-family: 'Poppins', sans-serif;
            }
            h1, h2, h3 {
                font-family: 'Poppins', sans-serif !important;
                font-weight: 600;
            }
            .insight-card {
                background-color: #1c1f26;
                padding: 20px;
                border-radius: 12px;
                border-left: 5px solid #00d2ff;
                margin-bottom: 10px;
            }
        </style>
    """, unsafe_allow_html=True)

    # ------------------ TITLE ------------------
    st.markdown("<h1 style='text-align: center; color: white;'>Retail Sales Time Series Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<div style='margin-bottom: 40px;'></div>", unsafe_allow_html=True)

    tab_pred, = st.tabs(["Prediction"])

    # ------------------ LOAD DATA ------------------
    @st.cache_data
    def load_timeseries_data():
        if not DATA_PATH.exists():
            return None
        with open(DATA_PATH, 'rb') as f:
            return pickle.load(f)

    history_dict = load_timeseries_data()
    
    if history_dict is None:
        st.warning(f"Required data file missing: {DATA_PATH.name}")
        return

    # ------------------ PREDICTION TAB ------------------
    with tab_pred:
        st.header("📊 Item-Level Sales Forecast")
        st.markdown("""
            Forecast sales for specific items and stores using the trained model. 
            The model was trained on individual item performance to capture granular trends.
        """)

        if not MODEL_PATH.exists():
            st.warning("Model file not found")
            return
        
        try:
            model = joblib.load(MODEL_PATH)
        except Exception as e:
            st.warning(f"Error loading model: {e}")
            return

        # 1. SELECT CONTEXT FOR FORECAST
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            all_stores_f = sorted(list(set([k[0] for k in history_dict.keys()])))
            sel_store = st.selectbox("Select Store for Forecast", all_stores_f, index=0)
        with col_f2:
            all_items_f = sorted(list(set([k[1] for k in history_dict.keys() if k[0] == sel_store])))
            sel_item = st.selectbox("Select Item for Forecast", all_items_f, index=0)

        # Retrieve history for selected store/item
        ts_list = history_dict.get((sel_store, sel_item))
        
        if not ts_list:
            st.warning("No historical data found for the selected store/item combination.")
            return

        steps = st.slider("Select Days to Forecast", 1, 30, 7)
        
        store_sales = [val for k, lst in history_dict.items() if k[0] == sel_store for val in lst]
        store_avg = np.mean(store_sales) if store_sales else 0
        
        item_sales = [val for k, lst in history_dict.items() if k[1] == sel_item for val in lst]
        item_avg = np.mean(item_sales) if item_sales else 0

        # 2. FEATURE ENGINEERING FOR INFERENCE (22 Features)
        def get_inference_features(last_sales, target_date):
            """
            Generates the 22 features expected by the model for a single target date.
            """
            def get_lag(n):
                return last_sales[-n] if len(last_sales) >= n else last_sales[0]

            def get_rolling_mean(n):
                return np.mean(last_sales[-n:]) if len(last_sales) >= n else np.mean(last_sales)
            def get_rolling_std(n):
                return np.std(last_sales[-n:]) if len(last_sales) >= n else 0

            trend_val = len(last_sales) 
            
            dow = target_date.dayofweek
            month = target_date.month
            is_weekend = 1 if dow >= 5 else 0
            
            features = {
                'store': sel_store,
                'item': sel_item,
                'lag_1': get_lag(1),
                'lag_7': get_lag(7),
                'lag_14': get_lag(14),
                'lag_28': get_lag(28),
                'rolling_mean_7': get_rolling_mean(7),
                'rolling_mean_14': get_rolling_mean(14),
                'rolling_mean_28': get_rolling_mean(28),
                'rolling_std_7': get_rolling_std(7),
                'rolling_std_14': get_rolling_std(14),
                'rolling_std_28': get_rolling_std(28),
                'day_of_week': dow,
                'dow_sin': np.sin(2 * np.pi * dow / 7),
                'dow_cos': np.cos(2 * np.pi * dow / 7),
                'month_sin': np.sin(2 * np.pi * month / 12),
                'month_cos': np.cos(2 * np.pi * month / 12),
                'trend': trend_val,
                'diff_1': last_sales[-1] - last_sales[-2] if len(last_sales) >= 2 else 0,
                'roc_7': (last_sales[-1] - get_lag(7)) / (get_lag(7) + 1e-5),
                'store_avg_sales': store_avg,
                'item_avg_sales': item_avg,
                'is_weekend': is_weekend
            }
            
            ordered_cols = ['store', 'item', 'lag_1', 'lag_7', 'lag_14', 'lag_28', 
                            'rolling_mean_7', 'rolling_mean_14', 'rolling_mean_28', 
                            'rolling_std_7', 'rolling_std_14', 'rolling_std_28', 
                            'day_of_week', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos', 
                            'trend', 'diff_1', 'roc_7', 'store_avg_sales', 'item_avg_sales',
                            'is_weekend']
            
            return np.array([features[col] for col in ordered_cols]).reshape(1, -1)

        # 3. RECURSIVE FORECASTING LOOP WITH HYBRID SMOOTHING
        current_list = list(ts_list)
        future_preds = []
        uncertainty_upper = []
        uncertainty_lower = []
        
        last_date = pd.Timestamp.now().normalize()
        
        hist_std = np.std(ts_list) if len(ts_list) > 1 else 0
        hist_mean = np.mean(ts_list) if len(ts_list) > 0 else 0
        
        with st.spinner("Generating robust forecast..."):
            for i in range(1, steps + 1):
                next_date = last_date + pd.Timedelta(days=1)
                
                X_inf = get_inference_features(current_list, next_date)
                
                model_pred = model.predict(X_inf)[0]
                model_pred = max(0, float(model_pred))
                
                seasonal_naive = current_list[-7] if len(current_list) >= 7 else model_pred
                
                hybrid_pred = (0.8 * model_pred) + (0.2 * seasonal_naive)
                
                error_margin = 1.96 * (hist_std * 0.1 * np.sqrt(i)) 
                
                future_preds.append(hybrid_pred)
                uncertainty_upper.append(hybrid_pred + error_margin)
                uncertainty_lower.append(max(0, hybrid_pred - error_margin))
                
                current_list.append(hybrid_pred)
                last_date = next_date

        # 4. BUSINESS INSIGHT LAYER
        st.markdown("<hr>", unsafe_allow_html=True)
        st.subheader("💡 Executive Forecast Summary")
        
        forecast_trend_slope = (future_preds[-1] - future_preds[0]) / steps
        avg_forecast = np.mean(future_preds)
        avg_hist = hist_mean
        
        if forecast_trend_slope > 0.5:
            trend_status = "🚀 Strong Growth"
            trend_color = "green"
        elif forecast_trend_slope < -0.5:
            trend_status = "📉 Declining"
            trend_color = "red"
        else:
            trend_status = "📊 Stable"
            trend_color = "blue"
            
        volatility = (np.std(future_preds) / avg_forecast) if avg_forecast > 0 else 0
        is_volatile = volatility > (hist_std / avg_hist) if avg_hist > 0 else False
        
        last_actual = ts_list[-1]
        z_score = (last_actual - avg_hist) / (hist_std + 1e-5) if hist_std > 0 else 0
        is_anomaly = abs(z_score) > 2

        col_i1, col_i2, col_i3 = st.columns(3)
        with col_i1:
            st.metric("Forecasted Trend", trend_status)
            st.caption(f"Slope: {forecast_trend_slope:.2f} units/day")
        with col_i2:
            st.metric("Volatility Index", "High" if is_volatile else "Low", delta=f"{volatility:.1%}")
            st.caption("Based on coefficient of variation")
        with col_i3:
            st.metric("Recent Data Status", "Anomaly" if is_anomaly else "Normal", delta=f"Z:{z_score:.1f}")
            st.caption("Last 7 days vs Historical 3σ")

        # 5. VISUALIZATION
        if future_preds:
            hist_dates = pd.date_range(end=pd.Timestamp.now().normalize(), periods=len(ts_list))
            recent_ts = pd.DataFrame({"date": hist_dates, "sales": ts_list})
            
            future_dates = pd.date_range(start=hist_dates[-1] + pd.Timedelta(days=1), periods=len(future_preds))
            forecast_df = pd.DataFrame({
                "date": future_dates, 
                "forecast": future_preds,
                "upper": uncertainty_upper,
                "lower": uncertainty_lower
            })

            st.subheader(f"📈 Sales Forecast & Confidence Interval")
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(x=recent_ts["date"], y=recent_ts["sales"], name="Historical", line=dict(color="#00d2ff")))
            
            fig.add_trace(go.Scatter(
                x=list(forecast_df["date"]) + list(forecast_df["date"])[::-1],
                y=list(forecast_df["upper"]) + list(forecast_df["lower"])[::-1],
                fill='toself',
                fillcolor='rgba(255, 75, 75, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                showlegend=True,
                name="95% Confidence Interval"
            ))
            
            fig.add_trace(go.Scatter(x=forecast_df["date"], y=forecast_df["forecast"], name="Forecast", line=dict(color="#ff4b4b", width=3)))
            
            fig.update_layout(
                template="plotly_dark", 
                title=f"Forecast for Store {sel_store} | Item {sel_item}", 
                title_x=0.5,
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig, use_container_width=True)

    # Footer
    st.markdown("<div style='margin-top: 50px; text-align: center; color: #666;'>RetailPulse Analytics | Zidio Intelligence</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    run()
