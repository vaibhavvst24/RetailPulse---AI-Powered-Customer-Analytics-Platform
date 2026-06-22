import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import os
import pulp
from pathlib import Path

def run():
    # ------------------ PAGE CONFIG & PATHS ------------------
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    BASE_DATA_PATH = BASE_DIR / "models" / "inventory_base.pkl"

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
            .debug-box {
                background-color: #12141a;
                padding: 10px;
                border: 1px solid #333;
                border-radius: 8px;
                font-family: monospace;
                font-size: 0.8rem;
                color: #888;
            }
        </style>
    """, unsafe_allow_html=True)

    # ------------------ TITLE ------------------
    st.markdown("<h1 style='text-align: center; color: white;'>📦 Inventory Management Analytics</h1>", unsafe_allow_html=True)
    st.markdown("<div style='margin-bottom: 40px;'></div>", unsafe_allow_html=True)

    # ------------------ LOAD DATA ------------------
    @st.cache_data
    def load_inventory_data():
        if not BASE_DATA_PATH.exists():
            return None
        return pd.read_pickle(BASE_DATA_PATH)

    df = load_inventory_data()

    if df is None:
        st.warning(f"Required model file missing: {BASE_DATA_PATH.name}")
        return

    # ------------------ SIDEBAR FILTERS ------------------
    st.sidebar.header("🔍 Filter Options")
    
    all_stores = sorted(df['Store_ID_enc'].unique())
    selected_store = st.sidebar.selectbox("Select Store ID", options=all_stores)
    
    categories = ['Electronics', 'Furniture', 'Groceries', 'Toys']
    selected_category = st.sidebar.selectbox("Select Category", ["All"] + categories)

    # Apply Filters
    mask = (df['Store_ID_enc'] == selected_store)
    
    if selected_category != "All":
        if 'category' in df.columns:
            mask &= (df['category'] == selected_category)
        elif 'Category' in df.columns:
            mask &= (df['Category'] == selected_category)

    # ------------------ TABS ------------------
    tab_pred, = st.tabs(["Demand Prediction"])

    with tab_pred:
        st.header("📦 Multi-Product Inventory Optimization")

        # ------------------ USER INPUT ------------------
        budget = st.sidebar.number_input(
            "💰 Budget",
            min_value=1000,
            max_value=1000000,
            value=50000,
            step=1000
        )

        holding_cost = 2
        stockout_cost = 25  # slightly higher → pushes fulfillment

        # ------------------ FILTER DATA ------------------
        agg = df[mask].copy()

        if agg.empty:
            st.warning("No data for selected filters")
            st.stop()

        # ------------------ INVENTORY POLICY ------------------
        if "needed" not in agg.columns:
            agg["needed"] = (agg["rop"] - agg["inventory"]).clip(lower=0)

        # Remove useless rows
        agg = agg[agg["needed"] > 0].copy()

        if agg.empty:
            st.success("✅ All products sufficiently stocked.")
            st.stop()

        # Reset index to ensure unique items for PuLP
        agg = agg.reset_index(drop=True)
        items = agg.index.tolist()

        # Ensure price exists for budget allocation
        if "price" not in agg.columns:
            if "Price" in agg.columns:
                agg["price"] = agg["Price"]
            else:
                # Recover price from priority if not explicitly provided: priority = needed / (price + 1)
                agg["price"] = np.where(agg["priority"] > 0, (agg["needed"] / agg["priority"]) - 1, 0).clip(min=0)

        # ------------------ OPTIMIZATION ------------------
        prob = pulp.LpProblem("Inventory_Optimization", pulp.LpMinimize)

        Q = pulp.LpVariable.dicts("order_qty", items, lowBound=0)
        U = pulp.LpVariable.dicts("unmet", items, lowBound=0)

        # Objective
        prob += pulp.lpSum([
            holding_cost * Q[i] +
            stockout_cost * U[i]
            for i in items
        ])

        # Constraints
        for i in items:
            prob += U[i] >= agg.loc[i, "needed"] - Q[i]

        # Budget constraint
        prob += pulp.lpSum([
            agg.loc[i, "price"] * Q[i]
            for i in items
        ]) <= budget

        # Solve
        prob.solve(pulp.PULP_CBC_CMD(msg=0))

        # ------------------ RESULTS ------------------
        agg["optimal_order"] = [Q[i].varValue or 0 for i in items]
        agg["unmet_after"] = [U[i].varValue or 0 for i in items]

        # ------------------ KPIs ------------------
        total_spend = (agg["optimal_order"] * agg["price"]).sum()
        total_unmet = agg["unmet_after"].sum()

        col1, col2, col3 = st.columns(3)
        col1.metric("💰 Total Spend", f"{int(total_spend):,}")
        col2.metric("⚠️ Unmet Demand", f"{int(total_unmet):,}")
        col3.metric("📦 Products Optimized", len(agg))

        # ------------------ STATUS ------------------
        if total_spend >= budget * 0.98:
            st.warning("⚠️ Budget fully utilized. Some demand may remain unmet.")
        else:
            st.success("✅ Budget sufficient to cover most demand.")

        # ------------------ DISPLAY ------------------
        st.subheader("📊 Optimization Results")

        st.dataframe(
            agg[[
                "Product",
                "mean_demand",
                "inventory",
                "rop",
                "needed",
                "optimal_order",
                "unmet_after",
                "priority"
            ]]
            .sort_values("priority", ascending=False)
            .round(2),
            use_container_width=True
        )

        # ------------------ TOP PRODUCTS ------------------
        st.subheader("🔥 Top 5 Critical Products")

        top5 = agg.sort_values("priority", ascending=False).head(5)

        st.dataframe(
            top5[[
                "Product",
                "needed",
                "optimal_order",
                "unmet_after"
            ]].round(2),
            use_container_width=True
        )

        # Footer
    st.markdown("<div style='margin-top: 50px; text-align: center; color: #666;'>RetailPulse Analytics | Zidio Intelligence</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    run()