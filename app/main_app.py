import streamlit as st

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="RetailPulse",
    layout="wide"
)

st.title("📊 RetailPulse AI Dashboard")

# ---------------- NAVIGATION ----------------
st.sidebar.title("Navigation")

page = st.sidebar.selectbox(
    "Select Module",
    [
        "Churn",
        "Time Series",
        "Segmentation",
        "Inventory"
    ]
)

# ---------------- ROUTING ----------------
if page == "Churn":
    from pages.Churn_page import run
    run()

elif page == "Time Series":
    from pages.Timeseries_page import run
    run()

elif page == "Segmentation":
    from pages.Customer_segmentation_page import run
    run()

elif page == "Inventory":
    from pages.inventory_page import run
    run()