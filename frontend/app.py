import streamlit as st
import sys
import os

# Page config
st.set_page_config(
    page_title="Manufacturing Output Predictor",
    page_icon="🏭",
    layout="centered"
)

# Hide Streamlit menu & footer
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stNumberInput > div > div > input {
        background-color: #f5f7fa;
    }
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Backend import
sys.path.append(os.path.abspath('../backend'))
from main import predict_output

# ===== HEADER =====
st.markdown(
    """
    <h1 style='text-align: center; color: #1F618D;'>
    🏭 Manufacturing Equipment Output Prediction
    </h1>
    <p style='text-align: center; font-size:17px;'>
    Predict hourly production output using Linear Regression
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)

# ===== INPUT SECTION =====
st.markdown("### ⚙️ Machine Parameters")

col1, col2 = st.columns(2)

with col1:
    inj_temp = st.number_input("Injection Temperature", min_value=0.0)
    inj_press = st.number_input("Injection Pressure", min_value=0.0)
    cycle_time = st.number_input("Cycle Time", min_value=0.0)
    cooling_time = st.number_input("Cooling Time", min_value=0.0)
    viscosity = st.number_input("Material Viscosity", min_value=0.0)
    ambient_temp = st.number_input("Ambient Temperature", min_value=0.0)
    machine_age = st.number_input("Machine Age", min_value=0.0)

with col2:
    operator_exp = st.number_input("Operator Experience", min_value=0.0)
    maintenance = st.number_input("Maintenance Hours", min_value=0.0)
    ratio = st.number_input("Temperature Pressure Ratio", min_value=0.0)
    total_cycle = st.number_input("Total Cycle Time", min_value=0.0)
    eff_score = st.number_input("Efficiency Score", min_value=0.0)
    utilization = st.number_input("Machine Utilization", min_value=0.0)

st.markdown("<br>", unsafe_allow_html=True)

# ===== PREDICT BUTTON =====
if st.button("🔍 Predict Output", use_container_width=True):
    features = [
        inj_temp, inj_press, cycle_time, cooling_time,
        viscosity, ambient_temp, machine_age, operator_exp,
        maintenance, ratio, total_cycle, eff_score, utilization
    ]

    result = predict_output(features)

    st.markdown("### 📊 Prediction Result")

    st.success(f"""
    🏭 **Estimated Production Output:**  
    ### {result:.2f} Parts per Hour
    """)
