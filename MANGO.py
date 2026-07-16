import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(
    page_title="Mango Disease Forecast System",
    page_icon="🥭",
    layout="wide"
)

def set_bg():
    st.markdown("""
    <style>

    .stApp {
        background: linear-gradient(
            rgba(245,255,245,0.92),
            rgba(230,255,230,0.92)
        );
    }

    .main-title {
        text-align:center;
        color:#1B5E20;
        font-size:42px;
        font-weight:bold;
    }

    .sub-title {
        text-align:center;
        color:#2E7D32;
        font-size:20px;
    }

    .card {
        background:white;
        padding:20px;
        border-radius:15px;
        box-shadow:0px 4px 10px rgba(0,0,0,0.15);
        margin-bottom:15px;
    }

    </style>
    """, unsafe_allow_html=True)

set_bg()

menu = st.sidebar.selectbox(
    "Navigation",
    [
        "🏠 Home",
        "🦠 Disease Description",
        "📊 Disease Prediction",
        "👨‍💻 About Us"
    ]
)

if menu == "🏠 Home":

    st.title("🥭 Mango Disease Forecast System")

    st.markdown("""
    ## Welcome

    This application forecasts major mango diseases one week in advance
    using weather parameters and machine learning models.

    ### Objectives

    ✅ Early disease warning

    ✅ Weather-based forecasting

    ✅ Risk assessment

    ✅ Decision support for farmers

    ### Diseases Covered

    - Leaf Anthracnose
    - Black Banded
    - Red Rust
    - Die Back
    - Sooty Mould
    """)
elif menu == "🦠 Disease Description":

    st.title("🦠 Mango Disease Description")

    disease_info = st.selectbox(
        "Select Disease",
        [
            "Leaf Anthracnose",
            "Black Banded",
            "Red Rust",
            "Die Back",
            "Sooty Mould"
        ],
        key="disease_description"
    )

    if disease_info == "Leaf Anthracnose":
       
        st.markdown("""
        ### Leaf Anthracnose

        **Pathogen:** Colletotrichum gloeosporioides

        **Symptoms**
        - Dark brown spots
        - Leaf blight
        - Fruit rot
        """)

    elif disease_info == "Black Banded":

        st.markdown("""
        ### Black Banded

        **Symptoms**
        - Black lesions on branches
        - Twig drying
        - Reduced vigour
        """)

    elif disease_info == "Red Rust":

        st.markdown("""
        ### Red Rust

        **Pathogen:** Cephaleuros virescens

        **Symptoms**
        - Orange-red velvety spots on leaves
        - Reduced photosynthetic activity
        - Premature leaf fall
        """)

    elif disease_info == "Die Back":

        st.markdown("""
        ### Die Back

        **Pathogen:** Lasiodiplodia theobromae

        **Symptoms**
        - Drying of twigs from tip downward
        - Browning and death of branches
        - Gum exudation
        """)

    elif disease_info == "Sooty Mould":
      
        st.markdown("""
        ### Sooty Mould

        **Causal Organism:** Capnodium spp.

        **Symptoms**
        - Black soot-like growth on leaves
        - Reduced photosynthesis
        - Poor fruit appearance
        """)

elif menu == "📊 Disease Prediction":

    st.title("📊 Disease Prediction")

    disease = st.selectbox(
        "Select Disease",
        [
            "Leaf Anthracnose",
            "Black Banded",
            "Red Rust",
            "Die Back",
            "Sooty Mould"
        ],
        key="disease_prediction"
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Weather Parameters")

        rf = st.number_input(
            "Rainfall (RF) mm",
            min_value=0.0,
            value=25.0
        )

        rd = st.number_input(
            "Rainy Days (RD)",
            min_value=0,
            value=2
        )

        rh = st.number_input(
            "Humidity (RH) %",
            min_value=0.0,
            max_value=100.0,
            value=80.0
        )

        tmax = st.number_input(
            "Maximum Temperature (°C)",
            value=34.0
        )

        tmin = st.number_input(
            "Minimum Temperature (°C)",
            value=25.0
        )

    with col2:
        st.subheader("Field Parameters")

        current_disease = st.number_input(
            "Current Disease Severity (%)",
            min_value=0.0,
            max_value=100.0,
            value=15.0
        )

        week = st.number_input(
            "Week Number (1-52)",
            min_value=1,
            max_value=52,
            value=24
        )

    model_files = {
        "Leaf Anthracnose": "LEAF ANTHRACNOSE_farmer_forecast.pkl",
        "Black Banded": "BLACK_BANDED_farmer_forecast.pkl",
        "Red Rust": "RED RUST_farmer_forecast.pkl",
        "Die Back": "DIE BACK_farmer_forecast.pkl",
        "Sooty Mould": "SOOTY MOULD_farmer_forecast.pkl"
    }

    if st.button("Forecast Disease Risk", key="forecast_button"):

        try:

            T_avg = (tmax + tmin) / 2
            T_Range = tmax - tmin

            week_sin = np.sin(2 * np.pi * week / 52)
            week_cos = np.cos(2 * np.pi * week / 52)

            input_data = pd.DataFrame([[
                rf,
                rd,
                rh,
                tmax,
                tmin,
                T_avg,
                T_Range,
                current_disease,
                week_sin,
                week_cos
            ]], columns=[
                "RF",
                "RD",
                "RH",
                "T_MAX",
                "T_MIN",
                "T_avg",
                "T_Range",
                "DISEASE",
                "week_sin",
                "week_cos"
            ])

            model_file = model_files[disease]

            if not os.path.exists(model_file):
                st.error(f"Model file not found: {model_file}")
                st.stop()

            model = joblib.load(model_file)

            forecast = float(model.predict(input_data)[0])

            if forecast < 20:
                risk = "🟢 Low"
                advice = "Routine monitoring is sufficient."

            elif forecast < 40:
                risk = "🟡 Moderate"
                advice = "Inspect orchard regularly."

            elif forecast < 60:
                risk = "🟠 High"
                advice = "Start disease management measures."

            else:
                risk = "🔴 Epidemic"
                advice = "Immediate control measures required."

            st.success("Forecast Completed")

            st.metric(
                "Predicted Disease Severity (%)",
                f"{forecast:.2f}"
            )

            st.subheader("Risk Assessment")
            st.write(f"**Risk Level:** {risk}")
            st.write(f"**Recommendation:** {advice}")

        except Exception as e:
            st.error(str(e))

elif menu == "👨‍💻 About Us":

    st.title("👨‍💻 About Us")

    st.markdown("""
    ## Mango Disease Forecast System

    This application was developed for forecasting major mango diseases
    using weather data and machine learning.

    ### Developer

    **Mr. Kathirvel M.**

    PG Student of Agricultural Statistics

    Department of Physical Sciences and Information Technology

    Agricultural Engineering College and Research Institute

    Tamil Nadu Agricultural University,Coimbatore, Tamil Nadu 641003, India

    ### Study Area

    Theni District, Tamil Nadu, India

    ### Technologies Used

    - Python
    - Streamlit
    - Machine Learning
    - XGBoost
    - Pandas
    - NumPy

    ### Purpose

    To provide an early warning system for mango disease outbreaks and
    support timely disease management decisions.
    """)