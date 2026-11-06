
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Pakistan House Price Estimator",
    page_icon="🏠",
    layout="centered"
)

@st.cache_resource
def load_model():
    return joblib.load("pakistan_house_price_model (2).pkl")

model = load_model()

st.title("🏠 Pakistan AI House Price Estimator")
st.write("Estimate your house price using Machine Learning.")

cities = [
    "Lahore",
    "Karachi",
    "Islamabad",
    "Rawalpindi",
    "Faisalabad"
]

with st.form("prediction_form"):

    city = st.selectbox("Select City", cities)

    area = st.number_input(
        "House Area (Marla)",
        min_value=1.0,
        max_value=200.0,
        value=10.0
    )

    bedrooms = st.number_input(
        "Bedrooms",
        min_value=1,
        max_value=15,
        value=3
    )

    baths = st.number_input(
        "Bathrooms",
        min_value=1,
        max_value=12,
        value=3
    )

    latitude = st.number_input(
        "Latitude",
        value=31.5204,
        format="%.6f"
    )

    longitude = st.number_input(
        "Longitude",
        value=74.3587,
        format="%.6f"
    )

    submit = st.form_submit_button("Predict House Price")

if submit:

    input_data = pd.DataFrame([{
        "city": city,
        "area_marla": area,
        "bedrooms": bedrooms,
        "baths": baths,
        "latitude": latitude,
        "longitude": longitude
    }])

    prediction = model.predict(input_data)[0]

    st.success("Estimated House Price")

    st.subheader(f"PKR {prediction:,.0f}")

    st.caption(
        "This is an estimated listing price, "
        "not a guaranteed market or sale price."
    )
