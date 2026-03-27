
import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# ==============================
# Load Model from Hugging Face
# ==============================
model_path = hf_hub_download(
    repo_id="hinaabcd/tourism-package-model",   # updated model repo
    filename="tourism_model.joblib"
)

model = joblib.load(model_path)

# ==============================
# Streamlit UI
# ==============================
st.set_page_config(page_title="Tourism Prediction", page_icon="")

st.title(" Tourism Package Prediction App")
st.markdown("### Predict whether a customer will purchase the Wellness Tourism Package")

# ==============================
# User Inputs
# ==============================

Age = st.number_input("Age", 18, 80, 30)
CityTier = st.selectbox("City Tier", [1, 2, 3])
NumberOfPersonVisiting = st.number_input("Number of Persons Visiting", 1, 10, 2)
PreferredPropertyStar = st.selectbox("Preferred Hotel Rating", [1, 2, 3, 4, 5])
NumberOfTrips = st.number_input("Number of Trips per Year", 0, 20, 2)

Passport = st.selectbox("Has Passport?", ["Yes", "No"])
OwnCar = st.selectbox("Owns Car?", ["Yes", "No"])
NumberOfChildrenVisiting = st.number_input("Children Visiting", 0, 5, 0)

MonthlyIncome = st.number_input("Monthly Income", 1000, 500000, 30000)

PitchSatisfactionScore = st.slider("Pitch Satisfaction Score", 1, 5, 3)
NumberOfFollowups = st.number_input("Number of Follow-ups", 0, 10, 2)
DurationOfPitch = st.number_input("Duration of Pitch (minutes)", 1, 60, 15)

TypeofContact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
Occupation = st.selectbox("Occupation", ["Salaried", "Freelancer", "Business", "Other"])
Gender = st.selectbox("Gender", ["Male", "Female"])
MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
Designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "VP"])
ProductPitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe"])

# ==============================
# Prepare Input Data
# ==============================
input_data = pd.DataFrame([{
    "Age": Age,
    "CityTier": CityTier,
    "NumberOfPersonVisiting": NumberOfPersonVisiting,
    "PreferredPropertyStar": PreferredPropertyStar,
    "NumberOfTrips": NumberOfTrips,
    "Passport": 1 if Passport == "Yes" else 0,
    "OwnCar": 1 if OwnCar == "Yes" else 0,
    "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
    "MonthlyIncome": MonthlyIncome,
    "PitchSatisfactionScore": PitchSatisfactionScore,
    "NumberOfFollowups": NumberOfFollowups,
    "DurationOfPitch": DurationOfPitch,
    "TypeofContact": TypeofContact,
    "Occupation": Occupation,
    "Gender": Gender,
    "MaritalStatus": MaritalStatus,
    "Designation": Designation,
    "ProductPitched": ProductPitched
}])

# ==============================
# Prediction
# ==============================
threshold = 0.45

if st.button("🔍 Predict"):
    prediction_proba = model.predict_proba(input_data)[0, 1]
    prediction = int(prediction_proba >= threshold)

    st.subheader(" Prediction Result")

    if prediction == 1:
        st.success(" Customer is likely to purchase the package")
    else:
        st.error(" Customer is NOT likely to purchase the package")

    st.metric("Prediction Probability", round(prediction_proba, 3))
