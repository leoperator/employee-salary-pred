import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Load model
model = joblib.load("best_model.pkl")

# Load label encoders (or recreate them)
# Ideally, save your encoders from training using joblib, but here we'll re-fit them on categories

# Define possible categories based on training
workclass_cats = ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Others']
marital_status_cats = ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent']
occupation_cats = ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty',
                   'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving',
                   'Priv-house-serv', 'Protective-serv', 'Armed-Forces', 'Others']
relationship_cats = ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried']
race_cats = ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other']
gender_cats = ['Male', 'Female']
country_cats = ['United-States', 'India', 'Mexico', 'Philippines', 'Germany', 'Canada', 'Others']

# Recreate encoders
def fit_encoder(categories):
    le = LabelEncoder()
    le.fit(categories)
    return le

enc_workclass = fit_encoder(workclass_cats)
enc_marital = fit_encoder(marital_status_cats)
enc_occupation = fit_encoder(occupation_cats)
enc_relationship = fit_encoder(relationship_cats)
enc_race = fit_encoder(race_cats)
enc_gender = fit_encoder(gender_cats)
enc_country = fit_encoder(country_cats)

st.set_page_config(page_title="Employee Salary Prediction", page_icon="💼")
st.title("💼 Employee Salary Classification App")
st.markdown("Predict whether an employee earns >50K or ≤50K based on input details.")

st.sidebar.header("📋 Input Employee Data")

# User Inputs
age = st.sidebar.slider("Age", 18, 75, 30)
workclass = st.sidebar.selectbox("Workclass", workclass_cats)
fnlwgt = st.sidebar.number_input("Final Weight (fnlwgt)", min_value=5000, max_value=1000000, value=100000)
education_num = st.sidebar.slider("Education Number", 1, 16, 10)
marital_status = st.sidebar.selectbox("Marital Status", marital_status_cats)
occupation = st.sidebar.selectbox("Occupation", occupation_cats)
relationship = st.sidebar.selectbox("Relationship", relationship_cats)
race = st.sidebar.selectbox("Race", race_cats)
gender = st.sidebar.selectbox("Gender", gender_cats)
capital_gain = st.sidebar.number_input("Capital Gain", min_value=0, value=0)
capital_loss = st.sidebar.number_input("Capital Loss", min_value=0, value=0)
hours_per_week = st.sidebar.slider("Hours per week", 1, 99, 40)
native_country = st.sidebar.selectbox("Native Country", country_cats)

# Encode categorical inputs
input_df = pd.DataFrame({
    'age': [age],
    'workclass': [enc_workclass.transform([workclass])[0]],
    'fnlwgt': [fnlwgt],
    'educational-num': [education_num],
    'marital-status': [enc_marital.transform([marital_status])[0]],
    'occupation': [enc_occupation.transform([occupation])[0]],
    'relationship': [enc_relationship.transform([relationship])[0]],
    'race': [enc_race.transform([race])[0]],
    'gender': [enc_gender.transform([gender])[0]],
    'capital-gain': [capital_gain],
    'capital-loss': [capital_loss],
    'hours-per-week': [hours_per_week],
    'native-country': [enc_country.transform([native_country])[0]]
})

st.subheader("🔍 Input Preview")
st.write(input_df)

if st.button("🔮 Predict Salary Class"):
    prediction = model.predict(input_df)[0]
    st.success(f"✅ Prediction: {'>50K' if prediction == 1 else '<=50K'}")
