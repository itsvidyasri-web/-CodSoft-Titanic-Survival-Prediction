
import streamlit as st
import pandas as pd
import pickle

# Load trained machine learning model
with open("titanic_model.pkl", "rb") as file:
    model = pickle.load(file)

# Website title
st.title("Titanic Survival Prediction")
st.write("Enter passenger details to predict survival.")

# Get passenger details
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=0, max_value=100, value=25)
sibsp = st.number_input("Siblings/Spouses", min_value=0, max_value=10, value=0)
parch = st.number_input("Parents/Children", min_value=0, max_value=10, value=0)
fare = st.number_input("Ticket Fare", min_value=0.0, value=30.0)
embarked = st.selectbox("Boarding Port", ["S", "C", "Q"])

# Convert gender into numbers
sex_value = 1 if sex == "Male" else 0

# Prepare passenger data
passenger = pd.DataFrame([{
    "Pclass": pclass,
    "Sex": sex_value,
    "Age": age,
    "SibSp": sibsp,
    "Parch": parch,
    "Fare": fare,
    "Embarked_C": 1 if embarked == "C" else 0,
    "Embarked_Q": 1 if embarked == "Q" else 0,
    "Embarked_S": 1 if embarked == "S" else 0
}])

# Prediction button
if st.button("Predict Survival"):
    prediction = model.predict(passenger)

    if prediction[0] == 1:
        st.success("Passenger is predicted to Survive!")
    else:
        st.error("Passenger is predicted Not to Survive!")
