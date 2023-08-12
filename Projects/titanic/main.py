from utils import columns, PrepProcesor
import pandas as pd
import numpy as np
import joblib
import warnings
import streamlit as st

warnings.simplefilter("ignore")


model = joblib.load("xgbpipe.joblib")

st.title("Titanic Prediction :ship:")

passengerid = st.text_input("Input Passenger ID", "123456")
passengerclass = st.slider("Choose Class", 1, 3)
name = st.text_input("Enter The Passenger Name", "John Doe")
gender = st.select_slider("Select Gender", ["Male", "Female"])
age = st.slider("Input The Age", 0, 100)
sibsp = st.slider("Input Siblings", 0, 10)
parch = st.slider("Input parents/children", 0, 2)
ticketid = st.number_input("Ticket Number", 12345)
fare = st.number_input("Fare Amount", 0, 100)
cabin = st.text_input("Enter Cabin", "C52")
embarked = st.selectbox("Choose Embarkation Point", ["S", "C", "Q"])


def predict():
    row = np.array(
        [
            passengerid,
            passengerclass,
            name,
            gender,
            age,
            sibsp,
            parch,
            ticketid,
            fare,
            cabin,
            embarked,
        ]
    )
    X = pd.DataFrame([row], columns=columns)
    prediction = model.predict(X)[0]

    if prediction == 1:
        st.success("Survive")
    else:
        st.error("Dead")


st.button("Predict", on_click=predict)
