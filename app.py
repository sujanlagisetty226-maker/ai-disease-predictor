import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
model = joblib.load("disease_model.pkl")

# --------------------------------
# Sample Training Dataset
# --------------------------------

data = {
"Age":[25,45,50,23,60,35,40,29,55,48],
"BMI":[22,31,33,21,35,29,28,23,36,32],
"Gender":[0,1,1,0,1,0,1,0,1,1],
"Smoking":[0,1,1,0,1,0,0,0,1,1],
"Alcohol":[0,1,1,0,1,0,0,0,1,1],
"Sleep":[7,5,6,8,5,6,6,7,5,6],
"Activity":[4,1,1,5,1,2,2,4,1,1],
"FamilyHistory":[0,1,1,0,1,0,1,0,1,1],
"Cholesterol":[180,230,250,170,260,200,210,190,270,240],
"BloodPressure":[120,150,160,110,170,130,135,118,175,155],
"Diet":[7,3,2,8,2,5,6,7,2,3],
"Stress":[3,8,7,2,8,5,6,3,9,7],
"ScreenTime":[4,9,8,3,10,6,7,4,11,9],
"Disease":[0,1,1,0,1,0,0,0,1,1]
}

df = pd.DataFrame(data)

X = df.drop("Disease",axis=1)
y = df["Disease"]

model = RandomForestClassifier()
model.fit(X,y)

# --------------------------------
# Streamlit Interface
# --------------------------------

st.title("AI Lifestyle Disease Risk Predictor")

st.write("Enter your health details")

age = st.slider("Age",18,80)
bmi = st.slider("BMI",15,45)
gender = st.selectbox("Gender (0=Female,1=Male)",[0,1])
smoking = st.selectbox("Smoking",[0,1])
alcohol = st.selectbox("Alcohol",[0,1])
sleep = st.slider("Sleep Hours",3,10)
activity = st.slider("Physical Activity Level",1,5)
family = st.selectbox("Family History",[0,1])
chol = st.slider("Cholesterol",150,300)
bp = st.slider("Blood Pressure",90,180)
diet = st.slider("Diet Quality Score",1,10)
stress = st.slider("Stress Level",1,10)
screen = st.slider("Screen Time (hours)",1,12)

# --------------------------------
# Prediction
# --------------------------------

if st.button("Predict Risk"):

    input_data = np.array([[age,bmi,gender,smoking,alcohol,sleep,activity,
                            family,chol,bp,diet,stress,screen]])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("High Lifestyle Disease Risk")
    else:
        st.success("Low Lifestyle Disease Risk")

    # AI Advice
    st.subheader("AI Health Advice")

    if bmi > 30:
        st.write("Your BMI indicates obesity risk. Increase physical activity.")

    if sleep < 6:
        st.write("Sleep duration is low. Aim for 7–8 hours.")

    if stress > 7:
        st.write("High stress detected. Practice relaxation techniques.")

    if screen > 8:
        st.write("Excess screen time may affect health. Reduce usage.")

    st.info("This tool provides risk estimation only and is not a medical diagnosis.")