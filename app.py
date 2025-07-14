# Import necessary libraries
import streamlit as st
import pandas as pd
import joblib

# App Title
st.title("Heart Disease Prediction App")
st.markdown("""
This app predicts the likelihood of heart disease based on user input.
""")

# Sidebar for User Input Features
st.sidebar.header('User  Input Features')

def user_input_features():
    age = st.sidebar.slider('Age', 1, 120, 50)
    sex = st.sidebar.selectbox('Sex (0: Female, 1: Male)', [0, 1])
    cp = st.sidebar.selectbox('Chest Pain Type (0-3)', [0, 1, 2, 3])
    trestbps = st.sidebar.slider('Resting Blood Pressure', 80, 200, 120)
    chol = st.sidebar.slider('Serum Cholesterol (mg/dl)', 100, 400, 200)
    fbs = st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dl (1: True, 0: False)', [0, 1])
    restecg = st.sidebar.selectbox('Resting ECG Results (0-2)', [0, 1, 2])
    thalach = st.sidebar.slider('Max Heart Rate Achieved', 60, 220, 150)
    exang = st.sidebar.selectbox('Exercise Induced Angina (1: Yes, 0: No)', [0, 1])
    oldpeak = st.sidebar.slider('ST Depression', 0.0, 10.0, 1.0, 0.1)
    slope = st.sidebar.selectbox('Slope of the ST Segment (0-2)', [0, 1, 2])
    ca = st.sidebar.selectbox('Major Vessels (0-4)', [0, 1, 2, 3, 4])
    thal = st.sidebar.selectbox('Thalassemia (0: Normal, 1: Fixed, 2: Reversible)', [0, 1, 2])

    data = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }
    return pd.DataFrame(data, index=[0])

# Get user input
input_df = user_input_features()

# Load dataset to align columns (excluding 'target')
data = pd.read_csv('heart.csv')
data = data.drop(columns=['target'])  # Drop target column
df = pd.concat([input_df, data], axis=0)  # Concatenate input with dataset (no target)
df = df[:1]  # Keep only user input row

# Load the trained model
try:
    model = joblib.load('Random_forest_model.joblib')
except FileNotFoundError:
    st.error("Model file not found. Please ensure `Random_forest_model.joblib` is in the directory.")
    st.stop()

# Predict and display results
prediction = model.predict(df)
prediction_proba = model.predict_proba(df)

st.subheader('Prediction')
st.write("Heart Disease" if prediction[0] == 1 else "No Heart Disease")

st.subheader('Prediction Probability')
# Safely display prediction probabilities without error
proba_df = pd.DataFrame(prediction_proba, columns=["No Heart Disease", "Heart Disease"])
st.write(proba_df)