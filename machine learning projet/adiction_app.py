import streamlit as st
import joblib
import numpy as np

# Load the model
model = joblib.load('rf_addiction.pkl')

st.title("Social Media Addiction Prediction App")
st.header("Enter user information:")

# Mappings
location_mapping = {
    'Pakistan': 6, 'Mexico': 5, 'United States': 8, 'Brazil': 0,
    'Vietnam': 9, 'India': 2, 'Indonesia': 3, 'Japan': 4,
    'Philippines': 7, 'Germany': 1
}
gender_mapping = {'Female': 0, 'Male': 1}
profession_mapping = {'Engineer': 2, 'Artist': 0, 'Waiting staff': 7, 'Manager': 4, 'driver': 8, 
                      'Students': 5, 'Labor/Worker': 3, 'Cashier': 1, 'Teacher': 6}
demographics_mapping = {'Urban': 1, 'Rural': 0}
platform_mapping = {'Instagram': 1, 'Facebook': 0, 'YouTube': 3, 'TikTok': 2}
video_category_mapping = {'Pranks': 6, 'Vlogs': 8, 'Gaming': 3, 'Jokes/Memes': 4, 
                          'Entertainment': 2,'ASMR': 0, 'Trends': 7, 'Comedy': 1, 'Life Hacks': 5}
frequency_mapping = {'Morning': 2, 'Afternoon': 0, 'Evening': 1, 'Night': 3}
watch_reason_mapping = {'Procrastination': 3, 'Habit': 2, 'Entertainment': 1, 'Boredom': 0}
device_type_mapping = {'Smartphone': 1, 'Computer': 0, 'Tablet': 2}
connection_type_mapping = {'Mobile Data': 0, 'Wi-Fi': 1}

# Inputs
age = st.number_input("Age", min_value=18, max_value=100)
gender = st.radio("Gender", list(gender_mapping.keys()))
location = st.selectbox("Location", list(location_mapping.keys()))
income = st.number_input("Income", step=100)
debt = st.radio("Debt", ["True", "False"])
owns_property = st.radio("Owns Property", ["True", "False"])
profession = st.selectbox("Profession", list(profession_mapping.keys()))
demographics = st.radio("Demographics", list(demographics_mapping.keys()))
platform = st.selectbox("Platform", list(platform_mapping.keys()))
total_time_spent = st.number_input("Total Time Spent")
number_of_sessions = st.number_input("Number of Sessions", step=1)
video_category = st.selectbox("Video Category", list(video_category_mapping.keys()))
video_length = st.number_input("Video Length (minutes)")
time_spent_on_video = st.number_input("Time Spent On Video (minutes)")
number_of_videos_watched = st.number_input("Number of Videos Watched")
scroll_rate = st.slider("Scroll Rate", 0, 100)
frequency = st.selectbox("Frequency", list(frequency_mapping.keys()))

# Second table inputs
productivity_loss = st.slider("Productivity Loss (1-10)", 1, 9)
satisfaction = st.slider("Satisfaction (1-10)", 1, 9)
watch_reason = st.selectbox("Watch Reason", list(watch_reason_mapping.keys()))
device_type = st.selectbox("Device Type", list(device_type_mapping.keys()))
self_control = st.slider("Self Control (3-10)", 3, 10)
connection_type = st.selectbox("Connection Type", list(connection_type_mapping.keys()))

# Convert to encoded values
gender = gender_mapping[gender]
location = location_mapping[location]
profession = profession_mapping[profession]
demographics = demographics_mapping[demographics]
platform = platform_mapping[platform]
video_category = video_category_mapping[video_category]
frequency = frequency_mapping[frequency]
watch_reason = watch_reason_mapping[watch_reason]
device_type = device_type_mapping[device_type]
connection_type = connection_type_mapping[connection_type]
debt = 1 if debt == "True" else 0
owns_property = 1 if owns_property == "True" else 0

# Feature vector (ensure order matches training)
features = np.array([[age, gender, location, income, debt, owns_property, profession,
                      demographics, platform, total_time_spent, number_of_sessions,
                      video_category, video_length, time_spent_on_video,
                      number_of_videos_watched, scroll_rate, frequency,
                      productivity_loss, satisfaction, watch_reason,
                      device_type, self_control, connection_type]])

# Predict
class_labels = {
    0: "Very Low",
    1: "Low",
    2: "Mild",
    3: "Moderate",
    4: "High",
    5: "Very High",
    6: "Severe",
    7: "Extreme"
}

# Prediction
if st.button("Predict Addiction Level"):
    prediction = model.predict(features)
    label = class_labels[prediction[0]]
    st.success(f"Predicted Addiction Level: {label}")

    probs = model.predict_proba(features)
    confidence = max(probs[0]) * 100

    st.info(f"Confidence: {confidence:.2f}%")