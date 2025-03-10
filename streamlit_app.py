import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# App Title and Description
st.title("🌿 Green Cloud AI: Smart Resource Optimization")
st.subheader("Green Cloud, Clean Future: Sustainable Computing for a Better Tomorrow!")
st.markdown("This app predicts **task priority** based on various resource usage metrics using Machine Learning.")

# Load Data
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/Leerish/data/refs/heads/main/data.csv"
    df = pd.read_csv(url)
    return df

data = load_data()
st.subheader("📊 Data Overview")
st.write("Here's a preview of the dataset:")
st.dataframe(data.head())

# Data Preprocessing
columns_to_normalize = ['cpu_usage', 'memory_usage', 'network_traffic', 'power_consumption', 'num_executed_instructions', 'execution_time', 'energy_efficiency']
scaler = MinMaxScaler()
data[columns_to_normalize] = scaler.fit_transform(data[columns_to_normalize])
data.fillna(data.mean(numeric_only=True), inplace=True)

for column in ['task_type', 'task_priority']:
    data[column] = data[column].fillna(data[column].mode()[0])

label_encoder = LabelEncoder()
data['task_type_encoded'] = label_encoder.fit_transform(data['task_type'])
data['task_priority_encoded'] = label_encoder.fit_transform(data['task_priority'])

X = data.drop(['task_priority', 'task_type'], axis=1)
y = data['task_priority_encoded']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Data Visualization
st.subheader("📈 Data Insights")



# CPU Usage Distribution


# User Input Section
st.sidebar.header("🔧 Enter Task Details")
task_type = st.sidebar.selectbox('Task Type', ['network', 'io', 'compute'])
cpu_usage = st.sidebar.slider('CPU Usage', 0.00, 100.00, 50.00)
memory_usage = st.sidebar.slider('Memory Usage', 0.00, 100.00, 50.00)
network_traffic = st.sidebar.slider('Network Traffic', 0.00, 1000.00, 500.00)
power_consumption = st.sidebar.slider('Power Consumption', 0.00, 500.00, 50.00)
num_executed_instructions = st.sidebar.slider('Executed Instructions', 0.00, 10000.00, 5000.00)
execution_time = st.sidebar.slider('Execution Time', 0.00, 100.00, 50.00)
energy_efficiency = st.sidebar.slider('Energy Efficiency', 0.00, 1.00, 0.5)

# Create DataFrame for Input Data
input_data = pd.DataFrame({
    'cpu_usage': [cpu_usage],
    'memory_usage': [memory_usage],
    'network_traffic': [network_traffic],
    'power_consumption': [power_consumption],
    'num_executed_instructions': [num_executed_instructions],
    'execution_time': [execution_time],
    'energy_efficiency': [energy_efficiency],
})

# Normalize Input Data
input_data[columns_to_normalize] = scaler.transform(input_data[columns_to_normalize])

# Make Prediction
prediction = model.predict(input_data)
prediction_prob = model.predict_proba(input_data)

# Display Results
st.subheader("📊 Prediction Results")
st.write(f"**Predicted Task Priority:** {prediction[0]}")
st.write("### 🔢 Prediction Probability")
st.bar_chart(prediction_prob[0])

st.success("✅ Prediction Completed! Adjust inputs on the sidebar to see changes.")
