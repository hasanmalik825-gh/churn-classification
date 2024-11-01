
path = 'DataScience/files/'

# Your Streamlit app code here
import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Load the trained model
model = tf.keras.models.load_model(path+'ann.h5')

# Load the encoders and scaler
with open(path+'ohe_encoder.pkl', 'rb') as file:
    onehot_encoder = pickle.load(file)

with open(path+'std_scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)


## streamlit app
st.title('Customer Churn Prediction')

# User input
geography = st.selectbox('Geography', onehot_encoder.categories_[0])
gender = st.selectbox('Gender', onehot_encoder.categories_[1])
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# Fit the encoder on the relevant columns
try:
    geog_gend = onehot_encoder.transform([[geography, gender]]).toarray()
except Exception as e:
    st.error(f"Error in OneHotEncoder transformation: {e}")


encoded_columns = onehot_encoder.get_feature_names_out()
geog_gend_df = pd.DataFrame(geog_gend, columns=encoded_columns)
# Combine one-hot encoded columns with input data
input_data=pd.concat([input_data.reset_index(drop=True), geog_gend_df], axis=1)

# Scale the input data
input_data_scaled = scaler.transform(input_data)

st.write(input_data_scaled, input_data_scaled.shape)
# Predict churn
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

st.write(f'Churn Probability: {prediction_proba:.2f}')

if prediction_proba > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn.')
