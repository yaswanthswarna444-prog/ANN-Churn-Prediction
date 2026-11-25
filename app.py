import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler
import pandas as pd
import numpy as np
import pickle

## load the model
model = tf.keras.models.load_model("model.h5")


with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl','rb') as file:
    onehot_encoder_geo = pickle.load(file)   

with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file)

st.title("customer churn prediction")

Geography = st.selectbox("Geography",onehot_encoder_geo.categories_[0])
gender = st.selectbox("Gender",label_encoder_gender.classes_)
age = st.slider('Age',18,92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure',0,10)
num_of_products = st.slider('Number of Products',1,4)
has_cr_card = st.selectbox("Has Credit Card",[0,1])
is_active_member = st.selectbox("Is Active Member",[0,1])

input_data =pd.DataFrame({ 
    
    'Geography':[Geography],
    'Gender':[label_encoder_gender.transform([gender])[0]],
    'Age':[age],
    'Balance':[balance],
    'CreditScore':[credit_score],
    'EstimatedSalary':[estimated_salary],
    'Tenure':[tenure],
    'NumOfProducts':[num_of_products],
    'HasCrCard':[has_cr_card],
    'IsActiveMember':[is_active_member]
})
geo_encoder = onehot_encoder_geo.transform(
    [[input_data['Geography'][0]]]
).toarray()
geo_encoder = pd.DataFrame(
    geo_encoder,
    columns=onehot_encoder_geo.get_feature_names_out(['Geography'])
)

input_data = pd.concat(
    [input_data.reset_index(drop=True), geo_encoder.reset_index(drop=True)],
    axis=1
)
input_data.drop('Geography', axis=1, inplace=True)
# Arrange input columns in the same order used during scaler training
input_data = input_data.reindex(columns=scaler.feature_names_in_)

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Make prediction using the trained ANN model
prediction = model.predict(input_data_scaled)

# Extract prediction probability
prediction_proba = float(prediction[0][0])

# Display prediction probability
st.write("Churn Prediction Probability:", round(prediction_proba * 100, 2), "%")

# Check whether the customer is likely to churn
if prediction_proba > 0.5:
    st.error("The customer is likely to churn.")
else:
    st.success("The customer is not likely to churn.")
