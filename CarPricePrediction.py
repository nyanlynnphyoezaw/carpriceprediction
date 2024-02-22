import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import RandomizedSearchCV
st.title('Car Price Prediction Demo')

# Get user input features
km_driven = st.slider('Kilometers Driven', min_value=0, max_value=500000, step=1000, value=70000)
fuel = st.selectbox('Fuel Type', ['Petrol', 'Diesel', 'CNG', 'LPG', 'Electric'], index=1)
seller_type = st.selectbox('Seller Type', ['Individual', 'Dealer', 'Trustmark Dealer'], index=1)
transmission = st.selectbox('Transmission', ['Manual', 'Automatic'], index=1)
past_owners = st.slider('Past Owners', min_value=0, max_value=5, value=1)
age = st.slider('Car Age (in years)', min_value=0, max_value=40, value=10)

# Dictionary mappings
dic_fuel = {'Petrol': 1, 'Diesel': 2, 'CNG': 3, 'LPG': 4, 'Electric': 5}
dic_seller_type = {'Individual': 1, 'Dealer': 2, 'Trustmark Dealer': 3}
dic_transmission = {'Manual': 1, 'Automatic': 2}

# Map selected options to numerical values
fuel_mapping = dic_fuel[fuel]
seller_type_mapping = dic_seller_type[seller_type]
transmission_mapping = dic_transmission[transmission]

# Create a numpy array with user input
user_features = np.array([[km_driven, fuel_mapping, seller_type_mapping,
                               transmission_mapping, past_owners, age]])

# Display user input
st.subheader('User Input:')
st.write(pd.DataFrame(user_features, columns=['km_driven', 'fuel', 'seller_type', 'transmission', 'past_owners', 'age']))

# Make predictions
if st.button('Predict Price'):
    try:
        model = joblib.load('carpriceprediction.pkl')
        predicted_price = model.predict(user_features)
        st.subheader('Predicted Price:')
        st.write(predicted_price)  # Display the predicted price
    except FileNotFoundError:
        st.error("Model file not found. Please check the file path.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
