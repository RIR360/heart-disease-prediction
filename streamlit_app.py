import streamlit as st
import pandas as pd
import pickle

# Load your pre-trained model
# model = pickle.load(open('house_model.pkl', 'rb'))

st.title("🏡 House Price Predictor")

# Create the input form
with st.form("prediction_form"):
    st.subheader("Enter Property Details")
    
    # Input widgets
    size = st.number_input("Size (sq ft)", min_value=100, max_value=10000, value=1500)
    bedrooms = st.slider("Number of Bedrooms", 1, 10, 3)
    location = st.selectbox("Location", ["Urban", "Suburban", "Rural"])
    
    # Every form must have a submit button
    submit_button = st.form_submit_button("Predict Price")

# Actions after clicking submit
if submit_button:
    # 1. Format inputs for the model
    input_data = pd.DataFrame([[size, bedrooms, location]], 
                              columns=['size', 'bedrooms', 'location'])
    
    # 2. (Optional) Apply your preprocessing/encoding here
    
    # 3. Make prediction
    # prediction = model.predict(input_data)
    prediction = 250000  # Placeholder for demo
    
    st.success(f"The estimated price is: ${prediction:,}")
