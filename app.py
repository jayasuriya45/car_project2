import pickle
import streamlit as st
import numpy as np
import pandas as pd
  
with open(r'D:\car_prediction\Rf_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)


# --- Page config ---

st.title("Car Price Predictor")
st.markdown("<h1 style='text-align:center; color:orange;'>🚗 Car Price Prediction App</h1>", unsafe_allow_html=True)
st.write("Enter the car details to predict its price:")


# --- Mapping dictionaries ---
car_mapping = {
    'toyota': 0, 'nissan': 1, 'mazda': 2, 'mitsubishi': 3, 'honda': 4,
    'volkswagen': 5, 'subaru': 6, 'peugot': 7, 'volvo': 8, 'dodge': 9,
    'mercedes-benz': 10, 'bmw': 11, 'plymouth': 12, 'audi': 13, 'saab': 14,
    'porsche': 15, 'jaguar': 16, 'chevrolet': 17, 'alfa-romero': 18,
    'isuzu': 19, 'renault': 20, 'mercury': 21
}

body_mapping = {'sedan': 0, 'hatchback': 1, 'wagon': 2, 'hardtop': 3, 'convertible': 4}
drive_mapping = {'fwd': 0, 'rwd': 1, '4wd': 2}
engine_mapping = {'dohc': 0, 'ohcv': 1, 'ohc': 2, 'l': 3, 'rotor': 4}
fuel_system_mapping = {'mpfi': 0, '2bbl': 1, 'idi': 2, '1bbl': 3, 'spdi': 4, '4bbl': 5}

# --- Input columns ---
col1, col2 = st.columns(2)

with col1:
    make_mapping = {'alfa-romero':0, 'audi':1, 'bmw':2, 'chevrolet':3, 'dodge':4,
                'honda':5, 'isuzu':6, 'jaguar':7, 'mazda':8, 'mercedes-benz':9,
                'mercury':10, 'mitsubishi':11, 'nissan':12, 'peugot':13, 'plymouth':14,
                'porsche':15, 'renault':16, 'saab':17, 'subaru':18, 'toyota':19, 'volkswagen':20, 'volvo':21}

body_mapping = {'sedan':0, 'hatchback':1, 'wagon':2, 'hardtop':3, 'convertible':4}
drive_mapping = {'fwd':0, 'rwd':1, '4wd':2}
engine_mapping = {'dohc':0, 'ohcv':1, 'ohc':2, 'l':3, 'rotor':4}
fuel_system_mapping = {'mpfi':0, '2bbl':1, 'idi':2, '1bbl':3, 'spdi':4, '4bbl':5}

fuel_type_mapping = {'gas':0, 'diesel':1}
aspiration_mapping = {'std':0, 'turbo':1}
doors_mapping = {'two':0, 'four':1}
engine_loc_mapping = {'front':0, 'rear':1}

# --- Input columns ---
col1, col2 = st.columns(2)

with col1:
    symboling = st.number_input("Symboling", value=3)
    Normalized_loss = st.number_input("Normalized Loss", value=122.0)
    
    Make_name = st.selectbox("Make", options=list(make_mapping.keys()))
    Make = make_mapping[Make_name]
    
    Fuel_type_name = st.selectbox("Fuel Type", options=list(fuel_type_mapping.keys()))
    Fuel_type = fuel_type_mapping[Fuel_type_name]
    
    aspiration_name = st.selectbox("Aspiration", options=list(aspiration_mapping.keys()))
    aspiration = aspiration_mapping[aspiration_name]
    
    Doors_name = st.selectbox("Doors", options=list(doors_mapping.keys()))
    Doors = doors_mapping[Doors_name]
    
    Body_style_name = st.selectbox("Body Style", options=list(body_mapping.keys()))
    Body_style = body_mapping[Body_style_name]
    
    Drive_wheels_name = st.selectbox("Drive Wheels", options=list(drive_mapping.keys()))
    Drive_wheels = drive_mapping[Drive_wheels_name]
    
    Engine_location_name = st.selectbox("Engine Location", options=list(engine_loc_mapping.keys()))
    Engine_location = engine_loc_mapping[Engine_location_name]

with col2:
    Engine_type_name = st.selectbox("Engine Type", options=list(engine_mapping.keys()))
    Engine_type = engine_mapping[Engine_type_name]
    
    num_of_cylinders_name = st.selectbox("Number of Cylinders", options=['three','four','five','six','eight'])
    num_of_cylinders = {'three':3,'four':4,'five':5,'six':6,'eight':8}[num_of_cylinders_name]
    
    Fuel_system_name = st.selectbox("Fuel System", options=list(fuel_system_mapping.keys()))
    Fuel_system = fuel_system_mapping[Fuel_system_name]
    
    Wheel_base = st.number_input("Wheel Base", value=88.6)
    Height = st.number_input("Height", value=48.8)
    Engine_size = st.number_input("Engine Size", value=130)
    Bore = st.number_input("Bore", value=3.47)
    Stroke = st.number_input("Stroke", value=2.68)
    Compression_ratio = st.number_input("Compression Ratio", value=9.0)
    Peak_rpm = st.number_input("Peak RPM", value=5000)
    City_mpg = st.number_input("City MPG", value=21)
    Width = st.number_input("Width", value=64.1)  # if part of model

# --- Combine inputs into array (numeric) ---
X_input = np.array([[
    Make, Fuel_type, aspiration, Doors, Body_style,
    Drive_wheels, Engine_location, Engine_type, num_of_cylinders, Fuel_system,
    Wheel_base, Height, Engine_size, Bore, Stroke, Compression_ratio, Peak_rpm, City_mpg, Width,
    symboling, Normalized_loss  # adjust order to match your trained model
]])


# --- Load model ---
with open(r'D:\\car_prediction\rf_model.pkl', 'rb') as f:
    model = pickle.load(f)

# --- Prediction button ---
if st.button("🔍 Predict Price"):
    prediction = model.predict(X_input)
    st.success(f"Predicted Car Price: ₹{prediction[0]:,.2f}")


