import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import math
import pickle
warnings.filterwarnings('ignore')


# 1️⃣ Load dataset
data = pd.read_csv('auto_imports.csv')

# 2️⃣ Rename columns safely
data.columns = [
    "symboling","normalized_losses","Make","Fuel_type","aspiration",
    "Doors","Body_style","Drive_wheels","Engine_location","Wheel_base",
    "Length","Width","Height","Curb_weight","Engine_type","Cylinders",
    "Engine_size","Fuel_system","Bore","Stroke","Compression_ratio",
    "Horsepower","Peak_rpm","City_mpg","Highway_mpg","price"
]

# 3️⃣ Convert numeric columns safely
numeric_cols = ['normalized_losses','Bore','Stroke','Horsepower','Peak_rpm']
for col in numeric_cols:
    data[col] = pd.to_numeric(data[col], errors='coerce')
    data[col].fillna(data[col].mean(), inplace=True)

# 4️⃣ Map categorical columns
make_map = {
    'toyota': 0,'nissan': 1,'mazda': 2,'mitsubishi': 3,'honda': 4,'volkswagen': 5,
    'subaru': 6,'peugot': 7,'volvo': 8,'dodge': 9,'mercedes-benz': 10,'bmw': 11,
    'plymouth': 12,'audi': 13,'saab': 14,'porsche': 15,'jaguar': 16,'chevrolet': 17,
    'alfa-romero': 18,'isuzu': 19,'renault': 20,'mercury': 21
}
data['Make'] = data['Make'].str.lower().str.strip().map(make_map)

body_map = {'sedan':0,'hatchback':1,'wagon':2,'hardtop':3,'convertible':4}
data['Body_style'] = data['Body_style'].str.lower().str.strip().map(body_map)

drive_map = {'fwd':0,'rwd':1,'4wd':2}
data['Drive_wheels'] = data['Drive_wheels'].str.lower().str.strip().map(drive_map)

engine_type_map = {"ohc":0,"ohcf":1,"ohcv":2,"dohc":3,"l":4,"rotor":5}
data['Engine_type'] = data['Engine_type'].str.lower().str.strip().map(engine_type_map)

num_cyl_map = {"twelve":0,"three":1,"eight":2,"two":3,"five":4,"six":5,"four":6}
data['Cylinders'] = data['Cylinders'].str.lower().str.strip().map(num_cyl_map)

fuel_system_map = {'mpfi':0,'2bbl':1,'idi':2,'1bbl':3,'spdi':4,'4bbl':5,'mfi':6,'spfi':7}
data['Fuel_system'] = data['Fuel_system'].str.lower().str.strip().map(fuel_system_map)

# 5️⃣ One-hot encoding
data = pd.get_dummies(data, columns=['Fuel_type','aspiration','Doors','Engine_location'], drop_first=True)

# 6️⃣ Drop irrelevant columns
data = data.drop(['Curb_weight','Highway_mpg','Horsepower','Length','Width'], axis=1)

# 7️⃣ Scale numeric columns
num_cols = data.select_dtypes(include=['int64','float64']).columns
num_cols = num_cols.drop('price')
scaler = StandardScaler()
data[num_cols] = scaler.fit_transform(data[num_cols])

# 8️⃣ Split data
X = data.drop('price', axis=1)
y = data['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=22)

# 9️⃣ Train RandomForestRegressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 10️⃣ Save model
with open(r'D:\car_prediction\rf_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)



print("Model trained and saved successfully!")
