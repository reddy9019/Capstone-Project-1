import pickle
import numpy as np
import os
import pandas as pd

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(BASE_DIR, 'model', 'linear_model.pkl')
scaler_path = os.path.join(BASE_DIR, 'model', 'scaler.pkl')
features_path = os.path.join(BASE_DIR, 'model', 'feature_columns.pkl')

# Load files
import os
import pickle

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

model_path = os.path.join(BASE_DIR, "model.pkl")
scaler_path = os.path.join(BASE_DIR, "scaler.pkl")

with open(model_path, "rb") as f:
    model = pickle.load(f)

with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)


with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

with open(features_path, 'rb') as f:
    feature_columns = pickle.load(f)


def predict_output(numeric_inputs):
    """
    numeric_inputs: list of 13 numeric values from Streamlit
    """

    # Create a dataframe with zeros for all 25 features
    input_df = pd.DataFrame([np.zeros(len(feature_columns))], columns=feature_columns)

    # Fill numeric columns only
    numeric_feature_names = [
        'Injection_Temperature',
        'Injection_Pressure',
        'Cycle_Time',
        'Cooling_Time',
        'Material_Viscosity',
        'Ambient_Temperature',
        'Machine_Age',
        'Operator_Experience',
        'Maintenance_Hours',
        'Temperature_Pressure_Ratio',
        'Total_Cycle_Time',
        'Efficiency_Score',
        'Machine_Utilization'
    ]

    for name, value in zip(numeric_feature_names, numeric_inputs):
        if name in input_df.columns:
            input_df[name] = value

    # Scale input
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)

    return prediction[0]
