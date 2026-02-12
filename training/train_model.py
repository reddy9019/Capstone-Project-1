import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv('../data/manufacturing_dataset.csv')

# Handle missing values
df['Material_Viscosity'].fillna(df['Material_Viscosity'].mean(), inplace=True)
df['Ambient_Temperature'].fillna(df['Ambient_Temperature'].mean(), inplace=True)
df['Operator_Experience'].fillna(df['Operator_Experience'].mean(), inplace=True)

# Drop Timestamp if exists
if 'Timestamp' in df.columns:
    df.drop('Timestamp', axis=1, inplace=True)

# Encode categorical variables
categorical_cols = ['Shift', 'Machine_Type', 'Material_Grade', 'Day_of_Week']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Define features & target
X = df.drop('Parts_Per_Hour', axis=1)
feature_columns = X.columns
y = df['Parts_Per_Hour']

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Predictions
y_pred = model.predict(X_test_scaled)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)

# Save model
with open('../model/linear_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('../model/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('../model/feature_columns.pkl', 'wb') as f:
    pickle.dump(feature_columns, f)

print("Model and scaler saved successfully!")
