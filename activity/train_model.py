import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pickle

# Load dataset
df = pd.read_csv("data.csv")

# Clean and preprocess
df['state_of_charge_%'] = df['state_of_charge_%'].astype(str).str.replace('%', '', regex=False)
df['state_of_charge_%'] = pd.to_numeric(df['state_of_charge_%'], errors='coerce')
df = df.dropna(subset=['battery_health_%'])

features = ['voltage_V', 'current_A', 'temperature_C', 'state_of_charge_%',
            'cycle_count', 'remaining_capacity_Ah']
X = df[features]
y = df['battery_health_%']

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Train models
models = {
    "RandomForest": RandomForestRegressor(random_state=42),
    "LinearRegression": LinearRegression(),
    "DecisionTree": DecisionTreeRegressor(random_state=42),
    "SVR": SVR()
}

mse_scores = {}
trained_models = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mse_scores[name] = round(mse, 2)
    trained_models[name] = model

# Save model, imputer and MSEs
with open("model.pkl", "wb") as f:
    pickle.dump((trained_models, imputer, mse_scores), f)
