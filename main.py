import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

# Load dataset
df = pd.read_csv('data/house_data.csv')

# Preprocess (simplified for illustration)
df = df.dropna()
X = df.drop(['Price'], axis=1)
y = df['Price']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
os.makedirs('model', exist_ok=True)
joblib.dump(model, 'model/house_price_model.pkl')
print("Model trained and saved successfully.")
