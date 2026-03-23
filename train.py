import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

# Create model folder
os.makedirs("model", exist_ok=True)

# Load dataset
data = pd.read_csv("data/iris.csv")

# Clean column names
data.columns = data.columns.str.strip().str.lower()

# Drop unnecessary column
if 'id' in data.columns:
    data = data.drop('id', axis=1)

# Rename columns (IMPORTANT)
data = data.rename(columns={
    'sepallengthcm': 'sepal_length',
    'sepalwidthcm': 'sepal_width',
    'petallengthcm': 'petal_length',
    'petalwidthcm': 'petal_width',
    'species': 'species'
})

# Features and target
X = data.drop("species", axis=1)
y = data["species"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
with open("model/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained successfully!")