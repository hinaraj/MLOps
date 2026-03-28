import pandas as pd
import os
from sklearn.model_selection import train_test_split

# ==============================
# Load Dataset
# ==============================
DATA_PATH = "tourism_project/data/tourism.csv"
df = pd.read_csv(DATA_PATH)

print(" Dataset loaded")

# ==============================
# Target
# ==============================
target = "ProdTaken"

# ==============================
# Drop ID column
# ==============================
if "CustomerID" in df.columns:
    df.drop(columns=["CustomerID"], inplace=True)

# ==============================
# Handle missing values
# ==============================
df = df.ffill()

# ==============================
# Features
# ==============================
numeric_features = [
    "Age","CityTier","NumberOfPersonVisiting","PreferredPropertyStar",
    "NumberOfTrips","Passport","OwnCar","NumberOfChildrenVisiting",
    "MonthlyIncome","PitchSatisfactionScore","NumberOfFollowups","DurationOfPitch"
]

categorical_features = [
    "TypeofContact","Occupation","Gender",
    "MaritalStatus","Designation","ProductPitched"
]

X = df[numeric_features + categorical_features]
y = df[target]

# ==============================
# Train Test Split
# ==============================
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==============================
# Save files
# ==============================
os.makedirs("tourism_project/data", exist_ok=True)

Xtrain.to_csv("tourism_project/data/Xtrain.csv", index=False)
Xtest.to_csv("tourism_project/data/Xtest.csv", index=False)
ytrain.to_csv("tourism_project/data/ytrain.csv", index=False)
ytest.to_csv("tourism_project/data/ytest.csv", index=False)

print(" Data split and saved successfully")
