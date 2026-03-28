
# ==============================
# Import Libraries
# ==============================
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi, login

# ==============================
# Hugging Face Authentication
# ==============================
HF_TOKEN = os.getenv("HF_TOKEN")
login(token=HF_TOKEN)

api = HfApi()

# ==============================
# Load Dataset from Hugging Face
# ==============================
DATASET_PATH = "hf://datasets/hinaabcd/tourism-package-dataset/tourism.csv"

df = pd.read_csv(DATASET_PATH)
print(" Dataset loaded successfully")

# ==============================
# Target Variable
# ==============================
target = "ProdTaken"

# ==============================
# Feature Selection
# ==============================

numeric_features = [
    "Age",
    "CityTier",
    "NumberOfPersonVisiting",
    "PreferredPropertyStar",
    "NumberOfTrips",
    "Passport",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "MonthlyIncome",
    "PitchSatisfactionScore",
    "NumberOfFollowups",
    "DurationOfPitch"
]

categorical_features = [
    "TypeofContact",
    "Occupation",
    "Gender",
    "MaritalStatus",
    "Designation",
    "ProductPitched"
]

# ==============================
# Data Cleaning
# ==============================

# Drop unnecessary columns if present
if "CustomerID" in df.columns:
    df.drop(columns=["CustomerID"], inplace=True)

# Handle missing values
df.fillna(method="ffill", inplace=True)

# ==============================
# Encoding
# ==============================

df = pd.get_dummies(df, columns=categorical_features, drop_first=True)

# ==============================
# Define X and y
# ==============================

X = df.drop(target, axis=1)
y = df[target]

# ==============================
# Train-Test Split
# ==============================

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==============================
# Save Locally
# ==============================

os.makedirs("tourism_project/data", exist_ok=True)

Xtrain.to_csv("tourism_project/data/Xtrain.csv", index=False)
Xtest.to_csv("tourism_project/data/Xtest.csv", index=False)
ytrain.to_csv("tourism_project/data/ytrain.csv", index=False)
ytest.to_csv("tourism_project/data/ytest.csv", index=False)

print(" Data saved locally")

# ==============================
# Upload to Hugging Face Dataset
# ==============================

repo_id = "hinaabcd/tourism-package-dataset"

files = [
    "tourism_project/data/Xtrain.csv",
    "tourism_project/data/Xtest.csv",
    "tourism_project/data/ytrain.csv",
    "tourism_project/data/ytest.csv"
]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],
        repo_id=repo_id,
        repo_type="dataset",
    )

print(" Files uploaded to Hugging Face successfully")
