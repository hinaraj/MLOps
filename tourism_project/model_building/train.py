import pandas as pd
import os

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

import mlflow
import joblib

from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# ==============================
# MLflow
# ==============================
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Tourism_Package_Prediction")

# ==============================
# Load Data (LOCAL FILES)
# ==============================
Xtrain = pd.read_csv("tourism_project/data/Xtrain.csv")
Xtest  = pd.read_csv("tourism_project/data/Xtest.csv")
ytrain = pd.read_csv("tourism_project/data/ytrain.csv").squeeze()
ytest  = pd.read_csv("tourism_project/data/ytest.csv").squeeze()

print(" Data loaded for training")

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

# ==============================
# Class imbalance
# ==============================
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

# ==============================
# Pipeline
# ==============================
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)

model = xgb.XGBClassifier(
    scale_pos_weight=class_weight,
    random_state=42,
    eval_metric="logloss"
)

pipeline = make_pipeline(preprocessor, model)

# ==============================
# Grid Search
# ==============================
param_grid = {
    'xgbclassifier__n_estimators': [50, 100],
    'xgbclassifier__max_depth': [3, 5],
    'xgbclassifier__learning_rate': [0.01, 0.1]
}

# ==============================
# Training
# ==============================
with mlflow.start_run():

    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        n_jobs=-1,
        scoring="accuracy"
    )

    grid.fit(Xtrain, ytrain)

    best_model = grid.best_estimator_

    mlflow.log_params(grid.best_params_)
    mlflow.log_metric("best_score", grid.best_score_)

    # ==============================
    # Evaluation
    # ==============================
    y_pred = best_model.predict(Xtest)

    report = classification_report(ytest, y_pred, output_dict=True)

    mlflow.log_metric("accuracy", report["accuracy"])

    # ==============================
    # Save Model
    # ==============================
    os.makedirs("tourism_project/models", exist_ok=True)

    model_path = "tourism_project/models/tourism_model.joblib"
    joblib.dump(best_model, model_path)

    mlflow.log_artifact(model_path)

    print(" Model trained & saved")

# ==============================
# Upload to Hugging Face
# ==============================
api = HfApi()
repo_id = "hinaabcd/tourism-package-model"

try:
    api.repo_info(repo_id=repo_id, repo_type="model")
except RepositoryNotFoundError:
    create_repo(repo_id=repo_id, repo_type="model", private=False)

api.upload_file(
    path_or_fileobj=model_path,
    path_in_repo="tourism_model.joblib",
    repo_id=repo_id,
    repo_type="model",
)

print(" Model uploaded to Hugging Face")
