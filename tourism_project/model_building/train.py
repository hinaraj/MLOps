
# ==============================
# Imports
# ==============================
import pandas as pd
import os

# preprocessing
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

# modeling
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

# tracking
import mlflow

# serialization
import joblib

# Hugging Face
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# ==============================
# MLflow Setup (FIXED)
# ==============================
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Tourism_Package_Prediction")

# ==============================
# Hugging Face API
# ==============================
api = HfApi()

# ==============================
# Load Data from Hugging Face Dataset
# ==============================
Xtrain = pd.read_csv("hf://datasets/hinaabcd/visit_with_us/Xtrain.csv")
Xtest  = pd.read_csv("hf://datasets/hinaabcd/visit_with_us/Xtest.csv")
ytrain = pd.read_csv("hf://datasets/hinaabcd/visit_with_us/ytrain.csv")
ytest  = pd.read_csv("hf://datasets/hinaabcd/visit_with_us/ytest.csv")

# Convert to Series
ytrain = ytrain.squeeze()
ytest = ytest.squeeze()

print(" Data loaded from Hugging Face")

# ==============================
# Feature Lists
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
# Handle Class Imbalance
# ==============================
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

# ==============================
# Preprocessing
# ==============================
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)

# ==============================
# Model
# ==============================
xgb_model = xgb.XGBClassifier(
    scale_pos_weight=class_weight,
    random_state=42,
    eval_metric="logloss"
)

# ==============================
# Hyperparameter Grid
# ==============================
param_grid = {
    'xgbclassifier__n_estimators': [50, 100],
    'xgbclassifier__max_depth': [3, 5],
    'xgbclassifier__learning_rate': [0.01, 0.1],
    'xgbclassifier__colsample_bytree': [0.6],
    'xgbclassifier__reg_lambda': [0.5]
}

# ==============================
# Pipeline
# ==============================
model_pipeline = make_pipeline(preprocessor, xgb_model)

# ==============================
# MLflow Run
# ==============================
with mlflow.start_run():

    # Train
    grid_search = GridSearchCV(
        model_pipeline,
        param_grid,
        cv=5,
        n_jobs=-1,
        scoring="accuracy"
    )

    grid_search.fit(Xtrain, ytrain)

    # ==============================
    # Log all runs
    # ==============================
    results = grid_search.cv_results_

    for i in range(len(results['params'])):
        with mlflow.start_run(nested=True):
            mlflow.log_params(results['params'][i])
            mlflow.log_metric("mean_test_score", results['mean_test_score'][i])
            mlflow.log_metric("std_test_score", results['std_test_score'][i])

    # ==============================
    # Best Model
    # ==============================
    best_model = grid_search.best_estimator_

    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("best_cv_score", grid_search.best_score_)

    # ==============================
    # Evaluation
    # ==============================
    threshold = 0.45

    y_pred_train = (best_model.predict_proba(Xtrain)[:, 1] >= threshold).astype(int)
    y_pred_test = (best_model.predict_proba(Xtest)[:, 1] >= threshold).astype(int)

    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    mlflow.log_metrics({
        "train_accuracy": train_report['accuracy'],
        "train_precision": train_report['1']['precision'],
        "train_recall": train_report['1']['recall'],
        "train_f1": train_report['1']['f1-score'],
        "test_accuracy": test_report['accuracy'],
        "test_precision": test_report['1']['precision'],
        "test_recall": test_report['1']['recall'],
        "test_f1": test_report['1']['f1-score']
    })

    # ==============================
    # Save Model
    # ==============================
    os.makedirs("tourism_project/models", exist_ok=True)

    model_path = "tourism_project/models/tourism_model.joblib"
    joblib.dump(best_model, model_path)

    mlflow.log_artifact(model_path)

    print(" Model saved and logged")

# ==============================
# Upload Model to Hugging Face
# ==============================
repo_id = "hinaabcd/tourism-package-model"
repo_type = "model"

try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f" Repo exists: {repo_id}")
except RepositoryNotFoundError:
    print(" Creating Hugging Face model repo...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)

api.upload_file(
    path_or_fileobj=model_path,
    path_in_repo="tourism_model.joblib",
    repo_id=repo_id,
    repo_type=repo_type,
)

print(" Model uploaded to Hugging Face")
