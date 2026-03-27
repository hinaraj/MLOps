
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import os

# ==============================
# Hugging Face Setup
# ==============================
HF_TOKEN = os.getenv("HF_TOKEN")
api = HfApi(token=HF_TOKEN)

# ==============================
# Repo Details
# ==============================
repo_id = "hinaabcd/Tourism-Package-Prediction"
repo_type = "space"

# ==============================
# Ensure Space Exists
# ==============================
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f" Space exists: {repo_id}")
except RepositoryNotFoundError:
    print(" Creating new Hugging Face Space...")
    create_repo(
        repo_id=repo_id,
        repo_type="space",
        private=False,
        space_sdk="streamlit"   # REQUIRED for Streamlit
    )
    print("Space created")

# ==============================
# IMPORTANT FIX 
# ==============================
# Upload ONLY the contents inside deployment folder
# so that app.py goes to ROOT of Space (mandatory)

api.upload_folder(
    folder_path="tourism_project/deployment",   # your local folder
    repo_id=repo_id,
    repo_type=repo_type,
    path_in_repo=""   # upload directly to root
)

print(" Deployment files uploaded successfully to Hugging Face Space")
