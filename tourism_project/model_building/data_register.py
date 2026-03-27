
from huggingface_hub.utils import RepositoryNotFoundError
from huggingface_hub import HfApi, create_repo
import os

# ==============================
# Hugging Face Dataset Repo
# ==============================
repo_id = "hinaabcd/tourism-package-dataset"   # dataset repo (recommended separate)
repo_type = "dataset"

# Initialize API client
api = HfApi(token=os.getenv("HF_TOKEN"))

# ==============================
# Check if dataset repo exists
# ==============================
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Dataset repo '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Dataset repo '{repo_id}' not found. Creating new repo...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Dataset repo '{repo_id}' created.")

# ==============================
# Upload dataset files
# ==============================
api.upload_folder(
    folder_path="tourism_project/data",   # ✅ updated path
    repo_id=repo_id,
    repo_type=repo_type,
)

print(" Dataset uploaded successfully to Hugging Face")
