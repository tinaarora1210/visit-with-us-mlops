# src/deploy_space.py
import os
import shutil
from pathlib import Path

from huggingface_hub import create_repo, upload_folder, whoami

# -----------------------------
# Config (update if you renamed your HF Space)
# -----------------------------
HF_SPACE_REPO_NAME = "visit-with-us-wellness-app"  # your Space repo name
LOCAL_DEPLOYMENT_DIR = "deployment"                # folder in this GitHub repo that contains app.py, Dockerfile, requirements.txt

def main():
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN is missing. Add it in GitHub Secrets as HF_TOKEN.")

    user = whoami(token=hf_token)["name"]
    space_repo_id = f"{user}/{HF_SPACE_REPO_NAME}"

    # Ensure local deployment folder exists in the repo
    deploy_path = Path(LOCAL_DEPLOYMENT_DIR)
    if not deploy_path.exists():
        raise FileNotFoundError(
            f"Deployment folder '{LOCAL_DEPLOYMENT_DIR}' not found in repo. "
            f"Make sure you committed your deployment files (app.py, Dockerfile, requirements.txt) into /deployment."
        )

    # Create HF Space repo (Streamlit Space)
    create_repo(
        repo_id=space_repo_id,
        repo_type="space",
        exist_ok=True,
        space_sdk="streamlit",
        token=hf_token
    )

    # Upload deployment folder contents to HF Space root
    upload_folder(
        repo_id=space_repo_id,
        repo_type="space",
        folder_path=str(deploy_path),
        path_in_repo=".",
        token=hf_token
    )

    print("Space deployed/updated successfully.")
    print("Space URL:", f"https://huggingface.co/spaces/{space_repo_id}")

if __name__ == "__main__":
    main()
