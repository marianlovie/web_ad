import gdown
import os

checkpoint_path = "checkpoints/patient_model.pt"
drive_file_id = "1vVqSQsMnLVkEbIGzxF4ac1aDG7DinR_t"  # Replace with your actual file ID

# Make sure the directory exists
os.makedirs("checkpoints", exist_ok=True)

# Download if not already present
if not os.path.exists(checkpoint_path):
    url = f"https://drive.google.com/uc?isd={drive_file_id}"
    gdown.download(url, checkpoint_path, quiet=False)
    print("✅ Model downloaded from Google Drive.")
