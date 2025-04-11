import torch
from train_patient import PatientMLP  # Make sure this import is correct
import pandas as pd

# Set device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define and load the model
model = PatientMLP(input_dim=30, num_classes=2).to(device)
model.load_state_dict(torch.load("checkpoints/patient_model.pt", map_location=device))
model.eval()

# Load patient data (adjust this based on your actual file/input format)
patient_data = pd.read_csv("path_to_patient_data.csv")  # Adjust file path

# Convert the patient data to a tensor (ensure the correct shape)
patient_tensor = torch.tensor(patient_data.values, dtype=torch.float32).to(device)

# Make prediction with no gradient tracking
with torch.no_grad():
    prediction = model(patient_tensor)
    probabilities = torch.softmax(prediction, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1)

# Output the prediction result
print(f"Predicted Class: {predicted_class.item()}")
print(f"Probabilities: {probabilities.cpu().numpy()}")
