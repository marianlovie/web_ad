from train_patient i
mport PatientMLP  # Or wherever your model class is defined
import torch

model = PatientMLP(input_dim=input_dim, num_classes=2)  # Ensure input_dim is passed correctly
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()
