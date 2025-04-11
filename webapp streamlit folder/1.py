import torch

# Assuming the model and data have been loaded correctly
with torch.no_grad():
    # Your inference code here
    prediction = model(patient_tensor.to(device))
    probabilities = torch.softmax(prediction, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1)

# Continue with the rest of your code...
