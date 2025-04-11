import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as f
from torchvision import transforms
from PIL import Image
import os
from cnn import ResNetClassifier
from train_patient import PatientMLP  # Adjust if it's from another file

# Assuming the original model was trained with 32 input features
model = PatientMLP(input_dim=32, num_classes=2)  # Use the correct input dimension
# Assuming the original model was trained with 32 input features
input_dim = 32  # Set this to the correct input dimension
num_classes = 2  # Adjust as necessary

# Initialize the model with the correct input dimension

# Load the saved checkpoint
checkpoint_path = "checkpoints/patient_model.pt"
if os.path.exists(checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()
else:
    print(f"Model checkpoint not found at {checkpoint_path}")
# Load CNN Model
@st.cache_resource
def load_cnn_model():
    model = ResNetClassifier(num_classes=4)  # Adjust if needed
    model.load_state_dict(torch.load("checkpoints/resnet50_model.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

# Load MLP Model
@st.cache_resource
def load_mlp_model():
    model_path = os.path.join(os.getcwd(), "checkpoints", "patient_model.pt")
    print(f"Loading model from: {model_path}")
    model = PatientMLP(input_dim=32, num_classes=2)  # Ensure input_dim is passed correctly
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Define the PatientMLP model
class PatientMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(PatientMLP, self).__init__()
        hidden_dim = 64  # Define hidden dimension
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# Check if CUDA (GPU) is available or fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the MLP model
checkpoint_path = "checkpoints/patient_model.pt"
if os.path.exists(checkpoint_path):
    model = PatientMLP(input_dim=32, num_classes=2).to("cpu")  # Adjust input_dim if needed
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()
else:
    print(f"Model checkpoint not found at {checkpoint_path}")

# Preprocess uploaded image
def preprocess_image(image_file):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    image = Image.open(image_file).convert("L")
    return transform(image).unsqueeze(0)  # shape: [1, 1, 224, 224]

# Preprocess patient info into tensor
def preprocess_patient_data(age, gender):
    gender_encoded = {"Male": 0, "Female": 1, "Other": 2}.get(gender, -1)
    features = [age, gender_encoded, age * 0.1]  # You can add more features here later
    padded_features = features + [0.0] * (32 - len(features))  # Pad to 32 features
    return torch.tensor([padded_features], dtype=torch.float32)

# Page setup
st.set_page_config(page_title="Alzheimer's Early Detection", layout="wide")
st.title("🧠 Alzheimer's Early Detection & Progression App")
st.markdown("""
Welcome to our prototype Streamlit app for early detection of Alzheimer's Disease.  
This interface is designed for research purposes using a CNN ensemble model and neuroimaging + patient data.
""")

tab1, tab2, tab3, tab4 = st.tabs([
    "1️⃣ Patient Details",
    "2️⃣ Upload MRI Image",
    "3️⃣ Run CNN Ensemble",
    "4️⃣ Output (Severity)"
])

# Global vars
patient_input = None
image_tensor = None
cnn_pred = None
mlp_pred = None
ensemble_pred = None

# ---- TAB 1: Patient Details ----
with tab1:
    st.subheader("👤 Enter Patient Details")
    name = st.text_input("Name")
    age = st.number_input("Age", min_value=60, max_value=90)
    gender = st.selectbox("Gender", ["Select", "Male", "Female", "Other"])

    # New fields for additional patient details
    ethnicity = st.selectbox("Ethnicity", ["Select", "Caucasian", "African American", "Hispanic", "Asian", "Other"])
    education_level = st.selectbox("Education Level", ["Select", "Less than High School", "High School", "Some College", "Bachelor's Degree", "Graduate Degree"])
    bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, format="%.2f")
    smoking = st.selectbox("Smoking Status", ["Select", "Non-Smoker", "Former Smoker", "Current Smoker"])
    alcohol_consumption = st.selectbox("Alcohol Consumption", ["Select", "None", "Moderate", "Heavy"])
    physical_activity = st.selectbox("Physical Activity Level", ["Select", "Sedentary", "Light", "Moderate", "Active"])
    diet_quality = st.selectbox("Diet Quality", ["Select", "Poor", "Fair", "Good", "Excellent"])
    sleep_quality = st.selectbox("Sleep Quality", ["Select", "Poor", "Fair", "Good", "Excellent"])
    family_history_alzheimers = st.selectbox("Family History of Alzheimer's", ["Select", "Yes", "No"])
    cardiovascular_disease = st.selectbox("Cardiovascular Disease", ["Select", "Yes", "No"])
    diabetes = st.selectbox("Diabetes", ["Select", "Yes", "No"])
    depression = st.selectbox("Depression", ["Select", "Yes", "No"])
    head_injury = st.selectbox("Head Injury", ["Select", "Yes", "No"])
    hypertension = st.selectbox("Hypertension", ["Select", "Yes", "No"])
    systolic_bp = st.number_input("Systolic BP", min_value=50, max_value=200)
    diastolic_bp = st.number_input("Diastolic BP", min_value=30, max_value=120)
    cholesterol_total = st.number_input("Total Cholesterol", min_value=100, max_value=400)
    cholesterol_ldl = st.number_input("LDL Cholesterol", min_value=50, max_value=300)
    cholesterol_hdl = st.number_input("HDL Cholesterol", min_value=20, max_value=100)
    cholesterol_triglycerides = st.number_input("Triglycerides", min_value=50, max_value=500)
    mmse = st.number_input("MMSE Score", min_value=0, max_value=30)
    functional_assessment = st.number_input("Functional Assessment Score", min_value=0, max_value=100)
    memory_complaints = st.selectbox("Memory Complaints", ["Select", "Yes", "No"])
    behavioral_problems = st.selectbox("Behavioral Problems", ["Select", "Yes", "No"])
    adl = st.selectbox("Activities of Daily Living (ADL)", ["Select", "Independent", "Some Assistance", "Dependent"])
    confusion = st.selectbox("Confusion", ["Select", "Yes", "No"])
    disorientation = st.selectbox("Disorientation", ["Select", "Yes", "No"])
    personality_changes = st.selectbox("Personality Changes", ["Select", "Yes", "No"])
    difficulty_completing_tasks = st.selectbox("Difficulty Completing Tasks", ["Select", "Yes", "No"])
    forgetfulness = st.selectbox("Forgetfulness", ["Select", "Yes", "No"])
    diagnosis = st.selectbox("Diagnosis", ["Select", "No AD", "Mild Cognitive Impairment", "Moderate AD", "Severe AD"])

    # Process patient data only if gender is selected
    if gender != "Select" and ethnicity != "Select" and education_level != "Select":
        patient_input = preprocess_patient_data(age, gender)  # You may need to adjust this function to include new features
        st.success("✅ Patient data processed")

with tab2:
    st.subheader("🖼️ Upload MRI Image")
    uploaded_image = st.file_uploader("Upload an MRI scan (JPG, PNG)", type=["jpg", "png"])

    if uploaded_image:
        st.image(uploaded_image, caption="🧠 Uploaded MRI Image", use_column_width=True)
        image_tensor = preprocess_image(uploaded_image)
        st.success("✅ Image preprocessed")

# ---- TAB 3: Run CNN Ensemble Model ----
with tab3:
    st.subheader("⚙️ Run Ensemble Model")
    if st.button("🧠 Run Model"):
        if image_tensor is not None and patient_input is not None:
            with st.spinner("Running models..."):
                # Load models
                cnn_model = load_cnn_model()
                mlp_model = load_mlp_model()

                # Run inference
                cnn_output = cnn_model(image_tensor)
                mlp_output = mlp_model(patient_input)

                cnn_probs = f.softmax(cnn_output, dim=1)
                mlp_probs = f.softmax(mlp_output, dim=1)

                # Ensemble (simple average of softmax outputs, padded if different class counts)
                cnn_probs_resized = cnn_probs[:, :2] if cnn_probs.shape[1] > 2 else cnn_probs
                ensemble_probs = (cnn_probs_resized + mlp_probs) / 2
                final_pred = torch.argmax(ensemble_probs, dim=1).item()
                confidence = torch.max(ensemble_probs).item()

                # Save for tab 4
                st.session_state.pred_label = final_pred
                st.session_state.confidence = round(confidence * 100, 2)

            st.success("✅ Model run complete. Check the Output tab.")
        else:
            st.warning("Please upload an MRI image and complete patient details.")

# ---- TAB 4: Show Output ----
with tab4:
    st.subheader("📊 Output & Severity Prediction")

    if "pred_label" in st.session_state:
        diagnosis_map = {0: "No AD", 1: "Mild Cognitive Impairment", 2: "Moderate AD", 3: "Severe AD"}
        diagnosis = diagnosis_map.get(st.session_state.pred_label, "Unknown")

        st.metric("🧠 Predicted Diagnosis", diagnosis)
        st.metric("🎯 Confidence Level", f"{st.session_state.confidence}%")
        st.metric("🚦 Severity Stage", diagnosis if diagnosis != "No AD" else "None")

        st.markdown("---")
        st.caption("Note: This is generated from a research ensemble model using MRI + patient features.")
    else:
        st.info("Please provide the necessary input to proceed.")