import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- MLP Model ---
class PatientMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_classes=2):
        super(PatientMLP, self).__init__()
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

def save_metrics(y_true, y_pred, y_prob, class_names, output_dir="results/patient_model"):
    os.makedirs(output_dir, exist_ok=True)

    # --- Classification Report ---
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
        f.write(report)
    print("✅ classification_report.txt saved.")

    # --- Confusion Matrix ---
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()
    print("✅ confusion_matrix.png saved.")

    # --- ROC + AUC ---
    if len(class_names) == 2:
        # Binary classification
        fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])  # Probabilities for class "1"
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}", color="darkorange")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.title("ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "roc_curve.png"))
        plt.close()
        print("✅ roc_curve.png saved.")

        with open(os.path.join(output_dir, "auc_scores.txt"), "w") as f:
            f.write(f"AUC: {roc_auc:.4f}\n")
        print("✅ auc_scores.txt saved.")
    else:
        # Multi-class fallback (previous version here if needed)
        pass


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    # --- Load data ---
    data = torch.load("D:/AD_Detection/preprocessing/data/processed/patient_data.pt")
    X, y = data['features'], data['labels']

    total = X.shape[0]
    split = int(0.8 * total)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    train_ds = torch.utils.data.TensorDataset(X_train, y_train)
    val_ds = torch.utils.data.TensorDataset(X_val, y_val)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=32, shuffle=False)

    model = PatientMLP(input_dim=X.shape[1], num_classes=len(torch.unique(y))).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # --- Training ---
    epochs = 20
    for epoch in range(epochs):
        model.train()
        correct, total, running_loss = 0, 0, 0.0

        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]")
        for features, labels in loop:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            running_loss += loss.item()

            loop.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "Acc": f"{(correct/total)*100:.2f}%"
            })

        print(f"✅ Epoch {epoch+1} | Loss: {running_loss/len(train_loader):.4f} | Acc: {(correct/total)*100:.2f}%")

    # --- Evaluation ---
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for features, labels in val_loader:
            features = features.to(device)
            outputs = model(features)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    # --- Save Metrics ---
    class_names = ["No AD", "AD"] if len(torch.unique(y)) == 2 else [f"Class {i}" for i in torch.unique(y).tolist()]
    save_metrics(all_labels, all_preds, np.array(all_probs), class_names)

    # --- Save Model ---
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/patient_model.pt")
    print("✅ Model saved to checkpoints/patient_model.pt")

if __name__ == "__main__":
    train()
