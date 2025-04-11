import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm

class ResNetClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super(ResNetClassifier, self).__init__()
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # grayscale
        self.model.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)  # multi-class output
        )

    def forward(self, x):
        return self.model(x)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    # --- Transforms ---
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # --- Load dataset ---
    data_dir = "D:/AD_Detection/data/processed/train"
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    class_names = dataset.classes
    num_classes = len(class_names)
    print(f"✅ Classes found: {class_names}")

    # --- Train/Val Split ---
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # --- Model, Loss, Optimizer ---
    model = ResNetClassifier(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # --- Training Loop ---
    epochs = 10
    for epoch in range(epochs):
        model.train()
        correct, total, running_loss = 0, 0, 0.0

        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]")
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
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

    # --- Save model ---
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/resnet50_model.pth")
    print("✅ Model saved to checkpoints/resnet50_model.pth")

if __name__ == "__main__":
    train()
