import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Define transformations
transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),  # Converts images to [0, 1] and turns into tensors
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
])

# Load datasets
train_dir = 'C:/Users/JILLIAN CLAIRE/PycharmProjects/PythonProject/datasets/Combined Dataset/train'
test_dir = 'C:/Users/JILLIAN CLAIRE/PycharmProjects/PythonProject/datasets/Combined Dataset/test'

train_dataset = datasets.ImageFolder(train_dir, transform=transform)
test_dataset = datasets.ImageFolder(test_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# View some images
images, labels = next(iter(train_loader))
plt.figure(figsize=(10, 5))
for i in range(6):
    plt.subplot(2, 3, i + 1)
    img = images[i].permute(1, 2, 0)  # rearrange for display
    plt.imshow(img.numpy() * 0.5 + 0.5)  # unnormalize
    plt.title(train_dataset.classes[labels[i]])
    plt.axis('off')
plt.show()

import torch.nn as nn
import torch.nn.functional as F

class AlzheimerCNN(nn.Module):
    def __init__(self):
        super(AlzheimerCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(64 * 35 * 35, 128)
        self.fc2 = nn.Linear(128, len(train_dataset.classes))  # output for each class

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 35 * 35)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
