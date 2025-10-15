import torch
import torchvision
import random
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as f
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np


batch_size = 64

# This section is responsible for altering the batch to introduce variation
# In the images to attempt to reduce overfitting

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  # Cropping the images
    transforms.RandomHorizontalFlip(p=0.5),  # 50% chance of image being flipped
    transforms.RandomRotation(10),  # Slight rotation as images are small
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    # Numbers specificially calculated from the CIFAR dataset
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
])


transform_test = transforms.Compose([
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
])

# Load the CIFAR-10 dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)


# Create DataLoaders for training and testing
train_loader = torch.utils.data.DataLoader(trainset, num_workers=2, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(testset, num_workers=2, batch_size=batch_size, shuffle=False)


# This is the amount of Experts / ConvLayer classes
K = 5

# The amount of channels in the CNN in the expert branch
Channels = 256

# What channeles will be divided by in the expert branch
r = 8

X, y = next(iter(train_loader))  # first batch


class ConvLayer(nn.Module):
    def __init__(self, given_kernal_size):
        super().__init__()
        padding_size = given_kernal_size // 2

        self.conv1 = nn.Conv2d(in_channels=Channels, out_channels=Channels, kernel_size=given_kernal_size, padding=padding_size)
        self.bn1 = nn.BatchNorm2d(Channels)
        self.relu = torch.nn.ReLU()
        self.dropout = nn.Dropout(0.3)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


# From my understanding, this wieghts each channel.
class ExpertBranch(nn.Module):
    def __init__(self):
        super().__init__()
        # Expert branch
        # Adaptive pool so I don't need to worry about the image size
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = torch.nn.Linear(Channels, Channels // r)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(Channels // r, K)
        self.softmax = torch.nn.Softmax(dim=1)


    def forward(self, x):
        x = self.avgpool(x)  # shape: (64, 128, 1, 1)
        x = x.view(x.size(0), -1)  # shape: (B, C)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


class Stem(nn.Module):
    def __init__(self):
        super().__init__()
        # Here I use two CNNS
        self.conv1 = nn.Conv2d(3, Channels//2, 3, padding=1)  # Initial Feature Extraction
        self.bn1 = nn.BatchNorm2d(Channels//2)
        self.conv2 = nn.Conv2d(Channels//2, Channels, 3, padding=1)  # Further Feature Extraction
        self.bn2 = nn.BatchNorm2d(Channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class combinedModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.stem = Stem()
        self.expertbranch = ExpertBranch()
        self.convs = nn.ModuleList([ConvLayer(3) for i in range(K)])

        self.fc1 = nn.Linear(Channels, 1024)  # Very Large - Could be too large
        self.fc2 = nn.Linear(1024, 128)  # Gradually going down
        self.fc3 = nn.Linear(128, 32)
        self.fc4 = nn.Linear(32, 10)

        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(32)

        self.dropout = nn.Dropout(0.3)

    def forward(self, x):

        x = self.stem(x)
        expert_out = self.expertbranch(x)
        # Do the 1st one outside of the loop so you can have weighted_sum
        weighted_sum = self.convs[0](x) * expert_out[:, 0].view(-1, 1, 1, 1)

        # use the value from expert to weight the CNN block
        for i in range(1, len(self.convs)):

            column = expert_out[:, i]

            x_out = self.convs[i](x)

            weighted_sum += x_out * column.view(-1, 1, 1, 1)


        weighted_sum_flat = f.adaptive_avg_pool2d(weighted_sum, (1, 1))
        weighted_sum_flat = weighted_sum_flat.squeeze(-1).squeeze(-1)

        x = f.relu(self.bn1(self.fc1(weighted_sum_flat)))
        x = self.dropout(x)  # Dropout ensures more generalisation
        x = f.relu(self.bn2(self.fc2(x)))
        x = f.relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)

        return x


if __name__ == "__main__":
    m = combinedModel()

    num_epochs = 100

    optimiser = torch.optim.AdamW(m.parameters(), lr=0.001)  # Adam with wieght decay

    scheduler = ReduceLROnPlateau(optimiser, mode='min', factor=0.5, patience=3, verbose=True, min_lr=1e-6)

    loss = nn.CrossEntropyLoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m.to(device)  # Move model to GPU if available

    train_losses = []
    train_accuracies = []  # New list to track training accuracy
    test_accuracies = []

    for epoch in range(num_epochs):
        # Training phase
        m.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)

            # Forward pass
            y_hat = m(X)
            l = loss(y_hat, y)

            # Calculate training accuracy
            _, predicted = torch.max(y_hat.data, 1)
            total_train += y.size(0)
            correct_train += (predicted == y).sum().item()

            # Backward pass and optimization
            optimiser.zero_grad()
            l.backward()
            optimiser.step()

            running_loss += l.item() * X.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        train_accuracy = 100 * correct_train / total_train

        train_losses.append(epoch_loss)
        train_accuracies.append(train_accuracy)  # Store training accuracy

        # Evaluation phase
        m.eval()
        correct_test, total_test = 0, 0
        validation_loss = 0.0

        with torch.no_grad():
            for X_test, y_test in test_loader:
                X_test, y_test = X_test.to(device), y_test.to(device)
                outputs = m(X_test)
                loss_value = loss(outputs, y_test)
                validation_loss += loss_value.item() * X_test.size(0)
                _, predicted = torch.max(outputs, 1)
                total_test += y_test.size(0)
                correct_test += (predicted == y_test).sum().item()

        validation_loss /= len(test_loader.dataset)
        scheduler.step(validation_loss)  # Update scheduler based on validation loss

        test_accuracy = 100 * correct_test / total_test
        test_accuracies.append(test_accuracy)

        # Use epoch+1 to start counting from 1 instead of 0 (as epoch 0 doesn't make sense)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%")

    # plotting after training
    plt.figure(figsize=(15, 5))

    epochs = list(range(1, num_epochs + 1))[:len(train_losses)]

    # Plot for losses
    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    # plot for training accuracy
    plt.subplot(1, 3, 2)
    plt.plot(epochs, train_accuracies, label='Training Accuracy', color='blue')
    plt.title('Training Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.legend()

    # plot for test accuracy
    plt.subplot(1, 3, 3)
    plt.plot(epochs, test_accuracies, label='Test Accuracy', color='green')
    plt.title('Test Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_accuracies, label='Training Accuracy', color='blue')
    plt.plot(epochs, test_accuracies, label='Test Accuracy', color='green')
    plt.title('Training vs Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
