import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


'''
simple image classifier on the MNIST dataset, numbers 0-9, run by 'python ml_model_basic.py' in terminal

images in dataser [0,1], 28x28
'''


# Transform: Convert images to tensors and normalize
# Compose combines multiple preprocessing steps, ToTensor convert PIL image to tensor, noarmalize take pixel values from [0,1] to [-1,1]
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Load dataset
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)#training dataset
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)#test dataset

# DataLoader: Batches of 64 samples
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)#class that organizes data to batches, here random
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)#test dont have to be randomized


class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten image
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # No activation, as we'll use CrossEntropyLoss
        return x

# Initialize model
model = SimpleNN()


criterion = nn.CrossEntropyLoss()  # Loss function for classification
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer

num_epochs = 5

for epoch in range(num_epochs):
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")


correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")
