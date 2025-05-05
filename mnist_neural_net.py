import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


# Hyperparameters
input_channels = 1
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 5


# CNN Model
#what is a cnn? why use a cnn and feed the hidden states through
#using a transformer is better than a cnn 
#maxpooling and feature maps
class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=8,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )
        self.fc1 = nn.Linear(16 * 7 * 7, num_classes)
        
    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = self.pool(x)
        x = nn.functional.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x


def train_and_save_model():
    # Training setup
    transform = transforms.ToTensor()
    train_dataset = torchvision.datasets.MNIST(
        root='./data', 
        train=True, 
        transform=transform,
        download=True
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, 
        batch_size=64, 
        shuffle=True
    )
    
    # Model initialization
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    for epoch in range(5):
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
        print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')
    
    # Save model only (no testing)
    torch.save(model.state_dict(), 'mnist_cnn.pth')
    print("Model saved to mnist_cnn.pth")

if __name__ == '__main__':
    train_and_save_model()