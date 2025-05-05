# test_model.py
import torch
from mnist_neural_net import CNN
from torchvision import datasets, transforms

def load_test_data():
    transform = transforms.ToTensor()
    test_dataset = datasets.MNIST(
        root='./data', 
        train=False, 
        transform=transform
    )
    return torch.utils.data.DataLoader(
        dataset=test_dataset, 
        batch_size=64, 
        shuffle=False
    )

def test_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNN().to(device)
    model.load_state_dict(torch.load('mnist_cnn.pth'))
    model.eval()
    
    test_loader = load_test_data()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    print(f'Test Accuracy: {100 * correct / total:.2f}%')

if __name__ == '__main__':
    test_model()