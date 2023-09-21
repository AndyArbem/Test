import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import time
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from torchvision import transforms, datasets
from torch.utils.data.dataloader import DataLoader
from mnist_lstm import GetNet


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = datasets.CIFAR10(root='/Code/CIFAR10', train=True, download=True, transform=transform)
    testset = datasets.CIFAR10(root='/Code/CIFAR10', train=False, download=True, transform=transform)

    BATCH_SIZE = 128
    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=6, pin_memory=True)
    test_loader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True, num_workers=6, pin_memory=True)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    net = GetNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    # optimizer = optim.Adam(net.parameters(), lr=0.01)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
    EPOCHS = 200
    for epoch in range(EPOCHS):
        train_loss = 0.0
        for i, (data, labels) in enumerate(train_loader):
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        print("Epoch: %d, Batch: %5d, Loss: %.3f" % (epoch + 1, i + 1, train_loss / len(train_loader.dataset)))
    PATH = './cifar_net.pth'
    torch.save(net.state_dict(), PATH)
    # scheduler.step()
    model = GetNet()
    model.load_state_dict(torch.load(PATH))
    correct = 0
    total = 0

    with torch.no_grad():
        for i, (data, labels) in enumerate(test_loader):
            outputs = model(data)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
        print("Test accuracy in 1000 pictures: {:.3f}%".format(correct / total * 100))


if __name__ == '__main__':
    main()
