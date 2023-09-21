import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets
from torch.utils.data.dataloader import DataLoader
import numpy as np
import matplotlib.pyplot as plt

# def imshow(img):
#     img = img / 2 + 0.5
#     np_img = img.numpy()
#     plt.imshow(np.transpose(np_img, (1, 2, 0)))
#     plt.show()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = datasets.CIFAR10(root='/Code/CIFAR10', train=True, download=True, transform=transform)
testset = datasets.CIFAR10(root='/Code/CIFAR10', train=False, download=True, transform=transform)

BATCH_SIZE = 100
train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 32*32*3 -> 28*28*6
        self.conv1 = nn.Conv2d(3, 6, 5)
        # 28*28*6 -> 14*14*6
        self.pool = nn.MaxPool2d(2, 2)
        # 14*14*6 -> 10*10*16
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def GetNet():
    return Net()


if __name__ == '__main__':
    net = Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    EPOCHS = 2
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
    model = Net()
    model.load_state_dict(torch.load(PATH))
    correct = 0
    total = 0

    with torch.no_grad():
        for i, (data, labels) in enumerate(test_loader):
            outputs = model(data)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
        print("Test accuracy in {} pictures: {:.3f}%".format(i + 1, correct / total * 100))
