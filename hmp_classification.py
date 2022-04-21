import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, random_split
from torch.utils.data import DataLoader
import config
from sklearn.model_selection import train_test_split
import random
from hmp_dataset import HMP
import numpy as np
from torchvision import models
from sklearn.metrics import confusion_matrix

# img aug
transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.2)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.14846641, 0.11407742, 0.10433987], [0.13520855, 0.11859692, 0.11858473])
])

dataset = HMP(transforms=transform)
dataset_len = len(dataset)
test_size = int(config.test_size * dataset_len)
train_dataset, test_dataset = random_split(dataset, [test_size, dataset_len - test_size],
                                           torch.Generator().manual_seed(1))
# prepare dataset
batch_size = config.batch_size
train_dataloader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=6)

test_dataloader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=6)

# design module
in_channel = 3
out_channel = len(config.ACTIONS)
padding = 0
kernel_size = 3


class CnnNet(nn.Module):
    def __init__(self):
        super(CnnNet, self).__init__()
        # design the cnn net
        self.conv1 = nn.Conv2d(in_channels=in_channel,
                               out_channels=32,
                               kernel_size=kernel_size,
                               stride=2,
                               padding=padding)
        # print(self.conv1.weight.shape)
        self.conv2 = nn.Conv2d(in_channels=32,
                               out_channels=64,
                               kernel_size=3,
                               stride=3,
                               padding=padding)
        # print(self.conv2.weight.shape)
        self.pooling = nn.MaxPool2d(2)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc = nn.Linear(5184, out_channel)

    def forward(self, x):
        # design fp
        # print("1:", x.size())
        batch_size = x.size(0)
        x = self.pooling(F.relu(self.conv1(x)))
        x = self.bn1(x)
        # print("conv1:", x.size())
        x = self.pooling(F.relu(self.conv2(x)))
        # print("conv2:", x.size())
        x = self.bn2(x)
        x = x.view(batch_size, -1)  # flatten
        x = self.fc(x)
        # print("fc:", x.size())
        x = F.softmax(x, dim=1)
        return x


module = CnnNet()

# loss and optimizer
criterion = nn.CrossEntropyLoss()
# criterion = nn.BCELoss()
optimal = optim.SGD(module.parameters(), lr=0.001, momentum=0.5)


# train
def train(epoch):
    module.train()
    loss_value = 0
    for i, (x, y) in enumerate(train_dataloader, 0):
        # print("iter:", i + 1)
        # FP
        y_pred = module(x)
        # get loss
        loss = criterion(y_pred, y)
        # BP
        optimal.zero_grad()
        loss.backward()
        # update
        optimal.step()
        loss_value += loss.item()
    print("Epoch:%d,loss:%lf" % (epoch + 1, loss_value / (i + 1)))

# test
def test(epoch):
    module.eval()
    correct = 0
    total = 0
    pred = []
    y_true = []
    for i, (x, y) in enumerate(test_dataloader, 0):
        y_pred = module(x)  # [0.1,0.2,0.3,0.1,0.1,0.1,0.1]  batch,class_num
        # print(type(y_pred))  # <class 'torch.Tensor'>
        # print(y_pred)
        # print("y_pred:", y_pred.size())
        _, predicted = torch.max(y_pred, dim=1)  # [2,idx1,idx2,...]
        # print(predicted)
        # print(y)
        total += predicted.size(0)
        correct += (predicted == y).sum().item()
        pred.extend(predicted)
        y_true.extend(y)
    print("Epoch:%d,Accuracy:%f" % (epoch + 1, correct / total))
    if (correct / total > 0.65):
        # save model
        torch.save(module.state_dict(), r'./model/model-%f.pth' % (correct / total))
        print(confusion_matrix(y_true, pred))
        # classes 7 classes
        classes = config.ACTIONS
        # get confusion matrix
        cm = confusion_matrix(y_true, pred)
        plot_confusion_matrix(cm, classes, 'confusion_matrix-%f.png' % (correct / total), title='confusion matrix')


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrix(cm, classes, savename, title='Confusion Matrix'):
    plt.figure(figsize=(12, 8), dpi=100)
    np.set_printoptions(precision=2)

    # The probability value of each cell in the confusion matrix
    ind_array = np.arange(len(classes))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val]
        if c > 0.001:
            plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=15, va='center', ha='center')

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.binary)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, classes, rotation=90)
    plt.yticks(xlocations, classes)
    plt.ylabel('Actual label')
    plt.xlabel('Predict label')

    # offset the tick
    tick_marks = np.array(range(len(classes))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)

    # show confusion matrix
    plt.savefig("./model/" + savename, format='png')
    plt.show()


if __name__ == '__main__':
    for epoch in range(200):
        train(epoch)
        test(epoch)
