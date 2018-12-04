import torch
import torch.nn as nn
import torch.nn.functional as F

class BlinkNeuralNetwork(nn.Module):
    def __init__(self):
        super(BlinkNeuralNetwork, self).__init__()

        self.conv1_1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            nn.ReLU()
        )
        self.conv1_1_bn = nn.BatchNorm2d(6)
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=3, stride=1, padding=0),
            nn.ReLU()
        )
        self.conv2_1_bn = nn.BatchNorm2d(16)

        self.conv1_2 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=3, stride=1, padding=0),
            nn.ReLU()
        )
        self.conv1_2_bn = nn.BatchNorm2d(6)
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=3, stride=1, padding=0),
            nn.ReLU()
        )
        self.conv2_2_bn = nn.BatchNorm2d(16)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.fc1 = nn.Sequential(
            nn.Linear(16*4*4 + 16*4*4, 120, bias=True),
            nn.Sigmoid(),
            nn.Dropout(0.05)
        )
        self.fc1_bn = nn.BatchNorm1d(120)
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84, bias=True),
            nn.Sigmoid(),
            nn.Dropout(0.05)
        )
        self.fc2_bn = nn.BatchNorm1d(84)

        # Fully connected layer from the hidden layer to a single output node
        self.outputLayer = nn.Sequential(
            nn.Linear(84, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Size is (1, 24, 24)
        out1 = self.conv1_1_bn(self.conv1_1(x))
        # Size is (6, 20, 20)
        out1 = self.pool(out1)
        # Size is (6, 10, 10)
        out1 = self.conv2_1_bn(self.conv2_1(out1))
        # Size is (16, 8, 8)
        out1 = self.pool(out1)
        # Size is (16, 4, 4)
        out1 = out1.reshape(out1.size(0), -1)
        # Size is (1, 16 * 4 * 4)

        # Size is (1, 24, 24)
        out2 = self.conv1_2_bn(self.conv1_2(x))
        # Size is (6, 22, 22)
        out2 = self.pool(out2)
        # Size is (6, 11, 11)
        out2 = self.conv2_2_bn(self.conv2_2(out2))
        # Size is (16, 9, 9)
        out2 = self.pool(out2)
        # Size is (16, 4, 4)
        out2 = out2.reshape(out2.size(0), -1)
        # Size is (1, 16 * 4 * 4)

        out = torch.cat((out1, out2), -1)

        out = self.fc1_bn(self.fc1(out))
        # Size is (1, 120)
        out = self.fc2_bn(self.fc2(out))
        # Size is (1, 84)
        out = self.outputLayer(out)
        # Size is (1, 1)
        return out