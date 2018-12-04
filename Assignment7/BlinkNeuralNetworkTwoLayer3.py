import torch.nn as nn
import torch.nn.functional as F

class BlinkNeuralNetwork(nn.Module):
    def __init__(self):
        super(BlinkNeuralNetwork, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=3, stride=1, padding=0),
            nn.ReLU()
        )
        self.conv1_bn = nn.BatchNorm2d(6)
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=3, stride=1, padding=0),
            nn.ReLU()
        )
        self.conv2_bn = nn.BatchNorm2d(16)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120, bias=True),
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
        out = self.conv1_bn(self.conv1(x))
        # Size is (conv1_out, 22, 22)
        out = self.pool(out)
        # Size is (conv1_out, 11, 11)
        out = self.conv2_bn(self.conv2(out))
        # Size is (conv2_out, 9, 9)
        out = self.pool(out)
        # Size is (conv2_out, 5, 5)
        out = out.reshape(out.size(0), -1)
        # Size is (1, conv2_out * 5 * 5)
        out = self.fc1_bn(self.fc1(out))
        # Size is (1, nn1_out)
        out = self.fc2_bn(self.fc2(out))
        # Size is (1, nn2_out)
        out = self.outputLayer(out)
        # Size is (1, 1)
        return out