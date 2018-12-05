import torch.nn as nn
import torch.nn.functional as F

class BlinkNeuralNetwork(nn.Module):
    def __init__(self, pool = 'Max', conv1_out = 16, conv1_kernel_size = 5, conv2_out = 16, conv2_kernel_size = 3, nn1_out = 120, nn2_out = 84, dropout = 0.4):
        super(BlinkNeuralNetwork, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, conv1_out, kernel_size=5, stride=1, padding=0),
            nn.ReLU()
        )
        self.conv1_bn = nn.BatchNorm2d(conv1_out)
        self.conv2 = nn.Sequential(
            nn.Conv2d(conv1_out, conv2_out, kernel_size=3, stride=1, padding=0),
            nn.ReLU()
        )
        self.conv2_bn = nn.BatchNorm2d(conv2_out)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0) if pool == 'Average' else nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Sequential(
            nn.Linear(conv2_out * 4 * 4, nn1_out, bias=True),
            nn.Sigmoid(),
            nn.Dropout(dropout)
        )
        self.fc1_bn = nn.BatchNorm1d(nn1_out)
        self.fc2 = nn.Sequential(
            nn.Linear(nn1_out, nn2_out, bias=True),
            nn.Sigmoid(),
            nn.Dropout(dropout)
        )
        self.fc2_bn = nn.BatchNorm1d(nn2_out)

        # Fully connected layer from the hidden layer to a single output node
        self.outputLayer = nn.Sequential(
            nn.Linear(nn2_out, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Size is (1, 24, 24)
        out = self.conv1_bn(self.conv1(x))
        # Size is (conv1_out, 20, 20)
        out = self.pool(out)
        # Size is (conv1_out, 10, 10)
        out = self.conv2_bn(self.conv2(out))
        # Size is (conv2_out, 8, 8)
        out = self.pool(out)
        # Size is (conv2_out, 4, 4)
        out = out.reshape(out.size(0), -1)
        # Size is (1, conv2_out * 4 * 4)
        out = self.fc1_bn(self.fc1(out))
        # Size is (1, nn1_out)
        out = self.fc2_bn(self.fc2(out))
        # Size is (1, nn2_out)
        out = self.outputLayer(out)
        # Size is (1, 1)
        return out