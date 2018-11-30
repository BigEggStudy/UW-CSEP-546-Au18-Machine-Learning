import torch

class SimpleBlinkNeuralNetwork(torch.nn.Module):
    def __init__(self, hiddenNodes = 20):
        super(SimpleBlinkNeuralNetwork, self).__init__()

        # Down sample the image to 12x12
        self.avgPooling = torch.nn.AvgPool2d(kernel_size = 2, stride = 2)

        # Fully connected layer to all the down-sampled pixels to all the hidden nodes
        self.fullyConnectedOne = torch.nn.Sequential(
           torch.nn.Linear(12 * 12, hiddenNodes),
           torch.nn.Sigmoid()
        )

        # Fully connected layer from the hidden layer to a single output node
        self.outputLayer = torch.nn.Sequential(
            torch.nn.Linear(hiddenNodes, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        # Apply the layers created at initialization time in order
        out = self.avgPooling(x)
        out = out.reshape(out.size(0), -1)
        out = self.fullyConnectedOne(out)
        out = self.outputLayer(out)

        return out