## Some of this references my answers to previous assignments.
##  Replace my references with references to your answers to those assignments.

## IMPORTANT NOTE !!
## Remember to install the Pillow library (which is required to execute 'import PIL')
## Remember to install Pytorch: https://pytorch.org/get-started/locally/ (if you want GPU you need to figure out CUDA...)

from PIL import Image
import torchvision.transforms as transforms
import torch
import numpy as np

import Assignment5Support
import EvaluationsStub
import CrossValidation
crossValidation = CrossValidation.CrossValidation(5)

if __name__=="__main__":
    kDataPath = "../dataset_B_Eye_Images"

    (xRaw, yRaw) = Assignment5Support.LoadRawData(kDataPath, includeLeftEye = True, includeRightEye = True)
    (xTrainRaw, yTrainRaw, xTestRaw, yTestRaw) = Assignment5Support.TrainTestSplit(xRaw, yRaw, percentTest = .25)

    print("Train is %f percent closed." % (sum(yTrainRaw)/len(yTrainRaw)))
    print("Test is %f percent closed." % (sum(yTestRaw)/len(yTestRaw)))

    # Load the images and then convert them into tensors (no normalization)
    xTrainImages = [ Image.open(path) for path in xTrainRaw ]
    xTrain = torch.stack([ transforms.ToTensor()(image) for image in xTrainImages ])

    xTestImages = [ Image.open(path) for path in xTestRaw ]
    xTest = torch.stack([ transforms.ToTensor()(image) for image in xTestImages ])

    yTrain = torch.Tensor([ [ yValue ] for yValue in yTrainRaw ])
    yTest = torch.Tensor([ [ yValue ] for yValue in yTestRaw ])

    import SimpleBlinkNeuralNetwork

    ############################

    # Create the model and set up:
    #     the loss function to use (Mean Square Error)
    #     the optimization method (Stochastic Gradient Descent) and the step size
    model = SimpleBlinkNeuralNetwork.SimpleBlinkNeuralNetwork(hiddenNodes = 5)
    lossFunction = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)

    for i in range(500):
        # Do the forward pass
        yTrainPredicted = model(xTrain)

        # Compute the training set loss
        loss = lossFunction(yTrainPredicted, yTrain)
        print(i, loss.item())

        # Reset the gradients in the network to zero
        optimizer.zero_grad()

        # Backprop the errors from the loss on this iteration
        loss.backward()

        # Do a weight update step
        optimizer.step()

    yTestPredicted = model(xTest)

    yPred = [ 1 if pred > 0.5 else 0 for pred in yTestPredicted ]

    print("Accuracy simple:", EvaluationsStub.Accuracy(yTest, yPred))
    simpleAccuracy = EvaluationsStub.Accuracy(yTest, yPred)
