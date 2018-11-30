## Some of this references my answers to previous assignments.
##  Replace my references with references to your answers to those assignments.

## IMPORTANT NOTE !!
## Remember to install the Pillow library (which is required to execute 'import PIL')
## Remember to install Pytorch: https://pytorch.org/get-started/locally/ (if you want GPU you need to figure out CUDA...)

from PIL import Image
import torchvision
import torchvision.transforms as transforms
import torch
import numpy as np

import Assignment5Support
import EvaluationsStub
import CrossValidation
crossValidation = CrossValidation.CrossValidation(5)
import Featurize

if __name__=='__main__':
    kDataPath = '../dataset_B_Eye_Images'

    (xRaw, yRaw) = Assignment5Support.LoadRawData(kDataPath, includeLeftEye = True, includeRightEye = True)
    (xTrainRaw, yTrainRaw, xTestRaw, yTestRaw) = Assignment5Support.TrainTestSplit(xRaw, yRaw, percentTest = .25)

    print('Train is %f percent closed.' % (sum(yTrainRaw)/len(yTrainRaw)))
    print('Test is %f percent closed.' % (sum(yTestRaw)/len(yTestRaw)))

    # Load the images and then convert them into tensors (no normalization)
    xTrainImages = [ Image.open(path) for path in xTrainRaw ]
    xTrain, yTrain = Featurize.Featurize(xTrainImages, yTrainRaw, transforms_data=False)
    print(f'Training data size: {xTrain.size()}')

    xTestImages = [ Image.open(path) for path in xTestRaw ]
    xTest, yTest = Featurize.Featurize(xTestImages, yTestRaw, transforms_data=False)

    ############################

    print('Set the random seed')
    seed = 100
    np.random.seed(seed)
    torch.manual_seed(seed)

    ############################

    print('========= Try Simple Neural Network ==========')
    import SimpleBlinkNeuralNetwork
    # Create the model and set up:
    #     the loss function to use (Mean Square Error)
    #     the optimization method (Stochastic Gradient Descent) and the step size
    model = SimpleBlinkNeuralNetwork.SimpleBlinkNeuralNetwork(hiddenNodes = 5)

    print('Training Model')
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

    print('Predict Test Data')
    yTestPredicted = model(xTest)
    yPred = [ 1 if pred > 0.5 else 0 for pred in yTestPredicted ]
    print('Accuracy simple:', EvaluationsStub.Accuracy(yTest, yPred))
    simpleAccuracy = EvaluationsStub.Accuracy(yTest, yPred)

    ############################

    import BlinkNeuralNetwork
    import BlinkNeuralNetworkTwoLayer

    print('========= Try My Neural Network ==========')
    model = BlinkNeuralNetwork.BlinkNeuralNetwork()

    print('Training Model')
    lossFunction = torch.nn.MSELoss(reduction='sum')
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters())
    for i in range(500):
        yTrainPredicted = model(xTrain)
        loss = lossFunction(yTrainPredicted, yTrain)
        print(i, loss.item() / len(xTrain))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('Predict Test Data')
    yTestPredicted = model(xTest)
    yPred = [ 1 if pred > 0.5 else 0 for pred in yTestPredicted ]
    print('Accuracy simple:', EvaluationsStub.Accuracy(yTest, yPred))

    ############################

    print('========= Try Find Best Model for Convolution Neural Network ==========')
    best_accuracy = 0
    best_parameter = (0, 0.0)
    for layer in [1, 2]:
        for optimizer_type in ['SGD', 'Adam']:
            for pool in ['Average', 'Max']:
                for conv1_out, conv2_out in [(6, 12), (6, 16), (8, 16), (16, 24), (16, 32)]:
                    for conv1_kernel_size, conv2_kernel_size in [(5, 3), (5, 5)]:
                        for nn1_out, nn2_out in [(20, 0), (40, 0), (60, 0), (80, 0)] if layer == 1 else [(20, 10), (40, 30), (40, 40), (80, 60), (120, 60), (120, 84)]:
                            for learning_rate in [0.01, 0.05, 0.1, 0.5, 1]:
                                for momentum in [0.15, 0.25, 0.33, 0.5, 0.75] if optimizer_type == 'SGD' else [0]:
                                    for iteration in [500, 750] if layer == 1 else [1000, 1250]:
                                        print(f'Training a {layer} fully connect layer(s) Model with {optimizer_type} optimizer and {pool} pooling')
                                        print(f'2 convolution layer output channel are ({conv1_out}, {conv2_out}), and kernel size are ({conv1_kernel_size}, {conv2_kernel_size})')
                                        print(f'2 fully connect layers output channel are ({nn1_out}, {nn2_out})')
                                        print(f'With learning rate {learning_rate}, momentum beta {momentum}, iteration {iteration}')

                                        accuracy = crossValidation.validate(xTrain, yTrain, layer, optimizer_type, pool, iteration, learning_rate, momentum, conv1_out, conv1_kernel_size, conv2_out, conv2_kernel_size, nn1_out, nn2_out)
                                        (lower, upper) = EvaluationsStub.Bound(accuracy, len(xTrainRaw))
                                        print('Accuracy from Cross Validation is %f, with lower bound %f and upper bound %f' % (accuracy, lower, upper))
                                        if best_accuracy < accuracy:
                                            best_accuracy = accuracy
                                            best_parameter = (layer, optimizer_type, pool, iteration, learning_rate, momentum, conv1_out, conv1_kernel_size, conv2_out, conv2_kernel_size, nn1_out, nn2_out)

    (layer, optimizer_type, pool, iteration, learning_rate, momentum, conv1_out, conv1_kernel_size, conv2_out, conv2_kernel_size, nn1_out, nn2_out) = best_parameter
    print('When')
    print(f'Training a {layer} layer(s) Model with {optimizer_type} optimizer and {pool} pooling')
    print(f'2 convolution layer output channel are ({conv1_out}, {conv2_out}), and kernel size are ({conv1_kernel_size}, {conv2_kernel_size})')
    print(f'2 fully connect layer output channel are ({nn1_out}, {nn2_out})')
    print(f'With learning rate {learning_rate}, momentum beta {momentum}, iteration {iteration}')
    print('Convolution Neural Networks has best accuracy %f' % best_accuracy)
