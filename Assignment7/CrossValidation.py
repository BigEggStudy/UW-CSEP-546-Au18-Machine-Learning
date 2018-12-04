import math
import numpy as np
from joblib import Parallel, delayed
import torchvision
import torchvision.transforms as transforms
import torch
from collections import defaultdict

import BlinkNeuralNetwork
import BlinkNeuralNetworkTwoLayer
import BlinkNeuralNetworkTwoLayer2
import BlinkNeuralNetworkTwoLayerSoftMax

class CrossValidation(object):
    def __init__(self, k):
        self.k = round(k)

    def __splitDataFold(self, x, y, index):
        x_chunk = torch.chunk(x, self.k, 0)
        y_chunk = torch.chunk(y, self.k, 0)

        if index == 0:
            newTrainX = torch.cat(x_chunk[1:], 0)
            newTrainY = torch.cat(y_chunk[1:], 0)
        elif index == self.k - 1:
            newTrainX = torch.cat(x_chunk[:-1], 0)
            newTrainY = torch.cat(y_chunk[:-1], 0)
        else:
            newTrainX1 = torch.cat(x_chunk[:index], 0)
            newTrainY1 = torch.cat(y_chunk[:index], 0)
            newTrainX2 = torch.cat(x_chunk[(index + 1):], 0)
            newTrainY2 = torch.cat(y_chunk[(index + 1):], 0)
            newTrainX = torch.cat((newTrainX1, newTrainX2), 0)
            newTrainY = torch.cat((newTrainY1, newTrainY2), 0)

        newValidationX = x_chunk[index]
        newValidationY = y_chunk[index]

        return (newTrainX, newTrainY, newValidationX, newValidationY)

    def __countCorrect(self, yPredicted, y):
        count = 0
        for i in range(len(y)):
            if y[i] == yPredicted[i]:
                count += 1

        return count

    def validate(self, x, y, layer = 1, optimizer_type = 'SGD', pool = 'Max', max_iteration = 500, learning_rate = 0.05, momentum = 0.25, conv1_out = 16, conv1_kernel_size = 5, conv2_out = 16, conv2_kernel_size = 3, nn1_out = 20, nn2_out = 20):
        def validation_core(i, x, y):
            (foldTrainX, foldTrainY, foldValidationX, foldValidationY) = self.__splitDataFold(x, y, i)

            model = BlinkNeuralNetworkTwoLayer.BlinkNeuralNetwork(pool=pool, conv1_out=conv1_out, conv1_kernel_size=conv1_kernel_size, conv2_out=conv2_out, conv2_kernel_size=conv2_kernel_size, nn1_out=nn1_out, nn2_out=nn2_out) \
                    if layer == 2 else \
                    BlinkNeuralNetwork.BlinkNeuralNetwork(pool=pool, conv1_out=conv1_out, conv1_kernel_size=conv1_kernel_size, conv2_out=conv2_out, conv2_kernel_size=conv2_kernel_size, nn1_out=nn1_out)
            lossFunction = torch.nn.MSELoss(reduction='sum')
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) if optimizer_type == 'Adam' else torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

            corrects = []
            model.train()
            for i in range(max_iteration):
                yTrainPredicted = model(foldTrainX)
                loss = lossFunction(yTrainPredicted, foldTrainY)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if i > 100 and (i + 1) % 100 == 0:
                    model.eval()
                    yPredicted = model(foldValidationX)
                    yPred = [ 1 if pred > 0.5 else 0 for pred in yPredicted ]
                    model.train()

                    corrects.append((i + 1, self.__countCorrect(yPred, foldValidationY)))
            return corrects

        totalCorrects = defaultdict(int)
        for i in range(self.k):
            results = validation_core(i, x, y)
            for iteration, correct_count in results:
                totalCorrects[iteration] += correct_count
        # totalCorrects = Parallel(n_jobs=6)(delayed(validation_core)(i, x, y) for i in range(self.k))
        # accuracy = sum(totalCorrects) / len(x)

        accuracies = []
        for key in totalCorrects.keys():
            accuracies.append((key, totalCorrects[key] / len(x)))
        return accuracies

    def validate2(self, x, y, layer = 2, optimizer_type = 'Adam', pool = 'Max', max_iteration = 500, learning_rate = 0.05, momentum = 0.25, conv1_out = 16, conv1_kernel_size = 5, conv2_out = 16, conv2_kernel_size = 3, nn1_out = 120, nn2_out = 84, dropout=0.4):
        def validation_core(i, x, y):
            (foldTrainX, foldTrainY, foldValidationX, foldValidationY) = self.__splitDataFold(x, y, i)

            model = BlinkNeuralNetworkTwoLayer.BlinkNeuralNetwork(pool=pool, conv1_out=conv1_out, conv1_kernel_size=conv1_kernel_size, conv2_out=conv2_out, conv2_kernel_size=conv2_kernel_size, nn1_out=nn1_out, nn2_out=nn2_out, dropout=dropout)
                    # if conv2_kernel_size == 3 else \
                    # BlinkNeuralNetworkTwoLayer2.BlinkNeuralNetwork(pool=pool, conv1_out=conv1_out, conv1_kernel_size=conv1_kernel_size, conv2_out=conv2_out, conv2_kernel_size=conv2_kernel_size, nn1_out=nn1_out, nn2_out=nn2_out)
            lossFunction = torch.nn.MSELoss(reduction='sum')
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) if optimizer_type == 'Adam' else torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

            corrects = []
            model.train()
            for i in range(max_iteration):
                yTrainPredicted = model(foldTrainX)
                loss = lossFunction(yTrainPredicted, foldTrainY)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if i > 100 and (i + 1) % 100 == 0:
                    model.eval()
                    yPredicted = model(foldValidationX)
                    yPred = [ 1 if pred > 0.5 else 0 for pred in yPredicted ]
                    model.train()

                    corrects.append((i + 1, self.__countCorrect(yPred, foldValidationY)))
            return corrects

        totalCorrects = defaultdict(int)
        for i in range(self.k):
            results = validation_core(i, x, y)
            for iteration, correct_count in results:
                totalCorrects[iteration] += correct_count
        # totalCorrects = Parallel(n_jobs=6)(delayed(validation_core)(i, x, y) for i in range(self.k))
        # accuracy = sum(totalCorrects) / len(x)

        accuracies = []
        for key in totalCorrects.keys():
            accuracies.append((key, totalCorrects[key] / len(x)))
        return accuracies

    def validate_softmax(self, x, y, max_iteration = 1500, pool = 'Max', learning_rate = 0.05, conv1_kernel_size = 5, conv2_kernel_size = 3):
        def validation_core(i, x, y):
            (foldTrainX, foldTrainY, foldValidationX, foldValidationY) = self.__splitDataFold(x, y, i)

            model = BlinkNeuralNetworkTwoLayerSoftMax.BlinkNeuralNetwork(pool=pool, conv1_out=6, conv1_kernel_size=conv1_kernel_size, conv2_out=16, conv2_kernel_size=conv2_kernel_size, nn1_out=120, nn2_out=84)
            lossFunction = torch.nn.MSELoss(reduction='sum')
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

            corrects = []
            model.train()
            for i in range(max_iteration):
                yTrainPredicted = model(foldTrainX)
                loss = lossFunction(yTrainPredicted, foldTrainY)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if i > 100 and (i + 1) % 100 == 0:
                    model.eval()
                    yPredicted = model(foldValidationX)
                    yPred = [ 1 if pred > 0.5 else 0 for pred in yPredicted ]
                    model.train()

                    corrects.append((i + 1, self.__countCorrect(yPred, foldValidationY)))
            return corrects

        totalCorrects = defaultdict(int)
        for i in range(self.k):
            results = validation_core(i, x, y)
            for iteration, correct_count in results:
                totalCorrects[iteration] += correct_count
        # totalCorrects = Parallel(n_jobs=6)(delayed(validation_core)(i, x, y) for i in range(self.k))
        # accuracy = sum(totalCorrects) / len(x)

        accuracies = []
        for key in totalCorrects.keys():
            accuracies.append((key, totalCorrects[key] / len(x)))
        return accuracies
