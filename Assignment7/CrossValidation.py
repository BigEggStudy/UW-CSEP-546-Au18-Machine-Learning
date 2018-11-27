import math
import numpy as np
from joblib import Parallel, delayed

class CrossValidation(object):
    def __init__(self, k):
        self.k = round(k)

    def __splitDataFold(self, x, y, index):
        size = round(len(x) / self.k)

        if (index * size == 0):
            newTrainX = x[(index + 1) * size:]
            newTrainY = y[(index + 1) * size:]
        elif ((index + 1) * size >= len(x)):
            newTrainX = x[:index * size]
            newTrainY = y[:index * size]
        else:
            newTrainX = np.concatenate((x[:index * size], x[(index + 1) * size:]), axis=0)
            newTrainY = np.concatenate((y[:index * size], y[(index + 1) * size:]), axis=0)

        newValidationX = x[index * size:(index + 1) * size]
        newValidationY = y[index * size:(index + 1) * size]

        return (newTrainX, newTrainY, newValidationX, newValidationY)

    def __countCorrect(self, yPredicted, y):
        count = 0
        for i in range(len(y)):
            if y[i] == yPredicted[i]:
                count += 1

        return count

    def validate(self, x, y, iteration = 200, mini_batch_size = 10, eta = 0.05, beta = 0):
        def validation_core(i, x, y):
            (foldTrainX, foldTrainY, foldValidationX, foldValidationY) = self.__splitDataFold(x, y, i)

            # model = NeuralNetworks.NeuralNetworks(len(foldTrainX[0]), [ 20 ], 1)
            # for i in range(iteration):
            #     model.fit_one(foldTrainX, foldTrainY, mini_batch_size, eta, beta)
            # return self.__countCorrect(model.predict(foldValidationX), foldValidationY)
            return 0

        totalCorrects = Parallel(n_jobs=6)(delayed(validation_core)(i, x, y) for i in range(self.k))

        accuracy = sum(totalCorrects) / len(x)
        return accuracy
