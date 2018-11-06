import math
import numpy as np

import FeatureSelection

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
            newTrainX = np.concatenate((x[:index * size], x[(index + 1) * size:]))
            newTrainY = y[:index * size] + y[(index + 1) * size:]

        newValidationX = x[index * size:(index + 1) * size]
        newValidationY = y[index * size:(index + 1) * size]

        return (newTrainX, newTrainY, newValidationX, newValidationY)

    def __countCorrect(self, yPredicted, y):
        count = 0
        for i in range(len(y)):
            if y[i] == yPredicted[i]:
                count += 1

        return count

    def validate(self, x, y, model):
        totalCorrect = 0

        for i in range(self.k):
            (foldTrainX, foldTrainY, foldValidationX, foldValidationY) = self.__splitDataFold(x, y, i)

            model.fit(foldTrainX, foldTrainY)
            totalCorrect += self.__countCorrect(model.predict(foldValidationX), foldValidationY)

        accuracy = totalCorrect / len(x)

        return accuracy

    def validateByFrequency(self, x, y, model):
        totalCorrect = 0

        for i in range(self.k):
            (foldTrainX, foldTrainY, foldValidationX, foldValidationY) = self.__splitDataFold(x, y, i)

            frequencyTable = FeatureSelection.byFrequency(foldTrainX)
            words = [word for word,_ in frequencyTable[:10]]
            print('For fold %d/%d, choose words:' % (i + 1, self.k))
            print(words)
            (xNewTrain, xNewValidation) = FeatureSelection.Featurize(foldTrainX, foldValidationX, words)

            model.fit(xNewTrain, foldTrainY)
            totalCorrect += self.__countCorrect(model.predict(xNewValidation), foldValidationY)

        accuracy = totalCorrect / len(x)

        return accuracy

    def validateByMutualInformation(self, x, y, model):
        totalCorrect = 0

        for i in range(self.k):
            (foldTrainX, foldTrainY, foldValidationX, foldValidationY) = self.__splitDataFold(x, y, i)

            mutualInformationTable = FeatureSelection.byMutualInformation(foldTrainX, foldTrainY)
            words = [word for word,_ in mutualInformationTable[:10]]
            print('For fold %d/%d, choose words:' % (i + 1, self.k))
            print(words)
            (xNewTrain, xNewValidation) = FeatureSelection.Featurize(foldTrainX, foldValidationX, words)

            model.fit(xNewTrain, foldTrainY)
            totalCorrect += self.__countCorrect(model.predict(xNewValidation), foldValidationY)

        accuracy = totalCorrect / len(x)

        return accuracy
