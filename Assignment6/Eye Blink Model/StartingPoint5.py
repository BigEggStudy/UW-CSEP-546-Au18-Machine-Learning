## Some of this references my answers to previous assignments.
##  Replace my references with references to your answers to those assignments.

## IMPORTANT NOTE !!
## Remember to install the Pillow library (which is required to execute 'import PIL')

import Assignment5Support
import EvaluationsStub
import matplotlib.pyplot as plt
import numpy as np

import Featurize

if __name__=="__main__":
    kDataPath = "..\\..\\dataset_B_Eye_Images"

    (xRaw, yRaw) = Assignment5Support.LoadRawData(kDataPath, includeLeftEye = True, includeRightEye = True)
    (xTrainRaw, yTrainRaw, xTestRaw, yTestRaw) = Assignment5Support.TrainTestSplit(xRaw, yRaw, percentTest = .25)
    print("Train is %f percent closed." % (sum(yTrainRaw)/len(yTrainRaw)))
    print("Test is %f percent closed." % (sum(yTestRaw)/len(yTestRaw)))

    yTrain = np.array(yTrainRaw)[np.newaxis].T
    yTest = np.array(yTestRaw)[np.newaxis].T

    import NeuralNetworks
    import CrossValidation
    crossValidation = CrossValidation.CrossValidation(5)
    ############################

    print('========== Compare Features ==========')
    best_accuracy = 0
    best_parameter = (0, 0.0)
    for step in [1, 2, 4]:
        print('Use Intensities with step %d' % step)
        (xTrain, xTest) = Featurize.ByIntensities(xTrainRaw, xTestRaw, step)
        for momentum_beta in [0.0, 0.25]:
            print("Training Model with momentum beta as %f" % momentum_beta)
            accuracy = crossValidation.validate(xTrain, yTrain, beta=momentum_beta)
            (lower, upper) = EvaluationsStub.Bound(accuracy, len(xTrainRaw))
            print("Accuracy from Cross Validation is %f, with lower bound %f and upper bound %f" % (accuracy, lower, upper))
            if best_accuracy < accuracy:
                best_accuracy = accuracy
                best_parameter = (step, momentum_beta)

    print('When Intensities with step %d and momentum beta as %f,' % best_parameter)
    print('Neural Networks has best accuracy %f' % best_accuracy)
