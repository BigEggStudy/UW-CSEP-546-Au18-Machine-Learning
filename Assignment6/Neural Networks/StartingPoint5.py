## Some of this references my answers to previous assignments.
##  Replace my references with references to your answers to those assignments.

## IMPORTANT NOTE !!
## Remember to install the Pillow library (which is required to execute 'import PIL')

import Assignment5Support
import EvaluationsStub
import matplotlib.pyplot as plt
import numpy as np

if __name__=="__main__":
    kDataPath = "..\\..\\dataset_B_Eye_Images"

    (xRaw, yRaw) = Assignment5Support.LoadRawData(kDataPath, includeLeftEye = True, includeRightEye = True)
    (xTrainRaw, yTrainRaw, xTestRaw, yTestRaw) = Assignment5Support.TrainTestSplit(xRaw, yRaw, percentTest = .25)
    print("Train is %f percent closed." % (sum(yTrainRaw)/len(yTrainRaw)))
    print("Test is %f percent closed." % (sum(yTestRaw)/len(yTestRaw)))

    print('Featurize the Samples')
    (xTrain, xTest) = Assignment5Support.Featurize(xTrainRaw, xTestRaw, includeGradients=False, includeRawPixels=False, includeIntensities=True)
    yTrain = np.array(yTrainRaw)[np.newaxis].T
    yTest = np.array(yTestRaw)[np.newaxis].T

    import NeuralNetworks
    ############################

    print('========== Compare Models ==========')
    fig, ax = plt.subplots()
    ax.grid(True)

    x = []
    y_test_loss = []
    for i in range(200):
        x.append(i + 1)

    for hidden_layer in [1, 2]:
        for hidden_nodes in [2, 5, 10, 15, 20]:
            print('Build Neural Network with %d hidden layer, and %s nodes for each layer' % (hidden_layer, hidden_nodes))
            hidden = [ hidden_nodes for i in range(hidden_layer) ]
            model = NeuralNetworks.NeuralNetworks(len(xTrain[0]), hidden, 1)

            y = []
            y_test = []
            for i in range(200):
                model.fit_one(xTrain, yTrain, 10, 0.05)
                y.append(model.loss(xTrain, yTrain))
                y_test.append(model.loss(xTest, yTest))

            yPredicted = model.predict(xTest)
            testAccuracy = EvaluationsStub.Accuracy(yTest, yPredicted)
            (lower, upper) = EvaluationsStub.Bound(testAccuracy, len(yPredicted))
            print("Test Set Accuracy is %f, with lower bound %f and upper bound %f" % (testAccuracy, lower, upper))

            y_test_loss.append(y_test)
            plt.plot(x, y, label = ('Model with %d layer and %d nodes' % (hidden_layer, hidden_nodes)))

    print('### Plot Training Set Loss ###')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Neural Networks\'s Loss on Training Set')
    plt.legend()
    print("Close the plot diagram to continue program")
    plt.show()

    print('### Plot Test Set Loss ###')
    fig, ax = plt.subplots()
    ax.grid(True)
    for y_loss in y_test_loss:
        plt.plot(x, y_loss, label = ('Model with %d layer and %d nodes' % (hidden_layer, hidden_nodes)))
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Neural Networks\'s Loss on Test Set')
    plt.legend()
    print("Close the plot diagram to continue program")
    plt.show()
