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

    ############################

    print('========== Visualize the Weights ==========')
    model = NeuralNetworks.NeuralNetworks(len(xTrain[0]), [ 2 ], 1)
    print('Training the model')
    for i in range(200):
        model.fit_one(xTrain, yTrain, 10, 0.05)
    hidden_layer_weights = model.weights[0]
    print('Output the weights')
    # for i in range(len(hidden_layer_weights)):
    #     weights = hidden_layer_weights[i]
    #     print(weights)
    Assignment5Support.VisualizeWeights(hidden_layer_weights[0], './weight-0.jpg')
    Assignment5Support.VisualizeWeights(hidden_layer_weights[1], './weight-1.jpg')

    ############################

    print('========== Find Underfitting and Overfiting ==========')
    fig, ax = plt.subplots()
    ax.grid(True)
    x = []
    for i in range(200):
        x.append(i)

    print('Build Neural Network with 1 hidden layer, and 15 nodes for each layer')
    model = NeuralNetworks.NeuralNetworks(len(xTrain[0]), [ 2 ], 1)

    y = []
    y_test = []
    y_accuracy = []
    y_test_accuracy = []
    for i in range(200):
        model.fit_one(xTrain, yTrain, 10, 0.05)

        y.append(model.loss(xTrain, yTrain))
        y_test.append(model.loss(xTest, yTest))

        predicted = model.predict(xTrain)
        accuracy = EvaluationsStub.Accuracy(yTrain, predicted)
        y_accuracy.append(accuracy)
        predicted = model.predict(xTest)
        testAccuracy = EvaluationsStub.Accuracy(yTest, predicted)
        y_test_accuracy.append(testAccuracy)

    plt.plot(x, y, label = 'Training Set Loss')
    plt.plot(x, y_test, label = 'Test Set Loss')

    print('### Plot Loss ###')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Neural Networks\'s Loss on Training and Test Set')
    plt.legend()
    print("Close the plot diagram to continue program")
    plt.show()

    plt.plot(x, y_accuracy, label = 'Training Set Accuracy')
    plt.plot(x, y_test_accuracy, label = 'Test Set Accuracy')

    print('### Plot Accuracy ###')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.title('Neural Networks\'s Accuracy vs Iterations')
    plt.legend()
    print("Close the plot diagram to continue program")
    plt.show()
