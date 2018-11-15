## Some of this references my answers to previous assignments.
##  Replace my references with references to your answers to those assignments.

## IMPORTANT NOTE !!
## Remember to install the Pillow library (which is required to execute 'import PIL')

import Assignment5Support
import EvaluationsStub
import matplotlib.pyplot as plt
import numpy as np

if __name__=="__main__":
    # xTrainRaw = [
    #     '../../dataset_B_Eye_Images/openRightEyes/Oscar_DLeon_0001_R.jpg',
    #     '../../dataset_B_Eye_Images/openRightEyes/Ahmed_Ghazi_0001_R.jpg',
    #     '../../dataset_B_Eye_Images/openLeftEyes/Ainsworth_Dyer_0001_L.jpg',
    #     '../../dataset_B_Eye_Images/openLeftEyes/Alex_Holmes_0001_L.jpg',
    #     '../../dataset_B_Eye_Images/closedRightEyes/closed_eye_0043.jpg_face_2_R.jpg',
    #     '../../dataset_B_Eye_Images/closedRightEyes/closed_eye_0071.jpg_face_1_R.jpg',
    #     '../../dataset_B_Eye_Images/closedLeftEyes/closed_eye_0014.jpg_face_2_L.jpg',
    #     '../../dataset_B_Eye_Images/closedLeftEyes/closed_eye_0068.jpg_face_1_L.jpg',
    # ]

    # print('Featurize the Samples')
    # (xTrain, xTest) = Assignment5Support.Featurize(xTrainRaw, [], includeGradients=False, includeRawPixels=False, includeIntensities=True)
    # yTrain = np.array([1, 1, 1, 1, 0, 0, 0, 0])[np.newaxis].T

    kDataPath = "..\\..\\dataset_B_Eye_Images"

    (xRaw, yRaw) = Assignment5Support.LoadRawData(kDataPath, includeLeftEye = True, includeRightEye = True)
    (xTrainRaw, yTrainRaw, xTestRaw, yTestRaw) = Assignment5Support.TrainTestSplit(xRaw, yRaw, percentTest = .25)
    print("Train is %f percent closed." % (sum(yTrainRaw)/len(yTrainRaw)))
    print("Test is %f percent closed." % (sum(yTestRaw)/len(yTestRaw)))

    print('Featurize the Samples')
    (xTrain, xTest) = Assignment5Support.Featurize(xTrainRaw, xTestRaw, includeGradients=False, includeRawPixels=False, includeIntensities=True)
    yTrain = np.array(yTrainRaw)[np.newaxis].T
    yTest = np.array(yTestRaw)[np.newaxis].T

    ############################

    import NeuralNetworks
    print('========== Compare Models ==========')
    x = []
    for i in range(200):
        x.append(i)

    hidden_layer = 1
    hidden_nodes = 2
    print('Build Neural Network with %d hidden layer, and %s nodes for each layer' % (hidden_layer, hidden_nodes))
    model = NeuralNetworks.NeuralNetworks(len(xTrain[0]), [ 2 ], 1)

    y = []
    y_test = []
    last_loss = -1
    for i in range(200):
        model.fit_one(xTrain, yTrain, 1, 0.05)
        loss = model.loss(xTrain, yTrain)
        if last_loss > 0 and last_loss < loss:
            print('Exist with error')
            exit()
        last_loss = loss
        y.append(loss)
        y_test.append(model.loss(xTest, yTest))

    yPredicted = model.predict(xTest)
    testAccuracy = EvaluationsStub.Accuracy(yTest, yPredicted)
    (lower, upper) = EvaluationsStub.Bound(testAccuracy, len(yPredicted))
    print("Test Set Accuracy is %f, with lower bound %f and upper bound %f" % (testAccuracy, lower, upper))

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
    plt.plot(x, y_test, label = ('Model with %d layer and %d nodes' % (hidden_layer, hidden_nodes)))
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Neural Networks\'s Loss on Test Set')
    plt.legend()
    print("Close the plot diagram to continue program")
    plt.show()
