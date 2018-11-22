## Some of this references my answers to previous assignments.
##  Replace my references with references to your answers to those assignments.

## IMPORTANT NOTE !!
## Remember to install the Pillow library (which is required to execute 'import PIL')

import Assignment5Support
import Featurize
import EvaluationsStub
import matplotlib.pyplot as plt

if __name__=="__main__":
    kDataPath = "..\\..\\dataset_B_Eye_Images"

    (xRaw, yRaw) = Assignment5Support.LoadRawData(kDataPath, includeLeftEye = True, includeRightEye = True)
    (xTrainRaw, yTrainRaw, xTestRaw, yTestRaw) = Assignment5Support.TrainTestSplit(xRaw, yRaw, percentTest = .25)
    print("Train is %f percent closed." % (sum(yTrainRaw)/len(yTrainRaw)))
    print("Test is %f percent closed." % (sum(yTestRaw)/len(yTestRaw)))

    print('Featurize the Samples')
    xTrain = Featurize.Featurize(xTrainRaw)
    xTest = Featurize.Featurize(xTestRaw)
    yTrain = yTrainRaw
    yTest = yTestRaw

    ############################

    import KNearestNeighbors
    model = KNearestNeighbors.KNearestNeighbors()
    model.fit(xTrain, yTrain)

    print('========== Compare Models ==========')
    fig, ax = plt.subplots()
    ax.grid(True)
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')

    for k in [1,3,5,10,20,50,100]:
        print('### For K = %d ###' % k)

        false_positives = []
        false_negatives = []

        for i in range(100):
            threshold = 0.01 + 0.99 * i / 99
            print('At threshold %f' % threshold)

            yTestPredicted = model.predict(xTest, k, threshold=threshold)
            false_positive = EvaluationsStub.FalsePositiveRate(yTest, yTestPredicted)
            false_negative = EvaluationsStub.FalseNegativeRate(yTest, yTestPredicted)
            false_positives.append(false_positive)
            false_negatives.append(false_negative)
            print('K Nearest Neighbors Model has False Positive Rate %f, False Negative Rate %f' % (false_positive, false_negative))

        plt.plot(false_positives, false_negatives, label = ('Model with K = %d' % k))

    plt.xlabel('False Positive Rate')
    plt.ylabel('False Negative Rate')
    plt.title('Test Set ROC Curve')
    plt.legend()

    print("Close the plot diagram to continue program")
    plt.show()
