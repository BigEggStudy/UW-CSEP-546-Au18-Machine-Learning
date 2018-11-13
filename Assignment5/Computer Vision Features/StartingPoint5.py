## Some of this references my answers to previous assignments.
##  Replace my references with references to your answers to those assignments.

## IMPORTANT NOTE !!
## Remember to install the Pillow library (which is required to execute 'import PIL')

import matplotlib.pyplot as plt
import Assignment5Support
import Featurize
import RandomForest
import EvaluationsStub

if __name__=="__main__":
    kDataPath = "..\\dataset_B_Eye_Images"

    (xRaw, yRaw) = Assignment5Support.LoadRawData(kDataPath, includeLeftEye = True, includeRightEye = True)
    (xTrainRaw, yTrainRaw, xTestRaw, yTestRaw) = Assignment5Support.TrainTestSplit(xRaw, yRaw, percentTest = .25)
    print("Train is %f percent closed." % (sum(yTrainRaw)/len(yTrainRaw)))
    print("Test is %f percent closed." % (sum(yTestRaw)/len(yTestRaw)))

    yTrain = yTrainRaw
    yTest = yTestRaw

    ############################

    print("========== Use Random Forest for Y-Gradient ==========")
    print("### Calculating features ###")
    xTrain_Y_Gradient = Featurize.Featurize_Y_Gradient(xTrainRaw)
    xTest_Y_Gradient = Featurize.Featurize_Y_Gradient(xTestRaw)

    print('### Parameter Sweeping for Random Forests ###')
    best_accuracy = 0
    best_parameter = (1, 125, False, 0)
    use_bagging = True
    for num_trees in range(25, 151, 25):
        for min_to_split in range(25, 152, 25):
            for restrict_features in [0, 10, 20]:
                print('Build Random Forests with num_trees = %d, min_to_split = %d, use_bagging = %s, restrict_features = %d' % (num_trees, min_to_split, use_bagging, restrict_features))
                model = RandomForest.RandomForest(num_trees = num_trees, min_to_split = min_to_split, use_bagging = use_bagging, restrict_features = restrict_features)
                model.fit(xTrain_Y_Gradient, yTrain)
                yPredicted = model.predict(xTest_Y_Gradient)
                testAccuracy = EvaluationsStub.Accuracy(yTest, yPredicted)
                (lower, upper) = EvaluationsStub.Bound(testAccuracy, len(yPredicted))
                print("Test Set Accuracy is %f, with lower bound %f and upper bound %f" % (testAccuracy, lower, upper))
                if best_accuracy < testAccuracy:
                    best_accuracy = testAccuracy
                    best_parameter = (num_trees, min_to_split, use_bagging, restrict_features)

    print('When parameter with num_trees = %d, min_to_split = %d, use_bagging = %s, restrict_features = %d' % best_parameter)
    print('Random Forests has best accuracy %f' % best_accuracy)

    ############################

    print("========== Use Random Forest for X-Gradient ==========")
    print("### Calculating features ###")
    xTrain_X_Gradient = Featurize.Featurize_X_Gradient(xTrainRaw)
    xTest_X_Gradient = Featurize.Featurize_X_Gradient(xTestRaw)

    print('### Parameter Sweeping for Random Forests ###')
    best_accuracy = 0
    best_parameter = (1, 125, False, 0)
    use_bagging = True
    for num_trees in range(25, 151, 25):
        for min_to_split in range(25, 152, 25):
            for restrict_features in [0, 10, 20]:
                print('Build Random Forests with num_trees = %d, min_to_split = %d, use_bagging = %s, restrict_features = %d' % (num_trees, min_to_split, use_bagging, restrict_features))
                model = RandomForest.RandomForest(num_trees = num_trees, min_to_split = min_to_split, use_bagging = use_bagging, restrict_features = restrict_features)
                model.fit(xTrain_X_Gradient, yTrain)
                yPredicted = model.predict(xTest_X_Gradient)
                testAccuracy = EvaluationsStub.Accuracy(yTest, yPredicted)
                (lower, upper) = EvaluationsStub.Bound(testAccuracy, len(yPredicted))
                print("Test Set Accuracy is %f, with lower bound %f and upper bound %f" % (testAccuracy, lower, upper))
                if best_accuracy < testAccuracy:
                    best_accuracy = testAccuracy
                    best_parameter = (num_trees, min_to_split, use_bagging, restrict_features)

    print('When parameter with num_trees = %d, min_to_split = %d, use_bagging = %s, restrict_features = %d' % best_parameter)
    print('Random Forests has best accuracy %f' % best_accuracy)

    ############################

    print("========== Use Random Forest for Y-Gradient Histogram ==========")
    print("### Calculating features ###")
    xTrain_Y_Histogram = Featurize.Featurize_Y_Histogram(xTrainRaw)
    xTest_Y_Histogram = Featurize.Featurize_Y_Histogram(xTestRaw)

    print('### Parameter Sweeping for Random Forests ###')
    best_accuracy = 0
    best_parameter = (1, 125, False, 0)
    use_bagging = True
    for num_trees in range(25, 151, 25):
        for min_to_split in range(25, 152, 25):
            for restrict_features in [0, 2, 4]:
                print('Build Random Forests with num_trees = %d, min_to_split = %d, use_bagging = %s, restrict_features = %d' % (num_trees, min_to_split, use_bagging, restrict_features))
                model = RandomForest.RandomForest(num_trees = num_trees, min_to_split = min_to_split, use_bagging = use_bagging, restrict_features = restrict_features)
                model.fit(xTrain_Y_Histogram, yTrain)
                yPredicted = model.predict(xTest_Y_Histogram)
                testAccuracy = EvaluationsStub.Accuracy(yTest, yPredicted)
                (lower, upper) = EvaluationsStub.Bound(testAccuracy, len(yPredicted))
                print("Test Set Accuracy is %f, with lower bound %f and upper bound %f" % (testAccuracy, lower, upper))
                if best_accuracy < testAccuracy:
                    best_accuracy = testAccuracy
                    best_parameter = (num_trees, min_to_split, use_bagging, restrict_features)

    print('When parameter with num_trees = %d, min_to_split = %d, use_bagging = %s, restrict_features = %d' % best_parameter)
    print('Random Forests has best accuracy %f' % best_accuracy)

    ############################

    print("========== Use Random Forest for X-Gradient Histogram ==========")
    print("### Calculating features ###")
    xTrain_X_Histogram = Featurize.Featurize_X_Histogram(xTrainRaw)
    xTest_X_Histogram = Featurize.Featurize_X_Histogram(xTestRaw)

    print('### Parameter Sweeping for Random Forests ###')
    best_accuracy = 0
    best_parameter = (1, 125, False, 0)
    use_bagging = True
    for num_trees in range(25, 151, 25):
        for min_to_split in range(25, 152, 25):
            for restrict_features in [0, 2, 4]:
                print('Build Random Forests with num_trees = %d, min_to_split = %d, use_bagging = %s, restrict_features = %d' % (num_trees, min_to_split, use_bagging, restrict_features))
                model = RandomForest.RandomForest(num_trees = num_trees, min_to_split = min_to_split, use_bagging = use_bagging, restrict_features = restrict_features)
                model.fit(xTrain_X_Histogram, yTrain)
                yPredicted = model.predict(xTest_X_Histogram)
                testAccuracy = EvaluationsStub.Accuracy(yTest, yPredicted)
                (lower, upper) = EvaluationsStub.Bound(testAccuracy, len(yPredicted))
                print("Test Set Accuracy is %f, with lower bound %f and upper bound %f" % (testAccuracy, lower, upper))
                if best_accuracy < testAccuracy:
                    best_accuracy = testAccuracy
                    best_parameter = (num_trees, min_to_split, use_bagging, restrict_features)

    print('When parameter with num_trees = %d, min_to_split = %d, use_bagging = %s, restrict_features = %d' % best_parameter)
    print('Random Forests has best accuracy %f' % best_accuracy)

    ############################

    print('========== Compare Models ==========')
    model_Y_Gradient = RandomForest.RandomForest(num_trees = 150, min_to_split = 25, use_bagging = True, restrict_features = 20)
    model_Y_Gradient.fit(xTrain_Y_Gradient, yTrain)
    false_positives_Y_Gradient = []
    false_negatives_Y_Gradient = []
    model_X_Gradient = RandomForest.RandomForest(num_trees = 75, min_to_split = 25, use_bagging = True, restrict_features = 10)
    model_X_Gradient.fit(xTrain_X_Gradient, yTrain)
    false_positives_X_Gradient = []
    false_negatives_X_Gradient = []
    model_Y_Histogram = RandomForest.RandomForest(num_trees = 25, min_to_split = 125, use_bagging = True, restrict_features = 0)
    model_Y_Histogram.fit(xTrain_Y_Histogram, yTrain)
    false_positives_Y_Histogram = []
    false_negatives_Y_Histogram = []
    model_X_Histogram = RandomForest.RandomForest(num_trees = 100, min_to_split = 25, use_bagging = True, restrict_features = 0)
    model_X_Histogram.fit(xTrain_X_Histogram, yTrain)
    false_positives_X_Histogram = []
    false_negatives_X_Histogram = []

    for i in range(100):
        threshold = 0.01 + 0.99 * i / 99
        print('At threshold %f' % threshold)

        yTestPredicted = model_Y_Gradient.predict(xTest_Y_Gradient, threshold)
        false_positive = EvaluationsStub.FalsePositiveRate(yTest, yTestPredicted)
        false_negative = EvaluationsStub.FalseNegativeRate(yTest, yTestPredicted)
        false_positives_Y_Gradient.append(false_positive)
        false_negatives_Y_Gradient.append(false_negative)
        print('Y-Gradient Model has False Positive Rate %f, False Negative Rate %f' % (false_positive, false_negative))

        yTestPredicted = model_X_Gradient.predict(xTest_X_Gradient, threshold)
        false_positive = EvaluationsStub.FalsePositiveRate(yTest, yTestPredicted)
        false_negative = EvaluationsStub.FalseNegativeRate(yTest, yTestPredicted)
        false_positives_X_Gradient.append(false_positive)
        false_negatives_X_Gradient.append(false_negative)
        print('X-Gradient Model has False Positive Rate %f, False Negative Rate %f' % (false_positive, false_negative))

        yTestPredicted = model_Y_Histogram.predict(xTest_Y_Histogram, threshold)
        false_positive = EvaluationsStub.FalsePositiveRate(yTest, yTestPredicted)
        false_negative = EvaluationsStub.FalseNegativeRate(yTest, yTestPredicted)
        false_positives_Y_Histogram.append(false_positive)
        false_negatives_Y_Histogram.append(false_negative)
        print('Y-Gradient Histogram Model has False Positive Rate %f, False Negative Rate %f' % (false_positive, false_negative))

        yTestPredicted = model_X_Histogram.predict(xTest_X_Histogram, threshold)
        false_positive = EvaluationsStub.FalsePositiveRate(yTest, yTestPredicted)
        false_negative = EvaluationsStub.FalseNegativeRate(yTest, yTestPredicted)
        false_positives_X_Histogram.append(false_positive)
        false_negatives_X_Histogram.append(false_negative)
        print('X-Gradient Histogram Model has False Positive Rate %f, False Negative Rate %f' % (false_positive, false_negative))

    print("")
    print("### Plot Precision vs Recall.")
    fig, ax = plt.subplots()
    ax.grid(True)
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')

    plt.plot(false_positives_Y_Gradient, false_negatives_Y_Gradient, label = 'Model use Y-Gradient')
    plt.plot(false_positives_X_Gradient, false_negatives_X_Gradient, label = 'Model use X-Gradient')
    plt.plot(false_positives_Y_Histogram, false_negatives_Y_Histogram, label = 'Model use Y-Gradient Histogram')
    plt.plot(false_positives_X_Histogram, false_negatives_X_Histogram, label = 'Model use X-Gradient Histogram')
    plt.xlabel('False Positive Rate')
    plt.ylabel('False Negative Rate')
    plt.title('Test Set ROC Curve')
    plt.legend()

    print("Close the plot diagram to continue program")
    plt.show()
