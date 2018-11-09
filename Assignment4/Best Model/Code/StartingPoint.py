import matplotlib.pyplot as plt
import numpy as np
import operator

import Assignment1Support
import EvaluationsStub
import AddNoise

if __name__=="__main__":
    ### UPDATE this path for your environment
    kDataPath = "..\\Data\\SMSSpamCollection"

    print('### Load the data and add the noise')
    (xRaw, yRaw) = Assignment1Support.LoadRawData(kDataPath)

    (xTrainRawOriginal, yTrainRawOriginal, xTestRawOriginal, yTestRawOriginal) = Assignment1Support.TrainTestSplit(xRaw, yRaw)
    (xTrainRaw, yTrainRaw) = AddNoise.MakeProblemHarder(xTrainRawOriginal, yTrainRawOriginal)
    (xTestRaw, yTestRaw) = AddNoise.MakeProblemHarder(xTestRawOriginal, yTestRawOriginal)
    yTrain = yTrainRaw
    yTest = yTestRaw

    print("Train is %f percent spam." % (sum(yTrainRaw)/len(yTrainRaw)))
    print("Test is %f percent spam." % (sum(yTestRaw)/len(yTestRaw)))

    ### Get the Mutual Information Words as features
    import FeatureSelection

    print('### Get the Mutual Information features')
    mutualInformationTable = FeatureSelection.byMutualInformation(xTrainRaw, yTrain)
    words = [word for word,_ in mutualInformationTable[:100]]
    (xTrainMI_Base, xTestMI_Base) = FeatureSelection.Featurize(xTrainRaw, xTestRaw, words)
    yTrain = yTrainRaw
    yTest = yTestRaw

    import CrossValidation
    crossValidation = CrossValidation.CrossValidation(5)
    import RandomForest
    ############################

    print("========== Use Random Forest as the default Model ==========")

    print("Default Model will use parameters: num_trees as %d, min_to_split as %d, use_bagging as %s, and restrict_features as %d" % (1, 2, False, 0))
    print("Default Model will use the 100 MI features")

    base_model = RandomForest.RandomForest(num_trees = 1, min_to_split = 2, use_bagging = False, restrict_features = 0)
    print("### Training with Random Forest")
    base_model.fit(xTrainMI_Base, yTrain)

    print("### Predicting with Random Forest")
    yTestPredicted = base_model.predict(xTestMI_Base)
    testAccuracy = EvaluationsStub.Accuracy(yTest, yTestPredicted)
    (lower, upper) = EvaluationsStub.Bound(testAccuracy, len(yTestPredicted))
    print("Test Set Accuracy is %f, with lower bound %f and upper bound %f" % (testAccuracy, lower, upper))

    ###########################

    print("========== Underfitting & Overfitting ==========")
    x_data = []
    y_data_training_accuracy = []
    y_data_test_accuracy = []

    for min_to_split in range(2, 403, 10):
        model = RandomForest.RandomForest(num_trees = 1, min_to_split = min_to_split, use_bagging = False, restrict_features = 0)
        print("### Training with Random Forest with min_to_split as %d" % min_to_split)
        model.fit(xTrainMI_Base, yTrain)

        print("### Predicting with Random Forest")
        yTrainingPredicted = model.predict(xTrainMI_Base)
        trainingAccuracy = EvaluationsStub.Accuracy(yTrain, yTrainingPredicted)
        (lower, upper) = EvaluationsStub.Bound(trainingAccuracy, len(yTrainingPredicted))
        print("Training Set Accuracy is %f, with lower bound %f and upper bound %f" % (trainingAccuracy, lower, upper))

        yTestPredicted = model.predict(xTestMI_Base)
        testAccuracy = EvaluationsStub.Accuracy(yTest, yTestPredicted)
        (lower, upper) = EvaluationsStub.Bound(testAccuracy, len(yTestPredicted))
        print("Test Set Accuracy is %f, with lower bound %f and upper bound %f" % (testAccuracy, lower, upper))

        x_data.append(min_to_split)
        y_data_training_accuracy.append(trainingAccuracy)
        y_data_test_accuracy.append(testAccuracy)

    print("### Plot the Accuracy to find the Underfitting & Overfitting")
    fig, ax = plt.subplots()
    ax.grid(True)

    plt.plot(x_data, y_data_training_accuracy, label = 'Accuracy on Training Set')
    plt.plot(x_data, y_data_test_accuracy, label = 'Accuracy on Test Set')
    plt.xlabel('Min to Split')
    plt.ylabel('Accuracy')
    plt.ylim(0.5, 1)
    plt.title('Accuracy vs Min to Split')
    plt.legend()

    print("Close the plot diagram to continue program")
    plt.show()
    print("Find that:")
    print("Find that when min_to_split < 50, the model is Overfitting")
    print("Find that when min_to_split > 150, the model is Underfitting")
    print("Choose min_to_split = 125")

    ############################

    print("========== Evaluation Manual Craft Features ==========")
    model = RandomForest.RandomForest(num_trees = 1, min_to_split = 125, use_bagging = False, restrict_features = 0)
    for i in range(3):
        if i == 1:
            print("##### Update the Manual Craft Features - Remove 'Word Count' feature ######")
        elif i == 2:
            print("###### Update the Manual Craft Features - Remove 'Has Uncommon Punctuation' feature ######")

        (xTrainHand, xTestHand, featuresName) = FeatureSelection.hand_craft_features(xTrainRaw, xTestRaw, i)
        for i in range(len(featuresName)):
            print("### Cross Validation without '" + featuresName[i] + "' feature")
            newXTrain = np.delete(xTrainHand, i, axis=1)
            accuracy = crossValidation.validate(newXTrain, yTrain, model)
            (lower, upper) = EvaluationsStub.Bound(accuracy, len(xTrainRaw))
            print("Accuracy from Cross Validation is %f, with lower bound %f and upper bound %f" % (accuracy, lower, upper))

        print("### Cross Validation with all %d manual hand craft features" % len(featuresName))
        accuracy = crossValidation.validate(newXTrain, yTrain, model)
        (lower, upper) = EvaluationsStub.Bound(accuracy, len(xTrainRaw))
        print("Accuracy from Cross Validation is %f, with lower bound %f and upper bound %f" % (accuracy, lower, upper))

        print('###### Categorize the Mistakes ######')
        model.fit(xTrainHand, yTrain)
        yTrainPredicted = model.predict(xTrainHand)
        testAccuracy = EvaluationsStub.Accuracy(yTrain, yTrainPredicted)
        print("Training Set Accuracy is %f" % (testAccuracy))

        yTrainPredictedRaw = model.predictRaw(xTrainHand)
        result = {}
        print('###### Get the Worst False Positives ######')
        for i in range(len(yTrain)):
            if yTrain[i] == 0 and yTrainPredictedRaw[i] > 0.5:
                result[xTrainRaw[i]] = yTrainPredictedRaw[i]

        sortedResult = sorted(result.items(), key=operator.itemgetter(1), reverse=True)
        length = min(20, len(sortedResult))
        for i in range(length):
            print(sortedResult[i])

        result = {}
        print('###### Get the Worst False Negative ######')
        for i in range(len(yTrain)):
            if yTrain[i] == 1 and yTrainPredictedRaw[i] <= 0.5:
                result[xTrainRaw[i]] = yTrainPredictedRaw[i]

        sortedResult = sorted(result.items(), key=operator.itemgetter(1))
        length = min(20, len(sortedResult))
        for i in range(length):
            print(sortedResult[i])

    ############################

    print("========== Preprocess the Data ==========")
    (xTrainRawNormalize, xTestRawNormalize) = FeatureSelection.preprocess(xTrainRaw, xTestRaw)

    ############################

    print("========== Evaluation Mutual Information Features ==========")
    model = RandomForest.RandomForest(num_trees = 1, min_to_split = 125, use_bagging = False, restrict_features = 0)

    for i in range(1, 21):
        print("### Cross Validation with top '%d' MI Word" % (i * 10))
        accuracy = crossValidation.validateByMutualInformation(xTrainRawNormalize, yTrain, model, i * 10)
        (lower, upper) = EvaluationsStub.Bound(accuracy, len(xTrainRawNormalize))
        print("Accuracy from Cross Validation is %f, with lower bound %f and upper bound %f" % (accuracy, lower, upper))

    ############################

    print('========== Merge Features ==========')
    print('Use 5 Hand Craft Words as Features')
    (xTrainHand, xTestHand, featuresName) = FeatureSelection.hand_craft_features(xTrainRaw, xTestRaw, 2)

    print('Use 70 Mutual Information Words as Features')
    mutualInformationTable = FeatureSelection.byMutualInformation(xTrainRawNormalize, yTrain)
    words = [word for word,_ in mutualInformationTable[:70]]
    (xTrainMI, xTestMI) = FeatureSelection.Featurize(xTrainRawNormalize, xTestRawNormalize, words)

    xTrain = np.hstack([xTrainHand, xTrainMI])
    xTest = np.hstack([xTestHand, xTestMI])

    ############################

    print('========== Parameter Sweeping for Random Forests ==========')
    best_accuracy = 0
    best_parameter = (1, 125, False, 0)
    for num_trees in range(50, 151, 25):
        for min_to_split in range(25, 152, 25):
            use_bagging = True
            for restrict_features in [0, 30, 70]:
                print('Build Random Forests with num_trees = %d, min_to_split = %d, use_bagging = %s, restrict_features = %d' % (num_trees, min_to_split, use_bagging, restrict_features))
                model = RandomForest.RandomForest(num_trees = num_trees, min_to_split = min_to_split, use_bagging = use_bagging, restrict_features = restrict_features)
                accuracy = crossValidation.validate(xTrain, yTrain, model)
                (lower, upper) = EvaluationsStub.Bound(accuracy, len(xTrain))
                print("Accuracy from Cross Validation is %f, with lower bound %f and upper bound %f" % (accuracy, lower, upper))
                if best_accuracy < accuracy:
                    best_accuracy = accuracy
                    best_parameter = (num_trees, min_to_split, use_bagging, restrict_features)

    print('When parameter with num_trees = %d, min_to_split = %d, use_bagging = %s, restrict_features = %d' % best_parameter)
    print('Random Forests has best accuracy %f' % best_accuracy)
    num_trees = best_parameter[0]
    min_to_split = best_parameter[1]
    use_bagging = best_parameter[2]
    restrict_features = best_parameter[3]

    # num_trees = 10
    # min_to_split = 12
    # use_bagging = True
    # restrict_features = 70

    ############################

    print('========== Evaluation on the Best Random Forests ==========')
    print('Build Random Forests with num_trees = %d, min_to_split = %d, use_bagging = %s, restrict_features = %d' % (num_trees, min_to_split, use_bagging, restrict_features))
    model = RandomForest.RandomForest(num_trees = num_trees, min_to_split = min_to_split, use_bagging = use_bagging, restrict_features = restrict_features)
    accuracy = crossValidation.validate(xTrain, yTrain, model)
    (lower, upper) = EvaluationsStub.Bound(accuracy, len(xTrain))
    print("Accuracy from Cross Validation is %f, with lower bound %f and upper bound %f" % (accuracy, lower, upper))
    model.fit(xTrain, yTrain)
    yPredicted = model.predict(xTest)
    testAccuracy = EvaluationsStub.Accuracy(yTest, yPredicted)
    (lower, upper) = EvaluationsStub.Bound(testAccuracy, len(yPredicted))
    print("Test Set Accuracy is %f, with lower bound %f and upper bound %f" % (testAccuracy, lower, upper))

    EvaluationsStub.ExecuteAll(yTest, yPredicted)

    ############################

    print('========== Compare Models ==========')
    false_positives1 = []
    false_negatives1 = []
    false_positives2 = []
    false_negatives2 = []

    for i in range(100):
        threshold = 0.01 + 0.99 * i / 99
        print('At threshold %f' % threshold)

        yTestPredicted1 = base_model.predict(xTestMI_Base, threshold)
        false_positive1 = EvaluationsStub.FalsePositiveRate(yTest, yTestPredicted1)
        false_negative1 = EvaluationsStub.FalseNegativeRate(yTest, yTestPredicted1)
        false_positives1.append(false_positive1)
        false_negatives1.append(false_negative1)
        print('First Model has False Positive Rate %f, False Negative Rate %f' % (false_positive1, false_negative1))

        yTestPredicted2 = model.predict(xTest, threshold)
        false_positive2 = EvaluationsStub.FalsePositiveRate(yTest, yTestPredicted2)
        false_negative2 = EvaluationsStub.FalseNegativeRate(yTest, yTestPredicted2)
        false_positives2.append(false_positive2)
        false_negatives2.append(false_negative2)
        print('Best Model has False Positive Rate %f, False Negative Rate %f' % (false_positive2, false_negative2))

    print("")
    print("### Plot Precision vs Recall.")
    fig, ax = plt.subplots()
    ax.grid(True)
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')

    plt.plot(false_positives1, false_negatives1, label = 'First Model')
    plt.plot(false_positives2, false_negatives2, label = 'Best Model')
    plt.xlabel('False Positive Rate')
    plt.ylabel('False Negative Rate')
    plt.title('Test Set ROC Curve')
    plt.legend()

    print("Close the plot diagram to continue program")
    plt.show()
