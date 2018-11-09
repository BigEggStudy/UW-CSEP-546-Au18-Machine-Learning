import matplotlib.pyplot as plt
import numpy as np

import Assignment1Support
import EvaluationsStub
import AddNoise

if __name__=="__main__":
    ### UPDATE this path for your environment
    kDataPath = "..\\Data\\SMSSpamCollection"

    print('### Load the data and add the noise')
    (xRaw, yRaw) = Assignment1Support.LoadRawData(kDataPath)

    # (xTrainRaw, yTrainRaw, xTestRaw, yTestRaw) = Assignment1Support.TrainTestSplit(xRaw, yRaw)
    (xTrainRawOriginal, yTrainRawOriginal, xTestRawOriginal, yTestRawOriginal) = Assignment1Support.TrainTestSplit(xRaw, yRaw)
    (xTrainRaw, yTrainRaw) = AddNoise.MakeProblemHarder(xTrainRawOriginal, yTrainRawOriginal)
    (xTestRaw, yTestRaw) = AddNoise.MakeProblemHarder(xTestRawOriginal, yTestRawOriginal)

    (xTrain, xTest) = Assignment1Support.Featurize(xTrainRaw, xTestRaw)
    yTrain = yTrainRaw
    yTest = yTestRaw

    ### Get the Mutual Information Words as features
    import FeatureSelection

    print('### Get the Mutual Information features')
    mutualInformationTable = FeatureSelection.byMutualInformation(xTrainRaw, yTrain)
    words = [word for word,_ in mutualInformationTable[:295]]
    (xNewTrain, xNewTest) = FeatureSelection.Featurize(xTrainRaw, xTestRaw, words)

    print('### Merge the features')
    xTrain = np.hstack([xTrain, xNewTrain])
    xTest = np.hstack([xTest, xNewTest])

    import RandomForest
    ############################

    print("========== Building one Model and output the accuracy ==========")

    model = RandomForest.RandomForest(num_trees = 10, min_to_split = 2, use_bagging = True, restrict_features = 20)
    print("### Training with Random Forest")
    model.fit(xTrain, yTrain)

    print("### Predicting with Random Forest")
    (yTestPredicted, yTestPredicted_Trees) = model.predict(xTest)
    testAccuracy = EvaluationsStub.Accuracy(yTest, yTestPredicted)
    (lower, upper) = EvaluationsStub.Bound(testAccuracy, len(yTestPredicted))
    print("Test Set Accuracy is %f, with lower bound %f and upper bound %f" % (testAccuracy, lower, upper))

    print("For Individual Trees:")
    for tree_predict in yTestPredicted_Trees:
        testAccuracy = EvaluationsStub.Accuracy(yTest, tree_predict)
        (lower, upper) = EvaluationsStub.Bound(testAccuracy, len(tree_predict))
        print("Test Set Accuracy is %f, with lower bound %f and upper bound %f" % (testAccuracy, lower, upper))

    ############################

    x_data = []
    y_data_model1 = []
    y_data_model2 = []
    y_data_model3 = []
    y_data_model4 = []

    print("========== Build multiple Models and plot the accuracy ==========")
    for num_trees in [1, 20, 40, 60, 80]:
        x_data.append(num_trees)
        y_data = []
        for (min_to_split, use_bagging, restrict_features) in [(2, True, 20), (50, True, 20), (2, False, 20), (2, True, 0)]:
            model = RandomForest.RandomForest(num_trees = num_trees, min_to_split = min_to_split, use_bagging = use_bagging, restrict_features = restrict_features)
            print("### Training with Random Forest with tree num as %d, min to split as %d, use bagging as %s, restrict features as %d" % (num_trees, min_to_split, use_bagging, restrict_features))
            model.fit(xTrain, yTrain)
            print("### Predicting with Random Forest")
            (yTestPredicted, yTestPredicted_Trees) = model.predict(xTest)
            testAccuracy = EvaluationsStub.Accuracy(yTest, yTestPredicted)
            (lower, upper) = EvaluationsStub.Bound(testAccuracy, len(yTestPredicted))
            print("Test Set Accuracy is %f, with lower bound %f and upper bound %f" % (testAccuracy, lower, upper))
            y_data.append(testAccuracy)
        y_data_model1.append(y_data[0])
        y_data_model2.append(y_data[1])
        y_data_model3.append(y_data[2])
        y_data_model4.append(y_data[3])


    plt.plot(x_data, y_data_model1, label = 'Split until 2 Samples, Bagging, Restrict Features to 20')
    plt.plot(x_data, y_data_model2, label = 'Split until 50 Samples, Bagging, Restrict Features to 20')
    plt.plot(x_data, y_data_model3, label = 'Split until 2 Samples, no Bagging, Restrict Features to 20')
    plt.plot(x_data, y_data_model4, label = 'Split until 2 Samples, Bagging, no Restrict Features')
    plt.xlabel('Number of Trees')
    plt.ylabel('Accuracy')
    plt.title('Test Set Accuracy vs Different Model')
    plt.legend()
    print("Close the plot diagram to continue program")
    plt.show()
