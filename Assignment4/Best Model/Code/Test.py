import matplotlib.pyplot as plt
import numpy as np
import operator

import Assignment1Support
import EvaluationsStub
import FeatureSelection
import RandomForest

if __name__=="__main__":
    ### UPDATE this path for your environment
    kDataPath = "..\\Data\\SMSSpamCollection"

    print('### Load the data and add the noise')
    (xRaw, yRaw) = Assignment1Support.LoadRawData(kDataPath)

    (xTrainRaw, yTrainRaw, xTestRaw, yTestRaw) = Assignment1Support.TrainTestSplit(xRaw, yRaw)
    yTrain = yTrainRaw
    yTest = yTestRaw

    print('========== Debug on raw data =========')

    num_trees = 10
    min_to_split = 12
    use_bagging = True
    restrict_features = 70
    print("========== Preprocess the Data ==========")
    (xTrainRawNormalize, xTestRawNormalize) = FeatureSelection.preprocess(xTrainRaw, xTestRaw)
    print('========== Merge Features ==========')
    print('Use 5 Hand Craft Words as Features')
    (xTrainHand, xTestHand, featuresName) = FeatureSelection.hand_craft_features(xTrainRaw, xTestRaw, 2)

    print('Use 70 Mutual Information Words as Features')
    model = RandomForest.RandomForest(num_trees = num_trees, min_to_split = min_to_split, use_bagging = use_bagging, restrict_features = restrict_features)
    mutualInformationTable = FeatureSelection.byMutualInformation(xTrainRawNormalize, yTrain)
    words = [word for word,_ in mutualInformationTable[:70]]
    (xTrainMI, xTestMI) = FeatureSelection.Featurize(xTrainRawNormalize, xTestRawNormalize, words)

    xTrain = np.hstack([xTrainHand, xTrainMI])
    xTest = np.hstack([xTestHand, xTestMI])

    model.fit(xTrain, yTrain)
    yPredicted = model.predict(xTest)
    testAccuracy = EvaluationsStub.Accuracy(yTest, yPredicted)
    (lower, upper) = EvaluationsStub.Bound(testAccuracy, len(yPredicted))
    print("Test Set Accuracy is %f, with lower bound %f and upper bound %f" % (testAccuracy, lower, upper))

    print('========== Debug on raw data =========')