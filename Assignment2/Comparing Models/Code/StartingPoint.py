import numpy as np

import Assignment1Support
import EvaluationsStub

### UPDATE this path for your environment
kDataPath = "..\\Data\\SMSSpamCollection"

(xRaw, yRaw) = Assignment1Support.LoadRawData(kDataPath)

(xTrainRaw, yTrainRaw, xTestRaw, yTestRaw) = Assignment1Support.TrainTestSplit(xRaw, yRaw)

print("Train is %f percent spam." % (sum(yTrainRaw)/len(yTrainRaw)))
print("Test is %f percent spam." % (sum(yTestRaw)/len(yTestRaw)))

(xTrain, xTest) = Assignment1Support.Featurize(xTrainRaw, xTestRaw)
yTrain = yTrainRaw
yTest = yTestRaw

import LogisticRegressionModel
model = LogisticRegressionModel.LogisticRegressionModel()

#############################

import FeatureSelection

print('### Get the Frequency Table')
frequencyTable = FeatureSelection.byFrequency(xTrainRaw)

#############################

print('### Get the Mutual Information Table')
mutualInformationTable = FeatureSelection.byMutualInformation(xTrainRaw, yTrain)

#############################

print('### Run Gradient Descent with the Top 10 Words by Frequency')
words = [word for word,_ in frequencyTable[:10]]
print(words)
(xNewTrain, xNewTest) = FeatureSelection.Featurize(xTrainRaw, xTestRaw, words)

model.fit(xNewTrain, yTrain, iterations=50000, step=0.01)
yTestPredicted = model.predict(xNewTest)
testAccuracy = EvaluationsStub.Accuracy(yTest, yTestPredicted)
(lower, upper) = EvaluationsStub.Bound(testAccuracy, len(yTestPredicted))
print("Test Set Accuracy is %f, with lower bound %f and upper bound %f" % (testAccuracy, lower, upper))

#############################

print('### Run Gradient Descent with the Top 10 Words by Mutual Information')
words = [word for word,_ in mutualInformationTable[:10]]
print(words)
(xNewTrain, xNewTest) = FeatureSelection.Featurize(xTrainRaw, xTestRaw, words)

model.fit(xNewTrain, yTrain, iterations=50000, step=0.01)
yTestPredicted = model.predict(xNewTest)
testAccuracy = EvaluationsStub.Accuracy(yTest, yTestPredicted)
(lower, upper) = EvaluationsStub.Bound(testAccuracy, len(yTestPredicted))
print("Test Set Accuracy is %f, with lower bound %f and upper bound %f" % (testAccuracy, lower, upper))

#############################

import CrossValidation
crossValidation = CrossValidation.CrossValidation(5)

#############################

print("### Cross Validation with the Top 10 Words by Frequency")
accuracy = crossValidation.validateByFrequency(xTrainRaw, yTrainRaw, model)
(lower, upper) = EvaluationsStub.Bound(accuracy, len(xTrainRaw))
print("Accuracy from Cross Validation is %f, with lower bound %f and upper bound %f" % (accuracy, lower, upper))

#############################

print("### Cross Validation with the Top 10 Words by Mutual Information")
accuracy = crossValidation.validateByMutualInformation(xTrainRaw, yTrainRaw, model)
(lower, upper) = EvaluationsStub.Bound(accuracy, len(xTrainRaw))
print("Accuracy from Cross Validation is %f, with lower bound %f and upper bound %f" % (accuracy, lower, upper))
