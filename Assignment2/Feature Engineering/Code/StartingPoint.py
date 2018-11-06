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

############################

print("### One Feature vs 5 Features")

featuresName = ["> 40", "Has Number", "Contains 'Call'", "Contains 'To'", "Contains 'Your'"]

for i in range(5):
    print("Train without '" + featuresName[i] + "' feature")
    newXTrain = np.delete(xTrain, i, axis=1)
    model.fit(newXTrain, yTrain, iterations=50000, step=0.01)

    newXTest = np.delete(xTest, i, axis=1)
    yTestPredicted = model.predict(newXTest)
    testAccuracy = EvaluationsStub.Accuracy(yTest, yTestPredicted)
    print("Test Set Accuracy is %f" % (testAccuracy))

print("Train with all 5 features")
model.fit(xTrain, yTrain, iterations=50000, step=0.01)
yTestPredicted = model.predict(xTest)
testAccuracy = EvaluationsStub.Accuracy(yTest, yTestPredicted)
print("Test Set Accuracy is %f" % (testAccuracy))

############################

import FeatureSelection

print('### Get the Frequency Table')

frequencyTable = FeatureSelection.byFrequency(xTrainRaw)
print('Top 10')
for i in range(10):
    print(frequencyTable[i])

#############################

print('### Get the Mutual Information Table')

mutualInformationTable = FeatureSelection.byMutualInformation(xTrainRaw, yTrain)
print('Top 10')
for i in range(10):
    print(mutualInformationTable[i])

#############################

print('### Run Gradient Descent with the Top 10 Words by Frequency')
words = [word for word,_ in frequencyTable[:10]]
print(words)
(xNewTrain, xNewTest) = FeatureSelection.Featurize(xTrainRaw, xTestRaw, words)

model.fit(xNewTrain, yTrain, iterations=50000, step=0.01)
yTestPredicted = model.predict(xNewTest)
testAccuracy = EvaluationsStub.Accuracy(yTest, yTestPredicted)
print("Test Set Accuracy is %f" % (testAccuracy))

#############################

print('### Run Gradient Descent with the Top 10 Words by Mutual Information')
words = [word for word,_ in mutualInformationTable[:10]]
print(words)
(xNewTrain, xNewTest) = FeatureSelection.Featurize(xTrainRaw, xTestRaw, words)

model.fit(xNewTrain, yTrain, iterations=50000, step=0.01)
yTestPredicted = model.predict(xNewTest)
testAccuracy = EvaluationsStub.Accuracy(yTest, yTestPredicted)
print("Test Set Accuracy is %f" % (testAccuracy))

#############################

print('Mutual Information is better')
print('### Run Gradient Descent with the Top 10 Words by Mutual Information PLUS the Hand Crafted Features')

model.fit(np.hstack([xTrain, xNewTrain]), yTrain, iterations=50000, step=0.01)
yTestPredicted = model.predict(np.hstack([xTest, xNewTest]))
testAccuracy = EvaluationsStub.Accuracy(yTest, yTestPredicted)
print("Test Set Accuracy is %f" % (testAccuracy))

#############################

print('### Run Gradient Descent with the Top 100 Words by Mutual Information PLUS the Hand Crafted Features')
(xNewTrain, xNewTest) = FeatureSelection.Featurize(xTrainRaw, xTestRaw, [word for word,_ in mutualInformationTable[:100]])

model.fit(np.hstack([xTrain, xNewTrain]), yTrain, iterations=50000, step=0.01)
yTestPredicted = model.predict(np.hstack([xTest, xNewTest]))
testAccuracy = EvaluationsStub.Accuracy(yTest, yTestPredicted)
print("Test Set Accuracy is %f" % (testAccuracy))
