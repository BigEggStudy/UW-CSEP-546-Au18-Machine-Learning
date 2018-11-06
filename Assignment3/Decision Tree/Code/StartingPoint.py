import matplotlib.pyplot as plt
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

import DecisionTreeModel
model = DecisionTreeModel.DecisionTree()

############################

# print("### UT for gain function")
# # set of example of the dataset
# playTennis = [9, 5] # Yes, No

# # attribute, number of members (feature)
# outlook = [
#     [4, 0],  # overcase
#     [2, 3],  # sunny
#     [3, 2]   # rain
# ]
# humidity = [
#     [3, 4],  # high
#     [6, 1]   # normal
# ]
# wind = [
#     [6, 2],  # weak
#     [3, 3]   # strong
# ]
# temperature = [
#     [2, 2],  # hot
#     [3, 1],  # cool
#     [4, 2]   # mild
# ]

# print('From Mitchell book (page 60) we can know the information gain of the following attributes are:')
# print('Gain(PlayTennis, Outlook) = 0.246, and the compute result is: %.4f' % model.gain(playTennis, outlook))
# print('Gain(PlayTennis, Humidity) = 0.151, and the compute result is: %.4f' % model.gain(playTennis, humidity))
# print('Gain(PlayTennis, Wind) = 0.048, and the compute result is: %.4f' % model.gain(playTennis, wind))
# print('Gain(PlayTennis, Temperature) = 0.029, and the compute result is: %.4f' % model.gain(playTennis, temperature))

############################

print("### Training with Decision Tree")
model.fit(xTrain, yTrain)
yTestPredicted = model.predict(xTest)
testAccuracy = EvaluationsStub.Accuracy(yTest, yTestPredicted)
(lower, upper) = EvaluationsStub.Bound(testAccuracy, len(yTestPredicted))
print("Test Set Accuracy is %f, with lower bound %f and upper bound %f" % (testAccuracy, lower, upper))

############################

print('Visualize the Decision Tree')
features_name = ["> 40", "Has Number", "Contains 'Call'", "Contains 'To'", "Contains 'Your'"]
model.visualize(features_name)

############################

import CrossValidation
crossValidation = CrossValidation.CrossValidation(10)

print('Get the new Features')
(xNewTrain, xNewTest) = Assignment1Support.Featurize_2(xTrainRaw, xTestRaw)

best_minToSplit = -1
best_accuracy = -1

for minToSplit in range(0, 201, 5):
    newModel = DecisionTreeModel.DecisionTree(minToSplit)
    accuracy = crossValidation.validate(xNewTrain, yTrain, newModel)
    (lower, upper) = EvaluationsStub.Bound(accuracy, len(xNewTrain))
    print("When minToSplit is %d, accuracy from Cross Validation is %f, with lower bound %f and upper bound %f" % (minToSplit, accuracy, lower, upper))
    if best_accuracy < accuracy:
        best_accuracy = accuracy
        best_minToSplit = minToSplit

############################

print("### Compare Models")
print("Choose minToSplit as %d, since it has the best accuracy %f" % (best_minToSplit, best_accuracy))

false_positives1 = []
false_negatives1 = []
false_positives2 = []
false_negatives2 = []

operating_points1 = []
operating_points2 = []

newModel = DecisionTreeModel.DecisionTree(best_minToSplit)
newModel.fit(xNewTrain, yTrain)

for i in range(100):
    threshold = 0.01 + 0.98 * i / 99

    yTestPredicted1 = model.predict(xTest, threshold)
    false_positive1 = EvaluationsStub.FalsePositiveRate(yTest, yTestPredicted1)
    false_negative1 = EvaluationsStub.FalseNegativeRate(yTest, yTestPredicted1)
    false_positives1.append(false_positive1)
    false_negatives1.append(false_negative1)

    yTestPredicted2 = newModel.predict(xNewTest, threshold)
    false_positive2 = EvaluationsStub.FalsePositiveRate(yTest, yTestPredicted2)
    false_negative2 = EvaluationsStub.FalseNegativeRate(yTest, yTestPredicted2)
    false_positives2.append(false_positive2)
    false_negatives2.append(false_negative2)

print("")
print("### Plot Precision vs Recall.")
fig, ax = plt.subplots()
ax.grid(True)
ax.invert_yaxis()
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')

plt.plot(false_positives1, false_negatives1, label = 'Model With Length Feature with Threshold 40 & minToSplit = 100')
plt.plot(false_positives2, false_negatives2, label = ('Model With Continuous Length Feature & minToSplit = %d' % best_minToSplit))
plt.xlabel('False Positive Rate')
plt.ylabel('False Negative Rate')
plt.title('Test Set ROC Curve')
plt.legend()

print("Close the plot diagram to continue program")
plt.show()

############################

features_name = ["Length", "Has Number", "Contains 'Call'", "Contains 'To'", "Contains 'Your'"]
newModel.visualize(features_name)
