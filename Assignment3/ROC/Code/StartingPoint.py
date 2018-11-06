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

import LogisticRegressionModel
model1 = LogisticRegressionModel.LogisticRegressionModel()
model2 = LogisticRegressionModel.LogisticRegressionModel()

import FeatureSelection

print('### Get the Mutual Information Table')
mutualInformationTable = FeatureSelection.byMutualInformation(xTrainRaw, yTrain)
words = [word for word,_ in mutualInformationTable[:10]]
print(words)
(xNewTrain, xNewTest) = FeatureSelection.Featurize(xTrainRaw, xTestRaw, words)

############################

print("### Training the Models")
model1.fit(xNewTrain, yTrain, iterations=50000, step=0.01)
model2.fit(np.hstack([xTrain, xNewTrain]), yTrain, iterations=50000, step=0.01)

############################

print("### Compare Models")

false_positives1 = []
false_negatives1 = []
false_positives2 = []
false_negatives2 = []

for i in range(100):
    threshold = 0.01 + 0.99 * i / 99
    print('At threshold %f' % threshold)

    yTestPredicted1 = model1.predictWithThreshold(xNewTest, threshold)
    false_positive1 = EvaluationsStub.FalsePositiveRate(yTest, yTestPredicted1)
    false_negative1 = EvaluationsStub.FalseNegativeRate(yTest, yTestPredicted1)
    false_positives1.append(false_positive1)
    false_negatives1.append(false_negative1)
    print('Model 10 MI Features without heuristics has False Positive Rate %f, False Negative Rate %f' % (false_positive1, false_negative1))

    yTestPredicted2 = model2.predictWithThreshold(np.hstack([xTest, xNewTest]), threshold)
    false_positive2 = EvaluationsStub.FalsePositiveRate(yTest, yTestPredicted2)
    false_negative2 = EvaluationsStub.FalseNegativeRate(yTest, yTestPredicted2)
    false_positives2.append(false_positive2)
    false_negatives2.append(false_negative2)
    print('Model 10 MI Features with heuristics has False Positive Rate %f, False Negative Rate %f' % (false_positive2, false_negative2))

print("")
print("### Plot Precision vs Recall.")
fig, ax = plt.subplots()
ax.grid(True)
ax.invert_yaxis()
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')

plt.plot(false_positives1, false_negatives1, label = 'Model Without Heuristics')
plt.plot(false_positives2, false_negatives2, label = 'Model With Heuristics')
plt.xlabel('False Positive Rate')
plt.ylabel('False Negative Rate')
plt.title('Test Set ROC Curve')
plt.legend()

print("Close the plot diagram to continue program")
plt.show()

############################

print('### Get the Threshold that Can Let the False Positive Rate achieve 10%')

print('For Model 10 MI Features without heuristics:')
operating_points_greater = []
operating_points_less = []
for i in range(100):
    threshold = 0.10 + 0.04 * i / 99

    yTestPredicted1 = model1.predictWithThreshold(xNewTest, threshold)
    false_positive1 = EvaluationsStub.FalsePositiveRate(yTest, yTestPredicted1)
    false_negative1 = EvaluationsStub.FalseNegativeRate(yTest, yTestPredicted1)
    if false_positive1 >= 0.10:
        operating_points_greater.append((threshold, (false_positive1, false_negative1)))
        if len(operating_points_greater) > 10:
            del operating_points_greater[0]
    if false_positive1 <= 0.10:
        operating_points_less.append((threshold, (false_positive1, false_negative1)))
    if len(operating_points_less) == 10:
        break

for data in operating_points_greater:
    print('When Threshold is %f, the False Positive Rate is %f, the False Negative Rate is %f' % (data[0], data[1][0], data[1][1]))
for data in operating_points_less:
    print('When Threshold is %f, the False Positive Rate is %f, the False Negative Rate is %f' % (data[0], data[1][0], data[1][1]))

print('For Model 10 MI Features with heuristics:')
operating_points_greater = []
operating_points_less = []
for i in range(100):
    threshold = 0.15 + 0.04 * i / 99

    yTestPredicted2 = model2.predictWithThreshold(np.hstack([xTest, xNewTest]), threshold)
    false_positive2 = EvaluationsStub.FalsePositiveRate(yTest, yTestPredicted2)
    false_negative2 = EvaluationsStub.FalseNegativeRate(yTest, yTestPredicted2)
    if false_positive2 >= 0.10:
        operating_points_greater.append((threshold, (false_positive2, false_negative2)))
        if len(operating_points_greater) > 10:
            del operating_points_greater[0]
    if false_positive2 <= 0.10:
        operating_points_less.append((threshold, (false_positive2, false_negative2)))
    if len(operating_points_less) == 10:
        break

for data in operating_points_greater:
    print('When Threshold is %f, the False Positive Rate is %f, the False Negative Rate is %f' % (data[0], data[1][0], data[1][1]))
for data in operating_points_less:
    print('When Threshold is %f, the False Positive Rate is %f, the False Negative Rate is %f' % (data[0], data[1][0], data[1][1]))