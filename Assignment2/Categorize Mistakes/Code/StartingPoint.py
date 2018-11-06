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

print('### Get the Mutual Information Table')
mutualInformationTable = FeatureSelection.byMutualInformation(xRaw, yRaw)

#############################

import EvaluationsStub

print('### Get the Confusion Matrix')
words = [word for word,_ in mutualInformationTable[:10]]
print(words)
(xNewTrain, xNewTest) = FeatureSelection.Featurize(xTrainRaw, xTestRaw, words)

model.fit(xNewTrain, yTrain, iterations=50000, step=0.01)
yTestPredicted = model.predict(xNewTest)

EvaluationsStub.ExecuteAll(yTest, yTestPredicted)

#############################

import operator

print('### Run Gradient Descent with the Top 10 Words by Mutual Information')
yTestPredicted = model.predictRaw(xNewTest)

result = {}
print('###### Get the Worst False Positives ######')
for i in range(len(yTest)):
    if yTest[i] == 0 and yTestPredicted[i] > 0.5:
        result[xTestRaw[i]] = yTestPredicted[i]

sortedResult = sorted(result.items(), key=operator.itemgetter(1), reverse=True)
length = min(20, len(sortedResult))
lengthLargerThan60Less150 = 0

for i in range(length):
    print(sortedResult[i])
    sentence = sortedResult[i][0]
    if len(sentence) > 60 and len(sentence) < 150:
        lengthLargerThan60Less150 += 1

if length < 20:
    print('Only %d sentence, are in False Positives' % length)

print('From the %d sentence, there have %d sentence length is less than 150 but greater than 60' % (length, lengthLargerThan60Less150))

if length < 20:
    for i in range(len(yTest)):
        if yTest[i] == 0 and yTestPredicted[i] < 1:
            result[xTestRaw[i]] = yTestPredicted[i]

    sortedResult = sorted(result.items(), key=operator.itemgetter(1), reverse=True)
    length = 20 - length
    print('Get more (%d) sentences to analyze' % length)
    lengthLessThan60 = 0
    lengthLargerThan150 = 0
    useUppercase = 0
    useUncommonPunctuation = 0
    for i in range(20 - length, 20):
        print(sortedResult[i])
        sentence = sortedResult[i][0]
        if len(sentence) <= 60:
            lengthLessThan60 += 1
        if len(sentence) >= 150:
            lengthLargerThan150 += 1
        if (any(word.upper() == word for word in sentence.split())):
            useUppercase += 1
        if (any((ch == '!' or ch == '/' or ch == ':' or ch == '@' or ch == ';') for ch in sentence)):
            useUncommonPunctuation += 1

    print('From the %d sentence, there have %d sentence length is less than 60' % (length, lengthLessThan60))
    print('From the %d sentence, there have %d sentence length is larger than 150' % (length, lengthLargerThan150))
    print('From the %d sentence, there have %d sentence use upper case' % (length, useUppercase))
    print('From the %d sentence, there have %d sentence use uncommon punctuation (! / : @ ;)' % (length, useUppercase))

result = {}
print('###### Get the Worst False Negative ######')
for i in range(len(yTest)):
    if yTest[i] == 1 and yTestPredicted[i] <= 0.5:
        result[xTestRaw[i]] = yTestPredicted[i]

sortedResult = sorted(result.items(), key=operator.itemgetter(1))
length = min(20, len(sortedResult))
urlInside = 0
digitalWord = 0
useUppercase = 0
useUncommonPunctuation = 0
for i in range(length):
    print(sortedResult[i])
    sentence = sortedResult[i][0]
    if 'www' in sentence or 'http' in sentence:
        urlInside += 1
    if (any(word.isdigit() for word in sentence.split())):
        digitalWord += 1
    if (any(word.upper() == word for word in sentence.split())):
        useUppercase += 1
    if (any((ch == '!' or ch == '/' or ch == ':' or ch == '@' or ch == ';') for ch in sentence)):
        useUncommonPunctuation += 1

print('From the %d sentence, there have %d sentence has URL' % (length, urlInside))
print('From the %d sentence, there have %d sentence has digit word' % (length, digitalWord))
print('From the %d sentence, there have %d sentence use upper case' % (length, useUppercase))
print('From the %d sentence, there have %d sentence use uncommon punctuation (! / : @ ;)' % (length, useUppercase))