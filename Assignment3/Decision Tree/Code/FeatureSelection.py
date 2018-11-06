import operator
import math

def byFrequency(xRaw):
    frequencyTable = {}

    for sentence in xRaw:
        for word in sentence.split():
            if word in frequencyTable:
                frequencyTable[word] += 1
            else:
                frequencyTable[word] = 1

    return sorted(frequencyTable.items(), key=operator.itemgetter(1), reverse=True)

def byMutualInformation(xRaw, y):
    totalSentenses = len(xRaw)
    py0 = __calcP(totalSentenses - sum(y), totalSentenses)
    py1 = __calcP(sum(y), totalSentenses)

    allWords = []
    for sentence in xRaw:
        for word in sentence.split():
            if word not in allWords:
                allWords.append(word)

    count_x0y0 = {}
    count_x1y0 = {}
    count_x0y1 = {}
    count_x1y1 = {}
    for word in allWords:
        count_x0y0[word] = 0
        count_x1y0[word] = 0
        count_x0y1[word] = 0
        count_x1y1[word] = 0

    for index in range(totalSentenses):
        for word in allWords:
            sentence = xRaw[index]
            y_i = y[index]
            if word in sentence.split():
                if y_i == 0:
                    count_x1y0[word] += 1
                if y_i == 1:
                    count_x1y1[word] += 1
            else:
                if y_i == 0:
                    count_x0y0[word] += 1
                if y_i == 1:
                    count_x0y1[word] += 1

    miTable = {}
    for word in allWords:
        wordCount = count_x1y0[word] + count_x1y1[word]
        px0 = __calcP(totalSentenses - wordCount, totalSentenses)
        px1 = __calcP(wordCount, totalSentenses)

        p_x0y0 = __calcP(count_x0y0[word], totalSentenses)
        mi_x0y0 = p_x0y0 * math.log2(p_x0y0/(px0 * py0))

        p_x1y0 = __calcP(count_x1y0[word], totalSentenses)
        mi_x1y0 = p_x1y0 * math.log2(p_x1y0/(px1 * py0))

        p_x0y1 = __calcP(count_x0y1[word], totalSentenses)
        mi_x0y1 = p_x0y1 * math.log2(p_x0y1/(px0 * py1))

        p_x1y1 = __calcP(count_x1y1[word], totalSentenses)
        mi_x1y1 = p_x1y1 * math.log2(p_x1y1/(px1 * py1))

        miTable[word] = mi_x0y0 + mi_x1y0 + mi_x0y1 + mi_x1y1

    return sorted(miTable.items(), key=operator.itemgetter(1), reverse=True)

def __calcP(obs, n):
    return (obs + 1) / (n + 2)


def Featurize(xTrainRaw, xTestRaw, words):
    # featurize the training data, may want to do multiple passes to count things.
    xTrain = []
    for x in xTrainRaw:
        features = []

        # Have features for a few words
        for word in words:
            if word in x.split():
                features.append(1)
            else:
                features.append(0)

        xTrain.append(features)

    # now featurize test using any features discovered on the training set. Don't use the test set to influence which features to use.
    xTest = []
    for x in xTestRaw:
        features = []

        # Have features for a few words
        for word in words:
            if word in x.split():
                features.append(1)
            else:
                features.append(0)

        xTest.append(features)

    return (xTrain, xTest)