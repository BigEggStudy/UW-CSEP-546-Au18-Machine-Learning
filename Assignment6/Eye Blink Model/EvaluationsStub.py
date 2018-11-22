import math

def __CheckEvaluationInput(y, yPredicted):
    # Check sizes
    if(len(y) != len(yPredicted)):
        print(len(y))
        print(len(yPredicted))
        raise UserWarning("Attempting to evaluate between the true labels and predictions.\n   Arrays contained different numbers of samples. Check your work and try again.")

    # Check values
    for value in y:
        if value not in [0, 1]:
            print(value)
            raise UserWarning("Attempting to evaluate between the true labels and predictions.\n   Arrays contained unexpected value. Must be 0 or 1.")
    for value in yPredicted:
        if value not in [0, 1]:
            raise UserWarning("Attempting to evaluate between the true labels and predictions.\n   Arrays contained unexpected value. Must be 0 or 1.")

def Accuracy(y, yPredicted):
    __CheckEvaluationInput(y, yPredicted)

    correct = []
    for i in range(len(y)):
        if(y[i] == yPredicted[i]):
            correct.append(1)
        else:
            correct.append(0)

    return sum(correct)/len(correct)

def Bound(accuracy, length, range=0.95):
    if range == 0.98:
        z = 2.33
    elif range == 0.99:
        z = 2.58
    else:
        z = 1.96

    upper = accuracy + z * math.sqrt( (accuracy * (1 - accuracy) ) / length )
    lower = accuracy - z * math.sqrt( (accuracy * (1 - accuracy) ) / length )
    return (lower, upper)

def _TruePositive(y, yPredicted):
    result = 0
    for i in range(len(y)):
        if y[i] == 1 and yPredicted[i] == 1:
            result += 1

    return result

def _FalsePositive(y, yPredicted):
    result = 0
    for i in range(len(y)):
        if y[i] == 0 and yPredicted[i] == 1:
            result += 1

    return result

def _TrueNegative(y, yPredicted):
    result = 0
    for i in range(len(y)):
        if y[i] == 0 and yPredicted[i] == 0:
            result += 1

    return result

def _FalseNegative(y, yPredicted):
    result = 0
    for i in range(len(y)):
        if y[i] == 1 and yPredicted[i] == 0:
            result += 1

    return result

def Precision(y, yPredicted):
    __CheckEvaluationInput(y, yPredicted)

    truePositive = _TruePositive(y, yPredicted)
    falsePositive = _FalsePositive(y, yPredicted)

    if truePositive + falsePositive == 0:
        return None
    else:
        return truePositive / (truePositive + falsePositive)

def Recall(y, yPredicted):
    __CheckEvaluationInput(y, yPredicted)

    truePositive = _TruePositive(y, yPredicted)
    falseNegative = _FalseNegative(y, yPredicted)

    if truePositive + falseNegative == 0:
        return None
    else:
        return truePositive / (truePositive + falseNegative)

def FalseNegativeRate(y, yPredicted):
    __CheckEvaluationInput(y, yPredicted)

    truePositive = _TruePositive(y, yPredicted)
    falseNegative = _FalseNegative(y, yPredicted)

    if falseNegative + truePositive == 0:
        return None
    else:
        return falseNegative / (falseNegative + truePositive)

def FalsePositiveRate(y, yPredicted):
    __CheckEvaluationInput(y, yPredicted)

    falsePositive = _FalsePositive(y, yPredicted)
    trueNegative = _TrueNegative(y, yPredicted)

    if falsePositive + trueNegative == 0:
        return None
    else:
        return falsePositive / (falsePositive + trueNegative)

def ConfusionMatrix(y, yPredicted):
    __CheckEvaluationInput(y, yPredicted)

    truePositive = _TruePositive(y, yPredicted)
    falsePositive = _FalsePositive(y, yPredicted)
    trueNegative = _TrueNegative(y, yPredicted)
    falseNegative = _FalseNegative(y, yPredicted)

    print('                 |  Predict True  |  Predict False ')
    print('---------------- | -------------- | ---------------')
    print(' Actually True   | ' + '{:>14}'.format(truePositive) + ' | ' + '{:>15}'.format(falseNegative))
    print(' Actually False  | ' + '{:>14}'.format(falsePositive) + ' | ' + '{:>15}'.format(trueNegative))


def ExecuteAll(y, yPredicted):
    print(ConfusionMatrix(y, yPredicted))
    print("Accuracy:", Accuracy(y, yPredicted))
    print("Precision:", Precision(y, yPredicted))
    print("Recall:", Recall(y, yPredicted))
    print("FPR:", FalsePositiveRate(y, yPredicted))
    print("FNR:", FalseNegativeRate(y, yPredicted))

