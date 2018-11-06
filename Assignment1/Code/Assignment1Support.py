import collections

def LoadRawData(path):
    f = open(path, 'r')
    
    lines = f.readlines()

    kNumberExamplesExpected = 5574

    if(len(lines) != kNumberExamplesExpected):
        message = "Attempting to load %s:\n" % (path)
        message += "   Expected %d lines, got %d.\n" % (kNumberExamplesExpected, len(lines))
        message += "    Check the path to training data and try again."
        raise UserWarning(message)

    x = []
    y = []

    for l in lines:
        if(l.startswith('ham')):
            y.append(0)
            x.append(l[4:])
        elif(l.startswith('spam')):
            y.append(1)
            x.append(l[5:])
        else:
            message = "Attempting to process %s\n" % (l)
            message += "   Did not match expected format."
            message += "    Check the path to training data and try again."
            raise UserWarning(message)

    return (x, y)

def TrainTestSplit(x, y, percentTest = .25):
    if(len(x) != len(y)):
        raise UserWarning("Attempting to split into training and testing set.\n\tArrays do not have the same size. Check your work and try again.")

    numTest = round(len(x) * percentTest)

    if(numTest == 0 or numTest > len(y)):
        raise UserWarning("Attempting to split into training and testing set.\n\tSome problem with the percentTest or data set size. Check your work and try again.")

    xTest = x[:numTest]
    xTrain = x[numTest:]
    yTest = y[:numTest]
    yTrain = y[numTest:]

    return (xTrain, yTrain, xTest, yTest)

def Featurize(xTrainRaw, xTestRaw):
    words = ['call', 'to', 'your']

    # featurize the training data, may want to do multiple passes to count things.
    xTrain = []
    for x in xTrainRaw:
        features = []

        # Have a feature for longer texts
        if(len(x)>40):
            features.append(1)
        else:
            features.append(0)

        # Have a feature for texts with numbers in them
        if(any(i.isdigit() for i in x)):
            features.append(1)
        else:
            features.append(0)

        # Have features for a few words
        for word in words:
            if word in x:
                features.append(1)
            else:
                features.append(0)

        xTrain.append(features)

    # now featurize test using any features discovered on the training set. Don't use the test set to influence which features to use.
    xTest = []
    for x in xTestRaw:
        features = []
        
        # Have a feature for longer texts
        if(len(x)>40):
            features.append(1)
        else:
            features.append(0)

        # Have a feature for texts with numbers in them
        if(any(i.isdigit() for i in x)):
            features.append(1)
        else:
            features.append(0)

        # Have features for a few words
        for word in words:
            if word in x:
                features.append(1)
            else:
                features.append(0)

        xTest.append(features)

    return (xTrain, xTest)

def InspectFeatures(xRaw, x):
    for i in range(len(xRaw)):
        print(x[i], xRaw[i])

