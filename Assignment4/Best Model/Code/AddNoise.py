import collections
import random

def FindMostCommonWords(x, n):
    counter = collections.Counter()

    for s in x:
        for word in str.split(s):
            counter[word] += 1

    return [ count[0] for count in counter.most_common(n) ]

def MakeProblemHarder(xRaw,yRaw,scaleUpFactor=1,featureNoisePercent=0.015,labelNoisePercent=0.08, seed=1000):
    random.seed(seed)

    xHarder = []
    yHarder = []

    # expand the size of the data set
    for i in range(scaleUpFactor):
        xHarder += xRaw
        yHarder += yRaw

    # add noise to the words
    vocabulary = FindMostCommonWords(xRaw, 5000)

    for i in range(len(xHarder)):
        noise = False

        xUpdated = ""
        for word in str.split(xHarder[i]):
            if random.uniform(0,1) < featureNoisePercent:
                word = vocabulary[random.randint(0, len(vocabulary))]
                noise = True

            if len(xUpdated) == 0:
                xUpdated = word
            else:
                xUpdated += " %s" % word
        xUpdated += "\n"

        #if noise:
        #    print(xHarder[i])
        #    print(xUpdated)
        #    print("---")

        xHarder[i] = xUpdated

    # add noise to the labels
    for i in range(len(yHarder)):
        if random.uniform(0,1) < labelNoisePercent:
            yHarder[i] = 0 if yHarder[i] == 1 else 1

    return (xHarder, yHarder)