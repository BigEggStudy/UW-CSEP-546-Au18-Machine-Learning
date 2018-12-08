import math

# This code converts the continuous space states returned from Gym into discrete states so we can implement Q-Learning with tables.
#   As part of your solution you may want to change the number of bins used.


# Mountain cart's state is described here: https://github.com/openai/gym/wiki/MountainCar-v0
mountainCarBinsPerDimension = 20
mountainCarLow = [ -1.2 , -0.07 ]
mountainCarHigh = [ 0.6 , 0.07 ]

def MountainCarStateSpaceShape():
    return [ mountainCarBinsPerDimension, mountainCarBinsPerDimension ]

def MountainCarObservationToStateSpace(observation):
    return [ _DimensionBinIndex(mountainCarLow[i], mountainCarHigh[i], mountainCarBinsPerDimension, observation[i]) for i in range(len(mountainCarLow)) ]

def _DimensionBinIndex(binMin, binMax, binsPerDimension, value):
    if value < binMin:
        return 0

    if value > binMax:
        return binsPerDimension - 1

    range = binMax - binMin
    binWidth = range / float(binsPerDimension)
    binIndex = math.floor((value - binMin) / binWidth)

    return min(binIndex, binsPerDimension - 1) # this min is so the max value ends up in the last bin, not beyond it

# CartPole's state is described here: 
cartPoleBinsPerDimension = 20
cartPoleLow = [ -4.8000002e+00, -4, -4.1887903e-01, -4 ]
cartPoleHigh = [ 4.8000002e+00, 4, 4.1887903e-01, 4 ]


def CartPoleStateSpaceShape():
    return [ cartPoleBinsPerDimension, cartPoleBinsPerDimension, cartPoleBinsPerDimension, cartPoleBinsPerDimension ]

def CartPoleObservationToStateSpace(observation):
    return [ _DimensionBinIndex(cartPoleLow[i], cartPoleHigh[i], cartPoleBinsPerDimension, observation[i]) for i in range(len(cartPoleLow)) ]