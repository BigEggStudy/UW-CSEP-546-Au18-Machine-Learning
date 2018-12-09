from collections import deque
import random
import math

import numpy as np

class QLearning(object):
    def __init__(self, stateSpaceShape, numActions, discountRate):
        self.stateSpaceShape = tuple(stateSpaceShape)
        self.numActions = numActions
        self.discountRate = discountRate

        self.q_table = np.zeros(self.stateSpaceShape + (numActions,))
        self.visit_n = np.zeros(self.stateSpaceShape + (numActions,))

        self.memory = deque(maxlen=2000)

    def GetAction(self, currentState, learningMode, randomActionRate = 0.01, actionProbabilityBase = 1.5):
        if learningMode:
            if np.random.rand() < randomActionRate:
                return random.randrange(self.numActions)

            powered = np.power(actionProbabilityBase, self.q_table[tuple(currentState)])
            actions_probability = powered / np.sum(powered)
            return np.random.choice(self.numActions, p=actions_probability)
        else:
            return np.argmax(self.q_table[tuple(currentState)])

    def ObserveAction(self, oldState, action, newState, reward, learningRateScale):
        learning_rate = 1 / (1 + learningRateScale * self.visit_n[tuple(oldState)][action])
        self.q_table[tuple(oldState)][action] += learning_rate * (reward + self.discountRate * np.max(self.q_table[tuple(newState)][:]) - self.q_table[tuple(oldState)][action])
        self.visit_n[tuple(oldState)][action] += 1


    def record(self, oldState, action, newState, reward):
        self.memory.append((oldState, action, newState, reward))

    def replay(self, learningRateScale):
        for oldState, action, newState, reward in self.memory:
            self.ObserveAction(oldState, action, newState, reward, learningRateScale)

    def clear_record(self):
        self.memory.clear()
