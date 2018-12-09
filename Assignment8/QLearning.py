from collections import deque
import random
import math

class QLearning(object):
    def __init__(self, stateSpaceShape, numActions, discountRate):
        self.stateSpaceShape = tuple(stateSpaceShape)
        self.num_actions = numActions
        self.discount_rate = discountRate

        max_index = 1
        for shape in stateSpaceShape:
            max_index = max_index * shape
        max_index = max_index * self.num_actions

        self.q_table = [0] * max_index
        self.visit_n = [0] * max_index

        self.memory = deque(maxlen=2000)

    def GetAction(self, currentState, learningMode, randomActionRate = 0.01, actionProbabilityBase = 1.5):
        if learningMode and random.random() < randomActionRate:
            return random.randrange(self.num_actions)

        current_state_index = 1
        for shape in currentState:
            current_state_index = current_state_index * shape
        action_scores = self.q_table[current_state_index: current_state_index + self.num_actions]
        assert len(action_scores) == self.num_actions

        if learningMode:
            powered = [ actionProbabilityBase ** action_score for action_score in action_scores ]
            actions_probability = [ power_value / sum(powered) for power_value in powered ]
            return random.choices(range(self.num_actions), actions_probability)[0]
        else:
            return action_scores.index(max(action_scores))

    def ObserveAction(self, oldState, action, newState, reward, learningRateScale):
        old_state_index = 1
        for shape in oldState:
            old_state_index = old_state_index * shape

        new_state_index = 1
        for shape in newState:
            new_state_index = new_state_index * shape

        learning_rate = 1 / (1 + learningRateScale * self.visit_n[old_state_index + action])
        self.q_table[old_state_index + action] += learning_rate * (reward + self.discount_rate * max(self.q_table[new_state_index:new_state_index + self.num_actions]) - self.q_table[old_state_index + action])
        self.visit_n[old_state_index + action] += 1


    def record(self, oldState, action, newState, reward):
        self.memory.append((oldState, action, newState, reward))

    def replay(self, learningRateScale):
        for oldState, action, newState, reward in self.memory:
            self.ObserveAction(oldState, action, newState, reward, learningRateScale)

    def clear_record(self):
        self.memory.clear()
