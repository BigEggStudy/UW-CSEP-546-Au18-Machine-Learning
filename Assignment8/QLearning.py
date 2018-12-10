from collections import deque
import random
import math

class QLearning(object):
    def __init__(self, stateSpaceShape, numActions, discountRate):
        self.stateSpaceShape = tuple(stateSpaceShape)
        self.num_actions = numActions
        self.discount_rate = discountRate

        max_index = self.num_actions
        for shape in stateSpaceShape:
            max_index *= shape

        self.q_table = [0] * max_index
        self.visit_n = [0] * max_index

        self.memory = deque(maxlen=2000)

    def GetAction(self, currentState, learningMode, randomActionRate = 0.01, actionProbabilityBase = 1.5):
        if learningMode and random.random() < randomActionRate:
            return random.randrange(self.num_actions)

        current_state_index = self.__get_index(currentState)
        action_scores = self.q_table[current_state_index:current_state_index + self.num_actions]

        if learningMode:
            powered = [ actionProbabilityBase ** action_score for action_score in action_scores ]
            sum_value = sum(powered)
            actions_probability = [ power_value / sum_value for power_value in powered ]
            return random.choices(range(self.num_actions), actions_probability)[0]
        else:
            return action_scores.index(max(action_scores))

    def ObserveAction(self, oldState, action, newState, reward, learningRateScale):
        old_state_index = self.__get_index(oldState)
        new_state_index = self.__get_index(newState)
        old_state_index_action = old_state_index + action

        learning_rate = 1 / (1 + learningRateScale * self.visit_n[old_state_index_action])
        self.q_table[old_state_index_action] += learning_rate * (reward + self.discount_rate * max(self.q_table[new_state_index:new_state_index + self.num_actions]) - self.q_table[old_state_index_action])
        self.visit_n[old_state_index_action] += 1


    def record(self, oldState, action, newState, reward):
        self.memory.append((oldState, action, newState, reward))

    def replay(self, learningRateScale):
        for _ in range(len(self.memory)):
            for oldState, action, newState, reward in self.memory:
                self.ObserveAction(oldState, action, newState, reward, learningRateScale)

    def clear_record(self):
        self.memory.clear()

    def __get_index(self, state):
        state_index = 0
        for i in range(len(state)):
            state_index = state_index * self.stateSpaceShape[i] + state[i]
        return state_index * self.num_actions
