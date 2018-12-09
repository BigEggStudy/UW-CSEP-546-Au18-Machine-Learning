import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

import random
import datetime

import gym
env = gym.make('MountainCar-v0')

import QLearning # your implementation goes here...
import Assignment7Support

discountRate = 0.98          # Controls the discount rate for future rewards -- this is gamma from 13.10
actionProbabilityBase = 1.8  # This is k from the P(a_i|s) expression from section 13.3.5 and influences how random exploration is
randomActionRate = 0.01      # Percent of time the next action selected by GetAction is totally random
learningRateScale = 0.01     # Should be multiplied by visits_n from 13.11.
trainingIterations = 20000

if __name__=="__main__":
    def training_ten(discountRate = 0.98, actionProbabilityBase = 1.8, trainingIterations = 20000, mountainCarBinsPerDimension = 20):
        print('10 Attempt for this parameters set')
        print(f'discountRate = {discountRate}, actionProbabilityBase = {actionProbabilityBase}, trainingIterations = {trainingIterations}, mountainCarBinsPerDimension = {mountainCarBinsPerDimension}')
        total_scores = Parallel(n_jobs=6)(delayed(training_one)(discountRate, actionProbabilityBase, trainingIterations, mountainCarBinsPerDimension, False) for _ in range(10))
        return (sum(total_scores) / float(len(total_scores)), total_scores)

    def training_one(discountRate = 0.98, actionProbabilityBase = 1.8, trainingIterations = 20000, mountainCarBinsPerDimension = 20, render = False):
        qlearner = QLearning.QLearning(stateSpaceShape=Assignment7Support.MountainCarStateSpaceShape(mountainCarBinsPerDimension), numActions=env.action_space.n, discountRate=discountRate)

        for trialNumber in range(trainingIterations):
            observation = env.reset()
            reward = 0
            for i in range(200):

                currentState = Assignment7Support.MountainCarObservationToStateSpace(observation, mountainCarBinsPerDimension)
                action = qlearner.GetAction(currentState, learningMode=True, randomActionRate=randomActionRate, actionProbabilityBase=actionProbabilityBase)

                oldState = Assignment7Support.MountainCarObservationToStateSpace(observation, mountainCarBinsPerDimension)
                observation, reward, isDone, info = env.step(action)
                newState = Assignment7Support.MountainCarObservationToStateSpace(observation, mountainCarBinsPerDimension)

                # learning rate scale
                qlearner.ObserveAction(oldState, action, newState, reward, learningRateScale=learningRateScale)

                if isDone:
                #     if (trialNumber + 1) % 1000 == 0:
                #         print(trialNumber + 1, i + 1, np.min(qlearner.q_table), np.mean(qlearner.q_table))
                    break

        n = 20
        totalRewards = []
        for runNumber in range(n):
            observation = env.reset()
            totalReward = 0
            reward = 0
            for i in range(200):
                if render:
                    renderDone = env.render()

                currentState = Assignment7Support.MountainCarObservationToStateSpace(observation, mountainCarBinsPerDimension)
                observation, reward, isDone, info = env.step(qlearner.GetAction(currentState, learningMode=False))

                totalReward += reward

                if isDone:
                    if render:
                        renderDone = env.render()
                    # print(runNumber + 1, i + 1, totalReward)
                    totalRewards.append(totalReward)
                    break

        if render:
            env.close()

        average_score = sum(totalRewards) / float(len(totalRewards))
        print(f'[{datetime.datetime.now()}] The average score of this one attempt is {average_score}')

        return average_score

    def plot_result(x, y, diagram_name, parameter_name, save_time = False):
        print('')
        print(f'### Plot {diagram_name}.')
        if save_time:
            print(x)
            print(y)
            return

        fig, ax = plt.subplots()
        ax.grid(True)

        plt.plot(x, y)
        plt.xlabel(parameter_name)
        plt.ylabel('Score')
        plt.title(diagram_name)
        plt.legend()

        print('Close the plot diagram to continue program')
        plt.show()

    #########################################

    best_score = float('-Inf')
    best_base = 0
    x = []
    y = []
    print('Tune the Action Probability Base')
    for base in [1.1, 1.2, 1.3, 1.4, 1.5, 1.8, 2.7, 5, 7]:
        print(f'[{datetime.datetime.now()}] Training with actionProbabilityBase {base}')
        score = training_ten(actionProbabilityBase=base)
        x.append(base)
        y.append(score)
        if score > best_score:
            best_score = score
            best_base = base
        print(f'[{datetime.datetime.now()}] The average score is {score}')

    plot_result(x, y, 'Action Probability Base vs Score', 'Action Probability Base', True)
    print(f'When Action Probability Base is {best_base}, the Q-Learning Agent performance the best')
    print(f'When Score is {best_score}')

    #########################################

    best_score = float('-Inf')
    best_bins = 0
    x = []
    y = []
    print('Tune the Bins per Dimension')
    for bins in range(20, 201, 10):
        print(f'[{datetime.datetime.now()}] mountainCarBinsPerDimension {bins}')
        score, all_score = training_ten(actionProbabilityBase=best_base, mountainCarBinsPerDimension=bins)
        x.append(bins)
        y.append(score)
        if score > best_score:
            best_score = score
            best_bins = bins
        print(f'[{datetime.datetime.now()}] The average score is {score}')

    plot_result(x, y, 'Bins per Dimension vs Score', 'Bins per Dimension', True)
    print(f'When Bins per Dimension is {best_bins}, the Q-Learning Agent performance the best')
    print(f'When Score is {best_score}')

    #########################################

    best_score = float('-Inf')
    best_discount_rate = 0
    x = []
    y = []
    print('Tune the Discount Rate')
    for discount_rate in [1, 0.99, 0.98, 0.97, 0.96, 0.95, 0.9, 0.8, 0.75]:
        print(f'[{datetime.datetime.now()}] Training with discountRate {discount_rate}')
        score = training_ten(mountainCarBinsPerDimension=best_bins, trainingIterations=best_iteration, actionProbabilityBase=best_base, discountRate=discount_rate)
        x.append(discount_rate)
        y.append(score)
        if score > best_score:
            best_score = score
            best_discount_rate = base
        print(f'[{datetime.datetime.now()}] The average score is {score}')

    plot_result(x, y, 'Discount Rate vs Score', 'Discount Rate', True)
    print(f'When Discount Rate is {best_discount_rate}, the Q-Learning Agent performance the best')
    print(f'When Score is {best_score}')

    #########################################

    best_score = float('-Inf')
    best_iteration = 0
    x = []
    y = []
    print('Tune the Training Iterations')
    for iteration in [20000, 25000, 30000, 35000, 40000, 50000]:
        print(f'[{datetime.datetime.now()}] Training with trainingIterations {iteration}')
        score = training_ten(actionProbabilityBase=best_base, mountainCarBinsPerDimension=best_bins, trainingIterations=iteration)
        x.append(iteration)
        y.append(score)
        if score > best_score:
            best_score = score
            best_iteration = iteration
        print(f'[{datetime.datetime.now()}] The average score is {score}')

    plot_result(x, y, 'Training Iterations vs Score', 'Training Iterations', True)
    print(f'When Training Iterations is {best_iteration}, the Q-Learning Agent performance the best')
    print(f'When Score is {best_score}')
