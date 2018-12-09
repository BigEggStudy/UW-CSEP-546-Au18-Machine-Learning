import numpy as np
import gym
import datetime

env = gym.make('CartPole-v0')

import random
import QLearning # Your implementation goes here...
import Assignment7Support

discountRate = 0.98          # Controls the discount rate for future rewards -- this is gamma from 13.10
actionProbabilityBase = 1.8  # This is k from the P(a_i|s) expression from section 13.3.5 and influences how random exploration is
randomActionRate = 0.01      # Percent of time the next action selected by GetAction is totally random
learningRateScale = 0.01     # Should be multiplied by visits_n from 13.11.
trainingIterations = 20000

scores = []
print('========== Start 10 runs of Cart Pole ==========')
for runs in range(10):
    qlearner = QLearning.QLearning(stateSpaceShape=Assignment7Support.CartPoleStateSpaceShape(), numActions=env.action_space.n, discountRate=discountRate)

    print(f'[{datetime.datetime.now()}] Start training, runs id {runs + 1}')
    for trialNumber in range(trainingIterations):
        observation = env.reset()
        reward = 0
        for i in range(300):
            #env.render()

            currentState = Assignment7Support.CartPoleObservationToStateSpace(observation)
            action = qlearner.GetAction(currentState, learningMode=True, randomActionRate=randomActionRate, actionProbabilityBase=actionProbabilityBase)

            oldState = Assignment7Support.CartPoleObservationToStateSpace(observation)
            observation, reward, isDone, info = env.step(action)
            newState = Assignment7Support.CartPoleObservationToStateSpace(observation)

            qlearner.ObserveAction(oldState, action, newState, reward, learningRateScale=learningRateScale)

            if isDone:
                if (trialNumber + 1) % 1000 == 0:
                    print(trialNumber + 1, i + 1, np.max(qlearner.q_table), np.mean(qlearner.q_table))
                break
    print(f'[{datetime.datetime.now()}] End of the traininig, runs id {runs + 1}')

    ## Now do the best n runs I can
    # input('Enter to continue...')

    n = 20
    totalRewards = []
    for runNumber in range(n):
        observation = env.reset()
        totalReward = 0
        reward = 0
        for i in range(300):
            # renderDone = env.render()

            currentState = Assignment7Support.CartPoleObservationToStateSpace(observation)
            observation, reward, isDone, info = env.step(qlearner.GetAction(currentState, learningMode=False))

            totalReward += reward

            if isDone:
                # renderDone = env.render()
                print(runNumber + 1, i + 1, totalReward)
                totalRewards.append(totalReward)
                break

    # env.close()

    average_score = sum(totalRewards) / float(len(totalRewards))
    print(f'[{datetime.datetime.now()}] End of the Test, runs id {runs + 1}')
    print(totalRewards)
    print(f'Your Score: {average_score}')
    scores.append(average_score)

print('All runs complete')
print(f'The scores are: {scores}')
print(f'All runs complete {sum(scores) / float(len(scores))}')