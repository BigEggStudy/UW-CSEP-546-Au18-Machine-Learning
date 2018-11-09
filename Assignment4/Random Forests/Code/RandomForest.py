import math
import sys
import numpy as np
import random
from joblib import Parallel, delayed

import DecisionTreeModel

class RandomForest(object):
    """A Random Forest Classifier"""

    def __init__(self, num_trees, min_to_split, use_bagging, restrict_features):
        self.num_trees = num_trees
        self.min_to_split = min_to_split
        self.use_bagging = use_bagging
        self.restrict_features = restrict_features

    def fit(self, x, y):
        self.trees = Parallel(n_jobs=6)(delayed(self.growTree)(i, x, y, self.min_to_split, random.randint(0,1000000), self.use_bagging, self.restrict_features) for i in range(self.num_trees))
        # self.trees = [ self.growTree(i, x, y, self.min_to_split, random.randint(0,1000000), self.use_bagging, self.restrict_features) for i in range(self.num_trees) ]

    def growTree(self, index, x, y, min_to_split, random_seed, use_bagging, restrict_features):
        (newX, newY) = self.__bagging(x, y, random_seed, use_bagging)
        tree = DecisionTreeModel.DecisionTree(min_to_split, random_seed, restrict_features)
        tree.fit(newX, newY)
        return tree

    def __bagging(self, x, y, random_seed, use_bagging):
        if not use_bagging:
            return (x, y)

        random.seed(random_seed)
        newX = []
        newY = []
        for i in range(len(x)):
            index = random.randint(0, len(x) - 1)
            newX.append(x[index])
            newY.append(y[index])

        return (newX, newY)

    def predict(self, x, threshold = 0.5):
        tree_predictions = []
        for tree in self.trees:
            tree_predictions.append(tree.predict(x, threshold))

        predictions = []
        for i in range(len(tree_predictions[0])):
            predictions.append(tree_predictions[0][i])
        for i in range(1, self.num_trees):
            for j in range(len(predictions)):
                predictions[j] += tree_predictions[i][j]
        for i in range(len(predictions)):
            predictions[i] = predictions[i] / self.num_trees
            predictions[i] = 1 if predictions[i] >= threshold else 0

        return (predictions, tree_predictions)
