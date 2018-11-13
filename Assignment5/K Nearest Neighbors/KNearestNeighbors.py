import math
import operator
from joblib import Parallel, delayed

class KNearestNeighbors(object):
    def fit(self, x, y):
        self.xTraining = x
        self.yTraining = y
        self.cache_neighbors = {}

    def predict(self, x, k, threshold = 0.5):
        results = []
        for idx, sample in enumerate(x):
            neighbors = self.__get_neighbors(sample, idx, k)
            result = self.__vote(neighbors, threshold)
            results.append(result)
        return results

    def __get_neighbors(self, sample, index, k):
        if index not in self.cache_neighbors:
            def compute(sample, training, label):
                dist = self.__euclidean_distance(sample, training)
                return (label, dist)

            distances = []
            for i in range(len(self.xTraining)):
                distances.append(compute(sample, self.xTraining[i], self.yTraining[i]))
            distances.sort(key=operator.itemgetter(1))

            self.cache_neighbors[index] = distances

        distances = self.cache_neighbors[index]

        neighbors = []
        for x in range(k):
            neighbors.append(distances[x][0])
        return neighbors

    def __vote(self, neighbors, threshold):
        value = sum(neighbors) / len(neighbors)
        return 1 if value >= threshold else 0

    def __euclidean_distance(self, a, b):
        def compute(a, b):
            return (a - b) ** 2

        sum_value = 0
        for i in range(len(a)):
            sum_value += compute(a[i], b[i])
        return math.sqrt(sum_value)
