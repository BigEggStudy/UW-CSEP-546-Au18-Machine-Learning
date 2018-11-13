import math
import random
from joblib import Parallel, delayed

class KMeansCluster(object):
    """A K-Means Cluster"""

    def find_clusters(self, x, k, iterations = 10):
        centroids_path = []
        centroids = [ x[random.randint(0, len(x) - 1)] for i in range(k) ]
        centroids_path.append(centroids)

        for i in range(iterations):
            assignments = self.__assign_samples(x, centroids)
            centroids = self.__update_centroid(assignments)
            centroids_path.append(centroids)

        return (centroids, assignments, centroids_path)

    def __assign_samples(self, x, centroids):
        assignments = {}
        for i in range(len(centroids)):
            assignments[i] = []

        def assign_sample(index, sample, centroids):
            nearest_distance = 1000000
            nearest_centroid = -1
            for i in range(len(centroids)):
                centroid = centroids[i]
                distance = self.euclidean_distance(sample, centroid)
                if distance < nearest_distance:
                    nearest_distance = distance
                    nearest_centroid = i
            return (nearest_centroid, sample, index)

        assigns = Parallel(n_jobs=6)(delayed(assign_sample)(i, x[i], centroids) for i in range(len(x)))
        for (nearest_centroid, sample, index) in assigns:
            assignments[nearest_centroid].append((sample, index))

        return assignments

    def __update_centroid(self, assignments):
        def compute(assignment):
            centroid = []
            dimension = len(assignment[0][0])

            for j in range(dimension):
                centroid.append(0)

            for (sample, index) in assignment:
                for j in range(dimension):
                    centroid[j] += sample[j]

            for j in range(dimension):
                centroid[j] /= len(assignment)

            return centroid

        new_centroids = Parallel(n_jobs=6)(delayed(compute)(assignment) for assignment in assignments.values())
        return new_centroids

    def euclidean_distance(self, sample, centroid):
        def compute(a, b):
            return (a - b) ** 2

        sum_value = 0
        for i in range(len(sample)):
            sum_value += compute(sample[i], centroid[i])
        return math.sqrt(sum_value)
