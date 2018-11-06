import math
import sys
import numpy as np

class DecisionTree(object):
    """A Decision Tree Classifier"""

    def __init__(self, minToSplit = 100):
        self.minToSplit = minToSplit

    def fit(self, x, y):
        self.root = self.__grow_tree(x, y)

    def visualize(self, feature_names):
        self.__visualize_core(self.root, feature_names)

    def predict(self, x, threshold = 0.5):
        predict_result = []
        for i in range(len(x)):
            predict_result.append(self.__predict_core(x[i], self.root, threshold))
        return predict_result

    def __entropy(self, S):
        '''
        Entropy(S) = SUM_i(- p_i * log_2(p_i))
        '''
        sum_value = 0
        for i in S:
            p = i / sum(S)
            if p != 0:
                sum_value += p * math.log2(p)
            else:
                sum_value += 0
        return -sum_value


    def gain(self, S, A):
        '''
        Gain(S, A) = Entropy(S) − SUM_v( |S_v| / |S| * Entropy(S_v) )
        '''
        sum_value = 0
        for v in A:
            sum_value += sum(v) / sum(S) * self.__entropy(v) if sum(v) > 0 else 0

        return self.__entropy(S) - sum_value

    def __compute_feature_thresholds(self, x):
        result = []
        for feature_index in range(len(x[0])):
            min_value = 1000
            max_value = -1
            for i in range(len(x)):
                min_value = min(min_value, x[i][feature_index])
                max_value = max(max_value, x[i][feature_index])
            result.append((min_value + max_value) / 2)
        return result

    def __grow_tree(self, x, y):
        thresholds =  self.__compute_feature_thresholds(x)

        if sum(y) == 0:
            return (None, None, len(y), 0)
        elif sum(y) == len(y):
            return (None, None, 0, len(y))
        elif len(y) < self.minToSplit:
            return (None, None, len(y) - sum(y), sum(y))
        else:
            feature_index = self.__choose_best_attribute(x, y, thresholds)
            if feature_index is None:
                return (None, None, len(y) - sum(y), sum(y))

            x_0 = []
            y_0 = []
            x_1 = []
            y_1 = []
            for i in range(len(x)):
                if x[i][feature_index] < thresholds[feature_index]:
                    x_0.append(x[i])
                    y_0.append(y[i])
                else:
                    x_1.append(x[i])
                    y_1.append(y[i])
            return (self.__grow_tree(x_0, y_0), self.__grow_tree(x_1, y_1), feature_index, thresholds[feature_index])

    def __choose_best_attribute(self, x, y, thresholds):
        max_gains = 0
        max_gains_feature_index = -1
        S = [sum(y), len(y) - sum(y)]
        for feature_index in range(len(x[0])):
            A = [[0, 0], [0, 0]]
            for i in range(len(x)):
                if x[i][feature_index] < thresholds[feature_index]:
                    if y[i] == 1:
                        A[0][0] += 1
                    else:
                        A[0][1] += 1
                else:
                    if y[i] == 1:
                        A[1][0] += 1
                    else:
                        A[1][1] += 1
            gain = self.gain(S, A)
            if max_gains < gain:
                max_gains = gain
                max_gains_feature_index = feature_index

        return max_gains_feature_index if max_gains_feature_index >= 0 else None

    def __visualize_core(self, node, feature_names, indentation = 0):
        if node is None:
            print(self.__repeat_to_length(' ', indentation * 4) + 'None')
        elif node[0] is None and node[1] is None:
            #   Leaf
            print(self.__repeat_to_length(' ', indentation * 4) + ('Leaf: %d with label 0, %d with label 1' % (node[2], node[3])))
        else:
            print(self.__repeat_to_length(' ', indentation * 4) + ('Feature (%d) "%s":' % (node[2], feature_names[node[2]])))
            print(self.__repeat_to_length(' ', (indentation + 1) * 4) + ('>= %f:' % node[3]))
            self.__visualize_core(node[1], feature_names, indentation + 2)
            print(self.__repeat_to_length(' ', (indentation + 1) * 4) + ('< %f:' % node[3]))
            self.__visualize_core(node[0], feature_names, indentation + 2)

    def __repeat_to_length(self, string_to_expand, length):
        return (string_to_expand * (int(length/len(string_to_expand))+1))[:length]

    def __predict_core(self, xi, node, threshold):
        if node is None:
            print('Error! Should not show this')
            return -1
        elif node[0] is None and node[1] is None:
            return 1 if node[3] / (node[3] + node[2]) >= threshold else 0
        else:
            if xi[node[2]] >= node[3]:
                return self.__predict_core(xi, node[1], threshold)
            else:
                return self.__predict_core(xi, node[0], threshold)
