import matplotlib.pyplot as plt
import numpy as np

class LogisticRegressionModel(object):
    """A Logistic Regression filter"""

    def __init__(self):
        pass

    def fit(self, x, y, iterations = 50000, step = 0.01, plot = False):
        (m, n) = np.shape(x)

        ones = np.ones([m, 1])
        X = np.hstack([ones, x])
        Y = np.array(y).reshape([-1, 1])
        theta = np.zeros([n + 1, 1])

        x_data = []
        y_data = []

        for i in range(iterations):
            theta = self._gradientDescent(X, Y, theta, step)
            if plot and i % 1000 == 0:
                x_data.append(i)

                h = self._sigmoid(np.matmul(X, theta))
                loss = (np.matmul(-Y.T, np.log(h)) - np.matmul((1 - Y.T), np.log(1 - h))) / m
                y_data.append(loss[0][0])

        self.weights = theta.T[0]

        if plot:
            plt.plot(x_data, y_data)
            plt.xlabel('Iteration')
            plt.ylabel('Training Set Loss')
            plt.title('Training Set Loss vs Iteration')
            print("Close the plot diagram to continue program")
            plt.show()


    def loss(self, x, y):
        (m, n) = np.shape(x)

        ones = np.ones([m, 1])
        X = np.hstack([ones, x])
        Y = np.array(y).reshape([-1, 1])
        theta = np.array(self.weights).reshape([-1, 1])

        h = self._sigmoid(np.matmul(X, theta))
        cost = (np.matmul(-Y.T, np.log(h)) - np.matmul((1 - Y.T), np.log(1 - h))) / m
        return cost

    def predict(self, x):
        (m, n) = np.shape(x)

        ones = np.ones([m, 1])
        X = np.hstack([ones, x])
        theta = np.array(self.weights).reshape([-1, 1])

        predictions = []

        h = self._sigmoid(np.matmul(X, theta))
        return (h > .5).astype(int).reshape(m)

    def predictWithThreshold(self, x, threshold = 0.5):
        (m, n) = np.shape(x)

        ones = np.ones([m, 1])
        X = np.hstack([ones, x])
        theta = np.array(self.weights).reshape([-1, 1])

        predictions = []

        h = self._sigmoid(np.matmul(X, theta))
        return (h > threshold).astype(int).reshape(m)

    def _sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))

    def _gradientDescent(self, x, y, theta, alpha):
        (m, n) = np.shape(x)

        h = self._sigmoid(np.matmul(x, theta))
        grad = np.matmul(x.T, (h - y)) / m
        theta = theta - alpha * grad
        return theta