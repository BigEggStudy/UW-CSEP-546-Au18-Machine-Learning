import numpy as np

class NeuralNetworks(object):
    def __init__(self, input_num, hidden_layer_num = 1, hidden_node_num = 2, output_num = 1):
        self.hidden_layer_num = hidden_layer_num
        self.network = {}
        self.network['weights'] = self.__initialize_network(input_num, hidden_layer_num, hidden_node_num, output_num)

    def fit(self, x, y, step):
        output = self.__forward_propagate(self.network, x)
        self.__backward_propagate(self.network, x, y, step, output)

    def predict(self, x, threshold = 0.5):
        (m, n) = np.shape(x)
        output = self.__forward_propagate(self.network, x)
        return (output > threshold).astype(int).reshape(m)

    def predict_raw(self, x):
        return self.__forward_propagate(self.network, x)

    def loss(self, x, y):
        return sum(np.square(y - self.predict_raw(x))) / 2

    def __initialize_network(self, inputs_num, hidden_layer_num, hidden_node_num, output_num):
        weights = list()

        first_hidden_layer = np.random.randn(inputs_num + 1, hidden_node_num)
        weights.append(first_hidden_layer)

        for layer in range(1, hidden_layer_num):
            hidden_layer = np.random.randn(hidden_node_num + 1, hidden_node_num)
            weights.append(hidden_layer)

        output_layer = np.random.randn(hidden_node_num + 1, output_num)
        weights.append(output_layer)
        return weights

    def __forward_propagate(self, network, x):
        outputs = list()
        inputs = x
        for i in range(self.hidden_layer_num + 1):
            weights = network['weights'][i]
            (m, n) = np.shape(inputs)
            ones = np.ones([m, 1])
            X = np.hstack([ones, inputs])

            output = self.__sigmoid(np.dot(X, weights))
            outputs.append(output)

            inputs = output
        network['outputs'] = outputs
        return output

    def __backward_propagate(self, network, x, y, step, output):
        output_error = y - output
        output_delta = output_error * self.__sigmoid_prime(output)

        deltas = list()
        deltas.append(output_delta)

        next_layer_delta = output_delta
        for i in range(self.hidden_layer_num - 1, -1, -1):
            current_layer_error = next_layer_delta.dot(network['weights'][i + 1].T)
            current_layer_delta = np.delete(current_layer_error, 0, 1) * self.__sigmoid_prime(network['outputs'][i])
            deltas.insert(0, current_layer_delta)
            next_layer_delta = current_layer_delta

        inputs = x
        for i in range(self.hidden_layer_num + 1):
            (m, n) = np.shape(inputs)
            ones = np.ones([m, 1])
            X = np.hstack([ones, inputs])

            network['weights'][i] += step * X.T.dot(deltas[i])
            inputs = network['outputs'][i]

    def __sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))

    def __sigmoid_prime(self, x):
        #derivative of sigmoid
        return x * (1 - x)