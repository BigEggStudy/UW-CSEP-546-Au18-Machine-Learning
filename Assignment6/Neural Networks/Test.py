import numpy as np
import NeuralNetworks
import matplotlib.pyplot as plt

X = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)
y = np.array(([92], [86], [89]), dtype=float)

# scale units
X = X/np.amax(X, axis=0) # maximum of X array
y = y/100 # max test score is 100

NN = NeuralNetworks.NeuralNetworks(2, [ 5, 3 ], 1)
iteration = []
loss = []
for i in range(1000):
    print('Input: %s' % X)
    print('Actual Output: %s' % y)
    print('Predicted Output: %s' % NN.predict_raw(X))
    print('Loss: %s' % NN.loss(X, y))
    # print('Weights: %s' % NN.weights)
    print('')

    NN.fit_one(X, y, 20, 0.05)

    iteration.append(i)
    loss.append(NN.loss(X, y))

fig, ax = plt.subplots()
ax.grid(True)
plt.plot(iteration, loss, label = 'Training')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Neural Networks\'s Loss on Training Set')
plt.legend()
print("Close the plot diagram to continue program")
plt.show()