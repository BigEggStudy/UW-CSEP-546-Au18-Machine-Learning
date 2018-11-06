import matplotlib.pyplot as plt

import Assignment1Support
import EvaluationsStub

### UPDATE this path for your environment
kDataPath = "..\\Data\\SMSSpamCollection"

(xRaw, yRaw) = Assignment1Support.LoadRawData(kDataPath)

(xTrainRaw, yTrainRaw, xTestRaw, yTestRaw) = Assignment1Support.TrainTestSplit(xRaw, yRaw)

print("Train is %f percent spam." % (sum(yTrainRaw)/len(yTrainRaw)))
print("Test is %f percent spam." % (sum(yTestRaw)/len(yTestRaw)))

(xTrain, xTest) = Assignment1Support.Featurize(xTrainRaw, xTestRaw)
yTrain = yTrainRaw
yTest = yTestRaw

############################
import MostCommonModel

model = MostCommonModel.MostCommonModel()
model.fit(xTrain, yTrain)
yTestPredicted = model.predict(xTest)

print("### 'Most Common' model")

EvaluationsStub.ExecuteAll(yTest, yTestPredicted)

############################
import SpamHeuristicModel
model = SpamHeuristicModel.SpamHeuristicModel()
model.fit(xTrain, yTrain)
yTestPredicted = model.predict(xTest)

print("### Heuristic model")

EvaluationsStub.ExecuteAll(yTest, yTestPredicted)

############################
import LogisticRegressionModel
model = LogisticRegressionModel.LogisticRegressionModel()

print("### Logistic regression model")
for i in [50000]:
    model.fit(xTrain, yTrain, iterations=i, step=0.01, plot=True)
    yTestPredicted = model.predict(xTest)
    print("%d, %f, %f, %f" % (i, model.weights[1], model.loss(xTest, yTest), EvaluationsStub.Accuracy(yTest, yTestPredicted)))

print("")
print("Plot the test set loss, test set accuracy, and value of weight[1] after every 10,000 iterations.")
x_data = []
y_data_loss = []
y_data_accuracy = []
y_data_weight = []
for i in range(0, 50001, 10000):
    model.fit(xTrain, yTrain, iterations=i, step=0.01)
    yTestPredicted = model.predict(xTest)
    x_data.append(i)
    loss = model.loss(xTest, yTest)[0][0]
    accuracy = EvaluationsStub.Accuracy(yTest, yTestPredicted)
    weight1 = model.weights[1]
    print('Test Set Statistics after %d iteration, loss %f, accuracy %f, and weight[1] %f' % (i, loss, accuracy, weight1))
    y_data_loss.append(loss)
    y_data_accuracy.append(accuracy)
    y_data_weight.append(weight1)

plt.plot(x_data, y_data_loss, label = 'Test Set Loss')
plt.plot(x_data, y_data_accuracy, label = 'Test Set Accuracy')
plt.plot(x_data, y_data_weight, label = 'Weight')
plt.xlabel('Iteration')
plt.ylabel('Test Set Statistics')
plt.title('Test Set Statistics vs Iteration')
plt.legend()
print("Close the plot diagram to continue program")
plt.show()

print("")
print("Calculate all the statistics from the evaluation framework on the 50,000 iteration run")
EvaluationsStub.ExecuteAll(yTest, yTestPredicted)
