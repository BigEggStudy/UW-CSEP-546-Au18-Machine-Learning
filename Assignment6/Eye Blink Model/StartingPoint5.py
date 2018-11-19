## Some of this references my answers to previous assignments.
##  Replace my references with references to your answers to those assignments.

## IMPORTANT NOTE !!
## Remember to install the Pillow library (which is required to execute 'import PIL')

import Assignment5Support
import EvaluationsStub
import matplotlib.pyplot as plt
import numpy as np

import Featurize

if __name__=="__main__":
    kDataPath = "..\\..\\dataset_B_Eye_Images"

    (xRaw, yRaw) = Assignment5Support.LoadRawData(kDataPath, includeLeftEye = True, includeRightEye = True)
    (xTrainRaw, yTrainRaw, xTestRaw, yTestRaw) = Assignment5Support.TrainTestSplit(xRaw, yRaw, percentTest = .25)
    print("Train is %f percent closed." % (sum(yTrainRaw)/len(yTrainRaw)))
    print("Test is %f percent closed." % (sum(yTestRaw)/len(yTestRaw)))

    yTrain = np.array(yTrainRaw)[np.newaxis].T
    yTest = np.array(yTestRaw)[np.newaxis].T

    import NeuralNetworks
    import CrossValidation
    crossValidation = CrossValidation.CrossValidation(5)
    ############################

    print('========== Compare Features ==========')
    best_accuracy = 0
    best_parameter = (0, 0.0)
    for step in [1, 2, 4]:
        print('Use Intensities with step %d' % step)
        (xTrain, xTest) = Featurize.ByIntensities(xTrainRaw, xTestRaw, step)
        for momentum_beta in [0.0, 0.25]:
            print("Training Model with momentum beta as %f" % momentum_beta)
            accuracy = crossValidation.validate(xTrain, yTrain, beta=momentum_beta)
            (lower, upper) = EvaluationsStub.Bound(accuracy, len(xTrainRaw))
            print("Accuracy from Cross Validation is %f, with lower bound %f and upper bound %f" % (accuracy, lower, upper))
            if best_accuracy < accuracy:
                best_accuracy = accuracy
                best_parameter = (step, momentum_beta)

    print('When Intensities with step %d and momentum beta as %f,' % best_parameter)
    print('Neural Networks has best accuracy %f' % best_accuracy)

    ############################

    print('========== Parameter Tuning ==========')
    print('Featurize the Data Set')
    (xTrain, xTest) = Featurize.ByIntensities(xTrainRaw, xTestRaw, 1)

    best_accuracy = 0
    best_parameter = (150, 10, 0.05, 0.0)
    for iteration in range(150, 251, 50):
        for mini_batch_size in [10, 30, 50, 75, 100, 125, 150, 200]:
            for eta in [0.01, 0.05, 0.1, 0.5, 1, 10]:
                for momentum_beta in [0.25, 0.33, 0.5]:
                    print("Training Model with %d iteration, mini batch size as %d, eta as %f, momentum beta as %f" % (iteration, mini_batch_size, eta, momentum_beta))
                    accuracy = crossValidation.validate(xTrain, yTrain, iteration, mini_batch_size, eta, momentum_beta)
                    (lower, upper) = EvaluationsStub.Bound(accuracy, len(xTrainRaw))
                    print("Accuracy from Cross Validation is %f, with lower bound %f and upper bound %f" % (accuracy, lower, upper))
                    if best_accuracy < accuracy:
                        best_accuracy = accuracy
                        best_parameter = (iteration, mini_batch_size, eta, momentum_beta)

    print('When %d iteration, mini batch size as %d, eta as %f, momentum beta as %f,' % best_parameter)
    print('Neural Networks has best accuracy %f' % best_accuracy)

    ############################

    # best_parameter = (150, 10, 1, 0.25)

    print('========== Mistake Categorize ==========')
    print('Build Neural Network with 1 hidden layer, and 20 nodes for each layer')
    print('Use %d iteration, mini batch size as %d, eta as %f, momentum beta as %f to Train the Model' % best_parameter)
    model = NeuralNetworks.NeuralNetworks(len(xTrain[0]), [ 20 ], 1)
    for i in range(best_parameter[0]):
        model.fit_one(xTrain, yTrain, best_parameter[1], best_parameter[2], best_parameter[3])
    predicted = model.predict_raw(xTrain)

    most_error = sorted(zip(predicted, yTrain, xTrainRaw), key=lambda data: data[0] - data[1])
    print("### Find the Most False Negative ###")
    print(most_error[:10])
    print("### Find the Most False Positive ###")
    print(most_error[-10:])

    ############################

    print('========== Evaluation on the Neural Networks ==========')
    print('Add X-Gradient and Y-Gradient as the Features')
    xTrain_X_Gradient = Featurize.Featurize_X_Gradient(xTrainRaw)
    xTrain_Y_Gradient = Featurize.Featurize_Y_Gradient(xTrainRaw)
    xTest_X_Gradient = Featurize.Featurize_X_Gradient(xTestRaw)
    xTest_Y_Gradient = Featurize.Featurize_Y_Gradient(xTestRaw)

    xTrain_Gradient = np.hstack((np.array(xTrain_X_Gradient), np.array(xTrain_Y_Gradient)))
    xTest_Gradient = np.hstack((np.array(xTest_X_Gradient), np.array(xTest_Y_Gradient)))

    xTrain_Best = np.hstack((np.array(xTrain), xTrain_Gradient))
    xTest_Best = np.hstack((np.array(xTest), xTest_Gradient))

    print('Build Neural Network with 1 hidden layer, and 20 nodes for each layer')
    print('Use %d iteration, mini batch size as %d, eta as %f, momentum beta as %f to Train the Model' % best_parameter)

    accuracy = crossValidation.validate(xTrain_Best, yTrain, best_parameter[0], best_parameter[1], best_parameter[2], best_parameter[3])
    (lower, upper) = EvaluationsStub.Bound(accuracy, len(xTrainRaw))
    print("Accuracy from Cross Validation is %f, with lower bound %f and upper bound %f" % (accuracy, lower, upper))
    model = NeuralNetworks.NeuralNetworks(len(xTrain_Best[0]), [ 20 ], 1)
    for i in range(best_parameter[0]):
        model.fit_one(xTrain_Best, yTrain, best_parameter[1], best_parameter[2], best_parameter[3])
    yPredicted = model.predict(xTest_Best)
    testAccuracy = EvaluationsStub.Accuracy(yTest, yPredicted)
    (lower, upper) = EvaluationsStub.Bound(testAccuracy, len(yPredicted))
    print("Test Set Accuracy is %f, with lower bound %f and upper bound %f" % (testAccuracy, lower, upper))

    EvaluationsStub.ExecuteAll(yTest, yPredicted)

    ############################

    import RandomForest

    print('========== Compare Models ==========')
    print('Featurize the Data Set')
    model_X_Gradient = RandomForest.RandomForest(num_trees = 75, min_to_split = 25, use_bagging = True, restrict_features = 10)
    model_X_Gradient.fit(xTrain_X_Gradient, yTrain)
    false_positives_X_Gradient = []
    false_negatives_X_Gradient = []

    (xTrain_Init, xTest_Init) = Featurize.ByIntensities(xTrainRaw, xTestRaw, 2)
    init_model = NeuralNetworks.NeuralNetworks(len(xTrain_Init[0]), [ 20 ], 1)
    for i in range(200):
        init_model.fit_one(xTrain_Init, yTrain, 10, 0.05, 0.25)
    false_positives_init_NN = []
    false_negatives_init_NN = []

    best_model = NeuralNetworks.NeuralNetworks(len(xTrain_Best[0]), [ 20 ], 1)
    for i in range(200):
        best_model.fit_one(xTrain_Best, yTrain, 10, 0.05, 0.33)
    false_positives_best_NN = []
    false_negatives_best_NN = []

    for i in range(100):
        threshold = 0.01 + 0.99 * i / 99
        print('At threshold %f' % threshold)

        yTestPredicted = model_X_Gradient.predict(xTest_X_Gradient, threshold)
        false_positive = EvaluationsStub.FalsePositiveRate(yTest, yTestPredicted)
        false_negative = EvaluationsStub.FalseNegativeRate(yTest, yTestPredicted)
        false_positives_X_Gradient.append(false_positive)
        false_negatives_X_Gradient.append(false_negative)
        print('X-Gradient Random Forests Model has False Positive Rate %f, False Negative Rate %f' % (false_positive, false_negative))

        yTestPredicted = init_model.predict(xTest_Init, threshold)
        false_positive = EvaluationsStub.FalsePositiveRate(yTest, yTestPredicted)
        false_negative = EvaluationsStub.FalseNegativeRate(yTest, yTestPredicted)
        false_positives_init_NN.append(false_positive)
        false_negatives_init_NN.append(false_negative)
        print('Initialize Neural Networks Model has False Positive Rate %f, False Negative Rate %f' % (false_positive, false_negative))

        yTestPredicted = best_model.predict(xTest_Best, threshold)
        false_positive = EvaluationsStub.FalsePositiveRate(yTest, yTestPredicted)
        false_negative = EvaluationsStub.FalseNegativeRate(yTest, yTestPredicted)
        false_positives_best_NN.append(false_positive)
        false_negatives_best_NN.append(false_negative)
        print('Best Neural Networks Model has False Positive Rate %f, False Negative Rate %f' % (false_positive, false_negative))

    print("")
    print("### Plot Precision vs Recall.")
    fig, ax = plt.subplots()
    ax.grid(True)
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')

    plt.plot(false_positives_X_Gradient, false_negatives_X_Gradient, label = 'Random Forest Model use X-Gradient')
    plt.plot(false_positives_init_NN, false_negatives_init_NN, label = 'Initialize Neural Network Model')
    plt.plot(false_positives_best_NN, false_negatives_best_NN, label = 'Best Neural Network Model')
    plt.xlabel('False Positive Rate')
    plt.ylabel('False Negative Rate')
    plt.title('Test Set ROC Curve')
    plt.legend()

    print("Close the plot diagram to continue program")
    plt.show()

    ############################

    print('========== Deep Learning Models ==========')
    print('Use %d iteration, mini batch size as %d, eta as %f, momentum beta as %f to Train the Model' % best_parameter)

    for node in [2, 5, 10, 15, 20]:
        print('Build Neural Network with 2 hidden layer, and 20 nodes for first layer, %d for second layer' % node)
        model = NeuralNetworks.NeuralNetworks(len(xTrain_Best[0]), [ 20, node ], 1)
        for i in range(best_parameter[0]):
            model.fit_one(xTrain_Best, yTrain, best_parameter[1], best_parameter[2], best_parameter[3])
        yPredicted = model.predict(xTest_Best)
        testAccuracy = EvaluationsStub.Accuracy(yTest, yPredicted)
        (lower, upper) = EvaluationsStub.Bound(testAccuracy, len(yPredicted))
        print("Test Set Accuracy is %f, with lower bound %f and upper bound %f" % (testAccuracy, lower, upper))
