## Some of this references my answers to previous assignments.
##  Replace my references with references to your answers to those assignments.

## IMPORTANT NOTE !!
## Remember to install the Pillow library (which is required to execute 'import PIL')

import Assignment5Support
import matplotlib.pyplot as plt
import numpy as np

if __name__=="__main__":
    kDataPath = "..\\dataset_B_Eye_Images"

    (xRaw, yRaw) = Assignment5Support.LoadRawData(kDataPath, includeLeftEye = True, includeRightEye = True)

    (xTrainRaw, yTrainRaw, xTestRaw, yTestRaw) = Assignment5Support.TrainTestSplit(xRaw, yRaw, percentTest = .25)

    print("Train is %f percent closed." % (sum(yTrainRaw)/len(yTrainRaw)))
    print("Test is %f percent closed." % (sum(yTestRaw)/len(yTestRaw)))

    print("Calculating features...")
    (xTrain, xTest) = Assignment5Support.Featurize(xTrainRaw, xTestRaw, includeGradients=True, includeRawPixels=False, includeIntensities=False)
    yTrain = yTrainRaw
    yTest = yTestRaw

    ############################

    import KMeansCluster
    print("========== Use K-Means Cluster ==========")
    k = 4
    print("Let K = %d" % k)
    cluster = KMeansCluster.KMeansCluster()

    (centroids, assignments, path) = cluster.find_clusters(xTrain, k, 10)
    colmap = ['r', 'g', 'b', 'm']

    fig = plt.figure(figsize=(5, 5))
    ax = plt.axes()

    for i in range(k):
        print("For centroid %d, it position is %f, %f" % (i, centroids[i][0], centroids[i][1]))
        samples = [ sample for (sample, index) in assignments[i] ]
        samples = np.array(samples)
        samples_x = samples[:,0]
        samples_y = samples[:,1]
        plt.scatter(samples_x, samples_y, color=colmap[i], alpha=0.25, edgecolor='k')

        nearest_samples = sorted(assignments[i], key=lambda data: cluster.euclidean_distance(data[0], centroids[i]))
        print("The Closest Sample to the Cluster center is:")
        for j in range(10):
            print("The file name is %s:" % xTrainRaw[nearest_samples[j][1]])

    print("")
    print("### Plot Clusters")
    old_centroids = path[0]
    for i in range(k):
        plt.scatter(*old_centroids[i], color=colmap[i], alpha=0.5, edgecolor='k')

    for i in range(1, len(path)):
        for j in range(k):
            plt.scatter(*path[i][j], color=colmap[j], alpha=0.5, edgecolor='k')
            old_x = old_centroids[j][0]
            old_y = old_centroids[j][1]
            dx = (path[i][j][0] - old_centroids[j][0]) * 0.85
            dy = (path[i][j][1] - old_centroids[j][1]) * 0.85
            ax.arrow(old_x, old_y, dx, dy, head_width=0.1, head_length=0.1, fc=colmap[j], ec=colmap[j])

        old_centroids = path[i]

    print("Close the plot diagram to continue program")
    plt.show()