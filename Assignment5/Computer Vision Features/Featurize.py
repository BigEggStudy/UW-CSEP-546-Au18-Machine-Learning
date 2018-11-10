from PIL import Image
import Assignment5Support

def CalculateGradientFeatures(gradients):
    grid_length = int(len(gradients) / 3)

    features = []
    for i in range(3):
        for j in range(3):
            min_value = 0
            max_value = 0
            average = 0
            for a in range(grid_length):
                for b in range(grid_length):
                    value = gradients[i * grid_length + a][j * grid_length + b]
                    min_value = min(value, min_value)
                    max_value = max(value, max_value)
                    average += value
            average /= grid_length * grid_length
            features.append(min_value)
            features.append(max_value)
            features.append(average)
    return features

def Featurize_Y_Gradient(x_raw):
    x = []
    for sample in x_raw:
        # y-gradient 9 grids of 8x8 pixels
        image = Image.open(sample)
        yGradients = Assignment5Support.Convolution3x3(image, [[1, 2, 1],[0, 0, 0],[-1, -2, -1]])
        yFeatures = CalculateGradientFeatures(yGradients)

        x.append(yFeatures)
    return x

def Featurize_X_Gradient(x_raw):
    x = []
    for sample in x_raw:
        # x-gradient 9 grids of 8x8 pixels
        image = Image.open(sample)
        xGradients = Assignment5Support.Convolution3x3(image, [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        xFeatures = CalculateGradientFeatures(xGradients)

        x.append(xFeatures)
    return x

def CalculateHistogramFeatures(gradients):
    image_length = int(len(gradients))
    features = [0, 0, 0, 0, 0]

    gradients_abs = []
    for gradients_row in gradients:
        for value in gradients_row :
            gradients_abs.append(abs(value))
    max_gradient = max(gradients_abs)
    min_gradient = min(gradients_abs)

    for value in gradients_abs:
        value = (value - min_gradient)/(max_gradient - min_gradient)

        if value < 0.2:
            features[0] += 1
        elif value >= 0.2 and value < 0.4:
            features[1] += 1
        elif value >= 0.4 and value < 0.6:
            features[2] += 1
        elif value >= 0.6 and value < 0.8:
            features[3] += 1
        else:
            features[4] += 1

    for i in range(5):
        features[i] /= image_length * image_length
    return features

def Featurize_Y_Histogram(x_raw):
    x = []
    for sample in x_raw:
        # y-gradient 9 grids of 8x8 pixels
        image = Image.open(sample)
        yGradients = Assignment5Support.Convolution3x3(image, [[1, 2, 1],[0, 0, 0],[-1, -2, -1]])
        yFeatures = CalculateHistogramFeatures(yGradients)

        x.append(yFeatures)
    return x

def Featurize_X_Histogram(x_raw):
    x = []
    for sample in x_raw:
        # x-gradient 9 grids of 8x8 pixels
        image = Image.open(sample)
        xGradients = Assignment5Support.Convolution3x3(image, [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        xFeatures = CalculateHistogramFeatures(xGradients)

        x.append(xFeatures)
    return x