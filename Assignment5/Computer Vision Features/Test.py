from PIL import Image
import Assignment5Support
import Featurize

image = Image.open('../dataset_B_Eye_Images/openRightEyes/Oscar_DLeon_0001_R.jpg')

# y-gradient 9 grids of 8x8 pixels
yGradients = Assignment5Support.Convolution3x3(image, [[1, 2, 1],[0, 0, 0],[-1, -2, -1]])
yFeatures = Featurize.CalculateGradientFeatures(yGradients)
print(yFeatures[:12])

# x-gradient 9 grids of 8x8 pixels
xGradients = Assignment5Support.Convolution3x3(image, [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
xFeatures = Featurize.CalculateGradientFeatures(xGradients)
print(xFeatures[:12])

# y-graident 5-bin histogram
yFeatures = Featurize.CalculateHistogramFeatures(yGradients)
print(yFeatures[:5])

# # x-gradient 5-bin histogram
xFeatures = Featurize.CalculateHistogramFeatures(xGradients)
print(xFeatures[:5])

# result:
# [-59, 63, -6.59375, -92, 78]
# [-125, 249, 13.3125, -196, 326]
# [0.6545138888888888, 0.2482638888888889, 0.06597222222222222, 0.024305555555555556, 0.006944444444444444]
# [0.5868055555555556, 0.234375, 0.11458333333333333, 0.046875, 0.017361111111111112]