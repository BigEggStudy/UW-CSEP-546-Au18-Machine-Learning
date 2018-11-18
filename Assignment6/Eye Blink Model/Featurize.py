from PIL import Image

def ByIntensities(xTrainRaw, xTestRaw, step = 2):
    xTrain = []
    for sample in xTrainRaw:
        xTrain.append(by_intensities_core(sample, step))

    xTest = []
    for sample in xTestRaw:
        xTest.append(by_intensities_core(sample, step))

    return (xTrain, xTest)

def by_intensities_core(path, step = 2):
    (pixels, xSize, ySize) = get_pixels(path)

    features = []
    for x in range(0, xSize, step):
        for y in range(0, ySize, step):
            features.append(pixels[x,y]/255.0)

    return features

def get_pixels(path):
    image = Image.open(path)

    xSize = image.size[0]
    ySize = image.size[1]

    return (image.load(), xSize, ySize)
