import collections

class MostCommonModel(object):
    """A model that predicts the most common label from the training data."""

    def __init__(self):
        pass

    def fit(self, x, y):
        count = collections.Counter()

        for label in y:
            count[label] += 1

        self.prediction = count.most_common(1)[0][0]

        print(self.predict)

    def predict(self, x):
        return [self.prediction for example in x]
