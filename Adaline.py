import numpy as np

class Adaline:
    def __init__(self, eta=0.01, epochs=50):
        self.eta = eta  # learning rate
        self.epochs = epochs  # number of iterations
        self.weights = None  # model weights

    def fit(self, X, y):
        # initialize weights to small random values
        self.weights = np.random.uniform(-0.01, 0.01, X.shape[1] + 1)

        for _ in range(self.epochs):
            # calculate net input
            net_input = self.net_input(X)

            # calculate errors
            errors = y - net_input

            # update weights
            self.weights[1:] += self.eta * X.T.dot(errors)
            self.weights[0] += self.eta * errors.sum()

    def net_input(self, X):
        return np.dot(X, self.weights[1:]) + self.weights[0]

    def predict(self, X, char1=-1, char2=1):
        return np.where(self.net_input(X) >= 0.0, char1, char2)