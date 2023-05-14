import numpy as np

class Adaline:
    def __init__(self, eta=0.01, epochs=50):
        self.eta = eta  # learning rate
        self.epochs = epochs  # number of iterations
        self.weights = np.random.uniform(-0.01, 0.01,101)  # initialize weights to small random values

    def fit(self, X, y):
        for _ in range(self.epochs):
            # calculate net input
            net_input = self.net_input(X)
            # calculate errors
            errors = y - net_input
            # update weights
            self.weights[1:] += self.eta * X.T.dot(errors * net_input * (1 - net_input))
            self.weights[0] += self.eta * errors.sum()

    def net_input(self, X):
        # apply sigmoid activation to the dot product between X and the weight matrix
        return 1.0 / (1.0 + np.exp(-(np.dot(X, self.weights[1:]) + self.weights[0])))

    def predict(self, X, char1=0, char2=1):
        # use a threshold of 0.5 to predict the class labels
        return np.where(self.net_input(X) >= 0.5, char1, char2)
