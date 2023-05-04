import numpy as np
class Adaline:
    def __init__(self, learning_rate=0.01, max_epochs=100):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs

    def fit(self, X, y):
        self.weights = np.random.rand(X.shape[1] + 1)
        X = np.insert(X, 0, 1, axis=1) # add bias
        for epoch in range(self.max_epochs):
            output = self.activation(np.dot(X, self.weights))
            error = y - output
            self.weights += self.learning_rate * np.dot(X.T, error)
            if np.sum(error**2) < 1e-4:
                break

    def predict(self, X, char1, char2):
        X = np.insert(X, 0, 1, axis=1) # add bias
        return np.where(self.activation(np.dot(X, self.weights)) > 0, char1, char2)

    def activation(self, x):
        return x
