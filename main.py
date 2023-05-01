
from setting import config
from utils import load_data, preprocess_data
import warnings
import os
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import numpy as np


warnings.simplefilter(action='ignore', category=FutureWarning)
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

def main():

    path = config['output_result_path']
    # # Load data
    X,Y = load_data(path)
    # # Optimization for results
    test_size = [0.2]
    results = []

    # for size in test_size:
    for idx in tqdm(range(0, len(test_size)), total=len(test_size),
                    desc=f"Run on: [0.2]"):
        # print(f'Run with test size of {test_size[idx]}\n')

        # Split data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, Y, test_size=test_size[idx], shuffle=True)
        algo = Adaline()
        algo.fit(X_train,y_train)
        result = algo.predict(X_test, 1,3)
        print('result',result)
        print('y_test',y_test)


if __name__ == '__main__':
    main()
