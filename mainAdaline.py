
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from setting import config
from utils import load_data, preprocess_data, export_to_json
import warnings
import os
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from Adaline import Adaline
import random
import numpy as np
warnings.simplefilter(action='ignore', category=FutureWarning)

def preprocess_data(X, Y):
    """"
        This function preprocesses the input data by scaling the feature values and encoding the labels as integers.

    Inputs:
        X (pandas.DataFrame or numpy.ndarray): The feature values to be processed.
        y (pandas.Series or numpy.ndarray): The labels to be processed.

    Returns:
        X_scaled (numpy.ndarray): The scaled feature values.
        y_encoded (numpy.ndarray): The encoded labels.

    """
    # Scale the feature values using a StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Encode the labels as integers
    le = LabelEncoder()
    y_encoded = le.fit_transform(Y)

    return X_scaled, y_encoded


def main():
    path = config['output_result_path']
    chars = config['chars']
    value_to_remove = 3
    images_dictionary = config['images_dictionary']
    images_dictionary = np.delete(images_dictionary, value_to_remove-1)
    chars.remove(value_to_remove)
    precentLow = 0.08  # the low precent of the letter paint
    precentHigh = 0.8  # the high precent of the letter paint
    shuffle = True
    # # Load data
    X, Y = load_data(path, chars, precentLow, precentHigh, shuffle)
    test_size = [0.2]
    X, Y = preprocess_data(X, Y)
    index = 0
    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=test_size[index], shuffle=False)
    x_train_split = np.array_split(x_train, 5)
    y_train_split = np.array_split(y_train, 5)
    algo = Adaline(epochs=1, eta=0.01)
    sumResult = []
    for i in range(5):
        arr = [0, 0, 0, 0, 0]
        while np.sum(arr) < 5:
            rand = random.randint(0, 4)
            if arr[rand] < 1:
                arr[rand] += 1
                algo.fit(x_train_split[rand], y_train_split[rand])
        resultPredict = algo.predict(x_test, 1, 0)
        result1 = accuracy_score(y_test, resultPredict)
        sumResult.append(result1)
    print(f"chars classification: {images_dictionary}")
    print(f"sumResults= {sumResult}")
    print(f"Average= {np.sum(sumResult)/5}")
    # Calculate the standard deviation of the array
    sumResultStd = np.std(sumResult)
    print("Standard deviation of the sumResult array:", sumResultStd)


if __name__ == '__main__':
    main()
