
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
    outputPath = f'{config["output_path"]}/output.txt'
    chars = config['chars']

    value_to_remove = 1
    chars.remove(value_to_remove)

    precentLow = 0.08  # the low precent of the letter paint
    precentHigh = 0.8  # the high precent of the letter paint
    shuffle = True
    # # Load data
    X, Y = load_data(path, chars, precentLow, precentHigh, shuffle)

    # print('X', len(X))
    # print('Y', len(Y))
    # print('Y', Y)
    # print('minY', min(Y.count(x) for x in set(Y)))
    # print('maxY', max(Y.count(x) for x in set(Y)))
    test_size = [0.2]
    X, Y = preprocess_data(X, Y)
    maxRun = 60
    result = []
    index = 0
    for idx in tqdm(range(0, len(test_size)), total=len(test_size),
                    desc=f"Run on: {test_size[index]}"):
        for run in range(10, maxRun):
            for run2 in range(1, maxRun):
                print(f'run number {run}')
                epochs=run
                eta = random.uniform(0, 0.01)

                data = {
                    'epochs': epochs,
                    'eta': eta,
                    'accuracy': 0,
                }
                # print(f'data {data}')

                X_train, X_test, y_train, y_test = train_test_split(
                    X, Y, test_size=test_size[index], shuffle=False)

                algo = Adaline(epochs=epochs, eta=eta)

                algo.fit(X_train, y_train)

                resultPredict = algo.predict(X_test, 0, 1)

                # print('result', resultPredict)
                # print('y_test', y_test)

                data['accuracy'] = accuracy_score(y_test, resultPredict)
                result.append(data)
        index += 1
        print("result:", result)
        export_to_json(result, outputPath)



if __name__ == '__main__':
    main()
