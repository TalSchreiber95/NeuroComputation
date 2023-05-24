import pandas as pd
import json
import numpy as np
import random
import warnings
from sklearn.preprocessing import StandardScaler, LabelEncoder
import ast
import random
from setting import config
from collections import defaultdict


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


def check_valid_vector(vector, precentLow, precentHigh):
    counterPositive = 0
    vectorSize = len(vector)
    if vectorSize != 100:
        return False
    for i in range(vectorSize):
        if vector[i] == 1:
            counterPositive += 1
    # print(
        # f"counterNegative= {vectorSize-counterPositive}, counterPositive= {counterPositive}")
    if counterPositive/vectorSize > precentLow and counterPositive/vectorSize < precentHigh:
        return True
    else:
        return False


def limit_vectors(X, Y, maximumInd):
    assert len(X) == len(Y), "X and Y must have the same length"
    counter = defaultdict(int)
    X_limited = []
    Y_limited = []
    for x, y in zip(X, Y):
        if counter[y] < maximumInd:
            X_limited.append(x)
            Y_limited.append(y)
            counter[y] += 1
    return X_limited, Y_limited


def shuffle_lists(X, Y):
    # combine the two lists into a list of tuples
    combined = list(zip(X, Y))
    # shuffle the combined list
    random.shuffle(combined)
    # unzip the shuffled list into separate X and Y lists
    return zip(*combined)


def export_to_json(result, file_name):
    result_json = json.dumps(result, indent=4)
    with open(file_name, 'w') as f:
        f.write(result_json)
    print(f'Successfully exported to {file_name}')


def read_information_from_result_from_models(files, resultModelsPath, resultApksPath):
    json_files = {}
    for file in files:
        nameKeys = file.replace(
            resultModelsPath, '').replace('.json', '').replace('/', '').split('-')
        json_files[nameKeys[0]] = {}

    for file in files:
        nameKeys = file.replace(
            resultModelsPath, '').replace('.json', '').replace('/', '').split('-')
        json_files[nameKeys[0]][nameKeys[1]] = {}
        data = read_from_results_models_json(file)
        json_files[nameKeys[0]][nameKeys[1]] = {
            'filePath': file,
            'values': data,
        }
    maxValues = {}
    algorithms = json_files.keys()
    for algo in algorithms:
        json_formatted_str = json.dumps(json_files[algo], indent=2)
        maxValue = getMaxValues(json_files[algo], algo)
        maxValues[algo] = maxValue

    path = f'{resultApksPath}/bestValuesForAlgorithms.json'
    # print('path', path)
    export_to_json(maxValues, path)

def getMaxValues(data, algoName):
    keys = data.keys()
    maxValue = {
        'size': 0,
        'recall': 0,
        'precision': 0,
        'accuracy': 0,
    }
    for size in keys:
        keyAlgo = f'model{algoName}'
        length = range(len(data[size]['values'][keyAlgo]))
        for idx in length:
            value = data[size]['values'][keyAlgo][idx]
            keyTrainAndTest = f'{keyAlgo}TrainAndTest'
            keyTrain = f'{keyAlgo}Train'
            TrainAndTestResult = value[keyTrainAndTest]
            onlyTrainResult = value[keyTrain]
            # print('TrainAndTestResult', TrainAndTestResult)
            # print('onlyTrainResult', onlyTrainResult)
            if isOverFitting(TrainAndTestResult, onlyTrainResult, algoName) == False:
                if maxValue['accuracy'] < TrainAndTestResult['accuracy']:
                    maxValue = TrainAndTestResult
                    maxValue['size'] = size
                    maxValue['algoName'] = algoName

    return maxValue


def isOverFitting(TrainAndTestResult, TrainResult, algoName):
    epsilon = 0.05

    accuracy = abs(TrainAndTestResult['accuracy'] - TrainResult['accuracy'])
    recall = abs(TrainAndTestResult['recall'] - TrainResult['recall'])
    precision = abs(TrainAndTestResult['precision'] - TrainResult['precision'])
    return accuracy > epsilon


def much_lists_size_and_shuffle(X, Y, shuffle):
    if shuffle:
        X, Y = shuffle_lists(X, Y)
    # return the minimum index of list Y
    minimumInd = min(Y.count(x) for x in set(Y))
    return limit_vectors(X, Y, minimumInd)


def load_data(filename, chars, precentLow=0.5, precentHigh=0.8, shuffle=True):
    X = []
    Y = []
    TempList = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            try:
                TempList.append(list(ast.literal_eval(lines[i])))
            except ValueError:
                print(f"Could not parse line {i+1} in {filename}: {lines[i]}")
        for i in range(len(TempList)):
            if TempList[i][0] in chars:
                if check_valid_vector(TempList[i][1:len(TempList[i])], precentLow, precentHigh):
                    X.append(TempList[i][1:len(TempList[i])])
                    Y.append(TempList[i][0])
    return much_lists_size_and_shuffle(X, Y, shuffle)


def validateFileNames(file, validateValues):
    result = False
    for name in validateValues:
        if file.startswith(name):
            result = True
    return result


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


def randomLines():
    path = config['output_result_path']

    # specify input and output file names
    input_file = 'input.txt'
    output_file = 'output.txt'

    # read in the rows from the input file
    with open(path, 'r') as f:
        rows = f.readlines()

    # shuffle the rows in a random order
    random.shuffle(rows)

    # write the shuffled rows to the output file
    with open(output_file, 'w') as f:
        for row in rows:
            f.write(row)
