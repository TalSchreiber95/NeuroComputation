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


def check_valid_vector(vector, precentLow, precentHigh):
    counterNegative = 0
    vectorSize = len(vector)
    if vectorSize != 100:
        return True
    for i in range(vectorSize):
        if vector[i] == -1:
            counterNegative += 1
    if counterNegative/vectorSize < precentLow or counterNegative/vectorSize > precentHigh:
        return False
    else:
        return True


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


def much_lists_size_and_shuffle(X, Y):
   # combine the two lists into a list of tuples
    combined = list(zip(X, Y))
    # shuffle the combined list
    random.shuffle(combined)
    # unzip the shuffled list into separate X and Y lists
    X, Y = zip(*combined)
    # return the minimum index of list Y
    minimumInd = min(Y.count(x) for x in set(Y))
    return limit_vectors(X, Y, minimumInd)


def load_data(filename, chars, precentLow=0.5, precentHigh=0.8):
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
    return much_lists_size_and_shuffle(X, Y)


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
