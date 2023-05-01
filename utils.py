import pandas as pd
import json
import numpy as np
import random
import warnings
from sklearn.preprocessing import StandardScaler, LabelEncoder
from numpy.lib.function_base import vectorize
import ast
import random
from setting import config


warnings.simplefilter(action='ignore', category=FutureWarning)

def load_data(filename):
    text_file = open(filename, "r")
    lines = text_file.readlines()
    X=[]
    Y=[]
    TempList = []
    for i in range(len(lines)):
        TempList.append(list(ast.literal_eval(lines[i])))
    X = np.array(TempList)[:, 1:] 
    Y = np.array(TempList)[:, 0]
    print('X',X)
    print('Y',Y)
    return X, Y

def validateFileNames(file , validateValues):
    result=False
    for name in validateValues:
        if file.startswith(name):
            result=True
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