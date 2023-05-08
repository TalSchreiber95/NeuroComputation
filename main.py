
from setting import config
from utils import load_data, preprocess_data
import warnings
import os
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from Adaline import Adaline
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.preprocessing import StandardScaler, LabelEncoder



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
    chars.remove(value_to_remove)

    # # Load data
    X,Y = load_data(path, chars)


    test_size = [0.2]
    X, Y = preprocess_data(X, Y)
    print('X',X)
    print('Y',Y)
    
    for size in test_size:
        for idx in tqdm(range(0, len(test_size)), total=len(test_size),
                        desc=f"Run on: [0.2]"):
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, Y, test_size=test_size[idx], shuffle=True)
            
            algo = Adaline()

            algo.fit(X_train,y_train)
            result = algo.predict(X_test,chars[0],chars[1])

            print('result',result)
            print('y_test',y_test)


if __name__ == '__main__':
    main()
