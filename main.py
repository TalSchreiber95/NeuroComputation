
from setting import config
from utils import load_data, preprocess_data
import warnings
import os
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from Adaline import Adaline
warnings.simplefilter(action='ignore', category=FutureWarning)

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
