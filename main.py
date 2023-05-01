from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split
from setting import config
from utils import get_random_number, load_data, count_apps, printResults, export_to_json, read_information_from_result_from_models
from classification_utils import preprocess_data, print_best_model
from classification import train_modelGradientBoostingClassifier, evaluate_model,\
    train_modelDecisionTreeClassifier, train_modelKNeighborsClassifier,\
    train_modelLogisticRegression, train_modelLinearSVC, trainAndEval
import warnings
import os
from glob import glob
from tqdm import tqdm
import json

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=ConvergenceWarning)

# need to
global c_val, epsilon_val, test_size_val, random_state_val
global c_val_max, epsilon_val_max, random_state_val_max
global accuracy_max, precision_max, recall_max
global num_benign_apps, num_malicious_apps


def main():
    """"
     This is the main function that executes the entire process of loading the data, preprocessing it,
     training a model, and evaluating its performance. It also allows for the classification of a new APK.

    Inputs:
        None.
    Returns:
        None.
    """

    path = config['apksResultJsonPath']
 
    resultApksPath = config['resultApksPath']
    resultModelsPath = f'{resultApksPath}/resultModels'
    # # Load data
    df, malicious_count, benign_count = load_data(
        path)

    df = df.rename(columns={'label': 'class'})

    # Split data into features and labels
    X = df.drop(columns=['class'])
    y = df['class']

    # Preprocess data
    X, y = preprocess_data(X, y)

    # Optimization for results
    test_size = [0.1, 0.2, 0.3]
    results = []

    # for size in test_size:
    for idx in tqdm(range(0, len(test_size)), total=len(test_size),
                    desc=f"Run on: [0.1, 0.2, 0.3]"):
        # print(f'Run with test size of {test_size[idx]}\n')

        # Split data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size[idx], shuffle=True)

        # Train model
        resultsTrainAndTest = trainAndEval(X_train, X_test, y_train,
                                           y_test, test_size[idx])

        results.append({test_size[idx]: [resultsTrainAndTest]})
        # print({test_size[idx]: [resultsTrainAndTest]})
    path = f'{resultApksPath}/result.json'
    # print('results', results)
    export_to_json(results, path)


if __name__ == '__main__':
    main()
