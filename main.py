
from setting import config
from utils import load_data
import warnings
import os
from glob import glob


warnings.simplefilter(action='ignore', category=FutureWarning)


def main():

    path = config['output_result_path']
 
    
    # # Load data
    df = load_data( path)
    print(df)

    # df = df.rename(columns={'label': 'class'})

    # # Split data into features and labels
    X = df.drop(columns=['class'])
    y = df['class']

    # # Preprocess data
    # X, y = preprocess_data(X, y)

    # # Optimization for results
    # test_size = [0.1, 0.2, 0.3]
    # results = []

    # # for size in test_size:
    # for idx in tqdm(range(0, len(test_size)), total=len(test_size),
    #                 desc=f"Run on: [0.1, 0.2, 0.3]"):
    #     # print(f'Run with test size of {test_size[idx]}\n')

    #     # Split data into training and test sets
    #     X_train, X_test, y_train, y_test = train_test_split(
    #         X, y, test_size=test_size[idx], shuffle=True)

    #     # Train model
    #     resultsTrainAndTest = trainAndEval(X_train, X_test, y_train,
    #                                        y_test, test_size[idx])

    #     results.append({test_size[idx]: [resultsTrainAndTest]})
    #     # print({test_size[idx]: [resultsTrainAndTest]})
    # path = f'{resultApksPath}/result.json'
    # # print('results', results)
    # export_to_json(results, path)


if __name__ == '__main__':
    main()
