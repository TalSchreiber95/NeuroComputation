import pandas as pd
import json
import random
import warnings


warnings.simplefilter(action='ignore', category=FutureWarning)

def validateFileNames(file , validateValues):
    result=False
    for name in validateValues:
        if file.startwith(name):
            result=True
    return result

def is_2d(value):
    if isinstance(value, list) and all(isinstance(i, list) for i in value):
        return True
    return False


def count_apps(app_list):
    malicious_count = 0
    benign_count = 0
    for item in app_list:
        if int(item['label']) == 1:
            malicious_count += 1
        else:
            benign_count += 1
    return malicious_count, benign_count


def get_random_number(start, end, isInt=True):
    rand = random.randrange(start, end+1)
    randU = random.uniform(start, end)
    if isInt:
        while rand == 0:
            rand = random.randrange(start, end+1)
        return rand
    while randU == 0:
        randU = random.uniform(start, end)
    return randU


def filter_apps(num_malicious, num_benign, app_list):
    malicious_apps = [app for app in app_list if app["label"] == 1]
    benign_apps = [app for app in app_list if app["label"] == 0]

    filtered_malicious_apps = random.sample(malicious_apps, num_malicious)
    filtered_benign_apps = random.sample(benign_apps, num_benign)

    filtered_apps = filtered_malicious_apps + filtered_benign_apps
    random.shuffle(filtered_apps)
    return filtered_apps


def removePropertyFromJson(property, json):
    for i in range(len(json)):
        del json[i][property]

    return json


def load_data(filename):
    """
    This function loads data from a JSON file and returns a Pandas DataFrame. If the DataFrame contains any NaN values, they are replaced with 0.

    Inputs:
        filename (str): The filepath of the JSON file to be loaded.

    Returns:
        result (pandas.DataFrame): The resulting DataFrame containing the data from the JSON file.
    """
    with open(filename, 'r') as f:
        data = json.load(f)
    data = removePropertyFromJson('sha256', data)

    # malicious_count, benign_count = count_apps(data)
    # print(f'malicious_apps_size in json file = {malicious_count}')
    # print(f'benign_apps_size in json file = {benign_count}')

    # benign_count = 1000
    # malicious_count = 1000
    # data = filter_apps(malicious_count, malicious_count, data)
    # print(f'malicious_apps_size selected by 0.1 = {malicious_count}')
    # print(f'benign_apps_size selected by 0.9 = {malicious_count}')
    # print(f'total application = {malicious_count * 2}')
    malicious_count, benign_count = count_apps(data)
    result = pd.DataFrame(data)
    result = result.fillna(0)
    return result, malicious_count, benign_count


def sort_by_malicious(result):
    return {k: v for k, v in sorted(result.items(), key=lambda item: item[1]['malicious'], reverse=True)}


def exportToTextFile(lst, text_file_path) -> bool:
    # open the text file in write mode
    with open(text_file_path, 'w') as f:
        # iterate through the feature list and write each feature to a new line in the text file
        for val in lst:
            f.write(val + '\n')
    print(f'Successfully exported lst to {text_file_path}')
    return True


def export_to_json(result, file_name):
    result_json = json.dumps(result, indent=4)
    with open(file_name, 'w') as f:
        f.write(result_json)
    print(f'Successfully exported to {file_name}')



def read_from_results_models_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
        return data


def get_max_accuracies(modelName, size, data):
    max_accuracies = []
    keyTrain = f'model{modelName}TrainAndTest'
    keyTest = f'model{modelName}Train'
    for item in data[modelName][size]['values'][keyTrain]:
        max_accuracies.append(
            item[keyTrain]['accuracy'])
        max_accuracies.append(item[keyTest]['accuracy'])
    return sorted(max_accuracies, reverse=True)[:3]


def isOverFitting(TrainAndTestResult, TrainResult, algoName):
    epsilon = 0.05
    # if algoName == 'LinearSVC':
    #     epsilon = 0.23
    # elif algoName == 'GradientBoostingClassifier':
    #     epsilon = 0.000001
    # elif algoName == 'KNeighborsClassifier':
    #     epsilon = 0.0008
    # elif algoName == 'LogisticRegression':
    #     epsilon = 0.0007
    # elif algoName == 'DecisionTreeClassifier':
    #     epsilon = 0.0001
    # else:
    #     epsilon = 0.03

    # accuracy10Percent = TrainAndTestResult['accuracy'] * 0.1
    # recall10Percent = TrainAndTestResult['recall'] * 0.1
    # precision10Percent = TrainAndTestResult['precision'] * 0.1

    accuracy = abs(TrainAndTestResult['accuracy'] - TrainResult['accuracy'])
    recall = abs(TrainAndTestResult['recall'] - TrainResult['recall'])
    precision = abs(TrainAndTestResult['precision'] - TrainResult['precision'])
    return accuracy > epsilon


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
