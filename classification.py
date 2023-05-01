from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.svm import LinearSVC
from plt import plotting
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
import random
from utils import export_to_json
from tqdm import tqdm
from setting import config
warnings.simplefilter(action='ignore', category=FutureWarning)


def train_modelGradientBoostingClassifier(X_train, y_train, n_estimators, learning_rate, random_state_val):
    """
        This function trains a Gradient Boosting classifier model on the input training data.

    Inputs:
        X_train (numpy.ndarray): The feature values for the training data.
        y_train (numpy.ndarray): The labels for the training data.
        n_estimators (int): The number of decision trees in the ensemble.
        learning_rate (float): The learning rate of the algorithm.
        random_state_val (int): The seed used by the random number generator.

    Returns:
        model (sklearn.ensemble.GradientBoostingClassifier): The trained Gradient Boosting model.
    """

    # Initialize the model
    model = GradientBoostingClassifier(
        n_estimators=n_estimators, learning_rate=learning_rate, random_state=random_state_val)
    model.fit(X_train, y_train)
    return model


def train_modelDecisionTreeClassifier(X_train, y_train, max_depth, random_state_val):
    """
        This function trains a Decision Tree classifier model on the input training data.

    Inputs:
        X_train (numpy.ndarray): The feature values for the training data.
        y_train (numpy.ndarray): The labels for the training data.
        max_depth (int): The maximum depth of the tree.
        random_state_val (int): The seed used by the random number generator.

    Returns:
        model (sklearn.tree.DecisionTreeClassifier): The trained Decision Tree model.
    """

    # Initialize the model
    model = DecisionTreeClassifier(
        max_depth=max_depth, random_state=random_state_val)
    model.fit(X_train, y_train)
    return model


def train_modelKNeighborsClassifier(X_train, y_train, n_neighbors):
    """
        This function trains a KNN classifier model on the input training data.

    Inputs:
        X_train (numpy.ndarray): The feature values for the training data.
        y_train (numpy.ndarray): The labels for the training data.
        n_neighbors (int): The number of nearest neighbors used for the classification.

    Returns:
        model (sklearn.neighbors.KNeighborsClassifier): The trained KNN model.
    """

    # Initialize the model
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    return model


def train_modelLogisticRegression(X_train, y_train, C, epsilon, random_state_val):
    """
        This function trains a Sec-SVM classifier model on the input training data.

    Inputs:
        X_train (numpy.ndarray): The feature values for the training data.
        y_train (numpy.ndarray): The labels for the training data.
        C (float): The regularization parameter.
        epsilon (float): A small constant used to determine when to stop the training.

    Returns:
        model (sklearn.linear_model.LogisticRegression): The trained Sec-SVM model.
    """

    # Initialize the model
    model = LogisticRegression(
        C=C, penalty='l2', random_state=random_state_val, tol=epsilon)
    model.fit(X_train, y_train)
    return model


def train_modelLinearSVC(X_train, y_train, C, epsilon, random_state_val):
    """
        This function trains a Sec-SVM classifier model on the input training data.

    Inputs:
        X_train (numpy.ndarray): The feature values for the training data.
        y_train (numpy.ndarray): The labels for the training data.
        C (float): The regularization parameter.
        epsilon (float): A small constant used to determine when to stop the training.

    Returns:
        model (sklearn.svm.LinearSVC): The trained Sec-SVM model.
    """

    # Initialize the model
    model = LinearSVC(C=C, random_state=random_state_val, tol=epsilon)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, modelName):
    """
    This function evaluates a model on the test data.

    Inputs:
        model (sklearn.svm.LinearSVC): The trained model.
        X_test (numpy.ndarray): The feature values for the test data.
        y_test (numpy.ndarray): The labels for the test data.

    Returns:
        accuracy (float): The accuracy score of the model.
        precision (float): The precision score of the model.
        recall (float): The recall score of the model.
    """
    # Predict the labels on the test data using the model
    y_pred = model.predict(X_test)

    # Compute the accuracy, precision, and recall scores
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    # plotting(y_pred[:100], y_test[:100], modelName)

    return accuracy, precision, recall


def classify_apk(model, apk_features):
    """
        This function makes a prediction for a new APK by using the trained model to classify the APK based on its features.

    Inputs:
        model (sklearn.svm.LinearSVC): The trained model.
        apk_features (numpy.ndarray): The feature values for the new APK.

    Returns:
        label (int): The predicted label for the new APK.
    """
    return model.predict(apk_features)


def runAndGetResultForOverFittingGradientBoostingClassifier(X_train, X_test, y_train, y_test, results):
    results['modelGradientBoostingClassifier'] = []
    result = []
    # maxRun = 10
    """
        learning_rate is a hyperparameter used in Gradient Boosting Algorithm that determines the step size at which the algorithm learns
        from the errors made in previous iterations. A smaller learning rate means the model will require more trees to learn the data,
        but will result in smaller steps and a more accurate model. A higher learning rate means the model will require fewer trees to
        learn the data, but will result in larger steps and a less accurate model.

        n_estimators is another hyperparameter in Gradient Boosting Algorithm that represents the number of trees in the model.
        The higher the number of trees, the more complex the model will be. However, if the number of trees is too high,
        it can lead to overfitting.

        Choosing the right values for these hyperparameters is a trade-off between model complexity and overfitting.
        A good way to choose the values for these parameters is to use a technique called grid search,
        which involves training the model with different combinations of the hyperparameters and evaluating the model's performance
        on a validation set. The combination of hyperparameters that result in the best performance on the validation set are chosen
        as the final values for the model.
    """
    learning_rate = 0.07900590739680002
    n_estimators = 93
    random_state_val = 0
    maxRun = 50

    for run in range(1, maxRun):
        learning_rate = random.uniform(0, 1)
        n_estimators = random.randint(1, 100)

        modelGradientBoostingClassifierTrainAndTest = train_modelGradientBoostingClassifier(
            X_train, y_train, n_estimators=n_estimators, learning_rate=learning_rate, random_state_val=random_state_val)
        accuracy1, precision1, recall1 = evaluate_model(
            modelGradientBoostingClassifierTrainAndTest, X_test, y_test, 'Gradient Boosting Classifier')

        modelGradientBoostingClassifierTrain = train_modelGradientBoostingClassifier(
            X_train, y_train, n_estimators=n_estimators, learning_rate=learning_rate, random_state_val=random_state_val)
        accuracy2, precision2, recall2 = evaluate_model(
            modelGradientBoostingClassifierTrain, X_train, y_train, 'Gradient Boosting Classifier')

        result = {
            "modelGradientBoostingClassifierTrainAndTest": {
                'accuracy': accuracy1,
                'precision': precision1,
                'recall': recall1,
                'n_estimators': n_estimators,
                'learning_rate': learning_rate,
                'random_state_val': random_state_val
            },
            "modelGradientBoostingClassifierTrain": {
                'accuracy': accuracy2,
                'precision': precision2,
                'recall': recall2,
                'n_estimators': n_estimators,
                'learning_rate': learning_rate,
                'random_state_val': random_state_val
            }
        }
        results['modelGradientBoostingClassifier'].append(result)
    return results, 'GradientBoostingClassifier'


def runAndGetResultForOverFittingDecisionTreeClassifier(X_train, X_test, y_train, y_test, results):
    results['modelDecisionTreeClassifier'] = []
    result = []
    """
        max_depth is a parameter in the DecisionTreeClassifier function that determines the maximum depth of the tree.
        The maximum depth is the number of levels in the tree from the root node to the farthest leaf node.
        A larger value for max_depth will lead to a more complex and potentially overfitting model,
        while a smaller value will lead to a simpler and potentially underfitting model.

        When choosing the value for max_depth, it's important to consider the trade-off between model complexity and accuracy.
        If the data is highly complex, a larger value for max_depth may be necessary to capture the complexity of the data. However,
        if the data is relatively simple, a smaller value for max_depth may be sufficient. In general,
        it's a good idea to start with a smaller value and gradually increase it,
        while monitoring the model's performance on the validation set.
        Additionally, using techniques like cross-validation can help to determine the optimal value for max_depth.
    """
    max_depth = 50
    random_state_val = 0

    for maxDepth in range(1, max_depth):

        modelDecisionTreeClassifier = train_modelDecisionTreeClassifier(
            X_train, y_train, max_depth=maxDepth, random_state_val=random_state_val)
        accuracy1, precision1, recall1 = evaluate_model(
            modelDecisionTreeClassifier, X_test, y_test, 'Decision Tree Classifier')

        modelDecisionTreeClassifier = train_modelDecisionTreeClassifier(
            X_train, y_train, max_depth=maxDepth, random_state_val=random_state_val)
        accuracy2, precision2, recall2 = evaluate_model(
            modelDecisionTreeClassifier, X_train, y_train, 'Decision Tree Classifier')

        result = {
            "modelDecisionTreeClassifierTrainAndTest": {
                'accuracy': accuracy1,
                'precision': precision1,
                'recall': recall1,
                'max_depth': max_depth,
                'random_state_val': random_state_val
            },
            "modelDecisionTreeClassifierTrain": {
                'accuracy': accuracy2,
                'precision': precision2,
                'recall': recall2,
                'max_depth': max_depth,
                'random_state_val': random_state_val
            }
        }
        results['modelDecisionTreeClassifier'].append(result)
    return results, 'DecisionTreeClassifier'


def runAndGetResultForOverFittingKNeighborsClassifier(X_train, X_test, y_train, y_test, results):
    results['modelKNeighborsClassifier'] = []
    """
        n_neighbors is a parameter in the KNeighborsClassifier function that specifies the number of nearest neighbors to consider
        when making a prediction for a new data point. It is an integer value, and the default value is 5.

        To choose the value of n_neighbors, one approach is to try different values and evaluate the performance of
        the model using a validation set. A smaller value of n_neighbors will result in a more complex model,
        as it will consider more nearest neighbors when making predictions, while a larger value will result in a
        simpler model that considers fewer neighbors. A good rule of thumb is to choose a value that gives the best performance
        on the validation set. Additionally, it's important to keep in mind that a large value of n_neighbors may lead to a high
        bias and a low variance in the model, while a small value may lead to a high variance and a low bias.
    """
    # n_neighbors = 5

    max_depth = 50
    for n_neighbors in range(1, max_depth):
        if n_neighbors % 2 != 0:
            modelKNeighborsClassifierTrainAndTest = train_modelKNeighborsClassifier(
                X_train, y_train, n_neighbors=n_neighbors)
            accuracy1, precision1, recall1 = evaluate_model(
                modelKNeighborsClassifierTrainAndTest, X_test, y_test, 'K Neighbors Classifier')

            modelKNeighborsClassifierTrain = train_modelKNeighborsClassifier(
                X_train, y_train, n_neighbors=n_neighbors)
            accuracy2, precision2, recall2 = evaluate_model(
                modelKNeighborsClassifierTrain, X_train, y_train, 'K Neighbors Classifier')

            result = {
                "modelKNeighborsClassifierTrainAndTest": {
                    'accuracy': accuracy1,
                    'precision': precision1,
                    'recall': recall1,
                    'n_neighbors': n_neighbors,
                },
                "modelKNeighborsClassifierTrain": {
                    'accuracy': accuracy2,
                    'precision': precision2,
                    'recall': recall2,
                    'n_neighbors': n_neighbors,
                }
            }
            results['modelKNeighborsClassifier'].append(result)
    return results, 'KNeighborsClassifier'


def runAndGetResultForOverFittingLogisticRegression(X_train, X_test, y_train, y_test, results):
    results['modelLogisticRegression'] = []
    """
        C is the inverse of regularization strength; smaller values specify stronger regularization.
        It is used to prevent overfitting by penalizing large weights. Common choices for C are 0.1, 1, and 10.

        epsilon is the tolerance for stopping criteria. A small value of epsilon will require a more accurate solution,
        while a larger value will be computationally faster but less accurate.

        random_state_val is used to set a seed for the random number generator, which is used to initialize the weights of the model.
        It ensures that the same random weights are generated every time the model is run,
        which can be useful for debugging or reproducibility. The value for this parameter is typically set to a random integer.
    """
    epsilon_val = 1e-3
    max_run = 50
    random_state_val = 1
    c_val = 0.7462138628

    for random_state_val in range(1, max_run):
        c_val = round(random.uniform(0.1, 1.0), 10)
        modelLogisticRegressionTrainAndTest = train_modelLogisticRegression(X_train, y_train, C=c_val,
                                                                            epsilon=epsilon_val, random_state_val=random_state_val)
        accuracy1, precision1, recall1 = evaluate_model(
            modelLogisticRegressionTrainAndTest, X_test, y_test, 'Logistic Regression')

        modelLogisticRegressionTrain = train_modelLogisticRegression(X_train, y_train, C=c_val,
                                                                     epsilon=epsilon_val, random_state_val=random_state_val)
        accuracy2, precision2, recall2 = evaluate_model(
            modelLogisticRegressionTrain, X_train, y_train, 'Logistic Regression')

        result = {
            "modelLogisticRegressionTrainAndTest": {
                'accuracy': accuracy1,
                'precision': precision1,
                'recall': recall1,
                'C': c_val,
                'epsilon': epsilon_val,
                'random_state_val': random_state_val,
            },
            "modelLogisticRegressionTrain": {
                'accuracy': accuracy2,
                'precision': precision2,
                'recall': recall2,
                'C': c_val,
                'epsilon': epsilon_val,
                'random_state_val': random_state_val,
            }
        }
        results['modelLogisticRegression'].append(result)
    return results, 'LogisticRegression'


def runAndGetResultForOverFittingLinearSVC(X_train, X_test, y_train, y_test, results):
    results['modelLinearSVC'] = []
    """
       In the LinearSVC function, C is a regularization parameter that controls the trade-off between maximizing the margin
       and minimizing the loss function. It is the inverse of the regularization strength, where a smaller value of C corresponds
       to a stronger regularization. The epsilon parameter is used to specify the tolerance for the stopping criterion.
       The random_state_val parameter is used to set a random seed for the algorithm.

        When choosing the values for C, epsilon, and random_state_val, it is important to consider the specific characteristics
        of your data and the task at hand. A good approach is to start with a relatively small value of C, and then increase it
        gradually to see how the performance of the model changes. The same goes for epsilon, where a smaller value will make the
        algorithm more sensitive to the stopping criterion. The random_state_val parameter can be set to any integer value,
        it is used to ensure that the results are reproducible.

        It is also important to consider the balance between underfitting and overfitting when choosing these parameters.
        In general, a smaller value of C and epsilon may lead to underfitting, while a larger value may lead to overfitting.
        One way to choose the optimal values for these parameters is by using techniques such as cross-validation and grid search.
    """
    epsilon_val = 1e-3
    max_run = 50
    c_val = 0.5410996905
    random_state_val = 0
    for run in range(0, max_run):
        c_val = round(random.uniform(0.1, 1.0), 10)  # 1.0

        modelLinearSVCTrainAndTest = train_modelLinearSVC(
            X_train, y_train, C=c_val, epsilon=epsilon_val, random_state_val=random_state_val)
        accuracy1, precision1, recall1 = evaluate_model(
            modelLinearSVCTrainAndTest, X_test, y_test, 'Linear SVC')

        modelLinearSVCTrain = train_modelLinearSVC(
            X_train, y_train, C=c_val, epsilon=epsilon_val, random_state_val=random_state_val)
        accuracy2, precision2, recall2 = evaluate_model(
            modelLinearSVCTrain, X_train, y_train, 'Linear SVC')

        result = {
            "modelLinearSVCTrainAndTest": {
                'accuracy': accuracy1,
                'precision': precision1,
                'recall': recall1,
                'C': c_val,
                'epsilon': epsilon_val,
                'random_state_val': random_state_val,
            },
            "modelLinearSVCTrain": {
                'accuracy': accuracy2,
                'precision': precision2,
                'recall': recall2,
                'C': c_val,
                'epsilon': epsilon_val,
                'random_state_val': random_state_val,
            }
        }
        results['modelLinearSVC'].append(result)
    return results, 'LinearSVC'


def trainAndEval(X_train, X_test, y_train, y_test, size):
    results = {}
    functions = [
        runAndGetResultForOverFittingGradientBoostingClassifier,
        runAndGetResultForOverFittingKNeighborsClassifier,
        runAndGetResultForOverFittingLogisticRegression,
        runAndGetResultForOverFittingDecisionTreeClassifier,
        runAndGetResultForOverFittingLinearSVC
    ]
    path = config['resultApksPath']
    algorithmsLength = len(functions)
    for idx in tqdm(range(0, algorithmsLength), total=algorithmsLength,
                    desc=f'Run of ML Algorithms...'):
        algoName = functions[idx].__qualname__
        algoName = algoName.replace('runAndGetResultForOverFitting', '')
        print(f'Run on {algoName}')
        res, algoName = functions[idx](
            X_train, X_test, y_train, y_test, {})
        # print('res', res)
        results[algoName] = res
    return results
