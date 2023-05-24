import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from setting import config
from utils import load_data, preprocess_data, export_to_json, preprocess_data
import random
from sklearn.metrics import accuracy_score

def main():
    # Load the data generated in Part I
    path = config['output_result_path']
    chars = config['chars']
    value_to_remove = 1
    images_dictionary = config['images_dictionary']
    images_dictionary = np.delete(images_dictionary, value_to_remove-1)
    chars.remove(value_to_remove)
    precentLow = 0.08  # the low precent of the letter paint
    precentHigh = 0.8  # the high precent of the letter paint
    shuffle = True
    
    # # Load data
    X, Y = load_data(path, chars, precentLow, precentHigh, shuffle)
    test_size = [0.2]
    X, Y = preprocess_data(X, Y)
    # Encode the labels
    # label_encoder = LabelEncoder()
    # labels_encoded = label_encoder.fit_transform(Y)
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    print('X_train',X_train)
    print('X_test',X_test)
    print('y_train',y_train)
    print('y_test',y_test)
    # Convert the input data to numpy.ndarray format
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    x_train_split = np.array_split(X_train, 5)
    y_train_split = np.array_split(y_train, 5)

    # Define the neural network model
    model = Sequential()
    model.add(Dense(32, activation='relu', input_shape=(100,)))  # Example architecture, adjust as needed
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    sumResult = []
    for i in range(5):
        arr = [0, 0, 0, 0, 0]
        while np.sum(arr) < 5:
            rand = random.randint(0, 4)
            if arr[rand] < 1:
                arr[rand] += 1
                # Train the neural network
                model.fit(x_train_split[rand], y_train_split[rand], epochs=10, batch_size=32)  # Example number of epochs and batch size, adjust as needed
        resultPredict = model.predict(X_test, 1, 0)
        print('resultPredict',resultPredict)
        print('len(resultPredict)', len(resultPredict))
        print('y_test',y_test)
        print('len(y_test)', len(y_test))
        # result1 = accuracy_score(y_test, resultPredict)
        loss, accuracy = model.evaluate(y_test, resultPredict)
        sumResult.append(accuracy)

    print(f"chars classification: {images_dictionary}")
    print(f"sumResults= {sumResult}")
    print(f"Average= {np.sum(sumResult)/5}")
    # Calculate the standard deviation of the array
    sumResultStd = np.std(sumResult)
    print("Standard deviation of the sumResult array:", sumResultStd)

    # Evaluate the model on the test set
    # loss, accuracy = model.evaluate(X_test, y_test)
    # print("Test loss:", loss)
    # print("Test accuracy:", accuracy)
    # sumResultStd = np.std(sumResult)  
if __name__ == '__main__':
    main()