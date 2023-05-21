import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from setting import config
from utils import load_data, preprocess_data, export_to_json

# Load the data generated in Part I
path = config['output_result_path']
chars = config['chars']
value_to_remove = 3
images_dictionary = config['images_dictionary']
images_dictionary = np.delete(images_dictionary, value_to_remove-1)
chars.remove(value_to_remove)
precentLow = 0.08  # the low precent of the letter paint
precentHigh = 0.8  # the high precent of the letter paint
shuffle = True
# # Load data
data,labels = load_data(path, chars, precentLow, precentHigh, shuffle)
# data = np.load("data.npy")
# labels = np.load("labels.npy")

# Encode the labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels_encoded, test_size=0.2)

# Convert the input data to numpy.ndarray format
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

# Define the neural network model
model = Sequential()
model.add(Dense(32, activation='relu', input_shape=(100,)))  # Example architecture, adjust as needed
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# Train the neural network
model.fit(X_train, y_train, epochs=10, batch_size=32)  # Example number of epochs and batch size, adjust as needed

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test)
print("Test loss:", loss)
print("Test accuracy:", accuracy)
