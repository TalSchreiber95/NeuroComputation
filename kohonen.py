import numpy as np
import matplotlib.pyplot as plt
from kohonenDatasetGenerator import generate_uniform_data, generate_non_uniform_data
from sklearn.preprocessing import MinMaxScaler

# Kohonen Algorithm
def kohonen_algorithm(data, num_neurons, learning_rate, neighborhood_radius, num_iterations, weight_change_threshold):
    num_features = data.shape[1]
    num_data_points = data.shape[0]

    # Initialize neurons with random weights
    neurons = np.random.uniform(low=0, high=1, size=(num_neurons, num_features))

    # Initialize previous neurons for weight change comparison
    prev_neurons = neurons.copy()

    # Calculate the decrease rate for the neighborhood radius
    neighborhood_radius_decrease_rate = (neighborhood_radius / num_iterations) * 5

    # Training loop
    for iteration in range(num_iterations):
        # Select a random data point
        data_point = data[np.random.randint(num_data_points)]

        # Find the best matching unit (BMU)
        bmu_idx = np.argmin(np.linalg.norm(neurons - data_point, axis=1))

        # Update the weights of BMU and its neighbors
        for i in range(num_neurons):
            distance = np.abs(i - bmu_idx)
            influence = np.exp(-distance / (2 * neighborhood_radius**2))
            neurons[i] += learning_rate * influence * (data_point - neurons[i])

        # Decay learning rate and neighborhood radius over iterations
        learning_rate *= 0.98
        neighborhood_radius -= neighborhood_radius_decrease_rate

        # Check weight change and break if below threshold
        weight_change = np.max(np.abs(neurons - prev_neurons))
        if weight_change < weight_change_threshold:
            break

        # Update previous neurons for the next iteration
        prev_neurons = neurons.copy()

    return neurons

# Generate the uniform dataset
num_points = 1000
uniform_data = generate_uniform_data(num_points)
non_uniform_data = generate_non_uniform_data(num_points)

# Normalize the data
scaler = MinMaxScaler()
normalized_uniform_data = scaler.fit_transform(uniform_data)
normalized_non_uniform_data = scaler.transform(non_uniform_data)

# Kohonen Algorithm parameters
num_neurons = 30
learning_rate = 0.1
neighborhood_radius = 0.1
num_iterations = 5000
weight_change_threshold = 1e-5  # Adjust the weight change threshold

# Run the Kohonen Algorithm for the uniform dataset
uniform_neurons = kohonen_algorithm(normalized_uniform_data, num_neurons, learning_rate, neighborhood_radius, num_iterations, weight_change_threshold)

# Run the Kohonen Algorithm for the non-uniform dataset
non_uniform_neurons = kohonen_algorithm(normalized_non_uniform_data, num_neurons, learning_rate, neighborhood_radius, num_iterations, weight_change_threshold)

# Generate line coordinates based on neuron weights
line_coords = np.linspace(0, 1, num_neurons).reshape(-1, 1)
# Plot the data, line, and neurons for the uniform dataset
plt.figure(figsize=(10, 8))  # Increase the figure height to accommodate the increased spacing
plt.subplot(2, 1, 1)  # Adjust the subplot arrangement to have two rows and one column
plt.scatter(uniform_data[:, 0], uniform_data[:, 1], color='blue', label='Data')
plt.plot(line_coords, line_coords, color='red', label='Line of Neurons')
plt.scatter(uniform_neurons[:, 0], uniform_neurons[:, 1], color='green', marker='s', label='Neurons')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Kohonen Algorithm - Fitting a Line of Neurons to Uniform Data')
plt.legend()

# Plot the data, line, and neurons for the non-uniform dataset
plt.subplot(2, 1, 2)  # Adjust the subplot arrangement to have two rows and one column
plt.scatter(non_uniform_data[:, 0], non_uniform_data[:, 1], color='blue', label='Data')
plt.plot(line_coords, line_coords, color='red', label='Line of Neurons')
plt.scatter(non_uniform_neurons[:, 0], non_uniform_neurons[:, 1], color='green', marker='s', label='Neurons')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Kohonen Algorithm - Fitting a Line of Neurons to Non-Uniform Data')
plt.legend()

plt.subplots_adjust(hspace=0.5)  # Increase the vertical spacing between the subplots

plt.show()
