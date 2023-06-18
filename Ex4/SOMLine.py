import numpy as np
import matplotlib.pyplot as plt

class KohonenLine:
    def __init__(self, data: np.ndarray, net_size: int , start , end):
        """
        Initializes the Kohonen_1D class.

        :param data: Training data.
        :param net_size: Number of neurons.
        """
        rand = np.random.RandomState(0)
        calc = rand.randint(0, 1000, (net_size, 2)).astype(float)
        self.SOM = calc / 1000  # Initialize the SOM weights randomly between 0 and 1
        self.data = data
        self.net_size = net_size
        self.start = start
        self.end = end

    def find_BMU(self, sample):
        """
        Finds the Best Matching Unit (BMU) for a given sample.

        Calculates the Euclidean distance from the sample to all neurons,
        and returns the indices of the neuron with the minimum distance.

        :param sample: Single training example.
        :return: Indices of the BMU.
        """
        distSq = (np.square(self.SOM - sample)).sum(axis=1)  # Calculate the squared Euclidean distances between the sample and all neurons
        return np.unravel_index(np.argmin(distSq, axis=None), distSq.shape)  # Return the indices of the neuron with the minimum distance

    def update_weights(self, sample, learn_rate, radius_sq, bmu_idx, step=3):
        """
        Updates the weights of the BMU and its neighbors.

        :param sample: Single training example.
        :param learn_rate: Learning rate.
        :param radius_sq: Square of the radius.
        :param bmu_idx: Indices of the BMU.
        :param step: Size of the neighborhood.
        :return: Updated SOM weights.
        """
        x = bmu_idx[0]
        if radius_sq < 1e-3:  # If the radius is very small, only update the weights of the BMU itself
            self.SOM[x, :] += learn_rate * (sample - self.SOM[x, :])
            return self.SOM
        for i in range(max(0, x - step), min(self.SOM.shape[0], x + step)):
            dist_sq = np.square(i - x)
            dist_func = np.exp(-dist_sq / 2 / radius_sq)  # Calculate the neighborhood function based on the distance between the neuron and the BMU
            self.SOM[i, :] += learn_rate * dist_func * (sample - self.SOM[i, :])  # Update the weights of the neuron based on the neighborhood function and the learning rate
        return self.SOM

    def train_SOM(self, learn_rate=.8, radius_sq=1, lr_decay=.1, radius_decay=.1, epochs=10):
        """
        Trains the Self-Organizing Map (SOM) model.

        For each sample:
            1. Find the BMU.
            2. Update the weights.
            3. Update the learning rate.
            4. Update the radius.

        :param lr_decay: Rate of decay of the learning rate.
        :param radius_decay: Rate of decay of the radius.
        :param epochs: Number of training epochs.
        :return: Trained SOM weights.
        """
        rand = np.random.RandomState(0)
        learn_rate_0 = learn_rate
        radius_0 = radius_sq
        for epoch in np.arange(0, epochs):
            rand.shuffle(self.data)  # Shuffle the training data for each epoch
            for sample in self.data:
                x = self.find_BMU(sample)  # Find the Best Matching Unit (BMU) for the sample
                self.SOM = self.update_weights(sample, learn_rate, radius_sq, x)  # Update the weights of the BMU and its neighbors
            self.plot(f"Algorithm: Line | iteration number:  {str(epoch)} | learning rate: {str(round(learn_rate, 3))}") # Plot the SOM grid for visualization
            learn_rate = learn_rate_0 * np.exp(-epoch * lr_decay)  # Update the learning rate with exponential decay
            radius_sq = radius_0 * np.exp(-epoch * radius_decay)  # Update the radius with exponential decay
        return self.SOM

    def plot(self, title):
        """
        Plots the SOM grid.

        :param title: Title of the plot.
        """
        X = self.SOM[:, 0]  # Extract the x-coordinates of the SOM neurons
        Y = self.SOM[:, 1]  # Extract the y-coordinates of the SOM neurons

        fig, ax = plt.subplots()
        ax.set_xlim(self.start, self.end)  # Set the x-axis limits of the plot
        ax.set_ylim(self.start, self.end)  # Set the y-axis limits of the plot
        xs = []  # List to store the x-coordinates of the SOM neurons
        ys = []  # List to store the y-coordinates of the SOM neurons
        for i in range(self.SOM.shape[0]):
            xs.append(self.SOM[i, 0])  # Append the x-coordinate of each neuron to the xs list
            ys.append(self.SOM[i, 1])  # Append the y-coordinate of each neuron to the ys list

        ax.plot(xs, ys, 'c-', markersize=0.5, linewidth=0.7)  # Plot lines connecting the neurons, color set to cyan
        ax.plot(X, Y, color='k', marker='*', linewidth=0, markersize=8, alpha=0.5)  # Plot markers for the neurons, color set to black, marker set to asterisk
        ax.scatter(self.data[:, 0], self.data[:, 1], c="m", alpha=0.2)  # Scatter plot of the training data, color set to magenta
        plt.title(title)  # Set the title of the plot
        plt.show()
