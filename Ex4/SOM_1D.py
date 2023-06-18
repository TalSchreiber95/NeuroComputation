import numpy as np
import matplotlib.pyplot as plt


class Kohonen_1D:
    def __init__(self, data: np.ndarray, net_size: int):
        """
        Initializes the Kohonen_1D class.

        :param data: Training data.
        :param net_size: Number of neurons.
        """
        rand = np.random.RandomState(0)
        h = np.sqrt(net_size).astype(int)
        self.SOM = rand.randint(0, 1000, (net_size, 2)).astype(float) / 1000
        self.data = data
        self.net_size = net_size

    def find_BMU(self, sample):
        """
        Finds the Best Matching Unit (BMU) for a given sample.

        Calculates the Euclidean distance from the sample to all neurons,
        and returns the indices of the neuron with the minimum distance.

        :param sample: Single training example.
        :return: Indices of the BMU.
        """
        distSq = (np.square(self.SOM - sample)).sum(axis=1)
        return np.unravel_index(np.argmin(distSq, axis=None), distSq.shape)

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
        # If the radius is close to zero, only the BMU is changed
        if radius_sq < 1e-3:
            self.SOM[x, :] += learn_rate * (sample - self.SOM[x, :])
            return self.SOM
        # Change all cells in a small neighborhood of the BMU
        for i in range(max(0, x - step), min(self.SOM.shape[0], x + step)):
            dist_sq = np.square(i - x)
            dist_func = np.exp(-dist_sq / 2 / radius_sq)
            self.SOM[i, :] += learn_rate * dist_func * (sample - self.SOM[i, :])
        return self.SOM

    def train_SOM(self, learn_rate=.9, radius_sq=1, lr_decay=.1, radius_decay=.1, epochs=10):
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
            rand.shuffle(self.data)
            for sample in self.data:
                x = self.find_BMU(sample)
                self.SOM = self.update_weights(sample, learn_rate, radius_sq, x)
            self.plot("line: curr iter: " + str(epoch) + " , learning rate: " + str(round(learn_rate, 3)))
            # Update learning rate and radius
            learn_rate = learn_rate_0 * np.exp(-epoch * lr_decay)
            radius_sq = radius_0 * np.exp(-epoch * radius_decay)
        return self.SOM

    def plot(self, title):
        """
        Plots the SOM grid.

        :param title: Title of the plot.
        """
        X = self.SOM[:, 0]  # The X coordinates of each point
        Y = self.SOM[:, 1]  # The Y coordinates of each point

        fig, ax = plt.subplots()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        xs = []  # X coordinates of each point in axis 1 (columns)
        ys = []  # Y coordinates of each point in axis 1 (rows)
        for i in range(self.SOM.shape[0]):
            xs.append(self.SOM[i, 0])
            ys.append(self.SOM[i, 1])

        ax.plot(xs, ys, 'c-', markersize=0.5, linewidth=0.7)  # Changed color to cyan
        ax.plot(X, Y, color='k', marker='*', linewidth=0, markersize=8, alpha=0.5)
        ax.scatter(self.data[:, 0], self.data[:, 1], c="m", alpha=0.2)  # Changed color to magenta
        plt.title(title)
        plt.show()
