import numpy as np
import matplotlib.pyplot as plt


class Kohonen:
    def __init__(self, data: np.ndarray, net_size: int):
        """
        Initializes the Kohonen class.

        :param data: Training data.
        :param net_size: Number of neurons.
        """
        rand = np.random.RandomState(0)
        h = np.sqrt(net_size).astype(int)
        self.SOM = rand.randint(0, 1000, (h, h, 2)).astype(float) / 1000
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
        distSq = (np.square(self.SOM - sample)).sum(axis=2)
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
        x, y = bmu_idx
        # If the radius is close to zero, only the BMU is changed
        if radius_sq < 1e-3:
            self.SOM[x, y, :] += learn_rate * (sample - self.SOM[x, y, :])
            return self.SOM
        # Change all cells in a small neighborhood of the BMU
        for i in range(max(0, x - step), min(self.SOM.shape[0], x + step)):
            for j in range(max(0, y - step), min(self.SOM.shape[1], y + step)):
                dist_sq = np.square(i - x) + np.square(j - y)
                dist_func = np.exp(-dist_sq / 2 / radius_sq)
                self.SOM[i, j, :] += learn_rate * dist_func * (sample - self.SOM[i, j, :])
        return self.SOM

    def train_SOM(self, learn_rate=.8, radius_sq=1, lr_decay=.1, radius_decay=.1, epochs=10):
        """
        Trains the Self-Organizing Map (SOM) model.

        For each sample:
            1. Find the BMU.
            2. Update the weights.
            3. Update the learning rate.
            4. Update the radius.

        :param learn_rate: Initial learning rate.
        :param radius_sq: Initial square of the radius.
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
                x, y = self.find_BMU(sample)
                self.SOM = self.update_weights(sample, learn_rate, radius_sq, (x, y))
            self.plot("Grid: curr iter: " + str(epoch) + " , learning rate: " + str(round(learn_rate, 3)))
            # Update learning rate and radius
            learn_rate = learn_rate_0 * np.exp(-epoch * lr_decay)
            radius_sq = radius_0 * np.exp(-epoch * radius_decay)
        return self.SOM

    def plot(self, title):
        """
        Plots the SOM grid.

        :param title: Title of the plot.
        """
        X = self.SOM[:, :, 0]  # X coordinates of each point
        Y = self.SOM[:, :, 1]  # Y coordinates of each point

        fig, ax = plt.subplots()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        for i in range(self.SOM.shape[0]):
            xs = []  # X coordinates of each point in axis 1 (columns)
            ys = []  # Y coordinates of each point in axis 1 (rows)
            xh = []  # X coordinates of each point in axis 0 (columns)
            yh = []  # Y coordinates of each point in axis 1 (rows)
            for j in range(self.SOM.shape[1]):
                xs.append(self.SOM[i, j, 0])
                ys.append(self.SOM[i, j, 1])
                xh.append(self.SOM[j, i, 0])
                yh.append(self.SOM[j, i, 1])

            ax.plot(xs, ys, 'c-', markersize=0.5, linewidth=0.7)  # Changed color to cyan
            ax.plot(xh, yh, 'c-', markersize=0.5, linewidth=0.7)  # Changed color to cyan

        ax.plot(X, Y, color='k', marker='*', linewidth=0, markersize=8, alpha=0.5)
        ax.scatter(self.data[:, 0], self.data[:, 1], c="m", alpha=0.2)  # Changed color to magenta
        plt.title(title)
        plt.show()
