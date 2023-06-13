import numpy as np

# Part A - Line of neurons in a square
def generate_uniform_data(num_points):
    # Generate uniform random (x, y) coordinates within the square
    x = np.random.uniform(low=0, high=1, size=num_points)
    y = np.random.uniform(low=0, high=1, size=num_points)
    return np.column_stack((x, y))

def generate_non_uniform_data(num_points):
    # Generate non-uniform random (x, y) coordinates within the square
    x = np.random.uniform(low=0, high=1, size=num_points)
    y = np.random.uniform(low=0, high=1, size=num_points) ** 2
    return np.column_stack((x, y))

# Part B - Circle of neurons in a "donut" shape
def generate_donut_data(num_points):
    # Generate random (r, theta) coordinates within the donut shape
    r = np.random.uniform(low=2, high=4, size=num_points)
    theta = np.random.uniform(low=0, high=2*np.pi, size=num_points)
    
    # Convert polar coordinates to Cartesian coordinates
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack((x, y))

# Example usage
num_points = 1000  # Number of data points to generate

# Generate uniform data in a square
uniform_data = generate_uniform_data(num_points)

# Generate non-uniform data in a square
non_uniform_data = generate_non_uniform_data(num_points)

# Generate data in a donut shape
donut_data = generate_donut_data(num_points)
