from scipy.special import gamma
import numpy as np
import matplotlib.pyplot as plt
import math
from numpy.linalg import norm


def generate_data(dim_count, num_points):
    """
    Creates a random number of points in n-dimensional space
    :param dim_count: The number of dimensions to work with.
    :param num_points: The number of points to generate in dim_count-dimensional space
    :return:
    """
    random_points = np.random.rand(num_points, dim_count)
    return random_points


def calculate_euclidean_distances(ref_point, data_points):
    """
    Calculates the Euclidean distance between two points
    :param ref_point: A reference point
    :param data_points: A set of random data points in space
    :return: The nearest point and the farthest point.
    """
    nearest_distance = float('inf')
    farthest_distance = 0
    for point in data_points:
        distance = np.linalg.norm(point - ref_point)
        if distance < nearest_distance:
            nearest_distance = distance
        if distance > farthest_distance:
            farthest_distance = distance
    return nearest_distance, farthest_distance


def calculate_cosine_similarity(ref_point, data_points):
    """
     Calculates the cosine similarity between two data points and the reference point.
    :param ref_point: A reference point
    :param data_points: A set of random data points in space
    :return:
    """
    cosine_similarities = []
    for point in data_points:
        cosine_similarity = np.dot(ref_point, point) / (norm(ref_point) * norm(point))
        cosine_similarities.append((cosine_similarity, point))
    cosine_similarities = sorted(cosine_similarities, key=lambda x1: x1[0], reverse=True)
    closest_point_distance = np.linalg.norm(cosine_similarities[0][1] - ref_point)
    farthest_point_distance = np.linalg.norm(cosine_similarities[-1][1] - ref_point)
    return closest_point_distance, farthest_point_distance


def test_measure():
    return 8


def test_measure2():
    return 12


def test_measure3():
    return 16


closest_points = []
farthest_points = []
dimension_count = 800
point_count = 10
plot_step = 5
method = 'euc'


for dimensions in range(1, dimension_count):
    data = generate_data(dimensions, point_count)
    reference_point = np.random.rand(dimensions)
    if method == 'euc':
        closest_point, farthest_point = calculate_euclidean_distances(reference_point, data)
    else:
        closest_point, farthest_point = calculate_cosine_similarity(reference_point, data)
    closest_points.append(closest_point)
    farthest_points.append(farthest_point)

# Creating dimensions array for x-axis
dimensions = np.arange(1, dimension_count)

# Plotting
plt.loglog(dimensions[::plot_step], closest_points[::plot_step], label="Closest Distances")
plt.loglog(dimensions[::plot_step], farthest_points[::plot_step], label="Farthest Distances")
plt.xlabel("Dimensions")
plt.ylabel("Distance")
plt.title("Closest and Farthest Distances as a Function of Dimensions")
plt.legend()
plt.show()
