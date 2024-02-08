import os
import time

import numpy as np
import trimesh
from matplotlib import pyplot as plt
from scipy.spatial import ConvexHull

from quick_hull_3d import QuickHull3D


def test_space_sampling() -> tuple:
    plot_params = {'show_indexes': True,
                   'marker_size': 10,
                   'elev': 45,
                   'azim': 45}
    points = np.array([[-1, 0, 0],
                       [-0.25, -0, -0.25],
                       [-0.5, 0, 0],
                       [0, 0, 0],
                       [0.5, 0, 0],
                       [1, 0, 0],
                       [0, -1, 0],
                       [0, -0.5, 0],
                       [0, 0.5, 0],
                       [0, 1, 0],
                       [0, 0, -1],
                       [0, 0, -0.5],
                       [0, 0, 0.5],
                       [0, 0, 1],
                       [0.8, 0.8, 0.8],
                       [0.8, 0.8, -0.8],
                       [0.8, -0.8, 0.8],
                       [-0.8, 0.8, 0.8],
                       [-0.8, 0.8, -0.8],
                       [-0.8, -0.8, 0.8],
                       [0.8, -0.8, -0.8],
                       [-0.8, -0.8, -0.8]
                      ])
    return points, plot_params


def random_space_sampling():
    plot_params = {'show_indexes': False,
                   'marker_size': 10,
                   'elev': 150,
                   'azim': 270}
    pts = np.random.rand(200, 3)
    pts = pts - 0.5
    pts = pts * 2
    return pts, plot_params


def load_space_sample(file_path) -> tuple:
    plot_params = {'show_indexes': False,
                   'marker_size': 1,
                   'elev': 150,
                   'azim': 270}
    mesh = trimesh.load(file_path)
    vertices = mesh.vertices
    return vertices, plot_params


if __name__ == "__main__":
    bunny_res2_file_path = os.path.join(os.getcwd(), "bunny\\reconstruction\\bun_zipper_res2.ply")
    dragon_file_path = os.path.join(os.getcwd(), "dragon\\dragon_vrip_res2.ply")
    # points, plot_params = load_space_sample(dragon_file_path)
    # points, plot_params = test_space_sampling()
    points, plot_params = random_space_sampling()
    start_time = time.time()
    solver = QuickHull3D(points, plot_params=plot_params)
    solver.find_hull(initial_shape='tetrahedron')
    solver.plot_hull(points)

    temp = 1
