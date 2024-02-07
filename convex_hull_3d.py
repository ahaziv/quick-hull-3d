from typing import List, Union, Dict
import trimesh
import os
import copy

import numpy as np
from matplotlib import pyplot as plt


def test_space_sampling():
    return np.array([[-1, 0, 0],
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


def random_space_sampling():
    pts = np.random.rand(200, 3)
    pts = pts - 0.5
    pts = pts * 2
    return pts


def load_space_sample(file_path):
    mesh = trimesh.load(file_path)
    vertices = mesh.vertices
    return vertices

class QuickHull3D:
    def __init__(self, points: np.array, log: bool = False):
        self._points = points
        self._hull_vertices = None
        self._hull_planes = None
        self._plane_neighbors = {}
        self._plane_queue = []
        self._log = log

    def initial_tetrahedron(self):
        hull_idxs = [self._points[:, 0].argmin(),
                     self._points[:, 1].argmin(),
                     self._points[:, 2].argmin(),
                     self._points[:, 1].argmax()]
        self._hull_vertices = np.array([self._points[index, :] for index in hull_idxs])
        hull_idxs.sort()
        hull_idxs.reverse()
        for index in hull_idxs:
            self._points = np.delete(self._points, index, 0)

        self._hull_planes = [(0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)]
        self._plane_neighbors = {(0, 2, 1): [(0, 3, 2), (1, 2, 3), (0, 1, 3)],
                                 (0, 1, 3): [(0, 2, 1), (1, 2, 3), (0, 3, 2)],
                                 (0, 3, 2): [(0, 1, 3), (1, 2, 3), (0, 2, 1)],
                                 (1, 2, 3): [(0, 2, 1), (0, 3, 2), (0, 1, 3)]}
        self.delete_internal_points(self._hull_planes)

    def initial_triangle(self):
        hull_idxs = [self._points[:, 0].argmin(),
                     self._points[:, 1].argmin(),
                     self._points[:, 2].argmin()]
        self._hull_vertices = np.array([self._points[index, :] for index in hull_idxs])
        hull_idxs.sort()
        hull_idxs.reverse()
        for index in hull_idxs:
            self._points = np.delete(self._points, index, 0)
        self._hull_planes = [(2, 1, 0), (0, 1, 2)]

    def remove_plane(self, plane: Union[tuple, int]):
        if type(plane) is int:
            plane = self._hull_planes.pop(plane)
        elif type(plane) is tuple:
            self._hull_planes.remove(plane)
        else:
            raise TypeError(f"illegal type for plane: {type(plane)} "
                            f"plane must be passed as either an integer (index) or tuple (plane).")
        neighbors = self._plane_neighbors.pop(plane)
        self.log(f'removing plane: {plane}')
        return neighbors

    def add_planes(self, planes: List[tuple], planes_neighbors: Dict[tuple, List[tuple]]):
        self._hull_planes += planes
        self._plane_neighbors.update(planes_neighbors)
        for plane, neighbors in planes_neighbors.items():
            for idx, neighbor in enumerate(neighbors):
                for in_idx, inner_neighbor in enumerate(self._plane_neighbors[neighbor]):
                    if inner_neighbor not in self._plane_neighbors:
                        self._plane_neighbors[neighbor][in_idx] = plane

    def find_hull(self, initial_shape: str):
        if initial_shape == 'triangle':
            self.initial_triangle()
        if initial_shape == 'tetrahedron':
            self.initial_tetrahedron()
        self._plane_queue = copy.deepcopy(self._hull_planes)
        while self._plane_queue:
            plane = self._plane_queue.pop(0)
            while plane not in self._plane_neighbors:
                plane = self._plane_queue.pop(0)
            new_pnt_idx = self.find_most_dist_point_index(np.array([self._hull_vertices[plane[0], :],
                                                                    self._hull_vertices[plane[1], :],
                                                                    self._hull_vertices[plane[2], :]]))
            if new_pnt_idx is not None:
                self._hull_vertices = np.vstack([self._hull_vertices, self._points[new_pnt_idx, :]])
                self._points = np.delete(self._points, new_pnt_idx, 0)
                new_hull_pt_idx = len(self._hull_vertices[:, 0]) - 1
                new_planes = [(plane[0], plane[1], new_hull_pt_idx),
                              (plane[1], plane[2], new_hull_pt_idx),
                              (plane[2], plane[0], new_hull_pt_idx)]

                new_planes_neighbors = {new_planes[0]: [self._plane_neighbors[plane][0], new_planes[1], new_planes[2]],
                                        new_planes[1]: [self._plane_neighbors[plane][1], new_planes[2], new_planes[0]],
                                        new_planes[2]: [self._plane_neighbors[plane][2], new_planes[0], new_planes[1]]}

                self.delete_internal_points(new_planes + [self.flip_triangle(plane)])
                neighbors = self.remove_plane(plane)
                self.add_planes(new_planes, new_planes_neighbors)

                new_planes = self.re_edge_adjacent_faces(plane, neighbors, new_planes, new_hull_pt_idx)
                self._plane_queue += new_planes
                self.plot_hull()

    def is_above_plane(self, plane: tuple, point: np.array):
        vec_a = self._hull_vertices[plane[1], :] - self._hull_vertices[plane[0], :]
        vec_b = self._hull_vertices[plane[2], :] - self._hull_vertices[plane[1], :]
        vec_p = point - self._hull_vertices[plane[2], :]
        return np.dot(vec_p, np.cross(vec_a, vec_b)) > 0

    def find_most_dist_point_index(self, surf_pts: np.array) -> int:
        plane_normal = np.cross((surf_pts[1] - surf_pts[0]), (surf_pts[2] - surf_pts[1]))
        plane_normal = plane_normal / np.linalg.norm(plane_normal)
        max_dist = 0
        point_index = None
        for index, pnt in enumerate(self._points):
            vec_to_pnt = pnt - surf_pts[0, :]
            dist_to_plane = np.dot(plane_normal, vec_to_pnt)
            if dist_to_plane > max_dist:
                point_index = index
                max_dist = dist_to_plane
        return point_index

    def internal_points_to_3d_polygon(self, surface_triangles: List) -> List[int]:
        internal_idxs = set(range(len(self._points)))
        for triangle in surface_triangles:
            idxs_to_remove = set()
            for idx in internal_idxs:
                if self.is_above_plane(triangle, self._points[idx, :]):
                    idxs_to_remove.add(idx)
            for idx in idxs_to_remove:
                internal_idxs.remove(idx)
        return list(internal_idxs)

    def delete_internal_points(self, polygon_planes: List[tuple]) -> np.array:
        internal_points_idxs = self.internal_points_to_3d_polygon(polygon_planes)
        internal_points_idxs.sort()
        internal_points_idxs.reverse()
        for pt_idx in internal_points_idxs:
            self._points = np.delete(self._points, pt_idx, 0)

    def re_edge_adjacent_faces(self, plane: tuple, plane_neighbors: List[tuple], new_planes: List[tuple],
                               new_hull_pt_idx: int) -> List[tuple]:
        """
        recursive function that re-edges all faces that are adjacent to a certain face and are below a given point
        """
        for adj_plane_idx, adj_plane in enumerate(plane_neighbors):
            if not self.is_above_plane(adj_plane, self._hull_vertices[-1, :]):
                continue
            self.log(f'plane of origin:{plane},   {new_hull_pt_idx} is above {adj_plane}')
            if adj_plane_idx < 2:
                base_points = [plane[adj_plane_idx], plane[adj_plane_idx+1]]
            else:
                base_points = [plane[adj_plane_idx], plane[0]]

            for idx, point in enumerate(adj_plane):
                if point not in base_points:
                    outer_point, outer_point_idx = point, idx
                    break

            for np_idx, new_plane in enumerate(new_planes):
                if base_points[0] in new_plane and base_points[1] in new_plane:
                    fixed_planes = [(base_points[0], outer_point, new_hull_pt_idx),
                                    (outer_point, base_points[1], new_hull_pt_idx)]

                    plane_neighbor_idx = self._plane_neighbors[adj_plane].index(new_plane)

                    fixed_planes_neighbors = {
                        fixed_planes[0]: [self._plane_neighbors[adj_plane][(plane_neighbor_idx + 1) % 3],
                                          fixed_planes[1],
                                          new_planes[(np_idx + len(new_planes) - 1) % len(new_planes)]],
                        fixed_planes[1]: [self._plane_neighbors[adj_plane][(plane_neighbor_idx + 2) % 3],
                                          new_planes[(np_idx + 1) % len(new_planes)],
                                          fixed_planes[0]]}

                    self.delete_internal_points([fixed_planes[0],
                                                 fixed_planes[1],
                                                 self.flip_triangle(new_plane),
                                                 self.flip_triangle(adj_plane)])

                    new_planes = new_planes[:np_idx] + fixed_planes + new_planes[np_idx+1:]

                    self.remove_plane(new_plane)
                    adj_plane_neighbors = self.remove_plane(adj_plane)
                    self.add_planes(fixed_planes, fixed_planes_neighbors)

                    new_planes = self.re_edge_adjacent_faces(adj_plane, adj_plane_neighbors, new_planes, new_hull_pt_idx)
        return new_planes

    def plot_hull(self, orig_points: np.array = None):
        show_indexes = False
        marker_size = 1
        fig = plt.figure(figsize=plt.figaspect(1))
        ax = fig.add_subplot(projection='3d')
        ax.view_init(elev=270, azim=90)
        if orig_points is not None:
            ax.scatter(orig_points[:, 0], orig_points[:, 1], orig_points[:, 2], color='blue', s=marker_size)
        ax.scatter(self._points[:, 0], self._points[:, 1], self._points[:, 2], color='blue', s=marker_size)
        ax.scatter(self._hull_vertices[:, 0], self._hull_vertices[:, 1], self._hull_vertices[:, 2], color='red', s=marker_size)
        if show_indexes:
            for i, point in enumerate(self._hull_vertices):
                ax.text(point[0] + 0.1, point[1] + 0.1, point[2] + 0.1, str(int(i)), color='black')
        if self._hull_vertices is not None:
            ax.plot_trisurf(self._hull_vertices[:, 0], self._hull_vertices[:, 1], self._hull_vertices[:, 2],
                            triangles=self._hull_planes, edgecolor=[[1, 0, 0]], linewidth=1.0, alpha=0.3, shade=False)
        fig.tight_layout()
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.show()

    def log(self, message):
        if self._log:
            print(message)

    @staticmethod
    def flip_triangle(triangle: tuple):
        return triangle[1], triangle[0], triangle[2]

    @staticmethod
    def find_plane_neighbors(plane, search_planes):
        adjacent_planes = {}
        for point in plane:
            for plane in search_planes:
                if point not in plane:
                    continue
                if plane in adjacent_planes:
                    adjacent_planes[plane] += 1
                else:
                    adjacent_planes[plane] = 1
        return [plane for plane, shared_points in adjacent_planes.items() if shared_points == 2]


# -------------------------------------- main ---------------------------------------- #
file_path = os.path.join(os.getcwd(), "quick_hull_3d\\bunny\\reconstruction\\bun_zipper_res2.ply")
pts = load_space_sample(file_path)
# pts = test_space_sampling()
solver = QuickHull3D(pts)
solver.find_hull(initial_shape='tetrahedron')
# solver.plot_hull(pts)
temp = 1

