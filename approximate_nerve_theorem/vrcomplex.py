import numpy as np
import matplotlib.pyplot as plt

from bisect import bisect
from sklearn.metrics import pairwise_distances
from sklearn.base import TransformerMixin
from collections import defaultdict
from matplotlib.collections import LineCollection
from matplotlib.axes import Axes
from typing import Dict, List, DefaultDict, Tuple, Iterable, Optional, Set


class VietorisRipsComplex:
    """
    Vietoris-Rips simplicial complex.
    """
    
    def __init__(self, points: np.array):
        """
        Args:
            points (np.array): point cloud over which the complex will be built
        """
        self.vertices = points
        self.simplices: DefaultDict[int, List[Tuple[int, ...]]] = defaultdict(list)
        # distance matrix for vertices
        self._distances = pairwise_distances(self.vertices)
        # time of birth for each simplex
        self.lower_distance: DefaultDict[int, List[float]] = defaultdict(list)
    
    @property
    def simplex_dim(self) -> int:
        """
        Max simplex dimensionality.
        """
        return max(key for key, item in self.simplices.items() if len(item) > 0)
        
    def _simplex_intersection(self, dist: float, simplex_a: Tuple[int, ...], 
                              simplex_b: Tuple[int, ...]) -> Tuple[Optional[Set[int]], Optional[float]]:
        """
        Find an intersection of two simplices and birth distance for the new intersection simplex.
        
        Args:
            dist (float): max distance
            simplex_a (Tuple[int, ...]): set of simplex vertices
            simplex_b (Tuple[int, ...]): set of simplex vertices
        
        Returns:
            (None, None) if simplices do not intersect.
            (simplex, birth distance) if simplices intersect.
        """
        distinct_vertices = set(simplex_b) - set(simplex_a)
        if len(distinct_vertices) != 1:
            return None, None
        distinct = distinct_vertices.pop()
        lower_distance = max(self._distances[a, distinct] for a in simplex_a)
        if lower_distance > dist:
            return None, None
        return tuple(set(simplex_a) | {distinct}), lower_distance 
    
    def _simplex_to_point_set(self, simplex: Tuple[int, ...]) -> np.array:
        """
        Convert the simplex as a set of vertex numbers to actual coordinates.
        
        Args:
            simplex (Tuple[int, ...]): set of simplex vertices
        
        Returns:
            Coordinates of simplex vertices.
        """
        return np.array([self.vertices[vtx] for vtx in simplex])
    
    def geometric_realization(self, dim: int, ax: Axes) -> None:
        """
        Draw simplices of one dimension.
        
        Args:
            dim (int): simplex dimension
            ax (matplotlib.axes.Axes): axes to draw on
        
        Returns:
            None
        """
        if dim > self.simplex_dim:
            raise ValueError(f"Dimensionality is too high: max dim for this complex is {self.simplex_dim} but got {dim} as dim")
        
        ax.set_xlim(self.vertices[:,0].min() - 1, self.vertices[:,0].max() + 1)
        ax.set_ylim(self.vertices[:,1].min() - 1, self.vertices[:,1].max() + 1)
        
        if dim == 0:    # points
            ax.scatter(self.vertices[:,0], self.vertices[:,1])
            return
        
        coords = [self._simplex_to_point_set(simplex) for simplex in self.simplices[dim]]
        if dim == 1:    # edges
            ax.add_collection(LineCollection(coords))
        else:           # volumes
            for polygon in coords:
                ax.fill(polygon[:,0], polygon[:,1], color=(0, 0, 1, 0.1))
    
    def show_complex(self) -> None:
        """
        Draw complex dimensions side by side.
        
        Returns:
            None
        """
        N = self.simplex_dim + 1
        fig, ax = plt.subplots(1, N)
        fig.set_figheight(5)
        fig.set_figwidth(5 * N)
        for dim in range(N):
            self.geometric_realization(dim, (ax[dim] if N > 1 else ax))

    def show_complex_single(self) -> None:
        """
        Draw complex on a single plot.
        
        Returns:
            None
        """
        N = self.simplex_dim + 1
        fig, ax = plt.subplots()
        fig.set_figheight(5)
        fig.set_figwidth(5)
        for dim in range(N):
            self.geometric_realization(dim, ax)
    
    def compute(self, max_dist: float = np.inf, max_dim: Optional[int] = None) -> None:
        """
        Compute simplices.
        
        Args:
            max_dist (float): max simplex diameter
            max_dim (int): max simplex dimensionality
        
        Returns:
            None
        """
        if max_dim is None:
            max_dim = self.vertices.shape[0] - 1
        self.simplices.clear()
        self.lower_distance.clear()
        self.simplices[0] = [(i,) for i in range(self.vertices.shape[0])]
        for dim in range(max_dim):
            for i, simplex_a in enumerate(self.simplices[dim]):
                for j, simplex_b in enumerate(self.simplices[dim][i+1:], start=i+1):
                    candidate, lower_distance = self._simplex_intersection(max_dist, simplex_a, simplex_b)
                    if candidate is None or candidate in self.simplices[dim + 1]:
                        continue
                    position = bisect(self.lower_distance[dim + 1], lower_distance)
                    self.simplices[dim + 1].insert(position, candidate)
                    self.lower_distance[dim + 1].insert(position, lower_distance)