import numpy as np
import matplotlib.pyplot as plt
import warnings

from sklearn.metrics import pairwise_distances
from collections import defaultdict
from matplotlib.collections import LineCollection
from typing import Dict, List, DefaultDict, Tuple, Iterable, Optional, Set

from ripser import ripser
from persim.bottleneck import bottleneck
from persim.visuals import plot_diagrams

from .vrcomplex import VietorisRipsComplex


class BallCover:
    """
    Ball Cover with An Approximate Nerve Theorem topological features.
    """
    
    def __init__(self, X: np.array, cover_centers: np.array, r: float = 1.0):
        """
        Args:
            X (np.array): point cloud
            cover_centers (np.array): coordinates of cover element centers
            r (float): coefficient to increase cover element radii
        """
        if r < 1.0:
            raise ValueError("r should not be less than 1.0")
        self._X               = X
        self.cover_centers    = cover_centers
        self.r                = r
        self._radii           = np.zeros(cover_centers.shape[0])
        self.epsilon          = np.inf
        self.furtherst_points = np.zeros_like(cover_centers)
        self._nerve: Optional[VietorisRipsComplex] = None
        self._diagram: Optional[np.array]          = None
    
    def fit(self) -> None:
        """
        Compute epsilon for the cover.
        
        Returns:
            None
        """
        self._compute_ball_radii()
        epsilon = 0.0
        for intersecting in self._find_intersecting_elements():
            intersection = self._find_intersection(intersecting[0], intersecting[1])
            if intersection.shape[0] < 2:
                continue
            epsilon = max(epsilon, self._compute_intersection_epsilon(intersection))
        self.epsilon = epsilon
    
    def _compute_ball_radii(self) -> None:
        """
        Compute radii of cover elements.
        
        Returns:
            None
        """
        distances = pairwise_distances(self._X, self.cover_centers, n_jobs=-1)
        closest_centers = distances.argmin(axis=1)
        for i in range(self.cover_centers.shape[0]):
            current_area = np.argwhere(closest_centers == i)
            current_distances = distances[current_area, i]
            self._radii[i] = current_distances.max()
            self.furtherst_points[i] = self._X[current_area[current_distances.argmax()]]
        self._radii *= self.r
        # Due to radii equality in VR-filtration
        self._radii[:] = self._radii.max()
    
    def _find_intersecting_elements(self) -> np.array:
        """
        Find intersecting cover elements.
        
        Returns:
            np.array of intersecting element pairs
        """
        distances = pairwise_distances(self.cover_centers)
        distances = np.triu(distances)
        # matrix of radii sum (for future VR-complex modification implementation)
        intersection_dist     = np.expand_dims(self._radii, 0) + np.expand_dims(self._radii, 1)
        intersecting_elements = np.argwhere(distances < intersection_dist)
        return intersecting_elements
    
    def _find_intersection(self, cover_element_a: int, cover_element_b: int) -> np.array:
        """
        Find points belonging to intersection of two cover elements.
        
        Args:
            cover_element_a (int): index of cover element
            cover_element_b (int): index of cover element
        
        Returns:
            np.array of points
        """
        distances       = pairwise_distances(self._X, self.cover_centers, n_jobs=-1)
        distances_to_ab = distances[:, [cover_element_a, cover_element_b]]
        radii_ab        = self._radii[[cover_element_a, cover_element_b]]
        in_intersection = np.all(distances_to_ab < radii_ab, axis=1)
        intersection    = self._X[np.argwhere(in_intersection).T[0]]
        return intersection
    
    def _compute_intersection_epsilon(self, intersection: np.array) -> float:
        """
        Compute epsilon for an intersection of cover elements.
        
        Args:
            intersection (np.array): points belonging to intersection
        
        Returns:
            epsilon for the intersection
        """
        diagrams = ripser(intersection)['dgms']
        epsilon = 0.0
        for i, diagram in enumerate(diagrams):
            # bottleneck distance to the diagonal for each point
            individual_bottleneck = (diagram[:, 1] - diagram[:, 0]) / 2
            if (individual_bottleneck == np.inf).sum() > (1 if i == 0 else 0):
                epsilon = np.inf
                warnings.warn("One of the cover component intersections is not epsilon-acyclic: epsilon = inf")
            else:
                epsilon = max(epsilon, individual_bottleneck[np.isfinite(individual_bottleneck)].max(initial=0))
        return epsilon
    
    def plot_cover(self) -> None:
        """
        Plot cover elements.
        
        Returns:
            None
        """
        if self._X.ndim != 2:
            raise ValueError(f"Only 2D data is acceptible to plot but current dimensionality is {self._X.ndim}")
        fig, ax = plt.subplots()
        ax.scatter(self.furtherst_points[:, 0], self.furtherst_points[:, 1])
        ax.scatter(self.cover_centers[:, 0], self.cover_centers[:, 1])
        for neuron, radius in zip(self.cover_centers, self._radii):
            ax.add_patch(plt.Circle(neuron, radius, fill=False))

        intersection = []
        for intersecting_elements in self._find_intersecting_elements():
            intersection.append(self._find_intersection(intersecting_elements[0], intersecting_elements[1]))
        intersection = np.vstack(intersection)
        ax.scatter(intersection[:, 0], intersection[:, 1], alpha=0.2, marker='.')
        ax.set_aspect(1)
    
    def _compute_nerve(self, max_dist: Optional[float] = None, max_dim: Optional[int] = None) -> None:
        """
        Compute nerve of the cover.
        
        Args:
            max_dist (float): max simplex diameter
            max_dim (int): max simplex dimensionality
            
        Returns:
            None
        """
        if self._nerve is None:
            self._nerve = VietorisRipsComplex(self.cover_centers)
            self._nerve.compute(max_dist or np.inf, max_dim)
    
    @property
    def nerve(self) -> VietorisRipsComplex:
        """
        Nerve of the cover.
        """
        if self._nerve is None:
            self._compute_nerve()
        return self._nerve
    
    def _nerve_dim(self, max_dist: Optional[float] = None, max_dim: Optional[int] = None) -> int:
        """
        Get nerve dimensionality.
        
        Args:
            max_dist (float): max simplex diameter
            max_dim (int): max simplex dimensionality
        
        Returns:
            Nerve dimensionality.
        """
        if max_dist is not None:
            self._compute_nerve(max_dist=max_dist, max_dim=max_dim)
        if self._nerve is not None:
            return self._nerve.simplex_dim
        return self.cover_centers.shape[0] - 1
    
    @property
    def error_bound(self) -> float:
        """
        Bottleneck distance error bound for the nerve diagram. 
        """
        return 2 * (1 + min(self._X.ndim, self._nerve_dim())) * self.epsilon
    
    def persistence_diagram(self, maxdim: int = 1) -> np.array:
        """
        Compute persistence diagram of the nerve.
        
        Args:
            maxdim (int): max simplex dimensionality
        
        Returns:
            Persistence diagrams in ripser format.
        """
        return ripser(self.cover_centers, maxdim=maxdim)['dgms']