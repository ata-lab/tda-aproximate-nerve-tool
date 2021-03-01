import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import pairwise_distances
from collections import defaultdict
from matplotlib.collections import LineCollection
from typing import Dict, List, DefaultDict, Tuple, Iterable, Optional, Set

from .vrcomplex import VietorisRipsComplex


class GrowingNeuralGas:
    """
    Implementation of the Growing Neural Gas model.
    """
    
    def __init__(self, epsilon_winner=0.2, epsilon_neighbor=0.05, birth_error_relief=0.5, birth_period=5, max_age=30):
        """
        Args:
            epsilon_winner (float): a coefficient determining how much the winner is affected by moving
            epsilon_neighbor (float): a coefficient determining how much winner neighbors are affected by moving
            birth_error_relief (float): a coefficient determining to what extent local error of the two neurons adjacent to the new one should be reduced
            birth_period (int): a neuron will be created once in *birth_period* epochs
            max_age (int): how many iterations are needed to consider an edge redundant
        """
        self._reinit(0.0, 1.0, (2,))
        self.epsilon_winner     = epsilon_winner
        self.epsilon_neighbor   = epsilon_neighbor
        self.birth_error_relief = birth_error_relief
        self.birth_period       = birth_period
        self.max_age            = max_age
    
    def fit(self, X: np.array, y: Optional[np.array] = None, epochs: int = 50) -> None:
        """
        Fit the model to the data.
        
        Args:
            X (np.array): point cloud
            y (np.array): for compatibility
            epochs (int): the number of epochs
        
        Returns:
            None
        """
        self._reinit(X.mean(), X.std(), X.shape[1:])
        
        for epoch in range(1, epochs):
            for point in X:
                # error is the distance from closest to the point
                closest_neuron, second_closest, error = self._get_winners(point)
                # accumulating the local error
                self._errors[closest_neuron]         += error ** 2
                # move neurons closer to the point
                self._move_neurons(closest_neuron, second_closest, point)
                # deleting inactive edges
                for dead_edge in [edge for edge, age in self._edge_age.items() if age > self.max_age]:
                    self._remove_edge(dead_edge)            
            if epoch % self.birth_period == 0:
                self._create_neuron()
    
    def _reinit(self, mean: np.array, std: np.array, dims: Tuple[int, ...]) -> None:
        """
        Reinintialize the fields.
        
        Args:
            mean (np.array): mean of the point cloud
            std (np.array): std of the point cloud
            dims (Tuple[int, ...]): dimensionality of the point cloud
        
        Returns:
            None
        """
        # neuron weights ("coordinates" in the space)
        self.neurons    = np.random.normal(loc=mean, scale=std, size=(2, *dims))
        # local errors for each neuron
        self._errors    = [0.0, 0.0]
        self._neighbors = defaultdict(set, {    # {neuron: {neighbors}}
            0: {1},
            1: {0},
        })
        self._edge_age = {     # {(neuron_0, neuron_1): age of the edge}
            (0, 1): 0,
        }
        self._neuron_counter = 2
    
    def _get_winners(self, point: np.array) -> Tuple[int, int, float]:
        """
        Find the closest and second closest to the point neurons and the distance from the closest one.
        
        Args:
            point (np.array): current point from the point cloud
        
        Returns:
            (closest neuron, second closest neuron, distance from the closest to the point)
        """
        distances      = pairwise_distances([point], self.neurons)[0]
        closest_neuron = distances.argmin()
        error          = distances[closest_neuron]
        distances[closest_neuron] = np.inf
        second_closest = distances.argmin()
        return closest_neuron, second_closest, error

    def _move_neurons(self, closest: int, second_closest: int, point: np.array) -> None:
        """
        Move neurons closer to the point.
        
        Args:
            closest (int): the closest to the point neuron
            second_closest (int): second closest to the point neuron
            point (np.array): a point from the point cloud
        
        Returns:
            None
        """
        # move the closest
        self.neurons[closest] += (point - self.neurons[closest]) * self.epsilon_winner
        # create edge (closest, second_closest) if required
        if second_closest not in self._neighbors[closest]:
            self._create_edge(closest, second_closest, age=-1)    
        # move neighbors of the closest, adjust edge ages
        for neighbor in self._neighbors[closest]:
            index = min(closest, neighbor), max(closest, neighbor)
            self._edge_age[index] += 1
            self.neurons[neighbor] += (point - self.neurons[neighbor]) * self.epsilon_neighbor
        # reset the (closest, second_closest) edge age
        self._edge_age[tuple(sorted((closest, second_closest)))] = 0

    def _create_edge(self, neuron_a: int, neuron_b: int, *, age: Optional[int] = None) -> None:
        """
        Create an edge between two neurons.
        
        Args:
            neuron_a (int): one neuron
            neuron_b (int): another neuron
            age (int): age of the edge
        
        Returns:
            None
        """
        index = min(neuron_a, neuron_b), max(neuron_a, neuron_b)
        self._neighbors[neuron_a].add(neuron_b)
        self._neighbors[neuron_b].add(neuron_a)
        self._edge_age[index] = age or 0

    def _remove_edge_safe(self, neuron_a: int, neuron_b: int) -> None:
        """
        Remove an edge, do not check neurons.
        
        Args:
            neuron_a (int): one neuron
            neuron_b (int): another neuron
        
        Returns:
            None
        """
        index = min(neuron_a, neuron_b), max(neuron_a, neuron_b)
        self._edge_age.pop(index)
        self._neighbors[neuron_a].remove(neuron_b)
        self._neighbors[neuron_b].remove(neuron_a)

    def _remove_edge(self, edge: Tuple[int, int]) -> None:
        """
        Remove an edge, check neurons.
        
        Args:
            edge (Tuple[int, int]): (neuron_a, neuron_b) edge
        
        Returns:
            None
        """
        self._edge_age.pop(edge)
        neuron_a, neuron_b = sorted(edge)
        self._neighbors[neuron_a].remove(neuron_b)
        self._neighbors[neuron_b].remove(neuron_a)
        
        # remove neurons if they have no neighbors
        if len(self._neighbors[neuron_b]) == 0:
            self._remove_neuron(neuron_b)         # neuron_b > neuron_a => neuron_a is still coherent
        if len(self._neighbors[neuron_a]) == 0:
            self._remove_neuron(neuron_a)

    def _create_neuron(self) -> None:
        """
        Create a new neuron.
        
        Returns:
            None
        """
        # a neuron with the greatest local error
        worst               = max(range(len(self._errors)), key = lambda n: self._errors[n])     # argmax(errors)
        second_worst        = max(self._neighbors[worst],   key = lambda n: self._errors[n]) 
        # between the worst and second_worst
        new_neuron_position = (self.neurons[worst] + self.neurons[second_worst]) / 2
        self.neurons        = np.vstack([self.neurons, new_neuron_position])

        # reduce worst and second_worst local errors
        self._errors[worst]        *= self.birth_error_relief
        self._errors[second_worst] *= self.birth_error_relief

        new_neuron               = self._neuron_counter
        self._neuron_counter    += 1
        self._errors.append(self._errors[worst])

        # replace (worst, second_worst) edge with edges to the new neuron
        self._remove_edge_safe(worst, second_worst)
        self._create_edge(worst, new_neuron)
        self._create_edge(second_worst, new_neuron)

    def _remove_neuron(self, neuron: int) -> None:
        """
        Remove a neuron.
        
        Args:
            neuron: the neuron to be removed
        
        Returns:
            None
        """
        # assuming len(self._neighbors[neuron]) == 0
        self.neurons = np.delete(self.neurons, neuron, axis=0)
        del self._errors[neuron]
        self._neighbors.pop(neuron)
        # correcting neuron indices
        self._edge_age = {
            tuple(i - (1 if i > neuron else 0) for i in edge): age 
            for edge, age in self._edge_age.items()
        }
        self._neighbors = defaultdict(set, {
            n - (1 if n > neuron else 0): {
                x - (1 if x > neuron else 0) for x in n_set
            } 
            for n, n_set in self._neighbors.items()
        })
        self._neuron_counter -= 1