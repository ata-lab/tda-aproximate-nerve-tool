import numpy as np
import matplotlib.pyplot as plt
from itertools import zip_longest
from typing import List
from persim.bottleneck import bottleneck
from persim.visuals import plot_diagrams


def filter_diagrams(diagrams: List[np.array], epsilon: float) -> List[np.array]:
    """
    Remove the points that are epsilon-close to the diagonal of the diagram.
    
    Args:
        diagrams (List[np.array]): persistence diagrams in ripser format
        epsilon (float): filtration threshold
    
    Returns:
        Filtered diagram with points epsilon-close to the diagonal removed.
    """
    filtered_diagram = []
    # iteration over homology levels
    for diagram in diagrams:
        # bottleneck distance to the diagonal for each point
        individual_bottleneck = (diagram[:, 1] - diagram[:, 0]) / np.sqrt(2)
        filtered_diagram.append(diagram[np.argwhere(individual_bottleneck > epsilon).T[0]])
    return filtered_diagram

def full_bottleneck(dgms_a: List[np.array], dgms_b: List[np.array]) -> float:
    """
    Max of pairwise bottleneck distances for each homology level.
    
    Args:
        dgms_a: persistence diagrams in ripser format
        dgms_b: persistence diagrams in ripser format
    
    Returns:
        None
    """
    return max(bottleneck(dgm_a, dgm_b) 
               for dgm_a, dgm_b in zip_longest(dgms_a, dgms_b, fillvalue=np.array([[]]))) 

def plot_two_diagrams(dgms_a: List[np.array], dgms_b: List[np.array]) -> None:
    """
    Plot two persistence diagrams side by side.
    
    Args:
        dgms_a: persistence diagrams in ripser format
        dgms_b: persistence diagrams in ripser format
    
    Returns:
        None
    """
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plot_diagrams(dgms_a)
    plt.subplot(1, 2, 2)
    plot_diagrams(dgms_b)