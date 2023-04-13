import warnings

import numpy as np
import networkx as nx

from .fairgraph import FairPairGraph


def random_list_to_pairs(G:FairPairGraph, nodes:list, seed: int | None = None):
    '''Split a list of nodes from FairPairGraph G into a list of disjoint pairs.'''
    arr = np.array(nodes)
    rng = np.random.default_rng(seed=seed)
    rng.shuffle(arr)
    if (len(arr) % 2 == 1):
        warnings.warn('Deleting one element to create pairs from list of uneven length', stacklevel=2)
        arr = arr[:-1]
    arr = arr.reshape(-1, 2)
    return list(map(tuple, arr))


class Sampling:

    def __init__(self, G:FairPairGraph, split_using=random_list_to_pairs):
        '''
        Initialize the Sampling

        Parameters
        ----------
        - G: the FairPairGraph this Sampling will be applied to
        - split_using: a function that splits lists into disjoint pairs
        '''
        self.G = G
        self.split_using = split_using
        # TODO: add options for logging info (degrees, etc.) while sampling
    
    def get_graph(self):
        '''Returns the FairPairGraph this Sampling is applied to.'''
        return self.G


class RandomSampling(Sampling):

    def __init__(self, G:FairPairGraph, split_using=random_list_to_pairs):
        Sampling.__init__(self, G, split_using)

    def apply(self, iter=1, k=10, p=0.4, seed: int | None = None):
        '''
        Apply random sampling with uniform probability

        Parameters
        ----------
        - iter: how many iterations of random sampling to perform
        - k: how often each sampled pair will be compared per iteration
        - p: probability of a node to be selected for comparison
        - seed: seed for the random number generator
        '''
        for iteration in range(iter):
            rng = np.random.default_rng(seed=seed)
            selected_nodes = [node for node in self.G.nodes if rng.binomial(1,p)]
            pairs = self.split_using(G=self.G, nodes=selected_nodes, seed=seed)
            for (i, j) in pairs:
                self.G.compare_pair(i, j, k, seed=seed)
