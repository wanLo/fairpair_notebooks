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
        rng = np.random.default_rng(seed=seed)
        for iteration in range(iter):
            selected_nodes = [node for node in self.G.nodes if rng.binomial(1,p)]
            pairs = self.split_using(G=self.G, nodes=selected_nodes, seed=seed)
            for (i, j) in pairs:
                self.G.compare_pair(i, j, k, seed=seed)


class ProbKnockoutSampling(Sampling):

    def __init__(self, G:FairPairGraph, split_using=random_list_to_pairs):
        Sampling.__init__(self, G, split_using)

    def apply(self, iter=1, k=10, seed: int | None = None):
        '''
        In each round, randomly pair nodes which are selected probabilistically based on their ratio of wins (success rate) so far.

        Parameters
        ----------
        - iter: how many iterations of ProbKnockout sampling to perform
        - k: how often each sampled pair will be compared per iteration
        - seed: seed for the random number generator
        '''
        rng = np.random.default_rng(seed=seed)
        for iteration in range(iter):
            selected_nodes = [node for node, prob in self.G.success_rates if rng.binomial(1,prob)]
            pairs = self.split_using(G=self.G, nodes=selected_nodes, seed=seed)
            for (i, j) in pairs:
                self.G.compare_pair(i, j, k, seed=seed)


class OversampleMinority(Sampling):

    def __init__(self, G:FairPairGraph, split_using=random_list_to_pairs):
        Sampling.__init__(self, G, split_using)

    def apply(self, iter=1, k=10, n=10, p=0.5, seed: int | None = None):
        '''
        For each node to select, select either group with probability p. Then select a node from this group randomly.

        Parameters
        ----------
        - iter: how many iterations of ProbKnockout sampling to perform
        - k: how often each sampled pair will be compared per iteration
        - n: how many nodes to sample in each iteration
        - p: probability of a minority node to be selected for comparison
        - seed: seed for the random number generator
        '''
        rng = np.random.default_rng(seed=seed)
        for iteration in range(iter):
            from_minority = rng.binomial(n,p) # how many nodes will come from the minority
            selected_minority = rng.choice(self.G.minority_nodes, from_minority, replace=False)
            selected_majority = rng.choice(self.G.majority_nodes, n-from_minority, replace=False)
            selected_nodes = np.concatenate((selected_minority, selected_majority))
            pairs = self.split_using(G=self.G, nodes=selected_nodes, seed=seed)
            for (i, j) in pairs:
                self.G.compare_pair(i, j, k, seed=seed)