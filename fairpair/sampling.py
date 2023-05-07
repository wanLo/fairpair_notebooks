import warnings

import numpy as np
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from .fairgraph import FairPairGraph


def random_list_to_pairs(G:FairPairGraph, nodes:list, seed: int | None = None, warn=True):
    '''Split a list of nodes from FairPairGraph G into a list of disjoint pairs.'''
    arr = np.array(nodes)
    rng = np.random.default_rng(seed=seed)
    rng.shuffle(arr)
    if (len(arr) % 2 == 1):
        if warn: warnings.warn('Deleting one element to create pairs from list of uneven length', stacklevel=2)
        arr = arr[:-1]
    arr = arr.reshape(-1, 2)
    return list(map(tuple, arr))


class Sampling:

    def __init__(self, G:FairPairGraph, split_using=random_list_to_pairs,
                 log_comparisons=False, log_success=False, warn=True):
        '''
        Initialize the Sampling

        Parameters
        ----------
        - G: the FairPairGraph this Sampling will be applied to
        - split_using: a function that splits lists into disjoint pairs
        - log_comparisons: whether to log #comparisons per node after each iteration
        - log_success: whether to log the success rate of each node after each iteration
        - warn: whether to throw warnings
        '''
        self.G = G
        self.split_using = split_using
        self.warn = warn

        self.comparisons_over_time = pd.DataFrame()
        self.log_comparisons = log_comparisons
        self.success_over_time = pd.DataFrame()
        self.log_success = log_success
    
    def get_graph(self):
        '''Returns the FairPairGraph this Sampling is applied to.'''
        return self.G
    
    def _split_and_compare(self, selected_nodes:list, k:int, iteration=0, seed: int | None = None):
        '''A helper for running k comparisons on selected nodes'''
        pairs = self.split_using(G=self.G, nodes=selected_nodes, seed=seed, warn=self.warn)
        for (i, j) in pairs:
            self.G.compare_pair(i, j, k, seed=seed)
        # logging
        if self.log_comparisons:
            for node, comparisons in self.G.comparisons:
                df = pd.DataFrame({'node': node, 'minority': self.G.nodes[node]['minority'], 'iteration': iteration, 'comparisons': comparisons}, index=[0])
                self.comparisons_over_time = pd.concat([self.comparisons_over_time, df], ignore_index=True)
        if self.log_success:
            for node, success in self.G.success_rates:
                df = pd.DataFrame({'node': node, 'minority': self.G.nodes[node]['minority'], 'iteration': iteration, 'success': success}, index=[0])
                self.success_over_time = pd.concat([self.success_over_time, df], ignore_index=True)

    def plot_comparisons_over_time(self):
        '''Plots the #comparisons for each node over time, colored by group membership'''
        self._plot_over_time(data=self.comparisons_over_time, y='comparisons', units='node', estimator=None)
    
    def plot_success_over_time(self):
        '''Plots the success rate for each node over time, colored by group membership'''
        self._plot_over_time(data=self.success_over_time, y='success')
    
    def _plot_over_time(self, data:pd.DataFrame, y:str, **kwargs):
        '''A helper for plotting stats over time'''
        ax = sns.lineplot(data=data, x='iteration', y=y, hue='minority', **kwargs)
        ax.legend(ax.get_legend().legendHandles, ['Majority', 'Minority'], title=None, frameon=False)
        plt.setp(ax.lines, alpha=0.3)
        sns.despine()
        plt.show()


class RandomSampling(Sampling):

    def apply(self, iter=1, k=10, f=0.2, seed: int | None = None):
        '''
        Apply random sampling with uniform probability

        Parameters
        ----------
        - iter: how many iterations of random sampling to perform
        - k: how often each sampled pair will be compared per iteration
        - f: fraction of nodes to sample in each iteration
        - seed: seed for the random number generator
        '''
        n = int(len(self.G)*f) # how many nodes to sample
        rng = np.random.default_rng(seed=seed)
        for iteration in range(iter):
            #selected_nodes = [node for node in self.G.nodes if rng.binomial(1,p)]
            selected_nodes = rng.choice(self.G.nodes, n, replace=False)
            self._split_and_compare(selected_nodes, k, iteration, seed)


class ProbKnockoutSampling(Sampling):

    def apply(self, iter=1, k=10, f=0.2, min_prob=0.001, seed: int | None = None):
        '''
        Select nodes probabilistically based on their ratio of wins (success rate) so far.

        Parameters
        ----------
        - iter: how many iterations of ProbKnockout sampling to perform
        - k: how often each sampled pair will be compared per iteration
        - f: fraction of nodes to sample in each iteration
        - min_prob: minimal probability of a node being selected (avoids being stuck at zero)
        - seed: seed for the random number generator
        '''
        n = int(len(self.G)*f) # how many nodes to sample
        rng = np.random.default_rng(seed=seed)
        for iteration in range(iter):
            rates = [rate for _, rate in self.G.success_rates]
            # rescale rates to (0,1)
            max_rate = max(rates)
            min_rate = min(rates)
            if (min_rate != max_rate):
                #normalized_rates = [max(min_prob, (rate-min_rate)/(max_rate-min_rate)) for rate in rates] # must be in (min_prob, 1)
                normalized_rates = [(rate-min_rate)/(max_rate-min_rate)*(1-min_prob)+min_prob for rate in rates] # min-max scaler
                normalized_rates = [rate/sum(normalized_rates) for rate in normalized_rates] # must sum to 1
                selected_nodes = rng.choice(self.G.nodes, n, replace=False, p=normalized_rates)
            else:
                # all node get equal chance of being selected
                selected_nodes = rng.choice(self.G.nodes, n, replace=False)
            #selected_nodes = [node for node, prob in normalized_success if rng.binomial(1,prob)]
            self._split_and_compare(selected_nodes, k, iteration, seed)


class GroupKnockoutSampling(Sampling):

    def apply(self, iter=1, k=10, f=0.2, seed: int | None = None):
        '''
        Select nodes probabilistically based on the highest ratio of wins (success rate) in their group (role models) so far.

        Parameters
        ----------
        - iter: how many iterations of GroupKnockout sampling to perform
        - k: how often each sampled pair will be compared per iteration
        - f: fraction of nodes to sample in each iteration
        - seed: seed for the random number generator
        '''
        n = int(len(self.G)*f) # how many nodes to sample
        rng = np.random.default_rng(seed=seed)
        for iteration in range(iter):
            # success rates per group
            minority_success = [rate for node, rate in self.G.success_rates if node in self.G.minority_nodes]
            majority_success = [rate for node, rate in self.G.success_rates if node in self.G.majority_nodes]
            # highest success per group
            minority_rate = max(minority_success)
            majority_rate = max(majority_success)
            # overall highest and lowest success
            max_rate = max([minority_rate, majority_rate])
            min_rate = min([min(minority_success), min(majority_success)])
            if (min_rate != max_rate):
                # normalized success rates per group membership
                minority_rate = (minority_rate-min_rate)/(max_rate-min_rate)
                majority_rate = (majority_rate-min_rate)/(max_rate-min_rate)
                normalized_rates = [minority_rate if node in self.G.minority_nodes else majority_rate for node in self.G.nodes]
                normalized_rates = [rate/sum(normalized_rates) for rate in normalized_rates] # must sum to 1
                selected_nodes = rng.choice(self.G.nodes, n, replace=False, p=normalized_rates)
            else:
                # all node get equal chance of being selected
                selected_nodes = rng.choice(self.G.nodes, n, replace=False)
            #selected_nodes = [node for node, prob in normalized_success if rng.binomial(1,prob)]
            self._split_and_compare(selected_nodes, k, iteration, seed)


class OversampleMinority(Sampling):

    def apply(self, iter=1, k=10, f=0.2, p=0.5, seed: int | None = None):
        '''
        Select n nodes randomly, with a share of p nodes from the minority

        Parameters
        ----------
        - iter: how many iterations of minority oversampling to perform
        - k: how often each sampled pair will be compared per iteration
        - f: fraction of nodes to sample in each iteration
        - p: share (from n) of minority nodes to be selected for comparison
        - seed: seed for the random number generator
        '''
        n = int(len(self.G)*f) # how many nodes to sample
        rng = np.random.default_rng(seed=seed)
        for iteration in range(iter):
            #from_minority = rng.binomial(n,p) # how many nodes will come from the minority
            from_minority = int(np.ceil(len(self.G)*f*p))
            selected_minority = rng.choice(self.G.minority_nodes, from_minority, replace=False)
            selected_majority = rng.choice(self.G.majority_nodes, n-from_minority, replace=False)
            selected_nodes = np.concatenate((selected_minority, selected_majority))
            self._split_and_compare(selected_nodes, k, iteration, seed)


class StargraphSampling(Sampling):

    def apply(self, iter=1, k=10, f=0.2, node: object = 0, node_prob: float | None = 1.0, seed: int | None = None):
        '''
        Select edges randomly, but for each pair, a designated `node` has a higher chance of being selected.
        In the default case of `node_prob=1.0`, this creates a star graph.
        The lower `node_prob`, the closer the graph is to a random graph.

        Parameters
        ----------
        - iter: how many iterations of Stargraph sampling to perform
        - k: how often each sampled pair will be compared per iteration
        - f: fraction of nodes to sample in each iteration
        - node: identifier of the designated node
        - node_prob: probability of selecting the designated node for comparison.
            Set `node_prob=None` to give the designated node the same chance as all other nodes.
        - seed: seed for the random number generator
        '''
        n = int(len(self.G)*f) # how many nodes to sample
        rng = np.random.default_rng(seed=seed)
        if node_prob is not None:
            others_prob = float(1-node_prob)/float(len(self.G)-1) # probability for other nodes to be selected for comparison
            probs = [node_prob if n==node else others_prob for n in self.G.nodes]
        for iteration in range(iter): 
            pairs = []
            for i in range(n//2): # number of pairs to create
                first_node = second_node = None
                while first_node == second_node: # no self-loops
                    if node_prob >= 1: first_node = node # always connect to the center
                    elif node_prob is None: first_node = rng.choice(self.G.nodes, size=1)[0] # with replacement since size=1
                    else: first_node = rng.choice(self.G.nodes, size=1, p=probs)[0] # different prob. for designated node
                    second_node = rng.choice(self.G.nodes, size=1)[0]
                    # don't compare the same pair multiple times per iteration
                    if ((first_node, second_node) in pairs or (second_node, first_node) in pairs):
                        first_node = second_node = None
                    else: pairs.append((first_node, second_node))
                self.G.compare_pair(first_node, second_node, k, seed=seed)
