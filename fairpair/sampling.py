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
        self._plot_over_time(data=self.comparisons_over_time, y='comparisons')
    
    def plot_success_over_time(self):
        '''Plots the success rate for each node over time, colored by group membership'''
        self._plot_over_time(data=self.success_over_time, y='success')
    
    def _plot_over_time(self, data:pd.DataFrame, y:str):
        '''A helper for plotting stats over time'''
        ax = sns.lineplot(data=data, x='iteration', y=y, hue='minority', units='node', estimator=None)
        ax.legend(ax.get_legend().legendHandles, ['Majority', 'Minority'], title=None, frameon=False)
        plt.setp(ax.lines, alpha=0.3)
        sns.despine()
        plt.show()


class RandomSampling(Sampling):

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
            self._split_and_compare(selected_nodes, k, iteration, seed)


class ProbKnockoutSampling(Sampling):

    def apply(self, iter=1, k=10, min_prob=0.01, seed: int | None = None):
        '''
        Select nodes probabilistically based on their ratio of wins (success rate) so far.

        Parameters
        ----------
        - iter: how many iterations of ProbKnockout sampling to perform
        - k: how often each sampled pair will be compared per iteration
        - min_prob: minimal probability of a node being selected (avoids being stuck at zero)
        - seed: seed for the random number generator
        '''
        rng = np.random.default_rng(seed=seed)
        for iteration in range(iter):
            rates = [rate for _, rate in self.G.success_rates]
            # rescale rates to (0,1)
            max_rate = max(rates)
            min_rate = min(rates)
            if (min_rate != max_rate):
                normalized_success = [(node, max(min_prob, (rate-min_rate)/(max_rate-min_rate))) for node, rate in self.G.success_rates]
            else:
                normalized_success = [(node, 0.5) for node in self.G.nodes] # all node get equal chance of being selected
            selected_nodes = [node for node, prob in normalized_success if rng.binomial(1,prob)]
            self._split_and_compare(selected_nodes, k, iteration, seed)


class GroupKnockoutSampling(Sampling):

    def apply(self, iter=1, k=10, seed: int | None = None):
        '''
        Select nodes probabilistically based on the highest ratio of wins (success rate) in their group (role models) so far.

        Parameters
        ----------
        - iter: how many iterations of ProbKnockout sampling to perform
        - k: how often each sampled pair will be compared per iteration
        - seed: seed for the random number generator
        '''
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
                normalized_success = [(node, minority_rate) if node in self.G.minority_nodes else (node, majority_rate) for node in self.G.nodes]
            else:
                normalized_success = [(node, 0.5) for node in self.G.nodes] # all node get equal chance of being selected
            selected_nodes = [node for node, prob in normalized_success if rng.binomial(1,prob)]
            self._split_and_compare(selected_nodes, k, iteration, seed)


class OversampleMinority(Sampling):

    def apply(self, iter=1, k=10, n=10, p=0.5, seed: int | None = None):
        '''
        Select n nodes randomly, with a share of p nodes from the minority

        Parameters
        ----------
        - iter: how many iterations of ProbKnockout sampling to perform
        - k: how often each sampled pair will be compared per iteration
        - n: how many nodes to sample in each iteration
        - p: share (from n) of minority nodes to be selected for comparison
        - seed: seed for the random number generator
        '''
        rng = np.random.default_rng(seed=seed)
        for iteration in range(iter):
            #from_minority = rng.binomial(n,p) # how many nodes will come from the minority
            from_minority = int(np.ceil(n*p))
            selected_minority = rng.choice(self.G.minority_nodes, from_minority, replace=False)
            selected_majority = rng.choice(self.G.majority_nodes, n-from_minority, replace=False)
            selected_nodes = np.concatenate((selected_minority, selected_majority))
            self._split_and_compare(selected_nodes, k, iteration, seed)