from functools import cached_property

import numpy as np
import networkx as nx

from .distributions import Distributions

class FairPairGraph(nx.DiGraph):


    def __init__(self):
        nx.DiGraph.__init__(self)


    def generate_groups(self, N:int, Nm: int):
        '''
        Generate nodes, and assign them a binary label

        Parameters
        ----------
        - N: number of nodes
        - Nm: number of minority nodes
        '''
        self.add_nodes_from(np.arange(N))
        self.label_minority(Nm)


    def label_minority(self, Nm: int, attr="minority", random=False, seed: int | None = None):
        '''
        Label a subset of nodes as minority

        Parameters
        ----------
        - Nm: number of minority nodes
        - attr: name of the node attribute to use as a group label
        - random: whether to shuffle nodes before assigning minority attribute
        - seed: seed for the random number generator
        '''
        N = len(self)
        if Nm >= N: raise ValueError('Minority can have at most N-1 nodes.')
        labels = [i >= N - Nm  for i in np.arange(N)]

        if random:
            rng = np.random.default_rng(seed=seed)
            rng.shuffle(labels)
        
        labels_dict = dict(zip(self.nodes, labels))
        nx.function.set_node_attributes(self, labels_dict, attr)


    def group_assign_scores(self, nodes: list, attr="score", distr = Distributions.uniform_distr, **kwargs):
        '''
        Randomly draw scores from a distribution and assign them to the nodes of a subgraph

        Parameters
        ----------
        - nodes: list of nodes to assign scores to
        - attr: name of the node attribute for storing scores
        - distr: distribution function with required keyword argument n (number of nodes)
        - **kwargs: keyword arguments passed to distr function
        '''
        scores_dict = dict(zip(nodes, distr(n = len(nodes), **kwargs)))
        nx.function.set_node_attributes(self, scores_dict, attr)


    def compare_pair(self, i, j, k = 1, node_attr="score", weight_attr="weight", wins_attr="wins", seed: int | None = None):
        '''
        Compares nodes i and j using the BTL-formula, k times (binomial distribution).

        Adds a weighted, directed edge between i and j, or updates the edge if the pair had already been compared before.

        Parameters
        ----------
        - i: the first node to compare
        - j: the second node to compare
        - k: how often to compare the nodes
        - node_attr: name of the node attribute for storing scores
        - weight_attr: name of the edge attribute for storing weights
        - wins_attr: name of the edge attribute for storing #wins
        - seed: seed for the random number generator
        '''
        rng = np.random.default_rng(seed=seed)
        prob = self.nodes[j][node_attr] / (self.nodes[i][node_attr] + self.nodes[j][node_attr])
        wins_j = rng.binomial(k, prob)

        # incorporate the results of previous comparisons
        if (self.has_edge(i, j) and self.has_edge(j, i) and wins_attr in self.edges[(i, j)] and wins_attr in self.edges[(j, i)]):
            wins_j += self.edges[(i,j)][wins_attr]
            k += self.edges[(i,j)][wins_attr] + self.edges[(j,i)][wins_attr]

        edge_i_j = {wins_attr: wins_j, # total wins of j over i, arrows are towards the winners
                    weight_attr: wins_j/k} # compared strength
        self.add_edge(i, j, **edge_i_j)

        edge_j_i = {wins_attr: k - wins_j, # the comparison in the other way
                    weight_attr: 1 - wins_j/k}
        self.add_edge(j, i, **edge_j_i)


    @cached_property
    def minority(self, attr="minority") -> nx.Graph:
        '''Returns a read-only graph view of the minority subgraph, see nx.subgraph_view'''
        return nx.graphviews.subgraph_view(self, filter_node=lambda x: x in self.minority_nodes(attr=attr))


    @cached_property
    def minority_nodes(self, attr="minority") -> list:
        '''Returns a list of minority nodes'''
        # TODO: implement this using nx.classes.reportviews.NodeView
        return [x for x,y in self.nodes(data=attr) if y]


    @cached_property
    def majority(self, attr="minority") -> nx.Graph:
        '''Returns a read-only graph view of the majority subgraph, see nx.subgraph_view'''
        return nx.graphviews.subgraph_view(self, filter_node=lambda x: x in self.majority_nodes(attr=attr))


    @cached_property
    def majority_nodes(self, attr="minority") -> list:
        '''Returns a list of minority nodes'''
        # TODO: implement this using nx.classes.reportviews.NodeView
        return [x for x,y in self.nodes(data=attr) if not y]