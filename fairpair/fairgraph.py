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


    def label_minority(self, Nm: int, attribute="minority", random=False, seed: int | None = None):
        '''
        Label a subset of nodes as minority

        Parameters
        ----------
        - Nm: number of minority nodes
        - attribute: name of the node attribute to use as a group label
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
        nx.function.set_node_attributes(self, labels_dict, attribute)


    def group_assign_scores(self, nodes: list, attribute="score", distr = Distributions.uniform_distr, **kwargs):
        '''
        Randomly draw scores from a distribution and assign them to the nodes of a subgraph

        Parameters
        ----------
        - nodes: list of nodes to assign scores to
        - attribute: name of the node attribute for storing scores
        - distr: distribution function with required keyword argument n (number of nodes)
        - **kwargs: keyword arguments passed to distr function
        '''
        scores_dict = dict(zip(nodes, distr(n = len(nodes), **kwargs)))
        nx.function.set_node_attributes(self, scores_dict, attribute)


    @cached_property
    def minority(self, attribute="minority") -> nx.Graph:
        '''Returns a read-only graph view of the minority subgraph, see nx.subgraph_view'''
        return nx.graphviews.subgraph_view(self, filter_node=lambda x: x in self.minority_nodes(attribute=attribute))


    @cached_property
    def minority_nodes(self, attribute="minority") -> list:
        '''Returns a list of minority nodes'''
        # TODO: implement this using nx.classes.reportviews.NodeView
        return [x for x,y in self.nodes(data=attribute) if y]


    @cached_property
    def majority(self, attribute="minority") -> nx.Graph:
        '''Returns a read-only graph view of the majority subgraph, see nx.subgraph_view'''
        return nx.graphviews.subgraph_view(self, filter_node=lambda x: x in self.majority_nodes(attribute=attribute))


    @cached_property
    def majority_nodes(self, attribute="minority") -> list:
        '''Returns a list of minority nodes'''
        # TODO: implement this using nx.classes.reportviews.NodeView
        return [x for x,y in self.nodes(data=attribute) if not y]