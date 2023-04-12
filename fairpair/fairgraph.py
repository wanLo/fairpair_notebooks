from functools import cached_property

import numpy as np
import networkx as nx

class FairGraph(nx.DiGraph):


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


    def label_minority(self, Nm: int, attribute="Minority", random=False, seed=None):
        '''
        Label a subset of nodes as minority

        Parameters
        ----------
        - Nm: number of minority nodes
        - attribute: name of the node attribute
        - random: whether to shuffle nodes before assigning minority attribute
        - seed: seed for the random number generator
        '''
        N = len(self)
        if Nm >= N: raise ValueError('Minority can have at most N-1 nodes.')
        labels = [i >= N - Nm  for i in np.arange(N)]

        if random:
            rng = np.random.default_rng(seed=seed)
            rng.shuffle(labels)
        
        labels_dict = dict(zip(self.nodes,labels))
        nx.set_node_attributes(self, labels_dict, attribute)


    @cached_property
    def minority(self, attribute="Minority") -> nx.Graph:
        '''
        Get the minority subgraph

        Parameters
        ----------
        - attribute: how the minority is labeled

        Returns
        -------
        - graph: a read-only graph view, see nx.subgraph_view
        '''
        return nx.graphviews.subgraph_view(self, filter_node=lambda x: x in self.minority_nodes(attribute=attribute))


    @cached_property
    def minority_nodes(self, attribute="Minority") -> list:
        '''
        Get the minority nodes

        Parameters
        ----------
        - attribute: how the minority is labeled

        Returns
        -------
        - nodes: a list of minority nodes
        '''
        # TODO: implement this using nx.classes.reportviews.NodeView
        return [x for x,y in self.nodes(data=attribute) if y]