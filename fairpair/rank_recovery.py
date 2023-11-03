from typing import Tuple, Union
from pathlib import Path
import os

import pandas as pd
import networkx as nx
import scipy.sparse as sp

from .fairgraph import FairPairGraph
from .recovery_baselines import *


def fairPageRank(G:FairPairGraph, cutoff=0.4, phi=0.5, path='data/tmp'):
    '''A wrapper for the C++-based implementation of Fairness-Aware PageRank (Tsioutsiouliklis et al., 2021)'''
    # get all edges with weights higher than or equal to cutof
    edges = G.edges(data='weight')
    edges = [(outgoing, incoming) for outgoing, incoming, weight in edges if weight>=cutoff]
    
    # write edgelist
    graphfile_path = Path(path,'out_graph.txt')
    with open(graphfile_path, 'w') as file:
        file.write(f'{len(G.nodes)}\n')
        for edge in edges:
            file.write(f'{edge[0]} {edge[1]}\n')
    
    # write nodelist with groups
    nodes = G.nodes(data='minority')
    nodes = [(node, 1) if minority else (node, 0) for node, minority in nodes]
    community_path = Path(path,'out_community.txt')
    with open(community_path, 'w') as file:
        file.write('2\n') # we always have two groups
        for node in nodes:
            file.write(f'{node[0]} {node[1]}\n')
    
    # write desired community sizes
    sizes_path = Path(path,'sizes.txt')
    with open(sizes_path, 'w') as file:
        file.write(f'0 {1-phi}\n') # priviledged group
        file.write(f'1 {phi}\n') # unpriviledged group
    
    # run the compiled fairPageRank program
    # make sure that pagerank.out is located in the path directory
    dir = os.getcwd()
    os.chdir(path)
    os.system('./pagerank.out -c sizes.txt > /dev/null') # run with muted stdout
    #os.system(f'./residual_optimization.out {phi} > /dev/null') # run with muted stdout
    os.chdir(dir)

    # read the finished file
    #result_path = Path(path,'out_pagerank_pagerank.txt')
    result_path = Path(path,'out_lfpr_p_pagerank.txt')
    #result_path = Path(path,'out_excess_sensitive_pagerank.txt')
    with open(result_path, 'r') as file:
        ranking = file.read().splitlines()
    ranking = [float(score) for score in ranking]

    return ranking


def randomRankRecovery(A: sp.spmatrix, seed: Union[int, None] = None):
    x, y = A.get_shape()
    rng = np.random.default_rng(seed=seed)
    return rng.random(x)


class RankRecovery:

    def __init__(self, G:FairPairGraph, class_attr='minority', weight_attr='weight', score_attr='score'):
        '''
        Initialize the ranking

        Parameters
        ----------
        - G: the FairPairGraph from which a ranking will be recovered
        - class_attr: name of the node attribute to use as a group label
        - weight_attr: name of the edge attribute for storing weights
        - score_attr: name of the node attribute for storing scores
        '''
        self.G = G
        self.class_attr = class_attr
        self.weight_attr = weight_attr
        self.score_attr = score_attr

    
    def apply(self, rank_using=rankCentrality, **kwargs) -> Tuple[dict, list]:
        '''
        Helper for applying a ranking function to a FairPairGraph.
        Preserves node names and calculates the ranking only if weakly connected.

        Parameters
        ----------
        - rank_using: a function that recovers a ranking (list) from an adjacency matrix
            OR 'fairPageRank', if Fairness-Aware PageRank should be applied
        - **kwargs: keyword arguments to be passed to the ranking function

        Returns
        -------
        - ranking: dict of nodes and their ranking results
        - other_nodes: list of all nodes NOT included in giant weakly connected component
        '''
        other_nodes = []
        ranking = None
        if nx.is_weakly_connected(self.G): # only apply ranking recovery if weakly connected
            if rank_using == 'fairPageRank':
                ranking = fairPageRank(self.G, **kwargs)
                ranking = dict(zip(self.G.nodes, ranking))
            else:
                adjacency = nx.linalg.graphmatrix.adjacency_matrix(self.G, weight=self.weight_attr)

                # The GNNRank implementation generally assumes i->j means "i beats j", while we mean the opposite
                adjacency = adjacency.transpose()
                
                ranking = rank_using(adjacency, **kwargs)
                ranking = [float(abs(score)) if isinstance(score, complex) else float(score) for score in ranking]
                ranking = dict(zip(self.G.nodes, ranking)) # nodes might have specific names, so we return a dict
        else:
            connected_nodes = max(nx.strongly_connected_components(self.G), key=len) # get the giant connected component
            other_nodes = [node for node in self.G.nodes if node not in connected_nodes]

        return ranking, other_nodes
    

    def _print_with_score(self, ranking:dict):

        # sort by ranking score
        ranking = {node: score for node, score in sorted(ranking.items(), key=lambda item: item[1], reverse=True)}

        # print with original score and group membership
        data = []
        for node, rank_score in ranking.items():
            data.append((node, self.G.nodes[node]['score'], rank_score))
        return pd.DataFrame(data, columns=['node', 'orig score', 'rank score'])
