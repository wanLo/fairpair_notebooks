from typing import Tuple

import pandas as pd
import networkx as nx
from sklearn.metrics import ndcg_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import rankdata

from .fairgraph import FairPairGraph
from .recovery_baselines import *

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
        Preserves node names and only calculates the ranking on the
        giant strongly connected component.

        Parameters
        ----------
        - rank_using: a function that recovers a ranking (list) from an adjacency matrix
        - **kwargs: keyword arguments to be passed to the ranking function

        Returns
        -------
        - ranking: dict of nodes and their ranking results
        - other_nodes: list of all nodes NOT included in the ranking because they were
          not strongly connected
        '''
        connected_nodes = max(nx.strongly_connected_components(self.G), key=len) # get the giant connected component
        connected_graph = self.G.subgraph(connected_nodes)
        other_nodes = [node for node in self.G.nodes if node not in connected_nodes]
        adjacency = nx.linalg.graphmatrix.adjacency_matrix(connected_graph, weight=self.weight_attr)

        # The GNNRank implementation generally assumes i->j means "i beats j", while we mean the opposite
        adjacency = adjacency.transpose()
        
        ranking = rank_using(adjacency, **kwargs)
        ranking = [float(score.real) if isinstance(score, complex) else float(score) for score in ranking]
        ranking = dict(zip(connected_nodes, ranking)) # nodes might have specific names, so we return a dict
        return ranking, other_nodes
    

    def _print_with_score(self, ranking:dict):

        # sort by ranking score
        ranking = {node: score for node, score in sorted(ranking.items(), key=lambda item: item[1], reverse=True)}

        # print with original score and group membership
        data = []
        for node, rank_score in ranking.items():
            data.append((node, self.G.nodes[node]['score'], rank_score))
        return pd.DataFrame(data, columns=['node', 'orig score', 'rank score'])


    @staticmethod
    def scores_to_rank(ranking:dict) -> dict:
        '''A helper to convert a ranking from scores to ranks'''
        # convert ranking from {node:score} dict to [(node, rank)] list
        rank_data = rankdata(list(ranking.values()))
        rank_data = [int(max(rank_data) - rank) for rank in rank_data] # we want 0 to be the highest rank
        ranks = zip(list(ranking.keys()), rank_data)
        # convert to {node:rank} dict
        return {node: rank for node, rank in ranks}


    def calc_NDCG(self, subgraph:FairPairGraph, ranking:dict, **kwargs) -> float:
        '''
        Calculates the normalized discounted cumulative gain (NDCG)
        for a ranking given initial scores from subgraph

        Parameters
        ----------
        - subgraph: a FairPairGraph subgraph 
        - ranking: dict of nodes and their ranking results
        - **kwargs: keyword arguments to be passed to sklearn's ndcg_score()
        '''
        # extract the ranking scores in the order of subgraph.nodes
        scores_predicted = [ranking[node] for node in subgraph.nodes if node in ranking]
        if len(scores_predicted) < 2:
            raise ValueError('subgraph must have at least 2 nodes contained in ranking to calculate NDCG')
        # extract true scores for the nodes
        scores_true = [score for node, score in subgraph.nodes(data=self.score_attr) if node in ranking]
        return ndcg_score([scores_true], [scores_predicted], **kwargs)


    def calc_MSE(self, subgraph:FairPairGraph, ranking:dict) -> float:
        '''
        Calculates the mean square error (MSE) for a ranking
        (scores normalized to (0,1)) given initial scores from subgraph

        Parameters
        ----------
        - subgraph: a FairPairGraph subgraph 
        - ranking: dict of nodes and their ranking results
        '''
        # extract the ranking scores in the order of subgraph.nodes
        scores_predicted = [ranking[node] for node in subgraph.nodes if node in ranking]
        # normalize to (0,1) to compare to true scores
        scaler = MinMaxScaler().fit([[score] for score in list(ranking.values())]) # shape: (n_samples, n_features=1)
        if len(scores_predicted)>0:
            scores_predicted = scaler.transform([[score] for score in scores_predicted])
        else: return None
        # extract true scores for the nodes
        scores_true = [score for node, score in subgraph.nodes(data=self.score_attr) if node in ranking]
        return mean_squared_error(scores_true, scores_predicted.flatten())
    

    def _extract_ranks(self, subgraph:FairPairGraph, ranking:dict) -> Tuple[list, list]:
        '''A helper to extract ranks (true + predicted from ranking) from nodes in a subgraph'''
        # convert to ranks
        ranking = RankRecovery.scores_to_rank(ranking)
        base_scores = {node: score for node, score in self.G.nodes(data=self.score_attr)}
        base_ranking = RankRecovery.scores_to_rank(base_scores)
        # extract the ranks in the order of subgraph.nodes
        ranks_predicted = [ranking[node] for node in subgraph.nodes if node in ranking]
        ranks_true = [base_ranking[node] for node in subgraph.nodes if node in ranking]
        return ranks_true, ranks_predicted


    def calc_rank_mean_error(self, subgraph:FairPairGraph, ranking:dict) -> float:
        '''
        Calculates the mean error for a ranking given the
        "ground-truth" ranking from initial scores of self.G

        Parameters
        ----------
        - subgraph: a FairPairGraph subgraph of self.G
        - ranking: dict of nodes and their ranking results
        '''
        ranks_true, ranks_predicted = self._extract_ranks(subgraph, ranking)

        errors = [true-predicted for true, predicted in zip(ranks_true, ranks_predicted)]
        return np.mean(errors)
    

    def calc_rank_MSE(self, subgraph:FairPairGraph, ranking:dict) -> float:
        '''
        Calculates the mean squared error (MSE) for a ranking given the
        "ground-truth" ranking from initial scores of self.G

        Parameters
        ----------
        - subgraph: a FairPairGraph subgraph of self.G
        - ranking: dict of nodes and their ranking results
        '''
        ranks_true, ranks_predicted = self._extract_ranks(subgraph, ranking)
        if len(ranks_true) == 0 or len(ranks_predicted) == 0: return None
        return mean_squared_error(ranks_true, ranks_predicted)
    
    def calc_rND(self, subgraph:FairPairGraph, ranking:dict) -> float:
        '''
        Calculates the rND score
        '''