from typing import Tuple

import numpy as np
from sklearn.metrics import ndcg_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import rankdata

from .fairgraph import FairPairGraph

def scores_to_rank(ranking:dict) -> dict:
    '''A helper to convert a ranking from scores to ranks'''
    # convert ranking from {node:score} dict to [(node, rank)] list
    rank_data = rankdata(list(ranking.values()))
    rank_data = [int(max(rank_data) - rank) for rank in rank_data] # we want 0 to be the highest rank
    ranks = zip(list(ranking.keys()), rank_data)
    # convert to {node:rank} dict
    return {node: rank for node, rank in ranks}


##### (Group-Conditioned) Accuracy #####


def NDCG(subgraph:FairPairGraph, ranking:dict, score_attr='score', **kwargs) -> float:
    '''
    Calculates the normalized discounted cumulative gain (NDCG)
    for a ranking given initial scores from subgraph

    Parameters
    ----------
    - subgraph: a FairPairGraph subgraph 
    - ranking: dict of nodes and their ranking results
    - score_attr: name of the node attribute for storing scores
    - **kwargs: keyword arguments to be passed to sklearn's ndcg_score()
    '''
    # extract the ranking scores in the order of subgraph.nodes
    scores_predicted = [ranking[node] for node in subgraph.nodes if node in ranking]
    if len(scores_predicted) < 2:
        raise ValueError('subgraph must have at least 2 nodes contained in ranking to calculate NDCG')
    # extract true scores for the nodes
    scores_true = [score for node, score in subgraph.nodes(data=score_attr) if node in ranking]
    return ndcg_score([scores_true], [scores_predicted], **kwargs)


def MSE(subgraph:FairPairGraph, ranking:dict, score_attr='score') -> float:
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
    scores_true = [score for node, score in subgraph.nodes(data=score_attr) if node in ranking]
    return mean_squared_error(scores_true, scores_predicted.flatten())


def _extract_ranks(graph:FairPairGraph, ranking:dict, subgraph:FairPairGraph | None = None, score_attr='score') -> Tuple[list, list]:
    '''A helper to extract ranks (true + predicted from ranking) from nodes in a subgraph'''
    if subgraph is None: subgraph = graph
    # convert to ranks
    ranking = scores_to_rank(ranking)
    base_scores = {node: score for node, score in graph.nodes(data=score_attr)}
    base_ranking = scores_to_rank(base_scores)
    # extract the ranks in the order of subgraph.nodes
    ranks_predicted = [ranking[node] for node in subgraph.nodes if node in ranking]
    ranks_true = [base_ranking[node] for node in subgraph.nodes if node in ranking]
    return ranks_true, ranks_predicted


def rank_mean_error(graph:FairPairGraph, ranking:dict, subgraph:FairPairGraph | None = None) -> float:
    '''
    Calculates the mean error for a ranking given the
    "ground-truth" ranking from initial scores of `graph`

    Parameters
    ----------
    - graph: the full FairPairGraph for ground-truth ranks
    - ranking: dict of nodes and their ranking results
    - subgraph: a FairPairGraph subgraph of `graph`, or identical to `graph` if None
    '''
    if subgraph is None: subgraph = graph
    ranks_true, ranks_predicted = _extract_ranks(graph, ranking, subgraph)
    if len(ranks_true) == 0 or len(ranks_predicted) == 0: return None
    errors = [true-predicted for true, predicted in zip(ranks_true, ranks_predicted)]
    return np.mean(errors)


def rank_MSE(graph:FairPairGraph, ranking:dict, subgraph:FairPairGraph | None = None) -> float:
    '''
    Calculates the mean squared error (MSE) for a ranking given the
    "ground-truth" ranking from initial scores of `graph`

    Parameters
    ----------
    - graph: the full FairPairGraph for ground-truth ranks
    - ranking: dict of nodes and their ranking results
    - subgraph: a FairPairGraph subgraph of `graph`, or identical to `graph` if None
    '''
    if subgraph is None: subgraph = graph
    ranks_true, ranks_predicted = _extract_ranks(graph, ranking, subgraph)
    if len(ranks_true) == 0 or len(ranks_predicted) == 0: return None
    return mean_squared_error(ranks_true, ranks_predicted)


##### Group-Representation #####


def rND(graph:FairPairGraph, ranking:dict, subgraph:FairPairGraph | None = None) -> float:
    '''
    Calculates the rND score
    '''