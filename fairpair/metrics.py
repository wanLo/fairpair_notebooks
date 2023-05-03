from typing import Tuple

import numpy as np
from sklearn.metrics import ndcg_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
import networkx as nx
import pandas as pd

from .fairgraph import FairPairGraph

def scores_to_rank(ranking:dict) -> dict:
    '''A helper to convert a ranking from scores to ranks'''
    # convert ranking from {node:score} dict to [(node, rank)] list
    rank_data = stats.rankdata(list(ranking.values()))
    rank_data = [int(max(rank_data) - rank) for rank in rank_data] # we want 0 to be the highest rank
    ranks = zip(list(ranking.keys()), rank_data)
    # convert to {node:rank} dict
    return {node: rank for node, rank in ranks}


##### (Group-Conditioned) Accuracy #####


def NDCG(subgraph:FairPairGraph, ranking:dict, score_attr='score', **kwargs) -> float:
    '''
    Calculates the normalized discounted cumulative gain (NDCG)
    for a `ranking` given initial scores from `subgraph`

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
    Calculates the mean square error (MSE) for scores from `ranking`,
    normalized to (0,1), given initial scores from `subgraph`.

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
    "ground-truth" ranking from initial scores of `graph`.

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
    "ground-truth" ranking from initial scores of `graph`.

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


def spearmanr(graph:FairPairGraph, ranking:dict, subgraph:FairPairGraph | None = None) -> float:
    '''
    Calculates Spearman's correlation coefficient for a `ranking` given the
    "ground-truth" ranking from initial scores of `graph`.

    Parameters
    ----------
    - graph: the full FairPairGraph for ground-truth ranks
    - ranking: dict of nodes and their ranking results
    - subgraph: a FairPairGraph subgraph of `graph`, or identical to `graph` if None
    '''
    if subgraph is None: subgraph = graph
    ranks_true, ranks_predicted = _extract_ranks(graph, ranking, subgraph)
    if len(ranks_true) == 0 or len(ranks_predicted) == 0: return None

    # This doesn't work, because it also converts to ranks first
    #return stats.spearmanr(ranks_true, ranks_predicted)

    return stats.pearsonr(ranks_true, ranks_predicted)


def weighted_tau(graph:FairPairGraph, ranking:dict, subgraph:FairPairGraph | None = None, score_attr='score') -> float:
    '''
    Calculates weighted Kendall tau for a `ranking` given the
    "ground-truth" ranking from initial scores of `graph`.
    See Negahban et al. (2012) for the implementation irrespective of groups.

    Parameters
    ----------
    - graph: the full FairPairGraph for ground-truth ranks
    - ranking: dict of nodes and their ranking results
    - subgraph: a FairPairGraph subgraph of `graph`, or identical to `graph` if None
    - score_attr: name of the node attribute for storing scores
    '''
    if subgraph is None: subgraph = graph

    # get rankings as dicts
    ranking = scores_to_rank(ranking)
    base_scores = {node: score for node, score in graph.nodes(data=score_attr)}

    # sum up weights of discordant pairs
    n_pairs = 0
    discordant_sum = 0
    worst_case_sum = 0
    subgraph_nodes = list(subgraph.nodes) # must faster to do this only once
    complementary_nodes = [node for node in graph.nodes if node not in subgraph.nodes]
    for i in subgraph_nodes:
        # consider within-group pairs till before the diagonal
        # and pairs with complementary nodes only one-way
        for j in subgraph_nodes[:i] + complementary_nodes:
            n_pairs += 1
            diff = (base_scores[i]-base_scores[j]) ** 2
            worst_case_sum += diff
            if (base_scores[i]-base_scores[j])*(ranking[i]-ranking[j]) > 0:
                discordant_sum += diff
    
    # The original formula uses Eucledian norm of weights for normalization.
    # But this is not compatible with subgraphs and yields a weird result range.
    # normed_weights = np.linalg.norm([weight for _, weight in subgraph.nodes(data=score_attr)])
    # return (discordant_sum / (2 * n_pairs * (normed_weights ** 2))) ** 0.5
    
    # We use a worst-case sum (all pairs are discordant) for normalization instead
    return (discordant_sum / worst_case_sum) ** 0.5


##### Group-Representation #####


def rND(graph:FairPairGraph, ranking:dict, subgraph:FairPairGraph | None = None) -> float:
    '''
    Calculates the rND score
    '''