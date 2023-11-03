from typing import Tuple, Union, List

import numpy as np
from sklearn.metrics import ndcg_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
import networkx as nx
import pandas as pd

from .fairgraph import FairPairGraph

def scores_to_rank(ranking:dict, invert=True) -> dict:
    '''A helper to convert a ranking from scores to ranks'''
    # convert ranking from {node:score} dict to {node:rank} dict
    rank_data = stats.rankdata(list(ranking.values()), method='ordinal') # use ordinal method to avoid same rank for ties
    if invert:
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


def _extract_ranks(graph:FairPairGraph, ranking:dict, subgraph: Union[FairPairGraph, None] = None, score_attr='score') -> Tuple[list, list]:
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


def rank_mean_error(graph:FairPairGraph, ranking:dict, subgraph: Union[FairPairGraph, None] = None) -> float:
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


def rank_MSE(graph:FairPairGraph, ranking:dict, subgraph: Union[FairPairGraph, None] = None) -> float:
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


def spearmanr(graph:FairPairGraph, ranking:dict, subgraph: Union[FairPairGraph, None] = None) -> float:
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


def weighted_tau(graph:FairPairGraph, ranking:dict, subgraph: Union[FairPairGraph, None] = None, score_attr='skill') -> float:
    '''
    Calculates the weighted Kemedy distance for a `ranking` given the
    "ground-truth" ranking from initial scores of `graph`.
    Adapted from Negahban et al. (2012)'s version of the weighted Kemedy distance.

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
    discordant_sum = 0
    worst_case_sum = 0
    subgraph_nodes = list(subgraph.nodes) # much faster to do this only once
    complementary_nodes = [node for node in graph.nodes if node not in subgraph.nodes]
    for num, i in enumerate(subgraph_nodes):
        # consider within-group pairs till before the diagonal
        # and pairs with complementary nodes only one-way
        for j in subgraph_nodes[:num] + complementary_nodes:
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


def weighted_tau_nodes(base_scores:dict, ranking:dict, subgraph_nodes:list, complementary_nodes:list) -> float:
    '''
    A faster version of the weighted Kemeny distance, operating directly on the score/ranking dicts

    Parameters
    ----------
    - base_scores: a {node:score} dictionary based on ground-truth scores
    - ranking: a {node:rank} dictionary based on the current ranking
    - subgraph_nodes: nodes from a subgraph for which to calculate the Kemeny distance
    '''

    # get rankings as dicts
    ranking = scores_to_rank(ranking)

    # sum up weights of discordant pairs
    discordant_sum = 0
    worst_case_sum = 0
    for num, i in enumerate(subgraph_nodes):
        # consider within-group pairs till before the diagonal
        # and pairs with complementary nodes only one-way
        for j in subgraph_nodes[:num] + complementary_nodes:
            diff = (base_scores[i]-base_scores[j]) ** 2
            worst_case_sum += diff
            if (base_scores[i]-base_scores[j])*(ranking[i]-ranking[j]) > 0:
                discordant_sum += diff
    
    # We use a worst-case sum (all pairs are discordant) for normalization instead
    return (discordant_sum / worst_case_sum) ** 0.5


def weighted_tau_separate(graph:FairPairGraph, ranking:dict, subgraph: Union[FairPairGraph, None] = None,
                          score_attr='skill', calc_within=True, calc_between=True) -> Tuple[float, float]:
    '''
    Calculates the within-group and the between-group weighted Kemedy distance for a `ranking` given the
    "ground-truth" ranking from initial scores of `graph`.

    Parameters
    ----------
    - graph: the full FairPairGraph for ground-truth ranks
    - ranking: dict of nodes and their ranking results
    - subgraph: a FairPairGraph subgraph of `graph`, or identical to `graph` if None
    - score_attr: name of the node attribute for storing scores
    - calc_within: whether to calculate within-group tau
    - calc_between: whether to calculate between-group tau

    Returns
    -------
    - tau_within: within-group weighted Kemedy distance
    - tau_between: between-group weighted Kemedy distance
    '''
    if subgraph is None: subgraph = graph

    # get rankings as dicts
    ranking = scores_to_rank(ranking)
    base_scores = {node: score for node, score in graph.nodes(data=score_attr)}

    subgraph_nodes = list(subgraph.nodes) # must faster to do this only once
    complementary_nodes = [node for node in graph.nodes if node not in subgraph.nodes]

    # sum up weights of within-group discordant pairs
    if calc_within:
        discordant_sum_int = 0
        worst_case_sum_int = 0
        for num, i in enumerate(subgraph_nodes):
            # consider within-group pairs till before the diagonal
            for j in subgraph_nodes[:num]:
                diff = (base_scores[i]-base_scores[j]) ** 2
                worst_case_sum_int += diff
                if (base_scores[i]-base_scores[j])*(ranking[i]-ranking[j]) > 0:
                    discordant_sum_int += diff
        tau_within = (discordant_sum_int / worst_case_sum_int) ** 0.5
    else: tau_within = None

    # sum up weights of between-group discordant pairs
    if calc_between:
        discordant_sum_ext = 0
        worst_case_sum_ext = 0
        for i in subgraph_nodes:
            # consider complementary nodes only one-way
            for j in complementary_nodes:
                diff = (base_scores[i]-base_scores[j]) ** 2
                worst_case_sum_ext += diff
                if (base_scores[i]-base_scores[j])*(ranking[i]-ranking[j]) > 0:
                    discordant_sum_ext += diff
        tau_between = (discordant_sum_ext / worst_case_sum_ext) ** 0.5
    else: tau_between = None
    
    return tau_within, tau_between


def weighted_individual_tau(graph:FairPairGraph, ranking:dict, subgraph: Union[FairPairGraph, None] = None, score_attr='skill') -> List[float]:
    '''
    Calculates the weighted Kemedy distance for each node in a `ranking` separately,
    given the "ground-truth" ranking from initial scores of `graph`.

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
    tau = []
    subgraph_nodes = [node for node, _ in sorted([(node, ranking[node]) for node in list(subgraph.nodes)], key=lambda n:n[1], reverse=True)]
    complementary_nodes = [node for node in graph.nodes if node not in subgraph.nodes]
    for i in subgraph_nodes:
        # Consider within-group pairs and pairs with complementary nodes.
        discordant_sum = 0
        worst_case_sum = 0
        within_group_nodes = [node for node in subgraph_nodes if node != i] # no self-loops
        for j in within_group_nodes + complementary_nodes:
            diff = (base_scores[i]-base_scores[j]) ** 2
            worst_case_sum += diff
            if (base_scores[i]-base_scores[j])*(ranking[i]-ranking[j]) > 0:
                discordant_sum += diff
        tau.append((ranking[i], (discordant_sum / worst_case_sum) ** 0.5))
    
    return tau # a list of (rank, tau) tuples, one for each node in subgraph


def weighted_topk_tau(graph:FairPairGraph, ranking:dict, subgraph: Union[FairPairGraph, None] = None,
                      topk=[10,20,50,100,200,500], score_attr='skill') -> List[float]:
    '''
    Calculates the weighted Kemedy distance for the top k nodes in a `ranking`,
    given the "ground-truth" ranking from initial scores of `graph`.

    Parameters
    ----------
    - graph: the full FairPairGraph for ground-truth ranks
    - ranking: dict of nodes and their ranking results
    - topk: at which cutoffs to calculate the Kemedy distance
    - subgraph: a FairPairGraph subgraph of `graph`, or identical to `graph` if None
    - score_attr: name of the node attribute for storing scores
    '''
    if subgraph is None: subgraph = graph

    # get rankings as dicts
    ranking = scores_to_rank(ranking)
    base_scores = {node: score for node, score in graph.nodes(data=score_attr)}

    # extract (adjacency) matrices of weight difference and concordance
    subgraph_nodes = [node for node, _ in sorted([(node, ranking[node]) for node in list(subgraph.nodes)], key=lambda n:n[1])]
    complementary_nodes = [node for node in graph.nodes if node not in subgraph.nodes]
    size = (len(subgraph), len(graph))
    weights = np.zeros(size)
    discordance = np.full(size, False)
    for i, node_i in enumerate(subgraph_nodes):
        # consider within-group pairs till before the diagonal
        # and pairs with complementary nodes only one-way
        # use "upper diagonal" to enable top-k evaluation
        for j, node_j in enumerate(subgraph_nodes[i:] + complementary_nodes):
            weights[i, j] = (base_scores[node_i]-base_scores[node_j]) ** 2
            discordance[i, j] = (base_scores[node_i]-base_scores[node_j])*(ranking[node_i]-ranking[node_j]) > 0

    tau = []
    for k in topk:
        if len(subgraph)>=k:
            discordant_sum = np.sum(weights[:k,:], where=discordance[:k,:])
            worst_case_sum = np.sum(weights[:k,:])
            tau.append((k, (discordant_sum / worst_case_sum) ** 0.5))

    return tau # a list of (k, tau) tuples, one for each cutoff k


##### Group-Representation #####


def rND(graph:FairPairGraph, ranking:dict, subgraph: Union[FairPairGraph, None] = None) -> float:
    '''
    Calculates the rND score
    '''


def exposure(graph:FairPairGraph, ranking:dict, subgraph: Union[FairPairGraph, None] = None) -> float:
    '''
    Calculates the exposure (Singh & Joachims, 2018) of the nodes of the subgraph in the ranking.
    Uses log_2 discount and normalizes by group size.

    Parameters
    ----------
    - graph: the full FairPairGraph for ground-truth ranks
    - ranking: dict of nodes and their ranking results
    - subgraph: a FairPairGraph subgraph of `graph`, or identical to `graph` if None
    '''
    if subgraph is None: subgraph = graph
    ranking = scores_to_rank(ranking)

    exp = np.array([ranking[node] for node in subgraph.nodes])
    exp = 1/np.log2(exp + 2) # use +2 such that the highest rank (0) works out fine
    return np.sum(exp)/len(subgraph)


def exposure_nodes(ranking:dict, subgraph_nodes:list) -> float:
    '''
    A faster version of calculating exposure (Singh & Joachims, 2018)

    Parameters
    ----------
    - ranking: dict of nodes and their ranking results
    - subgraph_nodes: nodes from a subgraph for which to calculate the exposure
    '''

    ranking = scores_to_rank(ranking)

    exp = np.array([ranking[node] for node in subgraph_nodes])
    exp = 1/np.log2(exp + 2) # use +2 such that the highest rank (0) works out fine
    return np.sum(exp)/len(subgraph_nodes)


def topk_exposure(graph:FairPairGraph, ranking:dict, subgraph: Union[FairPairGraph, None] = None,
                  topk=[10,20,50,100,200,500]) -> float:
    '''
    Calculates the exposure (Singh & Joachims, 2018) of the top k nodes of the subgraph in the ranking.
    Uses log_2 discount and normalizes by k.

    Parameters
    ----------
    - graph: the full FairPairGraph for ground-truth ranks
    - ranking: dict of nodes and their ranking results
    - subgraph: a FairPairGraph subgraph of `graph`, or identical to `graph` if None
    - topk: at which cutoffs to calculate the Exposure
    '''
    if subgraph is None: subgraph = graph
    ranking = scores_to_rank(ranking)

    exp = np.array(sorted([ranking[node] for node in subgraph.nodes])) # sort to enable topk
    exp = 1/np.log2(exp + 2) # use +2 such that the highest rank (0) works out fine

    exps = []
    for k in topk:
        if len(subgraph)>=k:
            exps.append((k, np.sum(exp[:k])/k))
    return exps