from scipy.stats import norm
from scipy import integrate
import mpmath
import numpy as np

from fairpair import *

def get_cutoff_accuracy(trial:int, N=400, Nm=200):
    accuracy = []
    H = FairPairGraph()
    H.generate_groups(N, Nm) # same size groups
    H.group_assign_scores(nodes=H.nodes, loc=0, scale=1) # general score distribution
    sampler = RandomSampling(H, warn=False)
    ranker = RankRecovery(H)
    ranking = None
    for j in range(101):
        sampler.apply(iter=100, k=1)
        for cutoff in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            # make sure to use separate directories for each thread
            ranking, other_nodes = ranker.apply(rank_using='fairPageRank', cutoff=cutoff, path=f'data/tmp{trial}')
            if len(other_nodes) == 0:
                tau = weighted_tau(H, ranking)
                accuracy.append((trial, j*100, cutoff, tau))
    return accuracy


def get_simulated_cutoff(trial:int, method:rankCentrality, N=400, Nm=200):
    accuracy = []
    H = FairPairGraph()
    H.generate_groups(N, Nm) # same size groups
    H.group_assign_scores(nodes=H.nodes, loc=0, scale=1) # general score distribution
    sampler = RandomSampling(H, warn=False)
    ranking = None
    for j in range(101):
        sampler.apply(iter=10, k=1)
        for cutoff in [0.4]:
            if method == 'fairPageRank':
                name = method
                ranker = RankRecovery(H)
                ranking, other_nodes = ranker.apply(rank_using='fairPageRank', cutoff=cutoff, path=f'data/tmp{trial}')
            else:
                name = method.__name__
                # simulate cut-off
                G = FairPairGraph()
                edges = H.edges(data='weight')
                edges = [(outgoing, incoming, 1) for outgoing, incoming, weight in edges if weight>=cutoff]
                G.add_weighted_edges_from(edges)
                ranker = RankRecovery(G)
                ranking, other_nodes = ranker.apply(rank_using=method) # by default, apply rankCentrality method
            if len(other_nodes) == 0:
                tau = weighted_tau(H, ranking)
                accuracy.append((trial, j*10, cutoff, tau, name))
    return accuracy


def get_method_accuracy(trial:int, method:rankCentrality, N=400, Nm=200):
    accuracy = []
    H = FairPairGraph()
    H.generate_groups(N, Nm) # same size groups
    H.assign_skills(loc=0, scale=0.86142674) # general skill distribution
    H.assign_bias(nodes=H.minority_nodes, loc=-1.43574282, scale=0.43071336) # add bias to unprivileged group
    sampler = RandomSampling(H, warn=False)
    ranker = RankRecovery(H)
    ranking = None
    for j in range(101):
        sampler.apply(iter=100, k=1)
        if method == 'fairPageRank':
            name = method
            ranking, other_nodes = ranker.apply(rank_using=method, path=f'data/tmp{trial}')
        else:
            name = method.__name__
            ranking, other_nodes = ranker.apply(rank_using=method) # by default, apply rankCentrality method
        if len(other_nodes) == 0:
            tau = weighted_tau(H, ranking)
            accuracy.append((trial, j*100, tau, 'Overall', name))
            #tau = weighted_tau(H, ranking, H.majority)
            #accuracy.append((trial, j*10, tau, 'Priviledged', name))
            #tau = weighted_tau(H, ranking, H.minority)
            #accuracy.append((trial, j*10, tau, 'Unpriviledged', name))
            #tau = weighted_tau_separate(H, ranking, H.majority)
            #accuracy.append((trial, j*10, tau[0], 'Priviledged within-group', name))
            #accuracy.append((trial, j*10, tau[1], 'Between groups', name))
            #tau = weighted_tau_separate(H, ranking, H.minority, calc_between=False)
            #accuracy.append((trial, j*10, tau[0], 'Unpriviledged within-group', name))
    return accuracy