import pandas as pd

from fairpair import *
import fairsearchcore as fsc
from fairsearchcore.models import FairScoreDoc

import sys
sys.path.append('../fairer_together/')
from src.epira import epiRA



def post_process(trial, sampling_method, ranking_method, bias_applied):

    #data = pd.read_csv('./correlations_combined.csv.bz2', compression='bz2')
    if ranking_method == 'GNNRank':
        filtered_df = pd.read_csv('./data/GNNRank_results/GNNRank_correlations_10trials.csv')
    else:
        data = pd.read_csv('./data/others_results/others_correlations_10trials.csv')
        filtered_df = data[data['ranker'] == ranking_method].copy()
        del data

    filtered_df = filtered_df[filtered_df['trial'] == trial] 
    filtered_df = filtered_df[filtered_df['sampling method'] == sampling_method] 
    filtered_df = filtered_df[filtered_df['bias_applied'] == bias_applied]
    filtered_df = filtered_df.reset_index(drop=True)

    # re-construct an empty comparison graph and label nodes
    # we need this for the evaluation of accuracy, group exposure, and group-conditioned accuracy later
    G = FairPairGraph()
    G.add_nodes_from(range(400))
    skill_scores = dict(filtered_df[filtered_df.iteration == filtered_df.iteration.min()]['skill score'])
    nx.set_node_attributes(G, skill_scores, 'skill')
    minority = dict(filtered_df[filtered_df.iteration == filtered_df.iteration.min()]['group'] == 'Unprivileged')
    nx.set_node_attributes(G, minority, 'minority')

    # initialize the FairSearchCore library
    fair = fsc.Fair(k=400, p=0.5, alpha=0.1)

    accuracy = []

    print(f'prepared {sampling_method}, {ranking_method}, with{"out" if not bias_applied else ""} bias, trial {trial}')

    for iteration in filtered_df.iteration.unique():
        # get the ranking from the filtered_df
        # luckily, the nodes are always stored in the same order (in each iteration, with each sampling strategy/ranking recovery method)
        tmp_df = filtered_df[filtered_df.iteration == iteration].reset_index(drop=True)
        tmp_df['group_bin'] = tmp_df['group'] == 'Unprivileged'

        # create a list of FairScoreDoc
        ranking_list = tmp_df.reset_index()[['index', 'rank', 'group_bin']].sort_values('rank').to_numpy()
        fair_score_list = [FairScoreDoc(entry[0], 400-entry[1], entry[2]) for entry in ranking_list]
        #for entry in fair_score_list:
        #    print(entry.id, entry.score, entry.is_protected)

        # apply the FA*IR re-ranking algorithm
        re_ranked_list = fair.re_rank(fair_score_list)
        #for entry in re_ranked_list:
        #    print(entry.id, entry.score, entry.is_protected)

        # re-format output
        ranking = {entry.id: 400-i for i, entry in enumerate(re_ranked_list)} # invert the ranks for evaluation

        tau = weighted_tau(G, ranking)
        accuracy.append((trial, iteration, tau, bias_applied, sampling_method, ranking_method, 'tau', 'Overall'))
        tau = weighted_tau(G, ranking, G.majority)
        accuracy.append((trial, iteration, tau, bias_applied, sampling_method, ranking_method, 'tau', 'Privileged'))
        tau = weighted_tau(G, ranking, G.minority)
        accuracy.append((trial, iteration, tau, bias_applied, sampling_method, ranking_method, 'tau', 'Unprivileged'))
        tau = weighted_tau_separate(G, ranking, G.majority)
        accuracy.append((trial, iteration, tau[0], bias_applied, sampling_method, ranking_method, 'tau', 'Privileged within-group'))
        accuracy.append((trial, iteration, tau[1], bias_applied, sampling_method, ranking_method, 'tau', 'Between groups'))
        tau = weighted_tau_separate(G, ranking, G.minority, calc_between=False)
        accuracy.append((trial, iteration, tau[0], bias_applied, sampling_method, ranking_method, 'tau', 'Unprivileged within-group'))
        exp = exposure(G, ranking, G.majority)
        accuracy.append((trial, iteration, exp, bias_applied, sampling_method, ranking_method, 'exposure', 'Privileged'))
        exp = exposure(G, ranking, G.minority)
        accuracy.append((trial, iteration, exp, bias_applied, sampling_method, ranking_method, 'exposure', 'Unprivileged'))

        if iteration % 100 == 0:
            print(f'post-processed {sampling_method}, {ranking_method}, with{"out" if not bias_applied else ""} bias, trial {trial}, iteration {iteration}')

    return accuracy



def post_process_epira(trial, sampling_method, ranking_method, bias_applied):

    #data = pd.read_csv('./correlations_combined.csv.bz2', compression='bz2')
    if ranking_method == 'GNNRank':
        filtered_df = pd.read_csv('./data/GNNRank_results/GNNRank_correlations_10trials.csv')
    else:
        data = pd.read_csv('./data/others_results/others_correlations_10trials.csv')
        filtered_df = data[data['ranker'] == ranking_method].copy()
        del data

    filtered_df = filtered_df[filtered_df['trial'] == trial] 
    filtered_df = filtered_df[filtered_df['sampling method'] == sampling_method] 
    filtered_df = filtered_df[filtered_df['bias_applied'] == bias_applied]
    filtered_df = filtered_df.reset_index(drop=True)

    # re-construct an empty comparison graph and label nodes
    # we need this for the evaluation of accuracy, group exposure, and group-conditioned accuracy later
    G = FairPairGraph()
    G.add_nodes_from(range(400))
    skill_scores = dict(filtered_df[filtered_df.iteration == filtered_df.iteration.min()]['skill score'])
    nx.set_node_attributes(G, skill_scores, 'skill')
    minority = dict(filtered_df[filtered_df.iteration == filtered_df.iteration.min()]['group'] == 'Unprivileged')
    nx.set_node_attributes(G, minority, 'minority')

    accuracy = []

    print(f'prepared {sampling_method}, {ranking_method}, with{"out" if not bias_applied else ""} bias, trial {trial}')

    for iteration in filtered_df.iteration.unique():
        # get the ranking from the filtered_df
        # luckily, the nodes are always stored in the same order (in each iteration, with each sampling strategy/ranking recovery method)
        tmp_df = filtered_df[filtered_df.iteration == iteration].reset_index(drop=True)
        tmp_df['group_bin'] = tmp_df['group'] == 'Unprivileged'

        # apply the EPIRA re-ranking algorithm
        ranking_list = tmp_df.reset_index()[['index', 'rank', 'group_bin']].to_numpy()
        base_ranks = np.arange(400)[ranking_list[:,1].argsort()].reshape((1,400)) #ranking_list[:,1].reshape((1,400))
        item_ids = ranking_list[:,0].astype(int)
        group_ids = ranking_list[:,2].astype(int)

        consensus, ranking_group_ids = epiRA(base_ranks, item_ids, group_ids, bnd=0.99, grporder=True, agg_method=None, print_swaps=False)

        # re-format output
        ranking = {id: 400-rank for rank, id in enumerate(consensus)} # invert the ranks for evaluation

        tau = weighted_tau(G, ranking)
        accuracy.append((trial, iteration, tau, bias_applied, sampling_method, ranking_method, 'tau', 'Overall'))
        tau = weighted_tau(G, ranking, G.majority)
        accuracy.append((trial, iteration, tau, bias_applied, sampling_method, ranking_method, 'tau', 'Privileged'))
        tau = weighted_tau(G, ranking, G.minority)
        accuracy.append((trial, iteration, tau, bias_applied, sampling_method, ranking_method, 'tau', 'Unprivileged'))
        tau = weighted_tau_separate(G, ranking, G.majority)
        accuracy.append((trial, iteration, tau[0], bias_applied, sampling_method, ranking_method, 'tau', 'Privileged within-group'))
        accuracy.append((trial, iteration, tau[1], bias_applied, sampling_method, ranking_method, 'tau', 'Between groups'))
        tau = weighted_tau_separate(G, ranking, G.minority, calc_between=False)
        accuracy.append((trial, iteration, tau[0], bias_applied, sampling_method, ranking_method, 'tau', 'Unprivileged within-group'))
        exp = exposure(G, ranking, G.majority)
        accuracy.append((trial, iteration, exp, bias_applied, sampling_method, ranking_method, 'exposure', 'Privileged'))
        exp = exposure(G, ranking, G.minority)
        accuracy.append((trial, iteration, exp, bias_applied, sampling_method, ranking_method, 'exposure', 'Unprivileged'))

        if iteration % 100 == 0:
            print(f'post-processed {sampling_method}, {ranking_method}, with{"out" if not bias_applied else ""} bias, trial {trial}, iteration {iteration}')

    return accuracy