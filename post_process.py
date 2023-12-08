from itertools import product
import multiprocessing

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
    fair = fsc.Fair(k=400, p=0.6, alpha=0.1)

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



def post_process_IMDB_WIKI_FAstarIR(trial, sampling_method, ranking_method):

    if ranking_method == 'randomRankRecovery':
        file = './data/imdb-wiki_results/randomRankRecovery_correlations_10trials.csv'
    elif ranking_method == 'davidScore':
        file = './data/imdb-wiki_results/davidScore_correlations_10trials.csv'
    elif ranking_method == 'rankCentrality':
        file = './data/imdb-wiki_results/rankCentrality_correlations_10trials.csv'
    elif ranking_method == 'fairPageRank':
        file = './data/imdb-wiki_results/fairPageRank_correlations_10trials.csv'
    elif ranking_method == 'GNNRank':
        file = './data/imdb-wiki_results/GNNRank_correlations_reTrained_syncRank_10trials.csv'
    else:
        raise ValueError('Unsupported Ranking Method')
    data = pd.read_csv(file)
    filtered_df = data[data['ranker'] == ranking_method]

    filtered_df = filtered_df[filtered_df['trial'] == trial] 
    filtered_df = filtered_df[filtered_df['sampling method'] == sampling_method] 
    #filtered_df = filtered_df[filtered_df['bias_applied'] == bias_applied]
    filtered_df = filtered_df.reset_index(drop=True)

    # re-construct an empty comparison graph and label nodes
    # we need this for the evaluation of accuracy, group exposure, and group-conditioned accuracy later
    try:
        first_iteration = filtered_df.iteration.unique()[0]
    except IndexError:
        raise IndexError(trial, sampling_method, ranking_method)

    base_scores = filtered_df[(filtered_df.iteration == first_iteration)]['skill score'].to_dict()
    all_nodes = list(filtered_df[filtered_df.iteration == first_iteration].index)
    majority_nodes = list(filtered_df[(filtered_df.iteration == first_iteration) & (filtered_df.group == 'Privileged')].index)
    minority_nodes = list(filtered_df[(filtered_df.iteration == first_iteration) & (filtered_df.group == 'Unprivileged')].index)

    # initialize the FairSearchCore library
    fair = fsc.Fair(k=len(all_nodes), p=0.6, alpha=0.1)  # default for p is 0.5

    results = []

    print(f'prepared {sampling_method}, {ranking_method}, trial {trial}')

    for iteration in filtered_df.iteration.unique():
        # get the ranking from the filtered_df
        # luckily, the nodes are always stored in the same order (in each iteration, with each sampling strategy/ranking recovery method)
        tmp_df = filtered_df[filtered_df.iteration == iteration].reset_index(drop=True)
        tmp_df['group_bin'] = tmp_df['group'] == 'Unprivileged'

        # create a list of FairScoreDoc
        ranking_list = tmp_df.reset_index()[['index', 'rank', 'group_bin']].sort_values('rank').to_numpy()
        fair_score_list = [FairScoreDoc(entry[0], len(ranking_list)-entry[1], entry[2]) for entry in ranking_list]
        #for entry in fair_score_list:
        #    print(entry.id, entry.score, entry.is_protected)

        # apply the FA*IR re-ranking algorithm
        re_ranked_list = fair.re_rank(fair_score_list)
        #for entry in re_ranked_list:
        #    print(entry.id, entry.score, entry.is_protected)

        # re-format output
        ranking = {entry.id: len(re_ranked_list)-i for i, entry in enumerate(re_ranked_list)} # invert the ranks for evaluation

        tau = weighted_tau_nodes(base_scores, ranking, subgraph_nodes=all_nodes, complementary_nodes=[])
        results.append((trial, iteration, tau, sampling_method, ranking_method, 'tau', 'Overall'))

        tau = weighted_tau_nodes(base_scores, ranking, subgraph_nodes=majority_nodes, complementary_nodes=minority_nodes)
        results.append((trial, iteration, tau, sampling_method, ranking_method, 'tau', 'Privileged'))

        tau = weighted_tau_nodes(base_scores, ranking, subgraph_nodes=minority_nodes, complementary_nodes=majority_nodes)
        results.append((trial, iteration, tau, sampling_method, ranking_method, 'tau', 'Unprivileged'))

        exp = exposure_nodes(ranking, subgraph_nodes=majority_nodes)
        results.append((trial, iteration, exp, sampling_method, ranking_method, 'exposure', 'Privileged'))

        exp = exposure_nodes(ranking, subgraph_nodes=minority_nodes)
        results.append((trial, iteration, exp, sampling_method, ranking_method, 'exposure', 'Unprivileged'))

        tau = weighted_tau_separate_nodes(base_scores, ranking, subgraph_nodes=majority_nodes, complementary_nodes=minority_nodes)
        results.append((trial, iteration, tau[0], sampling_method, ranking_method, 'tau', 'Privileged within-group'))
        results.append((trial, iteration, tau[1], sampling_method, ranking_method, 'tau', 'Between groups'))

        tau = weighted_tau_separate_nodes(base_scores, ranking, subgraph_nodes=minority_nodes, complementary_nodes=majority_nodes, calc_between=False)
        results.append((trial, iteration, tau[0], sampling_method, ranking_method, 'tau', 'Unprivileged within-group'))

        if iteration % 100 == 0:
            print(f'post-processed {sampling_method}, {ranking_method}, trial {trial}, iteration {iteration}')

    return results



def post_process_IMDB_WIKI_EPIRA(trial, sampling_method, ranking_method):

    if ranking_method == 'randomRankRecovery':
        file = './data/imdb-wiki_results/randomRankRecovery_correlations_10trials.csv'
    elif ranking_method == 'davidScore':
        file = './data/imdb-wiki_results/davidScore_correlations_10trials.csv'
    elif ranking_method == 'rankCentrality':
        file = './data/imdb-wiki_results/rankCentrality_correlations_10trials.csv'
    elif ranking_method == 'fairPageRank':
        file = './data/imdb-wiki_results/fairPageRank_correlations_10trials.csv'
    elif ranking_method == 'GNNRank':
        file = './data/imdb-wiki_results/GNNRank_correlations_reTrained_syncRank_10trials.csv'
    else:
        raise ValueError('Unsupported Ranking Method')
    data = pd.read_csv(file)
    filtered_df = data[data['ranker'] == ranking_method]

    filtered_df = filtered_df[filtered_df['trial'] == trial] 
    filtered_df = filtered_df[filtered_df['sampling method'] == sampling_method] 
    #filtered_df = filtered_df[filtered_df['bias_applied'] == bias_applied]
    filtered_df = filtered_df.reset_index(drop=True)

    # re-construct an empty comparison graph and label nodes
    # we need this for the evaluation of accuracy, group exposure, and group-conditioned accuracy later
    try:
        first_iteration = filtered_df.iteration.unique()[0]
    except IndexError:
        raise IndexError(trial, sampling_method, ranking_method)

    base_scores = filtered_df[(filtered_df.iteration == first_iteration)]['skill score'].to_dict()
    all_nodes = list(filtered_df[filtered_df.iteration == first_iteration].index)
    majority_nodes = list(filtered_df[(filtered_df.iteration == first_iteration) & (filtered_df.group == 'Privileged')].index)
    minority_nodes = list(filtered_df[(filtered_df.iteration == first_iteration) & (filtered_df.group == 'Unprivileged')].index)

    results = []

    print(f'prepared {sampling_method}, {ranking_method}, trial {trial}')

    for iteration in filtered_df.iteration.unique():
        # get the ranking from the filtered_df
        # luckily, the nodes are always stored in the same order (in each iteration, with each sampling strategy/ranking recovery method)
        tmp_df = filtered_df[filtered_df.iteration == iteration].reset_index(drop=True)
        tmp_df['group_bin'] = tmp_df['group'] == 'Unprivileged'

        # apply the EPIRA re-ranking algorithm
        ranking_list = tmp_df.reset_index()[['index', 'rank', 'group_bin']].to_numpy()
        base_ranks = np.arange(len(ranking_list))[ranking_list[:,1].argsort()].reshape((1,len(ranking_list))) #ranking_list[:,1].reshape((1,400))
        item_ids = ranking_list[:,0].astype(int)
        group_ids = ranking_list[:,2].astype(int)

        consensus, ranking_group_ids = epiRA(base_ranks, item_ids, group_ids, bnd=0.99, grporder=True, agg_method=None, print_swaps=False)

        # re-format output
        ranking = {id: len(consensus)-rank for rank, id in enumerate(consensus)} # invert the ranks for evaluation

        tau = weighted_tau_nodes(base_scores, ranking, subgraph_nodes=all_nodes, complementary_nodes=[])
        results.append((trial, iteration, tau, sampling_method, ranking_method, 'tau', 'Overall'))

        tau = weighted_tau_nodes(base_scores, ranking, subgraph_nodes=majority_nodes, complementary_nodes=minority_nodes)
        results.append((trial, iteration, tau, sampling_method, ranking_method, 'tau', 'Privileged'))

        tau = weighted_tau_nodes(base_scores, ranking, subgraph_nodes=minority_nodes, complementary_nodes=majority_nodes)
        results.append((trial, iteration, tau, sampling_method, ranking_method, 'tau', 'Unprivileged'))

        exp = exposure_nodes(ranking, subgraph_nodes=majority_nodes)
        results.append((trial, iteration, exp, sampling_method, ranking_method, 'exposure', 'Privileged'))

        exp = exposure_nodes(ranking, subgraph_nodes=minority_nodes)
        results.append((trial, iteration, exp, sampling_method, ranking_method, 'exposure', 'Unprivileged'))

        tau = weighted_tau_separate_nodes(base_scores, ranking, subgraph_nodes=majority_nodes, complementary_nodes=minority_nodes)
        results.append((trial, iteration, tau[0], sampling_method, ranking_method, 'tau', 'Privileged within-group'))
        results.append((trial, iteration, tau[1], sampling_method, ranking_method, 'tau', 'Between groups'))

        tau = weighted_tau_separate_nodes(base_scores, ranking, subgraph_nodes=minority_nodes, complementary_nodes=majority_nodes, calc_between=False)
        results.append((trial, iteration, tau[0], sampling_method, ranking_method, 'tau', 'Unprivileged within-group'))

        if iteration % 100 == 0:
            print(f'post-processed {sampling_method}, {ranking_method}, trial {trial}, iteration {iteration}')

    return results



if __name__ == '__main__':

    #tasks = list(product(range(10), ['RandomSampling', 'OversampleMinority', 'RankSampling'],
    #                     ['davidScore', 'randomRankRecovery', 'rankCentrality', 'GNNRank'], [False, True])) # trial, sampling_method, ranking_method, apply_bias
    tasks = list(product(range(10), ['randomSampling', 'oversampling', 'rankSampling'],
                         ['davidScore', 'randomRankRecovery', 'rankCentrality', 'GNNRank'])) # trial, sampling_method, ranking_method, apply_bias

    #tasks = list(product(range(10), ['RandomSampling'], ['davidScore'], [True]))

    pool = multiprocessing.Pool()
    accuracy = pool.starmap(post_process_IMDB_WIKI_EPIRA, tasks)

    accuracy = [result for pool in accuracy for result in pool]
    accuracy = pd.DataFrame(accuracy, columns=['trial', 'iteration', 'value', 'bias_applied', 'sampling strategy', 'recovery method', 'metric', 'group'])
    accuracy.to_csv('./data/post_processing/IMDB-WIKI_EPIRA99_10trials.csv', index=False)