import pandas as pd

from itertools import product
import multiprocessing

from fairpair import *

def ranking_evaluation(trial:int, sampling_method:str, ranking_method:str):

    if ranking_method == 'randomRankRecovery':
        file = './data/imdb-wiki_results/randomRankRecovery_correlations_10trials.csv'
    elif ranking_method == 'davidScore':
        file = './data/imdb-wiki_results/davidScore_correlations_10trials.csv'
    elif ranking_method == 'rankCentrality':
        file = './data/imdb-wiki_results/rankCentrality_correlations_10trials.csv'
    elif ranking_method == 'fairPageRank':
        file = './data/imdb-wiki_results/fairPageRank_correlations_10trials.csv'
    elif ranking_method == 'GNNRank':
        file = './data/imdb-wiki_results/GNNRank_correlations_10trials.csv'
    else:
        raise ValueError('Unsupported Ranking Method')

    data = pd.read_csv(file)

    filtered_df = data[data['ranker'] == ranking_method]#.copy()
    #del data
    
    filtered_df = filtered_df[filtered_df['trial'] == trial] 
    filtered_df = filtered_df[filtered_df['sampling method'] == sampling_method] 
    filtered_df = filtered_df.reset_index(drop=True)

    first_iteration = filtered_df.iteration.unique()[0]

    base_scores = filtered_df[(filtered_df.iteration == first_iteration)]['skill score'].to_dict()
    all_nodes = list(filtered_df[filtered_df.iteration == first_iteration].index)
    majority_nodes = list(filtered_df[(filtered_df.iteration == first_iteration) & (filtered_df.group == 'Privileged')].index)
    minority_nodes = list(filtered_df[(filtered_df.iteration == first_iteration) & (filtered_df.group == 'Unprivileged')].index)

    results = []

    for iteration in filtered_df.iteration.unique():

        _tmp_df = filtered_df[filtered_df.iteration == iteration].reset_index(drop=True)
        ranking = _tmp_df['rank'].to_dict()

        if len(filtered_df):
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
        
            print(f'trial {trial}, {sampling_method}, {ranking_method}: finished {iteration} iterations.')

    return results




if __name__ == '__main__':

    #file = './data/imdb-wiki_results/basicMethods_correlations_10trials.csv'

    tasks = list(product(range(10), ['randomSampling', 'oversampling', 'rankSampling'],
                         ['rankCentrality', 'randomRankRecovery', 'davidScore', 'fairPageRank'])) # trial, sampling_method, ranking_method

    pool = multiprocessing.Pool()
    accuracy = pool.starmap(ranking_evaluation, tasks)

    accuracy = [result for pool in accuracy for result in pool]
    accuracy = pd.DataFrame(accuracy, columns=['trial', 'iteration', 'value', 'sampling strategy', 'recovery method', 'metric', 'group'])
    accuracy.to_csv('./data/imdb-wiki_results/IMDB-WIKI_basicMethods_evaluated.csv', index=False)