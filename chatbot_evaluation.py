import pandas as pd

from itertools import product
import multiprocessing

from fairpair import *

def ranking_evaluation(trial:int, sampling_method:str, ranking_method:str, group_attribute:str, benchmark:str):

    file = '../fairpair/data/chatbot_arena_results/basicMethods_correlations_fullMedian_phiPriv.csv'
    #file = '../fairpair/data/chatbot_arena_results/GNNRank_correlations_fullMedian.csv'

    data = pd.read_csv(file)

    filtered_df = data[data['ranker'] == ranking_method]#.copy()
    #del data
    
    filtered_df = filtered_df[filtered_df['trial'] == trial] 
    filtered_df = filtered_df[filtered_df['sampling method'] == sampling_method]
    filtered_df = filtered_df[filtered_df['group attribute'] == group_attribute] 
    filtered_df = filtered_df[filtered_df['benchmark'] == benchmark]
    filtered_df = filtered_df.dropna(subset=['skill score'])
    filtered_df = filtered_df.reset_index(drop=True)

    if not len(filtered_df): return pd.DataFrame()

    first_iteration = filtered_df.iteration.unique()[0]

    base_scores = filtered_df[(filtered_df.iteration == first_iteration)]['skill score'].to_dict()
    all_nodes = list(filtered_df[filtered_df.iteration == first_iteration].index)
    majority_nodes = list(filtered_df[(filtered_df.iteration == first_iteration) & (filtered_df.group == 'Privileged')].index)
    minority_nodes = list(filtered_df[(filtered_df.iteration == first_iteration) & (filtered_df.group == 'Unprivileged')].index)

    results = pd.DataFrame()

    for iteration in filtered_df.iteration.unique():

        _tmp_df = filtered_df[filtered_df.iteration == iteration].reset_index(drop=True)
        ranking = _tmp_df['rank'].to_dict()

        iter_results = pd.DataFrame(columns=['trial', 'iteration', 'value', 'sampling strategy', 'recovery method',
                                             'group attribute', 'benchmark', 'metric', 'group'])

        if len(filtered_df):
            tau = weighted_tau_nodes(base_scores, ranking, subgraph_nodes=all_nodes, complementary_nodes=[])
            iter_results = pd.concat([iter_results, pd.DataFrame([{'value': tau, 'metric': 'tau', 'group': 'Overall'}])])

            tau = weighted_tau_nodes(base_scores, ranking, subgraph_nodes=majority_nodes, complementary_nodes=minority_nodes)
            iter_results = pd.concat([iter_results, pd.DataFrame([{'value': tau, 'metric': 'tau', 'group': 'Privileged'}])])

            tau = weighted_tau_nodes(base_scores, ranking, subgraph_nodes=minority_nodes, complementary_nodes=majority_nodes)
            iter_results = pd.concat([iter_results, pd.DataFrame([{'value': tau, 'metric': 'tau', 'group': 'Unprivileged'}])])

            exp = exposure_nodes(ranking, subgraph_nodes=majority_nodes)
            iter_results = pd.concat([iter_results, pd.DataFrame([{'value': exp, 'metric': 'exposure', 'group': 'Privileged'}])])

            exp = exposure_nodes(ranking, subgraph_nodes=minority_nodes)
            iter_results = pd.concat([iter_results, pd.DataFrame([{'value': exp, 'metric': 'exposure', 'group': 'Unprivileged'}])])

            tau = weighted_tau_separate_nodes(base_scores, ranking, subgraph_nodes=majority_nodes, complementary_nodes=minority_nodes)
            iter_results = pd.concat([iter_results, pd.DataFrame([{'value': tau[0], 'metric': 'tau', 'group': 'Privileged within-group'}])])
            iter_results = pd.concat([iter_results, pd.DataFrame([{'value': tau[1], 'metric': 'tau', 'group': 'Between groups'}])])

            tau = weighted_tau_separate_nodes(base_scores, ranking, subgraph_nodes=minority_nodes, complementary_nodes=majority_nodes, calc_between=False)
            iter_results = pd.concat([iter_results, pd.DataFrame([{'value': tau[0], 'metric': 'tau', 'group': 'Unprivileged within-group'}])])
        
            print(f'trial {trial}, {sampling_method}, {group_attribute}, {ranking_method}, {iteration} iterations.')

        iter_results['trial'] = trial
        iter_results['iteration'] = iteration
        iter_results['sampling strategy'] = sampling_method
        iter_results['recovery method'] = ranking_method
        iter_results['group attribute'] = group_attribute
        iter_results['benchmark'] = benchmark
        results = pd.concat([results, iter_results], ignore_index=True)

    return results




if __name__ == '__main__':

    # trial, benchmark
    tasks = list(product(range(10),
                         ['full dataset'],
                         ['rankCentrality', 'randomRankRecovery', 'davidScore', 'fairPageRank', 'btl'],
                         #['GNNRank'],
                         ['often_compared', 'often_first', 'often_formatted', 'open_source'],
                         ['helm', 'alpaca', 'arena_hard']))

    pool = multiprocessing.Pool()
    results = pool.starmap(ranking_evaluation, tasks)

    results = pd.concat(results, ignore_index=True)
    results.to_csv('../fairpair/data/chatbot_arena_results/chatbotArena_basicMethods_fullMedian_phiPriv_evaluated.csv', index=False)