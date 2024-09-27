import networkx as nx
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from itertools import product
import multiprocessing

from fairpair import *

import sys
sys.path.append('../GNNRank/')
from src.param_parser import ArgsNamespace # just import the class, not the parser
from src.Trainer import Trainer


def load_dataset(ground_truth_file:str, comparisons_file:str, group_attribute:str) -> nx.DiGraph:
    
    wins_df = pd.read_csv(comparisons_file, index_col=0)
    benchmark_df = pd.read_csv(ground_truth_file, index_col=0) # model names as index

    G = nx.from_pandas_adjacency(wins_df, create_using=nx.DiGraph)
    #G2 = G.subgraph(nodes=benchmark_df.index).copy()  # a smaller graph with only the models that we have benchmark data for
    G2 = G # recovery on the full dataset

    attr_df = benchmark_df[['score', group_attribute]].rename(columns={'score': 'skill'})
    attr_df['unpriv'] = attr_df[group_attribute] == 0
    attr_df = attr_df[['skill', 'unpriv']].to_dict(orient='index')

    nx.set_node_attributes(G2, attr_df)

    return G2


def rank_full_dataset(
        rank_using=rankCentrality,
        group_attribute='long_response',
        benchmark='helm',
        comparisons_file='../fairpair/data/chatbot_arena/comparisons_cleaned.csv') -> list:
    
    ground_truth_file=f'../fairpair/data/chatbot_arena/{benchmark}_combined.csv'
    
    G = load_dataset(ground_truth_file, comparisons_file, group_attribute)
    G = nx.convert_node_labels_to_integers(G)  # nodes need to be named 0…n in order for fairPageRank to work
    G = FairPairGraph(G)
    nx.set_edge_attributes(G, 1, 'weight')  # an edge either exists or it doesn't

    ranker = RankRecovery(G)
    ranks = pd.DataFrame(columns=['trial', 'iteration', 'skill score', 'rank', 'group',
                                  'sampling method', 'ranker', 'group attribute', 'benchmark'])
    
    if rank_using in [randomRankRecovery, btl]: seeds = [93, 55, 31, 4, 97, 13, 51, 5, 75, 74] # multiple runs because of randomness
    else: seeds = [1]

    for trial, seed in enumerate(seeds):
        if rank_using == 'fairPageRank':
            current_proc = multiprocessing.current_process()
            path = '../fairpair/data/fairPageRank/tmp_' + str(current_proc._identity[0])
            ranking, other_nodes = ranker.apply(rank_using=rank_using, path=path)
            ranker_name = 'fairPageRank'
        elif rank_using == randomRankRecovery:
            ranking, other_nodes = ranker.apply(rank_using=rank_using, seed=seed)
            ranker_name = rank_using.__name__
        else:
            ranking, other_nodes = ranker.apply(rank_using=rank_using)
            ranker_name = rank_using.__name__
        
        ranking_as_ranks = scores_to_rank(ranking, invert=True) # invert=True
        for node, data in G.priv.nodes(data=True):
            if 'skill' in data: skill = data['skill']
            else: skill = np.NaN
            ranks = pd.concat([ranks, pd.DataFrame([[trial, 0, skill, ranking_as_ranks[node],
                                                    'Privileged', 'full dataset', ranker_name, group_attribute, benchmark]], columns=ranks.columns)])
        for node, data in G.unpriv.nodes(data=True):
            if 'skill' in data: skill = data['skill']
            else: skill = np.NaN
            ranks = pd.concat([ranks, pd.DataFrame([[trial, 0, skill, ranking_as_ranks[node],
                                                    'Unprivileged', 'full dataset', ranker_name, group_attribute, benchmark]], columns=ranks.columns)])
    
    return ranks


def rank_full_dataset_GNNRank(
        group_attribute='long_response',
        benchmark='helm',
        comparisons_file='../fairpair/data/chatbot_arena/comparisons_cleaned.csv') -> list:
    
    ground_truth_file=f'../fairpair/data/chatbot_arena/{benchmark}_combined.csv'
    
    # customize `dataset` to properly save the model
    # get optimal settings from the paper: `baseline`, `pretrain_with`, `train_with`, `upset_margin_coeff`
    # use defaults for: `early_stopping`, `epochs`
    # handle output cleverly using: `load_only=True`, `regenerate_data=True`, `be_silent=True`
    # set K=10 for smaller dataset (used to be 20)
    args = ArgsNamespace(AllTrain=True, ERO_style='uniform', F=70, Fiedler_layer_num=5, K=20, N=350, SavePred=False, all_methods=['DIGRAC', 'ib'],
                     alpha=1.0, baseline='syncRank', cuda=True, data_path='/home/georg/fairpair/GNNRank/src/../data/',
                     dataset=f'IMDB-WIKI_correlations/full_dataset', be_silent=True,
                     debug=False, device=torch.device(type='cuda'), dropout=0.5, early_stopping=200, epochs=1000, eta=0.1, fill_val=0.5, hidden=8, hop=2,
                     load_only=True, log_root='/home/georg/fairpair/GNNRank/src/../logs/', lr=0.01, no_cuda=False, num_trials=1, optimizer='Adam', p=0.05,
                     pretrain_epochs=50, pretrain_with='dist', regenerate_data=True, season=1990, seed=31, seeds=[10], sigma=1.0, tau=0.5, test_ratio=1,
                     train_ratio=1, train_with='proximal_baseline', trainable_alpha=False, upset_margin=0.01, upset_margin_coeff=0, upset_ratio_coeff=1.0, weight_decay=0.0005)
    
    G = load_dataset(ground_truth_file, comparisons_file, group_attribute)
    G = nx.convert_node_labels_to_integers(G)  # nodes need to be named 0…n in order for fairPageRank to work
    G = FairPairGraph(G)
    nx.set_edge_attributes(G, 1, 'weight')  # an edge either exists or it doesn't

    ranks = pd.DataFrame(columns=['trial', 'iteration', 'skill score', 'rank', 'group', 'sampling method', 'ranker', 'group attribute', 'benchmark'])

    adj = nx.linalg.graphmatrix.adjacency_matrix(G, weight='weight') # returns a sparse matrix
    trainer = Trainer(args, random_seed=10, save_name_base='test', adj=adj) # initialize with the given adjacency matrix
    save_path_best, save_path_latest = trainer.train(model_name='ib')

    score, pred_label = trainer.predict_nn(model_name='ib', model_path=save_path_best, A=None, GNN_variant='proximal_baseline')
    ranking = {key: 1-score[0] for key, score in enumerate(score.cpu().detach().numpy())}
    
    ranking_as_ranks = scores_to_rank(ranking, invert=True) # invert=True
    for node, data in G.priv.nodes(data=True):
        if 'skill' in data: skill = data['skill']
        else: skill = np.NaN
        ranks = pd.concat([ranks, pd.DataFrame([[0, 0, skill, ranking_as_ranks[node],
                                                'Privileged', 'full dataset', 'GNNRank', group_attribute, benchmark]], columns=ranks.columns)])
    for node, data in G.unpriv.nodes(data=True):
        if 'skill' in data: skill = data['skill']
        else: skill = np.NaN
        ranks = pd.concat([ranks, pd.DataFrame([[0, 0, skill, ranking_as_ranks[node],
                                                'Unprivileged', 'full dataset', 'GNNRank', group_attribute, benchmark]], columns=ranks.columns)])
    
    return ranks


# custom function to simulate starmap
# see https://stackoverflow.com/a/67845088
def full_dataset_starmap(args):
    return rank_full_dataset(*args)

def full_dataset_GNNRank_starmap(args):
    return rank_full_dataset_GNNRank(*args)



if __name__ == '__main__':

    tasks = list(product([randomRankRecovery, davidScore, rankCentrality, 'fairPageRank', btl],
                         ['often_compared', 'often_first', 'often_formatted', 'open_source'],
                         ['helm', 'alpaca', 'arena_hard'])) # rank_using, group_attribute, benchmark
    
    # for GNNRank
    #tasks = list(product(['often_compared', 'often_first', 'often_formatted', 'open_source'],
    #                     ['helm', 'alpaca', 'arena_hard'])) # group_attribute, benchmark

    try: multiprocessing.set_start_method('spawn') # if it wasn't alrady set, make sure we use the `spawn` method.
    except RuntimeError: pass

    with multiprocessing.Pool() as pool:
        pbar = tqdm(pool.imap(full_dataset_starmap, tasks), total=len(tasks))
        results = list(pbar) # list invokes the evaluation
    
    ranks = pd.concat(results, ignore_index=True)

    #ranks = rank_full_dataset_GNNRank('often_first', 'helm')

    ranks.to_csv('../fairpair/data/chatbot_arena_results/basicMethods_correlations_fullDataset.csv', index=False)