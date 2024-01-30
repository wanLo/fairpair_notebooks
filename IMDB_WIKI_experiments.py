import networkx as nx
import numpy as np
import pandas as pd
import torch

from itertools import product
import multiprocessing

from fairpair import *

import sys
sys.path.append('../GNNRank/')
from src.param_parser import ArgsNamespace # just import the class, not the parser
from src.Trainer import Trainer

def load_dataset(ground_truth_file:str, pairwise_file:str) -> nx.DiGraph:
    
    pairwise_df = pd.read_csv(pairwise_file)
    df = pd.read_csv(ground_truth_file, index_col=0)

    win_lose = pairwise_df[['left', 'right', 'label']].copy().rename(columns={'label': 'win'})
    win_lose.loc[win_lose.left == win_lose.win, 'lose'] = win_lose.right
    win_lose.loc[win_lose.right == win_lose.win, 'lose'] = win_lose.left
    win_lose['lose'] = win_lose['lose'].astype(int)

    G = nx.from_pandas_edgelist(win_lose, source='lose', target='win', create_using=nx.DiGraph)
    G2 = G.subgraph(nodes=df.index).copy()  # a smaller graph with only the images we're fairly certain about

    tmp_df = df[['age', 'gender']].rename(columns={'age': 'skill'})
    tmp_df['minority'] = tmp_df['gender'] == 'Female'

    nx.set_node_attributes(G2, tmp_df[['skill', 'minority']].to_dict(orient='index'))

    return G2


def subsample_and_rank(trial:int, sampling_strategy='randomSampling', rank_using=rankCentrality,
        ground_truth_file='./data/imdb-wiki/ground_truth_selected_stratified.csv',
        pairwise_file='./data/imdb-wiki/comparisons_cleaned.csv') -> list:

    rng = np.random.default_rng()
    
    # init the ground-truth data
    G = load_dataset(ground_truth_file, pairwise_file)
    df = pd.read_csv(ground_truth_file, index_col=0)

    G = nx.convert_node_labels_to_integers(G)  # nodes need to be named 0…n in order for fairPageRank to work
    df = df.reset_index(drop=True)

    # init an empty graph to subsample into
    FPG = FairPairGraph()
    FPG.add_nodes_from(G.nodes(data=True))

    # init the ranker & ranking results
    ranking = None
    ranker = RankRecovery(FPG)
    ranks = []

    # settings for rankSampling
    min_prob=1
    factor=6

    step = 10
    for j in range(int(1000/step)):

        # sample [step] times before evaluating again
        for i in range(step):

            # subsample according to selected strategy
            if sampling_strategy == 'randomSampling':
                random_nodes = df.sample(int(len(df)*0.2))['gender'].index
                H = G.subgraph(random_nodes)
            elif sampling_strategy == 'oversampling':
                random_nodes_0 = df[df.gender == 'Female'].sample(int(len(df)*0.2*0.75))['gender'].index
                random_nodes_1 = df[df.gender == 'Male'].sample(int(len(df)*0.2*0.25))['gender'].index
                H = G.subgraph(list(random_nodes_0) + list(random_nodes_1))
            elif sampling_strategy == 'rankSampling':
                if ranking == None:  # use randomSampling as a fallback
                    random_nodes = df.sample(int(len(df)*0.2))['gender'].index
                    H = G.subgraph(random_nodes)
                else:
                    ranking2 = scores_to_rank(ranking, invert=False) # we want higher ranks for the top for higher probabilities
                    rates = [ranking2[node] for node in df.index] # ranking result in the same order that they appear in the df
                    rates = [np.exp(rate/len(rates)*factor)+min_prob for rate in rates] # rescale to (0,factor) to finetune exponential importance
                    normalized_rates = [rate/sum(rates) for rate in rates] # must sum to 1
                    selected_nodes = df.sample(n=int(len(df)*0.2), weights=normalized_rates)
                    H = G.subgraph(selected_nodes)

            # randomly select edges among the given nodes to add to the graph
            selected_edges = rng.choice(H.edges, size=len(H)//2, axis=0)
            FPG.add_edges_from(list(map(tuple, selected_edges)))
            nx.set_edge_attributes(FPG, 1, 'weight')  # an edge either exists or it doesn't

        # recover a ranking if the graph is at least weakly connected
        if nx.is_weakly_connected(FPG):
            if rank_using == 'fairPageRank':
                current_proc = multiprocessing.current_process()
                path = './data/fairPageRank/tmp_' + str(current_proc._identity[0])
                ranking, other_nodes = ranker.apply(rank_using=rank_using, path=path)
                ranker_name = 'fairPageRank'
            else:
                ranking, other_nodes = ranker.apply(rank_using=rank_using)
                ranker_name = rank_using.__name__
             
            ranking_as_ranks = scores_to_rank(ranking, invert=False) # invert=True
            for node, data in FPG.majority.nodes(data=True):
                ranks.append((trial, j*step+step, data['skill'], ranking_as_ranks[node], 'Privileged',
                                sampling_strategy, ranker_name))
            for node, data in FPG.minority.nodes(data=True):
                ranks.append((trial, j*step+step, data['skill'], ranking_as_ranks[node], 'Unprivileged',
                                sampling_strategy, ranker_name))
        
            if j%10 == 9:
                print(f'trial {trial}, {sampling_strategy}, {ranker_name}: finished {j*step+step} iterations.')
    
    return ranks


def subsample_and_rank_GNNRank(trial:int, sampling_strategy:str,
                               ground_truth_file='./data/imdb-wiki/ground_truth_selected.csv',
                               pairwise_file='./data/imdb-wiki/comparisons_cleaned.csv') -> list:
    # customize `dataset` to properly save the model
    # get optimal settings from the paper: `baseline`, `pretrain_with`, `train_with`, `upset_margin_coeff`
    # use defaults for: `early_stopping`, `epochs`
    # handle output cleverly using: `load_only=True`, `regenerate_data=True`, `be_silent=True`
    args = ArgsNamespace(AllTrain=True, ERO_style='uniform', F=70, Fiedler_layer_num=5, K=20, N=350, SavePred=False, all_methods=['DIGRAC', 'ib'],
                     alpha=1.0, baseline='syncRank', cuda=True, data_path='/home/georg/fairpair/GNNRank/src/../data/',
                     dataset=f'IMDB-WIKI_correlations/{sampling_strategy}', be_silent=True,
                     debug=False, device=torch.device(type='cuda'), dropout=0.5, early_stopping=200, epochs=1000, eta=0.1, fill_val=0.5, hidden=8, hop=2,
                     load_only=True, log_root='/home/georg/fairpair/GNNRank/src/../logs/', lr=0.01, no_cuda=False, num_trials=1, optimizer='Adam', p=0.05,
                     pretrain_epochs=50, pretrain_with='dist', regenerate_data=True, season=1990, seed=31, seeds=[10], sigma=1.0, tau=0.5, test_ratio=1,
                     train_ratio=1, train_with='proximal_baseline', trainable_alpha=False, upset_margin=0.01, upset_margin_coeff=0, upset_ratio_coeff=1.0, weight_decay=0.0005)
    torch.manual_seed(args.seed)

    rng = np.random.default_rng()
    
    # init the ground-truth data
    G = load_dataset(ground_truth_file, pairwise_file)
    df = pd.read_csv(ground_truth_file, index_col=0)

    G = nx.convert_node_labels_to_integers(G)  # nodes need to be named 0…n in order for fairPageRank to work
    df = df.reset_index(drop=True)

    # init an empty graph to subsample into
    FPG = FairPairGraph()
    FPG.add_nodes_from(G.nodes(data=True))

    ranking = None
    step = 50  # coarse evaluation if we re-train the model after each step
    connected = False
    ranks = []

    # settings for rankSampling
    min_prob=1
    factor=6

    # gradually infer ranking giving the initially trained model
    for j in range(int(1000/step)):

        # sample [step] times before evaluating again
        for i in range(step):

            # subsample according to selected strategy
            if sampling_strategy == 'randomSampling':
                random_nodes = df.sample(int(len(df)*0.2))['gender'].index
                H = G.subgraph(random_nodes)
            elif sampling_strategy == 'oversampling':
                random_nodes_0 = df[df.gender == 'Female'].sample(int(len(df)*0.2*0.75))['gender'].index
                random_nodes_1 = df[df.gender == 'Male'].sample(int(len(df)*0.2*0.25))['gender'].index
                H = G.subgraph(list(random_nodes_0) + list(random_nodes_1))
            elif sampling_strategy == 'rankSampling':
                if ranking == None:  # use randomSampling as a fallback
                    random_nodes = df.sample(int(len(df)*0.2))['gender'].index
                    H = G.subgraph(random_nodes)
                else:
                    ranking2 = scores_to_rank(ranking, invert=False) # we want higher ranks for the top for higher probabilities
                    rates = [ranking2[node] for node in df.index] # ranking result in the same order that they appear in the df
                    rates = [np.exp(rate/len(rates)*factor)+min_prob for rate in rates] # rescale to (0,factor) to finetune exponential importance
                    normalized_rates = [rate/sum(rates) for rate in rates] # must sum to 1
                    selected_nodes = df.sample(n=int(len(df)*0.2), weights=normalized_rates)
                    H = G.subgraph(selected_nodes)

            # randomly select edges among the given nodes to add to the graph
            selected_edges = rng.choice(H.edges, size=len(H)//2, axis=0)
            FPG.add_edges_from(list(map(tuple, selected_edges)))
            nx.set_edge_attributes(FPG, 1, 'weight')  # an edge either exists or it doesn't
        
        if (nx.is_weakly_connected(FPG)):

            #if not connected:
                # train once
            
            # re-train the model after each step
            print(f'{sampling_strategy}: training after {j*step+step} iterations…')
            adj = nx.linalg.graphmatrix.adjacency_matrix(FPG, weight='weight') # returns a sparse matrix
            trainer = Trainer(args, random_seed=10, save_name_base='test', adj=adj) # initialize with the given adjacency matrix
            save_path_best, save_path_latest = trainer.train(model_name='ib')
            
            #adj = nx.linalg.graphmatrix.adjacency_matrix(FPG, weight='weight') # returns a sparse matrix
            #trainer = Trainer(args, random_seed=10, save_name_base='test', adj=adj) # initialize with the given adjacency matrix
            print(f'{sampling_strategy}: predicting after {j*step+step} iterations…')
            score, pred_label = trainer.predict_nn(model_name='ib', model_path=save_path_best, A=None, GNN_variant='proximal_baseline')
            ranking = {key: 1-score[0] for key, score in enumerate(score.cpu().detach().numpy())}

            ranking_as_ranks = scores_to_rank(ranking, invert=False)
            for node, data in FPG.majority.nodes(data=True):
                ranks.append((trial, j*step+step, data['skill'], ranking_as_ranks[node], 'Privileged', sampling_strategy, 'GNNRank'))
            for node, data in FPG.minority.nodes(data=True):
                ranks.append((trial, j*step+step, data['skill'], ranking_as_ranks[node], 'Unprivileged', sampling_strategy, 'GNNRank'))
        
        print(f'{sampling_strategy}: finished {j*step+step} iterations.')
    
    ranks_df = pd.DataFrame(ranks, columns=['trial', 'iteration', 'skill score', 'rank', 'group', 'sampling method', 'ranker'])
    ranks_df.to_csv(f'./data/GNNRank_intermed/IMDB-WIKI_trial{trial}_{sampling_strategy}.csv', index=False)

    return ranks


def rank_full_dataset(rank_using=rankCentrality,
        ground_truth_file='./data/imdb-wiki/ground_truth_selected.csv',
        pairwise_file='./data/imdb-wiki/comparisons_cleaned.csv') -> list:
    
    G = load_dataset(ground_truth_file, pairwise_file)
    G = nx.convert_node_labels_to_integers(G)  # nodes need to be named 0…n in order for fairPageRank to work
    G = FairPairGraph(G)
    nx.set_edge_attributes(G, 1, 'weight')  # an edge either exists or it doesn't

    ranker = RankRecovery(G)
    ranks = []

    if rank_using == 'fairPageRank':
        current_proc = multiprocessing.current_process()
        path = './data/fairPageRank/tmp_' + str(current_proc._identity[0])
        ranking, other_nodes = ranker.apply(rank_using=rank_using, path=path)
        ranker_name = 'fairPageRank'
    else:
        ranking, other_nodes = ranker.apply(rank_using=rank_using)
        ranker_name = rank_using.__name__
    
    ranking_as_ranks = scores_to_rank(ranking, invert=False) # invert=True
    for node, data in G.majority.nodes(data=True):
        ranks.append((0, 0, data['skill'], ranking_as_ranks[node], 'Privileged', 'full dataset', ranker_name))
    for node, data in G.minority.nodes(data=True):
        ranks.append((0, 0, data['skill'], ranking_as_ranks[node], 'Unprivileged', 'full dataset', ranker_name))
    
    return ranks


def rank_full_dataset_GNNRank(
        ground_truth_file='./data/imdb-wiki/ground_truth_selected.csv',
        pairwise_file='./data/imdb-wiki/comparisons_cleaned.csv') -> list:
    
    # customize `dataset` to properly save the model
    # get optimal settings from the paper: `baseline`, `pretrain_with`, `train_with`, `upset_margin_coeff`
    # use defaults for: `early_stopping`, `epochs`
    # handle output cleverly using: `load_only=True`, `regenerate_data=True`, `be_silent=True`
    args = ArgsNamespace(AllTrain=True, ERO_style='uniform', F=70, Fiedler_layer_num=5, K=20, N=350, SavePred=False, all_methods=['DIGRAC', 'ib'],
                     alpha=1.0, baseline='syncRank', cuda=True, data_path='/home/georg/fairpair/GNNRank/src/../data/',
                     dataset=f'IMDB-WIKI_correlations/full_dataset', be_silent=True,
                     debug=False, device=torch.device(type='cuda'), dropout=0.5, early_stopping=200, epochs=1000, eta=0.1, fill_val=0.5, hidden=8, hop=2,
                     load_only=True, log_root='/home/georg/fairpair/GNNRank/src/../logs/', lr=0.01, no_cuda=False, num_trials=1, optimizer='Adam', p=0.05,
                     pretrain_epochs=50, pretrain_with='dist', regenerate_data=True, season=1990, seed=31, seeds=[10], sigma=1.0, tau=0.5, test_ratio=1,
                     train_ratio=1, train_with='proximal_baseline', trainable_alpha=False, upset_margin=0.01, upset_margin_coeff=0, upset_ratio_coeff=1.0, weight_decay=0.0005)
    
    G = load_dataset(ground_truth_file, pairwise_file)
    G = nx.convert_node_labels_to_integers(G)  # nodes need to be named 0…n in order for fairPageRank to work
    G = FairPairGraph(G)
    nx.set_edge_attributes(G, 1, 'weight')  # an edge either exists or it doesn't

    ranks = []

    adj = nx.linalg.graphmatrix.adjacency_matrix(G, weight='weight') # returns a sparse matrix
    trainer = Trainer(args, random_seed=10, save_name_base='test', adj=adj) # initialize with the given adjacency matrix
    save_path_best, save_path_latest = trainer.train(model_name='ib')

    score, pred_label = trainer.predict_nn(model_name='ib', model_path=save_path_best, A=None, GNN_variant='proximal_baseline')
    ranking = {key: 1-score[0] for key, score in enumerate(score.cpu().detach().numpy())}
    
    ranking_as_ranks = scores_to_rank(ranking, invert=False) # invert=True
    for node, data in G.majority.nodes(data=True):
        ranks.append((0, 0, data['skill'], ranking_as_ranks[node], 'Privileged', 'full dataset', 'GNNRank'))
    for node, data in G.minority.nodes(data=True):
        ranks.append((0, 0, data['skill'], ranking_as_ranks[node], 'Unprivileged', 'full dataset', 'GNNRank'))
    
    return ranks



if __name__ == '__main__':

    #tasks = list(product(range(10), ['randomSampling', 'oversampling', 'rankSampling'],
    #                    [randomRankRecovery, davidScore, rankCentrality, 'fairPageRank'])) # trial, sampling_strategy, rank_using
    #tasks = list(product(range(8), ['randomSampling', 'oversampling', 'rankSampling'],
    #                     [rankCentrality])) # trial, sampling_strategy, rank_using
    #tasks = list(product(range(10), ['randomSampling', 'oversampling', 'rankSampling']))
    #tasks = [randomRankRecovery, davidScore, rankCentrality, 'fairPageRank'] # only rank_using

    #try: multiprocessing.set_start_method('spawn') # if it wasn't alrady set, make sure we use the `spawn` method.
    #except RuntimeError: pass

    #pool = multiprocessing.Pool() # limit the num of processes in order to not overflow the GPU memory
    #ranks = pool.map(rank_full_dataset, tasks)

    #ranks = [result for pool in ranks for result in pool]
    ranks = rank_full_dataset_GNNRank()

    ranks = pd.DataFrame(ranks, columns=['trial', 'iteration', 'skill score', 'rank', 'group', 'sampling method', 'ranker'])

    ranks.to_csv('./data/imdb-wiki_results/GNNRank_correlations_full_dataset.csv', index=False)