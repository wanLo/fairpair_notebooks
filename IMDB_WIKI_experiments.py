import networkx as nx
import numpy as np
import pandas as pd

from itertools import product
import multiprocessing

from fairpair import *

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
        ground_truth_file='./data/imdb-wiki/ground_truth_selected.csv',
        pairwise_file='./data/imdb-wiki/comparisons_cleaned.csv') -> list:

    rng = np.random.default_rng()
    
    # init the ground-truth data
    G = load_dataset(ground_truth_file, pairwise_file)
    df = pd.read_csv(ground_truth_file, index_col=0)

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

        # recover a ranking if the graph is at least weakly connected
        if nx.is_weakly_connected(FPG):
            ranking, other_nodes = ranker.apply(rank_using=rank_using)
            
            ranking_as_ranks = scores_to_rank(ranking, invert=True)
            for node, data in FPG.majority.nodes(data=True):
                ranks.append((trial, j*step+step, data['skill'], ranking_as_ranks[node], 'Privileged',
                                sampling_strategy, rank_using.__name__))
            for node, data in FPG.minority.nodes(data=True):
                ranks.append((trial, j*step+step, data['skill'], ranking_as_ranks[node], 'Unprivileged',
                                sampling_strategy, rank_using.__name__))
        
            if j%10 == 9:
                print(f'trial {trial}, {sampling_strategy}, {rank_using.__name__}: finished {j*step+step} iterations.')
    
    return ranks




if __name__ == '__main__':

    #tasks = list(product(range(10), ['randomSampling', 'oversampling', 'rankSampling'],
    #                     [randomRankRecovery, davidScore, rankCentrality])) # trial, sampling_strategy, rank_using
    tasks = list(product(range(8), ['randomSampling', 'oversampling', 'rankSampling'],
                         [rankCentrality])) # trial, sampling_strategy, rank_using

    #try: multiprocessing.set_start_method('spawn') # if it wasn't alrady set, make sure we use the `spawn` method.
    #except RuntimeError: pass

    pool = multiprocessing.Pool() # limit the num of processes in order to not overflow the GPU memory
    ranks = pool.starmap(subsample_and_rank, tasks)

    ranks = [result for pool in ranks for result in pool]
    ranks = pd.DataFrame(ranks, columns=['trial', 'iteration', 'skill score', 'rank', 'group', 'sampling method', 'ranker'])

    ranks.to_csv('./data/imdb-wiki_results/test_rankCentrality_correlations_1trial.csv', index=False)