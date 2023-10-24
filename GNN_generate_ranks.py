from itertools import product
import multiprocessing

import pandas as pd

from fairpair import *
from GNN_workers import *

if __name__ == '__main__':

    tasks = list(product(range(10), [RandomSampling, OversampleMinority, RankSampling], [False, True])) # trial, samplingMethod, apply_bias

    try: multiprocessing.set_start_method('spawn') # if it wasn't alrady set, make sure we use the `spawn` method.
    except RuntimeError: pass

    pool = multiprocessing.Pool(processes=5) # limit the num of processes in order to not overflow the GPU memory
    ranks = pool.starmap(get_GNNRank_correlations, tasks)

    ranks = [result for pool in ranks for result in pool]
    ranks = pd.DataFrame(ranks, columns=['trial', 'iteration', 'skill score', 'average perceived score', 'rank', 'group', 'sampling method', 'bias_applied'])

    ranks.to_csv('./data/GNNRank_results/correlations_10trials.csv', index=False)