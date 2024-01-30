from itertools import product
import multiprocessing

import pandas as pd

from fairpair import *
from workers import *

if __name__ == '__main__':

    tasks = list(product(range(10), [RandomSampling, OversampleMinority, RankSampling], [False, True],
                         [randomRankRecovery, davidScore, rankCentrality, serialRank, btl, eigenvectorCentrality, PageRank])) # trial, samplingMethod, apply_bias, rank_using

    try: multiprocessing.set_start_method('spawn') # if it wasn't alrady set, make sure we use the `spawn` method.
    except RuntimeError: pass

    pool = multiprocessing.Pool() # limit the num of processes in order to not overflow the GPU memory
    ranks = pool.starmap(get_correlations, tasks)

    ranks = [result for pool in ranks for result in pool]
    ranks = pd.DataFrame(ranks, columns=['trial', 'iteration', 'skill score', 'average perceived score', 'rank', 'group', 'sampling method', 'ranker', 'bias_applied'])

    ranks.to_csv('./data/others_results/others_correlations_10trials.csv', index=False)