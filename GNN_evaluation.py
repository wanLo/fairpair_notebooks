from itertools import product
import multiprocessing

import pandas as pd

from fairpair import *
from GNN_workers import *


if __name__ == '__main__':

    tasks = list(product(range(10), [RandomSampling, OversampleMinority, RankSampling], [False, True])) # trial, samplingMethod, apply_bias

    try: multiprocessing.set_start_method('spawn') # if it wasn't alrady set, make sure we use the `spawn` method.
    except RuntimeError: pass

    pool = multiprocessing.Pool(processes=10) # limit the num of processes in order to not overflow the GPU memory
    accuracy = pool.starmap(get_GNNRank_weightedTau, tasks)

    accuracy = [result for pool in accuracy for result in pool]
    accuracy = pd.DataFrame(accuracy, columns=['trial', 'iteration', 'value', 'bias_applied', 'sampling method', 'metric', 'group'])
    accuracy.to_csv('./data/general_accuracy/generalAccuracy_GNNRank_coarse_extended.csv', index=False)