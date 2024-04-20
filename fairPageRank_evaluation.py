from itertools import product
import multiprocessing

import pandas as pd
import seaborn as sns
import matplotlib as plt
import scipy.sparse as sp
import numpy as np

from fairpair import *

from accuracy_workers import *

accuracy = pd.DataFrame()

for apply_bias in [True, False]:
    for sampling_method in [RandomSampling, Oversampling, RankSampling]:
        
        tasks = list(product(range(10), [sampling_method], ['fairPageRank'], [apply_bias])) #trial, samplingMethod, ranking_method, apply_bias

        if __name__ == '__main__':
            pool = multiprocessing.Pool(processes=10)
            accuracy_tmp = pool.starmap(get_method_accuracy, tasks)
            accuracy_tmp = [result for pool in accuracy_tmp for result in pool]
            accuracy_tmp = pd.DataFrame(accuracy_tmp, columns=['trial', 'iteration', 'value', 'bias_applied', 'sampling method', 'Ranking Method', 'metric', 'group'])

            accuracy = pd.concat([accuracy, accuracy_tmp], ignore_index=True).reset_index(drop=True)
            accuracy.to_csv('../fairpair/data/general_accuracy/generalAccuracy_fairPR_multigraph_intermed.csv', index=False)

accuracy.to_csv('../fairpair/data/general_accuracy/generalAccuracy_fairPR_multigraph.csv', index=False)