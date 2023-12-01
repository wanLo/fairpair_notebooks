from itertools import product
import multiprocessing

import pandas as pd

from post_processing_workers import *


if __name__ == '__main__':

    tasks = list(product(range(10), ['RandomSampling', 'OversampleMinority', 'RankSampling'],
                         ['davidScore', 'randomRankRecovery', 'rankCentrality', 'GNNRank'], [False, True])) # trial, sampling_method, ranking_method, apply_bias

    #tasks = list(product(range(10), ['RandomSampling'], ['davidScore'], [True]))

    pool = multiprocessing.Pool()
    accuracy = pool.starmap(post_process, tasks)

    accuracy = [result for pool in accuracy for result in pool]
    accuracy = pd.DataFrame(accuracy, columns=['trial', 'iteration', 'value', 'bias_applied', 'sampling strategy', 'recovery method', 'metric', 'group'])
    accuracy.to_csv('./data/post_processing/FAstarIRp60_10trials.csv', index=False)