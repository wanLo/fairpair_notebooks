from fairpair import *

#### just some functions that had to be excluded from jupyter notebooks in order to use multiprocessing.Pool

def get_representation(trial:int, N=500, Nm=100):
    contained = []
    H = FairPairGraph()
    H.generate_groups(N, Nm)
    H.group_assign_scores(nodes=H.majority_nodes, distr=Distributions.normal_distr)
    H.group_assign_scores(nodes=H.minority_nodes, distr=Distributions.normal_distr, loc=0.3, scale=0.2) # give a disadvantage to the minority
    for j in range(75):
        sampler = ProbKnockoutSampling(H, warn=False)
        sampler.apply(iter=1, k=1, min_prob=0.10)
        connected_nodes = max(nx.strongly_connected_components(H), key=len)
        contained.append((trial, j, len([n for n in connected_nodes if n in H.minority_nodes])/Nm, 'Minority'))
        contained.append((trial, j, len([n for n in connected_nodes if n in H.majority_nodes])/(N-Nm), 'Majority'))
    return contained


def get_accuracy(trial:int, N=500, Nm=100):
    accuracy = []
    connected = False
    H = FairPairGraph()
    H.generate_groups(N, Nm)
    H.group_assign_scores(nodes=H.majority_nodes, distr=Distributions.normal_distr)
    H.group_assign_scores(nodes=H.minority_nodes, distr=Distributions.normal_distr, loc=0.3, scale=0.2) # give a disadvantage to the minority
    for j in range(75):
        sampler = RandomSampling(H, warn=False)
        sampler.apply(iter=1, k=1)
        # apply davidScore for ranking recovery
        ranker = RankRecovery(H)
        ranking, other_nodes = ranker.apply(rank_using=davidScore) # by default, apply rankCentrality method
        if len(other_nodes) == 0 and not connected:
            print(f'Strongly connected after {j} iterations.')
            connected = True
        accuracy.append((trial, j, ranker.calc_MSE(H.majority, ranking), 'Majority'))
        accuracy.append((trial, j, ranker.calc_MSE(H.minority, ranking), 'Minority'))
    return accuracy