from scipy.stats import norm
from scipy import integrate

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
    sampler = ProbKnockoutSampling(H, warn=False)
    ranker = RankRecovery(H)
    for j in range(100):
        sampler.apply(iter=10, k=1, min_prob=0.1) #min_prob=0.1
        ranking, other_nodes = ranker.apply() # by default, apply rankCentrality method
        if len(other_nodes) == 0:
            if not connected:
                print(f'Strongly connected after {j*10} iterations.')
                connected = True
            #r_maj = spearmanr(H, ranking, H.majority)
            #if r_maj: accuracy.append((trial, j*5, r_maj[0], 'Majority'))
            #r_min = spearmanr(H, ranking, H.minority)
            #if r_min: accuracy.append((trial, j*5, r_min[0], 'Minority'))
            tau = weighted_tau(H, ranking, H.majority)
            accuracy.append((trial, j*10, tau, 'Majority'))
            tau = weighted_tau(H, ranking, H.minority)
            accuracy.append((trial, j*10, tau, 'Minority'))
            tau = weighted_tau_separate(H, ranking, H.majority)
            accuracy.append((trial, j*10, tau[0], 'Majority within-group'))
            accuracy.append((trial, j*10, tau[1], 'Between groups'))
            tau = weighted_tau_separate(H, ranking, H.minority, calc_between=False)
            accuracy.append((trial, j*10, tau[0], 'Minority within-group'))
    return accuracy


def group_representations(trial:int, representation:float, N=500, Nm=100):
    accuracy = []
    H = FairPairGraph()
    H.generate_groups(N, Nm)
    H.group_assign_scores(nodes=H.majority_nodes, distr=Distributions.normal_distr)
    H.group_assign_scores(nodes=H.minority_nodes, distr=Distributions.normal_distr, loc=0.3, scale=0.2) # give a disadvantage to the minority
    sampler = OversampleMinority(H, warn=False)
    ranker = RankRecovery(H)
    sampler.apply(iter=500, k=1, p=representation)
    ranking, other_nodes = ranker.apply() # by default, apply rankCentrality method
    tau = weighted_tau(H, ranking, H.majority)
    accuracy.append((trial, representation, tau, 'Majority'))
    tau = weighted_tau(H, ranking, H.minority)
    accuracy.append((trial, representation, tau, 'Minority'))
    tau = weighted_tau_separate(H, ranking, H.majority)
    accuracy.append((trial, representation, tau[0], 'Majority within-group'))
    accuracy.append((trial, representation, tau[1], 'Between groups'))
    tau = weighted_tau_separate(H, ranking, H.minority, calc_between=False)
    accuracy.append((trial, representation, tau[0], 'Minority within-group'))
    return accuracy


def minority_ratio(trial:int, ratio:float, N=500):
    accuracy = []
    H = FairPairGraph()
    Nm = int(N*ratio)
    H.generate_groups(N, Nm)
    H.group_assign_scores(nodes=H.majority_nodes, distr=Distributions.normal_distr)
    H.group_assign_scores(nodes=H.minority_nodes, distr=Distributions.normal_distr, loc=0.3, scale=0.2) # give a disadvantage to the minority
    sampler = OversampleMinority(H, warn=False)
    ranker = RankRecovery(H)
    sampler.apply(iter=500, k=1)
    ranking, other_nodes = ranker.apply() # by default, apply rankCentrality method
    tau = weighted_tau(H, ranking, H.majority)
    accuracy.append((trial, ratio, tau, 'Majority'))
    tau = weighted_tau(H, ranking, H.minority)
    accuracy.append((trial, ratio, tau, 'Minority'))
    tau = weighted_tau_separate(H, ranking, H.majority)
    accuracy.append((trial, ratio, tau[0], 'Majority within-group'))
    accuracy.append((trial, ratio, tau[1], 'Between groups'))
    tau = weighted_tau_separate(H, ranking, H.minority, calc_between=False)
    accuracy.append((trial, ratio, tau[0], 'Minority within-group'))
    return accuracy


def get_individual_tau(trial:int, iterations:int, sampling_method:Sampling, N=500, Nm=100):
    accuracy = []
    H = FairPairGraph()
    H.generate_groups(N, Nm)
    H.group_assign_scores(nodes=H.majority_nodes, distr=Distributions.normal_distr)
    H.group_assign_scores(nodes=H.minority_nodes, distr=Distributions.normal_distr, loc=0.3, scale=0.2) # give a disadvantage to the minority
    sampler = sampling_method(H, warn=False, use_exp_BTL=True)
    ranker = RankRecovery(H)
    sampler.apply(iter=iterations, k=1)
    ranking, other_nodes = ranker.apply() # by default, apply rankCentrality method
    if isinstance(sampler, RandomSampling): method = 'Random Sampling'
    elif isinstance(sampler, OversampleMinority): method = 'Oversample Minority'
    elif isinstance(sampler, ProbKnockoutSampling): method = 'ProbKnockout Sampling'
    elif isinstance(sampler, GroupKnockoutSampling): method = 'GroupKnockout Sampling'
    if len(other_nodes) == 0:
        taus = weighted_individual_tau(H, ranking, H.majority)
        accuracy += [(trial, rank, iterations, tau, method, 'Majority') for rank, tau in taus]
        taus = weighted_individual_tau(H, ranking, H.minority)
        accuracy += [(trial, rank, iterations, tau, method, 'Minority') for rank, tau in taus]
    return accuracy


def get_score_range(trial:int, scores_until:float, N=500):
    accuracy = []
    H = FairPairGraph()
    H.generate_groups(N, 0)
    H.group_assign_scores(nodes=H.majority_nodes, distr=Distributions.uniform_distr, low=0.0001, high=scores_until)
    sampler = RandomSampling(H, warn=False, use_exp_BTL=True)
    ranker = RankRecovery(H)
    for j in range(11):
        sampler.apply(iter=100, k=1)
        ranking, other_nodes = ranker.apply() # by default, apply rankCentrality method
        if len(other_nodes) == 0:
            accuracy.append((trial, j*100, scores_until, weighted_tau(H, ranking)))
    return accuracy


def get_star_graph(trial, stariness, N=500, Nm=0):
    accuracy = []
    connected = False
    H = FairPairGraph()
    H.generate_groups(N, Nm)
    H.group_assign_scores(nodes=H.nodes, distr=Distributions.normal_distr)
    sampler = StargraphSampling(H, warn=False)
    ranker = RankRecovery(H)
    for j in range(75):
        sampler.apply(iter=1, k=1, node_prob=stariness)
        # apply davidScore for ranking recovery
        ranking, other_nodes = ranker.apply(rank_using=davidScore) # by default, apply rankCentrality method
        if len(other_nodes) == 0 and not connected:
            print(f'Strongly connected after {j} iterations.')
            connected = True
        accuracy.append((trial, stariness, j, MSE(H, ranking)))
    return accuracy


def get_winning_prob(sigma:float):
    probs = []
    f = lambda y, x: 1/(1 + np.exp(x-y))*norm.pdf(x=x, scale=sigma)*norm.pdf(x=y, scale=sigma) # softmax, but less prone to overflow
    out, error = integrate.dblquad(f, -500, 500, lambda x:x, 500)
    out = 2*out # symmetric, so times 2
    probs.append((sigma, out, error))
    return probs