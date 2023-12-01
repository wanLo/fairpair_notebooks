from scipy.stats import norm
from scipy import integrate
import mpmath
import numpy as np

from fairpair import *

#### just some functions that had to be excluded from jupyter notebooks in order to use multiprocessing.Pool

def get_representation(trial:int, sampling_method=RandomSampling):
    contained = []
    H = FairPairGraph()
    H.generate_groups(400, 200)
    H.assign_skills(loc=0, scale=0.86142674) # general skill distribution
    H.assign_bias(nodes=H.minority_nodes, loc=-1.43574282, scale=0.43071336) # add bias to unprivileged group
    sampler = sampling_method(H, warn=False)
    for j in range(50):
        if isinstance(sampler, RandomSampling):
            method = 'Random Sampling'
            sampler.apply(iter=1, k=1)
        elif isinstance(sampler, OversampleMinority):
            method = 'Oversample 75 %'
            sampler.apply(iter=1, k=1, p=0.75)
        connected_nodes = max(nx.strongly_connected_components(H), key=len)
        #contained.append((trial, j, len([n for n in connected_nodes if n in H.minority_nodes])/Nm, 'Minority'))
        contained.append((trial, j, len(connected_nodes)/400, method))
    return contained


def get_accuracy(trial:int, N=400, Nm=200):
    accuracy = []
    connected = False
    H = FairPairGraph()
    H.generate_groups(N, Nm) # same size groups
    H.assign_skills(loc=0, scale=0.86142674) # general skill distribution
    H.assign_bias(nodes=H.minority_nodes, loc=-1.43574282, scale=0.43071336) # add bias to unprivileged group
    #H.assign_bias(nodes=H.minority_nodes, loc=-1.43574282, scale=0)
    sampler = RandomSampling(H, warn=False)
    ranker = RankRecovery(H)
    ranking = None
    for j in range(100):
        sampler.apply(iter=10, k=1) # p=0.75
        ranking, other_nodes = ranker.apply(rank_using='fairPageRank', path=f'data/tmp{trial}') # by default, apply rankCentrality method
        if len(other_nodes) == 0:
            if not connected:
                print(f'Strongly connected after {j*10} iterations.')
                connected = True
            tau = weighted_tau(H, ranking, H.majority)
            accuracy.append((trial, j*10, tau, 'Privileged'))
            tau = weighted_tau(H, ranking, H.minority)
            accuracy.append((trial, j*10, tau, 'Unprivileged'))
            tau = weighted_tau_separate(H, ranking, H.majority)
            accuracy.append((trial, j*10, tau[0], 'Privileged within-group'))
            accuracy.append((trial, j*10, tau[1], 'Between groups'))
            tau = weighted_tau_separate(H, ranking, H.minority, calc_between=False)
            accuracy.append((trial, j*10, tau[0], 'Unprivileged within-group'))
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


def get_individual_tau(trial:int, sampling_method:Sampling, N=500, Nm=100):
    accuracy = []
    H = FairPairGraph()
    H.generate_groups(N, Nm)
    H.group_assign_scores(nodes=H.majority_nodes, distr=Distributions.normal_distr)
    H.group_assign_scores(nodes=H.minority_nodes, distr=Distributions.normal_distr, loc=0.3, scale=0.2) # give a disadvantage to the minority
    sampler = sampling_method(H, warn=False, use_exp_BTL=True)
    ranker = RankRecovery(H)
    ranking = None
    for j in range(11):
        if isinstance(sampler, RankSampling):
            sampler.apply(iter=100, k=1, ranking=ranking)
        else: sampler.apply(iter=100, k=1)
        ranking, other_nodes = ranker.apply() # by default, apply rankCentrality method
        if isinstance(sampler, RandomSampling): method = 'Random Sampling'
        elif isinstance(sampler, OversampleMinority): method = 'Oversample Minority'
        elif isinstance(sampler, ProbKnockoutSampling): method = 'ProbKnockout Sampling'
        elif isinstance(sampler, RankSampling): method = 'Rank Sampling'
        elif isinstance(sampler, GroupKnockoutSampling): method = 'GroupKnockout Sampling'
        if len(other_nodes) == 0:
            taus = weighted_individual_tau(H, ranking, H.majority)
            accuracy += [(trial, rank, j*100, tau, method, 'Majority') for rank, tau in taus]
            taus = weighted_individual_tau(H, ranking, H.minority)
            accuracy += [(trial, rank, j*100, tau, method, 'Minority') for rank, tau in taus]
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


def get_topk_tau(trial:int, sampling_method:Sampling, topk=[10,50,100,200,400], N=400, Nm=200):
    accuracy = []
    H = FairPairGraph()
    H.generate_groups(N, Nm) # same size groups
    H.assign_skills(loc=0, scale=0.86142674) # general skill distribution
    H.assign_bias(nodes=H.minority_nodes, loc=-1.43574282, scale=0.43071336) # add bias to unprivileged group
    sampler = sampling_method(H, warn=False, use_exp_BTL=True)
    ranker = RankRecovery(H)
    ranking = None
    for j in range(101):
        if isinstance(sampler, RankSampling):
            sampler.apply(iter=10, k=1, ranking=ranking)
        elif isinstance(sampler, OversampleMinority):
            sampler.apply(iter=10, k=1, p=0.75)
        else: sampler.apply(iter=10, k=1)
        ranking, other_nodes = ranker.apply() # by default, apply rankCentrality method
        if isinstance(sampler, RandomSampling): method = 'Random Sampling'
        elif isinstance(sampler, OversampleMinority): method = 'Oversample Minority'
        elif isinstance(sampler, ProbKnockoutSampling): method = 'ProbKnockout Sampling'
        elif isinstance(sampler, RankSampling): method = 'Rank Sampling'
        elif isinstance(sampler, GroupKnockoutSampling): method = 'GroupKnockout Sampling'
        if len(other_nodes) == 0:
            taus = weighted_topk_tau(H, ranking, topk=topk)
            accuracy += [(trial, k, j*10, tau, method) for k, tau in taus]
            #taus = weighted_topk_tau(H, ranking, H.minority)
            #accuracy += [(trial, topk, j*100, tau, method, 'Minority') for topk, tau in taus]
    return accuracy


### Parameter optimization for uniform distribution ###

def _softmax_max(wj, wi): # highest softmax result for either wi or wj
    return 1/(1 + np.exp(min([wi,wj])-max([wi,wj])))

def _softmax(wj, wi): # softmax result for wj
    return 1/(1 + np.exp(wi-wj))

def _sep_stronger_prob_uniform(majority, minority, ratio=0.2):
    p_within_min = (-2)/(minority**2)*((np.pi**2)/12 + minority*np.log(2) + mpmath.fp.polylog(2,-np.exp(minority)))
    p_within_maj = (-2)/(majority**2)*((np.pi**2)/12 + majority*np.log(2) + mpmath.fp.polylog(2,-np.exp(majority)))
    p_between = 2/(minority*majority)*integrate.dblquad(_softmax_max, 0, minority, 0, majority)[0]
    return ratio**2*p_within_min + (1-ratio)**2*p_within_maj + (1-ratio)*ratio*p_between

def _sep_majority_prob_uniform(majority, minority):
    p_between = 1/(minority*majority)*integrate.dblquad(_softmax, 0, minority, 0, majority)[0]
    return p_between

def get_uniform_loss(x, prob_maj, prob_stronger, ratio=0.2):
    majority = x[0]
    minority = x[1]
    stronger_result = _sep_stronger_prob_uniform(majority, minority, ratio)
    maj_result = _sep_majority_prob_uniform(majority, minority)
    return np.linalg.norm(np.array([prob_maj, prob_stronger] - np.array([maj_result, stronger_result])))


### Parameter optimization for normal distribution ###

_min_range = -300
_max_range = 300
epsabs = 1e-4

def _softmax_normal_max(wj, wi, myu_j, sigma_j, myu_i, sigma_i): # highest softmax result for either wi or wj * prob density
    return 1/(1 + np.exp(min([wi,wj])-max([wi,wj])))*norm.pdf(x=wj, loc=myu_j, scale=sigma_j)*norm.pdf(x=wi, loc=myu_i, scale=sigma_i)

def _softmax_normal(wj, wi, myu_j, sigma_j, myu_i, sigma_i): # softmax result for wj * prob density
    return 1/(1 + np.exp(wi-wj))*norm.pdf(x=wj, loc=myu_j, scale=sigma_j)*norm.pdf(x=wi, loc=myu_i, scale=sigma_i)

def _sep_stronger_prob_normal(myu_maj, sigma_maj, myu_min, sigma_min, ratio=0.2):
    args = (myu_min, sigma_min, myu_min, sigma_min)
    p_within_min = 2*integrate.dblquad(_softmax_normal, _min_range, _max_range, lambda x:x, _max_range, args=args, epsabs=epsabs)[0]
    args = (myu_maj, sigma_maj, myu_maj, sigma_maj)
    p_within_maj = 2*integrate.dblquad(_softmax_normal, _min_range, _max_range, lambda x:x, _max_range, args=args, epsabs=epsabs)[0]
    args = (myu_maj, sigma_maj, myu_min, sigma_min)
    p_between = 2*integrate.dblquad(_softmax_normal_max, _min_range, _max_range, _min_range, _max_range, args=args, epsabs=epsabs)[0]
    return ratio**2*p_within_min + (1-ratio)**2*p_within_maj + (1-ratio)*ratio*p_between

def _sep_majority_prob_normal(myu_maj, sigma_maj, myu_min, sigma_min):
    args = (myu_maj, sigma_maj, myu_min, sigma_min)
    p_between = integrate.dblquad(_softmax_normal, _min_range, _max_range, _min_range, _max_range, args=args, epsabs=epsabs)[0]
    return p_between

def get_normal_loss(x, prob_maj, prob_stronger, ratio=0.2):
    myu_maj = x[0]
    sigma_maj = x[1]
    myu_min = x[2]
    sigma_min = x[3]
    stronger_result = _sep_stronger_prob_normal(myu_maj, sigma_maj, myu_min, sigma_min, ratio)
    maj_result = _sep_majority_prob_normal(myu_maj, sigma_maj, myu_min, sigma_min)
    return np.linalg.norm(np.array([prob_maj, prob_stronger] - np.array([maj_result, stronger_result])))


### Parallel double integration ###
#def _my_inner_quad(x, *args):
#    args = (x,) + args
#    return integrate.quad_vec(_softmax_normal, x, _max_range, args=args, workers=-1)[0]
#
#def _my_dblquad(args=None, _min_range=_min_range, _max_range=_max_range):
#    return integrate.quad_vec(_my_inner_quad, _min_range, _max_range, args=args, workers=-1)[0]


def get_sep_probs_normal(myu_maj, sigma_maj, myu_min, sigma_min, ratio=0.2):
    stronger_result = _sep_stronger_prob_normal(myu_maj, sigma_maj, myu_min, sigma_min, ratio)
    maj_result = _sep_majority_prob_normal(myu_maj, sigma_maj, myu_min, sigma_min)
    return myu_maj, sigma_maj, myu_min, sigma_min, maj_result, stronger_result


def get_exposure(trial:int, sampling_method:Sampling, apply_bias:bool):
    exps = []
    H = FairPairGraph()
    H.generate_groups(400, 200) # same size groups
    H.assign_skills(loc=0, scale=0.86142674) # general skill distribution
    if apply_bias:
        H.assign_bias(nodes=H.minority_nodes, loc=-1.43574282, scale=0.43071336) # add bias to unprivileged group
    sampler = sampling_method(H, warn=False)
    ranker = RankRecovery(H)
    ranking = None
    step = 10
    for j in range(int(1000/step)):
        if isinstance(sampler, RankSampling):
            sampler.apply(iter=step, k=1, ranking=ranking)
        elif isinstance(sampler, OversampleMinority):
            sampler.apply(iter=step, k=1, p=0.75)
        else: sampler.apply(iter=step, k=1)
        ranking, other_nodes = ranker.apply() # by default, apply rankCentrality method
        if len(other_nodes) == 0:
            exp = exposure(H, ranking, H.majority)
            exps += [(trial, j*step+step, exp, apply_bias, sampling_method.__name__, 'Privileged')]
            exp = exposure(H, ranking, H.minority)
            exps += [(trial, j*step+step, exp, apply_bias, sampling_method.__name__, 'Unprivileged')]
    return exps


def get_topk_exposure(trial:int, sampling_method:Sampling, topk=[10,50,100,250,500], N=500, Nm=100):
    exps = []
    H = FairPairGraph()
    H.generate_groups(N, Nm)
    H.group_assign_scores(nodes=H.majority_nodes, distr=Distributions.normal_distr)
    H.group_assign_scores(nodes=H.minority_nodes, distr=Distributions.normal_distr, loc=0.3, scale=0.2) # give a disadvantage to the minority
    sampler = sampling_method(H, warn=False, use_exp_BTL=True)
    ranker = RankRecovery(H)
    ranking = None
    for j in range(11):
        if isinstance(sampler, RankSampling):
            sampler.apply(iter=100, k=1, ranking=ranking)
        else: sampler.apply(iter=100, k=1)
        ranking, other_nodes = ranker.apply() # by default, apply rankCentrality method
        if isinstance(sampler, RandomSampling): method = 'Random Sampling'
        elif isinstance(sampler, OversampleMinority): method = 'Oversample Minority'
        elif isinstance(sampler, ProbKnockoutSampling): method = 'ProbKnockout Sampling'
        elif isinstance(sampler, RankSampling): method = 'Rank Sampling'
        elif isinstance(sampler, GroupKnockoutSampling): method = 'GroupKnockout Sampling'
        if len(other_nodes) == 0:
            exp = topk_exposure(H, ranking, H.majority, topk=topk)
            exps += [(trial, k, j*100, e, method, 'Majority') for k, e in exp]
            exp = topk_exposure(H, ranking, H.minority, topk=topk)
            exps += [(trial, k, j*100, e, method, 'Minority') for k, e in exp]
    return exps


### Parameter optimization with separate bias distribution ###

_min_range = -300
_max_range = 300
epsabs = 1e-4

def get_normal_loss_bias(x, prob_maj, prob_stronger):
    myu = x[0]
    sigma = x[1]
    myu_bias = x[2]
    sigma_bias = x[3]
    stronger_result = _sep_stronger_prob_normal(myu, sigma, myu + myu_bias, (sigma**2 + sigma_bias**2)**0.5, ratio=0.5)
    maj_result = _sep_majority_prob_normal(myu, sigma, myu + myu_bias, (sigma**2 + sigma_bias**2)**0.5)
    return np.linalg.norm(np.array([prob_maj, prob_stronger] - np.array([maj_result, stronger_result])))

def get_normal_loss_bias_constraint(x, prob_maj, prob_stronger, _myu, _sigma_bias):
    '''optimize with fixed values for myu_base and sigma_bias'''
    myu = _myu
    sigma = x[0]
    myu_bias = x[1]
    sigma_bias = _sigma_bias
    stronger_result = _sep_stronger_prob_normal(myu, sigma, myu + myu_bias, (sigma**2 + sigma_bias**2)**0.5, ratio=0.5)
    maj_result = _sep_majority_prob_normal(myu, sigma, myu + myu_bias, (sigma**2 + sigma_bias**2)**0.5)
    return np.linalg.norm(np.array([prob_maj, prob_stronger] - np.array([maj_result, stronger_result])))

def get_normal_loss_bias_half(x, prob_maj, prob_stronger, _myu):
    '''optimize with fixed value for myu_base and sigma_bias as sigma/2'''
    myu = _myu
    sigma = x[0]
    myu_bias = x[1]
    sigma_bias = x[0]/2
    stronger_result = _sep_stronger_prob_normal(myu, sigma, myu + myu_bias, (sigma**2 + sigma_bias**2)**0.5, ratio=0.43981708312918505) # 0.5
    maj_result = _sep_majority_prob_normal(myu, sigma, myu + myu_bias, (sigma**2 + sigma_bias**2)**0.5)
    return np.linalg.norm(np.array([prob_maj, prob_stronger] - np.array([maj_result, stronger_result])))

def get_sep_probs_normal_bias(myu, sigma, myu_bias, sigma_bias):
    stronger_result = _sep_stronger_prob_normal(myu, sigma, myu + myu_bias, (sigma**2 + sigma_bias**2)**0.5, ratio=0.43981708312918505) # 0.5
    maj_result = _sep_majority_prob_normal(myu, sigma, myu + myu_bias, (sigma**2 + sigma_bias**2)**0.5)
    return maj_result, stronger_result


def get_correlations(trial:int, samplingMethod, apply_bias:bool, rank_using=davidScore):

    # create a new graph for inference
    # fix seed=42 for reproducibility of single plots
    H = FairPairGraph()
    H.generate_groups(400, 200) # same size groups
    H.assign_skills(loc=0, scale=0.86142674) # general skill distribution
    if apply_bias:
        H.assign_bias(nodes=H.minority_nodes, loc=-1.43574282, scale=0.43071336) # add bias to unprivileged group
    
    sampler = samplingMethod(H, warn=False)
    ranker = RankRecovery(H)

    ranking = None
    step = 10
    ranks = []
    # gradually infer ranking giving the initially trained model
    for j in range(int(1000/step)):

        if samplingMethod.__name__ == 'OversampleMinority':
            sampler.apply(iter=step, k=1, p=0.75)
        elif samplingMethod.__name__ == 'RankSampling':
            sampler.apply(iter=step, k=1, ranking=ranking)
        else:
            sampler.apply(iter=step, k=1)
        
        if (nx.is_strongly_connected(H)):

            if rank_using == 'fairPageRank':
                ranker_name = 'Fairness-Aware PageRank'
                if samplingMethod.__name__ == 'RandomSampling' and apply_bias: path=f'data/tmp0'
                elif samplingMethod.__name__ == 'RandomSampling' and not apply_bias: path=f'data/tmp1'
                elif samplingMethod.__name__ == 'OversampleMinority' and apply_bias: path=f'data/tmp2'
                elif samplingMethod.__name__ == 'OversampleMinority' and not apply_bias: path=f'data/tmp3'
                elif samplingMethod.__name__ == 'RankSampling' and apply_bias: path=f'data/tmp4'
                elif samplingMethod.__name__ == 'RankSampling' and not apply_bias: path=f'data/tmp5'
                ranking, other_nodes = ranker.apply(rank_using=rank_using, path=path)
            #elif rank_using.__name__ == 'davidScore':
            #    ranker_name = "David's Score"
            #    ranking, other_nodes = ranker.apply(rank_using=rank_using)
            #elif rank_using.__name__ == 'rankCentrality':
            #    ranker_name = "RankCentrality"
            #    ranking, other_nodes = ranker.apply(rank_using=rank_using)
            ranker_name = rank_using.__name__
            ranking, other_nodes = ranker.apply(rank_using=rank_using)

            ranking_as_ranks = scores_to_rank(ranking, invert=True)
            for node, data in H.majority.nodes(data=True):
                ranks.append((trial, j*step+step, data['skill'], data['score'], ranking_as_ranks[node], 'Privileged',
                              samplingMethod.__name__, ranker_name, apply_bias))
            for node, data in H.minority.nodes(data=True):
                ranks.append((trial, j*step+step, data['skill'], data['score'], ranking_as_ranks[node], 'Unprivileged',
                              samplingMethod.__name__, ranker_name, apply_bias))
        
            if j%10 == 9:
                print(f'trial {trial}, {samplingMethod.__name__}, with{"out" if not apply_bias else ""} bias, {ranker_name}: finished {j*step+step} iterations.')
    
    return ranks