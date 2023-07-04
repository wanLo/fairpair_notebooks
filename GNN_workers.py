import numpy as np
import torch

from fairpair import *

import sys
sys.path.append('../GNNRank/')
from src.param_parser import ArgsNamespace # just import the class, not the parser
from src.Trainer import Trainer


def get_GNNRank_accuracy(trial:int):
    accuracy = []
    args = ArgsNamespace(AllTrain=True, ERO_style='uniform', F=70, Fiedler_layer_num=5, K=20, N=350, SavePred=False, all_methods=['DIGRAC', 'ib'],
                     alpha=1.0, baseline='syncRank', cuda=True, data_path='/home/jovyan/GNNRank/src/../data/', dataset='fairPair_test/',
                     debug=False, device=torch.device(type='cuda'), dropout=0.5, early_stopping=200, epochs=1000, eta=0.1, fill_val=0.5, hidden=8, hop=2,
                     load_only=False, log_root='/home/jovyan/GNNRank/src/../logs/', lr=0.01, no_cuda=False, num_trials=1, optimizer='Adam', p=0.05,
                     pretrain_epochs=50, pretrain_with='dist', regenerate_data=True, season=1990, seed=31, seeds=[10], sigma=1.0, tau=0.5, test_ratio=1,
                     train_ratio=1, train_with='proximal_baseline', trainable_alpha=False, upset_margin=0.01, upset_margin_coeff=0, upset_ratio_coeff=1.0, weight_decay=0.0005)
    torch.manual_seed(args.seed)

    H = FairPairGraph()
    H.generate_groups(400, 200) # same size groups
    H.assign_skills(loc=0, scale=0.86142674) # general skill distribution
    #H.assign_bias(nodes=H.minority_nodes, loc=-1.43574282, scale=0.43071336) # add bias to unprivileged group
    sampler = RandomSampling(H, warn=False)
    ranking = None
    for j in range(11):
        sampler.apply(iter=100, k=1)
        if (nx.is_strongly_connected(H)):
            adj = nx.linalg.graphmatrix.adjacency_matrix(H, weight='weight') # returns a sparse matrix
            trainer = Trainer(args, random_seed=10, save_name_base='test', adj=adj) # initialize with the given adjacency matrix
            save_path_best, save_path_latest = trainer.train(model_name='ib')
            score, pred_label = trainer.predict_nn(model_name='ib', model_path=save_path_best, A=adj, GNN_variant='proximal_baseline')
            ranking = {key: 1-score[0] for key, score in enumerate(score.cpu().detach().numpy())}
            tau = weighted_tau(H, ranking)
            accuracy.append((trial, j*100, tau, 'Overall'))
            #tau = weighted_tau(H, ranking, H.majority)
            #accuracy.append((trial, j*10, tau, 'Priviledged'))
            #tau = weighted_tau(H, ranking, H.minority)
            #accuracy.append((trial, j*10, tau, 'Unpriviledged'))
            #tau = weighted_tau_separate(H, ranking, H.majority)
            #accuracy.append((trial, j*10, tau[0], 'Priviledged within-group'))
            #accuracy.append((trial, j*10, tau[1], 'Between groups'))
            #tau = weighted_tau_separate(H, ranking, H.minority, calc_between=False)
            #accuracy.append((trial, j*10, tau[0], 'Unpriviledged within-group'))
    return accuracy


def get_GNNRank_generalizability(trial:int, train_after:int):
    accuracy = []
    
    # customize `dataset`
    # get optimal settings from the paper: `baseline`, `pretrain_with`, `train_with`, `upset_margin_coeff`
    # use defaults for: `early_stopping`, `epochs`
    # handle output cleverly using: `load_only=True`, `regenerate_data=True`, `be_silent=True`
    args = ArgsNamespace(AllTrain=True, ERO_style='uniform', F=70, Fiedler_layer_num=5, K=20, N=350, SavePred=False, all_methods=['DIGRAC', 'ib'],
                     alpha=1.0, baseline='syncRank', cuda=True, data_path='/home/jovyan/GNNRank/src/../data/',
                     dataset=f'fairPair_generalize/trial{trial}_after{train_after}_', be_silent=True,
                     debug=False, device=torch.device(type='cuda'), dropout=0.5, early_stopping=200, epochs=1000, eta=0.1, fill_val=0.5, hidden=8, hop=2,
                     load_only=True, log_root='/home/jovyan/GNNRank/src/../logs/', lr=0.01, no_cuda=False, num_trials=1, optimizer='Adam', p=0.05,
                     pretrain_epochs=50, pretrain_with='dist', regenerate_data=True, season=1990, seed=31, seeds=[10], sigma=1.0, tau=0.5, test_ratio=1,
                     train_ratio=1, train_with='proximal_baseline', trainable_alpha=False, upset_margin=0.01, upset_margin_coeff=0, upset_ratio_coeff=1.0, weight_decay=0.0005)
    torch.manual_seed(args.seed)

    # create an inital graph for training
    H = FairPairGraph()
    H.generate_groups(400, 200) # same size groups
    H.assign_skills(loc=0, scale=0.86142674) # general skill distribution
    #H.assign_bias(nodes=H.minority_nodes, loc=-1.43574282, scale=0.43071336) # add bias to unprivileged group
    sampler = RandomSampling(H, warn=False)
    sampler.apply(iter=train_after, k=1) # do inital sampling

    # train once on inital graph
    adj = nx.linalg.graphmatrix.adjacency_matrix(H, weight='weight') # returns a sparse matrix
    trainer = Trainer(args, random_seed=10, save_name_base='test', adj=adj) # initialize with the given adjacency matrix
    save_path_best, save_path_latest = trainer.train(model_name='ib')

    # create a new graph for inference
    H = FairPairGraph()
    H.generate_groups(400, 200) # same size groups
    H.assign_skills(loc=0, scale=0.86142674) # general skill distribution
    #H.assign_bias(nodes=H.minority_nodes, loc=-1.43574282, scale=0.43071336) # add bias to unprivileged group
    sampler = RandomSampling(H, warn=False)

    # gradually infer ranking giving the initially trained model
    for j in range(101):
        if j%10 == 0: print(f'trial {trial}, train_after {train_after}: finished {j*10} iterations.')
        sampler.apply(iter=10, k=1)
        if (nx.is_strongly_connected(H)):
            adj = nx.linalg.graphmatrix.adjacency_matrix(H, weight='weight') # returns a sparse matrix
            trainer = Trainer(args, random_seed=10, save_name_base='test', adj=adj) # initialize with the given adjacency matrix
            # DO NOT TRAIN THE MODEL AGAIN
            #save_path_best, save_path_latest = trainer.train(model_name='ib')
            score, pred_label = trainer.predict_nn(model_name='ib', model_path=save_path_best, A=None, GNN_variant='proximal_baseline')
            ranking = {key: 1-score[0] for key, score in enumerate(score.cpu().detach().numpy())}
            tau = weighted_tau(H, ranking)
            accuracy.append((trial, j*10, tau, train_after))
    
    return accuracy


def get_GNNRank_weightedTau(trial:int, samplingMethod:RandomSampling, apply_bias:bool):
    accuracy = []
    
    # customize `dataset` to properly save the model
    # get optimal settings from the paper: `baseline`, `pretrain_with`, `train_with`, `upset_margin_coeff`
    # use defaults for: `early_stopping`, `epochs`
    # handle output cleverly using: `load_only=True`, `regenerate_data=True`, `be_silent=True`
    args = ArgsNamespace(AllTrain=True, ERO_style='uniform', F=70, Fiedler_layer_num=5, K=20, N=350, SavePred=False, all_methods=['DIGRAC', 'ib'],
                     alpha=1.0, baseline='syncRank', cuda=True, data_path='/home/jovyan/GNNRank/src/../data/',
                     dataset=f'fairPair_weightedTau/trial{trial}_{samplingMethod.__name__}_with{"out" if not apply_bias else ""}bias', be_silent=True,
                     debug=False, device=torch.device(type='cuda'), dropout=0.5, early_stopping=200, epochs=1000, eta=0.1, fill_val=0.5, hidden=8, hop=2,
                     load_only=True, log_root='/home/jovyan/GNNRank/src/../logs/', lr=0.01, no_cuda=False, num_trials=1, optimizer='Adam', p=0.05,
                     pretrain_epochs=50, pretrain_with='dist', regenerate_data=True, season=1990, seed=31, seeds=[10], sigma=1.0, tau=0.5, test_ratio=1,
                     train_ratio=1, train_with='proximal_baseline', trainable_alpha=False, upset_margin=0.01, upset_margin_coeff=0, upset_ratio_coeff=1.0, weight_decay=0.0005)
    torch.manual_seed(args.seed)

    # create an inital graph for training
    H = FairPairGraph()
    H.generate_groups(400, 200) # same size groups
    H.assign_skills(loc=0, scale=0.86142674) # general skill distribution
    if apply_bias:
        H.assign_bias(nodes=H.minority_nodes, loc=-1.43574282, scale=0.43071336) # add bias to unprivileged group
    
    # do inital sampling
    sampler = samplingMethod(H, warn=False)
    while not nx.is_strongly_connected(H): # make sure it's a least strongly connected before we train GNNRank
        if samplingMethod.__name__ == 'OversampleMinority':
            sampler.apply(iter=50, k=1, p=0.75)
        elif samplingMethod.__name__ == 'RankSampling':
            ranker = RankRecovery(H)
            ranking, other_nodes = ranker.apply() # use default rankCentrality for pre-sampling
            sampler.apply(iter=50, k=1, ranking=ranking)
        else:
            sampler.apply(iter=50, k=1)
            
    # train once on inital graph
    adj = nx.linalg.graphmatrix.adjacency_matrix(H, weight='weight') # returns a sparse matrix
    trainer = Trainer(args, random_seed=10, save_name_base='test', adj=adj) # initialize with the given adjacency matrix
    save_path_best, save_path_latest = trainer.train(model_name='ib')

    # create a new graph for inference
    H = FairPairGraph()
    H.generate_groups(400, 200) # same size groups
    H.assign_skills(loc=0, scale=0.86142674) # general skill distribution
    if apply_bias:
        H.assign_bias(nodes=H.minority_nodes, loc=-1.43574282, scale=0.43071336) # add bias to unprivileged group
    
    sampler = samplingMethod(H, warn=False)
    ranking = None
    step = 20
    # gradually infer ranking giving the initially trained model
    for j in range(int(3000/step)):

        if samplingMethod.__name__ == 'OversampleMinority':
            sampler.apply(iter=step, k=1, p=0.75)
        elif samplingMethod.__name__ == 'RankSampling':
            sampler.apply(iter=step, k=1, ranking=ranking)
        else:
            sampler.apply(iter=step, k=1)
        
        if (nx.is_strongly_connected(H)):
            adj = nx.linalg.graphmatrix.adjacency_matrix(H, weight='weight') # returns a sparse matrix
            trainer = Trainer(args, random_seed=10, save_name_base='test', adj=adj) # initialize with the given adjacency matrix
            score, pred_label = trainer.predict_nn(model_name='ib', model_path=save_path_best, A=None, GNN_variant='proximal_baseline')
            ranking = {key: 1-score[0] for key, score in enumerate(score.cpu().detach().numpy())}

            tau = weighted_tau(H, ranking)
            accuracy.append((trial, j*step+step, tau, apply_bias, samplingMethod.__name__, 'tau', 'Overall'))
            tau = weighted_tau(H, ranking, H.majority)
            accuracy.append((trial, j*step+step, tau, apply_bias, samplingMethod.__name__, 'tau', 'Privileged'))
            tau = weighted_tau(H, ranking, H.minority)
            accuracy.append((trial, j*step+step, tau, apply_bias, samplingMethod.__name__, 'tau', 'Unprivileged'))
            tau = weighted_tau_separate(H, ranking, H.majority)
            accuracy.append((trial, j*step+step, tau[0], apply_bias, samplingMethod.__name__, 'tau', 'Privileged within-group'))
            accuracy.append((trial, j*step+step, tau[1], apply_bias, samplingMethod.__name__, 'tau', 'Between groups'))
            tau = weighted_tau_separate(H, ranking, H.minority, calc_between=False)
            accuracy.append((trial, j*step+step, tau[0], apply_bias, samplingMethod.__name__, 'tau', 'Unprivileged within-group'))
            exp = exposure(H, ranking, H.majority)
            accuracy.append((trial, j*step+step, exp, apply_bias, samplingMethod.__name__, 'exposure', 'Privileged'))
            exp = exposure(H, ranking, H.minority)
            accuracy.append((trial, j*step+step, exp, apply_bias, samplingMethod.__name__, 'exposure', 'Unprivileged'))
        
        if j%10 == 9:
            print(f'{samplingMethod.__name__}, with{"out" if not apply_bias else ""}bias, trial {trial}: finished {j*step+step} iterations.')
    
    accuracy_df = pd.DataFrame(accuracy, columns=['trial', 'iteration', 'value', 'bias_applied', 'sampling method', 'metric', 'group'])
    accuracy_df.to_csv(f'./data/GNNRank_intermed/trial{trial}_{samplingMethod.__name__}_with{"" if apply_bias else "out"}Bias.csv', index=False)

    return accuracy




def get_GNNRank_correlations(samplingMethod:RandomSampling, apply_bias:bool):
    # customize `dataset` to properly save the model
    # get optimal settings from the paper: `baseline`, `pretrain_with`, `train_with`, `upset_margin_coeff`
    # use defaults for: `early_stopping`, `epochs`
    # handle output cleverly using: `load_only=True`, `regenerate_data=True`, `be_silent=True`
    args = ArgsNamespace(AllTrain=True, ERO_style='uniform', F=70, Fiedler_layer_num=5, K=20, N=350, SavePred=False, all_methods=['DIGRAC', 'ib'],
                     alpha=1.0, baseline='syncRank', cuda=True, data_path='/home/jovyan/GNNRank/src/../data/',
                     dataset=f'fairPair_correlations/{samplingMethod.__name__}_with{"out" if not apply_bias else ""}bias', be_silent=True,
                     debug=False, device=torch.device(type='cuda'), dropout=0.5, early_stopping=200, epochs=1000, eta=0.1, fill_val=0.5, hidden=8, hop=2,
                     load_only=True, log_root='/home/jovyan/GNNRank/src/../logs/', lr=0.01, no_cuda=False, num_trials=1, optimizer='Adam', p=0.05,
                     pretrain_epochs=50, pretrain_with='dist', regenerate_data=True, season=1990, seed=31, seeds=[10], sigma=1.0, tau=0.5, test_ratio=1,
                     train_ratio=1, train_with='proximal_baseline', trainable_alpha=False, upset_margin=0.01, upset_margin_coeff=0, upset_ratio_coeff=1.0, weight_decay=0.0005)
    torch.manual_seed(args.seed)

    # create a new graph for inference
    # fix seed=42 for reproducibility of single plots
    H = FairPairGraph()
    H.generate_groups(400, 200) # same size groups
    H.assign_skills(loc=0, scale=0.86142674, seed=42) # general skill distribution
    if apply_bias:
        H.assign_bias(nodes=H.minority_nodes, loc=-1.43574282, scale=0.43071336, seed=42) # add bias to unprivileged group
    
    sampler = samplingMethod(H, warn=False)
    ranking = None
    step = 10
    connected = False
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

            if not connected:
                # train once
                adj = nx.linalg.graphmatrix.adjacency_matrix(H, weight='weight') # returns a sparse matrix
                trainer = Trainer(args, random_seed=10, save_name_base='test', adj=adj) # initialize with the given adjacency matrix
                save_path_best, save_path_latest = trainer.train(model_name='ib')
                connected = True
            
            adj = nx.linalg.graphmatrix.adjacency_matrix(H, weight='weight') # returns a sparse matrix
            trainer = Trainer(args, random_seed=10, save_name_base='test', adj=adj) # initialize with the given adjacency matrix
            score, pred_label = trainer.predict_nn(model_name='ib', model_path=save_path_best, A=None, GNN_variant='proximal_baseline')
            ranking = {key: 1-score[0] for key, score in enumerate(score.cpu().detach().numpy())}

            ranking_as_ranks = scores_to_rank(ranking, invert=True)
            for node, data in H.majority.nodes(data=True):
                ranks.append((j*step+step, data['skill'], data['score'], ranking_as_ranks[node], 'Privileged', samplingMethod.__name__, apply_bias))
            for node, data in H.minority.nodes(data=True):
                ranks.append((j*step+step, data['skill'], data['score'], ranking_as_ranks[node], 'Unprivileged', samplingMethod.__name__, apply_bias))
        
        if j%10 == 9:
            print(f'{samplingMethod.__name__}, with{"out" if not apply_bias else ""}bias: finished {j*step+step} iterations.')
    
    return ranks