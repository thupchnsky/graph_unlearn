#!/usr/bin/env python
# coding: utf-8

# In[6]:


import argparse
from itertools import product
from tqdm import tqdm

from datasets import get_dataset
from gcn import GCN, GCNWithJK
from gin import GIN, GIN0, GIN0WithJK, GINWithJK

import time

import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from torch import tensor
from torch.optim import Adam

from torch_geometric.utils import to_dense_adj, add_self_loops, contains_self_loops
from torch_geometric.loader import DataLoader
from torch_geometric.loader import DenseDataLoader as DenseLoader

import numpy as np
from gst import *
from utils import *
from train_unlearn import *

import ipdb


# In[7]:


parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--device', type=int, default=0, help='gpu id. For cpu, set it as -1')
parser.add_argument('--seed', type=int, default=12345)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--hidden', type=int, default=64)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--folds', type=int, default=10)
parser.add_argument('--lr', type=float, default=5e-1)
parser.add_argument('--wd', type=float, default=0, help='Weight decay factor for Adam')
parser.add_argument('--GNN_lr', type=float, default=1e-4)
parser.add_argument('--GNN_wd', type=float, default=0, help='Weight decay factor for Adam of GNN')
parser.add_argument('--display_step', type=int, default=10)
parser.add_argument('--rm_disp_step', type=int, default=1)
parser.add_argument('--J', type=int, default=5)
parser.add_argument('--Q', type=int, default=4)
parser.add_argument('--L', type=int, default=3)
parser.add_argument('--lam', type=float, default=1e-4, help='L2 regularization')
parser.add_argument('--optimizer', type=str, default='LBFGS', choices=['LBFGS', 'Adam'], help='Choice of optimizer.')
parser.add_argument('--model', type=str, default='GST', choices=['GST', 'GFT', 'GIN', 'GCN', 'linear-GST'], help='Choice of model')
parser.add_argument('--dataset', type=str, default='IMDB-BINARY', choices=['IMDB-BINARY', 'PROTEINS', 'COLLAB', 'MNIST', 'CIFAR10'], help='Choice of dataset')
parser.add_argument('--verbose', action='store_true', default=False, help='verbosity in optimizer')

parser.add_argument('--num_removes', type=int, default=50)
parser.add_argument('--remove_guo', action='store_true', default=False)
parser.add_argument('--retrain', action='store_true', default=False, help='Retrain GST from scratch or not. If this is true then remove_guo should be false!')
parser.add_argument('--compare_gnorm', action='store_true', default=False, help='Compute norm of worst case and real gradient each round.')
parser.add_argument('--std', type=float, default=1e-1, help='standard deviation for objective perturbation')
parser.add_argument('--eps', type=float, default=1.0, help='Eps coefficient for certified removal.')
parser.add_argument('--delta', type=float, default=1e-4, help='Delta coefficient for certified removal.')
args = parser.parse_args()

if args.retrain:
    assert args.remove_guo == False

# Set random seed for reproducibility
set_random_seed(seed=args.seed)

def get_GST_params(args):
    if args.dataset in ['COLLAB']:
        args.J, args.Q, args.L = 3, 2, 2
        args.lam = 1e-4
        args.std = 1e-1
    elif args.dataset in ['MNIST','CIFAR10']:
        args.lam = 1e-6
        args.std = 1e-1
    elif args.dataset in ['PROTEINS']:
        args.lam = 1e-4
        args.std = 1e-1
    elif args.dataset in ['IMDB-BINARY']:
        args.J, args.Q, args.L = 4, 3, 3
        args.lam = 1e-3
        args.std = 1e-1
    return args

def decide_num_remove(dataset,trn_ratio=0.1,unlearn_ratio=0.1):
    # We will unlearn unlearn_ratio*len(train_dataset) nodes.
    return round(round(len(dataset)*trn_ratio)*unlearn_ratio)

args = get_GST_params(args)
print(args.dataset)
dataset = get_dataset(args.dataset, sparse=True)
dataset = preprocess_data_from_dataset(dataset)  # pre-process the data
args.num_removes = decide_num_remove(dataset)
if args.device>-1:
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
else:
    device = torch.device('cpu')

scattering = GeometricScattering(args.J, args.Q, args.L).to(device)
gft = GraphFourierTransform().to(device)

# Records
grad_norm_approx = torch.zeros(args.num_removes, args.folds).float() # Data dependent res grad norm
grad_norm_real = torch.zeros(args.num_removes, args.folds).float() # true res grad norm
grad_norm_worst = torch.zeros(args.num_removes, args.folds).float() # worst case res grad norm
removal_times = torch.zeros(args.num_removes, args.folds).float() # record the time of each removal
acc_removal = torch.zeros((2, args.num_removes, args.folds)).float() # record the acc after removal
num_retrain = torch.zeros((args.folds,)).int()

acc = torch.zeros((2, args.folds)).float() # record the standard acc
times = torch.zeros((args.folds,)).float() # record the standard time


for fold in range(args.folds):
    print('='*20 + '  fold=' + str(fold) + '  ' + '='*20)
    # Get random splits
    train_idx, val_idx, test_idx = get_random_splits(dataset,ratio=[0.1,0.1,0.8])
    dataset = preprocess_data_from_dataset(dataset)
    
    train_list = to_datalist(dataset[train_idx])
    test_list = to_datalist(dataset[test_idx])
    val_list = to_datalist(dataset[val_idx])
    
    # Train original model
    ###########
    if args.model == "GST":
        # GST with non-linearity
        print('='*5+'non-linear GST'+'='*5)
        w, durations, acc[0,fold], acc[1,fold] = train_GST(args, train_list, scattering, device, val_list, test_list)
        times[fold] = durations[0]+durations[1]
    ########
    elif args.model == "GFT":
        # GFT
        print('='*5+'GFT'+'='*5)
        w, durations, acc[0,fold], acc[1,fold] = train_GFT(args,train_list,gft,device,val_list,test_list)
        times[fold] = durations[0]+durations[1]
    ###########
    elif args.model == "linear-GST":
        # GST without non-linearity
        print('='*5+'linear GST'+'='*5)
        w, durations, acc[0,fold], acc[1,fold] = train_GST(args,train_list,scattering,device,val_list,test_list, nonlin=False)
        times[fold] = durations[0]+durations[1]
    elif args.model == "GIN":
        # GIN
        print('='*5+'GIN'+'='*5)
        # Get data loader, only needed for GIN
        if 'adj' in train_list[0]:
            train_loader = DenseLoader(train_list, args.batch_size, shuffle=True)
            val_loader = DenseLoader(val_list, args.batch_size, shuffle=False)
            test_loader = DenseLoader(test_list, args.batch_size, shuffle=False)
        else:
            train_loader = DataLoader(train_list, args.batch_size, shuffle=True)
            val_loader = DataLoader(val_list, args.batch_size, shuffle=False)
            test_loader = DataLoader(test_list, args.batch_size, shuffle=False)

        durations, acc[0,fold], acc[1,fold] = train_GNN(args, train_loader, device, dataset, GIN, val_loader, test_loader)
        times[fold] = durations[0]
    else:
        raise ValueError("Unexpected models!")
    
    #=================Unlearning process====================#
    print('='*5+f'Start Unlearning Process for {args.model}'+'='*5)
    # budget for removal
    c_val = get_c(args.delta)
    # if we need to compute the norms, we should not retrain at all
    if args.compare_gnorm:
        budget = 1e5
    else:
        budget = get_budget(args.std, args.eps, c_val) * dataset.num_classes
    print('Budget:', budget)
    
    if not args.model == "GIN":
        w_approx = w.clone().detach() # copy the parameters to modify
    
    # The removal_queue will record the unlearning data [(graph_id,node_id)].
    if args.model == "GST":
        if args.remove_guo:
            removal_times[:, fold], num_retrain[fold], acc_removal[:, :, fold], grad_norm_approx[:, fold], grad_norm_real[:, fold], grad_norm_worst[:, fold], removal_queue = Unlearn_GST_guo(args, scattering, train_list, device, w_approx, budget,nonlin=True, gamma=1/4,val_list=val_list, test_list=test_list)
        elif args.retrain:
            removal_times[:, fold], num_retrain[fold], acc_removal[:, :, fold], grad_norm_approx[:, fold], grad_norm_real[:, fold], grad_norm_worst[:, fold], removal_queue = Retrain_GST(args, scattering, train_list, device, w_approx,budget=0,nonlin=True, gamma=1/4,val_list=val_list, test_list=test_list)
        else:
            removal_times[:, fold], num_retrain[fold], acc_removal[:, :, fold], grad_norm_approx[:, fold], grad_norm_real[:, fold], grad_norm_worst[:, fold], removal_queue = Unlearn_GST(args, scattering, train_list, device, w_approx, budget,nonlin=True, gamma=1/4,val_list=val_list, test_list=test_list)
    
    elif args.model == "linear-GST":
        if args.remove_guo:
            removal_times[:, fold], num_retrain[fold], acc_removal[:, :, fold], grad_norm_approx[:, fold], grad_norm_real[:, fold], grad_norm_worst[:, fold], removal_queue = Unlearn_GST_guo(args, scattering, train_list, device, w_approx, budget,nonlin=False, gamma=1/4,val_list=val_list, test_list=test_list)
        else:
            removal_times[:, fold], num_retrain[fold], acc_removal[:, :, fold], grad_norm_approx[:, fold], grad_norm_real[:, fold], grad_norm_worst[:, fold], removal_queue = Unlearn_GST(args, scattering, train_list, device, w_approx, budget,nonlin=False, gamma=1/4,val_list=val_list, test_list=test_list)
    
    elif args.model == "GFT": 
        if args.remove_guo:
            removal_times[:, fold], num_retrain[fold], acc_removal[:, :, fold], grad_norm_approx[:, fold], grad_norm_real[:, fold], grad_norm_worst[:, fold], removal_queue = Unlearn_GFT_guo(args, gft, train_list, device, w_approx, budget,gamma=1/4,val_list=val_list, test_list=test_list)
        else:
            removal_times[:, fold], num_retrain[fold], acc_removal[:, :, fold], grad_norm_approx[:, fold], grad_norm_real[:, fold], grad_norm_worst[:, fold], removal_queue = Unlearn_GFT(args, gft, train_list, device, w_approx, budget,gamma=1/4,val_list=val_list, test_list=test_list)
    
    elif args.model == "GIN":
        removal_times[:, fold], acc_removal[:, :, fold] = Retrain_GNN(args, GIN, dataset, train_list, device,graph_removal_queue=None, removal_queue=None, val_list=val_list, test_list=test_list)

            
# Save all results
savepath = './results/exp2'
results = {"Accs": acc, "Times": times,
           "Removal_times": removal_times, "Removal_acc": acc_removal,
           "num_retrain": num_retrain, "grad_norm_approx": grad_norm_approx}
torch.save(results,f'{savepath}/{args.model}_{args.dataset}_{args.remove_guo}_{args.retrain}.pt')
print(f'Experiment for {args.model} on {args.dataset} with UU:{args.remove_guo}/retrain:{args.retrain} is done!')