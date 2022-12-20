from tqdm import tqdm

from gcn import GCN, GCNWithJK
from gin import GIN, GIN0, GIN0WithJK, GINWithJK

import time

import torch
import torch.nn.functional as F
from torch import tensor
from torch.optim import Adam

from torch_geometric.utils import to_dense_adj, add_self_loops, contains_self_loops
from torch_geometric.loader import DataLoader
from torch_geometric.loader import DenseDataLoader as DenseLoader

import numpy as np
from gst import *
from utils import *

# import ipdb

def train(model, optimizer, loader, device):
    """
    For GIN.
    """
    model.train()
    total_loss = 0
    for data in loader:
        optimizer.zero_grad()
        data = data.to(device)
        out = model(data)
        loss = F.nll_loss(out, data.y.view(-1))
        loss.backward()
        total_loss += loss.item() * num_graphs(data)
        optimizer.step()
    return total_loss / len(loader.dataset)


def eval_acc(model, loader,device):
    """
    For GIN.
    """
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data).max(1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()
    return correct / len(loader.dataset)


def eval_loss(model, loader,device):
    """
    For GIN.
    """
    model.eval()
    loss = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            out = model(data)
        loss += F.nll_loss(out, data.y.view(-1), reduction='sum').item()
    return loss / len(loader.dataset)


def num_graphs(data):
    if hasattr(data, 'num_graphs'):
        return data.num_graphs
    else:
        return data.x.size(0)

    
def get_GST_emb(datalist, scattering, device, train_split=False, nonlin=True, batch=-1):
    """
    Input:
        datalist: a list of Data objects.
        scattering: Scattering Transform object.
        batch: if batch > 0, meaning we want to compute the embedding for all graphs, we should use loader; otherwise we should just Data
    """
    X = []
    y = []
    if batch > 0:        
        if 'adj' in datalist[0]:
            tmp_loader = DenseLoader(datalist, batch, shuffle=False)
        else:
            tmp_loader = DataLoader(datalist, batch, shuffle=False)
        for data in tqdm(tmp_loader):
            data = data.to(device)
            embed = scattering(data, nonlin, use_batch=True)
            X.append(embed)
            y.append(data.y.view(-1))
    else:
        for data in datalist:
            data = data.to(device)
            embed = scattering(data, nonlin, use_batch=False)
            X.append(embed)
            y.append(data.y.view(-1))
    X = torch.cat(X, dim=0)
    y = torch.cat(y)
    if train_split:
        y = F.one_hot(y) * 2 - 1
        y = y.float()
    return X.to(device), y.to(device)


def get_GFT_emb(datalist, gft, device, train_split=False, batch=-1):
    """
    Input:
        datalist: a list of Data objects.
        gft: GFT object.
    """
    X = []
    y = []
    if batch > 0:        
        if 'adj' in datalist[0]:
            tmp_loader = DenseLoader(datalist, batch, shuffle=False)
        else:
            tmp_loader = DataLoader(datalist, batch, shuffle=False)
        for data in tqdm(tmp_loader):
            data = data.to(device)
            embed = gft(data, use_batch=True)
            X.append(embed)
            y.append(data.y.view(-1))
    else:
        for data in datalist:
            data = data.to(device)
            embed = gft(data, use_batch=False)
            X.append(embed)
            y.append(data.y.view(-1))
    X = torch.cat(X, dim=0)
    y = torch.cat(y)
    if train_split:
        y = F.one_hot(y) * 2 - 1
        y = y.float()
    return X.to(device), y.to(device)


def train_GST(args, train_list, scattering, device, val_list=None, test_list=None, nonlin=True):
    """
    train_list, val_list, test_list are not dataloaders and should not be confused with the one used in GIN.
    """
    durations = []
    t_start = time.perf_counter()
    # generate GST embeddings, use batch for computing
    X_train, y_train = get_GST_emb(train_list, scattering, device, train_split=True, nonlin=nonlin, batch=args.batch_size)
    if val_list and test_list:
        X_val, y_val = get_GST_emb(val_list, scattering, device, nonlin=nonlin, batch=args.batch_size)
        X_test, y_test = get_GST_emb(test_list, scattering, device, nonlin=nonlin, batch=args.batch_size)
    
    t_end = time.perf_counter()
    duration = t_end - t_start
    durations.append(duration)
    print(f'Train Size: {X_train.size(0)}, Val Size: {X_val.size(0)}, Test Size: {X_test.size(0)}, Feature Size: {X_test.size(1)}, Class Size: {int(y_test.max())+1}')
    print(f'GST Data Processing Time: {duration:.3f} s')
    
    # train classifier
    t_start = time.perf_counter()
    b_std = args.std
    b = b_std * torch.randn(X_train.size(1), y_train.size(1)).float().to(device)
    w = ovr_lr_optimize(X_train, y_train, args.lam, None, b=b, num_steps=args.epochs, verbose=False, opt_choice=args.optimizer, lr=args.lr, wd=args.wd)
    t_end = time.perf_counter()
    duration = t_end - t_start
    durations.append(duration)
    
    val_acc, test_acc = 0, 0
    if val_list and test_list:
        val_acc = ovr_lr_eval(w, X_val, y_val)
        test_acc = ovr_lr_eval(w, X_test, y_test)
        print('Val accuracy = %.4f' % val_acc)
        print('Test accuracy = %.4f' % test_acc)
    print(f'GST Training Time: {duration:.3f} s')
    
    return w, durations, val_acc, test_acc


def train_GFT(args, train_list, gft, device, val_list=None, test_list=None):
    """
    train_list, val_list, test_list are not dataloaders and should not be confused with the one used in GIN.
    """
    durations = []
    t_start = time.perf_counter()
    # generate GFT embeddings
    X_train, y_train = get_GFT_emb(train_list, gft, device, train_split=True, batch=args.batch_size)
    if val_list and test_list:
        X_val, y_val = get_GFT_emb(val_list, gft, device, batch=args.batch_size)
        X_test, y_test = get_GFT_emb(test_list, gft, device, batch=args.batch_size)
    
    t_end = time.perf_counter()
    duration = t_end - t_start
    durations.append(duration)
    # print(f'Train Size: {X_train.size(0)}, Val Size: {X_val.size(0)}, Test Size: {X_test.size(0)}, Class Size: {y_train.max()+1}')
    print(f'GFT Data Processing Time: {duration:.3f} s')
    
    # train classifier
    t_start = time.perf_counter()
    b_std = args.std
    b = b_std * torch.randn(X_train.size(1), y_train.size(1)).float().to(device)
    w = ovr_lr_optimize(X_train, y_train, args.lam, None, b=b, num_steps=args.epochs, verbose=False, opt_choice=args.optimizer, lr=args.lr, wd=args.wd)
    t_end = time.perf_counter()
    duration = t_end - t_start
    durations.append(duration)
    
    val_acc, test_acc = 0, 0
    if val_list and test_list:
        val_acc = ovr_lr_eval(w, X_val, y_val)
        test_acc = ovr_lr_eval(w, X_test, y_test)
        print('Val accuracy = %.4f' % val_acc)
        print('Test accuracy = %.4f' % test_acc)
    print(f'GFT Training Time: {duration:.3f} s')
    
    return w, durations, val_acc, test_acc


def train_GNN(args, train_loader, device, dataset, Net=GIN, val_loader=None, test_loader=None):
    model = Net(dataset, args.num_layers, args.hidden)
    model.to(device).reset_parameters()
    optimizer = Adam(model.parameters(), lr=args.GNN_lr, weight_decay=args.GNN_wd)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    t_start = time.perf_counter()
    
    trn_loss, val_loss, val_acc, best_tst_acc = 0, 0, 0, 0
    best_val_acc, best_val_loss = 0, np.inf
    for epoch in range(1, args.epochs + 1):
        trn_loss = train(model, optimizer, train_loader,device)
        val_loss = eval_loss(model, val_loader,device)
        val_acc = eval_acc(model, val_loader,device)
        tst_acc = eval_acc(model, test_loader,device)
        
        # save best model base on val_loss!
        if val_loss <= best_val_loss:
            best_val_acc = val_acc
            best_tst_acc = eval_acc(model, test_loader,device)
            best_val_loss = val_loss
        
        if args.verbose:
            if epoch % args.display_step == 0:
                print(f'Epoch: {epoch}, Val Acc: {best_val_acc:.4f}, Test Acc: {best_tst_acc:.4f}')

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t_end = time.perf_counter()
    duration = t_end - t_start
    print(f'GIN Training Time: {duration:.3f} s')
    print(f'Epoch: {epoch}, Val Acc: {best_val_acc:.4f}, Test Acc: {best_tst_acc:.4f}')
    
    return [duration], best_val_acc, best_tst_acc


def Unlearn_GST(args, scattering, train_list, device, w_approx, budget, graph_removal_queue=None,
                removal_queue=None, val_list=None, test_list=None, nonlin=True, gamma=1/4):
    """
    train_list, val_list, test_list are not datasets, they are list of Data objects.
    The removal logic is Unlearn_GST will generate the removal_queue, and Retrain_GNN will use the exact same removal_queue
    """
    
    # F for Scattering Transform
    f = np.sqrt(args.L)
    
    grad_norm_approx = torch.zeros(args.num_removes).float() # Data dependent grad res norm
    grad_norm_real = torch.zeros(args.num_removes).float() # true grad res norm
    grad_norm_worst = torch.zeros(args.num_removes).float() # worst case grad res norm
    
    removal_time = torch.zeros(args.num_removes).float() # record the time of each removal
    acc_removal = torch.zeros((2, args.num_removes)).float() # record the acc after removal, 0 for val and 1 for test
    num_retrain = 0
    b_std = args.std
    
    if removal_queue is None:
        # Remove one node for a graph, generate removal graph_id in advance.
        graph_removal_queue = torch.randperm(len(train_list))
        removal_queue = []
    else:
        # will use the existing removal_queue
        graph_removal_queue = None
        
    # beta in paper
    grad_norm_approx_sum = 0
    
    X_train_old, y_train = get_GST_emb(train_list, scattering, device, train_split=True, nonlin=nonlin, batch=args.batch_size)  # y_train will not change during unlearning process for now
    if val_list and test_list:
        X_val, y_val = get_GST_emb(val_list, scattering, device, train_split=False, nonlin=nonlin, batch=args.batch_size)
        X_test, y_test = get_GST_emb(test_list, scattering, device, train_split=False, nonlin=nonlin, batch=args.batch_size)
    
    # start the removal process
    for i in range(args.num_removes):
        # we have generated the order of graphs to remove
        if graph_removal_queue is not None:
            # remove one node from graph_removal_queue[i % len(train_list)]
            train_list, removal_queue = remove_node_from_graph(train_list, graph_id=graph_removal_queue[i%len(train_list)], removal_queue=removal_queue)
        else:
            # remove one node based on removal_queue
            train_list = remove_node_from_graph(train_list, graph_id=removal_queue[i][0], node_id=removal_queue[i][1])
        
        X_train_new = X_train_old.clone().detach()
        
        t_start = time.perf_counter()
        # generate train embeddings AFTER removal, only for the affect graph
        new_graph_emb, _ = get_GST_emb([train_list[removal_queue[i][0]]], scattering, device, train_split=True, nonlin=nonlin, batch=-1)
        
        X_train_new[removal_queue[i][0], :] = new_graph_emb.view(-1)
        K = get_K_matrix(X_train_new).to(device)
        spec_norm = sqrt_spectral_norm(K)
        # update classifier for each class separately
        for k in range(y_train.size(1)):
            H_inv = lr_hessian_inv(w_approx[:, k], X_train_new, y_train[:, k], args.lam)
            # grad_i is the difference
            grad_old = lr_grad(w_approx[:, k], X_train_old, y_train[:,k], args.lam)
            grad_new = lr_grad(w_approx[:, k], X_train_new, y_train[:,k], args.lam)
            grad_i = grad_old - grad_new
            Delta = H_inv.mv(grad_i)
            Delta_p = X_train_new.mv(Delta)
            # update w here. If beta exceed the budget, w_approx will be retrained
            w_approx[:, k] += Delta
            
            # data dependent norm
            grad_norm_approx[i] += (Delta.norm() * Delta_p.norm() * spec_norm * gamma * f).cpu()
        
        # decide after all classes
        if grad_norm_approx_sum + grad_norm_approx[i] > budget:
            # retrain the model
            grad_norm_approx_sum = 0
            b = b_std * torch.randn(X_train_new.size(1), y_train.size(1)).float().to(device)
            w_approx = ovr_lr_optimize(X_train_new, y_train, args.lam, None, b=b, num_steps=args.epochs, verbose=False,
                                       opt_choice=args.optimizer, lr=args.lr, wd=args.wd)
            num_retrain += 1
        else:
            grad_norm_approx_sum += grad_norm_approx[i]
        
        # record acc each round
        if val_list and test_list:
            acc_removal[0, i] = ovr_lr_eval(w_approx, X_val, y_val)
            acc_removal[1, i] = ovr_lr_eval(w_approx, X_test, y_test)
        
        removal_time[i] = time.perf_counter() - t_start
        # Remember to replace X_old with X_new
        X_train_old = X_train_new.clone().detach()
        
        if (i+1) % args.rm_disp_step == 0:
            print('Remove iteration %d: time = %.2fs, number of retrain = %d' % (i+1, removal_time[i], num_retrain))
            print('Val acc = %.4f, Test acc = %.4f' % (acc_removal[0, i], acc_removal[1, i]))
    
    return removal_time, num_retrain, acc_removal, grad_norm_approx, grad_norm_real, grad_norm_worst, removal_queue

def Unlearn_GST_guo(args, scattering, train_list, device, w_approx, budget, graph_removal_queue=None,
                    removal_queue=None, val_list=None, test_list=None, nonlin=True, gamma=1/4):
    """
    train_list, val_list, test_list are not datasets, they are list of Data objects.
    The removal logic is Unlearn_GST will generate the removal_queue, and Retrain_GNN will use the exact same removal_queue
    """
    
    # F for Scattering Transform
    f = np.sqrt(args.L)
    
    grad_norm_approx = torch.zeros(args.num_removes).float() # Data dependent grad res norm
    grad_norm_real = torch.zeros(args.num_removes).float() # true grad res norm
    grad_norm_worst = torch.zeros(args.num_removes).float() # worst case grad res norm
    
    removal_time = torch.zeros(args.num_removes).float() # record the time of each removal
    acc_removal = torch.zeros((2, args.num_removes)).float() # record the acc after removal, 0 for val and 1 for test
    num_retrain = 0
    b_std = args.std
    
    grad_norm_approx_sum = 0
    
    X_train_old, y_train_old = get_GST_emb(train_list, scattering, device, train_split=True, nonlin=nonlin, batch=args.batch_size)  # y_train will not change during unlearning process for now
    if val_list and test_list:
        X_val, y_val = get_GST_emb(val_list, scattering, device, train_split=False, nonlin=nonlin, batch=args.batch_size)
        X_test, y_test = get_GST_emb(test_list, scattering, device, train_split=False, nonlin=nonlin, batch=args.batch_size)
    
    # start the removal process
    for i in range(args.num_removes):
        # Randomly choose which graph to remove at each round
        remove_idx = np.random.randint(len(train_list))
        X_train_new = X_train_old.clone().detach()[torch.arange(len(train_list)) != remove_idx,:]
        y_train_new = y_train_old.clone().detach()[torch.arange(len(train_list)) != remove_idx,:]
        train_list.pop(remove_idx)
        
        
        t_start = time.perf_counter()
        K = get_K_matrix(X_train_new).to(device)
        spec_norm = sqrt_spectral_norm(K)
        
        # update classifier for each class separately
        for k in range(y_train_new.size(1)):
            H_inv = lr_hessian_inv(w_approx[:, k], X_train_new, y_train_new[:, k], args.lam)
            
            # grad_i is the difference
            grad_old = lr_grad(w_approx[:, k], X_train_old, y_train_old[:,k], args.lam)
            grad_new = lr_grad(w_approx[:, k], X_train_new, y_train_new[:,k], args.lam)
            grad_i = grad_old - grad_new
            Delta = H_inv.mv(grad_i)
            Delta_p = X_train_new.mv(Delta)
            # update w here. If beta exceed the budget, w_approx will be retrained
            w_approx[:, k] += Delta
            
            # data dependent norm
            grad_norm_approx[i] += (Delta.norm() * Delta_p.norm() * spec_norm * gamma * f).cpu()
            
        # decide after all classes
        if grad_norm_approx_sum + grad_norm_approx[i] > budget:
            # retrain the model
            grad_norm_approx_sum = 0
            b = b_std * torch.randn(X_train_new.size(1), y_train_new.size(1)).float().to(device)
            w_approx = ovr_lr_optimize(X_train_new, y_train_new, args.lam, None, b=b, num_steps=args.epochs, verbose=False,
                                       opt_choice=args.optimizer, lr=args.lr, wd=args.wd)
            num_retrain += 1
        else:
            grad_norm_approx_sum += grad_norm_approx[i]
        
        
        removal_time[i] = time.perf_counter() - t_start
        # record acc each round
        if val_list and test_list:
            acc_removal[0, i] = ovr_lr_eval(w_approx, X_val, y_val)
            acc_removal[1, i] = ovr_lr_eval(w_approx, X_test, y_test)
        
        
        # Remember to replace X_old with X_new
        X_train_old = X_train_new.clone().detach()
        y_train_old = y_train_new.clone().detach()
        
        if (i+1) % args.rm_disp_step == 0:
            print('Remove iteration %d: time = %.2fs, number of retrain = %d' % (i+1, removal_time[i], num_retrain))
            print('Val acc = %.4f, Test acc = %.4f' % (acc_removal[0, i], acc_removal[1, i]))
    
    return removal_time, num_retrain, acc_removal, grad_norm_approx, grad_norm_real, grad_norm_worst, removal_queue

def Unlearn_GFT(args, gft, train_list, device, w_approx, budget, graph_removal_queue=None,
                removal_queue=None, val_list=None, test_list=None, gamma=1/4):
    """
    train_list, val_list, test_list are not datasets, they are list of Data objects.
    The removal logic is Unlearn_GST will generate the removal_queue, and Retrain_GNN will use the exact same removal_queue
    """
    
    grad_norm_approx = torch.zeros(args.num_removes).float() # Data dependent grad res norm
    grad_norm_real = torch.zeros(args.num_removes).float() # true grad res norm
    grad_norm_worst = torch.zeros(args.num_removes).float() # worst case grad res norm
    
    removal_time = torch.zeros(args.num_removes).float() # record the time of each removal
    acc_removal = torch.zeros((2, args.num_removes)).float() # record the acc after removal, 0 for val and 1 for test
    num_retrain = 0
    b_std = args.std
    
    if removal_queue is None:
        # Remove one node for a graph, generate removal graph_id in advance.
        graph_removal_queue = torch.randperm(len(train_list))
        removal_queue = []
    else:
        # will use the existing removal_queue
        graph_removal_queue = None
        
    # beta in paper
    grad_norm_approx_sum = 0
    
    X_train_old, y_train = get_GFT_emb(train_list, gft, device, train_split=True, batch=args.batch_size)  # y_train will not change during unlearning process for now
    if val_list and test_list:
        X_val, y_val = get_GFT_emb(val_list, gft, device, train_split=False, batch=args.batch_size)
        X_test, y_test = get_GFT_emb(test_list, gft, device, train_split=False, batch=args.batch_size)
    
    # start the removal process
    for i in range(args.num_removes):
        # we have generated the order of graphs to remove
        if graph_removal_queue is not None:
            # remove one node from graph_removal_queue[i % len(train_list)]
            train_list, removal_queue = remove_node_from_graph(train_list, graph_id=graph_removal_queue[i%len(train_list)], removal_queue=removal_queue)
        else:
            # remove one node based on removal_queue
            train_list = remove_node_from_graph(train_list, graph_id=removal_queue[i][0], node_id=removal_queue[i][1])
        
        X_train_new = X_train_old.clone().detach()
        
        t_start = time.perf_counter()
        # generate train embeddings AFTER removal, only for the affect graph
        new_graph_emb, _ = get_GFT_emb([train_list[removal_queue[i][0]]], gft, device, train_split=True, batch=-1)
        
        X_train_new[removal_queue[i][0], :] = new_graph_emb.view(-1)
        K = get_K_matrix(X_train_new).to(device)
        spec_norm = sqrt_spectral_norm(K)
        
        # update classifier for each class separately
        for k in range(y_train.size(1)):
            H_inv = lr_hessian_inv(w_approx[:, k], X_train_new, y_train[:, k], args.lam)
            
            # grad_i is the difference
            grad_old = lr_grad(w_approx[:, k], X_train_old, y_train[:,k], args.lam)
            grad_new = lr_grad(w_approx[:, k], X_train_new, y_train[:,k], args.lam)
            grad_i = grad_old - grad_new
            Delta = H_inv.mv(grad_i)
            Delta_p = X_train_new.mv(Delta)

            # update w here. If beta exceed the budget, w_approx will be retrained
            w_approx[:, k] += Delta
            
            # data dependent norm
            grad_norm_approx[i] += (Delta.norm() * Delta_p.norm() * spec_norm * gamma).cpu()

        # decide after all classes
        if grad_norm_approx_sum + grad_norm_approx[i] > budget:
            # retrain the model
            grad_norm_approx_sum = 0
            b = b_std * torch.randn(X_train_new.size(1), y_train.size(1)).float().to(device)
            w_approx = ovr_lr_optimize(X_train_new, y_train, args.lam, None, b=b, num_steps=args.epochs, verbose=False,
                                       opt_choice=args.optimizer, lr=args.lr, wd=args.wd)
            num_retrain += 1
        else:
            grad_norm_approx_sum += grad_norm_approx[i]
        
        
        removal_time[i] = time.perf_counter() - t_start
        # record acc each round
        if val_list and test_list:
            acc_removal[0, i] = ovr_lr_eval(w_approx, X_val, y_val)
            acc_removal[1, i] = ovr_lr_eval(w_approx, X_test, y_test)
        
        
        # Remember to replace X_old with X_new
        X_train_old = X_train_new.clone().detach()
        
        if (i+1) % args.rm_disp_step == 0:
            print('Remove iteration %d: time = %.2fs, number of retrain = %d' % (i+1, removal_time[i], num_retrain))
            print('Val acc = %.4f, Test acc = %.4f' % (acc_removal[0, i], acc_removal[1, i]))
    
    return removal_time, num_retrain, acc_removal, grad_norm_approx, grad_norm_real, grad_norm_worst, removal_queue

def Unlearn_GFT_guo(args, gft, train_list, device, w_approx, budget, graph_removal_queue=None,
                removal_queue=None, val_list=None, test_list=None, gamma=1/4):
    """
    train_list, val_list, test_list are not datasets, they are list of Data objects.
    The removal logic is Unlearn_GST will generate the removal_queue, and Retrain_GNN will use the exact same removal_queue
    """
    
    grad_norm_approx = torch.zeros(args.num_removes).float() # Data dependent grad res norm
    grad_norm_real = torch.zeros(args.num_removes).float() # true grad res norm
    grad_norm_worst = torch.zeros(args.num_removes).float() # worst case grad res norm
    
    removal_time = torch.zeros(args.num_removes).float() # record the time of each removal
    acc_removal = torch.zeros((2, args.num_removes)).float() # record the acc after removal, 0 for val and 1 for test
    num_retrain = 0
    b_std = args.std
        
    grad_norm_approx_sum = 0
    
    X_train_old, y_train_old = get_GFT_emb(train_list, gft, device, train_split=True, batch=args.batch_size)  # y_train will not change during unlearning process for now
    if val_list and test_list:
        X_val, y_val = get_GFT_emb(val_list, gft, device, train_split=False, batch=args.batch_size)
        X_test, y_test = get_GFT_emb(test_list, gft, device, train_split=False, batch=args.batch_size)
    
    # start the removal process
    for i in range(args.num_removes):
        # Randomly remove one graph each round
        remove_idx = np.random.randint(len(train_list))
        X_train_new = X_train_old.clone().detach()[torch.arange(len(train_list)) != remove_idx,:]
        y_train_new = y_train_old.clone().detach()[torch.arange(len(train_list)) != remove_idx,:]
        train_list.pop(remove_idx)
        
        t_start = time.perf_counter()
        
        
        
        K = get_K_matrix(X_train_new).to(device)
        spec_norm = sqrt_spectral_norm(K)
        
        
        # update classifier for each class separately
        for k in range(y_train_new.size(1)):
            H_inv = lr_hessian_inv(w_approx[:, k], X_train_new, y_train_new[:, k], args.lam)
            
            # grad_i is the difference
            grad_old = lr_grad(w_approx[:, k], X_train_old, y_train_old[:,k], args.lam)
            grad_new = lr_grad(w_approx[:, k], X_train_new, y_train_new[:,k], args.lam)
            grad_i = grad_old - grad_new
            Delta = H_inv.mv(grad_i) 
            Delta_p = X_train_new.mv(Delta)

            # update w here. If beta exceed the budget, w_approx will be retrained
            w_approx[:, k] += Delta
            
            # data dependent norm
            grad_norm_approx[i] += (Delta.norm() * Delta_p.norm() * spec_norm * gamma).cpu()
        
        # decide after all classes
        if grad_norm_approx_sum + grad_norm_approx[i] > budget:
            # retrain the model
            grad_norm_approx_sum = 0
            b = b_std * torch.randn(X_train_new.size(1), y_train_new.size(1)).float().to(device)
            w_approx = ovr_lr_optimize(X_train_new, y_train_new, args.lam, None, b=b, num_steps=args.epochs, verbose=False,
                                       opt_choice=args.optimizer, lr=args.lr, wd=args.wd)
            num_retrain += 1
        else:
            grad_norm_approx_sum += grad_norm_approx[i]
        
        
        removal_time[i] = time.perf_counter() - t_start
        # record acc each round
        if val_list and test_list:
            acc_removal[0, i] = ovr_lr_eval(w_approx, X_val, y_val)
            acc_removal[1, i] = ovr_lr_eval(w_approx, X_test, y_test)
        
        
        # Remember to replace X_old with X_new
        X_train_old = X_train_new.clone().detach()
        y_train_old = y_train_new.clone().detach()
        
        if (i+1) % args.rm_disp_step == 0:
            print('Remove iteration %d: time = %.2fs, number of retrain = %d' % (i+1, removal_time[i], num_retrain))
            print('Val acc = %.4f, Test acc = %.4f' % (acc_removal[0, i], acc_removal[1, i]))
    
    return removal_time, num_retrain, acc_removal, grad_norm_approx, grad_norm_real, grad_norm_worst, removal_queue

def Retrain_GNN(args, Net, dataset, train_list, device, graph_removal_queue=None, removal_queue=None, val_list=None, test_list=None):
    """
    Retrain GNN after each removal
    """
    removal_time = torch.zeros(args.num_removes).float()  # record the time of each removal
    acc_removal = torch.zeros((2, args.num_removes)).float()  # record the acc after removal, 0 for val and 1 for test
    if removal_queue is None:
        # Remove one node for a graph, generate removal graph_id in advance.
        graph_removal_queue = torch.randperm(len(train_list))
        removal_queue = []
    else:
        graph_removal_queue = None
        
    # Construct dataloader
    if 'adj' in train_list[0]:
        if val_list and test_list:
            val_loader = DenseLoader(val_list, args.batch_size, shuffle=False)
            test_loader = DenseLoader(test_list, args.batch_size, shuffle=False)
    else:
        if val_list and test_list:
            val_loader = DataLoader(val_list, args.batch_size, shuffle=False)
            test_loader = DataLoader(test_list, args.batch_size, shuffle=False)
    
    for i in range(args.num_removes):
        if graph_removal_queue is not None:
            # remove one node
            train_list, removal_queue = remove_node_from_graph(train_list, graph_id=graph_removal_queue[i%len(train_list)], removal_queue=removal_queue)
        else:
            # remove one node
            train_list = remove_node_from_graph(train_list, graph_id=removal_queue[i][0], node_id=removal_queue[i][1])
        # Construct train dataloader from scratch after 
        if 'adj' in train_list[0]:
            train_loader = DenseLoader(train_list, args.batch_size, shuffle=True)
        else:
            train_loader = DataLoader(train_list, args.batch_size, shuffle=True)
        
        # Re-initialize GNN
        model = Net(dataset, args.num_layers, args.hidden)
        model.to(device).reset_parameters()
        optimizer = Adam(model.parameters(), lr=args.GNN_lr, weight_decay=args.GNN_wd) 
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_start = time.perf_counter()
        trn_loss, val_loss, val_acc, best_tst_acc = 0, 0, 0, 0
        best_val_acc, best_val_loss = 0, np.inf
        for epoch in range(1, args.epochs+1):
            trn_loss = train(model, optimizer, train_loader,device)
            val_loss = eval_loss(model, val_loader,device)
            val_acc = eval_acc(model, val_loader,device)
            tst_acc = eval_acc(model, test_loader,device)

            # save best model base on val_loss!
            if val_loss <= best_val_loss:
                best_val_acc = val_acc
                best_tst_acc = eval_acc(model, test_loader,device)
                best_val_loss = val_loss

            if args.verbose:
                if epoch % args.display_step == 0:
                    print(f'Epoch: {epoch}, Val Acc: {best_val_acc:.4f}, Test Acc: {best_tst_acc:.4f}')

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_end = time.perf_counter()
        duration = t_end - t_start
        removal_time[i] = duration
        acc_removal[0, i] = best_val_acc
        acc_removal[1, i] = best_tst_acc
        if (i+1) % args.rm_disp_step == 0:
            print(f'Training Time: {duration:.3f} s')
            print(f'Remove Iteration: {i+1}, Val Acc: {best_val_acc:.4f}, Test Acc: {best_tst_acc:.4f}')
        
    return removal_time, acc_removal

def Retrain_GST(args, scattering, train_list, device, w_approx, budget, graph_removal_queue=None,
                removal_queue=None, val_list=None, test_list=None, nonlin=True, gamma=1/4):
    """
    train_list, val_list, test_list are not datasets, they are list of Data objects.
    The removal logic is Unlearn_GST will generate the removal_queue, and Retrain_GNN will use the exact same removal_queue
    """
    
    # F for Scattering Transform
    f = np.sqrt(args.L)
    
    grad_norm_approx = torch.zeros(args.num_removes).float() # Data dependent grad res norm
    grad_norm_real = torch.zeros(args.num_removes).float() # true grad res norm
    grad_norm_worst = torch.zeros(args.num_removes).float() # worst case grad res norm
    
    removal_time = torch.zeros(args.num_removes).float() # record the time of each removal
    acc_removal = torch.zeros((2, args.num_removes)).float() # record the acc after removal, 0 for val and 1 for test
    num_retrain = 0
    b_std = args.std
    
    if removal_queue is None:
        # Remove one node for a graph, generate removal graph_id in advance.
        graph_removal_queue = torch.randperm(len(train_list))
        removal_queue = []
    else:
        # will use the existing removal_queue
        graph_removal_queue = None
        
    grad_norm_approx_sum = 0
    
    X_train_old, y_train = get_GST_emb(train_list, scattering, device, train_split=True, nonlin=nonlin, batch=args.batch_size)  # y_train will not change during unlearning process for now
    if val_list and test_list:
        X_val, y_val = get_GST_emb(val_list, scattering, device, train_split=False, nonlin=nonlin, batch=args.batch_size)
        X_test, y_test = get_GST_emb(test_list, scattering, device, train_split=False, nonlin=nonlin, batch=args.batch_size)
    
    # start the removal process
    for i in range(args.num_removes):
        # we have generated the order of graphs to remove
        if graph_removal_queue is not None:
            # remove one node from graph_removal_queue[i % len(train_list)]
            train_list, removal_queue = remove_node_from_graph(train_list, graph_id=graph_removal_queue[i%len(train_list)], removal_queue=removal_queue)
        else:
            # remove one node based on removal_queue
            train_list = remove_node_from_graph(train_list, graph_id=removal_queue[i][0], node_id=removal_queue[i][1])
        
        X_train_new = X_train_old.clone().detach()
        
        t_start = time.perf_counter()
        # generate train embeddings AFTER removal, only for the affect graph
        new_graph_emb, _ = get_GST_emb([train_list[removal_queue[i][0]]], scattering, device, train_split=True, nonlin=nonlin, batch=-1)
        
        X_train_new[removal_queue[i][0], :] = new_graph_emb.view(-1)
        
        b = 0
        w_approx = ovr_lr_optimize(X_train_new, y_train, args.lam, None, b=b, num_steps=args.epochs, verbose=False,
                                   opt_choice=args.optimizer, lr=args.lr, wd=args.wd)
        
        
        removal_time[i] = time.perf_counter() - t_start
        # record acc each round
        if val_list and test_list:
            acc_removal[0, i] = ovr_lr_eval(w_approx, X_val, y_val)
            acc_removal[1, i] = ovr_lr_eval(w_approx, X_test, y_test)
        
        
        # Remember to replace X_old with X_new
        X_train_old = X_train_new.clone().detach()
        
        if (i+1) % args.rm_disp_step == 0:
            print('Remove iteration %d: time = %.2fs, number of retrain = %d' % (i+1, removal_time[i], num_retrain))
            print('Val acc = %.4f, Test acc = %.4f' % (acc_removal[0, i], acc_removal[1, i]))
    
    return removal_time, num_retrain, acc_removal, grad_norm_approx, grad_norm_real, grad_norm_worst, removal_queue