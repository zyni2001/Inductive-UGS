import dgl
import numpy as np
import os
import socket
import time
import random
import glob
import argparse, json
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from train.train_superpixels_graph_classification import train_epoch, evaluate_network
from tqdm import tqdm
from nets.superpixels_graph_classification.load_net import gnn_model # import all GNNS
from data.data import LoadData # import dataset
import pruning
import pdb

"""
    GPU Setup
"""
def gpu_setup(use_gpu, gpu_id):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  

    if torch.cuda.is_available() and use_gpu:
        print('cuda available with GPU:',torch.cuda.get_device_name(0))
        device = torch.device("cuda")
    else:
        print('cuda not available')
        device = torch.device("cpu")
    return device


"""
    VIEWING MODEL CONFIG AND PARAMS
"""
def view_model_param(MODEL_NAME, net_params):
    model = gnn_model(MODEL_NAME, net_params)
    total_param = 0
    print("MODEL DETAILS:\n")
    #print(model)
    for param in model.parameters():
        # print(param.data.size())
        total_param += np.prod(list(param.data.size()))
    print('MODEL/Total parameters:', MODEL_NAME, total_param)
    return total_param

"""
    TRAINING CODE
"""

def run_rprp(MODEL_NAME, dataset, params, net_params, rp_num, args):
    
    DATASET_NAME = dataset.name
    if MODEL_NAME in ['GCN', 'GAT']:
        if net_params['self_loop']:
            print("[!] Adding graph self-loops for GCN/GAT models (central node trick).")
            dataset._add_self_loops()
    
    dataset_pru = LoadData(DATASET_NAME, args)
    
    trainset_pru, valset_pru, testset_pru = dataset_pru.train, dataset_pru.val, dataset_pru.test 
    trainset, valset, testset = dataset.train, dataset.val, dataset.test

    sp_train = pruning.print_pruning_percent(trainset, trainset_pru)
    sp_test = pruning.print_pruning_percent(testset, testset_pru)
    sp_val = pruning.print_pruning_percent(valset, valset_pru)
    spa = (sp_train + sp_test + sp_val) / 3.0

    device = net_params['device']
 
    random.seed(params['seed'])
    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])
    torch.cuda.manual_seed(params['seed'])
    
    print("Training Graphs: ", len(trainset))
    print("Validation Graphs: ", len(valset))
    print("Test Graphs: ", len(testset))
    print("Number of Classes: ", net_params['n_classes'])

    model = gnn_model(MODEL_NAME, net_params)
    model = model.to(device)
    
    pruning.pruning_model(model, args.pw, random=True)
    spw = pruning.see_zero_rate(model)
    
    optimizer = optim.Adam(model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                     mode='min',
                                                     factor=params['lr_reduce_factor'],
                                                     patience=params['lr_schedule_patience'],
                                                     verbose=True)
    
    # batching exception for Diffpool
    drop_last = True if MODEL_NAME == 'DiffPool' else False
    # import train functions for all other GCNs
    train_loader = DataLoader(trainset_pru, batch_size=params['batch_size'], shuffle=True, drop_last=drop_last, collate_fn=dataset.collate)
    val_loader = DataLoader(valset_pru, batch_size=params['batch_size'], shuffle=False, drop_last=drop_last, collate_fn=dataset.collate)
    test_loader = DataLoader(testset_pru, batch_size=params['batch_size'], shuffle=False, drop_last=drop_last, collate_fn=dataset.collate)
    run_time, best_val_acc, best_epoch, update_test_acc  = 0, 0, 0, 0
    for epoch in range(args.epochs):

        t0 = time.time()
        epoch_train_loss, epoch_train_acc, optimizer = train_epoch(model, optimizer, device, train_loader, epoch, args)
        epoch_val_loss, epoch_val_acc = evaluate_network(model, device, val_loader, epoch)
        _, epoch_test_acc = evaluate_network(model, device, test_loader, epoch)                
        scheduler.step(epoch_val_loss)
        epoch_time = time.time() - t0
        run_time += epoch_time

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            update_test_acc = epoch_test_acc
            best_epoch = epoch

        print('-'*120)
        print( time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' | ' +
                'RP:[{}] spa[{:.2f}%] spw:[{:.2f}%] | Epoch [{}/{}]: Loss [{:.4f}] Train:[{:.2f}] Val:[{:.2f}] Test:[{:.2f}] | Update Test:[{:.2f}] at epoch:[{}] | Run Total Time: [{:.2f} min]'
                .format(rp_num,
                        spa * 100,
                        spw * 100,
                        epoch + 1, 
                        args.epochs,
                        epoch_train_loss, 
                        epoch_train_acc * 100,
                        epoch_val_acc * 100, 
                        epoch_test_acc * 100, 
                        update_test_acc * 100,
                        best_epoch,
                        run_time / 60)) 
        print('-'*120)
        
    print("syd: RP:[{}] | spa:[{:.2f}%] spw:[{:.2f}%] | Update Test:[{:.2f}] at epoch:[{}]"
            .format(rp_num,
                    spa * 100,
                    spw * 100,
                    update_test_acc * 100,
                    best_epoch))

def main():    
    """
        USER CONTROLS
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--pa', type=float, default=0)
    parser.add_argument('--pw', type=float, default=0)
    parser.add_argument('--random_type', type=str, default="rprp", help="rprp, rpimp, rpnp")
    parser.add_argument('--epochs', type=int, help="Please give a value for epochs")

    parser.add_argument('--config', help="Please give a config.json file with training/model/data/param details")
    parser.add_argument('--gpu_id', help="Please give a value for gpu id")
    parser.add_argument('--model', help="Please give a value for model name")
    parser.add_argument('--dataset', help="Please give a value for dataset name")
    parser.add_argument('--out_dir', help="Please give a value for out_dir")
    parser.add_argument('--seed', help="Please give a value for seed")

    parser.add_argument('--batch_size', help="Please give a value for batch_size")
    parser.add_argument('--init_lr', help="Please give a value for init_lr")
    parser.add_argument('--lr_reduce_factor', help="Please give a value for lr_reduce_factor")
    parser.add_argument('--lr_schedule_patience', help="Please give a value for lr_schedule_patience")
    parser.add_argument('--min_lr', help="Please give a value for min_lr")
    parser.add_argument('--weight_decay', help="Please give a value for weight_decay")
    parser.add_argument('--print_epoch_interval', help="Please give a value for print_epoch_interval")    
    parser.add_argument('--L', help="Please give a value for L")
    parser.add_argument('--hidden_dim', help="Please give a value for hidden_dim")
    parser.add_argument('--out_dim', help="Please give a value for out_dim")
    parser.add_argument('--residual', help="Please give a value for residual")
    parser.add_argument('--edge_feat', help="Please give a value for edge_feat")
    parser.add_argument('--readout', help="Please give a value for readout")
    parser.add_argument('--kernel', help="Please give a value for kernel")
    parser.add_argument('--n_heads', help="Please give a value for n_heads")
    parser.add_argument('--gated', help="Please give a value for gated")
    parser.add_argument('--in_feat_dropout', help="Please give a value for in_feat_dropout")
    parser.add_argument('--dropout', help="Please give a value for dropout")
    parser.add_argument('--layer_norm', help="Please give a value for layer_norm")
    parser.add_argument('--batch_norm', help="Please give a value for batch_norm")
    parser.add_argument('--sage_aggregator', help="Please give a value for sage_aggregator")
    parser.add_argument('--data_mode', help="Please give a value for data_mode")
    parser.add_argument('--num_pool', help="Please give a value for num_pool")
    parser.add_argument('--gnn_per_block', help="Please give a value for gnn_per_block")
    parser.add_argument('--embedding_dim', help="Please give a value for embedding_dim")
    parser.add_argument('--pool_ratio', help="Please give a value for pool_ratio")
    parser.add_argument('--linkpred', help="Please give a value for linkpred")
    parser.add_argument('--cat', help="Please give a value for cat")
    parser.add_argument('--self_loop', help="Please give a value for self_loop")
    parser.add_argument('--max_time', help="Please give a value for max_time")
    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)
        
    # device
    if args.gpu_id is not None:
        config['gpu']['id'] = int(args.gpu_id)
        config['gpu']['use'] = True
    device = gpu_setup(config['gpu']['use'], config['gpu']['id'])
    # model, dataset, out_dir
    if args.model is not None:
        MODEL_NAME = args.model
    else:
        MODEL_NAME = config['model']
    if args.dataset is not None:
        DATASET_NAME = args.dataset
    else:
        DATASET_NAME = config['dataset']
    dataset = LoadData(DATASET_NAME, args)
    if args.out_dir is not None:
        out_dir = args.out_dir
    else:
        out_dir = config['out_dir']
    # parameters
    params = config['params']
    if args.seed is not None:
        params['seed'] = int(args.seed)
    if args.epochs is not None:
        params['epochs'] = int(args.epochs)
    if args.batch_size is not None:
        params['batch_size'] = int(args.batch_size)
    if args.init_lr is not None:
        params['init_lr'] = float(args.init_lr)
    if args.lr_reduce_factor is not None:
        params['lr_reduce_factor'] = float(args.lr_reduce_factor)
    if args.lr_schedule_patience is not None:
        params['lr_schedule_patience'] = int(args.lr_schedule_patience)
    if args.min_lr is not None:
        params['min_lr'] = float(args.min_lr)
    if args.weight_decay is not None:
        params['weight_decay'] = float(args.weight_decay)
    if args.print_epoch_interval is not None:
        params['print_epoch_interval'] = int(args.print_epoch_interval)
    if args.max_time is not None:
        params['max_time'] = float(args.max_time)
    # network parameters
    net_params = config['net_params']
    net_params['device'] = device
    net_params['gpu_id'] = config['gpu']['id']
    net_params['batch_size'] = params['batch_size']
    if args.L is not None:
        net_params['L'] = int(args.L)
    if args.hidden_dim is not None:
        net_params['hidden_dim'] = int(args.hidden_dim)
    if args.out_dim is not None:
        net_params['out_dim'] = int(args.out_dim)   
    if args.residual is not None:
        net_params['residual'] = True if args.residual=='True' else False
    if args.edge_feat is not None:
        net_params['edge_feat'] = True if args.edge_feat=='True' else False
    if args.readout is not None:
        net_params['readout'] = args.readout
    if args.kernel is not None:
        net_params['kernel'] = int(args.kernel)
    if args.n_heads is not None:
        net_params['n_heads'] = int(args.n_heads)
    if args.gated is not None:
        net_params['gated'] = True if args.gated=='True' else False
    if args.in_feat_dropout is not None:
        net_params['in_feat_dropout'] = float(args.in_feat_dropout)
    if args.dropout is not None:
        net_params['dropout'] = float(args.dropout)
    if args.layer_norm is not None:
        net_params['layer_norm'] = True if args.layer_norm=='True' else False
    if args.batch_norm is not None:
        net_params['batch_norm'] = True if args.batch_norm=='True' else False
    if args.sage_aggregator is not None:
        net_params['sage_aggregator'] = args.sage_aggregator
    if args.data_mode is not None:
        net_params['data_mode'] = args.data_mode
    if args.num_pool is not None:
        net_params['num_pool'] = int(args.num_pool)
    if args.gnn_per_block is not None:
        net_params['gnn_per_block'] = int(args.gnn_per_block)
    if args.embedding_dim is not None:
        net_params['embedding_dim'] = int(args.embedding_dim)
    if args.pool_ratio is not None:
        net_params['pool_ratio'] = float(args.pool_ratio)
    if args.linkpred is not None:
        net_params['linkpred'] = True if args.linkpred=='True' else False
    if args.cat is not None:
        net_params['cat'] = True if args.cat=='True' else False
    if args.self_loop is not None:
        net_params['self_loop'] = True if args.self_loop=='True' else False
        
    # Superpixels
    net_params['in_dim'] = dataset.train[0][0].ndata['feat'][0].size(0)
    net_params['in_dim_edge'] = dataset.train[0][0].edata['feat'][0].size(0)
    num_classes = len(np.unique(np.array(dataset.train[:][1])))
    net_params['n_classes'] = num_classes
    net_params['total_param'] = view_model_param(MODEL_NAME, net_params)
    
    adj_pruning = 0.05
    wei_pruning = 0.2
    percent_list = [(1 - (1 - adj_pruning) ** (i + 1), 1 - (1 - wei_pruning) ** (i + 1)) for i in range(20)]
    percent_list.append((0.0,0.0)) # no pruning baseline

    for rp_num, (pa, pw) in enumerate(percent_list):
        args.pa = pa
        args.pw = 0
        run_rprp(MODEL_NAME, dataset, params, net_params, rp_num, args)

    
if __name__ == '__main__':
    main()

