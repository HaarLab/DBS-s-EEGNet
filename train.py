import os
import numpy as np
import json, pickle
from copy import deepcopy
from train_utils import *
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm, trange
from dbs_s_eegnet import EEGNet

def myTrain(data, input_params, out_folder, plot_types = ["loss_plt", "acc_plt"], to_int = ['F1','D']):
    """
    This runs the training of the EEGNet model with all the input parameters and saves the results in the specified folder.
    
    Args:
        data (dict): Dictionary containing the training and testing data and labels.
        input_params (dict): Dictionary containing the input parameters for the training.
        out_folder (str): The folder to save the results in.
        plot_types (list): List of plot types to generate.
        to_int (list): List of EEGNet hyperparameters to convert to integers.
    
    Returns:
        None
    """
    print(f"Starting training in mode = {input_params['mode']} for {input_params['study_name']}...")
    strFolder = getFolder(out_folder, save_verbose = input_params['save_verbose'])
    input_params['strFolder'] = strFolder
    if input_params['config'] != dict():
        argss = deepcopy(input_params)
        argss.pop('config')
        json_args = json.dumps(argss)
    else:
        json_args = json.dumps(input_params)
    with open(f"{strFolder}/params.json","w") as f:
        f.write(json_args)
    
    fold_indices = kFoldSplit(data['y_train'], input_params['n_folds'], input_params['val_size'], input_params['val_bal_ratio'], dataset_verbose = input_params['dataset_verbose'])
    results = {"val_accs": [],
                "train_accs": [],
                "train_losses":[],
                "val_losses": [],
                "sys_val_accs": [],
                "train_sets": [],
                "val_sets": [],
                "test_sets": []}
    val_mat = []
    train_mat = []
    net_params = input_params['net_params']
    if input_params['use_tune']:
        tune_path = f"{out_folder}/results/tune_params"
        print(f"Retrieving tune config from: {tune_path}")
        with open(tune_path,"rb") as f:
            tune_params = pickle.load(f)
        for key in tune_params:
            if key in to_int: tune_params[key] = int(tune_params[key])
            net_params[key] = tune_params[key] 
    for fold_nr, fold in enumerate(fold_indices):

        net = EEGNet(net_params = net_params, tune_bool = False,
                     num_channels = input_params['n_channels'],
                     seg_len = input_params['seg_len'],
                     device = input_params['device'])
        net.to(input_params['device'])
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=net_params['lr'])
        X_train, y_train, X_val, y_val = toModelData(data, fold)
        if input_params['metric'] == 'acc':
            max_val = 0
        elif input_params['metric'] == 'loss':
            min_val = 1000000
        if input_params['epoch_verbose']:
            train_iterator = trange(1,1+input_params['epochs'],desc = 'Epochs')
        else:
            train_iterator = range(1,1+input_params['epochs'])
        tr_accs = []
        tr_losses = []
        val_accs = []
        val_losses = []
        sys_val_accs = []
        for epoch in train_iterator:
            tr_acc, tr_loss, val_acc, val_loss = train_iter(net, X_train, y_train, X_val, y_val,
                                                            criterion, optimizer, input_params['batch_size'],
                                                            input_params['device'], net_params['L1'], input_params)
            if input_params['metric'] == 'acc':
                if val_acc > max_val:
                    best_net = deepcopy(net)
                    max_val = val_acc
                    best_optimizer = deepcopy(optimizer)
                    best_epoch = epoch
            elif input_params['metric'] == 'loss':
                if val_loss < min_val:
                    best_net = deepcopy(net)
                    min_val = val_loss
                    best_optimizer = deepcopy(optimizer)
                    best_epoch = epoch
            tr_accs.append(tr_acc)
            tr_losses.append(tr_loss)
            val_accs.append(val_acc)
            val_losses.append(val_loss)
            if epoch in input_params['checkpoints'] and input_params['save_model']:
                # Save checkpoint if the epoch is in the checkpoints list
                save_checkpoint(net,optimizer, filename = f"{strFolder}/checkpoint_f{fold_nr+1}_e{epoch}.tar")
                _, sys_val_acc, _ = evaluate(net,criterion, X_val, y_val, input_params['device'])
                sys_val_accs.append(sys_val_acc)
        save_checkpoint(best_net,best_optimizer, filename = f"{strFolder}/checkpoint_f{fold_nr+1}_best_e{best_epoch}.tar")
        input_params['actual_sizes'] = [X_train.shape[0], X_val.shape[0]]
        results['val_accs'].append(val_accs)
        results['train_accs'].append(tr_accs)
        results['train_losses'].append(tr_losses)
        results['val_losses'].append(val_losses)
        results['sys_val_accs'].append(sys_val_accs)
        savePlots(results, input_params, strFolder, fold_nr+1, plot_types)
        val_inds = fold[0]
        train_inds = fold[1]
        # The code below appends the indices to a matrix in a format that's easy to write into a csv file.
        results['val_sets'].append(val_inds)
        results['train_sets'].append(train_inds)
        del net 
    saveResults(results, input_params, strFolder)