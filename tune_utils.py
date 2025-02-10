import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import trange
from train_utils import train_epoch, evaluate
from dbs_s_eegnet import EEGNet
import ray.train as r_train
from train_utils import kFoldSplit, toModelData
import pickle, os, json
import matplotlib.pyplot as plt
import pandas as pd
from math import copysign as sign

def to_list(input):
    return [input]

def analyseTune(path, tune_params, folder, tune_verbose, save_verbose = False, save_format = "png"):
    """
    This function analyses the tuning results and plots the best hyperparameters.
    
    Args:
        path (str): The path to the folder containing the tuning results.
        tune_params (dict): The parameters used for tuning.
        folder (str): The name of the folder containing the tuning results.
        tune_verbose (bool): Whether to print verbose information during tuning.
        save_verbose (bool): Whether to print verbose information during saving.
        save_format (str): The format to save the plots in.
    """
    if save_verbose: print(f"Study folder: {path}")
    of_interest = ['val_acc','val_loss','train_acc','train_loss'] # Metrics of interest that were reported during tuning
    min_epochs = tune_params['grace_period'] # Length of the transient phase in terms of loss/acc curves

    full_params = []
    full_df = []
    results = []
    results_path = f"{path}/ray_results/{folder}/"
    num_start = len(next(os.walk(f"{results_path}."))[1]) # Number of tune samples
    num_start_pre = num_start
    c = 0 # Counter for how iterations of the loop below are skipped
    inv_idx = [i for i in range(num_start)][::-1] # Inverse indices, this is for the loop where we remove trials, and we want to remove trials starting from the end to avoid errors
    
    if save_verbose: print(f"Output tuning folder: ´{results_path}") # Debugging

    for i, trial in enumerate(next(os.walk(f"{results_path}."))[1]): # In this loop we go through each tune trial (sample) and extract the relevant information, starting from the last one
        i = inv_idx[i]
        if "error.txt" in os.listdir(f"{results_path}{trial}/") or "params.json" not in os.listdir(f"{results_path}{trial}/") or "progress.csv" not in os.listdir(f"{results_path}{trial}/"): # If the trial is not valid, remove it
            if i == num_start_pre-1: # If the first trial is invalid and it is the first one, just change the length of the trials
                num_start -= 1
                num_start_pre -=1
            else: # Otherwise, the matrices has been initialised (see below), and delete the corresponding entries
                param_matrix = np.delete(param_matrix,i,0)
                results_matrix = np.delete(results_matrix,i,1)
            c+=1 # Increase the skip counter
            continue # Skip this iteration
        # If the trial is valid:
        with open(f"{results_path}{trial}/params.json") as json_file:
            params = json.load(json_file) # Load the run parameters
        if i == num_start_pre-1: # If it is the first trial, initialise the matrices
            param_matrix = -5*np.ones((num_start, len(params))) # Initialise the parameter matrix, where the hyperparameters are stored
            results_matrix = -5*np.ones((len(of_interest),num_start, len(params))) # Initialise the results matrix, where the results are stored
            keys = [l for l in params.keys()] # Get the parameter names
        df = pd.read_csv(f"{results_path}{trial}/progress.csv") # Load tune progression data
        df = df[of_interest] # Only keeps the metrics that are useful
        full_df.append(df) # Append the dataframe to the list of dataframes
        full_params.append(params) # Append the parameters to the list of parameters
        # Results are saved as a dictionary first, then this is used to populate the matrix and they are appended to the list of results
        result = {"best_acc": df['val_acc'].max(),"avg_acc": df['val_acc'].loc[min_epochs-10:].mean(),"best_loss": df['val_loss'].min(),"avg_loss": df['val_loss'].loc[min_epochs-10:].mean()}
        for j,key in enumerate(params): # Iterate through parameters
            param_matrix[i,j] = params[key]
            for k, res_key in enumerate(result): # Iterate through result types
                results_matrix[k,i,j] = result[res_key]
        results.append(result)

    categorical = {"lr": False, "fc_dropout": True, "conv_dropout": True, "F1": True, "D": True, "L1": False, "L2": False} # Whether a result is categorical or continuous for plotting
    formats = {"lr": '{:.2e}'.format, "fc_dropout": '{:.2f}'.format, "conv_dropout":'{:.2f}'.format, "F1": '{:.0f}'.format, "D": '{:.0f}'.format, "L1": '{:.2e}'.format, "L2": '{:.2e}'.format} # Reporting format for plots
    of_interest = [key for key in result]
    n_cols = 1 # Number of columns in subplot
    fig, ax = plt.subplots(len(params)//n_cols+(len(params)%n_cols),n_cols,figsize=(22,22)) # Create the figure, one subplot for every hyperparameter being tuned
    if type(ax[0]) != np.ndarray:
        ax = list(map(to_list,ax)) # If there is only one subplot, make it a list
    best_params = {} # Dictionary to store the best parameters
    colours = ["b","c","r","m"] # Colours for the different results (best acc, avg acc, best loss, avg loss)  
    for res in range(4): # Iterate through the results
        for k,key in enumerate(keys): # Iterate through the hyperparameters
            curr_row = param_matrix[:,k] # Get the current row of the parameter matrix
            sort_idx = np.argsort(curr_row) # Sort the indices of the current row (so in respect to the parameter)
            sorted_matrix = results_matrix[res,sort_idx,0] # Sort the results matrix according to the current row
            if res < 2: # If the result is an accuracy, we want to maximise it
                np.nan_to_num(sorted_matrix,False, min(sorted_matrix)-0.1)
                best_result_idx = np.argmax(sorted_matrix)
            else: # If the result is a loss, we want to minimise it
                np.nan_to_num(sorted_matrix,False, max(sorted_matrix)+0.1)
                best_result_idx = np.argmin(sorted_matrix)
            best_param = np.sort(param_matrix[:,k])[best_result_idx] # Get the best parameter value
            best_params[key] = best_param
            if categorical[key]:
                add_factor = sign(1,res%2-1)*np.sort(param_matrix[:,k])[0]*0.1 # Add a small number to the x-axis to avoid overlap
                ax[k//n_cols][k%n_cols].plot(np.sort(param_matrix[:,k])+add_factor,sorted_matrix,f'{colours[res]}*',label = of_interest[res]) # Plot the results
                ax[k//n_cols][k%n_cols].set_xticks(np.unique(np.sort(param_matrix[:,k]))) # Set the x-ticks to the unique values of the parameter
            else:
                ax[k//n_cols][k%n_cols].plot(np.sort(param_matrix[:,k]),sorted_matrix,f'{colours[res]}*', label = of_interest[res]) # Plot the results
                # ax[k//n_cols][k%n_cols].bar()
                ax[k//n_cols][k%n_cols].set_xscale('log') # All continuous parameters are plotted on a log scale
            ax[k//n_cols][k%n_cols].set_xlabel(key) # Set the x-axis label to the hyperparameter name
            ax[k//n_cols][k%n_cols].plot(best_param,sorted_matrix[best_result_idx],'k*',label = f'Best result = {sorted_matrix[best_result_idx]:.2f}') # Plot the best result as a black star
            ax[k//n_cols][k%n_cols].legend()
        dict_str = [f"{key}: {formats[key](best_params[key])}" for key in best_params] # Create a string of the best parameters
        if tune_verbose: print(best_params) # Print the best parameters
    plt.suptitle(f"Best result obtained with:\n{dict_str}")
    plt.show()

    # Below the plots are saved and the results are saved in a pickle file for loading the best hyperparameters in training
    with open(f"{results_path}tune_params","wb") as f: # Save the best parameters dictionary in a pickle file	
        pickle.dump(best_params,f)
    # Handle the path for saving the plots and results to be inside the correct folder (different than tuning results)

    
    # Save the plots and results, since tuning is the first step, it creates the necessary folders
    if not os.path.exists(f"{path}/plots/"):
        os.makedirs(f"{path}/plots/")
    plt.savefig(f"{path}/plots/tune_plot.{save_format}", format = save_format)
    if save_verbose: print(f"Saving {path}/plots/tune_plot")
    

    if not os.path.exists(f"{path}/results/"):
        os.makedirs(f"{path}/results/")
    with open(f"{path}/results/tune_params","wb") as f:
        pickle.dump(best_params,f)
    if save_verbose: print(f"Saving {path}/results/tune_params")

def trainable_multifold(config: dict, data, input_params):
    """Performs the hyperparameter tuning of the model

    input_params:
        config (dict): Dictionary that contains the hyperparameters being tuned
        in_data (list): The generated data (dataframe, row1_ind, free_row1, labelled_dataset)
        input_params (dict): Input arguments
    """
    # Initialise lists since we are doing over multiple folds
    fold_indices = kFoldSplit(data['y_train'],
                              input_params['n_folds'],
                              input_params['val_size'],
                              input_params['val_bal_ratio'],
                              dataset_verbose = input_params['dataset_verbose'])
    # Set up the hyperparameters in the net_params dictionary
    net_params = input_params['net_params']
    if 'L2' in config:
        if input_params['tune_verbose']: print("L2 being configured...")
    else:
        config['L2'] = net_params['L2']
        if input_params['tune_verbose']: print("L2 not being configured...")
    if 'lr' in config:
        if input_params['tune_verbose']:print("lr being configured...")
    else:
        config['lr'] = net_params['lr'] 
        if input_params['tune_verbose']: print("lr not being configured...")
    if 'L1' in config:
        if input_params['tune_verbose']: print("L1 being configured...")
    else:
        config['L1'] = net_params['L1']
        if input_params['tune_verbose']: print("L1 not being configured")

    # Set up the optimizers and networks
    nets = []
    optimizers = []
    criterion = nn.CrossEntropyLoss()
    if input_params['epoch_verbose']:
        train_iterator = trange(1,1+input_params['epochs'],desc = 'Epochs')
    else:
        train_iterator = range(1,1+input_params['epochs'])
        
    # This loop generates the data for each fold
    for fold_nr, fold in enumerate(fold_indices):
        if fold_nr:
            tune_verbose = False
        else:
            tune_verbose = input_params['tune_verbose']
        net = EEGNet(num_classes=2,
                     config=config,
                     tune_bool = True,
                     num_channels = input_params['n_channels'],
                     seg_len = int(input_params['sample_freq']*input_params['seg_dur']),
                     device=input_params['device'],
                     net_params = net_params,
                     tune_verbose=tune_verbose).to(input_params['device'])
        nets.append(net)
        optimizer = optim.Adam(net.parameters(), lr = config['lr'], weight_decay = config['L2'])
        optimizers.append(optimizer)
    
    # Train for each epoch
    for epoch in train_iterator:
        tr_accs = []
        tr_losses = []
        val_accs = []
        val_losses = []
        # Train and evaluate across each fold
        for fold_nr, fold in enumerate(fold_indices):
            X_train, y_train, X_val, y_val = toModelData(data, fold)
            train_epoch(nets[fold_nr], X_train, y_train, criterion, optimizers[fold_nr], input_params['batch_size'], input_params['device'], config['L1']) # Train and update weights
            if X_train.shape[0] > 7000: # For memory purposes, split the data into smaller chunks
                tr_acc_i = []
                tr_loss_i = []
                for q in range(X_train.shape[0]//5000):
                    X_curr = X_train[5000*q:5000*(q+1)]
                    y_curr = y_train[5000*q:5000*(q+1)]
                    tr_loss_ii, tr_acc_ii, _ = evaluate(nets[fold_nr], criterion, X_curr, y_curr, input_params['device'])
                    tr_acc_i.append(tr_acc_ii)
                    tr_loss_i.append(tr_loss_ii)
                tr_acc_i = np.mean(tr_acc_i) # Train accuracy of this fold and epoch
                tr_loss_i = np.mean(tr_loss_i) # Train loss of this fold and epoch
            else: 
                tr_loss_i, tr_acc_i, _ = evaluate(nets[fold_nr], criterion, X_train, y_train, input_params['device'])
            tr_accs.append(tr_acc_i) # Append the accuracy of this fold and epoch
            tr_losses.append(tr_loss_i) # Append the loss of this fold and epoch
            val_loss_i, val_acc_i,_ = evaluate(nets[fold_nr], criterion, X_val, y_val, input_params['device'])
            val_losses.append(val_loss_i) # Append the validation loss of this fold and epoch
            val_accs.append(val_acc_i) # Append the validation accuracy of this fold and epoch

        # Average metrics across folds
        val_acc = np.mean(val_accs)
        val_loss = np.mean(val_losses)
        tr_acc = np.mean(tr_accs)
        tr_loss = np.mean(tr_losses)

        # Report to Ray the averaged metrics, this is later used to analyse the tuning iteration
        r_train.report({"val_loss": val_loss, "val_acc": val_acc, "train_loss": tr_loss, "train_acc": tr_acc})