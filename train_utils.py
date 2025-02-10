##### ----------------------------------------------------------------------------------------------------------------  #####
#                                                                                                                           #
# THIS PYTHON FILE IS USED TO SUPPORT THE TRAINING OF THE EEGNET MODEL ON THE EEG DATA AS WELL AS TO HELP TESTING           #
# This file doesn't import from any other python file but it exports its functions throughout the repository                #
#                                                                                                                           #
##### ----------------------------------------------------------------------------------------------------------------  #####


from tqdm import trange
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, recall_score, precision_score, roc_auc_score
import matplotlib.pyplot as plt
import torch
import csv
import ast
from copy import deepcopy
from numba import jit, njit
import operator
import os
import math
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
from tqdm import tqdm
from scipy.io import loadmat
from sklearn.model_selection import KFold

def saveResults(results, input_params, strFolder):
    """ Saves the results of the training process
    This saves the results in csv format and the training/val indices in csv format as well.
    
    Args:
        results (dict): Dictionary containing the results of the training.
        input_params (dict): Dictionary containing the input parameters for the training.
        strFolder (str): Path to the folder where the results are saved.
    """
    for key in list(results.keys())[0:5]: # Iterates through the accuracy/loss results
        path = gen_path(strFolder, key, ".csv") # Generates valid path for the results to be saved
        lengths = []
        for i in results[key]: # Iterates through each result (1 per fold)
            if key != 'test_accs':
                if type(i) == list:
                    lengths.append(len(i))
                max_len = max(lengths)
                for i in results[key]:
                    if type(i) == list:
                        if len(i)<max_len:
                            i.extend([0 for i in range(max_len-len(i))]) # Adds 0 to the end of the list if it's shorter than the longest list
        write_results(path, results[key], input_params['n_folds']+1, save_verbose=input_params['save_verbose']) # Writes the results in csv format


    # Saves the indices in csv format
    mode = 'w' if os.path.exists(f"{strFolder}/val_indices_fALL.csv") else 'w+'
    with open(f"{strFolder}/val_indices_fALL.csv", mode) as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerows(results['val_sets'])
        
    mode = 'w' if os.path.exists(f"{strFolder}/train_indices_fALL.csv") else 'w+'
    with open(f"{strFolder}/train_indices_fALL.csv", mode) as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerows(results['train_sets'])

def savePlots(results, input_params, strFolder, fold_nr,
              plot_types=["loss_plt", "acc_plt"]):
    """ Saves the plots for the training and cross-validation process
    
    Args:
        results (dict): Dictionary containing the results of the training.
        input_params (dict): Dictionary containing the input parameters for the training.
        strFolder (str): Path to the folder where the results are saved.
        fold_nr (int): Fold number.
        plot_types (list): List of plot types to save (e.g. ["loss_plt", "acc_plt"]).
    """
    save_format = input_params['save_format']
    format_len = len(save_format)
    # The following lines save the results as plots and csv files
    tr_acc = results['train_accs'][-1]
    val_acc = results['val_accs'][-1]
    tr_loss = results['train_losses'][-1]
    val_loss = results['val_losses'][-1]
    for i in range(len(plot_types)): # Iterates through every save type
        # The following lines adjust the filename to avoid overwriting old files
        file_counter = 1
        strFile = f"{plot_types[i]}{file_counter}.{save_format}"
        if os.path.isfile(strFolder+"/"+strFile):
            strFile = strFile[:-(2+format_len)]+str(file_counter)+"."+save_format
        while os.path.isfile(strFolder+"/"+strFile):
            ind = (2+format_len)+int(np.log10(file_counter-[1 if file_counter-1 else 0][0]))
            strFile = strFile[:-ind]+str(file_counter)+"."+save_format
            file_counter+=1
        
        if plot_types[i] == "loss_plt": # plot_types loss curves for train and validation datasets
            lossPlot(tr_loss, val_loss, fold_nr, input_params, path = strFolder+"/"+strFile)
        elif plot_types[i] == "acc_plt": # plot_types accuracy curves for train and validation datasets
            accPlot(tr_acc, val_acc, fold_nr, input_params, path = strFolder+"/"+strFile)

def train_iter(net, X_train, y_train, X_val, y_val,
               criterion, optimizer, batch_size,
               device, lambda1, input_params):
    """ Trains and evaluates for one iteration
    
    Args:
        net (EEGNet): Network
        X_train (np.array): N x 1 x 2*n_channels x seg_len array containing the EEG data of the first and second segment in each pair
        y_train (np.array): N x 2 array containing the labels of the pairs
        X_val (np.array): N x 1 x 2*n_channels x seg_len array containing the EEG data of the first and second segment in each pair
        y_val (np.array): N x 2 array containing the labels of the pairs
        criterion (nn.criterion): Loss function, most likely CrossEntropyLoss
        optimizer (nn.optimizer): Optimizer, most likely Adam
        batch_size (int): Batch size
        device (str): device, 'cuda' or 'cpu'
        lambda1 (float): L1 regularisation
        input_params (dict): Dictionary containing the input parameters for the training.
    Returns:
        tr_acc (float): Train accuracy
        tr_loss (float): Train loss
        val_acc (float): Validation accuracy
        val_loss (float): Validation loss
    """
    train_epoch(net, X_train, y_train, criterion, optimizer, batch_size, device, lambda1)
    tr_loss, tr_acc = memory_aware_eval(X_train, y_train, net, criterion, input_params)
    val_loss, val_acc = memory_aware_eval(X_val, y_val, net, criterion, input_params)
    return tr_acc, tr_loss, val_acc, val_loss

def memory_aware_eval(X_train, y_train, net, criterion, input_params):
    """ Evaluates network on training data, making sure memory capacity is not exceeded using the evaluate() function

    Args:
        net (EEGNet): Network
        X_train (np.array): N x 1 x 2*n_channels x seg_len array containing the EEG data of the first and second segment in each pair
        y_train (np.array): N x 2 array containing the labels of the pairs
        criterion (nn.criterion): Loss function, most likely CrossEntropyLoss
        input_params (dict): Dictionary containing the input parameters for the training.

    Returns:
        tr_loss_i (float): train loss for this epoch
        tr_acc_i (float): train accuracy for this epoch
    """

    
    if X_train.shape[0] > 10000: # If there is too much data
        tr_acc_i, tr_loss_i = [],[]
        for q in range(X_train.shape[0]//5000): # Divide it into chunks of 5000 samples
            X_curr = X_train[5000*q:5000*(q+1)]
            y_curr = y_train[5000*q:5000*(q+1)]
            tr_loss_ii, tr_acc_ii, _ = evaluate(net, criterion, X_train, y_train, input_params['device'])
            tr_acc_i.append(tr_acc_ii)
            tr_loss_i.append(tr_loss_ii)
        tr_acc_i = np.mean(tr_acc_i)
        tr_loss_i = np.mean(tr_loss_i)
    else:
        tr_loss_i, tr_acc_i, _ = evaluate(net, criterion, X_train, y_train, input_params['device']) #HERE
    return tr_loss_i, tr_acc_i

def evaluate(net, criterion, X, y, device):
    """ Evaluates the network on the data and returns the loss, accuracy, and predicted labels

    Args:
        net (EEGNet): Network
        criterion (nn.criterion): Loss function, most likely CrossEntropyLoss
        X (np.array): N x 1 x 2*n_channels x seg_len array containing the EEG data of the first and second segment in each pair
        y (np.array): N x 2 array containing the labels of the pairs
        device (str): 'cuda' or 'cpu'

    Returns:
        loss (float): Loss
        accuracy (float): Accuracy
        predicted (np.array): Predicted label
    """
    net.eval() # Evaluation mode
    predicted = []
    inputs = X.to(device)
    y = torch.tensor(list(map(to_label,y)))
    pred = net(inputs) # Forward pass
    predicted = pred.data.cpu().numpy() # Predicted labels in numpy form
    acc = accuracy_score(y, np.round(predicted)) # Accuracy    
    return criterion(pred,torch.FloatTensor(np.array(y)).to(device)).item(), acc, predicted

    
def train_epoch(net, X_train, y_train, criterion, optimizer, batch_size, device, lambda1):
    """ Runs 1 epoch of training in the network

    Args:
        net (EEGNet): Network
        X_train (np.array): N x 1 x 2*n_channels x seg_len array containing the EEG data of the first and second segment in each pair
        y_train (np.array): N x 2 array containing the labels of the pairs
        criterion (nn.criterion): Loss function, most likely CrossEntropyLoss
        optimizer (nn.optimizer): Optimizer, most likely Adam
        batch_size (int): batch size
        device (str): device, 'cuda' or 'cpu'
        lambda1 (float): L1 regularisation
    """
    y2D = torch.tensor(list(map(to_label,y_train))).to(device)
    net.train() # Train mode
    for i in range(len(X_train)//batch_size-1):
        s = i*batch_size
        e = (i+1)*batch_size
        inputs = X_train[s:e].to(device) # X_batch
        labels = y2D[s:e].to(device) # y_batch
        optimizer.zero_grad() # Resets optimizer
        #forward, backward, and optimize
        outputs = net(inputs) # Forward pass

        if lambda1 != 0:
            l1_reg = sum([torch.sum(torch.abs(p.cpu())) for p in net.parameters()]) # Absolute sum of weights
        else:
            l1_reg = 0
        loss = criterion(outputs, labels) + lambda1*l1_reg # Loss function
        loss.backward() # Backward pass
        optimizer.step() # Optimize
        del inputs, labels # Deletes the inputs and labels for GPU memory debugging


def toModelData(data, fold):
    """
    Converts data into a format that is compatible with the EEGNet model
    
    Args:
        data (dict): Dictionary containing the data.
        fold (tuple): Tuple containing the train and validation indices.
    
    Returns:
        X_train (torch.tensor): N_train x 1 x 2*n_channels x seg_len Tensor containing the training data.
        y_train (torch.tensor): N_train x 2 Tensor containing the training labels.
        X_val (torch.tensor): N_val x 1 x 2*n_channels x seg_len Tensor containing the validation data.
        y_val (torch.tensor): N_val x 2 Tensor containing the validation labels.
    """
    X_train = data['X_train'][fold[0]]
    y_train = data['y_train'][fold[0]]
    X_val = data['X_train'][fold[1]]
    y_val = data['y_train'][fold[1]]
    
    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).long()
    X_val = torch.from_numpy(X_val).float()
    y_val = torch.from_numpy(y_val).long()
        
    # Reshape data to (n_samples, 1, 2*n_channels, T) such that the channels of the two segments of a pair are concatenated in a way that is
    # the first half of the channels are the first segment and the second half are the second segment
    # Reshape data to have first half of channels from first segment and second half from second segment
    n_samples, n_channels, n_segments, T = X_train.shape
    X_train = torch.cat([X_train[:, :, 0, :], X_train[:, :, 1, :]], dim=1).unsqueeze(1)
    X_val = torch.cat([X_val[:, :, 0, :], X_val[:, :, 1, :]], dim=1).unsqueeze(1)
    
    return X_train, y_train, X_val, y_val


def kFoldSplit(data, n_folds, val_size, val_bal_ratio, dataset_verbose = True):
    """Creates k-fold split indices for cross validation with balanced classes
    
    Args:
        data (numpy.array): N x 3 array with N pairs, where each row contains [idx1, idx2, label]
        n_folds (int): Number of folds for cross-validation
        val_size (int): Target size of validation set per fold
        val_bal_ratio (float): Target ratio of class 1 ("different") in validation set (0.0-1.0)
        
    Returns:
        fold_indices (list): List of n_folds tuples, each containing (train_indices, val_indices) arrays
    """
    # Split data by class
    same_pairs = data[data[:,-1] == 0]  # Get pairs labeled as "same" (0)
    diff_pairs = data[data[:,-1] == 1]  # Get pairs labeled as "different" (1)
    
    # Create KFold splitter
    kf = KFold(n_splits=n_folds, shuffle=True)
    
    fold_indices = []
    
    # Generate splits for each class separately
    same_splits = list(kf.split(same_pairs))
    diff_splits = list(kf.split(diff_pairs))
    for fold in range(n_folds):
        # Get train/val indices for each class
        same_train_idx, same_val_idx = same_splits[fold]
        diff_train_idx, diff_val_idx = diff_splits[fold]
        
        # Calculate maximum possible samples for each class in validation set
        max_same_val = len(same_val_idx)
        max_diff_val = len(diff_val_idx)
        
        # Calculate the maximum validation set size while maintaining the ratio
        max_val_size_from_diff = int(max_diff_val / val_bal_ratio)
        max_val_size_from_same = int(max_same_val / (1 - val_bal_ratio))
        actual_val_size = min(val_size, max_val_size_from_diff, max_val_size_from_same)
        
        # Calculate samples per class to maintain ratio
        n_val_diff = int(actual_val_size * val_bal_ratio)
        n_val_same = actual_val_size - n_val_diff
        
        actual_ratio = n_val_diff / actual_val_size if actual_val_size > 0 else 0
        
        if dataset_verbose:
            print(f"Fold {fold+1}: Validation set size = {actual_val_size} (target: {val_size})")
            print(f"        Class ratio = {actual_ratio:.2f} (target: {val_bal_ratio:.2f})")
            print(f"        Same pairs = {n_val_same}, Different pairs = {n_val_diff}")
            
        # Sample validation set
        same_val_idx = np.random.choice(same_val_idx, size=n_val_same, replace=False)
        diff_val_idx = np.random.choice(diff_val_idx, size=n_val_diff, replace=False)
        
        # Get remaining indices for training
        same_train_idx = np.setdiff1d(same_train_idx, same_val_idx)
        diff_train_idx = np.setdiff1d(diff_train_idx, diff_val_idx)
        
        # Convert class-specific indices to indices in original data array
        val_indices = np.concatenate([
            np.where(data[:,-1] == 0)[0][same_val_idx],
            np.where(data[:,-1] == 1)[0][diff_val_idx]
        ])
        
        train_indices = np.concatenate([
            np.where(data[:,-1] == 0)[0][same_train_idx],
            np.where(data[:,-1] == 1)[0][diff_train_idx]
        ])
        
        fold_indices.append((train_indices, val_indices))
        
    return fold_indices

def loadData(load_mode, data_folder, noise, chan_to_exclude, channel_rescaling):
    """
    Loads the data from the specified folder and returns the training and testing data and labels.
    
    Args:
        load_mode (str): The mode of loading the data, either 'matlab' or 'numpy'.
        data_folder (str): The folder containing the data.
        noise (float): The amount of noise to add to the data.
        chan_to_exclude (list): The channels to exclude from the data.
        channel_rescaling (bool): Whether to rescale the channels.
    
    Returns:
        data (dict): A dictionary containing the training and testing data and labels.
    """
    if load_mode == 'matlab':
        X_train = loadmat(f'{data_folder}/data.mat')['X_train'] # (n_samples, n_channels, 2, T)
        y_train = loadmat(f'{data_folder}/data.mat')['y_train'] # (n_samples,)
        X_test = loadmat(f'{data_folder}/data.mat')['X_test'] # (n_samples, n_channels, 2, T)
        y_test = loadmat(f'{data_folder}/data.mat')['y_test'] # (n_samples,)
    elif load_mode == 'numpy':
        X_train = np.load(f'{data_folder}/X_train.npy') # (n_samples, n_channels, 2, T)
        y_train = np.load(f'{data_folder}/y_train.npy') # (n_samples,)
        X_test = np.load(f'{data_folder}/X_test.npy') # (n_samples, n_channels, 2, T)
        y_test = np.load(f'{data_folder}/y_test.npy') # (n_samples,)
        
    X_train = addNoise(X_train, noise)
    X_test = addNoise(X_test, noise)
    if channel_rescaling:
        X_train = X_train/abs(X_train).mean()
        X_test = X_test/abs(X_test).mean()
    if chan_to_exclude != []:
        X_train[:,chan_to_exclude,:,:] = 0
        X_test[:,chan_to_exclude,:,:] = 0
    data = {'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test}
    return data

def getFolder(out_folder, mode = "gen", save_verbose = True):
    """
    Generates a folder for the results to be saved in.
    
    Args:
        out_folder (str): The folder to save the results in.
        mode (str): The mode of the folder generation, either 'gen' or 'retrieve'.
        save_verbose (bool): Whether to print the folder path.
    
    Returns:
        strFolder (str): The path to the folder where the results will be saved.
    """
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    out_folder = out_folder + "/results"
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    strFolder = out_folder+"/run1"
    # The next lines are done in a try except block to avoid errors when doing multiple runs in parallel in the HPC
    # All it does is ensure that it is creating a new folder ending with runI, where I is the biggest number that is not already taken 
    file_counter = 1
    if mode == "gen":
        try:
            os.makedirs(strFolder)
        except:
            if os.path.exists(strFolder):
                strFolder = strFolder[:-1]+str(file_counter)
            while os.path.exists(strFolder):
                if file_counter == 1:
                    ind = 1
                else:
                    ind = 1+int(np.log10(file_counter-1))
                strFolder = strFolder[:-ind]+str(file_counter)
                file_counter+=1
            if not os.path.exists(strFolder):
                error_flag = True
                while error_flag == True:
                    try:
                        os.makedirs(strFolder)
                        error_flag = False
                    except Exception as error:
                        print(f"ERROR in folder creation:\n{error}")
                        strFolder = f"{strFolder[:-1]}{int(strFolder[-1])+1}"
            else:
                strFolder = f"{strFolder[:-1]}{int(strFolder[-1])+1}"
                os.makedirs(strFolder)
        if save_verbose: print(f"Processing...\nAll the results will be saved into {strFolder}.")
        return strFolder
    elif mode == "retrieve":
        while os.path.exists(strFolder):
            if file_counter == 1:
                ind = 1
            else:
                ind = 1+int(np.log10(file_counter-1))
            strFolder = strFolder[:-ind]+str(file_counter)
            file_counter+=1
        return file_counter-1 if file_counter > 1 else 1

def gen_path(strFolder, filename, file_type):
    """ Generates path for the file to be saved and makes sure that the file doesn't already exist

    Args:
        strFolder (str): Folder where the results are saved
        filename (str): Filename of the path to be generated
        file_type (str): file ending

    Returns:
        path (str) : The path of the file to be saved
    """
    strFile = filename
    file_counter = 1
    type_len = len(file_type)
    if os.path.isfile(strFolder+"/"+strFile):
        strFile = strFile[:-type_len]+str(file_counter)+file_type
    else:
        strFile = strFile+file_type

    while os.path.isfile(strFolder+"/"+strFile):
        ind = int(math.log10(file_counter))+type_len+1
        strFile = strFile[:-ind]+str(file_counter)+file_type
        file_counter+=1
    return strFolder+"/"+strFile

def write_results(path, data, folds, in_type = "listed", map_datatype = True, save_verbose = True):
    """ Writes the results as a csv file where each row is an instance and each column is a fold

    Args:
        path (str): Path of the csv file
        data (list): Data to be written to the csv file
        folds (int): Number of folds
        in_type (str, optional): "listed" or "unlisted", whether the input data is a list. Defaults to "listed".
        map_datatype (bool, optional): Whether to change numpy arrays to lists. Defaults to True.
        save_verbose (bool, optional): Whether to print the data shape. Defaults to True.
    """
    if map_datatype:
        if type(data[0]) == 'numpy.float64':
            data = list(map(float, data)) # Transforms numpy array into a list

    if save_verbose: print(f"{path}: data shape {np.shape(data)}") # Debugging
    with open(path, "w+") as f:
        writer = csv.writer(f,delimiter = ',', ) # Writes the data to a csv file with a delimiter of ','
        writer.writerow([f"f{i}" for i in range(1,folds)]) # Writes the header of the csv file as f1, f2, f3, ... (fold)
        if in_type == "listed":
            if len(np.shape(data)) > 1:
               writer.writerows(np.transpose(data))
            elif len(np.shape(data)) == 1:
                writer.writerows([data])
        elif in_type == "unlisted":
            writer.writerows([data])

def accPlot(tr_acc, val_acc, fold_nr=0,args = dict(), path = "", save_format = "png"):
    """ Plots the accuracy curves

    Args:
        tr_acc (list[float]): Train accuracy per epoch
        val_acc (list[float]): Validation accuracy per epoch
        fold_nr (int, optional): Fold Number. Defaults to 0.
        args (_type_, optional): Input arguments. Defaults to dict().
        path (str, optional): Path to save. Defaults to "".
        save_format (str, optional): Format to save the plot. Defaults to "png".
    """
    save_format = args['save_format']
    fig, ax = plt.subplots()
    ax.plot(tr_acc, color='g',label = 'Train Acc')
    ax.plot(val_acc, color='r',label = 'Validation Acc')
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Epoch")
    ax.legend()
    net_params = args['net_params']
    nps = net_params.copy()
    k = []
    for ki in range(1,len(nps.keys())+1):
        if ki%3 == 0:
            k.append("\n")
        else:
            k.append("")

    param_string2 = [f"{key[0:2-int(len(key)<2)]}: {nps[key]:.2e}{k[i]}" for i,key in enumerate(nps)]
    param_string = ""
    for s in param_string2:
        param_string += s+"| "
    if args != dict():
        ax.set_title(f"Net params: {param_string} \nN_train: {args['actual_sizes'][0]} | N_val: {args['actual_sizes'][1]}")
    fig.tight_layout()
    plt.savefig(f"{path}", format = save_format)
    plt.close()
    
def lossPlot(tr_loss, val_loss, fold_nr=0,args = dict(), path = "", save_format = "png"):
    """ Plots the loss curves

    Args:
        tr_loss (list[float]): Train loss per epoch
        val_loss (list[float]): Validation loss per epoch
        fold_nr (int, optional): Fold Number. Defaults to 0.
        args (_type_, optional): Input arguments. Defaults to dict().
        path (str, optional): Path to save. Defaults to "".
        save_format (str, optional): Format to save the plot. Defaults to "png".
    """
    save_format = args['save_format']
    fig, ax = plt.subplots()
    ax.plot(tr_loss, color='g', label='Training loss')
    ax.plot(val_loss, color='r', label='Validation loss')
    ax.set_ylabel("Loss")
    ax.set_xlabel("Epoch")
    net_params = args['net_params']
    net_p = net_params.copy()
    k = []

    for ki in range(1,len(net_p.keys())+1):
        if ki%3 == 0:
            k.append("\n")
        else:
            k.append("")
    
    param_string2 = [f"{key[0:2-int(len(key)<2)]}: {net_p[key]:.2e}{k[i]}" for i,key in enumerate(net_p)]
    param_string = ""
    for s in param_string2:
        param_string += s+"| "
    if args != dict():
        ax.set_title(f"Net params: {param_string} \nN_train: {args['actual_sizes'][0]} | N_val: {args['actual_sizes'][1]}")
    ax.legend()
    fig.tight_layout()
    if path == "":
        plt.savefig(f"./results/loss_plots_f{fold_nr}.{save_format}", format = save_format)
    else: 
        plt.savefig(f"{path}", format = save_format)
    plt.close()

def lossWrite(loss, fold_nr, path = ""):
    """ Writes the loss at each epoch into a text file

    Args:
        loss (list): Loss per epoch
        fold_nr (int): Fold number
        path (str, optional): Save path. Defaults to "".
    """
    loss = [f"{str(i)} " for i in loss]
    if path == "":
        f = open(f"./results/loss_{fold_nr}.txt","a+")
        f.writelines(loss)
        f.write("\n")
        f.close()
    else:
        f = open(path,"a+")
        f.writelines(loss)
        f.write("\n")
        f.close()
    

def writeMetrics(arr, mode, fold_nr, save_path = ""):
    """ Write evaluation metrics to text file

    Args:
        arr (list): List of [Accuracy, Recall, Precisiopn]
        mode (str): train/val/test
        fold_nr (int): Fold number
        save_path (str, optional): Save path. Defaults to "".
    """
    if save_path == "":
        f = open(f"./results/metrics_f{fold_nr}.txt","a+")
        f.write(f"\nEvaluation on {mode} data:")
        f.write(f"\nAccuracy: {arr[0]}")
        f.write(f"\nRecall: {arr[1]}")
        f.write(f"\nPrecision: {arr[2]}")
        f.close()
    else:
        f = open(f"{save_path}/metrics_f{fold_nr}.txt","a+")
        f.write(f"\nEvaluation on {mode} data:")
        f.write(f"\nAccuracy: {arr[0]}")
        f.write(f"\nRecall: {arr[1]}")
        f.write(f"\nPrecision: {arr[2]}")
        f.close()

def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    """ Save model checkpoint

    Args:
        model (EEGNet): Network
        optimizer (nn.optimizer): Optimizer
        filename (str, optional): File Name. Defaults to "my_checkpoint.pth.tar".
    """
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr, device):
    """ Load checkpoint

    Args:
        checkpoint_file (_type_): _description_
        model (EEGNet): Model that will have its weights updated
        optimizer (optimizer): Optimizer that will have its values updated
        lr (float): Learning rate to update optimizer
        device (str): 'cuda' or 'cpu'
    """
    checkpoint = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
        
def addNoise(X, noise):
    """Adding random Gaussian noise to data

    Args:
        X (numpy.array[float]): N x 2 x C x L array with N pairs, C channels and L samples 
        noise (float): % of standard deviation to add as noise

    Returns:
        X_out (numpy.array[float]): N x 2 x C x L array with added noise 
    """
    X_out = np.zeros_like(X)
    for i in range(X.shape[0]): #for each sample pair (or is each sample better????)        
        for segment in range(X.shape[2]): #for each channel
            std = np.std(X[i,:,segment,:])
            noise_vec = np.random.normal(0,std*noise, X.shape[-1]) #vec with len seglen for one segment
            X_out[i,:,segment,:] = noise_vec+X[i,:,segment,:]
    return X_out
    


def to_label(input):
    """ Transforms a boolean to a 2D label

    Args:
        input (int): 0 or 1

    Returns:
        output (list[int]): 1 x 2 label
    """
    if input == 0.:
        return [0., 1.]
    elif input == 1.:
        return [1., 0.]


def changeY(y):
    """ Transforms a 1D label list to a 2D label list

    Args:
        input (list[int]): List of 1D labels with N elements (0 or 1)

    Returns:
        output (list[list[int]]): N x 2 label
    """
    yb = []
    for j in y:
        if j[0] == 0.0:
            yb.append([1.0,0.0])
        elif j[0]== 1.0:
            yb.append([0.0,1.0])
        else: print("ERROR!!!!")
    return yb


        