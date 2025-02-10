from dbs_s_eegnet import EEGNet
import torch.nn as nn
import torch.optim as optim
import csv
import shutil
from glob import glob
import argparse
import numpy as np
import os
from warnings import warn
import pandas as pd
import torch
from train_utils import evaluate, to_label
from scipy.signal import butter, filtfilt, iirnotch
from copy import deepcopy
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import json
import matplotlib.pyplot as plt
from scipy.stats import sem
import pickle

def toModelData(X_test, y_test = []):
    """
    Args:
        X_test: numpy.array, shape (n_samples, n_channels, n_segments, T)
        y_test: numpy.array, shape (n_samples,)
    Returns:
        X_test: torch.Tensor, shape (n_samples, n_channels, n_segments, T)
        y_test: torch.Tensor, shape (n_samples,)
    Converts data into a format that is compatible with the EEGNet model
    """
    X_test = torch.from_numpy(X_test).float()
    if len(y_test) > 0:
        y_test = torch.from_numpy(y_test).long()

    n_samples, n_channels, n_segments, T = X_test.shape
    X_test = torch.cat([X_test[:, :, 0, :], X_test[:, :, 1, :]], dim=1).unsqueeze(1)
    if len(y_test) > 0:
        return X_test, y_test
    else:
        return X_test

def load_checkpoint(checkpoint_file, model, optimizer, lr, device):
    """ Load checkpoint from torch file

    Args:
        checkpoint_file (str): Path to the checkpoint file
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

def runComb(n_runs = 10, training_folder = "", folds_pr_folder = 10, runs = "", fold_select = False, test_verbose = False):
    """ This combines all runs into one folder for each training run, it is necessary for all the testing to work and should be run after all training is done.
    
    Args:
        n_runs (int, optional): Number of parallel runs that were run (1 result folder per run). Defaults to 10.
        folds_pr_folder (int, optional): Folds per folder, this can be put as list if variable number. Defaults to 10.
        training_folder (str, optional): Path to the training folder. Defaults to "".
        runs (list[str] or "", optional): Run numbers ([1,..,n_runs] if empty str is provided). Defaults to "".
        fold_select (bool, optional): Whether to skip folds that have low validation accuracy. Defaults to False.
        test_verbose (bool, optional): Whether to print verbose output. Defaults to False.
    """
    if runs =="": # If no runs are provided, use 1 to n_runs
        if test_verbose: print("Runs is empty, populating it...")
        runs = [str(i+1) for i in range(n_runs)]
    elif type(runs[0]) != str:  # If runs are provided as integers, convert to strings
        runs = list(map(str,runs))
        if test_verbose: print(f"Runs: {runs}")
    
    # Create output folder and create directories if it doesn't exist
    out_folder = f"fullrun"
    if fold_select:
        out_folder += "_fs"
    for q in runs:
        out_folder = f"{out_folder}_{q}"
    # print(out_folder)
    if not os.path.isdir(training_folder+"/results/"+out_folder):
        os.mkdir(training_folder+"/results/"+out_folder)

    new_f = 0 # Counter for new fold number
    skip_count = 0 # Counter for skipped folds
    for i in range(n_runs): # Loop through all runs
        if fold_select: # Skip folds with low validation accuracy
            # The way this skipping works is that it bins the accuracies of each fold and skips the lowest 25% of the folds
            # If any fold is below the threshold, but has an accuracy higher than 80% it is not skipped
            
            val_df = pd.read_csv(f"{training_folder}/results/run{runs[i]}/val_accs.csv", sep = ",")  # Load validation accuracy
            val_array = val_df.max(axis=0).to_numpy() # Keep only maximum of each epoch
            # print(np.unique(val_array,return_counts=True)[0])
            # print(np.unique(val_array,return_counts=True)[1])
            # Generate bins and threshold
            count_tally = 0
            count_tot = np.unique(val_array,return_counts=True)[1].sum()
            i_threshold = -1
            for i_count, count in enumerate(np.unique(val_array,return_counts=True)[1]):
                count_tally += count
                if count_tally > 0.25*count_tot:
                    if i_threshold==-1:
                        i_threshold = i_count
                    continue
            threshold = np.unique(val_array,return_counts=True)[0][i_threshold]
            in_flag = False
            while threshold > 0.8:
                if i_threshold==0: break
                i_threshold -= 1
                threshold = np.unique(val_array,return_counts=True)[0][i_threshold]
                in_flag = True
                
            if in_flag:
                threshold = np.unique(val_array,return_counts=True)[0][i_threshold]
                if test_verbose: print("/!\ The threshold was changed to be lower than 0.8 /!\ ")
            if test_verbose: print(f"Threshold used for run {runs[i]} is {threshold}\n(NOTE: THIS IS AN EXCLUSIVE > OPERATOR)")
        
        # Copy the best checkpoint from each fold to the new folder, and rename it to the new fold number
        if type(folds_pr_folder) == int: it_folds = folds_pr_folder
        else: it_folds = len(folds_pr_folder)
        for ff in range(1,1+it_folds): # Loop through all folds
            new_f = i*folds_pr_folder+ff-skip_count
            if fold_select and val_array[ff-1] <= threshold: 
                if test_verbose: print(f"Skipping fold {new_f+skip_count}...")
                skip_count += 1
                continue
            # Find the best checkpoint for each fold
            checkpoint_str = glob(training_folder+f"/results/run{runs[i]}/checkpoint_f{ff}_best_e*.tar")[0]
            epoch_nr = checkpoint_str.split('e')[-1].split('.')[0]
            # Copy the best checkpoint to the new folder
            shutil.copy(checkpoint_str,f"{training_folder}/results/{out_folder}/checkpoint_f{new_f}_e{epoch_nr}.tar")
    if test_verbose: print(f"{skip_count} folds were skipped")
    
def full_test():
    pass

def test(training_folder,sample_freq, seg_dur, seg_len,
         net_params, device, n_channels, fold_select,
         runs, mode, data, res, low_th, high_th, bw,
         ORDER, test_verbose, save_verbose, save_format):
    """
    This function runs the testing of the model on the test data.
    
    Args:
        training_folder (str): Path to the training folder.
        sample_freq (int): Sampling frequency (Hz).
        seg_dur (int): Segment duration (s).
        seg_len (int): Segment length (samples).
        net_params (dict): Parameters for the EEGNet model.
        device (str): 'cuda' or 'cpu'.
        n_channels (int): Number of channels.
        fold_select (bool): Whether to skip folds that have low validation accuracy.
        runs (list[str]): Run numbers.
        mode (str): 'BS', 'BP', 'LP', or 'HP' (Bandstop, Bandpass, Lowpass, Highpass).
        data (dict): Dictionary containing the test data.
        res (int): Resolution for the moving filter.
        low_th (int): Low threshold for the moving filter.
        high_th (int): High threshold for the moving filter.
        bw (int): Bandwidth for the moving filter.
        ORDER (int): Order for the moving filter.
        test_verbose (bool): Whether to print verbose output in the whole function.
        save_verbose (bool): Whether to print verbose output in terms of what is saved.
        save_format (str): Format for saving the results (e.g. 'svg', 'png').
    """
    loss_tot = []
    acc_tot = []
    y_preds = []
    freqs = []
    accs = []
    test_labels = []
    ffolder = training_folder.split('/')[-1]
    test_folder = training_folder
    net = EEGNet(num_classes=2, num_channels = n_channels, seg_len=seg_len, device = device, net_params=net_params).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters())
    # Folder name for the results
    out_folder = f"fullrun"
    if fold_select: out_folder += "_fs"
    for q in runs:
        out_folder = f"{out_folder}_{q}"
    # Load the test indices
    if mode == 'BS': # Bandstop mode
        # Initialise the dictionary for the heatmap results
        lo_chans_acc = {'~Allt':[], '~Alla':[], '~Allb':[], '~Allgl': [], '~Allgh':[], '~All':[]} # All theta, alpha, beta, gamma low, gamma high, all
        
    X_test, y_test = data['X_test'], data['y_test']
    X_test, y_test = toModelData(X_test, y_test)
    n_folds = max(list(map(lambda x: int(x.split("_")[-2].split("f")[-1]),glob(f"{training_folder}/results/{out_folder}/checkpoint_f*"))))
    heat_results = []
    for i in range(n_folds):
        if test_verbose: print(f"Fold {i+1} of {n_folds}")
        loss_i, acc_i = [],[]
        m_i = 0
        for filename in os.listdir(f"{training_folder}/results/{out_folder}/"): # Iterate through the folds
            if filename.startswith(f"checkpoint_f{i+1}_"):
                if test_verbose: print(f"Loading checkpoint {i+1} from {training_folder}/results/{out_folder}/{filename}")
                load_checkpoint(f"{training_folder}/results/{out_folder}/{filename}", net, optimizer, net_params['lr'], device) # Load the model
                _,ac,preds = evaluate(net, criterion, X_test, y_test, device)
                accs.append(ac)
                y_preds.append(preds.tolist())
                test_labels.append(list(map(to_label,y_test)))
                if mode == 'BS': #only if mode is BS (so we dont repeat it uneccasarily for all modes)
                    lo_chans_acc = acc_w_wo_chns(lo_chans_acc,net,criterion,X_test,y_test,device)
        for threshi in range(res*(high_th-low_th)+1):#change for better resolution
            thresh = threshi/res + low_th #change for better resolution
            Xfil = filterin(mode, thresh, X_test, bw, sample_freq, ORDER)            
            loss,acc,_ = evaluate(net, criterion, Xfil, y_test, device)
            loss_i.append(loss)
            acc_i.append(acc)
        loss_tot.append(loss_i)
        acc_tot.append(acc_i)
        # For BS mode, save the band-filtering results in a dictionary for future plotting
        if mode == 'BS':
            bands = [key for key in lo_chans_acc][-5:]
            band_acc = -5*np.ones(len(bands)) # -5 is a placeholder for debugging
            if i == 0: # If it's the first fold, initialise the dictionaries
                max_band_count = {} # Dictionary to count the number of times a band was the best (lowest accuracy when filtered)
                band_sort = {} # Dictionary to rank the band with respect to the others in terms of accuracy
                band_accs = {} # Dictionary to save the accuracies
            for q, key in enumerate(bands):
                # TO CHECK FIX: THIS WAS CHANGED FROM  band_acc[q] = lo_chans_acc[key][-1] TO band_acc[q] = np.mean(lo_chans_acc[key])
                # IT IS YET TO BE TESTED
                band_acc[q] = np.mean(lo_chans_acc[key])
                if i ==0: # Initialise the elements
                    max_band_count[key] = 0
                    band_sort[key] = []
                    band_accs[key] = []
            order = np.argsort(band_acc) # Sort the bands according to their accuracy
            for q, o in enumerate(order):
                band_sort[bands[o]].append(q) # Save the rank of the band (0 is the worst, 5 is the best)
            max_band_count[bands[order[0]]] += 1 #
            band_accs[bands[order[0]]].append(ac)
    full_results = np.array(accs)
    if mode == 'BS':
        heat_results = lo_chans_acc
    if not os.path.exists(f"{test_folder}/results/{out_folder}"):
        os.mkdir(f"{test_folder}/results/{out_folder}")
    if not os.path.exists(f"{test_folder}/plots"):
        os.mkdir(f"{test_folder}/plots")
    if not os.path.exists(f"{test_folder}/plots/{out_folder}"):
        os.mkdir(f"{test_folder}/plots/{out_folder}")
    if mode == 'BS': # For bandstop, save the previous filtering rank results
        json_max = json.dumps(max_band_count)
        json_sort = json.dumps(band_sort)
        json_accs = json.dumps(band_accs)
        with open(f"{test_folder}/results/{out_folder}/band_histogram.json","w") as f:
            f.write(json_max)
            f.write("\n")
            f.write(json_sort)
            f.write("\n")
            f.write(json_accs)
    if save_verbose: print(f"Saving into {test_folder}/results/{out_folder}/full_results")
    np.save(f"{test_folder}/results/{out_folder}/full_results", full_results)
    if mode == 'BS':
        if save_verbose: print(f"Saving into {test_folder}/results/{out_folder}/heat_results")
        with open(f"{test_folder}/results/{out_folder}/heat_results", "wb") as fp:
            pickle.dump(heat_results, fp)
    # Generate the ground truths and predictions
    try: # If the data is homogenous (i.e. equal number of folds per run and samples per fold)
        ygts = np.array(np.array(test_labels).argmax(axis=-1),dtype=float).flatten()
        yprs = np.array(np.array(y_preds).argmax(axis=-1),dtype=float).flatten()
    except: # Otherwise, flatten the data manually
        ygts = np.zeros(0)
        for v_l in test_labels:
            ygts = np.concatenate([ygts, v_l.argmax(axis=-1)]) # Transform to 1D label
        yprs = np.zeros(0)
        for y_p in y_preds:
            yprs = np.concatenate([yprs, np.array(y_p).argmax(axis=-1)]) # Transform to 1D prediction

    # Calculate accuracy using the ground truths and predictions

    acc1 = np.mean(accs)
    acc2 = accuracy_score(ygts, yprs)
    if np.round(acc1,5) != np.round(acc2,5): # Check if the accuracies match 
        raise("ERROR: the accuracies do not match")
    plotConfMat(ygts, yprs, ['Same','Different'], "test","ALL",test_folder,out_folder = f"{out_folder}/", save_format = save_format) # Plot confusion matrix   
    for threshj in range(res*(high_th-low_th)+1): # Create frequency x-axis
        freqs.append(threshj/res + low_th)

    plotAvg(test_folder,mode,acc_tot,freqs, out_folder= out_folder, full_acc = acc, freq_lims = [low_th, high_th], save_verbose=save_verbose, save_format = save_format) # Perform moving filter ablation plots
    acc_tot = np.array(acc_tot) # Transform the list to a numpy array and save it
    np.save(f"{test_folder}/results/{out_folder}/filter_array_{mode}", acc_tot)
    
    if mode == 'BS':
        getBandDrop(test_folder, out_folder, test_verbose, save_verbose)

def plotAvg(folder,mode,acc_tot, freqs, out_folder,
            full_acc = 0.0, save_format = "svg", comment = "",
            acc_sem = [], stim = 0, freq_lims = [4,91], save_verbose = True):
    """ Plots moving filter accuracy for each frequency. This function can also be used to plot the accuracy of different parameters for each patient and vice-versa
    This is done through the inter_param_filters and inter_patient_filters functions in this script

    Args:
        folder (str): Path to the folder where the results are stored
        mode (str): BS, BP, LP or HP (Bandstop, Bandpass, Lowpass or Highpass)
        acc_tot (list/np.array): Contains the accuracy for each fold and each frequency
        freqs (list): Contains the frequencies
        out_folder (str): Output folder
        full_acc (float, optional): Overall accuracy. Defaults to 0.
        save_format (str, optional): "svg" or "png". Defaults to "svg".
        comment (str, optional): Comment to add to end of folder. Defaults to "".
        acc_sem (list, optional): Standard error from the mean for each frequency, if not inputted, it is calculated here. Defaults to [].
        stim (int, optional): Stimulation frequency if first subharmonic is to be plotted. Defaults to 0.
        freq_lims (list, optional): Frequency limits for the plot. Defaults to [4,91].
        save_verbose (bool, optional): Whether to print verbose output in terms of what is saved. Defaults to True.
    """
    # If no accuracy standard error is given, calculate it
    if len(acc_sem) == 0:
        acc_sem = sem(np.array(acc_tot),axis=0)
    # Create folders if they don't exist
    if not os.path.isdir(f"{folder}/plots") and out_folder != "": os.makedirs(f"{folder}/plots")
    if not os.path.isdir(f"{folder}/plots/{out_folder}") and out_folder != "": os.makedirs(f"{folder}/plots/{out_folder}")

    modes = {'BS': 'Bandstop', 'BP': 'Bandpass', 'HP': 'Highpass', 'LP': 'Lowpass'}

    fig, ax2 = plt.subplots()
    acc_avg = np.mean(np.array(acc_tot), axis=0) # Average accuracy across folds for each filtering point
    ax2.plot(freqs, acc_avg, color='mediumblue', linewidth=1.5, label = 'Accuracy') # Plot the average accuracy
    ax2.fill_between(freqs,acc_avg-acc_sem,acc_avg+acc_sem,alpha=0.3,edgecolor='black',facecolor='mediumblue') # Fill between the standard error
    if stim != 0: # If stimulation frequency is provided, plot its first subharmonic vertically
        ax2.vlines(stim/2, 0,1,color='mediumblue')
    if out_folder != "" and type(acc_tot) != list: # Save results in text file
        f = open(f"{folder}/plots/{out_folder}/plot_avg_values.txt", 'w+')
        f.writelines(['Mode: ', str(mode), '\n', 
                    'Avg acc: ', '\n', str(acc_avg), '\n', 
                    'Freqs: ', '\n', str(freqs), '\n'])
        f.close()
    plt.suptitle(f"Accuracy without filtering: {full_acc}")

    ax2.set_xlabel(f"{modes[mode.split('_')[0]]} frequency (Hz)", fontsize=14)
    # ax2.set_ylabel("Accuracy", fontsize=14)
    ax2.set_ylim(0.45, 0.95)
    ax2.hlines(0.5,0,100,colors='crimson',label = 'Chance level')
    ax2.set_xlim(freq_lims[0],freq_lims[1])
    ax2.set_yticks([0.5, 0.7, 0.9])
    # The next 5 lines are to be commented when plots are being generated in psd_analysis.py
    ax2.set_xticks([4, 8, 13, 30, 60, 90]) # Frequency band boundaries
    ax2.tick_params(axis='both', labelsize=14) #
    ax2.legend(bbox_to_anchor=(0.3,0.9), fontsize='large') # 
    ax2.spines[['right', 'top']].set_visible(False) #
    if out_folder != "":
        if save_verbose: print(f"{folder}/plots/{out_folder}/loss_acc_{mode}_{freqs[0]}to{freqs[-1]}{comment}.{save_format}")
        fig.savefig(f"{folder}/plots/{out_folder}/loss_acc_{mode}_{freqs[0]}to{freqs[-1]}{comment}.{save_format}", format = save_format,bbox_inches='tight',pad_inches=0)
    else: 
        if save_verbose: print(f"{folder}/loss_acc_{mode}_{freqs[0]}to{freqs[-1]}{comment}.{save_format}")
        # # fig.savefig(f"{folder}/loss_acc_{mode}_{freqs[0]}to{freqs[-1]}{comment}.{save_format}", format = save_format,bbox_inches='tight',pad_inches=0)
    plt.close(fig)


def plotConfMat(y, y_pred, class_labels, mode,fold_nr, folder, save_path = "", out_folder = "", save_format = "svg", comment="", fold_select = False):
    """ Plot confusion matrix and saves it

    Args:
        y (np.array[int]): True labels
        y_pred (np.array[int]): Predicted labels
        class_labels (list[str]): Class name for plots
        mode (str): train/val/test
        fold_nr (int): Fold number
        folder (str): Mother path where the plots are saved when using this function post-hoc
        save_path (str, optional): Save Path. Defaults to "".
        out_folder (str, optional): Folder of the runs, if within a run leave empty. Defaults to "".
        save_format (str, optional): Picture format. Defaults to "svg".
        comment (str, optional): Appended to end of picture name when saving. Defaults to "".
        fold_select (bool, optional): Whether folds are being selected. Defaults to False.
    """
    cm = confusion_matrix(y, y_pred) # Compute confusion matrix
    acc = accuracy_score(y, y_pred) # Compute accuracy
    print(f"Accuracy: {100*acc:.2f}%")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = class_labels) # Plot confusion matrix
    disp.plot()
    plt.title(f"Overal accuracy: {100*acc:.2f}%")
    if save_path!="":
        if fold_select: fs = "_fs"
        else: fs = ""
        plt.savefig(f"{save_path}/{mode}confmat_f{fold_nr}{comment}{fs}")
    else:
        if not os.path.exists(f"{folder}/plots/{out_folder}"):
            os.makedirs(f"{folder}/plots/{out_folder}")
        plt.savefig(f"{folder}/plots/{out_folder}/{mode}confmat_f{fold_nr}.{save_format}", format = save_format)
    plt.close()
    
    

def acc_w_wo_chns(lo_chans_acc, net,criterion,X_te,y_te,device):
    """ Calculates the accuracy of the model with each channel and frequency band filtered out

    Args:
        lo_chans_acc (dict[list]): Contains 1 or 0 (accuracy) for each sample after filtering
        net (EEGNet): EEGNet Model
        criterion (): Loss function
        X_te (numpy.array): Test EEG data
        y_te (numpy.array): Test labels
        device (str): "cuda" or "cpu"

    Returns:
        lo_chans_acc (dict[list]): Populated dict with this fold's results
    """
    # Different conditions, different channels and frequency bands and their combinations
   
    bands = [[4,8],[8,12],[13,30],[30,45],[55,91]] # Theta, Alpha, Beta, Gamma low, Gamma high
    
    # Iterate through frequency bands and evaluate the model with the band filtered out for all channels
    allbands = ['t','a','b','gl','gh']
    for bandi in range(len(bands)):
        allband = '~All'+allbands[bandi]
        thresh = (bands[bandi][0]+bands[bandi][1])/2
        bw = bands[bandi][1]-bands[bandi][0]
        X_tex = deepcopy(X_te)
        
        f,e = iirnotch(thresh, thresh/bw, fs=300)
        sig = filtfilt(f,e,X_tex.numpy())
        sig = torch.from_numpy(sig.copy()).float()
        _,chac,_ = evaluate(net,criterion,deepcopy(sig), y_te, device)
        lo_chans_acc[allband].append(chac)
    
    # Make all the data 0 and save it under the ~All key, for heatmap plotting purposes
    X_tex = deepcopy(X_te)
    X_tex[:,:,:] = 0.0
    _,chac,_=evaluate(net,criterion,X_tex,y_te,device)
    lo_chans_acc['~All'].append(chac)
    
    return lo_chans_acc

def filterin(mode,thresh,X, bw, fs=300, ORDER=5):
    """ Filter the input signal using a series of filters

    Args:
        mode (str): Bandpass, bandstop, lowpass or highpass
        thresh (int): Frequency threshold (Hz)
        X (numpy.array): EEG data
        bw (int): Bandwidth (Hz)
        fs (int, optional): Sampling frequency (Hz). Defaults to 300.

    Returns:
        _type_: _description_
    """
    Xfil = np.empty(X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            sig = X[i,j,:]
            if mode == 'LP':
                b, a = butter(ORDER, thresh, fs=fs, btype='lowpass') #LP
                sig = filtfilt(b,a,sig)
            if mode == 'HP':
                d, c = butter(ORDER, thresh, fs=fs, btype='highpass') #HP
                sig = filtfilt(d,c,sig)
            if mode == 'BP':
                b, a = butter(ORDER, [max(thresh-bw/2-1,1), thresh+bw/2+1], fs=fs, btype='bandpass')
                sig = filtfilt(b,a,sig)
            if mode == 'BS':
                f, e = iirnotch(thresh,thresh/bw, fs=fs) #Notch
                sig = filtfilt(f,e,sig)
            Xfil[i,j,:] = sig
    return torch.from_numpy(Xfil).float()

def getBandDrop(test_folder, out_folder, test_verbose, save_verbose):
    band_dict = {'gh': 'Gamma-High', 'gl': 'Gamma-Low', 'b': 'Beta', 'a': 'Alpha', 't': 'Theta'}
    params = ['lAmp', 'rAmp', 'lContact', 'rContact']

    per_band_res = {key: [] for key in band_dict}
    # accuracy list for each fold
    acc_arr = np.load(f"{test_folder}/results/{out_folder}/full_results.npy")
    raw_acc = np.mean(acc_arr)

    # lo_chans_acc
    with open(f"{test_folder}/results/{out_folder}/heat_results","rb") as heat_f:
        heat_dict = pickle.load(heat_f)
    
    for band in band_dict:
        per_band_res[band].append(np.mean(heat_dict[f'~All{band}']))
        if test_verbose: print(f"{band_dict[band]}: {np.mean(per_band_res[band])}")

    # Create results dictionary with raw accuracy
    results_dict = {'Raw': raw_acc}
    
    # Add band results
    for band, name in band_dict.items():
        results_dict[name] = np.mean(per_band_res[band])

    # Save to CSV
    df = pd.DataFrame([results_dict])  # Create single row dataframe
    csv_path = f"{test_folder}/results/{out_folder}/band_drop_results.csv"
    df.to_csv(csv_path, index=False)
    if save_verbose: print(f"Results saved to {csv_path}")
    