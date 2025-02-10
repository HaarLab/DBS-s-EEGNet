from ray import tune
from ray.tune import ExperimentAnalysis
import argparse
from copy import deepcopy
import pickle, json
from torch.cuda import is_available
from matplotlib import pyplot as plt
from pandas import DataFrame as df
from tqdm import tqdm
import numpy as np
from train import myTrain
from test import myTest
from tune import myTune
import os
from train_utils import getFolder, loadData
from tqdm import trange


if is_available(): device = 'cuda'
else: device = 'cpu'

study_name = 'example_study11' # Name of the folder in which all outputs will be saved
# Each study can have multiple runs within it
folder = 'DBS-s-EEGNet' # Relative or absolute path to the repository folder
# Make the path absolute to avoid errors
if not os.path.isabs(folder):
    folder = os.path.abspath(folder)
    print(f"Converting to absolute path: {folder}")

out_folder = f'{folder}/outputs/{study_name}' # Folder where your outputs will be saved
data_folder = f"{folder}/your_data" # Folder where your data is present

mode = 'full_run' # mode, either 'normal', 'full_run', 'tune', 'data_estimation, 'param_xplore', or 'experiment'
load_mode = 'matlab' # 'matlab', 'numpy'
noise = 0.01 # noise level to add to the data
chan_to_exclude = [] # channels to exclude from the dataframe, STILL NEED TO IMPROVE THE IMPLEMENTATION OF THIS FEATURE
channel_rescaling = True # whether to rescale the channels or not. Default is True.

# This loads the data into numpy arrays and noise is added here and channels are excluded here*
data = loadData(load_mode, data_folder, noise, chan_to_exclude, channel_rescaling)

runs = [] # Which runs you want to analyse for testing | Can leave empty for full_run mode
default_nruns = 5 # Default number of runs to run in full_run
epochs = 450 # number of epochs per fold
sample_freq = 300 # sampling frequency in Hz
seg_dur = data['X_train'].shape[3] / sample_freq # duration of each segment in seconds
seg_len = data['X_train'].shape[3] # length of each segment in samples
n_channels = data['X_train'].shape[1] # number of channels in the EEG data
save_model = True # whether to save the model at checkpoints or not, default is True
checkpoints = [2,3] # At which epochs to save checkpoints (if left empty, just the best one is saved)
batch_size = 64 # batch size for training
class_labels = ['Same', 'Different'] # labels for the classes
tune_multifold = True # Whether to tune the hyperparameters across multiple folds or not
epoch_verbose = False # whether to print epoch progression
tune_verbose = False # Whether to print which parameters are/are not being tuned in tuning
save_verbose = False # Whether to print saved outputs to the terminal
test_verbose = False # Whether to print testing outputs to the terminal
dataset_verbose = False # whether to print the validation set size and balancing for each fold
metric = 'loss' # metric to use for saving the best model, either 'loss' or 'acc'
run_comb = True # whether to run the combination of runs or not (see test scripts)
fold_select = False # whether to select the folds or not with val accuracy thresholding
num_cpus = None # number of cpus to use for tuning, if None, the number of cpus will be automatically found
use_tune = True # Whether to use the parameters from the hyperparameter tuning or not.
tune_folder = "" # The folder in which tuning results are saved, if left empty, it will default to the study name
test_modes = ['BS','BP','HP','LP'] # Test modes, 'BS', 'BP', 'HP', 'LP' are Bandstop, Bandpass, Highpass, and Lowpass filtering respectively
save_format = "png" # Choose format for saving figures, (svg, png, jpg etc...)
n_folds = 10
val_bal_ratio = 0.5 # ratio of balancing the validation data (0.5 = 1:1 ratio of classes)
val_size = 20 # size of the validation set in samples per fold

base_params = {# Model parameters
                'L1': 0, # L1 penalty
               'fc_dropout': 0.45, # Dropout of fully-connected layers
               'conv_dropout': 0.45, # Dropout of convolutional filters
               'D': 2, # Number of spatial filters
               'F1': 8, # Number of temporal filters
               'L2': 1e-5, # L2 penalty
               'lr': 1e-4, # learning rate
                }  
config = { # Parameter space in which to sample for tuning
    "lr": tune.loguniform(5e-5, 5e-4), # learning rate
    "L2": tune.loguniform(1e-7,1e-4) # L2 penalty
    }
tune_params={ # tune_params for the Ray tuning process
    'num_samples': 4, # number of starting sample for tuning
    'metric': 'val_loss', # metric to use for tuning
    'mode': 'min', # 'min' to minimise metric, 'max' to maximise metric
    'grace_period': 5, # grace period for early stopping
    'reduction_factor': 2 # reduction factor for ASHAScheduler
}
# Defaults tune_folder to study_name
if tune_folder == "":
    tune_folder = study_name
# Defaults runs
if runs == []:
    n_runs = default_nruns
    previous_max_run = int(getFolder(out_folder, "retrieve"))
    runs = [i for i in range(previous_max_run, previous_max_run+n_runs)]
else:
    n_runs = len(runs)
# Validation dataset variables (set them yourself)
if mode == 'train' or mode == 'tune': # One fold
    n_folds = 1
else:
    n_folds = n_folds # number of folds for cross-validation

# the dataset_verbose variable is useful for debugging and safety checking.

input_params = {'epochs': epochs,
                'batch_size': batch_size,
                'class_labels': class_labels,
                'val_bal_ratio': val_bal_ratio,
                'val_size': val_size,
                'n_folds': n_folds,
                'save_model': save_model,
                'chan_to_exclude': chan_to_exclude,
                'channel_rescaling': channel_rescaling,
                'epoch_verbose': epoch_verbose,
                'tune_verbose': tune_verbose,
                'dataset_verbose': dataset_verbose,
                'save_verbose': save_verbose,
                'test_verbose': test_verbose,
                'metric': metric,
                'checkpoints': checkpoints,
                'tune_multifold': tune_multifold,
                'seg_dur': seg_dur,
                'sample_freq': sample_freq,
                'n_channels': n_channels,
                'net_params': base_params,
                'device': device,
                'noise': noise,
                'config': config,
                'tune_params': tune_params,
                'study_name': study_name,
                'mode': mode,
                'seg_dur': seg_dur,
                'seg_len': seg_len,
                'num_cpus': num_cpus,
                'tune_folder': tune_folder,
                'use_tune': use_tune,
                'save_format': save_format}

if mode == "train" or mode == "cross_val":
    myTrain(data, input_params, out_folder)
elif mode == "tune" or mode == "cross_tune" or mode == "analyse_tune":
    myTune(data, input_params, out_folder)
elif mode == "test":
        myTest(data, out_folder, test_modes, runs,
           {'fold_select': fold_select, 'run_comb': run_comb},
           test_verbose, save_verbose, save_format)
elif mode == "full_run":
    myTune(data, input_params, out_folder)
    for j in trange(n_runs, desc = 'Train Runs'):
        myTrain(data, input_params, out_folder)
    myTest(data, out_folder, test_modes, runs,
           {'fold_select': fold_select, 'run_comb': run_comb},
           test_verbose, save_verbose, save_format)

print("Done!")

