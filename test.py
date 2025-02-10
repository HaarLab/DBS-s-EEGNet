from test_utils import full_test, test, runComb
import json
from dbs_s_eegnet import EEGNet
import numpy as np
from glob import glob

def myTest(data, training_folder,modes, runs, test_params, test_verbose = False, save_verbose = True, save_format = "png"):
    """
    This function runs the testing of the model on the test data.
    Args:
        data (dict): Dictionary containing the test data.
        training_folder (str): Path to the training folder.
        modes (list): List of modes to test (e.g. ['BS','BP','HP','LP']).
        runs (list): List of run numbers.
        test_params (dict): Dictionary containing all the input parameters for the testing.
        test_verbose (bool): Whether to print verbose output.
        save_verbose (bool): Whether to print verbose output in terms of what is saved.
        save_format (str): Format for saving the results (e.g. 'svg', 'png').
    """
    if test_verbose: print(f"Loading params from {training_folder}/results/run{runs[0]}/params.json")
    with open(f"{training_folder}/results/run{runs[0]}/params.json","r") as f:
        input_params = json.load(f)
    
    
    device = input_params['device']
    n_channels = input_params['n_channels']
    seg_len = input_params['seg_len']
    seg_dur = input_params['seg_dur']
    sample_freq = input_params['sample_freq']
    net_params = input_params['net_params']

    fold_select = test_params['fold_select']
    run_comb = test_params['run_comb']
    defaults = {'low_th':4,'high_th':91,'bw':4,'res':1,'fold_select':False, 'ORDER': 5}
    for key in ['low_th','high_th','bw','res','fold_select', 'ORDER']:
        if key not in test_params or test_params[key] == None:
            test_params[key] = defaults[key]
    low_th = test_params['low_th']
    high_th = test_params['high_th']
    bw = test_params['bw']
    res = test_params['res']
    ORDER = test_params['ORDER']
    folds_pr_folder = [max(list(map(lambda x: int(x.split("_")[-3].split("f")[-1]),glob(f"{training_folder}/results/run{i}/checkpoint_f*_best*")))) for i in runs]
    if np.unique(folds_pr_folder).size == 1:
        folds_pr_folder = folds_pr_folder[0]
    if run_comb:
        runComb(n_runs = len(runs), training_folder = training_folder, folds_pr_folder = folds_pr_folder, runs = runs, fold_select = fold_select, test_verbose=test_verbose)

    for mode in modes: # Modes can be 'BS','BP','HP','LP' and represent the different filtering operations
        print(f"Running test with mode: {mode}")
        test(training_folder, sample_freq, seg_dur, seg_len,
                net_params, device, n_channels, fold_select,
                runs, mode, data, res, low_th, high_th, bw,
                ORDER, test_verbose, save_verbose, save_format) #mode, low_th, high_th, width, RES = resolution, folder (and FILE = val index file)
    