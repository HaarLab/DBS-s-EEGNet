import torch
import ray
import ray.tune as tune
import ray.train as r_train
from ray.tune.schedulers import ASHAScheduler
from tune_utils import trainable_multifold, analyseTune
import os

def myTune(data, input_params, out_folder):
    """
    This function runs the tuning of the EEGNet model.
    
    Args:
        data (dict): Dictionary containing the training and testing data and labels.
        input_params (dict): Dictionary containing the input parameters for the training.
        out_folder (str): The folder to save the results in.
    """
    if input_params['mode'] != "analyse_tune":
        if not os.path.exists(f"{out_folder}/ray_results"):
            os.makedirs(f"{out_folder}/ray_results")
        if input_params['num_cpus'] == None:
            # find number of cpus
            input_params['num_cpus'] = os.cpu_count()-4 if os.cpu_count() > 4 else 1
        if torch.cuda.is_available(): num_gpus = 1
        else: num_gpus = 0
        # Initialise Ray, this is necessary for it to work in a cluster
        if not ray.is_initialized():
            ray.init(num_cpus = input_params['num_cpus'], num_gpus = num_gpus)
        trainable_with_resources = tune.with_resources(trainable_multifold, {"cpu": input_params['num_cpus'], "gpu": num_gpus}) # This is the function that performs the training, we are initialising it with the available resources
        tune_params = input_params['tune_params']
        # The scheduler below is the ASHA scheduler, which allows early stopping of bad trials
        scheduler = ASHAScheduler(
                metric=tune_params['metric'], # Reported metric to be used
                mode=tune_params['mode'], # Maximise or minimise the metric
                max_t=input_params['epochs'], # Maximum number of epochs
                grace_period=tune_params['grace_period'], # Minimum number of epochs
                reduction_factor=tune_params['reduction_factor']) # Harshness of the early stopping
        
        storage_path = f"{out_folder}/ray_results"
        if not os.path.isabs(storage_path):
            storage_path = os.path.abspath(storage_path)
            print(f"Converting to absolute path: {storage_path}")
        
        # Create ray tuner and train it
        tuner = tune.Tuner(
            tune.with_parameters(trainable_with_resources, data = data, input_params = input_params), # Trainable function and its inputs
            param_space = input_params['config'], # Parameter space to search
            tune_config = tune.TuneConfig(num_samples = tune_params['num_samples'], scheduler = scheduler), # Number of samples and scheduler
            run_config = r_train.RunConfig(storage_path=storage_path, name=f"{input_params['tune_folder']}") # Save path and name
        )
        results = tuner.fit() # Train the tuner
        best_trial = results.get_best_result("val_loss", "min", "all") # Get the best trial
        if input_params['tune_verbose']: print(f"Best trial: {best_trial}")
        if input_params['tune_verbose']: print(f"Best trial config: {best_trial.config}")
    analyseTune(out_folder, input_params['tune_params'], input_params['tune_folder'], input_params['tune_verbose'], input_params['save_verbose'], input_params['save_format'])