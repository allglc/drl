import json
import logging
import time
import random
from pathlib import Path


class Params():
    """Class that loads hyperparameters from a json file.
    From: https://github.com/cs230-stanford/cs230-code-examples/blob/master/tensorflow/nlp/model/utils.py
    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path='params.json'):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__ = params

    def save(self, json_path='params.json'):
        """Saves parameters to json file"""
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)
            
            
def set_logger(log_path):
    """Sets the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    From: https://github.com/cs230-stanford/cs230-code-examples/blob/master/tensorflow/nlp/model/utils.py
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)
    
    return logger


def create_experiment_folder(path_results, env_name, agent):
    ''' 
    Creates a folder to save experiment results and parameters 
    Args:
        path_results: where to save
    '''
    
    # wait a random time to avoid problems with parallel calls
    time.sleep(5*random.random())
    
    timestamp = time.strftime('%Y-%m-%d_%H%M', time.localtime())
    path_results_exp = Path(path_results) / (timestamp + '_{}_{}'.format(env_name, agent))
    path_results_exp_i = path_results_exp
    i = 1
    while path_results_exp_i.exists():
        parts_i = list(path_results_exp.parts)
        parts_i[-1] += f'_({i})'
        path_results_exp_i = Path(*parts_i)
        i += 1
        
    path_results_exp_i.mkdir()
    
    return path_results_exp_i
