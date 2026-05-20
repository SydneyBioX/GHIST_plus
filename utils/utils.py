"""General utility functions for GHIST+ training and inference."""

import os
import datetime as dt
import json
import collections
import re
import torch
from scipy.special import softmax
import numpy as np
import random
import matplotlib.pyplot as plt
import natsort
from torchvision.utils import save_image, make_grid
from matplotlib.animation import FuncAnimation, PillowWriter
import tifffile
import scipy
import pandas as pd
import glob
import h5py
from sklearn.manifold import TSNE


def get_device(gpu_id):
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        print("Using GPUs: {}".format(visible))
    else:
        gpu_str = str(gpu_id)
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str
        print("Using GPUs: {}".format(gpu_str))
    device = torch.device("cuda")

    return device


def sorted_alphanumeric(data):
    """
    Alphanumerically sort a list
    """
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    return sorted(data, key=alphanum_key)


def read_txt(fp):
    with open(fp) as file:
        lines = [line.rstrip() for line in file]
    return lines


def delete_file(path):
    """
    Delete file if exists
    """
    if os.path.exists(path):
        os.remove(path)


def get_files_list(path, ext_array=[".tif"]):
    """
    Get all files in a directory with a specific extension
    """
    files_list = list()
    dirs_list = list()

    for root, dirs, files in os.walk(path, topdown=True):
        for file in files:
            if any(x in file for x in ext_array):
                files_list.append(os.path.join(root, file))
                folder = os.path.dirname(os.path.join(root, file))
                if folder not in dirs_list:
                    dirs_list.append(folder)

    return files_list, dirs_list


def json_file_to_pyobj(filename):
    """
    Read json config file
    """

    def _expand_config_paths(value):
        if isinstance(value, str):
            return os.path.expandvars(os.path.expanduser(value))
        if isinstance(value, list):
            return [_expand_config_paths(v) for v in value]
        if isinstance(value, dict):
            return {k: _expand_config_paths(v) for k, v in value.items()}
        return value

    def _json_object_hook(d):
        return collections.namedtuple("X", d.keys())(*d.values())

    def json2obj(data):
        payload = _expand_config_paths(json.loads(data))
        return json.loads(json.dumps(payload), object_hook=_json_object_hook)

    return json2obj(open(filename).read())


def get_newest_id(exp_dir="experiments", fold_id=1):
    """Get the latest experiment ID based on its timestamp

    Parameters
    ----------
    exp_dir : str, optional
        Name of the directory that contains all the experiment directories, by default 'experiments'

    Returns
    -------
    exp_id : str
        Name of the latest experiment directory
    """
    folders = next(os.walk(exp_dir))[1]
    folders = natsort.natsorted(folders)
    # folders = [x for x in folders if mode in x]
    folders = [x for x in folders if ("fold" + str(fold_id) + "_") in x]
    folder_last = folders[-1]
    exp_id = folder_last.replace("\\", "/")
    return exp_id


def get_experiment_id(make_new, load_dir, fold_id):
    """
    Get timestamp ID of current experiment
    """
    if make_new is False:
        if load_dir == "last":
            timestamp = get_newest_id("experiments", fold_id)
        else:
            timestamp = load_dir
    else:
        timestamp = (
            "fold"
            + str(fold_id)
            + "_"
            + dt.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        )

    return timestamp


def normalise_counts(df):
    total_counts_per_cell = df.sum(axis=1)

    scale_factor = 10000
    df_normalized = df.div(total_counts_per_cell, axis=0) * scale_factor

    df_normalized_with_pseudo = df_normalized + 1

    df_log_transformed = np.log(df_normalized_with_pseudo)

    return df_log_transformed
