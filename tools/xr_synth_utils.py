import argparse
import os, sys
import glob
from pathlib import Path
import time
import numpy as np
import torch
from copy import copy
import open3d
sys.path.insert(0, '../../OusterTesting')
import utils_ouster
def filter_predictions(pred_dict, classes_to_use):
    """
    Filter predictions to only include the classes we want to use.
    """
    if isinstance(pred_dict["pred_labels"],torch.Tensor):
            pred_dict["pred_labels"] = pred_dict["pred_labels"].cpu().numpy()
    if isinstance(pred_dict["pred_boxes"],torch.Tensor):
        pred_dict["pred_boxes"] = pred_dict["pred_boxes"].cpu().numpy()
    if isinstance(pred_dict["pred_scores"],torch.Tensor):
        pred_dict["pred_scores"] = pred_dict["pred_scores"].cpu().numpy()
    if classes_to_use is not None and len(pred_dict["pred_labels"]) > 0:
        indices = [np.nonzero(sum(pred_dict["pred_labels"]==x for x in classes_to_use))[0].tolist()][0]
        pred_dict["pred_boxes"] = pred_dict["pred_boxes"].reshape(pred_dict["pred_boxes"].shape[0],-1)[indices,:]
        pred_dict["pred_labels"] = pred_dict["pred_labels"].reshape(pred_dict["pred_labels"].shape[0],-1)[indices,:]
        pred_dict["pred_scores"] = pred_dict["pred_scores"].reshape(pred_dict["pred_scores"].shape[0],-1)[indices,:]
    return pred_dict
    
def generate_distance_matrix(pred_dict):
    if isinstance(pred_dict["pred_boxes"],torch.Tensor):
        pred_dict["pred_boxes"] = pred_dict["pred_boxes"].cpu().numpy()
    pred_dict["distance_matrix"] = np.zeros((pred_dict["pred_boxes"].shape[0],pred_dict["pred_boxes"].shape[0]))
    for i in range(pred_dict["pred_boxes"].shape[0]):
        for j in range(pred_dict["pred_boxes"].shape[0]):
            pred_dict["distance_matrix"][i,j] = np.linalg.norm(pred_dict["pred_boxes"][i,:3]-pred_dict["pred_boxes"][j,:3])
    return pred_dict
    