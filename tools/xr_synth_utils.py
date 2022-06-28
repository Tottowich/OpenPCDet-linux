import argparse
import os, sys
import glob
from pathlib import Path
import time
import numpy as np
import torch
from copy import copy
import open3d
from datetime import datetime as dt
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
        
        #indices = [np.nonzero(sum(pred_dict["pred_labels"]==x for x in classes_to_use))[0].tolist()][0]
        #print(np.nonzero((sum(pred_dict["pred_labels"]-1==x for x in classes_to_use))))
        indices = np.nonzero((sum(pred_dict["pred_labels"]-1==x for x in classes_to_use)))[0].tolist()
        #print(f"indices: {indices}")
        pred_dict["pred_boxes"] = pred_dict["pred_boxes"].reshape(pred_dict["pred_boxes"].shape[0],-1)[indices,:]
        pred_dict["pred_labels"] = pred_dict["pred_labels"].reshape(pred_dict["pred_labels"].shape[0],-1)[indices,:]-1
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


class CSVRecorder():
    def __init__(self, 
                 folder_name=f"csv_folder_{dt.now().strftime('%Y%m%d_%H%M%S')}",
                 main_folder="./lidarCSV",
                 class_names=None,
                 ):
        self.main_folder = main_folder
        self.folder_name = folder_name
        self.class_names = class_names
        self.path = os.path.join(self.main_folder, self.folder_name)
        
        self.labelfile = "label"
        self.cloudfile = "cloud"
        if not os.path.exists(self.main_folder):
            os.makedirs(self.main_folder)
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        self.frames = 0
    def process_labels(self,pred_dict):
        boxes = np.array(pred_dict["pred_boxes"][:,:6])
        labels = np.array([self.class_names[x] for x in pred_dict["pred_labels"]] if len(pred_dict["pred_labels"]) > 0 else []).reshape(-1,1)
        scores = np.array(pred_dict["pred_scores"]).reshape(-1,1)
        print(f"boxes: {boxes}")
        print(f"labels: {labels}")
        print(f"scores: {scores}")

        labels = np.concatenate((boxes,labels,scores),axis=1)
        return labels

    def add_frame_file(self, cloud,pred_dict):
        cloud_name = os.path.join(self.path, f"cloud_{self.frames}.csv")
        label_name = os.path.join(self.path, f"label_{self.frames}.csv")
        np.savetxt(cloud_name, cloud, header = "x, y, z, r",delimiter=",")
        np.savetxt(label_name, self.process_labels(pred_dict=pred_dict), header = "x, y, z, w, l, h, label, score",delimiter=",",fmt="%s")
        self.frames += 1
        