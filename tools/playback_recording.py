import argparse
import numpy as np
import os
from pcdet.config import cfg, cfg_from_yaml_file
from xr_synth_utils import CSVRecorder,filter_predictions,format_predictions,display_predictions,sorted_alphanumeric
from tools.visual_utils.open3d_live_vis import LiveVisualizer
parser = argparse.ArgumentParser(description='arg parser')
def parser_config():
    parser.add_argument('--data_path', type=str, default=None, help='specify the path to the lidar folder')
    parser.add_argument('--cfg_file', type=str, default="cfgs/nuscenes_models/cbgs_voxel0075_res3d_centerpoint.yaml",
                        help='specify the config for demo')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    args = parser.parse_args()
    if args.cfg_file is not None:
        cfg_from_yaml_file(args.cfg_file,cfg)
    return args, cfg

def load_labels(csv_path):
    data = np.loadtxt(csv_path, delimiter=',',dtype=str)
    data = data.reshape(-1,9)
    print(f"data.shape: {data.shape}")
    print(data)
    return data
def load_cloud(csv_path):
    data = np.loadtxt(csv_path, delimiter=',')
    #data = data.reshape(-1,4)
    print(f"data.shape: {data.shape}")
    return data
def numpy_to_cloud_dict(data):
    data_dict = {}
    data_dict['points'] = data
    data_dict['frame_id'] = 0
    return data_dict
def numpy_to_label_dict(data):
    pred_dict = {}
    print(data.shape)
    if len(data)==0:
        pred_dict["pred_boxes"] = None
        pred_dict["pred_scores"] = None
        pred_dict["pred_labels"] = None
        pred_dict['pred_labels_id'] = None
    else:
        pred_dict["pred_boxes"] = np.concatenate((data[:,:6].astype(np.float32),np.zeros((data.shape[0],3))),axis=1)
        pred_dict["pred_labels"] = data[:,6]
        pred_dict["pred_labels_id"] = data[:,7].astype(np.int32)
        pred_dict["pred_scores"] = data[:,8].astype(np.float16)
    return pred_dict

def get_label_and_cloud(csv_path,index):
    cloud = load_cloud(os.path.join(csv_path,f"cloud_{index}.csv"))
    print(cloud.shape)
    label = load_labels(os.path.join(csv_path,f"label_{index}.csv"))
    print(label)
    data_dict = numpy_to_cloud_dict(cloud)
    label_dict = numpy_to_label_dict(label)
    print(label_dict)
    return data_dict, label_dict

def loop_over_directory(directory,cfg):
    print(directory)
    dir = sorted_alphanumeric(os.listdir(directory))
    for index in range(0,len(dir)//2):
        data_dict, pred_dicts = get_label_and_cloud(directory,index)
        print(pred_dicts)
        if index==0:
            vis = LiveVisualizer("XR-SYNTHESIZER",
                                    class_names=cfg.CLASS_NAMES,
                                    first_cloud=data_dict['points'][:,:-1],
                                )
            vis.update(points=data_dict['points'][:,:-1], 
                        ref_boxes=pred_dicts['pred_boxes'],
                        ref_labels=pred_dicts['pred_labels_id'],
                        ref_scores=pred_dicts['pred_scores'],
            )
        else:
            #V.update_live_scene(vis,pts,points=data_dict['points'][:,1:], ref_boxes=pred_dicts[0]['pred_boxes'],
            #    ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels'],class_names=cfg.CLASS_NAMES)
            vis.update(points=data_dict['points'][:,:-1], 
                        ref_boxes=pred_dicts['pred_boxes'],
                        ref_labels=pred_dicts['pred_labels_id'],
                        ref_scores=pred_dicts['pred_scores'],
                        )

    input("Press Enter to continue...")
def main():
    args, cfg = parser_config()
    loop_over_directory(args.data_path,cfg)
if __name__ == "__main__":
    """
    Play back recordings stored in a directory of csv's.
    --data_path: path to the directory containing the csv's.
    --cfg_file: path to the config file of the model used to generate the predictions. Default CenterPoint 0.075m
    """
    main()

