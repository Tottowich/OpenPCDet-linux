import argparse
import os, sys
import glob
from pathlib import Path
import time
import numpy as np
from tools.xr_synth_utils import CSVRecorder
import torch
import threading
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from copy import copy
import open3d
sys.path.insert(0, '../../OusterTesting')
import utils_ouster
from transmitter import Transmitter
from tools.visual_utils.open3d_live_vis import LiveVisualizer
from ouster import client
from contextlib import closing
from visual_utils import open3d_vis_utils as V
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from xr_synth_utils import CSVRecorder,TimeLogger,filter_predictions,format_predictions,display_predictions
from pcdet.utils import common_utils
class live_stream(DatasetTemplate):
    """
    Class to stream data from a Sensor.
    Inheritance:
        DatasetTemplate:
            Uses batch processing and collation.
    """
    def __init__(self, dataset_cfg, class_names, root_path=None, logger=None):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names,training=False, root_path=root_path, logger=logger
        )
        self.name = dataset_cfg.DATASET
        self.training = False
        self.frame = 0
    def prep(self,points):
        """started
        Prepare data from the lidar sensor.
        args:
            points: xyz/xyzr points from sensor.
        """
        if self.name in ['NuScenesDataset']:
            points = np.concatenate((points,np.zeros((points.shape[0],1))),axis=1)
        input_dict = {
            'points': points,
            'frame_id': self.frame,
        }

        self.frame += 1
        data_dict = self.prepare_data(data_dict=input_dict)
        return self.collate_batch([data_dict])


def initialize_network(cfg,args,logger,live=None):
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=live)
    if args.ckpt is not None:
        model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    return model

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')
    parser.add_argument('--OU_ip', type=str, default=None, help='specify the ip of the sensor')
    parser.add_argument('--UE5_ip', type=str, default=None, help='specify the ip of the UE5 machine')
    parser.add_argument('--TD_ip', type=str, default=None, help='specify the ip of the TD machine')

    parser.add_argument('--name', type=str, default=None, help='specify the name of the sensor')
    parser.add_argument('--udp_port', type=int, default=7502, help='specify the udp port of the sensor')
    parser.add_argument('--tcp_port', type=int, default=7503, help='specify the tcp port of the sensor')
    parser.add_argument('--TD_port', type=int, default=7002, help='specify the port of the TD machine')
    parser.add_argument('--UE5_port', type=int, default=7000, help='specify the port of the UE5 machine')
    parser.add_argument('--time', type=int, default=100
    , help='specify the tcp port of the sensor')
    parser.add_argument('--save_dir', type=str, default="../lidarCSV", help='specify the save directory')
    parser.add_argument('--save_name', type=str, default="test_csv", help='specify the save name')
    if sys.version_info >= (3,9):
        parser.add_argument('--visualize', action=argparse.BooleanOptionalAction)
        parser.add_argument('--save_csv', action=argparse.BooleanOptionalAction)    
    else:
        parser.add_argument('--visualize', action='store_true')
        parser.add_argument('--no-visualize', dest='visualize', action='store_false')
        parser.add_argument('--save_csv', action='store_true')
        parser.add_argument('--no-save_csv', dest='save_csv', action='store_false')
        parser.add_argument('--log_time', action='store_true')
        parser.add_argument('--no-log_time', dest='log_time', action='store_false')
        parser.set_defaults(visualize=True)
        parser.set_defaults(save_csv=False)
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


@torch.no_grad()
def main():
    args, cfg = parse_config()
    if args.OU_ip is None and args.name is None:
        raise ValueError('Please specify the ip or sensor name of the ')
    # Select classes to use, None -> all.
    classes_to_use = [8]
    # Set up interactions
    logger = common_utils.create_logger()
    live = live_stream(cfg.DATA_CONFIG, cfg.CLASS_NAMES, logger=logger)
    if args.save_csv:
        recorder = CSVRecorder(args.save_name,args.save_dir, cfg.CLASS_NAMES)
        
    # Set up network
    model = initialize_network(cfg,args,logger,live)
    
    # Set up local network ports for IO
    transmitter = Transmitter(reciever_ip=args.TD_ip, reciever_port=args.TD_port, classes_to_send=[9])
    [cfg_ouster, host_ouster] = utils_ouster.sensor_config(args.name if args.name is not None else args.OU_ip,args.udp_port,args.tcp_port)
    transmitter.start_transmit_udp()
    transmitter.start_transmit_ml()
    log_time = False # False to let the program run for one loop to warm up :)
    if args.log_time:       
        time_logger = TimeLogger(logger)
        time_logger.create_metric("Data Prep")
        time_logger.create_metric("Load GPU")
        time_logger.create_metric("Infrence")
        time_logger.create_metric("Filter Predictions")
        if args.visualize:
            time_logger.create_metric("Visualize")
        if args.save_csv:
            time_logger.create_metric("Save CSV")
        if transmitter.started_udp:
            time_logger.create_metric("Transmit TD")
        if transmitter.started_ml:
            time_logger.create_metric("Transmit UE5")
        time_logger.create_metric("Full Pipeline")


    with closing(client.Scans.stream(host_ouster, args.udp_port,complete=False)) as stream:
        logger.info(f"Streaming lidar data to: {cfg.MODEL.NAME}")
        i = 0 # time 
        
        start_stream = time.monotonic()
        
        for scan in stream: # Ouster scan object
            xyz = utils_ouster.get_xyz(stream,scan)
            signal = utils_ouster.get_signal_reflection(stream,scan)
            xyzr = utils_ouster.convert_to_xyzr(xyz,signal)
            xyzr = utils_ouster.compress_mid_dim(xyzr)
            #print(f"Input point cloud shape: {xyzr.shape}")
            if i%2 == 0 and log_time:
                time_logger.start("Full Pipeline")
            if i%2 == 1 and log_time and i != 1:
                time_logger.stop("Full Pipeline")
            i+=1
            if log_time:
                time_logger.start("Data Prep")
            data_dict = live.prep(xyzr)
            if log_time:
                time_logger.stop("Data Prep")
            #print(f"data_dict: {data_dict}\npoints.shape = {data_dict['points'].shape}")
            #print(f"points: {sum(data_dict['points'][:,0]==0)}")
            #print(f"points: {(data_dict['points'][:8,:])}")

            #print(f"points.shape: {data_dict['points'][0].shape}")
            

            if log_time:
                time_logger.start("Load GPU")
            load_data_to_gpu(data_dict)
            if log_time:
                time_logger.stop("Load GPU")

            if log_time:
                time_logger.start("Infrence")
            pred_dicts, _ = model.forward(data_dict)
            if log_time:
                time_logger.stop("Infrence")
                
            if log_time:
                time_logger.start("Filter Predictions")
            # Only uses pred_dicts[0] since batch size is one at live infrence
            pred_dicts = filter_predictions(pred_dicts[0], classes_to_use)
            if log_time:
                time_logger.stop("Filter Predictions")
                
            
            if len(pred_dicts["pred_labels"]) > 0:
                display_predictions(pred_dicts,cfg.CLASS_NAMES,logger)
            if args.save_csv: # If recording, save to csv
                if log_time:
                    time_logger.start("Save CSV")
                recorder.add_frame_file(copy(data_dict["points"][:,1:-1]).cpu().numpy(),pred_dicts)
                if log_time:
                    time_logger.stop("Save CSV")
                #logger.info(f"Time to save to csv: {time.monotonic() - start:.3e}")
            
            if transmitter.started_ml:
                if log_time:
                    time_logger.start("Transmit UE5")
                transmitter.pcd = copy(data_dict["points"][:,1:])
                transmitter.pred_dict = copy(pred_dicts)
                transmitter.send_pcd()
                if log_time:
                    time_logger.stop("Transmit UE5")


            if transmitter.started_udp: # If transmitting, send to udp
                if log_time:
                    time_logger.start("Transmit TD")
                transmitter.pred_dict = copy(pred_dicts)
                transmitter.send_dict()
                if log_time:
                    time_logger.stop("Transmit TD")

            #logger.info(f"Frame {live.frame}")
            if live.frame == 1 and args.visualize:
                vis = LiveVisualizer("XR-SYNTHESIZER",
                                     class_names=cfg.CLASS_NAMES,
                                     first_cloud=data_dict['points'][:,1:],
                                     classes_to_visualize=classes_to_use
                                     )
                start_stream = time.monotonic()
                logger.info(f"Visualizing lidar data: {cfg.MODEL.NAME}:")
                vis.update(points=data_dict['points'][:,1:], 
                            pred_boxes=pred_dicts['pred_boxes'],
                            pred_labels=pred_dicts['pred_labels'],
                            pred_scores=pred_dicts['pred_scores'],
                            )
                #vis = V.create_live_scene(data_dict['points'][:,1:],ref_boxes=pred_dicts[0]['pred_boxes'],
                #ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels'])
            elif args.visualize:
                start = time.monotonic()
                #V.update_live_scene(vis,pts,points=data_dict['points'][:,1:], ref_boxes=pred_dicts[0]['pred_boxes'],
                #    ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels'],class_names=cfg.CLASS_NAMES)
                if log_time:
                    time_logger.start("Visualize")
                vis.update(points=data_dict['points'][:,1:], 
                            pred_boxes=pred_dicts['pred_boxes'],
                            pred_labels=pred_dicts['pred_labels'],
                            pred_scores=pred_dicts['pred_scores'],
                            )
                if log_time:
                    time_logger.stop("Visualize")
            if time.monotonic()-start_stream > args.time:
                break
            if log_time:
                print("\n")
            log_time = args.log_time
                
    transmitter.stop_transmit_udp()
    transmitter.stop_transmit_ml()
    if log_time:
        time_logger.visualize_results()
    logger.info("Stream Done")

"""
NuScenes uses the following labels:
    CLASS_NAMES: ['car','truck', 'construction_vehicle', 'bus', 'trailer',
              'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
    Note: 'pedestrian is predicted as index 9'
    This program uses has been tested with the Ouster OS0-64 sensor.
    Example Input:
        python3 live_predictions.py --cfg_file 'cfgs/nuscenes_models/cbgs_voxel0075_res3d_centerpoint.yaml' --ckpt "../checkpoints/cbgs_voxel0075_centerpoint_nds_6648.pth" --OU_ip "192.168.200.78" --TD_ip "192.168.200.103" --TD_port 7002  --time 300 --udp_port 7001 --tcp_port 7003 --name "OS0-64" --visualize

"""

if __name__ == '__main__':
    main()

