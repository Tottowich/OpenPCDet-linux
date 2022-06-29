import argparse
import os, sys
import glob
from pathlib import Path
import time
import numpy as np
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
from pcdet.utils import common_utils
class ouster_streamer:
    def __init__(self, stream):
        self.stream = stream
        self.started = False
        self.q_xyzr = Queue()

    def start_thread(self):
        self.started = True
        self.thread = threading.Thread(target=self.stream_loop)
        #self.thread.daemon = True
        self.thread.start()
        time.sleep(0.2)
    def stop_thread(self):
        self.started = False
        self.thread.join()
    def stream_loop(self):
        for scan in self.stream:
            if not self.started:
                break
            xyz = utils_ouster.get_xyz(self.stream,scan)
            signal = utils_ouster.get_signal_reflection(self.stream,scan)
            xyzr = utils_ouster.convert_to_xyzr(xyz,signal)
            self.q_xyzr.put(utils_ouster.compress_mid_dim(xyzr))
    def get_pcd(self):
        try:
            return self.q_xyzr.get(timeout=1e-6)
        except:
            return None
class live_stream(DatasetTemplate):
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
def display_predictions(pred_dict, class_names, logger=None):
    """
    Display predictions.
    args:
        pred_dict: prediction dictionary. "pred_boxes", "pred_labels", "pred_scores"
        class_names: list of class names
    """
    if logger is None:
        return
    logger.info(f"Model detected: {len(pred_dict['pred_labels'])} objects.")
    for lbls,score in zip(pred_dict['pred_labels'],pred_dict['pred_scores']):
        logger.info(f"lbls: {lbls} score: {score}")
        logger.info(f"\t Prediciton {class_names[lbls[0]-1]}, id: {lbls[0]-1} with confidence: {score:.5f}.")

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
    if sys.version_info >= (3,9):
        parser.add_argument('--visualize', action=argparse.BooleanOptionalAction)
    else:
        parser.add_argument('--visualize', action='store_true')
        parser.add_argument('--no-visualize', dest='visualize', action='store_false')
        parser.set_defaults(feature=True)
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg
def get_ouster_data(stream):
    scan = next(iter(stream))
    xyz = utils_ouster.get_xyz(stream,scan)
    signal = utils_ouster.get_signal_reflection(stream,scan)
    xyzr = utils_ouster.convert_to_xyzr(xyz,signal)
    return utils_ouster.compress_mid_dim(xyzr)


@torch.no_grad()
def main():
    args, cfg = parse_config()
    args.visualize = False
    if args.OU_ip is None and args.name is None:
        raise ValueError('Please specify the ip or sensor name of the ')
    
    classes_to_use = [9]

    logger = common_utils.create_logger()
    live = live_stream(cfg.DATA_CONFIG, cfg.CLASS_NAMES, logger=logger)

    model = initialize_network(cfg,args,logger,live)
    transmitter = Transmitter(reciever_ip=args.TD_ip, reciever_port=args.TD_port, classes_to_send=[9])
    [cfg_ouster, host_ouster] = utils_ouster.sensor_config(args.name if args.name is not None else args.OU_ip,args.udp_port,args.tcp_port)
    transmitter.start_transmit_udp()
    transmitter.start_transmit_ml()
    executor = ThreadPoolExecutor(max_workers=1)

    with closing(client.Scans.stream(host_ouster, args.udp_port,complete=False)) as stream:
        logger.info(f"Streaming lidar data: {cfg.MODEL.NAME}:")
        i = 0
        if False:
            ous_stream = ouster_streamer(stream=stream)
            ous_stream.start_thread()
            start_stream = time.monotonic()
            while time.monotonic()-start_stream <args.time:
                pcd = ous_stream.get_pcd()
                if pcd is not None:
                    data_dict = live.prep(pcd)
                    load_data_to_gpu(data_dict)
                    #logger.info(f"Time to prep: {time.monotonic() - start}")
                    start = time.monotonic()
                    pred_dicts, _ = model.forward(data_dict)
                    logger.info(f"Time to make predictions: {time.monotonic() - start:.5f} <=> {1/(time.monotonic() - start):.5f} Hz")
        #ous_stream.stop_thread()
        start_stream = time.monotonic()
        while True:
        #for scan in stream:
            if i == 0:
                start_time = time.monotonic()
                scan = next(iter(stream))
                xyz = utils_ouster.get_xyz(stream,scan)
                signal = utils_ouster.get_signal_reflection(stream,scan)
                xyzr = utils_ouster.convert_to_xyzr(xyz,signal)
                xyzr = utils_ouster.compress_mid_dim(xyzr)
            else:
                if loader.done():
                    logger.info(f"Loading from thread")
                    xyzr = loader.result()
                elif loader.running():
                    logger.info(f"Waiting for thread to finish")
                    while time.monotonic() - start_stream< args.time and not loader.done():
                        time.sleep(0.005)
                    xyzr = loader.result() if time.monotonic() - start_stream< args.time else None
                else:
                    logger.info(f"No data received")
                    xyzr = None

            if i == 1:
                start_time = time.monotonic()
            if i == 2:
                end_time = time.monotonic()
                logger.info(f"Time for one loop: {end_time-start_time:.5f} seconds.")
            i += 1
            
            start = time.monotonic()
            
            logger.info(f"Time to process lidar data: {time.monotonic()-start:.5f}")
            #print(f"Input point cloud shape: {xyzr.shape}")
            #start = time.monotonic()
            start = time.monotonic()
            data_dict = live.prep(xyzr)
            logger.info(f"Time to prepare data: {time.monotonic()-start:.5f}")
            load_data_to_gpu(data_dict)
            #logger.info(f"Time to prep: {time.monotonic() - start}")
            start = time.monotonic()
            pred_dicts, _ = model.forward(data_dict)
            loader = executor.submit(get_ouster_data, stream)
            logger.info(f"Time to make predictions: {time.monotonic() - start:.5f} <=> {1/(time.monotonic() - start):.5f} Hz")
            if classes_to_use is not None:
                pred_dicts = filter_predictions(pred_dicts[0], classes_to_use)
            else:
                pred_dicts = pred_dicts[0]
            #logger.info(f"Keys in pred_dicts: {pred_dicts[0].keys()}")
            
            #if len(pred_dicts["pred_labels"]) > 0:
            #    display_predictions(pred_dicts,cfg.CLASS_NAMES,logger)

            if transmitter.started_ml:
                start = time.monotonic()
                transmitter.pcd = copy(data_dict["points"][:,1:])
                transmitter.pred_dict = copy(pred_dicts)
                transmitter.send_pcd()
                logger.info(f"Time to send pcd: {time.monotonic() - start:.5f}")


            if transmitter.started_udp:
                start = time.monotonic()
                #transmitter.pcd = copy(data_dict['points'][:,1:])
                transmitter.pred_dict = copy(pred_dicts)
                transmitter.send_dict()
                logger.info(f"Time to send udp: {time.monotonic()-start:.5f}")

            #logger.info(f"Frame {live.frame}")
            if live.frame == 1 and args.visualize:
                vis = LiveVisualizer("XR-SYNTHESIZER",
                                     class_names=cfg.CLASS_NAMES,
                                     first_cloud=data_dict['points'][:,1:],
                                     classes_to_visualize=classes_to_use
                                     )
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
                vis.update(points=data_dict['points'][:,1:], 
                            pred_boxes=pred_dicts['pred_boxes'],
                            pred_labels=pred_dicts['pred_labels'],
                            pred_scores=pred_dicts['pred_scores'],
                            )
                logger.info(f"Visual time: {time.monotonic() - start:.5f} <=> {1/(time.monotonic() - start):.5f} Hz\n")
            if time.monotonic()-start_stream > args.time:
                break
            
    transmitter.stop_transmit_udp()
    transmitter.stop_transmit_ml()
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

