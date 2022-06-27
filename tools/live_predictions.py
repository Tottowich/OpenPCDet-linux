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
from transmitter import Transmitter
from tools.visual_utils.open3d_vis_utils import LiveVisualizer
from ouster import client
from contextlib import closing
from visual_utils import open3d_vis_utils as V
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
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
        """
        Prepare
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
    if logger is None:
        return
    logger.info(f"Model detected: {len(pred_dict[0]['pred_labels'])} objects.")
    for lbls,score in zip(pred_dict[0]['pred_labels'],pred_dict[0]['pred_scores']):
        logger.info(f"\t Prediciton {class_names[lbls-1]}, id: {lbls-1} with confidence: {score}.")



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


@torch.no_grad()
def main():
    args, cfg = parse_config()
    args.visualize = False
    if args.ip is None and args.name is None:
        raise ValueError('Please specify the ip or sensor name of the ')
    vis = None
    classes_to_visualize = [9]
    logger = common_utils.create_logger()
    live = live_stream(cfg.DATA_CONFIG, cfg.CLASS_NAMES, logger=logger)
    model = initialize_network(cfg,args,logger,live)
    transmitter = Transmitter(reciever_ip="192.168.200.103", reciever_port=7002, classes_to_send=[9])
    [cfg_ouster, host_ouster] = utils_ouster.sensor_config(args.name if args.name is not None else args.ip,args.udp_port,args.tcp_port)
    transmitter.start_transmit_udp()
    transmitter.start_transmit_pcd()

    with closing(client.Scans.stream(host_ouster, args.udp_port,complete=False)) as stream:
        logger.info(f"Streaming lidar data: {cfg.MODEL.NAME}:")
        start_stream = time.time()
        
        for scan in stream:
            xyz = utils_ouster.get_xyz(stream,scan)
            signal = utils_ouster.get_signal_reflection(stream,scan)
            xyzr = utils_ouster.convert_to_xyzr(xyz,signal)
            xyzr = utils_ouster.compress_mid_dim(xyzr)
            #print(f"Input point cloud shape: {xyzr.shape}")
            #start = time.time()
            data_dict = live.prep(xyzr)
            
            load_data_to_gpu(data_dict)
            #logger.info(f"Time to prep: {time.time() - start}")
            start = time.time()
            pred_dicts, _ = model.forward(data_dict)
            #logger.info(f"Keys in pred_dicts: {pred_dicts[0].keys()}")
            logger.info(f"Inference time: {time.time() - start:.5f} <=> {1/(time.time() - start):.5f} Hz\n")
            
            if len(pred_dicts[0]['pred_labels']) > 0:
                display_predictions(pred_dicts,cfg.CLASS_NAMES,logger)

            if transmitter.started_ml:
                transmitter.pcd = copy(data_dict["points"][:,1:])
                transmitter.pred_dict = copy(pred_dicts[0])
                transmitter.send_pcd()

            if transmitter.started_udp:
                #transmitter.pcd = copy(data_dict['points'][:,1:])
                start = time.time()
                transmitter.pred_dict = copy(pred_dicts[0])
                transmitter.send_dict("udp")

            #logger.info(f"Frame {live.frame}")
            if live.frame == 1 and args.visualize:
                vis = LiveVisualizer(59,True,class_names=cfg.CLASS_NAMES,first_cloud=data_dict['points'][:,1:],classes_to_visualize=classes_to_visualize)
                vis.update(points=data_dict['points'][:,1:], 
                            ref_boxes=pred_dicts[0]['pred_boxes'],
                            ref_labels=pred_dicts[0]['pred_labels'],
                            class_names=cfg.CLASS_NAMES,
                            )
                #vis = V.create_live_scene(data_dict['points'][:,1:],ref_boxes=pred_dicts[0]['pred_boxes'],
                #ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels'])
            elif args.visualize:
                start = time.time()
                #V.update_live_scene(vis,pts,points=data_dict['points'][:,1:], ref_boxes=pred_dicts[0]['pred_boxes'],
                #    ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels'],class_names=cfg.CLASS_NAMES)
                vis.update(points=data_dict['points'][:,1:], 
                            ref_boxes=pred_dicts[0]['pred_boxes'],
                            ref_labels=pred_dicts[0]['pred_labels'],
                            class_names=cfg.CLASS_NAMES,
                            )
                logger.info(f"Visual time: {time.time() - start:.5f} <=> {1/(time.time() - start):.5f} Hz\n")
            if time.time()-start_stream > args.time:
                break
            #break
    transmitter.stop_transmit_udp()
    logger.info("Stream Done")

if __name__ == '__main__':
    main()

