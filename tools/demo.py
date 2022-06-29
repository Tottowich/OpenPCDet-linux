import argparse
import glob
from pathlib import Path
import time
import re
import sys
import socket
from transmitter import Transmitter
from xr_synth_utils import CSVRecorder,filter_predictions,format_predictions,display_predictions,sorted_alphanumeric
from copy import copy

from matplotlib.pyplot import cla


from tools.visual_utils.open3d_live_vis import LiveVisualizer
#sys.path.insert(0, '../../OusterTesting')
#import utils_ouster
try:
    import open3d
    from visual_utils import open3d_vis_utils as V
    OPEN3D_FLAG = True
except:
    import mayavi.mlab as mlab
    from visual_utils import visualize_utils as V
    OPEN3D_FLAG = False

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from pythonosc import udp_client


class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.npy'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]
        if len(data_file_list) > 1:
            data_file_list = sorted_alphanumeric(data_file_list)
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
            #if len(points.shape) == 5 and len(self.dataset_cfg.POINT_FEATURE_ENCODING.used_feature_list) == 5:
                #print('Warning: The point cloud has 4 features, but the dataset config uses 5 features. ')
            points = np.concatenate((points,np.zeros((points.shape[0],1))),axis=1)
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')
    parser.add_argument('--udp_port', type=int, default=7502, help='specify the udp port of the sensor')
    parser.add_argument('--tcp_port', type=int, default=7503, help='specify the tcp port of the sensor')
    parser.add_argument('--TD_port', type=int, default=7002, help='specify the port of the TD machine')
    parser.add_argument('--UE5_port', type=int, default=7000, help='specify the port of the UE5 machine')
    parser.add_argument('--OU_ip', type=str, default=None, help='specify the ip of the sensor')
    parser.add_argument('--UE5_ip', type=str, default=None, help='specify the ip of the UE5 machine')
    parser.add_argument('--TD_ip', type=str, default=None, help='specify the ip of the TD machine')
    if sys.version_info >= (3,9):
            parser.add_argument('--visualize', action=argparse.BooleanOptionalAction)
            parser.add_argument('--save_csv', action=argparse.BooleanOptionalAction)    
    else:
        parser.add_argument('--visualize', action='store_true')
        parser.add_argument('--no-visualize', dest='visualize', action='store_false')
        parser.add_argument('--save_csv', action='store_true')
        parser.add_argument('--no-save_csv', dest='save_csv', action='store_false')
        parser.set_defaults(visualize=True)
        parser.set_defaults(save_csv=False)
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    if args.save_csv:
        recorder = CSVRecorder(args.save_name,args.save_dir, cfg.CLASS_NAMES)
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    classes_to_use = None
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')
    transmitter = Transmitter(reciever_ip=args.TD_ip, reciever_port=args.TD_port, classes_to_send=classes_to_use)
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    i = 0
    #vis = LiveVisualizer(100,True)
    #client = udp_client.SimpleUDPClient(args.ip, args.port)
    #transmitter.start_transmit_udp()
    #transmitter.start_transmit_ml()
    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            logger.info(f'Visualized sample index: \t{idx + 1}')

            start = time.monotonic()
            data_dict = demo_dataset.collate_batch([data_dict])
            logger.info(f"Time to process lidar data: {time.monotonic()-start:.3e}")

            start = time.monotonic()
            load_data_to_gpu(data_dict)
            logger.info(f"Time to prep: {time.monotonic() - start:.3e}")

            start = time.monotonic()
            pred_dicts, _ = model.forward(data_dict)
            logger.info(f"Time to make predictions: {time.monotonic() - start:.3e} <=> {1/(time.monotonic() - start):.3e} Hz")
            if classes_to_use is not None: # Only uses pred_dicts[0] since batch size is one at live infrence
                pred_dicts = filter_predictions(pred_dicts[0], classes_to_use)
                #print(f"Filtered pred_dicts: {pred_dicts}")
            else:
                pred_dicts = format_predictions(pred_dicts[0])
            #if transmitter is not None:
            #    transmitter.send_dict(copy(pred_dicts[0]))
            if len(pred_dicts['pred_labels']) > 0:
                display_predictions(pred_dicts,cfg.CLASS_NAMES,logger)

            if transmitter.started_ml:
                start = time.monotonic()
                transmitter.pcd = copy(data_dict["points"][:,1:])
                transmitter.pred_dict = copy(pred_dicts)
                transmitter.send_pcd()
                logger.info(f"Time to send pcd: {time.monotonic() - start:.3e}")

            if transmitter.started_udp: # If transmitting, send to udp
                start = time.monotonic()
                transmitter.pred_dict = copy(pred_dicts)
                transmitter.send_dict()
                logger.info(f"Time to send udp: {time.monotonic()-start:.3e}")
            #logger.info(f"Predction keys : ·{pred_dicts[0].keys()}")
            #logger.info(f"Predicitons : ·{pred_dicts[0]}")
            if i == 0 and args.visualize:
                i += 1
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
            elif args.visualize:
                start = time.monotonic()
                #V.update_live_scene(vis,pts,points=data_dict['points'][:,1:], ref_boxes=pred_dicts[0]['pred_boxes'],
                #    ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels'],class_names=cfg.CLASS_NAMES)
                vis.update(points=data_dict['points'][:,1:], 
                            pred_boxes=pred_dicts['pred_boxes'],
                            pred_labels=pred_dicts['pred_labels'],
                            pred_scores=pred_dicts['pred_scores'],
                            )
                logger.info(f"Visual time: {time.monotonic() - start:.3e} <=> {1/(time.monotonic() - start):.3e} Hz")
           
            #V.draw_scenes(
            #    points=data_dict['points'][:,1:], ref_boxes=pred_dicts[0]['pred_boxes'],
            #    ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
            #)

    transmitter.stop_transmit_udp()
    transmitter.stop_transmit_ml()
    input("Press \'Enter\' to exit demo...")
    logger.info('Demo done.')


if __name__ == '__main__':
    main()
