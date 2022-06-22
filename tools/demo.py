import argparse
import glob
from pathlib import Path
import time
import re
import sys
import socket
from transmitter import Transmitter
from matplotlib.pyplot import cla


from tools.visual_utils.open3d_vis_utils import LiveVisualizer
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

def sorted_alphanumeric(data):
    """
    Sort the given iterable in the way that humans expect.
    Args:
        data: An iterable.
    Returns: sorted version of the given iterable.
    """
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

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

def display_predictions(pred_dict, class_names, logger=None):
    if logger is None:
        return
    logger.info(f"Model detected: {len(pred_dict[0]['pred_labels'])} objects.")
    for lbls,score in zip(pred_dict[0]['pred_labels'],pred_dict[0]['pred_scores']):
        logger.info(f"\t Prediciton {class_names[lbls-1]} with confidence: {score}.")
    
def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')
    transmitter = Transmitter(reciever_ip="192.168.200.103", reciever_port= 7002)
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    #vis = LiveVisualizer(100,True)
    client = udp_client.SimpleUDPClient(args.ip, args.port)
    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            logger.info(f'Visualized sample index: \t{idx + 1}')
            data_dict = demo_dataset.collate_batch([data_dict])
            logger.info(f"Loading Data to GPU...")
            start = time.time()
            load_data_to_gpu(data_dict)
            logger.info(f"Loading Data to GPU loading took {time.time() - start:.5f} s.")
            
            logger.info(f"Running inference...")
            start = time.time()
            pred_dicts, _ = model.forward(data_dict)
            if transmitter is not None:
                transmitter.send_dict(copy(pred_dicts[0]))
            logger.info(f"Infrence time: {time.time() - start:.5f} <=> {1/(time.time() - start):.5f} Hz")
            if len(pred_dicts[0]['pred_labels']) > 0:
                display_predictions(pred_dicts,cfg.CLASS_NAMES,logger)
            #logger.info(f"Predction keys : ·{pred_dicts[0].keys()}")
            #logger.info(f"Predicitons : ·{pred_dicts[0]}")
            if idx==0: # This could be run on a seperate thread!
                #vis,pts, = V.create_live_scene(points=data_dict['points'][:,1:], 
                #                                ref_boxes=pred_dicts[0]['pred_boxes'],
                #                                ref_scores=pred_dicts[0]['pred_scores'], 
                #                                ref_labels=pred_dicts[0]['pred_labels'],
                #                                class_names=cfg.CLASS_NAMES)
                vis = LiveVisualizer(59,True,class_names=cfg.CLASS_NAMES,first_cloud=data_dict['points'][:,1:])
                vis.update(points=data_dict['points'][:,1:], 
                            ref_boxes=pred_dicts[0]['pred_boxes'],
                            ref_labels=pred_dicts[0]['pred_labels'],
                            class_names=cfg.CLASS_NAMES,
                            )
            else:
                start = time.time()
                #V.update_live_scene(vis,pts,points=data_dict['points'][:,1:], ref_boxes=pred_dicts[0]['pred_boxes'],
                #    ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels'],class_names=cfg.CLASS_NAMES)
                vis.update(points=data_dict['points'][:,1:], 
                            ref_boxes=pred_dicts[0]['pred_boxes'],
                            ref_labels=pred_dicts[0]['pred_labels'],
                            class_names=cfg.CLASS_NAMES,
                            )
                logger.info(f"Visual time: {time.time() - start:.5f} <=> {1/(time.time() - start):.5f} Hz\n")
           
            #V.draw_scenes(
            #    points=data_dict['points'][:,1:], ref_boxes=pred_dicts[0]['pred_boxes'],
            #    ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
            #)

            if not OPEN3D_FLAG:
                mlab.show(stop=True)
    input("Press \'Enter\' to exit demo...")
    logger.info('Demo done.')


if __name__ == '__main__':
    main()
