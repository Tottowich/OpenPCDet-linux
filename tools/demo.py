import argparse
import glob
from pathlib import Path
import time
import re
import sys
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

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
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

        data_file_list = sorted_alphanumeric(data_file_list)
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
            print(points.shape)
            points = np.concatenate((points,np.zeros((points.shape[0],1))),axis=1)
            print(points.shape)
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

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    
    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            start = time.time()
            logger.info(f'Visualized sample index: \t{idx + 1}')
            #print(f"Before: {data_dict}")
            data_dict = demo_dataset.collate_batch([data_dict])
            #print(f"After: {data_dict}")
            logger.info(f"Loading Data to GPU...")
            load_data_to_gpu(data_dict)
            #logger.info(f"data_dict.keys(): \t{data_dict.keys()}")
            #logger.info(f"data_dict['points'].shape: {data_dict['points'].shape}")
            #logger.info(f"data_dict['voxels'].shape: {data_dict['points'].shape}")
            #for key in data_dict:
            #    logger.info(f"{key} shape: \t{data_dict[key]}")
            logger.info(f"Running inference...")
            pred_dicts, _ = model.forward(data_dict)
            #logger.info(f"Predction keys : ·{pred_dicts[0].keys()}")
            #logger.info(f"Predicitons : ·{pred_dicts[0]}")
            if idx==0:
                vis = V.create_live_scene(points=data_dict['points'][:,1:], ref_boxes=pred_dicts[0]['pred_boxes'],
                    ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels'])
            else:
                V.update_live_scene(vis,points=data_dict['points'][:,1:], ref_boxes=pred_dicts[0]['pred_boxes'],
                    ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels'])
            logger.info(f"Infrence time: {time.time() - start} <=> {1/(time.time() - start)} Hz")
                #V.draw_scenes(
                #    points=data_dict['points'][:,1:], ref_boxes=pred_dicts[0]['pred_boxes'],
                #    ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
                #)

            if not OPEN3D_FLAG:
                mlab.show(stop=True)

    logger.info('Demo done.')


if __name__ == '__main__':
    main()
