import json
import os
import mmcv
import torch
import copy
import numpy as np
import tempfile
import warnings
from pathlib import Path
from os import path as osp
from torch.utils.data import Dataset
from mmdet3d.datasets.utils import extract_result_dict, get_loading_pipeline
# from .evaluation.precision_recall.average_precision_gen import eval_chamfer

from mmdet.datasets import DATASETS
from mmdet.datasets.pipelines import to_tensor
from mmcv.parallel import DataContainer as DC

from mmdet3d.datasets.pipelines import Compose
import time

from .renderer import Renderer

CAMERA_NAMES = [
    'camera0',
    'camera1',
    'camera2',
    'camera3',
    'camera4',
    'camera5'
]

def get_ego2img(extrinsics_mat, intrinsics_mat):
    ego2cam_rt = extrinsics_mat
    viewpad = np.eye(4)
    viewpad[:intrinsics_mat.shape[0], :intrinsics_mat.shape[1]] = intrinsics_mat
    ego2cam_rt = (viewpad @ ego2cam_rt)
    return ego2cam_rt

@DATASETS.register_module()
class Tier4MapDataset(Dataset):
    def __init__(self,
                 dataroot='',
                 pipeline=None,
                 visualize_cfg: dict = dict(),
                 eval_cfg: dict = dict(),
                 score_th=0.5,
                 work_dir=None,
                 img_norm_cfg=None,
                 **kwargs
                 ):
        super().__init__()
        self.dataroot = dataroot
        self.visualize_cfg = visualize_cfg
        self.eval_cfg = eval_cfg
        self.score_th = score_th
        self.work_dir = work_dir
        self.CLASSES = [] # Dummy
        self.img_norm_cfg = img_norm_cfg

        if pipeline is not None:
            self.pipeline = Compose(pipeline)
        else:
            self.pipeline = None

        metadata_path = Path(self.dataroot) / 'metadata.json'
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        self.samples = list(self.metadata.keys())
        self.flag = np.zeros(len(self), dtype=np.uint8)

    def get_sample(self, index):
        # index = 4174
        # print('OVERWRITING INDEX!!!!')
        timestamp_id = self.samples[index]
        info = self.metadata[timestamp_id]
        img_filenames = []
        for camera_name in CAMERA_NAMES:
            img_path = Path(self.dataroot) / 'images' / info['images'][camera_name]
            img_filenames.append(str(img_path))
        pose_filename = str(Path(self.dataroot) / 'poses' / info['pose'])
        intrinsics_filename = str(Path(self.dataroot) / 'intrinsics' / info['intrinsics'])
        extrinsics_filename = str(Path(self.dataroot) / 'extrinsics' / info['extrinsics'])

        with open(pose_filename, 'r') as f:
            pose = json.load(f)['pose']
        with open(intrinsics_filename, 'r') as f:
            intrinsics_dict = json.load(f)
        with open(extrinsics_filename, 'r') as f:
            extrinsics_dict = json.load(f)
        
        intrinsics = np.array([intrinsics_dict[c] for c in CAMERA_NAMES])
        extrinsics = np.array([extrinsics_dict[c] for c in CAMERA_NAMES])
        ###### WHY INVERSE?? #######
        extrinsics = np.array([np.linalg.inv(ex) for ex in extrinsics])
        ###### WHY INVERSE?? #######

        ego2img = np.array([
            get_ego2img(extr, intr) for extr, intr in zip(extrinsics, intrinsics)
        ])

        # input_dict = dict(
        #     sample_idx=timestamp_id,
        #     # img_filenames=img_filenames,
        #     img_filename=img_filenames,
        #     global2ego_pose=pose,
        #     # cam_intrinsics=copy.deepcopy(self.intrinsics),
        #     # cam_extrinsics=copy.deepcopy(self.extrinsics),
        #     # ego2img=copy.deepcopy(self.ego2img_rts),
        #     cam_intrinsics=intrinsics,
        #     cam_extrinsics=extrinsics,
        #     ego2img=ego2img,
        #     map_name=info['map_name']
        # )

        img_metas = dict(
            can_bus=None,
            lidar2img=ego2img,
            img_norm_cfg=self.img_norm_cfg,
            sample_idx=index,
            filename=img_filenames,
        )

        input_dict = dict(
            # sample_idx=timestamp_id,
            img_filename=img_filenames,
            global2ego_pose=pose,
            map_name=info['map_name'],
            # Would be gathered as img_metas in Collect3D
            can_bus=np.array([0.0 for _ in range(18)]),
            lidar2img=ego2img,
            sample_idx=index,
            scene_token=0,
            pts_filename=f'{timestamp_id}.hogehoge'
        )

        return input_dict

    def prepare_data(self, index):
        """Prepare data for testing.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Testing data dict of the corresponding index.
        """
        input_dict = self.get_sample(index)
        example = self.pipeline(input_dict)

        ### Only for MapTR ###
        # print('KOJI!!!!!!!!!!!! ', example['img'][0][0].shape)
        # print('KOJI!!!!!!!!!!!! ', example['img'][1][1].shape)
        if len(example['img'][0][0].shape) == 2:
            example['img'] = DC(torch.stack([
                torch.from_numpy(x.transpose(2, 0, 1)) for x in example['img']
            ]).unsqueeze(0), cpu_only=False, stack=True)
            example['img_metas'] = DC({0: example['img_metas'].data}, cpu_only=True)
        elif len(example['img'][0][0].shape) == 3:
            example['img'] = DC(torch.stack([
                torch.from_numpy(x.transpose(2, 0, 1)) for x in example['img'][0]
            ]).unsqueeze(0), cpu_only=False, stack=True)
            example['img_metas'] = DC({0: example['img_metas'][0].data}, cpu_only=True)
            # example['img_metas'] = DC(example['img_metas'][0].data, cpu_only=True)
        ### Only for MapTR ###
        return example

    def __len__(self):
        """Return the length of data infos.

        Returns:
            int: Length of data infos.
        """
        return len(self.samples)
        
    def _rand_another(self, idx):
        """Randomly get another item.

        Returns:
            int: Another index of item.
        """
        return np.random.choice(self.__len__)

    def __getitem__(self, idx):
        """Get item from infos according to the given index.

        Returns:
            dict: Data dictionary of the corresponding index.
        """
        data = self.prepare_data(idx)
        return data

    def evaluate(self,
                 results,
                 logger=None,
                 name=None,
                 **kwargs):
        '''
        Args:
            results (list[Tensor]): List of results.
            visualize_cfg (Dict): Config of test dataset.
            output_format (str): Model output format, should be either 'raster' or 'vector'.

        Returns:
            dict: Evaluation results.
        '''
        return dict(loss=314.0)
        # name = 'results_tier4' if name is None else name
        # result_path = self.format_results(
        #     results, name, prefix=self.work_dir, patch_size=self.eval_cfg.patch_size, origin=self.eval_cfg.origin)

        # self.eval_cfg.evaluation_cfg['result_path'] = result_path
        # # self.eval_cfg.evaluation_cfg['ann_file'] = self.ann_file

        # mean_ap = eval_chamfer(
        #     self.eval_cfg.evaluation_cfg, update=True, logger=logger)

        # # Create debug image
        # debug_token_id = results[0]['token']
        # debug_idx = np.where(np.array(self.samples) == debug_token_id)[0][0]

        # result = results[0]
        # vectors_pred = list()
        # for i in range(len(result['lines'])):
        #     label = result['labels'][i]
        #     score = result['scores'][i]
        #     if score < self.score_th:
        #         continue
        #     data = (result['lines'][i], len(result['lines'][i]), label)
        #     vectors_pred.append(data)

        # debug_img = self.visualize_data(debug_idx, vectors_pred=vectors_pred).transpose((2, 0, 1))

        # print('len of the results', len(results))
        # result_dict = {
        #     'mAP': mean_ap,
        #     # 'image_test': torch.full((3, 128, 128), 128, dtype=torch.uint8)
        #     'image/groundtruth': debug_img,
        # }
        # return result_dict

    # def visualize_data(self, idx, vectors_pred=None):
    #     data = self.__getitem__(idx)

    #     imgs = data['img'].data
    #     vectors = data['vectors'].data
    #     # intrinsics = data['img_metas'].data['cam_intrinsics']
    #     # extrinsics = data['img_metas'].data['cam_extrinsics']
    #     ego2img_list = data['img_metas'].data['ego2img']

    #     cat2id = self.visualize_cfg['class2label']
    #     roi_size = self.visualize_cfg['roi_size']
    #     renderer = Renderer(cat2id, roi_size, self.visualize_cfg['img_norm_cfg'], CAMERA_NAMES)

    #     img = renderer.render_all(vectors,
    #                               vectors_pred,
    #                               imgs,
    #                               ego2img_list,
    #                               thickness=3,
    #                               save_image=False,
    #                               return_image=True)
    #     return img

    def format_results(self, results, name, prefix=None, patch_size=(60, 30), origin=(0, 0)):

        meta = False
        submissions = {
            'meta': meta,
            'results': {},
            "groundTruth": {},  # for validation
        }
        patch_size = np.array(patch_size)
        origin = np.array(origin)

        for case in mmcv.track_iter_progress(results):
            '''
                vectorized_line {
                    "pts":               List[<float, 2>]  -- Ordered points to define the vectorized line.
                    "pts_num":           <int>,            -- Number of points in this line.
                    "type":              <0, 1, 2>         -- Type of the line: 0: ped; 1: divider; 2: boundary
                    "confidence_level":  <float>           -- Confidence level for prediction (used by Average Precision)
                }
            '''

            if case is None:
                continue

            vector_lines = []
            for i in range(case['nline']):
                vector = case['lines'][i] * patch_size + origin
                vector_lines.append({
                    'pts': vector,
                    'pts_num': len(case['lines'][i]),
                    'type': case['labels'][i],
                    'confidence_level': case['scores'][i],
                })
                submissions['results'][case['token']] = {}
                submissions['results'][case['token']]['vectors'] = vector_lines

            if 'groundTruth' in case:
                submissions['groundTruth'][case['token']] = {}
                vector_lines = []
                for i in range(case['groundTruth']['nline']):
                    line = case['groundTruth']['lines'][i] * \
                        patch_size + origin

                    vector_lines.append({
                        'pts': line,
                        'pts_num': len(case['groundTruth']['lines'][i]),
                        'type': case['groundTruth']['labels'][i],
                        'confidence_level': 1.,
                    })
                submissions['groundTruth'][case['token']
                                           ]['vectors'] = vector_lines

        # Use pickle format to minimize submission file size.
        print('Done!')
        mmcv.mkdir_or_exist(prefix)
        res_path = os.path.join(prefix, '{}.pkl'.format(name))
        mmcv.dump(submissions, res_path)

        return res_path