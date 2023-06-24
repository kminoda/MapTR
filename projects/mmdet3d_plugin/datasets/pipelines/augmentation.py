import numpy as np
from mmdet.datasets.builder import PIPELINES
import warnings
import itertools
from typing import Dict, List, Tuple
import albumentations as A
from scipy.spatial.transform import Rotation as R
import cv2
warnings.filterwarnings("ignore")

@PIPELINES.register_module(force=True)
class Augmentation(object):
    def __init__(self,
                 **kwargs):
        super().__init__()

        self.transform = A.Compose([
            A.HueSaturationValue(
                hue_shift_limit=50,
                sat_shift_limit=30,
                val_shift_limit=60,
                p=0.8
            ),
            A.Cutout(
                num_holes=4,
                max_h_size=64,
                max_w_size=64,
                fill_value=0,
                p=1.0
            ),
            A.RandomFog(
                fog_coef_upper=0.5,
                alpha_coef=0.05,
                p=0.5
            ),
            A.RandomSunFlare(
                src_radius=150,
                p=0.5
            )
        ])

    def __call__(self, input_dict: dict) -> dict:
        input_dict['img'] = np.array([img.astype(np.uint8) for img in input_dict['img']])
        input_dict['img'] = np.array([self.transform(image=img)['image'] for img in input_dict['img']])

        # input_dict = self.rotate_bev(input_dict, angle_lim=np.pi/8, p=0.5)
        # input_dict = self.translate_bev(input_dict, trans_lim=10.0, p=1.0)
        # input_dict = self.affine_camera(input_dict, p_angle = 1.0, p_scale = 0.5)
        # input_dict = self.random_drop(input_dict, p=0.5)
        # input_dict = self.shuffle_image(input_dict, p=1.0)
        return input_dict

    def rotate_bev(self, input_dict: dict, angle_lim: float = np.pi / 16, p = 1.0) -> dict:
        if p < np.random.rand():
            return input_dict
        rot = np.random.rand() * angle_lim * 2 - angle_lim
        mat = np.array([
            [np.cos(rot), -np.sin(rot), 0, 0],
            [np.sin(rot), np.cos(rot), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        # Rotate pose
        pose = input_dict['global2ego_pose']
        pose_rotated = np.linalg.inv(mat[:3, :3]) @ R.from_quat(pose[3:]).as_matrix()
        input_dict['global2ego_pose'][3:] = R.from_matrix(pose_rotated).as_quat()

        # Rotate extrinsics
        num_cameras = len(input_dict['cam_extrinsics'])
        for i in range(num_cameras):
            ext = input_dict['cam_extrinsics'][i].copy()
            input_dict['cam_extrinsics'][i] = np.linalg.inv(mat @ np.linalg.inv(ext))

        # Modify ego2img
        ego2img_rts = []
        for i in range(num_cameras):
            extrinsic = input_dict['cam_extrinsics'][i].copy()
            intrinsic = input_dict['cam_intrinsics'][i].copy()
            ego2cam_rt = extrinsic
            viewpad = np.eye(4)
            viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
            ego2cam_rt = (viewpad @ ego2cam_rt)
            ego2img_rts.append(ego2cam_rt)
        input_dict['ego2img'] = ego2img_rts

        return input_dict

    def translate_bev(self, input_dict: dict, trans_lim: float = 3, p = 1.0) -> dict:
        if p < np.random.rand():
            return input_dict
        trans_x = np.random.rand() * trans_lim * 2 - trans_lim
        trans_y = np.random.rand() * trans_lim * 2 - trans_lim

        trans_vec = np.array([trans_x, trans_y, 0])
        pose = input_dict['global2ego_pose']
        trans_glob = R.from_quat(pose[3:]).as_matrix() @ trans_vec.T

        # Translate pose
        input_dict['global2ego_pose'][0] += trans_glob[0]
        input_dict['global2ego_pose'][1] += trans_glob[1]

        # Translate extrinsics
        trans_mat = np.eye(4)
        trans_mat[0, 3] += trans_x
        trans_mat[1, 3] += trans_y
        num_cameras = len(input_dict['cam_extrinsics'])
        for i in range(num_cameras):
            ext = input_dict['cam_extrinsics'][i].copy()
            input_dict['cam_extrinsics'][i] = ext @ trans_mat

        # Modify ego2img
        ego2img_rts = []
        for i in range(num_cameras):
            extrinsic = input_dict['cam_extrinsics'][i].copy()
            intrinsic = input_dict['cam_intrinsics'][i].copy()
            ego2cam_rt = extrinsic
            viewpad = np.eye(4)
            viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
            ego2cam_rt = (viewpad @ ego2cam_rt)
            ego2img_rts.append(ego2cam_rt)
        input_dict['ego2img'] = ego2img_rts

        return input_dict

    def affine_camera(self,
                      input_dict: dict,
                      angle_lim: float = np.pi / 30,
                      scale_lower_lim: float = 0.7,
                    #   trans_lim: float = 50,
                      p_angle = 1.0,
                      p_scale = 0.5,
                    #   p_trans = 1.0
                      ) -> dict:
        num_cameras = len(input_dict['cam_extrinsics'])
        for i in range(num_cameras):
            img_origin = input_dict['img'][i].copy()
            ext_origin = input_dict['cam_extrinsics'][i].copy()
            int_origin = input_dict['cam_intrinsics'][i].copy()

            ###
            angle_rad = 0
            scale = 1
            # trans = (0, 0)
            if p_angle > np.random.rand():
                angle_rad = np.random.rand() * angle_lim * 2 - angle_lim
            if p_scale > np.random.rand():
                scale = np.random.rand() * (1 - scale_lower_lim) + scale_lower_lim
            # if p_trans > np.random.rand():
            #     trans_x = np.random.rand() * trans_lim * 2 - trans_lim
            #     trans_y = np.random.rand() * trans_lim * 2 - trans_lim
            #     trans = (trans_x, trans_y)

            center = np.array([int_origin[0, 2], int_origin[1, 2]])
            input_dict['img'][i] = affine_image(img_origin, angle_rad, scale, center)
            input_dict['cam_extrinsics'][i] = np.linalg.inv(
                rotate_by_optical_axis(np.linalg.inv(ext_origin), angle_rad))
            # input_dict['cam_intrinsics'][i][0, 2] += trans[0]
            # input_dict['cam_intrinsics'][i][1, 2] += trans[1]
            input_dict['cam_intrinsics'][i][:2, :2] *= scale

        # Modify ego2img
        ego2img_rts = []
        for i in range(num_cameras):
            extrinsic = input_dict['cam_extrinsics'][i].copy()
            intrinsic = input_dict['cam_intrinsics'][i].copy()
            ego2cam_rt = extrinsic
            viewpad = np.eye(4)
            viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
            ego2cam_rt = (viewpad @ ego2cam_rt)
            ego2img_rts.append(ego2cam_rt)
        input_dict['ego2img'] = ego2img_rts

        return input_dict
    
    def random_drop(self,
                    input_dict: dict,
                    p=0.5) -> dict:      
        if p < np.random.rand():
            return input_dict

        drop_idx = np.random.choice(6, 1)[0]
        input_dict['img'][drop_idx] *= 0
        return input_dict

    def shuffle_image(self,
                      input_dict: dict,
                      p=1.0) -> dict:      
        if p < np.random.rand():
            return input_dict

        permute_ids = np.random.permutation(6)
        input_dict['img'] = [input_dict['img'][i] for i in permute_ids]
        input_dict['cam_extrinsics'] = [input_dict['cam_extrinsics'][i] for i in permute_ids]
        input_dict['cam_intrinsics'] = [input_dict['cam_intrinsics'][i] for i in permute_ids]
        input_dict['ego2img'] = [input_dict['ego2img'][i] for i in permute_ids]
        return input_dict

def rotate_by_optical_axis(matrix, angle):
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle), 0, 0],
        [np.sin(angle), np.cos(angle), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    matrix_rotated = matrix @ rotation_matrix
    return matrix_rotated

def affine_image(image_input, angle, scale, center, trans=(0, 0)):
    image = image_input.copy()
    h, w = image.shape[:2]

    # Get rotation matrix
    mat_rot = cv2.getRotationMatrix2D(center, np.rad2deg(angle), scale)

    # Create translation matrix
    tx, ty = trans
    mat_trans = np.float32([[1, 0, tx], [0, 1, ty]])

    # Combine matrices
    mat = mat_trans @ np.vstack([mat_rot, [0, 0, 1]])

    image_transformed = cv2.warpAffine(image, mat[:2, :], (w, h))
    return image_transformed