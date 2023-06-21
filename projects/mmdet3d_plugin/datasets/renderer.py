import os.path as osp
import os
import mmcv
from IPython import embed
from pathlib import Path
import av2.geometry.interpolate as interp_utils
import numpy as np
import copy
import cv2
import io
import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

# COLOR_MAPS_BGR = {
#     # bgr colors
#     'divider': (0, 0, 255),
#     'boundary': (0, 255, 0),
#     'ped_crossing': (255, 0, 0),
#     'centerline': (51, 183, 255),
#     'drivable_area': (171, 255, 255),
#     'others': (128, 128, 128),  # gray
#     'contours': (255, 255, 51),  # yellow
# }

COLOR_MAPS_RGB = {
    'divider': (255, 0, 0),
    'boundary': (0, 255, 0),
    'ped_crossing': (0, 0, 255),
    'stop_line': (255, 255, 0),
    'centerline': (255, 183, 51),
    'drivable_area': (255, 255, 171),
    'others': (128, 128, 128),  # gray
    'contours': (51, 255, 255),  # yellow
}

COLOR_MAPS_PLT = {
    'divider': 'r',
    'boundary': 'g',
    'ped_crossing': 'b',
    'stop_line': 'y',
    'centerline': 'orange',
    'drivable_area': 'y',
    'others': 'gray',
    'contours': 'cyan',
}

CAM_NAMES_AV2 = ['ring_front_center', 'ring_front_right', 'ring_front_left',
    'ring_rear_right','ring_rear_left', 'ring_side_right', 'ring_side_left',
    ]

CAM_ORDER = [
    2, 0, 4,
    3, 1, 5
]

def remove_nan_values(uv):
    is_u_valid = np.logical_not(np.isnan(uv[:, 0]))
    is_v_valid = np.logical_not(np.isnan(uv[:, 1]))
    is_uv_valid = np.logical_and(is_u_valid, is_v_valid)

    uv_valid = uv[is_uv_valid]
    return uv_valid

# def points_ego2img(pts_ego, extrinsics, intrinsics):
#     pts_ego_4d = np.concatenate([pts_ego, np.ones([len(pts_ego), 1])], axis=-1)
#     pts_cam_4d = extrinsics @ pts_ego_4d.T
#     uv = (intrinsics @ pts_cam_4d[:3, :]).T
#     uv = remove_nan_values(uv)
#     depth = uv[:, 2]
#     uv = uv[:, :2] / uv[:, 2].reshape(-1, 1)
#     return uv, depth

# def points_ego2img(pts_ego, extrinsics, intrinsics):
#     viewpad = np.eye(4)
#     viewpad[:intrinsics.shape[0], :intrinsics.shape[1]] = intrinsics
#     ego2cam_rt = (viewpad @ extrinsics)
#     print("HOGE ", ego2cam_rt)
#     return points_ego2img_from_one_matrix(
#         pts_ego,
#         ego2cam_rt)

def points_ego2img_from_one_matrix(pts_ego, ego2img):
    pts_ego_4d = np.concatenate([pts_ego, np.ones([len(pts_ego), 1])], axis=-1)
    uv = pts_ego_4d @ ego2img.T
    uv = remove_nan_values(uv)
    depth = uv[:, 2]
    uv = uv[:, :2] / uv[:, 2].reshape(-1, 1)
    return uv, depth

def draw_polyline_ego_on_img(polyline_ego, img_bgr, ego2img, color_bgr, thickness):
    if polyline_ego.shape[1] == 2:
        zeros = np.zeros((polyline_ego.shape[0], 1))
        polyline_ego = np.concatenate([polyline_ego, zeros], axis=1)
    polyline_ego = interp_utils.interp_arc(t=500, points=polyline_ego)

    # uv, depth = points_ego2img(polyline_ego, extrinsics, intrinsics)
    uv, depth = points_ego2img_from_one_matrix(polyline_ego, ego2img)

    h, w, c = img_bgr.shape

    is_valid_x = np.logical_and(0 <= uv[:, 0], uv[:, 0] < w - 1)
    is_valid_y = np.logical_and(0 <= uv[:, 1], uv[:, 1] < h - 1)
    is_valid_z = depth > 0
    is_valid_points = np.logical_and.reduce([is_valid_x, is_valid_y, is_valid_z])

    if is_valid_points.sum() == 0:
        return
    
    uv = np.round(uv[is_valid_points]).astype(np.int32)

    draw_visible_polyline_cv2(
        copy.deepcopy(uv),
        valid_pts_bool=np.ones((len(uv), 1), dtype=bool),
        image=img_bgr,
        color=color_bgr,
        thickness_px=thickness,
    )

def draw_visible_polyline_cv2(line, valid_pts_bool, image, color, thickness_px):
    """Draw a polyline onto an image using given line segments.

    Args:
        line: Array of shape (K, 2) representing the coordinates of line.
        valid_pts_bool: Array of shape (K,) representing which polyline coordinates are valid for rendering.
            For example, if the coordinate is occluded, a user might specify that it is invalid.
            Line segments touching an invalid vertex will not be rendered.
        image: Array of shape (H, W, 3), representing a 3-channel BGR image
        color: Tuple of shape (3,) with a BGR format color
        thickness_px: thickness (in pixels) to use when rendering the polyline.
    """
    line = np.round(line).astype(int)  # type: ignore
    for i in range(len(line) - 1):

        if (not valid_pts_bool[i]) or (not valid_pts_bool[i + 1]):
            continue

        x1 = line[i][0]
        y1 = line[i][1]
        x2 = line[i + 1][0]
        y2 = line[i + 1][1]

        # Use anti-aliasing (AA) for curves
        # print(f'({x1}, {y1}) to ({x2}, {y2})')
        image = cv2.line(image, pt1=(x1, y1), pt2=(x2, y2), color=color, thickness=thickness_px, lineType=cv2.LINE_AA)

def add_label(image, text, position=(20,20), font=cv2.FONT_HERSHEY_SIMPLEX, 
              font_scale=0.7, color=(255,255,255), thickness=2):
    """
    Add a label (text) to an image.

    Args:
        image: The image to add the label to.
        text: The text to add as a label.
        position: A tuple (x, y) representing where to put the text on the image.
        font: Font type from OpenCV.
        font_scale: Font size.
        color: Text color.
        thickness: Text line thickness.

    Returns:
        The image with the label added.
    """
    return cv2.putText(image, text, position, font, font_scale, color, thickness, cv2.LINE_AA)

def dictionalize(vectors_list):
    """
    Transforms a list of tuples into a dictionary.
    
    Each tuple in the list contains a polyline, its length, and a label. The function groups 
    the polylines by their labels and returns a dictionary where keys are labels and values are
    lists of polylines associated with each label.

    Parameters:
    vectors_list (List[Tuple]): The list of tuples containing the polyline, line length, and label.

    Returns:
    dict: A dictionary grouping polylines by their labels.
    """
    vectors_dict = {}
    for polyline, line_length, label in vectors_list:
        if label not in vectors_dict:
            vectors_dict[label] = []
        vectors_dict[label].append(polyline)
    return vectors_dict

class Renderer(object):
    def __init__(self, cat2id, roi_size, img_norm_cfg, cam_names=CAM_NAMES_AV2):
        self.roi_size = roi_size
        self.cat2id = cat2id
        self.id2cat = {v: k for k, v in cat2id.items()}
        self.cam_names = cam_names
        self.img_norm_cfg = img_norm_cfg
    
    def denormalize_vectors(self, vectors):
        """
        vectors: dict
            {
                label1: [polyline1, polyline2, ...],
                label2: [polyline3, polyline4, ...],
                ...
            }
        """
        vectors_denormalized = copy.deepcopy(vectors)
        size = np.array([self.roi_size[0], self.roi_size[1]]) + 2
        origin = -np.array([self.roi_size[0]/2, self.roi_size[1]/2])
        for label, polylines in vectors_denormalized.items():
            for i, polyline in enumerate(polylines):
                polyline_denormalized = copy.deepcopy(polyline)
                polyline_denormalized[:, :2] = polyline_denormalized[:, :2] * size + origin
                # for point in polyline:
                #     x = point[0] * size[0] + origin[0]
                #     y = point[1] * size[1] + origin[1]
                # polyline_denormalized.append(polyline_denormalized)
                vectors_denormalized[label][i] = np.array(polyline_denormalized)
        return vectors_denormalized

    def render_all(self,
                   vectors_normalized_list,
                   vectors_normalized_pred_list,
                   imgs,
                   ego2img_list,
                   thickness,
                   out_dir='',
                   return_image=True,
                   save_image=True):
        vectors_normalized_dict = dictionalize(vectors_normalized_list)
        vectors_normalized_pred_dict = dictionalize(vectors_normalized_pred_list)
        vectors_dict = self.denormalize_vectors(vectors_normalized_dict)
        vectors_pred_dict = self.denormalize_vectors(vectors_normalized_pred_dict)

        fig = plt.figure(figsize=(14, 15))

        # Create a grid for the plots
        gs = gridspec.GridSpec(6, 3)

        out_path = Path(out_dir) / 'debug.jpg'

        for i, cam_id in enumerate(CAM_ORDER):
            img = imgs[cam_id]
            ego2img = ego2img_list[cam_id]
            img_bgr = copy.deepcopy(img.numpy().transpose((1, 2, 0)))
            img_bgr = mmcv.imdenormalize(
                img_bgr,
                mean=np.array(self.img_norm_cfg['mean']),
                std=np.array(self.img_norm_cfg['std']))
            img_bgr = self.draw_vectors_dict_on_img(img_bgr, vectors_dict, ego2img, thickness)
            img_bgr = add_label(img_bgr, f'camera {cam_id} (Ground truth)')

            ax = fig.add_subplot(gs[i // 3, i % 3])
            ax.imshow(img_bgr.astype(np.uint8))
            ax.axis('off')

        for i, cam_id in enumerate(CAM_ORDER):
            img = imgs[cam_id]
            ego2img = ego2img_list[cam_id]
            img_bgr = copy.deepcopy(img.numpy().transpose((1, 2, 0)))
            img_bgr = mmcv.imdenormalize(
                img_bgr,
                mean=np.array(self.img_norm_cfg['mean']),
                std=np.array(self.img_norm_cfg['std']))
            img_bgr = self.draw_vectors_dict_on_img(img_bgr, vectors_pred_dict, ego2img, thickness)
            img_bgr = add_label(img_bgr, f'camera {cam_id} (Predicted)')

            ax = fig.add_subplot(gs[i // 3 + 2, i % 3])
            ax.imshow(img_bgr.astype(np.uint8))
            ax.axis('off')

        ax_bev = fig.add_subplot(gs[4:, :])
        self.plot_bev_from_vectors(ax_bev, vectors_dict, vectors_pred_dict)
        ax_bev.axis('off')

        # To remove space between images
        plt.subplots_adjust(wspace=0.01, hspace=0.01)

        if save_image:
            plt.savefig(out_path, dpi=300)
        if return_image:
            buf = io.BytesIO()
            plt.savefig(buf, format='png', transparent=False)
            buf.seek(0)

            img = Image.open(buf).convert("RGB")
            img_arr = np.array(img)
            buf.close()
            return img_arr
        
    def render_ground_truth(self,
                            vectors_normalized_list,
                            imgs,
                            ego2img_list,
                            thickness,
                            out_dir='',
                            return_image=True,
                            save_image=True):
        vectors_normalized_dict = dictionalize(vectors_normalized_list)
        vectors_dict = self.denormalize_vectors(vectors_normalized_dict)

        fig = plt.figure(figsize=(14, 10))

        # Create a grid for the plots
        gs = gridspec.GridSpec(4, 3)

        out_path = Path(out_dir) / 'debug.jpg'

        for i, cam_id in enumerate(CAM_ORDER):
            img = imgs[cam_id]
            ego2img = ego2img_list[cam_id]
            img_bgr = copy.deepcopy(img.numpy().transpose((1, 2, 0)))
            img_bgr = mmcv.imdenormalize(
                img_bgr,
                mean=np.array(self.img_norm_cfg['mean']),
                std=np.array(self.img_norm_cfg['std']))
            img_bgr = self.draw_vectors_dict_on_img(img_bgr, vectors_dict, ego2img, thickness)
            img_bgr = add_label(img_bgr, f'camera {cam_id} (Ground truth)')

            ax = fig.add_subplot(gs[i // 3, i % 3])
            ax.imshow(img_bgr.astype(np.uint8))
            ax.axis('off')

        ax_bev = fig.add_subplot(gs[2:, :])
        self.plot_bev_from_vectors(ax_bev, vectors_dict)
        ax_bev.axis('off')

        # To remove space between images
        plt.subplots_adjust(wspace=0.01, hspace=0.01)

        if save_image:
            plt.savefig(out_path, dpi=300)
        if return_image:
            buf = io.BytesIO()
            plt.savefig(buf, format='png', transparent=False)
            buf.seek(0)

            img = Image.open(buf).convert("RGB")
            img_arr = np.array(img)
            buf.close()
            return img_arr

    def draw_vectors_dict_on_img(self, img, vectors_dict, ego2img, thickness):
        for label, vector_list in vectors_dict.items():
            cat = self.id2cat[label]
            color = COLOR_MAPS_RGB[cat]
            for vector in vector_list:
                img = np.ascontiguousarray(img)
                draw_polyline_ego_on_img(vector, img, ego2img, 
                    color, thickness)    
        return img

    def plot_bev_from_vectors(self, ax, vectors_in_dict, vectors_in_pred_dict=None):
        vectors_dict = copy.deepcopy(vectors_in_dict)
        vectors_pred_dict = copy.deepcopy(vectors_in_pred_dict)

        car_img = Image.open('car.png')

        ax.set_xlim(-self.roi_size[0] / 2, self.roi_size[0] / 2)
        ax.set_ylim(-self.roi_size[1] / 2, self.roi_size[1] / 2)
        ax.axis('off')
        ax.imshow(car_img, extent=[-1.5, 1.5, -1.2, 1.2])

        # Add grid lines
        num_x_lines_10m = int(self.roi_size[0] // 10)
        for i in range(-num_x_lines_10m, num_x_lines_10m + 1):
            ax.axvline(x=i * 10, color='black', linestyle='--', linewidth=0.2, alpha=0.9)
        num_y_lines_10m = int(self.roi_size[1] // 10)
        for i in range(-num_y_lines_10m, num_y_lines_10m + 1):
            ax.axhline(y=i * 10, color='black', linestyle='--', linewidth=0.2, alpha=0.9)
        num_x_lines_1m = int(self.roi_size[0])
        for i in range(-num_x_lines_1m, num_x_lines_1m + 1):
            ax.axvline(x=i * 1, color='black', linestyle='--', linewidth=0.2, alpha=0.3)
        num_y_lines_1m = int(self.roi_size[1])
        for i in range(-num_y_lines_1m, num_y_lines_1m + 1):
            ax.axhline(y=i * 1, color='black', linestyle='--', linewidth=0.2, alpha=0.3)

        for label, vector_list in vectors_dict.items():
            cat = self.id2cat[label]
            color = COLOR_MAPS_PLT[cat]
            for vector in vector_list:
                pts = vector[:, :2]
                x = np.array([pt[0] for pt in pts])
                y = np.array([pt[1] for pt in pts])
                # plt.quiver(x[:-1], y[:-1], x[1:] - x[:-1], y[1:] - y[:-1], angles='xy', color=color,
                #     scale_units='xy', scale=1)
                # ax.plot(x, y, color='black', linewidth=1, linestyle='-') # only line
                ax.plot(x, y, color=color, linewidth=1, linestyle='-') # only line

        if vectors_pred_dict is not None:
            for label, vector_list in vectors_pred_dict.items():
                cat = self.id2cat[label]
                color = COLOR_MAPS_PLT[cat]
                for vector in vector_list:
                    pts = vector[:, :2]
                    x = np.array([pt[0] for pt in pts])
                    y = np.array([pt[1] for pt in pts])
                    # plt.quiver(x[:-1], y[:-1], x[1:] - x[:-1], y[1:] - y[:-1], angles='xy', color=color,
                    #     scale_units='xy', scale=1)
                    ax.plot(x, y, color=color, linewidth=4, marker='o', linestyle='--', markersize=0.3, alpha=0.6)
        ax.text(-28, 12, 'Black line: Ground truth\nDashed line: Predicted', fontsize=12)
        ax.grid()
