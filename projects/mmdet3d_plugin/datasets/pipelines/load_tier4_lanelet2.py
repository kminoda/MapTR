import mmcv
import copy
import json
import cv2
import numpy as np
from pathlib import Path
from mmcv.parallel import DataContainer as DC
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import to_tensor

import warnings
import lanelet2
from lanelet2.projection import UtmProjector
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple
from scipy.spatial.transform import Rotation as R
from shapely.geometry import LineString
from ..lidar_instance_lines import LiDARInstanceLines

warnings.filterwarnings("ignore")

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


def get_possible_map_names_from_dataset(dataroot):
    metadata_path = Path(dataroot) / 'metadata.json'
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    map_name_set = set()
    for val in metadata.values():
        map_name_set.add(val['map_name'])
    return map_name_set

@PIPELINES.register_module(force=True)
class VectorizeLanelet2Map(object):

    def __init__(self,
                 maproot='/mount_hdd/data/maps/',
                 dataroot='',
                 extract_distance=250,
                 roi_size=(60, 30),
                 line_type_mapping=dict(),
                 class2label=dict(),
                 output_type="MapTR",
                 **kwargs):
        super().__init__()

        map_name_list = get_possible_map_names_from_dataset(dataroot)
        self.lanelet2_maps = {}
        for map_name in map_name_list:
            map_path = Path(maproot) / map_name / 'lanelet2_map.osm'
            self.lanelet2_maps[map_name] = load_lanelet2_my(map_path)
            print(f'Finished loading map {map_path}')

        # self.normalize = normalize
        self.roi_size = roi_size
        self.size = np.array([self.roi_size[0], self.roi_size[1]]) + 2
        self.line_type_mapping = line_type_mapping
        self.label2class = {v:k for k, v in class2label.items()}
        self.output_type = output_type

        # ##################### TODO #####################
        # self.distance = np.max(roi_size) / 2
        # ##################### TODO #####################

    def __call__(self, input_dict: dict) -> dict:
        if self.output_type == 'MapTR':
            return self.load_maptr_type(input_dict)
        elif self.output_type == "VectorMapNet":
            return self.load_vectormapnet_type(input_dict)
        else:
            raise NotImplementedError
    
    def load_maptr_type(self, input_dict: dict) -> dict:
        pose = input_dict['global2ego_pose']
        map_name = input_dict['map_name']
        vectors_dict_3d = self.vectorization(pose, map_name, normalize=False)

        gt_labels = []
        gt_instance = []
        for label, polylines_3d in vectors_dict_3d.items():
            for polyline_3d in polylines_3d:
                polyline_shapely_2d = LineString(polyline_3d[:, [1, 0]])
                gt_instance.append(polyline_shapely_2d)
                gt_labels.append(label)
        gt_instance = LiDARInstanceLines(
            gt_instance,
            sample_dist=1,
            num_samples=250,
            padding=False,
            fixed_num=20,
            padding_value=-10000,
            patch_size=self.roi_size)

        input_dict['gt_labels_3d'] = DC(to_tensor(gt_labels), cpu_only=False)
        input_dict['gt_bboxes_3d'] = DC(gt_instance, cpu_only=True)
        # input_dict['gt_labels_3d'] = to_tensor(gt_labels)
        # input_dict['gt_bboxes_3d'] = gt_instance

        return input_dict

    def load_vectormapnet_type(self, input_dict: dict) -> dict:
        pose = input_dict['global2ego_pose']
        map_name = input_dict['map_name']
        vectors_dict_3d = self.vectorization(pose, map_name, normalize=True)
        vectors_dict_2d = {label: [
                np.array(polyline)[:, :2] for polyline in polylines
            ] for label, polylines in vectors_dict_3d.items()}
        vectors_list_2d = self.flatten_vectors_dict(vectors_dict_2d)

        # input_dict['vectors_dict'] = vectors_dict_2d
        input_dict['vectors_dict'] = vectors_dict_3d
        input_dict['vectors'] = vectors_list_2d
        return input_dict

    def vectorization(self, pose: List[float], map_name: str, normalize: bool) -> Dict[int, List[List[Tuple[float, float]]]]:
        assert len(pose) == 7, "pose should be [px, py, pz, qx, qy, qz, qw]"

        ####
        # pose[0], pose[1] = 89380.01926014903, 43154.05710192098  # For debug
        # print("KOJI!!! ", pose[0], pose[1])
        ####

        lanelet2_map = self.lanelet2_maps[map_name]

        vectors = self.extract_lanelets(lanelet2_map, self.roi_size, pose, normalize=normalize)

        length = np.sum([len(polylines) for polylines in vectors.values()])
        if length == 0:
            print(f'No lanelet2 found at pose = ({pose[0]}, {pose[1]})')
        return vectors

    def generate_random_lanelet_image(self, pose: List[float], map_name: str, random_xy=30, random_yaw_deg=360):
        assert len(pose) == 7, "pose should be [px, py, pz, qx, qy, qz, qw]"

        lanelet2_map = self.lanelet2_maps[map_name]

        # Add random value to x, y, and yaw
        random_delta_xy = np.random.uniform(-random_xy, random_xy, size=2)
        random_delta_yaw_deg = np.random.uniform(-random_yaw_deg, random_yaw_deg)

        # Add random value to x, y, and yaw
        random_pose = pose.copy()
        random_pose[0] += random_delta_xy[0] # px
        random_pose[1] += random_delta_xy[1] # py

        # Convert quaternion to euler
        rotation = R.from_quat(pose[3:7])
        euler = rotation.as_euler('xyz')
        # Add random yaw
        euler[2] += np.deg2rad(random_delta_yaw_deg)
        # Convert back to quaternion
        random_pose[3:7] = R.from_euler('xyz', euler).as_quat()

        # Extract lanelet2
        roi_size = (max(self.roi_size) + random_xy, max(self.roi_size) + random_xy)
        vectors = self.extract_lanelets(lanelet2_map,
                                        roi_size=roi_size,
                                        pose=random_pose,
                                        normalize=True)
        lanelet2_img = self.draw_vectors_on_image(vectors, img_size=(256, 256))
        return lanelet2_img

    def extract_lanelets(self, lanelet2_map, roi_size, pose, normalize):
        distance = np.max(roi_size) / 2
        vectors = {}

        for line_string in lanelet2_map.lineStringLayer:
            # Check if the line type is of interest
            way_type = line_string.attributes['type']
            relation_subtype = get_relation_subtype(lanelet2_map, line_string)
            line_type = f'{way_type}:{relation_subtype}'
            if line_type in self.line_type_mapping.keys():
                if self.line_type_mapping[line_type] == -1:
                    continue
            else:
                continue
            label = self.line_type_mapping[line_type]
            polyline_all = line_string2list3d(line_string)

            polylines = extract_near_polylines(pose, polyline_all, distance)
            # if 'stop_line' in line_type:
            #     print(f'{line_type} found, and extracted length: {len(polylines)}')

            for polyline in polylines:
                polyline_local = global2local(polyline, pose)
                if label not in vectors.keys():
                    vectors[label] = []
                polylines_local_filtered = filter_polyline_by_roi_size(polyline_local, roi_size)
                # print('===================KOJI====================')
                # if len(polylines_local_filtered) == 0: ## DEBUG
                #     _ = filter_polyline_by_roi_size(polyline_local, roi_size, debug=True)
                #     # import matplotlib.pyplot as plt
                #     # plt.plot(np.array(polyline_local)[:, 0], np.array(polyline_local)[:, 1], marker='x', linestyle='--')
                #     # plt.savefig('/mount_hdd/vectormap_ws/test.png')
                if normalize:
                    polylines_local_filtered = [normalize_line(p, roi_size) for p in polylines_local_filtered]
                for polyline_local_filtered in polylines_local_filtered:
                    vectors[label].append(np.array(polyline_local_filtered))
        return vectors

    def flatten_vectors_dict(self, vectors_dict):
        vectors_list = []
        for label, polylines in vectors_dict.items():
            for polyline in polylines:
                vectors_list.append((polyline, len(polyline), label))
        return vectors_list

    def draw_vectors_on_image(self, vectors, img_size, line_thickness=2):
        # Create blank white image
        img = np.ones((img_size[0], img_size[1], 3), np.uint8) * 255

        # Draw each vector on the image
        for label, polylines in vectors.items():
            color = COLOR_MAPS_RGB.get(self.label2class[label], (0, 0, 0))  # Default to black if label color not defined
            for polyline in polylines:
                # Ignore z coordinate, and round and convert coordinates to integer for OpenCV
                polyline_xy = polyline[:,:2]

                # Normalize xy from roi_bbox to img_size
                polyline_xy[:, 0] *= img_size[0]
                polyline_xy[:, 1] *= img_size[1]
                polyline_xy = np.round(polyline_xy).astype(int)

                # Draw polyline on image
                cv2.polylines(img, [polyline_xy], isClosed=False, color=color, thickness=line_thickness)

        return img

def normalize_line(line, roi_size):
    '''
        prevent extrime pts such as 0 or 1. 
    '''
    size = np.array([roi_size[0], roi_size[1]]) + 2
    
    origin = -np.array([roi_size[0]/2, roi_size[1]/2])
    # for better learning
    line = np.array(line)
    line[:, :2] = line[:, :2] - origin
    line[:, :2] = line[:, :2] / size
    line = line.tolist()

    return line

def filter_polyline_by_roi_size(polyline: List[List[float]], roi_size: Tuple[int, int], debug=False) -> List[List[List[float]]]:
    polylines_filtered = []
    polyline_tmp = []
    if debug:
        print(roi_size)
    for i in range(len(polyline)):
        point_within_area = abs(polyline[i][0]) < roi_size[0] / 2 and abs(polyline[i][1]) < roi_size[1] / 2
        if i == 0:
            prev_point_within_area = point_within_area
        
        cur_point = polyline[i]
        prev_point = None if i == 0 else polyline[i - 1]

        if point_within_area and prev_point_within_area: # o-o -> add current point
            polyline_tmp.append(cur_point)
            if debug:
                print(f'{cur_point}: o-o')
        elif (not point_within_area) and prev_point_within_area and prev_point is not None: # o-x -> add interval point
            interval_point = rectangle_line_segment_intersection(cur_point, prev_point, roi_size)[0]
            polyline_tmp.append(interval_point)
            polylines_filtered.append(polyline_tmp)
            if debug:
                print('here!1', len(polyline_tmp))
            polyline_tmp = []
            if debug:
                print(f'{cur_point}: o-x')
        elif point_within_area and (not prev_point_within_area) and prev_point is not None: # x-o -> add interval point and current point
            interval_point = rectangle_line_segment_intersection(cur_point, prev_point, roi_size)[0]
            polyline_tmp.append(interval_point)
            polyline_tmp.append(cur_point)
            if debug:
                print(f'{cur_point}: x-o')
        elif (not point_within_area) and (not prev_point_within_area) and prev_point is not None: # x-x -> add nothing
            interval_points = rectangle_line_segment_intersection(cur_point, prev_point, roi_size)
            if len(interval_points) == 2:
                polylines_filtered.append(interval_points)
                # print('here!2', interval_points, cur_point, prev_point)
            if debug:
                print(f'{cur_point}: x-x')

        if i == len(polyline) - 1 and point_within_area and len(polyline_tmp) > 0:
            polylines_filtered.append(polyline_tmp)
            if debug:
                print('here!3', len(polyline_tmp))
            polyline_tmp = []
        prev_point_within_area = point_within_area
    return polylines_filtered

def line_segment_intersection(p0, p1, q0, q1):
    d = (p1[0] - p0[0]) * (q1[1] - q0[1]) - (p1[1] - p0[1]) * (q1[0] - q0[0])
    if d == 0:
        return None

    t = ((q0[0] - p0[0]) * (q1[1] - q0[1]) - (q0[1] - p0[1]) * (q1[0] - q0[0])) / d
    u = ((q0[0] - p0[0]) * (p1[1] - p0[1]) - (q0[1] - p0[1]) * (p1[0] - p0[0])) / d

    if 0 <= t <= 1 and 0 <= u <= 1:
        intersection = np.array(p0) + t * (np.array(p1) - np.array(p0))
        return intersection
    else:
        return None


def rectangle_line_segment_intersection(p0, p1, roi_size):
    A, B = roi_size[0] / 2, roi_size[1] / 2
    rectangle_edges = [
        (np.array([-A, -B]), np.array([A, -B])),
        (np.array([A, -B]), np.array([A, B])),
        (np.array([A, B]), np.array([-A, B])),
        (np.array([-A, B]), np.array([-A, -B])),
    ]

    intersections = []
    for q0, q1 in rectangle_edges:
        intersection = line_segment_intersection(p0, p1, q0, q1)
        if intersection is not None:
            intersections.append(intersection)

    # Sort intersections by distance to p0
    intersections.sort(key=lambda point: np.linalg.norm(point - np.array(p0)))
    return intersections


def get_subsequences(arr):
    result = []
    current_subsequence = []
    for index, element in enumerate(arr):
        if element == 'o':
            current_subsequence.append(index)
        else:
            if current_subsequence:
                result.append(current_subsequence)
                current_subsequence = []
    if current_subsequence:
        result.append(current_subsequence)
    return result

def extract_near_polylines(pose, polyline_all, distance):
    polylines = []
    polyline_to_add = []
    prev_within_area = False
    for i, p in enumerate(polyline_all):
        within_area = (p[0] - pose[0]) ** 2 + (p[1] - pose[1]) ** 2 < distance ** 2
        if within_area:
            polylines.append(polyline_all)
            break
    return polylines

def global2local(polyline: List[List[float]], pose: List[float]) -> List[List[float]]:
    """Transform a polyline from global coordinates to local coordinates based on a given pose.
    
    Args:
        polyline: A list of 3D points in global coordinates. E.g. [[x, y, z], [x, y, z], ...]
        pose: A list containing the position [px, py, pz] and orientation [qx, qy, qz, qw] of the local frame in global coordinates.
    
    Returns:
        polyline_local: A list of 3D points in local coordinates. E.g. [[x, y, z], [x, y, z], ...]
    """
    position = np.array(pose[:3])
    orientation = R.from_quat(pose[3:])

    polyline_local = []
    for point in polyline:
        point_global = np.array(point)
        point_local = orientation.inv().apply(point_global - position)
        polyline_local.append(point_local.tolist())

    return polyline_local

def line_string2list3d(line_string: lanelet2.core.LineString3d) -> List[List[float]]:
    result = []
    for p in line_string:
        result.append([p.x, p.y, p.z])
    return result

def extractNode(child):
    x,y = 0,0
    for tag in child:
        if tag.attrib['k'] == 'local_x':
            x = tag.attrib['v']
        elif tag.attrib['k'] == 'local_y':
            y = tag.attrib['v']
        elif tag.attrib['k'] == 'ele':
            z = tag.attrib['v']
    return int(child.attrib["id"]), float(x), float(y), float(z)

def extractWay(child):
    refs = []
    tags = {}
    for tag in child:
        if tag.tag == "nd":
            refs.append(int(tag.attrib["ref"]))
        elif tag.tag == "tag":
            tags[tag.attrib["k"]] = tag.attrib["v"]
    return int(child.attrib["id"]), refs, tags

def extractRelation(child):
    road = {}
    road_tags = {}
    for tag in child:
        if tag.tag == "member":
            road[tag.attrib["role"]] = int(tag.attrib["ref"])         
        if tag.tag == "tag":
            if tag.attrib["k"] == "type":
                if not tag.attrib["v"] == "lanelet":
                    return None, None
            road_tags[tag.attrib["k"]] = tag.attrib["v"]
    return road, road_tags

def dict2list(dic, key_list):
    output = []
    for key in key_list:
        output.append(dic[key])
    return output

def load_lanelet2_my(rfile: str) -> lanelet2.core.LaneletMap:
    tree = ET.parse(rfile)
    root = tree.getroot()

    nodes = {}
    ways = {}
    relations = {}
    ways_tags = {}
    relations_tags = {}
    stop_line_way_ids = []
    road_border_way_ids = []

    for child in root:
        if child.tag == "node":
            ref_id, x, y, z = extractNode(child)
            nodes[ref_id] = [x, y, z]
        elif child.tag == "way":
            idx, refs, tags = extractWay(child)
            ways[idx] = refs
            ways_tags[idx] = tags

            ### Exception for stop line ###
            if "type" in tags.keys():
                if tags["type"] == "stop_line":
                    stop_line_way_ids.append(idx)
                if tags["type"] == "road_border":
                    road_border_way_ids.append(idx)
            ### Exception for stop line ###
        elif child.tag == "relation":
            idx = int(child.attrib["id"])
            road, road_tags = extractRelation(child)
            if road:
                direction = road_tags["turn_direction"] if "turn_direction" in road_tags.keys() else None
                relations[idx] = [ road["left"], road["right"], direction]
                relations_tags[idx] = road_tags
    points = {}
    for idx in nodes.keys():
        points[idx] = lanelet2.core.Point3d(idx, *nodes[idx])    

    linestrings = {}
    for idx in ways.keys():
        point_list = ways[idx]
        attribute = lanelet2.core.AttributeMap(ways_tags[idx])
        linestrings[idx] = lanelet2.core.LineString3d(idx, dict2list(points, point_list), attribute)

    lanelets = {}
    lmap = lanelet2.core.LaneletMap()

    for idx in relations.keys():
        left, right, direction = relations[idx]
        attribute = lanelet2.core.AttributeMap(relations_tags[idx])
        lanelet = lanelet2.core.Lanelet(idx, linestrings[left], linestrings[right], attribute)
        lmap.add(lanelet)

    ### Exception for stop line ###
    for stop_line_way_id in stop_line_way_ids:
        lmap.add(linestrings[stop_line_way_id])
    ### Exception for stop line ###
    ### Exception for road border ###
    for road_border_way_id in road_border_way_ids:
        if lmap.lineStringLayer.exists(road_border_way_id):
            continue
        lmap.add(linestrings[road_border_way_id])
    ### Exception for road border ###
    return lmap

def get_relation_subtype(lmap, line_string):
    usages = [
        lanelet.attributes['subtype'] for lanelet in lmap.laneletLayer.findUsages(line_string)
    ]
    if 'crosswalk' in usages:
        return 'crosswalk'
    elif 'road' in usages:
        return 'road'
    elif 'walkway' in usages:
        return 'walkway'
    elif 'road_shoulder' in usages:
        return 'road_shoulder'
    elif len(usages) == 0:
        return 'none'
    else:
        print(f"ERROR! Usage {usages} not found")
        raise NotImplementedError

def render_bev_lanelets(polylines, img_size, radius=150):
    x_scale = img_size[0] / radius
    y_scale = img_size[1] / radius
    x_max, y_max = -np.inf, -np.inf
    x_min, y_min = np.inf, np.inf
    for polyline in polylines:
        for line in polyline:
            x_max = max(x_max, line[0])
            y_max = max(y_max, line[1])
            x_min = min(x_min, line[0])
            y_min = min(y_min, line[1])
    
    