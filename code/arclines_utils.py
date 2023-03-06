import matplotlib.pyplot as plt
import tqdm
import numpy as np
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.map_expansion import arcline_path_utils
from nuscenes.map_expansion.bitmap import BitMap
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion
import sys
from collections import defaultdict
import copy
from itertools import chain
import math
from typing import Dict, List, Tuple, Optional, Union
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from nuscenes.map_expansion.arcline_path_utils import pose_at_length,compute_segment_sign,_get_lie_algebra,get_transformation_at_step,apply_affine_transformation

def my_arclines_centerline(nusc_map,records:dict,
                               resolution_meters: float = 0.5,
                               figsize: Union[None, float, Tuple[float, float]] = None)-> Tuple[Figure, Axes]:
    """
    返回所有arclines的记录
    """
    pose_dict = {}
    each_pose_lists = []
    # self.lane在json文件中
    for lane in records['lane'] + records['lane_connector']:
        # 这里的get是dict的内置函数
        # 首先获得lane这条车道的一些列参数
        my_lane = nusc_map.arcline_path_3.get(lane, [])
        discretized = discretize_arclane(my_lane, resolution_meters)
        pose_dict[lane] = discretized
    return pose_dict


def discretize_arclane(lane,resolution_meters) :
    """
    离散化一条lane, 分成若干的arclines
    :param lane: Lanes are represented as a list of arcline paths.
    :param resolution_meters: How finely to discretize the lane. Smaller values ensure curved
        lanes are properly represented.
    :return: List of pose tuples along the lane.
    """
    arclines={} # 用来存储离散化的点和半径
    i = 0
    for path in lane:
        onearcline={} # 存储一条arcline的点和半径
        subpose_list = []
        # 得到一系列离散的点的坐标
        poses, radius = discretizeAddRadius(path, resolution_meters)
        # len(poses)=n_points
        for pose in poses:
            subpose_list.append(tuple(np.around(pose,5)))
        onearcline['points'] = subpose_list  # 用字典存储
        onearcline['radius']=radius
        arclines[i]=onearcline
        i += 1
    return arclines

def discretizeAddRadius(arcline_path,resolution_meters):
    """
    Discretize an arcline path.
    :param arcline_path: Arcline path record.
    :param resolution_meters: How finely to discretize the path.
    :return: List of pose tuples.
    """
    path_length = sum(arcline_path['segment_length'])
    radius = arcline_path['radius']

    # 求所有点的数量
    n_points = int(max(math.ceil(path_length / resolution_meters) + 1.5, 2)) # 这里是指点的数量，这里的resolution_meters=0.5其实是指每0.5米有一个点
    # 118

    # 重新计算resolution_meters
    resolution_meters = path_length / (n_points - 1) #0.49583110327766267， 原来是0.5

    discretization = []

    cumulative_length = [arcline_path['segment_length'][0],
                         arcline_path['segment_length'][0] + arcline_path['segment_length'][1],
                         path_length + resolution_meters]

    # 根据shape来生成sign,但是每个值代表的含义是？
    # shape='LSR'-> (1,0,-1)
    segment_sign = compute_segment_sign(arcline_path)
    # 获取 Lie algebra 李代数，李代数是啥？？
    poses = _get_lie_algebra(segment_sign, radius)
    #
    temp_pose = arcline_path['start_pose']

    g_i = 0
    g_s = 0.0

    for step in range(n_points):

        #0，0.5，1，1.5，2，2.5
        step_along_path = step * resolution_meters

        if step_along_path > cumulative_length[g_i]:
            # Retrieves pose at l meters along the arcline path
            temp_pose = pose_at_length(arcline_path, step_along_path)
            g_s = step_along_path
            g_i += 1
        # Get the affine transformation at s meters along the path.
        transformation = get_transformation_at_step(poses[g_i], step_along_path - g_s)
        new_pose = apply_affine_transformation(temp_pose, transformation)
        discretization.append(new_pose)

    return discretization,radius


def render_egoposes_on_fancy_map(location,nusc_map,nusc: NuScenes,
                                 scene_tokens: List = None,
                                 verbose: bool = True,
                                 out_path: str = None,
                                 render_egoposes: bool = True,
                                 render_egoposes_range: bool = True,
                                 render_legend: bool = True,
                                 bitmap: Optional[BitMap] = None) -> Tuple[np.ndarray, Figure, Axes]:
    """
    Renders each ego pose of a list of scenes on the map (around 40 poses per scene).
    This method is heavily inspired by NuScenes.render_egoposes_on_map(), but uses the map expansion pack maps.
    Note that the maps are constantly evolving, whereas we only released a single snapshot of the data.
    Therefore for some scenes there is a bad fit between ego poses and maps.
    :param nusc: The NuScenes instance to load the ego poses from.
    :param scene_tokens: Optional list of scene tokens corresponding to the current map location.
    :param verbose: Whether to show status messages and progress bar.
    :param out_path: Optional path to save the rendered figure to disk.
    :param render_egoposes: Whether to render ego poses.
    :param render_egoposes_range: Whether to render a rectangle around all ego poses.
    :param render_legend: Whether to render the legend of map layers.
    :param bitmap: Optional BitMap object to render below the other map layers.
    :return: <np.float32: n, 2>. Returns a matrix with n ego poses in global map coordinates.
    """
    # Settings
    patch_margin = 2
    min_diff_patch = 30

    # Ids of scenes with a bad match between localization and map.
    scene_blacklist = [499, 515, 517]

    # Get logs by location.
    # 首先筛某个地图的所有Log_tokens
    log_location = location# 这个是指地图的名字，比如 'boston-seaport'
    log_tokens = [log['token'] for log in nusc.log if log['location'] == log_location] # 找到所有在该地图上的tokens

    # nuscenes数据集一个'log.json'，里面'location'确定了是在哪张地图上跑。
    assert len(log_tokens) > 0, 'Error: This split has 0 scenes for location %s!' % log_location

    # Filter scenes.
    # 然后再筛场景
    scene_tokens_location = [e['token'] for e in nusc.scene if e['log_token'] in log_tokens]
    if scene_tokens is not None:
        scene_tokens_location = [t for t in scene_tokens_location if t in scene_tokens]

    assert len(scene_tokens_location) > 0, 'Error: Found 0 valid scenes for location %s!' % log_location

    map_poses = []
    # 记录该场景所有的ego pose的tokens
    ego_poses_tokens=[]
    # 记录每个ego pose对应的sample的tokens
    ego_sample_tokens=[]

    if verbose:
        print('Adding ego poses to map...')

    for scene_token in scene_tokens_location:
        # Check that the scene is from the correct location.
        # 只有一个scene？
        scene_record = nusc.get('scene', scene_token)
        scene_name = scene_record['name']
        scene_id = int(scene_name.replace('scene-', ''))
        log_record = nusc.get('log', scene_record['log_token'])
        assert log_record['location'] == log_location, \
            'Error: The provided scene_tokens do not correspond to the provided map location!'

        # Print a warning if the localization is known to be bad.
        if verbose and scene_id in scene_blacklist:
            print('Warning: %s is known to have a bad fit between ego pose and map.' % scene_name)

        # For each sample in the scene, store the ego pose.
        # 对于场景中的每个sample, 存储 ego pose
        sample_tokens = nusc.field2token('sample', 'scene_token', scene_token) # 找到所有sample['scene_token']==scene_token的sample的token

        # 遍历该场景的所有sample
        for sample_token in sample_tokens:
            sample_record = nusc.get('sample', sample_token)

            # Poses are associated with the sample_data. Here we use the lidar sample_data.
            sample_data_record = nusc.get('sample_data', sample_record['data']['LIDAR_TOP']) # 找到LIDAR_TOP在sample_data中的记录，包含了文件存放位置
            pose_record = nusc.get('ego_pose', sample_data_record['ego_pose_token'])

            # Calculate the pose on the map and append.
            map_poses.append(pose_record['translation'])
            ego_poses_tokens.append(pose_record['token'])
            ego_sample_tokens.append(sample_token)

    # Check that ego poses aren't empty.
    assert len(map_poses) > 0, 'Error: Found 0 ego poses. Please check the inputs.'

    # Compute number of close ego poses.
    if verbose:
        print('Creating plot...')
    map_poses = np.vstack(map_poses)[:, :2] # 取前两维

    # Render the map patch with the current ego poses.
    min_patch = np.floor(map_poses.min(axis=0) - patch_margin) # patch_margin是在边缘留出一些距离
    max_patch = np.ceil(map_poses.max(axis=0) + patch_margin)
    diff_patch = max_patch - min_patch
    if any(diff_patch < min_diff_patch):
        center_patch = (min_patch + max_patch) / 2
        diff_patch = np.maximum(diff_patch, min_diff_patch)
        min_patch = center_patch - diff_patch / 2
        max_patch = center_patch + diff_patch / 2
    my_patch = (min_patch[0], min_patch[1], max_patch[0], max_patch[1])

    return ego_sample_tokens, ego_poses_tokens, map_poses