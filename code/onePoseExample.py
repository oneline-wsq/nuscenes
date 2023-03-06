import matplotlib.pyplot as plt
import tqdm
import numpy as np
import sys
import os
import json
from utils import build_AdjacencyMatrix,getKeysControls,saveAll,saveJson
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.map_expansion import arcline_path_utils
from nuscenes.map_expansion.bitmap import BitMap
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion
from arclines_utils import render_egoposes_on_fancy_map


"""读取数据"""
if __name__ == '__main__':
    """数据参数"""
    scene_index = 8
    ego_index = 17
    height = 24
    width = 48
    save_path = './ego_poses'
    dataroot = '/Users/pengzai/实习/毫末智行/拓扑预测/nuScence数据处理/v1.0-mini'
    

    """读数据"""
    nusc = NuScenes(version='v1.0-mini', dataroot=dataroot, verbose=True)
    # 每一个场景创建一个文件夹
    scene_path = save_path + '/scene_' + str(scene_index)
    if not os.path.exists(scene_path):
        os.makedirs(scene_path)
    scene = nusc.scene[scene_index]
    # 获取具体位置，是在哪张地图上
    location = nusc.get('log', scene['log_token'])['location']
    # 读取该场景地图
    nusc_map = NuScenesMap(dataroot=dataroot, map_name=location)
    # 获取所有egopose
    ego_sample_tokens, ego_poses_tokens, ego_poses =render_egoposes_on_fancy_map(location,nusc_map,nusc, scene_tokens=[scene['token']], verbose=False)
    ego_poses_nums = len(ego_poses_tokens)
    ego_path = scene_path + '/egopose_' + str(ego_index)
    # 存tokens
    ego_name = 'egopose_' + str(ego_index)
    e1_data = nusc.get('ego_pose', ego_poses_tokens[ego_index])

    """获得所有的key points和control points"""
    new_arclines_record,allpointsList,fig,ax=getKeysControls(e1_data,nusc_map,width,height)
    fig.show()
    """建立邻接矩阵"""
    matrix=build_AdjacencyMatrix(new_arclines_record,allpointsList)


    print('end')






