import matplotlib.pyplot as plt
from tqdm import tqdm
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
from exchangeOrder.lanenetwork_gt_generator import LaneLanguageGTGenerator
from exchangeOrder.exchangePoints import exchangeOrder_by_carcenter



"""读取数据"""
if __name__ == '__main__':
    """数据参数"""
    height = 24
    width = 48
    save_path = './ego_poses'
    dataroot = '/mnt/share_disk/nuScenes/data'
    version='v1.0-trainval' # 
    
    ego_pose_tokens = []
    pose_tokens = os.listdir('ego_pose_token')
    for pose_token in pose_tokens:
      split_token = pose_token.split('.')
      if 'jpg' == split_token[-1]:
        ego_pose_tokens.append(''.join(split_token[:-1]))
    print('bad ego_pose_tokens', ego_pose_tokens)
    
    generated_gt_dict = {}

    """读数据"""
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)
    "拓扑生成器和换序器"
    generator = LaneLanguageGTGenerator()
    # 每一个场景创建一个文件夹
    nusc_map = NuScenesMap(dataroot=dataroot, map_name='singapore-onenorth')

    pbar = tqdm(range(len(nusc.scene)))
    pbar.set_description('testing')
    
    for ego_index in range(len(ego_pose_tokens)):
  
        e1_data = nusc.get('ego_pose', ego_pose_tokens[ego_index])

        """获得所有的key points和control points"""
        new_arclines_record, allpointsList, fig, ax = getKeysControls(ego_poses_tokens[ego_index], e1_data,nusc_map,width,height, isPlot=False)
                  
            

        # print(scene_index, '-finished')
        pbar.update()
     
    
 






