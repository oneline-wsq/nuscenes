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
    save_path = './visulization'
    dataroot = '/mnt/share_disk/nuScenes/data'
    version='v1.0-trainval'
    bad_scene = {
            "location": "singapore-onenorth",
            "sample_token": "eac9992171c446dabf9d738c23a96cde",
            "ego_token": "016a84823da74c43b9f6a2411cff159b"
    }
    
    if not os.path.exists(save_path):
      os.mkdir(save_path)
    
    with open(os.path.join('saved_gt', 'bad_scenes.json'), 'r') as f:
      bad_scenes_dict = json.load(f)

    """读数据"""
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)
    "拓扑生成器和换序器"
    generator = LaneLanguageGTGenerator()
    # 每一个场景创建一个文件夹
    nusc_map = NuScenesMap(dataroot=dataroot, map_name=bad_scene['location'])

    
    e1_data = nusc.get('ego_pose', bad_scene["ego_token"])
    
    """获得所有的key points和control points"""
    new_arclines_record, allpointsList, fig, ax = getKeysControls(bad_scene["ego_token"], e1_data,nusc_map,width,height, isPlot=True)
          
    fig.savefig(os.path.join(save_path, bad_scene["ego_token"]+'.jpg'))        

      
    
 






