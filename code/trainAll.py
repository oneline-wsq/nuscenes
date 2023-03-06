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
from exchangeOrder.exchangePoints import exchangeOrder_by_carcenter,exchangeOrder_by_direction



"""读取数据"""
if __name__ == '__main__':
    """数据参数"""
    height = 24
    width = 48
    dataroot = '/mnt/share_disk/nuScenes/data'
    version='v1.0-trainval'
    # version='v1.0-test' # 测试集
    # version='v1.0-mini'

    gt_save_path = './saved_gt_front'
    if not os.path.exists(gt_save_path):
        os.mkdir(gt_save_path)
    
    generated_gt_dict = {}
    
    
    empty_scene_dict = {}
    empty_scene_dict['empty_scene'] = []
    unknow_scene_dict = {}
    unknow_scene_dict['unknow_scene'] = []
    bad_scene_dict = {}
    bad_scene_dict['bad_scenes'] = []
    """读数据"""
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)
    "拓扑生成器和换序器"
    
    # 第一个轴是范围长的(car_length), 第二个轴范围是短的(car_width)
    generator = LaneLanguageGTGenerator(car_length = int(width/2), car_width=int(height/2))
    # 每一个场景创建一个文件夹
    
    prev_location = ""
    nusc_map = None
    pbar = tqdm(range(len(nusc.scene)))
    pbar.set_description('generating nuscense lane topology gt')
    
    count = 0
    
    for scene_index in pbar:
    # for scene_index in range(2):  
        scene = nusc.scene[scene_index]
        # 获取具体位置，是在哪张地图上
        location = nusc.get('log', scene['log_token'])['location']
        # 读取该场景地图
        if prev_location != location:
            nusc_map = NuScenesMap(dataroot=dataroot, map_name=location)
            prev_location = location
        # 获取所有egopose
        ego_sample_tokens, ego_poses_tokens, ego_poses = render_egoposes_on_fancy_map(location,nusc_map,nusc, scene_tokens=[scene['token']], verbose=False)
        ego_poses_nums = len(ego_poses_tokens)
        # 存tokens
        tokens = {}
        for ego_index in range(ego_poses_nums):
            # 遍历一个场景下的ego pose
            # if scene_index==1 and ego_index==1:
            #     print('scene_index:1; ego_index:1')
            ego_name = 'egopose_' + str(ego_index)
            tokens[ego_name] = {}
            tokens[ego_name]['sample_token'] = ego_sample_tokens[ego_index]
            tokens[ego_name]['ego_token'] = ego_poses_tokens[ego_index]
            e1_data = nusc.get('ego_pose', ego_poses_tokens[ego_index])
            
            """获得所有的key points和control points"""
            if scene_index<10:
                isPlot=True
            else:
                isPlot=False
            new_arclines_record, allpointsList, allpointsDict, fig, ax = getKeysControls(ego_poses_tokens[ego_index], e1_data,nusc_map,width,height, isPlot=isPlot)
            
            
            try:
                if(len(allpointsList)>0):
                    "根据预先设定的规则重新排序"
                    allpointsList = exchangeOrder_by_direction(new_arclines_record,allpointsList,allpointsDict)
                    
                    """画图"""
                    if isPlot:
                        from utils import plot_pointsAnno
                        fig, ax = plot_pointsAnno(allpointsList, fig, ax)
                        ax.set_axis_on
                        img_dir='saved_img_front/scene_'+str(scene_index)
                        if not os.path.exists(img_dir):
                            os.makedirs(img_dir)
                        img_name=img_dir+'/'+str(ego_index)+'.jpg'
                        fig.savefig(img_name) # 保存图像
                        plt.cla()
                        plt.close("all")
                    
                    """建立邻接矩阵"""
                    matrix=build_AdjacencyMatrix(new_arclines_record,allpointsList)
                    keypoint_list = generator.preprocessing(allpointsList, matrix)
                    if generator.isGoodData(keypoint_list):
                        gt_array, KeyMatrix, lane_path_list = generator.process(keypoint_list, matrix) 
                        generated_gt_dict[tokens[ego_name]['sample_token']] = {'gt_array':gt_array, 'gt_dict':lane_path_list, 'KeyMatrix':KeyMatrix}
                    else:
                        bad_scene = {}
                        bad_scene['location'] = location
                        bad_scene['sample_token'] = ego_sample_tokens[ego_index]
                        bad_scene['ego_token'] = ego_poses_tokens[ego_index]
                        bad_scene_dict['bad_scenes'].append(bad_scene)
                        print('bad_scene: ', bad_scene)
                else:
                    # pass
                    empty_scene = {}
                    empty_scene['location'] = location
                    empty_scene['sample_token'] = ego_sample_tokens[ego_index]
                    empty_scene['ego_token'] = ego_poses_tokens[ego_index]
                    empty_scene_dict['empty_scene'].append(empty_scene)
                    print('empty_scene: ', empty_scene)
                    
            except Exception as e:
                unknow_scene = {}
                unknow_scene['location'] = location
                unknow_scene['sample_token'] = ego_sample_tokens[ego_index]
                unknow_scene['ego_token'] = ego_poses_tokens[ego_index]
                unknow_scene_dict['unknow_scene'].append(unknow_scene)
                print('unknow_scene: ', unknow_scene)
                
            count+=1
        # if count > 10:
        #     break
        pbar.update()
        
    with open(os.path.join(gt_save_path, 'unknow_scene.json'), 'w') as f:
        json.dump(unknow_scene_dict, f, indent=4)
    with open(os.path.join(gt_save_path, 'empty_scene_dict.json'), 'w') as f:
        json.dump(empty_scene_dict, f, indent=4)
    with open(os.path.join(gt_save_path, 'bad_scenes.json'), 'w') as f:
        json.dump(bad_scene_dict, f, indent=4)
    with open(os.path.join(gt_save_path, 'lane_gt.json'), 'w') as f:
      json.dump(generated_gt_dict, f)
      
    print('end')






