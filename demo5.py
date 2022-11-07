"""遍历所有的场景下的所有samples"""
import matplotlib.pyplot as plt
import tqdm
import numpy as np
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.map_expansion import arcline_path_utils
from nuscenes.map_expansion.bitmap import BitMap
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion
import sys
from utils3 import *
import os
import json

"""读取数据"""
nusc = NuScenes(version='v1.0-mini', dataroot='', verbose=False)
# nusc.list_scenes()
save_path='ego_poses5'
height=24
weight=48

for i in range(2,3):
    # 遍历所有的scene，一共10个scene
    scene_path=save_path+'/scene_'+str(i)
    if not os.path.exists(scene_path):
        os.makedirs(scene_path)
    scene=nusc.scene[i]
    location=nusc.get('log',scene['log_token'])['location'] # 获取具体位置，是在哪张地图上
    # print(location)
    # 读取该场景地图
    nusc_map = NuScenesMap(dataroot='', map_name=location)
    ego_sample_tokens,ego_poses_tokens, ego_poses, fig, ax = nusc_map.\
        render_egoposes_on_fancy_map(nusc, scene_tokens=[scene['token']], verbose=False)
    ego_poses_nums=len(ego_poses_tokens)
    # fig.show()
    # for j in range(ego_poses_nums):

    json_path=scene_path+'/tokens.json'
    tokens = {}

    for j in range(ego_poses_nums): # (33,34)
        ego_path=scene_path+'/egopose_'+str(j)
        # 存tokens
        ego_name='egopose_'+str(j)
        tokens[ego_name]={}
        tokens[ego_name]['sample_token']=ego_sample_tokens[j]
        tokens[ego_name]['ego_token'] = ego_poses_tokens[j]

        e1_data = nusc.get('ego_pose', ego_poses_tokens[j])
        """利用api中的函数获得e1_data周围的所有lane和lane_connectors"""
        e1_pose = e1_data['translation']
        my_patch = (e1_pose[0] - 100, e1_pose[1] - 100, e1_pose[0] + 100, e1_pose[1] + 100)  # (x_min,y_min,x_max,y_max)
        records = nusc_map.get_records_in_patch(my_patch, layer_names=['lane', 'lane_connector'], mode='intersect')

        """转成中心线"""
        pose_dict = nusc_map.my_render_centerlines3(records, resolution_meters=0.2) # 分辨率为0.4

        """转到车身坐标系"""
        rotated_pose_dict= Turnto_vehicle(pose_dict, e1_data)

        """裁剪出一个32*64大小的矩形"""
        patch_lanes_record, fig, ax = get_points_in_patch(rotated_pose_dict,weight=weight,height=height)
        # fig.show()
        """建立一个所有arclines的dict"""
        patch_acrlines_record = {}
        arcnums = 0
        for key, value in patch_lanes_record.items():
            n = len(value)
            for aa in range(n):
                one_arcline = {}
                one_arcline['points'] = [tuple(ttt) for ttt in tuple(np.around(value[aa],5).transpose())]
                one_arcline['start'] = one_arcline['points'][0]
                one_arcline['end'] = one_arcline['points'][-1]
                one_arcline['mid'] = one_arcline['points'][int(np.ceil(len(one_arcline['points'])/2))]
                one_arcline['isvalid']=True
                patch_acrlines_record[arcnums] = one_arcline
                arcnums += 1

        # # 画图
        # fig1 = plt.figure(figsize=(20, 10))
        # ax1 = fig1.add_axes([0, 0, 1, 1])
        # plt.grid(None)
        # for key, value in patch_acrlines_record.items():
        #     points = np.array(value['points'])
        #     plt.plot(points[:, 0], points[:, 1])  # [3,nums]
        #     ax1.scatter(value['mid'][0],value['mid'][1],s=10, c='r', alpha=1.0, zorder=2)
        #     ax1.annotate(key, (value['mid'][0],value['mid'][1]))
        # center_point = [0, 0]
        # ax.scatter(center_point[0], center_point[1], s=20, c='r', alpha=1.0, zorder=2)
        # plt.gca().add_patch(
        #     plt.Rectangle((-weight / 2, -height / 2), weight, height, fill=False, edgecolor='r', linewidth=1))
        # # fig.show()

        """根据end 和end ,start和start的距离,融合太近的点"""
        # 首先计算start和start的距离
        new_arclines_record=merge_tooclose5(patch_acrlines_record)
        # 画图
        fig = plt.figure(figsize=(20,10))
        ax = fig.add_axes([0, 0, 1, 1])
        plt.grid(None)
        for key, value in new_arclines_record.items():
            points=np.array(value['points'])
            plt.plot(points[:, 0], points[:, 1])  # [3,nums]
        center_point = [0, 0]
        ax.scatter(center_point[0], center_point[1], s=20, c='r', alpha=1.0, zorder=2)
        plt.gca().add_patch(plt.Rectangle((-weight / 2, -height / 2), weight, height, fill=False, edgecolor='r', linewidth=1))
        # fig.show()

        """建立关键点list，去除重复点"""
        startend,mid= build_keypoints_list5(new_arclines_record)
        startend_array=np.array(startend)
        anno=0
        for jj in range(len(startend)):
            ax.scatter(startend_array[jj,0],startend_array[jj,1],s=20, c='#21f211', marker='o', alpha=1, zorder=6)
            ax.annotate(anno, (startend_array[jj,0],startend_array[jj,1]))
            anno+=1

        mid_array=np.array(mid)
        for jj in range(len(mid)):
            ax.scatter(mid_array[jj,0],mid_array[jj,1],s=20, c='#0000FF', marker='o', alpha=1, zorder=6)
            ax.annotate(anno, (mid_array[jj,0],mid_array[jj,1]))
            anno+=1

        """建立邻接矩阵"""
        all_keypoints=startend+mid
        matrix=build_matrix5(all_keypoints,new_arclines_record)

        """保存"""
        # 保存图像
        img_name = ego_path + '.jpg'.format(j)
        fig.savefig(img_name)
        # 保存关键点
        keypoints_path = ego_path + '_' + 'keypoints.npy'
        np.save(keypoints_path, all_keypoints)  # 将新的点保存到矩阵中
        # 保存矩阵
        matrix_path = ego_path + '_' + 'matrix.npy'
        np.save(matrix_path, matrix)
        print(j, '-finished')

    jsondata = json.dumps(tokens, indent=4, separators=(',', ': '))
    f = open(json_path, 'w')
    f.write(jsondata)
    f.close()

