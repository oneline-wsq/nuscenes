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

# 按照每个subline来画图和判断
def Turnto_vehicle(pose_dict,egopose_data):
    """
    从世界坐标系转为车身坐标系
    :param pose_dict:
    :param egopose_data:
    :param height:
    :param weight:
    :return:
    """
    # 遍历所有中心线
    rotated_pose_dict = {}
    for key, value in pose_dict.items():
        # 根据ego_Pose的坐标转到车身坐标系
        # print(value.shape)  # (40*3)
        # 将value中的每个点转到车身坐标系中
        rotated_sublines = {}
        for i in range(len(value)):
             # 字典用来存储所有旋转后的subline的点
            points=np.array(value[i]).transpose()
            # print(points.shape)  # (3,40)
            points = points - np.array(egopose_data['translation']).reshape((-1, 1))
            # 坐标转换
            points = np.dot(Quaternion(egopose_data['rotation']).rotation_matrix.T, points)
            # 然后将坐标保存到新的dict中
            rotated_sublines[i]=points
        rotated_pose_dict[key] = rotated_sublines

    # 旋转后的地图中心线参数和返回fig和ax
    # fig = plt.figure(figsize=(15, 15))
    # ax = fig.add_axes([0, 0, 1, 1])
    #
    # center_point = [0, 0]
    # for key, value in rotated_pose_dict.items():
    #     # 画每一条lane:
    #     for i in range(len(value)):
    #         points=value[i]
    #         plt.plot(points[0, :], points[1, :])
    # ax.scatter(center_point[0], center_point[1], s=20, c='r', alpha=1.0, zorder=2)

    return rotated_pose_dict#,fig,ax

def plot_centerlines(token_records,figsize=(10,10)):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    plt.grid(None)
    # 画dict内的所有点
    for key, value in token_records.items():
        for subkey, subvalue in value.items():
            plt.plot(subvalue[0, :], subvalue[1, :]) # [3,nums]

    center_point = [0, 0]
    ax.scatter(center_point[0], center_point[1], s=20, c='r', alpha=1.0, zorder=2)

    return fig,ax

def get_points_in_patch(rotated_pose,weight=64.0,height=32.0):
    """
    仿照 get_records_in_patch,
    :param roatated_pose: 转到车身坐标系的所有位姿
    :param weight: 矩形的宽
    :param height: 矩形的高
    :return: patch_lanes_record：字典，记录有点在patch内的所有lane的tokens
    """
    patch_lanes_record = {}  # 只要有点在patch内，就记录下这条lane，保存该lane下的所有在patch内的点

    # 第一层遍历：所有的lane和lane_connector
    for key, value in rotated_pose.items():
        # 因为原点是(0,0),所以就是判断x是否在[-16,16],y是否在[-32,32]
        # 判断字典里面的每一条subline
        sub_inpatch_nums = 0
        temp_subpoints={} # 用来存储点
        # 第二层遍历：遍历所有的sublines
        for subkey,sub_value in value.items():
            # 遍历subline,如果subline中有点在patch，则保存这些在patch内的点
            [rows, cols] = sub_value.shape  # rows=3,cols=40
            temp_list = []
            # 第三层遍历：遍历subLine中的所有的点
            for j in range(cols):
                pos = sub_value[:, j]
                # 判断这个pos是否在64*32内
                xn = pos[0] + weight/2
                xp = weight/2 - pos[0]
                yn = pos[1] + height/2
                yp = height/2 - pos[1]
                if xn >= 0 and xp >= 0 and yn >= 0 and yp >= 0:
                    # 在这个patch内，则将这个点记录在temp_list
                    temp_list.append(pos)
            # 判断该subline中是否有点在patch内
            if len(temp_list)>3:
                temp_subpoints[sub_inpatch_nums] = np.array(temp_list).transpose()  # shape=(3,nums)
                sub_inpatch_nums+=1

        # 判断该条lane是否有点在Patch内
        if temp_subpoints:
            patch_lanes_record[key]=temp_subpoints

    """画图看一看"""
    fig, ax = plot_centerlines(patch_lanes_record, figsize=(15, 10))
    plt.gca().add_patch(plt.Rectangle((-weight/2, -height/2), weight, height, fill=False, edgecolor='r', linewidth=1))

    """加入中心点"""

    return patch_lanes_record,fig,ax

def points_to_vehicle(point,egopose_data):
    """
    将某个点转为车身坐标系
    :param point: 三维点坐标 [621.047322330475, 1638.4606099101813, 2.5050109544695895]
    :param egopose_data:
    :return:  转换后的坐标
    """
    # 获取此时ego pose的坐标和旋转参数
    e1_rotation = egopose_data['rotation']
    e1_pos = egopose_data['translation']
    point = (point - np.array(egopose_data['translation'])).reshape((-1, 1))
    # 坐标转换
    point = np.dot(Quaternion(egopose_data['rotation']).rotation_matrix.T, point) # 正交阵，矩阵的逆=矩阵的转置
    return point

def find_start_end(nusc_map,patch_lanes_record,egopose_data,rotated_pose_dict,fig,ax):
    """
    找所有对应的arc_lines的start pose
    :param nusc_map: 当前的地图
    :param patch_lanes_record: 记录了所有patch中的lanes和lane_connector
    :param egopose_data: 车身的信息
    :param rotated_pose_dict: 旋转后的车道中心线的车身坐标系下的位置信息
    :param fig:
    :param ax:
    :return: 返回一个字典，用来保存在车身坐标系下patch内每个arcline的start_pose位置和end_pose位置
    """
    vehicle_arcline_start= {}
    # 遍历每条lane和lane_connector
    for key, value in patch_lanes_record.items():
        # 根据key(token)找到arclines
        # 遍历每条subline, 不用再回去找arcline的原始值了
        lane_keys = [] # 用来存储所有的sublines的起始点和终止点
        for subkey,subvalue in value.items():
            # 用来存储每一段subline的起点和终点
            start_end_dict = {}
            start_end_dict['start_pose']=np.around(subvalue[:,0],5)
            start_end_dict['end_pose'] = np.around(subvalue[:, -1],5)
            start_end_dict['mid_pose']= np.around(subvalue[:,int(np.ceil(subvalue.shape[1]/2))],5)
            # 画起始点和终止点
            ax.scatter(start_end_dict['start_pose'][0], start_end_dict['start_pose'][1], s=20, c='#1f57bf', marker='x',
                       alpha=0.5, zorder=2)
            ax.scatter(start_end_dict['end_pose'][0], start_end_dict['end_pose'][1], s=20, c='#f13219', marker='*',
                       alpha=0.5, zorder=2)
            lane_keys.append(start_end_dict)
        vehicle_arcline_start[key] = np.array(lane_keys)  # 将一系列的key points存储起来。

    return vehicle_arcline_start,fig,ax

def build_keypoints_list5(new_arclines_record):
    # 首先将所有的start_end点放入keypoints,去掉重复点
    startend=[]
    mid=[]
    for key,value in new_arclines_record.items():
        if value['start'] not in startend:
            startend.append(value['start'])
        if value['end'] not in startend:
            startend.append(value['end'])
        mid.append(value['mid'])
    return startend,mid

def build_keypoints_list(start_end):
    """
    读取所有start_end点，去除重复点,返回唯一的keypoints list
    :param start_end:
    :return:
    """
    all_points = []
    for key, value in start_end.items():
        value_len = len(value)
        for i in range(value_len):
            value[i]['start_pose'] = value[i]['start_pose'].reshape(1, 3)
            value[i]['end_pose'] = value[i]['end_pose'].reshape(1, 3)
            value[i]['mid_pose'] = value[i]['mid_pose'].reshape(1, 3)
            #
            all_points.append(list(value[i]['start_pose'][0]))
            if i == value_len - 1:
                all_points.append(list(value[i]['end_pose'][0]))

    all_points = np.around(np.array(all_points),5)
    all_norepeat = np.array(list(set([tuple(t) for t in all_points])))

    return all_norepeat

def build_matrix(keypoints,start_end):
    """
    生成邻接矩阵
    :param keypoints: 关键点List
    :param start_end: 起始点和终止点
    :return: 一个矩阵
    """
    keypoints_num = keypoints.shape[0]
    matrix = np.zeros([keypoints_num, keypoints_num])
    keypoints=np.around(keypoints,5)
    all_norepeat = [tuple(t) for t in keypoints]  # 转成tuple
    for key, value in start_end.items():
        value_len = len(value)
        for i in range(value_len):
            # 找到点在norepeate中的坐标
            startp = tuple(np.around(value[i]['start_pose'][0], 5)) # 取前5位判断
            endp = tuple(np.around(value[i]['end_pose'][0],5))
            start_index = all_norepeat.index(startp)
            end_index = all_norepeat.index(endp)
            matrix[start_index][end_index] = 1
            # print(start_index)
            # print(end_index)
    return matrix

def build_matrix2(keypoints,start_end):
    """
    生成邻接矩阵
    :param keypoints: 关键点List
    :param start_end: 起始点和终止点
    :return: 一个矩阵
    """
    keypoints_num = keypoints.shape[0] # 所有关键点的数量
    matrix = np.zeros([keypoints_num, keypoints_num])
    keypoints=np.around(keypoints,5)
    all_norepeat = [tuple(t) for t in keypoints]  # 转成tuple
    for key, value in start_end.items():
        value_len = len(value)
        for i in range(value_len):
            # 找到点在norepeate中的坐标
            startp = tuple(np.around(value[i]['start_pose'][0], 5)) # 取前5位判断
            endp = tuple(np.around(value[i]['end_pose'][0],5))
            midp=tuple(np.around(value[i]['mid_pose'][0],5))
            # 要判断是否能在all_norepeat中找到所有的点
            if (startp in all_norepeat) and (endp in all_norepeat) and (midp in all_norepeat):
                start_index = all_norepeat.index(startp)
                end_index = all_norepeat.index(endp)
                mid_index=all_norepeat.index(midp)
                matrix[start_index][mid_index] = 1
                matrix[mid_index][end_index] = 1
            # print(start_index)
            # print(end_index)
    return matrix


def build_matrix5(all_keypoints,new_arclines_record):
    keypoints_num = len(all_keypoints) # 所有关键点的数量
    matrix = np.zeros([keypoints_num, keypoints_num])

    # 遍历所有的arclines
    for key, value in new_arclines_record.items():
        startp=value['start']
        endp=value['end']
        midp=value['mid']
        # 找到在all_keypoints中的索引
        start_index=all_keypoints.index(startp)
        end_index = all_keypoints.index(endp)
        mid_index = all_keypoints.index(midp)

        matrix[start_index][mid_index] = 1
        matrix[mid_index][end_index] = 1

    return matrix



def merge_tooclose(keypoints,matrix):
    # 还要删除两个点距离小于某个阈值的点：
    keypoints=[tuple(t) for t in keypoints]
    # [nums，3]
    # 转为tuple
    should_del_index=[]
    should_del_points=[]
    after_del_points=[]
    before_after=defaultdict(list) # 建立索引字典
    for i in range(len(keypoints)-1):
        now_point=keypoints[i]
        key_new=keypoints[i+1:] # 后面的所有元素
        # 计算当前点与其他点的距离
        dis=np.sqrt(np.sum(np.asarray(np.array(now_point)[:2]-np.array(key_new)[:,0:2])**2,axis=1))
        # 返回距离最小值和最小值的索引
        for j,d in enumerate(dis):
            if d<1:
                # 获得最近点的坐标和索引
                nearest_point=key_new[j]
                nearest_index=keypoints.index(nearest_point)
                # 判断当前点与最近点是否有连接，如果有，则不报废
                if matrix[i,nearest_index]>0 or matrix[nearest_index,i]>0:
                    continue
                # 否则加上这个该删除的点
                should_del_index.append(i)
                before_after[i].append(nearest_index)
                # after_del_points.append(tuple(nearest_point))
                # 将当前这个点报废，
                # 首先将该点的横向和最近点的横向融合
                for ii in range(matrix.shape[1]):
                    matrix[nearest_index,ii]=matrix[nearest_index,ii] or matrix[i,ii]
                for ii in range(matrix.shape[0]):
                    matrix[ii,nearest_index]=matrix[ii,nearest_index] or matrix[ii,i]
    if should_del_index:
        # 将该融合的点融合
        new_keypoints=[n for i, n in enumerate(keypoints) if i not in should_del_index]
        # 处理邻接矩阵
        matrix=np.delete(matrix,should_del_index,0)
        matrix=np.delete(matrix, should_del_index, 1)
    else:
        new_keypoints=keypoints

    return new_keypoints,matrix,should_del_index,before_after

def merge_tooclose2(keypoints,patch_lanes_record):
    # 还要删除两个点距离小于某个阈值的点：
    keypoints=[tuple(t) for t in keypoints]
    # [nums，3]
    # 转为tuple
    should_del_index=[]
    changedict={}
    for i in range(len(keypoints)-1):
        now_point=keypoints[i]
        key_new=keypoints[i+1:] # 后面的所有元素
        # 计算当前点与其他点的距离
        dis=np.sqrt(np.sum(np.asarray(np.array(now_point)[:2]-np.array(key_new)[:,0:2])**2,axis=1))
        # 返回距离最小值和最小值的索引
        for j,d in enumerate(dis):
            if d<1: # 如果两个点之间的距离小于1米,则将其中一个点删除
                # 获得最近点的坐标和索引
                nearest_point=key_new[j]
                nearest_index=keypoints.index(nearest_point)

                # 否则加上这个该删除的点
                should_del_index.append(i)

    if should_del_index:
        # 将该融合的点融合
        new_keypoints=[n for i, n in enumerate(keypoints) if i not in should_del_index]
    else:
        new_keypoints=keypoints
    return new_keypoints

def add_mid_point(keypoints,start_end):
    # 首先判断点是否在start_end中
    keypoints_tuple=tuple(keypoints) # 将keypoints转为tuple
    for p in range(len(keypoints_tuple)):
        keypoint=keypoints_tuple[p] # 找到一个关键点
        for key, value in start_end.items():
            value_len = len(value)
            for i in range(value_len):
                # 找到点在norepeate中的坐标
                startp = tuple(np.around(value[i]['start_pose'][0], 5))  # 取前5位判断
                endp = tuple(np.around(value[i]['end_pose'][0], 5))

                start_index = keypoints.index(startp)
                end_index = keypoints.index(endp)

def merge_tooclose5(patch_acrlines_record):
    for key, now in patch_acrlines_record.items():
        for key2, after in patch_acrlines_record.items():
            # 只计算与该lane之后的所有Lane的距离
            if key2 > key:
                dis = np.linalg.norm(np.array(now['start']) - np.array(after['start']), ord=2)
                if 0<dis < 1:
                    # 开始点不能相等
                    if now['end']==after['end']:
                        now['isvalid'] = False
                    else:
                        if now['start'] != after['end'] or now['end'] != after['start']:
                            now['start'] = after['start']
                            now['points'][0] = after['start']
    # 计算end和end的距离
    for key, now in patch_acrlines_record.items():
        for key2, after in patch_acrlines_record.items():
            # 只计算与该lane之后的所有Lane的距离
            if key2 > key:
                dis = np.linalg.norm(np.array(now['end']) - np.array(after['end']), ord=2)
                if 0<dis < 1:
                    if now['start']==after['start']:
                        now['isvalid'] = False
                    else:
                        if now['start']!=after['start']:
                            now['end'] = after['end']
                            now['points'][-1] = after['end']
    # 将所有的invalid arcline删去
    new_arclines_record=copy.deepcopy(patch_acrlines_record)
    for key, value in patch_acrlines_record.items():
        if value['isvalid'] == False:
            del new_arclines_record[key]
    return new_arclines_record