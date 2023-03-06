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
from arclines_utils import my_arclines_centerline,render_egoposes_on_fancy_map
import json

def Turnto_vehicle2(pose_dict,egopose_data):
    # 转为车身坐标系
    rotated_pose_dict = {}
    # 遍历所有的Lanes
    for key, value in pose_dict.items():
        rotated_sublines = {}
        # 遍历所有的arclines
        for arckey,arcvalue in value.items():
            onearcline={}
            points=np.array(arcvalue['points']).transpose() # [3,40]
            points = points - np.array(egopose_data['translation']).reshape((-1, 1))
            # 坐标转换
            points = np.dot(Quaternion(egopose_data['rotation']).rotation_matrix.T, points)
            points=[tuple(t) for t in points.transpose()]
            # 然后将坐标保存到新的dict中
            onearcline['points'] = points
            # 保存半径
            onearcline['radius']=arcvalue['radius']
            rotated_sublines[arckey]=onearcline
        rotated_pose_dict[key] = rotated_sublines
    return rotated_pose_dict


def get_arclines_in_patch(rotated_pose, width = 48.0, height = 24.0):
    # 获得记录所有arcline离散点以及半径值的dict
    patch_arclanes_record = {}  # 只要有点在patch内，就记录下这条lane，保存该lane下的所有在patch内的点

    # 第一层遍历：所有的lane和lane_connector
    arclines_nums = 0
    for key, value in rotated_pose.items():
        # 图的大小为 24*48; 其中，车的位置为(0,12), 只向前看，
        # 因所以就是判断x是否在[0,48],y是否在[-12,12]
        # 判断字典里面的每一条subline
        # 第二层遍历：遍历所有的arclines
        for subkey, sub_value in value.items():
            one_arcline = {}  # 用来存储arclines的一系列参数
            # 遍历subline,如果subline中有点在patch，则保存这些在patch内的点
            points=np.array(sub_value['points']).transpose()
            [rows, cols] = points.shape  # rows=3,cols=40
            temp_list = []
            # 第三层遍历：遍历一条arcline中的所有points
            for j in range(cols):
                pos = points[:, j]
                # 判断这个pos是否在48*24内
                xn = pos[0]  # >=0
                xp = width - pos[0] #>=0
                yn = pos[1] + height / 2
                yp = height / 2 - pos[1]
                if xn >= 0 and xp >= 0 and yn >= 0 and yp >= 0:
                    # 在这个patch内，则将这个点记录在temp_list
                    temp_list.append(tuple(pos))
            # 判断该subline中是否有点在patch内
            if len(temp_list) > 3:
                one_arcline['points'] = temp_list  # shape=(3,nums)
                one_arcline['start'] = one_arcline['points'][0]
                one_arcline['end'] = one_arcline['points'][-1]
                one_arcline['mid'] = one_arcline['points'][int(np.ceil(len(one_arcline['points'])/2))]
                one_arcline['radius']=sub_value['radius']
                one_arcline['isvalid']=True # 用来判断后面是否要融合点或者去除
                patch_arclanes_record[arclines_nums]=one_arcline
                arclines_nums += 1

    return patch_arclanes_record

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

def merge_tooclose(patch_acrlines_record):
    for key, now in patch_acrlines_record.items():
        for key2, after in patch_acrlines_record.items():
            # 只计算与该lane之后的所有Lane的距离
            if key2 > key:
                dis = np.linalg.norm(np.array(now['start'])[:2] - np.array(after['start'])[:2], ord=2)
                if 0<dis < 1:
                    # 起点距离近，但终点相等
                    if now['end']==after['end']:
                        now['isvalid'] = False
                        now['radius']=max(now['radius'],after['radius'])
                        after['radius'] = max(now['radius'], after['radius'])
                    else:
                        # 虽然起始点之间的距离小于1，但是要排除两者首尾相连的情况
                        if now['start'] != after['end'] and now['end'] != after['start']:
                            now['start'] = after['start']
                            now['points'][0] = after['start']
    # 计算end和end的距离
    for key, now in patch_acrlines_record.items():
        for key2, after in patch_acrlines_record.items():
            # 只计算与该lane之后的所有Lane的距离
            if key2 > key:
                dis = np.linalg.norm(np.array(now['end'])[:2] - np.array(after['end'])[:2], ord=2)
                if 0<dis < 1:
                    if now['start']==after['start']:
                        now['isvalid'] = False
                        now['radius']=max(now['radius'],after['radius'])
                        after['radius'] = max(now['radius'], after['radius'])
                    else:
                        # 排除两者首尾相连的情况
                        if now['start'] != after['end'] and now['end'] != after['start']:
                            now['end'] = after['end']
                            now['points'][-1] = after['end']
    # 将所有的invalid arcline删去
    new_arclines_record=copy.deepcopy(patch_acrlines_record)
    for key, value in patch_acrlines_record.items():
        if value['isvalid'] == False:
            del new_arclines_record[key]
    return new_arclines_record

def add_tokens(new_arclines_record):
    new_dict = copy.deepcopy(new_arclines_record)
    for key, value in new_arclines_record.items():
        new_dict[key]['token']=key
    return new_dict

def polt_arclines(arlines,width,height,printRadius=False,printIndex=True):
    # # 画图
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_axes([0, 0, 1, 1])
    plt.grid(None)
    for key, value in arlines.items():
        points = np.array(value['points'])
        plt.plot(points[:, 0], points[:, 1])  # [3,nums]
        if printIndex:
            ax.scatter(value['mid'][0],value['mid'][1],s=10, c='r', alpha=1.0, zorder=2)
            ax.annotate(key, (value['mid'][0],value['mid'][1]))
        if printRadius:
            ax.annotate(value['radius'], (value['mid'][0]-3, value['mid'][1]+0.3))
    center_point = [0, 0]
    ax.scatter(center_point[0], center_point[1], s=20, c='r', alpha=1.0, zorder=2)
    plt.gca().add_patch(
        plt.Rectangle((0, -height / 2), width, height, fill=False, edgecolor='r', linewidth=1))

    return fig,ax

def plot_keypoints(keypoints,fig,ax,printIndex=True):
    color={'start':'#228B22','end':'#778899','merge':'#00BFFF','fork':'#FFA500','controls':'#FF00FF'}
    # 起始点是绿色，终止点是灰色，merge点是蓝色，fork点是黄色,控制点是紫色
    i=0
    for key, value in keypoints.items():
        points=np.array(value)
        for jj in range(len(value)):
            ax.scatter(points[jj, 0], points[jj, 1], s=20, c=color[key], marker='o', alpha=1, zorder=6)
            if printIndex:
                ax.annotate(i, (points[jj, 0], points[jj, 1]))
            i+=1
    return fig,ax
def plot_pointsList(keypointsList,fig,ax):
    i=0
    for point in keypointsList:
        point=np.array(point)
        ax.scatter(point[0], point[1], s=30, c="#FF0000", marker='o', alpha=1, zorder=6)
        ax.annotate(i, (point[0], point[1]))
        i+=1
    return fig,ax

def plot_pointsAnno(keypointsList,fig,ax):
    i=0
    for point in keypointsList:
        point=np.array(point)
        # ax.scatter(point[0], point[1], s=30, c="#FF0000", marker='o', alpha=1, zorder=6)
        ax.annotate(i, (point[0], point[1]),size=20)
        i+=1
    return fig,ax

def plot_pointsDict(keypointsDict,fig,ax,printIndex=False):
    i=0
    color = {'start': '#228B22', 'end': '#778899', 'merge': '#00BFFF', 'fork': '#FFA500', 'controls': '#FF00FF'}
    for key,value in keypointsDict.items():
        point=np.array(key)
        ax.scatter(point[0], point[1], s=30, c=color[value[0]], marker='o', alpha=1, zorder=6)
        if printIndex:
            ax.annotate(i, (point[0], point[1]))
        i+=1
    return fig,ax


def findStraightLines(new_arclines_record,threshold=100):
    all_straightLines={}
    for key, value in new_arclines_record.items():
        r=value['radius']
        if r>threshold:
            # 如果大于阈值，则为直线
            value['isStraight']=True

        else:
            value['isStraight'] = False
        all_straightLines[key] = value

    return all_straightLines

def mergeStraightLines(all_straightLines):
    mergedSrightLines={} # 将融合好的直线加入这个字典中
    begin_lanes=[]
    # 这里beginlanes有3种，分别是入度为0和入度>1的lane,以及出度大于1的所有的出的线
    for key, value in all_straightLines.items():
        if value['isStraight']:
            # 首先找到所有起始线段，incoming的长度为0
            if len(value['incoming'])<1 or len(value['incoming'])>1:
                # 入度为2，重新当作一个起始点, 出度为2，重新当作一个起始点
                begin_lanes.append(key)
            if len(value['outgoing'])>1:
                # 出度大于1，则将所有的出的线作为begin_lanes
                for j in value['outgoing']:
                    if  all_straightLines[j]['isStraight']:
                        begin_lanes.append(j)
    # 从begin开始遍历
    all_mergeList=[]
    for lane_index in begin_lanes:
        combine_list=[] # 每个begin lanes对应着一系列待拼接的点
        isend=False # 判断是否要终止
        now_lane_index=lane_index
        while not isend:
            # 判断当前这条线的入度和出度
            lane_record = all_straightLines[now_lane_index] # 当前这条线的所有记录
            if len(lane_record['incoming'])<2 and len(lane_record['outgoing'])==1:
                combine_list.append(lane_record['token'])
                next_index=lane_record['outgoing'][0]
                if next_index not in begin_lanes:
                    # 如果下一条线的索引不在起始线中，则继续
                    now_lane_index=lane_record['outgoing'][0]
                else:
                    # 排除下一条线是merge线的情况
                    isend=True
            elif len(lane_record['incoming'])<2 and len(lane_record['outgoing'])>1:
                # 此时这条线的终点开始分叉，那么将这条线的索引加上后，结束这条线
                combine_list.append(lane_record['token'])
                isend=True
            elif len(lane_record['incoming'])<2 and len(lane_record['outgoing'])==0:
                # 如果这条线是一条结束的线，那么将这条线的索引加上后，结束
                combine_list.append(lane_record['token'])
                isend=True

        all_mergeList.append(combine_list)

    # 遍历所有的需要融合的列表
    for i in range(len(all_mergeList)):
        oneMergeLine={}
        points=[]
        for j in all_mergeList[i]:
            # 遍历所有该融合的直线
            points.append(all_straightLines[j]['points'])
        # 设置起始点和终止点
        points=list(chain(*points))
        oneMergeLine['points']=points
        oneMergeLine['start']=points[0]
        oneMergeLine['end'] = points[-1]
        oneMergeLine['mid']=points[int(np.ceil(len(points)/2))]
        oneMergeLine['isStraight']=True
        oneMergeLine['token']=i
        mergedSrightLines[i]=oneMergeLine
    return mergedSrightLines

def calcInOut(new_arclines_record):
    # 计算每条laned的in和out
    for key, value in new_arclines_record.items():
        value['incoming']=[]
        value['outgoing']=[]
        for key2,value2 in new_arclines_record.items():
            if key != key2:
                if value['start']==value2['end']:
                    value['incoming'].append(key2)
                if value['end']==value2['start']:
                    value['outgoing'].append(key2)


def combineSC(all_straightLines,all_mergedSLines):
    mergedLineDict={}
    num=0
    for key, value in all_straightLines.items():
        if value['isStraight']==False:
            # 首先添加曲线
            mergedLineDict[num]=value
            num+=1
    # 然后添加融合后的直线
    for key, value in all_mergedSLines.items():
        mergedLineDict[num]=value
        num+=1
    return mergedLineDict

def mergeLines(new_arclines_record):
    # 按照关键点来重新融合lanes
    mergedLines = {}  # 将融合好的直线加入这个字典中
    begin_lanes = []
    keypoints={'start':[],'end':[],'merge':[],'fork':[]} # 用来存储关键点：起始点，终止点，分叉点和合并点
    # 这里beginlanes有3种，分别是入度为0和入度>1的lane,以及出度大于1的所有的出的线
    for key, value in new_arclines_record.items():
        # 首先找到所有起始线段，incoming的长度为0
        if len(value['incoming'])==0:
            # 入度=0，说明该条线的起始点为一个start点
            begin_lanes.append(key)
            keypoints['start'].append(value['start'])
        if len(value['incoming']) > 1:
            # 入度大于1，则该条线的起始点为一个merge点
            begin_lanes.append(key)
            keypoints['merge'].append(value['start'])

        if len(value['outgoing']) > 1:
            # 出度大于1，则将所有的出的线作为begin_lanes
            keypoints['fork'].append(value['end']) # 该条线的终止点为分叉点
            for j in value['outgoing']:
                begin_lanes.append(j)

    # 从begin开始遍历
    all_mergeList = []
    for lane_index in begin_lanes:
        combine_list = []  # 每个begin lanes对应着一系列待拼接的点
        isend = False  # 判断是否要终止
        now_lane_index = lane_index
        while not isend:
            # 判断当前这条线的入度和出度
            lane_record = new_arclines_record[now_lane_index]  # 当前这条线的所有记录
            if len(lane_record['incoming']) < 2 and len(lane_record['outgoing']) == 1:
                # 如果lane的入点是start或者continue, 终点为continue
                combine_list.append(lane_record['token'])
                next_index = lane_record['outgoing'][0]
                if next_index not in begin_lanes:
                    # 如果下一条线的索引不在起始线中，则继续
                    now_lane_index = lane_record['outgoing'][0]
                else:
                    # 排除下一条线是merge线的情况
                    isend = True
            elif len(lane_record['outgoing']) > 1:
                # 此时这条线的终点开始分叉，那么将这条线的索引加上后，结束这条线
                # 如果incoming为0或者1，则加在原来的combine list中
                # 入伏哦incoming为2， 则本身就是一个start lane
                combine_list.append(lane_record['token'])
                isend = True

            elif len(lane_record['outgoing']) == 0:
                # 如果这条线是一条结束的线，那么将这条线的索引加上后，结束
                combine_list.append(lane_record['token'])
                isend = True
                keypoints['end'].append(lane_record['end']) # 该条线的终止点为一个关键点
            elif len(lane_record['incoming'])>1 and len(lane_record['outgoing'])==1:
                # 如果这条线起点是一个分叉点，终点是一个continue的点
                combine_list.append(lane_record['token'])
                now_lane_index = lane_record['outgoing'][0] # 下一条lane


        all_mergeList.append(combine_list)

    # 遍历所有的需要融合的列表
    for i in range(len(all_mergeList)):
        oneMergeLine = {}
        points = []
        for j in all_mergeList[i]:
            # 遍历所有该融合的直线
            points.append(new_arclines_record[j]['points'])
        # 设置起始点和终止点
        points = list(chain(*points))
        oneMergeLine['points'] = points
        oneMergeLine['start'] = points[0]
        oneMergeLine['end'] = points[-1]
        oneMergeLine['mid'] = points[int(np.ceil(len(points) / 2))]
        oneMergeLine['token'] = i
        mergedLines[i] = oneMergeLine
    return mergedLines,keypoints

def addControlPoints(mergedLines_records,keypoints,segmentPNums=20,AngleThreshold=30):
    """在已有线段上加上control points"""
    newRecords=copy.deepcopy(mergedLines_records) # 复制一份
    newkeypoints=copy.deepcopy(keypoints) # 复制一份
    newkeypoints['controls'] = []  # 用来记录控制点
    for key,value in mergedLines_records.items():
        points=value['points']
        newRecords[key]['controls'] = []  # 用来存储控制点的坐标
        # 首先默认每个20个打一个点，如果points的数量不足20，则跳过
        if len(points)<20:
            continue
        # 计算一条线上该分的段数
        segmentNums=len(points)//segmentPNums

        for i in range(segmentNums):
            segmentPoints=points[i*segmentPNums:min((i+1)*segmentPNums,len(points))]
            # 将首尾的点连起来，计算这个向量的角度
            startP=segmentPoints[0]
            endP=segmentPoints[-1]
            angle1=calcAngle(startP,endP)
            if i==0:
                firstA=angle1
            else:
                # 计算与firstA之间的角度差值，如果角度差大于某个值，则加一个control point
                diffTwo=abs(firstA-angle1) # 计算两个角度之间的差值
                if diffTwo>AngleThreshold:
                    firstA=angle1
                    newRecords[key]['controls'].append(startP)
                    newkeypoints['controls'].append(startP)
    return newRecords, newkeypoints



def calcAngle(startP,endP):
    diff = np.array(endP) - np.array(startP)
    angle1 = math.atan2(diff[1], diff[0])
    angle1 = -int(angle1 * 180 / math.pi)  # 返回[0,360]范围内的值
    if angle1 < 0:
        angle1 = 360 + angle1
    return angle1

def addLongLines(new_arclines_record,keypoints):
    # 判断一条线，如果中间的控制点小于一个数值，二等分或者三等分
    newLines_record=copy.deepcopy(new_arclines_record)
    newKeypoints=copy.deepcopy(keypoints)
    for key, value in new_arclines_record.items():
        controlPoints=value['controls']
        points=value['points']
        if not controlPoints:
            # 如果没有controlPoints,则根据Points的点数判断是二等分还是3等分
            controlList=dividelanes(points)
            newLines_record[key]['controls']+=controlList
            newKeypoints['controls']+=controlList
        else:
            # 如果有control points，则分段判断每一小段的点数
            sublists=[]
            for p in range(len(controlPoints)):
                # 找到在points中的索引
                control_index = points.index(controlPoints[p])
                if p==0:
                    # 如果是第一个control point, 加上前面到start的点
                    subpoints=points[:control_index]
                    sublists.append(subpoints)
                    # 如果只有一个点，还要加上到末尾的点
                    if len(controlPoints)==1:
                        subpoints=points[control_index:]
                        sublists.append(subpoints)
                else:
                    # 是第2，3，4...的控制点
                    beforeControlIndex=points.index(controlPoints[p-1])
                    subpoints = points[beforeControlIndex:control_index]
                    sublists.append(subpoints)
                    if p==len(controlPoints)-1:
                        # 如果是最后一个控制点
                        subpoints = points[control_index:]
                        sublists.append(subpoints)

            for s in sublists:
                controlList=dividelanes(s)
                newLines_record[key]['controls'] += controlList
                newKeypoints['controls'] += controlList
    return newLines_record, newKeypoints



def dividelanes(points):
    nums = len(points)
    controlList=[]
    if 100 < nums <= 200:
        # 二等分,取终点
        mid_index = int(np.ceil(len(points) / 2))
        mid_point = points[mid_index]
        controlList.append(mid_point)
    elif 200 < nums <= 300:
        # 三分，中间加两个点
        mid1 = int(np.ceil(len(points) / 3))
        mid2 = mid1 * 2
        controlList.append(points[mid1])
        controlList.append(points[mid2])
    elif nums>300:
        # 四分，三个点
        mid1 = int(np.ceil(len(points) / 4))
        mid2 = mid1 * 2
        mid3 = mid1 * 3
        controlList.append(points[mid1])
        controlList.append(points[mid2])
        controlList.append(points[mid3])
    return controlList

def turnKeysList(keypoints):
    # 将字典转为List
    keypointsList=[]
    for key, value in keypoints.items():
        keypointsList+=value
    return keypointsList
def turnKeysDict(keypoints):
    keyDict={}
    for key, value in keypoints.items():
        for point in value:
            if not point in keyDict.keys():
                keyDict[point]=[key]
            else:
                keyDict[point].append(key)
    return keyDict

def build_AdjacencyMatrix(new_arclines_record,keypointsList):
    # 根据关键点建立邻接矩阵

    keypoints_num = len(keypointsList)
    matrix = np.zeros([keypoints_num, keypoints_num])

    for key, value in new_arclines_record.items():
        # 遍历每条lane， 里面记录了lane的起始点，终止点和控制点
        points=value['points']
        controlPoints=value['controls'] # 控制点不止有一个，要判断哪个控制点在前，哪个控制点在后
        newcontrols=sortControlPoints(points,controlPoints) # 对控制点进行排序
        startP=value['start']
        endP=value['end']
        startP_index=keypointsList.index(startP)
        endP_index=keypointsList.index(endP)
        if not newcontrols:
            # 如果中间没有控制点,那么只有起始点和终止点
            matrix[startP_index,endP_index]=1
        else:
            # 如果中间有控制点，首先解决起点和终点的问题
            # 找到控制点的坐标
            c_1=keypointsList.index(newcontrols[0])
            matrix[startP_index,c_1]=1
            c_1 = keypointsList.index(newcontrols[-1])
            matrix[c_1,endP_index]=1
        # 然后解决其他的控制点的邻接关系

        for i in range(1,len(newcontrols)):
            # 从第一个控制点开始，而不是从第0个控制点开始
            # 找到当前控制点和后一个控制点的索引
            c_1=keypointsList.index(newcontrols[i-1])
            c_2=keypointsList.index(newcontrols[i])
            matrix[c_1, c_2] = 1

    return matrix

def sortControlPoints(points, controls):
    # 判断每个点在points中的索引，然后按照索引从小到大排序
    indexDict={}
    for i in range(len(controls)):
        index=points.index(controls[i]) # 找到某个控制点的索引
        indexDict[index]=controls[i]
    # 对字典的键进行排序
    indexNew=sorted(indexDict.keys())
    newcontrols = []
    for key in indexNew:
        # 将字典对应的值放入list中
        newcontrols.append(indexDict[key])
    return newcontrols

def sortedDictValues3(adict):
    items = adict.items()
    sorted(items)
    return [value for key, value in items]

def detectAbnormal(arclines_record,allpointsDict):
    new_arclines_record=copy.deepcopy(arclines_record)
    new_keypoints=allpointsDict
    #
    for key, value in arclines_record.items():
        startP=value['start']
        endP=value['end']
        controlP=value['controls']
        points=value['points']
        startType=allpointsDict[startP]
        endType = allpointsDict[endP]

        if startType[0]=='start' and (len(endType)>1 or endType[0]=='merge') and len(controlP)==0 :
            # 如果lane的起点是start，终点是merge，且中间没有control point, 加一个control point
            # len(endType)>1，且如果所有的都是'end'
            addP_index=int(np.ceil(len(points)/2))
            cp=points[addP_index] # control points的点的具体位置
            # 加到关键点和原来的lane_record中
            new_arclines_record[key]['controls'].append(cp)
            new_keypoints[cp]=['controls']
        if (len(startType)>1 or startType[0]=='fork') and endType[0]=='end' and len(controlP)==0 :
            # 如果起点是两条线的start点或者就是一个fork点
            if set(startType)=={'start'}:
                continue
            addP_index=int(np.ceil(len(points)/2))
            cp=points[addP_index] # control points的点的具体位置
            # 加到关键点和原来的lane_record中
            new_arclines_record[key]['controls'].append(cp)
            new_keypoints[cp]=['controls']

        if startType[0]=='fork' and endType[0]=='merge' and len(controlP)==1:
            # 如果lane的起点是fork，终点是merge, 且中间只有一个control point, 则重新三等分（两个点）
            addP_index = int(np.ceil(len(points) / 3))
            i1=addP_index
            i2=addP_index*2
            cp1=points[i1]
            cp2=points[i2]
            # 将原来的control points点删去
            del new_keypoints[controlP[0]]
            # 加到关键点和原来的lane_record中
            new_arclines_record[key]['controls']=[cp1,cp2]
            new_keypoints[cp1]=['controls']
            new_keypoints[cp2] = ['controls']
    return new_arclines_record, new_keypoints

def turnDictList(allpointsDict):
    # 将所有点放在一个list中
    allPointsList=[]
    for key in allpointsDict.keys():
        allPointsList.append(key)
    return allPointsList


def getKeysControls(ego_pose_token, e1_data, nusc_map, width, height, isPlot=False):
    
    fig = None
    ax = None
    """利用api中的函数获得e1_data周围的所有lane和lane_connectors"""
    # 首先获得一个大范围内的所有lane和lane_connector
    e1_pose = e1_data['translation']
    my_patch = (e1_pose[0] - 20, e1_pose[1] - 60, e1_pose[0] + 120, e1_pose[1] + 60)  # (x_min,y_min,x_max,y_max)
    records = nusc_map.get_records_in_patch(my_patch, layer_names=['lane', 'lane_connector'], mode='intersect')

    """转成中心线"""
    # 然后转成中心线
    pose_dict = my_arclines_centerline(nusc_map, records, resolution_meters=0.1)

    """转到车身坐标系"""
    rotated_pose_dict = Turnto_vehicle2(pose_dict, e1_data)

    """获得24*48矩形内的所有arcline记录"""
    patch_acrlines_record = get_arclines_in_patch(rotated_pose_dict, width=width, height=height)
    if isPlot:
        fig, ax = polt_arclines(patch_acrlines_record, width=width, height=height,printRadius=False,printIndex=False)
        # fig.savefig('tmp.jpg')
    """根据end 和end ,start和start的距离,融合太近的点"""
    # 首先计算start和start的距离
    new_arclines_record = merge_tooclose(patch_acrlines_record)
    new_arclines_record = add_tokens(new_arclines_record)  # 为每个lane增加一个token

    """融合线"""
    # 计算所有lane的入度和出度
    calcInOut(new_arclines_record)
    # 以关键点为界，融合线段
    mergedLines_records, keypoints = mergeLines(new_arclines_record)
    # 删除太短的线
    mergedLines_records, keypoints = del_shortLane(mergedLines_records,keypoints,threshold=50)
    if isPlot:
        fig, ax = polt_arclines(mergedLines_records, width=width, height=height, printIndex=False)
    # fig.show()
    """沿着线先分割比较密集的点"""
    # 首先加上曲线上的控制点
    new_arclines_record, keypoints = addControlPoints(mergedLines_records, keypoints, AngleThreshold=20)
    # 然后加直线上的控制点
    new_arclines_record, keypoints = addLongLines(new_arclines_record, keypoints)
    if isPlot:
        fig, ax = plot_keypoints(keypoints, fig, ax,printIndex=False)
    # fig.show()
    """建立关键点list"""
    # 创建一个点对应一个类型的dict
    allpointsDict = turnKeysDict(keypoints)  # 将turnKeysDict
    # 判断异常值
    new_arclines_record, allpointsDict = detectAbnormal(new_arclines_record, allpointsDict)
    if isPlot:
        fig, ax = plot_pointsDict(allpointsDict, fig, ax,printIndex=False)
    # fig.show()
    # 将字典值转为List
    allpointsList = turnDictList(allpointsDict)

    return new_arclines_record, allpointsList, allpointsDict, fig, ax

def saveAll(ego_path,ego_index,allpointsList,matrix,fig,ax):
    img_name = ego_path + '.jpg'.format(ego_index)
    fig.savefig(img_name)
    # 保存关键点
    keypoints_path = ego_path + '_' + 'keypoints.npy'
    np.save(keypoints_path, allpointsList)  # 将新的点保存到矩阵中
    # 保存矩阵
    matrix_path = ego_path + '_' + 'matrix.npy'
    np.save(matrix_path, matrix)

def saveJson(tokens,json_path):
    jsondata = json.dumps(tokens, indent=4, separators=(',', ': '))
    f = open(json_path, 'w')
    f.write(jsondata)
    f.close()

def del_shortLane(mergedLines_records,keypoints,threshold):
    """
    删除太短的线
    """
    new_lanes_records=copy.deepcopy(mergedLines_records)
    for k,v in mergedLines_records.items():
        points=v['points']
        startP=v['start']
        endP=v['end']
        if len(points)<threshold and startP not in keypoints['merge'] and startP not in  keypoints['fork'] and endP not in keypoints['merge'] and endP not in keypoints['fork']:
            # 删除该条线段
            del new_lanes_records[k]
            if startP in keypoints['start']:
                keypoints['start'].remove(startP)
            if endP in keypoints['end']:
                keypoints['end'].remove(endP)

    return new_lanes_records,keypoints