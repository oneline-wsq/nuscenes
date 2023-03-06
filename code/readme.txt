onePoseExample.py: 一个场景下某一个ego pose中的keypoints, control points和邻接矩阵；

miniALL.py: 生成mini数据集10个场景下的所有ego pose 数据；

exchangeOrder: 
- exchangePoints.py: 两两换序，重新生成点序列和邻接矩阵并保存；

ego_poses ：Mini数据集的所有关键点+控制点和邻接矩阵。
- ego_poses/scene_0/egopose_0_exchange: 两两交换后的点序列和邻接矩阵；（不包括原始的点序列和邻接矩阵）
