import numpy as np
import json
import os
import torch

if __name__ == '__main__':
  
  root_test = 'saved_gt2'
  
  
  # with open(os.path.join('saved_gt', 'empty_scene_dict.json'), 'r') as f:
  #   empty_scene_data = json.load(f)
  
  # with open(os.path.join('saved_gt', 'unknow_scene.json'), 'r') as f:
  #   unknow_scene_data = json.load(f)
  
  # with open(os.path.join('saved_gt', 'bad_scenes.json'), 'r') as f:
  #   bad_data = json.load(f)
  
  with open(os.path.join('saved_gt', 'lane_gt.json'), 'r') as f:
    data = json.load(f)
  N = len(data)
  
  key_list = [
    'a923d5d2c2af4127ace3ff403fa21d78',
    '4f2c26aaed534dd68fe22793e7a0e30b',
    'b78599520c2b4993983a5f4feb718402',
    '722389bde00a4b57ace67686cdfd2325',
    '66202485ba6f4fd49162cb227c92a1d3',
    '4e6728fdd8cb487486c495a3bfed1cf3',
    '3fad9ea6895244ceaf1b0306e02eadda',
    'a05edf71b4b14052bb477918d0997c88',
    '4500107a2afe4854957be56ec0a11337',
    '89ca4a7f7c024623aac04759922e30bd',
    'd2f63505f315460c80b7a54560e2a8ea',
    '1cd46c8840dd4c9784b08034542e64aa',
    '8092c97c019b4996a8edba84c15ed168',
    '6a38e09fa2ba49828d0d15aa558dbb55',
    'a229d1703f844607b3e7b99caa8f565d',
    '10a0c1bd584641c297b9c2bceed6c4ce',
    'ba22536d4890443d9622199dc598fbe7',
    '7bd1d39fef274cd7801c8e4a5feac6fd',
    'a73d0005a38c44e69c6b722b97cf9aab',
    '44ed92effc924a79880b4b4176dd7b09'
  ]
  for i, key in enumerate(key_list):
    gt_item = data[key]
    gt_array = torch.tensor(gt_item['gt_array'])
    KeyMatrix = torch.tensor(gt_item['KeyMatrix']).float()
    print(key, gt_array.shape, KeyMatrix.shape)
    if gt_item['gt_array'] is None or gt_item['gt_dict'] is None or gt_item['KeyMatrix'] is None:
      print(f"{i}/{N},  {key}")
  

  print('end')