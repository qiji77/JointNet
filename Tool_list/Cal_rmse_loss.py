#Coding By DiaoJunqi 2022.7.23
from __future__ import print_function
import torch.utils.data as data
from torch.utils.data import DataLoader, Dataset

import torch
import numpy as np
import time
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import trimesh

Target_root="./VirtualBones/VirtualBonesDataset/dress03/"
Model1_root="./VirtualBones/"
Model2_root="./VirtualBones/"
dist_all=torch.zeros(5)
id=1
disend1=0

test_list=['86.npz', '46.npz', '58.npz', '81.npz', '26.npz']
for testid in test_list:
    Target_path=Target_root+testid
    targetlist=np.load(Target_path)
    Model1_path=Model1_root+"results_list/pretrain/vaegcn03_pose128/"+str(id)+"/"
    
    disttemp1=0
    for i in range(500):
        Target_res=targetlist['sim_res'][i]
        Target_vts=torch.from_numpy(Target_res)
        model1=trimesh.load(Model1_path+str(i)+".obj",process=False)
        model1_vts=torch.from_numpy(model1.vertices)
        model1_vts=model1_vts.float()
        dist1=torch.mean(torch.sqrt(torch.sum((model1_vts-Target_vts)**2,dim=1))) 
        disttemp1+=dist1
    dist_all[id-1]=disttemp1/500
    print("temp1 %.6f" %(disttemp1/500))
    id+=1
print("11 all dis is %.6f"%torch.mean(dist_all))
