#Coding By DiaoJunqi 2022.7.23
from __future__ import print_function
import torch.utils.data as data
from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
import time
import os
import trimesh


Target_root="./VirtualBones/VirtualBonesDataset/dress03/"
Model1_root="./VirtualBones/"
Model2_root="./VirtualBones/"


def Hausdorff(PointA_list,PointB_list):
    H_dist=torch.zeros(PointA_list.size(0)).cuda()
    for ith in range(PointA_list.size(0)):
        pointA_temp=PointA_list[ith]
        pointA_temp=pointA_temp.unsqueeze(0)
        eudic=torch.sqrt(torch.sum((PointB_list-pointA_temp)**2,dim=1))
        dist=torch.min(eudic)
        H_dist[ith]=dist
        #print(dist)
    return H_dist.max()
    

id=1
disend1=0
disend2=0
from hausdorff import hausdorff_distance
test_list=['86.npz', '46.npz', '58.npz', '81.npz', '26.npz']
for testid in test_list:
    disttemp1=0
    disttemp2=0
    Target_path=Target_root+testid
    targetlist=np.load(Target_path)
    Model1_path=Model1_root+"results_list/pretrain/vaegcn03_pose128/"+str(id)+"/"#Model1_root+"results_list/pretrain/virknn9/"+str(id)+"/"
    for i in range(500):
        Target_res=targetlist['sim_res'][i]
        Target_vts=Target_res
        model1=trimesh.load(Model1_path+str(i)+".obj",process=False)
        model1_vts=model1.vertices
        disend1+= hausdorff_distance(model1_vts, Target_vts, distance="euclidean") 

    id+=1
print("8model1",disend1/2500)


