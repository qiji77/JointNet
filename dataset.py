#Coding By DiaoJunqi 2022.7.23
from __future__ import print_function
import torch.utils.data as data
from torch.utils.data import DataLoader, Dataset
from scipy.spatial.transform import Rotation
import torch
import numpy as np
import time
import os
import trimesh

def unwrap_self(arg, **kwarg):
    return arg[0]._getitem(*(arg[1:]), **kwarg)
targetmodel_path="./VirtualBones/VirtualBonesDataset/dress03_mesh_res_2/"


class Anim_list(data.Dataset):
    def __init__(self, train,Train_path,Test_path,Low_path,Lap_path,faceidx):
        self.train = train
        self.datas_pose = np.zeros([1,52,3])
        self.datas_trans=np.zeros([1,3])
        self.datas_sim_res=np.zeros([1,8744,3])
        self.lapx_v=[]
        self.lapx_i=[]
        self.datas=[]
        self.low_res=[]
        self.targetmodel_list=[]
        self.faceidx=faceidx.long()
        if self.train:
            path=os.listdir(Train_path)
            for item in path:
                if item[-4:] == ".npz" and item not in  ['86.npz', '46.npz', '58.npz', '81.npz', '26.npz']:
                    path_temp=Train_path+item                            
                    data_temp=np.load(path_temp)                           #加载数据
                    if data_temp["pose"].shape[0]<500:
                        continue
                    ttemp=np.load(targetmodel_path+item[:-4]+".npy")
                    self.targetmodel_list.append(ttemp)
                    self.datas.append(data_temp)                           
                    low_data_temp=np.load(Low_path+item[:-4]+".npy")       #laplace平滑内容
                    self.low_res.append(low_data_temp)
                    

        else:
            path=os.listdir(Test_path)
            for item in ['58.npz','81.npz']:
                if item[-4:] == ".npz":
                    path_temp=Train_path+item                            
                    data_temp=np.load(path_temp)                           #加载数据
                    if data_temp["pose"].shape[0]<500:
                        continue
                    self.datas.append(data_temp)
                   
                    ttemp=np.load(targetmodel_path+item[:-4]+".npy")
                    
                    self.targetmodel_list.append(ttemp)                         
                    low_data_temp=np.load(Low_path+item[:-4]+".npy")       #laplace平滑内容
                    self.low_res.append(low_data_temp)

        self.targetmodel_list=np.array(self.targetmodel_list)




    def __getitem__(self, index):

        low_res=self.low_res[index]

        query2=np.load("/home/djq19/workfiles/VirtualBones/assets/dress03/SSDR/u_trans.npy")
        query2=[query2 for i in range(500)]
        query2=np.array(query2)
        query=query2.astype('float32')

        datas=self.datas[index]
        tarmodel=self.targetmodel_list[index]
        tarmodel=tarmodel.astype('float32')

        return datas,low_res,query,tarmodel



    def __len__(self):
        return len(self.datas)
if __name__ == '__main__':
    import random
    print("Import Dataset")
