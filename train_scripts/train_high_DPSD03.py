import copy
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import sys
sys.path.append('./')
sys.path.append('./src/')
sys.path.append('./Tool_list/')
sys.path.append('./VCMeshConv/')
import random
from tqdm import tqdm
import numpy as np
import scipy
from scipy.spatial.transform import Rotation
import json
import trimesh
import torch
from dataset import *
from torch.utils.data import DataLoader, Dataset
from torch_geometric.transforms import FaceToEdge
import torch.nn.functional as F
import time

from src.obj_parser import Mesh_obj
from src.SSDRLBS import SSDRLBS
from src.models import *

from VCMeshConv.GraphAutoEncoder import graphAE_param_iso as Param
from VCMeshConv.GraphAutoEncoder.graphVAE_train import Net_autoencpsdhigh
from psbody.mesh import Mesh, MeshViewers

config_path = "assets/dress03/config.json"
with open(config_path, "r") as f:
        config = json.load(f)
dir_name="./VirtualBones/NetWork_list/high_DPSD03/"
if not os.path.exists(dir_name):
    os.makedirs(dir_name)
train_path=config["train_path"]
test_path=config["test_path"]
low_path=config["low_path"]
lap_path=config["lap_path"]
learning_rate=0.001
time_steps=50

logname = os.path.join(dir_name, 'log.txt')

import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
writer = SummaryWriter("runs_high/DPSD3")

class AverageValueMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
train_loss = AverageValueMeter()
val_loss_L2 = AverageValueMeter()

#---------创建网络
BatchSize=config["batchsize"]
gru_dim = config["gru_dim"]
gru_out_dim = config["gru_out_dim"]
joint_list = config["joint_list"]
ssdrlbs_bone_num = config["ssdrlbs_bone_num"]

ssdrlbs_root_dir = config["ssdrlbs_root_dir"]
ssdrlbs_net_path = config["ssdrlbs_net_path"]
detail_net_path = config["detail_net_path"]
state_path = config["state_path"]
low_res_path = config["low_res_path"]
obj_template_path = config["obj_template_path"]

template_mesh=trimesh.load("./VirtualBones/assets/dress03/garment.obj",process=False)

# takes cuda torch variable repeated batch time

garment_template = Mesh_obj(obj_template_path)

cloth_vet_num=garment_template.v.shape[0]
cloth_tem=torch.from_numpy(garment_template.v).float().cuda()
from torch.optim import lr_scheduler
index_list=([3,0,2],[0,1,4],[0,2,5],[0,3,6],[1,4,7],[2,5,8],[3,6,9],[4,7,10],[5,8,11],[6,9,12],[7,10,10],[8,11,11],[9,12,15],[9,13,16],[9,14,17],[12,15,15],[13,16,18],[14,17,19],[16,18,18],[17,19,19])
joint_num = len(joint_list)
ssdr_model_list=[]
optimizer_list=[]
scheduler_list=[]
data = FaceToEdge()(Data(num_nodes=garment_template.v.shape[0],
                            face=torch.from_numpy(garment_template.f.astype(int).transpose() - 1).long()))



import itertools
device = torch.device(f'cuda' if torch.cuda.is_available() else "cpu")
param=Param.Parameters()
param.read_config("/home/djq19/workfiles/VirtualBones/VCMeshConv/GraphAutoEncoder/config_train.config")
GarmentNet=Net_autoencpsdhigh(param,data.edge_index,8744)
GarmentNet=GarmentNet.cuda()

def getoptim(learningrate, net_autoenc,fieldnet,mulweight):    
    ae_params = filter(lambda x: x.requires_grad,
                       itertools.chain(
                                       net_autoenc.net_geodec.parameters(),
                                       net_autoenc.net_geoenc.parameters(),
                                       net_autoenc.mcvcoeffsdec.parameters(),
                                       net_autoenc.mcvcoeffsenc.parameters(),
                                       ))
    
    ae_optim = torch.optim.Adam(itertools.chain(ae_params,fieldnet.parameters(),[mulweight]), lr=learning_rate, betas=(0.9, 0.999))    

    return ae_optim

ssdrlbs_net_path ="./NetWork_list/net_path/ssdr_modellist_ours03.pkl"
ssdrlbs_weight_path="./VirtualBones/NetWork_list/net_path/PoseWeightmat_ours03.pkl"
fieldpath="./VirtualBones/NetWork_list/net_path/FieldNet_ours03.pkl"

ssdr_model_list=torch.load(ssdrlbs_net_path)
for i in range(20):
    ssdr_model_list[i].eval()
mul_weight_list=torch.load(ssdrlbs_weight_path)
mul_weight_list=mul_weight_list.cuda()
FieldNet=torch.load(fieldpath)
FieldNet=FieldNet.cuda()
FieldNet.eval()

detailnet=VirtualposeMlP(480,128,64)
detailnet=detailnet.cuda()
DPSD_=torch.rand(20,64,8744,3).cuda()
DPSD_mat=torch.nn.Parameter(DPSD_.data,requires_grad=True)

optimizer = torch.optim.Adam(itertools.chain(detailnet.parameters(),[DPSD_mat]), lr=learning_rate, betas=(0.9, 0.999))
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)


#---------加载数据
Date_train=Anim_list(train=True,Train_path=train_path,Test_path=test_path,Low_path=low_path,Lap_path=lap_path,faceidx=data.edge_index)
Train_loader = torch.utils.data.DataLoader(Date_train, batch_size=BatchSize,
                                         shuffle=True, num_workers=int(1), drop_last=True)

Date_test=Anim_list(train=False,Train_path=train_path,Test_path=test_path,Low_path=low_path,Lap_path=lap_path,faceidx=data.edge_index)
Test_loader = torch.utils.data.DataLoader(Date_test, batch_size=1,
                                         shuffle=False, num_workers=int(1), drop_last=False)

ssdrlbs = SSDRLBS(os.path.join(ssdrlbs_root_dir, "u.obj"),
                      os.path.join(ssdrlbs_root_dir, "skin_weights.npy"),
                      os.path.join(ssdrlbs_root_dir, "u_trans.npy"),
                      DPSD_.device)

state = np.load(state_path)

cloth_pose_mean = torch.from_numpy(state["cloth_pose_mean"]).cuda()
cloth_pose_mean=cloth_pose_mean.unsqueeze(0)
cloth_pose_std = torch.from_numpy(state["cloth_pose_std"]).cuda()
cloth_pose_std=cloth_pose_std.unsqueeze(0)
cloth_trans_mean = torch.from_numpy(state["cloth_trans_mean"]).cuda()
cloth_trans_mean=cloth_trans_mean.unsqueeze(0)
cloth_trans_std = torch.from_numpy(state["cloth_trans_std"]).cuda()
cloth_trans_std=cloth_trans_std.unsqueeze(0)

ssdr_res_mean =  torch.from_numpy(state["ssdr_res_mean"]).cuda()
ssdr_res_mean=ssdr_res_mean.unsqueeze(0)
ssdr_res_std =  torch.from_numpy(state["ssdr_res_std"]).cuda()
ssdr_res_std=ssdr_res_std.unsqueeze(0)

vert_std = state["sim_res_std"]
vert_mean = state["sim_res_mean"]

#---------训练

for epoch in range(config["nepoch"]):

    detailnet.train()
    train_loss.reset()
    for data,low_res,query_list,trans_label,targetmodel_list in tqdm(Train_loader):
        pose_arr_all=data["pose"]
        trans_arr_all=data["trans"]
        sim_res_all=low_res#
        pose_arr_all=(pose_arr_all-state["pose_mean"])/state["pose_std"]
        trans_head=trans_arr_all[:,0,:].clone().detach()
        trans_head=trans_head.unsqueeze(1).expand(trans_arr_all.size())
        trans_arr_all=(trans_arr_all-trans_head- state["trans_mean"]) / state["trans_std"]
        pose_arr_all=pose_arr_all.transpose(0,1).cuda()
        trans_arr_all=trans_arr_all.transpose(0,1).cuda()
        sim_res_all=sim_res_all.transpose(0,1).cuda()        #从(Batch,500,...)转为(500,Batch)
        query_list=query_list.transpose(0,1).cuda()
        trans_label=trans_label.transpose(0,1).cuda()
        targetmodel_list=targetmodel_list.transpose(0,1).cuda()

        hidden_list=[]
        ssdr_hidden_list = []
        ssdr_hidden_temp=[]
        for i in range(20):
            ssdr_hidden_temp.append((None,None))
        ssdr_hidden_list.append(ssdr_hidden_temp)
        detail_hidden=None
        for pose_arr,trans_arr,sim_res,query,trans_label,targetmodel in zip(pose_arr_all.split(time_steps,0),trans_arr_all.split(time_steps,0),sim_res_all.split(time_steps,0),query_list.split(time_steps,0),trans_label.split(time_steps,0),targetmodel_list.split(time_steps,0)): #按time_step=50划分数据
            outputs=[]
            targets=[]
            output_detail=[]
            for data_item in range(time_steps):                                                                                             
                hidden_temp=[]
                pred_rot_trans_list_mu=torch.zeros((BatchSize,20,480)).cuda()# 480
                for itt in range(20):
                    motion_signature = np.zeros((1,BatchSize,3*6), dtype=np.float32) 
                    motion_signature = torch.from_numpy(motion_signature)
                    motion_signature=motion_signature.cuda()
                    for j in range(3):
                        motion_signature[:,:,j * 6: j * 6 + 3] = pose_arr[data_item,:, index_list[itt][j]]   
                        motion_signature[:,:,j * 6+3: j * 6 + 6] = trans_arr[data_item,:]
                    #构建运动特征
                    motion_signature = motion_signature.view((1*BatchSize, -1))
                    ssdr_hidden=ssdr_hidden_list[-1][itt][1]
                    tmu, new_ssdr_hidden = ssdr_model_list[itt](motion_signature, ssdr_hidden)
                    pred_rot_trans_list_mu[:,itt]=tmu
                    #保存隐层
                    hidden_temp.append((ssdr_hidden,new_ssdr_hidden))
                    
                ssdr_hidden_list.append(hidden_temp)

                loss_item,_=GarmentNet(targetmodel[data_item],tnr_temp[data_item],ssdrlbs,mul_weight_list,query[data_item],\
                            cloth_pose_std,cloth_pose_mean,cloth_trans_std,cloth_trans_mean,FieldNet,pred_rot_trans_list_mu,detailnet,DPSD_mat,ssdr_res_std,ssdr_res_mean)
            
                output_detail.append(loss_item)

            output_detail_list=torch.cat(output_detail,dim=0)
            output_detail_list=torch.mean(output_detail_list)

            optimizer.zero_grad()
            loss=output_detail_list
            loss.backward( torch.ones_like(output_detail_list),retain_graph=True)
            optimizer.step()
        train_loss.update(loss.item())
    scheduler.step()
    writer.add_scalar('Loss/Train_loss',train_loss.avg,epoch)
    if epoch%10==0:
        with torch.no_grad():
            detailnet.eval()
            detail_hidden=None
            val_loss_L2.reset()
            
            for data,low_res,query_list,trans_label,targetmodel_list in tqdm(Test_loader):

                pose_arr=data["pose"]
                trans_arr=data["trans"]
                sim_res=low_res   #data["sim_res"][:,:10]#   #
                pose_arr=pose_arr.transpose(0,1)
                trans_arr=trans_arr.transpose(0,1)
                pose_arr=(pose_arr-state["pose_mean"])/state["pose_std"]
                trans_head=trans_arr[0,:,:].clone().detach()
                trans_head=trans_head.unsqueeze(0).expand(trans_arr.size())
                trans_arr=(trans_arr-trans_head- state["trans_mean"]) / state["trans_std"]
                test_batch=sim_res.size(0) 
                pose_arr=pose_arr.cuda()
                trans_arr=trans_arr.cuda()
                sim_res=sim_res.cuda()
                query=query_list.transpose(0,1).cuda()
                targetmodel=targetmodel_list.transpose(0,1).cuda()
                tnr_temp=tnr_list.transpose(0,1).cuda()
                ssdr_hidden_list=[]
                ssdr_hidden_temp=[]
                for i in range(20):
                    ssdr_hidden_temp.append((None,None))
                ssdr_hidden_list.append(ssdr_hidden_temp)
                for i in range(500):
                    hidden_temp=[]
                    pred_rot_trans_mu=torch.zeros((test_batch,20,480)).cuda()
                    for itt in range(20):
                        motion_signature = np.zeros((1,test_batch,3*6), dtype=np.float32) 
                        motion_signature = torch.from_numpy(motion_signature)
                        motion_signature=motion_signature.cuda()
                        for j in range(3):
                            motion_signature[:,:,j * 6: j * 6 + 3] = pose_arr[i,:, index_list[itt][j]]   #数据格式(1,batch,52,3)
                            motion_signature[:,:,j * 6+3: j * 6 + 6] = trans_arr[i,:]
                        #构建运动特征
                        motion_signature = motion_signature.view((1*test_batch, -1))
                        #输入到网络
                        ssdr_hidden=ssdr_hidden_list[-1][itt][1]
                        tmu, new_ssdr_hidden= ssdr_model_list[itt](motion_signature, ssdr_hidden)
                        pred_rot_trans_mu[:,itt]=tmu

                        hidden_temp.append((ssdr_hidden,new_ssdr_hidden))
                        
                    ssdr_hidden_list.append(hidden_temp)
                    _,final_res=GarmentNet(targetmodel[i],tnr_temp[i],ssdrlbs,mul_weight_list,query[i],\
                            cloth_pose_std,cloth_pose_mean,cloth_trans_std,cloth_trans_mean,FieldNet,pred_rot_trans_mu,detailnet,DPSD_mat,ssdr_res_std,ssdr_res_mean)
                    final_res=final_res.reshape(sim_res[0,i].size())      
                    loss=torch.mean(torch.sqrt(torch.sum((final_res-sim_res[0,i])**2,dim=1)))
                    val_loss_L2.update(loss.item())
            print("end_loss",val_loss_L2.avg)   
            torch.save(detailnet, '%s/detailnet_64normal03.pkl' % (dir_name))
            torch.save(DPSD_mat, '%s/DPSD_64normal03.pkl' % (dir_name))
            log_table = {
                "train_loss_L2": train_loss.avg,
                "val_loss_L2": val_loss_L2.avg,
                "epoch": epoch,
            }
            print(log_table)
            with open(logname, 'a') as f:  # open and append
                f.write('json_stats: ' + json.dumps(log_table) + '\n')




#---------保存网络