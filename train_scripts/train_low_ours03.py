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
from torch.utils.data import DataLoader, Dataset
from torch_geometric.transforms import FaceToEdge
import torch.nn.functional as F
def seed_torch(seed=128):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) 
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
 
seed_torch()
from dataset import *
import time

from src.obj_parser import Mesh_obj
from src.SSDRLBS import SSDRLBS
from src.models import *

from VCMeshConv.GraphAutoEncoder import graphAE_param_iso as Param
from VCMeshConv.GraphAutoEncoder.graphVAE_train import garmentloss
from psbody.mesh import Mesh, MeshViewers

config_path = "assets/dress03/config.json"
with open(config_path, "r") as f:
        config = json.load(f)
dir_name="/home/djq19/workfiles/VirtualBones/NetWork_list/Net_"#config["Net_path"]
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
writer = SummaryWriter()

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
joint_list = config["joint_list"]
ssdrlbs_bone_num = config["ssdrlbs_bone_num"]
ssdrlbs_root_dir = config["ssdrlbs_root_dir"]
state_path = config["state_path"]
obj_template_path = config["obj_template_path"]


template_mesh=trimesh.load("./VirtualBones/assets/dress03/garment.obj",process=False)
faces = template_mesh.faces
faces = [faces for i in range(BatchSize*time_steps)]
faces = np.array(faces)
faces = torch.from_numpy(faces).cuda()
# takes cuda torch variable repeated batch time

garment_template = Mesh_obj(obj_template_path)

from torch.optim import lr_scheduler



index_list=([3,0,2],[0,1,4],[0,2,5],[0,3,6],[1,4,7],[2,5,8],[3,6,9],[4,7,10],[5,8,11],[6,9,12],[7,10,10],[8,11,11],[9,12,15],[9,13,16],[9,14,17],[12,15,15],[13,16,18],[14,17,19],[16,18,18],[17,19,19])
joint_num = len(joint_list)
ssdr_model_list=[]
optimizer_list=[]
scheduler_list=[]
mul_weight=torch.rand(80,20).cuda()
mul_weight_list=torch.nn.Parameter(mul_weight.data,requires_grad=True)

for ith in range(20):
    ssdr_model = GRU_Model(3 * 6, gru_dim, [ssdrlbs_bone_num * 6])
    ssdr_model = ssdr_model.cuda()
    ssdr_model_list.append(ssdr_model)

import itertools

data = FaceToEdge()(Data(num_nodes=garment_template.v.shape[0],
                            face=torch.from_numpy(garment_template.f.astype(int).transpose() - 1).long()))

param=Param.Parameters()
param.read_config("./VirtualBones/VCMeshConv/GraphAutoEncoder/config_train.config")
Garmentloss=garmentloss(param,data.edge_index)
Garmentloss=Garmentloss.cuda()

FieldNet=VposeFiled(480,1024,6)
FieldNet=FieldNet.cuda()
optimizer=torch.optim.RMSprop(itertools.chain( ssdr_model_list[0].parameters() ,\
ssdr_model_list[1].parameters() ,\
ssdr_model_list[2].parameters() ,\
ssdr_model_list[3].parameters() ,\
ssdr_model_list[4].parameters() ,\
ssdr_model_list[5].parameters() ,\
ssdr_model_list[6].parameters() ,\
ssdr_model_list[7].parameters() ,\
ssdr_model_list[8].parameters() ,\
ssdr_model_list[9].parameters() ,\
ssdr_model_list[10].parameters() ,\
ssdr_model_list[11].parameters() ,\
ssdr_model_list[12].parameters() ,\
ssdr_model_list[13].parameters() ,\
ssdr_model_list[14].parameters() ,\
ssdr_model_list[15].parameters() ,\
ssdr_model_list[16].parameters() ,\
ssdr_model_list[17].parameters() ,\
ssdr_model_list[18].parameters() ,\
ssdr_model_list[19].parameters() ,\
[mul_weight_list],\
FieldNet.parameters()
),lr=learning_rate)

scheduler = lr_scheduler.StepLR(optimizer, 30, gamma=0.7, last_epoch=-1)

#---------加载数据
Date_train=Anim_list(train=True,Train_path=train_path,Test_path=test_path,Low_path=low_path,Lap_path=lap_path,faceidx=data.edge_index)
Train_loader = torch.utils.data.DataLoader(Date_train, batch_size=BatchSize,
                                         shuffle=True, num_workers=int(4), drop_last=True)

Date_test=Anim_list(train=False,Train_path=train_path,Test_path=test_path,Low_path=low_path,Lap_path=lap_path,faceidx=data.edge_index)
Test_loader = torch.utils.data.DataLoader(Date_test, batch_size=1,
                                         shuffle=False, num_workers=int(4), drop_last=False)

ssdrlbs = SSDRLBS(os.path.join(ssdrlbs_root_dir, "u.obj"),
                      os.path.join(ssdrlbs_root_dir, "skin_weights.npy"),
                      os.path.join(ssdrlbs_root_dir, "u_trans.npy"),
                      faces.device)

state = np.load(state_path)

cloth_pose_mean = torch.from_numpy(state["cloth_pose_mean"]).cuda()
cloth_pose_mean=cloth_pose_mean.unsqueeze(0)
cloth_pose_std = torch.from_numpy(state["cloth_pose_std"]).cuda()
cloth_pose_std=cloth_pose_std.unsqueeze(0)
cloth_trans_mean = torch.from_numpy(state["cloth_trans_mean"]).cuda()
cloth_trans_mean=cloth_trans_mean.unsqueeze(0)
cloth_trans_std = torch.from_numpy(state["cloth_trans_std"]).cuda()
cloth_trans_std=cloth_trans_std.unsqueeze(0)
ssdr_res_mean = state["ssdr_res_mean"]
ssdr_res_std = state["ssdr_res_std"]


#---------训练
for epoch in range(config["nepoch"]):
    FieldNet.train()
    for moi in range(20):
        ssdr_model_list[moi].train()
    train_loss.reset()
    for data,low_res,query_list,trans_label in tqdm(Train_loader):
        pose_arr_all=data["pose"]
        trans_arr_all=data["trans"]
        sim_res_all=low_res
        pose_arr_all=(pose_arr_all-state["pose_mean"])/state["pose_std"]
        trans_head=trans_arr_all[:,0,:].clone().detach()
        trans_head=trans_head.unsqueeze(1).expand(trans_arr_all.size())
        trans_arr_all=(trans_arr_all-trans_head- state["trans_mean"]) / state["trans_std"]
        pose_arr_all=pose_arr_all.transpose(0,1).cuda()
        trans_arr_all=trans_arr_all.transpose(0,1).cuda()
        sim_res_all=sim_res_all.transpose(0,1).cuda()   
        query_list=query_list.transpose(0,1).cuda()
        trans_label=trans_label.transpose(0,1).cuda()

        hidden_list=[]
        ssdr_hidden_list = []
        ssdr_hidden_temp=[]
        for i in range(20):
            ssdr_hidden_temp.append((None,None))
        ssdr_hidden_list.append(ssdr_hidden_temp)
        for pose_arr,trans_arr,sim_res,query,trans_label in zip(pose_arr_all.split(time_steps,0),trans_arr_all.split(time_steps,0),sim_res_all.split(time_steps,0),query_list.split(time_steps,0),trans_label.split(time_steps,0)): #按time_step=50划分数据
            outputs=[]
            tnor_list=[]
            for data_item in range(time_steps):                                                                                                
                hidden_temp=[]
                sim_res_temp=sim_res[data_item]
                
                pred_rot_trans_list=torch.zeros((BatchSize,20,480)).cuda()# 480
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
                    pred_rot_trans_temp, new_ssdr_hidden = ssdr_model_list[itt](motion_signature, ssdr_hidden)
                    pred_rot_trans_list[:,itt]=pred_rot_trans_temp
                    #保存隐层
                    hidden_temp.append((ssdr_hidden,new_ssdr_hidden))
                    
                ssdr_hidden_list.append(hidden_temp)
                Field_input=torch.einsum("bjk,lj->blk",pred_rot_trans_list,F.relu(mul_weight_list))
                Field_output=FieldNet(query[data_item],Field_input)
                pred_rot_trans=Field_output.transpose(-1,-2).contiguous()
                pred_pose = pred_rot_trans[:,:, 0:3] * cloth_pose_std + \
                                                                cloth_pose_mean                                                       
                pred_trans = (query[data_item]+pred_rot_trans[:,:, 3:6] )* cloth_trans_std +cloth_trans_mean
                ssdr_res = ssdrlbs.batch_pose(pred_trans.view((1,BatchSize,80 , 3)),                                  
                                                torch.deg2rad(pred_pose).view(
                                                    (1,BatchSize, 80,3)))         
                final_res = ssdr_res                                 
                final_res=final_res.view(sim_res_temp.size())  

                outputs.append(final_res)                           
                targets.append(sim_res_temp)

            
            tar_list=torch.cat(targets,dim=0)
            out_list=torch.cat(outputs,dim=0)

            weight2loss=F.relu(mul_weight_list)
            weight2loss=weight2loss.sum(dim=0)
            loss_vt=Garmentloss(tar_list,out_list)+0.001*torch.mean(torch.abs(weight2loss))#
            
            optimizer.zero_grad()
            loss=loss_vt
            loss.backward(retain_graph=True)
            optimizer.step()
            temp=[]
            for moi in range(20):
                temp.append((ssdr_hidden_list[-1][moi][0].detach(),ssdr_hidden_list[-1][moi][1].detach()))
            ssdr_hidden_list=[]
            ssdr_hidden_list.append(temp)
        train_loss.update(loss.item())
    scheduler.step()
    writer.add_scalar('Loss/Train_loss',train_loss.avg,epoch) 
    if epoch%10==0:
        with torch.no_grad():
            FieldNet.eval()
            for moi in range(20):
                ssdr_model_list[moi].eval()
            ssdr_hidden_list = []
            ssdr_hidden_temp=[]
            for i in range(20):
                ssdr_hidden_temp.append((None,None))
            ssdr_hidden_list.append(ssdr_hidden_temp)
            val_loss_L2.reset()
            num=0
            
            for data,low_res,query_list,trans_label in tqdm(Test_loader):

                pose_arr=data["pose"]
                trans_arr=data["trans"]
                sim_res=low_res   
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
                for i in range(500):
                   
                    hidden_temp=[]
                    pred_rot_trans_list=torch.zeros((test_batch,20,480)).cuda()
                    for itt in range(20):
                        motion_signature = np.zeros((1,test_batch,3*6), dtype=np.float32) 
                        motion_signature = torch.from_numpy(motion_signature)
                        motion_signature=motion_signature.cuda()
                        for j in range(3):
                            motion_signature[:,:,j * 6: j * 6 + 3] = pose_arr[i,:, index_list[itt][j]]   #数据格式(1,batch,52,3)
                            motion_signature[:,:,j * 6+3: j * 6 + 6] = trans_arr[i,:]
                        #构建运动特征
                        motion_signature = motion_signature.view((1*test_batch, -1))
                        ssdr_hidden=ssdr_hidden_list[-1][itt][1]
                        pred_rot_trans_temp, new_ssdr_hidden = ssdr_model_list[itt](motion_signature, ssdr_hidden)
                        pred_rot_trans_list[:,itt]=pred_rot_trans_temp

                        hidden_temp.append((ssdr_hidden,new_ssdr_hidden))
                        
                    ssdr_hidden_list.append(hidden_temp)

                    Field_input=torch.einsum("bjk,lj->blk",pred_rot_trans_list,F.relu(mul_weight_list))
                    Field_output=FieldNet(query[i],Field_input)

                    pred_rot_trans=Field_output.transpose(-1,-2).contiguous()
                    
                    pred_pose = pred_rot_trans[:,:, 0:3] * cloth_pose_std + \
                                cloth_pose_mean
                    pred_trans = (query[i]+pred_rot_trans[:,:, 3:6] ) * cloth_trans_std + \
                                cloth_trans_mean
                    ssdr_res = ssdrlbs.batch_pose(pred_trans.reshape((-1, test_batch,80 ,3)),                                  
                                                    torch.deg2rad(pred_pose).reshape(
                                                        (-1, test_batch, 80, 3)))        
                    final_res=ssdr_res.reshape(sim_res[0,i].size())      
                    loss=torch.mean(torch.sqrt(torch.sum((final_res-sim_res[0,i])**2,dim=1)))
                    val_loss_L2.update(loss.item())
            print("end_loss",val_loss_L2.avg)
            writer.add_scalar('Loss/val_loss',val_loss_L2.avg,epoch)    
            
            torch.save(ssdr_model_list, '%s/ssdr_modellist_ours03.pkl' % (dir_name))
            torch.save(mul_weight_list, '%s/PoseWeightmat_ours03.pkl' % (dir_name))
            torch.save(FieldNet, '%s/FieldNet_ours03.pkl' % (dir_name))
            log_table = {
                "train_loss_L2": train_loss.avg,
                "val_loss_L2": val_loss_L2.avg,
                "epoch": epoch,
            }
            print(log_table)
            with open(logname, 'a') as f:  # open and append
                f.write('json_stats: ' + json.dumps(log_table) + '\n')




#---------保存网络