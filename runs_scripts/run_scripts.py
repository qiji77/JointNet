import copy
import os
import random
from tqdm import tqdm
import numpy as np
import scipy
from scipy.spatial.transform import Rotation
import json
import sys
sys.path.append('./')
sys.path.append('./src/')
sys.path.append('./Tool_list/')

import torch
from torch.utils.data import DataLoader, Dataset
from torch_geometric.transforms import FaceToEdge

from src.obj_parser import Mesh_obj
from src.SSDRLBS import SSDRLBS
seed=1024
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed) #cpu
torch.cuda.manual_seed_all(seed) #并行gpu
torch.backends.cudnn.deterministic = True #cpu/gpu结果一致
torch.backends.cudnn.benchmark = True
from src.models_spectrum import *
import torch.nn.functional as F

from Tool_list.Body import *

index_list=([3,0,2],[0,1,4],[0,2,5],[0,3,6],[1,4,7],[2,5,8],[3,6,9],[4,7,10],[5,8,11],[6,9,12],[7,10,10],[8,11,11],[9,12,15],[9,13,16],[9,14,17],[12,15,15],[13,16,18],[14,17,19],[16,18,18],[17,19,19])

class AverageMeter():
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_results(config_path, anim_path, out_path, device="cpu"):
    with open(config_path, "r") as f:
        config = json.load(f)

    gru_dim = config["gru_dim"]
    gru_out_dim = config["gru_out_dim"]
    joint_list = config["joint_list"]
    ssdrlbs_bone_num = config["ssdrlbs_bone_num"]

    ssdrlbs_root_dir = config["ssdrlbs_root_dir"]

    ssdrlbs_net_path ="./VirtualBones/NetWork_list/net_path/ssdr_modellist_ours03.pkl"
    ssdrlbs_weight_path="./VirtualBones/NetWork_list/net_path/PoseWeightmat_ours03.pkl"
    fieldpath="./VirtualBones/NetWork_list/net_path/FieldNet_ours03.pkl"
    detail_net_path = "./VirtualBones/NetWork_list/net_path/detailnet_8normal.pkl"#config["detail_net_path"]
    state_path = config["state_path"]
    obj_template_path = config["obj_template_path"]

    garment_template = Mesh_obj(obj_template_path)

    joint_num = len(joint_list)
    ssdr_model_list=torch.load(ssdrlbs_net_path)
    mul_weight_list=torch.load(ssdrlbs_weight_path)
    FieldNet=torch.load(fieldpath)
    #ssdr_model = ssdr_model.to(device)
    #ssdr_model.eval()

    data = FaceToEdge()(Data(num_nodes=garment_template.v.shape[0],
                             face=torch.from_numpy(garment_template.f.astype(int).transpose() - 1).long()))

    detail_model=torch.load(detail_net_path)
    detail_model=detail_model.cuda()
    detail_model.eval()
    DPSD=torch.load("./VirtualBones/NetWork_list/net_path/DPSD_8normal.pkl")
    ssdrlbs = SSDRLBS(os.path.join(ssdrlbs_root_dir, "u.obj"),
                      os.path.join(ssdrlbs_root_dir, "skin_weights.npy"),
                      os.path.join(ssdrlbs_root_dir, "u_trans.npy"),
                      device)

    state = np.load(state_path)

    cloth_pose_mean = torch.from_numpy(state["cloth_pose_mean"]).to(device)
    cloth_pose_std = torch.from_numpy(state["cloth_pose_std"]).to(device)
    cloth_trans_mean = torch.from_numpy(state["cloth_trans_mean"]).to(device)
    cloth_trans_std = torch.from_numpy(state["cloth_trans_std"]).to(device)

    ssdr_res_mean = state["ssdr_res_mean"]
    ssdr_res_mean=torch.from_numpy(ssdr_res_mean).cuda()
    ssdr_res_std = state["ssdr_res_std"]
    ssdr_res_std=torch.from_numpy(ssdr_res_std).cuda()

    vert_std = state["sim_res_std"]
    vert_mean = state["sim_res_mean"]

    anim = np.load(anim_path)
    pose_arr = (anim["pose"] - state["pose_mean"]) / state["pose_std"]
    trans_arr = (anim["trans"] - anim["trans"][0] - state["trans_mean"]) / state["trans_std"]
    item_length = pose_arr.shape[0]

    ssdr_hidden_list = []
    ssdr_hidden_temp=[]
    for i in range(20):
        ssdr_hidden_list.append((None,None))
    #ssdr_hidden_list.append(ssdr_hidden_temp)
    detail_hidden=None

    cloth_tem=torch.from_numpy(garment_template.v).float().cuda()
    query2=np.load("./VirtualBones/assets/dress03/SSDR/u_trans.npy")
    query2=np.array(query2)
    query2=query2.astype('float32')
    query2=torch.from_numpy(query2)
    query2=query2.unsqueeze(0)
    query2=query2.cuda()
    with torch.no_grad():
        for frame_it in tqdm(range(item_length)):
            frame=frame_it
            hidden_temp=[]
            pred_rot_trans_list=torch.zeros((1,20,480)).cuda()
        
            
            
            for itt in range(20):
                motion_signature = np.zeros((1,1,3*6), dtype=np.float32) 
                for j in range(3):
                    motion_signature[:,:,j * 6: j * 6 + 3] = pose_arr[frame, index_list[itt][j]]   #数据格式(1,batch,52,3)
                    motion_signature[:,:,j * 6+3: j * 6 + 6] = trans_arr[frame]
                #构建运动特征
                motion_signature = torch.from_numpy(motion_signature)
                motion_signature = motion_signature.view((1, -1)).to(device)
                #输入到网络
                ssdr_hidden=ssdr_hidden_list[itt][1]
                pred_rot_trans_temp, new_ssdr_hidden = ssdr_model_list[itt](motion_signature, ssdr_hidden)
                #pred_rot_trans=pred_rot_trans.unsqueeze(1)
                pred_rot_trans_list[:,itt]=pred_rot_trans_temp
                ssdr_hidden_list[itt]=(ssdr_hidden,new_ssdr_hidden)
                

            querytemp=query2.detach()
            Field_input=torch.einsum("bjk,lj->blk",pred_rot_trans_list,F.relu(mul_weight_list))
            Field_output=FieldNet(querytemp,Field_input)
           
            pred_rot_trans=Field_output.transpose(-2,-1).contiguous()
         
            pred_pose = pred_rot_trans[:,:, 0:3] * cloth_pose_std + \
                        cloth_pose_mean
            pred_trans = (querytemp+pred_rot_trans[:,:, 3:6] ) * cloth_trans_std + \
                            cloth_trans_mean    


            cloth_tem_input=cloth_tem.unsqueeze(0)
            cloth_tem_input=cloth_tem_input.unsqueeze(0)
            detailkey=detail_model(pred_rot_trans_list)
            detail_res=torch.zeros(1,8744,3).cuda()
            for i in range(20):
                detail_temp=torch.einsum("bj,jkl->bkl",detailkey[:,i],DPSD[i]) 
                detail_res+=detail_temp

            detail_res=detail_res * ssdr_res_std + ssdr_res_mean
            ssdr_res = ssdrlbs.batch_pose(pred_trans.reshape((-1, 1,80 , 3)),                                  
                                            torch.deg2rad(pred_pose).reshape(
                                                (-1, 1, 80, 3)))         #pred_trans.shape[0]=80 在这做lbs变形
            final_res = (ssdr_res).detach().cpu().numpy().reshape((-1, 3))

            pose = pose_arr[frame] * state["pose_std"] + state["pose_mean"]
            trans = trans_arr[frame] * state["trans_std"] + state["trans_mean"]

            trans_off = np.array([0,
                                  -2.1519510746002397,
                                  90.4766845703125]) / 100.0
            trans += trans_off

            final_res = np.matmul(Rotation.from_rotvec(pose[0]).as_matrix(),
                                  final_res.transpose()).transpose()
            final_res += trans
            
            
            out_obj = copy.deepcopy(garment_template)
            out_obj.v = final_res
            out_obj.write(os.path.join(out_path, "{}.obj".format(frame_it)))


if __name__ == "__main__":
    config_path = "assets/dress03/config.json"
    test_list=['86.npz', '46.npz', '58.npz', '81.npz', '26.npz']#['85.npz','87.npz','88.npz','9.npz']#
    ith=1
    for testid in test_list:
        anim_path ="./VirtualBones/VirtualBonesDataset/dress03/"+testid #"anim/anim"+str(ith)+".npz"
        out_path = "results_list/pretrain/"+testid[:-4]#str(ith)
        device = "cuda:0"
        if not os.path.exists(out_path):
            os.makedirs(out_path) 
        get_results(config_path, anim_path, out_path, device)
        ith+=1