"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import itertools

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

import numpy as np
from VCMeshConv.GraphAutoEncoder import graphVAESSW as vae_model
from VCMeshConv.GraphAutoEncoder import graphAE_param_iso as Param
from VCMeshConv.GraphAutoEncoder import graphAE_dataloader as Dataloader
from datetime import datetime
from plyfile import PlyData # the ply loader here is using. It is suggested to use faster load function to reduce the io overhead

import random
import json

import time
SCALE = 0.001  

bTrain = False

###############################################################################

class garmentloss(nn.Module):
    def __init__(self,param, facedata,norw=0.01):
        super(garmentloss, self).__init__()

        self.weight_num = 17
        self.motdim = 94

        self.net_loss = vae_model.MCLoss(param)
        
        
        self.register_buffer('t_facedata', facedata.long())

        self.w_pose = param.w_pose 
        self.w_laplace = param.w_laplace #0.5
        self.klweight = 1e-5 #0.00001
        self.w_nor = norw

        

    def forward(self, in_pc_batch, t_nor,ssdr_res): # meshvertices: B N 3, meshnormals: B N 3    

        nbat = in_pc_batch.size(0)
        npt = in_pc_batch.size(1)
        nch = in_pc_batch.size(2)
        
        out_pc_batch=ssdr_res.view(in_pc_batch.size())#out_pc_batchfull[:,:,0:3]#

        loss_pose_l1 = self.net_loss.compute_geometric_loss_l1(in_pc_batch[:,:,0:3], out_pc_batch[:,:,0:3])
        #loss_laplace_l1 = self.net_loss.compute_laplace_loss_l2(in_pc_batch[:,:,0:3], out_pc_batch[:,:,0:3])

        #loss = loss_pose_l1*self.w_pose +  loss_laplace_l1 * self.w_laplace  + klloss * self.klweight + loss_normal * self.w_nor+loss_vt+1000000*vaeloss

        loss = loss_pose_l1*self.w_pose#+ loss_normal * self.w_nor#+  loss_laplace_l1 * self.w_laplace 
        return loss[None]


class Net_autoencpsdhigh(nn.Module):
    def __init__(self,param, facedata,pointnum=12273):
        super(Net_autoencpsdhigh, self).__init__()

        self.weight_num = 17
        self.motdim = 94


        self.net_loss = vae_model.MCLoss(param)
        
        self.pointnum=pointnum
        self.register_buffer('t_facedata', facedata.long())
        self.w_pose = param.w_pose 
        self.w_laplace = param.w_laplace #0.5
        self.klweight = 1e-5 #0.00001
        self.w_nor = 10.0
        self.write_tmp_folder =  param.write_tmp_folder #+"%07d"%iteration+"_%02d_out"%n+suffix+".ply"
        

    def forward(self, in_pc_batch, t_nor,ssdrlbs,mul_weight_list,query,\
                            cloth_pose_std,cloth_pose_mean,cloth_trans_std,cloth_trans_mean,FieldNet,tmtemp,detailsnet,DPSD,ssdr_res_std,ssdr_res_mean): # meshvertices: B N 3, meshnormals: B N 3    

        nbat = in_pc_batch.size(0)
        npt = in_pc_batch.size(1)
        nch = in_pc_batch.size(2)

        
        Field_input=torch.einsum("bjk,lj->blk",tmtemp,F.relu(mul_weight_list))
        Field_output=FieldNet(query,Field_input)
        pred_rot_trans=Field_output.transpose(-1,-2).contiguous()
        pred_pose = pred_rot_trans[:,:, 0:3] * cloth_pose_std + \
                                                        cloth_pose_mean                                                       
        pred_trans = (query+pred_rot_trans[:,:, 3:6] )* cloth_trans_std +cloth_trans_mean
        
        detailkey=detailsnet(tmtemp)
        detail_res=torch.zeros(nbat,self.pointnum,3).cuda()
        for i in range(20):
            detail_temp=torch.einsum("bj,jkl->bkl",detailkey[:,i],DPSD[i]) 
            detail_res+=detail_temp

        detail_res=detail_res * ssdr_res_std + ssdr_res_mean
        ssdr_res = ssdrlbs.batch_pose(pred_trans.view((1,nbat,80 , 3)),                                  
                                        torch.deg2rad(pred_pose).view(
                                            (1,nbat, 80,3)),displacement=detail_res)
        out_pc_batch=ssdr_res.view(in_pc_batch.size())#out_pc_batchfull[:,:,0:3]#
        

        loss_pose_l1 = self.net_loss.compute_geometric_loss_l1(in_pc_batch[:,:,0:3], out_pc_batch[:,:,0:3])
        loss = loss_pose_l1*self.w_pose+0.0001*torch.mean((DPSD)**2)

        return loss[None],out_pc_batch
