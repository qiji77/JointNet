import copy
import math
import torch
torch.pi=torch.acos(torch.zeros(1)).item()*2
print("pi is",torch.pi)
import torch.nn as nn
from torch.nn import Embedding
import numpy as np
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.nn import EdgeConv, global_max_pool
from torch_geometric.utils import dropout_adj, get_laplacian

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

class Scheduler(_LRScheduler):
    def __init__(self, 
                 optimizer: Optimizer,
                 dim_embed: int,
                 warmup_steps: int,
                 last_epoch: int=-1,
                 verbose: bool=False,
                 initlr:float=0.00004) -> None:

        self.dim_embed = dim_embed
        self.warmup_steps = warmup_steps
        self.num_param_groups = len(optimizer.param_groups)
        self.initlr=initlr


        super().__init__(optimizer, last_epoch, verbose)
        
    def get_lr(self) -> float:
        lr = calc_linear_lr(self._step_count, self.warmup_steps,self.initlr)
        return [lr] * self.num_param_groups


def calc_linear_lr(step,warmup_steps,initlr=0.00004,endstep=20000):    #直接迁移时用的是0.0005
    
    if step<=warmup_steps:
        hymter=step/warmup_steps
        return hymter*initlr
    else:
        hymter= step/endstep
        return initlr-hymter*initlr

def calc_lr(step, dim_embed, warmup_steps):
    return 0.005*dim_embed**(-0.5) * min(step**(-0.5), step * warmup_steps**(-1.5))

def MLP(dimensions, dropout=False, batch_norm=False, batch_norm_momentum=1e-3):
    return nn.Sequential(*[
        nn.Sequential(
            nn.Dropout(p=0.5) if dropout else nn.Identity(),
            nn.Linear(dimensions[i - 1], dimensions[i]),
            nn.PReLU(dimensions[i]),
            nn.BatchNorm1d(dimensions[i], affine=True, momentum=batch_norm_momentum) if batch_norm else nn.Identity())
        for i in range(1, len(dimensions))])


class MLPModel(nn.Module):
    def __init__(self, dim):
        super(MLPModel, self).__init__()
        self.dim = dim
        self.layers = nn.Sequential(MLP(self.dim[0:-1], batch_norm=True),
                                    nn.Linear(self.dim[-2], self.dim[-1]))

    def forward(self, x):
        return self.layers(x)


class GRU_Model(nn.Module):
    def __init__(self, input_num, hidden_num, output_num, shortcut=False):
        super(GRU_Model, self).__init__()
        self.hidden_num = hidden_num
        #self.cell = nn.GRUCell(RFFdim*2+input_num, hidden_num)
        self.cell = nn.GRUCell(input_num, hidden_num)
        
        self.output_num = copy.deepcopy(output_num)
        self.output_num.insert(0, hidden_num)
        
        #self.inlinear=nn.Sequential(nn.Linear(18, 600))
        self.out_linear = nn.Sequential(MLP(self.output_num[0:-1], batch_norm=True, dropout=True),
                                        nn.Dropout(p=0.5),
                                        nn.Linear(self.output_num[-2], self.output_num[-1]))
        self.shortcut = shortcut
        

    def forward(self, x, hidden=None):
        
        if hidden is None:
            hidden = torch.zeros(x.shape[0], self.hidden_num).to(self.cell.bias_hh.device)
            # hidden = torch.randn(x.shape[0], self.hidden_num).to(self.cell.bias_hh.device)
        next_hidden = self.cell(x, hidden)
        y = self.out_linear(next_hidden)
        if self.shortcut:
            y = y + x[:, :-3]
        return y, next_hidden

class VirtualposeMlP(nn.Module):
    def __init__(self,input_num,hiddnum,outputnum):
        super(VirtualposeMlP, self).__init__()
        self.resizenet1=nn.Linear(input_num,hiddnum)
        self.resizenet2=nn.Linear(hiddnum,hiddnum)
        self.resizenet3=nn.Linear(hiddnum,outputnum)
        #self.RFFembed=RandomFourierFeatEnc(input_num//2,1,in_dim=input_num,include_input=True)# std=1

    def forward(self,x):
        #Bn,K,dim=x.size()
        #x=self.RFFembed(x)
        #x=x.reshape(Bn,-1)
        x=self.resizenet1(x)
        x=self.resizenet2(x)
        #x=self.RFFembed(x)
        x=self.resizenet3(x)
        
        
        #x=x.reshape(Bn,K,dim)
        return x

class GRU_Model_vae(nn.Module):
    def __init__(self, input_num, hidden_num, output_num, shortcut=False):
        super(GRU_Model_vae, self).__init__()
        self.hidden_num = hidden_num
        #self.cell = nn.GRUCell(RFFdim*2+input_num, hidden_num)
        self.cell = nn.GRUCell(input_num, hidden_num)
        
        self.output_num = copy.deepcopy(output_num)
        self.output_num.insert(0, hidden_num)
       
        self.out_linear = nn.Sequential(MLP(self.output_num[0:-1], batch_norm=True, dropout=True),
                                        nn.Dropout(p=0.5),
                                        nn.Linear(self.output_num[-2], self.output_num[-1]))
        self.shortcut = shortcut
        #self.VaeEncoder=VaeEncoder(hidden_num,1024,output_num[-1])
        

    def forward(self, x, hidden=None):
        
        if hidden is None:
            hidden = torch.zeros(x.shape[0], self.hidden_num).to(self.cell.bias_hh.device)
            # hidden = torch.randn(x.shape[0], self.hidden_num).to(self.cell.bias_hh.device)
        next_hidden = self.cell(x, hidden)
        t_mu=self.out_linear(next_hidden)
        '''t_mu,t_logstd = self.VaeEncoder(next_hidden)
        
        klloss = torch.mean(-0.5 - t_logstd + 0.5 * t_mu ** 2 + 0.5 * torch.exp(2 * t_logstd))
        t_std = t_logstd.exp()
        t_eps = torch.ones_like(t_std).normal_() #torch.FloatTensor(t_std.size()).normal_().to(device)
        t_z = t_mu + t_std * t_eps '''
        
        return t_mu, next_hidden#,klloss

        

class GRU_GNN_Model(nn.Module):
    # the implementation of pytorch geometric heavily relies on scatter_sum which can not be deterministic
    # so the GNN using this package can not achieve deterministic
    def __init__(self, input_num, hidden_num, output_num, gru_out_dim, gnn_dim, edge_index, batch_norm=True):
        super(GRU_GNN_Model, self).__init__()
        self.hidden_num = hidden_num
        self.cell = nn.GRUCell(input_num, hidden_num)
        self.output_num = copy.deepcopy(output_num)
        self.output_num.insert(0, hidden_num)
        self.gnn_dim = gnn_dim

        self.gru_out_linear = nn.Sequential(MLP(self.output_num, batch_norm=batch_norm, dropout=True))
        self.gru_out_dim = gru_out_dim
        self.out_linear = nn.Sequential(nn.Linear(self.gnn_dim[-1] + self.gru_out_dim, 3))
        self.edge_index = edge_index
        self.edge_conv_layers = nn.ModuleList()

        for i in range(len(self.gnn_dim) - 1):
            self.edge_conv_layers.append(EdgeConv(MLP([2 * self.gnn_dim[i], self.gnn_dim[i + 1]])))

    def dropout_edge(self, input_edge_index, p=0.8, force_undirected=True):
        if self.training:
            edge_index, _ = dropout_adj(input_edge_index, p=p, force_undirected=force_undirected)
        else:
            edge_index = input_edge_index
        return edge_index

    def forward(self, x, smoothed_vert_pos, hidden=None):
        batch_size = x.shape[0]
        smoothed_vert_pos = smoothed_vert_pos.view((batch_size, -1, 3))
        smoothed_x0 = torch.zeros((batch_size, smoothed_vert_pos.shape[1], self.gnn_dim[1])).to(x.device)
        smoothed_x1 = torch.zeros((batch_size, smoothed_vert_pos.shape[1], self.gnn_dim[2])).to(x.device)

        for i in range(batch_size):
            edge_index = self.dropout_edge(self.edge_index)
            smoothed_x0[i] = self.edge_conv_layers[0](smoothed_vert_pos[i], edge_index)

        if not self.training:
            smoothed_x0 = smoothed_x0.detach()

        for i in range(batch_size):
            edge_index = self.dropout_edge(self.edge_index)
            smoothed_x1[i] = self.edge_conv_layers[1](smoothed_x0[i], edge_index)

        if not self.training:
            smoothed_x1 = smoothed_x1.detach()

        if hidden is None:
            hidden = torch.zeros(x.shape[0], self.hidden_num).to(self.cell.bias_hh.device)
        next_hidden = self.cell(x, hidden)
        gru_out = self.gru_out_linear(next_hidden).view((batch_size, -1, self.gru_out_dim))
        y = self.out_linear(torch.cat([gru_out, smoothed_x1], axis=2)).view((batch_size, -1))
        return y, next_hidden

class GRU_GNN_Model_nogru(nn.Module):
    # the implementation of pytorch geometric heavily relies on scatter_sum which can not be deterministic
    # so the GNN using this package can not achieve deterministic
    def __init__(self, input_num, hidden_num, output_num, gru_out_dim, gnn_dim, edge_index, batch_norm=True):
        super(GRU_GNN_Model_nogru, self).__init__()
        self.hidden_num = hidden_num
        self.output_num = copy.deepcopy(output_num)
        self.output_num.insert(0, hidden_num)
        self.gnn_dim = gnn_dim

        self.gru_out_linear = nn.Sequential(MLP(self.output_num, batch_norm=False, dropout=True))
        self.gru_out_dim = gru_out_dim
        self.out_linear = nn.Sequential(nn.Linear(self.gnn_dim[-1] + self.gru_out_dim, 3))#Vae()#
        self.edge_index = edge_index
        self.edge_conv_layers = nn.ModuleList()

        for i in range(len(self.gnn_dim) - 1):
            self.edge_conv_layers.append(EdgeConv(MLP([2 * self.gnn_dim[i], self.gnn_dim[i + 1]])))
        self.RFFembed=RandomFourierFeatEnc(240,1,in_dim=480,include_input=True)# std=1
    def dropout_edge(self, input_edge_index, p=0.8, force_undirected=True):
        if self.training:
            edge_index, _ = dropout_adj(input_edge_index, p=p, force_undirected=force_undirected)
        else:
            edge_index = input_edge_index
        return edge_index

    def forward(self, x, smoothed_vert_pos, hidden=None):
        batch_size = x.shape[0]
        smoothed_vert_pos = smoothed_vert_pos.view((batch_size, -1, 3))
        smoothed_x0 = torch.zeros((batch_size, smoothed_vert_pos.shape[1], self.gnn_dim[1])).to(x.device)
        smoothed_x1 = torch.zeros((batch_size, smoothed_vert_pos.shape[1], self.gnn_dim[2])).to(x.device)
        x_input=self.RFFembed(x)
        for i in range(batch_size):
            edge_index = self.dropout_edge(self.edge_index)
            smoothed_x0[i] = self.edge_conv_layers[0](smoothed_vert_pos[i], edge_index)

        if not self.training:
            smoothed_x0 = smoothed_x0.detach()

        for i in range(batch_size):
            edge_index = self.dropout_edge(self.edge_index)
            smoothed_x1[i] = self.edge_conv_layers[1](smoothed_x0[i], edge_index)

        if not self.training:
            smoothed_x1 = smoothed_x1.detach()
        gru_out = self.gru_out_linear(x_input).view((batch_size, -1, self.gru_out_dim))
        y = self.out_linear(torch.cat([gru_out, smoothed_x1], axis=2)).view((batch_size, -1))
        next_hidden=0
        return y, next_hidden

class GRU_GNN_Model_REFbf(nn.Module):
    # the implementation of pytorch geometric heavily relies on scatter_sum which can not be deterministic
    # so the GNN using this package can not achieve deterministic
    def __init__(self, input_num, hidden_num, output_num, gru_out_dim, gnn_dim, edge_index, batch_norm=True):
        super(GRU_GNN_Model_REFbf, self).__init__()
        self.hidden_num = hidden_num
        self.output_num = copy.deepcopy(output_num)
        self.output_num.insert(0, hidden_num)
        self.gnn_dim = gnn_dim
        self.gru_out_linear = nn.Sequential(MLP(self.output_num, batch_norm=False, dropout=True))
        self.gru_out_dim = gru_out_dim
        self.out_linear = nn.Sequential(nn.Linear(self.gnn_dim[-1] , 3))#Vae()#
        self.edge_index = edge_index
        self.edge_conv_layers = nn.ModuleList()
   
        self.edge_conv_layers.append(EdgeConv(MLP([2 * self.gnn_dim[0]+2*self.gru_out_dim, self.gnn_dim[1]])))
        self.edge_conv_layers.append(EdgeConv(MLP([2 * self.gnn_dim[1], self.gnn_dim[2]])))

        self.RFFembed=RandomFourierFeatEnc(2160,1,in_dim=480,include_input=True)# std=1

    def dropout_edge(self, input_edge_index, p=0.8, force_undirected=True):
        if self.training:
            edge_index, _ = dropout_adj(input_edge_index, p=p, force_undirected=force_undirected)
        else:
            edge_index = input_edge_index
        return edge_index

    def forward(self, x, smoothed_vert_pos, hidden=None):
        batch_size = x.shape[0]
        smoothed_vert_pos = smoothed_vert_pos.view((batch_size, -1, 3))
        smoothed_x0 = torch.zeros((batch_size, smoothed_vert_pos.shape[1], self.gnn_dim[1])).to(x.device)
        smoothed_x1 = torch.zeros((batch_size, smoothed_vert_pos.shape[1], self.gnn_dim[2])).to(x.device)
        x_input=self.RFFembed(x)
        gru_out = self.gru_out_linear(x_input).view((batch_size, -1, self.gru_out_dim))
        smoothed_vert_pos=torch.cat([gru_out, smoothed_vert_pos], axis=2)
        for i in range(batch_size):
            edge_index = self.dropout_edge(self.edge_index)
            smoothed_x0[i] = self.edge_conv_layers[0](smoothed_vert_pos[i], edge_index)

        if not self.training:
            smoothed_x0 = smoothed_x0.detach()
        #smoothed_x0=torch.cat([gru_out, smoothed_x0], axis=2)
        for i in range(batch_size):
            edge_index = self.dropout_edge(self.edge_index)
            smoothed_x1[i] = self.edge_conv_layers[1](smoothed_x0[i], edge_index)

        if not self.training:
            smoothed_x1 = smoothed_x1.detach()
        y = self.out_linear(smoothed_x1).view((batch_size, -1))
        next_hidden=0
        return y, next_hidden

class PointNetfeat(nn.Module):
    def __init__(self, output_dim=1024, global_feat = True):
        super(PointNetfeat, self).__init__()
        self.output_dim=output_dim
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, self.output_dim, 1)

        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(self.output_dim)

    def forward(self, x):
        batchsize = x.size()[0]
        x = nn.functional.relu(self.bn1(self.conv1(x)))
        pointfeat = x
        x = nn.functional.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x,_ = torch.max(x, 2)
        x = x.view(-1, self.output_dim)
        return x

class GRU_GNN_Model_targettest(nn.Module):
    # the implementation of pytorch geometric heavily relies on scatter_sum which can not be deterministic
    # so the GNN using this package can not achieve deterministic
    def __init__(self, input_num, hidden_num, output_num, gru_out_dim, gnn_dim, edge_index, batch_norm=True):
        super(GRU_GNN_Model_targettest, self).__init__()
        self.hidden_num = hidden_num
        #self.cell = nn.GRUCell(input_num, hidden_num)
        self.output_num = copy.deepcopy(output_num)
        self.output_num.insert(0, hidden_num)
        self.gnn_dim = gnn_dim
        self.gru_out_linear = nn.Sequential(MLP(self.output_num, batch_norm=False, dropout=True))
        self.gru_out_dim = gru_out_dim
        self.out_linear = nn.Sequential(nn.Linear(self.gnn_dim[-1], 3))#Vae()#
        self.edge_index = edge_index
        self.edge_conv_layers = nn.ModuleList()
   
        #for i in range(len(self.gnn_dim) - 1):
        self.edge_conv_layers.append(EdgeConv(MLP([2 * self.gnn_dim[0]+2048, self.gnn_dim[1]])))
        self.edge_conv_layers.append(EdgeConv(MLP([2 * self.gnn_dim[1], self.gnn_dim[2]])))
        self.edge_conv_layers.append(EdgeConv(MLP([2 * self.gnn_dim[2], self.gnn_dim[3]])))

        self.RFFembed=RandomFourierFeatEnc(240,1,in_dim=480,include_input=True)# std=1

        
        self.z_mean=nn.Linear(1024,1024)
        self.z_log_var=nn.Linear(1024,1024)
        self.sample=Sample()
        

    def dropout_edge(self, input_edge_index, p=0.8, force_undirected=True):
        if self.training:
            edge_index, _ = dropout_adj(input_edge_index, p=p, force_undirected=force_undirected)
        else:
            edge_index = input_edge_index
        return edge_index

    def forward(self, x, smoothed_vert_pos, hidden=None):
        batch_size = x.shape[0]
        #xinput=self.inputfea(x.transpose(2,1))
        z_mean=self.z_mean(x)
        z_log_var=self.z_log_var(x)
        x=self.sample(z_mean,z_log_var)
        smoothed_vert_pos = smoothed_vert_pos.view((batch_size, -1, 3))

        xinput=x.unsqueeze(1).expand(batch_size,smoothed_vert_pos.size(1),-1)
        smoothed_vert_pos=torch.cat([xinput,smoothed_vert_pos],axis=2)
        smoothed_x0 = torch.zeros((batch_size, smoothed_vert_pos.shape[1], self.gnn_dim[1])).to(x.device)
        smoothed_x1 = torch.zeros((batch_size, smoothed_vert_pos.shape[1], self.gnn_dim[2])).to(x.device)
        smoothed_x2 = torch.zeros((batch_size, smoothed_vert_pos.shape[1], self.gnn_dim[3])).to(x.device)
        for i in range(batch_size):
            edge_index = self.dropout_edge(self.edge_index)
            smoothed_x0[i] = self.edge_conv_layers[0](smoothed_vert_pos[i], edge_index)

        if not self.training:
            smoothed_x0 = smoothed_x0.detach()

        for i in range(batch_size):
            edge_index = self.dropout_edge(self.edge_index)
            smoothed_x1[i] = self.edge_conv_layers[1](smoothed_x0[i], edge_index)

        if not self.training:
            smoothed_x1 = smoothed_x1.detach()
        
        for i in range(batch_size):
            edge_index = self.dropout_edge(self.edge_index)
            smoothed_x2[i] = self.edge_conv_layers[2](smoothed_x1[i], edge_index)

        if not self.training:
            smoothed_x2 = smoothed_x2.detach()

        

        y = self.out_linear(smoothed_x2)
        next_hidden=0
        return y, next_hidden

class GRU_GNN_Model_REF_atten(nn.Module):
    # the implementation of pytorch geometric heavily relies on scatter_sum which can not be deterministic
    # so the GNN using this package can not achieve deterministic
    def __init__(self, input_num, hidden_num, output_num, gru_out_dim, gnn_dim, edge_index, batch_norm=True):
        super(GRU_GNN_Model_REF_atten, self).__init__()
        self.hidden_num = hidden_num
        #self.cell = nn.GRUCell(input_num, hidden_num)
        self.output_num = copy.deepcopy(output_num)
        self.output_num.insert(0, hidden_num)
        self.gnn_dim = gnn_dim
        self.gru_out_linear = nn.Sequential(MLP(self.output_num, batch_norm=False, dropout=True))
        self.gru_out_dim = gru_out_dim
        self.out_linear = nn.Sequential(nn.Linear(self.gnn_dim[-1] + self.gru_out_dim, 3))#Vae()#
        self.edge_index = edge_index
        self.edge_conv_layers = nn.ModuleList()
   
        for i in range(len(self.gnn_dim) - 1):
            self.edge_conv_layers.append(EdgeConv(MLP([2 * self.gnn_dim[i], self.gnn_dim[i + 1]])))

        self.RFFembed=RandomFourierFeatEnc(240,1,in_dim=480,include_input=True)# std=1
        
        queryvec=torch.randn((1,hidden_num))
        self.queryvec=torch.nn.Parameter(queryvec.data,requires_grad=True)

        self.mlpq=nn.utils.weight_norm(nn.Conv1d(hidden_num, hidden_num, 1))
        self.mlpk=nn.utils.weight_norm(nn.Conv1d(hidden_num, hidden_num, 1))
        self.mlpv=nn.utils.weight_norm(nn.Conv1d(hidden_num, hidden_num, 1))

    def dropout_edge(self, input_edge_index, p=0.8, force_undirected=True):
        if self.training:
            edge_index, _ = dropout_adj(input_edge_index, p=p, force_undirected=force_undirected)
        else:
            edge_index = input_edge_index
        return edge_index

    def forward(self, x, smoothed_vert_pos, hidden=None):
        batch_size = x.shape[0]
        smoothed_vert_pos = smoothed_vert_pos.view((batch_size, -1, 3))
        smoothed_x0 = torch.zeros((batch_size, smoothed_vert_pos.shape[1], self.gnn_dim[1])).to(x.device)
        smoothed_x1 = torch.zeros((batch_size, smoothed_vert_pos.shape[1], self.gnn_dim[2])).to(x.device)
        x_input=self.RFFembed(x)
        for i in range(batch_size):
            edge_index = self.dropout_edge(self.edge_index)
            smoothed_x0[i] = self.edge_conv_layers[0](smoothed_vert_pos[i], edge_index)

        if not self.training:
            smoothed_x0 = smoothed_x0.detach()

        for i in range(batch_size):
            edge_index = self.dropout_edge(self.edge_index)
            smoothed_x1[i] = self.edge_conv_layers[1](smoothed_x0[i], edge_index)

        if not self.training:
            smoothed_x1 = smoothed_x1.detach()

        '''if hidden is None:
            hidden = torch.zeros(x.shape[0], self.hidden_num).to(self.cell.bias_hh.device)'''
        #next_hidden = self.cell(x, hidden)
        query=self.queryvec.unsqueeze(0).expand(batch_size,self.queryvec.size(0),self.queryvec.size(1))
        query=self.mlpq(query.transpose(-2,-1))
        query=query.transpose(-2,-1)
        key=self.mlpk(x_input.transpose(-2,-1))#B,20,D->B,d,20
        key=key.transpose(-2,-1)
        value=self.mlpv(x_input.transpose(-2,-1))
        value=value.transpose(-2,-1)
        linearinput,_=attention(query,key,value)
        linearinput=linearinput.view(batch_size,-1)
        gru_out = self.gru_out_linear(linearinput).view((batch_size, -1, self.gru_out_dim))
        y = self.out_linear(torch.cat([gru_out, smoothed_x1], axis=2)).view((batch_size, -1))
        next_hidden=0
        return y, next_hidden

class GRU_GNN_Model_REF(nn.Module):
    # the implementation of pytorch geometric heavily relies on scatter_sum which can not be deterministic
    # so the GNN using this package can not achieve deterministic
    def __init__(self, input_num, hidden_num, output_num, gru_out_dim, gnn_dim, edge_index, batch_norm=True):
        super(GRU_GNN_Model_REF, self).__init__()
        self.hidden_num = hidden_num
        #self.cell = nn.GRUCell(input_num, hidden_num)
        self.output_num = copy.deepcopy(output_num)
        self.output_num.insert(0, hidden_num)
        self.gnn_dim = gnn_dim
        self.gru_out_linear = nn.Sequential(MLP(self.output_num, batch_norm=False, dropout=True))
        self.gru_out_dim = gru_out_dim
        self.out_linear = nn.Sequential(nn.Linear(self.gnn_dim[-1] + self.gru_out_dim, 3))#Vae()#
        self.edge_index = edge_index
        self.edge_conv_layers = nn.ModuleList()
   
        for i in range(len(self.gnn_dim) - 1):
            self.edge_conv_layers.append(EdgeConv(MLP([2 * self.gnn_dim[i], self.gnn_dim[i + 1]])))

        self.RFFembed=RandomFourierFeatEnc(240,1,in_dim=480,include_input=True)# std=1

    def dropout_edge(self, input_edge_index, p=0.8, force_undirected=True):
        if self.training:
            edge_index, _ = dropout_adj(input_edge_index, p=p, force_undirected=force_undirected)
        else:
            edge_index = input_edge_index
        return edge_index

    def forward(self, x, smoothed_vert_pos, hidden=None):
        batch_size = x.shape[0]
        smoothed_vert_pos = smoothed_vert_pos.view((batch_size, -1, 3))
        smoothed_x0 = torch.zeros((batch_size, smoothed_vert_pos.shape[1], self.gnn_dim[1])).to(x.device)
        smoothed_x1 = torch.zeros((batch_size, smoothed_vert_pos.shape[1], self.gnn_dim[2])).to(x.device)
        x_input=self.RFFembed(x)
        for i in range(batch_size):
            edge_index = self.dropout_edge(self.edge_index)
            smoothed_x0[i] = self.edge_conv_layers[0](smoothed_vert_pos[i], edge_index)

        if not self.training:
            smoothed_x0 = smoothed_x0.detach()

        for i in range(batch_size):
            edge_index = self.dropout_edge(self.edge_index)
            smoothed_x1[i] = self.edge_conv_layers[1](smoothed_x0[i], edge_index)

        if not self.training:
            smoothed_x1 = smoothed_x1.detach()

        '''if hidden is None:
            hidden = torch.zeros(x.shape[0], self.hidden_num).to(self.cell.bias_hh.device)'''
        #next_hidden = self.cell(x, hidden)
        gru_out = self.gru_out_linear(x_input).view((batch_size, -1, self.gru_out_dim))
        y = self.out_linear(torch.cat([gru_out, smoothed_x1], axis=2)).view((batch_size, -1))
        next_hidden=0
        return y, next_hidden

class GRU_Field_Model(nn.Module):
    # the implementation of pytorch geometric heavily relies on scatter_sum which can not be deterministic
    # so the GNN using this package can not achieve deterministic
    def __init__(self, input_num, hidden_num, output_num, batch_norm=True):
        super(GRU_Field_Model, self).__init__()
        self.hidden_num = hidden_num
        self.cell = nn.GRUCell(input_num, hidden_num)
        #self.output_num = copy.deepcopy(output_num)
        #self.output_num.insert(0, hidden_num)
        self.field=VposeFiled_high(hidden_num,hidden_num,output_num,300)


    def forward(self, x, smoothed_vert_pos, hidden=None):
        batch_size = x.shape[0]
        smoothed_vert_pos = smoothed_vert_pos.view((batch_size, -1, 3))
        if hidden is None:
            hidden = torch.zeros(x.shape[0], self.hidden_num).to(self.cell.bias_hh.device)
        next_hidden = self.cell(x, hidden)

        y = self.field(smoothed_vert_pos,next_hidden.unsqueeze(1))
        return y, next_hidden
    
    def iterforward(self,smoothed_vert_pos, hidden=None):
        batch_size = smoothed_vert_pos.shape[0]
        smoothed_vert_pos = smoothed_vert_pos.view((batch_size, -1, 3))
        y = self.field(smoothed_vert_pos,hidden.unsqueeze(1))
        return y

class FourierFeatEnc(nn.Module):
    """
    Inspired by
    https://github.com/facebookresearch/pytorch3d/blob/fc4dd80208bbcf6f834e7e1594db0e106961fd60/pytorch3d/renderer/implicit/harmonic_embedding.py#L10
    """
    def __init__(self, k, include_input=True, use_logspace=False, max_freq=None):
        super(FourierFeatEnc, self).__init__()
        if use_logspace:
            freq_bands = 2 ** torch.arange(0, k) * torch.pi
        else:
            assert max_freq is not None
            freq_bands = 2 ** torch.linspace(0, max_freq, steps=k+1)[:-1] * torch.pi
        self.register_buffer("freq_bands", freq_bands, persistent=False)
        self.include_input = include_input

    def forward(self, x):
        embed = (x[..., None] * self.freq_bands).view(*x.size()[:-1], -1)
        if self.include_input:
            return torch.cat((embed.cos(), embed.sin(), x), dim=-1)
        return torch.cat((embed.cos(), embed.sin()), dim=-1)


class RandomFourierFeatEnc(nn.Module):
    def __init__(self, k, std=1., in_dim=3, dtype=torch.float32, include_input=False):
        super(RandomFourierFeatEnc, self).__init__()
        B = torch.randn((in_dim, k), dtype=dtype) * std
        self.register_buffer("B", B, persistent=True)
        self.include_input = include_input

    def forward(self, x):
        embed = (2 * torch.pi * x) @ self.B
        if self.include_input:
            return torch.cat((embed.cos(), embed.sin(), x), dim=-1)
        return torch.cat((embed.cos(), embed.sin()), dim=-1)
    
    
class Sine(nn.Module):
    def __init__(self):
        super(Sine, self).__init__()
        
    def forward(self, x):
        return torch.sin(x)


class LinearWithConcatAndActivation(nn.Module):
    def __init__(self, x_in_dim, y_in_dim, out_dim, batchnorm=False, activation=nn.ReLU):
        super(LinearWithConcatAndActivation, self).__init__()
        self.Lx = nn.Linear(x_in_dim, out_dim)
        self.Ly = nn.Linear(y_in_dim, out_dim)
        self.actn = activation()
        self.batchnorm = None
        if batchnorm:
            self.batchnorm = nn.BatchNorm1d(out_dim)

    def forward(self, x, y):
        out = self.actn(self.Lx(x) + self.Ly(y))
        return out if self.batchnorm is None else self.batchnorm(out)

class Onemlp(nn.Module):
    def __init__(self, feature_size,output_dim,vjointnum=80):
        super(Onemlp, self).__init__()
        self.vjnum=vjointnum
        self.fc_out = nn.utils.weight_norm(nn.Conv1d(feature_size, 6, 1))
        self.actvn = nn.LeakyReLU()
    
    def forward(self,point_features):
        
        point_features=point_features.transpose(-2,-1)
        point_features = self.fc_out(point_features)
        
        return point_features

class VposeFiled_Vjmlp(nn.Module):
    def __init__(self, feature_size,hidden_dim,output_dim,vjointnum=80):
        super(VposeFiled_Vjmlp, self).__init__()
        self.vjnum=vjointnum
        self.fc_0 = nn.utils.weight_norm(nn.Conv1d(feature_size*self.vjnum, hidden_dim*2*self.vjnum, 1,groups=self.vjnum))
        self.fc_1 = nn.utils.weight_norm(nn.Conv1d(hidden_dim*2*self.vjnum, hidden_dim*2*self.vjnum, 1,groups=self.vjnum))
        self.fc_2 = nn.utils.weight_norm(nn.Conv1d(hidden_dim*2*self.vjnum, hidden_dim*self.vjnum, 1,groups=self.vjnum))
        self.fc_out = nn.utils.weight_norm(nn.Conv1d(hidden_dim*self.vjnum, self.vjnum*6, 1,groups=self.vjnum))
        self.actvn = nn.LeakyReLU()
        self.RFFembed=RandomFourierFeatEnc(240,1,in_dim=3)# std=1
    
    def forward(self,querypoint,point_features): 
        queryfea=self.RFFembed(querypoint)
        B,N,C=queryfea.size()
        point_features=point_features+queryfea
        point_features=point_features.transpose(-2,-1)
        point_features=point_features.reshape(point_features.size(0),-1,1)
        point_features = self.actvn(self.fc_0(point_features))
        point_features = self.actvn(self.fc_1(point_features))
        point_features = self.actvn(self.fc_2(point_features))
        point_features = self.fc_out(point_features)
        point_features=point_features.reshape(B,N,6)#point_features.reshape(B,-1,N).transpose(-2,-1)
        return point_features

class VposeFiled_Vjmlp_weight(nn.Module):
    def __init__(self, feature_size,hidden_dim,output_dim,vjointnum=80):
        super(VposeFiled_Vjmlp_weight, self).__init__()
        self.vjnum=vjointnum
        self.fc_0 = nn.utils.weight_norm(nn.Conv1d(feature_size*self.vjnum, hidden_dim*2*self.vjnum, 1,groups=self.vjnum))
        #self.fc_1 = nn.utils.weight_norm(nn.Conv1d(hidden_dim*2*self.vjnum, hidden_dim*2*self.vjnum, 1,groups=self.vjnum))
        #self.fc_2 = nn.utils.weight_norm(nn.Conv1d(hidden_dim*2*self.vjnum, hidden_dim*self.vjnum, 1,groups=self.vjnum))
        self.fc_1 = nn.utils.weight_norm(nn.Conv1d(hidden_dim*2*self.vjnum, hidden_dim*self.vjnum, 1,groups=self.vjnum))
        self.fc_2 = nn.utils.weight_norm(nn.Conv1d(hidden_dim*self.vjnum, (hidden_dim//2)*self.vjnum, 1,groups=self.vjnum))
        self.fc_3 = nn.utils.weight_norm(nn.Conv1d((hidden_dim//2)*self.vjnum, (hidden_dim//4)*self.vjnum, 1,groups=self.vjnum))
        self.fc_out = nn.utils.weight_norm(nn.Conv1d((hidden_dim//4)*self.vjnum, self.vjnum*6, 1,groups=self.vjnum))
        self.actvn = nn.LeakyReLU()
        self.RFFembed=RandomFourierFeatEnc(240,1,in_dim=3)# std=1
    
    def forward(self,querypoint,point_features,adjmat): 
        queryfea=self.RFFembed(querypoint)
        B,N,C=queryfea.size()
        point_features=point_features+queryfea
        #point_features=point_features.transpose(-2,-1)
        point_features=point_features.reshape(point_features.size(0),-1,1)
        point_features = self.actvn(self.fc_0(point_features))
        #print("point_featuresf0",point_features.size())
        point_features=point_features.reshape(B,N,-1,1)
        point_features=torch.einsum("bkjl,kk->bkjl",point_features,nn.functional.relu(adjmat))
        point_features=point_features.reshape(B,-1,1)
        point_features = self.actvn(self.fc_1(point_features))
        point_features=point_features.reshape(B,N,-1,1)
        point_features=torch.einsum("bkjl,kk->bkjl",point_features,nn.functional.relu(adjmat))
        point_features=point_features.reshape(B,-1,1)
        point_features = self.actvn(self.fc_2(point_features))
        point_features=point_features.reshape(B,N,-1,1)
        point_features=torch.einsum("bkjl,kk->bkjl",point_features,nn.functional.relu(adjmat))
        point_features=point_features.reshape(B,-1,1)
        point_features = self.actvn(self.fc_3(point_features))
        point_features = self.fc_out(point_features)
        point_features=point_features.reshape(B,N,6)#point_features.reshape(B,-1,N).transpose(-2,-1)
        return point_features

class VposeFiled_mlp(nn.Module):
    def __init__(self, feature_size,hidden_dim,output_dim):
        super(VposeFiled_mlp, self).__init__()
        self.fc_0 = nn.utils.weight_norm(nn.Conv1d(feature_size, hidden_dim, 1))
        self.fc_1 = nn.utils.weight_norm(nn.Conv1d(hidden_dim, hidden_dim*2, 1))
        self.fc_2 = nn.utils.weight_norm(nn.Conv1d(hidden_dim*2, hidden_dim*2, 1))
        self.fc_out = nn.utils.weight_norm(nn.Conv1d(hidden_dim*2, output_dim, 1))
        self.actvn = nn.LeakyReLU()
        #self.RFFembed=RandomFourierFeatEnc(240,1,in_dim=3)# std=1
        self.pointmlp=nn.Sequential(nn.utils.weight_norm(nn.Conv1d(3, feature_size//2, 1)),
                                    nn.utils.weight_norm(nn.Conv1d( feature_size//2, feature_size, 1))                                                       )
    
    def forward(self,querypoint,point_features):
        point_features=point_features.transpose(-2,-1)
        querypoint=querypoint.transpose(-2,-1)
        queryfea=self.pointmlp(querypoint)#self.RFFembed(querypoint)
        if point_features.size(1)<queryfea.size(1):
            point_features=point_features.repeat(1,5,1)+queryfea
        else:
            point_features=point_features+queryfea
        
        point_features = self.actvn(self.fc_0(point_features))
        point_features = self.actvn(self.fc_1(point_features))
        point_features = self.actvn(self.fc_2(point_features))
        point_features = self.fc_out(point_features)
        
        return point_features

class VposeFiled_high(nn.Module):
    def __init__(self, feature_size,hidden_dim,output_dim,hidim=240):
        super(VposeFiled_high, self).__init__()
        #self.fc_0 = nn.utils.weight_norm(nn.Conv1d(feature_size, hidden_dim, 1))
        self.fc_1 = nn.utils.weight_norm(nn.Conv1d(hidden_dim, hidden_dim*2, 1))
        self.fc_out = nn.utils.weight_norm(nn.Conv1d(hidden_dim*2, output_dim, 1))
        self.actvn = nn.LeakyReLU()
        self.RFFembed=RandomFourierFeatEnc(hidim,1,in_dim=3)# std=1
    
    def forward(self,querypoint,point_features):
        
        queryfea=self.RFFembed(querypoint)
        '''if point_features.size(1)<queryfea.size(1):
            point_features=point_features.repeat(1,5,1)+queryfea
        else:'''
        point_features=point_features+queryfea
        point_features=point_features.transpose(-2,-1)
        #point_features = self.actvn(self.fc_0(point_features))
        point_features = self.actvn(self.fc_1(point_features))
        #point_features = self.actvn(self.fc_2(point_features))
        point_features = self.fc_out(point_features)

        return point_features

class VposeFiled_jav(nn.Module):
    def __init__(self, feature_size,hidden_dim,output_dim,hidim=240):
        super(VposeFiled_jav, self).__init__()
        self.fc_0 = nn.utils.weight_norm(nn.Conv1d(feature_size, hidden_dim, 1))
        self.fc_1 = nn.utils.weight_norm(nn.Conv1d(hidden_dim, hidden_dim*2, 1))
        self.fc_2 = nn.utils.weight_norm(nn.Conv1d(hidden_dim*2, hidden_dim, 1))
        '''self.fc_3 = nn.utils.weight_norm(nn.Conv1d(hidden_dim*2, hidden_dim*2, 1))
        self.fc_4 = nn.utils.weight_norm(nn.Conv1d(hidden_dim*2, hidden_dim, 1))'''
        self.fc_out = nn.utils.weight_norm(nn.Conv1d(hidden_dim, output_dim, 1))
        self.actvn = nn.LeakyReLU()
        self.RFFembed=RandomFourierFeatEnc(hidim,1,in_dim=3)# std=1
        self.RFFembed_jav=RandomFourierFeatEnc(hidim,1,in_dim=9)
    
    def forward(self,querypoint,point_features,jav):
        queryfea=self.RFFembed(querypoint)
        javfea=self.RFFembed_jav(jav)
        '''if point_features.size(1)<queryfea.size(1):
            point_features=point_features.repeat(1,5,1)+queryfea
        else:'''
        point_features=point_features+queryfea+javfea
        point_features=point_features.transpose(-2,-1)
        point_features = self.actvn(self.fc_0(point_features))
        point_features = self.actvn(self.fc_1(point_features))
        point_features = self.actvn(self.fc_2(point_features))
        '''point_features = self.actvn(self.fc_3(point_features))
        point_features = self.actvn(self.fc_4(point_features))'''
        point_features = self.fc_out(point_features)

        return point_features

class VposeFiled_noquery(nn.Module):
    def __init__(self, feature_size,hidden_dim,output_dim,hidim=240):
        super(VposeFiled_noquery, self).__init__()
        self.fc_0 = nn.utils.weight_norm(nn.Conv1d(feature_size, hidden_dim, 1))
        self.fc_1 = nn.utils.weight_norm(nn.Conv1d(hidden_dim, hidden_dim*2, 1))
        self.fc_2 = nn.utils.weight_norm(nn.Conv1d(hidden_dim*2, hidden_dim, 1))
        self.fc_out = nn.utils.weight_norm(nn.Conv1d(hidden_dim, output_dim, 1))
        self.actvn = nn.LeakyReLU()
        hidim=feature_size//2
        self.RFFembed=RandomFourierFeatEnc(hidim,1,in_dim=3)# std=1
    
    def forward(self,point_features):
        point_features=point_features.transpose(-2,-1)
        point_features = self.actvn(self.fc_0(point_features))
        point_features = self.actvn(self.fc_1(point_features))
        point_features = self.actvn(self.fc_2(point_features))

        point_features = self.fc_out(point_features)

        return point_features

class VposeFiled(nn.Module):
    def __init__(self, feature_size,hidden_dim,output_dim,hidim=240):
        super(VposeFiled, self).__init__()
        self.fc_0 = nn.utils.weight_norm(nn.Conv1d(feature_size, hidden_dim, 1))
        self.fc_1 = nn.utils.weight_norm(nn.Conv1d(hidden_dim, hidden_dim*2, 1))
        self.fc_2 = nn.utils.weight_norm(nn.Conv1d(hidden_dim*2, hidden_dim, 1))
        self.fc_out = nn.utils.weight_norm(nn.Conv1d(hidden_dim, output_dim, 1))
        self.actvn = nn.LeakyReLU()
        hidim=feature_size//2
        self.RFFembed=RandomFourierFeatEnc(hidim,1,in_dim=3)# std=1
    
    def forward(self,querypoint,point_features):
        queryfea=self.RFFembed(querypoint)
        '''if point_features.size(1)<queryfea.size(1):
            point_features=point_features.repeat(1,5,1)+queryfea
        else:'''
        point_features=point_features+queryfea
        point_features=point_features.transpose(-2,-1)
        point_features = self.actvn(self.fc_0(point_features))
        point_features = self.actvn(self.fc_1(point_features))
        point_features = self.actvn(self.fc_2(point_features))
        '''point_features = self.actvn(self.fc_3(point_features))
        point_features = self.actvn(self.fc_4(point_features))'''
        point_features = self.fc_out(point_features)

        return point_features

class VposeFiled_transformer(nn.Module):
    def __init__(self, feature_size,hidden_dim,output_dim,hidim=240):
        super(VposeFiled_transformer, self).__init__()
        decoder_layer = nn.TransformerDecoderLayer(d_model=feature_size, nhead=4)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        '''self.fc_0 = nn.utils.weight_norm(nn.Conv1d(feature_size, hidden_dim, 1))
        self.fc_1 = nn.utils.weight_norm(nn.Conv1d(hidden_dim, hidden_dim*2, 1))
        self.fc_2 = nn.utils.weight_norm(nn.Conv1d(hidden_dim*2, hidden_dim, 1))'''
        self.fc_out = nn.utils.weight_norm(nn.Conv1d(feature_size, output_dim, 1))
        self.actvn = nn.LeakyReLU()
        self.RFFembed=RandomFourierFeatEnc(hidim,1,in_dim=3)# std=1
    
    def forward(self,querypoint,point_features):
        queryfea=self.RFFembed(querypoint)
        queryfea=queryfea.permute(1,0,2)
        point_features=point_features.permute(1,0,2)
        point_features=self.transformer_decoder(queryfea,point_features)
        point_features=point_features.transpose(0,1)#.permute(1,2,0)
        point_features=point_features.transpose(2,1)
        '''point_features=point_features+queryfea
        point_features=point_features.transpose(-2,-1)
        point_features = self.actvn(self.fc_0(point_features))
        point_features = self.actvn(self.fc_1(point_features))
        point_features = self.actvn(self.fc_2(point_features))'''

        point_features = self.fc_out(point_features)

        return point_features

class VposeFiled_fea(nn.Module):
    def __init__(self, feature_size,hidden_dim,output_dim,hidim=240):
        super(VposeFiled_fea, self).__init__()
        self.fc_0 = nn.utils.weight_norm(nn.Conv1d(feature_size, hidden_dim, 1))
        self.fc_1 = nn.utils.weight_norm(nn.Conv1d(hidden_dim, hidden_dim*2, 1))
        self.fc_2 = nn.utils.weight_norm(nn.Conv1d(hidden_dim*2, hidden_dim, 1))
        self.fc_out = nn.utils.weight_norm(nn.Conv1d(hidden_dim, output_dim, 1))
        self.actvn = nn.LeakyReLU()
        self.RFFembed=RandomFourierFeatEnc(120,1,in_dim=240,include_input=True)# std=1
    
    def forward(self,querypoint,point_features):
        queryfea=self.RFFembed(querypoint)
        '''if point_features.size(1)<queryfea.size(1):
            point_features=point_features.repeat(1,5,1)+queryfea
        else:'''
        point_features=point_features+queryfea
        point_features=point_features.transpose(-2,-1)
        point_features = self.actvn(self.fc_0(point_features))
        point_features = self.actvn(self.fc_1(point_features))
        point_features = self.actvn(self.fc_2(point_features))
        '''point_features = self.actvn(self.fc_3(point_features))
        point_features = self.actvn(self.fc_4(point_features))'''
        point_features = self.fc_out(point_features)

        return point_features
        
def attention(query, key, value, mask=None, dropout=0.0):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
            / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = nn.functional.softmax(scores, dim = -1)
    # (Dropout described below)
    #p_attn = F.dropout(p_attn, p=dropout)
    
    return torch.einsum("bjl,blk->bjk",p_attn,value), p_attn
    
class Cross_att_VposeFiledpart(nn.Module):
    def __init__(self, feature_size,hidden_dim,output_dim,hidim=240):
        super(Cross_att_VposeFiledpart, self).__init__()
        self.fc_0 = nn.utils.weight_norm(nn.Conv1d(feature_size, hidden_dim, 1))
        self.fc_1 = nn.utils.weight_norm(nn.Conv1d(hidden_dim, hidden_dim*2, 1))
        self.fc_2 = nn.utils.weight_norm(nn.Conv1d(hidden_dim*2, hidden_dim, 1))
        self.fc_out = nn.utils.weight_norm(nn.Conv1d(hidden_dim, output_dim, 1))
        self.actvn = nn.LeakyReLU()
        self.RFFembed=RandomFourierFeatEnc(hidim,1,in_dim=3)# std=1
        self.mlpq= nn.utils.weight_norm(nn.Conv1d(feature_size, feature_size, 1))
        self.mlpk= nn.utils.weight_norm(nn.Conv1d(feature_size, feature_size, 1))
    
    def forward(self,querypoint,point_features,jpos,vposeFeaWeight):
        #point_features  B,20,d
        #vposeFeaWeight有20个，接口留为d
        pointfea_list=[]
        for i in range(point_features.size(1)):
            temp=torch.einsum("bl,lkq->bkq", point_features[:,i],vposeFeaWeight[i])
            pointfea_list.append(temp.unsqueeze(1))
        point_features=torch.cat(pointfea_list,dim=1)
        point_features=point_features.sum(dim=1)
        point_features=point_features.transpose(-2,-1)
        point_features = self.actvn(self.fc_0(point_features))
        point_features = self.actvn(self.fc_1(point_features))
        point_features = self.actvn(self.fc_2(point_features))
        point_features = self.fc_out(point_features)

        return point_features.transpose(-2,-1).contiguous()

class Cross_att_VposeFiled(nn.Module):
    def __init__(self, feature_size,hidden_dim,output_dim,hidim=240):
        super(Cross_att_VposeFiled, self).__init__()
        self.fc_0 = nn.utils.weight_norm(nn.Conv1d(feature_size, hidden_dim, 1))
        self.fc_1 = nn.utils.weight_norm(nn.Conv1d(hidden_dim, hidden_dim*2, 1))
        self.fc_2 = nn.utils.weight_norm(nn.Conv1d(hidden_dim*2, hidden_dim, 1))
        '''self.fc_3 = nn.utils.weight_norm(nn.Conv1d(hidden_dim*2, hidden_dim*2, 1))
        self.fc_4 = nn.utils.weight_norm(nn.Conv1d(hidden_dim*2, hidden_dim, 1))'''
        self.fc_out = nn.utils.weight_norm(nn.Conv1d(hidden_dim, output_dim, 1))
        self.actvn = nn.LeakyReLU()
        self.RFFembed=RandomFourierFeatEnc(hidim,1,in_dim=3)# std=1
        #self.atten=attention()
        self.mlpq= nn.utils.weight_norm(nn.Conv1d(feature_size, feature_size, 1))
        self.mlpk= nn.utils.weight_norm(nn.Conv1d(feature_size, feature_size, 1))
    
    def forward(self,querypoint,point_features,jpos,vposeFeaWeight):
        queryfea=self.RFFembed(querypoint)
        queryfea=queryfea.transpose(-2,-1)
        queryfea=self.mlpq(queryfea)
        queryfea=queryfea.transpose(-2,-1)
        
        '''jposfea=self.RFFembed(jpos)
        jposfea=jposfea.transpose(-2,-1)
        jposfea=self.mlpk(jposfea)
        jposfea=jposfea.transpose(-2,-1)'''
        jposfea=point_features
        point_features,_=attention(queryfea,jposfea,vposeFeaWeight)
        vbindex=np.arange(80)
        point_features=point_features[:,vbindex,vbindex]
        '''if point_features.size(1)<queryfea.size(1):
            point_features=point_features.repeat(1,5,1)+queryfea
        else:'''
        #point_features=point_features+queryfea
        point_features=point_features.transpose(-2,-1)
        point_features = self.actvn(self.fc_0(point_features))
        point_features = self.actvn(self.fc_1(point_features))
        point_features = self.actvn(self.fc_2(point_features))
        '''point_features = self.actvn(self.fc_3(point_features))
        point_features = self.actvn(self.fc_4(point_features))'''
        point_features = self.fc_out(point_features)

        return point_features.transpose(-2,-1).contiguous()

class VposeFiled_andhigh(nn.Module):
    def __init__(self, feature_size,hidden_dim,output_dim,hidim=240,interface=100):
        super(VposeFiled_andhigh, self).__init__()
        self.fc_0 = nn.utils.weight_norm(nn.Conv1d(feature_size, hidden_dim, 1))
        self.fc_1 = nn.utils.weight_norm(nn.Conv1d(hidden_dim, hidden_dim*2, 1))
        self.fc_2 = nn.utils.weight_norm(nn.Conv1d(hidden_dim*2, hidden_dim*2, 1))
        self.fc_out = nn.utils.weight_norm(nn.Conv1d(hidden_dim*2, output_dim, 1))
        self.actvn = nn.LeakyReLU()
        self.RFFembed=RandomFourierFeatEnc(hidim,1,in_dim=3)# std=1
        self.fc_high = nn.utils.weight_norm(nn.Conv1d(feature_size, interface, 1))
        
        self.output_num=[feature_size,output_dim*4,12273*3]
        self.gru_out_linear = nn.Sequential(MLP(self.output_num, batch_norm=False, dropout=True))
        self.gru_out_dim = 3
        '''self.out_linear = nn.Sequential(nn.Linear(self.gnn_dim[-1] + self.gru_out_dim, 3))
        self.edge_index = edge_index
        self.edge_conv_layers = nn.ModuleList()

        for i in range(len(self.gnn_dim) - 1):
            self.edge_conv_layers.append(EdgeConv(MLP([2 * self.gnn_dim[i], self.gnn_dim[i + 1]])))'''
    
    def forward(self,querypoint,point_features,high_psd=None,template=None):
        
        gru_out = self.gru_out_linear(point_features).view((point_features.size(0), -1, self.gru_out_dim))
        point_features=point_features.unsqueeze(-1)
        queryfea=self.RFFembed(querypoint)
        point_features=point_features+queryfea.transpose(-2,-1)
        point_features = self.actvn(self.fc_0(point_features))
        point_features = self.actvn(self.fc_1(point_features))
        point_features = self.actvn(self.fc_2(point_features))
        point_features = self.fc_out(point_features)

        return point_features,gru_out

class VposeFiled_posi(nn.Module):
    def __init__(self, feature_size,hidden_dim,output_dim,hidim=240):
        super(VposeFiled_posi, self).__init__()
        self.fc_0 = nn.utils.weight_norm(nn.Conv1d(feature_size, hidden_dim, 1))
        self.fc_1 = nn.utils.weight_norm(nn.Conv1d(hidden_dim, hidden_dim*2, 1))
        self.fc_2 = nn.utils.weight_norm(nn.Conv1d(hidden_dim*2, hidden_dim, 1))
        self.fc_out = nn.utils.weight_norm(nn.Conv1d(hidden_dim, output_dim, 1))
        self.actvn = nn.LeakyReLU()
        self.RFFembed=RandomFourierFeatEnc(hidim,1,in_dim=128)# std=1
    
    def forward(self,querypoint,point_features,posimap):
        queryfea=self.RFFembed(posimap)
        point_features=point_features+queryfea
        point_features=point_features.transpose(-2,-1)
        point_features = self.actvn(self.fc_0(point_features))
        point_features = self.actvn(self.fc_1(point_features))
        point_features = self.actvn(self.fc_2(point_features))
        point_features = self.fc_out(point_features)

        return point_features

class VposeFiled_norandfea(nn.Module):
    def __init__(self, feature_size,hidden_dim,output_dim,hidim=240):
        super(VposeFiled_norandfea, self).__init__()
        #self.fc_query = nn.utils.weight_norm(nn.Conv1d(3,feature_size, 1))
        self.fc_0 = nn.utils.weight_norm(nn.Conv1d(feature_size, hidden_dim, 1))
        self.fc_1 = nn.utils.weight_norm(nn.Conv1d(hidden_dim, hidden_dim*2, 1))
        self.fc_2 = nn.utils.weight_norm(nn.Conv1d(hidden_dim*2, hidden_dim, 1))
        '''self.fc_3 = nn.utils.weight_norm(nn.Conv1d(hidden_dim*2, hidden_dim*2, 1))
        self.fc_4 = nn.utils.weight_norm(nn.Conv1d(hidden_dim*2, hidden_dim, 1))'''
        self.fc_out = nn.utils.weight_norm(nn.Conv1d(hidden_dim, output_dim, 1))
        self.actvn = nn.LeakyReLU()
        self.RFFembed=RandomFourierFeatEnc(hidim,1,in_dim=480)# std=1
    
    def forward(self,querypoint,point_features):
        
        #queryfea=self.fc_query(querypoint.transpose(-2,-1))#self.RFFembed(querypoint)
        '''if point_features.size(1)<queryfea.size(1):
            point_features=point_features.repeat(1,5,1)+queryfea
        else:'''
        #point_features=point_features+queryfea
        point_features=self.RFFembed(point_features)
        point_features=point_features.transpose(-2,-1)#+queryfea
        point_features = self.actvn(self.fc_0(point_features))
        point_features = self.actvn(self.fc_1(point_features))
        point_features = self.actvn(self.fc_2(point_features))
        '''point_features = self.actvn(self.fc_3(point_features))
        point_features = self.actvn(self.fc_4(point_features))'''
        point_features = self.fc_out(point_features)

        return point_features
class Sample(nn.Module):
    def __init__(self):
        super(Sample, self).__init__()
    def forward(self,z_mean,z_log_var):
        epsilon=torch.randn(z_mean.shape)
        epsilon=epsilon.cuda()#to('cuda:1')
        return z_mean+(z_log_var/2).exp()*epsilon
class VaeEncoder(nn.Module):
    def __init__(self,original_dim,intermediate_dim,latent_dim):
        super(VaeEncoder, self).__init__()
        self.Dense=nn.Linear(original_dim,intermediate_dim)
        self.z_mean=nn.Linear(intermediate_dim,latent_dim)
        self.z_log_var=nn.Linear(intermediate_dim,latent_dim)
        self.sample=Sample()
    def forward(self,x):
        o=torch.nn.functional.relu(self.Dense(x))
        z_mean=self.z_mean(o)
        z_log_var=self.z_log_var(o)
        #o=self.sample(z_mean,z_log_var)
        return z_mean,z_log_var#o,z_mean,z_log_var
class VaeDecoder(nn.Module):
    def __init__(self,latent_dim,intermediate_dim,original_dim):
        super(VaeDecoder, self).__init__()
        self.Dense=nn.Linear(latent_dim,intermediate_dim)
        self.out=nn.Linear(intermediate_dim,original_dim)
        #self.sigmoid=nn.Sigmoid()
    def forward(self,z):
        o=nn.functional.relu(self.Dense(z))
        o=self.out(o)
        return o
class Vae(nn.Module):
    def __init__(self,Inputdim=600,interdim=600,latentdim=600,inter2dim=600,outputdim=600):
        super(Vae, self).__init__()
        self.encoder=VaeEncoder(Inputdim,interdim,latentdim)
        self.decoder=VaeDecoder(latentdim,inter2dim,outputdim)
    def forward(self,x):
        o,mean,var=self.encoder(x)
        return self.decoder(o)#,mean,var

class OutandIn(nn.Module):
    # the implementation of pytorch geometric heavily relies on scatter_sum which can not be deterministic
    # so the GNN using this package can not achieve deterministic
    def __init__(self, Bone_nums, hidden_num, fea_dim=1024):
        super(OutandIn, self).__init__()
        #trans 
        self.Trans_GRU=GRU_Model(Bone_nums*3, hidden_num, [fea_dim])
        self.Fea_MLP=MLP_Dec(2048)
    
    def forward(self,Body_Fea,Trans,trans_hidden):
        Trans_fea,trans_new_hidden=self.Trans_GRU(Trans,trans_hidden)
        Trans_fea=Trans_fea.unsqueeze(-1).expand(Trans_fea.size(0),Trans_fea.size(1),Body_Fea.size(2))
        all_fea=torch.cat((Body_Fea,Trans_fea),1)
        output=self.Fea_MLP(all_fea)

        return output,trans_new_hidden

class FullyConnect(nn.Module):
    def __init__(self, in_dim, out_dim,isnorm=True):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=0.05)
        self.bn=nn.BatchNorm1d(out_dim, affine=True)
        self.isnorm=isnorm

    def forward(self, x):
        if self.isnorm:
            #return self.dropout(self.bn(self.act(self.linear(x))))
            return self.dropout(self.act(self.linear(x)))
        else:
            return self.dropout(self.act(self.linear(x)))

class MLP2(nn.Module):
    def __init__(self, input_d, d_list:list):
        super().__init__()
        dims = [input_d, *d_list]
        self.mlpbn = nn.Sequential(*(FullyConnect(dims[i], dims[i+1],True) for i in range(len(dims)-2)))
        self.mlp=FullyConnect(dims[-2], dims[-1],False)

    def forward(self, x):
        x=self.mlpbn(x)
        return self.mlp(x)


class VaeMLP(nn.Module):
    def __init__(self, input_d, d_list:list,cond=None):
        super().__init__()
        if cond is not None:
            dims = [input_d+cond, *d_list]
        else:
            dims = [input_d, *d_list]
        self.mlpbn = nn.Sequential(*(FullyConnect(dims[i], dims[i+1],True) for i in range(len(dims)-2)))
        self.mlp_mean=FullyConnect(dims[-2], dims[-1],False)
        self.mlp_std=FullyConnect(dims[-2], dims[-1],False)

    def forward(self, x,c):
        if c is not None:
            x=torch.cat((x,c),dim=-1)
        x=self.mlpbn(x)
        return self.mlp_mean(x),self.mlp_std(x)

class Joint2Coarse_Network(nn.Module):
    def __init__(self, Garment_Dims:int, SE_Dims:list, SD_Dims:list, Motion_Dims:int, ME_Dims:list):
        super().__init__()
        # SD_Dims.append(Garment_Dims)
        self.ShapeEncoder = MLP2(Garment_Dims, SE_Dims)
        self.ShapeDecoder = nn.Sequential(MLP2(SD_Dims[0], SD_Dims[1:]),
                                          nn.Linear(SD_Dims[-1], Garment_Dims))

    def forward(self, x, motion_latent_space):
        shape_latent_space = self.ShapeEncoder(x)
         
        x_pred = self.ShapeDecoder(shape_latent_space)
        m_pred = self.ShapeDecoder(motion_latent_space)
        return x_pred, shape_latent_space, m_pred, motion_latent_space
    
    def forward_inference(self, motion_latent_space):
        m_pred = self.ShapeDecoder(motion_latent_space)
        return m_pred, motion_latent_space

class Joint2Coarse_Network_VAE(nn.Module):
    def __init__(self, Garment_Dims:int, SE_Dims:list, SD_Dims:list, Motion_Dims:int, ME_Dims:list):
        super().__init__()
        # SD_Dims.append(Garment_Dims)
        self.ShapeEncoder = VaeMLP(Garment_Dims, SE_Dims,480)
        self.ShapeDecoder = nn.Sequential(MLP2(SD_Dims[0]+480, SD_Dims[1:]),
                                          nn.Linear(SD_Dims[-1], Garment_Dims))

    def forward(self, x, motion_latent_space):
        means, log_var = self.ShapeEncoder(x, motion_latent_space)
        z = self.reparameterize(means, log_var)
        conditional_fea=torch.cat((z,motion_latent_space),dim=-1)
        recon_x = self.ShapeDecoder(conditional_fea)
        KLD = -0.5 * torch.sum(1 + log_var - means.pow(2) - log_var.exp())
        '''shape_latent_space = self.ShapeEncoder(x)
         
        x_pred = self.ShapeDecoder(shape_latent_space)
        m_pred = self.ShapeDecoder(motion_latent_space)'''
        return recon_x,KLD#x_pred, shape_latent_space, m_pred, motion_latent_space
    
    def inference(self, z,motion_latent_space):
        conditional_fea=torch.cat((z,motion_latent_space),dim=-1)
        m_pred = self.ShapeDecoder(conditional_fea)
        return m_pred

    '''def forward_inference(self, motion_latent_space):
        m_pred = self.ShapeDecoder(motion_latent_space)
        return m_pred, motion_latent_space'''
    
    def reparameterize(self, mu, log_var):

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return mu + eps * std

class Joint2Coarse_Network_cVAE(nn.Module):
    def __init__(self, Garment_Dims:int, SE_Dims:list, SD_Dims:list, Motion_Dims:int, ME_Dims:list):
        super().__init__()
        # SD_Dims.append(Garment_Dims)
        self.ShapeEncoder = VaeMLP(Garment_Dims, SE_Dims)
        self.ShapeDecoder = nn.Sequential(MLP2(SD_Dims[0], SD_Dims[1:]),
                                          nn.Linear(SD_Dims[-1], Garment_Dims))

    def forward(self, x):
        means, log_var = self.ShapeEncoder(x,None)
        z = self.reparameterize(means, log_var)

        #conditional_fea=torch.cat((z,motion_latent_space),dim=-1)
        recon_x = self.ShapeDecoder(z)
        KLD = -0.5 * torch.sum(1 + log_var - means.pow(2) - log_var.exp())
        #recon_y = self.ShapeDecoder(motion_latent_space)

        '''shape_latent_space = self.ShapeEncoder(x)
         
        x_pred = self.ShapeDecoder(shape_latent_space)
        m_pred = self.ShapeDecoder(motion_latent_space)'''
        #latloss=torch.mean((motion_latent_space-z)**2)
        return recon_x,KLD,z#,recon_y,latloss#x_pred, shape_latent_space, m_pred, motion_latent_space
    
    def inference(self, motion_latent_space):
        #conditional_fea=torch.cat((z,motion_latent_space),dim=-1)
        m_pred = self.ShapeDecoder(motion_latent_space)
        return m_pred

    '''def forward_inference(self, motion_latent_space):
        m_pred = self.ShapeDecoder(motion_latent_space)
        return m_pred, motion_latent_space'''
    
    def reparameterize(self, mu, log_var):

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return mu + eps * std

if __name__ == "__main__":
    pass
