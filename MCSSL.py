# -*- coding: utf-8 -*-
from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import math
from fusion import *
import numpy as np
import torch.nn.functional as F


def define_optimizer(opt, model):
    optimizer = None
    if opt.optimizer_type == 'adabound':
        #optimizer = adabound.AdaBound(model.parameters(), lr=opt.lr, final_lr=0.1)
         pass
    elif opt.optimizer_type == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2), weight_decay=opt.weight_decay)
    elif opt.optimizer_type == 'adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay, initial_accumulator_value=0.1)
    elif opt.optimizer_type == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2),
                                      weight_decay=opt.weight_decay)
    elif opt.optimizer_type == 'sgd':
        torch.optim.SGD(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % opt.optimizer)
    return optimizer


def init_parameters(layer):
    if type(layer)==nn.Linear:
        nn.init.kaiming_uniform_(layer.weight,0)

def define_scheduler(opt, optimizer):
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'exp':
        scheduler = lr_scheduler.ExponentialLR(optimizer, 0.1, last_epoch=-1)
    elif opt.lr_policy == 'step':
       scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
       scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
       scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
       return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


class graph_attention(nn.Module):
    """
    Simple
    """
    def __init__(self,atten_size):
        super(graph_attention, self).__init__()
        self.atten_size=atten_size
        self.weight = nn.Parameter(torch.FloatTensor(atten_size, atten_size))
        self.p = nn.Parameter(torch.FloatTensor(atten_size))
        self.b= nn.Parameter(torch.FloatTensor(atten_size))
        self.reset_parameters('Xavier')
    def reset_parameters(self, init):
        if init == 'Xavier':
            fan_in, fan_out = self.weight.shape
            fanp=1
            init_range = np.sqrt(6.0 / (fan_in + fan_out))
            init_range_p_q=np.sqrt(6.0/fanp+fan_in)
            self.weight.data.uniform_(-init_range, init_range)
            self.b.data.uniform_(-6.0/init_range_p_q,6.0/init_range_p_q)
            self.p.data.uniform_(-6.0 / init_range_p_q, 6.0 / init_range_p_q)
        elif init == 'Kaiming':
            torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            fan_in, _ = self.weight.shape
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.b, -bound, bound)
            torch.nn.init.uniform_(self.p, -bound, bound)
        else:
            stdv = 1. / math.sqrt(self.atten_size)
            self.weight.data.uniform_(-stdv, stdv)
            self.bias.data.uniform_(-stdv, stdv)
    def forward(self, feature):

        m=torch.tensordot(feature,self.weight,dims=1)
        m=torch.tensordot(F.relu(m+self.b),self.p,dims=1)
        attention=nn.functional.softmax(m,dim=-1)
        em=torch.sum(feature*attention.unsqueeze(-1),dim=1)
        attention=F.relu(feature*self.weight+self.b)

        return em


class MCSSL(nn.Module):
    def __init__(self,opt,multi_modal_dic):
        super(MCSSL, self).__init__()
        self.multi_modal_dic=multi_modal_dic
        self.opt=opt
        self.common = nn.Sequential(nn.Linear(opt.feature_size*2, opt.feature_size*8), nn.ReLU(),
                                    nn.Linear(opt.feature_size*8, opt.feature_size*4), nn.ReLU(),
                                    nn.Linear(opt.feature_size*4, opt.feature_size), nn.ReLU())
        self.lmf = LMF(opt.feature_size, int(opt.feature_size/4))
        self.att = nn.Sequential(nn.Linear(opt.feature_size * 2, opt.feature_size * 2), nn.ReLU())

        self.unique1 = nn.Sequential(nn.Linear(opt.feature_size*2, opt.feature_size*4), nn.ReLU(),
                                     nn.Linear(opt.feature_size*4, opt.feature_size*2), nn.ReLU(),
                                     nn.Linear(opt.feature_size*2, opt.feature_size), nn.ReLU())
    
        self.unique2 = nn.Sequential(nn.Linear(opt.feature_size*2, opt.feature_size*4), nn.ReLU(),
                                     nn.Linear(opt.feature_size*4, opt.feature_size*2), nn.ReLU(),
                                     nn.Linear(opt.feature_size*2, opt.feature_size), nn.ReLU())
       
        self.unique3 = nn.Sequential(nn.Linear(opt.feature_size*2, opt.feature_size*4), nn.ReLU(),
                                     nn.Linear(opt.feature_size*4, opt.feature_size*2), nn.ReLU(),
                                     nn.Linear(opt.feature_size*2, opt.feature_size), nn.ReLU())
        encoder1 = nn.Sequential(nn.Linear(opt.feature_size*4, opt.feature_size*2), nn.ReLU(), nn.Dropout(p=opt.dropout_rate))
        encoder2 = nn.Sequential(nn.Linear(opt.feature_size*2, int(opt.feature_size/2)), nn.ReLU(), nn.Dropout(p=opt.dropout_rate))

        self.encoder = nn.Sequential(encoder1, encoder2)
        self.classifier = nn.Sequential(nn.Linear(int(opt.feature_size/2), 1), nn.Sigmoid())

        ### Path
        self.linear_h1 = nn.Sequential(nn.Linear(opt.feature_size,opt.feature_size), nn.ReLU())
        self.linear_z1 = nn.Bilinear(opt.feature_size,opt.feature_size*2,opt.feature_size)
        self.linear_o1 = nn.Sequential(nn.Linear(opt.feature_size,opt.feature_size), nn.ReLU(), nn.Dropout(p=opt.dropout_rate))

        ### Graph
        self.linear_h2 = nn.Sequential(nn.Linear(opt.feature_size,opt.feature_size), nn.ReLU())
        self.linear_z2 = nn.Bilinear(opt.feature_size,opt.feature_size*2,opt.feature_size)
        self.linear_o2 = nn.Sequential(nn.Linear(opt.feature_size,opt.feature_size), nn.ReLU(), nn.Dropout(p=opt.dropout_rate))

        ### Omic
        self.linear_h3 = nn.Sequential(nn.Linear(opt.feature_size,opt.feature_size), nn.ReLU())
        self.linear_z3 = nn.Bilinear(opt.feature_size,opt.feature_size*2,opt.feature_size)
        self.linear_o3 = nn.Sequential(nn.Linear(opt.feature_size,opt.feature_size), nn.ReLU(), nn.Dropout(p=opt.dropout_rate))

        self.rec1=nn.Sequential(nn.Linear(opt.feature_size*2,opt.feature_size*4),nn.ReLU(),
                                   nn.Linear(opt.feature_size*4,opt.feature_size*2),nn.ReLU())
        self.rec2 = nn.Sequential(nn.Linear(opt.feature_size*2,opt.feature_size*4),nn.ReLU(),
                                   nn.Linear(opt.feature_size*4,opt.feature_size*2),nn.ReLU())
        self.rec3 = nn.Sequential(nn.Linear(opt.feature_size*2,opt.feature_size*4),nn.ReLU(),
                                   nn.Linear(opt.feature_size*4,opt.feature_size*2),nn.ReLU())
        init='Kaiming'
        if init=='Kaiming':
            self.rec1.apply(init_parameters)
            self.rec2.apply(init_parameters)
            self.rec3.apply(init_parameters)
            self.linear_h3.apply(init_parameters)
            self.linear_o3.apply(init_parameters)
            self.encoder.apply(init_parameters)
            self.unique1.apply(init_parameters)
            self.unique2.apply(init_parameters)
            self.unique3.apply(init_parameters)
            self.linear_h1.apply(init_parameters)
            self.linear_o1.apply(init_parameters)
            self.linear_h2.apply(init_parameters)
            self.linear_o2.apply(init_parameters)
            self.att.apply(init_parameters)

    def forward(self, x_gene, x_path, x_can):
        x_gene_common = self.common(x_gene)
        x_path_common = self.common(x_path)
        x_can_common = self.common(x_can)


        h1 = self.linear_h1(x_gene_common)
        vec31=torch.cat((x_path_common,x_can_common),dim=1)
        z1 = self.linear_z1(x_gene_common, vec31)
        o1 = self.linear_o1(nn.Sigmoid()(z1) * h1)

        h2 = self.linear_h1(x_path_common)
        vec32 = torch.cat((x_gene_common, x_can_common),dim=1)
        z2 = self.linear_z1(x_path_common, vec32)
        o2 = self.linear_o1(nn.Sigmoid()(z2) * h2)

        h3 = self.linear_h1(x_can_common)
        vec33 = torch.cat((x_gene_common, x_path_common),dim=1)
        z3 = self.linear_z1(x_can_common, vec33)
        o3 = self.linear_o1(nn.Sigmoid()(z3) * h3)
        
        lmf=self.lmf(o1,o2,o3)
        x_gene=torch.mul(self.att(x_gene),x_gene)
        x_path=torch.mul(self.att(x_path),x_path)
        x_can =torch.mul(self.att(x_can),x_can)


        x_gene_unique = self.unique1(x_gene)    #
        x_path_unique = self.unique2(x_path)
        x_can_unique = self.unique3(x_can)

        out_fusion = torch.cat((lmf, x_gene_unique, x_path_unique, x_can_unique),dim=1)
        encoder = self.encoder(out_fusion)
        out = self.classifier(encoder)

        gene_rec=self.rec1(torch.cat((lmf,x_gene_unique),dim=1))
        path_rec = self.rec2(torch.cat((lmf,x_path_unique),dim=1))
        can_rec = self.rec3(torch.cat((lmf,x_can_unique),dim=1))
        return out, x_gene, gene_rec, x_path, path_rec, x_can, can_rec, x_gene_common, x_path_common, x_can_common,x_gene_unique,x_path_unique,x_can_unique
       