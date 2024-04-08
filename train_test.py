# -*- coding: utf-8 -*-
import random
from torch.autograd import Variable
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from MCSSL import *
from fusion import Discriminator
from torch.utils.data import DataLoader
from data_loaders import *
from utils import  *
import torch.optim as optim
import pickle
import os
import gc


def model_train(opt,multi_modal_dic,train_samples,SL_test,device,k):
    cudnn.deterministic = True
    torch.cuda.manual_seed_all(123)
    torch.manual_seed(123)
    random.seed(123)
    model = MCSSL(opt,multi_modal_dic).to(device)
    diff=DiffLoss()
    mse=MSE()
    discr=Discriminator(opt.feature_size).to(device)
    adversarial_loss = torch.nn.BCELoss().cuda()
    optimizer = define_optimizer(opt, model)
    scheduler = define_scheduler(opt, optimizer)
    print(model)
    print("Number of Trainable Parameters: %d" % count_parameters(model))

    train_losses,test_losses,aucs,auprs=[],[],[],[]
    train_custom_data_loader = SL_DatasetLoader(train_samples)
    train_loader = DataLoader(dataset = train_custom_data_loader, batch_size = opt.train_batch_size, num_workers = 4, shuffle = True)
    result=[0,0,0,0,0,0,0]

    for epoch in range(1, opt.niter+opt.niter_decay+1):
        model.train()
        pred_all = np.array([])
        gc.collect()
        train_total_loss,train_SL_loss=0,0
        train_adv_loss,train_rec_loss,train_org_loss=0,0,0
        for batch_idx, (SL_ppi, SL_go, SL_gaus, label) in enumerate(train_loader):
            SL_ppi = SL_ppi.view(SL_ppi.size(0), -1)
            SL_go = SL_go.view(SL_go.size(0), -1)
            SL_gaus = SL_gaus.view(SL_gaus.size(0), -1)
            pred,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12=model(SL_ppi.to(device),SL_go.to(device),SL_gaus.to(device))
            valid = Variable(torch.cuda.FloatTensor(label.shape[0], 1).fill_(1.0),requires_grad=False)
            fake = Variable(torch.cuda.FloatTensor(label.shape[0], 1).fill_(0.0), requires_grad=False)
            diff_loss = (diff(x7, x10) + diff(x8, x11) + diff(x9, x12)) / 3
            rec_loss=mse(x1,x2)+mse(x3,x4)+mse(x5,x6)
            loss_cn =SL_BCE_loss(pred, label.view(label.size(0),-1).to('cuda'))
            train_SL_loss+=loss_cn.data.item()

            loss_reg = regularize_weights(model=model)
            real_loss = adversarial_loss(discr(x7), valid)
            fake_loss = adversarial_loss(discr(x8), fake) + adversarial_loss(discr(x9), fake)
            d_loss = 0.5 * (real_loss + fake_loss)
            loss = loss_cn + opt.lambda_reg * loss_reg +d_loss +  rec_loss +  0.05*diff_loss

            train_total_loss+= loss.data.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()
            pred_all = np.concatenate((pred_all, pred.detach().cpu().numpy().reshape(-1)))
            train_adv_loss+=d_loss.data.item()
            train_rec_loss+=rec_loss.data.item()
            train_org_loss+=diff_loss.data.item()

        train_losses.append(train_total_loss / len(train_loader.dataset))
        print('epoch:%d,train set SL loss:%f' % (epoch,train_SL_loss / len(train_loader.dataset)))

        scheduler.step()
        if epoch%10==0:
            test_loss,AUC,AUPR,RE,PRE,ACC,F1,MCC= model_test(opt,model,multi_modal_dic,SL_test,device)
            print('epoch:%d, AUC:%f,AUPR:%f,F1:%f,MCC:%f'%(epoch,AUC,AUPR,F1,MCC))
            test_losses.append(sum(test_loss))
            aucs.append(AUC)
            auprs.append(AUPR)
            if AUPR>result[1]:
                result[0],result[1],result[2],result[3],result[4],result[5],result[6]=AUC,AUPR,RE,PRE,ACC,F1,MCC

    return model,optimizer,result

def model_test(opt,model,multi_modal_dic,SL_test, device):
    model.eval()
    test_custom_data_loader = SL_test_DatasetLoader(SL_test)
    test_loader = DataLoader(dataset = test_custom_data_loader,batch_size = opt.test_batch_size,num_workers = 4, shuffle = True)
    pred_all,label_all = np.array([]),np.array([])
    test_adv_loss,test_rec_loss,test_org_loss,test_SL_loss = 0,0,0,0
    for batch_idx, (SL_sample,SL_label) in enumerate(test_loader):
        SL_sample = SL_sample.to(device)
        SL_label = SL_label.to(device)
        SL_ppi,SL_go,SL_gaus=construct_test_SL_features(SL_sample,multi_modal_dic)
        pred,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12 = model(SL_ppi.to(device),SL_go.to(device),SL_gaus.to(device))
        mse = MSE()
        valid = Variable(torch.cuda.FloatTensor(SL_label.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = Variable(torch.cuda.FloatTensor(SL_label.shape[0], 1).fill_(0.0), requires_grad=False)
        discr = Discriminator(opt.feature_size).to(device)
        adversarial_loss = torch.nn.BCELoss().cuda()
        real_loss = adversarial_loss(discr(x7), valid)
        fake_loss = adversarial_loss(discr(x8), fake) + adversarial_loss(discr(x9),fake)
        d_loss = 0.5 * (real_loss + fake_loss)
        rec_loss = mse(x1, x2) + mse(x3, x4) + mse(x5, x6)
        diff = DiffLoss()
        diff_loss = (diff(x7, x10) + diff(x8, x11) + diff(x9, x12)) / 3

        pred = pred.view(pred.size(0))
        loss_cn = SL_BCE_loss(pred, SL_label)
        test_SL_loss += loss_cn.data.item()
        pred_all = np.concatenate((pred_all, pred.data.cpu().numpy().reshape(-1)))
        label_all = np.concatenate((label_all, SL_label.data.cpu().numpy().reshape(-1)))
        torch.cuda.empty_cache()
        test_adv_loss += d_loss.data.item()
        test_rec_loss += rec_loss.data.item()
        test_org_loss += diff_loss.data.item()

    test_adv_loss = test_adv_loss / len(test_loader.dataset)
    test_rec_loss = test_rec_loss / len(test_loader.dataset)
    test_org_loss = test_org_loss / len(test_loader.dataset)
    test_SL_loss  = test_SL_loss  / len(test_loader.dataset)

    AUC,AUPR,RE,PRE,ACC,F1,MCC = evalution_all(pred_all, label_all)

    return [test_adv_loss,test_rec_loss,test_org_loss,test_SL_loss],AUC,AUPR,RE,PRE,ACC,F1,MCC

