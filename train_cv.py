# -*- coding: utf-8 -*-
import torch
import numpy as np
from utils import *
import pandas as pd
from data_loaders import *
from train_test import *
import argparse

def main(opt,cvmode='equal',datamode=1):
    kfold=5
    nodes_num=6375
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print(torch.cuda.device_count())

    inter_pairs, gene_ppi_features, gene_go_features, gene_gaussian_features = load_data(opt)
    pos_trains, pos_tests,neg_trains,neg_tests = cross_divide(cvmode, kfold,inter_pairs, nodes_num,seed=123)
    multi_modal_dic={'ppi':gene_ppi_features,'go':gene_go_features,'gaus':gene_gaussian_features}

    auc_pair, aupr_pair, recall_pair, precision_pair, accuracy_pair, f_measure_pair, MCC_pair = [], [], [], [], [], [], []
    for fold in range(0,5):
        print("fold:%d"%fold)
        SL_test={}
        train_samples=construct_SL_features(pos_trains[fold],neg_trains[fold],{'ppi':gene_ppi_features,'go':gene_go_features,'gaus':gene_gaussian_features})
        SL_test['sample']=np.concatenate((pos_tests[fold],neg_tests[fold]))
        SL_test['label']=np.concatenate((np.ones(len(pos_tests[fold])), np.zeros(len(neg_tests[fold]))))
        model, optimizer,result = model_train(opt,multi_modal_dic,train_samples,SL_test, device, fold)       #模型的训练

        auc_pair.append(result[0])
        aupr_pair.append(result[1])
        f_measure_pair.append(result[5])
        MCC_pair.append(result[6])
        print("result on fold%d AUC:%.6f, AUPR:%.6f,F1:%.6f,MCC:%.6f \n" % (fold+1,result[0],result[1],result[5],result[6]))


    aver_AUC, sdv_AUC  = mean_confidence_interval(auc_pair)
    aver_AUPR,sdv_AUPR = mean_confidence_interval(aupr_pair)
    aver_F1 , sdv_F1   = mean_confidence_interval(f_measure_pair)
    aver_MCC, sdv_MCC  = mean_confidence_interval(MCC_pair)

    print("Average results: AUC:%.6f, AUPR:%.6f,F1:%.6f,MCC:%.6f \n"%(aver_AUC,aver_AUPR,aver_F1,aver_MCC))




if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_size', type=int, default=128, help="input_size for feature vector")
    parser.add_argument('--dropout_rate', default=0.65, type=float,help='0 - 0.25. Increasing dropout_rate helps overfitting.')
    parser.add_argument('--hidden_size', default=512, type=int, help='the size of hidden units.')
    parser.add_argument('--weight_decay', default=1e-4, type=float,help='Used for Adam. L2 Regularization on weights. I normally turn this off if I am using L1. You should try')
    parser.add_argument('--niter_decay', type=int, default=10,help='# of iter to linearly decay learning rate to zero')
    parser.add_argument('--train_batch_size', type=int, default=256,help="Number of batches to train/test for. Default: 256")
    parser.add_argument('--test_batch_size', type=int, default=256,help="Number of batches to train/test for. Default: 256")
    parser.add_argument('--lr', default=1e-3, type=float, help='5e-4 for Adam | 1e-3 for AdaBound')
    parser.add_argument('--lambda_reg', type=float, default=1e-4)
    parser.add_argument('--gpu_ids', type=str, default='1',help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--label_dim', type=int, default=1, help='size of output')
    parser.add_argument('--beta1', type=float, default=0.9, help='0.9, 0.5 | 0.25 | 0')
    parser.add_argument('--beta2', type=float, default=0.999, help='0.999, 0.5 | 0.25 | 0')
    parser.add_argument('--niter', type=int, default=0, help='# of iter at starting learning rate')
    parser.add_argument('--lr_policy', default='linear', type=str,help='5e-4 for Adam | 1e-3 for AdaBound')
    parser.add_argument('--optimizer_type', type=str, default='adam')
    opt = parser.parse_known_args()[0]
    main(opt,cvmode='equal',datamode=1)