# -*- coding: utf-8 -*-
import torch
import random
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve, accuracy_score
from sklearn.metrics import auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score


def regularize_weights(model, reg_type=None):
    l1_reg = None
    for W in model.parameters():
        if l1_reg is None:
            l1_reg = torch.abs(W).sum()
        else:
            l1_reg = l1_reg + torch.abs(W).sum()
    return l1_reg



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def SL_BCE_loss(y_pred,y_true):

    eps = 1e-8
    loss = -torch.sum(torch.log(y_pred + eps) * y_true + torch.log(1 - y_pred + eps) * (1 - y_true))
    return loss

def SL_MSE_loss(y_pred,y_true):
    mask = torch.sign(y_true)
    y_pred = y_pred.float()
    y_true = y_true.float()
    ret = torch.pow( (y_pred-y_true)* mask , 2)
    return torch.sum( ret )




class DiffLoss(nn.Module):

    def __init__(self):
        super(DiffLoss, self).__init__()

    def forward(self, input1, input2):
        batch_size = input1.size(0)
        input1 = input1.view(batch_size, -1)
        input2 = input2.view(batch_size, -1)

        input1_mean = torch.mean(input1, dim=0, keepdims=True)
        input2_mean = torch.mean(input2, dim=0, keepdims=True)
        input1 = input1 - input1_mean
        input2 = input2 - input2_mean

        input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()
        input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)

        input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
        input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)

        diff_loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))

        return diff_loss


class CMD(nn.Module):
    """
    Adapted from https://github.com/wzell/cmd/blob/master/models/domain_regularizer.py
    """

    def __init__(self):
        super(CMD, self).__init__()

    def forward(self, x1, x2, n_moments=5):
        mx1 = torch.mean(x1, 0)
        mx2 = torch.mean(x2, 0)
        sx1 = x1 - mx1
        sx2 = x2 - mx2
        dm = self.matchnorm(mx1, mx2)
        scms = dm
        for i in range(n_moments - 1):
            scms += self.scm(sx1, sx2, i + 2)
        return scms

    def matchnorm(self, x1, x2):
        power = torch.pow(x1 - x2, 2)
        summed = torch.sum(power)
        sqrt = summed ** (0.5)
        return sqrt

    def scm(self, sx1, sx2, k):
        ss1 = torch.mean(torch.pow(sx1, k), 0)
        ss2 = torch.mean(torch.pow(sx2, k), 0)
        return self.matchnorm(ss1, ss2)
class CosineSimilarity(nn.Module):
    def __init__(self, dim=1, eps=1e-8):
        super(CosineSimilarity, self).__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x1, x2):
        return torch.sum(F.cosine_similarity(x1, x2, self.dim, self.eps))/x1.size(0)


class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, -pred)
        n = torch.numel(diffs.data)
        mse = torch.sum(diffs.pow(2)) / n-torch.sum(diffs.pow(2)) / (n*n)

        return mse



def load_data(opt):
    with open("./SLDB/List_Proteins_in_SL.txt", "r") as inf:
        gene_names = [line.rstrip() for line in inf]
        gene_id_mapping = dict(zip(gene_names, range(len(set(gene_names)))))
        gene_number = len(gene_names)

    gene_inter_pairs = []
    R_11 = np.zeros(shape=(gene_number, gene_number), dtype=np.int32)
    with open("./SLDB/SL.txt", "r") as inf:
      for line in inf:
          id1, id2, s = line.rstrip().split(" ")
          gene_inter_pairs.append((id1,id2))
          gene_inter_pairs.append((id2,id1))
      inter_pairs = np.array(gene_inter_pairs, dtype=int)


    gene_ppi_features = pd.read_csv(f"./SLDB/feature_ppi_{opt.feature_size}.txt", delimiter=' ',header=None).values
    gene_go_features = pd.read_csv(f"./SLDB/feature_go_{opt.feature_size}.txt", delimiter=' ', header=None).values
    gene_gaussian_features = pd.read_csv(f"./SLDB/feature_gauss_{opt.feature_size}.txt", delimiter=' ', header=None).values
    return inter_pairs,gene_ppi_features,gene_go_features,gene_gaussian_features


def cross_divide(cvmode,kfold, inter_pairs, nodes_num , seed=123):

    pos_train_kfold = []
    neg_train_kfold = []
    pos_test_kfold = []
    neg_test_kfold = []

    kf = KFold(n_splits=kfold, shuffle=True, random_state=seed)
    if cvmode == "equal":
        x, y = np.triu_indices(nodes_num, k=1)
        neg_set = set(zip(x, y)) - set(zip(inter_pairs[:, 0], inter_pairs[:, 1])) - set(
            zip(inter_pairs[:, 1], inter_pairs[:, 0]))
        noninter_pairs = np.array(list(neg_set))

        random.seed(seed)
        neg_random_sample_list = random.sample(range(1, len(noninter_pairs)), len(inter_pairs))
        noninter_pairs = noninter_pairs[neg_random_sample_list]

        pos_edge_kf = kf.split(inter_pairs)
        neg_edge_kf = kf.split(noninter_pairs)
        for pos_train_id, pos_test_id in pos_edge_kf:
            pos_train_kfold.append(inter_pairs[pos_train_id])
            pos_test_kfold.append(inter_pairs[pos_test_id])

        for neg_train_id, neg_test_id in neg_edge_kf:

            neg_train_kfold.append(noninter_pairs[neg_train_id])
            neg_test_kfold.append(noninter_pairs[neg_test_id])

    return pos_train_kfold, pos_test_kfold, neg_train_kfold, neg_test_kfold




def split_integer(m, n):
    assert n > 0
    quotient = int(m / n)
    remainder = m % n
    if remainder > 0:
        return [quotient] * (n - remainder) + [quotient + 1] * remainder
    if remainder < 0:
        return [quotient - 1] * -remainder + [quotient] * (n + remainder)
    return [quotient] * n


def construct_SL_features(pos_pairs,neg_pairs,features):
    SL_samples= {}
    SL_samples['label']=np.concatenate((np.ones(len(pos_pairs)),np.zeros(len(neg_pairs))))
    for modal, feature in features.items():
        pos_features=np.concatenate((feature[pos_pairs[:,0]],feature[pos_pairs[:,1]]),axis=1)
        neg_features=np.concatenate((feature[neg_pairs[:,0]],feature[neg_pairs[:,1]]),axis=1)
        SL_samples[modal]=np.concatenate((pos_features,neg_features))
    return SL_samples


def construct_test_SL_features(SL_pairs,multi_modal_dic):

    SL_ppi_features=torch.from_numpy(np.concatenate((multi_modal_dic['ppi'][SL_pairs[:,0].long().to('cpu')],multi_modal_dic['ppi'][SL_pairs[:,1].long().to('cpu')]),axis=1))
    SL_go_featuers=torch.from_numpy(np.concatenate((multi_modal_dic['go'][SL_pairs[:,0].long().to('cpu')],multi_modal_dic['go'][SL_pairs[:,1].long().to('cpu')]),axis=1))
    SL_gauss_features=torch.from_numpy(np.concatenate((multi_modal_dic['gaus'][SL_pairs[:,0].long().to('cpu')],multi_modal_dic['gaus'][SL_pairs[:,1].long().to('cpu')]),axis=1))
    SL_ppi_features=SL_ppi_features.type(torch.float32)
    SL_go_featuers=SL_go_featuers.type(torch.float32)
    SL_gauss_features=SL_gauss_features.type(torch.float32)
    return SL_ppi_features,SL_go_featuers,SL_gauss_features

def evalution(preds,labels):
    fpr, tpr, auc_thresholds = roc_curve(labels, preds)
    roc_score = auc(fpr, tpr)
    precisions, recalls, pr_thresholds = precision_recall_curve(labels, preds)
    aupr_score = auc(recalls, precisions)
    labels_all = labels.astype(np.float32)
    return roc_score, aupr_score

def evalution_all(preds,labels,outputfile=None,Kfold=1):

    preds_all = preds
    labels_all = labels


    fpr, tpr, auc_thresholds = roc_curve(labels_all, preds_all)
    roc_score = auc(fpr, tpr)
    precisions, recalls, pr_thresholds = precision_recall_curve(labels_all, preds_all)
    aupr_score = auc(recalls, precisions)
    #labels_all = labels_all.astype(np.float32)
    all_F_measure = np.zeros(len(pr_thresholds))
    max_index,max_f_measure=0,0
    for k in range(0, len(pr_thresholds)):
        if (precisions[k] + recalls[k]) > 0:
            f_measure= 2 * precisions[k] * recalls[k] / (precisions[k] + recalls[k])
            if f_measure>max_f_measure:
                max_f_measure=f_measure
                max_index=k

    threshold = pr_thresholds[max_index]
    predicted_labels = np.zeros(len(labels_all))
    predicted_labels[preds_all > threshold] = 1
    f_measure = f1_score(labels_all, predicted_labels)
    accuracy = accuracy_score(labels_all, predicted_labels)
    precision = precision_score(labels_all, predicted_labels)
    recall = recall_score(labels_all, predicted_labels)
    TP=np.multiply(labels_all,predicted_labels).sum()
    TN=np.multiply((1-labels_all),(1-predicted_labels)).sum()
    FP=predicted_labels.sum()-TP
    FN=labels_all.sum()-TP
    MCC=(TP*TN-FP*FN)/np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))

    return roc_score, aupr_score,recall,precision,accuracy,f_measure,MCC


def mean_confidence_interval(data, confidence=0.95):
    import scipy as sp
    import scipy.stats
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m, h

