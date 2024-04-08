### data_loaders.py
# -*- coding: utf-8 -*-
import torch
from torch.utils.data.dataset import Dataset

class SL_DatasetLoader(Dataset):
    def __init__(self,data):
        self.SL_ppi = data['ppi']
        self.SL_go = data['go']
        self.SL_gaus = data['gaus']
        self.label = data['label']

    def __getitem__(self, index):
        single_label = torch.tensor(self.label[index]).type(torch.FloatTensor)
        single_SL_ppi = torch.tensor(self.SL_ppi[index]).type(torch.FloatTensor)
        single_SL_go= torch.tensor(self.SL_go[index]).type(torch.FloatTensor)
        single_SL_gaus = torch.tensor(self.SL_gaus[index]).type(torch.FloatTensor)

        return single_SL_ppi, single_SL_go, single_SL_gaus,single_label

    def __len__(self):
        return len(self.label)

class SL_test_DatasetLoader(Dataset):
    def __init__(self, data):
        self.sample = data['sample']
        self.label = data['label']

    def __getitem__(self, index):
        single_sample = torch.tensor(self.sample[index]).type(torch.FloatTensor)
        single_label = torch.tensor(self.label[index]).type(torch.FloatTensor)
        return single_sample, single_label

    def __len__(self):
        return len(self.label)

class graph_fusion_DatasetLoader(Dataset):
    def __init__(self, data, split):
        self.X_gene = data[split]['x_gene']
        self.X_path = data[split]['x_path']
        self.X_cna = data[split]['x_cna']
        self.censored = data[split]['censored']
        self.survival = data[split]['survival']
    def __getitem__(self, index):
        single_censored = torch.tensor(self.censored[index]).type(torch.FloatTensor)
        single_survival = torch.tensor(self.survival[index]).type(torch.FloatTensor)
        single_X_gene = torch.tensor(self.X_gene[index]).type(torch.FloatTensor)
        single_X_path = torch.tensor(self.X_path[index]).type(torch.FloatTensor)
        single_X_cna = torch.tensor(self.X_cna[index]).type(torch.FloatTensor)

        return single_X_gene,single_X_path, single_X_cna,single_censored, single_survival

    def __len__(self):
        return len(self.X_gene)