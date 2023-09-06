import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torch.nn.utils import weight_norm
import math
import copy
from typing import Optional, Tuple
import matplotlib.pyplot as plt
from einops import rearrange




class STEAM_spatial_attention(nn.Module):
    def __init__(self, time, hidden_dim, spa_hidden_dim):
        super(STEAM_spatial_attention, self).__init__()

        self.spatial_embedding = nn.Sequential(nn.Conv1d(time, spa_hidden_dim, kernel_size=1, stride=1), nn.BatchNorm1d(spa_hidden_dim), nn.ReLU())
        self.mlp = nn.Sequential(nn.Linear(spa_hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1), nn.Sigmoid())



    def forward(self, X):
        minibatch_size, num_nodes, num_timepoints = X.shape[:3]
        x = rearrange(X, 'b c t -> b t c')
        y = self.spatial_embedding(x)
        x_embedded = rearrange(y, 'b t c -> b c t')


        diff = x_embedded.unsqueeze(2) - x_embedded.unsqueeze(1)
        distance = torch.sqrt(torch.relu(diff ** 2) + 1e-9)
        distance = rearrange(distance, 'b n1 n2 c -> (b n1 n2) c')
        clf_output = self.mlp(distance)
        clf_output = rearrange(clf_output, '(b n1 n2) c -> b n1 n2 c', b=minibatch_size, n1=num_nodes, n2=num_nodes).squeeze(-1)
        mask = torch.eye(num_nodes).unsqueeze(0).repeat(clf_output.size(0), 1, 1).bool()
        clf_output_copy = clf_output.clone()
        clf_output_copy[mask] = 1.
        Similarity_M = clf_output_copy


        values, vectors = torch.linalg.eigh(Similarity_M)
        values, values_ind = torch.sort(values, dim=1, descending=True)

        vectors_copy = vectors.clone()
        for j in range(Similarity_M.size(0)):
            vectors_copy[j] = vectors_copy[j, :, values_ind[j]]

        new_X_list = []
        for k in range(Similarity_M.size(0)):
            spatial_new_X_list = []
            for i in range(15):
                selected_eigenvectors = vectors_copy[k, :, i]
                selected_eigenvectors = torch.abs(selected_eigenvectors)
                new_X = X[k] * selected_eigenvectors.unsqueeze(1)
                spatial_new_X_list.append(new_X)
            aa = torch.stack(spatial_new_X_list)
            new_X_list.append(aa)
        spatial_attentive_signals = torch.stack(new_X_list)


        return spatial_attentive_signals




class ModuleTransformer(nn.Module):
    def __init__(self, n_region, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(n_region, num_heads)
        self.layer_norm1 = nn.LayerNorm(n_region)
        self.layer_norm2 = nn.LayerNorm(n_region)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.mlp = nn.Sequential(nn.Linear(n_region, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, n_region))

    def forward(self, q, k, v):
        x_attend, attn_matrix = self.multihead_attn(q, k, v)
        x_attend = self.dropout1(x_attend)
        x_attend = self.layer_norm1(x_attend)
        x_attend2 = self.mlp(x_attend)
        x_attend = x_attend + self.dropout2(x_attend2)
        x_attend = self.layer_norm2(x_attend)
        return x_attend, attn_matrix




class STEAM_temporal_attention(nn.Module):
    def __init__(self, n_region, hidden_dim):
        super(STEAM_temporal_attention, self).__init__()

        self.MLP = nn.Sequential(nn.Linear(n_region, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU())
        self.PE = nn.LSTM(hidden_dim, hidden_dim, 1)
        self.tatt = ModuleTransformer(hidden_dim, 2 * hidden_dim, num_heads=1, dropout=0.1)
        self.mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1), nn.Sigmoid())


    def forward(self, X):

        minibatch_size, num_timepoints, num_nodes = X.shape[:3]
        X_enc = rearrange(X, 'b t c -> (b t) c')
        X_enc = self.MLP(X_enc)
        X_enc = rearrange(X_enc, '(b t) c -> t b c', t=num_timepoints, b=minibatch_size)
        X_enc, (hn, cn) = self.PE(X_enc)
        X_enc, _ = self.tatt(X_enc, X_enc, X_enc)  # t b c
        X_enc = rearrange(X_enc, 't b c -> b t c', t=num_timepoints, b=minibatch_size)


        diff = X_enc.unsqueeze(2) - X_enc.unsqueeze(1)
        distance = torch.sqrt(torch.relu(diff ** 2) + 1e-9)
        distance = rearrange(distance, 'b n1 n2 c -> (b n1 n2) c')
        clf_output = self.mlp(distance)
        clf_output = rearrange(clf_output, '(b n1 n2) c -> b n1 n2 c', b=minibatch_size, n1=num_timepoints, n2=num_timepoints).squeeze(-1)
        mask = torch.eye(num_timepoints).unsqueeze(0).repeat(clf_output.size(0), 1, 1).bool()
        clf_output_copy = clf_output.clone()
        clf_output_copy[mask] = 1.
        Similarity_M = clf_output_copy





        values, vectors = torch.linalg.eigh(Similarity_M)
        values, values_ind = torch.sort(values, dim=1, descending=True)

        vectors_copy = vectors.clone()
        for j in range(Similarity_M.size(0)):
            vectors_copy[j] = vectors_copy[j, :, values_ind[j]]

        new_X_list = []
        for k in range(Similarity_M.size(0)):
            temporal_new_X_list = []
            for i in range(15):
                selected_eigenvectors = vectors_copy[k, :, i]
                selected_eigenvectors = torch.abs(selected_eigenvectors)
                new_X = X[k] * selected_eigenvectors.unsqueeze(1)
                temporal_new_X_list.append(new_X)
            aa = torch.stack(temporal_new_X_list)
            new_X_list.append(aa)
        temporal_attentive_signals = torch.stack(new_X_list)


        return temporal_attentive_signals









class STEAMNetwork(nn.Module):
    def __init__(self, roi, hidden_dim, spa_hidden_dim, time, num_classes, dropout=0.2):
        super(STEAMNetwork, self).__init__()

        self.temporal_model = STEAM_temporal_attention(roi, hidden_dim)
        self.spatial_model = STEAM_spatial_attention(time, hidden_dim, spa_hidden_dim)
        self.temporal_conv1 = nn.Conv2d(15, 32, kernel_size=(1, 116), stride=(1,))  # temporal_conv1 performs 1D-conv
        self.spatial_conv1 = nn.Conv2d(15, 32, kernel_size=(1, 176), stride=(1,))   # spatial_conv1 performs 1D-conv
        self.temporal_conv2 = nn.Conv1d(176, 1, kernel_size=1, stride=1)
        self.spatial_conv2 = nn.Conv1d(116, 1, kernel_size=1, stride=1)
        self.last_layer = nn.Sequential(nn.Linear(32, hidden_dim), nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes))


    def forward(self, inputs):
        st_attention = {'temporal-result': [], 'spatial-result': []}


        temporal_result = self.temporal_model(inputs)
        inputs = rearrange(inputs, 'b t c -> b c t')
        spatio_result = self.spatial_model(inputs)

        tem_out = self.temporal_conv1(temporal_result).squeeze(-1)
        spa_out = self.spatial_conv1(spatio_result).squeeze(-1)


        for s in range(spa_out.shape[0]):
            a = (spa_out[s].cpu().detach().numpy() - np.min(spa_out[s].cpu().detach().numpy())) / (np.max(spa_out[s].cpu().detach().numpy()) - np.min(spa_out[s].cpu().detach().numpy()))
            b = (tem_out[s].cpu().detach().numpy() - np.min(tem_out[s].cpu().detach().numpy())) / (np.max(tem_out[s].cpu().detach().numpy()) - np.min(tem_out[s].cpu().detach().numpy()))
            st_attention['spatial-result'].append(a)
            st_attention['temporal-result'].append(b)



        spa_out = rearrange(spa_out, 'b f c -> b c f')
        tem_out = rearrange(tem_out, 'b f c -> b c f')

        tem_out = self.temporal_conv2(tem_out).squeeze(1)
        spa_out = self.spatial_conv2(spa_out).squeeze(1)


        spatio_temporal_input = spa_out + tem_out
        logit = self.last_layer(spatio_temporal_input)



        return logit, st_attention



