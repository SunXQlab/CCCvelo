import torch
torch.cuda.empty_cache()
from collections import OrderedDict
import scanpy as sc
import pandas as pd
# import TFvelo as TFv
import os
import numpy as np

import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
import warnings
import time
from tqdm import tqdm
from collections import Counter
# import scvelo as scv

# CUDA support
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
warnings.filterwarnings("ignore")

def PrepareData(adata, hidden_dims):
    TFs_expr = torch.tensor(adata.layers['Imputate'][:, adata.var['TFs'].astype(bool)])
    TGs_expr = torch.tensor(adata.layers['Imputate'][:, adata.var['TGs'].astype(bool)])

    TGTF_regulate = torch.tensor(adata.varm['TGTF_regulate'])
    nonzero_idx = torch.nonzero(torch.sum(TGTF_regulate, dim=1)).squeeze()
    TGTF_regulate = TGTF_regulate[nonzero_idx].float()  # torch.Size([539, 114])

    TFLR_allscore = torch.tensor(adata.obsm['TFLR_signaling_score'])

    # adata = root_cell(adata, select_root)
    iroot = torch.tensor(adata.uns['iroot'])
    # print('the root cell is:', adata.uns['iroot'])

    N_TGs = TGs_expr.shape[1]
    layers = hidden_dims
    layers.insert(0, N_TGs+1)  # 在第一位插入90
    layers.append(N_TGs)  # 在最后一位追加89
    data = [TGs_expr, TFs_expr, TFLR_allscore, TGTF_regulate, iroot, layers]
    return data

def root_cell(adata, select_root):

    if select_root == 'STAGATE':
        max_cell_for_subsampling = 5000
        if adata.shape[0] < max_cell_for_subsampling:
            sub_adata_x = adata.obsm['X_umap']
            sum_dists = distance_matrix(sub_adata_x, sub_adata_x).sum(axis=1)
            adata.uns['iroot'] = np.argmax(sum_dists)
        else:
            indices = np.arange(adata.shape[0])
            selected_ind = np.random.choice(indices, max_cell_for_subsampling, False)
            sub_adata_x = adata.obsm['X_umap'][selected_ind, :]
            sum_dists = distance_matrix(sub_adata_x, sub_adata_x).sum(axis=1)
            adata.uns['iroot'] = np.argmax(sum_dists)
    elif select_root == 'UMAP':
        adata_copy = adata.copy()

        del adata_copy.obsm['X_umap']
        del adata_copy.uns['neighbors']
        del adata_copy.uns['umap']
        del adata_copy.obsp['distances']
        del adata_copy.obsp['connectivities']

        sc.tl.pca(adata_copy, svd_solver="arpack")
        sc.pp.neighbors(adata_copy)
        # sc.pp.neighbors(adata_copy, n_pcs=50)
        sc.tl.umap(adata_copy)

        max_cell_for_subsampling = 5000
        if adata_copy.shape[0] < max_cell_for_subsampling:
            sub_adata_x = adata_copy.obsm['X_umap']
            sum_dists = distance_matrix(sub_adata_x, sub_adata_x).sum(axis=1)
            adata.uns['iroot'] = np.argmax(sum_dists)
        else:
            indices = np.arange(adata_copy.shape[0])
            selected_ind = np.random.choice(indices, max_cell_for_subsampling, False)
            sub_adata_x = adata_copy.obsm['X_umap'][selected_ind, :]
            sum_dists = distance_matrix(sub_adata_x, sub_adata_x).sum(axis=1)
            adata.uns['iroot'] = np.argmax(sum_dists)
    elif select_root == 'CCC_genes':
        lig = adata.var['ligand'].astype(bool)
        rec = adata.var['receptor'].astype(bool)
        tf = adata.var['TFs'].astype(bool)
        tg = adata.var['TGs'].astype(bool)
        combined_bool = lig | rec | tf | tg
        sub_adata = adata[:, combined_bool]
        sub_adata = np.unique(sub_adata.var_names)
        sub_adata = adata[:, sub_adata]
        max_cell_for_subsampling = 5000
        if sub_adata.shape[0] < max_cell_for_subsampling:
            sub_adata_x = sub_adata.layers['Imputate']
            sum_dists = distance_matrix(sub_adata_x, sub_adata_x).sum(axis=1)
            adata.uns['iroot'] = np.argmax(sum_dists)
        else:
            indices = np.arange(sub_adata.shape[0])
            selected_ind = np.random.choice(indices, max_cell_for_subsampling, False)
            sub_adata_x = sub_adata.layers['Imputate'][selected_ind, :]
            sum_dists = distance_matrix(sub_adata_x, sub_adata_x).sum(axis=1)
            adata.uns['iroot'] = np.argmax(sum_dists)
    elif select_root == "spatial":
        max_cell_for_subsampling = 50000
        if adata.shape[0] < max_cell_for_subsampling:
            sub_adata_x = adata.obsm['spatial']
            sum_dists = distance_matrix(sub_adata_x, sub_adata_x).sum(axis=1)
            adata.uns['iroot'] = np.argmax(sum_dists)
        else:
            indices = np.arange(adata.shape[0])
            selected_ind = np.random.choice(indices, max_cell_for_subsampling, False)
            sub_adata_x = adata.obsm['spatial'][selected_ind, :]
            sum_dists = distance_matrix(sub_adata_x, sub_adata_x).sum(axis=1)
            adata.uns['iroot'] = np.argmax(sum_dists)
    elif type(select_root) == int:
        adata.uns['iroot'] = select_root

    return adata

# +++++++++++++++++++++++++++++++++++++ the deep neural network ++++++++++++++++++++++++++++++++++++++++
class DNN(torch.nn.Module):
    def __init__(self, layers):
        super(DNN, self).__init__()

        # parameters
        self.depth = len(layers) - 1

        # set up layer order dict
        self.activation = torch.nn.Tanh

        layer_list = list()
        for i in range(self.depth - 1):
            layer_list.append(
                ('layer_%d' % i, torch.nn.Linear(layers[i], layers[i + 1]))
            )
            layer_list.append(('activation_%d' % i, self.activation()))

        layer_list.append(
            ('layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1]))
        )
        layerDict = OrderedDict(layer_list)

        # deploy layers
        self.layers = torch.nn.Sequential(layerDict)

    def forward(self, x):
        out = self.layers(x)
        return out

class SpatialVelocity():
    def __init__(self, TGs_expr, TFs_expr, TFLR_allscore, TGTF_regulate, iroot, layers, lr, Lambda):
        # data
        self.TGs_expr = TGs_expr.clone().detach().float().to(device)
        self.TFs_expr = TFs_expr.clone().detach().float().to(device)
        self.TFLR_allscore = TFLR_allscore.clone().detach().float().to(device)
        self.regulate = TGTF_regulate.clone().detach().to(device)  # torch.Size([539, 114])
        self.iroot = iroot.int().to(device)
        self.t = torch.linspace(0, 1, 2000).unsqueeze(1).requires_grad_(True).to(device)
        self.Lambda = Lambda
        self.N_cell = TGs_expr.shape[0]
        self.N_TFs = TFs_expr.shape[1]
        self.N_TGs = TGs_expr.shape[1]
        self.N_LRs = TFLR_allscore.shape[2]

        self.rootcell_exp = self.TGs_expr[self.iroot, :]

        # setting parameters

        self.V1 = torch.empty((self.N_TFs, self.N_LRs), dtype=torch.float32).uniform_(0, 1).float().requires_grad_(True).to(device)
        self.K1 = torch.empty((self.N_TFs, self.N_LRs), dtype=torch.float32).uniform_(0, 1).float().requires_grad_(True).to(device)
        self.V2 = torch.empty((self.N_TGs, self.N_TFs), dtype=torch.float32).uniform_(0, 1).float().requires_grad_(True).to(device)
        self.K2 = torch.empty((self.N_TGs, self.N_TFs), dtype=torch.float32).uniform_(0, 1).float().requires_grad_(True).to(device)
        self.beta = torch.empty((self.N_TFs),dtype=torch.float32).uniform_(0, 1).float().requires_grad_(True).to(device)
        self.gamma = torch.empty((self.N_TGs),dtype=torch.float32).uniform_(0, 1).float().requires_grad_(True).to(device)

        self.V1 = torch.nn.Parameter(self.V1)  # 反向传播需要更新的参数
        self.K1 = torch.nn.Parameter(self.K1)  # 反向传播需要更新的参数
        self.V2 = torch.nn.Parameter(self.V2)  # 反向传播需要更新的参数
        self.K2 = torch.nn.Parameter(self.K2)  # 反向传播需要更新的参数
        self.beta = torch.nn.Parameter(self.beta)
        self.gamma = torch.nn.Parameter(self.gamma)

        # deep neural networks
        self.dnn = DNN(layers).to(device)
        self.dnn.register_parameter('V1', self.V1)
        self.dnn.register_parameter('K1', self.K1)
        self.dnn.register_parameter('V2', self.V2)
        self.dnn.register_parameter('K2', self.K2)
        self.dnn.register_parameter('beta', self.beta)
        self.dnn.register_parameter('gamma', self.gamma)

        self.optimizer_Adam = torch.optim.Adam(self.dnn.parameters(), lr=lr)
        self.iter = 0

    def net_z(self):
        t = self.t
        N_TGs = self.N_TGs
        z0 = self.rootcell_exp.repeat(t.size(0), 1)
        z_and_t = torch.cat([z0, t], dim=1)
        z_dnn = self.dnn(z_and_t)  # dim = 1 :按行并排

        for i in range(N_TGs):
            z_t_pre = torch.autograd.grad(
                z_dnn[:, i], t,
                grad_outputs=torch.ones_like(z_dnn[:, i]),
                retain_graph=True,
                create_graph=True
            )[0]
            if i == 0:
                dz_dt = z_t_pre
            else:
                dz_dt = torch.cat((dz_dt, z_t_pre), 1)
        z_dnn = torch.where(z_dnn > 0, z_dnn, torch.full_like(z_dnn, 0))

        return z_dnn, dz_dt

    def assign_latenttime(self,TGs_expr):
        tpoints = self.t
        z_dnn = self.net_z()[0]
        z_obs = TGs_expr
        loss_cell_to_t = torch.sum((z_dnn.unsqueeze(1) - z_obs.unsqueeze(0)) ** 2, dim=2)  # torch.Size([2000, 2515])
        pos = torch.argmin(loss_cell_to_t, dim=0)
        fit_t = tpoints[pos]
        fit_t = fit_t.flatten()[:, None].squeeze()
        # print('the minial loss position of cells is:', pos[1450:1500])
        # print('the shape of loss_cell_to_t is:', loss_cell_to_t.shape) # torch.Size([2000, 2515])
        # print('the fit_t is :\n',fit_t)
        return pos, fit_t

    def calculate_initial_y0(self):
        # calculate initial y0
        V1 = self.V1
        K1 = self.K1
        iroot = self.iroot
        TFLR_allscore = self.TFLR_allscore
        TFs_expr = self.TFs_expr
        # calculate initial y0
        x0 = TFLR_allscore[iroot,:,:]
        Y0 = TFs_expr[iroot,:]
        zero_y = torch.zeros(self.N_TFs, self.N_LRs).float().to(device)
        V1_ = torch.where(x0 > 0, V1, zero_y)  # torch.Size([10, 88, 63])
        K1_ = torch.where(x0 > 0, K1, zero_y)  # torch.Size([10, 88, 63])
        y0 = torch.sum((V1_ * x0) / ((K1_ + x0) + (1e-12)),dim=1) * Y0  # torch.Size([10, 88])
        return y0

    def hill_fun(self, y0, cell_i, t_i):  # trapezoidal rule approximation
        V1 = self.V1
        K1 = self.K1
        beta = self.beta
        TFLR_allscore = self.TFLR_allscore
        TFs_expr = self.TFs_expr
        x_i = TFLR_allscore[int(cell_i), :, :]
        Y_i = TFs_expr[int(cell_i), :]
        # zero_y = torch.zeros(self.N_TFs, self.N_LRs)
        zero_y = torch.zeros_like(self.V1)
        V1_ = torch.where(x_i > 0, V1, zero_y)  # torch.Size([88, 63])
        K1_ = torch.where(x_i > 0, K1, zero_y)  # torch.Size([88, 63])
        tmp1 = torch.sum((V1_ * x_i) / ((K1_ + x_i) + (1e-12)), dim=1) * Y_i
        tmp2 = tmp1 * torch.exp(beta*t_i)
        y_i = (((y0 + tmp2)*t_i)/2 + y0) * torch.exp(-beta*t_i)
        return y_i

    def solve_ym(self, fit_t):
        y0_ = self.calculate_initial_y0()
        N_cell = self.N_cell
        N_TFs = self.N_TFs
        y_ode = torch.zeros((N_cell,N_TFs)).to(device)
        for i in range(N_cell):
            t_i = fit_t[i]
            if t_i.item() == 0:
                y_ode[i] = y0_
            else:
                y_ode[i] = self.hill_fun(y0_,i,t_i)
        return y_ode

    def net_f2(self):
        N_cell = self.N_cell
        V2 = self.V2
        K2 = self.K2
        regulate = self.regulate
        N_TGs = self.N_TGs
        N_TFs = self.N_TFs
        TGs_expr = self.TGs_expr
        z_dnn, dz_dt = self.net_z()
        fit_t_pos, fit_t = self.assign_latenttime(TGs_expr)
        # print('the fit latent time is:\n', fit_t)

        # calculate ym
        y_ode = self.solve_ym(fit_t)

        # zero_z = torch.zeros(N_TGs, N_TFs)
        zero_z = torch.zeros_like(self.V2)
        V2_ = torch.where(regulate == 1, V2, zero_z)
        K2_ = torch.where(regulate == 1, K2, zero_z)
        tmp1 = V2_.unsqueeze(0) * y_ode.unsqueeze(1)
        tmp2 = (K2_.unsqueeze(0) + y_ode.unsqueeze(1)) + (1e-12)
        tmp3 = torch.sum(tmp1 / tmp2, dim=2)

        z_pred_exp = torch.zeros((N_cell, N_TGs)).to(device)
        dz_dt_pred = torch.zeros((N_cell, N_TGs)).to(device)
        for i in range(N_cell):
            z_pred_exp[i, :] = z_dnn[fit_t_pos[i]]
            dz_dt_pred[i, :] = dz_dt[fit_t_pos[i]]

        # print('[gamma] value is:', self.gamma[0:5])
        # print('[beta] value is:', self.beta[0:5])

        dz_dt_ode = tmp3 - self.gamma*z_pred_exp
        f = dz_dt_pred - dz_dt_ode

        return z_pred_exp, f

    def pre_velo(self,y_ode):
        N_cell = self.N_cell
        N_TGs = self.N_TGs
        # y_ode = self.solve_ym(t)
        pre_velo = torch.zeros((N_cell, N_TGs)).to(device)
        for i in range(N_cell):
            y_i = y_ode[i, :]
            ym_ = self.regulate * y_i
            tmp1 = self.V2 * ym_
            tmp2 = (self.K2 + ym_) + (1e-12)
            tmp3 = torch.sum(tmp1 / tmp2, dim=1)
            dz_dt = tmp3 - self.gamma * self.TGs_expr[i, :]
            pre_velo[i, :] = dz_dt
        return pre_velo

    def train(self, nIter):
        print('Training SpatialVelocity model...')
        self.dnn.train()
        loss_adam = []
        iteration_adam = []
        a = 0
        for epoch in range(nIter):
            z_pred, f_pred = self.net_f2()
            loss1 = torch.mean((self.TGs_expr - z_pred) ** 2)
            loss2 = torch.mean(f_pred ** 2)
            # loss = 0.1 * torch.mean((self.TGs_expr - z_pred) ** 2) + self.Lambda * torch.mean(f_pred ** 2) # for cortex and prostate
            loss = torch.mean((self.TGs_expr - z_pred) ** 2) + self.Lambda * torch.mean(f_pred ** 2) # for slide-seqv2 trunk
            # Backward and optimize
            self.optimizer_Adam.zero_grad()
            loss.backward()
            self.optimizer_Adam.step()
            iteration_adam.append(a)
            a += 1
            loss_adam.append(loss.item())
            # print('It: %d, Loss: %.3e' % (epoch, loss.item()))
            if epoch % 100 == 0:
                print('loss1: %.3e, loss2: %.3e'%(loss1.item(), loss2.item()))
                print('It: %d, Loss: %.3e' %(epoch, loss.item()))

        return iteration_adam, loss_adam

