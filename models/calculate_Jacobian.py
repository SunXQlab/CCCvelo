import os
import torch
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.spatial.distance import pdist, squareform

# from models2.preprocess_CCCvelo import *
from models2.calculateLRscore import *

def calculate_y_ode(t, y0, x, Y, model):
    V1 = model.V1
    K1 = model.K1
    x_i = x
    Y_i = Y
    t_i = t
    zero_y = torch.zeros(model.N_TFs, model.N_LRs)
    V1_ = torch.where(x_i > 0, V1, zero_y)  # torch.Size([88, 63])
    K1_ = torch.where(x_i > 0, K1, zero_y)  # torch.Size([88, 63])
    tmp1 = torch.sum((V1_ * x_i) / ((K1_ + x_i) + (1e-12)), dim=1) * Y_i
    tmp2 = tmp1 * torch.exp(t_i)
    y_ode = (((y0 + tmp2) * t_i) / 2 + y0) * torch.exp(-t_i)
    return y_ode

def pre_velo(y_ode, z, model):
    regulate = model.regulate
    V2 = model.V2.detach()
    K2 = model.K2.detach()
    # y_ode = self.solve_ym(t)
    y_i = y_ode
    ym_ = regulate * y_i
    tmp1 = V2 * ym_
    tmp2 = (K2 + ym_) + (1e-12)
    tmp3 = torch.sum(tmp1 / tmp2, dim=1)
    dz_dt = tmp3 - z
    pre_velo = dz_dt
    return pre_velo

### calculate LRTG regulate matrix for batch training
def Jacobian_TFTG_batch(model,batch):
    TGs_expr, TFs_expr, TFLR_allscore = batch
    t = model.assign_latenttime(TGs_expr)[1]
    # t = model.assign_latenttime()[1]
    y_ode_ = model.solve_ym(t).detach()  # torch.Size([710, 3])
    y_ode = torch.tensor(y_ode_, requires_grad=True)

    jac = []
    for i in range(y_ode.shape[0]):
        y_ode_i = y_ode[i, :]
        z = TGs_expr[i, :]
        pre_velo_i = pre_velo(y_ode_i, z, model)
        dv_dy_list = [torch.autograd.grad(pre_velo_i[j], y_ode_i, retain_graph=True)[0] for j in
                      range(len(pre_velo_i))]
        dv_dy_i = torch.stack(dv_dy_list)
        jac.append(dv_dy_i)
    jac_tensor = torch.stack(jac, dim=0)  # torch.Size([710, 4, 3])
    regulate_mtx = torch.mean(jac_tensor, dim=0)
    return jac_tensor

def Jacobian_LRTF_batch(model,batch):
    TGs_expr, TFs_expr, TFLR_allscore = batch
    t = model.assign_latenttime(TGs_expr)[1]
    y0 = model.calculate_initial_y0()
    TFLR_allscore = TFLR_allscore
    TFs_expr = TFs_expr
    x = torch.tensor(TFLR_allscore, requires_grad=True)
    jac = []
    for i in range(x.shape[0]):
        # print(f"========================第{i}个cell===============================")
        t_i = t[i]
        x_i = x[i, :, :]
        Y_i = TFs_expr[i, :]
        y_ode_i = calculate_y_ode(t_i, y0, x_i, Y_i, model)
        dy_dx_list = [torch.autograd.grad(y_ode_i[j], x_i, retain_graph=True)[0] for j in
                      range(len(y_ode_i))]
        dy_dx_i = torch.stack(dy_dx_list)
        # print('1.the shape of dy_dx_i is:', dy_dx_i.shape)  # torch.Size([3, 3, 10])
        dy_dx_i = torch.sum(dy_dx_i, dim=1)
        # print('2.the shape of dy_dx_i is:', dy_dx_i.shape)  # torch.Size([3, 10])
        jac.append(dy_dx_i)
    jac_tensor = torch.stack(jac, dim=0)  # torch.Size([687, 3, 10])
    return jac_tensor

def Jacobian_TFTG(model,isbatch):
    TGs_expr = model.TGs_expr

    if isbatch:
       t = model.assign_latenttime(TGs_expr)[1]
    else:
        t = model.assign_latenttime()[1]

    y_ode_ = model.solve_ym(t).detach()  # torch.Size([710, 3])
    y_ode = torch.tensor(y_ode_, requires_grad=True)

    jac = []
    for i in range(y_ode.shape[0]):
        y_ode_i = y_ode[i, :]
        z = model.TGs_expr[i, :]
        pre_velo_i = pre_velo(y_ode_i, z, model)
        dv_dy_list = [torch.autograd.grad(pre_velo_i[j], y_ode_i, retain_graph=True)[0] for j in
                      range(len(pre_velo_i))]
        dv_dy_i = torch.stack(dv_dy_list)
        jac.append(dv_dy_i)
    jac_tensor = torch.stack(jac, dim=0)  # torch.Size([710, 4, 3])
    regulate_mtx = torch.mean(jac_tensor, dim=0)
    return jac_tensor

def Jacobian_LRTF(model,isbatch):
    TGs_expr = model.TGs_expr

    if isbatch:
        t = model.assign_latenttime(TGs_expr)[1]
    else:
        t = model.assign_latenttime()[1]
    y0 = model.calculate_initial_y0()
    TFLR_allscore = model.TFLR_allscore
    TFs_expr = model.TFs_expr
    x = torch.tensor(TFLR_allscore, requires_grad=True)
    jac = []
    for i in range(x.shape[0]):
        # print(f"========================第{i}个cell===============================")
        t_i = t[i]
        x_i = x[i, :, :]
        Y_i = TFs_expr[i, :]
        y_ode_i = calculate_y_ode(t_i, y0, x_i, Y_i, model)
        dy_dx_list = [torch.autograd.grad(y_ode_i[j], x_i, retain_graph=True)[0] for j in
                      range(len(y_ode_i))]
        dy_dx_i = torch.stack(dy_dx_list)
        # print('1.the shape of dy_dx_i is:', dy_dx_i.shape)  # torch.Size([3, 3, 10])
        dy_dx_i = torch.sum(dy_dx_i, dim=1)
        # print('2.the shape of dy_dx_i is:', dy_dx_i.shape)  # torch.Size([3, 10])
        jac.append(dy_dx_i)
    jac_tensor = torch.stack(jac, dim=0)  # torch.Size([687, 3, 10])
    return jac_tensor


def calculate_Jacobian(adata_velo,model,path):
    
    cell_num = adata_velo.shape[0]
    TG_num = adata_velo.shape[1]
    TF_num = adata_velo.obsm['TFLR_signaling_score'][0].shape[0]
    LR_num = adata_velo.obsm['TFLR_signaling_score'][0].shape[1]


     # calculate regulatory matrix between LR and TG
    all_jac_TFTG = torch.zeros(cell_num,TG_num,TF_num)
    all_jac_LRTF = torch.zeros(cell_num,TF_num,LR_num)
    all_jac_LRTG = torch.zeros(cell_num,TG_num,LR_num)
    all_matched_indices = []
    for batch_idx, batch in enumerate(model.dataloader):
        TGs_expr_batch, TFs_expr_batch, TFLR_allscore_batch = [x.to(device) for x in batch]
    
        jac_TFTG = Jacobian_TFTG_batch(model,batch)  # torch.Size([687, 4, 3])
        jac_LRTF = Jacobian_LRTF_batch(model,batch)  # torch.Size([687, 3, 10])
        jac_TFTG = jac_TFTG.float()  # 转换为 float32
        jac_LRTF = jac_LRTF.float()  # 转换为 float32
        jac_LRTG = torch.bmm(jac_TFTG, jac_LRTF)  # 批量矩阵相乘
    
        print('the shape of jac_TFTG:', jac_TFTG.shape)
        print('the shape of jac_LRTF:', jac_LRTF.shape)
        print('the shape of jac_LRTG:', jac_LRTG.shape)
    
        # 存储每行的匹配索引
        matched_indices = []
    
        for i, tf_batch_row in enumerate(TFs_expr_batch):
            # 逐行比较 TFs_expr_batch 的每一行
            match_found = False
            tf_batch_row = tf_batch_row.float()
            for j, tf_original_row in enumerate(model.TFs_expr):
                if torch.all(torch.eq(tf_batch_row, tf_original_row)):  
                    # print(f"================Row {i} in TFs_expr_batch matches with row {j} in TFs_expr.")
                    matched_indices.append((i, j))  
                    match_found = True
                    break
    
            if not match_found:
                print(f"Row {i} in TFs_expr_batch has no exact match in TFs_expr.")
    
            if match_found:
                all_jac_TFTG[j,:,:] = jac_TFTG[i,:,:]
                all_jac_LRTF[j,:,:] = jac_LRTF[i,:,:]
                all_jac_LRTG[j,:,:] = jac_LRTG[i,:,:]
    
        all_matched_indices.append(matched_indices)
    
    print('the shape of all_jac_TFTG:', all_jac_TFTG.shape)
    print('the shape of all_jac_LRTF:', all_jac_LRTF.shape)
    print('the shape of all_jac_LRTG:', all_jac_LRTG.shape)
    
    torch.save(all_jac_TFTG, os.path.join(path, "Jacobian_TFTG.pt"))
    torch.save(all_jac_LRTF, os.path.join(path, "Jacobian_LRTF.pt"))
    torch.save(all_jac_LRTG, os.path.join(path, "Jacobian_LRTG.pt"))
    torch.save(all_matched_indices, os.path.join(path, "all_matched_indices.pt"))

    return all_jac_LRTG
