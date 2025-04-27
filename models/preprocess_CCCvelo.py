import pandas as pd
import numpy as np
import os
import scanpy as sc
import scvelo as scv
import torch

def PrepareInputData(adata,LR_link_file,TFTG_link_file,LRTF_score_file):

    LR_link = pd.read_csv(LR_link_file)  # LR link
    TFTG_link = pd.read_csv(TFTG_link_file)  # TFTG link

    Ligs = list(np.unique(LR_link['ligand'].values))
    Recs = list(np.unique(LR_link['receptor'].values))
    TFs = list(np.unique(TFTG_link['TF'].values))
    TGs = list(np.unique(TFTG_link['TG'].values))
    ccc_factors = np.unique(np.hstack((Ligs, Recs, TFs, TGs)))

    print('the number of ligands is:', len(np.unique(LR_link['ligand'].values)))
    print('the number of receptors is:', len(np.unique(LR_link['receptor'].values)))
    print('the number of TFs is:', len(np.unique(TFTG_link['TF'].values)))
    print('the number of TGs is:', len(np.unique(TFTG_link['TG'].values)))

    n_gene = adata.shape[1]
    adata.var['ligand'] = np.full(n_gene, False, dtype=bool).astype(int)
    adata.var['receptor'] = np.full(n_gene, False, dtype=bool).astype(int)
    adata.var['TFs'] = np.full(n_gene, False, dtype=bool).astype(int)
    adata.var['TGs'] = np.full(n_gene, False, dtype=bool).astype(int)

    for gene in list(adata.var_names):
        if gene in Ligs:
            adata.var['ligand'][gene] = 1
        if gene in Recs:
            adata.var['receptor'][gene] = 1
        if gene in TFs:
            adata.var['TFs'][gene] = 1
        if gene in TGs:
            adata.var['TGs'][gene] = 1

    adata.varm['TGTF_pair'] = np.full([n_gene, len(TFs)], 'blank')
    adata.varm['TGTF_regulate'] = np.full([n_gene, len(TFs)], 0)
    gene_names = list(adata.var_names)
    for target in TGs:
        if target in gene_names:
            target_idx = gene_names.index(target)
            df_tf_idx = np.where(TFTG_link['TG'].values == target)[0]
            tf_name = list(TFTG_link['TF'].values[df_tf_idx])
            tf_idx = [index for index, element in enumerate(TFs) if element in tf_name]

            for item1, item2 in zip(tf_idx, tf_name):
                adata.varm['TGTF_pair'][target_idx][item1] = item2
                adata.varm['TGTF_regulate'][target_idx][item1] = 1

    # add TFLR score
    folder_path = LRTF_score_file
    file_names = os.listdir(folder_path)
    obs_names = adata.obs_names
    TFLR_allscore = []
    for i in obs_names:
        obs_name = i + "_"
        index = [index for index, name in enumerate(file_names) if obs_name in name]

        if not index:  # Handle case where no files match
            print(f"Error: No file found matching {obs_name}")
            continue

        file_name = file_names[index[0]]
        data_tmp = pd.read_csv(folder_path + file_name)
        LR_pair = data_tmp.columns.tolist()
        # print('the LR_pair is:\n', LR_pair)
        data = data_tmp.values
        # print('the data is:\n', data)
        # print('the raw LR signaling score is:\n', data)
        TFLR_allscore.append(data)
    TFLR_allscore = np.array(TFLR_allscore)  # (2515, 44, 20)
    adata.obsm['TFLR_signaling_score'] = TFLR_allscore

    # Normalization
    if adata.shape[1]<3000:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    else:
        sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=3000)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

        for factor in ccc_factors:
            if factor in adata.var.index:
                if not adata.var['highly_variable'][factor]:
                    adata.var['highly_variable'][factor] = True

    sc.tl.pca(adata, svd_solver="arpack")
    # sc.pp.neighbors(adata, n_pcs=50)
    scv.pp.neighbors(adata)
    sc.tl.umap(adata)

    return adata



def PrerocessRealData(count_file,imput_file,meta_file,loca_file,LR_link_file,TFTG_link_file,LRTF_score_file,using_low_emdding):

    df_count = pd.read_csv(count_file)  # raw expression matrix
    df_imput = pd.read_csv(imput_file)  # imputated expression matrix
    df_meta = pd.read_csv(meta_file)  # meta data info
    df_loca = pd.read_csv(loca_file)  # cell location
    LR_link = pd.read_csv(LR_link_file)  # LR link
    TFTG_link = pd.read_csv(TFTG_link_file)  # TFTG link

    # creat AnnData object
    adata = sc.AnnData(X=df_count.values.astype(np.float64))  # 2515 × 9875
    adata.obs_names = df_count.index  # 设置观测名称
    adata.var_names = df_count.columns  # 设置变量名称
    adata.obs['Cluster'] = df_meta['Cluster'].values
    adata.obsm['spatial'] = df_loca.values.astype(np.float64)
    adata.layers['Imputate'] = df_imput.values

    Ligs = list(np.unique(LR_link['ligand'].values))
    Recs = list(np.unique(LR_link['receptor'].values))
    TFs = list(np.unique(TFTG_link['TF'].values))
    TGs = list(np.unique(TFTG_link['TG'].values))
    ccc_factors = np.unique(np.hstack((Ligs, Recs, TFs, TGs)))

    print('the number of ligands is:', len(np.unique(LR_link['ligand'].values)))
    print('the number of receptors is:', len(np.unique(LR_link['receptor'].values)))
    print('the number of TFs is:', len(np.unique(TFTG_link['TF'].values)))
    print('the number of TGs is:', len(np.unique(TFTG_link['TG'].values)))

    n_gene = adata.shape[1]
    adata.var['ligand'] = np.full(n_gene, False, dtype=bool).astype(int)
    adata.var['receptor'] = np.full(n_gene, False, dtype=bool).astype(int)
    adata.var['TFs'] = np.full(n_gene, False, dtype=bool).astype(int)
    adata.var['TGs'] = np.full(n_gene, False, dtype=bool).astype(int)

    for gene in list(adata.var_names):
        if gene in Ligs:
            adata.var['ligand'][gene] = 1
        if gene in Recs:
            adata.var['receptor'][gene] = 1
        if gene in TFs:
            adata.var['TFs'][gene] = 1
        if gene in TGs:
            adata.var['TGs'][gene] = 1

    adata.varm['TGTF_pair'] = np.full([n_gene, len(TFs)], 'blank')
    adata.varm['TGTF_regulate'] = np.full([n_gene, len(TFs)], 0)
    gene_names = list(adata.var_names)
    for target in TGs:
        if target in gene_names:
            target_idx = gene_names.index(target)
            df_tf_idx = np.where(TFTG_link['TG'].values == target)[0]
            tf_name = list(TFTG_link['TF'].values[df_tf_idx])
            tf_idx = [index for index, element in enumerate(TFs) if element in tf_name]

            for item1, item2 in zip(tf_idx, tf_name):
                adata.varm['TGTF_pair'][target_idx][item1] = item2
                adata.varm['TGTF_regulate'][target_idx][item1] = 1

    # add TFLR score
    folder_path = LRTF_score_file
    file_names = os.listdir(folder_path)
    obs_names = adata.obs_names
    TFLR_allscore = []
    for i in obs_names:
        obs_name = i + "_"
        index = [index for index, name in enumerate(file_names) if obs_name in name]

        if not index:  # Handle case where no files match
            print(f"Error: No file found matching {obs_name}")
            continue

        file_name = file_names[index[0]]
        data_tmp = pd.read_csv(folder_path + file_name)
        LR_pair = data_tmp.columns.tolist()
        # print('the LR_pair is:\n', LR_pair)
        data = data_tmp.values
        # print('the data is:\n', data)
        # print('the raw LR signaling score is:\n', data)
        TFLR_allscore.append(data)
    TFLR_allscore = np.array(TFLR_allscore)  # (2515, 44, 20)
    adata.obsm['TFLR_signaling_score'] = TFLR_allscore

    # Normalization
    if adata.shape[1]<3000:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    else:
        sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=3000)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

        for factor in ccc_factors:
            if factor in adata.var.index:
                if not adata.var['highly_variable'][factor]:
                    adata.var['highly_variable'][factor] = True

    if using_low_emdding:
        # Constructing the spatial network
        # STAGATE.Cal_Spatial_Net(adata, rad_cutoff=150)
        # STAGATE.Stats_Spatial_Net(adata)
        # adata = STAGATE.train_STAGATE(adata, alpha=0)

        sc.pp.neighbors(adata, use_rep='STAGATE')
        sc.tl.umap(adata)
    else:
        sc.tl.pca(adata, svd_solver="arpack")
        # sc.pp.neighbors(adata, n_pcs=50)
        scv.pp.neighbors(adata)
        sc.tl.umap(adata)

    return adata

def PrerocessData(count_file, meta_file, loca_file, LR_link_file, TFTG_link_file, LRTF_score_file, TF_activity_file,LRTF_para_file,TFTG_para_file,
                  using_low_emdding):
    df_count = pd.read_csv(count_file)  # raw expression matrix
    df_meta = pd.read_csv(meta_file)  # meta data info
    df_loca = pd.read_csv(loca_file)  # cell location
    LR_link = pd.read_csv(LR_link_file)  # LR link
    TFTG_link = pd.read_csv(TFTG_link_file)  # TFTG link
    TF_activity = pd.read_csv(TF_activity_file)  # TFTG link
    LRTF_paras = pd.read_csv(LRTF_para_file).values  # LRTF parameters (ground truth)
    TFTG_paras = pd.read_csv(TFTG_para_file).values # TFTG parameters (ground truth)

    # creat AnnData object
    adata = sc.AnnData(X=df_count.values.T)
    print(adata)
    adata.obs_names = df_count.columns  # 设置观测名称
    adata.var_names = df_count.index  # 设置观测名称
    adata.obs['groundTruth_psd'] = df_meta['pseudotime'].values
    adata.obsm['spatial'] = df_loca.values.astype(np.float64)
    adata.obsm['groundTruth_TF_activity'] = TF_activity.values.T
    adata.layers['Imputate'] = df_count.values.T

    cell_total_counts = adata.X[:,10:].sum(axis=1)
    non_zero_cells = np.where(cell_total_counts != 0)[0]
    adata = adata[non_zero_cells]

    adata.uns["ground_truth_para"] = {}
    adata.uns["ground_truth_para"]['gd_V1'] = LRTF_paras[:10].T
    adata.uns["ground_truth_para"]['gd_K1'] = LRTF_paras[10:20].T
    adata.uns["ground_truth_para"]['gd_beta'] = LRTF_paras[20].T
    adata.uns["ground_truth_para"]['gd_V2'] = TFTG_paras[:3].T
    adata.uns["ground_truth_para"]['gd_K2'] = TFTG_paras[3:6].T
    adata.uns["ground_truth_para"]['gd_gamma'] = TFTG_paras[6].T

    Ligs = list(np.unique(LR_link['ligand'].values))
    Recs = list(np.unique(LR_link['receptor'].values))
    TFs = list(np.unique(TFTG_link['TF'].values))
    TGs = list(np.unique(TFTG_link['TG'].values))
    ccc_factors = np.unique(np.hstack((Ligs, Recs, TFs, TGs)))

    n_gene = adata.shape[1]
    adata.var['ligand'] = np.full(n_gene, False, dtype=bool).astype(int)
    adata.var['receptor'] = np.full(n_gene, False, dtype=bool).astype(int)
    adata.var['TFs'] = np.full(n_gene, False, dtype=bool).astype(int)
    adata.var['TGs'] = np.full(n_gene, False, dtype=bool).astype(int)

    for gene in list(adata.var_names):
        if gene in Ligs:
            adata.var['ligand'][gene] = 1
        if gene in Recs:
            adata.var['receptor'][gene] = 1
        if gene in TFs:
            adata.var['TFs'][gene] = 1
        if gene in TGs:
            adata.var['TGs'][gene] = 1

    adata.varm['TGTF_pair'] = np.full([n_gene, len(TFs)], 'blank')
    adata.varm['TGTF_regulate'] = np.full([n_gene, len(TFs)], 0)
    gene_names = list(adata.var_names)
    for target in TGs:
        if target in gene_names:
            target_idx = gene_names.index(target)
            df_tf_idx = np.where(TFTG_link['TG'].values == target)[0]
            tf_name = list(TFTG_link['TF'].values[df_tf_idx])
            tf_idx = [index for index, element in enumerate(TFs) if element in tf_name]

            for item1, item2 in zip(tf_idx, tf_name):
                adata.varm['TGTF_pair'][target_idx][item1] = item2
                adata.varm['TGTF_regulate'][target_idx][item1] = 1

    # add TFLR score
    folder_path = LRTF_score_file
    file_names = os.listdir(folder_path)
    obs_names = adata.obs_names
    TFLR_allscore = []
    for i in obs_names:
        obs_name = i + "_"
        index = [index for index, name in enumerate(file_names) if obs_name in name]
        file_name = file_names[index[0]]
        data = pd.read_csv(folder_path + file_name).values
        # data = data.astype(np.float32)
        # print('the raw LR signaling score is:\n', data)
        # normalize
        # min_val = np.min(data)
        # max_val = np.max(data)
        # norm_data = (data - min_val) / (max_val - min_val) * (1 - 0) + 0
        # TFLR_allscore.append(norm_data)
        TFLR_allscore.append(data)
    TFLR_allscore = np.array(TFLR_allscore)  # (2515, 44, 20)
    adata.obsm['TFLR_signaling_score'] = TFLR_allscore

    # Normalization
    sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=3000)
    # sc.pp.normalize_total(adata, target_sum=1e4)
    # sc.pp.log1p(adata)

    for factor in ccc_factors:
        if factor in adata.var.index:
            if not adata.var['highly_variable'][factor]:
                adata.var['highly_variable'][factor] = True

    if using_low_emdding:
        # Constructing the spatial network
        # STAGATE.Cal_Spatial_Net(adata, rad_cutoff=150)
        # STAGATE.Stats_Spatial_Net(adata)
        # adata = STAGATE.train_STAGATE(adata, alpha=0)

        sc.pp.neighbors(adata, use_rep='STAGATE')
        sc.tl.umap(adata)
    else:
        sc.tl.pca(adata, svd_solver="arpack")
        # sc.pp.neighbors(adata, n_pcs=50)
        scv.pp.neighbors(adata)
        sc.tl.umap(adata)

    return adata

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory created at: {path}")

def save_model_and_data(model, data, path):
    torch.save(model, os.path.join(path, "model_spa_velo.pth"))
    torch.save(data, os.path.join(path, "CCCvelo.pt"))
    # print(f"Model and data saved at: {path}")

def calculate_groundTruth_velo(adata,GWnosie):  # 给定模型中的参数和TF活性，根据模型的公式计算的velocity
    adata = adata[:, adata.var['TGs'].astype(bool)]
    regulate = adata.varm['TGTF_regulate']
    y_ode = adata.obsm['groundTruth_TF_activity']
    TGs_expr = adata.layers['Imputate'][:, adata.var['TGs'].astype(bool)]
    N_cell, N_TGs = TGs_expr.shape
    V2 = adata.uns["ground_truth_para"]['gd_V2']
    K2 = adata.uns["ground_truth_para"]['gd_K2']
    gamma = adata.uns["ground_truth_para"]['gd_gamma']
    gd_velo = []
    for i in range(N_cell):
        y_i = y_ode[i, :]
        ym_ = regulate * y_i
        tmp1 = V2 * ym_
        tmp2 = (K2 + ym_) + (1e-12)
        tmp3 = np.sum(tmp1 / tmp2, axis=1)
        dz_dt = tmp3 - gamma * TGs_expr[i, :]
        gd_velo.append(dz_dt)
    gd_velo = np.array(gd_velo)
    # add GWnoise
    gd_velo = gd_velo+GWnosie
    # print('the groundTruth_velo is:\n',gd_velo.shape)
    adata.layers['groundTruth_velo'] = gd_velo
    return adata

def calculate_groundTruth_velo_v2(adata):  # gene表达量关于时间的导数，中心差分法

    adata = adata[:, adata.var['TGs'].astype(bool)]
    TGs_expr = adata.layers['Imputate']
    gdt_psd = adata.obs['groundTruth_psd']
    dTG_dt = []
    for i in range(TGs_expr.shape[1]):
        TG_expr_i = np.array(TGs_expr[:, i])
        dTG_dt_i = np.gradient(TG_expr_i, gdt_psd)
        # print('the shape of dTG_dt_i is:', dTG_dt_i.shape)
        dTG_dt.append(dTG_dt_i)
    dTG_dt = np.array(dTG_dt).T
    adata.layers['groundTruth_velo_with_dG_dt'] = dTG_dt

    return adata

# def calculate_pseudo_velocity(adata, model):
#     N_cell, N_TGs = adata.shape
#     regulate = model.regulate
#     TGs_expr = model.TGs_expr
#     V2 = model.V2.detach()
#     K2 = model.K2.detach()
#     y_gdt = adata.obsm['groundTruth_TF_activity']
#     velo_raw = torch.zeros((N_cell, N_TGs)).to(device)
#     for i in range(N_cell):
#         y_i = y_gdt[i, :]
#         ym_ = regulate * y_i
#         tmp1 = V2 * ym_
#         tmp2 = (K2 + ym_) + (1e-12)
#         tmp3 = torch.sum(tmp1 / tmp2, dim=1)
#         dz_dt = tmp3 - TGs_expr[i, :]
#         velo_raw[i, :] = dz_dt
#
#     adata_copy = adata.copy()
#     adata_copy.layers['pseudo_velocity'] = velo_raw.detach().numpy()
#     return adata_copy

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

def calclulate_TFactivity(model,batch):
    TGs_expr, TFs_expr, TFLR_allscore = batch
    N_TGs = TGs_expr.size(1)
    N_TFs = TFs_expr.size(1)
    t = model.assign_latenttime(TGs_expr)[1]
    # t = model.assign_latenttime()[1]
    y_ode = model.solve_ym(t).detach()  # torch.Size([710, 3])

    V2 = model.V2.detach()
    K2 = model.K2.detach()
    regulate = model.regulate
    print('the shape of regulate is:', regulate.size())
    zero_z = torch.zeros(N_TGs, N_TFs)
    V2_ = torch.where(regulate == 1, V2, zero_z)
    K2_ = torch.where(regulate == 1, K2, zero_z)
    tmp1 = V2_.unsqueeze(0) * y_ode.unsqueeze(1)
    tmp2 = (K2_.unsqueeze(0) + y_ode.unsqueeze(1)) + (1e-12)
    tmp3 = torch.sum(tmp1 / tmp2, dim=2)

    return y_ode,tmp3

def calclulate_TFactivity_v0(model,isbatch):
    TGs_expr = model.TGs_expr
    TFs_expr = model.TFs_expr
    N_TGs = TGs_expr.size(1)
    N_TFs = TFs_expr.size(1)

    if isbatch:
       t = model.assign_latenttime(TGs_expr)[1]
    else:
        t = model.assign_latenttime()[1]

    y_ode = model.solve_ym(t).detach()  # torch.Size([710, 3])

    V2 = model.V2.detach()
    K2 = model.K2.detach()
    regulate = model.regulate
    print('the shape of regulate is:', regulate.size())
    zero_z = torch.zeros(N_TGs, N_TFs)
    V2_ = torch.where(regulate == 1, V2, zero_z)
    K2_ = torch.where(regulate == 1, K2, zero_z)
    tmp1 = V2_.unsqueeze(0) * y_ode.unsqueeze(1)
    tmp2 = (K2_.unsqueeze(0) + y_ode.unsqueeze(1)) + (1e-12)
    tmp3 = torch.sum(tmp1 / tmp2, dim=2)

    return y_ode,tmp3


# def calculate_initial_y0_(model):
#     # calculate initial y0
#     V1 = model.V1
#     K1 = model.K1
#     iroot = model.iroot
#     TFLR_allscore = model.TFLR_allscore
#     TFs_expr = model.TFs_expr
#     # calculate initial y0
#     x0 = TFLR_allscore[iroot,:,:]
#     Y0 = TFs_expr[iroot,:]
#     zero_y = torch.zeros(model.N_TFs, model.N_LRs).float()
#     V1_ = torch.where(x0 > 0, V1, zero_y)  # torch.Size([10, 88, 63])
#     K1_ = torch.where(x0 > 0, K1, zero_y)  # torch.Size([10, 88, 63])
#     y0 = torch.sum((V1_ * x0) / ((K1_ + x0) + (1e-12)),dim=1) * Y0  # torch.Size([10, 88])
#     return y0
#
# def hill_fun_(y0, x_i, TFs_expr_i,t_i, model):  # trapezoidal rule approximation
#     K1 = model.K1.detach()
#     V1 = model.V1.detach()
#     # TFLR_allscore = self.TFLR_allscore
#     Y_i = TFs_expr_i
#     zero_y = torch.zeros(model.N_TFs, model.N_LRs)
#     V1_ = torch.where(x_i > 0, V1, zero_y)  # torch.Size([88, 63])
#     K1_ = torch.where(x_i > 0, K1, zero_y)  # torch.Size([88, 63])
#     tmp1 = torch.sum((V1_ * x_i) / ((K1_ + x_i) + (1e-12)), dim=1) * Y_i
#     tmp2 = tmp1 * torch.exp(t_i)
#     y_i = (((y0 + tmp2)*t_i)/2 + y0) * torch.exp(-t_i)
#     return y_i
#
# def solve_ym_(t_i,x_i, TFs_expr_i,model):
#     y0_ = calculate_initial_y0_(model)
#     # N_cell = model.N_cell
#     # N_TFs = model.N_TFs
#     y_ode_i = hill_fun_(y0_,x_i,TFs_expr_i,t_i,model)
#     return y_ode_i
#
# def pre_velo_new(t_i, x_i, TFs_expr_i, z_i, model):
#
#     V2 = model.V2.detach()
#     K2 = model.K2.detach()
#     regulate = model.regulate
#     y_ode_i = solve_ym_(t_i, x_i, TFs_expr_i,model)
#     y_ode_i = torch.tensor(y_ode_i, requires_grad=True)
#     ym_ = regulate * y_ode_i
#     tmp1 = V2 * ym_
#     tmp2 = (K2 + ym_) + (1e-12)
#     tmp3 = torch.sum(tmp1 / tmp2, dim=1)
#     dz_dt = tmp3 - z_i
#     pre_velo = dz_dt
#     return pre_velo

# t = model.assign_latenttime()[1]
# TFLR_allscore = model.TFLR_allscore
# TFs_expr = model.TFs_expr
# TGs_expr = model.TGs_expr
# x = torch.tensor(TFLR_allscore, requires_grad=True)
# jac = []
# for i in range(x.shape[0]):
#     t_i = t[i]
#     x_i = x[i, :, :]
#     TFs_expr_i = TFs_expr[i, :]
#     z_i = TGs_expr[i, :]
#     pre_velo_i = pre_velo_new(t_i, x_i, TFs_expr_i, z_i, model)
#
#     for j in range(len(pre_velo_i)):
#         # Compute the gradient of each element in pre_velo_i with respect to x_i
#         pre_velo_ij = pre_velo_i[j]
#         pre_velo_ij.backward(retain_graph=True)  # Retain the graph for subsequent backward passes
#
#         gradient_xij = x_i.grad.clone()  # Clone the gradient to save it
#         print('The gradient_xij is:\n', gradient_xij)
#
#         x_i.grad.zero_()  # Clear gradients for the next iteration


# def Jacobian_LRTG(model):
#     t = model.assign_latenttime()[1]
#     TFLR_allscore = model.TFLR_allscore
#     TFs_expr = model.TFs_expr
#     TGs_expr = model.TGs_expr
#     x = torch.tensor(TFLR_allscore, requires_grad=True)
#
#     jac = []
#     for i in range(x.shape[0]):
#         t_i = t[i]
#         x_i = x[i,:,:]
#         TFs_expr_i = TFs_expr[i, :]
#         z_i = TGs_expr[i, :]
#         pre_velo_i = pre_velo_new(t_i, x_i, TFs_expr_i, z_i, model)
#         dv_dx_list = [torch.autograd.grad(pre_velo_i[j], x_i, retain_graph=True)[0] for j in
#                       range(len(pre_velo_i))]
#         dv_dx_i = torch.stack(dv_dx_list)
#         jac.append(dv_dx_i)
#     jac_tensor = torch.stack(jac, dim=0)  # torch.Size([710, 4, 3])
#     regulate_mtx = torch.mean(jac_tensor, dim=0)
#     return regulate_mtx






