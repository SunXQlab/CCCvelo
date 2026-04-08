import pandas as pd
import numpy as np
import os
import scanpy as sc
import scvelo as scv
import torch
import warnings
from scipy.spatial import distance_matrix

# CUDA support
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
warnings.filterwarnings("ignore")

def ReadData(count_file, imput_file, meta_file, loca_file):
    """
    create AnnData object.

    Parameters:
        count_file (str): raw expression data path.
        imput_file (str): imputation expression data path.
        meta_file (str): cell mate information data path.
        loca_file (str): spatial location data path.

    Returns:
        AnnData: adata.
    """
    df_count = pd.read_csv(count_file, index_col=0)
    df_imput = pd.read_csv(imput_file, index_col=0)
    df_meta = pd.read_csv(meta_file)
    df_loca = pd.read_csv(loca_file)

    adata = sc.AnnData(X=df_count.values.astype(np.float64))
    adata.obs_names = df_count.index
    adata.var_names = df_count.columns
    adata.obs['Cluster'] = df_meta['Cluster'].values
    adata.obsm['spatial'] = df_loca.values.astype(np.float64)
    adata.layers['Imputate'] = df_imput.values
    return adata

def PrepareInputData(adata,LR_link_file,TFTG_link_file,LRTF_score_file):

    LR_link = pd.read_csv(LR_link_file)  # LR link
    TFTG_link = pd.read_csv(TFTG_link_file)  # TFTG link

    Ligs = list(np.unique(LR_link['ligand'].values))
    Recs = list(np.unique(LR_link['receptor'].values))
    TFs = list(np.unique(TFTG_link['TF'].values))
    TGs = list(np.unique(TFTG_link['TG'].values))
    ccc_factors = np.unique(np.hstack((Ligs, Recs, TFs, TGs)))

    # print('the number of ligands is:', len(np.unique(LR_link['ligand'].values)))
    # print('the number of receptors is:', len(np.unique(LR_link['receptor'].values)))
    # print('the number of TFs is:', len(np.unique(TFTG_link['TF'].values)))
    # print('the number of TGs is:', len(np.unique(TFTG_link['TG'].values)))

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

def PrepareData(adata, hidden_dims):
    TFs_expr = torch.tensor(adata.layers['Imputate'][:, adata.var['TFs'].astype(bool)])
    TGs_expr = torch.tensor(adata.layers['Imputate'][:, adata.var['TGs'].astype(bool)])

    TGTF_regulate = torch.tensor(adata.varm['TGTF_regulate'])
    nonzero_idx = torch.nonzero(torch.sum(TGTF_regulate, dim=1)).squeeze()
    TGTF_regulate = TGTF_regulate[nonzero_idx].float()  # torch.Size([539, 114])

    TFLR_allscore = torch.tensor(adata.obsm['TFLR_signaling_score'])

    # adata = root_cell(adata, select_root)
    iroot = torch.tensor(adata.uns['iroot'])
    print('the root cell is:', adata.uns['iroot'])

    N_TGs = TGs_expr.shape[1]
    layers = hidden_dims
    layers.insert(0, N_TGs+1)  
    layers.append(N_TGs)  
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
        sc.pp.neighbors(adata_copy, n_pcs=50)
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

def calculate_groundTruth_velo(adata,GWnosie):  
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
        t = model.assign_latenttime(TGs_expr)[1]

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

def get_raw_velo(adata, model):

    N_TGs = model.N_TGs
    N_TFs = model.N_TFs
    N_cell = model.N_cell
    regulate = model.regulate
    TGs_expr = model.TGs_expr
    V1 = model.V1.detach()
    K1 = model.K1.detach()
    V2 = model.V2.detach()
    K2 = model.K2.detach()
    beta = model.beta.detach()
    gamma = model.gamma.detach()
    fit_t = model.assign_latenttime(TGs_expr)[1]
    y_ode = model.solve_ym(fit_t)
    zero_z = torch.zeros(N_TGs, N_TFs)
    V2_ = torch.where(regulate == 1, V2, zero_z)
    K2_ = torch.where(regulate == 1, K2, zero_z)
    velo_raw = torch.zeros((N_cell, N_TGs)).to(device)
    for i in range(N_cell):
        y_i = y_ode[i,:]
        ym_ = regulate * y_i
        tmp1 = V2_ * ym_
        tmp2 = (K2_ + ym_) + (1e-12)
        tmp3 = torch.sum(tmp1 / tmp2, dim=1)
        dz_dt = tmp3 - gamma*TGs_expr[i, :]
        velo_raw[i,:] = dz_dt

    velo_norm = (velo_raw - velo_raw.min()) / (velo_raw.max() - velo_raw.min() + 1e-6)

    adata_copy = adata.copy()
    adata_copy.uns["velo_para"] = {}
    adata_copy.uns["velo_para"]['fit_V1'] = V1.detach().numpy()
    adata_copy.uns["velo_para"]['fit_K1'] = K1.detach().numpy()
    adata_copy.uns["velo_para"]['fit_V2'] = V2.detach().numpy()
    adata_copy.uns["velo_para"]['fit_K2'] = K2.detach().numpy()
    adata_copy.obs['fit_t'] = fit_t.detach()
    adata_copy.layers['velo_raw'] = velo_raw.detach().numpy()
    adata_copy.layers['velo_norm'] = velo_norm.detach().numpy()
    adata_copy.layers['velocity'] = adata_copy.layers['velo_raw']

    return adata_copy

def get_raw_velo_v2(adata, model):

    N_TGs = model.N_TGs
    N_TFs = model.N_TFs
    N_cell = model.N_cell
    regulate = model.regulate
    TGs_expr = model.TGs_expr
    TGs_pred = model.net_f2()[0]
    V1 = model.V1.detach()
    K1 = model.K1.detach()
    V2 = model.V2.detach()
    K2 = model.K2.detach()
    fit_t = model.assign_latenttime()[1]
    y_ode = model.solve_ym(fit_t)
    print('the shape of y_ode is:', y_ode.shape)
    velo_raw = torch.zeros((N_cell, N_TGs)).to(device)
    for i in range(N_cell):
        y_i = y_ode[i,:]
        ym_ = regulate * y_i
        tmp1 = V2 * ym_
        tmp2 = (K2 + ym_) + (1e-6)
        tmp3 = torch.sum(tmp1 / tmp2, dim=1)
        dz_dt = tmp3 - TGs_expr[i, :]
        velo_raw[i,:] = dz_dt

    loss = torch.mean((TGs_expr - TGs_pred) ** 2,dim=0)
    velo_norm = (velo_raw - velo_raw.min()) / (velo_raw.max() - velo_raw.min() + 1e-6)

    adata_copy = adata.copy()
    lig = adata.var['ligand'].astype(bool)
    rec = adata.var['receptor'].astype(bool)
    tf = adata.var['TFs'].astype(bool)
    tg = adata.var['TGs'].astype(bool)
    combined_bool = lig | rec | tf | tg
    adata_copy = adata_copy[:, combined_bool]  # genes consist with ligand, receptor, TF, TG

    y_ode_ = torch.zeros((adata_copy.shape))
    tfs_mask = adata_copy.var['TFs'].astype(bool)
    tfs_index = tfs_mask[tfs_mask].index
    tfs_index = [adata_copy.var_names.get_loc(ind) for ind in tfs_index]  # Convert to integer indices

    for i, ind in enumerate(tfs_index):
        y_ode_[:, ind] = y_ode[:, i]

    # print('the old y_ode shape is:\n', y_ode.shape)
    # print('the new y_ode_ shape is:\n', y_ode_.shape)

    velo_raw_ = torch.zeros((adata_copy.shape))
    velo_norm_ = torch.zeros((adata_copy.shape))
    loss_ = torch.zeros((adata_copy.shape[1],))
    tgs_mask = adata_copy.var['TGs'].astype(bool)
    tgs_index = tgs_mask[tgs_mask].index
    tgs_index = [adata_copy.var_names.get_loc(ind) for ind in tgs_index]  # Convert to integer indices

    for i, ind in enumerate(tgs_index):
        velo_raw_[:, ind] = velo_raw[:, i]
        velo_norm_[:, ind] = velo_norm[:, i]
        loss_[ind] = loss[i]

    adata_copy.uns["velo_para"] = {}
    adata_copy.uns["velo_para"]['fit_V1'] = V1
    adata_copy.uns["velo_para"]['fit_K1'] = K1
    adata_copy.uns["velo_para"]['fit_V2'] = V2
    adata_copy.uns["velo_para"]['fit_K2'] = K2
    adata_copy.obs['fit_t'] = fit_t.detach()
    adata_copy.varm['loss'] = loss_.detach().numpy()
    adata_copy.layers['TFs_activity'] = y_ode_.detach().numpy()
    adata_copy.layers['velo_raw'] = velo_raw_.detach().numpy()
    adata_copy.layers['velo_norm'] = velo_norm_.detach().numpy()
    adata_copy.layers['velocity'] = adata_copy.layers['velo_raw']

    return adata_copy

