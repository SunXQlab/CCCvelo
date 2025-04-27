import os
import dfply
from dfply import *
import pickle
import pandas as pd
import numpy as np
import scanpy as sc
import anndata as ad
import json
import itertools
from scipy.stats import fisher_exact
from scipy.spatial.distance import pdist, squareform

from models.runMLnet import *
from models.preprocess_CCCvelo import *
from models.calculateLRscore import *

def get_index1(lst=None, item=''):
    return [index for (index, value) in enumerate(lst) if value == item]

def ReadData(count_file,imput_file,meta_file,loca_file):

    df_count = pd.read_csv(count_file)  # raw expression matrix
    df_imput = pd.read_csv(imput_file)  # imputated expression matrix
    df_meta = pd.read_csv(meta_file)  # meta data info
    df_loca = pd.read_csv(loca_file)  # cell location

    # creat AnnData object
    adata = sc.AnnData(X=df_count.values.astype(np.float64))  # 2515 × 9875
    adata.obs_names = df_count.index  # 设置观测名称
    adata.var_names = df_count.columns  # 设置变量名称
    adata.obs['Cluster'] = df_meta['Cluster'].values
    adata.obsm['spatial'] = df_loca.values.astype(np.float64)
    adata.layers['Imputate'] = df_imput.values

    return adata

# 1. load data from R
slide_num= 25
base_path = "E:/CCCvelo/apply_in_cortex/"
input_dir = os.path.join(base_path, f"Input/bin60_input_with_para{slide_num}")
files = {
    'count_file': 'raw_expression_mtx.csv',
    'imput_file':'imputation_expression_mtx.csv',
    'meta_file': 'cell_meta.csv',
    'loca_file': 'cell_location.csv',
}

paths = {key: os.path.join(input_dir, fname) for key, fname in files.items()}
adata = ReadData(**paths)  # n_obs × n_vars = 1976 × 14832
print(adata)

# 2. load candinate feature genes, candinate ligands and receptors (from R)
with open(os.path.join(input_dir, f"TGs_list.json"), "r") as f:
    TGs_list = json.load(f)
with open(os.path.join(input_dir, f"Ligs_list.json"), "r") as f:
    Ligs_list = json.load(f)
with open(os.path.join(input_dir, f"Recs_list.json"), "r") as f:
    Recs_list = json.load(f)

# print(TGs_list.keys())  # dict_keys(['Isocortex L23', 'Isocortex L4', ...])
# print(TGs_list['Isocortex L23'][:5])  # 输出前5个基因名

# 3. prepare runMLnet input
ExprMat = adata.X  # 注意：这里是原始表达矩阵
ExprMat = np.log1p(ExprMat) 
ExprMat = pd.DataFrame(ExprMat, index=adata.obs_names, columns=adata.var_names)

sub_anno = pd.DataFrame({
    "Barcode": adata.obs_names,                  # 等价于 rownames(neur_meta)
    "Cluster": adata.obs["Cluster"].values      # 等价于 neur_meta$scc_anno
})

outputDir = os.path.join(base_path, f"Output/bin60_input_with_para{slide_num}/RtoPy_test/")
create_directory(outputDir)

# 4. construct multilayer signaling network
resMLnet = runMLnet(ExprMat=ExprMat,AnnoMat=sub_anno,
         LigClus=None, RecClus=None,OutputDir=outputDir, Databases=None,RecTF_method = "Search", TFTG_method = "Search",
         TGList=TGs_list, LigList = Ligs_list, RecList = Recs_list)

ex_mulnetlist = {}
for receiver, sender_dict in resMLnet["mlnets"].items():
    for name, mlnet in sender_dict.items():
        if not mlnet["LigRec"].empty:
            ex_mulnetlist[name] = mlnet
summary_MLnetnode = summarize_multinet(ex_mulnetlist)
print(summary_MLnetnode)

# 5. calculate LR-TF signaling strength
ExprMat_Impute = adata.layers['Imputate']
ExprMat_Impute = pd.DataFrame(ExprMat_Impute.T, index=adata.var_names, columns= adata.obs_names)

coords = adata.obsm['spatial']
DistMat = squareform(pdist(coords, metric="euclidean"))
DistMat = pd.DataFrame(DistMat, index=adata.obs_names, columns=adata.obs_names)
np.fill_diagonal(DistMat.values, 1)

wd_model = os.path.join(outputDir, 'runModel')
create_directory(wd_model)
print(f"saveDir: {wd_model}")

loop_calculate_LRTF_allscore(
    exprMat=ExprMat_Impute,
    distMat=DistMat,
    annoMat=sub_anno,
    ex_mulnetlist=ex_mulnetlist,
    neuronal_ct=['Isocortex L4', 'Isocortex L6', 'Isocortex L23', 'Isocortex L5'],
    wd_model=wd_model
)

# 6. calculate TF-LR score of each cell and save as csv file
files = os.listdir(wd_model)
files_tf = [f for f in files if "LRTF" in f]
TFLR_all_score = get_TFLR_allactivity(
    mulNetList=ex_mulnetlist,
    LRTF_score_files=files_tf,
    wd_model=wd_model
)

save_LRscore_and_MLnet(
    adata,
    mulNetList=ex_mulnetlist,
    TFLR_all_score=TFLR_all_score, 
    save_path=outputDir)
