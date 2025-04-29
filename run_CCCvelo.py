import os
import time
import random
import psutil
import pickle
import json
import torch
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.spatial.distance import pdist, squareform

from models.runMLnet import *
from models.preprocess_CCCvelo import *
from models.calculateLRscore import *

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_index1(lst, item):
    return [idx for idx, val in enumerate(lst) if val == item]

def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def ReadData(count_file, imput_file, meta_file, loca_file):
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

def main(
    seed,
    base_path="E:/CCCvelo/apply_in_prostate/",
    project_name="humanprostate_test",
    rec_clusters=None,
    hidden_dims=[200, 200, 200],
    batch_size=1500,
    learning_rate=0.001,
    lambda_reg=0.01,
    n_epochs=20
):
    # if rec_clusters is None:
    #     rec_clusters = ['E.state tumor', 'ICS.state tumor', 'M.state tumor']

    # set input and output path
    input_dir = os.path.join(base_path, "Input", project_name)
    output_dir = os.path.join(base_path, "Output", project_name)
    create_directory(output_dir)

    # step1: load raw dataset
    print("Loading data...")
    data_files = {
        'count_file': 'raw_expression_mtx.csv',
        'imput_file': 'imputation_expression_mtx.csv',
        'meta_file': 'cell_meta.csv',
        'loca_file': 'cell_location.csv'
    }
    paths = {key: os.path.join(input_dir, fname) for key, fname in data_files.items()}
    adata = ReadData(**paths)

    # step2: load prior database and candidate ligands, receptors, and feature genes
    print("Loading database...")
    Databases = load_json(os.path.join(input_dir, "Databases.json"))
    TGs_list = load_json(os.path.join(input_dir, "TGs_list.json"))
    Ligs_list = load_json(os.path.join(input_dir, "Ligs_list.json"))
    Recs_list = load_json(os.path.join(input_dir, "Recs_list.json"))

    # step3: construct multilayer signaing network 
    ExprMat = pd.DataFrame(np.log1p(adata.X), index=adata.obs_names, columns=adata.var_names)
    sub_anno = pd.DataFrame({
        "Barcode": adata.obs_names,
        "Cluster": adata.obs["Cluster"].values
    })

    print("Building multilayer network...")
    resMLnet = runMLnet(
        ExprMat=ExprMat,
        AnnoMat=sub_anno,
        LigClus=None,
        RecClus=rec_clusters,
        OutputDir=output_dir,
        Databases=None,
        RecTF_method="Search",
        TFTG_method="Search",
        TGList=TGs_list,
        LigList=Ligs_list,
        RecList=Recs_list
    )

    ex_mulnetlist = {
        name: mlnet
        for receiver, sender_dict in resMLnet["mlnets"].items()
        for name, mlnet in sender_dict.items()
        if not mlnet["LigRec"].empty
    }
    print("Multilayer network nodes summary:")
    print(summarize_multinet(ex_mulnetlist))

    # step4: calclulate LR signaling strength
    ExprMat_Impute = pd.DataFrame(
        adata.layers['Imputate'].T,
        index=adata.var_names,
        columns=adata.obs_names
    )
    coords = adata.obsm['spatial']
    DistMat = pd.DataFrame(
        squareform(pdist(coords, metric="euclidean")),
        index=adata.obs_names,
        columns=adata.obs_names
    )
    np.fill_diagonal(DistMat.values, 1)

    wd_model = os.path.join(output_dir, 'runModel')
    create_directory(wd_model)

    print(f"Saving model intermediate results to {wd_model}")
    loop_calculate_LRTF_allscore(
        exprMat=ExprMat_Impute,
        distMat=DistMat,
        annoMat=sub_anno,
        ex_mulnetlist=ex_mulnetlist,
        neuronal_ct=rec_clusters,
        wd_model=wd_model
    )

    # save LR signaling strength for each cells
    files_tf = [f for f in os.listdir(wd_model) if "LRTF" in f]
    TFLR_all_score = get_TFLR_allactivity(
        mulNetList=ex_mulnetlist,
        LRTF_score_files=files_tf,
        wd_model=wd_model
    )

    save_LRscore_and_MLnet(
        adata,
        mulNetList=ex_mulnetlist,
        TFLR_all_score=TFLR_all_score,
        save_path=output_dir
    )

    with open(os.path.join(output_dir, 'TFLR_all_score.pkl'), 'wb') as f:
        pickle.dump(TFLR_all_score, f)

    # step5: select receiver cells
    print("Selecting receiver cells...")
    celltype_ls = adata.obs['Cluster'].to_list()
    ct_index_ls = []
    for name in rec_clusters:
        ct_index_ls.extend(get_index1(celltype_ls, name))

    adata = adata[ct_index_ls, :].copy()

    results_path = os.path.join(output_dir, "Output")
    create_directory(results_path)

    # step6: prepare the input of CCCvelo
    link_files = {
        'LR_link_file': 'LR_links.csv',
        'TFTG_link_file': 'TFTG_links.csv',
        'LRTF_score_file': 'TFLR_score/'
    }
    paths = {key: os.path.join(output_dir, fname) for key, fname in link_files.items()}
    print('Loading link files from:', paths)

    adata = PrepareInputData(adata, **paths)
    adata.uns['Cluster_colors'] = ["#DAA0B0", "#908899", "#9D5A38"]

    torch.save(adata, os.path.join(results_path, "pp_adata.pt"))

    adata = root_cell(adata, select_root='UMAP')
    print('Root cell cluster is:', adata.obs['Cluster'][adata.uns['iroot']])

    # step7: trainging CCCvelo model
    print("Training CCCvelo model...")

    n_cells = adata.n_obs
    print(f"Number of receiver cells: {n_cells}")

    if n_cells <= 10000:
        print("Training with standard SpatialVelocity (full batch)...")
        
        from models.train_CCCvelo import SpatialVelocity

        data = PrepareData(adata, hidden_dims=hidden_dims)
        model = SpatialVelocity(*data, lr=learning_rate, Lambda=lambda_reg)
        iteration_adam, loss_adam = model.train(n_epochs)

        plt_path = os.path.join(results_path, "figure/")
        create_directory(plt_path)
        plot_gene_dynamic(adata_velo, model, plt_path)
    
    else:
        print("Training with batch SpatialVelocity (mini-batch mode)...")

        from models.train_CCCvelo_batchs import SpatialVelocity

        data = PrepareData(adata, hidden_dims=hidden_dims)
        model = SpatialVelocity(*data, lr=learning_rate, Lambda=lambda_reg, batch_size=batch_size)
        iteration_adam, loss_adam = model.train(n_epochs)

        plt_path = os.path.join(results_path, "figure/")
        create_directory(plt_path)
        plot_gene_dynamic(adata_velo, model, plt_path)

    adata.write_h5ad(os.path.join(output_dir, 'adata_pyinput.h5ad'))

    adata_copy = adata[:, adata.var['TGs'].astype(bool)]
    adata_velo = get_raw_velo(adata_copy, model)

    save_model_and_data(model, adata_velo, results_path)

    print("Pipeline finished successfully!")

# Running
if __name__ == "__main__":

    seed = 1 # Replace with your seed value

    torch.manual_seed(seed)  
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    random.seed(seed) 
    np.random.seed(seed)  
    start_time = time.time()
    process = psutil.Process(os.getpid())
    before_memory = process.memory_info().rss / 1024 ** 2  

    main(
        seed,
        base_path="E:/CCCvelo/apply_in_prostate/",
        project_name="humanprostate_test",
        rec_clusters=['E.state tumor', 'ICS.state tumor', 'M.state tumor'],
        hidden_dims=[200, 200, 200],
        batch_size=1500,
        learning_rate=0.001,
        lambda_reg=0.01,
        n_epochs=5)
    
    after_memory = process.memory_info().rss / 1024 ** 2  
    print(f"Memory usage is: {after_memory - before_memory} MB")
    end_time = time.time()
    run_time = (end_time - start_time) / 60
    print(f"Running time is: {run_time} mins")




