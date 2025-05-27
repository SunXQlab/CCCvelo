import os
import re
import numpy as np
import pandas as pd
import pickle
from collections import defaultdict
from scipy.spatial.distance import pdist, squareform
from models2.utils import create_directory 
import scipy.sparse as sp
from scipy.spatial import Delaunay
from anndata import AnnData

def build_DT_neighbors(adata, r_eps_real = 200, scale_factor = 0.73):
    """
    Construct an adjacency graph based on Delaunay Triangulation and store it in adata.obsp["dt_connectivities"]

    Parameters:
    - adata (AnnData): AnnData object, 
    - r_eps_real：the radius of the epsilon ball in tech resolution in um, default 200 um
    - scale_factor: 1 spatial coord unit equals to how many µm

    Input:
       AnnData: adata
    """
    assert "spatial" in adata.obsm, "adata.obsm['spatial'] is required."

    coords = adata.obsm["spatial"]
    tri = Delaunay(coords)
    edges = tri.simplices  # shape (n_tri, 3)

    edge_list = set()
    for tri in edges:
        i, j, k = tri
        edge_list.update({(i, j), (j, i), (i, k), (k, i), (j, k), (k, j)})

    edge_array = np.array(list(edge_list))  # shape (n_edge, 2)

    node1 = coords[edge_array[:, 0]]
    node2 = coords[edge_array[:, 1]]
    dist = np.linalg.norm(node1 - node2, axis=1)
    print('the max dist is:', max(dist))

    max_r = r_eps_real / scale_factor
    valid = dist <= max_r
    valid_edges = edge_array[valid]
    
    sorted_edges = np.sort(valid_edges, axis=1)
    unique_edges = np.unique(sorted_edges, axis=0)

    n_cells = coords.shape[0]
    row = np.concatenate([unique_edges[:, 0], unique_edges[:, 1]])
    col = np.concatenate([unique_edges[:, 1], unique_edges[:, 0]])
    data = np.ones_like(row, dtype=float)
    W = sp.coo_matrix((data, (row, col)), shape=(n_cells, n_cells)).tocsr()

    deg = np.array(W.sum(axis=1)).flatten()
    has_neighbor = deg > 0
    # print('the has_neighbor is:\n',has_neighbor)
    W[has_neighbor, has_neighbor] = 1.0

    adata.obsp["DT_connectivities"] = W
    # print(f"Stored Delaunay neighbor graph with threshold {max_r} in adata.obsp['DT_connectivities']")

def loop_calculate_LRTF_allscore(adata, ex_mulnetlist, receiver_celltype, diff_LigRecDB_path, cont_LigRecDB_path, OutputDir):

    diff_LigRecDB = pd.read_csv(diff_LigRecDB_path)
    cont_LigRecDB = pd.read_csv(cont_LigRecDB_path)

    wd_model = os.path.join(OutputDir, 'runModel')
    os.makedirs(wd_model, exist_ok=True)
    print(f"Saving model intermediate results to {wd_model}")

    for receiver in receiver_celltype:
        Receiver = receiver
        Sender = None  

        LRTF_allscore = calculate_LRTF_allscore(
            adata=adata,
            mulNetList=ex_mulnetlist,
            diff_LigRecDB=diff_LigRecDB, 
            cont_LigRecDB=cont_LigRecDB,
            Receiver=Receiver,
            Sender=Sender,
        )

        if len(LRTF_allscore['LRs_score']) != 0:
            filename = os.path.join(wd_model, f"LRTF_allscore_TME-{Receiver}.pkl")
            with open(filename, 'wb') as f:
                pickle.dump(LRTF_allscore, f)

    return "Done"

# Define calculate_LRTF_allscore
def calculate_LRTF_allscore(adata, mulNetList, diff_LigRecDB, cont_LigRecDB, Receiver, Sender=None,
                            group=None, far_ct=0.75, close_ct=0.25, downsample=False):
    
    # prepare data
    exprMat = pd.DataFrame(
        adata.layers['Imputate'].T,
        index=adata.var_names,
        columns=adata.obs_names
    )
    annoMat = pd.DataFrame({
        "Barcode": adata.obs_names,
        "Cluster": adata.obs["Cluster"].values
    })
    coords = adata.obsm['spatial']
    distMat = pd.DataFrame(
        squareform(pdist(coords, metric="euclidean")),
        index=adata.obs_names,
        columns=adata.obs_names
    )
    np.fill_diagonal(distMat.values, 1)

    build_DT_neighbors(adata, mode="weight_sum_2")
    DT_neighbor = adata.obsp["DT_connectivities"]
    
    if Sender is None:
        filtered_nets = {k: v for k, v in mulNetList.items() if k.endswith(f"_{Receiver}")}
        
        mulNet_tab = []
        for mlnet in filtered_nets.values():
            ligrec = pd.DataFrame({'Ligand': mlnet['LigRec']['source'], 'Receptor': mlnet['LigRec']['target']})
            rectf = pd.DataFrame({'Receptor': mlnet['RecTF']['source'], 'TF': mlnet['RecTF']['target']})
            tftg = pd.DataFrame({'TF': mlnet['TFTar']['source'], 'Target': mlnet['TFTar']['target']})
            merged = ligrec.merge(rectf, on='Receptor').merge(tftg, on='TF')[['Ligand', 'Receptor', 'TF', 'Target']]
            mulNet_tab.append(merged.sort_values(by=['Ligand', 'Receptor']).reset_index(drop=True))
        
        mulNet_tab = pd.concat(mulNet_tab, ignore_index=True)
        
        diff_mulNet_tab = mulNet_tab[
                          mulNet_tab['Ligand'].isin(diff_LigRecDB['source']) & mulNet_tab['Receptor'].isin(diff_LigRecDB['target'])
                          ].copy()
        num_diffligrec = diff_mulNet_tab[['Ligand', 'Receptor']].drop_duplicates().shape[0]
        print(f"diffusion Ligand-Receptor 对总数: {num_diffligrec}")
        print(f"calculate the regulatory score of LR pairs from microenvironment to {Receiver}")
        diff_LRTF_allscore = calculate_diff_LRTF_score(exprMat, distMat, annoMat, Receiver,Sender=Sender,mulNet_tab=diff_mulNet_tab,
                                             group=group,far_ct=far_ct,close_ct=close_ct, downsample=downsample)
        print('the keys of diff_LRTF_allscore is:', diff_LRTF_allscore.keys())
        cont_mulNet_tab = mulNet_tab[
                          mulNet_tab['Ligand'].isin(cont_LigRecDB['source']) & mulNet_tab['Receptor'].isin(cont_LigRecDB['target'])
                          ].copy()
        num_contligrec = cont_mulNet_tab[['Ligand', 'Receptor']].drop_duplicates().shape[0]
        print(f"contact Ligand-Receptor 对总数: {num_contligrec}")  
        cont_LRTF_allscore = calculate_cont_LRTF_score(exprMat,DT_neighbor,annoMat,Receiver,mulNet_tab=cont_mulNet_tab,
                                                       group=group,far_ct=far_ct,close_ct=close_ct, downsample=downsample)
        print('the keys of cont_LRTF_allscore is:', cont_LRTF_allscore.keys())
        exit()
                
    else:
        cellpair = f"{Sender}-{Receiver}"
        if cellpair not in mulNetList:
            return None
        mlnet = mulNetList[cellpair]
        TFs = list(set(mlnet['TFTar']['source']))
        LRpairs = defaultdict(list)
        for _, row in mlnet['LigRec'].iterrows():
            for tf in TFs:
                LRpairs[tf].append(f"{row['source']}_{row['target']}")
        LRpairs = {k: list(set(v)) for k, v in LRpairs.items()}
        print('the LRpairs is:',type(LRpairs))
        exit()
        print(f"calculate the regulatory score of diffusion based LR pairs from {Sender} to {Receiver}")
        diff_LRTF_allscore = calculate_diff_LRTF_score(exprMat, distMat, annoMat, Receiver,Sender=Sender,mulNet_tab=diff_mulNet_tab,
                                             group=group,far_ct=far_ct,close_ct=close_ct, downsample=downsample)

    LRs_score_combined = {}

    for tf, df in diff_LRTF_allscore['LRs_score'].items():
        LRs_score_combined[f'diff_{tf}'] = df

    for tf, df in cont_LRTF_allscore['LRs_score'].items():
        LRs_score_combined[f'cont_{tf}'] = df

    TFs_expr_diff = diff_LRTF_allscore['TFs_expr'].add_prefix('diff_')
    TFs_expr_cont = cont_LRTF_allscore['TFs_expr'].add_prefix('cont_')

    TFs_expr_combined = pd.concat([TFs_expr_diff, TFs_expr_cont], axis=1)

    combined_allscore = {
    'LRs_score': LRs_score_combined,
    'TFs_expr': TFs_expr_combined
    }

    return combined_allscore

# Python version of the R function 'calculate_LRTF_score'
def calculate_diff_LRTF_score(exprMat, distMat, annoMat, Receiver,mulNet_tab,Sender=None, 
                              group=None,far_ct=0.75,close_ct=0.25, downsample=False):
    
    mulNet_tab['LR'] = mulNet_tab['Ligand'] + '_' + mulNet_tab['Receptor']
    LRpairs = mulNet_tab.groupby('TF')['LR'].apply(lambda x: list(set(x))).to_dict()
    TFs = list(LRpairs.keys())
    # for tf in TFs:
    #     print(f"TF: {tf}, #diffusion LR pairs: {len(LRpairs[tf])}")

    Receptors = {tf: [lr.split("_")[1] for lr in LRpairs[tf]] for tf in TFs}
    Ligands = {tf: [lr.split("_")[0] for lr in LRpairs[tf]] for tf in TFs}

    receBars = annoMat[annoMat['Cluster'] == Receiver]['Barcode'].tolist()
    sendBars = exprMat.columns.tolist()

    LigMats = {}
    for tf in TFs:
        ligs = Ligands[tf]
        lig_count = exprMat.loc[ligs,sendBars].values
        LigMats[tf] = pd.DataFrame(lig_count, index=LRpairs[tf], columns= sendBars)

    RecMats = {}
    for tf in TFs:
        recs = Receptors[tf]
        rec_count = exprMat.loc[recs,receBars].values
        RecMats[tf] = pd.DataFrame(rec_count, index=LRpairs[tf], columns=receBars)

    distMat = distMat.loc[sendBars, receBars]
    distMat = 1 / distMat.replace(0, np.nan)

    cpMat = None
    if group is not None:
        cpMat = get_cell_pairs(group, distMat, far_ct, close_ct)
    
    LRs_score = {}
    for tf in TFs:
        LigMat = LigMats[tf]
        RecMat = RecMats[tf]
        lr = LRpairs[tf]

        if cpMat is None:
            LR_score = RecMat.values * (LigMat.values @ distMat.values)
            LR_score_df = pd.DataFrame(LR_score.T, columns=lr, index=receBars)
        else:
            rec_cells = cpMat['Receiver'].unique()
            rows = []
            for j in rec_cells:
                senders = cpMat[cpMat['Receiver'] == j]['Sender'].unique()
                if len(senders) == 1:
                    val = RecMat.loc[:, j].values * (LigMat.loc[:, senders].values * distMat.loc[senders, j].values)
                else:
                    val = RecMat.loc[:, j].values * (LigMat.loc[:, senders].values @ distMat.loc[senders, j].values)
                rows.append(val)
            LR_score_df = pd.DataFrame(rows, index=rec_cells, columns=lr)
        LRs_score[tf] = LR_score_df

    if cpMat is None:
        TFs_expr = {tf: exprMat.loc[tf, receBars].values for tf in TFs}
    else:
        TFs_expr = {tf: exprMat.loc[tf, cpMat['Receiver'].unique()].values for tf in TFs}

    if len(receBars) > 500 and downsample:
        np.random.seed(2021)
        if cpMat is None:
            keep_cell = np.random.choice(receBars, size=500, replace=False)
        else:
            keep_cell = np.random.choice(cpMat['Receiver'].unique(), size=500, replace=False)

        LRs_score = {tf: df.loc[keep_cell] for tf, df in LRs_score.items()}
        TFs_expr = {tf: expr[keep_cell] for tf, expr in TFs_expr.items()}

    return {"LRs_score": LRs_score, "TFs_expr": TFs_expr}

def calculate_cont_LRTF_score(exprMat, DT_neighbor, annoMat, Receiver,mulNet_tab,Sender=None, 
                              group=None,far_ct=0.75,close_ct=0.25, downsample=False):
    
    mulNet_tab['LR'] = mulNet_tab['Ligand'] + '_' + mulNet_tab['Receptor']
    LRpairs = mulNet_tab.groupby('TF')['LR'].apply(lambda x: list(set(x))).to_dict()
    TFs = list(LRpairs.keys())
    # for tf in TFs:
    #     print(f"TF: {tf}, #contact LR pairs: {len(LRpairs[tf])}")
    
    Receptors = {tf: [lr.split("_")[1] for lr in LRpairs[tf]] for tf in TFs}
    Ligands = {tf: [lr.split("_")[0] for lr in LRpairs[tf]] for tf in TFs}
    
    receBars = annoMat[annoMat['Cluster'] == Receiver]['Barcode'].tolist()
    rece_indices = annoMat.index[annoMat['Cluster'] == Receiver]

    LRs_score = {}
    for tf in TFs:
        ligs = Ligands[tf]
        recs = Receptors[tf]
        lr = LRpairs[tf]
        
        rows = []
        for idx_pos, j in enumerate(rece_indices):
            sender_neighbors = DT_neighbor[j].nonzero()[1] 
            sendBar = exprMat.columns.values[sender_neighbors]
            receBar = receBars[idx_pos]

            # print('the receBars is:',receBars)
            lig_count = exprMat.loc[ligs,sendBar].values
            rec_count = exprMat.loc[recs,receBar].values
            # print('the shape of lig_count is:',lig_count.shape)
            # print('the shape of rec_count is:',rec_count.shape)

            lig_sum = lig_count.sum(axis=1)
            val = rec_count * lig_sum
            rows.append(val)
        LR_score_df = pd.DataFrame(rows, index=receBars, columns=lr)

    LRs_score[tf] = LR_score_df
    TFs_expr = {tf: exprMat.loc[tf, receBars].values for tf in TFs}

    return {"LRs_score": LRs_score, "TFs_expr": TFs_expr}


def get_cell_pairs(distMat, group=None, far_ct=0.75, close_ct=0.25):
 
    distMat_long = distMat.reset_index().melt(id_vars='index', var_name='Receiver', value_name='Distance')
    distMat_long.rename(columns={'index': 'Sender'}, inplace=True)

    distMat_long['Sender'] = distMat_long['Sender'].astype(str)
    distMat_long['Receiver'] = distMat_long['Receiver'].astype(str)

    if group is None or group == 'all':
        respon_cellpair = distMat_long[['Sender', 'Receiver']]
    elif group == 'close':
        threshold = distMat_long['Distance'].quantile(close_ct)
        respon_cellpair = distMat_long[distMat_long['Distance'] <= threshold][['Sender', 'Receiver']]
    elif group == 'far':
        threshold = distMat_long['Distance'].quantile(far_ct)
        respon_cellpair = distMat_long[distMat_long['Distance'] >= threshold][['Sender', 'Receiver']]
    else:
        raise ValueError("Invalid group. Must be None, 'close', or 'far'.")

    return respon_cellpair

def get_TFLR_activity(mulNet_tab, LRTF_allscore):
    mulNet_tab['LRpair'] = mulNet_tab['Ligand'] + "_" + mulNet_tab['Receptor']
    LRpairs = mulNet_tab['LRpair'].unique()
    TFs = list(LRTF_allscore['LRs_score'].keys())
    cell_ids = LRTF_allscore['LRs_score'][TFs[0]].index

    TFLR_score = {}
    for i in cell_ids:
        tflr_score = pd.DataFrame(0, index=TFs, columns=LRpairs)
        for tf in TFs:
            LR_score = LRTF_allscore['LRs_score'][tf]
            cell_score = LR_score.loc[i]
            intersecting_cols = tflr_score.columns.intersection(LR_score.columns)
            tflr_score.loc[tf, intersecting_cols] = cell_score[intersecting_cols].values
        TFLR_score[i] = tflr_score

    return TFLR_score

def get_TFLR_allactivity(mulNetList, OutputDir):

    TFLR_allscore = {}
    wd_model = os.path.join(OutputDir, 'runModel')
    LRTF_score_files = [f for f in os.listdir(wd_model) if "LRTF" in f]

    for f in LRTF_score_files:
        print('Loading ',f)

        cellpair = re.sub(r"LRTF_allscore_|\.pkl", "", f)
        Receiver = cellpair.split("-")[-1]
        Sender = cellpair.split("-")[0]

        LRTF_allscore = pd.read_pickle(os.path.join(wd_model, f))

        if Sender == "TME":
            mulNet = {k: v for k, v in mulNetList.items() if f"_{Receiver}" in k}
            mulNet_tab = []
            for mlnet in mulNet.values():
                ligrec = pd.DataFrame({'Ligand': mlnet['LigRec']['source'], 'Receptor': mlnet['LigRec']['target']})
                rectf = pd.DataFrame({'Receptor': mlnet['RecTF']['source'], 'TF': mlnet['RecTF']['target']})
                tftg = pd.DataFrame({'TF': mlnet['TFTar']['source'], 'Target': mlnet['TFTar']['target']})
                res = ligrec.merge(rectf, on='Receptor').merge(tftg, on='TF')[['Ligand', 'Receptor', 'TF', 'Target']]
                mulNet_tab.append(res.sort_values(by=['Ligand', 'Receptor']))
            mulNet_tab = pd.concat(mulNet_tab)
            TFLR_score = get_TFLR_activity(mulNet_tab, LRTF_allscore)
        else:
            mulNet = mulNetList[cellpair]
            ligrec = pd.DataFrame({'Ligand': mulNet['LigRec']['source'], 'Receptor': mulNet['LigRec']['target']})
            rectf = pd.DataFrame({'Receptor': mulNet['RecTF']['source'], 'TF': mulNet['RecTF']['target']})
            tftg = pd.DataFrame({'TF': mulNet['TFTar']['source'], 'Target': mulNet['TFTar']['target']})
            mulNet_tab = ligrec.merge(rectf, on='Receptor').merge(tftg, on='TF')[['Ligand', 'Receptor', 'TF', 'Target']]
            mulNet_tab = mulNet_tab.sort_values(by=['Ligand', 'Receptor'])
            TFLR_score = get_TFLR_activity(mulNet_tab, LRTF_allscore)

        if len(LRTF_allscore.get('LRs_score', {})) != 0:
            with open(os.path.join(wd_model, f"TFLR_allscore_{Sender}_{Receiver}.pkl"), "wb") as f_out:
                pd.to_pickle(TFLR_score, f_out)

        TFLR_allscore.update(TFLR_score)

    mulNet_alltab = []
    for mlnet in mulNetList.values():
        ligrec = pd.DataFrame({'Ligand': mlnet['LigRec']['source'], 'Receptor': mlnet['LigRec']['target']})
        rectf = pd.DataFrame({'Receptor': mlnet['RecTF']['source'], 'TF': mlnet['RecTF']['target']})
        tftg = pd.DataFrame({'TF': mlnet['TFTar']['source'], 'Target': mlnet['TFTar']['target']})
        res = ligrec.merge(rectf, on='Receptor').merge(tftg, on='TF')[['Ligand', 'Receptor', 'TF', 'Target']]
        mulNet_alltab.append(res.sort_values(by=['Ligand', 'Receptor']))
    mulNet_alltab = pd.concat(mulNet_alltab)

    TFs = mulNet_alltab['TF'].unique()
    TGs = mulNet_alltab['Target'].unique()
    LRpairs = (mulNet_alltab['Ligand'] + "_" + mulNet_alltab['Receptor']).unique()
    cell_ids = list(TFLR_allscore.keys())

    TFLR_allscore_new = {}
    for i in cell_ids:
        tflr_score = pd.DataFrame(0, index=TFs, columns=LRpairs)
        LR_score = TFLR_allscore[i]
        tflr_score.loc[LR_score.index, LR_score.columns] = LR_score
        TFLR_allscore_new[i] = tflr_score

    TFTG_link = mulNet_tab[['TF', 'Target']]
    TFLR_all = {
        'TFLR_allscore': TFLR_allscore_new,
        'LR_links': LRpairs,
        'TFTG_link': TFTG_link
    }

    return TFLR_all

def save_LRscore_and_MLnet(adata, mulNetList, TFLR_all_score, save_path):

    adata.write_h5ad(save_path+'adata_raw.h5ad')
    mulNet_tab = []
    for mlnet in mulNetList.values():
        ligrec = pd.DataFrame({'Ligand': mlnet['LigRec']['source'], 'Receptor': mlnet['LigRec']['target']})
        rectf = pd.DataFrame({'Receptor': mlnet['RecTF']['source'], 'TF': mlnet['RecTF']['target']})
        tftg = pd.DataFrame({'TF': mlnet['TFTar']['source'], 'Target': mlnet['TFTar']['target']})

        merged = ligrec.merge(rectf, on='Receptor').merge(tftg, on='TF')
        merged = merged[['Ligand', 'Receptor', 'TF', 'Target']].sort_values(by=['Ligand', 'Receptor'])
        mulNet_tab.append(merged)

    mulNet_tab = pd.concat(mulNet_tab, ignore_index=True)

    LR_link = mulNet_tab[['Ligand', 'Receptor']].rename(columns={'Ligand': 'ligand', 'Receptor': 'receptor'})
    TFTG_link = mulNet_tab[['TF', 'Target']].rename(columns={'Target': 'TG'})

    wd_score = os.path.join(save_path, "TFLR_score")
    os.makedirs(wd_score, exist_ok=True)

    TFLR_allscore = TFLR_all_score['TFLR_allscore']
    cell_ids = list(TFLR_allscore.keys())

    for cell_id in cell_ids:
        score_df = TFLR_allscore[cell_id]
        score_df.to_csv(os.path.join(wd_score, f"{cell_id}_TFLR_score.csv"), index=False)

    LR_link.to_csv(os.path.join(save_path, 'LR_links.csv'), index=False)
    TFTG_link.to_csv(os.path.join(save_path, 'TFTG_links.csv'), index=False)
