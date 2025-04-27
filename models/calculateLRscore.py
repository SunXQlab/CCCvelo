import numpy as np
import pandas as pd
import os
import pickle
from collections import defaultdict
import os
import re

def loop_calculate_LRTF_allscore(exprMat, distMat, annoMat, ex_mulnetlist, neuronal_ct, wd_model):

    for receiver in neuronal_ct:
        Receiver = receiver
        Sender = None  # kept for future compatibility

        LRTF_allscore = calculate_LRTF_allscore(
            exprMat=exprMat,
            distMat=distMat,
            annoMat=annoMat,
            mulNetList=ex_mulnetlist,
            Receiver=Receiver,
            Sender=Sender
        )

        if len(LRTF_allscore['LRs_score']) != 0:
            filename = os.path.join(wd_model, f"LRTF_allscore_TME-{Receiver}.pkl")
            with open(filename, 'wb') as f:
                pickle.dump(LRTF_allscore, f)

    return "Done"

# Define calculate_LRTF_allscore
def calculate_LRTF_allscore(exprMat, distMat, annoMat, mulNetList, Receiver, Sender=None,
                            group=None, far_ct=0.75, close_ct=0.25, downsample=False):
    
    if Sender is None:
        filtered_nets = {k: v for k, v in mulNetList.items() if k.endswith(f"_{Receiver}")}
        
        mulNet_tab = []
        for mlnet in filtered_nets.values():
            ligrec = pd.DataFrame({'Ligand': mlnet['LigRec']['source'], 'Receptor': mlnet['LigRec']['target']})
            rectf = pd.DataFrame({'Receptor': mlnet['RecTF']['source'], 'TF': mlnet['RecTF']['target']})
            tftg = pd.DataFrame({'TF': mlnet['TFTar']['source'], 'Target': mlnet['TFTar']['target']})
            merged = ligrec.merge(rectf, on='Receptor').merge(tftg, on='TF')[['Ligand', 'Receptor', 'TF', 'Target']]
            mulNet_tab.append(merged.sort_values(by=['Ligand', 'Receptor']))
        
        mulNet_tab = pd.concat(mulNet_tab, ignore_index=True)
        
        LRpairs = defaultdict(list)
        for _, row in mulNet_tab.iterrows():
            LRpairs[row['TF']].append(f"{row['Ligand']}_{row['Receptor']}")
        LRpairs = {k: list(set(v)) for k, v in LRpairs.items()}
        TFs = list(LRpairs.keys())

        print(f"calculate the regulatory score of LR pairs from microenvironment to {Receiver}")
        LRTF_allscore = calculate_LRTF_score(exprMat, distMat, annoMat, LRpairs, TFs, Receiver,
                                             group=group, Sender=Sender, far_ct=far_ct,
                                             close_ct=close_ct, downsample=downsample)

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

        print(f"calculate the regulatory score of LR pairs from {Sender} to {Receiver}")
        LRTF_allscore = calculate_LRTF_score(exprMat, distMat, annoMat, LRpairs, TFs, Receiver,
                                             group=group, Sender=Sender, far_ct=far_ct,
                                             close_ct=close_ct, downsample=downsample)

    return LRTF_allscore

# Python version of the R function 'calculate_LRTF_score'
def calculate_LRTF_score(exprMat, distMat, annoMat, LRpairs, TFs, Receiver, group=None,
                         Sender=None, far_ct=0.75, close_ct=0.25, downsample=False):
    receBars = annoMat[annoMat['Cluster'] == Receiver]['Barcode'].tolist()
    sendBars = exprMat.columns.tolist()

    Receptors = {tf: [lr.split("_")[1] for lr in LRpairs[tf]] for tf in TFs}
    Ligands = {tf: [lr.split("_")[0] for lr in LRpairs[tf]] for tf in TFs}

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
    
    # 这里要把Ligand分为contact和diffusion两种方法写
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

def get_TFLR_allactivity(mulNetList, LRTF_score_files, wd_model):
    TFLR_allscore = {}
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
        'LRpairs': LRpairs,
        'TFTG_link': TFTG_link
    }

    return TFLR_all

import os
import pandas as pd

def save_LRscore_and_MLnet(adata, mulNetList, TFLR_all_score, save_path):

    adata.write_h5ad(save_path+'adata_pp.h5ad')
    
    # 合并 mulNetList 为一个总的 DataFrame
    mulNet_tab = []
    for mlnet in mulNetList.values():
        ligrec = pd.DataFrame({'Ligand': mlnet['LigRec']['source'], 'Receptor': mlnet['LigRec']['target']})
        rectf = pd.DataFrame({'Receptor': mlnet['RecTF']['source'], 'TF': mlnet['RecTF']['target']})
        tftg = pd.DataFrame({'TF': mlnet['TFTar']['source'], 'Target': mlnet['TFTar']['target']})

        merged = ligrec.merge(rectf, on='Receptor').merge(tftg, on='TF')
        merged = merged[['Ligand', 'Receptor', 'TF', 'Target']].sort_values(by=['Ligand', 'Receptor'])
        mulNet_tab.append(merged)

    mulNet_tab = pd.concat(mulNet_tab, ignore_index=True)

    # 构建边表
    LR_link = mulNet_tab[['Ligand', 'Receptor']].rename(columns={'Ligand': 'ligand', 'Receptor': 'receptor'})
    TFTG_link = mulNet_tab[['TF', 'Target']].rename(columns={'Target': 'TG'})

    # 创建评分输出目录
    wd_score = os.path.join(save_path, "TFLR_score")
    os.makedirs(wd_score, exist_ok=True)

    # 获取 TFLR 打分
    TFLR_allscore = TFLR_all_score['TFLR_allscore']
    cell_ids = list(TFLR_allscore.keys())

    for cell_id in cell_ids:
        score_df = TFLR_allscore[cell_id]
        score_df.to_csv(os.path.join(wd_score, f"{cell_id}_TFLR_score.csv"), index=False)

    # 保存边表
    LR_link.to_csv(os.path.join(save_path, 'LR_links.csv'), index=False)
    TFTG_link.to_csv(os.path.join(save_path, 'TFTG_links.csv'), index=False)






