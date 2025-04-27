import os
import dfply
from dfply import *
import pickle
import pandas as pd
import numpy as np
import scanpy as sc
import anndata as ad
import itertools
from scipy.stats import fisher_exact

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory created at: {path}")
        
def runMLnet(ExprMat, AnnoMat,TGList, LigList, RecList, LigClus = None, RecClus = None, OutputDir = None, Databases = None,
             RecTF_method = "Fisher", TFTG_method = "Fisher", logfc_ct = 0.1, pct_ct = 0.05,
             pval_ct = 0.05, expr_ct = 0.1):
    # 载入数据库
    if Databases is None:
        print("Load default database")
        # 读取 CSV 文件
        LigRecDB = pd.read_csv("E:/CCCvelo/create_MLnet_python/Database/LigRecDB.csv")
        RecTFDB = pd.read_csv("E:/CCCvelo/create_MLnet_python/Database/RecTFDB.csv")
        TFTGDB = pd.read_csv("E:/CCCvelo/create_MLnet_python/Database/TFTGDB.csv")

        # 将 DataFrame 存入字典
        Databases = {
            'LigRec_DB': LigRecDB,
            'RecTF_DB': RecTFDB,
            'TFTG_DB': TFTGDB
        }

        Databases['RecTF_DB'] = Databases['RecTF_DB'][['source', 'target']].drop_duplicates()
        Databases['LigRec_DB'] = Databases['LigRec_DB'][['source', 'target']].drop_duplicates()
        Databases['LigRec_DB'] = Databases['LigRec_DB'][
            Databases['LigRec_DB']['target'].isin(Databases['RecTF_DB']['source'])]
        Databases['TFTG_DB'] = Databases['TFTG_DB'][['source', 'target']].drop_duplicates()
        Databases['TFTG_DB'] = Databases['TFTG_DB'][
            Databases['TFTG_DB']['source'].isin(Databases['RecTF_DB']['target'])]
    else:
        print("Load user database")
        for key in Databases:
            Databases[key]=pd.DataFrame(Databases[key])
        Databases['RecTF.DB']=Databases['RecTF.DB'].drop_duplicates(subset=['source','target'])
        Databases['LigRec.DB'] = Databases['LigRec.DB'].drop_duplicates(subset=['source', 'target'])
        Databases['LigRec.DB'] = Databases['LigRec.DB'].loc[Databases['LigRec.DB']['target'].isin(Databases['RecTF.DB']['source'])]
        Databases['TFTG.DB'] = Databases['TFTG.DB'].drop_duplicates(subset=['source', 'target'])
        Databases['TFTG.DB'] = Databases['TFTG.DB'].loc[
            Databases['TFTG.DB']['source'].isin(Databases['RecTF.DB']['target'])]

    # 创建工作目录
    WorkDir = os.path.join(OutputDir, 'runscMLnet')
    create_directory(WorkDir)
    print(f"WorkDir: {WorkDir}")

    # 检查 LigClus 和 RecClus
    if LigClus is None:
        LigClus = pd.Series(AnnoMat['Cluster'].unique()).astype(str)
    if RecClus is None:
        RecClus = pd.Series(AnnoMat['Cluster'].unique()).astype(str)

    # 输入对象
    inputs = {
        'parameters': {
            'LigClus': pd.Series(LigClus),
            'RecClus': pd.Series(RecClus),
            'WorkDir': WorkDir,
            'logfc_ct': logfc_ct,
            'pct_ct': pct_ct,
            'pval_ct': pval_ct,
            'expr_ct': expr_ct,
            'RecTF_method': RecTF_method,
            'TFTG_method': TFTG_method
        },
        'data': {
            'df_norm': ExprMat,
            'df_anno': AnnoMat,
            'ls_clusters': AnnoMat['Cluster'].unique().astype(str),
            'ls_targets': TGList,
            'ls_ligands': LigList,
            'ls_receptors': RecList
        }
    }  # 到这里还是正常的
    # 规范化

    # 获取多层网络
    outputs = {'mlnets': {}, 'details': {}}
    for RecClu in inputs['parameters']['RecClus'].values:
        LigClus = inputs['parameters']['LigClus'].values
        LigClus = LigClus[LigClus != RecClu]
        combination = [lig + "_" + RecClu for lig in LigClus]
        details = pd.DataFrame(index=['Lig_bk', 'Rec_bk', 'target_bk',
                                      "LRpair", "RecTFpair", "TFTGpair",
                                      "Ligand", "Receptor", "TF", "Target"], columns=combination)
        mlnets = {}

        for i, LigClu in enumerate(LigClus):
            print(f"{LigClu}_{RecClu}")
            resMLnet = getCellPairMLnet(inputs, LigClu, RecClu, Databases)
            mlnets[LigClu + '_' + RecClu] = resMLnet['mlnet']
            details[LigClu + "_" + RecClu] = resMLnet['detail']
        details.to_csv(os.path.join(WorkDir, f"TME-{RecClu}.csv"))
        outputs['mlnets'][RecClu] = mlnets
        outputs['details'][RecClu] = details

    return outputs

# Python version of getCellPairMLnet (no p-value section)
def getCellPairMLnet(inputs, ligclu, recclu, databases):
    
    # 1. Unpack inputs
    df_norm = inputs['data']['df_norm']
    ls_ligands = inputs['data']['ls_ligands']
    ls_receptors = inputs['data']['ls_receptors']
    ls_targets = inputs['data']['ls_targets']
    workdir = inputs['parameters']['WorkDir']
    RecTF_method = inputs['parameters']['RecTF_method']
    TFTG_method = inputs['parameters']['TFTG_method']

    # 2. Unpack databases
    LigRec_DB = databases['LigRec_DB']
    TFTG_DB = databases['TFTG_DB']
    RecTF_DB = databases['RecTF_DB']

    # 3. LigRec
    source_abundant = ls_ligands.get(ligclu, [])
    print("source_background:", len(source_abundant))
    target_abundant = ls_receptors.get(recclu, [])
    print("target_background:", len(target_abundant))

    try:
        LigRecTab = getLigRec(LigRec_DB, source_abundant, target_abundant)
    except Exception as e:
        print(str(e))
        LigRecTab = pd.DataFrame()

    # 4. TFTG
    target_gene = df_norm.columns[df_norm.mean(axis=0) > 0].tolist()
    # target_gene = pd.Series(df_norm.columns[df_norm.mean(axis=0) > 0])
    print("target_background:", len(target_gene))

    target_icg = ls_targets.get(recclu, {})
    if isinstance(target_icg, dict) and ligclu in target_icg:
        target_icg = target_icg[ligclu]
    print("target_icg:", len(target_icg))

    try:
        TFTGTab = getTFTG(TFTG_DB, target_icg, target_gene, TFTG_method)
    except Exception as e:
        print(str(e))
        TFTGTab = pd.DataFrame()

    # 5. RecTF
    if not LigRecTab.empty and not TFTGTab.empty:
        Rec_list = getNodeList(LigRecTab, "target")
        TF_list = getNodeList(TFTGTab, "source")
        try:
            RecTFTab = getRecTF(RecTF_DB, Rec_list, TF_list, RecTF_method)
        except Exception as e:
            print(str(e))
            RecTFTab = pd.DataFrame()
    else:
        RecTFTab = pd.DataFrame()

    # 6. Merge
    if not RecTFTab.empty and not LigRecTab.empty:
        receptors_in_tab = getNodeList(RecTFTab, "source")
        LigRecTab_new = LigRecTab[LigRecTab.iloc[:, 1].isin(receptors_in_tab)]
    else:
        LigRecTab_new = pd.DataFrame()
    print("LR pairs:", LigRecTab_new.shape[0])

    if not RecTFTab.empty and not TFTGTab.empty:
        TFs_in_tab = getNodeList(RecTFTab, "target")
        TFTGTab_new = TFTGTab[TFTGTab.iloc[:, 0].isin(TFs_in_tab)]
    else:
        TFTGTab_new = pd.DataFrame()
    print("TFTG pairs:", TFTGTab_new.shape[0])

    # 7. Result
    mlnet = {
        "LigRec": LigRecTab_new,
        "RecTF": RecTFTab,
        "TFTar": TFTGTab_new
    }

    foldername = f"{ligclu}_{recclu}"
    workdir = os.path.join(workdir, foldername)
    os.makedirs(workdir, exist_ok=True)
    pd.to_pickle(mlnet, os.path.join(workdir, "scMLnet.pkl"))

    # 8. Detail info
    detail = [
        len(source_abundant),
        len(target_abundant),
        len(target_icg),
        mlnet['LigRec'].shape[0],
        mlnet['RecTF'].shape[0],
        mlnet['TFTar'].shape[0],
        mlnet['LigRec']['source'].nunique() if not mlnet['LigRec'].empty else 0,
        mlnet['LigRec']['target'].nunique() if not mlnet['LigRec'].empty else 0,
        mlnet['TFTar']['source'].nunique() if not mlnet['TFTar'].empty else 0,
        mlnet['TFTar']['target'].nunique() if not mlnet['TFTar'].empty else 0
    ]

    return {
        'mlnet': mlnet,
        'detail': detail
    }

def getLigRec(LigRec_DB, source_up, target_up):
    # Check input
    if not isinstance(LigRec_DB, pd.DataFrame):
        raise ValueError("LigRec_DB must be a pandas DataFrame")
    if 'source' not in LigRec_DB.columns:
        raise ValueError("LigRec_DB must contain a column named 'source'")
    if 'target' not in LigRec_DB.columns:
        raise ValueError("LigRec_DB must contain a column named 'target'")

    # Get ligand and receptor list
    LigGene = LigRec_DB['source'].unique()
    RecGene = LigRec_DB['target'].unique()
    TotLigRec = (LigRec_DB['source'] + "_" + LigRec_DB['target']).unique()

    # Get highly expressed ligand and receptor
    LigHighGene = set(LigGene).intersection(source_up)
    RecHighGene = set(RecGene).intersection(target_up)

    # Generate all possible LR combinations
    LRList = [f"{lig}_{rec}" for lig in LigHighGene for rec in RecHighGene]
    LRList = list(set(LRList).intersection(TotLigRec))

    if len(LRList) == 0:
        raise ValueError("Error: No significant LigRec pairs")

    # Create result table
    LRTable = pd.DataFrame([pair.split('_') for pair in LRList], columns=['source', 'target'])

    print(f"get {len(LRList)} activated LR pairs")
    return LRTable

def getBarList(Aclu, BarCluTable):
    AcluBar = BarCluTable[BarCluTable['Cluster'] == Aclu]['Barcode'].tolist()
    AllBar = BarCluTable['Barcode'].tolist()
    OtherBar = list(set(AllBar) - set(AcluBar))
    return [AcluBar, OtherBar]

def runFisherTest(subset1, subset2, background):
    a = len(set(subset1).intersection(subset2))
    b = len(subset1) - a
    c = len(subset2) - a
    d = len(background) - a - b - c
    matrix = [[a, c], [b, d]]
    _, p_value = fisher_exact(matrix, alternative='greater')
    return p_value

def getNodeList(Database, Nodetype):
    return Database[Nodetype].unique().tolist()

def getTFTG(TFTG_DB, target_degs, target_genes, method='Fisher'):
    if method == 'Search':
        return getTFTGSearch(TFTG_DB, target_degs, target_genes)
    elif method == 'Fisher':
        return getTFTGFisher(TFTG_DB, target_degs, target_genes)

def getTFTGSearch(TFTG_DB, target_degs, target_genes):
    TFGene = TFTG_DB['source'].unique()
    TargetGene = TFTG_DB['target'].unique()
    TotTFTG = (TFTG_DB['source'] + "_" + TFTG_DB['target']).unique()

    TargetHighGene = list(set(TargetGene).intersection(target_degs))
    TFGene = [gene for gene in TFGene if gene in target_genes]

    TFTGList = [f"{tf}_{tg}" for tf in TFGene for tg in TargetHighGene]
    TFTGList = list(set(TFTGList).intersection(TotTFTG))

    if len(TFTGList) == 0:
        raise ValueError("Error: No significant TFTG pairs")

    TFTGTable = pd.DataFrame([x.split('_') for x in TFTGList], columns=['source', 'target'])
    print(f"get {len(TFTGList)} activated TFTG pairs")
    return TFTGTable

def getTFTGFisher(TFTG_DB, target_degs, target_genes):
    TF_list = TFTG_DB['source'].unique()
    TG_list = {
        tf: TFTG_DB[TFTG_DB['source'] == tf]['target'].unique().tolist()
        for tf in TF_list
    }
    TFs = {
        tf: runFisherTest(targets, target_degs, target_genes)
        for tf, targets in TG_list.items()
    }
    significant_TFs = [tf for tf, p in TFs.items() if p <= 0.05 and tf in target_genes]

    TFTGList = [
        f"{tf}_{tg}"
        for tf in significant_TFs
        for tg in set(TG_list[tf]).intersection(target_degs)
    ]

    if len(TFTGList) == 0:
        raise ValueError("Error: No significant TFTG pairs")

    TFTGTable = pd.DataFrame([x.split('_') for x in TFTGList], columns=['source', 'target'])
    print(f"get {len(TFTGList)} activated TFTG pairs")
    return TFTGTable

def getRecTF(RecTF_DB, Rec_list, TF_list, method='Fisher'):
    if method == 'Search':
        return getRecTFSearch(RecTF_DB, Rec_list, TF_list)
    elif method == 'Fisher':
        return getRecTFFisher(RecTF_DB, Rec_list, TF_list)

def getRecTFSearch(RecTF_DB, Rec_list, TF_list):
    TotRecTF = (RecTF_DB['source'] + "_" + RecTF_DB['target']).unique()
    RecTFList = [f"{rec}_{tf}" for rec in Rec_list for tf in TF_list]
    RecTFList = list(set(RecTFList).intersection(TotRecTF))

    if len(RecTFList) == 0:
        raise ValueError("Error: No significant RecTF pairs")

    RecTFTable = pd.DataFrame([x.split('_') for x in RecTFList], columns=['source', 'target'])
    print(f"get {len(RecTFList)} activated RecTF pairs")
    return RecTFTable

def getRecTFFisher(RecTF_DB, Rec_list, TF_list):
    RecTF_DB = RecTF_DB[RecTF_DB['source'].isin(Rec_list) & RecTF_DB['target'].isin(TF_list)]
    TFofRec = {
        rec: RecTF_DB[RecTF_DB['source'] == rec]['target'].unique().tolist()
        for rec in Rec_list
    }
    TFofALL = RecTF_DB['target'].unique().tolist()

    Recs = {
        rec: runFisherTest(tfs, TF_list, TFofALL)
        for rec, tfs in TFofRec.items()
    }
    significant_Recs = [rec for rec, p in Recs.items() if p <= 0.05]

    RecTFList = [
        f"{rec}_{tf}"
        for rec in significant_Recs
        for tf in set(TFofRec[rec]).intersection(TF_list)
    ]

    if len(RecTFList) == 0:
        raise ValueError("Error: No significant RecTF pairs")

    RecTFTable = pd.DataFrame([x.split('_') for x in RecTFList], columns=['source', 'target'])
    print(f"get {len(RecTFList)} activated RecTF pairs")
    return RecTFTable

def getRecpval(RecTF_DB, Rec_list, TF_list, method='Fisher'):
    if method == 'Search':
        return getRecTFSearch(RecTF_DB, Rec_list, TF_list)
    elif method == 'Fisher':
        return getRecFisherpval(RecTF_DB, Rec_list, TF_list)

def getRecFisherpval(RecTF_DB, Rec_list, TF_list):
    RecTF_DB = RecTF_DB[RecTF_DB['source'].isin(Rec_list) & RecTF_DB['target'].isin(TF_list)]
    TFofRec = {
        rec: RecTF_DB[RecTF_DB['source'] == rec]['target'].unique().tolist()
        for rec in Rec_list
    }
    TFofALL = RecTF_DB['target'].unique().tolist()

    Recs = {
        rec: runFisherTest(tfs, TF_list, TFofALL)
        for rec, tfs in TFofRec.items()
    }
    return pd.Series(Recs, name='RecTFpval')

def summarize_multinet(ex_mulnetlist):
    mulNet_tab_list = []

    for name, mlnet in ex_mulnetlist.items():
        ligrec = pd.DataFrame({
            "Ligand": mlnet['LigRec']['source'],
            "Receptor": mlnet['LigRec']['target']
        })
        rectf = pd.DataFrame({
            "Receptor": mlnet['RecTF']['source'],
            "TF": mlnet['RecTF']['target']
        })
        tftg = pd.DataFrame({
            "TF": mlnet['TFTar']['source'],
            "Target": mlnet['TFTar']['target']
        })

        res = pd.merge(ligrec, rectf, on='Receptor')
        res = pd.merge(res, tftg, on='TF')
        res = res[['Ligand', 'Receptor', 'TF', 'Target']].sort_values(by=['Ligand', 'Receptor'])

        mulNet_tab_list.append(res)

    mulNet_tab = pd.concat(mulNet_tab_list, ignore_index=True)

    summary = {
        "Number of ligands": mulNet_tab['Ligand'].nunique(),
        "Number of receptors": mulNet_tab['Receptor'].nunique(),
        "Number of TFs": mulNet_tab['TF'].nunique(),
        "Number of TGs": mulNet_tab['Target'].nunique()
    }

    return summary


# 后续进一步分析的代码，上面的是runMLnet函数涉及的代码
def extract_MLnet(resMLnet):
    ex_mulnetlist = {}

    mlnets = resMLnet["mlnets"]
    for name, mlnet in mlnets.items():
        if not mlnet["LigRec"].empty:
            ex_mulnetlist[name] = mlnet

    mulNet_tab_list = []

    for name, mlnet in ex_mulnetlist.items():
        ligrec = pd.DataFrame({
            "Ligand": mlnet['LigRec']['source'],
            "Receptor": mlnet['LigRec']['target']
        })
        rectf = pd.DataFrame({
            "Receptor": mlnet['RecTF']['source'],
            "TF": mlnet['RecTF']['target']
        })
        tftg = pd.DataFrame({
            "TF": mlnet['TFTar']['source'],
            "Target": mlnet['TFTar']['target']
        })

        res = ligrec.merge(rectf, on='Receptor').merge(tftg, on='TF')
        res = res[['Ligand', 'Receptor', 'TF', 'Target']].sort_values(by=['Ligand', 'Receptor'])
        mulNet_tab_list.append(res)

    mulNet_tab = pd.concat(mulNet_tab_list, ignore_index=True)

    print("The number of ligands is:", mulNet_tab['Ligand'].nunique())
    print("The number of receptors is:", mulNet_tab['Receptor'].nunique())
    print("The number of TFs is:", mulNet_tab['TF'].nunique())
    print("The number of TGs is:", mulNet_tab['Target'].nunique())

    return ex_mulnetlist



