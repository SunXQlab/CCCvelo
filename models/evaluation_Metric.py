import numpy as np
import scipy
from sklearn.metrics.pairwise import cosine_similarity

def calculate_SpearmanMtx(X,Y):
    coef = []
    for i in range(X.shape[0]):
        # X_i = X[:,i]
        # Y_i = Y[:,i]
        X_i = X[i,:]
        Y_i = Y[i,:]
        velo_spear = scipy.stats.spearmanr(X_i, Y_i)[0]
        # print(f'the {i} spearman correlication is:',velo_spear)
        coef.append(velo_spear)
    correlation_mean = np.mean(coef)
    return correlation_mean

def calculate_Spearman(adata_velo):
    adata_copy = adata_velo.copy()
    # calculate the spearman coefficient between groundtruth pseudotime and velocity pseudotime
    velo_psd = adata_copy.obs['velocity_pseudotime']
    fit_t = adata_copy.obs['fit_t']
    gdt_psd = adata_copy.obs['groundTruth_psd']
    # correlation_fit = scipy.stats.spearmanr(fit_t, velo_psd)[0]
    # correlation_fit_with_gd = scipy.stats.spearmanr(fit_t, gdt_psd)[0]
    correlation_psd = scipy.stats.spearmanr(gdt_psd, velo_psd)[0]

    # calculate the spearman coefficient between groundtruth velocity and inferred velocity
    gdt_velo = adata_copy.layers['groundTruth_velo']
    pred_velo = adata_copy.layers['velocity']
    correlation_velo = calculate_SpearmanMtx(gdt_velo, pred_velo)
    # correlation_velo = []
    # for i in range(pred_velo.shape[1]):
    #     gdt_velo_i = gdt_velo[:,i]
    #     pred_velo_i = pred_velo[:,i]
    #     velo_spear = scipy.stats.spearmanr(pred_velo_i, gdt_velo_i)[0]
    #     correlation_velo.append(velo_spear)
    # correlation_velo_mean = np.mean(correlation_velo)

    # calculate the spearman coefficient between groundtruth parameters and inferred parameters
    gdt_V1 = adata_copy.uns["ground_truth_para"]['gd_V1']
    gdt_K1 = adata_copy.uns["ground_truth_para"]['gd_K1']
    gdt_V2 = adata_copy.uns["ground_truth_para"]['gd_V2']
    gdt_K2 = adata_copy.uns["ground_truth_para"]['gd_K2']

    pred_V1 = adata_copy.uns["velo_para"]['fit_V1']
    pred_K1 = adata_copy.uns["velo_para"]['fit_K1']
    pred_V2 = adata_copy.uns["velo_para"]['fit_V2']
    pred_K2 = adata_copy.uns["velo_para"]['fit_K2']

    # correlation_V1 = calculate_SpearmanMtx(gdt_V1, pred_V1)
    # correlation_K1 = calculate_SpearmanMtx(gdt_K1, pred_K1)
    # correlation_V2 = calculate_SpearmanMtx(gdt_V2, pred_V2)
    # correlation_K2 = calculate_SpearmanMtx(gdt_K2, pred_K2)

    print('Correlation between groundTruth_psd and velocity pseudotime:', correlation_psd)
    print('Correlation between groundTruth_velo and inferred velocity:', correlation_velo)
    # print('Correlation between groundTruth_V1 and inferred V1:', correlation_V1)
    # print('Correlation between groundTruth_K1 and inferred K1:', correlation_K1)
    # print('Correlation between groundTruth_V2 and inferred V2:', correlation_V2)
    # print('Correlation between groundTruth_K2 and inferred K2:', correlation_K2)

    # correlation_res = [correlation_psd, correlation_velo, correlation_V1, correlation_K1, correlation_V2, correlation_K2]
    correlation_res = [correlation_psd, correlation_velo]
    return correlation_res

def calculate_mse(adata_velo):

    # calculate the mse between velocity pseudotime and groundtruth latent time
    adata_copy = adata_velo.copy()
    velo_psd = adata_copy.obs['velocity_pseudotime']
    gdt_psd = adata_copy.obs['groundTruth_psd']
    mse_psd = np.mean((gdt_psd - velo_psd) ** 2)

    # calculate the mse between inferred velocity and groundtruth velocity
    gdt_velo = adata_copy.layers['groundTruth_velo']
    pred_velo = adata_copy.layers['velocity']
    mse_velo = np.mean((gdt_velo - pred_velo) ** 2)

    # calculate the spearman coefficient between groundtruth parameters and inferred parameters
    gdt_V1 = adata_copy.uns["ground_truth_para"]['gd_V1']
    gdt_K1 = adata_copy.uns["ground_truth_para"]['gd_K1']
    gdt_V2 = adata_copy.uns["ground_truth_para"]['gd_V2']
    gdt_K2 = adata_copy.uns["ground_truth_para"]['gd_K2']

    pred_V1 = adata_copy.uns["velo_para"]['fit_V1']
    pred_K1 = adata_copy.uns["velo_para"]['fit_K1']
    pred_V2 = adata_copy.uns["velo_para"]['fit_V2']
    pred_K2 = adata_copy.uns["velo_para"]['fit_K2']

    # mse_V1 = np.mean((gdt_V1 - pred_V1) ** 2)
    # mse_K1 = np.mean((gdt_K1 - pred_K1) ** 2)
    # mse_V2 = np.mean((gdt_V2 - pred_V2) ** 2)
    # mse_K2 = np.mean((gdt_K2 - pred_K2) ** 2)

    print('MSE between groundTruth pseudotime and inferred pseudotime is:', mse_psd)
    print('MSE between groundTruth velocity and inferred velocity is:', mse_velo)
    # print('MSE between groundTruth V1 and inferred V1 is:', mse_V1)
    # print('MSE between groundTruth K1 and inferred K1 is:', mse_K1)
    # print('MSE between groundTruth V2 and inferred V2 is:', mse_V2)
    # print('MSE between groundTruth K2 and inferred K2 is:', mse_K2)

    # mse_res = [mse_psd, mse_velo, mse_V1, mse_K1, mse_V2,mse_K2]
    mse_res = [mse_psd, mse_velo]
    return mse_res

    # if adata_spa.obs['groundTruth_psd'].isna().any():
    #     gd_nonan = np.where(~adata_spa.obs['groundTruth_psd'].isna())[0]
    #     velo_psd = adata_spa.obs['velocity_pseudotime'][gd_nonan]
    #     # velo_psd_nonan = np.where(~velo_psd.isna())[0]
    #
    #     fit_t_new = adata_spa.obs['fit_t'][gd_nonan]
    #     gdt_psd_new = adata_spa.obs['groundTruth_psd'][gd_nonan]
    #     velo_psd_new = velo_psd[gd_nonan]
    #
    #     correlation_fit = scipy.stats.spearmanr(fit_t_new, velo_psd_new)[0]
    #     correlation_groundtruth = scipy.stats.spearmanr(gdt_psd_new, velo_psd_new)[0]
    #     correlation_fit_with_gdt = scipy.stats.spearmanr(fit_t_new, gdt_psd_new)[0]
    #
    # elif adata_spa.obs['velocity_pseudotime'].isna().any():
    #     velo_psd = adata_spa.obs['velocity_pseudotime']
    #     velo_psd_nonan = np.where(~velo_psd.isna())[0]
    #
    #     fit_t_new = adata_spa.obs['fit_t'][velo_psd_nonan]
    #     gdt_psd_new = adata_spa.obs['groundTruth_psd'][velo_psd_nonan]
    #     velo_psd_new = velo_psd[velo_psd_nonan]
    #
    #     correlation_fit = scipy.stats.spearmanr(fit_t_new, velo_psd_new)[0]
    #     correlation_groundtruth = scipy.stats.spearmanr(gdt_psd_new, velo_psd_new)[0]
    #     correlation_fit_with_gd = scipy.stats.spearmanr(fit_t_new, gdt_psd_new)[0]
    #
    # else:
    #     velo_psd = adata_spa.obs['velocity_pseudotime']
    #     fit_t = adata_spa.obs['fit_t']
    #     gdt_psd = adata_spa.obs['groundTruth_psd']
    #
    #     correlation_fit = scipy.stats.spearmanr(fit_t, velo_psd)[0]
    #     correlation_groundtruth = scipy.stats.spearmanr(gdt_psd, velo_psd)[0]
    #     correlation_fit_with_gd = scipy.stats.spearmanr(fit_t, gdt_psd)[0]

    # print('Correlation between fit_t and velocity pseudotime:', correlation_fit)
    # print('Correlation between fit_t and groundTruth_psd:', correlation_fit_with_gd)

def summary_scores(all_scores):
    """Summarize group scores.

    Args:
        all_scores (dict{str,list}):
            {group name: score list of individual cells}.

    Returns:
        dict{str,float}:
            Group-wise aggregation scores.
        float:
            score aggregated on all samples

    """
    sep_scores = {k: np.mean(s) for k, s in all_scores.items() if s}
    overal_agg = np.mean([s for k, s in sep_scores.items() if s])
    return sep_scores, overal_agg


def keep_type(adata, nodes, target, k_cluster):
    """Select cells of targeted type

    Args:
        adata (Anndata):
            Anndata object.
        nodes (list):
            Indexes for cells
        target (str):
            Cluster name.
        k_cluster (str):
            Cluster key in adata.obs dataframe

    Returns:
        list:
            Selected cells.

    """
    return nodes[adata.obs[k_cluster][nodes].values == target]

def _select_emb(adata, k_velocity, x_emb_key):
    if x_emb_key in adata.layers.keys():
        # using embedding from raw space
        x_emb = adata.layers[x_emb_key]
        v_emb = adata.layers[k_velocity]

    else:  # embedding from visualization dimensions
        if x_emb_key.startswith("X_"):
            v_emb_key = k_velocity + x_emb_key[1:]
        else:
            v_emb_key = k_velocity + "_" + x_emb_key
            x_emb_key = "X_" + x_emb_key
        assert x_emb_key in adata.obsm.keys()
        assert v_emb_key in adata.obsm.keys()
        x_emb = adata.obsm[x_emb_key]
        v_emb = adata.obsm[v_emb_key]
    return x_emb, v_emb

def cross_boundary_correctness(
        adata,
        k_cluster,
        k_velocity,
        cluster_edges,
        return_raw=False,
        x_emb="X_umap"
):
    """Cross-Boundary Direction Correctness Score (A->B)

    Args:
        adata (Anndata):
            Anndata object.
        k_cluster (str):
            key to the cluster column in adata.obs DataFrame.
        k_velocity (str):
            key to the velocity matrix in adata.obsm.
        cluster_edges (list of tuples("A", "B")):
            pairs of clusters has transition direction A->B
        return_raw (bool):
            return aggregated or raw scores.
        x_emb (str):
            key to x embedding for visualization.

    Returns:
        dict:
            all_scores indexed by cluster_edges or mean scores indexed by cluster_edges
        float:
            averaged score over all cells.

    """
    scores = {}
    all_scores = {}

    # if x_emb == "X_umap":
    #     v_emb = adata.obsm['{}_umap'.format(k_velocity)]
    # else:
    #     v_emb = adata.obsm[[key for key in adata.obsm if key.startswith(k_velocity)][0]]

    x_emb, v_emb = _select_emb(adata, k_velocity, x_emb)

    for u, v in cluster_edges:
        sel = adata.obs[k_cluster] == u
        nbs = adata.uns['neighbors']['indices'][sel]  # [n * 30]

        boundary_nodes = map(lambda nodes: keep_type(adata, nodes, v, k_cluster), nbs)
        x_points = x_emb[sel]
        x_velocities = v_emb[sel]

        type_score = []
        for x_pos, x_vel, nodes in zip(x_points, x_velocities, boundary_nodes):
            if len(nodes) == 0: continue

            position_dif = x_emb[nodes] - x_pos
            dir_scores = cosine_similarity(position_dif, x_vel.reshape(1, -1)).flatten()
            type_score.append(np.mean(dir_scores))

        scores[(u, v)] = np.mean(type_score)
        all_scores[(u, v)] = type_score

    if return_raw:
        return all_scores

    return scores, np.mean([sc for sc in scores.values()])


def inner_cluster_coh(adata, k_cluster, k_velocity, return_raw=False):
    """In-cluster Coherence Score.

    Args:
        adata (Anndata):
            Anndata object.
        k_cluster (str):
            key to the cluster column in adata.obs DataFrame.
        k_velocity (str):
            key to the velocity matrix in adata.obsm.
        return_raw (bool):
            return aggregated or raw scores.

    Returns:
        dict:
            all_scores indexed by cluster_edges mean scores indexed by cluster_edges
        float:
            averaged score over all cells.

    """
    clusters = np.unique(adata.obs[k_cluster])
    scores = {}
    all_scores = {}

    for cat in clusters:
        sel = adata.obs[k_cluster] == cat
        nbs = adata.uns['neighbors']['indices'][sel]
        same_cat_nodes = map(lambda nodes: keep_type(adata, nodes, cat, k_cluster), nbs)

        velocities = adata.layers[k_velocity]
        cat_vels = velocities[sel]
        cat_score = [cosine_similarity(cat_vels[[ith]], velocities[nodes]).mean()
                     for ith, nodes in enumerate(same_cat_nodes)
                     if len(nodes) > 0]
        all_scores[cat] = cat_score
        scores[cat] = np.mean(cat_score)

    if return_raw:
        return all_scores

    return scores, np.mean([sc for sc in scores.values()])


def evaluate(
        adata,
        cluster_edges,
        k_cluster,
        k_velocity="velocity",
        x_emb="X_umap",
        verbose=True
):
    """Evaluate velocity estimation results using 5 metrics.

    Args:
        adata (Anndata):
            Anndata object.
        cluster_edges (list of tuples("A", "B")):
            pairs of clusters has transition direction A->B
        k_cluster (str):
            key to the cluster column in adata.obs DataFrame.
        k_velocity (str):
            key to the velocity matrix in adata.obsm.
        x_emb (str):
            key to x embedding for visualization.

    Returns:
        dict:
            aggregated metric scores.

    """
    crs_bdr_crc = cross_boundary_correctness(adata, k_cluster, k_velocity, cluster_edges, True, x_emb)
    ic_coh = inner_cluster_coh(adata, k_cluster, k_velocity, True)

    if verbose:
        print("# Cross-Boundary Direction Correctness (A->B)\n{}\nTotal Mean: {}".format(*summary_scores(crs_bdr_crc)))
        print("# In-cluster Coherence\n{}\nTotal Mean: {}".format(*summary_scores(ic_coh)))

    return {
        "Cross-Boundary Direction Correctness (A->B)": crs_bdr_crc,
        "In-cluster Coherence": ic_coh,
    }
