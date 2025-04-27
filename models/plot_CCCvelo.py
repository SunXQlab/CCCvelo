import torch
import numpy as np

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import scvelo as scv
import matplotlib.cm as cm
import scipy
import os

# for simulation data
def plot_gene_dynamic_v2(adata, model, plt_path):
    latent_time = model.assign_latenttime()[0]
    gene_obs = adata.layers['Imputate']
    gene_pre = model.net_z()[0]
    gene_pre = torch.where(gene_pre > 0, gene_pre, torch.full_like(gene_pre, 0)).detach()  # 2000*2

    # 主循环，遍历每个基因
    for i in range(gene_pre.shape[1]):
        # 创建新的绘图窗口
        plt.figure(figsize=(6, 4))
        # 绘制预测曲线
        lab, = plt.plot(gene_pre[:, i], label=f'Gene {i + 1} Prediction')
        # 绘制实际观测值的散点图
        plt.scatter(latent_time, gene_obs[:, i], color='red', s=1, label=f'Gene {i + 1} Observation')
        # 添加标题和坐标轴标签
        plt.title('Gene Dynamics')
        plt.xlabel('Latent Time')
        plt.ylabel('Gene Expression')
        # 添加图例
        plt.legend(loc='best', fontsize='small')  #  fontsize=10
        # 保存图像
        plt.savefig(os.path.join(plt_path, f"gene{i + 1}_dynamics_cluster.pdf"), bbox_inches='tight', dpi=200)
        plt.close()

def plot_gene_dynamic(adata, model, save_path):
    # data prepare
    latent_time = model.assign_latenttime()[0]
    gene_obs = adata.layers['Imputate']
    gene_pre = model.net_z()[0]
    gene_pre = torch.where(gene_pre > 0, gene_pre, torch.full_like(gene_pre, 0)).detach()  # 2000*2
    adata.obs['Cluster'] = adata.obs['Cluster'].astype('category')
    cluster_codes = adata.obs['Cluster'].cat.codes.values

    # color = ['green', 'blue', 'orange']
    cluster = np.unique(cluster_codes)
    num_clusters = len(cluster)
    # colormap = cm.get_cmap('viridis', num_clusters)
    # color_scatter = [colormap(i) for i in range(num_clusters)]
    color_scatter = adata.uns['Cluster_colors']
    # color_scatter = [np.array([31, 119, 180]) / 255,
    #                  np.array([255, 127, 14]) / 255,
    #                  np.array([44, 160, 44]) / 255,
    #                  np.array([214, 39, 40]) / 255]


    for i in range(gene_pre.shape[1]):
        plt.figure(figsize=(6, 4))
        labels = []
        labels_ins = []
        for id, num in zip(cluster, cluster_codes):
            idx = np.where(cluster_codes == id)[0]
            x = latent_time[idx]
            y = gene_obs[idx, i]
            plt.scatter(x, y, color=color_scatter[id], s=2)
            # lab, = plt.plot(gene_pre[:, i], c=color[i])
            lab, = plt.plot(gene_pre[:, i],color='blue')
            if id == 0:  # 只添加一次图例对象
                labels_ins.append(lab)
                labels.append("Gene %s" % (i + 1))

        plt.title('Gene dynamics cluster')
        plt.legend(handles=labels_ins, labels=labels)
        plt.savefig(save_path + "gene%s_dynamics_cluster.png" % i, bbox_inches='tight', dpi=200)
        plt.close()

def plotLossAdam(loss_adam, save_path):
    torch.save(loss_adam, save_path + "Loss_adam.pt")
    plt.figure(figsize=(10, 4))
    plt.plot(loss_adam)  # linestyle为线条的风格（共五种）,color为线条颜色
    # plt.title('Loss of Adam at time %s'%(timepoint+1))
    plt.xlabel('iteration')
    plt.ylabel('loss of Adam')
    plt.savefig(save_path + "Loss_adam.png", bbox_inches='tight', dpi=600)
    plt.close()

def plot_velocity_streamline_v2(adata, basis, vkey, xkey, root_key, density, smooth, cutoff_perc,plt_path,save_name):

    adata_copy = adata.copy()

    scv.tl.velocity_graph(adata_copy, basis=basis, vkey=vkey, xkey=xkey)
    scv.pl.velocity_embedding_stream(adata_copy, basis=basis, density=density, smooth=smooth, cutoff_perc=cutoff_perc
                                     , save=plt_path + basis+'_embedding_stream_with_'+save_name)
    if root_key:
        scv.tl.velocity_pseudotime(adata_copy, root_key=adata_copy.uns['iroot'])  # root_key=adata_copy.uns['iroot']
        scv.pl.scatter(adata_copy, basis=basis, color='velocity_pseudotime', cmap='gnuplot',
                       save=plt_path + basis + '_velocity_pseudotime_with_set_root'+save_name)
    else:
        scv.tl.velocity_pseudotime(adata_copy)  # root_key=adata_copy.uns['iroot']
        scv.pl.scatter(adata_copy, basis=basis, color='velocity_pseudotime', cmap='gnuplot',
                       save=plt_path + basis + '_velocity_pseudotime_with_'+save_name)

    # torch.save(adata_copy, plt_path + "CCCvelo_results.pt")
    # calculate
    correlation, _ = scipy.stats.spearmanr(adata_copy.obs['fit_t'], adata_copy.obs['velocity_pseudotime'])
    print('the correlation between fit_t and velocity_pesudotime is:', correlation)

    correlation, _ = scipy.stats.spearmanr(adata_copy.obs['groundTruth_psd'], adata_copy.obs['velocity_pseudotime'])
    print('the correlation between groundTruth_psd and velocity_pesudotime is:', correlation)


def plot_velocity_streamline(adata, basis, vkey, xkey, root_key, density, smooth, cutoff_perc,plt_path,save_name):

    adata_copy = adata.copy()

    scv.tl.velocity_graph(adata_copy, basis=basis, vkey=vkey, xkey=xkey)
    scv.pl.velocity_embedding_stream(adata_copy, basis=basis, density=density, smooth=smooth, cutoff_perc=cutoff_perc,
                                     color='Cluster',legend_loc='right margin',
                                     save=plt_path + basis+'_streamline_'+save_name)
    # scv.pl.velocity_embedding(adata_copy, basis=basis, color='Cluster', dpi=300, arrow_size=2, arrow_length=2,
    #                           save=plt_path + basis+'_velocity_embedding_'+save_name)
    scv.pl.scatter(adata_copy, basis=basis, color='fit_t', color_map='gnuplot',save=plt_path + basis + '_latent_time_'+save_name)
    if root_key:
        scv.tl.velocity_pseudotime(adata_copy, root_key=adata_copy.uns['iroot'])  # root_key=adata_copy.uns['iroot']
        scv.pl.scatter(adata_copy, basis=basis, color='velocity_pseudotime', cmap='gnuplot',
                       save=plt_path + basis + '_velocity_psd_with_set_root_'+save_name)
        # torch.save(adata_copy, plt_path + basis+ "_CCCvelo_results_with_root.pt")
    else:
        scv.tl.velocity_pseudotime(adata_copy)  # root_key=adata_copy.uns['iroot']
        scv.pl.scatter(adata_copy, basis=basis, color='velocity_pseudotime', cmap='gnuplot',
                       save=plt_path + basis + '_velocity_psd_with_'+save_name)
        # torch.save(adata_copy, plt_path + basis + "_CCCvelo_results.pt")

    # torch.save(adata_copy, plt_path + "CCCvelo_results.pt")
    # calculate
    correlation, _ = scipy.stats.spearmanr(adata_copy.obs['fit_t'], adata_copy.obs['velocity_pseudotime'])
    print('the correlation between fit_t and velocity_pesudotime is:', correlation)
    # scv.pl.heatmap(adata_copy, var_names=adata_copy.var_names, sortby='velocity_pseudotime', col_color='Cluster', n_convolve=100,
    #                save=plt_path + basis+'_heatmap_TGs_with_psd')
    # scv.pl.heatmap(adata_copy, var_names=adata_copy.var_names, sortby='fit_t', col_color='Cluster', n_convolve=100,
    #                save=plt_path + basis+'_heatmap_TGs_with_fit_t')

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory created at: {path}")

def plot_gene_expr_with_latenttime(adata_velo, velo_psd_new, results_path):
    plt_path = os.path.join(results_path, "figure")
    create_directory(plt_path)

    gene_obs = adata_velo.layers['Imputate']
    for i in range(gene_obs.shape[1]):
        plt.figure(figsize=(6, 4))
        plt.scatter(velo_psd_new, gene_obs[:, i], color='blue', s=2, label=f'Gene {i + 1} Observation')
        plt.title('Gene Dynamics')
        plt.xlabel('Latent Time')
        plt.ylabel('Gene Expression')
        plt.legend(loc='best', fontsize='small')
        plt.savefig(os.path.join(plt_path, f"gene{i + 1}_dynamics_with_velo_psd.png"), bbox_inches='tight', dpi=200)
        plt.close()
    # print(f"Gene dynamics plots saved at: {plt_path}")



