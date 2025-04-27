# CCCvelo: Decoding dynamic cell-cell communication-driven cell state transitions in spatial transcriptomics

<p align="center">
  <img src="https://github.com/SunXQlab/CCCvelo/blob/main/fig1.framework.png">
</p>

CCCvelo is a computational framework designed to reconstruct CCC-driven CST dynamics by jointly optimizing a dynamic CCC signaling network and a latent cell-state transitions (CST) clock through a multiscale, nonlinear network kinetics model. CCCvelo can estimated RNA velocity, cell pseudotime, pseudo-temporal dynamics of TGs’ expressions or TFs’ activities, and the cell state-specific multilayer signaling network of CCC. These functionalities enable the reconstruction of spatiotemporal trajectories of cells while simultaneously capturing dynamic cellular communication driving CST. CCCvelo employs several visualization strategies to facilitate the analysis of CCC-driven CST dynamics. These visualizations mainly include velocity streamlines illustrating CST trajectories, heatmap visualizations of gene expression and TF activity along pseudotime, and multilayer network plots of CCC displaying the signaling paths from upstream LR pairs to TFs and then to the downstream TGs.

The main features of CCCvelo are：

* (1) the reconstruction of spatiotemporal dynamics of CCC-regulated CSTs within a spatial context <br>
* (2) quantitative ordering of cellular progression states through velocity vector field embedding <br>
* (3) the identification of dynamic rewiring of CCC signaling <br>

# Environment
h5py                3.11.0 <br>
matplotlib          3.7.5 <br>
mpmath              1.3.0 <br>
networkx            3.1 <br>
numba               0.58.1 <br>
numpy               1.24.4 <br>
pandas              2.0.3 <br>
pip                 23.2.1 <br>
python-dateutil     2.9.0.post0 <br>
python-utils        3.8.2 <br>
scanpy              1.9.8 <br>
scipy               1.10.1 <br>
scvelo              0.3.2 <br>
seaborn             0.13.2 <br>
setuptools          68.2.0 <br>
threadpoolctl       3.5.0 <br>
torch               2.0.1 <br>
anndata             0.9.2  <br>     
# Usage
The package CCCvelo can be directly downloaded for usage. 

To learn how to run CCCvelo, Please check the `0_preprocess_inputData.R`, `1_run_MLnet_demo.py`, and  `2_run_CCCvelo_demo.py` files. These files shows the application of CCCvelo on the mouse cortex dataset, which can be download from (https://www.dropbox.com/s/c5tu4drxda01m0u/mousebrain_bin60.h5ad?dl=0). 

* `0_preprocess_inputData.R` contains the scripts to prepare the condinate ligands, receptors, and feature genes for constructing multilayer signaling network <br>
* `1_run_MLnet_demo.py` contains the scripts to construct mulitlayer signling network and calculate the LR signaling strength <br>
* `2_run_CCCvelo_demo.py` contains the scripts to infer the CCC-driven RNA velocity <br>





