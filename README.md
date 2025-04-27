# CCCvelo: Decoding dynamic cell-cell communication-driven cell state transitions in spatial transcriptomics

<p align="center">
  <img src="https://github.com/SunXQlab/CCCvelo/blob/main/fig1.framework.png">
</p>

CCCvelo is a computational framework designed to reconstruct CCC-driven CST dynamics by jointly optimizing a dynamic CCC signaling network and a latent CST clock through a multiscale, nonlinear network kinetics model. CCCvelo can estimated RNA velocity, cell pseudotime, pseudo-temporal dynamics of TGs’ expressions or TFs’ activities, and the cell state-specific multilayer signaling network of CCC. These functionalities enable the reconstruction of spatiotemporal trajectories of cells while simultaneously capturing dynamic cellular communication driving CST. 

CCCvelo employs several visualization strategies to facilitate the analysis of CCC-driven CST dynamics. These visualizations mainly include velocity streamlines illustrating CST trajectories, heatmap visualizations of gene expression and TF activity along pseudotime, and multilayer network plots of CCC displaying the signaling paths from upstream LR pairs to TFs and then to the downstream TGs.

* `0_preprocess_inputData.R` contains the scripts to prepare the condinate ligands, receptors, and feature genes for constructing multilayer signaling network <br>
* `1_run_MLnet_demo.py` contains the scripts to construct mulitlayer signling network and calculate the LR signaling strength <br>
* `2_run_CCCvelo_demo.py` contains the scripts to infer the CCC-driven RNA velocity <br>

# Environment
anndata             0.9.2         
h5py                3.11.0
matplotlib          3.7.5
mpmath              1.3.0
networkx            3.1
numba               0.58.1
numpy               1.24.4
pandas              2.0.3
pip                 23.2.1
python-dateutil     2.9.0.post0
python-utils        3.8.2
scanpy              1.9.8
scipy               1.10.1
scvelo              0.3.2
seaborn             0.13.2
setuptools          68.2.0
threadpoolctl       3.5.0
torch               2.0.1





