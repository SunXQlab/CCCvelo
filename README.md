# CCCvelo: Decoding dynamic cell-cell communication-driven cell state transitions in spatial transcriptomics

<p align="center">
  <img src="https://github.com/SunXQlab/CCCvelo/blob/main/fig1.framework.png">
</p>

CCCvelo is a computational framework designed to reconstruct CCC-driven CST dynamics by jointly optimizing a dynamic CCC signaling network and a latent CST clock through a multiscale, nonlinear network kinetics model. CCCvelo can estimated RNA velocity, cell pseudotime, pseudo-temporal dynamics of TGs’ expressions or TFs’ activities, and the cell state-specific multilayer signaling network of CCC. These functionalities enable the reconstruction of spatiotemporal trajectories of cells while simultaneously capturing dynamic cellular communication driving CST. 

CCCvelo employs several visualization strategies to facilitate the analysis of CCC-driven CST dynamics. These visualizations mainly include velocity streamlines illustrating CST trajectories, heatmap visualizations of gene expression and TF activity along pseudotime, and multilayer network plots of CCC displaying the signaling paths from upstream LR pairs to TFs and then to the downstream TGs.

* `0_preprocess_inputData.R` contains the scripts to prepare the condinate ligands, receptors, and feature genes for constructing multilayer signaling network <br>
* `1_run_MLnet_demo.py` contains the scripts to construct mulitlayer signling network and calculate the LR signaling strength <br>
* `2_run_CCCvelo_demo.py` contains the scripts to infer the CCC-driven RNA velocity <br>



