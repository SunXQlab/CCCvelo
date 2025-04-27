import warnings
import psutil
import anndata as ad

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="tkinter")

from models.train_CCCvelo import *
from models.plot_CCCvelo import *
from models.preprocess_CCCvelo import *
from models.evaluation_Metric import *
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import roc_curve, precision_recall_curve

# CUDA support
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
warnings.filterwarnings("ignore")

def main(seed, inputDir):
    # Step 1: Load data
    adata = ad.read_h5ad(os.path.join(inputDir,'adata_raw.h5ad'))
    results_path = os.path.join(inputDir, f"Output/")
    create_directory(results_path)

    files = {
    'LR_link_file': 'LR_links.csv',
    'TFTG_link_file': 'TFTG_links.csv',
    'LRTF_score_file': 'TFLR_score/',
    }

    paths = {key: os.path.join(inputDir, fname) for key, fname in files.items()}
    adata = PrepareInputData(adata,**paths)
    adata = root_cell(adata, select_root='UMAP')
    # sc.pl.embedding(adata, basis="spatial", color="Cluster", s=6, show=True)
    adata.uns['Cluster_colors'] = ['#ff7f0e', '#2ca02c', '#1f77b4', '#d62728']
    torch.save(adata, os.path.join(results_path, "pp_adata.pt"))
 
    print('the shape of TGTF_regulate',adata.varm['TGTF_regulate'].shape)
    print('the shape of TFLR_signaling_score',adata.obsm['TFLR_signaling_score'].shape)

    # Step 2: Train model
    data = PrepareData(adata, hidden_dims=[150, 150, 150])

    model = SpatialVelocity(*data, lr=0.005, Lambda=0.01)
    iteration_adam, loss_adam = model.train(200)
    plotLossAdam(loss_adam, results_path)

    # Extract and save velocity data
    adata_copy = adata[:, adata.var['TGs'].astype(bool)]
    adata_velo = get_raw_velo(adata_copy, model)
    save_model_and_data(model, adata_velo, results_path)

    plt_path = os.path.join(results_path, "figure/")
    create_directory(plt_path)
    # plot_gene_dynamic_v2(adata_velo, model, plt_path)
    plot_gene_dynamic(adata_velo, model, plt_path)

if __name__ == "__main__":
    process = psutil.Process(os.getpid())
    before_memory = process.memory_info().rss / 1024 ** 2  

    seed = 3 # Replace with your seed value
    torch.manual_seed(seed)  
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    random.seed(seed)  
    np.random.seed(seed)  
    start_time = time.time()

    base_path = "E:/CCCvelo/apply_in_cortex/"
    outputDir = os.path.join(base_path, f"Output/RtoPy_test/")

    main(seed, inputDir=outputDir)  

    after_memory = process.memory_info().rss / 1024 ** 2 
    print(f"Using memory is: {after_memory - before_memory} MB")

    end_time = time.time()
    run_time = (end_time - start_time) / 60
    print(f"Running time is: {run_time} mins")


