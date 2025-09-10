import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score, silhouette_score, rand_score
import anndata as ad
import scipy.io as sio
import scanpy as sc
import bbknn

# --- Configuration ---
DATA_FOLDER = './Dataset'
OUTPUT_FOLDER = './Figures_CSVs'
DATA_FILES = ['dataBaronX.mat', 'dataMuraroX.mat', 'dataScapleX.mat', 'dataWangX.mat', 'dataXinX.mat']
CLASS_LABEL_FILE = 'classLabel.mat'
BATCH_LABEL_FILE = 'batchLabel.mat'

# --- Helper Functions (Standardized) ---
def plot_embedding(X, y, title, filename):
    """Generates and saves a 2D scatter plot of an embedding."""
    X, y = np.asarray(X), np.asarray(y)
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='jet', s=18)
    plt.xlabel(f'{title} Dimension 1', fontsize=8)
    plt.ylabel(f'{title} Dimension 2', fontsize=8)
    plt.title(filename, fontsize=10)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, f'{filename}.png'), dpi=300)
    plt.close()

def plot_metrics_bar(ari, rand, silhouette, filename_prefix):
    """Generates and saves a bar chart of evaluation metrics."""
    metrics = ["Adjusted Rand Index (ARI)", "Rand Index", "Silhouette Score"]
    values = [ari, rand, silhouette]
    plt.figure(figsize=(8, 6))
    bars = plt.bar(metrics, values, color=['coral', 'lightgreen', 'blue'], width=0.3)
    plt.ylabel("Score")
    plt.title(f"ARI_RI_SC_Metrics_For_{filename_prefix}")
    plt.ylim(min(0, min(values) - 0.1), max(1, max(values) + 0.1))
    plt.xticks(rotation=45)
    plt.tight_layout()
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f'{value:.2f}', ha='center',
                 color='black', fontsize=12, fontweight='bold')
    plt.savefig(os.path.join(OUTPUT_FOLDER, f"ARI_RI_SC_Metrics_For_{filename_prefix}.png"), dpi=300)
    plt.close()

def save_metrics_csv(ari, rand, silhouette, filename):
    """Saves evaluation metrics to a CSV file."""
    df = pd.DataFrame([{"ARI": ari, "Rand": rand, "Silhouette": silhouette}])
    df.to_csv(os.path.join(OUTPUT_FOLDER, f'{filename}.csv'), index=False)

def load_and_prepare_data():
    """Loads all .mat files, combines them into a single AnnData object, and performs initial preprocessing."""
    adatas = []
    for file in DATA_FILES:
        data_dict = sio.loadmat(os.path.join(DATA_FOLDER, file))
        key = next((k for k in data_dict if 'data' in k.lower()), None)
        if key:
            adatas.append(ad.AnnData(data_dict[key]))
        else:
            logging.warning(f"No valid data key found in {file}")

    class_labels = sio.loadmat(os.path.join(DATA_FOLDER, CLASS_LABEL_FILE))['classLabel'].squeeze()
    batch_labels = sio.loadmat(os.path.join(DATA_FOLDER, BATCH_LABEL_FILE))['batchLabel'].squeeze()

    adata = ad.concat(adatas, label="batch_id", keys=[str(i) for i in range(len(adatas))])
    
    # Ensure labels match the number of cells
    limit = min(adata.n_obs, len(class_labels), len(batch_labels))
    adata = adata[:limit, :].copy()
    adata.obs['class'] = class_labels[:limit]
    adata.obs['batch'] = batch_labels[:limit]

    # --- THIS IS THE FIX ---
    # Convert 'batch' column to category dtype as required by sc.pp.highly_variable_genes
    adata.obs['batch'] = adata.obs['batch'].astype('category')
    # ----------------------

    # Standard preprocessing
    #adata.X = SimpleImputer(strategy="mean").fit_transform(adata.X)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
    return adata

# --- Main Analysis Function ---
def main():
    """Main function to run the BBKNN analysis workflow."""
    sc.settings.verbosity = 2
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # 1. Load and Prepare Data (Standardized)
    logging.info("Loading and preparing data...")
    adata = load_and_prepare_data()

    # 2. Preprocessing and Integration with BBKNN
    logging.info("Finding HVGs, running PCA, and BBKNN integration...")
    
    # Find highly variable genes to focus the analysis
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, batch_key='batch')
    # Subset the data to only the highly variable genes
    adata = adata[:, adata.var.highly_variable]

    sc.pp.pca(adata, n_comps=50)
    # BBKNN operates on the PCA embedding to correct batch effects
    bbknn.bbknn(adata, batch_key='batch')

    # 3. Clustering and Visualization (Standardized)
    logging.info("Running Leiden clustering on BBKNN graph...")
    # BBKNN creates a corrected neighborhood graph, which is ideal for Leiden clustering
    sc.tl.leiden(adata, resolution=0.8)

    logging.info("Generating UMAP and t-SNE plots...")
    sc.tl.umap(adata)
    sc.tl.tsne(adata, n_pcs=50) # t-SNE on PCA still useful for comparison

    # Plotting based on true class labels for visual inspection
    plot_embedding(adata.obsm['X_umap'], adata.obs['class'].astype('category').cat.codes, "UMAP", "UMAP_Plot_BBKNN_Algorithm")
    plot_embedding(adata.obsm['X_tsne'], adata.obs['class'].astype('category').cat.codes, "t-SNE", "TSNE_Plot_BBKNN_Algorithm")

    # 4. Evaluation (Standardized)
    logging.info("Calculating evaluation metrics...")
    true_labels = adata.obs['class']
    predicted_labels = adata.obs['leiden']

    ari = adjusted_rand_score(true_labels, predicted_labels)
    rand = rand_score(true_labels, predicted_labels)
    # Silhouette score is calculated on the UMAP embedding, as it reflects the final clustered space
    silhouette = silhouette_score(adata.obsm['X_umap'], predicted_labels)
    
    logging.info(f"ARI: {ari:.3f}, Rand Index: {rand:.3f}, Silhouette Score: {silhouette:.3f}")

    plot_metrics_bar(ari, rand, silhouette, "BBKNN_Algorithm_With_Genomap_dataset")
    save_metrics_csv(ari, rand, silhouette, "CSV_BBKNN_Algorithm_With_Genomap_dataset")
    
    logging.info("BBKNN analysis complete.")

if __name__ == "__main__":
    main()