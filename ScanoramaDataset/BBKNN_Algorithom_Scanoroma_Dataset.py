import os
import logging
import scanpy as sc
import bbknn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score, rand_score, silhouette_score

# UMPA and t-SNE plotting function
def plot_embedding(X, y, title, filename):
    """
    Plot 2D embedding (e.g., UMAP or t-SNE) and save as PNG.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    if X.shape[1] != 2:
        raise ValueError("Input X must have exactly 2 columns for 2D embedding.")
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='jet', s=18)
    plt.xlabel(f'{title} Dimension 1', fontsize=5)
    plt.ylabel(f'{title} Dimension 2', fontsize=5)
    plt.title(f'{filename}', fontsize=5)
    plt.xticks(fontsize=5)
    plt.yticks(fontsize=5)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(f'./Figures_CSVs/{filename}.png', dpi=300)
    plt.close()

# Metrics plotting function
def plot_metrics_bar(ari, rand, silhouette, filename_prefix):
    """
    Plot ARI, Rand Index, and Silhouette Score as a bar chart and save as PNG.
    """
    metrics = ["Adjusted Rand Index (ARI)", "Rand Index", "Silhouette Score"]
    values = [ari, rand, silhouette]

    plt.figure(figsize=(8, 6))
    bars = plt.bar(metrics, values, color=['coral', 'lightgreen', 'blue'], width=0.3)
    plt.ylabel("Score")
    plt.title(f"ARI_RI_SC_Metrics_For_{filename_prefix}")
    plt.ylim(min(0, min(values) - 0.1), max(1, max(values) + 0.1))
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Annotate bars
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f'{value:.2f}',
                 ha='center', color='black', fontsize=12, fontweight='bold')

    plt.savefig(f"./Figures_CSVs/ARI_RI_SC_Metrics_For_{filename_prefix}.png", dpi=300)
    plt.close()

# Metrics saving function
def save_metrics_csv(ari, rand, silhouette, filename):
    pd.DataFrame([{"ARI": ari, "Rand": rand, "Silhouette": silhouette}])\
      .to_csv(f"./Figures_CSVs/{filename}.csv", index=False)

def main():
    # Setup logging and directories
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    os.makedirs('./Figures_CSVs', exist_ok=True)

    # Load dataset
    data_folder = './Dataset'
    try:
        adata = sc.read(os.path.join(data_folder, 'human_pancreas_norm_complexBatch.h5ad'))
    except FileNotFoundError as e:
        logging.error(f"Data file not found: {e}")
        return
    
    adata.var_names_make_unique()

    # Filter out cells with no gene counts to prevent NaN errors
    sc.pp.filter_cells(adata, min_genes=1)

    # --- CORRECTED PREPROCESSING WORKFLOW ---
    # 1. Normalize and log-transform the data
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # --- Explicitly handle any remaining NaNs ---
    # This ensures the data is clean before finding variable genes.
    if np.isnan(adata.X).any():
        adata.X[np.isnan(adata.X)] = 0
    # -----------------------------------------------------------

    # 2. Identify Highly Variable Genes (HVGs) across batches
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, batch_key='tech')
    adata = adata[:, adata.var.highly_variable]

    # 3. Scale data to unit variance and zero mean, clipping at max_value 10
    sc.pp.scale(adata, max_value=10)
    
    # 4. Run PCA on the scaled HVGs
    #sc.pp.pca(adata)
    sc.tl.pca(adata, svd_solver='arpack', n_comps=50)

    # --- Batch Correction and Clustering ---
    # BBKNN batch correction modifies the neighborhood graph
    bbknn.bbknn(adata, batch_key='tech')

    # Leiden clustering is performed on the BBKNN-corrected graph
    sc.tl.leiden(adata, resolution=0.5)
    #sc.tl.leiden(adata)

    # UMAP and t-SNE embeddings are also based on the corrected graph
    sc.tl.umap(adata)
    sc.tl.tsne(adata)
    
    # --- Plotting and Evaluation ---
    true_labels = adata.obs['celltype'].astype('category').cat.codes.to_numpy()
    cluster_labels = adata.obs['leiden'].astype('category').cat.codes.to_numpy()

    plot_embedding(adata.obsm['X_umap'], true_labels, "UMAP", "UMAP_Plot_Using_BBKNN_Algorithom_With_Scanorama_Dataset")
    plot_embedding(adata.obsm['X_tsne'], true_labels, "t-SNE", "t-SNE_Plot_Using_BBKNN_Algorithom_With_Scanorama_Dataset")

    # Clustering evaluation
    ari = adjusted_rand_score(true_labels, cluster_labels)
    rand = rand_score(true_labels, cluster_labels)
    silhouette = silhouette_score(adata.obsm['X_umap'], cluster_labels)
  
    # Plot metrics bar chart and save it to CSV
    plot_metrics_bar(ari, rand, silhouette, "BBKNN_Algorithm_With_Scanorama_data")
    save_metrics_csv(ari, rand, silhouette, "CVS_BBKNN_Algorithm_With_Scanorama_data")

if __name__ == "__main__":
    main()