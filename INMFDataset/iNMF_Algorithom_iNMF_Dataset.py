import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
from sklearn.decomposition import NMF
from sklearn.metrics import adjusted_rand_score, silhouette_score, rand_score

# --- Configuration ---
#DATA_FILE = './Dataset/pbmcs_ctrl_converted.h5ad'
DATA_FILE = './Dataset/pbmcs_ctrl_annotated.h5ad'
OUTPUT_FOLDER = './Figures_CSVs'
# Assuming the AnnData object has these .obs columns
BATCH_KEY = 'batch'
CELLTYPE_KEY = 'celltype'

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
    adata = sc.read(DATA_FILE)
    adata.obs_names_make_unique()

    # If HVGs already exist in var, use them and skip re-computing
    if "highly_variable" in adata.var:
        adata = adata[:, adata.var["highly_variable"]].copy()
    else:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=2000)
        adata = adata[:, adata.var["highly_variable"]].copy()

    return adata    
# def load_and_prepare_data():
#     """Loads and preprocesses the PBMC dataset."""
#     try:
#         adata = sc.read(DATA_FILE)
#     except FileNotFoundError:
#         logging.error(f"Data file not found at {DATA_FILE}")
#         return None

#     # --- Standard Preprocessing ---
#     sc.pp.normalize_total(adata, target_sum=1e4)
#     sc.pp.log1p(adata)
#     sc.pp.highly_variable_genes(adata, n_top_genes=2000, batch_key=BATCH_KEY)
#     adata = adata[:, adata.var.highly_variable]
#     # NMF requires non-negative data. The log1p data is already non-negative.
#     # No scaling is applied before NMF as it can create negative values.
    
#     # --- Added data scaling ---
#     # This step centers the data, which is a standard practice.
#     #sc.pp.scale(adata, max_value=10) 

#     return adata

# --- Main Analysis Function ---
def main():
    """Main function to run the iNMF analysis workflow."""
    sc.settings.verbosity = 2
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    logging.info("Loading and preparing data...")
    adata = load_and_prepare_data()
    if adata is None:
        return
    if CELLTYPE_KEY not in adata.obs.columns:
        logging.error(f"Required column '{CELLTYPE_KEY}' not in adata.obs.")
        return

    # 2. 
    # Shift data to be non-negative after scaling
    min_val = adata.X.min()
    if min_val < 0:
        logging.warning(f"Negative values found after scaling. Shifting data to be non-negative for NMF.")
        adata.X = adata.X - min_val
    
    # Dimensionality reduction using NMF
    logging.info("Applying iNMF...")
    n_components = 30
    model = NMF(n_components=n_components, init='nndsvda', random_state=42)
    adata.obsm['X_nmf'] = model.fit_transform(adata.X)

    # 3. Clustering and Visualization
    logging.info("Building neighborhood graph, running Leiden, and generating embeddings...")
    sc.pp.neighbors(adata, use_rep='X_nmf', n_neighbors=15)
    sc.tl.leiden(adata, resolution=0.5)
    sc.tl.umap(adata)
    sc.tl.tsne(adata, use_rep='X_nmf') # Add t-SNE calculation
    
    true_labels_cat = adata.obs[CELLTYPE_KEY].astype('category').cat.codes

    plot_filename_prefix = "iNMF_Algorithm_iNMF_Dataset"
    plot_embedding(adata.obsm['X_umap'], true_labels_cat, "UMAP", f"UMAP_Plot_{plot_filename_prefix}")
    plot_embedding(adata.obsm['X_tsne'], true_labels_cat, "t-SNE", f"TSNE_Plot_{plot_filename_prefix}")

    # 4. Evaluation
    logging.info("Calculating evaluation metrics...")
    true_labels = adata.obs[CELLTYPE_KEY].values
    predicted_labels = adata.obs['leiden'].values

    ari = adjusted_rand_score(true_labels, predicted_labels)
    rand = rand_score(true_labels, predicted_labels)
    silhouette = silhouette_score(adata.obsm['X_nmf'], predicted_labels)     
    logging.info(f"ARI: {ari:.3f}, Rand Index: {rand:.3f}, Silhouette Score: {silhouette:.3f}")

    plot_metrics_bar(ari, rand, silhouette, f"{plot_filename_prefix}")
    save_metrics_csv(ari, rand, silhouette, f"CSV_{plot_filename_prefix}")

    logging.info("iNMF analysis complete.")

if __name__ == "__main__":
    main()