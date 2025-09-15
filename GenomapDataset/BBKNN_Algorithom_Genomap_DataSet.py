
import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
    bars = plt.bar(metrics, values, width=0.3)
    plt.ylabel("Score")
    plt.title(f"ARI_RI_SC_Metrics_For_{filename_prefix}")
    ymin = min(0.0, min(values) - 0.1)
    ymax = max(1.0, max(values) + 0.1)
    plt.ylim(ymin, ymax)
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

def load_data():
    """Loads all .mat files into a list of AnnData objects and loads labels."""
    adatas = []
    for file in DATA_FILES:
        data_dict = sio.loadmat(os.path.join(DATA_FOLDER, file))
        key = next((k for k in data_dict if 'data' in k.lower()), None)
        if key:
            # Perform basic preprocessing on each batch individually
            X = data_dict[key]
            if not isinstance(X, np.ndarray):
                X = np.array(X)
            adata_batch = ad.AnnData(X)
            sc.pp.normalize_total(adata_batch, target_sum=1e4)
            sc.pp.log1p(adata_batch)
            adatas.append(adata_batch)
        else:
            logging.warning(f"No valid data key found in {file}")

    class_labels = sio.loadmat(os.path.join(DATA_FOLDER, CLASS_LABEL_FILE))['classLabel'].squeeze()
    return adatas, class_labels

def tune_leiden_to_target_k(adata, target_k, flavor="igraph", directed=False, max_iter=20,
                            lo=0.05, hi=2.0, tol=0):
    """
    Binary-search-like tuning of Leiden resolution so that the number of clusters ~ target_k.
    Stops when cluster count == target_k or iterations exhausted.
    Returns the final resolution used.
    """
    resolution = 0.5
    best_res = resolution
    best_diff = float('inf')

    for _ in range(max_iter):
        sc.tl.leiden(adata, resolution=resolution, flavor=flavor, directed=directed)
        k = adata.obs['leiden'].nunique()
        diff = abs(k - target_k)
        if diff < best_diff:
            best_diff = diff
            best_res = resolution
        if diff <= tol:
            break
        # adjust bounds
        if k > target_k:
            hi = resolution
            resolution = (lo + resolution) / 2.0
        else:
            lo = resolution
            resolution = (resolution + hi) / 2.0

    # run once more at best_res to set labels
    sc.tl.leiden(adata, resolution=best_res, flavor=flavor, directed=directed)
    return best_res

# --- Main Analysis Function ---
def main():
    """Main function to run the BBKNN analysis workflow (updated)."""
    sc.settings.verbosity = 2
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # 1. Load Data
    logging.info("Loading and preparing data...")
    adatas, class_labels = load_data()

    # Concatenate all batches into a single AnnData object with a batch key
    adata = ad.concat(adatas, label="batch", keys=[f"batch_{i}" for i in range(len(adatas))])
    adata.obs_names_make_unique()  # fix duplicate obs names warning

    # 2. Preprocessing and Integration with BBKNN
    logging.info("Selecting HVGs (per batch) and running PCA...")
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, batch_key='batch')
    adata = adata[:, adata.var.highly_variable].copy()
    sc.tl.pca(adata, svd_solver='arpack', n_comps=50)
    bbknn.bbknn(adata, batch_key='batch')

    logging.info("Building BBKNN graph...")
    bbknn.bbknn(adata, batch_key='batch')  # modifies adata.obsp and adata.uns

    # 3. Visualization embedding (UMAP) to align with other pipelines for evaluation
    logging.info("Computing UMAP for evaluation and visualization...")
    sc.tl.umap(adata)  # uses BBKNN neighbor graph

    # 4. Attach true class labels (truncate if needed)
    limit = min(adata.n_obs, len(class_labels))
    if adata.n_obs != len(class_labels):
        logging.warning(f"Number of cells ({adata.n_obs}) != number of labels ({len(class_labels)}). Using first {limit}.")
        adata = adata[:limit, :].copy()
    adata.obs['class'] = pd.Categorical(class_labels[:adata.n_obs])

    # 5. Tune Leiden to match number of true classes
    n_true = int(len(np.unique(adata.obs['class'])))
    logging.info(f"Tuning Leiden resolution to target {n_true} clusters...")
    final_res = tune_leiden_to_target_k(adata, target_k=n_true, flavor="igraph", directed=False)
    logging.info(f"Final Leiden resolution used: {final_res:.3f}; clusters found: {adata.obs['leiden'].nunique()}")

    # 6. Optional: t-SNE for plotting only
    sc.tl.tsne(adata, use_rep='X_pca')

    # Plotting by true labels
    plot_filename_prefix = "BBKNN_Algorithm_Genomap_Dataset"
    plot_embedding(adata.obsm['X_umap'], adata.obs['class'].astype('category').cat.codes, "UMAP", f"UMAP_Plot_{plot_filename_prefix}")
    plot_embedding(adata.obsm['X_tsne'], adata.obs['class'].astype('category').cat.codes, "t-SNE", f"TSNE_Plot_{plot_filename_prefix}")

    # 7. Evaluation: use UMAP embedding to be comparable with other pipelines
    logging.info("Calculating evaluation metrics on UMAP embedding...")
    true_labels = adata.obs['class']
    predicted_labels = adata.obs['leiden']

    ari = adjusted_rand_score(true_labels, predicted_labels)
    rand = rand_score(true_labels, predicted_labels)
    silhouette = silhouette_score(adata.obsm['X_pca'], predicted_labels)
    logging.info(f"ARI: {ari:.3f}, Rand Index: {rand:.3f}, Silhouette Score: {silhouette:.3f}")

    plot_metrics_bar(ari, rand, silhouette, plot_filename_prefix)
    save_metrics_csv(ari, rand, silhouette, f"CSV_{plot_filename_prefix}")

    logging.info("BBKNN analysis complete.")

if __name__ == "__main__":
    main()
