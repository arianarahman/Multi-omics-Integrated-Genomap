import os
import logging
import scanpy as sc
import bbknn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Patch
from sklearn.metrics import adjusted_rand_score, rand_score, silhouette_score

# --- Configuration ---
DATA_FILE = './Dataset/human_pancreas_norm_complexBatch.h5ad'
OUTPUT_FOLDER = './Figures_CSVs'
BATCH_KEY = 'batch'
CELLTYPE_KEY = 'celltype'

# --- Helper Functions (Standardized) ---
def create_custom_colormap(labels, colors):
    """
    Creates a custom color mapping dictionary and a list of legend patches.
    """
    label_to_color = dict(zip(labels, colors))
    legend_handles = [Patch(color=label_to_color[label], label=label) for label in labels]
    return label_to_color, legend_handles


def plot_embedding(X, y, title, filename, custom_colors=None):
    """
    Generates and saves a 2D scatter plot of an embedding with a custom legend.
    """
    X, y = np.asarray(X), np.asarray(y)
    plt.figure()

    if custom_colors:
        unique_labels = np.unique(y)
        for label in unique_labels:
            indices = np.where(y == label)
            plt.scatter(X[indices, 0], X[indices, 1],
                        color=custom_colors.get(label),
                        label=label,
                        s=18)

        # Add legend outside the plot
        #plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    else:
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap='jet', s=18)

    plt.xlabel(f'{title} Dimension 1', fontsize=8)
    plt.ylabel(f'{title} Dimension 2', fontsize=8)
    plt.title(filename, fontsize=10)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(os.path.join('./Figures_CSVs', f'{filename}.png'), dpi=300)
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
    #sc.pp.scale(adata, max_value=10)
    
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
    sc.tl.tsne(adata, use_rep='X_pca')
    
    # --- Plotting and Evaluation ---
    true_labels = adata.obs['celltype'].astype('category').cat.codes.to_numpy()
    # --- 4. CLUSTERING AND VISUALIZATION ---
    logging.info("Clustering and visualizing on GenoMap result...")
    sc.pp.neighbors(adata, use_rep='X_pca', n_neighbors=15)
    sc.tl.leiden(adata, resolution=0.5)
    cluster_labels = adata.obs['leiden'].astype('category').cat.codes.to_numpy()


    # --- PLOTTING AND EVALUATION ---
    # Get the unique labels from your data
    unique_labels = sorted(list(np.unique(adata.obs['celltype'])))
    num_labels = len(unique_labels)
    # Dynamically generate colors from a matplotlib colormap
    colors = cm.get_cmap('tab20', num_labels)
    colors_list = [colors(i) for i in range(num_labels)]
    # Create the custom color map dictionary
    label_to_color_map, _ = create_custom_colormap(unique_labels, colors_list)

    plot_filename_prefix = "BBKNN_Algorithm_Scanorama_Dataset"
    
    # Use the new plotting function with the custom colormap
    plot_embedding(adata.obsm['X_umap'], adata.obs['celltype'].astype(str), "UMAP", f"UMAP_Plot_{plot_filename_prefix}", custom_colors=label_to_color_map)
    plot_embedding(adata.obsm['X_tsne'], adata.obs['celltype'].astype(str), "t-SNE", f"TSNE_Plot_{plot_filename_prefix}", custom_colors=label_to_color_map)


    # plot_filename_prefix = "BBKNN_Algorithm_Scanorama_Dataset"
    # plot_embedding(adata.obsm['X_umap'], true_labels, "UMAP", f"UMAP_Plot_{plot_filename_prefix}")
    # plot_embedding(adata.obsm['X_tsne'], true_labels, "t-SNE", f"t-SNE_Plot_{plot_filename_prefix}")

    # Clustering evaluation
    ari = adjusted_rand_score(true_labels, cluster_labels)
    rand = rand_score(true_labels, cluster_labels)
    silhouette = silhouette_score(adata.obsm['X_pca'], cluster_labels)
    logging.info(f"ARI: {ari:.3f}, Rand Index: {rand:.3f}, Silhouette Score: {silhouette:.3f}")
    
    # Plot metrics bar chart and save it to CSV
    plot_metrics_bar(ari, rand, silhouette, f"{plot_filename_prefix}")
    save_metrics_csv(ari, rand, silhouette, f"CSV_{plot_filename_prefix}")

if __name__ == "__main__":
    main()