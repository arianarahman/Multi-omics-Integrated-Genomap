import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Patch
import scanpy as sc
import scanorama
import anndata as ad
from sklearn.metrics import adjusted_rand_score, silhouette_score, rand_score
from sklearn.preprocessing import StandardScaler
import genomap.genoDR as gp

# --- Configuration ---
DATA_FILE = './Dataset/pancreas.h5ad'
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
    try:
        adata = sc.read(DATA_FILE)
    except FileNotFoundError:
        logging.error(f"Data file not found at {DATA_FILE}")
        return None

    # Filter out cells with no gene counts to prevent NaN errors
    sc.pp.filter_cells(adata, min_genes=1)
    
    # Basic preprocessing before splitting for Scanorama
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
    # Explicitly handle any remaining NaNs
    if np.isnan(adata.X).any():
        adata.X[np.isnan(adata.X)] = 0

    return adata

# --- Main Analysis Function ---
def main():
    """Main function to run the GenoMap analysis workflow with Scanorama."""
    sc.settings.verbosity = 2
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    logging.info("Loading and preparing data...")
    adata_raw = load_and_prepare_data()
    if adata_raw is None:
        return

    # 1. Prepare data and run Scanorama for batch correction
    logging.info("Splitting data by batch for Scanorama...")
    adatas_list = [adata_raw[adata_raw.obs[BATCH_KEY] == batch].copy() for batch in adata_raw.obs[BATCH_KEY].unique()]
    
    logging.info("Running Scanorama for batch correction...")
        # Note: hvg is set inside correct_scanpy to align genes across batches
    adatas_corrected = scanorama.correct_scanpy(adatas_list, return_dimred=True, hvg=2000)
    
    # Concatenate the corrected data back into a single object
    adata = ad.concat(adatas_corrected, join='outer', label="batch_id")


    # # 2. Apply GenoMap on the corrected data
    # X = StandardScaler().fit_transform(adata.obsm['X_scanorama'])

    # # choose a grid that matches input dimensionality
    # D = X.shape[1]              
    # side = int(np.sqrt(D))
    # if side * side != D:
    #     # zero-pad to the next square so we can use a square grid
    #     newD = side*side if side*side >= D else (side+1)*(side+1)
    #     pad = np.zeros((X.shape[0], newD - D))
    #     X = np.hstack([X, pad])
    #     side = int(np.sqrt(X.shape[1]))

    # adata.obsm['X_genomap'] = gp.genoDR(
    #     X,
    #     n_dim=32,                        # give GenoMap more capacity
    #     n_clusters=adata.obs['celltype'].nunique(),
    #     colNum=side,
    #     rowNum=side
    # )


    #2. Apply GenoMap on the corrected data
    n_clusters = adata.obs[CELLTYPE_KEY].nunique()
    logging.info("Applying GenoMap on Scanorama-corrected data...")
    adata.obsm['X_genomap'] = gp.genoDR(
        adata.obsm['X_scanorama'], 
        n_dim=32, 
        n_clusters=n_clusters, 
        colNum=33, 
        rowNum=33
    )

    # 3. Clustering and Visualization
    logging.info("Building neighborhood graph, running Leiden, and generating embeddings...")
    sc.pp.neighbors(adata, use_rep='X_genomap', n_neighbors=15)
    sc.tl.leiden(adata, resolution=0.5)

    sc.tl.umap(adata)
    sc.tl.tsne(adata, use_rep='X_genomap')

    # Get the unique labels from your data
    unique_labels = sorted(list(np.unique(adata.obs[CELLTYPE_KEY])))
    num_labels = len(unique_labels)
    # Dynamically generate colors from a matplotlib colormap
    colors = cm.get_cmap('tab20', num_labels)
    colors_list = [colors(i) for i in range(num_labels)]
    # Create the custom color map dictionary
    label_to_color_map, _ = create_custom_colormap(unique_labels, colors_list)

    plot_filename_prefix = "Genomap_Algorithm_BBKNN_Dataset"
    
    # Use the new plotting function with the custom colormap
    plot_embedding(adata.obsm['X_umap'], adata.obs[CELLTYPE_KEY].astype(str), "UMAP", f"UMAP_Plot_{plot_filename_prefix}", custom_colors=label_to_color_map)
    plot_embedding(adata.obsm['X_tsne'], adata.obs[CELLTYPE_KEY].astype(str), "t-SNE", f"TSNE_Plot_{plot_filename_prefix}", custom_colors=label_to_color_map)


    # true_labels_cat = adata.obs[CELLTYPE_KEY].astype('category').cat.codes
    # #true_labels = adata.obs[CELLTYPE_KEY].astype('category').cat.codes

    # plot_filename_prefix = "Genomap_Algorithm_BBKNN_Dataset"
    # plot_embedding(adata.obsm['X_umap'], true_labels_cat, "UMAP", f"UMAP_Plot_{plot_filename_prefix}")
    # plot_embedding(adata.obsm['X_tsne'], true_labels_cat, "t-SNE", f"TSNE_Plot_{plot_filename_prefix}")

    # 4. Evaluation
    logging.info("Calculating evaluation metrics...")
    true_labels = adata.obs[CELLTYPE_KEY].values
    predicted_labels = adata.obs['leiden'].values

    ari = adjusted_rand_score(true_labels, predicted_labels)
    rand = rand_score(true_labels, predicted_labels)
    silhouette = silhouette_score(adata.obsm['X_genomap'], predicted_labels)
    logging.info(f"ARI: {ari:.3f}, Rand Index: {rand:.3f}, Silhouette Score: {silhouette:.3f}")

    plot_metrics_bar(ari, rand, silhouette, f"{plot_filename_prefix}")
    save_metrics_csv(ari, rand, silhouette, f"CSV_{plot_filename_prefix}")
    
    logging.info("GenoMap analysis with BBKNN Dataset complete.")

if __name__ == "__main__":
    main()