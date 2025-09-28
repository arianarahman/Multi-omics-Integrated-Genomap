import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Patch
from sklearn.metrics import adjusted_rand_score, silhouette_score, rand_score
import anndata as ad
import scipy.io as sio
import scanpy as sc
import scanorama
import genomap.genoDR as gp

# --- Configuration ---
DATA_FOLDER = './Dataset'
OUTPUT_FOLDER = './Figures_CSVs'
DATA_FILES = ['dataBaronX.mat', 'dataMuraroX.mat', 'dataScapleX.mat', 'dataWangX.mat', 'dataXinX.mat']
CLASS_LABEL_FILE = 'classLabel.mat'
BATCH_LABEL_FILE = 'batchLabel.mat'

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

def load_data():
    """Loads all .mat files into a list of AnnData objects and loads labels."""
    adatas = []
    for file in DATA_FILES:
        data_dict = sio.loadmat(os.path.join(DATA_FOLDER, file))
        key = next((k for k in data_dict if 'data' in k.lower()), None)
        if key:
            # Perform basic preprocessing on each batch individually
            adata_batch = ad.AnnData(data_dict[key])
            sc.pp.normalize_total(adata_batch, target_sum=1e4)
            sc.pp.log1p(adata_batch)
            adatas.append(adata_batch)
        else:
            logging.warning(f"No valid data key found in {file}")

    class_labels = sio.loadmat(os.path.join(DATA_FOLDER, CLASS_LABEL_FILE))['classLabel'].squeeze()
    #batch_labels = sio.loadmat(os.path.join(DATA_FOLDER, BATCH_LABEL_FILE))['batchLabel'].squeeze()

    return adatas, class_labels

# --- Main Analysis Function ---
def main():
    """Main function to run the GenoMap analysis workflow with the corrected pipeline."""
    sc.settings.verbosity = 2
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # 1. Load Data
    logging.info("Loading and preparing data...")
    adatas_list, class_labels = load_data()

    # Create a mapping from numerical labels to class names
    class_name_map = {
        1: 'MHC class II', 2: 'acinare', 3: 'ductal', 4: 'gamma', 5: 'macrophage', 6: 'alpha', 
        7: 'beta', 8: 'endothelial', 9: 'epsilon', 10: 'mast', 11: 'mesenchymal', 12: 'stellate', 
        13: 'delta', 14: 'schwann'
    }

    # Map the numerical labels to their corresponding names
    mapped_labels = np.array([class_name_map.get(lbl, f'Unknown-{lbl}') for lbl in class_labels])

    # 2. Batch Correction with Scanorama
    # This is now the FIRST major processing step.
    # Scanorama internally handles HVG selection with the `hvg` parameter.
    logging.info("Running Scanorama for batch correction...")
    # The `correct_scanpy` function is convenient and returns a list of corrected AnnData objects
    adatas_corrected = scanorama.correct_scanpy(adatas_list, return_dimred=True, hvg=2000)

    # 3. Concatenate the corrected results into a single object for downstream analysis
    logging.info("Constructing unified AnnData object from corrected data...")
    adata = ad.concat(adatas_corrected, join='outer', label="batch_id")

    # # Ensure labels match the number of cells after integration
    # limit = min(adata.n_obs, len(class_labels))
    # adata = adata[:limit, :].copy()
    # adata.obs['class'] = class_labels[:limit]
    # #adata.obs['batch'] = batch_labels[:limit]

    limit = min(adata.n_obs, len(mapped_labels))
    adata = adata[:limit, :].copy()
    adata.obs['class'] = mapped_labels[:limit]

    n_clusters = len(np.unique(adata.obs['class']))

    # 4. Apply GenoMap on the Batch-Corrected Data
    logging.info("Applying GenoMap on the Scanorama-corrected embedding...")
    # The input `X_scanorama` now correctly corresponds to the data in `adata`.
    resDR = gp.genoDR(adata.obsm['X_scanorama'], n_dim=64, n_clusters=n_clusters, colNum=33, rowNum=33)
    adata.obsm['X_genomap'] = resDR

    # 5. Clustering and Visualization
    logging.info("Building neighborhood graph and running Leiden clustering...")
    sc.pp.neighbors(adata, use_rep='X_genomap', n_neighbors=15)
    sc.tl.leiden(adata, resolution=0.5)

    logging.info("Generating UMAP and t-SNE plots...")
    sc.tl.umap(adata)
    sc.tl.tsne(adata, use_rep='X_genomap')

    # Get the unique labels from the data
    unique_labels = sorted(list(np.unique(adata.obs['class'])))
    num_labels = len(unique_labels)
    # Dynamically generate colors from a matplotlib colormap
    colors = cm.get_cmap('tab20', num_labels)
    colors_list = [colors(i) for i in range(num_labels)]
    # Create the custom color map dictionary
    label_to_color_map, _ = create_custom_colormap(unique_labels, colors_list)

    plot_filename_prefix = "Genomap_Algorithm_Genomap_Dataset"
    
    # Use the new plotting function with the custom colormap
    plot_embedding(adata.obsm['X_umap'], adata.obs['class'].astype(str), "UMAP", f"UMAP_Plot_{plot_filename_prefix}", custom_colors=label_to_color_map)
    plot_embedding(adata.obsm['X_tsne'], adata.obs['class'].astype(str), "t-SNE", f"TSNE_Plot_{plot_filename_prefix}", custom_colors=label_to_color_map)

    # # Plotting based on true class labels for visual inspection
    # plot_filename_prefix = "Genomap_Algorithm_Genomap_Dataset"
    # plot_embedding(adata.obsm['X_umap'], adata.obs['class'].astype('category').cat.codes, "UMAP", f"UMAP_Plot_{plot_filename_prefix}")
    # plot_embedding(adata.obsm['X_tsne'], adata.obs['class'].astype('category').cat.codes, "t-SNE", f"TSNE_Plot_{plot_filename_prefix}")

    # 6. Evaluation
    logging.info("Calculating evaluation metrics...")
    true_labels = adata.obs['class']
    predicted_labels = adata.obs['leiden']

    ari = adjusted_rand_score(true_labels, predicted_labels)
    rand = rand_score(true_labels, predicted_labels)
    silhouette = silhouette_score(adata.obsm['X_genomap'], predicted_labels)
    #silhouette = silhouette_score(adata.obsm['X_umap'], predicted_labels)
    logging.info(f"ARI: {ari:.3f}, Rand Index: {rand:.3f}, Silhouette Score: {silhouette:.3f}")

    plot_metrics_bar(ari, rand, silhouette, plot_filename_prefix)
    save_metrics_csv(ari, rand, silhouette, f"CSV_{plot_filename_prefix}")

    logging.info("GenoMap analysis complete.")

if __name__ == "__main__":
    main()