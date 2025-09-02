import h5py
import pandas as pd
from scipy.sparse import csc_matrix # Import csc_matrix instead of csr_matrix
from anndata import AnnData
import os

# --- File Path Configuration ---
# Ensure this path points to the directory containing your .h5 file.
# Using a relative path like './Dataset' is often more portable.
data_folder = './Dataset' 
input_file = os.path.join(data_folder, "pbmcs_ctrl.h5")
output_file = os.path.join(data_folder, "pbmcs_ctrl_converted.h5ad")

# Create the output directory if it doesn't exist
os.makedirs(data_folder, exist_ok=True)

print(f"Loading data from: {input_file}")

# --- Load Data from the HDF5 File ---
# This block reads the different components of the 10x Genomics HDF5 format.
with h5py.File(input_file, "r") as f:
    # Access the main matrix group
    mat_group = f['matrix']
    
    # Extract sparse matrix components
    data = mat_group['data'][:]
    indices = mat_group['indices'][:]
    indptr = mat_group['indptr'][:]
    
    # Extract cell barcodes and gene names, decoding from bytes to strings
    barcodes = [s.decode('utf-8') for s in mat_group['barcodes'][:]]
    
    feature_group = mat_group['features']
    genes = [s.decode('utf-8') for s in feature_group['name'][:]]

    # Infer shape from the number of genes and barcodes
    shape = (len(genes), len(barcodes))

    # --- THIS IS THE FIX: Use csc_matrix for column-compressed data ---
    # Create the sparse matrix using the correct format
    X = csc_matrix((data, indices, indptr), shape=shape)
    # -----------------------------------------------------------------
    
print("Data loaded successfully.")

# --- Create AnnData Object and Save ---
# AnnData is the standard format for single-cell data in Scanpy.
# We create DataFrames for observations (cells) and variables (genes).
obs = pd.DataFrame(index=barcodes)
var = pd.DataFrame(index=genes)

# Assemble the AnnData object
# Transpose X to get the standard (cells x genes) shape
adata = AnnData(X=X.T, obs=obs, var=var) 

# --- Add Metadata Columns ---
# Add a 'batch' column. Since this is a single control file,
# we'll assign the same batch ID to all cells.
adata.obs['batch'] = 'ctrl'

# Add a placeholder 'celltype' column.
# This makes the file compatible with analysis scripts that expect this column.
adata.obs['celltype'] = 'PBMC_ctrl'

# Convert the new columns to 'category' dtype for compatibility with Scanpy functions
adata.obs['batch'] = adata.obs['batch'].astype('category')
adata.obs['celltype'] = adata.obs['celltype'].astype('category')
# --------------------------------------------

print(f"AnnData object created with shape: {adata.shape}")
print("Added 'batch' and 'celltype' columns to adata.obs")
print(f"Saving to: {output_file}")

# Write the AnnData object to an .h5ad file
adata.write(output_file)

print("Conversion complete.")