import anndata as ad
import matplotlib.pyplot as plt

# 1. Load the dataset
# It might take a few seconds as it loads ~97,000 cells into memory
adata = ad.read_h5ad("src/data/4i/8h.h5ad")

# 2. Print the overall summary
print("--- Basic Summary ---")
print(adata)
print(f"Total number of cells: {adata.n_obs}")
print(f"Total number of features: {adata.n_vars}\n")

# 3. Look at the cell metadata (What information do we have per cell?)
print("--- Cell Metadata (First 5 rows) ---")
print(adata.obs.head(), "\n")

# 4. Look at the feature names (What exactly was measured?)
# You should see ~47 features here (proteins and morphology)
print("--- Feature Names ---")
print(adata.var_names.tolist(), "\n")

# 5. Find out what the 'drug' column is named
# Look at the output of step 3. The column indicating the drug is 
# usually named 'condition', 'treatment', or 'drug'. 
# Let's assume it is called 'condition' for the next step.