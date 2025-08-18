# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "anndata",
#     "pdex",
# ]
#
# [tool.uv.sources]
# pdex = { git = "https://github.com/drbh/pdex.git", branch = "pairwise_fdr" }
# ///
import os
import anndata as ad
from pdex._single_cell import (
    parallel_differential_expression_vec_wrapper as parallel_differential_expression,
)

adata = ad.read_h5ad("../bspc/convert/vcc_data/adata_Training.h5ad")
os.makedirs("de_results", exist_ok=True)

ctrl_mask = adata.obs["target_gene"] == "non-targeting"
control_cells = adata[ctrl_mask]

targets = ["LAD1", "TWF2"]

cells = [control_cells]
for target in targets:
    print(f"Selected target: {target}")
    target_mask = adata.obs["target_gene"] == target
    target_cells = adata[target_mask]
    cells.append(target_cells)


de_adata = ad.concat(cells)


results = parallel_differential_expression(
    de_adata,
    reference="non-targeting",
    groupby_key="target_gene",
)


results["diff"] = results["pairwise_fdr"] - results["fdr"]
results = results.sort_values("diff")

filtered = results[results["pairwise_fdr"] < 0.05]

print(filtered)
# print the number of each target
for target in targets:
    count = filtered[filtered["target"] == target].shape[0]
    print(f"Target: {target}, DE genes: {count}")

print("=" * 30)

filtered2 = results[results["fdr"] < 0.05]
print(filtered2)
for target in targets:
    count = filtered2[filtered2["target"] == target].shape[0]
    print(f"Target: {target}, DE genes: {count}")
