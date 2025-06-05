import anndata as ad
import pandas as pd
import polars as pl
from adpbulk import ADPBulk
from pydeseq2.dds import DeseqDataSet
from pydeseq2.default_inference import DefaultInference
from pydeseq2.ds import DeseqStats
from tqdm import tqdm

KNOWN_METHODS = ["sum", "mean", "median"]


def pseudobulk_dex(
    adata: ad.AnnData,
    groupby: list[str],
    groups: list[str] | None = None,
    test_col: str = "target_gene",
    reference: str = "non-targeting",
    design: str | None = None,
    num_workers: int = 1,
    aggr_method: str = "sum",
    as_polars: bool = False,
) -> pd.DataFrame | pl.DataFrame:
    """Calculate differential expression between groups of cells after performing a pseudobulk.


    Parameters
    ----------
    adata: ad.AnnData
        Annotated data matrix containing raw gene expression data (count data)
    groupby: list[str]
        Which columns to pseudobulk over - must include `test_col` (perturbation column)
    groups: list[str], optional
        List of targets to compare, defaults to None which compares all groups (all perturbations)
    reference: str, optional
        Reference group to compare against, defaults to "non-targeting"
    test_col: str, optional
        Column in obs which designates the perturbation identity (or what you want to measure differential expression)
    design: str, optional
        Design formula to use in DESeq2 - defaults to "~ test_col + (remaining groupby keys)"
    num_workers: int
        Number of workers to use for parallel processing, defaults to 1
    aggr_method: str
        Aggregation method to use in pseudobulk
    as_polars: bool
        Return the dataframe as a polars dataframe

    Returns
    -------
    pd.DataFrame containing differential expression results for each group and feature
    """

    # Check for known aggregation method
    if aggr_method not in KNOWN_METHODS:
        raise ValueError(
            f"Unknown aggr method: {aggr_method} :: Expecting: {KNOWN_METHODS} "
        )

    # Validate the test column is in groupby
    if test_col not in groupby:
        raise ValueError(
            f"The test column: {test_col} must be in the groupby set: {groupby}"
        )

    if len(groupby) < 2:
        raise ValueError(
            "DESeq2 requires replicates to properly estimate variance - you will need to provide an additional groupby "
            "column alongside the test column to pseudobulk across."
        )

    if design is None:
        design = "~" + "+".join(groupby)

    print(f"DESIGN: {design}")

    if groups is None:
        groups = [x for x in adata.obs[test_col].unique() if x != reference]

    bulk = ADPBulk(adata, groupby=groupby, method=aggr_method)
    matrix = bulk.fit_transform()
    meta = bulk.get_meta()

    inference = DefaultInference(n_cpus=num_workers)
    dds = DeseqDataSet(
        counts=matrix,
        metadata=meta.set_index("SampleName"),
        design=design,
        refit_cooks=True,
        inference=inference,
        quiet=True,
    )
    dds.deseq2()

    all_results = []
    for target in tqdm(groups, desc="Evaluating contrasts..."):
        ds = DeseqStats(
            dds,
            contrast=[test_col, target, reference],
            inference=inference,
            quiet=True,
        )
        ds.summary()
        results = ds.results_df.reset_index(names="gene")
        results["target"] = target
        results["reference"] = reference
        all_results.append(results)

    all_results = pd.concat(all_results)

    if as_polars:
        return pl.DataFrame(all_results)

    return all_results
