"""
This file contains fuctions that interact with AnnData/SaptialData object.
"""


import os
import numpy as np
import pandas as pd
import seaborn as sns
import textwrap as tw
import matplotlib.pyplot as plt


def show_stat(adata):
    no_gene = np.sum(adata.var["total_counts"] == 0)
    no_read = np.sum(adata.obs["total_counts"] == 0)
    batches = adata.obs.batch.unique()

    count_dict = {}
    for i in batches:
        count_dict[i] = sum(adata.obs['batch'] == i)

    message = tw.dedent(f"""\
     Number of cells: {adata.X.shape[0]}
     Cells without gene: {no_gene}
     Cells without reads: {no_read}
     Number of genes: {adata.X.shape[1]}
     Number of batches: {len(batches)}
     Cells in each experiment: {count_dict}
     """)
    print(message)


def show_reads_quantile(adata):
    read_quantile = adata.obs['total_counts'].quantile([0.1, 0.25, 0.5, 0.75, 0.9])
    print(f"Reads per cell quantile:\n{read_quantile}")


# Filter by cell area
def filter_cells_by_area(adata, min_area, max_area, save=False):

    # Plot cell area distribution
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(adata.obs['area'])

    # Filter by cell area
    adata = adata[adata.obs.area > min_area, :]
    adata = adata[adata.obs.area < max_area, :]

    # Plot cell area distribution after filtration
    plt.subplot(1, 2, 2)
    sns.histplot(adata.obs['area'])

    plt.tight_layout()

    if save:
        current_fig_path = "./figures/cell_filter_by_area.pdf"
        plt.savefig(current_fig_path)

    plt.show()
    print(f"Number of cell left: {adata.X.shape[0]}")

    return adata



