"""
This file contains basic visualization fuctions.
"""


# Import packages
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats.mstats import zscore


# ==== Statistics ====
# Plot per cell stats plot
def plot_stats_per_cell(adata, 
                        color='sample', 
                        figsize=(15, 5), 
                        save=False, 
                        figpath="./figures/cell_stats.pdf"):
    """Plot statistics info for each cell in the dataset, including reads & genes per cell and correlation between these two vectors. 

    Parameters
    ----------
    adata
        Annotated data matrix
    color, optional
        the input field determines the scatter colors in the correlation plot, by default 'sample'
    figsize, optional
        by default (15, 5)        
    save, optional
        by default False
    """
    plt.figure(figsize)

    reads_per_cell = adata.obs['total_counts']
    genes_per_cell = adata.obs['n_genes_by_counts']

    plt.subplot(1, 3, 1)
    sns.histplot(reads_per_cell)
    plt.ylabel('# cells')
    plt.xlabel('# reads')

    plt.subplot(1, 3, 2)
    sns.histplot(genes_per_cell)
    plt.ylabel('# cells')
    plt.xlabel('# genes')

    plt.subplot(1, 3, 3)
    plt.title(
        'R=%f' % np.corrcoef(reads_per_cell.T, genes_per_cell)[0, 1])  # Pearson product-moment correlation coefficients
    sns.scatterplot(data=adata.obs, x='total_counts', y='n_genes_by_counts', hue=color, s=5)
    plt.xlabel("Reads per cell")
    plt.ylabel("Genes per cell")
    plt.legend(loc='lower right')
    plt.tight_layout()

    if save:
        plt.savefig(figpath)
    plt.show()


# Plot heatmap
def plot_heatmap(adata, 
                 genes, 
                 groupby, 
                 cmap=plt.cm.get_cmap('bwr'), 
                 fontsize=16,
                 use_imshow=False, 
                 ax=None, 
                 show_vlines=True, 
                 annotation=None):
    """Plot gene expression heatmap

    Parameters
    ----------
    adata
        Annotated data matrix
    genes
        Genes to plot
    groupby
        Labels to group cells 
    cmap, optional
        matplotlib color map, by default plt.cm.get_cmap('bwr')
    fontsize, optional
        by default 16
    use_imshow, optional
        by default False
    ax, optional
        by default None
    show_vlines, optional
        by default True
    annotation, optional
        by default None
    """

    input_data = adata.X.copy()
    cluster_vector = adata.obs[groupby].values.astype(int)

    if ax is None:
        ax = plt.axes()

    cluster_sizes = [sum(cluster_vector == i) for i in np.unique(cluster_vector)]
    # data = np.vstack([input_data.iloc[cluster_vector == i, :].loc[:, gene_names].values for i in np.unique(cluster_vector)]).T
    gene_index = []
    for v in genes:
        curr_index = np.where(adata.var.index.to_numpy() == v)[0]
        if len(curr_index) != 0:
            gene_index.append(curr_index[0])
        else:
            raise Exception(f"Gene: {v} not included!")

    data = np.vstack(
        [input_data[cluster_vector == i, :][:, gene_index] for i in np.unique(cluster_vector)]).T


    if use_imshow:
        ax.imshow(np.flipud(zscore(data, axis=1)), vmin=-2.5, vmax=2.5, cmap=cmap, interpolation='none', aspect='auto')
    else:
        ax.pcolor(np.flipud(zscore(data, axis=1)), vmin=-2.5, vmax=2.5, cmap=cmap)
    # plt.imshow(np.flipud(zscore(data,axis=1)),vmin=-2.5, vmax=2.5, cmap=cmap, aspect='auto', interpolation='none')
    ax.set_xlim([0, data.shape[1]])
    ax.set_ylim([0, data.shape[0]])
    if show_vlines:
        for i in np.cumsum(cluster_sizes[:-1]):
            ax.axvline(i, color='k', linestyle='-')
    ax.set_yticks(np.arange(data.shape[0])+0.5)

    if not annotation:
        y_labels = genes[::-1]
    else:
        y_labels = []
        for gene in genes[::-1]:
            y_labels.append("%s - %s" % (gene, annotation[gene]))

    ax.set_yticklabels(y_labels, fontsize=fontsize)
    # ax.get_xaxis().set_fontsize(fontsize)
    plt.tick_params(axis='both', which='major', labelsize=fontsize)


# Plot heatmap of gene markers of each cluster
def plot_heatmap_with_labels(adata, 
                             genes, 
                             groupby, 
                             cmap=plt.cm.get_cmap('tab10'), 
                             show_axis=True,
                             show_top_ticks=True, 
                             use_labels=None, 
                             font_size=15, 
                             annotation=None):
    """Plot gene expression heatmap with cluster label

    Parameters
    ----------
    adata
        Annotated data matrix
    genes
        Genes to plot
    groupby
        Labels to group cells 
    cmap, optional
        by default plt.cm.get_cmap('tab10')
    show_axis, optional
        by default True
    show_top_ticks, optional
        by default True
    use_labels, optional
        by default None
    font_size, optional
        by default 15
    annotation, optional
        by default None
    """

    g = plt.GridSpec(2, 1, wspace=0.01, hspace=0.01, height_ratios=[0.5, 10])
    cluster_vector = adata.obs[groupby].values.astype(int)
    cluster_array = np.expand_dims(np.sort(cluster_vector), 1).T
    ax = plt.subplot(g[0])
    ax.imshow(cluster_array, aspect='auto', interpolation='none', cmap=cmap)
    if show_top_ticks:
        locations = []
        for i in np.unique(cluster_vector):
            locs = np.median(np.argwhere(cluster_array == i)[:, 1].flatten())
            locations.append(locs)
        ax.xaxis.tick_top()
        if use_labels is not None:
            plt.xticks(locations, use_labels, rotation=45)
        else:
            plt.xticks(locations, np.unique(cluster_vector))
        ax.get_yaxis().set_visible(False)
    # ax.axis('off')

    ax = plt.subplot(g[1])
    plot_heatmap(adata, list(genes), groupby, fontsize=font_size, use_imshow=False, ax=ax, annotation=annotation)
    if not show_axis:
        plt.axis('off')