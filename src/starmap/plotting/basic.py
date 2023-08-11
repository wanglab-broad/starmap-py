from anndata import AnnData


def basic_plot(adata: AnnData) -> int:
    """Generate a basic plot for an AnnData object.

    Parameters
    ----------
    adata
        The AnnData object to preprocess.

    Returns
    -------
    Some integer value.
    """
    print("Import matplotlib and implement a plotting function here.")
    return 0


class BasicClass:
    """A basic class.

    Parameters
    ----------
    adata
        The AnnData object to preprocess.
    """

    my_attribute: str = "Some attribute."
    my_other_attribute: int = 0

    def __init__(self, adata: AnnData):
        print("Implement a class here.")

    def my_method(self, param: int) -> int:
        """A basic method.

        Parameters
        ----------
        param
            A parameter.

        Returns
        -------
        Some integer value.
        """
        print("Implement a method here.")
        return 0

    def my_other_method(self, param: str) -> str:
        """Another basic method.

        Parameters
        ----------
        param
            A parameter.

        Returns
        -------
        Some integer value.
        """
        print("Implement a method here.")
        return ""


"""
scanpy related utilites.

"""

# Import packages
import os
import numpy as np
import pandas as pd
import textwrap as tw
import tifffile as tf
from anndata import AnnData
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Polygon, FancyArrowPatch
from matplotlib.collections import PatchCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import ttest_ind, norm, ranksums, spearmanr
from scipy.spatial import ConvexHull
from skimage.measure import regionprops
from scipy.stats.mstats import zscore
from tqdm.notebook import tqdm



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


# Plot per cell stats plot
def plot_stats_per_cell(adata, color='sample', save=False):
    plt.figure(figsize=(15, 5))

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
        if type(save) == str:
            plt.savefig(save)
        else:
            # current_fig_path = os.path.join(os.getcwd(), "output/figures/cell_stats.pdf")
            current_fig_path = "./figures/cell_stats.pdf"
            plt.savefig(current_fig_path)
    plt.show()


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





# Plot heatmap of gene markers of each cluster
def plot_heatmap_with_labels(adata, degenes, cluster_key, cmap=plt.cm.get_cmap('tab10'), show_axis=True,
                             show_top_ticks=True, use_labels=None, font_size=15, annotation=None):

    g = plt.GridSpec(2, 1, wspace=0.01, hspace=0.01, height_ratios=[0.5, 10])
    cluster_vector = adata.obs[cluster_key].values.astype(int)
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
    plot_heatmap(adata, list(degenes), cluster_key, fontsize=font_size, use_imshow=False, ax=ax, annotation=annotation)
    if not show_axis:
        plt.axis('off')


# Plot heatmap
def plot_heatmap(adata, gene_names, cluster_key, cmap=plt.cm.get_cmap('bwr'), fontsize=16,
                 use_imshow=False, ax=None, show_vlines=True, annotation=None):

    input_data = adata.X
    cluster_vector = adata.obs[cluster_key].values.astype(int)

    if ax is None:
        ax = plt.axes()

    clust_sizes = [sum(cluster_vector == i) for i in np.unique(cluster_vector)]
    # data = np.vstack([input_data.iloc[cluster_vector == i, :].loc[:, gene_names].values for i in np.unique(cluster_vector)]).T
    gene_index = []
    for v in gene_names:
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
        for i in np.cumsum(clust_sizes[:-1]):
            ax.axvline(i, color='k', linestyle='-')
    ax.set_yticks(np.arange(data.shape[0])+0.5)

    if not annotation:
        y_labels = gene_names[::-1]
    else:
        y_labels = []
        for gene in gene_names[::-1]:
            y_labels.append("%s - %s" % (gene, annotation[gene]))

    ax.set_yticklabels(y_labels, fontsize=fontsize)
    # ax.get_xaxis().set_fontsize(fontsize)
    plt.tick_params(axis='both', which='major', labelsize=fontsize)


def merge_multiple_clusters(adata, clusts, cluster_key='louvain'):
    cluster_vector = adata.obs[cluster_key].values.astype(int)
    for idx, c in enumerate(clusts[1:]):
        cluster_vector[cluster_vector == c] = clusts[0]
    temp = cluster_vector.copy()
    # relabel clusters to be contiguous
    for idx, c in enumerate(np.unique(cluster_vector)):
        temp[cluster_vector == c] = idx
    adata.obs[cluster_key] = temp
    adata.obs[cluster_key] = adata.obs[cluster_key].astype('category')


# Get Convexhull for each cell
def get_qhulls(labels):
    hulls = []
    coords = []
    centroids = []
    print('Geting ConvexHull...')
    for i, region in enumerate(tqdm(regionprops(labels))):
        current_convex = ConvexHull(region.coords)
        hulls.append(ConvexHull(region.coords))
        coords.append(region.coords)
        centroids.append(region.centroid)
    num_cells = len(hulls)
    print(f"Used {num_cells} / {i + 1}")
    return hulls, coords, centroids

# Get Convexhull for each cell
def get_qhulls_test(labels):
    hulls = []
    coords = []
    centroids = []
    print('Geting ConvexHull...')
    for i, region in enumerate(tqdm(regionprops(labels))):
        try:
          current_convex = ConvexHull(region.coords)
          hulls.append(ConvexHull(region.coords))
        except:
          print(f"Cannot generate convexhull for region {region.label}")
          hulls.append([])
        coords.append(region.coords)
        centroids.append(region.centroid)
    num_cells = len(hulls)
    print(f"Used {num_cells} / {i + 1}")
    return hulls, coords, centroids

# Plot 2D polygon figure indicating cell typing and clustering
def plot_poly_cells_cluster_by_sample(adata, sample, cmap, linewidth=0.1,
                            show_plaque=None, show_tau=None, show_tau_cells=None, show_gfap=False, bg_color='#ededed', 
                            width=2, height=9, figscale=10, save=False, show=True, save_as_real_size=False,
                            output_dir='./figures', rescale_colors=False, alpha=1, vmin=None, vmax=None):
    sample_key = f"{sample}_morph"
    nissl = adata.uns[sample_key]['label_img']
    hulls = adata.uns[sample_key]['qhulls']
    colors = adata.uns[sample_key]['colors']
    good_cells = adata.uns[sample_key]['good_cells']

    if save_as_real_size:
        # print(nissl.shape[1]/1000, nissl.shape[0]/1000)
        plt.figure(figsize=(nissl.shape[0]/1000, nissl.shape[1]/1000), dpi=100)
    else:
        # plt.figure(figsize=(figscale*width/float(height), figscale))
        plt.figure(figsize=(nissl.shape[0]/1000 * figscale, nissl.shape[1]/1000 * figscale), dpi=100)

    polys = []
    for h in hulls:
        if h == []:
            polys.append([])
        else:
            polys.append(hull_to_polygon(h))
    # polys = [hull_to_polygon(h) for h in hulls]

    if good_cells is not None:
        others = [p for i, p in enumerate(polys) if i not in good_cells and p != []]
        polys = [p for i, p in enumerate(polys) if i in good_cells]

    p = PatchCollection(polys, alpha=alpha, cmap=cmap, edgecolor='k', linewidth=linewidth, zorder=3)

    other_cmap = sns.color_palette([bg_color]) # d1d1d1 ededed d9d9d9 b3b3b3
    other_cmap = ListedColormap(other_cmap)
    o = PatchCollection(others, alpha=1, cmap=other_cmap, linewidth=0, zorder=1)

    if vmin or vmax is not None:
        p.set_array(colors)
        p.set_clim(vmin=vmin, vmax=vmax)
    else:
        if rescale_colors:
            p.set_array(colors+1)
            p.set_clim(vmin=0, vmax=max(colors+1))
        else:
            p.set_array(colors)
            p.set_clim(vmin=0, vmax=len(cmap.colors))

            o_colors = np.ones(len(others)).astype(int)
            o.set_array(o_colors)
            o.set_clim(vmin=0, vmax=max(o_colors))

    # show background image (nissl | DAPI | device)
    if show_plaque:
        plaque = adata.uns[sample_key]['plaque']
        # plt.imshow(plaque.T, cmap=plt.cm.get_cmap('binary'), zorder=0)

        masked_plaque = np.ma.masked_where(plaque == 0, plaque)

        plaque_cmap = sns.color_palette(['#000000', '#ffffff'])
        plaque_cmap = ListedColormap(plaque_cmap)
        plt.imshow(masked_plaque.T, plaque_cmap, interpolation='none', alpha=1, zorder=0)


    nissl = (nissl > 0).astype(np.int)
    # masked_nissl = np.ma.masked_where(nissl == 0, nissl)
    # nissl_cmap = sns.color_palette(['#7a7a7a', '#ffffff'])
    # nissl_cmap = ListedColormap(nissl_cmap)
    plt.imshow(nissl.T, cmap=plt.get_cmap('gray_r'), alpha=0, zorder=1)

    plt.gca().add_collection(p)
    plt.gca().add_collection(o)

    if show_gfap:
        gfap = adata.uns[sample_key]['Gfap']
        gfap_cmap = sns.color_palette(['#00db00', '#ffffff']) # 00a6b5 00db00
        gfap_cmap = ListedColormap(gfap_cmap)
        masked = np.ma.masked_where(gfap == 0, gfap)
        plt.imshow(masked.T, gfap_cmap, interpolation='none', alpha=.8, zorder=5)


    if show_tau:
        tau = adata.uns[sample_key]['tau']
        masked = np.ma.masked_where(tau == 0, tau)
        # plt.imshow(masked.T, 'gray', interpolation='none', alpha=0.5)
        # plt.imshow(masked.T, 'Set1', interpolation='none', alpha=0.7)
        tau_cmap = sns.color_palette(['#ff0088', '#000000']) # 00a6b5 ff0088
        # tau_cmap = sns.color_palette(['#7a7a7a', '#000000'])
        tau_cmap = ListedColormap(tau_cmap)
        plt.imshow(masked.T, tau_cmap, interpolation='none', alpha=.8, zorder=4)

        ## test
        # tau = adata.uns['AD_mouse9494_morph']['tau']
        # masked = np.ma.masked_where(tau == 0, tau)
        # tau_test = tau > 0
        # tau_test = tau_test.astype(np.uint8) * 255
        # blobs_labels = label(tau_test, background=0)
        #
        # # good_blobs = []
        # # for i,region in enumerate(regionprops(blobs_labels)):
        # #     if region.area > 2:
        # #         good_blobs.append(region.coords)
        # #     #print(region.label, region.area)
        #
        # hulls, _, _ = su.get_qhulls(blobs_labels)
        # polys = [su.hull_to_polygon(h) for h in hulls]
        #
        # p = PatchCollection(polys, alpha=1, cmap='Reds', edgecolor='k', linewidth=0.1, zorder=2)
        #
        # plt.figure(figsize=(10, 10))
        # plt.imshow(tau_test, cmap='binary')
        # plt.gca().add_collection(p)

    if show_tau_cells == 'arrow':
        tau_index = adata.uns[sample_key]['tau_index']
        tau_cells = adata.obs.loc[tau_index, :].index
        print(tau_cells.shape[0])
        x = adata.obs.loc[tau_cells, 'x'].values
        y = adata.obs.loc[tau_cells, 'y'].values
        # x = x * .3
        # y = y * .3
        # plt.plot(y, x, 'r+', markersize=7)
        x_head = x - 2
        y_head = y - 2

        x_tail = x - 100
        y_tail = y - 100

        arrows = []
        for i in range(len(x)):
            curr_arrow = FancyArrowPatch((y_tail[i], x_tail[i]), (y_head[i], x_head[i]),
                                          mutation_scale=100, arrowstyle='->')
            arrows.append(curr_arrow)

        arrow_collection = PatchCollection(arrows, facecolor="red", alpha=1)
        plt.gca().add_collection(arrow_collection)

    elif show_tau_cells == 'hatch':
        # hatch option
        tau_index = adata.uns[sample_key]['tau_index']
        tau_cells = adata.obs.loc[tau_index, 'orig_index'].to_list()
        tau_polys = [hull_to_polygon(h, hatch='//') for i, h in enumerate(hulls) if i in tau_cells]
        print(len(tau_polys))
        mpl.rcParams['hatch.linewidth'] = 0.3
        t = PatchCollection(tau_polys, alpha=1, hatch='/////////////', facecolor="none", edgecolor="k", linewidth=0.0)
        plt.gca().add_collection(t)

    plt.axis('off')
    plt.tight_layout(pad=0)

    if save:
        if isinstance(save, str):
            if save_as_real_size:
                current_fig_path = f"{output_dir}/sct_{sample}_{save}.tif"
                # plt.savefig(current_fig_path, dpi=1000, pil_kwargs={"compression": "tiff_lzw"})
                plt.savefig(current_fig_path, dpi=1000)
            else:
                current_fig_path = f"{output_dir}/sct_{sample}_{save}.pdf"
                plt.savefig(current_fig_path, bbox_inches='tight', pad_inches=0)
        else:
            if save_as_real_size:
                current_fig_path = f"{output_dir}/sct_{sample}.tif"
                # plt.savefig(current_fig_path, dpi=1000, pil_kwargs={"compression": "tiff_lzw"})
                plt.savefig(current_fig_path, dpi=1000)
            else:
                current_fig_path = f"{output_dir}/sct_{sample}.pdf"
                plt.savefig(current_fig_path, bbox_inches='tight', pad_inches=0)

    if show:
        plt.show()
    else:
        plt.clf()
        plt.close()


def plot_poly_cells_expr_by_sample(adata, sample, gene, cmap, linewidth=0.5, use_raw=True,
                            show_plaque=None, show_tau=None, show_tau_cells=None,
                            show_colorbar=False, width=2, height=9, figscale=10,
                            save=False, show=True, alpha=1, vmin=None, vmax=None):
    sample_key = f"{sample}_morph"
    nissl = adata.uns[sample_key]['label_img']
    hulls = adata.uns[sample_key]['qhulls']
    good_cells = adata.uns[sample_key]['good_cells']

    current_index = adata.obs['sample'] == sample

    if use_raw:
        total_expr = adata.raw[:, gene].X.flatten()
        expr = adata.raw[current_index, gene].X.flatten()
    else:
        total_expr = adata[:, gene].layers['scaled'].flatten()
        expr = adata[current_index, gene].layers['scaled'].flatten()

    if vmax is None:
        vmax = total_expr.max()
    else:
        vmax = vmax

    if vmin is None:
        vmin = total_expr.min()
    else:
        vmin = vmin

    plt.figure(figsize=(figscale*width/float(height), figscale))

    polys = []
    for h in hulls:
        if isinstance(h, list):
            polys.append([])
        else:
            polys.append(hull_to_polygon(h))
    # polys = [hull_to_polygon(h) for h in hulls]

    if good_cells is not None:
        polys = [p for i, p in enumerate(polys) if i in good_cells]

    p = PatchCollection(polys, alpha=alpha, cmap=cmap, edgecolor='k', linewidth=linewidth, zorder=2)
    p.set_array(expr)
    p.set_clim(vmin=vmin, vmax=vmax)


    nissl = (nissl > 0).astype(np.int)
    bg_cmap = sns.color_palette(['#383838'])
    # bg_cmap = sns.color_palette(['#383838', '#000000'])
    # bg_cmap = sns.color_palette(['#000000', '#383838'])
    bg_cmap = ListedColormap(bg_cmap)
    masked_nissl = np.ma.masked_where(nissl == 0, nissl)
    plt.imshow(masked_nissl.T, cmap=bg_cmap, alpha=.9, zorder=1)

    plt.gca().add_collection(p)
    # plt.gca().add_collection(o)

    if show_plaque:
        plaque = adata.uns[sample_key]['plaque']
        plaque_cmap = sns.color_palette(['#000000', '#ffffff'])
        plaque_cmap = ListedColormap(plaque_cmap)
        plt.imshow(plaque.T, cmap=plaque_cmap, alpha=1, zorder=0)

    if show_tau:
        tau = adata.uns[sample_key]['tau']
        masked = np.ma.masked_where(tau == 0, tau)
        # plt.imshow(masked.T, 'gray', interpolation='none', alpha=0.5)
        # plt.imshow(masked.T, 'Set1', interpolation='none', alpha=0.7)
        tau_cmap = sns.color_palette(['#ff0000', '#000000'])
        tau_cmap = ListedColormap(tau_cmap)
        plt.imshow(masked.T, tau_cmap, interpolation='none', alpha=.7, zorder=3)
        # plt.imshow(masked.T, tau_cmap)

    if show_tau_cells == 'arrow':
        tau_index = adata.uns[sample_key]['tau_index']
        tau_cells = adata.obs.loc[tau_index, :].index
        print(tau_cells.shape[0])
        x = adata.obs.loc[tau_cells, 'x'].values
        y = adata.obs.loc[tau_cells, 'y'].values
        # x = x * .3
        # y = y * .3
        # plt.plot(y, x, 'r+', markersize=7)
        x_head = x - 2
        y_head = y - 2

        x_tail = x - 100
        y_tail = y - 100

        arrows = []
        for i in range(len(x)):
            curr_arrow = FancyArrowPatch((y_tail[i], x_tail[i]), (y_head[i], x_head[i]),
                                          mutation_scale=100, arrowstyle='->')
            arrows.append(curr_arrow)

        arrow_collection = PatchCollection(arrows, facecolor="red", alpha=1)
        plt.gca().add_collection(arrow_collection)

    elif show_tau_cells == 'hatch':
        # hatch option
        tau_index = adata.uns[sample_key]['tau_index']
        tau_cells = adata.obs.loc[tau_index, 'orig_index'].to_list()
        tau_polys = [hull_to_polygon(h, hatch='//') for i, h in enumerate(hulls) if i in tau_cells]
        print(len(tau_polys))
        mpl.rcParams['hatch.linewidth'] = 0.3
        t = PatchCollection(tau_polys, alpha=1, hatch='/////////////', facecolor="none", edgecolor="k", linewidth=0.0)
        plt.gca().add_collection(t)

    plt.axis('off')
    if show_colorbar:
        ax = plt.gca()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.05)
        plt.colorbar(p, cax=cax)

    print(vmin, vmax)

    plt.tight_layout()

    if save:
        if isinstance(save, str):
            current_fig_path = f"./figures/sct_{sample}_{save}.pdf"
            plt.savefig(current_fig_path)
        else:
            current_fig_path = f"./figures/sct_{sample}.pdf"
            plt.savefig(current_fig_path)

    if show:
        plt.show()
    else:
        plt.clf()
        plt.close()


# Convert convexhull to polygon
def hull_to_polygon(hull, hatch=None):
    cent = np.mean(hull.points, 0)
    pts = []
    for pt in hull.points[hull.simplices]:
        pts.append(pt[0].tolist())
        pts.append(pt[1].tolist())
    pts.sort(key=lambda p: np.arctan2(p[1] - cent[1],
                                      p[0] - cent[0]))
    pts = pts[0::2]  # Deleting duplicates
    pts.insert(len(pts), pts[0])
    k = 1.1

    if hatch is not None:
        poly = Polygon(k * (np.array(pts) - cent) + cent, edgecolor='k', linewidth=1, fill=False, hatch=hatch)
    else:
        poly = Polygon(k * (np.array(pts) - cent) + cent, edgecolor='k', linewidth=1)
    # poly.set_capstyle('round')
    return poly


def plot_tau_dist(adata, sample, label_key, pl_dict, save=False, show=True):
    current_key = f"{sample}_morph"
    tau_index = adata.uns[current_key]['tau_index']
    tau_dist = pd.DataFrame(adata.obs.loc[tau_index, label_key].value_counts())
    tau_dist['counts'] = tau_dist[label_key]
    tau_dist[label_key] = tau_dist.index
    tau_dist[label_key] = tau_dist[label_key].astype(object)
    tau_dist = tau_dist.reset_index(drop=True)
    tau_dist = tau_dist.loc[tau_dist['counts'] != 0, :]

    tau_pl = []
    for i in tau_dist[label_key].unique():
        print(i)
        tau_pl.append(pl_dict[i])

    g = sns.barplot(x=label_key, y='counts', data=tau_dist, palette=tau_pl)
    for index, row in tau_dist.iterrows():
        g.text(index, row.counts, row.counts, color='black', ha="center")
    plt.xticks(size=10, rotation=45)
    plt.title('Cell type count')
    plt.tight_layout()

    if save:
        plt.savefig(f'./figures/{sample}_{label_key}_tau.pdf')

    if show:
        plt.show()
    else:
        plt.clf()
        plt.close()


def plot_poly_cells_cluster(adata, cmap, width=2, height=9, figscale=10, save=False,
                            show=True, save_as_real_size=False,
                            rescale_colors=False, alpha=1, vmin=None, vmax=None):

    nissl = adata.uns['label_img']
    hulls = adata.uns['qhulls']
    colors = adata.uns['colors']
    good_cells = adata.uns['good_cells']

    if save_as_real_size:
        # print(nissl.shape[1]/1000, nissl.shape[0]/1000)
        plt.figure(figsize=(nissl.shape[0]/1000, nissl.shape[1]/1000), dpi=100)
    else:
        plt.figure(figsize=(figscale*width/float(height), figscale))

    # polys = [hull_to_polygon(h) for h in hulls]
    polys = []
    for h in hulls:
        # print(type(h))
        if type(h) != list:
            polys.append(hull_to_polygon(h))
        else:
            polys.append(Polygon(np.zeros([2,2])))

    if good_cells is not None:
        others = [p for i, p in enumerate(polys) if i not in good_cells]
        polys = [p for i, p in enumerate(polys) if i in good_cells]

    p = PatchCollection(polys, alpha=alpha, cmap=cmap, edgecolor='k', linewidth=0.1, zorder=2)

    other_cmap = sns.color_palette(['#ededed'])
    other_cmap = ListedColormap(other_cmap)
    o = PatchCollection(others, alpha=1, cmap=other_cmap, linewidth=0, zorder=0)

    if vmin or vmax is not None:
        p.set_array(colors)
        p.set_clim(vmin=vmin, vmax=vmax)
    else:
        if rescale_colors:
            p.set_array(colors+1)
            p.set_clim(vmin=0, vmax=max(colors+1))
        else:
            p.set_array(colors)
            p.set_clim(vmin=0, vmax=max(colors))

            o_colors = np.ones(len(others)).astype(int)
            o.set_array(o_colors)
            o.set_clim(vmin=0, vmax=max(o_colors))


    nissl = (nissl > 0).astype(np.int)
    plt.imshow(nissl.T, cmap=plt.get_cmap('gray_r'), alpha=0, zorder=1)

    plt.gca().add_collection(p)
    plt.gca().add_collection(o)
    plt.axis('off')
    plt.tight_layout(pad=0)

    if save:
        if isinstance(save, str):
            if save_as_real_size:
                current_fig_path = f"../output/figures/sct_{save}.tif"
                plt.savefig(current_fig_path, dpi=1000, pil_kwargs={"compression": "tiff_lzw"})
            else:
                current_fig_path = f"../output/figures/sct_{save}.pdf"
                plt.savefig(current_fig_path, bbox_inches='tight', pad_inches=0)
        else:
            if save_as_real_size:
                current_fig_path = f"../output/figures/sct.tif"
                plt.savefig(current_fig_path, dpi=1000, pil_kwargs={"compression": "tiff_lzw"})
            else:
                current_fig_path = f"../output/figures/sct.pdf"
                plt.savefig(current_fig_path, bbox_inches='tight', pad_inches=0)

    if show:
        plt.show()
    else:
        plt.clf()
        plt.close()


def plot_poly_cells_meta_by_sample(adata, sample, meta, cmap, linewidth=0.5, use_raw=True,
                            show_plaque=None, show_tau=None, show_tau_cells=None,
                            show_colorbar=False, width=2, height=9, figscale=10, save_as_real_size=False,
                            output_dir='./figures', save=False, show=True, alpha=1, vmin=None, vmax=None):

    # Get sample information
    sample_key = f"{sample}_morph"
    nissl = adata.uns[sample_key]['label_img']
    hulls = adata.uns[sample_key]['qhulls']
    good_cells = adata.uns[sample_key]['good_cells']

    current_index = adata.obs['sample'] == sample

    if use_raw:
        total_expr = adata.obs[meta].values
        expr = adata.obs.loc[current_index, meta].values
    else:
        print('TBA')

    # Set vmax/vmin for colormap
    if vmax is None:
        vmax = total_expr.max()
    else:
        vmax = vmax

    if vmin is None:
        vmin = total_expr.min()
    else:
        vmin = vmin

    # Set figure size
    if save_as_real_size:
        # print(nissl.shape[1]/1000, nissl.shape[0]/1000)
        plt.figure(figsize=(nissl.shape[0]/1000, nissl.shape[1]/1000), dpi=100)
    else:
        # plt.figure(figsize=(figscale*width/float(height), figscale))
        plt.figure(figsize=(nissl.shape[0]/1000 * figscale, nissl.shape[1]/1000 * figscale), dpi=100)


    # Construct polygon collection
    polys = []
    for h in hulls:
        if isinstance(h, list):
            polys.append([])
        else:
            polys.append(hull_to_polygon(h))
    # polys = [hull_to_polygon(h) for h in hulls]

    if good_cells is not None:
        others = [p for i, p in enumerate(polys) if i not in good_cells]
        polys = [p for i, p in enumerate(polys) if i in good_cells]

    p = PatchCollection(polys, alpha=alpha, cmap=cmap, edgecolor='k', linewidth=linewidth, zorder=2)
    p.set_array(expr)
    p.set_clim(vmin=vmin, vmax=vmax)

    other_cmap = sns.color_palette(['#b3b3b3']) # d1d1d1 ededed
    other_cmap = ListedColormap(other_cmap)
    o = PatchCollection(others, alpha=1, cmap=other_cmap, linewidth=0, zorder=1)
    o_colors = np.ones(len(others)).astype(int)
    o.set_array(o_colors)
    o.set_clim(vmin=0, vmax=max(o_colors))

    nissl = (nissl > 0).astype(np.int)
    # bg_cmap = sns.color_palette(['#383838'])
    # bg_cmap = sns.color_palette(['#383838', '#000000'])
    # bg_cmap = sns.color_palette(['#000000', '#383838'])
    # bg_cmap = ListedColormap(bg_cmap)
    # masked_nissl = np.ma.masked_where(nissl == 0, nissl)
    # plt.imshow(masked_nissl.T, cmap=bg_cmap, alpha=.9, zorder=1)
    plt.imshow(nissl.T, cmap=plt.get_cmap('gray_r'), alpha=.1, zorder=1)
    plt.gca().add_collection(p)
    plt.gca().add_collection(o)

    if show_plaque:
        plaque = adata.uns[sample_key]['plaque']
        # plt.imshow(plaque.T, cmap=plt.cm.get_cmap('binary'), zorder=0)

        masked_plaque = np.ma.masked_where(plaque == 0, plaque)

        plaque_cmap = sns.color_palette(['#000000', '#ffffff'])
        plaque_cmap = ListedColormap(plaque_cmap)
        plt.imshow(masked_plaque.T, plaque_cmap, interpolation='none', alpha=1, zorder=0)


    if show_tau:
        tau = adata.uns[sample_key]['tau']
        masked = np.ma.masked_where(tau == 0, tau)
        # plt.imshow(masked.T, 'gray', interpolation='none', alpha=0.5)
        # plt.imshow(masked.T, 'Set1', interpolation='none', alpha=0.7)
        tau_cmap = sns.color_palette(['#ff0000', '#000000'])
        tau_cmap = ListedColormap(tau_cmap)
        plt.imshow(masked.T, tau_cmap, interpolation='none', alpha=.7, zorder=3)
        # plt.imshow(masked.T, tau_cmap)

    if show_tau_cells == 'arrow':
        tau_index = adata.uns[sample_key]['tau_index']
        tau_cells = adata.obs.loc[tau_index, :].index
        print(tau_cells.shape[0])
        x = adata.obs.loc[tau_cells, 'x'].values
        y = adata.obs.loc[tau_cells, 'y'].values
        # x = x * .3
        # y = y * .3
        # plt.plot(y, x, 'r+', markersize=7)
        x_head = x - 2
        y_head = y - 2

        x_tail = x - 100
        y_tail = y - 100

        arrows = []
        for i in range(len(x)):
            curr_arrow = FancyArrowPatch((y_tail[i], x_tail[i]), (y_head[i], x_head[i]),
                                          mutation_scale=100, arrowstyle='->')
            arrows.append(curr_arrow)

        arrow_collection = PatchCollection(arrows, facecolor="red", alpha=1)
        plt.gca().add_collection(arrow_collection)

    elif show_tau_cells == 'hatch':
        # hatch option
        tau_index = adata.uns[sample_key]['tau_index']
        tau_cells = adata.obs.loc[tau_index, 'orig_index'].to_list()
        tau_polys = [hull_to_polygon(h, hatch='//') for i, h in enumerate(hulls) if i in tau_cells]
        print(len(tau_polys))
        mpl.rcParams['hatch.linewidth'] = 0.3
        t = PatchCollection(tau_polys, alpha=1, hatch='/////////////', facecolor="none", edgecolor="k", linewidth=0.0)
        plt.gca().add_collection(t)

    plt.axis('off')
    if show_colorbar:
        ax = plt.gca()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.05)
        plt.colorbar(p, cax=cax)

    print(vmin, vmax)

    plt.tight_layout(pad=0)

    if save:
        if isinstance(save, str):
            if save_as_real_size:
                current_fig_path = f"{output_dir}/sct_{sample}_{save}.tif"
                plt.savefig(current_fig_path, dpi=1000, pil_kwargs={"compression": "tiff_lzw"})
            else:
                current_fig_path = f"{output_dir}/sct_{sample}_{save}.pdf"
                plt.savefig(current_fig_path, bbox_inches='tight', pad_inches=0)
        else:
            if save_as_real_size:
                current_fig_path = f"{output_dir}/sct_{sample}.tif"
                plt.savefig(current_fig_path, dpi=1000, pil_kwargs={"compression": "tiff_lzw"})
            else:
                current_fig_path = f"{output_dir}/sct_{sample}.pdf"
                plt.savefig(current_fig_path, bbox_inches='tight', pad_inches=0)

    if show:
        plt.show()
    else:
        plt.clf()
        plt.close()


####### Test

def plot_poly_cells_cluster_by_sample_test(adata, sample, cmap, linewidth=0.1,
                            show_plaque=None, show_tau=None, show_tau_cells=None, show_gfap=False, bg_color='#ededed', 
                            width=2, height=9, figscale=10, save=False, show=True, save_as_real_size=False,
                            output_dir='./figures', rescale_colors=False, alpha=1, vmin=None, vmax=None):
    sample_key = f"{sample}_morph"
    nissl = adata.uns[sample_key]['label_img']
    hulls = adata.uns[sample_key]['qhulls']
    colors = adata.uns[sample_key]['colors']
    good_cells = adata.uns[sample_key]['good_cells']

    if save_as_real_size:
        # print(nissl.shape[1]/1000, nissl.shape[0]/1000)
        plt.figure(figsize=(nissl.shape[0]/1000, nissl.shape[1]/1000), dpi=100)
    else:
        # plt.figure(figsize=(figscale*width/float(height), figscale))
        plt.figure(figsize=(nissl.shape[0]/1000 * figscale, nissl.shape[1]/1000 * figscale), dpi=100)

    polys = []
    for h in hulls:
        if h == []:
            polys.append([])
        else:
            polys.append(hull_to_polygon(h))
    # polys = [hull_to_polygon(h) for h in hulls]

    if good_cells is not None:
        others = [p for i, p in enumerate(polys) if i not in good_cells and p != []]
        polys = [p for i, p in enumerate(polys) if i in good_cells]

    p = PatchCollection(polys, alpha=alpha, cmap=cmap, edgecolor='k', linewidth=linewidth, zorder=3)

    other_cmap = sns.color_palette([bg_color]) # d1d1d1 ededed d9d9d9 b3b3b3
    other_cmap = ListedColormap(other_cmap)
    o = PatchCollection(others, alpha=1, cmap=other_cmap, linewidth=0, zorder=1)

    if vmin or vmax is not None:
        p.set_array(colors)
        p.set_clim(vmin=vmin, vmax=vmax)
    else:
        if rescale_colors:
            p.set_array(colors+1)
            p.set_clim(vmin=0, vmax=max(colors+1))
        else:
            p.set_array(colors)
            p.set_clim(vmin=0, vmax=len(cmap.colors))

            o_colors = np.ones(len(others)).astype(int)
            o.set_array(o_colors)
            o.set_clim(vmin=0, vmax=max(o_colors))

    # show background image (nissl | DAPI | device)
    if show_plaque:
        plaque = adata.uns[sample_key]['plaque']
        # plt.imshow(plaque.T, cmap=plt.cm.get_cmap('binary'), zorder=0)

        masked_plaque = np.ma.masked_where(plaque == 0, plaque)

        plaque_cmap = sns.color_palette(['#000000', '#ffffff'])
        plaque_cmap = ListedColormap(plaque_cmap)
        plt.imshow(masked_plaque.T, plaque_cmap, interpolation='none', alpha=1, zorder=0)


    nissl = (nissl > 0).astype(np.int)
    plt.imshow(nissl.T, cmap=plt.get_cmap('gray_r'), alpha=0, zorder=1)

    plt.gca().add_collection(p)
    plt.gca().add_collection(o)

    if show_gfap:
        gfap = adata.uns[sample_key]['Gfap']
        gfap_cmap = sns.color_palette(['#00a31b', '#ffffff']) # 00a6b5 00db00
        gfap_cmap = ListedColormap(gfap_cmap)
        masked = np.ma.masked_where(gfap == 0, gfap)
        plt.imshow(masked.T, gfap_cmap, interpolation='none', alpha=1, zorder=5)

    if show_tau:
        tau = adata.uns[sample_key]['tau']
        tau_cmap = sns.color_palette(['#ff0088', '#ffffff']) # 00a6b5 ff0088
        tau_cmap = ListedColormap(tau_cmap)
        masked = np.ma.masked_where(tau == 0, tau)
        plt.imshow(masked.T, tau_cmap, interpolation='none', alpha=1, zorder=4)

    plt.axis('off')
    plt.tight_layout(pad=0)

    if save:
        if isinstance(save, str):
            if save_as_real_size:
                current_fig_path = f"{output_dir}/sct_{sample}_{save}.tif"
                # plt.savefig(current_fig_path, dpi=1000, pil_kwargs={"compression": "tiff_lzw"})
                plt.savefig(current_fig_path, dpi=1000)
            else:
                current_fig_path = f"{output_dir}/sct_{sample}_{save}.pdf"
                plt.savefig(current_fig_path, bbox_inches='tight', pad_inches=0)
        else:
            if save_as_real_size:
                current_fig_path = f"{output_dir}/sct_{sample}.tif"
                # plt.savefig(current_fig_path, dpi=1000, pil_kwargs={"compression": "tiff_lzw"})
                plt.savefig(current_fig_path, dpi=1000)
            else:
                current_fig_path = f"{output_dir}/sct_{sample}.pdf"
                plt.savefig(current_fig_path, bbox_inches='tight', pad_inches=0)

    if show:
        plt.show()
    else:
        plt.clf()
        plt.close()


"""
This file contains fuctions for visualization.
"""

# Import packages
from . import utilities as ut

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from scipy.stats import ttest_ind, norm, ranksums, spearmanr
from scipy.spatial import ConvexHull
from skimage.measure import regionprops
from scipy.stats.mstats import zscore
from anndata import AnnData


# ==== STATS ====
# Plot per cell stats plot
def plot_stats_per_cell(self):
    plt.figure(figsize=(15, 10))

    if isinstance(self, AnnData):
        reads_per_cell = self.obs['total_counts']
        genes_per_cell = self.obs['n_genes_by_counts']
        color = self.obs['orig_ident']
    else:
        reads_per_cell = self._meta["reads_per_cell"]
        genes_per_cell = self._meta["genes_per_cell"]
        color = self._meta["orig_ident"]

    plt.subplot(2, 3, 1)
    plt.hist(reads_per_cell, 10, color='k')
    plt.ylabel('# cells')
    plt.xlabel('# reads')

    plt.subplot(2, 3, 2)
    plt.hist(genes_per_cell, 10, color='k')
    plt.ylabel('# cells')
    plt.xlabel('# genes')

    plt.subplot(2, 3, 3)
    plt.title(
        'R=%f' % np.corrcoef(reads_per_cell.T, genes_per_cell)[0, 1])  # Pearson product-moment correlation coefficients
    plt.scatter(reads_per_cell, genes_per_cell, marker='.', s=30, c=color, cmap=plt.cm.get_cmap('jet'), lw=0)
    plt.xlabel("Reads per cell")
    plt.ylabel("Genes per cell")


# Plot correlation between pair of experiments
def plot_correlation_pair(self, exp_id_0, exp_id_1):
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    rep1 = np.sum(self.get_cells_by_experiment(idx=exp_id_0, use_scaled=False), axis=0) + 1
    rep2 = np.sum(self.get_cells_by_experiment(idx=exp_id_1, use_scaled=False), axis=0) + 1

    plt.title('Spearman R=%.4f' % spearmanr(rep1, rep2).correlation)
    plt.scatter(rep1, rep2, c="black")
    plt.xlabel("log2(rep1 + 1)")
    plt.ylabel("log2(rep2 + 1)")
    plt.xscale('log', basex=2)
    plt.yscale('log', basey=2)
    plt.show()


# ==== DIMENSIONALITY REDUCTION ====
# Plot explained variances of PCA
def plot_explained_variance(self):
    if self._pca is not None:
        plt.plot(self._pca.explained_variance_ratio_, 'ko-')


# Plot PCA
def plot_pca(self, use_cells=None, dim0=0, dim1=1, colorby="orig_ident", s=10, cmap=plt.cm.get_cmap('jet')):
    if self._active_cells is None:
        plt.scatter(self._transformed_pca[:, dim0], self._transformed_pca[:, dim1],
                         c=self.get_metadata(colorby), cmap=cmap, s=s, lw=0)
    else:
        new_color = self.get_metadata(colorby).iloc[self._active_cells]
        plt.scatter(self._transformed_pca[:, dim0], self._transformed_pca[:, dim1],
                    c=new_color, cmap=cmap, s=s, lw=0)
    # if self._clusts is None:
    #     plt.scatter(self._transformed_pca[:, dim0], self._transformed_pca[:, dim1],
    #                 c=self.get_metadata("orig_ident"), cmap=cmap, s=s, lw=0)
    # else:
    #     plt.scatter(self._transformed_pca[:, dim0], self._transformed_pca[:, dim1],
    #                 c=self._clusts, cmap=cmap, s=s, lw=0)


# Plot dimensionality reduction results (tsne/umap)
def plot_dim(self, cmap=None, colorby=None, s=10, renumber_clusts=False):
    """
    :param self:
    :param cmap: color map
    :param colorby
    :param s: dot size
    :param renumber_clusts:
    :return:
    """
    if self._clusts is None:
        plt.plot(self._tsne[:, 0], self._tsne[:, 1], 'o')
    elif colorby is not None:
        c = self._meta[colorby].cat.codes.to_list()
        plt.scatter(self._tsne[:, 0], self._tsne[:, 1], c=c, s=s, cmap=cmap, lw=0)
    else:
        if cmap is None:
            # get color map based on number of clusters
            cmap = plt.cm.get_cmap('jet', len(np.unique(self._clusts)))
        if not renumber_clusts:
            plt.scatter(self._tsne[:, 0], self._tsne[:, 1], c=self._clusts, s=s, cmap=cmap, lw=0)
        else:
            plt.scatter(self._tsne[:, 0], self._tsne[:, 1], c=self._clusts+1,
                        s=s, cmap=cmap, vmin=0, vmax=self._clusts.max() + 1, lw=0)
        plt.title(f"Number of clusters: {len(np.unique(self._clusts))}")


# Plot dimensionality reduction results of a specific sample
def plot_dim_org_id(self):
    plt.scatter(self._tsne[:, 0], self._tsne[:, 1], c=self._meta['orig_ident'], s=10, cmap=plt.cm.get_cmap('gist_rainbow'), lw=0)


# === EXPRESSION ====
# Plot expression between groups for sepecifc genes for all clusters
def plot_expression_between_groups(self, gene_names, test="bimod", plot_type="bar",
                                   figsize=(10,10), vmin=0, vmax=None, use_raw=False):
    # convert gene_names to list
    if not isinstance(gene_names, list):
        gene_names = [gene_names]

    group_vals = np.unique(self._meta["group"].values)
    # cluster_df = []
    # ncells = []

    f, ax = plt.subplots(nrows=len(gene_names), ncols=len(np.unique(self._clusts)), figsize=figsize)
    f.tight_layout(pad=0.5, w_pad=0.5, h_pad=0.5)

    for i, g in enumerate(gene_names):
        for j, c in enumerate(np.unique(self._clusts)):
            cells = self.get_cells_by_cluster(c, use_raw=use_raw)

            # normalize to TPM
            meta = self.get_metadata_by_cluster(c)
            cells0 = cells.iloc[ut.get_subset_index(meta["group"], group_vals[0]), :]
            cells1 = cells.iloc[ut.get_subset_index(meta["group"], group_vals[1]), :]
            n0 = cells0.shape[0]
            n1 = cells1.shape[0]
            expr = np.hstack((cells0[g].values, cells1[g].values))

            ids = np.hstack((np.zeros((n0,)), np.ones((n1,))))
            temp = np.zeros_like(ids)
            d = pd.DataFrame(data=np.vstack((expr, ids)).T, columns=["expr", "group"])

            if len(gene_names) == 1:
                curr_ax = ax[j]
            else:
                curr_ax = ax[i][j]

            if plot_type is "bar":
                sns.barplot(x="group", y="expr", data=d, ax=curr_ax, capsize=.2, errwidth=1)
            elif plot_type is "violin":
                sns.violinplot(x="group", y="expr", data=d, ax=curr_ax, capsize=.2, errwidth=1,
                               palette="Set2_r", inner=None, linewidth=0)
                sns.swarmplot(x="group", y="expr", data=d, ax=curr_ax, size=4, color=".3", linewidth=0)

            if test is "bimod":
                pval = ut.differential_lrt(cells0[g].values, cells1[g].values)
            if test is "wilcox":
                pval = ranksums(cells0[g].values, cells1[g].values)[1]
            if test is "t":
                pval = ttest_ind(cells0[g].values, cells1[g].values)[1]

            if vmax is not None:
                curr_ax.set_ylim([vmin, vmax])

            curr_ax.set_title("Gene %s\nCluster %d\nP=%2.4f" % (g, c, pval))
            curr_ax.get_xaxis().set_visible(False)
            sns.despine(fig=f, ax=curr_ax, bottom=True, left=True)
            # sns.violinplo


# Plot volcano plot
def plot_vocano(self, log_pval_thresh=5, log_fc_thresh=0.5, test="bimod", use_genes=None, use_raw=False):
    comparisons, ncells = self.compare_expression_between_groups(test=test, use_genes=use_genes, use_raw=use_raw)
    comparisons["pval"] = -np.log10(comparisons["pval"])
    ymax = comparisons["pval"].replace([np.inf, -np.inf], np.nan).dropna(how="all").max()
    n_clusts = len(np.unique(self._clusts))

    for i, c in enumerate(np.unique(self._clusts)):
        ax = plt.subplot(1, n_clusts, i+1)
        curr_vals = comparisons[comparisons["cluster"] == c]
        m = 6
        plt.plot(curr_vals["log_fc"], curr_vals["pval"], 'ko', markersize=m, markeredgewidth=0, linewidth=0)
        good_genes = curr_vals.loc[curr_vals["pval"] > log_pval_thresh, :]
        for g in good_genes.index:
            if g == "Egr2":
                print("Clu=%d,Name=%s,logFC=%f,Pval=%f" % (c, str(g), good_genes.loc[g, "log_fc"], good_genes.loc[g, "pval"]))
        for g in good_genes.index:
            x = good_genes.loc[g,"log_fc"]
            y = good_genes.loc[g,"pval"]
            plt.plot(x, y, 'go', markersize=m, markeredgewidth=0, linewidth=0)
        good_genes = good_genes.loc[good_genes["log_fc"].abs() > log_fc_thresh, :]

        for g in good_genes.index:
            x = good_genes.loc[g, "log_fc"]
            y = good_genes.loc[g, "pval"]
            plt.plot(x, y, 'ro', markersize=m, markeredgewidth=0, linewidth=0)
            plt.text(x, y, str(g), fontsize=18)
        plt.xlim([-2, 2])
        plt.ylim([-0.5, 1.2*ymax])
        ax.set_xticks([-2, 0, 2])
        if i > 0:
            ax.get_yaxis().set_visible(False)
        sns.despine()
        plt.tick_params(axis='both', which='major', labelsize=18)


# Plot dotplot for expression across clusters
def dotplot_expression_across_clusters(self, gene_names, scale_max=500, cmap=plt.cm.get_cmap('viridis'), clust_order=False):
    n_genes = len(gene_names)
    n_clusts = len(np.unique(self._clusts))
    uniq_clusts, clust_counts = np.unique(self._clusts, return_counts=True)

    avg = []  # averge expression value of the genes
    num = []  # number of cells epxressed these genes

    for i in range(n_genes):
        expr = self.get_expr_for_gene(gene_names[i], scaled=True).values
        d = pd.DataFrame(np.array([expr, self._clusts]).T, columns=["expr", "cluster"])
        avg_expr = d.groupby("cluster").mean()
        avg_expr /= avg_expr.sum()
        avg_expr = avg_expr.values.flatten()
        num_expr = d.groupby("cluster").apply(lambda x: (x["expr"] > 0).sum()).values.astype(np.float)
        num_expr /= clust_counts
        avg.append(avg_expr)
        num.append(num_expr)
    avg = np.vstack(avg)
    num = np.vstack(num)

    if clust_order:
        pos = []
        for i in range(n_genes):
            idx = np.argsort(-avg[i,:])
            for k in idx:
                if k not in pos:
                    pos.append(k)
                    break
        print("Number of Genes: %d\nNumber of Clusters: %d" % avg.shape)
        print("Indexes of Cluster shown: ", pos)
        num = num[:, pos]
        avg = avg[:, pos]
        pos = range(num.shape[1])
    else:
        pos = range(n_clusts)

    for i in range(n_genes):
        plt.scatter(pos, -i*np.ones_like(pos), s=num[i, :]*scale_max, c=avg[i, :],
                    cmap=cmap, vmin=0, vmax=avg[i, :].max(), lw=0)

    plt.yticks(-np.array(range(n_genes)), gene_names)
    plt.axes().set_xticks(pos)
    plt.axes().set_xticklabels(pos)
    plt.xlabel('Cluster')


# Plot array of genes x clusters
def plot_expression_across_clusters(self, gene_names, plot_type="bar", figsize=None,
                                    clust_order=None, show_frame=True, palette='gray'):
    if clust_order is not None:
        clusts = self._clusts.copy()
        for i,c in enumerate(clust_order):
            clusts[self._clusts == c] = i
    else:
        clusts = self._clusts

    n_genes = len(gene_names)

    if figsize is None:
        f, ax = plt.subplots(n_genes, 1)
    else:
        f, ax = plt.subplots(n_genes, 1, figsize=figsize)
    f.tight_layout()

    for i in range(n_genes):
        expr = self.get_expr_for_gene(gene_names[i], scaled=False).values
        # plt.title(name)
        d = pd.DataFrame(np.array([expr, clusts]).T, columns=["expr", "cluster"])
        if plot_type == "bar":
            sns.barplot(x="cluster", y="expr", data=d, ax=ax[i], capsize=.2, errwidth=1, palette=palette)
        elif plot_type == "violin":
            sns.violinplot(x="cluster", y="expr", data=d, ax=ax[i], capsize=.2, errwidth=1)
        # get rid of the frame
        if not show_frame:
            for spine in ax[i].spines.values():
                spine.set_visible(False)
            ax[i].get_xaxis().set_visible(False)
            ax[i].get_yaxis().set_visible(True)
            ax[i].set_ylabel(gene_names[i])
            if i == n_genes-1:
                ax[i].tick_params(top='off', bottom='on', left='off', right='off', labelleft='off', labelbottom='on')
            else:
                ax[i].tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='on')


# Plot bar chart for gene expression
def plot_bar_gene_expression(self, gene_names, nrow=None, ncol=None, ylim=None, figsize=(5,5), cmap=None):
    def _bar_plot(name, ylim=None, ax=None):
        expr = self.get_expr_for_gene(name, scaled=False).values
        # plt.title(name)
        d = pd.DataFrame(np.array([expr, self._clusts]).T, columns=["expr", "cluster"])
        if cmap is None:
            sns.barplot(x="cluster", y="expr", data=d, ax=ax)
        else:
            sns.barplot(x="cluster", y="expr", data=d, ax=ax,palette=cmap)
        if ax is None:
            ax = plt.axes()
        ax.set_title(name)
        if ylim is not None:
            ax.set_ylim([-1, ylim])
        sns.despine(ax=ax)
        ax.set_xlabel("")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    if isinstance(gene_names, list):
        if nrow is None and ncol is None:
            nrow = np.ceil(np.sqrt(len(gene_names)))
            ncol = nrow
        f,ax = plt.subplots(int(nrow), int(ncol), figsize=figsize)
        f.tight_layout()
        ax = np.array(ax).flatten()
        for i, name in enumerate(gene_names):
            _bar_plot(name, ylim, ax=ax[i])
    else:
        _bar_plot(gene_names, ylim)


# Plot heatmap
def plot_heatmap(self, gene_names, cmap=plt.cm.get_cmap('bwr'), fontsize=16,
                 use_imshow=False, ax=None, show_vlines=True, annotation=None):
    if self._active_cells is not None:
        input_data = self._data.iloc[self._active_cells, :]
    else:
        input_data = self._data

    if ax is None:
        ax = plt.axes()

    clust_sizes = [sum(self._clusts == i) for i in np.unique(self._clusts)]
    data = np.vstack([input_data.iloc[self._clusts == i, :].loc[:, gene_names].values for i in np.unique(self._clusts)]).T

    if use_imshow:
        ax.imshow(np.flipud(zscore(data, axis=1)), vmin=-2.5, vmax=2.5, cmap=cmap, interpolation='none', aspect='auto')
    else:
        im = ax.pcolor(np.flipud(zscore(data, axis=1)), vmin=-2.5, vmax=2.5, cmap=cmap)
    # plt.imshow(np.flipud(zscore(data,axis=1)),vmin=-2.5, vmax=2.5, cmap=cmap, aspect='auto', interpolation='none')
    ax.set_xlim([0, data.shape[1]])
    ax.set_ylim([0, data.shape[0]])
    if show_vlines:
        for i in np.cumsum(clust_sizes[:-1]):
            ax.axvline(i, color='k', linestyle='-')
    ax.set_yticks(np.arange(data.shape[0])+0.5)

    if not annotation:
        y_labels = gene_names[::-1]
    else:
        y_labels = []
        for gene in gene_names[::-1]:
            y_labels.append("%s - %s" % (gene, annotation[gene]))

    ax.set_yticklabels(y_labels, fontsize=fontsize)
    # ax.figure.colorbar(im, ax=ax)
    # ax.get_xaxis().set_fontsize(fontsize)
    plt.tick_params(axis='both', which='major', labelsize=fontsize)
    return im

# Plot violin plot for gene expression
def plot_violin_gene_expression(self, gene_names, nrow=None, ncol=None, ylim=None):
    def _violin_plot(name, ylim=None):
        expr = self.get_expr_for_gene(name, scaled=False).values
        plt.title(name)
        d = pd.DataFrame(np.array([expr, self._clusts]).T, columns=["expr", "cluster"])
        sns.violinplot(x="cluster", y="expr", data=d)
        if ylim is not None:
            plt.ylim([-1, ylim])

    if isinstance(gene_names, list):
        if nrow is None and ncol is None:
            nrow = np.ceil(np.sqrt(len(gene_names)))
            ncol = nrow
        for i, name in enumerate(gene_names):
            plt.subplot(nrow, ncol, i+1)
            _violin_plot(name, ylim)
    else:
        _violin_plot(gene_names, ylim)


# Plot the expression of a single gene in tSNE space
def plot_tsne_gene_expression(self, gene_names, scaled=True, nrow=None, ncol=None, s=10):
    if isinstance(gene_names, list):
        if nrow is None and ncol is None:
            nrow = np.ceil(np.sqrt(len(gene_names)))
            ncol = nrow
        for i,name in enumerate(gene_names):
            plt.subplot(nrow, ncol, i+1)
            expr = self.get_expr_for_gene(name, scaled=scaled)
            plt.title(name, fontsize=16)
            plt.scatter(self._tsne[:, 0], self._tsne[:, 1], c=expr, cmap=plt.cm.get_cmap('jet'), s=s, lw=0)
            plt.axis('off')
    else:
        expr = self.get_expr_for_gene(gene_names, scaled=scaled)
        plt.title(gene_names, fontsize=16)
        plt.scatter(self._tsne[:, 0], self._tsne[:, 1], c=expr, cmap=plt.cm.get_cmap('jet'), s=s, lw=0)
    # plt.axis('off')


# ==== CELL TYPING WITH MORPHOLOGY ====
# Plot heatmap of gene markers of each cluster
def plot_heatmap_with_labels(self, degenes, cmap=plt.cm.get_cmap('jet'), show_axis=True,
                             show_top_ticks=True, use_labels=None, font_size=15, annotation=None):
    g = plt.GridSpec(2, 1, wspace=0.01, hspace=0.01, height_ratios=[0.5, 10])
    cluster_array = np.expand_dims(np.sort(self._clusts), 1).T
    ax1 = plt.subplot(g[0])
    ax1.imshow(cluster_array, aspect='auto', interpolation='none', cmap=cmap)
    if show_top_ticks:
        locations = []
        for i in np.unique(self._clusts):
            locs = np.median(np.argwhere(cluster_array == i)[:, 1].flatten())
            locations.append(locs)
        ax1.xaxis.tick_top()
        if use_labels is not None:
            plt.xticks(locations, use_labels)
        else:
            plt.xticks(locations, np.unique(self._clusts))
        ax1.get_yaxis().set_visible(False)
    # ax.axis('off')

    ax2 = plt.subplot(g[1])
    im = plot_heatmap(self, list(degenes), fontsize=font_size, use_imshow=False, ax=ax2, annotation=annotation)
    plt.colorbar(im, ax=[ax1, ax2])
    if not show_axis:
        plt.axis('off')


# Plot 2D polygon figure indicating gene expression pattern
def plot_poly_cells_expression(nissl, hulls, expr, cmap, good_cells=None, width=2, height=9, figscale=10, alpha=1, vmin=0, vmax=None):
    # define figure dims
    plt.figure(figsize=(figscale*width/float(height), figscale))
    # get polygon from convexhull
    polys = [hull_to_polygon(h) for h in hulls]
    # filter based on cells
    if good_cells is not None:
        polys = [p for i, p in enumerate(polys) if i in good_cells]
    p = PatchCollection(polys, alpha=alpha, cmap=cmap, linewidths=0)
    p.set_array(expr)
    if vmax is None:
        vmax = expr.max()
    else:
        vmax = vmax
    p.set_clim(vmin=vmin, vmax=vmax)
    plt.gca().add_collection(p)
    plt.imshow(nissl.T, cmap=plt.cm.get_cmap('gray_r'), alpha=0.15)
    plt.axis('off')


# Plot 2D polygon figure indicating cell typing and clustering
def plot_poly_cells_cluster(nissl, hulls, colors, cmap, other_cmap='gray',
                            good_cells=None, width=2, height=9, figscale=10,
                            rescale_colors=False, alpha=1, vmin=None, vmax=None):
    plt.figure(figsize=(figscale*width/float(height),figscale))
    polys = [hull_to_polygon(h) for h in hulls]

    if good_cells is not None:
        others = [p for i, p in enumerate(polys) if i not in good_cells]
        polys = [p for i, p in enumerate(polys) if i in good_cells]

    p = PatchCollection(polys, alpha=alpha, cmap=cmap, edgecolor='k', linewidth=0.5)
    # o = PatchCollection(others, alpha=0.1, cmap=other_cmap, edgecolor='k', linewidth=0.5)

    if vmin or vmax is not None:
        p.set_array(colors)
        p.set_clim(vmin=vmin, vmax=vmax)
    else:
        if rescale_colors:
            p.set_array(colors+1)
            p.set_clim(vmin=0, vmax=max(colors+1))
        else:
            p.set_array(colors)
            p.set_clim(vmin=0, vmax=max(colors))

    # show background image (nissl | DAPI | device)
    nissl = (nissl > 0).astype(np.int)
    plt.imshow(nissl.T, cmap=plt.cm.get_cmap('gray_r'), alpha=0.15)
    plt.gca().add_collection(p)
    # plt.gca().add_collection(o)
    plt.axis('off')
    # return polys


# Plot 2D polygon figure indicating cell typing and clustering
def plot_poly_cells_cluster_with_plaque(nissl, plaque, hulls, colors, cmap,
                            good_cells=None, width=2, height=9, figscale=10,
                            rescale_colors=False, alpha=1, vmin=None, vmax=None):
    plt.figure(figsize=(figscale*width/float(height),figscale))
    polys = [hull_to_polygon(h) for h in hulls]

    if good_cells is not None:
        others = [p for i, p in enumerate(polys) if i not in good_cells]
        polys = [p for i, p in enumerate(polys) if i in good_cells]

    p = PatchCollection(polys, alpha=alpha, cmap=cmap, edgecolor='k', linewidth=0.5)
    # o = PatchCollection(others, alpha=0.1, cmap=other_cmap, edgecolor='k', linewidth=0.5)

    if vmin or vmax is not None:
        p.set_array(colors)
        p.set_clim(vmin=vmin, vmax=vmax)
    else:
        if rescale_colors:
            p.set_array(colors+1)
            p.set_clim(vmin=0, vmax=max(colors+1))
        else:
            p.set_array(colors)
            p.set_clim(vmin=0, vmax=max(colors))

    # show background image (nissl | DAPI | device)
    nissl = (nissl > 0).astype(np.int)
    plt.imshow(plaque.T, cmap=plt.cm.get_cmap('binary'))
    plt.imshow(nissl.T, cmap=plt.cm.get_cmap('gray_r'), alpha=0.15)
    plt.gca().add_collection(p)
    # plt.gca().add_collection(o)
    plt.axis('off')
    # return polys


# Convert convexhull to polygon
def hull_to_polygon(hull):
    cent = np.mean(hull.points, 0)
    pts = []
    for pt in hull.points[hull.simplices]:
        pts.append(pt[0].tolist())
        pts.append(pt[1].tolist())
    pts.sort(key=lambda p: np.arctan2(p[1] - cent[1],
                                    p[0] - cent[0]))
    pts = pts[0::2]  # Deleting duplicates
    pts.insert(len(pts), pts[0])
    k = 1.1
    poly = Polygon(k*(np.array(pts) - cent) + cent, edgecolor='k', linewidth=1)
    # poly.set_capstyle('round')
    return poly


# Get Convexhull for each cell
def get_qhulls(labels):
    hulls = []
    coords = []
    centroids = []
    print('Geting ConvexHull...')
    for i, region in enumerate(regionprops(labels)):
        if 100000 > region.area > 1000:
            hulls.append(ConvexHull(region.coords))
            coords.append(region.coords)
            centroids.append(region.centroid)
    num_cells = len(hulls)
    print(f"Used {num_cells} / {i + 1}")
    return hulls, coords, centroids


# Plot cells of each cluster
def plot_cells_cluster(nissl, coords, good_cells, colors, cmap, width=2, height=9, figscale=100, vmin=None, vmax=None):
    plt.figure(figsize=(figscale*width/float(height), figscale))
    img = -1 * np.ones_like(nissl)
    curr_coords = [coords[k] for k in range(len(coords)) if k in good_cells]
    for i, c in enumerate(curr_coords):
        for k in c:
            if k[0] < img.shape[0] and k[1] < img.shape[1]:
                img[k[0], k[1]] = colors[i]
    plt.imshow(img.T, cmap=cmap, vmin=-1, vmax=colors.max())
    plt.axis('off')


# Plot cells of each cluster as dots
def plot_dot_cells_cluster_2d(centroids, good_cells, colors, cmap):
    good_centroids = []
    for i in range(len(centroids)):
        if i in good_cells:
            good_centroids.append(centroids[i])

    transformed_centroids = ut.rc2xy_2d(np.array(good_centroids))

    plt.figure(figsize=(20, 10))
    plt.scatter(transformed_centroids[:, 0], transformed_centroids[:, 1], s=100, c=colors, cmap=cmap)
    plt.axis('off')
    plt.show()


# Get colormap
def get_colormap(colors):
    pl = sns.color_palette(colors)
    cmap = ListedColormap(pl.as_hex())

    sns.palplot(sns.color_palette(pl))
    return cmap

