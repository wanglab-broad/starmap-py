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
    for i, region in enumerate(regionprops(labels)):
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
    for i, region in enumerate(regionprops(labels)):
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
                            show_plaque=None, show_tau=None, show_tau_cells=None, show_gfap=False,
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

    polys = [hull_to_polygon(h) for h in hulls]

    if good_cells is not None:
        others = [p for i, p in enumerate(polys) if i not in good_cells]
        polys = [p for i, p in enumerate(polys) if i in good_cells]

    p = PatchCollection(polys, alpha=alpha, cmap=cmap, edgecolor='k', linewidth=linewidth, zorder=3)

    other_cmap = sns.color_palette(['#b3b3b3']) # d1d1d1 ededed d9d9d9 b3b3b3
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
