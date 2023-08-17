"""
This file contains spatial visualization fuctions.
"""


# Import packages
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Polygon, FancyArrowPatch
from matplotlib.collections import PatchCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.spatial import ConvexHull
from skimage.measure import regionprops


# ==== Basics ====
# Get Convexhull for each cell
def get_qhulls(labels):
    """Generate convex hulls from input label image

    Parameters
    ----------
    labels
        input label/segmentation image

    Returns
    -------
    hulls 
        convex hull object for each segmentation region
    coords 
        pixel coordinates of each segmentation region
    centroids 
        segmentation region centroids 
    """
        
    hulls = []
    coords = []
    centroids = []
    print('Geting ConvexHull...')
    for i, region in enumerate(tqdm(regionprops(labels))):
        try:
          hulls.append(ConvexHull(region.coords))
        except:
          print(f"Cannot generate convexhull for region {region.label}")
          hulls.append([])
        coords.append(region.coords)
        centroids.append(region.centroid)
    num_cells = len(hulls)
    print(f"Used {num_cells} / {i + 1}")

    return hulls, coords, centroids


# Convert convexhull to polygon
def hull_to_polygon(hull, 
                    hatch=None):
    """Convert convex hulls to polygons for visualization 

    Parameters
    ----------
    hull
        a convex hull object
    hatch
        hatch shape of the polygon

    Returns
    -------
    poly 
        a polygon object
    """
        
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


# Plot 2D polygon figure indicating cell typing and clustering
def plot_poly_cells_cluster_by_sample(adata, 
                                      sample, 
                                      cmap, 
                                      linewidth=0.1,
                                      show_plaque=None, 
                                      show_tau=None, 
                                      show_tau_cells=None, 
                                      show_gfap=False, 
                                      bg_color='#ededed', 
                                      figscale=10, 
                                      save=False, 
                                      show=True, 
                                      save_as_real_size=False,
                                      output_dir='./figures', 
                                      rescale_colors=False, 
                                      alpha=1, 
                                      vmin=None, 
                                      vmax=None):
    """Plot ploygon spatial map

    Parameters
    ----------
    adata
        Annotated data matrix 
    sample
        labels to separate different groups 
    cmap
        matplotlib color map
    linewidth, optional
        the line width of the polygon, by default 0.1
    show_plaque, optional
        by default None
    show_tau, optional
        by default None
    show_tau_cells, optional
        by default None
    show_gfap, optional
        by default False
    bg_color, optional
        by default '#ededed'
    figscale, optional
        by default 10
    save, optional
        by default False
    show, optional
        y default True
    save_as_real_size, optional
        by default False
    output_dir, optional
        by default './figures'
    rescale_colors, optional
        by default False
    alpha, optional
        by default 1
    vmin, optional
        by default None
    vmax, optional
        by default None
    """
    

    sample_key = f"{sample}_morph"
    labels = adata.uns[sample_key]['label_img']
    hulls = adata.uns[sample_key]['qhulls']
    colors = adata.uns[sample_key]['colors']
    good_cells = adata.uns[sample_key]['good_cells']

    if save_as_real_size:
        plt.figure(figsize=(labels.shape[0]/1000, labels.shape[1]/1000), dpi=100)
    else:
        plt.figure(figsize=(labels.shape[0]/1000 * figscale, labels.shape[1]/1000 * figscale), dpi=100)

    polys = []
    for h in hulls:
        if h == []:
            polys.append([])
        else:
            polys.append(hull_to_polygon(h))

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

    # show background image (labels | DAPI | device)
    if show_plaque:
        plaque = adata.uns[sample_key]['plaque']
        masked_plaque = np.ma.masked_where(plaque == 0, plaque)
        plaque_cmap = sns.color_palette(['#000000', '#ffffff'])
        plaque_cmap = ListedColormap(plaque_cmap)
        plt.imshow(masked_plaque.T, plaque_cmap, interpolation='none', alpha=1, zorder=0)


    labels = (labels > 0).astype(np.int)
    # masked_labels = np.ma.masked_where(labels == 0, labels)
    # labels_cmap = sns.color_palette(['#7a7a7a', '#ffffff'])
    # labels_cmap = ListedColormap(labels_cmap)
    plt.imshow(labels.T, cmap=plt.get_cmap('gray_r'), alpha=0, zorder=1)

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
        tau_cmap = sns.color_palette(['#ff0088', '#000000']) # 00a6b5 ff0088
        tau_cmap = ListedColormap(tau_cmap)
        plt.imshow(masked.T, tau_cmap, interpolation='none', alpha=.8, zorder=4)

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


# Plot 2D polygon figure indicating gene expression
def plot_poly_cells_expr_by_sample(adata, 
                                   sample, 
                                   gene, 
                                   cmap, 
                                   linewidth=0.5, 
                                   use_raw=True,
                                   show_plaque=None, 
                                   show_tau=None, 
                                   show_tau_cells=None,
                                   show_colorbar=False, 
                                   width=2, 
                                   height=9, 
                                   figscale=10,
                                   save=False, 
                                   show=True, 
                                   alpha=1, 
                                   vmin=None, 
                                   vmax=None):
    """_summary_

    Parameters
    ----------
    adata
        Annotated data matrix
    sample
        labels to separate different groups
    gene
        Gene to plot
    cmap
        matplotlib color map
    linewidth, optional
        by default 0.5
    use_raw, optional
        by default True
    show_plaque, optional
        by default None
    show_tau, optional
        by default None
    show_tau_cells, optional
        by default None
    show_colorbar, optional
        by default False
    width, optional
        by default 2
    height, optional
        by default 9
    figscale, optional
        by default 10
    save, optional
        by default False
    show, optional
        by default True
    alpha, optional
        by default 1
    vmin, optional
        by default None
    vmax, optional
        by default None
    """
    
    sample_key = f"{sample}_morph"
    labels = adata.uns[sample_key]['label_img']
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

    if good_cells is not None:
        polys = [p for i, p in enumerate(polys) if i in good_cells]

    p = PatchCollection(polys, alpha=alpha, cmap=cmap, edgecolor='k', linewidth=linewidth, zorder=2)
    p.set_array(expr)
    p.set_clim(vmin=vmin, vmax=vmax)


    labels = (labels > 0).astype(np.int)
    bg_cmap = sns.color_palette(['#383838'])
    # bg_cmap = sns.color_palette(['#383838', '#000000'])
    # bg_cmap = sns.color_palette(['#000000', '#383838'])
    bg_cmap = ListedColormap(bg_cmap)
    masked_labels = np.ma.masked_where(labels == 0, labels)
    plt.imshow(masked_labels.T, cmap=bg_cmap, alpha=.9, zorder=1)

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


# Plot 2D polygon figure indicating other numric metadata field 
def plot_poly_cells_meta_by_sample(adata, 
                                   sample, 
                                   field, 
                                   cmap, 
                                   linewidth=0.5, 
                                   use_raw=True,
                                   show_plaque=None, 
                                   show_tau=None, 
                                   show_tau_cells=None,
                                   show_colorbar=False, 
                                   figscale=10, 
                                   save_as_real_size=False,
                                   output_dir='./figures', 
                                   save=False, 
                                   show=True, 
                                   alpha=1, 
                                   vmin=None, 
                                   vmax=None):
    """_summary_

    Parameters
    ----------
    adata
        Annotated data matrix 
    sample
        labels to separate different groups
    field
        Metadata field to plot 
    cmap
        matplotlib color map
    linewidth, optional
        by default 0.5
    use_raw, optional
        by default True
    show_plaque, optional
        by default None
    show_tau, optional
        by default None
    show_tau_cells, optional
        by default None
    show_colorbar, optional
        by default False
    figscale, optional
        by default 10
    save_as_real_size, optional
        by default False
    output_dir, optional
        by default './figures'
    save, optional
        by default False
    show, optional
        by default True
    alpha, optional
        by default 1
    vmin, optional
        by default None
    vmax, optional
        by default None
    """


    # Get sample information
    sample_key = f"{sample}_morph"
    labels = adata.uns[sample_key]['label_img']
    hulls = adata.uns[sample_key]['qhulls']
    good_cells = adata.uns[sample_key]['good_cells']

    current_index = adata.obs['sample'] == sample

    if use_raw:
        total_expr = adata.obs[field].values
        expr = adata.obs.loc[current_index, field].values
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
        # print(labels.shape[1]/1000, labels.shape[0]/1000)
        plt.figure(figsize=(labels.shape[0]/1000, labels.shape[1]/1000), dpi=100)
    else:
        # plt.figure(figsize=(figscale*width/float(height), figscale))
        plt.figure(figsize=(labels.shape[0]/1000 * figscale, labels.shape[1]/1000 * figscale), dpi=100)


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

    labels = (labels > 0).astype(np.int)
    # bg_cmap = sns.color_palette(['#383838'])
    # bg_cmap = sns.color_palette(['#383838', '#000000'])
    # bg_cmap = sns.color_palette(['#000000', '#383838'])
    # bg_cmap = ListedColormap(bg_cmap)
    # masked_labels = np.ma.masked_where(labels == 0, labels)
    # plt.imshow(masked_labels.T, cmap=bg_cmap, alpha=.9, zorder=1)
    plt.imshow(labels.T, cmap=plt.get_cmap('gray_r'), alpha=.1, zorder=1)
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
    labels = adata.uns[sample_key]['label_img']
    hulls = adata.uns[sample_key]['qhulls']
    colors = adata.uns[sample_key]['colors']
    good_cells = adata.uns[sample_key]['good_cells']

    if save_as_real_size:
        # print(labels.shape[1]/1000, labels.shape[0]/1000)
        plt.figure(figsize=(labels.shape[0]/1000, labels.shape[1]/1000), dpi=100)
    else:
        # plt.figure(figsize=(figscale*width/float(height), figscale))
        plt.figure(figsize=(labels.shape[0]/1000 * figscale, labels.shape[1]/1000 * figscale), dpi=100)

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

    # show background image (labels | DAPI | device)
    if show_plaque:
        plaque = adata.uns[sample_key]['plaque']
        # plt.imshow(plaque.T, cmap=plt.cm.get_cmap('binary'), zorder=0)

        masked_plaque = np.ma.masked_where(plaque == 0, plaque)

        plaque_cmap = sns.color_palette(['#000000', '#ffffff'])
        plaque_cmap = ListedColormap(plaque_cmap)
        plt.imshow(masked_plaque.T, plaque_cmap, interpolation='none', alpha=1, zorder=0)


    labels = (labels > 0).astype(np.int)
    plt.imshow(labels.T, cmap=plt.get_cmap('gray_r'), alpha=0, zorder=1)

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


# ==== Others ====
# Plot tau positive cells 
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


