"""
This file contains deprecated functions, some of them might not be working any more due to the changes of the overall package architecture. 

"""

# Import related packages 
import os
import numpy as np
import pandas as pd
import textwrap as tw
from anndata import AnnData

# Dimensionality reduction
import umap
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA, FactorAnalysis, NMF

# Clustering
import hdbscan
import igraph as ig
from sklearn.mixture import GaussianMixture
from sklearn.cluster import SpectralClustering, DBSCAN
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.preprocessing import scale, MinMaxScaler
from scipy.stats import ttest_ind, ranksums
from statsmodels.stats.multitest import multipletests

# Visualization 
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from scipy.stats import ttest_ind, norm, ranksums, spearmanr
from scipy.spatial import ConvexHull
from skimage.measure import regionprops
from scipy.stats.mstats import zscore


# The old STARMapDataset object class for downstream analysis used in Wang et al. 2018 Science 
class STARMapDataset(object):
    """This is the fundamental class"""

    def __init__(self):
        self._raw_data = None  # raw data
        self._data = None  # data that has been normalized (log + total count)
        self._scaled = None  # scaled data
        self._pca = None  # pca (object) for cells
        self._transformed_pca = None  # pca (values) for cells
        self._tsne = None  # tsne for cells
        self._clusts = None  # per cell clustering
        self._meta = None  # per cell metadata
        self._meta_out = None  # output metadata
        self._nexpt = 0  # number of experiment

        self._nissl = None  # nissl channel image
        self._hulls = None  # convexhull generated from the label image (nissl or others)

        self._all_ident = None  # original identity of all cells (including those filtered out)

        # binary array of which cells/genes are included and which are filtered out
        self._good_cells = None
        self._good_genes = None
        self._active_cells = None
        self._ncell, self._ngene = 0, 0

    # ==== INPUT FUNCTIONS ====
    # Loading data into the object
    def add_data(self, data, group=None, tpm_norm=False, use_genes=None, cluster_id_path=None):
        """
        Add data to data matrix, keeping track of source.
        Inputs:
            group: numeric id of experiment
            tpm_norm: normalize raw data to TPM (transcripts per million)
            use_genes: only include listed genes
            cluster_id_path: load cluster identities from CSV file (of format cell_num,cluster_id,cluster_name per line)
        """
        reads_per_cell = data.sum(axis=1)  # row sum
        genes_per_cell = (data > 0).sum(axis=1)  # genes with reads per cell (row sum)
        # cells_per_gene = (data > 0).sum(axis=0)  # cells with reads per gene (column sum)

        # transfer to tpm
        if tpm_norm:
            data = data / (data.sum().sum() / 1e6)  # can get how many reads per million

        # get expression profile for subset of genes
        if use_genes:
            data = data[use_genes]

        # get stats
        n_cells, n_genes = data.shape
        # construct metadata table with group field or not
        if group is not None:
            meta = pd.DataFrame(np.vstack((np.ones(n_cells, dtype=np.int) * self._nexpt,
                                           reads_per_cell, genes_per_cell, np.repeat(group, n_cells))).T,
                                columns=["orig_ident", "reads_per_cell", "genes_per_cell", "group"])
        else:
            meta = pd.DataFrame(np.vstack((np.ones(n_cells, dtype=np.int) * self._nexpt,
                                           reads_per_cell, genes_per_cell)).T,
                                columns=["orig_ident", "reads_per_cell", "genes_per_cell"])

        # load cluster information from files
        if cluster_id_path is not None:
            labels = pd.read_csv(cluster_id_path)
            meta['cluster_id'] = labels["Cluster_ID"]
            meta['cluster_name'] = labels['Cluster_Name']

        # assign dataframes to the analysis object
        if self._nexpt == 0:
            self._raw_data = data
            self._meta = meta
        else:  # add data to existing dataframe
            self._raw_data = self._data.append(data)
            self._meta = self._meta.append(meta)

        self._data = self._raw_data
        self._nexpt += 1
        self._ncell, self._ngene = self._data.shape

        self._all_ident = np.array(self._meta['orig_ident'])

        self._good_cells = np.ones((self._ncell,), dtype=np.bool)
        self._good_genes = np.ones((self._ngene,), dtype=np.bool)

        # add filtration field
        # self._meta['Keep'] = False

        # add meta_out
        self._meta_out = self._meta.copy()

    # Load meta data
    def add_meta_data(self, input_meta=None):
        if isinstance(input_meta, str):
            self._meta = pd.read_csv(input_meta)
        elif isinstance(input_meta, pd.DataFrame):
            self._meta = input_meta
        else:
            print("Please provide a valid path of cluster label file or a pandas dataframe!")

    # Add meta data field
    def add_meta_data_field(self, field_name, field_data):
        self._meta[field_name] = field_data
        self._meta.head()

    # Load cluster labels for cells
    def add_cluster_data(self, input_cluster=None):
        if isinstance(input_cluster, str):
            labels = pd.read_csv(input_cluster)
            self._meta['cluster_id'] = labels["Cluster_ID"]
            self._meta['cluster_name'] = labels['Cluster_Name']
            self._clusts = self._meta['cluster_id'].to_numpy(dtype=np.int32)
        elif isinstance(input_cluster, pd.DataFrame):
            self._meta['cluster_id'] = input_cluster["Cluster_ID"]
            self._meta['cluster_name'] = input_cluster['Cluster_Name']
            self._clusts = self._meta['cluster_id'].to_numpy(dtype=np.int32)
        else:
            print("Please provide a valid path of cluster label file or a pandas data frame!")

    # Map numeric cluster labels to user defined labels
    def map_cluster_data(self, input_dict, id_field="cluster_id", label_field="cluster_label"):
        if id_field not in self._meta.columns:
            self._meta[id_field] = self._clusts

        self._meta[label_field] = self._meta[id_field]
        self._meta = self._meta.replace({label_field: input_dict})

    # Add cluster labels to output meta table
    def add_cluster_label(self, input_dict, label_field):
        # obj meta table
        if label_field not in self._meta.columns:
            self._meta[label_field] = "NA"

        if label_field not in self._meta_out.columns:
            self._meta_out[label_field] = "NA"

        if self._active_cells is not None:
            self._meta[label_field].iloc[self._active_cells] = self._clusts
            replace_cells = np.argwhere(self._good_cells == True).flatten()
            self._meta_out[label_field].iloc[replace_cells[self._active_cells]] = self._clusts
        else:
            self._meta[label_field] = self._clusts
            self._meta_out.loc[self._good_cells, label_field] = self._clusts


        self._meta = self._meta.replace({label_field: input_dict})
        self._meta_out = self._meta_out.replace({label_field: input_dict})
        # fill NA
        is_na = self._meta_out[label_field].isnull()
        self._meta_out.loc[is_na, label_field] = 'NA'

    # Add locations
    def add_location(self, input_location):
        dims = ['r', 'c', 'z']

        for c in range(input_location.shape[1]):
            self._meta_out[dims[c]] = input_location[:, c]

    # ==== DATA ACCESS ====
    # Get genes
    def get_gene_names(self):
        return self._raw_data.columns

    # Get features in metadata
    def get_metadata_names(self):
        return self._meta.columns

    # Get specific feature in metadata
    def get_metadata(self, feature):
        return self._meta[feature]

    # Get metadata of specific cluster
    def get_metadata_by_cluster(self, clust_id):
        cells = self._clusts == clust_id
        return self._meta.iloc[cells, :]

    def get_metaout_by_experiment(self, expt_id):
        return self._meta_out.loc[self._meta_out["orig_ident"] == expt_id, :]

    # Get expression profile for specific gene
    def get_expr_for_gene(self, gene_name, scaled=True):
        if scaled:
            return np.log1p(self._raw_data[gene_name])
        else:
            return self._raw_data[gene_name]

    # Get expression profile for specific cell
    def get_expr_for_cell(self, cell_index, scaled=True):
        if scaled:
            return np.log1p(self._raw_data.iloc[cell_index, :])
        else:
            return self._raw_data.iloc[cell_index, :]

    # Get mean expression profile for all clusters
    def get_mean_expr_across_clusters(self, scaled=True):
        expr = [self.get_mean_expr_for_cluster(i, scaled) for i in range(max(self._clusts))]
        return pd.concat(expr, axis=1).transpose()

    # Get mean expression profile for specific cluster
    def get_mean_expr_for_cluster(self, clust_id, scaled=True):
        data = self.get_cells_by_cluster(clust_id, use_raw=True)
        if scaled:
            return np.log1p(data)
        else:
            return data

    # Get cell index for specific experiment
    def get_cells_by_experiment(self, idx, use_genes=None, use_scaled=False):
        condition = self._meta["orig_ident"] == idx
        expt_idx = np.argwhere(condition.to_numpy()).flatten()

        if use_scaled:
            data = self._raw_data
        else:
            data = self._data

        if use_genes:
            return data.iloc[expt_idx, :].loc[:, use_genes]
        else:
            return data.iloc[expt_idx, :]

    # Get expression profile of cells in specific cluster
    def get_cells_by_cluster(self, clust_id, use_raw=False):
        cells = self._clusts == clust_id
        if use_raw:
            return self._raw_data.iloc[cells, :]
        else:
            return self._data.iloc[cells, :]

    # Get cluster information of specific experiment
    def get_cluster_for_experiment(self, expt_id):
        return self._clusts[self._meta["orig_ident"] == expt_id]

    # Get cluster information base on meta
    def get_cluster_for_group(self, meta_value, meta_field="group"):
        return self._clusts[self._meta[meta_field] == meta_value]

    # Get cluster labels
    def get_cluster_labels(self, cell_type_names=None):
        if cell_type_names is None:
            cluster_labels = np.array([cell_type_names[i] for i in self._clusts])
        else:
            cluster_labels = np.array(['NA' for _ in self._clusts])
        return cluster_labels

    # Get cell indexs and cluster labels of specific experiment
    def get_cells_and_clusts_for_experiment(self, expt_id, colorby=None):
        if self._active_cells is not None:
            meta = self._meta.iloc[self._active_cells, :]
        else:
            meta = self._meta

        if colorby is not None:
            good_cells = meta.index[(meta["orig_ident"] == expt_id)].values
            colors = self._meta[colorby].loc[self._meta['orig_ident'] == expt_id].cat.codes
        else:
            good_cells = meta.index[(meta["orig_ident"] == expt_id)].values
            colors = self._clusts[meta["orig_ident"] == expt_id]
        return good_cells, colors

    def show_stat(self):
        no_gene = np.sum(self._meta["genes_per_cell"] == 0)
        no_read = np.sum(self._meta["reads_per_cell"] == 0)
        count_dict = {}
        for i in range(self._nexpt):
            count_dict[i] = sum(self._meta['orig_ident'] == i)

        message = tw.dedent(f"""\
        Number of cells: {self._ncell}
        Cells without gene: {no_gene}
        Cells without reads: {no_read}
        Number of genes: {self._ngene}
        Number of experiments: {self._nexpt}
        Cells in each experiment: {count_dict}
        """)
        print(message)

    def show_reads_quantile(self):
        read_quantile = self._meta['reads_per_cell'].quantile([0.1, 0.25, 0.5, 0.75, 0.9])
        print(f"Reads per cell quantile:\n{read_quantile}")

    # ==== SUBSET FUNCTIONS ====
    # Get subset of data by using cell index
    def subset_by_cell(self, cell_ids):
        subset = STARMapDataset()
        subset._raw_data = self._raw_data.iloc[cell_ids, :]  # raw data
        subset._data = self._data.iloc[cell_ids, :]  # data that has been normalized (log + total count)
        subset._scaled = self._scaled.iloc[cell_ids, :]  # scaled data
        subset._meta = self._meta.iloc[cell_ids]  # per cell metadata
        subset._nexpt = self._nexpt
        subset._ncell, subset._ngene = subset._data.shape
        return subset

    # Get subset of data by using cluster id
    def subset_by_cluster(self, cluster_id):
        cell_ids = np.argwhere(np.array(self._clusts) == cluster_id).flatten()
        subset = self.subset_by_cell(cell_ids)
        return subset

    # ==== MANIPULATE CLUSTER ====
    # Merge each cluster into first in list
    def merge_multiple_clusters(self, clusts):
        for idx, c in enumerate(clusts[1:]):
            self._clusts[self._clusts == c] = clusts[0]
        temp = self._clusts.copy()
        # relabel clusters to be contiguous
        for idx, c in enumerate(np.unique(self._clusts)):
            temp[self._clusts == c] = idx
        self._clusts = temp

    # Merge cluster1 into cluster0
    def merge_clusters(self, clust0, clust1):
        self._clusts[self._clusts == clust1] = clust0
        temp = self._clusts.copy()
        for idx, c in enumerate(np.unique(self._clusts)):
            temp[self._clusts == c] = idx
        self._clusts = temp

    # Sort clusters with input order
    def order_clusters(self, orders):
        temp = self._clusts.copy()
        n_clusters = len(np.unique(self._clusts))
        for idx, c in enumerate(np.unique(self._clusts)):
            temp[self._clusts == c] = idx + n_clusters

        for idx, c in enumerate(np.unique(temp)):
            temp[temp == c] = orders[idx]
            print(f"{idx} --> {orders[idx]}")
        self._clusts = temp

    # ==== Data Structure ====
    # Transfer to AnnData object
    def transfer_to_anndata(self):

        # Get X (expression profile)
        if self._scaled is None:
            X = self._data.values
        else:
            X = self._scaled.values

        # Get obs (metadata of obervations)
        obs = self._meta
        # if self._meta_out is None:
        #     obs = self._meta
        # else:
        #     obs = self._meta_out

        adata = AnnData(X=X, obs=obs)
        if self._transformed_pca is not None:
            adata.obsm['X_pca'] = self._transformed_pca
        if self._tsne is not None:
            adata.obsm['X_umap'] = self._tsne
        if self._clusts is not None:
            adata.obs['clusts'] = self._clusts

        return adata


# ====Additional IO function====
# Load gene expression table (output files from sequencing pipeline)
def load_data(data_dir, prefix="Cell"):
    expr = pd.read_csv(os.path.join(data_dir, "cell_barcode_count.csv"), header=None)
    gene_names = pd.read_csv(os.path.join(data_dir, "cell_barcode_names.csv"), header=None)
    row_names = ["%s_%05d" % (prefix, i) for i in range(expr.shape[0])]
    names = gene_names[2]
    names.name = "Gene"
    return pd.DataFrame(data=expr.values, columns=names, index=row_names)


# Load gene expression table (output file from MATLAB pipeline)
def load_new_data(data_dir):
    expr = pd.read_csv(os.path.join(data_dir, "geneByCell.csv"), header=0, index_col=0)
    return expr.T


# ==== PRE-PROCESSING ====
# Filter the cell and gene by its expression profile
def filter_cells_by_expression(self, min_genes=10, min_cells=10):
    good_cells = (self._raw_data.values > 0).sum(axis=1) > min_genes  # at least min_genes expressed in these cells
    good_genes = (self._raw_data.values > 0).sum(axis=0) > min_cells  # at least min_cells have the reads of the genes

    print("Filter cells by expression:")
    print(f"#Cells left after filtered by genes: {np.sum(good_cells)}")

    # construct logical array and filter the data with it
    self._good_cells = np.logical_and(self._good_cells, good_cells)
    self._good_genes = np.logical_and(self._good_genes, good_genes)
    self._raw_data = self._raw_data.loc[good_cells, :]
    self._raw_data = self._raw_data.loc[:, good_genes]
    self._data = self._data.loc[good_cells, :]
    self._data = self._data.loc[:, good_genes]
    if self._scaled is not None:
        self._scaled = self._scaled.loc[good_cells, :]
    self._meta = self._meta.loc[good_cells]
    self._meta_out['Keep'] = self._good_cells
    self._ncell, self._ngene = self._data.shape

    print(f"#Genes left after filtered by cells: {self._ncell}")

    # return self


# Filter the cells based on features in metadata table
def filter_cells_by_meta_feature(self, feature_name, low_thresh, high_thresh):
    current_field = self._meta[feature_name].values
    to_keep = np.logical_and(current_field > low_thresh, current_field <= high_thresh)

    print(f"Filter cells by {feature_name} with threshold {low_thresh} to {high_thresh}:")
    print(f"#Cells left after filtered by genes: {self._ncell}")

    self._good_cells[self._good_cells] = to_keep
    self._raw_data = self._raw_data.loc[to_keep, :]
    self._data = self._data.loc[to_keep, :]
    if self._scaled is not None:
        self._scaled = self._scaled.loc[to_keep, :]
    self._meta = self._meta.loc[to_keep, :]
    self._meta_out['Keep'] = self._good_cells
    self._ncell, self._ngene = self._data.shape

    print(f"#Genes left after filtered by cells: {self._ncell}")

    # return self


# Apply normalization
def normalize(self, norm_method="none", use_genes=None, scale_factor=10000):
    # only use a subset of genes for normalization
    if use_genes:
        data = self._data.loc[:, use_genes]
    else:
        data = self._data

    median_transcripts = np.median(self._raw_data.sum(axis=1))  # get median number of reads for all cells
    for i in range(self._ncell):
        # normalized reads per cell = natural logrithm (ln) ((1 + raw reads / total reads of the cell) * scaling factor)
        if norm_method == "abs":
            self._data.iloc[i, :] = np.log1p((self._data.iloc[i, :] / data.iloc[i, :].sum()) * scale_factor)
        # normalized reads per cell = natural logrithm (ln) ((1 + raw reads / total reads of the cell) * median number of reads for all cells)
        elif norm_method == "median":
            self._data.iloc[i, :] = np.log1p((self._data.iloc[i, :] / data.iloc[i, :].sum()) * median_transcripts)
        # normalized reads of each cell = natural logarithm (1 + raw reads)
        elif "none":
            self._data.iloc[i, :] = np.log1p(self._data.iloc[i, :])

    # return self


# Apply scaling and fit data with model
def scaling(self, model_type="none", do_trim=False, do_scale=True, do_center=True, scale_max=10):
    """ Regress out reads per cell and identity """

    scaled = np.zeros((self._ncell, self._ngene))
    reads_per_cell = self._meta["reads_per_cell"]
    genes_per_cell = self._meta["genes_per_cell"]
    ident = self._meta["orig_ident"]
    group = self._meta["group"]

    if model_type is "none":
        scaled = self._data.values.copy()
    else:
        # for each gene
        for i in range(self._ngene):
            expr = self._data.iloc[:, i]  # expression value for each gene across all cells
            d = pd.DataFrame(np.array((expr.astype(np.float), reads_per_cell, genes_per_cell, ident, group)).T,
                             columns=["expr", "reads_per_cell", "genes_per_cell", "orig_ident", "group"])
            # fit linear model
            if model_type is "linear":
                results = smf.ols('expr ~ reads_per_cell + orig_ident + group', data=d).fit()
                # print(results.summary())
                scaled[:, i] = results.resid
            # fit poisson distribution
            elif model_type is "poisson":
                results = smf.glm('expr ~ reads_per_cell + orig_ident + group', data=d,
                                  family=sm.families.Poisson()).fit()
                # print(results.summary())
                scaled[:, i] = results.resid_pearson

    self._scaled = pd.DataFrame(scaled, columns=self._data.columns, index=self._data.index)

    if do_trim:
        x = self._scaled.mean(axis=0)
        y = self._scaled.var(axis=0) / x  # variance
        plt.plot(x, y, '.')
        good_genes = np.array(np.logical_and(x.values > 0.1, y.values > 1))
        self._scaled = self._scaled.iloc[:, good_genes]

    if do_center or do_scale:
        # for each gene
        temp_max = []
        for i in range(self._scaled.shape[1]):
            temp = self._scaled.iloc[:, i].values
            # from sklearn.preprocessing
            # Center to the mean and component wise scale to unit variance.
            temp = scale(temp, with_mean=do_center, with_std=do_scale)
            temp_max.append(max(temp))
            temp[temp > scale_max] = scale_max
            self._scaled.iloc[:, i] = temp

        # plt.plot(temp_max, '.')
        # plt.axhline(scale_max, color='r')

    # return self


# ==== DIMENSIONALITY REDUCTION ====
# Perform PCA
def run_pca(self, n_components=10, use_cells=None, use_genes=None, use_corr=False):
    # subset by genes
    if use_genes is not None:
        d = self._scaled.loc[:, use_genes].dropna(axis=1)
    else:
        d = self._scaled

    if self._active_cells is not None:
        d = d.iloc[self._active_cells, :].dropna(axis=0)
    # if use_cells is not None:
    #     d = d.iloc[use_cells, :].dropna(axis=0)

    # use Pearson product-moment correlation coefficients
    if use_corr:
        d = np.corrcoef(d)

    self._pca = PCA(n_components=n_components).fit(d)
    self._transformed_pca = self._pca.transform(d)

    # return self


# Perform tSNE
def run_tsne(self, max_pc=5, perplexity=20):
    """
    :param self:
    :param max_pc:
    :param perplexity: it is related to the number of nearest neighbors that is used in other manifold learning algorithms.
    :return:
    n_components: Dimension of the embedded space
    Larger dataset usually requires a larger perplexity.
    Consider selecting a value between 5 and 50. Different values can result in significanlty different results.
    random_state: If int, random_state is the seed used by the random number generator
    """
    self._tsne = TSNE(n_components=3, perplexity=perplexity, random_state=1).fit_transform(self._transformed_pca[:, :max_pc])


# Perform UMAP
def run_umap(self, max_pc=5, n_neighbors=10, min_dist=0.3, metric="euclidean"):
    """
    :param self:
    :param max_pc:
    :param n_neighbors: the higher the number, the more global structure we got
    :param min_dist: controls how tightly UMAP is allowed to pack points together
    :param metric:
    :return:
    """
    self._tsne = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric).fit_transform(self._transformed_pca[:, :max_pc])


# ==== CLUSTERING ====
# Hierarchical Density-Based Spatial Clustering of Applications with Noise
def cluster_hdbscan(self, max_pc=5):
    """
    :param self:
    :param max_pc:
    :return:
    """
    # min_cluster_size: the smallest size grouping that you wish to consider a cluster
    # clusterer = hdbscan.RobustSingleLinkage(cut=0.125, k=7)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=10, alpha=2.)
    self._clusts = np.array(clusterer.fit_predict(self._tsne))


# Density-Based Spatial Clustering of Applications with Noise
def cluster_dbscan(self, eps=0.5):
    """
    :param self:
    :param eps: The maximum distance between two samples for them to be considered as in the same neighborhood.
    :return:
    """
    self._clusts = np.array(DBSCAN(eps=eps).fit_predict(self._tsne))


# Unsupervised Shared Nearest Neighbors
def cluster_snn(self, max_pc=5, k=30):
    data = self._transformed_pca[:, :max_pc]
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree').fit(data)
    neighbor_graph = nbrs.kneighbors_graph(data)
    g = ig.Graph()
    g = ig.GraphBase.Adjacency(neighbor_graph.toarray().tolist(), mode=ig.ADJ_UNDIRECTED)
    sim = np.array(g.similarity_jaccard())
    g = ig.GraphBase.Weighted_Adjacency(sim.tolist(), mode=ig.ADJ_UNDIRECTED)
    self._clusts = np.array(g.community_multilevel(weights="weight", return_levels=False))


# Gaussian Mixture
def cluster_gmm(self, n_clusts=5, max_pc=5):
    model = GaussianMixture(n_components=n_clusts).fit(self._transformed_pca[:, :max_pc])
    self._clusts = np.array(model.predict(self._transformed_pca[:, :max_pc]))


# ==== DIFFERENTIAL EXPRESSION ====
# Find all marker genes
def find_all_markers(self, test="bimod", use_genes=None, only_pos=True,
                     log_fc_thresh=0.25, min_pct=0.1, fdr_correct=True):
    dfs = []
    # for each cluster
    for i in np.unique(self._clusts):
        if i >= 0:
            markers = find_markers(self, i, test=test, use_genes=use_genes, only_pos=only_pos,
                                        log_fc_thresh=log_fc_thresh, min_pct=min_pct)
            markers['cluster'] = i
            dfs.append(markers.sort_values(["pval", "log_fc"]))
    dfs = pd.concat(dfs)
    if fdr_correct:
        _, qvals, _, _ = multipletests(dfs["pval"], method="fdr_bh")
        dfs["pval"] = qvals
    return dfs


# Find marker gene of specific cluster
def find_markers(self, clust0, clust1=None, test="bimod", use_genes=None, only_pos=True, log_fc_thresh=0.25, min_pct=0.1):
    # only_pos: only positive

    if use_genes is None:
        curr_data = self._data  # raw_data
    else:
        curr_data = self._data.loc[:, use_genes]

    if self._active_cells is not None:
        curr_data = curr_data.iloc[self._active_cells, :]

    cells1 = np.argwhere(self._clusts == clust0).flatten()

    # reference cells
    if clust1 is not None:
        cells2 = np.argwhere(self._clusts == clust1).flatten()
    else:
        cells2 = np.argwhere(self._clusts != clust0).flatten()

    # select genes based on being expressed in a minimum fraction of cells
    fraction1 = (curr_data.iloc[cells1, :] > 0).sum(axis=0)/float(len(cells1))  # number of cells expressed each gene / total number of cells
    fraction2 = (curr_data.iloc[cells2, :] > 0).sum(axis=0)/float(len(cells2))
    good_frac = np.logical_or(fraction1 > min_pct, fraction2>min_pct)
    fraction1 = fraction1[good_frac]
    fraction2 = fraction2[good_frac]
    curr_data = curr_data.loc[:, good_frac]

    # select genes based on FC
    # np.expm1: Calculate exp(x) - 1 for all elements in the array
    log_fc = np.array([np.log1p(np.expm1(curr_data.iloc[cells1, i].values).mean()) - np.log1p(np.expm1(curr_data.iloc[cells2, i].values).mean()) for i in range(curr_data.shape[1])])

    # only positive (only up-regulated)
    if only_pos:
        good_fc = log_fc > log_fc_thresh
    else:
        good_fc = np.abs(log_fc) > log_fc_thresh

    # get good genes
    curr_data = curr_data.iloc[:, good_fc]
    log_fc = log_fc[good_fc]
    fraction1 = fraction1[good_fc]
    fraction2 = fraction2[good_fc]

    # run statistical test
    # Calculate the T-test for the means of two independent samples of scores.
    # This is a two-sided test for the null hypothesis that 2 independent samples have identical average (expected) values.
    # This test assumes that the populations have identical variances by default.
    # Get the two-tailed p-value.
    if test == "t":
        pvals = [ttest_ind(curr_data.iloc[cells1,i], curr_data.iloc[cells2,i])[1] for i in range(curr_data.shape[1])]
    elif test == "bimod":
        pvals = [ut.differential_lrt(curr_data.iloc[cells1,i].values, curr_data.iloc[cells2,i].values) for i in range(curr_data.shape[1])]
    # Compute the Wilcoxon rank-sum statistic for two samples.
    # The Wilcoxon rank-sum test tests the null hypothesis that two sets of measurements are drawn from the same distribution.
    # The alternative hypothesis is that values in one sample are more likely to be larger than the values in the other sample.
    elif test == "wilcox":
        pvals = [ranksums(curr_data.iloc[cells1,i].values, curr_data.iloc[cells2,i].values)[1] for i in range(curr_data.shape[1])]

    d = pd.DataFrame(data=np.array((pvals, log_fc, fraction1, fraction2)).T, columns=["pval", "log_fc", "pct.1", "pct.2"], index=curr_data.columns)
    return d


# Get top genes for each cluster
def get_top_markers_of_cluster(markers, n=5, pval_thresh=1e-6, return_unique=False):
    top_genes = []
    clusts = np.unique(markers["cluster"])
    for c in clusts:
        curr_genes = markers[markers["cluster"] == c]
        curr_genes = curr_genes[curr_genes["pval"] < pval_thresh]
        top_genes.extend(list(curr_genes.index[:n]))
    if return_unique:
        dup_top_genes = top_genes
        top_genes = []
        for i in dup_top_genes:
            if i not in top_genes:
                top_genes.append(i)
        # top_genes = list(set(top_genes))
    return top_genes


# Compare expression profile between groups
def compare_expression_between_groups(self, test="bimod", use_genes=None, use_raw=False):
    """
    Get log FC and P-value for each gene for each cluster between groups
    return: dataframe containing each cluster and number of cells
    """
    group_vals = np.unique(self._meta["group"].values)
    cluster_df = []
    ncells = []
    for c in np.unique(self._clusts):
        cells = self.get_cells_by_cluster(c, use_raw=use_raw)
        meta = self.get_metadata_by_cluster(c)
        cells0 = cells.iloc[ut.get_subset_index(meta["group"], group_vals[0]), :]
        cells1 = cells.iloc[ut.get_subset_index(meta["group"], group_vals[1]), :]

        if use_genes is not None:
            cells0 = cells0.loc[:, use_genes]
            cells1 = cells1.loc[:, use_genes]

        # log fold change for each gene for this cluster
        if use_raw:
            log_fc = np.array([np.log2((0.12+cells0.iloc[:,i].values.mean()))-np.log2(0.12+cells1.iloc[:, i].values.mean()) for i in range(cells0.shape[1])])
        else:
            log_fc = np.array([np.log1p(np.expm1(cells0.iloc[:,i].values).mean()) - np.log1p(np.expm1(cells1.iloc[:, i].values).mean()) for i in range(cells0.shape[1])])

        if test == "bimod":
            pvals = [ut.differential_lrt(cells0.iloc[:,i].values, cells1.iloc[:, i].values) for i in range(cells1.shape[1])]
        elif test == "t":
            pvals = [ttest_ind(cells0.iloc[:,i].values, cells1.iloc[:, i].values)[1] for i in range(cells1.shape[1])]
        elif test == "wilcox":
            pvals = [ranksums(cells0.iloc[:,i].values, cells1.iloc[:, i].values)[1] for i in range(cells1.shape[1])]

        d = pd.DataFrame(data=np.array((cells0.mean(), cells1.mean(), pvals, log_fc, np.repeat(c, cells0.shape[1]).astype(np.int))).T, columns=["mean0", "mean1", "pval", "log_fc", "cluster"], index=cells0.columns)
        _, d["pval"], _, _ = multipletests(d["pval"], method="fdr_bh")
        d = d.sort_values(["pval", "log_fc"])
        cluster_df.append(d)
        ncells.append((cells0.shape[0], cells1.shape[0]))
    return pd.concat(cluster_df), ncells


# ==== VISUALIZATION ====
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

