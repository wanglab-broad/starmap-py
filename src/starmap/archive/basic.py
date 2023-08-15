"""
This file contains deprecated functions, some of them might not be working any more due to the changes of the overall package architecture. 

"""

# Import related packages 
import os
import numpy as np
import pandas as pd
import textwrap as tw
from anndata import AnnData
from matplotlib import pyplot as plt

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