import numpy as np
from anndata import AnnData
import anndata as ad
import scanpy as sc
import pandas as pd
from typing import Optional
import scipy.sparse
from typing import Optional, Union
from sklearn.preprocessing import normalize
import sklearn.utils.extmath
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import dijkstra
from scipy.sparse import csr_matrix

###
### Alignment
###

def load_data(path):
    """
    Load data.
    
    Args:
        path: Path of anndata.
    """
    ann = ad.read_h5ad(path)
    return ann

def process_anndata(adata, highly_variable_genes=True, normalize_total=True,
                    log1p=True, scale=True, pca=True, neighbors=True,
                    umap=True, n_top_genes=3000, n_comps=100,
                    mode='rna', ndim=30):

    # Processing for RNA data
    if mode == 'rna':
        print("Processing RNA data...")
        
        if highly_variable_genes:
            print("Identifying highly variable genes...")
            sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor="seurat_v3", span=0.6, min_disp=0.1)

        if normalize_total:
            print("Normalizing total counts...")
            adata.layers["counts"] = adata.X.copy()
            sc.pp.normalize_total(adata)

        if log1p:
            print("Applying log1p transformation...")
            sc.pp.log1p(adata)

        print("Saving pre-log1p counts to a layer...")
        adata.layers["data"] = adata.X.copy()

        if scale:
            print("Scaling the data...")
            sc.pp.scale(adata)

        if pca:
            print("Performing PCA...")
            sc.tl.pca(adata, n_comps=n_comps, svd_solver="auto")
            
        if neighbors:
            print("Calculating neighbors based on cosine metric...")
            sc.pp.neighbors(adata, metric="cosine", n_pcs=ndim)
            
    # Processing for ATAC data
    elif mode == 'atac':
        print("Processing ATAC data...")
        
        print("Running LSI...")
        run_lsi(adata, n_components=n_comps, n_iter=15)
        
        if neighbors:
            print("Calculating neighbors based on cosine metric using X_lsi...")
            sc.pp.neighbors(adata, metric="cosine", use_rep="X_lsi", n_pcs=ndim)
            
    if umap:
        print("Performing UMAP...")
        sc.tl.umap(adata)
    
    print("Processing completed.")
    return adata

def calculate_gene(adata, top_marker_num=100, maker_by="leiden"):
    """
    Calculate marker genes based on groupings/clusters in the AnnData object.

    Args:
        adata (AnnData): AnnData object.
        top_marker_num (int): Number of top marker genes to retrieve. Defaults to 100.
        maker_by (str): Key for group information based on which marker genes will be calculated. Defaults to "leiden".

    Returns:
        adata (AnnData): Updated AnnData object with a new 'group_marker' variable in `.var` attribute.
    """
    sc.tl.rank_genes_groups(adata, maker_by, method='wilcoxon', use_raw=False)
    marker_df = pd.DataFrame(adata.uns['rank_genes_groups']['names']).head(top_marker_num)
    marker_array = np.array(marker_df)
    marker_array = np.ravel(marker_array)
    marker_array = np.unique(marker_array)
    marker = list(marker_array)
    group_marker_genes = marker
    is_group_marker = pd.Series(False, index=adata.var_names)
    is_group_marker[group_marker_genes] = True
    adata.var['group_marker'] = is_group_marker.values
    return adata

def find_marker(
    adata1: AnnData, adata2: AnnData, 
    top_marker_num: int = 10, 
    marker1_by: str = "leiden", 
    marker2_by: str = "leiden", 
    min_cells: int = 0
) -> tuple:
    """
    Find marker genes between two AnnData objects based on specified groupings/clusters.

    Args:
        adata1 (AnnData): First AnnData object.
        adata2 (AnnData): Second AnnData object.
        top_marker_num (int): Number of top marker genes to retrieve. Defaults to 10.
        marker1_by (str): Key for group information in adata1 based on which marker genes will be calculated. Defaults to "leiden".
        marker2_by (str): Key for group information in adata2 based on which marker genes will be calculated. Defaults to "leiden".
        min_cells (int): Minimum number of cells expressing a gene to be considered. Defaults to 0.

    Returns:
        Tuple containing updated adata1 and adata2 with calculated marker genes based on common genes.
    """
    print("Finding marker genes...")

    # Filter genes based on min_cells threshold
    if min_cells > 0:
        print(f"Filtering genes with minimum {min_cells} cells...")
        sc.pp.filter_genes(adata1, min_cells=min_cells)
        sc.pp.filter_genes(adata2, min_cells=min_cells)   
    
    # Select common genes between adata1 and adata2
    common_genes = np.intersect1d(adata1.var.index, adata2.var.index)
    print(f"Number of common genes: {len(common_genes)}")
    adata1 = adata1[:, common_genes]
    adata2 = adata2[:, common_genes]
    
    # Calculate marker genes for adata1 and adata2 based on specified groupings
    print(f"Calculating marker genes based on '{marker1_by}' for adata1...")
    adata1 = calculate_gene(adata1, top_marker_num=top_marker_num, maker_by=marker1_by)
    
    print(f"Calculating marker genes based on '{marker2_by}' for adata2...")
    adata2 = calculate_gene(adata2, top_marker_num=top_marker_num, maker_by=marker2_by)
    
    print("Marker gene calculation completed.")
    return adata1, adata2

def average_expression(adata: AnnData, avg_by: str = 'leiden', layer: str = None) -> pd.DataFrame:
    """
    Calculate the average gene expression for each category/group defined by 'avg_by'.

    Args:
        adata (AnnData): AnnData object.
        avg_by (str): Key for grouping categories based on which average expression will be calculated. Defaults to 'leiden'.
        layer (str): Optional - Layer of data to calculate average expression from. Defaults to None (use adata.X).

    Returns:
        mean_expression_df (pd.DataFrame): DataFrame containing mean expression values for each category.
    """
    unique_categories = np.unique(adata.obs[avg_by])
    mean_expression_by_category = []
    
    if layer is None:
        data_to_avg = adata.X
    elif layer in adata.layers:
        data_to_avg = adata.layers[layer]
    else:
        raise ValueError(f"Layer '{layer}' not found in adata.layers.")
    
    for category in unique_categories:
        category_indices = adata.obs[avg_by] == category
        gene_expression = data_to_avg[category_indices]
        mean_expression = np.mean(gene_expression, axis=0)
        mean_expression = np.asarray(mean_expression).ravel()
        mean_expression_by_category.append(mean_expression)
    
    mean_expression_df = pd.DataFrame(mean_expression_by_category, columns=adata.var_names, index=unique_categories)
    return mean_expression_df


def random_sample_cells(adata, num_each_type=None, cell_types=None, num_random=None, cell_type_num=None):
    """
    Randomly sample cells from an AnnData object based on different criteria.

    Args:
        adata (AnnData): AnnData object.
        num_each_type (list or None): List specifying the number of cells to sample from each cell type.
        cell_types (list or None): List of cell types for sampling.
        num_random (int or None): Number of cells to randomly sample.
        cell_type_num (int or None): Number of cell types to sample.

    Returns:
        sample_adata (AnnData): AnnData object containing the sampled cells.
        cell_indices (list): List of indices corresponding to the sampled cells in the original AnnData.
    """
    
    if num_random is not None:
        if num_random > len(adata):
            raise ValueError(f"Requested {num_random} cells, but only {len(adata)} cells available.")
        cell_indices = np.random.choice(len(adata), num_random, replace=False)
    elif cell_type_num is not None:
        all_cell_types = adata.obs['cell_type'].unique()
        if cell_type_num > len(all_cell_types):
            raise ValueError(f"Requested {cell_type_num} cell types, but only {len(all_cell_types)} cell types available.")
        
        selected_cell_types = np.random.choice(all_cell_types, cell_type_num, replace=False)
        cell_indices = []
        for cell_type in selected_cell_types:
            indices = np.where(adata.obs['cell_type'] == cell_type)[0]
            if len(indices) < 1:
                raise ValueError(f"Cell type {cell_type} has no cells.")
            cell_indices.extend(indices)  # Add all cells of the selected cell type
            
    else:
        cell_indices = []
        for cell_type, n in zip(cell_types, num_each_type):
            indices = np.where(adata.obs['cell_type'] == cell_type)[0]
            if len(indices) < n:
                raise ValueError(f"Cell type {cell_type} only has {len(indices)} cells, less than {n} cells requested")
            cell_indices.extend(np.random.choice(indices, n, replace=False))
    
    # Extract the data of the sampled cells from the original anndata
    sample_adata = adata[cell_indices].copy()
    
    return sample_adata, cell_indices

def shuffle_obs(adata):
    """
    Shuffle the observations in the given AnnData object.

    Args:
        adata (AnnData): AnnData object.

    Returns:
        shuffled_adata (AnnData): AnnData object with observations shuffled.
    """
    obs_names = adata.obs_names
    shuffled_indices = np.random.permutation(len(obs_names))
    shuffled_adata = adata[shuffled_indices, :]
    return shuffled_adata

def extract_exp(data, layer=None, gene = None):
    """
    Extract gene expression data from the given data object.

    Args:
        data (AnnData): AnnData object.
        layer (str): Optional - Layer of data from which to extract expression data. Defaults to None (use data.X).
        gene (str or list): Optional - Gene name or list of gene names to extract expression data for.

    Returns:
        exp_data (pd.DataFrame): DataFrame containing gene expression data.
    """
    if layer is None:
        expression_data = data.X
    elif layer in data.layers:
        expression_data = data.layers[layer].toarray()
    else:
        raise ValueError(f"Layer '{layer}' not found in data.layers.")

    exp_data = pd.DataFrame(expression_data)
    exp_data.columns = data.var.index.tolist()
    exp_data.index = data.obs.index.tolist()
    
    if gene is not None:
        exp_data = exp_data.loc[:,gene]
    
    return exp_data

def extract_reduction(data: AnnData, use_rep: str = 'reduction') -> pd.DataFrame:
    """
    Extract the reduced dimensions (e.g., PCA, tSNE) from an AnnData object.

    Args:
        data (AnnData): AnnData object.
        use_rep (str): Key for the representation in the 'obsm' attribute of AnnData objects. Defaults to 'reduction'.

    Returns:
        reduction_df (pd.DataFrame): DataFrame containing the reduced dimensions.
    """
    reduction_df = pd.DataFrame(data.obsm[use_rep])
    reduction_df.index = data.obs.index.tolist()
    return reduction_df

def scale_dataframe_to_01(df: pd.DataFrame) -> pd.DataFrame:
    """
    Scale values in a DataFrame to the range [0, 1].

    Args:
        df (pd.DataFrame): Input DataFrame containing numerical values.

    Returns:
        scaled_df (pd.DataFrame): DataFrame with values scaled to the range [0, 1].
    """
    min_value = df.min().min()
    max_value = df.max().max()
    scaled_df = (df - min_value) / (max_value - min_value)
    return scaled_df

Array = Union[np.ndarray, scipy.sparse.spmatrix]
def tfidf(X: np.ndarray) -> np.ndarray:
    """
    Calculate TF-IDF (Term Frequency-Inverse Document Frequency) matrix.

    Args:
        X (np.ndarray): Input matrix or sparse matrix.

    Returns:
        np.ndarray: TF-IDF weighted matrix.
    """
    idf = X.shape[0] / X.sum(axis=0)
    if scipy.sparse.issparse(X):
        tf = X.multiply(1 / X.sum(axis=1))
        return tf.multiply(idf)
    else:
        tf = X / X.sum(axis=1, keepdims=True)
        return tf * idf

def run_lsi(
    adata: AnnData, n_components: int = 20,
    use_highly_variable: Optional[bool] = None, **kwargs
) -> None:
    """
    Run Latent Semantic Indexing (LSI) on input AnnData object.

    Args:
        adata (AnnData): Annotated data object.
        n_components (int): Number of components for LSI.
        use_highly_variable (bool, optional): Whether to use highly variable genes. Defaults to None.
        **kwargs: Additional keyword arguments for randomized_svd.
    """
    if "random_state" not in kwargs:
        kwargs["random_state"] = 0  # Keep deterministic as the default behavior
    if use_highly_variable is None:
        use_highly_variable = "highly_variable" in adata.var
    adata_use = adata[:, adata.var["highly_variable"]] if use_highly_variable else adata
    X = tfidf(adata_use.X)
    X_norm = normalize(X, norm="l1")
    X_norm = np.log1p(X_norm * 1e4)
    X_lsi = sklearn.utils.extmath.randomized_svd(X_norm, n_components, **kwargs)[0]
    X_lsi -= X_lsi.mean(axis=1, keepdims=True)
    X_lsi /= X_lsi.std(axis=1, ddof=1, keepdims=True)
    adata.obsm["X_lsi"] = X_lsi

###
### Alignment
###

def construct_graph(X, k, mode= "connectivity", metric="minkowski",p=2):
    """
    Construct graph with KNN.
    
    Args:
        X: Input data containing features.
        k (int): Number of neighbors for each data point.
        mode (str): Optional - Mode for constructing the graph, either 'connectivity' or 'distance'. Defaults to 'connectivity'.
        metric (str): Optional - Name of the distance metric to use. Defaults to 'minkowski'.
        p (int): Optional - Parameter for the Minkowski metric. Defaults to 2.

    Returns:
        -The knn graph of input data. 
    """
    assert (mode in ["connectivity", "distance"]), "Norm argument has to be either one of 'connectivity', or 'distance'. "
    if mode=="connectivity":
        include_self=True
    else:
        include_self=False
    c_graph=kneighbors_graph(X, k, mode=mode, metric=metric, include_self=include_self,p=p)
    return c_graph

def distances_cal(graph):
    """
    Calculate distance between cells/spots based on graph.
    
    Args:
        graph: KNN graph.
        
    Returns:
        -The distance matrix of cells/spots. 
    """
    
    shortestPath=dijkstra(csgraph= csr_matrix(graph), directed=False, return_predecessors=False)
    the_max=np.nanmax(shortestPath[shortestPath != np.inf])
    shortestPath[shortestPath > the_max] = the_max
    C_dis=shortestPath/shortestPath.max()
    C_dis -= np.mean(C_dis)
    return C_dis

def intersect_datasets(data1, data2, by="annotation"):
    """
    Get the intersection of two AnnData objects based on annotation.

    Args:
        data1 (AnnData): The first AnnData object.
        data2 (AnnData): The second AnnData object.

    Returns:
        AnnData: The intersection of data1 and data2.
    """
    # Find the intersection of annotations
    common_annotations = list(set(data1.obs[by]).intersection(data2.obs[by]))

    # Filter data1 and data2 based on the common_annotations
    intersected_data1 = data1[data1.obs[by].isin(common_annotations)]
    intersected_data2 = data2[data2.obs[by].isin(common_annotations)]

    return intersected_data1, intersected_data2

def sort_datasets(adata1=None, adata2=None, suffix1=None, suffix2=None):
    """
    Sort and subset two AnnData objects based on their cell names.

    Args:
    - adata1 (AnnData): First AnnData object.
    - adata2 (AnnData): Second AnnData object.
    - suffix1 (str): Suffix used in the cell names of adata1 for sorting.
    - suffix2 (str): Suffix used in the cell names of adata2 for sorting.

    Returns:
    - Tuple containing sorted and subsetted adata1 and adata2 based on matching cell names.
    """
    cell_names1 = sorted([cell_name[:-len(suffix1)] for cell_name in adata1.obs_names if cell_name.endswith(suffix1)])
    cell_names2 = sorted([cell_name[:-len(suffix2)] for cell_name in adata2.obs_names if cell_name.endswith(suffix2)])
    merged_cell_names = intersect(cell_names1,cell_names2)
    adata1 = adata1[[cell_name + suffix1 for cell_name in merged_cell_names], :]
    adata2 = adata2[[cell_name + suffix2 for cell_name in merged_cell_names], :]
    return adata1, adata2

def intersect_cells(adata1=None, adata2=None, suffix1=None, suffix2=None):
    """
    Intersect and subset two AnnData objects based on intersecting cell names.

    Args:
        adata1 (AnnData): First AnnData object.
        adata2 (AnnData): Second AnnData object.
        suffix1 (str): Suffix used in the cell names of adata1 for intersecting cells.
        suffix2 (str): Suffix used in the cell names of adata2 for intersecting cells.

    Returns:
        Tuple containing intersected and subsetted adata1 and adata2 based on matching cell names.
    """
    cell_names1 = sorted([cell_name[:-len(suffix1)] for cell_name in adata1.obs_names if cell_name.endswith(suffix1)])
    cell_names2 = sorted([cell_name[:-len(suffix2)] for cell_name in adata2.obs_names if cell_name.endswith(suffix2)])
    merged_cell_names = list(zip(cell_names1, cell_names2))
    adata1 = adata1[[cell_name + suffix1 for cell_name, _ in merged_cell_names], :]
    adata2 = adata2[[cell_name + suffix2 for _, cell_name in merged_cell_names], :]
    return adata1, adata2

def intersect(lst1, lst2): 
    """
    Gets and returns intersection of two lists.

    Args:
        lst1: List
        lst2: List
    
    Returns:
        lst3: List of common elements.
    """
    
    temp = set(lst2)
    lst3 = [value for value in lst1 if value in temp]
    return lst3 

def subset_ndimension(adata1=None, adata2=None,dim_num =None):
    """
    Subset the n dimensions from reduction data in two AnnData objects.

    Args:
        adata1 (anndata.AnnData): First AnnData object.
        adata2 (anndata.AnnData): Second AnnData object.
        dim_num (int): Number of dimensions to subset. Defaults to None.

    Returns:
        adata1 (anndata.AnnData): Updated AnnData object 1 with subsetted dimensions.
        adata2 (anndata.AnnData): Updated AnnData object 2 with subsetted dimensions.
    """
    if dim_num is None:
        dim_num = min(adata1.obsm['reduction'].shape[1], adata2.obsm['reduction'].shape[1])
    if isinstance(adata1.obsm['reduction'], pd.DataFrame):
        adata1.obsm['reduction'] = adata1.obsm['reduction'].iloc[:, :dim_num]
        adata2.obsm['reduction'] = adata2.obsm['reduction'].iloc[:, :dim_num]
    elif isinstance(adata1.obsm['reduction'], np.ndarray):
        adata1.obsm['reduction'] = adata1.obsm['reduction'][:, :dim_num]
        adata2.obsm['reduction'] = adata2.obsm['reduction'][:, :dim_num]
    return adata1, adata2

def top_n(df, n=3, column='APM'):
    """
    Get a subset of the DataFrame according to the values of a column.
    
    """
    return df.sort_values(by=column, ascending=False)[:n]

def save_integrated_data(adata, name, path):
    integrated_emb = extract_reduction(adata, use_rep='integrated')
    integrated_umap = extract_reduction(adata, use_rep='X_umap')
    integrated_emb.to_csv(f"{path}/{name}_integrated_emb.csv", index=True)
    integrated_umap.to_csv(f"{path}/{name}_integrated_umap.csv", index=True)
    adata.obs.to_csv(f"{path}/{name}_integrated_obs.csv", index=True)
    adata.write_h5ad(f"{path}/{name}_integrated.h5ad")