import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.cross_decomposition import CCA
import ot
from ot import emd
from ot.partial import gwloss_partial, gwgrad_partial
from .helper import *

def map_test_to_reference(data, transfer_df, type_in_data='leiden'):
    """
    Map labels from test data to reference data based on a transfer probability DataFrame.

    Args:
        data (AnnData): Test data (AnnData object).
        transfer_df (DataFrame): DataFrame containing transfer probabilities between test and reference data.
        type_in_data (str): Key for the annotation type in the test data. Defaults to 'leiden'.

    Returns:
        filtered_data (AnnData): Test data with updated labels transferred from reference data.
    """
    melted_df = transfer_df.reset_index().melt(id_vars='index', var_name='Cell_Type', value_name='score')
    sorted_df = melted_df.sort_values(by='score', ascending=False)
    sorted_df = sorted_df.rename(columns={sorted_df.columns[0]: 'reference', sorted_df.columns[1]: 'test'})
    sorted_df = sorted_df[sorted_df['score'] > 0]
    
    filtered_data = data[data.obs[type_in_data].isin(sorted_df['test'].unique())]

    mapping_dict = {}
    for test_cell_type in sorted_df['test'].unique():
        test_rows = sorted_df[sorted_df['test'] == test_cell_type]
        best_match = test_rows[test_rows['score'] == test_rows['score'].max()]
        if not best_match.empty:
            mapping_dict[test_cell_type] = best_match['reference'].values[0]
    filtered_data.obs['annotation'] = filtered_data.obs[type_in_data].map(mapping_dict)
    
    return filtered_data        
        
def scotch_transfer(test_data,
                    reference_data,
                    layer=None,
                    test_avg_by = 'leiden',
                    reference_avg_by = 'leiden',
                    reg = 1,
                    cutoff = 0.5):
    """
    Perform transfer of labels from reference_data to test_data based on gene expression correlations.

    Args:
        test_data (AnnData): Data containing test samples.
        reference_data (AnnData): Data containing reference samples.
        layer (str, optional): Layer to use from AnnData object. Default is None.
        test_avg_by (str, optional): Method to average test data by. Default is 'leiden'.
        reference_avg_by (str, optional): Method to average reference data by. Default is 'leiden'.
        reg (int, optional): Regularization parameter for unbalanced optimal transport. Default is 1.
        cutoff (float, optional): Cutoff value for mapping probabilities. Default is 0.5.

    Returns:
        test_data (AnnData): Test data with transferred labels.
        reference_data (AnnData): Reference data.
        transfer_df (DataFrame): DataFrame containing transfer probabilities.
    """
    print("Performing label transfer...")

    # Extract common genes in both datasets
    lst1 = list(test_data.var.index[test_data.var['group_marker']])
    lst2 = list(reference_data.var.index[reference_data.var['group_marker']])
    lst1.extend(lst2)
    deg_list = np.unique(lst1).tolist()
    
    # Calculate average expression for test and reference data
    print("Calculating average expression...")
    test_exp = average_expression(test_data[:, deg_list], layer=layer, avg_by=test_avg_by).T
    reference_exp = average_expression(reference_data[:, deg_list], layer=layer, avg_by=reference_avg_by).T
    
    # Compute correlation matrix
    print("Computing correlation matrix...")
    exp1 = reference_exp.T
    exp2 = test_exp.T
    correlation_df = pd.DataFrame(index=exp1.index, columns=exp2.index)
    for cell_type1 in exp1.index:
        for cell_type2 in exp2.index:
            gene_expr1 = exp1.loc[cell_type1]
            gene_expr2 = exp2.loc[cell_type2]
            correlation, _ = pearsonr(gene_expr1, gene_expr2)
            correlation_df.loc[cell_type1, cell_type2] = correlation
    correlation_df = scale_dataframe_to_01(correlation_df).astype('float')
    
    # Compute transfer probabilities using unbalanced optimal transport
    print("Performing unbalanced optimal transport...")
    M = 1 - correlation_df.values
    a = np.ones((correlation_df.shape[0],)) / correlation_df.shape[0]
    b = np.ones((correlation_df.shape[1],)) / correlation_df.shape[1]
    
    l2_uot = ot.unbalanced.mm_unbalanced(a, b, M, reg, div='kl')
    transfer_df = pd.DataFrame(l2_uot, index=correlation_df.index, columns=correlation_df.columns).astype(float)
    transfer_df = scale_dataframe_to_01(transfer_df).astype('float')
    
    # Apply cutoff to transfer probabilities and map labels
    print("Applying cutoff to transfer probabilities and mapping labels...")
    transfer_df = transfer_df.applymap(lambda x: 0 if x < cutoff else x)
    test_data = map_test_to_reference(test_data, transfer_df, type_in_data=test_avg_by)
    reference_data.obs['annotation'] = reference_data.obs[reference_avg_by]
    
    print("Label transfer completed.")
    return test_data, reference_data, transfer_df


def integrate_adata(adata1=None, 
                    adata2=None, 
                    df=None, 
                    n_components=30, 
                    cca_components=10, 
                    max_iter=2000, 
                    index_unique=None):
    """
    Integrate two AnnData objects using Canonical Correlation Analysis (CCA) and concatenate them.

    Args:
        adata1 (AnnData): First AnnData object.
        adata2 (AnnData): Second AnnData object.
        df (DataFrame): DataFrame containing cell mappings between adata1 and adata2.
        n_components (int): Number of components for dimensionality reduction. Defaults to 30.
        cca_components (int): Number of CCA components to retain. Defaults to 10.
        max_iter (int): Maximum number of iterations for CCA. Defaults to 2000.
        index_unique (str): Key for unique index in concatenated AnnData object. Defaults to None.

    Returns:
        adata1 (AnnData): First AnnData object with integrated information.
        adata2 (AnnData): Second AnnData object with integrated information.
        adata_merged (AnnData): Concatenated AnnData object after integration.
    """
    print("Integrating and concatenating AnnData objects...")

    df = df[df.duplicated('cell1') == False]
    df = df[df.duplicated('cell2') == False]
    
    adata1 = adata1[df.cell1]
    adata2 = adata2[df.cell2]
    
    emb_d1 = pd.DataFrame(adata1.obsm['reduction'],index=adata1.obs.index)
    emb_d2 = pd.DataFrame(adata2.obsm['reduction'],index=adata2.obs.index)
    
    emb_d1 = emb_d1.loc[df.cell1]
    emb_d2 = emb_d2.loc[df.cell2]
    
    #n_components = min(emb_d1.shape[1], emb_d2.shape[1])
    emb_d1 = emb_d1.iloc[:,0:n_components]
    emb_d2 = emb_d2.iloc[:,0:n_components]
    
    X, Y = emb_d1, emb_d2
    X = normalize(X)
    Y = normalize(Y)

    cca = CCA(n_components=cca_components, max_iter=max_iter)
    cca.fit(X, Y)
    cca_d1, cca_d2 = cca.transform(X, Y)
    
    cca_d1 = normalize(cca_d1)
    cca_d2 = normalize(cca_d2)
    
    cca_d1 = pd.DataFrame(cca_d1,index=emb_d1.index)
    cca_d2 = pd.DataFrame(cca_d2,index=emb_d2.index)
    
    cca_d1 = cca_d1.loc[adata1.obs.index]
    cca_d2 = cca_d2.loc[adata2.obs.index]
    
    adata1.obsm['integrated'] = cca_d1.values
    adata2.obsm['integrated'] = cca_d2.values

    adata_merged = adata1.concatenate(adata2, 
                                      batch_key='batch',
                                      index_unique=index_unique,
                                      join='outer')
    print("Integration and concatenation completed.")
    return adata1,adata2,adata_merged

def scotch_alignment(ann1, 
                     ann2, 
                     norm='l2', 
                     graph_mode='connectivity', 
                     k=10, 
                     backend=ot.backend.NumpyBackend(),
                     **kwargs):
    """
    Perform SCOTCH alignment between two AnnData objects.

    Args:
        ann1 (AnnData): The first AnnData object.
        ann2 (AnnData): The second AnnData object.
        top_num (int, optional): The number of top matches to select for each cell. Default is 5.

    Returns:
        final_result (pd.DataFrame): A DataFrame containing the top matching cell pairs with their values.
    """
    print("Performing SCOTCH alignment...")
    # Get the intersecting cell types
    intersect_type = intersect(ann1.obs['annotation'].unique().tolist(), ann2.obs['annotation'].unique().tolist())

    # Initialize an empty DataFrame to store the results
    result_df = pd.DataFrame()

    # Iterate over each intersect_type with a progress display
    for idx, cell_type in enumerate(intersect_type):
        print(f"Processing cell type {idx + 1}/{len(intersect_type)}: {cell_type}")

        # Select the corresponding cell types from ann1 and ann2
        ann1_subset = ann1[ann1.obs['annotation'] == cell_type]
        ann2_subset = ann2[ann2.obs['annotation'] == cell_type]

        # Perform uot_alignment on the subsets
        if k > min(ann1_subset.shape[0], ann2_subset.shape[0]):
            k = min(ann1_subset.shape[0], ann2_subset.shape[0])
            
        alignment_result = alignment(ann1_subset,
                                     ann2_subset,
                                     norm=norm,
                                     graph_mode=graph_mode,
                                     k=k, 
                                     backend=backend,
                                     **kwargs)
        alignment_result = pd.DataFrame(alignment_result, index=ann1_subset.obs_names, columns=ann2_subset.obs_names).astype(float)
        alignment_result['cell1'] = alignment_result.index
        alignment_result = alignment_result.melt(id_vars=['cell1'], var_name='cell2', value_name='value')
        alignment_result = alignment_result.sort_values(by="value", ascending=False)
        alignment_result = alignment_result[alignment_result['value'] > 0]
        
        #alignment_result = extract_unique_pairs(alignment_result)
        # Concatenate the alignment result with the existing results
        result_df = pd.concat([result_df, alignment_result])

    # Reset the index of the result DataFrame
    result_df.reset_index(drop=True, inplace=True)

    final_result = result_df[['cell1', 'cell2', 'value']]
    
    meta1_dict = {
        'cell1': pd.Series(ann1.obs.index.tolist(),index=ann1.obs.index.tolist()),
        'cell_type1': pd.Series(ann1.obs['cell_type'],index=ann1.obs.index.tolist())
    }
    meta1 = pd.DataFrame(meta1_dict)
    final_result = pd.merge(final_result, meta1, on='cell1',how="left")
    meta2_dict = {
        'cell2': pd.Series(ann2.obs.index.tolist(),index=ann2.obs.index.tolist()),
        'cell_type2': pd.Series(ann2.obs['cell_type'],index=ann2.obs.index.tolist())
    }
    meta2 = pd.DataFrame(meta2_dict)
    final_result = pd.merge(final_result, meta2, on='cell2',how="left")
    print("SCOTCH alignment completed.")
    return final_result

def partial_gromov_wasserstein(C1, C2, p, q, m=None, nb_dummies=1, G0=None,
                               thres=1, numItermax=1000, tol=1e-4,numItermax_emd=1e6,
                               log=False, verbose=False, **kwargs):
    if m is None:
        m = np.min((np.sum(p), np.sum(q)))
    elif m < 0:
        raise ValueError("Problem infeasible. Parameter m should be greater"
                         " than 0.")
    elif m > np.min((np.sum(p), np.sum(q))):
        m = np.min((np.sum(p), np.sum(q)))
        # raise ValueError("Problem infeasible. Parameter m should lower or"
        #                  " equal than min(|a|_1, |b|_1).")

    if G0 is None:
        G0 = np.outer(p, q)

    if nb_dummies < int(np.min((np.sum(p), np.sum(q)))/100):
        nb_dummies = int(np.min((np.sum(p), np.sum(q)))/100)
    
    dim_G_extended = (len(p) + nb_dummies, len(q) + nb_dummies)
    q_extended = np.append(q, [(np.sum(p) - m) / nb_dummies] * nb_dummies)
    p_extended = np.append(p, [(np.sum(q) - m) / nb_dummies] * nb_dummies)

    cpt = 0
    err = 1

    if log:
        log = {'err': []}

    while (err > tol and cpt < numItermax):

        Gprev = np.copy(G0)

        M = gwgrad_partial(C1, C2, G0)
        M_emd = np.zeros(dim_G_extended)
        M_emd[:len(p), :len(q)] = M
        M_emd[-nb_dummies:, -nb_dummies:] = np.max(M) * 1e2
        M_emd = np.asarray(M_emd, dtype=np.float64)

        Gc, logemd = emd(p_extended, q_extended, M_emd,numItermax =numItermax_emd,  log=True, **kwargs)

        if logemd['warning'] is not None:
            raise ValueError("Error in the EMD resolution: try to increase the"
                             " number of dummy points")

        G0 = Gc[:len(p), :len(q)]

        if cpt % 10 == 0:  # to speed up the computations
            err = np.linalg.norm(G0 - Gprev)
            if log:
                log['err'].append(err)
            if verbose:
                if cpt % 200 == 0:
                    print('{:5s}|{:12s}|{:12s}'.format(
                        'It.', 'Err', 'Loss') + '\n' + '-' * 31)
                print('{:5d}|{:8e}|{:8e}'.format(cpt, err,
                                                 gwloss_partial(C1, C2, G0)))

        deltaG = G0 - Gprev
        a = gwloss_partial(C1, C2, deltaG)
        b = 2 * np.sum(M * deltaG)
        if b > 0:  # due to numerical precision
            gamma = 0
            cpt = numItermax
        elif a > 0:
            gamma = min(1, np.divide(-b, 2.0 * a))
        else:
            if (a + b) < 0:
                gamma = 1
            else:
                gamma = 0
                cpt = numItermax

        G0 = Gprev + gamma * deltaG
        cpt += 1

    if log:
        log['partial_gw_dist'] = gwloss_partial(C1, C2, G0)
        return G0[:len(p), :len(q)], log
    else:
        return G0[:len(p), :len(q)]

def alignment(
    ann1: AnnData,  
    ann2: AnnData,
    p_distribution = None, 
    q_distribution = None, 
    norm: str = 'l2', 
    backend = ot.backend.NumpyBackend(),  
    return_obj: bool = False,
    k: int = 10,
    graph_mode: str = "connectivity",
    **kwargs):
    """
    Aligns two AnnData objects using partial Gromov-Wasserstein alignment.

    Args:
        ann1 (AnnData): First AnnData object.
        ann2 (AnnData): Second AnnData object.
        p_distribution (array-like, optional): Probability distribution for ann1. Defaults to None.
        q_distribution (array-like, optional): Probability distribution for ann2. Defaults to None.
        norm (str, optional): Normalization method. Defaults to 'l2'.
        backend: Backend used for calculations.
        return_obj (bool, optional): Whether to return the alignment object. Defaults to False.
        k (int, optional): Number of nearest neighbors for constructing the graph. Defaults to 10.
        graph_mode (str, optional): Mode for constructing the graph ('connectivity' or other). Defaults to "connectivity".
        **kwargs: Additional keyword arguments.

    Returns:
        pi (array-like): Optimal transport plan.
    """
    nx = backend
    n1 = ann1.shape[0]
    n2 = ann2.shape[0]
    
    mass = min(n1, n2) / max(n1, n2)
    print("mass : "+str(mass))
    
    # Construct the graph
    #num_columns = min(ann1.obsm['reduction'].shape[1], ann2.obsm['reduction'].shape[1])
    reduction_array_1 = ann1.obsm['reduction']
    reduction_array_2 = ann2.obsm['reduction']
    reduction_array_1 = normalize(reduction_array_1, norm=norm, axis=1)
    reduction_array_2 = normalize(reduction_array_2, norm=norm, axis=1)

    # Construct graphs for alignment
    print('Constructing ' + str(graph_mode) + '...')
    print('k = ' + str(k))
    Xgraph = construct_graph(reduction_array_1, k=k, mode=graph_mode)
    ygraph = construct_graph(reduction_array_2, k=k, mode=graph_mode)

    # Calculate distances and convert to NetworkX graphs
    Cx = distances_cal(Xgraph)
    Cy = distances_cal(ygraph)
    
    Cx = nx.from_numpy(Cx)
    Cy = nx.from_numpy(Cy)
    
    # Init distributions
    if p_distribution is None:
        p = np.ones((n1,)) / n1
        p = nx.from_numpy(p)
    else:
        p = nx.from_numpy(p_distribution)
    if q_distribution is None:
        q = np.ones((n2,)) / n2
        q = nx.from_numpy(q)
    else:
        q = nx.from_numpy(q_distribution)
    
    pi, log = partial_gromov_wasserstein(Cx, Cy, p, q, m=mass, log=True,**kwargs)
   
    print('Running OT...')

    if return_obj:
        return pi, log['partial_fgw_cost']
    return pi


