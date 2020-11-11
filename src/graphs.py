import h5py
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from skimage.morphology import binary_dilation
import graphviz  # https://pypi.org/project/graphviz/
from pathlib import Path


def img_to_graph(img, compute_node_features=None, compute_edge_features=None, node_external_features=None):
    """
    Converts array of grain ids to a graph of connected grains.

    *img* is an image where each pixel indicates the grain it belongs to.
     The grains are considered nodes on the graph, and the boundaries between
     adjacent grains are considered edges. The grain boundary network forms
     a simple undirected graph. The graph is described by its adjacency and
     degree matrices.

    Parameters
    --------------
    img: ndarray
        m x n array where each pixel is an integer indicating the grain it belongs to

    compute_node_features, compute_edge_features: callable or None
        if None- only the graph topology/connectivity is determined
        else- functions that take the image, neighbor list, and external features as inputs  # TODO clarify format

    node_external_features: array, dictionary, or None
        if None- external features are not incorporated in computing node and edge features
        if array- N_node x f array of features for each node
        if dictionary- keys are node_idx or tuple(source_node, target_node), value is n-element array of features


    Returns
    -------------
    G- networkx graph object

    Examples
    -------------

    References
    -------------
    Z. Liu, J. Zhou Introduction to Graph Neural Networks
      (https://www.morganclaypoolpublishers.com/catalog_Orig/samples/9781681737669_sample.pdf)

    """
    if compute_node_features is None:
        def compute_node_features(img, edges, node_features):
            return {}
    if compute_edge_features is None:
        def compute_edge_features(img, edges, node_features):
            return [(*e, {}) for e in edges]

    grain_ids = np.unique(img)
    n_grain = len(grain_ids)

    G = nx.Graph()  # simple, undirected graph

    img_pad = np.pad(img, 6,
                     mode='wrap')  # note that padding increases sizes of grains around edges- either use roll or preserve original image?
    for idx in grain_ids:
        grain_mask = img == idx  # selects single grain

        # to find neighbors of grain, apply binary dilation tho the mask and look for overlap
        grain_mask_dilate = binary_dilation(grain_mask, selem=np.ones((5, 5), np.int))

        # TODO find threshold for min number of shared pixels to be considered a neighbor?
        #      would have to be scaled by perimiter or something
        neighbors, counts = np.unique(img[grain_mask_dilate], return_counts=True)
        neighbor_mask = neighbors != idx  # a grain cannot be its own neighbor (ie no self loops on graph)
        neighbors = neighbors[neighbor_mask]
        counts = counts[neighbor_mask]
        # neighbors = neighbors[counts>thresh]  # implement if neighbors needs to be filtered by number of overlapping pixels

        source_idx = np.zeros(neighbors.shape, np.int) + idx
        edges = np.stack((source_idx, neighbors),
                         axis=0).T  # format: [[source_idx, neighbor_1_idx], [source_idx, neighbor_2_idx], ...]

        # node and edge features computed from immage, edge list (source node can be inferred from this), and pre-computed features
        node_features = compute_node_features(img, edges, node_external_features)  # dictionary of features for the node
        ebunch = compute_edge_features(img, edges,
                                       node_external_features)  # list of tuples of the format (source_idx, target_idx, {**features})

        G.add_node(idx, **node_features)
        G.add_edges_from(ebunch)

    return G


def agg_stats(grain_sizes, grain_labels, timesteps):
    """



    References:
    DeCost, Holm, Phenomenology of Abnormal Grain Growth in Systems with Nonuniform Grain Boundary Mobility,
    MMTA, 2017
    """
    total_area = grain_sizes[0].sum()  # total number of pixels in simulation box

    candidate_idx = np.argmax(grain_labels == 1)

    initial_average_size = grain_sizes[0].mean()
    initial_candidate_size = grain_sizes[0, candidate_idx]
    # todo factor out destroyed grains
    final_candidate_size = grain_sizes[-1, candidate_idx]

    initial_grains = np.unique(grain_labels, return_counts=True)

    mask = grain_sizes[-1] > 0  # select grains that were not consumed during simulation
    final_grains = np.unique(grain_labels[mask], return_counts=True)
    final_average_size = grain_sizes[-1, mask].mean()

    growth_ratio_all = grain_sizes[-1] / grain_sizes[0]  # A(final)/A(initial), includes all (including consumed) grains
    growth_ratio = growth_ratio_all[mask]  # only include grains that survived the simulation
    growth_ratio_avg = growth_ratio.mean()
    growth_ratio_avg_bulk = _masked_mean(growth_ratio_all[np.logical_and(mask, np.arange(
        len(growth_ratio_all)) != candidate_idx)])  # growth ratio ignoring candidate grain
    candidate_area_fraction = grain_sizes[-1, candidate_idx] / total_area

    agg_bool = final_candidate_size / final_average_size > 3

    initial_u, initial_c = np.unique(grain_labels, return_counts=True)
    initial_stats = {}
    final_stats = {}
    for t in range(1, 4):
        if t not in list(initial_u):  # sometimes one grain type does not appear. force all types (candidate,
                                      # high, low mobility) to appear in u
            initial_u = np.concatenate([initial_u, [t]], axis=0)
            initial_c = np.concatenate([initial_c, [0]], axis=0)
    for u, c in zip(initial_u, initial_c):  # separate grains by type (red/blue/white)
        init_mask = grain_labels == u
        initial_stats['type_{}'.format(u)] = {
            'count': c,
            'average_size': _masked_mean(grain_sizes[0, init_mask]),
            'area_fraction': grain_sizes[0, init_mask].sum() / total_area}

        final_mask = np.logical_and(init_mask,
                                    mask)  # select grains of correct type and still exist at end of simulation (ie haven't been consumed)

        final_stats['type_{}'.format(u)] = {
            'count': final_mask.sum(),
            'average_size': _masked_mean(grain_sizes[-1, final_mask]),
            'avg_growth_ratio': _masked_mean(growth_ratio_all[final_mask]),
            'area_fraction': grain_sizes[-1, final_mask].sum() / total_area
        }

    results = {'initial_average_size': initial_average_size,
               'initial_candidate_size': initial_candidate_size,
               'final_average_size': final_average_size,
               'final_candidate_size': final_candidate_size,
               'final_candidate_area_fraction': candidate_area_fraction,
               'candidate_growth_ratio': growth_ratio_all[candidate_idx],
               'growth_ratio_avg': growth_ratio_avg,
               'growth_ratio_avg_bulk': growth_ratio_avg_bulk,
               'agg_bool': agg_bool,
               'initial_stats': initial_stats,
               'final_stats': final_stats,
               'grain_ids': {'type_1': 'candidate grain (white)',
                             'type_2': 'low mobility (blue)',
                             'type_3': 'high mobility (red)'},
               'grain_sizes': grain_sizes,
               'timesteps': timesteps,
               'grain_labels': grain_labels}

    return results


## TODO add node and edge features
def create_graph(root):
    """
    Read the SPPARKS meso input/output files and build a graph of the data.

    Parameters
    -----------
    root: string or Path object
        directory containing 2 files:
            1) initial.dream3d contains pixel map of initial grains (n x m array, each pixel is an
               integer indicating which grain it belongs to)
            2) stats.h5: contains arrays of timesteps and grain area (in pixels) of each grain at the time step

    Returns
    -----------
    ?
    """
    root = Path(root)  # forces root to be path object
    init = h5py.File(root / 'initial.dream3d', 'r')
    stats = h5py.File(root / 'stats.h5', 'r')

    sv = init['DataContainers']['SyntheticVolume']
    # grain_ids contains n x m array of pixels. Each pixel has integer value indicating the grain id that it belongs to
    # shifted by 1 so pixel ids start at 0
    grain_ids = np.asarray(sv['CellData']['FeatureIds']) - 1

    # grain_labels contains the class of each grain.
    # The row index corresponds to the grain in grain_ids (also shifted by 1 for consistency- the first row is zeros
    # to account for how the first grain id is 1- ie there is no 0 index, so this is removed)
    # In the original file there are 3 different vectors describing grain mobility:
    # [0.577, 0.577, 0.577] indicates high-mobility  (red in animation)
    # [0.894, 0.447, 0] indicates low-mobility grain (blue in animation)
    # [1, 0, 0] indicates candidate grain (white in animation)
    # Rather than keeping the entire vector, we can get the label by simply counting the number of non-zero elements
    # 3 indicates high mobility, 2 indicates low-mobility, 1 indicates candidate grain
    grain_labels = np.asarray(sv['CellFeatureData']['AvgQuats'])[1:]
    grain_labels = (grain_labels > 0).sum(1)

    timesteps = np.asarray(stats['time'])
    # again, the first column is all 0's due to original grain indexing starting at 1
    # shift by one element to account for zero-indexing
    grain_sizes = np.asarray(stats['grainsize'])[:, 1:]

    return agg_stats(grain_sizes, grain_labels, timesteps)

    G = img_to_graph(grain_ids)
    return G


def _masked_mean(x):
    """Returns mean if list is non-empty, otherwise return None.

    Parameters
    ----------
    x: ndarray
        array of valuse (usually result of masking a larger array to select specific values)

    Returns
    -------
    mean: float or None
        mean of x if x is not empty, or None if x is empty
    """
    return np.mean(x) if len(x) else None

