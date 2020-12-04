import h5py
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from skimage.morphology import binary_dilation
import graphviz  # https://pypi.org/project/graphviz/
from pathlib import Path

from . import graph_features as gf


def img_to_graph(img, grain_labels, grain_sizes, timesteps, compute_node_features=None, compute_edge_features=None,
                 compute_graph_features=None):
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

    grain_labels: ndarray
        n_grain element array where values at index i indicate the class of grain with label i.
          Values should be 0, 1, or 2 and correspond to the grain's class (candidate, high, low mobility, respectively)

    compute_node_features, compute_edge_features, compute_graph_features: callable or None
        if None- only the graph topology/connectivity is determined
        else- functions that take the image, neighbor list, and external features as inputs  # TODO clarify format

    external_features: dictionary
        contains ## TODO make format consistent for graph features
        if None- external features are not incorporated in computing node and edge features
        if array- N_node x f array of features for each node
        if dictionary- keys are node_idx or tuple(source_node, target_node), value is n-element array of features


    Returns
    -------------
    graph- dict
        results with structure {'attributes': <dict containing graph level features,
                                'graph': <networkx DiGraph containing graph structure and node/edge attributes>}


    Examples
    -------------

    References
    -------------
    Z. Liu, J. Zhou Introduction to Graph Neural Networks
      (https://www.morganclaypoolpublishers.com/catalog_Orig/samples/9781681737669_sample.pdf)

    """

    if compute_node_features is None:
        compute_node_features = gf.compute_node_features

    if compute_edge_features is None:
        compute_edge_features = gf.compute_edge_features

    if compute_graph_features is None:
        compute_graph_features = gf.compute_graph_features


    graph_features = compute_graph_features(img, grain_labels, grain_sizes, timesteps)

    G = Graph()  # simple, undirected graph

    # TODO this part is slow, look into parallelizing?
    #     Compute node/edge features in parallel and then add everything
    #     to Graph object afer loop? How to avoid double computation of
    #     edge properties?
    #       Alternatively: many graphs will have to be computed, keep
    #      this function as single-thread and run in parallel for multiple graphs
    for idx in range(len(grain_labels)):
        img_roll = _roll_img(img, idx)
        grain_mask = img_roll == idx  # selects single grain

        # to find neighbors of grain, apply binary dilation tho the mask and look for overlap
        # TODO (select smaller window around rolled image, use where to get coords, look for neighbors directly?)

        grain_mask_dilate = binary_dilation(grain_mask, selem=np.ones((3, 3), np.int))

        # TODO find threshold for min number of shared pixels to be considered a neighbor?
        #      would have to be scaled by perimiter or something
        neighbors = np.unique(img_roll[grain_mask_dilate])
        neighbors = neighbors[neighbors != idx]  # a grain cannot be its own neighbor (ie no self loops on graph)

        source_idx = np.zeros(neighbors.shape, np.int) + idx
        edges = np.stack((neighbors, source_idx),
                         axis=0).T  # format: [[source_idx, neighbor_1_idx], [source_idx, neighbor_2_idx], ...]

        # node and edge features computed from immage, edge list (source node can be inferred from this), and pre-computed features
        node_features = compute_node_features(grain_mask, grain_labels[idx], len(neighbors))  # dictionary of features for the node

        G.add_node(idx, **node_features)

        # list of tuples of the format (source_idx, target_idx, {**features})
        ebunch = [(*e, compute_edge_features(img, *e)) for e in edges]


        G.add_edges_from(ebunch)

    G.graph_attr = graph_features

    return G

# TODO Refine this function once you figure out exactly which quantities are needed.
# TODO groups- candidate, bulk, red, blue
# TODO mask- grow
# TODO for each group, and each subset (grow/nogrow):
#   number, avg size, avg grow ratio, subset_mask (so individual stats can be captured)

# TODO offset grain_ids by 1 to account for 0 indexing
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

    not_candidate_mask = grain_labels != 1
    growth_ratio_all = grain_sizes[-1] / grain_sizes[0]  # A(final)/A(initial), includes all (including consumed) grains
    growth_mask = growth_ratio_all > 1.0  # do not consider grains that shrunk or were consumed during simulation
    growth_ratio = growth_ratio_all[growth_mask]  # only include grains that survived the simulation
    growth_ratio_avg = growth_ratio.mean()

    # growth ratio ignoring candidate grain
    growth_ratio_avg_bulk = _masked_mean(growth_ratio_all[np.logical_and(growth_mask, not_candidate_mask)])

    candidate_area_fraction = grain_sizes[-1, candidate_idx] / total_area
    growth_final_average_size = grain_sizes[-1, growth_mask].mean()
    growth_final_average_size_bulk = grain_sizes[-1, np.logical_and(growth_mask, not_candidate_mask)]

    agg_bool = final_candidate_size / final_average_size > 3

    initial_u, initial_c = np.unique(grain_labels, return_counts=True)
    initial_stats = {}
    final_stats = {}
    for t in range(3):
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

        final_mask = np.logical_and(init_mask, mask)  # select grains of correct type and still exist at
                                                      # end of simulation (ie haven't been consumed)
        final_growth_mask = np.logical_and(init_mask, growth_mask) # select only grains of correct type that
                                                                   # grew during simulation

        final_stats['type_{}'.format(u)] = {
            'count': final_mask.sum(),
            'average_size': _masked_mean(grain_sizes[-1, final_mask]),
            'avg_growth_ratio': _masked_mean(growth_ratio_all[final_mask]),
            'area_fraction': grain_sizes[-1, final_mask].sum() / total_area,
            # same statistics as above but only include grains that grew during simulation
            'grow_count': final_growth_mask.sum(),
            'grow_average_size': _masked_mean(grain_sizes[-1, final_growth_mask]),
            'grow_avg_growth_ratio': _masked_mean(growth_ratio_all[final_growth_mask]),
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
               'grain_ids': {'type_0': 'candidate grain (white)',
                             'type_1': 'low mobility (blue)',
                             'type_2': 'high mobility (red)'},
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
    initfile = root / 'initial.dream3d'
    statsfile = root / 'stats.h5'

    assert initfile.is_file(), f'{str(initfile.absolute())} not found!'
    assert statsfile.is_file(), f'{str(statsfile.absolute())} not found!'

    init = h5py.File(initfile, 'r')
    stats = h5py.File(statsfile, 'r')



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
    #  and subtract 1 for zero-indexing
    # 2 indicates high mobility, 1 indicates low-mobility, 0 indicates candidate grain
    grain_labels = np.asarray(sv['CellFeatureData']['AvgQuats'])[1:]
    grain_labels = (grain_labels > 0).sum(1) - 1

    timesteps = np.asarray(stats['time'])
    # again, the first column is all 0's due to original grain indexing starting at 1
    # shift by one element to account for zero-indexing
    grain_sizes = np.asarray(stats['grainsize'])[:, 1:]

    G = img_to_graph(grain_ids, grain_labels, grain_sizes, timesteps)
    G.root = root.absolute()
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


def _roll_img(img, i):
    """
    Roll image *img* such that grain *i* is approximately centered.

    This is needed to account for periodic boundary conditions, which may cause an individual grain
    or grain's neighbor to be wrapped around an edge of the image. Wrapping to the center ensures that small grain
    neighborhoods will be centered and not cross the edges. Note that this only applies to small grain neighborhoods
    (ie the initial grain structure.) For grains that dominate the size of the image (ie after abnormal growth occurs,)
    then a more sophisticated approach (ie using np.where() to get coords and then operating on coords directly) will
    be needed, but this is likely not needed for determining the graph structure (ie initial grains are small,) and is
    not compatible with functions that operate on binary masks (ie skimage regionprops.)

    Parameters
    ----------
    img: ndarray
        r x c integer numpy array where each element is the grain ID corresponding to the pixel

    i: int
        index of grain in *img* to center

    Returns
    -------
    img_roll: ndarray
        same format as *img* with coordinates rolled to center grain *i*
    """

    # center indices of image
    r, c = [x//2 for x in img.shape]

    # coordinates of grain of interest
    rows, cols = np.where(img == i)

    # for both row and column indices:

    # if mask is only 1 pixel, roll it to center index directly

    # if mask has more than 1 pixels, look for discontinuities (difference in coordinates between consecutive
    # pixels in mask is larger than some threshold ie 10px)

    # if a discontinuity is found: roll image by half of its coords (will approximately land in center)
    # (later this could be improved by finding weighted average location of pixels)
    # if one is not found, roll to center by the difference in center and mean coordinate of mask


    if len(rows) == 1:  # mask is only 1 pixel, can't take difference in coords to detect split
        row_shift = r - rows[0]
    elif (rows[1:] - rows[:-1]).max() > 10:
        row_shift = r
    else:
        row_shift = int(r - rows.mean())

    if len(cols) == 1:
        col_shift = c - cols[0]
    elif (cols[1:] - cols[:-1]).max() > 10:
        col_shift = c
    else:
        col_shift = int(c - cols.mean())

    img_roll = np.roll(img, (row_shift, col_shift), axis=(0, 1))

    return img_roll

class Graph(nx.DiGraph):
    """
    Wraps networkx.DiGraph to add support for graph level features and export to pytorch geometric (pyg)
    and dgl datasets.
    """

    def __init__(self):
        super().__init__()
        self._graph_attributes = None
        self._root = None

    @property
    def graph_attr(self):
        return self._graph_attributes

    @graph_attr.setter
    def graph_attr(self, ga):
        assert type(ga) == dict
        self._graph_attributes = ga

    @property
    def root(self):
        return self._root

    @root.setter
    def root(self, r):
        assert type(r) is str or isinstance(r, Path)
        self._root = r

    def to_pyg_dataset(self):
        # TODO implement this
        pass

    def to_dgl_dataset(self):
        # TODO implement this
        pass

    def to_numpydict(self):
        # TODO implement this
        """
        returns edge list, node features, edge features, graph features
        """
        pass

    def to_json(self):
        # TODO implement this
        pass

    def from_json(self):
        # TODO implement this
        pass

    # def __repr__(self):
    #     # TODO implement this
    #     pass

    def copy(self):
        # TODO implement this
        pass
