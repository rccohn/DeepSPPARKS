import h5py
import json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from skimage.morphology import binary_dilation
import graphviz  # https://pypi.org/project/graphviz/
from pathlib import Path, PurePath
from src.image import _roll_img
import src.graph_features as gf


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


# TODO add node and edge features
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
    G: Graph object
        graph where nodes are grains, edges are boundaries between grains.
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


# TODO add metadata separate from features?
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
        """
        Export graph to torch_geometric dataset.

        Note that only features are exported. Labels are subjective
        are therefore determined externally (ie data.y must be set later.)

        Returns
        -------

        """
        import torch
        from torch_geometric.data import Data

        nd = self.to_numpydict()

        # torch_geometric uses 2 x N, not N x 2 edgelist
        edges = torch.tensor(nd['edge_list'].T, dtype=torch.long)

        # copy node and edge features from numpydict as-is
        node_features = torch.tensor(nd['node_features'], dtype=torch.double)
        edge_attr = torch.tensor(nd['edge_features'], dtype=torch.double)

        # for now, mask selects candidate grain only
        mask = torch.BoolTensor([self.nodes[n]['mobility_label'][0] == 1. for n in self.nodes])

        d = Data(x=node_features, edge_index=edges, edge_attr=edge_attr, mask=mask)

        return d

    def to_dgl_dataset(self):
        # TODO implement this
        pass

    def to_numpydict(self):
        # TODO implement this
        """
        returns edge list, node features, edge features, graph features
        """
        nd = {}

        nd['edge_list'] = np.asarray(self.edges)

        #  id of first node and edges
        n0 = next((x for x in self.nodes))
        e0 = next((x for x in self.edges))

        # names of features for nodes and edges
        # note that single labels may correspond to multiple values (ie unit vector has 2 components for edges)
        node_feat_ids = sorted(self.nodes[n0].keys())
        edge_feat_ids = sorted(self.edges[e0].keys())

        # node features
        node_feat = []
        for n in self.nodes:
            node_feat.append(np.concatenate([self.nodes[n][f] for f in node_feat_ids]))
        node_feat = np.asarray(node_feat)

        edge_feat = []
        for e in self.edges:
            edge_feat.append(np.concatenate([self.edges[e][f] for f in edge_feat_ids]))
        edge_feat = np.asarray(edge_feat)

        nd['node_features'] = node_feat
        nd['edge_features'] = edge_feat

        nd['node_feature_ids'] = node_feat_ids
        nd['edge_feature_ids'] = edge_feat_ids
        # TODO add graph features, normalization, split into continuous vs categorical?

        return nd

    def to_json(self, path):
        """
        Saves graph data to json file. Can be loaded with graph.from_json(path).

        Parameters
        ----------
        path: str or Path object
            Path to save data to
        """

        nd = {}

        # add nodes
        nd['nodes'] = {n: self.nodes[n] for n in self.nodes}

        # add edges
        nd['edges'] = {e: self.edges[e] for e in self.edges}

        # add graph level features
        nd['graph_attr'] = self.graph_attr

        # add path
        nd['path'] = self.root

        jd = DictTransformer.to_jsondict(nd)

        with open(path, 'w') as f:
            json.dump(jd, f,)

    @staticmethod
    def from_json(path):
        r"""
        Loads data from json file into Graph object.

        Parameters
        ----------
        path: str or Path object
            path to graph data stored as json (from Graph().to_json())

        Returns
        -------
        g: Graph object
            Graph with data loaded from path

        """
        g = Graph()
        with open(path, 'r') as f:
            jd = json.load(f)

        nd = DictTransformer.from_jsondict(jd)

        # add nodes and node features
        g.add_nodes_from(((node, attr) for node, attr in nd['nodes'].items()))

        # add edges and edge features
        g.add_edges_from(((*edge, attr) for edge, attr in nd['edges'].items()))

        # add graph attributes
        g.graph_attr = nd['graph_attr']

        # file root
        g.root = nd['path']

        return g

    @staticmethod
    def from_spparks_out(path):
        r"""
        Create graph from raw outputs from spparks.

        Parameters
        ----------
        path: str or Path object
            path to directory containing  initial.dream3d and stats.h5

        Returns
        -------
        g: Graph object
            Extracted graph from data
        """
        # TODO custom features
        return create_graph(path)

    # def __repr__(self):
    #     # TODO implement this
    #     pass

    def copy(self):
        # TODO implement this
        pass


class DictTransformer:

    @staticmethod
    def to_jsondict(nd):
        """
        Converts a dictionary containing numpy arrays to a format that can be saved with json.dump()

        Parameters
        ----------
        nd: dict
            dictionary that may contain numpy arrays.

        Returns
        -------
        jd: dict
            json-formatted dict that can be saved with json.dump().
        """
        jd = {}
        for k, v in nd.items():
            # keys for json must be one of following format
            if type(k) not in (str, int, float, bool, None):
                k = str(k)


            t = type(v)
            if t == dict:  # if v is dict, recursively apply method to cover all k-v pairs in sub-dictionaries
                jd[k] = DictTransformer.to_jsondict(v)
            elif t == np.ndarray:  # if v is array, convert to list so it's compatible with json
                jd[k] = v.tolist()
            elif isinstance(v, PurePath):  # path object, note we check v and not t
                jd[k] = str(v)
            else:  # otherwise, store v as-is
                jd[k] = v

        return jd

    @staticmethod
    def from_jsondict(jd):
        """
        Convert json-compatible dictionary into dictionary containing numpy arrays.

        Applies the following changes:
            Where applicable, converts keys (always str for json) to numeric, and
            Converts numeric lists to arrays (json always stores list-like objects as lists)

        Parameters
        ----------
        jd: dict
            json-formatted dictionary

        Returns
        -------
        nd: dict
            numpy-formatted dictionary with keys that may be numeric and values that may be ndarrays.
        """
        nd = {}
        for k, v in jd.items():
            k = _str_to_numeric(k)
            t = type(v)
            if t == dict:
                # if v is a dict, recursively apply method to cover all k-v pairs in sub-dictionaries
                nd[k] = DictTransformer.from_jsondict(v)
            elif t == list:  # list or array
                arr = np.asarray(v)
                if np.issubdtype(arr.dtype, np.number):  # array is numeric
                    nd[k] = arr
                else:  # array is not numeric, keep item as a list
                    nd[k] = v
            elif t == str and '/' in v:  # object appears to be a path
                nd[k] = Path(v)
            else:  # other type, leave v as-is
                nd[k] = v

        return nd


def _str_to_numeric(s):
    """
    Convert string s to numeric value, if applicable.

    Necessary for reading json dicts (keys always strings) back to consistent format for graph
    (ie keys for node id's are ints)

    Parameters
    ----------
    s: str
        string

    Returns
    -------
    int, float, tuple(int, int), or str
        converted value.

    """
    try:  # case 1: s is integer (ie '3')
        return int(s)
    except ValueError:
        try:  # case 2 s is float (ie '3.5')
            return float(s)
        except ValueError:
            try:  # case 3 s is a tuple of integers- like from an edge list (ie '(3, 5)')
                return tuple(int(x) for x in s.strip('()').split(', '))
            except:  # case 4 s is a path
                return s

if __name__ == "__main__":
    p2 = Path('/media/ryan/TOSHIBA EXT/Research/datasets/AFRL_AGG/SPPARKS_simulations/candidate-grains/2021_01_04_01_06_candidate_grains_master/spparks_results/run_11')
    assert p2.exists()
    g3 = Graph.from_spparks_out(p2)
    g3.to_json('test_debug.json')