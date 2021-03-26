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
import pycocotools.mask as RLE
import matplotlib.pyplot as plt

def img_to_graph(img, grain_labels):
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

    NOTE: THESE ARE NOT CURRENTLY USED. TODO UPDATE DOCSTRING. Node/edge/graph features can be updated after graph
                                                                is created.
    compute_node_features, compute_edge_features, compute_graph_features: callable or None
        if None- only the graph topology/connectivity is determined
        else- functions that take the image, neighbor list, and external features as inputs  # TODO clarify format

    NOTE: THESE ARE NOT CURRENTLY USED. TODO UPDATE DOCSTRING.
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

        # node and edge features computed from image, edge list (source node can be inferred from this)
        node_features = gf.compute_node_features(grain_mask, grain_labels[idx])  # dictionary of features for the node

        G.add_node(idx, **node_features)

        # list of tuples of the format (source_idx, target_idx, {**features})
        ebunch = [(*e, gf.compute_edge_features(img_roll, *e)) for e in edges]

        G.add_edges_from(ebunch)

    return G


# TODO add node and edge features
# TODO separate graph metadata from graph level features?
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

    # metadata
    metadata = {}

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
    grain_labels_raw = np.asarray(sv['CellFeatureData']['AvgQuats'])[1:]
    # preserve original mobility labels

    # reduced representation of mobility- store as single value instead of vector of 3 elements
    grain_labels = (grain_labels_raw > 0).sum(1) - 1

    timesteps = np.asarray(stats['time'])
    # again, the first column is all 0's due to original grain indexing starting at 1
    # shift by one element to account for zero-indexing
    grain_sizes = np.asarray(stats['grainsize'])[:, 1:]

    #TODO fix inconsistencies in naming ie grain labels vs grain ids, especially for compute_metadata()
    G = img_to_graph(grain_ids, grain_labels)
    G.root = root.absolute()
    G.metadata = gf.compute_metadata(grain_ids, grain_labels_raw, grain_sizes, timesteps)
    G.targets = gf.compute_targets(grain_labels, grain_sizes)
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
        self._graph_attributes = None  # graph level attributes
        self._root = None  # path to file
        self._metadata = None  # info about the original graph itself
        self._targets = None  # labels, regression outputs, etc

    @property
    def graph_attr(self):
        return self._graph_attributes

    @graph_attr.setter
    def graph_attr(self, ga):
        assert type(ga) in (dict, type(None)), 'graph_attr must be dict or None!'
        self._graph_attributes = ga

    @property
    def metadata(self):
        return self._metadata

    @metadata.setter
    def metadata(self, md):
        assert type(md) == dict
        self._metadata = md

    @property
    def targets(self):
        return self._targets

    @targets.setter
    def targets(self, y):
        assert type(y) == dict
        self._targets = y

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

        # add metadata
        nd['metadata'] = self.metadata

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

        for n in g.nodes:
            g.nodes[n]['rle']['counts'] = bytes(g.nodes[n]['rle']['counts'], 'utf-8')

        metadata = {}
        nd = nd['metadata']
        metadata['img_size'] = tuple(nd['img_size'])
        metadata['timesteps'] = np.array(nd['timesteps'])
        metadata['mobility_vectors'] = nd['mobility_vectors']
        metadata['grain_sizes'] = np.array(nd['grain_sizes'])
        metadata['center_id'] = nd['center_id']  # id of grain located at center of image
        metadata['center_bounds_rrcc'] = np.array(nd['center_bounds_rrcc'])
        g.metadata = metadata




        return g

    def to_image(self):
        """
        turn graph back into image from spparks output
        """
        size = self.metadata['img_size']
        img = np.zeros(size, np.int) - 1  # -1 to differentiate between grain 0 and unfilled pixels

        # place center grain on image first
        cid = self.metadata['center_id']
        rle = self.nodes[cid]['rle']
        center = RLE.decode(rle)

        where = np.stack(np.where(center), axis=1)
        wmin = np.min(where, axis=0)

        shift1 = -wmin  # to top left corner of grain bbox in bitmask
        shift2 = G.metadata['center_bounds_rrcc'][::2]  # center of img

        shift = shift1 + shift2

        newcoords = (where + shift).T

        img[newcoords[0], newcoords[1]] = cid
        visited = {x: False for x in self.nodes}
        visited[cid] = True
        img = _add_neighbors_to_img(self, img, cid, visited)

        img = _roll_img(img, cid)  # not sure why this is needed but rolling the
                                   # image to its final position doesn't work without it

        where = np.stack(np.where(img == cid), axis=1).min(axis=0)
        shift = shift2 - where
        img = np.roll(img, shift, axis=(0, 1))
        assert not np.any(img == -1), 'error in reconstructing image, some pixels not filled'
        return img

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

    def to_spparks_out(self):
        # TODO finish this to verify computations perform as expected
        # compare graid_ids, mobilities, timesteps, grain_sizes, etc to
        # original hd5 file
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

            elif t in (tuple, list, set):
                jd[k] = DictTransformer.to_jsonlist(v)
            elif t == bytes:
                jd[k] = v.decode('utf-8')
            elif isinstance(v, PurePath):  # path object, note we check v and not t
                jd[k] = str(v)
            else:  # otherwise, store v as-is
                jd[k] = v

        return jd

    @staticmethod
    def to_jsonlist(x):
        """
        Convert list to a json-compatible list. Iterates through items
        in the list, and applies the following changes:
            converts tuples, ndarrays, sets to list
            converts dicts to jsondict (see to_jsondict)

        Parameters
        ----------
        x: list-like
            list of values to iterae over

        Returns
        --------
        x_json: list
            json-compatible list

        """
        x_json = []
        for item in x:
            t = type(item)
            if t == dict:
                x_json.append(DictTransformer.to_jsondict(item))

            elif t == np.ndarray:
                x_json.append(item.tolist())

            elif t in (tuple, list, set):
                x_json.append(DictTransformer.to_jsonlist(item))

            else:
                x_json.append(item)

        return x_json

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
            if k == 'metadata':
                nd[k] = v
            else:
                if t == dict:
                    # if v is a dict, recursively apply method to cover all k-v pairs in sub-dictionaries
                    nd[k] = DictTransformer.from_jsondict(v)
                elif t == list:  # list or array
                    if type(v[0]) == list:
                        if all([len(x) == len(v[0]) for x in v]):
                            nd[k] = np.array(v)
                        else:
                            if all([type(x) in (int, float, bool) for x in v]):
                                nd[k] = np.array(v)
                            else:
                                nd[k] = DictTransformer.to_jsonlist(v)
                    else:
                        nd[k] = DictTransformer.to_jsonlist(v)

                    arr = np.array(v)
                    if np.issubdtype(arr.dtype, np.number):  # array is numeric
                        nd[k] = arr
                    else:  # array is not numeric, keep item as a list
                        nd[k] = v
                elif t == str and '/' in v:  # object appears to be a path
                    nd[k] = Path(v)
                else:  # other type, leave v as-is
                    nd[k] = v

        return nd

    @staticmethod
    def from_jsonlist(x):
        pass

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


def _add_node_to_img(g, img, src, target):
    """
    img: image to put grains on
    g: graph
    src, target: nodes in graph
        src should already be on graph, target should be a neighbor of src
        (ie the edge (src, target) should exist)
    """
    # note- need to handle wrapped grains later
    xcenter = np.mean(np.where(img == src), axis=1)  # centroid of source grain
    edge = g.edges[(src, target)]

    bitmask = RLE.decode(g.nodes[target]['rle'])
    shift1 = -np.mean(np.where(bitmask), axis=1)  # top left of bitmask to center of grain
    shift2 = xcenter - (edge['unit_vector'] * edge['length'])
    coords = np.round((np.stack(np.where(bitmask), axis=1) + shift1 + shift2)).astype(np.int).T
    #coords = coords % np.reshape(img.shape, (2, 1))

    img[coords[0], coords[1]] = target

    return img

# TODO this is a breadth-first search problem
def _add_neighbors_to_img(g, img, src, visited):
    neighborhoods = [(src, [n for n in g.neighbors(src)])]
    for n in neighborhoods[0][1]:  # all of these nodes will be visited
        visited[n] = True

    i = 0


    while neighborhoods:  # continue while there are still neighbors
        edgelist = neighborhoods.pop(0)  # (src, [n for n in g.neighbors(src)] if not visited[n])
        src = edgelist[0]
        img = _roll_img(img, src)  # avoid issues from periodic boundary conditions by always working at center of image
        while edgelist[1]:
            i += 1

            n = edgelist[1].pop(0)  # get target node

            img = _add_node_to_img(g, img, src, n,)  # add target node to graph

            new_edgelist = (n, [x for x in g.neighbors(n) if not visited.get(x)])

            for node in new_edgelist[1]:
                visited[node] = True # all of these nodes will be visited

            if new_edgelist[1]:
                neighborhoods.append(new_edgelist)


    return img


if __name__ == "__main__":
    p2 = Path('/media/ryan/TOSHIBA EXT/Research/datasets/AFRL_AGG/'
     'SPPARKS_simulations/candidate-grains/2020_12_26_21_38_candidate_grains_master/spparks_results/run_472')
    assert p2.exists()
    json_path = Path('..', 'data', 'temp', 'test_graph_2.json')
    if not json_path.is_file():
        G = Graph.from_spparks_out(p2)

        G.to_json(json_path)
    else:
        G = Graph.from_json(json_path)
    img = G.to_image()

#    runs = list(p2.glob('*run*'))[:10]
#    from multiprocessing import Pool
#    def#
#    with Pool(4) as p:
#        graphs = p.map_async(Graph.from_spparks_out(p2))
#    graphs

    plt.imshow(img, cmap='jet')
    plt.colorbar()
    plt.show()
