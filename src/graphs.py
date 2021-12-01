import h5py
import json
import numpy as np
import networkx as nx
#import matplotlib.pyplot as plt
from skimage.morphology import binary_dilation
#import graphviz  # https://pypi.org/project/graphviz/
from pathlib import Path, PurePath
from src.image import roll_img
from src import graph_features as gf
from pycocotools import mask as RLE
from copy import deepcopy


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

    grain_rles = [RLE.encode(np.asfortranarray(img == i))
                  for i in range(img.max() + 1)]

    G = Graph()  # simple, undirected graph

    # TODO this part is slow, look into parallelizing?
    #     Compute node/edge features in parallel and then add everything
    #     to Graph object afer loop? How to avoid double computation of
    #     edge properties?
    #       Alternatively: many graphs will have to be computed, keep
    #      this function as single-thread and run in parallel for multiple graphs
    for idx in range(len(grain_labels)):
        img_roll = roll_img(img, idx)
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
        # TODO temporary fix for testing, move this to graph_features.py if it works
        node_features['rle'] = grain_rles[idx]

        G.add_node(idx, **node_features)

        # list of tuples of the format (source_idx, target_idx, {**features})
        # TODO update method with final version (with or without edge features)
        ebunch = [(*e, ) for e in edges]
        #ebunch = [(*e, gf.compute_edge_features(img_roll, *e)) for e in edges]

        G.add_edges_from(ebunch)

    return G


def create_graph(path):
    """
    Read the SPPARKS meso input/output files and build a graph of the data.

    Parameters
    -----------
    path: string or Path object
        directory containing 2 files:
            1) initial.dream3d contains pixel map of initial grains (n x m array, each pixel is an
               integer indicating which grain it belongs to)
            2) stats.h5: contains arrays of timesteps and grain area (in pixels) of each grain at the time step

    Returns
    -----------
    G: Graph object
        graph where nodes are grains, edges are boundaries between grains.
    """
    path = Path(path)  # forces root to be path object
    initfile = path / 'initial.dream3d'
    if not initfile.is_file():  # for repeat graphs
        initfile = path / 'spparks_init/initial.dream3d'

    statsfile = path / 'stats.h5'
    assert initfile.is_file(), f'{str(initfile.absolute())} not found!'

    if statsfile.is_file():
        gtype = 'single'  # graph corresponds to single spparks simulation
    else:
        statsfile = path / 'spparks_results'
        stats_files = [x / 'stats.h5' for x in sorted(statsfile.glob('*')) if (x/'stats.h5').is_file()]
        assert len(stats_files), "stats.h5 file not found!"
        gtype = 'repeat'  # graph corresponds to multiple repeated spparks
                    # simulations with same initial state, and repeated
                    # growth simulations with different seeds

    # for both single and repeat graphs, the initial state is the same

    init = h5py.File(initfile, 'r')



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
    # TODO double check low, high, candidate mobilities against image
    #  Consider making candidate 2, high mobility 1, low mobility 0
    #  to make overall mobility more linear
    # reduced representation of mobility- store as single value instead of vector of 3 elements
    grain_labels = (grain_labels_raw > 0).sum(1) - 1

    # TODO fix inconsistencies in naming ie grain labels vs grain ids, especially for compute_metadata()
    G = img_to_graph(grain_ids, grain_labels)

    # processing of stats.h5 is different for single and repeat graphs
    # for single graphs, simply get trajectory of each grain and store it as metadata
    if gtype == "single":
        stats = h5py.File(statsfile, 'r')
        timesteps = np.asarray(stats['time'])
        # again, the first column is all 0's due to original grain indexing starting at 1
        # shift by one element to account for zero-indexing
        grain_sizes = np.asarray(stats['grainsize'])[:, 1:]

        G.metadata = gf.compute_metadata(grain_ids, grain_labels_raw, grain_sizes, timesteps, path.absolute().resolve())
        G.metadata['gtype'] = gtype
    # repeat graph
    else:
        metadata = {'path': [], 'timesteps': [], 'grain_sizes': [], 'gtype': 'repeat'}
        for f in stats_files:
            stats = h5py.File(f, 'r')
            timesteps = np.asarray(stats['time'])
            grain_sizes = np.asarray(stats['grainsize'])[:,1:]
            single_metadata = gf.compute_metadata(grain_ids, grain_labels_raw,
                                                  grain_sizes, timesteps, f.absolute().resolve())
            # append the filename, timesteps,
            # todo timesteps probably the same for every trial, double check this
            #      >> timesteps are not exactly equal, but very, very close
            # doesn't require arrays to be same size (ie if one simulation exited early for some reason)
            # if this causes slowdown, appending can be replaced by filling fixed size numpy array later
            metadata['path'].append(f)
            metadata['timesteps'].append(timesteps)
            metadata['grain_sizes'].append(grain_sizes)

        # other metadata (img size, grain_types, center_id, etc) should be the same for
        # all trials.
        for k,v in single_metadata.items():
            if k not in ('path', 'timesteps', 'grain_sizes'):
                metadata[k] = v
        G.metadata = metadata

    return G


# TODO warn user when using g.nodes that g.nodes is not sorted and they probably want g.nodelist instead
class Graph(nx.DiGraph):
    """
    Wraps networkx.DiGraph to add support for metadata and some convenient
    built-in functions.
    """
    # TODO to further reduce graph size, make node and edge features single list so you don't have to store key
    #     value pairs for every node and edge
    def __init__(self):
        super().__init__()
        self._metadata = {}  # info about the original graph itself

    @property
    def metadata(self):
        return self._metadata

    @metadata.setter
    def metadata(self, md):
        assert type(md) == dict, 'metada must be a dict object!'
        self._metadata = md

    # TODO update code to use edgelist where applicable (instead of explicitly calling
    #      [self.edges[e] for e in sorted(self.edges)]
    @property
    def edgelist(self):
        """
        Sorted list of edge indices.
]
        Returns
        -------
        edgelist: list(tuple)
            sorted list of edges. Each edge is represented as (source, target) where
            source and target are items in self.nodes.
        """
        return [self.edges[e] for e in self.eli]

    # TODO update code to use nodelist where applicable (instead of explicitly calling
    #      [self.nodes[n] for n in sorted(self.nodes)]
    @property
    def nodelist(self):
        """
        Sorted list of nodes.

        Returns
        -------
        nodelist: list
            sorted list of node indices [idx1, idx2, ...]. Node objects can be accessed through
            self.nodes[idx].
        """
        return [self.nodes[n] for n in self.nli]

    @property
    def nli(self):
        """
        sorted list of node indices
        Returns
        -------
        node_list_indices: list
            sorted list of nodi indices [idx1, idx2, ...]. Individual node objects can be accessed through
            self.nodes[idx].
        """
        return sorted(self.nodes)

    @property
    def eli(self):
        """
        Sorted list of edge indices.

        Returns
        -------
        edgelist: list(tuple)
            sorted list of edges. Each edge is represented as (source, target) where
            source and target are items in self.nodes.
        """
        return sorted(self.edges)

    @property
    def cidx(self):
        """
        Candidate idx.
        Returns
        -------
        cidx:int
            self.nodelist[self.cidx] returns the node corresponding
            to the candidate grain
        """
        return np.argmin([n['grain_type'] for n in self.nodelist])

    def to_image(self, color_by_type=False, flip_horizontal=False):
        """
        # color by type: add edges to grains and color by 'type' instead of id
        # flip horizontal: reverses order of each column of pixels.
        # if both color_by_type and match_spparks are True, the image should match
        # the first frame of the grains.mov output generated by spparks/meso
        Returns
        -------

        """
        img = np.zeros(self.metadata['img_size'], np.int16) - 1
        for i in self.nli:
            n = self.nodes[i]
            mask = RLE.decode(n['rle']).astype(np.bool)
            img[mask] = i

        if color_by_type:
            s = img.shape
            # coordinates of pixels in image
            cc, rr = np.meshgrid(np.arange(s[1]), np.arange(s[0]))
            # True where grain id of pixel [i,j] != pixel[i+1,j] or pixel[i, j+1]
            # including periodic boundary conditions
            edge_mask = np.logical_or(img[rr, cc] != img[(rr + 1) % s[0], cc],
                                      img[rr, cc] != img[rr, (cc + 1) % s[1]])
            edge_mask = binary_dilation(edge_mask, selem=np.ones((2, 2), np.bool))

            new_img = np.zeros((*img.shape, 3), np.uint8)

            type_colors = [(255, 255, 255), (106, 139, 152), (151, 0, 0)]

            # assign color to image
            for n in sorted(self.nodes):
                new_img[img == n, :] = type_colors[self.nodes[n]['grain_type']]

            # add edges
            new_img[edge_mask] = (0, 0, 0)

            img = new_img
        if flip_horizontal:
            img = img[::-1]
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

    def __repr__(self):
        gtype = self.metadata.get('gtype', 'single')
        if gtype == 'repeat':
            n_runs = len(self.metadata['path'])
            repr = f"{gtype} graph ({n_runs} runs, {len(self.nodes)} nodes, {len(self.edges)} edges))"
        else:
            repr = f'{gtype} graph ({len(self.nodes)} nodes, {len(self.edges)} edges)'
        return repr

    def __str__(self):
        return self.__repr__()

    def copy(self):
        """
        Return a copy of the graph
        Returns
        -------
        gc: Graph object
            copy of graph.
        """
        return deepcopy(self)

    def to_spparks_out(self):
        # compare graid_ids, mobilities, timesteps, grain_sizes, etc to
        # original hd5 file

        spparks_dict = {'initial.dream3d': {}, 'stats.h5': {}}

        # shift to_image() by 1 to get ids from 1-N instead of 0-N-1
        spparks_dict['initial.dream3d']['grain_ids'] = self.to_image2() + 1

        # go from integer grain type labels back to the vectors from spparks
        mob_map = np.stack([self.metadata['grain_types'][f'type_{x}'][0]
                            for x in range(3)])
        mobs = mob_map[[n['grain_type'] for n in self.nodelist]]
        mobs = np.pad(mobs, pad_width=((1, 0), (0, 1)), constant_values=0)
        spparks_dict['initial.dream3d']['grain_labels'] = mobs

        # stack grain sizes on array of zeros to account for offset (ie grain idx starts at 1)
        grain_sizes = self.metadata['grain_sizes']
        grain_sizes = np.pad(grain_sizes, pad_width=((0, 0), (1, 0)), constant_values=0)
        spparks_dict['stats.h5']['grainsize'] = grain_sizes

        spparks_dict['stats.h5']['time'] = self.metadata['timesteps']

        return spparks_dict

    def validate_with_spparks_outputs(self, root="", verbose=True):
        """
        check to make sure all data in graph exactly matches data contained in h5 files
        """

        # TODO implement this case if needed
        gtype = self.metadata.get('gtype', 'single')
        if gtype != 'single':
            print('spparks validation only available for single graphs')
            return 0

        if root == "":  # default, read from metadata
            root = self.metadata['path']

        sd = self.to_spparks_out()

        root = Path(root)  # forces root to be path object
        initfile = root / 'initial.dream3d'
        statsfile = root / 'stats.h5'

        assert initfile.is_file(), f'{str(initfile.absolute())} not found!'
        assert statsfile.is_file(), f'{str(statsfile.absolute())} not found!'

        init = h5py.File(initfile, 'r')
        stats = h5py.File(statsfile, 'r')
        sv = init['DataContainers']['SyntheticVolume']

        grain_ids = np.asarray(sv['CellData']['FeatureIds'])
        grain_labels_raw = np.asarray(sv['CellFeatureData']['AvgQuats'])
        timesteps = np.asarray(stats['time'])
        grain_sizes = np.asarray(stats['grainsize'])

        h5d = {'initial.dream3d': {'grain_ids': grain_ids, 'grain_labels': grain_labels_raw},
               'stats.h5': {'time': timesteps, 'grainsize': grain_sizes}}
        match = True
        for k, v in h5d.items():
            for kk, vv in v.items():
                match_i = np.all(vv == sd[k][kk])
                match = match and match_i
                if verbose:
                    print("file: {}, key: {}\n\tmatch: {}".format(k, kk, match_i))
        return match

    # TODO make this work for multi graph
    def to_dict(self):
        """
        Returns a json-compatible dictionary of graph nodes, edges, and metadata (ie all information needed to exactly reconstruct the graph exactly with self.from_dict()

        Parameters
        -----------
        None

        Returns
        -----------
        jd: dict
            json-compatible dictionary with keys 'nodes', 'edges', 'metadata', and values contain all of the information contained in the graph.
        """
        gtype = self.metadata.get('gtype', 'single')
        nodes = {str(k): deepcopy(v) for k, v in self.nodes.items()}
        for k in nodes.keys():
            nodes[k]['rle']['counts'] = nodes[k]['rle']['counts'].decode('utf-8')

        edges = {str(k): deepcopy(v) for k, v in self.edges.items()}

        # TODO remove when no longer needed
        #for k in edges.keys():
        #    edges[k]['dr'] = edges[k]['dr'].tolist()

        metadata = {'gtype': gtype}
        if self.metadata:  # if metadata is not an empty dict
            metadata = {k: deepcopy(v) for k, v in self.metadata.items()}

            for k, v in self.metadata.items():

                #  I think this is now redundant. I had to handle cases individually
                # so I don't think this step is repeated later.
                # However, for the sake of not breaking anything I'll leave it in for now.
                if type(v) == np.ndarray:
                    metadata[k] = v.tolist()
                    # self.metadata[int(k)] = np.asarray(g.edges[int(k)]['unit_vector'])
                elif k == "grain_types":
                    metadata[k] = {kk: (vv[0], vv[1]) for kk, vv in v.items()}
                elif k == 'path':
                    if gtype == 'single':
                        metadata[k] = str(v)  # single path to string
                    else:
                        metadata[k] = [str(x) for x in v]  # list of paths to list of strings
                elif k in ('timesteps', 'grain_sizes'):
                    if gtype == 'repeat':
                        metadata[k] = [x.tolist() for x in v]
                    else:
                        metada[k] = v.tolist()
                else:
                    metadata[k] = v

        jd = {'nodes': nodes,
              'edges': edges,
              'metadata': metadata, }
        return jd

    def to_json(self, path):
        """
        Saves graph to json file.

        Parameters
        -----------
        path: str or Path object
            path to save json file to
        """
        with open(path, 'w') as f:
            jd = self.to_dict()
            json.dump(jd, f)

    @staticmethod
    def from_dict(jd):
        """
        Loads graph object from dictionary. See to_dict() for format of dictionary.

        Parameters
        -----------
        jd: dict
            json-compatible dictionary with format defined in Graph.to_dict().

        Returns
        ----------
        g: Graph object
            Graph with data loaded from dict
        """
        G = Graph()
        for k, v in jd['nodes'].items():
            v['rle']['counts'] = bytes(v['rle']['counts'], 'utf-8')
            G.add_node(int(k), **v)

        for k, v in jd['edges'].items():
            k = [int(x) for x in k.strip('()').split(', ')]
            #todo remove when no longer needed
            #v['dr'] = np.asarray(v['dr'])
            G.add_edge(*k)#, **v)

        metadata = jd['metadata']
        md = {}
        if metadata:  # if metadata is not empty
            gtype = metadata.get('gtype', 'single')
            if gtype == 'single':  # read values for single run
                md['timesteps'] = np.asarray(metadata['timesteps'])
                md['grain_sizes'] = np.asarray(metadata['grain_sizes'])
                md['path'] = Path(metadata['path'])
            else:  # read values as lists for repeated trials
                md['timesteps'] = [np.asarray(x) for x in metadata['timesteps']]
                md['grain_sizes'] = [np.asarray(x) for x in metadata['grain_sizes']]
                md['path'] = [Path(x) for x in metadata['path']]
            md['img_size'] = tuple(metadata['img_size'])
            gt = {}
            for k, v in metadata['grain_types'].items():
                gt[k] = (v[0], v[1])
            md['grain_types'] = gt
            md['center_id'] = metadata['center_id']
            md['center_rc'] = np.asarray(metadata['center_rc'])

            md['gtype'] = gtype

        G.metadata = md

        return G

    @staticmethod
    def from_json(path, keep_path=True):
        """
        Loads graph object from json file.

        Parameters
        ----------
        path: str or Path object

        keep_path: Bool
            if True, path from original data source is preserved.
            Otherwise, the path of the json file itself is used.

        """
        with open(path, 'r') as f:
            G = Graph.from_dict(json.load(f))
        if not keep_path:
            G.metadata['path'] = path
        return G

class ListNode:
    def __init__(self, value=None, nxt=None):
        """
        Node for singly-linked list
        Parameters
        ----------
        value: Object
            value stored in list node
        nxt: ListNode object or None
            pointer to next node in list
        """

class LLQueue:
    def __init__(self, head=None, tail=None):
        """
        Queue with linked list backend
        Parameters
        ----------
        head, tail: ListNode or None
            pointer to first (head) and last (tail) items in list, respectively,
            or None if queue is empty
        """
           

def _add_node_to_img(g, img, src, target):
    """
    img: image to put grains on
    g: graph
    src, target: nodes in graph
        src should already be on graph, target should be a neighbor of src
        (ie the edge (src, target) should exist)
    """
    # note- need to handle wrapped grains later
    #xcenter = np.mean(np.where(img == src), axis=1)  # centroid of source grain
    edge = g.edges[(src, target)]

    bitmask = RLE.decode(g.nodes[target]['rle'])
    # shift1 = -np.mean(np.where(bitmask), axis=1)  # top left of bitmask to center of grain
    # shift2 = xcenter - (edge['unit_vector'] * edge['length'])
    # shift = shift1 + shift2
    shift = np.min(np.where(img == src), axis=1) - edge['dr']
    coords = np.round((np.stack(np.where(bitmask), axis=1) + shift)).astype(np.int).T
    coords = coords % np.reshape(img.shape, (2, 1))

    img[coords[0], coords[1]] = target

    return img


def _add_neighbors_to_img(g, img, src, visited):
    neighborhoods = [(src, [n for n in g.neighbors(src)])]
    for n in neighborhoods[0][1]:  # all of these nodes will be visited
        visited[n] = True

    while neighborhoods:  # continue while there are still neighbors
        edgelist = neighborhoods.pop(0)  # (src, [n for n in g.neighbors(src)] if not visited[n])
        src = edgelist[0]
        img = roll_img(img, src)  # avoid issues from periodic boundary conditions by always working at center of image
        while edgelist[1]:

            n = edgelist[1].pop(0)  # get target node

            img = _add_node_to_img(g, img, src, n,)  # add target node to graph

            new_edgelist = (n, [x for x in g.neighbors(n) if not visited.get(x)])

            for node in new_edgelist[1]:
                visited[node] = True  # all of these nodes will be visited

            if new_edgelist[1]:
                neighborhoods.append(new_edgelist)

    return img


def quickstats(g: Graph) -> dict:
    """
    Generates a dictionary of commonly used metrics to characterize a graph by.
    Currently only implemented for repeat graphs (may not work for single graphs).

    Stats are given in a dictinary with the following formatting:
    {
    fr: float
        number fraction of red grains in the system
    cgr: ndarray
        n_repeat element array of candidate growth ratio values for each trial in the system
    'size': tuple(int, int)
        system size
    'time': tuple(float, float, int)
        timesteps (start, stop, number of steps)
    }


    Parameters
    ----------
    g: Graph
        repeat graph to analyze.

    Returns
    -------
    stats: dict
        dictionary with formatting described above.

    """
    assert g.metadata['gtype']
    # 0: candidate, 1: blue (low mobility) 2: red (high mobility)
    grain_types = np.array([x['grain_type'] for x in g.nodelist])

    # index of candidate grain (grain_type = 0, which is the minimum value)
    cidx = np.argmin(grain_types)

    # growth ratio of each trial. Final size/initial size of candidate grain.
    cgr = np.array([x[-1, cidx]/x[0, cidx] for x in g.metadata['grain_sizes']])

    # fraction of red grains in initial microstructure
    fr = (grain_types == 2).sum()/len(grain_types)

    # img_size
    size = g.metadata['img_size']

    # timesteps
    t0 = g.metadata['timesteps'][0]
    for t in g.metadata['timesteps'][1:]:
        assert t[0] == t0[0]
        assert t[-1] == t0[-1]
        assert len(t) == len(t0)
    time = (t0[0], t0[-1], len(t0))
    results = {
        'fr': fr,
        'cgr': cgr,
        'size': size,
        'time': time
    }

    return results


if __name__ == "__main__":
    path = '/home/ryan/Desktop/tempruns/399363-2021-09-20-12-00-04-290587326-candidate-grains-repeat/spparks_init'
    path = '/home/ryan/Desktop/tempruns/399350-2021-09-17-23-31-08-923308997-candidate-grains-repeat'
    G = Graph.from_spparks_out(path)
    print(G.__repr__())
    p1 = '/home/ryan/Desktop/test1.json'
    G.to_json(p1)
    G2 = Graph.from_json(p1)
    p2 = '/home/ryan/Desktop/test2.json'
    G2.to_json(p2)
    with open(p1, 'r') as f:
        d1 = json.loads(f.read())
    with open(p2, 'r') as f:
        d2 = json.loads(f.read())
    # files are not exaclty the same because ordering is different
    # however, we can load json files and verify that
    # ALL of the key-value pairs match between both files
    for k in d1.keys():
        assert d1[k] == d2[k]
    for k in d2.keys():
        assert d2[k] == d1[k]
    print(d1.keys())
    print(d1['metadata']['gtype'])
    print(d2['metadata']['gtype'])
    # path = '/media/ryan/TOSHIBA EXT/Research/datasets/AFRL_AGG/datasets/candidate-grains-small-v1.0.1/train/2020_11_06_12_23_candidate_grains_master-run_533.json'
    # g = Graph.from_json(path)
    # img = g.to_image()
    # # img = g.to_image()
    # # fig, ax = plt.subplots()
    # # ax.imshow(img, cmap='rainbow')
    # # plt.show()
    # # print((img == -1).sum())