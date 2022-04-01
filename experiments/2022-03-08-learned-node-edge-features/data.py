import os
from pathlib import Path
from pycocotools.mask import decode
import torch
from torch.utils.data import Dataset as TorchDataset
from patch_compression import recover_node_patch, recover_edge_patch
from typing import Union, Callable, Optional


class DatasetBackend(TorchDataset):
    """
    Allows fast loading of compressed images from disk for very large datasets.
    """

    def __init__(
        self, root: Union[str, Path], which: str, transform: Callable, size: int
    ):
        """
        Parameters
        ----------
        root: str or Path object
            root directory containing all text files in a single directory.

        which: str
            Valid values are "node" or "edge"
            Determines the method for loading data from disk.

        transform: Callable
            function to take dictionary representation of patch
            and convert it into an input for the autoencoder.
            The function should take a single argument, the dictionary
            representation of the loaded patch, and return a single
            torch Tensor in the correct format for the autoencoder as
            output.

        size: int
            length of square patch. Not storing the size
            with the rest of the patch reduces the amount of
            repeated information stored within each patch, since
            all patches in the same directory should have the same size.
        """

        super(DatasetBackend, self).__init__()
        # maps int 'grain type' to float intensity to use
        # grain type    intensity
        # 0 (candidate)  0.6
        # 1 (blue)       0.2
        # 2 (red)        0.4

        self.mp = (0.6, 0.2, 0.4)
        self.root = Path(root)
        n = 0  # length computation
        # do not convert to list and call len() --> memory efficient
        for _ in self.root.glob("*"):
            n += 1
        self.n = n
        assert self.n, "no files found in {}".format(self.root)
        if which == "node":
            self.loader = recover_node_patch
        elif which == "edge":
            self.loader = recover_edge_patch
        else:
            raise ValueError("parameter 'which' must be 'node' or 'edge'")

        self.transform = transform
        self.size = size

    def __getitem__(self, i):
        """
        Loads item from disk to memory
        """
        # read text file
        # convert string of data into dictionary
        # containing patch information
        # convert dictionary into torch tensor
        # that can be used as input to the auto-encoder

        with open(self.root / "{:09d}".format(i), "r") as f:
            return self.transform(self.loader(f.read(), self.size))

    def __len__(self):
        """
        Returns
        -------
        n: int
            number of items in dataset
        """
        return self.n


class Dataset:
    """
    contains root directory for train, val,
    and testing subsets
    """

    def __init__(
        self,
        name: str,
        which: str,
        raw_root: Union[str, os.PathLike] = "/root/data/datasets",
        mapper: Optional[dict] = None,
        log: bool = True,
    ):
        """

        If no mapper is specified, the following default one is used:
        {
            'background': 0.2, # intensity of background pixels
            'edge_offset': 0.1, # how much more intense pixels
                      # on grain boundary are
            'grain_intensities': (0.8, 0.4, 0.6),
            # mapper['grain_intensities'][node['grain_type']]
            # gives the intensity for a given grain
            # default values give values of 0.8 for the candidate grain,
            # 0.6 for red grains, and 0.4 for blue grains
        }
        Parameters
        ----------
        name: str
            name of dataset
        which: str
            valid values are "node" or "edge".
            Determines which patches will be loaded.
        raw_root: path-like
            Path of dataset parent directory (in docker container!)
        mapper: dict or None
            contains values for determining the intensity
            of pixel. If None, the default (described above)
            is used.
        """
        self.raw_root = Path(raw_root, name)
        sizes = parse_patch_sizes(self.raw_root / "patch_sizes.txt")

        self.which = which
        self.size = sizes[which]

        self.train = None
        self.val = None
        self.test = None

        if mapper is None:  # use default mapper
            # see docstring for description of values
            mapper = {
                "background": 0.2,
                "edge_offset": 0.1,
                "grain_intensities": (0.8, 0.4, 0.6),
            }
        self.mapper = mapper

    def process(
        self,
    ):
        """
        Store training, validation, and subset directories as torch-compatible datasets.
        """
        # TODO easy: store dataset size as text file
        #      more involved: compute and store sha256 checksum of file contents
        #      so that correctness of dataset can be verified without re-processing

        # need transform for loading patch
        # todo clean up transform vs loader --> the same information (ie size param)
        #     is spread across multiple locations, and is a bit messy
        datasets = {
            k: DatasetBackend(
                root=Path(self.raw_root, k, self.which),
                which=self.which,
                transform=_get_transform(self.which, self.mapper),
                size=self.size,
            )
            for k in ("train", "val", "test")
        }
        self.train = datasets["train"]
        self.val = datasets["val"]
        self.test = datasets["test"]


def _get_transform(which: str, mapper: dict) -> Callable:
    """ "
    gets appropriate transform for the given type

    Params
    ------
    which: str
        can be "node" or "edge"

    mapper: dict
        see Dataset.mapper

    Returns
    --------
    transform_fn: Callable
        appropriate transform to load node or edge patches, respectively.
    """
    if which == "node":
        return _node_transform(mapper)
    elif which == "edge":
        return _edge_transform(mapper)
    else:
        raise ValueError("parameter 'which' should be 'node' or 'edge'")


def _node_transform(
    mapper: dict,
) -> Callable:
    """
    Transform converts compressed node dictionary into torch tensor.

    Params
    ------
    mapper: dict
        see Dataset.mapper
    """

    def transform_fn(patch: dict):
        # load image and set bakcground pixels to fixed value
        img = torch.zeros(patch["size"], dtype=torch.double) + mapper["background"]
        # fill in grain pixels with different intensity
        img[decode(patch).astype(bool)] = mapper["grain_intensities"][patch["type"]]
        # add dimension for color channel, so img is channels x rows x columns
        img = img.unsqueeze(0)
        return img

    return transform_fn


def _edge_transform(mapper: dict) -> Callable:
    """
    Params
    ------
    mapper: dict
        see Dataset.mapper

    """

    def transform_fn(patch: dict):
        # create image and fill in background with appropriate value
        img = torch.zeros(patch["size"], dtype=torch.double) + mapper["background"]
        # for each of the 2 grains on the grain boundary...
        for c, t, ec in zip(patch["counts"], patch["types"], patch["edge_coords"]):
            rle = {"size": patch["size"], "counts": c}
            # fill in the appropriate pixels for each grain with their respective
            # intensity value
            img[decode(rle).astype(bool)] = mapper["grain_intensities"][t]
            # ec: edge coords: [[row_indices], [col_indices]]
            # also offset pixels belonging to the grain that are on the grain boundary
            # (even if 2 grains have same intensity, the boundary will still be
            # highlighted)
            img[ec[0], ec[1]] += mapper["edge_offset"]
        # add dimension for color channel, so img is channels x rows x columns
        img = img.unsqueeze(0)
        return img

    return transform_fn


def parse_patch_sizes(size_file):
    """
    Read file that stores the common patch size
    information for all patches.

    Since all patches are the same size, storing the size
    only once reduces redundant information stored in each
    compressed file.
    """
    sizes = {}
    with open(size_file, "r") as f:
        data = f.readlines()
        window = [int(x) for x in data[0].split(": ")[1].split("_")]
        assert (
            window[2] - window[0] == window[3] - window[1]
        ), "node patch should be square"
        sizes["node"] = window[2] - window[0]
        sizes["edge"] = int(data[1].split(": ")[1])
    return sizes
