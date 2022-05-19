import mlflow
import torch
import numpy as np
import os
import json
from dirhash import dirhash
from shutil import rmtree
from typing import Union
from torch import flatten, Tensor
import torch.cuda
from tqdm import tqdm

from torch_geometric.data import Data, Dataset as TGDataset
from deepspparks.utils import aggregate_targets

from pathlib import Path

from deepspparks.graphs import Graph
from deepspparks.image import node_patch_to_image, edge_patch_to_image
from deepspparks.paths import RAW_DATA_ROOT, PROCESSED_DATA_ROOT

from typing import Optional

#  input a json graph
#  output a torch_geometric data object/save to disk
#  uses trained auto-encoders to generate node and edge features in the graph
# auto-encoders are trained and stored in the experiment:
#            2022-03-08-learned-node-edge-features
# auto-encoders can either be neural network or pca as a comparison


class Dataset:
    def __init__(
        self,
        params: dict,
        raw_root: Union[str, Path] = RAW_DATA_ROOT,
        processed_root: Union[str, Path] = PROCESSED_DATA_ROOT,
        log: bool = True,
    ):
        """
        params should have the following fields:
        "name": str
            name of directory containing raw dataset in raw_root
        "mode": str
            "pca" if encoder is a pca model, "autoencoder' if encoder
            is a pytorch autoencoder, or "test" to skip using a pre-trained model
            (simply flattens inputs for testing code)
        "node_feature_model_uri": str
            only needed if mode != "test", uri to load encoder from
        "edge_feature_model_uri": str
            only needed if mode != "test", uri to load encoder from
        "node_n_components": Optional[int] = None,  # only needed if mode == 'pca'
            only needed if mode == "pca", number of components to use
        "edge_n_components": Optional[int] = None,  # only needed if mode == 'pca'
            only needed if mode == "pca", number of components to use
        "subgraph_radius": int = 4,
            number of neighborhood shells of candidate grain to include in graph
        node_patch_bounds: list[int, int, int int]
            min_row, min_col, max_row, max_c for extracting node patches
            see argument 'bounds' from deepspparks.image.extract_node_patch()
        """

        self.dataset_name = params["dataset"]["name"]
        self.mode = params["encoder"]["mode"]  # either "autoencoder" or "pca" or "test"
        self.node_feature_model_uri = params["encoder"]["node_feature_model_uri"]
        self.edge_feature_model_uri = params["encoder"]["edge_feature_model_uri"]
        self.target_name = "candidate_growth_ratio_repeats"
        self.node_patch_bounds = params["encoder"]["node_patch_bounds"]

        self.subgraph_radius = params["dataset"]["subgraph_radius"]

        # maximum to extract during processing
        self.node_n_components_max = params["encoder"]["node_n_components_max"]
        self.edge_n_components_max = params["encoder"]["edge_n_components_max"]

        # number of components to actually load into dataloader
        self.node_n_components = self.node_n_components_max
        self.edge_n_components = self.edge_n_components_max

        self.raw_root = Path(raw_root, self.dataset_name)

        if self.mode == "test":
            self.processed_root = Path(
                processed_root, self.dataset_name, self.mode, self.target_name
            )

        else:
            self.processed_root = Path(
                processed_root,
                self.dataset_name,
                self.node_feature_model_uri.replace("://", "__").replace("/", "_"),
                "-".join([str(x) for x in self.node_patch_bounds]),
                "{}-components".format(self.node_n_components_max)
                if self.mode == "pca"
                else "",
                self.edge_feature_model_uri.replace(":/", "__").replace("/", "_"),
                "{}-components".format(self.edge_n_components_max)
                if self.mode == "pca"
                else "",
                "r={}".format(self.subgraph_radius),
                self.target_name,
            )

        self.info_file = self.processed_root / "info.json"

        self.train = None
        self.val = None
        self.test = None

        self.subset_labels = ("train", "val", "test")

        # only enable gpu for  autoencoders (cannot be used by pca or test)
        self.device = params["device"] if self.mode == "autoencoder" else "cpu"

        # aggregator and threshold for aggregating targets
        self.aggregator: Optional[str] = None
        self.threshold: Optional[str] = None

        # mask --> which grains are used for training and evaulation
        mask_key = params["dataset"]["mask"]
        mlflow.log_param("mask", mask_key)
        self.mask_fn = mask_functions(mask_key)

        if log:
            self._log()

    def _log(self):
        mlflow.set_tags(
            {
                "dataset": self.dataset_name,
                "node_features": self.node_feature_model_uri,
                "edge_features": self.edge_feature_model_uri,
                "targets": self.target_name,
            }
        )
        mlflow.log_params({"node_patch_bounds": str(self.node_patch_bounds)})

    def set_threshold_aggregator_ncomponents(
        self,
        threshold: float,
        aggregator: str,
        node_n_components: int = None,
        edge_n_components: int = None,
        log: bool = True,
    ):
        self.aggregator = aggregator
        self.threshold = threshold

        params = {"cgr_thresh": str(threshold), "repeat_aggregator": aggregator}

        if node_n_components is not None:
            self.node_n_components = node_n_components
            params["node_n_components"] = node_n_components
        if edge_n_components is not None:
            self.edge_n_components = edge_n_components
            params["edge_n_components"] = edge_n_components

        if log:
            mlflow.log_params(params)

    def load(self) -> bool:
        """
        Load existing processed data, if available. Returns True if processed data
        is successfully loaded, or False if it is not.

        Processed data is not successfully loaded if for ANY subset (train, val, test),
        the corresponding processed directory does not exist, or if its directory hash
        does not match the hash stored in the info file.

        """
        # returns True if loaded successfully, otherwise False

        print(
            "checking for existing processed data: {}".format(self.info_file.absolute())
        )

        if not self.info_file.is_file():
            print("Info file not found! {}".format(self.info_file))
            return False

        with open(self.info_file, "r") as f:
            info = json.load(f)

        for subset in self.subset_labels:
            subset_root = self.processed_root / subset
            print("Checking {}".format(subset_root))
            if not subset_root.is_dir():
                print("Directory not found")
                return False

            md5_ref = info[subset]["md5"]
            md5_actual = dirhash(subset_root, "md5")
            if md5_ref != md5_actual:
                print("hash mismatch! ref: {} actual: {}".format(md5_ref, md5_actual))
                return False
            print("hashes match")
            db = DatasetBackend(subset_root, info[subset]["n_files"], parent=self)
            self.__setattr__(subset, db)

        print("Processed data successfully loaded!")
        return True

    def process(self, force=False):
        # make feature extraction agnostic to pca vs neural autoencoder

        if not force:
            print("checking for processed data")
            result = self.load()
            if result:
                print("using existing processed data")
                # processed data successfully loaded, nothing to do
                return
            print("processed data not found")

        # force == true or processed data not loaded --> (re)process data
        print("processing data")

        if self.node_n_components == 0:
            # turn off node features -> set all to 1
            def node_encode(_: Tensor):
                return torch.DoubleTensor([1.0])

        elif self.mode == "pca":
            node_encode = pca_encode(
                self.node_feature_model_uri, n_components=self.node_n_components_max
            )

        elif self.mode == "autoencoder":
            node_encode = autoencoder_encode(self.node_feature_model_uri)

        elif self.mode == "test":
            # for testing only --> "encoder" returns the input
            # flattened for dimensional compatibility
            def node_encode(x: Tensor):
                return flatten(x, start_dim=0)

        else:
            raise ValueError('mode must be "pca" or "autoencoder"!')

        # repeat the above process for edge features
        if self.edge_n_components == 0:

            def edge_encode(_: Tensor):
                return torch.DoubleTensor([1.0])

        elif self.mode == "pca":
            edge_encode = pca_encode(
                self.edge_feature_model_uri, n_components=self.edge_n_components_max
            )
        elif self.mode == "autoencoder":
            edge_encode = autoencoder_encode(self.edge_feature_model_uri)

        elif self.mode == "test":

            def edge_encode(x: Tensor):
                return flatten(x, start_dim=0)

        else:
            raise ValueError('mode must be "pca" or "autoencoder"!')

        # keep track of number of files for re-loading
        file_info = {k: {} for k in self.subset_labels}

        # ~100 nodes and 525 edges within r=4 of candidate grain
        # store results for each graph as a separate file?

        for subset in self.subset_labels:
            subset_root = self.processed_root / subset
            if subset_root.is_dir():
                # remove existing processed data to prevent inconsistencies
                rmtree(subset_root)

            os.makedirs(subset_root, exist_ok=True)
            # iterate through all json graphs in directory
            # sort to preserve order for saving
            files = sorted((self.raw_root / subset).glob("*.json"))
            for i, f in tqdm(
                enumerate(files), desc="processing files {} ".format(subset)
            ):
                g = Graph.from_json(f)
                # get subgraph centered at candidate grain with specified radius
                sg: Graph = g.get_subgraph(r=self.subgraph_radius)
                sg.reset_node_idx()

                # extract node and edge features
                node_features = []
                node_targets = []
                grain_types = []

                for idx, node in enumerate(sg.nodelist):
                    node_features.append(
                        node_encode(
                            # TODO on repeated dataset, default bounds on node_patch()
                            #     do not extract center of image. Update node patch
                            #     to do this (will also need to rename processed data
                            #     in learned_features dataset)
                            torch.DoubleTensor(
                                node_patch_to_image(
                                    sg.node_patch(idx, self.node_patch_bounds)
                                )
                            )
                        )
                    )
                    # aggregate targets and store y values
                    gs = node["grain_size"]
                    growth_ratio = [x[-1] / x[0] for x in gs]
                    node_targets.append(growth_ratio)

                    # keep track of type of grain, allowing for selection of candidate
                    # (or other groupings)
                    grain_types.append(node["grain_type"])

                edge_index = [[], []]
                edge_attr = []
                for idx in sg.eli:
                    edge_index[0].append(idx[0])  # source node
                    edge_index[1].append(idx[1])  # target node
                    try:
                        edge_attr.append(
                            edge_encode(
                                torch.DoubleTensor(
                                    edge_patch_to_image(sg.edge_patch(idx))
                                )
                            )
                        )
                    except Exception as e:
                        print(f)
                        raise e
                # convert to torch-geometric data format
                x = torch.stack(node_features)
                edge_index = torch.LongTensor(edge_index)
                edge_attr = torch.stack(edge_attr)
                y = torch.DoubleTensor(node_targets)
                types = torch.ByteTensor(grain_types)
                data = Data(
                    x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    y=y,
                    grain_types=types,
                )
                # save graph in
                torch.save(data, subset_root / "{:04}.pt".format(i))

            file_info[subset]["n_files"] = len(files)
            file_info[subset]["md5"] = dirhash(subset_root, "md5")
            dsb = DatasetBackend(subset_root, file_info[subset]["n_files"], parent=self)
            self.__setattr__(subset, dsb)

        # store dataset size and hash of directory to ensure consistency
        with open(self.info_file, "w") as f:
            json.dump(file_info, f)


class DatasetBackend(TGDataset):
    def __init__(self, root: Path, n: int, parent: Dataset):
        super().__init__()
        self.root = root
        self.n = n
        self.parent = parent

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        # TODO store repeat aggregator as dataset attribute,
        #     apply the aggregator to item.y after loading?
        data = torch.load(self.root / ("{:04d}.pt".format(i)))

        data = aggregate_targets(
            data=data,
            aggregator=self.parent.aggregator,
            threshold=self.parent.threshold,
        )

        # mask can be for selecting which grains to generate predictions for
        # (ie only candidate grains)
        data.mask = self.parent.mask_fn(data)

        nc = self.parent.node_n_components
        if nc:  # not 0
            data.x = data.x[:, :nc]
        else:  # == 0
            data.x = torch.ones((len(data.x), 1), dtype=data.x.dtype)

        ec = self.parent.edge_n_components
        if ec:  # not 0
            data.edge_attr = data.edge_attr[:, :ec]
        else:  # == 0
            data.edge_attr = torch.ones(
                (len(data.edge_attr), 1), dtype=data.edge_attr.dtype
            )

        return data


def mask_functions(key):
    _mask_functions = {
        "candidate_grain": lambda x: x.grain_types == 0,  # only select candidate grain
        # no mask, select all grains in system
        "all_grains": lambda x: torch.BoolTensor(x.grain_types.shape),
    }

    assert key in _mask_functions.keys(), "invalid mask type {}".format(key)
    return _mask_functions[key]


def pca_encode(model_uri, n_components):
    # wraps pca model to be compatible with 4d torch tensors
    # (mini batch of multichannel images)
    # using the desired number of components

    # load pca model from uri
    model = mlflow.sklearn.load_model(model_uri)

    def encode_fn(x: torch.Tensor) -> torch.DoubleTensor:
        # tensor to numpy

        x = x.detach().cpu().numpy()
        # 4d list of (channel, row, col) images
        # to 2d data matrix
        x = x.reshape(-1) - model.mean_

        # encode only using number of desired components
        x_encode = np.dot(x, model.components_[:n_components].T)
        if model.whiten:
            x_encode /= np.sqrt(model.explained_variance_[:n_components])

        # array to tensor
        x_encode = torch.DoubleTensor(x_encode)
        return x_encode

    return encode_fn


def autoencoder_encode(model_uri):
    # load model
    model = mlflow.pytorch.load_model(model_uri)

    def encode_fn(x: torch.Tensor) -> torch.DoubleTensor:
        # only use encoder portion
        return model.encode(x)

    return encode_fn
