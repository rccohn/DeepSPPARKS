import mlflow

# import os
from pathlib import Path

# from deepspparks.graphs import Graph

#  input a json graph
#  output a torch_geometric data object/save to disk
#  uses trained auto-encoders to generate node and edge features in the graph
# autoencoders are trained and stored in the experiment:
#            2022-03-08-learned-node-edge-features
# autoencoders can either be neural network or pca as a comparison


class Dataset:
    def __init__(
        self,
        name,
        mode,
        node_feature_model_uri,
        edge_feature_model_uri,
        raw_root="/root/data/datasets",
        processed_root="/root/data/processed",
        log=True,
    ):

        self.dataset_name = name
        self.mode = mode  # either "autoencoder" or "pca"
        self.node_feature_uri = node_feature_model_uri
        self.edge_feature_uri = edge_feature_model_uri
        self.target_name = "candidate_growth_ratio_repeats"

        self.raw_root = Path(raw_root, name)
        self.processed_root = Path(
            processed_root,
            name,
            node_feature_model_uri.replace("://", "__").replace("/", "_"),
            edge_feature_model_uri.replace("://", "__").replace("/", "_"),
            self.target_name,
        )

        self.train = None
        self.val = None
        self.test = None

        if log:
            self._log()

    def _log(self):
        mlflow.set_tags(
            {
                "dataset": self.dataset_name,
                "node_features": self.node_feature_uri,
                "edge_features": self.edge_feature_uri,
                "targets": self.target_name,
            }
        )

    def process(self):
        # TODO save hash of train, val, test directories to verify
        #      processed data has not been changed
        if self.mode == "pca":
            pass  # need pca model wrapper and number of components
