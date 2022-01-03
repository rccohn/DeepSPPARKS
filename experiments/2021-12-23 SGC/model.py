import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from typing import Union
from torch_geometric.nn import SGConv
from mlflow import log_params, set_tags, log_artifact


class SGCNet(torch.nn.Module):
    def __init__(self, k: int, data: Union[Data, None] = None, num_features: Union[int, None] = None,
                 num_classes: Union[int, None] = None, log: bool = True):
        super().__init__()

        # get number of features (input shape) and number of classes (output shape)
        if data is not None:
            num_features = data.num_features
            num_classes = data.num_classes = int(data.y.max()+1)

        # SGCNet only has one layer, the SGConv layer
        # multiple iterations of message passing handled with K parameter
        # from pyg documentation: cached should only be set to true for transductive learning
        # (ie where the same graph is used and only the labels of some nodes are unknown)
        self.conv = SGConv(in_channels=num_features, out_channels=num_classes, K=k, cached=False)

        self.double()  # forces weight tensors to double type, preventing difficult-to-debug errors later

        # for logging
        self.model_name = "SGCNet-classification-v1"
        self.k = k
        if log:
            self._log()

    def forward(self, data):
        x = self.conv(data.x, data.edge_index)
        return F.log_softmax(x, dim=1)

    def predict(self, data, mask=None):
        yp = self.forward(data).argmax(dim=1)
        if mask is None:
            return yp
        else:
            return yp[mask]

    def _log(self):
        set_tags({
            'model_name': self.model_name
            }
        )
        log_params({
            'k': self.k
            }
        )


    def mlflow_save_state(self, path):
        torch.save(self.state_dict(), path)
        log_artifact(path)


