import mlflow
import numpy as np
import os
from pathlib import Path
from pycocotools.mask import decode
from skimage.measure import regionprops_table
from deepspparks.graphs import Graph, combine_graphs, roll_img
import torch
from torch_geometric.data import Data


class Dataset:
    def __init__(self, name, raw_root='/root/data/datasets',
                 processed_root='/root/data/processed', log=True):

        if name is None:
            self.dataset_name = self.raw_root.name  # folder should be name of dataset
        else:
            self.dataset_name = name
        self.feature_name = "grain_properties_single_v1"  # description of features
        self.target_name = "candidate_growth_ratio_r20"  # description of targets

        self.raw_root = Path(raw_root, name)
        self.processed_root = Path(processed_root, name, self.feature_name, self.target_name)

        self.class_labels = ("Normal grain growth", "Abnormal grain growth")
        self.means = None  # means for normalizing
        self.stds = None  # stds for normalizing
        self.train = None  # training dataset
        self.val = None  # validation dataset
        self.test = None  # testing dataset
        if log:
            self._log()

    def load(self):
        """
        Loads existing processed data to self.{train, val, test}.
        """
        for filename in ('train', 'val', 'test'):
            assert Path(self.processed_root, '{}.pt'.format(filename)).is_file()

        self.train = torch.load(Path(self.processed_root, 'train.pt'))
        self.val = torch.load(Path(self.processed_root, 'val.pt'))
        self.test = torch.load(Path(self.processed_root, 'test.pt'))

        meanspath = Path(self.processed_root, 'normalization_means.pt')
        if meanspath.is_file():
            self.means = torch.load(meanspath)
        stdspath = Path(self.processed_root, 'normalization_stds.pt')
        if stdspath.is_file():
            self.stds = torch.load(stdspath)

    def process(self, force=False):
        """
        Process json-graphs saved in self.raw_root/['train','val','test']

        Compute features associated with grain_properties_single_v1
        and targets associated with candidate_grain_thresh_cgr>10

        Save as pytorch objects in self.processed_root
        Parameters
        ----------
            force: bool
                if True, raw data is always processed, even if
                         files exist in self.processed_root
                else, raw data is only processed if all files
                    are not in self.processed_root

        Returns:
            None

        Saves:
            train.pt, val.pt, test.pt: torch_geometric.Data objects
                datasets containing training, validation, and test data
            normalization_means, normalization_stds: torch.Tensor objects
                tensors containing mean and std values used for feature normalization
        """
        keys = ('train', 'val', 'test')
        if not force:  # if force == True, skip this step and always process files
            try:  # load existing data
                self.load()
                print('Loaded existing processed data')
                return
            except AssertionError:  # if any files are not found, re-process whole dataset
                print('Processed data not found, processing data now.')

        os.makedirs(self.processed_root, exist_ok=True)
        subsets = {}
        for subset in keys:
            raw = Path(self.raw_root, subset)
            files = sorted(raw.glob('*.json'))
            print("processing {} ({} files)".format(raw.absolute(), len(list(raw.glob('*.json')))))
            assert raw.is_dir() and len(files)
            # load all json graphs, combine into one large multi graph
            subgraphs = []
            for f in files:
                sg = Graph.from_json(f)
                # won't use more than 4 iterations of message passing, only need subgraph with r=4
                # only 1 candidate grain per graph, with node index g.cidx[0]
                sg = sg.get_subgraph(center=sg.cidx[0], r=4)
                subgraphs.append(sg)
            g = combine_graphs(subgraphs)

            # regionprops for featurization
            props = ('area', 'perimeter', 'equivalent_diameter', 'major_axis_length', 'minor_axis_length')
            x = np.zeros((len(g.nodes), len(props) + 2), dtype=float)  # props + grain "type" + degree

            # Note it is assumed all nodes in graph have same number of repeats
            y = np.zeros((len(g.nodes), len(g.nodes[0]['grain_size'])), float)  # True/False classification of abnormal grain growth

            nli = g.nli  # ordered node indices

            for n in nli:
                node = g.nodes[n]  # select node
                bitmask = roll_img(decode(node['rle']), 1.)  # get grain mask, and center in image
                # compute features from rprops
                rprops = regionprops_table(bitmask.astype(np.uint8), properties=props)
                feat = np.zeros(len(props) + 2)

                # first 2 features are grain type (1 for candidate, 0.5 for red grain, 0 for blue grain)
                #   and number of neighbors (node degree)
                feat[0] = [1, 0, 0.5][node['grain_type']]  # 1 -> candidate, 0 -> low mobility, 0.5 -> high mobility
                feat[1] = g.in_degree[n]  # number of incoming edges
                # rest of features are determined from regionprops
                for i, p in enumerate(props):
                    feat[2 + i] = float(rprops[p][0])

                # target is growth ratio for
                gs = np.stack(node['grain_size'])

                y[n] = (gs[:, -1]/gs[:, 0])  # candidate growth ratio

                x[n] = feat

            edge_index = np.array(g.eli, int).T  # edge index in pyg format
            candidate_mask = np.zeros(len(g.nodes), bool)  # mask with True value at index of candidate grains
            candidate_mask[g.cidx] = 1

            x = torch.from_numpy(x)
            edge_index = torch.from_numpy(edge_index)
            candidate_mask = torch.from_numpy(candidate_mask)
            y = torch.from_numpy(y)
            subsets[subset] = Data(x=x, edge_index=edge_index, y=y, candidate_mask=candidate_mask)

        # mean and standard deviation for normalization
        # note that we skip the first column (grain type) as it is categorical and already bound between 0 and 1

        self.means = torch.mean(subsets['train'].x[:, 1:], dim=0)
        self.stds = torch.std(subsets['train'].x[:, 1:], dim=0)

        # normalize continuous data to 0-mean and unit std
        for k, v in subsets.items():
            subsets[k].x[:, 1:] = (v.x[:, 1:] - self.means) / self.stds
            # save data to processed_root
            processed = Path(self.processed_root, '{}.pt'.format(k))
            torch.save(subsets[k], processed)

        torch.save(self.means, Path(self.processed_root, "normalization_means.pt"))
        torch.save(self.stds, Path(self.processed_root, "normalization_stds.pt"))

        self.train = subsets['train']
        self.val = subsets['val']
        self.test = subsets['test']

    @staticmethod
    def load_from(raw_root, processed_root):
        """
        Load previously saved data

        Parameters
        ----------
        raw_root, processed_root: str or Path object
            path to raw and processed data

        Returns
        ---------
            data: Dataset object
                loaded dataset
        """
        # initialize Dataset object
        data = Dataset(raw_root, processed_root)

        # load training, validation, and test set
        data.train = torch.load(Path(data.processed_root, 'train.pt'))
        data.val = torch.load(Path(data.processed_root, 'val.pt'))
        data.test = torch.load(Path(data.processed_root, 'test.pt'))

        # load normalization constants
        data.means = torch.load(Path(data.processed_root, "normalization_means.pt"))
        data.stds = torch.load(Path(data.processed_root, "normalization_stds.pt"))

        return data

    def _log(self):
        """
        Logs relevant tags to mlflow run.

        Note that it is assumed that a run has already been started. If not, mlflow will start
        a new one with default parameters, which is usually not desirable.

        Returns
        -------
            None

        Logs
        ------
            dataset name, feature name, target name saved as tags in mlflow tracking server
        """
        mlflow.set_tags({
            'dataset': self.dataset_name,
            'features': self.feature_name,
            'targets': self.target_name
            }
        )
