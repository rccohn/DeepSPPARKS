import mlflow
import numpy as np
import os
from pathlib import Path
from pycocotools.mask import encode, decode
from deepspparks.graphs import Graph, roll_img
import torch
from torch.utils.data import Dataset as TorchDataset, DataLoader


class DatasetBackend(TorchDataset):
    """
    Allows fast loading of compressed images from disk for very large datasets.
    """

    def __init__(self, root):
        """
        Parameters
        ----------
        root: string or Path object
            root directory containing all .pt files in dataset.
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
        for _ in self.root.glob("*.pt"):
            n += 1
        self.n = n

    def __getitem__(self, i):
        """
        Loads item from disk to memory
        """
        rle = torch.load(self.root / "{:08d}.pt".format(i))
        # modifying decode to use modified intensity values instead of
        # 0,1 would require re-writing C code. Instead we load the
        # binary image and then transform the data afterwards.
        img = decode(rle) * self.mp[rle["id"]] + 0.2
        return img

    def __len__(self):
        """
        Returns
        -------
        n: int
            number of items in dataset
        """
        return self.n


class Dataset:
    def __init__(
        self,
        name,
        raw_root="/root/data/datasets",
        processed_root="/root/data/processed",
        log=True,
    ):

        if name is None:
            self.dataset_name = self.raw_root.name  # folder should be name of dataset
        else:
            self.dataset_name = name
        # extracts image of individual grain, with pixel values determined by grain type
        # centers grain using deepspparks.image.roll_img, and crops to a 96x96 patch
        # scales all pixels to fit within sigmoid prediction (ie no values of 0 or 1,
        # which can never be predicted exactly)
        # background pixels are 0.2, blue grain pixels are 0.4, red grain pixels are
        # 0.6, and candidate grain pixels are 0.8
        self.feature_name = "grain_patches_v1"  # description of features
        # for an auto-encoder model, the
        # target is to reconstruct the inputs
        self.target_name = self.feature_name

        self.raw_root = Path(raw_root, name)
        self.processed_root = Path(
            processed_root, name, self.feature_name, self.target_name
        )

        self.means = None  # means for normalizing
        self.stds = None  # stds for normalizing

        # torch datasets for train, validation, test
        self.train = None
        self.val = None
        self.test = None

        # torch data loaders for train, validation, test
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        if log:
            self._log()

    def make_dataloaders(self, **kwargs):
        """
        Creates torch dataloaders for training, validation, and test sets.
        Datasets are created in the Dataset processed root after calling process()

        Parameters
        ----------
        kwargs: arguments passed to the torch DataLoader used to load batches of data.

        """

        # idea: verify that number of images in the processed data folder matches the
        # number of graphs in the raw data folder

        keys = ("train", "val", "test")

        for k in keys:
            json_graph_dir = self.raw_root / k
            image_dir = self.processed_root / k
            count_imgs = 0
            for _ in image_dir.glob("*.pt"):
                count_imgs += 1
            assert count_imgs, "no images found in {}".format(image_dir)
            count_nodes = 0
            for f in json_graph_dir.glob("*.json"):
                g = Graph.from_json(f)
                count_nodes += len(g.nodes)
            assert (
                count_imgs == count_nodes
            ), "{}: {} graphs found but only {}" "images found".format(
                k, count_nodes, count_imgs
            )
            dataset = DatasetBackend(image_dir)
            loader = DataLoader(dataset, **kwargs)
            self.__setattr__(k, dataset)
            self.__setattr__("{}_loader".format(k), loader)
        return

    def process(self, force=False, **kwargs):
        """
        Parameters
        ----------
        force: bool
            if True, dataset is always processed and over-writes existing processed
               data, if any. Otherwise, if processed dat is found on disk, it
               is loaded with make_dataloaders()
        kwargs:
            passed to torch dataloaders used to load batches from disk
        """

        if not force:  # if force == True, skip this step and always process files
            try:  # load existing data
                self.make_dataloaders(**kwargs)
                print("Loaded existing processed data")
                return
            except AssertionError:  # if any files are missing, re-process whole dataset
                print("Processed data not found, processing data now.")
        if not self.processed_root.exists():
            os.makedirs(self.processed_root)

        keys = ("train", "val", "test")
        # saving individual images for every grain in every graph requires too much
        # memory. Thus, images are RLE compressed, along with the grain 'type'

        # sort files to ensure consistent ordering every time features are computed.
        for k in keys:
            subset_raw_root = self.raw_root / k
            subset_processed_root = self.processed_root / k
            if not subset_processed_root.exists():
                os.makedirs(subset_processed_root)

            node_idx = 0
            print("processing subset {}".format(k))
            print(
                subset_raw_root.absolute(),
                subset_raw_root.exists(),
                list(subset_raw_root.glob("*")),
            )
            for f in sorted(subset_raw_root.glob("*.json")):
                g = Graph.from_json(f)
                # use sorted nodelist to ensure constant ordering
                for n in g.nodelist:
                    # extract only the rle representation and grain type
                    # to minimize overhead of loading data during experiments
                    rle = n["rle"]
                    img = roll_img(decode(rle), 1)[208:304, 208:304]
                    rle = encode(np.asfortranarray(img))
                    rle["id"] = n["grain_type"]
                    savepath = subset_processed_root / "{:08d}.pt".format(node_idx)
                    torch.save(rle, savepath)
                    node_idx += 1

        self.make_dataloaders(**kwargs)

    def _log(self):
        """
        Logs relevant tags to mlflow run.

        Note that it is assumed that a run has already been started. If not, mlflow will
        start a new one with default parameters, which is usually not desirable.

        Returns
        -------
            None

        Logs
        ------
            dataset name, feature name, target name saved as tags in tracking server
        """
        mlflow.set_tags(
            {
                "dataset": self.dataset_name,
                "features": self.feature_name,
                "targets": self.target_name,
            }
        )


if __name__ == "__main__":
    raw_root = Path("/mlflow-data")
    processed_root = Path("/scratch/tmp-processed")
    assert raw_root.is_dir()
    ds = Dataset("candidate-grains-toy-lmd-v2.0.0", raw_root, processed_root, log=False)
    ds.process(force=False, **{"batch_size": 3})
    print(next(iter(ds.val_loader)).shape)
