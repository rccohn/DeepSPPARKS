class Dataset():
    def __init__(self, root, loader, featurizer):
        """

        Parameters
        ----------
        root: str or Path object
            path to data on disk
        loader: DataLoader object
            object to load data into memory
        featurizer

        """
        self.root = root

    def load

class BaseDataLoader():
    def __init__(self, root):
        pass

class BaseFeaturizer():
    def __init__(self, in_root, out_root):
        pass
