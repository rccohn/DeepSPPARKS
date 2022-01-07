import mlflow
import numpy as np
import os
from pathlib import Path
import skimage
import skimage.io
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model, load_model


class Dataset:
    def __init__(self, raw_root, processed_root, dataset_name, log=True):
        self.raw_root = Path(raw_root)  # path to directory containing dataset folder
        self.processed_root = Path(processed_root)  # root directory to save processed data in
        self.dataset_name = dataset_name
        self.feature_name = "imagenet-tfkeras-vgg16-fc1"
        self.target_name = "NEU-cls"
        self.labels_inv = ['Cr', 'In', 'Pa', 'PS', 'RS', 'Sc']  # maps int to str label
        self.labels_fwd = {y: x for x, y in enumerate(self.labels_inv)}  # maps str label to int
        self.class_labels = ("Normal grain growth", "Abnormal grain growth")
        self.X = None
        self.y = None
        self.files = None
        self.featurizer = None
        self._raw_path = self.raw_root / self.dataset_name  # full path to raw data
        # full path to processed data: root/dataset_name/feature_name/target_name
        self._processed_path = Path(self.processed_root, self.dataset_name,
                self.feature_name, self.target_name)

        if log:
            self._log()

    def load(self):
        """
        Loads existing processed data to self.{train, val, test}
        """
        data_file = Path(self._processed_path, 'data.npz')
        print("looking for existing data: ", data_file)
        assert data_file.is_file

        data = np.load(data_file)
        assert data['X'].shape == (1800, 4096)
        assert data['y'].shape == (1800, )

        self.X = data['X']
        self.y = data['y']
        self.files = data['files']

    def _log(self):
        mlflow.set_tags({
                'dataset': self.dataset_name,
                'features': self.feature_name,
                'targets': self.target_name
            }
        )

    def process(self, vgg16_path=None, force=False, log_featurizer=False, artifact_path=None):
        if not force:  # if force == True, skip this step and always process files
            try:  # load existing data
                self.load()
                print('Loaded existing processed data')
                return
            except AssertionError:  # if any files are not found, re-process whole dataset
                print('Processed data not found, processing data now.')

        os.makedirs(self._processed_path, exist_ok=True)

        if vgg16_path is None:
            vgg16 = VGG16(include_top=True, weights="imagenet")
        else:
            vgg16 = VGG16(include_top=True, weights=vgg16_path)
        # preprocess inputs: normalize by rgb mean [0.485 0.456 0.406]
        # and standard deviation [0.229, 0.224, 0.225], or equivalent based on format of images
        fc1_extractor = Model(inputs=vgg16.inputs, outputs=vgg16.get_layer('fc1').output)
        # since we are only extracting, we don't actually need to compile the model
        # however, if we don't, tensorflow will throw a warning
        # so it is easier to just compile with default settings
        fc1_extractor.compile('sgd', 'categorical_crossentropy', ['accuracy'])
        self.featurizer = fc1_extractor
        if log_featurizer:
            feat_path = Path(artifact_path, 'featurizer.h5')
            self.featurizer.save(feat_path)
            mlflow.log_artifact(str(feat_path), artifact_path='models')

        #  use sort to keep order consistent --> easier to look at individual samples
        files = sorted(self._raw_path.glob('*.bmp'))

        n = len(files)
        assert n == 1800

        features = np.zeros((len(files), 4096), float)  # x data
        targets = np.zeros(len(files), bool)  # classification targets

        for i, f in enumerate(files):  # train/graph_xxx.json
            targets[i] = self.labels_fwd[f.name.split('_')[0]]
            imgray = skimage.io.imread(f, as_gray=True)
            # resize image to get to 224x224
            imgray = skimage.transform.resize(imgray, (224, 224))  # imagenet size for vgg16
            imgray = skimage.img_as_ubyte(imgray)  # vgg16 input expects ubyte images (pixels are int (0,255))
            # vgg16 takes array of images with 3 color channels
            imgray = np.expand_dims(np.stack([imgray for _ in range(3)], axis=2), axis=0)
            # subtract mean, divide by stdev
            imgray = preprocess_input(imgray)
            # extract features
            # predict returns numpy array, no conversion from tensor needed
            fc1 = fc1_extractor.predict(imgray)
            features[i, :] = fc1

        self.X = features
        self.y = targets
        self.files = files

        data = {'files': [x.name for x in files],
                'X': features,
                'y': targets}
        np.savez(self._processed_path / 'data.npz', **data)
