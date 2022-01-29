import mlflow
import numpy as np
import os
from pathlib import Path
import skimage
import skimage.io
from deepspparks.graphs import Graph
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model, load_model


def crop_img(img, newr, newc):
    """
    Crop image such that the center is preserved.

    For example, cropping 2 pixels off of the width of the image
    will remove the first and last columns, so that the image is still centered.

    Parameters
    ----------
    img: ndarray
        2 (grayscale) or 3 (color) dimensional array of pixels

    newr, newc: int
        new dimensions of image after cropping.

    Returns
    -----------
    img_crop: ndarray
        cropped image, newr * newc * 2 (grayscale) or 3 (color) array
    """
    r, c = img.shape[:2]
    r, c = r // 2, c // 2

    offsetr = newr % 2  # if image size is odd
    offsetc = newr % 2  # if image size is odd

    shiftr = newr // 2
    shiftc = newc // 2

    return img[r - shiftr - offsetr:r + shiftr, c - shiftc - offsetr:c + shiftc]


class Dataset:
    def __init__(self, dataset_name, crop, raw_root='/root/data/datasets',
                 processed_root='/root/data/processed', log=True):
        self.raw_root = Path(raw_root)  # path to directory containing dataset folder
        self.processed_root = Path(processed_root)  # root directory to save processed data in
        self.dataset_name = dataset_name
        self.feature_name = "imagenet-tfkeras-vgg16-fc1"
        self.target_name = "candidate_growth_ratio"
        self.class_labels = ("Normal grain growth", "Abnormal grain growth")
        self.train = None
        self.val = None
        self.test = None
        self.featurizer = None
        self.crop = crop  # whether image will be cropped during processing to focus on candidate grain
        self._raw_path = self.raw_root / self.dataset_name  # full path to raw data
        # full path to processed data: root/dataset_name/feature_name/target_name
        if crop:
            crop_str = "crop"
        else:
            crop_str = "no_crop"
        self._processed_path = Path(self.processed_root, self.dataset_name, 
                                    self.feature_name, crop_str, self.target_name)
        if log:
            self.log_()

    def load(self):
        """
        Loads existing processed data to self.{train, val, test}
        """
        for filename in ('train', 'val', 'test'):
            assert Path(self._processed_path, '{}.npz'.format(filename)).is_file()

        train = np.load(Path(self._processed_path, 'train.npz'))
        self.train = {'X': train['X'], 'y': train['y']}
        val = np.load(Path(self._processed_path, 'val.npz'))
        self.val = {'X': val['X'], 'y': val['y']}
        test = np.load(Path(self._processed_path, 'test.npz'))
        self.test = {'X': test['X'], 'y': test['y']}

    def log_(self):
        mlflow.set_tags({
                'dataset': self.dataset_name,
                'features': self.feature_name,
                'targets': self.target_name
            }
        )
        mlflow.log_param('crop', int(self.crop))

    def process(self, vgg16_path=None, force=False, log_featurizer=False, artifact_path=None):
        keys = ('train', 'val', 'test')
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

        # for mapping color of grain in image to const value
        mapper = {(255, 255, 255): 1.,  # candidate grain: highest signal
                  (106, 139, 152): 1 / 3,  # blue grain: low signal
                  (151, 0, 0): 2 / 3,  # red grain: medium signal
                  (0, 0, 0,): 0.}  # grain boundaries: 0 signal
        
        for subset in keys:  # train, val, test

            # use sort to keep order consistent --> easier to look at individual samples
            files = sorted((self._raw_path / subset).glob('*.json'))
            n = len(files)
            checkpoint = (0, n//2, n-1)  # index to save sample images for verification
            features = np.zeros((len(files), 4096), float)  # x data
            targets = np.zeros(len(files), float)  # classification targets
            for i, f in enumerate(files):  # train/graph_xxx.json
                g = Graph.from_json(f)
                cidx = g.cidx[0]  # only 1 candidate per simulation
                candidate_node = g.nodelist[cidx]
                size = candidate_node['grain_size'][0]
                targets[i] = size[-1]/size[0]

                # generate image where each grain is colored by its 'type',
                # the candidate grain is centered in the image, and
                # black edges ('grain boundaries') are added between grains
                img = g.to_image(True, center_grain=cidx)
                # map colors in image to mobility label
                r, c = img.shape[:2]
                imgray = np.zeros((r, c), float)
                for k, v in mapper.items():  # each color in image
                    # [1 x 1 x 3 array] so mapper can match with all color channels for single pixel
                    map_array = np.array(k)[np.newaxis, np.newaxis, :]
                    mask = np.all(img == map_array, axis=2)  # True where all color channels match for given pixel
                    imgray[mask] = v
                if self.crop:  # 'zoom in' to get to 224x224
                    imgray = crop_img(imgray, 224, 224)
                else:  # resize image to get to 224x224
                    imgray = skimage.transform.resize(imgray, (224, 224))  # imagenet size for vgg16
                imgray = skimage.img_as_ubyte(imgray)  # vgg16 input expects ubyte images (pixels are int (0,255))
                if i in checkpoint:
                    savepath = Path(artifact_path, '{}-{}-{}-processed.png'.format(subset, i, f.stem))
                    skimage.io.imsave(savepath, imgray)
                    mlflow.log_artifact(str(savepath), "sample-images")
                # vgg16 takes array of images with 3 color channels
                imgray = np.expand_dims(np.stack([imgray for _ in range(3)], axis=2), axis=0)
                # subtract mean, divide by stdev
                imgray = preprocess_input(imgray)
                # extract features
                # predict returns numpy array, no conversion from tensor needed
                fc1 = fc1_extractor.predict(imgray)
                features[i, :] = fc1

            data = {'X': features, 'y': targets}
            self.__setattr__(subset, data)
            np.savez(self._processed_path / '{}.npz'.format(subset), **data)
