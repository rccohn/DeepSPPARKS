import os

from pathlib import Path
import pycocotools.mask as RLE
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import skimage.measure
from src.graphs import Graph
#from src.utils import batcher
from skimage.transform import resize
from skimage.io import imsave

from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
import h5py
from skimage.morphology import binary_dilation
from src.image import roll_img

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
    r,c = img.shape[:2]
    r,c = r//2, c//2
    
    offsetr = newr % 2
    offsetc = newr % 2
    
    shiftr = newr//2
    shiftc = newc//2
    
    return img[r-shiftr-offsetr:r+shiftr, c-shiftc-offsetr:c+shiftc]


def graph_to_img(g):
    """
    Turns graph into image of grains. Grayscale intensity is determined by
    the 'type' of grain- dark grey/light grey/white correspond to blue/red/candidate
    in the simulations. Grain boundaries are overlaid in black.
    
    Parameters
    -----------
    g: Graph object
        graph to extract image from.
    
    Returns
    -----------
    img: ndarray
        r x c grayscale image of intensity values.
    """
    img = g.to_image2()
    s = img.shape
    cc, rr = np.meshgrid(np.arange(s[1]), np.arange(s[0]))
    
    edge_mask = np.logical_or(img[rr, cc] != img[(rr + 1) % s[0], cc],
                                      img[rr, cc] != img[rr, (cc + 1) % s[1]])
    
    edge_mask = binary_dilation(edge_mask, selem=np.ones((2,2), np.bool))
    new_img = np.zeros((*s, 3), np.uint8)
    
    
    type_colors = [(255, 255, 255), (106, 139, 152), (151, 0, 0)]
    
    for n in g.nli:
        new_img[img == n, :] = type_colors[g.nodes[n]['grain_type']]
    
    new_img[edge_mask] = (0, 0, 0)
    
    img = new_img
    
    return img


def featurize(g, preprocess_path, extractor, crop=False):
    """
    turn graph into pyg dataset
    """
    img = graph_to_img(g)
    mapper = {(255, 255, 255):1.,
             (106, 139, 152): 1/3,
             (151, 0, 0): 2/3,
             (0, 0, 0,): 0.}
    r, c = img.shape[:2]
    imgray = np.zeros((r,c), np.float64)
    for k,v in mapper.items():
        kb = np.array(k)[np.newaxis, np.newaxis, :]
        mask = np.all(img == kb, axis=2)
        imgray[mask] = v
    imgray = skimage.img_as_ubyte(imgray)
    imgray = roll_img(imgray, 255)
    if crop:
        imgray = crop_img(imgray, 224, 224)
    else:
        resize(imgray, (224, 224))
    imgray = skimage.img_as_ubyte(imgray)
    imgray = roll_img(imgray, 255)
    imsave(preprocess_path, imgray)
    img = load_img(preprocess_path)
    img = img_to_array(img)
    img = preprocess_input(img)
    x = extractor(img[np.newaxis]).numpy().squeeze()


    cidx = np.argmax([n['grain_type'] == 0 for n in g.nodelist])
    sizes = g.metadata['grain_sizes'][[0,-1], cidx]
    gr = sizes[-1]/sizes[0]
    y = int(gr >= 10.)

    features = {'path': str(g.metadata['path']),
                'x': x,
               'y': y}    
    return features


class Dataset():
    def __init__(self, processed_dir, preprocessed_dir=None, root=None, artifact_path=None):
        self.processed_dir = Path(processed_dir)
        self.preprocessed_dir = Path(preprocessed_dir) if preprocessed_dir is not None else preprocessed_dir
        self.root = Path(root) if root is not None else root
        self.extractor = None # feature extraction model
        self.artifact_path = Path(artifact_path) if artifact_path is not None else artifact_path
        
    @property
    def processed_file_names(self):
        return sorted(Path(self.processed_dir).glob('*.pt'))
    
    def process(self, crop=False):
        os.makedirs(self.processed_dir, exist_ok=True)
        if self.preprocessed_dir is not None:
            os.makedirs(self.preprocessed_dir, exist_ok=True)
        path = Path(self.root)
        for i, file in enumerate(sorted(path.glob('*.json'))):
            g = Graph.from_json(file)
            g.metadata['path'] = file
            d = featurize(g, preprocess_path=Path(self.preprocessed_dir,
                          f'{i}.png'), extractor=self.extractor, crop=crop)
            np.savez(Path(self.processed_dir, f'{i}.npz'), **d)
        
    def __len__(self):
        return len(tuple(self.processed_dir.glob('*.npz')))
    
    def get(self, idx):
        data = np.load(Path(self.processed_dir, f'{idx}.npz'))
        return data
    
    def make_extractor(self, savepath=None, save=True):
        if savepath is None:
            savepath = Path(self.artifact_path)
        from tensorflow.keras.applications.vgg16 import VGG16
        from tensorflow.keras.models import Model
        vgg16 = VGG16()
        extractor = Model(inputs=vgg16.inputs, outputs=vgg16.get_layer('fc1').output)
        extractor.compile('rmsprop','categorical_crossentropy',['accuracy']) # since we are not training the model, this is not
        # needed, however, compiling prevents warnings from
        # showing up
        self.extractor = extractor
        os.makedirs(Path(savepath).parent, exist_ok=True)
        if save is not None:
            extractor.save(savepath)
    
    def load_extractor(self, path=None, force=False):
        if path==None:
            path = self.artifact_path
            
        if self.extractor is None or force:
            from tensorflow.keras.models import load_model
            self.extractor = load_model(path)
    
    def __repr__(self):
        return "Dataset ({} items)".format(len(self))


def plot_c_search(cvals, tr_accs, va_accs, plot=False, return_fig=True):
    """
    Plot SVM training curve (training/validation acc vs svm hyperparameter c)
    
    Parameters
    ----------
    """
    with mpl.rc_context({'xtick.labelsize':8,'ytick.labelsize':8}): # move this to a plotting function
        fig, ax = plt.subplots(figsize=(4,2), dpi=300, facecolor='w')
        ax.plot(cvals, tr_accs, '--k', label='training')
        ax.plot(cvals, va_accs, '-b', label='validation')
        ax.set_xscale('log')
        ax.set_xlabel('regularization C', fontsize=8)
        ax.set_ylabel('accuracy', fontsize=8)
        ax.legend(fontsize=8, handlelength=1.75)
        fig.tight_layout()
        if plot:
            plt.show()
        if return_fig:
            return fig
