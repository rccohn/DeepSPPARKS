import torch
from torch.nn import MSELoss

# from sklearn.decomposition import IncrementalPCA
# from pathlib import Path
# from data import Dataset
import numpy as np


# fit incremental pca to dataset
# pca is used as a baseline -> if an autoencoder
# requires more features to represent the data
# with the same amount of loss, then it is better to simply use pca instead

#  fit pca with all components, and generate scree plot
#  with % variance observed
# can representation loss for number of components be determined as well?
# if not feasable for all components, do a range of components (train and val loss)

# goal: match auto encoder.predict() so that functions
# take in tensor, convert to numpy, apply prediction,
# convert back to torch, return
class PCAEncoder:
    """
    Wrapper for PCA model for compressing images
        Can encode 2d torch tensors into 1d tensors. Can apply transformation
        with a variable number of components, avoiding the need to re-fit the
        sklearn model to reduce the smaller number of components.
        Can decode 1d compressed tensors back into 2d images
    """

    def __init__(
        self,
        pca_model,
        channels=None,
        im_width=None,
        im_height=None,
    ):
        """
        Parameters
        ----------
        pca_model: already fitted sklearn PCA model
        model used as backend
        channels, im_width, im_height: int
            image dimensions: number of color channels, image width, and image height
        """

        self.pca_model = pca_model
        self.channels = channels
        self.im_width = im_width
        self.im_height = im_height

    def encode(self, X, n_components=-1):
        """
        Encodes tensor of images into representation with reduced dimension.

        Automatically reads image dimensions if they are not yet set.

        Parameters
        ----------
        X: tensor
            n_sample x n_channel x im_width x im_width -> array of images

        n_components: int
            number of components to apply when encoding. -1 uses all components.

        Returns
        ---------
        X_encode: tensor
            n_sample x n_component reduced dimensionality
        """
        if n_components == -1:
            n_components = self.pca_model.n_components

        X = X.detach().numpy()

        if self.channels is None:
            self.channels, self.im_height, self.im_width = X.shape[1:]
        else:
            assert (self.channels, self.im_height, self.im_width) == X.shape[
                1:
            ], "image dimensions do not match fitted model"

        X = X.reshape((len(X), -1)) - self.pca_model.mean_

        X_encode = np.dot(X, (self.pca_model.components_[:n_components].T))
        if self.pca_model.whiten:
            X_encode /= np.sqrt(self.pca_model.explained_variance_[:n_components])

        X_encode = torch.DoubleTensor(X_encode)

        return X_encode

    def decode(self, X_encode):
        """
        Decodes tensor of PCA-encodings back into an image

        Parameters
        ----------
        X_encode: tensor
            n_sample x n_component tensor of encoded features

        Returns
        --------
        X_decode: tensor
            n_sample x n_channel x im_width x im_height tensor of decoded images
        """
        n_components = len(X_encode[0])
        X_encode = X_encode.detach().numpy()

        if self.pca_model.whiten:
            X_decode = (
                np.dot(
                    X_encode,
                    np.sqrt(
                        self.pca_model.explained_variance_[:n_components, np.newaxis]
                    )
                    * self.pca_model.components_[:n_components],
                )
                + self.pca_model.mean_
            )
        else:
            X_decode = (
                np.dot(X_encode, self.pca_model.components_[:n_components])
                + self.pca_model.mean_
            )
        X_decode = X_decode.reshape(
            len(X_decode), self.channels, self.im_height, self.im_width
        )
        X_decode = torch.DoubleTensor(X_decode)

        return X_decode

    def predict(self, X, n_components):
        """
        Encodes and decodes a collection of images.
        Used for loss computation

        Parameters
        ----------
        X: tensor
            n_sample x n_channel x im_height x im_width tensor
            of images to generate predictions for

        Returns
        -----------
        y_pred: tensor
            tensor with dimensions matching X of reconstructed
            images
        n_components: int
            number of components to use during encoding step
        """

        # note that this is not the most computationally
        # efficient, as vectors are cast back and forth
        # between torch and numpy between the encoding and
        # decoding steps. This can be changed later if needed,
        # but would require re-writing the code for each
        # method without the extra casting steps
        X_enc = self.encode(X, n_components)
        y_pred = self.decode(X_enc)
        return y_pred


def mse_loss(model, dataloader):
    """
    Compute mse loss for a given dataset.

    Parameters
    ----------
    model: object with predict() method
    """

    mean_loss = 0.0
    samples = 0.0

    # efficient running mean does not rely on total sum / n
    # making it more accurace for larger datasets
    for batch in dataloader:
        yp = model.predict(batch)
        loss = MSELoss(batch, yp)
        n = len(batch)

        mean_loss = (samples / samples + n) * mean_loss + loss / (samples + n)
        samples += n

    return mean_loss
