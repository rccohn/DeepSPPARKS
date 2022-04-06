import torch
import torch.nn as nn
import torch.nn.functional as F

# from sklearn.decomposition import IncrementalPCA
# from pathlib import Path
# from data import Dataset
import numpy as np
import mlflow


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
        Can decode 1d compressed tensors back into 2d images.
    """

    def __init__(
        self,
        pca_model,
        channels=None,
        im_width=None,
        im_height=None,
        n_components=-1,
    ):
        """
        Parameters
        ----------
        pca_model: already fitted sklearn PCA model
        model used as backend
        channels, im_width, im_height: int
            image dimensions: number of color channels, image width, and image height
        n_components: int
            number of components. -1 includes all components.

        """

        self.pca_model = pca_model
        self.channels = channels
        self.im_width = im_width
        self.im_height = im_height
        self.n_components = n_components

    def encode(self, X, n_components=None):
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
        if n_components is None:
            n_components = self.n_components

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

    def predict(self, X, n_components=None):
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
        if n_components is None:
            n_components = self.n_components
        # note that this is not the most computationally
        # efficient, as vectors are cast back and forth
        # between torch and numpy between the encoding and
        # decoding steps. This can be changed later if needed,
        # but would require re-writing the code for each
        # method without the extra casting steps
        X_enc = self.encode(X, n_components)
        y_pred = self.decode(X_enc)
        return y_pred

    def to(self, device):
        """
        Placeholder for torch.Module models that can be moved to cpu/gpu with
        model.to(device). Allows common functions to be used for both models
        """
        # remove pep 8 linter warnings about unused variable "device"
        assert 1 < 2 or device
        return self


class ConvAutoEncoderBase(nn.Module):
    # template for autoencoder models, to reduce amount of repeated code
    # requires encode and decode methods
    def __init__(self, name, log=True):
        super(ConvAutoEncoderBase, self).__init__()
        self.name = name

        if log:
            # log model properties to mlflow tracking server, allows different
            # model versions to be distinguished
            self._log()

    def _log(self):
        # can be over-ridden on subclasses to log more parameters if needed
        mlflow.set_tag("model_name", self.name)

    def encode(self, x):
        # needs to be implemented by subclass
        raise NotImplementedError

    def decode(self, x):
        # needs to be implemented by subclass
        raise NotImplementedError

    def predict(self, x):
        # compress and recover image
        # needed because some evaluation functions
        # require a predict() method
        return self(x)

    def forward(self, x):
        # compress and recover image
        # loss will be distance between original and recovered image
        x = self.encode(x)
        x = self.decode(x)
        return x


class ConvAutoEncoderTest(ConvAutoEncoderBase):
    def __init__(self, log=True):
        name = "ConvAutoEncoderTest"
        super(ConvAutoEncoderTest, self).__init__(name=name)

        # Encoder
        self.conv1 = nn.Conv2d(1, 4, 3, padding=1)
        self.conv2 = nn.Conv2d(4, 4, 3, padding=1)
        self.conv3 = nn.Conv2d(4, 8, 3, padding=1)
        self.conv4 = nn.Conv2d(8, 8, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # Decoder
        self.t_conv1 = nn.ConvTranspose2d(8, 4, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(4, 1, 2, stride=2)

        # ensure all weights are double, avoiding type errors during training
        self.double()

    def encode(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        return x

    def decode(self, x):
        x = F.relu(self.t_conv1(x))
        x = torch.sigmoid(self.t_conv2(x))
        return x


# idea taken from torchvision/vgg module
# instead of manually entering each layer,
# generate layers from config list
def make_layers_feature(cfg, in_channels=1, batch_norm=False):
    layers = []
    for c in cfg:
        if c == "p":  # pool
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:  # conv layer
            layers += [nn.Conv2d(in_channels, c, kernel_size=3, padding=1)]
            if batch_norm:
                layers += [nn.BatchNorm2d(c)]
            layers += [nn.ReLU(inplace=True)]
            in_channels = c

    return nn.Sequential(*layers).double()


def make_layers_dense(cfg, in_features, batch_norm=False):
    layers = []
    for c in cfg:
        layers += [
            nn.Linear(in_features, c, bias=True),
        ]
        if batch_norm:
            layers += [nn.BatchNorm1d(c)]
        layers += [nn.ReLU(inplace=True)]

    return nn.Sequential(*layers).double()


def make_layers_decoder(cfg, in_channels, batch_norm=False):
    layers = []
    for c in cfg:
        if c == "u":  # up sample
            layers += [nn.ConvTranspose2d(in_channels, in_channels // 2, 2, stride=2)]
            in_channels = in_channels // 2
        else:
            layers += [nn.Conv2d(in_channels, c, kernel_size=3, padding=1)]
            in_channels = c
        if batch_norm:
            layers += [nn.BatchNorm2d()]
        layers += [nn.ReLU(inplace=True)]

    return nn.Sequential(*layers).double()


class ConvAutoEncoderEdgeV1(ConvAutoEncoderBase):
    # for edge patches
    def __init__(
        self,
    ):
        # input shape: (row, col)
        name = "ConvAutoEncoderV1"
        super(ConvAutoEncoderEdgeV1, self).__init__(name=name)
        # TODO look at max vs avg pool
        # cfg 1 probably only useful for edge encoder, node encoder likely needs
        # 1 more pooling + 256 channel features
        # for feature map
        cfg1_f = [16, 16, "p", 32, 32, "p", 64, 64, 64, "p", 128, 128, 128, "p"]
        # linear layers
        # encode to bottleneck
        cfg1_l_a = [9 * 128, 3 * 128]
        # decode
        cfg1_l_b = [9 * 128]
        # decoder
        cfg1_d = [
            "u",
            128,
            128,
            128,
            "u",
            64,
            64,
            64,
            "u",
            32,
            32,
            "u",
            16,
            16,
            1,
            1,
            1,
        ]
        input_shape = (48, 48)
        self.input_shape = input_shape
        self.n = input_shape[0] // (2**4) * input_shape[1] // (2**4) * 128
        self.double()

        self.feat = make_layers_feature(cfg1_f)

        self.l1 = make_layers_dense(cfg1_l_a, self.n)
        self.l2 = make_layers_dense(cfg1_l_b, 3 * 128)
        self.decoder = make_layers_decoder(cfg1_d, 128)

    def encode(self, x):
        x = self.feat(x)
        x = torch.flatten(x, 1)
        x = self.l1(x)
        return x

    def decode(self, x):
        x = self.l2(x)
        x = x.reshape(
            x.shape[0],
            128,
            self.input_shape[0] // 2**4,
            self.input_shape[1] // 2**4,
        )
        x = torch.sigmoid(self.decoder(x))

        return x


def batch_mse_loss(model, dataloader, device="cpu"):
    """
    Compute mse loss for a given dataset.

    Parameters
    ----------
    model: object with predict() method
    dataloader: loads data
    device: device to move model and data to
    """
    # efficient running mean loss does not rely on large sums
    # derived from mu[j]-mu[i] = (sum[x_i]/ + sum[xj])/(n+m) - (sum[xi]/n)
    # set mu[j]-mu[i] = delta, solve for delta, then mu[j] = mu[i] + delta
    model = model.to(device)
    mean_loss = 0.0
    total_samples = 0
    loss_fn = nn.MSELoss(reduction="sum")
    for batch in dataloader:
        batch = batch.to(device)
        yp = model.predict(batch)
        # without detaching, gradients accumulate and memory blows up very
        # quickly. This is a very subtle issue but will cause massive problem.
        loss = float(loss_fn(batch, yp).detach().cpu())
        #
        m = len(batch)
        mean_loss += (loss - (m * mean_loss)) / (total_samples + m)
        total_samples += m

    # average loss over number of pixels
    mean_loss /= np.prod(batch.shape[1:])
    return mean_loss


def get_autoencoder(model_name: str) -> torch.nn.Module:
    """
    Selects model architecture to use for autoencoder experiments.
    New architectures can be added as needed.

    Parameters
    ----------
    model_name: str
        name of model architecture to use.

    Returns
    ---------
    model_class: torch.nn.Module
        class of model. model(*args, **kwargs) should build the model
        with appropriate configurations.
    """
    models = {
        "test": ConvAutoEncoderTest,
        "ConvAutoEncoderEdgeV1": ConvAutoEncoderEdgeV1,
    }
    return models[model_name]


def main():
    net = ConvAutoEncoderEdgeV1()
    x = torch.ones((3, 1, 48, 48), dtype=torch.double)
    test = net(x).shape == x.shape
    print(test)
    assert test


if __name__ == "__main__":
    main()
