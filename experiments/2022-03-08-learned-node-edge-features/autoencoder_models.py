import torch
import torch.functional as F
import torch.nn as nn
import mlflow


class Conv2d_std(nn.Conv2d):
    """standardize conv2d layers to have kernel size 3
    and padding 1 (maintains size of input)"""

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(Conv2d_std, self).__init__(
            in_channels, out_channels, kernel_size=kernel_size, padding=padding
        )


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


class ConvAutoEncoderSimple(ConvAutoEncoderBase):
    def __init__(self, log=True):
        name = "ConvAutoEncoderSimplev1"
        super(ConvAutoEncoderSimple, self).__init__(name=name, log=log)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU(inplace=True)

        # encoder
        self.features = nn.Sequential(
            Conv2d_std(1, 4),  # 96 x 96 x 4
            self.relu,
            Conv2d_std(4, 4),  # 96 x 96 x 4
            self.relu,
            self.pool,
            Conv2d_std(4, 8),  # 48 x 48 x 8
            self.relu,
            Conv2d_std(8, 8),  # 48 x 48 x 8
            self.relu,
            self.pool,
            Conv2d_std(8, 16),  # 24 x 24 x 16
            self.relu,
            Conv2d_std(16, 16),  # 24 x 24 x 16
            self.relu,
            self.pool,  # 12 x 12 x 16
        )

        self.dense_enc = nn.Sequential(
            nn.Linear(16 * 12 * 12, 2000),
            nn.Dropout(0.5),
            self.relu,
            nn.Linear(2000, 200),
            nn.Dropout(0.5),
            self.relu,
        )

        self.dense_dec = nn.Sequential(
            nn.Linear(200, 16 * 12 * 12),
            nn.Dropout(0.5),
            self.relu,
        )

        self.reconstruction = nn.Sequential(
            nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2),
            self.relu,
            Conv2d_std(8, 8),
            self.relu,
            nn.ConvTranspose2d(8, 4, kernel_size=2, stride=2),
            self.relu,
            Conv2d_std(4, 4),
            self.relu,
            nn.ConvTranspose2d(4, 1, kernel_size=2, stride=2),
            self.relu,
            Conv2d_std(1, 1),
            self.relu,
        )

    def encode(self, x):
        x = self.features(x)
        x = x.reshape(len(x), 16 * 12 * 12)
        x = self.dense_enc(x)
        return x

    def decode(self, x):
        x = self.dense_dec(x)
        x = x.reshape(len(x), 16, 12, 12)
        x = self.reconstruction(x)
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
    def __init__(self, log=True):
        # input shape: (row, col)
        name = "ConvAutoEncoderV1"
        super(ConvAutoEncoderEdgeV1, self).__init__(name=name, log=log)
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
        # sigmoid possibly introduces vanishing gradient --> try relu
        x = nn.ReLU(inplace=True)(self.decoder(x))

        return x
