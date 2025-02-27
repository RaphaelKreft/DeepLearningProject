import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base_model import BaseModel

"""
CNN structure taken from https://github.com/gist-ailab/SleePyCo

Returns:
    pretrain: flag to only return the last feature layer of the pyramid
    init_weights: initialise weights by constant
    num_scales: values 1-3 how many of the feature pyramid layers are returned

    Function forward calls the whole autoencoder structure,
    function forwards_encoder only calls the encoder structure
"""

class CnnBackbone(BaseModel):

    SUPPORTED_MODES = ["pretrain-hybrid", "pretrain_mp", "pretrain", "train-classifier", "classification", "gen-embeddings"]  # support Contrastive Learning and Masked Prediction

    def __init__(self, mode: str, conf: dict):
        super(CnnBackbone, self).__init__(mode)

        # architecture setup (params for encoder blocks)
        arch_args = [[1, 64, 128, 192, 256],  # in channels
                     [64, 128, 192, 256, 128],  # out channels
                     [2, 2, 3, 3, 3],  # n-layers
                     [None, 5, 5, 5, 4]]  # max-pool size
        self.encoder = Encoder(*arch_args, use_gate=True)
        self.decoder = Decoder(*arch_args, use_gate=True)

        self.fp_dim = 128

        self.projection_head = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
        )

        if conf["init_weights"]:
            self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Depending on mode acts as backbone only, outputting latent space or also has decoder outputting reconstructed image.
        1. if mode == pretrain_mp: return reconstructed sequence/or reconstructed latent (used in train_mp.py)
        2. if mode == pretrain: in contrastive learning, only use normal encoder without masking. Return latent (used in train_crl.py)
        3. all other: assume we want classification - only return latent (so classifier can be used on top) -> here we return feature pyramid
        """
        c3, c4, c5 = self.encoder(x)
        if self.mode == 'pretrain_mp':
            return self.decoder(c5)  # Note that here the mp training happens on latent dim (128,6) while for all other modes we output (128,1) latent
                                       # Mean on eval benchmarks of MP train we take latent trained on (128,6) and AvgPool
        elif self.mode == 'pretrain-hybrid':
            non_normalized_latent = self.projection_head(c5)
            normalized_latent = F.normalize(non_normalized_latent, dim=1)
            return [self.decoder(c5), normalized_latent.squeeze(-1)]
        else:
            # includes all other modes including 'pretrain'. Outputs normalized 128 dim latent
            non_normalized_latent = self.projection_head(c5)
            normalized_latent = F.normalize(non_normalized_latent, dim=1)
            return normalized_latent.squeeze(-1) # remove dummy dimension in (B, 128, 1)


class Decoder(nn.Module):
    def __init__(self, in_channels: list, out_channels: list, n_layers: list, maxpool_size: list, use_gate: bool):
        super(Decoder, self).__init__()
        self.init_layer = DecoderBlock(in_channels=in_channels[0], out_channels=out_channels[0], n_layers=n_layers[0],
                                       maxpool_size=maxpool_size[0], use_gate=use_gate, first=True)
        self.layer1 = DecoderBlock(in_channels=in_channels[1], out_channels=out_channels[1], n_layers=n_layers[1],
                                   maxpool_size=maxpool_size[1], use_gate=use_gate)
        self.layer2 = DecoderBlock(in_channels=in_channels[2], out_channels=out_channels[2], n_layers=n_layers[2],
                                   maxpool_size=maxpool_size[2], use_gate=use_gate)
        self.layer3 = DecoderBlock(in_channels=in_channels[3], out_channels=out_channels[3], n_layers=n_layers[3],
                                   maxpool_size=maxpool_size[3], use_gate=use_gate)
        self.layer4 = DecoderBlock(in_channels=in_channels[4], out_channels=out_channels[4], n_layers=n_layers[4],
                                   maxpool_size=maxpool_size[4], use_gate=use_gate)

    def forward(self, x: torch.Tensor):
        c5 = self.layer4(x)
        c4 = self.layer3(c5)
        c3 = self.layer2(c4)
        c2 = self.layer1(c3)
        c1 = self.init_layer(c2)

        #print(f"Decoder with shapes {c5.shape}, {c4.shape}, {c3.shape}, {c2.shape}, {c1.shape}")

        return c1


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_layers, maxpool_size, use_gate, first=False):
        super(DecoderBlock, self).__init__()
        self.use_gate = use_gate
        self.first = first
        self.transConv = nn.ConvTranspose1d(out_channels, out_channels, maxpool_size,
                                            maxpool_size) if not first else None
        self.layers = self.make_layers(in_channels, out_channels, n_layers)
        self.prelu = nn.PReLU()

    def make_layers(self, out_channels, in_channels, n_layers):
        layers = []
        for i in range(n_layers):
            conv1d = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
            layers += [conv1d, nn.BatchNorm1d(out_channels)]
            if i == n_layers - 1:
                self.gate = ChannelGate(in_channels)
            if i != n_layers - 1:
                layers += [nn.PReLU()]
            in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):

        if not self.first:
            x = self.transConv(x)
        x = self.layers(x)
        if self.use_gate:
            x = self.gate(x)
        return self.prelu(x)


class Encoder(nn.Module):
    def __init__(self, in_channels: list, out_channels: list, n_layers: list, maxpool_size: list, use_gate: bool):
        super(Encoder, self).__init__()
        self.init_layer = EncoderBlock(in_channels=in_channels[0], out_channels=out_channels[0], n_layers=n_layers[0],
                                       maxpool_size=maxpool_size[0], use_gate=use_gate, first=True)
        self.layer1 = EncoderBlock(in_channels=in_channels[1], out_channels=out_channels[1], n_layers=n_layers[1],
                                   maxpool_size=maxpool_size[1], use_gate=use_gate)
        self.layer2 = EncoderBlock(in_channels=in_channels[2], out_channels=out_channels[2], n_layers=n_layers[2],
                                   maxpool_size=maxpool_size[2], use_gate=use_gate)
        self.layer3 = EncoderBlock(in_channels=in_channels[3], out_channels=out_channels[3], n_layers=n_layers[3],
                                   maxpool_size=maxpool_size[3], use_gate=use_gate)
        self.layer4 = EncoderBlock(in_channels=in_channels[4], out_channels=out_channels[4], n_layers=n_layers[4],
                                   maxpool_size=maxpool_size[4], use_gate=use_gate)

    def forward(self, x: torch.Tensor):
        c1 = self.init_layer(x)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        # print(f"Encoder with shapes {c1.shape}, {c2.shape}, {c3.shape}, {c4.shape}, {c5.shape}")

        return c3, c4, c5


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_layers, maxpool_size, use_gate, first=False):
        super(EncoderBlock, self).__init__()
        self.first = first
        self.pool = MaxPool1d(maxpool_size)
        self.layers = self.make_layers(in_channels, out_channels, n_layers)
        self.prelu = nn.PReLU()
        self.use_gate = use_gate

    def make_layers(self, in_channels, out_channels, n_layers):
        layers = []
        for i in range(n_layers):
            conv1d = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
            layers += [conv1d, nn.BatchNorm1d(out_channels)]
            if i == n_layers - 1:
                self.gate = ChannelGate(in_channels)
            if i != n_layers - 1:
                layers += [nn.PReLU()]
            in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        if not self.first:
            x = self.pool(x)
        x = self.layers(x)
        if self.use_gate:
            x = self.gate(x)
        return self.prelu(x)


class MaxPool1d(nn.Module):
    def __init__(self, maxpool_size):
        super(MaxPool1d, self).__init__()
        self.maxpool_size = maxpool_size
        self.maxpool = nn.MaxPool1d(kernel_size=maxpool_size, stride=maxpool_size)

    def forward(self, x):
        _, _, n_samples = x.size()
        if n_samples % self.maxpool_size != 0:
            pad_size = self.maxpool_size - (n_samples % self.maxpool_size)
            if pad_size % 2 != 0:
                left_pad = pad_size // 2
                right_pad = pad_size // 2 + 1
            else:
                left_pad = pad_size // 2
                right_pad = pad_size // 2
            x = F.pad(x, (left_pad, right_pad), mode='constant')

        x = self.maxpool(x)

        return x


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm1d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool1d(x, x.size(2), stride=x.size(2))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool1d(x, x.size(2), stride=x.size(2))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(lp_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid(channel_att_sum).unsqueeze(2).expand_as(x)
        return x * scale


if __name__ == '__main__':
    x0 = torch.randn((10, 1, 3000))
    conf = {
        "name": "CnnOnly",
        "init_weights": False,
        "num_scales": 1
    }
    mode = "pretrain_mp"
    print(f"X shape {x0.shape}")
    m0 = CnnBackbone(mode, conf)
    forw = m0.forward(x0)
    print(f"Out: {type(forw)}, len={len(forw)}")
    print(f"Out-Shape: {forw[0].shape}")