import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.core.lightning import LightningModule
from collections import OrderedDict


class PositionEmbedding(nn.Module):
    def __init__(self, in_channels, N_freqs=10):
        super(PositionEmbedding, self).__init__()
        self.N_freqs = N_freqs
        self.in_channels = in_channels
        self.funcs = [torch.sin, torch.cos]
        self.out_channels = in_channels*(len(self.funcs)*N_freqs+1)
        self.freq_bands = torch.cat([2**torch.linspace(0, N_freqs-1, N_freqs)])

    def forward(self, x):
        out = [x]

        for freq in self.freq_bands:
            for func in self.funcs:
                out += [func(freq*x)]
        return torch.cat(out, -1)


class FourierEmbedding(LightningModule):
    def __init__(self, in_channels, N_freqs=256, scale=10.):
        super(FourierEmbedding, self).__init__()
        self.B = scale * torch.randn((in_channels, N_freqs)).cuda()

    def forward(self, x):
        x_proj = 2 * np.pi * x.matmul(self.B)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class SineLayer(LightningModule):
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                            1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                            np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))


class SineMLP(LightningModule):
    def __init__(self, in_features, out_features, hidden_layers, hidden_features, outermost_linear=False,
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()

        self.net = []
        self.net.append(SineLayer(in_features, hidden_features,
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features,
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)

            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                             np.sqrt(6 / hidden_features) / hidden_omega_0)

            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features,
                                      is_first=False, omega_0=hidden_omega_0))

        self.net = nn.Sequential(*self.net)

    def forward(self, coords, gating_layers=None):
        output = self.net(coords)
        return output


class GatedSineMLP(LightningModule):
    def __init__(self, in_features, out_features, hidden_layers, hidden_features, outermost_linear=False,
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()

        self.hidden_layers = hidden_layers
        self.first_layer = SineLayer(in_features, hidden_features,
                                     is_first=True, omega_0=first_omega_0)

        for i in range(hidden_layers):
            layer = SineLayer(hidden_features, hidden_features,
                              is_first=False, omega_0=hidden_omega_0)
            setattr(self, f'hidden_layer_{i}', layer)

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)

            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                             np.sqrt(6 / hidden_features) / hidden_omega_0)

            self.final_layer = final_linear
        else:
            self.final_layer = SineLayer(hidden_features, out_features,
                                         is_first=False, omega_0=hidden_omega_0)

    def forward(self, coords, gating_layers):
        out = self.first_layer(coords)
        for i in range(self.hidden_layers):
            out = gating_layers[i] * getattr(self, f'hidden_layer_{i}')(out)
        output = self.final_layer(out)
        return output


class ReLUMLP(LightningModule):
    def __init__(self, in_features, out_features, hidden_layers, hidden_features):
        super().__init__()

        self.first_layer = nn.Sequential(nn.Linear(in_features, hidden_features),
                                         nn.ReLU())
        self.hidden_layers = hidden_layers

        for i in range(hidden_layers):
            layer = nn.Sequential(
                nn.Linear(hidden_features, hidden_features), nn.ReLU())
            setattr(self, f'hidden_layer_{i}', layer)

        self.final_layer = nn.Sequential(
            nn.Linear(hidden_features, out_features))

    def forward(self, x, gating_layers=None):
        out = self.first_layer(x)
        for i in range(self.hidden_layers):
            if gating_layers:
                out = gating_layers[i] * \
                    getattr(self, f'hidden_layer_{i}')(out)
            else:
                out = getattr(self, f'hidden_layer_{i}')(out)

        output = self.final_layer(out)
        return output


class ModulationMLP(LightningModule):
    def __init__(self, in_features, hidden_features, hidden_layers):
        super().__init__()

        self.first_layer = nn.Sequential(nn.Linear(in_features, hidden_features),
                                         nn.ReLU(True))
        self.hidden_layers = hidden_layers  # since there is no final layer
        for i in range(self.hidden_layers):
            layer = nn.Sequential(nn.Linear(hidden_features, hidden_features),
                                  nn.ReLU(True))
            setattr(self, f'layer_{i}', layer)

    def forward(self, coords):
        output = self.first_layer(coords)
        gating_layers = []
        for i in range(self.hidden_layers):
            output = getattr(self, f'layer_{i}')(output)
            gating_layers.append(output)
        return gating_layers

class Conv2dResBlock(LightningModule):
    '''Aadapted from https://github.com/makora9143/pytorch-convcnp/blob/master/convcnp/modules/resblock.py'''

    def __init__(self, in_channel, out_channel=128):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 5, 1, 2),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, 5, 1, 2),
            nn.ReLU()
        )

        self.final_relu = nn.ReLU()

    def forward(self, x):
        shortcut = x
        output = self.convs(x)
        output = self.final_relu(output + shortcut)
        return output

# class ConvImgEncoder(LightningModule):
    # def __init__(self, channel, image_width, latent_dim):
        # super().__init__()

        # self.latent_dim = latent_dim
        # self.conv_theta = nn.Conv2d(channel, 128, 3, 1, 1)
        # self.relu = nn.ReLU(inplace=True)

        # self.cnn = nn.Sequential(
            # nn.Conv2d(128, 256, 3, 1, 1),
            # nn.ReLU(),
            # Conv2dResBlock(256, 256),
            # # Conv2dResBlock(256, 256),
            # Conv2dResBlock(256, 256),
            # nn.Conv2d(256, 128, 3, 1, 1)
        # )

        # self.relu_2 = nn.ReLU(inplace=True)
        # self.fc = nn.Linear(128 * (image_width ** 2), latent_dim)
        # self.image_resolution = image_width

    # def forward(self, model_input):
        # o = self.relu(self.conv_theta(model_input))
        # o = self.cnn(o)
        # o = self.fc(self.relu_2(o).view(o.shape[0], -1))
        # return o

class ConvImgEncoder(LightningModule):
    def __init__(self, channel, image_resolution, latent_dim):
        super().__init__()

        self.latent_dim = latent_dim
        # conv_theta is input convolution
        self.conv_theta = nn.Conv2d(channel, 128, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)

        self.cnn = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(),
            Conv2dResBlock(256, 256),
            Conv2dResBlock(256, 256),
            Conv2dResBlock(256, 256),
            Conv2dResBlock(256, 256),
            # nn.Conv2d(256, latent_dim, 2, 1, 0)
            nn.Conv2d(256, 128, 3, 1, 1)
        )

        self.relu_2 = nn.ReLU(inplace=True)
        self.fc = nn.Linear(128 * (image_resolution ** 2), latent_dim)
        self.image_resolution = image_resolution

    def forward(self, model_input):
        o = self.relu(self.conv_theta(model_input))
        o = self.cnn(o)
        o = self.fc(self.relu_2(o).view(o.shape[0], -1))
        return o

class LocalMLP(LightningModule):
    def __init__(self, in_features, out_features, hidden_layers=4, hidden_features=256, latent_dim=None,
                 synthesis_activation=None, modulation_activation=None, embedding=None, N_freqs=None, 
                 encoder=False, patch_res=32):
        super().__init__()

        if embedding == 'ffn':
            total_in_features = N_freqs * in_features
            self.embed = FourierEmbedding(in_features, N_freqs)
        elif embedding == 'pe':
            total_in_features = N_freqs * 2 * in_features + in_features
            self.embed = PositionEmbedding(in_features, N_freqs)
        else:
            self.embed = None
            total_in_features = in_features

        if modulation_activation:
            if synthesis_activation == 'sine':
                self.synthesis_nw = GatedSineMLP(total_in_features, out_features, hidden_layers, hidden_features,
                                                 outermost_linear=True)
            else:
                self.synthesis_nw = ReLUMLP(
                    total_in_features, out_features, hidden_layers, hidden_features)

            if modulation_activation == 'relu':
                self.modulation_nw = ModulationMLP(
                    latent_dim + in_features, hidden_features, hidden_layers)
            else:
                print("Modulation sine not implemented yet!")
                exit()
        else:
            self.modulation_nw = None
            if synthesis_activation == 'sine':
                self.synthesis_nw = SineMLP(
                    total_in_features + latent_dim, out_features, hidden_layers, hidden_features)
            else:
                self.synthesis_nw = ReLUMLP(
                    total_in_features + latent_dim, out_features, hidden_layers, hidden_features)

        if encoder:
            self.encoder = ConvImgEncoder(out_features, patch_res, latent_dim)
        else:
            self.encoder = None


    def forward(self, model_input):

        coords = model_input['coords']
        BS, PS, D = coords.shape

        if self.encoder:
            latent_vec = self.encoder(model_input['img'].cuda().float()) 
        else:
            latent_vec = model_input['embedding']
        latent_vec = latent_vec.unsqueeze(-2).repeat(1, PS, 1) 

        if self.embed:
            coords = self.embed(coords)

        gating_layers = None
        if self.modulation_nw:
            gating_layers = self.modulation_nw(
                # torch.cat([1000 * latent_vec, coords], dim=-1))
                torch.cat([latent_vec, coords], dim=-1))
            model_output = self.synthesis_nw(coords, gating_layers)

        else:
            model_output = self.synthesis_nw(
                torch.cat([latent_vec, coords], dim=-1), gating_layers)

        return {'model_in': coords, 'model_out': model_output, 'latent_vec': latent_vec}


class GlobalMLP(LightningModule):
    def __init__(self, in_features, out_features, hidden_layers=4, hidden_features=256,
                 synthesis_activation=None, embedding=None, N_freqs=None):
        super().__init__()

        if embedding == 'ffn':
            total_in_features = N_freqs * in_features
            self.embed = FourierEmbedding(in_features, N_freqs)
        elif embedding == 'pe':
            total_in_features = N_freqs * 2 * in_features + in_features
            self.embed = PositionEmbedding(in_features, N_freqs)
        else:
            self.embed = None
            total_in_features = in_features

        if synthesis_activation == 'sine':
            self.synthesis_nw = SineMLP(
                total_in_features, out_features, hidden_layers, hidden_features, outermost_linear=True)
        else:
            self.synthesis_nw = ReLUMLP(
                total_in_features, out_features, hidden_layers, hidden_features)

    def forward(self, model_input):
        coords = model_input['global_coords']
        if self.embed:
            coords = self.embed(coords)
        model_output = self.synthesis_nw(coords, None)
        return {'model_in': coords, 'model_out': model_output, 'latent_vec': 0}

if __name__ == '__main__':
    # Test encoder
    encoder = ConvImgEncoder(3, 16, 128)
    test = torch.rand((64, 3, 16, 16))
    o = encoder(test)
    print(o.shape)
