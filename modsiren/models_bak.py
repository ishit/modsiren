import torch
import numpy as np
import torch.nn as nn
from pytorch_lightning import LightningModule
from torchmeta.modules import (MetaModule, MetaSequential)
from torchmeta.modules.utils import get_subdict
from collections import OrderedDict
from modsiren.resnet_encoder import *
from pytorch_lightning import Trainer, seed_everything

seed_everything(42)


class PositionEmbedding(nn.Module):
    def __init__(self, in_channels, N_freqs=10):
        super(PositionEmbedding, self).__init__()
        self.N_freqs = N_freqs
        self.in_channels = in_channels
        self.funcs = [torch.sin, torch.cos]
        self.out_channels = in_channels*(len(self.funcs)*N_freqs+1)
        # TODO(mgharbi): Pi missing here?
        self.freq_bands = torch.cat([2**torch.linspace(0, N_freqs-1, N_freqs)])

    def forward(self, x):
        out = [x]

        for freq in self.freq_bands:
            for func in self.funcs:
                out += [func(freq*x)]
        return torch.cat(out, -1)


class FourierEmbedding(nn.Module):
    def __init__(self, in_channels, N_freqs=256, scale=10.):
        super(FourierEmbedding, self).__init__()
        self.B = scale * torch.randn((in_channels, N_freqs)).cuda()
        self.B = torch.nn.Parameter(self.B, requires_grad=False)

    def forward(self, x):
        x_proj = 2 * np.pi * x.matmul(self.B)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Sine(LightningModule):
    def __init(self):
        super().__init__()

    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(30 * input)


def sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30,
                              np.sqrt(6 / num_input) / 30)


def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-1 / num_input, 1 / num_input)


def init_weights_normal(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(
                m.weight, a=0.0, nonlinearity='relu', mode='fan_in')


def hyper_weight_init(m, in_features_main_net):
    if hasattr(m, 'weight'):
        nn.init.kaiming_normal_(
            m.weight, a=0.0, nonlinearity='relu', mode='fan_in')
        m.weight.data = m.weight.data / 1.e2

    if hasattr(m, 'bias'):
        with torch.no_grad():
            m.bias.uniform_(-1/in_features_main_net, 1/in_features_main_net)


def hyper_bias_init(m):
    if hasattr(m, 'weight'):
        nn.init.kaiming_normal_(
            m.weight, a=0.0, nonlinearity='relu', mode='fan_in')
        m.weight.data = m.weight.data / 1.e2

    if hasattr(m, 'bias'):
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
        with torch.no_grad():
            m.bias.uniform_(-1/fan_in, 1/fan_in)


class BatchLinear(nn.Linear, MetaModule):
    __doc__ = nn.Linear.__doc__

    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        bias = params.get('bias', None)
        weight = params['weight']

        output = input.matmul(weight.permute(
            *[i for i in range(len(weight.shape) - 2)], -1, -2))
        output += bias.unsqueeze(-2)
        return output


class SineLayer(nn.Module):
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


class SineMLP(nn.Module):
    def __init__(self, in_features, out_features, hidden_layers,
                 hidden_features, outermost_linear=False, first_omega_0=30,
                 hidden_omega_0=30.):
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


class GatedSineMLP(nn.Module):
    def __init__(self, in_features, out_features, hidden_layers, hidden_features, outermost_linear=False,
                 first_omega_0=30., hidden_omega_0=30.):
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


class ReLUMLP(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 hidden_layers,
                 hidden_features):
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


class ModulationMLP(nn.Module):
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
        skip = output
        gating_layers = []
        for i in range(self.hidden_layers):
            output = getattr(self, f'layer_{i}')(output) + skip
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


class ConvImgEncoder(LightningModule):
    # Try vectorized patch
    # Add stride or pooling
    # Limit the number of parameters
    # learn AE and see the limit of the encoder

    def __init__(self, channel, image_resolution, latent_dim):
        super().__init__()

        self.latent_dim = latent_dim
        # conv_theta is input convolution
        self.conv_theta = nn.Conv2d(channel, 128, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)

        self.cnn = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 1),  # 16 x 16
            nn.ReLU(),
            Conv2dResBlock(256, 256),
            nn.Conv2d(256, 256, 3, 2, 1),  # 8 x 8
            nn.ReLU(),
            Conv2dResBlock(256, 256),
            Conv2dResBlock(256, 256),
            nn.Conv2d(256, 128, 3, 2, 1)  # 4 x 4
        )

        self.relu_2 = nn.ReLU(inplace=True)
        self.fc = nn.Linear(128 * ((image_resolution//8) ** 2),
                            latent_dim)  # Ton of parameters
        # self.fc = nn.Linear(128 * (4 ** 2), latent_dim) # Ton of parameters
        self.image_resolution = image_resolution

    def forward(self, model_input):
        o = self.relu(self.conv_theta(model_input))
        o = self.cnn(o)
        o = self.fc(self.relu_2(o).view(o.shape[0], -1))
        return o


class HyperConvImgEncoder(nn.Module):
    def __init__(self, channel, image_resolution):
        super().__init__()

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
            nn.Conv2d(256, 256, 1, 1, 0)
        )

        self.relu_2 = nn.ReLU(inplace=True)
        self.fc = nn.Linear(image_resolution ** 2, 1)
        self.image_resolution = image_resolution

    def forward(self, I):
        o = self.relu(self.conv_theta(I))
        o = self.cnn(o)
        print(o.shape)
        exit()

        o = self.fc(self.relu_2(o).view(o.shape[0], 256, -1)).squeeze(-1)
        return o


class SimpleConvImgEncoder(LightningModule):
    def __init__(self, channel, image_resolution, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim

        self.cnn = nn.Sequential(
            nn.Conv2d(channel, 128, 3, 2, 1),  # 16 x 16
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, 2, 1),  # 8 x 8
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 2, 1),  # 4 x 4
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 2, 1),  # 2 x 2
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 2, 1),  # 1 x 1
            nn.ReLU(),
        )
        self.fc = nn.Linear(256, latent_dim)

    def forward(self, model_input):
        o = self.cnn(model_input)
        o = self.fc(o.view(o.shape[0], -1))
        return o


class Encoder3(LightningModule):
    def __init__(self, in_channels, image_resolution, latent_dim, batch_norm=False):
        super().__init__()
        self.latent_dim = latent_dim

        modules = []
        hidden_dims = [2 ** (i + 5)
                       for i in range(int(np.log2(image_resolution)) - 1)]

        # Build Encoder
        for h_dim in hidden_dims:
            if batch_norm:
                modules.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, out_channels=h_dim,
                                  kernel_size=3, stride=2, padding=1),

                        nn.BatchNorm2d(h_dim),
                        nn.LeakyReLU())
                )
            else:
                modules.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, out_channels=h_dim,
                                  kernel_size=3, stride=2, padding=1),

                        nn.LeakyReLU())
                )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc = nn.Linear(hidden_dims[-1]*4, latent_dim)

    def forward(self, x):
        res = self.encoder(x)
        res = torch.flatten(res, start_dim=1)

        return self.fc(res)


def maxpool(x, dim=-1, keepdim=False):
    out, _ = x.max(dim=dim, keepdim=keepdim)
    return out


class SimplePointnet(nn.Module):
    # taken from occupancy networks

    def __init__(self, c_dim=128, dim=3, hidden_dim=128):
        super().__init__()
        self.c_dim = c_dim

        self.fc_pos = nn.Linear(dim, 2*hidden_dim)
        self.fc_0 = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc_1 = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc_2 = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc_3 = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc_c = nn.Linear(hidden_dim, c_dim)

        self.actvn = nn.ReLU()
        self.pool = maxpool

    def forward(self, p):
        batch_size, T, D = p.size()

        # output size: B x T X F
        net = self.fc_pos(p)
        net = self.fc_0(self.actvn(net))
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.fc_1(self.actvn(net))
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.fc_2(self.actvn(net))
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.fc_3(self.actvn(net))

        # Recude to  B x F
        net = self.pool(net, dim=1)
        c = self.fc_c(self.actvn(net))

        return c


class Decoder3(LightningModule):
    def __init__(self, in_channels, image_resolution, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        modules = []
        hidden_dims = [2 ** (i + 5)
                       for i in range(int(np.log2(image_resolution)) - 1)]
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)
        hidden_dims.reverse()

        self.hidden_dims = hidden_dims
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1],
                               hidden_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=3,
                      kernel_size=3, padding=1),
            nn.Tanh())

    def forward(self, z):
        res = self.decoder_input(z)
        res = res.view(-1, self.hidden_dims[0], 2, 2)
        res = self.decoder(res)
        res = self.final_layer(res)
        return res


class AE(LightningModule):
    def __init__(self, out_features, latent_dim, patch_res):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder3(out_features, patch_res[0], latent_dim)
        self.decoder = Decoder3(out_features, patch_res[0], latent_dim)

    def forward(self, model_input):
        in_img = model_input['img'].cuda().float()
        latent_vec = self.encoder(in_img)
        out = self.decoder(latent_vec)
        model_output = out.permute(0, 2, 3, 1).reshape(
            in_img.shape[0], -1, in_img.shape[1])
        return {'model_in': 0, 'model_out': model_output, 'latent_vec': latent_vec}


# class VideoEncoder(LightningModule):
    # def __init__(self, in_channels, patch_res, latent_dim):
        # super(VideoEncoder, self).__init__()
        # self.latent_dim = latent_dim
        # modules = []
        # # hidden_dims = [2 ** (i + 5)
                       # # for i in range(int(np.log2(patch_res[1])) - 1)]
        # hidden_dims = [512
                       # for i in range(int(np.log2(patch_res[1])) - 1)]

        # # Build Encoder
        # for h_dim in hidden_dims:
            # modules.append(
                # nn.Sequential(
                    # nn.Conv3d(in_channels, out_channels=h_dim,
                              # kernel_size=3, stride=2, padding=1),
                    # nn.LeakyReLU())
            # )
            # in_channels = h_dim

        # self.encoder = nn.Sequential(*modules)
        # self.fc = nn.Linear(hidden_dims[-1]*4, latent_dim)

    # def forward(self, x):
        # res = self.encoder(x)
        # res = torch.flatten(res, start_dim=1)
        # return self.fc(res)

class VideoEncoder(LightningModule):
    def __init__(self, in_channels, patch_res, latent_dim):
        super(VideoEncoder, self).__init__()

        im_latent_dim = 256
        self.encoder = Encoder3(in_channels, patch_res[1], im_latent_dim)
        self.max_pool = nn.MaxPool1d(patch_res[0])

        self.fc1 = nn.Linear(im_latent_dim * patch_res[0], latent_dim)
        self.fc2 = nn.Linear(latent_dim, latent_dim)

        self.fc = nn.Sequential(
                    self.fc1,
                    nn.LeakyReLU(),
                    self.fc2
                )

    def forward(self, x):

        x_2d = x.permute(0, 2, 1, 3, 4)
        BS, F, C, H, W = x_2d.shape
        input_2d = x_2d.reshape(BS * F, C, H, W)
        output_2d = self.encoder(input_2d) # BS * F, latent_dim
        input_3d = output_2d.view(BS, F, -1)
        input_3d = input_3d.reshape(BS, -1)
        # input_3d = input_3d.permute(0, 2, 1)
        # pooled_3d = self.max_pool(input_3d).squeeze(-1)
        # output_3d = self.fc(pooled_3d) 
        output_3d = self.fc(input_3d) 

        return output_3d

class LocalMLP(nn.Module):
    def __init__(self, in_features, out_features, hidden_layers=4, hidden_features=256, latent_dim=None,
                 synthesis_activation=None, modulation_activation=None, concat=True, embedding=None, freq_scale=1., N_freqs=None,
                 encoder=False, encoder_type='simple', patch_res=32):
        super().__init__()

        if embedding == 'ffn':
            total_in_features = N_freqs * 2
            self.embed = FourierEmbedding(
                in_features, N_freqs, scale=freq_scale)
        elif embedding == 'pe':
            total_in_features = N_freqs * 2 * in_features + in_features
            self.embed = PositionEmbedding(in_features, N_freqs)
        else:
            self.embed = None
            total_in_features = in_features

        self.concat = concat

        if modulation_activation:
            if synthesis_activation == 'sine':
                first_omega_0 = 30. / freq_scale
                hidden_omega_0 = 30. / freq_scale
                self.synthesis_nw = GatedSineMLP(total_in_features, out_features, hidden_layers, hidden_features,
                                                 outermost_linear=True, first_omega_0=freq_scale, hidden_omega_0=hidden_omega_0)
            else:
                self.synthesis_nw = ReLUMLP(
                    total_in_features, out_features, hidden_layers, hidden_features)

            if modulation_activation == 'relu':

                if concat:
                    self.modulation_nw = ModulationMLP(
                        latent_dim + total_in_features, hidden_features, hidden_layers)
                else:
                    self.modulation_nw = ModulationMLP(
                        latent_dim, hidden_features, hidden_layers)

            else:
                print("Modulation sine not implemented yet!")
                exit()
        else:
            self.modulation_nw = None
            if synthesis_activation == 'sine':
                self.synthesis_nw = SineMLP(
                    total_in_features + latent_dim, out_features, hidden_layers, hidden_features,
                    outermost_linear=True)
            else:
                self.synthesis_nw = ReLUMLP(
                    total_in_features + latent_dim, out_features, hidden_layers, hidden_features)

        if encoder:
            if len(patch_res) > 2:
                self.encoder = VideoEncoder(
                    out_features, patch_res, latent_dim)
            elif len(patch_res) == 0:
                self.encoder = SimplePointnet(c_dim=latent_dim)
            else:
                if encoder_type == 'simple':
                    self.encoder = Encoder3(
                        out_features, patch_res[0], latent_dim)
                else:
                    self.encoder = ConvImgEncoder(
                        out_features, patch_res[0], latent_dim)
        else:
            self.encoder = None

    def forward(self, model_input):
        coords = model_input['coords'].float()

        if self.encoder:
            latent_vec = self.encoder(model_input['img'].cuda().float())
        else:
            latent_vec = model_input['embedding'].float()

        if len(latent_vec.shape) != len(coords.shape):
            BS, PS, D = coords.shape
            latent_vec = latent_vec.unsqueeze(-2).repeat(1, PS, 1)

        if self.embed:
            coords = self.embed(coords)

        gating_layers = None
        if self.modulation_nw:
            if self.concat:
                gating_layers = self.modulation_nw(
                    torch.cat([latent_vec, coords], dim=-1))
            else:
                gating_layers = self.modulation_nw(latent_vec)
            model_output = self.synthesis_nw(coords, gating_layers)

        else:
            model_output = self.synthesis_nw(
                torch.cat([latent_vec, coords], dim=-1), gating_layers)

        return {'model_in': coords, 'model_out': model_output, 'latent_vec': latent_vec}


class GlobalMLP(nn.Module):
    def __init__(self, in_features=2, out_features=1, hidden_layers=4,
                 hidden_features=256, synthesis_activation=None,
                 embedding=None, N_freqs=None, freq_scale=1.):
        super().__init__()

        if embedding == 'ffn':
            total_in_features = N_freqs * 2
            self.embed = FourierEmbedding(
                in_features, N_freqs, scale=freq_scale)
        elif embedding == 'pe':
            total_in_features = N_freqs * 2 * in_features + in_features
            self.embed = PositionEmbedding(in_features, N_freqs)
        else:
            self.embed = None
            total_in_features = in_features

        if synthesis_activation == 'sine':
            first_omega_0 = 30. / freq_scale
            hidden_omega_0 = 30. / freq_scale
            print(first_omega_0)
            self.synthesis_nw = SineMLP(
                total_in_features, out_features, hidden_layers,
                hidden_features, outermost_linear=True, first_omega_0=first_omega_0, hidden_omega_0=hidden_omega_0)
        else:
            self.synthesis_nw = ReLUMLP(
                total_in_features, out_features, hidden_layers,
                hidden_features)

    def forward(self, model_input):
        coords = model_input['global_coords']
        if self.embed:
            coords = self.embed(coords)
        model_output = self.synthesis_nw(coords, None)
        return {'model_in': coords, 'model_out': model_output, 'latent_vec': 0.}


class FCBlock(MetaModule):
    def __init__(self, in_features, out_features, num_hidden_layers, hidden_features,
                 outermost_linear=True, nonlinearity='relu', weight_init=None):
        super().__init__()

        self.first_layer_init = None
        nls_and_inits = {'sine': (Sine(), sine_init, first_layer_sine_init),
                         'relu': (nn.ReLU(inplace=True), init_weights_normal, None)}

        nl, nl_weight_init, first_layer_init = nls_and_inits[nonlinearity]

        if weight_init is not None:  # Overwrite weight init if passed
            self.weight_init = weight_init
        else:
            self.weight_init = nl_weight_init

        self.net = []
        self.net.append(MetaSequential(
            BatchLinear(in_features, hidden_features), nl
        ))

        for i in range(num_hidden_layers):
            self.net.append(MetaSequential(
                BatchLinear(hidden_features, hidden_features), nl
            ))

        if outermost_linear:
            self.net.append(MetaSequential(
                BatchLinear(hidden_features, out_features)))
        else:
            self.net.append(MetaSequential(
                BatchLinear(hidden_features, out_features), nl
            ))

        self.net = MetaSequential(*self.net)

        if self.weight_init is not None:
            self.net.apply(self.weight_init)

        # Apply special initialization to first layer, if applicable.
        if first_layer_init is not None:
            self.net[0].apply(first_layer_init)

    def forward(self, coords, params=None, **kwargs):
        if params is None:
            params = OrderedDict(self.named_parameters())

        output = self.net(coords, params=get_subdict(params, 'net'))
        return output


class HyperNetwork(LightningModule):
    # TODO: Try with smaller batch-size with 64x64 patches
    def __init__(self, hyper_in_features, hyper_hidden_layers, hyper_hidden_features, hypo_module):
        super().__init__()

        hypo_parameters = hypo_module.meta_named_parameters()

        self.names = []
        self.nets = nn.ModuleList()
        self.param_shapes = []

        for name, param in hypo_parameters:
            self.names.append(name)
            self.param_shapes.append(param.size())

            hn = FCBlock(in_features=hyper_in_features, out_features=int(torch.prod(torch.tensor(param.size()))),
                         num_hidden_layers=hyper_hidden_layers, hidden_features=hyper_hidden_features,
                         outermost_linear=True, nonlinearity='relu')
            self.nets.append(hn)

            if 'weight' in name:
                self.nets[-1].net[-1].apply(
                    lambda m: hyper_weight_init(m, param.size()[-1]))
            elif 'bias' in name:
                self.nets[-1].net[-1].apply(lambda m: hyper_bias_init(m))

    def forward(self, z):
        params = OrderedDict()
        for name, net, param_shape in zip(self.names, self.nets, self.param_shapes):
            batch_param_shape = (-1,) + param_shape
            params[name] = net(z).reshape(batch_param_shape)
        return params


class NeuralProcessImplicit2DHypernet(LightningModule):
    def __init__(self, in_features, out_features, hidden_layers, hidden_features, latent_dim, encoder=True,
                 encoder_type='simple', patch_res=(32, 32)):
        super().__init__()
        self.latent_dim = latent_dim
        self.hypo_net = FCBlock(in_features=in_features, out_features=out_features, num_hidden_layers=hidden_layers, hidden_features=hidden_features,
                                outermost_linear=True, nonlinearity='sine')
        self.hyper_net = HyperNetwork(hyper_in_features=self.latent_dim, hyper_hidden_layers=hidden_layers, hyper_hidden_features=hidden_features,
                                      hypo_module=self.hypo_net)

        if encoder:
            if len(patch_res) > 2:
                self.encoder = VideoEncoder(
                    out_features, patch_res, latent_dim)
            else:
                if encoder_type == 'simple':
                    self.encoder = Encoder3(
                        out_features, patch_res[0], latent_dim)
                else:
                    self.encoder = ConvImgEncoder(
                        out_features, patch_res[0], latent_dim)
        else:
            self.encoder = None

    def forward(self, model_input):

        if self.encoder:
            embedding = self.encoder(model_input['img'].cuda().float())
        else:
            embedding = model_input['embedding']

        hypo_params = self.hyper_net(embedding)
        coords = model_input['coords']
        model_output = self.hypo_net(coords, params=hypo_params)

        return {'model_in': coords, 'model_out': model_output, 'latent_vec': embedding,
                'hypo_params': hypo_params}


if __name__ == '__main__':
    # Test encoder
    # encoder = ConvImgEncoder(3, 32, 128)
    # encoder = VideoEncoder([10, 32, 32], 128)
    encoder = VideoEncoder(in_channels=3, patch_res=[10, 32, 32], latent_dim=128)
    # test = torch.rand((64, 3, 32, 32))
    test = torch.rand((64, 3, 10, 32, 32))
    o = encoder(test)
    breakpoint()

    e = SimplePointnet(c_dim=256)
    # def __init__(self, c_dim=128, dim=3, hidden_dim=128):
    test = torch.rand((64, 1024, 3))
    ret = e.forward(test)
    print(ret.shape)
