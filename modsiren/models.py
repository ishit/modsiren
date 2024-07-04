"""Models."""
import numpy as np
import torch as th
import torch.nn as nn


class ReLUMLP(nn.Sequential):
    def __init__(self, input_features=2, output_features=1,
                 hidden_layers=4, hidden_features=256):
        super().__init__()

        if hidden_layers < 1:
            raise ValueError("Model should have at least 1 hidden layer.")

        self.add_module("input", 
                        nn.Sequential(
                            nn.Linear(input_features, hidden_features),
                            nn.ReLU(True)))

        for i in range(hidden_layers-1):
            self.add_module(f"hidden{i:02d}",
                            nn.Sequential(
                                nn.Linear(hidden_features, hidden_features),
                                nn.ReLU(True)))

        self.add_module("output", nn.Linear(hidden_features, output_features))

        # Reset biases
        for n, p in self.named_parameters():
            if 'bias' in n:
                p.data.zero_()


class SineLayer(nn.Sequential):
    """Linear layer followed by a sine activation.

    Args:
        num_in(int): number of input channels.
        num_out(int): number of output channels.
        freq_scaling(float): scale the initial weights if provided.
        init_scale(float or None): if provided, overrides the scale factor for
            the weight initialization.
    """

    class _Activation(nn.Module):
        def __init__(self, scale):
            super().__init__()
            self.scale = scale

        def forward(self, x):
            return th.sin(x*self.scale)

    def __init__(self, num_in, num_out, freq_scaling=30.0, init_scale=None):
        super().__init__()

        self.add_module("linear", nn.Linear(num_in, num_out))
        self.add_module("activation", SineLayer._Activation(freq_scaling))

        # Reset weights
        if init_scale is None:
            init_scale = np.sqrt(6.0 / num_in) / freq_scaling
        self.linear.weight.data.uniform_(-init_scale, init_scale)

        # TODO(mgharbi): should we randomize the phase?
        self.linear.bias.data.zero_()


class Siren(nn.Sequential):
    def __init__(self, input_features=2, output_features=1,
                 hidden_layers=4, hidden_features=256,
                 freq_scaling=30.0):
        super().__init__()

        if hidden_layers < 1:
            raise ValueError("Model should have at least 1 hidden layer.")

        self.add_module(
            "input",
            SineLayer(input_features, hidden_features,
                      init_scale=1.0/input_features,
                      freq_scaling=freq_scaling))

        for i in range(hidden_layers-1):
            self.add_module(
                f"hidden{i:02d}",
                SineLayer(hidden_features, hidden_features,
                          freq_scaling=freq_scaling))

        self.add_module("output", nn.Linear(hidden_features, output_features))


class GlobalModel(nn.Module):
    def __init__(self, num_cells):
        super().__init__()
        self.synthesizer = None

    def forward(self, x):
        gcoords = x['global_coords']
        bs, c, h, w = gcoords.shape
        # Make it a batch
        gcoords = gcoords.permute(0, 2, 3, 1).reshape(-1, c)
        out = self.synthesizer(gcoords)
        nc = out.shape[1]
        out = out.view(bs, h, w, nc).permute(0, 3, 1, 2).contiguous()
        return out


class GlobalReLUMLP(GlobalModel):
    def __init__(self, num_cells, *args, **kwargs):
        super().__init__(num_cells)
        self.synthesizer = ReLUMLP(*args, **kwargs)


class PositionalEncoding(nn.Module):
    def __init__(self, in_channels, num_frequencies):
        super().__init__()
        self.funcs = [th.sin, th.cos]
        self.freq_bands = th.cat(
            [2**th.arange(0, num_frequencies).float()])*np.pi

    def forward(self, x):
        out = []
        for freq in self.freq_bands:
            for func in self.funcs:
                out.append(func(freq*x))
        out = th.cat(out, -1)
        return out


class FourierFeatures(nn.Module):
    def __init__(self, in_channels, num_frequencies, scale=10.0):
        super().__init__()
        self.register_buffer(
            "weight", 2.0*scale*np.pi*th.randn((in_channels, num_frequencies)))

    def forward(self, x):
        x_proj = th.matmul(x, self.weight)
        return th.cat([th.sin(x_proj), th.cos(x_proj)], dim=-1)


class GlobalPositionalEncoding(GlobalModel):
    def __init__(self, num_cells, input_features=2,
                 output_features=1, hidden_layers=4,
                 hidden_features=256, num_frequencies=16):
        super().__init__(num_cells)
        self.synthesizer = nn.Sequential(
            PositionalEncoding(input_features, num_frequencies),
            ReLUMLP(
                input_features=2*input_features*num_frequencies,
                output_features=output_features,
                hidden_layers=hidden_layers,
                hidden_features=hidden_features))


class GlobalFourierFeatures(GlobalModel):
    def __init__(self, num_cells, input_features=2,
                 output_features=1, hidden_layers=4,
                 hidden_features=256, num_frequencies=16,
                 scale=10.0):
        super().__init__(num_cells)
        self.synthesizer = nn.Sequential(
            FourierFeatures(input_features, num_frequencies, scale=scale),
            ReLUMLP(
                input_features=2*num_frequencies,
                output_features=output_features,
                hidden_layers=hidden_layers,
                hidden_features=hidden_features))


class GlobalSiren(GlobalModel):
    def __init__(self, num_cells, *args, **kwargs):
        super().__init__(num_cells)
        self.synthesizer = Siren(*args, **kwargs)


class LocalModel(nn.Module):
    def __init__(self, num_cells, z_dim=64):
        super().__init__()
        self.synthesizer = None
        self.latent_codes = th.nn.Parameter(
            th.randn(num_cells, z_dim)
        )

    def get_coords_and_latent(self, x):
        lcoords = x['local_coords']
        bs, c, h, w = lcoords.shape

        latent = self.latent_codes[x['idx']].unsqueeze(-1).unsqueeze(-1)
        zdim = latent.shape[1]
        latent = latent.repeat(1, 1, h, w)

        lcoords = lcoords.permute(0, 2, 3, 1).reshape(-1, c)
        latent = latent.permute(0, 2, 3, 1).reshape(-1, zdim)

        return lcoords, latent

    def forward(self, x):
        lcoords, latent = self.get_coords_and_latent(x)
        out = self.synthesizer(th.cat([lcoords, latent], 1))
        nc = out.shape[1]
        bs, c, h, w = x["local_coords"].shape
        out = out.view(bs, h, w, nc).permute(0, 3, 1, 2).contiguous()
        return out


class LocalReLUMLP(LocalModel):
    def __init__(self, num_cells, input_features=2, output_features=1,
                 hidden_layers=4, hidden_features=256,
                 z_dim=64):
        super().__init__(num_cells, z_dim)

        self.synthesizer = ReLUMLP(
            input_features=input_features + z_dim, 
            output_features=output_features,
            hidden_layers=hidden_layers,
            hidden_features=hidden_features)


class LocalSiren(LocalModel):
    def __init__(self, num_cells, input_features=2, output_features=1,
                 hidden_layers=4, hidden_features=256,
                 z_dim=64):
        super().__init__(num_cells, z_dim)

        self.synthesizer = Siren(
            input_features=input_features + z_dim, 
            output_features=output_features,
            hidden_layers=hidden_layers,
            hidden_features=hidden_features)


class ModSiren(LocalModel):
    """Our model."""

    def __init__(self, num_cells, input_features=2, output_features=1,
                 hidden_layers=4, hidden_features=256,
                 freq_scaling=30.0,
                 z_dim=64):
        super().__init__(num_cells, z_dim)

        self.modulator = ReLUMLP(
            input_features=z_dim, 
            output_features=output_features,
            hidden_layers=hidden_layers,
            hidden_features=hidden_features)

        self.synthesizer = Siren(
            input_features=input_features, 
            output_features=output_features,
            hidden_layers=hidden_layers,
            hidden_features=hidden_features,
            freq_scaling=freq_scaling)

        self.hidden_layers = hidden_layers

    def forward(self, x):
        lcoords, latent = self.get_coords_and_latent(x)

        h_synth = self.synthesizer.input(lcoords)
        h_mod = self.modulator.input(latent)

        for i in range(self.hidden_layers-1):
            synth_layer = getattr(self.synthesizer, f"hidden{i:02d}")
            mod_layer = getattr(self.modulator, f"hidden{i:02d}")

            h_mod = mod_layer(h_mod)
            h_synth = synth_layer(h_synth) * h_mod

        out = self.synthesizer.output(h_synth)
        nc = out.shape[1]
        bs, c, h, w = x["local_coords"].shape
        out = out.view(bs, h, w, nc).permute(0, 3, 1, 2).contiguous()
        return out


# class Encoder(nn.Module):
#     def __init__(self, channels, latent_dim, image_resolution):
#         super().__init__()
#         self.conv1 = nn.Conv2d(3, 128, 3, 1, 1)
#         self.relu = nn.ReLU(inplace=True)
#         self.cnn = nn.Sequential(
#             nn.Conv2d(128, 128, 3, 2, 1),
#             nn.ReLU(),
#             nn.Conv2d(128, 128, 3, 1, 1),
#             nn.ReLU(),
#             nn.Conv2d(128, 128, 3, 1, 1),
#             nn.ReLU()
#         )
#         self.fc = nn.Linear(128 * ((image_resolution // 2) ** 2), latent_dim)
#
#     def forward_encoder(self, im):
#         P, C, H, W = im.shape
#         model_in = im.view(P, C, H, W)
#         o = self.relu(self.conv1(model_in))
#         o = self.cnn(o)
#         z = self.fc(o.view(o.shape[0], -1))
#         return z
#
#
# class LocalMLP(nn.Module):
#     def __init__(self, in_features, out_features, hidden_layers=4, hidden_features=256, latent_dim=None,
#                  synthesis_activation=None, modulation_activation=None, embedding=None, N_freqs=None):
#         super().__init__()
#
#         if embedding == 'ffn':
#             total_in_features = N_freqs * in_features
#             self.embed = FourierEmbedding(in_features, N_freqs)
#         elif embedding == 'pe':
#             total_in_features = N_freqs * 2 * in_features + in_features
#             self.embed = PositionEmbedding(in_features, N_freqs)
#         else:
#             self.embed = None
#             total_in_features = in_features
#
#         if modulation_activation:
#             if synthesis_activation == 'sine':
#                 self.synthesis_nw = GatedSineMLP(total_in_features, out_features, hidden_layers, hidden_features,
#                                                  outermost_linear=True)
#             else:
#                 self.synthesis_nw = ReLUMLP(
#                     total_in_features, out_features, hidden_layers, hidden_features)
#
#             if modulation_activation == 'relu':
#                 self.modulation_nw = ModulationMLP(
#                     latent_dim + in_features, hidden_features, hidden_layers)
#             else:
#                 print("Modulation sine not implemented yet!")
#                 exit()
#         else:
#             self.modulation_nw = None
#             if synthesis_activation == 'sine':
#                 self.synthesis_nw = SineMLP(
#                     total_in_features + latent_dim, out_features, hidden_layers, hidden_features)
#             else:
#                 self.synthesis_nw = ReLUMLP(
#                     total_in_features + latent_dim, out_features, hidden_layers, hidden_features)
#
#     def forward(self, model_input):
#         latent_vec = model_input['embedding']
#         coords = model_input['coords']
#         if self.embed:
#             coords = self.embed(coords)
#
#         gating_layers = None
#         if self.modulation_nw:
#             gating_layers = self.modulation_nw(
#                 torch.cat([1000 * latent_vec, coords], dim=-1))
#             model_output = self.synthesis_nw(coords, gating_layers)
#
#         else:
#             model_output = self.synthesis_nw(
#                 torch.cat([latent_vec, coords], dim=-1), gating_layers)
#
#         return {'model_in': coords, 'model_out': model_output, 'latent_vec': latent_vec}
