import math
from dataclasses import dataclass
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from s4 import FFTConv
"""

This file closely follows the mamba.py file from https://github.com/alxndrTL/mamba.py, which in turn follow 
the official Mamba implementation, and the mamba-minimal by @johnma2006.
The major differences are :
- We add implementation for complex parametrization of the SSM (S6-Complex, S4D-Complex) 
- We add an option to use S4D inside the MambaBlock instead of S6 (S4D-Real, S4D-Complex)
"""


@dataclass
class MambaConfig:
    ssm_type: str
    n_layers: int
    d_model: int  # D
    bidirectional: bool
    d_state: int = 16  # N in paper/comments
    d_conv: int = 4
    expand_factor: int = 2  # E in paper/comments

    dt_is_selective: bool = True
    B_is_selective: bool = True
    C_is_selective: bool = True
    param_A_imag: str = "normal"
    deterministic: bool = False
    pscan: bool = False
    initA_imag: str = "S4"
    initA_real: str = "S6"
    S4_init: str = None
    discretizationA: str = "normal"
    discretizationB: str = "s6"
    channel_sharing: bool = True
    dt_rank: Union[int, str] = 'auto'
    A_imag_using_weight_decay: str = True

    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random"  # "random" or "constant"
    dt_scale: float = 1.0
    dt_init_floor = 1e-4

    bias: bool = False
    conv_bias: bool = True  # use parallel scan mode or sequential mode when training
    use_cuda: bool = True

    def __post_init__(self):
        self.d_inner = int(self.expand_factor * self.d_model)  # E*D = ED in comments

        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)

        if self.ssm_type == "S6-Complex":
            self.d_state = self.d_state // 2  # apples to apples comparison with S6-Real w.r.t number of parameters

        if "S4" in self.ssm_type:
            self.channel_sharing = False
            if self.S4_init is None:
                if self.ssm_type == "S4D-Real":
                    self.S4_init = "diag-real"
                elif self.ssm_type == "S4D-Complex":
                    self.S4_init = "diag-lin"
                else:
                    raise NotImplementedError


class Mamba(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()

        self.config = config

        self.layers = nn.ModuleList([ResidualBlock(config) for _ in range(config.n_layers)])
        # self.norm_f = RMSNorm(config.d_model)

    def forward(self, x):
        # x : (B, L, D)

        # y : (B, L, D)

        for layer in self.layers:
            x = layer(x)

        # x = self.norm_f(x)

        return x

    def step(self, x, caches):
        # x : (B, L, D)
        # caches : [cache(layer) for all layers], cache : (h, inputs)

        # y : (B, L, D)
        # caches : [cache(layer) for all layers], cache : (h, inputs)

        for i, layer in enumerate(self.layers):
            x, caches[i] = layer.step(x, caches[i])

        return x, caches


class ResidualBlock(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()

        self.mixer = MambaBlock(config)
        self.norm = RMSNorm(config.d_model)
        # for param in self.norm.parameters():
        #     param.requires_grad = False

    def forward(self, x):
        # x : (B, L, D)

        # output : (B, L, D)

        output = self.mixer(self.norm(x)) + x
        return output

    def step(self, x, cache):
        # x : (B, D)
        # cache : (h, inputs)
        # h : (B, ED, N)
        # inputs: (B, ED, d_conv-1)

        # output : (B, D)
        # cache : (h, inputs)

        output, cache = self.mixer.step(self.norm(x), cache)
        output = output + x
        return output, cache


class MambaBlock(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()

        self.config = config

        if config.deterministic:
            torch.manual_seed(0)

        # projects block input from D to 2*ED (two branches)
        self.in_proj = nn.Linear(config.d_model, 2 * config.d_inner, bias=config.bias)

        self.conv1d = nn.Conv1d(in_channels=config.d_inner, out_channels=config.d_inner,
                                kernel_size=config.d_conv, bias=config.conv_bias,
                                groups=config.d_inner,
                                padding=config.d_conv - 1,)

        # projects block output from ED back to D
        self.out_proj = nn.Linear(config.d_inner, config.d_model, bias=config.bias)

        if config.ssm_type == "S6-Real":
            if not config.channel_sharing:
                self.BC_dims = config.d_state * config.d_inner
            elif config.channel_sharing:
                self.BC_dims = config.d_state
            else:
                raise NotImplementedError

            if config.B_is_selective:
                self.x_proj_B = nn.Linear(config.d_inner, config.d_state, config.bias)
            else:
                self.x_proj_B = nn.Linear(config.d_inner, config.d_state * config.d_inner, True)
                self.x_proj_B.weight.data.fill_(0)  # Initialize weights to 0
                self.x_proj_B.weight.requires_grad = False
                self.x_proj_B.bias.requires_grad = True

            if config.C_is_selective:
                self.x_proj_C = nn.Linear(config.d_inner, config.d_state, config.bias)
            else:
                self.x_proj_C = nn.Linear(config.d_inner, config.d_state * config.d_inner, True)
                self.x_proj_C.weight.data.fill_(0)  # Initialize weights to 0
                self.x_proj_C.weight.requires_grad = False
                self.x_proj_C.bias.requires_grad = True

            self.x_proj_dt = nn.Linear(config.d_inner, config.dt_rank, config.bias)

            # projects Δ from dt_rank to d_inner
            self.dt_proj = nn.Linear(config.dt_rank, config.d_inner, bias=True)

            if config.dt_is_selective:
                self.dt_proj = nn.Linear(config.dt_rank, config.d_inner, bias=True, dtype=torch.float)

                # dt initialization
                # dt weights
                dt_init_std = config.dt_rank ** -0.5 * config.dt_scale
                if config.dt_init == "constant":
                    nn.init.constant_(self.dt_proj.weight, dt_init_std)
                elif config.dt_init == "random":
                    nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
                else:
                    raise NotImplementedError

                # dt bias
                dt = torch.exp(
                    torch.rand(config.d_inner) * (math.log(config.dt_max) - math.log(config.dt_min)) + math.log(
                        config.dt_min)
                ).clamp(min=config.dt_init_floor)
                inv_dt = dt + torch.log(
                    -torch.expm1(-dt))  # inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
                with torch.no_grad():
                    self.dt_proj.bias.copy_(inv_dt)
                self.dt_proj.bias._no_reinit = True  # initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
            else:
                inv_dt = torch.rand(config.d_inner) * (math.log(config.dt_max) - math.log(config.dt_min)) + math.log(
                        config.dt_min)
                self.inv_dt = nn.Parameter(inv_dt)

            # S4D real initialization
            A = torch.arange(1, config.d_state + 1, dtype=torch.float32).repeat(config.d_inner, 1)
            self.A_log = nn.Parameter(
                torch.log(A))  # why store A in log ? to keep A < 0 (cf -torch.exp(...)) ? for gradient stability ?
            self.D = nn.Parameter(torch.ones(config.d_inner))

        elif config.ssm_type == "S6-Complex":
            # projects x to input-dependent Δ, B, C
            if not config.channel_sharing:
                self.BC_dims = config.d_state * config.d_inner
            elif config.channel_sharing:
                self.BC_dims = config.d_state
            else:
                raise NotImplementedError
            # self.x_proj_real = nn.Linear(config.d_inner, config.dt_rank + 2 * self.BC_dims, config.bias)
            # self.x_proj_complex = nn.Linear(config.d_inner, 2 * self.BC_dims, config.bias)
            if config.B_is_selective:
                self.x_proj_real_B = nn.Linear(config.d_inner, config.d_state, config.bias)
                self.x_proj_imag_B = nn.Linear(config.d_inner, config.d_state, config.bias)
            else:
                self.x_proj_real_B = nn.Linear(config.d_inner, config.d_state * config.d_inner, True)
                self.x_proj_imag_B = nn.Linear(config.d_inner, config.d_state * config.d_inner, True)
                self.x_proj_real_B.weight.data.fill_(0)  # Initialize weights to 0
                self.x_proj_imag_B.weight.data.fill_(0)  # Initialize weights to 0
                self.x_proj_real_B.weight.requires_grad = False
                self.x_proj_imag_B.weight.requires_grad = False
                self.x_proj_real_B.bias.requires_grad = True
                self.x_proj_imag_B.bias.requires_grad = True

            if config.C_is_selective:
                self.x_proj_real_C = nn.Linear(config.d_inner, config.d_state, config.bias)
                self.x_proj_imag_C = nn.Linear(config.d_inner, config.d_state, config.bias)
            else:
                self.x_proj_real_C = nn.Linear(config.d_inner, config.d_state * config.d_inner, True)
                self.x_proj_imag_C = nn.Linear(config.d_inner, config.d_state * config.d_inner, True)
                self.x_proj_real_C.weight.data.fill_(0)  # Initialize weights to 0
                self.x_proj_imag_C.weight.data.fill_(0)  # Initialize weights to 0
                self.x_proj_real_C.weight.requires_grad = False
                self.x_proj_imag_C.weight.requires_grad = False
                self.x_proj_real_C.bias.requires_grad = True
                self.x_proj_imag_C.bias.requires_grad = True
            self.x_proj_dt = nn.Linear(config.d_inner, config.dt_rank, config.bias)

            # # init the imaginary part of x_proj to 0
            # self.x_proj.weight.imag.data.zero_()

            # projects Δ from dt_rank to d_inner
            if config.dt_is_selective:
                self.dt_proj = nn.Linear(config.dt_rank, config.d_inner, bias=True, dtype=torch.float)

                # dt initialization
                # dt weights
                dt_init_std = config.dt_rank ** -0.5 * config.dt_scale
                if config.dt_init == "constant":
                    nn.init.constant_(self.dt_proj.weight, dt_init_std)
                elif config.dt_init == "random":
                    nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
                else:
                    raise NotImplementedError

                # dt bias
                dt = torch.exp(
                    torch.rand(config.d_inner) * (math.log(config.dt_max) - math.log(config.dt_min)) + math.log(
                        config.dt_min)
                ).clamp(min=config.dt_init_floor)
                inv_dt = dt + torch.log(
                    -torch.expm1(-dt))  # inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
                with torch.no_grad():
                    self.dt_proj.bias.copy_(inv_dt)
                self.dt_proj.bias._no_reinit = True  # initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
            else:
                inv_dt = torch.rand(config.d_inner) * (math.log(config.dt_max) - math.log(config.dt_min)) + math.log(
                        config.dt_min)
                self.inv_dt = nn.Parameter(inv_dt)


            if config.initA_real == "S4":
                log_A_real = torch.log(0.5 * torch.ones(config.d_inner, config.d_state))
            elif config.initA_real == "S6":
                A = torch.arange(1, config.d_state + 1, dtype=torch.float32).repeat(config.d_inner, 1)
                log_A_real = torch.log(A)
            else:
                raise NotImplementedError

            if config.initA_imag == "uniform":
                A_imag = 2*math.pi * torch.linspace(0, 1, config.d_state).repeat(config.d_inner, 1)
            elif config.initA_imag == "uniform_small":
                A_imag = 2 * math.pi * torch.linspace(0, 0.01, config.d_state).repeat(config.d_inner, 1)
            elif config.initA_imag == "zero":
                A_imag = 0 * torch.linspace(0, 0.01, config.d_state).repeat(config.d_inner, 1)
            elif config.initA_imag == "rand":
                A_imag = (torch.rand(log_A_real.shape) * 2 * torch.pi) - torch.pi
            elif config.initA_imag == "rand_small":
                A_imag = ((torch.rand(log_A_real.shape) * 2 * torch.pi) - torch.pi)/10
            elif config.initA_imag == "S4":
                A_imag = math.pi * torch.arange(config.d_state).repeat(config.d_inner, 1)
            else:
                raise NotImplementedError

            self.log_A_real = nn.Parameter(log_A_real)
            self.log_A_real._no_weight_decay = True
            self.A_imag = nn.Parameter(A_imag)
            if config.param_A_imag == "fixed":
                raise
                self.A_imag.requires_grad = False
            elif config.param_A_imag == "normal":
                pass
            else:
                raise NotImplementedError

            if config.A_imag_using_weight_decay:
                pass
            else:
                self.A_imag._no_weight_decay = True

            # D does not need to be complex since it is multiplied by x, and we take real part of the output
            self.D = nn.Parameter(torch.randn(config.d_inner))

        elif config.ssm_type == "S4D-Complex":
            self.ssm_kernel = FFTConv(d_model=config.d_inner,
                                      d_state=config.d_state,
                                      activation='id',
                                      transposed=False,
                                      mode='s4d',
                                      is_real=False,
                                      shared=config.channel_sharing,
                                      init=config.S4_init,
                                      deterministic = self.config.deterministic,
                                      bidirectional=self.config.bidirectional)
        elif config.ssm_type == "S4D-Real":
            self.ssm_kernel = FFTConv(d_model=config.d_inner,
                                      d_state=config.d_state,
                                      activation='id',
                                      transposed=False,
                                      mode='s4d',
                                      is_real=True,
                                      init=config.S4_init,
                                      shared=config.channel_sharing,
                                      deterministic = self.config.deterministic,
                                      bidirectional=self.config.bidirectional
                                      )
        elif config.ssm_type == "conv":
            inner_conv_state = config.d_state*3
            self.ssm_kernel = nn.Conv1d(in_channels=config.d_inner, out_channels=config.d_inner,
                                        kernel_size=inner_conv_state, bias=False,
                                        groups=config.d_inner,
                                        padding=inner_conv_state - 1)
        else:
            print("type", config.ssm_type)
            raise NotImplementedError

        if self.config.use_cuda:
            try:
                from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
                self.selective_scan_cuda = selective_scan_fn
            except ImportError:
                print("Failed to import mamba_ssm. Falling back to mamba.py.")
                raise NotImplementedError
        elif self.config.pscan:
            try:
                from simple_mamba.pscan import selective_scan
                self.selective_scan = selective_scan
            except ImportError:
                print("Failed to import pscan. Falling back to sequential scan.")

    def forward(self, x):
        # x : (B, L, D)

        # y : (B, L, D)

        _, L, _ = x.shape

        xz = self.in_proj(x)  # (B, L, 2*ED)
        x, z = xz.chunk(2, dim=-1)  # (B, L, ED), (B, L, ED)

        # x branch
        x = x.transpose(1, 2)  # (B, ED, L)
        x = self.conv1d(x)[:, :, :L]  # depthwise convolution over time, with a short filter
        x = x.transpose(1, 2)  # (B, L, ED)

        x = F.silu(x)

        y = self.ssm(x, z)

        if self.config.use_cuda:
            output = self.out_proj(y) # (B, L, D)
            return output

        # z branch
        z = F.silu(z)
        output = y * z
        output = self.out_proj(output)  # (B, L, D)

        return output

    def ssm(self, x, z):
        # x : (B, L, ED)

        # y : (B, L, ED)
        if self.config.ssm_type == "S6-Real":
            A = -torch.exp(self.A_log.float())  # (ED, N)
            D = self.D
            # TODO remove .float()

            delta = self.x_proj_dt(x)
            B = self.x_proj_B(x)
            C = self.x_proj_C(x)

            b, l, ed = x.shape
            if not self.config.B_is_selective:
                B = B.reshape(b, l, ed, self.config.d_state)
            if not self.config.C_is_selective:
                C = C.reshape(b, l, ed, self.config.d_state)

            if self.config.dt_is_selective:
                delta = self.dt_proj.weight @ delta.transpose(1, 2)  # (ED, dt_rank) @ (B, L, dt_rank) -> (B, ED, L)
            else:
                # delta = torch.zeros(delta.shape) + torch.exp(self.inv_dt)
                delta_new = torch.exp(self.inv_dt)
                delta = torch.zeros([B.shape[0], B.shape[1], A.shape[0]],
                                    device=A.device) + delta_new  # (B, L, ED)
                delta = delta.transpose(1, 2)


            if self.config.use_cuda:
                # these are unfortunately needed for the selective_scan_cuda function
                x = x.transpose(1, 2)
                B = B.transpose(1, 2)
                C = C.transpose(1, 2)
                z = z.transpose(1, 2)
                if self.config.dt_is_selective:
                    # "softplus" + "bias" + "y * silu(z)" operations are fused
                    y = self.selective_scan_cuda(x, delta, A, B, C, D, z=z, delta_softplus=True,
                                                delta_bias=self.dt_proj.bias.float())
                else:
                    y = self.selective_scan_cuda(x, delta, A, B, C, D, z=z, delta_softplus=False,
                                                 delta_bias=None)
                y = y.transpose(1, 2)  # (B, L, ED)

            else:
                delta = delta.transpose(1, 2)
                if self.config.dt_is_selective:
                    delta = F.softplus(delta + self.dt_proj.bias)
                if self.config.pscan:
                    y = self.selective_scan(x, delta, A, B, C, D,
                                            B_shape='bln' if self.config.B_is_selective else 'bldn',
                                            C_shape='bln' if self.config.C_is_selective else 'bldn')
                else:
                    y = self.selective_scan_seq(x, delta, A, B, C, D)

            return y

        elif self.config.ssm_type == "S6-Complex":
            b, l, ed = x.shape
            A = -torch.exp(self.log_A_real) + 1j * self.A_imag  # (ED, N)
            D = self.D

            if self.config.initA_imag == "zero" and self.config.param_A_imag == "fixed":
                if torch.any(A.imag != 0):
                    print("zeros did learn something on fixed")
                    raise

            delta = self.x_proj_dt(x)
            B_real = self.x_proj_real_B(x)
            C_real = self.x_proj_real_C(x)
            B_imag = self.x_proj_imag_B(x)
            C_imag = self.x_proj_imag_C(x)

            B = B_real + 1j * B_imag
            C = C_real - 1j * C_imag
            b, l, ed = x.shape
            if not self.config.B_is_selective:
                B = B.reshape(b, l, ed, self.config.d_state)
            if not self.config.C_is_selective:
                C = C.reshape(b, l, ed, self.config.d_state)

            if self.config.dt_is_selective:
                delta = self.dt_proj.weight @ delta.transpose(1, 2)  # (ED, dt_rank) @ (B, L, dt_rank) -> (B, ED, L)
            else:
                # delta = torch.zeros(delta.shape) + torch.exp(self.inv_dt)
                delta_new = torch.exp(self.inv_dt)
                delta = torch.zeros([B.shape[0], B.shape[1], A.shape[0]],
                                    device=A.device) + delta_new  # (B, L, ED)
                delta = delta.transpose(1, 2)

            if self.config.use_cuda:
                # these are unfortunately needed for the selective_scan_cuda function
                x = x.transpose(1, 2)
                B = B.transpose(1, 2)
                B = torch.view_as_real(B).reshape(b, self.config.d_state, l*2)
                C = C.transpose(1, 2)
                C = torch.view_as_real(C).reshape(b, self.config.d_state, l*2)
                z = z.transpose(1, 2)
                if self.config.dt_is_selective:
                    # "softplus" + "bias" + "y * silu(z)" operations are fused
                    y = self.selective_scan_cuda(x, delta, A, B, C, D, z=z, delta_softplus=True,
                                                 delta_bias=self.dt_proj.bias.float())
                else:
                    y = self.selective_scan_cuda(x, delta, A, B, C, D, z=z, delta_softplus=False,
                                                 delta_bias=None)
                y = y.transpose(1, 2)  # (B, L, ED)


            else:
                delta = delta.transpose(1, 2)
                if self.config.dt_is_selective:
                    delta = F.softplus(delta + self.dt_proj.bias)
                if self.config.pscan:
                    y = self.selective_scan(x, delta, A, B, C, D,
                                            B_shape='bln' if self.config.B_is_selective else 'bldn',
                                            C_shape='bln' if self.config.C_is_selective else 'bldn')
                else:
                    y = self.selective_scan_seq(x, delta, A, B, C, D)

            return y

        elif self.config.ssm_type == "S4D-Complex" or self.config.ssm_type == "S4D-Real":
            return self.ssm_kernel(x)[0]
        elif self.config.ssm_type == "conv":
            x_bdl = x.transpose(-1, -2)
            L = x_bdl.size(-1)
            out = self.ssm_kernel(x_bdl)[:, :, :L]
            return out.transpose(-1, -2)
        else:
            raise NotImplementedError


# taken straight from https://github.com/johnma2006/mamba-minimal/blob/master/model.py
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output
