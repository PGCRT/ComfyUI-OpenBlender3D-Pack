import comfy.ops
import torch.nn as nn
import torch.nn.functional as F

ops = comfy.ops.disable_weight_init


# Convolutions
class EncBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        norm_kwargs={},
        pad_mode="zeros",
        norm_fn=None,
        activation_fn=nn.SiLU,
        use_conv_shortcut=False,
        bias=True,
        residual=False,
        dtype=None,
        device=None,
        operations=ops,
    ):
        super().__init__()
        self.use_conv_shortcut = use_conv_shortcut
        # norm_fn is supplied by the encoder() factory below as
        # operations.GroupNorm (a curried partial that already has dtype/device
        # baked in). Calling norm_fn(**norm_kwargs) instantiates a fresh
        # GroupNorm per call.
        self.norm1 = norm_fn(**norm_kwargs)
        self.conv1 = operations.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            padding_mode=pad_mode,
            bias=bias,
            dtype=dtype,
            device=device,
        )
        self.norm2 = norm_fn(**norm_kwargs)
        self.conv2 = operations.Conv2d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            padding_mode=pad_mode,
            bias=bias,
            dtype=dtype,
            device=device,
        )
        self.activation_fn = activation_fn()
        if in_channels != out_channels:
            self.shortcut = operations.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                padding=0,
                padding_mode=pad_mode,
                bias=bias,
                dtype=dtype,
                device=device,
            )
        self.residual = residual

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.activation_fn(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.activation_fn(x)
        x = self.conv2(x)
        if self.use_conv_shortcut or residual.shape != x.shape:
            residual = self.shortcut(residual)
        if self.residual:
            return x + residual
        return x


def encoder(in_dim, hidden_dim, kernel_size=1, ks_res=1, num_layers=2, bias=True, num_groups=8, residual=False, dtype=None, device=None, operations=ops):
    # GroupNorm needs to be a callable so EncBlock can re-instantiate it per
    # layer with its own num_groups/num_channels. Wrap operations.GroupNorm
    # in a lambda that forwards the dtype/device closure -- cleaner than
    # constructing partials and avoids accidentally sharing parameters.
    def norm_fn(num_groups, num_channels):
        return operations.GroupNorm(num_groups, num_channels, dtype=dtype, device=device)

    return nn.Sequential(
        operations.Conv2d(
            in_dim,
            hidden_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            padding_mode="reflect",
            bias=bias,
            dtype=dtype,
            device=device,
        ),
        *[
            EncBlock(
                hidden_dim,
                hidden_dim,
                kernel_size=ks_res,
                pad_mode="reflect",
                norm_fn=norm_fn,
                norm_kwargs={"num_groups": num_groups, "num_channels": hidden_dim},
                activation_fn=nn.SiLU,
                use_conv_shortcut=False,
                bias=bias,
                residual=residual,
                dtype=dtype,
                device=device,
                operations=operations,
            )
            for _ in range(num_layers)
        ],
    )
