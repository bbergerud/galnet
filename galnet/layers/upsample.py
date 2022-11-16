"""
Methods for upsampling tensors.

Class Methods
-------------
ConvTranspose2d
    Convolutional Transpose operation. Has been redefined from the pytorch
    implementation to apply the correct padding to return the proper spatial
    size. The sequence of operations is
        Pad -> ConvTranspose -> Crop

ConvBilinearUpsample
    Applies a bilinear upsampling operation followed by a convolution
    to increase the spatial dimensions and change the number of channels.
    The sequence of operations is
        Bilinear Upsample -> Pad -> Conv -> BN -> Nonlinear

ConvUpsample
    Applies a transpose convolution to increase the spatial dimensions
    and change the number of channels. The sequence of operations is
        Pad -> ConvTranspose -> BN -> Nonlinear

PreactivationBilinearUpsample
    Applies a bilinear upsampling operation followed by a convolution
    to increase the spatial dimensions and change the number of channels.
    The sequence of operations is
        Bilinear Upsample -> BN -> Nonlinear -> Pad -> Conv

PreactivationUpsample
    Applies a transpose convolution to increase the spatial dimensions
    and change the number of channels. The sequence of operations is
        BN -> Nonlinear -> Pad -> Conv

WrapUpsampling2d
    Performs upsampling, but applies wrap padding to the input first and
    crops the output after the interpolation to restore the desired size.

BilinearUpsample2d
    Bilinear upsampling operation that wraps around pytorch's UpsamplingBilinear2d and
    the implementation WrapUpsampleBilinear2d through the argument `wrap_padding`.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .padding import WrapPad2d
from ..utils import get_padding

class ConvTranspose2d(nn.Module):
    """
    Convolutional Transpose operation. Has been redefined from the pytorch
    implementation to apply the correct padding to return the proper spatial
    size. The sequence of operations is

        Pad -> ConvTranspose -> Crop

    Parameters
    ----------
    in_channels : int
        The number of input channels

    out_channels : int
        The number of output channels

    kernel_size : int
        The size of the kernel.

    stride : int
        The stride factor.

    dilation : int
        The dilation factor.

    wrap_padding : bool
        Boolean indicating whether to apply wrap padding (True) or
        zero padding (False). Default is False.

    bias : bool
        Boolean indicating whether to apply bias. Default is True.

    Methods
    -------
    forward(input, *args, **kwargs)
        Returns the upsampled tensor

    Examples
    ---------
    >>> import torch
    >>> from galnet.layers.upsample import ConvTranspose2d
    >>> upsample = ConvTranspose2d(in_channels=1, out_channels=1, stride=2)
    >>> y = upsample(torch.randn(1,1,16,16))
    >>> print(y.size())
    torch.Size([1, 1, 32, 32])
    """
    def __init__(self,
        in_channels:int,
        out_channels:int,
        kernel_size:int = 4,
        stride:int = 2,
        dilation:int = 1,
        wrap_padding:bool = False,
        bias:bool = True,
    ):
        super().__init__()

        # ======================================================================
        # For the case of pre-padding, we have
        #       H_out = (H_in + 2P - 1) * S - 2*D*(K-1) + D*(K-1) + OutPad + 1 - 2C
        # where C is the crop size, so we solve to find the relation
        #       D(K-1) + S - 1 = 2P*S + OutPad - 2C
        # and set
        #       P = ceil((D(K-1) + S - 1) / (2*S))
        # Then solve for C and OutPad
        # ======================================================================
        left = dilation*(kernel_size-1) + stride - 1
        padding = math.ceil( left / (2 * stride))
        difference = left - 2*padding*stride
        crop = math.ceil(-difference/2)
        output_padding = difference + 2*crop
        self.crop = crop

        # ======================================================================
        # Generate sequence
        # ======================================================================
        Pad = WrapPad2d if wrap_padding else nn.ZeroPad2d
        self.block = nn.Sequential(
            Pad(padding),
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=dilation*(kernel_size-1), # No additional padding
                dilation=dilation,
                output_padding=output_padding,
                bias=bias,
            )
        )

    def forward(self, input:torch.Tensor, *args, **kwargs) -> torch.Tensor:
        output = self.block(input)
        # Check for crop == 0, otherwise a zero-sized spatial array is extracted
        if self.crop != 0:
            output = output[...,self.crop:-self.crop, self.crop:-self.crop]
        return output

class ConvBilinearUpsample(nn.Module):
    """
    Applies a bilinear upsampling operation followed by a convolution
    to increase the spatial dimensions and change the number of channels.
    The sequence of operations is

        Bilinear Upsample -> Pad -> Conv -> BN -> Nonlinear

    Parameters
    ----------
    in_channels : int
        The number of input channels

    out_channels : int
        The number of output channels

    kernel_size : int
        The size of the kernel.

    stride : int
        The stride factor.

    dilation : int
        The dilation factor.

    wrap_padding : bool
        Boolean indicating whether to apply wrap padding (True) or
        zero padding (False). Default is False.

    nonlinear : torch.nn.Module
        A nonlinear operation to apply. 

    Methods
    -------
    forward(input, *args, **kwargs)
        Returns the upsampled tensor

    Examples
    --------
    >>> import torch
    >>> from galnet.layers.upsample import ConvBilinearUpsample
    >>> upsample = ConvBilinearUpsample(in_channels=1, out_channels=1, stride=2)
    >>> y = upsample(torch.randn(1,1,16,16))
    >>> print(y.size())
    torch.Size([1, 1, 32, 32])
    """
    def __init__(self,
        in_channels:int,
        out_channels:int,
        kernel_size:int = 3,
        stride:int = 2,
        dilation:int = 1,
        wrap_padding:bool = False,
        nonlinear:nn.Module = nn.ReLU(),
    ):
        super().__init__()

        pad = (WrapPad2d if wrap_padding else nn.ZeroPad2d)(get_padding(kernel_size, dilation))
        up  = (WrapUpsampling2d if wrap_padding else nn.UpsamplingBilinear2d)(scale_factor=stride)

        self.block = nn.Sequential(
            pad,
            nn.Conv2d(
                in_channels  = in_channels,
                out_channels = out_channels,
                kernel_size  = kernel_size,
                dilation     = dilation,
                bias         = False,
            ),
            nn.BatchNorm2d(out_channels),
            nonlinear,
            up
        )

    def forward(self, input:torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.block(input)

class ConvUpsample(nn.Module):
    """
    Applies a transpose convolution to increase the spatial dimensions
    and change the number of channels. The sequence of operations is

        Pad -> ConvTranspose -> BN -> Nonlinear

    Parameters
    ----------
    in_channels : int
        The number of input channels

    out_channels : int
        The number of output channels

    kernel_size : int
        The size of the kernel.

    stride : int
        The stride factor.

    dilation : int
        The dilation factor.

    wrap_padding : bool
        Boolean indicating whether to apply wrap padding (True) or
        zero padding (False). Default is False.

    nonlinear : torch.nn.Module
        A nonlinear operation to apply. 

    Methods
    -------
    forward(input, *args, **kwargs)
        Returns the upsampled tensor

    Examples
    --------
    >>> import torch
    >>> from galnet.layers.upsample import ConvUpsample
    >>> upsample = ConvUpsample(in_channels=1, out_channels=1, stride=2)
    >>> y = upsample(torch.randn(1,1,16,16))
    >>> print(y.size())
    torch.Size([1, 1, 32, 32])
    """
    def __init__(self,
        in_channels:int,
        out_channels:int,
        kernel_size:int = 3,
        stride:int = 2,
        dilation:int = 1,
        wrap_padding:bool = False,
        nonlinear:nn.Module = nn.ReLU()
    ):
        super().__init__()

        self.block = nn.Sequential(
            ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                wrap_padding=wrap_padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nonlinear
        )

    def forward(self, input:torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.block(input)

class PreactivationBilinearUpsample(nn.Module):
    """
    Applies a bilinear upsampling operation followed by a convolution
    to increase the spatial dimensions and change the number of channels.
    The sequence of operations is

        Bilinear Upsample -> BN -> Nonlinear -> Pad -> Conv

    Parameters
    ----------
    in_channels : int
        The number of input channels

    out_channels : int
        The number of output channels

    kernel_size : int
        The size of the kernel.

    stride : int
        The stride factor.

    dilation : int
        The dilation factor.

    wrap_padding : bool
        Boolean indicating whether to apply wrap padding (True) or
        zero padding (False). Default is False.

    nonlinear : torch.nn.Module
        A nonlinear operation to apply. 

    Methods
    -------
    forward(input, *args, **kwargs)
        Returns the upsampled tensor

    Examples
    --------
    >>> import torch
    >>> from galnet.layers.upsample import PreactivationBilinearUpsample
    >>> upsample = PreactivationBilinearUpsample(in_channels=1, out_channels=1, stride=2)
    >>> y = upsample(torch.randn(1,1,16,16))
    >>> print(y.size())
    torch.Size([1, 1, 32, 32])
    """
    def __init__(self,
        in_channels:int,
        out_channels:int,
        kernel_size:int = 3,
        stride:int = 2,
        dilation:int = 1,
        wrap_padding:bool = False,
        nonlinear:nn.Module = nn.ReLU(),
    ):
        super().__init__()

        pad = (WrapPad2d if wrap_padding else nn.ZeroPad2d)(get_padding(kernel_size, dilation))
        up  = (WrapUpsampling2d if wrap_padding else nn.UpsamplingBilinear2d)(scale_factor=stride)

        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nonlinear,
            pad,
            nn.Conv2d(
                in_channels  = in_channels,
                out_channels = out_channels,
                kernel_size  = kernel_size,
                dilation     = dilation,
                bias         = False,
            ),
            up
        )

    def forward(self, input:torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.block(input)

class PreactivationUpsample(nn.Module):
    """
    Applies a transpose convolution to increase the spatial dimensions
    and change the number of channels. The sequence of operations is

        BN -> Nonlinear -> Pad -> Conv

    Parameters
    ----------
    in_channels : int
        The number of input channels

    out_channels : int
        The number of output channels

    kernel_size : int
        The size of the kernel.

    stride : int
        The stride factor.

    dilation : int
        The dilation factor.

    wrap_padding : bool
        Boolean indicating whether to apply wrap padding (True) or
        zero padding (False). Default is False.

    nonlinear : torch.nn.Module
        A nonlinear operation to apply. 

    Methods
    -------
    forward(input, *args, **kwargs)
        Returns the upsampled tensor

    Examples
    --------
    >>> import torch
    >>> from galnet.layers.upsample import PreactivationUpsample
    >>> upsample = PreactivationUpsample(in_channels=1, out_channels=1, stride=2)
    >>> y = upsample(torch.randn(1,1,16,16))
    >>> print(y.size())
    torch.Size([1, 1, 32, 32])
    """
    def __init__(self,
        in_channels:int,
        out_channels:int,
        kernel_size:int = 4,
        stride:int = 2,
        dilation:int = 1,
        wrap_padding:bool = False,
        nonlinear:nn.Module = nn.ReLU(),
    ):
        super().__init__()

        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nonlinear,
            ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                stride=stride,
                wrap_padding=wrap_padding,
                bias=False
            ),
        )

    def forward(self, input:torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.block(input)

class WrapUpsampling2d(nn.Module):
    """
    Performs upsampling, but applies wrap padding to the input first and
    crops the output after the interpolation to restore the desired size.

    The main issue with directly upsampling is the last azimuthal angle
    will be larger than before, so an interpolation between the last
    azimuthal angle and the first needs to be performed.

    Parameters
    ----------
    scale_factor : int
        The scale factor for upsampling.

    dim : int
        The dimension along which to apply the wrap padding.
        Should be one of either 0 or 1.

    mode : str
        The upsampling mode. Default is bilinear.

    Methods
    -------
    forward(input, *args, **kwargs)
        Returns the upsampled tensor

    Examples
    --------
    >>> import torch
    >>> from galnet.layers.upsample import WrapUpsampling2d
    >>> x = torch.arange(4, dtype=torch.float32).view(1,1,2,2).cos()
    >>> print(x)
    tensor([[[[ 1.0000,  0.5403],
              [-0.4161, -0.9900]]]])
    >>> WrapUpsampling2d(scale_factor=2)(x)
    tensor([[[[ 1.0000,  0.8468,  0.6935,  0.5403],
              [ 0.2919,  0.1197, -0.0526, -0.2248],
              [-0.4161, -0.6074, -0.7987, -0.9900],
              [ 0.2919,  0.1197, -0.0526, -0.2248]]]])
    """
    def __init__(self, scale_factor:int=2, dim:int=0, mode='bilinear'):
        super().__init__()
        self.scale_factor = scale_factor
        self.dim = dim
        self.mode = mode

    def forward(self, input:torch.Tensor, *args, **kwargs) -> torch.Tensor:
        size = input.shape[-2:]
        size = [s*self.scale_factor for s in size]  # size from upsampling
        size  = [(s+1) if i == self.dim else s for i,s in enumerate(size)] # Add extra along wrap padded dimension

        # For wrap padding, the last azimuthal  
        input = F.pad(input, pad=(0,0,0,1) if self.dim == 0 else (0,1,0,0), mode='circular')
        input = F.interpolate(input, size=size, align_corners=True, mode='bilinear')
        return input[...,:-1,:] if self.dim == 0 else input[...,:-1]

class BilinearUpsample2d(nn.Module):
    """
    Bilinear upsampling operation that wraps around pytorch's UpsamplingBilinear2d and
    the implementation WrapUpsampleBilinear2d through the argument `wrap_padding`.

    If the input and output channels are different, then a 1x1 convolution is applied.

    Parameters
    ----------
    in_channels : int
        The number of input channels

    out_channels : int
        The number of output channels

    stride : int
        The stride factor.

    wrap_padding : bool
        Boolean indicating whether to apply wrap padding (True) or
        zero padding (False). Default is False.

    **kwargs
        Collects unused parameters to allow seamless interfacing
        with other upsampling methods.

    Methods
    -------
    forward(input, *args, **kwargs)
        Returns the upsampled tensor

    Examples
    --------
    >>> import torch
    >>> from galnet.layers.upsample import BilinearUpsample2d
    >>> x = torch.arange(4, dtype=torch.float32).view(1,1,2,2).cos()
    >>> print(x)
    tensor([[[[ 1.0000,  0.5403],
              [-0.4161, -0.9900]]]])
    >>> BilinearUpsample2d(1, 1, stride=2, wrap_padding=True)(x)
    tensor([[[[ 1.0000,  0.8468,  0.6935,  0.5403],
              [ 0.2919,  0.1197, -0.0526, -0.2248],
              [-0.4161, -0.6074, -0.7987, -0.9900],
              [ 0.2919,  0.1197, -0.0526, -0.2248]]]])
    >>> BilinearUpsample2d(1, 1, stride=2, wrap_padding=False)(x)
    tensor([[[[ 1.0000,  0.8468,  0.6935,  0.5403],
              [ 0.5280,  0.3620,  0.1961,  0.0302],
              [ 0.0559, -0.1227, -0.3013, -0.4799],
              [-0.4161, -0.6074, -0.7987, -0.9900]]]])
    """
    def __init__(self,
        in_channels:int,
        out_channels:int,
        stride:int=2,
        wrap_padding:bool=False,
        **kwargs
    ):
        super().__init__()
        if in_channels != out_channels:
            self.conv = nn.Conv(in_channels, out_channels, kernel_size=1)

        self.upsample = (WrapUpsampling2d if wrap_padding else nn.UpsamplingBilinear2d)(scale_factor=stride)

    def forward(self, input, *args, **kwargs):
        if 'conv' in self.__dict__['_modules']:
            input = self.conv(input)
        return self.upsample(input)