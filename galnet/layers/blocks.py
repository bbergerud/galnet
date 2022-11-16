"""
Convolutional blocks with flexible lengths.

Classes
-------
ConvBlock
    Generic convolution block with only one nonlinear operation.
    The sequence of operations is
        N × (Pad -> Conv -> BN) -> Nonlinear

ConvResidualBlock
    Generic residual block with only one nonlinear operation. 
    The sequence of operations is
        N × (Pad -> Conv -> BN) -> Nonlinear
    which is then added to the input.

PreactivationBlock
    Preactivation block with only one nonlinear operation. 
    The sequence of operations is
        (N - 1) × (BN -> Pad -> Conv) -> BN -> nonlinear -> Pad -> Conv

PreactivationResidualBlock
    Preactivation residual block with only one nonlinear operation. 
    The sequence of operations is
        (N - 1) × (BN -> Pad -> Conv) -> BN -> nonlinear -> Pad -> Conv
    which is then added to the input.

Methods
-------
block_wrapper(block, num_repeats, **kwargs):
    Interface to the block class that takes as input the length
    and any other keyword arguments and provides an accessor to
    the class that passes along the specified values.

conv_block(num_repeats, **kwargs)
    Interface to ConvBlock that stores input arguments for use
    when the method is constructed at a later point.

conv_residual_block(num_repeats, **kwargs)
    Interface to ConvResidualBlock that stores input arguments for use
    when the method is constructed at a later point.

preactivation_block(num_repeats, **kwargs)
    Interface to PreactivationBlock that stores input arguments for use
    when the method is constructed at a later point.

preactivation_residual_block(num_repeats, **kwargs)
    Interface to PreactivationResidualBlock that stores input arguments
    for use when the method is constructed at a later point.
"""
import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
from .padding import WrapPad2d
from ..utils import get_padding, parse_parameter

def block_wrapper(block:nn.Module, num_repeats:int, **kwargs) -> callable:
    """
    Interface to the block class that stores input arguments for use
    when the block is constructed at a later point.

    Parameters
    ----------
    block : torch.nn.Module
        A pytorch module object that represents a block operation.

    num_repeats : int
        The number of repeated block elements

    **kwargs
        Additional parameters and their arguments to retain.

    Returns
    -------
    class_constructor : callable
        Function that takes additional inputs and passes then to the
        class constructor.

    Examples
    --------
    >>> import torch
    >>> from galnet.layers.blocks import block_wrapper, ConvBlock
    >>> block = block_wrapper(ConvBlock, num_repeats=3, dilation=2, kernel_size=3)
    >>> input = torch.randn(1,1,5,5)
    >>> block(in_channels=1, out_channels=1)(input).size()
    torch.Size([1, 1, 5, 5])
    """
    return lambda *_args, **_kwargs: block(*_args, num_repeats=num_repeats, **kwargs, **_kwargs)

def conv_block(num_repeats:int, **kwargs) -> callable:
    """
    Interface to ConvBlock that stores input arguments for use
    when the method is constructed at a later point.

    Parameters
    ----------
    num_repeats : int
        The number of repeated block elements

    **kwargs
        Additional parameters and their arguments to retain. See
        ConvBlock for a listing of the available parameters.

    Examples
    --------
    >>> import torch
    >>> from galnet.layers.blocks import conv_block as block_wrapper
    >>> block = block_wrapper(num_repeats=3, dilation=2, kernel_size=3)
    >>> input = torch.randn(1,1,5,5)
    >>> block(in_channels=1, out_channels=1)(input).size()
    torch.Size([1, 1, 5, 5])
    """
    return block_wrapper(ConvBlock, num_repeats, **kwargs)

def conv_residual_block(num_repeats:int, **kwargs) -> callable:
    """
    Interface to ConvResidualBlock that stores input arguments for use
    when the method is constructed at a later point.

    Parameters
    ----------
    num_repeats : int
        The number of repeated block elements

    **kwargs
        Additional parameters and their arguments to retain. See
        ConvResidualBlock for a listing of the available parameters.

    Examples
    --------
    >>> import torch
    >>> from galnet.layers.blocks import conv_residual_block as block_wrapper
    >>> block = block_wrapper(num_repeats=3, dilation=2, kernel_size=3)
    >>> input = torch.randn(1,1,5,5)
    >>> block(in_channels=1, out_channels=1)(input).size()
    torch.Size([1, 1, 5, 5])
    """
    return block_wrapper(ConvResidualBlock, num_repeats, **kwargs)

def preactivation_block(num_repeats:int, **kwargs) -> callable:
    """
    Interface to PreactivationBlock that stores input arguments for use
    when the method is constructed at a later point.

    Parameters
    ----------
    num_repeats : int
        The number of repeated block elements

    **kwargs
        Additional parameters and their arguments to retain. See
        PreactivationBlock for a listing of the available parameters.

    Examples
    --------
    >>> import torch
    >>> from galnet.layers.blocks import preactivation_block as block_wrapper
    >>> block = block_wrapper(num_repeats=3, dilation=2, kernel_size=3)
    >>> input = torch.randn(1,1,5,5)
    >>> block(in_channels=1, out_channels=1)(input).size()
    torch.Size([1, 1, 5, 5])
    """
    return block_wrapper(PreactivationBlock, num_repeats, **kwargs)

def preactivation_residual_block(num_repeats:int, **kwargs) -> callable:
    """
    Interface to PreactivationResidualBlock that stores input arguments for use
    when the method is constructed at a later point.

    Parameters
    ----------
    num_repeats : int
        The number of repeated block elements

    **kwargs
        Additional parameters and their arguments to retain. See
        PreactivationResidualBlock for a listing of the available
        parameters.

    Examples
    --------
    >>> import torch
    >>> from galnet.layers.blocks import preactivation_residual_block as block_wrapper
    >>> block = block_wrapper(num_repeats=3, dilation=2, kernel_size=3)
    >>> input = torch.randn(1,1,5,5)
    >>> block(in_channels=1, out_channels=1)(input).size()
    torch.Size([1, 1, 5, 5])
    """
    return block_wrapper(PreactivationResidualBlock, num_repeats, **kwargs)

class ConvBlock(nn.Module):
    """
    Generic convolution block. The sequence of operations is

        N × (Pad -> Conv -> BN) -> Nonlinear

    Parameters
    ----------
    num_repeats : int
        The number of repeated operation sequences.

    in_channels : int
        The number of input channels
    
    out_channels : int, Tuple[int]
        Either a single integer, or a tuple of integers that
        represent the number of output channels for each convolution.

    kernel_size : int, Tuple[int]
        Either a single integer, or a tuple of integers that
        represent the kernel_size for each convolution.

    stride : int, Tuple[int]
        Either a single integer, or a tuple of integers that
        represent the stride for each convolution.

    dilation : int, Tuple[int]
        Either a single integer, or a tuple of integers that
        represent the dilation for each convolution.

    wrap_padding : bool
        Boolean indicating whether to apply wrap padding (True) or
        zero padding (False). Default is False.

    nonlinear : torch.nn.Module, optional
        A nonlinear operation. If set to None, then no operation
        is performed. Default is ReLU.

    Examples
    --------
    >>> import torch
    >>> from galnet.layers.blocks import ConvBlock as Block
    >>> input = torch.randn(1,1,16,16)
    >>> Block(num_repeats=2, in_channels=1, out_channels=1, stride=(1,2))(input).size()
    torch.Size([1, 1, 8, 8])
    """
    def __init__(self,
        num_repeats:int,
        in_channels:int,
        out_channels:Union[int, Tuple[int]],
        kernel_size:Union[int, Tuple[int]] = 3,
        stride:Union[int, Tuple[int]] = 1,
        dilation:Union[int, Tuple[int]] = 1,
        wrap_padding:bool = False,
        nonlinear:Optional[nn.Module] = nn.ReLU(),
    ):
        super().__init__()

        out_channels = parse_parameter(out_channels, num_repeats)
        kernel_size  = parse_parameter(kernel_size, num_repeats)
        dilation     = parse_parameter(dilation, num_repeats)
        stride       = parse_parameter(stride, num_repeats)

        Pad = WrapPad2d if wrap_padding else nn.ZeroPad2d

        blocks = []
        for i in range(num_repeats):
            blocks.extend([
                Pad(padding=get_padding(kernel_size[i], dilation[i])),
                nn.Conv2d(
                    in_channels=in_channels if i == 0 else out_channels[i-1],
                    out_channels=out_channels[i],
                    kernel_size=kernel_size[i],
                    dilation=dilation[i],
                    stride=stride[i],
                    bias=False,     # Batch normalization takes care of it
                ),
                nn.BatchNorm2d(out_channels[i]),                
            ])
        if nonlinear is not None:
            blocks.append(nonlinear)
        self.block = nn.Sequential(*blocks)

    def forward(self, input:torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.block(input)

class ConvResidualBlock(ConvBlock):
    """
    Generic residual block with only one nonlinear operation. 
    The sequence of operations is

        N × (Pad -> Conv -> BN) -> Nonlinear

    which is then added to the input.

    Parameters
    ----------
    num_repeats : int
        The number of repeated operation sequences.

    in_channels : int
        The number of input channels
    
    out_channels : int, Tuple[int]
        Either a single integer, or a tuple of integers that
        represent the number of output channels for each convolution.

    kernel_size : int, Tuple[int]
        Either a single integer, or a tuple of integers that
        represent the kernel_size for each convolution.

    stride : int, Tuple[int]
        Either a single integer, or a tuple of integers that
        represent the stride for each convolution.

    dilation : int, Tuple[int]
        Either a single integer, or a tuple of integers that
        represent the dilation for each convolution.

    wrap_padding : bool
        Boolean indicating whether to apply wrap padding (True) or
        zero padding (False). Default is False.

    nonlinear : torch.nn.Module, optional
        A nonlinear operation. If set to None, then no operation
        is performed. Default is ReLU.

    Examples
    --------
    >>> import torch
    >>> from galnet.layers.blocks import ConvResidualBlock as Block
    >>> input = torch.randn(1,1,16,16)
    >>> Block(num_repeats=2, in_channels=1, out_channels=1, kernel_size=(3,5))(input).size()
    torch.Size([1, 1, 16, 16])
    """
    def __init__(self,
        num_repeats:int,
        in_channels:int,
        out_channels:Union[int, Tuple[int]],
        kernel_size :Union[int, Tuple[int]] = 3,
        stride:Union[int, Tuple[int]] = 1,
        dilation:Union[int, Tuple[int]] = 1,
        wrap_padding:bool = False,
        nonlinear:Optional[nn.Module] = nn.ReLU(),
    ):
        out_channels = parse_parameter(out_channels, num_repeats)
        if in_channels != out_channels[-1]:        
            raise ValueError('in_channels != out_channels[-1]')

        super().__init__(
            num_repeats = num_repeats,
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = kernel_size,
            stride = stride,
            dilation = dilation,
            wrap_padding = wrap_padding,
            nonlinear = nonlinear,
        )

    def forward(self, input:torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return input + self.block(input)

class PreactivationBlock(nn.Module):
    """
    Preactivation block with only one nonlinear operation. 
    The sequence of operations is

        (N - 1) × (BN -> Pad -> Conv) -> BN -> nonlinear -> Pad -> Conv
    
    For residual connections, see PreactivationResidualBlock

    Parameters
    ----------
    num_repeats : int
        The number of repeated operation sequences.

    in_channels : int
        The number of input channels
    
    out_channels : int, Tuple[int]
        Either a single integer, or a tuple of integers that
        represent the number of output channels for each convolution.

    kernel_size : int, Tuple[int]
        Either a single integer, or a tuple of integers that
        represent the kernel_size for each convolution.

    stride : int, Tuple[int]
        Either a single integer, or a tuple of integers that
        represent the stride for each convolution.

    dilation : int, Tuple[int]
        Either a single integer, or a tuple of integers that
        represent the dilation for each convolution.

    wrap_padding : bool
        Boolean indicating whether to apply wrap padding (True) or
        zero padding (False). Default is False.

    nonlinear : torch.nn.Module, optional
        A nonlinear operation. If set to None, then no operation
        is performed. Default is ReLU.

    Examples
    --------
    >>> import torch
    >>> from galnet.layers.blocks import PreactivationBlock as Block
    >>> input = torch.randn(1,1,16,16)
    >>> Block(num_repeats=2, in_channels=1, out_channels=1, stride=(1,2))(input).size()
    torch.Size([1, 1, 8, 8])
    """
    def __init__(self,
        num_repeats:int,
        in_channels:int,
        out_channels:Union[int, Tuple[int]],
        kernel_size:Union[int, Tuple[int]] = 3,
        stride:Union[int, Tuple[int]] = 1,
        dilation:Union[int, Tuple[int]] = 1,
        wrap_padding:bool = False,
        nonlinear:Optional[nn.Module] = nn.ReLU(),
    ):
        super().__init__()

        out_channels = parse_parameter(out_channels, num_repeats)
        kernel_size  = parse_parameter(kernel_size, num_repeats)
        dilation     = parse_parameter(dilation, num_repeats)
        stride       = parse_parameter(stride, num_repeats)

        Pad = WrapPad2d if wrap_padding else nn.ZeroPad2d

        blocks = []
        for i in range(num_repeats):
            blocks.extend([
                nn.BatchNorm2d(in_channels if i == 0 else out_channels[i-1]),
                Pad(padding=get_padding(kernel_size[i], dilation[i])),
                nn.Conv2d(
                    in_channels=in_channels if i == 0 else out_channels[i-1],
                    out_channels=out_channels[i],
                    kernel_size=kernel_size[i],
                    dilation=dilation[i],
                    stride=stride[i],
                    bias=True if i == num_repeats - 1 else False, # Batch norm takes care of bias; keep last for nonlinear operation
                ),
            ])

        if nonlinear is not None:
            blocks.insert(-2, nonlinear)
        self.block = nn.Sequential(*blocks)

    def forward(self, input:torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.block(input)

class PreactivationResidualBlock(PreactivationBlock):
    """
    Preactivation residual block with only one nonlinear operation. 
    The sequence of operations is

        (N - 1) × (BN -> Pad -> Conv) -> BN -> nonlinear -> Pad -> Conv

    which is then added to the input.

    Parameters
    ----------
    num_repeats : int
        The number of repeated operation sequences.

    in_channels : int
        The number of input channels
    
    out_channels : int, Tuple[int]
        Either a single integer, or a tuple of integers that
        represent the number of output channels for each convolution.

    kernel_size : int, Tuple[int]
        Either a single integer, or a tuple of integers that
        represent the kernel_size for each convolution.

    stride : int, Tuple[int]
        Either a single integer, or a tuple of integers that
        represent the stride for each convolution.

    dilation : int, Tuple[int]
        Either a single integer, or a tuple of integers that
        represent the dilation for each convolution.

    wrap_padding : bool
        Boolean indicating whether to apply wrap padding (True) or
        zero padding (False). Default is False.

    nonlinear : torch.nn.Module, optional
        A nonlinear operation. If set to None, then no operation
        is performed. Default is ReLU.

    Examples
    --------
    >>> import torch
    >>> from galnet.layers.blocks import PreactivationResidualBlock as Block
    >>> input = torch.randn(1,1,16,16)
    >>> Block(num_repeats=2, in_channels=1, out_channels=1, kernel_size=(3,5))(input).size()
    torch.Size([1, 1, 16, 16])
    """
    def __init__(self,
        num_repeats:int,
        in_channels:int,
        out_channels:Union[int, Tuple[int]],
        kernel_size:Union[int, Tuple[int]] = 3,
        stride:Union[int, Tuple[int]] = 1,
        dilation:Union[int, Tuple[int]] = 1,
        wrap_padding:bool = False,
        nonlinear:Optional[nn.Module] = nn.ReLU(),
    ):
        out_channels = parse_parameter(out_channels, num_repeats)
        if in_channels != out_channels[-1]:        
            raise ValueError('in_channels != out_channels[-1]')

        super().__init__(
            num_repeats = num_repeats,
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = kernel_size,
            stride = stride,
            dilation = dilation,
            wrap_padding = wrap_padding,
            nonlinear = nonlinear,
        )
        
    def forward(self, input:torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return input + self.block(input)

class SkipBlock(nn.Module):
    """
    A skip block that simply returns the input. May be useful
    for create skip connections that involve no operations.
    """
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, input:torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return input