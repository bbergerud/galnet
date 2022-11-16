"""
General utility functions for constructing model networks.

Functions
---------
count_parameters(model)
    Returns the number of parameters in the model that are trainable.

get_merged_channels(channels, merge_operation)
    Calculates the number of input channels based on the merge operation when joining.

get_padding(kernel_size, dilation)
    Calculates the padding factor for maintaining the spatial size
    given the kernel size and dilation factors. The returned value
    is
        (kernel_size + (kernel_size - 1) * (dilation - 1)) // 2
    Note that an improper value will be returned if the padding is
    not symmetric on each side, thus odd kernel sizes should generally
    be used.

parse_parameter(input, length)
    Takes a value or iterable collection of values and parses
    the input to make sure it has the proper size.

Variables
----------
merge_operations : dict
    A dictionary containing different merge options. The available
    keys are {'add', 'cat', 'concat', 'mul'}. Each operation should
    take as input a sequence of tensors.
"""

import torch
import torch.nn as nn
from numbers import Number
from dataops.utils import flatten
from functools import reduce
from typing import List, Tuple, Union

merge_operations = {
    'cat'   : lambda *x: torch.cat(tuple(flatten(x, torch.Tensor)), dim=1),
    'concat': lambda *x: torch.cat(tuple(flatten(x, torch.Tensor)), dim=1),
    'add'   : lambda *x: reduce(torch.add, tuple(flatten(x, torch.Tensor))),
    'mul'   : lambda *x: reduce(torch.mul, tuple(flatten(x, torch.Tensor))),
}

def count_parameters(
    model:nn.Module
) -> int:
    """
    Returns the number of parameters in the model that are trainable.

    Parameters
    ----------
    model : Module
        A torch.nn.Module class that represents a network model

    Returns
    -------
    number : int
        The number of parameters.

    Examples
    --------
    >>> from galnet.utils import count_parameters
    >>> import torch.nn as nn
    >>> model = nn.Linear(10, 10, bias=False)
    >>> print(count_parameters(model))
    100
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_merged_channels(
    channels:List[int],
    merge_operation:str,
) -> int:
    """
    Calculates the number of input channels based on the merge operation when joining.

    Parameters
    ----------
    channels : List[int]
        A sequence of channel sizes to combine. 

    merge_operation : str
        The merging operation to join multiple inputs.

    Returns
    -------
    channels : int
        The number of input channels corresponding to the provided
        channel numbers and the merge operation.

    Raises
    ------
    NotImplementedError
        If the provided merge operation is not implemented, then an
        exception is raised.

    ValueError
        If the provided channels are not compatible with the
        designated merge operation, then a ValueError is raised.

    Examples
    --------
    >>> from galnet.utils import get_merged_channels
    >>> print(get_merged_channels([5,5,5,5], merge_operation='add'))
    5
    >>> print(get_merged_channels([5,4,3,2,1], merge_operation='concat'))
    15
    """
    if merge_operation not in merge_operations.keys():
        raise NotImplementedError('Not a valid merge operation. Should be one of: {}'.format(merge_operations.keys()))
    elif merge_operation in ['cat', 'concat']:
        return sum(channels)
    else:
        if len(set(channels)) != 1:
            raise ValueError("Channels must have the same size when merge_operation={}".format(merge_operation))
        return channels[0]

def get_padding(
    kernel_size:int,
    dilation:int = 1,
) -> int:
    """
    Calculates the padding factor for maintaining the spatial size
    given the kernel size and dilation factors. The returned value
    is

        (kernel_size + (kernel_size - 1) * (dilation - 1)) // 2

    Note that an improper value will be returned if the padding is
    not symmetric on each side, thus odd kernel sizes should generally
    be used.

    Parameters
    ----------
    kernel_size : int
        The kernel size. Should be an odd number.

    dilation : int
        The dilation factor

    Returns
    -------
    padding : int
        The padding factor

    Examples
    --------
    >>> import torch
    >>> from galnet.utils import get_padding
    >>> conv = torch.nn.Conv2d(
    ...    in_channels=1,
    ...    out_channels=1,
    ...    kernel_size=5,
    ...    dilation=3,
    ...    padding=get_padding(
    ...        kernel_size=5,
    ...        dilation=3
    ...    )
    ... )
    >>> y = conv(torch.randn(1,1,10,10))
    >>> print(y.shape)
    torch.Size([1, 1, 10, 10])
    """
    return (kernel_size + (kernel_size - 1)*(dilation - 1))//2

def parse_parameter(
    input:Union[Number, Tuple[Number], List[Number]],
    length:int
) -> Tuple[Number]:
    """
    Takes a value or iterable collection of values and parses
    the input to make sure it has the proper size.

    Parameter
    ---------
    input : number, tuple
        The input values to parse. Should either be a single value
        or a collection of values.
    
    length : int
        The length of the output value.

    Returns
    -------
    output : tuple
        If input is a single value, then the value is returned as a tuple 
        with length `length`. Otherwise, the input value is returned if
        no exception is raised.

    Raises
    ------
    ValueError
        If input is an iterable but not of the proper length, then an
        exception is raised.

    Examples
    --------
    >>> from galnet.utils import parse_parameter
    >>> parse_parameter(input=1, length=3)
    (1, 1, 1)
    >>> parse_parameter(input=(1,2,3), length=3)
    (1, 2, 3)
    """
    if isinstance(input, (list, tuple)):
        if len(input) != length:
            raise ValueError(f"len({input}) != {length}")
        return tuple(input)
    else:
        return (input,) * length