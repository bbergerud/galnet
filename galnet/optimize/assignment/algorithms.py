"""
Methods for identifying the optimal pair matchings given a cost matrix

Functions
---------
first_come_first_serve(cost_matrix)
    Assigns the optimal matchings in order of the network output. Assumes
    that the output instance is the row and the target instance the column.

greedy(cost_matrix)
    Uses a greedy approach towards minimizing the overall cost function.
"""

import torch
from dataops.utils import unravel_index
from typing import Tuple

@torch.no_grad()
def first_come_first_serve(
    cost_matrix:torch.Tensor,
) -> Tuple[Tuple[int], Tuple[int]]:
    """
    Assigns the optimal matchings in order of the network output. Assumes
    that the output instance is the row and the target instance the column.

    Parameters
    ----------
    cost_matrix : torch.Tensor
        The cost matrix

    Returns
    -------
    rows : Tuple[int]
        The optimal row indices for matching

    cols : Tuple[int]
        The optimal column indices for matching

    Examples
    --------
    >>> import torch
    >>> from galnet.optimize.assignment.algorithms import first_come_first_serve
    >>> x = torch.tensor([[2., 3., 4., 5.],
    ...                   [3., 2., 1., 0.],
    ...                   [4., 5., 3., 2.],
    ...                   [4., 5., 1., 3.]])
    >>> rows, cols = first_come_first_serve(x)
    >>> print(rows)
    (0, 1, 2, 3)
    >>> print(cols)
    (0, 3, 2, 1)
    """
    if cost_matrix.ndim != 2:
        raise ValueError("cost_matrix must be an NxM matrix")

    cost_matrix = cost_matrix.detach().clone()

    size = min(cost_matrix.shape)
    rows, cols = [], []

    inf = torch.tensor(float('inf'), device=cost_matrix.device)
    for row in range(size):
        col = torch.argmin(cost_matrix[row]).item()
        rows.append(row)
        cols.append(col)
        cost_matrix[:,col] = inf

    return tuple(rows), tuple(cols)

@torch.no_grad()
def greedy(
    cost_matrix:torch.Tensor,
) -> Tuple[Tuple[int], Tuple[int]]:
    """
    Uses a greedy approach towards minimizing the overall cost function.

    Parameters
    ----------
    cost_matrix : torch.Tensor
        The cost matrix

    Returns
    -------
    rows : Tuple[int]
        The optimal row indices for matching

    cols : Tuple[int]
        The optimal column indices for matching

    Examples
    --------
    >>> import torch
    >>> from galnet.optimize.assignment.algorithms import greedy
    >>> x = torch.tensor([[2., 3., 4., 5.],
    ...                   [3., 2., 1., 0.],
    ...                   [4., 5., 3., 2.],
    ...                   [4., 5., 1., 3.]])
    >>> rows, cols = greedy(x)
    >>> print(rows)
    (1, 3, 0, 2)
    >>> print(cols)
    (3, 2, 0, 1)
    """
    cost_matrix = cost_matrix.detach().clone()

    size = min(cost_matrix.shape)
    rows, cols = [], []

    inf = torch.tensor(float('inf'), device=cost_matrix.device)
    for _ in range(size):
        row, col = unravel_index(torch.argmin(cost_matrix).item(), cost_matrix.shape)
        rows.append(row)
        cols.append(col)

        cost_matrix[row, :] = inf
        cost_matrix[:,col] = inf
    
    return tuple(rows), tuple(cols)