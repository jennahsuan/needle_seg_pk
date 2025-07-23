# torch.optim.swa_utils
## https://github.com/pytorch/pytorch/blob/main/torch/optim/swa_utils.py
r"""Implementation for Stochastic Weight Averaging implementation."""
import itertools
import math
import warnings
from copy import deepcopy
from typing import Any, Callable, Iterable, List, Literal, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Module
# from torch.optim.lr_scheduler import _format_param, LRScheduler
# from torch.utils._foreach_utils import _get_foreach_kernels_supported_devices
# from .optimizer import Optimizer

@torch.no_grad()
def update_bn_override(
    loader: Iterable[Any],
    model: Module,
    device: Optional[Union[int, torch.device]] = None,
    buffer_num_sample: Optional[int] = None,
    time_window: Optional[int] = None,
):
    r"""Update BatchNorm running_mean, running_var buffers in the model.

    It performs one pass over data in `loader` to estimate the activation
    statistics for BatchNorm layers in the model.
    Args:
        loader (torch.utils.data.DataLoader): dataset loader to compute the
            activation statistics on. Each data batch should be either a
            tensor, or a list/tuple whose first element is a tensor
            containing data.
        model (torch.nn.Module): model for which we seek to update BatchNorm
            statistics.
        device (torch.device, optional): If set, data will be transferred to
            :attr:`device` before being passed into :attr:`model`.

    Example:
        >>> # xdoctest: +SKIP("Undefined variables")
        >>> loader, model = ...
        >>> torch.optim.swa_utils.update_bn(loader, model)

    .. note::
        The `update_bn` utility assumes that each data batch in :attr:`loader`
        is either a tensor or a list or tuple of tensors; in the latter case it
        is assumed that :meth:`model.forward()` should be called on the first
        element of the list or tuple corresponding to the data batch.
    """
    momenta = {}
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.reset_running_stats()
            momenta[module] = module.momentum

    if not momenta:
        return

    was_training = model.training
    model.train()
    for module in momenta.keys():
        module.momentum = None

    for input in loader:
        if buffer_num_sample and time_window:
            for t in range(buffer_num_sample):  # iterate over the buffer to get samples
                # Image data
                input_imgs = input["images"][:, t : t + time_window, :, :]  # [N, T, H, W]
                if device is not None:
                    input_imgs = input_imgs.to(device)
                model(input_imgs)
        else:
            if isinstance(input, (list, tuple)):
                input = input[0]
            if device is not None:
                input = input.to(device)

            model(input)

    for bn_module in momenta.keys():
        bn_module.momentum = momenta[bn_module]
    model.train(was_training)
