import numpy as np

from .loss import Loss
from ..utils.model import NeuronModel

from typing import Callable

from jax import grad, vmap


class ManualGradient(Loss):
    def __init__(self, loss_func: Callable):
        self.loss = loss_func
        # the provided loss function should take the input/predictions as the first argument.
        self.grad_loss = grad(loss_func)
        # if we use self.grad_loss "as is", this can lead to gradients being interdependent
        # between batches (which are represented by the 0-th axis in set_target).
        # So we use jax.vmap to vectorize over the 0-th axis, which means that the losses/gradients
        # are calculated independently across batches.
        self.grad_loss_batch = vmap(self.grad_loss, in_axes=0, out_axes=0)

    def add_to_neuron(
        self, model: NeuronModel, shape, batch_size: int, example_timesteps: int
    ):
        pass

    def set_target(
        self,
        genn_pop,
        y_true: np.ndarray,
        shape: tuple,
        batch_size: int,
        example_timesteps: int,
    ):
        expected_shape = (batch_size, example_timesteps) + shape

        y_true = np.asarray(y_true)

        # check if the number of dimensions match
        assert y_true.ndim == len(
            expected_shape
        ), f"Expected {len(expected_shape)} axes, but got {y_true.ndim} for y_true."

        batch_size_y_true = y_true.shape[0]

        # check if the number of batches is smaller or equal to batch_size.
        assert (
            batch_size_y_true <= batch_size
        ), f"Number of batches in y_true was {batch_size_y_true}, but must be less or equal than {batch_size}."

        # check if the rest of the axes match
        assert (
            y_true.shape[1:] == expected_shape[1:]
        ), f"Expected shape {expected_shape[1:]} for non-batch axes, got {y_true.shape[1:]} for y_true."

        genn_pop.extra_global_params["RingV"].pull_from_device()

        voltages = np.reshape(
            genn_pop.extra_global_params["RingV"].view[:], expected_shape
        )

        gradients = np.zeros(expected_shape)
        gradients[:batch_size_y_true] = self.grad_loss_batch(
            voltages[:batch_size_y_true], y_true
        )

        genn_pop.extra_global_params["Gradient"].view[:] = gradients.flatten()
        genn_pop.extra_global_params["Gradient"].push_to_device()
