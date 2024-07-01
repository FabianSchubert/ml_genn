import numpy as np

from .loss import Loss
from ..utils.model import NeuronModel

from typing import Callable

from jax import grad


class ManualGradient(Loss):
    def __init__(self, loss_func: Callable):
        self.loss = loss_func
        # the provided loss function should take the input/predictions as the first argument.
        self.grad_loss = grad(loss_func)

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

        assert (
            y_true.shape == expected_shape
        ), f"Expected shape {expected_shape}, got {y_true.shape} for y_true"

        genn_pop.extra_global_params["RingV"].pull_from_device()

        voltages = np.reshape(
            genn_pop.extra_global_params["RingV"].view[:], expected_shape
        )

        gradients = -self.grad_loss(voltages, y_true)

        genn_pop.extra_global_params["Gradient"].view[:] = gradients.flatten()
        genn_pop.extra_global_params["Gradient"].push_to_device()
