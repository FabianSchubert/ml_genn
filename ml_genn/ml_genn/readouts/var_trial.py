import numpy as np

from .readout import Readout
from ..utils.model import NeuronModel

from copy import deepcopy


class VarTrial(Readout):
    """Read out instantaneous value of neuron model's output variable"""

    def add_readout_logic(self, model: NeuronModel, **kwargs) -> NeuronModel:
        self.output_var_name = model.output_var_name

        self.batch_size = kwargs["batch_size"]
        self.pop_shape = kwargs["pop_shape"]
        self.example_timesteps = int(kwargs["example_timesteps"])

        if "vars" not in model.model:
            raise RuntimeError(
                "Var readout can only be used " "with models with state variables"
            )
        if self.output_var_name is None:
            raise RuntimeError(
                "Var readout requires that models " "specify an output variable name"
            )

        # Find output variable
        try:
            _ = next(v for v in model.model["vars"] if v[0] == self.output_var_name)
        except StopIteration:
            raise RuntimeError(
                f"Model does not have variable " f"{self.output_var_name} to read"
            )

        model_copy = deepcopy(model)

        ring_size = self.batch_size * self.example_timesteps * np.prod(self.pop_shape)

        model_copy.add_egp(
            f"Ring{self.output_var_name}",
            "scalar*",
            np.empty(ring_size, dtype=np.float32),
        )

        model_copy.append_sim_code(
            f"""
            const unsigned int ts = min((int)(t / dt), {self.example_timesteps - 1});
            const unsigned int recindex = batch * {self.example_timesteps} * num_neurons + (ts * num_neurons) + id;
            Ring{self.output_var_name}[recindex] = {self.output_var_name};
            """
        )

        return model_copy

    def get_readout(self, genn_pop, batch_size: int, shape) -> np.ndarray:
        # Pull variable from genn
        genn_pop.extra_global_params[f"Ring{self.output_var_name}"].pull_from_device()

        # Return contents, reshaped as desired
        return np.reshape(
            genn_pop.extra_global_params[f"Ring{self.output_var_name}"].view,
            (batch_size, self.example_timesteps) + shape,
        )
