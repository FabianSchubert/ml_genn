from pygenn.genn_wrapper.Models import VarAccess_READ_ONLY_DUPLICATE
from .input import InputBase
from .neuron import Neuron
from ..utils.model import NeuronModel

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .. import Population

class PoissonInput(Neuron, InputBase):
    def __init__(self, signed_spikes=False, input_frames=1, 
                 input_frame_time=1):
        super(PoissonInput, self).__init__(egp_name="Input", 
                                           input_frames=input_frames,
                                           input_frame_time=input_frame_time)

        self.signed_spikes = signed_spikes
        if self.signed_spikes and input_frames > 1:
            raise NotImplementedError("Signed spike input cannot currently "
                                      "be used with time-varying inputs ")

    def get_model(self, population: "Population", dt: float, batch_size: int):
        genn_model = {
            "sim_code":
                """
                const bool spike = $(gennrand_uniform) >= exp(-fabs($(Input)) * DT);
                """,
            "threshold_condition_code":
                """
                $(Input) > 0.0 && spike
                """,
            "is_auto_refractory_required": False}

        # If signed spikes are enabled, add negative threshold condition
        if self.signed_spikes:
            genn_model["negative_threshold_condition_code"] =\
                """
                $(Input_pre) < 0.0 && spike
                """
        return self.create_input_model(
            NeuronModel(genn_model, None, {}, {}),
            batch_size, population.shape, replace_input="$(Input)")
