import numpy as np

from .synapse import Model, Synapse
from ..utils import InitValue, Value

genn_model = {
    "param_name_types": [("ExpDecay", "scalar")],
    "apply_input_code":
        """
        $(Isyn) += $(inSyn);
        """,
    "decay_code":
        """
        $(inSyn) *= $(ExpDecay);
        """}
        
class Exponential(Synapse):
    def __init__(self, tau=5.0):
        super(Exponential, self).__init__()
        
        self.tau = Value(tau)
        
        if self.tau.is_initializer:
            raise NotImplementedError("Exponential synapse model does not "
                                      "currently support tau values specified"
                                      " using Initialiser objects")

    def get_model(self, population, dt):
        return Model(genn_model, 
                     {"ExpDecay": Value(np.exp(-dt / self.tau.value))}, 
                     {})
