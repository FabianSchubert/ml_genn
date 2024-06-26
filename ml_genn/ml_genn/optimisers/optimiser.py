from abc import ABC

from ..utils.model import CustomUpdateModel

from abc import abstractmethod


class Optimiser(ABC):
    """Base class for all optimisers"""

    @abstractmethod
    def set_step(self, genn_cu, step: int):
        """Performs optimiser-specific update to compiled optimier
        object at given training step e.g. recalculating learning rates

        Args:
            genn_cu:    GeNN ``CustomUpdate`` object optimiser has been
                        compiled into
            step:       Training step
        """
        pass

    @abstractmethod
    def get_model(self, gradient_ref, var_ref,
                  zero_gradient: bool) -> CustomUpdateModel:
        """Gets model described by this optimiser

        Args:
            gradient_ref:    GeNN variable reference for model 
                             to read gradient from
            var_ref:         GeNN variable refernece to variable to update
            zero_gradient:   Should gradient be zeroed at the end of the
                             optimiser custom update? This is typically the
                             behaviour we want when batch size is 1 but, 
                             otherwise, gradient_ref points to an intermediate
                             reduced gradient which there is no point in zeroing.
        """
        pass
