from .readout import Readout
from .spike_count import SpikeCount
from .sum_var import SumVar
from .var import Var

from ..utils.module import get_module_classes

default_readouts = get_module_classes(globals(), Readout)