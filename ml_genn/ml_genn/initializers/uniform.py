from .initializer import Initializer
from ..utils.snippet import ConstantValueDescriptor, InitializerSnippet


class Uniform(Initializer):
    min = ConstantValueDescriptor()
    max = ConstantValueDescriptor()

    def __init__(self, min: float = 0.0, max: float = 1.0):
        super(Uniform, self).__init__()

        self.min = min
        self.max = max

    def get_snippet(self):
        return InitializerSnippet("Uniform",
                                  {"min": self.min, "max": self.max}, {})

    def __repr__(self):
        return f"(Uniform) Min: {self.min}, Max: {self.max}"
