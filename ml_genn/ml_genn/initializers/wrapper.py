from .initializer import Initializer, Snippet

class Wrapper(Initializer):
    def __init__(self, snippet, param_vals, egp_vals):
        super(Wrapper, self).__init__()
        
        self.snippet = snippet
        self.param_vals = param_vals
        self.egp_vals = egp_vals
    
    def get_snippet(self):
        return Snippet(self.snippet, self.param_vals, self.egp_vals)
