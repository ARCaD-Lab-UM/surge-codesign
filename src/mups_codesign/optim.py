"""
Two entry points: optimize_gradient(...) and optimize_cmaes(...), both calling the same objective() interface.
"""
class Optimizer:
    def optimize_gradient(self):
        ...
    
    def optimize_cmaes(self):
        ...
