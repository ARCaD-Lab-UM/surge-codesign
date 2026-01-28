"""
One function that returns objective, components from SRB + policy rollout. No more ad-hoc objective in multiple places.
"""
class Objective:
    def evaluate(self):
        ...
