"""Content-adaptive filter probe bank for no-reference video quality estimation.

Every probe follows one template:
  1. Naturalness model  : parametric model of natural content (sparse DCT,
                          locally linear, directional, neighbor-consistent, ...)
  2. Blind theta fit    : parameters estimated from the (degraded) content itself
  3. Probe residual     : r = y - F_theta(y)  -- what the model could not explain
  4. Features           : statistics of r, its spatial localization, and theta itself
"""

from .pipeline import FeatureExtractor, to_vector
from .naturalness import NaturalnessModel
from .fusion import RidgeFusion, GbtFusion

__all__ = [
    "FeatureExtractor",
    "to_vector",
    "NaturalnessModel",
    "RidgeFusion",
    "GbtFusion",
]
