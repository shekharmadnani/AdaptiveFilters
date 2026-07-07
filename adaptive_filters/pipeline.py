"""Frame -> feature-vector pipeline: probe bank at two scales, pooled names."""

import numpy as np

from .utils import to_float, downsample2
from .probes import (
    DctKeepKProbe,
    DeblockProbe,
    DirectionalProbe,
    GuidedProbe,
    SaoProbe,
    TemporalProbe,
)


def default_probes():
    return [
        DctKeepKProbe(),
        GuidedProbe(),
        DeblockProbe(),
        SaoProbe(),
        DirectionalProbe(),
        TemporalProbe(),
    ]


class FeatureExtractor:
    """Runs the probe bank on a luma frame at `num_scales` dyadic scales.

    Returns a flat {name: value} dict with keys like "s0_dct_k_mean".
    Temporal features appear only when prev_frame is given -- keep that
    consistent across a dataset so vectors have a fixed length.
    """

    def __init__(self, probes=None, num_scales=2):
        self.probes = probes if probes is not None else default_probes()
        self.num_scales = num_scales

    def extract(self, frame, prev_frame=None):
        f = to_float(frame)
        p = to_float(prev_frame) if prev_frame is not None else None

        features = {}
        for s in range(self.num_scales):
            for probe in self.probes:
                result = probe.run(f, p)
                if result is None:
                    continue
                for k, v in result.features.items():
                    features[f"s{s}_{probe.name}_{k}"] = float(v)
            if s + 1 < self.num_scales:
                f = downsample2(f)
                p = downsample2(p) if p is not None else None
        return features


def to_vector(features, names=None):
    """Stable (names, vector) from a feature dict; sorted keys by default."""
    if names is None:
        names = sorted(features.keys())
    return names, np.array([features[n] for n in names], dtype=np.float64)
