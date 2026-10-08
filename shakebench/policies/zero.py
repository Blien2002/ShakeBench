"""Zero-action policy for installation and evaluation-pipeline smoke checks."""

import numpy as np


class ZeroPolicy:
    """Return a finite action chunk; this is not a learned manipulation policy."""

    chunk_size = 1

    def predict(self, observation):
        return np.zeros((self.chunk_size, 7), dtype=np.float32)


def make_policy():
    return ZeroPolicy()
