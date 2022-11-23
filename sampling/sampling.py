import random
from typing import Callable, Generator

import numpy as np


def nums_vs(nums: np.ndarray, vs: np.ndarray) -> np.ndarray:
    return np.multiply(vs.T, nums).T 

def randdirs3d(n: int) -> np.ndarray:
    n *= 10
    xs = np.random.uniform(-1, 1, (n, 2))
    xnorms = np.sum(xs**2, axis=1)

    xs = xs[xnorms < 1.]
    xnorms = xnorms[xnorms < 1.]

    nxs = 2 * xs[:, 0] * np.sqrt(1-xnorms)
    nys = 2 * xs[:, 1] * np.sqrt(1-xnorms)
    nzs = 1 - 2*xnorms

    n //= 10
    return np.hstack((nxs[:n].reshape(-1, 1), nys[:n].reshape(-1, 1), nzs[:n].reshape(-1, 1)))

# Proposal function for Metropolis–Hastings algorithm, using here a gaussian with width sigma
def proposal(x: np.ndarray, sigma: float = 1.) -> np.ndarray:
    return np.random.normal(x, sigma)

# Metropolis–Hastings algorithm for sampling from a probability distribution
def metropolis(pdistr: Callable, n: int, x0: np.ndarray, sigma: float = 1., burn_in: int = 10_000) -> Generator:
    x = x0 # start somewhere
    for _ in range(n+burn_in+1):
        trial = proposal(x, sigma) # random neighbor from the proposal distribution
        acceptance = pdistr(trial) / pdistr(x)
        
        # accept the move conditionally
        if random.random() < acceptance:
            x = trial

        if _ > burn_in:
            yield x