import random
from typing import Callable, Generator

import numpy as np


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