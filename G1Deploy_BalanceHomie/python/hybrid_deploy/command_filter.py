"""The first-order command filter used to collect the Balance switch states."""
import numpy as np


def filter_command(current, target, dt, tau):
    if not np.isfinite(dt) or dt < 0 or not np.isfinite(tau) or tau <= 0:
        raise ValueError('Invalid command filter interval/time constant')
    return current + min(1., dt / tau) * (target - current)
