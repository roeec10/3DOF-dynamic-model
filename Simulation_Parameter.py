import numpy as np

def get_parameters(rps):
    cyc_num = 76
    lcmTeeth = 38 * 17
    nominalFs = 1e5
    gmsCoarseInterval = 50
    shaft_polar_stiff = 6.343600550517852e3
    Fs = np.ceil(nominalFs / (lcmTeeth * rps) + 6) * lcmTeeth * rps
    T = cyc_num / rps
    dt = 1 / Fs
    time_vctr = np.arange(0, T, dt)
    time_vctr = time_vctr[:, np.newaxis]
    rps_sig = rps * np.ones_like(time_vctr)
    cyc_motor = 2 * np.pi * np.cumsum(rps_sig * dt)
    cyc_motor = cyc_motor - cyc_motor[0]
    dCycGMSFine = 1 / (gmsCoarseInterval ** 2 * lcmTeeth * np.ceil(nominalFs / (gmsCoarseInterval * lcmTeeth)))
    dCycGMSCoarse = dCycGMSFine * gmsCoarseInterval
    return cyc_motor, gmsCoarseInterval, shaft_polar_stiff, time_vctr, dCycGMSCoarse