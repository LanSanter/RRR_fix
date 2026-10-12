import math


config = {
    'num_devices' : 100, #experimental variables
    'BANDWIDTH': 1 * 10**6,  # Hz (1 MHz)
    'num_antennas': 4, 
    'data_quantity_constraint': 12000, #(change in main)
    'data_coverage_constraint': 0.6,
    'label_class': 10, #(change in main)
    'theta_SDCI': 0.4,
    'count_limit': 4,
    'com_weight': 1, 
    'cmp_weight': 1e-9, #使用GIGA Flops計算
    'importance_epsilon': 0.6,
    'delay_constraint': 1, #experimental variables
    'feature_bits': 16 * 1024 * 8,  # bits per monitoring location
    'pvt_ratio': 0.4,
    'cmn_ratio': 0.6
}
#math
PI = 3.14159

# powerBS = 6.31 #Watt
powerBS = 15.62
AWGNoise = 10.0 ** ((-174.0 - 30.0) / 10.0)  # W/Hz, from -174 dBm/Hz
TX_POWER_PSD = 10.0 ** ((1.0 - 30.0) / 10.0)  # W/Hz, from 1 dBm/Hz

def rate_mapping(sinr_linear):
    """Shannon spectral efficiency in bit/s/Hz for a linear SINR.

    The old MCS table had no calibrated units and accepted dB thresholds.
    Keep the public name so existing callers now use the paper's rate model.
    """
    if not math.isfinite(sinr_linear) or sinr_linear < 0:
        raise ValueError("SINR must be finite, non-negative, and in linear units.")
    return math.log1p(sinr_linear) / math.log(2.0)


def power_split():
    """Return configured common/private fractions, independent of data overlap."""
    common, private = config['cmn_ratio'], config['pvt_ratio']
    if not (0 <= common <= 1 and 0 <= private <= 1):
        raise ValueError("RSMA power fractions must be in [0, 1].")
    if not math.isclose(common + private, 1.0, abs_tol=1e-12):
        raise ValueError("RSMA power fractions must sum to one.")
    return common, private


def payload_per_hz(data_bits, efficiency):
    """Return seconds * Hz; an empty stream takes zero time even at zero rate."""
    if data_bits <= 0:
        return 0.0
    return data_bits / efficiency if efficiency > 0 else float('inf')


def required_bandwidth(payload, time_budget):
    """Return Hz; no remaining transmission time is infeasible."""
    if time_budget <= 0:
        return float('inf')
    return payload / time_budget
