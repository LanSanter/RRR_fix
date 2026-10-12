import numpy as np
from phase2 import find_longest_path
from phase3 import phase3, compute_common_message_ratio, distributed_beamforming_gain, compute_private_rate_group
from config import AWGNoise, config, payload_per_hz, power_split, rate_mapping, required_bandwidth


def main():
    rsma_groups, collaborative_groups=phase3()
    print("Final RSMA groups")
    for i, g in enumerate(rsma_groups):
        print(f"Group {i+1}: {[siot.index for siot in g]}")
        cost = compute_RSMA_cost(g, bandwidth_cost_unit=1, mec_cost_unit=1)
        print(f"RSMA Group {i+1} Cost: {cost}\n")

    print("Final Collaborative groups:")
    for i, g in enumerate(collaborative_groups):
        print(f"Group {i+1}: {[siot.index for siot in g]}")
        cost = compute_collaborative_cost(g, bandwidth_cost_unit=1, eta=1)
        print(f"Collaborative Group {i+1} Cost: {cost}\n")

def compute_RSMA_cost(rsma_group, bandwidth_cost_unit, mec_cost_unit):
    """
    計算 collaborative group c 的總成本 (Bandwidth cost + Communication cost + Computation cost)
    collaborative_group: 目前的 collaborative group
    siots: 所有 SIoTs 資訊
    bandwidth_cost_unit: c_bw, bandwidth unit cost
    eta: collaborative parameter
    """
    if not rsma_group:
        return 0  # 空群組直接返回 0

    # 1. Bandwidth cost C_c^bw
    C_bw = bandwidth_cost_unit * compute_rsma_required_bandwidth(rsma_group, T_threshold=5)
    

    # 2. Computation cost C_c^comp at MEC server
    C_comp = mec_cost_unit * len(set.union(*(set(siot.locations) for siot in rsma_group)))


    # 總成本
    total_cost = C_bw + C_comp
    print(f"Total:{total_cost}, Bandwith: {C_bw}, Computation:{C_comp}")
    return total_cost

def compute_collaborative_cost(collaborative_group, bandwidth_cost_unit, eta):
    """
    計算 collaborative group c 的總成本 (Bandwidth cost + Communication cost + Computation cost)
    collaborative_group: 目前的 collaborative group
    siots: 所有 SIoTs 資訊
    bandwidth_cost_unit: c_bw, bandwidth unit cost
    eta: collaborative parameter
    """
    if not collaborative_group:
        return 0  # 空群組直接返回 0

    # 1. Bandwidth cost C_c^bw
    C_bw = bandwidth_cost_unit * compute_collaborative_required_bandwidth(collaborative_group, T_threshold=5)
    
    # 2. Communication cost C_c^com
    num_siot = len(collaborative_group)  # 群組內 SIoT 數量
    if num_siot == 1:
        C_com = 0  # 只有一個設備，無需通信
    else:
        C_com = eta * (num_siot - 1)  # 可選用 |V|-1 (確保連通) 或 Edge 數

    # 3. Computation cost C_c^comp
    C_comp = eta *(sum(abs(len(siot.locations)) for siot in collaborative_group) ) # Covered Locations 絕對值計算

    # 總成本
    total_cost = C_bw + C_com + C_comp
    print(f"Total:{total_cost}, Bandwith: {C_bw}, Communication:{C_com}, Computation:{C_comp}")
    return total_cost

def compute_rsma_required_bandwidth(rsma_group, T_threshold):
    """Bandwidth in Hz; PSD SINRs are independent of allocated bandwidth."""
    if not rsma_group:
        return 0
    if len(rsma_group) == 1:
        bandwidth = compute_single_required_bandwidth(rsma_group, T_threshold)
        return bandwidth

    total_locations = len(set.union(*(set(siot.locations) for siot in rsma_group)))
    mec_cpu = 1
    time_budget = len(rsma_group) * T_threshold - total_locations / mec_cpu
    common_message_ratio, common_locations = compute_common_message_ratio(rsma_group)
    common_fraction, private_fraction = power_split()
    common_signal = sum(abs(siot.gain)**2 * common_fraction * siot.power for siot in rsma_group)
    private_interference = sum(abs(siot.gain)**2 * private_fraction * siot.power for siot in rsma_group)
    common_sinr = common_signal / (private_interference + AWGNoise)
    beamforming_gain = distributed_beamforming_gain(rsma_group, common_message_ratio)
    common_efficiency = rate_mapping(beamforming_gain * common_sinr)
    private_efficiencies = compute_private_rate_group(rsma_group, common_message_ratio)

    # Location counts determine payload, never the stream power split.
    common_message_data_size = len(common_locations) * config['feature_bits']
    common_payload = payload_per_hz(common_message_data_size, common_efficiency)
    private_payload = max(
        (payload_per_hz(
            len(set(siot.locations) - common_locations) * config['feature_bits'],
            private_efficiencies[siot.index],
        ) for siot in rsma_group),
        default=0.0,
    )
    bandwidth = required_bandwidth(max(common_payload, private_payload), time_budget)
    return bandwidth


def compute_single_required_bandwidth(rsma_group, T_threshold):
    """Single-device uplink using the full transmit PSD, returning Hz."""
    if not rsma_group:
        return 0.0
    total_locations = len(set.union(*(set(siot.locations) for siot in rsma_group)))
    time_budget = len(rsma_group) * T_threshold - total_locations / 2.0
    snr = sum(abs(siot.gain)**2 * siot.power for siot in rsma_group) / AWGNoise
    payload = payload_per_hz(total_locations * config['feature_bits'], rate_mapping(snr))
    return required_bandwidth(payload, time_budget)

def compute_collaborative_required_bandwidth(collaborative_group, T_threshold):
    """
    計算該 collaborative group 需要的 Bandwidth.
    
    Parameters:
    - collaborative_group: 該 group 的 SIoT 設備集合
    - T_threshold: 時間門檻
    - SNR_d2d: D2D 通訊的 SNR
    - SNR_u: 上行傳輸的 SNR
    - eta: 加權參數
    """
    
    if not collaborative_group:
        return 0
 
    representative_siot = max(collaborative_group, key=lambda s: s.gain)  # 挑最高 channel gain
    

    # **1. 計算計算時間 (Computation Time)**
    T_comp = max(len(siot.locations) / siot.cpu_frequency for siot in collaborative_group)
    
    # **2. 計算 D2D 通訊時間**
    longest_path_length, critical_siot = find_longest_path(representative_siot,collaborative_group)

    T_d2d = ((len(critical_siot.locations) * 0.4) / critical_siot.d2d_rate) * longest_path_length
    
    # **3. 計算上行傳輸時間**
    
    #剩餘時間 (也扣過整合時間)
    T_rest=(len(collaborative_group) * T_threshold) - T_comp - T_d2d - ((sum(len(siot.locations) * 0.2 for siot in collaborative_group) / representative_siot.cpu_frequency))
    
    #Representative SIoT data rate
    representative_bits = len(set.union(*(set(siot.locations) for siot in collaborative_group))) * 0.2 * config['feature_bits']
    gamma_representative_siot= (abs(representative_siot.gain) ** 2 * representative_siot.power) / AWGNoise
    
    B_u_c = required_bandwidth(payload_per_hz(representative_bits, rate_mapping(gamma_representative_siot)), T_rest)
    
    #print(f"Bandwidth:{B_u_c}")
    return B_u_c



if __name__ == '__main__':
    main()