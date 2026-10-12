import numpy as np
from itertools import combinations
from phase2 import phase2, check_distributed_beamforming_constraint, compute_collaborative_cost
from config import AWGNoise, config, payload_per_hz, power_split, rate_mapping, required_bandwidth

def give_rsma_collaborative_groups(rsma_groups, collaborative_groups):
    return rsma_groups, collaborative_groups

def phase3(rsma_groups, collaborative_groups,unselected_siot_set):
    total_cost=0
    total_bandwidth_cost=0
    total_computation_cost=0
    total_communication_cost=0
    total_common_message_data_size=0
    total_collaborative_coverage=0
    total_RSMA_coverage=0
    total_collaborative_cost=0
    total_beamforming_gain=0
    total_local_computation_cost=0
    total_siot=0
    #rsma_groups, collaborative_groups, unselected_siot_set=phase2()
    # print("From Phase2 RSMA groups:")
    # for i, g in enumerate(rsma_groups):
    #     print(f"Group {i+1}: {[siot.index for siot in g]}")
    # print("From Phase2 Collaborative groups:")
    # for i, g in enumerate(collaborative_groups):
    #     print(f"Group {i+1}: {[siot.index for siot in g]}")

    rsma_groups, collaborative_groups=siot_elimination(rsma_groups, collaborative_groups)

    # print("After elimination RSMA groups")
    # for i, g in enumerate(rsma_groups):
    #     print(f"Group {i+1}: {[siot.index for siot in g]}")
    # print("After elimination Collaborative groups:")
    # for i, g in enumerate(collaborative_groups):
    #     print(f"Group {i+1}: {[siot.index for siot in g]}")
    #print("Unselected SIoTs:", [siot.index for siot in unselected_siot_set])

    if rsma_groups:
        for i, rsma_group in enumerate(rsma_groups):
            rsma_groups[i] = replace_siot_in_group(rsma_group, unselected_siot_set)

    rsma_groups = [group for group in rsma_groups if group]
    collaborative_groups=[group for group in collaborative_groups if group]

    print("Final RSMA groups")
    for i, g in enumerate(rsma_groups):
        print(f"Group {i+1}: {[siot.index for siot in g]}")
        cost, bandwidth_cost, computation_cost,RSMA_coverage,common_message_data_size,beamforming_gain = compute_RSMA_cost(g, bandwidth_cost_unit=1, mec_cost_unit=1)
        total_cost+=cost
        total_bandwidth_cost+=bandwidth_cost
        total_computation_cost+=computation_cost
        total_RSMA_coverage+=RSMA_coverage
        total_common_message_data_size+=common_message_data_size
        total_beamforming_gain+=beamforming_gain
        total_siot+=len(g)
        print(f"RSMA Group {i+1} Cost: {cost}\n")
    if len(rsma_groups) == 0:
        total_beamforming_gain=0
    else:
        total_beamforming_gain= total_beamforming_gain/len(rsma_groups)
    print("Final Collaborative groups:")
    for i, g in enumerate(collaborative_groups):
        print(f"Group {i+1}: {[siot.index for siot in g]}")
        coverage = len(set.union(*[set(m.locations) for m in g]))
        cost, C_bw, C_com, C_comp = compute_collaborative_cost(g, bandwidth_cost_unit=1, eta=1)
        total_cost+=cost
        total_collaborative_cost+=cost
        total_bandwidth_cost+=C_bw
        total_computation_cost+=C_comp
        total_communication_cost+=C_com
        total_collaborative_coverage+=coverage
        total_local_computation_cost+=C_comp
        total_siot+=len(g)
        print(f"Collaborative Group {i+1} Cost: {cost}\n")
    return total_cost,total_bandwidth_cost,total_computation_cost,total_communication_cost,total_RSMA_coverage,total_common_message_data_size,total_collaborative_cost,total_collaborative_coverage,total_beamforming_gain, total_local_computation_cost,total_siot

def siot_elimination(rsma_groups, collaborative_groups):
    """
    Phase 3: SIoT Elimination and Replacement (SER)
    - Remove SIoTs whose coverage is entirely encompassed by others.
    - Ensure location coverage constraints remain satisfied.
    """

    # 組合所有群組
    all_groups = collaborative_groups + rsma_groups

    # 建立一個待檢查的 SIoT 列表
    siot_list = sorted(
        [siot for group in all_groups for siot in group],
        key=lambda x: len(x.locations)  # 根據監測位置數量排序 (最少的優先)
    )
    #print(f"sorted_list {[i.index for i in siot_list]}")

    # 遍歷所有 SIoT，嘗試移除冗餘的 SIoT
    for siot in siot_list:

        # 計算當前 SIoT 覆蓋的監測位置
        siot_coverage = set(siot.locations)

        # 計算群組內其他 SIoT 的聯合覆蓋 (排除當前 SIoT)
        remaining_coverage = set().union(*(set(s.locations) for g in all_groups for s in g if s != siot))

        # 如果 siot_coverage ⊆ remaining_coverage，則 n 是冗餘的
        if siot_coverage.issubset(remaining_coverage):
            if siot in collaborative_groups:
                # Find the specific group that contains siot
                siot_group = next((group for group in collaborative_groups if siot in group), None)
    
                # Ensure siot_group exists before checking connectivity
                if siot_group and not satisfies_social_connectivity_constraint(siot, siot_group):
                    continue  # Skip removal if connectivity is not preserved
            print(f"Removing redundant SIoT {siot.index} from group")
            # **遍歷所有群組，找到 `siot` 所在的群組並移除**
            for group in all_groups:
                if siot in group:
                    group.remove(siot)

    # **列出剩下的群組及其 locations**
    # print("\nFinal Groups and Their Locations:")
    # for i, group in enumerate(all_groups, 1):
    #     group_locations = set().union(*(s.locations for s in group))
    #     print(f"Group {i}: Locations {group_locations}")

    return rsma_groups, collaborative_groups

def replace_siot_in_group(g_j, unselected_siot_set):
    """
    替換 RSMA 群組 g_j 內的 SIoT，使 common message ratio 提高、beamforming 條件滿足、成本降低
    """
    best_group = g_j.copy()  # 目前最好的群組
    best_common_ratio = compute_common_message_ratio(g_j)  # 計算當前 common message ratio
    best_cost, C_bw, C_comp,coverage,common_message_data_size,beamforming_gain = compute_RSMA_cost(g_j, bandwidth_cost_unit=1, mec_cost_unit=0.5)  # 計算當前群組成本
    

    # 嘗試用 unselected_siot_set 內的 SIoT 替換 g_j 的部分成員
    for n_prime in unselected_siot_set:
        # 遍歷 g_j 的所有可能的子集 N_replace
        for r in range(1, len(g_j) + 1):  # 選擇 1 到 |g_j| 個成員替換
            for N_replace in combinations(g_j, r):
                # 確保 n_prime 覆蓋 N_replace 的所有監測位置
                if set(n_prime.locations) >= set.union(*(set(n.locations) for n in N_replace)):
                
                    # 建立新的群組
                    temp_group = g_j.copy()
                    for n in N_replace:
                        temp_group.remove(n)  # 移除原 SIoTs
                    temp_group.add(n_prime)  # 加入新的 SIoT
                
                    # 計算 common message ratio
                    new_common_ratio = compute_common_message_ratio(temp_group)

                    # 檢查 beamforming 條件
                    phase_terms = {siot.index: np.exp(1j * (2 * np.pi * siot.distance * np.cos(np.radians(siot.angle))) / 0.125) for siot in temp_group}
                    if not check_distributed_beamforming_constraint(temp_group, phase_terms):
                        continue  # 若不滿足 beamforming 條件則跳過

                    # 計算新群組的成本
                    new_cost, C_bw, C_comp,coverage,common_message_data_size,beamforming_gain = compute_RSMA_cost(temp_group,bandwidth_cost_unit=1,mec_cost_unit=0.5)

                    # 確保群組的 common message ratio 更高，且成本更低
                    if new_common_ratio > best_common_ratio and new_cost < best_cost:
                        best_group = temp_group.copy()
                        best_common_ratio = new_common_ratio
                        best_cost = new_cost
    
    return best_group  # 返回最佳替換後的群組



def compute_RSMA_cost(rsma_group, bandwidth_cost_unit, mec_cost_unit):
    """
    計算 collaborative group c 的總成本 (Bandwidth cost + Communication cost + Computation cost)
    collaborative_group: 目前的 collaborative group
    siots: 所有 SIoTs 資訊
    bandwidth_cost_unit: c_bw, bandwidth unit cost
    eta: collaborative parameter
    """
    if not rsma_group:
        return 0, 0, 0, 0, 0, 0  # 空群組直接返回 0

    C_bw=0
    C_comp=0
    B_g=0
    common_message_ratio=0
    common_message_data_size=0

    # 1. Bandwidth cost C_c^bw
    B_g,common_message_ratio,common_message_data_size,beamforming_gain=compute_rsma_required_bandwidth(rsma_group, T_threshold=10)
    C_bw = bandwidth_cost_unit * B_g
    

    # 2. Computation cost C_c^comp at MEC server
    coverage=len(set.union(*(set(siot.locations) for siot in rsma_group)))
    C_comp = mec_cost_unit * coverage


    # 總成本
    total_cost = C_bw + C_comp
    #print(f"Total:{total_cost}, Bandwith: {C_bw}, Communication:{C_com}, Computation:{C_comp}")
    return total_cost, C_bw, C_comp,coverage,common_message_data_size,beamforming_gain

def compute_rsma_required_bandwidth(rsma_group, T_threshold):
    """Bandwidth in Hz; PSD SINRs are independent of allocated bandwidth."""
    if not rsma_group:
        return 0, 0, 0, 0
    if len(rsma_group) == 1:
        bandwidth = compute_single_required_bandwidth(rsma_group, T_threshold)
        return bandwidth, 0, 0, 1

    total_locations = len(set.union(*(set(siot.locations) for siot in rsma_group)))
    mec_cpu = 2
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
    return bandwidth, common_message_ratio, common_message_data_size, beamforming_gain


def compute_single_required_bandwidth(rsma_group, T_threshold):
    """Single-device uplink using the full transmit PSD, returning Hz."""
    if not rsma_group:
        return 0.0
    total_locations = len(set.union(*(set(siot.locations) for siot in rsma_group)))
    time_budget = len(rsma_group) * T_threshold - total_locations / 2.0
    snr = sum(abs(siot.gain)**2 * siot.power for siot in rsma_group) / AWGNoise
    payload = payload_per_hz(total_locations * config['feature_bits'], rate_mapping(snr))
    return required_bandwidth(payload, time_budget)

def compute_common_message_ratio(g_j):
    
    if not g_j:  # 避免 g_j 為空集合
        return 0

    # 計算共用訊息比例 (Common Message Ratio)
    intersection_size = len(set.intersection(*(set(m.locations) for m in g_j)))
    union_size = len(set.union(*(set(m.locations) for m in g_j)))
    intersection=set.intersection(*(set(m.locations) for m in g_j))

    common_message_ratio = intersection_size / union_size if union_size > 0 else 0  # 避免除以零

    return common_message_ratio, intersection

def distributed_beamforming_gain(g_j, common_message_ratio):
    """檢查 Distributed Beamforming Constraint 是否滿足"""
    

    # 如果群組為空，直接回傳 False
    if not g_j:
        return False
    
    common_fraction, _ = power_split()
    wavelength=0.125
    phase_terms = {siot.index: np.exp(1j * (2 * np.pi * siot.distance * np.cos(np.radians(siot.angle))) / wavelength) for siot in g_j}
    


    # 計算分子 (Numerator)
    numerator = sum(
        np.sqrt(i.power * common_fraction) * abs(i.gain) * phase_terms[i.index] for i in g_j
    )

    # 計算分母 (Denominator)
    denominator = sum(np.sqrt(i.power * common_fraction) * abs(i.gain) for i in g_j)

    
    if denominator == 0:
        return 0

    # 計算 beamforming gain
    gamma = abs(numerator) / denominator

    return gamma  # 這裡的 gamma 需根據實際公式計算

def compute_private_rate_group(rsma_group, common_message_ratio):
    """
    計算 RSMA 群組內所有 SIoT 的 Private SINR
    
    參數：
    - rsma_group (list of SIoT): RSMA 群組內的所有 SIoT
    - noise_power (float): 熱噪聲功率 N_0
    
    返回：
    - rate_dict (dict): 每個 SIoT 的 spectral efficiency (bit/s/Hz)
    common_message_ratio is kept for API compatibility; power uses config.
    """
    _, private_fraction = power_split()
    rate_dict = {}
    # **Step 1: 先計算分子 (訊號功率)**
    power_list = []
    for siot in rsma_group:
        power_value = abs(siot.gain)**2 * (private_fraction * siot.power)
        power_list.append((siot, power_value))
    
    # **Step 2: 按照分子值排序（大到小）**
    power_list.sort(key=lambda x: x[1], reverse=True)

    # **Step 3: 按照排序計算 SINR**
    for siot, _ in power_list:
        numerator = abs(siot.gain)**2 * (private_fraction * siot.power)
        denominator = sum(abs(m.gain)**2 * (private_fraction * m.power) for m in rsma_group if m != siot) + AWGNoise  # 避免除零
        private_sinr = numerator / denominator if denominator > 0 else float('inf')

        rate_dict[siot.index] = rate_mapping(private_sinr)  # 映射到對應的速率

    return rate_dict

def satisfies_social_connectivity_constraint(siot, collaborative_group):
    """
    Checks if removing an SIoT maintains the social connectivity of its own distributed computing group.
    
    Args:
        siot: The SIoT object to check.
        siot_group: The specific distributed computing group (set of SIoTs) that contains `siot`.
    
    Returns:
        True if removing `siot` does not break the social connectivity of its group, otherwise False.
    """
    
    # If the group contains only one SIoT, removing it does not break connectivity
    if len(collaborative_group) == 1:
        return True  

    # Simulate removal of the SIoT
    remaining_siot_set = collaborative_group - {siot}

    # Start BFS or DFS from any remaining SIoT
    visited = set()
    start_siot = next(iter(remaining_siot_set))  # Pick any SIoT from the remaining set
    queue = [start_siot]
    
    while queue:
        current_siot = queue.pop()
        if current_siot in visited:
            continue
        visited.add(current_siot)
        
        # Add neighbors that are still in the remaining group
        for neighbor_idx in current_siot.neighbors:
            neighbor = next((s for s in remaining_siot_set if s.index == neighbor_idx), None)
            if neighbor and neighbor not in visited:
                queue.append(neighbor)

    # If all remaining SIoTs were visited, connectivity is preserved
    return len(visited) == len(remaining_siot_set)







if __name__ == '__main__':
    main()