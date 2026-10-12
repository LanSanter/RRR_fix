import numpy as np
from input_settings import SIoT
from phase1 import phase1
from config import AWGNoise, config, payload_per_hz, rate_mapping, required_bandwidth








def phase2(siot_set,unselected_siot_set):
    #siot_set,unselected_siot_set =phase1()
    collaborative_groups=[]
    rsma_groups=[]
    unassigned_siot_set=set()
    print("From Phase1 Selected SIoTs:", [siot.index for siot in siot_set])
    wavelength=0.125

    unassigned_siot_set, collaborative_groups = initial_classification(siot_set, unassigned_siot_set, collaborative_groups)
    unassigned_siot_set, collaborative_groups = refine_collaborative_groups( unassigned_siot_set,collaborative_groups)
    rsma_groups=rsma_grouping(rsma_groups,unassigned_siot_set,wavelength)

    print(f"End Phase 2--------------------------------------------------------------")

    return rsma_groups, collaborative_groups,unselected_siot_set



def initial_classification(siot_set, unassigned_siot_set, collaborative_groups):
    """
    執行初始分類，將 SIoTs 根據社交關係分成 Collaborative Groups。
    若無更多 SIoTs 有社交連結，則轉移至 RSMA Grouping。

    參數:
    - siot_set: (set) 仍未分類的 SIoT 集合
    - unassigned_siot_set: (set) 用於儲存所有未分類的 SIoT
    - collaborative_groups: (list) 儲存已建立的 Collaborative Groups
    """
    
    while siot_set:  # 直到所有 SIoT 被分配
        # 選擇擁有最高社交度的 SIoT
        highest_degree_siot = max(siot_set, key=lambda siot: len(siot.neighbors & {s.index for s in siot_set}))
        #print(f"SIoT{highest_degree_siot.index} has {highest_degree_siot.neighbors & {s.index for s in siot_set}}")
        
        #Check 是否還有social degree
        if len(highest_degree_siot.neighbors & {s.index for s in siot_set}) == 0:
            #print("No more SIoTs with social relations. Moving to RSMA Grouping.")
            unassigned_siot_set.update(siot_set)  # 將所有剩餘的 SIoT 存入未分配集合
            siot_set.clear()  # 清空 siot_set
            break  # 跳出迴圈

        # 建立新群組
        new_collaborative_group = {highest_degree_siot}  
        siot_set.remove(highest_degree_siot)

        # 在該群組內尋找所有相鄰的 SIoT，並加入群組
        queue = [highest_degree_siot]  # BFS

        while queue:
            current_siot = queue.pop(0)  # 取出當前 SIoT

            for neighbor_idx in current_siot.neighbors:
                neighbor_siot = next((siot for siot in siot_set if siot.index == neighbor_idx), None)

                if neighbor_siot:  # 確保鄰居還在 S 裡
                    new_collaborative_group.add(neighbor_siot)
                    siot_set.remove(neighbor_siot)
                    queue.append(neighbor_siot)

        # 群組建立完畢，加入 collaborative_groups
        collaborative_groups.append(new_collaborative_group)

    print("Collaborative Groups:")
    for group in collaborative_groups:
        print([siot.index for siot in group])
        # total_cost=compute_collaborative_cost(group, bandwidth_cost_unit=1, eta=1)
        # print(f"cost:{total_cost}")
    # print("Unassigned SIoTs:")
    # print([u_siot.index for u_siot in unassigned_siot_set])
    return unassigned_siot_set, collaborative_groups

def refine_collaborative_groups( unassigned_siot_set,collaborative_groups):
    """
    根據 coverage-cost ratio 進一步篩選與最佳化 collaborative groups。

    參數:
    - collaborative_groups: (list) 初始的 collaborative groups
    - unassigned_siot_set: (set) 存儲未確認分組的 SIoT
    - compute_collaborative_cost: (function) 用於計算 collaborative cost 的函式

    回傳:
    - final_groups: (list) 經過優化的 collaborative groups
    """
    
    confirmed_groups = []
    unconfirmed_groups = collaborative_groups.copy()

    for group in unconfirmed_groups:
        check_collaborative = set()  # 初始化新的最佳群組
        all_neighbors=set()
        # 創建一個對照表，將 SIoT 的 index 對應到 SIoT 物件
        group_dict = {siot.index: siot for siot in group}
        # Step 2: 找出與其他 SIoTs 最少重疊的 SIoT
        min_overlap_siot = min(group, key=lambda n: sum(len(set(n.locations) & set(m.locations) ) for m in group if m != n))
        # 初始化 
        check_collaborative.add(min_overlap_siot)
        group.remove(min_overlap_siot)
        # 選擇最佳 SIoT
        while group:
            # 確保 `best_siot` 來自目前 check_collaborative 內 SIoT 的 neighbor           
            # 從 check_collaborative 內的 SIoT 取得鄰居索引，並轉換回 SIoT 物件
            neighbor_indices = set.union(*(set(siot.neighbors) for siot in check_collaborative)) & set(group_dict.keys())
            all_neighbors = {group_dict[i] for i in neighbor_indices}  # 轉換回 SIoT 物件
            print(f"Current check_collaborative: {[siot.index for siot in check_collaborative]}")
            #print(f"All_neighbors (only in group): {[n.index for n in all_neighbors]}")
            best_siot = min(
                all_neighbors, 
                key=lambda n: len(set(n.locations) & set.union(*[set(m.locations) for m in check_collaborative]))
                    / len(set(n.locations) | set.union(*[set(m.locations) for m in check_collaborative]))
                )
            # Step 4: 計算 coverage-cost ratio
            coverage = len(set.union(*[set(m.locations) for m in check_collaborative | {best_siot}]))
            cost, C_bw, C_com, C_comp = compute_collaborative_cost(check_collaborative, bandwidth_cost_unit=1, eta=1)            
            current_ratio = coverage / cost if cost != 0 else 0
            if current_ratio > (len(set.union(*[set(m.locations) for m in check_collaborative])) / cost):
                check_collaborative.add(best_siot)
                group.remove(best_siot)
            else:
                break  # 停止加入
        # Step 5: 確認新的 collaborative group
        if len(check_collaborative) > 1:  # 只有超過 1 個 SIoT 才能成為 collaborative group
            confirmed_groups.append(check_collaborative)
        else:
            unassigned_siot_set.update(check_collaborative)
        unassigned_siot_set.update(group)
    # 更新 collaborative_groups
    collaborative_groups = confirmed_groups
    print("Collaborative Groups:")
    for group in collaborative_groups:
        print([siot.index for siot in group])
    print(f"Unassigned SIoTs: {[siot.index for siot in unassigned_siot_set]}")

    # 更新 collaborative_groups
    return unassigned_siot_set,confirmed_groups

def rsma_grouping(rsma_groups,unassigned_siot_set,wavelength):
    """
    根據 SPCI、Distributed Beamforming Constraint 和 RSMA Grouping Constraint 來分類 RSMA 群組。

    參數:
    - unassigned_siot_set: (set) 未分配的 SIoT 集合
    - spci: (function) 用於計算 SPCI 的函式
    - check_distributed_beamforming_constraint: (function) 用於檢查 Distributed Beamforming Constraint
    - check_rsma_grouping_constraint: (function) 用於檢查 RSMA Grouping Constraint
    - wavelength: (float) 計算相位時所需的波長

    回傳:
    - rsma_groups: (list) 已經分組的 RSMA 群組
    """


    # 計算所有 SIoT 的相位
    phase_terms = {siot.index: np.exp(1j * (2 * np.pi * siot.distance * np.cos(np.radians(siot.angle))) / wavelength) for siot in unassigned_siot_set}

    # 選擇未分配的 SIoTs，根據空間相位對齊來創建新群組
    while unassigned_siot_set:
        # 計算每個 SIoT 的平均空間相位對齊
        best_siot = max(
            unassigned_siot_set,
            key=lambda n: np.abs(phase_terms[n.index] + sum(phase_terms[m.index] for m in unassigned_siot_set if m != n)) / len(unassigned_siot_set)
        )
        
        

        # 創建新 RSMA 群組
        new_rsma_group = {best_siot}
        rsma_groups.append(new_rsma_group)

        # 移除已分配的 SIoT
        unassigned_siot_set.remove(best_siot)

        print(f"New RSMA group created with SIoT {best_siot.index}")
        # highest= np.abs(phase_terms[best_siot.index] + sum(phase_terms[m.index] for m in unassigned_siot_set if m != best_siot)) / len(unassigned_siot_set)
        # print(f"Highest phase alighment:{highest}")

        # 選擇適合的 SIoT 加入此群組
        remaining_siot_set = unassigned_siot_set.copy()

        while remaining_siot_set:
            # 按照 SPCI 選擇最佳 SIoT
            sorted_siot_list = sorted(
                remaining_siot_set, 
                key=lambda n: max(spci(n, g_j, phase_terms) for g_j in rsma_groups) if rsma_groups else 0, reverse=True
            )
            #print(f"sorted_list {[i.index for i in sorted_siot_list]}")

            # 嘗試將符合 Distributed Beamforming Constraint 的 SIoT 加入 RSMA 群組
            selected_siot = None
            for candidate_siot in sorted_siot_list:
                for g_j in rsma_groups:  # 遍歷所有現有 RSMA 群組
                    # 預先測試加入後的群組是否滿足條件
                    if check_distributed_beamforming_constraint(g_j, phase_terms, candidate_siot) and check_rsma_grouping_constraint(g_j, candidate_siot):
                        selected_siot = candidate_siot
                        target_group = g_j
                        break  # 找到符合條件的 SIoT，結束迴圈
                
            # 如果沒有找到符合條件的 SIoT，則創建新群組
            if selected_siot is None:
                new_group = {sorted_siot_list[0]}  # 選擇最高 SPCI 的 SIoT 創建新群組
                rsma_groups.append(new_group)
                remaining_siot_set.remove(sorted_siot_list[0])
                unassigned_siot_set.remove(sorted_siot_list[0])
                print(f"New RSMA group created with SIoT {sorted_siot_list[0].index}")
                # new_highest= np.abs(phase_terms[sorted_siot_list[0].index] + sum(phase_terms[m.index] for m in unassigned_siot_set)) / (len(unassigned_siot_set)+1)
                # print(f"New Highest Phase:{new_highest}")
                # new_highest=new_highest*(len(unassigned_siot_set)+1)
                # print(f"New Highest Phase 分子:{new_highest}")
                continue

            # 加入 RSMA 群組
            target_group.add(selected_siot)
            remaining_siot_set.remove(selected_siot)
            unassigned_siot_set.remove(selected_siot)

            print(f"SIoT {selected_siot.index} added to RSMA group [{', '.join(str(i.index) for i in new_rsma_group)}]")

    return rsma_groups


def spci(siot, group_j, phase_terms):
    """
    計算 Spatial Phase and Common Message Indicator (SPCI)
    
    參數:
    - siot: 當前考慮加入 RSMA 群組 g_j 的 SIoT
    - group_j: 目前 RSMA 群組 (set of SIoTs)
    - phase_terms: 事先計算好的 phase term 字典 {siot.index: 相位值}

    回傳:
    - SPCI 值
    """
    if not group_j:
        return 0  # 如果群組為空，避免錯誤


    # 計算空間相位對齊 (Spatial Phase Alignment)
    spatial_phase = abs(sum(phase_terms[m.index] for m in group_j) + phase_terms[siot.index]) / (len(group_j) + 1)

    # 計算共用訊息比例 (Common Message Ratio)
    intersection_size = len(set(siot.locations) & set.intersection(*(set(m.locations) for m in group_j)))
    union_size = len(set(siot.locations) | set.union(*(set(m.locations) for m in group_j)))

    common_message_ratio = intersection_size / union_size if union_size > 0 else 0  # 避免除以零

    
    #print(f"SPCI:{spatial_phase}*{common_message_ratio}")
    
    # SPCI 值
    return spatial_phase * common_message_ratio

def check_distributed_beamforming_constraint(g_j, phase_terms, n=None):
    """檢查 Distributed Beamforming Constraint 是否滿足"""
    gamma_threshold=0.8

    # 如果群組為空，直接回傳 False
    if not g_j:
        return False
    # 計算群組內的 Common Message Ratio
    if n is None:
        intersection_size = len(set.intersection(*(set(m.locations) for m in g_j)))
        union_size = len(set.union(*(set(m.locations) for m in g_j)))  # 群內的 union size 即為所有組員的覆蓋範圍
    else:
        intersection_size = len(set(n.locations) & set.intersection(*(set(m.locations) for m in g_j)))
        union_size = len(set(n.locations) | set.union(*(set(m.locations) for m in g_j)))

    common_message_ratio = intersection_size / union_size if union_size > 0 else 0  # 避免除零


    # 計算分子 (Numerator)
    numerator = sum(
        np.sqrt(i.power * common_message_ratio) * abs(i.gain) * phase_terms[i.index] for i in g_j
    )

    if n is not None:  # 若 n 存在，則加入 n 的 beamforming 計算
        numerator += np.sqrt(n.power * common_message_ratio) * abs(n.gain) * phase_terms[n.index]

    # 計算分母 (Denominator)
    denominator = sum(np.sqrt(i.power * common_message_ratio) * abs(i.gain) for i in g_j)

    if n is not None:  # 若 n 存在，則加入 n 的分母計算
        denominator += np.sqrt(n.power * common_message_ratio) * abs(n.gain)
    
    if denominator == 0:
        return False

    # 計算 beamforming gain
    gamma = abs(numerator) / denominator
    if n is not None:
        print(f"SIoT {n.index} and Group [{', '.join(str(i.index) for i in g_j)}]'s Gamma: {gamma}")
    return (gamma>= gamma_threshold)  # 這裡的 gamma 需根據實際公式計算

def check_rsma_grouping_constraint(g_j,n=None):
    """檢查 RSMA Grouping Constraint 是否滿足"""
    grouping_threshold=5
    new_group_size= len(g_j)+(1 if n is not None else 0)
    return (new_group_size <= grouping_threshold)  # 這裡的條件需根據實際公式計算


def compute_collaborative_cost(collaborative_group, bandwidth_cost_unit, eta):
    """
    計算 collaborative group c 的總成本 (Bandwidth cost + Communication cost + Computation cost)
    collaborative_group: 目前的 collaborative group
    siots: 所有 SIoTs 資訊
    bandwidth_cost_unit: c_bw, bandwidth unit cost
    eta: collaborative parameter
    """
    if not collaborative_group:
        return 0, 0, 0, 0 # 空群組直接返回 0

    C_bw=0
    C_com=0
    C_comp=0

    # 1. Bandwidth cost C_c^bw
    C_bw = bandwidth_cost_unit * compute_collaborative_required_bandwidth(collaborative_group, T_threshold=10)
    
    # 2. Communication cost C_c^com
    num_siot = len(collaborative_group)  # 群組內 SIoT 數量
    if num_siot == 1:
        C_com = 0  # 只有一個設備，無需通信
    else:
        C_com = eta * (num_siot - 1)  # 可選用 |V|-1 (確保連通) 或 Edge 數
    # C_com = 0
    # for siot_n in collaborative_group:
    #     for siot_m in collaborative_group:
    #         if siot_n != siot_m and siot_n.index in siot_m.neighbors:  # 確保有社交連結
    #             C_com += eta * siot_n.communication_cost[siot_m.index]  # 兩者之間的通信成本

    # 3. Computation cost C_c^comp
    C_comp = eta *(sum(abs(len(siot.locations)) for siot in collaborative_group) ) # Covered Locations 絕對值計算

    # 總成本
    total_cost = C_bw + C_com + C_comp
    #print(f"Total:{total_cost}, Bandwith: {C_bw}, Communication:{C_com}, Computation:{C_comp}")
    return total_cost, C_bw, C_com, C_comp

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

    T_d2d = ((len(critical_siot.locations) * 0.2) / critical_siot.d2d_rate) * longest_path_length
    
    # **3. 計算上行傳輸時間**
    
    #剩餘時間 (也扣過整合時間)
    T_rest=(len(collaborative_group) * T_threshold) - T_comp - T_d2d - ((sum(len(siot.locations) * 0.2 for siot in collaborative_group) / representative_siot.cpu_frequency))
    
    #Representative SIoT data rate
    representative_bits = len(set.union(*(set(siot.locations) for siot in collaborative_group))) * 0.2 * config['feature_bits']
    gamma_representative_siot= (abs(representative_siot.gain) ** 2 * representative_siot.power) / AWGNoise
    
    B_u_c = required_bandwidth(payload_per_hz(representative_bits, rate_mapping(gamma_representative_siot)), T_rest)
    
    print(f"Bandwidth:{B_u_c}")
    return B_u_c


def find_longest_path(start_siot, group):
    """
    使用 DFS (Stack-based) 找最長的路徑，並且只考慮 group 內的 SIoT
    """

    # 建立索引到 SIoT 物件的映射
    siot_dict = {siot.index: siot for siot in group}
    group_indexes = set(siot_dict.keys())  # 確保 `group` 是索引集合


    # for siot in group:
    #     print(f"find_longest_path phase SIoT: {siot.index}, Neighbors in Group: {[n for n in siot.neighbors]}")
    #     print(f"find_longest_path phase SIoT: {siot.index}, Neighbors in Group: {[n for n in siot.neighbors if n in siot_dict]}")
        

    stack = [(start_siot.index, 0, start_siot.index)]  # (當前節點索引, 當前深度, 來源索引)
    longest_path_length = 0
    visited = set()
    critical_siot_index = start_siot.index  # 預設為代表性 SIoT

    #print(f"Starting DFS from SIoT {start_siot.index}")

    while stack:
        current_index, depth, source_index = stack.pop()
        
        # **確保取出的 `current_siot` 是 `SIoT` 物件**
        current_siot = siot_dict[current_index]
        
        if current_index in visited:
            continue
        visited.add(current_index)

        # 更新最長路徑
        if depth > longest_path_length:
            longest_path_length = depth
            critical_siot_index = source_index

        # **這裡不改變 SIoT，只是查看他的 neighbors 哪些在 group 內**
        valid_neighbors = [n for n in current_siot.neighbors if n in group_indexes]
        #print(f"Checking SIoT {current_index}, Depth: {depth}, Neighbors in Group: {valid_neighbors}")

        # 遍歷當前節點的鄰居，確保只考慮 group 內的節點
        for neighbor_index in valid_neighbors:
            if neighbor_index not in visited:
                stack.append((neighbor_index, depth + 1, current_index))  # 繼續探索

    #print(f"SIoT {critical_siot_index} has the longest path: {longest_path_length}")

    return longest_path_length, siot_dict[critical_siot_index]  # 返回對應的 SIoT 物件

if __name__ == '__main__':
    main()