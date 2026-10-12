import numpy as np
import networkx as nx
import json
from config import TX_POWER_PSD



class SIoT:
    def __init__(self, index):
        self.index = index
        self.distance = np.random.randint(10, 300)  # 隨機距離
        self.angle = np.random.randint(0, 360)  # 隨機角度
        self.neighbors = set()  # 社交網絡連接
        self.locations = []  # 監控位置清單
        self.num_locations=0
        self.cpu_frequency=1
        self.d2d_rate=1
        self.power = TX_POWER_PSD  # Transmit PSD in W/Hz
        self.gain = self.channel_gain(self.distance)  # 計算信道增益
    def __hash__(self):
        return hash(self.index)

    def __eq__(self, other):
        return isinstance(other, SIoT) and self.index == other.index    
    def assign_locations(self, universal_locations, uncovered_locations):
        """ 確保每個 SIoT 至少負責一個監控點，並分配額外的監控點 """
        if uncovered_locations:
            must_cover = uncovered_locations.pop()  # 取出一個未被覆蓋的點
            self.locations.append(must_cover)

        # **額外分配監控點，確保無重複**
        num_locations = np.random.randint(2, 7)  # 每個 SIoT 監控的數量

        available_locations = list(set(universal_locations) - set(self.locations))
        
        if available_locations:
            additional_locations = np.random.choice(
                available_locations,
                #list(set(universal_locations) - set(self.locations)),  # 避免重複
                size=min(num_locations - len(self.locations), len(set(universal_locations) - set(self.locations))), 
                replace=False
            ).tolist()

            self.locations.extend(additional_locations)
        if not self.locations:
            print(f"Warning: SIoT {self.index} has no locations assigned!")
    
    def add_neighbor(self, neighbor_index):
        """Add a neighbor to the SIoT social network."""
        self.neighbors.add(neighbor_index)

    
    def get_neighbors(self):
        """Return the list of neighbors."""
        return list(self.neighbors)
    
    
    def channel_gain(self, distance):
        """計算信道增益"""
        path_loss = 50 + 15 * np.log10(distance) 
        loss = 10 ** (path_loss / 10)

        shadowing_db = np.random.normal(0, 1)  # 陰影衰落
        shadowing = 10 ** (shadowing_db / 10)

        small_scale_fading = (np.random.normal(0, 1) + 1j * np.random.normal(0, 1)) / np.sqrt(2.0)  # CN(0,1)
        
        h_k = np.abs(small_scale_fading) / np.sqrt(loss * shadowing)  # Amplitude

        return h_k

def build_siot_social_network_barabasi_albert(siots):
    """
    Generate an SIoT social network using the Barabási–Albert model.
    
    Parameters:
    - num_siot: Number of SIoTs (nodes).
    - m: Number of edges a new SIoT connects to existing SIoTs.
    
    Returns:
    - siots: List of SIoT objects with assigned social neighbors.
    """
    num_siot = len(siots)
    if num_siot < 2:
        return  # 若SIoT數量不足，不建立網絡
    m = max(1, min(num_siot - 1, np.random.randint(0, 4)))

    # Create a scale-free network
    G = nx.barabasi_albert_graph(n=num_siot, m=m)

    # **清空每個SIoT的鄰居（避免重複添加）**
    for siot in siots:
        siot.neighbors.clear()

    # **設定鄰居關係**
    for i in range(num_siot):
        siots[i].neighbors.update(G.neighbors(i))
    
    return siots

def get_siot(num_siot=10,num_locations_total=15):
    # 參數設定 設定 SIoT 數量 設定監控點的總數量
    
    universal_locations = [f"l_{i}" for i in range(1, num_locations_total + 1)]  # 生成 L


    # **確保所有監控位置至少分配給一個 SIoT**
    uncovered_locations = set(universal_locations)
    siots = [SIoT(i) for i in range(num_siot)]  # 先初始化 SIoT，不處理監控點

    # 為每個 SIoT 指派監控位置
    for siot in siots:
        siot.assign_locations(universal_locations, uncovered_locations)

    
    # **確保所有監控點都被覆蓋**
    covered_locations = set(loc for siot in siots for loc in siot.locations)
    missing_locations = set(universal_locations) - covered_locations
    if missing_locations:
        # **強制補充未覆蓋的監控點**
        for loc in missing_locations:
            siots[np.random.randint(0, num_siot)].locations.append(loc)

        # **再次檢查**
        covered_locations = set(loc for siot in siots for loc in siot.locations)
        assert covered_locations == set(universal_locations), f"監控位置未完全覆蓋！缺少：{set(universal_locations) - covered_locations}"
    
    for siot in siots:
        siot.num_locations=len(siot.locations)

    build_siot_social_network_barabasi_albert(siots)

    # for siot in siots:
    #     print(f"SIoT {siot.index}: Locations: {siot.locations}, Neighbors: {siot.get_neighbors()}, Distance: {siot.distance:.2f}, Angle: {siot.angle:.2f}, Gain: {siot.gain:.10f}")


    return siots, num_locations_total, universal_locations



# def parse_siots_from_json(file_path="siots.json"):
#     with open(file_path, "r", encoding="utf-8") as f:
#         siot_data = json.load(f)

#     siots = []
#     for data in siot_data["siots"]:
#         siot = SIoT(data["index"])  # 只傳入 index，其他屬性後續設定
#         siot.distance = data["distance"]
#         siot.angle = data["angle"]
#         siot.neighbors = set(data["neighbors"])  # 轉回 set
#         siot.locations = data["locations"]  # 轉回 set
#         siot.gain = data["gain"]
#         siot.num_locations = data["num_locations"]
#         siot.cpu_frequency = data["cpu_frequency"]
#         siot.d2d_rate = data["d2d_rate"]
#         siot.power = data["power"]
#         siots.append(siot)


#     return siots

# def parse_requests_from_json(file_path="requests.json"):
#     """從 requests.json 讀取多個 requests"""
#     with open(file_path, "r", encoding="utf-8") as f:
#         request_data = json.load(f)
    
#     requests = {}
#     for request in request_data["requests"]["requests"]:  # 這邊多取一層
#         request_id = request["request_id"]
#         request_locations = set(request["request_locations"])  # 轉回 set
#         requests[request_id] = request_locations  # 存入字典
    
#     return requests

def parse_siots_from_json(file_path="siots.json",batch_id=0):
    with open(file_path, "r", encoding="utf-8") as f:
        siot_data = json.load(f)
    # 找到對應的 batch
    batch_data = next((batch for batch in siot_data["iterations"] if batch["batch_id"] == batch_id), None)
    
    if batch_data is None:
        raise ValueError(f"Batch ID {batch_id} not found in JSON file.")

    siots = []
    for data in batch_data["siots"]:
        siot = SIoT(data["index"])  # 假設 SIoT 類別有 index 屬性
        siot.distance = data["distance"]
        siot.angle = data["angle"]
        siot.neighbors = set(data["neighbors"])  # 轉回 set
        siot.locations = data["locations"]  # 轉回 set
        siot.gain = data["gain"]
        siot.num_locations = data["num_locations"]
        siot.cpu_frequency = data["cpu_frequency"]
        siot.d2d_rate = data["d2d_rate"]
        siot.power = data["power"]
        siots.append(siot)


    return siots

def parse_requests_from_json(file_path="requests.json", batch_id=0):
    """從 requests.json 讀取多個 requests"""
    with open(file_path, "r", encoding="utf-8") as f:
        request_data = json.load(f)

    # 找到對應的 batch
    batch_data = next((batch for batch in request_data["iterations"] if batch["batch_id"] == batch_id), None)
    
    if batch_data is None:
        raise ValueError(f"Batch ID {batch_id} not found in JSON file.")

    requests = {}
    for request in batch_data["requests"]:
        request_id = request["request_id"]
        request_locations = set(request["request_locations"])
        requests[request_id] = request_locations

    return requests



   
        





        


