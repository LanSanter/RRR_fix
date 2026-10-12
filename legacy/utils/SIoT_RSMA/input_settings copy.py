import numpy as np
import networkx as nx
import math
import random
from random import randint
# from config import config
# from utils import *
# from RSMA import *
# from models import MLP
#from options import args_parser


class SIoT:
    def __init__(self, index, universal_locations=None, uncovered_locations=None):
        self.index = index
        self.distance = np.random.randint(10, 300)  # 隨機距離
        self.angle = np.random.randint(0, 360)  # 隨機角度
        self.neighbors = set()  # Social network connections
        

        # 確保每個 SIoT 至少負責一個監控點
        self.locations = []
        if universal_locations is not None and uncovered_locations is not None:
            if uncovered_locations:
                must_cover = uncovered_locations.pop()  # 取出一個未被覆蓋的點
                self.locations.append(must_cover)

            # **額外分配監控點，確保無重複**
            num_locations = np.random.randint(2, 7)  # 每個 SIoT 監控的數量
            additional_locations = np.random.choice(
                list(set(universal_locations) - set(self.locations)),  # 避免重複
                size=min(num_locations - len(self.locations), len(set(universal_locations) - set(self.locations))), 
                replace=False
            ).tolist()

            self.locations.extend(additional_locations)
        #self.locations = np.random.choice(universal_locations, size=num_locations, replace=False).tolist()

        # 計算信道增益
        self.gain = self.channel_gain(self.distance)
    def assigned_locations(self, index, universal_locations, uncovered_locations):
        if uncovered_locations:
            must_cover = uncovered_locations.pop()  # 取出一個未被覆蓋的點
            self.locations.append(must_cover)

        # **額外分配監控點，確保無重複**
        num_locations = np.random.randint(2, 7)  # 每個 SIoT 監控的數量
        additional_locations = np.random.choice(
            list(set(universal_locations) - set(self.locations)),  # 避免重複
            size=min(num_locations - len(self.locations), len(set(universal_locations) - set(self.locations))), 
            replace=False
        ).tolist()

        self.locations.extend(additional_locations)
        #self.locations = np.random.choice(universal_locations, size=num_locations, replace=False).tolist()
    
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

def build_siot_social_network_barabasi_albert(num_siot, universal_locations=None, uncovered_locations=None):
    """
    Generate an SIoT social network using the Barabási–Albert model.
    
    Parameters:
    - num_siot: Number of SIoTs (nodes).
    - m: Number of edges a new SIoT connects to existing SIoTs.
    
    Returns:
    - siots: List of SIoT objects with assigned social neighbors.
    """
    m = np.random.randint(0, 3)

    # Create a scale-free network
    G = nx.barabasi_albert_graph(n=num_siot, m=m)

    # Initialize SIoT objects
    siots = [SIoT(i,universal_locations, uncovered_locations) for i in range(num_siot)]

    # Assign neighbors based on the BA graph
    for i in range(num_siot):
        for neighbor in G.neighbors(i):
            siots[i].add_neighbor(neighbor)
    
    return siots

def main():
    # 參數設定
    num_siot = 10  # 設定 SIoT 數量
    num_locations_total = 15  # 設定監控點的總數量
    if 'universal_locations' not in globals():
        universal_locations = [f"l_{i}" for i in range(1, num_locations_total + 1)] #生成L

    # **確保所有監控位置至少分配給一個 SIoT**
    uncovered_locations = set(universal_locations)
    siots = []

    for i in range(num_siot):
        siot = SIoT((i+1), universal_locations, uncovered_locations)
        siots.append(siot)
    
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
    
    siots = build_siot_social_network_barabasi_albert(num_siot, universal_locations, uncovered_locations)

    # 顯示結果
    for siot in siots:
        print(f"SIoT {siot.index}: Locations: {siot.locations}, Neighbors: {siot.get_neighbors()}, Distance: {siot.distance:.2f}, Angle: {siot.angle:.2f}, Gain: {siot.gain:.10f}")


if __name__ == '__main__':
    main()


   
        





        


