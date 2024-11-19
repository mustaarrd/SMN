import random
import networkx as nx

class Graph:
    def __init__(self):
        self.seed_value = None  # 초기 시드 값

    def set_seed(self, seed_value):
        """
        랜덤 시드 설정 함수
        """
        self.seed_value = seed_value
        random.seed(seed_value)
    # 함수 정의
    def generate_random_graph(self, num_nodes, num_edges):
        """
        랜덤 그래프 생성 함수
        """
        G = nx.gnm_random_graph(num_nodes, num_edges, directed=True, seed=self.seed_value)
        for node in G.nodes():
            G.nodes[node]['traffic_flow'] = 0  # 현재 노드의 리소스 수
            G.nodes[node]['capacity'] = random.randint(5, 10)  # 노드의 용량 추가
            G.nodes[node]['signal_cycle'] = random.uniform(30, 120)
            G.nodes[node]['signal_timer'] = random.uniform(0, G.nodes[node]['signal_cycle'])
        for u, v in G.edges():
            G.edges[u, v]['capacity'] = random.randint(5, 10)  # 간선 용량을 작게 설정하여 병목 유발
            G.edges[u, v]['current_flow'] = 0  # 현재 간선을 통과하는 리소스 수
        return G

    def find_shortest_path(graph, start_node, end_node):
        """
        두 노드 간의 최단 경로를 찾는 함수
        """
        try:
            path = nx.shortest_path(graph, source=start_node, target=end_node)
            return path
        except nx.NetworkXNoPath:
            return None