# 클래스 정의
class Resource:
    """
    리소스(예: 차량) 클래스 정의
    """
    def __init__(self, resource_id, path, destination):
        self.resource_id = resource_id
        self.path = path  # 이동해야 할 노드의 리스트
        self.current_index = 0  # 현재 위치한 노드의 인덱스 (path 내에서)
        self.finished = False  # 목적지 도달 여부
        self.destination = destination  # 목적지 노드
        self.distance_traveled = 0  # 이동한 거리 누적

    def current_node(self):
        return self.path[self.current_index]

    def next_node(self):
        if self.current_index + 1 < len(self.path):
            return self.path[self.current_index + 1]
        else:
            return None

    def move(self):
        self.current_index += 1
        self.distance_traveled += 1  # 이동 거리 증가