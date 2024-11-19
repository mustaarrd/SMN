import random
import time
import threading
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import logging  # 로깅을 위한 모듈 추가
from matplotlib.lines import Line2D  # 범례를 위한 라이브러리
import sys
import os

# 현재 파일 기준으로 상대 경로 설정
sys.path.append(os.path.join(os.path.dirname(__file__), '../class'))
from Graph import Graph  # Graph 클래스를 가져옴
from Resource import Resource  # Resource 클래스를 가져옴

# 새로운 리소스 생성 및 시작 노드에 추가하는 함수
def generate_resource(resource_id, graph, node_resources):
    start_node = random.choice(list(graph.nodes()))
    end_node = random.choice(list(graph.nodes()))
    while end_node == start_node:
        end_node = random.choice(list(graph.nodes()))
    path = Graph.find_shortest_path(graph, start_node, end_node)
    if path is None:
        return None  # 경로가 없으면 생성하지 않음
    resource = Resource(resource_id, path, end_node)
    node_resources[start_node].append(resource)
    # 리소스 생성 정보 로깅
    print(f"Resource {resource_id} generated: Start={start_node}, End={end_node}, Path={path}")
    return resource

#리소스 이동 시뮬레이션 함수
def move_resources(simulation_graph, node_resources, finished_resources, lock, bottleneck_counter, total_distance, stop_event):
    time_step = 0  # 타임스텝 카운터
    max_time_steps = 100  # 최대 타임스텝 설정

    while time_step < max_time_steps:
        time_step += 1
        with lock:
            # 신호등 타이머 업데이트
            for node in simulation_graph.nodes():
                signal_cycle = simulation_graph.nodes[node]['signal_cycle']
                simulation_graph.nodes[node]['signal_timer'] = (simulation_graph.nodes[node]['signal_timer'] + 1) % signal_cycle

            # 각 간선의 현재 흐름 초기화
            for u, v in simulation_graph.edges():
                simulation_graph.edges[u, v]['current_flow'] = 0

            # 노드의 차량 흐름 업데이트
            for node in simulation_graph.nodes():
                simulation_graph.nodes[node]['traffic_flow'] = len(node_resources[node])

            # 병목 현상 체크 및 카운트
            bottleneck_nodes = []
            for node in simulation_graph.nodes():
                if simulation_graph.nodes[node]['traffic_flow'] >= simulation_graph.nodes[node]['capacity']:
                    bottleneck_nodes.append(node)
            bottleneck_counter['total'] += len(bottleneck_nodes)

            # 리소스 이동 처리
            for node in list(simulation_graph.nodes()):
                resources = node_resources[node]
                if resources:
                    resource = resources[0]  # FIFO 구조로 첫 번째 리소스 선택
                    if resource.finished:
                        continue
                    current_node = resource.current_node()
                    next_node = resource.next_node()
                    if next_node is None:
                        # 목적지에 도달한 경우
                        resource.finished = True
                        node_resources[node].remove(resource)
                        simulation_graph.nodes[node]['traffic_flow'] -= 1  # 노드의 리소스 수 감소
                        finished_resources.append(resource)
                        continue

                    # 이동하려는 간선 확인
                    edge_data = simulation_graph.get_edge_data(current_node, next_node)
                    if edge_data is None:
                        # 간선이 없으면 이동 불가
                        continue

                    # 신호등 상태 확인
                    signal_cycle = simulation_graph.nodes[current_node]['signal_cycle']
                    signal_timer = simulation_graph.nodes[current_node]['signal_timer']
                    if signal_timer < signal_cycle / 2:
                        can_move = False  # 빨간불
                    else:
                        can_move = True   # 초록불

                    # 이동하려는 간선의 용량과 현재 흐름 확인
                    capacity_edge = edge_data['capacity']
                    current_flow_edge = edge_data['current_flow']

                    # 다음 노드의 용량과 현재 리소스 수 확인
                    capacity_node = simulation_graph.nodes[next_node]['capacity']
                    traffic_flow_node = simulation_graph.nodes[next_node]['traffic_flow']

                    # 이동 조건 확인
                    if can_move and current_flow_edge < capacity_edge and traffic_flow_node < capacity_node:
                        # 이동
                        node_resources[node].remove(resource)
                        simulation_graph.nodes[node]['traffic_flow'] -= 1  # 현재 노드의 리소스 수 감소
                        resource.move()
                        next_node = resource.current_node()
                        node_resources[next_node].append(resource)
                        simulation_graph.nodes[next_node]['traffic_flow'] += 1  # 다음 노드의 리소스 수 증가
                        # 간선의 현재 흐름 증가
                        simulation_graph.edges[current_node, next_node]['current_flow'] += 1
                    else:
                        # 이동 불가 시 대기
                        pass
        time.sleep(1)  # 리소스 이동 주기

    # 시뮬레이션 완료 후 총 이동거리 계산
    total_distance['distance'] = sum(resource.distance_traveled for resource in finished_resources)

    # 종료 이벤트 설정
    stop_event.set()

# 시각화 업데이트 함수
def update_gui(frame, simulation_graph, node_resources, pos, lock):
    plt.cla()
    with lock:
        # 노드의 상태에 따라 색상 결정
        node_colors = []
        for node in simulation_graph.nodes():
            traffic_flow = simulation_graph.nodes[node]['traffic_flow']
            capacity = simulation_graph.nodes[node]['capacity']
            if traffic_flow >= capacity:
                node_colors.append('red')  # 용량이 가득 찬 경우
            else:
                node_colors.append('green')  # 여유가 있는 경우

        edge_colors = []
        for u, v in simulation_graph.edges():
            capacity = simulation_graph.edges[u, v]['capacity']
            edge_colors.append('black' if capacity > 5 else 'gray')

        # 노드 및 간선 그리기
        nx.draw_networkx_nodes(simulation_graph, pos, node_color=node_colors, node_size=500)
        nx.draw_networkx_edges(simulation_graph, pos, edge_color=edge_colors, arrows=True, arrowstyle='->', arrowsize=15)
        nx.draw_networkx_labels(simulation_graph, pos, font_size=10, font_color='white')

        # 노드 및 간선의 속성 표시
        for node in simulation_graph.nodes():
            x, y = pos[node]
            traffic_flow = simulation_graph.nodes[node]['traffic_flow']
            capacity = simulation_graph.nodes[node]['capacity']
            signal_timer = simulation_graph.nodes[node]['signal_timer']
            signal_cycle = simulation_graph.nodes[node]['signal_cycle']
            signal_state = "Green" if signal_timer >= signal_cycle / 2 else "Red"
            plt.text(x, y - 0.15, f"Flow: {traffic_flow}/{capacity}\nSignal: {signal_state}", horizontalalignment='center',
                     fontsize=8, color='black')

        for u, v in simulation_graph.edges():
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            capacity = simulation_graph.edges[u, v]['capacity']
            plt.text(mid_x, mid_y, f"Cap: {capacity}", horizontalalignment='center',
                     fontsize=8, color='purple')

        # 리소스 위치 및 목적지 표시
        for node in simulation_graph.nodes():
            resources = node_resources[node]
            if resources:
                x, y = pos[node]
                resource_info = [f"{r.resource_id}->{r.destination}" for r in resources]
                plt.text(x, y + 0.1, f"IDs: {', '.join(resource_info)}", horizontalalignment='center',
                         fontsize=8, color='blue')

        # 범례 추가
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='Available Node', markerfacecolor='green', markersize=10),
            Line2D([0], [0], marker='o', color='w', label='Full Capacity Node', markerfacecolor='red', markersize=10),
            Line2D([0], [0], color='black', lw=2, label='High Capacity Edge'),
            Line2D([0], [0], color='gray', lw=2, label='Low Capacity Edge'),
            Line2D([0], [0], marker='s', color='w', label='Signal: Green', markerfacecolor='none', markersize=10, markeredgecolor='black'),
            Line2D([0], [0], marker='s', color='w', label='Signal: Red', markerfacecolor='none', markersize=10, markeredgecolor='black', linestyle='--'),
        ]
        plt.legend(handles=legend_elements, loc='upper right')

        plt.title('Resource Flow Simulation')
        plt.axis('off')

# 주기적으로 새로운 리소스를 생성하는 함수
def resource_generator(simulation_graph, node_resources, lock, stop_event):
    resource_id = 0
    while not stop_event.is_set():
        with lock:
            # 리소스 발생 빈도 조정
            for _ in range(random.randint(2, 4)):
                resource = generate_resource(resource_id, simulation_graph, node_resources)
                if resource:
                    resource_id += 1
        time.sleep(1.5)  # 리소스 생성 주기


# 로깅 설정
logging.basicConfig(
    filename='simulation_log.txt',
    filemode='w',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 랜덤 시드 설정
seed_value = 42
np.random.seed(seed_value)

# 메인 코드
if __name__ == "__main__":
    # 그래프 생성 및 데이터 변환
    num_nodes = 10  # 노드 수
    num_edges = 15
    graph = Graph()
    graph.set_seed(seed_value)
    simulation_graph = graph.generate_random_graph(num_nodes, num_edges)
    node_resources = {node: [] for node in simulation_graph.nodes()}

    finished_resources = []  # 목적지에 도달한 리소스 목록
    bottleneck_counter = {'total': 0}  # 병목 현상 발생 횟수 저장
    total_distance = {'distance': 0}  # 총 이동거리 저장
    lock = threading.Lock()  # 스레드 동기화를 위한 락
    stop_event = threading.Event()  # 스레드 종료를 위한 이벤트

    # 그래프 구조 로깅
    logging.info("Initial Graph Structure:")
    for node in simulation_graph.nodes():
        capacity = simulation_graph.nodes[node]['capacity']
        signal_cycle = simulation_graph.nodes[node]['signal_cycle']
        logging.info(f"Node {node}: Capacity={capacity}, Signal Cycle={signal_cycle:.2f}")
    for u, v in simulation_graph.edges():
        capacity = simulation_graph.edges[u, v]['capacity']
        logging.info(f"Edge {u}->{v}: Capacity={capacity}")

    # 그래프 시각화를 위한 위치 계산 (노드 간격 증가)
    pos = nx.spring_layout(simulation_graph, k=2, scale=2, seed=seed_value)  # 랜덤 시드 추가

    # 리소스 이동 스레드 시작
    move_thread = threading.Thread(target=move_resources, args=(simulation_graph, node_resources, finished_resources, lock, bottleneck_counter, total_distance, stop_event))
    move_thread.start()

    # 리소스 생성 스레드 시작 (데몬 스레드가 아님)
    generator_thread = threading.Thread(target=resource_generator, args=(simulation_graph, node_resources, lock, stop_event))
    generator_thread.start()

    # 애니메이션 실행
    fig = plt.figure(figsize=(12, 8))
    ani = FuncAnimation(fig, update_gui, fargs=(simulation_graph, node_resources, pos, lock), interval=1000)
    plt.show()

    # 리소스 이동 및 생성 스레드가 종료될 때까지 대기
    move_thread.join()
    generator_thread.join()

    # 시뮬레이션 종료 후 총합계 출력 및 로깅
    total_resources_moved = len(finished_resources)
    total_bottlenecks = bottleneck_counter['total']
    total_movement_distance = total_distance['distance']

    print("\nSimulation Completed.")
    print(f"Total Time Steps: 100")
    print(f"Total Resources Moved and Released: {total_resources_moved}")
    print(f"Total Movement Distance of Resources: {total_movement_distance}")
    print(f"Total Bottleneck Occurrences: {total_bottlenecks}")

    # 로깅
    logging.info("\nSimulation Completed.")
    logging.info(f"Total Time Steps: 100")
    logging.info(f"Total Resources Moved and Released: {total_resources_moved}")
    logging.info(f"Total Movement Distance of Resources: {total_movement_distance}")
    logging.info(f"Total Bottleneck Occurrences: {total_bottlenecks}")
