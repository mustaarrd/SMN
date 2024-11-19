
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from GNN_Model import GNNModel
from torch_geometric.data import Data, DataLoader

import sys
import os
# 현재 파일 기준으로 상대 경로 설정
sys.path.append(os.path.join(os.path.dirname(__file__), '../class'))
from Graph import Graph  # Graph 클래스를 가져옴
from Resource import Resource  # Resource 클래스를 가져옴

# 데이터 수집 함수
def collect_data(graph_instance, num_samples=100):
    data_list = []
    for _ in range(num_samples):
        # 랜덤 그래프 생성
        G = graph_instance.generate_random_graph(num_nodes=10, num_edges=15)

        # 노드 특징과 레이블 생성
        node_features = []
        labels = []
        for node in G.nodes():
            traffic_flow = G.nodes[node]['traffic_flow']
            capacity = G.nodes[node]['capacity']
            signal_timer = G.nodes[node]['signal_timer']
            signal_cycle = G.nodes[node]['signal_cycle']
            node_feature = [traffic_flow, capacity, signal_timer, signal_cycle]
            node_features.append(node_feature)
            label = int(traffic_flow >= capacity)  # 병목 여부 레이블
            labels.append(label)

        # 간선 인덱스 및 특징 생성
        edge_index = []
        edge_attr = []
        for u, v in G.edges():
            edge_index.append([u, v])
            capacity = G.edges[u, v]['capacity']
            current_flow = G.edges[u, v]['current_flow']
            edge_attr.append([capacity, current_flow])

        # PyTorch Geometric의 Data 객체 생성
        data = Data(
            x=torch.tensor(node_features, dtype=torch.float),
            edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
            edge_attr=torch.tensor(edge_attr, dtype=torch.float),
            y=torch.tensor(labels, dtype=torch.long)
        )
        data_list.append(data)
    return data_list

# 모델 학습 함수
def train_model(data_list):
    dataloader = DataLoader(data_list, batch_size=1, shuffle=True)
    num_node_features = data_list[0].num_node_features
    model = GNNModel(num_node_features, hidden_channels=32)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    num_epochs = 10
    loss_values = []  # 손실 값을 저장할 리스트

    for epoch in range(num_epochs):
        total_loss = 0
        for data in dataloader:
            optimizer.zero_grad()
            out = model(data.x.squeeze(0), data.edge_index.squeeze(0), data.edge_attr.squeeze(0))
            loss = criterion(out, data.y.squeeze(0))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        loss_values.append(avg_loss)  # 에포크별 평균 손실 저장
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    # 손실 그래프 그리기
    plt.figure()
    plt.plot(range(1, num_epochs+1), loss_values, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss per Epoch')
    plt.grid(True)
    plt.savefig('training_loss.png')  # 그래프를 이미지 파일로 저장
    plt.show()

    # 모델 저장
    torch.save(model.state_dict(), 'model.pth')
    print("모델이 model.pth 파일로 저장되었습니다.")

    return model

# 메인 함수
def main():
    # 그래프 인스턴스 생성
    graph_instance = Graph()
    graph_instance.set_seed(42)

    # 데이터 수집
    print("데이터 수집 중...")
    data_list = collect_data(graph_instance, num_samples=100)

    # 모델 학습
    print("모델 학습 중...")
    model = train_model(data_list)

if __name__ == "__main__":
    main()
