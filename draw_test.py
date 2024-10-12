# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

# 读取并解析aaa.txt文件
data_send = defaultdict(defaultdict(list))
data_send_meta = defaultdict(list)
data_recv = defaultdict(defaultdict(list))
data_recv_meta = defaultdict(list)

max_time = 0
min_time = float('inf')

with open('mix/output/test_0/test_0_snd_rcv_record_file_first_10000.txt', 'r') as file:
    for line in file:
        # print(line)
        parts = line.split()
        timestamp_ns = int(parts[0][:-1])  # 去掉冒号
        character = parts[1]
        node_id = int(parts[2])
        assert parts[3] == 'NIC'
        port_num = int(parts[4])
        action = parts[5]  # send or recv
        assert parts[6] == 'a'
        pkt_type = parts[7]
        assert parts[8] == 'pkt.'
        size = int(parts[9].split('=')[1])
        flowid = int(parts[10].split('=')[1])
        seq = int(parts[11].split('=')[1])

        max_time = max(max_time, timestamp_ns)
        min_time = min(min_time, timestamp_ns)

        if character == 'host':
            if action == 'send':
                if flowid not in data_send_meta:
                    data_send_meta[flowid] = [node_id, port_num, pkt_type]
                else:
                    assert node_id == data_send_meta[flowid][0]
                    assert port_num == data_send_meta[flowid][1]
                    assert pkt_type == data_send_meta[flowid][2]
                data_send[flowid]['time'].append(timestamp_ns)
                data_send[flowid]['size'].append(size)
                data_send[flowid]['seq'].append(seq)
            else:
                if flowid not in data_recv_meta:
                    data_recv_meta[flowid] = [node_id, port_num, pkt_type]
                else:
                    assert node_id == data_recv_meta[flowid][0]
                    assert port_num == data_recv_meta[flowid][1]
                    assert pkt_type == data_recv_meta[flowid][2]
                # data_recv[flowid].append([timestamp_ns, size, seq]
                data_recv[flowid]['time'].append(timestamp_ns)
                data_recv[flowid]['size'].append(size)
                data_recv[flowid]['seq'].append(seq)

print(df_send)
print(df_recv)
# 按flowid分组
grouped = df_send.groupby('flowid')

# 遍历每个flowid，计算发送和接收的flow rate
for flowid, group in grouped:
    # 按时间戳排序
    group = group.sort_values('timestamp')
    
    # 计算时间间隔（秒）和速率（Mbps）
    group['time_diff'] = group['timestamp'].diff().fillna(0) / 1e6  # 微秒转秒
    group['rate'] = group['size'] * 8 / group['time_diff'] / 1e6  # bits to Mbps

    # 分开发送和接收
    send_data = group[group['action'] == 'send']
    recv_data = group[group['action'] == 'recv']
    
    # 绘制发送速率
    plt.figure(figsize=(10, 6))
    plt.plot(send_data['timestamp'] / 1e6, send_data['rate'], label='Send Rate')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Rate (Mbps)')
    plt.title(f'Flow {flowid} Send Rate')
    plt.legend()
    plt.show()

    # 绘制接收速率
    plt.figure(figsize=(10, 6))
    plt.plot(recv_data['timestamp'] / 1e6, recv_data['rate'], label='Receive Rate', color='orange')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Rate (Mbps)')
    plt.title(f'Flow {flowid} Receive Rate')
    plt.legend()
    plt.show()
