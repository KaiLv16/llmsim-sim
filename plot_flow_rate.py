import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import defaultdict
import argparse


def parse_args():
    # 创建解析器
    parser = argparse.ArgumentParser(description="绘制曲线并在指定范围内控制图例显示")

    # 添加参数
    parser.add_argument('--x_min', type=float, default=-1, help='x 轴的最小值，默认值为 -1')
    parser.add_argument('--x_max', type=float, default=10000000000000, help='x 轴的最大值，默认值为 1000000000000')
    parser.add_argument('--threshold', type=float, default=5, help='曲线最大值的阈值，超过该值时才显示图例，默认值为 0.5')
    parser.add_argument('--type', type=str, default='send_recv', help='指定绘制的曲线属于 \'send\' 还是 \'recv\' 。默认是\'send_recv\'')

    # 解析命令行参数
    args = parser.parse_args()
    return args


# 定义每个包的大小 (字节)
pkt_size = 1000 * 8  # 转换为比特
time_interval = 5000  # 区间长度，以ns为单位

# 参数：是否平滑以及平滑窗口的范围
enable_smoothing = True  # 是否启用平滑
smoothing_range = 1  # 平滑窗口大小（单位：区间）

# 读取并解析aaa.txt文件
data_send = defaultdict(lambda: defaultdict(list))
data_send_meta = defaultdict(list)
data_recv = defaultdict(lambda: defaultdict(list))
data_recv_meta = defaultdict(list)
flow_path = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

max_time = 0
min_time = float('inf')

use_pkl = True
mode = 'send'
mode = 'recv'

def read_flowid_from_file(filename):
    with open(filename, 'r') as file:
        # 读取第一行，获取i和j
        first_line = file.readline().strip().split()
        i, j = int(first_line[0]), int(first_line[1])
        
        # 如果 i 或 j 为 -1，退出
        if i == -1 or j == -1:
            return None
        
        # 读取剩余的所有行
        lines = file.readlines()
        
        # 确保 i 和 j 在有效范围内
        if 1 <= i <= len(lines) and 1 <= j <= len(lines) and i <= j:
            # 读取第i到第j行（包括i和j），将这些行的数字组合成一个列表
            combined_list = []
            for line in lines[i - 2:j - 1]:  # 获取从第i行到第j行的内容
                # 跳过以#开头的行
                if not line.strip().startswith('#'):
                    numbers = list(map(int, line.strip().split()))
                    combined_list.extend(numbers)
            return combined_list
        else:
            return None  # 如果i或j超出文件行数范围，返回None


if use_pkl == False or not os.path.exists('results/flow_send_rate.pkl'):
    with open('mix/output/test_0/test_0_snd_rcv_record_file.txt', 'r') as file:
        for line in file:
            # print(line)
            parts = line.split()
            timestamp_ns = int(parts[0][:-1]) - 2000000000  # 时间（ns）  # 去掉冒号
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
                        data_send_meta[(flowid, pkt_type)] = [node_id, port_num, pkt_type]
                    else:
                        print(line)
                        assert node_id == data_send_meta[(flowid, pkt_type)][0], f"{node_id} != {data_send_meta[(flowid, pkt_type)][0]}"
                        assert port_num == data_send_meta[(flowid, pkt_type)][1], f"{node_id} != {data_send_meta[(flowid, pkt_type)][1]}"
                        assert pkt_type == data_send_meta[(flowid, pkt_type)][2], f"{node_id} != {data_send_meta[(flowid, pkt_type)][2]}"
                    data_send[(flowid, pkt_type)]['time'].append(timestamp_ns)
                    data_send[(flowid, pkt_type)]['size'].append(size)
                    data_send[(flowid, pkt_type)]['seq'].append(seq)
                else:
                    if flowid not in data_recv_meta:
                        data_recv_meta[(flowid, pkt_type)] = [node_id, port_num, pkt_type]
                    else:
                        assert node_id == data_recv_meta[(flowid, pkt_type)][0]
                        assert port_num == data_recv_meta[(flowid, pkt_type)][1]
                        assert pkt_type == data_recv_meta[(flowid, pkt_type)][2]
                    # data_recv[flowid].append([timestamp_ns, size, seq]
                    data_recv[(flowid, pkt_type)]['time'].append(timestamp_ns)
                    data_recv[(flowid, pkt_type)]['size'].append(size)
                    data_recv[(flowid, pkt_type)]['seq'].append(seq)

            else:
                if action == 'send':
                    if (node_id, port_num) not in flow_path[flowid]['send'][pkt_type]:
                        flow_path[flowid]['send'][pkt_type].insert(0, (node_id, port_num))
                        # data_send_meta[(flowid, pkt_type)] = [node_id, port_num, pkt_type]
                else:
                    if (node_id, port_num) not in flow_path[flowid]['recv'][pkt_type]:
                        flow_path[flowid]['recv'][pkt_type].insert(0, (node_id, port_num))
                        # data_recv_meta[(flowid, pkt_type)] = [node_id, port_num, pkt_type]


    for k, v in data_send_meta.items():
        if (v[0], v[1]) not in flow_path[k[0]]['send'][k[1]]:
            flow_path[k[0]]['send'][k[1]].insert(0, (v[0], v[1]))
    for k, v in data_recv_meta.items():
        if (v[0], v[1]) not in flow_path[k[0]]['recv'][k[1]]:
            flow_path[k[0]]['recv'][k[1]].insert(0, (v[0], v[1]))

    with open('results/flow_path_output.txt', 'w') as file:
        # 遍历 flow_path 按 flow_id 写入文件
        for flow_id in sorted(flow_path.keys()):
            flow_data = flow_path[flow_id]
            file.write(f'Flow ID: {flow_id}\n')
            for type, data in flow_data.items():
                file.write(f'  {type} Path:\n')
                for pkt_type, path in data.items():
                    file.write(f'    {pkt_type}: {path}\n')

            file.write('\n')  # 每个 flow_id 之间加一个空行



    # 存储数据的字典
    # flow_data = {}

    # # 读取并解析文件
    # with open('mix/output/test_0/test_0_snd_rcv_record_file.txt', 'r') as file:
    #     for line in file:
    #         if 'host' in line and 'NIC' in line and 'send' in line:
    #             # 提取时间、flowid和seq
    #             match = re.search(r'(\d+):\s+host\s+\d+\s+NIC\s+\d+\s+send\s+a\s+pkt.\s+size=\d+\s+flowid=(\d+)\s+seq=(\d+)', line)
    #             if match:
    #                 time = int(match.group(1)) - 2000000000  # 时间（ns）
    #                 flowid = int(match.group(2))  # flowid
    #                 seq = int(match.group(3))  # seq
    #                 max_time = max(max_time, time)
    #                 min_time = min(min_time, time)

    #                 if flowid not in flow_data:
    #                     # 初始化数据
    #                     flow_data[flowid] = {
    #                         'time': [],        # 保存包的发送时间
    #                         'seq': [],         # 保存包的序列号
    #                     }
                        
    #                 # 保存时间和序列号
    #                 # 每次保存表示在时间 time 到达了一个大小为 pkt_size 的数据包，包的序列号为 seq
    #                 flow_data[flowid]['time'].append(time)
    #                 flow_data[flowid]['seq'].append(seq)

    print(f"min_time: {min_time}, max_time: {max_time}")

    flow_send_rate = {}
    flow_recv_rate = {}
    plot_min_time = 0

    # 这里画的是 data_send
    for flowid_type, data in data_send.items():
        if flowid_type[1] == 'UDP':
            flow_send_rate[flowid_type] = np.zeros((max_time - plot_min_time) // time_interval + 1)  # 初始化速率为0
            # print(data['time'])
            for time in data['time']:
                # 计算当前包属于哪个时间区间
                index = (time - plot_min_time) // time_interval
                flow_send_rate[flowid_type][index] += pkt_size  # 在该时间区间内累加数据包大小

    # 这里画的是 data_recv
    for flowid_type, data in data_recv.items():
        if flowid_type[1] == 'UDP':
            flow_recv_rate[flowid_type] = np.zeros((max_time - plot_min_time) // time_interval + 1)  # 初始化速率为0
            # print(data['time'])
            for time in data['time']:
                # 计算当前包属于哪个时间区间
                index = (time - plot_min_time) // time_interval
                flow_recv_rate[flowid_type][index] += pkt_size  # 在该时间区间内累加数据包大小

    # 保存 flow_send_rate 到文件
    with open('results/flow_send_rate.pkl', 'wb') as file:
        pickle.dump(flow_send_rate, file)

    with open('results/flow_recv_rate.pkl', 'wb') as file:
        pickle.dump(flow_recv_rate, file)

else:
    # 从文件中读取 flow_send_rate
    with open('results/flow_send_rate.pkl', 'rb') as file:
        flow_send_rate = pickle.load(file)
    
    # 从文件中读取 flow_recv_rate
    with open('results/flow_recv_rate.pkl', 'rb') as file:
        flow_recv_rate = pickle.load(file)

print('aaa')

# 将每个区间的发送数据量转换为发送速率 (Gbps)
for flowid_type, rate in flow_send_rate.items():
    # rate 是比特/纳秒，刚好是 Gbps
    flow_send_rate[flowid_type] = flow_send_rate[flowid_type] / time_interval

# 将每个区间的发送数据量转换为发送速率 (Gbps)
for flowid_type, rate in flow_recv_rate.items():
    # rate 是比特/纳秒，刚好是 Gbps
    flow_recv_rate[flowid_type] = flow_recv_rate[flowid_type] / time_interval

print('bbb')

# 可选：应用平滑
# if enable_smoothing:
#     for flowid, rate in flow_send_rate.items():
#         smooth_rate = np.copy(rate)
#         for i in range(smoothing_range, len(rate) - smoothing_range):
#             smooth_rate[i] = np.mean(rate[i-smoothing_range:i+smoothing_range+1])
#         flow_send_rate[flowid] = smooth_rate

if enable_smoothing:
    for flowid_type, rate in flow_send_rate.items():
        # smooth_rate = np.copy(rate)
        df = pd.DataFrame(rate)
        df = df.rolling(window=3, center=True).mean()
        # 先前向填充，再后向填充，防止两端出现nan
        df_filled = df.fillna(method='ffill').fillna(method='bfill')
        flow_send_rate[flowid_type] = df_filled.to_numpy().T[0]
        # print(flow_send_rate[flowid_type])
    
    for flowid_type, rate in flow_recv_rate.items():
        # smooth_rate = np.copy(rate)
        df = pd.DataFrame(rate)
        df = df.rolling(window=3, center=True).mean()
        # 先前向填充，再后向填充，防止两端出现nan
        df_filled = df.fillna(method='ffill').fillna(method='bfill')
        flow_recv_rate[flowid_type] = df_filled.to_numpy().T[0]
        # print(flow_recv_rate[flowid_type])

print('ccc')

flow_list = read_flowid_from_file('config/flow_to_be_plotted.py')
# assert flow_list is not None
print(f"current curve in \'flow_to_be_plotted.py\': {flow_list}")

args = parse_args()

# 绘制图表
plt.figure(figsize=(10, 6))

if 'send' in args.type:
    for flowid_type, rate in flow_send_rate.items():
        if flow_list is None or (flow_list is not None and flowid_type[0] in flow_list):
            # 转换为 Gbps
            time_axis = np.arange(len(rate)) * time_interval / 1000000  # 转换为 ms
            # x_max = min(time_axis[-1], args.x_max)
            # x_min = max(time_axis[0],  args.x_min)
            mask = (time_axis >= args.x_min) & (time_axis <= args.x_max)
            x_sub = time_axis[mask]
            y_sub = rate[mask]
            
            # print(flowid_type, rate, 'Global max:', max(rate), '; ', y_sub, 'Interval max:', max(y_sub))
            
            if len(x_sub) > 0 and len(y_sub) > 0:
                label = f'Flow {flowid_type} Snd Rate' if np.max(y_sub) > args.threshold else '_nolegend_'
                plt.plot(x_sub, y_sub, label=label)
            # plt.plot(time_axis, rate, label=f'Flow {flowid_type} Sending Rate')

if 'recv' in args.type:
    for flowid_type, rate in flow_recv_rate.items():
        if flow_list is None or (flow_list is not None and flowid_type[0] in flow_list):
            # 转换为 Gbps
            time_axis = np.arange(len(rate)) * time_interval / 1000000  # 转换为 ms
            # x_max = min(time_axis[-1], args.x_max)
            # x_min = max(time_axis[0],  args.x_min)
            mask = (time_axis >= args.x_min) & (time_axis <= args.x_max)
            x_sub = time_axis[mask]
            y_sub = rate[mask]
            
            # print(flowid_type, rate, 'Global max:', max(rate), '; ', y_sub, 'Interval max:', max(y_sub))
            
            if len(x_sub) > 0 and len(y_sub) > 0:
                label = f'Flow {flowid_type} Rcv Rate' if np.max(y_sub) > args.threshold else '_nolegend_'
                plt.plot(x_sub, y_sub, label=label)
            # plt.plot(time_axis, rate, label=f'Flow {flowid_type} Sending Rate')


# 设置 x 轴范围
# plt.xlim([x_min, x_max])

plt.xlabel('Time (ms)')
plt.ylabel('Rate (Gbps)')
plt.title('Sending / Receiving Rates per Flow')

# plt.legend(loc='best')

# 设置图例，并将其放在图片下方
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4)

# 设置图例，并将其放在图片外
# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.grid(True)

# 保存为PDF，不裁切边界
plt.savefig(f'results/flow_rates.pdf', format='pdf', bbox_inches='tight', pad_inches=0.1)

# plt.savefig(f'results/flow_rates_{args.x_min}_{args.x_max}.pdf', format='pdf', bbox_inches='tight', pad_inches=0.1)


plt.show()
