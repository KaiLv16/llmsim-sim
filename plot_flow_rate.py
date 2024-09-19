import re
import matplotlib.pyplot as plt
import numpy as np

# 定义每个包的大小 (字节)
pkt_size = 1000 * 8  # 转换为比特
time_interval = 1e3  # 固定的时间间隔为 1 毫秒 (1e6 纳秒)

# 存储数据的字典
flow_data = {}

max_time = 0
min_time = 100000000000000000000

# 读取并解析文件
with open('mix/output/test_0/test_0_snd_rcv_record_file.txt', 'r') as file:
    for line in file:
        if 'host' in line and 'NIC' in line and 'send' in line:
            # 提取时间、flowid和seq
            match = re.search(r'(\d+):\s+host\s+\d+\s+NIC\s+\d+\s+send\s+a\s+pkt.\s+size=\d+\s+flowid=(\d+)\s+seq=(\d+)', line)
            if match:
                time = int(match.group(1)) - 2000000000 # 时间
                flowid = int(match.group(2))  # flowid
                seq = int(match.group(3))  # seq
                if max_time < time:
                    max_time = time
                if min_time > time:
                    min_time = time

                if flowid not in flow_data:
                    flow_data[flowid] = {'time': [], 'seq': [], 'rate': [0,]}
                
                # 保存时间和seq
                flow_data[flowid]['time'].append(time)
                flow_data[flowid]['seq'].append(seq)

                if len(flow_data[flowid]['time']) > 1:
                    rate = 1000 * 8 / (flow_data[flowid]['time'][-1] - flow_data[flowid]['time'][-2])  # Gbps
                    flow_data[flowid]['rate'].append(rate)


# 绘制图表
plt.figure(figsize=(10, 6))

for flowid, data in flow_data.items():
    flow_data[flowid]['time'].append(flow_data[flowid]['time'][-1] + 100)
    flow_data[flowid]['rate'].append(0)

    plt.plot(np.array(flow_data[flowid]['time']) / 1000000, np.array(flow_data[flowid]['rate']), label=f'Flow {flowid} Sending Rate')

    

# # 有效发送速率
# for flowid, rates in flow_effective_rates.items():
#     plt.plot(range(len(rates)), rates, '--', label=f'Flow {flowid} Effective Sending Rate')

plt.xlabel(f'Time (ms)')
plt.ylabel('Rate (Gbps)')
plt.title('Total and Effective Sending Rates per Flow')
plt.legend(loc='best')
plt.grid(True)

# 保存为PDF，不裁切边界
plt.savefig('flow_rates.pdf', format='pdf', bbox_inches=None, pad_inches=0.1)
plt.show()
