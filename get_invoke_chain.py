import re

def get_invoke_flows(filename, flow_id):
    flow_maps = {}

    # 读取文件并构建 flow_maps 字典
    with open(filename, 'r') as file:
        for line in file:
            parts = line.split(", ")
            current_flow_id = int(parts[0])
            match = re.search(r'invoke_flow=\[(.*?)\]', line)
            invoke_flows = []
            
            if match:
                invoke_flow_str = match.group(1).strip()
                if invoke_flow_str:  # 确保不为空
                    invoke_flows = list(map(int, invoke_flow_str.split(', ')))

            flow_maps[current_flow_id] = invoke_flows

    # 用于存储所有能够被 invoke 的流
    all_invoke_flows = []

    # 深度优先搜索函数
    def dfs(current_flow):
        if current_flow not in all_invoke_flows:
            all_invoke_flows.append(current_flow)
            invoke_flow_list = flow_maps.get(current_flow, [])
            if len(invoke_flow_list) > 1:
                print(invoke_flow_list)
            for invoke_flow in invoke_flow_list:
                if len(invoke_flow_list) > 1:
                    all_invoke_flows.append('\n')
                dfs(invoke_flow)

    # 从给定的 flow_id 开始 DFS
    dfs(flow_id)

    return all_invoke_flows


filename = '../ResNet50-MNIST-pytorch/mix/llm_flow.txt'  # 替换为你的文件名
flow_id = 0  # 你想查询的流编号
result = get_invoke_flows(filename, flow_id)
print(f"流 {flow_id} 能够 invoke 的其他流编号: \n{' '.join(map(str, result))}")
