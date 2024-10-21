import pickle
from pprint import pprint
import pandas as pd

def parse_flow_statistics(file_path):
    # 读取文件
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # 解析每一行
    data = []
    for line in lines:
        if line.startswith("FlowId"):
            # 提取 note 字段
            start = line.find('note="') + 6
            end = line.rfind('"')
            note = line[start:end] if start > 5 and end > start else ''

            # 删除 note 部分
            line = line[:line.find('note="')] + line[line.find('"', end + 1) + 1:]

            # 按照逗号分割每一项
            parts = line.strip().split(',')
            entry = {}

            for part in parts:
                key_value = part.split('=', 1)  # 只分割成键值对
                if len(key_value) == 2:
                    key = key_value[0].strip()
                    value = key_value[1].strip()
                    # print(f"{key}-{value}-")
                    # 尝试将值转换为数字
                    if value.isdigit():  # 检查是否为整数
                        # print(f"{key}--{value}--")
                        entry[key] = int(value)
                    else:
                        try:
                            entry[key] = float(value)  # 尝试转换为浮点数
                        except ValueError:
                            entry[key] = value  # 如果转换失败，保留为字符串

            entry['note'] = note  # 将 note 添加到字典中
            data.append(entry)

    # 创建 DataFrame
    df = pd.DataFrame(data)

    # 将 note 列移到最后
    cols = [col for col in df.columns if col != 'note'] + ['note']
    df = df[cols]

    # 设置打印选项，避免横向折叠
    pd.set_option('display.expand_frame_repr', False)
    pd.set_option('display.max_columns', None)  # 显示所有列
    pd.set_option('display.max_colwidth', None)  # 显示每列的最大宽度

    return df


def contains_all(line, keywords):
    """检查行中是否包含所有关键字。"""
    return all(keyword in line for keyword in keywords)


def find_matching_flow_ids(filename, keyword_lists):
    """查找满足条件的 FlowId 列表。"""
    matching_flow_ids = []
    matching_flow_rtts = []
    matching_flow_notes = []

    with open(filename, 'r') as file:
        for line in file:
            if line.startswith("FlowId"):
                # 提取 note 字段
                start = line.find('note="') + 6
                end = line.rfind('"')
                note = line[start:end] if start > 5 and end > start else ''

                # 删除 note 部分
                line = line[:line.find('note="')] + line[line.find('"', end + 1) + 1:]

                # 按照逗号分割每一项
                parts = line.strip().split(',')
                entry = {}

                flow_id = None
                rtt = None
            
                for part in parts:
                    if part.strip().startswith("FlowId="):
                        flow_id = int(part.split("=")[1])
                        break
                
                for part in parts:
                    if part.strip().startswith("RTT="):
                        rtt = int(part.split("=")[1])
                        break
                
                if flow_id is None:
                    print('flowId not found!')
                    continue  # 如果没有找到 FlowId，跳过此行

                if rtt is None:
                    print(line)
                    print('rtt not found!')
                    continue  # 如果没有找到 FlowId，跳过此行

                # 检查每个关键字列表
                for keywords in keyword_lists:
                    if contains_all(line, keywords):
                        matching_flow_ids.append(flow_id)
                        matching_flow_rtts.append(rtt)
                        matching_flow_notes.append(note)
                        break  # 找到匹配后跳出循环，继续处理下一行

    return matching_flow_ids, matching_flow_rtts, matching_flow_notes


# 当列表中的任意一个子列表的所有字符串都在一行中出现时，给出所有满足条件的flow的FlowId的列表
def main(fname=None):
        
    # 读取 ConfigId
    with open("mix/last_param.txt", 'r') as file:
        config_id = file.readline().strip()  # 读取第一行并去除空白字符

    filename = f"mix/output/{config_id}/{config_id}_flow_statistics_output.txt"

    if fname is not None:
        filename = fname

    print(filename)
    
    df = parse_flow_statistics(filename)
    print(df.head(20))  # n 是您想打印的行数

    keyword_lists1 = [[" "]]
    
    keyword_lists2 = [
        ["priority=3", "DP", "71, 143", "chunk 1"],  # Flow, DP
    ]

    ids_tot, rtts_tot, notes_tot = find_matching_flow_ids(filename, keyword_lists1)

    # 创建字典并反转键值对
    result = {key: [value1, value2] for key, value1, value2 in zip(ids_tot, rtts_tot, notes_tot)}
    # pprint(result)
    print("metadata of all Flows and Deps is saved as config/flowid_rtt.pkl")

    ids, rtts, notes = find_matching_flow_ids(filename, keyword_lists2)
    print("Matching FlowIds:", ids)
    print("Matching Item Cnt:", len(ids))

    with open(f'config/flowid_rtt.pkl', 'wb') as file:
        pickle.dump(result, file)
    
    return df

if __name__ == "__main__":
    df = main()
    
    print("\n")
    
    col_names= ["priority", "size", "lat", "RTT", "idealDelta"]
    for col_name in col_names:
        unique_values_set = set(df[col_name])
        print(f"{col_name}的所有取值: {unique_values_set}")
    
    print("\n")
    
    df = df[df['priority'] == 3]   # Flow only, not Dep

    avg_slowdown = df['slowDown'].mean()
    max_slowdown = df['slowDown'].max()
    min_slowdown = df['slowDown'].min()
    print(f"所有Flow的平均/最大/最小 slowDown: {avg_slowdown}/{max_slowdown}/{min_slowdown}")

    avg_slowdown_dp = df[df['note'].str.contains("DP", na=False)]['slowDown'].mean()
    max_slowdown_dp = df[df['note'].str.contains("DP", na=False)]['slowDown'].max()
    min_slowdown_dp = df[df['note'].str.contains("DP", na=False)]['slowDown'].min()
    print(f"所有DP Flow的平均/最大/最小 slowDown: {avg_slowdown_dp}/{max_slowdown_dp}/{min_slowdown_dp}")
    
    # 计算 RTT 小于 5000 的 slowDown 平均值
    avg_slowdown_rtt_below_5000 = df[df['RTT'] < 5000]['slowDown'].mean()
    max_slowdown_rtt_below_5000 = df[df['RTT'] < 5000]['slowDown'].max()
    min_slowdown_rtt_below_5000 = df[df['RTT'] < 5000]['slowDown'].min()
    print(f"RTT 小于 5000ns (在同一个ToR下) 的平均/最大/最小 slowDown: {avg_slowdown_rtt_below_5000}/{max_slowdown_rtt_below_5000}/{min_slowdown_rtt_below_5000}")
    
    # 计算 RTT 在 5000 到 10000 的 slowDown 平均值
    avg_slowdown_rtt_5000_to_10000 = df[(df['RTT'] >= 5000) & (df['RTT'] <= 10000)]['slowDown'].mean()
    max_slowdown_rtt_5000_to_10000 = df[(df['RTT'] >= 5000) & (df['RTT'] <= 10000)]['slowDown'].max()
    min_slowdown_rtt_5000_to_10000 = df[(df['RTT'] >= 5000) & (df['RTT'] <= 10000)]['slowDown'].min()
    print(f"RTT 在 5000ns 到 10000ns (同DC不同ToR) 的平均/最大/最小 slowDown: {avg_slowdown_rtt_5000_to_10000}/{max_slowdown_rtt_5000_to_10000}/{min_slowdown_rtt_5000_to_10000}")

    # 计算 RTT 大于 10000 的 slowDown 平均值
    avg_slowdown_rtt_above_10000 = df[df['RTT'] > 10000]['slowDown'].mean()
    max_slowdown_rtt_above_10000 = df[df['RTT'] > 10000]['slowDown'].max()
    min_slowdown_rtt_above_10000 = df[df['RTT'] > 10000]['slowDown'].min()
    print(f"RTT 大于 10000ns (不同DC) 的平均/最大/最小 slowDown: {avg_slowdown_rtt_above_10000}/{max_slowdown_rtt_above_10000}/{min_slowdown_rtt_above_10000}")

    # print(contains_all('FlowId=128 priority=3 src=17 dst=30 size=8388608 ', ["priority=3", "1"]))
