import pickle
from pprint import pprint

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
            # 提取 FlowId
            parts = line.split()
            flow_id = None
            rtt = None
            
            for part in parts:
                if part.startswith("FlowId="):
                    flow_id = int(part.split("=")[1])
                    break
            
            for part in parts:
                if part.startswith("RTT="):
                    rtt = int(part.split("=")[1])
                    break
            
            if flow_id is None:
                print('flowId not found!')
                continue  # 如果没有找到 FlowId，跳过此行

            if rtt is None:
                print('rtt not found!')
                continue  # 如果没有找到 FlowId，跳过此行
            
            start = line.find('note="') + 6
            end = line.rfind('"')
            note = line[start:end]

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

if __name__ == "__main__":
    main()

    # print(contains_all('FlowId=128 priority=3 src=17 dst=30 size=8388608 ', ["priority=3", "1"]))
