def contains_all(line, keywords):
    """检查行中是否包含所有关键字。"""
    return all(keyword in line for keyword in keywords)

def find_matching_flow_ids(filename, keyword_lists):
    """查找满足条件的 FlowId 列表。"""
    matching_flow_ids = []

    with open(filename, 'r') as file:
        for line in file:
            # 提取 FlowId
            parts = line.split()
            flow_id = None
            
            for part in parts:
                if part.startswith("FlowId="):
                    flow_id = int(part.split("=")[1])
                    break
            
            if flow_id is None:
                print('flowId not found!')
                continue  # 如果没有找到 FlowId，跳过此行

            # 检查每个关键字列表
            for keywords in keyword_lists:
                if contains_all(line, keywords):
                    matching_flow_ids.append(flow_id)
                    break  # 找到匹配后跳出循环，继续处理下一行

    return matching_flow_ids


# 当列表中的任意一个子列表的所有字符串都在一行中出现时，给出所有满足条件的flow的FlowId的列表
def main(fname=None):
        
    # 读取 ConfigId
    with open("mix/last_param.txt", 'r') as file:
        config_id = file.readline().strip()  # 读取第一行并去除空白字符

    # 构造输出文件名
    filename = f"mix/output/{config_id}/{config_id}_flow_statistics_output.txt"

    if fname is not None:
        filename = fname

    keyword_lists = [
        ["priority=3", "DP"],  # Flow, DP
        # ["priority=-1", "Reduce-Scatter"]
    ]

    matching_flow_ids = find_matching_flow_ids(filename, keyword_lists)

    print("Matching FlowIds:", matching_flow_ids)

if __name__ == "__main__":
    main()
