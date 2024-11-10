def parse_ids(filename):
    switch_dict = {}
    host_dict = {}
    
    with open(filename, 'r') as file:
        lines = file.readlines()
        
        # 标志变量，用于判断是否在对应的ID部分
        switch_section = False
        host_section = False
        
        for line in lines:
            # 检查是否到达switch IDs部分
            if line.strip() == "switch IDs:":
                switch_section = True
                host_section = False
                continue
            # 检查是否到达host IDs部分
            elif line.strip() == "host IDs:":
                host_section = True
                switch_section = False
                continue
            
            # 如果在switch IDs部分
            if switch_section:
                # 检查行是否是空白行
                if not line.strip():
                    continue
                
                # 提取AZ编号和对应的switch ID列表
                parts = line.strip().split(": ")
                if len(parts) == 2:
                    az_key = parts[0].split()[-1]  # 获取AZ编号
                    ids = list(map(int, parts[1].split(',')))  # 提取并转换ID为整数列表
                    switch_dict[az_key] = ids
            
            # 如果在host IDs部分
            elif host_section:
                # 检查行是否是空白行
                if not line.strip():
                    continue
                
                # 提取AZ编号和对应的host ID列表
                parts = line.strip().split(": ")
                if len(parts) == 2:
                    az_key = parts[0].split()[-1]  # 获取AZ编号
                    ids = list(map(int, parts[1].split(',')))  # 提取并转换ID为整数列表
                    host_dict[az_key] = ids
    
    return switch_dict, host_dict



if __name__ == '__main__':
    filename = '../config/topo_dumbbell_4dcilink_2dci_4core_4tor_16host_100Gbps_100Gbps_100Gbps.txt'
    switch_ids, host_ids = parse_ids(filename)
    print("Switch IDs:", switch_ids)
    print("Host IDs:", host_ids)