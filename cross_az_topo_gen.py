import sys
import ast
import argparse
import itertools
from functools import reduce
import operator

# Generate switch node IDs
switch_node_ids = set()
server_node_ids = dict()


def translate_delay_ns(delay):
    if delay is None:
        return None
    if not isinstance(delay, str):
        return None
    delay = delay.strip().lower()  # 去除首尾空格并转换为小写
    if delay.endswith('ms'):
        return float(delay[:-2]) * 1e6
    elif delay.endswith('us'):
        return float(delay[:-2]) * 1e3
    elif delay.endswith('ns'):
        return float(delay[:-2])
    else:
        print("Unrecognized delay! Exiting...")
        sys.exit(0)


def parse_list(az_num, dci_lat_string):
    try:
        input_list = ast.literal_eval(dci_lat_string)
        if not isinstance(input_list, list) or not all(isinstance(item, tuple) for item in input_list):
            raise ValueError("Input is not a valid list of tuples.")
        
        expected_length = (az_num - 1) * az_num // 2
        if len(input_list) != expected_length:
            raise ValueError(f"Length of input list ({len(input_list)}) does not match expected ({expected_length}).")
        
        return input_list
    except (ValueError, SyntaxError) as e:
        print(f"Error parsing or validating input: {e}")
        sys.exit(1)


"""
    @param 
    nAZ是数据中心的数量
	每个数据中心有ndciswitch个边界交换机
	每个数据中心有ntorswitch个tor交换机
    每个数据中心有ncoreswitch个core交换机
	serverPertorswitch是每一个内部交换机上连接的服务器数量
	dcirate表示了跨数据中心的链路带宽
	dcnrate表示了数据中心内部的链路带宽
    dci_lat是一个list，list中的每个元素是一个三元组{a,b,c}，代表了从数据中心a到数据中心b的时延是c.
	dcn_lat是一个数字，代表了数据中心内的所有连接的延迟
	outputfile代表了输出文件名称
	
连接方式：
	1. 每个core交换机都和该数据中心的所有边界交换机有一条连接。
    2. 每个tor交换机都连接着该数据中心的全部core交换机
	3. 每个tor交换机都连接着serverPertorswitch个server
	4. 数据中心之间的连接遵循按顺序链接，比如，每个数据中心的第一个边界交换机之间进行全连接，每个数据中心的第2个边界交换机之间进行全连接，等等
"""
# Example usage:
# topo_file_gen1(2, 2, 2, 3, 4, 10, 1, [(0, 1, 5)], 1, 'topology.txt')
def topo_file_gen(nAZ, nwire_num, ndciswitch, ncoreswitch, ntorswitch, serverPertorswitch, dcirate, dcnrate, dci_lat, dcn_lat, outputfile):
    total_nodes = nAZ * (ndciswitch + ncoreswitch + ntorswitch + ntorswitch * serverPertorswitch)
    switch_nodes = nAZ * (ndciswitch + ncoreswitch + ntorswitch)
    links = []

    # Helper function to get node ID
    def get_node_id(az, sw_type, sw_id, server_id=0):
        if sw_type == 'dci':
            node_id = az * (ndciswitch + ncoreswitch + ntorswitch + ntorswitch * serverPertorswitch) + sw_id
        elif sw_type == 'core':
            node_id = az * (ndciswitch + ncoreswitch + ntorswitch + ntorswitch * serverPertorswitch) + ndciswitch + sw_id
        elif sw_type == 'tor':
            node_id = az * (ndciswitch + ncoreswitch + ntorswitch + ntorswitch * serverPertorswitch) + ndciswitch + ncoreswitch + sw_id
        elif sw_type == 'server':
            node_id = az * (ndciswitch + ncoreswitch + ntorswitch + ntorswitch * serverPertorswitch) \
                            + ndciswitch + ncoreswitch + ntorswitch + sw_id + server_id
            # print(az, sw_type, sw_id, server_id, '->', node_id)
        if sw_type != 'server':
            switch_node_ids.add(node_id)
        else:
            if server_node_ids.get(az) == None:
                server_node_ids[az] = [node_id]
            else:
                server_node_ids[az].append(node_id)
        return node_id

    # Generate connections within each data center
    for az in range(nAZ):
        for core_sw in range(ncoreswitch):
            core_sw_id = get_node_id(az, 'core', core_sw)
            # Connect core switches to dci switches
            for dci_sw in range(ndciswitch):
                dci_sw_id = get_node_id(az, 'dci', dci_sw)
                links.append((dci_sw_id, core_sw_id, dcnrate, dcn_lat))
                
        for tor_sw in range(ntorswitch):
            tor_sw_id = get_node_id(az, 'tor', tor_sw)
            # Connect tor switches to core switches
            for core_sw in range(ncoreswitch):
                core_sw_id = get_node_id(az, 'core', core_sw)
                links.append((core_sw_id, tor_sw_id, dcnrate, dcn_lat))

            # Connect servers to tor switches
            for server in range(serverPertorswitch):
                server_id = get_node_id(az, 'server', tor_sw * serverPertorswitch + server)
                links.append((server_id, tor_sw_id, dcnrate, dcn_lat))  # 这里为了配合conweave代码，把host放前面

    # Generate connections between data centers
    for (a, b, c) in dci_lat:
        for dci_sw in range(ndciswitch):
            dci_sw_id_a = get_node_id(a, 'dci', dci_sw)
            dci_sw_id_b = get_node_id(b, 'dci', dci_sw)
            for ii in range(nwire_num):
                links.append((dci_sw_id_a, dci_sw_id_b, dcirate, c))

    # Write to output file
    with open(outputfile, 'w') as f:
        f.write(f"{total_nodes} {switch_nodes} {len(links)}\n")
        f.write(" ".join(map(str, switch_node_ids)) + "\n")
        for src, dst, rate, delay in links:
            f.write(f"{src} {dst} {rate} {delay} 0\n")
        f.write(f"""
First line: total node #, switch node #, link #
Second line: switch node IDs...
src0 dst0 rate delay error_rate
src1 dst1 rate delay error_rate
...
""")
        f.write(f"switch IDs: ")
        f.write(','.join(map(str, list(switch_node_ids))))
    
# python3 cross_az_topo_gen.py -n 2 -i 1 -c 2 -t 4 -s 4 -B 100Gbps -b 50Gbps -L "[(0, 1, '3ms')]" -l 1us -o dumbbell
# python3 cross_az_topo_gen.py -n 2 -i 1 -c 1 -t 1 -s 4 -B 100Gbps -b 100Gbps -L "[(0, 1, '3ms')]" -l 1us -o dumbbell
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process topology parameters.")
    parser.add_argument("-n", "--num_az", type=int, default=2, help="Number of AZs")
    parser.add_argument("-w", "--wire_num", type=int, default=2, help="Number of links per DCI switch")
    parser.add_argument("-i", "--dci_switch_num", type=int, default=1, help="Number of DCI switches per AZ")
    parser.add_argument("-c", "--core_switch_num", type=int, default=2, help="Number of core switches per AZ")
    parser.add_argument("-t", "--tor_switch_num", type=int, default=2, help="Number of ToR switches per AZ")
    parser.add_argument("-s", "--n_server_per_tor", type=int, default=1, help="Number of servers per ToR switch")

    parser.add_argument("-B", "--dci_rate", type=str, default="100Gbps", help="DCI link rate (e.g., '100Gbps')")
    parser.add_argument("-b", "--dcn_rate", type=str, default="100Gbps", help="DCN link rate (e.g., '100Gbps')")
    parser.add_argument("-r", "--tor_rate", type=str, default="100Gbps", help="DCN link rate (e.g., '100Gbps')")
    parser.add_argument("-L", "--dci_lat", type=str, 
                        default="",
                        # default="[(0, 1, '3ms'), (0, 2, '5ms'), (1, 2, '7ms')]",
                        help="List of DCI latencies in format [(a, b, 'c'), ...]")
    parser.add_argument("-l", "--dcn_lat", type=str, default="1us", help="DCN link rate (e.g., '1us')")
    parser.add_argument("-o", "--topo_type", type=str, default="dumbbell", help="topo_type")
    parser.add_argument("-O", "--output_file_name", type=str, default="", help="output_file_name")
    
    args = parser.parse_args()

    n_AZ = args.num_az
    n_wirenum = args.wire_num
    n_dci_switch = args.dci_switch_num
    n_core_switch = args.core_switch_num
    n_tor_switch = args.tor_switch_num
    n_server_per_tor = args.n_server_per_tor
    dci_rate = args.dci_rate
    dcn_rate = args.dcn_rate
    if n_AZ > 1:
        dci_lat = parse_list(args.num_az, args.dci_lat)
        dci_lat = [(a, b, delay) for a, b, delay in dci_lat]
    # dcn_lat = translate_delay_ns(args.dcn_lat)
    outputfile = args.topo_type +'_az'+str(n_AZ)+'_dci'+str(n_dci_switch)+'_core'+str(n_core_switch)+'_tor'+str(n_tor_switch) +'_host'+str(n_server_per_tor) + '.txt'
    if args.output_file_name != "":
        outputfile = 'config/' + args.output_file_name + '.txt'

    topo_file_gen(
        nAZ=n_AZ,
        nwire_num=n_wirenum,
        ndciswitch=n_dci_switch,
        ncoreswitch = n_core_switch,
        ntorswitch = n_tor_switch,
        serverPertorswitch = n_server_per_tor,
        dcirate = dci_rate,
        dcnrate = dcn_rate,
        dci_lat = dci_lat,
        dcn_lat = args.dcn_lat,
        outputfile = outputfile
    )
    print(list(switch_node_ids))
    print(server_node_ids)
    
