#!/bin/bash

cecho(){  # source: https://stackoverflow.com/a/53463162/2886168
    RED="\033[0;31m"
    GREEN="\033[0;32m"
    YELLOW="\033[0;33m"
    # ... ADD MORE COLORS
    NC="\033[0m" # No Color

    printf "${!1}${2} ${NC}\n"
}

cecho "GREEN" "Running RDMA Network Load Balancing Simulations (leaf-spine topology)"

# TOPOLOGY="leaf_spine_128_100G_OS2" # or, fat_k8_100G_OS2
# NETLOAD="50" # network load 50%
# RUNTIME="0.1" # 0.1 second (traffic generation)

NUM_AZ="2"
dci_link_num="4"
dci_switch_num="2"
core_switch_num="4"
tor_switch_num="4"
n_server_per_tor="4"
DCI_SPEED="100Gbps"
DCN_SPEED="100Gbps"
TOR_SPEED="100Gbps"
DCI_LAT="2ms"

host_num=$((tor_switch_num * tor_switch_num))

TOPOLOGY="topo_dumbbell_${dci_link_num}dcilink_${dci_switch_num}dci_${core_switch_num}core_${tor_switch_num}tor_$((host_num))host_${DCI_SPEED}_${DCN_SPEED}_${TOR_SPEED}" # or, fat_k8_100G_OS2

FLOWFILE="dumbbell_flow_test1"
FLOWFILE2="dumbbell_cc_test"
FLOWFILE_LLM="../../ResNet50-MNIST-pytorch/mix/llm_flow"
NODE_MAPPING="../../ResNet50-MNIST-pytorch/mix/node_mapping"


NETLOAD="50" # network load 50%
RUNTIME="100" # 0.1 second (traffic generation)
FLOW_RELATION="1"   # "1" means using ralational flow file, while "0" use traditional flow file


cecho "YELLOW" "\n----------------------------------"
cecho "YELLOW" "TOPOLOGY: ${TOPOLOGY}" 
cecho "YELLOW" "FLOWFILE: ${FLOWFILE}" 
cecho "YELLOW" "NETWORK LOAD: ${NETLOAD} (may be useless)" 
cecho "YELLOW" "TIME: ${RUNTIME}" 
cecho "YELLOW" "----------------------------------\n"

# 生成拓扑
cecho "GREEN" "Generating Topo file..."
python3 cross_az_topo_gen.py -n ${NUM_AZ} -w ${dci_link_num} -i ${dci_switch_num} -c ${core_switch_num} -t ${tor_switch_num} -s ${n_server_per_tor} \
                             -B ${DCI_SPEED} -b ${DCN_SPEED} -r ${TOR_SPEED} -L "[(0, 1, '${DCI_LAT}')]" -l 1us -o dumbbell -O ${TOPOLOGY}

# 命令最后加上 &，可以让这个进行在后台运行。
# running test 
cecho "GREEN" "Run Test..."

# # GDB debug 方法
    # ./waf shell
    # gdb
    # file ./build/scratch/network-load-balance
    # set args mix/output/test_0/config.txt
    # r
    # bt
    # info thread
    # thread 1
    # frame 1`

# rm -r mix/output/

LOG_FILE="my_test_log.txt"  
  
if [ -e "$LOG_FILE" ]; then
    rm "$LOG_FILE"  
    echo "File '$LOG_FILE' has been removed."  
else  
    echo "File '$LOG_FILE' does not exist."  
fi

# cc_modes = {"dcqcn", "hpcc", "timely", "dctcp", "hpccPint"}
# lb_modes = {"fecmp", "drill", "conga", "letflow", "conweave", "host_spray", "switch_spray", "host_switch_spray"}

# `debug` parameter is useless
python3 run.py  --flow_file_name ${FLOWFILE_LLM} --flow_relation ${FLOW_RELATION} --node_mapping ${NODE_MAPPING} \
   --cc dcqcn --lb switch_spray --spray 1 --pfc 0 --irn 1 --debug 0 \
   --simul_time ${RUNTIME} --netload ${NETLOAD} --topo ${TOPOLOGY}
cp mix/output/'dcqcn(1)_switch_spray(12)_pfc0_irn1'/'dcqcn(1)_switch_spray(12)_pfc0_irn1_flow_statistics_output.txt' results/irn_${DCI_LAT}.txt

#    > results/my_test_log.txt

# Lossless RDMA
# python3 run.py --lb fecmp --pfc 1 --irn 0 --simul_time ${RUNTIME} --netload ${NETLOAD} --topo ${TOPOLOGY} 2>&1 > /dev/null & 
# sleep 5
# python3 run.py --lb letflow --pfc 1 --irn 0 --simul_time ${RUNTIME} --netload ${NETLOAD} --topo ${TOPOLOGY} 2>&1 > /dev/null &
# sleep 0.1
# python3 run.py --lb conga --pfc 1 --irn 0 --simul_time ${RUNTIME} --netload ${NETLOAD} --topo ${TOPOLOGY} 2>&1 > /dev/null &
# sleep 0.1
# python3 run.py --lb conweave --pfc 1 --irn 0 --simul_time ${RUNTIME} --netload ${NETLOAD} --topo ${TOPOLOGY} 2>&1 > /dev/null &
# sleep 0.1

# # IRN RDMA
# cecho "GREEN" "Run IRN RDMA experiments..."
# python3 run.py --lb fecmp --pfc 0 --irn 1 --simul_time ${RUNTIME} --netload ${NETLOAD} --topo ${TOPOLOGY} 2>&1 > /dev/null &
# sleep 5
# python3 run.py --lb letflow --pfc 0 --irn 1 --simul_time ${RUNTIME} --netload ${NETLOAD} --topo ${TOPOLOGY} 2>&1 > /dev/null &
# sleep 0.1
# python3 run.py --lb conga --pfc 0 --irn 1 --simul_time ${RUNTIME} --netload ${NETLOAD} --topo ${TOPOLOGY} 2>&1 > /dev/null &
# sleep 0.1
# python3 run.py --lb conweave --pfc 0 --irn 1 --simul_time ${RUNTIME} --netload ${NETLOAD} --topo ${TOPOLOGY} 2>&1 > /dev/null &
# sleep 0.1

# cecho "GREEN" "Runing all in parallel. Check the processors running on background!"