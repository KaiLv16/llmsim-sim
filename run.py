#!/usr/bin/python3
from genericpath import exists
import subprocess
import os
import time
from xmlrpc.client import boolean
import numpy as np
import copy
import shutil
import random
from datetime import datetime
import sys
import os
import argparse
from datetime import date
import shutil

from cut_output_file import * 

# randomID
random.seed(datetime.now())
MAX_RAND_RANGE = 1000000000

# config template
config_template = """TOPOLOGY_FILE config/{topo}.txt  ~
FLOW_FILE config/{flow}.txt                           ~
FLOW_RELATIONAL {flow_relation}                       newly_add
NODE_MAPPING config/{node_mapping}.txt                newly_add
SIM_NODE_MAPPING_FILE config/{simnode_mapping_file}.txt   newly_add
IS_SPRAY {spray_test}                                   newly_add

FLOW_INPUT_FILE mix/output/{id}/{id}_in.txt           new
CNP_OUTPUT_FILE mix/output/{id}/{id}_out_cnp.txt      new
FCT_OUTPUT_FILE mix/output/{id}/{id}_out_fct.txt      ~
PFC_OUTPUT_FILE mix/output/{id}/{id}_out_pfc.txt      ~
QLEN_MON_FILE mix/output/{id}/{id}_out_qlen.txt       ~
VOQ_MON_FILE mix/output/{id}/{id}_out_voq.txt         new
VOQ_MON_DETAIL_FILE mix/output/{id}/{id}_out_voq_per_dst.txt    new
UPLINK_MON_FILE mix/output/{id}/{id}_out_uplink.txt             new
CONN_MON_FILE mix/output/{id}/{id}_out_conn.txt                 new
EST_ERROR_MON_FILE mix/output/{id}/{id}_out_est_error.txt       new
SND_RCV_OUTPUT_FILE mix/output/{id}/{id}_snd_rcv_record_file.txt       newly_added
ROUTING_TABLE_OUTPUT_FILE mix/output/{id}/{id}_routing_table_record.txt newly_added
QLEN_MON_START {qlen_mon_start}    ~
QLEN_MON_END {qlen_mon_end}        ~
SW_MONITORING_INTERVAL {sw_monitoring_interval}      new

FLOWGEN_START_TIME {flowgen_start_time}              new
FLOWGEN_STOP_TIME {flowgen_stop_time}                new
BUFFER_SIZE {buffer_size}                            ~

CC_MODE {cc_mode}               ~
LB_MODE {lb_mode}               new
ENABLE_PFC {enabled_pfc}        new
ENABLE_IRN {enabled_irn}        new

CONWEAVE_TX_EXPIRY_TIME {cwh_tx_expiry_time}                        new
CONWEAVE_REPLY_TIMEOUT_EXTRA {cwh_extra_reply_deadline}             new
CONWEAVE_PATH_PAUSE_TIME {cwh_path_pause_time}                      new
CONWEAVE_EXTRA_VOQ_FLUSH_TIME {cwh_extra_voq_flush_time}            new
CONWEAVE_DEFAULT_VOQ_WAITING_TIME {cwh_default_voq_waiting_time}    new

ALPHA_RESUME_INTERVAL 1         ~~~~~~(is_50_in_dcqcn_paper)
RATE_DECREASE_INTERVAL 4        ~~~~~~(is_50_in_dcqcn_paper)
CLAMP_TARGET_RATE 0             ~
RP_TIMER 300                    ~~~~~~(is_300_in_dcqcn_paper)
FAST_RECOVERY_TIMES 1           ~
EWMA_GAIN {ewma_gain}           is_0.0625_in_dctcp,_else_is_0.00390625
RATE_AI {ai}Mb/s                ~
RATE_HAI {hai}Mb/s              ~
MIN_RATE 100Mb/s                ~~~~~~(is_1000Mb/s_in_hpcc_code)
DCTCP_RATE_AI {dctcp_ai}Mb/s    ~

ERROR_RATE_PER_LINK 0.0000      ~
L2_CHUNK_SIZE 4000              ~
L2_ACK_INTERVAL 1               ~
L2_BACK_TO_ZERO 0               ~
ACK_HIGH_PRIO {ack_high_prio}   ~

RATE_BOUND {rate_bound}         仅在确认下一个包的发送时间时有用
HAS_WIN {has_win}               ~
VAR_WIN {var_win}               is_0_in_dcqcn_and_timely
GLOBAL_T {use_global_win}       是否使用全局统一的窗口大小（maxBDP）

ENABLE_PLB {enable_plb}         ~
ENABLE_QLEN_AWARE_EG {qlen_aware_egress} ~

FAST_REACT {fast_react}         is_1_in_hpcc_algo,_else=0
MI_THRESH {mi}                  ~
INT_MULTI {int_multi}           ~

U_TARGET 0.95                   ~
MULTI_RATE 0                    ~
SAMPLE_FEEDBACK 0               ~

ENABLE_QCN 1                    ~
USE_DYNAMIC_PFC_THRESHOLD 1     ~
PACKET_PAYLOAD_SIZE 1000        ~


LINK_DOWN 0 0 0                 ~
KMAX_MAP {kmax_map}             ~
KMIN_MAP {kmin_map}             ~
PMAX_MAP {pmax_map}             ~
LOAD {load}                     new
RANDOM_SEED 1                   new
"""


# LB/CC mode matching，与HPCC的定义相同
cc_modes = {
    "dcqcn": 1,
    "hpcc": 3,
    "timely": 7,
    "dctcp": 8,
    "hpccPint": 10,
}

lb_modes = {
    "fecmp": 0,
    "drill": 2,
    "conga": 3,
    "letflow": 6,
    "conweave": 9,

    "host_spray": 11,
    "switch_spray": 12,
    "host_switch_spray": 13,
}

topo2bdp = {
    "leaf_spine_128_100G_OS2": 104000,  # 2-tier -> all 100Gbps
    "fat_k8_100G_OS2": 156000,  # 3-tier -> all 100Gbps
}

FLOWGEN_DEFAULT_TIME = 2.0  # see /traffic_gen/traffic_gen.py::base_t


def main():
    # make directory if not exists
    isExist = os.path.exists(os.getcwd() + "/mix/output/")
    if not isExist:
        os.makedirs(os.getcwd() + "/mix/output/")
        print("The new directory is created - {}".format(os.getcwd() + "/mix/output/"))

    parser = argparse.ArgumentParser(description='run simulation')
    parser.add_argument('--debug', dest='debug', action='store',
                        type=int, default=0, help="debug mode (default not enabled)")
    parser.add_argument('--spray', dest='spray_test', action='store',
                        type=int, default=0, help="enable Spray (default: 0)")
    parser.add_argument('--cc', dest='cc', action='store',
                        default='dcqcn', help="hpcc/dcqcn/timely/timely_vwin/dctcp (default: dcqcn)")
    parser.add_argument('--lb', dest='lb', action='store',
                        default='fecmp', help="fecmp/pecmp/drill/conga (default: fecmp)")
    parser.add_argument('--pfc', dest='pfc', action='store',
                        type=int, default=1, help="enable PFC (default: 1)")
    parser.add_argument('--irn', dest='irn', action='store',
                        type=int, default=0, help="enable IRN (default: 0)")
    parser.add_argument('--high_prio', dest='high_prio', action='store',
                        type=int, default=1, help="ACK/NACK goes to high prio (prio 0) (default: 1)")
    parser.add_argument('--simul_time', dest='simul_time', action='store',
                        default='0.1', help="traffic time to simulate (up to 3 seconds) (default: 0.1)")
    parser.add_argument('--buffer', dest="buffer", action='store',
                        default='9', help="the switch buffer size (MB) (default: 9)")
    parser.add_argument('--netload', dest='netload', action='store', type=int,
                        default=40, help="Network load at NIC to generate traffic (default: 40.0)")
    parser.add_argument('--bw', dest="bw", action='store',
                        default='100', help="the NIC bandwidth (Gbps) (default: 100)")
    parser.add_argument('--topo', dest='topo', action='store',
                        default='leaf_spine_128_100G', help="the name of the topology file (default: leaf_spine_128_100G_OS2)")
    parser.add_argument('--flow_file_name', dest='flow_file_name', action='store',
                        default='null', help="the name of the flow file. If this is not 'null', the program will use this given flow file. (default: 'null')")
    parser.add_argument('--flow_relation', dest='flow_relation', action='store',
                        default=0, help="If this is 1, read flow file from relational flow txt. (default: '0')")
    parser.add_argument('--node_mapping', dest='node_mapping', action='store',
                        default='../ResNet50-MNIST-pytorch/mix/node_mapping', help="Node Mappig file. (default: '../ResNet50-MNIST-pytorch/mix/node_mapping')")
    parser.add_argument('--simnode_mapping_file', dest='simnode_mapping_file', action='store',
                        default='layer_Id_to_nic_Id', help="SimNode Mappig file. (default: 'layer_Id_to_nic_Id')")
    parser.add_argument('--cdf', dest='cdf', action='store',
                        default='AliStorage2019', help="the name of the cdf file (default: AliStorage2019)")
    parser.add_argument('--enforce_win', dest='enforce_win', action='store',
                        type=int, default=0, help="enforce to use window scheme (default: 0)")
    parser.add_argument('--sw_monitoring_interval', dest='sw_monitoring_interval', action='store',
                        type=int, default=10000, help="interval of sampling statistics for queue status (default: 10000ns)")

    # #### CONWEAVE PARAMETERS ####
    # parser.add_argument('--cwh_extra_reply_deadline', dest='cwh_extra_reply_deadline', action='store',
    #                     type=int, default=4, help="extra-timeout, where reply_deadline = base-RTT + extra-timeout (default: 4us)")
    # parser.add_argument('--cwh_path_pause_time', dest='cwh_path_pause_time', action='store',
    #                     type=int, default=16, help="Time to pause the path with ECN feedback (default: 8us")
    # parser.add_argument('--cwh_extra_voq_flush_time', dest='cwh_extra_voq_flush_time', action='store',
    #                     type=int, default=16, help="Extra VOQ Flush Time (default: 8us for IRN)")
    # parser.add_argument('--cwh_default_voq_waiting_time', dest='cwh_default_voq_waiting_time', action='store',
    #                     type=int, default=400, help="Default VOQ Waiting Time (default: 400us)")
    # parser.add_argument('--cwh_tx_expiry_time', dest='cwh_tx_expiry_time', action='store',
    #                     type=int, default=1000, help="timeout value of ConWeave Tx for CLEAR signal (default: 1000us)")

    args = parser.parse_args()

    # make running ID of this config
    # need to check directory exists or not
    isExist = True
    spray_test = args.spray_test

    config_ID = "test_0"
    # while (isExist):      # 找一个新的序列号
    #     config_ID = str(random.randrange(MAX_RAND_RANGE))
    #     isExist = os.path.exists(os.getcwd() + "/mix/output/" + config_ID)

    # input parameters
    cc_mode = cc_modes[args.cc]
    lb_mode = lb_modes[args.lb]
    enabled_pfc = int(args.pfc)
    enabled_irn = int(args.irn)

    config_ID = f"{args.cc}({cc_mode})_{args.lb}({lb_mode})_pfc{enabled_pfc}_irn{enabled_irn}"
    print(f"config_ID: {config_ID}")

    # 如果本次参数和上次相等，就删除上次的输出
    with open('mix/last_param.txt', 'r') as file:
        first_line = file.readline().strip()  # 使用 .strip() 去除可能的换行符
        if first_line == config_ID:
            output_path = f'mix/output/{first_line}'
            # 如果相等，删除 out.txt 文件
            if os.path.exists(output_path):
                # 判断是文件还是目录，并删除
                if os.path.isfile(output_path):
                    os.remove(output_path)  # 删除文件
                    print(f"{output_path} 旧文件已删除")
                elif os.path.isdir(output_path):
                    shutil.rmtree(output_path)  # 递归删除目录及其内容
                    print(f"{output_path} 旧目录已删除")

    with open('mix/last_param.txt', 'w') as file:
        file.write(config_ID)


    bw = int(args.bw)
    buffer = args.buffer
    topo = args.topo    # 生成拓扑的文件名
    flow = args.flow_file_name # 直接读取的文件名
    flow_relation = int(args.flow_relation)
    node_mapping = args.node_mapping
    simnode_mapping_file = args.simnode_mapping_file
    enforce_win = args.enforce_win
    cdf = args.cdf      # 流分布的文件名
    flowgen_start_time = FLOWGEN_DEFAULT_TIME  # default: 2.0
    flowgen_stop_time = flowgen_start_time + \
        float(args.simul_time)  # default: 2.0
    sw_monitoring_interval = int(args.sw_monitoring_interval)

    # get over-subscription ratio from topoogy name

    hostload = 0
    netload = args.netload
    if (not spray_test):    
        oversub = int(topo.replace("\n", "").split("OS")[-1].replace(".txt", ""))
        assert (int(args.netload) % oversub == 0)
        hostload = int(args.netload) / oversub
        assert (hostload > 0)

    # Sanity checks
    if (args.cc == "timely" or args.cc == "hpcc") and args.lb == "conweave":
        raise Exception(
            "CONFIG ERROR : ConWeave currently does not support RTT-based protocols. Plz modify its logic accordingly.")
    if enabled_irn == 1 and enabled_pfc == 1:
        raise Exception(
            "CONFIG ERROR : If IRN is turn-on, then you should turn off PFC (for better perforamnce).")
    if enabled_irn == 0 and enabled_pfc == 0:
        raise Exception(
            "CONFIG ERROR : Either IRN or PFC should be true (at least one).")
    if float(args.simul_time) < 0.005:
        raise Exception("CONFIG ERROR : Runtime must be larger than 5ms (= warmup interval).")

    # sniff number of servers
    # 读取真正的拓扑文件
    with open("config/{topo}.txt".format(topo=args.topo), 'r') as f_topo:
        line = f_topo.readline().split(" ")
        n_host = int(line[0]) - int(line[1])

    assert (hostload >= 0 and hostload < 100)
    # 在config文件夹下
    if (args.flow_file_name == 'null'):
        print("no flow file specified")
        if (spray_test):
            flow = "spray_N_{n_host}_T_{time}s_B_{bw}_flow".format(
                n_host=n_host, time=int(float(args.simul_time)), bw=bw)
        else:
            flow = "L_{load:.2f}_CDF_{cdf}_N_{n_host}_T_{time}ms_B_{bw}_flow".format(
                load=hostload, cdf=args.cdf, n_host=n_host, time=int(float(args.simul_time)*1000), bw=bw)

    # check the file exists
    if (exists(os.getcwd() + "/config/" + flow + ".txt")):
        if (spray_test):
            print("Input SPRAY traffic <{_flow}.txt> already exists".format(_flow=flow))
        else:
            print("Input traffic file with load:{load:.2f}, cdf:{cdf}, n_host:{n_host} already exists".format(
                load=hostload, cdf=cdf, n_host=n_host))
    else:  # make the input traffic file
        if (args.flow_file_name != 'null'):
            print(f"You specified an input flow file name {args.flow_file_name}, but it cannot be found under /config/ folder. Exit...")
            exit(1)
        print("Generate a input traffic file...")
        if (spray_test):
            print("python ./traffic_gen/traffic_gen_spray.py -n {n_host} -o {output}".format(
                n_host=n_host,
                output=os.getcwd() + "/config/" + flow + ".txt"))

            os.system("python ./traffic_gen/traffic_gen_spray.py -n {n_host} -o {output}".format(
                n_host=n_host,
                output=os.getcwd() + "/config/" + flow + ".txt"))
        else:
            print("python ./traffic_gen/traffic_gen.py -c {cdf} -n {n_host} -l {load} -b {bw} -t {time} -o {output}".format(
                cdf=os.getcwd() + "/../traffic_gen/" + args.cdf + ".txt",
                n_host=n_host,
                load=hostload / 100.0,
                bw=args.bw + "G",
                time=args.simul_time,
                output=os.getcwd() + "/config/" + flow + ".txt"))

            os.system("python ./traffic_gen/traffic_gen.py -c {cdf} -n {n_host} -l {load} -b {bw} -t {time} -o {output}".format(
                cdf=os.getcwd() + "/traffic_gen/" + args.cdf + ".txt",
                n_host=n_host,
                load=hostload / 100.0,
                bw=args.bw + "G",
                time=args.simul_time,
                output=os.getcwd() + "/config/" + flow + ".txt"))

    # sanity check - bandwidth
    if (not spray_test):
        with open("config/{topo}.txt".format(topo=args.topo), 'r') as f_topo:
            first_line = f_topo.readline().split(" ")
            n_host = int(first_line[0]) - int(first_line[1])
            n_link = int(first_line[2])
            i = 0
            for line in f_topo.readlines()[1:]:
                i += 1
                if (i > n_link):
                    break
                parsed = line.split(" ")
                if len(parsed) > 2 and (int(parsed[0]) < n_host or int(parsed[1]) < n_host):
                    assert (int(parsed[2].replace("Gbps", "")) == int(bw))
        print("All NIC bandwidth is {bw}Gbps".format(bw=bw))

    ##################################################################
    ##########              ConWeave parameters             ##########
    ##################################################################
    if (lb_mode == 9):
        cwh_extra_reply_deadline = 4  # 4us, NOTE: this is "extra" term to base RTT
        cwh_path_pause_time = 16  # 8us (K_min) or 16us

        if "leaf_spine" in topo:  # 2-tier
            cwh_extra_voq_flush_time = 16
            cwh_default_voq_waiting_time = 200
            cwh_tx_expiry_time = 300  # 300us
        elif "fat" in topo and enabled_pfc == 0 and enabled_irn == 1:  # 3-tier, IRN
            cwh_extra_voq_flush_time = 16
            cwh_default_voq_waiting_time = 300
            cwh_tx_expiry_time = 1000  # 1ms
        elif "fat" in topo and enabled_pfc == 1 and enabled_irn == 0:  # 3-tier, Lossless
            cwh_extra_voq_flush_time = 64
            cwh_default_voq_waiting_time = 600
            cwh_tx_expiry_time = 1000  # 1ms
        else:
            raise Exception(
                "Unsupported ConWeave Parameter Setup")
    else:
        #### CONWEAVE PARAMETERS (DUMMY) ####
        cwh_extra_reply_deadline = 4
        cwh_path_pause_time = 16
        cwh_extra_voq_flush_time = 64
        cwh_default_voq_waiting_time = 400
        cwh_tx_expiry_time = 1000

    ##################################################################

    # make directory if not exists
    isExist = os.path.exists(os.getcwd() + "/mix/output/" + config_ID + "/")
    assert (not isExist)
    # if not isExist:
    os.makedirs(os.getcwd() + "/mix/output/" + config_ID + "/")
    print("The new directory is created  - {}".format(os.getcwd() +
          "/mix/output/" + config_ID + "/"))

    config_name = os.getcwd() + "/mix/output/" + config_ID + "/config.txt"

    ##################################################################
    ############             IMPORTANT ARGS               ############
    ##################################################################
    # By default, DCQCN uses no window (rate-based).
    has_win = 0
    var_win = 0
    rate_bound = 1
    enable_plb = 0    # 还没有实现，需要端侧先实现RTT检测
    qlen_aware_egress = 0
    if (cc_mode == 3 or cc_mode == 8 or enforce_win == 1):  # HPCC or DCTCP or enforcement
        has_win = 1
        var_win = 1
        if enforce_win == 1:
            print("### INFO: Enforced to use window scheme! ###")

    if (enabled_irn == 1 and 'spray' in args.lb and 'switch' in args.lb):  # 对于irn + packet_spray，使用可变窗口、逐QP窗口。
        rate_bound = 0
        has_win = 1
        var_win = 1
        use_global_max_win = 0
        qlen_aware_egress = 1

    # record to history
    simulday = datetime.now().strftime("%m/%d/%y")
    with open("./mix/.history", "a") as history:
        history.write("{simulday},{config_ID},{cc_mode},{lb_mode},{cwh_tx_expiry_time},{cwh_extra_reply_deadline},{cwh_path_pause_time},{cwh_extra_voq_flush_time},{cwh_default_voq_waiting_time},{pfc},{irn},{has_win},{var_win},{topo},{bw},{cdf},{load},{time}\n".format(
            simulday=simulday,
            config_ID=config_ID,
            cc_mode=cc_mode,
            lb_mode=lb_mode,
            cwh_tx_expiry_time=cwh_tx_expiry_time,
            cwh_extra_reply_deadline=cwh_extra_reply_deadline,
            cwh_path_pause_time=cwh_path_pause_time,
            cwh_extra_voq_flush_time=cwh_extra_voq_flush_time,
            cwh_default_voq_waiting_time=cwh_default_voq_waiting_time,
            pfc=enabled_pfc,
            irn=enabled_irn,
            has_win=has_win,
            var_win=var_win,
            topo=topo,
            bw=bw,
            cdf=cdf,
            load=netload,
            time=args.simul_time,
        ))

    if not spray_test:    # conweave原生
        # 1 BDP calculation（我们的拓扑不在这里）
        if topo2bdp.get(topo) == None:
            print("ERROR - topology is not registered in run.py!!", flush=True)
            return
        bdp = int(topo2bdp[topo])
        print("1BDP = {}".format(bdp))

        # DCQCN parameters (NOTE: HPCC's 400KB/1600KB is too large, although used in Microsoft)
        kmax_map = "6 %d %d %d %d %d %d %d %d %d %d %d %d" % (
            bw*200000000, 400, bw*500000000, 400, bw*1000000000, 400, bw*2*1000000000, 400, bw*2500000000, 400, bw*4*1000000000, 400)
        kmin_map = "6 %d %d %d %d %d %d %d %d %d %d %d %d" % (
            bw*200000000, 100, bw*500000000, 100, bw*1000000000, 100, bw*2*1000000000, 100, bw*2500000000, 100, bw*4*1000000000, 100)
        pmax_map = "6 %d %d %d %d %d %.2f %d %.2f %d %.2f %d %.2f" % (
            bw*200000000, 0.2, bw*500000000, 0.2, bw*1000000000, 0.2, bw*2*1000000000, 0.2, bw*2500000000, 0.2, bw*4*1000000000, 0.2)

        # queue monitoring
        qlen_mon_start = flowgen_start_time
        qlen_mon_end = flowgen_stop_time

        if (cc_mode == 1):  # DCQCN  不使用window
            ai = 10 * bw / 25
            hai = 25 * bw / 25
            dctcp_ai = 1000
            fast_react = 0
            mi = 0
            int_multi = 1
            ewma_gain = 0.00390625

            hprio = args.high_prio
            
            config = config_template.format(id=config_ID, topo=topo, flow=flow, flow_relation=flow_relation, node_mapping=node_mapping, simnode_mapping_file=simnode_mapping_file,
                                            spray_test=spray_test,
                                            qlen_mon_start=qlen_mon_start, qlen_mon_end=qlen_mon_end, flowgen_start_time=flowgen_start_time,
                                            flowgen_stop_time=flowgen_stop_time, sw_monitoring_interval=sw_monitoring_interval,
                                            load=netload, buffer_size=buffer, lb_mode=lb_mode, cwh_tx_expiry_time=cwh_tx_expiry_time,
                                            cwh_extra_reply_deadline=cwh_extra_reply_deadline, cwh_default_voq_waiting_time=cwh_default_voq_waiting_time,
                                            cwh_path_pause_time=cwh_path_pause_time, cwh_extra_voq_flush_time=cwh_extra_voq_flush_time,
                                            enabled_pfc=enabled_pfc, enabled_irn=enabled_irn,
                                            cc_mode=cc_mode,
                                            ai=ai, hai=hai, dctcp_ai=dctcp_ai,
                                            rate_bound=rate_bound, has_win=has_win, var_win=var_win, use_global_win=use_global_max_win,
                                            enable_plb=enable_plb, qlen_aware_egress=qlen_aware_egress, 
                                            fast_react=fast_react, mi=mi, int_multi=int_multi, ewma_gain=ewma_gain,
                                            kmax_map=kmax_map, kmin_map=kmin_map, pmax_map=pmax_map,
                                            ack_high_prio=hprio)
        else:
            print("unknown cc:{}".format(args.cc))
    ###################################################
    ##             MODIFIED FROM HPCC                ##
    ###################################################
    else:
        print('Using parameters in HPCC ...')
        # 还是用回hpcc的参数吧
        kmax_map = "2 %d %d %d %d"%(bw*1000000000, 400*bw/25, bw*4*1000000000, 400*bw*4/25)
        kmin_map = "2 %d %d %d %d"%(bw*1000000000, 100*bw/25, bw*4*1000000000, 100*bw*4/25)
        pmax_map = "2 %d %.2f %d %.2f"%(bw*1000000000, 0.2, bw*4*1000000000, 0.2)
        
        # queue monitoring
        qlen_mon_start = flowgen_start_time
        qlen_mon_end = flowgen_stop_time

        # some common params. modified in each CC
        ewma_gain = 0.00390625
        dctcp_ai = 1000

        if (cc_mode == 1):  # DCQCN 不使用window
            ai = 10 * bw / 25
            hai = 25 * bw / 25
            # 保持和HPCC一致
            dctcp_ai = 1000
            fast_react = 0  # HPCC中的 ‘us’ 参数
            mi = 0
            int_multi = 1
            ewma_gain = 0.00390625
            hprio = args.high_prio

            config = config_template.format(id=config_ID, topo=topo, flow=flow, flow_relation=flow_relation, node_mapping=node_mapping, simnode_mapping_file=simnode_mapping_file,
                                            spray_test=spray_test,
                                            qlen_mon_start=qlen_mon_start, qlen_mon_end=qlen_mon_end, flowgen_start_time=flowgen_start_time,
                                            flowgen_stop_time=flowgen_stop_time, sw_monitoring_interval=sw_monitoring_interval,
                                            load=netload, buffer_size=buffer, lb_mode=lb_mode, cwh_tx_expiry_time=cwh_tx_expiry_time,
                                            cwh_extra_reply_deadline=cwh_extra_reply_deadline, cwh_default_voq_waiting_time=cwh_default_voq_waiting_time,
                                            cwh_path_pause_time=cwh_path_pause_time, cwh_extra_voq_flush_time=cwh_extra_voq_flush_time,
                                            enabled_pfc=enabled_pfc, enabled_irn=enabled_irn,
                                            cc_mode=cc_mode,
                                            ai=ai, hai=hai, dctcp_ai=dctcp_ai,
                                            rate_bound=rate_bound, has_win=has_win, var_win=var_win, use_global_win=use_global_max_win,
                                            enable_plb=enable_plb, qlen_aware_egress=qlen_aware_egress, 
                                            fast_react=fast_react, mi=mi, int_multi=int_multi, ewma_gain=ewma_gain,
                                            kmax_map=kmax_map, kmin_map=kmin_map, pmax_map=pmax_map,
                                            ack_high_prio=hprio)
        elif (cc_mode == 3):    # hpcc 使用window
            ai = 10 * bw / 25
            hpcc_hpai = 50
            hpcc_utgt = 0.95
            mi = 0
            if hpcc_hpai > 0:
                ai = hpcc_hpai
            hai = ai # useless
            int_multi = bw / 25
            fast_react = 1
            ewma_gain = 0.00390625
            cc = "%s%d"%(args.cc, hpcc_utgt)
            if (mi > 0):
                cc += "mi%d"%mi
            if hpcc_hpai > 0:
                cc += "ai%d"%ai

            hprio = args.high_prio
            hprio = 0       # hpcc代码里写的
            # config_name = "mix/config_%s_%s_%s%s.txt"%(topo, trace, cc, failure)
            # config = config_template.format(bw=bw, trace=trace, topo=topo, cc=cc, mode=3, t_alpha=1, t_dec=4, t_inc=300, g=0.00390625, ai=ai, hai=hai, dctcp_ai=1000, has_win=1, vwin=1, us=1, u_tgt=u_tgt, mi=mi, int_multi=int_multi, pint_log_base=pint_log_base, pint_prob=pint_prob, ack_prio=0, link_down=args.down, failure=failure, kmax_map=kmax_map, kmin_map=kmin_map, pmax_map=pmax_map, buffer_size=bfsz, enable_tr=enable_tr)
            config = config_template.format(id=config_ID, topo=topo, flow=flow, flow_relation=flow_relation, node_mapping=node_mapping, simnode_mapping_file=simnode_mapping_file,
                                            spray_test=spray_test,
                                            qlen_mon_start=qlen_mon_start, qlen_mon_end=qlen_mon_end, flowgen_start_time=flowgen_start_time,
                                            flowgen_stop_time=flowgen_stop_time, sw_monitoring_interval=sw_monitoring_interval,
                                            load=netload, buffer_size=buffer, lb_mode=lb_mode, cwh_tx_expiry_time=cwh_tx_expiry_time,
                                            cwh_extra_reply_deadline=cwh_extra_reply_deadline, cwh_default_voq_waiting_time=cwh_default_voq_waiting_time,
                                            cwh_path_pause_time=cwh_path_pause_time, cwh_extra_voq_flush_time=cwh_extra_voq_flush_time,
                                            enabled_pfc=enabled_pfc, enabled_irn=enabled_irn,
                                            cc_mode=cc_mode,
                                            ai=ai, hai=hai, dctcp_ai=dctcp_ai,
                                            rate_bound=rate_bound, has_win=has_win, var_win=var_win, use_global_win=use_global_max_win,
                                            enable_plb=enable_plb, qlen_aware_egress=qlen_aware_egress, 
                                            fast_react=fast_react, mi=mi, int_multi=int_multi, ewma_gain=ewma_gain,
                                            kmax_map=kmax_map, kmin_map=kmin_map, pmax_map=pmax_map,
                                            ack_high_prio=hprio)
        elif (cc_mode == 8):    # dctcp 使用window
            ai = 10 # ai is useless for dctcp
            hai = ai  # also useless
            dctcp_ai=615 # calculated from RTT=13us and MTU=1KB, because DCTCP add 1 MTU per RTT.
            ewma_gain = 0.0625
            kmax_map = "2 %d %d %d %d"%(bw*1000000000, 30*bw/10, bw*4*1000000000, 30*bw*4/10)
            kmin_map = "2 %d %d %d %d"%(bw*1000000000, 30*bw/10, bw*4*1000000000, 30*bw*4/10)
            pmax_map = "2 %d %.2f %d %.2f"%(bw*1000000000, 1.0, bw*4*1000000000, 1.0)
            fast_react = 0
            mi = 0
            int_multi = 1
            hprio = args.high_prio
            hprio = 0       # hpcc代码里写的
            # config = config_template.format(bw=bw, trace=trace, topo=topo, cc=args.cc, mode=8, t_alpha=1, t_dec=4, t_inc=300, g=0.0625, ai=ai, hai=hai, dctcp_ai=dctcp_ai, has_win=1, vwin=1, us=0, u_tgt=u_tgt, mi=mi, int_multi=1, pint_log_base=pint_log_base, pint_prob=pint_prob, ack_prio=0, link_down=args.down, failure=failure, kmax_map=kmax_map, kmin_map=kmin_map, pmax_map=pmax_map, buffer_size=bfsz, enable_tr=enable_tr)
            config = config_template.format(id=config_ID, topo=topo, flow=flow, flow_relation=flow_relation, node_mapping=node_mapping, simnode_mapping_file=simnode_mapping_file,
                                            spray_test=spray_test,
                                            qlen_mon_start=qlen_mon_start, qlen_mon_end=qlen_mon_end, flowgen_start_time=flowgen_start_time,
                                            flowgen_stop_time=flowgen_stop_time, sw_monitoring_interval=sw_monitoring_interval,
                                            load=netload, buffer_size=buffer, lb_mode=lb_mode, cwh_tx_expiry_time=cwh_tx_expiry_time,
                                            cwh_extra_reply_deadline=cwh_extra_reply_deadline, cwh_default_voq_waiting_time=cwh_default_voq_waiting_time,
                                            cwh_path_pause_time=cwh_path_pause_time, cwh_extra_voq_flush_time=cwh_extra_voq_flush_time,
                                            enabled_pfc=enabled_pfc, enabled_irn=enabled_irn,
                                            cc_mode=cc_mode,
                                            ai=ai, hai=hai, dctcp_ai=dctcp_ai,
                                            rate_bound=rate_bound, has_win=has_win, var_win=var_win, use_global_win=use_global_max_win,
                                            enable_plb=enable_plb, qlen_aware_egress=qlen_aware_egress, 
                                            fast_react=fast_react, mi=mi, int_multi=int_multi, ewma_gain=ewma_gain,
                                            kmax_map=kmax_map, kmin_map=kmin_map, pmax_map=pmax_map,
                                            ack_high_prio=hprio)
        elif (cc_mode == 7):    # timely 不使用window
            ai = 10 * bw / 10
            hai = 50 * bw / 10
            ewma_gain = 0.00390625
            fast_react = 0
            mi = 0
            int_multi = 1
            hprio = args.high_prio
            # config = config_template.format(bw=bw, trace=trace, topo=topo, cc=args.cc, mode=7, t_alpha=1, t_dec=4, t_inc=300, g=0.00390625, ai=ai, hai=hai, dctcp_ai=1000, has_win=0, vwin=0, us=0, u_tgt=u_tgt, mi=mi, int_multi=1, pint_log_base=pint_log_base, pint_prob=pint_prob, ack_prio=1, link_down=args.down, failure=failure, kmax_map=kmax_map, kmin_map=kmin_map, pmax_map=pmax_map, buffer_size=bfsz, enable_tr=enable_tr)
            config = config_template.format(id=config_ID, topo=topo, flow=flow, flow_relation=flow_relation, node_mapping=node_mapping, simnode_mapping_file=simnode_mapping_file,
                                            spray_test=spray_test,
                                            qlen_mon_start=qlen_mon_start, qlen_mon_end=qlen_mon_end, flowgen_start_time=flowgen_start_time,
                                            flowgen_stop_time=flowgen_stop_time, sw_monitoring_interval=sw_monitoring_interval,
                                            load=netload, buffer_size=buffer, lb_mode=lb_mode, cwh_tx_expiry_time=cwh_tx_expiry_time,
                                            cwh_extra_reply_deadline=cwh_extra_reply_deadline, cwh_default_voq_waiting_time=cwh_default_voq_waiting_time,
                                            cwh_path_pause_time=cwh_path_pause_time, cwh_extra_voq_flush_time=cwh_extra_voq_flush_time,
                                            enabled_pfc=enabled_pfc, enabled_irn=enabled_irn,
                                            cc_mode=cc_mode,
                                            ai=ai, hai=hai, dctcp_ai=dctcp_ai,
                                            rate_bound=rate_bound, has_win=has_win, var_win=var_win, use_global_win=use_global_max_win,
                                            enable_plb=enable_plb, qlen_aware_egress=qlen_aware_egress, 
                                            fast_react=fast_react, mi=mi, int_multi=int_multi, ewma_gain=ewma_gain,
                                            kmax_map=kmax_map, kmin_map=kmin_map, pmax_map=pmax_map,
                                            ack_high_prio=hprio)
        elif args.cc == "timely_vwin":  # not used for now
            ai = 10 * bw / 10
            hai = 50 * bw / 10
            ewma_gain = 0.00390625
            fast_react = 0
            mi = 0
            int_multi = 1
            hprio = args.high_prio
            # config = config_template.format(bw=bw, trace=trace, topo=topo, cc=args.cc, mode=7, t_alpha=1, t_dec=4, t_inc=300, g=0.00390625, ai=ai, hai=hai, dctcp_ai=1000, has_win=1, vwin=1, us=0, u_tgt=u_tgt, mi=mi, int_multi=1, pint_log_base=pint_log_base, pint_prob=pint_prob, ack_prio=1, link_down=args.down, failure=failure, kmax_map=kmax_map, kmin_map=kmin_map, pmax_map=pmax_map, buffer_size=bfsz, enable_tr=enable_tr)
            config = config_template.format(id=config_ID, topo=topo, flow=flow, flow_relation=flow_relation, node_mapping=node_mapping, simnode_mapping_file=simnode_mapping_file,
                                            spray_test=spray_test,
                                            qlen_mon_start=qlen_mon_start, qlen_mon_end=qlen_mon_end, flowgen_start_time=flowgen_start_time,
                                            flowgen_stop_time=flowgen_stop_time, sw_monitoring_interval=sw_monitoring_interval,
                                            load=netload, buffer_size=buffer, lb_mode=lb_mode, cwh_tx_expiry_time=cwh_tx_expiry_time,
                                            cwh_extra_reply_deadline=cwh_extra_reply_deadline, cwh_default_voq_waiting_time=cwh_default_voq_waiting_time,
                                            cwh_path_pause_time=cwh_path_pause_time, cwh_extra_voq_flush_time=cwh_extra_voq_flush_time,
                                            enabled_pfc=enabled_pfc, enabled_irn=enabled_irn,
                                            cc_mode=cc_mode,
                                            ai=ai, hai=hai, dctcp_ai=dctcp_ai,
                                            rate_bound=rate_bound, has_win=has_win, var_win=var_win, use_global_win=use_global_max_win,
                                            enable_plb=enable_plb, qlen_aware_egress=qlen_aware_egress, 
                                            fast_react=fast_react, mi=mi, int_multi=int_multi, ewma_gain=ewma_gain,
                                            kmax_map=kmax_map, kmin_map=kmin_map, pmax_map=pmax_map,
                                            ack_high_prio=hprio)
        
        else:
            print("unknown cc:{}".format(args.cc))

    with open(config_name, "w") as file:
        file.write(config)

    # run program
    print("Running simulation...")
    output_log = config_name.replace(".txt", ".log")
    run_command = "./waf --run 'scratch/network-load-balance {config_name}' > {output_log} 2>&1".format(
        config_name=config_name, output_log=output_log)
    with open("./mix/.history", "a") as history:
        history.write(run_command + "\n")
        history.write(
            "./waf --run 'scratch/network-load-balance' --command-template='gdb --args %s {config_name}'\n".format(
                config_name=config_name)
        )
        history.write("\n")

    # print(run_command)
    # os.system("./waf --run 'scratch/network-load-balance {config_name}' > {output_log} 2>&1".format(
    #     config_name=config_name, output_log=output_log))

    # os.system("./waf --run 'scratch/network-load-balance {config_name}' > {output_log}".format(
    #     config_name=config_name, output_log=output_log))
    print(    "./waf --run 'scratch/network-load-balance {config_name}'".format(
        config_name=config_name, output_log=output_log))
    
    if args.debug == 0:
        # print("111111111")
        os.system("./waf --run 'scratch/network-load-balance {config_name}'".format(
            config_name=config_name, output_log=output_log))
    else:
        print("./waf shell")
        # os.system("./waf shell")
        os.system(""" ./waf --run scratch/network-load-balance --command-template="gdb --args %s {config_name}" """.format(
            config_name=config_name, output_log=output_log))
    
    print("config_ID: ", config_ID)
    
    if spray_test:
        print("TODO: Analyze the output")
        pass
    else:
        ####################################################
        #                 Analyze the output FCT           #
        ####################################################
        # NOTE: collect data except warm-up and cold-finish period
        fct_analysis_time_limit_begin = int(
            flowgen_start_time * 1e9) + int(0.005 * 1e9)  # warmup
        fct_analysistime_limit_end = int(
            flowgen_stop_time * 1e9) + int(0.05 * 1e9)  # extra term

        print("Analyzing output FCT...")
        print("python3 fctAnalysis.py -id {config_ID} -dir {dir} -bdp {bdp} -sT {fct_analysis_time_limit_begin} -fT {fct_analysistime_limit_end} > /dev/null 2>&1".format(
            config_ID=config_ID, dir=os.getcwd(), bdp=bdp, fct_analysis_time_limit_begin=fct_analysis_time_limit_begin, fct_analysistime_limit_end=fct_analysistime_limit_end))
        os.system("python3 fctAnalysis.py -id {config_ID} -dir {dir} -bdp {bdp} -sT {fct_analysis_time_limit_begin} -fT {fct_analysistime_limit_end} > /dev/null 2>&1".format(
            config_ID=config_ID, dir=os.getcwd(), bdp=bdp, fct_analysis_time_limit_begin=fct_analysis_time_limit_begin, fct_analysistime_limit_end=fct_analysistime_limit_end))

        if lb_mode == 9: # ConWeave Logging
            ################################################################
            #             Analyze hardware resource of ConWeave            #
            ################################################################
            # NOTE: collect data except warm-up and cold-finish period
            queue_analysis_time_limit_begin = int(
                flowgen_start_time * 1e9) + int(0.005 * 1e9)  # warmup
            queue_analysistime_limit_end = int(flowgen_stop_time * 1e9)
            print("Analyzing output Queue...")
            print("python3 queueAnalysis.py -id {config_ID} -dir {dir} -sT {queue_analysis_time_limit_begin} -fT {queue_analysistime_limit_end} > /dev/null 2>&1".format(
                config_ID=config_ID, dir=os.getcwd(), queue_analysis_time_limit_begin=queue_analysis_time_limit_begin, queue_analysistime_limit_end=queue_analysistime_limit_end))
            os.system("python3 queueAnalysis.py -id {config_ID} -dir {dir} -sT {queue_analysis_time_limit_begin} -fT {queue_analysistime_limit_end} > /dev/null 2>&1".format(
                config_ID=config_ID, dir=os.getcwd(), queue_analysis_time_limit_begin=queue_analysis_time_limit_begin, queue_analysistime_limit_end=queue_analysistime_limit_end,
                monitoringInterval=sw_monitoring_interval))  # TODO: parameterize

        print("\n\n============== Done ============== ")
    return config_ID


if __name__ == "__main__":

    config_ID = main()
    # 调用函数
    save_first_xxx_lines(100000, f'mix/output/{config_ID}/{config_ID}_snd_rcv_record_file')



