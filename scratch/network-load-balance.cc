/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2023 NUS
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation;
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 * Authors: Chahwan Song <skychahwan@gmail.com>
 */

#include <ns3/assert.h>
#include <ns3/rdma-client-helper.h>
#include <ns3/rdma-client.h>
#include <ns3/rdma-driver.h>
#include <ns3/rdma.h>
#include <ns3/sim-setting.h>
#include <ns3/switch-node.h>
#include <time.h>
#include <regex>
#include <algorithm> // 必须包含

#include <fstream>
#include <iostream>
#include <unordered_map>

#include "ns3/applications-module.h"
#include "ns3/broadcom-node.h"
#include "ns3/conga-routing.h"
#include "ns3/conweave-voq.h"
#include "ns3/core-module.h"
#include "ns3/error-model.h"
#include "ns3/global-route-manager.h"
#include "ns3/internet-module.h"
#include "ns3/ipv4-static-routing-helper.h"
#include "ns3/letflow-routing.h"
#include "ns3/packet.h"
#include "ns3/point-to-point-helper.h"
#include "ns3/qbb-helper.h"
#include "ns3/qbb-net-device.h"
#include "ns3/rdma-hw.h"
#include "ns3/settings.h"

#include <fstream>
#include <sstream>
#include <queue>
#include <map>
#include <vector>

using namespace ns3;
using namespace std;

NS_LOG_COMPONENT_DEFINE("GENERIC_SIMULATION");

std::ostringstream oss;

float global_sim_start_time = 2.0;
/*------Load balancing parameters-----*/
// mode for load balancer, 0: flow ECMP, 2: DRILL, 3: Conga, 6: Letflow, 9: ConWeave
uint32_t lb_mode = 0;

// Conga params (based on paper recommendation)
Time conga_flowletTimeout = MicroSeconds(100);  // 100us
Time conga_dreTime = MicroSeconds(50);
Time conga_agingTime = MicroSeconds(500);
uint32_t conga_quantizeBit = 3;
double conga_alpha = 0.2;

// Letflow params
Time letflow_flowletTimeout = MicroSeconds(100);  // 100us
Time letflow_agingTime = MilliSeconds(2);  // just to clear the unused map entries for simul speed

// Conweave params
Time conweave_extraReplyDeadline = MicroSeconds(4);       // additional term to reply deadline
Time conweave_pathPauseTime = MicroSeconds(8);            // time to send packets to congested path
Time conweave_txExpiryTime = MicroSeconds(1000);          // waiting time for CLEAR
Time conweave_extraVOQFlushTime = MicroSeconds(32);       // extra for uncertainty
Time conweave_defaultVOQWaitingTime = MicroSeconds(500);  // default flush timer if no history
bool conweave_pathAwareRerouting = true;

/*------------------------ simulation variables -----------------------------*/
uint64_t one_hop_delay = 1000;  // nanoseconds
uint32_t cc_mode = 1;           // mode for congestion control, 1: DCQCN
bool enable_qcn = true, enable_pfc = true, use_dynamic_pfc_threshold = true;
uint32_t packet_payload_size = 1000, l2_chunk_size = 0, l2_ack_interval = 0;
double pause_time = 5;  // PFC pause, microseconds
double flowgen_start_time = 2.0, flowgen_stop_time = 2.5, simulator_extra_time = 0.1;
// queue length monitoring time is not used in this simulator
// uint32_t qlen_dump_interval = 100000000, qlen_mon_interval = 1000;  // ns
uint64_t qlen_mon_start;               // ns
uint64_t qlen_mon_end;                 // ns
uint32_t switch_mon_interval = 10000;  // 10000 ns
uint64_t cnp_mon_start;                // ns
uint64_t cnp_monitor_bucket = 100000;  // 100000 ns
uint64_t irn_mon_start;                // ns
uint64_t irn_monitor_bucket = 100000;  // 100000 ns

FILE *pfc_file = NULL;              // 被get_pfc()调用
FILE *fct_output = NULL;            // 被RdmaDriver->QpComplete()回调，回调函数在third.cc的qp_finish()
FILE *flow_input_stream = NULL;     // 这个事实上没有用到。调度之后的流才会写进到这个file里面
FILE *cnp_output = NULL;            // 在cnp_freq_monitoring()中写入
FILE *est_error_output = NULL;      // 在conweave_history_print()中写入
FILE *voq_output = NULL;            // 仅供conweave使用，在periodic_monitoring()中写入
FILE *voq_detail_output = NULL;     // 仅供conweave使用，在periodic_monitoring()中写入
FILE *uplink_output = NULL;         // 每10us调用一次periodic_monitoring()
FILE *conn_output = NULL;           // 

FILE *routing_table_output = NULL;
FILE *snd_rcv_output = NULL;

std::string data_rate, link_delay, topology_file, flow_file, node_mapping, simnode_mapping_file;
uint32_t is_flow_relational = 0;
std::string flow_input_file = "flow.txt";   // 这个事实上没有用到。调度之后的流才会写进到这个file里面
std::string fct_output_file = "fct.txt";
std::string pfc_output_file = "pfc.txt";
std::string cnp_output_file = "cnp.txt";
std::string qlen_mon_file = "qlen.txt";
std::string voq_mon_file = "voq.txt";
std::string voq_mon_detail_file = "voq_detail.txt";
std::string uplink_mon_file = "uplink.txt";
std::string conn_mon_file = "conn.txt";
std::string est_error_output_file = "est_error.txt";

std::string routing_table_output_file = "routing_table.txt";
std::string snd_rcv_output_file = "snd_rcv_output_file.txt";
std::string flow_statistics_output_file = "flow_statistics_output.txt";

// CC params
double alpha_resume_interval = 55, rp_timer = 300, ewma_gain = 1 / 16;
double rate_decrease_interval = 4;
uint32_t fast_recovery_times = 1;
std::string rate_ai, rate_hai, min_rate = "100Mb/s";
std::string dctcp_rate_ai = "1000Mb/s";

bool clamp_target_rate = false, l2_back_to_zero = false;
double error_rate_per_link = 0.0;
uint32_t is_spray = 1;
uint32_t has_win = 1;
uint32_t enable_plb = 0;
uint32_t qlen_aware_egress = 0;
uint32_t per_qp_window = 0;
uint32_t global_t = 1;
uint32_t mi_thresh = 5;
bool var_win = false, fast_react = true;
bool multi_rate = true;
bool sample_feedback = false;
double u_target = 0.95;
uint32_t int_multi = 1;
bool rate_bound = true;

unordered_map<uint64_t, uint32_t> rate2kmax, rate2kmin;
unordered_map<uint64_t, double> rate2pmax;
unordered_map<uint32_t, Ptr<SwitchNode>> idxNodeToR;  // nodeId和ToR switch指针的对应关系, Id -> Ptr

// 默认是1，走高优先级
uint32_t ack_high_prio = 1;

// config of link-down scenario, ACK priority, and buffer
uint64_t link_down_time = 0;
uint32_t link_down_A = 0, link_down_B = 0;
uint32_t buffer_size = 0;  // 0 to set buffer size automatically

// HPCC里没迁移过来的参数
// uint32_t enable_trace = 1;

// Added from Here
double load = 10.0;
int enable_irn = 0;
int random_seed = 1;  // change this randomly if you want random expt


/************************************************
 * Runtime varibles (from HPCC)
 ***********************************************/

uint64_t maxRtt, maxBdp;

// app parameters
struct Interface {
    uint32_t idx;
    uint32_t peerIdx;
    bool up;
    uint64_t delay;
    uint64_t bw;

    Interface() : idx(0), up(false) {}
};

// 从node1到下一跳node2的port是哪个
map<Ptr<Node>, map<Ptr<Node>, vector<Interface>>> nbr2if;
// Mapping destination to next hop for each node: <node, <dest, <nexthop0, ...> > >
// 从node到dest可以走哪些nexthop
map<Ptr<Node>, map<Ptr<Node>, vector<Ptr<Node>>>> nextHop;
map<Ptr<Node>, map<Ptr<Node>, uint64_t>> pairDelay;
map<Ptr<Node>, map<Ptr<Node>, uint64_t>> pairTxDelay;
map<Ptr<Node>, map<Ptr<Node>, uint64_t>> pairBw;
map<Ptr<Node>, map<Ptr<Node>, uint64_t>> pairBdp;
map<Ptr<Node>, map<Ptr<Node>, uint64_t>> pairRtt;

// for uplink/Downlink monitoring at TOR switches (load balance performance)
std::map<uint32_t, std::vector<uint32_t>> torId2UplinkIf;
std::map<uint32_t, std::vector<uint32_t>> torId2DownlinkIf;

// 新增
// 从node1到下一跳node2 有多少条link
map<uint32_t, map<uint32_t, uint32_t>> nodeId2NodeIdConnNum;

// input files
std::ifstream topof, flowf, mappingf;

NodeContainer n;                         // node container
std::vector<Ipv4Address> serverAddress;  // server address

// flow generator
std::unordered_map<uint32_t, uint32_t> flows_per_host;
uint32_t flow_id = 0;
std::unordered_map<uint32_t, uint16_t> portNumber;
std::unordered_map<uint32_t, uint16_t> dportNumber;
uint16_t *port_per_host;

// Scheduling input flows from flow.txt
struct FlowInput {
    uint32_t src, dst, pg, maxPacketCount, port;    // 取消了 hpcc 的 dport
    double start_time;
    uint32_t idx;
};
FlowInput flow_input = {0};  // global variable
uint32_t flow_num = 0;


/*------  parsing flows and their dependencies  -----*/
struct Flow {
    uint32_t id;
    int pg; // 新增优先级字段
    uint32_t src;
    uint32_t dst;
    int size;
    uint32_t lat;
    int64_t baseTxTime;
    int64_t TxTime;
    int64_t TxStartTime;
    int64_t TxFinishTime;
    float slowDown;
    int64_t theoreticalStartTime;
    int64_t theoreticalFinishTime;

    vector<uint32_t> dependFlows;
    vector<uint32_t> invokeFlows;
    std::string note; // 新增成员，用于存储 note 内容
    bool sent;
    
    Flow() {
        size = -1;
        TxTime = 0;
        baseTxTime = 0;
        TxStartTime = 0;
        TxFinishTime = 0;
        slowDown = -1;
        theoreticalStartTime = 0;
        theoreticalFinishTime = 0;
    }

    void print(bool simple = true, const std::string& outputTarget = "stdout") {
        std::ostream* outStream = &std::cout; // 默认输出到标准输出
        std::ofstream outFile;

        // 如果输出目标是文件，则打开文件
        if (outputTarget != "stdout") {
            outFile.open(outputTarget, std::ios::app);
            if (!outFile) {
                std::cerr << "Error opening file: " << outputTarget << std::endl;
                return;
            }
            outStream = &outFile; // 改变输出流为文件流
        }

        uint64_t base_ns = global_sim_start_time * 1000000000;

        // 使用 outStream 进行输出
        *outStream << "FlowId=" << id << " priority=" << pg << " src=" << src << " dst=" << dst 
                << " size=" << size << " lat=" << lat << " dep_flow=[";
        
        for (int i = 0; i < dependFlows.size(); i++) {
            *outStream << dependFlows[i] << " ";
        }
        *outStream << "] invoke_flow=[";

        for (int i = 0; i < invokeFlows.size(); i++) {
            *outStream << invokeFlows[i] << " ";
        }
        *outStream << "], note=\"" << note << "\" \n";

        if (!simple) {
            *outStream << "    TxTime=" << TxTime 
                    << ", baseTxTime=" << baseTxTime 
                    << ", TxStartTime=" << (TxStartTime - base_ns) 
                    << ", TxFinishTime=" << (TxFinishTime - base_ns) 
                    << ", slowDown=" << slowDown 
                    << ", theoreticalStartTime=" << (theoreticalStartTime - base_ns) 
                    << ", theoreticalFinishTime=" << (theoreticalFinishTime - base_ns) << "\n";
        }

        // 如果使用文件输出，则关闭文件
        if (outputTarget != "stdout") {
            outFile.close();
        }
    }
};
float maxSlowDown = 0;

// key: Flow.id   value: Flow结构体
map<uint32_t, Flow> flowMap;
void PrintFlowMap(bool flow_only=true, bool simple=false, const std::string& outputTarget="flowstatistics.txt") {
    std::cout << "flowMap" << std::endl;
    // can do some statistics here
    int64_t RealEndTime = 0;
    int64_t IdealEndTime = 0;
    for (auto it = flowMap.begin(); it != flowMap.end(); ++it) {
        uint32_t flowId = it->first;
        Flow flow = it->second;
        if (!(flow_only && flow.pg == -1)){
            // printf("Flow ID: %u\n", flowId);
            flow.print(simple, outputTarget);  // 调用 Flow 结构中的打印函数
            IdealEndTime = std::max(flow.theoreticalFinishTime - int64_t(global_sim_start_time * 1000000000.0), IdealEndTime);
            RealEndTime = std::max(flow.TxFinishTime - int64_t(global_sim_start_time * 1000000000), RealEndTime);
        }
    }
    std::ostream* outStream = &std::cout; // 默认输出到标准输出
    std::ofstream outFile;
    if (outputTarget != "stdout") {
        outFile.open(outputTarget, std::ios::app);
        if (!outFile) {
            std::cerr << "Error opening file: " << outputTarget << std::endl;
            return;
        }
        outStream = &outFile; // 改变输出流为文件流
    }
    *outStream << "    IdealEndTime=" << IdealEndTime / 1000000.0
               << "ms, RealEndTime=" << RealEndTime / 1000000.0
               << "ms, totalSlowDown=" << double(RealEndTime) / double(IdealEndTime)
               << "\n";
    if (outputTarget != "stdout") {
        outFile.close();
    }
}

// key: Flow.id   value: 每条流的依赖数量
map<uint32_t, int> dependencies;
void PrintDependencies() {
    std::cout << "dependencies" << std::endl;
    for (auto it = dependencies.begin(); it != dependencies.end(); ++it) {
        uint32_t flowId = it->first;
        int dependCount = it->second;
        printf("Flow ID: %u, Dependencies: %d\n", flowId, dependCount);
    }
    printf("\n");
}

// 存放ready的FLow的id
queue<uint32_t> readyQueue;
void PrintReadyQueue() {
    queue<uint32_t> tempQueue = readyQueue;  // 创建队列的副本
    printf("Ready Queue: ");
    while (!tempQueue.empty()) {
        uint32_t flowId = tempQueue.front();
        printf("%u ", flowId);
        tempQueue.pop();
    }
    printf("\n");
    printf("\n");
}

void trim(string &s) {
    // Remove leading and trailing spaces
    s.erase(0, s.find_first_not_of(' ')); // trim leading spaces
    s.erase(s.find_last_not_of(' ') + 1); // trim trailing spaces
}

void ParseRelationalFlowFile(string fileName) {
    std::cout << "read   relational   flow    input" << std::endl;
    ifstream file(fileName);
    string line;

    // 读取第一行，获取流的总数量
    // flowf.open(flow_file.c_str());
    // flowf >> flow_num;
    getline(file, line);
    flow_num = stoi(line);
    printf("Total flow num: %d\n", flow_num);

    uint32_t cur_flow_num = 0;
    // 逐行解析每条流的信息
    while (cur_flow_num < flow_num) {
        getline(file, line);
        istringstream iss(line);
        Flow flow;
        string dependStr, invokeStr;
        float fsize;

        // 解析流ID、源、目的和大小
        iss >> flow.id;
        iss.ignore(2);  // 跳过 ", "
        iss >> flow.pg; // 解析优先级
        iss.ignore(6);  // 跳过 ", src="
        iss >> flow.src;
        iss.ignore(6);  // 跳过 ", dst="
        iss >> flow.dst;
        // 检查是否有大小字段
        if (line.find("size=") != string::npos) {
            iss.ignore(7);  // 跳过 ", size="
            iss >> fsize;
            flow.size = (int)fsize;
        } else {
            flow.size = -1; // 或者设定为一个默认值
        }
        iss.ignore(6);  // 跳过 ", dst="
        float latency;
        iss >> latency;
        flow.lat = (int)(latency * 1000000);
        iss.ignore(15); // 跳过 ", depend_flow=["
        getline(iss, dependStr, ']');
        iss.ignore(15); // 跳过 ", invoke_flow=["
        getline(iss, invokeStr, ']');

        // 解析 depend_flow
        istringstream depStream(dependStr);
        string dep;
        while (getline(depStream, dep, ',')) {
            trim(dep); // 去除空格
            if (!dep.empty()) {
                // cout << dep << " ";
                flow.dependFlows.push_back(stoi(dep));
            }
        }

        // 解析 invoke_flow
        istringstream invStream(invokeStr);
        string inv;
        while (getline(invStream, inv, ',')) {
            trim(inv); // 去除空格
            if (!inv.empty()) {
                flow.invokeFlows.push_back(stoi(inv));
            }
        }
        // 使用正则表达式提取 note 内容
        std::regex note_regex(R"(note=<([^>]*)>)");
        std::smatch note_match;
        if (std::regex_search(line, note_match, note_regex)) {
            std::string note_content = note_match[1]; // 获取 <xxx> 中的内容
            flow.note = note_content; // 假设 Flow 类中有一个 string 类型的 note 成员
        }
        // if (cur_flow_num < 40)
        //     flow.print();

        // assert(n.Get(flow.src)->GetNodeType() == 0 &&
            //    n.Get(flow.dst)->GetNodeType() == 0);

        // flow.print();

        flow.sent = false;
        flowMap[flow.id] = flow;
        dependencies[flow.id] = flow.dependFlows.size();

        // 如果没有依赖流，直接加入readyQueue
        if (flow.dependFlows.empty()) {
            readyQueue.push(flow.id);
        }

        cur_flow_num += 1;
    }
}

map<uint32_t, uint32_t> vnode2node;     // 从node_mapping文件中解析的
void ParseNodeMapping(string filename) {
    ifstream infile(filename);
    string line;

    // 检查文件是否成功打开
    if (!infile.is_open()) {
        cerr << "无法打开文件: " << filename << endl;
        return;
    }

    // 逐行读取文件
    while (getline(infile, line)) {
        uint32_t key, value=-1;
        char arrow; // 用于读取 '->' 中的字符

        // 使用字符串流解析每一行
        stringstream ss(line);
        ss >> key >> arrow >> arrow >> value; // 读取key和value

        // 存储到map中
        vnode2node[key] = value;
    }
    
    // // 打印结果以验证
    // for (const auto& pair : vnode2node) {
    //     cout << "vnode: " << pair.first << ", node: " << pair.second << endl;
    // }

    infile.close(); // 关闭文件
}


map<uint32_t, uint32_t> node2phynode;   // 仿真中进行映射的
void DoNodeSimulationMapping(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "无法打开文件: " << filename << std::endl;
        return;
    }

    std::string line;
    int lineCount;
    
    // 读取行数
    if (std::getline(file, line)) {
        lineCount = std::stoi(line);
    } else {
        std::cerr << "无法读取行数" << std::endl;
        return;
    }

    // 读取每一行
    for (int i = 0; i < lineCount; ++i) {
        if (std::getline(file, line)) {
            std::istringstream iss(line);
            uint32_t node, phynode;
            char comma; // 用于读取逗号
            
            if (iss >> node >> comma >> phynode) {
                node2phynode[node] = phynode; // 保存映射关系
            } else {
                std::cerr << "格式错误: " << line << std::endl;
            }
        } else {
            std::cerr << "读取行失败或行数不足" << std::endl;
            break;
        }
    }
    
    file.close();
}

void RelationalFlowStart(uint32_t flowid);
void RelationalFlowEnd(uint32_t flowid);
void ScheduleFlowRelational();

void RelationalFlowStart(uint32_t flowid) {
    NS_LOG_DEBUG("Schedule RelationalFlow at " << Simulator::Now());
    Flow& currentFlow = flowMap[flowid];

    uint32_t pg, src, dst, sport, dport, maxPacketCount, target_len;
    pg = currentFlow.pg;
    // src = currentFlow.src;
    // dst = currentFlow.dst;
    src = node2phynode[vnode2node[currentFlow.src]];
    dst = node2phynode[vnode2node[currentFlow.dst]];

    // only works for those flow doesn't have pre-conditions.
    if (currentFlow.dependFlows.size() == 0) {      // 这些flow没有任何前置流。
        currentFlow.theoreticalStartTime = Simulator::Now().GetTimeStep();
        printf("Flow %u: Bind the start time to the starting stream.\n", flowid);
    }

    if (pg == 3) {
        uint64_t baseTxTime_ns = double(currentFlow.size) * 8.0 / 95.0 + double(pairRtt[n.Get(src)][n.Get(dst)]);      // ms, 95 means 95b/ns
        currentFlow.theoreticalFinishTime = currentFlow.theoreticalStartTime + baseTxTime_ns;        // ns
        currentFlow.TxStartTime = Simulator::Now().GetTimeStep();
        currentFlow.baseTxTime = baseTxTime_ns;
        std::cerr << Simulator::Now() << " Sending flow " << currentFlow.id << 
        ": node " << currentFlow.src << " (" << src << ") -> node " << currentFlow.dst << " (" << dst << 
        "), note=\"" << currentFlow.note << "\", size=" << currentFlow.size << ", IdealTxtime=" << baseTxTime_ns/1000 << "us" << std::endl;
        if (src == dst) {
            std::cerr << "\nSRC node == DST node!\n\n";
        }
        // src port
        sport = portNumber[src];  // get a new port number
        portNumber[src] = portNumber[src] + 1;

        // dst port
        dport = dportNumber[dst];
        dportNumber[dst] = dportNumber[dst] + 1;
        target_len = currentFlow.size;  // this is actually not packet-count, but bytes
        // if (target_len == 0) {
        //     target_len = 1;
        // }
        // 没变
        RdmaClientHelper clientHelper(
            pg, serverAddress[src], serverAddress[dst], sport, dport, target_len,
            has_win ? (global_t == 1 ? maxBdp : pairBdp[n.Get(src)][n.Get(dst)]) : 0,
            global_t == 1 ? maxRtt : pairRtt[n.Get(src)][n.Get(dst)]);
            
        clientHelper.SetAttribute("StatFlowID", IntegerValue(currentFlow.id));

        ApplicationContainer appCon = clientHelper.Install(n.Get(src));  // SRC，开始调度流
        appCon.Start(Seconds(Time(0)));
        appCon.Stop(Seconds(100.0));            // 新增
    }
    else {   // 处理 Dep
        std::cerr << Simulator::Now() << " dealing dep " << currentFlow.id << 
            ": node " << currentFlow.src << " (" << src << ") -> node " << currentFlow.dst << " (" << dst << ")" << std::endl;
        currentFlow.theoreticalFinishTime = currentFlow.theoreticalStartTime;
        Simulator::Schedule(MicroSeconds(currentFlow.lat), &RelationalFlowEnd, currentFlow.id);

        // Simulator::Schedule(MicroSeconds(0), &RelationalFlowEnd, currentFlow.id);   // 时延已经在 ScheduleFlowRelational() 加过了
    }

    // this is only for test. the really scheduling should be done @qp_finish()
    // Simulator::Schedule(NanoSeconds(currentFlow.size * 8), &RelationalFlowEnd, flowid);
}

void RelationalFlowEnd(uint32_t flowid) {
    Flow& currentFlow = flowMap[flowid];
    // 标记该流已发送
    currentFlow.sent = true;
    currentFlow.TxFinishTime = Simulator::Now().GetTimeStep();
    currentFlow.TxTime = currentFlow.TxFinishTime - currentFlow.TxStartTime;
    string ftype1 = (currentFlow.pg == -1)? " Dep " : " Flow ";
    std::cerr << Simulator::Now() << ftype1 << currentFlow.id << " Finish. ";
    if (currentFlow.pg == 3) {
        currentFlow.slowDown = double(currentFlow.TxTime) / double(currentFlow.baseTxTime);
        std::cerr << "note=\"" << currentFlow.note << "\", IdealTxTime=" << currentFlow.baseTxTime / 1000 
        << "us, RealTxTime=" << currentFlow.TxTime / 1000.0 << "us, SlowDown=" << currentFlow.slowDown;
        maxSlowDown = std::max(maxSlowDown, currentFlow.slowDown);
    }
    std::cout << "\n";

    // 处理 invoke_flow 中的流，将其依赖关系减1
    for (uint32_t invokedFlowId : currentFlow.invokeFlows) {
        dependencies[invokedFlowId]--;
        Flow& newFlow = flowMap[invokedFlowId];
        newFlow.theoreticalStartTime = std::max(currentFlow.theoreticalFinishTime, newFlow.theoreticalStartTime);
        if (dependencies[invokedFlowId] == 0) {
            readyQueue.push(invokedFlowId);
            string ftype2 = (newFlow.pg == -1)? " Dep " : " Flow ";
            std::cerr << Simulator::Now() << ftype2 << invokedFlowId << " becomes Ready.\n";
        }
    }
    ScheduleFlowRelational();
}

void ScheduleFlowRelational() {
    while (!readyQueue.empty()) {
        uint32_t currentFlowId = readyQueue.front();
        readyQueue.pop();
        Flow& currentFlow = flowMap[currentFlowId];
        // Simulator::Schedule(MicroSeconds(currentFlow.lat), &RelationalFlowStart, currentFlowId);
        Simulator::Schedule(MicroSeconds(0), &RelationalFlowStart, currentFlowId);
        // RelationalFlowStart(currentFlowId);
    }
}

/**
 * Read flow input from file "flowf"
 */
void ReadFlowInput() {
    std::cout << "read   flow    input" << std::endl;
    if (flow_input.idx < flow_num) {
        flowf >> flow_input.src >> flow_input.dst >> flow_input.pg >> flow_input.maxPacketCount >>
            flow_input.start_time;
            // std::cout << "node " << flow_input.src << ": " << n.Get(flow_input.src)->GetNodeType() << std::endl <<
            //    "node " << flow_input.dst << ": " << n.Get(flow_input.dst)->GetNodeType() << std::endl;
        assert(n.Get(flow_input.src)->GetNodeType() == 0 &&
               n.Get(flow_input.dst)->GetNodeType() == 0);
    } else {
        std::cout << "*** input flow is over the prefixed number -- flow number : " << flow_num
                  << std::endl;
        std::cout << "*** flow_input.idx : " << flow_input.idx << std::endl;
        std::cout << "*** THIS IS THE LAST FLOW TO SEND :) \n" << std::endl;
    }
}

/**
 * Scheduling flows given in /config/L_XX....txt file
 */
void ScheduleFlowInputs(FILE *infile) {
    NS_LOG_DEBUG("ScheduleFlowInputs at " << Simulator::Now());
    while (flow_input.idx < flow_num && Seconds(flow_input.start_time) == Simulator::Now()) {
        uint32_t pg, src, dst, sport, dport, maxPacketCount, target_len;
        pg = flow_input.pg;
        src = flow_input.src;
        dst = flow_input.dst;

        // src port
        sport = portNumber[src];  // get a new port number
        portNumber[src] = portNumber[src] + 1;

        // dst port
        dport = dportNumber[dst];
        dportNumber[dst] = dportNumber[dst] + 1;

        target_len = flow_input.maxPacketCount;  // this is actually not packet-count, but bytes
        if (target_len == 0) {
            target_len = 1;
        }
        assert(n.Get(src)->GetNodeType() == 0 && n.Get(dst)->GetNodeType() == 0);

        /**
         * Turn on if you want to record all input streams into output file for logging.
         * But, the input stream can be found in config. We do not recommend to do this
         * as it consumes storage resource, redundantly.
         */
        if (0) {  // logging input streams to "XXXX_out_in.txt"
            /************************
             * record flow's 4-tuple
             ************************/
            fprintf(infile, "%u %u %u %u %u %lu\n", src, dst, sport, dport, target_len,
                    (uint64_t)(flow_input.start_time * (uint64_t)1000000000));
            fflush(infile);

            /***********    FCT Tracking    **************/
            UdpServerHelper server0(dport);
            server0.SetAttribute("FlowSize", UintegerValue(target_len));    // 用这个传进去做流完成检查
            server0.SetAttribute("irn", BooleanValue(enable_irn));
            server0.SetAttribute("StatHostSrc", UintegerValue(src));
            server0.SetAttribute("StatHostDst", UintegerValue(dst));
            server0.SetAttribute("StatRxLen", UintegerValue(target_len));
            server0.SetAttribute("StatFlowID", UintegerValue(flow_input.idx));
            server0.SetAttribute("Port", UintegerValue(dport));

            ApplicationContainer apps0s = server0.Install(n.Get(dst));  // DST，接受测做传输完成检查，使用的是UdpClient的HandleRead()函数
            apps0s.Start(Seconds(Time(0)));
            apps0s.Stop(Seconds(100.0));
        }  // end of logging input streams

        if (pairRtt.find(n.Get(src)) == pairRtt.end() ||
            pairRtt[n.Get(src)].find(n.Get(dst)) == pairRtt[n.Get(src)].end()) {
            std::cerr << "pairRtt src: " << src << " -> dst: " << dst
                      << " ==> cannot be found from database" << std::endl;
            assert(false);
        }

        // 没变 
        RdmaClientHelper clientHelper(
            pg, serverAddress[src], serverAddress[dst], sport, dport, target_len,
            has_win ? (global_t == 1 ? maxBdp : pairBdp[n.Get(src)][n.Get(dst)]) : 0,
            global_t == 1 ? maxRtt : pairRtt[n.Get(src)][n.Get(dst)]);
            
        clientHelper.SetAttribute("StatFlowID", IntegerValue(flow_input.idx));

        ApplicationContainer appCon = clientHelper.Install(n.Get(src));  // SRC，开始调度流
        appCon.Start(Seconds(Time(0)));
        appCon.Stop(Seconds(100.0));            // 新增

        // get the next flow input
        std::cout << "get the next flow input" << std::endl;
        flow_input.idx++;
        ReadFlowInput();
    }

    // schedule the next time to run this function
    if (flow_input.idx < flow_num) {
        Simulator::Schedule(Seconds(flow_input.start_time) - Simulator::Now(), &ScheduleFlowInputs,
                            infile);
    } else {  // no more flows, close the file
        flowf.close();
    }
}

/**
 * @brief CNP frequency monitoring (timestamp nodeId ECN OoO Total)
 */
void cnp_freq_monitoring(FILE *fout, Ptr<RdmaHw> rdmahw) {
    if (rdmahw->cnp_total > 0) {
        // flush
        fprintf(fout, "%lu %u %u %u %u\n", Simulator::Now().GetNanoSeconds(),
                rdmahw->m_node->GetId(), rdmahw->cnp_by_ecn, rdmahw->cnp_by_ooo, rdmahw->cnp_total);
        fflush(fout);

        // initialize
        rdmahw->cnp_by_ecn = 0;
        rdmahw->cnp_by_ooo = 0;
        rdmahw->cnp_total = 0;
    }

    // recursive callback，每100us记录一次
    Simulator::Schedule(NanoSeconds(cnp_monitor_bucket), &cnp_freq_monitoring, fout, rdmahw);
}

/**
 * @brief TOR Switch monitoring
 * - VOQ number and uplink throughput at switches
 * - the number of active connections at RNICS
 */
void periodic_monitoring(FILE *fout_voq, FILE *fout_voq_detail, FILE *fout_uplink, FILE *fout_conn,
                         uint32_t *lb_mode) {
    uint32_t lb_mode_val = *lb_mode;
    uint64_t now = Simulator::Now().GetNanoSeconds();

    // std::cout << "periodic_monitoring()" << std::endl;
    for (const auto &tor2If : torId2UplinkIf) {  // for each TOR switches
        Ptr<Node> node = n.Get(tor2If.first);    // tor id
        auto swNode = DynamicCast<SwitchNode>(node);
        assert(swNode->m_isToR == true);  // sanity check

        if (lb_mode_val == 9) {  // Conweave
            // monitor VOQ number per switch <time, ToRId, #VOQ, #Pkts>
            uint32_t nVOQ = swNode->m_mmu->m_conweaveRouting.GetNumVOQ();
            uint32_t nVolumeVOQ = swNode->m_mmu->m_conweaveRouting.GetVolumeVOQ();
            fprintf(fout_voq, "%lu,%u,%u,%u\n", now, tor2If.first, nVOQ, nVolumeVOQ);

            // monitor VOQ per destination IP <time, dstip, #VOQ, #Pkts>
            std::unordered_map<uint32_t, std::pair<uint32_t, uint32_t>> dip_to_nvoq_npkt;
            for (auto voq : swNode->m_mmu->m_conweaveRouting.GetVOQMap()) {
                auto &nvoq_npkt = dip_to_nvoq_npkt[voq.second.getDIP()];
                nvoq_npkt.first += 1;
                nvoq_npkt.second += voq.second.getQueueSize();
            }
            for (auto x : dip_to_nvoq_npkt) {
                fprintf(fout_voq_detail, "%lu,%u,%u,%u\n", now, x.first, x.second.first,
                        x.second.second);
            }
        }

        // common: monitor TOR's uplink to measure load balancing performance
        for (const auto &iface : tor2If.second) {
            // monitor uplink txBytes <time, ToRId, OutDev, Bytes>
            uint64_t uplink_txbyte = swNode->GetTxBytesOutDev(iface);
            fprintf(fout_uplink, "%lu,%u,%u,%lu\n", now, tor2If.first, iface, uplink_txbyte);
        }
    }

    // common: get number of concurrent connections at each server
    for (uint32_t i = 0; i < Settings::node_num; i++) {
        if (n.Get(i)->GetNodeType() == 0) {  // is server
            Ptr<Node> server = n.Get(i);
            Ptr<RdmaDriver> rdmaDriver = server->GetObject<RdmaDriver>();
            Ptr<RdmaHw> rdmaHw = rdmaDriver->m_rdma;
            // monitor total/active QP number <time, serverId, #ExistingQP, #ActiveQP>
            uint64_t nQP = rdmaHw->m_qpMap.size();
            uint64_t nActiveQP = 0;
            for (auto qp : rdmaHw->m_qpMap) {
                if (qp.second->GetBytesLeft() > 0) {  // conns with bytes left。
                    nActiveQP++;                       // TODO: 只有发送端有这个统计，可能需要在接收端一起加上
                }
            }
            fprintf(fout_conn, "%lu,%u,%lu,%lu\n", now, i, nQP, nActiveQP);
        }
    }
    // cout << "flowgen_stop_time: " << flowgen_stop_time << std::endl;
    if (Simulator::Now() < Seconds(flowgen_stop_time + 0.05)) {
        // recursive callback
        Simulator::Schedule(NanoSeconds(switch_mon_interval), &periodic_monitoring, fout_voq,
                            fout_voq_detail, fout_uplink, fout_conn, lb_mode);  // every 10us
    }
    return;
}

/**
 * @brief Conga timeout number recording
 */
void conga_history_print() {
    std::cout << "\n------------CONGA History---------------" << std::endl;
    std::cout << "Number of flowlet's timeout:" << CongaRouting::nFlowletTimeout
              << "Conga's timeout: " << conga_flowletTimeout << std::endl;
}

/**
 * @brief Letflow timeout number recording
 */
void letflow_history_print() {
    std::cout << "\n------------Letflow History---------------" << std::endl;
    std::cout << "Number of flowlet's timeout:" << LetflowRouting::nFlowletTimeout
              << "\nLetflow's timeout: " << letflow_flowletTimeout << std::endl;
}

/**
 * @brief Conweave rerouting/VOQ number recording
 */
void conweave_history_print() {
    // Conweave params
    std::cout << "\n------ConWeave parameters-----" << std::endl;
    std::cout << "Param - extraReplyDeadline:" << conweave_extraReplyDeadline << std::endl;
    std::cout << "Param - extraVOQFlushTime:" << conweave_extraVOQFlushTime << std::endl;
    std::cout << "Param - txExpiryTime:" << conweave_txExpiryTime << std::endl;
    std::cout << "Param - defaultVOQWaitingTime:" << conweave_defaultVOQWaitingTime << std::endl;
    std::cout << "Param - pathPauseTime:" << conweave_pathPauseTime << std::endl;
    std::cout << "Param - pathAwareRerouting:" << conweave_pathAwareRerouting << std::endl;

    std::cout << "\n------------ConWeave History---------------" << std::endl;
    std::cout << "Number of INIT's Reply sent (RTT_REPLY):" << ConWeaveRouting::m_nReplyInitSent
              << "\nNumber of Timely RTT_REPLY (INIT's Reply):" << ConWeaveRouting::m_nTimelyInitReplied
              << "\nNumber of TAIL's Reply Sent (CLEAR):" << ConWeaveRouting::m_nReplyTailSent
              << "\nNumber of Timely CLEAR (TAIL's Reply):" << ConWeaveRouting::m_nTimelyTailReplied
              << "\nNumber of NOTIFY Sent:" << ConWeaveRouting::m_nNotifySent
              << "\nNumber of Rerouting:" << ConWeaveRouting::m_nReRoute
              << "\nNumber of OoO enqueued pkts:" << ConWeaveRouting::m_nOutOfOrderPkts
              << "\nNumber of VOQ Flush Total:" << ConWeaveRouting::m_nFlushVOQTotal
              << "\nNumber of VOQ Flush From History:" << ConWeaveRouting::m_historyVOQSize.size()
              << "\nNumber of VOQ Flush by TAIL:" << ConWeaveRouting::m_nFlushVOQByTail
              << std::endl;

    std::cout << "--------------------------" << std::endl;

    /** VOQ: Sanity check*/
    for (size_t ToRId = 0; ToRId < Settings::node_num; ToRId++) {
        Ptr<Node> node = n.Get(ToRId);
        if (node->GetNodeType() == 1) {  // switches
            auto swNode = DynamicCast<SwitchNode>(n.Get(ToRId));
            if (swNode->m_isToR) {  // TOR switch
                uint32_t num_remained_voq = swNode->m_mmu->m_conweaveRouting.GetNumVOQ();
                if (num_remained_voq > 0) {
                    printf("*******************************\n");
                    printf("*** WARNING - Tor Sw (%lu) - VOQ (num=%u) is not flushed yet!! ***\n",
                           ToRId, num_remained_voq);
                    printf(
                        " -- Probably the history print is too early so simulation might not be "
                        "finished?");
                    printf("********************************\n");
                }
            }
        }
    }

    /** Get ConWeave Flush Time Estimation Error */
    if (0) {
        // sanity check - extraVOQFlushTime must be large enough to get accuracy
        assert(conweave_extraVOQFlushTime >= MicroSeconds(128) && "PARAMETER ERROR!!");

        std::cout << "\n--------------------------" << std::endl;
        std::cout << "Extracting ConWeave Estimation Error Data..." << std::endl;
        est_error_output = fopen(est_error_output_file.c_str(), "w");
        for (auto x : ConWeaveVOQ::m_flushEstErrorhistory) {
            fprintf(est_error_output, "%d\n", x);
        }
        ConWeaveVOQ::m_flushEstErrorhistory.clear();
        std::cout << "---------D O N E---------" << std::endl;
    }
}

/**
 * @brief When one RDMA is finished, so does (1) QP, (2) RxQP, (3) write it on file fct.txt.
 */
void qp_finish(FILE *fout, Ptr<RdmaQueuePair> q) {
    // 调度 relational 的流
    uint32_t flow_id = q->m_flow_id;
    RelationalFlowEnd(flow_id);

    uint32_t sid = Settings::ip_to_node_id(q->sip), did = Settings::ip_to_node_id(q->dip);
    uint64_t base_rtt = pairRtt[n.Get(sid)][n.Get(did)];
    uint64_t b = pairBw[n.Get(sid)][n.Get(did)];
    uint32_t total_bytes = 
        q->m_size + ((q->m_size - 1) / packet_payload_size + 1) *
                        (CustomHeader::GetStaticWholeHeaderSize() -
                         IntHeader::GetStaticSize());  // translate to the minimum bytes required
                                                       // (with header but no INT)
    uint64_t standalone_fct = base_rtt + total_bytes * 8000000000lu / b;    // ns

    // XXX: remove rxQP from the receiver
    Ptr<Node> dstNode = n.Get(did);
    Ptr<RdmaDriver> rdma = dstNode->GetObject<RdmaDriver>();
    rdma->m_rdma->DeleteRxQp(q->sip.Get(), q->sport, q->dport, q->m_pg);

    // fprintf(fout, "%lu QP complete\n", Simulator::Now().GetTimeStep());
    fprintf(fout, "%u %u %u %u %lu %lu %lu %lu: %lu %u %lu\n", Settings::ip_to_node_id(q->sip),
            Settings::ip_to_node_id(q->dip), q->sport, q->dport, q->m_size,
            q->startTime.GetTimeStep(), (Simulator::Now() - q->startTime).GetTimeStep(),
            standalone_fct, base_rtt, total_bytes, b);

    oss.str("");oss.clear();
    // for debugging
    oss << Settings::ip_to_node_id(q->sip) << " "
        << Settings::ip_to_node_id(q->dip) << " "
        << q->sport << " "
        << q->dport << " "
        << q->m_size << " "
        << q->startTime.GetTimeStep() << " "
        << (Simulator::Now() - q->startTime).GetTimeStep() << " "
        << standalone_fct << "\n";

    NS_LOG_DEBUG(oss.str());

    // NS_LOG_DEBUG("%u %u %u %u %lu %lu %lu %lu\n" %
    //              (Settings::ip_to_node_id(q->sip), Settings::ip_to_node_id(q->dip), q->sport,
    //               q->dport, q->m_size, q->startTime.GetTimeStep(),
    //               (Simulator::Now() - q->startTime).GetTimeStep(), standalone_fct));
    Settings::cnt_finished_flows++;
    fflush(fout);
}

/**
 * @brief TODO: when RxQP finish. Not used for now. 
 */
void rx_qp_finish(FILE *fout, Ptr<RdmaQueuePair> q) {
    uint32_t sid = Settings::ip_to_node_id(q->sip), did = Settings::ip_to_node_id(q->dip);
    uint64_t base_rtt = pairRtt[n.Get(sid)][n.Get(did)];
    uint64_t b = pairBw[n.Get(sid)][n.Get(did)];
    uint32_t total_bytes = 
        q->m_size + ((q->m_size - 1) / packet_payload_size + 1) *
                        (CustomHeader::GetStaticWholeHeaderSize() -
                         IntHeader::GetStaticSize());  // translate to the minimum bytes required
                                                       // (with header but no INT)
    uint64_t standalone_fct = base_rtt + total_bytes * 8000000000lu / b;    // ns

    // XXX: remove rxQP from the receiver
    Ptr<Node> dstNode = n.Get(did);
    Ptr<RdmaDriver> rdma = dstNode->GetObject<RdmaDriver>();
    rdma->m_rdma->DeleteRxQp(q->sip.Get(), q->sport, q->dport, q->m_pg);

    // fprintf(fout, "%lu QP complete\n", Simulator::Now().GetTimeStep());
    fprintf(fout, "%u %u %u %u %lu %lu %lu %lu: %lu %u %lu\n", Settings::ip_to_node_id(q->sip),
            Settings::ip_to_node_id(q->dip), q->sport, q->dport, q->m_size,
            q->startTime.GetTimeStep(), (Simulator::Now() - q->startTime).GetTimeStep(),
            standalone_fct, base_rtt, total_bytes, b);

    oss.str("");oss.clear();
    // for debugging
    oss << Settings::ip_to_node_id(q->sip) << " "
        << Settings::ip_to_node_id(q->dip) << " "
        << q->sport << " "
        << q->dport << " "
        << q->m_size << " "
        << q->startTime.GetTimeStep() << " "
        << (Simulator::Now() - q->startTime).GetTimeStep() << " "
        << standalone_fct << "\n";

    NS_LOG_DEBUG(oss.str());

    // NS_LOG_DEBUG("%u %u %u %u %lu %lu %lu %lu\n" %
    //              (Settings::ip_to_node_id(q->sip), Settings::ip_to_node_id(q->dip), q->sport,
    //               q->dport, q->m_size, q->startTime.GetTimeStep(),
    //               (Simulator::Now() - q->startTime).GetTimeStep(), standalone_fct));
    Settings::cnt_finished_flows++;
    fflush(fout);
}

/**
 * @brief PFC event logging
 */
void get_pfc(FILE *fout, Ptr<QbbNetDevice> dev, uint32_t type) {
    // time, nodeID, nodeType, Interface's Idx, 0:resume, 1:pause
    fprintf(fout, "%lu %u %u %u %u\n", Simulator::Now().GetTimeStep(), dev->GetNode()->GetId(),
            dev->GetNode()->GetNodeType(), dev->GetIfIndex(), type);
}

void record_send(FILE *fout, Ptr<QbbNetDevice> dev, uint32_t type) {
    // time, nodeID, nodeType, Interface's Idx, 0:resume, 1:pause
    fprintf(fout, "%lu %u %u %u %u\n", Simulator::Now().GetTimeStep(), dev->GetNode()->GetId(),
            dev->GetNode()->GetNodeType(), dev->GetIfIndex(), type);
}


inline const char* getPriorityString(int pkt_type) {
    switch (pkt_type) {
        case 0:
            return "UDP";
        case 1:
            return "CNP";
        case 2:
            return "NACK";
        case 3:
            return "ACK";
        case 4:
            return "PFC";
        default:
            return "Undefined";
    }
}
/**
 * @brief record a send/recv event. 0: recv, 1: send; size: pkt_size
 */
void snd_rcv_record(FILE *fout, Ptr<QbbNetDevice> dev, 
                uint32_t rcv_snd_type, uint32_t pkt_type, uint32_t pkt_size, int flowid=-1, int seq=-1) {
    // time, nodeID, nodeType, Interface's Idx, 0:resume, 1:pause
    fprintf(fout, "%lu: %s %u NIC %u %s a %s pkt. size=%u flowid=%d seq=%d\n", 
            Simulator::Now().GetTimeStep(), 
            (dev->GetNode()->GetNodeType() == 0) ? " host  " : "switch ", 
            dev->GetNode()->GetId(),
            dev->GetIfIndex(), 
            // 下面是动态填充的
            (rcv_snd_type == 0) ? " recv " : " send ",
            getPriorityString(pkt_type),
            pkt_size,
            flowid,
            seq);
}

void switch_spray_event_record(FILE *fout, Ptr<SwitchNode> sw, uint32_t port_picked, const std::vector<int> &nexthops, const char *explain_str){
    uint32_t nsize = nexthops.size();
    if (nsize < 1) {
        std::cout << "Error encountered in switch_spray_event_record(): nexthops.size() < 1" << std::endl;
        return;
    }
    fprintf(fout, "%lu: switch %u do spray (%lu, %s) [", Simulator::Now().GetTimeStep(), sw->GetId(), nexthops.size(), explain_str);
    size_t i = 0;
    for (i = 0; i < nsize - 1; ++i) {
        fprintf(fout, "%d, ", nexthops[i]);
    }
    fprintf(fout, "%d] -> %u\n", nexthops[i], port_picked);
}
/*******************************************************************/
#if (false)

/**
 * @brief Qlen monitoring at switches (output: qlen.txt), I think "periodically"...
 *
 */
struct QlenDistribution {
    vector<uint32_t> cnt;  // cnt[i] is the number of times that the queue len is i KB
    void add(uint32_t qlen) {
        uint32_t kb = qlen / 1000;
        if (cnt.size() < kb + 1) cnt.resize(kb + 1);
        cnt[kb]++;
    }
};

map<uint32_t, map<uint32_t, QlenDistribution>> queue_result;
void monitor_buffer(FILE *qlen_output, NodeContainer *n) {
    /*******************************************************************/
    /************************** UNUSED NOW *****************************/
    /*******************************************************************/
    for (uint32_t i = 0; i < n->GetN(); i++) {
        if (n->Get(i)->GetNodeType() == 1) {  // is switch
            Ptr<SwitchNode> sw = DynamicCast<SwitchNode>(n->Get(i));
            if (queue_result.find(i) == queue_result.end()) queue_result[i];
            for (uint32_t j = 1; j < sw->GetNDevices(); j++) {
                uint32_t size = 0;
                for (uint32_t k = 0; k < SwitchMmu::qCnt; k++)
                    size += sw->m_mmu->egress_bytes[j][k];
                queue_result[i][j].add(size);
            }
        }
    }
    if (Simulator::Now().GetTimeStep() % qlen_dump_interval == 0) {
        fprintf(qlen_output, "time: %lu\n", Simulator::Now().GetTimeStep());
        for (auto &it0 : queue_result) {
            for (auto &it1 : it0.second) {
                fprintf(qlen_output, "%u %u", it0.first, it1.first);
                auto &dist = it1.second.cnt;
                for (uint32_t i = 0; i < dist.size(); i++) fprintf(qlen_output, " %u", dist[i]);
                fprintf(qlen_output, "\n");
            }
        }
        fflush(qlen_output);
    }
    if (Simulator::Now().GetTimeStep() < qlen_mon_end)
        Simulator::Schedule(NanoSeconds(qlen_mon_interval), &monitor_buffer, qlen_output, n);
}
#endif
/*******************************************************************/

/**
 * @brief Stop simulation in the middle (when almost all flows are done).
 * This function allows to finish simulation quickly when all messages are sent.
 */
void stop_simulation_middle() {
    // return;
    uint32_t target_flow_num = flow_num - 0;  // can be lower than flownum

    // std::cout << "target_flow_num: " << target_flow_num << std::endl;
    if (Settings::cnt_finished_flows >= target_flow_num) {
        std::cout << "\n*** Simulator is enforced to be finished, finished so far: "
                  << Settings::cnt_finished_flows << "/ total: " << target_flow_num
                  << ", Time:" << Simulator::Now() << std::endl;

        // schedule conga timeout monitor
        if (lb_mode == 3) {  // CONGA
            conga_history_print();
        }
        if (lb_mode == 6) {  // LETFLOW
            letflow_history_print();
        }
        if (lb_mode == 9) {  // CONWEAVE
            conweave_history_print();
        }
        Simulator::Stop(NanoSeconds(1));  // finish soon, stop this schedule (NECESSARY!)
        return;
    }

    Simulator::Schedule(MicroSeconds(100), &stop_simulation_middle);  // check every 100us
}

/**
 * @brief Calculate edge-to-edge delays, TX delays, and bandwidths
 * 为每一个server节点计算路由
 */
void CalculateRoute(Ptr<Node> host) {
    // queue for the BFS.
    vector<Ptr<Node>> q;
    // Distance from the host to each node.
    map<Ptr<Node>, int> dis;
    map<Ptr<Node>, uint64_t> delay;
    map<Ptr<Node>, uint64_t> txDelay;
    map<Ptr<Node>, uint64_t> bw;
    // init BFS.
    q.push_back(host);
    dis[host] = 0;
    delay[host] = 0;
    txDelay[host] = 0;
    bw[host] = 0xfffffffffffffffflu;
    
    
    // BFS.
    for (int i = 0; i < (int)q.size(); i++) {
        Ptr<Node> now = q[i];
        int d = dis[now];
        // std::map<ns3::Ptr<ns3::Node>, std::map< ns3::Ptr<ns3::Node>, std::vector<Interface>> > nbr2if
        // 从node1到下一跳node2的port是哪个
        for (auto it = nbr2if[now].begin(); it != nbr2if[now].end(); it++) {    // 对每一个邻居节点及其互联inf
            Ptr<Node> next = it->first;
            vector<Interface> nextInfs = it->second;  // 下一跳节点的vector
            // skip down link
            bool no_activate_connections = true;
            for (int k = 0; k < (int)nextInfs.size(); k++) {
                if (nextInfs[k].up)     // Interface up
                    no_activate_connections = false;
            }
            if (no_activate_connections) {
                continue;
            }
            // If 'next' have not been visited.
            if (dis.find(next) == dis.end()) {
                dis[next] = d + 1;
                delay[next] = delay[now] + nextInfs[0].delay;  // maybe nanoseconds?
                txDelay[next] = txDelay[now] + packet_payload_size * 1000000000lu * 8 / nextInfs[0].bw;  // maybe nanoseconds?
                bw[next] = std::min(bw[now], nextInfs[0].bw);
                // we only enqueue switch, because we do not want packets to go through host as
                // middle point
                if (next->GetNodeType() == 1) {
                    q.push_back(next);
                }
            }
            // if 'now' is on the shortest path from 'next' to 'host'.
            if (d + 1 == dis[next]) {
                nextHop[next][host].push_back(now);
            }
        }
    }
    for (auto it : delay) {
        pairDelay[it.first][host] = it.second;
    }
    for (auto it : txDelay) {
        pairTxDelay[it.first][host] = it.second;
    }
    for (auto it : bw) {
        pairBw[it.first][host] = it.second;
    }
}

void CalculateRoutes(NodeContainer &n) {
    for (int i = 0; i < (int)n.GetN(); i++) {
        Ptr<Node> node = n.Get(i);
        if (node->GetNodeType() == 0) {
            CalculateRoute(node);
        }
    }
}

/**
 * @brief Set the Routing Entries object
 */
void SetRoutingEntries() {
    // For each node.
    for (auto i = nextHop.begin(); i != nextHop.end(); i++) {
        Ptr<Node> node = i->first;  // 当前节点
        auto &table = i->second;    // {目标节点, vector(下一跳节点)}
        for (auto j = table.begin(); j != table.end(); j++) {
            // The destination node.
            Ptr<Node> dst = j->first;
            // The IP address of the dst.
            Ipv4Address dstAddr = dst->GetObject<Ipv4>()->GetAddress(1, 0).GetLocal();
            // The next hops towards the dst.
            vector<Ptr<Node>> nexts = j->second;  // 下一跳节点的vector
            for (int k = 0; k < (int)nexts.size(); k++) {
                Ptr<Node> next = nexts[k];
                vector<Interface> intfs = nbr2if[node][next];
                // printf("------- %d -------\n", (int)intfs.size());
                for (int k = 0; k < (int)intfs.size(); k++) {
                    uint32_t interface = intfs[k].idx;    // node到下一跳的inf
                    if (node->GetNodeType() == 1) {
                        fprintf(routing_table_output, "Adding rtTable entry into switch %u: dst = %u, Intf = %u\n",
                                    node->GetId(), dst->GetId(), interface);
                        DynamicCast<SwitchNode>(node)->AddTableEntry(dstAddr, interface);
                    }
                    else {
                        fprintf(routing_table_output, "Adding rtTable entry into host %u: dst = %u, Intf = %u\n",
                                    node->GetId(), dst->GetId(), interface);
                        node->GetObject<RdmaDriver>()->m_rdma->AddTableEntry(dstAddr, interface);
                    }
                }
            }
        }
    }
}
/**
 * @brief take down the link between a and b, and redo the routing
 */
void TakeDownLink(NodeContainer n, Ptr<Node> a, Ptr<Node> b, uint32_t a_interface) {
    if (!nbr2if[a][b][a_interface].up)
        return;
    uint32_t b_interface = nbr2if[a][b][a_interface].peerIdx;     // 正和反不一样！
    // take down link between a and b
    nbr2if[a][b][a_interface].up = nbr2if[b][a][b_interface].up = false;
    nextHop.clear();
    CalculateRoutes(n);
    // clear routing tables
    for (uint32_t i = 0; i < n.GetN(); i++) {
        if (n.Get(i)->GetNodeType() == 1)
            DynamicCast<SwitchNode>(n.Get(i))->ClearTable();
        else
            n.Get(i)->GetObject<RdmaDriver>()->m_rdma->ClearTable();
    }
    DynamicCast<QbbNetDevice>(a->GetDevice(nbr2if[a][b][a_interface].idx))->TakeDown();
    DynamicCast<QbbNetDevice>(b->GetDevice(nbr2if[b][a][b_interface].idx))->TakeDown();
    // reset routing table
    SetRoutingEntries();

    // redistribute qp on each host
    for (uint32_t i = 0; i < n.GetN(); i++) {
        if (n.Get(i)->GetNodeType() == 0)
            n.Get(i)->GetObject<RdmaDriver>()->m_rdma->RedistributeQp();
    }
}

uint64_t get_nic_rate(NodeContainer &n) {
    uint64_t avg_nic_rate;
    uint64_t n_servers = 0;
    for (uint32_t i = 0; i < n.GetN(); i++) {
        if (n.Get(i)->GetNodeType() == 0) {
            avg_nic_rate +=
                DynamicCast<QbbNetDevice>(n.Get(i)->GetDevice(1))->GetDataRate().GetBitRate();
            n_servers += 1;
        }
    }
    return avg_nic_rate / n_servers;
}

int extractInteger(const std::string& str) {
    std::stringstream ss(str);
    int result;
    ss >> result;
    return result;
}

float extractFloat(const std::string& str) {
    std::stringstream ss(str);
    float result;
    ss >> result;
    return result;
}

/************************************************************************/
//                                                                      //
//                                M A I N                               //
//                                                                      //
/************************************************************************/

int main(int argc, char *argv[]) {
    uint32_t *workload_cdf = nullptr;
    clock_t begint, endt;
    begint = clock();
#ifndef PGO_TRAINING
    if (argc > 1)
#else
    if (true)
#endif
    {
        // Read the configuration file
        std::ifstream conf;
#ifndef PGO_TRAINING
        conf.open(argv[1]);
#else
        conf.open(PATH_TO_PGO_CONFIG);
#endif
        while (!conf.eof()) {
            std::string key;
            conf >> key;
            if (key.compare("FLOW_INPUT_FILE") == 0) {
                std::string v;
                conf >> v;
                flow_input_file = v;
                std::cerr << "FLOW_INPUT_FILE\t\t\t" << flow_input_file << "\n";
            }
            if (key.compare("FLOW_STATISTICS_OUTPUT_FILE") == 0) {
                std::string v;
                conf >> v;
                flow_statistics_output_file = v;
                std::cerr << "FLOW_STATISTICS_OUTPUT_FILE\t\t\t" << flow_statistics_output_file << "\n";
            }
            if (key.compare("FLOW_RELATIONAL") == 0) {
                uint32_t v;
                conf >> v;
                is_flow_relational = v;
                std::cerr << "FLOW_RELATIONAL\t\t\t" << is_flow_relational << "\n";
            }
            if (key.compare("NODE_MAPPING") == 0) {
                std::string v;
                conf >> v;
                node_mapping = v;
                std::cerr << "NODE_MAPPING\t\t\t" << node_mapping << "\n";
            }
            if (key.compare("SIM_NODE_MAPPING_FILE") == 0) {
                std::string v;
                conf >> v;
                simnode_mapping_file = v;
                std::cerr << "SIM_NODE_MAPPING_FILE\t\t\t" << simnode_mapping_file << "\n";
            }
             else if (key.compare("CNP_OUTPUT_FILE") == 0) {
                std::string v;
                conf >> v;
                cnp_output_file = v;
                std::cerr << "CNP_OUTPUT_FILE\t\t\t" << cnp_output_file << "\n";
            } else if (key.compare("IS_SPRAY") == 0) {
                uint32_t v;
                conf >> v;
                is_spray = v;
                std::cerr << "IS_SPRAY\t\t\t" << is_spray << "\n";
            } else if (key.compare("EST_ERROR_MON_FILE") == 0) {
                std::string v;
                conf >> v;
                est_error_output_file = v;
                std::cerr << "EST_ERROR_MON_FILE\t\t\t" << est_error_output_file << "\n";
            } else if (key.compare("LB_MODE") == 0) {
                uint32_t v;
                conf >> v;
                lb_mode = v;
                std::cerr << "LB_MODE\t\t\t" << lb_mode << "\n";
            } else if (key.compare("SW_MONITORING_INTERVAL") == 0) {
                uint32_t v;
                conf >> v;
                switch_mon_interval = v;
                std::cerr << "SW_MONITORING_INTERVAL\t\t\t" << switch_mon_interval << "\n";
            } else if (key.compare("CONWEAVE_TX_EXPIRY_TIME") == 0) {
                uint32_t v;
                conf >> v;
                conweave_txExpiryTime = Time(MicroSeconds(v));
                std::cerr << "CONWEAVE_TX_EXPIRY_TIME\t\t\t" << conweave_txExpiryTime << "\n";
            } else if (key.compare("CONWEAVE_REPLY_TIMEOUT_EXTRA") == 0) {
                uint32_t v;
                conf >> v;
                conweave_extraReplyDeadline = Time(MicroSeconds(v));
                std::cerr << "CONWEAVE_REPLY_TIMEOUT_EXTRA\t\t\t" << conweave_extraReplyDeadline
                          << "\n";
            } else if (key.compare("CONWEAVE_EXTRA_VOQ_FLUSH_TIME") == 0) {
                uint32_t v;
                conf >> v;
                conweave_extraVOQFlushTime = Time(MicroSeconds(v));
                std::cerr << "CONWEAVE_EXTRA_VOQ_FLUSH_TIME\t\t\t" << conweave_extraVOQFlushTime
                          << "\n";
            } else if (key.compare("CONWEAVE_PATH_PAUSE_TIME") == 0) {
                uint32_t v;
                conf >> v;
                conweave_pathPauseTime = Time(MicroSeconds(v));
                std::cerr << "CONWEAVE_PATH_PAUSE_TIME\t\t\t" << conweave_pathPauseTime << "\n";
            } else if (key.compare("CONWEAVE_DEFAULT_VOQ_WAITING_TIME") == 0) {
                uint32_t v;
                conf >> v;
                conweave_defaultVOQWaitingTime = Time(MicroSeconds(v));
                std::cerr << "CONWEAVE_DEFAULT_VOQ_WAITING_TIME\t\t\t"
                          << conweave_defaultVOQWaitingTime << "\n";
            } else if (key.compare("ENABLE_PFC") == 0) {
                uint32_t v;
                conf >> v;
                enable_pfc = v;
                if (enable_pfc)
                    std::cerr << "ENABLE_PFC\t\t\t"
                              << "Yes"
                              << "\n";
                else
                    std::cerr << "ENABLE_PFC\t\t\t"
                              << "No"
                              << "\n";
            } else if (key.compare("ENABLE_QCN") == 0) {
                uint32_t v;
                conf >> v;
                enable_qcn = v;
                if (enable_qcn)
                    std::cerr << "ENABLE_QCN\t\t\t"
                              << "Yes"
                              << "\n";
                else
                    std::cerr << "ENABLE_QCN\t\t\t"
                              << "No"
                              << "\n";
            } else if (key.compare("USE_DYNAMIC_PFC_THRESHOLD") == 0) {
                uint32_t v;
                conf >> v;
                use_dynamic_pfc_threshold = v;
                if (use_dynamic_pfc_threshold)
                    std::cerr << "USE_DYNAMIC_PFC_THRESHOLD\t"
                              << "Yes"
                              << "\n";
                else
                    std::cerr << "USE_DYNAMIC_PFC_THRESHOLD\t"
                              << "No"
                              << "\n";
            } else if (key.compare("CLAMP_TARGET_RATE") == 0) {
                uint32_t v;
                conf >> v;
                clamp_target_rate = v;
                if (clamp_target_rate)
                    std::cerr << "CLAMP_TARGET_RATE\t\t"
                              << "Yes"
                              << "\n";
                else
                    std::cerr << "CLAMP_TARGET_RATE\t\t"
                              << "No"
                              << "\n";
            } else if (key.compare("PAUSE_TIME") == 0) {
                double v;
                conf >> v;
                pause_time = v;
                std::cerr << "PAUSE_TIME\t\t\t" << pause_time << "\n";
            } else if (key.compare("DATA_RATE") == 0) {
                std::string v;
                conf >> v;
                data_rate = v;
                std::cerr << "DATA_RATE\t\t\t" << data_rate << "\n";
            } else if (key.compare("LINK_DELAY") == 0) {
                std::string v;
                conf >> v;
                link_delay = v;
                std::cerr << "LINK_DELAY\t\t\t" << link_delay << "\n";
            } else if (key.compare("PACKET_PAYLOAD_SIZE") == 0) {
                uint32_t v;
                conf >> v;
                packet_payload_size = v;
                std::cerr << "PACKET_PAYLOAD_SIZE\t\t" << packet_payload_size << "\n";
            } else if (key.compare("L2_CHUNK_SIZE") == 0) {
                uint32_t v;
                conf >> v;
                l2_chunk_size = v;
                std::cerr << "L2_CHUNK_SIZE\t\t\t" << l2_chunk_size << "\n";
            } else if (key.compare("L2_ACK_INTERVAL") == 0) {
                uint32_t v;
                conf >> v;
                l2_ack_interval = v;
                std::cerr << "L2_ACK_INTERVAL\t\t\t" << l2_ack_interval << "\n";
            } else if (key.compare("L2_BACK_TO_ZERO") == 0) {
                uint32_t v;
                conf >> v;
                l2_back_to_zero = v;
                if (l2_back_to_zero)
                    std::cerr << "L2_BACK_TO_ZERO\t\t\t"
                              << "Yes"
                              << "\n";
                else
                    std::cerr << "L2_BACK_TO_ZERO\t\t\t"
                              << "No"
                              << "\n";
            } else if (key.compare("TOPOLOGY_FILE") == 0) {
                std::string v;
                conf >> v;
                topology_file = v;
                std::cerr << "TOPOLOGY_FILE\t\t\t" << topology_file << "\n";
            } else if (key.compare("FLOW_FILE") == 0) {     // 输入的流文件
                std::string v;
                conf >> v;
                flow_file = v;
                std::cerr << "FLOW_FILE\t\t\t" << flow_file << "\n";
            } else if (key.compare("FLOWGEN_START_TIME") == 0) {
                double v;
                conf >> v;
                flowgen_start_time = v;
                qlen_mon_start = v;
                qlen_mon_end = v;
                cnp_mon_start = v;
                irn_mon_start = v;
                std::cerr << "FLOWGEN_START_TIME\t\t" << flowgen_start_time << "\n";
            } else if (key.compare("FLOWGEN_STOP_TIME") == 0) {
                double v;
                conf >> v;
                flowgen_stop_time = v;
                std::cerr << "FLOWGEN_STOP_TIME\t\t" << flowgen_stop_time << "\n";
            } else if (key.compare("ALPHA_RESUME_INTERVAL") == 0) {
                double v;
                conf >> v;
                alpha_resume_interval = v;
                std::cerr << "ALPHA_RESUME_INTERVAL\t\t" << alpha_resume_interval << "\n";
            } else if (key.compare("RP_TIMER") == 0) {
                double v;
                conf >> v;
                rp_timer = v;
                std::cerr << "RP_TIMER\t\t\t" << rp_timer << "\n";
            } else if (key.compare("EWMA_GAIN") == 0) {
                double v;
                conf >> v;
                ewma_gain = v;
                std::cerr << "EWMA_GAIN\t\t\t" << ewma_gain << "\n";
            } else if (key.compare("FAST_RECOVERY_TIMES") == 0) {
                uint32_t v;
                conf >> v;
                fast_recovery_times = v;
                std::cerr << "FAST_RECOVERY_TIMES\t\t" << fast_recovery_times << "\n";
            } else if (key.compare("RATE_AI") == 0) {
                std::string v;
                conf >> v;
                rate_ai = v;
                std::cerr << "RATE_AI\t\t\t\t" << rate_ai << "\n";
            } else if (key.compare("RATE_HAI") == 0) {
                std::string v;
                conf >> v;
                rate_hai = v;
                std::cerr << "RATE_HAI\t\t\t" << rate_hai << "\n";
            } else if (key.compare("ERROR_RATE_PER_LINK") == 0) {
                double v;
                conf >> v;
                error_rate_per_link = v;
                std::cerr << "ERROR_RATE_PER_LINK\t\t" << error_rate_per_link << "\n";
            } else if (key.compare("CC_MODE") == 0) {
                conf >> cc_mode;
                std::cerr << "CC_MODE\t\t" << cc_mode << '\n';
            } else if (key.compare("RATE_DECREASE_INTERVAL") == 0) {
                double v;
                conf >> v;
                rate_decrease_interval = v;
                std::cerr << "RATE_DECREASE_INTERVAL\t\t" << rate_decrease_interval << "\n";
            } else if (key.compare("MIN_RATE") == 0) {
                conf >> min_rate;
                std::cerr << "MIN_RATE\t\t" << min_rate << "\n";
            } else if (key.compare("FCT_OUTPUT_FILE") == 0) {
                conf >> fct_output_file;
                std::cerr << "FCT_OUTPUT_FILE\t\t" << fct_output_file << '\n';
            } else if (key.compare("SND_RCV_OUTPUT_FILE") == 0) {
                conf >> snd_rcv_output_file;
                std::cerr << "SND_RCV_OUTPUT_FILE\t\t" << snd_rcv_output_file << '\n';
            } else if (key.compare("ROUTING_TABLE_OUTPUT_FILE") == 0) {
                conf >> routing_table_output_file;
                std::cerr << "ROUTING_TABLE_OUTPUT_FILE\t\t" << routing_table_output_file << '\n';
            } else if (key.compare("HAS_WIN") == 0) {
                conf >> has_win;
                std::cerr << "HAS_WIN\t\t" << has_win << "\n";
            } else if (key.compare("GLOBAL_T") == 0) {
                conf >> global_t;
                std::cerr << "GLOBAL_T\t\t" << global_t << '\n';
            }
            else if (key.compare("ENABLE_PLB") == 0) {
                conf >> enable_plb;
                std::cerr << "ENABLE_PLB\t\t" << enable_plb << '\n';
            } else if (key.compare("ENABLE_QLEN_AWARE_EG") == 0) {
                conf >> qlen_aware_egress;
                std::cerr << "ENABLE_QLEN_AWARE_EG\t\t" << qlen_aware_egress << '\n';
            } 
            
            else if (key.compare("MI_THRESH") == 0) {
                conf >> mi_thresh;
                std::cerr << "MI_THRESH\t\t" << mi_thresh << '\n';
            } else if (key.compare("VAR_WIN") == 0) {
                uint32_t v;
                conf >> v;
                var_win = v;
                std::cerr << "VAR_WIN\t\t" << v << '\n';
            } else if (key.compare("FAST_REACT") == 0) {
                uint32_t v;
                conf >> v;
                fast_react = v;
                std::cerr << "FAST_REACT\t\t" << v << '\n';
            } else if (key.compare("U_TARGET") == 0) {
                conf >> u_target;
                std::cerr << "U_TARGET\t\t" << u_target << '\n';
            } else if (key.compare("INT_MULTI") == 0) {
                conf >> int_multi;
                std::cerr << "INT_MULTI\t\t\t\t" << int_multi << '\n';
            } else if (key.compare("RATE_BOUND") == 0) {
                uint32_t v;
                conf >> v;
                rate_bound = v;
                std::cerr << "RATE_BOUND\t\t" << rate_bound << '\n';
            } else if (key.compare("ACK_HIGH_PRIO") == 0){
				conf >> ack_high_prio;
				std::cout << "ACK_HIGH_PRIO\t\t" << ack_high_prio << '\n';
			}
             else if (key.compare("DCTCP_RATE_AI") == 0) {
                conf >> dctcp_rate_ai;
                std::cerr << "DCTCP_RATE_AI\t\t\t\t" << dctcp_rate_ai << "\n";
            } else if (key.compare("PFC_OUTPUT_FILE") == 0) {
                conf >> pfc_output_file;
                std::cerr << "PFC_OUTPUT_FILE\t\t\t\t" << pfc_output_file << '\n';
            } else if (key.compare("LINK_DOWN") == 0) {
                conf >> link_down_time >> link_down_A >> link_down_B;
                std::cerr << "LINK_DOWN\t\t\t\t" << link_down_time << ' ' << link_down_A << ' '
                          << link_down_B << '\n';
            } else if (key.compare("KMAX_MAP") == 0) {
                int n_k;
                conf >> n_k;
                std::cerr << "KMAX_MAP\t\t\t\t";
                for (int i = 0; i < n_k; i++) {
                    uint64_t rate;
                    uint32_t k;
                    conf >> rate >> k;
                    rate2kmax[rate] = k;
                    std::cerr << ' ' << rate << ' ' << k;
                }
                std::cerr << '\n';
            } else if (key.compare("KMIN_MAP") == 0) {
                int n_k;
                conf >> n_k;
                std::cerr << "KMIN_MAP\t\t\t\t";
                for (int i = 0; i < n_k; i++) {
                    uint64_t rate;
                    uint32_t k;
                    conf >> rate >> k;
                    rate2kmin[rate] = k;
                    std::cerr << ' ' << rate << ' ' << k;
                }
                std::cerr << '\n';
            } else if (key.compare("PMAX_MAP") == 0) {
                int n_k;
                conf >> n_k;
                std::cerr << "PMAX_MAP\t\t\t\t";
                for (int i = 0; i < n_k; i++) {
                    uint64_t rate;
                    double p;
                    conf >> rate >> p;
                    rate2pmax[rate] = p;
                    std::cerr << ' ' << rate << ' ' << p;
                }
                std::cerr << '\n';
            } else if (key.compare("BUFFER_SIZE") == 0) {
                conf >> buffer_size;
                std::cerr << "BUFFER_SIZE\t\t\t\t" << buffer_size << '\n';
            } else if (key.compare("QLEN_MON_FILE") == 0) {
                conf >> qlen_mon_file;
                std::cerr << "QLEN_MON_FILE\t\t\t\t" << qlen_mon_file << '\n';
            } else if (key.compare("VOQ_MON_FILE") == 0) {
                conf >> voq_mon_file;
                std::cerr << "VOQ_MON_FILE\t\t\t\t" << voq_mon_file << '\n';
            } else if (key.compare("VOQ_MON_DETAIL_FILE") == 0) {
                conf >> voq_mon_detail_file;
                std::cerr << "VOQ_MON_DETAIL_FILE\t\t\t\t" << voq_mon_detail_file << '\n';
            } else if (key.compare("UPLINK_MON_FILE") == 0) {
                conf >> uplink_mon_file;
                std::cerr << "UPLINK_MON_FILE\t\t\t\t" << uplink_mon_file << '\n';
            } else if (key.compare("CONN_MON_FILE") == 0) {
                conf >> conn_mon_file;
                std::cerr << "CONN_MON_FILE\t\t\t\t" << conn_mon_file << '\n';
            } else if (key.compare("QLEN_MON_START") == 0) {
                conf >> qlen_mon_start;
                std::cerr << "QLEN_MON_START\t\t\t\t" << qlen_mon_start << '\n';
            } else if (key.compare("QLEN_MON_END") == 0) {
                conf >> qlen_mon_end;
                std::cerr << "QLEN_MON_END\t\t\t\t" << qlen_mon_end << '\n';
            } else if (key.compare("MULTI_RATE") == 0) {
                int v;
                conf >> v;
                multi_rate = v;
                std::cerr << "MULTI_RATE\t\t\t\t" << multi_rate << '\n';
            } else if (key.compare("SAMPLE_FEEDBACK") == 0) {
                int v;
                conf >> v;
                sample_feedback = v;
                std::cerr << "SAMPLE_FEEDBACK\t\t\t\t" << sample_feedback << '\n';
            } else if (key.compare("LOAD") == 0) {
                double v;
                conf >> v;
                load = v;
                std::cerr << "LOAD\t\t\t" << load << "\n";
            } else if (key.compare("ENABLE_IRN") == 0) {
                bool v;
                conf >> v;
                enable_irn = v;
                std::cerr << "ENABLE_IRN\t\t" << enable_irn << "\n";
            } else if (key.compare("RANDOM_SEED") == 0) {
                int v;
                conf >> v;
                random_seed = v;
                std::cerr << "RANDOM_SEED\t\t\t" << random_seed << "\n";
            }
            std::string comment;
            conf >> comment;
            fflush(stdout);
        }
        conf.close();

    } else {
        std::cerr << "Error: require a config file\n";
        fflush(stdout);
        return 1;
    }

    /******************* READING CONFIG FILE IS DONE ***********************/

    /**
     * Activate ns3 logging
     */
    LogComponentEnable("GENERIC_SIMULATION", LOG_LEVEL_DEBUG);

    /**
     * @brief Random seed setup
     */
    NS_LOG_INFO("Initialize random seed: " << random_seed);
    srand((unsigned)random_seed);
    SeedManager::SetSeed(random_seed);

    /**
     * @brief PFC/QCN setup
     */
    bool dynamicth = use_dynamic_pfc_threshold;
    Config::SetDefault("ns3::QbbNetDevice::PauseTime", UintegerValue(pause_time));
    Config::SetDefault("ns3::QbbNetDevice::QcnEnabled", BooleanValue(enable_qcn));
    Config::SetDefault("ns3::QbbNetDevice::DynamicThreshold", BooleanValue(dynamicth));
    Config::SetDefault("ns3::QbbNetDevice::QbbEnabled", BooleanValue(enable_pfc));      // newly added

    if (cc_mode != 1 && lb_mode == 9) {
        std::cout << "Currently, ConWeave supports only DCQCN congestion control for RDMA. \nIf "
                     "you want to extend, the reordering delay at DstTor must be considered."
                  << std::endl;
        exit(1);
    }

    /**
     * @brief INT header setup
     */
    IntHop::multi = int_multi;
    // IntHeader::mode
    if (cc_mode == 7)  // timely, use ts
        IntHeader::mode = 1;
    else if (cc_mode == 3)  // hpcc, use int
        IntHeader::mode = 0;
    else  // others, no extra header
        IntHeader::mode = 5;

    /**
     * @brief open topology config, input-flows config.
     */
    topof.open(topology_file.c_str());
    uint32_t node_num, switch_num, link_num;
    topof >> node_num >> switch_num >> link_num;
    for (int i = 0; i < node_num; i++) {
        for (int j = 0; j < node_num; j++) {
            nodeId2NodeIdConnNum[i][j] = 0;
            nodeId2NodeIdConnNum[j][i] = 0;
        }
    }

    if (is_flow_relational == 0) {
        flowf.open(flow_file.c_str());
        flowf >> flow_num;
    } else {
        // do nothing, things will be handled in <ParseRelationalFlowFile()>
        // flowf.open(relational_flow_file.c_str());
    }

    /*-------Parameter of Settings-------*/
    Settings::node_num = node_num;
    Settings::host_num = node_num - switch_num;
    Settings::switch_num = switch_num;
    Settings::lb_mode = lb_mode;
    Settings::packet_payload = packet_payload_size;

    Settings::record_switch_spray = 1;
    // Settings::MTU = packet_payload_size + 48;  // for simplicity
    /*------------------------------------*/

    std::vector<uint32_t> node_type(node_num, 0);
    for (uint32_t i = 0; i < switch_num; i++) {
        uint32_t sid;
        topof >> sid;
        node_type[sid] = 1;
    }

    // configure switch
    for (uint32_t i = 0; i < node_num; i++) {
        if (node_type[i] == 0)
            n.Add(CreateObject<Node>());
        else {
            Ptr<SwitchNode> sw = CreateObject<SwitchNode>();
            n.Add(sw);
            sw->SetAttribute("EcnEnabled", BooleanValue(enable_qcn));
            sw->SetAttribute("QlenAwareEgressSelection", BooleanValue(bool(qlen_aware_egress)));
        }
    }
    NS_LOG_INFO("Create nodes.");

    /*----------------------------------------*/

    InternetStackHelper internet;
    internet.Install(n);  // aggregate ipv4, ipv6, udp, tcp, etc

    //
    // Assign IP to each server
    //
    for (uint32_t i = 0; i < node_num; i++) {
        if (n.Get(i)->GetNodeType() == 0) {  // is server
            serverAddress.resize(i + 1);
            serverAddress[i] = Settings::node_id_to_ip(i);
        }
    }

    NS_LOG_INFO("Create channels.");

    //
    // Explicitly create the channels required by the topology.
    //
    Ptr<RateErrorModel> rem = CreateObject<RateErrorModel>();
    Ptr<UniformRandomVariable> uv = CreateObject<UniformRandomVariable>();
    rem->SetRandomVariable(uv);
    uv->SetStream(50);
    rem->SetAttribute("ErrorRate", DoubleValue(error_rate_per_link));
    rem->SetAttribute("ErrorUnit", StringValue("ERROR_UNIT_PACKET"));

    pfc_file = fopen(pfc_output_file.c_str(), "w");
    routing_table_output = fopen(routing_table_output_file.c_str(), "w");
    snd_rcv_output = fopen(snd_rcv_output_file.c_str(), "w");

    QbbHelper qbb;
    Ipv4AddressHelper ipv4;
    uint32_t max_bw = 0;
    uint32_t max_delay = 0;
    
    std::vector<std::pair<uint32_t, uint32_t>> link_pairs;  // src, dst link pairs
    for (uint32_t i = 0; i < link_num; i++) {
        uint32_t src, dst;
        std::string data_rate, link_delay;
        double error_rate;
        topof >> src >> dst >> data_rate >> link_delay >> error_rate;
        if (extractInteger(data_rate) > max_bw)
            max_bw = extractInteger(data_rate);
        if (extractFloat(link_delay) > max_delay)
            max_delay = (uint32_t)extractFloat(link_delay);
        // std::cout << max_bw << max_delay << std::endl;

        /** ASSUME: fixed one-hop delay across network */
        // assert(std::to_string(one_hop_delay) + "ns" == link_delay);

        link_pairs.push_back(std::make_pair(src, dst));
        Ptr<Node> snode = n.Get(src), dnode = n.Get(dst);

        nodeId2NodeIdConnNum[src][dst] += 1;
        nodeId2NodeIdConnNum[dst][src] += 1;
        
        qbb.SetDeviceAttribute("DataRate", StringValue(data_rate));     // 重要
        qbb.SetChannelAttribute("Delay", StringValue(link_delay));

        if (error_rate > 0) {
            Ptr<RateErrorModel> rem = CreateObject<RateErrorModel>();
            Ptr<UniformRandomVariable> uv = CreateObject<UniformRandomVariable>();
            rem->SetRandomVariable(uv);
            uv->SetStream(50);
            rem->SetAttribute("ErrorRate", DoubleValue(error_rate));
            rem->SetAttribute("ErrorUnit", StringValue("ERROR_UNIT_PACKET"));
            qbb.SetDeviceAttribute("ReceiveErrorModel", PointerValue(rem));
        } else {
            qbb.SetDeviceAttribute("ReceiveErrorModel", PointerValue(rem));
        }

        fflush(stdout);

        // Assigne server IP
        // Note: this should be before the automatic assignment below (ipv4.Assign(d)),
        // because we want our IP to be the primary IP (first in the IP address list),
        // so that the global routing is based on our IP
        NetDeviceContainer d = qbb.Install(snode, dnode);
        if (snode->GetNodeType() == 0) {
            Ptr<Ipv4> ipv4 = snode->GetObject<Ipv4>();
            ipv4->AddInterface(d.Get(0));
            ipv4->AddAddress(1, Ipv4InterfaceAddress(serverAddress[src], Ipv4Mask(0xff000000)));
        }
        if (dnode->GetNodeType() == 0) {
            Ptr<Ipv4> ipv4 = dnode->GetObject<Ipv4>();
            ipv4->AddInterface(d.Get(1));
            ipv4->AddAddress(1, Ipv4InterfaceAddress(serverAddress[dst], Ipv4Mask(0xff000000)));
        }

        // used to create a graph of the topology，具体数值在 qbb.SetDeviceAttribute 中赋值的
        Interface inf1, inf2;
        inf1.idx = DynamicCast<QbbNetDevice>(d.Get(0))->GetIfIndex();
        inf1.peerIdx = DynamicCast<QbbNetDevice>(d.Get(1))->GetIfIndex();
        inf1.up = true;
        inf1.delay = DynamicCast<QbbChannel>(DynamicCast<QbbNetDevice>(d.Get(0))->GetChannel())->GetDelay().GetTimeStep();
        inf1.bw = DynamicCast<QbbNetDevice>(d.Get(0))->GetDataRate().GetBitRate();
        nbr2if[snode][dnode].push_back(inf1);

        inf2.idx = DynamicCast<QbbNetDevice>(d.Get(1))->GetIfIndex();
        inf2.peerIdx = DynamicCast<QbbNetDevice>(d.Get(0))->GetIfIndex();
        inf2.up = true;
        inf2.delay = DynamicCast<QbbChannel>(DynamicCast<QbbNetDevice>(d.Get(1))->GetChannel())->GetDelay().GetTimeStep();
        inf2.bw = DynamicCast<QbbNetDevice>(d.Get(1))->GetDataRate().GetBitRate();
        nbr2if[dnode][snode].push_back(inf2);
        
        // This is just to set up the connectivity between nodes. The IP addresses are useless
        char ipstring[16];
        Ipv4Address x;
        sprintf(ipstring, "10.%d.%d.0", i / 254 + 1, i % 254 + 1);
        ipv4.SetBase(ipstring, "255.255.255.0");
        ipv4.Assign(d);

        // setup PFC trace
        DynamicCast<QbbNetDevice>(d.Get(0))->TraceConnectWithoutContext(
            "QbbPfc", MakeBoundCallback(&get_pfc, pfc_file, DynamicCast<QbbNetDevice>(d.Get(0))));
        DynamicCast<QbbNetDevice>(d.Get(1))->TraceConnectWithoutContext(
            "QbbPfc", MakeBoundCallback(&get_pfc, pfc_file, DynamicCast<QbbNetDevice>(d.Get(1))));

        // 新增
        DynamicCast<QbbNetDevice>(d.Get(0))->TraceConnectWithoutContext(
            "SndRcvRecord", MakeBoundCallback(&snd_rcv_record, snd_rcv_output, DynamicCast<QbbNetDevice>(d.Get(0))));
        DynamicCast<QbbNetDevice>(d.Get(1))->TraceConnectWithoutContext(
            "SndRcvRecord", MakeBoundCallback(&snd_rcv_record, snd_rcv_output, DynamicCast<QbbNetDevice>(d.Get(1))));
    }
    std::cout << "max_bw: " << max_bw << "max_delay: " << max_delay << std::endl;
    std::cout << "(AVG) NIC RATE: " << get_nic_rate(n) << std::endl;

    /* Get IP address <-> NodeID pairs */
    Ipv4Address empty_ip;
    for (uint32_t i = 0; i < node_num; ++i) {
        if (n.Get(i)->GetNodeType() == 0) {  // is server
            if (serverAddress[i].IsEqual(empty_ip)) {
                printf("XXX ERROR %d\n", i);
                printf("size of serverAddress: %lu", serverAddress.size());
                NS_FATAL_ERROR("An end-host belongs to no link");
            }
        }
        Settings::hostId2IpMap[i] = serverAddress[i].Get();
        Settings::hostIp2IdMap[serverAddress[i].Get()] = i;
    }

    // config switch
    for (uint32_t i = 0; i < node_num; i++) {
        if (n.Get(i)->GetNodeType() == 1) {  // is switch
            Ptr<SwitchNode> sw = DynamicCast<SwitchNode>(n.Get(i));
            uint32_t shift = 3;  // by default 1/8
            for (uint32_t j = 1; j < sw->GetNDevices(); j++) {
                Ptr<QbbNetDevice> dev = DynamicCast<QbbNetDevice>(sw->GetDevice(j));
                // set ecn
                uint64_t rate = dev->GetDataRate().GetBitRate();
                NS_ASSERT_MSG(rate2kmin.find(rate) != rate2kmin.end(),
                              "must set kmin for each link speed");
                NS_ASSERT_MSG(rate2kmax.find(rate) != rate2kmax.end(),
                              "must set kmax for each link speed");
                NS_ASSERT_MSG(rate2pmax.find(rate) != rate2pmax.end(),
                              "must set pmax for each link speed");
                assert(rate2kmin.find(rate) != rate2kmin.end() &&
                       rate2kmax.find(rate) != rate2kmax.end() &&
                       rate2pmax.find(rate) != rate2pmax.end());
                sw->m_mmu->ConfigEcn(j, rate2kmin[rate], rate2kmax[rate], rate2pmax[rate]);
                // set pfc
                uint64_t delay = DynamicCast<QbbChannel>(dev->GetChannel())->GetDelay().GetTimeStep();
                // 2RTT + 2MTU
                uint32_t headroom = rate * delay / 8 / 1000000000 * 2 + 2 * sw->m_mmu->MTU;
                sw->m_mmu->ConfigHdrm(j, headroom);
            }
            sw->m_mmu->ConfigNPort(sw->GetNDevices() - 1);
            sw->m_mmu->ConfigBufferSize(buffer_size * 1024 * 1024);  // default 0, specify in run.py!!
            sw->m_mmu->node_id = sw->GetId();
            oss.str("");oss.clear();
            oss << "Node " << i << " : Broadcom switch (" << (sw->GetNDevices() - 1)
                << " ports / " << (sw->m_mmu->GetMmuBufferBytes() / 1000000.0) << "MB MMU)\n";
            NS_LOG_INFO(oss.str());
            // 添加统计回调函数
            sw->TraceConnectWithoutContext(
            "SwitchSprayEventRecord", MakeBoundCallback(&switch_spray_event_record, snd_rcv_output, sw));
        }
    }

    fct_output = fopen(fct_output_file.c_str(), "w");
    flow_input_stream = fopen(flow_input_file.c_str(), "w");
    if (cc_mode == 1) {
        cnp_output = fopen(cnp_output_file.c_str(), "w");
    }

    /**
     * @brief install RDMA driver (Mellanox parameters)
     *
     * [ClampTargetRate]clamp_tgt_rate (false) - when receiving a CNP, the target rate is always
     *updated to be the current rate
     *[-]clamp_tgt_rate_after_time_inc (true) - when receiving a CNP, the target rate is updated to
     *be the current rate also if the last rate increase event was due to the timer, and not only
     *due to the byte counter
     * [-]initial_alpha_value(1023) -
     * [RateDecreaseInterval]rate_reduce_monitor_period(4) - Minimal interval for rate reduction for
     *a flow. If a CNP is received during the interval, the flow rate is reduced at the beginning of
     *the next rate_reduce_monitor_period interval to (1-Alpha/Gd)*CurrentRate. rpg_gd is given as
     *log2(Gd), where Gd may only be powers of 2.
     * [-]rpg_gd(11) - If an CNP is received, the flow rate is reduced at the beginning of the next
     *rate_reduce_monitor_period interval to (1-Alpha/Gd)*CurrentRate.
     * -> in this simulator, (alpha / gd) ~ 0.5 setup, initially. We do not need rpg_gd parameter.
     * [RateOnFirstCnp]rate_to_set_on_first_cnp(0) - The rate that is set for the flow, upon first
     *CNP received, in Mbps. [RPTimer]rpg_time_reset(300us) - Time counter for rate increase event
     *[FastRecoveryTimes]rpg_threshold(1) - Number of rate increase events for switching between
     *Fast Recovery, Active Increase, Hyper Active Increase modes.
     * [AlphaResumInterval]dce_tcp_rtt(1) - Window for sampling of moving average calculation of
     *alpha
     * [-]dce_tcp_g(1019) - Weight of the new sampling in moving average calculation of alpha
     * [-]rpg_byte_reset(32767) - Byte counter for rate increase event
     * [-]rpg_min_dec_fac(50) -  Maximal factor by which the rate can be reduced (2 means that the
     *new rate can be divided by 2 at maximum)
     */
    if (is_spray) {

    }
    // manually type BDP
    std::map<std::string, uint32_t> topo2bdpMap;
    topo2bdpMap[std::string("leaf_spine_128_100G_OS2")] = 104000;  // RTT=8320
    topo2bdpMap[std::string("fat_k8_100G_OS2")] = 156000;      // RTT=12480 --> all 100G links


    // 在异构网络场景下，原有的map不能满足需求
    std::map<std::pair<uint32_t, uint32_t>, uint32_t> nodepair2bdpMap;
    /*
        // 定义两个节点
        uint32_t src_node = 1;
        uint32_t dst_node = 2;
        
        // 插入一个 key (src_node, dst_node) 对应的值
        nodepair2bdpMap[std::make_pair(src_node, dst_node)] = 100;
        
        // 访问这个 key 的值
        uint32_t bdp = nodepair2bdpMap[std::make_pair(src_node, dst_node)];
    */

    // // 计算跨AZ和AZ内的BDP
    // uint32_t rtt_cross_az = max_delay * 2;      // ns，1-hop * 2
    // uint32_t bw_cross_az = 100;     // Gb
    // uint32_t rtt_within_az = 6000;  // ns，6-hops
    // uint32_t bw_within_az = 100;    // Gb
    // uint32_t tx_delay_per_hop = 80;    // Gb

    // topo2bdpMap[std::string("dumbbell_cross_AZ")] = rtt_cross_az * bw_cross_az;
    // topo2bdpMap[std::string("dumbbell_within_AZ")] = rtt_cross_az * bw_cross_az;
    // topo2bdpMap[std::string("dumbbell")] = topo2bdpMap[std::string("dumbbell_cross_AZ")] 
    //                                     + 2 * topo2bdpMap[std::string("dumbbell_within_AZ")] / 2;
    topo2bdpMap[std::string("dumbbell")] = 25157000;      // 手动计算：源DC 3跳 + DC间一跳 + 目的DC 3跳，txdelay只计算oneway
    // topo2bdpMap[std::string("dumbbell")] = 1560000;

    // topology_file
    bool found_topo2bdpMap = false;
    uint32_t irn_bdp_lookup = 0;
    for (auto pair : topo2bdpMap) {
        if (topology_file.find(pair.first) != std::string::npos) {  // if topology file string includes the word
            irn_bdp_lookup = pair.second;
            found_topo2bdpMap = true;
            break;
        }
    }
    if (found_topo2bdpMap == false) {
        std::cout << __FILE__ << "(" << __LINE__ << ")"
                  << " ERROR - topo2bdpMap has no matched item with " << topology_file << std::endl;
        assert(false);
    }

    // rdmaHw config
    for (uint32_t i = 0; i < node_num; i++) {
        if (n.Get(i)->GetNodeType() == 0) {  // is server
            // create RdmaHw
            Ptr<RdmaHw> rdmaHw = CreateObject<RdmaHw>();
            rdmaHw->SetAttribute("ClampTargetRate", BooleanValue(clamp_target_rate));
            rdmaHw->SetAttribute("AlphaResumInterval", DoubleValue(alpha_resume_interval));
            rdmaHw->SetAttribute("RPTimer", DoubleValue(rp_timer));
            rdmaHw->SetAttribute("FastRecoveryTimes", UintegerValue(fast_recovery_times));
            rdmaHw->SetAttribute("EwmaGain", DoubleValue(ewma_gain));
            rdmaHw->SetAttribute("RateAI", DataRateValue(DataRate(rate_ai)));
            rdmaHw->SetAttribute("RateHAI", DataRateValue(DataRate(rate_hai)));
            rdmaHw->SetAttribute("L2BackToZero", BooleanValue(l2_back_to_zero));
            rdmaHw->SetAttribute("L2ChunkSize", UintegerValue(l2_chunk_size));
            rdmaHw->SetAttribute("L2AckInterval", UintegerValue(l2_ack_interval));
            rdmaHw->SetAttribute("CcMode", UintegerValue(cc_mode));     // 设置 host 上的 CC mode
            rdmaHw->SetAttribute("RateDecreaseInterval", DoubleValue(rate_decrease_interval));
            rdmaHw->SetAttribute("MinRate", DataRateValue(DataRate(min_rate)));
            rdmaHw->SetAttribute("Mtu", UintegerValue(packet_payload_size));
            rdmaHw->SetAttribute("MiThresh", UintegerValue(mi_thresh));
            rdmaHw->SetAttribute("VarWin", BooleanValue(var_win));
            rdmaHw->SetAttribute("FastReact", BooleanValue(fast_react));
            rdmaHw->SetAttribute("MultiRate", BooleanValue(multi_rate));
            rdmaHw->SetAttribute("SampleFeedback", BooleanValue(sample_feedback));
            rdmaHw->SetAttribute("TargetUtil", DoubleValue(u_target));
            rdmaHw->SetAttribute("RateBound", BooleanValue(rate_bound));
            rdmaHw->SetAttribute("DctcpRateAI", DataRateValue(DataRate(dctcp_rate_ai)));
            // 新加的
            rdmaHw->SetAttribute("IrnEnable", BooleanValue(enable_irn));
            // topo2bdpMap (e.g., longest BDP 25000: 8us * 25Gbps)
            rdmaHw->SetAttribute("IrnRtoHigh", TimeValue(MicroSeconds(1000000)));  // 1s
            rdmaHw->SetAttribute("IrnRtoLow", TimeValue(MicroSeconds(1000000)));   // 1s
            rdmaHw->SetAttribute("IrnBdp", UintegerValue(irn_bdp_lookup));
            rdmaHw->SetAttribute("L2Timeout", TimeValue(MicroSeconds(1000000)));    // 超时 1s
            // Monitoring CNP Marking frequency of DCQCN
            if (cc_mode == 1) {
                Simulator::Schedule(NanoSeconds(cnp_mon_start), &cnp_freq_monitoring, cnp_output,
                                    rdmaHw);
            }

            // create and install RdmaDriver
            Ptr<RdmaDriver> rdma = CreateObject<RdmaDriver>();
            Ptr<Node> node = n.Get(i);
            rdma->SetNode(node);
            rdma->SetRdmaHw(rdmaHw);

            node->AggregateObject(rdma);
            rdma->Init();
            rdma->TraceConnectWithoutContext("QpComplete",
                                             MakeBoundCallback(qp_finish, fct_output));
        }
    }

    // set ACK priority on hosts
    int ack_hi_prio = 1;
	if (ack_high_prio == 0)
		ack_hi_prio = 0;

    /**
     * @brief setup switch's CcMode and ACK with high priority
     */
    for (uint32_t i = 0; i < node_num; i++) {
        if (n.Get(i)->GetNodeType() == 1) {  // switch
            Ptr<SwitchNode> sw = DynamicCast<SwitchNode>(n.Get(i));
            sw->SetAttribute("CcMode", UintegerValue(cc_mode));     // 设置 switch 上的 CC mode
            sw->SetAttribute("AckHighPrio", UintegerValue(ack_hi_prio));
        }
    }

    /**
     * @brief setup routing
     */
    CalculateRoutes(n);
    SetRoutingEntries();

    /**
     * @brief get BDP and delay
     */
    maxRtt = maxBdp = 0;
    fprintf(stderr, "node_num=%d\n", node_num);
    for (uint32_t i = 0; i < node_num; i++) {
        if (n.Get(i)->GetNodeType() != 0) 
            continue;
        for (uint32_t j = i + 1; j < node_num; j++) {
            if (n.Get(j)->GetNodeType() != 0) 
                continue;
            uint64_t delay = pairDelay[n.Get(i)][n.Get(j)];
            uint64_t txDelay = pairTxDelay[n.Get(i)][n.Get(j)];
            uint64_t rtt = delay * 2 + txDelay;                 // 这里计算RTT的方法有误吧，txDelay也得乘以2
            uint64_t bw = pairBw[n.Get(i)][n.Get(j)];
            uint64_t bdp = rtt * bw / 1000000000 / 8;
            pairBdp[n.Get(i)][n.Get(j)] = bdp;
            pairBdp[n.Get(j)][n.Get(i)] = bdp;
            pairRtt[n.Get(i)][n.Get(j)] = rtt;
            pairRtt[n.Get(j)][n.Get(i)] = rtt;
            if (bdp > maxBdp) 
                maxBdp = bdp;
                // std::cout << delay << ", " << txDelay << ", " << rtt << ", " << bw << ", " << bdp << std::endl;
            if (rtt > maxRtt) 
                maxRtt = rtt;
        }
    }
    fprintf(stderr, "maxRtt: %lu, maxBdp: %lu\n", maxRtt, maxBdp);
    // assert(maxBdp == irn_bdp_lookup);

    // for (uint32_t i = 0; i < node_num; i++) {
    //     if (n.Get(i)->GetNodeType() != 0) 
    //         continue;
    //     for (uint32_t j = 0; j < node_num; j++) {
    //         if (n.Get(j)->GetNodeType() != 0) 
    //             continue;
    //         printf("node %u <--> node %u: Bdp=%lu, Rtt=%lu\n", i, j, pairBdp[n.Get(i)][n.Get(j)], pairRtt[n.Get(i)][n.Get(j)]);
    //     }
    // }
    // printf("actively pause, It's time to check RTT and BDP between server nodes\n");
    // exit(1);
    
// 新加开始
// TODO: Check： 必须要在topo文件中把host放在src位置
    std::cout << "Configuring switches" << std::endl;
    /* config ToR Switch */
    for (auto &pair : link_pairs) {     // std::vector<std::pair<uint32_t, uint32_t>> link_pairs;  // src, dst link pairs
        Ptr<Node> probably_host = n.Get(pair.first);
        Ptr<Node> probably_switch = n.Get(pair.second);
        // host-switch link
        if (probably_host->GetNodeType() == 0 && probably_switch->GetNodeType() == 1) {
            Ptr<SwitchNode> sw = DynamicCast<SwitchNode>(probably_switch);
            // std::cout << "a ToR switch: " << sw->GetId() << std::endl;
            sw->m_isToR = true;
            uint32_t hostIP = serverAddress[pair.first].Get();
            sw->m_isToR_hostIP.insert(hostIP);      // 每个switch上维护一个本交换机连接的host IP列表
            if (idxNodeToR.find(sw->GetId()) == idxNodeToR.end()) {
                idxNodeToR[sw->GetId()] = sw;
            };
        }
    }

    /* config load balancer's switches using ToR-to-ToR routing */
    if (lb_mode == 3 || lb_mode == 6 || lb_mode == 9) {  // Conga, Letflow, Conweave
        NS_LOG_INFO("Configuring Load Balancer's Switches");
        for (auto &pair : link_pairs) {
            Ptr<Node> probably_host = n.Get(pair.first);
            Ptr<Node> probably_switch = n.Get(pair.second);

            // host-switch link
            if (probably_host->GetNodeType() == 0 && probably_switch->GetNodeType() == 1) {
                Ptr<SwitchNode> sw = DynamicCast<SwitchNode>(probably_switch);
                uint32_t hostIP = serverAddress[pair.first].Get();
                // 每个host都知道连的是哪个switch
                Settings::hostIp2SwitchId[hostIP] = sw->GetId();  // hostIP -> connected switch's ID
            }
        }

        // 这个大的for循环干的事情就是，为每一个{src-dst} pair 计算它们的路径（第一跳的出端口，第二跳的出端口，…）
        // 最大四跳，并插入到每个交换机的路由表中
        // Conga: m_congaFromLeafTable, m_congaToLeafTable, m_congaRoutingTable
        // Letflow: m_letflowRoutingTable
        // Conweave: m_ConWeaveRoutingTable, m_rxToRId2BaseRTT
        for (auto i = nextHop.begin(); i != nextHop.end(); i++) {  // every node
            if (i->first->GetNodeType() == 1) {                    // switch
                Ptr<Node> nodeSrc = i->first;
                Ptr<SwitchNode> swSrc = DynamicCast<SwitchNode>(nodeSrc);  // switch
                uint32_t swSrcId = swSrc->GetId();

                if (swSrc->m_isToR) {
                    // printf("--- ToR Switch %d\n", swSrcId);

                    auto table1 = i->second;
                    for (auto j = table1.begin(); j != table1.end(); j++) { // 对该switch的每一个dest及其可走的下一跳
                        Ptr<Node> dst = j->first;  // dst
                        uint32_t dstIP = Settings::hostId2IpMap[dst->GetId()];
                        uint32_t swDstId = Settings::hostIp2SwitchId[dstIP];  // Rx(dst)ToR

                        if (swSrcId == swDstId) {
                            continue;  // if in the same pod, then skip
                        }

                        if (lb_mode == 3) {
                            // initialize `m_congaFromLeafTable` and `m_congaToLeafTable`
                            swSrc->m_mmu->m_congaRouting
                                .m_congaFromLeafTable[swDstId];  // dynamically will be added in
                                                                 // conga
                            swSrc->m_mmu->m_congaRouting.m_congaToLeafTable[swDstId];
                        }

                        // construct paths    这里犯懒，直接用第0根链路，但是其实最好是真正实现一下
                        uint32_t pathId;
                        uint8_t path_ports[4] = {0, 0, 0, 0};  // interface is always large than 0
                        vector<Ptr<Node>> nexts1 = j->second;   // 前往 dest 的所有下一跳
                        for (auto next1 : nexts1) {     // 对每一个下一跳的节点
                            uint32_t outPort1 = nbr2if[nodeSrc][next1][0].idx; // 从src ToR 到 目的地的每一个下一跳端口
                            auto nexts2 = nextHop[next1][dst];              // 从每个<下一跳到dst>的下一跳节点
                            if (nexts2.size() == 1 && nexts2[0]->GetId() == swDstId) {
                                // this destination has 2-hop distance
                                uint32_t outPort2 = nbr2if[next1][nexts2[0]][0].idx;
                                // printf("[IntraPod-2hop] %d (%d)-> %d (%d) -> %d -> %d\n",
                                // nodeSrc->GetId(), outPort1, next1->GetId(), outPort2,
                                // nexts2[0]->GetId(), dst->GetId());
                                path_ports[0] = (uint8_t)outPort1;  // 第一跳节点上的出端口号
                                path_ports[1] = (uint8_t)outPort2;  // 第二跳节点上的出端口号
                                pathId = *((uint32_t *)path_ports);
                                if (lb_mode == 3) {
                                    swSrc->m_mmu->m_congaRouting.m_congaRoutingTable[swDstId]
                                        .insert(pathId);
                                }
                                if (lb_mode == 6) {
                                    swSrc->m_mmu->m_letflowRouting.m_letflowRoutingTable[swDstId]
                                        .insert(pathId);
                                }
                                if (lb_mode == 9) {
                                    swSrc->m_mmu->m_conweaveRouting.m_ConWeaveRoutingTable[swDstId]
                                        .insert(pathId);
                                    swSrc->m_mmu->m_conweaveRouting.m_rxToRId2BaseRTT[swDstId] =
                                        one_hop_delay * 4;
                                }
                                continue;
                            }

                            for (auto next2 : nexts2) {
                                uint32_t outPort2 = nbr2if[next1][next2][0].idx;
                                auto nexts3 = nextHop[next2][dst];
                                if (nexts3.size() == 1 && nexts3[0]->GetId() == swDstId) {
                                    // this destination has 3-hop distance
                                    uint32_t outPort3 = nbr2if[next2][nexts3[0]][0].idx;
                                    // printf("[IntraPod-3hop] %d (%d)-> %d (%d) -> %d (%d) -> %d ->
                                    // %d\n", nodeSrc->GetId(), outPort1, next1->GetId(), outPort2,
                                    // next2->GetId(), outPort3, nexts3[0]->GetId(), dst->GetId());
                                    path_ports[0] = (uint8_t)outPort1;
                                    path_ports[1] = (uint8_t)outPort2;
                                    path_ports[2] = (uint8_t)outPort3;
                                    pathId = *((uint32_t *)path_ports);
                                    if (lb_mode == 3) {
                                        swSrc->m_mmu->m_congaRouting.m_congaRoutingTable[swDstId]
                                            .insert(pathId);
                                    }
                                    if (lb_mode == 6) {
                                        swSrc->m_mmu->m_letflowRouting
                                            .m_letflowRoutingTable[swDstId]
                                            .insert(pathId);
                                    }
                                    if (lb_mode == 9) {
                                        swSrc->m_mmu->m_conweaveRouting
                                            .m_ConWeaveRoutingTable[swDstId]
                                            .insert(pathId);
                                        swSrc->m_mmu->m_conweaveRouting.m_rxToRId2BaseRTT[swDstId] =
                                            one_hop_delay * 6;
                                    }
                                    continue;
                                }

                                for (auto next3 : nexts3) {
                                    uint32_t outPort3 = nbr2if[next2][next3][0].idx;
                                    auto nexts4 = nextHop[next3][dst];
                                    if (nexts4.size() == 1 && nexts4[0]->GetId() == swDstId) {
                                        // this destination has 4-hop distance
                                        uint32_t outPort4 = nbr2if[next3][nexts4[0]][0].idx;
                                        // printf("[IntraPod-4hop] %d (%d)-> %d (%d) -> %d (%d) ->
                                        // %d (%d) -> %d -> %d\n", nodeSrc->GetId(), outPort1,
                                        // next1->GetId(), outPort2, next2->GetId(), outPort3,
                                        // next3->GetId(), outPort4, nexts4[0]->GetId(),
                                        // dst->GetId());
                                        path_ports[0] = (uint8_t)outPort1;
                                        path_ports[1] = (uint8_t)outPort2;
                                        path_ports[2] = (uint8_t)outPort3;
                                        path_ports[3] = (uint8_t)outPort4;
                                        pathId = *((uint32_t *)path_ports);
                                        if (lb_mode == 3) {
                                            swSrc->m_mmu->m_congaRouting
                                                .m_congaRoutingTable[swDstId]
                                                .insert(pathId);
                                        }
                                        if (lb_mode == 6) {
                                            swSrc->m_mmu->m_letflowRouting
                                                .m_letflowRoutingTable[swDstId]
                                                .insert(pathId);
                                        }
                                        if (lb_mode == 9) {
                                            swSrc->m_mmu->m_conweaveRouting
                                                .m_ConWeaveRoutingTable[swDstId]
                                                .insert(pathId);
                                            swSrc->m_mmu->m_conweaveRouting
                                                .m_rxToRId2BaseRTT[swDstId] = one_hop_delay * 8;
                                        }
                                        continue;
                                    } else {
                                        printf("Too large topology?\n");
                                        assert(false);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // m_outPort2BitRateMap - only for Conga
        for (auto i = nextHop.begin(); i != nextHop.end(); i++) {  // every node
            if (i->first->GetNodeType() == 1) {                    // switch
                Ptr<Node> node = i->first;
                Ptr<SwitchNode> sw = DynamicCast<SwitchNode>(node);  // switch
                uint32_t swId = sw->GetId();

                auto table = i->second;
                for (auto j = table.begin(); j != table.end(); j++) {
                    Ptr<Node> dst = j->first;  // dst
                    uint32_t dstIP = Settings::hostId2IpMap[dst->GetId()];
                    uint32_t swDstId = Settings::hostIp2SwitchId[dstIP];

                    for (auto next : j->second) {
                        uint32_t outPort = nbr2if[node][next][0].idx;
                        uint64_t bw = nbr2if[node][next][0].bw;
                        sw->m_mmu->m_congaRouting.SetLinkCapacity(outPort, bw);
                        // printf("Node: %d, interface: %d, bw: %lu\n", swId, outPort, bw);
                    }
                }
            }
        }

        // Constant setup, and switchInfo
        for (auto i = nextHop.begin(); i != nextHop.end(); i++) {  // every node
            if (i->first->GetNodeType() == 1) {
                Ptr<Node> node = i->first;
                Ptr<SwitchNode> sw = DynamicCast<SwitchNode>(node);  // switch
                // NS_LOG_INFO("Switch Info - ID:%u, ToR:%d\n" % (sw->GetId(), sw->m_isToR));
                oss.str("");oss.clear();
                oss << "Switch Info - ID:" << sw->GetId() << ", ToR:" << sw->m_isToR << "\n";
                NS_LOG_INFO(oss.str());
                if (lb_mode == 3) {
                    sw->m_mmu->m_congaRouting.SetConstants(conga_dreTime, conga_agingTime,
                                                           conga_flowletTimeout, conga_quantizeBit,
                                                           conga_alpha);
                    sw->m_mmu->m_congaRouting.SetSwitchInfo(sw->m_isToR, sw->GetId());  // 设置自己是不是ToR，以及自己的ID是多少
                }
                if (lb_mode == 6) {
                    sw->m_mmu->m_letflowRouting.SetConstants(letflow_agingTime,
                                                             letflow_flowletTimeout);
                    sw->m_mmu->m_letflowRouting.SetSwitchInfo(sw->m_isToR, sw->GetId());
                }
                if (lb_mode == 9) {
                    sw->m_mmu->m_conweaveRouting.SetConstants(
                        conweave_extraReplyDeadline, conweave_extraVOQFlushTime,
                        conweave_txExpiryTime, conweave_defaultVOQWaitingTime,
                        conweave_pathPauseTime, conweave_pathAwareRerouting);
                    sw->m_mmu->m_conweaveRouting.SetSwitchInfo(sw->m_isToR, sw->GetId());
                }
            }
        }

        // schedule conga timeout monitor
        if (lb_mode == 3) {  // CONGA
            Simulator::Schedule(Seconds(flowgen_stop_time + simulator_extra_time),
                                conga_history_print);
        }
        if (lb_mode == 6) {  // LETFLOW
            Simulator::Schedule(Seconds(flowgen_stop_time + simulator_extra_time),
                                letflow_history_print);
        }
        if (lb_mode == 9) {  // CONWEAVE
            Simulator::Schedule(Seconds(flowgen_stop_time + simulator_extra_time),
                                conweave_history_print);
        }
    }


// 新加结束

    // populate routing tables (although we use our custom impl in switch_node.cc)
    Ipv4GlobalRoutingHelper::PopulateRoutingTables();

    // maintain port number for each host
    for (uint32_t i = 0; i < node_num; i++) {
        if (n.Get(i)->GetNodeType() == 0) {
            portNumber[i] = 10000;  // each host use port number from 10000
            dportNumber[i] = 100;
        }
    }

    // 添加流
    if (is_flow_relational == 0){
        flow_input.idx = 0;
        port_per_host = new uint16_t[node_num - switch_num];    // ？host_num = port_per_host ？
        if (flow_num > 0) {
            // generate flows
            ReadFlowInput();
            // 流是现读、现安装、现运行的，不是一次性读完的
            Simulator::Schedule(Seconds(0), &ScheduleFlowInputs, flow_input_stream);
        }
    }
    else {
        // 解析依赖流
        printf("aaaaa: %s\n", flow_file.c_str());
        fflush(stdout);
        ParseRelationalFlowFile(flow_file);
        ParseNodeMapping(node_mapping);
        DoNodeSimulationMapping(simnode_mapping_file);
        // 打印 flowMap, dependencies 和 readyQueue
        // PrintFlowMap();
        // PrintDependencies();
        PrintReadyQueue();

        // 调度并发送流
        Simulator::Schedule(Seconds(global_sim_start_time), &ScheduleFlowRelational);
    }

    topof.close();

    // schedule link down
    if (link_down_time > 0) {
        Simulator::Schedule(Seconds(flowgen_start_time) + MicroSeconds(link_down_time),
                            &TakeDownLink, n, n.Get(link_down_A), n.Get(link_down_B), 0);   // 默认是intf 0
    }

    if (lb_mode == 9) {
        voq_output = fopen(voq_mon_file.c_str(), "w");                // specific to ConWeave
        voq_detail_output = fopen(voq_mon_detail_file.c_str(), "w");  // specific to ConWeave
    }

    uplink_output = fopen(uplink_mon_file.c_str(), "w");  // common
    conn_output = fopen(conn_mon_file.c_str(), "w");      // common

    // 对每个ToR switch，记录其上联端口和下联端口
    // update torId2UplinkIf, torId2DownlinkIf
    for (size_t ToRId = 0; ToRId < Settings::node_num; ToRId++) {
        Ptr<Node> node = n.Get(ToRId);
        if (node->GetNodeType() == 1) {  // switches
            auto swNode = DynamicCast<SwitchNode>(n.Get(ToRId));
            if (swNode->m_isToR) {  // TOR switch
                for (auto &nextNodeIf : nbr2if[node]) {
                    if (nextNodeIf.first->GetNodeType() == 1) {  // nextNode is switch (i.e., uplink)
                        auto &vec = torId2UplinkIf[ToRId];
                        for (int i = 0; i < nextNodeIf.second.size(); i++) {
                            vec.push_back(nextNodeIf.second[i].idx);  // record this uplink port (outDev index)
                            // printf("Sw %lu - uplink port %u\n", ToRId, nextNodeIf.second.idx);  //
                            // debugging
                        }
                    } else {
                        auto &vec = torId2DownlinkIf[ToRId];
                        for (int i = 0; i < nextNodeIf.second.size(); i++) {
                            vec.push_back(nextNodeIf.second[i].idx);  // record this downlink port (outDev index)
                            // printf("Sw %lu - downlink port %u\n", ToRId, nextNodeIf.second.idx);  //
                            // debugging
                        }
                    }
                }
            }
        }
    }
    Simulator::Schedule(Seconds(flowgen_start_time), &periodic_monitoring, voq_output,
                        voq_detail_output, uplink_output, conn_output, &lb_mode);

    //
    // Now, do the actual simulation.
    //
    std::cout << "------------------------------------------" << std::endl;
    std::cout << "Running Simulation.\n";
    fflush(stdout);
    NS_LOG_INFO("Run Simulation.");
    Simulator::Schedule(Seconds(flowgen_start_time),
                        &stop_simulation_middle);  // check every 100us
    cout << "flowgen_stop_time: " << flowgen_stop_time << std::endl;
    Simulator::Stop(Seconds(flowgen_stop_time + 10.0));
    Simulator::Run();
    /*
        打印flow信息
    */

    PrintFlowMap(true, false, flow_statistics_output_file);
    // PrintFlowMap(true, false, "stdout");         // 也可以输出到标准输出
    /*-----------------------------------------------------------------------------*/
    /*----- we don't need below. Just we can enforce to close this simulation. -----*/
    /*-----------------------------------------------------------------------------*/
    Simulator::Destroy();
    NS_LOG_INFO("Total number of packets: " << RdmaHw::nAllPkts);
    NS_LOG_INFO("Done.");
    endt = clock();
    std::cerr << (double)(endt - begint) / CLOCKS_PER_SEC << "\n";
    std::cerr << "maxSlowDown" << maxSlowDown << std::endl;
}
