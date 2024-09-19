#include <ns3/core-module.h>
#include <ns3/network-module.h>
#include <ns3/internet-module.h>
#include <ns3/point-to-point-module.h>
#include <fstream>
#include <sstream>
#include <queue>
#include <map>
#include <vector>

using namespace ns3;
using namespace std;

struct Flow {
    uint32_t id;
    uint32_t priority; // 新增优先级字段
    uint32_t src;
    uint32_t dst;
    uint32_t size;
    vector<uint32_t> dependFlows;
    vector<uint32_t> invokeFlows;
    bool sent;

    void print() {
        printf("id=%u priority=%u src=%u dst=%u size=%u dep=[", id, priority, src, dst, size);
        for (int i = 0; i < dependFlows.size(); i++) {
            printf("%u ", dependFlows[i]);
        }
        printf("] invoke=[");
        for (int i = 0; i < invokeFlows.size(); i++) {
            printf("%u ", invokeFlows[i]);
        }
        printf("]\n");
    }
};


// key: Flow.id   value: Flow结构体
map<uint32_t, Flow> flowMap;
void PrintFlowMap() {
    cout << "flowMap" << endl;
    for (auto it = flowMap.begin(); it != flowMap.end(); ++it) {
        uint32_t flowId = it->first;
        Flow flow = it->second;
        printf("Flow ID: %u\n", flowId);
        flow.print();  // 调用 Flow 结构中的打印函数
    }
    printf("\n");
}

// key: Flow.id   value: 每条流的依赖数量
map<uint32_t, int> dependencies;
void PrintDependencies() {
    cout << "dependencies" << endl;
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

void ParseFlowFile(string fileName) {
    ifstream file(fileName);
    string line;

    // 读取第一行，获取流的总数量
    getline(file, line);
    int totalFlows = stoi(line);

    uint32_t flow_num = 0;
    // 逐行解析每条流的信息
    while (flow_num < totalFlows) {
        getline(file, line);
        istringstream iss(line);
        Flow flow;
        string dependStr, invokeStr;

        // 解析流ID、源、目的和大小
        iss >> flow.id;
        iss.ignore(2);  // 跳过 ", "
        iss >> flow.priority; // 解析优先级
        iss.ignore(6);  // 跳过 ", src="
        iss >> flow.src;
        iss.ignore(6);  // 跳过 ", dst="
        iss >> flow.dst;
        iss.ignore(7);  // 跳过 ", size="
        iss >> flow.size;
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

        // flow.print();

        flow.sent = false;
        flowMap[flow.id] = flow;
        dependencies[flow.id] = flow.dependFlows.size();

        // 如果没有依赖流，直接加入readyQueue
        if (flow.dependFlows.empty()) {
            readyQueue.push(flow.id);
        }

        flow_num += 1;
    }
}

void FlowStart(uint32_t flowid);
void FlowEnd(uint32_t flowid);
void ScheduleFlows();

void FlowStart(uint32_t flowid) {
    Flow& currentFlow = flowMap[flowid];
    cout << Simulator::Now() << " Sending flow " << currentFlow.id << " : node " << currentFlow.src << " -> node " << currentFlow.dst<< endl;
    // cout << currentFlow.size << endl;
    Simulator::Schedule(NanoSeconds(currentFlow.size * 8), &FlowEnd, flowid);
}

void FlowEnd(uint32_t flowid) {
    Flow& currentFlow = flowMap[flowid];
    // 标记该流已发送
    currentFlow.sent = true;
    cout << Simulator::Now() << " Flow " << currentFlow.id << " send Finish" << endl;
    // 处理 invoke_flow 中的流，将其依赖关系减1
    for (uint32_t invokedFlowId : currentFlow.invokeFlows) {
        dependencies[invokedFlowId]--;
        if (dependencies[invokedFlowId] == 0) {
            readyQueue.push(invokedFlowId);
            cout << Simulator::Now() << " Flow " << invokedFlowId << " becomes Ready." << endl;
        }
    }
    ScheduleFlows();
}

void ScheduleFlows() {
    while (!readyQueue.empty()) {
        uint32_t currentFlowId = readyQueue.front();
        readyQueue.pop();
        Flow& currentFlow = flowMap[currentFlowId];
        FlowStart(currentFlowId);
    }
}

int main(int argc, char *argv[]) {
    // 解析输入文件
    ParseFlowFile("flows.txt");

    // 打印 flowMap, dependencies 和 readyQueue
    PrintFlowMap();
    PrintDependencies();
    PrintReadyQueue();

    // 调度并发送流
    Simulator::Schedule(Seconds(2.0), &ScheduleFlows);

    Simulator::Run();
    Simulator::Destroy();

    return 0;
}
