# -*- coding: utf-8 -*-

import pandas as pd
import argparse
from build_llm_exec_graph import *
from utils import *
from collections import defaultdict
import matplotlib.pyplot as plt
import networkx as nx
import math


tf_layers = None
lid_2_idx_dict = {}         # 用transformer或者mlp的全局唯一的ID，找Transformer对象的五个index :)
vnode_list = []
flow_list = []
flowtype_cnt = {'dp_flow': 0, 'fwd_tp_flow': 0, 'bkwd_tp_flow': 0, 'between_tf_layers': 0, \
                # 'start_mid_layer': 0, 'in_shadow_node': 0, 'cc_flow': 0, \
                'attn_to_mlp_layer': 0, 'mlp_to_attn_layer': 0, \
                'between_microbatch': 0, 'fwd_bkwd_connect': 0, \
                'FWD_Attention_calc' : 0, 'FWD_Mlp_calc' : 0, 'BKWD_Attention_calc' : 0, 'BKWD_Mlp_calc' : 0, }

print_vnode = True          # 生成节点依赖时，务必将其置为True
vnodeid_2_nodeid = {}
nodeid_2_inherent_id = {}
inherent_id_2_NIC_dict = {}       # 做inherent ID到物理网卡上的映射


def vid_to_pid(print_file=None):
    global vnodeid_2_nodeid
    global nodeid_2_inherent_id
    global inherent_id_2_NIC_dict

    id_to_pid_dict = {}
    
    with open(print_file, 'w') as f:
        for nodeid in nodeid_2_inherent_id.keys():
            id_to_pid_dict[nodeid] = inherent_id_2_NIC_dict[nodeid_2_inherent_id[nodeid]]
            if print_file is not None:
                f.write(f"{nodeid} -> {id_to_pid_dict[nodeid]} (inherent_id {nodeid_2_inherent_id[nodeid]})\n")

        for vnodeid in vnodeid_2_nodeid.keys():
            id_to_pid_dict[vnodeid] = inherent_id_2_NIC_dict[nodeid_2_inherent_id[vnodeid_2_nodeid[vnodeid]]]
            if print_file is not None:
                f.write(f"{vnodeid} -> {id_to_pid_dict[vnodeid]} (Layer {vnodeid_2_nodeid[vnodeid]}, inherent_id {nodeid_2_inherent_id[vnodeid_2_nodeid[vnodeid]]})\n")

    return id_to_pid_dict


def parse_arguments():
    parser = argparse.ArgumentParser(description="Transformers Layer Configuration")
    parser.add_argument('--topo', type=str, default='None', help='Topo aware model placement.')
    parser.add_argument('--num_of_layers', type=int, default=3, help='Number of layers in the transformer model (default: 3)')
    parser.add_argument('--global_batch', type=int, default=8192, help='Global batch size (default: 8192)')
    parser.add_argument('--micro_batch', type=int, default=1, help='Micro batch size (default: 1)')
    parser.add_argument('--seq_length', type=int, default=4096, help='Sequence length (default: 4096)')
    parser.add_argument('--pp_cut', type=int, default=-1, help='Degree of compression, usually 0 (minimal) or -1 (no compression) (default: -1)')
    parser.add_argument('--passes', type=int, default=2, help='Number of passes')
    parser.add_argument('--pp', type=int, default=-1, help='Number of stages')
    parser.add_argument('--bypass_first_fwd', type=bool, default=True, help='Bypass first forward pass')
    parser.add_argument('--bypass_last_bkwd', type=bool, default=True, help='Bypass last backward pass')
    parser.add_argument('--enable_ar', type=str, default='True', help='Enable All-Reduce (default True)')

    return parser.parse_args()


# 全局 node ID 分发器
global_lid_register = -1
def get_node_id():
    global global_lid_register
    global_lid_register += 1
    return global_lid_register


# 全局 vnode ID 分发器
global_vlid_register = 99999  # 应当足够大，以免与 global_lid_register 产生重合
def get_vnode_id():
    global global_vlid_register
    global_vlid_register += 1
    return global_vlid_register


# 全局 flow ID 分发器
global_fid_register = -1
def get_flow_id():
    global global_fid_register
    global_fid_register += 1
    return global_fid_register


# 修改函数来处理可能的空格
def get_parameters_by_name(df, name):
    # 打印列名检查
    # 如果有空格或隐藏字符，可以尝试清理列名
    df.columns = df.columns.str.strip()
    # print(df.columns)
    
    # 去掉 name 两边的空格，避免匹配错误
    row = df[df['Name'].str.strip() == name.strip()]
    
    if not row.empty:
        parameters = row.iloc[0].drop('Name').to_dict()
        Parameter_size = parameters['Parameter_size']
        Hidden_size = parameters['Hidden_size']
        Num_of_layers = parameters['Num_of_layers']
        Attention_heads = parameters['Attention_heads']
        Sequence_length = parameters['Sequence_length']
        FFN_hidden_size = parameters['FFN_hidden_size']
        World_size = parameters['World_size']
        TP = parameters['TP']
        PP = parameters['PP']
        DP = int(World_size/(PP*TP))
            # 打印返回的值
        print(f"Name: {name}")
        print(f"Parameter_size: {Parameter_size}")
        print(f"Hidden_size: {Hidden_size}")
        print(f"Num_of_layers: {Num_of_layers}")
        print(f"Attention_heads: {Attention_heads}")
        print(f"Sequence_length: {Sequence_length}")
        print(f"FFN_hidden_size: {FFN_hidden_size}")
        print(f"World_size: {World_size}")
        print(f"TP: {TP}")
        print(f"PP: {PP}")
        print(f"DP: {DP}")

        return Parameter_size, Hidden_size, Num_of_layers, Attention_heads, Sequence_length, FFN_hidden_size, World_size, TP, PP, DP
    else:
        return None, None, None, None, None, None, None, None, None, None   # 如果没有找到，返回 None
    

# Usage: 
# dep = Dep(src=, dst=, depend_node=[], invoke_node=[])
# print(dep)
class Dep:
    def __init__(self, prio=-1, src=-1, dst=-1, lat=0, depend_node=[], invoke_node=[], depend_flow=[], invoke_flow=[], note=''):
        self.id = get_flow_id()
        self.prio=prio
        self.src = src
        self.dst = dst
        self.lat = lat
        self.depend_node = depend_node
        self.invoke_node = invoke_node
        self.depend_flow = depend_flow
        self.invoke_flow = invoke_flow
        self.note = note

    def __repr__(self) -> str:
        global vnodeid_2_nodeid
        global print_vnode
        src = self.src if (print_vnode is True or self.src not in vnodeid_2_nodeid) else vnodeid_2_nodeid[self.src]
        dst = self.dst if (print_vnode is True or self.dst not in vnodeid_2_nodeid) else vnodeid_2_nodeid[self.dst]
        # type = -1 表示 Dep, = 3 表示 Flow
        repr_str = f"{self.id}, {self.prio}, src={src}, dst={dst}, lat={self.lat}, "
        repr_str += f"depend_flow={self.depend_flow}, invoke_flow={self.invoke_flow}, depend_node={self.depend_node}, invoke_node={self.invoke_node}"
        repr_str += f" note=<<{self.note}>>"
        return repr_str


# Usage: 
# flow = Flow(src=, dst=, size=, lat=, depend_flow=[], invoke_flow=[])
# print(flow)
class Flow(Dep):
    def __init__(self, prio=3, src=-1, dst=-1, size=10000, lat=0, depend_node=[], invoke_node=[], depend_flow=[], invoke_flow=[], note=''):
        super().__init__(prio=prio, src=src, dst=dst, lat=lat, depend_node=depend_node, invoke_node=invoke_node, depend_flow=depend_flow, invoke_flow=invoke_flow)
        self.size = size
        self.note = note

    def __repr__(self) -> str:
        global vnodeid_2_nodeid
        global print_vnode
        src = self.src if (print_vnode is True or self.src not in vnodeid_2_nodeid) else vnodeid_2_nodeid[self.src]
        dst = self.dst if (print_vnode is True or self.dst not in vnodeid_2_nodeid) else vnodeid_2_nodeid[self.dst]
        repr_str = f"""{self.id}, {self.prio}, src={src}, dst={dst}, size={self.size}, lat={self.lat}, depend_flow={self.depend_flow}, invoke_flow={self.invoke_flow}, depend_node={self.depend_node}, invoke_node={self.invoke_node}"""
        repr_str += f" note=<<{self.note}>>"
        return repr_str


class SimpleNode:
    def __init__(self, id, lat=0, depend_flow_id=None, invoke_flow_id=None):
        self.id = id
        self.lat=lat
        self.depend_flows = []
        self.invoke_flows = []
        if depend_flow_id is not None:
            self.depend_flows.append(depend_flow_id)
        if invoke_flow_id is not None:
            self.invoke_flows.append(invoke_flow_id)


class Node:
    def __init__(self, layer_start_id, layer_mid_id, layer_end_id, lat=0, name='node'):
        global flow_list
        self.class_name = name
        self.layer_start_id = layer_start_id
        self.layer_mid_id = layer_mid_id
        self.layer_end_id = layer_end_id
        self.flow_dep_id = []
        self.flow_invoke_id = []
        self.lat = lat
        # if name == 'cc_shadow' and layer_start_id != layer_end_id:
        #     flow_list.append(Dep(src=layer_start_id, dst=layer_end_id, lat=0, note=name))
        #     flowtype_cnt['in_shadow_node'] += 1 
        # elif name == 'inner_layer' and layer_start_id != layer_end_id:
        #     flow_list.append(Dep(src=layer_start_id, dst=layer_mid_id, lat=lat, note=name))
        #     flowtype_cnt['start_mid_layer'] += 1 
        # else:
        #     pass

    def print_node(self, file_name=None, end='\n'):
        if file_name is None:
            print(f"{self.class_name}_id=<{self.layer_start_id}, {self.layer_end_id}>{end} ")
            print(f"dep={self.flow_dep_id}{end} ")
            print(f"invoke={self.flow_invoke_id}")
        else:
            with open(file_name, 'w') as f:
                f.write(f"{self.class_name}_id=<{self.layer_start_id}, {self.layer_end_id}>{end} ")
                f.write(f"dep={self.flow_dep_id}{end} ")
                f.write(f"invoke={self.flow_invoke_id}")



class ShadowNode(Node):
    def __init__(self, node_start_id=None, node_end_id=None, origin_node_start_id=None, origin_node_end_id=None, lat=-1):
        global vnodeid_2_nodeid
        assert node_start_id is not None and origin_node_start_id is not None
        assert (node_end_id is not None and origin_node_end_id is not None) or (node_end_id is None and origin_node_end_id is None)
        if node_end_id is None:
            node_end_id = node_start_id
            origin_node_end_id = origin_node_start_id

        super().__init__(layer_start_id=node_start_id, layer_mid_id=-1, layer_end_id=node_end_id, lat=0)
        self.origin_node_start_id = origin_node_start_id
        self.origin_node_end_id = origin_node_end_id

        vnodeid_2_nodeid[self.layer_start_id] = origin_node_start_id
        vnodeid_2_nodeid[self.layer_end_id] = origin_node_end_id

    def __repr__(self):
        # repr_str = 'WARNING: printing method of class ShadowNode is not implemented yet.'
        if self.layer_start_id == self.layer_end_id:
            repr_str = f"node_id=<{self.layer_start_id}> "
            repr_str += f"origin_node_id=<{self.origin_node_start_id}> "
        else:
            repr_str = f"node_id=<{self.layer_start_id}, {self.layer_end_id}> "
            repr_str += f"origin_node_id=<{self.origin_node_start_id}, {self.origin_node_end_id}> "

        repr_str += f"lat={self.lat} "
        repr_str += f"dep={self.flow_dep_id} "
        repr_str += f"invoke={self.flow_invoke_id}"
        return repr_str

    def print_vnode(self, file_name=None, end=' '):
        if file_name is None:
            print(f"node_id=<{self.layer_start_id}, {self.layer_end_id}>{end} ")
            print(f"origin_node_id=<{self.origin_node_start_id}, {self.origin_node_end_id}>{end} ")
            print(f"dep={self.flow_dep_id}{end} ")
            print(f"invoke={self.flow_invoke_id}")
        else:
            with open(file_name, 'w') as f:
                f.write(f"node_id=<{self.layer_start_id}, {self.layer_end_id}>{end} ")
                f.write(f"origin_node_id=<{self.origin_node_start_id}, {self.origin_node_end_id}>{end} ")
                f.write(f"dep={self.flow_dep_id}{end} ")
                f.write(f"invoke={self.flow_invoke_id}")


class Layer(Node):
    def __init__(self, layer_start_id, layer_mid_id, layer_end_id, inherent_id, input_size, output_size, calc_time, param_size, former_layer_id=None, next_layer_id=None):
        super().__init__(layer_start_id, layer_mid_id, layer_end_id, lat=calc_time)
        self.inherent_id = inherent_id
        self.input_size = input_size
        self.output_size = output_size
        self.calc_time = calc_time
        self.param_size = param_size
        self.former_layer_id = former_layer_id
        self.next_layer_id = next_layer_id

        self.tp_grp_start = []
        self.tp_grp_end = []
        self.tp_fwd_type = None
        self.tp_bkwd_type = None

        self.dp_src_grp = []
        self.dp_dst_grp = []
        self.dp_type = None

        self.pp_dep = []
        self.pp_invoke = []

    def __repr__(self):
        return (f"Layer <{self.layer_start_id}, {self.layer_mid_id}, {self.layer_end_id}>: "
                # f"Input_Size = {self.input_size}, Output_Size = {self.output_size}, "
                # f"Calc_Time = {self.calc_time}s, Param_Size = {self.param_size}, "
                f"Former_Layer = {self.former_layer_id}, Next_Layer = {self.next_layer_id}, "
                f"TP_Grp_start = {self.tp_grp_start}, TP_Grp_end = {self.tp_grp_end}, TP_fwd_Type = {self.tp_fwd_type}, TP_bkwd_Type = {self.tp_bkwd_type}, "
                f"DP_src = {self.dp_src_grp}, DP_dst = {self.dp_dst_grp}, DP_Type = {self.dp_type}, "
                f"pp_dep={self.pp_dep}, pp_invoke={self.pp_invoke}")
    
    def set_tp_grp(self, tp_grp_start, tp_grp_end, tp_fwd_type=None, tp_bkwd_type=None):
        self.tp_grp_start = tp_grp_start
        self.tp_grp_end = tp_grp_end
        self.tp_fwd_type = tp_fwd_type if tp_fwd_type else None
        self.tp_bkwd_type = tp_bkwd_type if tp_bkwd_type else None

    def set_dp_grp(self, dp_grp, dp_type):
        self.dp_grp = dp_grp
        self.dp_type = dp_type if dp_grp else None


# 包括QKV层 和输出层：4 * N^2
class Attention(Layer):
    def __init__(self, tf_object, layer_start_id, layer_mid_id, layer_end_id, inherent_id, input_size, output_size, calc_time, param_size, former_layer_id=None, next_layer_id=None):
        super().__init__(layer_start_id, layer_mid_id, layer_end_id, inherent_id, input_size, output_size, calc_time, param_size, former_layer_id, next_layer_id)
        self.layer_type = "Attn"
        self.tf_layer = tf_object

    def __repr__(self):
        repo_str = self.layer_type + ' '
        repo_str += super().__repr__()
        return repo_str


# 包括两个线性层： N -> 4N  &&  4N -> N, N为隐藏层大小。参数量 = (N * 4 * N + 4 * N) + (4 * N * N + N)
class MLP(Layer):
    def __init__(self, tf_object, layer_start_id, layer_mid_id, layer_end_id, inherent_id, input_size, output_size, calc_time, param_size, former_layer_id=None, next_layer_id=None):
        super().__init__(layer_start_id, layer_mid_id, layer_end_id, inherent_id, input_size, output_size, calc_time, param_size, former_layer_id, next_layer_id)
        self.layer_type = "MLP"
        self.tf_layer = tf_object

    def __repr__(self):
        repo_str = self.layer_type + ' '
        repo_str += super().__repr__()
        return repo_str


class TransformerLayer:
    '''
    inherent_id: 用于确定这一层的物理位置
    '''
    def __init__(self, inherent_id, step, pass_type, layer_start_id_attn, layer_mid_id_attn, layer_end_id_attn, layer_start_id_mlp, layer_mid_id_mlp, layer_end_id_mlp, input_size, output_size, \
                 mlp_calc_time, attn_calc_time, mlp_param_size, attn_param_size, former_layer_id=None, next_layer_id=None, did=0, mbs=0, Num_of_layers=1, TP=1, DP=1):
        output_input_size = input_size
        self.inherent_id = inherent_id
        self.step = step
        self.pass_type = pass_type
        self.did = did
        self.mbs = mbs
        self.Num_of_layers = Num_of_layers  # 设置适当的层数
        self.TP = TP             # 设置适当的TP值
        self.DP = DP             # 设置适当的TP值

        self.physical_layer_id = self.extract_ids(pass_type)[4]
        self.physical_tp_id = self.extract_ids(pass_type)[5]
        
        
        if self.pass_type == 'FWD':
            self.attention_layer = Attention(self, layer_start_id=layer_start_id_attn, layer_mid_id=layer_mid_id_attn, layer_end_id=layer_end_id_attn, inherent_id=inherent_id,
                                             input_size=input_size, output_size=output_input_size, calc_time=attn_calc_time, param_size=attn_param_size, 
                                             former_layer_id=former_layer_id, next_layer_id=layer_start_id_mlp)
        
            self.mlp_layer = MLP(self, layer_start_id=layer_start_id_mlp, layer_mid_id=layer_mid_id_mlp, layer_end_id=layer_end_id_mlp, inherent_id=inherent_id, 
                                 input_size=output_input_size, output_size=input_size, calc_time=mlp_calc_time, param_size=mlp_param_size, 
                                 former_layer_id=layer_end_id_attn, next_layer_id=next_layer_id)
            flow_list.append(Dep(src=layer_end_id_attn, dst=layer_start_id_mlp, note=f'FWD_attn_to_mlp_id{self.inherent_id}_dp{self.did}'))
            flowtype_cnt['attn_to_mlp_layer'] += 1
        else:
            self.mlp_layer = MLP(self, layer_start_id=layer_start_id_mlp, layer_mid_id=layer_mid_id_mlp, layer_end_id=layer_end_id_mlp, inherent_id=inherent_id, 
                                 input_size=output_input_size, output_size=input_size, calc_time=mlp_calc_time, param_size=mlp_param_size, 
                                 former_layer_id=former_layer_id, next_layer_id=layer_start_id_attn)
            
            self.attention_layer = Attention(self, layer_start_id=layer_start_id_attn, layer_mid_id=layer_mid_id_attn, layer_end_id=layer_end_id_attn, inherent_id=inherent_id, 
                                             input_size=input_size, output_size=output_input_size, calc_time=attn_calc_time, param_size=attn_param_size, 
                                             former_layer_id=layer_end_id_mlp, next_layer_id=next_layer_id)
            flow_list.append(Dep(src=layer_end_id_mlp, dst=layer_start_id_attn, note=f'BKWD_mlp_to_attn_id{self.inherent_id}_dp{self.did}'))
            flowtype_cnt['mlp_to_attn_layer'] += 1
    def __repr__(self):
        result = self.extract_ids(self.pass_type)
        # return f"Transformer Layer {self.inherent_id} (step={result[0]} pass_type={result[1]} did={result[2]} mbid={result[3]} lid={result[4]} tid={result[5]}):\n{self.attention_layer}\n{self.mlp_layer}"
        
        if self.pass_type == 'FWD':
            return f"FWD Transformer Layer {self.inherent_id} (step, type, did, mbid, lid, tid = {result}):\n{self.attention_layer}\n{self.mlp_layer}"
        else:
            return f"BKWD Transformer Layer {self.inherent_id} (step, type, did, mbid, lid, tid = {result}):\n{self.mlp_layer}\n{self.attention_layer}"
       
    def extract_ids(self, pass_type=None):
        assert pass_type is not None
        if pass_type == 'BKWD':
            inherent_id = (self.DP * self.mbs * self.Num_of_layers * self.TP - 1) - self.inherent_id
        else:
            inherent_id = self.inherent_id
        # 计算 did
        did = inherent_id // (self.mbs * self.Num_of_layers * self.TP)
        # 计算 mbid
        remaining_after_did = inherent_id % (self.mbs * self.Num_of_layers * self.TP)
        mbid = remaining_after_did // (self.Num_of_layers * self.TP)
        # 计算 lid
        remaining_after_mbid = remaining_after_did % (self.Num_of_layers * self.TP)
        lid = remaining_after_mbid // self.TP
        # 计算 tid
        tid = remaining_after_mbid % self.TP
        return self.step, self.pass_type, did, mbid, lid, tid

    def set_tp_groups(self, tp_grp_attn_start, tp_grp_attn_end, tp_grp_mlp_start, tp_grp_mlp_end, tp_fwd_type=None, tp_bkwd_type=None):
        self.attention_layer.set_tp_grp(tp_grp_attn_start, tp_grp_attn_end, tp_fwd_type, tp_bkwd_type)
        self.mlp_layer.set_tp_grp(tp_grp_mlp_start, tp_grp_mlp_end, tp_fwd_type, tp_bkwd_type)
        
    def set_dp_groups(self, dp_grp, dp_type):
        self.mlp_layer.set_dp_grp(dp_grp, dp_type)
        self.attention_layer.set_dp_grp(dp_grp, dp_type)


###################################
###         FUNCTIONS           ###
###################################
def RingAllReduce_1(label, src_list, dst_list, total_data_size, op=None):
    global tf_layers
    global lid_2_idx_dict
    assert op in ['mlp', 'attn']

    N = len(src_list)
    virt_nodes = []
    cc_flows = []

    flow_data_size = total_data_size / N

    for chunk in range(N):
        src_idx = chunk
        dst_idx = (chunk + 2 * N - 2) % N
        for round in range(2 * N - 2):
            src_node_id = src_list[src_idx] if round == 0 else virt_nodes[-1].layer_end_id
            if round == (2 * N - 3):
                dst_node_id = dst_list[dst_idx] 
                # print(f"op = {op}, dst_node_id = {dst_list[dst_idx]}")
            else:
                sidxs = lid_2_idx_dict[src_list[(round + chunk) % N]]  # [step, did, mbid, lid, tid]
                didxs = lid_2_idx_dict[dst_list[(round + chunk + 1) % N]]
                
                slayer = tf_layers[sidxs[0]][sidxs[1]][sidxs[2]][sidxs[3]][sidxs[4]]
                dlayer = tf_layers[didxs[0]][didxs[1]][didxs[2]][didxs[3]][didxs[4]]
                if op == 'mlp':
                    dst_node_start_id = dlayer.mlp_layer.layer_start_id
                    dst_node_mid_id = dlayer.mlp_layer.layer_mid_id
                    dst_node_end_id = dlayer.mlp_layer.layer_end_id
                    calc_time = slayer.mlp_layer.calc_time
                elif op == 'attn':
                    dst_node_start_id = dlayer.attention_layer.layer_start_id
                    dst_node_mid_id = dlayer.attention_layer.layer_mid_id
                    dst_node_end_id = dlayer.attention_layer.layer_end_id
                    calc_time = slayer.attention_layer.calc_time

                # print(f"op = {op}, dst_node_start_id = {dst_node_start_id}, dst_node_end_id = {dst_node_end_id}, dst_list[(round + chunk + 1) % N] = {dst_list[(round + chunk + 1) % N]}")
                assert dst_node_end_id == dst_list[(round + chunk + 1) % N] or dst_node_start_id == dst_list[(round + chunk + 1) % N], \
                    f"{op}  {src_list}, {dst_list}  {dst_node_end_id}, {dst_list[(round + chunk + 1) % N]}"
                if 'DP' in label:
                    virt_nodes.append(ShadowNode(get_vnode_id(), get_vnode_id(), dst_node_start_id, dst_node_end_id, lat=calc_time))
                else:
                    virt_nodes.append(ShadowNode(get_vnode_id(), get_vnode_id(), dst_node_mid_id, dst_node_end_id, lat=calc_time))
                dst_node_id = virt_nodes[-1].layer_start_id

            cc_flows.append(Flow(src=src_node_id, dst=dst_node_id, size=flow_data_size, note=f'{label}, chunk {chunk}, round {round}'))
    # print(f"{virt_nodes}")
    return virt_nodes, cc_flows


def RingAllReduce_2(label, src_list, dst_list, total_data_size, op=None):
    global tf_layers
    global lid_2_idx_dict
    assert op in ['mlp', 'attn']    # actually 不需要这个限制

    N = len(src_list)
    virt_nodes = []
    cc_flows = []
    flow_data_size = total_data_size / N

    for chunk in range(N):
        src_node_id = src_list[chunk]
        for round in range(1, 2 * N - 1):
            src_idx = (chunk + round - 1) % N     # 这两个用来标记当前的 dst vnode 和哪个 src_list / dst_list 对应
            dst_idx = (chunk + round) % N
            vnode = ShadowNode(get_vnode_id(), None, dst_list[dst_idx], None)
            dst_node_id = vnode.layer_start_id
            virt_nodes.append(vnode)

            cc_flows.append(Flow(src=src_node_id, dst=dst_node_id, size=flow_data_size, note=f'{label}, chunk {chunk}, round {round}'))
            if round <= N - 1:
                cc_flows.append(Dep(src=src_list[dst_idx], dst=dst_node_id, note=f'{label}, Reduce-Scatter dependency, chunk {chunk}, round {round}'))
            if round >= N - 1 and round <= 2 * N - 2:
                cc_flows.append(Dep(src=dst_node_id, dst=dst_list[dst_idx], note=f'{label}, All-Gather dependency, chunk {chunk}, round {round}'))
            src_node_id = virt_nodes[-1].layer_end_id
    # print(f"{virt_nodes}")
    return virt_nodes, cc_flows


# 返回值1是class VirtNode对象的列表, 返回值2是 class Flow 的列表
def AllReduce(label, src_list, dst_list, size, method='RingAllReduce', op=None):
    print(f"All_Reduce {label}: {src_list} -> {dst_list}")

    '''
    global tf_layers
    global lid_2_idx_dict

    sidxs = lid_2_idx_dict[src_list[0]]
    didxs = lid_2_idx_dict[dst_list[0]]

    if 'tp' in label.lower():
        # tp的src是节点的start，dst是节点的end
        src_mlp_id = tf_layers[sidxs[0]][sidxs[1]][sidxs[2]][sidxs[3]][sidxs[4]].mlp_layer.layer_mid_id
        dst_mlp_id = tf_layers[didxs[0]][didxs[1]][didxs[2]][didxs[3]][didxs[4]].mlp_layer.layer_end_id
        src_attn_id = tf_layers[sidxs[0]][sidxs[1]][sidxs[2]][sidxs[3]][sidxs[4]].attention_layer.layer_mid_id
        dst_attn_id = tf_layers[didxs[0]][didxs[1]][didxs[2]][didxs[3]][didxs[4]].attention_layer.layer_end_id

        if src_list[0] == src_mlp_id and dst_list[0] == dst_mlp_id:
            op = 'mlp'
        elif src_list[0] == src_attn_id and dst_list[0] == dst_attn_id:
            op = 'attn'
        else:
            print(f"assert {src_list[0]} == {src_mlp_id} and {dst_list[0]} == {dst_mlp_id}")
            print(f"assert {src_list[0]} == {src_attn_id} and {dst_list[0]} == {dst_attn_id}")
            assert False

    elif 'dp' in label.lower():
        # dp的src是节点A的end，dst是节点B的start
        src_mlp_id = tf_layers[sidxs[0]][sidxs[1]][sidxs[2]][sidxs[3]][sidxs[4]].mlp_layer.layer_end_id
        dst_mlp_id = tf_layers[didxs[0]][didxs[1]][didxs[2]][didxs[3]][didxs[4]].mlp_layer.layer_start_id
        src_attn_id = tf_layers[sidxs[0]][sidxs[1]][sidxs[2]][sidxs[3]][sidxs[4]].attention_layer.layer_end_id
        dst_attn_id = tf_layers[didxs[0]][didxs[1]][didxs[2]][didxs[3]][didxs[4]].attention_layer.layer_start_id

        if src_list[0] == src_mlp_id and dst_list[0] == dst_mlp_id:
            op = 'mlp'
        elif src_list[0] == src_attn_id and dst_list[0] == dst_attn_id:
            op = 'attn'
        else:
            print(f"assert {src_list[0]} == {src_mlp_id} and {dst_list[0]} == {dst_mlp_id}")
            print(f"assert {src_list[0]} == {src_attn_id} and {dst_list[0]} == {dst_attn_id}")
            assert False
    else:
        assert False
    '''
    if method == 'RingAllReduce':
        assert len(src_list) > 0
        assert len(dst_list) > 0
        assert len(src_list) == len(dst_list)
        virt_nodes, cc_flows = RingAllReduce_2(label, src_list, dst_list, size, op)
    else:
        assert False, f'Method {method} has not implemented yet!'

    return virt_nodes, cc_flows



def define_inherentId_to_NICId(DP, mbs, Num_of_layers, TP):
    # inherent_id = did * (mbs * Num_of_layers * TP) + mbid * (Num_of_layers * TP) + lid * TP + tid
    global inherent_id_2_NIC_dict       # 做inherent ID到物理网卡上的映射
    layers_per_node = 1
    for did in range(DP):
        for mbid in range(mbs):
            for lid in range(Num_of_layers):
                for tid in range(TP):
                    inherent_id = did * (mbs * Num_of_layers * TP) + mbid * (Num_of_layers * TP) + lid * TP + tid
                    dpid = inherent_id // (mbs * Num_of_layers * TP)
                    inherent_id_2_NIC_dict[inherent_id] = did * (TP * math.ceil(Num_of_layers / layers_per_node)) + (lid // layers_per_node) + tid * math.ceil(Num_of_layers / layers_per_node)
                    print(f"Layer {inherent_id} ({did}, {mbid}, {lid}, {tid}) @ node {inherent_id_2_NIC_dict[inherent_id]}")


def gen_flow_dependency():
    global flow_list
    global tf_layers
    global vnode_list

    def remove_common_elements(list1, list2):
        # 找出两个列表中共同的元素
        common_elements = set(list1) & set(list2)
        
        # 过滤掉共同元素，保持顺序
        filtered_list1 = [item for item in list1 if item not in common_elements]
        filtered_list2 = [item for item in list2 if item not in common_elements]
        
        return filtered_list1, filtered_list2
    
    NodeDependencyDict = defaultdict(SimpleNode)

    for flow in flow_list:
        if flow.src not in NodeDependencyDict:
            NodeDependencyDict[flow.src] = SimpleNode(flow.src, flow.lat, invoke_flow_id=flow.id)
        else:
            NodeDependencyDict[flow.src].invoke_flows.append(flow.id)

        if flow.dst not in NodeDependencyDict:
            NodeDependencyDict[flow.dst] = SimpleNode(flow.dst, flow.lat, depend_flow_id=flow.id)
        else:
            NodeDependencyDict[flow.dst].depend_flows.append(flow.id)
    
    for i in range(len(flow_list)):
        flow_list[i].depend_flow = NodeDependencyDict[flow_list[i].src].depend_flows
        flow_list[i].invoke_flow = NodeDependencyDict[flow_list[i].dst].invoke_flows
        assert len(flow_list[i].depend_flow) == len(set(flow_list[i].depend_flow))
        assert len(flow_list[i].invoke_flow) == len(set(flow_list[i].invoke_flow))

        flow_list[i].depend_flow, flow_list[i].invoke_flow = \
            remove_common_elements(flow_list[i].depend_flow, flow_list[i].invoke_flow)

#######################################
###            M A I N              ###
#######################################
def main():
    global tf_layers
    global lid_2_idx_dict
    global vnode_list
    global flow_list
    global inherent_id_2_NIC_dict       # 做inherent ID到物理网卡上的映射
    global nodeid_2_inherent_id
    global vnodeid_2_nodeid

    args = parse_arguments()
    enable_ar = args.enable_ar.lower() == 'true'

    # 读取 CSV 文件到 DataFrame
    workload_df = pd.read_csv('workload.csv', sep='\t')
    Parameter_size, Hidden_size, Num_of_layers, Attention_heads, Seq_len, FFN_hidden_size, World_size, TP, PP, DP \
        = get_parameters_by_name(workload_df, 'GPT_7B_2')
    # 开始构造流
    if args.pp != -1:
        PP = args.pp

    # Num_of_layers = 3
    # global_batch = 8192
    # micro_batch = 1
    # seq_length = 4096
    # pp_cut = -1    # 压缩的程度。一般来讲应该是 0(最小压缩) 或者 -1 （不压缩）
    passes = args.passes
    bypass_first_fwd = args.bypass_first_fwd
    bypass_last_bkwd = args.bypass_last_bkwd

    # 部分参数更新, for ease of simulation
    Num_of_layers = args.num_of_layers
    global_batch = args.global_batch
    micro_batch = args.micro_batch
    seq_length = args.seq_length
    pp_cut = args.pp_cut    # 压缩的程度。一般来讲应该是 0(最小压缩) 或者 -1 （不压缩）
    
    mbs = global_batch // micro_batch
    assert global_batch == mbs * micro_batch
    
    if pp_cut >= 0:
        mbs = min(global_batch // micro_batch, PP + pp_cut)

    mb_input_size = Seq_len * micro_batch * Hidden_size
    assert FFN_hidden_size == Hidden_size * 4

    mlp_param_size_tp = (Hidden_size * FFN_hidden_size + FFN_hidden_size + FFN_hidden_size * Hidden_size + Hidden_size) / TP
    Head_size = Hidden_size / Attention_heads
    attn_param_size_tp = (3 * (Hidden_size * Head_size * Attention_heads) + Hidden_size * Hidden_size) / TP   #  = 4 * Hidden_size^2 / TP

    steps = passes * 2
    first_step = 1 if bypass_first_fwd else 0
    last_step = 1 if bypass_last_bkwd else 0
    
    tf_layers = [[[[[] for _ in range(Num_of_layers)] for _ in range(mbs)] for _ in range(DP)] for _ in range(steps)]
    total_layer_cnt = 0

    dp_grp_fwd = [[[[] for _ in ['attn', 'mlp']] for _ in range(TP)] for _ in range(Num_of_layers)]  # 对最后一个 mbid 的每一个 Transformer layer，都维护一个 DP 组
    dp_grp_bkwd = [[[[] for _ in ['attn', 'mlp']] for _ in range(TP)] for _ in range(Num_of_layers)]  # 对最后一个 mbid 的每一个 Transformer layer，都维护一个 DP 组

    for step in range(first_step, steps - last_step):
        pass_type = 'FWD' if step % 2 == 0 else 'BKWD'

        if pass_type == 'FWD':
            dp_grp_fwd = [[[[] for _ in ['attn', 'mlp']] for _ in range(TP)] for _ in range(Num_of_layers)]
            for did in range(DP):
                for mbid in range(mbs):
                    # 为每一层构建多个 TP 组
                    for lid in range(Num_of_layers):
                        # 构建并设置 TP 组
                        tp_grp_attn_start = []
                        tp_grp_attn_end = []
                        tp_grp_mlp_start = []
                        tp_grp_mlp_end = []
                        for tid in range(TP):
                            inherent_id = did * (mbs * Num_of_layers * TP) + mbid * (Num_of_layers * TP) + lid * TP + tid
                            lid_start_attn = get_node_id()
                            lid_mid_attn = get_node_id()
                            lid_end_attn = get_node_id()
                            lid_start_mlp = get_node_id()
                            lid_mid_mlp = get_node_id()
                            lid_end_mlp = get_node_id()
                            tp_grp_attn_start.append(lid_mid_attn)
                            tp_grp_attn_end.append(lid_end_attn)
                            tp_grp_mlp_start.append(lid_mid_mlp)
                            tp_grp_mlp_end.append(lid_end_mlp)

                            # 为FWD的最后一个 microbatch 添加 DP 依赖
                            if mbid == mbs - 1:
                                dp_grp_fwd[lid][tid][0].append(lid_start_attn)
                                dp_grp_fwd[lid][tid][1].append(lid_start_mlp)

                            attn_calc_time=0.0007
                            mlp_calc_time=0.001

                            tf_layers[step][did][mbid][lid].append(
                                TransformerLayer(inherent_id=inherent_id, step=step, pass_type=pass_type,
                                            layer_start_id_attn=lid_start_attn, layer_mid_id_attn=lid_mid_attn, layer_end_id_attn=lid_end_attn, 
                                            layer_start_id_mlp=lid_start_mlp, layer_mid_id_mlp=lid_mid_mlp, layer_end_id_mlp=lid_end_mlp, 
                                            input_size=mb_input_size, output_size=mb_input_size, 
                                            attn_calc_time=attn_calc_time, mlp_calc_time=mlp_calc_time, 
                                            mlp_param_size=mlp_param_size_tp, attn_param_size=attn_param_size_tp,
                                            did=did, mbs=mbs, Num_of_layers=Num_of_layers, TP=TP, DP=DP)
                                )
                            flow_list.append(Dep(src=lid_start_attn, dst=lid_mid_attn, lat=attn_calc_time, note=f'layer {inherent_id} Attention FWD calc latency'))     # attn在本层的计算。用来仿真前向计算的时延
                            flow_list.append(Dep(src=lid_start_mlp, dst=lid_mid_mlp, lat=mlp_calc_time, note=f'layer {inherent_id} MLP FWD calc latency'))     # mlp 在本层的计算。用来仿真前向计算的时延
                            flowtype_cnt['FWD_Attention_calc'] += 1
                            flowtype_cnt['FWD_Mlp_calc'] += 1
                            
                            nodeid_2_inherent_id[lid_start_attn] = inherent_id
                            nodeid_2_inherent_id[lid_mid_attn] = inherent_id
                            nodeid_2_inherent_id[lid_end_attn] = inherent_id
                            nodeid_2_inherent_id[lid_start_mlp] = inherent_id
                            nodeid_2_inherent_id[lid_mid_mlp] = inherent_id
                            nodeid_2_inherent_id[lid_end_mlp] = inherent_id
                            
                            total_layer_cnt += 1

                            lid_2_idx_dict[lid_start_attn] = [step, did, mbid, lid, tid]
                            lid_2_idx_dict[lid_mid_attn] = [step, did, mbid, lid, tid]
                            lid_2_idx_dict[lid_end_attn] = [step, did, mbid, lid, tid]
                            lid_2_idx_dict[lid_start_mlp] = [step, did, mbid, lid, tid]
                            lid_2_idx_dict[lid_mid_mlp] = [step, did, mbid, lid, tid]
                            lid_2_idx_dict[lid_end_mlp] = [step, did, mbid, lid, tid]


                        # for all tids in TP Grp
                        for tid in range(TP):
                            tf_layers[step][did][mbid][lid][tid].set_tp_groups(tp_grp_attn_start=tp_grp_attn_start, tp_grp_attn_end=tp_grp_attn_end, \
                                                                               tp_grp_mlp_start=tp_grp_mlp_start, tp_grp_mlp_end=tp_grp_mlp_end, \
                                                                               tp_fwd_type='ALLREDUCE', tp_bkwd_type='ALLREDUCE')
                        if enable_ar == True:
                            attn_vnode, attn_ccflow = AllReduce(f'TP attn FWD ({tp_grp_attn_start}-> {tp_grp_attn_end})', tp_grp_attn_start, tp_grp_attn_end, size=tf_layers[step][did][mbid][lid][-1].attention_layer.param_size, method='RingAllReduce', op='attn')
                            mlp_vnode, mlp_ccflow = AllReduce(f'TP mlp FWD ({tp_grp_mlp_start} -> {tp_grp_mlp_end})', tp_grp_mlp_start, tp_grp_mlp_end, size=tf_layers[step][did][mbid][lid][-1].mlp_layer.param_size, method='RingAllReduce', op='mlp')
                            vnode_list += attn_vnode
                            vnode_list += mlp_vnode
                            flow_list.extend(attn_ccflow)
                            flowtype_cnt['fwd_tp_flow'] += len(attn_ccflow)
                            flow_list.extend(mlp_ccflow)
                            flowtype_cnt['fwd_tp_flow'] += len(mlp_ccflow)
                    
                    # 连接多层Transformer网络
                    for tid in range(TP):
                        for lid in range(1, Num_of_layers):
                            tf_layers[step][did][mbid][lid][tid].attention_layer.former_layer_id = tf_layers[step][did][mbid][lid - 1][tid].mlp_layer.layer_end_id
                            tf_layers[step][did][mbid][lid - 1][tid].mlp_layer.next_layer_id = tf_layers[step][did][mbid][lid][tid].attention_layer.layer_start_id
                            
                            # 添加前后层连接的Flow
                            inherent_id1 = tf_layers[step][did][mbid][lid - 1][tid].inherent_id
                            inherent_id2 = tf_layers[step][did][mbid][lid][tid].inherent_id
                            src_id = tf_layers[step][did][mbid][lid - 1][tid].mlp_layer.layer_end_id
                            dst_id = tf_layers[step][did][mbid][lid][tid].attention_layer.layer_start_id
                            size = tf_layers[step][did][mbid][lid - 1][tid].mlp_layer.output_size
                            lat = tf_layers[step][did][mbid][lid - 1][tid].mlp_layer.calc_time
                            flow_list.append(Flow(src=src_id, dst=dst_id, size=size, lat=0, note=f'layer {inherent_id1} -> layer {inherent_id2}: connect between layers (FWD)'))
                            flowtype_cnt['between_tf_layers'] += 1
                    
                # 添加 PP 依赖
                for mbid in range(1, mbs):     # 每个mb
                    for lid in range(Num_of_layers):  # 每层
                        for tid in range(TP):       # 每个 TP 组
                            for tgrp in range(TP):
                                tf_layers[step][did][mbid-1][lid][tid].mlp_layer.pp_invoke.append(tf_layers[step][did][mbid][lid][tgrp].attention_layer.layer_start_id)
                                tf_layers[step][did][mbid][lid][tid].attention_layer.pp_dep.append(tf_layers[step][did][mbid-1][lid][tgrp].mlp_layer.layer_end_id)
                            # 添加跨不同的mbid的前后层依赖
                            inherent_id1 = tf_layers[step][did][mbid-1][lid][tid].inherent_id
                            inherent_id2 = tf_layers[step][did][mbid][lid][tid].inherent_id
                            src_id = tf_layers[step][did][mbid-1][lid][tid].mlp_layer.layer_end_id
                            dst_id = tf_layers[step][did][mbid][lid][tid].attention_layer.layer_start_id
                            flow_list.append(Dep(src=src_id, dst=dst_id, note=f'layer {inherent_id1} -> layer {inherent_id2}: dep across mbid (fwd)'))
                            flowtype_cnt['between_microbatch'] += 1
                            
            # 最后：TP并行下，每个 TP 部分分别做 DP 的 all-reduce
            for did in range(1):    # 每个DP的dependency都是一样的，为了避免多次生成流，只做第一个DP的
                for lid in range(Num_of_layers):            
                    # if len(dp_grp_bkwd[0]) != 0 and len(dp_grp_fwd[0]) != 0:
                    if step > first_step:
                        for tid in range(TP):       # 每个 TP 部分分别做 DP 的 all-reduce
                            for dgrp in range(DP):
                                tf_layers[step - 1][did][mbs - 1][Num_of_layers - 1 - lid][TP - 1 - tid].attention_layer.dp_dst_grp.insert(0, tf_layers[step][dgrp][0][lid][tid].attention_layer.layer_start_id)
                                tf_layers[step][did][0][lid][tid].attention_layer.dp_src_grp.append(tf_layers[step - 1][dgrp][mbs - 1][Num_of_layers - 1 - lid][TP - 1 - tid].attention_layer.layer_end_id)

                                tf_layers[step - 1][did][mbs - 1][Num_of_layers - 1 - lid][TP - 1 - tid].mlp_layer.dp_dst_grp.insert(0, tf_layers[step][dgrp][0][lid][tid].mlp_layer.layer_start_id)
                                tf_layers[step][did][0][lid][tid].mlp_layer.dp_src_grp.append(tf_layers[step - 1][dgrp][mbs - 1][Num_of_layers - 1 - lid][TP - 1 - tid].mlp_layer.layer_end_id)

                            tf_layers[step - 1][did][mbs - 1][Num_of_layers - 1 - lid][TP - 1 - tid].attention_layer.dp_src_grp = tf_layers[step][did][0][lid][tid].attention_layer.dp_src_grp
                            tf_layers[step][did][0][lid][tid].attention_layer.dp_dst_grp = tf_layers[step - 1][did][mbs - 1][Num_of_layers - 1 - lid][TP - 1 - tid].attention_layer.dp_dst_grp
                            tf_layers[step - 1][did][mbs - 1][Num_of_layers - 1 - lid][TP - 1 - tid].mlp_layer.dp_src_grp = tf_layers[step][did][0][lid][tid].mlp_layer.dp_src_grp
                            tf_layers[step][did][0][lid][tid].mlp_layer.dp_dst_grp = tf_layers[step - 1][did][mbs - 1][Num_of_layers - 1 - lid][TP - 1 - tid].mlp_layer.dp_dst_grp

                            tf_layers[step - 1][did][mbs - 1][Num_of_layers - 1 - lid][TP - 1 - tid].attention_layer.dp_type = 'ALLREDUCE'
                            tf_layers[step][did][0][lid][tid].attention_layer.dp_type = 'ALLREDUCE'
                            tf_layers[step - 1][did][mbs - 1][Num_of_layers - 1 - lid][TP - 1 - tid].mlp_layer.dp_type = 'ALLREDUCE'
                            tf_layers[step][did][0][lid][tid].mlp_layer.dp_type = 'ALLREDUCE'

                            if enable_ar == True:
                                attn_vnode, attn_ccflow = AllReduce(f'DP attn ({tf_layers[step][did][0][lid][tid].attention_layer.dp_src_grp} -> {tf_layers[step][did][0][lid][tid].attention_layer.dp_dst_grp})', 
                                                                tf_layers[step][did][0][lid][tid].attention_layer.dp_src_grp, tf_layers[step][did][0][lid][tid].attention_layer.dp_dst_grp, \
                                                                size=tf_layers[step][did][0][lid][tid].attention_layer.param_size, method='RingAllReduce', op='attn')
                                mlp_vnode, mlp_ccflow = AllReduce(f'DP mlp ({tf_layers[step][did][0][lid][tid].mlp_layer.dp_src_grp} -> {tf_layers[step][did][0][lid][tid].mlp_layer.dp_dst_grp})',
                                                                tf_layers[step][did][0][lid][tid].mlp_layer.dp_src_grp, tf_layers[step][did][0][lid][tid].mlp_layer.dp_dst_grp, \
                                                                size=tf_layers[step][did][0][lid][tid].mlp_layer.param_size, method='RingAllReduce', op='mlp')
                                vnode_list += attn_vnode
                                vnode_list += mlp_vnode
                                flow_list.extend(attn_ccflow)
                                flowtype_cnt['dp_flow'] += len(attn_ccflow)
                                flow_list.extend(mlp_ccflow)
                                flowtype_cnt['dp_flow'] += len(mlp_ccflow)
                    else:
                        pass



        else:       # BKWD
            dp_grp_bkwd = [[[[] for _ in ['attn', 'mlp']] for _ in range(TP)] for _ in range(Num_of_layers)]
            for did in range(DP):
                for mbid in range(mbs):
                    # 为每一层构建多个 TP 组
                    for lid in range(Num_of_layers):
                        # 构建并设置 TP 组
                        tp_grp_attn_start = []
                        tp_grp_attn_end = []
                        tp_grp_mlp_start = []
                        tp_grp_mlp_end = []
                        for tid in range(TP):
                            # inherent_id = did * (mbs * Num_of_layers * TP) + mbid * (Num_of_layers * TP) + lid * TP + tid
                            inherent_id = (DP - 1 - did) * (mbs * Num_of_layers * TP) + (mbs - 1 - mbid) * (Num_of_layers * TP) + (Num_of_layers - 1 - lid) * TP + (TP - 1 - tid)
                            lid_start_mlp = get_node_id()
                            lid_mid_mlp = get_node_id()
                            lid_end_mlp = get_node_id()
                            lid_start_attn = get_node_id()
                            lid_mid_attn = get_node_id()
                            lid_end_attn = get_node_id()
                            
                            tp_grp_attn_start.append(lid_mid_attn)
                            tp_grp_attn_end.append(lid_end_attn)
                            tp_grp_mlp_start.append(lid_mid_mlp)
                            tp_grp_mlp_end.append(lid_end_mlp)

                            # 为 BKWD 的第一个 microbatch 添加 DP 依赖
                            if mbid == 0:
                                dp_grp_bkwd[lid][tid][0].append(lid_end_attn)
                                dp_grp_bkwd[lid][tid][1].append(lid_end_mlp)

                            attn_calc_time=0.001     # 反向传播计算时间是正向传播的两倍
                            mlp_calc_time=0.002

                            tf_layers[step][did][mbid][lid].append(
                                TransformerLayer(inherent_id=inherent_id, step=step, pass_type=pass_type,
                                            layer_start_id_attn=lid_start_attn, layer_mid_id_attn=lid_mid_attn, layer_end_id_attn=lid_end_attn, 
                                            layer_start_id_mlp=lid_start_mlp, layer_mid_id_mlp=lid_mid_mlp, layer_end_id_mlp=lid_end_mlp, 
                                            input_size=mb_input_size, output_size=mb_input_size, 
                                            attn_calc_time=attn_calc_time, mlp_calc_time=mlp_calc_time,       
                                            mlp_param_size=mlp_param_size_tp, attn_param_size=attn_param_size_tp,
                                            did=did, mbs=mbs, Num_of_layers=Num_of_layers, TP=TP, DP=DP)
                                )

                            flow_list.append(Dep(src=lid_start_attn, dst=lid_mid_attn, lat=attn_calc_time, note=f'layer {inherent_id} Attention BKWD calc latency'))     # attn在本层的计算。用来仿真前向计算的时延
                            flow_list.append(Dep(src=lid_start_mlp, dst=lid_mid_mlp, lat=mlp_calc_time, note=f'layer {inherent_id} MLP BKWD calc latency'))     # mlp 在本层的计算。用来仿真前向计算的时延
                            flowtype_cnt['BKWD_Attention_calc'] += 1
                            flowtype_cnt['BKWD_Mlp_calc'] += 1

                            nodeid_2_inherent_id[lid_start_attn] = inherent_id
                            nodeid_2_inherent_id[lid_mid_attn] = inherent_id
                            nodeid_2_inherent_id[lid_end_attn] = inherent_id
                            nodeid_2_inherent_id[lid_start_mlp] = inherent_id
                            nodeid_2_inherent_id[lid_mid_mlp] = inherent_id
                            nodeid_2_inherent_id[lid_end_mlp] = inherent_id

                            total_layer_cnt += 1

                            lid_2_idx_dict[lid_start_attn] = [step, did, mbid, lid, tid]
                            lid_2_idx_dict[lid_mid_attn] = [step, did, mbid, lid, tid]
                            lid_2_idx_dict[lid_end_attn] = [step, did, mbid, lid, tid]
                            lid_2_idx_dict[lid_start_mlp] = [step, did, mbid, lid, tid]
                            lid_2_idx_dict[lid_mid_mlp] = [step, did, mbid, lid, tid]
                            lid_2_idx_dict[lid_end_mlp] = [step, did, mbid, lid, tid]
                            
                        for tid in range(TP):
                            tf_layers[step][did][mbid][lid][tid].set_tp_groups(tp_grp_attn_start=tp_grp_attn_start, tp_grp_attn_end=tp_grp_attn_end, \
                                                                               tp_grp_mlp_start=tp_grp_mlp_start, tp_grp_mlp_end=tp_grp_mlp_end, \
                                                                               tp_fwd_type='ALLREDUCE', tp_bkwd_type='ALLREDUCE')
                            
                        # print(f'layer {lid}: {tp_grp_mlp_start}, {tp_grp_mlp_end}')
                        if enable_ar == True:
                            mlp_vnode, mlp_ccflow = AllReduce(f"TP mlp BKWD ({tp_grp_mlp_start} -> {tp_grp_mlp_end})", tp_grp_mlp_start, tp_grp_mlp_end, size=tf_layers[step][did][mbid][lid][-1].mlp_layer.param_size, method='RingAllReduce', op='mlp')
                            attn_vnode, attn_ccflow = AllReduce(f'TP attn BKWD ({tp_grp_attn_start} -> {tp_grp_attn_end})', tp_grp_attn_start, tp_grp_attn_end, size=tf_layers[step][did][mbid][lid][-1].attention_layer.param_size, method='RingAllReduce', op='attn')
                            vnode_list += mlp_vnode
                            vnode_list += attn_vnode
                            flow_list.extend(mlp_ccflow)
                            flowtype_cnt['bkwd_tp_flow'] += len(mlp_ccflow)
                            flow_list.extend(attn_ccflow)
                            flowtype_cnt['bkwd_tp_flow'] += len(attn_ccflow)

                        
                    # 连接多层Transformer网络 (反向传播)
                    for tid in range(TP):
                        for lid in range(1, Num_of_layers):
                            tf_layers[step][did][mbid][lid][tid].mlp_layer.former_layer_id = tf_layers[step][did][mbid][lid - 1][tid].attention_layer.layer_end_id
                            tf_layers[step][did][mbid][lid - 1][tid].attention_layer.next_layer_id = tf_layers[step][did][mbid][lid][tid].mlp_layer.layer_start_id
                            
                            # 添加前后层连接的Flow
                            inherent_id1 = tf_layers[step][did][mbid][lid - 1][tid].inherent_id
                            inherent_id2 = tf_layers[step][did][mbid][lid][tid].inherent_id
                            src_id = tf_layers[step][did][mbid][lid - 1][tid].attention_layer.layer_end_id
                            dst_id = tf_layers[step][did][mbid][lid][tid].mlp_layer.layer_start_id
                            size = tf_layers[step][did][mbid][lid - 1][tid].attention_layer.output_size
                            lat = tf_layers[step][did][mbid][lid - 1][tid].attention_layer.calc_time
                            flow_list.append(Flow(src=src_id, dst=dst_id, size=size, lat=0, depend_flow=[], invoke_flow=[], note=f'layer {inherent_id1} -> layer {inherent_id2}: connect between layers (BKWD)'))
                            flowtype_cnt['between_tf_layers'] += 1

                            
                    # 连接前向和反向传播的代码。这里使用PP进行连接。note: 这里是dependency而非flow
                    if mbid == 0:
                        if step - 1 >= first_step:
                            for tid in range(TP):
                                # X 是复制的，所以不需要group操作
                                tf_layers[step][did][mbid][0][tid].mlp_layer.pp_dep.append(tf_layers[step - 1][did][mbs - 1][Num_of_layers - 1][tid].mlp_layer.layer_end_id)
                                tf_layers[step - 1][did][mbs - 1][Num_of_layers - 1][tid].mlp_layer.pp_invoke.append(tf_layers[step][did][mbid][0][tid].mlp_layer.layer_start_id)
                                # for tgrp in range(TP):
                                #     tf_layers[step][did][mbid][0][tid].mlp_layer.pp_dep.append(tf_layers[step - 1][did][mbs - 1][Num_of_layers - 1][tgrp].mlp_layer.layer_end_id)
                                #     tf_layers[step - 1][did][mbs - 1][Num_of_layers - 1][tid].mlp_layer.pp_invoke.append(tf_layers[step][did][mbid][0][tgrp].mlp_layer.layer_start_id)
  
                                # 添加前后层连接的Flow
                                inherent_id1 = tf_layers[step - 1][did][mbs - 1][Num_of_layers - 1][tid].inherent_id
                                inherent_id2 = tf_layers[step][did][mbid][0][tid].inherent_id
                                src_id = tf_layers[step - 1][did][mbs - 1][Num_of_layers - 1][tid].mlp_layer.layer_end_id
                                dst_id = tf_layers[step][did][mbid][0][tid].mlp_layer.layer_start_id
                                size = tf_layers[step - 1][did][mbs - 1][Num_of_layers - 1][tid].mlp_layer.output_size
                                lat = tf_layers[step - 1][did][mbs - 1][Num_of_layers - 1][tid].mlp_layer.calc_time
                                flow_list.append(Flow(src=src_id, dst=dst_id, size=size, lat=0, depend_flow=[], invoke_flow=[], note=f'layer {inherent_id1} -> layer {inherent_id2}: connect between FWD & BKWD'))
                                flowtype_cnt['fwd_bkwd_connect'] += 1

                # 添加 PP 依赖
                for mbid in range(1, mbs):     # 每个mb
                    for lid in range(Num_of_layers):  # 每层
                        for tid in range(TP):       # 每个 TP 组
                            for tgrp in range(TP):
                                tf_layers[step][did][mbid-1][lid][tid].attention_layer.pp_invoke.append(tf_layers[step][did][mbid][lid][tgrp].mlp_layer.layer_start_id)
                                tf_layers[step][did][mbid][lid][tid].mlp_layer.pp_dep.append(tf_layers[step][did][mbid-1][lid][tgrp].attention_layer.layer_end_id)
                            # 添加跨不同的mbid的前后层依赖
                            inherent_id1 = tf_layers[step][did][mbid-1][lid][tid].inherent_id
                            inherent_id2 = tf_layers[step][did][mbid][lid][tid].inherent_id
                            src_id = tf_layers[step][did][mbid-1][lid][tid].attention_layer.layer_end_id
                            dst_id = tf_layers[step][did][mbid][lid][tid].mlp_layer.layer_start_id
                            flow_list.append(Dep(src=src_id, dst=dst_id, note=f'layer {inherent_id1} -> layer {inherent_id2}: dep across mbid (bkwd)'))
                            flowtype_cnt['between_microbatch'] += 1

    print(f'\npasses: {list(range(first_step, steps - last_step))}')                
    print(f'new_DP: {DP}')
    print(f'new_mbs: {mbs}')
    print(f'new_Num_of_layers: {Num_of_layers}')
    print(f'new_TP: {TP}')
    print(f'total_layers: {total_layer_cnt}')

    return DP, mbs, Num_of_layers, TP


def print_details():
    global tf_layers
    global vnode_list
    global flow_list
    # node
    with open('mix/llm_node.txt', 'w') as f:
        for step in range(len(tf_layers)):
            for dp in range(len(tf_layers[step])):
                for mb in range(len(tf_layers[step][dp])):
                    for layer in range(len(tf_layers[step][dp][mb])):
                        for tid in range(len(tf_layers[step][dp][mb][layer])):
                            f.write(f"\n\n{tf_layers[step][dp][mb][layer][tid]}")
    
    # vnode
    with open('mix/llm_vnode.txt', 'w') as f:
        for vnode in vnode_list:
            f.write(repr(vnode) + '\n')

    # Flow and Dep
    with open('mix/llm_flow.txt', 'w') as f:
        f.write(str(len(flow_list)) + '\n')
        for flow in flow_list:
            f.write(repr(flow) + '\n')

    print(flowtype_cnt)
    total_sum = sum(flowtype_cnt.values())
    print(f"total_flows = {total_sum}")

def read_flows(filename):
    edges = []
    elliminate_shadow = False
    with open(filename, 'r') as file:
        first_line = file.readline()
        flow_num = int(first_line)
        print(f'Read flow: number = {flow_num}')
        for line in file:
            parts = line.strip().split(',')
            edge_id = parts[0].strip()  # 获取边编号
            second_column_value = int(parts[1].strip())  # 获取第二列的值
            src = parts[2].split('=')[1].strip()  # 获取起始节点编号
            dst = parts[3].split('=')[1].strip()  # 获取终止节点编号
            src_num = int(src)
            dst_num = int(dst)
            if elliminate_shadow == True:
                if src_num in vnodeid_2_nodeid:
                    src = '(' + str(vnodeid_2_nodeid[src_num]) + ')'
                if dst_num in vnodeid_2_nodeid:
                    dst = '(' + str(vnodeid_2_nodeid[dst_num]) + ')'
                edges.append((src, dst, edge_id, second_column_value))
            else:
                edges.append((src, dst, edge_id, second_column_value))
    return edges

def create_graph(edges):
    G = nx.DiGraph()
    for src, dst, edge_id, second_column_value in edges:
        # 根据第二列的值设置边的颜色和样式
        if second_column_value == 3:
            color = "black"
            style = "solid"
        elif second_column_value == -1:
            color = "lightgray"
            style = "dashed"
        
        # 添加带属性的边
        G.add_edge(src, dst, label=edge_id, color=color, style=style)
    return G

def flip_positions(pos):
    """上下翻转节点的位置，通过翻转y坐标"""
    flipped_pos = {}
    for node, (x, y) in pos.items():
        flipped_pos[node] = (x, -y)  # 通过将y坐标取反实现上下翻转
    return flipped_pos

def draw_graph(G):
    # 使用 graphviz 的 'dot' 布局，带 rankdir 参数实现从左到右排列
    pos = nx.nx_agraph.graphviz_layout(G, prog="dot")  #, args='-Grankdir=LR')

    # 上下翻转节点位置
    # pos = flip_positions(pos)

    # 创建图形和轴
    fig, ax = plt.subplots(figsize=(20, 15))  # 增大图形尺寸

    # 绘制节点
    nx.draw(G, pos, ax=ax, with_labels=True, node_color='lightblue', 
            node_size=200, font_size=7, font_weight='bold', arrows=True)

    # 获取边的颜色和样式
    edges = G.edges(data=True)
    edge_colors = [attr['color'] for _, _, attr in edges]
    edge_styles = [attr['style'] for _, _, attr in edges]

    # 绘制边
    for edge, color, style in zip(edges, edge_colors, edge_styles):
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=[edge], edge_color=color, 
                               style=style, arrows=True)

    # 绘制边的标签
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, ax=ax, edge_labels=edge_labels, 
                                 font_color='red', font_size=7)  # 减小字体避免重叠
    
    plt.title("Directed Graph of Flows (Flipped & Rotated 90°)")
    plt.savefig("flows_graph_flipped_rotated.pdf", format="pdf")  # 保存为PDF
    plt.show()
    plt.close()  # 关闭图形

if __name__ == '__main__':

    filename = '../config/topo_dumbbell_4dcilink_2dci_4core_4tor_16host_100Gbps_100Gbps_100Gbps.txt'
    switch_ids, host_ids = parse_ids(filename)
    
    DP, mbs, Num_of_layers, TP = main()
    # define_inherentId_to_NICId(2, 2, 3, 2)  # in a mocked case: (2, 2, 3, 2)
    define_inherentId_to_NICId(DP, mbs, Num_of_layers, TP)  # in a mocked case: (2, 2, 3, 2)
    gen_flow_dependency()
    print_details()
    node_2_pid = vid_to_pid('mix/node_mapping.txt')

    filename = 'mix/llm_flow.txt'
    edges = read_flows(filename)
    G = create_graph(edges)
    draw_graph(G)

    srcs = [
        [0,1],
        [0,1,2],
        [0,1,2,3],
        [0,1,2,3,4],
    ]

    dsts = [
        [100,101],
        [100,101,102],
        [100,101,102,103],
        [100,101,102,103,104],
    ]

    assert len(srcs) == len(dsts)
    for i in range(len(srcs)):
        assert len(srcs[i]) == len(dsts[i])
        v_nodes, v_flows = RingAllReduce_2('label', srcs[i], dsts[i], 30, op='mlp')
        # 大小为i的allreduce，有i*(2*i-2)个vnode，i*(2*i-2)个flow和 i*(2*i-1)个Dep
        print(f"AllReduce of size {len(srcs[i])} has {len(v_nodes)} new nodes and {len(v_flows)} Deps and Flows")

    # for vnode in v_nodes:
    #     print(vnode)
    # for vflow in v_flows:
    #     print(vflow)
    