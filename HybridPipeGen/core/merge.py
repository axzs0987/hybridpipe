"""
input: Graph, Code, Operations
output: new code -> run result
"""
# step 1 : Add one operation on one Edge.
"""
Input: Graph, Code, Operation, Edge
output: new code
"""

from HybridPipeGen.core.read_ipynb import ipynb2py
import json
import pickle
from HybridPipeGen.core.operation_code import OperationCode, OperationType, EdgeOperationType
from HybridPipeGen.core.NotebookGraph import NotebookGraph, Node, Edge
import os
import pprint
import copy
import time
from func_timeout import func_set_timeout
import func_timeout

import numpy as np

cleaning = ["confusion_matrix", "sum", "remove", "drop", "unstack", "reshape", "replace", "drop_duplicates", "groupby", "merge", "reset_index", "join", "sort_values", "concat"]
fit_transform_ope = ["fit_transform", "transform","fit"]


res_data_path = 'hi_res_data/'
val_res_data_path = 'hi_val_res_data/'

def get_all_aiseq():
    seqs = [
    ['Scaler', 'FeatureEngine_simple', 'FeatureSelection'],
    ['FeatureEngine_simple', 'FeatureSelection', 'Scaler'],
    ['FeatureSelection', 'Scaler', 'FeatureEngine_simple'],
    ['FeatureSelection', 'FeatureEngine_simple', 'Scaler'],
    ]
    ai_sequences = []
    for seq in seqs:
        temp_list = []
        for index, lop in enumerate(seq):
            for pop in OperationType[lop]:
                if index == 0:
                    temp_list.append([pop])
                else:
                    for index1,temp_seq in enumerate(temp_list1):
                        temp_seq1 = temp_seq.copy()
                        temp_seq1.append(pop)
                        temp_list.append(temp_seq1)
            temp_list1 = temp_list.copy()
        ai_sequences += temp_list
    # pprint.p#print(ai_sequences)
    ai_sequences_new = [i for i in ai_sequences if len(i) ==3]
    # #print(len(ai_sequences_new))
    return ai_sequences_new

class Step():
    def __init__(self, operator, edge_id):
        self.operator = operator
        self.edge_id = edge_id

    
class Merger():
    def __init__(self, running_id):
        self.code = ''
        self.origin_code = ''
        self.operations = []
        self.graph = None
        self.all_seq_num_dict = {}
        self.cuted_seq_num_dict = {}
        self.running_id = running_id

    def load_origin_code(self, notebook_id):
        filepath = "HybridPipeGen/core/tmpdata/prenotebook_code/" + str(notebook_id) + '.py'
        self.file_path = filepath
        output_path = filepath
        with open(output_path, 'r') as src_file:
            self.origin_code = src_file.read()
        self.code = self.origin_code
        # #print('load_origin_code', self.code.split('\n')[49])
        
    def load_operations(self, notebook_id):
        # filepath = "deepline_only_new_"+str(self.running_id)+"/" + str(notebook_id) + "_seq.json"
        filepath = "deepline_only_tmpdata"+ "/" + str(notebook_id) + "_seq.json"
        self.operations = []
        with open(filepath, 'r') as f:
            operations = json.load(f)
        self.can_add_operaiton = []
        for ope_type in OperationType:
            self.can_add_operaiton += OperationType[ope_type]
        for item in operations:
            if item != -1 and item != 'BLANK' and item != 'FINISH' and item in self.can_add_operaiton:
                self.operations.append(item)

    def load_planB_operations(self, notebook_id):
        self.can_add_operaiton = []
        for ope_type in OperationType:
            self.can_add_operaiton += OperationType[ope_type]
        with open('clean_task_no_1_fix_label_s.json','r') as f:
            clean_data = json.load(f)
        for task in clean_data:
            for nid in clean_data[task]['notebook_list']:
                if nid == notebook_id:
                    found_task  = task
                    break
        #print(found_task)
        with open('planB_cross.json','r') as f:
            planB_test = json.load(f)
        if found_task not in planB_test.keys():
            return 'need_wait'
        best_seq_id = list(planB_test[found_task].keys())[0]
        self.operations = get_all_aiseq()[int(best_seq_id)]

    def load_rl_operations(self, notebook_id):
        self.can_add_operaiton = []
        for ope_type in OperationType:
            self.can_add_operaiton += OperationType[ope_type]
        with open("clean_task_rl_200.json",'r') as f:
            clean_task = json.load(f)
        with open("firefly_seq_56000.json",'r') as f:
            firefly_seq = json.load(f)
        for task in clean_task:
            for nid in clean_task[task]['notebook_list']:
                if nid == notebook_id:
                    found_task = task
        self.operations = firefly_seq[found_task]
    def load_one_rl_operation(self, notebook_id):
        self.can_add_operaiton = []
        for ope_type in OperationType:
            self.can_add_operaiton += OperationType[ope_type]
        ops = []
        for ope in self.operations:
            if ope in self.can_add_operaiton:
                ops.append(ope)
        self.operations = ops
        # with open("clean_task_rl_200.json",'r') as f:
        #     clean_task = json.load(f)
        # with open("HybridPipeGen/core/firefly_seq_56000.json",'r') as f:
        #     firefly_seq = json.load(f)
        # with open("HybridPipeGen/core/merged_dataset_label.json",'r') as f:
        #     dataset_label = json.load(f)
        # info_triple = np.load('HybridPipeGen/core/merged_info_triple.npy', allow_pickle=True).item()
        # found_task = dataset_label[notebook_id]['dataset']+'_'+info_triple[notebook_id]['model_type']+'_'+str(dataset_label[notebook_id]['index'][0])
        # # for task in clean_task:
        # #     for nid in clean_task[task]['notebook_list']:
        # #         if nid == notebook_id:
        # #             found_task = task
        # self.operations = firefly_seq[found_task]
    def load_graph(self, notebook_id):     
        if os.path.exists('HybridPipeGen/core/tmpdata/prenotebook_graph/'+str(notebook_id)+".pkl"):
            filepath = 'HybridPipeGen/core/tmpdata/prenotebook_graph/'+str(notebook_id)+".pkl"
        else:
            filepath = 'HybridPipeGen/core/tmpdata/HybridPipeGen/core/tmpdata/prenotebook_graph/'+str(notebook_id)+".pkl"
        with open(filepath, 'rb') as f:
            self.graph = pickle.load(f)  

    def print_code(self):
        codelist = self.code.split("\n")
        # for index,line in enumerate(codelist):
        #     print(index, line)

    def print_one_seq(self,seq):
        str_ = ''
        for item in seq:
            str_ += str((item.operator, item.edge_id))
            str_ += ','
        #print(str_)

    def print_all_seq(self):
        for seq in self.all_seq:
            self.print_one_seq(seq)

    def enum_adding(self, notebook_id):
        def is_all_in(ope_id, seq):
            res = True
            need_operations = set()
            for index,ope in enumerate(self.operations):
                if index == ope_id:
                    break
                need_operations.add(ope)
            has_operations = set()
            for item in seq:
                has_operations.add(item.operator)
            if len(has_operations) < len(need_operations):
                return False
            else:
                return True
        #print("###############")
        #print('notebook_id', notebook_id)
        
        self.load_graph(notebook_id)
        self.load_operations(notebook_id)
        #print('len(self.operations)', len(self.operations))
        self.need_add_edge_id = []
        added_sorted_id = []
        
        for edge in self.graph.result_edges:
            if edge.edge_type == 1 and edge.sorted_id not in added_sorted_id and 'sum' not in edge.func_name:
                self.need_add_edge_id.append(edge.edge_id)
                added_sorted_id.append(edge.sorted_id)
        #print('len(self.operations)', len(self.operations))
        #print('len(need_add_edge_id)', len(self.need_add_edge_id))
    
        self.all_seq = []
        if len(self.need_add_edge_id) != 0:
            self.all_seq = [[]]
            for ope_index,operation in enumerate(self.operations):
                new_seq = copy.copy(self.all_seq)
                # #print('ope_index', ope_index)
                for edge_index in self.need_add_edge_id:
                    # #print('edge_index', edge_index)
                    for position in ['before', 'after']:
                        step = Step(operation, str(edge_index)+"---"+position)
                        for seq in self.all_seq:
                            last_step_id = 0
                            last_step_position = 'before'
        
                            if len(seq) != 0:
                                last_step = seq[-1]
                                last_step_id, last_step_position = last_step.edge_id.split("---")
                                last_step_id = int(last_step_id)
                            if edge_index < last_step_id: # 该图能添加的边已经大于现有的边
                                continue
                            if last_step_position == 'after' and position == 'before':
                                continue
                            if not is_all_in(ope_index, seq):
                                continue
                            temp_seq = copy.copy(seq)
                            temp_seq.append(step)
                            new_seq.append(temp_seq)
                    
                self.all_seq = new_seq
        if len(self.all_seq) != 0:
            if len(self.all_seq[0]) == 0:
                self.all_seq.pop(0)
        # else:
        seq = []
        # #print('self.operations', self.operations)
        for operation in self.operations:
            step = Step(operation, 'end')
            seq.append(step)
            # #print("seq",seq)
            temp = copy.copy(seq)
            # #print('temp',temp)
            self.all_seq.append(temp)

        if len(self.all_seq) not in self.all_seq_num_dict:
            self.all_seq_num_dict[len(self.all_seq)] = 0
        self.all_seq_num_dict[len(self.all_seq)] += 1

        #print('len(self.all_seq)', len(self.all_seq))
        # #print(self.all_seq)

    def enum_adding_rl(self, notebook_id):
        def is_all_in(ope_id, seq):
            res = True
            need_operations = set()
            for index,ope in enumerate(self.operations):
                if index == ope_id:
                    break
                need_operations.add(ope)
            has_operations = set()
            for item in seq:
                has_operations.add(item.operator)
            if len(has_operations) < len(need_operations):
                return False
            else:
                return True
        # #print("###############")
        # #print('notebook_id', notebook_id)
        
        self.load_graph(notebook_id)
        # res = self.load_rl_operations(notebook_id)
        
        res = self.load_one_rl_operation(notebook_id)
        
        # #print('len(self.operations)', len(self.operations))
        self.need_add_edge_id = []
        added_sorted_id = []
        
        for edge in self.graph.result_edges:
            if edge.edge_type == 1 and edge.sorted_id not in added_sorted_id and 'sum' not in edge.func_name:
                self.need_add_edge_id.append(edge.edge_id)
                added_sorted_id.append(edge.sorted_id)
        # #print('len(self.operations)', len(self.operations))
        # #print('len(need_add_edge_id)', len(self.need_add_edge_id))
    
        self.all_seq = []
        if len(self.need_add_edge_id) != 0:
            self.all_seq = [[]]
            for ope_index,operation in enumerate(self.operations):
                new_seq = copy.copy(self.all_seq)
                # #print('ope_index', ope_index)
                for edge_index in self.need_add_edge_id:
                    # #print('edge_index', edge_index)
                    for position in ['before', 'after']:
                        step = Step(operation, str(edge_index)+"---"+position)
                        for seq in self.all_seq:
                            last_step_id = 0
                            last_step_position = 'before'
        
                            if len(seq) != 0:
                                last_step = seq[-1]
                                last_step_id, last_step_position = last_step.edge_id.split("---")
                                last_step_id = int(last_step_id)
                            if edge_index < last_step_id: # 该图能添加的边已经大于现有的边
                                continue
                            if last_step_position == 'after' and position == 'before':
                                continue
                            # if not is_all_in(ope_index, seq):
                                # continue
                            temp_seq = copy.copy(seq)
                            temp_seq.append(step)
                            new_seq.append(temp_seq)
                    
                self.all_seq = new_seq
        if len(self.all_seq) != 0:
            if len(self.all_seq[0]) == 0:
                self.all_seq.pop(0)
        # else:
        end_seqs = []
        if len(self.operations) == 3:
            end_seqs = [
                [self.operations[0]],
                [self.operations[0], self.operations[1]],
                [self.operations[0], self.operations[2]],
                [self.operations[0], self.operations[1], self.operations[2]],
                [self.operations[1]],
                [self.operations[2]],
                [self.operations[1], self.operations[2]],
            ]
        elif len(self.operations) == 2:
            end_seqs = [
                [self.operations[0]],
                [self.operations[0], self.operations[1]],
                [self.operations[1]],
            ]
        elif len(self.operations) == 1:
            end_seqs = [
                [self.operations[0]],
            ]
        # seq = []
        # for operation in self.operations:
        #     step = Step(operation, 'end')
        #     seq.append(step)
        #     # #print("seq",seq)
        #     temp = copy.copy(seq)
        #     # #print('temp',temp)
        #     self.all_seq.append(temp)
        
        for seq in end_seqs:
            step_seq = []
            for operation in seq:
                step = Step(operation, 'end')
                step_seq.append(step)
            temp = copy.copy(step_seq)
            # #print('step_seq', step_seq)
            self.all_seq.append(temp)
        # #print('self.operations', self.operations)
        # self.all_seq += seq
        # pprint.p#print(self.all_seq)
        if len(self.all_seq) not in self.all_seq_num_dict:
            self.all_seq_num_dict[len(self.all_seq)] = 0
        self.all_seq_num_dict[len(self.all_seq)] += 1

    def enum_adding_planb(self, notebook_id):
        def is_all_in(ope_id, seq):
            res = True
            need_operations = set()
            for index,ope in enumerate(self.operations):
                if index == ope_id:
                    break
                need_operations.add(ope)
            has_operations = set()
            for item in seq:
                has_operations.add(item.operator)
            if len(has_operations) < len(need_operations):
                return False
            else:
                return True
        # #print("###############")
        # #print('notebook_id', notebook_id)
        
        self.load_graph(notebook_id)
        res = self.load_planB_operations(notebook_id)
        if res == 'need_wait':
            return res
        # #print('len(self.operations)', len(self.operations))
        self.need_add_edge_id = []
        added_sorted_id = []
        
        for edge in self.graph.result_edges:
            if edge.edge_type == 1 and edge.sorted_id not in added_sorted_id and 'sum' not in edge.func_name:
                self.need_add_edge_id.append(edge.edge_id)
                added_sorted_id.append(edge.sorted_id)
        # #print('len(self.operations)', len(self.operations))
        # #print('len(need_add_edge_id)', len(self.need_add_edge_id))
    
        self.all_seq = []
        if len(self.need_add_edge_id) != 0:
            self.all_seq = [[]]
            for ope_index,operation in enumerate(self.operations):
                new_seq = copy.copy(self.all_seq)
                # #print('ope_index', ope_index)
                for edge_index in self.need_add_edge_id:
                    # #print('edge_index', edge_index)
                    for position in ['before', 'after']:
                        step = Step(operation, str(edge_index)+"---"+position)
                        for seq in self.all_seq:
                            last_step_id = 0
                            last_step_position = 'before'
        
                            if len(seq) != 0:
                                last_step = seq[-1]
                                last_step_id, last_step_position = last_step.edge_id.split("---")
                                last_step_id = int(last_step_id)
                            if edge_index < last_step_id: # 该图能添加的边已经大于现有的边
                                continue
                            if last_step_position == 'after' and position == 'before':
                                continue
                            # if not is_all_in(ope_index, seq):
                                # continue
                            temp_seq = copy.copy(seq)
                            temp_seq.append(step)
                            new_seq.append(temp_seq)
                    
                self.all_seq = new_seq
        if len(self.all_seq) != 0:
            if len(self.all_seq[0]) == 0:
                self.all_seq.pop(0)
        # else:
        end_seqs = [
            [self.operations[0]],
            [self.operations[0], self.operations[1]],
            [self.operations[0], self.operations[2]],
            [self.operations[0], self.operations[1], self.operations[2]],
            [self.operations[1]],
            [self.operations[2]],
            [self.operations[1], self.operations[2]],
        ]

        # seq = []
        # for operation in self.operations:
        #     step = Step(operation, 'end')
        #     seq.append(step)
        #     # #print("seq",seq)
        #     temp = copy.copy(seq)
        #     # #print('temp',temp)
        #     self.all_seq.append(temp)
        
        for seq in end_seqs:
            step_seq = []
            for operation in seq:
                step = Step(operation, 'end')
                step_seq.append(step)
            temp = copy.copy(step_seq)
            # #print('step_seq', step_seq)
            self.all_seq.append(temp)
        # #print('self.operations', self.operations)
        # self.all_seq += seq
        # pprint.p#print(self.all_seq)
        if len(self.all_seq) not in self.all_seq_num_dict:
            self.all_seq_num_dict[len(self.all_seq)] = 0
        self.all_seq_num_dict[len(self.all_seq)] += 1

        # #print('len(self.all_seq)', len(self.all_seq))
        # #print(self.all_seq)
    
    def cut_by_rule(self):
        """
        rule0: imputer, encoder is not need.
        rule1: operations must be done after cleaning.
        rule2: no operation can be done between 'fit' and 'transform'.        
        """
        
        def check_rule0(seq):       
            
            for step in seq:
                if step.operator not in self.can_add_operaiton:
                    return False
            return True

        def check_rule1(seq, edge_dict):
            for step in seq:
                if "---" not in step.edge_id:
                    continue
                edge_id, position = step.edge_id.split("---")
                edge_id = int(edge_id)
                after_list = []
                for one_edge_id in edge_dict:
                    if position == 'before':
                        if one_edge_id < edge_id:
                            continue
                        after_list.append(edge_dict[one_edge_id])
                    elif position == 'after':
                        if one_edge_id <= edge_id:
                            continue
                        after_list.append(edge_dict[one_edge_id])
                # #print('after_list', after_list)
                for operation in after_list:
                    for ope in EdgeOperationType['Cleaning']:
                        if ope in operation:
                            # #print('rule1_false', operation)
                            return False
            return True

        def check_rule2(seq, edge_dict):
            for step in seq:
                if "---" not in step.edge_id:
                    continue
                edge_id, position = step.edge_id.split("---")
                edge_id = int(edge_id)
                before_fit_num = 0
                before_transform_num = 0
                for one_edge_id in edge_dict:
                    if position == 'before':
                        if one_edge_id >= edge_id:
                            continue
                        if edge_dict[one_edge_id] == 'fit':
                            before_fit_num += 1
                        if edge_dict[one_edge_id] == 'transform':
                            before_transform_num += 1
                    elif position == 'after':
                        if one_edge_id > edge_id:
                            continue
                        if edge_dict[one_edge_id] == 'fit':
                            before_fit_num += 1
                        if edge_dict[one_edge_id] == 'transform':
                            before_transform_num += 1
                if before_fit_num > 0 and before_transform_num and before_fit_num == before_transform_num:
                    return False
            return True
        def check_rule3(seq, edge_dict):
            for step in seq:
                if "---" not in step.edge_id:
                    continue
                edge_id, position = step.edge_id.split("---")
                edge_id = int(edge_id)
                before_list = []
                for one_edge_id in edge_dict:
                    if one_edge_id < edge_id:
                        before_list.append(edge_dict[one_edge_id])
                        
                # #print('after_list', after_list)
                # #print('before list', before_list)
                for operation in before_list:
                    if operation == 'train_test_split':
                        return False
            return True

        self.cuted_all_seq = []
        self.edge_dict = {}
        self.edge_all_dict = {}
        for edge in self.graph.result_edges:
            if edge.edge_id in self.need_add_edge_id:
                self.edge_dict[edge.edge_id] = edge.func_name
            if 'train_test_split(' in edge.original_code:
                edge.func_name = 'train_test_split'
            self.edge_all_dict[edge.edge_id] = edge.func_name
            # #print('edge_all_dict funcname', edge.func_name)
            # #print('edge_all_dict origincode', edge.original_code)
        # #print(self.edge_dict)
        # #print(self.operations)
        if len(self.edge_dict) == 0:
            self.cuted_all_seq = copy.copy(self.all_seq)
        else:
            for seq in self.all_seq:
                if check_rule0(seq) and check_rule1(seq, self.edge_dict) and check_rule2(seq, self.edge_dict) and check_rule3(seq, self.edge_all_dict):
                    self.cuted_all_seq.append(seq)
                    # #print("can add")
                    self.print_one_seq(seq)
                # else:
                    # #print('cant add')
                    # self.print_one_seq(seq)

        if len(self.cuted_all_seq) not in self.cuted_seq_num_dict:
            self.cuted_seq_num_dict[len(self.cuted_all_seq)] = 0
        self.cuted_seq_num_dict[len(self.cuted_all_seq)] += 1
        # #print('len(self.cuted_all_seq)',len(self.cuted_all_seq))
    def add_one_ope(self, notebook_id, edge_id, ope, position, varible):
        code_list = self.code.split('\n')
        # self.print_code()
        if ope not in list(OperationCode.keys()):
            return
        
        
        # #print('position', position)
        if position != 'end':
            # #print('xxxx')
            operation_code = OperationCode[ope]['pre_code'] + OperationCode[ope]['code']
            operation_code = operation_code.replace("-[PLACEHOLDER]-", varible) + '\n'
            for edge in self.graph.result_edges:
                if edge.edge_id == edge_id:
                    # #print(edge.line_id)
                    # #print(edge.original_code[0:-1])
                    # #print(edge.line_id[0])
                    add_position = edge.line_id[0] + self.added_rows

            if add_position == 0:
                pre_code_list = []
            else:
                pre_code_list = code_list[0:add_position]
                
            pre_code = ''
            for item in pre_code_list:
                pre_code += item
                pre_code += '\n'
            # #print(self.added_rows)
            # #print('add_position', add_position)
            # #print(len(code_list))
            # #print(code_list[add_position])
            # #print('position', position)
            edge_code = code_list[add_position] + '\n'
            # #print('edge_code', edge_code)
            if add_position == len(code_list)-1:
                after_code_list = []
            else:
                after_code_list = code_list[add_position+1:]
            after_code = ''
            for item in after_code_list:
                after_code += item
                after_code += '\n'
            if position == 'before': # before
                self.code = pre_code + operation_code + edge_code + after_code
            elif position == 'after':
                self.code = pre_code  + edge_code + operation_code + after_code
        else:
            # #print('ope',ope)
            with open("HybridPipeGen/core/tmpdata/prenotebook_varibles_index/"+str(notebook_id)+'.json', 'r') as f:
                varible_index = json.load(f)
            x_varible = varible_index['x_varible']
            operation_code = OperationCode[ope]['pre_code'] + OperationCode[ope]['code']
            operation_code = operation_code.replace("-[PLACEHOLDER]-", x_varible) + '\n'
            # #print(varible_index['end_idnex'])
            end_index = varible_index['end_idnex'] + self.added_rows
            # #print('add_rows',str(self.added_rows))
            # #print('end_index', str(end_index))

            # #print("varible_index['end_idnex']", varible_index['end_idnex'])
            # #print('self.added_rows', self.added_rows)
            code_list = self.code.split("\n")
            # #print(code_list[end_index])
            pre_code_list = code_list[0:end_index+1]
            after_code_list = code_list[end_index+1:]

            pre_code = ''
            for line in pre_code_list:
                pre_code += line
                pre_code += '\n'

            after_code = ''
            for line in after_code_list:
                after_code += line
                after_code += '\n'

            self.code = pre_code + operation_code + after_code
            # #print(self.code)
            
            
        self.added_rows += len(operation_code.split('\n'))-1

    def merging_one_notebook(self,notebook_id):
        self.enum_adding(notebook_id)
        self.cut_by_rule()
        self.load_graph(notebook_id)
        seq_id = 0
        # #print("sss")
        
        # #print(self.cuted_all_seq)
        for seq in self.cuted_all_seq:
            self.added_rows = 0
            self.load_origin_code(notebook_id)
            # if seq_id == 2:
                # self.print_code()
            for step in seq:
                now_ope = step.operator
                if step.edge_id != 'end':
                    edge_id, position = step.edge_id.split("---")
                    edge_id = int(edge_id)
                else:
                    edge_id = 0
                    position = 'end'
                # #print('node_edge.edge_id', node_edge.edge_id)
                varible = ''
                for node in self.graph.result_nodes:
                    # #print('node')
                    for edge_index,node_edge in enumerate(node.children_edges):
                        # #print('edge_index', edge_index)
                        if node_edge.edge_id == edge_id:
                            varible = node.varible_name
                            assign_num = 0
                            temp_varible = ''
                            # #print('varible', varible)
                            for child_index,child_edge in enumerate(node.childrens[edge_index].children_edges):
                                # #print(child_edge.func_name)
                                if child_edge.func_name == '-Assign-':
                                    assign_num += 1
                                    temp_varible = node.childrens[edge_index].childrens[child_index].varible_name
                            if assign_num == 1:
                                varible = temp_varible
                # #print('end varible', varible)
                # #print('seq_id', seq_id)
                self.add_one_ope(notebook_id, edge_id, now_ope, position, varible)
        
            # #print(self.code)
            # if seq_id == 1:
            #     self.print_code()
            #     self.print_one_seq(seq)
            #     break

            save_seq = []
            for item in seq:
                save_seq.append({"operator": item.operator, "edge_id": item.edge_id})
            # if not os.path.exists('merge_code_'+str(self.running_id)+'/'+str(notebook_id)):
            #     os.mkdir('merge_code_'+str(self.running_id)+'/'+str(notebook_id))
            # with open('merge_code_'+str(self.running_id)+'/'+str(notebook_id)+'/'+str(seq_id)+'.json', 'w') as f:
            #     json.dump({'seq':save_seq, 'code': self.code}, f)
            if not os.path.exists('merge_code_tmpdata_1600'+'/'+str(notebook_id)):
                os.mkdir('merge_code_tmpdata_1600'+'/'+str(notebook_id))
            with open('merge_code_tmpdata_1600'+'/'+str(notebook_id)+'/'+str(seq_id)+'.json', 'w') as f:
                json.dump({'seq':save_seq, 'code': self.code}, f)
            seq_id += 1
            
            # break
    def merging_one_notebook_planb(self,notebook_id):
        res = self.enum_adding_planb(notebook_id)
        if res == 'need_wait':
            #print('.....................')
            return res
        self.cut_by_rule()
        self.load_graph(notebook_id)
        seq_id = 0
        # #print("sss")
        
        # #print(self.cuted_all_seq)
        for seq in self.cuted_all_seq:
            self.added_rows = 0
            self.load_origin_code(notebook_id)
            # if seq_id == 2:
                # self.print_code()
            for step in seq:
                now_ope = step.operator
                if step.edge_id != 'end':
                    edge_id, position = step.edge_id.split("---")
                    edge_id = int(edge_id)
                else:
                    edge_id = 0
                    position = 'end'
                # #print('node_edge.edge_id', node_edge.edge_id)
                varible = ''
                for node in self.graph.result_nodes:
                    # #print('node')
                    for edge_index,node_edge in enumerate(node.children_edges):
                        # #print('edge_index', edge_index)
                        if node_edge.edge_id == edge_id:
                            varible = node.varible_name
                            assign_num = 0
                            temp_varible = ''
                            # #print('varible', varible)
                            for child_index,child_edge in enumerate(node.childrens[edge_index].children_edges):
                                # #print(child_edge.func_name)
                                if child_edge.func_name == '-Assign-':
                                    assign_num += 1
                                    temp_varible = node.childrens[edge_index].childrens[child_index].varible_name
                            if assign_num == 1:
                                varible = temp_varible
                # #print('end varible', varible)
                # #print('seq_id', seq_id)
                self.add_one_ope(notebook_id, edge_id, now_ope, position, varible)
        
            # #print(self.code)
            # if seq_id == 1:
            #     self.print_code()
            #     self.print_one_seq(seq)
            #     break

            save_seq = []
            for item in seq:
                save_seq.append({"operator": item.operator, "edge_id": item.edge_id})
            # if not os.path.exists('merge_code_'+str(self.running_id)+'/'+str(notebook_id)):
            #     os.mkdir('merge_code_'+str(self.running_id)+'/'+str(notebook_id))
            # with open('merge_code_'+str(self.running_id)+'/'+str(notebook_id)+'/'+str(seq_id)+'.json', 'w') as f:
            #     json.dump({'seq':save_seq, 'code': self.code}, f)
            if not os.path.exists('planB_test_merge_code_add_rule3'+'/'+str(notebook_id)):
                os.mkdir('planB_test_merge_code_add_rule3'+'/'+str(notebook_id))
            with open('planB_test_merge_code_add_rule3'+'/'+str(notebook_id)+'/'+str(seq_id)+'.json', 'w') as f:
                json.dump({'seq':save_seq, 'code': self.code}, f)
            if not os.path.exists('planB_test_merge_code_add_rule3_py'+'/'+str(notebook_id)):
                os.mkdir('planB_test_merge_code_add_rule3_py'+'/'+str(notebook_id))
            with open('planB_test_merge_code_add_rule3_py'+'/'+str(notebook_id)+'/'+str(seq_id)+'.py', 'w') as f:
                f.write(self.code)
            seq_id += 1

    def merging_one_notebook_rl(self,notebook_id, ai_sequence):
        self.operations = ai_sequence
        res = self.enum_adding_rl(notebook_id)
        self.cut_by_rule()
        self.load_graph(notebook_id)
        seq_id = 0
        # #print("sss")
        
        # #print(self.cuted_all_seq)
        for seq in self.cuted_all_seq:
            self.added_rows = 0
            self.load_origin_code(notebook_id)
            # if seq_id == 2:
                # self.print_code()
            for step in seq:
                now_ope = step.operator
                if step.edge_id != 'end':
                    edge_id, position = step.edge_id.split("---")
                    edge_id = int(edge_id)
                else:
                    edge_id = 0
                    position = 'end'
                # #print('node_edge.edge_id', node_edge.edge_id)
                varible = ''
                for node in self.graph.result_nodes:
                    # #print('node')
                    for edge_index,node_edge in enumerate(node.children_edges):
                        # #print('edge_index', edge_index)
                        if node_edge.edge_id == edge_id:
                            varible = node.varible_name
                            assign_num = 0
                            temp_varible = ''
                            # #print('varible', varible)
                            for child_index,child_edge in enumerate(node.childrens[edge_index].children_edges):
                                # #print(child_edge.func_name)
                                if child_edge.func_name == '-Assign-':
                                    assign_num += 1
                                    temp_varible = node.childrens[edge_index].childrens[child_index].varible_name
                            if assign_num == 1:
                                varible = temp_varible
                # #print('end varible', varible)
                # #print('seq_id', seq_id)
                self.add_one_ope(notebook_id, edge_id, now_ope, position, varible)
        
            # #print(self.code)
            # if seq_id == 1:
            #     self.print_code()
            #     self.print_one_seq(seq)
            #     break

            save_seq = []
            for item in seq:
                save_seq.append({"operator": item.operator, "edge_id": item.edge_id})
            # if not os.path.exists('merge_code_'+str(self.running_id)+'/'+str(notebook_id)):
            #     os.mkdir('merge_code_'+str(self.running_id)+'/'+str(notebook_id))
            # with open('merge_code_'+str(self.running_id)+'/'+str(notebook_id)+'/'+str(seq_id)+'.json', 'w') as f:
            #     json.dump({'seq':save_seq, 'code': self.code}, f)
            if not os.path.exists('HybridPipeGen/core/tmpdata/rl_test_merge_code'+'/'+str(notebook_id)):
                os.mkdir('HybridPipeGen/core/tmpdata/rl_test_merge_code'+'/'+str(notebook_id))
            with open('HybridPipeGen/core/tmpdata/rl_test_merge_code'+'/'+str(notebook_id)+'/'+str(seq_id)+'.json', 'w') as f:
                json.dump({'seq':save_seq, 'code': self.code}, f)
            if not os.path.exists('HybridPipeGen/core/tmpdata/rl_test_merge_code_py'+'/'+str(notebook_id)):
                os.mkdir('HybridPipeGen/core/tmpdata/rl_test_merge_code_py'+'/'+str(notebook_id))
            with open('HybridPipeGen/core/tmpdata/rl_test_merge_code_py'+'/'+str(notebook_id)+'/'+str(seq_id)+'.py', 'w') as f:
                f.write(self.code)
            seq_id += 1

    def count_operations(self):
        # filelist = os.listdir('deepline_only_new_'+str(self.running_id))
        filelist = os.listdir('deepline_only_tmpdata')
        notebooks = set()
        for item in filelist:
            notebooks.add(int(item.split('_')[0]))
    
        hnum_dict = {}
        num2notebooks = {}
        for notebook_id in list(notebooks):
            self.load_operations(notebook_id)
            if len(self.operations) not in hnum_dict:
                hnum_dict[len(self.operations)] = 0
            if len(self.operations) not in num2notebooks:
                num2notebooks[len(self.operations)] = []
            hnum_dict[len(self.operations)] += 1
            num2notebooks[len(self.operations)].append(notebook_id)
        # pprint.p#print(sorted(hnum_dict.items(), key=lambda d: d[0]))
        
    def count_hightlight(self):
        # filelist = os.listdir('deepline_only_new_'+str(self.running_id))
        filelist = os.listdir('deepline_only_tmpdata')
        notebooks = set()
        for item in filelist:
            notebooks.add(int(item.split('_')[0]))
    
        hnum_dict = {}
        num2notebooks = {}
        for notebook_id in list(notebooks):
            highlight = set()
            # with open('HybridPipeGen/core/tmpdata/prenotebook_graph/'+str(notebook_id)+".pkl", 'rb') as f:
            with open('/home/yxm/staticfg-master/HybridPipeGen/core/tmpdata/HybridPipeGen/core/tmpdata/prenotebook_graph/'+str(notebook_id)+".pkl", 'rb') as f:
                graph = pickle.load(f)
                for edge in graph.result_edges:
                    if edge.edge_type == 1:
                        highlight.add(edge.sorted_id)
            # #print(notebook_id, len(highlight))
            if len(highlight) not in hnum_dict:
                hnum_dict[len(highlight)] = 0
            if len(highlight) not in num2notebooks:
                num2notebooks[len(highlight)] = []
            hnum_dict[len(highlight)] += 1
            num2notebooks[len(highlight)].append(notebook_id)
        # #print(hnum_dict)
        # pprint.p#print(sorted(hnum_dict.items(), key=lambda d: d[0]))
        # #print(num2notebooks[2])

    def batch_enuming_planb(self):
        with open('planB_test_tasks.json','r') as f:
            test_tasks = json.load(f)
        with open('clean_task_no_1_fix_label_s.json','r') as f:
            clean_data = json.load(f)
        
        notebooks = []
        for task in clean_data:
            if task in test_tasks:
                for nid in clean_data[task]['notebook_list']:
                    notebooks.append(nid)

        # with open('return_cross.json','r') as f:
            # notebooks = json.load(f)
        exist_f = open("/home/yxm/staticfg-master/all_s.txt", "r")
        exist = exist_f.readlines()
        exitst_ = [x.strip("\n") for x in exist]
        notebooks = list(set(notebooks) & set(exitst_))
        count = 0
        failed = 0
        for notebook_id in notebooks:
            # if os.path.exists('planB_test_merge_code'+'/'+str(notebook_id)):
                # continue
            # if notebook_id != 'rishighybrid97_linear-svm-classification':
                # continue
            #print('notebook_id', notebook_id)
            count += 1
            if not os.path.exists('/home/yxm/staticfg-master/HybridPipeGen/core/tmpdata/HybridPipeGen/core/tmpdata/prenotebook_graph/' + notebook_id + '.pkl'): # graph failed
                continue
            try:
                merger.merging_one_notebook_planb(notebook_id)
            except:
                failed += 1
                continue
            end = time.time()
            validation_running_time[str(notebook_id)] = end-start_time
            with open("rl_step2_merge_time.json", 'w') as f:
                json.dump(validation_running_time, f)

            # break

        merger.all_seq_num_dict = sorted(merger.all_seq_num_dict.items(), key=lambda d: d[0])
        # pprint.p#print(merger.all_seq_num_dict)
        origin_num = 0
        for item in merger.all_seq_num_dict:
            origin_num += item[0]*item[1]
        merger.cuted_seq_num_dict = sorted(merger.cuted_seq_num_dict.items(), key=lambda d: d[0])
        # pprint.p#print(merger.cuted_seq_num_dict)
        cut_num = 0
        for item in merger.cuted_seq_num_dict:
            cut_num += item[0]*item[1]
        #print(origin_num)
        #print(cut_num)
        #print('count', count)
        #print('failed', failed)
    def batch_enuming_rl(self):
        with open('planB_test_tasks.json','r') as f:
            test_tasks = json.load(f)
        with open('clean_task_rl_200.json','r') as f:
            clean_data = json.load(f)
        if os.path.exists("rl_step2_merge_time.json"):
            with open("rl_step2_merge_time.json", 'r') as f:
                validation_running_time = json.load(f)
        else:
            validation_running_time = {}
        notebooks = []
        for task in clean_data:
            if task in test_tasks:
                for nid in clean_data[task]['notebook_list']:
                    notebooks.append(nid)

        # with open('return_cross.json','r') as f:
            # notebooks = json.load(f)
        exist_f = open("/home/yxm/staticfg-master/all_s.txt", "r")
        exist = exist_f.readlines()
        exitst_ = [x.strip("\n") for x in exist]
        notebooks = list(set(notebooks) & set(exitst_))
        count = 0
        failed = 0
        for notebook_id in notebooks:
            # if os.path.exists('planB_test_merge_code'+'/'+str(notebook_id)):
                # continue
            # if notebook_id != 'rishighybrid97_linear-svm-classification':
                # continue
            #print('notebook_id', notebook_id)
            count += 1
            if not os.path.exists('/home/yxm/staticfg-master/HybridPipeGen/core/tmpdata/HybridPipeGen/core/tmpdata/prenotebook_graph/' + notebook_id + '.pkl'): # graph failed
                continue
            if os.path.exists("rl_step2_merge_time.json"):
                with open("rl_step2_merge_time.json", 'r') as f:
                    validation_running_time = json.load(f)
            if str(notebook_id) not in validation_running_time:
                validation_running_time[str(notebook_id)] = {}
            start_time = time.time()
            # try:
            merger.merging_one_notebook_rl(notebook_id)
            # except:
                # failed += 1
                # continue
            # break
            end = time.time()
            validation_running_time[str(notebook_id)] = end-start_time
            with open("rl_step2_merge_time.json", 'w') as f:
                json.dump(validation_running_time, f)


        merger.all_seq_num_dict = sorted(merger.all_seq_num_dict.items(), key=lambda d: d[0])
        # pprint.p#print(merger.all_seq_num_dict)
        origin_num = 0
        for item in merger.all_seq_num_dict:
            origin_num += item[0]*item[1]
        merger.cuted_seq_num_dict = sorted(merger.cuted_seq_num_dict.items(), key=lambda d: d[0])
        # pprint.p#print(merger.cuted_seq_num_dict)
        cut_num = 0
        for item in merger.cuted_seq_num_dict:
            cut_num += item[0]*item[1]
        #print(origin_num)
        #print(cut_num)
        #print('count', count)
        #print('failed', failed)

    def batch_enuming(self):
        # filelist = os.listdir('deepline_only_new_'+str(self.running_id))
        # filelist = os.listdir('deepline_only_tmpdata')
        # notebooks = []
        count = 0
        # for file_ in filelist:
        #     if int(file_.split('_')[0]) not in notebooks:
        #         notebooks.append(int(file_.split('_')[0]))
        failed = 0

        # for file_ in filelist:
        #     if str(file_.rsplit('_',1)[0]) not in notebooks:
        #         notebooks.append(str(file_.rsplit('_',1)[0]))
        with open('return_cross.json','r') as f:
            notebooks = json.load(f)
        exist_f = open("/home/yxm/staticfg-master/all_s.txt", "r")
        exist = exist_f.readlines()
        exitst_ = [x.strip("\n") for x in exist]
        notebooks = list(set(notebooks) & set(exitst_))

        for notebook_id in notebooks:
            count += 1
            # exist_f = open("/home/yxm/staticfg-master/origin_new.txt", "r")
            # exist = exist_f.readlines()
            # exitst_ = [x.strip("\n") for x in exist]
            # if notebook_id not in exitst_:
            #     continue
            # if count <= -1:
            #     count += 1
            #     continue
            # if os.path.exists('merge_code_tmpdata_1600/'+str(notebook_id)):
                # continue
            # if notebook_id !='bilal75210_project-data-science':
                # continue
            
            # if notebook_id == 'srikanthmalyala_zs-challenge' or notebook_id == 'shreyajune_leads-score-eda-and-simple-regression-tutorial' or notebook_id == 'surya08084_loan-starter-notebook' or notebook_id == 'vijit2911_lead-scoring' or notebook_id == 'roh245_a-statistical-analysis-of-imdb-ratings-of-us-films':
            #     continue
            if not os.path.exists('/home/yxm/staticfg-master/HybridPipeGen/core/tmpdata/HybridPipeGen/core/tmpdata/prenotebook_graph/' + notebook_id + '.pkl'): # graph failed
                continue
            # try:
                # self.enum_adding(notebook_id)
                # self.cut_by_rule()
            # if not os.path.exists('HybridPipeGen/core/tmpdata/HybridPipeGen/core/tmpdata/prenotebook_graph/'+notebook_id+'.pkl'):
                # continue
            try:
                merger.merging_one_notebook(notebook_id)
            except:
                failed += 1
                continue
            # break
        # #print(merger.all_seq_num_dict)
        merger.all_seq_num_dict = sorted(merger.all_seq_num_dict.items(), key=lambda d: d[0])
        # pprint.p#print(merger.all_seq_num_dict)
        origin_num = 0
        for item in merger.all_seq_num_dict:
            origin_num += item[0]*item[1]
        merger.cuted_seq_num_dict = sorted(merger.cuted_seq_num_dict.items(), key=lambda d: d[0])
        # pprint.p#print(merger.cuted_seq_num_dict)
        cut_num = 0
        for item in merger.cuted_seq_num_dict:
            cut_num += item[0]*item[1]
        #print(origin_num)
        #print(cut_num)
        #print('count', count)

        #print('failed', failed)
def transform_origin_validation_code(split_random_state=0, running_id=1):
    # notebooks = os.listdir('/home/yxm/staticfg-master/HybridPipeGen/core/tmpdata/prenotebook_code')
    # notebooks.sort()
    res = []
    with open('return_cross.json','r') as f:
        notebooks = json.load(f)
    exist_f = open("/home/yxm/staticfg-master/all_s.txt", "r")
    exist = exist_f.readlines()
    exitst_ = [x.strip("\n") for x in exist]
    # notebooks = list(set(notebooks) & set(exitst_))
    for notebook_id in notebooks:
        if notebook_id != '351382':
            continue
        # notebook_id = int(notebook_id_py.split('.')[0])
        # notebook_id = str(notebook_id_py.split('.')[0])
        notebook_id_py = notebook_id +'.py'
        # if notebook_id != 'adkspence_heart-disease-prediction':
            # continue
        # #print(notebook_id)
        test_num = 0
        exist_f = open("/home/yxm/staticfg-master/all_s.txt", "r")
        exist = exist_f.readlines()
        exitst_ = [x.strip("\n") for x in exist]
        # if notebook_id not in exitst_:
            # continue
        # if notebook_id != "gabrielfrborges_iris-ds":
        #     continue
        # if os.path.exists('/home/yxm/staticfg-master/HybridPipeGen/core/tmpdata/merge_validation_code/'+str(notebook_id)+'/origin.json'):
        #     continue
        # if notebook_id == "tphaterp_exploring-gender-differences-in-heart-disease":
        #     continue
        with open("/home/yxm/staticfg-master/HybridPipeGen/core/tmpdata/prenotebook_code/" + notebook_id_py, 'r') as f:
            test_code = f.read()
        # #print("------------1----------------------------------------")
        # #print(test_code)
        test_code = cleaning_origin(test_code)
        # #print("------------2----------------------------------------")
        # #print(test_code)
        
        validation_code = ''
        test_code_list = test_code.split('\n')
        seq_code_dict = {}
        ours_index =0
        # #print("------------3----------------------------------------")
        for index, line in enumerate(test_code_list):
            if 'train_test_split(' in line:
                if '=' not in line:
                    continue
                
                #print(line)
                try:
                    x_train_varible , x_test_varible , y_train_varible , y_test_varible = line.split('=')[0].strip().split(',')
                    train_test_index = index
                except:
                    continue
                train_test_split_code = line.split('=')[1].strip()
                kuohao = train_test_split_code.find('(')
                train_test_split_code = train_test_split_code[kuohao+1:]
                arglist = train_test_split_code.split(",")
                # #print(arglist)
                x_varible = arglist[0].strip()
                y_varible = arglist[1].strip()
            if "model.fit(" in line:
                ours_index = index
        # #print(ours_index)
        # #print("------------4----------------------------------------")

        for index, line in enumerate(test_code_list):
            # if index < train_test_index -1:
            #     continue
            if index == train_test_index-1:
                validation_code += line
                validation_code += '\n'
                new_line0 = 'import numpy as np\n'
                # new_line = 'np.save("'+"merge_result_data/"+str(notebook_id)+"/"+"origin_data_x.npy\"," +x_varible+')\n'
                # new_line1 = 'np.save("'+"merge_result_data/"+str(notebook_id)+"/"+"origin_data_y.npy\"," +y_varible+')\n'
                new_line = 'np.save("'+"/home/yxm/staticfg-master/HybridPipeGen/core/tmpdata/merge_result_data/"+str(notebook_id)+"/"+"origin_data_x.npy\"," +x_varible+')\n'
                new_line1 = 'np.save("'+"/home/yxm/staticfg-master/HybridPipeGen/core/tmpdata/merge_result_data/"+str(notebook_id)+"/"+"origin_data_y.npy\"," +y_varible+')\n'
                # validation_code += new_line0
                # validation_code += new_line
                # validation_code += new_line1
            if index == train_test_index:
                validation_code += line
                validation_code += '\n'
                #print('index', line)
                try:
                    x_train_varible , x_test_varible , y_train_varible , y_test_varible = line.split('=')[0].strip().split(',')
                    x_train_varible = x_train_varible.strip()
                    not_null_index = line.find(x_train_varible)
                    start_null = line[0:not_null_index]
                except:
                    continue
                x_train_varible = x_train_varible.strip()
                x_test_varible = x_test_varible.strip()
                y_train_varible = y_train_varible.strip()
                y_test_varible = y_test_varible.strip()
                x_validation_varible = 'x_validation_varible'
                y_validation_varible = 'y_validation_varible'
    
                # new_train_test_split = start_null + x_train_varible + ', ' + x_validation_varible + ', ' +  y_train_varible + ', ' +  y_validation_varible
                # new_train_test_split = new_train_test_split + " = train_test_split(" + x_train_varible + ', ' + y_train_varible + ', train_size='+str(0.875)+', test_size='+str(0.125)+', random_state='+str(split_random_state)+')' + '\n'
                # validation_code += new_train_test_split
                cross_validation_code = 'from sklearn.model_selection import cross_val_score\n'
                cross_validation_code += 'cross_score = cross_val_score(model, ' + x_train_varible +', ' + y_train_varible + ',cv=4)\n'
                    

            elif 'model.fit(' in line  and index >= ours_index:
                    validation_code += '#'+line.replace(x_test_varible, x_validation_varible)
                    validation_code += '\n'
            elif 'model.predict(' in line and index >= ours_index:
                validation_code += '#'+line.replace(x_test_varible, x_validation_varible)
                validation_code += '\n'
            
            elif 'score = accuracy_score(' in line  and index >= ours_index:
                validation_code += '#'+line.replace(y_test_varible, y_validation_varible)
                validation_code += '\n'
                validation_code += cross_validation_code
            elif 'np.save("HybridPipeGen/core/tmpdata/prenotebook_res/' in line:
                validation_code += '#'+line.replace(y_test_varible, y_validation_varible)
                validation_code += '\n'
                validation_code += line.replace('prenotebook_res', 'cross_val_res').replace(': score }', ': cross_score }')
                validation_code += '\n'
            else:
                validation_code += line
                validation_code += '\n'
            if index > train_test_index and index < ours_index:
                # #print(x_test_varible)
                if x_test_varible in line:
                    test_num +=1
        if test_num > 0:
            res.append(notebook_id)
        # #print(res)
        # #print(len(res))
        with open("test_error.txt","w")as f:
            for x in res:
                f.write(x+"\n")
        # validation_code = validation_code.replace("prenotebook_res", 'validation_prenotebook_res')
        # #print(validation_code)
        # validation_code = clean_kuohao(validation_code)
        # #print(validation_code)
        seq_code_dict['validation_code'] = validation_code
        notebook_id= str(notebook_id)
        # if not os.path.exists('merge_validation_code_'+str(running_id)+'/'+notebook_id):
        #     os.mkdir('merge_validation_code_'+str(running_id)+'/'+notebook_id)
        # if not os.path.exists('merge_validation_code_py_'+str(running_id)+'/'+notebook_id):
        #     os.mkdir('merge_validation_code_py_'+str(running_id)+'/'+notebook_id)
        # with open('merge_validation_code_'+str(running_id)+'/'+notebook_id+'/origin.json', 'w') as f:
        #     json.dump(seq_code_dict, f)
        # with open('merge_validation_code_py_'+str(running_id)+'/'+notebook_id+'/origin.py', 'w') as f:
        #     f.write(validation_code)
        if not os.path.exists('/home/yxm/staticfg-master/HybridPipeGen/core/tmpdata/cross_validation_code'+'/'+notebook_id):
            os.mkdir('/home/yxm/staticfg-master/HybridPipeGen/core/tmpdata/cross_validation_code'+'/'+notebook_id)
        if not os.path.exists('/home/yxm/staticfg-master/HybridPipeGen/core/tmpdata/cross_validation_code_py'+'/'+notebook_id):
            os.mkdir('/home/yxm/staticfg-master/HybridPipeGen/core/tmpdata/cross_validation_code_py'+'/'+notebook_id)
        with open('/home/yxm/staticfg-master/HybridPipeGen/core/tmpdata/cross_validation_code'+'/'+notebook_id+'/origin.json', 'w') as f:
            json.dump(seq_code_dict, f)
        with open('/home/yxm/staticfg-master/HybridPipeGen/core/tmpdata/cross_validation_code_py'+'/'+notebook_id+'/origin.py', 'w') as f:
            f.write(validation_code)


def cleaning_origin(code):
    lines = code.split('\n')
    res_str = ''
    # #print("*************************************************")
    for line in lines:
        # #print(line)
        line1 = line.strip()
        # #print(line1)
        # if '#' in line and "color = \"#" not in line and "color= \"#" not in line and "color =\"#" not in line and "color = '#" not in line and "color= ''#" not in line and "color ='#" not in line:
        if '#' in line:
            if line1[0]=="#":
                index = line.index("#")
                line = line[0:index]
        if len(line) != 0:
            line1 = line.strip()
            # #print(line1)
            if line[-1] == '\\':
                res_str += line[0:-1]
            elif len(line1) > 0:
                if line1[-1] == ',':
                    res_str += line
                elif line1[-1] == '(':
                    res_str += line
                else:
                    res_str += line
                    res_str += '\n'
            else:
                res_str += line
                res_str += '\n'

    return res_str

if os.path.exists('need_modify.npy'):
    need_modify = list(np.load('need_modify.npy', allow_pickle=True))
else:
    need_modify = []

def transform_validation_planB():
    notebooks = os.listdir('planB_test_merge_code/')
    # notebooks.sort()
    # with open('return_cross.json','r') as f:
    #     notebooks = json.load(f)
    # exist_f = open("/home/yxm/staticfg-master/all_s.txt", "r")
    # exist = exist_f.readlines()
    # exitst_ = [x.strip("\n") for x in exist]
    # notebooks = list(set(notebooks) & set(exitst_) & set(notebook_m))
    # #print(len(notebooks))
    for notebook_id in notebooks:
        # if notebook_id != "gabrielfrborges_iris-ds":
        #     continue
        if notebook_id == 'tmpfile':
            continue
        seq_files = os.listdir('planB_test_merge_code/'+notebook_id)
        need_continue=False
        for seq_file in seq_files:
            if os.path.exists('/home/yxm/staticfg-master/HybridPipeGen/core/tmpdata/planB_cross_validation_code/'+notebook_id+'/'+seq_file.replace('.json', '.py')):
                continue
            seq_index = seq_file.split('.')[0]

            with open('planB_test_merge_code/'+notebook_id+'/'+seq_file, 'r') as f:
                seq_code_dict = json.load(f)
            test_code = seq_code_dict['code']
            test_code = cleaning_origin(test_code)
            validation_code = ''
            test_code_list = test_code.split('\n')

            train_test_index = 0
            for index, line in enumerate(test_code_list):
                if '=train_test_split(' in line or ' train_test_split(' in line:
                    if '=' not in line:
                        continue
                    try:
                        x_train_varible , x_test_varible , y_train_varible , y_test_varible = line.split('=')[0].strip().split(',')
                        train_test_index = index
                    except:
                        continue
                    train_test_split_code = line.split('=')[1].strip()
                    kuohao = train_test_split_code.find('(')
                    train_test_split_code = train_test_split_code[kuohao+1:]
                    arglist = train_test_split_code.split(",")
                    # #print(arglist)
                    try:
                        x_varible = arglist[0].strip()
                        y_varible = arglist[1].strip()
                    except:
                        need_modify.append(notebook_id)
                        np.save('need_modify.npy', need_modify)
                        need_continue = True
                if "model.fit(" in line:
                    ours_index = index
            if need_continue:
                break
            # #print(train_test_index)
            for index, line in enumerate(test_code_list):
                if index == train_test_index-1:
                    validation_code += line
                    validation_code += '\n'
                    # new_line0 = 'import numpy as np\n'
                    # new_line = 'np.save("'+"/home/yxm/staticfg-master/HybridPipeGen/core/tmpdata/merge_result_data/"+str(notebook_id)+"/"+seq_index+"_data_x.npy\"," +x_varible+')\n'
                    # new_line1 = 'np.save("'+"/home/yxm/staticfg-master/HybridPipeGen/core/tmpdata/merge_result_data/"+str(notebook_id)+"/"+seq_index+"_data_y.npy\"," +y_varible+')\n'
                    # validation_code += new_line0
                    # validation_code += new_line
                    # validation_code += new_line1
                if index == train_test_index:
                    validation_code += line
                    validation_code += '\n'
                    try:
                        x_train_varible , x_test_varible , y_train_varible , y_test_varible = line.split('=')[0].strip().split(',')
                        x_train_varible = x_train_varible.strip()
                        not_null_index = line.find(x_train_varible)
                        start_null = line[0:not_null_index]
                    except:
                        continue
                    # #print("....")
                    x_train_varible = x_train_varible.strip()
                    x_test_varible = x_test_varible.strip()
                    y_train_varible = y_train_varible.strip()
                    y_test_varible = y_test_varible.strip()
                    x_validation_varible = 'x_validation_varible'
                    y_validation_varible = 'y_validation_varible'
        

                    cross_validation_code = 'from sklearn.model_selection import cross_val_score\n'
                    cross_validation_code += 'cross_score = cross_val_score(model, ' + x_train_varible +', ' + y_train_varible + ',cv=4)\n'
                    # new_train_test_split = start_null + x_train_varible + ', ' + x_validation_varible + ', ' +  y_train_varible + ', ' +  y_validation_varible
                    # new_train_test_split = new_train_test_split + " = train_test_split(" + x_train_varible + ', ' + y_train_varible + ', train_size='+str(0.875)+', test_size='+str(0.125)+', random_state='+str(split_random_state)+')' + '\n'
                    # validation_code += new_train_test_split
                    
                elif 'model.fit(' in line  and index >= ours_index:
                    validation_code += '#'+line.replace(x_test_varible, x_validation_varible)
                    validation_code += '\n'
                elif 'model.predict(' in line and index >= ours_index:
                    validation_code += '#'+line.replace(x_test_varible, x_validation_varible)
                    validation_code += '\n'
                
                elif 'score = accuracy_score(' in line  and index >= ours_index:
                    validation_code += '#'+line.replace(y_test_varible, y_validation_varible)
                    validation_code += '\n'
                    validation_code += cross_validation_code
                elif 'np.save("HybridPipeGen/core/tmpdata/prenotebook_res/' in line:
                    validation_code += '#'+line.replace(y_test_varible, y_validation_varible)
                    validation_code += '\n'
                    validation_code += line.replace('prenotebook_res', 'cross_val_res').replace(': score }', ': cross_score }')
                    validation_code += '\n'
                else:
                    validation_code += line
                    validation_code += '\n'
            seq_code_dict['validation_code'] = validation_code
            # #print(seq_code_dict)
            #print('????????????????????????')
            if not os.path.exists('/home/yxm/staticfg-master/HybridPipeGen/core/tmpdata/planB_cross_validation_code_add3/'+notebook_id):
                os.mkdir('/home/yxm/staticfg-master/HybridPipeGen/core/tmpdata/planB_cross_validation_code_add3/'+notebook_id)
            if not os.path.exists('/home/yxm/staticfg-master/HybridPipeGen/core/tmpdata/planB_cross_validation_code_add3_py/'+notebook_id):
                os.mkdir('/home/yxm/staticfg-master/HybridPipeGen/core/tmpdata/planB_cross_validation_code_add3_py/'+notebook_id)
            with open('/home/yxm/staticfg-master/HybridPipeGen/core/tmpdata/planB_cross_validation_code_add3/'+notebook_id+'/'+seq_file, 'w') as f:
                json.dump(seq_code_dict, f)
            with open('/home/yxm/staticfg-master/HybridPipeGen/core/tmpdata/planB_cross_validation_code_add3_py/'+notebook_id+'/'+seq_file.replace('.json', '.py'), 'w') as f:
                f.write(validation_code)

def transform_validation_rl():
    notebooks = os.listdir('HybridPipeGen/core/tmpdata/rl_test_merge_code/')
    # notebooks.sort()
    # with open('return_cross.json','r') as f:
    #     notebooks = json.load(f)
    # exist_f = open("/home/yxm/staticfg-master/all_s.txt", "r")
    # exist = exist_f.readlines()
    # exitst_ = [x.strip("\n") for x in exist]
    # notebooks = list(set(notebooks) & set(exitst_) & set(notebook_m))
    # #print(len(notebooks))
    for notebook_id in notebooks:
        # if notebook_id != "gabrielfrborges_iris-ds":
        #     continue
        if notebook_id == 'tmpfile':
            continue
        seq_files = os.listdir('HybridPipeGen/core/tmpdata/rl_test_merge_code/'+notebook_id)
        need_continue=False
        with open("rl_step2_merge_time.json", 'r') as f:
            validation_running_time = json.load(f)
        start_time = time.time()
        for seq_file in seq_files:
            if os.path.exists('/home/yxm/staticfg-master/HybridPipeGen/core/tmpdata/rl_cross_validation_code/'+notebook_id+'/'+seq_file.replace('.json', '.py')):
                continue
            seq_index = seq_file.split('.')[0]
            
            with open('HybridPipeGen/core/tmpdata/rl_test_merge_code/'+notebook_id+'/'+seq_file, 'r') as f:
                seq_code_dict = json.load(f)
            test_code = seq_code_dict['code']
            test_code = cleaning_origin(test_code)
            validation_code = ''
            test_code_list = test_code.split('\n')

            train_test_index = 0
            for index, line in enumerate(test_code_list):
                if '=train_test_split(' in line or ' train_test_split(' in line:
                    if '=' not in line:
                        continue
                    try:
                        x_train_varible , x_test_varible , y_train_varible , y_test_varible = line.split('=')[0].strip().split(',')
                        train_test_index = index
                    except:
                        continue
                    train_test_split_code = line.split('=')[1].strip()
                    kuohao = train_test_split_code.find('(')
                    train_test_split_code = train_test_split_code[kuohao+1:]
                    arglist = train_test_split_code.split(",")
                    # #print(arglist)
                    try:
                        x_varible = arglist[0].strip()
                        y_varible = arglist[1].strip()
                    except:
                        need_modify.append(notebook_id)
                        np.save('need_modify.npy', need_modify)
                        need_continue = True
                if "model.fit(" in line:
                    ours_index = index
            if need_continue:
                break
            # #print(train_test_index)
            for index, line in enumerate(test_code_list):
                if index == train_test_index-1:
                    validation_code += line
                    validation_code += '\n'
                    # new_line0 = 'import numpy as np\n'
                    # new_line = 'np.save("'+"/home/yxm/staticfg-master/HybridPipeGen/core/tmpdata/merge_result_data/"+str(notebook_id)+"/"+seq_index+"_data_x.npy\"," +x_varible+')\n'
                    # new_line1 = 'np.save("'+"/home/yxm/staticfg-master/HybridPipeGen/core/tmpdata/merge_result_data/"+str(notebook_id)+"/"+seq_index+"_data_y.npy\"," +y_varible+')\n'
                    # validation_code += new_line0
                    # validation_code += new_line
                    # validation_code += new_line1
                if index == train_test_index:
                    validation_code += line
                    validation_code += '\n'
                    try:
                        x_train_varible , x_test_varible , y_train_varible , y_test_varible = line.split('=')[0].strip().split(',')
                        x_train_varible = x_train_varible.strip()
                        not_null_index = line.find(x_train_varible)
                        start_null = line[0:not_null_index]
                    except:
                        continue
                    # #print("....")
                    x_train_varible = x_train_varible.strip()
                    x_test_varible = x_test_varible.strip()
                    y_train_varible = y_train_varible.strip()
                    y_test_varible = y_test_varible.strip()
                    x_validation_varible = 'x_validation_varible'
                    y_validation_varible = 'y_validation_varible'
        

                    cross_validation_code = 'from sklearn.model_selection import cross_val_score\n'
                    cross_validation_code += 'cross_score = cross_val_score(model, ' + x_train_varible +', ' + y_train_varible + ',cv=4)\n'
                    # new_train_test_split = start_null + x_train_varible + ', ' + x_validation_varible + ', ' +  y_train_varible + ', ' +  y_validation_varible
                    # new_train_test_split = new_train_test_split + " = train_test_split(" + x_train_varible + ', ' + y_train_varible + ', train_size='+str(0.875)+', test_size='+str(0.125)+', random_state='+str(split_random_state)+')' + '\n'
                    # validation_code += new_train_test_split
                    
                elif 'model.fit(' in line  and index >= ours_index:
                    validation_code += '#'+line.replace(x_test_varible, x_validation_varible)
                    validation_code += '\n'
                elif 'model.predict(' in line and index >= ours_index:
                    validation_code += '#'+line.replace(x_test_varible, x_validation_varible)
                    validation_code += '\n'
                
                elif 'score = accuracy_score(' in line  and index >= ours_index:
                    validation_code += '#'+line.replace(y_test_varible, y_validation_varible)
                    validation_code += '\n'
                    validation_code += cross_validation_code
                elif 'np.save("HybridPipeGen/core/tmpdata/prenotebook_res/' in line:
                    validation_code += '#'+line.replace(y_test_varible, y_validation_varible)
                    validation_code += '\n'
                    validation_code += line.replace('prenotebook_res', 'cross_val_res').replace(': score }', ': cross_score }')
                    validation_code += '\n'
                else:
                    validation_code += line
                    validation_code += '\n'
            seq_code_dict['validation_code'] = validation_code
            # #print(seq_code_dict)
            # #print('????????????????????????')
            if not os.path.exists('/home/yxm/staticfg-master/HybridPipeGen/core/tmpdata/rl_cross_validation_code/'+notebook_id):
                os.mkdir('/home/yxm/staticfg-master/HybridPipeGen/core/tmpdata/rl_cross_validation_code/'+notebook_id)
            if not os.path.exists('/home/yxm/staticfg-master/HybridPipeGen/core/tmpdata/rl_cross_validation_code_py/'+notebook_id):
                os.mkdir('/home/yxm/staticfg-master/HybridPipeGen/core/tmpdata/rl_cross_validation_code_py/'+notebook_id)
            with open('/home/yxm/staticfg-master/HybridPipeGen/core/tmpdata/rl_cross_validation_code/'+notebook_id+'/'+seq_file, 'w') as f:
                json.dump(seq_code_dict, f)
            with open('/home/yxm/staticfg-master/HybridPipeGen/core/tmpdata/rl_cross_validation_code_py/'+notebook_id+'/'+seq_file.replace('.json', '.py'), 'w') as f:
                f.write(validation_code)
        end = time.time()
        validation_running_time[notebook_id] += end - start_time
        with open("rl_step2_merge_time.json", 'w') as f:
            json.dump(validation_running_time, f)



def transform_validation_code(running_id = 1,split_random_state=0):

    notebook_m = os.listdir('merge_code_tmpdata_1600')
    # notebooks.sort()
    with open('return_cross.json','r') as f:
        notebooks = json.load(f)
    exist_f = open("/home/yxm/staticfg-master/all_s.txt", "r")
    exist = exist_f.readlines()
    exitst_ = [x.strip("\n") for x in exist]
    notebooks = list(set(notebooks) & set(exitst_) & set(notebook_m))
    # #print(len(notebooks))
    for notebook_id in notebooks:
        # if notebook_id != "gabrielfrborges_iris-ds":
        #     continue
        if notebook_id == 'tmpfile':
            continue
        seq_files = os.listdir('merge_code_tmpdata_1600/'+notebook_id)
        need_continue=False
        for seq_file in seq_files:
            # if os.path.exists('/home/yxm/staticfg-master/HybridPipeGen/core/tmpdata/merge_validation_code/'+notebook_id+'/'+seq_file.replace('.json', '.py')):
            #     continue
            seq_index = seq_file.split('.')[0]
            # #print(notebook_id)
            exist_f = open("/home/yxm/staticfg-master/origin_1756.txt", "r")
            exist = exist_f.readlines()
            exitst_ = [x.strip("\n") for x in exist]
            # if notebook_id not in exitst_:
                # continue
                
            # #print(seq_file)
            with open('merge_code_tmpdata_1600/'+notebook_id+'/'+seq_file, 'r') as f:
                seq_code_dict = json.load(f)
            test_code = seq_code_dict['code']
            test_code = cleaning_origin(test_code)
            validation_code = ''
            test_code_list = test_code.split('\n')

            train_test_index = 0
            for index, line in enumerate(test_code_list):
                if '=train_test_split(' in line or ' train_test_split(' in line:
                    if '=' not in line:
                        continue
                    try:
                        x_train_varible , x_test_varible , y_train_varible , y_test_varible = line.split('=')[0].strip().split(',')
                        train_test_index = index
                    except:
                        continue
                    train_test_split_code = line.split('=')[1].strip()
                    kuohao = train_test_split_code.find('(')
                    train_test_split_code = train_test_split_code[kuohao+1:]
                    arglist = train_test_split_code.split(",")
                    # #print(arglist)
                    try:
                        x_varible = arglist[0].strip()
                        y_varible = arglist[1].strip()
                    except:
                        need_modify.append(notebook_id)
                        np.save('need_modify.npy', need_modify)
                        need_continue = True
                if "model.fit(" in line:
                    ours_index = index
            if need_continue:
                break
            # #print(train_test_index)
            for index, line in enumerate(test_code_list):
                if index == train_test_index-1:
                    validation_code += line
                    validation_code += '\n'
                    new_line0 = 'import numpy as np\n'
                    new_line = 'np.save("'+"/home/yxm/staticfg-master/HybridPipeGen/core/tmpdata/merge_result_data/"+str(notebook_id)+"/"+seq_index+"_data_x.npy\"," +x_varible+')\n'
                    new_line1 = 'np.save("'+"/home/yxm/staticfg-master/HybridPipeGen/core/tmpdata/merge_result_data/"+str(notebook_id)+"/"+seq_index+"_data_y.npy\"," +y_varible+')\n'
                    # validation_code += new_line0
                    # validation_code += new_line
                    # validation_code += new_line1
                if index == train_test_index:
                    validation_code += line
                    validation_code += '\n'
                    try:
                        x_train_varible , x_test_varible , y_train_varible , y_test_varible = line.split('=')[0].strip().split(',')
                        x_train_varible = x_train_varible.strip()
                        not_null_index = line.find(x_train_varible)
                        start_null = line[0:not_null_index]
                    except:
                        continue
                    # #print("....")
                    x_train_varible = x_train_varible.strip()
                    x_test_varible = x_test_varible.strip()
                    y_train_varible = y_train_varible.strip()
                    y_test_varible = y_test_varible.strip()
                    x_validation_varible = 'x_validation_varible'
                    y_validation_varible = 'y_validation_varible'
        

                    cross_validation_code = 'from sklearn.model_selection import cross_val_score\n'
                    cross_validation_code += 'cross_score = cross_val_score(model, ' + x_train_varible +', ' + y_train_varible + ',cv=4)\n'
                    # new_train_test_split = start_null + x_train_varible + ', ' + x_validation_varible + ', ' +  y_train_varible + ', ' +  y_validation_varible
                    # new_train_test_split = new_train_test_split + " = train_test_split(" + x_train_varible + ', ' + y_train_varible + ', train_size='+str(0.875)+', test_size='+str(0.125)+', random_state='+str(split_random_state)+')' + '\n'
                    # validation_code += new_train_test_split
                    
                elif 'model.fit(' in line  and index >= ours_index:
                    validation_code += '#'+line.replace(x_test_varible, x_validation_varible)
                    validation_code += '\n'
                elif 'model.predict(' in line and index >= ours_index:
                    validation_code += '#'+line.replace(x_test_varible, x_validation_varible)
                    validation_code += '\n'
                
                elif 'score = accuracy_score(' in line  and index >= ours_index:
                    validation_code += '#'+line.replace(y_test_varible, y_validation_varible)
                    validation_code += '\n'
                    validation_code += cross_validation_code
                elif 'np.save("HybridPipeGen/core/tmpdata/prenotebook_res/' in line:
                    validation_code += '#'+line.replace(y_test_varible, y_validation_varible)
                    validation_code += '\n'
                    validation_code += line.replace('prenotebook_res', 'cross_val_res').replace(': score }', ': cross_score }')
                    validation_code += '\n'
                else:
                    validation_code += line
                    validation_code += '\n'
            seq_code_dict['validation_code'] = validation_code
            # #print(seq_code_dict)
            #print('????????????????????????')
            if not os.path.exists('/home/yxm/staticfg-master/HybridPipeGen/core/tmpdata/cross_validation_code/'+notebook_id):
                os.mkdir('/home/yxm/staticfg-master/HybridPipeGen/core/tmpdata/cross_validation_code/'+notebook_id)
            if not os.path.exists('/home/yxm/staticfg-master/HybridPipeGen/core/tmpdata/cross_validation_code_py/'+notebook_id):
                os.mkdir('/home/yxm/staticfg-master/HybridPipeGen/core/tmpdata/cross_validation_code_py/'+notebook_id)
            with open('/home/yxm/staticfg-master/HybridPipeGen/core/tmpdata/cross_validation_code/'+notebook_id+'/'+seq_file, 'w') as f:
                json.dump(seq_code_dict, f)
            with open('/home/yxm/staticfg-master/HybridPipeGen/core/tmpdata/cross_validation_code_py/'+notebook_id+'/'+seq_file.replace('.json', '.py'), 'w') as f:
                f.write(validation_code)
        # break
                        
    

    
def clean_kuohao(code):
    lines = code.split('\n')
    left_num = 0
    right_num = 0
    left_z_num = 0
    right_z_num = 0
    str_ = ''
    is_in_beizhu = False
    for line in lines:
        num_p = 0
        num_pp = 0
        if line.count('"""') % 2 == 1 or line.count("'''") %2 == 1:
            is_in_beizhu = not is_in_beizhu
        
        if not is_in_beizhu:
            for index,char in enumerate(line):
                if num_pp % 2 ==0 and num_p % 2 ==0:
                    if char == '#':
                        continue
                if char == '"':
                    if index > 0:
                        if line[index-1] != '\\':
                            if num_p % 2 ==0:
                                num_pp += 1
                    else:
                        if num_p % 2 ==0:
                                num_pp += 1
                if char == "'":
                    if index > 0:
                        if line[index-1] != '\\':
                            if num_pp % 2 ==0:
                                num_p += 1
                    else:
                        if num_pp % 2 ==0:
                                num_p += 1
                if char == '(':
                    if num_pp %2 ==0 and num_p %2 == 0:
                        left_num += 1
                if char == ')':
                    if num_pp %2 ==0 and num_p %2 == 0:
                        right_num += 1
                if char == '[':
                    if num_pp %2 ==0 and num_p %2 == 0:
                        left_z_num += 1
                if char == ']':
                    if num_pp %2 ==0 and num_p %2 == 0:
                        right_z_num += 1
                    
        if left_num == right_num and left_z_num == right_z_num:
            str_ += line
            str_ += '\n'
            left_num =0
            right_num = 0
        else:
            str_ += line
    return str_
def transform_validation_code_new(running_id = 1,split_random_state=0):
    # notebooks = os.listdir('merge_code_tmpdata_1600')
    # notebooks.sort()
    with open('need_rerun.json','r') as f:
        notebooks = json.load(f)
    exist_f = open("/home/yxm/staticfg-master/all_s.txt", "r")
    exist = exist_f.readlines()
    exitst_ = [x.strip("\n") for x in exist]
    # notebooks = list(set(notebooks) & set(exitst_))
    #print(len(notebooks))
    for notebook_id in notebooks:
        if notebook_id != '351382':
            continue
        if notebook_id == 'tmpfile':
            continue
        try:
            seq_files = os.listdir('merge_code_tmpdata_1600/'+notebook_id)
        except:
            continue
        need_continue=False
        for seq_file in seq_files:
            # if os.path.exists('/home/yxm/staticfg-master/HybridPipeGen/core/tmpdata/merge_validation_code_new/'+notebook_id+'/'+seq_file.replace('.json', '.py')):
                # continue
            seq_index = seq_file.split('.')[0]
            # #print(notebook_id)
            # exist_f = open("/home/yxm/staticfg-master/test_error.txt", "r")
            # exist = exist_f.readlines()
            # exitst_ = [x.strip("\n") for x in exist]
            # if notebook_id not in exitst_:
                # continue
            # #print(seq_file)
            with open('merge_code_tmpdata_1600/'+notebook_id+'/'+seq_file, 'r') as f:
                seq_code_dict = json.load(f)
            test_code = seq_code_dict['code']
            test_code = cleaning_origin(test_code)
            validation_code = ''
            test_code_list = test_code.split('\n')

            train_test_index = 0
            for index, line in enumerate(test_code_list):
                if '=train_test_split(' in line or ' train_test_split(' in line:
                    if '=' not in line:
                        continue
                    train_test_index = index
                    # #print(line)
                    # x_train_varible , x_test_varible , y_train_varible , y_test_varible = line.split('=')[0].strip().split(',')
                    train_test_split_code = line.split('=')[1].strip()
                    kuohao = train_test_split_code.find('(')
                    train_test_split_code = train_test_split_code[kuohao+1:]
                    arglist = train_test_split_code.split(",")
                    # #print(arglist)
                    try:
                        x_varible = arglist[0].strip()
                        y_varible = arglist[1].strip()
                    except:
                        need_modify.append(notebook_id)
                        np.save('need_modify.npy', need_modify)
                        need_continue = True
                if "#print(\"start running model training........\")" == line:
                    ours_index = index
            if need_continue:
                break
            # #print(train_test_index)
            for index, line in enumerate(test_code_list):
                if index == train_test_index-1:
                    validation_code += line
                    validation_code += '\n'
                    # new_line0 = 'import numpy as np\n'
                    # new_line = 'np.save("'+"/home/yxm/staticfg-master/HybridPipeGen/core/tmpdata/merge_result_data/"+str(notebook_id)+"/"+seq_index+"_data_x.npy\"," +x_varible+')\n'
                    # new_line1 = 'np.save("'+"/home/yxm/staticfg-master/HybridPipeGen/core/tmpdata/merge_result_data/"+str(notebook_id)+"/"+seq_index+"_data_y.npy\"," +y_varible+')\n'
                    # validation_code += new_line0
                    # validation_code += new_line
                    # validation_code += new_line1
                if index == train_test_index:
                    validation_code += line
                    validation_code += '\n'
                    try:
                        x_train_varible , x_test_varible , y_train_varible , y_test_varible = line.split('=')[0].strip().split(',')
                        x_train_varible = x_train_varible.strip()
                        not_null_index = line.find(x_train_varible)
                        start_null = line[0:not_null_index]
                    except:
                        continue
                    x_train_varible = x_train_varible.strip()
                    x_test_varible = x_test_varible.strip()
                    y_train_varible = y_train_varible.strip()
                    y_test_varible = y_test_varible.strip()
                    x_validation_varible = x_test_varible
                    y_validation_varible = y_test_varible
        
                    new_train_test_split = start_null + x_train_varible + ', ' + x_validation_varible + ', ' +  y_train_varible + ', ' +  y_validation_varible
                    new_train_test_split = new_train_test_split + " = train_test_split(" + x_train_varible + ', ' + y_train_varible + ', train_size='+str(0.875)+', test_size='+str(0.125)+', random_state='+str(split_random_state)+')' + '\n'
                    validation_code += new_train_test_split
                    

                # elif 'model.predict(' in line and index >= ours_index:
                #     validation_code += line.replace(x_test_varible, x_validation_varible)
                #     validation_code += '\n'
                
                # elif 'score = accuracy_score(' in line  and index >= ours_index:
                #     validation_code += line.replace(y_test_varible, y_validation_varible)
                #     validation_code += '\n'
                
                else:
                    validation_code += line
                    validation_code += '\n'


            seq_code_dict['validation_code'] = validation_code
            if not os.path.exists('/home/yxm/staticfg-master/HybridPipeGen/core/tmpdata/merge_validation_code_1/'+notebook_id):
                os.mkdir('/home/yxm/staticfg-master/HybridPipeGen/core/tmpdata/merge_validation_code_1/'+notebook_id)
            if not os.path.exists('/home/yxm/staticfg-master/HybridPipeGen/core/tmpdata/merge_validation_code_py_1/'+notebook_id):
                os.mkdir('/home/yxm/staticfg-master/HybridPipeGen/core/tmpdata/merge_validation_code_py_1/'+notebook_id)
            with open('/home/yxm/staticfg-master/HybridPipeGen/core/tmpdata/merge_validation_code_1/'+notebook_id+'/'+seq_file, 'w') as f:
                json.dump(seq_code_dict, f)
            with open('/home/yxm/staticfg-master/HybridPipeGen/core/tmpdata/merge_validation_code_py_1/'+notebook_id+'/'+seq_file.replace('.json', '.py'), 'w') as f:
                f.write(validation_code)
def transform_origin_validation_code_new(split_random_state=0, running_id=1):
    # notebooks = os.listdir('/home/yxm/staticfg-master/HybridPipeGen/core/tmpdata/prenotebook_code')
    # notebooks.sort()
    with open('need_rerun.json','r') as f:
        notebooks = json.load(f)
    exist_f = open("/home/yxm/staticfg-master/all_s.txt", "r")
    exist = exist_f.readlines()
    exitst_ = [x.strip("\n") for x in exist]
    notebooks = list(set(notebooks) & set(exitst_))
    #print(len(notebooks))
    for notebook_id in notebooks:
        # notebook_id = int(notebook_id_py.split('.')[0])
        # notebook_id = str(notebook_id_py.split('.')[0])
        notebook_id_py  = notebook_id + '.py'
        # #print(notebook_id)
        
        # exist_f = open("/home/yxm/staticfg-master/test_error.txt", "r")
        # exist = exist_f.readlines()
        # exitst_ = [x.strip("\n") for x in exist]
        # if notebook_id not in exitst_:
        #     continue
        
        # if os.path.exists('/home/yxm/staticfg-master/HybridPipeGen/core/tmpdata/merge_validation_code_new/'+str(notebook_id)+'/origin.json'):
            # continue
        with open("/home/yxm/staticfg-master/HybridPipeGen/core/tmpdata/prenotebook_code/" + notebook_id_py, 'r') as f:
            test_code = f.read()
        # #print("------------1----------------------------------------")
        # #print(test_code)
        test_code = cleaning_origin(test_code)
        # #print("------------2----------------------------------------")
        # #print(test_code)
        
        validation_code = ''
        test_code_list = test_code.split('\n')
        seq_code_dict = {}
        ours_index =0
        test_num = 0
        #print("------------3----------------------------------------")
        for index, line in enumerate(test_code_list):
            if 'train_test_split(' in line:
                if '=' not in line:
                    continue
                train_test_index = index
                #print(line)
                # x_train_varible , x_test_varible , y_train_varible , y_test_varible = line.split('=')[0].strip().split(',')
                train_test_split_code = line.split('=')[1].strip()
                kuohao = train_test_split_code.find('(')
                train_test_split_code = train_test_split_code[kuohao+1:]
                arglist = train_test_split_code.split(",")
                #print(arglist)
                x_varible = arglist[0].strip()
                y_varible = arglist[1].strip()
            if "#print(\"start running model training........\")" == line:
                ours_index = index
        #print(ours_index)
        #print("------------4----------------------------------------")

        for index, line in enumerate(test_code_list):
            # if index < train_test_index -1:
            #     continue
            if index == train_test_index-1:
                validation_code += line
                validation_code += '\n'
                # new_line0 = 'import numpy as np\n'
                # new_line = 'np.save("'+"merge_result_data/"+str(notebook_id)+"/"+"origin_data_x.npy\"," +x_varible+')\n'
                # new_line1 = 'np.save("'+"merge_result_data/"+str(notebook_id)+"/"+"origin_data_y.npy\"," +y_varible+')\n'
                # new_line = 'np.save("'+"/home/yxm/staticfg-master/HybridPipeGen/core/tmpdata/merge_result_data/"+str(notebook_id)+"/"+"origin_data_x.npy\"," +x_varible+')\n'
                # new_line1 = 'np.save("'+"/home/yxm/staticfg-master/HybridPipeGen/core/tmpdata/merge_result_data/"+str(notebook_id)+"/"+"origin_data_y.npy\"," +y_varible+')\n'
                # validation_code += new_line0
                # validation_code += new_line
                # validation_code += new_line1
            if index == train_test_index:
                validation_code += line
                validation_code += '\n'
                # #print('index', line)
                try:
                    x_train_varible , x_test_varible , y_train_varible , y_test_varible = line.split('=')[0].strip().split(',')
                    x_train_varible = x_train_varible.strip()
                    not_null_index = line.find(x_train_varible)
                    start_null = line[0:not_null_index]
                except:
                    continue
                x_train_varible = x_train_varible.strip()
                x_test_varible = x_test_varible.strip()
                y_train_varible = y_train_varible.strip()
                y_test_varible = y_test_varible.strip()
                x_validation_varible = x_test_varible
                y_validation_varible = y_test_varible
    
                new_train_test_split = start_null + x_train_varible + ', ' + x_validation_varible + ', ' +  y_train_varible + ', ' +  y_validation_varible
                new_train_test_split = new_train_test_split + " = train_test_split(" + x_train_varible + ', ' + y_train_varible + ', train_size='+str(0.875)+', test_size='+str(0.125)+', random_state='+str(split_random_state)+')' + '\n'
                validation_code += new_train_test_split
                    

            # elif index > ours_index and 'model.predict' in line:
            #     # #print(index)
            #     validation_code += line.replace(x_test_varible, x_validation_varible)
            #     validation_code += '\n'
            
            # elif  index > ours_index and 'accuracy_score(' in line:
            #     validation_code += line.replace(y_test_varible, y_validation_varible)
            #     validation_code += '\n'
            
            else:
                validation_code += line
                validation_code += '\n'
        #     if index > train_test_index and index < ours_index:
        #         # #print(x_test_varible)
        #         if x_test_varible in line:
        #             test_num +=1
        # if test_num > 0:
        #     res.append(notebook_id)
        # #print(res)
        # #print(len(res))
        # with open("test_error.txt","w")as f:
        #     for x in res:
        #         f.write(x+"\n")

        # validation_code = validation_code.replace("prenotebook_res", 'validation_prenotebook_res')
        # #print(validation_code)
        # validation_code = clean_kuohao(validation_code)
        seq_code_dict['validation_code'] = validation_code
        notebook_id= str(notebook_id)
        # if not os.path.exists('merge_validation_code_'+str(running_id)+'/'+notebook_id):
        #     os.mkdir('merge_validation_code_'+str(running_id)+'/'+notebook_id)
        # if not os.path.exists('merge_validation_code_py_'+str(running_id)+'/'+notebook_id):
        #     os.mkdir('merge_validation_code_py_'+str(running_id)+'/'+notebook_id)
        # with open('merge_validation_code_'+str(running_id)+'/'+notebook_id+'/origin.json', 'w') as f:
        #     json.dump(seq_code_dict, f)
        # with open('merge_validation_code_py_'+str(running_id)+'/'+notebook_id+'/origin.py', 'w') as f:
        #     f.write(validation_code)
        if not os.path.exists('/home/yxm/staticfg-master/HybridPipeGen/core/tmpdata/merge_validation_code_1'+'/'+notebook_id):
            os.mkdir('/home/yxm/staticfg-master/HybridPipeGen/core/tmpdata/merge_validation_code_1'+'/'+notebook_id)
        if not os.path.exists('/home/yxm/staticfg-master/HybridPipeGen/core/tmpdata/merge_validation_code_py_1'+'/'+notebook_id):
            os.mkdir('/home/yxm/staticfg-master/HybridPipeGen/core/tmpdata/merge_validation_code_py_1'+'/'+notebook_id)
        with open('/home/yxm/staticfg-master/HybridPipeGen/core/tmpdata/merge_validation_code_1'+'/'+notebook_id+'/origin.json', 'w') as f:
            json.dump(seq_code_dict, f)
        with open('/home/yxm/staticfg-master/HybridPipeGen/core/tmpdata/merge_validation_code_py_1'+'/'+notebook_id+'/origin.py', 'w') as f:
            f.write(validation_code)
def create_base_code():
    with open('task_all.json','r') as f:
        tasks = json.load(f)
    exist_f = open("/home/yxm/staticfg-master/all_same.txt", "r")
    exist = exist_f.readlines()
    exitst_ = [x.strip("\n") for x in exist]
    old_path = os.listdir('prenotebook_code')
    res = []
    # notebooks = list(set(notebooks) & set(exitst_))
    for task in tasks:
        # if task != '22000-scotch-whisky-reviews_LogisticRegression_2':
        #     continue
        notebook_id = list(tasks[task]['notebook_list'])[0]
        notebook_id_py = notebook_id +'.py'

        if notebook_id in exitst_:
            # #print(notebook_id)
            # if notebook_id != 'azsadic_churn-prediction-for-banking':
            #     continue
            # #print('???')
            read_index = -1
            train_test_index = -1
            ours_index = -1
            seq_code_dict = {}
            with open("/home/yxm/staticfg-master/HybridPipeGen/core/tmpdata/prenotebook_code/" + notebook_id_py, 'r') as f:
                test_code = f.read()
            test_code = cleaning_origin(test_code)
            test_code_list = test_code.split('\n')
            base_code = ''
            for index, line in enumerate(test_code_list):
                if 'read_csv(' in line and '=' in line: 
                    read_index = index
                if 'train_test_split(' in line:
                    if '=' not in line:
                        continue
                    train_test_index = index
                    #print(line)
                    try:
                        x_train_varible , x_test_varible , y_train_varible , y_test_varible = line.split('=')[0].strip().split(',')
                    except:
                        continue
                    # train_test_split_code = line.split('=')[1].strip()
                    # kuohao = train_test_split_code.find('(')
                    # train_test_split_code = train_test_split_code[kuohao+1:]
                    # arglist = train_test_split_code.split(",")
                    # #print(arglist)
                    # x_varible = arglist[0].strip()
                    # y_varible = arglist[1].strip()
                if "#print(\"start running model training........\")" == line:
                    ours_index = index
            df = ''
            for index, line in enumerate(test_code_list):
                # if index < read_index:
                #     base_code += line
                #     base_code +='\n'
                if index  == read_index:
                    df = line.split('=')[0].strip()
                    if 'index_col' in line:
                        res.append(notebook_id)
                        #print(line)
                    #     line =  line.split(',')[0] + ',index_col=False)'
                    # else:
                    #     line = line.split(')')[0]+',index_col=False)'
                    # base_code += line+'\n'
                    # base_code += 'import json' + '\n'
                    # base_code += 'import pandas as pd' + '\n'
                    # base_code += 'import numpy as np' + '\n'
                    # base_code += 'from sklearn.preprocessing import OneHotEncoder, LabelEncoder' +'\n'
                    # base_code += 'from sklearn.impute import SimpleImputer, MissingIndicator' +'\n'
                    # base_code += 'dataset = pd.DataFrame(' + df +')'+'\n'
                    # base_code += 'with open(\'task_all.json\',\'r\') as f:' + '\n'
                    # base_code += '    tasks = json.load(f)' + '\n'
                    # base_code += 'notebook_id = list(tasks[\''+task+'\'][\'notebook_list\'])[0]' + '\n'
                    # base_code += 'with open(\"../statsklearn/notebook_all.json\", \'r\') as f:' + '\n'
                    # base_code += '    notebook_column_info = json.load(f)' + '\n'
                    # base_code += 'column_index = notebook_column_info[str(notebook_id)][\'index\'][0]' + '\n'
                    # base_code += 'for column in notebook_column_info[str(notebook_id)][\'column_index\']:' + '\n'
                    # base_code += '    if notebook_column_info[str(notebook_id)][\'column_index\'][column] ==column_index:' +'\n'
                    # base_code += '        label_column_name = column'+'\n'
                    # base_code += 'label_column_name = dataset.columns[column_index]' + '\n'
                    # base_code += '#print(label_column_name)' + '\n'
                    # base_code += 'dataset.dropna(subset=[label_column_name])' + '\n'
                    # base_code += 'label_col = dataset.pop(label_column_name).astype(\'str\')' + '\n'
                    # base_code += 'if str(label_col.dtype) == \'object\':' + '\n'
                    # base_code += '    label_encoder = LabelEncoder()' + '\n'
                    # base_code += '    label_col = label_encoder.fit_transform(label_col)' + '\n'
                    # base_code += '#print(label_col.shape)'+'\n'
                    # base_code += 'imp=SimpleImputer(strategy="most_frequent")' + '\n'
                    # base_code += 'cols = dataset.columns' + '\n'
                    # base_code += 'num_cols = dataset._get_numeric_data().columns' + '\n'
                    # base_code += 'cat_cols = list(set(cols) - set(num_cols))' + '\n'
                    # base_code += 'if len(num_cols) > 0:' + '\n'
                    # base_code += '    dataset_num = pd.DataFrame(imp.fit_transform(dataset[num_cols]), columns=num_cols).reset_index(drop=True).infer_objects()' + '\n'
                    # base_code += '    if len(cat_cols) > 0:' + '\n'
                    # base_code += '        dataset_cat = dataset[cat_cols].fillna(\'NaN\')' + '\n'
                    # base_code += '        dataset = pd.concat([dataset_num, dataset_cat], axis=1)' + '\n'
                    # base_code += '    else:' + '\n'
                    # base_code += '        dataset = dataset_num' + '\n'
                    # base_code += 'else:' + '\n'
                    # base_code += '    if len(cat_cols) > 0:' + '\n'
                    # base_code += '        dataset = dataset[cat_cols].fillna(\'NaN\')' + '\n'
                    # base_code += 'num_cols = dataset._get_numeric_data().columns' + '\n'
                    # base_code += 'cat_cols = list(set(cols) - set(num_cols))' + '\n'
                    # base_code += 'encoder = False' + '\n'
                    # base_code += 'if len(cat_cols)!= 0:' + '\n'
                    # base_code += '    encoder = True' + '\n'
                    # base_code += '    for col_name in cat_cols:' + '\n'
                    # base_code += '        column = dataset.pop(col_name).astype(\'str\')' + '\n'
                    # base_code += '        label_encoder = LabelEncoder()' + '\n'
                    # base_code += '        column = label_encoder.fit_transform(column)' + '\n'
                    # base_code += '        dataset.insert(loc=len(dataset.columns), column=col_name, value=column)' + '\n'
                    # base_code += 'dataset.insert(loc=len(dataset.columns), column=label_column_name, value=label_col)' + '\n'

                    # temp_x = 'x = dataset'
                    # temp_y = 'y = label_col'
                    # temp_x = 'x = dataset.drop(dataset.columns[[' + str(tasks[task]['label']) + ']],axis=1)'
                    # temp_y = 'y = dataset.iloc[:,' + str(tasks[task]['label'])+']'
                    # #print(temp_x)
                    # #print(temp_y)
                    
                    # base_code += 'import numpy as np\n'
                    # base_code += 'import pandas as pd\n'
                    # base_code += temp_x + '\n'
                    # base_code += temp_y + '\n'
                    # base_code += '#print(y.shape)\n'
                # if index == train_test_index:
                #     temp_split = x_train_varible + ', ' + x_test_varible + ', ' +  y_train_varible + ', ' +  y_test_varible
                #     base_code +='from sklearn.model_selection import train_test_split\n'
                #     temp_split = temp_split + ' = train_test_split(x, y, train_size=0.8, test_size=1-0.8, random_state=0)'+ '\n'
                #     #print(temp_split)
                #     base_code += temp_split
                # if index >= ours_index - 2 and index <= ours_index + 6:
                #     base_code +=line+'\n'
                # seq_code_dict['code'] = base_code
                # with open('/home/yxm/staticfg-master/HybridPipeGen/core/tmpdata/base_code_add_ai'+'/'+task+'.json', 'w') as f:
                #     json.dump(seq_code_dict, f)
                # with open('/home/yxm/staticfg-master/HybridPipeGen/core/tmpdata/base_code_add_ai_py'+'/'+task+'.py', 'w') as f:
                #     f.write(base_code)
        
        if notebook_id_py in old_path:
            read_index = 99999999999
            train_test_index = -1
            ours_index = -1
            seq_code_dict = {}
            with open("/home/yxm/staticfg-master/prenotebook_code/" + notebook_id_py, 'r') as f:
                test_code = f.read()
            test_code = cleaning_origin(test_code)
            test_code_list = test_code.split('\n')
            base_code = ''
            for index, line in enumerate(test_code_list):
                if 'read_csv(' in line and '=' in line and index < read_index: 
                    read_index = index
                if 'train_test_split(' in line:
                    if '=' not in line:
                        continue
                    train_test_index = index
                    #print(line)
                    try:
                        x_train_varible , x_test_varible , y_train_varible , y_test_varible = line.split('=')[0].strip().split(',')
                    except:
                        continue
                if 'np.save(\"prenotebook_res' in line:
                    ours_index = index
            df = ''
            for index, line in enumerate(test_code_list):
                if index < read_index:
                    base_code += line
                    base_code +='\n'
                if index  == read_index:
                    df = line.split('=')[0].strip()
                    if 'index_col' in line:
                        res.append(notebook_id)
                        #print(line)
                    # if 'index_col' in line:
                    #     line =  line.split(',')[0] + ',index_col=False)'
                    # else:
                    #     line = line.split(')')[0]+',index_col=False)'
                    # base_code += line+'\n'
                    # base_code += 'import json' + '\n'
                    # base_code += 'import pandas as pd' + '\n'
                    # base_code += 'import numpy as np' + '\n'
                    # base_code += 'from sklearn.preprocessing import OneHotEncoder, LabelEncoder' +'\n'
                    # base_code += 'from sklearn.impute import SimpleImputer, MissingIndicator' +'\n'
                    # base_code += 'dataset = pd.DataFrame(' + df +')'+'\n'
                    # base_code += 'with open(\'task_all.json\',\'r\') as f:' + '\n'
                    # base_code += '    tasks = json.load(f)' + '\n'
                    # base_code += 'notebook_id = list(tasks[\''+task+'\'][\'notebook_list\'])[0]' + '\n'
                    # base_code += 'with open(\"../statsklearn/notebook_all.json\", \'r\') as f:' + '\n'
                    # base_code += '    notebook_column_info = json.load(f)' + '\n'
                    # base_code += 'column_index = notebook_column_info[str(notebook_id)][\'index\'][0]' + '\n'
                    # base_code += 'for column in notebook_column_info[str(notebook_id)][\'column_index\']:' + '\n'
                    # base_code += '    if notebook_column_info[str(notebook_id)][\'column_index\'][column] ==column_index:' +'\n'
                    # base_code += '        label_column_name = column'+'\n'
                    # base_code += 'label_column_name = dataset.columns[column_index]' + '\n'
                    # base_code += '#print(label_column_name)' + '\n'
                    # base_code += 'dataset.dropna(subset=[label_column_name])' + '\n'
                    # base_code += 'label_col = dataset.pop(label_column_name).astype(\'str\')' + '\n'
                    # base_code += 'if str(label_col.dtype) == \'object\':' + '\n'
                    # base_code += '    label_encoder = LabelEncoder()' + '\n'
                    # base_code += '    label_col = label_encoder.fit_transform(label_col)' + '\n'
                    # base_code += '#print(label_col.shape)'+'\n'
                    # base_code += 'imp=SimpleImputer(strategy="most_frequent")' + '\n'
                    # base_code += 'cols = dataset.columns' + '\n'
                    # base_code += 'num_cols = dataset._get_numeric_data().columns' + '\n'
                    # base_code += 'cat_cols = list(set(cols) - set(num_cols))' + '\n'
                    # base_code += 'if len(num_cols) > 0:' + '\n'
                    # base_code += '    dataset_num = pd.DataFrame(imp.fit_transform(dataset[num_cols]), columns=num_cols).reset_index(drop=True).infer_objects()' + '\n'
                    # base_code += '    if len(cat_cols) > 0:' + '\n'
                    # base_code += '        dataset_cat = dataset[cat_cols].fillna(\'NaN\')' + '\n'
                    # base_code += '        dataset = pd.concat([dataset_num, dataset_cat], axis=1)' + '\n'
                    # base_code += '    else:' + '\n'
                    # base_code += '        dataset = dataset_num' + '\n'
                    # base_code += 'else:' + '\n'
                    # base_code += '    if len(cat_cols) > 0:' + '\n'
                    # base_code += '        dataset = dataset[cat_cols].fillna(\'NaN\')' + '\n'
                    # base_code += 'num_cols = dataset._get_numeric_data().columns' + '\n'
                    # base_code += 'cat_cols = list(set(cols) - set(num_cols))' + '\n'
                    # base_code += 'encoder = False' + '\n'
                    # base_code += 'if len(cat_cols)!= 0:' + '\n'
                    # base_code += '    encoder = True' + '\n'
                    # base_code += '    for col_name in cat_cols:' + '\n'
                    # base_code += '        column = dataset.pop(col_name).astype(\'str\')' + '\n'
                    # base_code += '        label_encoder = LabelEncoder()' + '\n'
                    # base_code += '        column = label_encoder.fit_transform(column)' + '\n'
                    # base_code += '        dataset.insert(loc=len(dataset.columns), column=col_name, value=column)' + '\n'
                    # base_code += 'dataset.insert(loc=len(dataset.columns), column=label_column_name, value=label_col)' + '\n'

                    # temp_x = 'x = dataset'
                    # temp_y = 'y = label_col'
                    # temp_x = 'x = dataset.drop(dataset.columns[[' + str(tasks[task]['label']) + ']],axis=1)'
                    # temp_y = 'y = dataset.iloc[:,' + str(tasks[task]['label'])+']'
                    # #print(temp_x)
                    # #print(temp_y)
                    
                    # base_code += 'import numpy as np\n'
                    # base_code += 'import pandas as pd\n'
                    # base_code += temp_x + '\n'
                    # base_code += temp_y + '\n'
                    # base_code += '#print(y.shape)\n'
                # if index == train_test_index:
                #     temp_split = x_train_varible + ', ' + x_test_varible + ', ' +  y_train_varible + ', ' +  y_test_varible
                #     base_code +='from sklearn.model_selection import train_test_split\n'
                #     temp_split = temp_split + ' = train_test_split(x, y, train_size=0.8, test_size=1-0.8, random_state=0)'+ '\n'
                #     #print(temp_split)
                #     base_code += temp_split
                # #print(base_code)
                # #print(ours_index)
                # if index >= ours_index - 7:
                #     base_code +=line+'\n'
                # seq_code_dict['code'] = base_code
                # with open('/home/yxm/staticfg-master/HybridPipeGen/core/tmpdata/base_code_add_ai'+'/'+task+'.json', 'w') as f:
                #     json.dump(seq_code_dict, f)
                # with open('/home/yxm/staticfg-master/HybridPipeGen/core/tmpdata/base_code_add_ai_py'+'/'+task+'.py', 'w') as f:
                #     f.write(base_code)   
    #print(res)
    #print(len(res))
def create_base_code_all():
    with open('task_all.json','r') as f:
        tasks = json.load(f)
    exist_f = open("/home/yxm/staticfg-master/all_same.txt", "r")
    exist = exist_f.readlines()
    exitst_ = [x.strip("\n") for x in exist]
    old_path = os.listdir('prenotebook_code')
    # notebooks = list(set(notebooks) & set(exitst_))
    res =[]
    for task in tasks:
        # if task != 'arshid_iris-flower-dataset_RandomForestClassifier_4':
        #     continue
        notebook_id = list(tasks[task]['notebook_list'])[0]
        notebook_id_py = notebook_id +'.py'

        if notebook_id in exitst_ :
            #print(notebook_id)
            # if notebook_id != 'azsadic_churn-prediction-for-banking':
            #     continue
            # #print('???')
            # read_index = 999999999
            train_test_index = -1
            ours_index = -1
            seq_code_dict = {}
            with open("/home/yxm/staticfg-master/HybridPipeGen/core/tmpdata/prenotebook_code/" + notebook_id_py, 'r') as f:
                test_code = f.read()
            test_code = cleaning_origin(test_code)
            test_code_list = test_code.split('\n')
            base_code = ''
            # for index, line in enumerate(test_code_list):
            #     # if 'read_csv(' in line and '=' in line and index < read_index: 
            #     #     read_index = index
            #     if 'train_test_split(' in line:
            #         if '=' not in line:
            #             continue
            #         train_test_index = index
            #         # #print(line)
            #         try:
            #             x_train_varible , x_test_varible , y_train_varible , y_test_varible = line.split('=')[0].strip().split(',')
            #         except:
            #             continue
            #         # train_test_split_code = line.split('=')[1].strip()
            #         # kuohao = train_test_split_code.find('(')
            #         # train_test_split_code = train_test_split_code[kuohao+1:]
            #         # arglist = train_test_split_code.split(",")
            #         # #print(arglist)
            #         # x_varible = arglist[0].strip()
            #         # y_varible = arglist[1].strip()
            #     if "#print(\"start running model training........\")" == line:
            #         ours_index = index
            df = ''
            # for index, line in enumerate(test_code_list):
                # if index < read_index:
                #     base_code += line
                #     base_code +='\n'
                # if index  == read_index:
                # df = line.split('=')[0].strip()
                # if 'index_col' in line:
                #     #print(line)
                #     res.append(task)
                #     line =  line.split(',')[0] + ',index_col=False)'
                # else:
                #     line = line.split(')')[0]+',index_col=False)'
            base_code += 'import json' + '\n'
            base_code += 'import pandas as pd' + '\n'
            base_code += 'import numpy as np' + '\n'
            # datasets = os.listdir('KGTorrent/dataset/'+tasks[task]['dataset']) 
            dataset_name = tasks[task]['dataset_file']
            base_code += 'df = pd.read_csv(\''+'/home/yxm/KGTorrent/dataset/'+tasks[task]['dataset']+'/'+dataset_name+'\')'+'\n'
            df = 'df'
            
            base_code += 'from sklearn.preprocessing import OneHotEncoder, LabelEncoder' +'\n'
            base_code += 'from sklearn.impute import SimpleImputer, MissingIndicator' +'\n'
            base_code += 'dataset = pd.DataFrame(' + df +')'+'\n'
            base_code += 'with open(\'task_all.json\',\'r\') as f:' + '\n'
            base_code += '    tasks = json.load(f)' + '\n'
            base_code += 'notebook_id = list(tasks[\''+task+'\'][\'notebook_list\'])[0]' + '\n'
            base_code += 'with open(\"../statsklearn/notebook_all.json\", \'r\') as f:' + '\n'
            base_code += '    notebook_column_info = json.load(f)' + '\n'
            base_code += 'column_index = notebook_column_info[str(notebook_id)][\'index\'][0]' + '\n'
            base_code += 'for column in notebook_column_info[str(notebook_id)][\'column_index\']:' + '\n'
            base_code += '    if notebook_column_info[str(notebook_id)][\'column_index\'][column] ==column_index:' +'\n'
            base_code += '        label_column_name = column'+'\n'
            # base_code += 'label_column_name = dataset.columns[column_index]' + '\n'
            # base_code += '#print(label_column_name)' + '\n'
            base_code += 'dataset.dropna(subset=[label_column_name])' + '\n'
            base_code += 'label_col = dataset.pop(label_column_name).astype(\'str\')' + '\n'
            base_code += 'if str(label_col.dtype) == \'object\':' + '\n'
            base_code += '    label_encoder = LabelEncoder()' + '\n'
            base_code += '    label_col = label_encoder.fit_transform(label_col)' + '\n'
            # base_code += '#print(label_col.shape)'+'\n'
            base_code += 'imp=SimpleImputer(strategy="most_frequent")' + '\n'
            base_code += 'cols = dataset.columns' + '\n'
            base_code += 'num_cols = dataset._get_numeric_data().columns' + '\n'
            base_code += 'cat_cols = list(set(cols) - set(num_cols))' + '\n'
            base_code += 'if len(num_cols) > 0:' + '\n'
            base_code += '    dataset_num = pd.DataFrame(imp.fit_transform(dataset[num_cols]), columns=num_cols).reset_index(drop=True).infer_objects()' + '\n'
            base_code += '    if len(cat_cols) > 0:' + '\n'
            base_code += '        dataset_cat = dataset[cat_cols].fillna(\'NaN\')' + '\n'
            base_code += '        dataset = pd.concat([dataset_num, dataset_cat], axis=1)' + '\n'
            base_code += '    else:' + '\n'
            base_code += '        dataset = dataset_num' + '\n'
            base_code += 'else:' + '\n'
            base_code += '    if len(cat_cols) > 0:' + '\n'
            base_code += '        dataset = dataset[cat_cols].fillna(\'NaN\')' + '\n'
            base_code += 'num_cols = dataset._get_numeric_data().columns' + '\n'
            base_code += 'cat_cols = list(set(cols) - set(num_cols))' + '\n'
            base_code += 'encoder = False' + '\n'
            base_code += 'if len(cat_cols)!= 0:' + '\n'
            base_code += '    encoder = True' + '\n'
            base_code += '    for col_name in cat_cols:' + '\n'
            base_code += '        column = dataset.pop(col_name).astype(\'str\')' + '\n'
            base_code += '        label_encoder = LabelEncoder()' + '\n'
            base_code += '        column = label_encoder.fit_transform(column)' + '\n'
            base_code += '        dataset.insert(loc=len(dataset.columns), column=col_name, value=column)' + '\n'
            # base_code += 'dataset.insert(loc=len(dataset.columns), column=label_column_name, value=label_col)' + '\n'

            temp_x = 'x = dataset'
            temp_y = 'y = label_col'
            # temp_x = 'x = dataset.drop(dataset.columns[[' + str(tasks[task]['label']) + ']],axis=1)'
            # temp_y = 'y = dataset.iloc[:,' + str(tasks[task]['label'])+']'
            # #print(temp_x)
            # #print(temp_y)
            
            # base_code += 'import numpy as np\n'
            # base_code += 'import pandas as pd\n'
            base_code += temp_x + '\n'
            base_code += temp_y + '\n'
                # base_code += '#print(y.shape)\n'
            # if index == train_test_index:
            #     temp_split = x_train_varible + ', ' + x_test_varible + ', ' +  y_train_varible + ', ' +  y_test_varible
            #     base_code +='from sklearn.model_selection import train_test_split\n'
            #     temp_split = temp_split + ' = train_test_split(x, y, train_size=0.8, test_size=1-0.8, random_state=0)'+ '\n'
            #     # #print(temp_split)
            #     base_code += temp_split
            # if index >= ours_index - 2 and index <= ours_index + 6:
            #     base_code +=line+'\n'
            seq_code_dict['code'] = base_code
            with open('/home/yxm/staticfg-master/HybridPipeGen/core/tmpdata/base_code_add_ai'+'/'+task+'.json', 'w') as f:
                json.dump(seq_code_dict, f)
            with open('/home/yxm/staticfg-master/HybridPipeGen/core/tmpdata/base_code_add_ai_py'+'/'+task+'.py', 'w') as f:
                f.write(base_code) 
        # if notebook_id_py in old_path:
        #     seq_code_dict = {}
        #     base_code = ''
        #     base_code += 'import json' + '\n'
        #     base_code += 'import pandas as pd' + '\n'
        #     base_code += 'import numpy as np' + '\n'
        #     # datasets = os.listdir('KGTorrent/dataset/'+tasks[task]['dataset']) 
        #     dataset_name = tasks[task]['dataset_file']
        #     base_code += 'df = pd.read_csv(\''+'/home/datamanager/dataset/statsklearn/dataset/'+tasks[task]['dataset']+'/'+dataset_name+'\')'+'\n'
        #     df = 'df'
            
        #     base_code += 'from sklearn.preprocessing import OneHotEncoder, LabelEncoder' +'\n'
        #     base_code += 'from sklearn.impute import SimpleImputer, MissingIndicator' +'\n'
        #     base_code += 'dataset = pd.DataFrame(' + df +')'+'\n'
        #     base_code += 'with open(\'task_all.json\',\'r\') as f:' + '\n'
        #     base_code += '    tasks = json.load(f)' + '\n'
        #     base_code += 'notebook_id = list(tasks[\''+task+'\'][\'notebook_list\'])[0]' + '\n'
        #     base_code += 'with open(\"../statsklearn/notebook_all.json\", \'r\') as f:' + '\n'
        #     base_code += '    notebook_column_info = json.load(f)' + '\n'
        #     base_code += 'column_index = notebook_column_info[str(notebook_id)][\'index\'][0]' + '\n'
        #     base_code += 'for column in notebook_column_info[str(notebook_id)][\'column_index\']:' + '\n'
        #     base_code += '    if notebook_column_info[str(notebook_id)][\'column_index\'][column] ==column_index:' +'\n'
        #     base_code += '        label_column_name = column'+'\n'
        #     # base_code += 'label_column_name = dataset.columns[column_index]' + '\n'
        #     # base_code += '#print(label_column_name)' + '\n'
        #     base_code += 'dataset.dropna(subset=[label_column_name])' + '\n'
        #     base_code += 'label_col = dataset.pop(label_column_name).astype(\'str\')' + '\n'
        #     base_code += 'if str(label_col.dtype) == \'object\':' + '\n'
        #     base_code += '    label_encoder = LabelEncoder()' + '\n'
        #     base_code += '    label_col = label_encoder.fit_transform(label_col)' + '\n'
        #     # base_code += '#print(label_col.shape)'+'\n'
        #     base_code += 'imp=SimpleImputer(strategy="most_frequent")' + '\n'
        #     base_code += 'cols = dataset.columns' + '\n'
        #     base_code += 'num_cols = dataset._get_numeric_data().columns' + '\n'
        #     base_code += 'cat_cols = list(set(cols) - set(num_cols))' + '\n'
        #     base_code += 'if len(num_cols) > 0:' + '\n'
        #     base_code += '    dataset_num = pd.DataFrame(imp.fit_transform(dataset[num_cols]), columns=num_cols).reset_index(drop=True).infer_objects()' + '\n'
        #     base_code += '    if len(cat_cols) > 0:' + '\n'
        #     base_code += '        dataset_cat = dataset[cat_cols].fillna(\'NaN\')' + '\n'
        #     base_code += '        dataset = pd.concat([dataset_num, dataset_cat], axis=1)' + '\n'
        #     base_code += '    else:' + '\n'
        #     base_code += '        dataset = dataset_num' + '\n'
        #     base_code += 'else:' + '\n'
        #     base_code += '    if len(cat_cols) > 0:' + '\n'
        #     base_code += '        dataset = dataset[cat_cols].fillna(\'NaN\')' + '\n'
        #     base_code += 'num_cols = dataset._get_numeric_data().columns' + '\n'
        #     base_code += 'cat_cols = list(set(cols) - set(num_cols))' + '\n'
        #     base_code += 'encoder = False' + '\n'
        #     base_code += 'if len(cat_cols)!= 0:' + '\n'
        #     base_code += '    encoder = True' + '\n'
        #     base_code += '    for col_name in cat_cols:' + '\n'
        #     base_code += '        column = dataset.pop(col_name).astype(\'str\')' + '\n'
        #     base_code += '        label_encoder = LabelEncoder()' + '\n'
        #     base_code += '        column = label_encoder.fit_transform(column)' + '\n'
        #     base_code += '        dataset.insert(loc=len(dataset.columns), column=col_name, value=column)' + '\n'
        #     # base_code += 'dataset.insert(loc=len(dataset.columns), column=label_column_name, value=label_col)' + '\n'

        #     temp_x = 'x = dataset'
        #     temp_y = 'y = label_col'

        #     base_code += temp_x + '\n'
        #     base_code += temp_y + '\n'

        #     seq_code_dict['code'] = base_code
        #     with open('/home/yxm/staticfg-master/HybridPipeGen/core/tmpdata/base_code_add_ai'+'/'+task+'.json', 'w') as f:
        #         json.dump(seq_code_dict, f)
        #     with open('/home/yxm/staticfg-master/HybridPipeGen/core/tmpdata/base_code_add_ai_py'+'/'+task+'.py', 'w') as f:
        #         f.write(base_code) 
    # #print(res)
    # #print(len(res))
def transform_one_validation_rl(notebook_id):
    notebooks = os.listdir('HybridPipeGen/core/tmpdata/rl_test_merge_code/')
    seq_files = os.listdir('HybridPipeGen/core/tmpdata/rl_test_merge_code/'+notebook_id)
    start_time = time.time()
    for seq_file in seq_files:
        if os.path.exists('HybridPipeGen/core/tmpdata/rl_cross_validation_code/'+notebook_id+'/'+seq_file.replace('.json', '.py')):
            continue
        seq_index = seq_file.split('.')[0]
        
        with open('HybridPipeGen/core/tmpdata/rl_test_merge_code/'+notebook_id+'/'+seq_file, 'r') as f:
            seq_code_dict = json.load(f)
        test_code = seq_code_dict['code']
        test_code = cleaning_origin(test_code)
        validation_code = ''
        test_code_list = test_code.split('\n')

        train_test_index = 0
        for index, line in enumerate(test_code_list):
            if '=train_test_split(' in line or ' train_test_split(' in line:
                if '=' not in line:
                    continue
                try:
                    x_train_varible , x_test_varible , y_train_varible , y_test_varible = line.split('=')[0].strip().split(',')
                    train_test_index = index
                except:
                    continue
                train_test_split_code = line.split('=')[1].strip()
                kuohao = train_test_split_code.find('(')
                train_test_split_code = train_test_split_code[kuohao+1:]
                arglist = train_test_split_code.split(",")
                # #print(arglist)
                try:
                    x_varible = arglist[0].strip()
                    y_varible = arglist[1].strip()
                except:
                    need_modify.append(notebook_id)
                    np.save('need_modify.npy', need_modify)
            if "model.fit(" in line:
                ours_index = index

        # #print(train_test_index)
        for index, line in enumerate(test_code_list):
            if index == train_test_index-1:
                validation_code += line
                validation_code += '\n'
                # new_line0 = 'import numpy as np\n'
                # new_line = 'np.save("'+"/home/yxm/staticfg-master/HybridPipeGen/core/tmpdata/merge_result_data/"+str(notebook_id)+"/"+seq_index+"_data_x.npy\"," +x_varible+')\n'
                # new_line1 = 'np.save("'+"/home/yxm/staticfg-master/HybridPipeGen/core/tmpdata/merge_result_data/"+str(notebook_id)+"/"+seq_index+"_data_y.npy\"," +y_varible+')\n'
                # validation_code += new_line0
                # validation_code += new_line
                # validation_code += new_line1
            if index == train_test_index:
                validation_code += line
                validation_code += '\n'
                try:
                    x_train_varible , x_test_varible , y_train_varible , y_test_varible = line.split('=')[0].strip().split(',')
                    x_train_varible = x_train_varible.strip()
                    not_null_index = line.find(x_train_varible)
                    start_null = line[0:not_null_index]
                except:
                    continue
                # #print("....")
                x_train_varible = x_train_varible.strip()
                x_test_varible = x_test_varible.strip()
                y_train_varible = y_train_varible.strip()
                y_test_varible = y_test_varible.strip()
                x_validation_varible = 'x_validation_varible'
                y_validation_varible = 'y_validation_varible'
    

                cross_validation_code = 'from sklearn.model_selection import cross_val_score\n'
                cross_validation_code += 'cross_score = cross_val_score(model, ' + x_train_varible +', ' + y_train_varible + ',cv=4)\n'
                # new_train_test_split = start_null + x_train_varible + ', ' + x_validation_varible + ', ' +  y_train_varible + ', ' +  y_validation_varible
                # new_train_test_split = new_train_test_split + " = train_test_split(" + x_train_varible + ', ' + y_train_varible + ', train_size='+str(0.875)+', test_size='+str(0.125)+', random_state='+str(split_random_state)+')' + '\n'
                # validation_code += new_train_test_split
                
            elif 'model.fit(' in line  and index >= ours_index:
                validation_code += '#'+line.replace(x_test_varible, x_validation_varible)
                validation_code += '\n'
            elif 'model.predict(' in line and index >= ours_index:
                validation_code += '#'+line.replace(x_test_varible, x_validation_varible)
                validation_code += '\n'
            
            elif 'score = accuracy_score(' in line  and index >= ours_index:
                validation_code += '#'+line.replace(y_test_varible, y_validation_varible)
                validation_code += '\n'
                validation_code += cross_validation_code
            elif 'np.save("HybridPipeGen/core/tmpdata/prenotebook_res/' in line:
                validation_code += '#'+line.replace(y_test_varible, y_validation_varible)
                validation_code += '\n'
                validation_code += line.replace('prenotebook_res', 'cross_val_res').replace(': score }', ': cross_score }')
                validation_code += '\n'
            else:
                validation_code += line
                validation_code += '\n'
        seq_code_dict['validation_code'] = validation_code

        if not os.path.exists('HybridPipeGen/core/tmpdata/rl_cross_validation_code/'+notebook_id):
            os.mkdir('HybridPipeGen/core/tmpdata/rl_cross_validation_code/'+notebook_id)
        if not os.path.exists('HybridPipeGen/core/tmpdata/rl_cross_validation_code_py/'+notebook_id):
            os.mkdir('HybridPipeGen/core/tmpdata/rl_cross_validation_code_py/'+notebook_id)
        with open('HybridPipeGen/core/tmpdata/rl_cross_validation_code/'+notebook_id+'/'+seq_file, 'w') as f:
            json.dump(seq_code_dict, f)
        with open('HybridPipeGen/core/tmpdata/rl_cross_validation_code_py/'+notebook_id+'/'+seq_file.replace('.json', '.py'), 'w') as f:
            f.write(validation_code)
def transform_one_origin_validation_code(notebook_id):

    notebook_id_py = notebook_id +'.py'
    # if os.path.exists('HybridPipeGen/core/tmpdata/merge_validation_code/'+str(notebook_id)+'/origin.json'):
    #     continue

    with open("HybridPipeGen/core/tmpdata/prenotebook_code/" + notebook_id_py, 'r') as f:
        test_code = f.read()

    test_code = cleaning_origin(test_code)
    
    validation_code = ''
    test_code_list = test_code.split('\n')
    seq_code_dict = {}
    ours_index =0
    for index, line in enumerate(test_code_list):
        if 'train_test_split(' in line:
            if '=' not in line:
                continue
            
            #print(line)
            try:
                x_train_varible , x_test_varible , y_train_varible , y_test_varible = line.split('=')[0].strip().split(',')
                train_test_index = index
            except:
                continue
            train_test_split_code = line.split('=')[1].strip()
            kuohao = train_test_split_code.find('(')
            train_test_split_code = train_test_split_code[kuohao+1:]
            arglist = train_test_split_code.split(",")
            # #print(arglist)
            x_varible = arglist[0].strip()
            y_varible = arglist[1].strip()
        if "model.fit(" in line:
            ours_index = index
    # #print(ours_index)
    # #print("------------4----------------------------------------")

    for index, line in enumerate(test_code_list):
        # if index < train_test_index -1:
        #     continue
        if index == train_test_index:
            validation_code += line
            validation_code += '\n'
            #print('index', line)
            try:
                x_train_varible , x_test_varible , y_train_varible , y_test_varible = line.split('=')[0].strip().split(',')
                x_train_varible = x_train_varible.strip()
                not_null_index = line.find(x_train_varible)
                start_null = line[0:not_null_index]
            except:
                continue
            x_train_varible = x_train_varible.strip()
            x_test_varible = x_test_varible.strip()
            y_train_varible = y_train_varible.strip()
            y_test_varible = y_test_varible.strip()
            x_validation_varible = 'x_validation_varible'
            y_validation_varible = 'y_validation_varible'
            cross_validation_code = 'from sklearn.model_selection import cross_val_score\n'
            cross_validation_code += 'cross_score = cross_val_score(model, ' + x_train_varible +', ' + y_train_varible + ',cv=4)\n'
                

        elif 'model.fit(' in line  and index >= ours_index:
                validation_code += '#'+line.replace(x_test_varible, x_validation_varible)
                validation_code += '\n'
        elif 'model.predict(' in line and index >= ours_index:
            validation_code += '#'+line.replace(x_test_varible, x_validation_varible)
            validation_code += '\n'
        
        elif 'score = accuracy_score(' in line  and index >= ours_index:
            validation_code += '#'+line.replace(y_test_varible, y_validation_varible)
            validation_code += '\n'
            validation_code += cross_validation_code
        elif 'np.save("HybridPipeGen/core/tmpdata/prenotebook_res/' in line:
            validation_code += '#'+line.replace(y_test_varible, y_validation_varible)
            validation_code += '\n'
            validation_code += line.replace('prenotebook_res', 'cross_val_res').replace(': score }', ': cross_score }')
            validation_code += '\n'
        else:
            validation_code += line
            validation_code += '\n'

    seq_code_dict['validation_code'] = validation_code
    notebook_id= str(notebook_id)
    if not os.path.exists('HybridPipeGen/core/tmpdata/rl_cross_validation_code'+'/'+notebook_id):
        os.mkdir('HybridPipeGen/core/tmpdata/rl_cross_validation_code'+'/'+notebook_id)
    if not os.path.exists('HybridPipeGen/core/tmpdata/rl_cross_validation_code_py'+'/'+notebook_id):
        os.mkdir('HybridPipeGen/core/tmpdata/rl_cross_validation_code_py'+'/'+notebook_id)
    with open('HybridPipeGen/core/tmpdata/rl_cross_validation_code'+'/'+notebook_id+'/origin.json', 'w') as f:
        json.dump(seq_code_dict, f)
    with open('HybridPipeGen/core/tmpdata/rl_cross_validation_code_py'+'/'+notebook_id+'/origin.py', 'w') as f:
        f.write(validation_code)

if __name__ == "__main__":
    merger = Merger(4)
    # merger.batch_enuming_planb()
    merger.batch_enuming_rl()
    # merger.merging_one_notebook_planb('960455')
    # merger.enum_adding(5476391)
    # merger.count_hightlight()
    # merger.count_operations()

    # transform_validation_code()
    # transform_origin_validation_code()
    # transform_validation_planB()
    transform_validation_rl()

    # transform_validation_code_new()
    # transform_origin_validation_code_new()
    # create_base_code_all()
    # create_base_code()
    # test_time()