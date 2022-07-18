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
import gc

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
    # pprint.pprint(ai_sequences)
    ai_sequences_new = [i for i in ai_sequences if len(i) ==3]
    # print(len(ai_sequences_new))
    return ai_sequences_new

class Step():
    def __init__(self, operator, edge_id):
        self.operator = operator
        self.edge_id = edge_id

    
class Merger():
    def __init__(self):
        self.code = ''
        self.origin_code = ''
        self.operations = []
        self.graph = None
        self.all_seq_num_dict = {}
        self.cuted_seq_num_dict = {}

        self.enum_num = []
        self.cut_num = []

    def load_origin_code(self, notebook_id):
        filepath = 'HybridPipeGen/core/tmpdata/prenotebook_code/' + str(notebook_id) + '.py'
        self.file_path = filepath
        output_path = filepath
        with open(output_path, 'r') as src_file:
            self.origin_code = src_file.read()
        self.code = self.origin_code
        # print('load_origin_code', self.code.split('\n')[49])

    def load_graph(self, notebook_id):     
        filepath = 'HybridPipeGen/core/tmpdata/prenotebook_graph/' + str(notebook_id) + ".pkl"
        with open(filepath, 'rb') as f:
            self.graph = pickle.load(f)  

    def print_code(self):
        codelist = self.code.split("\n")
        for index,line in enumerate(codelist):
            print(index, line)

    def print_one_seq(self,seq):
        str_ = ''
        for item in seq:
            str_ += str((item.operator, item.edge_id))
            str_ += ','
        print(str_)

    def print_all_seq(self):
        for seq in self.all_seq:
            self.print_one_seq(seq)


    def enum_adding_rl(self, notebook_id, ai_seq):
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
        # print("###############")
        # print('notebook_id', notebook_id)
        
        self.load_graph(notebook_id)
        # res = self.load_rl_operations(notebook_id)
        self.operations = [ i for i in ai_seq if i != 'blank']

        # print('len(self.operations)', len(self.operations))
        self.need_add_edge_id = []
        added_sorted_id = []
        
        for edge in self.graph.result_edges:
            if edge.edge_type == 1 and edge.sorted_id not in added_sorted_id and 'sum' not in edge.func_name:
                self.need_add_edge_id.append(edge.edge_id)
                added_sorted_id.append(edge.sorted_id)
        # print('len(self.operations)', len(self.operations))
        # print('len(need_add_edge_id)', len(self.need_add_edge_id))
    
        self.all_seq = []
        if len(self.need_add_edge_id) != 0:
            self.all_seq = [[]]
            for ope_index,operation in enumerate(self.operations):
                new_seq = copy.copy(self.all_seq)
                # print('ope_index', ope_index)
                for edge_index in self.need_add_edge_id:
                    # print('edge_index', edge_index)
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
        def remove_ope(remove_num, seq):
            from itertools import combinations
            import copy
            new_seq = []
            combinations_items = list(combinations(seq, remove_num))
            for remove_items in combinations_items:
                seq_copy = copy.copy(seq)
                for remove_item in remove_items:
                    seq_copy.remove(remove_item)
                new_seq.append(seq_copy)
            # print(new_seq)
            return new_seq

        for remove_num in range(0, len(self.operations)):
            end_seqs += remove_ope(remove_num, self.operations)
        # if len(self.operations) == 3:
        #     end_seqs = [qw`                       `
        #         [self.operations[0]],
        #         [self.operations[0], self.operations[1]],
        #         [self.operations[0], self.operations[2]],
        #         [self.operations[0], self.operations[1], self.operations[2]],
        #         [self.operations[1]],
        #         [self.operations[2]],
        #         [self.operations[1], self.operations[2]],
        #     ]
        # elif len(self.operations) == 2:
        #     end_seqs = [
        #         [self.operations[0]],
        #         [self.operations[0], self.operations[1]],
        #         [self.operations[1]],
        #     ]
        # elif len(self.operations) == 1:
        #     end_seqs = [
        #         [self.operations[0]],
        #     ]
        # seq = []
        # for operation in self.operations:
        #     step = Step(operation, 'end')
        #     seq.append(step)
        #     # print("seq",seq)
        #     temp = copy.copy(seq)
        #     # print('temp',temp)
        #     self.all_seq.append(temp)
        
        for seq in end_seqs:
            step_seq = []
            for operation in seq:
                step = Step(operation, 'end')
                step_seq.append(step)
            temp = copy.copy(step_seq)
            # print('step_seq', step_seq)
            self.all_seq.append(temp)
        # print('self.operations', self.operations)
        # self.all_seq += seq
        # pprint.pprint(self.all_seq)

        if len(self.all_seq) not in self.all_seq_num_dict:
            self.all_seq_num_dict[len(self.all_seq)] = 0
        self.all_seq_num_dict[len(self.all_seq)] += 1
        self.enum_num.append(len(self.all_seq))
    
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
                # print('after_list', after_list)
                for operation in after_list:
                    for ope in EdgeOperationType['Cleaning']:
                        if ope in operation:
                            # print('rule1_false', operation)
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
            # train_test_split_num = 0
            found = False
            for one_edge_id in edge_dict:
                # print('edge_dict[one_edge_id]', edge_dict[one_edge_id])
                if edge_dict[one_edge_id] == 'train_test_split':
                   train_test_split_num  = one_edge_id
                   found = True

            for step in seq:
                if "---" not in step.edge_id:
                    continue
                edge_id, position = step.edge_id.split("---")
                edge_id = int(edge_id)
                before_list = []
                for one_edge_id in edge_dict:
                    if one_edge_id < edge_id:
                        before_list.append(one_edge_id)
                        
                # print('after_list', after_list)
                # print('before list', before_list)
                for one_edge_id in before_list:
                    if found:
                        if one_edge_id == train_test_split_num:
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
            # print('edge_all_dict funcname', edge.func_name)
            # print('edge_all_dict origincode', edge.original_code)
        # print(self.edge_dict)
        # print(self.operations)
        cant_add_index = []
        cant_add_index3 = []
        if len(self.edge_dict) == 0:
            self.cuted_all_seq = copy.copy(self.all_seq)
        else:
            for index,seq in enumerate(self.all_seq):
                if check_rule0(seq) and check_rule1(seq, self.edge_dict) and check_rule2(seq, self.edge_dict) and check_rule3(seq, self.edge_all_dict):
                # if check_rule3(seq, self.edge_all_dict):
                    self.cuted_all_seq.append(seq)
                    # print("can add")
                    # self.print_one_seq(seq)
                else:
                    if check_rule0(seq) and check_rule1(seq, self.edge_dict) and check_rule2(seq, self.edge_dict) and check_rule3(seq, self.edge_all_dict) == False:
                        cant_add_index3.append(index)
                    cant_add_index.append(index)
                # else:
                    # print('cant add')
                    # self.print_one_seq(seq)

        if len(self.cuted_all_seq) not in self.cuted_seq_num_dict:
            self.cuted_seq_num_dict[len(self.cuted_all_seq)] = 0
        self.cuted_seq_num_dict[len(self.cuted_all_seq)] += 1
        self.cut_num.append(len(self.cuted_all_seq))

        return cant_add_index, cant_add_index3
        # print('len(self.cuted_all_seq)',len(self.cuted_all_seq))
    def add_one_ope(self, notebook_id, edge_id, ope, position, varible):
        with open('HybridPipeGen/core/tmpdata/prenotebook_varibles_index/'+str(notebook_id)+'.json', 'r') as f:
            varible_index = json.load(f)
        code_list = self.code.split('\n')
        # self.print_code()
        if ope not in list(OperationCode.keys()):
            return False
        
        
        # print('position', position)
        if position != 'end':
            # print('xxxx')
            operation_code = OperationCode[ope]['pre_code'] + OperationCode[ope]['code']
            operation_code = operation_code.replace("-[PLACEHOLDER]-", varible) + '\n'
            found = False
            for edge in self.graph.result_edges:
                if edge.edge_id == edge_id:
                    # print('edge.line_id', edge.line_id)
                    # print('edge.original_code[0:-1]', edge.original_code[0:-1])
                    # print('edge.line_id[0]', edge.line_id[0])
                    add_position = edge.line_id[0] + self.added_rows
                    found_edge_id = edge.line_id[0]
                    found = True

            
            if add_position == 0:
                pre_code_list = []
            else:
                pre_code_list = code_list[0:add_position]
                
            pre_code = ''
            for item in pre_code_list:
                pre_code += item
                pre_code += '\n'
            # print(self.added_rows)
            # print('add_position', add_position)
            # print(len(code_list))
            # print(code_list[add_position])
            # print('position', position)
            # print('add_position', add_position)
            # print("varible_index['end_idnex']", varible_index['end_idnex'])
            edge_code = code_list[add_position] + '\n'
            # print('edge_code', edge_code)
            if add_position == len(code_list)-1:
                after_code_list = []
            else:
                after_code_list = code_list[add_position+1:]

            if found_edge_id > varible_index['end_idnex'] and found:
                return False
            after_code = ''
            for item in after_code_list:
                after_code += item
                after_code += '\n'
            # print('####')
            # print('operation_code', operation_code)
            if position == 'before': # before
                self.code = pre_code + operation_code + edge_code + after_code
            elif position == 'after':
                self.code = pre_code  + edge_code + operation_code + after_code
        else:
            # print('ope',ope)
            
            x_varible = varible_index['x_varible']
            operation_code = OperationCode[ope]['pre_code'] + OperationCode[ope]['code']
            operation_code = operation_code.replace("-[PLACEHOLDER]-", x_varible) + '\n'
            # print(varible_index['end_idnex'])
            end_index = varible_index['end_idnex'] + self.added_rows
            # print('add_rows',str(self.added_rows))
            # print('end_index', str(end_index))

            # print("varible_index['end_idnex']", varible_index['end_idnex'])
            # print('self.added_rows', self.added_rows)
            code_list = self.code.split("\n")
            # print(code_list[end_index])
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
            # print(self.code)
            
            
        self.added_rows += len(operation_code.split('\n'))-1
            # break
        return True
   
   
    def merging_one_notebook_rl(self,notebook_id, ai_seq):
        time1 = time.time()
        res = self.enum_adding_rl(notebook_id, ai_seq)
        time2 = time.time()

        # cant_add_index, cant_add_index3 = self.cut_by_rule()
        time3 = time.time()
        self.load_graph(notebook_id)
        time_load_graph = time.time()
        seq_id = 0
        # print("sss")
        
        # print(self.cuted_all_seq)
        self.cuted_all_seq = self.all_seq
        note_len = {}
        note_len['len'] = len(self.cuted_all_seq)
        # with open('data/analysis/offline_26000/'+notebook_id + '.json', 'w') as f:
        #     json.dump(note_len, f)
        for seq in self.cuted_all_seq:
            self.added_rows = 0
            self.load_origin_code(notebook_id)
            # print('seq', seq)
            # if seq_id == 2:
                # self.print_code()
            subtime1 = time.time()
            is_all_no_add = True
            for step in seq:

                now_ope = step.operator
                if step.edge_id != 'end':
                    edge_id, position = step.edge_id.split("---")
                    edge_id = int(edge_id)
                else:
                    edge_id = 0
                    position = 'end'
                # print('node_edge.edge_id', node_edge.edge_id)
                varible = ''
                for node in self.graph.result_nodes:
                    # print('node')
                    for edge_index,node_edge in enumerate(node.children_edges):
                        # print('edge_index', edge_index)
                        if node_edge.edge_id == edge_id:
                            varible = node.varible_name
                            assign_num = 0
                            temp_varible = ''
                            # print('varible', varible)
                            for child_index,child_edge in enumerate(node.childrens[edge_index].children_edges):
                                # print(child_edge.func_name)
                                if child_edge.func_name == '-Assign-':
                                    assign_num += 1
                                    temp_varible = node.childrens[edge_index].childrens[child_index].varible_name
                            if assign_num == 1:
                                varible = temp_varible
                # print('end varible', varible)
                # print('seq_id', seq_id)
                
                add_res = self.add_one_ope(notebook_id, edge_id, now_ope, position, varible)
                if add_res == True:
                    is_all_no_add = False

            if is_all_no_add == True:
                continue
            subtime2 = time.time()
            # print('subtime2', subtime2-subtime1)
       
            # print('generate code.....')
            # print('enum code', sub_time1)
            # print('add one ope', sub_time2)
        
            # print(self.code)
            # if seq_id == 1:
            #     self.print_code()
            #     self.print_one_seq(seq)
            #     break

            save_seq = []
            for item in seq:
                save_seq.append({"operator": item.operator, "edge_id": item.edge_id})
            self.code = self.code.replace('HybridPipeGen/core/tmpdata/prenotebook_res/'+ notebook_id, 'HybridPipeGen/core/tmpdata/merge_max_result_rl/' + notebook_id + '/' + str(seq_id))
            subtime3 = time.time()
            # print('subtime3', subtime3-subtime2)
  
            if not os.path.exists('HybridPipeGen/core/tmpdata/rl_test_merge_code/'+str(notebook_id)):
                os.mkdir('HybridPipeGen/core/tmpdata/rl_test_merge_code/'+str(notebook_id))
            with open('HybridPipeGen/core/tmpdata/rl_test_merge_code/' + str(notebook_id)+'/'+str(seq_id)+'.json', 'w') as f:
                json.dump({'seq':save_seq, 'code': self.code}, f)
            if not os.path.exists('HybridPipeGen/core/tmpdata/rl_test_merge_code_py/'+str(notebook_id)):
                os.mkdir('HybridPipeGen/core/tmpdata/rl_test_merge_code_py/'+str(notebook_id))
            with open('HybridPipeGen/core/tmpdata/rl_test_merge_code_py/'+str(notebook_id)+'/'+str(seq_id)+'.py', 'w') as f:
                f.write(self.code)
            seq_id += 1
            # gc.collect()
            subtime4 = time.time()
            # print('subtime4', subtime4-subtime3)
        time4 = time.time()

        gc.collect()
    

def transform_one_validation_rl(notebook_id):

    seq_files = os.listdir('HybridPipeGen/core/tmpdata/rl_test_merge_code/' + notebook_id)
    for seq_file in seq_files:
        seq_index = seq_file.split('.')[0]
        
        with open('HybridPipeGen/core/tmpdata/rl_test_merge_code/' + notebook_id + '/' + seq_file, 'r') as f:
            seq_code_dict = json.load(f)
        # print('seq_file', seq_file)
        test_code = seq_code_dict['code']
        # test_code = cleaning_origin(test_code)
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
                # print(arglist)
                try:
                    x_varible = arglist[0].strip()
                    y_varible = arglist[1].strip()
                except:
                    need_continue = True
            if "model.fit(" in line:
                ours_index = index

        # print(train_test_index)
        for index, line in enumerate(test_code_list):
            # print(index,line)
            if index == train_test_index-1:
                # print('train_test_indexs',line)
                validation_code += line
                validation_code += '\n'

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
                # print("....")
                x_train_varible = x_train_varible.strip()
                x_test_varible = x_test_varible.strip()
                y_train_varible = y_train_varible.strip()
                y_test_varible = y_test_varible.strip()
                x_validation_varible = 'x_validation_varible'
                y_validation_varible = 'y_validation_varible'
    
                # cross_validation_code = 'cross_score = xgb.cv(params, xgb_train, nfold = 4, seed = 0)\n'
                cross_validation_code = 'from sklearn.model_selection import cross_val_score\n'
                cross_validation_code += 'cross_score = cross_val_score(model, ' + x_train_varible +', ' + y_train_varible + ',cv=4)\n'
                
            elif 'model.fit(' in line  and index >= ours_index:
                validation_code += '#'+line.replace(x_test_varible, x_validation_varible)
                validation_code += '\n'
            elif 'model.predict(' in line and index >= ours_index:
                validation_code += '#'+line.replace(x_test_varible, x_validation_varible)
                validation_code += '\n'
            
            elif 'score = accuracy_score(' in line and index >= ours_index:
                validation_code += '#'+line.replace(y_test_varible, y_validation_varible)
                validation_code += '\n'
                validation_code += cross_validation_code
            elif 'HybridPipeGen/core/tmpdata/merge_max_result_rl/' in line:
                validation_code += '#'+line.replace(y_test_varible, y_validation_varible)
                validation_code += '\n'
                validation_code += line.replace('HybridPipeGen/core/tmpdata/merge_max_result_rl/', 'HybridPipeGen/core/tmpdata/rl_cross_val_res/').replace(': score }', ': cross_score }')
                validation_code += '\n'
                # print('validation_code', validation_code)
            else:
                validation_code += line
                validation_code += '\n'
        seq_code_dict['validation_code'] = validation_code
    
        if not os.path.exists('HybridPipeGen/core/tmpdata/rl_cross_validation_code/' + notebook_id):
            os.mkdir('HybridPipeGen/core/tmpdata/rl_cross_validation_code/'+ notebook_id)
        if not os.path.exists('HybridPipeGen/core/tmpdata/rl_cross_validation_code_py/' + notebook_id):
            os.mkdir('HybridPipeGen/core/tmpdata/rl_cross_validation_code_py/' + notebook_id)
        with open('HybridPipeGen/core/tmpdata/rl_cross_validation_code/' + notebook_id + '/' + seq_file, 'w') as f:
            json.dump(seq_code_dict, f)
        with open('HybridPipeGen/core/tmpdata/rl_cross_validation_code_py/' + notebook_id + '/' + seq_file.replace('.json', '.py'), 'w') as f:
            f.write(validation_code)


def transform_one_validation_rl_origin(notebook_id):
    notebook_py = notebook_id + '.py'
    with open('HybridPipeGen/core/tmpdata/prenotebook_code/' + notebook_py, 'r') as f:
        test_code = f.read()
    # test_code = cleaning_origin(test_code)
    validation_code = ''
    test_code_list = test_code.split('\n')
    seq_code_dict = {}
    need_continue=False
    train_test_index = 0
    start_time = time.time()
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
            # print(arglist)
            try:
                x_varible = arglist[0].strip()
                y_varible = arglist[1].strip()
            except:

                need_continue = True
        if "model.fit(" in line:
            ours_index = index

    for index, line in enumerate(test_code_list):
        if index == train_test_index-1:
            validation_code += line
            validation_code += '\n'
        # print('all line', line)
        if index == train_test_index:
            # print('train test index', line)
            validation_code += line
            validation_code += '\n'
            try:
                x_train_varible , x_test_varible , y_train_varible , y_test_varible = line.split('=')[0].strip().split(',')
                x_train_varible = x_train_varible.strip()
                not_null_index = line.find(x_train_varible)
                start_null = line[0:not_null_index]
            except:
                continue
            # print("....")
            x_train_varible = x_train_varible.strip()
            x_test_varible = x_test_varible.strip()
            y_train_varible = y_train_varible.strip()
            y_test_varible = y_test_varible.strip()
            x_validation_varible = 'x_validation_varible'
            y_validation_varible = 'y_validation_varible'

            # cross_validation_code = 'cross_score = xgb.cv(params, xgb_train, nfold = 4, seed = 0)\n'
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
        elif 'HybridPipeGen/core/tmpdata/prenotebook_res/' in line:
            # print('line', line)
            validation_code += '#'+line.replace(y_test_varible, y_validation_varible)
            validation_code += '\n'
            validation_code += line.replace('HybridPipeGen/core/tmpdata/prenotebook_res/', 'HybridPipeGen/core/tmpdata/rl_cross_val_res/'+notebook_id+'/').replace(': score }', ': cross_score }').replace(notebook_id +'.npy','origin.npy')
            validation_code += '\n'
            
    
        else:
            validation_code += line
            validation_code += '\n'
    seq_code_dict['code'] = validation_code
    if not os.path.exists('HybridPipeGen/core/tmpdata/rl_cross_validation_code/' + notebook_id):
        os.mkdir('HybridPipeGen/core/tmpdata/rl_cross_validation_code/' + notebook_id)
    if not os.path.exists('HybridPipeGen/core/tmpdata/rl_cross_validation_code_py/'  + notebook_id):
        os.mkdir('HybridPipeGen/core/tmpdata/rl_cross_validation_code_py/' + notebook_id)
    with open('HybridPipeGen/core/tmpdata/rl_cross_validation_code/' + notebook_id + '/' + 'origin.json', 'w') as f:
        json.dump(seq_code_dict, f)
    with open('HybridPipeGen/core/tmpdata/rl_cross_validation_code_py/' + notebook_id + '/origin.py', 'w') as f:
        f.write(validation_code)


if __name__ == "__main__":
    # merger = Merger()
    # merger.batch_count_number()
    # merger.batch_enuming_rl(1,1)
    # para_merge()
    # transform_validation_rl()
    # transform_validation_rl_origin()
    # batch_pkl_to_code('test')
    count_test_py()