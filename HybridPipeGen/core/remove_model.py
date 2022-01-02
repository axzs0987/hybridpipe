import ast
import astunparse
import re

class FindVariables(ast.NodeVisitor):
    def __init__(self):
        self.variables = []

    def visit_Name(self,node):
        self.variables.append(node.id)
        ast.NodeVisitor.generic_visit(self, node)

def remove_model(code):
    try:
        r_node = ast.parse(code)
    except:
        return code
    lines = code.split('\n')
    fit_caller = {}
    str_ = ''

    need_remove_variables = []
    need_remove_index = []

    for index,line in enumerate(lines): 
        if ('ProfileReport(' in line or 'cross_val_predict(' in line or 'plot(' in line or 'axes[' in line or 'axes.' in line or 'sns.' in line or '.HoloMap(' in line or '.boxcox(' in line or 'plt.' in line) and 'def ' not in line:
            line1 = line.strip()
            try:
                r_node = ast.parse(line1)
            except:
                continue
            if type(r_node.body[0]).__name__ == 'Assign':
                for tar in r_node.body[0].targets:
                    #print('tar', tar)
                    pred_variable = astunparse.unparse(tar)[0:-1]
                    if ',' in pred_variable:
                        pvars = pred_variable[1:-1].split(',')
                        for pvar in pvars:
                            need_remove_variables.append(pvar.strip())
                    else:
                        need_remove_variables.append(pred_variable)

                    #print("*4", pred_variable)
            need_remove_index.append(index)
            # #print("#7", index)
        if '.fit(' in line or '.fit_resample(' in line:
            line1 = line.strip()
            try:
                r_node = ast.parse(line1)
            except:
                continue
            
            assigned = r_node.body[0].value
            if type(assigned).__name__ == 'Attribute':
                assigned = assigned.value

            args = assigned.args
            if len(args) == 2:
                caller = astunparse.unparse(assigned.func.value)[0:-1]
                if '(' in caller and ')' in caller:
                    if type(r_node.body[0]).__name__ == 'Assign':
                        caller = astunparse.unparse(r_node.body[0].targets[0])[0:-1]
                    else:
                        if 'GridSearchCV' in line:
                            need_remove_index.append(index)
                            # #print("#6", index)
                        continue
                    func_name = astunparse.unparse(assigned.func.value)[0:-1].split('(')[0]
                    if caller not in fit_caller:
                        fit_caller[caller] = {}
                        fit_caller[caller]['index'] = []
                        fit_caller[caller]['func_name']= func_name
        
                if caller not in fit_caller:
                    fit_caller[caller] = {}
                    fit_caller[caller]['index'] = []
                fit_caller[caller]['index'].append(index)
                fit_caller[caller]['need_remove']= True
            else:
                # #print(line)
                for keyword in assigned.keywords:
                    # #print('keyword', keyword)
                    if keyword.arg == 'y':
                        caller = astunparse.unparse(assigned.func.value)[0:-1]
                        if '(' in caller and ')' in caller:
                            if type(r_node.body[0]).__name__ == 'Assign':
                                caller = astunparse.unparse(r_node.body[0].targets[0])[0:-1]
                            else:
                                if 'GridSearchCV' in line:
                                    need_remove_index.append(index)
                                    # #print("#6", index)
                                continue
                            func_name = astunparse.unparse(assigned.func.value)[0:-1].split('(')[0]
                            if caller not in fit_caller:
                                fit_caller[caller] = {}
                                fit_caller[caller]['index'] = []
                                fit_caller[caller]['func_name']= func_name
                
                        if caller not in fit_caller:
                            fit_caller[caller] = {}
                            fit_caller[caller]['index'] = []
                        fit_caller[caller]['index'].append(index)
                        fit_caller[caller]['need_remove']= True
                        break
        if '.best_score_' in line:
            # #print(line)
            for caller in fit_caller:
                if caller in line:
                    fit_caller[caller]['index'].append(index)
        if '.best_params_' in line:
            # #print(line)
            for caller in fit_caller:
                if caller in line:
                    fit_caller[caller]['index'].append(index)
        if '.best_estimator_' in line:
            # #print(line)
            for caller in fit_caller:
                if caller in line:
                    fit_caller[caller]['index'].append(index)
                    
    # #print(fit_caller)
    
    for index,line in enumerate(lines): 
        try:
            r_node = ast.parse(line).body[0]
        except:
            continue
        if type(r_node).__name__ == 'Assign':
            if astunparse.unparse(r_node.targets[0])[0:-1] in list(fit_caller.keys()):
                # #print('line', line)
                if type(r_node.value).__name__ != 'Call':
                    continue
                if type(r_node.value.func).__name__ == 'Name':
                    
                    func_name = astunparse.unparse(r_node.value.func)[0:-1]
                    # #print(func_name)
                    fit_caller[astunparse.unparse(r_node.targets[0])[0:-1]]['func_name']= func_name
                    
    
    for index,line in enumerate(lines):
        if 'from' in line and 'import' in line and 'feature_selection' in line:
            for caller in fit_caller:
                if 'func_name' in fit_caller[caller]:
                    if fit_caller[caller]['func_name'] in line:
                        fit_caller[caller]['need_remove'] = False
    for caller in fit_caller:
        if fit_caller[caller]['need_remove']==True:
            for ind in fit_caller[caller]['index']:
                need_remove_index.append(ind)
                # #print("#5", index)
            need_remove_variables.append(caller)
            #print("*5", caller)

    ###### check definition
    is_in_def = []
    def_info = {}
    next_line = []
    suojin_num = []
    def_name = []
    for index,line in enumerate(lines):
        # #print(line)
        if len(is_in_def) > 0: 
            if index == def_info[def_name[-1]]['start'] + next_line[-1]:
                all_blank = True
                for char in line:
                    if ord(char) != 32:
                        all_blank=False
                        break
                if all_blank:
                    next_line[-1] += 1
                    continue
                suojin_num[-1] = 0
                for char in line:
                    if char != ' ':
                        break
                    suojin_num[-1] += 1
                # #print(suojin_num)
            else:
                all_blank = True
                for char in line:
                    if ord(char) != 32:
                        all_blank=False
                        break
                if all_blank:
                    continue
                line_suojin = 0
                for char in line:
                    if char != ' ':
                        break
                    line_suojin += 1
                #print('line_suojin', line_suojin)
                #print('suojin_num', suojin_num[-1])
                if line_suojin < suojin_num[-1]:
                    #print(def_name[-1], "end")
                    def_info[def_name[-1]]['end'] = index -1
                    is_in_def = is_in_def[0:-1]
                    next_line = next_line[0:-1]
                    suojin_num = suojin_num[0:-1]
                    def_name = def_name[0:-1]
        if 'def ' in line:
            # #print(line)
            def_name.append(line.strip()[4:-1].split('(')[0])
            def_info[def_name[-1]] = {}
            def_info[def_name[-1]]['start'] = index
            is_in_def.append(True)
            next_line.append(1)
            suojin_num.append(0)
            # continue
        if line == '':
            continue
        
        if line[0:7] == 'return ':
            # #print(def_name[-1], "return")
            def_info[def_name[-1]]['end'] = index
            is_in_def = is_in_def[0:-1]
            next_line = next_line[0:-1]
            suojin_num = suojin_num[0:-1]
            def_name = def_name[0:-1]
            # #print("return end")

    if len(is_in_def) > 0:
        def_info[def_name[-1]]['end'] = len(lines)-1
        is_in_def = is_in_def[0:-1]
        next_line = next_line[0:-1]
        suojin_num = suojin_num[0:-1]
        def_name = def_name[0:-1]
    str_ = ''
    # #print(need_remove_variables)
    #print(def_info)
    ######## find need remove variables
    for index,line in enumerate(lines):
        can_parse = True
        for key in def_info:
            all_drop = True
            if index == def_info[key]['end']+1:
                #print('key', key)
                #print(def_info[key]['start'], def_info[key]['end'])
                for ind in range(def_info[key]['start']+1, def_info[key]['end']+1):
                    all_blank = True
                    for char in lines[ind]:
                        if ord(char) != 32:
                            all_blank=False
                            break
                    if all_blank:
                        continue
                    #print(lines[ind], ind in need_remove_index)
                    if ind not in need_remove_index:
                        all_drop = False
                        break
                if all_drop:
                    def_info[key]['drop'] = True
                    need_remove_variables.append(key)
        try:
            r_node = ast.parse(line.strip()).body[0]
        except:
            can_parse = False
        if '.show(' in line or 'input(' in line or '.imshow(' in line:
            need_remove_index.append(index)
            # #print("#4", index)
        if can_parse:
            if type(r_node).__name__ == 'Assign':
                # #print(line)
                # #print(need_remove_variables)
                for variable in need_remove_variables:
                    if '[' + variable + ']' in astunparse.unparse(r_node.targets)[0:-1]:
                        need_remove_index.append(index)
                        # #print("#3", index)
                        need_remove_variables.append(astunparse.unparse(r_node.targets)[0:-1].split('[')[0])
                        # #print("*7", )
                        #print("*1", astunparse.unparse(r_node.targets)[0:-1].split('[')[0])
                    elif variable + '.' in astunparse.unparse(r_node.targets)[0:-1]:
                        need_remove_index.append(index)
                        # #print("#3", index)
                        #print("*1", astunparse.unparse(r_node.targets)[0:-1].split('[')[0])
                    elif variable + '[' in astunparse.unparse(r_node.targets)[0:-1]:
                        need_remove_index.append(index)
                        # #print("#3", index)
                        #print("*1", astunparse.unparse(r_node.targets)[0:-1].split('[')[0])
                    fv = FindVariables()
                    fv.visit(r_node.value)
                    # #print(fv.variables)
                    if variable in fv.variables:
                        for taget in r_node.targets:
                            target_str = astunparse.unparse(taget)[0:-1]
                            if '(' == target_str[0] and ')' == target_str[-1]:
                                target_str = astunparse.unparse(taget)[0:-1][1:-1]
                            split_dou = target_str.split(',')
                            # #print('split_dou', split_dou)
                            for target_var in split_dou:
                                target_var = target_var.strip()
                                if target_var not in need_remove_variables:
                                    need_remove_variables.append(target_var)
                                    #print("*2", target_var)
                                    #print(line)
                                    #print(variable)
                        
                        need_remove_index.append(index)
                        # #print("#2", index)
            else:
                fv = FindVariables()
                fv.visit(r_node)
                for variable in need_remove_variables:          
                    if variable in fv.variables:
                        # #print('remove index', index)
                        need_remove_index.append(index)
                        for item in fv.variables:
                            if item not in need_remove_variables and item+'.append(' in line:
                                need_remove_variables.append(item)
                                #print("*7", item)
                        break
            if line.strip()[0:7] == 'return ':
                #print('return,,,,', line)
                try:
                    r_node = ast.parse(line.strip())
                except:
                    continue
                #print('need_remove_variables', need_remove_variables)
                fv = FindVariables()
                fv.visit(r_node)
                variables = fv.variables
                #print('fv.variables', fv.variables)
                for variable in need_remove_variables:
                    #print('return xx', variables)
                    if variable in variables:
                        #print('return variable', variable)
                        for key in def_info:
                            if index == def_info[key]['end']:
                                def_info[key]['drop'] = True
                                need_remove_variables.append(key)
                                #print("*3", key)
                        break
    
   
   ##### deal with all for,if,while #ed
    is_suojin = []
    suojin_num = []
    suojin_index = []
    need_remove = []
    next_line = []
    before_suojin_num = []
    for index,line in enumerate(lines):
        
        if len(is_suojin) > 0:
            if index == suojin_index[-1]+next_line[-1]:
                #print("xxxxxxx282")
                all_blank = True
                for char in line:
                    if ord(char) != 32:
                        all_blank=False
                        break
                if all_blank:
                    next_line[-1] += 1
                    continue
                suojin_num[-1] =0
                for char in line:
                    if ord(char) == 32:
                        suojin_num[-1] += 1
                    elif ord(char) == 9:
                        suojin_num[-1] += 4
                    else:
                        break
                if suojin_num[-1] == before_suojin_num[-1]:
                    is_suojin = is_suojin[0:-1]
                    suojin_num = suojin_num[0:-1]
                    need_remove = need_remove[0:-1]
                    suojin_index = suojin_index[0:-1]
                    next_line = next_line[0:-1]
                # if need_remove[-1]:
                    # need_remove_index.append(index)
        # if len(is_suojin) > 0:
            else:
                #print("xxxxxxx303")
                all_blank = True
                for char in line:
                    if ord(char) != 32:
                        all_blank=False
                        break
                if all_blank:
                    continue
                line_suojin =0
                for char in line:
                    if ord(char) == 32:
                        line_suojin += 1
                    elif ord(char) == 9:
                        line_suojin += 4
                    else:
                        break
                if line_suojin != suojin_num[-1]:
                    #print("xxxxxxx320")
                    is_suojin = is_suojin[0:-1]
                    is_all_in = True
                    suojin_num = suojin_num[0:-1]
                    need_remove = need_remove[0:-1]
                    #print(lines[suojin_index[-1]])
                    #print(line)
                    for ind in range(suojin_index[-1]+1, index):
                        all_blank = True
                        for char in lines[ind]:
                            if ord(char) != 32:
                                all_blank=False
                                break
                        if all_blank:
                            continue
                        #print('line', lines[ind])
                        #print('ind in need_remove_index', ind in need_remove_index)
                        if ind not in need_remove_index:
                            is_all_in=False
                            break
                    
                    if is_all_in:
                        need_remove_index.append(suojin_index[-1])
                    suojin_index = suojin_index[0:-1]
                    next_line = next_line[0:-1]

                    while(len(is_suojin)>0 and line_suojin != suojin_num[-1]):
                        #print("xxxxxxx351")
                        is_suojin = is_suojin[0:-1]
                        is_all_in = True
                        suojin_num = suojin_num[0:-1]
                        need_remove = need_remove[0:-1]
                        #print(lines[suojin_index[-1]])
                        #print(line)
                        for ind in range(suojin_index[-1]+1, index):
                            all_blank = True
                            for char in lines[ind]:
                                if ord(char) != 32:
                                    all_blank=False
                                    break
                            if all_blank:
                                continue
                            #print('line', lines[ind])
                            #print('ind in need_remove_index', ind in need_remove_index)
                            if ind not in need_remove_index:
                                is_all_in=False
                                break
                        
                        if is_all_in:
                            need_remove_index.append(suojin_index[-1])
                        suojin_index = suojin_index[0:-1]
                        next_line = next_line[0:-1]
                # else:
                #     #print("xxxxxxx342")
            if len(need_remove) > 0:
                if need_remove[-1]:
                    need_remove_index.append(index)
        if line.strip()[0:4] == 'for ' or line.strip()[0:3] ==  'if ' or line.strip()[0:6] == 'while ' or line.strip()[0:5] == 'else:' or line.strip()[0:5] == 'elif ' or line.strip()[0:5] == 'with ' or line.strip()[0:4] == 'try:' or line.strip()[0:7] == 'except:' or line.strip()[0:7] == 'except ':
            #print("xxxxxxx346")
            is_suojin.append(True)
            suojin_index.append(index)
            suojin_num.append(0)
            need_remove_by_v = False
            next_line.append(1)
            before_suojin_num.append(0)
            for char in line:
                if ord(char) == 32:
                    before_suojin_num[-1] += 1
                elif ord(char) == 9:
                    before_suojin_num[-1] += 4
                else:
                    break
            # #print("33333333")
            # #print(line)
            for var in need_remove_variables:
                # #print('var', var)
                # #print(line)
                if var in line:
                    need_remove_by_v = True
                    break
            if len(need_remove) > 0:
                if need_remove[-1]:
                    need_remove_by_v = True
            # #print('need_remove_variables', need_remove_variables)
            # #print('need_remove_by_v',need_remove_by_v)
            # #print('index in need_remove_index', index in need_remove_index)
            need_remove.append(index in need_remove_index or need_remove_by_v)
            if need_remove[-1]:
                need_remove_index.append(index)
            # continue
        # if index< 856:
        #     #print("3333333")
        #     #print(line)
        #     #print('is_suojin', is_suojin)
        #     #print('need_remove', need_remove)
        #     #print('suojin_num', suojin_num)
        #     #print('next_line', next_line)
        

    #print('def_info', def_info)
    # #print('need_remove_index', need_remove_index)
    is_suojin = []
    suojin_num = []
    suojin_index = []

    #print('need_remove_variables', need_remove_variables)
    next_line = 1
    for index,line in enumerate(lines): 
        # #print(line)
        is_def_drop = False
        is_for = False
        for key in def_info:
            if 'drop' in def_info[key]:
                if index >= def_info[key]['start'] and index <= def_info[key]['end']:
                    # #print('1', line)
                    # #print('6#', line)
                    str_ += '#'
                    str_ += line
                    str_ += '\n'
                    is_def_drop = True
        if is_def_drop:
            continue
        if index in need_remove_index:
            # #print('2', index, line)
            # #print('5#', line)
            str_ += '#'
            str_ += line
            str_ += '\n'
        else:
            # str_ += '#'
            str_ += line
            str_ += '\n'
            #print('@1', line)
            # if len(is_suojin) > 0: 
            #     if index == suojin_index[-1] + next_line:
            #         all_blank=True
            #         for char in line:
            #             if ord(char) != 32:
            #                 all_blank=False
            #         if all_blank:
            #             next_line +=1
            #             continue
            #         suojin_num[-1] =0
            #         for char in line:
            #             if ord(char) == 32:
            #                 suojin_num[-1] += 1
            #             elif ord(char) == 9:
            #                 suojin_num[-1] += 4
            #             else:
            #                 break
            # if line[0:4] == 'for ' or line[0:3] ==  'if ' or line[0:6] == 'while ' in line or line.strip()[0:5] == 'else:':
            #     #print('00#', line)
            #     is_suojin.append(True)
            #     suojin_index.append(index)
            #     suojin_num.append(0)
            # continue  
                # #print('is_suojin', line)
        # else:
        #     for variable in need_remove_variables:
        #         if variable in line and 'def' not in line:
        #             if line[0:4] == 'for ' or line[0:3] ==  'if ' or line[0:6] == 'while ' in line or line.strip()[0:5] == 'else:':
        #                 #print('4#', index, line)
        #                 str_ += '#'
        #                 str_ += line
        #                 str_ += '\n'
                        
        #                 is_suojin = True
        #                 suojin_index = index
        #                 is_for = True
        #                 break


        # if is_for:
        #     continue

        # if is_suojin and index == suojin_index + next_line:
        #     all_blank=True
        #     for char in line:
        #         if ord(char) != 32:
        #             all_blank=False
        #     if all_blank:
        #         next_line +=1
        #         continue
        #     suojin_num =0
        #     for char in line:
        #         if ord(char) == 32:
        #             suojin_num += 1
        #         elif ord(char) == 9:
        #             suojin_num += 4
        #         else:
        #             break
        #     #print('1#', line)
        #     str_ += '#'
        #     str_ += line
        #     str_ += '\n'
        # elif is_suojin:
        #     all_blank=True
        #     for char in line:
        #         if ord(char) != 32:
        #             all_blank=False
        #     if all_blank:
        #         continue
        #     line_suojin = 0
        #     for char in line:
        #         if ord(char) == 32:
        #             line_suojin += 1
        #         elif ord(char) == 9:
        #             line_suojin += 4
        #         else:
        #             break
        #     if line_suojin == suojin_num:
        #         #print('3#',line_suojin, suojin_num, line)
        #         str_ += '#'
        #         str_ += line
        #         str_ += '\n'
        #     else:
        #         #print('@2', line)
        #         is_suojin = False
        #         suojin_num = 0
        #         str_ += line
        #         str_ += '\n'
        #         next_line = 1
        # else:
        #     #print('@1', line)
        #     str_ += line
        #     str_ += '\n'
    # #print(str_)
    return str_

# def remove_model_2(code):
#     lines = code.split('\n')
#     code_res = ''
#     for line in lines:
#         temp= ''
#         ret = re.findall('sns\.',line)
#         if len(ret)>0:
#             temp ='#'+line+'\n'
#             #print(temp)
#         else :
#             temp = line + '\n'
#         code_res+=temp
#     return code_res
def remove_model_2(code):
    lines = code.split('\n')
    code_res = ''
    for line in lines:
        line = line.replace("LogisticRegression(","LogisticRegression(solver='liblinear',")
        line = line.replace("display(","#display(")
        code_res+=line+"\n"
    return code_res