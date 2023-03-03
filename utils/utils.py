#! -*- coding:utf-8 -*-

import json
import pandas as pd 
def statistics_text_length(filename,tokenizer):
    data = pd.read_csv(filename)
    lines = data[['text']].to_dict(orient="records")
    max_length = 0
    for line in lines:
        max_length = max(max_length,len(tokenizer.tokenize(line['text'])))
    
    return max_length


def rematch(offsets):
    mapping = []
    for offset in offsets:
        if offset[0] == 0 and offset[1] == 0:
            mapping.append([])
        else:
            mapping.append([i for i in range(offset[0], offset[1])])
    return mapping


def find_head_idx(source, target, ent_start, offset):
    sub_index = 0
    for i,offs in enumerate(offset[:-1]):
        if offs[0] == ent_start:
            sub_index = i
    target_len = len(target)
    for i in range(sub_index,len(source)):
        if source[i: i + target_len] == target:
            return i
    return -1

def update_arguments(args,config):
    """将config中的参数更新到args中
    Args:
        args ([type]): [description]
        config ([type]): [description]
    """
    for key,value in config.items():
        args.__setattr__(key,value)
    return args
