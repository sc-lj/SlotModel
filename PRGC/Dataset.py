# -*- encoding: utf-8 -*-
'''
@File    :   Dataset.py
@Time    :   2022/08/29 19:09:38
@Author  :   lujun
@Version :   1.0
@License :   (C)Copyright 2021-2022, Liugroup-NLPR-CASIA
@Desc    :   None
'''
import os
import torch
import json
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils.utils import find_head_idx
from collections import defaultdict
from torch.utils.data import DataLoader, Dataset
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast


class InputExample(object):
    """a single set of samples of data
    """

    def __init__(self, text, en_pair_list, relations, rel2ens):
        self.text = text
        self.en_pair_list = en_pair_list
        self.relations = relations
        self.rel2ens = rel2ens


class PRGCDataset(Dataset):
    def __init__(self, args, is_training):
        super().__init__()
        self.args = args
        self.tokenizer = BertTokenizerFast.from_pretrained(
            args.pretrain_path, cache_dir="./bertbaseuncased")
        self.is_training = is_training
        self.batch_size = args.batch_size
        with open(os.path.join(args.data_dir, "intent_label.json"), 'r') as f:
            intents = json.load(f)
        with open(os.path.join(args.data_dir, "slot_name.json"), 'r') as f:
            slot2id = json.load(f)
        self.slot2id = slot2id
        
        self.rel2id = intents
        self.rels_set = list(self.rel2id.values())
        self.intent_number = len(self.rel2id)
        if is_training:
            filenames = os.path.join(args.data_dir, "train_data.csv")
        else:
            filenames = os.path.join(args.data_dir, "val_data.csv")
        
        data = pd.read_csv(filenames)
        lines = data[['intent','target',"new_text"]].to_dict(orient="records")
        self.datas = self.preprocess(lines)

    def preprocess(self, lines):
        examples = []
        for sample in lines:
            text = sample['new_text']
            en_pair_list = []
            relations = self.rel2id[sample['intent']]
            rel2ens = defaultdict(list)
            for triple in eval(sample['target']):
                # {'entity': 'time', 'start': 0, 'end': 2, 'group': None, 'value': '三点'}
                entity_type = triple['entity']
                entity_value = triple['value']
                start = triple['start']
                en_pair_list.append([entity_value,entity_type,start])
                group = triple['group']
                if group is None:
                    group = 0
                rel2ens[group].append(entity_value)
            example = InputExample(text=text, en_pair_list=en_pair_list, relations=relations, rel2ens=rel2ens)
            examples.append(example)
        max_text_len = min(self.args.max_length+2,512)
        # multi-process
        # with Pool(10) as p:
        #     convert_func = functools.partial(self.convert, max_text_len=max_text_len, tokenizer=self.tokenizer, rel2idx=self.rel2id,
        #                                     ensure_rel=self.args.ensure_rel,num_negs=self.args.num_negs)
        #     features = p.map(func=convert_func, iterable=examples)
        # # return list(chain(*features))
        features = []
        for example in tqdm(examples, desc="convert example"):
            feature = self.convert(example, max_text_len=max_text_len, tokenizer=self.tokenizer, rel2idx=self.rel2id,
                                   ensure_rel=self.args.ensure_rel, num_negs=self.args.num_negs)
            features.extend(feature)
        return features

    def convert(self, example: InputExample, max_text_len: int, tokenizer, rel2idx, ensure_rel, num_negs):
        """转换函数 for CarFaultRelation data
        Args:
            example (_type_): 一个样本示例
            max_text_len (_type_): 样本的最大长度
            tokenizer (_type_): _description_
            rel2idx (dict): 关系的索引
            ex_params (_type_): 额外的参数
        Returns:
            _type_: _description_
        """
        outputs = tokenizer(example.text,return_offsets_mapping=True)
        text_tokens = ["[CLS]"]+tokenizer.tokenize(example.text)+["[SEP]"]
        
        # token to id
        input_ids =outputs['input_ids']
        offset = outputs['offset_mapping']
        attention_mask = outputs['attention_mask']
        # zero-padding up to the sequence length
        if len(input_ids) < max_text_len:
            pad_len = max_text_len - len(input_ids)
            # token_pad_id=0
            input_ids += [0] * pad_len
            attention_mask += [0] * pad_len

        # construct tags of correspondence and relation
        # subject和object相关性 target
        relations = example.relations
        tags_sub = max_text_len * [self.slot2id['O']]
        entity_index = defaultdict(list)
        for ent,ent_type,ent_start in example.en_pair_list:
            sub = tokenizer(ent,add_special_tokens=False)['input_ids']
            ent_head = find_head_idx(input_ids, sub, ent_start,offset)
            tags_sub[ent_head] = self.slot2id['{}-B'.format(ent_type)]
            tags_sub[ent_head + 1:ent_head + len(sub)] = (len(sub) - 1) * [self.slot2id['{}-I'.format(ent_type)]]
            entity_index[ent].append(ent_head)
            
        sub_feats = []
        corres_tag = np.zeros((max_text_len, max_text_len))
        # positive samples，标记subject和object的序列
        for rel, en_ll in example.rel2ens.items():
            if rel ==0:
                continue
            # 两个实体的头部相交位置为1，表示其存在关系
            ent_index = [entity_index[en].pop(0)  for en in en_ll]
            corres_tag[ent_index[0]][ent_index[1]] = 1

        sub_feats.append(InputFeatures(
            input_tokens=text_tokens,
            input_ids=input_ids,
            attention_mask=attention_mask,
            corres_tag=corres_tag,
            seq_tag=tags_sub,
            relation=relations,
            text=example.text
            ))
        return sub_feats
    
    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        data = self.datas[index]
        return data


class InputFeatures(object):
    """
    Desc:
        a single set of features of data
    """

    def __init__(self, text, input_tokens, input_ids, attention_mask, seq_tag=None, corres_tag=None, relation=None, triples=None, rel_tag=None):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.seq_tag = seq_tag
        self.corres_tag = corres_tag
        self.relation = relation
        self.triples = triples
        self.rel_tag = rel_tag
        self.text = text


def collate_fn(features):
    """将InputFeatures转换为Tensor
    Args:
        features (List[InputFeatures])
    Returns:
        tensors (List[Tensors])
    """
    input_ids = np.array([f.input_ids for f in features],dtype=np.int64)
    input_ids = torch.from_numpy(input_ids)
    attention_mask = np.array([f.attention_mask for f in features],dtype=np.int64)
    attention_mask = torch.from_numpy(attention_mask)
    seq_tags = np.array([f.seq_tag for f in features],dtype=np.int64)
    seq_tags = torch.from_numpy(seq_tags)
    poten_relations = np.array([f.relation for f in features],dtype=np.int64)
    poten_relations = torch.from_numpy(poten_relations)
    corres_tags = np.array([f.corres_tag for f in features],dtype=np.int64)
    corres_tags = torch.from_numpy(corres_tags)
    texts = [f.text for f in features]
    input_tokens = [f.input_tokens for f in features]
    tensors = [input_ids, attention_mask, seq_tags, poten_relations, corres_tags,texts,input_tokens]
    return tensors


