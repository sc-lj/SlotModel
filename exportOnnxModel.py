import os
import yaml
import sys
import numpy as np 
import pandas as pd 

class Dict(dict):
    def __getattr__(self, key):
        return self.get(key)

    def __setattr__(self, key, value):
        self[key] = value


def npsigmoid(x):
    return 1/(1+np.exp(-x))

def softmax(x):
    # max = np.max(x,axis=1,keepdims=True) #returns max of each row and keeps same dims
    e_x = np.exp(x) #subtracts each row with its max value
    sum = np.sum(e_x,axis=1,keepdims=True) #returns sum of each row and keeps same dims
    f_x = e_x / sum 
    return f_x

def get_chunk_type(tok, idx_to_tag):
    """
    Args:
        tok: id of token, ex 4
        idx_to_tag: dictionary {4: "B-PER", ...}
    Returns:
        tuple: "B", "PER"
    """
    tag_name = idx_to_tag[tok]
    content = tag_name.split('-')
    tag_class = content[0]
    if len(content) == 1:
        return tag_class
    bi = content[-1]
    return tag_class, bi


def get_chunks(seq, tags):
    """Given a sequence of tags, group entities and their position
    Args:
        seq: np.array[4, 4, 0, 0, ...] sequence of labels
        tags: dict["O"] = 4
    Returns:
        list of (chunk_type, chunk_start, chunk_end)
    Example:
        seq = [4, 5, 0, 3]
        tags = {"B-PER": 4, "I-PER": 5, "B-LOC": 3}
        result = [("PER", 0, 2), ("LOC", 3, 4)]
    """
    default1 = tags['O']
    idx_to_tag = {idx: tag for tag, idx in tags.items()}
    chunks = []
    chunk_type, chunk_start = None, None
    for i, tok in enumerate(seq):
        # End of a chunk 1
        if tok == default1 and chunk_type is not None:
            # Add a chunk.
            chunk = (chunk_type, chunk_start, i)
            chunks.append(chunk)
            chunk_type, chunk_start = None, None

        # End of a chunk + start of a chunk!
        elif tok != default1:
            res = get_chunk_type(tok, idx_to_tag)
            if len(res) == 1:
                continue
            tok_chunk_class, bi = get_chunk_type(tok, idx_to_tag)
            tok_chunk_type = bi
            if chunk_type is None:
                chunk_type, chunk_start = tok_chunk_class, i
            elif tok_chunk_class != chunk_type or tok_chunk_type == "B":
                chunk = (chunk_type, chunk_start, i)
                chunks.append(chunk)
                chunk_type, chunk_start = tok_chunk_class, i
        else:
            pass
    # end condition
    if chunk_type is not None:
        chunk = (chunk_type, chunk_start, len(seq))
        chunks.append(chunk)

    return chunks

def span2str(triples, tokens,offset):
    def _concat(token_list):
        result = ''
        for idx, t in enumerate(token_list):
            if t =='[CLS]':
                t = ''
            if idx == 0:
                result = t
            elif t.startswith('##'):
                result += t.lstrip('##')
            else:
                result += '' + t
        return result

    output = []
    for triple in triples:
        typename= triple[0]
        sub_tokens = tokens[triple[1]:triple[-1]]
        sub = _concat(sub_tokens)
        start = offset[triple[1]][0]
        end = offset[triple[-1]][0]
        output.append((typename,sub,start,end))
    return output


def export_group_data(text,entities,group_infos):
    group_number = len(group_infos)
    
        

def eval_prgc_model(onnx_model_file,eval_file):
    import onnxruntime
    import json
    import numpy as np 
    from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast
    session = onnxruntime.SessionOptions()
    session.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    providers = ['CUDAExecutionProvider']
    session = onnxruntime.InferenceSession(onnx_model_file, session,providers=providers)
    with open(os.path.join("data/intent_label.json"), 'r') as f:
        intents = json.load(f)
    with open(os.path.join("data/slot_name.json"), 'r') as f:
        slot2id = json.load(f)
    
    id2rel = {i:v  for v,i in intents.items()}
    data = pd.read_excel(eval_file)
    texts = data['text'].values.tolist()
    number = len(data)
    tokenizer = BertTokenizerFast.from_pretrained("./rtb3")
    batch_size = 10
    pred_result = []
    for i in range(0,number,batch_size):
        batch_text = texts[i:i+batch_size]
        outputs = tokenizer(batch_text,truncation=True,padding=True,max_length=512,return_offsets_mapping=True)
        text_tokens = [["[CLS]"]+tokenizer.tokenize(t)+["[SEP]"] for t in batch_text]
        input_ids =outputs['input_ids']
        offsets = outputs['offset_mapping']
        attention_mask = outputs['attention_mask']
        input_ids = np.array(input_ids).astype(np.int32)
        attention_mask = np.array(attention_mask).astype(np.float32)
        output_sub,intent_pred_hidden,corres_pred,corres_mask = session.run(None,{"input_ids":input_ids,"attention_mask":attention_mask})
        pred_seq_prob = softmax(output_sub)
        pred_seqs = np.argmax(pred_seq_prob, axis=-1)
        attention_mask[:,0]=0
        attention_mask[:,-1]=0
        pred_seqs = pred_seqs*attention_mask
        corres_preds = npsigmoid(corres_pred) * corres_mask
        pred_rels = np.argmax(softmax(intent_pred_hidden),-1)
        bs = len(batch_text)
        for idx in range(bs):
            text = batch_text[idx]
            offset = offsets[idx]
            intent_name = id2rel[pred_rels[idx]]
            pre_tag = pred_seqs[idx]
            pred_corre = corres_preds[idx]
            pred_chunks_sub = get_chunks(pre_tag, slot2id)
            pre_entities = span2str(pred_chunks_sub, text_tokens[idx],offset)
            pred_group_entity = [(pre_entities[i], pre_entities[j]) for i,h in enumerate(pred_chunks_sub) for j,t in enumerate(pred_chunks_sub) if pred_corre[h[1]][t[1]] == 1]
            # [{'entity': 'amount', 'start': 0, 'end': 1, 'group': None, 'value': '1'}]
            entities_info = {}
            had_group = set()
            if len(pred_group_entity)!=0:
                for i in range(len(pred_group_entity)):
                    group_entity = pred_group_entity[i]
                    for g in group_entity:
                        entities_info['entity'] = g[0]
                        entities_info['value'] = g[1]
                        entities_info['start'] = g[2]
                        entities_info['end'] = g[3]
                        entities_info['group'] = i
                        had_group.add("".join(map(str,g)))
            
            for o in pre_entities:
                strs = "".join(map(str,o))
                if strs in had_group:
                    continue
                else:
                    entities_info['entity'] = o[0]
                    entities_info['value'] = o[1]
                    entities_info['start'] = o[2]
                    entities_info['end'] = o[3]
                    entities_info['group'] = None
            entities_info = sorted(entities_info,key=lambda x:x['start'])
            pred_result.append({"text":text,"res":entities_info,"intent_name":intent_name})
    pred_result = pd.DataFrame(pred_result)
    csv_file = eval_file.replace("xlsx","csv")
    pred_result.to_csv(csv_file,index=False)

def repair_group_info(entities):
    pass

def change_prgc_model_type(model_file):
    """[将pytorch_lightning保存的模型格式切换为原torch.save保存的格式]

    Args:
        model_file ([type]): [description]
    """

    parent_path = os.path.dirname(os.path.dirname(model_file))
    sys.path.append(os.path.abspath(parent_path))
    from  PRGC.Model import PRGCPytochLighting
    import torch
    path = os.path.dirname(model_file)
    
    hparams_file = os.path.join(parent_path,"hparams.yaml")
    with open(hparams_file,'r') as f:
        params = yaml.load(f,Loader=yaml.FullLoader)
    new_params = Dict(params)
    model = PRGCPytochLighting.load_from_checkpoint(model_file,strict=True,args=new_params)
    model = model.model
    
    torch.save(model.state_dict(),path+"/best_model.pt")
    model.eval()  # 不进行梯度传递
    # device = torch.device("cuda")
    device = torch.device("cpu")
    # model = model.cuda()
    input_ids = torch.zeros((1, 512), dtype=torch.int32).to(device)  # 网络输入大小
    attention_mask = torch.zeros((1, 512), dtype=torch.float32).to(device)  # 网络输入大小
    token_type_ids = torch.zeros((1, 512), dtype=torch.int32).to(device)  # 网络输入大小
    model_path = path+"/best_model.onnx"
    with torch.no_grad():
        torch.onnx.export(model,
                        (input_ids, attention_mask),
                        model_path,
                        export_params=True,  # 是否保存模型的训练好的参数
                        verbose=False,  # 是否输出debug描述
                        input_names=['input_ids', 'attention_mask'], # 定义输入结点的名字，有几个输入就定义几个
                        output_names=['entity_pred',"intent_prd","corres_pred", "corres_mask"],  # 定义输出结点的名字
                        dynamic_axes={"input_ids": {0: "batch_size", 1: "input_dim"}, 
                                      "attention_mask": {0: "batch_size", 1: "input_dim"},
                                      "entity_pred":{0: "batch_size"},
                                      "intent_prd":{0: "batch_size"},
                                      "corres_pred": {0: "batch_size",1:"input_dim"},
                                      "corres_mask": {0: "batch_size",1:"input_dim"}
                                      },
                        opset_version=12,  # onnx opset的库版本
                        # whether do constant-folding optimization 该优化会替换全为常数输入的分支为最终结果
                        do_constant_folding=True,
                        operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
                        #   operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK
                        )


    from onnxruntime_tools import optimizer
    from onnxruntime_tools.transformers.onnx_model_bert import BertOptimizationOptions
    optimization_options = BertOptimizationOptions("bert")
    optimization_options.enable_skip_layer_norm=False
    optimized_model = optimizer.optimize_model(model_path, model_type='bert',optimization_options=optimization_options, num_heads=12, hidden_size=768,use_gpu=False)
    optimized_model.convert_model_float32_to_float16()
    model_path = path+"/best_model_float16.onnx"
    optimized_model.save_model_to_file(model_path)
    import onnxruntime as ort
    infer_session = ort.InferenceSession(model_path,providers=["CUDAExecutionProvider"])



if __name__ == "__main__":
    model_path_file = "lightning_logs/prgc/version_0/checkpoints/epoch=13int_acc=0.991n_f1=0.934n_acc=0.931n_rec=0.937g_f1=0.811g_acc=1.000g_rec=0.682.ckpt"
    # change_prgc_model_type(model_path_file)
    onnx_model_file = "lightning_logs/prgc/version_0/checkpoints/best_model.onnx"
    eval_file = "data/提槽测试题1W条-1020.xlsx"
    eval_prgc_model(onnx_model_file,eval_file)



