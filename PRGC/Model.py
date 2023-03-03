import os
import json
import torch
import numpy as np
import torch.nn as nn
from collections import Counter
import pytorch_lightning as pl
import torch.nn.functional as F
from utils.loss_func import GlobalCrossEntropy
from transformers import BertPreTrainedModel, BertModel
from PRGC.utils import get_chunks
import math

class MultiNonLinearClassifier(nn.Module):
    def __init__(self, hidden_size, tag_size, dropout_rate):
        super(MultiNonLinearClassifier, self).__init__()
        self.tag_size = tag_size
        self.linear = nn.Linear(hidden_size, int(hidden_size / 2))
        self.hidden2tag = nn.Linear(int(hidden_size / 2), self.tag_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_features):
        features_tmp = self.linear(input_features)
        features_tmp = nn.ReLU()(features_tmp)
        features_tmp = self.dropout(features_tmp)
        features_output = self.hidden2tag(features_tmp)
        return features_output


class ConditionalLayerNorm(nn.Module):
    def __init__(self,normalized_shape,cond_shape,eps=1e-12):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.Tensor(normalized_shape))
        self.bias = nn.Parameter(torch.Tensor(normalized_shape))
        self.weight_dense = nn.Linear(cond_shape, normalized_shape, bias=False)
        self.bias_dense = nn.Linear(cond_shape, normalized_shape, bias=False)
        self.reset_weight_and_bias()

    def reset_weight_and_bias(self):
        """
        此处初始化的作用是在训练开始阶段不让 conditional layer norm 起作用
        """
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)

        nn.init.zeros_(self.weight_dense.weight)
        nn.init.zeros_(self.bias_dense.weight)

    def forward(self, inputs, cond=None):
        # cond = torch.unsqueeze(cond, 1)  # (b, 1, h*2)
        weight = self.weight_dense(cond) + self.weight  # (b, 1, h)
        bias = self.bias_dense(cond) + self.bias  # (b, 1, h)
        mean = torch.mean(inputs, dim=-1, keepdim=True)  # （b, s, 1）
        outputs = inputs - mean  # (b, s, h)
        variance = torch.mean(outputs ** 2, dim=-1, keepdim=True)
        std = torch.sqrt(variance + self.eps)  # (b, s, 1)
        outputs = outputs / std  # (b, s, h)
        outputs = outputs*weight + bias
        return outputs


class biaffine(nn.Module):
    def __init__(self, in_size, bias_x=True, bias_y=True):
        super().__init__()
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.U = torch.nn.Parameter(torch.randn(in_size + int(bias_x), 1, in_size + int(bias_y)))

    def forward(self, x, y):
        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), dim=-1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), dim=-1)

        bilinar_mapping = torch.einsum('bxi,ioj,byj->bxyo', x, self.U, y)
        return bilinar_mapping


class BiaffineTagger(nn.Module):
    def __init__(self, hidden_size,biaffine_hidden):
        super(BiaffineTagger, self).__init__()
        self.start_layer = nn.Linear(hidden_size, biaffine_hidden)
        self.end_layer = nn.Linear(hidden_size, biaffine_hidden)
        self.biaffne_layer = biaffine(biaffine_hidden)

    def forward(self, hidden):
        start_logits = self.start_layer(hidden)
        end_logits = self.end_layer(hidden)
        span_logits = self.biaffne_layer(start_logits, end_logits)
        span_logits = span_logits.squeeze(-1).contiguous()
        return span_logits
    
class GlobalCorres(nn.Module):
    # 使用BiaffineTagger计算global Correspondence
    def __init__(self,config,params) -> None:
        super().__init__()
        self.ap_tagger = BiaffineTagger(config.hidden_size, params.biaffine_hidden)
        self.layernormal = nn.LayerNorm(config.hidden_size)

    def forward(self,sequence_output,attention_mask,entity_ids):
        entiti_hidden = sequence_output*entity_ids.unsqueeze(-1) # [batch_size,seq_len,hidden_size]
        entiti_hidden = torch.mean(entiti_hidden,dim=1,keepdim=True)
        hiddens = sequence_output+entiti_hidden
        hidden = self.layernormal(hiddens)
        pred_corres = self.ap_tagger(hidden)
        mask_tmp1 = attention_mask.unsqueeze(-1)
        mask_tmp2 = attention_mask.unsqueeze(1)
        corres_mask = mask_tmp1 * mask_tmp2
        return pred_corres,corres_mask

class GlobalCorres_v1(nn.Module):
    # 使用BiaffineTagger计算global Correspondence,加入实体类型embedding
    def __init__(self,config,params) -> None:
        super().__init__()
        self.ap_tagger = BiaffineTagger(config.hidden_size, params.biaffine_hidden)
        self.layernormal = nn.LayerNorm(config.hidden_size)
        self.tag_size = math.ceil(params.seq_tag_size/2)
        self.tag_type_embedding = nn.Embedding(self.tag_size,config.hidden_size,padding_idx=0)
        self.tag_linear = nn.Linear(config.hidden_size,config.hidden_size)
        self.linear = nn.Linear(config.hidden_size,config.hidden_size)
        self.linear_rel = nn.Linear(config.hidden_size,config.hidden_size)

    def forward(self,sequence_output,attention_mask,entity_type,rel_emb):
        tag_type_embedding = self.tag_type_embedding(entity_type)
        tag_type_embedding = self.tag_linear(tag_type_embedding)
        rel_emb = self.linear_rel(rel_emb)
        rel_emb = rel_emb.unsqueeze(1)
        # tag_type_embedding = torch.mean(tag_type_embedding,dim=1,keepdim=True)
        hiddens = sequence_output+tag_type_embedding # [batch_size,seq_len,hidden_size]
        hiddens = self.linear(hiddens)
        hiddens = self.layernormal(hiddens)
        pred_corres = self.ap_tagger(hiddens)
        mask_tmp1 = attention_mask.unsqueeze(-1)
        mask_tmp2 = attention_mask.unsqueeze(1)
        corres_mask = mask_tmp1 * mask_tmp2
        return pred_corres,corres_mask

class EntityModel(nn.Module):
    def __init__(self,config,params) -> None:
        super().__init__()
        self.emb_fusion = params.emb_fusion
        self.intent_num = params.intent_number
        self.seq_tag_size = params.seq_tag_size
        self.sequence_tagging_sub = MultiNonLinearClassifier(config.hidden_size, self.seq_tag_size, params.drop_prob)
        self.cond_layer = ConditionalLayerNorm(config.hidden_size,config.hidden_size)
        self.linear = nn.Linear(config.hidden_size*2,config.hidden_size)
        self.activate_layer = nn.ReLU6()
        self.intent_linear = nn.Linear(config.hidden_size,config.hidden_size)
        
    def forward(self,sequence_output,rel_emb,attention_mask):
        bs, seq_len, h = sequence_output.size()
        # 获取关系的embedding信息
        # rel_emb = self.rel_embedding(potential_rels)
        rel_emb = self.intent_linear(rel_emb)
        # relation embedding vector fusion
        rel_emb = rel_emb.unsqueeze(1).expand(-1, seq_len, h)
        if self.emb_fusion == 'concat':
            # 将关系的信息与句子的token信息融合在一起
            decode_input = torch.cat([sequence_output, rel_emb], dim=-1)
            decode_input = self.linear(decode_input)
        elif self.emb_fusion == 'sum':
            # (bs/sum(x_i), seq_len, h)
            decode_input = sequence_output + rel_emb
        elif self.emb_fusion == "normal":
            # (bs/sum(x_i), seq_len, h)
            decode_input = self.cond_layer(sequence_output,rel_emb)*attention_mask.unsqueeze(-1)
            # decode_input = self.intermediate_label_layer(decode_input)
            decode_input = self.activate_layer(decode_input)
        output_sub = self.sequence_tagging_sub(decode_input)
        
        return output_sub


class PRGC(BertPreTrainedModel):
    def __init__(self, config, params):
        super().__init__(config)
        self.seq_tag_size = params.seq_tag_size
        self.intent_num = params.intent_number
        # pretrain model
        self.bert = BertModel(config)
        # relation judgement
        self.intent_judgement = MultiNonLinearClassifier(config.hidden_size, self.intent_num, params.drop_prob)
        self.global_corres = GlobalCorres(config,params)
        self.global_corres_v1 = GlobalCorres_v1(config,params)
        self.entity_model = EntityModel(config,params)
        self.rel_embedding = nn.Embedding(self.intent_num, config.hidden_size)
        self.init_weights()

    @staticmethod
    def masked_avgpool(sent, mask):
        mask_ = mask.masked_fill(mask == 0, -1e9).float()
        score = torch.softmax(mask_, -1)
        return torch.matmul(score.unsqueeze(1), sent).squeeze(1)
    
    def forward(self,input_ids,attention_mask,potential_intents=None,seq_tags=None):
        """
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            rel_tags: (bs, rel_num)
            potential_intents: (bs,), only in train stage.
            seq_tags: (bs, 2, seq_len)
            corres_tags: (bs, seq_len, seq_len)
            ex_params: experiment parameters
        """
        batch,seq_len = input_ids.shape
        # pre-train model
        outputs = self.bert(input_ids,attention_mask=attention_mask,output_hidden_states=True)  # sequence_output, pooled_output, (hidden_states), (attentions)
        sequence_output = outputs[0]
        pooled_output = outputs[1]
        # 预测关系 (bs, h)
        # h_k_avg = self.masked_avgpool(sequence_output, attention_mask)
        # rel_pred = self.rel_judgement(h_k_avg) # (bs, rel_num)
        intent_pred_hidden = self.intent_judgement(pooled_output) # [bs, rel_num]
        # corres_pred,corres_mask = self.global_corres_v0(sequence_output,attention_mask,seq_len)
        if potential_intents is None:
            # 使用预测的关系获取关系的embedding信息，也可以使用gold关系获取关系的embedding信息
            # (bs, rel_num)
            intent_pred_prob = torch.softmax(intent_pred_hidden,-1)
            intent_pred = intent_pred_prob.argmax(-1)
            potential_intents = intent_pred
        
        rel_emb = self.rel_embedding(potential_intents)

        output_sub = self.entity_model(sequence_output,rel_emb,attention_mask)
        attention_mask = attention_mask.clone()
        attention_mask[:,0] = 0 
        attention_mask[:,-1] = 0
        if seq_tags is None:
            entity_pred_prob = torch.softmax(output_sub,-1)
            entity_pred = entity_pred_prob.argmax(-1)
            entity_pred = entity_pred*attention_mask
            entity_pred_index = (entity_pred>0).int()
            entity_ids = entity_pred_index
            # 向上取整
            entity_type = torch.ceil(entity_pred/2).long()
        else:
            entity_ids = (seq_tags>0).int()
            entity_type = torch.ceil(seq_tags/2).long()
        # corres_pred,corres_mask = self.global_corres(sequence_output,attention_mask,entity_ids)
        corres_pred,corres_mask = self.global_corres_v1(sequence_output,attention_mask,entity_type,rel_emb)
        return output_sub,intent_pred_hidden,corres_pred,corres_mask


class PRGCPytochLighting(pl.LightningModule):
    def __init__(self,args) -> None:
        super().__init__()
        self.seq_tag_size = args.seq_tag_size
        
        with open(os.path.join(args.data_dir, "intent_label.json"), 'r') as f:
            relation = json.load(f)
        self.id2rel = {v:k for k,v in relation.items()} 
        with open(os.path.join(args.data_dir, "slot_name.json"), 'r') as f:
            slot2id = json.load(f)
        self.slot2id = slot2id
        
        self.model = PRGC.from_pretrained(args.pretrain_path,args)
        
        self.save_hyperparameters(args)
        self.corres_threshold = 0.
        self.corres_global_loss_func = GlobalCrossEntropy()
        self.rel_loss_func = nn.CrossEntropyLoss(reduction='mean')
        self.ent_loss_func = nn.CrossEntropyLoss(reduction='none')
        self.args = args
        self.epoch = 0

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)

    def compute_kl_loss(self, p, q, pad_mask=None):
        p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
        q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')

        if pad_mask is not None:
            p_loss.masked_fill_(pad_mask.unsqueeze(-1).type(torch.bool), 0.)
            q_loss.masked_fill_(pad_mask.unsqueeze(-1).type(torch.bool), 0.)

        p_loss = p_loss.sum()
        q_loss = q_loss.sum()
        loss = (p_loss + q_loss) / 2
        return loss

    def training_step(self, batches,batch_idx):
        # input_id,attention_mask,实体序列id，意图，group
        input_ids, attention_mask, seq_tags, relation, corres_tags,_,_ = batches
        bs = input_ids.shape[0]
        # seq_tag_ids = torch.nonzero(seq_tags>0).squeeze()
        # compute model output and loss
        output_sub,rel_pred_hidden,corres_pred,corres_mask = self.model(input_ids,attention_mask,relation,seq_tags)
        # calculate loss
        pos_attention_mask = attention_mask.view(-1)
        # sequence label loss
        loss_seq_sub = (self.ent_loss_func(output_sub.view(-1, self.seq_tag_size), seq_tags.reshape(-1)) * pos_attention_mask).sum() / pos_attention_mask.sum()
        corres_pred = corres_pred.view(bs, -1)
        corres_mask = corres_mask.view(bs, -1)
        corres_tags = corres_tags.view(bs, -1)

        # loss_matrix = (self.corres_loss_func(corres_pred,corres_tags.float()) * corres_mask).sum() / corres_mask.sum()
        loss_matrix = self.corres_global_loss_func(corres_pred,corres_tags.float())
        loss_rel = self.rel_loss_func(rel_pred_hidden, relation)
        loss = loss_seq_sub + loss_matrix + loss_rel
        return loss
    
    def validation_step(self,batches,batch_idx):
        input_ids, attention_mask, seq_tags, relations, corres_tags,texts,input_tokens = batches
        # compute model output and loss
        output_sub,rel_pred_hidden,corres_pred,corres_mask = self.model(input_ids, attention_mask=attention_mask)
        # (sum(x_i), seq_len)
        pred_seq_sub = torch.argmax(torch.softmax(output_sub, dim=-1), dim=-1)
        pred_seqs = pred_seq_sub.cpu().numpy()
        corres_pred = torch.sigmoid(corres_pred) * corres_mask
        pred_rels = torch.argmax(torch.softmax(rel_pred_hidden,-1),-1)
        relations = relations.cpu().numpy()
        corres_pred = corres_pred.cpu().numpy()
        pred_rels = pred_rels.cpu().numpy()
        seq_tags = seq_tags.cpu().numpy()
        corres_tags = corres_tags.cpu().numpy()
        return texts, pred_seqs, corres_pred, pred_rels,relations,input_tokens,corres_tags,seq_tags
    
    def validation_epoch_end(self, outputs):
        os.makedirs(os.path.join(self.args.output_path,
                    self.args.model_type), exist_ok=True)
        writer = open(os.path.join(self.args.output_path, self.args.model_type,
                      'val_output_{}.json'.format(self.epoch)), 'w')

        predictions = []
        ground_truths = []
        entity_correct_num, entity_predict_num, entity_gold_num = 0, 0, 0
        group_correct_num, group_predict_num, group_gold_num = 0, 0, 0
        error_result = []
        rel_correct_num = 0
        rel_num = 0
        orders = ["ner_type","ner_name"]
        for texts, pred_seqs, pred_corres, pred_rels,relations,input_tokens,corres_tags,seq_tags in outputs:
            bs = len(texts)
            for idx in range(bs):
                error_type = []
                text = texts[idx]
                rel_num += 1
                pre_rel_name = self.id2rel[pred_rels[idx]]
                tar_rel_name = self.id2rel[relations[idx]]
                rel_correct = False
                if pre_rel_name == tar_rel_name:
                    rel_correct_num += 1
                    rel_correct = True
                else:error_type.append("intent")
                pre_tag = pred_seqs[idx]
                tar_tag = seq_tags[idx]
                pred_corre = pred_corres[idx]
                corres_tag = corres_tags[idx]
                pred_chunks_sub = get_chunks(pre_tag, self.slot2id)
                target_chunks_sub = get_chunks(tar_tag, self.slot2id)
                
                gold_entities = self.span2str(target_chunks_sub, input_tokens[idx])
                pre_entities = self.span2str(pred_chunks_sub, input_tokens[idx])
                
                pred_group_entity = [(pre_entities[i], pre_entities[j]) for i,h in enumerate(pred_chunks_sub) for j,t in enumerate(pred_chunks_sub) if pred_corre[h[1]][t[1]] == 1]
                target_group_entity = [(gold_entities[i], gold_entities[j]) for i,h in enumerate(target_chunks_sub) for j,t in enumerate(target_chunks_sub) if corres_tag[h[1]][t[1]] == 1]


                ground_truths.append(gold_entities)
                predictions.append(pre_entities)
                # counter
                entity_correct_num += len(set(pre_entities) & set(gold_entities))
                entity_predict_num += len(set(pre_entities))
                entity_gold_num += len(set(gold_entities))
                
                group_correct_num += len(set(pred_group_entity) & set(target_group_entity))
                group_predict_num += len(set(pred_group_entity))
                group_gold_num += len(set(target_group_entity))
                
                ner_new = [dict(zip(orders, triple)) for triple in set(pre_entities) - set(gold_entities)]
                ner_lack = [dict(zip(orders, triple)) for triple in set(gold_entities) - set(pre_entities)]
                ner_pred = True
                if len(ner_new) or len(ner_lack):
                    ner_pred = False
                    error_type.append("ner")

                ner_group_new = [dict(zip(orders, triple)) for triple in set(pred_group_entity) - set(target_group_entity)]
                ner_group_lack = [dict(zip(orders, triple)) for triple in set(target_group_entity) - set(pred_group_entity)]
                
                ner_group = True
                if len(ner_group_lack) or len(ner_group_new):
                    ner_group = False
                    error_type.append("group")
                
                if not ner_pred or not ner_group or not rel_correct:
                    result = {'text': texts[idx],
                            "error_type":error_type,
                            "gold_rel":tar_rel_name,
                            "pred_rel":pre_rel_name,
                            'gold_ner': [dict(zip(orders, triple)) for triple in gold_entities],
                            'pred_ner': [dict(zip(orders, triple)) for triple in pre_entities],
                            'ner_new': ner_new,
                            'ner_lack': ner_lack,
                            "group_new":ner_group_new,
                            "group_lack":ner_group_lack
                            }
                    error_result.append(result)
        writer.write(json.dumps(error_result,ensure_ascii=False) + '\n')
        writer.close()
        rel_p = rel_correct_num/rel_num
        self.log("int_acc",rel_p, prog_bar=True)
        ner_p = entity_correct_num / entity_predict_num if entity_predict_num > 0 else 0
        ner_r = entity_correct_num / entity_gold_num if entity_gold_num > 0 else 0
        ner_f1 = 2 * ner_p * ner_r / (ner_p + ner_r) if (ner_p + ner_r) > 0 else 0
        self.log("n_f1",ner_f1, prog_bar=True)
        self.log("n_acc",ner_p, prog_bar=True)
        self.log("n_rec",ner_r, prog_bar=True)
        
        group_p = group_correct_num / group_predict_num if group_predict_num > 0 else 0
        group_r = group_correct_num / group_gold_num if group_gold_num > 0 else 0
        group_f1 = 2 * group_p * group_r / (group_p + group_r) if (group_p + group_r) > 0 else 0
        self.log("g_f1",group_f1, prog_bar=True)
        self.log("g_acc",group_p, prog_bar=True)
        self.log("g_rec",group_r, prog_bar=True)

        self.log("g_g_num",group_gold_num, prog_bar=True)
        self.log("g_p_num",group_predict_num, prog_bar=True)
        self.log("g_c_num",group_correct_num, prog_bar=True)

        self.log("n_g_num",entity_gold_num, prog_bar=True)
        self.log("n_p_num",entity_predict_num, prog_bar=True)
        self.log("n_c_num",entity_correct_num, prog_bar=True)
        
        
        self.epoch += 1

    def span2str(self,triples, tokens):
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
            output.append((typename,sub))
        return output


    def configure_optimizers(self):
        """[配置优化参数]
        """
        param_optimizer = list(self.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay) and 'bert' in n], 'weight_decay': 0.8,'lr':2e-5},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay) and 'bert' in n], 'weight_decay': 0.0,'lr':2e-5},
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay) and 'bert' not in n], 'weight_decay': 0.8,'lr':2e-4},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay) and 'bert' not in n], 'weight_decay': 0.0,'lr':2e-4}
                ]
    
        # optimizer = torch.optim.AdamW(self.parameters(), lr=1e-5)
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-5)
        # StepLR = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
        milestones = list(range(2, 50, 2))
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.85)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", verbose = True, patience = 6)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.decay_steps, gamma=self.args.decay_rate)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, self.num_step * self.args.rewarm_epoch_num, self.args.T_mult)
        # StepLR = WarmupLR(optimizer,25000)
        optim_dict = {'optimizer': optimizer, 'lr_scheduler': scheduler}
        return optim_dict




    