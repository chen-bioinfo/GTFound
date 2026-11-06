import torch
from torch.nn.init import normal_
from .BertEmbedding import BertEmbeddings
from .MyTransformer1 import MyMultiheadAttention
from .OtherAttention import FLASHAttention
import torch.nn as nn
import os
import logging
import time
from copy import deepcopy

import pynvml

# from longformer.longformer import LongformerSelfAttention,LongformerConfig

 

def get_activation(activation_string):
    act = activation_string.lower()
    if act == "linear":
        return None
    elif act == "relu":
        return nn.ReLU()
    elif act == "gelu":
        return nn.GELU()
    elif act == "tanh":
        return nn.Tanh()
    else:
        raise ValueError("Unsupported activation: %s" % act)


class BertSelfAttention(nn.Module):


    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.attention == 'FLASH':
            MultiHeadAttention = FLASHAttention
            self.multi_head_attention = MultiHeadAttention(embed_dim=config.hidden_size,
                                                           num_heads=config.num_attention_heads,
                                                           src_len=config.max_position_embeddings,
                                                           dropout=config.attention_probs_dropout_prob,
                                                           device=config.device)
        else:
            if 'use_torch_multi_head' in config.__dict__ and config.use_torch_multi_head:
                MultiHeadAttention = nn.MultiheadAttention
            else:
                MultiHeadAttention = MyMultiheadAttention
            # longformer_config = LongformerConfig(
            #     attention_window=64,
            #     attention_dilation=1,
            #     hidden_size=config.hidden_size,
            #     num_attention_heads=config.num_attention_heads,
            #     max_position_embeddings=config.max_position_embeddings,
            #     attention_probs_dropout_prob=config.attention_probs_dropout_prob)
            # self.longformer_self_attention = LongformerSelfAttention(longformer_config)
            self.multi_head_attention = MultiHeadAttention(embed_dim=config.hidden_size,
                                                           num_heads=config.num_attention_heads,
                                                           dropout=config.attention_probs_dropout_prob)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None, ld_matrices=None):

        #!!!!!!!!!
        res = self.multi_head_attention(query, key, value, attn_mask=attn_mask, key_padding_mask=key_padding_mask, ld_matrices=ld_matrices)

        # return self.multi_head_attention(query, key, value, attn_mask=attn_mask, key_padding_mask=key_padding_mask, ld_matrices=ld_matrices)
        return res

class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        """
        :param hidden_states: [src_len, batch_size, hidden_size]
        :param input_tensor: [src_len, batch_size, hidden_size]
        :return: [src_len, batch_size, hidden_size]
        """
        #modify
        # input_tensor = input_tensor.transpose(0,1)
        # hidden_states = self.dense(hidden_states)  # [src_len, batch_size, hidden_size]
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self,
                hidden_states,
                attention_mask=None,
                ld_matrices=None):
        """

        :param hidden_states: [src_len, batch_size, hidden_size]
        :param attention_mask: [batch_size, src_len]
        :return: [src_len, batch_size, hidden_size]
        """
        self_outputs, attn_weight = self.self(hidden_states,
                                 hidden_states,
                                 hidden_states,
                                 attn_mask=None,
                                 key_padding_mask=attention_mask,
                                 ld_matrices=ld_matrices)
        # self_outputs[0] shape: [src_len, batch_size, hidden_size]
        attention_output = self.output(self_outputs[0], hidden_states)
        return attention_output, attn_weight


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = get_activation(config.hidden_act)
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        """

        :param hidden_states: [src_len, batch_size, hidden_size]
        :return: [src_len, batch_size, intermediate_size]
        """
        hidden_states = self.dense(hidden_states)  # [src_len, batch_size, intermediate_size]
        if self.intermediate_act_fn is None:
            hidden_states = hidden_states
        else:
            hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        """

        :param hidden_states: [src_len, batch_size, intermediate_size]
        :param input_tensor: [src_len, batch_size, hidden_size]
        :return: [src_len, batch_size, hidden_size]
        """
        hidden_states = self.dense(hidden_states)  # [src_len, batch_size, hidden_size]
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bert_attention = BertAttention(config)
        self.bert_intermediate = BertIntermediate(config)
        self.bert_output = BertOutput(config)

    def forward(self,
                hidden_states,
                attention_mask=None,
                ld_matrices=None):
        """

        :param hidden_states: [src_len, batch_size, hidden_size]
        :param attention_mask: [batch_size, src_len] mask掉padding部分的内容
        :return: [src_len, batch_size, hidden_size]
        """
        attention_output, attn_weight = self.bert_attention(hidden_states, attention_mask,ld_matrices=ld_matrices)
        # [src_len, batch_size, hidden_size]
        intermediate_output = self.bert_intermediate(attention_output)
        # [src_len, batch_size, intermediate_size]
        layer_output = self.bert_output(intermediate_output, attention_output)
        # [src_len, batch_size, hidden_size]
        return layer_output, attn_weight


class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.bert_layers = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            ld_matrices=None):
        """

        :param hidden_states: [src_len, batch_size, hidden_size]
        :param attention_mask: [batch_size, src_len]
        :return:
        """
        # all_encoder_layers = []
        layer_output = hidden_states
        attention_mask = attention_mask.transpose(0,1)
        for i, layer_module in enumerate(self.bert_layers):
            layer_output, attn_weight = layer_module(layer_output,
                                        attention_mask,ld_matrices)
            #  [src_len, batch_size, hidden_size]
            # all_encoder_layers.append(layer_output)
        # return all_encoder_layers, attn_weight
        return layer_output, attn_weight #修改


class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        self.config = config

    def forward(self, hidden_states):
        """

        :param hidden_states:  [src_len, batch_size, hidden_size]
        :return: [batch_size, hidden_size]
        """
        if self.config.pooler_type == "first_token_transform":
            token_tensor = hidden_states[0, :].reshape(-1, self.config.hidden_size)
        elif self.config.pooler_type == "all_token_average":
            token_tensor = torch.mean(hidden_states, dim=0)
        pooled_output = self.dense(token_tensor)  # [batch_size, hidden_size]
        pooled_output = self.activation(pooled_output)
        return pooled_output  # [batch_size, hidden_size]


def format_paras_for_torch(loaded_paras_names, loaded_paras):

    qkv_weight_names = ['query.weight', 'key.weight', 'value.weight']
    qkv_bias_names = ['query.bias', 'key.bias', 'value.bias']
    qkv_weight, qkv_bias = [], []
    torch_paras = []
    for i in range(len(loaded_paras_names)):
        para_name_in_pretrained = loaded_paras_names[i]
        para_name = ".".join(para_name_in_pretrained.split('.')[-2:])
        if para_name in qkv_weight_names:
            qkv_weight.append(loaded_paras[para_name_in_pretrained])
        elif para_name in qkv_bias_names:
            qkv_bias.append(loaded_paras[para_name_in_pretrained])
        else:
            torch_paras.append(loaded_paras[para_name_in_pretrained])
        if len(qkv_weight) == 3:
            torch_paras.append(torch.cat(qkv_weight, dim=0))
            qkv_weight = []
        if len(qkv_bias) == 3:
            torch_paras.append(torch.cat(qkv_bias, dim=0))
            qkv_bias = []
    return torch_paras


def replace_512_position(init_embedding, loaded_embedding, config):

    init_embedding[:512, :] = loaded_embedding[:512, :]
    return init_embedding


class BertModel(nn.Module):
    """

    """

    def __init__(self, config):
        super().__init__()
        self.bert_embeddings = BertEmbeddings(config)
        self.bert_encoder = BertEncoder(config)
        self.bert_pooler = BertPooler(config)
        self.config = config
        self._reset_parameters()

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                ld_matrices=None):

        
        embedding_output = self.bert_embeddings(input_ids=input_ids,
                                                position_ids=position_ids,
                                                token_type_ids=token_type_ids)
        # embedding_output: [src_len, batch_size, hidden_size]
        #modify
        embedding_output = embedding_output.transpose(0,1)
        #ld_matrices
        all_encoder_outputs, attn_weight = self.bert_encoder(embedding_output,
                                                attention_mask=attention_mask,ld_matrices=ld_matrices)

        # all_encoder_outputs 为一个包含有num_hidden_layers个层的输出
        # sequence_output = all_encoder_outputs[-1]  # 取最后一层
        # sequence_output: [src_len, batch_size, hidden_size]
        # pooled_output = self.bert_pooler(sequence_output)
        pooled_output = self.bert_pooler(all_encoder_outputs)
        # 默认是最后一层的first token 即[cls]位置经dense + tanh 后的结果
        # pooled_output: [batch_size, hidden_size]
        return pooled_output, all_encoder_outputs, attn_weight

    def _reset_parameters(self):

        for p in self.parameters():
            if p.dim() > 1:
                normal_(p, mean=0.0, std=self.config.initializer_range)

    @classmethod
    def from_pretrained(cls, config, pretrained_model_dir=None):
        model = cls(config)  # 初始化模型，cls为未实例化的对象，即一个未实例化的BertModel对象
        pretrained_model_path = os.path.join(pretrained_model_dir, "pytorch_model.bin")
        if not os.path.exists(pretrained_model_path):
            raise ValueError(f"<路径：{pretrained_model_path} 中的模型不存在，请仔细检查！>")
        loaded_paras = torch.load(pretrained_model_path)
        state_dict = deepcopy(model.state_dict())
        loaded_paras_names = list(loaded_paras.keys())[:-8]
        model_paras_names = list(state_dict.keys())[1:]
        if 'use_torch_multi_head' in config.__dict__ and config.use_torch_multi_head:
            torch_paras = format_paras_for_torch(loaded_paras_names, loaded_paras)
            for i in range(len(model_paras_names)):
                logging.debug(f"## 成功赋值参数:{model_paras_names[i]},形状为: {torch_paras[i].size()}")
                if "position_embeddings" in model_paras_names[i]:
                    # 这部分代码用来消除预训练模型只能输入小于512个字符的限制
                    if config.max_position_embeddings > 512:
                        new_embedding = replace_512_position(state_dict[model_paras_names[i]],
                                                             loaded_paras[loaded_paras_names[i]],
                                                             config)
                        state_dict[model_paras_names[i]] = new_embedding
                        continue
                state_dict[model_paras_names[i]] = torch_paras[i]
            logging.info(f"## 注意，正在使用torch框架中的MultiHeadAttention实现")
        else:
            for i in range(len(loaded_paras_names)):
                logging.debug(f"## 成功将参数:{loaded_paras_names[i]}赋值给{model_paras_names[i]},"
                              f"参数形状为:{state_dict[model_paras_names[i]].size()}")
                if "position_embeddings" in model_paras_names[i]:
                    # 这部分代码用来消除预训练模型只能输入小于512个字符的限制
                    if config.max_position_embeddings > 512:
                        new_embedding = replace_512_position(state_dict[model_paras_names[i]],
                                                             loaded_paras[loaded_paras_names[i]],
                                                             config)
                        state_dict[model_paras_names[i]] = new_embedding
                        continue
                state_dict[model_paras_names[i]] = loaded_paras[loaded_paras_names[i]]
            logging.info(f"## 注意，正在使用本地MyTransformer中的MyMultiHeadAttention实现")
        model.load_state_dict(state_dict)
        return model
 