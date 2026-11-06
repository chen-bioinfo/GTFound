import torch.nn as nn
import torch
from torch.nn.init import normal_
import pandas as pd
import math, time


class PositionalEmbedding(nn.Module):


    def __init__(self, hidden_size, batch_size, max_position_embeddings=1024, initializer_range=0.02, all_possible_position=None, device=None):
        super(PositionalEmbedding, self).__init__()
        assert max_position_embeddings >= 512, "config.max_position_embeddings参数必须大于等于512"
        # 因为BERT预训练模型的长度为512
        self.device = device
        self._reset_parameters(initializer_range)
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.max_position_embeddings = max_position_embeddings
        
        # self.all_possible_position = all_possible_position.reshape(-1,1)
        # self.all_possible_position_len = self.all_possible_position.shape[0]
        # # print(self.all_possible_position_len)
        # self.pe = torch.zeros(self.all_possible_position_len, hidden_size)
        # self.div_term = torch.exp(torch.arange(0, hidden_size, 2) * -(math.log(10000.0) / hidden_size))
        # # self.div_term = self.div_term.to(self.device)
        
        # # df = pd.read_csv(self.all_possible_position,sep='\t')
        # # dfpos = df['POS']
        # # data = dfpos[:max_position_embeddings]
        # # tensorData = torch.from_numpy(data.to_numpy())
        # # self.position = tensorData.reshape(-1,1)
        
        # # 将前面定义的变化矩阵 进行技术，偶数分别赋值
        # self.pe[:, 0::2] = torch.sin(self.all_possible_position * self.div_term)  # 用正弦波给偶数部分赋值
        # self.pe[:, 1::2] = torch.cos(self.all_possible_position * self.div_term)  # 用余弦波给奇数部分赋值
        # self.pe = self.pe.to(self.device)
        self.pre_dense = torch.empty((self.batch_size, self.max_position_embeddings, self.hidden_size))
        # self.pre_dense = self.pre_dense.to(self.device)
        # # 从正余弦编码学习新的编码
        # print("pre_device",self.pre_dense.device)
        # print("pe_device",self.pe.device)
        self.dense = nn.Linear(hidden_size, hidden_size, bias=True)
        self.simple_pe = nn.Linear(1, hidden_size, bias=True)
        self.activation = nn.Sigmoid()

        self.tensor_to_gpu_flag = False

    def forward(self, position_ids):
        """
        :param position_ids: [position_ids_len, batch_size]
        :return: [position_ids_len, batch_size, hidden_size]
        """
        # print(f"in BERT_PE, position_emb.shape: {position_ids.shape}")
        # print("deviceaaaaa",position_ids.device)
        # if not self.tensor_to_gpu_flag:
        #     self.device = position_ids.device
        #     self.all_possible_position = self.all_possible_position.reshape(-1,1).to(self.device)
        #     self.div_term = self.div_term.to(self.device)
        #     self.pe = self.pe.to(self.device)
        #     self.pre_dense = self.pre_dense.to(self.device)
        #     self.tensor_to_gpu_flag = True
        # print(f"in_forward: {position_ids.shape}")
        # print(self.all_possible_position.shape)
        # return self.activation(self.simple_pe((position_ids.T).to(torch.float32))).unsqueeze(1)
        # position_ids = position_ids.transpose(0, 1)
        # print(f"position_ids in pos_emb: {position_ids.shape}")


        # pos_emb_start = time.perf_counter()
        # for i in range(position_ids.shape[1]): # i for batch_size
        #     for j in range(self.all_possible_position_len - self.max_position_embeddings): # j for start_pos
        #         if position_ids[0][i] == self.all_possible_position[j][0]:
        #             self.pre_dense[i] = self.pe[j:j+self.max_position_embeddings]
        # for i in range(position_ids.shape[1]): # i for batch_size
        #     j = position_ids[0][i]
        #     self.pre_dense[i] = self.pe[j:j+self.max_position_embeddings]
        
        # self.pre_dense = torch.split(self.pre_dense,32,dim=0)#spilt batchsize32 to 2*16 
        # print("for ward.device",self.pre_dense[0].device)
        # print("abc device",self.pe.device)
        # print(f"pos forward for a batch: {(end_time - start_time):.3f}s")
        # return self.activation(self.dense(self.pre_dense.transpose(0, 1)))
        result = self.activation(self.dense(position_ids)).transpose(0, 1)
        # print(f"after BERT_PE, position_emb.shape:{result.shape}")
        # pos_emb_end = time.perf_counter()
        # print(f"pos_emb_time: {pos_emb_end - pos_emb_start}s")
        return result


    def _reset_parameters(self, initializer_range):
        r"""Initiate parameters."""

        for p in self.parameters():
            if p.dim() > 1:
                normal_(p, mean=0.0, std=initializer_range)


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, hidden_size, pad_token_id=0, initializer_range=0.02):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)
        self._reset_parameters(initializer_range)

    def forward(self, input_ids):
        """
        :param input_ids: shape : [input_ids_len, batch_size]
        :return: shape: [input_ids_len, batch_size, hidden_size]
        """
        return self.embedding(input_ids)

    def _reset_parameters(self, initializer_range):

        for p in self.parameters():
            if p.dim() > 1:
                normal_(p, mean=0.0, std=initializer_range)


class SegmentEmbedding(nn.Module):
    def __init__(self, type_vocab_size, hidden_size, initializer_range=0.02):
        super(SegmentEmbedding, self).__init__()
        self.embedding = nn.Embedding(type_vocab_size, hidden_size)
        self._reset_parameters(initializer_range)

    def forward(self, token_type_ids):
        """

        :param token_type_ids:  shape: [token_type_ids_len, batch_size]
        :return: shape: [token_type_ids_len, batch_size, hidden_size]
        """
        return self.embedding(token_type_ids)

    def _reset_parameters(self, initializer_range):

        for p in self.parameters():
            if p.dim() > 1:
                normal_(p, mean=0.0, std=initializer_range)


class BertEmbeddings(nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : normal embedding matrix
        2. SegmentEmbedding : adding sentence segment info, (sent_A:1, sent_B:2)
        sum of all these features are output of BERTEmbedding
    """

    def __init__(self, config):
        super().__init__()
        self.device = config.device
        self.word_embeddings = TokenEmbedding(vocab_size=config.vocab_size,
                                              hidden_size=config.hidden_size,
                                              pad_token_id=config.pad_token_id,
                                              initializer_range=config.initializer_range)
        # return shape [src_len,batch_size,hidden_size]

        self.position_embeddings = PositionalEmbedding(max_position_embeddings=config.max_position_embeddings,
                                                       hidden_size=config.hidden_size,
                                                       batch_size=config.batch_size,
                                                       initializer_range=config.initializer_range,
                                                       all_possible_position=config.all_possible_position,
                                                       device=config.device)
        # return shape [src_len,1,hidden_size]

        self.token_type_embeddings = SegmentEmbedding(type_vocab_size=config.type_vocab_size,
                                                      hidden_size=config.hidden_size,
                                                      initializer_range=config.initializer_range)
        # return shape  [src_len,batch_size,hidden_size]

        self.LayerNorm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.register_buffer("position_ids",
                             torch.arange(config.max_position_embeddings).expand((1, -1)))
        # shape: [1, max_position_embeddings]

    def forward(self,
                input_ids=None,
                position_ids=None,
                token_type_ids=None):

        src_len = input_ids.size(0)
        token_embedding = self.word_embeddings(input_ids).transpose(0,1)
        # shape:[src_len,batch_size,hidden_size]
        # print(f"token_embedding devices: {token_embedding.device}")

        if position_ids is None:  # 在实际建模时这个参数其实可以不用传值
            position_ids = self.position_ids[:, :src_len]  # [1,src_len]
        positional_embedding = self.position_embeddings(position_ids)



        embeddings = token_embedding + positional_embedding

        # [src_len,batch_size,hidden_size] + [src_len,1,hidden_size] + [src_len,batch_size,hidden_size]

        embeddings = self.LayerNorm(embeddings)  # [src_len, batch_size, hidden_size]

        embeddings = embeddings.transpose(0,1)
        embeddings = self.dropout(embeddings)
        return embeddings
