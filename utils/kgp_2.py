import math
import os
import random
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import time
import logging
import threading
from io import StringIO
max_length = 1024
# random.seed(42)
def vcfdf_to_hapsnp(df: pd.DataFrame, ridx: pd.Index=None, is_impute: bool=False, temp="./temp/vcfdf_to_hapsnp_temp") -> tuple:
    """
    将读入vcf的dataframe中的基因型的列数据转化为单倍型的列数据
    """
    ids = df.columns
    # print(ids)
    if is_impute:
        df = df.reindex(columns=ridx)
        df.to_csv(temp, sep='\t', index=False)
        df = pd.read_csv(temp, sep='\t')
        ids = df.columns
    # 从第9列开始是位点信息
    
    for id in ids[9:]: 
        df[str(id) + "_0"] = df[id].apply(lambda x: int(x[0]))
        df[str(id) + "_1"] = df[id].apply(lambda x: int(x[-1]))
    
    # print(df)
    # 删除原有信息
    df.drop(columns=ids[5:], inplace=True)
    df.drop(columns=['#CHROM', 'ID'], inplace=True)
    df['POS'] = pd.to_numeric(df['POS'])
    nda = df.to_numpy().T
    # print(df['POS'].dtype)
    # print(nda)
    return nda

def vcf_process(vcf_file):
        # 手动过滤掉以 ## 开头的行
    with open(vcf_file, 'r') as file:
        lines = [line for line in file if not line.startswith('##')]

    # 将过滤后的内容传递给 pandas 进行读取
    try:
        val_df = pd.read_csv(StringIO(''.join(lines)), sep='\t', low_memory=False)
        print(val_df.head())  # 查看前几行数据
    except pd.errors.ParserError as e:
        print(f"ParserError: {e}")
    val_nda = vcfdf_to_hapsnp(val_df)
    return val_nda

class Vocab:
    """
    根据本地的vocab文件,构造一个词表
    vocab = Vocab()
    print(vocab.itos)  # 得到一个列表，返回词表中的每一个词；
    print(vocab.itos[2])  # 通过索引返回得到词表中对应的词；
    print(vocab.stoi)  # 得到一个字典，返回词表中每个词的索引；
    print(vocab.stoi['我'])  # 通过单词返回得到词表中对应的索引
    print(len(vocab))  # 返回词表长度
    """
    UNK = '[UNK]'

    def __init__(self, vocab_path):
        self.stoi = {}
        self.itos = []
        with open(vocab_path, 'r', encoding='utf-8') as f:
            for i, word in enumerate(f):
                w = word.strip('\n')
                self.stoi[w] = i
                self.itos.append(w)

    def __getitem__(self, token):
        return self.stoi.get(token, self.stoi.get(Vocab.UNK))

    def __len__(self):
        return len(self.itos)


def pad_sequence(sequences, batch_first=False, max_len=None, padding_value=0):
    """
    对一个List中的元素进行padding
    Pad a list of variable length Tensors with ``padding_value``
    a = torch.ones(25)
    b = torch.ones(22)
    c = torch.ones(15)
    pad_sequence([a, b, c],max_len=None).size()
    torch.Size([25, 3])
        sequences:
        batch_first: 是否把batch_size放到第一个维度
        padding_value:
        max_len :
                当max_len = 50时，表示以某个固定长度对样本进行padding，多余的截掉；
                当max_len=None是，表示以当前batch中最长样本的长度对其它进行padding；
    Returns:
    """
    if max_len is None:
        max_len = max([s.size(0) for s in sequences])
    out_tensors = []
    for tensor in sequences:
        if tensor.size(0) < max_len:
            tensor = torch.cat([tensor, torch.tensor([padding_value] * (max_len - tensor.size(0)))], dim=0)
        else:
            tensor = tensor[:max_len]
        out_tensors.append(tensor)
    out_tensors = torch.stack(out_tensors, dim=1)
    if batch_first:
        return out_tensors.transpose(0, 1)
    return out_tensors

# def generate_batch(data_batch, PAD_IDX=2, max_len=512):
#     # print(data_batch[0][0])
    
#     b_token_ids, b_mlm_label, b_positions = [], [], []
#     for (token_ids, mlm_label, positions) in data_batch:
#         # 开始对一个batch中的每一个样本进行处理
#         b_token_ids.append(token_ids)
#         b_mlm_label.append(mlm_label)
#         b_positions.append(positions)
   
#     b_token_ids = torch.stack(b_token_ids,dim=1).transpose(0,1)

#     b_mask = (b_token_ids == PAD_IDX)
#     print(b_mask)
   
#     # print(f"b_positions.type: {type(b_positions)}")
#     b_positions = torch.stack(b_positions)
#     # print(f"b_positions.shape: {b_positions.shape}")
#     #b_positions:  [batch_size, max_len, hidden_size]
#     # logging.info(f"batch::::::")
#     b_mlm_label = torch.stack(b_mlm_label)
#     return b_token_ids, b_mask, b_mlm_label, b_positions
import torch

def pad_sequences_3d(sequences, padding_value=0, max_len = max_length,batch_first=False):
    out_tensors = []
    for tensor in sequences:
        # print(tensor.shape)
        if tensor.size(0) < max_len:
            pad_dim1 = torch.tensor([padding_value] * (max_len - tensor.size(0)), dtype=tensor.dtype, device=tensor.device)
            tensor = torch.cat([tensor, pad_dim1.unsqueeze(1).expand(-1, max_len)], dim=0)


        if tensor.size(1) < max_len:
            pad_dim2 = torch.tensor([padding_value] * (max_len - tensor.size(1)), dtype=tensor.dtype, device=tensor.device)
            tensor = torch.cat([tensor, pad_dim2.unsqueeze(0).expand(max_len, -1)], dim=1)


        out_tensors.append(tensor)

    out_tensors = torch.stack(out_tensors, dim=0)

    if batch_first:
        out_tensors = out_tensors.permute(0, 2, 1)

    return out_tensors

# Example usage:
# sequences_3d = your_three_dimensional_array
# padded_tensor = pad_sequences_3d(sequences_3d, padding_value=0, batch_first=True)


def generate_batch(data_batch, PAD_IDX=3, max_len=max_length):
    b_token_ids, b_mlm_label, b_positions,b_LD_matrices = [], [], [], []
    b_MAF_g= []
    chunk_id_list = []

    for (token_ids, mlm_label, positions,LD_matrix,MAF_g,chunk_id) in data_batch:
        # 开始对一个batch中的每一个样本进行处理
        b_token_ids.append(token_ids)
        b_mlm_label.append(mlm_label)
        b_positions.append(positions)
        b_LD_matrices.append(LD_matrix)
        b_MAF_g.append(MAF_g)
        chunk_id_list.append(chunk_id)


    # print("aaabbbccc")
    b_token_ids = pad_sequence(b_token_ids,  # [batch_size,max_len]
                               padding_value=PAD_IDX,
                               batch_first=False,
                               max_len=max_len)
    b_token_ids = b_token_ids.transpose(0,1)                        
    #b_token_ids:  [batch_size,src_len]
    
    
    b_mlm_label = pad_sequence(b_mlm_label,  # [batch_size,max_len]
                               padding_value=PAD_IDX,
                               batch_first=False,
                               max_len=max_len)
   # b_mlm_label:  [src_len, batch_size]
   
    b_mask = (b_token_ids == PAD_IDX).transpose(0, 1)
    
   # b_mask: [max_len, batch_size]
    b_mlm_label = b_mlm_label.transpose(0, 1)
    b_mask = b_mask.transpose(0, 1) 
    # print(f"b_positions.type: {type(b_positions)}")
    b_positions = torch.stack(b_positions)

    b_LD_matrices = pad_sequences_3d(b_LD_matrices,
                                     padding_value=PAD_IDX,
                                     batch_first=False,
                                     )
    # print(b_LD_matrices[0].shape)
    chunk_id_t = torch.tensor(chunk_id_list)
    return b_token_ids, b_mask, b_mlm_label, b_positions, b_LD_matrices,b_MAF_g,chunk_id_t

def generate_batch_pre(data_batch, PAD_IDX=3, max_len=max_length):
    b_token_ids, b_mlm_label, b_positions,b_LD_matrices = [], [], [], []
    b_MAF_g= []
    chunk_id_list = []

    for (token_ids, mlm_label, positions,LD_matrix,MAF_g,chunk_id) in data_batch:
        # 开始对一个batch中的每一个样本进行处理
        b_token_ids.append(token_ids)
        b_mlm_label.append(mlm_label)
        b_positions.append(positions)
        b_LD_matrices.append(LD_matrix)
        b_MAF_g.append(MAF_g)
        chunk_id_list.append(chunk_id)


    # print("aaabbbccc")
    b_token_ids = pad_sequence(b_token_ids,  # [batch_size,max_len]
                               padding_value=PAD_IDX,
                               batch_first=False,
                               max_len=max_len+3)
    b_token_ids = b_token_ids.transpose(0,1)                        
    #b_token_ids:  [batch_size,src_len]
    
    
    b_mlm_label = pad_sequence(b_mlm_label,  # [batch_size,max_len]
                               padding_value=PAD_IDX,
                               batch_first=False,
                               max_len=max_len+3)
   # b_mlm_label:  [src_len, batch_size]
   
    b_mask = (b_token_ids == PAD_IDX).transpose(0, 1)
    
   # b_mask: [max_len, batch_size]
    b_mlm_label = b_mlm_label.transpose(0, 1)
    b_mask = b_mask.transpose(0, 1) 
    # print(f"b_positions.type: {type(b_positions)}")
    b_positions = torch.stack(b_positions)

    b_LD_matrices = pad_sequences_3d(b_LD_matrices,
                                     padding_value=PAD_IDX,
                                     batch_first=False,
                                     )
    # print(b_LD_matrices[0].shape)
    chunk_id_t = torch.tensor(chunk_id_list)
    return b_token_ids, b_mask, b_mlm_label, b_positions, b_LD_matrices,b_MAF_g,chunk_id_t

def add_zero_padding(tensor):
    """
    在二维张量的第一行和第一列添加全为0的维度
    
    参数:
        tensor (torch.Tensor): 输入的二维张量
        
    返回:
        torch.Tensor: 扩展后的张量
    """
    # 获取原始张量的尺寸
    rows, cols = tensor.size()
    
    # 在第一列添加全0列
    zeros_col = torch.zeros((rows, 1), dtype=tensor.dtype, device=tensor.device)
    tensor_with_col = torch.cat([zeros_col, tensor], dim=1)
    
    # 在第一行添加全0行
    zeros_row = torch.zeros((1, cols + 1), dtype=tensor.dtype, device=tensor.device)
    result = torch.cat([zeros_row, tensor_with_col], dim=0)
    
    return result
    
class randomDataset_torch(Dataset):
    def __init__(self, 
                 nda:torch.tensor, 
                 LD_data:torch.tensor,
                 hidden_size:int,
                 max_position_embeddings:int, 
                 masked_rate=0.4, 
                 vocab:Vocab=None, 
                 pad_index=0,
                 masked_token_rate=0.8,
                 masked_token_unchanged_rate=0.5,
                 step = 1,
                 length_edge = 0,
                 ) -> None:
        super().__init__()
        self.positions = nda[0]

        # self.positions = self.positions.reshape(-1,1024)
        # first_pos = self.positions[:,0].unsqueeze(1)
        # self.positions = self.positions - first_pos

        # self.positions = self.positions - self.positions[0]
        # self.positions = (self.positions - self.positions[0]) / (self.positions[-1] - self.positions[0]) * 10000
        self.positions_tensor= self.positions.reshape(-1,1)
        # self.positions_tensor = torch.arange(1024).reshape(-1,1)
        # print(self.positions_tensor)
        self.data = nda[1:]

        self.LD_data = LD_data.to(torch.float32)
        self.samples = self.data.shape[0]
        self.length = self.data.shape[1]
        self.max_position_embeddings = max_position_embeddings
        self.max_start = self.length - max_position_embeddings
        self.vocab = vocab
        # self.masked_rate = 0.2
        self.masked_rate = 0.4
        # self.masked_rate = 0.6
        # self.masked_rate = 0.8
        # self.masked_rate = random.uniform(0.85, 0.95)
        # self.masked_rate = random.uniform(0.35, 0.45)
        # self.masked_rate = 0.01
        self.masked_token_rate = masked_token_rate
        self.masked_token_unchanged_rate = masked_token_unchanged_rate
        self.PAD_IDX = pad_index
        self.SEP_IDX = self.vocab['[SEP]']
        self.CLS_IDX = self.vocab['[CLS]']
        self.MASK_IDS = self.vocab['[MASK]']
        self.step = step
        self.length_edge = length_edge
        self.start = 0
        self.arr = list(range(0,self.max_start+1,self.step))
        self.index_arr = list(range(0,self.samples,1))
        self.length_arr = np.arange(self.max_position_embeddings)
        self.hidden_size = hidden_size
        # random.shuffle(self.index_arr)

        print("len",self.max_start)
        print("nda data shape",self.data.shape)
        print("numbers of site",len(self.arr))
        print("samples numbers",self.samples)
        print("length of index_arr",len(self.index_arr))
        print("length_edge",self.length_edge)
        print("LD_data",LD_data)
        # 预先计算正余弦编码
        self.pe = torch.zeros(self.length, hidden_size)
        print(f"hiddensize:{hidden_size}")
        # self.CLS_PE = torch.zeros(1, hidden_size)
        self.div_term = torch.exp(torch.arange(0, hidden_size, 2) * -(math.log(10000.0) / hidden_size))
        # self.positions_tensor = self.positions_tensor / 10000
        self.pe[:, 0::2] = torch.sin(self.positions_tensor * self.div_term)  # 用正弦波给偶数部分赋值
        self.pe[:, 1::2] = torch.cos(self.positions_tensor * self.div_term)  # 用余弦波给奇数部分赋值


        # print(self.data[0][0], self.vocab[str(self.data[0][0])])
    
    def get_masked_sample(self, token_ids,maf):
        """
        本函数的作用是将传入的 一段token_ids的其中部分进行mask处理
        :param token_ids:         e.g. [101, 1031, 4895, 2243, 1033, 10029, 2000, 2624, 1031,....]
        :return: mlm_input_tokens_id:  [101, 1031, 103, 2243, 1033, 10029, 2000, 103,  1031, ...]
                           mlm_label:  [ 0,   0,   4895,  0,    0,    0,    0,   2624,  0,...]
        """
        # masked_rate = self.masked_rate
        # self.masked_rate = random.uniform(0.85, 0.95)
        candidate_pred_positions = []  # 候选预测位置的索引
        for i, ids in enumerate(token_ids):
            # 在遮蔽语言模型任务中不会预测特殊词元，所以如果该位置是特殊词元
            # 那么该位置就不会成为候选mask位置
            if ids in [self.CLS_IDX, self.SEP_IDX]:
                continue
            if i < self.length_edge or i > (self.max_position_embeddings - self.length_edge -1):
                continue
            candidate_pred_positions.append(i)
            # 保存候选位置的索引， 例如可能是 [ 2,3,4,5, ....]
        # random.shuffle(candidate_pred_positions)  # 将所有候选位置打乱，更利于后续随机
        # 被掩盖位置的数量，BERT模型中默认将15%的Token进行mask
        # num_mlm_preds = max(1, round(len(token_ids) * self.masked_rate))
        # num_mlm_preds = max(1, round(len(token_ids) * random.uniform(0.85, 0.95)))
        # maskrate = random.uniform(0.90, 0.95)
        maskrate = 0.4
        num_mlm_preds = max(1, round(len(token_ids) * self.masked_rate))

        # logging.debug(f" ## Mask数量为: {num_mlm_preds}")
        mlm_input_tokens_id, mlm_label = self.replace_masked_tokens(
            token_ids, candidate_pred_positions, num_mlm_preds,maf,maskrate)
        return mlm_input_tokens_id, mlm_label
        
    def replace_masked_tokens(self, token_ids, candidate_pred_positions, num_mlm_preds,maf,maskrate):
        """
        本函数的作用是根据给定的token_ids、候选mask位置以及需要mask的数量来返回被mask后的token_ids以及标签信息
        :param token_ids:
        :param candidate_pred_positions:
        :param num_mlm_preds:
        :return:
        """


        
        pred_positions = []

        mlm_input_tokens_id = [token_id for token_id in token_ids]
        for mlm_pred_position in candidate_pred_positions:
            if len(pred_positions) >= num_mlm_preds:
                break  # 如果已经mask的数量大于等于num_mlm_preds则停止mask
            masked_token_id = None
            # 80%的时间：将词替换为['MASK']词元，但这里是直接替换为['MASK']对应的id
            if random.random() < self.masked_rate:
                random_num = random.randint(0, 9)
                # maf_site = maf[i]
                if random_num < 1:
                    masked_token_id = self.MASK_IDS
                    mlm_input_tokens_id[mlm_pred_position] = masked_token_id
                    
                elif random_num < 9:
                    # 10%的时间：将词被替换成任意其他的标记
                    random_token_id = random.choice(range(2))  # 假设vocab_size是词汇表大小
                    mlm_input_tokens_id[mlm_pred_position] = random_token_id

                pred_positions.append(mlm_pred_position)  # 保留被mask位置的索引信息
            else:
                continue
        # 构造mlm任务中需要预测位置对应的正确标签，如果其没出现在pred_positions则表示该位置不是mask位置
        # 则在进行损失计算时需要忽略掉这些位置（即为PAD_IDX）；而如果其出现在mask的位置，则其标签为原始token_ids对应的id

        mlm_label = [self.PAD_IDX if idx not in pred_positions
                     else token_ids[idx] for idx in range(len(token_ids))]
        # print(f'mlm_label :{mlm_label}')
        return mlm_input_tokens_id, mlm_label
    
    def __getitem__(self, index):
        # start = time.time()
        # 转化为token_id
        # 取随机起始位点的索引
        # rand_start = random.randint(0, self.max_start)
        # rand_start = random.randint(0, self.max_start)
        # random.seed(43)
        gene_idx = index//len(self.arr)
        gene_site = self.arr[index%len(self.arr)]


        chunk_id  = index%len(self.arr)
    
        # token_ids = self.data[gene_idx][rand_start:rand_start + self.max_position_embeddings]

        token_ids = self.data[gene_idx][gene_site:gene_site + self.max_position_embeddings]#恢复恢复
        
        # token_ids.tolist()
        # np.save('new.npy',token_ids)
        token_ids = np.array(token_ids)
        token_ids = [self.vocab[str(token)] for token in token_ids]
        # np.save('new.npy',token_ids)
        # position_embedding = self.pe[rand_start: rand_start + self.max_position_embeddings]
        position_embedding = self.pe[gene_site: gene_site + self.max_position_embeddings]#恢复恢复
        # print(gene_idx,gene_site)
        # for key,value in self.vocab.stoi.items():
        #     token_ids[token_ids==key]=value
        # token_ids = [self.vocab[str(token)] for token in token_id]
        # self.data[gene_idx][gene_site: gene_site + self.max_position_embeddings]
     
        # token_ids = np.array()
        maf = self.MAF[gene_site:gene_site+self.max_position_embeddings]
        mlm_input_tokens_id, mlm_label = self.get_masked_sample(token_ids,maf) 
        # LD_matrix = torch.arange(gene_site,gene_site + self.max_position_embeddings)
        # print(type(LD_matrix))
        LD_matrix = self.LD_data[gene_site:gene_site + self.max_position_embeddings,:]

        # LD_matrix = add_zero_padding(LD_matrix)
        # mlm_input_tokens_id = mlm_input_tokens_id.tolist()
        # print(f'fuc:{ [6] + mlm_input_tokens_id}')
        # mlm_input_tokens_id = [6] + mlm_input_tokens_id
        token_ids = torch.tensor(mlm_input_tokens_id, dtype=torch.long)

        mlm_label = torch.tensor(mlm_label, dtype=torch.long)

        # LD_matrix = torch.tensor(LD_matrix,dtype=torch.float32)
        MAF_g = self.MAF[:self.max_position_embeddings] 
        # position_embedding = torch.cat([self.CLS_PE,position_embedding],dim=0)
        # print(f'cywcnm:{token_ids.shape}')

        # print(pos_res)
        return token_ids, mlm_label, position_embedding,LD_matrix,MAF_g,chunk_id
    def __len__(self):
        return len(self.arr)*self.samples
        # return self.samples
    


def generate_batch_1(data_batch, PAD_IDX=3, max_len=1024):
    b_token_ids, b_mlm_label, b_positions,b_LD_matrices = [], [], [], []
    for (token_ids, mlm_label, positions,LD_matrix) in data_batch:
        # 开始对一个batch中的每一个样本进行处理
        b_token_ids.append(token_ids)
        b_mlm_label.append(mlm_label)
        b_positions.append(positions)
        b_LD_matrices.append(LD_matrix)
    # print("aaabbbccc")
    b_token_ids = pad_sequence(b_token_ids,  # [batch_size,max_len]
                               padding_value=PAD_IDX,
                               batch_first=False,
                               max_len=max_len)
    b_token_ids = b_token_ids.transpose(0,1)                        
    #b_token_ids:  [batch_size,src_len]
    
    
    b_mlm_label = pad_sequence(b_mlm_label,  # [batch_size,max_len]
                               padding_value=PAD_IDX,
                               batch_first=False,
                               max_len=max_len)
   # b_mlm_label:  [src_len, batch_size]
   
    b_mask = (b_token_ids == PAD_IDX).transpose(0, 1)
    
   # b_mask: [max_len, batch_size]
    b_mlm_label = b_mlm_label.transpose(0, 1)
    b_mask = b_mask.transpose(0, 1) 
    # print(f"b_positions.type: {type(b_positions)}")
    b_positions = torch.stack(b_positions)
    b_LD_matrices = pad_sequences_3d(b_LD_matrices,
                                     padding_value=PAD_IDX,
                                     batch_first=False,

                                     )


    return b_token_ids, b_mask, b_mlm_label, b_positions, b_LD_matrices




class randomDataset(Dataset):
    def __init__(self, 
                 nda:np.ndarray, 
                 LD_data:np.ndarray,
                 hidden_size:int,
                 MAF:np.ndarray,
                 max_position_embeddings:int, 
                 masked_rate=0.4, 
                 vocab:Vocab=None, 
                 pad_index=0,
                 masked_token_rate=0.8,
                 masked_token_unchanged_rate=0.5,
                 step = 1,
                 length_edge = 0
                 ) -> None:
        super().__init__()
        self.positions = nda[0]
        self.positions_tensor = torch.from_numpy(self.positions.astype(np.int32)).reshape(-1,1)
        self.ref = nda[1]
        self.alt = nda[2]
        self.MAF = MAF

        self.data = nda[3:]
        self.LD_data = LD_data
        self.samples = self.data.shape[0]
        self.length = self.data.shape[1]
        self.max_position_embeddings = max_position_embeddings
        self.max_start = self.length - max_position_embeddings
        self.vocab = vocab
        self.masked_rate = masked_rate
        self.masked_token_rate = masked_token_rate
        self.masked_token_unchanged_rate = masked_token_unchanged_rate
        self.PAD_IDX = pad_index
        self.SEP_IDX = self.vocab['[SEP]']
        self.CLS_IDX = self.vocab['[CLS]']
        self.MASK_IDS = self.vocab['[MASK]']
        self.step = step
        self.length_edge = length_edge
        self.start = 0
        self.arr = list(range(0,self.max_start+1,self.step))
        self.index_arr = list(range(0,self.samples,1))
        self.length_arr = np.arange(self.max_position_embeddings)
        # random.shuffle(self.index_arr)

        print("len",self.max_start)
        print("nda data shape",self.data.shape)
        print("numbers of site",len(self.arr))
        print("samples numbers",self.samples)
        print("length of index_arr",len(self.index_arr))
        print("length_edge",self.length_edge)
        print("LD_data",LD_data)
        # 预先计算正余弦编码
        self.pe = torch.zeros(self.length, hidden_size)
        self.div_term = torch.exp(torch.arange(0, hidden_size, 2) * -(math.log(10000.0) / hidden_size))
        self.pe[:, 0::2] = torch.sin(self.positions_tensor * self.div_term)  # 用正弦波给偶数部分赋值
        self.pe[:, 1::2] = torch.cos(self.positions_tensor * self.div_term)  # 用余弦波给奇数部分赋值

        # print(self.data[0][0], self.vocab[str(self.data[0][0])])
    
    def get_masked_sample(self, token_ids):
        """
        本函数的作用是将传入的 一段token_ids的其中部分进行mask处理
        :param token_ids:         e.g. [101, 1031, 4895, 2243, 1033, 10029, 2000, 2624, 1031,....]
        :return: mlm_input_tokens_id:  [101, 1031, 103, 2243, 1033, 10029, 2000, 103,  1031, ...]
                           mlm_label:  [ 0,   0,   4895,  0,    0,    0,    0,   2624,  0,...]
        """
        masked_rate = self.masked_rate
        candidate_pred_positions = []  # 候选预测位置的索引
        for i, ids in enumerate(token_ids):
            # 在遮蔽语言模型任务中不会预测特殊词元，所以如果该位置是特殊词元
            # 那么该位置就不会成为候选mask位置
            if ids in [self.CLS_IDX, self.SEP_IDX]:
                continue
            if i < self.length_edge or i > (self.max_position_embeddings - self.length_edge -1):
                continue
            candidate_pred_positions.append(i)
            # 保存候选位置的索引， 例如可能是 [ 2,3,4,5, ....]
        # random.shuffle(candidate_pred_positions)  # 将所有候选位置打乱，更利于后续随机
        # 被掩盖位置的数量，BERT模型中默认将15%的Token进行mask
        num_mlm_preds = max(1, round(len(token_ids) * masked_rate))
        # logging.debug(f" ## Mask数量为: {num_mlm_preds}")
        mlm_input_tokens_id, mlm_label = self.replace_masked_tokens(
            token_ids, candidate_pred_positions, num_mlm_preds)
        return mlm_input_tokens_id, mlm_label
        
    def replace_masked_tokens(self, token_ids, candidate_pred_positions, num_mlm_preds):
        """
        本函数的作用是根据给定的token_ids、候选mask位置以及需要mask的数量来返回被mask后的token_ids以及标签信息
        :param token_ids:
        :param candidate_pred_positions:
        :param num_mlm_preds:
        :return:
        """


        
        pred_positions = []
        mlm_input_tokens_id = [token_id for token_id in token_ids]
        for mlm_pred_position in candidate_pred_positions:
            if len(pred_positions) >= num_mlm_preds:
                break  # 如果已经mask的数量大于等于num_mlm_preds则停止mask
            masked_token_id = None
            # 80%的时间：将词替换为['MASK']词元，但这里是直接替换为['MASK']对应的id
            if random.random() < self.masked_rate:
                random_num = random.randint(0, 9)
                if random_num < 8:
                   masked_token_id = self.MASK_IDS
                   mlm_input_tokens_id[mlm_pred_position] = masked_token_id
                   pred_positions.append(mlm_pred_position)  # 保留被mask位置的索引信息
                elif random_num < 9:
                     # 10%的时间：将词被替换成任意其他的标记
                     random_token_id = random.choice(range(2))  # 假设vocab_size是词汇表大小
                     mlm_input_tokens_id[mlm_pred_position] = random_token_id
            else:
                continue
        # 构造mlm任务中需要预测位置对应的正确标签，如果其没出现在pred_positions则表示该位置不是mask位置
        # 则在进行损失计算时需要忽略掉这些位置（即为PAD_IDX）；而如果其出现在mask的位置，则其标签为原始token_ids对应的id
        print(f'pad{self.PAD_IDX}')
        mlm_label = [self.PAD_IDX if idx not in pred_positions
                     else token_ids[idx] for idx in range(len(token_ids))]

        return mlm_input_tokens_id, mlm_label
    
    def __getitem__(self, index):
        random.seed(42)
        # 转化为token_id

        # 取随机起始位点的索引
        # rand_start = random.randint(0, self.max_start)
        # rand_start = random.randint(0, self.max_start)
        gene_idx = index//len(self.arr)
        gene_site = self.arr[index%len(self.arr)]
    
        # token_ids = self.data[gene_idx][rand_start:rand_start + self.max_position_embeddings]

        token_ids = self.data[gene_idx][gene_site:gene_site + self.max_position_embeddings]#恢复恢复

        token_ids = [self.vocab[str(token)] for token in token_ids] 
        # print(token_ids)  
        # np.save('old.npy',token_ids)
        # position_embedding = self.pe[rand_start: rand_start + self.max_position_embeddings]
        position_embedding = self.pe[gene_site: gene_site + self.max_position_embeddings]#恢复恢复
        # for key,value in self.vocab.stoi.items():
        #     token_ids[token_ids==key]=value
        # token_ids = [self.vocab[str(token)] for token in token_id]
        # self.data[gene_idx][gene_site: gene_site + self.max_position_embeddings]
     
        # token_ids = np.array()
        mlm_input_tokens_id, mlm_label = self.get_masked_sample(token_ids) 
        LD_matrix = self.LD_data[gene_site:gene_site + self.max_position_embeddings,:]
        # print(type(LD_matrix))
       
        # print(LD_matrix)
        # print("mlm type",type(mlm_input_tokens_id))
        # print(mlm_input_tokens_id.shape)
        # print("mlm_label type",type(mlm_label))
        # mlm_input_tokens_id = mlm_input_tokens_id.tolist()
        token_ids = torch.tensor(mlm_input_tokens_id, dtype=torch.long)
        mlm_label = torch.tensor(mlm_label, dtype=torch.long)
        LD_matrix = torch.tensor(LD_matrix,dtype=torch.float32)
        MAF_g = self.MAF[gene_site:gene_site+self.max_position_embeddings] 
        return token_ids, mlm_label, position_embedding,LD_matrix,MAF_g
        
    def __len__(self):
        return len(self.arr)*self.samples
    

class randomDataset_torch_double_test(Dataset):
    def __init__(self, 
                 nda:torch.tensor, 
                 ref:torch.tensor,
                 LD_data:torch.tensor,
                 MAF:np.ndarray,
                 hidden_size:int,
                 max_position_embeddings:int, 
                 masked_rate=0.4, 
                 vocab:Vocab=None, 
                 pad_index=0,
                 masked_token_rate=0.8,
                 masked_token_unchanged_rate=0.5,
                 step = 1,
                 length_edge = 0,
                 ) -> None:
        super().__init__()
        self.positions = nda[0]

        # self.positions = self.positions.reshape(-1,1024)
        # first_pos = self.positions[:,0].unsqueeze(1)
        # self.positions = self.positions - first_pos

        self.positions_tensor = self.positions.reshape(-1,1)
        # self.positions_tensor = torch.arange(1024).reshape(-1,1)
        print(self.positions_tensor)
        self.data = nda[1:]
        self.ref = ref[1:]
        self.MAF = MAF

        self.LD_data = LD_data.to(torch.float32)
        self.samples = self.data.shape[0]
        self.length = self.data.shape[1]
        self.max_position_embeddings = max_position_embeddings
        self.max_start = self.length - max_position_embeddings
        self.vocab = vocab
        self.masked_rate = masked_rate
        self.masked_token_rate = masked_token_rate
        self.masked_token_unchanged_rate = masked_token_unchanged_rate
        self.PAD_IDX = pad_index
        self.SEP_IDX = self.vocab['[SEP]']
        self.CLS_IDX = self.vocab['[CLS]']
        self.MASK_IDS = self.vocab['[MASK]']
        self.step = step
        self.length_edge = length_edge
        self.start = 0
        self.arr = list(range(0,self.max_start+1,self.step))
        # print(f'step:{self.step}')
        # print(self.arr)
        self.index_arr = list(range(0,self.samples,1))
        self.length_arr = np.arange(self.max_position_embeddings)
        # random.shuffle(self.index_arr)

        print("len",self.max_start)
        print("nda data shape",self.data.shape)
        print("numbers of site",len(self.arr))
        print("samples numbers",self.samples)
        print("length of index_arr",len(self.index_arr))
        print("length_edge",self.length_edge)
        print("LD_data",LD_data)
        # 预先计算正余弦编码
        self.pe = torch.zeros(self.length, hidden_size)
        self.div_term = torch.exp(torch.arange(0, hidden_size, 2) * -(math.log(10000.0) / hidden_size))
        self.pe[:, 0::2] = torch.sin(self.positions_tensor * self.div_term)  # 用正弦波给偶数部分赋值
        self.pe[:, 1::2] = torch.cos(self.positions_tensor * self.div_term)  # 用余弦波给奇数部分赋值

        # print(self.data[0][0], self.vocab[str(self.data[0][0])])
    
    def get_masked_sample(self, token_ids):
        """
        本函数的作用是将传入的 一段token_ids的其中部分进行mask处理
        :param token_ids:         e.g. [101, 1031, 4895, 2243, 1033, 10029, 2000, 2624, 1031,....]
        :return: mlm_input_tokens_id:  [101, 1031, 103, 2243, 1033, 10029, 2000, 103,  1031, ...]
                           mlm_label:  [ 0,   0,   4895,  0,    0,    0,    0,   2624,  0,...]
        """
        masked_rate = self.masked_rate
        candidate_pred_positions = []  # 候选预测位置的索引
        for i, ids in enumerate(token_ids):
            # 在遮蔽语言模型任务中不会预测特殊词元，所以如果该位置是特殊词元
            # 那么该位置就不会成为候选mask位置
            if ids in [self.CLS_IDX, self.SEP_IDX]:
                continue
            if i < self.length_edge or i > (self.max_position_embeddings - self.length_edge -1):
                continue
            candidate_pred_positions.append(i)
            # 保存候选位置的索引， 例如可能是 [ 2,3,4,5, ....]
        # random.shuffle(candidate_pred_positions)  # 将所有候选位置打乱，更利于后续随机
        # 被掩盖位置的数量，BERT模型中默认将15%的Token进行mask
        num_mlm_preds = max(1, round(len(token_ids) * masked_rate))
        # logging.debug(f" ## Mask数量为: {num_mlm_preds}")
        mlm_input_tokens_id, mlm_label = self.replace_masked_tokens(
            token_ids, candidate_pred_positions, num_mlm_preds)
        return mlm_input_tokens_id, mlm_label
        
    def replace_masked_tokens(self, token_ids, candidate_pred_positions, num_mlm_preds):
        """
        本函数的作用是根据给定的token_ids、候选mask位置以及需要mask的数量来返回被mask后的token_ids以及标签信息
        :param token_ids:
        :param candidate_pred_positions:
        :param num_mlm_preds:
        :return:
        """


        
        pred_positions = []
        mlm_input_tokens_id = [token_id for token_id in token_ids]
        for mlm_pred_position in candidate_pred_positions:
            if len(pred_positions) >= num_mlm_preds:
                break  # 如果已经mask的数量大于等于num_mlm_preds则停止mask
            masked_token_id = None
            # 80%的时间：将词替换为['MASK']词元，但这里是直接替换为['MASK']对应的id
            if random.random() < self.masked_rate:
                random_num = random.randint(0, 9)
                if random_num < 8:
                   masked_token_id = self.MASK_IDS
                   mlm_input_tokens_id[mlm_pred_position] = masked_token_id
                   pred_positions.append(mlm_pred_position)  # 保留被mask位置的索引信息
                elif random_num < 9:
                     # 10%的时间：将词被替换成任意其他的标记
                     random_token_id = random.choice(range(2))  # 假设vocab_size是词汇表大小
                     mlm_input_tokens_id[mlm_pred_position] = random_token_id
            else:
                continue
        # 构造mlm任务中需要预测位置对应的正确标签，如果其没出现在pred_positions则表示该位置不是mask位置
        # 则在进行损失计算时需要忽略掉这些位置（即为PAD_IDX）；而如果其出现在mask的位置，则其标签为原始token_ids对应的id
        mlm_label = [self.PAD_IDX if idx not in pred_positions
                     else token_ids[idx] for idx in range(len(token_ids))]
        return mlm_input_tokens_id, mlm_label
    


    
    def process_list(self,main_list, reference_list):
        """
        Processes a list containing 0s, 1s, and 2s based on specific rules for 2s.

        Args:
            main_list (list): The list to be processed, containing only 0, 1, or 2.
            reference_list (list): The list used for replacement when a 2 falls into
                                    the 10% replacement category. Must be of the same
                                    length as main_list.

        Returns:
            list: The processed list.
        """
        if len(main_list) != len(reference_list):
            raise ValueError("The main list and reference list must have the same length.")
        processed_list = []
        for i, item in enumerate(main_list):
            if item == 2:
                rand_num = random.random() # Generates a float between 0.0 and 1.0
                
                if rand_num < 0:  
                    processed_list.append(random.choice([0, 1]))
                    # reference_list[i] = self.PAD_IDX
                elif rand_num < 1:  
                    processed_list.append(self.MASK_IDS) # Keep it as 2
                else:  # 10% chance (between 0.9 and 1.0)
                    processed_list.append(reference_list[i])
                    # reference_list[i] = self.PAD_IDX
                # processed_list.append(self.MASK_IDS) # Keep it as 2

            else:
                processed_list.append(item) # Keep 0s and 1s as they are
                # processed_list.append(self.MASK_IDS) # Keep 0s and 1s as they are
                reference_list[i] = self.PAD_IDX

        return np.array(processed_list),np.array(reference_list)
    def __getitem__(self, index):
        # start = time.time()
        # 转化为token_id
        # 取随机起始位点的索引
        # rand_start = random.randint(0, self.max_start)
        # rand_start = random.randint(0, self.max_start)

        gene_idx = index//len(self.arr)
        gene_site = self.arr[index%len(self.arr)]
        # gene_idx = index
        # gene_site = 0
    
        # token_ids = self.data[gene_idx][rand_start:rand_start + self.max_position_embeddings]

        token_ids = self.data[gene_idx][gene_site:gene_site + self.max_position_embeddings]#恢复恢复
        ref_token_ids = self.ref[gene_idx][gene_site:gene_site + self.max_position_embeddings]

        # token_ids.tolist()
        # np.save('new.npy',token_ids)
        token_ids = np.array(token_ids)
        ref_token_ids = np.array(ref_token_ids)
        # print(token_ids[0].dtype)
        token_ids = [self.vocab[str(token)] for token in token_ids]
        # print(ref_token_ids[0].dtype)

        ref_token_ids = [self.vocab[str(token)] for token in ref_token_ids]   

        # np.save('new.npy',token_ids)
        # position_embedding = self.pe[rand_start: rand_start + self.max_position_embeddings]
        position_embedding = self.pe[gene_site: gene_site + self.max_position_embeddings]#恢复恢复
        # for key,value in self.vocab.stoi.items():
        #     token_ids[token_ids==key]=value
        # token_ids = [self.vocab[str(token)] for token in token_id]
        # self.data[gene_idx][gene_site: gene_site + self.max_position_embeddings]
     
        # token_ids = np.array()
        # mlm_input_tokens_id, mlm_label = self.get_masked_sample(token_ids) 
        MAF_g = self.MAF[gene_site:gene_site+self.max_position_embeddings] 

        mlm_input_tokens_id, mlm_label = self.process_list(token_ids,ref_token_ids)
        # LD_matrix = torch.arange(gene_site,gene_site + self.max_position_embeddings)
        # print(type(LD_matrix))
       
        LD_matrix = self.LD_data[gene_site:gene_site + self.max_position_embeddings,:]
        # print(LD_matrix.shape)

        # mlm_input_tokens_id = mlm_input_tokens_id.tolist()
        token_ids = torch.tensor(mlm_input_tokens_id, dtype=torch.long)
        mlm_label = torch.tensor(mlm_label, dtype=torch.long)
        # LD_matrix = torch.tensor(LD_matrix,dtype=torch.float32)
        chunk_id = index%len(self.arr)
        # print(mlm_label)
        # raise RuntimeError
        return token_ids, mlm_label, position_embedding ,LD_matrix,MAF_g,chunk_id
    def __len__(self):
        return len(self.arr)*self.samples
        # return self.samples
    


class randomDataset_MACHRSQ(Dataset):
    def __init__(self, 
                 nda:torch.tensor, 
                 LD_data:torch.tensor,
                 MAF:np.ndarray,
                 hidden_size:int,
                 max_position_embeddings:int, 
                 masked_rate=0.4, 
                 vocab:Vocab=None, 
                 pad_index=0,
                 masked_token_rate=0.8,
                 masked_token_unchanged_rate=0.5,
                 step = 1,
                 length_edge = 0,
                 ) -> None:
        super().__init__()
        self.positions = nda[0]
        self.positions_tensor = self.positions.reshape(-1,1)

        self.data = nda[1:]
        self.MAF = MAF

        self.LD_data = LD_data.to(torch.float32)
        self.samples = self.data.shape[0]
        self.length = self.data.shape[1]
        self.max_position_embeddings = max_position_embeddings
        self.max_start = self.length - max_position_embeddings
        self.vocab = vocab
        self.masked_rate = random.uniform(0.85, 0.95)
        # self.masked_rate = 0.1
        self.masked_token_rate = masked_token_rate
        self.masked_token_unchanged_rate = masked_token_unchanged_rate
        self.PAD_IDX = pad_index
        self.SEP_IDX = self.vocab['[SEP]']
        self.CLS_IDX = self.vocab['[CLS]']
        self.MASK_IDS = self.vocab['[MASK]']
        self.step = step
        self.length_edge = length_edge
        self.start = 0
        self.arr = list(range(0,self.max_start+1,self.step))
        self.index_arr = list(range(0,self.samples,1))
        self.length_arr = np.arange(self.max_position_embeddings)
        # random.shuffle(self.index_arr)

        print("len",self.max_start)
        print("nda data shape",self.data.shape)
        print("numbers of site",len(self.arr))
        print("samples numbers",self.samples)
        print("length of index_arr",len(self.index_arr))
        print("length_edge",self.length_edge)
        print("LD_data",LD_data)
        # 预先计算正余弦编码
        self.pe = torch.zeros(self.length, hidden_size)
        self.div_term = torch.exp(torch.arange(0, hidden_size, 2) * -(math.log(10000.0) / hidden_size))
        self.pe[:, 0::2] = torch.sin(self.positions_tensor * self.div_term)  # 用正弦波给偶数部分赋值
        self.pe[:, 1::2] = torch.cos(self.positions_tensor * self.div_term)  # 用余弦波给奇数部分赋值

        # print(self.data[0][0], self.vocab[str(self.data[0][0])])
    
    def get_masked_sample(self, token_ids):
        """
        本函数的作用是将传入的 一段token_ids的其中部分进行mask处理
        :param token_ids:         e.g. [101, 1031, 4895, 2243, 1033, 10029, 2000, 2624, 1031,....]
        :return: mlm_input_tokens_id:  [101, 1031, 103, 2243, 1033, 10029, 2000, 103,  1031, ...]
                           mlm_label:  [ 0,   0,   4895,  0,    0,    0,    0,   2624,  0,...]
        """
        # masked_rate = self.masked_rate
        # self.masked_rate = random.uniform(0.85, 0.95)
        candidate_pred_positions = []  # 候选预测位置的索引
        for i, ids in enumerate(token_ids):
            # 在遮蔽语言模型任务中不会预测特殊词元，所以如果该位置是特殊词元
            # 那么该位置就不会成为候选mask位置
            if ids in [self.CLS_IDX, self.SEP_IDX]:
                continue
            if i < self.length_edge or i > (self.max_position_embeddings - self.length_edge -1):
                continue
            candidate_pred_positions.append(i)
            # 保存候选位置的索引， 例如可能是 [ 2,3,4,5, ....]
        # random.shuffle(candidate_pred_positions)  # 将所有候选位置打乱，更利于后续随机
        # 被掩盖位置的数量，BERT模型中默认将15%的Token进行mask
        num_mlm_preds = max(1, round(len(token_ids) * self.masked_rate))
        # logging.debug(f" ## Mask数量为: {num_mlm_preds}")
        mlm_input_tokens_id, mlm_label = self.replace_masked_tokens(
            token_ids, candidate_pred_positions, num_mlm_preds)
        return mlm_input_tokens_id, mlm_label
        
    def replace_masked_tokens(self, token_ids, candidate_pred_positions, num_mlm_preds):
        """
        本函数的作用是根据给定的token_ids、候选mask位置以及需要mask的数量来返回被mask后的token_ids以及标签信息
        :param token_ids:
        :param candidate_pred_positions:
        :param num_mlm_preds:
        :return:
        """


        
        pred_positions = []

        mlm_input_tokens_id = [token_id for token_id in token_ids]
        for mlm_pred_position in candidate_pred_positions:
            if len(pred_positions) >= num_mlm_preds:
                break  # 如果已经mask的数量大于等于num_mlm_preds则停止mask
            masked_token_id = None
            # 80%的时间：将词替换为['MASK']词元，但这里是直接替换为['MASK']对应的id
            if random.random() < self.masked_rate:
                random_num = random.randint(0, 9)
                if random_num < 8:
                    masked_token_id = self.MASK_IDS
                    mlm_input_tokens_id[mlm_pred_position] = masked_token_id
                    
                elif random_num < 9:
                    # 10%的时间：将词被替换成任意其他的标记
                    random_token_id = random.choice(range(2))  # 假设vocab_size是词汇表大小
                    mlm_input_tokens_id[mlm_pred_position] = random_token_id

                pred_positions.append(mlm_pred_position)  # 保留被mask位置的索引信息
            else:
                continue

        # 构造mlm任务中需要预测位置对应的正确标签，如果其没出现在pred_positions则表示该位置不是mask位置
        # 则在进行损失计算时需要忽略掉这些位置（即为PAD_IDX）；而如果其出现在mask的位置，则其标签为原始token_ids对应的id

        mlm_label = [self.PAD_IDX if idx not in pred_positions
                     else token_ids[idx] for idx in range(len(token_ids))]
        # print(f'mlm_label :{mlm_label}')

        return mlm_input_tokens_id, mlm_label
    
    def __getitem__(self, index):
        # start = time.time()
        # 转化为token_id
        # 取随机起始位点的索引
        # rand_start = random.randint(0, self.max_start)
        # rand_start = random.randint(0, self.max_start)
        # random.seed(43)
        gene_idx = index//len(self.arr)//4
        gene_site = self.arr[index%len(self.arr)]
    
        # token_ids = self.data[gene_idx][rand_start:rand_start + self.max_position_embeddings]

        token_ids = self.data[gene_idx:gene_idx+4,gene_site:gene_site + self.max_position_embeddings]#恢复恢复

        # token_ids.tolist()
        # np.save('new.npy',token_ids)
        token_ids = np.array(token_ids)
        token_ids = [[self.vocab[str(token)] for token in token_ids[i]] for i in range(4)]
        # np.save('new.npy',token_ids)
        # position_embedding = self.pe[rand_start: rand_start + self.max_position_embeddings]
        position_embedding = self.pe[gene_site: gene_site + self.max_position_embeddings]#恢复恢复
        # for key,value in self.vocab.stoi.items():
        #     token_ids[token_ids==key]=value
        # token_ids = [self.vocab[str(token)] for token in token_id]
        # self.data[gene_idx][gene_site: gene_site + self.max_position_embeddings]
     
        # token_ids = np.array()


        # mlm_input_tokens_id, mlm_label = self.get_masked_sample(token_ids) 

        mlm_input_tokens_id,mlm_label = self.get_masked_sample(token_ids[0])
        mlm_input_tokens_id,mlm_label = torch.tensor(mlm_input_tokens_id).unsqueeze(0),torch.tensor(mlm_label).unsqueeze(0)
        for i in range(1,len(token_ids)):
            tt,ll = self.get_masked_sample(token_ids[i])
            tt,ll = torch.tensor(tt),torch.tensor(ll)
            mlm_input_tokens_id = torch.cat([mlm_input_tokens_id,tt.unsqueeze(0)],dim = 0)
            mlm_label = torch.cat([mlm_label,ll.unsqueeze(0)],dim = 0)

        # LD_matrix = torch.arange(gene_site,gene_site + self.max_position_embeddings)
        # print(type(LD_matrix))
        LD_matrix = self.LD_data[gene_site:gene_site + self.max_position_embeddings,:]

        


        # mlm_input_tokens_id = mlm_input_tokens_id.tolist()
        # token_ids = torch.tensor(mlm_input_tokens_id, dtype=torch.long)
        # mlm_label = torch.tensor(mlm_label, dtype=torch.long)
        # LD_matrix = torch.tensor(LD_matrix,dtype=torch.float32)
        MAF_g = self.MAF[:self.max_position_embeddings] 

        return mlm_input_tokens_id, mlm_label, position_embedding.unsqueeze(0).repeat(4, 1,1),LD_matrix.unsqueeze(0).repeat(4, 1,1),MAF_g
    def __len__(self):
        return len(self.arr)*self.samples//4
    

def generate_batch_MACH(data_batch, PAD_IDX=3, max_len=max_length):
    b_token_ids, b_mlm_label, b_positions,b_LD_matrices = [], [], [], []
    b_MAF_g= []

    for (token_ids, mlm_label, positions,LD_matrix,MAF_g) in data_batch:
        # 开始对一个batch中的每一个样本进行处理
        for i in range(len(token_ids)):
            b_token_ids.append(token_ids[i])
            b_mlm_label.append(mlm_label[i])
            b_positions.append(positions[i])

            b_LD_matrices.append(LD_matrix[i])
        b_MAF_g.append(MAF_g)


    # print("aaabbbccc")
    b_token_ids = pad_sequence(b_token_ids,  # [batch_size,max_len]
                               padding_value=PAD_IDX,
                               batch_first=False,
                               max_len=max_len)
    b_token_ids = b_token_ids.transpose(0,1)                        
    #b_token_ids:  [batch_size,src_len]
    
    
    b_mlm_label = pad_sequence(b_mlm_label,  # [batch_size,max_len]
                               padding_value=PAD_IDX,
                               batch_first=False,
                               max_len=max_len)
   # b_mlm_label:  [src_len, batch_size]
   
    b_mask = (b_token_ids == 100000).transpose(0, 1)
    
   # b_mask: [max_len, batch_size]
    b_mlm_label = b_mlm_label.transpose(0, 1)
    b_mask = b_mask.transpose(0, 1) 
    # print(f"b_positions.type: {type(b_positions)}")
    b_positions = torch.stack(b_positions)
    b_LD_matrices = pad_sequences_3d(b_LD_matrices,
    
                                     padding_value=PAD_IDX,
                                     batch_first=False,
                                     )
    
    # print(b_LD_matrices[0].shape)

    return b_token_ids, b_mask, b_mlm_label, b_positions, b_LD_matrices,b_MAF_g









class randomDataset_torch_prefix(Dataset):
    def __init__(self, 
                 nda:torch.tensor, 
                 LD_data:torch.tensor,
                 MAF:np.ndarray,
                 hidden_size:int,
                 max_position_embeddings:int, 
                 masked_rate=0.4, 
                 vocab:Vocab=None, 
                 pad_index=0,
                 masked_token_rate=0.8,
                 masked_token_unchanged_rate=0.5,
                 step = 1,
                 length_edge = 0,
                 ) -> None:
        super().__init__()
        self.positions = nda[0]
        # self.positions_tensor= self.positions[:96*1024].reshape(-1,1)
        #change for rela&abs
        # self.positions_tensor = self.positions_tensor.reshape(-1,1024)
        # self.positions_tensor = self.positions_tensor - self.positions_tensor[:, 0].unsqueeze(1)
        # print(self.positions_tensor)

        self.positions_tensor = self.positions.reshape(-1,1)
        # raise RuntimeError
        # self.positions_tensor = torch.arange(self.positions.shape[0]).reshape(-1,1)
        print(self.positions_tensor)
        self.data = nda[1:]
        self.MAF = MAF
        self.pre_pos = torch.tensor([-3,-2,-1]).reshape(-1,1)

        self.LD_data = LD_data.to(torch.float32)
        self.samples = self.data.shape[0]
        self.length = self.data.shape[1]
        self.max_position_embeddings = max_position_embeddings
        self.max_start = self.length - max_position_embeddings
        self.vocab = vocab

        
        self.masked_rate = random.uniform(0.85, 0.95)
        # self.masked_rate = 0.4
        # self.masked_rate = random.uniform(0.35, 0.45)
        # self.masked_rate = 0.01
        self.masked_token_rate = masked_token_rate
        self.masked_token_unchanged_rate = masked_token_unchanged_rate
        self.PAD_IDX = pad_index
        self.SEP_IDX = self.vocab['[SEP]']
        self.CLS_IDX = self.vocab['[CLS]']
        self.MASK_IDS = self.vocab['[MASK]']
        self.step = step
        self.length_edge = length_edge
        self.start = 0
        self.arr = list(range(0,self.max_start+1,self.step))
        self.index_arr = list(range(0,self.samples,1))
        self.length_arr = np.arange(self.max_position_embeddings)
        self.hidden_size = hidden_size
        # random.shuffle(self.index_arr)

        print("len",self.max_start)
        print("nda data shape",self.data.shape)
        print("numbers of site",len(self.arr))
        print("samples numbers",self.samples)
        print("length of index_arr",len(self.index_arr))
        print("length_edge",self.length_edge)
        print("LD_data",LD_data)
        # 预先计算正余弦编码
        # self.pe = torch.zeros(self.length, hidden_size)
        self.pe = torch.zeros(self.length, hidden_size)
        self.pre_pe = torch.zeros(3, hidden_size)

        # self.CLS_PE = torch.zeros(1, hidden_size)
        self.div_term = torch.exp(torch.arange(0, hidden_size, 2) * -(math.log(1000000.0) / hidden_size))
        # self.positions_tensor = self.positions_tensor / 10000
        self.pe[:, 0::2] = torch.sin(self.positions_tensor * self.div_term)  # 用正弦波给偶数部分赋值
        self.pe[:, 1::2] = torch.cos(self.positions_tensor * self.div_term)  # 用余弦波给奇数部分赋值
        # self.pre_pe[:, 1::2] = torch.cos(self.pre_pos * self.div_term)  # 用余弦波给奇数部分赋值
        # self.pre_pe[:, 1::2] = torch.cos(self.pre_pos * self.div_term)  # 用余弦波给奇数部分赋值
        # print(self.data[0][0], self.vocab[str(self.data[0][0])])
    
    def get_masked_sample(self, token_ids):
        """
        本函数的作用是将传入的 一段token_ids的其中部分进行mask处理
        :param token_ids:         e.g. [101, 1031, 4895, 2243, 1033, 10029, 2000, 2624, 1031,....]
        :return: mlm_input_tokens_id:  [101, 1031, 103, 2243, 1033, 10029, 2000, 103,  1031, ...]
                           mlm_label:  [ 0,   0,   4895,  0,    0,    0,    0,   2624,  0,...]
        """
        # masked_rate = self.masked_rate
        # self.masked_rate = random.uniform(0.85, 0.95)
        candidate_pred_positions = []  # 候选预测位置的索引
        for i, ids in enumerate(token_ids):
            # 在遮蔽语言模型任务中不会预测特殊词元，所以如果该位置是特殊词元
            # 那么该位置就不会成为候选mask位置
            if ids in [self.CLS_IDX, self.SEP_IDX]:
                continue
            if i < self.length_edge or i > (self.max_position_embeddings - self.length_edge -1):
                continue
            candidate_pred_positions.append(i)
            # 保存候选位置的索引， 例如可能是 [ 2,3,4,5, ....]
        # random.shuffle(candidate_pred_positions)  # 将所有候选位置打乱，更利于后续随机
        # 被掩盖位置的数量，BERT模型中默认将15%的Token进行mask
        num_mlm_preds = max(1, round(len(token_ids) * random.uniform(0.85, 0.95)))
        # logging.debug(f" ## Mask数量为: {num_mlm_preds}")
        mlm_input_tokens_id, mlm_label = self.replace_masked_tokens(
            token_ids, candidate_pred_positions, num_mlm_preds)
        return mlm_input_tokens_id, mlm_label
        
    def replace_masked_tokens(self, token_ids, candidate_pred_positions, num_mlm_preds):
        """
        本函数的作用是根据给定的token_ids、候选mask位置以及需要mask的数量来返回被mask后的token_ids以及标签信息
        :param token_ids:
        :param candidate_pred_positions:
        :param num_mlm_preds:
        :return:
        """


        
        pred_positions = []

        mlm_input_tokens_id = [token_id for token_id in token_ids]
        for mlm_pred_position in candidate_pred_positions:
            if len(pred_positions) >= num_mlm_preds:
                break  # 如果已经mask的数量大于等于num_mlm_preds则停止mask
            masked_token_id = None
            # 80%的时间：将词替换为['MASK']词元，但这里是直接替换为['MASK']对应的id
            if random.random() < self.masked_rate:
                random_num = random.randint(0, 9)
                if random_num < 8:
                    masked_token_id = self.MASK_IDS
                    mlm_input_tokens_id[mlm_pred_position] = masked_token_id
                    
                elif random_num < 9:
                    # 10%的时间：将词被替换成任意其他的标记
                    random_token_id = random.choice(range(2))  # 假设vocab_size是词汇表大小
                    mlm_input_tokens_id[mlm_pred_position] = random_token_id

                pred_positions.append(mlm_pred_position)  # 保留被mask位置的索引信息
            else:
                continue
        # 构造mlm任务中需要预测位置对应的正确标签，如果其没出现在pred_positions则表示该位置不是mask位置
        # 则在进行损失计算时需要忽略掉这些位置（即为PAD_IDX）；而如果其出现在mask的位置，则其标签为原始token_ids对应的id

        mlm_label = [self.PAD_IDX if idx not in pred_positions
                     else token_ids[idx] for idx in range(len(token_ids))]
        # print(f'mlm_label :{mlm_label}')
        return mlm_input_tokens_id, mlm_label
    
    def __getitem__(self, index):
        # start = time.time()
        # 转化为token_id
        # 取随机起始位点的索引
        # rand_start = random.randint(0, self.max_start)
        # rand_start = random.randint(0, self.max_start)
        # random.seed(43)

        gene_idx = index//len(self.arr)
        gene_site = self.arr[index%len(self.arr)]
        # gene_idx = index
        # gene_site = 0
        chunk_id  = index%len(self.arr)
    
        # token_ids = self.data[gene_idx][rand_start:rand_start + self.max_position_embeddings]

        token_ids = self.data[gene_idx][gene_site:gene_site + self.max_position_embeddings]#恢复恢复
        
        # token_ids.tolist()
        # np.save('new.npy',token_ids)
        token_ids = np.array(token_ids)
        token_ids = [self.vocab[str(token)] for token in token_ids]
        # np.save('new.npy',token_ids)
        # position_embedding = self.pe[rand_start: rand_start + self.max_position_embeddings]
        position_embedding = self.pe[gene_site:gene_site + self.max_position_embeddings]#恢复恢复
        # for key,value in self.vocab.stoi.items():
        #     token_ids[token_ids==key]=value
        # token_ids = [self.vocab[str(token)] for token in token_id]
        # self.data[gene_idx][gene_site: gene_site + self.max_position_embeddings]
     
        # token_ids = np.array()
        mlm_input_tokens_id, mlm_label = self.get_masked_sample(token_ids) 
        # LD_matrix = torch.arange(gene_site,gene_site + self.max_position_embeddings)
        # print(type(LD_matrix))
        LD_matrix = self.LD_data[gene_site:gene_site + self.max_position_embeddings,:]

        # LD_matrix = add_zero_padding(LD_matrix)
        # mlm_input_tokens_id = mlm_input_tokens_id.tolist()
        # print(f'fuc:{ [6] + mlm_input_tokens_id}')
        mlm_input_tokens_id = [self.vocab.__getitem__('[CLS]'),self.vocab.__getitem__('Chr_22'),self.vocab.__getitem__(f'Chunk_{chunk_id}')] + mlm_input_tokens_id
        token_ids = torch.tensor(mlm_input_tokens_id, dtype=torch.long)
 
        mlm_label = [3,3,3] + mlm_label
        mlm_label = torch.tensor(mlm_label, dtype=torch.long)

        # LD_matrix = torch.tensor(LD_matrix,dtype=torch.float32)
        MAF_g = self.MAF[:self.max_position_embeddings] 
        # position_embedding = torch.cat([self.CLS_PE,position_embedding],dim=0)
        # print(f'cywcnm:{token_ids.shape}')
        position_embedding = torch.cat([self.pre_pe,position_embedding],dim=0)
        # print(pos_res)
        return token_ids, mlm_label, position_embedding,LD_matrix,MAF_g,chunk_id
    def __len__(self):
        return len(self.arr)*self.samples
        # return self.samples
    







class randomDataset_torch_double_test_pre(Dataset):
    def __init__(self, 
                 nda:torch.tensor, 
                 ref:torch.tensor,
                 LD_data:torch.tensor,
                 MAF:np.ndarray,
                 hidden_size:int,
                 max_position_embeddings:int, 
                 masked_rate=0.4, 
                 vocab:Vocab=None, 
                 pad_index=0,
                 masked_token_rate=0.8,
                 masked_token_unchanged_rate=0.5,
                 step = 1,
                 length_edge = 0,
                 ) -> None:
        super().__init__()
        self.positions = nda[0]
        
        self.positions_tensor = self.positions.reshape(-1,1)

        # self.positions_tensor = torch.arange(1024).reshape(-1,1)
        # self.positions_tensor = self.positions_tensor.reshape(-1,1024)
        # self.positions_tensor = self.positions_tensor - self.positions_tensor[:, 0].unsqueeze(1)
        # self.positions_tensor = self.positions_tensor.reshape(-1,1)
        self.data = nda[1:]
        self.ref = ref[1:]
        self.MAF = MAF

        self.LD_data = LD_data.to(torch.float32)
        self.samples = self.data.shape[0]
        self.length = self.data.shape[1]
        self.max_position_embeddings = max_position_embeddings
        self.max_start = self.length - max_position_embeddings
        self.vocab = vocab
        self.masked_rate = masked_rate
        self.masked_token_rate = masked_token_rate
        self.masked_token_unchanged_rate = masked_token_unchanged_rate
        self.PAD_IDX = pad_index
        self.SEP_IDX = self.vocab['[SEP]']
        self.CLS_IDX = self.vocab['[CLS]']
        self.MASK_IDS = self.vocab['[MASK]']
        self.step = step
        self.length_edge = length_edge
        self.start = 0
        self.arr = list(range(0,self.max_start+1,self.step))
        # print(f'step:{self.step}')
        # print(self.arr)
        self.index_arr = list(range(0,self.samples,1))
        self.length_arr = np.arange(self.max_position_embeddings)
        # random.shuffle(self.index_arr)

        print("len",self.max_start)
        print("nda data shape",self.data.shape)
        print("numbers of site",len(self.arr))
        print("samples numbers",self.samples)
        print("length of index_arr",len(self.index_arr))
        print("length_edge",self.length_edge)
        print("LD_data",LD_data)
        # 预先计算正余弦编码
        self.pre_pos = torch.tensor([-3,-2,-1]).reshape(-1,1)

        self.pe = torch.zeros(self.length, hidden_size)
        self.pre_pe = torch.zeros(3, hidden_size)
        self.div_term = torch.exp(torch.arange(0, hidden_size, 2) * -(math.log(1000000.0) / hidden_size))
        self.pe[:, 0::2] = torch.sin(self.positions_tensor * self.div_term)  # 用正弦波给偶数部分赋值
        self.pe[:, 1::2] = torch.cos(self.positions_tensor * self.div_term)  # 用余弦波给奇数部分赋值
        # self.pre_pe[:, 0::2] = torch.sin(self.pre_pos * self.div_term)  # 用正弦波给偶数部分赋值
        # self.pre_pe[:, 1::2] = torch.cos(self.pre_pos * self.div_term)  # 用余弦波给奇数部分赋值
        # print(self.data[0][0], self.vocab[str(self.data[0][0])])
    
    def get_masked_sample(self, token_ids):
        """
        本函数的作用是将传入的 一段token_ids的其中部分进行mask处理
        :param token_ids:         e.g. [101, 1031, 4895, 2243, 1033, 10029, 2000, 2624, 1031,....]
        :return: mlm_input_tokens_id:  [101, 1031, 103, 2243, 1033, 10029, 2000, 103,  1031, ...]
                           mlm_label:  [ 0,   0,   4895,  0,    0,    0,    0,   2624,  0,...]
        """
        masked_rate = self.masked_rate
        candidate_pred_positions = []  # 候选预测位置的索引
        for i, ids in enumerate(token_ids):
            # 在遮蔽语言模型任务中不会预测特殊词元，所以如果该位置是特殊词元
            # 那么该位置就不会成为候选mask位置
            if ids in [self.CLS_IDX, self.SEP_IDX]:
                continue
            if i < self.length_edge or i > (self.max_position_embeddings - self.length_edge -1):
                continue
            candidate_pred_positions.append(i)
            # 保存候选位置的索引， 例如可能是 [ 2,3,4,5, ....]
        # random.shuffle(candidate_pred_positions)  # 将所有候选位置打乱，更利于后续随机
        # 被掩盖位置的数量，BERT模型中默认将15%的Token进行mask
        num_mlm_preds = max(1, round(len(token_ids) * masked_rate))
        # logging.debug(f" ## Mask数量为: {num_mlm_preds}")
        mlm_input_tokens_id, mlm_label = self.replace_masked_tokens(
            token_ids, candidate_pred_positions, num_mlm_preds)
        return mlm_input_tokens_id, mlm_label
        
    def replace_masked_tokens(self, token_ids, candidate_pred_positions, num_mlm_preds):
        """
        本函数的作用是根据给定的token_ids、候选mask位置以及需要mask的数量来返回被mask后的token_ids以及标签信息
        :param token_ids:
        :param candidate_pred_positions:
        :param num_mlm_preds:
        :return:
        """


        
        pred_positions = []
        mlm_input_tokens_id = [token_id for token_id in token_ids]
        for mlm_pred_position in candidate_pred_positions:
            if len(pred_positions) >= num_mlm_preds:
                break  # 如果已经mask的数量大于等于num_mlm_preds则停止mask
            masked_token_id = None
            # 80%的时间：将词替换为['MASK']词元，但这里是直接替换为['MASK']对应的id
            if random.random() < self.masked_rate:
                random_num = random.randint(0, 9)
                if random_num < 8:
                   masked_token_id = self.MASK_IDS
                   mlm_input_tokens_id[mlm_pred_position] = masked_token_id
                   pred_positions.append(mlm_pred_position)  # 保留被mask位置的索引信息
                elif random_num < 9:
                     # 10%的时间：将词被替换成任意其他的标记
                     random_token_id = random.choice(range(2))  # 假设vocab_size是词汇表大小
                     mlm_input_tokens_id[mlm_pred_position] = random_token_id
            else:
                continue
        # 构造mlm任务中需要预测位置对应的正确标签，如果其没出现在pred_positions则表示该位置不是mask位置
        # 则在进行损失计算时需要忽略掉这些位置（即为PAD_IDX）；而如果其出现在mask的位置，则其标签为原始token_ids对应的id
        mlm_label = [self.PAD_IDX if idx not in pred_positions
                     else token_ids[idx] for idx in range(len(token_ids))]
        return mlm_input_tokens_id, mlm_label
    


    
    def process_list(self,main_list, reference_list):
        """
        Processes a list containing 0s, 1s, and 2s based on specific rules for 2s.

        Args:
            main_list (list): The list to be processed, containing only 0, 1, or 2.
            reference_list (list): The list used for replacement when a 2 falls into
                                    the 10% replacement category. Must be of the same
                                    length as main_list.

        Returns:
            list: The processed list.
        """
        if len(main_list) != len(reference_list):
            raise ValueError("The main list and reference list must have the same length.")
        processed_list = []
        for i, item in enumerate(main_list):
            if item == 2:
                rand_num = random.random() # Generates a float between 0.0 and 1.0

                if rand_num < 0.1:  
                    processed_list.append(random.choice([0, 1]))
                    # reference_list[i] = self.PAD_IDX
                elif rand_num < 0.9:  
                    processed_list.append(self.MASK_IDS) # Keep it as 2
                else:  # 10% chance (between 0.9 and 1.0)
                    processed_list.append(reference_list[i])
                    # reference_list[i] = self.PAD_IDX
                # processed_list.append(self.MASK_IDS) # Keep it as 2

            else:
                processed_list.append(item) # Keep 0s and 1s as they are
                reference_list[i] = self.PAD_IDX

        return np.array(processed_list),np.array(reference_list)
    def __getitem__(self, index):
        # start = time.time()
        # 转化为token_id
        # 取随机起始位点的索引
        # rand_start = random.randint(0, self.max_start)
        # rand_start = random.randint(0, self.max_start)
        # print('what')
        gene_idx = index//len(self.arr)
        gene_site = self.arr[index%len(self.arr)]
        # gene_idx = index
        # gene_site = 0

        # token_ids = self.data[gene_idx][rand_start:rand_start + self.max_position_embeddings]

        token_ids = self.data[gene_idx][gene_site:gene_site + self.max_position_embeddings]#恢复恢复
        ref_token_ids = self.ref[gene_idx][gene_site:gene_site + self.max_position_embeddings]

        # token_ids.tolist()
        # np.save('new.npy',token_ids)
        token_ids = np.array(token_ids)
        ref_token_ids = np.array(ref_token_ids)
        # print(token_ids[0].dtype)
        token_ids = [self.vocab[str(token)] for token in token_ids]
        # print(ref_token_ids[0].dtype)

        ref_token_ids = [self.vocab[str(token)] for token in ref_token_ids]   

        # np.save('new.npy',token_ids)
        # position_embedding = self.pe[rand_start: rand_start + self.max_position_embeddings]
        position_embedding = self.pe[gene_site: gene_site + self.max_position_embeddings]#恢复恢复
        # for key,value in self.vocab.stoi.items():
        #     token_ids[token_ids==key]=value
        # token_ids = [self.vocab[str(token)] for token in token_id]
        # self.data[gene_idx][gene_site: gene_site + self.max_position_embeddings]
     
        # token_ids = np.array()
        # mlm_input_tokens_id, mlm_label = self.get_masked_sample(token_ids) 
        mlm_input_tokens_id, mlm_label = self.process_list(token_ids,ref_token_ids)
        # LD_matrix = torch.arange(gene_site,gene_site + self.max_position_embeddings)
        # print(type(LD_matrix))
       
        LD_matrix = self.LD_data[gene_site:gene_site + self.max_position_embeddings,:]
        # print(LD_matrix.shape)

        # mlm_input_tokens_id = mlm_input_tokens_id.tolist()
        chunk_id = index%len(self.arr)
        
        mlm_input_tokens_id = [self.vocab.__getitem__('[CLS]'),self.vocab.__getitem__('Chr_22'),self.vocab.__getitem__(f'Chunk_{chunk_id}')] + mlm_input_tokens_id.tolist()

        token_ids = torch.tensor(mlm_input_tokens_id, dtype=torch.long)
        mlm_label = [3,3,3] + mlm_label.tolist()

        mlm_label = torch.tensor(mlm_label, dtype=torch.long)
        # LD_matrix = torch.tensor(LD_matrix,dtype=torch.float32)
        MAF_g = self.MAF[gene_site:gene_site+self.max_position_embeddings] 
        position_embedding = torch.cat([self.pre_pe,position_embedding],dim=0)

        return token_ids, mlm_label, position_embedding,LD_matrix,MAF_g,chunk_id
    def __len__(self):
        return len(self.arr)*self.samples
        # return self.samples
    




class randomDataset_torch_sampling(Dataset):
    def __init__(self, 
                 nda:torch.tensor, 
                 LD_data:torch.tensor,
                 MAF:np.ndarray,
                 hidden_size:int,
                 max_position_embeddings:int, 
                 masked_rate=0.4, 
                 vocab:Vocab=None, 
                 pad_index=0,
                 masked_token_rate=0.8,
                 masked_token_unchanged_rate=0.5,
                 step = 1,
                 length_edge = 0,
                 ) -> None:
        super().__init__()
        self.positions = nda[0]

        # self.positions = self.positions.reshape(-1,1024)
        # first_pos = self.positions[:,0].unsqueeze(1)
        # self.positions = self.positions - first_pos

        # self.positions = self.positions - self.positions[0]
        # self.positions = (self.positions - self.positions[0]) / (self.positions[-1] - self.positions[0]) * 10000
        self.positions_tensor= self.positions.reshape(-1,1)
        # self.positions_tensor = torch.arange(1024).reshape(-1,1)
        # print(self.positions_tensor)
        self.data = nda[1:]
        self.MAF = MAF

        self.LD_data = LD_data.to(torch.float32)
        self.samples = self.data.shape[0]
        self.length = self.data.shape[1]
        self.max_position_embeddings = max_position_embeddings
        self.max_start = self.length - max_position_embeddings
        self.vocab = vocab
        # self.masked_rate = 0.2
        # self.masked_rate = 0.4
        self.masked_rate = random.uniform(0.85, 0.95)
        # self.masked_rate = random.uniform(0.35, 0.45)
        # self.masked_rate = 0.01
        self.masked_token_rate = masked_token_rate
        self.masked_token_unchanged_rate = masked_token_unchanged_rate
        self.PAD_IDX = pad_index
        self.SEP_IDX = self.vocab['[SEP]']
        self.CLS_IDX = self.vocab['[CLS]']
        self.MASK_IDS = self.vocab['[MASK]']
        self.step = step
        self.length_edge = length_edge
        self.start = 0
        self.arr = list(range(0,self.max_start+1,self.step))
        self.index_arr = list(range(0,self.samples,1))
        self.length_arr = np.arange(self.max_position_embeddings)
        self.hidden_size = hidden_size
        # random.shuffle(self.index_arr)

        print("len",self.max_start)
        print("nda data shape",self.data.shape)
        print("numbers of site",len(self.arr))
        print("samples numbers",self.samples)
        print("length of index_arr",len(self.index_arr))
        print("length_edge",self.length_edge)
        print("LD_data",LD_data)
        # 预先计算正余弦编码
        self.pe = torch.zeros(self.length, hidden_size)
        print(f"hiddensize:{hidden_size}")
        # self.CLS_PE = torch.zeros(1, hidden_size)
        self.div_term = torch.exp(torch.arange(0, hidden_size, 2) * -(math.log(10000.0) / hidden_size))
        # self.positions_tensor = self.positions_tensor / 10000
        self.pe[:, 0::2] = torch.sin(self.positions_tensor * self.div_term)  # 用正弦波给偶数部分赋值
        self.pe[:, 1::2] = torch.cos(self.positions_tensor * self.div_term)  # 用余弦波给奇数部分赋值


        # print(self.data[0][0], self.vocab[str(self.data[0][0])])
    
    def get_masked_sample(self, token_ids,maf):
        """
        本函数的作用是将传入的 一段token_ids的其中部分进行mask处理
        :param token_ids:         e.g. [101, 1031, 4895, 2243, 1033, 10029, 2000, 2624, 1031,....]
        :return: mlm_input_tokens_id:  [101, 1031, 103, 2243, 1033, 10029, 2000, 103,  1031, ...]
                           mlm_label:  [ 0,   0,   4895,  0,    0,    0,    0,   2624,  0,...]
        """
        # masked_rate = self.masked_rate
        # self.masked_rate = random.uniform(0.85, 0.95)
        candidate_pred_positions = []  # 候选预测位置的索引
        for i, ids in enumerate(token_ids):
            # 在遮蔽语言模型任务中不会预测特殊词元，所以如果该位置是特殊词元
            # 那么该位置就不会成为候选mask位置
            if ids in [self.CLS_IDX, self.SEP_IDX]:
                continue
            if i < self.length_edge or i > (self.max_position_embeddings - self.length_edge -1):
                continue
            candidate_pred_positions.append(i)
            # 保存候选位置的索引， 例如可能是 [ 2,3,4,5, ....]
        # random.shuffle(candidate_pred_positions)  # 将所有候选位置打乱，更利于后续随机
        # 被掩盖位置的数量，BERT模型中默认将15%的Token进行mask
        num_mlm_preds = max(1, round(len(token_ids) * random.uniform(0.85, 0.95)))
        # logging.debug(f" ## Mask数量为: {num_mlm_preds}")
        mlm_input_tokens_id, mlm_label = self.replace_masked_tokens(
            token_ids, candidate_pred_positions, num_mlm_preds,maf)
        return mlm_input_tokens_id, mlm_label
        
    def replace_masked_tokens(self, token_ids, candidate_pred_positions, num_mlm_preds,maf):
        """
        本函数的作用是根据给定的token_ids、候选mask位置以及需要mask的数量来返回被mask后的token_ids以及标签信息
        :param token_ids:
        :param candidate_pred_positions:
        :param num_mlm_preds:
        :return:
        """


        
        pred_positions = []

        mlm_input_tokens_id = [token_id for token_id in token_ids]
        for idx,mlm_pred_position in enumerate(candidate_pred_positions):
            if len(pred_positions) >= num_mlm_preds:
                break  # 如果已经mask的数量大于等于num_mlm_preds则停止mask
            masked_token_id = None
            # 80%的时间：将词替换为['MASK']词元，但这里是直接替换为['MASK']对应的id
            if random.random() < self.masked_rate:
                random_num = random.randint(0, 9)
                maf_site = maf[idx]
                if random_num < 5:
                    masked_token_id = self.MASK_IDS
                    mlm_input_tokens_id[mlm_pred_position] = masked_token_id
                    
                elif random_num < 9:
                    ratio = random.random()
                    if ratio < maf_site:
                        random_token_id = 1
                    else:
                        random_token_id = 0
                    # 10%的时间：将词被替换成任意其他的标记
                    mlm_input_tokens_id[mlm_pred_position] = random_token_id

                pred_positions.append(mlm_pred_position)  # 保留被mask位置的索引信息
            else:
                continue
        # 构造mlm任务中需要预测位置对应的正确标签，如果其没出现在pred_positions则表示该位置不是mask位置
        # 则在进行损失计算时需要忽略掉这些位置（即为PAD_IDX）；而如果其出现在mask的位置，则其标签为原始token_ids对应的id

        mlm_label = [self.PAD_IDX if idx not in pred_positions
                     else token_ids[idx] for idx in range(len(token_ids))]
        # print(f'mlm_label :{mlm_label}')
        return mlm_input_tokens_id, mlm_label
    
    def __getitem__(self, index):
        # start = time.time()
        # 转化为token_id
        # 取随机起始位点的索引
        # rand_start = random.randint(0, self.max_start)
        # rand_start = random.randint(0, self.max_start)
        # random.seed(43)
        gene_idx = index//len(self.arr)
        gene_site = self.arr[index%len(self.arr)]


        chunk_id  = index%len(self.arr)
    
        # token_ids = self.data[gene_idx][rand_start:rand_start + self.max_position_embeddings]

        token_ids = self.data[gene_idx][gene_site:gene_site + self.max_position_embeddings]#恢复恢复
        
        # token_ids.tolist()
        # np.save('new.npy',token_ids)
        token_ids = np.array(token_ids)
        token_ids = [self.vocab[str(token)] for token in token_ids]
        # np.save('new.npy',token_ids)
        # position_embedding = self.pe[rand_start: rand_start + self.max_position_embeddings]
        position_embedding = self.pe[gene_site: gene_site + self.max_position_embeddings]#恢复恢复
        # print(gene_idx,gene_site)
        # for key,value in self.vocab.stoi.items():
        #     token_ids[token_ids==key]=value
        # token_ids = [self.vocab[str(token)] for token in token_id]
        # self.data[gene_idx][gene_site: gene_site + self.max_position_embeddings]
     
        # token_ids = np.array()
        maf = self.MAF[gene_site:gene_site+self.max_position_embeddings]
        mlm_input_tokens_id, mlm_label = self.get_masked_sample(token_ids,maf) 
        # LD_matrix = torch.arange(gene_site,gene_site + self.max_position_embeddings)
        # print(type(LD_matrix))
        LD_matrix = self.LD_data[gene_site:gene_site + self.max_position_embeddings,:]

        # LD_matrix = add_zero_padding(LD_matrix)
        # mlm_input_tokens_id = mlm_input_tokens_id.tolist()
        # print(f'fuc:{ [6] + mlm_input_tokens_id}')
        # mlm_input_tokens_id = [6] + mlm_input_tokens_id
        token_ids = torch.tensor(mlm_input_tokens_id, dtype=torch.long)

        mlm_label = torch.tensor(mlm_label, dtype=torch.long)

        # LD_matrix = torch.tensor(LD_matrix,dtype=torch.float32)
        MAF_g = self.MAF[:self.max_position_embeddings] 
        # position_embedding = torch.cat([self.CLS_PE,position_embedding],dim=0)
        # print(f'cywcnm:{token_ids.shape}')

        # print(pos_res)
        return token_ids, mlm_label, position_embedding,LD_matrix,MAF_g,chunk_id
    def __len__(self):
        return len(self.arr)*self.samples
        # return self.samples



class randomDataset_torch_double_test_sampling(Dataset):
    def __init__(self, 
                 nda:torch.tensor, 
                 ref:torch.tensor,
                 LD_data:torch.tensor,
                 MAF:np.ndarray,
                 hidden_size:int,
                 max_position_embeddings:int, 
                 masked_rate=0.4, 
                 vocab:Vocab=None, 
                 pad_index=0,
                 masked_token_rate=0.8,
                 masked_token_unchanged_rate=0.5,
                 step = 1,
                 length_edge = 0,
                 ) -> None:
        super().__init__()
        self.positions = nda[0]

        # self.positions = self.positions.reshape(-1,1024)
        # first_pos = self.positions[:,0].unsqueeze(1)
        # self.positions = self.positions - first_pos

        self.positions_tensor = self.positions.reshape(-1,1)
        # self.positions_tensor = torch.arange(1024).reshape(-1,1)
        print(self.positions_tensor)
        self.data = nda[1:]
        self.ref = ref[1:]
        self.MAF = MAF

        self.LD_data = LD_data.to(torch.float32)
        self.samples = self.data.shape[0]
        self.length = self.data.shape[1]
        self.max_position_embeddings = max_position_embeddings
        self.max_start = self.length - max_position_embeddings
        self.vocab = vocab
        self.masked_rate = masked_rate
        self.masked_token_rate = masked_token_rate
        self.masked_token_unchanged_rate = masked_token_unchanged_rate
        self.PAD_IDX = pad_index
        self.SEP_IDX = self.vocab['[SEP]']
        self.CLS_IDX = self.vocab['[CLS]']
        self.MASK_IDS = self.vocab['[MASK]']
        self.step = step
        self.length_edge = length_edge
        self.start = 0
        self.arr = list(range(0,self.max_start+1,self.step))
        # print(f'step:{self.step}')
        # print(self.arr)
        self.index_arr = list(range(0,self.samples,1))
        self.length_arr = np.arange(self.max_position_embeddings)
        # random.shuffle(self.index_arr)

        print("len",self.max_start)
        print("nda data shape",self.data.shape)
        print("numbers of site",len(self.arr))
        print("samples numbers",self.samples)
        print("length of index_arr",len(self.index_arr))
        print("length_edge",self.length_edge)
        print("LD_data",LD_data)
        # 预先计算正余弦编码
        self.pe = torch.zeros(self.length, hidden_size)
        self.div_term = torch.exp(torch.arange(0, hidden_size, 2) * -(math.log(10000.0) / hidden_size))
        self.pe[:, 0::2] = torch.sin(self.positions_tensor * self.div_term)  # 用正弦波给偶数部分赋值
        self.pe[:, 1::2] = torch.cos(self.positions_tensor * self.div_term)  # 用余弦波给奇数部分赋值

        # print(self.data[0][0], self.vocab[str(self.data[0][0])])
    
    def get_masked_sample(self, token_ids):
        """
        本函数的作用是将传入的 一段token_ids的其中部分进行mask处理
        :param token_ids:         e.g. [101, 1031, 4895, 2243, 1033, 10029, 2000, 2624, 1031,....]
        :return: mlm_input_tokens_id:  [101, 1031, 103, 2243, 1033, 10029, 2000, 103,  1031, ...]
                           mlm_label:  [ 0,   0,   4895,  0,    0,    0,    0,   2624,  0,...]
        """
        masked_rate = self.masked_rate
        candidate_pred_positions = []  # 候选预测位置的索引
        for i, ids in enumerate(token_ids):
            # 在遮蔽语言模型任务中不会预测特殊词元，所以如果该位置是特殊词元
            # 那么该位置就不会成为候选mask位置
            if ids in [self.CLS_IDX, self.SEP_IDX]:
                continue
            if i < self.length_edge or i > (self.max_position_embeddings - self.length_edge -1):
                continue
            candidate_pred_positions.append(i)
            # 保存候选位置的索引， 例如可能是 [ 2,3,4,5, ....]
        # random.shuffle(candidate_pred_positions)  # 将所有候选位置打乱，更利于后续随机
        # 被掩盖位置的数量，BERT模型中默认将15%的Token进行mask
        num_mlm_preds = max(1, round(len(token_ids) * masked_rate))
        # logging.debug(f" ## Mask数量为: {num_mlm_preds}")
        mlm_input_tokens_id, mlm_label = self.replace_masked_tokens(
            token_ids, candidate_pred_positions, num_mlm_preds)
        return mlm_input_tokens_id, mlm_label
        
    def replace_masked_tokens(self, token_ids, candidate_pred_positions, num_mlm_preds):
        """
        本函数的作用是根据给定的token_ids、候选mask位置以及需要mask的数量来返回被mask后的token_ids以及标签信息
        :param token_ids:
        :param candidate_pred_positions:
        :param num_mlm_preds:
        :return:
        """


        
        pred_positions = []
        mlm_input_tokens_id = [token_id for token_id in token_ids]
        for mlm_pred_position in candidate_pred_positions:
            if len(pred_positions) >= num_mlm_preds:
                break  # 如果已经mask的数量大于等于num_mlm_preds则停止mask
            masked_token_id = None
            # 80%的时间：将词替换为['MASK']词元，但这里是直接替换为['MASK']对应的id
            if random.random() < self.masked_rate:
                random_num = random.randint(0, 9)
                if random_num < 8:
                   masked_token_id = self.MASK_IDS
                   mlm_input_tokens_id[mlm_pred_position] = masked_token_id
                   pred_positions.append(mlm_pred_position)  # 保留被mask位置的索引信息
                elif random_num < 9:
                     # 10%的时间：将词被替换成任意其他的标记
                     random_token_id = random.choice(range(2))  # 假设vocab_size是词汇表大小
                     mlm_input_tokens_id[mlm_pred_position] = random_token_id
            else:
                continue
        # 构造mlm任务中需要预测位置对应的正确标签，如果其没出现在pred_positions则表示该位置不是mask位置
        # 则在进行损失计算时需要忽略掉这些位置（即为PAD_IDX）；而如果其出现在mask的位置，则其标签为原始token_ids对应的id
        mlm_label = [self.PAD_IDX if idx not in pred_positions
                     else token_ids[idx] for idx in range(len(token_ids))]
        return mlm_input_tokens_id, mlm_label
    


    
    def process_list(self,main_list, reference_list,maf):
        """
        Processes a list containing 0s, 1s, and 2s based on specific rules for 2s.

        Args:
            main_list (list): The list to be processed, containing only 0, 1, or 2.
            reference_list (list): The list used for replacement when a 2 falls into
                                    the 10% replacement category. Must be of the same
                                    length as main_list.

        Returns:
            list: The processed list.
        """
        if len(main_list) != len(reference_list):
            raise ValueError("The main list and reference list must have the same length.")
        processed_list = []
        for i, item in enumerate(main_list):
            if item == 2:
                rand_num = random.random() # Generates a float between 0.0 and 1.0
                maf_site = maf[i]
                if rand_num < 0.6:  
                    sample = random.random()
                    if sample<maf_site:
                        processed_list.append(1)
                    else:
                        processed_list.append(0)
                    reference_list[i] = self.PAD_IDX

                    # reference_list[i] = self.PAD_IDX
                else:  
                    processed_list.append(self.MASK_IDS) # Keep it as 2

                    # reference_list[i] = self.PAD_IDX
                # processed_list.append(self.MASK_IDS) # Keep it as 2

            else:
                processed_list.append(item) # Keep 0s and 1s as they are
                # processed_list.append(self.MASK_IDS) # Keep 0s and 1s as they are
                reference_list[i] = self.PAD_IDX

        return np.array(processed_list),np.array(reference_list)
    def __getitem__(self, index):
        # start = time.time()
        # 转化为token_id
        # 取随机起始位点的索引
        # rand_start = random.randint(0, self.max_start)
        # rand_start = random.randint(0, self.max_start)

        gene_idx = index//len(self.arr)
        gene_site = self.arr[index%len(self.arr)]
        # gene_idx = index
        # gene_site = 0
    
        # token_ids = self.data[gene_idx][rand_start:rand_start + self.max_position_embeddings]

        token_ids = self.data[gene_idx][gene_site:gene_site + self.max_position_embeddings]#恢复恢复
        ref_token_ids = self.ref[gene_idx][gene_site:gene_site + self.max_position_embeddings]

        # token_ids.tolist()
        # np.save('new.npy',token_ids)
        token_ids = np.array(token_ids)
        ref_token_ids = np.array(ref_token_ids)
        # print(token_ids[0].dtype)
        token_ids = [self.vocab[str(token)] for token in token_ids]
        # print(ref_token_ids[0].dtype)

        ref_token_ids = [self.vocab[str(token)] for token in ref_token_ids]   

        # np.save('new.npy',token_ids)
        # position_embedding = self.pe[rand_start: rand_start + self.max_position_embeddings]
        position_embedding = self.pe[gene_site: gene_site + self.max_position_embeddings]#恢复恢复
        # for key,value in self.vocab.stoi.items():
        #     token_ids[token_ids==key]=value
        # token_ids = [self.vocab[str(token)] for token in token_id]
        # self.data[gene_idx][gene_site: gene_site + self.max_position_embeddings]
     
        # token_ids = np.array()
        # mlm_input_tokens_id, mlm_label = self.get_masked_sample(token_ids) 
        MAF_g = self.MAF[gene_site:gene_site+self.max_position_embeddings] 

        mlm_input_tokens_id, mlm_label = self.process_list(token_ids,ref_token_ids,MAF_g)
        # LD_matrix = torch.arange(gene_site,gene_site + self.max_position_embeddings)
        # print(type(LD_matrix))
       
        LD_matrix = self.LD_data[gene_site:gene_site + self.max_position_embeddings,:]
        # print(LD_matrix.shape)

        # mlm_input_tokens_id = mlm_input_tokens_id.tolist()
        token_ids = torch.tensor(mlm_input_tokens_id, dtype=torch.long)
        mlm_label = torch.tensor(mlm_label, dtype=torch.long)
        # LD_matrix = torch.tensor(LD_matrix,dtype=torch.float32)
        chunk_id = index%len(self.arr)
        # print(mlm_label)
        # raise RuntimeError
        return token_ids, mlm_label, position_embedding ,LD_matrix,MAF_g,chunk_id
    def __len__(self):
        return len(self.arr)*self.samples

if __name__ == '__main__':
    voc = Vocab("/geniusland/home/wagnqiyu/bert-msa/1/LD_bert_conv/bert_1KGPMLM/newvocab.txt")
    start = time.time()

    # if os.path.exists("./data/temp/chr22/beyond0.001/train.npy"):
    # LD = '/geniusland/home/wagnqiyu/bert-msa/1/LD_bert_conv/data/temp/Ld.pt'
    # LD = '//geniusland/home/wagnqiyu/bert-msa/1/stici_ld5.pt'
    LD = '/geniusland/home/wagnqiyu/bert-msa/1/LD_bert_conv/stici_ld2048.pt'
    # LD = '/geniusland/home/wagnqiyu/bert-msa/1/LD_bert_conv/data/temp/Ld.pt'
    # LD = '/geniusland/home/wagnqiyu/bert-msa/1/stici_ld4096.pt'
    # LD_file = '/geniusland/dataset/GenomeImputation/30x_unzip/chr6_1024_LD_file2.npy'
    LD_data = torch.load(LD)
    # train_nda = torch.load('/geniusland/home/wagnqiyu/bert-msa/1/LD_bert_conv/data/temp/chr22_train.pt')
    # train_nda = torch.load('/geniusland/home/wagnqiyu/bert-msa/1/stici_chr22_double_new.pt')#stici_dataset
    train_nda = torch.load('/geniusland/home/wagnqiyu/bert-msa/1/99314_train.pt')
    # train_nda = torch.load('/geniusland/home/wagnqiyu/bert-msa/1/LD_bert_conv/data/temp/chr22_train.pt')
    ref = torch.load('/geniusland/home/wagnqiyu/bert-msa/1/stici_chr22_gt.pt')
    end_f = time.time()
    print("read process_file time:", end_f-start)

    MAF = np.load('/geniusland/home/wagnqiyu/bert-msa/1/LD_bert_conv/stici_train_distribution.npy',allow_pickle=True)
    #读取LD数据集
    # LD_file = '/geniusland/home/wagnqiyu/bert-msa/1/LD_bert_conv/data/temp/chr22_1024_LD_file.npy'
    # # LD_file = '/geniusland/home/shiqi/1/LD_bert/utils/LD_utils/1024_LD_file.npy'
    # LD_data = np.load(LD_file)
    print("before data",LD_data.shape)
    train_ds = randomDataset_torch(train_nda,
                             LD_data=LD_data,
                             MAF=MAF,
                             hidden_size=1024,
                             max_position_embeddings=2048,
                             masked_rate=0.4,
                             vocab=voc,
                             pad_index=voc['[PAD]'],
                             step = 1024,
                             length_edge = 0)
    train_iter = DataLoader(dataset=train_ds,
                            batch_size=1,
                            collate_fn=generate_batch,
                            drop_last=True,
                            num_workers=4,
                            shuffle =False,
                            pin_memory=True)
    for idx,i in enumerate(train_iter):
        # print(i[0].shape,i[1].shape,i[2].shape,i[3].shape)
        print(i[0],i[1],i[2],i[3])
        torch.save(i,f'/geniusland/home/wagnqiyu/bert-msa/1/LD_bert_conv/check_input_mask_check.pt')
        raise RuntimeError
        pass