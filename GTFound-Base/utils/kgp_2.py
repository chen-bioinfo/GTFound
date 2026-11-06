"""Legacy experimental datasets; use utils.tensor_data for the public CLI."""

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
    'Convert VCF sample GT columns into a haplotype array.'
    ids = df.columns
    if is_impute:
        df = df.reindex(columns=ridx)
        df.to_csv(temp, sep='\t', index=False)
        df = pd.read_csv(temp, sep='\t')
        ids = df.columns
    # Sample genotype columns start at zero-based column 9.

    for id in ids[9:]:
        df[str(id) + "_0"] = df[id].apply(lambda x: int(x[0]))
        df[str(id) + "_1"] = df[id].apply(lambda x: int(x[-1]))

    # Remove the original genotype and unused metadata columns.
    df.drop(columns=ids[5:], inplace=True)
    df.drop(columns=['#CHROM', 'ID'], inplace=True)
    df['POS'] = pd.to_numeric(df['POS'])
    nda = df.to_numpy().T
    return nda

def vcf_process(vcf_file):
        # Skip VCF metadata lines beginning with ##.
    with open(vcf_file, 'r') as file:
        lines = [line for line in file if not line.startswith('##')]

    # Parse the remaining tab-separated VCF records with pandas.
    try:
        val_df = pd.read_csv(StringIO(''.join(lines)), sep='\t', low_memory=False)
        print(val_df.head())  # Preview the parsed records.
    except pd.errors.ParserError as e:
        print(f"ParserError: {e}")
    val_nda = vcfdf_to_hapsnp(val_df)
    return val_nda

class Vocab:
    'Read a fixed vocabulary. itos maps indices to tokens; stoi maps tokens to indices.'
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
    'Pad or truncate one-dimensional tensors. None uses the longest sequence; batch_first selects [batch, length] output.'
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


#     b_token_ids, b_mlm_label, b_positions = [], [], []

#         b_token_ids.append(token_ids)
#         b_mlm_label.append(mlm_label)
#         b_positions.append(positions)


import torch

def pad_sequences_3d(sequences, padding_value=0, max_len = max_length,batch_first=False):
    out_tensors = []
    for tensor in sequences:
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


def generate_batch(data_batch, PAD_IDX=3, max_len=max_length):
    b_token_ids, b_mlm_label, b_positions,b_LD_matrices = [], [], [], []
    b_MAF_g= []
    chunk_id_list = []

    for (token_ids, mlm_label, positions,LD_matrix,MAF_g,chunk_id) in data_batch:
        # Collect each sample in the batch.
        b_token_ids.append(token_ids)
        b_mlm_label.append(mlm_label)
        b_positions.append(positions)
        b_LD_matrices.append(LD_matrix)
        b_MAF_g.append(MAF_g)
        chunk_id_list.append(chunk_id)


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
    b_positions = torch.stack(b_positions)

    b_LD_matrices = pad_sequences_3d(b_LD_matrices,
                                     padding_value=PAD_IDX,
                                     batch_first=False,
                                     )
    chunk_id_t = torch.tensor(chunk_id_list)
    return b_token_ids, b_mask, b_mlm_label, b_positions, b_LD_matrices,b_MAF_g,chunk_id_t

def generate_batch_pre(data_batch, PAD_IDX=3, max_len=max_length):
    b_token_ids, b_mlm_label, b_positions,b_LD_matrices = [], [], [], []
    b_MAF_g= []
    chunk_id_list = []

    for (token_ids, mlm_label, positions,LD_matrix,MAF_g,chunk_id) in data_batch:
        # Collect each sample in the batch.
        b_token_ids.append(token_ids)
        b_mlm_label.append(mlm_label)
        b_positions.append(positions)
        b_LD_matrices.append(LD_matrix)
        b_MAF_g.append(MAF_g)
        chunk_id_list.append(chunk_id)


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
    b_positions = torch.stack(b_positions)

    b_LD_matrices = pad_sequences_3d(b_LD_matrices,
                                     padding_value=PAD_IDX,
                                     batch_first=False,
                                     )
    chunk_id_t = torch.tensor(chunk_id_list)
    return b_token_ids, b_mask, b_mlm_label, b_positions, b_LD_matrices,b_MAF_g,chunk_id_t

def add_zero_padding(tensor):
    'Prepend a zero row and column to a two-dimensional tensor.'
    # Read the original tensor dimensions.
    rows, cols = tensor.size()

    # Prepend a zero column.
    zeros_col = torch.zeros((rows, 1), dtype=tensor.dtype, device=tensor.device)
    tensor_with_col = torch.cat([zeros_col, tensor], dim=1)

    # Prepend a zero row.
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


        self.positions_tensor= self.positions.reshape(-1,1)
        self.data = nda[1:]

        self.LD_data = LD_data.to(torch.float32)
        self.samples = self.data.shape[0]
        self.length = self.data.shape[1]
        self.max_position_embeddings = max_position_embeddings
        self.max_start = self.length - max_position_embeddings
        self.vocab = vocab
        self.masked_rate = 0.4
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
        # Precompute sinusoidal position encodings.
        self.pe = torch.zeros(self.length, hidden_size)
        print(f"hiddensize:{hidden_size}")
        self.div_term = torch.exp(torch.arange(0, hidden_size, 2) * -(math.log(10000.0) / hidden_size))
        self.pe[:, 0::2] = torch.sin(self.positions_tensor * self.div_term)  # Assign sine values to even embedding dimensions.
        self.pe[:, 1::2] = torch.cos(self.positions_tensor * self.div_term)  # Assign cosine values to odd embedding dimensions.


    def get_masked_sample(self, token_ids,maf):
        'Select prediction sites and return masked token IDs and MLM labels.'
        candidate_pred_positions = []  # Candidate prediction site indices.
        for i, ids in enumerate(token_ids):
            # Exclude special tokens from prediction candidates.
            # Skip this position.
            if ids in [self.CLS_IDX, self.SEP_IDX]:
                continue
            if i < self.length_edge or i > (self.max_position_embeddings - self.length_edge -1):
                continue
            candidate_pred_positions.append(i)
            # Collect candidate site indices.

        # Choose the number of masked sites using the configured rate.
        maskrate = 0.4
        num_mlm_preds = max(1, round(len(token_ids) * self.masked_rate))


        mlm_input_tokens_id, mlm_label = self.replace_masked_tokens(
            token_ids, candidate_pred_positions, num_mlm_preds,maf,maskrate)
        return mlm_input_tokens_id, mlm_label

    def replace_masked_tokens(self, token_ids, candidate_pred_positions, num_mlm_preds,maf,maskrate):
        'Apply the legacy masking policy to candidate sites and construct MLM labels.'


        pred_positions = []

        mlm_input_tokens_id = [token_id for token_id in token_ids]
        for mlm_pred_position in candidate_pred_positions:
            if len(pred_positions) >= num_mlm_preds:
                break  # Stop after selecting the requested number of prediction sites.
            masked_token_id = None
            # Replace selected sites with the mask token in this branch.
            if random.random() < self.masked_rate:
                random_num = random.randint(0, 9)
                if random_num < 1:
                    masked_token_id = self.MASK_IDS
                    mlm_input_tokens_id[mlm_pred_position] = masked_token_id

                elif random_num < 9:
                    # Replace selected sites with random allele tokens in this branch.
                    random_token_id = random.choice(range(2))  # Sample an allele token from the biallelic vocabulary.
                    mlm_input_tokens_id[mlm_pred_position] = random_token_id

                pred_positions.append(mlm_pred_position)  # Record the selected prediction site.
            else:
                continue
        # Build MLM labels only for selected prediction sites.
        # Use original tokens at prediction sites and PAD_IDX elsewhere.

        mlm_label = [self.PAD_IDX if idx not in pred_positions
                     else token_ids[idx] for idx in range(len(token_ids))]
        return mlm_input_tokens_id, mlm_label

    def __getitem__(self, index):
        # Convert alleles to vocabulary token IDs.
        # Select the window start index.
        # random.seed(43)
        gene_idx = index//len(self.arr)
        gene_site = self.arr[index%len(self.arr)]


        chunk_id  = index%len(self.arr)


        token_ids = self.data[gene_idx][gene_site:gene_site + self.max_position_embeddings]

        # token_ids.tolist()
        # np.save('new.npy',token_ids)
        token_ids = np.array(token_ids)
        token_ids = [self.vocab[str(token)] for token in token_ids]
        # np.save('new.npy',token_ids)
        position_embedding = self.pe[gene_site: gene_site + self.max_position_embeddings]

        maf = self.MAF[gene_site:gene_site+self.max_position_embeddings]
        mlm_input_tokens_id, mlm_label = self.get_masked_sample(token_ids,maf)
        LD_matrix = self.LD_data[gene_site:gene_site + self.max_position_embeddings,:]

        token_ids = torch.tensor(mlm_input_tokens_id, dtype=torch.long)

        mlm_label = torch.tensor(mlm_label, dtype=torch.long)

        MAF_g = self.MAF[:self.max_position_embeddings]

        return token_ids, mlm_label, position_embedding,LD_matrix,MAF_g,chunk_id
    def __len__(self):
        return len(self.arr)*self.samples


def generate_batch_1(data_batch, PAD_IDX=3, max_len=1024):
    b_token_ids, b_mlm_label, b_positions,b_LD_matrices = [], [], [], []
    for (token_ids, mlm_label, positions,LD_matrix) in data_batch:
        # Collect each sample in the batch.
        b_token_ids.append(token_ids)
        b_mlm_label.append(mlm_label)
        b_positions.append(positions)
        b_LD_matrices.append(LD_matrix)
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
        # Precompute sinusoidal position encodings.
        self.pe = torch.zeros(self.length, hidden_size)
        self.div_term = torch.exp(torch.arange(0, hidden_size, 2) * -(math.log(10000.0) / hidden_size))
        self.pe[:, 0::2] = torch.sin(self.positions_tensor * self.div_term)  # Assign sine values to even embedding dimensions.
        self.pe[:, 1::2] = torch.cos(self.positions_tensor * self.div_term)  # Assign cosine values to odd embedding dimensions.


    def get_masked_sample(self, token_ids):
        'Select prediction sites and return masked token IDs and MLM labels.'
        masked_rate = self.masked_rate
        candidate_pred_positions = []  # Candidate prediction site indices.
        for i, ids in enumerate(token_ids):
            # Exclude special tokens from prediction candidates.
            # Skip this position.
            if ids in [self.CLS_IDX, self.SEP_IDX]:
                continue
            if i < self.length_edge or i > (self.max_position_embeddings - self.length_edge -1):
                continue
            candidate_pred_positions.append(i)
            # Collect candidate site indices.

        # Choose the number of masked sites using the configured rate.
        num_mlm_preds = max(1, round(len(token_ids) * masked_rate))

        mlm_input_tokens_id, mlm_label = self.replace_masked_tokens(
            token_ids, candidate_pred_positions, num_mlm_preds)
        return mlm_input_tokens_id, mlm_label

    def replace_masked_tokens(self, token_ids, candidate_pred_positions, num_mlm_preds):
        'Apply the legacy masking policy to candidate sites and construct MLM labels.'


        pred_positions = []
        mlm_input_tokens_id = [token_id for token_id in token_ids]
        for mlm_pred_position in candidate_pred_positions:
            if len(pred_positions) >= num_mlm_preds:
                break  # Stop after selecting the requested number of prediction sites.
            masked_token_id = None
            # Replace selected sites with the mask token in this branch.
            if random.random() < self.masked_rate:
                random_num = random.randint(0, 9)
                if random_num < 8:
                   masked_token_id = self.MASK_IDS
                   mlm_input_tokens_id[mlm_pred_position] = masked_token_id
                   pred_positions.append(mlm_pred_position)  # Record the selected prediction site.
                elif random_num < 9:
                     # Replace selected sites with random allele tokens in this branch.
                     random_token_id = random.choice(range(2))  # Sample an allele token from the biallelic vocabulary.
                     mlm_input_tokens_id[mlm_pred_position] = random_token_id
            else:
                continue
        # Build MLM labels only for selected prediction sites.
        # Use original tokens at prediction sites and PAD_IDX elsewhere.
        print(f'pad{self.PAD_IDX}')
        mlm_label = [self.PAD_IDX if idx not in pred_positions
                     else token_ids[idx] for idx in range(len(token_ids))]

        return mlm_input_tokens_id, mlm_label

    def __getitem__(self, index):
        random.seed(42)
        # Convert alleles to vocabulary token IDs.

        # Select the window start index.
        gene_idx = index//len(self.arr)
        gene_site = self.arr[index%len(self.arr)]


        token_ids = self.data[gene_idx][gene_site:gene_site + self.max_position_embeddings]

        token_ids = [self.vocab[str(token)] for token in token_ids]
        # np.save('old.npy',token_ids)
        position_embedding = self.pe[gene_site: gene_site + self.max_position_embeddings]

        mlm_input_tokens_id, mlm_label = self.get_masked_sample(token_ids)
        LD_matrix = self.LD_data[gene_site:gene_site + self.max_position_embeddings,:]

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


        self.positions_tensor = self.positions.reshape(-1,1)
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
        # Precompute sinusoidal position encodings.
        self.pe = torch.zeros(self.length, hidden_size)
        self.div_term = torch.exp(torch.arange(0, hidden_size, 2) * -(math.log(10000.0) / hidden_size))
        self.pe[:, 0::2] = torch.sin(self.positions_tensor * self.div_term)  # Assign sine values to even embedding dimensions.
        self.pe[:, 1::2] = torch.cos(self.positions_tensor * self.div_term)  # Assign cosine values to odd embedding dimensions.


    def get_masked_sample(self, token_ids):
        'Select prediction sites and return masked token IDs and MLM labels.'
        masked_rate = self.masked_rate
        candidate_pred_positions = []  # Candidate prediction site indices.
        for i, ids in enumerate(token_ids):
            # Exclude special tokens from prediction candidates.
            # Skip this position.
            if ids in [self.CLS_IDX, self.SEP_IDX]:
                continue
            if i < self.length_edge or i > (self.max_position_embeddings - self.length_edge -1):
                continue
            candidate_pred_positions.append(i)
            # Collect candidate site indices.

        # Choose the number of masked sites using the configured rate.
        num_mlm_preds = max(1, round(len(token_ids) * masked_rate))

        mlm_input_tokens_id, mlm_label = self.replace_masked_tokens(
            token_ids, candidate_pred_positions, num_mlm_preds)
        return mlm_input_tokens_id, mlm_label

    def replace_masked_tokens(self, token_ids, candidate_pred_positions, num_mlm_preds):
        'Apply the legacy masking policy to candidate sites and construct MLM labels.'


        pred_positions = []
        mlm_input_tokens_id = [token_id for token_id in token_ids]
        for mlm_pred_position in candidate_pred_positions:
            if len(pred_positions) >= num_mlm_preds:
                break  # Stop after selecting the requested number of prediction sites.
            masked_token_id = None
            # Replace selected sites with the mask token in this branch.
            if random.random() < self.masked_rate:
                random_num = random.randint(0, 9)
                if random_num < 8:
                   masked_token_id = self.MASK_IDS
                   mlm_input_tokens_id[mlm_pred_position] = masked_token_id
                   pred_positions.append(mlm_pred_position)  # Record the selected prediction site.
                elif random_num < 9:
                     # Replace selected sites with random allele tokens in this branch.
                     random_token_id = random.choice(range(2))  # Sample an allele token from the biallelic vocabulary.
                     mlm_input_tokens_id[mlm_pred_position] = random_token_id
            else:
                continue
        # Build MLM labels only for selected prediction sites.
        # Use original tokens at prediction sites and PAD_IDX elsewhere.
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
                elif rand_num < 1:
                    processed_list.append(self.MASK_IDS) # Keep it as 2
                else:  # 10% chance (between 0.9 and 1.0)
                    processed_list.append(reference_list[i])
                # processed_list.append(self.MASK_IDS) # Keep it as 2

            else:
                processed_list.append(item) # Keep 0s and 1s as they are
                # processed_list.append(self.MASK_IDS) # Keep 0s and 1s as they are
                reference_list[i] = self.PAD_IDX

        return np.array(processed_list),np.array(reference_list)
    def __getitem__(self, index):
        # Convert alleles to vocabulary token IDs.
        # Select the window start index.

        gene_idx = index//len(self.arr)
        gene_site = self.arr[index%len(self.arr)]


        token_ids = self.data[gene_idx][gene_site:gene_site + self.max_position_embeddings]
        ref_token_ids = self.ref[gene_idx][gene_site:gene_site + self.max_position_embeddings]

        # token_ids.tolist()
        # np.save('new.npy',token_ids)
        token_ids = np.array(token_ids)
        ref_token_ids = np.array(ref_token_ids)
        token_ids = [self.vocab[str(token)] for token in token_ids]

        ref_token_ids = [self.vocab[str(token)] for token in ref_token_ids]

        # np.save('new.npy',token_ids)
        position_embedding = self.pe[gene_site: gene_site + self.max_position_embeddings]

        # mlm_input_tokens_id, mlm_label = self.get_masked_sample(token_ids)
        MAF_g = self.MAF[gene_site:gene_site+self.max_position_embeddings]

        mlm_input_tokens_id, mlm_label = self.process_list(token_ids,ref_token_ids)

        LD_matrix = self.LD_data[gene_site:gene_site + self.max_position_embeddings,:]

        token_ids = torch.tensor(mlm_input_tokens_id, dtype=torch.long)
        mlm_label = torch.tensor(mlm_label, dtype=torch.long)
        chunk_id = index%len(self.arr)
        return token_ids, mlm_label, position_embedding ,LD_matrix,MAF_g,chunk_id
    def __len__(self):
        return len(self.arr)*self.samples


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
        # Precompute sinusoidal position encodings.
        self.pe = torch.zeros(self.length, hidden_size)
        self.div_term = torch.exp(torch.arange(0, hidden_size, 2) * -(math.log(10000.0) / hidden_size))
        self.pe[:, 0::2] = torch.sin(self.positions_tensor * self.div_term)  # Assign sine values to even embedding dimensions.
        self.pe[:, 1::2] = torch.cos(self.positions_tensor * self.div_term)  # Assign cosine values to odd embedding dimensions.


    def get_masked_sample(self, token_ids):
        'Select prediction sites and return masked token IDs and MLM labels.'
        candidate_pred_positions = []  # Candidate prediction site indices.
        for i, ids in enumerate(token_ids):
            # Exclude special tokens from prediction candidates.
            # Skip this position.
            if ids in [self.CLS_IDX, self.SEP_IDX]:
                continue
            if i < self.length_edge or i > (self.max_position_embeddings - self.length_edge -1):
                continue
            candidate_pred_positions.append(i)
            # Collect candidate site indices.

        # Choose the number of masked sites using the configured rate.
        num_mlm_preds = max(1, round(len(token_ids) * self.masked_rate))

        mlm_input_tokens_id, mlm_label = self.replace_masked_tokens(
            token_ids, candidate_pred_positions, num_mlm_preds)
        return mlm_input_tokens_id, mlm_label

    def replace_masked_tokens(self, token_ids, candidate_pred_positions, num_mlm_preds):
        'Apply the legacy masking policy to candidate sites and construct MLM labels.'


        pred_positions = []

        mlm_input_tokens_id = [token_id for token_id in token_ids]
        for mlm_pred_position in candidate_pred_positions:
            if len(pred_positions) >= num_mlm_preds:
                break  # Stop after selecting the requested number of prediction sites.
            masked_token_id = None
            # Replace selected sites with the mask token in this branch.
            if random.random() < self.masked_rate:
                random_num = random.randint(0, 9)
                if random_num < 8:
                    masked_token_id = self.MASK_IDS
                    mlm_input_tokens_id[mlm_pred_position] = masked_token_id

                elif random_num < 9:
                    # Replace selected sites with random allele tokens in this branch.
                    random_token_id = random.choice(range(2))  # Sample an allele token from the biallelic vocabulary.
                    mlm_input_tokens_id[mlm_pred_position] = random_token_id

                pred_positions.append(mlm_pred_position)  # Record the selected prediction site.
            else:
                continue

        # Build MLM labels only for selected prediction sites.
        # Use original tokens at prediction sites and PAD_IDX elsewhere.

        mlm_label = [self.PAD_IDX if idx not in pred_positions
                     else token_ids[idx] for idx in range(len(token_ids))]

        return mlm_input_tokens_id, mlm_label

    def __getitem__(self, index):
        # Convert alleles to vocabulary token IDs.
        # Select the window start index.
        # random.seed(43)
        gene_idx = index//len(self.arr)//4
        gene_site = self.arr[index%len(self.arr)]


        token_ids = self.data[gene_idx:gene_idx+4,gene_site:gene_site + self.max_position_embeddings]

        # token_ids.tolist()
        # np.save('new.npy',token_ids)
        token_ids = np.array(token_ids)
        token_ids = [[self.vocab[str(token)] for token in token_ids[i]] for i in range(4)]
        # np.save('new.npy',token_ids)
        position_embedding = self.pe[gene_site: gene_site + self.max_position_embeddings]


        # mlm_input_tokens_id, mlm_label = self.get_masked_sample(token_ids)

        mlm_input_tokens_id,mlm_label = self.get_masked_sample(token_ids[0])
        mlm_input_tokens_id,mlm_label = torch.tensor(mlm_input_tokens_id).unsqueeze(0),torch.tensor(mlm_label).unsqueeze(0)
        for i in range(1,len(token_ids)):
            tt,ll = self.get_masked_sample(token_ids[i])
            tt,ll = torch.tensor(tt),torch.tensor(ll)
            mlm_input_tokens_id = torch.cat([mlm_input_tokens_id,tt.unsqueeze(0)],dim = 0)
            mlm_label = torch.cat([mlm_label,ll.unsqueeze(0)],dim = 0)

        LD_matrix = self.LD_data[gene_site:gene_site + self.max_position_embeddings,:]


        MAF_g = self.MAF[:self.max_position_embeddings]

        return mlm_input_tokens_id, mlm_label, position_embedding.unsqueeze(0).repeat(4, 1,1),LD_matrix.unsqueeze(0).repeat(4, 1,1),MAF_g
    def __len__(self):
        return len(self.arr)*self.samples//4


def generate_batch_MACH(data_batch, PAD_IDX=3, max_len=max_length):
    b_token_ids, b_mlm_label, b_positions,b_LD_matrices = [], [], [], []
    b_MAF_g= []

    for (token_ids, mlm_label, positions,LD_matrix,MAF_g) in data_batch:
        # Collect each sample in the batch.
        for i in range(len(token_ids)):
            b_token_ids.append(token_ids[i])
            b_mlm_label.append(mlm_label[i])
            b_positions.append(positions[i])

            b_LD_matrices.append(LD_matrix[i])
        b_MAF_g.append(MAF_g)


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
    b_positions = torch.stack(b_positions)
    b_LD_matrices = pad_sequences_3d(b_LD_matrices,

                                     padding_value=PAD_IDX,
                                     batch_first=False,
                                     )


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
        #change for rela&abs

        self.positions_tensor = self.positions.reshape(-1,1)
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
        # Precompute sinusoidal position encodings.
        self.pe = torch.zeros(self.length, hidden_size)
        self.pre_pe = torch.zeros(3, hidden_size)

        self.div_term = torch.exp(torch.arange(0, hidden_size, 2) * -(math.log(1000000.0) / hidden_size))
        self.pe[:, 0::2] = torch.sin(self.positions_tensor * self.div_term)  # Assign sine values to even embedding dimensions.
        self.pe[:, 1::2] = torch.cos(self.positions_tensor * self.div_term)  # Assign cosine values to odd embedding dimensions.


    def get_masked_sample(self, token_ids):
        'Select prediction sites and return masked token IDs and MLM labels.'
        candidate_pred_positions = []  # Candidate prediction site indices.
        for i, ids in enumerate(token_ids):
            # Exclude special tokens from prediction candidates.
            # Skip this position.
            if ids in [self.CLS_IDX, self.SEP_IDX]:
                continue
            if i < self.length_edge or i > (self.max_position_embeddings - self.length_edge -1):
                continue
            candidate_pred_positions.append(i)
            # Collect candidate site indices.

        # Choose the number of masked sites using the configured rate.
        num_mlm_preds = max(1, round(len(token_ids) * random.uniform(0.85, 0.95)))

        mlm_input_tokens_id, mlm_label = self.replace_masked_tokens(
            token_ids, candidate_pred_positions, num_mlm_preds)
        return mlm_input_tokens_id, mlm_label

    def replace_masked_tokens(self, token_ids, candidate_pred_positions, num_mlm_preds):
        'Apply the legacy masking policy to candidate sites and construct MLM labels.'


        pred_positions = []

        mlm_input_tokens_id = [token_id for token_id in token_ids]
        for mlm_pred_position in candidate_pred_positions:
            if len(pred_positions) >= num_mlm_preds:
                break  # Stop after selecting the requested number of prediction sites.
            masked_token_id = None
            # Replace selected sites with the mask token in this branch.
            if random.random() < self.masked_rate:
                random_num = random.randint(0, 9)
                if random_num < 8:
                    masked_token_id = self.MASK_IDS
                    mlm_input_tokens_id[mlm_pred_position] = masked_token_id

                elif random_num < 9:
                    # Replace selected sites with random allele tokens in this branch.
                    random_token_id = random.choice(range(2))  # Sample an allele token from the biallelic vocabulary.
                    mlm_input_tokens_id[mlm_pred_position] = random_token_id

                pred_positions.append(mlm_pred_position)  # Record the selected prediction site.
            else:
                continue
        # Build MLM labels only for selected prediction sites.
        # Use original tokens at prediction sites and PAD_IDX elsewhere.

        mlm_label = [self.PAD_IDX if idx not in pred_positions
                     else token_ids[idx] for idx in range(len(token_ids))]
        return mlm_input_tokens_id, mlm_label

    def __getitem__(self, index):
        # Convert alleles to vocabulary token IDs.
        # Select the window start index.
        # random.seed(43)

        gene_idx = index//len(self.arr)
        gene_site = self.arr[index%len(self.arr)]
        chunk_id  = index%len(self.arr)


        token_ids = self.data[gene_idx][gene_site:gene_site + self.max_position_embeddings]

        # token_ids.tolist()
        # np.save('new.npy',token_ids)
        token_ids = np.array(token_ids)
        token_ids = [self.vocab[str(token)] for token in token_ids]
        # np.save('new.npy',token_ids)
        position_embedding = self.pe[gene_site:gene_site + self.max_position_embeddings]

        mlm_input_tokens_id, mlm_label = self.get_masked_sample(token_ids)
        LD_matrix = self.LD_data[gene_site:gene_site + self.max_position_embeddings,:]

        mlm_input_tokens_id = [self.vocab.__getitem__('[CLS]'),self.vocab.__getitem__('Chr_22'),self.vocab.__getitem__(f'Chunk_{chunk_id}')] + mlm_input_tokens_id
        token_ids = torch.tensor(mlm_input_tokens_id, dtype=torch.long)

        mlm_label = [3,3,3] + mlm_label
        mlm_label = torch.tensor(mlm_label, dtype=torch.long)

        MAF_g = self.MAF[:self.max_position_embeddings]
        position_embedding = torch.cat([self.pre_pe,position_embedding],dim=0)
        return token_ids, mlm_label, position_embedding,LD_matrix,MAF_g,chunk_id
    def __len__(self):
        return len(self.arr)*self.samples


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
        # Precompute sinusoidal position encodings.
        self.pre_pos = torch.tensor([-3,-2,-1]).reshape(-1,1)

        self.pe = torch.zeros(self.length, hidden_size)
        self.pre_pe = torch.zeros(3, hidden_size)
        self.div_term = torch.exp(torch.arange(0, hidden_size, 2) * -(math.log(1000000.0) / hidden_size))
        self.pe[:, 0::2] = torch.sin(self.positions_tensor * self.div_term)  # Assign sine values to even embedding dimensions.
        self.pe[:, 1::2] = torch.cos(self.positions_tensor * self.div_term)  # Assign cosine values to odd embedding dimensions.


    def get_masked_sample(self, token_ids):
        'Select prediction sites and return masked token IDs and MLM labels.'
        masked_rate = self.masked_rate
        candidate_pred_positions = []  # Candidate prediction site indices.
        for i, ids in enumerate(token_ids):
            # Exclude special tokens from prediction candidates.
            # Skip this position.
            if ids in [self.CLS_IDX, self.SEP_IDX]:
                continue
            if i < self.length_edge or i > (self.max_position_embeddings - self.length_edge -1):
                continue
            candidate_pred_positions.append(i)
            # Collect candidate site indices.

        # Choose the number of masked sites using the configured rate.
        num_mlm_preds = max(1, round(len(token_ids) * masked_rate))

        mlm_input_tokens_id, mlm_label = self.replace_masked_tokens(
            token_ids, candidate_pred_positions, num_mlm_preds)
        return mlm_input_tokens_id, mlm_label

    def replace_masked_tokens(self, token_ids, candidate_pred_positions, num_mlm_preds):
        'Apply the legacy masking policy to candidate sites and construct MLM labels.'


        pred_positions = []
        mlm_input_tokens_id = [token_id for token_id in token_ids]
        for mlm_pred_position in candidate_pred_positions:
            if len(pred_positions) >= num_mlm_preds:
                break  # Stop after selecting the requested number of prediction sites.
            masked_token_id = None
            # Replace selected sites with the mask token in this branch.
            if random.random() < self.masked_rate:
                random_num = random.randint(0, 9)
                if random_num < 8:
                   masked_token_id = self.MASK_IDS
                   mlm_input_tokens_id[mlm_pred_position] = masked_token_id
                   pred_positions.append(mlm_pred_position)  # Record the selected prediction site.
                elif random_num < 9:
                     # Replace selected sites with random allele tokens in this branch.
                     random_token_id = random.choice(range(2))  # Sample an allele token from the biallelic vocabulary.
                     mlm_input_tokens_id[mlm_pred_position] = random_token_id
            else:
                continue
        # Build MLM labels only for selected prediction sites.
        # Use original tokens at prediction sites and PAD_IDX elsewhere.
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
                elif rand_num < 0.9:
                    processed_list.append(self.MASK_IDS) # Keep it as 2
                else:  # 10% chance (between 0.9 and 1.0)
                    processed_list.append(reference_list[i])
                # processed_list.append(self.MASK_IDS) # Keep it as 2

            else:
                processed_list.append(item) # Keep 0s and 1s as they are
                reference_list[i] = self.PAD_IDX

        return np.array(processed_list),np.array(reference_list)
    def __getitem__(self, index):
        # Convert alleles to vocabulary token IDs.
        # Select the window start index.
        gene_idx = index//len(self.arr)
        gene_site = self.arr[index%len(self.arr)]


        token_ids = self.data[gene_idx][gene_site:gene_site + self.max_position_embeddings]
        ref_token_ids = self.ref[gene_idx][gene_site:gene_site + self.max_position_embeddings]

        # token_ids.tolist()
        # np.save('new.npy',token_ids)
        token_ids = np.array(token_ids)
        ref_token_ids = np.array(ref_token_ids)
        token_ids = [self.vocab[str(token)] for token in token_ids]

        ref_token_ids = [self.vocab[str(token)] for token in ref_token_ids]

        # np.save('new.npy',token_ids)
        position_embedding = self.pe[gene_site: gene_site + self.max_position_embeddings]

        # mlm_input_tokens_id, mlm_label = self.get_masked_sample(token_ids)
        mlm_input_tokens_id, mlm_label = self.process_list(token_ids,ref_token_ids)

        LD_matrix = self.LD_data[gene_site:gene_site + self.max_position_embeddings,:]

        chunk_id = index%len(self.arr)

        mlm_input_tokens_id = [self.vocab.__getitem__('[CLS]'),self.vocab.__getitem__('Chr_22'),self.vocab.__getitem__(f'Chunk_{chunk_id}')] + mlm_input_tokens_id.tolist()

        token_ids = torch.tensor(mlm_input_tokens_id, dtype=torch.long)
        mlm_label = [3,3,3] + mlm_label.tolist()

        mlm_label = torch.tensor(mlm_label, dtype=torch.long)
        MAF_g = self.MAF[gene_site:gene_site+self.max_position_embeddings]
        position_embedding = torch.cat([self.pre_pe,position_embedding],dim=0)

        return token_ids, mlm_label, position_embedding,LD_matrix,MAF_g,chunk_id
    def __len__(self):
        return len(self.arr)*self.samples


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


        self.positions_tensor= self.positions.reshape(-1,1)
        self.data = nda[1:]
        self.MAF = MAF

        self.LD_data = LD_data.to(torch.float32)
        self.samples = self.data.shape[0]
        self.length = self.data.shape[1]
        self.max_position_embeddings = max_position_embeddings
        self.max_start = self.length - max_position_embeddings
        self.vocab = vocab
        self.masked_rate = random.uniform(0.85, 0.95)
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
        # Precompute sinusoidal position encodings.
        self.pe = torch.zeros(self.length, hidden_size)
        print(f"hiddensize:{hidden_size}")
        self.div_term = torch.exp(torch.arange(0, hidden_size, 2) * -(math.log(10000.0) / hidden_size))
        self.pe[:, 0::2] = torch.sin(self.positions_tensor * self.div_term)  # Assign sine values to even embedding dimensions.
        self.pe[:, 1::2] = torch.cos(self.positions_tensor * self.div_term)  # Assign cosine values to odd embedding dimensions.


    def get_masked_sample(self, token_ids,maf):
        'Select prediction sites and return masked token IDs and MLM labels.'
        candidate_pred_positions = []  # Candidate prediction site indices.
        for i, ids in enumerate(token_ids):
            # Exclude special tokens from prediction candidates.
            # Skip this position.
            if ids in [self.CLS_IDX, self.SEP_IDX]:
                continue
            if i < self.length_edge or i > (self.max_position_embeddings - self.length_edge -1):
                continue
            candidate_pred_positions.append(i)
            # Collect candidate site indices.

        # Choose the number of masked sites using the configured rate.
        num_mlm_preds = max(1, round(len(token_ids) * random.uniform(0.85, 0.95)))

        mlm_input_tokens_id, mlm_label = self.replace_masked_tokens(
            token_ids, candidate_pred_positions, num_mlm_preds,maf)
        return mlm_input_tokens_id, mlm_label

    def replace_masked_tokens(self, token_ids, candidate_pred_positions, num_mlm_preds,maf):
        'Apply the legacy masking policy to candidate sites and construct MLM labels.'


        pred_positions = []

        mlm_input_tokens_id = [token_id for token_id in token_ids]
        for idx,mlm_pred_position in enumerate(candidate_pred_positions):
            if len(pred_positions) >= num_mlm_preds:
                break  # Stop after selecting the requested number of prediction sites.
            masked_token_id = None
            # Replace selected sites with the mask token in this branch.
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
                    # Replace selected sites with random allele tokens in this branch.
                    mlm_input_tokens_id[mlm_pred_position] = random_token_id

                pred_positions.append(mlm_pred_position)  # Record the selected prediction site.
            else:
                continue
        # Build MLM labels only for selected prediction sites.
        # Use original tokens at prediction sites and PAD_IDX elsewhere.

        mlm_label = [self.PAD_IDX if idx not in pred_positions
                     else token_ids[idx] for idx in range(len(token_ids))]
        return mlm_input_tokens_id, mlm_label

    def __getitem__(self, index):
        # Convert alleles to vocabulary token IDs.
        # Select the window start index.
        # random.seed(43)
        gene_idx = index//len(self.arr)
        gene_site = self.arr[index%len(self.arr)]


        chunk_id  = index%len(self.arr)


        token_ids = self.data[gene_idx][gene_site:gene_site + self.max_position_embeddings]

        # token_ids.tolist()
        # np.save('new.npy',token_ids)
        token_ids = np.array(token_ids)
        token_ids = [self.vocab[str(token)] for token in token_ids]
        # np.save('new.npy',token_ids)
        position_embedding = self.pe[gene_site: gene_site + self.max_position_embeddings]

        maf = self.MAF[gene_site:gene_site+self.max_position_embeddings]
        mlm_input_tokens_id, mlm_label = self.get_masked_sample(token_ids,maf)
        LD_matrix = self.LD_data[gene_site:gene_site + self.max_position_embeddings,:]

        token_ids = torch.tensor(mlm_input_tokens_id, dtype=torch.long)

        mlm_label = torch.tensor(mlm_label, dtype=torch.long)

        MAF_g = self.MAF[:self.max_position_embeddings]

        return token_ids, mlm_label, position_embedding,LD_matrix,MAF_g,chunk_id
    def __len__(self):
        return len(self.arr)*self.samples


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


        self.positions_tensor = self.positions.reshape(-1,1)
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
        # Precompute sinusoidal position encodings.
        self.pe = torch.zeros(self.length, hidden_size)
        self.div_term = torch.exp(torch.arange(0, hidden_size, 2) * -(math.log(10000.0) / hidden_size))
        self.pe[:, 0::2] = torch.sin(self.positions_tensor * self.div_term)  # Assign sine values to even embedding dimensions.
        self.pe[:, 1::2] = torch.cos(self.positions_tensor * self.div_term)  # Assign cosine values to odd embedding dimensions.


    def get_masked_sample(self, token_ids):
        'Select prediction sites and return masked token IDs and MLM labels.'
        masked_rate = self.masked_rate
        candidate_pred_positions = []  # Candidate prediction site indices.
        for i, ids in enumerate(token_ids):
            # Exclude special tokens from prediction candidates.
            # Skip this position.
            if ids in [self.CLS_IDX, self.SEP_IDX]:
                continue
            if i < self.length_edge or i > (self.max_position_embeddings - self.length_edge -1):
                continue
            candidate_pred_positions.append(i)
            # Collect candidate site indices.

        # Choose the number of masked sites using the configured rate.
        num_mlm_preds = max(1, round(len(token_ids) * masked_rate))

        mlm_input_tokens_id, mlm_label = self.replace_masked_tokens(
            token_ids, candidate_pred_positions, num_mlm_preds)
        return mlm_input_tokens_id, mlm_label

    def replace_masked_tokens(self, token_ids, candidate_pred_positions, num_mlm_preds):
        'Apply the legacy masking policy to candidate sites and construct MLM labels.'


        pred_positions = []
        mlm_input_tokens_id = [token_id for token_id in token_ids]
        for mlm_pred_position in candidate_pred_positions:
            if len(pred_positions) >= num_mlm_preds:
                break  # Stop after selecting the requested number of prediction sites.
            masked_token_id = None
            # Replace selected sites with the mask token in this branch.
            if random.random() < self.masked_rate:
                random_num = random.randint(0, 9)
                if random_num < 8:
                   masked_token_id = self.MASK_IDS
                   mlm_input_tokens_id[mlm_pred_position] = masked_token_id
                   pred_positions.append(mlm_pred_position)  # Record the selected prediction site.
                elif random_num < 9:
                     # Replace selected sites with random allele tokens in this branch.
                     random_token_id = random.choice(range(2))  # Sample an allele token from the biallelic vocabulary.
                     mlm_input_tokens_id[mlm_pred_position] = random_token_id
            else:
                continue
        # Build MLM labels only for selected prediction sites.
        # Use original tokens at prediction sites and PAD_IDX elsewhere.
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

                else:
                    processed_list.append(self.MASK_IDS) # Keep it as 2

                # processed_list.append(self.MASK_IDS) # Keep it as 2

            else:
                processed_list.append(item) # Keep 0s and 1s as they are
                # processed_list.append(self.MASK_IDS) # Keep 0s and 1s as they are
                reference_list[i] = self.PAD_IDX

        return np.array(processed_list),np.array(reference_list)
    def __getitem__(self, index):
        # Convert alleles to vocabulary token IDs.
        # Select the window start index.

        gene_idx = index//len(self.arr)
        gene_site = self.arr[index%len(self.arr)]


        token_ids = self.data[gene_idx][gene_site:gene_site + self.max_position_embeddings]
        ref_token_ids = self.ref[gene_idx][gene_site:gene_site + self.max_position_embeddings]

        # token_ids.tolist()
        # np.save('new.npy',token_ids)
        token_ids = np.array(token_ids)
        ref_token_ids = np.array(ref_token_ids)
        token_ids = [self.vocab[str(token)] for token in token_ids]

        ref_token_ids = [self.vocab[str(token)] for token in ref_token_ids]

        # np.save('new.npy',token_ids)
        position_embedding = self.pe[gene_site: gene_site + self.max_position_embeddings]

        # mlm_input_tokens_id, mlm_label = self.get_masked_sample(token_ids)
        MAF_g = self.MAF[gene_site:gene_site+self.max_position_embeddings]

        mlm_input_tokens_id, mlm_label = self.process_list(token_ids,ref_token_ids,MAF_g)

        LD_matrix = self.LD_data[gene_site:gene_site + self.max_position_embeddings,:]

        token_ids = torch.tensor(mlm_input_tokens_id, dtype=torch.long)
        mlm_label = torch.tensor(mlm_label, dtype=torch.long)
        chunk_id = index%len(self.arr)
        return token_ids, mlm_label, position_embedding ,LD_matrix,MAF_g,chunk_id
    def __len__(self):
        return len(self.arr)*self.samples
