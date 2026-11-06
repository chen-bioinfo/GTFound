from torch.nn.init import xavier_uniform_
import torch.nn.functional as F
import torch.nn as nn
import copy
import torch


class MyTransformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 ):
        super(MyTransformer, self).__init__()



        #  ================ 编码部分 =====================
        encoder_layer = MyTransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = MyTransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        # ================ 解码部分 =====================
        decoder_layer = MyTransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = MyTransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None,
                memory_mask=None, src_key_padding_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """
        :param src:   [src_len,batch_size,embed_dim]
        :param tgt:  [tgt_len, batch_size, embed_dim]
        :param src_mask:  None
        :param tgt_mask:  [tgt_len, tgt_len]
        :param memory_mask: None
        :param src_key_padding_mask: [batch_size, src_len]
        :param tgt_key_padding_mask: [batch_size, tgt_len]
        :param memory_key_padding_mask:  [batch_size, src_len]
        :return: [tgt_len, batch_size, num_heads * kdim] <==> [tgt_len,batch_size,embed_dim]
        """
        memory, attn_weight = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        # [src_len, batch_size, num_heads * kdim] <==> [src_len,batch_size,embed_dim]
        output = self.decoder(tgt=tgt, memory=memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)
        return output  # [tgt_len, batch_size, num_heads * kdim] <==> [tgt_len,batch_size,embed_dim]

    def generate_square_subsequent_mask(self, sz):
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask  # [sz,sz]


class MyTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(MyTransformerEncoderLayer, self).__init__()

        self.self_attn = MyMultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.activation = F.relu

        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):

        # src2 = self.self_attn(src, src, src, attn_mask=src_mask,
        #                       key_padding_mask=src_key_padding_mask, )[0]  # 计算多头注意力
        src2, attn_weight = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask, )[0]  # 计算多头注意力
        # src2: [src_len,batch_size,num_heads*kdim] num_heads*kdim = embed_dim
        src = src + self.dropout1(src2)  # 残差连接
        src = self.norm1(src)  # [src_len,batch_size,num_heads*kdim]

        src2 = self.activation(self.linear1(src))  # [src_len,batch_size,dim_feedforward]
        src2 = self.linear2(self.dropout(src2))  # [src_len,batch_size,num_heads*kdim]
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attn_weight  # [src_len, batch_size, num_heads * kdim] <==> [src_len,batch_size,embed_dim]


class MyTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(MyTransformerEncoder, self).__init__()

        self.layers = _get_clones(encoder_layer, num_layers)  # 克隆得到多个encoder layers 论文中默认为6
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None):

        output = src
        for mod in self.layers:
            output = mod(output, src_mask=mask,
                         src_key_padding_mask=src_key_padding_mask)  # 多个encoder layers层堆叠后的前向传播过程
        if self.norm is not None:
            output = self.norm(output)
        return output  # [src_len, batch_size, num_heads * kdim] <==> [src_len,batch_size,embed_dim]


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MyTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(MyTransformerDecoderLayer, self).__init__()

        self.self_attn = MyMultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout)
        self.multihead_attn = MyMultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = F.relu

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None):

        tgt2 = self.self_attn(tgt, tgt, tgt,  # [tgt_len,batch_size, embed_dim]
                              attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        # 解码部分输入序列之间'的多头注意力（也就是论文结构图中的Masked Multi-head attention)

        tgt = tgt + self.dropout1(tgt2)  # 接着是残差连接
        tgt = self.norm1(tgt)  # [tgt_len,batch_size, embed_dim]

        tgt2 = self.multihead_attn(tgt, memory, memory,  # [tgt_len, batch_size, embed_dim]
                                   attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]

        # 解码部分的输入经过多头注意力后同编码部分的输出（memory）通过多头注意力机制进行交互
        tgt = tgt + self.dropout2(tgt2)  # 残差连接
        tgt = self.norm2(tgt)  # [tgt_len, batch_size, embed_dim]

        tgt2 = self.activation(self.linear1(tgt))  # [tgt_len, batch_size, dim_feedforward]
        tgt2 = self.linear2(self.dropout(tgt2))  # [tgt_len, batch_size, embed_dim]
        # 最后的两层全连接
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt  # [tgt_len, batch_size, num_heads * kdim] <==> [tgt_len,batch_size,embed_dim]


class MyTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None):
        super(MyTransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None):

        output = tgt  # [tgt_len,batch_size, embed_dim]

        for mod in self.layers:  # 这里的layers就是N层解码层堆叠起来的
            output = mod(output, memory,
                         tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)
        if self.norm is not None:
            output = self.norm(output)

        return output  # [tgt_len, batch_size, num_heads * kdim] <==> [tgt_len,batch_size,embed_dim]


class MyMultiheadAttention(nn.Module):


    def __init__(self, embed_dim, num_heads, dropout=0., bias=True):
        super(MyMultiheadAttention, self).__init__()

        self.embed_dim = embed_dim  # 前面的d_model参数
        self.head_dim = embed_dim // num_heads  # head_dim 指的就是d_k,d_v
        self.kdim = self.head_dim
        self.vdim = self.head_dim

        self.num_heads = num_heads  # 多头个数
        self.dropout = dropout

        assert self.head_dim * num_heads == self.embed_dim, "embed_dim 除以 num_heads必须为整数"
        # 上面的限制条件就是论文中的  d_k = d_v = d_model/n_head 条件

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)  # embed_dim = kdim * num_heads
        # 这里第二个维度之所以是embed_dim，实际上这里是同时初始化了num_heads个W_q堆叠起来的, 也就是num_heads个头
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)  # W_k,  embed_dim = kdim * num_heads
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)  # W_v,  embed_dim = vdim * num_heads
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        # 最后将所有的Z组合起来的时候，也是一次性完成， embed_dim = vdim * num_heads

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):

        return multi_head_attention_forward(query, key, value, self.num_heads,
                                            self.dropout,
                                            self.out_proj.weight, self.out_proj.bias,
                                            training=self.training,
                                            key_padding_mask=key_padding_mask,
                                            q_proj=self.q_proj,
                                            k_proj=self.k_proj,
                                            v_proj=self.v_proj,
                                            attn_mask=attn_mask)


def multi_head_attention_forward(query,  # [tgt_len,batch_size, embed_dim]
                                 key,  # [src_len, batch_size, embed_dim]
                                 value,  # [src_len, batch_size, embed_dim]
                                 num_heads,
                                 dropout_p,
                                 out_proj_weight,  # [embed_dim = vdim * num_heads, embed_dim = vdim * num_heads]
                                 out_proj_bias,
                                 training=True,
                                 key_padding_mask=None,  # [batch_size,src_len/tgt_len]
                                 q_proj=None,  # weight: [embed_dim,kdim * num_heads]  , bias: [embed_dim]
                                 k_proj=None,  # weight: [embed_dim,kdim * num_heads]  , bias: [embed_dim]
                                 v_proj=None,  # weight: [embed_dim,kdim * num_heads]  , bias: [embed_dim]
                                 attn_mask=None,  # [tgt_len,src_len] or [num_heads*batch_size,tgt_len, src_len]
                                 ):
    q = F.linear(query, q_proj.weight, q_proj.bias)
    #  [tgt_len,batch_size, embed_dim] x [embed_dim,kdim * num_heads] = [tgt_len,batch_size,kdim * num_heads]

    k = F.linear(key, k_proj.weight, k_proj.bias)
    # [src_len, batch_size, embed_dim] x [embed_dim, kdim * num_heads] = [src_len, batch_size, kdim * num_heads]

    v = F.linear(value, v_proj.weight, v_proj.bias)
    # [src_len, batch_size, embed_dim] x [embed_dim, vdim * num_heads] = [src_len, batch_size, vdim * num_heads]
    tgt_len, bsz, embed_dim = query.size()  # [tgt_len,batch_size, embed_dim]
    src_len = key.size(0)
    head_dim = embed_dim // num_heads  # num_heads * head_dim = embed_dim
    scaling = float(head_dim) ** -0.5
    q = q * scaling  # [query_len,batch_size,kdim * num_heads]

    if attn_mask is not None:  # [tgt_len,src_len] or [num_heads*batch_size,tgt_len, src_len]
        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0)  # [1, tgt_len,src_len]
            if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                raise RuntimeError('The size of the 2D attn_mask is not correct.')
        elif attn_mask.dim() == 3:
            if list(attn_mask.size()) != [bsz * num_heads, query.size(0), key.size(0)]:
                raise RuntimeError('The size of the 3D attn_mask is not correct.')


    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    # [batch_size * num_heads,tgt_len,kdim]

    k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)  # [batch_size * num_heads,src_len,kdim]
    v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)  # [batch_size * num_heads,src_len,vdim]
    attn_output_weights = torch.bmm(q, k.transpose(1, 2))
    # [batch_size * num_heads,tgt_len,kdim] x [batch_size * num_heads, kdim, src_len]

    if attn_mask is not None:
        attn_output_weights += attn_mask  # [batch_size * num_heads, tgt_len, src_len]

    if key_padding_mask is not None:
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        
        attn_output_weights = attn_output_weights.masked_fill(
            key_padding_mask.unsqueeze(1).unsqueeze(2),  
            float('-inf'))  #
        attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len,
                                                       src_len)  # [batch_size * num_heads, tgt_len, src_len]

    attn_output_weights = F.softmax(attn_output_weights, dim=-1)  # [batch_size * num_heads, tgt_len, src_len]
    attn_output_weights = F.dropout(attn_output_weights, p=dropout_p, training=training)
    attn_output = torch.bmm(attn_output_weights, v)
    # Z = [batch_size * num_heads, tgt_len, src_len]  x  [batch_size * num_heads,src_len,vdim]
    # = # [batch_size * num_heads,tgt_len,vdim]
    # 这就num_heads个Attention(Q,K,V)结果

    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    # 先transpose成 [tgt_len, batch_size* num_heads ,kdim]
    # 再view成 [tgt_len,batch_size,num_heads*kdim]
    attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)

    Z = F.linear(attn_output, out_proj_weight, out_proj_bias)
    return Z, attn_output_weights.sum(dim=1) / num_heads  # average attention weights over heads
