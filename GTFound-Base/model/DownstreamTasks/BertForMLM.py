import logging
from ..BasicBert.Bert1 import BertModel
from ..BasicBert.Bert1 import get_activation
from .BertForNSPAndMLM import BertForLMTransformHead
from .BertForNSPAndMLM import BertForMaskedLM
import torch
import torch.nn as nn
import time
import torch.nn.functional as F

class ImputationLoss(nn.Module):
    def __init__(self, use_r2_loss=True):
        super(ImputationLoss, self).__init__()
        self.ce_loss_obj = nn.CrossEntropyLoss(reduction='mean', ignore_index=3)
        self.kld_loss_obj = nn.KLDivLoss(reduction='none')
        self.use_r2_loss = use_r2_loss

    def calculate_Minimac_R2_1(self,pred_alt_allele_probs, y_true):
        gt_alt_af = torch.sum(torch.eq(y_true, 1), dim=-1).float() / y_true.shape[-1]

        gt_alt_af = gt_alt_af.unsqueeze(dim=-1)
        mask = torch.logical_or(torch.eq(gt_alt_af, 0.0), torch.eq(gt_alt_af, 1.0))
        gt_alt_af = torch.where(mask, torch.tensor(0.5, device=gt_alt_af.device), gt_alt_af)
        denom = gt_alt_af * (1.0 - gt_alt_af)
        denom = torch.where(denom < 0.01, torch.tensor(0.01, device=denom.device), denom)

        r2 = torch.mean(torch.square(pred_alt_allele_probs - gt_alt_af), dim=-1).unsqueeze(dim=-1) / denom
        r2 = torch.where(mask, torch.zeros_like(r2), r2)
        return r2
    def calculate_Minimac_R2(self,pred_alt_allele_probs, gt_alt_af):
        mask = torch.logical_or(torch.eq(gt_alt_af, 0.0), torch.eq(gt_alt_af, 1.0))
        gt_alt_af = torch.where(mask, 0.5, gt_alt_af)
        denom = gt_alt_af * (1.0 - gt_alt_af)
        denom = torch.where(denom < 0.01, 0.01, denom)
        r2 = torch.mean(torch.square(pred_alt_allele_probs - gt_alt_af.unsqueeze(-1).repeat(1,4)), dim=-1) / denom
        r2 = torch.where(mask, torch.zeros_like(r2), r2)
        return r2


    def forward(self, y_true, y_pred,input):
        # Convert y_true to same dtype as y_pred
        cat_loss = self.ce_loss_obj(y_pred.reshape([-1,y_pred.shape[-1]]), y_true.reshape(-1)).unsqueeze(dim=0)
        # KL divergence loss (need to apply log_softmax to input)
        # Note: KLDivLoss in PyTorch expects log probabilities for input
        input_log_softmax = F.log_softmax(y_pred.reshape([-1,y_pred.shape[-1]])[:,:2], dim=-1)
        y_true_tr = torch.where(torch.eq(y_true, 3), input, y_true).to(y_true)
        y_true_tr_1 = F.one_hot(y_true_tr, num_classes=2).to(input_log_softmax)
        attention_mask = torch.ones_like(y_true)  # Initialize a mask over all target positions.
        attention_mask[y_true != 3] = 0  #
        mask = attention_mask.unsqueeze(-1)  # Expand the mask to [batch_size, sequence_length, 1].
        kl_loss = self.kld_loss_obj(input_log_softmax.reshape(-1,2), y_true_tr_1.reshape(-1,2))
        kl_loss = (kl_loss * mask.reshape(-1,1)).sum() / mask.sum()  # Average the loss over the selected mask positions.

        total_loss = cat_loss
        y_alt = F.softmax(y_pred,dim = -1)[:,:,:,1].transpose(1,2).reshape(-1,4)
        gt_alt_af = y_true_tr.sum(dim=1).reshape(-1) / 4
        if self.use_r2_loss:

            r2_loss = self.calculate_Minimac_R2(y_alt,gt_alt_af)


            total_loss += -r2_loss.mean()*4

        return total_loss

class BertForMLM(nn.Module):


    def __init__(self, config, bert_pretrained_model_dir=None):
        super(BertForMLM, self).__init__()
        if bert_pretrained_model_dir is not None:
            self.bert = BertModel.from_pretrained(config, bert_pretrained_model_dir)
        else:  # Initialize model weights when no pretrained directory is supplied.
            self.bert = BertModel(config)
        weights = None
        if 'use_embedding_weight' in config.__dict__ and config.use_embedding_weight:
            weights = self.bert.bert_embeddings.word_embeddings.embedding.weight
            logging.info(f"Sharing token embedding weights with the output layer: {weights.shape}")
        self.mlm_prediction = BertForLMTransformHead(config, weights)
        self.config = config
        self.criterion = ImputationLoss(use_r2_loss=False)
    def forward(self, input_ids,  # [batch_size, src_len]
                attention_mask=None,  # [batch_size, src_len]; True masks padding positions.
                token_type_ids=None,  # [src_len, batch_size]
                position_ids=None,
                masked_lm_labels=None,  # [batch_size, src_len]
                ld_matrices = None,
                next_sentence_labels=None):  # [batch_size]
        if attention_mask is None:
            attention_mask = input_ids.eq(self.config.pad_token_id)
        attention_mask = attention_mask.transpose(0, 1)
        # pooled_output, all_encoder_outputs, attn_weight = self.bert(


        _, all_encoder_outputs, _ = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            ld_matrices = ld_matrices)


        # sequence_output: [src_len, batch_size, hidden_size]

        mlm_prediction_logits = self.mlm_prediction(all_encoder_outputs)


        # mlm_prediction_logits: [src_len, batch_size, vocab_size]
        mlm_prediction_logits = mlm_prediction_logits.transpose(0, 1)
        # mlm_prediction_logits: [batch_size, src_len, vocab_size]


        if masked_lm_labels is not None:
            #             masked_lm_labels.reshape(-1))

            #                     masked_lm_labels.reshape(-1))
            #                     masked_lm_labels.reshape([-1,4,masked_lm_labels.shape[1]]),mlm_prediction_logits,input_ids.reshape([-1,4,input_ids.shape[1]]))
            loss_fct = nn.CrossEntropyLoss(ignore_index=3)
            mlm_loss = loss_fct(mlm_prediction_logits.reshape(-1, self.config.vocab_size),
                                masked_lm_labels.reshape(-1))
            mlm_loss = mlm_loss.unsqueeze(dim=0)
            return mlm_loss, mlm_prediction_logits
        else:
            return mlm_prediction_logits
        # [batch_size, src_len, vocab_size], [batch_size, 2]
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, ignore_index=-1):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.ce_loss = nn.CrossEntropyLoss(reduction='none', ignore_index=ignore_index)

    def forward(self, inputs, targets):
        ce_loss = self.ce_loss(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = (self.alpha * (1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss
