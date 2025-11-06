import os

from torch.cuda import amp
from torch.utils.data import DataLoader
from utils import vcfdf_to_hapsnp, generate_batch, Vocab, vcf_process,randomDataset_torch,randomDataset,generate_batch_1,randomDataset_torch_double_test,randomDataset_MACHRSQ,generate_batch_MACH
from model import BertForMLM
from model import BertConfig
import random
from utils import logger_init,randomDataset_torch_prefix,generate_batch_pre,randomDataset_torch_double_test_pre,randomDataset_torch_sampling,randomDataset_torch_double_test_sampling
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter  
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix, matthews_corrcoef
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import pandas as pd
import math
import argparse
import time
import torch
import os
import logging
import sys
from torch import multiprocessing as mp
import pynvml
import warnings
from itertools import chain
from torch.optim.lr_scheduler import CosineAnnealingLR
from save_load import save_checkpoint, load_checkpoint
import csv

saved_model_path = "./cache"

class ModelConfig:
    def __init__(self,
                 masked_rate=0.5,
                 test_masked_rate=0.5,
                 train_set="chr22_train_5000.csv",
                 test_set="chr22_test_5000.csv",
                 do_logging=True,
                 gpu=0,
                 learning_rate=2e-4,
                 batch_size=32,
                 length_edge = 0,
                 resume = False,
                 checkpoint = '',
                 step = 512,
                 add_information='',
                 gpus = [0,1,2,3],
                 epoch_resume = 0,
                 test_dir = ''):
        self.project_dir = os.path.dirname(os.path.abspath(__file__))
        self.dataset_dir = os.path.join(self.project_dir, 'data', '1KGPMLM')
        self.pretrained_model_dir = os.path.join(
            self.project_dir, "bert_1KGPMLM")
        self.vocab_path = os.path.join(self.pretrained_model_dir, 'vocab.txt')
        self.device = torch.device(
            f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
        self.train_set = train_set
        self.test_set = test_set
        self.train_file_path = os.path.join(
            self.dataset_dir, train_set)
        self.val_file_path = os.path.join(
            self.dataset_dir, test_set)
        self.test_file_path = os.path.join(
            self.dataset_dir, test_set)
        self.model_save_dir = os.path.join(self.project_dir, 'cache/')
        # self.logs_save_dir = os.path.join(self.project_dir, 'logs')
        self.logs_save_dir = os.path.join(
            self.project_dir, '../records/bertmlm/wqy')
        self.data_name = '1KGPMLM'
        self.model_save_path = os.path.join(
            self.model_save_dir, f'model_{self.data_name}.pt')
        self.is_sample_shuffle = True
        self.use_embedding_weight = True
        self.batch_size = batch_size
        self.max_sen_len = None  # 为None时则采用每个batch中最长的样本对该batch中的样本进行padding
        self.pad_index = 0
        self.random_state = 2021
        self.learning_rate = learning_rate
        self.masked_rate = masked_rate
        self.test_masked_rate = test_masked_rate
        self.masked_token_rate = 1
        self.masked_token_unchanged_rate = 1
        self.log_level = logging.DEBUG
        self.use_torch_multi_head = False  # False表示使用model/BasicBert/MyTransformer中的多头实现
        self.epochs = 10000
        self.model_val_per_epoch = 1
        self.do_logging = do_logging
        self.attention = ''
        self.resume = resume
        self.checkpoint = checkpoint
        self.add_information = add_information
        self.if_rotary = False
        #新增代码 date:5.04
        self.length_edge = length_edge
        self.step = step
        self.gpus = gpus
        self.epoch_resume = epoch_resume
        self.test_dir = test_dir
        if do_logging:
            logger_init(log_file_name='1KGP', log_level=self.log_level,
                        log_dir=self.logs_save_dir)
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)
        bert_config_path = os.path.join(
            self.pretrained_model_dir, "config.json")
        bert_config = BertConfig.from_json_file(bert_config_path)
        for key, value in bert_config.__dict__.items():
            self.__dict__[key] = value

        if do_logging:
            logging.info(" ###print the configuration ")
            for key, value in self.__dict__.items():
                logging.info(f"### {key} = {value}")
        self.vocab_size = 6

class suppress_stdout_stderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    '''

    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])


def fakeTokenizer(s):
    return list(s)


def train(config):
    writer = SummaryWriter(f"Bert_{config.hidden_size}_{config.num_hidden_layers}_{config.add_information}/chr22_{config.batch_size}_{config.step}_{config.learning_rate}")  # !!!!!!


    voc = Vocab("./bert_1KGPMLM/vocab.txt")
    start = time.time()


    #CHR22dataset
    LD = './data/temp/Ld.pt'
    train_data = './data/temp/chr22_train.pt'

    #mircoarray dataset
    # LD = '/geniusland/home/wagnqiyu/bert-msa/1/LD_bert_conv/stici_ld2048.pt'
    # LD = '/geniusland/home/wagnqiyu/bert-msa/1/stici_ld.pt'
    # train_data = '/geniusland/home/wagnqiyu/bert-msa/1/99314_train.pt'

    LD_data = torch.load(LD)

    train_nda = torch.load(train_data)

    end_f = time.time()
    print("read process_file time:", end_f-start)


    print("before data",LD_data.shape)
    train_ds = randomDataset_torch(train_nda,
                             LD_data=LD_data,
                             hidden_size=config.hidden_size,
                             max_position_embeddings=config.max_position_embeddings,
                             masked_rate=config.masked_rate,
                             vocab=voc,
                             pad_index=voc['[PAD]'],
                             step = config.max_position_embeddings,
                             length_edge = config.length_edge)
    train_iter = DataLoader(dataset=train_ds,
                            batch_size=config.batch_size,
                            collate_fn=generate_batch,
                            drop_last=True,
                            num_workers=32,
                            shuffle =True,
                            pin_memory=True)


    all_possible_position = train_ds.positions.to(torch.long)
    config.all_possible_position = all_possible_position  # slow

    gpus = config.gpus

    config.device = gpus[0]
    model = BertForMLM(config)

    model.to(config.device)


    model = nn.DataParallel(model, device_ids=gpus, output_device=gpus[0]) 





    # model.train()
    # optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    print("learning rate ",config.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=20, eta_min = 5e-7)


    scaler = amp.GradScaler()
    torch.backends.cuda.matmul.allow_tf32 = True

    # save path
    directory = os.path.join(config.model_save_dir,f'chr22/hs{config.hidden_size}_ln{config.num_hidden_layers}_{config.add_information}')
    if not os.path.exists(directory):
        os.makedirs(directory)

    #resume
    resume_epoch = -1

    if config.resume:
        load_checkpoint(model,config.checkpoint)
        model.train()
        logging.info(f'##load{config.checkpoint},resume epoch：{config.epoch_resume}retrain')

    # resume_epoch = 0
    for epoch in range(config.epochs):
        if config.epoch_resume >= epoch:
            scheduler.step()
            logging.info(f"current learning rate{optimizer.param_groups[0]['lr']}")
            
            continue
        print("start training......")


        start_time = time.time()


        for idx, (b_token_ids, b_mask, b_mlm_label, b_positions,b_LD_matrices,b_maf,b_chunk_id) in enumerate(train_iter):
            global_iter_num = epoch * len(train_iter) + idx + 1  
            b_token_ids = b_token_ids.to(gpus[0])
            b_mask = b_mask.to(gpus[0])


            b_mlm_label = b_mlm_label.to(gpus[0])
            b_positions = b_positions.to(gpus[0])
            b_LD_matrices = b_LD_matrices.to(gpus[0])
            b_chunk_id = b_chunk_id.to(gpus[0])

            model.train()
            optimizer.zero_grad()



            with amp.autocast():
                loss, mlm_logits = model(input_ids=b_token_ids,
                                                      attention_mask=b_mask,
                                                      token_type_ids=None,
                                                      position_ids=b_positions,
                                                      masked_lm_labels=b_mlm_label,
                                                      ld_matrices = b_LD_matrices,
                                                      next_sentence_labels=None,
                                                      if_rotary = config.if_rotary,
                                                      chunk_id = b_chunk_id)


            writer.add_scalar('train-Loss', loss.sum().item(), global_iter_num)       # !!!!!!!!!!!

            scaler.scale(loss).backward(torch.ones(loss.shape).to(gpus[0]))


            if global_iter_num>= 2000 and global_iter_num%1000 == 0 :

                model.eval()

                mlm_acc_seg, _, _ = accuracy(mlm_logits, b_mlm_label, voc['[PAD]'])

                auc = roc_auc(mlm_logits, b_mlm_label, voc['[PAD]'])
                matrix, f1, precision, recall, mcc = f1_precision_recall_mcc(mlm_logits, b_mlm_label, voc['[PAD]'])
                writer.add_scalar('Train mlm acc', mlm_acc_seg, global_iter_num)  
                writer.add_scalar('auc', auc, global_iter_num)
                writer.add_scalar('f1-score', f1, global_iter_num)
                writer.add_scalar('precision', precision, global_iter_num)
                writer.add_scalar('recall', recall, global_iter_num)
                writer.add_scalar('mcc', mcc, global_iter_num)
                model.train()

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            losses += loss.sum().item()
            # sss1 = time.time()  
            
            if idx % 2000 == 0:

                save_path = os.path.join(directory,f'lrdecay_cosine_{epoch}_{idx}.pt')

                model.eval()

                mlm_acc_seg, _, _ = accuracy(mlm_logits, b_mlm_label, voc['[PAD]'])
                auc = roc_auc(mlm_logits, b_mlm_label, voc['[PAD]'])
                matrix, f1, precision, recall, mcc= f1_precision_recall_mcc(mlm_logits, b_mlm_label, voc['[PAD]'])
                logging.info(f"Epoch: {epoch}, Batch[{idx}/{len(train_iter)}], "
                             f"Train loss :{loss.sum().item():.3f}, Train mlm acc: {mlm_acc_seg:.3f}, roc_auc: {auc:.3f}, f1: {f1}, precision: {precision}, recall: {recall}, mcc: {mcc}")  # 一轮的结果
                logging.info(f"\n{matrix}")
                end_time = time.time()
                logging.info(f"training time:{end_time-start_time}")
                start_time = end_time

        scheduler.step()
        logging.info(f"current learning rate{optimizer.param_groups[0]['lr']}")
        save_checkpoint(model, optimizer, epoch, losses, save_path, idx, scheduler,scaler)





def attn_to_figure(attn_weight, epoch):
    mat = attn_weight.sum(dim=0).detach().cpu()
    with suppress_stdout_stderr():
        plt.matshow(mat, cmap=plt.cm.hot, vmin=mat.min(), vmax=mat.max())
        plt.colorbar()
        plt.savefig(f"../figure/epoch_{epoch}.jpg")

def get_most_free_memory_gpu():
    device_count = pynvml.nvmlDeviceGetCount()
    max_free_memory = 0
    max_memory_gpu = 0
    
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        free_memory = mem_info.free
        print(f"GPU {i}: Total Memory: {mem_info.total / 1024**2} MB, Used Memory: {mem_info.used / 1024**2} MB, Free Memory: {free_memory / 1024**2} MB")

        if free_memory > max_free_memory:
            max_free_memory = free_memory
            max_memory_gpu = i
    
    return max_memory_gpu


def roc_auc(mlm_logits, mlm_labels, PAD_IDX):
    with torch.no_grad():
        mlm_pred = mlm_logits.reshape(-1, mlm_logits.shape[2])
        mlm_true = mlm_labels.reshape(-1)

        mask = torch.logical_not(mlm_true.eq(PAD_IDX))  

        mlm_pred = mlm_pred[mask,:2] 


        mlm_pred = mlm_pred.to(torch.float32)
        mlm_pred_sm = torch.softmax(mlm_pred, dim= 1).cpu()
        mlm_pred_sm = mlm_pred_sm[:,1:2]
        mlm_pred_sm = mlm_pred_sm.reshape(-1)
        mlm_true = mlm_true[mask]

        mlm_true = mlm_true.cpu()
        mlm_true = mlm_true.cpu()
        mlm_true = mlm_true.numpy()
        mlm_pred_sm = mlm_pred_sm.numpy()
        return roc_auc_score(y_true=mlm_true, y_score=mlm_pred_sm)


def f1_precision_recall_mcc(mlm_logits, mlm_labels, PAD_IDX):
    """
    :param mlm_logits:  [batch_size, src_len, vocab_size]
    :param mlm_labels:  [batch_size, src_len]
    :param PAD_IDX:
    :return:
    """
    mlm_pred = mlm_logits.argmax(axis=2).reshape(-1).cpu()
    mlm_true = mlm_labels.reshape(-1).cpu()
    mask = mlm_true != PAD_IDX
    

    mask = torch.logical_not(mlm_true.eq(PAD_IDX)).cpu().numpy()
    labels = list(range(mlm_logits.shape[-1]))
    matrix = confusion_matrix(mlm_true, mlm_pred, labels=labels, sample_weight=mask)
    TP = matrix[1][1]
    TN = matrix[0][0]
    FP = matrix[0][1]+matrix[0][2]+matrix[0][3]+matrix[0][4]+matrix[0][5]
    FN = matrix[1][0]+matrix[1][2]+matrix[1][3]+matrix[1][4]+matrix[1][5]
    
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = (2 * precision * recall) / (precision + recall)
    mcc = matthews_corrcoef(mlm_true, mlm_pred, sample_weight=mask)
    
    return matrix, f1, precision, recall, mcc,
    

def accuracy(mlm_logits, mlm_labels, PAD_IDX):
    """
    :param mlm_logits:  [batch_size, src_len, vocab_size]
    :param mlm_labels:  [batch_size, src_len]
    :param PAD_IDX:
    :return:
    """
    # print("mlm_logits",mlm_logits.shape)
    mlm_pred = mlm_logits.argmax(axis=2).reshape(-1)
    mlm_true = mlm_labels.reshape(-1)
    mlm_acc = mlm_pred.eq(mlm_true)  

    mask = torch.logical_not(mlm_true.eq(PAD_IDX))


    mlm_acc = mlm_acc.logical_and(mask) 
    mlm_correct = mlm_acc.sum().item()
    mlm_total = mask.sum().item()
    epi = 1e-10  
    mlm_acc = float(mlm_correct) / (mlm_total + epi)
    return (mlm_acc, mlm_correct, mlm_total)












def compute_metrics(tensorC, label,res):

    assert tensorC.shape == label.shape, "tensorC dismatch label shape"
    

    mask = (label != 3)
    

    tensorC = tensorC[mask]
    label = label[mask]
    

    TP = torch.sum((tensorC == 1) & (label == 1)).item()
    TN = torch.sum((tensorC == 0) & (label == 0)).item()
    FP = torch.sum((tensorC == 1) & (label == 0)).item()
    FN = torch.sum((tensorC == 0) & (label == 1)).item()

    return [x + y for x, y in zip(res, [TP, TN, FP, FN])]  
def extract(tensor_a, tensor_b, padidx,logit_in,label_in):


    mask = (tensor_b != padidx)

    logit = tensor_a[mask]
    label = tensor_b[mask]

    logit_list = logit.tolist()
    label_list = label.tolist()
    logit_out = logit_in+logit_list
    label_out = label_in+label_list

    return logit_out,label_out







def evaluate(config, data_iter, model,epoch, PAD_IDX, inference=False):

    res_ref = []
    res_label = []
    with torch.no_grad():
        model.eval()
        mlm_corrects, mlm_totals, auc, cnt = 0, 0, 0, 0
        total_idx = 0
        recall_total = 0
        sss1 = time.time()
        total_res = []
        mlm_logits_save = []
        mlm_labels_save = []
        logit85 = []
        label_highld = []
        logit4 = []
        logit10 = []
        for idx, (b_token_ids, b_mask, b_mlm_label, b_positions,b_LD_matrices,MAF_g,b_chunk_id) in enumerate(data_iter):


            b_token_ids = b_token_ids.to(config.device)

            # print(f'token{b_token_ids}')
            b_mask = b_mask.to(config.device)
            b_mlm_label = b_mlm_label.to(config.device)
            # b_LD_matrices = b_positions.to(config.device)
            # print(b_LD_matrices.shape)
            b_positions = b_positions.to(config.device)
            b_LD_matrices = b_LD_matrices.to(config.device)
            b_chunk_id = b_chunk_id.to(config.gpus[0])



            _, mlm_logits_85 = model(input_ids=b_token_ids,
                                ld_matrices = b_LD_matrices,
                                attention_mask=b_mask,
                                token_type_ids=None,
                                position_ids=b_positions,
                                masked_lm_labels=b_mlm_label,
                                next_sentence_labels=None,
                                chunk_id = b_chunk_id)

            mlm_logits_85 = mlm_logits_85.argmax(dim=-1)
            logit85.append(mlm_logits_85)
            label_highld.append(b_mlm_label)
            torch.save(logit85,f"./predict/{config.add_information}_epoch_{epoch}.pt")
            torch.save(label_highld,f".v/predict/{config.add_information}_epoch_{epoch}_label.pt")







        return (0,0,0,0,0,0,0)



def model_test(config):
    logging.info('test')
    start = time.time()

    voc = Vocab('./bert_1KGPMLM/vocab.txt')




    #chr22 data
    LD = ',/data/temp/Ld.pt'
    LD_data = torch.load(LD)
    ref = torch.load(',/data/temp/chr22_test.pt')
    val_nda = torch.load(',/chr22_ratiomask_testdata/20mask.pt')

    MAF = np.load('./stici_train_distribution.npy',allow_pickle=True)


    val_ds = randomDataset_torch_double_test(val_nda,
                                            ref=ref,
                                            LD_data=LD_data,
                                            hidden_size=config.hidden_size,
                                            MAF=MAF,
                                            max_position_embeddings=config.max_position_embeddings,
                                            masked_rate=config.masked_rate,
                                            vocab=voc,
                                            pad_index=voc['[PAD]'],
                                            step = config.max_position_embeddings,
                                            length_edge = config.length_edge)

    val_iter = DataLoader(dataset=val_ds,
                          batch_size=config.batch_size,
                          collate_fn=generate_batch,
                          drop_last=True,
                          shuffle=False,
                          num_workers=32,
                          pin_memory=True)



    positions_list = [val_ds.positions]
    print("position_list",type(val_ds.positions))
    print("list element type",type(positions_list[0][0]))



    all_possible_position = val_ds.positions.to(torch.long)

    config.all_possible_position = all_possible_position  # slow

    directory_path = config.test_dir

    config.device = config.gpus[0]

    


    model_names = []

    epoches = []
    for epoch in range(0,20):

        model_name = f"{epoch}_100000.pt"

        epoches.append(epoch)
        # model_names.append(model_name)
        print(model_name)
        model_names.append(os.path.join(directory_path,f'lrdecay_cosine_{model_name}'))

# 输出生成的模型名称列表




    model_count = 0
    step = -1
    for saved_model,epoch in zip(model_names,epoches):
        step += 1
        model = BertForMLM(config)
        # low_model = BertForMLM(config)
        # third_model = BertForMLM(config)
        model_path = saved_model
        logging.info(f"load model{model_path}")

        model = model.to(config.device)

        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)

        logging.info(f"## 成功载入已有模型{model_path}，进行推断......")

        model = nn.DataParallel(model, device_ids=config.gpus, output_device=config.gpus[0])

        time_load = time.time()
        mlm_acc, f1, precision, recall, mcc, matrix,auc = evaluate(config, val_iter, model,epoch,
                            voc['[PAD]'],inference=False)
    
        time_end = time.time()
        del model


        logging.info(
            f" ### MLM Accuracy on model: {model_path}")
        logging.info(
            f" ### MLM Accuracy on val: {mlm_acc:.6f}, f1: {f1:.4f}, precision: {precision:.4f}, recall: {recall:.4f}, mcc: {mcc},auc:{auc}")


        logging.info(f"\n{matrix}")

        logging.info(f'impute cost:{time.time()-start}')
    return



def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True



def int_list(string):
    return [int(x) for x in string.split(',')]



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="To do MLM task on 1KGP dataset.")
    parser.add_argument('-lr', '--learning_rate', type=float,
                        help='Masking rate of SNPs.', default=5e-5)
    parser.add_argument('-mr', '--masked_rate', type=float,
                        help='Masking rate of SNPs.', default=0.4)
    parser.add_argument('-tmr', '--test_masked_rate', type=float,
                        help='Masking rate of SNPs in testing set.', default=0.4)
    parser.add_argument('-i', '--inference', action='store_true',
                        help='To do inference on testing set with trained model.')
    # parser.add_argument('-trs', '--train_set', type=str,
    #                     help='train_set', default="simplify.train.recode.vcf")
    # parser.add_argument('-tes', '--test_set', type=str,
    #                     help='test_set', default="simplify.test.recode.vcf")
    parser.add_argument('-trs', '--train_set', type=str,
                        help='train_set', default="1kGP_high_coverage_Illumina.chr22.filtered.SNV_INDEL_SV_phased_panel_out_4_train_1.vcf")
    parser.add_argument('-tes', '--test_set', type=str,
                        help='test_set', default="1kGP_high_coverage_Illumina.chr22.filtered.SNV_INDEL_SV_phased_panel_out_4_common_4addSA_1.vcf")
    parser.add_argument('-g', '--gpu', type=str,
                        help='gpu', default=None)
    parser.add_argument('-bsz', '--batch_size', type=int,
                        help='batch_size', default=32)
    
    parser.add_argument('-m','--model_test',action='store_true')

    parser.add_argument('-s','--step',type=int,
                        help='Length of step  when select next segment to train',default=1024)
    parser.add_argument('-le','--length_edge',type=int,
                        help='length of edge when prediction masked sites on one side',default=0)
    parser.add_argument('-seed','--random_seed',type=int,
                        help='length of edge when prediction masked sites on one side',default=42)
    parser.add_argument('-re','--resume',action='store_true',
                        help='resume from the designed checkpoint')
    parser.add_argument('-ckp', '--checkpoint', type=str,
                        help='the path of the checkpoint')
    parser.add_argument('-add', '--add_information', type=str,default='',
                        help='add_information')
    parser.add_argument('-gpus', '--gpu_list', type=int_list,default=[0,1,2,3],
                        help='using gpus')
    parser.add_argument('-re_epoch', '--resume_epoch', type=int,default=0,
                        help='resume_from_epoch')
    parser.add_argument('-test_dir', '--test_dir', type=str,
                        help='test_dir', default='')
    args = parser.parse_args()
    if args.gpu is None:
        args.gpu = "1"

    if args.model_test == True:
        args.batch_size = 128

    config = ModelConfig(masked_rate=args.masked_rate,
                         learning_rate=args.learning_rate,
                         test_masked_rate=args.test_masked_rate,
                         train_set=args.train_set,
                         test_set=args.test_set,
                         gpu=args.gpu,
                         batch_size=args.batch_size,
                         step=args.step,
                         resume=args.resume,
                         checkpoint=args.checkpoint,
                         length_edge=args.length_edge,
                         add_information=args.add_information,
                         gpus = args.gpu_list,
                         epoch_resume = args.resume_epoch,
                         test_dir = args.test_dir)
    setup_seed(args.random_seed)

    if args.model_test:
        model_test(config)
        exit(1)
    print("123456789")
    if not args.inference:
        if not args.model_test:
            train(config)
