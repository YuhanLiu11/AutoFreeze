# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import queue
import torch.multiprocessing as mp
import multiprocessing

import csv
import os
import logging
import argparse
import random
from tqdm import tqdm, trange
import pandas as pd
import math
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torch.utils.data.sampler import RandomSampler, SequentialSampler, SubsetRandomSampler
from torch.utils.data.distributed import DistributedSampler
import time
import tokenization
from modeling_caching import BertConfig, BertForSequenceClassification, BertForSequenceClassificationTest
from optimization_lr import BERTAdam
import json
import statistics 
import gc
import copy
from utils import *
os.environ["LRU_CACHE_CAPACITY"] = "3"
os.environ["OMP_NUM_THREADS"] = "1"
mp.set_sharing_strategy('file_system')
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', 
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    #TODO: Do we need deterministic in cudnn ? Double check
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # print ("Seeded everything")

global_step = 0 
nb_tr_examples = 0
nb_tr_steps = 0
tr_loss = 0         
first_caching = {}
all_length = []

def eval_model(model, args, eval_dataloader, device, epoch, tr_loss, nb_tr_steps, eval_step):
    model.eval()
    eval_step = int(eval_step)
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    with open(os.path.join(args.output_dir, "results_ep_"+str(epoch)+".txt"),"w") as f:
        for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)

            with torch.no_grad():
                (tmp_eval_loss, logits), _ = model(input_ids, segment_ids, input_mask, label_ids)

            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()
            outputs = np.argmax(logits, axis=1)
            for output in outputs:
                f.write(str(output)+"\n")
            tmp_eval_accuracy=np.sum(outputs == label_ids)

            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy

            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_examples

    result = {'eval_loss': eval_loss,
              'eval_accuracy': eval_accuracy,
              'global_step': global_step,
              'loss': tr_loss/nb_tr_steps}
    
    output_eval_file = os.path.join(args.output_dir, "eval_results_ep_"+str(eval_step)+".txt")
    print("output_eval_file=",output_eval_file)
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))       
def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs==labels)

lengths = []
def frozen_model(bert_config, label_list, args, global_step, num_train_steps, device):
    """ 
    """
    no_decay = ['bias', 'gamma', 'beta']
    new_model = BertForSequenceClassification(bert_config, len(label_list))
    new_optimizer_parameters = [
                {'params': [p for n, p in new_model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
                {'params': [p for n, p in new_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
                ]
    
    new_optimizer = BERTAdam(new_optimizer_parameters,
                            lr=args.learning_rate, 
                            warmup=args.warmup_proportion,
                            t_total=num_train_steps,
                            decay_factor=args.decay_factor)
    torch.cuda.empty_cache()
    new_model.load_state_dict(torch.load(args.output_dir + "/checkpoint-{}.pth.tar".format(int(global_step / args.grad_eval_step)))['state_dict'])
    new_model.to(device)
    model = new_model
    optimizer = new_optimizer
    return model, optimizer


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_config_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The config json file corresponding to the pre-trained BERT model. \n"
                             "This specifies the model architecture.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--vocab_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The vocabulary file that the BERT model was trained on.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--init_checkpoint",
                        default=None,
                        type=str,
                        help="Initial checkpoint (usually from a pre-trained BERT model).")
    parser.add_argument("--do_lower_case",
                        default=False,
                        action='store_true',
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        default=False,
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--discr",
                        default=False,
                        action='store_true',
                        help="Whether to do discriminative fine-tuning.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--save_checkpoints_steps",
                        default=1000,
                        type=int,
                        help="How often to save the model checkpoint.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--accumulate_gradients",
                        type=int,
                        default=1,
                        help="Number of steps to accumulate gradient on (divide the batch_size and accumulate)")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', 
                        type=int, 
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumualte before performing a backward/update pass.")
    
    parser.add_argument('--grad_eval_step',
                        type=int,
                        default=500,
                        help="Number of iterations that evaluation.")  
    parser.add_argument('--num_datas',
                        type=int,
                        default=None,
                        help="Number of data points.")   
    parser.add_argument('--percentile',
                        type=int,
                        default=50,
                        help="Percentile for freezing. ")
    parser.add_argument('--random_seeds',
                        type=str,
                        default=None,
                        help="Random seeds for each training epoch, separated by comma.")
    parser.add_argument('--decay_factor',
                        type=int,
                        default=10,
                        help="Learning rate decay factor") 
    parser.add_argument('--num_intervals',
                        type=int,
                        default=10,
                        help="Number of evaluation intervals.")
    args = parser.parse_args()

    processors = {
        "ag": AGNewsProcessor,
        "ag_sep": AGNewsProcessor_sep,
        "ag_sep_aug": AGNewsProcessor_sep_aug,
        "imdb": IMDBProcessor,
        "imdb_sep": IMDBProcessor_sep,
        "imdb_sep_aug": IMDBProcessor_sep_aug,
        "yelp_p": Yelp_p_Processor,
        "yelp_f": Yelp_f_Processor,
        "yahoo": Yahoo_Processor,
        "trec": Trec_Processor,
        "dbpedia":Dbpedia_Processor,
        "mrpc": MrpcProcessor,
        "sogou": Sogou_Processor
    }

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device %s n_gpu %d distributed training %r", device, n_gpu, bool(args.local_rank != -1))

    if args.accumulate_gradients < 1:
        raise ValueError("Invalid accumulate_gradients parameter: {}, should be >= 1".format(
                            args.accumulate_gradients))

    args.train_batch_size = int(args.train_batch_size / args.accumulate_gradients)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    bert_config = BertConfig.from_json_file(args.bert_config_file)

    if args.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length {} because the BERT model was only trained up to sequence length {}".format(
            args.max_seq_length, bert_config.max_position_embeddings))

    # if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
    #     raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    label_list = processor.get_labels()

    tokenizer = tokenization.FullTokenizer(
        vocab_file=args.vocab_file, do_lower_case=args.do_lower_case)

    train_examples = None
    num_train_steps = None
    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir, data_num = args.num_datas)
        num_train_steps = int(
            len(train_examples) / args.train_batch_size * args.num_train_epochs)
    set_seed(0)
    model = BertForSequenceClassification(bert_config, len(label_list))
    if args.init_checkpoint is not None:
        model.bert.load_state_dict(torch.load(args.init_checkpoint, map_location='cpu'))
        
    model.to(device)
    
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    no_decay = ['bias', 'gamma', 'beta']
    if args.discr:
        group1=['layer.0.','layer.1.','layer.2.','layer.3.']
        group2=['layer.4.','layer.5.','layer.6.','layer.7.']
        group3=['layer.8.','layer.9.','layer.10.','layer.11.']
        group_all=['layer.0.','layer.1.','layer.2.','layer.3.','layer.4.','layer.5.','layer.6.','layer.7.','layer.8.','layer.9.','layer.10.','layer.11.']
        optimizer_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and not any(nd in n for nd in group_all)],'weight_decay_rate': 0.01},
            
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and any(nd in n for nd in group1)],'weight_decay_rate': 0.01, 'lr': args.learning_rate/2.6},
            
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and any(nd in n for nd in group2)],'weight_decay_rate': 0.01, 'lr': args.learning_rate},
            
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and any(nd in n for nd in group3)],'weight_decay_rate': 0.01, 'lr': args.learning_rate*2.6},
            
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and not any(nd in n for nd in group_all)],'weight_decay_rate': 0.0},
            
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and any(nd in n for nd in group1)],'weight_decay_rate': 0.0, 'lr': args.learning_rate/2.6},
            
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and any(nd in n for nd in group2)],'weight_decay_rate': 0.0, 'lr': args.learning_rate},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and any(nd in n for nd in group3)],'weight_decay_rate': 0.0, 'lr': args.learning_rate*2.6},
        ]
    else:
        optimizer_parameters = [
             {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
             {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
             ]
		
    optimizer = BERTAdam(optimizer_parameters,
                         lr=args.learning_rate,
                         warmup=args.warmup_proportion,
                         t_total=num_train_steps,
                         decay_factor=args.decay_factor)
	
    eval_examples = processor.get_dev_examples(args.data_dir)
    eval_features = convert_examples_to_features(
        eval_examples, label_list, args.max_seq_length, tokenizer)

    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)

    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    eval_dataloader = DataLoader(eval_data, batch_size=args.eval_batch_size, shuffle=False, num_workers=0)
    global global_step
    global nb_tr_examples, nb_tr_steps, tr_loss    
    if args.do_train:
        train_features = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)
        print(len(train_features))
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        set_seed(0)
        train_dataloader = DataLoader(train_data, sampler = train_sampler, batch_size=args.train_batch_size, num_workers=0)
        orig_dataloader = train_dataloader # to be used in cache_data
        old_param_dict = {}
        model.eval()
        for i in range(12):
            old_param_dict[i] = []
        
        grad_dict = None
        all_dicts = []
        start_layer = 0
        prev_intermediate_grad_dict = None
        prev_min_trainable_layer = 0
        grad_tensor_dict = {} # gradient accumulator
        cached_data = False
        for name, param in model.bert.named_parameters():
            grad_tensor_dict[name] = torch.zeros(param.shape).to(device)
        set_seed(0)

        all_indices = []
        
        
        torch.multiprocessing.set_start_method('spawn', force=True)


        prev_num_files = 0
        
        train_indices = [] # List of dictionary storing the new index after shuffling -> original index mapping
        train_indices_orig = [] 
        seeds = [int(seed) for seed in args.random_seeds.split(',')]
        if len(seeds) != int(args.num_train_epochs):
            raise ValueError("Length of random seeds must equal to number of training epochs. ")  
        for i in range(int(args.num_train_epochs)):
            set_seed(seeds[i])
            tmp_indices = [i for i in range(len(train_examples))]
            shuffle_to_original = {}
            original_to_shuffle = {}
            train_indices_dataloader = DataLoader(tmp_indices, sampler=RandomSampler(tmp_indices), num_workers=0)
            for step, batch in enumerate(tqdm(train_indices_dataloader)):
                shuffle_to_original[step] = batch[0].item()
                original_to_shuffle[batch[0].item()] = step
            train_indices.append(shuffle_to_original)
            train_indices_orig.append(original_to_shuffle)
        epoch = 0
        start_layer = 0
        switch_interval = 0
        unmodified_layer = False # Set to True if the number of frozen layers haven't changed
        current_loading_epoch = None # The last epoch which we write new intermediate output data at
        args.grad_eval_step = int(args.num_train_epochs * len(train_dataloader)* (1/args.num_intervals)) 
        epoch_layer = 0 # Record the layer at the beginning of an epoch
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            epoch+=1
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            
            logger.info("*********Start layer*********")
            print(start_layer)            
            loaded_bert = [] # store the intermediate output from disk
        
            start_time = time.time()
            manager = multiprocessing.Manager()
            data_queue = mp.Queue() # queue to save intermediate outputs to be written to disk
            prev_min_trainable_layer = epoch_layer
            epoch_layer = start_layer # The layer to start caching at the current epoch
            print("EPOCH layer is: ", epoch_layer)

            set_seed(seeds[epoch-1])
            
            logger.info("START TRAINING AT {}".format(epoch))
            current_dict = {}
            last_time = time.time()
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                
                batch = tuple(t.to(device) for t in batch)
                bert_last = None
                model.train()
                
                input_ids, input_mask, segment_ids, label_ids = batch
                bert_last = None
                
                # if epoch >= 3, we could start loading the hidden states for training from new_cached_queue (poping from the queue)
                if epoch >= 3 and prev_min_trainable_layer > 0:
                    
                    bert_last = torch.zeros(len(input_ids), 512, 768, device=torch.device("cuda"))
                    for i in range(len(input_ids)):
                        # x = torch.zeros((1, 512, 768))
                        x = prev_dict[step*args.train_batch_size+i]
                        bert_last[i,:, :] = x.to(device)
                        del prev_dict[step*args.train_batch_size+i]
                
                
                (loss,_),hidden_states = model(input_ids, segment_ids, input_mask, label_ids, start_layer=start_layer, prev_min_trainable_layer = prev_min_trainable_layer, caching_training=True, bert_last = bert_last)
                # if epoch >= 2, Save the hidden states  to data_queue (put to the queue)
                if epoch >= 2 and epoch < args.num_train_epochs and epoch_layer > 0:
                    
                    for i in range(len(input_ids)):
                        translated_index = train_indices_orig[epoch][train_indices[epoch-1][step*args.train_batch_size+i]]
                        # first_caching[repr(input_ids[i].cpu().detach().numpy())] = hidden_states[start_layer - 1][i].cpu().detach().reshape((1, 512, 768)).numpy()
                        if prev_min_trainable_layer > 0:

                            if epoch_layer != prev_min_trainable_layer:
                                current_dict[translated_index] = hidden_states[epoch_layer-prev_min_trainable_layer-1][i].cpu()
                                
                            else:
                                current_dict[translated_index] = bert_last[i].cpu()
                        elif epoch_layer > 0:
                            current_dict[translated_index] = hidden_states[epoch_layer - 1][i].cpu()
                            # data_queue.put((step*args.train_batch_size+i, activation['bert.encoder.layer.11.output'][i]))
                del bert_last
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                loss.backward()
                
                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                for name, param in model.bert.named_parameters():
                    param_list = name.split(".")
                    layer_num = 0
                    if param.grad is not None:
                        
                        if name not in grad_tensor_dict.keys():
                            grad_tensor_dict[name] = param.grad
                        else:
                            grad_tensor_dict[name] += param.grad
                current_grad_dict = {}
                for i in range(12):
                    current_grad_dict[i] = 0
                
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step(current_step = global_step)    # We have accumulated enought gradients
        
                    model.zero_grad()
                if (global_step + 1) % args.grad_eval_step == 0 and global_step / args.grad_eval_step > 0:
                
                    logger.info("Start evaluating!")
                    # Saving checkpoints
                    state = {
                    'epoch': epoch+1,
                    'state_dict': model.state_dict(),
                    }
                    torch.save(state, args.output_dir + "/checkpoint-{}.pth.tar".format(int(global_step / args.grad_eval_step))) 
                    eval_model(model, args, eval_dataloader, device, epoch, tr_loss, nb_tr_steps, global_step / args.grad_eval_step)
                    # Calculate gradient changing ratio
                    for name in grad_tensor_dict.keys():
                        param_list = name.split(".")
                        layer_num = 0
                        for split_param in param_list:
                            try:
                                layer_num = int(split_param)
                                if "encoder" in name:
                                    current_grad_dict[layer_num] += torch.norm(grad_tensor_dict[name].cpu().detach(), p=1).item() 
                            except ValueError:
                                pass
                        
                    print(current_grad_dict)                    
                    print("grad dict", current_grad_dict) 
                    
                    # Clear gradient accumulator
                    grad_tensor_dict = {}
                    for name, param in model.bert.named_parameters():
                        grad_tensor_dict[name] = torch.zeros(param.shape).to(device)

                    if prev_intermediate_grad_dict is None:
                        # Set gradient dict to be compared with for the first time
                        prev_intermediate_grad_dict = current_grad_dict
                    else:
                        threshold_dict = {}
                        for key in range(12):
                            threshold_dict[key] = 0
                        # Calculate gradient changing threshold
                        for key in current_grad_dict.keys() :
                            if current_grad_dict[key] > 0:
                                threshold_dict[key] = abs(prev_intermediate_grad_dict[key] - current_grad_dict[key]) / prev_intermediate_grad_dict[key]    
                            
                        median_value = np.percentile(list(threshold_dict.values())[start_layer:], args.percentile)   
                        # Find out the first layer with ratio ge to the median value      
                        for key in threshold_dict.keys():
                            if threshold_dict[key] >= median_value:
                                start_layer = key
                                break
                        prev_intermediate_grad_dict = current_grad_dict
                        print("threshold: ", threshold_dict)
                        print("layer num: ", start_layer)
                        if start_layer > 0 :
                            # New optimizer 
                            
                            model, optimizer = frozen_model(bert_config, label_list, args, global_step, num_train_steps, device)
                        logger.info("TRAINING FROM {}".format(str(start_layer)))
                global_step += 1
            
            prev_dict = current_dict


if __name__ == "__main__":
    main()