import re
import os

import logging
from pprint import pformat
from logging import handlers
import ujson as json

import torch
import numpy as np
import pandas as pd
from functools import reduce
from copy import deepcopy
import uuid

from text import torchtext

from tensorboardX import SummaryWriter

import arguments
import models
from validate import validate
from multiprocess import Multiprocess, DistributedDataParallel
from metrics import compute_metrics
from util import elapsed_time, get_splits, batch_fn, set_seed, preprocess_examples, get_trainable_params, count_params

def get_single(raw,global_dict):
    uid = uuid.uuid1().hex
    raw.answer = list(map(lambda x:x.strip() ,raw.answer))
    raw.context = list(map(lambda x:x.strip() ,raw.context))
    context = ' '.join(raw.context).strip()
    question = ''.join(raw.question).strip()
    answer = ' '.join(raw.answer).strip()
    #print("context:", context)
    #print("question:", question)
    #print("answer:", answer)
    #print("------------")
    if context in global_dict:
        global_dict[context].append({"id":uid,"answers":[{"answer_start": None,"text":answer}],"question":question})
    else:
        global_dict[context] = [{"id":uid,"answers":[{"answer_start": None,"text":answer}],"question":question}]
    return

def transform_single_v2(row,version=2):
    return transform_single(row,version)

def transform_single(row,version=1):

    '''
    v1:
    data = {
        "title":"dummy",
        "paragraphs":[{
          "context": None,
          "qas":[{
            "id": uuid.uuid1(),
            "answers":[{
              "answer_start": None,
              "text": None
            }],
            "question": None
          }]
        }],
    }

    v2 (is_impossible: false):
    data = {
        "title":"dummy",
        "paragraphs":[{
          "context": None,
          "qas":[{
            "id": uuid.uuid1(),
            "answers":[{
              "answer_start": None,
              "text": None
            }],
            "question": None,
            "is_impossible": False
          }]
        }],
    }

    v2 (is_impossible: true):
    data = {
        "title":"dummy",
        "paragraphs":[{
          "context": None,
          "qas":[{
            "id": uuid.uuid1(),
            "answers":[{
              "answer_start": None,
              "text": None
            }],
            "question": None,
            "is_impossible": False,
            "plausible_answers":[{
              "text": "Normans",
              "answer_start": 4
            }],
          }]
        }],
    }
    '''
    '''
    row = (  
          context,
          [
              {'id': uid,
               'answers': [{"answer_start": None,"text":answer},...],
               'question': question},
              {'id': uid,
               'answers': [{"answer_start": None,"text":answer},...],
               'question': question},
              ...
          ]
    ) 
    '''

    context, qas = row
    if version == 2:
        for qa in qas:
            answers = qa["answers"]
            for ans in answers:
                if ans["text"] == "unanswerable":
                    qa["is_impossible"] = True
                else:
                    qa["is_impossible"] = False

    content = {
      "context": context,
      "qas": qas
    }
    data = {
        "title":"dummy",
        "paragraphs":[content]
    }

    return data

def initialize_logger(args, rank='main'):
    # set up file logger
    logger = logging.getLogger(f'process_{rank}')
    logger.setLevel(logging.DEBUG)
    handler = handlers.RotatingFileHandler(os.path.join(args.log_dir, f'process_{rank}.log'), maxBytes=1024*1024*10, backupCount=1)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(name)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.propagate = False
    return logger

def prepare_data(args, field, logger):
    if field is None:
        logger.info(f'Constructing field')
        FIELD = torchtext.data.ReversibleField(batch_first=True, init_token='<init>', eos_token='<eos>', lower=args.lower, include_lengths=True)
    else:
        FIELD = field

    train_sets, val_sets, vocab_sets = [], [], []
    for task in args.train_tasks:
        logger.info(f'Loading {task}')
        kwargs = {'test': None}
        kwargs['subsample'] = args.subsample
        kwargs['validation'] = None
        logger.info(f'Adding {task} to training datasets')
        split = get_splits(args, task, FIELD, **kwargs)[0]
        logger.info(f'{task} has {len(split)} training examples')
        train_sets.append(split)
        if args.vocab_tasks is not None and task in args.vocab_tasks:
            vocab_sets.extend(split)

    for task in args.val_tasks:
        logger.info(f'Loading {task}')
        kwargs = {'test': None}
        kwargs['subsample'] = args.subsample
        kwargs['train'] = None
        logger.info(f'Adding {task} to validation datasets')
        split = get_splits(args, task, FIELD, **kwargs)[0]
        logger.info(f'{task} has {len(split)} validation examples')
        val_sets.append(split)
        if args.vocab_tasks is not None and task in args.vocab_tasks:
            vocab_sets.extend(split)
    return train_sets, val_sets

def get_data(args):
    if args is None:
        return
    set_seed(args)
    logger = initialize_logger(args)
    logger.info(f'Arguments:\n{pformat(vars(args))}')

    field, save_dict = None, None
    if args.load is not None:
        logger.info(f'Loading field from {os.path.join(args.save, args.load)}')
        save_dict = torch.load(os.path.join(args.save, args.load))
        field = save_dict['field']
    train_sets, val_sets = prepare_data(args, field, logger)
    return train_sets, val_sets

if __name__ == '__main__':
    tasks = ["squad", "iwslt.en.de", "cnn_dailymail", "multinli.in.out", "sst", "srl", "zre", "woz.en", "wikisql", "schema"]
    tasks = ["iwslt.en.de", "cnn_dailymail", "multinli.in.out", "sst", "srl", "zre", "woz.en", "wikisql", "schema"]
    args = arguments.parse()
    v = "1.1" if args.version == 1 else "2.0"
    for task in tasks:
        train_dict = {}
        val_dict = {}
        args.train_tasks = [task]
        args.val_tasks = [task]
        train_sets, val_sets = get_data(args)
        train_sets = train_sets[0]
        list(map(lambda x: get_single(x,train_dict), train_sets))
        train_sets = map(lambda x: transform_single(x,args.version), list(train_dict.items()))
        train_sets = list(filter(lambda x: x is not None, train_sets))
        train_sets = {"data": train_sets}
        val_sets = val_sets[0]
        list(map(lambda x: get_single(x,val_dict), val_sets))
        val_sets = map(lambda x: transform_single(x,args.version), list(val_dict.items()))
        val_sets = list(filter(lambda x: x is not None, val_sets))
        val_sets = {"data": val_sets}
        with open("lll_data/%s_to_squad-dev-v%s.json"%(v,task),"w") as f:
            json.dump(val_sets,f)
        with open("lll_data/%s_to_squad-train-v%s.json"%(v,task),"w") as f:
            json.dump(train_sets,f)
