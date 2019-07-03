import re
import os

import logging
from pprint import pformat
from logging import handlers
import ujson as json

import torch
import numpy as np
import uuid

from text import torchtext

from tensorboardX import SummaryWriter

import arguments
import models
from validate import validate
from multiprocess import Multiprocess, DistributedDataParallel
from metrics import compute_metrics
from util import elapsed_time, get_splits, batch_fn, set_seed, preprocess_examples, get_trainable_params, count_params

def transform_single_v2(raw,task):
    return transform_single(raw,task,version=2)

def transform_single(raw,task,version=1):
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

    data = {
        "title":"dummy",
        "paragraphs":[]
    }

    content = {
      "context": None,
      "qas":[]
    }

    qas = {
      "id": uuid.uuid1().hex,
      "answers":[],
      "question": None
    }

    answers = {
      "answer_start": None,
      "text": None
    }

    raw.answer = list(map(lambda x:x.strip() ,raw.answer)) 
    raw.context = list(map(lambda x:x.strip() ,raw.context)) 
    answers["text"] = ' '.join(raw.answer).strip() 
    if task in ["multinli.in.out", "sst", "schema"]:
        raw.context_question = list(map(lambda x:x.strip() ,raw.context_question)) 
        content["context"] = ' '.join(raw.context_question).strip() 
    else:
        content["context"] = ' '.join(raw.context).strip() 
    print("text:", answers["text"])
    print("context:", content["context"])
    print(answers["text"] in content["context"])
    print("------------")

    if version == 1:
        if answers["text"] == "unanswerable": return
        substitute(answers,content)
        qas["answers"].append(answers)
    elif version == 2:
        if answers["text"] == "unanswerable":
            qas["is_impossible"] = True
        else:
            qas["is_impossible"] = False 
            substitute(answers,content)
            qas["answers"].append(answers)

    qas["question"] = ''.join(raw.question).strip()
    content["qas"].append(qas)
    data["paragraphs"].append(content)
    return data

def substitute(answers,content):
    subs = [",","and","or","to","/"]
    while len(subs) > 0:
        try:
            answers["answer_start"] = re.search(re.escape(answers["text"][:10]), content["context"]).start()
        except AttributeError:
            prev_sub = subs.pop(0)
            try:
                answers["text"] = re.sub(prev_sub,subs[0],answers["text"])
            except IndexError:
                return
            continue
        break

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

    #if args.load is None:
    #    logger.info(f'Getting pretrained word vectors')
    #    char_vectors = torchtext.vocab.CharNGram(cache=args.embeddings)
    #    glove_vectors = torchtext.vocab.GloVe(cache=args.embeddings)
    #    vectors = [char_vectors, glove_vectors]
    #    vocab_sets = (train_sets + val_sets) if len(vocab_sets) == 0 else vocab_sets
    #    logger.info(f'Building vocabulary')
    #    FIELD.build_vocab(*vocab_sets, max_size=args.max_effective_vocab, vectors=vectors)

    #FIELD.decoder_itos = FIELD.vocab.itos[:args.max_generative_vocab]
    #FIELD.decoder_stoi = {word: idx for idx, word in enumerate(FIELD.decoder_itos)}
    #FIELD.decoder_to_vocab = {idx: FIELD.vocab.stoi[word] for idx, word in enumerate(FIELD.decoder_itos)}
    #FIELD.vocab_to_decoder = {idx: FIELD.decoder_stoi[word] for idx, word in enumerate(FIELD.vocab.itos) if word in FIELD.decoder_stoi}

    #logger.info(f'Vocabulary has {len(FIELD.vocab)} tokens')
    #logger.info(f'The first 500 tokens:')
    #print(FIELD.vocab.itos[:500])

    #logger.info('Preprocessing training data')
    #preprocess_examples(args, args.train_tasks, train_sets, FIELD, logger, train=True)
    #logger.info('Preprocessing validation data')
    #preprocess_examples(args, args.val_tasks, val_sets, FIELD, logger, train=args.val_filter)

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
    tasks = ["multinli.in.out", "sst", "srl", "zre", "wikisql", "schema"]
    args = arguments.parse()
    for task in tasks: 
        args.train_tasks = [task]
        args.val_tasks = [task]
        train_sets, val_sets = get_data(args)
        train_sets = train_sets[0]
        train_sets = list(map(lambda x: transform_single(x,task,version=args.version), train_sets))
        train_sets = list(filter(lambda x: x is not None, train_sets))
        train_sets = {"data": train_sets}
        val_sets = val_sets[0]
        val_sets = list(map(lambda x: transform_single(x,task,version=args.version), val_sets))
        val_sets = list(filter(lambda x: x is not None, val_sets))
        val_sets = {"data": val_sets}
        with open("clean_data/%s_to_squad-dev-v1.1.json"%task,"w") as f:
            json.dump(val_sets,f)
        with open("clean_data/%s_to_squad-train-v1.1.json"%task,"w") as f:
            json.dump(train_sets,f)
