import logging
import random
import torch
import numpy as np
import time
from torch.optim import AdamW
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from transformers import get_linear_schedule_with_warmup


def setup_device(args):
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.n_gpu = torch.cuda.device_count()


def setup_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def setup_logging(key_word: str = 'test'):
    log_f_name = './log/' + f"{key_word}_{time.strftime('%m%d', time.localtime())}.log"  # %m%d_%H%M

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[
                            # logging.FileHandler(log_f_name),
                            logging.StreamHandler()
                        ]
                        )
    logger = logging.getLogger(__name__)

    return logger


def build_optimizer(args, model):
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                                num_training_steps=args.max_steps)
    return optimizer, scheduler


def build_optimizer_vlbert(args, model):  # , mode='pretrain'
    # Prepare optimizer and schedule (linear warmup and decay)
    # for n, p in model.named_parameters():
    #     print(n)
    lr_dict = {'visual_backbone': args.pretrain_model_lr, 'bert': args.pretrain_model_lr, 'others': args.other_model_lr}
    no_decay = ['bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = []
    for layer_name in lr_dict:
        lr = lr_dict[layer_name]
        if layer_name != 'others':
            optimizer_grouped_parameters += [
                {'params': [p for n, p in model.named_parameters() if (not any(nd in n for nd in no_decay)
                                                                       and layer_name in n)],
                 'weight_decay': args.weight_decay,
                 'lr': lr},
                {'params': [p for n, p in model.named_parameters() if (any(nd in n for nd in no_decay)
                                                                       and layer_name in n)],
                 'weight_decay': 0.0,
                 'lr': lr}
            ]
        else:
            optimizer_grouped_parameters += [
                {'params': [p for n, p in model.named_parameters() if (not any(nd in n for nd in no_decay)
                                                                       and not any(name in n for name in lr_dict))],
                 'weight_decay': args.weight_decay,
                 'lr': lr},
                {'params': [p for n, p in model.named_parameters() if (any(nd in n for nd in no_decay)
                                                                       and not any(name in n for name in lr_dict))],
                 'weight_decay': 0.0,
                 'lr': lr}
            ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=lr_dict['bert'], eps=args.adam_epsilon)

    def _invsqrt_lr(step):
        return np.sqrt(args.warmup_steps) * 2 / np.sqrt(max(1, step))

    def _warmup_lr(step):
        return step / args.warmup_steps

    def _invert_lr_with_warmup(step):
        return _warmup_lr(step) if step < args.warmup_steps else _invsqrt_lr(step)

    # if mode == 'pretrain':
        # scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                    # num_training_steps=args.max_steps)
    # else:
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_invert_lr_with_warmup)

    # swa_scheduler = torch.optim.swa_utils.SWALR(optimizer, swa_lr=args.minimum_lr)
    return optimizer, scheduler


def evaluate(prediction, label, raw_prediction):
    '''
    prediction: [bs, 25]  0 1
    label: [bs, 25] 0 1
    '''
    
    prediction = np.vstack(prediction)
    label = np.vstack(label)
    raw_prediction = np.vstack(raw_prediction)
    
    macro_per_cls = f1_score(label, prediction, average=None)
    macro_list = [round(i, 4) for i in macro_per_cls]

    metrics = {"macro_f1": round(f1_score(label, prediction, average="macro"), 6),
               "micro_f1": round(f1_score(label, prediction, average="micro"), 6),
               "weighted_f1": round(f1_score(label, prediction, average="weighted"), 6),
               "macro_list": macro_list,
               "auc_pr_macro": round(average_precision_score(label, raw_prediction, average="macro"), 6),
               "auc_pr_micro": round(average_precision_score(label, raw_prediction, average="micro"), 6),
               "auc_macro": round(roc_auc_score(label, raw_prediction, average="macro"), 6),
               "auc_micro": round(roc_auc_score(label, raw_prediction, average="micro"), 6)
               }

    return metrics


