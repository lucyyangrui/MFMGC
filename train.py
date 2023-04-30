import json
import logging
import time

import numpy as np
import pandas as pd
import torch
from utils import evaluate, setup_logging, setup_seed, setup_device, build_optimizer, build_optimizer_vlbert
import tempfile
from sklearn.model_selection import KFold

from config import parse_args
from data_helper import create_dataloader, create_dataloader_all
from models.CTT_base1.model import Base1CttMmc
from models.GMU_base2.model import GMU_MGC
from models.base3.model import Base3Moviescope
from models.MFMGC.model import MFMGC
from models.base4.model import Base4
from models.base5.model import Base5
from models.base6.model import Base6


def validation(model, val_dataloader):
    model.eval()
    losses = []
    predictions_01 = []
    raw_predictions = []
    labels = []
    hidden_matrix = []
    with torch.no_grad():
        t_begin = time.time()
        for step, batch in enumerate(val_dataloader):
            loss, accuracy, pred_label_id, raw_prediction, hidden = model(batch)

            loss = loss.mean()
            accuracy = accuracy.mean()
            predictions_01.extend(pred_label_id.cpu().numpy())
            raw_predictions.extend(raw_prediction.cpu().numpy())
            labels.extend(batch['label'].cpu().numpy())
            # hidden_matrix.extend(hidden.cpu().numpy())
            losses.append(loss)
            if step % 5 == 0:
                t = time.time()
                logging.info(f"step={step:4}/{len(val_dataloader)}|"
                             f"loss={loss:6.4}|acc={accuracy:0.4}|time={t - t_begin:.4}s")
                t_begin = time.time()

    loss = sum(losses) / len(losses)
    metrics = evaluate(predictions_01, labels, raw_predictions)
    return loss, metrics, hidden_matrix


def test_model(args, test_dataloader, model_save, dataloader):
    logging.info('>>> loading model...')
    if args.model_name == 'base1':
        model = Base1CttMmc(args)
    elif args.model_name == 'mymodel':
        model = MFMGC(args)
    elif args.model_name == 'base2':
        model = GMU_MGC(args)
    elif args.model_name == 'base3':
        model = Base3Moviescope(args)
    elif args.model_name == 'base4':
        model = Base4(args)
    elif args.model_name == 'base5':
        model = Base5(args)
    elif args.model_name == 'base6':
        model = Base6(args)
    model_save.seek(0)
    model_dict = torch.load(model_save, map_location='cpu')
    model.load_state_dict(model_dict[0])
    
    if args.device == 'cuda':
        model = torch.nn.parallel.DataParallel(model.to(args.device))
        print('>> using gpu to loading model...')
    
    val_loss, metrics, _ = validation(model, test_dataloader)
    logging.info(f">>> val_loss: {val_loss}, matrics: {metrics}")
    hidden_matrix = []
    if args.model_name == 'base6':
        with torch.no_grad():
            for step, batch in enumerate(dataloader):
                loss, accuracy, pred_label_id, raw_prediction, hidden = model(batch)
                hidden_matrix.extend(hidden.cpu().numpy())

        hidden_matrix = pd.DataFrame(hidden_matrix)
        print(hidden_matrix.shape)
        hidden_matrix.to_csv('./' + args.modals[0] + '-features-mb.csv', header=False, index=False)
    return metrics


def train_and_validation(args):
    skf = KFold(n_splits=5, random_state=42, shuffle=True)
    labels = pd.read_csv(args.label_path, header=None, index_col=None).values
    args.data_num = len(labels)
    note_log = dict(
        note='-'.join(args.modals), # args.note,
        best_epoch=[0 for _ in range(5)],
        macro=[],
        micro=[],
        weighted_f1=[],
        auc_pr_macro=[],
        auc_pr_micro=[],
        auc_macro=[],
        auc_micro=[],
    )
    for i, (train_val_index, test_index) in enumerate(skf.split(range(args.data_num), labels)):
        if i != 0:
            continue
        print('==================== %d fold ====================' % i)
        train_index = train_val_index[: int(len(train_val_index) * 7 / 8)]
        val_index = train_val_index[int(len(train_val_index) * 7 / 8):]
        # train_index, val_index, test_index = None, None, None
        train_dataloader, val_dataloader, test_dataloader, _, _, _ = create_dataloader(args, train_index, val_index, test_index)
        dataloader = create_dataloader_all(args)
        if args.model_name == 'base1':
            model = Base1CttMmc(args)
            optimizer, schedual = build_optimizer(args, model)
        elif args.model_name == 'mymodel':
            model = MFMGC(args)
            optimizer, schedual = build_optimizer_vlbert(args, model)
        elif args.model_name == 'base2':
            model = GMU_MGC(args)
            optimizer, schedual = build_optimizer(args, model)
        elif args.model_name == 'base3':
            model = Base3Moviescope(args)
            optimizer, schedual = build_optimizer(args, model)
        elif args.model_name == 'base4':
            model = Base4(args)
            optimizer, schedual = build_optimizer(args, model)
        elif args.model_name == 'base5':
            model = Base5(args)
            optimizer, schedual = build_optimizer(args, model)
        elif args.model_name == 'base6':
            model = Base6(args)
            optimizer, schedual = build_optimizer(args, model)

        # model_dict = torch.load('./models/CCT_MMC_base1/save/model.bin', map_location='cpu')
        # model.load_state_dict(model_dict['model_state_dict'])
        if args.device == 'cuda':
            model = torch.nn.parallel.DataParallel(model.to(args.device))
            print('>> using gpu to loading model...')

        micro_begin = 0.0
        no_decay_epoch = 0

        logging.info('>>> begin training...')
        model_save = tempfile.TemporaryFile()
        for epoch in range(100):
            t_begin = time.time()
            t_log = t_begin
            train_loss = []
            for step, batch in enumerate(train_dataloader):
                model.train()
                optimizer.zero_grad()

                loss, accuracy, _, _, _ = model(batch)
                loss = loss.mean()
                accuracy = accuracy.mean()
                loss.backward()

                optimizer.step()
                if args.model_name == 'mymodel':
                    schedual.step()

                train_loss.append(loss)

                elap_t = time.time() - t_begin
                if step == 2 or step and step % 10 == 0:
                    lr = optimizer.param_groups[0]['lr']
                    logging.info(f"Epoch={epoch + 1}|step={step:4}/{len(train_dataloader)}|"
                                 f"loss={loss:6.4}|lr={lr:0.8}|acc={accuracy:0.4}|time={elap_t:.4}s")
                    t_begin = time.time()

            logging.info(f"train_loss={sum(train_loss) / len(train_loss):.4}")
            # validation
            logging.info('>>> begin validation...')
            val_loss, metrics, _ = validation(model, val_dataloader)
            logging.info(f">>> val_loss: {val_loss}, matrics: {metrics}")
            micro_f1 = metrics['micro_f1']

            if micro_f1 - micro_begin > 0.001:
                micro_begin = micro_f1
                no_decay_epoch = 0
                logging.info('>> model saving...')
                model_save.close()
                model_save = tempfile.TemporaryFile()
                dict_list = [model.module.state_dict()]  #
                torch.save(dict_list, model_save)
                note_log['best_epoch'][i] = epoch
            else:
                no_decay_epoch += 1
                if no_decay_epoch >= args.early_stop_epoch:
                    logging.info('the micro_f1 is not increase over %d epoches, STOP Training!' % args.early_stop_epoch)
                    break

        # 开始test
        logging.info('==========================================================')
        logging.info('>>> begin testing...')
        metrics = test_model(args, test_dataloader, model_save, dataloader)
        note_log['macro'].append(metrics['macro_f1'])
        note_log['micro'].append(metrics['micro_f1'])
        note_log['weighted_f1'].append(metrics['weighted_f1'])
        note_log['auc_pr_macro'].append(metrics['auc_pr_macro'])
        note_log['auc_pr_micro'].append(metrics['auc_pr_micro'])
        note_log['auc_macro'].append(metrics['auc_macro'])
        note_log['auc_micro'].append(metrics['auc_micro'])

    result = 'macro: %.4f|micro: %.4f|weighted: %.4f|auc_pr_macro:%.4f|auc_pr_micro:%.4f|auc_macro:%.4f|auc_micro:%.4f' % \
             (np.array(note_log['macro']).mean(),
              np.array(note_log['micro']).mean(),
              np.array(note_log['weighted_f1']).mean(),
              np.array(note_log['auc_pr_macro']).mean(),
              np.array(note_log['auc_pr_micro']).mean(),
              np.array(note_log['auc_macro']).mean(),
              np.array(note_log['auc_micro']).mean())
    # 保存日志
    if args.dataset_name == 'moviebricks':
        result_save_path = './moviebricks.json'
    else:
        result_save_path = './moviescope.json'
    with open(result_save_path, 'a+', encoding='utf-8') as f:
        json.dump(note_log, f)
        f.write('\n')
        f.write(result)
        f.write('\n')
    print('======result=====', result)


def rewrite_args(args, root):
    args.annotation = root + args.annotation
    args.graph_path = root + args.graph_path
    args.pic_video_path = root + args.pic_video_path
    args.poster_pic_path = root + args.poster_pic_path
    args.audio_feature_path = root + args.audio_feature_path
    args.label_path = root + args.label_path
    args.poster_feature_path = root + args.poster_feature_path
    args.word2vec_path = root + args.word2vec_path


if __name__ == '__main__':
    args = parse_args()
    if args.dataset_name == 'moviebricks':
        root = './data/'
    elif args.dataset_name == 'moviescope':
        root = './data-moviescope/'
    else:
        root = ''
    setup_seed(args)
    setup_device(args)
    setup_logging(args.model_name + '_' + args.ablation)
    logging.info("Training/evaluation parameters: %s", args)
    rewrite_args(args, root)

    if root == './data/':
        cls_freqs = {'动作': 744, '惊悚': 861, '冒险': 579, '剧情': 2244, '科幻': 403, '爱情': 591, '奇幻': 337, '喜剧': 1189, '恐怖': 414, '犯罪': 496}
        movie_num = 4063
        cls_weight = torch.FloatTensor([int(v) / movie_num for v in cls_freqs.values()]) ** -1
    else:
        cls_freqs = [953, 200, 237, 1556, 731, 2114, 461, 760, 451, 415, 910, 494, 1181]
        movie_num = 4076
        cls_weight = torch.FloatTensor([int(v) / movie_num for v in cls_freqs]) ** -1
        args.cls_num = len(cls_freqs)
        args.bert_dir = 'roberta-base'
        args.pretrain_model_lr = 5e-6
        args.warmup_steps = 200  # 6000
        args.bert_cache = './models/cache/bert'
        args.word2vec_path = './data-moviescope/word2vec/GoogleNews-vectors-negative300.bin.gz'

    print(cls_weight)
    args.cls_weight = cls_weight
    
    train_and_validation(args)
