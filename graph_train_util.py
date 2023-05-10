import json
import logging
import time
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from utils import evaluate, setup_logging, setup_seed, setup_device, build_optimizer, build_optimizer_vlbert
import tempfile
import torch.nn as nn
import dgl
import torch.optim as optim
from dgl.dataloading import MultiLayerNeighborSampler, NodeDataLoader
from config import parse_args
from data_helper import create_dataloader, create_dataloader_all
from train import validation
# from models.CCT_MMC_base1.model import Base1CttMmc
# from models.GMU_base2.model import GMU_MGC
# from models.moviescope_base3.model import Base3Moviescope
from models.MFMGC.model import MFMGC
from models.GRAPH.GAT import GAT
from models.GRAPH.GCN import GCN
from models.GRAPH.graph_data_helper import load_graph_data

from train import rewrite_args

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class ModelWithGraph(nn.Module):
    def __init__(self, args, train_index=None, val_index=None, test_index=None):
        super(ModelWithGraph, self).__init__()
        self.device = args.device
        self.train_dataloader, self.val_dataloader, self.test_dataloader, \
        self.train_index, self.val_index, self.test_index = create_dataloader(args, train_index, val_index, test_index)
        self.dataloader = create_dataloader_all(args)
        self.ModalFusion = MFMGC(args)
        if args.device == 'cuda':
            self.ModalFusion = self.ModalFusion.to(
                args.device)  # torch.nn.parallel.DataParallel(self.ModalFusion.to(args.device))
            print('>> using gpu to loading model...')
        self.graph_alg = args.graph_alg
        self.hidden_size = args.hidden_size
        if self.graph_alg == 'GCN':
            self.GraphLearn = GCN(257, 257, 10, 0.1).to(args.device)
        else:
            self.GraphLearn = GAT(self.hidden_size, 512, 10, 0.6, 0.1, 2).to(args.device)

        self.optimizer_MF, self.schedule_MF = build_optimizer_vlbert(args,
                                                                     self.ModalFusion)  # optim.Adam(self.ModalFusion.parameters(), lr=0.0005, weight_decay=0.0036)
        # self.optimizer_GL, self.schedule_GL = build_optimizer_vlbert(args, self.GraphLearn) # optim.Adam(self.GraphLearn.parameters(), lr=0.00005, weight_decay=5e-4)
        self.optimizer_GL = optim.AdamW(self.GraphLearn.parameters(), lr=0.001, weight_decay=5e-4)

        self.graph_data = load_graph_data(args.annotation, args.graph_path)

    def forward(self, test='train', epoch=0):
        # 进行多模态数据融合 --> 得到节点的表示
        t_begin = time.time()
        if test == 'train':
            for step, batch in enumerate(self.train_dataloader):
                self.ModalFusion.train()
                self.optimizer_MF.zero_grad()
                loss, accuracy, _, _, _ = self.ModalFusion(batch)
                loss.backward()
                self.optimizer_MF.step()
                self.schedule_MF.step()
                elap_t = time.time() - t_begin
                lr = self.optimizer_MF.param_groups[0]['lr']
                print_step(epoch, step, len(self.train_dataloader), lr, loss, accuracy, elap_t)
                t_begin = time.time()

        hidden_matrix = torch.empty((0)).to(self.device)
        targ = []
        if test == 'graph':
            for step, batch in enumerate(self.dataloader):
                with torch.no_grad():
                    loss, accuracy, _, _, hidden = self.ModalFusion(batch)
                hidden_matrix = torch.cat([hidden_matrix, hidden], dim=0)

                targ.extend(batch['label'].cpu().numpy())
        else:
            validation(self.ModalFusion, self.val_dataloader)

        return hidden_matrix, targ

    def forward_graph(self, hidden_matrix, label, test='train'):
        # 图表示学习
        loss_total = 0
        G = dgl.from_scipy(self.graph_data['adj']).to(self.device)
        G.ndata['feat'] = hidden_matrix.to(self.device)
        G.ndata['label'] = torch.tensor(np.array(label)).to(self.device)
        G.edata['w'] = torch.tensor(self.graph_data['adj'].data).to(self.device)

        if test == 'train':
            idx = self.train_index
        elif test == 'val':
            idx = self.val_index
        else:
            idx = self.test_index
        sampler = MultiLayerNeighborSampler([5, 10])
        node_loader = NodeDataLoader(G,
                                     torch.tensor(idx).to(self.device),
                                     sampler,
                                     batch_size=3000,
                                     shuffle=False,
                                     drop_last=False)
        # num_batches, size = len(node_loader), len(node_loader.dataset)
        pre_ids = []
        pred = []
        labels = []
        for input_nodes, output_nodes, blocks in node_loader:
            self.optimizer_GL.zero_grad()
            blocks = [b.to(torch.device('cuda')) for b in blocks]
            input_feat = blocks[0].srcdata['feat']
            label = blocks[-1].dstdata['label']
            loss, accuracy, pred_label_id, prediction = self.GraphLearn(blocks, input_feat, label)
            if test == 'train':
                loss.backward()
                self.optimizer_GL.step()
                # self.schedule_GL.step()
            loss_total += loss.item()

            pred.extend(prediction.detach().cpu().numpy())
            pre_ids.extend(pred_label_id.detach().cpu().numpy())
            labels.extend(label.detach().cpu().numpy())

        metrics = evaluate(pre_ids, labels, pred)

        return metrics, loss_total

    def run_epoch(self, test='train', epoch=0):
        hidden_matrix, targ = self.forward(test, epoch)
        mat, _ = self.forward_graph(hidden_matrix, targ, test)
        print(mat)
        print('begin validation...')
        metric, loss = self.forward_graph(hidden_matrix, targ, 'val')
        # print('epoch: %d, graph_loss: %.4f' % (epoch, loss))
        return metric


def print_step(epoch, step, total_step, lr, loss, acc, elap_t):
    if step == 2 or step and step % 100 == 0:
        logging.info(f"epoch={epoch:4}|step={step:4}/{total_step}|"
                     f"loss={loss:6.4}|lr={lr:0.8}|acc={acc:0.4}|time={elap_t:.4}s")


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

    cls_freqs = {'动作': 744, '惊悚': 861, '冒险': 579, '剧情': 2244, '科幻': 403, '爱情': 591, '奇幻': 337, '喜剧': 1189, '恐怖': 414,
                 '犯罪': 496}
    movie_num = 4063
    cls_weight = torch.FloatTensor([int(v) / movie_num for v in cls_freqs.values()]) ** -1
    print(cls_weight)
    args.cls_weight = cls_weight

    note_log = dict(
        note='-'.join(args.modals),  # args.note,
        best_epoch=[0 for _ in range(5)],
        macro=[],
        micro=[],
        weighted_f1=[],
        auc_pr_macro=[],
        auc_pr_micro=[],
        auc_macro=[],
        auc_micro=[],
    )

    skf = KFold(n_splits=5, random_state=42, shuffle=True)
    labels = pd.read_csv(args.label_path, header=None, index_col=None).values
    args.data_num = len(labels)
    for i, (train_val_index, test_index) in enumerate(skf.split(range(args.data_num), labels)):
        print('==================== %d fold ====================' % i)
        train_index = train_val_index[: int(len(train_val_index) * 7 / 8)]
        val_index = train_val_index[int(len(train_val_index) * 7 / 8):]
        model = ModelWithGraph(args, train_index, val_index, test_index)
        micro_best, no_decay_epoch = 0.0, 0
        model_save = tempfile.TemporaryFile()
        for epoch in range(5000):
            met = model.run_epoch('train', epoch)
            print(met)
            if met['micro_f1'] - micro_best > 0.001:
                micro_best = met['micro_f1']
                print('save model...')
                model_save.close()
                model_save = tempfile.TemporaryFile()
                dict_list = [model.state_dict()]
                torch.save(dict_list, model_save)
                no_decay_epoch = 0
                note_log['best_epoch'].append(epoch)
            else:
                no_decay_epoch += 1
                if no_decay_epoch >= 50:  # args.early_stop_epoch:
                    break

        model_save.seek(0)
        dict_list = torch.load(model_save)
        model.load_state_dict(dict_list[0])
        print('begin testing...')
        hidden_matrix, targ = model.forward('test')
        metrics, _ = model.forward_graph(hidden_matrix, targ, 'test')
        print(metrics)
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

    print('======result=====', result)
