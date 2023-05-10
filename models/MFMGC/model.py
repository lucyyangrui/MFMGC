import torch
import torch.nn as nn
from transformers import PretrainedConfig, SwinConfig, SwinModel
import torch.nn.functional as F
from transformers.models.bert.modeling_bert import BertConfig, BertPreTrainedModel, \
    BertEmbeddings, BertEncoder, BertModel

from transformers.models.roberta.modeling_roberta import RobertaConfig, RobertaPreTrainedModel, \
    RobertaEncoder, RobertaEmbeddings

import sys

sys.path.append(
    '/public/home/robertchen/yxr20214227065/multi-modal-mgc/movie-genres-classification-multimodal-master/models/')

from swin.swin import swin_small


def get_fusion_size(modals):
    ans = 0
    if 'audio' in modals:
        ans += 1
    if 'summary' in modals or 'video' in modals:
        ans += 1
    if 'poster' in modals:
        ans += 1
    return ans


class MFMGC(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.modals = args.modals
        self.device = args.device
        self.cls_weight = args.cls_weight
        self.bert_cfg = BertConfig.from_pretrained(args.bert_dir)
        self.hidden_size = args.hidden_size
        self.poster_hidden_size = args.poster_hidden_size

        if 'video' in args.modals:
            self.visual_backbone = swin_small(args.swin_pretrained_path)
            self.visual_backbone.requires_grad_(False)
        if 'summary' in args.modals:  #  'video' in args.modals:
            if args.dataset_name == 'moviebricks':
                self.bert = VlBertModel.from_pretrained(args.bert_dir)
            else:
                self.bert = VlRoBertModel.from_pretrained(args.bert_dir)
            # self.bert.requires_grad_(False)
            self.t_v_linear = nn.Linear(self.bert_cfg.hidden_size, self.hidden_size, bias=False)
            nn.init.kaiming_normal_(self.t_v_linear.weight)

        if 'poster' in args.modals:
            self.poster_linear_model = linear_model(self.poster_hidden_size, self.hidden_size)

        if 'audio' in args.modals:
            self.audio_hidden_size = args.audio_hidden_size
            self.audio_linear_model = linear_model(self.audio_hidden_size, self.hidden_size)

        self.fusion_size = get_fusion_size(self.modals)
        print('======fusion_size:', self.fusion_size)
        self.modal_fusion = ModalAttention(self.hidden_size)
        if self.fusion_size == -1:
            out_len = self.hidden_size
        else:
            out_len = self.hidden_size * self.fusion_size + self.fusion_size ** 2
        self.classifier = nn.Linear(out_len, args.cls_num)

        self.dropout = nn.Dropout(0.5)
        self.layer_norm = nn.LayerNorm(self.hidden_size, eps=1e-6)

    def encoder_frames(self, visual_input):
        assert len(visual_input.size()) == 5, print('visual input size must equal to 5!')
        bs = visual_input.size(0)
        frames = visual_input.size(1)
        visual_input = visual_input.view((bs * frames,) + visual_input.size()[2:])
        encoder_output = self.visual_backbone(visual_input)
        encoder_output = encoder_output.view(bs, frames, encoder_output.size(-1))
        return encoder_output

    def cal_loss(self, prediction, label):
        # label --> [bs, 10(cls_num)]
        label = label.to(self.device)  # new
        assert prediction.size() == label.size(), 'prediction size must equal to label'
        label_float = label.float()
        loss = nn.BCEWithLogitsLoss(pos_weight=self.cls_weight.to(self.device))  # (pos_weight=cls_weight.cuda())
        prediction = torch.sigmoid(prediction)
        loss = loss(prediction, label_float)
        with torch.no_grad():
            pred_label_id = torch.where(prediction > 0.5,
                                        torch.ones_like(label), torch.zeros_like(label))  # [bs, 25]  0 1 组成
            true_label_num = torch.sum(pred_label_id == label)

            accuracy = true_label_num / (label.size(0) * label.size(1))
        return loss, accuracy, pred_label_id

    def forward(self, data):
        inputs = {}
        inputs_mask = {}
        inputs_linear = {}

        out = []
        if 'summary' in self.modals:
            text_input = data['text_input']  # [bs, seq_len]
            text_mask = data['text_mask']  # [bs, seq_len]
            inputs['text_input'] = text_input
            inputs_mask['text_mask'] = text_mask

        if 'video' in self.modals:
            frame_input = data['frame_input']  # [bs, 32, 3, 224, 224]
            frame_mask = data['frame_mask']  # [bs, 32]
            visual_feature = self.encoder_frames(frame_input)  # [bs, 32, 768]
            inputs['visual_feature'] = visual_feature
            inputs_mask['visual_mask'] = frame_mask

        if 'audio' in self.modals:
            audio_feature = data['audio_feature']  # [bs, 1024]
            inputs_linear['audio_feature'] = audio_feature

        if 'poster' in self.modals:
            poster_feature = data['poster_feature'].to(self.device)   # new
            inputs_linear['poster_feature'] = poster_feature

        if 'summary' in self.modals:  # or 'video' in self.modals:
            output = self.bert(inputs, inputs_mask)  # [bs, seq_len + 32, 768]
            output = torch.mean(output, dim=1)
            output = self.t_v_linear(output)
            out.append(output.unsqueeze(dim=1))
        elif 'video' in self.modals:
            output = torch.mean(inputs['visual_feature'], dim=1)
            output = self.layer_norm(self.dropout(self.t_v_linear(output)))
            out.append(output.unsqueeze(dim=1))
        if 'poster' in self.modals:
            poster_output = self.poster_linear_model(inputs_linear['poster_feature'])
            out.append(poster_output.unsqueeze(dim=1))
        if 'audio' in self.modals:
            audio_output = self.audio_linear_model(inputs_linear['audio_feature'])
            out.append(audio_output.unsqueeze(dim=1))

        if self.fusion_size > 0:
            pooled_output = torch.cat(out, dim=1)  # [bs, fusion_size, hidden_size]
            pooled_output = self.modal_fusion(pooled_output)  # [bs, fusion_size * hidden + fusion_size ** 2]
        else:
            pooled_output = out[0].squeeze(dim=1)

        # pooled_output = self.dropout(pooled_output)
        classifier = self.classifier(pooled_output)  # [bs, 10]

        loss, acc, pred_label_id = self.cal_loss(classifier, data['label'])
        return loss, acc, pred_label_id, classifier, pooled_output


class VlBertModel(BertPreTrainedModel):
    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value: nn.Module):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(self, inputs: dict, inputs_mask: dict):  # ):
        encoder_input = []
        encoder_mask = []
        if 'text_input' in inputs:
            text_input = self.embeddings(inputs['text_input'])
            encoder_input.append(text_input)
            encoder_mask.append(inputs_mask['text_mask'])
        if 'visual_feature' in inputs:
            visual_emb = self.embeddings(inputs_embeds=inputs['visual_feature'],
                                         token_type_ids=torch.ones_like(inputs_mask['visual_mask']))
            encoder_input.append(visual_emb)
            encoder_mask.append(inputs_mask['visual_mask'])

        embeddings = torch.cat(encoder_input, dim=1)  #
        mask = torch.cat(encoder_mask, dim=1)  #
        # embeddings = text_input
        # mask = text_mask
        mask = mask[:, None, None, :]
        mask = (1.0 - mask) * -10000.0

        encoder_output = self.encoder(embeddings, attention_mask=mask)['last_hidden_state']
        return encoder_output


class VlRoBertModel(RobertaPreTrainedModel):
    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.config = config

        self.embeddings = RobertaEmbeddings(config)
        self.encoder = RobertaEncoder(config)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value: nn.Module):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(self, inputs: dict, inputs_mask: dict):  # ):
        encoder_input = []
        encoder_mask = []
        if 'text_input' in inputs:
            text_input = self.embeddings(inputs['text_input'])
            encoder_input.append(text_input)
            encoder_mask.append(inputs_mask['text_mask'])
        if 'visual_feature' in inputs:
            visual_emb = inputs['visual_feature'] # self.embeddings(inputs_embeds=inputs['visual_feature'],
                                         # token_type_ids=torch.ones_like(inputs_mask['visual_mask']))
            encoder_input.append(visual_emb)
            encoder_mask.append(inputs_mask['visual_mask'])

        embeddings = torch.cat(encoder_input, dim=1)  #
        mask = torch.cat(encoder_mask, dim=1)  #
        # embeddings = text_input
        # mask = text_mask
        mask = mask[:, None, None, :]
        mask = (1.0 - mask) * -10000.0

        encoder_output = self.encoder(embeddings, attention_mask=mask)['last_hidden_state']
        return encoder_output


class linear_model(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.linear1 = nn.Linear(in_size, out_size)
        self.act1 = nn.ReLU()
        self.linear2 = nn.Linear(out_size, out_size)

    def forward(self, inputs):
        out = self.linear1(inputs)
        out = self.act1(out)
        out = self.linear2(out)
        out = self.act1(out)
        return out


class Attention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(-2, -1))
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = attn / abs(attn.min())
        attn = self.dropout(F.softmax(F.normalize(attn, dim=-1), dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


class ModalAttention(nn.Module):
    def __init__(self, hidden):
        super(ModalAttention, self).__init__()
        self.w_q = nn.Linear(hidden, hidden, bias=False)
        self.w_k = nn.Linear(hidden, hidden, bias=False)
        self.w_v = nn.Linear(hidden, hidden, bias=False)

        self.attention = Attention(temperature=hidden ** 0.5)

        self.dropout = nn.Dropout(0.5)
        self.layer_norm = nn.LayerNorm(hidden, eps=1e-6)  # 这两行很重要~~

    def forward(self, data):
        bs = data.size(0)
        residual = data
        q = self.w_q(data)  # [bs, fusion_size, hidden]
        k = self.w_k(data)
        v = self.w_v(data)

        out, attn = self.attention(q, k, v)  # out [bs, fusion_size, hidden] attn [bs, fusion_size, hidden]
        out = self.dropout(out)
        out += residual
        out = self.layer_norm(out)
        out = out.view(bs, -1)
        attn = attn.view(bs, -1)
        out = torch.cat((out, attn), dim=1)
        return out  # [bs, fusion_size * hidden + fusion_size ** 2]
