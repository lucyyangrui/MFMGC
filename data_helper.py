import json
import random
import time
import zipfile
from io import BytesIO
import os
from PIL import Image
import logging

import numpy as np
import torch
from functools import partial
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import torch.nn.functional as F
from transformers import BertTokenizer, AutoFeatureExtractor, AutoTokenizer
from transformers import Wav2Vec2Tokenizer
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize, ToTensor
import jieba
import librosa
import pandas as pd
import math
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim.models.keyedvectors import KeyedVectors


def create_dataloader(args, train_index=None, val_index=None, test_index=None, test_mode=False):
    droplast = True
    if args.model_name == 'mymodel':
        droplast = False
        
    ann_path = args.annotation
    pic_video_path = args.pic_video_path

    logging.info('>>> loading data... from %s %s' % (ann_path, pic_video_path))
    dataset = MultiModalDataset(args, test_mode)
    if train_index is None and val_index is None:
        data_size = len(dataset)
        val_size = int(data_size * args.val_ratio)
        test_size = int(data_size * args.test_ratio)
        train_size = data_size - val_size - test_size
        index_shuffle = [i for i in range(data_size)]
        random.shuffle(index_shuffle)
        train_index = index_shuffle[: train_size]
        val_index = index_shuffle[train_size: train_size + val_size]
        test_index = index_shuffle[train_size + val_size: data_size]

    logging.info('>>> loading data... train_size: %d, val_size: %d, test_size %d' % (
        len(train_index), len(val_index), len(test_index)))
    train_dataset, val_dataset, test_dataset = torch.utils.data.Subset(dataset, train_index), \
                                               torch.utils.data.Subset(dataset, val_index), torch.utils.data.Subset(
        dataset, test_index),
    if args.num_workers > 0:
        dataloader_class = partial(DataLoader, pin_memory=True, num_workers=args.num_workers,
                                   prefetch_factor=args.prefetch)
    else:
        # single-thread reading does not support prefetch_factor arg
        dataloader_class = partial(DataLoader, pin_memory=True, num_workers=0)

    train_sampler = RandomSampler(train_dataset)
    val_sampler = RandomSampler(val_dataset)
    test_sampler = RandomSampler(test_dataset)
    train_dataloader = dataloader_class(train_dataset,
                                        batch_size=args.train_batch_size,
                                        sampler=train_sampler,
                                        drop_last=droplast)
    val_dataloader = dataloader_class(val_dataset,
                                      batch_size=args.val_batch_size,
                                      sampler=val_sampler,
                                      drop_last=droplast)
    test_dataloader = dataloader_class(test_dataset,
                                       batch_size=args.val_batch_size,
                                       sampler=test_sampler,
                                       drop_last=droplast)

    train_index_pd = pd.DataFrame(train_index)
    val_index_pd = pd.DataFrame(val_index)
    test_index_pd = pd.DataFrame(test_index)
    train_index_pd.to_csv('./data/graph_data/train_index.csv', header=False, index=False)
    val_index_pd.to_csv('./data/graph_data/val_index.csv', header=False, index=False)
    test_index_pd.to_csv('./data/graph_data/test_index.csv', header=False, index=False)

    return train_dataloader, val_dataloader, test_dataloader, train_index, val_index, test_index


def create_dataloader_all(args):
    dataset = MultiModalDataset(args)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset,
                            batch_size=args.val_batch_size,
                            sampler=sampler,
                            drop_last=False)
    return dataloader


class MultiModalDataset(Dataset):
    """ A simple class that supports multi-modal inputs.

        Args:
            ann_path (str): annotation file path, with the '.json' suffix.
            pic_video_path (str): poster and video frames.
            test_mode (bool): if it's for testing.  -- no label
        """

    def __init__(self,
                 args,
                 test_mode: bool = False):
        self.dataset = args.dataset_name
        self.model_name = args.model_name
        self.modals = args.modals
        self.max_frame = args.max_frame
        self.partial_num = args.partial_num
        self.partial_len = args.partial_len
        self.frequency_dimension = args.frequency_dimension
        self.temporal_dimension = args.temporal_dimension
        self.bert_seq_len = args.max_seq_len
        self.test_mode = test_mode

        self.cls_num = args.cls_num
        self.audio_feature_path = args.audio_feature_path
        
        self.audio_wav2vec_path = args.audio_wav2vec_path

        self.handles = [None for _ in range(args.num_workers)]

        self.pic_video_path = args.pic_video_path
        self.poster_pic_path = args.poster_pic_path
        with open(args.annotation, 'r', encoding='utf-8') as f:
            self.anns = json.load(f)

        if 'poster' in args.modals:
            if args.model_name == 'mymodel' or args.model_name == 'base5':
                self.poster_features = pd.read_csv(args.poster_feature_path, header=None, index_col=None).values

        self.labels = pd.read_csv(args.label_path, header=None, index_col=None).values
        
        if 'audio' in args.modals:
            self.audio_features = pd.read_csv(args.audio_feature_path, header=None, index_col=None).values
            
        self.audio_part_num = args.audio_part_num

        def str_to_float(x):
            return float(x)

        self.word_vec_dict = {}
        self.stop_word = []
        if args.model_name != 'mymodel':
            if 'summary' in self.modals:
                if self.dataset == 'moviebricks':
                    with open('./data/word2vec/stop_word.txt', 'r') as f:
                        self.stop_word = [w.strip() for w in f]
                    with open(args.word2vec_path, 'rb') as f:
                        for i, line_b in enumerate(f):
                            line_u = line_b.decode('utf-8')
                            if i >= 1:
                                word_vec = line_u.strip('\n ').split(' ')
                                self.word_vec_dict[word_vec[0]] = list(map(str_to_float, word_vec[1:]))
                else:
                    self.stopWords_en = set(stopwords.words('english'))
                    self.word_vec_dict = KeyedVectors.load_word2vec_format(args.word2vec_path, binary=True)
        self.tokenizer = AutoTokenizer.from_pretrained(args.bert_dir, cache_dir=args.bert_cache)

        self.visual_feature_exact = AutoFeatureExtractor.from_pretrained("microsoft/resnet-152",
                                                                         cache_dir=args.resnet_cache)
        self.transform = Compose([
            Resize(224),
            CenterCrop(224),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.anns)

    def get_visual_frames(self, idx: int) -> tuple:
        vid = str(self.anns[idx]['movie_id'])
        pic_video_path = os.path.join(self.pic_video_path, vid)

        namelist = os.listdir(pic_video_path)
        num_frames = min(self.max_frame, len(namelist))

        frame = torch.zeros((self.max_frame, 3, 224, 224), dtype=torch.float32)
        mask = torch.zeros((self.max_frame,), dtype=torch.long)

        select_inds = list(range(num_frames))

        for i, j in enumerate(select_inds):
            mask[i] = 1
            with Image.open(os.path.join(pic_video_path, str(i) + '.jpg')) as image:
                if self.model_name == 'mymodel':
                    img_tensor = self.transform(image)
                else:
                    img_tensor = self.visual_feature_exact(image, return_tensors='pt')['pixel_values'][0]
                frame[i] = img_tensor

        return frame, mask

    def get_poster(self, idx: int):
        vid = str(self.anns[idx]['movie_id'])
        poster_pic_path = os.path.join(self.poster_pic_path, vid)
        if self.dataset == 'moviebricks':
            poster_name = 'data/poster.webp'  # os.listdir(poster_pic_path)[0]
            poster_path = os.path.join(poster_pic_path, poster_name)
        else:
            poster_path = poster_pic_path + '.jpg'
        with Image.open(poster_path) as img:
            poster_tensor = self.transform(img.convert('RGB'))
        return poster_tensor

    def get_audio_feature(self, idx: int):
        vid = self.anns[idx]['movie_id']
        sus = self.base1_audio_sus[str(vid)]
        if sus:
            # zip_handler = zipfile.ZipFile(self.audio_feature_path, 'r')
            # BytesIO(zip_handler.read(name=str(vid) + '/audio_feature.npy'))
            audio_feature = np.load(self.audio_feature_path + '/' + str(vid) + '/audio_feature.npy')
        else:
            audio_mel_matric = []
            for i in range(5):
                mel_spect = torch.zeros((130, 513), dtype=torch.float)
                audio_mel_matric.append(np.array(mel_spect))
            audio_feature = np.array(audio_mel_matric)
        return torch.Tensor(audio_feature)

    def tokenizer_text(self, title: str, summary: str) -> tuple:
        summary = summary.replace(' ', '')
        if title:
            title = title.split(' ')[0]
            text = '[SEP]' + title + '[SEP]' + summary
        else:
            text = '[SEP]' + summary
        encoded_input = self.tokenizer(text, max_length=self.bert_seq_len, padding='max_length', truncation=True)
        input_ids = torch.LongTensor(encoded_input['input_ids'])
        mask = torch.LongTensor(encoded_input['attention_mask'])

        return input_ids, mask

    def get_text_vec(self, title: str, summary: str):
        text_words = []
        if self.dataset == 'moviebricks':
            title = title.split(' ')[0]
            summary = summary.replace(' ', '')
            tmp_words = jieba.cut(title + summary)
            for w in tmp_words:
                if w not in self.stop_word:
                    text_words.append(w)
        else:
            words = word_tokenize(summary)
            for w in words:
                if w not in self.stopWords_en:
                    text_words.append(w)
            text_words = text_words[:64]
        word_vec = None
        len_text_word = 0
        for word in text_words:
            if word in self.word_vec_dict:
                len_text_word += 1
                if word_vec is None:
                    word_vec = torch.FloatTensor(torch.tensor(self.word_vec_dict[word]))
                else:
                    word_vec += torch.FloatTensor(torch.tensor(self.word_vec_dict[word]))
        if len_text_word == 0:
            word_vec = torch.zeros((300))
            len_text_word = 1
        word_vec /= len_text_word
        return word_vec

    def __getitem__(self, idx: int) -> dict:
        data = {}
        if 'poster' in self.modals:
            if self.model_name == 'mymodel':
                poster_feature = self.poster_features[idx]
                data['poster_feature'] = torch.Tensor(poster_feature)
            else:
                poster_tensor = self.get_poster(idx)
                data['poster_input'] = poster_tensor
        if 'video' in self.modals:
            frame_tensor, frame_mask = self.get_visual_frames(idx)
            data['frame_input'] = frame_tensor
            data['frame_mask'] = frame_mask
        if 'summary' in self.modals:
            if self.dataset == 'moviescope':
                summary = self.anns[idx]['plot']
                title = ''
            else:
                title = self.anns[idx]['title']
                summary = self.anns[idx]['summary']
            if self.model_name == 'mymodel':
                input_ids, input_mask = self.tokenizer_text(title, summary)
                data['text_input'] = input_ids
                data['text_mask'] = input_mask
            else:
                word_vec = self.get_text_vec(title, summary)
                data['word_vec'] = word_vec

        # audio_feature = self.get_audio_feature(idx)
        if 'audio' in self.modals:
            audio_feature = self.audio_features[idx]  # mymodel
            data['audio_feature'] = torch.Tensor(audio_feature)

        if not self.test_mode:
            label = self.labels[idx]
            data['label'] = torch.Tensor(label)

        return data

# test
# setup_logging()
# import config
# from utils import evaluate, setup_logging, setup_seed, setup_device, build_optimizer, build_optimizer_vlbert
# args = config.parse_args()
# setup_seed(args)
# setup_device(args)
# train_dataloader, val_dataloader, test_dataloader = create_dataloader(args)
# for i, t in enumerate(train_dataloader):
    # print(t['audio_list'].shape)
    # if i == 2:
        # break
