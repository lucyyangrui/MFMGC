import json
import os

import torch
import librosa
import numpy as np
import math
from transformers import Wav2Vec2Tokenizer, Wav2Vec2Model


def get_audio_list(audio_path, partial_num, partial_len):
    mask = torch.zeros((partial_num,), dtype=torch.long)
    y_list = []
    try:
        y, sr = librosa.load(audio_path, sr=16000)
    except:
        y = []
        sr = 16000
    partial_duration = max(math.floor(len(y) / partial_num), int(partial_len) * sr)
    for i in range(partial_num):
        begin_time = int(partial_duration * i)
        end_time = int(begin_time + sr * partial_len)
        if end_time <= len(y):
            y_list.append(y[begin_time: end_time])
            mask[i] = 1
        else:
            y_list.append(np.zeros((end_time - begin_time,)))
    # audio_list = audio_tokenizer(y_list, return_tensors="pt").input_values
    return y_list, np.array(mask)


if __name__ == '__main__':
    pic_video_path = '../../data/pic-videos/'
    movie_ids = os.listdir(pic_video_path)
    audio_tokenizer = Wav2Vec2Tokenizer.from_pretrained('audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim')
    model = Wav2Vec2Model.from_pretrained('audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim')
    audio_feature_map = {}
    for i, idx in enumerate(movie_ids):
        print(idx)
        audio_path = os.path.join(pic_video_path, idx, 'data', 'audio.wav')
        y_list, mask = get_audio_list(audio_path, 16, 5)
        audio_tk = audio_tokenizer(y_list, return_tensors="pt")['input_values']  # [16, sr * 5]
        audio_feature = model(audio_tk).last_hidden_state  # [16, ,768]
        audio_feature = torch.mean(audio_feature, dim=1)  # [16, 768]
        audio_tmp_dict = {'audio_feature': audio_feature.detach().numpy().tolist(), 'mask': mask.tolist()}
        audio_feature_map[idx] = audio_tmp_dict

        if i % 100 == 0 and i:
            print('================= ', i)
            # with open('./audio_feature.json', 'a+', encoding='utf-8') as f:
                # json.dump(audio_feature_map, f)
            # audio_feature_map = {}

    if len(audio_feature_map):
        print(len(audio_feature_map))
        with open('./audio_feature.json', 'w', encoding='utf-8') as f:
            json.dump(audio_feature_map, f)

    # with open('./audio_feature.json', 'r', encoding='utf-8') as f:
    #     data = json.load(f)
    #
    # print(data)
