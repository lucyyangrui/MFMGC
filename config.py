import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='args for multimodal movie genre classification')

    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--dataset_name', type=str, default='moviebricks')  # moviebricks moviescope
    parser.add_argument('--model_name', type=str, default='base1')  # base1 base2 base3 mymodel
    parser.add_argument('--modals', nargs='+', type=str, default=['summary'])  # summary poster video audio
    parser.add_argument('--fusion_size', type=int, default=3)
    parser.add_argument('--mode', type=str, default='train')  # train test

    # ========================= Data Configs ==========================
    parser.add_argument('--annotation', type=str, default='annotation/filter_data.json')
    parser.add_argument('--graph_path', type=str, default='graph_data/filter_node_graph.csv')
    parser.add_argument('--pic_video_path', type=str, default='Trailers-frames/')
    parser.add_argument('--poster_pic_path', type=str, default='/pic-videos/')
    parser.add_argument('--audio_feature_path', type=str, default='features/audio_features.csv')
    parser.add_argument('--audio_wav2vec_path', type=str, default='./models/MFMGR/audio_feature.json')
    parser.add_argument('--label_path', type=str, default='features/label.csv')
    parser.add_argument('--poster_feature_path', type=str, default='features/poster_features.csv')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='validation data ratio')
    parser.add_argument('--test_ratio', type=float, default=0.2, help='test data ratio')
    parser.add_argument('--train_batch_size', type=int, default=16, help='batch size for training')  # 23
    parser.add_argument('--val_batch_size', type=int, default=16, help='batch size for validation')
    parser.add_argument('--prefetch', default=64, type=int, help="use for training duration per worker")
    parser.add_argument('--num_workers', type=int, default=2)  # 4
    parser.add_argument('--cls_num', type=int, default=10)

    # ========================= train Configs ==========================
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument('--warmup_steps', type=int, default=200)  # 200
    parser.add_argument('--max_steps', type=int, default=4000)
    parser.add_argument('--model_save_path', type=str, default='./models/MFMGC/save/')
    parser.add_argument('--note', type=str, default='mymodel')
    parser.add_argument('--hidden_size', type=int, default=256)

    # ========================= text Configs ==========================
    parser.add_argument('--max_seq_len', type=int, default=256)  # 256
    parser.add_argument('--bert_dir', type=str, default='hfl/chinese-roberta-wwm-ext')
    parser.add_argument('--bert_cache', type=str, default='./models/cache/bert/')

    # ========================= video Configs ==========================
    parser.add_argument('--max_frame', type=int, default=32)  # 32
    parser.add_argument('--resnet_cache', type=str, default='./models/cache/resnet152')
    parser.add_argument('--swin_pretrained_path', type=str,
                        default='./models/cache/swin/swin_small_patch4_window7_224_22k.pth')

    # ========================= audio Configs ==========================
    parser.add_argument('--partial_num', type=int, default=5, help='audio partial num, default 10')
    parser.add_argument('--partial_len', type=float, default=3.0, help='audio partial length, default 3.0s')
    parser.add_argument('--temporal_dimension', type=int, default=130, help='audio melspectrogram temporal dimension')
    parser.add_argument('--frequency_dimension', type=int, default=513, help='audio melspectrogram frequency dimension')
    parser.add_argument('--wav2vec_hidden_size', type=int, default=1024, help='wav2vec pretrain model output dim')

    parser.add_argument('--poster_hidden_size', type=int, default=1024)
    parser.add_argument('--audio_hidden_size', type=int, default=1024)

    # ========================= baseline1 CTT-MMC Configs ==========================
    parser.add_argument('--bs1_audio_hidden_size', type=int, default=513, help='audio feature length, for conv')
    parser.add_argument('--bs1_video_hidden_size', type=int, default=2048, help='video feature length, for conv')
    parser.add_argument('--bs1_out_channel', type=int, default=256, help='ctt module conv out channel')
    parser.add_argument('--bs1_kernel_size', type=int, default=3, help='ctt module conv kernel size')
    parser.add_argument('--base1_audio_sus', type=str, default='./models/CCT_MMC_base1/audio_sus.json')

    parser.add_argument('--word2vec_path', type=str,
                        default='word2vec/sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5')

    parser.add_argument('--vgg_model_path', type=str, default='./data/vgg16/vgg16-397923af.pth')
    parser.add_argument('--graph_alg', type=str, default='GCN')

    # ========================= ablation ===========================================
    parser.add_argument('--ablation', type=str, default='')
    parser.add_argument('--save_model_name', type=str, default='/model.bin')
    parser.add_argument('--audio_part_num', type=int, default=16)
    parser.add_argument('--pretrain_model_lr', type=float, default=5e-5)  # 5e-5 using for the utils vlbert_lr
    parser.add_argument('--other_model_lr', type=float, default=5e-4)  # 5e-4
    parser.add_argument('--early_stop_epoch', type=int, default=10)

    # ========================= baseline4 =========================
    parser.add_argument('--poster_model', type=str, default='resnet')

    return parser.parse_args()
