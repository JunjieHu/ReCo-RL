import argparse
import os

def get_parser(desc):
    parser = argparse.ArgumentParser(
        description='Sequence to Sequence -- ' + desc)
    parser.add_argument('--log_interval', type=int, default=50, help='log progress every N iterations')
    parser.add_argument('--log_file', default=None, type=str, help='logging file')
    parser.add_argument('--seed', default=1, type=int, help='random seed')
    return parser

def get_training_parser():
    parser = get_parser('Training')
    add_dataset_args(parser, train=True)
    add_model_args(parser)
    add_optimization_args(parser)
    add_checkpoint_args(parser)
    return parser

def get_testing_parser():
    parser = get_parser('Testing')
    add_dataset_args(parser, train=False)
    add_model_args(parser)
    add_optimization_args(parser)
    add_checkpoint_args(parser)
    return parser

def parse_args(parser):
    args = parser.parse_args()
    return args

def add_dataset_args(parser, train=True):
    group = parser.add_argument_group('Dataset and data processing')
    group.add_argument('--s', '--source_lang', default=None, help='source language')
    group.add_argument('--t', '--target_lang', default=None, help='target language')
    group.add_argument('--train_src_file', default=None, type=str, help='train source data')
    group.add_argument('--train_trg_file', default=None, type=str, help='train target data')
    group.add_argument('--dev_src_file', default=None, type=str, help='dev source data')
    group.add_argument('--dev_trg_file', default=None, type=str, help='dev target data')
    group.add_argument('--test_src_file', default=None, type=str, help='test source data')
    group.add_argument('--test_trg_file', default=None, type=str, help='test target data')
    group.add_argument('--train_bitext_file', default=None, type=str)
    group.add_argument('--dev_bitext_file', default=None, type=str)
    group.add_argument('--test_bitext_file', default=None, type=str)
    group.add_argument('--delimiter', default="|||", type=str)
    group.add_argument('--src_vocab_size', default=50000, type=int)
    group.add_argument('--trg_vocab_size', default=50000, type=int)
    group.add_argument('--share_vocab', action='store_true', default=False)
    group.add_argument('--vocab', default=None, type=str, help='vocabulary file')
    group.add_argument('--include_singleton', default=False, action='store_true', help='whether to include singleton in the vocabulary (default=False)')
    group.add_argument('--batch_size', default=64, type=int, help='number of training pairs in a batch')
    group.add_argument('--skip_invalid_data', action='store_true', default=False, help='skip too long or too short sentences')
    group.add_argument('--data_dir', default=os.getenv('PT_DATA_DIR', None), type=str, help='data directory')
    return group

def add_model_args(parser):
    group = parser.add_argument_group('Model Configuration')
    group.add_argument('--uniform_init', default=0.1, type=float)
    group.add_argument('--cuda', default=False, action='store_true')
    group.add_argument('--arch', '-a', default='lstm', type=str)
    group.add_argument('--beam_size', default=5, type=int, help='beam size')
    group.add_argument('--decode_max_length', default=100, type=int, help='maximun decoding step')
    group.add_argument('--save_nbest_format', default='cdec', type=str, help='format to save nbest list')
    group.add_argument('--save_decode_file', default=None, type=str, help='path to save the decoding output')
    group.add_argument('--model_dir', default=os.getenv('PT_DATA_DIR', None), type=str, help='directory for pretrained model')
    group.add_argument('--python_dir', default=None, type=str)
    group.add_argument('--decode_type', default='beam', type=str)
    return group

def add_optimization_args(parser):
    group = parser.add_argument_group('Optimization')
    group.add_argument('--max_epoch', default=5000, type=int, help='force stop training at maximum epoch')
    group.add_argument('--optim', default='adam', help='optimizer')
    group.add_argument('--lr', '--learning-rate', default=0.001, type=float, help='learning rate')
    group.add_argument('--lr_decay', default=0.5, type=float, help='learning rate decay')
    group.add_argument('--clip_norm', default=5.0, type=float, help='clip the gradient norm to 5')
    group.add_argument('--objective', default='MLE', type=str, help='objective functions, e.g. MLE, REINFORCE')
    group.add_argument('--rl_reward', default='BLEU', type=str, help='REINFORCE reward function')
    group.add_argument('--sample_size', default=5, type=int, help='sample size in REINFORCE')
    group.add_argument('--valid_metric', default='loss', type=str, help='validation metric')
    return group

def add_checkpoint_args(parser):
    group = parser.add_argument_group('Save/Load Checkpoints')
    group.add_argument('--save_dir', default=os.getenv('PT_OUTPUT_DIR', None), type=str, help='directory to save model')
    group.add_argument('--save_model_to', default=None, help='path to save chekcpoints')
    group.add_argument('--load_model_from', default=None, help='path to load checkpoints')
    group.add_argument('--valid_interval', type=int, default=100, help='validate every N iterations')
    group.add_argument('--save_model_after', type=int, default=2, help='save checkpoints after N validations')
    group.add_argument('--patience', default=5, type=int, help='patience for early stop')
    group.add_argument('--save_last_K_model', default=5, type=int, help='save the last K models')
    return group
