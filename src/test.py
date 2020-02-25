from __future__ import print_function
import torch
import torch.nn as nn
from options import get_training_parser, parse_args
from vist_dataloader import VistDataLoader
from vist_model import VistModel
from scorer import score_func
import logging
# import tqdm
import time
import numpy as np
import os, sys
sys.path.append(os.path.abspath('./'))

def init_logging(args, mode='Train'):
    logger = logging.getLogger(mode)
    file_handler = logging.FileHandler(args.log_file)
    logger.addHandler(file_handler)
    logger.setLevel(logging.DEBUG)
    return logger

def test(args):
	# load data
    data = VistDataLoader(args)
    trg_vocab = data.vocab
	# initialize or load model
    print('load model from %s' % args.load_model_from)
    model = VistModel.load(args.load_model_from, args)

    decode(args, trg_vocab, model, data.tst)
    # tst_output = " ".join("{}:{}".format(m, v) for m, v in tst_scores.items())
    # print('Test: tst. metric %s' % (tst_output))


def decode(args, vocab, model, data):
    cum_loss = cum_trg_word = cum_example = 0.0
    all_hyps = []
    all_refs = []
    all_albums = []

    for sample in VistDataLoader.ref_iter(data, batch_size=args.batch_size, cuda=args.cuda):
        with torch.no_grad():
            raw_hyps, scores = model.generate(
                sample['src_seq'], sample['src_lengths'],
                beam_size=args.beam_size,
                decode_max_length=args.decode_max_length,
                decode_type=args.decode_type,
                to_word=True)
        hyps = [" ".join(" ".join(h[1:-1]) for h in beam[0]) for beam in raw_hyps]
        all_hyps.extend(hyps)  # [batch, num_seq * decode_max_length]
        refs = sample['trg_seq']

        #for i, (hyp, score) in enumerate(zip(raw_hyps, scores)):
        #    print('------------------\n batch id', i)
        #    for b, (hs, ss) in enumerate(zip(hyp, score)):
        #        print('{} {} {}'.format(b, ss, " || ".join(' '.join(ww) for ww in hs)))
        #exit(0)

        all_refs.extend(refs)
        all_albums.extend(sample['album'])
    if args.save_decode_file is not None:
        with open(args.save_decode_file, 'w') as f:
            for a, h in zip(all_albums, all_hyps):
                f.write('{}\t {}\n'.format(a, h))

if __name__ == '__main__':
    parser = get_training_parser()
    parser = VistDataLoader.add_args(parser)
    parser = VistModel.add_args(parser)
    args = parser.parse_args()

    # seed the RNG
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    np.random.seed(int(args.seed * 13 / 7))

    if args.python_dir is not None:
        sys.path.append(args.python_dir)
        print('sys.path', sys.path)

    # print('args', args)
    print('Test Arguments')
    for arg in vars(args):
        print('\t{}\t:{}'.format(arg, getattr(args, arg)))
    test(args)
