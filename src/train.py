from __future__ import print_function
import torch
import torch.nn as nn 
from options import get_training_parser, parse_args
from trainer import Trainer
from vist_dataloader import VistDataLoader 
from vist_model import VistModel
# from scorer import score_func
from album_eval import AlbumEvaluator
import logging
import time 
import numpy as np
import os, sys
from bert_nsp import get_nsp

def init_logging(args, mode='Train'):
    logger = logging.getLogger(mode)
    file_handler = logging.FileHandler(args.log_file)
    logger.addHandler(file_handler)
    logger.setLevel(logging.DEBUG)
    return logger

def train(args):
    # load data
    data = VistDataLoader(args) 
    trg_vocab = data.vocab
    # initialize or load model
    if args.load_model_from is not None:
        print('load model from %s' % args.load_model_from)
        model = VistModel.load(args.load_model_from, args)
    else:
        model = VistModel(args, trg_vocab)
        print('uniformly initialize parameters [-%f, +%f]' % (args.uniform_init, args.uniform_init))
        model.uniform_init()

    # initialize loss
    vocab_mask = torch.ones(len(trg_vocab))
    vocab_mask[trg_vocab['<pad>']] = 0
    cross_entropy_loss = nn.CrossEntropyLoss(weight=vocab_mask, size_average=False)
    # initialize Trainer
    trainer = Trainer(args, model, cross_entropy_loss)
    # logging

    print('Begining training')
    train_iter = valid_num = 0
    begin_time = time.time()
    train_loss = cum_train_loss = trg_word = cum_trg_word = train_example = cum_train_example = 0.0
    hist_valid_scores = []
    save_model_list = []
    for epoch in range(args.max_epoch):
        for sample in VistDataLoader.data_iter(data.trn, trg_vocab, batch_size=args.batch_size, shuffle=True, cuda=args.cuda):
            train_iter += 1
            model.train_step = train_iter
            loss, log_outputs = trainer.train_step(sample, args.objective)

            train_loss += loss.item() * sample['num_trg_seq']
            cum_train_loss += loss.item() * sample['num_trg_seq']
            trg_word += sample['num_trg_word']
            cum_trg_word += sample['num_trg_word']
            train_example += sample['num_trg_seq']
            cum_train_example += sample['num_trg_seq']

            if train_iter % args.log_interval ==  0:
                print('epoch %d, iter %d, avg. loss %.6f, avg. ppl %.2f ' \
                      'example %d, time elapsed %.2f seconds' 
                      % (epoch, train_iter, 
                         train_loss / trg_word,
                         np.exp(train_loss / trg_word),
                         train_example,
                         time.time() - begin_time))
                # Reset the training log
                train_loss = trg_word = train_example = 0.0 

            # Perform validation
            if train_iter % args.valid_interval == 0:
                print('epoch %d, iter %d, cum. loss %.2f, cum. ppl %.2f ' \
                      'time elapsed %.2f seconds' % (epoch, train_iter, 
                                                     cum_train_loss / cum_train_example,
                                                     np.exp(cum_train_loss / cum_trg_word),
                                                     time.time() - begin_time))
                cum_train_loss = cum_trg_word = cum_train_example = 0.0 
                print('Begin validation ...')
                valid_num += 1
                dev_loss, dev_ppl = validation(args, trg_vocab, trainer, data.dev)
                dev_scores = compute_metric(args, trg_vocab, model, data.dev)
                tst_scores = compute_metric(args, trg_vocab, model, data.tst, train_iter)
                tst_output = " ".join("{}:{}".format(m, v) for m, v in tst_scores.items())
                print('Test: epoch %d, iter %d, dev. metric %s' % (epoch, train_iter, tst_output))
                if args.valid_metric == 'loss':
                    print('Validation: epoch %d, iter %d, dev. loss %.2f, dev. ppl %.2f' % (epoch, train_iter, dev_loss, dev_ppl))
                    dev_metric = -dev_loss
                else:
                    dev_output = " ".join("{}:{}".format(m, v) for m, v in dev_scores.items())
                    print('Validation: epoch %d, iter %d, dev. metric %s' % (epoch, train_iter, dev_output))
                    dev_metric = dev_scores[args.valid_metric]

                is_better = len(hist_valid_scores) == 0 or dev_metric > max(hist_valid_scores)
                is_better_than_last = len(hist_valid_scores) == 0  or dev_metric > hist_valid_scores[-1]
                hist_valid_scores.append(dev_metric)

                if valid_num > args.save_model_after:
                    if len(save_model_list) >= args.save_last_K_model:
                        old_model_file = args.save_model_to + '.iter%d.bin' % save_model_list[0]
                        print('delete old model ' + old_model_file)
                        os.system('rm -rf %s' % (old_model_file))
                        save_model_list = save_model_list[1:]
                    model_file = args.save_model_to + '.iter%d.bin' % train_iter
                    print('save model to [%s]' % model_file)
                    model.save(model_file)
                    save_model_list.append(train_iter)
                
                if (not is_better_than_last) and args.lr_decay:
                    lr = trainer.optimizer.param_groups[0]['lr'] * args.lr_decay
                    print('decay learning rate to %f' % lr)
                    trainer.optimizer.param_groups[0]['lr'] = lr
                
                if is_better:
                    patience = 0
                    best_model_iter = train_iter
                    if valid_num > args.save_model_after:
                        print('save the current best model ..')
                        model_file_abs_path = os.path.abspath(model_file)
                        symlin_file_abs_path = os.path.abspath(args.save_model_to + '.bin')
                        os.system('cp %s %s' % (model_file_abs_path, symlin_file_abs_path))
                else:
                    patience += 1
                    print('hit patience %d ' % patience)
                    if patience == args.patience:
                        print('early stop!')
                        print('the best model is from iteration [%d] ' % best_model_iter)
                        exit(0)
                
            
def validation(args, vocab, trainer, data):
    cum_loss = cum_trg_word = cum_example = 0.0
    for sample in VistDataLoader.data_iter(data, vocab, batch_size=args.batch_size, shuffle=False, is_test=True, cuda=args.cuda):
        loss, log_outputs = trainer.valid_step(sample)

        #word_loss_var = loss.item() / sample['num_trg_word']
        cum_loss += loss.item() * sample['num_trg_seq']
        cum_trg_word += sample['num_trg_word']
        cum_example += sample['num_trg_seq']
    return cum_loss / cum_example, np.exp(cum_loss / cum_trg_word)


def compute_metric(args, vocab, model, data, train_iter=None):
    cum_loss = cum_trg_word = cum_example = 0.0
    all_hyps = []
    all_refs = []
    all_albums = []
    all_ent_BLEU = []
    all_sent_pairs = []
    all_db_hyps = []
    
    start = time.time()    
    for sample in VistDataLoader.ref_iter(data, batch_size=args.batch_size, cuda=args.cuda):
        with torch.no_grad():
            raw_hyps, scores = model.generate(sample['src_seq'], sample['src_lengths'], beam_size=args.beam_size, decode_max_length=args.decode_max_length, to_word=True)
        hyps = [" ".join(" ".join(h[:-1]) for h in beam[0]) for beam in raw_hyps]
        all_hyps.extend(hyps)  # [batch, num_seq * decode_max_length]
        refs = sample['trg_seq']
        all_refs.extend(refs)
        all_albums.extend(sample['album'])
        all_db_hyps.extend([[h[:-1] for h in beam[0]] for beam in raw_hyps])

        src_img_entities = sample['src_img_entities']
        for story_ents, beam, ref in zip(src_img_entities, raw_hyps, refs):
            for i, (entities, h, r) in enumerate(zip(story_ents, beam[0], ref)):
                ent_bleu = model.compute_reward(h[:-1], [r[1:-1]], entities)
                all_ent_BLEU.append(ent_bleu)
                if i == 0:
                    prev = ''
                else:
                    prev = " ".join(beam[0][i-1][:-1]).strip().replace('<unk>', '[UNK]').replace('<pad>', '[PAD]') + ' '
                cur = " ".join(h[:-1]).strip().replace('<unk>', '[UNK]').replace('<pad>', '[PAD]')
                all_sent_pairs.append([prev, cur])
    print('decode all hyps using time ', time.time() - start)
    if args.save_decode_file is not None and train_iter is not None:
        with open(args.save_decode_file + '.iter' + str(train_iter), 'w') as f:
            for a, h in zip(all_albums, all_hyps):
                f.write('{}\t {}\n'.format(a, h))

    start = time.time()
    hyps = {a:[h] for a, h in zip(all_albums, all_hyps)}
    evaluator = AlbumEvaluator()
    score = evaluator.evaluate(data[2], hyps)
    print('scoring automatic metirc using time', time.time() - start)

    start = time.time()
    score['relevance'] = np.mean(all_ent_BLEU)
    print('Relevance', score['ent_BLEU'], ' using time ', time.time() - start)
    rl_reward = model.args.rl_reward.split('-')
    if 'coherence' in rl_reward:
        start = time.time()
        all_coherence = get_nsp(all_sent_pairs, model.bert_tokenizer, model.bert_nsp)
        score['coherence'] = np.mean(all_coherence)
        print('Coherence', np.mean(all_coherence), ' using time ', time.time() - start)
   # if 'expressiveness' in rl_reward:
   #     start = time.time()
   #     all_dBLEU = [0.0]
   #     #all_dBLEU = model.compute_expressiveness_reward(all_db_hyps[0:1000])
   #     score['expressiveness'] = np.mean(all_dBLEU)
   #     print('Expressiveness', score['dBLEU'], ' using time ', time.time() - start)

    return score


if __name__ == '__main__':
    parser = get_training_parser()
    parser = VistDataLoader.add_args(parser)
    parser = VistModel.add_args(parser)
    args = parser.parse_args()
    print('args', args)
    
    # seed the RNG
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    np.random.seed(int(args.seed * 13 / 7))

    if args.save_dir is not None:
        args.save_model_to = args.save_dir + '/model'
        args.save_decode_file = args.save_dir + '/decode-len100'
    
    if args.python_dir is not None:
        sys.path.append(args.python_dir)
        print('sys.path', sys.path)

    # print('args', args)
    print('Training Arguments')
    for arg in vars(args):
        print('\t{}\t:{}'.format(arg, getattr(args, arg)))
    train(args)
