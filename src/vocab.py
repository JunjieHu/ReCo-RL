from __future__ import print_function
import argparse
from collections import Counter
from itertools import chain

import torch
import sys, os
sys.path.append(os.path.abspath('./'))

def read_corpus(file_path, pad_bos_eos=False):
    data = []
    for line in open(file_path):
        sent = line.strip().split(' ')
        # only append <s> and </s> to the target sentence
        if pad_bos_eos:
            sent = ['<s>'] + sent + ['</s>']
        data.append(sent)

    return data


def read_bitext(file_path, delimiter="|||"):
    """ Read parallel text with the format: 'src ||| trg' """
    src_sents, trg_sents = [], []
    for line in open(file_path):
        src_trg = line.strip().split(delimiter)
        src = src_trg[0].strip().split(' ')
        trg = ['<s>'] + src_trg[1].strip().split(' ') + ['</s>']
        src_sents.append(src)
        trg_sents.append(trg)
    return src_sents, trg_sents



class VocabEntry(object):
    def __init__(self):
        self.word2id = dict()
        self.pad_id = 0
        self.bos_id = 1
        self.eos_id = 2
        self.unk_id = 3
        self.word2id['<pad>'] = 0
        self.word2id['<s>'] = 1
        self.word2id['</s>'] = 2
        self.word2id['<unk>'] = 3
        self.word2lemma = None
        self.word2pos = None
        self.id2word = {self.word2id[w]:w for w in self.word2id}
        # self.id2word = {v: k for k, v in self.word2id.iteritems()}

    def __getitem__(self, word):
        return self.word2id.get(word, self.unk_id)

    def __contains__(self, word):
        return word in self.word2id

    def __setitem__(self, key, value):
        raise ValueError('vocabulary is readonly')

    def __len__(self):
        return len(self.word2id)

    def __repr__(self):
        return 'Vocabulary[size=%d]' % len(self)

    def id2word(self, wid):
        return self.id2word[wid]

    def add(self, word):
        if word not in self:
            wid = self.word2id[word] = len(self)
            self.id2word[wid] = word
            return wid
        else:
            return self[word]

    @staticmethod
    def from_corpus(corpus, size, remove_singleton=True):
        """
        Args:
            corpus: list of list, eg. [['today', 'is', 'a', 'good', 'day'], ['good', 'morning']]
        """
        vocab_entry = VocabEntry()

        word_freq = Counter(chain(*corpus))
        non_singletons = [w for w in word_freq if word_freq[w] > 1]
        print('number of word types: %d, number of word types w/ frequency > 1: %d' % (len(word_freq),
                                                                                       len(non_singletons)))

        top_k_words = sorted(word_freq.keys(), reverse=True, key=word_freq.get)[:size]

        for word in top_k_words:
            if len(vocab_entry) < size:
                if not (word_freq[word] == 1 and remove_singleton):
                    vocab_entry.add(word)

        return vocab_entry

    @staticmethod
    def from_bilingual_corpus(src_corpus, trg_corpus, src_size, trg_size, remove_singleton=True):
        src_vocab_entry = VocabEntry()
        trg_vocab_entry = VocabEntry()
        vocab_entry = VocabEntry()

        src_word_freq = Counter(chain(*src_corpus))
        trg_word_freq = Counter(chain(*trg_corpus))

        non_src_singletons = [w for w in src_word_freq if src_word_freq[w] > 1]
        non_trg_singletons = [w for w in trg_word_freq if trg_word_freq[w] > 1]
        print('SRC: no. of word types: %d, no. of word types w/ frequency > 1: %d' % (len(src_word_freq), len(non_src_singletons)))
        print('TRG: no. of word types: %d, no. of word types w/ frequency > 1: %d' % (len(trg_word_freq), len(non_trg_singletons)))

        top_src_words = sorted(src_word_freq.keys(), reverse=True, key=src_word_freq.get)[:src_size]
        top_trg_words = sorted(trg_word_freq.keys(), reverse=True, key=trg_word_freq.get)[:trg_size*2]

        for word in top_src_words:
            if len(vocab_entry) < src_size:
                if not (src_word_freq[word] == 1 and remove_singleton):
                    vocab_entry.add(word)
                    src_vocab_entry.add(word)
        print('generate %d / %d source words' % (len(src_vocab_entry), len(src_word_freq)))

        for word in top_trg_words:
            # if word in src_vocab_entry:
            #     continue
            if len(trg_vocab_entry) < trg_size:
                if not (trg_word_freq[word] == 1 and remove_singleton):
                    vocab_entry.add(word)
                    trg_vocab_entry.add(word)
        print('generate %d / %d target words' % (len(trg_vocab_entry), len(trg_word_freq)))
        print('generate %d / %d shared words' % (len(vocab_entry), len(src_word_freq) + len(trg_word_freq)))
        return vocab_entry, src_vocab_entry, trg_vocab_entry

    @staticmethod
    def from_dict(w2id):
        vocab_entry = VocabEntry()
        vocab_entry.word2id = w2id
        vocab_entry.add('<pad>')
        vocab_entry.add('<s>')
        vocab_entry.add('</s>')
        vocab_entry.add('<unk>')
        vocab_entry.pad_id = vocab_entry['<pad>']
        vocab_entry.bos_id = vocab_entry['<s>']
        vocab_entry.eos_id = vocab_entry['</s>']
        vocab_entry.unk_id = vocab_entry['<unk>']
        vocab_entry.id2word = {vocab_entry[w]: w for w in vocab_entry.word2id}
        return vocab_entry

class Vocab(object):
    def __init__(self, src_sents=None, trg_sents=None, src_vocab_size=50000, trg_vocab_size=50000, remove_singleton=True, enc_share_vocab=False, dec_share_vocab=False):
        #assert len(src_sents) == len(trg_sents)
        if src_sents is not None and trg_sents is not None:
            print('initialize vocabulary ..')
            self.share, self.src, self.trg = VocabEntry.from_bilingual_corpus(src_sents, trg_sents, src_vocab_size, trg_vocab_size, remove_singleton=remove_singleton)
            if enc_share_vocab:
                self.src = self.share
            if dec_share_vocab:
                self.trg = self.share

            # if share_vocab:
            #     print('initialize share vocabulary ..')
            #     self.share, self.src_only, self.trg_only = VocabEntry.from_bilingual_corpus(src_sents, trg_sents, src_vocab_size, trg_vocab_size, remove_singleton=remove_singleton)
            #     self.src = self.trg = self.share
            # else:
            #     print('initialize source vocabulary ..')
            #     self.src = VocabEntry.from_corpus(src_sents, src_vocab_size, remove_singleton=remove_singleton)

            #     print('initialize target vocabulary ..')
            #     self.trg = VocabEntry.from_corpus(trg_sents, trg_vocab_size, remove_singleton=remove_singleton)

    def __repr__(self):
        return 'Vocab(source %d words, target %d words)' % (len(self.src), len(self.trg))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_vocab_size', default=50000, type=int, help='source vocabulary size')
    parser.add_argument('--trg_vocab_size', default=50000, type=int, help='target vocabulary size')
    parser.add_argument('--include_singleton', action='store_true', default=False, help='whether to include singleton'
                                                                                        'in the vocabulary (default=False)')

    parser.add_argument('--train_bitext', type=str, default='', help='file of parallel sentences')
    parser.add_argument('--train_src_file', type=str, help='path to the source side of the training sentences')
    parser.add_argument('--train_trg_file', type=str, help='path to the target side of the training sentences')
    parser.add_argument('--output', default='vocab.bin', type=str, help='output vocabulary file')
    parser.add_argument('--share_vocab', action='store_true', default=False)
    parser.add_argument('--enc_share_vocab', action='store_true', default=False)
    parser.add_argument('--dec_share_vocab', action='store_true', default=False)

    args = parser.parse_args()

    print('read in parallel sentences: %s' % args.train_bitext)
    if args.train_bitext:
        src_sents, trg_sents = read_bitext(args.train_bitext)
    else:
        src_sents = read_corpus(args.train_src_file, pad_bos_eos=False)
        trg_sents = read_corpus(args.train_trg_file, pad_bos_eos=False)

    vocab = Vocab(src_sents, trg_sents, args.src_vocab_size, args.trg_vocab_size,
                  remove_singleton=not args.include_singleton,
                  enc_share_vocab=args.enc_share_vocab,
                  dec_share_vocab=args.dec_share_vocab)
    print('generated vocabulary, source %d words, target %d words' % (len(vocab.src), len(vocab.trg)))

    torch.save(vocab, args.output)
    print('vocabulary saved to %s' % args.output)


