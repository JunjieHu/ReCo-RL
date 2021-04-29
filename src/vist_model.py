
from decoder import Decoder, LSTMDecoder
from StackedRNN import StackedLSTM
from decoder import repeat, topK_2d_ngrams, update_ngrams, select_hid, topK_2d, new_state, update_complete_hid, select_sequences_by_pointer, update_top_seqs
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from bert_nsp import get_nsp
from pytorch_pretrained_bert import BertTokenizer, BertForNextSentencePrediction
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from reward_function import compute_relevance_reward
from collections import Counter, defaultdict
import pickle
import sys, os
import copy
sys.path.append(os.path.abspath('./'))


class MeanEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, src_seq, src_lengths):
        """ Return the mean of the source sequence as state and cell"""
        return src_seq, (src_seq.mean(dim=1), src_seq.mean(dim=1))


class FeudalDecoder(Decoder):
    def __init__(self, vocab, embed_size=300, hidden_size=512, num_layers=1,
        dropout=0.5, attention=None, encoder_hidden_size=2048, tie_weight=False, empty_cell=False
        ):
        super().__init__(vocab)
        self.num_layers = num_layers
        self.dropout = dropout
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.img_size = encoder_hidden_size
        self.cur_manager_hidden = None
        self.tie_weight = tie_weight
        self.empty_cell = empty_cell

        # Embedding lookup table
        self.trg_embed = nn.Embedding(len(vocab), embed_size, padding_idx=vocab['<pad>'])
        # Project source embeddings to target hidden space
        self.src_linear = nn.Linear(encoder_hidden_size, hidden_size, bias=True)
        # manager LSTMCell
        self.manager_lstm = nn.LSTMCell(hidden_size + 2 * encoder_hidden_size, hidden_size)
        self.manager_linear = nn.Linear(hidden_size + 2 * encoder_hidden_size, hidden_size)
        print('worker input size=', embed_size + hidden_size + encoder_hidden_size)
        print('worker output size = ', hidden_size)
        # self.manager_lstm = StackedLSTM(num_layers, hidden_size, hidden_size, dropout=dropout)
        # worker LSTMCell
        self.worker_init_linear = nn.Linear(encoder_hidden_size + hidden_size, hidden_size, bias=True)
        self.worker_lstm = nn.LSTMCell(embed_size + hidden_size + encoder_hidden_size, hidden_size)
        # self.worker_lstm = StackedLSTM(num_layers, hidden_size, hidden_size, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        # prediction layer of the target vocabulary
        # self.readout = nn.Linear(hidden_size, len(vocab), bias=False)
        if self.tie_weight:
            print('tie the weight of output layer with word embedding weight')
            self.readout_weight = Parameter(torch.Tensor(hidden_size, embed_size))
            self.readout_bias = Parameter(torch.Tensor(len(vocab)))
            self.readout = self.readout_tie_weight
        else:
            self.readout = nn.Linear(hidden_size, len(vocab), bias=False)

    def init_state(self, final_src_ctx, linear=None):
        src_ctx_linear = linear(final_src_ctx[0]) if linear is not None else final_src_ctx[0]
        init_state = F.tanh(src_ctx_linear)
        if self.empty_cell:
            init_cell = init_state.data.new(init_state.size()).zero_()
        else:
            init_cell = src_ctx_linear
        return (init_state, init_cell)

    def readout_tie_weight(self, input):
        return input.matmul(self.readout_weight).matmul(self.trg_embed.weight.t()) + self.readout_bias

    def forward(self, src_hidden, final_src_ctx, trg_seq):
        """
        Args:
            src_hidden: [batch, num_seq(5), D]
            final_src_ctx: [batch, D]
            trg_seq: [batch, num_seq(5), trg_len]
        Return:
            scores: [batch, num_seq(5), trg_len, vocab_size]
        """

        batch_size = src_hidden.size(0)
        new_tensor = src_hidden.data.new

        img_mean = final_src_ctx[0]
        manager_init_state = F.tanh(self.src_linear(img_mean))
        manager_init_cell = manager_init_state.data.new(manager_init_state.size()).zero_()
        manager_hidden = (manager_init_state, manager_init_cell)     # [batch, hidden_size]

        trg_seq_embed = self.trg_embed(trg_seq)  # [batch, num_seq, trg_len, embed_size]
        trg_seq_embed = self.dropout(trg_seq_embed)
        # print('trg_seq_embed', trg_seq_embed.size())
        scores = []
        worker_hm1 = new_tensor(batch_size, self.hidden_size).zero_()
        for trg_word_embed, src_img_embed in zip(trg_seq_embed.split(split_size=1, dim=1), src_hidden.split(split_size=1, dim=1)):
            trg_word_embed = trg_word_embed.squeeze(1)   # [batch, trg_len, embed_size]
            src_img_embed = src_img_embed.squeeze(1)     # [batch, hidden_size]

            # Manager produces the next goal
            manager_x = torch.cat([img_mean, src_img_embed, worker_hm1], dim=1)  # [batch, hidden*3]
            manager_hidden = self.manager_lstm(manager_x, manager_hidden)
            manager_hidden = (self.dropout(manager_hidden[0]), self.dropout(manager_hidden[1]))
            # Initialize the worker's hidden state
            worker_h0 = torch.cat([src_img_embed, manager_hidden[0]], dim=1)
            worker_init_state = F.tanh(self.worker_init_linear(worker_h0))
            worker_init_cell = worker_init_state.data.new(worker_init_state.size()).zero_()
            worker_hidden = (worker_init_state, worker_init_cell)

            trg_word_scores = []
            # start from <s>, util y_{T-1}
            for y_tm1_embed in trg_word_embed.split(split_size=1, dim=1):
                x = torch.cat([y_tm1_embed.squeeze(1), worker_h0], dim=1)
                # Go through LSTMCell to get h_t [batch_size, hidden_size]
                worker_hidden = self.worker_lstm(x, worker_hidden)
                worker_hidden = (self.dropout(worker_hidden[0]), self.dropout(worker_hidden[1]))
                h_t = worker_hidden[0]
                h_t = F.tanh(h_t)
                score_t = self.readout(h_t)
                trg_word_scores.append(score_t)

            trg_word_scores = torch.stack(trg_word_scores, dim=1)  # [batch, trg_len, vocab_size]
            scores.append(trg_word_scores)
            worker_hm1 = worker_hidden[0]
        scores = torch.stack(scores, dim=1)   # [batch, num_seq, trg_len, vocab_size]
        # print('scores', scores.size())
        return scores

    def decode_one_step(self, input, hidden, src_hidden, ctx_vec, beam_size=None):
        """
        Args:
            input: [batch]
            ctx_vec: [batch, d]
        Return:
            log_prob: [batch, |V|]
        """
        trg_embed = self.trg_embed(input)  #[batch, D]
        x = torch.cat([trg_embed, ctx_vec], dim=-1)  # [batch, D+d]
        worker_hidden = self.worker_lstm(x, hidden)
        h_t = F.tanh(worker_hidden[0])
        score_t = self.readout(h_t)     # [batch, |V|]
        log_prob = F.log_softmax(score_t, dim=-1) # [batch, |V|]
        return log_prob, worker_hidden, ctx_vec

    def sequence_beam_search(self, src_hidden, final_src_ctx, beam_size=5, decode_max_length=100, to_word=True, avoid_ngram=2, decode_len_constraint=13):
        """
        Args:
            src_hidden: [batch, src_len]
            final_src_ctx: [batch, hidden_size]
        Return:
            completed_sequences: [batch, beam_size, num_seq, decode_max_length]
            completed_scores: Tensor, [batch, beam_size, num_seq]
        """
        batch_size = src_hidden.size(0)
        num_seq = src_hidden.size(1)
        new_tensor = src_hidden.data.new

        img_mean = final_src_ctx[0]
        img_mean = repeat(img_mean, dim=1, k=beam_size) # [batch*beam_size, encoder_hidden_size]
        manager_hidden = self.init_state(final_src_ctx, self.src_linear)
        manager_hidden = repeat(manager_hidden, dim=1, k=beam_size)

        completed_sequences, completed_scores = [], []
        top_score = new_tensor(batch_size, beam_size).fill_(-float('inf'))  #[batch, beam]
        top_score.data[:, 0].fill_(0)
        ngrams = [[defaultdict(list) for _ in range(beam_size)] for _ in range(batch_size)] if avoid_ngram > 0 else None

        worker_hm1 = new_tensor(batch_size * beam_size, self.hidden_size).zero_()
        i = 0
        for src_img_embed in src_hidden.split(split_size=1, dim=1):
            src_img_embed = src_img_embed.squeeze(1)
            src_img_repeat = repeat(src_img_embed, dim=1, k=beam_size) # [batch*beam_size, img_size]

            manager_x = torch.cat([img_mean, src_img_repeat, worker_hm1], dim=1)  # [batch*beam_size, 2*img_size+hid_size]
            manager_hidden = self.manager_lstm(manager_x, manager_hidden) # [batch*beam_size, hidden_size]

            worker_h0 = torch.cat([src_img_repeat, manager_hidden[0]], dim=1)  # [batch*beam, img_size+hid_size]
            worker_init_state = self.worker_init_linear(worker_h0)   # [batch*beam, hid_size] no tanh yet
            worker_hidden = (worker_init_state, worker_init_state)

            # sequence: [batch, beam_size, decode_max_length], score: [batch, beam_size]
            sequence, score, worker_hidden, ngrams, beam_bptr = self.beam_search(
                src_hidden=src_hidden, final_src_ctx=worker_hidden,
                ctx_vec=worker_h0, top_score=top_score,
                repeat_hidden=False, repeat_ctx=False,
                beam_size=beam_size, decode_max_length=decode_max_length,
                to_word=to_word, ngrams=ngrams, avoid_ngram=avoid_ngram, decode_len_constraint=decode_len_constraint)

            # add the sequence to the correct completed_sequences by the beam_bptr
            completed_sequences, completed_scores = select_sequences_by_pointer(completed_sequences, completed_scores, sequence, score, beam_bptr)
            top_score = new_tensor(completed_scores).mean(dim=2).float()
            worker_hm1 = worker_hidden[0]
            i += 1
        return completed_sequences, completed_scores

    def sequence_sample(self, src_hidden, final_src_ctx, sample_size=5, decode_max_length=100, to_word=True, sample_method='random'):
        """
        Args:
            src_hidden: [batch, src_len]
            final_src_ctx: [batch, hidden_size]
        Return:
            sequences: [batch, sample_size, num_seq, decode_max_length]
            scores: [batch, sample_size, num_seq]
            manager_xs: [batch, sample_size, 2*enc_hidden_size + hidden_size]
            manager_hs: [batch, sample_size, hidden_size]
        """
        batch_size = src_hidden.size(0)
        num_seq = src_hidden.size(1)
        new_tensor = src_hidden.data.new

        img_mean = final_src_ctx[0]
        img_mean = repeat(img_mean, dim=1, k=sample_size) # [batch*beam_size, encoder_hidden_size]

        manager_hidden = self.init_state(final_src_ctx, self.src_linear)
        manager_hidden = repeat(manager_hidden, dim=1, k=sample_size)
        manager_inputs = []
        manager_goals = []

        completed_sequences, completed_scores = [], []
        worker_hm1 = new_tensor(batch_size * sample_size, self.hidden_size).zero_()
        for src_img_embed in src_hidden.split(split_size=1, dim=1):
            src_img_embed = src_img_embed.squeeze(1)
            src_img_repeat = repeat(src_img_embed, dim=1, k=sample_size) # [batch*beam_size, hidden_size]

            manager_x = torch.cat([img_mean, src_img_repeat, worker_hm1], dim=1)  # [batch*beam_size, 2*hidden_size]
            manager_hidden = self.manager_lstm(manager_x, manager_hidden) # [batch*beam_size, hidden_size]

            # Collect manager inputs
            manager_inputs.append(manager_x.view(batch_size, sample_size, -1))
            manager_goals.append(manager_hidden[0].view(batch_size, sample_size, -1))

            worker_h0 = torch.cat([src_img_repeat, manager_hidden[0]], dim=1)
            # print('worker_h0, ctx', worker_h0.size())
            worker_init_state = self.worker_init_linear(worker_h0)
            worker_hidden = (worker_init_state, worker_init_state)
            # sequence: [batch, beam_size, decode_max_length], score: [batch, beam_size, decode_max_length]
            # worker_hidden: [batch, beam_size, hidden_size]
            sequence, score, worker_hidden = self.sample(
                src_hidden=src_hidden, final_src_ctx=worker_hidden,
                ctx_vec=worker_h0, repeat_hidden=False, repeat_ctx=False,
                sample_size=sample_size, decode_max_length=decode_max_length,
                to_word=to_word, sample_method=sample_method, detached=True)

            completed_sequences.append(sequence)   # [batch, sample_size, decode_max_length]
            completed_scores.append(score)         # [batch, sample_size, decode_max_length]
            worker_hm1 = worker_hidden[0].view(-1, worker_hidden[0].size(-1))  # [batch*beam_size, hidden_size]
        manager_inputs = torch.stack(manager_inputs, dim=2)
        manager_goals = torch.stack(manager_goals, dim=2)

        # Augment the score tensor to have the same sequence length
        scores = completed_scores

        # sequences: [batch, sample_size, num_seq, decode_max_length]
        sequences = []
        for i in range(batch_size):
            beams = []
            for k in range(sample_size):
                beam_k = []
                for j in range(num_seq):
                    beam_k.append(completed_sequences[j][i][k])
                beams.append(beam_k)
            sequences.append(beams)

        return sequences, scores, manager_inputs, manager_goals

    def beam_search(self, src_hidden, final_src_ctx, ctx_vec, top_score=None, repeat_hidden=True, repeat_ctx=False, beam_size=5, decode_max_length=100, to_word=True, ngrams=None, avoid_ngram=2, decode_len_constraint=13):
        """
        Args:
            src_hidden: [batch, src_seq_len, hidden_size]
            final_src_ctx: tensor or tuple of tensor [batch, hidden_size], used to initialize decoder hidden state
            ctx_vec: [batch, hidden_size], context vector that adds to the input at each decoding step
        Return:
            sequence: [batch, beam_size, decode_max_length]
            sort_score: [batch, beam_size]
            complete_hidden: [batch*beam_size, hidden_size]
            ngram:
            beam_bptr: [batch, beam_size]
        """
        batch_size = src_hidden.size(0)
        num_vocab = len(self.vocab)
        new_tensor = src_hidden.data.new
        pad_id = self.vocab['<pad>']
        eos_id = self.vocab['</s>']
        bos_id = self.vocab['<s>']
        unk_id = self.vocab['<UNK>']
        per_id = self.vocab['.']
        exc_id = self.vocab['!']
        sem_id = self.vocab[';']

        # repeat `src_hidden`, `pad_mask` for k times
        dec_input = new_tensor(batch_size * beam_size).long().fill_(bos_id)  # [batch*beam_size]
        dec_hidden = self.init_state(final_src_ctx)  # [batch, hidden_size]
        hidden_size = dec_hidden[0].size(-1)  # LSTM

        # repeat dec_hidden from [batch, hidden] to [batch*beam, hidden]
        if repeat_hidden:
            dec_hidden = repeat(dec_hidden, dim=1, k=beam_size)  # [batch*beam_size, hidden]
        complete_hidden = new_state(dec_hidden)

        # repeat ctx_vec from [batch, hidden] to [batch*beam, hidden]
        if repeat_ctx:
            ctx_vec = repeat(ctx_vec, dim=1, k=beam_size)  # [batch*beam_size, hidden_size]

        batch_id = torch.LongTensor(range(batch_size))[:, None].repeat(1, beam_size).type_as(dec_input.data)  # [batch, beam] [[0,..0], [1,..1]..]
        if top_score is None:
            top_score = new_tensor(batch_size, beam_size).fill_(-float('inf'))  #[batch, beam]
            top_score.data[:, 0].fill_(0)

        # [1, |V|]
        pad_vec = new_tensor(1, num_vocab).float().fill_(-float('inf'))
        pad_vec[0, pad_id] = 0
        eos_vec = new_tensor(1, num_vocab).float().fill_(-float('inf'))
        eos_vec[0, eos_id] = 0
        end_mark = new_tensor(batch_size, beam_size).byte().fill_(0)

        top_colid = new_tensor(batch_size, beam_size).long().fill_(bos_id)
        eos_mask = new_tensor(batch_size, beam_size).byte().fill_(0)  # [batch, beam_size]
        sent_len = new_tensor(batch_size, beam_size).float().fill_(0)
        beam_bptr = torch.LongTensor(range(beam_size))[None, :].repeat(batch_size, 1).type_as(dec_input.data)  # [batch, beam]
        # print('init beam_bptr', beam_bptr)

        # top_rowids, top_colids = [], []
        top_seqs = [[[bos_id] for _ in range(beam_size)] for _ in range(batch_size)]  # [batch, beam, num_words]

        if ngrams is None:
            ngrams = [[defaultdict(list) for _ in range(beam_size)] for _ in range(batch_size)] # [batch, beam]'s dict: key= n-1 gram, value n-th word list
        # else:
        #     # for ng in ngrams:
        #     new_ngrams = [[ng.copy() for _ in range(beam_size)] for ng in ngrams]
        #     ngrams = new_ngrams


        for i in range(decode_max_length):
            # [batch*beam_size, |V|]
            log_prob, dec_hidden, ctx_vec = self.decode_one_step(dec_input, dec_hidden, src_hidden, ctx_vec, beam_size)
            log_prob = log_prob.view(batch_size, beam_size, num_vocab)  # [batch, beam_size, |V|]

            log_prob.data[:,:,unk_id].fill_(-float('inf'))

            # if the sequence has already produce <eos> token, add <pad> token to the end
            if eos_mask is not None and eos_mask.sum() > 0:
                log_prob.data.masked_scatter_(eos_mask.unsqueeze(2), pad_vec.expand((eos_mask.sum(), num_vocab))) # [batch]

            # if the sequence is longer than 20 and the last word is . or ! predict </s>
            if i > 1 and end_mark.sum() > 0:
                log_prob.data.masked_scatter_(end_mark.unsqueeze(2), eos_vec.expand((end_mark.sum(), num_vocab)))

            # score1 = top_score.unsqueeze(2) + log_prob   # [batch, beam_size, |V|]
            # ss1, sidx1 = score1.topk(k=beam_size, dim=1)

            # mask repeated n-gram for some words
            if avoid_ngram > 0:
                nm1_grams = []
                log_mask = [[[0 for _ in range(num_vocab)] for _ in range(beam_size)] for _ in range(batch_size)]
                for ii, beam_seq in enumerate(top_seqs):  # batch
                    for jj, seq in enumerate(beam_seq):   # beam
                        nm1_gram = tuple(seq[-(avoid_ngram-1):])
                        exist_words = ngrams[ii][jj][nm1_gram]  # a list of existing n-th word
                        for widx in exist_words:
                            log_mask[ii][jj][widx] = 1
                log_mask = new_tensor(log_mask).byte()
                log_prob.data.masked_fill_(log_mask, -float('inf'))

            # if a sentence is too short, force it to continue
            if (sent_len < 5).sum() > 0:
                log_prob[:,:,eos_id].fill_(-float('inf'))

            if i == 0:
                # print('i=0, skip')
                score = top_score.unsqueeze(2) + log_prob
            else:
                # print('LN: i=', i)
                score = ((top_score * sent_len).unsqueeze(2) + log_prob) / (sent_len + 1 - eos_mask.float()).unsqueeze(2)

            # top_score, top_rowid, top_colid, ngrams, top_seqs = topK_2d_ngrams(score, ngrams, top_seqs, avoid_ngram)
            top_score, top_rowid, top_colid = topK_2d(score) # [batch, beam_size]
            # top_rowids.append(top_rowid)    # select the beam id
            # top_colids.append(top_colid)    # select the word id

            # update the curent hidden state
            dec_hidden = select_hid(dec_hidden, batch_id, top_rowid)  # [batch*beam_size, hidden_size]
            dec_input = top_colid.view(-1)  # [batch_size * beam_size]

            # update the eos mask
            # print(i, '-iter bptr')
            # print('before eos', eos_mask)
            eos_mask = eos_mask.gather(dim=1, index=top_rowid.data) | top_colid.data.eq(eos_id)
            # print('after eos', eos_mask)
            # print('top_rowid', top_rowid)

            # update complete hidden state, if a sentence in a beam is complete,
            # then do not update the complete hidden state
            complete_hidden = select_hid(complete_hidden, batch_id, top_rowid)
            complete_hidden = update_complete_hid(complete_hidden, eos_mask, dec_hidden)

            # update beam-back-pointer
            beam_bptr = beam_bptr.gather(dim=1, index=top_rowid.data)

            # update top sequence and add top_colid
            top_seqs = update_top_seqs(top_seqs, top_rowid.tolist(), top_colid.tolist())  # [batch, beam, num_words]

            # update ngrams
            new_ngrams = []
            for ii, (bi_bptr, bi_ngram) in enumerate(zip(top_rowid.tolist(), ngrams)):
                new_bi_ngrams = []
                for jj, bptr in enumerate(bi_bptr):
                    ngram_copy = defaultdict(list, {k:[w for w in v] for k, v in bi_ngram[bptr].items()})
                    # print(f'ii={ii}, jj={jj}, bptr={bptr}', bi_ngram[bptr])
                    # ngram_copy = defaultdict(list)
                    # for nm1gram, next_words in bi_ngram[bptr].items():
                    #     ngram_copy[nm1gram] = [w for w in next_words]
                    new_bi_ngrams.append(ngram_copy)
                    # new_bi_ngrams.append(copy.deepcopy(bi_ngram[bptr]))  # too slow to use deepcopy
                new_ngrams.append(new_bi_ngrams)
            ngrams = new_ngrams

            if avoid_ngram > 0:
                for ii, bi_seqs in enumerate(top_seqs):
                    for jj, seqs in enumerate(bi_seqs):
                        if seqs[-1] == pad_id or seqs[-1] == eos_id:
                            continue
                        nm1_ngram = tuple(seqs[-avoid_ngram:-1])
                        # print(f'ii={ii}, jj={jj}, n-1-gram={nm1_ngram}, seqs[-1]={seqs[-1]}', ngrams[ii][jj][nm1_ngram])
                        ngrams[ii][jj][nm1_ngram].append(seqs[-1])
            # update sentence length
            sent_len = sent_len.gather(dim=1, index=top_rowid.data) + (1 - eos_mask.float())
            end_mark = (sent_len > decode_len_constraint) & (top_colid.eq(per_id) | top_colid.eq(exc_id) | top_colid.eq(sem_id)) & (1 - eos_mask)

            #----DEBUG----
            # end_mark_list = end_mark.tolist()
            # print(f'================\n\n decode {i}-th word')
            # for ii, bi_seqs in enumerate(top_seqs):
            #     print('\n\n---------------------\n batch id:', ii)
            #     for jj, seq_jj in enumerate(bi_seqs):
            #         print('\n  +++++++\n beam id: ', jj)
            #         print('  seq_jj', seq_jj)
            #         print('  sent len', sent_len.tolist()[ii][jj])
            #         print('  cur seq:' + ' '.join([self.vocab.id2word[wi] for wi in seq_jj]))
            #         if end_mark_list[ii][jj] == 1:
            #             print('NOTE!!END', end_mark_list[ii][jj])
            #         else:
            #             print('NOTYET', end_mark_list[ii][jj])
            #         print('  prev beam:', top_rowid.tolist()[ii][jj])
            #         print('  cur word:', self.vocab.id2word[top_colid.tolist()[ii][jj]])
            #         for k, v in ngrams[ii][jj].items():
            #             # print(' n-1 grams:', ' '.join(self.vocab.id2word[wi] for wi in list(k)))
            #             # print(' n-th words', [self.vocab.id2word[wi] for wi in v])
            #             for w in v:
            #                 print('     ngrams:', ' '.join(self.vocab.id2word[wi] for wi in list(k) + [w]))


            # print('beam_bptr', beam_bptr, '\n-----------\n\n')


            if eos_mask.sum() == batch_size * beam_size:
                break

        # tokens = []
        # for i in reversed(range(len(top_colids))):
        #     if i == len(top_colids) - 1:
        #         sort_score, sort_idx = torch.sort(top_score, dim=1, descending=True)  # [batch_size, beam_size]
        #         # print('sort_score', sort_score.size())
        #         # sort_score, sort_idx = sort_score[:, :N], sort_idx[:, :N]
        #     else:
        #         sort_idx = top_rowids[i+1].gather(dim=1, index=sort_idx)
        #     token = top_colids[i].gather(dim=1, index=sort_idx)

        #     tokens.insert(0, token)
        #     # print('i', i, 'token', token.size)
        # sequence = torch.stack(tokens, dim=2)  # [batch, beam_size, decode_max_lengths]
        # sequence = sequence.cpu().data.numpy().tolist()

        sequence = top_seqs
        sort_score = top_score
        if to_word:
            # print('sequence', sequence.size())
            # sequence = sequence.cpu().data.numpy().tolist()
            sequence = [[[self.vocab.id2word[w] for w in seq if w != self.vocab['<pad>']] for seq in beam] for beam in sequence]
            # print('one sentence', sequence[0][0])
            sort_score = sort_score.cpu().data.numpy().tolist()

        # print('batch')
        # for i, mseq in enumerate(sequence):
        #     print('batch', i)
        #     for j, seq in enumerate(mseq):
        #         print(' beam-%d: %s' % (j, ' '.join(seq)))

        # TODO
        # dec_hidden_h = dec_hidden[0].contiguous().view(batch_size, beam_size, hidden_size)
        # dec_hidden_c = dec_hidden[1].contiguous().view(batch_size, beam_size, hidden_size)
        # dec_hidden = (dec_hidden_h, dec_hidden_c)
        return sequence, sort_score, complete_hidden, ngrams, beam_bptr

class VistModel(nn.Module):
    def __init__(self, args, vocab):
        super().__init__()
        self.args = args
        # self.encoder = encoder
        # self.decoder = decoder
        self.encoder = MeanEncoder()
        self.decoder = FeudalDecoder(
            vocab, args.embed_size, args.hidden_size,
            args.num_layers, args.dropout, None,
            args.encoder_hidden_size, args.tie_weight, True
        )

        self.train_step = 0
        if args.cuda:
            print('use GPU in Model')
            self.encoder = self.encoder.cuda()
            self.decoder = self.decoder.cuda()
        self.bert_tokenizer = self.bert_nsp = None

    @staticmethod
    def add_args(parser):
        parser.add_argument('--embed_size', default=300, type=int)
        parser.add_argument('--hidden_size', default=512, type=int)
        parser.add_argument('--num_layers', default=1, type=int)
        parser.add_argument('--dropout', default=0.5, type=float)
        parser.add_argument('--encoder_hidden_size', default=2048, type=int)
        parser.add_argument('--tie_weight',  action='store_true', default=False)
        parser.add_argument('--empty_cell', action='store_true', default=False)
        parser.add_argument('--reward_alpha', default=1, type=float)
        parser.add_argument('--hrl', default=False, action='store_true')
        parser.add_argument('--ngram', default=2, type=int)
        parser.add_argument('--rl_baseline', default='greedy', type=str)
        parser.add_argument('--rl_weight', default=0.5, type=float)
        parser.add_argument('--avoid_ngram', default=2, type=int)
        parser.add_argument('--new_vocab', default=None, type=str)
        parser.add_argument('--rl_bleu', default=0.5, type=float)
        parser.add_argument('--rl_f1', default=0.5, type=float)
        parser.add_argument('--rl_relevance', default=0.5, type=float)
        parser.add_argument('--rl_relevance_weight', default='0.4-0.3-0.2-0.1', type=str)
        parser.add_argument('--rl_relevance_beta', default=5, type=float)
        parser.add_argument('--rl_coherence', default=0.5, type=float)
        parser.add_argument('--rl_expressiveness', default=0.5, type=float)
        parser.add_argument('--bert_weight_path', default='bert-base-uncased', type=str)
        parser.add_argument('--bert_vocab_path', default='bert-base-uncased', type=str)
        parser.add_argument('--decode_len_constraint', default=13, type=int)
        return parser

    def save(self, path):
        params = {
            'args': self.args,
            'encoder_state_dict': self.encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'decoder_vocab': self.decoder.vocab
        }
        torch.save(params, path)

    @staticmethod
    def load(path, new_args=None):
        params = torch.load(path, map_location=lambda storage, loc: storage)
        args = params['args']
        print('new args', new_args)

        args.rl_baseline = new_args.rl_baseline
        args.hrl = new_args.hrl
        args.ngram = new_args.ngram
        args.rl_weight = new_args.rl_weight
        args.reward_alpha = new_args.reward_alpha
        args.avoid_ngram = new_args.avoid_ngram
        args.optim = new_args.optim
        args.cuda = new_args.cuda
        args.rl_bleu = new_args.rl_bleu
        args.rl_f1 = new_args.rl_f1
        args.rl_relevance = new_args.rl_relevance
        args.rl_reward = new_args.rl_reward
        args.rl_relevance_weight = new_args.rl_relevance_weight
        args.rl_relevance_beta = new_args.rl_relevance_beta
        args.rl_expressiveness = new_args.rl_expressiveness
        args.rl_coherence = new_args.rl_coherence
        args.bert_weight_path = new_args.bert_weight_path
        args.bert_vocab_path = new_args.bert_vocab_path
        args.decode_len_constraint = new_args.decode_len_constraint
        if new_args.new_vocab is not None:
            print('loading new vocab from ' + new_args.new_vocab)
            vocab = torch.load(new_args.new_vocab)
        else:
            vocab = params['decoder_vocab']
        # model = VistModel.build_model(args, vocab)
        model = VistModel(args, vocab)
        try:
            model.encoder.load_state_dict(params['encoder_state_dict'])
        except KeyError:
            print('****Warming loading stat dict missing parameters ****')

        try:
            print('decoder params', params['decoder_state_dict'].keys())
            model.decoder.load_state_dict(params['decoder_state_dict'], strict=False)
        except KeyError:
            print('****Warming loading stat dict missing parameters ****')
            params_name = params['decoder_state_dict'].keys()
            for n, p in model.named_parameters():
                if n not in params_name:
                    print('uniformly initial new parameters %s in [-%f,%f]' % (n, -args.uniform_init, args.uniform_init))
                    p.data.uniform_(-args.uniform_init, args.uniform_init)
        if 'coherence' in args.rl_reward:
            model.bert_tokenizer = BertTokenizer.from_pretrained(new_args.bert_vocab_path)
            model.bert_nsp = BertForNextSentencePrediction.from_pretrained(new_args.bert_weight_path)
            model.bert_nsp.eval()

        return model

    def uniform_init(self):
        # print('uniformly initialize parameters [-%f, +%f]' % (self.args.uniform_init, self.args.uniform_init))
        for p in self.parameters():
            p.data.uniform_(-self.args.uniform_init, self.args.uniform_init)

    def generate(self, src_seq, src_lengths, beam_size=5, decode_max_length=100, to_word=True, decode_type='beam'):
        src_hidden, final_src_ctx = self.encoder(src_seq, src_lengths)
        if decode_type == 'beam':
            completed_sequences, completed_scores = self.decoder.sequence_beam_search(src_hidden, final_src_ctx, beam_size, decode_max_length, to_word, self.args.avoid_ngram, decode_len_constraint=self.args.decode_len_constraint)
        else:
            completed_sequences, completed_scores = self.decoder.sequence_greedy_decode(src_hidden, final_src_ctx, decode_max_length, to_word)
        return completed_sequences, completed_scores

    def sample(self, src_seq, src_lengths, sample_size, decode_max_length):
        src_hidden, final_src_ctx = self.encoder(src_seq, src_lengths)
        completed_sequences, completed_scores, manager_inputs, manager_goals = self.decoder.sequence_sample(src_hidden, final_src_ctx, sample_size, decode_max_length, to_word=True)
        return completed_sequences, completed_scores, manager_inputs, manager_goals

    def reinforce(self, sample, sample_size, decode_max_length, rl_reward='BLEU'):
        avg_loss = self.worker_reinforce(sample, sample_size, decode_max_length, rl_reward)
        return avg_loss

    def worker_reinforce(self, sample, sample_size, decode_max_length, rl_reward='BLEU'):
        """
        Args:
            src_seq: [batch, src_len]
            trg_seq: [batch, num_seq, trg_len]
        """
        src_seq, src_lengths, trg_seq = sample['src_seq'], sample['src_lengths'], sample['trg_seq']
        entities = sample['src_img_entities']
        src_img_ids = sample['src_img_ids']
        # Encode
        src_hidden, final_src_ctx = self.encoder(src_seq, src_lengths)

        # Sample, [batch, sample_size, num_seq, decode_max_length]
        sequences, scores, _, _ = self.decoder.sequence_sample(src_hidden, final_src_ctx, sample_size, decode_max_length, to_word=True)

        # Compute reward and baseline
        rl_reward = rl_reward.split('-')
        batch_size = len(src_seq)
        num_seq = len(sequences[0][0])
        rewards = [[[0.0 for _ in range(num_seq)] for _ in range(sample_size)] for _ in range(batch_size)]
        baseline = [[0.0 for _ in range(num_seq)] for _ in range(batch_size)]
        # Compute baseline (self-critic) [batch, beam_size, num_seq, decode_max_length]
        if self.args.rl_baseline == 'greedy':
            greedy_sequences, _ = self.decoder.sequence_beam_search(src_hidden, final_src_ctx, 1, decode_max_length, True)
            greedy_baseline = [[0 for _ in range(num_seq)] for _ in range(batch_size)]

        sent_idxs = []
        sent_pairs = []
        for i in range(batch_size):
            for k in range(num_seq):
                ref = trg_seq[i][k][1:].data.tolist()
                ref = [self.decoder.vocab.id2word[w] for w in ref]  # remove <s> </s>
                ref = [w for w in ref if w != '</s>' and w != '<pad>']

            for j in range(sample_size):
                ngrams = set()
                for k in range(num_seq):
                    hyp = sequences[i][j][k][1:-1] # remove <s> </s>
                    ref = trg_seq[i][k][1:].data.tolist()
                    ref = [self.decoder.vocab.id2word[w] for w in ref]  # remove <s> </s>
                    ref = [w for w in ref if w != '</s>' and w != '<pad>']
                    bleu_reward, div_reward, dBLEU_reward, ent_bleu = 0, 0, 0, 0
                    if 'BLEU' in rl_reward:
                        bleu_reward = sentence_bleu([ref], hyp, smoothing_function=SmoothingFunction().method3)
                    if 'expressiveness' in rl_reward:
                        if k == 0:
                            expressiveness_reward = 1
                        else:
                            pre_hs = [sequences[i][j][s][1:-1] for s in range(0, k)]
                            # print('pre_hs', pre_hs, k, list(range(0, k)))
                            dBLEU = sentence_bleu(pre_hs, hyp, smoothing_function=SmoothingFunction().method3)
                            expressiveness_reward = 1 - dBLEU
                    if 'relevance' in rl_reward:
                        #hyp, refs, entities, vocab, beta=5, weights=(0.7, 0.3)
                        weights = [float(ww) for ww in self.args.rl_relevance_weight.split('-')]
                        ent_bleu, _ = compute_relevance_reward(hyp, [ref], entities[i][k], self.decoder.vocab, beta=self.args.rl_relevance_beta, weights=weights)
                        # print('hyp: ', hyp)
                        # print('ref: ', ref)
                        # print('ent_bleu', ent_bleu)
                    if 'coherence' in rl_reward:
                        if k == 0:
                            sent_pair = ["", " ".join(hyp)]
                        else:
                            prev_hyp = sequences[i][j][k-1][1:-1]
                            sent_pair = [" ".join(prev_hyp), " ".join(hyp)]
                        # print(i, j, k, 'sent_pair', sent_pair)
                        sent_pairs.append(sent_pair)
                        sent_idxs.append((i,j,k))
                    # stat = " ".join(['p%d=%.3f' % (pi+1, p) for pi, p in enumerate(stat)])
                    # print('    s-%d: bleu=%.4f, stat=%s, div=%.4f, dbleu=%.4f, nll=%.4f, %s' % (k, bleu_reward, stat, div_reward, expressiveness_reward, scores[k][i][j].sum().item()/(1e-10+len(hyp)), " ".join(sequences[i][j][k])))
                    # print('          nll={}'.format(scores[k][i][j].tolist()))
                    #rewards[i][j][k] = self.args.rl_bleu * bleu_reward + div_reward + expressiveness_reward + self.args.rl_f1 * f1
                    rewards[i][j][k] = self.args.rl_bleu * bleu_reward + self.args.rl_expressiveness * expressiveness_reward + self.args.rl_relevance * ent_bleu

        if 'coherence' in rl_reward:
            # print(len(sent_pairs))
            # print('s0', sent_pairs[0])
            coh_reward = get_nsp(sent_pairs, self.bert_tokenizer, self.bert_nsp)
            for cr, (i,j,k) in zip(coh_reward, sent_idxs):
                rewards[i][j][k] += self.args.rl_coherence * cr

        # compute baseline
        baseline = [[0.0 for _ in range(num_seq)] for _ in range(batch_size)]
        for i in range(batch_size):
            for j in range(sample_size):
                for k in range(num_seq):
                    baseline[i][k] += rewards[i][j][k]
                baseline[i][k] /= sample_size

        new_tensor = src_hidden.data.new
        rewards = new_tensor(rewards)   # [batch, sample_size, num_seq]
        if self.args.rl_baseline == 'greedy':
            baseline = new_tensor(greedy_baseline)
        elif self.args.rl_baseline == 'average':
            baseline = new_tensor(baseline) # [batch, num_seq]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         masks = new_tensor(masks).byte()
        else:
            baseline = new_tensor(batch_size, num_seq).zero_()

        # max_seq_len = scores.size(-1)
        # rewards = rewards.unsqueeze(3).repeat(1, 1, 1, max_seq_len)
        # baseline = baseline.unsqueeze(1).unsqueeze(3).repeat(1, sample_size, 1, max_seq_len)
        # loss = (rewards - baseline) * (-scores)
        # masks = [[[[0] * len(seq[1:]) + [1] * (max_seq_len - len(seq[1:])) for seq in seqs] for seqs in sample] for sample in sequences]
        # masks = new_tensor(masks).byte()
        # loss.data.masked_fill_(masks, 0.0)
        # num_word = float((1 - masks).sum())
        # avg_loss = loss.sum() / num_word
        # Compute loss
        num_word = 0
        sent_scores = []
        for i in range(num_seq):
            max_seq_len = scores[i].size(-1)
            # [batch, sample_size, max_seq_len]
            masks = [[[0] * len(seqs[i][1:]) + [1] * (max_seq_len - len(seqs[i][1:])) for seqs in sample] for sample in sequences]
            masks = new_tensor(masks).byte()
            scores[i].data.masked_fill_(masks, 0.0)
            sent_scores.append(scores[i].sum(dim=-1))
            num_word += float((1-masks).sum())
            # print('sent_scores', sent_scores[-1].size())
        sent_scores = torch.stack(sent_scores, dim=2)   # [batch, sample_size, num_seq]
        baseline = baseline.unsqueeze(1).repeat(1, sample_size, 1)
        loss = (rewards - baseline) * (-sent_scores)
        avg_loss = loss.sum() / num_word

        # avg_loss = loss.sum() / batch_size / num_seq
        return avg_loss

    def train_reward(self, sample):
        if self.args.objective == 'REWARD-LM':
            return self.train_lm(sample)

    def train_lm(self, sample):
        src_seq = sample['src_seq']
        src_lengths = sample['src_lengths']
        src_hidden, final_src_ctx = self.encoder(src_seq, src_lengths)
        scores = self.decoder(src_hidden, final_src_ctx, trg_seq)

    def compute_reward(self, hyp, refs, entities):
        rl_reward = self.args.rl_reward.split('-')
        reward = 0
        if 'relevance' in rl_reward:
            weights = [float(ww) for ww in self.args.rl_relevance_weight.split('-')]
            reward, _ = compute_relevance_reward(hyp, refs, entities, self.decoder.vocab, beta=self.args.rl_relevance_beta, weights=weights)
        return reward

    def compute_expressiveness_reward(self, hyps):
        """ hyp: [batch, num_seq, num_tokens] """
        scores = []
        for i, hyp in enumerate(hyps):
            for k, h in enumerate(hyp):
                pre_hs = h[:k-1]
                dBLEU = sentence_bleu(pre_hs, h, smoothing_function=SmoothingFunction().method3)
                dBLEU_reward = 1 - dBLEU
                scores.append(dBLEU_reward)
        return scores

    def forward(self, src_seq, src_lengths, trg_seq):
        src_hidden, final_src_ctx = self.encoder(src_seq, src_lengths)
        scores = self.decoder(src_hidden, final_src_ctx, trg_seq)
        return scores
