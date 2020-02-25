import torch 
import torch.nn as nn
import torch.nn.functional as F
from StackedRNN import StackedAttentionLSTM, StackedAttentionGRU


def new_state(hidden_state):
    if type(hidden_state) is tuple or type(hidden_state) is list:
        new_state = []
        for h in hidden_state:
            nh = h.data.new(h.size()).fill_(0)
            new_state.append(nh)
        new_state = tuple(new_state)
    else:
        new_state = hidden_state.data.new(hidden_state.size()).fill_(0)
    return new_state

def repeat(input, dim, k):
    """Repeat the input tensor k times along the dim dimention
        input: [dim, d]
        output: [dim*k, d]
    """
    if type(input) is tuple:
        size = [-1] * len(input[0].size())
        size.insert(dim, k)
        new_size = list(input[0].size())
        new_size[dim-1] *= k
        input = list(input)
        for idx in range(len(input)):
            input[idx] = input[idx].unsqueeze(dim).expand(tuple(size)).contiguous().view(new_size)
        input = tuple(input)
    else:
        size = [-1] * len(input.size())
        size.insert(dim, k)
        new_size = list(input.size())
        new_size[dim-1] *= k
        input = input.unsqueeze(dim).expand(tuple(size)).contiguous().view(new_size)
    return input

def resize(input, size):
    if type(input) is tuple:
        input = list(input)
        for idx in range(len(input)):
            input[idx] = input[idx].view(size)
        input = tuple(input)
    else:
        input.view(size)
    return input

def topK(score):
    """
    Args: 
        score: [beam, num_vocab]
    Return:
        top_score: [beam]
        top_rowid: [beam], beam id 
        top_colid: [beam], word id
    """
    beam_size, num_vocab = score.size() 
    flat_score = score.view(beam_size * num_vocab)
    top_score, top_index = flat_score.topk(k=beam_size, dim=0)
    top_rowid, top_colid = top_index / num_vocab, top_index % num_vocab
    return top_score, top_rowid, top_colid 

def topK_2d(score):
    """
    Args: 
        score: [batch, beam_size, num_vocab]
    Return:
        top_score: [batch, beam_size], select score
        top_rowid: [batch, beam_size], beam id
        top_colid: [batch, beam_size], word id
    """
    batch_size, beam_size, num_vocab = score.size()
    flat_score = score.view(batch_size, beam_size * num_vocab)
    top_score, top_index = flat_score.topk(k=beam_size, dim=1)
    top_rowid, top_colid = top_index / num_vocab, top_index % num_vocab 
    return top_score, top_rowid, top_colid  


def copy_dict(d):
    di = defaultdict(list)
    for k, v in d.items():
        di[k] = [vi for vi in v]
    return di 
    
def topK_2d_ngrams(orig_score, ngrams, sequence, n=2, word_seq=None, word_ngrams=None, vocab=None):
    """
    Args:
        orig_score: [batch, beam_size, num_vocab]
        ngrams: [batch, beam_size], each is a set
        sequence: [batch, beam_size] 
    Return:
        top_score: [batch, beam_size]
        top_rowid: [batch, beam_size]
        top_colid: [batch, beam_size]
    """
    batch_size, beam_size, num_vocab = orig_score.size()
    flat_score = orig_score.view(batch_size, beam_size * num_vocab)
    top_score, top_index = torch.sort(flat_score, dim=1, descending=True)  # [batch, beam*|V|]
    
    if ngrams is None:
        new_score = top_score[:, 0:beam_size]
        new_index = top_index[:, 0:beam_size]
        new_rowid, new_colid = new_index / num_vocab, new_index % num_vocab
        for i in range(batch_size):
            for j in range(beam_size):
                sequence[i][j] = sequence[i][j] + [int(new_colid[i][j])]
    else:
        new_tensor = orig_score.data.new
        new_score = new_tensor(batch_size, beam_size).zero_()
        new_rowid = new_tensor(batch_size, beam_size).zero_().long()
        new_colid = new_tensor(batch_size, beam_size).zero_().long()

        for i in range(batch_size):
            b = 0
            seq_b = sequence[i][b]
            ngram_b = ngrams[i][b]
            wseq_b = word_seq[i][b]
            wngram_b = word_ngrams[i][b]
            for j in range(top_score.size(1)):
                score = top_score[i, j].item()
                index = top_index[i, j].item()
                rowidx = int(index / num_vocab)
                colidx = int(index % num_vocab)
                pre_word = seq_b[-(n-1):]
                cur_word = colidx
                ngram = tuple(pre_word + [cur_word])
                pre_w = wseq_b[-(n-1):]
                cur_w = vocab.id2word[colidx]
                wngram = tuple(pre_w + [cur_w])
                tmp = wngram in wngram_b
                if ngram in ngram_b:
                    continue
                ngram_b.add(ngram)
                wngram_b.add(wngram)
                new_score[i][b] = score 
                new_rowid[i][b] = rowidx 
                new_colid[i][b] = colidx 
                sequence[i][b] = sequence[i][b] + [colidx]
                word_seq[i][b] = word_seq[i][b] + [cur_w]
                b += 1
                if b >= beam_size:
                    break    
                seq_b = sequence[i][b]
                ngram_b = ngrams[i][b]
                wseq_b = word_seq[i][b]
                wngram_b = word_ngrams[i][b]
    return new_score, new_rowid, new_colid, ngrams, sequence

def update_ngrams(top_rowid, top_colid, top_seqs, ngrams, n):
    batch_size, beam_size = len(top_seqs), len(top_seqs[0])
    new_seqs = [[[] for _ in range(beam_size)] for _ in range(batch_size)]
    new_ngrams = [[set() for _ in range(beam_size)] for _ in range(batch_size)]
    for i in range(batch_size):
        for j in range(beam_size):
            beam = int(top_rowid[i][j])
            word = int(top_colid[i][j])
            new_seqs[i][j] = top_seqs[i][beam] + [word] 
            new_ngrams[i][j] = ngrams[i][beam].copy()
            new_ngrams[i][j].add(top_seqs[i][beam][-n:])
    return new_seqs, new_ngrams 

def select_hid_1d(hidden, row_id):
    """
    Args: 
        hidden: [beam, hidden_size]
        row_id: [beam]
    Return:
        new_hidden: [beam, hidden_size]
    """
    new_hidden = []
    for h in hidden:
        new_h = h[row_id.data]
        new_hidden.append(new_h)
    new_hidden = tuple(new_hidden)
    return new_hidden

def select_hid(hidden, batch_id, row_id):
    """ Re-arange the hidden state according to the selected beams in the previous step
    Args:
        hidden: [batch*beam_size, hidden_size] 
        batch_id: [batch, beam_size] 
        row_id: [batch, beam_size] 
    Return:
        new_hidden: [batch*beam_size, hidden_size]
    """
    batch_size, beam_size = row_id.size()
    if type(hidden) is tuple or type(hidden) is list:
        new_hidden = []
        for h in hidden:
            new_h = h.view(batch_size, beam_size, -1)[batch_id.data, row_id.data]
            new_h = new_h.view(batch_size * beam_size, -1)
            new_hidden.append(new_h)
        new_hidden = tuple(new_hidden)
    else:
        new_hidden = hidden.view(batch_size, beam_size, hidden.size(2))[:, batch_id.data, row_id.data]
        new_hidden = new_hidden.view(batch_size * beam_size, hidden.size(2))
    return new_hidden 

def update_complete_hid(last_complete_hid, eos_mask, cur_dec_hid):
    """
    Args: 
        last_complete_hid: [batch*beam, hidden_size]
        eos_mask: [batch, beam]
        cur_dec_hid: [batch*beam, hidden_size]
    Return: 
        complete_hid: [batch*beam, hidden_size]
    """
    hidden_size = last_complete_hid[0].size(-1)
    eos = eos_mask.view(-1, 1).repeat(1, hidden_size)
    if type(last_complete_hid) is tuple or type(last_complete_hid) is list:
        complete_hid = []
        for lch, cdh in zip(last_complete_hid, cur_dec_hid):
            h = eos.float() * lch + (1 - eos).float() * cdh 
            complete_hid.append(h)
        complete_hid = tuple(complete_hid)
    else:
        complete_hid = eos.float() * last_complete_hid + (1 - eos).float() * cur_dec_hid 
    return complete_hid


def update_top_seqs(top_seqs, beam_bptr, top_words):
    """
    Args:
        top_seqs: [batch, beam, num_words]
        beam_bptr: [batch, beam]
        top_words: [batch, beam]
    Return:
        top_seqs: [batch, beam, num_words+1]
    """
    new_seqs = []
    for bi_seqs, bi_bptr, bi_words in zip(top_seqs, beam_bptr, top_words):
        new_bi_seqs = [bi_seqs[bptr] + [word] for bptr, word in zip(bi_bptr, bi_words)]
        new_seqs.append(new_bi_seqs)
    return new_seqs


def select_sequences_by_pointer(completed_sequences, completed_scores, sequences, scores, beam_bptr):
    """
    Args:
        completed_sequences: [batch, beam, num_seq, max_seq_len]
        completed_scores: [batch, beam, num_seq]
        sequences: [batch, beam, max_seq_len］
        scores: [batch, beam］
        beam_bptr: [batch, beam]
    """
    if len(completed_sequences) == 0:
        completed_sequences = [[[seq] for seq in beams] for beams in sequences]
        completed_scores = [[[score] for score in beams] for beams in scores] 
        return completed_sequences, completed_scores 
    else:
        new_sequences, new_scores = [], []
        for cseqs, csc, seqs, sc, bptr in zip(completed_sequences, completed_scores, sequences, scores, beam_bptr):
            ncseqs = [cseqs[b] + [s] for b, s in zip(bptr, seqs)]
            ncsc = [csc[b] + [s] for b, s in zip(bptr, sc)]
            new_sequences.append(ncseqs)
            new_scores.append(ncsc)
    return new_sequences, new_scores
            

class Decoder(nn.Module):
    """ Base class for decoder """
    def __init__(self, vocab):
        super().__init__()
        self.vocab = vocab
    
    def forward(self, src_hidden, final_src_ctx, trg_seq):
        """ 
        Args:
            src_hidden: [batch, src_len, encoder_hidden_size]
            final_src_ctx: tuple of (last_state, last_cell), [batch, encoder_hidden_size]
            trg_seq: [batch, trg_len]
        Return:
            scores: probability of words [batch, trg_len, vocab_size]
        """
        raise NotImplementedError
    
    def get_normalized_probs(self, net_output, log_probs):
        """Get normalized probabilities (or log probs) from a net's output."""
        logits = net_output[0]
        if log_probs:
            return F.log_softmax(logits, dim=-1)
        else:
            return F.softmax(logits, dim=-1)  

    def init_state(self, final_src_ctx):
        print('using base init_state')
        raise NotImplementedError      
    
    def decode_one_step(self, input, hidden, src_hidden, ctx_vec, beam_size):
        """ Decode only one step
        Args: 
            input: [batch], the current input
            hidden: [batch, hidden_size], the previous hidden state
            ctx_vec: [batch, d], (optional) context vector to append to the input at each decoding step
            src_hidden: [batch, src_seq, D], (optional) for attention on source sequence
        Return:
            log_prob: [batch, num_vocab]
            hidden: [batch, hidden_size], next hidden state
        """
        raise NotImplementedError
    
    def beam_search(self, src_hidden, final_src_ctx, ctx_vec, repeat_hidden=True, repeat_ctx=False, beam_size=5, decode_max_length=100, to_word=True):
        """
        Args:
            src_hidden: [batch, src_seq_len, hidden_size]
            final_src_ctx: tensor or tuple of tensor [batch, hidden_size], used to initialize decoder hidden state
            ctx_vec: [batch, hidden_size], context vector that adds to the input at each decoding step
        Return:
            sequence: [batch, beam_size, decode_max_length] 
            sort_score: [batch, beam_size]
        """
        batch_size = src_hidden.size(0)
        num_vocab = len(self.vocab)   
        new_tensor = src_hidden.data.new
        pad_id = self.vocab['<pad>']
        eos_id = self.vocab['</s>']
        bos_id = self.vocab['<s>']
        # print('batch_size', batch_size)

        # repeat `src_hidden`, `pad_mask` for k times
        dec_input = new_tensor(batch_size * beam_size).long().fill_(bos_id)  # [batch*beam_size]
        dec_hidden = self.init_state(final_src_ctx)  # [batch, hidden_size]

        if repeat_hidden:
            dec_hidden = repeat(dec_hidden, dim=1, k=beam_size)  # [batch*beam_size, hidden]

        if repeat_ctx:
            ctx_vec = repeat(ctx_vec, dim=1, k=beam_size)  # [batch*beam_size, hidden_size]

        batch_id = torch.LongTensor(range(batch_size))[:, None].repeat(1, beam_size).type_as(dec_input.data)
        
        top_score = new_tensor(batch_size, beam_size).fill_(-float('inf'))
        top_score.data[:, 0].fill_(0)

        # [1, |V|]
        pad_vec = new_tensor(1, num_vocab).float().fill_(-float('inf'))
        pad_vec[0, pad_id] = 0
        eos_mask = new_tensor(batch_size, beam_size).byte().fill_(0)  # [batch, beam_size]

        top_rowids, top_colids = [], []
        for i in range(decode_max_length):
            # [batch*beam_size, |V|]
            log_prob, dec_hidden, ctx_vec = self.decode_one_step(dec_input, dec_hidden, src_hidden, ctx_vec, beam_size)
            log_prob = log_prob.view(batch_size, beam_size, num_vocab)  # [batch, beam_size, |V|]
            
            if eos_mask is not None and eos_mask.sum() > 0:
                log_prob.data.masked_scatter_(eos_mask.unsqueeze(2), pad_vec.expand((eos_mask.sum(), num_vocab))) # [batch]
            
            score = top_score.unsqueeze(2) + log_prob   # [batch, beam_size, |V|]

            # [batch, beam_size]
            top_score, top_rowid, top_colid = topK_2d(score)  
            top_rowids.append(top_rowid)    # select the beam id
            top_colids.append(top_colid)    # select the word id

            dec_hidden = select_hid(dec_hidden, batch_id, top_rowid)  # [batch*beam_size, hidden_size]
            dec_input = top_colid.view(-1)  # [batch_size * beam_size]

            eos_mask = eos_mask.gather(dim=1, index=top_rowid.data) | top_colid.data.eq(eos_id)
            if eos_mask.sum() == batch_size * beam_size:
                break
        
        tokens = []
        for i in reversed(range(len(top_colids))):
            if i == len(top_colids) - 1:
                sort_score, sort_idx = torch.sort(top_score, dim=1, descending=True)  # [batch_size, beam_size]
            else:
                sort_idx = top_rowids[i+1].gather(dim=1, index=sort_idx)
            token = top_colids[i].gather(dim=1, index=sort_idx)
            
            tokens.insert(0, token)
        sequence = torch.stack(tokens, dim=2)  # [batch, beam_size, decode_max_lengths]
        if to_word:
            sequence = sequence.cpu().data.numpy().tolist()
            sequence = [[[self.vocab.id2word[w] for w in seq if w != self.vocab['<pad>']] for seq in beam] for beam in sequence]
            sort_score = sort_score.cpu().data.numpy().tolist()
        return sequence, sort_score, dec_hidden

    def sample(self, src_hidden, final_src_ctx, ctx_vec, repeat_hidden=True, repeat_ctx=False, sample_size=5, decode_max_length=100, to_word=True, sample_method='random', detached=False):
        """
        Args:
            src_hidden: [batch, src_seq_len, hidden_size]
            final_src_ctx: tensor or tuple of tensor [batch, hidden_size], used to initialize decoder hidden state
            ctx_vec: [batch, hidden_size], context vector that adds to the input at each decoding step
        Return:
            completed_sequences: [batch, sample_size, decode_max_length] 
            completed_scores: [batch, sample_size, decode_max_length]
            last_dec_hidden: [batch, sample_size, hidden_size]
        """
        batch_size =  int(src_hidden.size(0))
        aug_batch_size = batch_size * sample_size  # augmented batch size
        eos_id = self.vocab['</s>']
        bos_id = self.vocab['<s>']
        pad_id = self.vocab['<pad>']
        
        # Initial the first decoding input and the init state and cell
        new_tensor = src_hidden.data.new
        dec_input = new_tensor(aug_batch_size).long().fill_(bos_id)  # [batch*sample_size]
        dec_hidden = self.init_state(final_src_ctx)  # [batch, hidden_size]

        if repeat_hidden:
            dec_hidden = repeat(dec_hidden, dim=1, k=sample_size)  # [batch*sample_size, hidden_size]
        
        if repeat_ctx:
            ctx_vec = repeat(ctx_vec, dim=1, k=sample_size)  # [batch*sample_size, hidden_size]   

        # Repeat the source hidden states
        src_hidden_repeat = repeat(src_hidden, dim=1, k=sample_size)  # [batch*sample_size, hidden]

        # Ending conditions
        sample_ends = new_tensor([0] * aug_batch_size).byte()
        all_ones = new_tensor([1] * aug_batch_size).byte()
        # hyp_scores = new_tensor(aug_batch_size).float().zero_()

        di = dec_input.unsqueeze(1).transpose(0, 1).view(batch_size, sample_size)
        samples = [di]
        samples_losses = []
        samples_scores = []

        for t in range(decode_max_length):
            # Decoding one step and get the log-probability
            log_prob, dec_hidden, ctx_vec = self.decode_one_step(dec_input, dec_hidden, src_hidden, ctx_vec, sample_size)
            
            if sample_method == 'random':
                dec_input = torch.multinomial(F.softmax(log_prob), num_samples=1)
                dec_input.masked_fill_(sample_ends.unsqueeze(1), pad_id)

                sampled_log_prob = log_prob.gather(1, dec_input).squeeze(1) * (1 - sample_ends).float()
                dec_input = dec_input.squeeze(1)

            if detached:
                dec_input = dec_input.detach()

            slp = sampled_log_prob.unsqueeze(1).transpose(0, 1).view(batch_size, sample_size)
            di = dec_input.unsqueeze(1).transpose(0, 1).view(batch_size, sample_size)

            samples_scores.append(slp)
            samples.append(di)

            # Check whether all samples are completed
            sample_ends |= torch.eq(dec_input, eos_id).byte().data
            if torch.equal(sample_ends, all_ones):
                break
        last_dec_hidden = resize(dec_hidden, (batch_size, sample_size, -1))
        
        # Post-process log-probability of each token at each decoding step
        completed_scores = torch.stack(samples_scores, dim=2)
        completed_sequences = torch.stack(samples, dim=2)   # [batch, sample_size, max_seq_len]
        masks = (completed_sequences[:,:,1:] == pad_id).byte()
        completed_sequences = completed_sequences.tolist()

        if to_word:
            for i, src_sent_samples in enumerate(completed_sequences):
                completed_sequences[i] = [[self.vocab.id2word[w] for w in s if w != pad_id] for s in src_sent_samples]
                # print('********************\nsrc-{} '.format(i))
                # for j, seq in enumerate(completed_sequences[i]):
                #     print('    hyp-{}, {}, nll={}'.format(j, ' '.join(seq), completed_scores[i][j].tolist()))

        return completed_sequences, completed_scores, last_dec_hidden


class LSTMDecoder(Decoder):
    def __init__(
        self, vocab, embed_size=512, hidden_size=512, num_layers=1, 
        dropout=0.2, encoder_hidden_size=512, encoder_ctx_size=512
        ):
        super().__init__(vocab)
        self.num_layers = num_layers
        self.dropout = dropout
        self.embed_size = embed_size
        self.hidden_size = hidden_size

        # Embedding lookup table
        self.trg_embed = nn.Embedding(len(vocab), embed_size, padding_idx=vocab['<pad>'])
        # decoder LSTMCell
        self.lstm = StackedAttentionLSTM(num_layers, embed_size + hidden_size, hidden_size, dropout=dropout)
        # Attention: project source hidden vectors to the decoder RNN's hidden space
        self.att_src_linear = nn.Linear(encoder_hidden_size, hidden_size, bias=False)
        # Initialize decoder hidden state
        self.decoder_init_linear = nn.Linear(encoder_ctx_size, hidden_size)
        # prediction layer of the target vocabulary
        self.readout = nn.Linear(hidden_size, len(vocab), bias=False)

    def init_state(self, final_src_ctx):
        last_state, last_cell = final_src_ctx    # [batch, encoder_hidden_size]
        hsz = last_cell.size()
        last_cell = last_cell.view(hsz[0], self.num_layers, -1)
        init_cell = self.decoder_init_linear(last_cell)
        init_state = F.tanh(init_cell)
        init_cell = init_cell.view(hsz[0], -1)
        init_state = init_state.view(hsz[0], -1)
        return (init_state, init_cell)

    def forward(self, src_hidden, final_src_ctx, trg_seq):
        init_cell = self.decoder_init_linear(final_src_ctx[1])
        init_state = F.tanh(init_cell)
        # init_state, init_cell = self.init_state(final_src_ctx)
        init_state = init_state.unsqueeze(0).repeat(self.num_layers, 1, 1)   # [num_layers, batch_size, decoder_hidden_size] 
        init_cell = init_cell.unsqueeze(0).repeat(self.num_layers, 1, 1)
        hidden = (init_state, init_cell)
        batch_size = src_hidden.size(0)
        new_tensor = src_hidden.data.new

        # For attention, project source hidden vectors to decoder hidden dimension space
        src_hidden_att_linear = self.att_src_linear(src_hidden)   # [batch, src_len, hidden_size]
        # initialize attentional vector
        att_tm1 = new_tensor(batch_size, self.hidden_size).zero_()

        trg_word_embed = self.trg_embed(trg_seq)  # [batch, trg_len, embed_size]
        scores = []
        # start from <s>, util y_{T-1}
        for y_tm1_embed in trg_word_embed.split(split_size=1, dim=1):
            # input feeding: concatenate y_{t-1} and previous attentional vector
            x = torch.cat([y_tm1_embed.squeeze(1), att_tm1], 1)
            # Go through LSTMCell to get h_t [batch_size, hidden_size]
            att_t, (h_t, cell_t) = self.lstm(x, hidden, src_hidden, src_hidden_att_linear)
            score_t = self.readout(att_t)   # [batch, |V|]
            scores.append(score_t)

            att_tm1 = att_t
            hidden = (h_t, cell_t)
        scores = torch.stack(scores, dim=1)  # [batch, seq_len, |V|]
        return scores

    def decode_one_step(self, input, hidden, src_hidden, ctx_vec, beam_size=5):
        y_tm1 = self.trg_embed(input)
        src_hidden = repeat(src_hidden, dim=1, k=beam_size)
        src_hidden_att_linear = self.att_src_linear(src_hidden)
        x = torch.cat([y_tm1, ctx_vec], dim=1)
        
        # For multi-layer LSTM
        # reshape [batch*beam_size, num_layers*hidden_size] to [num_layers, batch*beam_size, hidden_size]
        hsz = hidden[0].size()
        h_t = hidden[0].view(hsz[0], self.num_layers, -1).transpose(0, 1)
        c_t = hidden[1].view(hsz[0], self.num_layers, -1).transpose(0, 1) # [num_layers, batch*beam_size, hidden_size]
        att_t, (h_t, c_t) = self.lstm(x, (h_t, c_t), src_hidden, src_hidden_att_linear)
        score_t = self.readout(att_t)  # [num_seq, |V|]
        # reshape back to [batch*beam_size, num_layers*hidden_size]
        h_t = h_t.transpose(0, 1).contiguous().view(hsz[0], -1)
        c_t = c_t.transpose(0, 1).contiguous().view(hsz[0], -1)
        return F.log_softmax(score_t, dim=-1), (h_t, c_t), att_t

    def generate(self, src_hidden, final_src_ctx, beam_size=5, decode_max_length=100, to_word=True, length_norm=True):
        # initialize the first decoder hidden state
        batch_size = src_hidden.size(0)
        ctx_vec = src_hidden.data.new(src_hidden.size(0), self.hidden_size).zero_()
        final_src_ctx = repeat(final_src_ctx, dim=1, k=beam_size)    # [batch*beam_size, enc_hidden_size]
        h_T, c_T = final_src_ctx
        h_T = h_T.unsqueeze(1).repeat(1, self.num_layers, 1).contiguous().view(batch_size * beam_size, -1)
        c_T = c_T.unsqueeze(1).repeat(1, self.num_layers, 1).contiguous().view(batch_size * beam_size, -1)
        final_src_ctx = (h_T, c_T)   # [batch*beam_size, num_layer * enc_hidden_size]

        completed_sequences, completed_scores, dec_hidden = self.beam_search(src_hidden, final_src_ctx, ctx_vec, False, True, beam_size, decode_max_length, to_word)
        if length_norm:
            completed_sequences, completed_scores = self.sort_by_len(completed_sequences, completed_scores)

        return completed_sequences, completed_scores

    def sort_by_len(self, seqs, scores):
        """
        Args:
            seqs: [num_seq, beam, seq_len]
            scores: [num_seq, beam]
        """
        new_seqs, new_scores = [], []
        for seq, score in zip(seqs, scores):
            for i in range(len(score)):
                score[i] /= len(seq[i])
            pairs = list(zip(score, seq))
            sorted_pairs = sorted(pairs, key=lambda x: x[0], reverse=True)
            sorted_score, sorted_seq = zip(*sorted_pairs)
            new_seqs.append(sorted_seq)
            new_scores.append(sorted_score)
        return new_seqs, new_scores    
