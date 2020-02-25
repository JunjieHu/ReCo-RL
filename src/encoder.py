import torch 
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

class Encoder(nn.Module):
    """ Base class for encoder """
    def __init__(self, vocab):
        super().__init__()
        self.vocab = vocab 
    
    def forward(self, src_seq, src_len):
        raise NotImplementedError
    
    
class MeanEncoder(Encoder):
    def __init__(self):
        super().__init__(vocab=None)
    
    def forward(self, src_seq, src_lengths):
        """ Return the mean of the source sequence as state and cell"""
        return src_seq, (src_seq.mean(dim=1), src_seq.mean(dim=1))


class LSTMEncoder(Encoder):
    def __init__(self, vocab, embed_size, hidden_size, num_layers, dropout, bidirectional=True):
        super().__init__(vocab)
        self.vocab = vocab  
        self.bidirectional = bidirectional
        self.embed = nn.Embedding(len(self.vocab), embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)
        self.num_layers = num_layers
        self.dropout = dropout
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        # print('embed_size', embed_size)
        # print('embed', self.embed.weight.size())

    def forward(self, src_seq, src_lengths):
        src_embed = self.embed(src_seq)                 # [batch, src_len, embed_size]
        packed_src_embed = pack_padded_sequence(src_embed, src_lengths, batch_first=True)

        src_hidden, last_hidden = self.lstm(packed_src_embed)  # [batch, src_len, hidden_size*2]
        src_hidden, _ = pad_packed_sequence(src_hidden, batch_first=True)

        batch_size = src_hidden.size(0)
        h_T = last_hidden[0].transpose(0, 1).contiguous().view(batch_size, -1)  # [batch, num_layers * 2 * hidden_size]
        c_T = last_hidden[1].transpose(0, 1).contiguous().view(batch_size, -1)
        last_hidden = (h_T, c_T)
        return src_hidden, last_hidden 
        