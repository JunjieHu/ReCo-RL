
import torch
import torch.nn as nn
import torch.nn.functional as F


def dot_prod_attention(h_t, src_encoding, src_encoding_att_linear, mask=None):
    """
    :param h_t: (batch_size, hidden_size)
    :param src_encoding: (batch_size, src_sent_len, hidden_size * 2)
    :param src_encoding_att_linear: (batch_size, src_sent_len, hidden_size)
    :param mask: (batch_size, src_sent_len)
    """
    # (batch_size, src_sent_len)
    att_weight = torch.bmm(src_encoding_att_linear, h_t.unsqueeze(2)).squeeze(2)
    if mask:
        att_weight.data.masked_fill_(mask, -float('inf'))
    att_weight = F.softmax(att_weight)

    att_view = (att_weight.size(0), 1, att_weight.size(1))
    # (batch_size, hidden_size)
    ctx_vec = torch.bmm(att_weight.view(*att_view), src_encoding).squeeze(1)

    return ctx_vec, att_weight


class StackedAttentionLSTM(nn.Module):
    """
    stacked LSTM.
    Needed for the decoder, because we do input feeding.
    """
    def __init__(self, num_layers, input_size, rnn_size, dropout, ctx_vec_size=None):
        super(StackedAttentionLSTM, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        # Map the concatenated [h_t; ctx_t] vector to the rnn_size vector 
        self.ctx_vec_size = rnn_size * 2 if ctx_vec_size is None else ctx_vec_size
        self.att_vec_linear = nn.Linear(rnn_size + self.ctx_vec_size, rnn_size, bias=False)

        for i in range(num_layers):
            self.layers.append(nn.LSTMCell(input_size, rnn_size))
            input_size = rnn_size
        #print('input size', input_size, rnn_size)

    def forward(self, input, hidden, src_encoding, src_encoding_att_linear):
        """
        :param input: (batch_size, input_size)
        :param hidden : (num_layer, batch_size, hidden_size)
        :param src_encoding: (batch_size, src_len, ctx_vec_size)
        :param src_encoding_att_linear: (batch_size, src_len, hidden_size)
        return: input: (batch_size, hidden_size)
                h_1, c_1: (num_layers, batch_size, hidden_size)
        """
        h_0, c_0 = hidden
        #print('layer 0', len(hidden), h_0.size(), c_0.size())
        #print('input', input.size())
        #print('self.layers[0]', self.layers[0])
        h_1_0, c_1_0 = self.layers[0](input, (h_0[0], c_0[0]))
        h_1, c_1 = [h_1_0], [c_1_0]
        # Only use the first decoding outputs to do attention and copy the context vectors
        # to the subsequent decoding layers
        ctx_t, alpha_t = dot_prod_attention(h_1_0, src_encoding, src_encoding_att_linear)
        input = self.att_vec_linear(torch.cat([h_1_0, ctx_t], 1))  # (batch, hidden_size)

        for i in range(1, len(self.layers)):
            layer = self.layers[i]
            h_1_i, c_1_i = layer(input, (h_0[i], c_0[i]))
            input = self.att_vec_linear(torch.cat([h_1_i, ctx_t], 1))
            if i + 1 != self.num_layers:
                input = self.dropout(input)
            h_1 += [h_1_i]
            c_1 += [c_1_i]
        input = F.tanh(input)
        h_1 = torch.stack(h_1)
        c_1 = torch.stack(c_1)
        h_1 = self.dropout(h_1)
        return input, (h_1, c_1)


class StackedAttentionGRU(nn.Module):

    def __init__(self, num_layers, input_size, rnn_size, dropout):
        super(StackedAttentionGRU, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(nn.GRUCell(input_size, rnn_size))
            input_size = rnn_size

    def forward(self, input, hidden, src_encoding, src_encoding_att_linear):
        """
        :param input: (batch_size, input_size)
        :param hidden : (num_layer, batch_size, hidden_size)
        :param src_encoding: (batch_size, src_len, hidden_size * 2)
        :param src_encoding_att_linear: (batch_size, src_len, hidden_size)
        return: input: (batch_size, hidden_size)
                h_1, c_1: (num_layers, batch_size, hidden_size)
        """
        h_0, c_0 = hidden
        h_1_0, c_1_0 = self.layers[0](input, (h_0[0], c_0[0]))
        h_1, c_1 = [h_1_0], [c_1_0]
        # Only use the first decoding outputs to do attention and copy the context vectors
        # to the subsequent decoding layers
        ctx_t, alpha_t = dot_prod_attention(h_1_0, src_encoding, src_encoding_att_linear)
        input = self.att_vec_linear(torch.cat([h_1_0, ctx_t], 1))  # (batch, hidden_size)

        for i in range(1, len(self.layers)):
            layer = self.layers[i]
            h_1_i, c_1_i = layer(input, (h_0[i], c_0[i]))
            input = self.att_vec_linear(torch.cat([h_1_i, ctx_t], 1))
            if i + 1 != self.num_layers:
                input = self.dropout(input)
            h_1 += [h_1_i]
            c_1 += [c_1_i]
        input = F.tanh(input)
        h_1 = torch.stack(h_1)
        c_1 = torch.stack(c_1)
        h_1 = self.dropout(h_1)
        return input, (h_1, c_1)


class StackedLSTM(nn.Module):
    def __init__(self, num_layers, input_size, rnn_size, dropout):
        super(StackedLSTM, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(nn.LSTMCell(input_size, rnn_size))
            input_size = rnn_size

    def forward(self, input, hidden):
        h_0, c_0 = hidden
        h_1, c_1 = [], []
        for i, layer in enumerate(self.layers):
            h_1_i, c_1_i = layer(input, (h_0[i], c_0[i]))
            input = h_1_i
            if i + 1 != self.num_layers:
                input = self.dropout(input)
            h_1 += [h_1_i]
            c_1 += [c_1_i]

        h_1 = torch.stack(h_1)
        c_1 = torch.stack(c_1)

        return input, (h_1, c_1)


class StackedGRU(nn.Module):

    def __init__(self, num_layers, input_size, rnn_size, dropout):
        super(StackedGRU, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(nn.GRUCell(input_size, rnn_size))
            input_size = rnn_size

    def forward(self, input, hidden):
        h_1 = []
        for i, layer in enumerate(self.layers):
            h_1_i = layer(input, hidden[0][i])
            input = h_1_i
            if i + 1 != self.num_layers:
                input = self.dropout(input)
            h_1 += [h_1_i]

        h_1 = torch.stack(h_1)
        return input, (h_1,)


class AttentionLSTM(nn.Module):
    """
    stacked LSTM.
    Needed for the decoder, because we do input feeding.
    """
    def __init__(self, input_size, rnn_size, dropout, ctx_vec_size=None):
        super(StackedAttentionLSTM, self).__init__()
        self.dropout = nn.Dropout(dropout)

        # Map the concatenated [h_t; ctx_t] vector to the rnn_size vector 
        self.ctx_vec_size = rnn_size * 2 if ctx_vec_size is None else ctx_vec_size
        self.att_vec_linear = nn.Linear(rnn_size + self.ctx_vec_size, rnn_size, bias=False)

        self.lstm = nn.LSTMCell(input_size, rnn_size)
        # for i in range(num_layers):
        #     self.layers.append(nn.LSTMCell(input_size, rnn_size))
        #     input_size = rnn_size
        #print('input size', input_size, rnn_size)

    def forward(self, input, hidden, src_encoding, src_encoding_att_linear):
        """
        :param input: (batch_size, input_size)
        :param hidden : (batch_size, hidden_size)
        :param src_encoding: (batch_size, src_len, ctx_vec_size)
        :param src_encoding_att_linear: (batch_size, src_len, hidden_size)
        return: input: (batch_size, hidden_size)
                h_1, c_1: (batch_size, hidden_size)
        """
        # h_0, c_0 = hidden
        h_1, c_1 = self.lstm(input, hidden)
        # h_1, c_1 = h_1_0, [c_1_0]
        ctx_t, alpha_t = dot_prod_attention(h_1, src_encoding, src_encoding_att_linear)
        input = self.att_vec_linear(torch.cat([h_1, ctx_t], 1))
        input = F.tanh(input)
        return input, (h_1, c_1) 
