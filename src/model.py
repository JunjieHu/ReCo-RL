import torch 
import torch.nn as nn

from encoder import Encoder, LSTMEncoder
from decoder import Decoder, LSTMDecoder, repeat
from vocab import Vocab


class BaseModel(nn.Module):
    """ Base class for encoder-decoder models """

    def __init__(self, args, encoder, decoder):
        super().__init__()

        self.args = args 
        self.encoder = encoder
        self.decoder = decoder 
        assert isinstance(self.encoder, Encoder)
        assert isinstance(self.decoder, Decoder)
    
    @staticmethod
    def add_args(parser):
        pass
    
    def forward(self, src_seq, src_lengths, trg_seq):
        src_hidden, final_src_ctx = self.encoder(src_seq, src_lengths)
        scores = self.decoder(src_hidden, final_src_ctx, trg_seq)
        return scores
    
    def get_normalized_probs(self, net_output, log_probs):
        return self.decoder.get_normalized_probs(net_output, log_probs)

    def sample(self, src_seq, sample_size=None, to_word=True, sample_method='random'):
        """ Sampling for each source sequence
        Args:
            src_seq: [batch, src_seq_len] or [batch, src_seq_len, dim] 
        Return:
            samples, scores
        """
        pass
    
    def reinforce(self, src_seq, src_lengths, trg_seq, sample_size=None, to_word=True, rl_reward='BLEU'):
        pass 

    def uniform_init(self):
        print('uniformly initialize parameters [-%f, +%f]' % (self.args.uniform_init, self.args.uniform_init))
        for name, param in self.named_parameters():
            print(name, param.size())
            param.data.uniform_(-self.args.uniform_init, self.args.uniform_init)

    def save(self, path):
        params = {
            'args': self.args,
            'encoder_state_dict': self.encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'encoder_vocab': self.encoder.vocab,
            'decoder_vocab': self.decoder.vocab
        }
        torch.save(params, path)

    @staticmethod
    def load(path, MODEL=None):
        if MODEL is None:
            print('Please specify which model to load')
            exit(0)
        params = torch.load(path, map_location=lambda storage, loc: storage)
        args = params['args']
        vocab = Vocab()
        vocab.src = params['encoder_vocab']
        vocab.trg = params['decoder_vocab']
        model = MODEL.build_model(args, vocab)
        model.encoder.load_state_dict(params['encoder_state_dict'])
        model.decoder.load_state_dict(params['decoder_state_dict'])
        return model 

class LSTMSeq2SeqModel(BaseModel):
    def __init__(self, args, encoder, decoder):
        super().__init__(args,encoder, decoder)

    @staticmethod
    def add_args(parser):
        parser.add_argument('--embed_size', default=512, type=int)
        parser.add_argument('--hidden_size', default=512, type=int)
        parser.add_argument('--num_encoder_layers', default=1, type=int)
        parser.add_argument('--num_decoder_layers', default=1, type=int)
        parser.add_argument('--dropout', default=0.3, type=int)
        parser.add_argument('--bidirectional', action='store_true', default=False)
        return parser
    
    @staticmethod
    def build_model(args, vocab):
        print('build LSTMSeq2SeqModel')
        encoder_hidden_size = 2 * args.hidden_size if args.bidirectional else args.hidden_size
        encoder_ctx_size = encoder_hidden_size * args.num_encoder_layers
        encoder = LSTMEncoder(vocab.src, args.embed_size, args.hidden_size, args.num_encoder_layers, args.dropout)
        decoder = LSTMDecoder(vocab.trg, args.embed_size, args.hidden_size, args.num_decoder_layers, args.dropout, encoder_hidden_size, encoder_ctx_size)
        return LSTMSeq2SeqModel(args, encoder, decoder)

    def generate(self, src_seq, src_lengths, beam_size=5, decode_max_length=100, to_word=True):
        src_hidden, final_src_ctx = self.encoder(src_seq, src_lengths)   # [batch, src_len, enc_hidden_size]

        completed_sequences, completed_scores = self.decoder.generate(src_hidden, final_src_ctx,  beam_size, decode_max_length, to_word)       
        return completed_sequences, completed_scores 
    
    @staticmethod
    def load(path):
        return BaseModel.load(path, LSTMSeq2SeqModel)


        
