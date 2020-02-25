import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM, BertForNextSentencePrediction
from pytorch_pretrained_bert.modeling import BertPreTrainedModel
import numpy as np
import logging
logging.basicConfig(level=logging.INFO)

class BertOnlyNSRHead(nn.Module):
    def __init__(self, config):
        super(BertOnlyNSRHead, self).__init__()
        self.seq_regression = nn.Linear(config.hidden_size, 1)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_regression(pooled_output)
        return seq_relationship_score
    
class BertForNextSentenceRegression(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForNextSentenceRegression, self).__init__(config)

        self.bert = BertModel(config)
        self.cls = BertOnlyNSRHead(config)
        self.apply(self.init_bert_weights)
        
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, next_sentence_label=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask,
                                     output_all_encoded_layers=False)
        seq_relationship_score = self.cls( pooled_output)

        if next_sentence_label is not None:
            loss_fct = torch.nn.MSELoss() 
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1), next_sentence_label.view(-1))
            return next_sentence_loss
        else:
            return seq_relationship_score

def get_nsp_reg(pairs, tokenizer, bert_nsp, batch_size=40):
    # construct pairs to sequence 
    num_pair = len(pairs)
    seqs, typs = [], []
    for p in pairs:
        tp = [tokenizer.tokenize(pi) for pi in p]
        lp = [len(pi) for pi in tp]
        s = ["[CLS]"] + tp[0] + ['[SEP]'] + tp[1] + ['[SEP]'] 
        seqs.append(s)
        typs.append([0]*(lp[0]+2) + [1]*(lp[1]+1))

    z = zip(seqs, typs, range(num_pair))
    sz = sorted(z, key=lambda x: len(x[1]))
    scores = [0 for _ in range(num_pair)]

    num_batch = int(np.ceil(num_pair/ float(batch_size))) 
    for i in range(num_batch):
        cur_batch_size = batch_size if i < num_batch - 1 else num_pair - batch_size * i 
        seq = [sz[i*batch_size+b][0] for b in range(cur_batch_size)]
        typ = [sz[i*batch_size+b][1] for b in range(cur_batch_size)]
        idx = [sz[i*batch_size+b][2] for b in range(cur_batch_size)]
        max_len = max(len(l) for l in typ)

        marks = [[1]*len(s) + [0]*(max_len-len(s)) for s in seq]
        seq_aug = [tokenizer.convert_tokens_to_ids(s) + [0]*(max_len-len(s)) for s in seq]
        typ_aug = [t + [0]*(max_len-len(t)) for t in typ]

        marks = torch.LongTensor(marks).to('cuda')
        seq_aug = torch.LongTensor(seq_aug).to('cuda')
        typ_aug = torch.LongTensor(typ_aug).to('cuda')
        bert_nsp.to('cuda')
        with torch.no_grad():
            seq_logits = bert_nsp(seq_aug, typ_aug, marks)
            seq_probs = seq_logits.tolist()
            #seq_probs = torch.softmax(seq_logits, dim=1)[:,0].tolist() # [batch]
            for j, sp in zip(idx, seq_probs):
                scores[j] = sp 
    return scores 

def get_nsp(pairs, tokenizer, bert_nsp, batch_size=40):
    # construct pairs to sequence 
    num_pair = len(pairs)
    seqs, typs = [], []
    for p in pairs:
        tp = [tokenizer.tokenize(pi) for pi in p]
        lp = [len(pi) for pi in tp]
        s = ["[CLS]"] + tp[0] + ['[SEP]'] + tp[1] + ['[SEP]'] 
        seqs.append(s)
        typs.append([0]*(lp[0]+2) + [1]*(lp[1]+1))

    z = zip(seqs, typs, range(num_pair))
    sz = sorted(z, key=lambda x: len(x[1]))
    scores = [0 for _ in range(num_pair)]

    num_batch = int(np.ceil(num_pair/ float(batch_size))) 
    for i in range(num_batch):
        cur_batch_size = batch_size if i < num_batch - 1 else num_pair - batch_size * i 
        seq = [sz[i*batch_size+b][0] for b in range(cur_batch_size)]
        typ = [sz[i*batch_size+b][1] for b in range(cur_batch_size)]
        idx = [sz[i*batch_size+b][2] for b in range(cur_batch_size)]
        max_len = max(len(l) for l in typ)
        marks = [[1]*len(s) + [0]*(max_len-len(s)) for s in seq]
        seq_aug = [tokenizer.convert_tokens_to_ids(s) + [0]*(max_len-len(s)) for s in seq]
        typ_aug = [t + [0]*(max_len-len(t)) for t in typ]

        marks = torch.LongTensor(marks).to('cuda')
        seq_aug = torch.LongTensor(seq_aug).to('cuda')
        typ_aug = torch.LongTensor(typ_aug).to('cuda')
        bert_nsp.to('cuda')
        with torch.no_grad():
            seq_logits = bert_nsp(seq_aug, typ_aug, marks)
            seq_probs = torch.softmax(seq_logits, dim=1)[:,0].tolist() # [batch]
            for j, sp in zip(idx, seq_probs):
                scores[j] = sp 
    return scores 

def score_nsp(file, bert_weight, bert_vocab):
    pairs = []
    for l in open(file):
        sents = l.strip().split('\t')[1].strip().split('.')
        for i in range(len(sents)):
            pre = "" if i == 0 else sents[i-1].strip() + ' . '
            cur = sents[i].strip() + ' .'
            pairs.append("[CLS] {}[SEP] {} [SEP]".format(pre, cur))

    bert_tokenizer = BertTokenizer.from_pretrained(bert_vocab)
    bert_nsp = BertForNextSentencePrediction.from_pretrained(bert_weight)
    bert_nsp.eval()
    scores = get_nsp(pairs, bert_tokenizer, bert_nsp)
    # print('scores', scores)
    for s, p in zip(pairs, scores):
        print('s={}, p={}'.format(s, p))
    return np.mean(scores)

def convert_wx_to_pair_file(wx_file, pair_file):
    with open(pair_file, 'w') as f:
        for l in open(wx_file):
            ls = l.strip().split('\t')[1].strip() 
            sents = [s.strip()+' .' for s in ls.split('.')[:-1]]
            if len(sents) < 2:
                print('l=', l)
            for i in range(1, len(sents)):
                pre = sents[i-1]
                cur = sents[i]
                f.write(f'[CLS] {pre} [SEP] {cur} [SEP]\n')
