import pickle 
from collections import defaultdict, Counter
import time
import nltk
import json
import h5py
import os 
import sys
sys.path.append(os.path.abspath('./'))
import os.path
import numpy as np
import random
from vocab import VocabEntry
import torch
import math

from nltk.util import ngrams
from scipy import stats
import math 

import fractions
try:
    fractions.Fraction(0, 1000, _normalize=False)
    from fractions import Fraction
except TypeError:
    from nltk.compat import Fraction

pos_set = set(['ADP', 'X', 'SYM', 'NUM', 'NOUN', 'ADV', 'DET', 'PUNCT', 'VERB', 'PROPN', 'ADJ', 'PRON', '<pad>', '<s>', '</s>', 'PART', 'CCONJ', '<unk>', 'INTJ'])
propn = {'hers':'she', 'theirs':'they', 'their':'they', 'myself':'i', 'yourself':'you', 'shes':'she', 'herself':'she',
         'him':'he', 'yours':'you', 'himself': 'he', 'ours': 'we', 'our':'we', 'me':'i', 'us': 'we', 'im':'i', 'hes':'he',
         'themselves': 'they', 'them': 'they', 'ourselves': 'we', 'itself': 'it', 'y':'y', 'you':'you', 'they':'they',
         'i':'i', 'thee':'thee', 'id':'i', 'shes':'she', 'em':'em', 'wed':'we', 'ya':'ya', 'we':'we', 'it':'it', 
         'she':'she', 'he':'he', 'her':'she', "'s":"'s", 'one':'one', 'my':'me', 'mine':'me', 'his':'his', 't.v':'t.v',
         'thy':'they', 'yo':'you', 'hime':'he', 'itsself':'it', 'hats|they':'hat', 'him.|a':'he', 'egg.s':'egg',
         'tthey':'they'}

def brevity_penalty(closest_ref_len, hyp_len):
    if hyp_len > closest_ref_len:
        return 1
    # If hypothesis is empty, brevity penalty = 0 should result in BLEU = 0.0
    elif hyp_len == 0:
        return 0
    else:
        return math.exp(1 - closest_ref_len / hyp_len)

def closest_ref_length(references, hyp_len):
    """
    This function finds the reference that is the closest length to the
    hypothesis. The closest reference length is referred to as *r* variable
    from the brevity penalty formula in Papineni et. al. (2002)

    :param references: A list of reference translations.
    :type references: list(list(str))
    :param hyp_len: The length of the hypothesis.
    :type hyp_len: int
    :return: The length of the reference that's closest to the hypothesis.
    :rtype: int
    """
    ref_lens = (len(reference) for reference in references)
    closest_ref_len = min(ref_lens, key=lambda ref_len:
    (abs(ref_len - hyp_len), ref_len))
    return closest_ref_len

def avg_ref_length(references, hyp_len):
    return int(np.mean([len(reference) for reference in references]))+1

def reweight_counter(counts, entities, beta=2):
    #return counts
    sum_entities = 1.0 * sum(entities.values())
    new_counts = {}
    weights = {}
    for ngram, count in counts.items():
        hit = 0
        for w in ngram:
            if w in entities:
                hit += entities[w] / sum_entities
        weight = 1 + beta * hit #) if hit > 0 else 1
        new_counts[ngram] = count * weight
        weights[ngram] = weight
    return new_counts 

def reweight_two_counter(hit_cnt, all_cnt, entities, beta):
    sum_entities = 1.0 * sum(entities.values())
    new_hit_cnt = {}
    new_all_cnt = {}
    weights = {}
    for ngram, count in hit_cnt.items():
        hit = 0
        for w in ngram:
            if w in entities:
                hit += entities[w] / sum_entities
        weight = 1 + beta * hit
        new_hit_cnt[ngram] = hit_cnt[ngram] * weight
        new_all_cnt[ngram] = all_cnt[ngram] * weight 
        weights[ngram] = weight
    return new_hit_cnt, new_all_cnt 

def modified_precision(hyp_ents, refs, entities, n, beta=2):
    counts = Counter(ngrams(hyp_ents, n) if len(hyp_ents) >=n else Counter())
    max_counts = {}
    for ri, r in enumerate(refs):
        ref_counts =  (Counter(ngrams(r, n) if len(r) >= n else Counter()))
        for ngram in counts:
            max_counts[ngram] = max(max_counts.get(ngram, 0), ref_counts.get(ngram, 0))
    clipped_counts = {ngram: min(count, max_counts[ngram]) for ngram, count in counts.items()}
    clipped_counts, counts = reweight_two_counter(clipped_counts, counts, entities, beta)

    numerator = sum(clipped_counts.values())
    denominator = max(1, sum(counts.values()))
    return numerator, denominator

def compute_f1_reward(tok, entities, vocab):
    hyp_ents = [vocab.word2lemma[w] for w in tok if vocab.word2pos[w] in ['NOUN', 'PRON']]
    num_ents = sum([entities[k] for k in entities]) + 1e-10
    num_hyp = len(hyp_ents) + 1e-10
    hit = 0.0
    hit_ents = set()
    for e in hyp_ents:
        if e in entities:
            hit += 1
            hit_ents.add(e)
    hit_cnt = float(sum([entities[k] for k in hit_ents]))
    precision = hit / num_hyp 
    recall = hit_cnt / num_ents 
    f1 = 2.0 * precision * recall / (precision + recall + 1e-10)
    return f1

def compute_relevance_reward(hyp, refs, entities, vocab, beta=5, weights=(0.7, 0.3), auto_reweigh=False):
    """
    Args:
        hyp: list of words
        refs: list of list of words
        entities: 
        vocab:
    Return:
        s: scalar score
        stat: statistic
    """
    hyp_ent = []
    for w in hyp:
        if w not in vocab.word2pos:
            hyp_ent.append(w)
        else:
            hyp_ent.append(vocab.word2lemma[w] if vocab.word2pos[w] == 'NOUN' else w)
    p_num = Counter()
    p_dem = Counter()
    hyp_lengths, ref_lengths = 0, 0

    n_gram = len(weights)
    for i in range(1, 1+n_gram):
        numerator, denominator = modified_precision(hyp_ent, refs, entities, i, beta)
        p_num[i] = numerator
        p_dem[i] = denominator
    hyp_len = len(hyp_ent)
    hyp_lengths += hyp_len 
    ref_lengths += max(5, closest_ref_length(refs, hyp_len))

    bp = brevity_penalty(ref_lengths, hyp_lengths)
    p_n = [(p_num[i]+1) / (p_dem[i]+1) for i in range(1, 1+n_gram)]

    if p_num[1] == 0:
        return (0, [0] * (len(weights) +1))
    if auto_reweigh:
        if hyp_lengths < 4 and weights == (0.25, 0.25, 0.25, 0.25):
            weights = (1 / hyp_lengths,) * hyp_lengths

    s = (w_i * math.log(p_i) for w_i, p_i in zip(weights, p_n))
    s = bp * math.exp(math.fsum(s))
    stat = p_n + [bp]
    return s, stat
    
def load(p_file, id_file, vocab_file, sample_file, img_dict, out_file):
    img2idx = json.load(open(img_dict))
    print('img2idx', list(img2idx.keys())[0:10])
    

    lemma, pos, tok = pickle.load(open(p_file, 'rb'))
    ids = [l.strip() for l in open(id_file)]
    assert len(ids) == len(lemma)

    refs = defaultdict(list)
    miss = []
    new_propn = []
    for j, ls, ps, ts in zip(ids, lemma, pos, tok):
        if j not in img2idx:
            miss.append(j)
            continue
        i = img2idx[j]
        if i not in refs:
            refs[i] = []
        for l, p, t in zip(ls, ps, ts):
            if p == 'NOUN':
                refs[i].append(l)
            elif p == 'PRON':
                if t in propn: 
                    refs[i].append(propn[t])
                else:
                    propn[t] = t
                    refs[i].append(t)
                    new_propn.append(t)
            # TODO: PROPN PRON
    print('new', len(new_propn), new_propn)
    refs = {i: Counter(v) for i, v in refs.items()}
    print('refs keys', list(refs.keys())[0:5])
    print('miss', set(miss))
    print('len(refs)', len(refs))

    vocab = torch.load(vocab_file)
    samples = defaultdict(list)
    for l in open(sample_file):
        its = l.strip().split('|||')
        img_id = int(its[0].strip())
        samples[img_id].append(its[1].strip().split(' ')[1:-1])

    fout = open(out_file, 'w')
    cnt = total = 0.0
    x = []
    for idx in samples:
        fout.write('refs ||| ' + str(refs[idx]) + '\n')
        f1_pairs = []
        for s in samples[idx]:
            f1 = compute_f1_reward(s, refs[idx], vocab)
            f1_pairs.append((f1, " ".join(s)))
            if f1 > 1e-8:
                cnt += 1
            total += 1
            x.append(f1)
        for (f1, s) in sorted(f1_pairs, reverse=True):
            fout.write("%d ||| %.4f ||| %s\n" % (idx, f1, s))
    fout.close()
    print('percent: {}/{} = {}'.format(cnt, total, cnt/total))
    print('mean(f1)=', np.mean(x))

def process_data(p_file, id_file, img_dict, vocab_file, out_p_file):
    img2idx = json.load(open(img_dict))
    print('img2idx', list(img2idx.keys())[0:10])
    

    lemma, pos, tok = pickle.load(open(p_file, 'rb'))
    ids = [l.strip() for l in open(id_file)]
    assert len(ids) == len(lemma)

    refs = defaultdict(list)
    refs_multi = defaultdict(list)
    refs_lem_sent = defaultdict(list)
    refs_tok_sent = defaultdict(list)
    refs_pos_sent = defaultdict(list)
    miss = []
    new_propn = []
    out_ids, out_lemma = [], []
    for j, ls, ps, ts in zip(ids, lemma, pos, tok):
        if j not in img2idx:
            miss.append(j)
            continue
        i = img2idx[j]
        if i not in refs:
            refs[i] = []
            refs_multi[i] = []
            refs_lem_sent[i] = []
            refs_tok_sent[i] = []
            refs_pos_sent[i] = []
        new_ls = []
        ent_i = []
        for l, p, t in zip(ls, ps, ts):
            if p == 'NOUN':
                refs[i].append(l)
                new_ls.append(l)
                ent_i.append(l)
            else:
                new_ls.append(t)
            
        # refs_multi[i].append(Counter(ent_i))
        refs_multi[i].append(ent_i)
        refs_lem_sent[i].append(' '.join(new_ls))
        refs_tok_sent[i].append(' '.join(ts))
        refs_pos_sent[i].append(' '.join(ps))

        out_ids.append(i)
        out_lemma.append(new_ls)
    
    print('new', len(new_propn), new_propn)
    refs = {i: Counter(v) for i, v in refs.items()}
    print('refs keys', list(refs.keys())[0:5])
    print('miss', len(miss))
    print('len(refs)', len(refs))
    print('img2idx', len(img2idx))
    print('save data to ', out_p_file)
    print('no. images in refs multi', len(refs_multi))
    print('average no. refs per image: ', np.mean([len(v) for k,v in refs_multi.items()]))
    keys = list(refs_multi.keys())
    print('refs multi: ')
    for k in keys[0:2]:
        print('   ', k, refs_multi[k])
        print('   ', refs_lem_sent[k])
        print('   ', refs_tok_sent[k])
        print('   ', refs_pos_sent[k])
        print('   ', refs[k], '--------------------\n')
    pickle.dump([out_ids, out_lemma, pos, tok, refs_multi, refs_pos_sent, refs_lem_sent, refs_tok_sent, refs], open(out_p_file, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)





