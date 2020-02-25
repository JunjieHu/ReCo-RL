#from __future__ import print_function
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.spice.spice import Spice

import sys
import pickle

def score_func(ref, hypo, idx=None):
    """
    ref, dictionary of reference sentences (id, sentence)
    hypo, dictionary of hypothesis sentences (id, sentence)
    score, dictionary of scores
    """
    scorers = [
        (Spice(), "SPICE"),
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr"),
        (Meteor(),"METEOR")
    ]
    final_scores = {}
    if idx is not None:
        scorers = [scorers[idx]]
    for scorer, method in scorers:
        score, scores = scorer.compute_score(ref, hypo)
        print('score', method, score)
        if type(score) == list:
            for m, s in zip(method, score):
                final_scores[m] = s
        else:
            final_scores[method] = score
    return final_scores


if __name__ == '__main__':
    ref_file = sys.argv[1]
    hyp_file = sys.argv[2]
    print(ref_file)
    print(hyp_file)

    refs = pickle.load(open(ref_file, 'rb'))[3]
    print('len(refs)', len(refs))

    hyps = {}
    for line in open(hyp_file, 'r'):
        key, hyp = line.strip().split('\t')
        if key not in hyps:
            hyps[key] = [hyp.strip()]
    print('len(hyps)', len(hyps))
    print('keys', set(hyps.keys()) == set(refs.keys()))

    scores = score_func(refs, hyps)
    print(scores)



