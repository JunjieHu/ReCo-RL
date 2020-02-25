from __future__ import print_function
import torch
import numpy as np
import pickle
import scipy.io
from vocab import VocabEntry
import os.path
import json
import h5py


class VistDataLoader(object):
    def __init__(self, args):
        self.args = args
        self.vocab = None
        self.load_data()
        self.load_vocab()

    @staticmethod
    def add_args(parser):
        """ Add Vist-specific arguments to the parser """
        parser.add_argument('--dataset_file', type=str, default=None, help='path to train/dev/test sets')
        parser.add_argument('--ref_file', type=str, default=None, help='path to train/dev/test ref sentences')
        parser.add_argument('--img_feats_train_file', type=str, default=None, help='path to train image features')
        parser.add_argument('--img_feats_dev_file', type=str, default=None, help='path to dev image features')
        parser.add_argument('--img_feats_test_file', type=str, default=None, help='path to test image features')
        parser.add_argument('--rebuild_vocab', default=False, action='store_true', help='whether to rebuild the vocab or not')
        #parser.add_argument('--tag_feat_file', type=str, default=None, help='path to tag features')
        parser.add_argument('--story_h5', default=None, type=str)
        parser.add_argument('--story_line', default=None, type=str)
        # parser.add_argument('--data_dir', default=None, type=str)
        return parser

    def load_data(self):
        # load captions and references
        story_h5_path = os.path.join(self.args.data_dir, 'story.h5') if not self.args.story_h5 else self.args.story_h5
        story_h5 = h5py.File(story_h5_path, 'r', driver='core')['story']
        story_line_path = os.path.join(self.args.data_dir, 'story_line.json') if not self.args.story_line else self.args.story_line
        story_line = json.load(open(story_line_path))
        self.id2w = {int(k): v for k, v in story_line['id2words'].items()}
        self.w2id = {k: int(v) for k, v in story_line['words2id'].items()}

        self.trn = self.transform_story(story_h5, story_line, 'train')
        self.dev = self.transform_story(story_h5, story_line, 'val')
        self.tst = self.transform_story(story_h5, story_line, 'test')

    def transform_story(self, story_h5, story_line, split):
        # load entities
        entities_file = os.path.join(self.args.data_dir, 'spacy/{}-spacy-lemma-lg-mix.p'.format(split))
        print('loading entities from ' + entities_file)
        entities = pickle.load(open(entities_file, 'rb'))[-1]
        print('# of images with entities: ', len(entities))
        # load data if the file exists
        data_file = os.path.join(self.args.data_dir, '{}_dataset.p'.format(split))
        print('data_file', data_file)
        if os.path.isfile(data_file):
            [captions, imgs, album_ids, reference] = pickle.load(open(data_file, 'rb'))
            img_file = os.path.join(self.args.data_dir, 'img_{}_feat.npy'.format(split))
            img_feats = np.load(img_file)
            print('Split', split, '# captions', len(captions), ', # img streams', len(imgs), len(imgs[0]), '# reference', len(reference))
            return captions, imgs, reference, img_feats, album_ids, entities

        # Generate the data file
        id2w = self.id2w
        data = story_line[split]
        captions = []
        imgs = []
        img_id_set = set()
        album_ids = []
        for sid, story in data.items():
            story_ids = story_h5[story['text_index']]
            sp = [['<s>'] + [id2w[idx] for idx in sent if idx != 0] + ['</s>'] for sent in story_ids]
            captions.append(sp)
            album_ids.append(story['album_id'])
            img_ids = story['flickr_id']
            img_id_set |= set(img_ids)
            imgs.append(img_ids)
        print('captions', len(captions), len(captions[0]))
        print('imgs', len(imgs), len(imgs[0]))

        # generate reference
        if not os.path.exists(self.args.data_dir):
            os.makedirs(self.args.data_dir)

        ref_file = os.path.join(self.args.data_dir, '{}_reference.json'.format(split))
        if os.path.isfile(ref_file):
            reference = json.load(open(ref_file, 'r'))
        else:
            reference = {}
            for story in story_line[split].values():
                if story['album_id'] not in reference:
                    reference[story['album_id']] = [story['origin_text']]
                else:
                    reference[story['album_id']].append(story['origin_text'])
            with open(ref_file, 'w') as f:
                json.dump(reference, f)

        # form image feature vectors
        img_file = os.path.join(self.args.data_dir, 'img_{}_feat.npy'.format(split))
        if os.path.isfile(img_file):
            print('loading feats from %s' % img_file)
            img_feats = np.load(img_file)
            print('loading img dict from %s' % (img_file+'.json'))
            img_dict = json.load(open(img_file+'.json', 'r'))
            img_dict = {k: int(v) for k, v in img_dict.items()}
        else:
            img_feats = []
            img_dict = {}
            cnt = 0
            for img_id in img_id_set:
                fc_path = os.path.join(self.args.data_dir, 'resnet_features/fc', split, img_id + '.npy')
                feat_fc = np.load(fc_path)
                img_feats.append(feat_fc)
                img_dict[img_id] = cnt
                cnt += 1
            img_feats = np.stack(img_feats, axis=0)  # [num_img, img_size]
            print('saving feats to %s' % img_file)
            np.save(img_file, img_feats)
            print('saving img dict from %s' % (img_file+'.json'))
            json.dump(img_dict, open(img_file+'.json', 'w'))
        print('imgs_feat', img_feats.shape)
        #Important! Reindexing the image ids! different from reading the original one
        imgs = [[img_dict[i] for i in img_id] for img_id in imgs]  # [num_story, num_seq]
        print('len(imgs)', len(imgs), len(imgs[0]))

        data_file = os.path.join(self.args.data_dir, '{}_dataset.p'.format(split))
        if not os.path.isfile(data_file):
            pickle.dump([captions, imgs, album_ids, reference], open(data_file, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        return captions, imgs, reference, img_feats, album_ids, entities


    def load_vocab(self):
        # Load the vocabulary or create vocabulary if not exists
        if self.args.vocab is not None:
            if not os.path.isfile(self.args.vocab):
                print('create new vocab and save to %s' % self.args.vocab)
                corpus = []
                for story in self.trn[0]:
                    for sent in story:
                        corpus.append(sent)
                if self.args.rebuild_vocab:
                    self.vocab = VocabEntry.from_corpus(corpus, 50000, remove_singleton=not self.args.include_singleton)
                else:
                    self.vocab = VocabEntry.from_dict(self.w2id)
                torch.save(self.vocab, self.args.vocab)
            else:
                self.vocab = torch.load(self.args.vocab)
        else:
            print('vocab file is required')
            exit(0)

    def _batch_slice(self, batch_size, sort=True):
        pass

    @staticmethod
    def pad_sequence(seqs, pad_token):
        max_len = max(max(len(s) for s in story) for story in seqs)
        new_seqs = []
        for story in seqs:
            new_story = []
            for sent in story:
                new_story.append(sent + [pad_token] * (max_len - len(sent)))
            new_seqs.append(new_story)
        return new_seqs

    @staticmethod
    def data_iter(data, vocab, batch_size, shuffle=True, cuda=False, is_test=False):
        # print('cuda in dataloader', cuda)
        batched_data = list(zip(data[0], data[1]))
        img_entities = data[5]  # key: img idx, value: Counter of entities
        k = list(img_entities.keys())
        print('img entities in batch', len(img_entities))
        print('key', k[0:10], img_entities[k[0]])
        img_feats = data[3]
        if shuffle:
            np.random.shuffle(batched_data)
        batch_num = int(np.ceil(len(batched_data) / float(batch_size)))
        for i in range(batch_num):
            cur_batch_size = batch_size if i < batch_num - 1 else len(batched_data) - batch_size * i
            trg_seqs = [batched_data[i * batch_size +b][0] for b in range(cur_batch_size)]
            src_imgs = [batched_data[i * batch_size +b][1] for b in range(cur_batch_size)]

            # [batch_size, num_seq (5), max_num_words]
            trg_seqs_idx = [[[vocab[w] for w in sent] for sent in story] for story in trg_seqs]
            num_trg_word = sum([sum([len(sent[1:]) for sent in story]) for story in trg_seqs])  # skip leading <s>
            num_trg_seq = len(trg_seqs_idx) * len(trg_seqs_idx[0])
            trg_seqs_idx = VistDataLoader.pad_sequence(trg_seqs_idx, vocab['<pad>'])

            # [batsh_size, num_seq (5), img_feat_dim]
            src_img_feats = [[img_feats[idx] for idx in story] for story in src_imgs]

            trg_seqs_var = torch.LongTensor(trg_seqs_idx)
            src_seqs_var = torch.FloatTensor(src_img_feats)
            if cuda:
                trg_seqs_var = trg_seqs_var.cuda()
                src_seqs_var = src_seqs_var.cuda()

            # load entities of each image
            src_img_entities = [[img_entities[idx] for idx in story] for story in src_imgs]

            yield {
                    'src_seq': src_seqs_var, 'src_lengths': None,
                    'trg_seq': trg_seqs_var[:,:,:-1],
                    'target': trg_seqs_var[:,:,1:],
                    'num_trg_word': num_trg_word, 'num_trg_seq': num_trg_seq,
                    'src_img_entities': src_img_entities,
                    'src_img_ids': src_imgs
                   }

    @staticmethod
    def ref_iter(data, batch_size, cuda=False):
        captions, imgs, reference, img_feats, album_ids, img_entities = data
        num_example = len(imgs)
        batch_num = int(np.ceil(len(imgs) / float(batch_size)))
        for i in range(batch_num):
            cur_batch_size = batch_size if i < batch_num - 1 else num_example - batch_size * i
            src_imgs = [imgs[i * batch_size + b] for b in range(cur_batch_size)]  # [batch, num_seq(5)]
            # ref_seqs = [reference[album_ids[i * batch_size + b]] for b in range(cur_batch_size)]  # [batch, num_ref, story_len]
            ref_seqs = [captions[i*batch_size + b] for b in range(cur_batch_size)]
            album = [album_ids[i * batch_size + b] for b in range(cur_batch_size)]  # [batch]

            src_img_feats = [[img_feats[idx] for idx in story] for story in src_imgs]
            src_seqs_var = torch.FloatTensor(src_img_feats)
            if cuda:
                src_seqs_var = src_seqs_var.cuda()

            src_img_entities = [[img_entities[idx] for idx in story] for story in src_imgs]
            yield {
                    'src_seq': src_seqs_var, 'src_lengths': None,
                    'trg_seq': ref_seqs,
                    'album': album,
                    'src_img_entities': src_img_entities,
                    'src_img_ids': src_imgs
            }

