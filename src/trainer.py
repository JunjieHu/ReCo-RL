from __future__ import print_function
import math
import torch
import numpy as np

class Trainer(object):

    def __init__(self, args, model, criterion):
        self.args = args

        self.model = model
        self.criterion = criterion

        # initialize optimizer and learning rate
        if self.args.optim == 'sgd':
            self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        else:
            self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, amsgrad=True)
            
        # initialize loger
        self._num_updates = 0
        if args.cuda:
            self.model = self.model.cuda()
            self.criterion = self.criterion.cuda()
            
    def save_checkpoint(self, filename):
        pass

    def load_checkpoint(self, filename):
        pass

    def train_step(self, sample, objective='MLE'):
        self.model.train()
        self.optimizer.zero_grad()
        # forward pass
        if objective == 'MLE':
            loss, log_outputs = self._forward(sample)
        elif objective == 'REINFORCE':
            loss, log_outputs = self._reinforce(sample)
        elif objective == 'MIXER':
            mle_loss, log_outputs = self._forward(sample)
            rl_loss, _ = self._reinforce(sample)
            loss = (1 - self.args.rl_weight) * mle_loss + self.args.rl_weight * rl_loss
            log_outputs['rl_loss'] = rl_loss
        elif objective.startswith('REWARD'):
            loss, log_outputs = self._reward(sample, torch.nn.BCELoss())
        else:
            print('Specify objective: MLE or REINFORCE')
            exit(0)

        # backward pass
        grad_norm = self._backward(loss)
        return loss, log_outputs

    def _forward(self, sample):
        # get the model's prediction
        lprobs = self.model(sample['src_seq'], sample['src_lengths'], sample['trg_seq'])
        target = sample['target']

        # get the loss 
        loss = self.criterion(lprobs.contiguous().view(-1, lprobs.size(-1)), target.contiguous().view(-1))
        loss = loss / sample['num_trg_seq']
        logging_outputs = {'loss': loss, 'nsample': sample['target'].size(0)}
        return loss, logging_outputs

    def _backward(self, loss):
    	loss.backward()

    	if self.args.clip_norm > 0:
    		grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_norm)
    	else:
            grad_norm = math.sqrt(sum(p.grad.data.norm()**2 for p in self.model.parameters()))
    	self.optimizer.step()
    	self._num_updates += 1
    	return grad_norm

    def valid_step(self, sample, objective='MLE'):
        self.model.eval()
        with torch.no_grad():
            if objective == 'MLE':
                loss, log_outputs = self._forward(sample)
            elif objective == 'REINFORCE':
                loss, log_outputs = self._reinforce(sample)
            elif objective == 'MIXER':
                mle_loss, log_outputs = self._forward(sample)
                rl_loss, _ = self._reinforce(sample)
                loss = (1 - self.args.rl_weight) * mle_loss + self.args.rl_weight * rl_loss
                log_outputs['rl_loss'] = rl_loss
            elif objective.startswith('REWARD'):
                loss, log_outputs = self._reward(sample, torch.nn.BCELoss())
            # loss, log_outputs = self._forward(sample, eval=True)
        return loss, log_outputs

    def _reinforce(self, sample):
        self.model.train()
        self.optimizer.zero_grad()
        loss = self.model.reinforce(sample, self.args.sample_size, self.args.decode_max_length, self.args.rl_reward)
        logging_outputs = {'loss': loss, 'nsample': sample['target'].size(0)}
        return loss, logging_outputs 

    def _reward(self, sample, criteria):
        self.model.train()
        self.optimizer.zero_grad()
        return self.model.train_reward(sample, criteria)
        

