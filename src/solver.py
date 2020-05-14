# Created on 2018/12
# Author: Kaituo XU

import os
import time

import torch
from tqdm import tqdm

from pit_criterion import cal_loss


class Solver(object):
    
    def __init__(self, data, model, optimizer, args):
        self.tr_loader = data['tr_loader']
        self.cv_loader = data['cv_loader']
        self.model = model
        self.optimizer = optimizer

        # Training config
        self.use_cuda = args.use_cuda
        self.epochs = args.epochs
        self.half_lr = args.half_lr
        self.early_stop = args.early_stop
        self.max_norm = args.max_norm * args.batch_per_step
        self.batch_per_step = args.batch_per_step
        # save and load model
        self.save_folder = args.save_folder
        self.checkpoint = args.checkpoint
        self.valid_interval = args.valid_interval
        self.continue_from = args.continue_from
        self.model_path = args.model_path
        # logging

        self.tr_loss = torch.Tensor(self.epochs)
        self.cv_loss = torch.Tensor(self.epochs)

        self._reset()

    def _reset(self):
        # Reset
        if self.continue_from:
            print(f'Loading checkpoint model {self.continue_from}')
            package = torch.load(self.continue_from)
            self.model.module.load_state_dict(package['state_dict'])
            self.optimizer.load_state_dict(package['optim_dict'])
            self.start_epoch = int(package.get('epoch', 1))
            self.tr_loss[:self.start_epoch] = package['tr_loss'][:self.start_epoch]
            self.cv_loss[:self.start_epoch] = package['cv_loss'][:self.start_epoch]
            self.batches = int(package.get('batches', 0))
        else:
            self.start_epoch = 0
            self.batches = 0
        # Create save folder
        os.makedirs(self.save_folder, exist_ok=True)
        self.prev_val_loss = float("inf")
        self.best_val_loss = float("inf")
        self.halving = False
        self.val_no_impv = 0
        self.optimizer.zero_grad()

    def train(self):
        # Train model multi-epoches
        for epoch in range(self.start_epoch, self.epochs):
            # Train one epoch
            self.model.train()  # Turn on BatchNorm & Dropout
            start = time.time()
            tr_avg_loss = self._run_one_epoch(epoch)

            # Save model each epoch
            if self.checkpoint:
                file_path = os.path.join(
                    self.save_folder, 'epoch%d.pth.tar' % (epoch + 1))
                torch.save(self.model.module.serialize(self.model.module,
                                                       self.optimizer,
                                                       epoch + 1,
                                                       self.batches,
                                                       tr_loss=self.tr_loss,
                                                       cv_loss=self.cv_loss),
                           file_path)
                print('Saving checkpoint model to %s' % file_path)

            # Validation
            if (epoch + 1) % self.valid_interval == 0:
                print('Cross validation...')
                self.model.eval()  # Turn off Batchnorm & Dropout
                val_loss = self._run_one_epoch(epoch, cross_valid=True)

                # Adjust learning rate (halving)
                if self.half_lr:
                    if val_loss >= self.prev_val_loss:
                        self.val_no_impv += self.valid_interval
                        if self.val_no_impv >= 3:
                            self.halving = True
                        if self.val_no_impv >= 10 and self.early_stop:
                            print("No imporvement for 10 epochs, early stopping.")
                            break
                    else:
                        self.val_no_impv = 0
                if self.halving:
                    optim_state = self.optimizer.state_dict()
                    optim_state['param_groups'][0]['lr'] = \
                        optim_state['param_groups'][0]['lr'] / 2.0
                    self.optimizer.load_state_dict(optim_state)
                    print('Learning rate adjusted to: {lr:.6f}'.format(
                        lr=optim_state['param_groups'][0]['lr']))
                    self.halving = False
                self.prev_val_loss = val_loss

            # Save the best model
            self.tr_loss[epoch] = tr_avg_loss
            try:
                self.cv_loss[epoch] = val_loss
            except:
                import math
                self.cv_loss[epoch] = math.inf

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                file_path = os.path.join(self.save_folder, self.model_path)
                torch.save(
                    self.model.module.serialize(
                        self.model.module,
                        self.optimizer, epoch + 1,
                        self.batches,
                        tr_loss=self.tr_loss,
                        cv_loss=self.cv_loss
                    ),
                    file_path
                )
                print("Find better validated model, saving to %s" % file_path)

    def _run_one_epoch(self, epoch, cross_valid=False):
        start = time.time()
        total_loss = 0

        data_loader = self.tr_loader if not cross_valid else self.cv_loader

        with tqdm(total=len(data_loader.dataset), unit="iter") as t:
            for i, (data) in enumerate(data_loader):
                self.batches += 1

                t.set_description(f"Epoch {epoch}") if not cross_valid else t.set_description("Validation")

                padded_mixture, mixture_lengths, padded_source = data
                if self.use_cuda:
                    padded_mixture = padded_mixture.cuda()
                    mixture_lengths = mixture_lengths.cuda()
                    padded_source = padded_source.cuda()
                estimate_source = self.model(padded_mixture)
                loss, max_snr, estimate_source, reorder_estimate_source = cal_loss(padded_source, estimate_source, mixture_lengths)

                if not cross_valid:
                    loss.backward()

                    if self.batches % self.batch_per_step == 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                total_loss += loss.item()

                postfix = f"Iter {i + 1} | Steps {self.batches // self.batch_per_step} | Average Loss {(total_loss / (i + 1)):.{3}} | Loss {loss.item():.{6}}"
                t.set_postfix_str(postfix)
                t.update(padded_mixture.size(0))

        return total_loss / (i + 1)
