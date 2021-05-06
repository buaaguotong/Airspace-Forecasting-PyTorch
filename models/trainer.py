import os
import time
import torch
import copy

import numpy as np

from tqdm import tqdm
from datetime import datetime
from utils.get_logger import get_logger
from utils.get_acc import get_acc

class Trainer:
    def __init__(self, 
                 model, 
                 loss, 
                 optimizer, 
                 lr_scheduler,
                 train_loader, 
                 val_loader, 
                 test_loader, 
                 args):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.args = args

        self.train_per_epoch = len(train_loader)
        self.val_per_epoch = len(val_loader)
        if args.checkpoint:
            self.log_dir = os.path.join(args.log_dir, f'log_{datetime.now().strftime("%m%d%H%M%S")}_test')
        else:
            self.log_dir = os.path.join(args.log_dir, f'log_{datetime.now().strftime("%m%d%H%M")}')
        if os.path.isdir(self.log_dir) == False:
            os.makedirs(self.log_dir, exist_ok=True)
        self.logger = get_logger(self.log_dir, debug=args.debug)
        self.best_path = os.path.join(self.log_dir, f'best_model_{datetime.now().strftime("%m%d%H%M")}.pth')


    def val_epoch(self, epoch):
        self.model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for (val_input, val_label) in iter(self.val_loader):
                val_output = self.model(val_input)
                pred, true = val_output, val_label
                loss = self.loss(pred, true)
                total_val_loss += loss.item()
        val_loss = total_val_loss / self.val_per_epoch
        return val_loss


    def train_epoch(self, epoch):
        self.model.train()
        total_train_loss = 0
        for (train_input, train_label) in tqdm(iter(self.train_loader)):
            self.optimizer.zero_grad()
            train_output = self.model(train_input)
            pred, true = train_output, train_label
            loss = self.loss(pred, true)
            loss.backward()
            if self.args.grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            self.optimizer.step()
            total_train_loss += loss.item()
        train_epoch_loss = total_train_loss / self.train_per_epoch
        self.lr_scheduler.step()
        return train_epoch_loss


    def train(self,):
        train_loss_list, val_loss_list = [], []
        best_loss, not_improved_count = float('inf'), 0

        start_time = time.time()
        for epoch in range(1, self.args.epochs + 1):
            train_epoch_loss = self.train_epoch(epoch)
            val_epoch_loss = self.val_epoch(epoch)
            train_loss_list.append(train_epoch_loss)
            val_loss_list.append(val_epoch_loss)

            if val_epoch_loss < best_loss:
                best_loss, not_improved_count, best_state = val_epoch_loss, 0, True
            else:
                not_improved_count, best_state = not_improved_count + 1, False
            if not_improved_count == self.args.patience:
                self.logger.info(f'Early stop at epoch: {epoch}.')
                break

            if best_state == True:
                self.logger.info(f'epoch {epoch:03d} | train_loss: {train_epoch_loss:.6f} | ' +
                    f'val_loss: {val_epoch_loss:.6f} | lr: {self.lr_scheduler.get_last_lr()[0]:.6f} (Current best)')
                best_model = copy.deepcopy(self.model.state_dict())
            else:
                self.logger.info(f'epoch {epoch:03d} | train_loss: {train_epoch_loss:.6f} | ' +
                    f'val_loss: {val_epoch_loss:.6f} | lr: {self.lr_scheduler.get_last_lr()[0]:.6f}')

        training_time = time.time() - start_time
        torch.save(best_model, self.best_path)
        self.logger.info('*********************** Train Finish ***********************')
        self.logger.info(f'Saving current best model at {self.best_path}')
        self.logger.info(f"Total training time: {(training_time / 60):.4f} min, best loss: {best_loss:.6f}")


    def test(self):
        self.logger.info('*********************** Test Step ***********************')
        outputs, labels = [], []
        self.model.eval()
        with torch.no_grad():
            for (test_input, test_label) in iter(self.test_loader):
                test_output = self.model(test_input)
                pred, true = test_output, test_label
                outputs.append(pred.squeeze())
                labels.append(true.squeeze())

        outputs_stack = torch.cat(outputs, dim=0).permute(1,0,2).cpu().detach().numpy()
        labels_stack = torch.cat(labels, dim=0).permute(1,0,2).cpu().detach().numpy()
        result = {'prediction': outputs_stack, 'truth': labels_stack}
        np.savez_compressed(os.path.join(self.log_dir,f'tcn_predictions_{datetime.now().strftime("%m%d%H%M")}.npz'), **result)
        self.logger.info(f'prediction shape: {outputs_stack.shape} | truth shape: {labels_stack.shape}')
        get_acc(outputs_stack, labels_stack, self.logger)

        