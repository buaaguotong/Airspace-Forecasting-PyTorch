import time
import torch

class Trainer:
    def __init__(self, model, loss, optimizer, lr_scheduler,  
                    train_loader, val_loader, test_loader, scaler, args):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.scaler = scaler
        self.args = args

        self.train_per_epoch = len(train_loader)
        self.val_per_epoch = len(val_loader)


    def val_epoch(self, epoch):
        self.model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch_idx, (val_input, val_label) in enumerate(self.val_loader):
                model_output = self.model(val_input)
                label = self.scaler.inverse_transform(val_label)
                loss = self.loss(model_output, label)
                total_val_loss += loss.item()
        val_loss = total_val_loss / self.val_per_epoch
        return val_loss


    def train_epoch(self, epoch):
        self.model.train()
        total_train_loss = 0
        for batch_idx, (train_input, train_label) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            model_output = self.model(train_input)
            label = self.scaler.inverse_transform(train_label)
            loss = self.loss(model_output, label)
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
            print(f'epoch {epoch:03d} train_epoch_loss: {train_epoch_loss} val_epoch_loss: {val_epoch_loss}')
            if val_epoch_loss < best_loss:
                best_loss, not_improved_count, best_state = val_epoch_loss, 0, True
            else:
                not_improved_count, best_state = not_improved_count + 1, False
            if not_improved_count == self.args.patience:
                self.logger.info(f'Early stop at epoch: {epoch}.')
                break
        training_time = time.time() - start_time