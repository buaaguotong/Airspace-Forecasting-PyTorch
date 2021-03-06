import os
import time

import numpy as np
import torch
# from torch.utils.tensorboard import SummaryWriter

from libs import utils
from libs.metrics import masked_rmse_np, masked_mape_np, masked_mae_np
from model.pytorch.dcrnn_model import DCRNNModel
from model.pytorch.loss import masked_mae_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DCRNNSupervisor:
    def __init__(self, adj_mx, **kwargs):
        self._kwargs = kwargs
        self._data_kwargs = kwargs.get('data')
        self._model_kwargs = kwargs.get('model')
        self._train_kwargs = kwargs.get('train')

        self.max_grad_norm = self._train_kwargs.get('max_grad_norm', 1.)

        # logging.
        self._log_dir = self._get_log_dir(kwargs)
        # self.  = SummaryWriter('runs/' + self._log_dir)

        log_level = self._kwargs.get('log_level', 'INFO')
        self._logger = utils.get_logger(self._log_dir, __name__, 'info.log', level=log_level)

        # data set
        self._data = utils.load_dataset(**self._data_kwargs)
        self.standard_scaler = self._data['scaler']

        self.num_nodes = int(self._model_kwargs.get('num_nodes', 1))
        self.input_dim = int(self._model_kwargs.get('input_dim', 1))
        self.seq_len = int(self._model_kwargs.get('seq_len'))  # for the encoder
        self.output_dim = int(self._model_kwargs.get('output_dim', 1))
        self.use_curriculum_learning = bool(
            self._model_kwargs.get('use_curriculum_learning', False))
        self.horizon = int(self._model_kwargs.get('horizon', 1))  # for the decoder

        # setup model
        dcrnn_model = DCRNNModel(adj_mx, self._logger, **self._model_kwargs)
        self.dcrnn_model = dcrnn_model.cuda() if torch.cuda.is_available() else dcrnn_model
        self._logger.info("Model created")

        self._epoch_num = self._train_kwargs.get('epoch', 0)
        if self._epoch_num > 0:
            self.load_model()

    @staticmethod
    def _get_log_dir(kwargs):
        log_dir = kwargs['train'].get('log_dir')
        if log_dir is None:
            batch_size = kwargs['data'].get('batch_size')
            learning_rate = kwargs['train'].get('base_lr')
            max_diffusion_step = kwargs['model'].get('max_diffusion_step')
            num_rnn_layers = kwargs['model'].get('num_rnn_layers')
            rnn_units = kwargs['model'].get('rnn_units')
            structure = '-'.join(
                ['%d' % rnn_units for _ in range(num_rnn_layers)])
            horizon = kwargs['model'].get('horizon')
            filter_type = kwargs['model'].get('filter_type')
            filter_type_abbr = 'L'
            if filter_type == 'random_walk':
                filter_type_abbr = 'R'
            elif filter_type == 'dual_random_walk':
                filter_type_abbr = 'DR'
            run_id = 'dcrnn_%s_%d_h_%d_%s_lr_%g_bs_%d_%s/' % (
                filter_type_abbr, max_diffusion_step, horizon,
                structure, learning_rate, batch_size,
                time.strftime('%m%d%H%M%S'))
            base_dir = kwargs.get('base_dir')
            log_dir = os.path.join(base_dir, run_id)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        return log_dir

    def save_model(self, epoch):
        if not os.path.exists('models/'):
            os.makedirs('models/')

        config = dict(self._kwargs)
        config['model_state_dict'] = self.dcrnn_model.state_dict()
        config['epoch'] = epoch
        torch.save(config, 'models/epo%d.tar' % epoch)
        self._logger.info("Saved model at {}".format(epoch))
        return 'models/epo%d.tar' % epoch

    def load_model(self):
        self._setup_graph()
        assert os.path.exists('models/epo%d.tar' % self._epoch_num), 'Weights at epoch %d not found' % self._epoch_num
        checkpoint = torch.load('models/epo%d.tar' % self._epoch_num, map_location='cpu')
        self.dcrnn_model.load_state_dict(checkpoint['model_state_dict'])
        self._logger.info("Loaded model at {}".format(self._epoch_num))

    def _setup_graph(self):
        with torch.no_grad():
            self.dcrnn_model = self.dcrnn_model.eval()

            val_iterator = self._data['val_loader'].get_iterator()

            for _, (x, y) in enumerate(val_iterator):
                x, y = self._prepare_data(x, y)
                output = self.dcrnn_model(x)
                break

    def train(self, **kwargs):
        kwargs.update(self._train_kwargs)
        return self._train(**kwargs)

    def evaluate(self, dataset='val', batches_seen=0):
        """
        Computes mean L1Loss
        :return: mean L1Loss
        """
        with torch.no_grad():
            self.dcrnn_model = self.dcrnn_model.eval()

            val_iterator = self._data['{}_loader'.format(dataset)].get_iterator()
            y_truths, y_preds, losses = [], [], []

            for _, (x, y) in enumerate(val_iterator):
                x, y = self._prepare_data(x, y)

                output = self.dcrnn_model(x)
                loss = self._compute_loss(y, output)
                losses.append(loss.item())
                y_truths.append(y.cpu())
                y_preds.append(output.cpu())

            mean_loss = np.mean(losses)
            # self._writer.add_scalar('{} loss'.format(dataset), mean_loss, batches_seen)
            y_preds = np.concatenate(y_preds, axis=1)
            y_truths = np.concatenate(y_truths, axis=1)  # concatenate on batch dimension

            y_truths_scaled, y_preds_scaled = [], []
            for t in range(y_preds.shape[0]):
                y_truth = self.standard_scaler.inverse_transform(y_truths[t])
                y_pred = self.standard_scaler.inverse_transform(y_preds[t])
                y_truths_scaled.append(y_truth)
                y_preds_scaled.append(y_pred)

            return mean_loss, {'prediction': y_preds_scaled, 'truth': y_truths_scaled}

    def _train(self, base_lr,
               steps, patience=50, epochs=300, lr_decay_ratio=0.1, log_every=1, save_model=1,
               test_every_n_epochs=10, epsilon=1e-8, **kwargs):
        # steps is used in learning rate - will see if need to use it?
        min_val_loss = float('inf')
        wait = 0
        optimizer = torch.optim.Adam(self.dcrnn_model.parameters(), lr=base_lr, eps=epsilon)

        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=steps,
                                                            gamma=lr_decay_ratio)

        self._logger.info('Start training ...')

        # this will fail if model is loaded with a changed batch_size
        num_batches = self._data['train_loader'].num_batch
        self._logger.info("num_batches:{}".format(num_batches))

        batches_seen = num_batches * self._epoch_num

        for epoch_num in range(self._epoch_num, epochs):
            self.dcrnn_model = self.dcrnn_model.train()
            train_iterator = self._data['train_loader'].get_iterator()
            losses = []
            start_time = time.time()

            for _, (x, y) in enumerate(train_iterator):
                optimizer.zero_grad()
                x, y = self._prepare_data(x, y)
                output = self.dcrnn_model(x, y, batches_seen)
                if batches_seen == 0:
                    # this is a workaround to accommodate dynamically registered parameters in DCGRUCell
                    optimizer = torch.optim.Adam(self.dcrnn_model.parameters(), lr=base_lr, eps=epsilon)

                loss = self._compute_loss(y, output)
                self._logger.debug(loss.item())
                losses.append(loss.item())
                batches_seen += 1
                loss.backward()
                # gradient clipping - this does it in place
                torch.nn.utils.clip_grad_norm_(self.dcrnn_model.parameters(), self.max_grad_norm)
                optimizer.step()

            self._logger.info("epoch complete")
            lr_scheduler.step()
            self._logger.info("evaluating now!")
            val_loss, _ = self.evaluate(dataset='val', batches_seen=batches_seen)
            end_time = time.time()
            # self._writer.add_scalar('training loss', np.mean(losses), batches_seen)

            if (epoch_num % log_every) == log_every - 1:
                message = 'Epoch [{}/{}] ({}) train_mae: {:.4f}, val_mae: {:.4f}, lr: {:.6f}, ' \
                          '{:.1f}s'.format(epoch_num, epochs, batches_seen,
                                           np.mean(losses), val_loss, lr_scheduler.get_lr()[0],
                                           (end_time - start_time))
                self._logger.info(message)

            if (epoch_num % test_every_n_epochs) == test_every_n_epochs - 1:
                test_loss, _ = self.evaluate(dataset='test', batches_seen=batches_seen)
                message = 'Epoch [{}/{}] ({}) train_mae: {:.4f}, test_mae: {:.4f},  lr: {:.6f}, ' \
                          '{:.1f}s'.format(epoch_num, epochs, batches_seen,
                                           np.mean(losses), test_loss, lr_scheduler.get_lr()[0],
                                           (end_time - start_time))
                self._logger.info(message)

            if val_loss < min_val_loss:
                wait = 0
                if save_model:
                    model_file_name = self.save_model(epoch_num)
                    self._logger.info(
                        'Val loss decrease from {:.4f} to {:.4f}, '
                        'saving to {}'.format(min_val_loss, val_loss, model_file_name))
                min_val_loss = val_loss

            elif val_loss >= min_val_loss:
                wait += 1
                if wait == patience:
                    self._logger.warning('Early stopping at epoch: %d' % epoch_num)
                    break

    def _prepare_data(self, x, y):
        x, y = self._get_x_y(x, y)
        x, y = self._get_x_y_in_correct_dims(x, y)
        return x.to(device), y.to(device)

    def _get_x_y(self, x, y):
        """
        :param x: shape (batch_size, seq_len, num_sensor, input_dim)
        :param y: shape (batch_size, horizon, num_sensor, input_dim)
        :returns x shape (seq_len, batch_size, num_sensor, input_dim)
                 y shape (horizon, batch_size, num_sensor, input_dim)
        """
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        self._logger.debug("X: {}".format(x.size()))
        self._logger.debug("y: {}".format(y.size()))
        x = x.permute(1, 0, 2, 3)
        y = y.permute(1, 0, 2, 3)
        return x, y

    def _get_x_y_in_correct_dims(self, x, y):
        """
        :param x: shape (seq_len, batch_size, num_sensor, input_dim)
        :param y: shape (horizon, batch_size, num_sensor, input_dim)
        :return: x: shape (seq_len, batch_size, num_sensor * input_dim)
                 y: shape (horizon, batch_size, num_sensor * output_dim)
        """
        batch_size = x.size(1)
        x = x.view(self.seq_len, batch_size, self.num_nodes * self.input_dim)
        y = y[..., :self.output_dim].view(self.horizon, batch_size,
                                          self.num_nodes * self.output_dim)
        return x, y

    def _compute_loss(self, y_true, y_predicted):
        y_true = self.standard_scaler.inverse_transform(y_true)
        y_predicted = self.standard_scaler.inverse_transform(y_predicted)
        return masked_mae_loss(y_predicted, y_true)

    def evaluate_acc(self, dataset='test', batches_seen=0):
        with torch.no_grad():
            self.dcrnn_model = self.dcrnn_model.eval()

            data_iterator = self._data['{}_loader'.format(dataset)].get_iterator()
            y_truth, y_pred, losses = [], [], []

            for _, (x, y) in enumerate(data_iterator):
                x, y = self._prepare_data(x, y)
                output = self.dcrnn_model(x)
                y_truth.append(y.cpu())
                y_pred.append(output.cpu())

            y_pred, y_truth = np.concatenate(y_pred, axis=1), np.concatenate(y_truth, axis=1)  # concatenate on batch dimension
            y_truth_scaled, y_pred_scaled = \
                self.standard_scaler.inverse_transform(y_truth), self.standard_scaler.inverse_transform(y_pred) 
            
            for step in range(self.horizon):
                y_truth_scaled_step, y_pred_scaled_step = y_truth_scaled[step].reshape(-1), y_pred_scaled[step].reshape(-1)
                y_truth_cls, y_pred_cls = np.zeros(shape=y_truth_scaled_step.size), np.zeros(shape=y_pred_scaled_step.size)
                high_idx, normal_idx, low_idx = [], [], []
                for i in range(y_truth_cls.size):
                    if y_truth_scaled_step[i] < 2/3:
                        y_truth_cls[i] = 0
                        low_idx.append(i)
                    elif 2/3 <= y_truth_scaled_step[i] < 4/3:
                        y_truth_cls[i] = 1
                        normal_idx.append(i)
                    else:
                        y_truth_cls[i] = 2
                        high_idx.append(i)

                for i in range(y_pred_cls.size):
                    if y_pred_scaled_step[i] < 2/3:
                        y_pred_cls[i] = 0
                    elif 2/3 <= y_pred_scaled_step[i] < 4/3:
                        y_pred_cls[i] = 1
                    else:
                        y_pred_cls[i] = 2

                acc = sum(y_truth_cls==y_pred_cls)/(y_truth_cls.size)
                accH = sum(y_truth_cls[high_idx]==y_pred_cls[high_idx])/(y_truth_cls[high_idx].size)
                accN = sum(y_truth_cls[normal_idx]==y_pred_cls[normal_idx])/(y_truth_cls[normal_idx].size)
                accL = sum(y_truth_cls[low_idx]==y_pred_cls[low_idx])/(y_truth_cls[low_idx].size)
                # print(f'======= Class count: High {len(high_idx)}, Normal {len(normal_idx)}, Low {len(low_idx)}, All {y_truth_cls.size}')
                print(f'Horizon {step:02d}: Acc {acc:.4f}, AccH {accH:.4f}, AccN {accN:.4f}, AccL {accL:.4f}')
            return acc, {'prediction': y_pred_scaled, 'truth': y_truth_scaled}

    def evaluate_pred(self, dataset='test', batches_seen=0):
        with torch.no_grad():
            self.dcrnn_model = self.dcrnn_model.eval()

            data_iterator = self._data['{}_loader'.format(dataset)].get_iterator()
            y_truth, y_pred, losses = [], [], []

            for _, (x, y) in enumerate(data_iterator):
                x, y = self._prepare_data(x, y)
                output = self.dcrnn_model(x)
                y_truth.append(y.cpu())
                y_pred.append(output.cpu())

            y_pred, y_truth = np.concatenate(y_pred, axis=1), np.concatenate(y_truth, axis=1)  # concatenate on batch dimension
            y_truth_scaled, y_pred_scaled = \
                self.standard_scaler.inverse_transform(y_truth), self.standard_scaler.inverse_transform(y_pred) 
            mae, rmse, mape = [], [], []
            for step in range(self.horizon):
                y_truth_scaled_step, y_pred_scaled_step = y_truth_scaled[step].reshape(-1), y_pred_scaled[step].reshape(-1)
                mae.append(masked_mae_np(y_pred_scaled_step, y_truth_scaled_step, 0))
                rmse.append(masked_rmse_np(y_pred_scaled_step, y_truth_scaled_step, 0))
                mape.append(masked_mape_np(y_pred_scaled_step, y_truth_scaled_step, 0))
                print(f'Horizon {step:02d}: MAE {mae[-1]:.4f}, RMSE {rmse[-1]:.4f}, MAPE {mape[-1]:.4f}')
            return mae, {'prediction': y_pred_scaled, 'truth': y_truth_scaled}