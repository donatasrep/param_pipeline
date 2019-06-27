import os
import csv
import sys
import time
from collections import OrderedDict
from collections import Iterable
import numpy as np
import tensorflow as tf
from hyperopt import hp
from ruamel.yaml import YAML
from collections.abc import Iterable
import logging

from tensorflow.python.keras.backend import epsilon
from tensorflow.python.keras.callbacks import TensorBoard, CSVLogger, Callback

logging.basicConfig(stream=sys.stdout, format='%(asctime)s-%(levelname)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.DEBUG)

def coef_det_k(y_true, y_pred):
    """Computer coefficient of determination R^2
    """
    SS_res = tf.reduce_sum(tf.square(y_true - y_pred))
    SS_tot =  tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    return 1 - SS_res / (SS_tot + epsilon())


best_check = {'monitor': 'val_loss', 'verbose': 0, 'save_best_only': True, 'save_weights_only': True, 'mode': 'min'}
last_check = {'monitor': 'val_loss', 'verbose': 0, 'save_best_only': False, 'save_weights_only': True, 'mode': 'min'}


class TrainValTensorBoard(TensorBoard):
    """Keras callback to display train and validation metrics on same tensorboard plot
    """
    def __init__(self, log_dir='./tensorboard_logs', **kwargs):
        # Make the original `TensorBoard` log to a subdirectory 'training'
        training_log_dir = os.path.join(log_dir, 'training')
        super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)

        # Log the validation metrics to a separate subdirectory
        self.val_log_dir = os.path.join(log_dir, 'validation')

    def set_model(self, model):
        # Setup writer for validation metrics
        self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        super(TrainValTensorBoard, self).set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        # Pop the validation logs and handle them separately with
        # `self.val_writer`. Also rename the keys so that they can
        # be plotted on the same figure with the training metrics
        logs = logs or {}
        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}
        for name, value in val_logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.val_writer.add_summary(summary, epoch)
        self.val_writer.flush()

        # Pass the remaining logs to `TensorBoard.on_epoch_end`
        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        super(TrainValTensorBoard, self).on_epoch_end(epoch, logs)

        # print learning rate for testing purposes
        # learning rate decay in optimizers changes internally and is not shown
        # https://stackoverflow.com/questions/37091751/keras-learning-rate-not-changing-despite-decay-in-sgd
        lr = float(K.get_value(self.model.optimizer.lr))
        #print("Learning rate:", lr)

    def on_train_end(self, logs=None):
        super(TrainValTensorBoard, self).on_train_end(logs)
        self.val_writer.close()


class MyCSVLogger(CSVLogger):
    """Callback that streams epoch results to a csv file.

    Supports all values that can be represented as a string,
    including 1D iterables such as np.ndarray.

    # Example

    ```python
    csv_logger = my_CSVLogger('training.log')
    model.fit(X_train, Y_train, callbacks=[csv_logger])
    ```

    # Arguments
        filename: filename of the csv file, e.g. 'run/log.csv'.
        hpars: dictionary with additional values to store, e.g. current hyperparameters
        separator: string used to separate elements in the csv file.
        append: True: append if file exists (useful for continuing
            training). False: overwrite existing file,

    """

    def __init__(self, filename, hpars, test, separator=',', append=False):

        self.sep = separator
        self.filename = filename
        self.append = append
        self.writer = None
        self.keys = None
        self.append_header = True
        self.hpars = hpars
        #self.test_dict = {'test_loss': test.loss,'test_val_det_k': test.acc} 
        self.test = test
        self.file_flags = ''
        self._open_args = {'newline': '\n'}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.test_dict = {'test_loss': self.test.loss,'test_val_det_k': self.test.acc} 
        logs = {**logs, **self.hpars, **self.test_dict}

        def handle_value(k):
            is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
            if isinstance(k, Iterable) and not is_zero_dim_ndarray:
                return '"[%s]"' % (', '.join(map(str, k)))
            else:
                return k

        if self.keys is None:
            self.keys = sorted(logs.keys())

        if self.model.stop_training:
            # We set NA so that csv parsers do not fail for this last epoch.
            logs = dict([(k, logs[k] if k in logs else 'NA') for k in self.keys])

        if not self.writer:
            class CustomDialect(csv.excel):
                delimiter = self.sep

            fieldnames = ['epoch'] + self.keys

            self.writer = csv.DictWriter(self.csv_file,
                                         fieldnames=fieldnames,
                                         dialect=CustomDialect)
            if self.append_header:
                self.writer.writeheader()

        row_dict = OrderedDict({'epoch': epoch})
        row_dict.update((key, handle_value(logs[key])) for key in self.keys)
        self.writer.writerow(row_dict)
        self.csv_file.flush()


class TestCallback(Callback): 
    def __init__(self, test_data): 
        self.test_data = test_data
        self.loss = -1e8
        self.acc = -1e8

    def on_train_begin(self, logs=None):
        self.times = []
        self.training_time_start = time.time()
        logging.info("Training started")

    def on_train_end(self, logs=None):
        x, y = self.test_data
        self.loss, self.acc = self.model.evaluate(x, y, batch_size=x[0].shape[0], verbose=0)
        elapse_time = time.time() - self.training_time_start
        logging.info('Training ended | Loss: {}, acc: {} (Training took {} seconds | Avg. epoch duration: {} seconds)'.format(
            self.loss, self.acc, elapse_time, np.mean(self.times)))


    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)
        if False:
            x, y = self.test_data
            self.loss, self.acc = self.model.evaluate(x, y, batch_size=x[0].shape[0], verbose=0)
            elapse_time = time.time() - self.epoch_time_start
            logging.info('Testing loss for epoch {}: {}, acc: {} (Took: {} seconds)'.format(epoch, self.loss, self.acc, elapse_time))
        
        
def split_data(x, y, validation_split=0.1):
    if validation_split and 0. < validation_split < 1.:
        split_at = int(len(x[0][:]) * (1. - validation_split))
        x, x_val = ([np.array(sl[0:split_at]) for sl in x], [np.array(sl[split_at:]) for sl in x])
        y, y_val = (np.array(y[0:split_at]), np.array(y[split_at:]))
    else:
        raise ValueError("validation_split must be [0,1)")
    return x, x_val, y, y_val


def get_section_hparams(name, ub, lb, type):
    ub = eval(str(ub))
    lb = eval(str(lb))
    switcher = {
        "log": hp.loguniform(name, lb, ub),
        "uniform": hp.uniform(name, lb, ub),
        "choice": hp.choice(name, [ub])
    }
    # Get the function from switcher dictionary
    func = switcher.get(type, None)
    # Execute the function
    return func


def create_hparams(param_config_file):
    """Create the hparams object for generic training hyperparameters."""
    hparams = {}

    if param_config_file:
        with open(param_config_file) as fp:
            cfg = YAML().load(fp)
            section = 'default'
            if section in cfg:
                for k, v in cfg[section].items():
                    hparams[k] = hp.choice(k, v)
            section = 'sampling'
            schema = ['ub', 'lb', 'type']
            if section in cfg:
                for k in cfg[section].keys():
                    if isinstance(cfg[section][k], Iterable) and all(s in cfg[section][k] for s in schema):
                        v = get_section_hparams(name=k,
                                                ub=cfg[section][k]['ub'],
                                                lb=cfg[section][k]['lb'],
                                                type=cfg[section][k]['type'])
                        hparams[k] = v

    return hparams