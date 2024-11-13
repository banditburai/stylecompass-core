import numpy as np
from collections import deque, defaultdict
import torch
from torch import distributed as dist
from src.utils.distributed import is_dist_avail_and_initialized
import time
import datetime

class Metrics:
    """Evaluation metrics for retrieval tasks"""
    def __init__(self):
        self.data = None

    @staticmethod
    def get_recall(preds, gts, topk=5):
        """Get recall@k for continuous predictions"""
        preds = preds[:, :topk]
        preds -= gts[:, None]
        found = np.where(np.amin(np.absolute(preds), axis=1) == 0)[0]
        return found.shape[0] / gts.shape[0]

    @staticmethod
    def get_mrr(preds, gts, topk=5):
        """Get Mean Reciprocal Rank for continuous predictions"""
        preds = preds[:, :topk]
        preds -= gts[:, None]
        rows, cols = np.where(preds == 0)
        _, unique_rows = np.unique(rows, return_index=True)
        valid_cols = cols[unique_rows]
        valid_cols += 1
        return np.mean(1/valid_cols)

    @staticmethod
    def get_map(preds, gts, topk=5):
        """Get Mean Average Precision for continuous predictions"""
        preds = preds[:, :topk]
        preds -= gts[:, None]
        rows, cols = np.where(preds == 0)
        _, unique_rows = np.unique(rows, return_index=True)
        row_cols = np.split(cols, unique_rows)[1:]
        row_cols = [np.hstack([x[0], np.diff(x), topk - x[-1]]) for x in row_cols]
        row_cols = [np.pad(x, (0, topk + 1 - x.shape[0]), 'constant', constant_values=(0, 0)) for x in row_cols]
        precision = np.asarray([np.repeat(np.arange(topk + 1), x) / np.arange(1, topk + 1) for x in row_cols])
        return np.sum(np.mean(precision, axis=1)) / preds.shape[0]

    @staticmethod
    def get_recall_bin(preds, topk=5):
        """Get recall@k for binary predictions"""
        preds = preds[:, :topk]
        found = np.where(np.amax(preds, axis=1) == True)[0]
        return found.shape[0] / preds.shape[0]

    @staticmethod
    def get_mrr_bin(preds, topk=5):
        """Get Mean Reciprocal Rank for binary predictions"""
        preds = preds[:, :topk]
        rows, cols = np.where(preds)
        _, unique_rows = np.unique(rows, return_index=True)
        valid_cols = cols[unique_rows]
        valid_cols += 1
        return np.mean(1/valid_cols)

    @staticmethod
    def get_map_bin(preds, topk=5):
        """Get Mean Average Precision for binary predictions"""
        preds = preds[:, :topk]
        rows, cols = np.where(preds)
        _, unique_rows = np.unique(rows, return_index=True)
        row_cols = np.split(cols, unique_rows)[1:]
        row_cols = [np.hstack([x[0], np.diff(x), topk - x[-1]]) for x in row_cols]
        row_cols = [np.pad(x, (0, topk + 1 - x.shape[0]), 'constant', constant_values=(0, 0)) for x in row_cols]
        precision = np.asarray([np.repeat(np.arange(topk + 1), x) / np.arange(1, topk + 1) for x in row_cols])
        return np.sum(np.mean(precision, axis=1)) / preds.shape[0]


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.6f} ({global_avg:.6f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.6f}')
        data_time = SmoothedValue(fmt='{avg:.6f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.6f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))