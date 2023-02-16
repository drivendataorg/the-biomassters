import time
import torch
import datetime

from collections import defaultdict, deque


class SmoothedValue(object):

    def __init__(self, window_size=100, fmt=None):
        if fmt is None:
            fmt = '{global_avg:.6f}'

        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

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
            value=self.value
        )


class MetricLogger(object):

    def __init__(self, header, print_freq=100, delimiter=' '):
        self.header = header
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.print_freq = print_freq

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
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
            loss_str.append('{}:{}'.format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable):
        start = time.time()
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [self.header, '[{0' + space_fmt + '}/{1}]', '{meters}']

        log_msg = self.delimiter.join(log_msg)
        for i, obj in enumerate(iterable):
            yield obj
            if (i % self.print_freq == 0 or i == len(iterable) - 1):
                print(log_msg.format(i, len(iterable), meters=str(self)))

        total_time = time.time() - start
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s/it)'.format(
            self.header, total_time_str, total_time / len(iterable)))
