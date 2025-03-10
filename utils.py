from __future__ import print_function
from collections import defaultdict, deque, OrderedDict
import datetime
import math
import itertools
import time
import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import errno
import os
import sys
from torch.utils.data.dataloader import default_collate
from torch.optim.lr_scheduler import _LRScheduler
from torch._six import inf
import numpy as np
import random

from torch.utils.hipify.hipify_python import bcolors


def fix_randseed(seed):
    r""" Set random seeds for reproducibility """
    if seed is None:
        seed = int(random.random() * 1e5)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def all_reduce_tensor(tensor, op=dist.ReduceOp.SUM, world_size=1, norm=True):
    with torch.no_grad():
        tensor = tensor.detach()
        dist.all_reduce(tensor, op)
        if norm:
            tensor.div_(world_size)
    return tensor


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
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
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = self.delimiter.join([
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}',
            'max mem: {memory:.0f}'
        ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                print(log_msg.format(
                    i, len(iterable), eta=eta_string,
                    meters=str(self),
                    time=str(iter_time), data=str(data_time),
                    memory=torch.cuda.max_memory_allocated() / MB))
                sys.stdout.flush()

            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {}'.format(header, total_time_str))


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environment: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1

    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()
    setup_for_distributed(is_main_process())

    if args.output_dir:
        mkdir(args.output_dir)
    # if args.model_id:
    #     mkdir(os.path.join('./models/', args.model_id))


def collate_func(batch):
    '''Support different numbers of instance masks within a batch
    '''

    elem = batch[0]
    elem_type = type(elem)
    # print(elem_type)
    if isinstance(elem, dict):
        return {key: collate_func([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, list):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if isinstance(elem[0], torch.Tensor):
            return [torch.stack(sample, dim=0) for sample in batch]
        elif not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        else:
            transposed = zip(*batch)
            return [collate_func(samples) for samples in transposed]
    elif isinstance(elem, tuple):
        # it = iter(batch)
        # elem_size = len(next(it))
        # if not all(len(elem) == elem_size for elem in it):
        #     raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [collate_func(samples) for samples in transposed]
    else:
        return default_collate(batch)


# def collate_func(batch, path="root"):
#     """
#     Debugging version of collate_func to locate tensor shape mismatches in a batch of data.
#     """
#     elem = batch[0]
#     elem_type = type(elem)
#
#     try:
#         if isinstance(elem, dict):
#             # Debug output for dictionary keys
#             print(f"Processing dict at path: {path}, keys: {list(elem.keys())}")
#             return {
#                 key: collate_func([d[key] for d in batch], path=f"{path}.{key}")
#                 for key in elem
#             }
#         elif isinstance(elem, list):
#             print(f"Processing list at path: {path}")
#             if isinstance(elem[0], torch.Tensor):
#                 tensor_shapes = [e.shape for e in elem]
#                 print(f"List tensor shapes at path: {path} -> {tensor_shapes}")
#                 if not all(shape == tensor_shapes[0] for shape in tensor_shapes):
#                     print(f"Shape mismatch in list at {path}: {tensor_shapes}")
#                     raise RuntimeError(f"Inconsistent tensor shapes at {path}: {tensor_shapes}")
#                 return [torch.stack(sample, dim=0) for sample in batch]
#             else:
#                 return [collate_func(samples, path=path) for samples in zip(*batch)]
#         elif isinstance(elem, tuple):
#             print(f"Processing tuple at path: {path}")
#             return tuple(collate_func(samples, path=path) for samples in zip(*batch))
#         elif isinstance(elem, torch.Tensor):
#             print(f"Processing tensor at path: {path}, shape: {elem.shape}")
#             return default_collate(batch)
#         else:
#             print(f"Processing other type at path: {path}, type: {elem_type}")
#             return default_collate(batch)
#     except Exception as e:
#         # Log details for debugging before re-raising exception
#         print(f"Error occurred at path: {path}")
#         print(f"Batch content at error (limited to shapes or types):")
#         if isinstance(batch, list) and all(isinstance(item, torch.Tensor) for item in batch):
#             print([item.shape for item in batch])
#         else:
#             print(batch)  # Fallback to raw batch content
#         raise e



def load_model(model, model_file, is_restore=False):
    t_start = time.time()
    if isinstance(model_file, str):
        device = torch.device('cpu')
        checkpoint = torch.load(model_file, map_location=device)
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        if 'model' in state_dict.keys():
            state_dict = state_dict['model']
    else:
        state_dict = model_file
    t_ioend = time.time()

    if is_restore:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        state_dict = new_state_dict

    model.load_state_dict(state_dict, strict=False)
    ckpt_keys = set(state_dict.keys())
    own_keys = set(model.state_dict().keys())
    missing_keys = own_keys - ckpt_keys
    unexpected_keys = ckpt_keys - own_keys

    if len(missing_keys) > 0:
        print('Missing key(s) in state_dict: {}'.format(
            ', '.join('{}'.format(k) for k in missing_keys)))

    if len(unexpected_keys) > 0:
        print('Unexpected key(s) in state_dict: {}'.format(
            ', '.join('{}'.format(k) for k in unexpected_keys)))

    del state_dict
    t_end = time.time()
    print(
        "Load model from {}, Time usage:\n\tIO: {}, initialize parameters: {}".format(model_file,
                                                                                      t_ioend - t_start,
                                                                                      t_end - t_ioend))

    return model


class WarmUpPolyLRScheduler(_LRScheduler):
    """Decays the learning rate of each parameter group using a polynomial function
    in the given total_iters. When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        total_iters (int): The number of steps that the scheduler decays the learning rate. Default: 5.
        power (int): The power of the polynomial. Default: 1.0.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

    """

    def __init__(self, optimizer, total_iters=5, power=1.0, last_epoch=-1, verbose=False,
                 min_lr=0, warmup=False, warmup_iters=0, warmup_ratio=0.1):
        self.total_iters = total_iters
        self.power = power
        self.min_lr = min_lr
        self.warmup = warmup
        self.warmup_iters = warmup_iters
        self.warmup_ratio = warmup_ratio
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):

        if self.last_epoch == 0 or self.last_epoch > self.total_iters:
            return [group["lr"] for group in self.optimizer.param_groups]

        return self._get_closed_form_lr()
        # decay_factor = ((1.0 - self.last_epoch / self.total_iters) / (1.0 - (self.last_epoch - 1) / self.total_iters)) ** self.power
        # return [group["lr"] * decay_factor for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        if not self.warmup or self.last_epoch > self.warmup_iters:
            coeff = (1 - self.last_epoch / self.total_iters) ** self.power
            return [
                (
                        (base_lr - self.min_lr) * coeff + self.min_lr
                )
                for base_lr in self.base_lrs
            ]
        else:
            coeff = (1 - self.last_epoch / self.warmup_iters) * (1 - self.warmup_ratio)
            return [
                (
                        base_lr * (1 - coeff)
                )
                for base_lr in self.base_lrs
            ]


def ampscaler_get_grad_norm(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(),
                                                        norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True,
                 model=None):
        # with torch.autograd.detect_anomaly(True):  # 反向传播时：在求导时开启侦测
        self._scaler.scale(loss).backward(create_graph=create_graph)
        # # print grad check
        # v_n = []
        # v_v = []
        # v_g = []
        # for name, parameter in model.named_parameters():
        #     v_n.append(name)
        #     v_v.append(parameter.detach().cpu().numpy() if parameter is not None else [0])
        #     v_g.append(parameter.grad.detach().cpu().numpy() if parameter.grad is not None else [0])
        # for i in range(len(v_n)):
        #     if np.max(v_v[i]).item() - np.min(v_v[i]).item() < 1e-6:
        #         color = bcolors.FAIL + '*'
        #     else:
        #         color = bcolors.OKGREEN + ' '
        # for name, param in model.named_parameters():
        #     if param.grad is not None and torch.isnan(param.grad).any():
        #         print(f"[ERROR] NaN detected in gradients of parameter: {name}")

        # for name, param in model.named_parameters():
        #     if param.grad is not None:
        #         if torch.isnan(param.grad).any():
        #             print(f"NaN detected in gradients of {name}")
        #         if torch.isinf(param.grad).any():
        #             print(f"Inf detected in gradients of {name}")

        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = ampscaler_get_grad_norm(parameters)

            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


class NativeScalerWithGradNormCount2:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None):
        self._scaler.scale(loss).backward()
        if clip_grad is not None:
            self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
            all_params = itertools.chain(*[x["params"] for x in optimizer.param_groups])
            norm = torch.nn.utils.clip_grad_norm_(all_params, clip_grad)
        else:
            self._scaler.unscale_(optimizer)
            all_params = itertools.chain(*[x["params"] for x in optimizer.param_groups])
            norm = ampscaler_get_grad_norm(all_params)
        self._scaler.step(optimizer)
        self._scaler.update()
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)