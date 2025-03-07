import copy
import os
import time

import numpy as np
import torch
import logging
import datetime
from functools import reduce
from collections import OrderedDict
import itertools
import operator
from typing import Any, Dict, List, Set
# from detectron2.engine import DefaultTrainer, launch,default_argument_parser
# from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
# from detectron2.data import MetadataCatalog, build_detection_train_loader, build_detection_test_loader
# from detectron2.utils.logger import setup_logger
from detectron2.projects.deeplab import add_deeplab_config,build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
# import detectron2.utils.comm as comm

# Custom imports for your specific model
from gres_model import (
    RefCOCOMapper,
    ReferEvaluator,
    add_maskformer2_config,
    add_refcoco_config,
    load_config
)
from importlib import import_module
import utils
import torch.backends.cudnn as cudnn
from datasets.dataloader_gres import loader,RefCOCODataSet
from datasets.dataloader_gvpcoco import GvpCOCODataSet
# from test import batch_evaluate, batch_evaluate_v1
from datasets.dataset_refer_bert import ReferDataset,ReferDatasetTest
# from test_v1 import batch_evaluate_v1
import torch.nn.functional as F
class ModelLoader:
    def __init__(self, args):
        self.model_use = args.model_name
        model_moudle_path = 'gres_model.models.' + self.model_use + '.builder'
        self.model_moudle = import_module(model_moudle_path)

    def Net(self, cfg,args):
        return self.model_moudle.__dict__[args.model](cfg)


def is_distributed():
    distributed = False
    if 'WORLD_SIZE' in os.environ:
        distributed = int(os.environ['WORLD_SIZE']) > 1
    return distributed




def build_optimizer(cfg, model):
    weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
    weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED
    defaults = {}
    defaults["lr"] = cfg.SOLVER.BASE_LR
    defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

    norm_module_types = (
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
        torch.nn.SyncBatchNorm,
        torch.nn.GroupNorm,
        torch.nn.InstanceNorm1d,
        torch.nn.InstanceNorm2d,
        torch.nn.InstanceNorm3d,
        torch.nn.LayerNorm,
        torch.nn.LocalResponseNorm,
    )

    params: List[Dict[str, Any]] = []
    memo: Set[torch.nn.parameter.Parameter] = set()
    for module_name, module in model.named_modules():
        for module_param_name, value in module.named_parameters(recurse=False):
            if "text_encoder" in module_name:
                continue
            if not value.requires_grad:
                continue
            if value in memo:
                continue
            memo.add(value)

            hyperparams = copy.copy(defaults)

            if (
                    "relative_position_bias_table" in module_param_name
                    or "absolute_pos_embed" in module_param_name
            ):
                hyperparams["weight_decay"] = 0.0

            if isinstance(module, norm_module_types):
                hyperparams["weight_decay"] = weight_decay_norm

            if isinstance(module, torch.nn.Embedding):
                hyperparams["weight_decay"] = weight_decay_embed
            params.append({"params": [value], **hyperparams})

    hyperparams = copy.copy(defaults)
    params.append({"params": reduce(operator.concat,
                                    [[p for p in model.text_encoder.encoder.layer[i].parameters()
                                      if p.requires_grad] for i in range(10)]),
                   **hyperparams
                   })
    def maybe_add_full_model_gradient_clipping(optim):
        # detectron2 doesn't have full model gradient clipping now
        clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
        enable = (
            cfg.SOLVER.CLIP_GRADIENTS.ENABLED
            and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
            and clip_norm_val > 0.0
        )

        class FullModelGradientClippingOptimizer(optim):
            def step(self, closure=None):
                all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                super().step(closure=closure)

        return FullModelGradientClippingOptimizer if enable else optim

    optimizer_type = cfg.SOLVER.OPTIMIZER
    if optimizer_type == "SGD":
        optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
            params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
        )
    elif optimizer_type == "ADAMW":
        optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
            params, cfg.SOLVER.BASE_LR
        )

    else:
        raise NotImplementedError(f"no optimizer type {optimizer_type}")
    if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
        optimizer = maybe_add_gradient_clipping(cfg, optimizer)
    return optimizer


def setup(args):
    """
    Create configs and perform basic setups.
    """
    # cfg = get_cfg()
    # add_deeplab_config(cfg)
    # add_maskformer2_config(cfg)
    # add_refcoco_config(cfg)
    # cfg=load_config(args.config_file)
    # cfg.MODEL.SWIN.QK_SCALE = None
    # add_maskformer2_config(cfg)
    # add_refcoco_config(cfg)
    # cfg.merge_from_list(args.opts)
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_refcoco_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.MODEL.SWIN.SWIN_PRETRAINED_WEIGHTS='swin_base_patch4_window12_384_22k.pth'
    cfg.OUTPUT_DIR=args.output_dir
    cfg.DATASETS.DATASET_NAME=args.dataset
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "full_model"
    cfg.SOLVER.CLIP_GRADIENTS.ENABLED= False
    cfg.CONDITION = args.condition

    cfg.freeze()
    # print(cfg)
    # exit(0)
    # Setup logger and log config
    return cfg
def computeIoU(pred_seg, gd_seg):
    I = np.sum(np.logical_and(pred_seg, gd_seg))
    U = np.sum(np.logical_or(pred_seg, gd_seg))

    return I, U
def evaluate(model, data_loader, dataset_name,device = 'cuda'):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")

    acc_ious = torch.zeros(1).cuda()

    # evaluation variables
    # cum_I, cum_U = 0, 0
    cum_I = torch.zeros(1).cuda()
    cum_U = torch.zeros(1).cuda()

    eval_seg_iou_list = [.5, .7, .8,.9]
    seg_correct = torch.zeros(len(eval_seg_iou_list)).cuda()
    total_num = len(data_loader.dataset)
    # evaluation variables
    # cum_I, cum_U = 0, 0
    # eval_seg_iou_list = [.5, .6, .7, .8, .9]
    # seg_correct = np.zeros(len(eval_seg_iou_list), dtype=np.int32)
    seg_total = torch.zeros(1).cuda()
    mean_IoU = []
    header = 'Test:'
    _available_sources = ["refcoco","refcoco+","refcocog", "grefcoco", "gvpcoco"]
    _cpu_device = torch.device("cpu")
    with torch.no_grad():
        for data in metric_logger.log_every(data_loader, 100, header):
            for key in data.keys():
                data[key] = data[key].cuda(non_blocking=True)
                # with torch.cuda.amp.autocast():
            output = model(data)
            batch_size = data['image'].shape[0]
            for j in range(batch_size):
                src = dataset_name
                assert src in _available_sources
                # 提取 GT 和预测 mask
                gt_mask = data['gt_mask_merged'][j].to(_cpu_device)
                gt = np.array(gt_mask, dtype=np.int8)
                output_mask = output[j]["ref_seg"].argmax(dim=0).to(_cpu_device)
                # output_mask = (output[j]["ref_seg"]>0.5).to(_cpu_device)
                pred_mask = np.array(output_mask, dtype=np.int8)



                I, U = computeIoU(pred_mask, gt)
                if U == 0:
                    this_iou = 0.0
                else:
                    this_iou = I*1.0/U
                # mean_IoU.append(this_iou)
                acc_ious+=this_iou
                cum_I += I
                cum_U += U
                for n_eval_iou in range(len(eval_seg_iou_list)):
                    eval_seg_iou = eval_seg_iou_list[n_eval_iou]
                    seg_correct[n_eval_iou] += (this_iou >= eval_seg_iou)
                seg_total += 1

    torch.cuda.synchronize()
    if is_distributed():
        cum_I = utils.all_reduce_tensor(cum_I, norm=False).cpu().numpy()
        cum_U = utils.all_reduce_tensor(cum_U, norm=False).cpu().numpy()
        acc_ious = utils.all_reduce_tensor(acc_ious, norm=False).cpu().numpy()
        seg_correct = utils.all_reduce_tensor(seg_correct, norm=False).cpu().numpy()
        seg_total = utils.all_reduce_tensor(seg_total, norm=False).cpu().numpy()
    else:
        cum_I = cum_I.cpu().numpy()
        cum_U = cum_U.cpu().numpy()
        acc_ious = acc_ious.cpu().numpy()
        seg_correct = seg_correct.cpu().numpy()
        seg_total = seg_total.cpu().numpy()

    # print(len(mean_IoU),seg_total,len(data_loader.dataset))
    mIoU = acc_ious / seg_total
    # mean_IoU = np.array(mean_IoU)
    # mIoU = np.mean(mean_IoU)

    print('Final results:')
    print('Mean IoU is %.2f\n' % (mIoU * 100.))
    # print('Mean2 IoU is %.2f\n' % (mIoU2 * 100.))
    results_str = ''
    for n_eval_iou in range(len(eval_seg_iou_list)):
        results_str += '    precision@%s = %.2f\n' % \
                       (str(eval_seg_iou_list[n_eval_iou]), seg_correct[n_eval_iou] * 100. / seg_total)
    results_str += '    overall IoU = %.2f\n' % (cum_I * 100. / cum_U)
    print(results_str)

    return 100 * mIoU, 100 * cum_I / cum_U

def train_one_epoch(model, optimizer, data_loader, lr_scheduler, epoch, print_freq, loss_scaler, clip_grad):
    model.train()
    optimizer.zero_grad()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.4e}'))
    header = 'Epoch: [{}]'.format(epoch)


    for data in metric_logger.log_every(data_loader, print_freq, header):


        for key in data.keys():
            data[key]=data[key].cuda(non_blocking=True)


        # with torch.cuda.amp.autocast():
        loss_dict = model(data)

        total_loss = loss_dict['loss_mask']

        # grad_norm = loss_scaler(total_loss, optimizer, clip_grad=clip_grad, parameters=model.parameters(),model=model)
        grad_norm = loss_scaler(total_loss, optimizer, clip_grad=clip_grad)
        optimizer.zero_grad()  # set_to_none=True is only available in pytorch 1.6+
        lr_scheduler.step()

        torch.cuda.synchronize()
        metric_logger.update(lr=optimizer.param_groups[-1]["lr"])
        metric_logger.update(grad_norm=grad_norm)
        metric_logger.update(**loss_dict)

        # for name, param in model.module.named_parameters():
        #     if param.grad is None:
        #         print(f"Parameter {name} in {type(param).__name__} has no gradient.")
        # exit(0)

def main(args,distributed):
    cfg = setup(args)
    utils.fix_randseed(114514)
    if cfg.DATASETS.DATASET_NAME=='grefcoco':
        train_set = RefCOCODataSet(cfg, split='train',splitby='unc')
        val_set = RefCOCODataSet(cfg, split='val',splitby='unc')
    elif cfg.DATASETS.DATASET_NAME=='gvpcoco':
        train_set = GvpCOCODataSet(cfg, split='train', splitby='unc')
        val_set = GvpCOCODataSet(cfg, split='val', splitby='unc')
    elif cfg.DATASETS.DATASET_NAME=='refcoco':
        train_set = ReferDatasetTest(cfg, split='train', splitby='unc')
        val_set = ReferDatasetTest(cfg, split='val', splitby='unc',eval_mode=True)
    elif cfg.DATASETS.DATASET_NAME=='refcoco+':
        train_set = ReferDatasetTest(cfg, split='train', splitby='unc')
        val_set = ReferDatasetTest(cfg, split='val', splitby='unc',eval_mode=True)
    elif cfg.DATASETS.DATASET_NAME=='refcocog':
        train_set = ReferDatasetTest(cfg, split='train', splitby='umd')
        val_set = ReferDatasetTest(cfg, split='val', splitby='umd',eval_mode=True)

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()

    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, num_replicas=num_tasks, rank=global_rank,
                                                                        shuffle=True, drop_last=True)
        # test_sampler = torch.utils.data.SequentialSampler(dataset_test)
        test_sampler = torch.utils.data.distributed.DistributedSampler(val_set, shuffle=False, drop_last=False)
        shuffle = False
    else:
        train_sampler = None
        test_sampler = None
        shuffle = True
    # data loader
    data_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=shuffle,
        sampler=train_sampler, num_workers=args.workers, pin_memory=args.pin_mem,
        drop_last=True, collate_fn=utils.collate_func)

    data_loader_test = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, sampler=test_sampler, num_workers=args.workers)

    criterion = None
    single_model = ModelLoader(args).Net(cfg,args)
    # single_model.init_weights(pretrained=cfg.MODEL.SWIN.SWIN_PRETRAINED_WEIGHTS)
    single_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(single_model)

    # optimizer = torch.optim.AdamW(params_to_optimize,
    #                               lr=args.lr,
    #                               weight_decay=args.weight_decay,
    #                               amsgrad=args.amsgrad
    #                               )
    optimizer = build_optimizer(cfg, single_model)
    single_model.cuda()

    # if args.resume:
    #     checkpoint = torch.load(args.resume, map_location='cpu')
    #     single_model.load_state_dict(checkpoint['model'])

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(single_model, device_ids=[args.local_rank],
                                                          find_unused_parameters=True)
    else:
        model = torch.nn.DataParallel(single_model)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        single_model.load_state_dict(checkpoint['model'])


    # lr_scheduler =build_lr_scheduler(cfg, optimizer)
    # total_iters = (len(data_loader) * args.epochs)
    # lr_scheduler = utils.WarmUpPolyLRScheduler(optimizer, total_iters, power=0.9, min_lr=args.min_lr,
    #                                            warmup=args.warmup, warmup_iters=args.warmup_iters,
    #                                            warmup_ratio=args.warmup_ratio)

    lr_scheduler=build_lr_scheduler(cfg, optimizer)
    # loss_scaler = utils.NativeScalerWithGradNormCount()
    loss_scaler=utils.NativeScalerWithGradNormCount2()
    clip_grad = args.clip_value if args.clip_grads else None

    start_time = time.time()
    best_oIoU = -0.1
    best_mIoU = -0.1
    if args.resume:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        resume_epoch = checkpoint['epoch']
    else:
        resume_epoch = -999

    # if not args.training:
    #     single_model=utils.load_model(single_model, args.resume)
    #     model = single_model.cuda()
    #     mIoU, oIoU = evaluate(model, data_loader_test, cfg.DATASETS.DATASET_NAME)
    #     exit(0)

    for epoch in range(max(0, resume_epoch + 1), args.epochs):
        if distributed:
            data_loader.sampler.set_epoch(epoch)
        train_one_epoch(model, optimizer, data_loader, lr_scheduler, epoch, args.print_freq, loss_scaler, clip_grad)
        if epoch + 1:
            mIoU,oIoU = evaluate(model, data_loader_test,cfg.DATASETS.DATASET_NAME)
            src=cfg.DATASETS.DATASET_NAME

            save_checkpoint = (best_oIoU < oIoU)
            save_checkpoint_mIoU = (best_mIoU < mIoU)
            # if epoch % 10 == 0 or epoch >= args.epochs - 16:
            if save_checkpoint:
                print('Better epoch: {}\n'.format(epoch))
                dict_to_save = {'model': single_model.state_dict(),
                                'optimizer': optimizer.state_dict(), 'epoch': epoch, 'args': args,
                                'lr_scheduler': lr_scheduler.state_dict(), 'scaler': loss_scaler.state_dict()}

                utils.save_on_master(dict_to_save, os.path.join(args.output_dir,
                                                                'model_best_{}.pth'.format(src)))
                best_oIoU = oIoU

            if save_checkpoint_mIoU:
                print('Better epoch miou: {}\n'.format(epoch))
                dict_to_save = {'model': single_model.state_dict(),
                                'optimizer': optimizer.state_dict(), 'epoch': epoch, 'args': args,
                                'lr_scheduler': lr_scheduler.state_dict(), 'scaler': loss_scaler.state_dict()}

                utils.save_on_master(dict_to_save, os.path.join(args.output_dir,
                                                                'model_best_miou_{}.pth'.format(src)))
                best_mIoU = mIoU

        # summarize
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))





if __name__ == "__main__":
    from args import get_parser

    parser = get_parser()
    args = parser.parse_args()
    # args = default_argument_parser().parse_args()
    # set up distributed learning
    cfg = setup(args)
    distributed = is_distributed()
    if distributed:
        utils.init_distributed_mode(args)
    cudnn.benchmark = True
    # print('Image size: {}'.format(str(args.img_size)))
    main(args, distributed)
