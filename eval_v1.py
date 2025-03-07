import os

import torch
import torch.utils.data
from torch import nn


import utils

import numpy as np
from PIL import Image
import torch.nn.functional as F
from detectron2.projects.deeplab import add_deeplab_config,build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.config import get_cfg
import utils
from datasets.dataloader_gres import loader,RefCOCODataSet
from datasets.dataloader_gvpcoco import GvpCOCODataSet
from datasets.dataset_refer_bert import ReferDataset,ReferDatasetTest
from importlib import import_module
from gres_model import (
    add_maskformer2_config,
    add_refcoco_config,
)
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

def batch_IoU(pred, gt):
    intersection = torch.logical_and(pred, gt).sum(1)
    union = torch.logical_or(pred, gt).sum(1)

    iou = intersection.float() / union.float()

    return iou, intersection, union


def evaluate(model, data_loader,dataset_name,split):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")

    total_num = len(data_loader.dataset)
    acc_ious = torch.zeros(1).cuda()

    # evaluation variables
    cum_I = torch.zeros(1).cuda()
    cum_U = torch.zeros(1).cuda()
    eval_seg_iou_list = [.5, .7,.8, .9]
    seg_correct = torch.zeros(len(eval_seg_iou_list)).cuda()
    seg_total = torch.zeros(1).cuda()
    header = 'Test:' if split is None else split+':'
    _available_sources = ["refcoco", "refcoco+", "refcocog", "grefcoco", "gvpcoco"]
    _cpu_device = torch.device("cpu")
    with torch.no_grad():
        for data in metric_logger.log_every(data_loader, 100, header):
            for key in data.keys():
                data[key] = data[key].cuda(non_blocking=True)

            output = model(data)
            batch_size = data['image'].shape[0]
            # for j in range(batch_size):
            src = dataset_name
            assert src in _available_sources
            # 提取 GT 和预测 mask
            gt_mask = data['gt_mask_merged'].to(_cpu_device)
            gt = gt_mask.to(torch.int8)
            output_mask=[]
            for j in range(batch_size):
                output_mask.append(output[j]["ref_seg"].argmax(dim=0).to(_cpu_device))
                # output_mask.append((output[j]["ref_seg"]>0.5).to(_cpu_device))
            # output_mask = output[j]["ref_seg"].argmax(dim=0).to(_cpu_device)
                # output_mask = (output[j]["ref_seg"]>0.5).to(_cpu_device)
            output_mask=torch.stack(output_mask)
            pred_mask = output_mask.to(torch.int8)  # 保持为 Tensor 类型

            # I, U = computeIoU(pred_mask, gt)
            iou, I, U = batch_IoU(pred_mask.flatten(1), gt.flatten(1))
            acc_ious += iou.sum()
            cum_I += I.sum()
            cum_U += U.sum()
            for n_eval_iou in range(len(eval_seg_iou_list)):
                eval_seg_iou = eval_seg_iou_list[n_eval_iou]
                seg_correct[n_eval_iou] += (iou >= eval_seg_iou).sum()
            seg_total += batch_size


    torch.cuda.synchronize()
    cum_I = cum_I.cpu().numpy()
    cum_U = cum_U.cpu().numpy()
    acc_ious = acc_ious.cpu().numpy()
    seg_correct = seg_correct.cpu().numpy()
    seg_total = seg_total.cpu().numpy()
    # print( seg_total, len(data_loader.dataset))
    mIoU = acc_ious / seg_total
    print('Final results:')
    print('Mean IoU is %.2f\n' % (mIoU * 100.))
    results_str = ''
    for n_eval_iou in range(len(eval_seg_iou_list)):
        results_str += '    precision@%s = %.2f\n' % \
                       (str(eval_seg_iou_list[n_eval_iou]), seg_correct[n_eval_iou] * 100. / seg_total)
    results_str += '    overall IoU = %.2f\n' % (cum_I * 100. / cum_U)
    print(results_str)



def computeIoU(pred_seg, gd_seg):
    I = np.sum(np.logical_and(pred_seg, gd_seg))
    U = np.sum(np.logical_or(pred_seg, gd_seg))

    return I, U
def setup(args):

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
def main(args):
    device = torch.device(args.device)
    # dataset_test, _ = get_dataset(args.split, get_transform(args=args), args)
    cfg = setup(args)
    # from datasets.utils import config
    # __C = config.load_cfg_from_cfg_file(args.config)
    # val_set = RefCOCODataSet(__C, split='val')
    if cfg.DATASETS.DATASET_NAME=='grefcoco':
        train_set = RefCOCODataSet(cfg, split='train',splitby='unc')
        val_set = RefCOCODataSet(cfg, split='val',splitby='unc')
    elif cfg.DATASETS.DATASET_NAME=='gvpcoco':
        train_set = GvpCOCODataSet(cfg, split='train', splitby='unc')
        val_set = GvpCOCODataSet(cfg, split='val', splitby='unc')
    elif cfg.DATASETS.DATASET_NAME=='refcoco':
        train_set = ReferDataset(cfg, split='train', splitby='unc')
        val_set = ReferDatasetTest(cfg, split=args.split, splitby='unc',eval_mode=True)
    elif cfg.DATASETS.DATASET_NAME=='refcoco+':
        train_set = ReferDataset(cfg, split='train', splitby='unc')
        val_set = ReferDatasetTest(cfg, split=args.split, splitby='unc',eval_mode=True)
    elif cfg.DATASETS.DATASET_NAME=='refcocog':
        train_set = ReferDataset(cfg, split='train', splitby='umd')
        val_set = ReferDatasetTest(cfg, split=args.split, splitby='umd',eval_mode=True)
    # print(len(val_set))
    test_sampler = torch.utils.data.SequentialSampler(val_set)
    data_loader_test = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size,
                                                   sampler=test_sampler, num_workers=args.workers)
    # print(args.model)
    # single_model = builder.__dict__[args.model](pretrained='',args=args)
    single_model = ModelLoader(args).Net(cfg,args)

    utils.load_model(single_model, args.resume)
    model = single_model.to(device)

    evaluate(model, data_loader_test,cfg.DATASETS.DATASET_NAME,split=args.split)

if __name__ == "__main__":
    from args import get_parser
    parser = get_parser()
    args = parser.parse_args()
    print('Image size: {}'.format(str(args.img_size)))
    if args.eval_ori_size:
        print('Eval mode: original')
    else:
        print('Eval mode: resized')
    main(args)
