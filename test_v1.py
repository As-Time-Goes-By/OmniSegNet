import itertools
import json
import logging
import numpy as np
import os
from collections import OrderedDict
import torch
import torch.distributed as dist
from detectron2.utils.comm import all_gather, is_main_process, synchronize
from detectron2.utils.file_io import PathManager
from detectron2.evaluation.evaluator import DatasetEvaluator
from gres_model.evaluation.refer_evaluation import ReferEvaluator
from gres_model.evaluation.gvp_refer_evaluation import GVPReferEvaluator

import utils
def is_distributed():
    distributed = False
    if 'WORLD_SIZE' in os.environ:
        distributed = int(os.environ['WORLD_SIZE']) > 1
    return distributed
def computeIoU(pred_seg, gd_seg):
    I = np.sum(np.logical_and(pred_seg, gd_seg))
    U = np.sum(np.logical_or(pred_seg, gd_seg))

    return I, U



def batch_evaluate(cfg,model, data_loader):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    total_num = len(data_loader.dataset)
    acc_ious = torch.zeros(1).cuda()

    # evaluation variables
    # cum_I, cum_U = 0, 0
    cum_I = torch.zeros(1).cuda()
    cum_U = torch.zeros(1).cuda()
    eval_seg_iou_list = [.5, .7, .9]
    seg_correct = torch.zeros(len(eval_seg_iou_list)).cuda()

    # if output_folder is None:
    output_folder = os.path.join(cfg.OUTPUT_DIR, cfg.DATASETS.DATASET_NAME,"inference")
    os.makedirs(output_folder, exist_ok=True)

    evaluator=GVPReferEvaluator(
        cfg.DATASETS.DATASET_NAME,
        distributed=True,
        output_dir=output_folder,
    )

    with torch.no_grad():
        for data in metric_logger.log_every(data_loader, 100, header):

            for key in data.keys():
                data[key] = data[key].cuda(non_blocking=True)

            # with torch.cuda.amp.autocast():
            output = model(data)
            # 处理评估器
            evaluator.process(data, output,cfg.DATASETS.DATASET_NAME)



    # result = OrderedDict()
    results = evaluator.evaluate()
    # predictions=evaluator.get_all_predictions()
    # print(len(predictions))
    # results= evaluate(predictions,output_folder,cfg.DATASETS.DATASET_NAME)
    # exit(0)
    if is_main_process() and results is not None:

        # if results is not None:
            print('*'*20,type(results))
            for src,result in results.items():
                msg=src
                for key,value in result.items():

                    msg+=f'    {key} = %.4f' % (value)
                print(msg)
    # exit(0)
    return results

def batch_evaluate_v1(cfg,model, data_loader,dataset_name):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    total_num = len(data_loader.dataset)
    # acc_ious = torch.zeros(1).cuda()

    # evaluation variables
    # cum_I, cum_U = 0, 0
    # cum_I = torch.zeros(1).cuda()
    # cum_U = torch.zeros(1).cuda()
    # eval_seg_iou_list = [.5, .7, .9]
    # seg_correct = torch.zeros(len(eval_seg_iou_list)).cuda()

    # if output_folder is None:
    output_folder = os.path.join(cfg.OUTPUT_DIR, cfg.DATASETS.DATASET_NAME,"inference")
    os.makedirs(output_folder, exist_ok=True)

    # evaluator=GVPReferEvaluator(
    #     cfg.DATASETS.DATASET_NAME,
    #     distributed=True,
    #     output_dir=output_folder,
    # )

    predictions = []
    _cpu_device = torch.device("cpu")
    _available_sources = ["refcoco", "grefcoco", "gvpcoco"]
    pr_thres = [.7, .8, .9]
    accum_I = {src: torch.zeros(1).cuda() for src in _available_sources}
    accum_U = {src: torch.zeros(1).cuda() for src in _available_sources}
    accum_IoU = {src: torch.zeros(1).cuda() for src in _available_sources}
    pr_count = {src: {thres: torch.zeros(1).cuda() for thres in pr_thres} for src in _available_sources}
    total_count = {src: torch.zeros(1).cuda() for src in _available_sources}
    not_empty_count = {src: torch.zeros(1).cuda() for src in _available_sources}
    empty_count = {src: torch.zeros(1).cuda() for src in _available_sources}
    nt = {src: {"TP": torch.zeros(1).cuda(), "TN": torch.zeros(1).cuda(), "FP": torch.zeros(1).cuda(), "FN": torch.zeros(1).cuda()} for src in _available_sources}

    with torch.no_grad():
        for data in metric_logger.log_every(data_loader, 100, header):

            for key in data.keys():
                data[key] = data[key].cuda(non_blocking=True)

            # with torch.cuda.amp.autocast():
            output = model(data)
            # 处理评估器
            # batch_predictions = process(data, output, cfg.DATASETS.DATASET_NAME)
            # predictions.extend(batch_predictions)  # Use extend instead of +=
            batch_size = data['image'].shape[0] # 假设 batch 中每个字段的第一个维度是 batch_size
            for i in range(batch_size):
                src = dataset_name
                assert src in _available_sources
                # 提取 GT 和预测 mask
                gt_mask = data['gt_mask_merged'][i].to(_cpu_device)
                output_mask = output[i]["ref_seg"].argmax(dim=0).to(_cpu_device)

                pred_mask = np.array(output_mask, dtype=np.int8)
                gt = np.array(gt_mask, dtype=np.int8)
                # 提取 NT label
                gt_nt = data.get('empty', [False])[i]  # 如果 'empty' 是可选的，提供默认值
                output_nt = output[i]["nt_label"].argmax(dim=0).bool().to(_cpu_device)
                pred_nt = bool(output_nt)
                I, U = computeIoU(pred_mask, gt)

                if gt_nt:
                    empty_count[src] += 1
                    # True Positive
                    if pred_nt:
                        nt[src]["TP"] += 1
                        accum_IoU[src] += 1
                        accum_I[src] += 0
                        accum_U[src] += 0
                    # False Negative
                    else:
                        nt[src]["FN"] += 1
                        accum_IoU[src] += 0
                        accum_I[src] += 0
                        accum_U[src] += int(U)
                # Targeted Samples
                else:
                    # False Positive
                    if pred_nt:
                        nt[src]["FP"] += 1
                        I = 0
                    # True Negative
                    else:
                        nt[src]["TN"] += 1

                    this_iou = float(0) if U == 0 else float(I) / float(U)

                    accum_IoU[src] += this_iou
                    accum_I[src] += I
                    accum_U[src] += U

                    not_empty_count[src] += 1

                    for thres in pr_thres:
                        if this_iou >= thres:
                            pr_count[src][thres] += 1

                total_count[src] += 1

    torch.cuda.synchronize()
    if is_distributed():
        total_count[src] = utils.all_reduce_tensor(total_count[src], norm=False).cpu().numpy()
        accum_U[src] = utils.all_reduce_tensor(accum_U[src], norm=False).cpu().numpy()
        accum_I[src] = utils.all_reduce_tensor(accum_I[src], norm=False).cpu().numpy()
        accum_IoU[src] = utils.all_reduce_tensor(accum_IoU[src], norm=False).cpu().numpy()
        nt[src]["TN"] = utils.all_reduce_tensor(nt[src]["TN"], norm=False).cpu().numpy()
        nt[src]["FP"] = utils.all_reduce_tensor(nt[src]["FP"], norm=False).cpu().numpy()
        nt[src]["FN"] = utils.all_reduce_tensor(nt[src]["FN"], norm=False).cpu().numpy()
        nt[src]["TP"] = utils.all_reduce_tensor(nt[src]["TP"], norm=False).cpu().numpy()
        not_empty_count[src] = utils.all_reduce_tensor(not_empty_count[src], norm=False).cpu().numpy()
        empty_count[src] = utils.all_reduce_tensor(empty_count[src], norm=False).cpu().numpy()

        for thres in pr_thres:
            pr_count[src][thres] = utils.all_reduce_tensor(pr_count[src][thres], norm=False).cpu().numpy()

    else:
        total_count[src] = total_count[src].cpu().numpy()
        accum_U[src] = accum_U[src].cpu().numpy()
        accum_I[src] = accum_I[src].cpu().numpy()
        accum_IoU[src] = accum_IoU[src].cpu().numpy()
        nt[src]["TN"] = nt[src]["TN"].cpu().numpy()
        nt[src]["FP"] = nt[src]["FP"].cpu().numpy()
        nt[src]["FN"] = nt[src]["FN"].cpu().numpy()
        nt[src]["TP"] = nt[src]["TP"].cpu().numpy()
        not_empty_count[src] = not_empty_count[src].cpu().numpy()
        empty_count[src] = empty_count[src].cpu().numpy()
        for thres in pr_thres:
            pr_count[src][thres] = pr_count[src][thres].cpu().numpy()

    detected_srcs = [src for src in _available_sources if total_count[src] > 0]

    final_results_list = []
    # print(total_count[src],not_empty_count[src], empty_count[src],len(data_loader.dataset))
    # results for each source
    for src in detected_srcs:
        res = {}
        # 转换为可序列化格式
        res['gIoU'] = float(100. * (accum_IoU[src] / total_count[src]))  # 确保转换为浮动值
        res['cIoU'] = float(accum_I[src] * 100. / accum_U[src])  # 确保转换为浮动值

        if empty_count[src] > 0:
            res['T_acc'] = float(nt[src]['TN'] / (nt[src]['TN'] + nt[src]['FP']))
            res['N_acc'] = float(nt[src]['TP'] / (nt[src]['TP'] + nt[src]['FN']))
        else:
            res['T_acc'] = res['N_acc'] = 0.0  # 如果没有空样本，直接赋值为0

        # 处理 Pr@{thres} 的部分
        for thres in pr_thres:
            pr_name = f'Pr@{thres:1.1f}'
            res[pr_name] = float(pr_count[src][thres] * 100. / not_empty_count[src])  # 转换为浮动值

        final_results_list.append((src, res))
    # predictions=evaluator.get_all_predictions()
    # print(len(predictions))
    # results= evaluate(predictions,output_folder,cfg.DATASETS.DATASET_NAME)

    if len(detected_srcs) > 1:
        res_full = {}
        res_full['gIoU'] = 100. * _sum_values(accum_IoU) / _sum_values(total_count)
        res_full['cIoU'] = 100. * _sum_values(accum_I) / _sum_values(accum_U)

        for thres in pr_thres:
            pr_name = 'Pr@{0:1.1f}'.format(thres)
            res_full[pr_name] = sum([pr_count[src][thres] for src in detected_srcs]) * 100. / _sum_values(
                not_empty_count)

        final_results_list.append(('full', res_full))

    if output_folder:
        file_path = os.path.join(output_folder, f"{dataset_name}_results.json")
        with PathManager.open(file_path, "w") as f:
            f.write(json.dumps(final_results_list, indent=4))

    results = OrderedDict(final_results_list)

    for src, result in results.items():
        msg = src
        for key, value in result.items():
            msg += f'    {key} = %.4f' % (value)
        print(msg)

    return results

def _sum_values(x):
    return sum(x.values())
def process( inputs, outputs,dataset_name):
    _cpu_device = torch.device("cpu")
    _available_sources = ["refcoco", "grefcoco", "gvpcoco"]

    _predictions=[]
    batch_size = len(inputs['image'])  # 假设 batch 中每个字段的第一个维度是 batch_size
    for i in range(batch_size):
        # 从 batch 提取单个样本
        # img_id = inputs['image_id'][i]
        src = dataset_name
        assert src in _available_sources

        # 提取 GT 和预测 mask
        gt_mask = inputs['gt_mask_merged'][i].to(_cpu_device)
        output_mask = outputs[i]["ref_seg"].argmax(dim=0).to(_cpu_device)

        pred_mask = np.array(output_mask, dtype=np.int8)
        gt = np.array(gt_mask, dtype=np.int8)

        # 提取 NT label
        gt_nt = inputs.get('empty', [False])[i]  # 如果 'empty' 是可选的，提供默认值
        output_nt = outputs[i]["nt_label"].argmax(dim=0).bool().to(_cpu_device)
        pred_nt = bool(output_nt)

        I,U=computeIoU(pred_mask, gt)

        del gt,pred_mask
        _predictions.append({
            # 'img_id': img_id,
            'source': src,
            # 'sent': input['sentence']['raw'],
            # 'sent_info':input['sentence'],
            'pred_nt': pred_nt,
            'gt_nt': gt_nt,
            'I': I,
            'U': U
            # 'pred_mask': pred_mask,
            # 'gt_mask': gt
        })

    return _predictions

def evaluate(predictions,output_dir,dataset_name):

    _available_sources = ["refcoco", "grefcoco", "gvpcoco"]



    pr_thres = [.7, .8, .9]
    accum_I = {src: 0 for src in _available_sources}
    accum_U = {src: 0 for src in _available_sources}
    accum_IoU = {src: 0 for src in _available_sources}
    pr_count = {src: {thres: 0 for thres in pr_thres} for src in _available_sources}
    total_count = {src: 0 for src in _available_sources}
    not_empty_count = {src: 0 for src in _available_sources}
    empty_count = {src: 0 for src in _available_sources}
    nt = {src: {"TP": 0, "TN": 0, "FP": 0, "FN": 0} for src in _available_sources}

    for eval_sample in predictions:
        src = eval_sample['source']
        assert src in _available_sources

        ref_result = {}
        ref_result['source'] = src
        # ref_result['img_id'] = eval_sample['img_id']
        ref_result['gt_nt'] = eval_sample['gt_nt']
        ref_result['pred_nt'] = eval_sample['pred_nt']
        # ref_result['sent'] = eval_sample['sent']
        # ref_result['sent_info'] = eval_sample['sent_info']

        # I, U = computeIoU(eval_sample['pred_mask'], eval_sample['gt_mask'])
        I, U = eval_sample['I'], eval_sample['U']

        # No-target Samples
        if eval_sample['gt_nt']:
            empty_count[src] += 1
            ref_result['I'] = int(0)

            # True Positive
            if eval_sample['pred_nt']:
                nt[src]["TP"] += 1
                accum_IoU[src] += 1
                accum_I[src] += 0
                accum_U[src] += 0

                ref_result['U'] = int(0)
                ref_result['cIoU'] = float(1)

            # False Negative
            else:
                nt[src]["FN"] += 1
                accum_IoU[src] += 0
                accum_I[src] += 0
                accum_U[src] += int(U)

                ref_result['U'] = int(U)
                ref_result['cIoU'] = float(0)

        # Targeted Samples
        else:
            # False Positive
            if eval_sample['pred_nt']:
                nt[src]["FP"] += 1
                I = 0

            # True Negative
            else:
                nt[src]["TN"] += 1

            this_iou = float(0) if U == 0 else float(I) / float(U)

            accum_IoU[src] += this_iou
            accum_I[src] += I
            accum_U[src] += U

            not_empty_count[src] += 1

            for thres in pr_thres:
                if this_iou >= thres:
                    pr_count[src][thres] += 1

            ref_result['I'] = int(I)
            ref_result['U'] = int(U)
            ref_result['cIoU'] = float(this_iou)

        total_count[src] += 1
        # results_dict.append(ref_result)

    detected_srcs = [src for src in _available_sources if total_count[src] > 0]

    final_results_list = []

    # results for each source
    for src in detected_srcs:
        res = {}
        res['gIoU'] = 100. * (accum_IoU[src] / total_count[src])  ##miou
        res['cIoU'] = accum_I[src] * 100. / accum_U[src]  ##oiou


        if empty_count[src] > 0:
            res['T_acc'] = nt[src]['TN'] / (nt[src]['TN'] + nt[src]['FP'])
            res['N_acc'] = nt[src]['TP'] / (nt[src]['TP'] + nt[src]['FN'])
        else:
            res['T_acc'] = res['N_acc'] = 0

        for thres in pr_thres:
            pr_name = 'Pr@{0:1.1f}'.format(thres)
            res[pr_name] = pr_count[src][thres] * 100. / not_empty_count[src]

        final_results_list.append((src, res))

    def _sum_values(x):
        return sum(x.values())

    # global results
    if len(detected_srcs) > 1:
        res_full = {}
        res_full['gIoU'] = 100. * _sum_values(accum_IoU) / _sum_values(total_count)
        res_full['cIoU'] = 100. * _sum_values(accum_I) / _sum_values(accum_U)

        for thres in pr_thres:
            pr_name = 'Pr@{0:1.1f}'.format(thres)
            res_full[pr_name] = sum([pr_count[src][thres] for src in detected_srcs]) * 100. / _sum_values(
                not_empty_count)

        final_results_list.append(('full', res_full))

    if output_dir:
        file_path = os.path.join(output_dir, f"{dataset_name}_results.json")
        with PathManager.open(file_path, "w") as f:
            f.write(json.dumps(final_results_list, indent=4))


    results = OrderedDict(final_results_list)

    return results