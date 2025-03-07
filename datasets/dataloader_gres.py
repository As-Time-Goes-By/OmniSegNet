# coding=utf-8
# Copyright 2022 The SimREC Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import copy
import os
from random import random

import cv2
import io
import numpy as np
import contextlib
import torch
import torch.utils.data as Data
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

import transformers
from transformers import BertTokenizer
from PIL import Image
from fvcore.common.timer import Timer
import torch.nn.functional as F
from detectron2.utils.file_io import PathManager
from detectron2.structures import BoxMode
import pycocotools.mask as mask_util

from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

from transformers import BertTokenizer
from pycocotools import mask as coco_mask

from .grefer import G_REFER

def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    # for polygons in segmentations:
    if isinstance(segmentations[0], list) and len(segmentations[0]) > 0:
        rles = coco_mask.frPyObjects(segmentations, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        # masks.append(mask)
    # if masks:
    #     masks = torch.stack(masks, dim=0)
    # else:
    #     masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return mask

    # if isinstance(segmentations[0], list) and len(segmentations[0]) > 0:  # polygon
    #     rles = mask.frPyObjects(segmentations, height, width)
    #     rle = mask.merge(rles)
    # else:
    #     rle = segmentations
    # m = mask.decode(rle)  # 生成的掩码可能为 (H, W) 或 (H, W, N)
    # if len(m.shape) == 3:  # 多通道掩码
    #     m = np.any(m, axis=2).astype(np.uint8)  # 合并为单通道

def build_transform_train(cfg):
    image_size = cfg.INPUT.IMAGE_SIZE
    min_scale = cfg.INPUT.MIN_SCALE

    augmentation = []

    augmentation.extend([
        T.Resize((image_size, image_size))
    ])

    return augmentation


def build_transform_test(cfg):
    image_size = cfg.INPUT.IMAGE_SIZE

    augmentation = []

    augmentation.extend([
        T.Resize((image_size, image_size))
    ])

    return augmentation

def load_grefcoco_json(refer_root, dataset_name, splitby, split, image_root, extra_annotation_keys=None,
                       extra_refer_keys=None):
    if dataset_name == 'refcocop':
        dataset_name = 'refcoco+'
    if dataset_name == 'refcoco' or dataset_name == 'refcoco+':
        splitby == 'unc'
    if dataset_name == 'refcocog':
        assert splitby == 'umd' or splitby == 'google'

    dataset_id = '_'.join([dataset_name, splitby, split])


    # logger.info('Loading dataset {} ({}-{}) ...'.format(dataset_name, splitby, split))
    # logger.info('Refcoco root: {}'.format(refer_root))
    # print('refer_root*****************',refer_root)
    timer = Timer()
    refer_root = PathManager.get_local_path(refer_root)
    with contextlib.redirect_stdout(io.StringIO()):
        refer_api = G_REFER(data_root=refer_root,
                            dataset=dataset_name,
                            splitBy=splitby)
    if timer.seconds() > 1:
        print("Loading {} takes {:.2f} seconds.".format(dataset_id, timer.seconds()))

    ref_ids = refer_api.getRefIds(split=split)
    img_ids = refer_api.getImgIds(ref_ids)
    refs = refer_api.loadRefs(ref_ids)
    imgs = [refer_api.loadImgs(ref['image_id'])[0] for ref in refs]
    anns = [refer_api.loadAnns(ref['ann_id']) for ref in refs]
    imgs_refs_anns = list(zip(imgs, refs, anns))

    print(
        "Loaded {} images, {} referring object sets in G_RefCOCO format from {}".format(len(img_ids), len(ref_ids),
                                                                                        dataset_id))

    dataset_dicts = []

    ann_keys = ["iscrowd", "bbox", "category_id"] + (extra_annotation_keys or [])
    ref_keys = ["raw", "sent_id"] + (extra_refer_keys or [])

    ann_lib = {}

    NT_count = 0
    MT_count = 0

    for (img_dict, ref_dict, anno_dicts) in imgs_refs_anns:
        record = {}
        record["source"] = 'grefcoco'
        record["file_name"] = os.path.join(image_root, img_dict["file_name"])
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        image_id = record["image_id"] = img_dict["id"]

        # Check that information of image, ann and ref match each other
        # This fails only when the data parsing logic or the annotation file is buggy.
        assert ref_dict['image_id'] == image_id
        assert ref_dict['split'] == split
        if not isinstance(ref_dict['ann_id'], list):
            ref_dict['ann_id'] = [ref_dict['ann_id']]

        # No target samples
        if None in anno_dicts:
            assert anno_dicts == [None]
            assert ref_dict['ann_id'] == [-1]
            record['empty'] = True
            obj = {key: None for key in ann_keys if key in ann_keys}
            obj["bbox_mode"] = BoxMode.XYWH_ABS
            obj["empty"] = True
            obj = [obj]

        # Multi target samples
        else:
            record['empty'] = False
            obj = []
            for anno_dict in anno_dicts:
                ann_id = anno_dict['id']
                if anno_dict['iscrowd']:
                    continue
                assert anno_dict["image_id"] == image_id
                assert ann_id in ref_dict['ann_id']

                if ann_id in ann_lib:
                    ann = ann_lib[ann_id]
                else:
                    ann = {key: anno_dict[key] for key in ann_keys if key in anno_dict}
                    ann["bbox_mode"] = BoxMode.XYWH_ABS
                    ann["empty"] = False

                    segm = anno_dict.get("segmentation", None)
                    assert segm  # either list[list[float]] or dict(RLE)
                    if isinstance(segm, dict):
                        if isinstance(segm["counts"], list):
                            # convert to compressed RLE
                            segm = mask_util.frPyObjects(segm, *segm["size"])
                    else:
                        # filter out invalid polygons (< 3 points)
                        segm = [poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6]
                        if len(segm) == 0:
                            num_instances_without_valid_segmentation += 1
                            continue  # ignore this instance
                    ann["segmentation"] = segm
                    ann_lib[ann_id] = ann

                obj.append(ann)

        record["annotations"] = obj

        # Process referring expressions
        sents = ref_dict['sentences']
        for sent in sents:
            ref_record = record.copy()
            ref = {key: sent[key] for key in ref_keys if key in sent}
            ref["ref_id"] = ref_dict["ref_id"]
            ref_record["sentence"] = ref
            dataset_dicts.append(ref_record)
    #         if ref_record['empty']:
    #             NT_count += 1
    #         else:
    #             MT_count += 1

    # logger.info("NT samples: %d, MT samples: %d", NT_count, MT_count)

    # Debug mode
    # return dataset_dicts[:100]

    return dataset_dicts

class RefCOCODataSet(Data.Dataset):
    def __init__(self, cfg,split,splitby):
        super(RefCOCODataSet, self).__init__()
        self.__C = cfg
        self.split=split
        assert cfg.DATASETS.DATASET_NAME in ['refcoco', 'refcoco+', 'refcocog','grefcoco','gvpcoco','merge']

        if cfg.DATASETS.DATASET_NAME == 'refcocop':
            cfg.DATASETS.DATASET_NAME = 'refcoco+'
        if cfg.DATASETS.DATASET_NAME == 'refcoco' or cfg.DATASETS.DATASET_NAME == 'refcoco+':
            assert splitby == 'unc'
        if cfg.DATASETS.DATASET_NAME == 'refcocog':
            assert splitby == 'umd' or splitby == 'google'
        # if cfg.DATASETS.DATASET_NAME == 'merge':
        dataset_name='grefcoco'

        # --------------------------
        # ---- Raw data loading ---
        # --------------------------
        self.tokenizer = BertTokenizer.from_pretrained(cfg.REFERRING.BERT_TYPE)
        self.max_tokens = cfg.REFERRING.MAX_TOKENS
        self.img_format=cfg.INPUT.FORMAT
        self.merge = True

        if split=='train':
            self.tfm_gens = build_transform_train(cfg)
            self.is_train=True
        else:
            self.tfm_gens = build_transform_test(cfg)
            self.is_train = False

        # self.image_root=os.path.join('/data/zqc/datasets/MSCOCO2014', f'{split}2014')
        self.image_root='/data/zqc/datasets/ref/images/train2014'
        #stat_refs_list=json.load(open(__C.ANN_PATH[__C.DATASET], 'r'))
        self.datadicts=load_grefcoco_json(cfg.DATASETS.REF_ROOT, dataset_name, splitby, split, self.image_root, extra_annotation_keys=None, extra_refer_keys=None)

        #self.transforms=transforms.Compose([transforms.ToTensor(), transforms.Normalize(__C.MEAN, __C.STD)])



    
    def __getitem__(self, idx):
        #ref_iter = self.load_refs(idx)
        #ref_iter = ""
        dataset_dict=self.datadicts[idx]
        new_dataset_dict={}
        # dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # image = Image.open(dataset_dict["file_name"]).convert('RGB')
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)
        # if image.size[0]!=dataset_dict['height'] or image.size[1]!=dataset_dict['width']:
        #     print('图片尺寸不对',image.size,dataset_dict['height'],dataset_dict['width'])
        #     exit(0)

        image_shape = image.shape[:2]  # h, w
        # print(image.shape)
        # exit(0)
        # TODO: get padding mask
        # by feeding a "segmentation mask" to the same transforms
        padding_mask = np.ones(image.shape[:2])

        image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        # the crop transformation has default padding value 0 for segmentation
        padding_mask = transforms.apply_segmentation(padding_mask)
        padding_mask = ~ padding_mask.astype(bool)


        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        new_dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        # dataset_dict["padding_mask"] = torch.as_tensor(np.ascontiguousarray(padding_mask))

        # USER: Implement additional transformations if you have other types of data
        # annos = [
        #     utils.transform_instance_annotations(obj, transforms, image_shape)
        #     for obj in dataset_dict.pop("annotations")
        #     if (obj.get("iscrowd", 0) == 0) and (obj.get("empty", False) == False)
        # ]
        # instances = utils.annotations_to_instances(annos, image_shape)

        annos = [
            self.transform_annotation(obj, image.shape[:2], image_shape)
            for obj in dataset_dict["annotations"]
            if (obj.get("iscrowd", 0) == 0) and not obj.get("empty", False)
        ]

        empty = dataset_dict.get("empty", False)

        # Process masks and boxes
        gt_masks = [anno["mask"] for anno in annos]
        # gt_boxes = [anno["bbox"] for anno in annos]
        gt_classes =[int(obj["category_id"]) for obj in dataset_dict["annotations"]
                     if (obj.get("iscrowd", 0) == 0) and not obj.get("empty", False)]

        if len(gt_masks) > 0:
            gt_masks_tensor = [mask for mask in gt_masks]
            gt_classes_tensor = torch.tensor(list(set(gt_classes)), dtype=torch.int64)
        else:
            gt_masks_tensor = [torch.zeros(( image.shape[0], image.shape[1]), dtype=torch.uint8)]
            # gt_boxes = torch.zeros((0, 4), dtype=torch.float32)
            gt_classes_tensor = torch.tensor([-1], dtype=torch.int64)

        # if self.is_train:
        #     dataset_dict["instances"] = {"gt_mask":gt_masks,'gt_classes':gt_classes}
        # else:
        #     dataset_dict["gt_mask"] =

        # dataset_dict["gt_masks"] = gt_masks_tensor
        # # dataset_dict["gt_boxes"] = gt_boxes

        # new_dataset_dict["gt_classes"] = gt_classes_tensor
        # assert gt_classes_tensor.shape[-1]==1,gt_classes_tensor

        # print('get_clsses:', gt_classes_tensor)

        new_dataset_dict["empty"] = empty
        # new_dataset_dict["empty"] = torch.tensor([empty], dtype=torch.bool)
        new_dataset_dict["gt_mask_merged"] = self._merge_masks(gt_masks_tensor) if self.merge else None
        # print('mask shape', new_dataset_dict["gt_mask_merged"].shape,empty)


        # Language data
        sentence_raw = dataset_dict['sentence']['raw']
        attention_mask = [0] * self.max_tokens
        padded_input_ids = [0] * self.max_tokens

        input_ids = self.tokenizer.encode(text=sentence_raw, add_special_tokens=True)
        input_ids = input_ids[:self.max_tokens]
        padded_input_ids[:len(input_ids)] = input_ids
        attention_mask[:len(input_ids)] = [1] * len(input_ids)

        new_dataset_dict['lang_tokens'] = torch.tensor(padded_input_ids).unsqueeze(0)
        new_dataset_dict['lang_mask'] = torch.tensor(attention_mask).unsqueeze(0)

        return new_dataset_dict
        # return dataset_dict["image"],dataset_dict['lang_tokens'],dataset_dict['lang_mask'],dataset_dict["gt_mask_merged"],dataset_dict["gt_classes"]

    def __len__(self):
        return len(self.datadicts)

    def _merge_masks(self,x):
        merged_masks=torch.stack(x,dim=0)
        # merged_masks = sum([mask for mask in x])
        # merged_masks[np.where(merged_masks > 1)] = 1

        return merged_masks.sum(dim=0, keepdim=True).clamp(max=1)
        # return merged_masks

    def transform_annotation(self, obj, resize_image_shape, image_shape):
        """
        Transforms annotation to a standard format (dict with 'mask' and 'bbox').
        """
        mask = convert_coco_poly_to_mask(obj["segmentation"], image_shape[0], image_shape[1])
        # print('mask shape',mask.shape,image_shape)
        mask=F.interpolate(mask.float().unsqueeze(0).unsqueeze(0).float(),resize_image_shape,mode='nearest').squeeze().long()
        # bbox = transforms.apply_box(obj["bbox"])
        return {"mask": mask}

    def shuffle_list(self, list):
        random.shuffle(list)

def loader(cfg,dataset: torch.utils.data.Dataset, rank: int, shuffle,drop_last=False):
    if cfg.SOLVER.MULTIPROCESSING_DISTRIBUTED:
        assert cfg.SOLVER.BATCH_SIZE % len(cfg.SOLVER.GPU) == 0
        assert cfg.SOLVER.NUM_WORKER % len(cfg.SOLVER.GPU) == 0
        assert dist.is_initialized()

        dist_sampler = DistributedSampler(dataset,
                                          num_replicas=cfg.SOLVER.WORLD_SIZE,
                                          rank=rank)

        data_loader = DataLoader(dataset,
                                 batch_size=cfg.SOLVER.BATCH_SIZE // len(cfg.SOLVER.GPU),
                                 shuffle=shuffle,
                                 sampler=dist_sampler,
                                 num_workers=cfg.SOLVER.NUM_WORKER //len(cfg.SOLVER.GPU),
                                 pin_memory=True,
                                 drop_last=drop_last)  # ,
                                # prefetch_factor=_C['PREFETCH_FACTOR'])  only works in PyTorch 1.7.0
    else:
        data_loader = DataLoader(dataset,
                                 batch_size=cfg.SOLVER.BATCH_SIZE,
                                 shuffle=shuffle,
                                 num_workers=cfg.SOLVER.NUM_WORKER,
                                 pin_memory=True,
                                 drop_last=drop_last)
    return data_loader

if __name__ == '__main__':

    class Cfg():
        def __init__(self):
            super(Cfg, self).__init__()
            self.ANN_PATH= {
                'refcoco': './data/anns/refcoco.json',
                'refcoco+': './data/anns/refcoco+.json',
                'refcocog': './data/anns/refcocog.json',
                'vg': './data/anns/vg.json',
                'gres':"/data/huangxiaorui/data/gref_100.json"
            }

            self.IMAGE_PATH={
                'refcoco': './data/images/train2014',
                'refcoco+': './data/images/train2014',
                'refcocog': './data/images/train2014',
                'vg': './data/images/VG',
                'gres':'/data/sunjiamu/home/data/images/train2014'
            }

            self.MASK_PATH={
                'refcoco': './data/masks/refcoco',
                'refcoco+': './data/masks/refcoco+',
                'refcocog': './data/masks/refcocog',
                'vg': './data/masks/vg',
                'gres': './data/masks/refcoco'}
            self.INPUT_SHAPE = (416, 416)
            self.USE_GLOVE = True
            self.DATASET = 'gres'
            self.MAX_TOKEN = 15
            self.MEAN = [0., 0., 0.]
            self.STD = [1., 1., 1.]
            self.LANG_ENC = 'bert'
    cfg=Cfg()
    dataset=RefCOCODataSet(cfg,'val')
    data_loader = DataLoader(dataset,
                             batch_size=10,
                             shuffle=False,
                             pin_memory=True)
    i=0
    '''torch.from_numpy(ref_iter).long(), \
            input_dict['img'], \
            input_dict['mask'], \
            input_dict['box'], \
            torch.from_numpy(gt_box_iter).float(), \
            mask_id,\
            np.array(info_iter),\
            ref_mask_iter'''
    for ref_iter,image_iter, mask_iter, box_iter,gt_box,mask_id,info_iter,ref_mask in data_loader:
        image = image_iter.numpy()[0].transpose((1, 2, 0)) * 255
        mask = mask_iter.numpy()[0].transpose((1, 2, 0)) * 255

        # Ensure image and mask are in uint8 format
        image = image.astype(np.uint8)
        mask = mask.astype(np.uint8)

        # Convert mask to a binary mask (if needed)
        if mask.shape[2] == 3:  # If mask has 3 channels
            mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        else:
            mask_gray = mask.squeeze()  # Remove the last dimension if it exists

        # Create a color mask (optional)
        color_mask = np.zeros_like(image)
        color_mask[:, :, 1] = mask_gray  # Use green channel for the mask

        # Overlay the mask on the image
        alpha = 0.5  # Transparency factor
        overlayed_image = cv2.addWeighted(image, 1 - alpha, color_mask, alpha, 0)

        # Save the images
        cv2.imwrite(f'./mask_test/overlayed_{i}.jpg', overlayed_image)
        i += 1
        if i > 500:
            break





