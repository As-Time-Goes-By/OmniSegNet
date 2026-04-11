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

from .omniref_refer import G_REFER

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
     
    return mask

    

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

def load_data_json(refer_root, dataset_name, split, image_root, extra_annotation_keys=None,
                       extra_refer_keys=None):
    
    dataset_id = '_'.join([dataset_name,  split])

    timer = Timer()
    refer_root = PathManager.get_local_path(refer_root)
    with contextlib.redirect_stdout(io.StringIO()):
        refer_api = G_REFER(data_root=refer_root,image_root=image_root,
                            dataset=dataset_name,
                            split=split)
    if timer.seconds() > 1:
        print("Loading {} takes {:.2f} seconds.".format(dataset_id, timer.seconds()))

    ref_ids = refer_api.getRefIds(split=split)
    img_ids = refer_api.getImgIds(ref_ids)
    refs = refer_api.loadRefs(ref_ids)
    imgs = [refer_api.loadImgs(ref['image_id'])[0] for ref in refs]
    anns = [refer_api.loadAnns(ref['ann_id']) for ref in refs]
    imgs_refs_anns = list(zip(imgs, refs, anns))

    print(
        "Loaded {} images, {} referring object sets in OmniRef text-test format from {}".format(len(img_ids), len(ref_ids),
                                                                                        dataset_id))

    dataset_dicts = []

    ann_keys = ["iscrowd", "bbox", "category_id"] + (extra_annotation_keys or [])
    ref_keys = ["raw", "sent_id"] + (extra_refer_keys or [])

    ann_lib = {}

   

    for (img_dict, ref_dict, anno_dicts) in imgs_refs_anns:
        record = {}
        record["file_name"] = os.path.join(image_root, img_dict["file_name"])
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        image_id = record["image_id"] = img_dict["id"]

        # Check that information of image, ann and ref match each other
        # This fails only when the data parsing logic or the annotation file is buggy.
        assert ref_dict['image_id'] == image_id
        assert ref_dict['split'] == split
        # assert ref_dict['split'] in ['val','test','testA','testB']
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
                    assert segm  
                    if isinstance(segm, dict):
                        if isinstance(segm["counts"], list):
                            # convert to compressed RLE
                            segm = mask_util.frPyObjects(segm, *segm["size"])
                    else:
                        
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
   


    return dataset_dicts

class OmniRefTextDataSet(Data.Dataset):
    def __init__(self, cfg,split):
        super(OmniRefTextDataSet, self).__init__()
        self.__C = cfg
        self.split=split
        assert cfg.DATASETS.DATASET_NAME in ['OmniRef']

    
        self.dataset_name=cfg.DATASETS.DATASET_NAME

        # --------------------------
        # ---- Raw data loading ---
        # --------------------------
        self.tokenizer = BertTokenizer.from_pretrained(cfg.REFERRING.BERT_TYPE)
        self.max_tokens = cfg.REFERRING.MAX_TOKENS
        self.img_format=cfg.INPUT.FORMAT
        self.merge = True

        
        self.tfm_gens = build_transform_test(cfg)
        self.is_train = False

        
        self.image_root=cfg.DATASETS.IMAGE_ROOT
        self.datadicts=load_data_json(cfg.DATASETS.REF_ROOT, self.dataset_name, split, self.image_root, extra_annotation_keys=None, extra_refer_keys=None)




    
    def __getitem__(self, idx):
        
        dataset_dict=self.datadicts[idx]
        new_dataset_dict={}
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)
      

        image_shape = image.shape[:2]  # h, w
       
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

            gt_classes_tensor = torch.tensor([-1], dtype=torch.int64)

        

        new_dataset_dict["empty"] = empty
       
        new_dataset_dict["gt_mask_merged"] = self._merge_masks(gt_masks_tensor) if self.merge else None
       


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

    def __len__(self):
        return len(self.datadicts)

    def _merge_masks(self,x):
        merged_masks=torch.stack(x,dim=0)
        

        return merged_masks.sum(dim=0, keepdim=True).clamp(max=1)
       

    def transform_annotation(self, obj, resize_image_shape, image_shape):
        """
        Transforms annotation to a standard format (dict with 'mask' and 'bbox').
        """
        mask = convert_coco_poly_to_mask(obj["segmentation"], image_shape[0], image_shape[1])
        mask=F.interpolate(mask.float().unsqueeze(0).unsqueeze(0).float(),resize_image_shape,mode='nearest').squeeze().long()
        return {"mask": mask}

    def shuffle_list(self, list):
        random.shuffle(list)







