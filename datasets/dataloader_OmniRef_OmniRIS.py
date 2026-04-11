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
import collections
import copy
import os
import time
from random import random
import json
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




def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    
    if isinstance(segmentations[0], list) and len(segmentations[0]) > 0:
        rles = coco_mask.frPyObjects(segmentations, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        
    return mask

def build_transform_image(cfg):
    image_size = cfg.INPUT.IMAGE_SIZE
    image_transform = transforms.Compose([
        transforms.Resize(size=(image_size,image_size)),  # 调整大小
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return image_transform
def build_transform_mask(cfg):
    image_size = cfg.INPUT.IMAGE_SIZE
    mask_transform = transforms.Compose([
        transforms.Resize(size=(image_size,image_size), interpolation=transforms.InterpolationMode.NEAREST),  # 使用最近邻插值以保留类别信息
        transforms.Lambda(lambda img: torch.from_numpy(np.array(img, dtype=np.uint8)))
    ])
    return  mask_transform


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


def build_imgs_refs_anns(refs_file, instances_file, split="val"):
    start = time.time()

    # 加载数据
    with open(refs_file, 'r') as f:
        refs_data = json.load(f)[split]
    with open(instances_file, 'r') as f:
        instances_data = json.load(f)[split]

   
    refs = [ref for ref in refs_data if ref.get("split") == split]

   
    img_dict = {img['id']: img for img in instances_data['images']}
    ann_dict = {ann['id']: ann for ann in instances_data['annotations']}
    ann_dict[-1]=None
    
    imgs_refs_anns = []
    for ref in refs:
        img = img_dict.get(ref['image_id'])
        

        ann=[ann_dict[ann_id] for ann_id in ref['ann_id']]

        if img and ann:
            imgs_refs_anns.append((img, ref, ann))

    elapsed = time.time() - start
    print(f"Loading OmniRef {split} takes {elapsed:.2f}s. Total triples: {len(imgs_refs_anns)}")

    return imgs_refs_anns

def process_visual_data(image_root,ref_dict,ann_id_to_data):
    ann_keys = ["iscrowd", "bbox", "category_id"]
    

    ann_lib = {}
    record = {}
    
    record["file_name"] = os.path.join(image_root,ref_dict["file_name"])
    record["height"] = ref_dict["height"]
    record["width"] = ref_dict["width"]
    image_id = record["image_id"] = ref_dict["image_id"]

    
    assert ref_dict['image_id'] == image_id
    
    if not isinstance(ref_dict['ann_id'], list):
        ref_dict['ann_id'] = [ref_dict['ann_id']]

    
    if -1 in ref_dict['ann_id'] and len(ref_dict['ann_id'])==1:
        anno_dicts=[None]
    else:
        
        anno_dicts = [ann_id_to_data[aid] for aid in ref_dict['ann_id'] if aid in ann_id_to_data]


    if None in anno_dicts:
        assert anno_dicts == [None]
        assert ref_dict['ann_id'] == [-1]
        record['empty'] = True
        obj = {key: None for key in ann_keys if key in ann_keys}
        obj["bbox_mode"] = BoxMode.XYWH_ABS
        obj["empty"] = True
        obj = [obj]
        record['supp_image_id'] = ref_dict['supp_image_id']
        record['supp_cat_ids'] = ref_dict['supp_cat_ids']

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
                        
                        segm = mask_util.frPyObjects(segm, *segm["size"])
                else:
                   
                    segm = [poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6]
                    if len(segm) == 0:
                        num_instances_without_valid_segmentation += 1
                        continue 
                ann["segmentation"] = segm
                ann_lib[ann_id] = ann

            obj.append(ann)




    record["annotations"] = obj
    record["category_id"]= ref_dict['category_id']
    record["categories"] = ref_dict['categories']

    return record

def load_data_json(refer_root, dataset_name, split, image_root, extra_annotation_keys=None,
                       extra_refer_keys=None):
   

    dataset_id = '_'.join([dataset_name,  split])


    timer = Timer()
    refer_root = PathManager.get_local_path(refer_root)
   
    if timer.seconds() > 1:
        print("Loading {} takes {:.2f} seconds.".format(dataset_id, timer.seconds()))
   

  

    

    ref_file = f'{refer_root}/OmniRef.json'
    instances_file=f'{refer_root}/instances.json'

    imgs_refs_anns=build_imgs_refs_anns(ref_file,instances_file, split=split)

    with open(instances_file, 'r') as f:
            instances_data = json.load(f)
           
    visual_test_anns = instances_data['visual-test']['annotations']
    visual_test_ann_dict = {ann['id']: ann for ann in visual_test_anns}

    dataset_dicts = []

    ann_keys = ["iscrowd", "bbox", "category_id"] + (extra_annotation_keys or [])
    ref_keys = ["raw", "sent_id"] + (extra_refer_keys or [])

    ann_lib = {}

    

    for (img_dict, ref_dict, anno_dicts) in imgs_refs_anns:
        record = {}
        # record["source"] = 'supp_grefcocog'
        record["file_name"] = os.path.join(image_root, img_dict["file_name"])
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        image_id = record["image_id"] = img_dict["id"]
        # record["visual_data"]={}
        # record["visual_data"]["source"] = 'supp_grefcocog'
        # record["visual_data"]["file_name"] = os.path.join(image_root,ref_dict["file_name"])
        # record["visual_data"]["height"] = img_dict["height"]
        # record["visual_data"]["width"] = img_dict["width"]
        # record["visual_data"]['image_id']=ref_dict["visual_data"]["image_id"]

        # Check that information of image, ann and ref match each other
        # This fails only when the data parsing logic or the annotation file is buggy.
        assert ref_dict['image_id'] == image_id
        # assert record["visual_data"]['image_id'] == image_id
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
            # ref["ref_id"] = ref_dict["ref_id"]
            ref_record["sentence"] = ref
            ref_record['visual_data']=ref_dict['visual_data']
            dataset_dicts.append(ref_record)
    

    return dataset_dicts,visual_test_ann_dict





class OmniRefOmniRISDataSet(Data.Dataset):
    def __init__(self, cfg, split):
        super(OmniRefOmniRISDataSet, self).__init__()
        self.__C = cfg
        self.split = split
        assert cfg.DATASETS.DATASET_NAME in ['OmniRef']

       
        self.dataset_name = cfg.DATASETS.DATASET_NAME

        # --------------------------
        # ---- Raw data loading ---
        # --------------------------
        self.tokenizer = BertTokenizer.from_pretrained(cfg.REFERRING.BERT_TYPE)
        self.max_tokens = cfg.REFERRING.MAX_TOKENS
        self.img_format = cfg.INPUT.FORMAT
        self.merge = True

        
        self.tfm_gens = build_transform_test(cfg)
        self.is_train = False
        self.supp_image_tfm = build_transform_image(cfg)
        self.supp_mask_tfm = build_transform_mask(cfg)

        self.image_path=cfg.DATASETS.IMAGE_ROOT
        self.ref_root=cfg.DATASETS.REF_ROOT
        
        self.datadicts,self.visual_test_ann_dict = load_data_json(cfg.DATASETS.REF_ROOT, self.dataset_name,  split, self.image_path,
                                            extra_annotation_keys=None, extra_refer_keys=None)
        self.supp_image_anns = json.load(
            open(f'{self.ref_root}/SuppImageAnns.json', 'r', encoding="utf-8"))
        

    def __getitem__(self, idx):
        
        dataset_dict = self.datadicts[idx]
        new_dataset_dict = {}
        
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)
       

        image_shape = image.shape[:2]  # h, w

        padding_mask = np.ones(image.shape[:2])

        image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        
        padding_mask = transforms.apply_segmentation(padding_mask)
        padding_mask = ~ padding_mask.astype(bool)

      
        new_dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
       

        annos = [
            self.transform_annotation(obj, image.shape[:2], image_shape)
            for obj in dataset_dict["annotations"]
            if (obj.get("iscrowd", 0) == 0) and not obj.get("empty", False)
        ]

        empty = dataset_dict.get("empty", False)

        
        gt_masks = [anno["mask"] for anno in annos]
        
        gt_classes = [int(obj["category_id"]) for obj in dataset_dict["annotations"]
                      if (obj.get("iscrowd", 0) == 0) and not obj.get("empty", False)]

        if len(gt_masks) > 0:
            gt_masks_tensor = [mask for mask in gt_masks]
            gt_classes_tensor = torch.tensor(list(set(gt_classes)), dtype=torch.int64)
        else:
            gt_masks_tensor = [torch.zeros((image.shape[0], image.shape[1]), dtype=torch.uint8)]
            
            gt_classes_tensor = torch.tensor([-1], dtype=torch.int64)

       
        supp_dataset_dict = self.get_visual_data(dataset_dict['visual_data'])

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

        return  new_dataset_dict,supp_dataset_dict

    def __len__(self):
        return len(self.datadicts)

    def _merge_masks(self, x):
        merged_masks = torch.stack(x, dim=0)
       

        return merged_masks.sum(dim=0, keepdim=True).clamp(max=1)
        

    def transform_annotation(self, obj, resize_image_shape, image_shape):
        """
        Transforms annotation to a standard format (dict with 'mask' and 'bbox').
        """
        mask = convert_coco_poly_to_mask(obj["segmentation"], image_shape[0], image_shape[1])
        # print('mask shape',mask.shape,image_shape)
        mask = F.interpolate(mask.float().unsqueeze(0).unsqueeze(0).float(), resize_image_shape,
                             mode='nearest').squeeze().long()
        # bbox = transforms.apply_box(obj["bbox"])
        return {"mask": mask}

    def shuffle_list(self, list):
        random.shuffle(list)

    def get_visual_data(self,ref_dict):
       
        dataset_dict=process_visual_data(self.image_path,ref_dict,self.visual_test_ann_dict)
        new_dataset_dict={}
        
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)
        

        image_shape = image.shape[:2]  # h, w
        
        padding_mask = np.ones(image.shape[:2])

        image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        
        padding_mask = transforms.apply_segmentation(padding_mask)
        padding_mask = ~ padding_mask.astype(bool)


        new_dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        

        annos = [
            self.transform_annotation(obj, image.shape[:2], image_shape)
            for obj in dataset_dict["annotations"]
            if (obj.get("iscrowd", 0) == 0) and not obj.get("empty", False)
        ]

        empty = dataset_dict.get("empty", False)

        # Process masks and boxes
        gt_masks = [anno["mask"] for anno in annos]
        
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
        

        if not empty:
            start_time = time.time()
            while True:
                supp = np.random.choice(self.supp_image_anns[str(dataset_dict['category_id'])], 1, replace=False)[0]
                if supp['image_id'] != dataset_dict['image_id']:
                    if supp['image_id'] != dataset_dict['image_id']:
                        
                        supp_image, supp_mask = self.get_supp_mask_image(supp)
                       
                        if sorted(supp_mask.unique().tolist()) == [0, 1]:
                            new_dataset_dict["supp_image"] = supp_image
                            new_dataset_dict["supp_mask"] = supp_mask
                            break
                if time.time() - start_time > 30:
                    raise TimeoutError("Loop Timeout")
            
        else:
            supp_cat_ids = dataset_dict['supp_cat_ids']
            
            supp_image_id = dataset_dict['supp_image_id']

            start_time = time.time()
            while True:
                
                supp_cat_id = np.random.choice(supp_cat_ids, 1, replace=False)[0]
                supp = self.get_supp_ann(supp_cat_id, supp_image_id)
                if supp['image_id'] != dataset_dict['image_id']:
                   
                    supp_image, supp_mask = self.get_supp_mask_image(supp)
                   
                    if sorted(supp_mask.unique().tolist()) == [0, 1]:
                        new_dataset_dict["supp_image"] = supp_image
                        new_dataset_dict["supp_mask"] = supp_mask
                        break
                if time.time() - start_time > 30:
                    raise TimeoutError("Loop Timeout")


        return new_dataset_dict
    

    def get_supp_mask_image(self, supp):
        annotations = supp['annotations']
        
        supp_image_path = os.path.join(self.image_path, supp['file_name'])

        supp_image=Image.open(supp_image_path).convert('RGB')
       

        if isinstance(annotations['segmentation'], list):
           
            rle = coco_mask.frPyObjects(annotations['segmentation'], supp['height'], supp['width'])
            rle = coco_mask.merge(rle)  
        elif isinstance(annotations['segmentation'], dict) and 'counts' in annotations['segmentation']:
           
            rle = annotations['segmentation']
        else:
            raise ValueError("Unknown segmentation format!")

        supp_mask = coco_mask.decode(rle)  
        supp_mask = Image.fromarray(supp_mask, mode="P")

        supp_image=self.supp_image_tfm(supp_image)
        
        supp_mask=self.supp_mask_tfm(supp_mask).clamp(min=0,max=1)
       
        return supp_image,supp_mask

    def get_supp_ann(self,supp_cat_id,supp_image_id):
        for ann in self.supp_image_anns[str(supp_cat_id)]:
            if ann['image_id']==supp_image_id:
                return ann







    