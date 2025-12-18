
import contextlib
import io
import logging
import time

import cv2
import numpy as np
import os
import random
import copy
import pycocotools.mask as mask_util
from fvcore.common.timer import Timer
from PIL import Image
import json
from detectron2.structures import Boxes, BoxMode, PolygonMasks, RotatedBoxes
from detectron2.utils.file_io import PathManager
from detectron2.data import transforms as T
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask
from torchvision.transforms import transforms
import numpy as np
import torch
from PIL import Image
import torch.utils.data as Data
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from detectron2.data import detection_utils as utils
import torch.nn.functional as F


logger = logging.getLogger(__name__)

__all__ = ["load_OmniRef_json"]

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

def build_transform_image(cfg):
    image_size = cfg.INPUT.IMAGE_SIZE
    image_transform = transforms.Compose([
        transforms.Resize(size=(image_size,image_size)),  
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return image_transform
def build_transform_mask(cfg):
    image_size = cfg.INPUT.IMAGE_SIZE
    mask_transform = transforms.Compose([
        transforms.Resize(size=(image_size,image_size), interpolation=transforms.InterpolationMode.NEAREST),   
        transforms.Lambda(lambda img: torch.from_numpy(np.array(img, dtype=np.uint8)))
    ])
    return  mask_transform
def load_OmniRef_json(split, image_root, dataset_peth,extra_annotation_keys=None):


    # ref_file=f'./datasets/OmniRef_{split}.json'
    ref_file = f'{dataset_peth}/OmniRef_{split}.json'
    coco_file=f'./coco2014/annotations/instances_train2014.json'
    coco = COCO(coco_file)

    ref_data=json.load(open(ref_file, 'r'))

 

    dataset_dicts = []

    ann_keys = ["iscrowd", "bbox", "category_id"] + (extra_annotation_keys or [])

    ann_lib = {}

    NT_count = 0
    MT_count = 0

    for ref_dict in ref_data:
        record = {}
        record["source"] = 'OmniRef'
        record["file_name"] = os.path.join(image_root,ref_dict["file_name"])
        record["height"] = ref_dict["height"]
        record["width"] = ref_dict["width"]
        image_id = record["image_id"] = ref_dict["image_id"]

        # Check that information of image, ann and ref match each other
        # This fails only when the data parsing logic or the annotation file is buggy.
        assert ref_dict['image_id'] == image_id
   
        if not isinstance(ref_dict['ann_id'], list):
            ref_dict['ann_id'] = [ref_dict['ann_id']]

        # No target samples
        if -1 in ref_dict['ann_id'] and len(ref_dict['ann_id'])==1:
            anno_dicts=[None]
        else:
            anno_dicts=coco.loadAnns(ref_dict['ann_id'])


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
        record["category_id"]= ref_dict['category_id']
        record["categories"] = ref_dict['categories']



        dataset_dicts.append(record)

    return dataset_dicts

class OmniRefDataSet(Data.Dataset):
    def __init__(self, cfg,split):
        super(OmniRefDataSet, self).__init__()
        self.__C = cfg
        self.split=split
        assert cfg.DATASETS.DATASET_NAME in ['OmniRef']



        self.max_tokens = cfg.REFERRING.MAX_TOKENS
        self.img_format=cfg.INPUT.FORMAT
        self.merge = True

        if split=='train':
            self.tfm_gens = build_transform_train(cfg)
            self.is_train=True
        else:
            self.tfm_gens = build_transform_test(cfg)
            self.is_train = False

        self.supp_image_tfm = build_transform_image(cfg)
        self.supp_mask_tfm = build_transform_mask(cfg)

        self.image_path = './coco2014/train2014'
        self.omniref_path='./datasets'

        self.supp_image_anns = json.load(
            open(f'{self.omniref_path}/suppImageAnns.json', 'r', encoding="utf-8"))
        self.datadicts=load_OmniRef_json(split, self.image_path, self.omniref_path, extra_annotation_keys=None)

    
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
                    raise TimeoutError("time out")

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
                    raise TimeoutError("time out")


        return new_dataset_dict

    def __len__(self):
        return len(self.datadicts)
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








