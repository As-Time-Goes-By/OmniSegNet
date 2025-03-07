
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
"""
This file contains functions to parse RefCOCO-format annotations into dicts in "Detectron2 format".
"""

logger = logging.getLogger(__name__)

__all__ = ["load_gvpcoco_json"]

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
def load_gvpcoco_json(split, image_root, extra_annotation_keys=None):


    ref_file=f'/data/zqc/datasets/ref/select_image/supp_images/refcoco/{split}_gvpcoco_nt.json'
    coco_file=f'/data/zqc/datasets/ref/images/annotations/instances_train2014.json'
    coco = COCO(coco_file)

    ref_data=json.load(open(ref_file, 'r'))

    # print(
    #     "Loaded  {} referring object sets in G_RefCOCO format from ".format(len(ref_data)))

    dataset_dicts = []

    ann_keys = ["iscrowd", "bbox", "category_id"] + (extra_annotation_keys or [])
    # ref_keys = ["raw", "sent_id"] + (extra_refer_keys or [])

    ann_lib = {}

    NT_count = 0
    MT_count = 0

    for ref_dict in ref_data:
        record = {}
        record["source"] = 'gvpcoco'
        record["file_name"] = os.path.join(image_root,ref_dict["file_name"])
        record["height"] = ref_dict["height"]
        record["width"] = ref_dict["width"]
        image_id = record["image_id"] = ref_dict["image_id"]

        # Check that information of image, ann and ref match each other
        # This fails only when the data parsing logic or the annotation file is buggy.
        assert ref_dict['image_id'] == image_id
        # assert ref_dict['split'] == split
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


        #     ref_record["sentence"] = ref
        dataset_dicts.append(record)

    #     dataset_dicts=dataset_dicts[:12000]
    return dataset_dicts

class GvpCOCODataSet(Data.Dataset):
    def __init__(self, cfg,split,splitby):
        super(GvpCOCODataSet, self).__init__()
        self.__C = cfg
        self.split=split
        assert cfg.DATASETS.DATASET_NAME in ['refcoco', 'refcoco+', 'refcocog','grefcoco','gvpcoco','merge']

        if cfg.DATASETS.DATASET_NAME == 'refcocop':
            cfg.DATASETS.DATASET_NAME = 'refcoco+'
        if cfg.DATASETS.DATASET_NAME == 'refcoco' or cfg.DATASETS.DATASET_NAME == 'refcoco+':
            assert splitby == 'unc'
        if cfg.DATASETS.DATASET_NAME == 'refcocog':
            assert splitby == 'umd' or splitby == 'google'
        # --------------------------
        # ---- Raw data loading ---
        # --------------------------
        # self.tokenizer = BertTokenizer.from_pretrained(cfg.REFERRING.BERT_TYPE)
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
        # self.image_root=os.path.join('/data/zqc/datasets/MSCOCO2014', f'{split}2014')
        # self.image_root='/data/zqc/datasets/MSCOCO2014'
        self.image_path = '/data/zqc/datasets/ref/images/train2014'
        self.supp_image_anns = json.load(
            open('/data/zqc/datasets/ref/select_image/supp_images/refcoco/suppImageAnns.json', 'r', encoding="utf-8"))
        #stat_refs_list=json.load(open(__C.ANN_PATH[__C.DATASET], 'r'))
        self.datadicts=load_gvpcoco_json(split, self.image_path, extra_annotation_keys=None)

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


        new_dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        # dataset_dict["padding_mask"] = torch.as_tensor(np.ascontiguousarray(padding_mask))

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


        new_dataset_dict["empty"] = empty
        # new_dataset_dict["empty"] = torch.tensor([empty], dtype=torch.bool)
        new_dataset_dict["gt_mask_merged"] = self._merge_masks(gt_masks_tensor) if self.merge else None
        # print('mask shape', new_dataset_dict["gt_mask_merged"].shape,empty)

        if not empty:
            start_time = time.time()
            while True:
                supp = np.random.choice(self.supp_image_anns[str(dataset_dict['category_id'])], 1, replace=False)[0]
                if supp['image_id'] != dataset_dict['image_id']:
                    if supp['image_id'] != dataset_dict['image_id']:
                        # supp_image, supp_instance, supp_mask = self.get_supp_instance(supp)
                        supp_image, supp_mask = self.get_supp_mask_image(supp)
                        # print(sorted(supp_mask.unique().tolist()))
                        if sorted(supp_mask.unique().tolist()) == [0, 1]:
                            new_dataset_dict["supp_image"] = supp_image
                            new_dataset_dict["supp_mask"] = supp_mask
                            break
                if time.time() - start_time > 30:
                    raise TimeoutError("循环超时，超过10秒未找到有效的supp")
            # supp_cat_id=dataset_dict['category_id']
        else:
            supp_cat_ids = dataset_dict['supp_cat_ids']
            # supp_cat_id = np.random.choice(supp_cat_ids, 1, replace=False)[0]
            supp_image_id = dataset_dict['supp_image_id']

            start_time = time.time()
            while True:
                # supp = np.random.choice(self.supp_image_anns[str(supp_cat_id)], 1, replace=False)[0]
                supp_cat_id = np.random.choice(supp_cat_ids, 1, replace=False)[0]
                supp = self.get_supp_ann(supp_cat_id, supp_image_id)
                if supp['image_id'] != dataset_dict['image_id']:
                    # supp_image,supp_instance,supp_mask=self.get_supp_instance(supp)
                    supp_image, supp_mask = self.get_supp_mask_image(supp)
                    # print(sorted(supp_mask.unique().tolist()))
                    if sorted(supp_mask.unique().tolist()) == [0, 1]:
                        new_dataset_dict["supp_image"] = supp_image
                        new_dataset_dict["supp_mask"] = supp_mask
                        break
                if time.time() - start_time > 30:
                    raise TimeoutError("循环超时，超过10秒未找到有效的supp")


        return new_dataset_dict

    def __len__(self):
        return len(self.datadicts)
    def get_supp_mask_image(self, supp):
        annotations = supp['annotations']
        # assert len(annotations)==1,'signle not only signle ann!!!'
        supp_image_path = os.path.join(self.image_path, supp['file_name'])

        supp_image=Image.open(supp_image_path).convert('RGB')
        # print('supp_image shape0',supp_image.size,np.array(supp_image).shape)
        # if not isinstance(annotations['segmentation'][0], list):
        #     # print(type(obj['segmentation']), type(obj['segmentation'][0]))
        #     # print('*' * 40)
        #     annotations['segmentation'][0] = annotations['segmentation'][0].tolist()

        if isinstance(annotations['segmentation'], list):
            # Polygon 格式
            rle = coco_mask.frPyObjects(annotations['segmentation'], supp['height'], supp['width'])
            rle = coco_mask.merge(rle)  # 合并为 RLE
        elif isinstance(annotations['segmentation'], dict) and 'counts' in annotations['segmentation']:
            # RLE 格式
            rle = annotations['segmentation']
        else:
            raise ValueError("Unknown segmentation format!")

        supp_mask = coco_mask.decode(rle)  # 生成二值化 mask (H, W)
        # print('supp_maske shape0', supp_mask.shape,np.unique(supp_mask))
        supp_mask = Image.fromarray(supp_mask, mode="P")

        supp_image=self.supp_image_tfm(supp_image)
        # print('supp_image shape1', supp_image.shape)
        supp_mask=self.supp_mask_tfm(supp_mask).clamp(min=0,max=1)
        # print('supp_maske shape1', supp_mask.shape)
        # exit(0)
        return supp_image,supp_mask

    def get_supp_ann(self,supp_cat_id,supp_image_id):
        for ann in self.supp_image_anns[str(supp_cat_id)]:
            if ann['image_id']==supp_image_id:
                return ann
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





