import os
import sys
import torch.utils.data as data
import torch
from torchvision import transforms
from torch.autograd import Variable
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import random

# from bert.tokenization_bert import BertTokenizer
from transformers import BertTokenizer
# from torchvision.transforms import transforms
# import h5py
from .refer import REFER
import datasets.transforms as T
from args import get_parser

# Dataset configuration initialization
parser = get_parser()
args = parser.parse_args()

def get_transform(cfg):
    transforms = [T.Resize(cfg.INPUT.IMAGE_SIZE, cfg.INPUT.IMAGE_SIZE),
                  T.ToTensor(),
                  T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                  ]

    return T.Compose(transforms)

# def build_transform_image(cfg):
#     image_size = cfg.INPUT.IMAGE_SIZE
#     image_transform = transforms.Compose([
#         transforms.Resize(size=(image_size,image_size)),  # 调整大小
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
#     return image_transform

class ReferDataset(data.Dataset):

    def __init__(self,
                 cfg,
                 split='train',
                 splitby='unc',
                 eval_mode=False):

        self.classes = []
        self.image_transforms = get_transform(cfg)
        # self.target_transform = target_transforms

        assert cfg.DATASETS.DATASET_NAME in ['refcoco', 'refcoco+', 'refcocog','grefcoco','gvpcoco','merge']

        if cfg.DATASETS.DATASET_NAME == 'refcocop':
            cfg.DATASETS.DATASET_NAME = 'refcoco+'
        if cfg.DATASETS.DATASET_NAME == 'refcoco' or cfg.DATASETS.DATASET_NAME == 'refcoco+':
            assert splitby == 'unc'
        if cfg.DATASETS.DATASET_NAME == 'refcocog':
            assert splitby == 'umd' or splitby == 'google'
        self.split = split
        self.refer = REFER(cfg.DATASETS.REF_ROOT, cfg.DATASETS.DATASET_NAME, splitby)

        self.max_tokens = 20

        ref_ids = self.refer.getRefIds(split=self.split)
        img_ids = self.refer.getImgIds(ref_ids)

        all_imgs = self.refer.Imgs
        self.imgs = list(all_imgs[i] for i in img_ids)
        self.ref_ids = ref_ids

        self.input_ids = []
        self.attention_masks = []
        self.tokenizer = BertTokenizer.from_pretrained(cfg.REFERRING.BERT_TYPE)

        self.eval_mode = eval_mode
        # if we are testing on a dataset, test all sentences of an object;
        # o/w, we are validating during training, randomly sample one sentence for efficiency
        for r in ref_ids:
            ref = self.refer.Refs[r]

            sentences_for_ref = []
            attentions_for_ref = []

            for i, (el, sent_id) in enumerate(zip(ref['sentences'], ref['sent_ids'])):
                sentence_raw = el['raw']
                attention_mask = [0] * self.max_tokens
                padded_input_ids = [0] * self.max_tokens

                input_ids = self.tokenizer.encode(text=sentence_raw, add_special_tokens=True)

                # truncation of tokens
                input_ids = input_ids[:self.max_tokens]

                padded_input_ids[:len(input_ids)] = input_ids
                attention_mask[:len(input_ids)] = [1]*len(input_ids)

                sentences_for_ref.append(torch.tensor(padded_input_ids).unsqueeze(0))
                attentions_for_ref.append(torch.tensor(attention_mask).unsqueeze(0))

            self.input_ids.append(sentences_for_ref)
            self.attention_masks.append(attentions_for_ref)

    def get_classes(self):
        return self.classes

    def __len__(self):
        return len(self.ref_ids)

    def __getitem__(self, index):
        this_ref_id = self.ref_ids[index]
        this_img_id = self.refer.getImgIds(this_ref_id)
        this_img = self.refer.Imgs[this_img_id[0]]

        img = Image.open(os.path.join(self.refer.IMAGE_DIR, this_img['file_name'])).convert("RGB")

        ref = self.refer.loadRefs(this_ref_id)

        ref_mask = np.array(self.refer.getMask(ref[0])['mask'])
        annot = np.zeros(ref_mask.shape)
        annot[ref_mask == 1] = 1

        annot = Image.fromarray(annot.astype(np.uint8), mode="P")

        if self.image_transforms is not None:
            # resize, from PIL to tensor, and mean and std normalization
            img, gt_masks_tensor = self.image_transforms(img, annot)

        # gt_masks_tensor = F.interpolate(mask.float().unsqueeze(0).unsqueeze(0).float(), img.shape[-2],
        #                      mode='nearest').squeeze().long()
        if self.eval_mode:
            embedding = []
            att = []
            for s in range(len(self.input_ids[index])):
                e = self.input_ids[index][s]
                a = self.attention_masks[index][s]
                embedding.append(e.unsqueeze(-1))
                att.append(a.unsqueeze(-1))

            tensor_embeddings = torch.cat(embedding, dim=-1)
            attention_mask = torch.cat(att, dim=-1)
        else:
            choice_sent = np.random.choice(len(self.input_ids[index]))
            tensor_embeddings = self.input_ids[index][choice_sent]
            attention_mask = self.attention_masks[index][choice_sent]

        new_dataset_dict={}
        new_dataset_dict['image']=img
        new_dataset_dict['lang_tokens']=tensor_embeddings
        new_dataset_dict['lang_mask'] = attention_mask
        new_dataset_dict["empty"] = False
        # new_dataset_dict["empty"] = torch.tensor([empty], dtype=torch.bool)
        new_dataset_dict["gt_mask_merged"] = gt_masks_tensor.unsqueeze(0)
        # print('mask shape', new_dataset_dict["gt_mask_merged"].shape)
        # exit(0)


        # return img, target, tensor_embeddings, attention_mask
        return new_dataset_dict

class ReferDatasetTest(data.Dataset):
    def __init__(self,
                 cfg,
                 split='train',
                 splitby='unc',
                 eval_mode=False):

        self.classes = []
        self.image_transforms = get_transform(cfg)

        assert cfg.DATASETS.DATASET_NAME in ['refcoco', 'refcoco+', 'refcocog','grefcoco','gvpcoco','merge']

        if cfg.DATASETS.DATASET_NAME == 'refcocop':
            cfg.DATASETS.DATASET_NAME = 'refcoco+'
        if cfg.DATASETS.DATASET_NAME == 'refcoco' or cfg.DATASETS.DATASET_NAME == 'refcoco+':
            assert splitby == 'unc'
        if cfg.DATASETS.DATASET_NAME == 'refcocog':
            assert splitby == 'umd' or splitby == 'google'

        self.split = split
        self.refer = REFER(cfg.DATASETS.REF_ROOT, cfg.DATASETS.DATASET_NAME, splitby)

        self.max_tokens = 20

        ref_ids = self.refer.getRefIds(split=self.split)
        img_ids = self.refer.getImgIds(ref_ids)

        all_imgs = self.refer.Imgs
        self.imgs = list(all_imgs[i] for i in img_ids)
        self.ref_ids = ref_ids

        self.input_ids = []
        self.attention_masks = []
        self.sentence_ref_ids = []
        self.tokenizer = BertTokenizer.from_pretrained(cfg.REFERRING.BERT_TYPE)

        self.eval_mode = eval_mode
        # if we are testing on a dataset, test all sentences of an object;
        # o/w, we are validating during training, randomly sample one sentence for efficiency
        for r in ref_ids:
            ref = self.refer.Refs[r]
            for i, (el, sent_id) in enumerate(zip(ref['sentences'], ref['sent_ids'])):
                sentence_raw = el['raw']
                attention_mask = [0] * self.max_tokens
                padded_input_ids = [0] * self.max_tokens

                input_ids = self.tokenizer.encode(text=sentence_raw, add_special_tokens=True)

                input_ids = input_ids[:self.max_tokens] # truncation of tokens

                padded_input_ids[:len(input_ids)] = input_ids
                attention_mask[:len(input_ids)] = [1]*len(input_ids)

                self.sentence_ref_ids.append(r)
                self.input_ids.append(torch.tensor(padded_input_ids).unsqueeze(0)) # [[1 x max_tokens]]
                self.attention_masks.append(torch.tensor(attention_mask).unsqueeze(0)) # [[1 x max_tokens]]

        # print(self.refer.getCatIds())
        # self.cat_names = self.refer.loadCats(list(self.refer.getCatIds()))
        # print(self.cat_names)
        # print(len(self.input_ids))

    def get_classes(self):
        return self.classes

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        '''
            target:
                Tensor [HxW]
                or
                List of Tensor [HxW], len(target) == num_masks, referent_mask = target[0]
        '''
        this_ref_id = self.sentence_ref_ids[index]
        this_img_id = self.refer.getImgIds(this_ref_id)
        this_img = self.refer.Imgs[this_img_id[0]]

        img = Image.open(os.path.join(self.refer.IMAGE_DIR, this_img['file_name'])).convert("RGB")

        ref = self.refer.loadRefs(this_ref_id)
        # ref_cls_id = ref[0]['category_id']
        ref_mask = np.array(self.refer.getMask(ref[0])['mask'])
        annot = np.zeros(ref_mask.shape)
        annot[ref_mask == 1] = 1
        annot = Image.fromarray(annot.astype(np.uint8), mode="P")

        # cat = self.refer.loadCats([ref_cls_id])
        # cat = cat[0]
        # ref_cat = self.cat_names.index(cat)    # ref_cat ranges from 0 to 79

        if self.image_transforms is not None:
            # resize, from PIL to tensor, and mean and std normalization
            img, gt_masks_tensor = self.image_transforms(img, annot)

        tensor_embeddings = self.input_ids[index]  # [1 x max_tokens]
        attention_mask = self.attention_masks[index]

        new_dataset_dict = {}
        new_dataset_dict['image'] = img
        new_dataset_dict['lang_tokens'] = tensor_embeddings
        new_dataset_dict['lang_mask'] = attention_mask
        new_dataset_dict["empty"] = False
        # new_dataset_dict["empty"] = torch.tensor([empty], dtype=torch.bool)
        new_dataset_dict["gt_mask_merged"] = gt_masks_tensor.unsqueeze(0)
        # print('mask shape', new_dataset_dict["gt_mask_merged"].shape)
        # exit(0)
        # return img, target, tensor_embeddings, attention_mask
        return new_dataset_dict
