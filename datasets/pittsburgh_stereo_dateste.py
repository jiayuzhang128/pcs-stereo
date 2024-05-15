import torch.nn as nn
import torch.utils.data as data
import cv2
import random
import numpy as np
import torch
import sys
from torchvision import transforms
from pathlib import Path
from PIL import Image, ImageOps
from collections import OrderedDict

root = sys.path[0].split('/')[0]
sys.path.append(root)

from utils.ops import *

def pil_loader(path):
    """使用PIL读取文件"""
    img = Image.open(path)
    return img

class PittburghStereoDataset(data.Dataset):
    def __init__(self,
                 data_path,
                 list_path,
                 splits,
                 height,
                 width,
                 is_train=False,
                 ftype='png',
                 fill=0,
                 pad_mode='constant',
                 no_norm=True,
                 is_flip=False,
                 flip_direction='h'):
        super(PittburghStereoDataset, self).__init__()
        self.data_path = data_path
        self.img_suffix = 'Resize'
        self.height = height
        self.width = width
        self.is_train = is_train
        self.ftype = ftype
        self.fill = fill
        self.pad_mode = pad_mode

        self.records = OrderedDict()
        self.records = []
        # 读取list中的内容：collection, imgID, RGBExposureTime, NIRExposureTime, RedGain, BlueGain
        self.count = 0
        for split in splits.split(','):
            f = open(Path(list_path) / (split + '.txt'), 'r')
            lines = f.readlines()
            f.close()
            for i, line in enumerate(lines):
                if i > 6:
                  break
                items = line.split()
                collection = items[0]
                img_id = items[1]
                # rgb_exp = float(items[2])
                # nir_exp = float(items[3])
                # red_gain = float(items[4])
                # blue_gain = float(items[5])
                record = (collection, img_id)
                # self.records[self.count] = record
                self.records.append(record)
                self.count += 1

        self.num_records = len(self.records)

        self.interp = Image.LANCZOS
        self.loader = pil_loader
        self.pfm_reader = read_pfm
        self.padder = InputPadder((self.width, self.height), fill=self.fill, pad_mode=self.pad_mode)
        self.padded_height = self.padder.padded_ht
        self.padded_width = self.padder.padded_wd
        self.flip_direction = flip_direction
        self.is_flip = is_flip
        if no_norm:
            self.transform = transforms.ToTensor()
        else:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(std=std, mean =mean)])

        # 设置数据增强参数
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.ColorJitter.get_params(
                self.brightness,
                self.contrast,
                self.saturation,
                self.hue
            )

        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

    def __len__(self):
        return self.num_records

    def __getitem__(self, index):
        """以字典形式返回一个训练数据
            value: torch张量
            key: 
                "rgb"  for rgb images
                "rgb_aug" for augmented rgb images
                "nir" for nir images
                "nir_aug" for augmented nir images
                "semi_proxy_label" for pretrained stereo proxy label 
                "dense_proxy_label" for pretrained stereo & mono proxy label
                
        """
        inputs = {}
        # print("index: ", str(index))
        
        # 随机增强与翻转
        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5
        
        if not self.is_train:
            do_color_aug = False
            do_flip =False

        if not self.is_flip:
            do_flip =False

        collection, img_id = self.records[index]
        inputs["rgb"] = self.get_rgb(collection, img_id, do_flip)
        inputs["nir"] = self.get_nir(collection, img_id, do_flip)

        inputs["semi_proxy_label"] = self.get_proxy_label(collection, img_id, 'semi')
        inputs["dense_proxy_label"] = self.get_proxy_label(collection, img_id, 'dense')
        inputs["mono_proxy_label"] = self.get_proxy_label(collection, img_id, 'mono')

        if do_color_aug:
            color_aug = transforms.ColorJitter(
                self.brightness,
                self.contrast,
                self.saturation,
                self.hue)
        else:
            # 啥也不干
            color_aug = (lambda x:x)
        self.preprocess(inputs, color_aug)
        
        return inputs

    def preprocess(self, inputs, color_aug):
        """数据增强，对所有输入进行同样的数据增强
            inputs: 输入
            color_aug: 预先定义好的颜色增强对象
        """
        # 增强
        for k in list(inputs):
            f = inputs[k]
            if "rgb" == k:
                inputs[k] = self.padder.pad(self.transform(f))
                # inputs["rgb_aug"] = self.padder.pad(self.transform(color_aug(f)))
            elif "nir" == k:
                inputs[k] = self.padder.pad(self.transform(f))
                # inputs["nir_aug"] = self.padder.pad(self.transform(color_aug(f)))
            elif "semi_proxy_label" == k:
                inputs[k] = torch.tensor(f).unsqueeze(0)
            elif "dense_proxy_label" == k:
                inputs[k] = torch.tensor(f).unsqueeze(0)
            elif "mono_proxy_label" == k:
                inputs[k] = torch.tensor(f).unsqueeze(0)

    def get_rgb(self, collection, img_id, do_flip):
        rgb = self.loader(self.filename(collection, img_id, self.img_suffix, 'RGB', self.ftype))
        rgb.convert('RGB')
        size = rgb.size
        if do_flip:
            if self.flip_direction == 'w':
                rgb = rgb.transpose(method=Image.FLIP_LEFT_RIGHT)
            elif self.flip_direction == 'h':
                rgb = rgb.transpose(method=Image.FLIP_TOP_BOTTOM)

        return rgb

    def get_nir(self, collection, img_id, do_flip):
        nir = self.loader(self.filename(collection, img_id, self.img_suffix, 'NIR', self.ftype))
        nir = nir.convert('L')
        size = nir.size

        if do_flip:
            if self.flip_direction == 'w':
                nir = nir.transpose(method=Image.FLIP_LEFT_RIGHT)
            elif self.flip_direction == 'h':
                nir = nir.transpose(method=Image.FLIP_TOP_BOTTOM)

        nir = ImageOps.colorize(nir, black='black', white='white')
        return nir

    def get_proxy_label(self, collection, img_id, proxy='semi'):
        """获取代理标签, 及其mask, 即disp为0的mask

        Args:
            collection (str): dir name
            img_id (str): image id
            proxy (str, optional): type of proxy label. Defaults to 'semi', choice=['semi', 'dense', 'mono'].
        """
        proxy_label = np.load(self.filename(collection, img_id, self.img_suffix, 'RGB', 'npy', proxy))
        return proxy_label
        
    def filename(self, collection, img_id, suffix, camera, ftype, proxy=None):
        """拼接完整文件路径"""
        assert camera in ['RGB', 'NIR', '']
        assert ftype in ['png', 'npz', 'npy', 'pfm']
        if proxy is None:
            return str(Path(self.data_path) / collection / (camera + suffix)/ (img_id + '_' + camera + suffix + '.' + ftype))
        assert proxy in ['semi', 'dense', 'mono']
        if proxy == 'semi':
            return str(Path(self.data_path) / collection  / 'SemiProxy' / 'npy' / (img_id + '_' + camera + suffix + '_semi_proxy' + '.' + ftype))
        elif proxy == 'dense':
            return str(Path(self.data_path) / collection  / 'DenseProxy' / 'npy' / (img_id + '_' + camera + suffix + '_dense_proxy' + '.' + ftype))
        elif proxy == 'mono':
            return str(Path(self.data_path) / collection / 'MonoDisp' / 'npy' / (img_id + '_' + camera + suffix + '-dpt_beit_large_512' + '.' + ftype))

if __name__ == "__main__":

    dataset = PittburghStereoDataset
    root = "/media/jiayu/My Passport1/academic/depth_estimation/datasets/RGB-NIR_stereo/tzhi/RGBNIRStereoRelease/rgbnir_stereo/data/metric3d_rgbnir_stereo"
    data_path = root + "/data"
    list_path = root + "/lists"
    train_splits = "20170222_0951"
    # train_splits = "20170221_1357,20170222_0715,20170222_1207,20170222_1638,20170223_0920,20170223_1217,20170223_1445,20170224_1022"
    test_splits = "20170222_0951,20170222_1423,20170223_1639,20170224_0742"
    height = 429
    width = 582
    ftype = 'png'
    batch_size = 2
    num_workers = 2
    pad_mode = 'constant'

    train_dataset = dataset(data_path,
                            list_path, 
                            train_splits, 
                            height, width, 
                            is_train=True, 
                            ftype=ftype, 
                            pad_mode=pad_mode,
                            no_norm=True)

    train_loader = data.DataLoader(train_dataset, 
                                   batch_size, 
                                   False, 
                                   num_workers=num_workers, 
                                   pin_memory=True, 
                                   drop_last=True)

    for idx, inputs in enumerate(train_loader):
        keys = []
        for key in inputs.keys():
            print(key)
            keys.append(key)
            print(inputs[key].shape)

        k=4
        print(len(keys))
        print(inputs[keys[k]].shape)
        if "semi_proxy_label" == keys[k]:
            inputs[keys[k]][0] = (inputs[keys[k]][0]-inputs[keys[k]][0].min()) / (inputs[keys[k]][0].max() - inputs[keys[k]][0].min())
        elif "dense_proxy_label" == keys[k]:
            inputs[keys[k]][0] = (inputs[keys[k]][0]-inputs[keys[k]][0].min()) / (inputs[keys[k]][0].max() - inputs[keys[k]][0].min())
        elif "mono_proxy_label" == keys[k]:
            inputs[keys[k]][0] = (inputs[keys[k]][0]-inputs[keys[k]][0].min()) / (inputs[keys[k]][0].max() - inputs[keys[k]][0].min())

        img = transforms.ToPILImage()(inputs[keys[k]][0])
        # img.save("./tmp.png")
        img.show()
            
        break
