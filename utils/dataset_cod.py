import os
import random
import numpy as np
from PIL import Image, ImageEnhance
from typing import List, Union

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from .simple_tokenizer import SimpleTokenizer as _Tokenizer

def cv_random_flip(img, label, fix):
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
        fix = fix.transpose(Image.FLIP_LEFT_RIGHT)
    return img, label, fix


def randomCrop(image, label, fix):
    border = 30
    image_width = image.size[0]
    image_height = image.size[1]
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    return image.crop(random_region), label.crop(random_region), fix.crop(random_region)


def randomRotation(image, label, fix):
    mode = Image.BICUBIC
    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        image = image.rotate(random_angle, mode)
        label = label.rotate(random_angle, mode)
        fix = fix.rotate(random_angle, mode)
    return image, label, fix


def colorEnhance(image):
    bright_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity = random.randint(0, 20) / 10.0
    image = ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity = random.randint(0, 30) / 10.0
    image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image


def randomGaussian(image, mean=0.1, sigma=0.35):
    def gaussianNoisy(im, mean=mean, sigma=sigma):
        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        return im

    img = np.asarray(image)
    width, height = img.shape
    img = gaussianNoisy(img[:].flatten(), mean, sigma)
    img = img.reshape([width, height])
    return Image.fromarray(np.uint8(img))


def randomPeper(img):
    img = np.array(img)
    noiseNum = int(0.0015 * img.shape[0] * img.shape[1])
    for i in range(noiseNum):

        randX = random.randint(0, img.shape[0] - 1)

        randY = random.randint(0, img.shape[1] - 1)

        if random.randint(0, 1) == 0:
            img[randX, randY] = 0
        else:
            img[randX, randY] = 255
    return Image.fromarray(img)

_tokenizer = _Tokenizer()
def tokenize(texts: Union[str, List[str]],
             context_length: int = 77,
             truncate: bool = False) -> torch.LongTensor:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token]
                  for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(
                    f"Input {texts[i]} is too long for context length {context_length}"
                )
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result


# dataset for training
class CamObjDataset(data.Dataset):
    def __init__(self, image_root, gt_root, fix_root, overall_desc_root, camo_desc_root, attri_root, trainsize, word_length):
        self.trainsize = trainsize
        self.word_length = word_length
        # get filenames
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')
                       or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                    or f.endswith('.png')]
        self.fix = [fix_root + f for f in os.listdir(fix_root) if f.endswith('.jpg')
                      or f.endswith('.png')]
        self.overall_desc = [overall_desc_root + f for f in os.listdir(overall_desc_root) if f.endswith('.txt')]
        self.camo_desc = [camo_desc_root + f for f in os.listdir(camo_desc_root) if f.endswith('.txt')]
        self.attri = [attri_root + f for f in os.listdir(attri_root) if f.endswith('.txt')]
        

        # sorted files
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.fix = sorted(self.fix)
        self.overall_desc = sorted(self.overall_desc)
        self.camo_desc = sorted(self.camo_desc)
        self.attri = sorted(self.attri)

        # filter mathcing degrees of files
        self.filter_files()

        # transforms
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        # get size of dataset
        self.size = len(self.images)
        print('>>> trainig/validing with {} samples'.format(self.size))

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        fix = self.gray_loader(self.fix[index])
        with open(self.overall_desc[index], 'r') as file:
                overall_desc = file.read()
        with open(self.camo_desc[index], 'r') as file:
                camo_desc = file.read()
        with open(self.attri[index], 'r') as file:
                attrs = file.readlines()
                for i in range(len(attrs)):
                    attrs[i] = float(attrs[i])

        # data augumentation
        image, gt, fix = cv_random_flip(image, gt, fix)
        image, gt, fix = randomCrop(image, gt, fix)
        image, gt, fix = randomRotation(image, gt, fix)

        image = colorEnhance(image)
        gt = randomPeper(gt)
        fix = randomPeper(fix)
        
        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        fix = self.gt_transform(fix)

        overall_desc_vec = tokenize(overall_desc, self.word_length, True).squeeze(0)
        camo_desc_vec = tokenize(camo_desc, self.word_length, True).squeeze(0)
        attrs = torch.tensor(attrs)

        return image, gt, fix, overall_desc_vec, camo_desc_vec, attrs

    def filter_files(self):
        assert all(len(lst) == len(self.images) for lst in [self.gts, self.fix, self.overall_desc])
        images, gts, fixs, desc = [], [], [], []
        for img_pth, gt_pth, fix_pth, desc_pth in zip(self.images, self.gts, self.fix, self.overall_desc):
            img = Image.open(img_pth)
            gt = Image.open(gt_pth)
            fix = Image.open(fix_pth)
            
            if img.size == gt.size == fix.size:
                images.append(img_pth)
                gts.append(gt_pth)
                fixs.append(fix_pth)
        self.images = images
        self.gts = gts
        self.fixs = fixs

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
    
    def gray_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size


# dataloader for training
def get_loader(image_root, gt_root, edge_root, batchsize, trainsize,
               shuffle=True, num_workers=12, pin_memory=True):
    dataset = CamObjDataset(image_root, gt_root, edge_root, trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader


# test dataset and loader
class TestDataset(data.Dataset):
    def __init__(self, image_root, gt_root, testsize, word_length):
        self.testsize = testsize
        self.word_length = word_length
        # get filenames
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')
                       or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                    or f.endswith('.png')]

        # sorted files
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)


        # filter mathcing degrees of files
        self.filter_files()

        # transforms
        self.img_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor()])
        # get size of dataset
        self.size = len(self.images)
        print('>>> validing with {} samples'.format(self.size))

    def __getitem__(self, index):
        # read assest/gts/fix/desc
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])


        # save img shape
        shape = gt.size
        
        # data augumentation
        # image, gt, _ = cv_random_flip(image, gt, gt)
        # image, gt, _ = randomCrop(image, gt, gt)
        # image, gt, _ = randomRotation(image, gt, gt)

        # image = colorEnhance(image)
        # gt = randomPeper(gt)
        
        image = self.img_transform(image)
        gt = self.gt_transform(gt)

        # word_vec = tokenize(desc, self.word_length, True).squeeze(0)

        name = self.images[index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'

        return image, gt, name, shape

    def filter_files(self):
        assert len(self.gts) == len(self.images) 
        images, gts= [], []
        for img_pth, gt_pth in zip(self.images, self.gts):
            img = Image.open(img_pth)
            gt = Image.open(gt_pth)
            
            if img.size == gt.size:
                images.append(img_pth)
                gts.append(gt_pth)
        self.images = images
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
    
    def gray_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size

