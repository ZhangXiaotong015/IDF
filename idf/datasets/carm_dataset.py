import numpy as np
import random
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
from concurrent.futures import ThreadPoolExecutor
from os import listdir
from os.path import join
import os
from typing import Tuple
import numpy as np
from natsort import natsorted
from idf.datasets.utils import augment_img
from idf.utils.noise import add_Gaussian_noise
from typing import Optional


class CFM_train_paired(Dataset) :
    def __init__(self, dataroot:str, patch_size:int, augmentation:bool,
                 preload:bool, parallel_preload:bool, test:bool,
                 lq_folder:str='noisy', gt_folder:str='clean') :
        super(CFM_train_paired, self).__init__()
        
        # Initialize Variables
        self.dataroot = dataroot
        self.patch_size = patch_size
        self.test = test
        self.preload = preload
        self.augmentation = augmentation
        
        # Get Dataset Instances
        self.lq_folder, self.gt_folder = lq_folder, gt_folder
        self.noisyDataset, self.cleanDataset = self.getPathList()    

        self.GT_dir = [join(self.cleanDataset[1], fn) for fn in self.cleanDataset[0]]
        self.LQ_dir = [join(self.noisyDataset[1], fn) for fn in self.noisyDataset[0]]

        if self.preload:
            if parallel_preload:
                # Preload images into RAM in parallel
                with ThreadPoolExecutor() as executor:
                    self.GT = list(executor.map(self.load_image, self.GT_dir))
                with ThreadPoolExecutor() as executor:
                    self.LQ = list(executor.map(self.load_image, self.LQ_dir))
            else:
                self.GT, self.LQ = [], []
                for dir in self.GT_dir:
                    self.GT.append(np.array(Image.open(dir)))
                for dir in self.LQ_dir:
                    self.LQ.append(np.array(Image.open(dir)))
    
    def load_image(self, img_path):
        image = np.array(Image.open(img_path))
        return image

    def __getitem__(self, index) :
        # Load Data
        if self.preload:
            noisy = self.LQ[index]
            clean = self.GT[index]
        else:
            noisy = np.array(Image.open(self.LQ_dir[index]))
            clean = np.array(Image.open(self.GT_dir[index]))

        if not self.test:
            h, w, _ = clean.shape
            rnd_h = random.randint(0, max(0, h - self.patch_size))
            rnd_w = random.randint(0, max(0, w - self.patch_size))
            clean_patch = clean[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
            noisy_patch = noisy[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
        else:
            if self.patch_size is not None:
                # perform center crop
                h, w, _ = clean.shape
                start_h = (h - self.patch_size) // 2
                start_w = (w - self.patch_size) // 2
                
                end_h = start_h + self.patch_size
                end_w = start_w + self.patch_size

                clean_patch = clean[start_h:end_h, start_w:end_w, :]
                noisy_patch = noisy[start_h:end_h, start_w:end_w, :]
            else:
                clean_patch = clean
                noisy_patch = noisy
            
        if not self.test and self.augmentation:
            mode = random.randint(0, 7)
            clean_patch = augment_img(clean_patch, mode)
            noisy_patch = augment_img(noisy_patch, mode)

        clean_patch = clean_patch.transpose(2, 0, 1).astype(np.float32) / 255.
        noisy_patch = noisy_patch.transpose(2, 0, 1).astype(np.float32) / 255.

        img_item = {}
        img_item['GT'] = clean_patch
        img_item['LQ'] = noisy_patch
        img_item['file_name'] = self.noisyDataset[0][index]
        # img_item['metadata'] = label
        
        return img_item

    def __len__(self):
        return len(self.noisyDataset[0])

    def getPathList(self) :            
        noisyPath = join(self.dataroot, self.lq_folder)
        cleanPath = join(self.dataroot, self.gt_folder)
    
        # Create List Instance for Adding Dataset Path
        noisyPathList = listdir(noisyPath)
        cleanPathList = listdir(cleanPath)
        
        # Create List Instance for Adding File Name
        noisyNameList = [imageName for imageName in noisyPathList if imageName.split(".")[-1] in ["png", "tif"]]
        cleanNameList = [imageName for imageName in cleanPathList if imageName.split(".")[-1] in ["png", "tif"]]
        
        # Sort List Instance
        noisyNameList = natsorted(noisyNameList)
        cleanNameList = natsorted(cleanNameList)
        
        return (noisyNameList, noisyPath), (cleanNameList, cleanPath)
    
class CFM_valid_paired(Dataset) :
    def __init__(self, dataroot:str, patch_size:Optional[int], augmentation:bool,
                 preload:bool, parallel_preload:bool, test:bool,
                 lq_folder:str='noisy', gt_folder:str='clean') :
        super(CFM_valid_paired, self).__init__()
        
        # Initialize Variables
        self.dataroot = dataroot
        self.patch_size = patch_size
        self.test = test
        self.preload = preload
        self.augmentation = augmentation
        
        # Get Dataset Instances
        self.lq_folder, self.gt_folder = lq_folder, gt_folder
        self.noisyDataset, self.cleanDataset = self.getPathList()    

        # self.GT_dir = [join(self.cleanDataset[1], fn) for fn in self.cleanDataset[0]]
        # self.LQ_dir = [join(self.noisyDataset[1], fn) for fn in self.noisyDataset[0]]
        self.GT_dir = self.cleanDataset
        self.LQ_dir = self.noisyDataset

        if self.preload:
            if parallel_preload:
                # Preload images into RAM in parallel
                with ThreadPoolExecutor() as executor:
                    self.GT = list(executor.map(self.load_image, self.GT_dir))
                with ThreadPoolExecutor() as executor:
                    self.LQ = list(executor.map(self.load_image, self.LQ_dir))
            else:
                self.GT, self.LQ = [], []
                for dir in self.GT_dir:
                    self.GT.append(np.array(Image.open(dir)))
                for dir in self.LQ_dir:
                    self.LQ.append(np.array(Image.open(dir)))
    
    def load_image(self, img_path):
        image = np.array(Image.open(img_path))
        return image

    def __getitem__(self, index) :
        # Load Data
        if self.preload:
            noisy = self.LQ[index]
            clean = self.GT[index]
        else:
            noisy = np.array(Image.open(self.LQ_dir[index]).convert("RGB"))
            clean = np.array(Image.open(self.GT_dir[index]).convert("RGB"))

        if not self.test:
            h, w, _ = clean.shape
            rnd_h = random.randint(0, max(0, h - self.patch_size))
            rnd_w = random.randint(0, max(0, w - self.patch_size))
            clean_patch = clean[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
            noisy_patch = noisy[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
        else:
            if self.patch_size is not None:
                # perform center crop
                h, w, _ = clean.shape
                start_h = (h - self.patch_size) // 2
                start_w = (w - self.patch_size) // 2
                
                end_h = start_h + self.patch_size
                end_w = start_w + self.patch_size

                clean_patch = clean[start_h:end_h, start_w:end_w, :]
                noisy_patch = noisy[start_h:end_h, start_w:end_w, :]
            else:
                clean_patch = clean
                noisy_patch = noisy
            
        if not self.test and self.augmentation:
            mode = random.randint(0, 7)
            clean_patch = augment_img(clean_patch, mode)
            noisy_patch = augment_img(noisy_patch, mode)

        clean_patch = clean_patch.transpose(2, 0, 1).astype(np.float32) / 255.
        noisy_patch = noisy_patch.transpose(2, 0, 1).astype(np.float32) / 255.

        img_item = {}
        img_item['GT'] = clean_patch
        img_item['LQ'] = noisy_patch
        img_item['file_name'] = self.noisyDataset[index]
        # img_item['metadata'] = label
        
        return img_item

    def __len__(self):
        return len(self.noisyDataset)

    def getPathList(self) :            
        noisyPath = join(self.dataroot, self.lq_folder)
        cleanPath = join(self.dataroot, self.gt_folder)
    
        # Create List Instance for Adding Dataset Path
        # noisyPathList = listdir(noisyPath)
        # cleanPathList = listdir(cleanPath)
        cleanPathList = []
        noisyPathList = []
        for root, dirs, files in os.walk(cleanPath):
            for f in files:
                cleanPathList.append(os.path.join(root, f))
        for root, dirs, files in os.walk(noisyPath):
            for f in files:
                noisyPathList.append(os.path.join(root, f))
        
        # Create List Instance for Adding File Name
        noisyNameList = [imageName for imageName in noisyPathList if imageName.split(".")[-1] in ["png", "tif"]]
        cleanNameList = [imageName for imageName in cleanPathList if imageName.split(".")[-1] in ["png", "tif"]]
        
        # Sort List Instance
        noisyNameList = natsorted(noisyNameList)
        cleanNameList = natsorted(cleanNameList)
        
        # return (noisyNameList, noisyPath), (cleanNameList, cleanPath)
        return noisyNameList, cleanNameList

def augment_img(img, mode=0):
    '''Kai Zhang (github: https://github.com/cszn)
    '''
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(np.rot90(img))
    elif mode == 2:
        return np.flipud(img)
    elif mode == 3:
        return np.rot90(img, k=3)
    elif mode == 4:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 5:
        return np.rot90(img)
    elif mode == 6:
        return np.rot90(img, k=2)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))

class CFM_train_unpaired(Dataset) :
    def __init__(self, dataroot:str, patch_size:int, augmentation:bool,
                 noise_level:Tuple[int, int], channel_wise_noise:bool,
                 preload:bool, parallel_preload:bool, test:bool) :
        super(CFM_train_unpaired, self).__init__()
        
        # Initialize Variables
        self.dataroot = dataroot
        self.patch_size = patch_size
        self.test = test
        self.preload = preload
        self.augmentation = augmentation

        self.noise_level = noise_level
        self.channel_wise_noise = channel_wise_noise
        
        # Get Dataset Instances
        self.cleanDataset = self.getPathList()    

        # self.GT_dir = [join(self.cleanDataset[1], fn) for fn in self.cleanDataset[0]]
        self.GT_dir = self.cleanDataset

        if self.preload:
            if parallel_preload:
                # Preload images into RAM in parallel
                with ThreadPoolExecutor() as executor:
                    self.GT = list(executor.map(self.load_image, self.GT_dir))
            else:
                self.GT= []
                for dir in self.GT_dir:
                    self.GT.append(np.array(Image.open(dir).convert('RGB')))
    
    def load_image(self, img_path):
        image = np.array(Image.open(img_path).convert('RGB'))
        return image

    def __getitem__(self, index) :
        # Load Data
        if self.preload:
            clean = self.GT[index]
        else:
            clean = np.array(Image.open(self.GT_dir[index]).convert("RGB"))
            # clean = np.array(Image.open(self.GT_dir[index]))[:,:,None]

        h, w, _ = clean.shape
        rnd_h = random.randint(0, max(0, h - self.patch_size))
        rnd_w = random.randint(0, max(0, w - self.patch_size))
        clean_patch = clean[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
        
        if not self.test and self.augmentation:
            mode = random.randint(0, 7)
            clean_patch = augment_img(clean_patch, mode)

        clean_patch = clean_patch.transpose(2, 0, 1).astype(np.float32) / 255.
        noisy_patch, noise_level = add_Gaussian_noise(clean_patch, 
                                                      noise_level1=self.noise_level[0], 
                                                      noise_level2=self.noise_level[1],
                                                      channel_wise=self.channel_wise_noise)

        img_item = {}
        img_item['GT'] = clean_patch
        img_item['LQ'] = noisy_patch
        img_item['file_name'] = self.cleanDataset[index] # self.cleanDataset[0][index]
        img_item['noise_level'] = noise_level
        
        return img_item

    def __len__(self):
        return len(self.cleanDataset)

    def getPathList(self) :
        # Get Dataset Path
        cleanPath = self.dataroot
    
        # Create List Instance for Adding Dataset Path
        # cleanPathList = os.listdir(cleanPath)
        cleanPathList = []
        for root, dirs, files in os.walk(cleanPath):
            for f in files:
                cleanPathList.append(os.path.join(root, f))

        
        # Create List Instance for Adding File Name
        cleanNameList = [imageName for imageName in cleanPathList if imageName.split('.')[-1] in ["png", "bmp", "jpg"]]
        
        # Sort List Instance
        cleanNameList = natsorted(cleanNameList)
        
        # return (cleanNameList, cleanPath)
        return cleanNameList
    
class CFM_valid_unpaired(Dataset) :
    def __init__(self, dataroot:str, patch_size:int, augmentation:bool,
                 noise_level:Tuple[int, int], channel_wise_noise:bool,
                 preload:bool, parallel_preload:bool, test:bool) :
        super(CFM_valid_unpaired, self).__init__()
        
        # Initialize Variables
        self.dataroot = dataroot
        self.patch_size = patch_size
        self.test = test
        self.preload = preload
        self.augmentation = augmentation

        self.noise_level = noise_level
        self.channel_wise_noise = channel_wise_noise
        
        # Get Dataset Instances
        self.cleanDataset = self.getPathList()    

        # self.GT_dir = [join(self.cleanDataset[1], fn) for fn in self.cleanDataset[0]]
        self.GT_dir = self.cleanDataset

        if self.preload:
            if parallel_preload:
                # Preload images into RAM in parallel
                with ThreadPoolExecutor() as executor:
                    self.GT = list(executor.map(self.load_image, self.GT_dir))
            else:
                self.GT= []
                for dir in self.GT_dir:
                    self.GT.append(np.array(Image.open(dir).convert('RGB')))
    
    def load_image(self, img_path):
        image = np.array(Image.open(img_path).convert('RGB'))
        return image

    def __getitem__(self, index) :
        # Load Data
        if self.preload:
            clean = self.GT[index]
        else:
            clean = np.array(Image.open(self.GT_dir[index]).convert("RGB"))

        h, w, _ = clean.shape
        rnd_h = random.randint(0, max(0, h - self.patch_size))
        rnd_w = random.randint(0, max(0, w - self.patch_size))
        clean_patch = clean[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
        
        if not self.test and self.augmentation:
            mode = random.randint(0, 7)
            clean_patch = augment_img(clean_patch, mode)

        clean_patch = clean_patch.transpose(2, 0, 1).astype(np.float32) / 255.
        noisy_patch, noise_level = add_Gaussian_noise(clean_patch, 
                                                      noise_level1=self.noise_level[0], 
                                                      noise_level2=self.noise_level[1],
                                                      channel_wise=self.channel_wise_noise)

        img_item = {}
        img_item['GT'] = clean_patch
        img_item['LQ'] = noisy_patch
        img_item['file_name'] = self.cleanDataset[index] # self.cleanDataset[0][index]
        img_item['noise_level'] = noise_level
        
        return img_item

    def __len__(self):
        return len(self.cleanDataset)

    def getPathList(self) :
        # Get Dataset Path
        cleanPath = self.dataroot
    
        # Create List Instance for Adding Dataset Path
        # cleanPathList = os.listdir(cleanPath)
        cleanPathList = []
        for root, dirs, files in os.walk(cleanPath):
            for f in files:
                cleanPathList.append(os.path.join(root, f))

        
        # Create List Instance for Adding File Name
        cleanNameList = [imageName for imageName in cleanPathList if imageName.split('.')[-1] in ["png", "bmp", "jpg"]]
        
        # Sort List Instance
        cleanNameList = natsorted(cleanNameList)
        
        # return (cleanNameList, cleanPath)
        return cleanNameList