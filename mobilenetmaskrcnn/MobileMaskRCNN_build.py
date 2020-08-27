#!/usr/bin/env python
# coding: utf-8

# In[25]:


import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

from mmrcnn.config import Config
from mmrcnn import model as modellib
from mmrcnn import visualize
import mmrcnn
from mmrcnn.utils import Dataset
from mmrcnn.model import MaskRCNN
import numpy as np
from numpy import zeros
from numpy import asarray
import colorsys
import argparse
import imutils
import random
import cv2
import os
import time
import sys
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from keras.models import load_model
import skimage.draw
import json
import matplotlib
from os import listdir
from xml.etree import ElementTree
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True




#root path
ROOT_DIR = os.path.abspath("C:\\Users\\ものづくり改革室\\MobileMaskRCNN")

#mmrcnn
import mmrcnn.model as modellib
import mmrcnn
from mmrcnn.utils import Dataset
from mmrcnn.config import Config

get_ipython().run_line_magic('matplotlib', 'inline')

#logs and models
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

#images to run detect on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")


# In[26]:


class mobileconfig(Config):
    BACKBONE = "mobilenetv1"
    NAME = "mobile_config"
    IMAGE_MAX_DIM = 1024
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1+1
    STEPS_PER_EPOCH = 348
    LEARNING_RATE=5e-5
    DETECTION_MIN_CONFIDENCE = 0.9
    MAX_GT_INSTANCES=10
    IMAGE_SHAPE = 1024
    
config = mobileconfig()
config.display()


# In[27]:


model = modellib.MaskRCNN(mode = 'training', model_dir = MODEL_DIR, config = config)


# In[28]:


def get_ax(rows=1, cols=1, size=8):
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax


# In[31]:


class MobileDataset(Dataset):
    def load_burr(self, dataset_dir, subset):
        self.add_class("defects", 1, "burr")
        assert subset in ["train", "val"]
        if subset is "train":
            dataset_dir = os.path.join(dataset_dir, subset)
            
            annotations = json.load(open(os.path.join(dataset_dir, "via_region_burr.json")))
            annotations = list(annotations.values())
            annotations = [a for a in annotations if a["regions"]]
            
            for a in annotations:
                polygons = [r["shape_attributes"] for r in a["regions"]]
                
                image_path = os.path.join(dataset_dir, a["filename"])
                image = skimage.io.imread(image_path)
                height, width = image.shape[:2]
                
                self.add_image(
                    "defects",
                    image_id = a["filename"],
                    path = image_path,
                    width = width,
                    height = height,
                    polygons = polygons
                )
        else:
            dataset_dir = os.path.join(dataset_dir, subset)
            
            annotations = json.load(open(os.path.join(dataset_dir, "via_region_burr.json")))
            annotations = list(annotations.values())
            annotations = [a for a in annotations if a["regions"]]
            
            for a in annotations:
                polygons = [r["shape_attributes"] for r in a["regions"]]
                
                image_path = os.path.join(dataset_dir, a["filename"])
                image = skimage.io.imread(image_path)
                height, width = image.shape[:2]
                
                self.add_image(
                    "defects",
                    image_id = a["filename"],
                    path = image_path,
                    width = width,
                    height = height,
                    polygons = polygons
                )
#         else:
#             dataset_dir = os.path.join(dataset_dir, subset)
#             print(dataset_dir)
#             filenames = os.listdir(dataset_dir)
#             filenames = filenames[:-1]
#             for f in filenames:
#                 image_path = os.path.join(dataset_dir, f)
#                 print(image_path)
#                 image = skimage.io.imread(image_path)
#                 height, width = image.shape[:2]
                
#                 self.add_image(
#                     "defects",
#                     image_id = f,
#                     path = image_path,
#                     width = width,
#                     height = height
#                     )
                    
    def load_mask(self, image_id):
        image_info = self.image_info[image_id]
        if image_info["source"] != "defects":
            return super(self.__class__, self).load_mask(image_id)
        
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])], dtype = np.uint8)
        
        for i, p in enumerate(info["polygons"]):
            rr, cc = skimage.draw.polygon(p["all_points_y"], p["all_points_x"])
            mask[rr, cc, i] = 1
        
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype = np.int32)
    
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        if info["source"] == "defects":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)
            

            
def train(model):
    dataset_train = burr_dataset()
    dataset_train.load_burr(args.dataset, "train")
    dataset_train.prepare()
    
    dataset_val = burr_dataset()
    dataset_val.load_burr(args.dataset, "val")
    dataset_val.prepare()
    
    print("training network heads")
    model.train(dataset_train, 
                dataset_val,
                learning_rate = config.LEARNING_RATE,
                epochs = 50,
                layers = "heads"
               )


# In[ ]:




