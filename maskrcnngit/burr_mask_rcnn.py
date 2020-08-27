from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn import visualize
import mrcnn
from mrcnn.utils import Dataset
from mrcnn.model import MaskRCNN
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

ROOT_DIR = os.path.abspath(".\\")
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs_burr")

class BurrConfig(Config):
    # give the configuration a recognizable name
    NAME = "Burr_config"
 
    # set the number of GPUs to use along with the number of images
    # per GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
 
    # number of classes (we would normally add +1 for the background)
     # kangaroo + BG
    NUM_CLASSES = 1+1
   
    # Number of training steps per epoch
    STEPS_PER_EPOCH = 348
    
    # Learning rate
    LEARNING_RATE=5e-5
    
    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9
    
    # setting Max ground truth instances
    MAX_GT_INSTANCES=10
   
   
config = BurrConfig()
config.display()

class burr_dataset(Dataset):
    def load_burr(self, dataset_dir, subset):
        self.add_class("defects", 1, "burr")
        assert subset in ["train", "val", "val_new"]
        if subset is "train" or "val":
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
            print(dataset_dir)
            filenames = os.listdir(dataset_dir)
            filenames = filenames[:-1]
            for f in filenames:
                image_path = os.path.join(dataset_dir, f)
                print(image_path)
                image = skimage.io.imread(image_path)
                height, width = image.shape[:2]
                
                self.add_image(
                    "defects",
                    image_id = f,
                    path = image_path,
                    width = width,
                    height = height
                    )
            
                    
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

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect kangaroos.')
    
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/kangaroo/dataset/",
                        help='Directory of the Kangaroo dataset')
    
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    
    args = parser.parse_args()
    
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
        
    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = BurrConfig()
        
        
        
        ########################################################
    else:
        class InferenceConfig(BurrConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image,
                                video_path=args.video)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))
