""" Pneumonia detection """
import sys
import os
import cv2
import glob
import time
import pydicom
import math
import random
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from tqdm import tqdm
from imgaug import augmenters as iaa
from keras.callbacks import CSVLogger
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau

import utils
import pneu_dataset
import model as modellib
from config import Config

sys.path.append("C:/Users/hliu/Desktop/DL/toolbox")
import tool



class DetectorConfig(Config):
    """ MRCNN configuration """
    GPU_COUNT = 1
    NUM_CLASSES = 2
    NAME = 'pneumonia'
    BACKBONE = 'resnet101'
    IMAGE_RESIZE_MODE = "square"
    ORIG_SIZE = 512

    IMAGES_PER_GPU = 32 # Batch size
    NUM_EPOCHS = 30
    LEARNING_RATE = 1e-3
    
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    TRAIN_STEPS = 1067
    VAL_STEPS = 134

    TRAIN_CSV_FP = "C:/Users/hliu/Desktop/DL/dataset/Kaggle/mrcnn_set/train.csv"
    VAL_CSV_FP = "C:/Users/hliu/Desktop/DL/dataset/Kaggle/mrcnn_set/val.csv"

    # MRCNN
    MAX_GT_INSTANCES = 5
    LOSS_WEIGHTS = {"rpn_class_loss":   1.,
                    "rpn_bbox_loss":    1.,
                    "mrcnn_class_loss": 1.,
                    "mrcnn_bbox_loss":  1.,
                    "mrcnn_mask_loss":  1.}
    
    # RPN branch
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)
    TRAIN_ROIS_PER_IMAGE = 50

    # Mask branch
    MINI_MASK_SHAPE = (28,28)

    # Detection
    DETECTION_MAX_INSTANCES = 10
    DETECTION_MIN_CONFIDENCE = 0.75
    DETECTION_NMS_THRESHOLD = 0.3



def augmentation(num):
    """ Real-time image augmentation """
    return iaa.SomeOf((0,num),
                [    
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5),
                iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, 
                        translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}, 
                        rotate=(-25, 25), 
                        shear=(-10, 10)),
                iaa.CropAndPad(percent=(-0.2,0.2)),
                iaa.Add((-10, 10), per_channel=0.5),
                iaa.ContrastNormalization((0.8,1.5),per_channel=0.5),
                iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
                iaa.Emboss(alpha=(0, 1.0), strength=(0, 1.5)),
                iaa.Multiply((0.9, 1.1), per_channel=0.5),
                iaa.PerspectiveTransform((0.01, 0.1)),
                iaa.PiecewiseAffine((0.02, 0.04)),
                iaa.ElasticTransformation(alpha=130, sigma=(10,13)),
                iaa.OneOf([
                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),    
                    iaa.Dropout((0.01,0.03)),
                    iaa.SaltAndPepper((0.01,0.02)),
                    ]),
                iaa.OneOf([
                    iaa.GaussianBlur((0, 1.2)),
                    iaa.AverageBlur(k=(1, 5)),
                    iaa.MedianBlur(k=(1, 5)),
                    ]),
                ])



class InferenceConfig(DetectorConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1



class MyDetector(object):
    def __init__(self):
        self.config = DetectorConfig()
        self.model = None


    def display_config(self):
        self.config.display()


    def get_dataset(self, csv_fp):
            df = pd.read_csv(csv_fp)
            image_fps = tool.get_uniq(df["patientId"].tolist())
            ant = {image_fp: [] for image_fp in image_fps}
            for _,row in df.iterrows():
                ant[row["patientId"]].append(row)
            dataset = pneu_dataset.DetectorDataset(image_fps, ant,
                self.config.ORIG_SIZE, self.config.ORIG_SIZE)
            dataset.prepare()
            return dataset


    def train(self, log_dir):
        """ Training MRCNN 
        """
        #########################################################################
        csv_logger = CSVLogger(os.path.join(log_dir, "csv_logger.csv"))
        early_stop = EarlyStopping(monitor='val_loss', patience=10)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5)
        #########################################################################

        # Make log directory if not exist
        if not os.path.isdir(log_dir): 
            os.makedirs(log_dir)

        train_dataset = self.get_dataset(self.config.TRAIN_CSV_FP)
        val_dataset = self.get_dataset(self.config.VAL_CSV_FP)

        # Load MRCNN model
        model = modellib.MaskRCNN(mode='training', config=self.config, model_dir=log_dir)

        # Initialization
        model.load_weights(filepath=model.get_imagenet_weights(), by_name=True)
        
        # Training RPN
        model.train(train_dataset, val_dataset, 
                    learning_rate=1e-3, 
                    epochs=30, 
                    layers='heads',
                    augmentation=augmentation(7),
                    custom_callbacks=[csv_logger, early_stop, reduce_lr]
                    )
        """ fix rpn weights, train only mask heads:
        model.train(train_dataset, val_dataset, 
                    learning_rate=1e-3, 
                    epochs=30, 
                    layers='heads',
                    augmentation=augmentation(7),
                    custom_callbacks=[csv_logger, early_stop, reduce_lr]
                    )
        """

    def load_model(self, weights_fp):
        self.config =InferenceConfig()
        model = modellib.MaskRCNN(mode='inference', config=self.config, model_dir=None)
        model.load_weights(filepath=weights_fp, by_name=True)
        print("successfully loaded MRCNN model")
        self.model = model


    def predict(self, image_fp, show=False, min_conf=0.95):
        """Predict single image.
        Returns: 
        Empty list or [[score1, x1, y1, w1, h1], [score2, x2, ......]
        """
        assert self.model is not None, "Please load model"
        boxes_list = []
        min_dim = self.config.IMAGE_MIN_DIM
        max_dim = self.config.IMAGE_MAX_DIM
        mode = self.config.IMAGE_RESIZE_MODE
        image = tool.read_image(image_fp, 3)
        rf = image.shape[0]/self.config.IMAGE_SHAPE[0]
        image = tool.normalize(image)
        image, window, scale, padding, crop = utils.resize_image(image,
            min_dim=min_dim, max_dim=max_dim, mode=mode)
        pred = self.model.detect([image])[0]
        num_box = len(pred['rois'])
        assert num_box == len(pred['scores'])
        if num_box != 0:
            for i in range(num_box):
                if pred["scores"][i] > min_conf:
                    score = round(pred['scores'][i],2)
                    x = pred['rois'][i][1]
                    y = pred['rois'][i][0]
                    w = pred['rois'][i][3] - x
                    h = pred['rois'][i][2] - y
                    pneu_box = [score, x*rf, y*rf, w*rf, h*rf]
                    boxes_list.append(pneu_box)
        if show:
            if num_box == 0:
                print("No pneumonia detected")
            else:
                print(f"{len(boxes_list)} boxes are above the min_conf")
                for idx, box in enumerate(boxes_list):
                    print(f"box{idx}:{box}")
        return boxes_list


    def get_mask(self, image_fp, show=False, min_conf=0.95):
        assert self.model is not None, "Please load model"
        boxes_list = []
        min_dim = self.config.IMAGE_MIN_DIM
        max_dim = self.config.IMAGE_MAX_DIM
        mode = self.config.IMAGE_RESIZE_MODE
        image = tool.read_image(image_fp, 3)
        rf = image.shape[0]/self.config.IMAGE_SHAPE[0]
        image = tool.normalize(image)
        image, window, scale, padding, crop = utils.resize_image(image,
            min_dim=min_dim, max_dim=max_dim, mode=mode)
        pred = self.model.detect([image])[0]
        mask1 = pred["masks"][:,:,0]
        mask1 = mask1.astype(int)
        mask1 = np.uint8(mask1) * 255
        print(mask1)
        print(np.max(mask1))
        cv2.imshow("image", mask1)
        cv2.waitKey(0)
        return
        num_box = len(pred['rois'])
        assert num_box == len(pred['scores'])
        if num_box != 0:
            for i in range(num_box):
                if pred["scores"][i] > min_conf:
                    score = round(pred['scores'][i],2)
                    x = pred['rois'][i][1]
                    y = pred['rois'][i][0]
                    w = pred['rois'][i][3] - x
                    h = pred['rois'][i][2] - y
                    pneu_box = [score, x*rf, y*rf, w*rf, h*rf]
                    boxes_list.append(pneu_box)
        if show:
            if num_box == 0:
                print("No pneumonia detected")
            else:
                print(f"{len(boxes_list)} boxes are above the min_conf")
                for idx, box in enumerate(boxes_list):
                    print(f"box{idx}:{box}")
        return boxes_list


    def generate_submission(self, image_dir, save_fp, min_conf):
        """Generate submission for Kaggle competition.
        image_dir: Directory of testing images
        """     
        test_fps = tool.get_image_fps(image_dir)
        with open(save_fp, 'w') as file:
            file.write("patientId,PredictionString\n")
            for image_fp in tqdm(test_fps):
                image_id = tool.get_id(image_fp)
                line = image_id + ","
                boxes_list = self.predict2(image_fp=image_fp,
                    show=False, min_conf=min_conf)
                boxes_str = tool.get_box_str(boxes_list)
                line += boxes_str
                file.write(line+"\n")


    def visualize(self, image_fp, gt_csv=None, save_fp=None,
                  show=False, min_conf=0.95):
        """Visualize the detected boxes and groundtruth boxes, if provided.
        Note: Green: Detected
              Red:  Groundtruth
        """
        boxes_list = self.predict(image_fp, min_conf=min_conf)
        image = tool.read_image(image_fp, 3)
        for i, box in enumerate(boxes_list):
            score = box[0]
            if score > min_conf:
                x_min = int(box[1])
                y_min = int(box[2])
                w = int(box[3])
                h = int(box[4])
                x_max = x_min + w
                y_max = y_min + h
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3, 1)
                cv2.putText(image,str(score),(x_min+10, y_min+10), cv2.FONT_HERSHEY_PLAIN, 3,(255,0,0),2,cv2.LINE_AA)
                print("**************************************")
                print(f"confidence level: {score:.2f}\nbox [x, y, w, h]:{[x_min, y_min, w, h]}")

        if gt_csv is not None:
            assert os.path.exists(gt_csv)
            image_id = tool.get_id(image_fp)
            df = pd.read_csv(gt_csv)
            df_matched = df[df["patientId"]==image_id]
            assert df_matched.shape[0] != 0
            x = df_matched["x"].tolist()
            y = df_matched["y"].tolist()
            w = df_matched["width"].tolist()
            h = df_matched["height"].tolist()
            if not math.isnan(x[0]):
                for j in range(len(x)):
                    x_min, y_min, width, height = int(x[j]), int(y[j]), int(w[j]), int(h[j])
                    x_max, y_max = x_min + width, y_min + height
                    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 3, 1)
        if save_fp is not None:
            cv2.imwrite(save_fp, image)
        if show:
            plt.figure() 
            plt.imshow(image, cmap=plt.cm.gist_gray)
            plt.show()
