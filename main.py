from my_detector import MyDetector


import sys
sys.path.append("C:/Users/hliu/Desktop/DL/toolbox")
import tool
import lung_seg
import glob
import pandas as pd
sys.path.append("C:/Users/hliu/Desktop/DL/models/classification")



if __name__ == "__main__":

    weight_fp = "C:/Users/hliu/Desktop/tmp/final/mask_rcnn_pneumonia_0019_1.1470_lei.h5"
    image_fp = "C:/Users/hliu/Desktop/DL/dataset/Kaggle/stage_1_test_images/000db696-cf54-4385-b10b-6b16fbb3f985.dcm"


    my_detector = MyDetector()
    my_detector.load_model(weight_fp)
    my_detector.visualize(image_fp, show=True, min_conf=0.95)