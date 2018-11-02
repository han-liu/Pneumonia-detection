""" customized utility functions """
import os
import cv2
import math
import pydicom
import glob
import numpy as np
import pandas as pd
from skimage.transform import resize

############################################################
#  Utility
############################################################

def get_image_fps(image_dir):
    return glob.glob(image_dir+'/'+'*.*')


def get_extension(image_fp):
    return os.path.splitext(os.path.basename(image_fp))[1]


def get_id(image_fp):
    return os.path.splitext(os.path.basename(image_fp))[0]


def read_image(image_fp, target_depth=None):
    """ Read using pydicom or openCV 
    image_fp: Filepath of a gray image

    Returns:
    image: Image with customized depth
    """
    SUPPORTED_FORMAT = [".bmp",".jpg",".png",".dcm"]
    if not os.path.exists(image_fp):
        print(image_fp)
        input("")
    assert os.path.exists(image_fp)
    ext = get_extension(image_fp)
    assert ext in SUPPORTED_FORMAT

    if ext==".dcm":
        ds = pydicom.read_file(image_fp)
        image = ds.pixel_array
    else:
        image = cv2.imread(image_fp, 0)

    if target_depth is not None and target_depth > 1:
        image = np.stack((image,)*target_depth, -1)
    return image


def gray2rgb(image):
    """ Convert gray image to RGB image_dir
    """
    assert len(image.shape) == 2
    return np.stack((image,)*3, -1)    


def normalize(image):
    """ Normalize to 0-255, dtype: np.uint8
    """
    if np.min(image) < 0:
        image = image - np.min(image)

    if np.max(image) != 0:
        image = image / np.max(image)
        image = image * 255
        image = np.uint8(image)
    return image


def resize_image(image, target_size):
    """ Resize image using skimage

    target_size: Tuple of (width, height)
    """
    return resize(image,(target_size[1],target_size[0]),
        order=1, mode="constant", preserve_range=True)


def get_uniq(my_list): 
    """ Get the unique elements of a list 
        while preserving the order (order preserving) 
    """
    target_list = []
    [target_list.append(i) for i in my_list if not target_list.count(i)]
    return target_list


############################################################
#  Bounding boxes
############################################################

def get_box_str(boxes_list):
    boxes_str = ""
    if len(boxes_list) != 0:
        for i, box in enumerate(boxes_list):
            score = box[0]
            x = box[1]
            y = box[2]
            w = box[3]
            h = box[4]
            boxes_str += f" {score} {x} {y} {w} {h}"
    return boxes_str


def iou(box1, box2):
    """ Calculate intersection over union between two boxes 
    """
    x_min_1, y_min_1, width_1, height_1 = box1
    x_min_2, y_min_2, width_2, height_2 = box2
    assert width_1 * height_1 > 0
    assert width_2 * height_2 > 0
    x_max_1, y_max_1 = x_min_1 + width_1, y_min_1 + height_1
    x_max_2, y_max_2 = x_min_2 + width_2, y_min_2 + height_2
    area1, area2 = width_1 * height_1, width_2 * height_2
    x_min_max, x_max_min = max([x_min_1, x_min_2]), min([x_max_1, x_max_2])
    y_min_max, y_max_min = max([y_min_1, y_min_2]), min([y_max_1, y_max_2])
    if x_max_min <= x_min_max or y_max_min <= y_min_max:
        return 0
    else:
        intersect = (x_max_min-x_min_max) * (y_max_min-y_min_max)
        union = area1 + area2 - intersect
        return intersect / union
    

def ap_iou(boxes_gt, boxes_pred, scores, thresholds = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]):
    """ Average precision at differnet intersection over union (IoU) threshold 
    boxes_gt, boxes_pred: Mx4 and Nx4 numpy arrays
    """
    if boxes_gt is None and boxes_pred is None:
        return None
    elif (boxes_gt is None and boxes_pred is not None) \
    or (boxes_gt is not None and boxes_pred is None):
        return 0
    else:
        assert boxes_gt.shape[1] == 4 or boxes_pred.shape[1] == 4
        assert len(scores) == len(boxes_pred)
        # sort boxes_pred by scores in decreasing order
        boxes_pred = boxes_pred[np.argsort(scores)[::-1], :] 
        ap_total = 0
        # loop over thresholds
        for t in thresholds:
            tp, fn, fp = 0, 0, 0
            for i, box_gt in enumerate(boxes_gt):
                matched = False
                for j, box_pred in enumerate(boxes_pred):
                    miou = iou(box_gt, box_pred)
                    if miou >= t:
                        tp += 1 
                        matched = True
                        break
                if not matched:
                    fn += 1
            fp = len(boxes_pred) - tp
            precision = tp / (tp + fn + fp)
            ap_total += precision
        return ap_total / len(thresholds)


def compute_kaggle_metric(pred_csv_fp, gt_csv_fp):
    """ Given the prediction and groundtruth csvs, 
    compute the mean average precision 
    """
    df_gt = pd.read_csv(gt_csv_fp)
    df_pred = pd.read_csv(pred_csv_fp)
    assert set(df_gt["patientId"].tolist())\
     == set(df_pred["patientId"].tolist())
    image_ids = get_uniq(df_pred["patientId"].tolist())

    dic_gt = {image_id: [] for image_id in image_ids}
    dic_pred = {image_id: [] for image_id in image_ids}
    dic_score = {image_id: [] for image_id in image_ids}

    # Groundtruth boxes 
    for idx, row in df_gt.iterrows():
        image_id = row["patientId"]
        if math.isnan(row["x"]):
            dic_gt[image_id] = None
        else:
            box_gt = [row["x"], row["y"], row["width"], row["height"]]
            dic_gt[image_id].append(box_gt)

    # Detected boxes 
    for idx in range(df_pred.shape[0]):
        image_id = df_pred.loc[idx, "patientId"]
        pstr = df_pred.loc[idx, "PredictionString"]
        if type(pstr) is not str:
            dic_pred[image_id] = None 
        else:
            ps_list = pstr.split()
            assert len(ps_list)%5 == 0  
            num_boxes = int(len(ps_list)/5)
            for i in range(num_boxes):
                box_info = [float(ps_list[i*5+1]),float(ps_list[i*5+2]),
                float(ps_list[i*5+3]), float(ps_list[i*5+4])]
                score_info = ps_list[i*5]
                dic_pred[image_id].append(box_info)
                dic_score[image_id].append(score_info)
    
    # Compute mean average precision
    mean_ap, count = 0, 0
    for idx, image_id in enumerate(image_ids):
        boxes_gt, boxes_pred, scores = None, None, None
        if dic_gt[image_id] is not None:
            boxes_gt = np.array(dic_gt[image_id])
        if dic_pred[image_id] is not None:
            boxes_pred = np.array(dic_pred[image_id])
        if dic_score[image_id] is not None:
            scores = np.array(dic_score[image_id])
        ap = ap_iou(boxes_gt, boxes_pred, scores)
        if ap is not None:
            mean_ap += ap
            count += 1
    return mean_ap/count


############################################################
#  Preprocessing
############################################################

def setup(root_dir, csv_dir, split_name, class_names):
    """ Create dataset split folders, class folders, and csv directory
    """
    for x in range(len(split_name)):
        dataset_dir = os.path.join(root_dir, "dataset_"+split_name[x])
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
        for y in range(len(class_names)):
            sub_folder_dir = os.path.join(dataset_dir, class_names[y])
            if not os.path.exists(sub_folder_dir):
                os.makedirs(sub_folder_dir)
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)


def update_pid(df, suffix):
    """ Add a customized suffix to the 'patientId' 
    """
    df_update = df.copy()
    df_update["patientId"] = df["patientId"] + suffix
    return df_update


def get_flipped_dfs(df):
    """ Generate the dataframes of horizontal, 
        vertical, and diagonal flipping 

    Returns:
    df_flipped: List of flipped dataframes [df_h, df_v, df_d]
    """
    df_flipped = []
    annotations = ["_h", "_v", "_d"]
    for i in range(3):
        tmp_df = update_pid(df, annotations[i])
        for idx, row in df.iterrows():
            if i==0 and row["Target"]==1: 
                tmp_df.loc[idx, "x"] = 1024-row["x"]-row["width"]
            if i==1 and row["Target"]==1:
                tmp_df.loc[idx, "y"] = 1024-row["y"]-row["height"]
            if i==2 and row["Target"]==1:
                tmp_df.loc[idx, "x"] = 1024-row["x"]-row["width"]
                tmp_df.loc[idx, "y"] = 1024-row["y"]-row["height"]
        df_flipped.append(tmp_df)
    return df_flipped


def update_pneu_gt(lung_box, pneu_box):
    """ update penumonia groundtruth box based on the lung box
    lung_box: [x_min1, y_min1, w1, h1]
    pneu_box: [x_min2, y_min2, w2, h2]

    Returns:
    [x3, y3, w3, h3]: Re-calculated box coordinates
    """
    [x_min1, y_min1, w1, h1] = lung_box
    [x_min2, y_min2, w2, h2] = pneu_box
    iou_score = iou(lung_box, pneu_box)
    if iou_score == 0:
        return None
    x_max1 = x_min1 + w1
    y_max1 = y_min1 + h1
    x_max2 = x_min2 + w2
    y_max2 = y_min2 + h2
    if x_min2 < x_min1:
        x_min2 = x_min1
    if y_min2 < y_min1:
        y_min2 = y_min1
    if x_max2 > x_max1:
        x_max2 = x_max1
    if y_max2 > y_max1:
        y_max2 = y_max1
    return [x_min2-x_min1, y_min2-y_min1, x_max2-x_min2, y_max2-y_min2]


def scale_csv_dir(csv_dir, scale_factor):
    """ 
    Example: tool.scale_csv_folder("./csv_dataset", 0.5)
    """
    def scale_csv(csv, scale_factor):
        df = pd.read_csv(csv)
        df_out = df.copy()
        count = []
        for i, row in df.iterrows():
            if math.isnan(row["x"]):
                continue
            df_out.loc[i, "x"] = int(row["x"]*scale_factor)
            df_out.loc[i, "y"] = int(row["y"]*scale_factor)
            df_out.loc[i, "width"] = int(row["width"]*scale_factor)
            df_out.loc[i, "height"] = int(row["height"]*scale_factor)
        return df_out
    csv_fps = glob.glob(csv_dir+'/'+'*.csv')
    print(f"Found {len(csv_fps)} csv files")
    for i, csv_fp in enumerate(csv_fps):
        print("Updating ", i+1)
        scale_csv(csv_fp, scale_factor).to_csv(csv_fp, index=False)
    print("All updated!")


############################################################
#  Preprocessing
############################################################
from imgaug import augmenters as iaa

def visualize_aug(image_fp, aug, show=False, save_fp=None):
    seq = iaa.Sequential([aug])
    img_list = []
    img = read_image(image_fp)
    img_list.append(img)
    img_aug = seq.augment_images(img_list)
    if show:
        cv2.imshow("Augmented image", img_aug[0])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    if save_fp is not None:
        cv2.imwrite(save_fp, img_aug[0])


############################################################
#  Classification
############################################################

def compute_class_acc(target_csv_fp, gt_csv_fp):
    """ Calculate the classification accuracy
    target_csv_fp: filepath of csv with predicted label "Target" 
    gt_csv_fp: groundtruth "Target" csv filepath 
    """
    df_target = pd.read_csv(target_csv_fp)
    df_gt = pd.read_csv(gt_csv_fp)
    # Assume both csvs contain 'Target' and 'patientId'columns
    assert "Target" in df_target.columns and "patientId" in df_target.columns
    assert "Target" in df_gt.columns and "patientId" in df_gt.columns
    # Check if the csvs have the same group of patients
    assert set(df_target["patientId"].tolist()) == set(df_gt["patientId"].tolist())
    hit = 0
    total = df_target.shape[0]
    for idx, row in df_target.iterrows():
        if row["Target"] == df_gt.loc[idx, "Target"]:
            hit += 1
    print("classification accuracy: ", hit/total)
    return hit/total


def update_submission(submission_csv_fp, target_csv_fp, output_csv_fp):
    """ Update the MRCNN result based on the classification result 
    """
    df_submission = pd.read_csv(submission_csv_fp)
    df_target = pd.read_csv(target_csv_fp)
    df_out = df_submission.copy()
    fp, fn = 0, 0
    for idx, row in df_target.iterrows():
        pid =row["patientId"]
        if (row["Target"] == 0) and \
        (not df_submission.loc[df_submission["patientId"]==pid, "PredictionString"].isnull().values.any()):
            fp += 1
            df_out.loc[df_submission["patientId"]==pid, "PredictionString"] = ""
        if (row["Target"] == 1) and \
        (df_submission.loc[df_submission["patientId"]==pid, "PredictionString"].isnull().values.any()):
            fn +=1
    df_out.to_csv(output_csv_fp, index=False)
    print("false positive (modified): ", fp)
    print("false negative (missing):  ", fn)


def prepare_csv(image_dir):
    class_names = os.listdir(image_dir)
    num_classes = len(class_names)
    save_fp = os.path.join(image_dir, "dataset.csv")
    base_label = "0,"*num_classes
    base_label = base_label[:-1]

    with open(save_fp, "w") as f:
        first_row = "image_fp,"
        for class_name in class_names:
            first_row += class_name +","
        first_row = first_row[:-1] + "\n"
        f.write(first_row)

        for i in range(num_classes):
            class_dir = os.path.join(image_dir, class_names[i])
            image_fps = glob.glob(class_dir + "/*.*")
            ind = 2*i
            class_label = list(base_label)
            class_label[ind] = '1'
            class_label = "".join(class_label)
            
            for image_fp in image_fps:
                f.write(f"{image_fp},{class_label}\n")
    
    return class_names, save_fp


############################################################
#  CSV, DATAFRAME, TXT
############################################################

def get_csv_from_folder(folder_dir, class_name, label):
    """ Generate csv from a image folder

    Example: get_csv_from_folder("./pneumonia", "Target", "1")
    """
    save_dir = folder_dir[:-len(os.path.basename(folder_dir))-1]
    save_fp = os.path.join(save_dir, os.path.basename(folder_dir)+".csv")

    with open(save_fp,"w") as f:
        f.write("patientId,"+class_name+"\n")
        image_fps = glob.glob(folder_dir + "/*.*")
        assert len(image_fps) != 0
        for image_fp in image_fps:
            f.write(f"{image_fp},{label}\n")
    print("save csv at: ", save_fp)


def combine_csv(csv_dir):
    """
    Example: combine_csv("./train")
    """
    df_out = pd.DataFrame()
    csv_fps = glob.glob(csv_dir+"/*.csv")
    assert len(csv_fps) != 0
    for csv_fp in csv_fps:
        df = pd.read_csv(csv_fp)
        df_out = df_out.append(df)
    save_fp = os.path.join(csv_dir, "combined.csv")
    df_out.to_csv(save_fp, index=False)
    print("save merged csv at: ", save_fp)
