import sys
import cv2
import numpy as np
from skimage import morphology
import matplotlib.pyplot as plt

sys.path.append("C:/Users/hliu/Desktop/DL/toolbox")
import tool


def remove_small_regions(img, size):
    img = morphology.remove_small_objects(img, size)
    img = morphology.remove_small_holes(img, size)
    return img



def get_lungmask(unet, image_fp, save_fp=None, show=False):
    ''' Generate binary lung mask
    
    unet: Pre-trained lung segmentation model

    Returns:
    resized_pred: The lung mask of size 256 x 256, binary pixels 0 and 255

    Example: lung_mask = get_lung(model, "./chestX-ray.jpg")
    '''
    def preprocess(image, net_input_shape):
        image = cv2.resize(image, net_input_shape)
        image = image.astype(np.float64)
        image = np.expand_dims(image, -1)
        image = np.expand_dims(image, 0)
        image -= image.mean()
        image /= image.std()
        return image

    def postprocess(pred, threshold, rsrr):
        pred = pred > threshold
        pred = remove_small_regions(pred, rsrr)
        pred = np.float32(pred)
        pred = np.uint8(pred) * 255
        return pred

    THRESHOLD = 0.5
    COLORMAP_THRESHOLD = 0.3
    NET_INPUT_DIM = unet.layers[0].output_shape[1:3]
    REMOVE_SMALL_REGION_RATIO = 0.02*np.prod(NET_INPUT_DIM)
    
    image = tool.read_image(image_fp)
    h, w = image.shape[0], image.shape[1]
    prc_image = preprocess(image, NET_INPUT_DIM)
    pred = unet.predict(prc_image).reshape(NET_INPUT_DIM)
    pred = postprocess(pred, THRESHOLD, REMOVE_SMALL_REGION_RATIO)

    lung_mask = cv2.applyColorMap(pred, cv2.COLORMAP_JET)*COLORMAP_THRESHOLD
    orig_image = cv2.cvtColor(cv2.resize(image,(256,256)),cv2.COLOR_GRAY2RGB)
    visual = lung_mask + orig_image
    visual = tool.normalize(visual)

    if save_fp != None:
        cv2.imwrite(save_fp, visual)
        print("Successfully created lung mask at: ", save_fp)
    
    if show:
        cv2.imshow("Predicted lung mask", visual)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return pred



def get_bbox(unet, image_fp, save_fp=None, show=False):
    ''' Generate customized lung bounding box.

    unet: Pre-trained lung segmentation model

    Returns: 
    y_min, y_max, x_min, x_max, cropped bbox

    Example: y_min, y_max, x_min, x_max, bbox_img = get_bbox(model, "./chestX-ray.jpg")
    '''
    def compute_cw(cw, CW1, CW2):
            if cw <= CW1 and cw > 0:
                cw = CW1
            elif cw <= 0:
                cw = CW2
            else:
                pass
            return cw

    LUNG_COMPARE_RATIO = 3
    CENTRAL_WIDTH_RATIO = [0.1367,0.0195]
    AREA_THRESHOLD = 0.2
    BOX_MARGIN_RATIO = [0.0391, 0.0391, 0.0391, 0.0856]
    NET_INPUT_DIM = unet.layers[0].output_shape[1:3]

    HALF_DIM = NET_INPUT_DIM[0] / 2
    CW1 = CENTRAL_WIDTH_RATIO[0] * NET_INPUT_DIM[0]
    CW2 = CENTRAL_WIDTH_RATIO[1] * NET_INPUT_DIM[0]
    THRESHOLD_AREA = NET_INPUT_DIM[0] * NET_INPUT_DIM[0] * AREA_THRESHOLD

    image = tool.read_image(image_fp)
    h, w = image.shape[0], image.shape[1]
    rf = [h/NET_INPUT_DIM[0], w/NET_INPUT_DIM[0]]
    pred = get_lungmask(unet, image_fp)
    output = cv2.connectedComponentsWithStats(pred)
    num_instances = output[0] # 1 background + number of segs
    num_seg = output[0] - 1
    label = np.float32(output[1])
    stats = output[2]  
    x_min, x_max, y_min, y_max = 0, 0, 0, 0

    # ************************************************************
    # Postprocess the segmentation mask to generate a bounding box
    # ************************************************************

    # More than two segmentation regions, keep 
    # the two regions closest to the image center
    if num_seg > 2:
        scores = [0 for i in range(num_seg)]
        idx1, idx2 = 0, 0
        for i in range(num_seg):
            v_cl = stats[i+1,0] + 0.5*stats[i+1,2]    
            h_cl = stats[i+1,1] + 0.5*stats[i+1,3]
            ab1 = abs(v_cl - (NET_INPUT_DIM[0]/2))
            ab2 = abs(h_cl - (NET_INPUT_DIM[0]/2))
            scores[i] = np.sqrt(ab1*ab1+ab2*ab2)
        score1, idx1 = min((val, idx) for (idx, val) in enumerate(scores))
        tmp = list(scores)
        tmp.remove(score1)
        score2, idx2 = min((val, idx) for (idx, val) in enumerate(tmp))
        for j in range(num_seg):
            if scores[j] == score1:
                idx1 = j+1
            if scores[j] == score2:
                idx2 = j+1
        label[(label != idx1)&(label != idx2)] = 0
        stats[0,:] = stats[0,:]
        stats[1,:] = stats[idx1,:]
        stats[2,:] = stats[idx2,:]
        stats = stats[:3,:]

    # Two segmentation regions, remove the area which 
    # is smaller than LUNG_COMPARE_RATIO of the other
    if stats.shape[0] == 3: 
        area1 = stats[1,4]
        area2 = stats[2,4]
        if area1 >= area2 * LUNG_COMPARE_RATIO:
            label[label == 2] = 0
            label[label != 0] = 1
            stats = np.delete(stats,2,0)
        elif area2 >= area1 * LUNG_COMPARE_RATIO:
            label[label == 1] = 0
            label[label != 0] = 1
            stats = np.delete(stats,1,0)
        else:
            # Update predicted lung mask
            pred = np.uint8(label) * 255
            pred[pred != 0] = 255
            x, y = [], []
            for i in range(NET_INPUT_DIM[0]):
                for j in range(NET_INPUT_DIM[0]):
                    if pred[i,j] == 255:
                        y.append(i)
                        x.append(j)
            x_min, x_max = min(x), max(x)
            y_min, y_max = min(y), max(y)

    # Only one segmentation region, mirror this region
    # with respect to a short distance (trachea)
    if stats.shape[0] == 2:
        cl = stats[1,0] + 0.5 * stats[1,2] 
        # Left lung
        if cl <= HALF_DIM: 
            cw = 2 * (HALF_DIM - stats[1,0] - stats[1,2])
            cw = compute_cw(cw, CW1, CW2)
            x_min = int(stats[1,0])
            x_max = int(x_min + 2 * stats[1,2] + cw)
            y_min = int(stats[1,1])
            y_max = int(y_min + stats[1,3])
        # Right lung
        else: 
            cw = 2 * (stats[1,0] - HALF_DIM) 
            cw = compute_cw(cw, CW1, CW2)
            x_min = int(x_max - 2 * stats[1,2] - cw)
            x_max = int(stats[1,0] + stats[1,2])
            y_min = int(stats[1,1])
            y_max = int(y_min + stats[1,3])

    # If bbox area is too small, center-crop the image
    bbox_area = (x_max - x_min) * (y_max - y_min)
    if (bbox_area <= THRESHOLD_AREA) or (num_seg == 0): 
        x_min = int((15/256)*NET_INPUT_DIM[0])
        x_max = int((239/256)*NET_INPUT_DIM[0])
        y_min = int((15/256)*NET_INPUT_DIM[0])
        y_max = int((239/256)*NET_INPUT_DIM[0])
    else: # expand the box
        x_min = int(x_min - BOX_MARGIN_RATIO[0]*NET_INPUT_DIM[0])
        y_min = int(y_min - BOX_MARGIN_RATIO[1]*NET_INPUT_DIM[0])
        x_max = int(x_max + BOX_MARGIN_RATIO[2]*NET_INPUT_DIM[0])
        y_max = int(y_max + BOX_MARGIN_RATIO[3]*NET_INPUT_DIM[0])

    if x_min < 0:
        x_min = 0
    if y_min < 0:
        y_min = 0
    if x_max > (NET_INPUT_DIM[0]-1):
        x_max = (NET_INPUT_DIM[0]-1)
    if y_max > (NET_INPUT_DIM[0]-1):
        y_max = (NET_INPUT_DIM[0]-1)

    # Mapping to original dimension
    yy_min = int(y_min*rf[0])
    yy_max = int(y_max*rf[0])
    xx_min = int(x_min*rf[1])
    xx_max = int(x_max*rf[1])
    bbox_image = image[yy_min:yy_max, xx_min:xx_max]

    if save_fp!=None:
        cv2.imwrite(save_fp, bbox_image)
        print("Successfully created lung bounding box at: ", save_fp)

    if show:
        plt.figure() 
        plt.imshow(bbox_image, cmap=plt.cm.gist_gray)
        plt.show()

    return yy_min, yy_max, xx_min, xx_max, bbox_image
