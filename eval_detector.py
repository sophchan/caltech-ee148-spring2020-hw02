import os
import json
import numpy as np
import matplotlib.pyplot as plt

def compute_iou(box_1, box_2):
    '''
    This function takes a pair of bounding boxes and returns intersection-over-
    union (IoU) of two bounding boxes.
    '''
    # [x1, y1, x2, y2]
    # intersection coordinates
    x1 = max(box_1[0], box_2[0])
    y1 = max(box_1[1], box_2[1])
    x2 = min(box_1[2], box_2[2])
    y2 = min(box_1[3], box_2[3])

    if x2 < x1 or y2 < y1:
        iou = 0
    else: 
        intersect = (y2-y1)*(x2-x1)
        box_1_total = (box_1[2]-box_1[0]+1)*(box_1[3]-box_1[1])
        box_2_total = (box_2[2]-box_2[0]+1)*(box_2[3]-box_2[1])
        iou = intersect/(box_1_total + box_2_total - intersect)
    #intersect = (min(box_1[2], box_2[2]) - max(box_1[0], box_2[0])+1) * \
    #    ( - max(box_1[1], box_2[1])+1)
    
    #print(intersect, box_1_total, box_2_total, iou)
    
    assert (iou >= 0) and (iou <= 1.0)

    return iou


def compute_counts(preds, gts, iou_thr=0.5, conf_thr=0.5):
    '''
    This function takes a pair of dictionaries (with our JSON format; see ex.) 
    corresponding to predicted and ground truth bounding boxes for a collection
    of images and returns the number of true positives, false positives, and
    false negatives. 
    <preds> is a dictionary containing predicted bounding boxes and confidence
    scores for a collection of images.
    <gts> is a dictionary containing ground truth bounding boxes for a
    collection of images.
    '''
    TP = 0
    FP = 0
    FN = 0

    '''
    BEGIN YOUR CODE
    '''
    for pred_file, pred in preds.items():
        gt = gts[pred_file]
        if len(pred)>0: 
            for k in range(len(pred)):
                below_conf = []
                if pred[k][-1] > conf_thr:
                    FP += 1 
                else: 
                    below_conf.append(k)
            for index in below_conf: 
                pred.pop(index)
            for i in range(len(gt)):
                gt_conf = 0
                gt_pred_id = None
                for j in range(len(pred)):
                    iou = compute_iou(pred[j][:4], gt[i])
                    if iou > iou_thr and pred[j][-1] > conf_thr:
                        if pred[j][-1] > gt_conf:
                            gt_conf = pred[j][-1]
                            gt_pred_id = j
                if gt_pred_id is None: 
                    FN += 1
                else:
                    TP += 1
                    pred.pop(gt_pred_id)
    FP -= TP


    '''
    END YOUR CODE
    '''

    return TP, FP, FN

def plot_PR(preds, gts):
    n = 100
    precision_arr = [] #np.zeros(n)
    recall_arr = [] #np.zeros(n)
    for k in range(n-1, 0, -1):
        TP, FP, FN = compute_counts(preds, gts, iou_thr=0.10, conf_thr=k/100)
        precision = TP/(TP+FP) #if TP+FP!=0 else 0
        recall = TP/(TP+FN) #if TP+FN!=0 else 1
        print(k/100, TP, FP, FN, precision, recall)
        precision_arr.append(precision)
        recall_arr.append(recall)
        #precision_arr[k] = precision
        #recall_arr[k] = recall
    print(precision_arr, '\n', recall_arr)
    plt.scatter(precision_arr, recall_arr)
    plt.axis('equal')
    plt.show()

# set a path for predictions and annotations:
preds_path = '../data/hw02_preds'
gts_path = '../data/hw02_annotations'

# load splits:
split_path = '../data/hw02_splits'
file_names_train = np.load(os.path.join(split_path,'file_names_train.npy'))
file_names_test = np.load(os.path.join(split_path,'file_names_test.npy'))

# Set this parameter to True when you're done with algorithm development:
done_tweaking = False

'''
Load training data. 
''' #preds_path, preds_train_set
with open(os.path.join(preds_path,'preds_train_final.json'),'r') as f:
    preds_train = json.load(f)
    
with open(os.path.join(gts_path, 'annotations_train.json'),'r') as f:
    gts_train = json.load(f)
#box_1 = [0, 0, 4, 4]
#box_2 = [1, 1, 5, 5]
#compute_iou(box_1, box_2)

#preds_train = {'1': [[0, 0, 4, 4, 0.9]]}
#gts_train = {'1': [[1, 1, 5, 5]]}

plot_PR(preds_train, gts_train)

if done_tweaking:
    
    '''
    Load test data.
    '''
    
    with open(os.path.join(preds_path,'preds_test.json'),'r') as f:
        preds_test = json.load(f)
        
    with open(os.path.join(gts_path, 'annotations_test.json'),'r') as f:
        gts_test = json.load(f)


# For a fixed IoU threshold, vary the confidence thresholds.
# The code below gives an example on the training set for one IoU threshold. 

'''
confidence_thrs = np.sort(np.array([preds_train[fname][4] for fname in preds_train],dtype=float)) # using (ascending) list of confidence scores as thresholds
tp_train = np.zeros(len(confidence_thrs))
fp_train = np.zeros(len(confidence_thrs))
fn_train = np.zeros(len(confidence_thrs))
for i, conf_thr in enumerate(confidence_thrs):
    tp_train[i], fp_train[i], fn_train[i] = compute_counts(preds_train, gts_train, iou_thr=0.5, conf_thr=conf_thr)
'''
# Plot training set PR curves

if done_tweaking:
    print('Code for plotting test set PR curves.')
