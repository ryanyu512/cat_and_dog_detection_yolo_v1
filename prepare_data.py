'''
UPDATED ON 2023/04/03

1. aims at providing custom library of data preparation of classification and detection
'''

import os
import cv2
import copy
import json
import uuid
import random
from bs4 import BeautifulSoup

def yolo2minmax(yboxes, img_w, img_h):
    '''
    args:
        yboxes: [xc, yc, w, h] in normalized yolo format 
        img_w, img_h: shape of image
        
    returns:
        boxes: [x1, y1, x2, y2] - VOC format    
    '''
    
    boxes = [None]*len(yboxes)
    for i, ybox in enumerate(yboxes):
        xc, yc, w, h = ybox
        x1 = (xc - w/2)*img_w
        y1 = (yc - h/2)*img_h
        x2 = (xc + w/2)*img_w
        y2 = (yc + h/2)*img_h
        boxes[i] = [x1, y1, x2, y2]
    
    return boxes

def minmax2yolo(boxes, img_w, img_h):
    '''
    args:
        boxes: [x1, y1, x2, y2] - VOC format
        img_w, img_h: shape of image
        
    returns:
        yboxes: [xc, yc, w, h] in normalized yolo format    
    '''
    
    yboxes = [None]*len(boxes)
    for i, box in enumerate(boxes):
        xc = (box[0] + box[2])/2./img_w
        yc = (box[1] + box[3])/2./img_h

        w = (box[2] - box[0])/img_w
        h = (box[3] - box[1])/img_h

        yboxes[i] = [xc, yc, w, h]
    
    return yboxes

def load_data_path(roots, target = 'train'):
    '''
    args:
        roots:  root of data
        target: which type of data (train or valid or test)
        
    returns:
        img_path: path of images 
        lab_path: path of labels
    '''
    
    img_path = []
    lab_path = []
    if target != 'train' and target != 'valid' and target != 'test':
        return img_path, lab_path
    
    for root in roots:
        sub_name = (root.split('/')[0]).split('_')[-1]

        if target == 'train':
            if sub_name == 'trainval':
                file_load = os.path.join(root, 'ImageSets', 'Main', 'train.txt')
            else:
                file_load = os.path.join(root, 'ImageSets', 'Main', 'test.txt')
        elif target == 'valid':
            file_load = os.path.join(root, 'ImageSets', 'Main', 'val.txt')
        elif target == 'test':
            file_load = os.path.join(root, 'ImageSets', 'Main', 'test.txt')
            
        f = open(file_load, 'r')
        lines = f.readlines()
        img_path += [os.path.join(root,  'JPEGImages', line.replace('\n','') + '.jpg') for line in lines]
        lab_path += [os.path.join(root, 'Annotations', line.replace('\n','') + '.xml') for line in lines]
        
    return img_path, lab_path 

def split_VOC_data(img_path, lab_path, cls2ind, target_root = None, is_save = True):
    
    '''
    aim:
        gather and save VOC image and label into specific folder
    args:
        img_path: path of images
        lab_path: path of labels
        cls2ind: class to one hot vector index
        target_root: the root of saved data
        is_save: flag to decide if data is saved 
    return:
        None
    '''
    
    seed = 0
    rd = random.Random()
    rd.seed(0)
    
    if target_root is None or len(img_path) != len(lab_path):
        return

    cls_num = len(list(cls2ind.keys()))
    for img_load, lab_load in zip(img_path, lab_path):

        img = cv2.imread(img_load)

        with open(lab_load, 'r') as f:
            lab = f.read()
        lab = BeautifulSoup(lab, "xml")
        img_name = lab.find_all('filename')[0].text

        lab_obj = lab.find_all('object')

        boxes   = [None]*len(lab_obj)
        objs    = [None]*len(lab_obj)
        d_one_hots = [None]*len(lab_obj)
        c_one_hot = [0]*cls_num
        for j, obj in enumerate(lab_obj):
            x1 = float(obj.find_all('xmin')[0].text)
            y1 = float(obj.find_all('ymin')[0].text)
            x2 = float(obj.find_all('xmax')[0].text)
            y2 = float(obj.find_all('ymax')[0].text)

            obj = obj.find_all('name')[0].text
            obj_ind = cls2ind[obj]
            d_one_hot = [0]*cls_num
            d_one_hot[obj_ind] = 1

            boxes[j] = [x1, y1, x2, y2]
            objs[j]  = obj
            d_one_hots[j] = d_one_hot
            c_one_hot[obj_ind] = 1

        annotation = {}
        annotation['img_name'] = img_name
        annotation['num'] = len(lab_obj)
        annotation['box'] = boxes
        annotation['obj'] = objs
        annotation['d_one_hot'] = d_one_hots
        annotation['c_one_hot'] = c_one_hot

        if is_save:
            uid = uuid.UUID(int=rd.getrandbits(128))
            img_save = os.path.join(target_root, f'{uid}' + '.jpg')
            lab_save = os.path.join(target_root, f'{uid}' + '.json')
            cv2.imwrite(img_save, img)
            with open(lab_save, 'w') as f:
                json.dump(annotation, f)
            
def get_aug_data(img_path, lab_path, transform, aug_num = 1, target_root = 'data/aug_train'):
    
    '''
    aim:
        convert VOC data format into YOLO format and other additional data augmentation
    args:
        img_path: path of images
        lab_path: path of labels
        transform: augmentation transform
        target_root: the root of saved data
    return:
        None
    '''
    
    img_path = sorted(img_path)
    lab_path = sorted(lab_path)
    
    for img_load, lab_load in zip(img_path, lab_path):

        img = cv2.cvtColor(cv2.imread(img_load), cv2.COLOR_BGR2RGB)
        img_h, img_w, _ = img.shape
        
        lab = json.load(open(lab_load, 'r'))
        yboxes = minmax2yolo(boxes = lab['box'], img_w = img_w, img_h = img_h)  
        
        uid = (img_load.split('.')[0]).split('/')[-1]
        
        if transform is None:
            aug_num = 1
            
        for i in range(aug_num):
            
            if transform is not None:
                img_cpy = copy.deepcopy(img)
                transformed = transform(image = img_cpy, 
                                        bboxes = yboxes,
                                        class_labels = lab['obj'])

                timg   = transformed['image']
                tboxes = transformed['bboxes']
                tobjs  = transformed['class_labels']
                tlab   = copy.deepcopy(lab)
                tlab['obj'] = tobjs
                tlab['box'] = tboxes      
            else:
                timg   = copy.deepcopy(img)
                tlab   = copy.deepcopy(lab)
                tlab['box'] = yboxes   
            
            img_save = os.path.join(target_root, f'{uid}-{i}' + '.jpg')
            lab_save = os.path.join(target_root, f'{uid}-{i}' + '.json')
            cv2.imwrite(img_save, cv2.cvtColor(timg, cv2.COLOR_RGB2BGR))
            
            with open(lab_save, 'w') as f:
                json.dump(tlab, f)
                