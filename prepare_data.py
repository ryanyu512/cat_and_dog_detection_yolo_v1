import os
import cv2
import json
import uuid
import random
from marco import *
from bs4 import BeautifulSoup

def load_data_path(roots, target = 'train'):
    
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

def split_data(img_path, lab_path, target_root = None):
    
    seed = 0
    rd = random.Random()
    rd.seed(0)
    
    if target_root is None or len(img_path) != len(lab_path):
        return

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
        c_one_hot = [0]*20
        for j, obj in enumerate(lab_obj):
            x1 = float(obj.find_all('xmin')[0].text)
            y1 = float(obj.find_all('ymin')[0].text)
            x2 = float(obj.find_all('xmax')[0].text)
            y2 = float(obj.find_all('ymax')[0].text)

            obj = obj.find_all('name')[0].text
            obj_ind = CLS2IND[obj]
            d_one_hot = [0]*20
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

        uid = uuid.UUID(int=rd.getrandbits(128))
        img_save = os.path.join(target_root, f'{uid}' + '.jpg')
        lab_save = os.path.join(target_root, f'{uid}' + '.json')
        cv2.imwrite(img_save, img)
        with open(lab_save, 'w') as f:
            json.dump(annotation, f)