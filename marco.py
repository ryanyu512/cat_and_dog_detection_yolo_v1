'''
    Updated on 2023/04/03
    
    1. aim at defining the architechure of yolo_v1
'''

CLS2IND = {
    'person': 0,
    'bird': 1,
    'cat': 2,
    'cow': 3,
    'dog': 4,
    'horse': 5,
    'sheep': 6,
    'aeroplane': 7,
    'bicycle': 8,
    'boat': 9,
    'bus': 10,
    'car': 11,
    'motorbike': 12,
    'train': 13,
    'bottle': 14,
    'chair': 15,
    'diningtable': 16,
    'pottedplant': 17,
    'sofa': 18,
    'tvmonitor': 19
}

#Augmentation parameters
#probability 
AUG_P = 0.5
#number of augmented data per image
AUG_NUM = 5
#rotation angle limit
AUG_ANG_LIM = 45