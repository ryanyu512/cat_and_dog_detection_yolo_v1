'''
    Updated on 2023/03/30
    
    1. aim at defining the architechure of yolo_v1
'''

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import ZeroPadding2D, Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.layers import LeakyReLU, Softmax, Activation

CONV_MAX_CONFIG = [
    [7,   64, 2, 3], #1
    'max_pool',
    [3,  192, 1, 1], #2
    'max_pool',
    [1,  128, 1, 0], #3
    [3,  256, 1, 1], #4
    [1,  256, 1, 0], #5
    [3,  512, 1, 1], #6
    'max_pool',
    [1,  256, 1, 0], #7
    [3,  512, 1, 1], #8
    [1,  256, 1, 0], #9
    [3,  512, 1, 1], #10
    [1,  256, 1, 0], #11
    [3,  512, 1, 1], #12
    [1,  256, 1, 0], #13
    [3,  512, 1, 1], #14
    [1,  512, 1, 0], #15
    [3, 1024, 1, 1], #16
    'max_pool',
    [1,  512, 1, 0], #17
    [3, 1024, 1, 1], #18
    [1,  512, 1, 0], #19
    [3, 1024, 1, 1], #20 <= first 20 needs to train for classification firstly
    [3, 1024, 1, 1], #21
    [3, 1024, 2, 1], #22
    [3, 1024, 1, 1], #23
    [3, 1024, 1, 1]  #24
]
        
class Network:
    
    def __init__(self, class_num = 20):
        
        self.backbone = Sequential(name = 'backbone')
        self.neck = Sequential(name = 'neck')
        self.fc = Sequential(name = 'fc_layers')
        self.cls_num = class_num
        
        #define backbone
        for i in range(24):
            config = CONV_MAX_CONFIG[i]
            if CONV_MAX_CONFIG[i] != 'max_pool':
                k, ch, s, p = config
                z2d = ZeroPadding2D(padding = (p, p))
                c2d = Conv2D(filters = ch,
                             kernel_size = (k, k),
                             strides = (s, s),
                             padding = 'valid')
                bn = BatchNormalization(axis = -1)
                lr = LeakyReLU(alpha = 0.1)
                self.backbone.add(z2d)
                self.backbone.add(c2d)
                self.backbone.add(bn)
                self.backbone.add(lr)
            else:
                m2d = MaxPooling2D(pool_size = (2, 2), 
                                   strides   = (2, 2), 
                                   padding   = 'valid')
                self.backbone.add(m2d)
        
        self.ap2d = AveragePooling2D(pool_size = (2, 2), 
                                        strides   = (2, 2), 
                                        padding   = 'valid')
        self.flatten = Flatten()
        self.dense = Dense(class_num)
        self.sig   = Activation('sigmoid')
       
        #define neck
        for i in range(24, len(CONV_MAX_CONFIG)):
            config = CONV_MAX_CONFIG[i]
            '''
            k, ch, s, p = config[0], config[1], config[2], config[3]
            cb = ConvBlock(ch, k, s, p)
            '''
            k, ch, s, p = config
            z2d = ZeroPadding2D(padding = (p, p))
            c2d = Conv2D(filters = ch,
                            kernel_size = (k, k),
                            strides = (s, s),
                            padding = 'valid')
            bn = BatchNormalization(axis = -1)
            lr = LeakyReLU(alpha = 0.1)
            self.neck.add(z2d)
            self.neck.add(c2d)
            self.neck.add(bn)
            self.neck.add(lr)
        
        #define fully connected network
        '''
        1. grid cell = 7*7
        2. class = 20
        3. # of bbox = 2
        4. 49*(20 + 2*5) = 1470
        '''
        self.fc.add(Flatten())
        self.fc.add(Dense(4096))
        self.fc.add(LeakyReLU(alpha = 0.1))
        self.fc.add(Dense(1470))
    
    def YOLO_v1_pretrain(self):
        
        feed = Input((224, 224, 3))
        x = self.backbone(feed)
        x = self.ap2d(x)
        x = self.flatten(x)
        x = self.dense(x)
        out = self.sig(x)
           
        model = Model(feed, [out])  
            
        return model
    
    def YOLO_v1(self):
        
        feed = Input((448, 448, 3))
        x = self.backbone(feed)
        x = self.neck(x)
        out = self.fc(x)
        
        model = Model(feed, [out])
        
        return model