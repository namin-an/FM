import datetime
date = datetime.datetime.now()
print(f'Today is Happy{date: %A, %d, %m, %Y}.', '\n')
import os
import gc
import time
import random
from random import sample
import itertools
import argparse

from PIL import Image, ImageOps
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt 
import pandas as pd
from pandas import DataFrame
import json # instead of "Pickle" and "Joblib"
import joblib
import gzip
import collections
import cv2 as cv
import seaborn as sns
import sklearn
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.svm import SVC, LinearSVC
#from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc, roc_auc_score
import skimage
from skimage import color, io, viewer
from skimage.color import label2rgb
import tensorflow as tf
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, MultiMarginLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout, AdaptiveAvgPool2d
from torch.optim import Adam, SGD

print(torch.__version__)
print(torch.cuda.get_device_name(0)) # my GPU
print(torch.cuda.is_available()) 
print(torch.version.cuda)


def loadData(com_list_n, which_data_mode):

    if which_data_mode == 'train':
        path = train_path
    elif which_data_mode == 'test':
        path = test_path

    # Note: 학습 데이터 중 배치 사이즈만큼의 데이터를 고려하여 모델의 weight를 갱신함
    # img_features = lis|t()
    file_path_list, label_list = list(), list()
    start_time = time.time()

    if args.finetune == 'ft' and which_data_mode == 'train':
        tot_num_trains = len(os.listdir(os.path.join(path, str(com_list_n[0]))))
        zeros_and_ones = [0]*500 + [1]*500 + [2]*(tot_num_trains-1000) # 아예 처음부터 지정
        random.seed(random)
        rand_indexes = random.shuffle(zeros_and_ones)
        ft_path = 'E:\\ANNA_INTERN\\Middle_Resolution_137_unzipped_parcropped_128_removed_train_finetune'
        
        filePaths = []
        for i in range(1000): 
            ran_val = zeros_and_ones[i]
            if ran_val == 0: # high-resolution
                for face in com_list_n:
                    one_folder = os.path.join(path, str(face))
                    file_name = os.listdir(one_folder)[i] # same index i for all files of faces
                    
                    sl_split = file_name.split('L')
                    file_name2 = sl_split[-1]
                    e_split = file_name2.split('E')
                    file_name3 = e_split[-1]
                    c_split = file_name3.split('C')
                    file_name4 = c_split[-1]
                    dot_split = file_name4.split('.')
                    
                    if (str(sl_split[0][-1]) not in accessory_list) or (str(e_split[0]) not in light_list) or (str(c_split[0][-1]) not in expression_list) or (str(dot_split[0]) not in camera_list):                       
                        file_path_list.append(os.path.join(one_folder, file_name))
                        label_list.append(str(face))

            elif ran_val == 1: # args.finetune
                for face in com_list_n:
                    one_folder = os.path.join(ft_path, str(face))
                    file_name = os.listdir(one_folder)[i]
                    
                    sl_split = file_name.split('L')
                    file_name2 = sl_split[-1]
                    e_split = file_name2.split('E')
                    file_name3 = e_split[-1]
                    c_split = file_name3.split('C')
                    file_name4 = c_split[-1]
                    dot_split = file_name4.split('.')
                    
                    if (str(sl_split[0][-1]) not in accessory_list) or (str(e_split[0]) not in light_list) or (str(c_split[0][-1]) not in expression_list) or (str(dot_split[0]) not in camera_list):                       
                        file_path_list.append(os.path.join(one_folder, file_name))
                        label_list.append(str(face))
                
    else:
        folderPaths = [os.path.join(path, str(folder_name)) for folder_name in com_list_n]
        for folderPath in folderPaths: # 모든 사람 폴더에 대하여
            
            folder_name = os.path.basename(folderPath) # 실제 사람 번호 (com_list_n[i], i=0,1,...,r)
            filePaths = [os.path.join(folderPath, file_name) for file_name in os.listdir(folderPath)]

            if which_data_mode == 'train':
                # print(folder_name)
                # print(folder_name, len(filePaths))
                # assert len(filePaths) == 4872 # 한 사람당 준비된 total training 이미지 수 (par 고려)
                # if model_type == 'PCALR' or model_type == 'PCASVC':
                random.seed(seed)
                train_int_list = random.sample(range(len(filePaths)), 1000) # ~ 1K images from over 4,000 images (same for all ppl (folders))
                # elif model_type == 'CNN_LR' or model_type == 'CNN_SVC':
                #     train_int_list = random.sample(range(len(filePaths)), len(filePaths)) # 딥러닝할 때는 최대한의 데이터 사용

            for i, filePath in enumerate(filePaths): # 한 사람의 모든 파일에 대하여
                # _, _, _, folder_name, _ = filePath.split('\\')
                file_name = os.path.basename(filePath)
            
                sl_split = file_name.split('L')
                file_name2 = sl_split[-1]
                e_split = file_name2.split('E')
                file_name3 = e_split[-1]
                c_split = file_name3.split('C')
                file_name4 = c_split[-1]
                dot_split = file_name4.split('.')

                if which_data_mode == 'train': 
                    if i in train_int_list: # 1) 위에서 4,872장에서 임의로 고른 1000장에 대해서만
                        # if model_type == 'CNN_LR' or model_type == 'CNN_SVC':
                        #     # train image 뽑기
                        #     if (str(sl_split[0][-1]) in tr_accessory_list) and (str(e_split[0]) in tr_light_list) and (str(c_split[0][-1]) in tr_expression_list) and (str(dot_split[0]) in tr_camera_list):
                        #         # test image 걸러내기
                        #         if (str(sl_split[0][-1]) not in accessory_list) or (str(e_split[0]) not in light_list) or (str(c_split[0][-1]) not in expression_list) or (str(dot_split[0]) not in camera_list):                       
                        #             file_path_list.append(filePath)
                        #             label_list.append(folder_name)
                        # elif model_type == 'PCALR' or model_type == 'PCASVC':
                        #     # test image 걸러내기
                        # 2) test image의 parameter과 겹치지 않게
                        if (str(sl_split[0][-1]) not in accessory_list) or (str(e_split[0]) not in light_list) or (str(c_split[0][-1]) not in expression_list) or (str(dot_split[0]) not in camera_list):                       
                            # image = Image.open(filePath)
                            # image = ImageOps.grayscale(image)
                            # bimage = image.tobytes() 
                            # img = load_img(filePath, target_size=(150, 150, 3), color_mode='grayscale')
                            # img = img_to_array(img)
                            # img = np.ravel(img)
                            # img_features.append(img)
                            file_path_list.append(filePath)
                            label_list.append(folder_name)

                elif which_data_mode == 'test':
                    if (str(sl_split[0][-1]) in accessory_list) and (str(e_split[0]) in light_list) and (str(c_split[0][-1]) in expression_list) and (str(dot_split[0]) in camera_list):
                        file_path_list.append(filePath)
                        label_list.append(folder_name)

    if which_data_mode == 'train': 
        assert len(file_path_list) <= 1000*r # 한 사람당 training 이미지 수 * 모든 사람 수
    elif which_data_mode == 'test':
        if test_type == 'normal':
            assert len(file_path_list) == 9*r # 한 사람당 test 이미지 수 * 모든 사람 수


    print("Files found in {:.2f} sec.".format((time.time()-start_time))) # to 2 decimal place

  
    #if which_data_mode == 'test':
    #    X = np.array([np.array(Image.open(filePath).convert('L')) for filePath in file_path_list]) # PIL.Image
    #elif which_data_mode == 'train':
    #    X = np.array([cv.resize(np.array(Image.open(filePath).convert('L')), (256, 256), interpolation=cv.INTER_CUBIC) for filePath in file_path_list]) # PIL.Image
    if args.model_type2 == 'CNN_ResNet' or args.model_type2 == 'CNN_AlexNet': # 이전거
        X = np.array([np.array(Image.open(filePath).convert('RGB')) for filePath in file_path_list])
    else:
        X = np.array([np.array(Image.open(filePath).convert('L')) for filePath in file_path_list])
    y = np.array(label_list)

    print("Data shape: ", X.shape, y.shape)

    X, y = X.astype(np.uint8), y.astype(np.float64) # unsigned positive integers (0~255(2^8만큼 표현 가능), 0~4,294,967,295(2^32만큼 표현 가능))
    print("Data type: ", type(X[0][0][0]))

    old_uniq_labels = list(set(y))
    key_list = list(range(r))
    dictionary = dict(zip(old_uniq_labels, key_list))
    #for idx in range(r): # 나중에 torch label 쓸 때 필요
    #    temp_labels = [idx if i==old_uniq_labels[idx] else i for i in y]
    #    y = temp_labels
    for i, face_num in enumerate(y):
        y[i] = dictionary[face_num]
    new_uniq_labels = list(set(y))
    assert len(old_uniq_labels) == r
    assert len(new_uniq_labels) == r
    print(f'Old unique labels: {old_uniq_labels}')
    print(f'New unique labels: {new_uniq_labels}')
    y = np.array(y)

    return X, y, file_path_list, old_uniq_labels, new_uniq_labels


def visualizeData(Xtrain, ytrain, Xtest, ytest):
    random_idx = random.randint(0, len(Xtrain))
    plt.imshow(Xtrain[random_idx], cmap='gray')
    plt.title(f"Training example #{random_idx} (Class: {ytrain[random_idx]})")
    #plt.axis('off')
    plt.show()

    random_idx = random.randint(0, len(Xtest))
    plt.imshow(Xtest[random_idx], cmap='gray')
    plt.title(f"Testing example #{random_idx} (Class: {ytest[random_idx]})")
    #plt.axis('off')
    plt.show()