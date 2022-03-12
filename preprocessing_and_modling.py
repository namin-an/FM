"""
Running this file trains ANNs described in the paper.

*Preview of this code:
- Packages

- Fig. 2a (Fig. S5a, Fig. S6a)
- Fig. 2b (Fig. S5c, Fig. S5d, Fig. S6c, Fig. S6d)
- Fig. 2c (Fig. S5e, Fig. S5f, Fig. S6e, Fig. S6f, Fig. S6g)
- Fig. 2d (Fig. S5g, Fig. S6h, Fig. S6i)

"""



#%%
# Packages

import datetime
date = datetime.datetime.now()
print(f'Today is Happy{date: %A, %d, %m, %Y}.', '\n')

import os
import gc
import time
import random
import itertools
from random import sample
from PIL import Image, ImageOps

import numpy as np 
import matplotlib
import matplotlib.pyplot as plt 
import pandas as pd
from pandas import DataFrame
#from IPython.display import Image
import json # instead of "Pickle" and "Joblib"
import joblib
import gzip
import collections
print(np.__version__)
print(matplotlib.__version__)
print(pd.__version__, '\n')

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
print(sklearn.__version__)
print(skimage.__version__, '\n')

import tensorflow as tf
from tensorflow.train import Feature, BytesList, Int64List, Example, Features
from tensorflow.image import rgb_to_grayscale
from tensorflow.data import TFRecordDataset
from tensorflow.io import FixedLenFeature, parse_single_example, decode_raw, decode_image, TFRecordWriter
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
from tensorflow.keras import Sequential
from tensorflow.keras.models import load_model
import keras
from keras.applications import resnet, vgg19
from keras.applications.resnet import preprocess_input, ResNet50, ResNet101, ResNet152
from keras.applications.vgg19 import preprocess_input, VGG19
from keras.applications.densenet import preprocess_input, DenseNet121, DenseNet169, DenseNet201
from keras.layers import GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from tensorflow.keras.utils import to_categorical
print(tf.__version__)
print(keras.__version__)


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
print(torch.cuda.is_available()) # True면 GPU 가속 사용 가능
print(torch.version.cuda)

from mypackages import pytorchtools # after adding an empty file named '__init__.py'



#%%
# *Buttons of hyper-parameters

# CHANGE
test_type = 'elec' # 'opt' or 'elec' or 'normal'

model_type1 = '' # 'PCA', 'PCA', 'PCA', '', '', '', '', '', '', '', '', '', ''
model_type2 = 'CNN_SVC'
# 'SVC2'(Old version) 'SVC', 'LR', 'CNN_LR', 'CNN_SVC', 'PIXEL_LR', 'PIXEL_SVC',
# 'CNN_ResNet', 'CNN_ResNet2', CNN_ResNet2_SVC', 'CNN_AlexNet', 'CNN_AlexNet2', 'CNN_AlexNet2_SVC', 'CNN_VggNet2', 'CNN_VggNet2_SVC'
model_type = model_type1 + model_type2
xai = 'no' # 'yes', 'no'
finetune = '' # 'ft', ''

r = 4

# CAN BE CHANGED (BUT PREFER NOT TO)
#if model_type2 == 'CNN_LR':
#    img_dim = 128 # to run experiments efficiently
#else:
#    img_dim = 256


# DO NOT CHANGE
# df = pd.read_csv('C:\\Users\\user\\Desktop\\KFace_data_information_Folder1_400.csv')
df = pd.read_csv('C:\\Users\\user\\Desktop\\210827_ANNA_Removing_uncontaminated_data.csv')

l = list(range(df.shape[0]))
l_2030_f = list(df.loc[(df['연령대'].isin(['20대','30대'])) & (df['성별']=='여')].index)
l_4050_f = list(df.loc[(df['연령대'].isin(['40대','50대'])) & (df['성별']=='여')].index)
l_2030_m = list(df.loc[(df['연령대'].isin(['20대','30대'])) & (df['성별']=='남')].index)
l_4050_m = list(df.loc[(df['연령대'].isin(['40대','50대'])) & (df['성별']=='남')].index)
l_20304050_f = list(df.loc[df['성별'] == '여'].index)
l_20304050_m = list(df.loc[df['성별'] == '남'].index)
n = 16

random.seed(22)
set_1 = random.sample(l, n)
set_2 = random.sample(l_2030_f, n)
set_3 = random.sample(l_4050_f, n)
set_4 = random.sample(l_2030_m, n)
set_5 = random.sample(l_4050_m, n)
set_6 = random.sample(l_20304050_f, n)
set_7 = random.sample(l_20304050_m, n)

random.seed(116)
set_8 = random.sample(l, n)
set_9 = random.sample(l_2030_f, n)
set_10 = random.sample(l_4050_f, n)
set_11 = random.sample(l_2030_m, n)
set_12 = random.sample(l_4050_m, n)
set_13 = random.sample(l_20304050_f, n)
set_14 = random.sample(l_20304050_m, n)

sets = [set_1, set_2, set_3, set_4, set_5, set_6, set_7, set_8, set_9, set_10, set_11, set_12, set_13, set_14] # 14 independent sets are ready! :)
# Below three lines are not necessarily needed, but I wrote it to check how many "faces" we are dealing with.
# ppl_l = set_1 + set_2 + set_3 + set_4 + set_5 + set_6 + set_7 + set_8 + set_9 + set_10 + set_11 + set_12 + set_13 + set_14
# ppl_l = list(set(ppl_l))
# print(len(ppl_l))

# train_path =  'C:\\Users\\user\\Desktop\\Middle_Resolution_177_unzipped_parcropped_128' # Combination 돌릴 준비된 사람 데이터들
# test_path =  f'C:\\Users\\user\\Desktop\\Middle_Resolution_177_unzipped_parcropped_128_{test_type}' # Combination 돌릴 준비된 사람 데이터들
train_path =  'E:\\ANNA_INTERN\\Middle_Resolution_137_unzipped_parcropped_128_removed_train' 
if test_type == 'opt' or test_type == 'elec':
    test_path =  f'E:\\ANNA_INTERN\\Middle_Resolution_137_unzipped_parcropped_128_{test_type}'
elif test_type == 'normal':
    test_path = 'E:\\ANNA_INTERN\\Middle_Resolution_137_unzipped_parcropped_128_removed_train'

# test 이미지
camera_list = ['4','7','10'] #4(C), 정면은 7
light_list = ['1'] #2(L)
accessory_list =['1'] #1(S)
expression_list = ['1','2','3'] #3(E)

# train 이미지
tr_camera_list = list(map(str, [4,5,6,7,8,9,10,14,15,16,17,18,19,20])) #4(C), 정면은 7
tr_light_list = list(map(str, [1,2,3,4,5,6, # 검은거(L7) 제거
    8,9,10,11,12,13,14,15,16,17,18,
    19,20,21,22,23,24,25,26,27,28,29,30])) #2(L)
tr_accessory_list =['1'] #1(S)
tr_expression_list = ['1','2','3'] #3(E)

if model_type1 == 'PCA':
    ext_name1 = 'gz'
elif model_type1 == '':
    ext_name1 = ''

if model_type2 == 'SVC2':
    ext_name2 = 'gz'
elif model_type2 == 'LR' or model_type2 == 'SVC' or model_type2 == 'CNN_LR' or model_type2 == 'CNN_SVC' or model_type2 == 'PIXEL_LR' or model_type2 == 'PIXEL_SVC' or model_type2 == 'CNN_ResNet' or model_type2 == 'CNN_ResNet2' or model_type2 == 'CNN_ResNet2_SVC' or model_type2 == 'CNN_AlexNet' or model_type2 == 'CNN_AlexNet2' or model_type2 == 'CNN_AlexNet2_SVC' or model_type2 == 'CNN_VggNet2' or model_type2 == 'CNN_VggNet2_SVC':
    ext_name2 = 'pt'

seed_list = [100] #[22, 77, 2, 100, 81, 42, 7, 1, 55, 50] # different 1,000 training images (total of 10 permutations)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 


for m in range(14): # 14개의 identically, independent distributed set
    input_folder = [df.iloc[i, 0] for i in sets[m]] # 무조건 16명의 사람
    assert len(input_folder) == 16
    com_obj = itertools.combinations(input_folder, r)
    com_list = list(com_obj)
    
    for seed in seed_list:
        for n in range(len(com_list)): # for every combination # range(178, 210, 1)

            # data_path = f'C:\\Users\\user\\Desktop\\Question Banks AI Hub\\{r}classes\\set{m}'
            # data_path = f'D:\\ANNA_INTERN\\Question Banks AI Hub_final\\{r}classes\\set{m}\\seed{seed}'
            data_path = f'E:\\ANNA_INTERN\\Question Banks AI Hub_final\\{r}classes\\set{m}\\seed{seed}'
            preprocessed_data_path =  os.path.join(data_path, f'comb{n}')

            model_path = os.path.join(preprocessed_data_path, 'Saved_Models')
            high_analysis_path = os.path.join(preprocessed_data_path, f'High_Analysis_{test_type}')
            low_analysis_path = os.path.join(preprocessed_data_path, f'Low_Analysis_{test_type}')

            os.makedirs(data_path, exist_ok=True)
            os.makedirs(preprocessed_data_path, exist_ok=True) 
            os.makedirs(model_path, exist_ok=True)
            os.makedirs(high_analysis_path, exist_ok=True)
            # os.makedirs(low_analysis_path, exist_ok=True)

            model_file1 = os.path.join(model_path, f'Model_{model_type1}{finetune}.{ext_name1}')
            model_file2 = os.path.join(model_path, f'Model_{model_type2}{finetune}.{ext_name2}')
            checkpoint_file = os.path.join(model_path, f'Checkpoint_{model_type2}{finetune}.{ext_name2}') # for PCA_LR, CNN_LR, CNN_SVC
            earlystop_file = os.path.join(model_path, f'Early_Stop_{model_type2}{finetune}.{ext_name2}') # for PCA_LR, CNN_LR, CNN_SVC
            high_csv_file = os.path.join(high_analysis_path, f'High_Level_Data_Analysis_{model_type1}_{model_type2}{finetune}.csv')
            
            low_csv_file = os.path.join(low_analysis_path, f'Classification Report_{model_type1}_{model_type2}.csv') # 안 씀 
            auc_info_file = os.path.join(low_analysis_path, f'TP_FP_AUC_Dictionary_{model_type1}_{model_type2}.csv') # 안 씀
            roc_file = os.path.join(low_analysis_path, f'ROC_{model_type1}_{model_type2}.png') # 안 씀
            # if model_type2 == 'SVC':
            #     check_file = low_csv_file
            # else:
            #     check_file = auc_info_file
            check_file = high_csv_file

            if os.path.isfile(check_file) and xai == 'no': # 특정 조합에 대해서 맨 마지막에 저장될 파일이 이미 존재한다면,
                pass
            else:
                print(f'\n START {m}th set {n}th comb (seed{seed}): comb{com_list[n]} \n')
                """1. Loading a data"""
                Xtrain, ytrain, _, old_uniq_labels, tr_unique_items = loadData(com_list[n], 'train') 
                Xtest, ytest, file_path_list, old_uniq_labels2, test_unique_items = loadData(com_list[n], 'test')
                assert set(old_uniq_labels) == set(old_uniq_labels2)

                # visualizeData(Xtrain, ytrain, Xtest, ytest)
                
                assert set(tr_unique_items) == set(test_unique_items)
                unique_labels = tr_unique_items

                """2. Modeling"""
                # print(gc.collect())
                # print(torch.cuda.empty_cache())
                # print(gc.collect())
                instance = beginModeling(n, model_type1, model_type2, Xtrain, ytrain, Xtest, ytest, unique_labels, model_file1, model_file2, high_csv_file, low_csv_file, auc_info_file, checkpoint_file, earlystop_file, roc_file) # INITIALIZATION

                # method 호출
                model, Xtrain, Xtest = instance.loadOrSaveModel1() # FEATURE EXTRACTION (PART 1)
                train_loader, val_loader, test_loader = instance.convertAndVisualData(model, Xtrain, Xtest, ytrain, ytest) # (OPTIONAL) VISUALIZATION 1|
                ytest, yfit, yprob, mod, y_test_oh= instance.loadOrSaveModel2andEval(train_loader, val_loader, test_loader, Xtrain, Xtest, old_uniq_labels2, file_path_list) # CLASSIFICATION (PART 2)
                # instance.visualPredData(testdataset, mod) # (OPTIONAL) VISUALIZATION 2
                instance.ready4Visualization(ytest, yfit, yprob, file_path_list, old_uniq_labels2, unique_labels, y_test_oh) # VISUALIZATION 3


#%%
# Loading a data

def loadData(com_list_n, which_data_mode):

    if which_data_mode == 'train':
        path = train_path
    elif which_data_mode == 'test':
        path = test_path

    # Note: 학습 데이터 중 배치 사이즈만큼의 데이터를 고려하여 모델의 weight를 갱신함
    # img_features = lis|t()
    file_path_list, label_list = list(), list()
    start_time = time.time()

    if finetune == 'ft' and which_data_mode == 'train':
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

            elif ran_val == 1: # finetune
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
    if model_type2 == 'CNN_ResNet' or model_type2 == 'CNN_AlexNet': # 이전거
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



#%%
# Training ANNs

class beginModeling():
    # class variables
    n_components = 128
    val_percent = 0.15
    batch_size = 16
    num_epoch = 1000 #1000
    learning_rate = 1e-4 # 1e-4 
    n_class = 4


    ##### INITIALIZATION
    def __init__(self, n, model_type1, model_type2, Xtrain, ytrain, Xtest, ytest, unique_labels, model_file1, model_file2, high_csv_file, low_csv_file, auc_info_file, checkpoint_file, earlystop_file, roc_file):

        self.model_type1 = model_type1
        self.model_type2 = model_type2
        self.Xtrain = Xtrain
        self.ytrain = ytrain
        self.Xtest = Xtest
        self.ytest = ytest
        self.unique_labels = unique_labels
        
        self.model_file1 = model_file1
        self.model_file2 = model_file2
        self.high_csv_file = high_csv_file
        self.low_csv_file = low_csv_file
        self.auc_info_file = auc_info_file
        self.checkpoint_file = checkpoint_file
        self.earlystop_file = earlystop_file
        self.roc_file = roc_file

        #self.LRModel = self.LR_Model()


    ##### FEATURE EXTRACTION (PART 1)
    def loadOrSaveModel1(self):
            
        if os.path.isfile(self.model_file1): # If there IS a saved model file...
            print(f'Loading a model 1 ({self.model_type1})...')            

            if self.model_type1 == 'PCA':
                with gzip.GzipFile(self.model_file1, 'rb') as f:
                    model = joblib.load(f)
                Xtrain = self.Xtrain.reshape(self.Xtrain.shape[0], self.Xtrain.shape[1]*self.Xtrain.shape[2])
                Xtest = self.Xtest.reshape(self.Xtest.shape[0], self.Xtest.shape[1]*self.Xtest.shape[2])
                print("Original data shape: ", Xtrain.shape, Xtest.shape)
                Xtrain, Xtest = model.transform(Xtrain), model.transform(Xtest) # Reducing the dimensionality of Xtrain and Xtest
                print("PCA transformed data shape: ", Xtrain.shape, Xtest.shape)

            elif self.model_type1 == '':
                model, Xtrain, Xtest = None, self.Xtrain, self.Xtest

                inputs, targets, testinputs, testtargets = torch.from_numpy(Xtrain).float(), torch.from_numpy(self.ytrain).long(), torch.from_numpy(Xtest).float(), torch.from_numpy(self.ytest).long() # long: integers
                dataset, testdataset = TensorDataset(inputs, targets), TensorDataset(testinputs, testtargets)
                train_loader, test_loader = DataLoader(dataset), DataLoader(testdataset)

                ytrain, ytrain_fit = np.array([]), np.array([])
                for i, (inputs, targets) in enumerate(train_loader):
                    y, yhat = targets, model(inputs)
                    ytrain, ytrain_fit = np.append(ytrain, y), np.append(ytrain_fit, yhat)

                ytest, ytest_fit = np.array([]), np.array([])
                for i, (inputs, targets) in enumerate(test_loader):
                    y, yhat = targets, model(inputs)
                    ytest, ytest_fit = np.append(ytest, y), np.append(ytest_fit, yhat)
                Xtrain, Xtest = ytrain_fit, ytest_fit
                print('CNN transformed data shape: ', Xtrain.shape, Xtest.shape) # (batch, 1, beginModeling.n_components)


        else: # If there ISN'T...
            print(f'Making a model 1 ({self.model_type1})...')

            if self.model_type1 == 'PCA':
                Xtrain = self.Xtrain.reshape(self.Xtrain.shape[0], self.Xtrain.shape[1]*self.Xtrain.shape[2])
                Xtest = self.Xtest.reshape(self.Xtest.shape[0], self.Xtest.shape[1]*self.Xtest.shape[2])
                model = PCA(beginModeling.n_components, whiten=True, random_state=42)
                model.fit(Xtrain)
                #eigenfaces = model.components_.reshape((beginModeling.n_components, img_dim, img_dim)) # 2D에서 3D로(flatten되어 있던 것을 (150, 150)으로), eigen-vectors
                # 여기에 나중에 eigenface 시각화
                plt.plot(np.cumsum(model.explained_variance_ratio_))
                plt.xlabel('Number of Components')
                plt.ylabel('Cumulative Explained Variance')
                plt.show()
                print(f'Saving a model 1 ({self.model_type1})...')
                with gzip.GzipFile(self.model_file1, 'wb', compresslevel=3) as f:
                    joblib.dump(model, f) # 만든 mod를 파일로
                print("Original data shape: ", Xtrain.shape, Xtest.shape)
                Xtrain, Xtest = model.transform(Xtrain), model.transform(Xtest) # Reducing the dimensionality of Xtrain and Xtest
                print("PCA transformed data shape: ", Xtrain.shape, Xtest.shape)

            elif self.model_type1 == '':
                model, Xtrain, Xtest = None, self.Xtrain, self.Xtest

        return model, Xtrain, Xtest
    


    ##### (OPTIONAL) VISUALIZATION (PART 1)
    def convertAndVisualData(self, model, Xtrain, Xtest, ytrain, ytest):
       
        inputs, targets, testinputs, testtargets = torch.from_numpy(Xtrain).float(), torch.from_numpy(self.ytrain).long(), torch.from_numpy(Xtest).float(), torch.from_numpy(self.ytest).long() # long: integers
        dataset, testdataset = TensorDataset(inputs, targets), TensorDataset(testinputs, testtargets)

        if self.model_type2 != 'SVC2':
            val_size = int(self.Xtrain.shape[0]*beginModeling.val_percent)
            train_size = self.Xtrain.shape[0] - val_size
            train_ds, val_ds = random_split(dataset, [train_size, val_size])

            train_loader, val_loader = DataLoader(train_ds, beginModeling.batch_size, shuffle=True), DataLoader(val_ds, beginModeling.batch_size*2, shuffle=True)
            test_loader = DataLoader(testdataset, shuffle=False) #, batch_size*2, shuffle=True)
        else:
            train_loader, val_loader, test_loader = None, None, None

        # print("Visualizing a transformed data...")
        # random_idx = random.randint(0, len(Xtrain))
        # if self.model_type1 == 'PCA':
        #     plt.eventplot(Xtrain[random_idx])
        # elif self.model_type1 == '':
        #     plt.imshow(Xtrain[random_idx], cmap='gray')
        # #plt.title('Face #', ytrain[random_idx])
        # plt.show()
        
        return train_loader, val_loader, test_loader


    ###### CLASSIFICATION (PART 2)
    def loadOrSaveModel2andEval(self, train_loader, val_loader, test_loader, Xtrain, Xtest, old_uniq_labels2, file_path_list):

        #----------------------------------------------------------------------------------------------------------
        # Below two functions are needed to change list of torch.tensors to list of floats.
        def flatten(L):
            for item in L:
                try:
                    yield from flatten(item)
                except TypeError:
                    yield item

        def gpu2cpu(L):
            temp_list = list()
            for item in L:
                try:
                    item = item.cpu().numpy()
                except:
                    pass
                temp_list.append(item)
            temp_list2 = list(map(float, temp_list))
            return temp_list2
        #----------------------------------------------------------------------------------------------------------

        """Load or save a model."""
        if os.path.isfile(self.model_file2): # If there IS a saved model file...
            print(f'Loading a model 2 ({self.model_type2})...')
            if self.model_type2 == 'LR' or self.model_type2 == 'SVC' or self.model_type2 == 'CNN_LR' or self.model_type2 == 'CNN_SVC' or self.model_type2 == 'PIXEL_LR' or self.model_type2 == 'PIXEL_SVC' or self.model_type2 == 'CNN_ResNet' or model_type2 == 'CNN_ResNet2' or model_type2 == 'CNN_ResNet2_SVC' or model_type2 == 'CNN_AlexNet' or model_type2 == 'CNN_AlexNet2' or model_type2 == 'CNN_AlexNet2_SVC' or model_type2 == 'CNN_VggNet2' or model_type2 == 'CNN_VggNet2_SVC':
                model = torch.load(self.model_file2)
                if os.path.isfile(self.checkpoint_file):
                    os.remove(self.checkpoint_file)
                if os.path.isfile(self.earlystop_file):
                    os.remove(self.earlystop_file)
                #final_checkpoint = torch.load(earlystop_file) # checkpoint_file 아님
                #model.load_state_dict(final_checkpoint)  

            elif self.model_type2 == 'SVC2':
                with gzip.GzipFile(self.model_file2, 'rb') as f:
                    model = joblib.load(f)

        else:
            print(f'Making a model 2 ({self.model_type2})...')
            if self.model_type2 == 'SVC2':
                print('Grid searching...')
                param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100]} #[1, 5, 10, 50],
                # C를 낮게 설정하면 이상치들이 있을 가능성을 크게 보아 일반적인 결정 경계를 찾는다는 의미이고(underfitting)
                # C를 높게 설정하면 이상치들의 존재 가능성을 적게 보아 좀 더 세심하게 결정 경계를 찾는다는 의미임(overfitting)
                               # 'gamma' : [0.001, 0.01, 0.1, 1, 10, 100]} #[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1]} 
                               # 선형 SVM으로 데이터를 제대로 분류할 수 없는 상황에서 비선형 SVM으로 대체
                               # 감마가 클수록 한 데이터 포인터가 영향력을 행사하는 거리가 짧아짐(overfitting)
                # model = GridSearchCV(SVC(kernel='rbf', class_weight='balanced', probability=False), param_grid, verbose=3)
                # rbf(radical based function): 비선형 커널 중 가장 많이 사용되는 커널, degree: 3(default), probability: 5-fold cross-validation
                model = GridSearchCV(LinearSVC(penalty='l2', loss='hinge', multi_class='ovr'), param_grid=param_grid) #, verbose=1)
                %time model.fit(Xtrain, self.ytrain) # score: train accuracy
                #print('Cross Val Results: ', model.cv_results_)
                #print('Best Score: ', model.best_score_)
                print('Best Parameters: ', model.best_params_)
                #print('Best Index: ', model.best_index_)
                #print('Scorer: ', model.scorer_)
                #print('N Splits: ', model.n_splits_)
                #print('Multimetric?: ', model.multimetric_)

                print(f"Saving a model 2 ({self.model_type2})...")
                with gzip.GzipFile(self.model_file2, 'wb', compresslevel=3) as f:
                    joblib.dump(model, f) # 만든 mod를 파일로

            elif self.model_type2 == 'LR' or self.model_type2 == 'SVC' or self.model_type2 == 'CNN_LR' or self.model_type2 == 'CNN_SVC' or self.model_type2 == 'PIXEL_LR' or self.model_type2 == 'PIXEL_SVC' or self.model_type2 == 'CNN_ResNet' or model_type2 == 'CNN_ResNet2' or model_type2 == 'CNN_ResNet2_SVC' or model_type2 == 'CNN_AlexNet' or model_type2 == 'CNN_AlexNet2' or model_type2 == 'CNN_AlexNet2_SVC'or model_type2 == 'CNN_VggNet2' or model_type2 == 'CNN_VggNet2_SVC':

                m1, m2 = self.model_type1, self.model_type2
                model = self.LR_Model(m1, m2) # 1) 아래에 있는 sub-class 활용

                # print("Model's structure:", '\n', model)
                model, avg_train_losses, avg_valid_losses, avg_train_accs, avg_valid_accs = model.fit(beginModeling.num_epoch, beginModeling.learning_rate, model, train_loader, val_loader, self.checkpoint_file, self.earlystop_file) # 2)

                # avg_train_losses, avg_valid_losses, avg_train_accs, avg_valid_accs = gpu2cpu(flatten(avg_train_losses)), gpu2cpu(flatten(avg_valid_losses)), gpu2cpu(flatten(avg_train_accs)), gpu2cpu(flatten(avg_valid_accs))
                # model.visualize(avg_train_losses, avg_valid_losses, avg_train_accs, avg_valid_accs)
            
                print(f'Saving a model 2 ({self.model_type2})...')
                torch.save(model, self.model_file2) 
        

        """Fit the model."""
        print(f'Evaluating a model 2 ({self.model_type2})...')
        if self.model_type2 == 'SVC2':
            res_model = model.best_estimator_ # estimator that was chosen by the search
            print('Best Estimator: ', res_model)

            testinputs, testtargets = torch.from_numpy(Xtest).float(), torch.from_numpy(self.ytest).long()
            testdataset = TensorDataset(testinputs, testtargets)
            test_loader = DataLoader(testdataset, shuffle=False)

            yfit, ytest = np.array([]), np.array([])
            yprob = np.array([])
            for i, (inputs, targets) in enumerate(test_loader):
                yhat, y = res_model.predict(inputs), targets
                ytest, yfit = np.append(ytest, y), np.append(yfit, yhat)

        elif self.model_type2 == 'LR' or self.model_type2 == 'SVC' or self.model_type2 == 'CNN_LR' or self.model_type2 == 'CNN_SVC' or self.model_type2 == 'PIXEL_LR' or self.model_type2 == 'PIXEL_SVC' or self.model_type2 == 'CNN_ResNet' or self.model_type2 == 'CNN_ResNet2' or self.model_type2 == 'CNN_ResNet2_SVC' or self.model_type2 == 'CNN_AlexNet' or self.model_type2 == 'CNN_AlexNet2' or model_type2 == 'CNN_AlexNet2_SVC' or model_type2 == 'CNN_VggNet2' or model_type2 == 'CNN_VggNet2_SVC':
            #print("Model's parameters: ", model.state_dict()) # model parameters
            ytest, yfit, yprob = np.array([]), np.array([]), np.array([])
            model = model.to(device)
            model.eval()

            for i, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                inputs.requires_grad_()
                y = targets
                #inputs = inputs.reshape(-1, beginModeling.n_components)
                #lin = Linear(beginModeling.n_components, beginModeling.n_class) 
                #inputs = lin(inputs) # shape: (-1, beginModeling.n_class)
                #out = F.softmax(inputs, dim=1)
                #yhat = out
                model = model.to(device)
                yhat = model(inputs) # predicted probabilities (of k개의 classes)
                temp_yhat = yhat
                pred_probs = yhat.detach() # np.ndarray object does not have extra "computational graph," unlike torch.tensor
                _, yhat = torch.max(yhat, dim=1) # max, max_indices
                y, yhat = y.cpu().numpy(), yhat.cpu().numpy()
                pred_probs = gpu2cpu(flatten(pred_probs))
                ytest, yfit, yprob = np.append(ytest, y), np.append(yfit, yhat), np.append(yprob, pred_probs)

                #_, _, _, _, _, f1, f2 = file_path_list[i].split('\\')
                _, _, _, f1, f2 = file_path_list[i].split('\\')
                full_file_name = str(f1) + '_' + str(f2)

            yprob = yprob.reshape(yfit.shape[0], beginModeling.n_class)

        print("Real ppl.: ", ytest.shape, "Predicted ppl.: ", yfit.shape, "Predicted prob.: ", yprob.shape) # 예측된 사람 / 실제 사람 / 예측 확률 # Doesn't matter of what type of the model 2 is...
        assert ytest.shape == yfit.shape

        """Getting ready to evaluate the model."""
        y_test_oh = to_categorical(ytest)
        for idx in range(ytest.shape[0]):
            temp_test, temp_fit = old_uniq_labels2[int(ytest[idx])], old_uniq_labels2[int(yfit[idx])]
            ytest[idx], yfit[idx] = temp_test, temp_fit

        return ytest, yfit, yprob, model, y_test_oh


    class LR_Model(torch.nn.Module):

        """1) INITIALIZATION"""
        def __init__(self, m1, m2):
            super().__init__() # super(LRModel, self).__init__()으로 써도 무방(옛날 방식), to call the __init__ method of the superclass, which in this case is Module.

            self.model_type1, self.model_type2 = m1, m2

            # For a logistic regression model, or when model_type1 == 'PCA'
            self.linear = Linear(in_features=beginModeling.n_components, out_features=beginModeling.n_class).to(device) # Logistic Regression, # of input nodes & # of output nodes

            # For a 'lite' CNN, or when model_type == 'CNN_LR' or model_type == 'CNN_SVC'
            # Oliveira[2017], 23 classes, 99% end-to-end acc., 25,000 test images
            self.cnn_num_block = nn.Sequential(
                Conv2d(in_channels=1, out_channels=32, kernel_size=7, stride=1, padding=0).to(device),
                ReLU(inplace=True).to(device), # perform th eoperation w/ using any additional memory
                MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False).to(device),

                Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=0).to(device),
                ReLU(inplace=True).to(device),
                MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False).to(device),

                Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0).to(device),
                ReLU(inplace=True).to(device),
                MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False).to(device),

                Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=0).to(device),
                ReLU(inplace=True).to(device))
             
            self.linear_num_block = nn.Sequential(
                 Linear(in_features=256*11*11, out_features=beginModeling.n_components, bias=True).to(device), # 논문에서는 1024 대신 128 뉴런 사용
                 ReLU(inplace=True).to(device), # No negative #'s
                 Linear(in_features=beginModeling.n_components, out_features=beginModeling.n_class, bias=True)).to(device)

            # For a pixel model, or when model_type == 'PIXEL_LR' or model_type == 'PIXEL_SVC'
            self.pixel = Linear(in_features=128*128, out_features=beginModeling.n_class).to(device)

            # Source code for torchvision.models.alexnet
            self.alexnet_features = nn.Sequential(
                Conv2d(1, 64, kernel_size=11, stride=4, padding=2).to(device),
                ReLU(inplace=True).to(device),
                MaxPool2d(kernel_size=3, stride=2).to(device),

                Conv2d(64, 192, kernel_size=5, padding=2).to(device),
                ReLU(inplace=True).to(device),
                MaxPool2d(kernel_size=3, stride=2).to(device),

                Conv2d(192, 384, kernel_size=3, padding=1).to(device),
                ReLU(inplace=True).to(device),

                Conv2d(384, 256, kernel_size=3, padding=1).to(device),
                ReLU(inplace=True).to(device),

                Conv2d(256, 256, kernel_size=3, padding=1).to(device),
                ReLU(inplace=True).to(device),
                MaxPool2d(kernel_size=3, stride=2).to(device)
            )
            self.alexnet_avgpool = AdaptiveAvgPool2d((1, 1)).to(device)
            self.alexnet_classifier = nn.Sequential(
                # Dropout().to(device),
                # Linear(256 * 6 * 6, 4096).to(device),
                # ReLU(inplace=True).to(device),

                # Dropout().to(device),
                # Linear(4096, 4096).to(device),
                # ReLU(inplace=True).to(device),

                # Linear(4096, beginModeling.n_class).to(device),
                Linear(in_features=1*1*256, out_features=beginModeling.n_components, bias=True).to(device),
                ReLU(inplace=True).to(device), # No negative #'s
                Linear(in_features=beginModeling.n_components, out_features=beginModeling.n_class, bias=True).to(device)
            )

            # Source code for torchvision.models.vggnet
            self.vggnet_features = nn.Sequential(
                Conv2d(1, 64, kernel_size=3, stride=1, padding=1).to(device),
                ReLU(inplace=True).to(device),
                MaxPool2d(kernel_size=2, stride=2).to(device),

                Conv2d(64, 128, kernel_size=3, padding=1).to(device),
                ReLU(inplace=True).to(device),
                MaxPool2d(kernel_size=2, stride=2).to(device),

                Conv2d(128, 256, kernel_size=3, padding=1).to(device),
                ReLU(inplace=True).to(device),
                Conv2d(256, 256, kernel_size=3, padding=1).to(device),
                ReLU(inplace=True).to(device),
                MaxPool2d(kernel_size=2, stride=2).to(device),

                Conv2d(256, 512, kernel_size=3, padding=1).to(device),
                ReLU(inplace=True).to(device),
                Conv2d(512, 512, kernel_size=3, padding=1).to(device),
                ReLU(inplace=True).to(device),
                MaxPool2d(kernel_size=2, stride=2).to(device),

                Conv2d(512, 512, kernel_size=3, padding=1).to(device),
                ReLU(inplace=True).to(device),
                Conv2d(512, 512, kernel_size=3, padding=1).to(device),
                ReLU(inplace=True).to(device),
                MaxPool2d(kernel_size=2, stride=2).to(device),
            )
            self.vggnet_avgpool = AdaptiveAvgPool2d((1, 1)).to(device)
            self.vggnet_classifier = nn.Sequential(
                # Linear(512 * 7 * 7, 4096).to(device),
                # ReLU(inplace=True).to(device),
                # Dropout().to(device),

                # Linear(4096, 4096).to(device),
                # ReLU(inplace=True).to(device),
                # Dropout().to(device),

                # Linear(4096, beginModeling.n_class).to(device),
                Linear(in_features=1*1*512, out_features=beginModeling.n_components, bias=True).to(device),
                ReLU(inplace=True).to(device), # No negative #'s
                Linear(in_features=beginModeling.n_components, out_features=beginModeling.n_class, bias=True).to(device)
            )

            # Source code for torchvision.models.resnet
            # 2*padding = kernel_size - 1이고, stride=2이면 이미지 크기가 반으로 
            # 예: padding=1, kernel_size=3
            # 예: padding=0, kernel_size=1
            self.resnet_filters = [64, 128, 256, 512]

            self.resnet_conv = nn.Sequential(
                Conv2d(1, self.resnet_filters[0], kernel_size=7, stride=2, padding=3, bias=False).to(device),
                BatchNorm2d(self.resnet_filters[0]).to(device),
                ReLU(inplace=True).to(device),
                MaxPool2d(kernel_size=3, stride=2, padding=1).to(device)
            )
            
            # BLOCK 1 (이미지 크기 변하진 않음, 채널만 많아짐)
            self.resnet_conv1 = nn.Sequential(
                Conv2d(self.resnet_filters[0], self.resnet_filters[0], kernel_size=3, stride=1, padding=1, bias=False).to(device),
                BatchNorm2d(self.resnet_filters[0]).to(device),
                ReLU().to(device),

                Conv2d(self.resnet_filters[0], self.resnet_filters[0] , kernel_size=3, stride=1, padding=1, bias=False).to(device),
                BatchNorm2d(self.resnet_filters[0] ).to(device)
            )
            self.resnet_conv1_id = nn.Sequential(
                        Conv2d(self.resnet_filters[0], self.resnet_filters[0] , kernel_size=1, stride=1, bias=False).to(device),
                        BatchNorm2d(self.resnet_filters[0] ).to(device)
                    )
        
            # BLOCK 2
            self.resnet_conv2 = nn.Sequential(
                Conv2d(self.resnet_filters[0] , self.resnet_filters[1], kernel_size=3, stride=2, padding=1, bias=False).to(device), # ((h or w) + 2*padding(=1) - (kernel_size(=3)-1)) // stride(=2) = (h or w) // stride(=2)
                BatchNorm2d(self.resnet_filters[1]).to(device),
                ReLU().to(device),

                Conv2d(self.resnet_filters[1], self.resnet_filters[1], kernel_size=3, stride=1, padding=1, bias=False).to(device),
                BatchNorm2d(self.resnet_filters[1]).to(device)
            )
            self.resnet_conv2_id = nn.Sequential(
                        Conv2d(self.resnet_filters[0] , self.resnet_filters[1] , kernel_size=1, stride=2, bias=False).to(device),
                        BatchNorm2d(self.resnet_filters[1] ).to(device)
                    )
            self.resnet_conv2_2 = nn.Sequential(
                Conv2d(self.resnet_filters[1], self.resnet_filters[1] , kernel_size=1, stride=1, padding=0, bias=False).to(device), # ((h or w) + 2*padding(=1) - (kernel_size(=3)-1)) // stride(=2) = (h or w) // stride(=2)
                BatchNorm2d(self.resnet_filters[1] ).to(device),
                ReLU().to(device),

                Conv2d(self.resnet_filters[1] , self.resnet_filters[1], kernel_size=1, stride=1, padding=0, bias=False).to(device),
                BatchNorm2d(self.resnet_filters[1] ).to(device)
            )
            self.resnet_conv2_2_id = nn.Sequential(
                        Conv2d(self.resnet_filters[1], self.resnet_filters[1] , kernel_size=1, stride=1, bias=False).to(device),
                        BatchNorm2d(self.resnet_filters[1] ).to(device)
                    )

            # BLOCK 3
            self.resnet_conv3 = nn.Sequential(
                Conv2d(self.resnet_filters[1] , self.resnet_filters[2], kernel_size=3, stride=2, padding=1, bias=False).to(device),
                BatchNorm2d(self.resnet_filters[2]).to(device),
                ReLU().to(device),

                Conv2d(self.resnet_filters[2], self.resnet_filters[2], kernel_size=3, stride=1, padding=1, bias=False).to(device),
                BatchNorm2d(self.resnet_filters[2]).to(device)
            )
            self.resnet_conv3_id = nn.Sequential(
                        Conv2d(self.resnet_filters[1] , self.resnet_filters[2], kernel_size=1, stride=2, bias=False).to(device),
                        BatchNorm2d(self.resnet_filters[2]).to(device)
                    )
            self.resnet_conv3_2 = nn.Sequential(
                Conv2d(self.resnet_filters[2], self.resnet_filters[2], kernel_size=1, stride=1, padding=0, bias=False).to(device),
                BatchNorm2d(self.resnet_filters[2]).to(device),
                ReLU().to(device),

                Conv2d(self.resnet_filters[2], self.resnet_filters[2] , kernel_size=1, stride=1, padding=0, bias=False).to(device),
                BatchNorm2d(self.resnet_filters[2]).to(device)
            )
            self.resnet_conv3_2_id = nn.Sequential(
                        Conv2d(self.resnet_filters[2], self.resnet_filters[2] , kernel_size=1, stride=1, bias=False).to(device),
                        BatchNorm2d(self.resnet_filters[2] ).to(device)
                    )

            # BLOCK 4
            self.resnet_conv4 = nn.Sequential(
                Conv2d(self.resnet_filters[2] , self.resnet_filters[3], kernel_size=3, stride=2, padding=1, bias=False).to(device),
                BatchNorm2d(self.resnet_filters[3]).to(device),
                ReLU().to(device),
            
                Conv2d(self.resnet_filters[3], self.resnet_filters[3], kernel_size=3, stride=1, padding=1, bias=False).to(device),
                BatchNorm2d(self.resnet_filters[3]).to(device)
            )
            self.resnet_conv4_id = nn.Sequential(
                        Conv2d(self.resnet_filters[2] , self.resnet_filters[3], kernel_size=1, stride=2, bias=False).to(device),
                        BatchNorm2d(self.resnet_filters[3]).to(device)
                    )
            self.resnet_conv4_2 = nn.Sequential(
                Conv2d(self.resnet_filters[3], self.resnet_filters[3], kernel_size=1, stride=1, padding=0, bias=False).to(device),
                BatchNorm2d(self.resnet_filters[3]).to(device),
                ReLU().to(device),
            
                Conv2d(self.resnet_filters[3], self.resnet_filters[3] , kernel_size=1, stride=1, padding=0, bias=False).to(device),
                BatchNorm2d(self.resnet_filters[3]).to(device)
            )
            self.resnet_conv4_2_id = nn.Sequential(
                        Conv2d(self.resnet_filters[3], self.resnet_filters[3] , kernel_size=1, stride=1, bias=False).to(device),
                        BatchNorm2d(self.resnet_filters[3] ).to(device)
                    )

            self.resnet_avgpool = AdaptiveAvgPool2d((1, 1)).to(device)
            self.resnet_classifier = nn.Sequential(
                Linear(in_features=self.resnet_filters[-1], out_features=beginModeling.n_components, bias=True).to(device),
                ReLU(inplace=True).to(device), # No negative #'s
                Linear(beginModeling.n_components, beginModeling.n_class).to(device)
            )
            

        """ FROM torch.Module (AUTOMATICALLY RUN) """
        def forward(self, xb):
            if (self.model_type1 == 'PCA' and self.model_type2 == 'LR') or (self.model_type1 == 'PCA' and self.model_type2 == 'SVC'):
                xb = xb.reshape(-1, beginModeling.n_components).to(device)
                xb = self.linear(xb)
                # if self.model_type2 == 'LR':
                #     out = F.softmax(xb, dim=1).to(device)
                # elif self.model_type2 == 'SVC':
                out = xb.to(device) 

            elif self.model_type2 == 'CNN_LR' or self.model_type2 == 'CNN_SVC':
                xb = torch.unsqueeze(xb, 1).to(device) #(batch_size, h, w) -> (batch_size, 1, h, w)
                xb = self.cnn_num_block(xb)
                xb = xb.view(xb.size(0), -1).to(device) # flatten
                xb = self.linear_num_block(xb)
                out = xb.to(device) # (batch_size, beginModeling.n_class)

            elif self.model_type2 == 'PIXEL_LR' or self.model_type2 == 'PIXEL_SVC':
                xb = xb.reshape(-1, 128*128).to(device)
                xb = self.pixel(xb)
                # if self.model_type2 == 'PIXEL_LR':
                
                #     out = F.softmax(xb, dim=1).to(device) # (batch_size, beginModeling.n_class)
                # elif self.model_type2 == 'PIXEL_SVC':
                out = xb.to(device) # (batch_size, beginModeling.n_class)

            elif self.model_type2 == 'CNN_ResNet' or self.model_type2 == 'CNN_AlexNet' or self.model_type2 == 'CNN_VGGNet':
                xb = xb.permute(0, 3, 1, 2) # (batch_size, w, h, c) -> (batch_size, c, w, h)
                if self.model_type2 == 'CNN_ResNet':
                    model = torchvision.models.resnet18(pretrained=False).to(device)
                    # for param in model.parameters():
                    #     param.requires_grad = False # Freezing weights.
                    num_infeat = model.fc.in_features
                    model.fc = Linear(num_infeat, beginModeling.n_class).to(device)
                elif self.model_type2 == 'CNN_AlexNet':
                    model = torchvision.models.alexnet(pretrained=False).to(device)
                    # print(model, '\n', model.classifier[-1], '\n')
                    num_infeat = model.classifier[-1].in_features
                    model.classifier[-1] = Linear(num_infeat, beginModeling.n_class).to(device)
                elif self.model_type2 == 'CNN_VGGNet':
                    model = torchvision.models.vgg16(pretrained=False).to(device)               
                    # print(model, '\n', model.classifier[-1], '\n')
                    num_infeat = model.classifier[-1].in_features
                    model.classifier[-1] = Linear(num_infeat, beginModeling.n_class).to(device)
                xb = model(xb)
                out = xb.to(device)
            
            elif self.model_type2 == 'CNN_AlexNet2' or self.model_type2 == 'CNN_AlexNet2_SVC':
                xb = torch.unsqueeze(xb, 1).to(device)
                xb = self.alexnet_features(xb)
                xb = self.alexnet_avgpool(xb)
                xb = torch.flatten(xb, 1).to(device) # batch 다음 걸로 flatten
                xb = self.alexnet_classifier(xb)
                out = xb.to(device)
                return out
            
            elif self.model_type2 == 'CNN_VggNet2' or self.model_type2 == 'CNN_VggNet2_SVC':
                xb = torch.unsqueeze(xb, 1).to(device)
                xb = self.vggnet_features(xb)
                xb = self.vggnet_avgpool(xb)
                xb = torch.flatten(xb, 1).to(device) # batch 다음 걸로 flatten
                xb = self.vggnet_classifier(xb)
                out = xb.to(device)
                return out

            elif model_type2 == 'CNN_ResNet2' or model_type2 == 'CNN_ResNet2_SVC':
                xb = torch.unsqueeze(xb, 1).to(device)
                # xb = self.ResNet(self.BasicBlock, [1, 1, 1, 1])(xb) # ResNet8: [1, 1, 1], ResNet10: [1, 1, 1, 1], ResNet18: [2, 2, 2, 2], ResNet34: [3, 4, 6, 3]
                # 윗줄은 아래 sub-class ResNet, BasicBlock, BottleNeck 이용.

                xb = self.resnet_conv(xb)

                # BLOCK 1
                xb1_1 = self.resnet_conv1(xb)
                xb1_2 = self.resnet_conv1_id(xb)
                assert xb1_1.shape == xb1_2.shape
                xb = xb1_1 + xb1_2
                xb = ReLU(inplace=True)(xb).to(device)

                xb1_3 = self.resnet_conv1(xb)
                xb1_4 = self.resnet_conv1_id(xb)
                assert xb1_3.shape == xb1_4.shape
                xb = xb1_3 + xb1_4
                xb = ReLU(inplace=True)(xb).to(device)
                ####################################

                # BLOCK 2
                xb2_1 = self.resnet_conv2(xb)
                xb2_2 = self.resnet_conv2_id(xb)
                assert xb2_1.shape == xb2_2.shape
                xb = xb2_1 + xb2_2
                xb = ReLU(inplace=True)(xb).to(device)

                xb2_3 = self.resnet_conv2_2(xb)
                xb2_4 = self.resnet_conv2_2_id(xb)
                assert xb2_3.shape == xb2_4.shape
                xb = xb2_3 + xb2_4
                xb = ReLU(inplace=True)(xb).to(device)
                ####################################

                # BLOCK 3
                xb3_1 = self.resnet_conv3(xb)
                xb3_2 = self.resnet_conv3_id(xb)
                assert xb3_1.shape == xb3_2.shape
                xb = xb3_1 + xb3_2
                xb = ReLU(inplace=True)(xb).to(device)

                xb3_3 = self.resnet_conv3_2(xb)
                xb3_4 = self.resnet_conv3_2_id(xb)
                assert xb3_3.shape == xb3_4.shape
                xb = xb3_3 + xb3_4
                xb = ReLU(inplace=True)(xb).to(device)
                ####################################

                # BLOCK 4
                xb4_1 = self.resnet_conv4(xb)
                xb4_2 = self.resnet_conv4_id(xb)
                assert xb4_1.shape == xb4_2.shape
                xb = xb4_1 + xb4_2
                xb = ReLU(inplace=True)(xb).to(device)

                xb4_3 = self.resnet_conv4_2(xb)
                xb4_4 = self.resnet_conv4_2_id(xb)
                assert xb4_3.shape == xb4_4.shape
                xb = xb4_3 + xb4_4
                xb = ReLU(inplace=True)(xb).to(device)
                ####################################

                xb = self.resnet_avgpool(xb)
                xb = torch.flatten(xb, 1).to(device) # batch 다음 걸로 flatten
                # x = x.view(x.size(0), -1).to(device) # (batch_size, c, h=1, w=1) -> (batch_size, c)
                xb = self.resnet_classifier(xb)
                out = xb.to(device)
                return out

            return out
        

        """ 2) FIT """
        def fit(self, epochs, lr, model, train_loader, val_loader, checkpoint_file, earlystop_file, opt_func=torch.optim.Adam):

            train_losses, valid_losses, avg_train_losses, avg_valid_losses = [], [], [], []
            train_accs, valid_accs, avg_train_accs, avg_valid_accs = [], [], [], []  

            optimizer = opt_func(model.parameters(), lr=lr, weight_decay=1e-4) # weight_decay is used to prevent overfitting (It keeps the weights small and avoid exploding gradients.)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.1, patience=5, min_lr=1e-8, verbose=False)
            early_stopping = pytorchtools.EarlyStopping(patience=20, verbose=True, path=earlystop_file) # early-stop한 모델의 모수가 저장된 파일                                                                                                       

            if os.path.isfile(checkpoint_file):
                checkpoint = torch.load(checkpoint_file)
                model.load_state_dict(checkpoint['model_state_dict']) # update the model where it left off
                optimizer.load_state_dict(checkpoint['optimizer_state_dict']) # update the optimizer where it left off
                past_epoch = checkpoint['epoch']
                avg_train_losses.append(checkpoint['avg_train_losses']) # 나중에 그래프 그릴 때 처음 정보부터 시각화하려고
                avg_valid_losses.append(checkpoint['avg_valid_losses'])
                avg_train_accs.append(checkpoint['avg_train_accs'])
                avg_valid_accs.append(checkpoint['avg_valid_accs'])
            else:
                past_epoch = 0

            if (past_epoch+1) < epochs+1:
                for epoch in range(past_epoch+1, epochs+1): # whew what a relief (that I do not have to start all over again every time I train the model)
                    # Training Phase
                    for i, (images, labels) in enumerate(train_loader):
                        images, labels = images.to(device), labels.to(device)
                        out = self(images) # forward-propagation (predict outputs)
                        if self.model_type2 == 'LR' or self.model_type2 == 'SVC' or self.model_type2 == 'CNN_LR' or self.model_type2 == 'PIXEL_LR' or self.model_type2 == 'CNN_ResNet' or model_type2 == 'CNN_ResNet2' or self.model_type2 == 'CNN_AlexNet' or self.model_type2 == 'CNN_AlexNet2' or model_type2 == 'CNN_VggNet2':
                            loss = F.cross_entropy(out, labels) # calculate the loss
                        elif self.model_type2 == 'CNN_SVC' or self.model_type2 == 'PIXEL_SVC' or self.model_type2 == 'CNN_AlexNet2_SVC' or model_type2 == 'CNN_ResNet2_SVC' or model_type2 == 'CNN_VggNet2_SVC':
                            loss = F.multi_margin_loss(out, labels) # calculate the loss
                        _, preds = torch.max(out, dim=1) # 한 batch 당 예측된 번호 사람들
                        acc = torch.tensor(torch.sum(preds==labels).item() / len(preds))

                        loss.backward() # back-propagation (compute gradients of the loss)
                        optimizer.step() # perform a single optimization step (parameter update)
                        optimizer.zero_grad() # clear the gradients of all optimized variables

                        train_losses.append(loss.detach()) # record training loss
                        train_accs.append(acc.detach())
                        #print(f'{i}th training loss: {loss}')

                    for i, (images, labels) in enumerate(val_loader):
                        images, labels = images.to(device), labels.to(device)
                        out = self(images)
                        if self.model_type2 == 'LR' or self.model_type2 == 'SVC' or self.model_type2 == 'CNN_LR' or self.model_type2 == 'PIXEL_LR' or self.model_type2 == 'CNN_ResNet' or model_type2 == 'CNN_ResNet2' or self.model_type2 == 'CNN_AlexNet' or self.model_type2 == 'CNN_AlexNet2' or model_type2 == 'CNN_VggNet2':
                            loss = F.cross_entropy(out, labels) # calculate the loss
                        elif self.model_type2 == 'CNN_SVC' or self.model_type2 == 'PIXEL_SVC' or self.model_type2 == 'CNN_AlexNet2_SVC' or model_type2 == 'CNN_ResNet2_SVC' or model_type2 == 'CNN_VggNet2_SVC':
                            loss = F.multi_margin_loss(out, labels) # calculate the loss
                        _, preds = torch.max(out, dim=1) # 
                        acc = torch.tensor(torch.sum(preds==labels).item() / len(preds))

                        valid_losses.append(loss.item()) # record validation loss
                        valid_accs.append(acc.detach())
                    
                    scheduler.step(loss) # This step should be called after validation.

                    train_loss = sum(train_losses) / len(train_losses)
                    valid_loss = sum(valid_losses) / len(valid_losses)
                    avg_train_losses.append(train_loss)
                    avg_valid_losses.append(valid_loss)

                    train_acc = sum(train_accs) / len(train_accs)
                    valid_acc = sum(valid_accs) / len(valid_accs)
                    avg_train_accs.append(train_acc)
                    avg_valid_accs.append(valid_acc)

                    epoch_len = len(str(epochs))

                    msg = (f'{epoch}/{epochs} train_loss: {train_loss:.4f} valid_loss: {valid_loss:.4f}')
                    print(msg)

                    torch.save({'epoch' : epoch,
                                'model_state_dict' : model.state_dict(),
                                'optimizer_state_dict' : optimizer.state_dict(),
                                'avg_train_losses' : avg_train_losses,
                                'avg_valid_losses' : avg_valid_losses,
                                'avg_train_accs' : avg_train_accs,
                                'avg_valid_accs' : avg_valid_accs}, checkpoint_file) # 마지막까지 업데이트된 모델의 모수 저장된 파일

                    train_losses, valid_losses = [], [] # clear lists to track next epoch

                    early_stopping(valid_loss, model)
                    if early_stopping.early_stop: # mypackages\\pytorchtools.py\\instance variable 중 하나
                        print('Early stopped.')
                        break # 다음 for문으로
      
            else:
                final_checkpoint = torch.load(earlystop_file) # checkpoint_file 아님
                model.load_state_dict(final_checkpoint)       

            return model, avg_train_losses, avg_valid_losses, avg_train_accs, avg_valid_accs

        """ 3) VISUALIZE """
        def visualize(self, avg_train_losses, avg_valid_losses, avg_train_accs, avg_valid_accs): 
            # list 안의 list 안의 list를 flatten 시키기 위해
            def flatten(x):
                if isinstance(x, collections.Iterable):
                    return [a for i in x for a in flatten(i)]
                else:
                    return [x]
                    

            avg_train_losses = flatten(avg_train_losses)
            avg_valid_losses = flatten(avg_valid_losses)
            avg_train_accs = flatten(avg_train_accs)
            avg_valid_accs = flatten(avg_valid_accs)

            # fig, acc_ax = plt.subplots()
            
            # acc_ax.plot(range(1, len(avg_train_losses)+1), avg_train_losses, label='Training Loss')
            # acc_ax.plot(range(1, len(avg_valid_losses)+1), avg_valid_losses, label='Valid Loss')
            # acc_ax.set_ylabel('Loss')
            # acc_ax.legend(loc='lower left')

            # acc_ax2 = acc_ax.twinx()
            # acc_ax2.plot(range(1, len(avg_train_accs)+1), avg_train_accs, label='Training Accuracy', color='green')
            # acc_ax2.plot(range(1, len(avg_valid_accs)+1), avg_valid_accs, label='Valid Accuracy', color='red')
            # acc_ax2.set_ylabel('Accuracy')
            # acc_ax2.legend(loc='upper left')

            # minpos = avg_valid_losses.index(min(avg_valid_losses)) + 1 # x축의 1에서부터 plot하니까
            # plt.axvline(minpos, linestyle='--', color='r', label='Early Stopping Checkpoint')
            # plt.legend(loc='center left')

            # plt.xlabel('Epochs')
            # plt.xlim(0, len(avg_train_losses)+1) # for a consistent scale
            # plt.grid(True)
            # plt.tight_layout()
            # plt.show()

    ### END of class LR_model.

    
    ##### (OPTIONAL) VISUALIZATION (PART 2)
    # def visualPredData(self, testdataset, mod):
    #     print("Prediction Result Samples: ")
    #     for i in range(1):
    #         img, label = testdataset[i]

    #         if self.model_type1 == 'PCA' or self.model_type2 == 'PIXEL_LR' or self.model_type2 == 'PIXEL_SVC':
    #             plt.eventplot(img, orientation='horizontal', colors='k')
    #             img = np.reshape(img, (1, -1)) # one-example
    #             if self.model_type2 == 'SVC2':
    #                 n_pred_face = mod.predict(img)
    #             elif self.model_type2 == 'LR' or self.model_type2 == 'SVC':
    #                 n_pred_face = mod(img)
    #         elif self.model_type2 == 'CNN_LR' or self.model_type2 == 'CNN_SVC':
    #             img = torch.unsqueeze(img, 0) #(h, w) -> (1, h, w) batch
    #             plt.imshow(img[0], cmap='gray')
    #             n_pred_face = mod(img) # one-hot encoded 되어서 나옴

    #         print('Face #', label, 'Predicted Face #', n_pred_face)
    #         plt.show()



    ##### VISUALIZATION (PART 3)
    def ready4Visualization(self, ytest, yfit, yprob, file_path_list, old_uniq_labels, unique_labels, y_test_oh):
        """ High Analysis """
        if os.path.isfile(self.high_csv_file): # If there IS a saved high_csv file...
            pass
        else:
            df_high = pd.DataFrame()
            # if self.model_type2 != 'SVC' and self.model_type2 != 'SVC2' and self.model_type2 != 'CNN_SVC' and self.model_type2 != 'PIXEL_SVC' and self.model_type2 != 'CNN_AlexNet2_SVC' and model_type2 != 'CNN_ResNet2_SVC' and model_type2 != 'CNN_VggNet2_SVC':
            for i in range(yfit.shape[0]):
                idx = old_uniq_labels.index(int(ytest[i])) # 사람 폴더 위치(0, 1, ..., beginModeling.n_class-1)
                actual_prob = yprob[i][idx]
                pred_prob = np.max(yprob[i], axis=-1)
                if yfit[i] == ytest[i]:
                    ans = 'correct'
                else:
                    ans = 'wrong'
                chance_prob = 1 / beginModeling.n_class
                if ((chance_prob+1)/2 < pred_prob) and (pred_prob <= 1):
                    conf_level = 1
                elif (0 < pred_prob) and (pred_prob <= (chance_prob+1)/2):
                    conf_level = 0
                else:
                    conf_level = None
                # _, _, _, _, _, folder, file_name_list = file_path_list[i].split('\\')
                _, _, _, folder, file_name_list = file_path_list[i].split('\\')
                new_data = {
                            'file_name' : str(file_name_list), # 저장 3
                            'correctness' : ans, # 저장 4
                            'pred_prob_vector' : yprob[i], # 저장 5
                            'actual_person' : ytest[i], # 저장 6
                            'actual_prob' : actual_prob, # 저장 7
                            'pred_person' : yfit[i], # 저장 8
                            'pred_prob' : pred_prob, # 저장 9
                            'conf_level' : conf_level # 저장 10
                            }
                df_high = df_high.append(new_data, ignore_index=True)  
            df_high = df_high[['file_name', 'correctness', 'pred_prob_vector', 'actual_person', 'actual_prob', 'pred_person', 'pred_prob', 'conf_level']] # 열 순서 재배치     
            # else: # SVC일 땐 확률 생성X
            #     for i in range(yfit.shape[0]):
            #         idx = old_uniq_labels.index(int(ytest[i])) # 사람 폴더 위치(0, 1, ..., beginModeling.n_class-1)
            #         if yfit[i] == ytest[i]:
            #             ans = 'correct'
            #         else:
            #             ans = 'wrong'
            #         new_data = {
            #                     'file_name' : os.path.basename(file_path_list[i]), # 저장 3
            #                     'correctness' : ans, # 저장 4
            #                     'actual_person' : ytest[i], # 저장 6
            #                     'pred_person' : yfit[i], # 저장 8
            #                     }
            #         df_high = df_high.append(new_data, ignore_index=True)  
            #     df_high = df_high[['file_name', 'correctness', 'actual_person',
            #                     'pred_person']] # 열 

            print(df_high.groupby(df_high['correctness']).count()) # 얼마나 맞췄는지 미리 보기 위하여   
            df_high.to_csv(self.high_csv_file, index=False) # 저장 1, 2

        #people_ids = list(set(ytest)) # unique list
        # """ Classification Report """
        # if os.path.isfile(self.low_csv_file): # If there IS a saved high_csv file...
        #     pass
        # else:
        #     c_report = classification_report(ytest, yfit, output_dict=True)
        #     #print('Classification Report: ', '\n', c_report)
        #     c_report_df = pd.DataFrame(c_report).transpose()
        #     c_report_df.to_csv(self.low_csv_file, index=True) # 저장 1, 2

        #""" Confusion Matrix """
        #if os.path.isfile(self.hm_file): # If there IS a saved high_csv file...
        #    pass
        #else:
        #    mat = confusion_matrix(ytest, yfit)
        #    plt.figure(figsize=(7,7))
        #    sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
        #                xticklabels=old_uniq_labels,
        #                yticklabels=old_uniq_labels)
        #    plt.xlabel('True Labels')
        #    plt.ylabel('Predicted Labels')
        #    plt.title(f'{self.model_type1}_{self.model_type2}_{old_uniq_labels}')
        #    plt.savefig(self.hm_file)
        #    plt.show()

        # """ Receiviing Operatinc Characteristic Curve"""
        # if self.model_type2 != 'SVC' and self.model_type2 != 'SVC2' and self.model_type2 != 'CNN_SVC' and self.model_type2 != 'PIXEL_SVC' and self.model_type2 != 'CNN_AlexNet2_SVC' and model_type2 != 'CNN_ResNet2_SVC' and model_type2 == 'CNN_VggNet2_SVC':
        #     if os.path.isfile(self.auc_info_file):
        #         pass
        #     else:
        #         y_test_oh = y_test_oh # (# of training samples, beginModeling.n_class)
        #         y_pred_prob = yprob # (# of training samples, beginModeling.n_class)
        #         print('Shape of y_test_oh:', y_test_oh.shape,'Shape of y_pred_prob:', y_pred_prob.shape)

        #         auc_value = roc_auc_score(y_true=y_test_oh, y_score=y_pred_prob, average='micro', multi_class='ovr') # 'ovr': One-vs-rest, 'ovo: One-vs-one, average=None하면 각 class별 점수가 출력됨 
        #         print('(MICRO) ROC-AUC = %.4f'%(auc_value)) # micro_precision = Aver(all roc_auc_score)
        #         auc_value = roc_auc_score(y_true=y_test_oh, y_score=y_pred_prob, average='macro', multi_class='ovr') # 'ovr': One-vs-rest, 'ovo: One-vs-one, average=None하면 각 class별 점수가 출력됨 
        #         print('(MACRO) ROC-AUC = %.4f'%(auc_value)) # macro_precision = Sum(TP_i) /(Sum(TP_i) + Sum(FP_i)))

        #         # MICRO ROC-AUC
        #         fpr_dict, tpr_dict, auc_dict = {}, {}, {} # increasing fpr (or tpr) s.t. element i is the fpr (or tpr) of predictions with score >= thresholds[i]
        #         for i in range(beginModeling.n_class):
        #             fpr, tpr, _ = roc_curve(y_true=y_test_oh[:,i], y_score=y_pred_prob[:,i])
        #             fpr_dict[i], tpr_dict[i] = fpr, tpr
        #             roc_auc = sklearn.metrics.auc(fpr, tpr)
        #             auc_dict[i] = roc_auc
        #         fpr, tpr, _ = roc_curve(y_test_oh.ravel(), y_pred_prob.ravel())
        #         fpr_dict['micro'], tpr_dict['micro'] = fpr, tpr
        #         roc_auc = sklearn.metrics.auc(fpr, tpr)
        #         auc_dict['micro'] = roc_auc

        #         # MACRO ROC-AUC
        #         all_fpr = list(fpr_dict[i] for i in range(beginModeling.n_class))
        #         all_fpr = np.unique(np.concatenate(all_fpr))
        #         mean_tpr = np.zeros_like(all_fpr) # 전부 다 0(초기화)
        #         for i in range(beginModeling.n_class):
        #             mean_tpr += np.interp(all_fpr, fpr_dict[i], tpr_dict[i]) # the x-coordinate at which to evaluate the interpolated values, xp, fp (length: beginModeling.n_class)
        #         mean_tpr /= beginModeling.n_class # update mean_tpr
        #         fpr, tpr = all_fpr, mean_tpr
        #         fpr_dict['macro'], tpr_dict['macro'] = fpr, tpr
        #         roc_auc = sklearn.metrics.auc(fpr, tpr)
        #         auc_dict['macro'] = roc_auc

        #         plt.figure(figsize=(7, 7))

        #         plt.plot(fpr_dict['micro'], tpr_dict['micro'],
        #                 color='salmon', linestyle=':', linewidth=2,
        #                 label='Micro: {0:0.3f}'.format(auc_dict['micro']))
        #         plt.plot(fpr_dict['macro'], tpr_dict['macro'],
        #                 color='tomato', linestyle='-.', linewidth=2,
        #                 label='Macro: {0:0.3f}'.format(auc_dict['macro']))

        #         color_list = sns.color_palette('hls', beginModeling.n_class)
        #         for i, color in zip(range(beginModeling.n_class), color_list):
        #             plt.plot(fpr_dict[i], tpr_dict[i],
        #                     color=color, lw=1,
        #                     label='Class {0}: {1:0.3f}'.format(i, auc_dict[i]))
        #         plt.plot([0,1], [0,1], color='whitesmoke', lw=1, linestyle='--') # base-line

        #         plt.xlabel('False Positive Rate', fontsize=12) # 실제 negative들 중에서 positive로 예측한 비율(1-specificity)
        #         plt.ylabel('True Positive Rate', fontsize=12) # 실제 positive들 중에서 positive로 예측한 비율(sensitivity=recall)
        #         plt.legend(loc='lower right') # bbox_to_anchor=(1,1): legend를 plot 바깥에 위치하고 싶을 때 사용 
        #         # plt.savefig(self.roc_file)
        #         plt.show()
                
        #         # micro, macro만 저장(용량 최소화, 나중에 모델별로 비교하기 위해)
        #         sel_fpr_dict = {sel_k:fpr_dict[sel_k] for sel_k in ['micro', 'macro']}
        #         sel_tpr_dict = {sel_k:tpr_dict[sel_k] for sel_k in ['micro', 'macro']}
        #         sel_auc_dict = {sel_k:auc_dict[sel_k] for sel_k in ['micro', 'macro']}

        #         fpr_df, tpr_df, auc_df = pd.DataFrame.from_dict(sel_fpr_dict, orient='index'), pd.DataFrame.from_dict(sel_tpr_dict, orient='index'), pd.DataFrame.from_dict(sel_auc_dict, orient='index')
        #         fpr_df, tpr_df, auc_df = fpr_df.transpose(), tpr_df.transpose(), auc_df.transpose()
        #         final_df = pd.concat([fpr_df, tpr_df, auc_df], axis=1) # 열 덧붙이기
        #         # final_df.to_csv(self.auc_info_file, index=False) # index 굳이 필요 없음
        #         print('FP_TP dictionary for every face', final_df.head())