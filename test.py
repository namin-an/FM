import datetime
# date = datetime.datetime.now()
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

from mypackages import pytorchtools 
from loadData import loadData, visualizeData
from trainANNs import beginModeling


if __name__ == 'main':
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_type', type=str, default='opt') # 'opt' or 'elec' or 'normal'
    parser.add_argument('--model_type1', type=str, default='opt') # 'PCA', 'PCA', 'PCA', '', '', '', '', '', '', '', '', '', ''
    parser.add_argument('--model_type2', type=str, default='') # 'SVC2'(Old version) 'SVC', 'LR', 'CNN_LR', 'CNN_SVC', 'PIXEL_LR', 'PIXEL_SVC',
    # 'CNN_ResNet', 'CNN_ResNet2', CNN_ResNet2_SVC', 'CNN_AlexNet', 'CNN_AlexNet2', 'CNN_AlexNet2_SVC', 'CNN_VggNet2', 'CNN_VggNet2_SVC'
    parser.add_argument('--xai', type=str, default='no') # 'yes', 'no'
    parser.add_argument('--finetune', type=str, default='') # 'ft', ''
    parser.add_argument('--r', type=int, default=4) # 2, 4, 16
    parser.add_argument('--meta_path', type=str, default='C:\\Users\\user\\Desktop\\210827_ANNA_Removing_uncontaminated_data.csv')
    parser.add_argument('--train_path', type=str, default='E:\\ANNA_INTERN\\Middle_Resolution_137_unzipped_parcropped_128_removed_train')
    parser.add_argument('--test_path', type=str, default='E:\\ANNA_INTERN\\Middle_Resolution_137_unzipped_parcropped_128')
    args = parser.parse_args()

    model_type = args.model_type1 + args.model_type2
    df = pd.read_csv(args.meta_path)

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


    if args.test_type == 'opt' or args.test_type == 'elec':
        test_path =  f'{args.test_path}_{args.test_type}'
    elif args.test_type == 'normal':
        test_path = f'{args.train_path}'

    # test images
    camera_list = ['4','7','10'] #4(C)
    light_list = ['1'] #2(L)
    accessory_list =['1'] #1(S)
    expression_list = ['1','2','3'] #3(E)

    # train images
    tr_camera_list = list(map(str, [4,5,6,7,8,9,10,14,15,16,17,18,19,20])) #4(C)
    tr_light_list = list(map(str, [1,2,3,4,5,6, # 7: all black
        8,9,10,11,12,13,14,15,16,17,18,
        19,20,21,22,23,24,25,26,27,28,29,30])) #2(L)
    tr_accessory_list =['1'] #1(S)
    tr_expression_list = ['1','2','3'] #3(E)

    if args.model_type1 == 'PCA':
        ext_name1 = 'gz'
    elif args.model_type1 == '':
        ext_name1 = ''

    if args.model_type2 == 'SVC2':
        ext_name2 = 'gz'
    elif args.model_type2 == 'LR' or args.model_type2 == 'SVC' or args.model_type2 == 'CNN_LR' or args.model_type2 == 'CNN_SVC' or args.model_type2 == 'PIXEL_LR' or args.model_type2 == 'PIXEL_SVC' or args.model_type2 == 'CNN_ResNet' or args.model_type2 == 'CNN_ResNet2' or args.model_type2 == 'CNN_ResNet2_SVC' or args.model_type2 == 'CNN_AlexNet' or args.model_type2 == 'CNN_AlexNet2' or args.model_type2 == 'CNN_AlexNet2_SVC' or args.model_type2 == 'CNN_VggNet2' or args.model_type2 == 'CNN_VggNet2_SVC':
        ext_name2 = 'pt'

    seed_list = [22, 77, 2, 100, 81, 42, 7, 1, 55, 50] # different 1,000 training images (total of 10 permutations)


    for m in range(14): # 14 iid set
        input_folder = [df.iloc[i, 0] for i in sets[m]] 
        assert len(input_folder) == 16
        com_obj = itertools.combinations(input_folder, r)
        com_list = list(com_obj)
        
        for seed in seed_list:
            for n in range(len(com_list)): # for every combination of question set

                data_path = f'E:\\ANNA_INTERN\\Question Banks AI Hub_final\\{args.r}classes\\set{m}\\seed{seed}'
                preprocessed_data_path =  os.path.join(data_path, f'comb{n}')

                model_path = os.path.join(preprocessed_data_path, 'Saved_Models')
                high_analysis_path = os.path.join(preprocessed_data_path, f'High_Analysis_{args.test_type}')
                low_analysis_path = os.path.join(preprocessed_data_path, f'Low_Analysis_{args.test_type}')

                os.makedirs(data_path, exist_ok=True)
                os.makedirs(preprocessed_data_path, exist_ok=True) 
                os.makedirs(model_path, exist_ok=True)
                os.makedirs(high_analysis_path, exist_ok=True)

                model_file1 = os.path.join(model_path, f'Model_{args.model_type1}{args.finetune}.{ext_name1}')
                model_file2 = os.path.join(model_path, f'Model_{args.model_type2}{args.finetune}.{ext_name2}')
                checkpoint_file = os.path.join(model_path, f'Checkpoint_{args.model_type2}{args.finetune}.{ext_name2}') # for PCA_LR, CNN_LR, CNN_SVC
                earlystop_file = os.path.join(model_path, f'Early_Stop_{args.model_type2}{args.finetune}.{ext_name2}') # for PCA_LR, CNN_LR, CNN_SVC
                high_csv_file = os.path.join(high_analysis_path, f'High_Level_Data_Analysis_{args.model_type1}_{args.model_type2}{args.finetune}.csv')
                
                low_csv_file = os.path.join(low_analysis_path, f'Classification Report_{args.model_type1}_{args.model_type2}.csv') # 안 씀 
                auc_info_file = os.path.join(low_analysis_path, f'TP_FP_AUC_Dictionary_{args.model_type1}_{args.model_type2}.csv') # 안 씀
                roc_file = os.path.join(low_analysis_path, f'ROC_{args.model_type1}_{args.model_type2}.png') # 안 씀
                check_file = high_csv_file

                if os.path.isfile(check_file) and args.xai == 'no':
                    pass
                else:
                    print(f'\n START {m}th set {n}th comb (seed{seed}): comb{com_list[n]} \n')
                    """1. Loading a data"""
                    Xtrain, ytrain, _, old_uniq_labels, tr_unique_items = loadData(com_list[n], 'train') 
                    Xtest, ytest, file_path_list, old_uniq_labels2, test_unique_items = loadData(com_list[n], 'test')
                    assert set(old_uniq_labels) == set(old_uniq_labels2)
                    assert set(tr_unique_items) == set(test_unique_items)
                    unique_labels = tr_unique_items

                    """2. Modeling"""
                    instance = beginModeling(n, args.model_type1, args.model_type2, Xtrain, ytrain, Xtest, ytest, unique_labels, model_file1, model_file2, high_csv_file, low_csv_file, auc_info_file, checkpoint_file, earlystop_file, roc_file) # INITIALIZATION

                    model, Xtrain, Xtest = instance.loadOrSaveModel1() # FEATURE EXTRACTION (PART 1)
                    train_loader, val_loader, test_loader = instance.convertAndVisualData(model, Xtrain, Xtest, ytrain, ytest) # (OPTIONAL) VISUALIZATION 1|
                    ytest, yfit, yprob, mod, y_test_oh= instance.loadOrSaveModel2andEval(train_loader, val_loader, test_loader, Xtrain, Xtest, old_uniq_labels2, file_path_list) # CLASSIFICATION (PART 2)
                    # instance.visualPredData(testdataset, mod) # (OPTIONAL) VISUALIZATION 2
                    instance.ready4Visualization(ytest, yfit, yprob, file_path_list, old_uniq_labels2, unique_labels, y_test_oh) # VISUALIZATION 3
