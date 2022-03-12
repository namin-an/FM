"""
Running this file trains ANNs described in the paper.

*Preview of this code:
- Packages

- Loading human and ANN data
- Fig. 4a (Fig. S8a, Fig. S8b)
- Fig. 4b
"""



#%%
# Packages

import datetime
date = datetime.datetime.now()
print(f'Today is Happy{date: %A, %d, %m, %Y}.', '\n')

import os
import math
import random 
import itertools 

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from matplotlib.lines import Line2D

import scipy
import scipy as sp
from scipy import stats
from scipy.stats import norm, wilcoxon, linregress # norm.cdf와 norm.ppf(percent point function, inverse of cdf-percentiles)은 역함수 관계
import scipy.stats as sp

from statannot import add_stat_annotation
from PIL import Image
import cv2 as cv
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score, classification_report
from sklearn.utils import shuffle



#%%
# Loading human and ANN data

# Button
type = 'opt' # 'opt' or 'elec'
test_type = type

# 사람 데이터
human_data = 'E:\\ANNA_INTERN\\Human_Exp\\211202'
# human_data = 'C:\\Users\\user\\Desktop\\Human_Exp\\211124'
# human_data = 'C:\\Users\\user\\Desktop\\Human_Exp\\211118'
# human_data = '/content/drive/MyDrive/Human_Exp/211118'
# sel_ppl = [92, 93] # 정현, 서연

if type == 'opt':
  sel_ppl = list(range(300, 309)) + list(range(400, 408)) + [611] # 18개
elif type == 'elec': 
  # sel_ppl = [499, 500, 502] + list(range(504, 509)) + list(range(602, 606)) + list(range(608, 612)) # 16개 (잘한 남자랑 못한 여자 제거)
  sel_ppl = [499, 500, 502] + list(range(503, 509)) + list(range(602, 607)) + list(range(608, 612)) # 18개

human_df = pd.DataFrame()
n = 9
for i in range(1, 80*n+1, 80):
    try:
      # temp_df = pd.read_csv(f'C:\\Users\\user\\Desktop\\211108_ANNA_main_test_{i}.csv') 
      # temp_df = pd.read_csv(os.path.join(human_data, f'211116_ANNA_main_test_{i}.csv'))
      j = i+79
      temp_df = pd.read_csv(os.path.join(human_data, f'main_test({i}_{j}).xls.csv'))
      if i == 1:
        pass
      else:
        temp_df = temp_df.rename(columns = {'유저식별아이디':'useless', 'MC구분':'useless', '성별':'useless', '나이':'useless', '학력':'useless'})
      human_df = pd.concat([human_df, temp_df], axis=1)
    except:
      print(i)

# temp_list = [f'선택_A_{p}' for p in range(i, n+1) ]
# human_df = human_df[['유저식별아이디', '나이', '성별', '학력', '시력', *temp_list]]
human_df = human_df[human_df['유저식별아이디'].isin(sel_ppl)]
# human_df = human_df[human_df.index.isin([34, 37, 38])] # 윤서, 세인, 나민
# new_sel_cols = [col for col in human_df.columns if col != 503 or col!= 506] # 결측치 제거
# human_df = human_df[new_sel_cols]
orig_human_df = human_df
human_df = human_df.fillna(0)


sel_col = []
for j in range(1, 80*n+1):
    temp_str = f'선택_A_{j}'
    sel_col.append(temp_str)

human_df = human_df[sel_col]
human_df.index = sel_ppl 
human_df.columns = list(range(80*n))

# 결측치(0) 확인 (x: 사람아이디 번호, 즉, 0이면 대답 안했거나 못했다는 의미)
plt.hist(human_df.values, density=True)
plt.show()

# 머신 데이터 
# answer_df = pd.read_csv(f'C:\\Users\\user\\Documents\\Namin\\210930_MCs_for_dev.csv')
answer_df = pd.read_csv(f'E:\\ANNA_INTERN\\Human_Exp\\211105_QAs_for_Set0_CNN_SVC_4classes_partial.csv')

act_per_list, pix_list, gs_list, par_list = [], [], [], []
for answer in answer_df['Answer']:
    img, _ = answer.split('.jpg')
    act_per, pix, gs, par = img.split('_')
    
    act_per_list.append(act_per)
    pix_list.append(pix)
    gs_list.append(gs)
    par_list.append(par)

answer_df['act_per'] = act_per_list
answer_df['PIX'] = pix_list
answer_df['GS'] = gs_list
answer_df['par'] = par_list

answer_df = answer_df[:80*n]
answer_df = answer_df.T

orig_answer_df = answer_df


# 사람과 머신 데이터
# human_df이랑 answer_df 합치기

mer_df = pd.concat([human_df, answer_df], axis=0)
mer_df = mer_df.T
mer_df = mer_df.fillna(0)
orig_mer_df = mer_df

mer_df

#%%
# Fig. 4a (Fig. S8a, Fig. S8b)
# 여러가지 머신러닝 모델들의 성능

test_type_list = [test_type] #['opt', 'elec']
model_type1_list = ['', '', '', '', '', '', '', '', 'PCA', 'PCA'] #['PCA', 'PCA', '', '']
model_type2_list = ['PIXEL_LR','PIXEL_SVC', 'CNN_LR', 'CNN_SVC', 'CNN_VggNet2', 'CNN_VggNet2_SVC', 'CNN_AlexNet2', 'CNN_AlexNet2_SVC', 'SVC', 'LR']
seed_list = [22, 77, 2, 100, 81, 42, 7, 1, 55, 50] # different 1,000 training images (total of 10 permutations)

for test_type in test_type_list:
    for (model_type1, model_type2) in zip(model_type1_list, model_type2_list):
        model_type = model_type1 + model_type2

        high_df = pd.DataFrame()
        for m in range(1): # 제일 첫 번째 set에 대해서만
            for seed in seed_list:
                data_path = f'E:\\ANNA_INTERN\\Question Banks AI Hub_final\\16classes\\set{m}\\seed{seed}'
                preprocessed_data_path =  os.path.join(data_path, 'comb0') # 16 classes 는 1 comb 밖에 없음.

                high_analysis_path = os.path.join(preprocessed_data_path, f'High_Analysis_{test_type}')
                try:
                    add_high_df = pd.read_csv(os.path.join(high_analysis_path, f'High_Level_Data_Analysis_{model_type1}_{model_type2}.csv'))
                    add_high_df['hit_rate'] = add_high_df['correctness'].replace(['correct', 'wrong'], [1, 0]) 
                    high_df = pd.concat([high_df, add_high_df], axis=0)
                    high_df = high_df[['file_name', 'actual_person', 'hit_rate']]
                except:
                    print(model_type, seed, 'error')
                    pass
                
        image_list = []
        for i, file_name in enumerate(high_df['file_name']):
            face = high_df['actual_person'].iloc[i]
            face = str(int(float(face)))
            image_list.append(f'{face}_{file_name}') # for img (scatter-plot of two probs)
        high_df['image'] = image_list

        high_df = high_df[['image', 'hit_rate']]

        old_high_df_cols = high_df.columns
        new_high_df_cols = [f'{model_type}_' + col_name for (i, col_name) in enumerate(old_high_df_cols) if i != 0] # file_name 제외하고 나머지에 모델 이름 앞에 달아주기
        new_high_df_cols = [old_high_df_cols.tolist()[0], *new_high_df_cols]
        high_df.columns = new_high_df_cols

        high_df = pd.DataFrame(high_df.groupby('image').mean().reset_index()) # reset_index to bring 'file_name' to one of the columns.

        if model_type == 'PCALR':
            pca_lr_high_df = high_df
        elif model_type == 'PCASVC':
            pca_svc_high_df = high_df
        elif model_type == 'PCASVC2':
            pca_svc2_high_df = high_df
        elif model_type == 'CNN_LR':
            cnn_lr_high_df = high_df
        elif model_type == 'CNN_SVC':
            cnn_svc_high_df = high_df
        elif model_type == 'CNN_VggNet2':
            cnn_vgg_high_df = high_df
        elif model_type == 'CNN_VggNet2_SVC':
            cnn_vgg_svc_high_df = high_df
        elif model_type == 'CNN_AlexNet2':
            cnn_alexnet_high_df = high_df
        elif model_type == 'CNN_AlexNet2_SVC':
            cnn_alexnet_svc_high_df = high_df
        elif model_type == 'PIXEL_SVC':
            pixel_svc_high_df = high_df
        elif model_type == 'PIXEL_LR':
            pixel_lr_high_df = high_df

# 나중에 시각화할 순서
model_list = [
              pixel_svc_high_df, pixel_lr_high_df,
              pca_svc_high_df, pca_lr_high_df,
              cnn_svc_high_df, cnn_lr_high_df,
              cnn_alexnet_svc_high_df, cnn_alexnet_high_df,
              cnn_vgg_svc_high_df, cnn_vgg_high_df]
model_name_list = ['PIXEL_SVC', 'PIXEL_LR',
                   'PCASVC', 'PCALR',
                   'CNN_SVC', 'CNN_LR',
                   'CNN_AlexNet2_SVC', 'CNN_AlexNet2',
                   'CNN_VggNet2_SVC', 'CNN_VggNet2'] # 위와 순서 일치해야 함.
temp_high_df = model_list[0].merge(model_list[1], on='image')
for i in range(2, len(model_list)):
    temp_high_df = temp_high_df.merge(model_list[i], on='image')

temp_high_df = temp_high_df.drop(temp_high_df.tail(1).index)
temp_high_df = temp_high_df.set_index('image')
temp_high_df.columns = model_name_list

# Archetypal 사람들의 성능 + 여러가지 머신러닝 모델들의 성능

temp_acc_df = mer_df3.copy()

archetypal_df = temp_acc_df[sel_ppl]
# archetypal_df = temp_acc_df[sel_ppl].mean(axis=1).to_frame()
# archetypal_df.columns = ['Human']

merged_df = pd.merge(archetypal_df, temp_high_df, left_index=True, right_index=True, how='inner')

chosen_hyperpar = ''

# Face-level

temp_index = []
pix_list = []
h_list = []

for file_name in merged_df.index:
    split = file_name.split('.')
    face, pix, gs, _ = split[0].split('_')
    temp_index.append(face)
    pix_list.append(pix)
    h_list.append(f'{pix}_{gs}')

merged_df['face'] = temp_index
merged_df['PIX'] = pix_list
merged_df['hyperpar'] = h_list
#merged_df = merged_df[merged_df['PIX'] == '64PIX']
if chosen_hyperpar != '':
    new_merged_df = merged_df[merged_df['hyperpar'] == chosen_hyperpar]
else:
    new_merged_df = merged_df

temp_acc_df = new_merged_df.groupby('face').mean()

# --------------- 사람 성능 순서대로 사진 배열 ---------------
temp_df = temp_acc_df.sort_values([sel_ppl[0]]) # 16_hit_rate를 기준으로 difficult(성능 낮은)한 순서부터 easy(성능 높은) face ID 나열

color_index_dict = {k:[] for k in range(1, temp_df.shape[1]+1)}
for (t, col) in enumerate(temp_df.columns):
    temp_col = temp_df[col]
    # color_index_dict[t+1] = sorted(range(len(temp_col)), key = lambda k:temp_col[k])
    color_index_dict[t+1] = temp_col.values

num_class = temp_acc_df.shape[0] # 16

fig, ax = plt.subplots(num_class, temp_df.shape[1]+1, figsize=(2*(temp_df.shape[1]+1), 2*num_class))
for i, person in enumerate(temp_df.index):
    person = int(person)
    if chosen_hyperpar != '':
        filePath = f'E:\\ANNA_INTERN\\Middle_Resolution_137_unzipped_parcropped_128_{test_type}\\{person}\\{chosen_hyperpar}_S001L1E01C7.jpg'
    else:
        filePath = f'E:\\ANNA_INTERN\\Middle_Resolution_137_unzipped_parcropped_128_{test_type}\\{person}\\32PIX_4GS_S001L1E01C7.jpg'
    if os.path.isfile(filePath):
        for j in range(temp_df.shape[1]+1):
            # ax[i].set_title(merged_df.columns[j+1])
            if j == 0:
                img = cv.imread(filePath, cv.IMREAD_GRAYSCALE)
                ax[i][0].imshow(np.asarray(img), cmap='gray')
                ax[i][0].set_xticks([])
                ax[i][0].set_yticks([])
                ax[i][0].set_aspect('equal')
            else:
                # flipped_color = int(color_index_dict[j][i]) # j는 model의 개수이므로 dataframe의 column에 해당, i는 face-class의 개수이므로 dataframe의 index에 해당
                # color = num_class-1 - flipped_color
                # color = sns.color_palette('Spectral', num_class)[color]
                
                hit_rate = color_index_dict[j][i]
                assert hit_rate >= 0 and hit_rate <= 1
                num_val = 1000
                # conv_hr = (max(temp_df.max()) - 1) + (hit_rate - min(temp_df.min())) / (max(temp_df.max()) - min(temp_df.min()))  # M-1(빨) ~ M(파)
                # conv_hr = min(temp_df.min()) + (hit_rate - min(temp_df.min())) / (max(temp_df.max()) - min(temp_df.min())) # m(빨) ~ m+1(파)
                # conv_hr = (hit_rate - min(temp_df.min())) / (max(temp_df.max()) - min(temp_df.min()))# m(빨) ~ M(파) # 네모마다 상대적
                conv_hr = hit_rate # 0(빨) ~ 1(파) # 절대적
                conv_hr = conv_hr * (num_val - 1) # num_val 등분
                conv_hr = round(conv_hr)

                # if conv_hr < 0:
                #     new_conv_hr = 0
                # elif conv_hr >= num_val:
                #     new_conv_hr = num_val-1
                # else:
                new_conv_hr = conv_hr
                color = sns.color_palette('Spectral', num_val)[new_conv_hr] 
                
                ax[i][j].set_facecolor(color)
                ax[i][j].set_xticks([])
                ax[i][j].set_yticks([])
                ax[i][j].set_aspect('equal')
    else:
        print('The file does not exist.')
fig.subplots_adjust(wspace=0, hspace=0)
plt.savefig(f'C:\\Users\\user\\Desktop\\face_{test_type}_{chosen_hyperpar}.png')
#plt.show()
# --------------------------------------------

temp_acc_df.std()

# Face-attribute level

temp_index = []
for file_name in merged_df.index:
    split = file_name.split('.')
    face, _, _, par = split[0].split('_')
    temp_index.append(face + '_' + par)

merged_df['facepar'] = temp_index

temp_acc_df = merged_df.groupby('facepar').mean()

# --------------- 사람 성능 순서대로 사진 배열 ---------------
temp_df = temp_acc_df.sort_values([sel_ppl[0]]) # 16_hit_rate를 기준으로 difficult(성능 낮은)한 순서부터 easy(성능 높은) face ID 나열

color_index_dict = {k:[] for k in range(1, temp_df.shape[1]+1)}
for (t, col) in enumerate(temp_df.columns):
    temp_col = temp_df[col]
    # color_index_dict[t+1] = sorted(range(len(temp_col)), key = lambda k:temp_col[k])
    color_index_dict[t+1] = temp_col.values

num_class = temp_acc_df.shape[0] # 16*5

fig, ax = plt.subplots(num_class, temp_df.shape[1]+1, figsize=(0.1*2*(temp_df.shape[1]+1), 0.1*2*num_class)) # multiply 5 b/c there are 5 face-attributes per one face-class

for i, perpar in enumerate(temp_df.index):
    person, par = perpar.split('_')
    person=int(person)
    filePath = f'E:\\ANNA_INTERN\\Middle_Resolution_137_unzipped_parcropped_128_{test_type}\\{person}\\32PIX_4GS_{par}.jpg'
    if os.path.isfile(filePath):
        for j in range(temp_df.shape[1]+1):
            # ax[i].set_title(merged_df.columns[j+1])
            if j == 0:
                img = cv.imread(filePath, cv.IMREAD_GRAYSCALE)
                #img = cv.resize(img, (16, 16))
                ax[i][0].imshow(np.asarray(img), cmap='gray')
                ax[i][0].set_xticks([])
                ax[i][0].set_yticks([])
                ax[i][0].set_aspect('equal')
            else:
                # flipped_color = int(color_index_dict[j][i]) # j는 model의 개수이므로 dataframe의 column에 해당, i는 face-class의 개수이므로 dataframe의 index에 해당
                # color = num_class-1 - flipped_color
                # color = sns.color_palette('Spectral', num_class)[color]
                
                hit_rate = color_index_dict[j][i]
                num_val = 1000
                # conv_hr = (max(temp_df.max()) - 1) + (hit_rate - min(temp_df.min())) / (max(temp_df.max()) - min(temp_df.min()))  # M-1(빨) ~ M(파)
                # conv_hr = min(temp_df.min()) + (hit_rate - min(temp_df.min())) / (max(temp_df.max()) - min(temp_df.min())) # m(빨) ~ m+1(파)
                assert hit_rate >= 0 and hit_rate <= 1
                # conv_hr = (hit_rate - min(temp_df.min())) / (max(temp_df.max()) - min(temp_df.min()))# m(빨) ~ M(파)
                conv_hr = hit_rate # 0(빨) ~ 1(파) # 절대적
                conv_hr = conv_hr * (num_val - 1) # num_val 등분
                conv_hr = round(conv_hr)

                # if conv_hr < 0:
                #     new_conv_hr = 0
                # elif conv_hr >= num_val:
                #     new_conv_hr = num_val-1
                # else:
                new_conv_hr = conv_hr
                color = sns.color_palette('Spectral', num_val)[new_conv_hr] 
                
                ax[i][j].set_facecolor(color)
                ax[i][j].set_xticks([])
                ax[i][j].set_yticks([])
                ax[i][j].set_aspect('equal')
    else:
        print('The file does not exist.')
fig.subplots_adjust(wspace=0, hspace=0)
plt.savefig(f'C:\\Users\\user\\Desktop\\face_attribute_{test_type}.png')
plt.show()


sel_par_list = ['19071131_S001L1E01C7', '19071131_S001L1E02C7', '19071131_S001L1E03C7', '19071131_S001L1E01C4', '19071131_S001L1E01C10',
                '19071821_S001L1E01C7', '19071821_S001L1E02C7', '19071821_S001L1E03C7', '19071821_S001L1E01C4', '19071821_S001L1E01C10']

temp_df_par = temp_df[temp_df.index.isin(sel_par_list)]
temp_df_par = temp_df_par.reindex(sel_par_list)

color_index_dict = {k:[] for k in range(1, temp_df_par.shape[1]+1)}
for (t, col) in enumerate(temp_df.columns):
    temp_col = temp_df[col]
    # color_index_dict[t+1] = sorted(range(len(temp_col)), key = lambda k:temp_col[k])
    color_index_dict[t+1] = temp_col.values

num_class = temp_df_par.shape[0] # 16*5

# 부분적으로만 그려보자
fig, ax = plt.subplots(num_class, temp_df_par.shape[1]+1, figsize=(2*(temp_df_par.shape[1]+1), 2*num_class)) # multiply 5 b/c there are 5 face-attributes per one face-class

for i, perpar in enumerate(temp_df_par.index):
    person, par = perpar.split('_')
    person=int(person)
    filePath = f'E:\\ANNA_INTERN\\Middle_Resolution_137_unzipped_parcropped_128_{test_type}\\{person}\\32PIX_4GS_{par}.jpg'
    if os.path.isfile(filePath):
        for j in range(temp_df_par.shape[1]+1):
            # ax[i].set_title(merged_df.columns[j+1])
            if j == 0:
                img = cv.imread(filePath, cv.IMREAD_GRAYSCALE)
                #img = cv.resize(img, (16, 16))
                ax[i][0].imshow(np.asarray(img), cmap='gray')
                ax[i][0].set_xticks([])
                ax[i][0].set_yticks([])
                ax[i][0].set_aspect('equal')
            else:
                # flipped_color = int(color_index_dict[j][i]) # j는 model의 개수이므로 dataframe의 column에 해당, i는 face-class의 개수이므로 dataframe의 index에 해당
                # color = num_class-1 - flipped_color
                # color = sns.color_palette('Spectral', num_class)[color]
                
                hit_rate = color_index_dict[j][i]
                num_val = 16
                # conv_hr = (max(temp_df.max()) - 1) + (hit_rate - min(temp_df.min())) / (max(temp_df.max()) - min(temp_df.min()))  # M-1(빨) ~ M(파)
                # conv_hr = min(temp_df.min()) + (hit_rate - min(temp_df.min())) / (max(temp_df.max()) - min(temp_df.min())) # m(빨) ~ m+1(파)
                assert hit_rate >= 0 and hit_rate <= 1
                # conv_hr = (hit_rate - min(temp_df.min())) / (max(temp_df.max()) - min(temp_df.min()))# m(빨) ~ M(파)
                conv_hr = hit_rate
                conv_hr = conv_hr * (num_val - 1) # num_val 등분
                conv_hr = round(conv_hr)

                # if conv_hr < 0:
                #     new_conv_hr = 0
                # elif conv_hr >= num_val:
                #     new_conv_hr = num_val-1
                # else:
                new_conv_hr = conv_hr
                color = sns.color_palette('Spectral', num_val)[new_conv_hr] 
                
                ax[i][j].set_facecolor(color)
                ax[i][j].set_xticks([])
                ax[i][j].set_yticks([])
                ax[i][j].set_aspect('equal')
    else:
        print('The file does not exist.')
fig.subplots_adjust(wspace=0, hspace=0)
plt.savefig(f'C:\\Users\\user\\Desktop\\face_partial_attribute_{test_type}.png')
plt.show()

#%%
# Fig. 4b

# 현재 몇 개 째 돌리고 있는지

test_type_list = ['elec'] #['opt', 'elec']
model_type1_list = [''] #['PCA', 'PCA', '', '']
model_type2_list = ['CNN_AlexNet2_SVC'] #['SVC', 'LR', 'CNN_LR', 'CNN_SVC']
seed_list = [22, 77, 2, 100, 81, 42, 7, 1, 55, 50] # different 1,000 training images (total of 10 permutations)
pix_order_list = ['16PIX', '24PIX', '32PIX', '64PIX', '128PIX']
gs_order_list = ['2GS', '4GS', '6GS', '8GS', '16GS']
class_list = [4] #[2, 4, 16]

for test_type in test_type_list:
    for (model_type1, model_type2) in zip(model_type1_list, model_type2_list): # 제일 첫 번째 model에 대해서만
        model_type = model_type1 + '_' + model_type2
        for c in class_list:
            for m in range(1): # 제일 첫 번째 set에 대해서만
                high_df = pd.DataFrame()
                for seed in seed_list:
                
                    agg_list = []
                    # seed = random.choice(seed_list) # For each set, we have random seed.
                    data_path = f'E:\\ANNA_INTERN\\Question Banks AI Hub_final\\{c}classes\\set{m}\\seed{seed}'
                    try:
                        for n in range(len(os.listdir(data_path))):
                            model_file = os.path.join(data_path, f'comb{n}\\High_Analysis_{test_type}\\High_Level_Data_Analysis_{model_type}.csv')
                            if os.path.isfile(model_file):
                                agg_list.append(1)
                        print(seed, len(agg_list))
                    except:
                        print(seed, 0)

                        seed_list = [22, 77, 2, 100, 81, 42, 7, 1, 55, 50] # different 1,000 training images (total of 10 permutations)
pix_order_list = ['16PIX', '24PIX', '32PIX', '64PIX', '128PIX']
gs_order_list = ['2GS', '4GS', '6GS', '8GS', '16GS']
class_list = [4, 16] # [2, 4, 16]

for test_type in test_type_list:
    for (model_type1, model_type2) in zip(model_type1_list, model_type2_list): # 제일 첫 번째 model에 대해서만
        model_type = model_type1 + model_type2

        for c in class_list:
            high_df = pd.DataFrame()
            for m in range(1): # 제일 첫 번째 set에 대해서만
                for seed in seed_list:
                    if c == 16 or (c == 4 and seed == 22): 
                        data_path = f'E:\\ANNA_INTERN\\Question Banks AI Hub_final\\{c}classes\\set{m}\\seed{seed}'
                        try:
                            for n in range(len(os.listdir(data_path))):
                                preprocessed_data_path =  os.path.join(data_path, f'comb{n}') # 16 classes 는 1 comb 밖에 없음.

                                high_analysis_path = os.path.join(preprocessed_data_path, f'High_Analysis_{test_type}')
                                
                                add_high_df = pd.read_csv(os.path.join(high_analysis_path, f'High_Level_Data_Analysis_{model_type1}_{model_type2}.csv'))
                                add_high_df['hit_rate'] = add_high_df['correctness'].replace(['correct', 'wrong'], [1, 0]) 

                                full_hyper_list = []
                                for i, file_name in enumerate(add_high_df['file_name']):
                                    file_name_list = file_name.split('_')
                                    pix, gs, _ = file_name_list[0], file_name_list[1], file_name_list[-1]
                                    full_hyper = pix + '_' + gs
                                    full_hyper_list.append(full_hyper)
                                    
                                add_high_df['full_hyperpar'] = full_hyper_list

                                add_high_df = add_high_df[['actual_person', 'full_hyperpar', 'hit_rate']]

                                high_df = pd.concat([high_df, add_high_df], axis=0)
                                print(c, seed, n)
                        except:
                            print(c, seed, n, 'error')
                            break
                    else:
                        pass

            old_high_df_cols = high_df.columns
            new_high_df_cols = [f'{c}_' + col_name for (i, col_name) in enumerate(old_high_df_cols) if i == 2] # hit_rate에만 c 달아주기
            new_high_df_cols = [old_high_df_cols.tolist()[0], old_high_df_cols.tolist()[1], *new_high_df_cols]
            high_df.columns = new_high_df_cols

            high_df = pd.DataFrame(high_df.groupby(['actual_person', 'full_hyperpar']).mean().reset_index()) # reset_index to bring 'file_name' to one of the columns.
            high_df = high_df[['actual_person', 'full_hyperpar', f'{c}_hit_rate']]

            if c == 2:
                high_df_2 = high_df
            elif c == 4:
                high_df_4 = high_df
            elif c == 16:
                high_df_16 = high_df


high_df = high_df_4.merge(high_df_16, on=['actual_person', 'full_hyperpar'])
high_df = high_df.astype({'actual_person':int})

# --------- low-level (class별) ---------

df = high_df
df = pd.DataFrame(df.groupby('actual_person').agg(np.mean)).reset_index()

# --------------- 사람별 확인용 ---------------
temp_df = df.set_index('16_hit_rate')
temp_df = temp_df.sort_index(axis=0, ascending=True) # 16_hit_rate를 기준으로 difficult, easy face ID 찾기

num_class = len(list(map(int, list(df[f'actual_person'].unique())))) # 16

fig, ax = plt.subplots(1, num_class, figsize=(5*num_class, 5))
for i, person in enumerate(temp_df['actual_person']):
    filePath = f'E:\\ANNA_INTERN\\Middle_Resolution_137_unzipped_parcropped_128_{test_type}\\{person}\\128PIX_2GS_S001L1E01C7.jpg'
    img = cv.imread(filePath, cv.IMREAD_GRAYSCALE)
    ax[i].imshow(np.asarray(img), cmap='gray')
    ax[i].set_title(f'{person}')
plt.show()
# --------------------------------------------

com_class_list = list(itertools.combinations(class_list, 2))

for i in com_class_list:
    if i[0] >= i[1]:
        M, m = i[0], i[1]
    else:
        M, m = i[1], i[0]

    # --------------- 사람별 확인용 ---------------
    # def label_point(df, ax):
    #     for i, point in df.iterrows():
    #         ax.annotate(point['actual_person'], xy = (point[f'{M}_hit_rate'], point[f'{m}_hit_rate']), xytext=(2,-2), textcoords="offset points")

    # fig, ax = plt.subplots(1, 1)
    #g = sns.lmplot(data=df, x=f'{M}_hit_rate', y=f'{m}_hit_rate', palette='Spectral', hue='actual_person', ci=95)
    # label_point(df, ax)
    # plt.title(f'{model_type}')
    # plt.xlabel(f'{M} classes')
    # plt.ylabel(f'{m} classes')
    # plt.title(f'{model_type}')
    # plt.xlim([0, 1.1])
    # plt.ylim([0, 1.1])
    # plt.show()
    # --------------------------------------------

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(5,5))
    
    # g_extra =  sns.lmplot(data=df, x=f'{M}_hit_rate', y=f'{m}_hit_rate', palette='Spectral', ci=95,
    # scatter_kws={'color':sns.color_palette('Spectral')[5]}, line_kws={'color':sns.color_palette('Spectral')[5]}) # 8 for CNN_SVC

    slope, intercept, r_value, p_value, std_error = linregress(df[f'{M}_hit_rate'], df[f'{m}_hit_rate'])
    plot = sns.regplot(x=f'{M}_hit_rate', y=f'{m}_hit_rate', data=df, color=sns.color_palette('Spectral')[5], line_kws={'label':"y={0:.1f}x+{1:.1f}".format(slope, intercept)})
    plot.legend(loc='lower right')
    sns.despine() # removes the top and right spines from plots

    r, p = sp.stats.pearsonr(df[f'{M}_hit_rate'], df[f'{m}_hit_rate'])
    #ax = plt.gca()
    #plt.text(x=1.05, y=1.05, s=(f'Pearson r={r: .2f}, p={p: .2e}'), transform=ax.transAxes) # (0, 0): lower-left, (1, 1): upper-right
    print(r, p)
    # r, p = sp.stats.spearmanr(df[f'{M}_hit_rate'], df[f'{m}_hit_rate'])
    # plt.text(x=0.05, y=0.1, s=(f'Spearman r={r: .2f}, p={p: .2e}'), transform=ax.transAxes)
    # r, p = sp.stats.kendalltau(df[f'{M}_hit_rate'], df[f'{m}_hit_rate'])
    # plt.text(x=0.05, y=0.05, s=(f'Kendall tau={r: .2f}, p={p: .2e}'), transform=ax.transAxes)

    #plt.title(f'{model_type}')
    plt.xlabel(f'Hit Rate of {model_type} with Class Size of {M}')
    plt.ylabel(f'Hit Rate of {model_type} with Class Size of {m}')
    plt.xlim([0, 1.1])
    plt.ylim([0, 1.1])
    plt.show()