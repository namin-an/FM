"""
Running this file trains ANNs described in the paper.

*Preview of this code:
- Packages

- Loading human and ANN data
- Fig. 3a (Fig. S7a)
- Fig. 3b 
- Fig. 3c (Fig. S7b)
- Fig. 3d
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

