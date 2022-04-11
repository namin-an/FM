"""
Running this file makes visualizions of results in the paper.

- Loading human and ANN data
- Fig. 3a (Fig. S7a)
- Fig. 3b 
- Fig. 3c (Fig. S7b)
- Fig. 3d
"""



#%%
import datetime
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
from scipy.stats import norm, wilcoxon, linregress
import scipy.stats as sp
from statannot import add_stat_annotation
from PIL import Image
import cv2 as cv
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score, classification_report
from sklearn.utils import shuffle
from tensorflow.keras import models, layers
        from tensorflow import keras



#%%
# Loading human and ANN data

# Button
type = 'opt' # 'opt' or 'elec'
test_type = type
human_data = 'E:\\ANNA_INTERN\\Human_Exp\\211202'

if type == 'opt':
  sel_ppl = list(range(300, 309)) + list(range(400, 408)) + [611] 
elif type == 'elec': 
  # sel_ppl = [499, 500, 502] + list(range(504, 509)) + list(range(602, 606)) + list(range(608, 612)) 
  sel_ppl = [499, 500, 502] + list(range(503, 509)) + list(range(602, 607)) + list(range(608, 612)) 

human_df = pd.DataFrame()
n = 9
for i in range(1, 80*n+1, 80):
    try:
      j = i+79
      temp_df = pd.read_csv(os.path.join(human_data, f'main_test({i}_{j}).xls.csv'))
      if i == 1:
        pass
      else:
        temp_df = temp_df.rename(columns = {'유저식별아이디':'useless', 'MC구분':'useless', '성별':'useless', '나이':'useless', '학력':'useless'})
      human_df = pd.concat([human_df, temp_df], axis=1)
    except:
      print(i)

human_df = human_df[human_df['유저식별아이디'].isin(sel_ppl)]
orig_human_df = human_df
human_df = human_df.fillna(0)

sel_col = []
for j in range(1, 80*n+1):
    temp_str = f'선택_A_{j}'
    sel_col.append(temp_str)

human_df = human_df[sel_col]
human_df.index = sel_ppl 
human_df.columns = list(range(80*n))

# check outliers (zeros)
plt.hist(human_df.values, density=True)
plt.show()

# machine data
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


# human and machine data
mer_df = pd.concat([human_df, answer_df], axis=0)
mer_df = mer_df.T
mer_df = mer_df.fillna(0)
orig_mer_df = mer_df

#%%
# Getting ready...

human_df = orig_human_df.copy()
human_df = human_df.fillna(0)
human_df['유저식별아이디'] = human_df['유저식별아이디'].astype(int)
human_df.index = human_df['유저식별아이디']

sel_col = []
for (i, col) in enumerate(human_df.columns):
    split = col.split('_')
    if split[0] == '선택' and split[1] == 'A':
        sel_col.append(col)
    
acc_df = human_df[sel_col]
acc_df.columns = [n for n in range(acc_df.shape[1])]
acc_df = acc_df.astype(int)

for q in range(acc_df.shape[1]):
    act_per = answer_df.loc['act_per'][q] # 실제 사람 데이터
    for s in range(acc_df.shape[0]):
        pred_per = acc_df.iloc[s, q]
        try:
            if str(int(pred_per)) == str(int(act_per)):
                acc_df.iloc[s, q] = 1 # modify the value of the cell (s, t)
            else:
                acc_df.iloc[s, q] = 0
        except:
            acc_df.iloc[s, q] = 0

acc_df.columns = answer_df.T['Answer']
acc_df

test_type_list = [type] #['opt', 'elec']
model_type1_list = [''] #['', ''] #['PCA', 'PCA', '', '']
model_type2_list = ['CNN_SVC'] #['CNN_SVC', 'CNN_SVCft'] #['SVC', 'LR', 'CNN_LR', 'CNN_SVC']
seed_list = [22, 77, 2, 100, 81, 42, 7, 1, 55, 50] # [22, 77, 2, 100, 81, 42, 7, 1, 55, 50] # different 1,000 training images (total of 10 permutations)
pix_order_list = ['16PIX', '24PIX', '32PIX', '64PIX', '128PIX']
gs_order_list = ['2GS', '4GS', '6GS', '8GS', '16GS']
class_list = [16] # [2, 4, 16]

df = pd.read_csv('E:\\ANNA_INTERN\\210827_ANNA_Removing_uncontaminated_data.csv')
l = list(range(df.shape[0]))
n = 16
random.seed(22)
set_1 = random.sample(l, n)
sets = [set_1]
r = class_list[0]

for test_type in test_type_list:
    mac_df = pd.DataFrame()
    for (model_type1, model_type2) in zip(model_type1_list, model_type2_list): 
        model_type = model_type1 + model_type2
        for c in class_list:
            for m in range(1):
                input_folder = [df.iloc[i, 0] for i in sets[m]] 
                assert len(input_folder) == 16
                com_obj = itertools.combinations(input_folder, r)
                com_list = list(com_obj)

               # mac_df = pd.DataFrame()
                for seed in seed_list:
                    data_path = f'E:\\ANNA_INTERN\\Question Banks AI Hub_final\\{c}classes\\set{m}\\seed{seed}'

                    for n in range(len(os.listdir(data_path))): # len(com_list)
                        print(seed, n)

                        preprocessed_data_path =  os.path.join(data_path, f'comb{n}') 
                        high_analysis_path = os.path.join(preprocessed_data_path, f'High_Analysis_{test_type}')
                        
                        add_high_df = pd.read_csv(os.path.join(high_analysis_path, f'High_Level_Data_Analysis_{model_type1}_{model_type2}.csv'))
                        add_high_df['Hit Rate'] = add_high_df['correctness'].replace(['correct', 'wrong'], [1, 0]) 
                        add_high_df = add_high_df[['file_name', 'actual_person', 'Hit Rate']]
                        add_high_df['actual_person'] = add_high_df['actual_person'].astype(int)
                        
                        img_list, hyperpar_list, par_list = [], [], []
                        for i in range(add_high_df.shape[0]):
                            file_name = add_high_df['file_name'][i].split('.')[0]
                            pix, gs, par = file_name.split('_')
                            per = add_high_df['actual_person'][i]
                            img = str(per) + '_' + str(pix) + '_' + str(gs) + '_' + str(par) + '.jpg'
                            img_list.append(img)
                            hyperpar_list.append(str(pix) + '_' + str(gs))
                            par_list.append(str(par))
                        add_high_df['img'] = img_list
                        add_high_df['hyperpar'] = hyperpar_list
                        add_high_df['par'] = par_list
                        
                        add_high_df = add_high_df[['img', 'hyperpar', 'par', 'Hit Rate']]
                    
                    add_high_df['Seed'] = [seed] * add_high_df.shape[0] #[f'{model_type}_{seed}'] * add_high_df.shape[0]
                    
                    mac_df = pd.concat([mac_df, add_high_df], axis=0)

new_hyperpar_name_list = ['16PIX_2GS', '16PIX_4GS', '16PIX_8GS', 
                          '32PIX_2GS', '32PIX_4GS', '32PIX_8GS',  
                          '64PIX_2GS', '64PIX_4GS', '64PIX_8GS']
par_list = ['S001L1E01C4', 'S001L1E01C7', 'S001L1E01C10',
             'S001L1E02C7',
            'S001L1E03C7']
mac_df = mac_df[mac_df['hyperpar'].isin(new_hyperpar_name_list)]
mac_df = mac_df[mac_df['par'].isin(par_list)]
# mac_df = mac_df.groupby('img').mean().reset_index()

mac_df = mac_df.pivot(index='Seed', columns='img', values='Hit Rate')

mac_df

mer_df3 = pd.concat([acc_df, mac_df], join='inner')
mer_df3 = mer_df3[mer_df3.index != 'img']

mer_df3 = mer_df3.T.astype(float)


mer_df3

#%%
# Fig. 3a (Fig. S7a)

temp_acc_df = mer_df3.copy()

temp_index = []
for file_name in answer_df.T['Answer']:
    split = file_name.split('.')
    _, pix, gs, par = split[0].split('_')
    par = pix + '_' + gs + '_' + par
    temp_index.append(par)
    
temp_acc_df['par'] = temp_index

temp_acc_df = temp_acc_df.groupby('par').mean()

corrMatrix = temp_acc_df.corr()

plt.figure(figsize=(25, 20))
sns.heatmap(corrMatrix, vmin=0, vmax=1, annot=True, cmap='vlag')
plt.title(f'Phophene quality condition & face-attribute level')
plt.xlabel('')
plt.ylabel('')
plt.show()



#%%
# Fig. 3b

# lm plot for dr. MI 
pix_list, hyperpar_list = [], []
for hyperpar in temp_acc_df['par']:
    pix, gs, _ = hyperpar.split('_')
    pix_list.append(pix)
    hyperpar_list.append(pix + '_' + gs)
    
temp_acc_df['Pixels'] = pix_list
temp_acc_df['hyperpar'] = hyperpar_list
temp_acc_df['Color'] = ['0'] * temp_acc_df.shape[0]

#temp_acc_df = temp_acc_df.reset_index()

temp_acc_df.columns = temp_acc_df.columns.astype(str)


g = sns.lmplot(data=temp_acc_df, x='402', y='306', ci=None, hue='Color', palette=[sns.color_palette('Paired')[5]]) # hue_order
for ax in g.axes.ravel():
    ax.plot((0, 1), (0, 1), color='gray', lw=1, linestyle='--')
plt.xlim([0, 1.1])
plt.ylim([0, 1.1])
# plt.legend('')
plt.show()

g = sns.lmplot(data=temp_acc_df, x='401', y='100', ci=None, hue='Color', palette=[sns.color_palette('Paired')[9]])
for ax in g.axes.ravel():
    ax.plot((0, 1), (0, 1), color='gray', lw=1, linestyle='--')
plt.xlim([0, 1.1])
plt.ylim([0, 1.1])
# plt.legend('')
sns.despine()
plt.show()

g = sns.lmplot(data=temp_acc_df, x='407', y='2', ci=None, hue='Color', palette=[sns.color_palette('Paired')[9]])
for ax in g.axes.ravel():
    ax.plot((0, 1), (0, 1), color='gray', lw=1, linestyle='--')
plt.xlim([0, 1.1])
plt.ylim([0, 1.1])
# plt.legend('')
sns.despine()
plt.show()

g = sns.lmplot(data=temp_acc_df, x='2', y='7', ci=None, hue='Color', palette=[sns.color_palette('Paired')[1]])
for ax in g.axes.ravel():
    ax.plot((0, 1), (0, 1), color='gray', lw=1, linestyle='--')
plt.xlim([0, 1.1])
plt.ylim([0, 1.1])
# plt.legend('')
sns.despine()
plt.show()



#%%
# Fig. 3d

class LinearReg():

    def __init__(self, new_x):
        self.new_x = new_x
    
    def equation_val(self, new_x):
        x, y = mer_avg['Machines'], mer_avg['Humans']
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        new_y = slope * new_x + intercept
        return new_y
    
    @staticmethod
    def equation():
        x, y = mer_avg['Machines'], mer_avg['Humans']
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        print(f'R: {r_value}, p-value: {p_value}')
        return slope, intercept

new_hyperpar_name_list = ['16PIX_2GS', '16PIX_4GS', '16PIX_8GS',
                          '32PIX_2GS', '32PIX_4GS', '32PIX_8GS',
                          '64PIX_2GS', '64PIX_4GS', '64PIX_8GS']
par_mac_df_T_gp = mac_df_T_gp[np.logical_not(mac_df_T_gp.index.isin(new_hyperpar_name_list))]
df = par_mac_df_T_gp.mean(axis=1).to_frame()
df.columns = ['Hit Rate of Machines']
diction = {key: [] for key in df.index}
for i in range(df.shape[0]):
    linear_eq = LinearReg(df['Hit Rate of Machines'].iloc[i])
    new_y = linear_eq.equation_val(df['Hit Rate of Machines'].iloc[i])
    
    diction[df.index[i]].append(df['Hit Rate of Machines'].iloc[i])
    diction[df.index[i]].append(new_y)

new_df = pd.DataFrame.from_dict(diction).T#.reset_index()
new_df.columns = ['Machines', 'Humans']
new_df

hyperpar_list = []
for hyperpar in mer_avg['hyperpar']:
    pix, gs = hyperpar.split('_')
    hyperpar_list.append(f'{pix}_{gs}')
mer_avg['hyperpar'] = hyperpar_list
mer_avg_gp = mer_avg.groupby('hyperpar').mean().reset_index()

g = sns.lmplot(data=mer_avg_gp, x='Machines', y='Humans', ci=None, palette='rocket', hue='hyperpar') #, palette='rocket', hue='hyperpar') # hue_order
for ax in g.axes.ravel():
    ax.plot((0, 1), (0, 1), color='gray', lw=1, linestyle='--')

x = np.linspace(0, 1, 100)
linear_eq = LinearReg(x)
slope_intercept = linear_eq.equation()
slope, intercept = slope_intercept[0], slope_intercept[-1]
y = slope*x + intercept
print(f'y={slope}x+{intercept}')
plt.plot(x, y, '-r', label=f'y={slope}x+{intercept}', alpha=0.5)

sns.scatterplot(x='Machines', y='Humans', data=new_df, color='r', alpha=0.5, s=150, marker='+') #, legend = False)
    
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.legend([])
plt.show()



#%%
# Fig. 3c (Fig. S7b)

# Humans
df = orig_human_df.copy()

ans_list, pix_list, gs_list, par_list = list(), list(), list(), list()
for (i, col) in enumerate(df.columns):
    split_col = col.split('_')
    if split_col[0] == '선택' and split_col[1] == 'B':
        k = int(split_col[-1]) - 1
        file_name = answer_df[k].loc['Answer'].split('.')[0]
        # ans, pix, gs, par = file_name.split('_')
        ans_list.append(file_name)

    else:
        ans_list.append(col)

df.columns = ans_list

sel_col = []
for col in df.columns:
    split = col.split('_')
    if len(split) == 4: # 시간만
        sel_col.append(col)
df = df[sel_col]

temp_df = df.isna().sum().to_frame()
sum_na = sum(temp_df[0].values)
sum_tot = df.shape[0]*df.shape[1]

a = sum_na / sum_tot * 100
print(a)


# Top-1 Accuracy

temp_mer_df0 = orig_mer_df.copy()

d = dict()
for per in sel_ppl:
    d[per] = []
pix_par_list = ['16PIX', '32PIX', '64PIX']
gs_par_list = ['2GS', '4GS', '8GS']

for pix in pix_par_list:
    temp_mer_df = temp_mer_df0[temp_mer_df0['PIX'] == pix]
    for gs in gs_par_list:
        temp_mer_df2 = temp_mer_df[temp_mer_df['GS'] == gs]
        temp_mer_df2 = temp_mer_df2.reset_index()
        
        for per in sel_ppl:
            new_l = []
            for i in range(temp_mer_df2.shape[0]):
                if str(int(temp_mer_df2[per][i])) == str(int(temp_mer_df2['act_per'][i])):
                    new_l.append(1)
                else:
                    new_l.append(0)
            score = sum(new_l) / len(new_l)
            d[per].append(score)
            # print(pix, '_', gs, '_', per, ': ', round((sum(new_l) / len(new_l))*100, 4))

per_df1 = pd.DataFrame.from_dict(d)
hyperpar_name_list = ['16PIX_2GS', '16PIX_4GS', '16PIX_8GS', 
                      '32PIX_2GS', '32PIX_4GS', '32PIX_8GS',  
                      '64PIX_2GS', '64PIX_4GS', '64PIX_8GS']
per_df1.index = hyperpar_name_list

per_df1 = per_df1.stack().to_frame().reset_index(level=0)
per_df1.columns = ['Resolution', 'Hit Rate']

per_df1


# Top-2 Accuracy

human_df2 = pd.DataFrame()
n = 9
for i in range(1, 80*n+1, 80):
    try:
      j = i+79
      temp_df = pd.read_csv(os.path.join(human_data, f'main_test({i}_{j}).xls.csv'))
      if i == 1:
        pass
      else:
        temp_df = temp_df.rename(columns = {'유저식별아이디':'useless', 'MC구분':'useless', '성별':'useless', '나이':'useless', '학력':'useless'})
      human_df2 = pd.concat([human_df2, temp_df], axis=1)
    except:
      print(i)


human_df2 = human_df2[human_df2['유저식별아이디'].isin(sel_ppl)]
human_df2 = human_df2.fillna(0)

sel_col = []
for j in range(1, 80*n+1):
    temp_str = f'선택_B_{j}'
    sel_col.append(temp_str)

human_df2 = human_df2[sel_col]
human_df2.index = sel_ppl 
human_df2.columns = list(range(80*n))

mer_df2 = pd.concat([human_df2, answer_df], axis=0)
mer_df2 = mer_df2.T
mer_df2 = mer_df2.fillna(0)

temp_mer_df0 = mer_df2.copy()

d = dict()
for per in sel_ppl:
    d[per] = []
pix_par_list = ['16PIX', '32PIX', '64PIX']
gs_par_list = ['2GS', '4GS', '8GS']

for pix in pix_par_list:
    temp_mer_df = temp_mer_df0[temp_mer_df0['PIX'] == pix]
    for gs in gs_par_list:
        temp_mer_df2 = temp_mer_df[temp_mer_df['GS'] == gs]
        temp_mer_df2 = temp_mer_df2.reset_index()
        
        for per in sel_ppl:
            new_l = []
            for i in range(temp_mer_df2.shape[0]):
                if str(int(temp_mer_df2[per][i])) == str(int(temp_mer_df2['act_per'][i])):
                    new_l.append(1)
                else:
                    new_l.append(0)
            score = sum(new_l) / len(new_l)
            d[per].append(score)
            # print(pix, '_', gs, '_', per, ': ', round((sum(new_l) / len(new_l))*100, 4))

per_df2 = pd.DataFrame.from_dict(d)
hyperpar_name_list = ['16PIX_2GS', '16PIX_4GS', '16PIX_8GS', 
                      '32PIX_2GS', '32PIX_4GS', '32PIX_8GS',  
                      '64PIX_2GS', '64PIX_4GS', '64PIX_8GS']
per_df2.index = hyperpar_name_list

per_df2 = per_df2.stack().to_frame().reset_index(level=0)
per_df2.columns = ['Resolution', 'Hit Rate']
per_df2['Hit Rate'] = per_df1['Hit Rate'] + per_df2['Hit Rate'] 

per_df2

per_df2.groupby('Resolution').mean()


# Prediction (TOP-1) - # 1 Linear Regression

## human
human_df = orig_human_df.copy()
human_df = human_df.fillna(0)
human_df['유저식별아이디'] = human_df['유저식별아이디'].astype(int)
human_df.index = human_df['유저식별아이디']

sel_col = []
for (i, col) in enumerate(human_df.columns):
    split = col.split('_')
    if split[0] == '선택' and split[1] == 'A':
        sel_col.append(col)
    
acc_df = human_df[sel_col]
acc_df.columns = [n for n in range(acc_df.shape[1])]
acc_df = acc_df.astype(int)

for q in range(acc_df.shape[1]):
    act_per = answer_df.loc['act_per'][q] 
    for s in range(acc_df.shape[0]):
        pred_per = acc_df.iloc[s, q]
        try:
            if str(int(pred_per)) == str(int(act_per)):
                acc_df.iloc[s, q] = 1 # modify the value of the cell (s, t)
            else:
                acc_df.iloc[s, q] = 0
        except:
            acc_df.iloc[s, q] = 0

acc_df.columns = answer_df.T['Answer']

## machine
test_type_list = [type] #['opt', 'elec']
model_type1_list = [''] #['PCA', 'PCA', '', '']
model_type2_list = ['CNN_SVC'] #['SVC', 'LR', 'CNN_LR', 'CNN_SVC']
seed_list = [22, 77, 2, 100, 81, 42, 7, 55, 50] # [22, 77, 2, 100, 81, 42, 7, 1, 55, 50] # different 1,000 training images (total of 10 permutations)
pix_order_list = ['16PIX', '24PIX', '32PIX', '64PIX', '128PIX']
gs_order_list = ['2GS', '4GS', '6GS', '8GS', '16GS']
class_list = [16] # [2, 4, 16]

df = pd.read_csv('E:\\ANNA_INTERN\\210827_ANNA_Removing_uncontaminated_data.csv')
l = list(range(df.shape[0]))
n = 16
random.seed(22)
set_1 = random.sample(l, n)
sets = [set_1]
r = class_list[0]

new_seed_list = []
for test_type in test_type_list:
    mac_df = pd.DataFrame()
    for (model_type1, model_type2) in zip(model_type1_list, model_type2_list):
        model_type = model_type1 + model_type2
        for c in class_list:
            for m in range(1): 
                input_folder = [df.iloc[i, 0] for i in sets[m]] 
                assert len(input_folder) == 16
                com_obj = itertools.combinations(input_folder, r)
                com_list = list(com_obj)

                # mac_df = pd.DataFrame()
                for seed in seed_list:
                    new_seed_list.append(f'{model_type}_{seed}')
                    data_path = f'E:\\ANNA_INTERN\\Question Banks AI Hub_final\\{c}classes\\set{m}\\seed{seed}'

                    for n in range(len(os.listdir(data_path))): # len(com_list)
                        print(seed, n)

                        preprocessed_data_path =  os.path.join(data_path, f'comb{n}') 
                        high_analysis_path = os.path.join(preprocessed_data_path, f'High_Analysis_{test_type}')
                        
                        add_high_df = pd.read_csv(os.path.join(high_analysis_path, f'High_Level_Data_Analysis_{model_type1}_{model_type2}.csv'))
                        add_high_df['Hit Rate'] = add_high_df['correctness'].replace(['correct', 'wrong'], [1, 0]) 
                        add_high_df = add_high_df[['file_name', 'actual_person', 'Hit Rate']]
                        add_high_df['actual_person'] = add_high_df['actual_person'].astype(int)
                        
                        img_list, hyperpar_list, par_list = [], [], []
                        for i in range(add_high_df.shape[0]):
                            file_name = add_high_df['file_name'][i].split('.')[0]
                            pix, gs, par = file_name.split('_')
                            per = add_high_df['actual_person'][i]
                            img = str(per) + '_' + str(pix) + '_' + str(gs) + '_' + str(par) + '.jpg'
                            img_list.append(img)
                            hyperpar_list.append(str(pix) + '_' + str(gs))
                            par_list.append(str(par))
                        add_high_df['img'] = img_list
                        add_high_df['hyperpar'] = hyperpar_list
                        add_high_df['par'] = par_list
                        
                        add_high_df = add_high_df[['img', 'hyperpar', 'par', 'Hit Rate']]
                    
                    add_high_df['Seed'] = [f'{model_type}_{seed}'] * add_high_df.shape[0]
                    
                    mac_df = pd.concat([mac_df, add_high_df], axis=0)


par_list = ['S001L1E01C4', 'S001L1E01C7', 'S001L1E01C10',
             'S001L1E02C7',
            'S001L1E03C7']
mac_df = mac_df[mac_df['par'].isin(par_list)]

mac_df = mac_df.pivot(index='Seed', columns='img', values='Hit Rate')

mac_df_copy = mac_df.copy()

mac_df_T = mac_df_copy.T

hyper_par_list = []
for img in mac_df_T.index:
    _, pix, gs, _ = img.split('_')
    hyper_par_list.append(f'{pix}_{gs}')
mac_df_T['hyperpar'] = hyper_par_list
#mac_df_T_par = mac_df_T.iloc[:, 2]
mac_df_T_gp = mac_df_T.groupby('hyperpar').mean()



# both
# scatter plot (feedback) # 6 -> Feedback -> 6A
mer_df3 = pd.concat([acc_df, mac_df], join='inner')
mer_df3 = mer_df3[mer_df3.index != 'img']

mer_df3 = mer_df3.T.astype(float)

temp_acc_df = mer_df3.copy()

pix_list, hyperpar_list = [], []
for hyperpar in temp_acc_df.index:
    _, pix, gs, _ = hyperpar.split('_')
    pix_list.append(pix)
    hyperpar_list.append(hyperpar)
    
temp_acc_df['Pixels'] = pix_list
temp_acc_df['hyperpar'] = hyperpar_list

temp_acc_df.columns = temp_acc_df.columns.astype(str)

# People
sel_ppl_str = list(str(per) for per in sel_ppl)
ppl_df = temp_acc_df[sel_ppl_str]
ppl_df_avg = ppl_df.mean(axis=1)

# Machines
sel_mac_str = list(str(per) for per in new_seed_list)
mac_df = temp_acc_df[sel_mac_str]
mac_df_avg = mac_df.mean(axis=1)

mer_avg = pd.concat([ppl_df_avg, mac_df_avg], axis=1)
mer_avg.columns = ['Humans', 'Machines']

hyper_list = []
i = 0
for img in mer_avg.index:
    _, pix, gs, _ = img.split('_')
    hyper_list.append(f'{pix}_{gs}')
    i+=1
mer_avg['hyperpar'] = hyper_list

mer_avg = mer_avg.groupby('hyperpar').mean()
mer_avg = mer_avg.reset_index()


class LinearReg():

    def __init__(self, new_x):
        self.new_x = new_x
    
    def equation_val(self, new_x):
        x, y = mer_avg['Machines'], mer_avg['Humans']
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        new_y = slope * new_x + intercept
        return new_y
    
    @staticmethod
    def equation():
        x, y = mer_avg['Machines'], mer_avg['Humans']
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        print(f'R: {r_value}, p-value: {p_value}')
        return slope, intercept

new_hyperpar_name_list = ['16PIX_2GS', '16PIX_4GS', '16PIX_8GS',
                          '32PIX_2GS', '32PIX_4GS', '32PIX_8GS',
                          '64PIX_2GS', '64PIX_4GS', '64PIX_8GS']
par_mac_df_T_gp = mac_df_T_gp[np.logical_not(mac_df_T_gp.index.isin(new_hyperpar_name_list))]
df = par_mac_df_T_gp.mean(axis=1).to_frame()
df.columns = ['Hit Rate of Machines']
diction = {key: [] for key in df.index}
for i in range(df.shape[0]):
    linear_eq = LinearReg(df['Hit Rate of Machines'].iloc[i])
    new_y = linear_eq.equation_val(df['Hit Rate of Machines'].iloc[i])
    
    diction[df.index[i]].append(df['Hit Rate of Machines'].iloc[i])
    diction[df.index[i]].append(new_y)

new_df = pd.DataFrame.from_dict(diction).T#.reset_index()
new_df.columns = ['Machines', 'Humans']
new_df1 = new_df.reset_index()
new_df1.columns = ['Resolution', 'Machines', 'Humans']

new_df1


# Prediction (TOP-1) - # 2 ANN

# human
human_df = orig_human_df.copy()
human_df = human_df.fillna(0)
human_df['유저식별아이디'] = human_df['유저식별아이디'].astype(int)
human_df.index = human_df['유저식별아이디']

sel_col = []
for (i, col) in enumerate(human_df.columns):
    split = col.split('_')
    if split[0] == '선택' and split[1] == 'A':
        sel_col.append(col)
    
acc_df = human_df[sel_col]
acc_df.columns = [n for n in range(acc_df.shape[1])]
acc_df = acc_df.astype(int)

for q in range(acc_df.shape[1]):
    act_per = answer_df.loc['act_per'][q] # 실제 사람 데이터
    for s in range(acc_df.shape[0]):
        pred_per = acc_df.iloc[s, q]
        try:
            if str(int(pred_per)) == str(int(act_per)):
                acc_df.iloc[s, q] = 1 # modify the value of the cell (s, t)
            else:
                acc_df.iloc[s, q] = 0
        except:
            acc_df.iloc[s, q] = 0

acc_df.columns = answer_df.T['Answer']

# machine
test_type_list = [type] #['opt', 'elec']
model_type1_list = [''] #['PCA', 'PCA', '', '']
model_type2_list = ['CNN_SVC'] #['SVC', 'LR', 'CNN_LR', 'CNN_SVC']
seed_list = [22, 77, 2, 100, 81, 42, 7, 55, 50] # [22, 77, 2, 100, 81, 42, 7, 1, 55, 50] # different 1,000 training images (total of 10 permutations)
pix_order_list = ['16PIX', '24PIX', '32PIX', '64PIX', '128PIX']
gs_order_list = ['2GS', '4GS', '6GS', '8GS', '16GS']
class_list = [16] # [2, 4, 16]

df = pd.read_csv('E:\\ANNA_INTERN\\210827_ANNA_Removing_uncontaminated_data.csv')
l = list(range(df.shape[0]))
n = 16
random.seed(22)
set_1 = random.sample(l, n)
sets = [set_1]
r = class_list[0]

new_seed_list = []
for test_type in test_type_list:
    mac_df = pd.DataFrame()
    for (model_type1, model_type2) in zip(model_type1_list, model_type2_list):
        model_type = model_type1 + model_type2
        for c in class_list:
            for m in range(1): 
                input_folder = [df.iloc[i, 0] for i in sets[m]] 
                assert len(input_folder) == 16
                com_obj = itertools.combinations(input_folder, r)
                com_list = list(com_obj)

                # mac_df = pd.DataFrame()
                for seed in seed_list:
                    new_seed_list.append(f'{model_type}_{seed}')
                    data_path = f'E:\\ANNA_INTERN\\Question Banks AI Hub_final\\{c}classes\\set{m}\\seed{seed}'

                    for n in range(len(os.listdir(data_path))): # len(com_list)
                        print(seed, n)

                        preprocessed_data_path =  os.path.join(data_path, f'comb{n}') 
                        high_analysis_path = os.path.join(preprocessed_data_path, f'High_Analysis_{test_type}')
                        
                        add_high_df = pd.read_csv(os.path.join(high_analysis_path, f'High_Level_Data_Analysis_{model_type1}_{model_type2}.csv'))
                        add_high_df['Hit Rate'] = add_high_df['correctness'].replace(['correct', 'wrong'], [1, 0]) 
                        add_high_df = add_high_df[['file_name', 'actual_person', 'Hit Rate']]
                        add_high_df['actual_person'] = add_high_df['actual_person'].astype(int)
                        
                        img_list, hyperpar_list, par_list = [], [], []
                        for i in range(add_high_df.shape[0]):
                            file_name = add_high_df['file_name'][i].split('.')[0]
                            pix, gs, par = file_name.split('_')
                            per = add_high_df['actual_person'][i]
                            img = str(per) + '_' + str(pix) + '_' + str(gs) + '_' + str(par) + '.jpg'
                            img_list.append(img)
                            hyperpar_list.append(str(pix) + '_' + str(gs))
                            par_list.append(str(par))
                        add_high_df['img'] = img_list
                        add_high_df['hyperpar'] = hyperpar_list
                        add_high_df['par'] = par_list
                        
                        add_high_df = add_high_df[['img', 'hyperpar', 'par', 'Hit Rate']]
                    
                    add_high_df['Seed'] = [f'{model_type}_{seed}'] * add_high_df.shape[0]
                    
                    mac_df = pd.concat([mac_df, add_high_df], axis=0)

new_hyperpar_name_list = ['16PIX_2GS', '16PIX_4GS', '16PIX_8GS', 
                          '32PIX_2GS', '32PIX_4GS', '32PIX_8GS',  
                          '64PIX_2GS', '64PIX_4GS', '64PIX_8GS']
par_list = ['S001L1E01C4', 'S001L1E01C7', 'S001L1E01C10',
            'S001L1E02C7',
            'S001L1E03C7']

######### Ext. Fig. 2-1
mac_df_ext = mac_df[mac_df['hyperpar'].isin(new_hyperpar_name_list)]
mac_df_ext = mac_df_ext[mac_df_ext['par'].isin(par_list)]
mac_df_ext = mac_df_ext.pivot(index='Seed', columns='img', values='Hit Rate')
#########
mac_df = mac_df[np.logical_not(mac_df['hyperpar'].isin(new_hyperpar_name_list))] # exclude human parameters
mac_df = mac_df[mac_df['par'].isin(par_list)]
# mac_df = mac_df.groupby('img').mean().reset_index()

mac_df = mac_df.pivot(index='Seed', columns='img', values='Hit Rate')

mac_df_copy = mac_df.copy()
mac_df_T = mac_df_copy.T

con_list = []
for img in mac_df_T.index:
    _, pix, gs, par = img.split('_')
    con_list.append(f'{pix}_{gs}_{par}')
mac_df_T['con'] = con_list
mac_df_T_gp = mac_df_T.groupby('con').mean()


hyper_par_list, par_list = [], []
for img in mac_df_T_gp.index:
    pix, gs, par = img.split('_')
    hyper_par_list.append(f'{pix}_{gs}')
    par_list.append(par.split('.')[0])
mac_df_T_gp['hyperpar'] = hyper_par_list
mac_df_T_gp['par'] = par_list

mac_df_T_gp['Avg of ANNs'] = mac_df_T_gp[new_seed_list].mean(axis=1)
mac_df_T_gp = mac_df_T_gp[['Avg of ANNs', 'hyperpar', 'par']]


class ANN():
    def __init__(self, new_x, checkpoint_file):
        self.new_x = new_x
        self.checkpoint_file = checkpoint_file
    
    @staticmethod
    def create_model():
        
        model = models.Sequential()
        model.add(keras.Input(shape=(6, )))
        model.add(layers.Dense(3, activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(5, activation='relu'))
        print(model.summary())
        
        return model
    
    def load_model(self, model):
        
        model = keras.models.load_model(self.checkpoint_file)
        
        return model
    
    def eval_mod(self, w_model):
        new_ys = w_model(self.new_x)
        return new_ys

df = pd.DataFrame()
new_h_list = list(set(mac_df_T_gp['hyperpar'].values))
for l, hyperpar in enumerate(new_h_list): # for all resolutions
    par_old_df = mac_df_T_gp[mac_df_T_gp['hyperpar'] == hyperpar]
    temp_df = par_old_df['Avg of ANNs'].to_frame().T
    
    par_list = list(range(1, 6))
    temp_df.columns = [f'ANN_{j}' for j in par_list]
    
    df = pd.concat([df, temp_df], axis=0)
df['Cor'] = [0.7598]*df.shape[0]
df.index = new_h_list

ann_inst = ANN(df.values, 'Chcekpoints_FB/Checkpoint_3.h5') # Checkpoint_3, Checkpoint_4
mod = ann_inst.create_model()
w_mod = ann_inst.load_model(mod)
new_ys = ann_inst.eval_mod(w_mod)

diction = {x: [] for x in df.index}
for i in range(df.shape[0]):
    diction[df.index[i]].append(np.mean(df.iloc[i, :].values))
    diction[df.index[i]].append(np.mean(new_ys[i, :]))

new_df = pd.DataFrame.from_dict(diction).T#.reset_index()
new_df.columns = ['Machines', 'Humans']
new_df1_ann = new_df.reset_index()
new_df1_ann.columns = ['Resolution', 'Machines', 'Humans']

new_df1_ann


# Ext Fig. 2-1

mer_df3 = pd.concat([acc_df, mac_df_ext], join='inner')
mer_df3 = mer_df3[mer_df3.index != 'img']

mer_df3 = mer_df3.T.astype(float)

#####################################
temp_acc_df = mer_df3.copy()

temp_index = []
for file_name in answer_df.T['Answer']:
    split = file_name.split('.')
    _, pix, gs, par = split[0].split('_')
    par = pix + '_' + gs + '_' + par
    temp_index.append(par)
    
temp_acc_df['par'] = temp_index

temp_acc_df = temp_acc_df.groupby('par').mean()

corrMatrix = temp_acc_df.corr()

plt.figure(figsize=(25, 20))
sns.heatmap(corrMatrix, vmin=0, vmax=1, annot=True, cmap='vlag')
plt.title(f'Phophene quality condition & face-attribute level')
plt.xlabel('')
plt.ylabel('')
plt.show()


# Prediction (TOP-2) - # 1 Linear Regression

human_df = orig_human_df.copy()
human_df = human_df.fillna(0)
human_df['유저식별아이디'] = human_df['유저식별아이디'].astype(int)
human_df.index = human_df['유저식별아이디']

sel_col1, sel_col2 = [], []
for (i, col) in enumerate(human_df.columns):
    split = col.split('_')
    if split[0] == '선택' and split[1] == 'A':
        sel_col1.append(col)
    if split[0] == '선택' and split[1] == 'B':
        sel_col2.append(col)
        
acc_df1 = human_df[sel_col1]
acc_df1.columns = [n for n in range(acc_df1.shape[1])]
acc_df1 = acc_df1.astype(int)
acc_df2 = human_df[sel_col2]
acc_df2.columns = [n for n in range(acc_df2.shape[1])]
acc_df2 = acc_df2.astype(int)

for q in range(acc_df1.shape[1]):
    act_per = answer_df.loc['act_per'][q]
    for s in range(acc_df1.shape[0]):
        pred_per1 = acc_df1.iloc[s, q]
        pred_per2 = acc_df2.iloc[s, q]
        try:
            if str(int(pred_per1)) == str(int(act_per)) or str(int(pred_per2)) == str(int(act_per)):
                acc_df1.iloc[s, q] = 1 # modify the value of the cell (s, t)
            else:
                acc_df1.iloc[s, q] = 0
        except:
            acc_df1.iloc[s, q] = 0

acc_df1.columns = answer_df.T['Answer']
acc_df = acc_df1.T# .reset_index(drop=True)
acc_df = acc_df.mean(axis=1).reset_index()
acc_df.columns = ['file_name', 'Top-2 Hit Rate of Humans']
h_list = []
for file in acc_df['file_name']:
    _, pix, gs, _ = file.split('_')
    h_list.append(f'{pix}_{gs}')
acc_df['Resolution'] = h_list
new_acc_df2 = acc_df.groupby('Resolution').mean().reset_index()
new_acc_df2.columns = ['Resolution', 'Humans']


## machine
test_type_list = [type] #['opt', 'elec']
model_type1_list = [''] #['PCA', 'PCA', '', '']
model_type2_list = ['CNN_SVC'] #['SVC', 'LR', 'CNN_LR', 'CNN_SVC']
seed_list = [22, 77, 2, 100, 81, 42, 7, 55, 50] # [22, 77, 2, 100, 81, 42, 7, 1, 55, 50] # different 1,000 training images (total of 10 permutations)
pix_order_list = ['16PIX', '24PIX', '32PIX', '64PIX', '128PIX']
gs_order_list = ['2GS', '4GS', '6GS', '8GS', '16GS']
class_list = [16] # [2, 4, 16]

df = pd.read_csv('E:\\ANNA_INTERN\\210827_ANNA_Removing_uncontaminated_data.csv')
l = list(range(df.shape[0]))
n = 16
random.seed(22)
set_1 = random.sample(l, n)
sets = [set_1]
r = class_list[0]
face_labels = [19081632,
               19080133,
               19092711,
               19070311,
               19090631,
               19092521,
               19071821,
               19081421,
               19090222,
               19082032,
               19082131,
               19070231, 
               19070522,
               19071131,
               19072221,
               19072922]

new_seed_list = []
for test_type in test_type_list:
    for (model_type1, model_type2) in zip(model_type1_list, model_type2_list): 
        model_type = model_type1 + model_type2
        for c in class_list:
            for m in range(1): 
                input_folder = [df.iloc[i, 0] for i in sets[m]] 
                assert len(input_folder) == 16
                com_obj = itertools.combinations(input_folder, r)
                com_list = list(com_obj)

                mac_df2 = pd.DataFrame()
                for seed in seed_list:
                    new_seed_list.append(f'{model_type}_{seed}')
                    data_path = f'E:\\ANNA_INTERN\\Question Banks AI Hub_final\\{c}classes\\set{m}\\seed{seed}'
                    
                    res_d = {'16PIX_2GS': [], '16PIX_4GS': [], '16PIX_6GS': [], '16PIX_8GS': [], '16PIX_16GS': [],
                             '24PIX_2GS': [], '24PIX_4GS': [], '24PIX_6GS': [], '24PIX_8GS': [], '24PIX_16GS': [],
                             '32PIX_2GS': [], '32PIX_4GS': [], '32PIX_6GS': [], '32PIX_8GS': [], '32PIX_16GS': [],
                             '64PIX_2GS': [], '64PIX_4GS': [], '64PIX_6GS': [], '64PIX_8GS': [], '64PIX_16GS': [],
                             '128PIX_2GS': [], '128PIX_4GS': [], '128PIX_6GS': [], '128PIX_8GS': [], '128PIX_16GS': [],}

                    for n in range(len(os.listdir(data_path))): # len(com_list)
                        print(seed, n)

                        preprocessed_data_path =  os.path.join(data_path, f'comb{n}') # 16 classes 는 1 comb 밖에 없음.
                        high_analysis_path = os.path.join(preprocessed_data_path, f'High_Analysis_{test_type}')
                        
                        add_high_df = pd.read_csv(os.path.join(high_analysis_path, f'High_Level_Data_Analysis_{model_type1}_{model_type2}.csv'))
                        
                        # Split pred_prob_vector
                        d = {'p0' : [], 'p1': [], 'p2': [], 'p3': [], 'p4': [], 'p5': [], 'p6': [], 'p7': [],
                             'p8': [], 'p9': [], 'p10': [], 'p11': [], 'p12': [], 'p13': [], 'p14': [], 'p15': []}
                        
                        img_list, hyperpar_list, par_list = [], [], []
                        for i in range(add_high_df.shape[0]):
                            file_name = add_high_df['file_name'][i].split('.')[0]
                            pix, gs, par = file_name.split('_')
                            per = add_high_df['actual_person'][i]
                            img = str(per) + '_' + str(pix) + '_' + str(gs) + '_' + str(par) + '.jpg'
                            img_list.append(img)
                            hyperpar_list.append(str(pix) + '_' + str(gs))
                            par_list.append(str(par))
                            
                            # For Top-2 accuracy
                            temp_str = add_high_df['pred_prob_vector'].iloc[i]
                            temp_split = temp_str.split('[')
                            temp_split = temp_split[1].split(']')
                            temp_split = temp_split[0].split(' ')
                    
                            temp_list = []
                            for num in temp_split:
                                if num != '':
                                    num = float(num)
                                    try:
                                        num = math.exp(num)
                                    except OverflowError:
                                        num = float('inf')
                                    temp_list.append(num)
                            s = sum(temp_list)
                            for j in range(c):
                                temp_val = temp_list[j] / s
                                d[f'p{j}'].append(temp_val)

                            try:
                                add_high_df['actual_prob'].iloc[i] = math.exp(float(add_high_df['actual_prob'].iloc[i])) / s
                            except OverflowError:
                                add_high_df['actual_prob'].iloc[i] = 1
                            try:
                                add_high_df['pred_prob'].iloc[i] = math.exp(float(add_high_df['pred_prob'].iloc[i])) / s
                            except OverflowError:
                                add_high_df['pred_prob'].iloc[i] = 1

                        add_high_df['img'] = img_list
                        add_high_df['hyperpar'] = hyperpar_list
                        add_high_df['par'] = par_list
                        
                        for j in range(c):
                            add_high_df[face_labels[j]] = d[f'p{j}']
                            add_high_df[face_labels[j]] = add_high_df[face_labels[j]].astype(float)
                                                        
                        hyperpar_list = []
                        for i, file_name in enumerate(add_high_df['file_name']):
                            pix, gs, _ = file_name.split('_')
                            hyperpar = pix + '_' + gs
                            hyperpar_list.append(hyperpar) 
                        add_high_df['Resolution'] = hyperpar_list
                        # add_high_df['Seed'] = [seed] * add_high_df.shape[0]
                        add_high_df['Seed'] = [f'{model_type}_{seed}'] * add_high_df.shape[0]
                    
                        mac_df2 = pd.concat([mac_df2, add_high_df], axis=0)

probs_df = mac_df2[face_labels]

top2_hit_rate = []

for i in range(probs_df.shape[0]):
    prob_vec = probs_df.iloc[i]
    prob_vec = list(prob_vec.values)
    
    sorted_prob_vec = sorted(prob_vec)
    sorted_prob_vec_ind = [i[0] for i in sorted(enumerate(prob_vec), key=lambda x: x[1])]
    
    M1 = sorted_prob_vec[-1]
    M1_index = sorted_prob_vec_ind[-1]
    M2 = sorted_prob_vec[-2]
    M2_index = sorted_prob_vec_ind[-2]
    
    first_pred_per = face_labels[M1_index]
    assert int(mac_df2['pred_person'].iloc[i]) == int(first_pred_per)
    sec_pred_per = face_labels[M2_index]
    
    if int(mac_df2['actual_person'].iloc[i]) == int(first_pred_per) or int(mac_df2['actual_person'].iloc[i]) == int(sec_pred_per):
        top2_hit_rate.append(1)
    else:
        top2_hit_rate.append(0)

mac_df2['Machines'] = top2_hit_rate

hyperpar_list = []
for i, file_name in enumerate(mac_df2['file_name']):
    pix, gs, _ = file_name.split('_')
    hyperpar = pix + '_' + gs
    hyperpar_list.append(hyperpar) 
mac_df2['Resolution'] = hyperpar_list

new_mac_df2 = mac_df2.groupby(['Resolution']).mean()['Machines']


## both
# scatter plot (feedback) # 6 -> Feedback -> 6A
mer_df3 = new_acc_df2.merge(new_mac_df2, on='Resolution')

class LinearReg():

    def __init__(self, new_x):
        self.new_x = new_x
    
    def equation_val(self, new_x):
        x, y = mer_df3['Machines'], mer_df3['Humans']
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        new_y = slope * new_x + intercept
        return new_y
    
    @staticmethod
    def equation():
        x, y = mer_df3['Machines'], mer_df3['Humans']
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        print(f'R: {r_value}, p-value: {p_value}')
        return slope, intercept

new_hyperpar_name_list = ['16PIX_2GS', '16PIX_4GS', '16PIX_8GS',
                          '32PIX_2GS', '32PIX_4GS', '32PIX_8GS',
                          '64PIX_2GS', '64PIX_4GS', '64PIX_8GS']
df = new_mac_df2[np.logical_not(new_mac_df2.index.isin(new_hyperpar_name_list))].to_frame() # Machines (all resolutions)
df.columns = ['Hit Rate of Machines']
diction = {key: [] for key in df.index}
for i in range(df.shape[0]):
    linear_eq = LinearReg(df['Hit Rate of Machines'].iloc[i])
    new_y = linear_eq.equation_val(df['Hit Rate of Machines'].iloc[i])
    
    diction[df.index[i]].append(df['Hit Rate of Machines'].iloc[i])
    diction[df.index[i]].append(new_y)

new_df = pd.DataFrame.from_dict(diction).T#.reset_index()
new_df.columns = ['Machines', 'Humans']
new_df2 = new_df.reset_index()
new_df2.columns = ['Resolution', 'Machines', 'Humans']

new_df2


# Prediction (TOP-2) - # 2 ANN

## human
human_df = orig_human_df.copy()
human_df = human_df.fillna(0)
human_df['유저식별아이디'] = human_df['유저식별아이디'].astype(int)
human_df.index = human_df['유저식별아이디']

sel_col = []
for (i, col) in enumerate(human_df.columns):
    split = col.split('_')
    if split[0] == '선택' and split[1] == 'A':
        sel_col.append(col)
    
acc_df = human_df[sel_col]
acc_df.columns = [n for n in range(acc_df.shape[1])]
acc_df = acc_df.astype(int)

for q in range(acc_df.shape[1]):
    act_per = answer_df.loc['act_per'][q] 
    for s in range(acc_df.shape[0]):
        pred_per = acc_df.iloc[s, q]
        try:
            if str(int(pred_per)) == str(int(act_per)):
                acc_df.iloc[s, q] = 1 # modify the value of the cell (s, t)
            else:
                acc_df.iloc[s, q] = 0
        except:
            acc_df.iloc[s, q] = 0

acc_df.columns = answer_df.T['Answer']

## machine
test_type_list = [type] #['opt', 'elec']
model_type1_list = [''] #['PCA', 'PCA', '', '']
model_type2_list = ['CNN_SVC'] #['SVC', 'LR', 'CNN_LR', 'CNN_SVC']
seed_list = [22, 77, 2, 100, 81, 42, 7, 55, 50] # [22, 77, 2, 100, 81, 42, 7, 1, 55, 50] # different 1,000 training images (total of 10 permutations)
pix_order_list = ['16PIX', '24PIX', '32PIX', '64PIX', '128PIX']
gs_order_list = ['2GS', '4GS', '6GS', '8GS', '16GS']
class_list = [16] # [2, 4, 16]

df = pd.read_csv('E:\\ANNA_INTERN\\210827_ANNA_Removing_uncontaminated_data.csv')
l = list(range(df.shape[0]))
n = 16
random.seed(22)
set_1 = random.sample(l, n)
sets = [set_1]
r = class_list[0]

new_seed_list = []
for test_type in test_type_list:
    mac_df2 = pd.DataFrame()
    for (model_type1, model_type2) in zip(model_type1_list, model_type2_list): 
        model_type = model_type1 + model_type2
        for c in class_list:
            for m in range(1): 
                input_folder = [df.iloc[i, 0] for i in sets[m]] 
                assert len(input_folder) == 16
                com_obj = itertools.combinations(input_folder, r)
                com_list = list(com_obj)

                # mac_df2 = pd.DataFrame()
                for seed in seed_list:
                    new_seed_list.append(f'{model_type}_{seed}')
                    data_path = f'E:\\ANNA_INTERN\\Question Banks AI Hub_final\\{c}classes\\set{m}\\seed{seed}'

                    for n in range(len(os.listdir(data_path))): # len(com_list)
                        print(seed, n)

                        preprocessed_data_path =  os.path.join(data_path, f'comb{n}') # 16 classes 는 1 comb 밖에 없음.
                        high_analysis_path = os.path.join(preprocessed_data_path, f'High_Analysis_{test_type}')
                        
                        add_high_df = pd.read_csv(os.path.join(high_analysis_path, f'High_Level_Data_Analysis_{model_type1}_{model_type2}.csv'))
                        
                        # Split pred_prob_vector
                        d = {'p0' : [], 'p1': [], 'p2': [], 'p3': [], 'p4': [], 'p5': [], 'p6': [], 'p7': [],
                             'p8': [], 'p9': [], 'p10': [], 'p11': [], 'p12': [], 'p13': [], 'p14': [], 'p15': []}
                        
                        img_list, hyperpar_list, par_list = [], [], []
                        for i in range(add_high_df.shape[0]):
                            file_name = add_high_df['file_name'][i].split('.')[0]
                            pix, gs, par = file_name.split('_')
                            per = add_high_df['actual_person'][i]
                            img = str(per) + '_' + str(pix) + '_' + str(gs) + '_' + str(par) + '.jpg'
                            img_list.append(img)
                            hyperpar_list.append(str(pix) + '_' + str(gs))
                            par_list.append(str(par))
                            
                            # For Top-2 accuracy
                            temp_str = add_high_df['pred_prob_vector'].iloc[i]
                            temp_split = temp_str.split('[')
                            temp_split = temp_split[1].split(']')
                            temp_split = temp_split[0].split(' ')
                    
                            temp_list = []
                            for num in temp_split:
                                if num != '':
                                    num = float(num)
                                    try:
                                        num = math.exp(num)
                                    except OverflowError:
                                        num = float('inf')
                                    temp_list.append(num)
                            s = sum(temp_list)
                            for j in range(c):
                                temp_val = temp_list[j] / s
                                d[f'p{j}'].append(temp_val)

                            try:
                                add_high_df['actual_prob'].iloc[i] = math.exp(float(add_high_df['actual_prob'].iloc[i])) / s
                            except OverflowError:
                                add_high_df['actual_prob'].iloc[i] = 1
                            try:
                                add_high_df['pred_prob'].iloc[i] = math.exp(float(add_high_df['pred_prob'].iloc[i])) / s
                            except OverflowError:
                                add_high_df['pred_prob'].iloc[i] = 1

                        add_high_df['img'] = img_list
                        add_high_df['hyperpar'] = hyperpar_list
                        add_high_df['par'] = par_list
                        
                        for j in range(c):
                            add_high_df[face_labels[j]] = d[f'p{j}']
                            add_high_df[face_labels[j]] = add_high_df[face_labels[j]].astype(float)
                                                        
                        hyperpar_list = []
                        for i, file_name in enumerate(add_high_df['file_name']):
                            pix, gs, _ = file_name.split('_')
                            hyperpar = pix + '_' + gs
                            hyperpar_list.append(hyperpar) 
                        add_high_df['Resolution'] = hyperpar_list
                        # add_high_df['Seed'] = [seed] * add_high_df.shape[0]
                        add_high_df['Seed'] = [f'{model_type}_{seed}'] * add_high_df.shape[0]
                    
                        mac_df2 = pd.concat([mac_df2, add_high_df], axis=0)

probs_df = mac_df2[face_labels]

top2_hit_rate = []

for i in range(probs_df.shape[0]):
    prob_vec = probs_df.iloc[i]
    prob_vec = list(prob_vec.values)
    
    sorted_prob_vec = sorted(prob_vec)
    sorted_prob_vec_ind = [i[0] for i in sorted(enumerate(prob_vec), key=lambda x: x[1])]
    
    M1 = sorted_prob_vec[-1]
    M1_index = sorted_prob_vec_ind[-1]
    M2 = sorted_prob_vec[-2]
    M2_index = sorted_prob_vec_ind[-2]
    
    first_pred_per = face_labels[M1_index]
    assert int(mac_df2['pred_person'].iloc[i]) == int(first_pred_per)
    sec_pred_per = face_labels[M2_index]
    
    if int(mac_df2['actual_person'].iloc[i]) == int(first_pred_per) or int(mac_df2['actual_person'].iloc[i]) == int(sec_pred_per):
        top2_hit_rate.append(1)
    else:
        top2_hit_rate.append(0)

mac_df2['Hit Rate'] = top2_hit_rate
              
new_hyperpar_name_list = ['16PIX_2GS', '16PIX_4GS', '16PIX_8GS', 
                          '32PIX_2GS', '32PIX_4GS', '32PIX_8GS',  
                          '64PIX_2GS', '64PIX_4GS', '64PIX_8GS']
par_list = ['S001L1E01C4', 'S001L1E01C7', 'S001L1E01C10',
            'S001L1E02C7',
            'S001L1E03C7']

mac_df = mac_df2[np.logical_not(mac_df2['hyperpar'].isin(new_hyperpar_name_list))] # except for human parameters
mac_df = mac_df[mac_df['par'].isin(par_list)]
# mac_df = mac_df.groupby('img').mean().reset_index()

mac_df = mac_df.pivot(index='Seed', columns='img', values='Hit Rate')

mac_df_copy = mac_df.copy()
mac_df_T = mac_df_copy.T

con_list = []
for img in mac_df_T.index:
    _, pix, gs, par = img.split('_')
    con_list.append(f'{pix}_{gs}_{par}')
mac_df_T['con'] = con_list
mac_df_T_gp = mac_df_T.groupby('con').mean()


hyper_par_list, par_list = [], []
for img in mac_df_T_gp.index:
    pix, gs, par = img.split('_')
    hyper_par_list.append(f'{pix}_{gs}')
    par_list.append(par.split('.')[0])
mac_df_T_gp['hyperpar'] = hyper_par_list
mac_df_T_gp['par'] = par_list

mac_df_T_gp['Avg of ANNs'] = mac_df_T_gp[new_seed_list].mean(axis=1)
mac_df_T_gp = mac_df_T_gp[['Avg of ANNs', 'hyperpar', 'par']]


class ANN():
    def __init__(self, new_x, checkpoint_file):
        self.new_x = new_x
        self.checkpoint_file = checkpoint_file
    
    @staticmethod
    def create_model():
        
        model = models.Sequential()
        model.add(keras.Input(shape=(6, )))
        model.add(layers.Dense(3, activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(5, activation='relu'))
        print(model.summary())
        
        return model
    
    def load_model(self, model):
        
        model = keras.models.load_model(self.checkpoint_file)
        
        return model
    
    def eval_mod(self, w_model):
        new_ys = w_model(self.new_x)
        return new_ys

df = pd.DataFrame()
new_h_list = list(set(mac_df_T_gp['hyperpar'].values))
for l, hyperpar in enumerate(new_h_list): # for all resolutions
    par_old_df = mac_df_T_gp[mac_df_T_gp['hyperpar'] == hyperpar]
    temp_df = par_old_df['Avg of ANNs'].to_frame().T
    
    par_list = list(range(1, 6))
    temp_df.columns = [f'ANN_{j}' for j in par_list]
    
    df = pd.concat([df, temp_df], axis=0)
df['Cor'] = [0.7598]*df.shape[0]
df.index = new_h_list

ann_inst = ANN(df.values, 'Chcekpoints_FB/Checkpoint_3.h5') # Checkpoint_3, Checkpoint_4
mod = ann_inst.create_model()
w_mod = ann_inst.load_model(mod)
new_ys = ann_inst.eval_mod(w_mod)

diction = {x: [] for x in df.index}
for i in range(df.shape[0]):
    diction[df.index[i]].append(np.mean(df.iloc[i, :].values))
    diction[df.index[i]].append(np.mean(new_ys[i, :]))

new_df = pd.DataFrame.from_dict(diction).T#.reset_index()
new_df.columns = ['Machines', 'Humans']
new_df2_ann = new_df.reset_index()
new_df2_ann.columns = ['Resolution', 'Machines', 'Humans']

new_df2_ann

# Graph for humans

new_hyperpar_name_list = ['16PIX_2GS', '16PIX_4GS', '16PIX_8GS', 
                          '32PIX_2GS', '32PIX_4GS', '32PIX_8GS',  
                          '64PIX_2GS', '64PIX_4GS', '64PIX_8GS']
new_hyperpar_name_list = ['16PIX_2GS', '16PIX_4GS', '16PIX_6GS', '16PIX_8GS', '16PIX_16GS',
                          '24PIX_2GS', '24PIX_4GS', '24PIX_6GS', '24PIX_8GS', '24PIX_16GS',
                          '32PIX_2GS', '32PIX_4GS', '32PIX_6GS', '32PIX_8GS', '32PIX_16GS',
                          '64PIX_2GS', '64PIX_4GS', '64PIX_6GS', '64PIX_8GS', '64PIX_16GS',
                          '128PIX_2GS', '128PIX_4GS', '128PIX_6GS', '128PIX_8GS', '128PIX_16GS']

df_list = [per_df2, per_df1]
df2_list = [new_df2_ann, new_df1_ann] #[new_df2_ann, new_df1_ann] # [new_df2, new_df1]
color_list = ['pink', 'crimson']
label_list = ['Top-2 Accuracy', 'Top-1 Accuracy']

plt.figure(figsize=(10, 4))
for i in range(len(color_list)): # Top-1 Acc & Top-2 Acc
    df = df_list[i]
    df = df[df['Resolution'].isin(new_hyperpar_name_list)]
    df['Resolution'] = pd.Categorical(df['Resolution'], categories=new_hyperpar_name_list, ordered=True)
    sns.barplot(x='Resolution', y=f'Hit Rate', data=df, color=color_list[i], label=label_list[i], errwidth=1, capsize=0.3)

    df2 = df2_list[i]
    df2['Resolution'] = pd.Categorical(df2['Resolution'], categories=new_hyperpar_name_list, ordered=True)
    sns.barplot(x='Resolution', y='Humans', data=df2, edgecolor=color_list[i], facecolor=(1,1,1,0))

plt.xticks(rotation=90)
plt.legend()
plt.ylim([0, 1])
sns.despine()
plt.show()

pix_list = []
for (i, res) in enumerate(df['Resolution']):
    pix, _ = res.split('_')
    pix_list.append(pix)
    
df['PIX'] = pix_list

df.groupby('PIX').mean()

# Machines

test_type_list = [type]
model_type1_list = ['PCA'] #['PCA', 'PCA', '', '']
model_type2_list = ['SVC'] #['SVC', 'LR', 'CNN_LR', 'CNN_SVC']
seed_list = [22, 77, 2, 100, 81, 42, 7, 55, 50] # [22, 77, 2, 100, 81, 42, 7, 1, 55, 50] # different 1,000 training images (total of 10 permutations)
pix_order_list = ['16PIX', '24PIX', '32PIX', '64PIX', '128PIX']
gs_order_list = ['2GS', '4GS', '6GS', '8GS', '16GS']
class_list = [16] # [2, 4, 16]

df = pd.read_csv('C:\\Users\\user\\Desktop\\210827_ANNA_Removing_uncontaminated_data.csv')
l = list(range(df.shape[0]))
n = 16
random.seed(22)
set_1 = random.sample(l, n)
sets = [set_1]
r = class_list[0]

for test_type in test_type_list:
    for (model_type1, model_type2) in zip(model_type1_list, model_type2_list): 
        model_type = model_type1 + model_type2
        for c in class_list:
            for m in range(1): 
                input_folder = [df.iloc[i, 0] for i in sets[m]]
                assert len(input_folder) == 16
                com_obj = itertools.combinations(input_folder, r)
                com_list = list(com_obj)

                mac_df1 = pd.DataFrame()
                for seed in seed_list:
                    # seed = random.choice(seed_list) # For each set, we have random seed.
                    data_path = f'E:\\ANNA_INTERN\\Question Banks AI Hub_final\\{c}classes\\set{m}\\seed{seed}'
                    res_d = {'16PIX_2GS': [], '16PIX_4GS': [], '16PIX_6GS': [], '16PIX_8GS': [], '16PIX_16GS': [],
                             '24PIX_2GS': [], '24PIX_4GS': [], '24PIX_6GS': [], '24PIX_8GS': [], '24PIX_16GS': [],
                             '32PIX_2GS': [], '32PIX_4GS': [], '32PIX_6GS': [], '32PIX_8GS': [], '32PIX_16GS': [],
                             '64PIX_2GS': [], '64PIX_4GS': [], '64PIX_6GS': [], '64PIX_8GS': [], '64PIX_16GS': [],
                             '128PIX_2GS': [], '128PIX_4GS': [], '128PIX_6GS': [], '128PIX_8GS': [], '128PIX_16GS': [],}
                    #try:
                    for n in range(len(os.listdir(data_path))): # len(com_list)
                        print(seed, n)

                        preprocessed_data_path =  os.path.join(data_path, f'comb{n}') 
                        high_analysis_path = os.path.join(preprocessed_data_path, f'High_Analysis_{test_type}')
                        
                        add_high_df = pd.read_csv(os.path.join(high_analysis_path, f'High_Level_Data_Analysis_{model_type1}_{model_type2}.csv'))
                        add_high_df['hit_rate'] = add_high_df['correctness'].replace(['correct', 'wrong'], [1, 0]) 
                        add_high_df = add_high_df[['file_name', 'hit_rate']]
                        
                        hyperpar_list = []
                        for i, file_name in enumerate(add_high_df['file_name']):
                            pix, gs, _ = file_name.split('_')
                            hyperpar = pix + '_' + gs
                            hyperpar_list.append(hyperpar) 
                        add_high_df['Resolution'] = hyperpar_list
                        
                        for (p, res) in enumerate(add_high_df['Resolution']):
                            v = add_high_df['hit_rate'][p]
                            res_d[res].append(v)

                    # except:
                    #     print(c, seed, n, 'error')
                    #     pass
                    temp_df = pd.DataFrame.from_dict(res_d)
                    temp_df = temp_df.mean().to_frame().reset_index()
                    temp_df.columns = ['Resolution', 'Hit Rate']
                    seed_rows = [seed] * temp_df.shape[0]
                    temp_df['Seed'] = seed_rows
                    
                    mac_df1 = pd.concat([mac_df1, temp_df], axis=0)
                #high_df = high_df.groupby('image').mean().reset_index()

mac_df1.head()            

new_hyperpar_name_list = ['16PIX_2GS', '16PIX_4GS', '16PIX_8GS', 
                          '32PIX_2GS', '32PIX_4GS', '32PIX_8GS',  
                          '64PIX_2GS', '64PIX_4GS', '64PIX_8GS']
new_hyperpar_name_list = ['16PIX_2GS', '16PIX_4GS', '16PIX_6GS', '16PIX_8GS', '16PIX_16GS',
                          '24PIX_2GS', '24PIX_4GS', '24PIX_6GS', '24PIX_8GS', '24PIX_16GS',
                          '32PIX_2GS', '32PIX_4GS', '32PIX_6GS', '32PIX_8GS', '32PIX_16GS',
                          '64PIX_2GS', '64PIX_4GS', '64PIX_6GS', '64PIX_8GS', '64PIX_16GS',
                          '128PIX_2GS', '128PIX_4GS', '128PIX_6GS', '128PIX_8GS', '128PIX_16GS']


plt.figure(figsize=(8, 4))
df_list = [mac_df1]
color_list = ['steelblue']
color_list = ['darkolivegreen']
label_list = ['Top-1 Accuracy']
hit_rate_list = ['Hit Rate']

for i in range(len(color_list)): # Top-1 Acc & Top-2 Acc
    df = df_list[i]
    df = df[df['Resolution'].isin(new_hyperpar_name_list)]
    df['Resolution'] = pd.Categorical(df['Resolution'], categories=new_hyperpar_name_list, ordered=True)
    sns.barplot(x='Resolution', y=hit_rate_list[i], data=df, color=color_list[i], label=label_list[i])

plt.xticks(rotation=60)
plt.legend()
plt.ylim([0, 1])
plt.show()

test_type_list = [type]
model_type1_list = ['PCA'] #['PCA', 'PCA', '', '']
model_type2_list = ['LR'] #['SVC', 'LR', 'CNN_LR', 'CNN_SVC']
seed_list = [22, 77, 2, 100, 81, 42, 7, 55, 50] # [22, 77, 2, 100, 81, 42, 7, 1, 55, 50] # different 1,000 training images (total of 10 permutations)
pix_order_list = ['16PIX', '24PIX', '32PIX', '64PIX', '128PIX']
gs_order_list = ['2GS', '4GS', '6GS', '8GS', '16GS']
class_list = [16] # [2, 4, 16]

df = pd.read_csv('C:\\Users\\user\\Desktop\\210827_ANNA_Removing_uncontaminated_data.csv')
l = list(range(df.shape[0]))
n = 16
random.seed(22)
set_1 = random.sample(l, n)
sets = [set_1]
r = class_list[0]
face_labels = [19081632,
               19080133,
               19092711,
               19070311,
               19090631,
               19092521,
               19071821,
               19081421,
               19090222,
               19082032,
               19082131,
               19070231, 
               19070522,
               19071131,
               19072221,
               19072922]

for test_type in test_type_list:
    for (model_type1, model_type2) in zip(model_type1_list, model_type2_list): 
        model_type = model_type1 + model_type2
        for c in class_list:
            for m in range(1): 
                input_folder = [df.iloc[i, 0] for i in sets[m]] 
                assert len(input_folder) == 16
                com_obj = itertools.combinations(input_folder, r)
                com_list = list(com_obj)

                mac_df2 = pd.DataFrame()
                for seed in seed_list:
                    # seed = random.choice(seed_list) # For each set, we have random seed.
                    data_path = f'E:\\ANNA_INTERN\\Question Banks AI Hub_final\\{c}classes\\set{m}\\seed{seed}'
                    res_d = {'16PIX_2GS': [], '16PIX_4GS': [], '16PIX_6GS': [], '16PIX_8GS': [], '16PIX_16GS': [],
                             '24PIX_2GS': [], '24PIX_4GS': [], '24PIX_6GS': [], '24PIX_8GS': [], '24PIX_16GS': [],
                             '32PIX_2GS': [], '32PIX_4GS': [], '32PIX_6GS': [], '32PIX_8GS': [], '32PIX_16GS': [],
                             '64PIX_2GS': [], '64PIX_4GS': [], '64PIX_6GS': [], '64PIX_8GS': [], '64PIX_16GS': [],
                             '128PIX_2GS': [], '128PIX_4GS': [], '128PIX_6GS': [], '128PIX_8GS': [], '128PIX_16GS': [],}
                    #try:
                    for n in range(len(os.listdir(data_path))): # len(com_list)
                        print(seed, n)

                        preprocessed_data_path =  os.path.join(data_path, f'comb{n}') # 16 classes 는 1 comb 밖에 없음.
                        high_analysis_path = os.path.join(preprocessed_data_path, f'High_Analysis_{test_type}')
                        
                        add_high_df = pd.read_csv(os.path.join(high_analysis_path, f'High_Level_Data_Analysis_{model_type1}_{model_type2}.csv'))
                        
                        # Splitting pred_prob_vector
                        d = {'p0' : [], 'p1': [], 'p2': [], 'p3': [], 'p4': [], 'p5': [], 'p6': [], 'p7': [],
                             'p8': [], 'p9': [], 'p10': [], 'p11': [], 'p12': [], 'p13': [], 'p14': [], 'p15': []}
                        
                        for i in range(add_high_df.shape[0]):
                            temp_str = add_high_df['pred_prob_vector'].iloc[i]
                            temp_split = temp_str.split('[')
                            temp_split = temp_split[1].split(']')
                            temp_split = temp_split[0].split(' ')
                    
                            temp_list = []
                            for num in temp_split:
                                if num != '':
                                    num = float(num)
                                    try:
                                        num = math.exp(num)
                                    except OverflowError:
                                        num = float('inf')
                                    temp_list.append(num)
                            s = sum(temp_list)
                            for j in range(c):
                                temp_val = temp_list[j] / s
                                d[f'p{j}'].append(temp_val)

                            try:
                                add_high_df['actual_prob'].iloc[i] = math.exp(float(add_high_df['actual_prob'].iloc[i])) / s
                            except OverflowError:
                                add_high_df['actual_prob'].iloc[i] = 1
                            try:
                                add_high_df['pred_prob'].iloc[i] = math.exp(float(add_high_df['pred_prob'].iloc[i])) / s
                            except OverflowError:
                                add_high_df['pred_prob'].iloc[i] = 1

                        for j in range(c):
                            add_high_df[face_labels[j]] = d[f'p{j}']
                            add_high_df[face_labels[j]] = add_high_df[face_labels[j]].astype(float)
                                                        
                        hyperpar_list = []
                        for i, file_name in enumerate(add_high_df['file_name']):
                            pix, gs, _ = file_name.split('_')
                            hyperpar = pix + '_' + gs
                            hyperpar_list.append(hyperpar) 
                        add_high_df['Resolution'] = hyperpar_list
                                            
                    seed_rows = [seed] * add_high_df.shape[0]
                    add_high_df['Seed'] = seed_rows
                    
                    mac_df2 = pd.concat([mac_df2, add_high_df], axis=0)


mac_df2.head()            

probs_df = mac_df2[face_labels]

top2_hit_rate = []

for i in range(probs_df.shape[0]):
    prob_vec = probs_df.iloc[i]
    prob_vec = list(prob_vec.values)
    
    sorted_prob_vec = sorted(prob_vec)
    sorted_prob_vec_ind = [i[0] for i in sorted(enumerate(prob_vec), key=lambda x: x[1])]
    
    M1 = sorted_prob_vec[-1]
    M1_index = sorted_prob_vec_ind[-1]
    M2 = sorted_prob_vec[-2]
    M2_index = sorted_prob_vec_ind[-2]
    
    first_pred_per = face_labels[M1_index]
    assert int(mac_df2['pred_person'].iloc[i]) == int(first_pred_per)
    sec_pred_per = face_labels[M2_index]
    
    if int(mac_df2['actual_person'].iloc[i]) == int(first_pred_per) or int(mac_df2['actual_person'].iloc[i]) == int(sec_pred_per):
        top2_hit_rate.append(1)
    else:
        top2_hit_rate.append(0)

mac_df2['top2_hit_rate'] = top2_hit_rate

mac_df2

hyperpar_list = []
for i, file_name in enumerate(mac_df2['file_name']):
    pix, gs, _ = file_name.split('_')
    hyperpar = pix + '_' + gs
    hyperpar_list.append(hyperpar) 
mac_df2['Resolution'] = hyperpar_list

new_mac_df2 = mac_df2.groupby(['Resolution', 'Seed']).mean().reset_index()[['Resolution', 'Seed', 'top2_hit_rate']]
new_mac_df2

new_mac_df2[new_mac_df2['Seed'] == 22].groupby('Resolution').mean()

new_hyperpar_name_list = ['16PIX_2GS', '16PIX_4GS', '16PIX_8GS', 
                          '32PIX_2GS', '32PIX_4GS', '32PIX_8GS',  
                          '64PIX_2GS', '64PIX_4GS', '64PIX_8GS']
new_hyperpar_name_list = ['16PIX_2GS', '16PIX_4GS', '16PIX_6GS', '16PIX_8GS', '16PIX_16GS',
                          '24PIX_2GS', '24PIX_4GS', '24PIX_6GS', '24PIX_8GS', '24PIX_16GS',
                          '32PIX_2GS', '32PIX_4GS', '32PIX_6GS', '32PIX_8GS', '32PIX_16GS',
                          '64PIX_2GS', '64PIX_4GS', '64PIX_6GS', '64PIX_8GS', '64PIX_16GS',
                          '128PIX_2GS', '128PIX_4GS', '128PIX_6GS', '128PIX_8GS', '128PIX_16GS']


plt.figure(figsize=(10, 4))
df_list = [new_mac_df2, mac_df1]
color_list = ['lightsteelblue', 'steelblue']
#color_list = ['olivedrab', 'darkolivegreen']
label_list = ['Top-2 Accuracy', 'Top-1 Accuracy']
hit_rate_list = ['top2_hit_rate', 'Hit Rate']

for i in range(len(color_list)): # Top-1 Acc & Top-2 Acc
    df = df_list[i]
    df = df[df['Resolution'].isin(new_hyperpar_name_list)]
    df['Resolution'] = pd.Categorical(df['Resolution'], categories=new_hyperpar_name_list, ordered=True)
    sns.barplot(x='Resolution', y=hit_rate_list[i], data=df, color=color_list[i], label=label_list[i], errwidth=1, capsize=0.3)

plt.xticks(rotation=90)
plt.legend()
plt.ylim([0, 1])
sns.despine()
plt.show()

new_hyperpar_name_list = ['16PIX_2GS', '16PIX_4GS', '16PIX_8GS', 
                          '32PIX_2GS', '32PIX_4GS', '32PIX_8GS',  
                          '64PIX_2GS', '64PIX_4GS', '64PIX_8GS']
df = df[df['Resolution'].isin(new_hyperpar_name_list)]

pix_list = []
for (i, res) in enumerate(df['Resolution']):
    pix, _ = res.split('_')
    pix_list.append(pix)
df['PIX'] = pix_list

df.groupby('PIX').mean()
