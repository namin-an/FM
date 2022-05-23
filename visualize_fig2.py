"""
Running this file makes visualizions of results in the paper.

- Fig. 2a (Fig. S5a, Ext. Fig. 1a)
- Fig. 2b (Fig. S5c, Fig. S5d, Ext. Fig. 1c, Ext. Fig. 1d)
- Fig. 2c (Fig. S5e, Fig. S5f, Ext. Fig. 1e, Ext. Fig. 1f, Ext. Fig. 1g)
- Fig. 2d (Fig. S5g, Ext. Fig. 1h, Ext. Fig. 1i)

"""



#%%
import os
import random 
import itertools 
from itertools import combinations
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy
from scipy import stats
from scipy.stats import norm
import scipy.stats as sp
from statannot import add_stat_annotation


#%%
# Fig. 2a (Fig. S5a, Ext. Fig. 1a)

test_type_list = ['normal'] #'opt' for Fig. 2a, 'normal' for Fig. S5a, 'elec' for Fig. S6a
model_type1_list = ['', '', '', '', '', '', '', '', '', '', 'PCA', 'PCA', '', ''] #['PCA', 'PCA', '', '']
model_type2_list = ['PIXEL_LR','PIXEL_SVC', 'CNN_LR', 'CNN_SVC', 'CNN_VggNet2', 'CNN_VggNet2_SVC', 'CNN_AlexNet2', 'CNN_AlexNet2_SVC',
                    'CNN_ResNet2', 'CNN_ResNet2_SVC', 'SVC', 'LR', 'CNN_SVCft', 'CNN_AlexNet2ft']
seed_list = [22, 77, 2, 100, 81, 42, 7, 1, 55, 50] # different 1,000 training images (total of 10 permutations)

for test_type in test_type_list:
    for (model_type1, model_type2) in zip(model_type1_list, model_type2_list):
        model_type = model_type1 + model_type2

        high_df = pd.DataFrame()
        for m in range(1): 
            for seed in seed_list:
                data_path = f'E:\\ANNA_INTERN\\Question Banks AI Hub_final\\16classes\\set{m}\\seed{seed}'
                preprocessed_data_path =  os.path.join(data_path, 'comb0') 

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
            image_list.append(f'{face}_{file_name}') # for img (scatter-plot of two probs)
        high_df['image'] = image_list

        high_df = high_df[['image', 'hit_rate']]

        old_high_df_cols = high_df.columns
        new_high_df_cols = [f'{model_type}_' + col_name for (i, col_name) in enumerate(old_high_df_cols) if i != 0]
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
        elif model_type == 'CNN_ResNet2':
            cnn_resnet_high_df = high_df
        elif model_type == 'CNN_ResNet2_SVC':
            cnn_resnet_svc_high_df = high_df
        elif model_type == 'PIXEL_SVC':
            pixel_svc_high_df = high_df
        elif model_type == 'PIXEL_LR':
            pixel_lr_high_df = high_df
        elif model_type == 'CNN_SVCft':
            cnn_svcft_high_df = high_df
        elif model_type == 'CNN_AlexNet2ft':
            cnn_alexnetft_high_df = high_df    

model_list = [
              pixel_svc_high_df, pixel_lr_high_df,
              pca_svc_high_df, pca_lr_high_df,
              cnn_svc_high_df, cnn_lr_high_df,
              cnn_alexnet_svc_high_df, cnn_alexnet_high_df,
              cnn_vgg_svc_high_df, cnn_vgg_high_df,  
              cnn_resnet_svc_high_df, cnn_resnet_high_df,
              cnn_svcft_high_df, cnn_alexnetft_high_df]
model_name_list = [
                   'PIXEL_SVC', 'PIXEL_LR',
                   'PCASVC', 'PCALR',
                   'CNN_SVC', 'CNN_LR',
                   'CNN_AlexNet2_SVC', 'CNN_AlexNet2',
                   'CNN_VggNet2_SVC', 'CNN_VggNet2', 
                   'CNN_ResNet2_SVC', 'CNN_ResNet2',
                   'CNN_SVCft', 'CNN_AlexNet2ft'] # same order as the above list
temp_high_df = model_list[0].merge(model_list[1], on='image')
for i in range(2, len(model_list)):
    temp_high_df = temp_high_df.merge(model_list[i], on='image')

temp_high_df = temp_high_df.drop(temp_high_df.tail(1).index)
temp_high_df = temp_high_df.set_index('image')
temp_high_df.columns = model_name_list

temp_high_df2 = temp_high_df.copy()

temp_high_df2 = temp_high_df2.stack().to_frame()
temp_high_df2.columns = ['Hit Rate']

if test_type == 'opt' or test_type == 'elec':
    h_list, pix_list, gs_list = [], [], []
    for i in range(temp_high_df2.shape[0]):
        filename = temp_high_df2.index[i][0]
        _, pix, gs, _ = filename.split('_')
        h_list.append(f'{pix}_{gs}')
        pix_list.append(pix)
        gs_list.append(gs)
    temp_high_df2['hyperpar'] = h_list
    temp_high_df2['PIX'] = pix_list
    temp_high_df2['GS'] = gs_list

    temp_high_df2.index = temp_high_df2.index.droplevel(level=0)
    temp_high_df2 = temp_high_df2.reset_index()
    temp_high_df2.columns = ['Models', 'Hit Rate', 'hyperpar', 'PIX', 'GS']
    
elif test_type == 'normal':
    temp_high_df2.index = temp_high_df2.index.droplevel(level=0)
    temp_high_df2 = temp_high_df2.reset_index()
    temp_high_df2.columns = ['Models', 'Hit Rate']
    # temp_high_df2['0'] = ['0'] * temp_high_df2.shape[0]
                             
pal =  ["#bbd17b", "#bbd17b",
        "#50b161", "#50b161",
        "#8ee5e8", "#8ee5e8",
        "#60c9f6", "#60c9f6",
        "#3475b7", "#3475b7",
        "#355d94", "#355d94",
        "#8ee5e8", '#60c9f6'
        ] # add more colors

if test_type == 'opt' or test_type == 'elec':

    pix_name_list = ['16PIX', '24PIX', '32PIX', '64PIX', '128PIX']
    cri_len = len(pix_name_list) * 2
    plt.figure(figsize=(12, 5))
    plot = sns.barplot(x='PIX', y='Hit Rate', hue='Models', data=temp_high_df2, order=pix_name_list, palette=pal, errwidth=1, capsize=0.03)

    j = 0
    for (i, onebar) in enumerate(plot.patches):
        if i % cri_len == 0:
            j = j+1
        k = i - cri_len*(j-1) # for each parameter
        if k >= (cri_len//2):
            clr = onebar.get_facecolor()
            onebar.set_edgecolor(clr)
            onebar.set_facecolor((1, 1, 1)) # set it to white

    plt.legend(bbox_to_anchor=(1, 0.5))
    sns.despine()
    plt.xticks(rotation=60)
    plt.ylim([0, 1])
    plt.show()



    gs_name_list = ['2GS', '4GS', '6GS', '8GS', '16GS']
    cri_len = len(gs_name_list) * 2
    plt.figure(figsize=(12, 5))
    plot = sns.barplot(x='GS', y='Hit Rate', hue='Models', data=temp_high_df2, order=gs_name_list, palette=pal, errwidth=1, capsize=0.03)

    j = 0
    for (i, onebar) in enumerate(plot.patches):
        if i % cri_len == 0:
            j = j+1
        k = i - cri_len*(j-1) 
        if k >= (cri_len//2):
            clr = onebar.get_facecolor()
            onebar.set_edgecolor(clr)
            onebar.set_facecolor((1, 1, 1)) # set it to white
            
    plt.legend(bbox_to_anchor=(1, 0.5))
    sns.despine()
    plt.xticks(rotation=60)
    plt.ylim([0, 1])
    plt.show()

elif test_type == 'normal':
    plt.figure(figsize=(6, 5))
    plot = sns.barplot(x='Models', y='Hit Rate', data=temp_high_df2, palette=pal, errwidth=1, capsize=0.3)

    j = 0
    for (i, onebar) in enumerate(plot.patches):
        if i % len(model_name_list) == 0:
            j = j+1
        i = i - len(model_name_list)*(j-1)
        if i % 2 != 0 and i != len(model_name_list)-1:
            clr = onebar.get_facecolor()
            onebar.set_edgecolor(clr)
            onebar.set_facecolor((1, 1, 1)) # set it to white
        elif i == len(model_name_list)-2 or i == len(model_name_list)-1:
            onebar.set_hatch('//')
            
    plt.legend(bbox_to_anchor=(1, 0.5))
    sns.despine()
    plt.xticks(rotation=90)
    plt.ylim([0, 1])
    plt.show()



#%%
# Fig. 2b (Fig. S5c, Fig. S5d, Ext. Fig. 1c, Ext. Fig. 1d)

test_type_list = ['elec'] #['opt', 'elec']
set_type1_list = [''] #['PCA', 'PCA', '', '']
set_type2_list = ['CNN_SVC'] #['SVC', 'LR', 'CNN_LR', 'CNN_SVC']
seed_list = [22, 77, 2, 100, 81, 42, 7, 1, 55, 50] # different 1,000 training images (total of 10 permutations)
pix_order_list = ['16PIX', '24PIX', '32PIX', '64PIX', '128PIX']
gs_order_list = ['2GS', '4GS', '6GS', '8GS', '16GS']


for test_type in test_type_list:
    for (set_type1, set_type2) in zip(set_type1_list, set_type2_list): # for the first set only
        set_type = set_type1 + '_' + set_type2
 
        high_df = pd.DataFrame() # initialize variables
        for m in range(14):
            for seed in seed_list:
                data_path = f'E:\\ANNA_INTERN\\Question Banks AI Hub_final\\16classes\\set{m}\\seed{seed}'
                preprocessed_data_path =  os.path.join(data_path, 'comb0') # only one combination for 16 classes

                high_analysis_path = os.path.join(preprocessed_data_path, f'High_Analysis_{test_type}')
                high_file_path = os.path.join(high_analysis_path, f'High_Level_Data_Analysis_{set_type}.csv')
                
                if os.path.isfile(high_file_path):
                    add_high_df = pd.read_csv(high_file_path)
                    add_high_df['hit_rate'] = add_high_df['correctness'].replace(['correct', 'wrong'], [1, 0]) 
                    add_high_df['Set'] = [m] * add_high_df.shape[0]
                    add_high_df = add_high_df[['file_name', 'hit_rate', 'Set']]
                    add_high_df['seed'] = [seed] * add_high_df.shape[0]
                    
                    high_df = pd.concat([high_df, add_high_df], axis=0) 
                else:
                    print(set_type, m, seed, 'error')
                    pass

temp_high_df = high_df.copy()

hyperpar_list = []
for (i, file_name) in enumerate(temp_high_df['file_name']):
    file_name = file_name.split('.')[0]
    pix, gs, par = file_name.split('_')
    
    hyperpar_list.append(str(pix)+'_'+str(gs)+'_'+str(par))

temp_high_df['hyperpar'] = hyperpar_list

temp_high_df = temp_high_df.groupby(['Set', 'hyperpar']).mean().reset_index()

pix_list = []
for (i, file_name) in enumerate(temp_high_df['hyperpar']):
    pix, _, _ = file_name.split('_')
    
    pix_list.append(pix)

temp_high_df['PIX'] = pix_list

pix_name_list = ['16PIX', '24PIX', '32PIX', '64PIX', '128PIX']
temp_high_df['PIX'] = pd.Categorical(temp_high_df['PIX'], categories=pix_name_list, ordered=True)

g = sns.scatterplot(x='Set', y='hit_rate', hue='PIX', data=temp_high_df,
                   palette='rocket') #, marker='s')
avg_high_df = temp_high_df.groupby('Set').mean().reset_index()
avg_high_df.columns = ['Set', 'hit_rate', 'Seed']
h = sns.lineplot(data=avg_high_df, x='Set', y='hit_rate', color=sns.color_palette('mako_r', 7)[1], linewidth=5, alpha=0.5)
plt.axhline(y=1/16, color='k', linestyle=':', alpha=0.5)

plt.legend(bbox_to_anchor=(1.05, 1))
sns.despine()
plt.ylim([0, 1])
plt.xlabel('Set #')
plt.ylabel('Hit Rate')
plt.savefig('C:\\Users\\user\\Desktop\\Fig.3-2.png')
plt.show()


l = list(set(temp_high_df['Set'].values))
set_com_list = list(combinations(l, 2))
temp_high_df['Set'] = temp_high_df['Set'].astype(str)
new_temp_high_df = temp_high_df.groupby(['hyperpar', 'Set']).mean().reset_index()

for (i, set_num) in enumerate(set_com_list):
    set_num1 = str(set_num[0])
    set_num2 = str(set_num[1])
    
    com1 = new_temp_high_df[new_temp_high_df['Set'] == set_num1] # 225 (5*5*9)
    com2 = new_temp_high_df[new_temp_high_df['Set'] == set_num2]    
    com1 = com1['hit_rate'].astype(float).values
    com2 = com2['hit_rate'].astype(float).values
    
    statistic, pvalue = scipy.stats.mannwhitneyu(x=com1, y=com2, use_continuity=False, alternative='two-sided')
    pvalue = '{:.2e}'.format(pvalue)
    pvalue = float(pvalue)
    if set_num1 == '10' or set_num2 == '10':
        if pvalue < 0.01:
            star = '***'
        elif pvalue >= 0.01 and pvalue < 0.05:
            star = '**'
        elif pvalue >= 0.05 and pvalue < 0.1:
            star = '*'
        else:
            star = 'ns'
        print(f'{set_num1}_{set_num2}: {star}') # Mann-Whitney-Wilcoxon test two-sided with Bonferroni correction, P-val={pvalue}, U_stat={statistic}', '\n')



#%%
# Fig. 2c (Fig. S5e, Fig. S5f, Ext. Fig. 1e, Ext. Fig. 1f, Ext. Fig. 1g)

test_type_list = ['elec'] #['opt', 'elec']
model_type1_list = [''] #['PCA', 'PCA', '', '']
model_type2_list = ['CNN_SVC'] #['SVC', 'LR', 'CNN_LR', 'CNN_SVC']
seed_list = [22, 77, 2, 100, 81, 42, 7, 1, 55, 50] # different 1,000 training images (total of 10 permutations)
pix_order_list = ['16PIX', '24PIX', '32PIX', '64PIX', '128PIX']
gs_order_list = ['2GS', '4GS', '6GS', '8GS', '16GS']
class_list = [2, 4, 16]

for test_type in test_type_list:
    for (model_type1, model_type2) in zip(model_type1_list, model_type2_list): # only the first model
        model_type = model_type1 + model_type2
        for c in class_list:
            for m in range(1): # only the first set
                high_df = pd.DataFrame()
                for seed in seed_list:
                    # seed = random.choice(seed_list) # For each set, we have random seed.
                    data_path = f'E:\\ANNA_INTERN\\Question Banks AI Hub_final\\{c}classes\\set{m}\\seed{seed}'
                    try:
                        for n in range(len(os.listdir(data_path))):
                            print(c, seed, n)
                            preprocessed_data_path =  os.path.join(data_path, f'comb{n}') 

                            high_analysis_path = os.path.join(preprocessed_data_path, f'High_Analysis_{test_type}')
                            
                            add_high_df = pd.read_csv(os.path.join(high_analysis_path, f'High_Level_Data_Analysis_{model_type1}_{model_type2}.csv'))
                            add_high_df['hit_rate'] = add_high_df['correctness'].replace(['correct', 'wrong'], [1, 0]) 
                            add_high_df = add_high_df[['file_name', 'hit_rate', 'actual_person']]
                            
                            image_list = []
                            for i, file_name in enumerate(add_high_df['file_name']):
                                file_name_list = file_name.split('.')
                                full_file_name = file_name_list[0]
                                face = add_high_df['actual_person'].iloc[i]
                                image = str(face) + '_' + str(full_file_name)
                                image_list.append(image)
                            add_high_df['image'] = image_list

                            add_high_df = add_high_df[['image', 'hit_rate']]
                            high_df = pd.concat([high_df, add_high_df], axis=0)
                    except:
                        print(c, seed, n, 'error')
                        break

                old_high_df_cols = high_df.columns
                new_high_df_cols = [f'{c}_' + col_name for (i, col_name) in enumerate(old_high_df_cols) if i != 0] 
                new_high_df_cols = [old_high_df_cols.tolist()[0], *new_high_df_cols]
                high_df.columns = new_high_df_cols

                high_df = pd.DataFrame(high_df.groupby('image').mean().reset_index()) 
                high_df = high_df[['image', f'{c}_hit_rate']]

                if c == 2:
                    high_df_2 = high_df
                elif c == 4:
                    high_df_4 = high_df
                elif c == 16:
                    high_df_16 = high_df


temp_high_df = high_df_2.merge(high_df_4, on='image')
temp_high_df = temp_high_df.merge(high_df_16, on='image')

pix_list, gs_list, full_hyperpar_list, par_list = [], [], [], []
for i, file_name in enumerate(temp_high_df['image']):
    _, pix, gs, par = file_name.split('_')
    full_hyperpar = str(pix) + '_' + str(gs)
    par = full_hyperpar + '_' + str(par)
    
    pix_list.append(pix)
    gs_list.append(gs)
    full_hyperpar_list.append(full_hyperpar)
    par_list.append(par)
    
temp_high_df['Pixels'] = pix_list 
temp_high_df['Gray-scales'] = gs_list
temp_high_df['full_hyperpar'] = full_hyperpar_list
temp_high_df['par'] = par_list

# --------- low-level ---------

df = temp_high_df

df = pd.DataFrame(df.groupby('full_hyperpar').agg(np.mean)).reset_index()
pix_list, gs_list = [], []
for i, file_name in enumerate(df['full_hyperpar']):
    file_name_list = file_name.split('_')
    pix, gs, _ = file_name_list[0], file_name_list[1], file_name_list[-1]
    pix_list.append(pix)
    gs_list.append(gs)

df['Pixels'] = pix_list 
df['Gray-scales'] = gs_list

com_class_list = list(itertools.combinations(class_list, 2))

for i in com_class_list:
    if i[0] >= i[1]:
        M, m = i[0], i[1]
    else:
        M, m = i[1], i[0]

    g = sns.lmplot(data=df, x=f'{M}_hit_rate', y=f'{m}_hit_rate', palette='rocket',
                   hue='Pixels', hue_order=['16PIX', '24PIX', '32PIX', '64PIX', '128PIX'], ci=None, legend_out=False)
    for ax in g.axes.ravel():
        ax.plot((0,1), (0,1), color='gray', lw=1, linestyle='--')

    r, p = sp.stats.pearsonr(df[f'{M}_hit_rate'], df[f'{m}_hit_rate'])
    ax = plt.gca()
    plt.text(x=0.5, y=0.15, s=(f'Pearson r={r: .2f}, p={p: .2e}'), transform=ax.transAxes) # (0, 0): lower-left, (1, 1): upper-right
    r, p = sp.stats.spearmanr(df[f'{M}_hit_rate'], df[f'{m}_hit_rate'])
    plt.text(x=0.5, y=0.1, s=(f'Spearman r={r: .2f}, p={p: .2e}'), transform=ax.transAxes)
    r, p = sp.stats.kendalltau(df[f'{M}_hit_rate'], df[f'{m}_hit_rate'])
    plt.text(x=0.5, y=0.05, s=(f'Kendall tau={r: .2f}, p={p: .2e}'), transform=ax.transAxes)

    # plt.title(f'Regression Accuracy Plot per Resolution ({model_type})')
    plt.xlabel(f'Hit Rate of {M} classes (16 C {M} combinations)')
    plt.ylabel(f'Hit Rate of {m} classes (16 C {m} combinations)')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.legend(loc='center right')
    ax.set_aspect('equal', adjustable='box')
    plt.show()

# --------- middle-level ---------

df = temp_high_df

df = pd.DataFrame(df.groupby('par').agg(np.mean)).reset_index()

pix_list, gs_list = [], []
for i, file_name in enumerate(df['par']):
    file_name_list = file_name.split('_')
    pix, gs, _ = file_name_list[0], file_name_list[1], file_name_list[-1]
    pix_list.append(pix)
    gs_list.append(gs)

df['Pixels'] = pix_list 
df['Gray-scales'] = gs_list

com_class_list = list(itertools.combinations(class_list, 2))

for i in com_class_list:
    if i[0] >= i[1]:
        M, m = i[0], i[1]
    else:
        M, m = i[1], i[0]

    g = sns.lmplot(data=df, x=f'{M}_hit_rate', y=f'{m}_hit_rate', palette='rocket',
                   hue='Pixels', hue_order=['16PIX', '24PIX', '32PIX', '64PIX', '128PIX'], ci=None, legend_out=False)
    for ax in g.axes.ravel():
        ax.plot((0,1), (0,1), color='gray', lw=1, linestyle='--')

    r, p = sp.stats.pearsonr(df[f'{M}_hit_rate'], df[f'{m}_hit_rate'])
    #ax = plt.gca()
    #plt.text(x=1.03, y=0.03, s=(f'Pearson r={r: .2f}'), transform=ax.transAxes) # (0, 0): lower-left, (1, 1): upper-right
    print(r)
    # r, p = sp.stats.spearmanr(df[f'{M}_hit_rate'], df[f'{m}_hit_rate'])
    # plt.text(x=0.5, y=0.1, s=(f'Spearman r={r: .2f}, p={p: .2e}'), transform=ax.transAxes)
    # r, p = sp.stats.kendalltau(df[f'{M}_hit_rate'], df[f'{m}_hit_rate'])
    # plt.text(x=0.5, y=0.05, s=(f'Kendall tau={r: .2f}, p={p: .2e}'), transform=ax.transAxes)

    # plt.title(f'Regression Accuracy Plot per Resolution ({model_type})')
    plt.xlabel(f'Hit Rate of {model_type} with Class Size of {M}')
    plt.ylabel(f'Hit Rate of {model_type} with Class Size of {m}')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.legend() #bbox_to_anchor=(1.1, 1.05))
    ax.set_aspect('equal', adjustable='box')
    plt.show()


#%%
# Fig. 2d (Fig. S5g, Ext. Fig. 1h, Ext. Fig. 1i)

type = 'opt' # 'opt' or 'elec'
human_data = 'E:\\ANNA_INTERN\\Human_Exp\\211202'

if type == 'opt':
  sel_ppl = list(range(300, 309)) + list(range(400, 408)) + [611] # 18 ppl
elif type == 'elec': 
  # sel_ppl = [499, 500, 502] + list(range(504, 509)) + list(range(602, 606)) + list(range(608, 612)) # 16 ppl (remove two outliers)
  sel_ppl = [499, 500, 502] + list(range(503, 509)) + list(range(602, 607)) + list(range(608, 612)) # 18 ppl

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


# machine data
test_type_list = ['opt']
model_type1_list = [''] #['PCA', 'PCA', '', '']
model_type2_list = ['CNN_SVC'] #['SVC', 'LR', 'CNN_LR', 'CNN_SVC']
seed_list = [22, 77, 2, 100, 81, 42] #, 7, 1, 55, 50] # different 1,000 training images (total of 10 permutations)
pix_order_list = ['16PIX', '24PIX', '32PIX', '64PIX', '128PIX']
gs_order_list = ['2GS', '4GS', '6GS', '8GS', '16GS']
class_list = [4] # [2, 4, 16]

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

                mac_df = pd.DataFrame()
                for seed in seed_list:
                    # seed = random.choice(seed_list) # For each set, we have random seed.
                    data_path = f'E:\\ANNA_INTERN\\Question Banks AI Hub_final\\{c}classes\\set{m}\\seed{seed}'
                    res_d = {'16PIX_2GS': [], '16PIX_4GS': [], '16PIX_6GS': [], '16PIX_8GS': [], '16PIX_16GS': [],
                             '24PIX_2GS': [], '24PIX_4GS': [], '24PIX_6GS': [], '24PIX_8GS': [], '24PIX_16GS': [],
                             '32PIX_2GS': [], '32PIX_4GS': [], '32PIX_6GS': [], '32PIX_8GS': [], '32PIX_16GS': [],
                             '64PIX_2GS': [], '64PIX_4GS': [], '64PIX_6GS': [], '64PIX_8GS': [], '64PIX_16GS': [],
                             '128PIX_2GS': [], '128PIX_4GS': [], '128PIX_6GS': [], '128PIX_8GS': [], '128PIX_16GS': [],}
                    try:
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
                    except:
                        print(c, seed, n, 'error')
                        pass
                    temp_df = pd.DataFrame.from_dict(res_d)
                    temp_df = temp_df.mean().to_frame().reset_index()
                    temp_df.columns = ['Resolution', 'Hit Rate']
                    seed_rows = [seed] * temp_df.shape[0]
                    temp_df['Seed'] = seed_rows
                    
                    mac_df = pd.concat([mac_df, temp_df], axis=0)
                #high_df = high_df.groupby('image').mean().reset_index()

answer_df = pd.read_csv(f'E:\\ANNA_INTERN\\Human_Exp\\211105_QAs_for_Set0_CNN_SVC_4classes_partial.csv')

n = 9
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
answer_df

# both
mer_df = pd.concat([human_df, answer_df], axis=0)
mer_df = mer_df.T
mer_df = mer_df.fillna(0)
orig_mer_df = mer_df

temp_mer_df0 = orig_mer_df.copy()

sel_ppl = []
for col in temp_mer_df0.columns:
    if col == 'Answer':
        break
    sel_ppl.append(col)

d = dict()
for per in sel_ppl:
    d[per] = []
pix_par_list = ['16PIX', '32PIX', '64PIX']
gs_par_list = ['2GS', '4GS', '8GS']

for pix in pix_par_list:
    temp_mer_df1 = temp_mer_df0[temp_mer_df0['PIX'] == pix]
    for gs in gs_par_list:
        temp_mer_df2 = temp_mer_df1[temp_mer_df1['GS'] == gs]
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

per_df = pd.DataFrame.from_dict(d)
hyperpar_name_list = ['16PIX_2GS', '16PIX_4GS', '16PIX_8GS', 
                      '32PIX_2GS', '32PIX_4GS', '32PIX_8GS',  
                      '64PIX_2GS', '64PIX_4GS', '64PIX_8GS']
per_df.index = hyperpar_name_list
new_hyperpar_name_list = ['16PIX_2GS', '16PIX_4GS', '16PIX_8GS', 
                          '32PIX_2GS', '64PIX_2GS', '32PIX_4GS', '32PIX_8GS',  
                          '64PIX_4GS', '64PIX_8GS']
per_df = per_df.reindex(new_hyperpar_name_list)

avg_list = []
for i in range(per_df.shape[0]):
    temp_l = []
    for per in sel_ppl:
        temp_l.append(per_df[per][i])
    a = np.mean(temp_l)
    avg_list.append(a)
per_df['avg_acc'] = avg_list

new_hyperpar_name_list = ['16PIX_2GS', '16PIX_4GS', '16PIX_8GS', 
                          '32PIX_2GS', '32PIX_4GS', '32PIX_8GS',  
                          '64PIX_2GS', '64PIX_4GS', '64PIX_8GS']


# ANN
ann_df = mac_df.copy()
ann_df = ann_df[ann_df['Resolution'].isin(new_hyperpar_name_list)]
ann_df['Resolution'] = pd.Categorical(ann_df['Resolution'], categories=new_hyperpar_name_list, ordered=True)

# Human
hum_df = per_df.stack().to_frame()
hum_df = hum_df.reset_index()
hum_df.columns = ['Resolution', 'Seed', 'Hit Rate']
hum_df = hum_df[hum_df['Resolution'].isin(new_hyperpar_name_list)]
hum_df['Resolution'] = pd.Categorical(hum_df['Resolution'], categories=new_hyperpar_name_list, ordered=True)
hum_df = hum_df[hum_df['Seed'].isin(sel_ppl)]


# Plot graphs
ann_hum_df = [ann_df, hum_df]
palette_list = [sns.color_palette('Blues', len(list(set(list(ann_df['Seed'].values))))), sns.color_palette('Reds', len(list(set(list(hum_df['Seed'].values)))))]
                
grouped_df1 = ann_df.groupby('Resolution').mean()
grouped_df2 = hum_df.groupby('Resolution').mean()
grouped_df = [grouped_df1, grouped_df2]
label_list = ['Avg of ANNs', 'Avg of humans']
color_list = ['steelblue', 'crimson']

for (i, df) in enumerate(grouped_df):
    sns.lineplot(x='Resolution', y='Hit Rate', hue='Seed', data=ann_hum_df[i], palette=palette_list[i], lw=1, alpha=0.5)
    globals()[f'plot{i}'] = sns.lineplot(data=grouped_df[i]['Hit Rate'], label=label_list[i], linestyle='-', lw=3, color=color_list[i])

plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.xticks(rotation=60)
plt.ylabel('Hit Rate')
plt.xlabel('Resolution')
plt.ylim([0, 1])
sns.despine()
plt.show()

for (i, hyperpar) in enumerate(new_hyperpar_name_list):
    ann = np.array(ann_df[ann_df['Resolution'] == hyperpar]['Hit Rate'].astype(float).values)
    hum = np.array(hum_df[hum_df['Resolution'] == hyperpar]['Hit Rate'].astype(float).values)
    
    statistic, pvalue = scipy.stats.mannwhitneyu(x=ann, y=hum, use_continuity=True, alternative='two-sided')
    pvalue = '{:.2e}'.format(pvalue)
    print(f'{hyperpar}: Mann-Whiteney-Wilcoxon test two-sided with Bonferroni correction, P-val={pvalue}, U_stat={statistic}', '\n')
