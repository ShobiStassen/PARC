'''
plotting feature histograms
'''

import LungCancer_function_minClusters as LCC
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss

df_h2170 = LCC.make_subtype_df('h21702018Jan23_gatedH2170',0,n_eachsubtype=None, randomseedval=1, HTC_filename ="\\\\Desktop-u14r2et\\G\\2018Jan23\\" ) #h2170_Raw, 28 x 302,635
df_h1975 = LCC.make_subtype_df('h19752018Jan23_gatedH1975',1,n_eachsubtype=None, randomseedval=1, HTC_filename= 'F:\\') #60447
df_h526 = LCC.make_subtype_df('H5262018May24_gatedH526',2,n_eachsubtype=None, randomseedval=1, HTC_filename = '\\\\Desktop-p9kngca\htc-4(b)\\2018May24_cancer\\')#375889
df_h520 = LCC.make_subtype_df('h5202018Jan03_gatedH520',3,n_eachsubtype=None, randomseedval=1, HTC_filename= '\\\\DESKTOP-H5E5CH1\\DVDproc - 2 (G)\\20180103 H520\\')#451208
df_h358 = LCC.make_subtype_df('h3582018Jan03_gatedH358',4,n_eachsubtype=None, randomseedval=1, HTC_filename='\\\\DESKTOP-H5E5CH1\\DVDproc - 2\\2018Jan3_h358_520_526\\')#170198
df_h69 = LCC.make_subtype_df('h692018Jan23_gatedH69',5,n_eachsubtype=None, randomseedval=1,HTC_filename= 'F:\\') #130075
df_hcc827 = LCC.make_subtype_df('hcc8272018Jun05_gatedHcc827',6,n_eachsubtype=None, randomseedval=1, HTC_filename='\\\\Desktop-p9kngca\\htc-4(b)\\2018Jun05_lungcancer\\')

column_biofeatures = ['Area', 'Volume', 'Circularity', 'Attenuation density', 'Amplitude var', 'Amplitude skewness', 'Amplitude kurtosis',
                      'Focus factor 1', 'Focus factor 2', 'Dry mass', 'Dry mass density', 'Dry mass var', 'Dry mass skewness', 'Peak phase',
                      'Phase var', 'Phase skewness', 'Phase kurtosis', 'DMD contrast 1', 'DMD contrast 2', 'DMD contrast 3', 'DMD contrast 4',
                      'Mean phase arrangement', 'Phase arrangement var', 'Phase arrangement skewness', 'Phase orientation var', 'Phase orientation kurtosis']
print('num bio features',len(column_biofeatures))
#data = df_h2170.columns.values
#fig = plt.figure()
#ax = fig.add_subplot(111)
#data =df_h2170['Area'].values
#ax.hist(data, weights=np.zeros_like(data) + 1. / data.size)
frames = [df_h2170, df_h1975, df_h526, df_h520, df_h358, df_h69, df_hcc827]
frames_str = ['df_h2170', 'df_h1975', 'df_h526', 'df_h520', 'df_h358', 'df_h69', 'df_hcc827']
'''
for frame_i, frame_i_str in zip(frames,frames_str):
    for feature in column_biofeatures:
        data = frame_i[feature].values
        skew = ss.skew(data)
        if skew >=0.8:
            print(str(frame_i_str), feature,'has skew ', skew)
'''

fig, axes = plt.subplots(nrows=3, ncols=3)

for ax, i_feature in zip(axes.flat[0:], column_biofeatures[0:9]):
    ax.set_title(i_feature,size=8)
    data = df_h2170[i_feature].values
    mean = np.mean(data)
    std = np.std(data)
    ax.hist(data,bins=20, weights=np.zeros_like(data) + 1. / data.size)
    ax.tick_params(labelsize=6)
    if i_feature!= 'Dry mass var' and min(data)>=0:
        ax.set_xlim(max(0, mean-5*std),min(max(data),mean+5*std))
fig.tight_layout()
fig.suptitle('H2170 set1 \n')

fig.tight_layout()
fig.suptitle('H2170 set1')
fig1Norm, axes1Norm = plt.subplots(nrows=3, ncols=3)
for ax, i_feature in zip(axes1Norm.flat[0:], column_biofeatures[0:9]):
    ax.set_title(i_feature,size=8)
    data = df_h2170[i_feature].values

    skew = ss.skew(data)
    print('skew of ', i_feature, 'is ', skew)
    #if abs(skew)>0.8:
    if min(data)<=0: data= data + abs(min(data))+1
    data = np.log(data)
    #data = ss.boxcox(data, lmbda = 0)
    print('doing boxcox')
    mean = np.mean(data)
    std = np.std(data)
    normdata =data#(data-mean)/std
    ax.hist(normdata,bins=20, weights=np.zeros_like(data) + 1. / data.size)
    ax.tick_params(labelsize=6)

fig1Norm.tight_layout()
fig1Norm.suptitle('H2170 set1Norm \n')


fig1, axes1 = plt.subplots(nrows=3, ncols=3)

for ax, i_feature in zip(axes1.flat[0:], column_biofeatures[9:18]):
    ax.set_title(i_feature,size=8)
    data = df_h2170[i_feature].values
    mean = np.mean(data)
    std = np.std(data)
    ax.hist(data,bins=20, weights=np.zeros_like(data) + 1. / data.size)
    ax.tick_params(labelsize=6)
    if min(data)>=0 and i_feature!= 'Dry mass var':
        ax.set_xlim(max(0, mean-5*std),min(max(data),mean+5*std))
fig1.tight_layout()
fig1.suptitle('H2170 set2')

fig2, axes2 = plt.subplots(nrows=3, ncols=3)

for ax, i_feature in zip(axes2.flat[0:], column_biofeatures[18:]):
    ax.set_title(i_feature,size=8)
    data = df_h2170[i_feature].values
    mean = np.mean(data)
    std = np.std(data)
    ax.hist(data,bins=20,weights=np.zeros_like(data) + 1. / data.size)
    ax.tick_params(labelsize=6)
    if min(data)>=0:
        ax.set_xlim(max(0, mean-5*std),min(max(data),mean+5*std))
fig2.tight_layout()
fig2.suptitle('H2170 set3')


plt.show()
