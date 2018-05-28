'''
Created on 5 May, 2018

@author: shobi
'''
# benchmark speeds on mnist for various tsne implementations: https://github.com/scikit-learn/scikit-learn/issues/10044
# from bhtsnevdm import run_bh_tsne #https://github.com/dominiek/p11ython-bhtsne
from MulticoreTSNE import MulticoreTSNE as multicore_tsne  # https://github.com/DmitryUlyanov/Multicore-TSNE
import bhtsne  # (pip install bhtsne)
from tsne import bh_sne  # https://github.com/danielfrg/tsne
import numpy as np
import time
from sklearn.manifold import TSNE
import scipy.io
import pandas as pd
import matplotlib.pyplot as plt
from mst_clustering import MSTClustering
# import seaborn as sns;
import time
# sns.set()
from matplotlib.widgets import Lasso
from matplotlib.collections import RegularPolyCollection
from matplotlib import colors as mcolors, path
from sklearn.cluster import DBSCAN
import graph_tool as gt
import graph_tool.topology as gtt
import graph_tool.draw as gtd
import networkx as nx
from convertnx2gt import *

version = 'Multicore tsne'
lr = 1000
# cancer = 'acc220'
cancer = 'k562_gated'
# cancer = 'fluor_nsclc'
fluor = 0
min_cluster_size = 20
print('min size', min_cluster_size)
sigma_factor = 2.5

# 0: no fluor
# 1: only fluor
# 2: all features (fluor + non-fluor)
perp = 30

new_file_name = 'n' + str(n_total) + "_pbmcsubtypes"

label_file_name = '/home/shobi/Thesis/BarnesHutMC_data/' + new_folder_name + '/BH_label_' + new_file_name + '.txt'
tag_file_name = '/home/shobi/Thesis/BarnesHutMC_data/' + new_folder_name + '/BH_tag_' + new_file_name + '.txt'
data_file_name = '/home/shobi/Thesis/BarnesHutMC_data/' + new_folder_name + '/BH_data_' + new_file_name + '.txt'

# KELVINS K562 AND ACC220 DONT HAVE columns for FLUOR DATA
featureName_k562_acc220 = ['File ID', 'Cell ID', 'Area', 'Volume', 'Circularity', 'Attenuation density',
                           'Amplitude var', 'Amplitude skewness', 'Amplitude kurtosis', 'Focus factor 1',
                           'Focus factor 2', 'Dry mass', 'Dry mass density', 'Dry mass var', 'Dry mass skewness',
                           'Peak phase', 'Phase var', 'Phase skewness', 'Phase kurtosis', 'DMD contrast 1',
                           'DMD contrast 2', 'DMD contrast 3', 'DMD contrast 4', 'Mean phase arrangement',
                           'Phase arrangement var', 'Phase arrangement skewness', 'Phase orientation var',
                           'Phase orientation kurtosis']
# DICKSON'S NSCLC HAVE FLUOR DATA. FOCUS FACTOR IS NOT THE FINAL FEATURE
featureName_fluor = ['File ID', 'Cell ID', 'Area', 'Volume', 'Circularity', 'Attenuation density', 'Amplitude var',
                     'Amplitude skewness', 'Amplitude kurtosis', 'Focus factor 1', 'Focus factor 2', 'Dry mass',
                     'Dry mass density', 'Dry mass var', 'Dry mass skewness', 'Peak phase', 'Phase var',
                     'Phase skewness', 'Phase kurtosis', 'DMD contrast 1', 'DMD contrast 2', 'DMD contrast 3',
                     'DMD contrast 4', 'Mean phase arrangement', 'Phase arrangement var', 'Phase arrangement skewness',
                     'Phase orientation var', 'Phase orientation kurtosis', 'Fluorescence (Peak)',
                     'Fluorescence (Area)', 'Fluorescence density', 'Fluorescence-Phase correlation']
# KELVINS PBMC AND THP1 HAVE EMPTY COLUMNS FOR FLUOR WHICH WE WILL DROP LATER. THE FOCUS FACTOR FEATURE IS THE FINAL FEATURE
featureName = ['File ID', 'Cell ID', 'Area', 'Volume', 'Circularity', 'Attenuation density', 'Amplitude var',
               'Amplitude skewness', 'Amplitude kurtosis', 'Dry mass', 'Dry mass density', 'Dry mass var',
               'Dry mass skewness', 'Peak phase', 'Phase var', 'Phase skewness', 'Phase kurtosis', 'DMD contrast 1',
               'DMD contrast 2', 'DMD contrast 3', 'DMD contrast 4', 'Mean phase arrangement', 'Phase arrangement var',
               'Phase arrangement skewness', 'Phase orientation var', 'Phase orientation kurtosis',
               'Fluorescence (Peak)', 'Fluorescence (Area)', 'Fluorescence density', 'Fluorescence-Phase correlation',
               'Focus factor 1', 'Focus factor 2']
# ALL FEATURES EXCLUDING FILE AND CELL ID:
feat_cols = ['Area', 'Volume', 'Circularity', 'Attenuation density', 'Amplitude var', 'Amplitude skewness',
             'Amplitude kurtosis', 'Dry mass', 'Dry mass density', 'Dry mass var', 'Dry mass skewness', 'Peak phase',
             'Phase var', 'Phase skewness', 'Phase kurtosis', 'DMD contrast 1', 'DMD contrast 2', 'DMD contrast 3',
             'DMD contrast 4', 'Mean phase arrangement', 'Phase arrangement var', 'Phase arrangement skewness',
             'Phase orientation var', 'Phase orientation kurtosis', 'Focus factor 1', 'Focus factor 2']
feat_cols_includefluor = ['Area', 'Volume', 'Circularity', 'Attenuation density', 'Amplitude var', 'Amplitude skewness',
                          'Amplitude kurtosis', 'Dry mass', 'Dry mass density', 'Dry mass var', 'Dry mass skewness',
                          'Peak phase', 'Phase var', 'Phase skewness', 'Phase kurtosis', 'DMD contrast 1',
                          'DMD contrast 2', 'DMD contrast 3', 'DMD contrast 4', 'Mean phase arrangement',
                          'Phase arrangement var', 'Phase arrangement skewness', 'Phase orientation var',
                          'Phase orientation kurtosis', 'Focus factor 1', 'Focus factor 2', 'Fluorescence (Peak)',
                          'Fluorescence (Area)', 'Fluorescence density', 'Fluorescence-Phase correlation']
feat_cols_fluor_only = ['Fluorescence (Peak)', 'Fluorescence (Area)', 'Fluorescence density',
                        'Fluorescence-Phase correlation']
feat_cols1 = ['Area', 'Volume', 'Circularity', 'Attenuation density', 'Amplitude var']
# THP1_Raw = scipy.io.loadmat('/home/shobi/Thesis/Data/ShobiGatedData/thp12017Nov22_gatedThp1.mat') #28 x 307,339
# PBMC_Raw = scipy.io.loadmat('/home/shobi/Thesis/Data/ShobiGatedData/pbmc2017Nov22_gatedPbmc.mat') #28 x 466,266

# MCF7_Raw = scipy.io.loadmat('/home/shobi/Thesis/Data/MCF7_clean_real.mat') #32 x 306,968
# K562_Raw = scipy.io.loadmat('/home/shobi/Thesis/Data/ShobiGatedData/k5622017Nov08_gatedK562.mat') #28 x 716,088
acc220_Raw = scipy.io.loadmat('/home/shobi/Thesis/Data/ShobiGatedData/acc2202017Nov22_gatedAcc220.mat')  # 28 x 416,421
# print('loaded acc220')
# pbmc_fluor_raw = scipy.io.loadmat('/home/shobi/Thesis/Data/pbmc_fluor_clean_real.mat') #42,308 x 32
# nsclc_fluor_raw = scipy.io.loadmat('/home/shobi/Thesis/Data/nsclc_fluor_clean_real.mat') #1,031 x 32
pbmc_monocyte_Raw = scipy.io.loadmat(
    '/home/shobi/Thesis/Data/ShobiGatedData/monocyte2018Apr30_gatedMonocyte.mat')  # 32*488,831
pbmct_Raw = scipy.io.loadmat('/home/shobi/Thesis/Data/ShobiGatedData/pbmct2018Apr30_gatedPbmct.mat')  # 32*4,474
pbmcb_Raw = scipy.io.loadmat('/home/shobi/Thesis/Data/ShobiGatedData/monocyte2018Apr30_gatedPbmcb.mat')  # 32*890
pbmcnk_Raw = scipy.io.loadmat('/home/shobi/Thesis/Data/ShobiGatedData/nk2018Apr30_gatedNk.mat')  # 32*7,475

print('loaded pbmc subtypes')

print('getting monocyte')
monocyte_struct = pbmc_monocyte_Raw['monocyte2017Apr30_gatedMonocyte']
df_monocyte = pd.DataFrame(monocyte_struct[0, 0]['cellparam'].transpose().real)
monocyte_features = monocyte_struct[0, 0]['cellparam_label'][0].tolist()
monocyte_fileidx = pd.DataFrame(monocyte_struct[0, 0]['gated_idx'].transpose())
flist = []
for element in monocyte_features:
    flist.append(element[0])
df_monocyte.columns = flist
monocyte_fileidx.columns = ['filename', 'index']
print('shape of fileidx', monocyte_fileidx.shape)
df_monocyte['cell_tag'] = 'monocyte2018Apr30_' + monocyte_fileidx["filename"].map(int).map(str) + '_' + \
                          monocyte_fileidx["index"].map(int).map(str)
df_monocyte['label'] = 'Monocyte'
df_monocyte['class'] = 0
print(df_monocyte.head(5))
df_monocyte = df_monocyte.sample(frac=1).reset_index(drop=True)
print(df_monocyte.shape)

print('getting T-cell')
pbmct_struct = pbmct_Raw['pbmct2018Apr30_gatedPbmct']
df_pbmct = pd.DataFrame(pbmct_struct[0, 0]['cellparam'].transpose().real)
pbmct_features = pbmct_struct[0, 0]['cellparam_label'][0].tolist()
pbmct_fileidx = pd.DataFrame(pbmct_struct[0, 0]['gated_idx'].transpose())
flist = []
for element in pbmct_features:
    flist.append(element[0])
df_pbmct.columns = flist
pbmct_fileidx.columns = ['filename', 'index']
print('shape of fileidx', pbmct_fileidx.shape)
df_pbmct['cell_tag'] = 'pbmct2018Apr30_' + pbmct_fileidx["filename"].map(int).map(str) + '_' + pbmct_fileidx[
    "index"].map(int).map(str)
df_pbmct['label'] = 'T-cell'
df_pbmct['class'] = 1
print(df_pbmct.head(5))
df_pbmct = df_pbmct.sample(frac=1).reset_index(drop=True)
print(df_pbmct.shape)

print('getting B-cell')
pbmcb_struct = pbmcb_Raw['pbmcb2018Apr30_gatedPbmcb']
df_pbmcb = pd.DataFrame(pbmcb_struct[0, 0]['cellparam'].transpose().real)
pbmcb_features = pbmcb_struct[0, 0]['cellparam_label'][0].tolist()
pbmcb_fileidx = pd.DataFrame(pbmcb_struct[0, 0]['gated_idx'].transpose())
flist = []
for element in pbmcb_features:
    flist.append(element[0])
df_pbmcb.columns = flist
pbmcb_fileidx.columns = ['filename', 'index']
print('shape of fileidx', pbmcb_fileidx.shape)
df_pbmcb['cell_tag'] = 'pbmct2018Apr30_' + pbmcb_fileidx["filename"].map(int).map(str) + '_' + pbmcb_fileidx[
    "index"].map(int).map(str)
df_pbmcb['label'] = 'T-cell'
df_pbmcb['class'] = 2
print(df_pbmcb.head(5))
df_pbmcb = df_pbmcb.sample(frac=1).reset_index(drop=True)
print(df_pbmcb.shape)

print('getting NK-cell')
pbmcnk_struct = pbmcb_Raw['nk2018Apr30_gatedNk']
df_pbmcnk = pd.DataFrame(pbmcnk_struct[0, 0]['cellparam'].transpose().real)
pbmcnk_features = pbmcnk_struct[0, 0]['cellparam_label'][0].tolist()
pbmcnk_fileidx = pd.DataFrame(pbmcnk_struct[0, 0]['gated_idx'].transpose())
flist = []
for element in pbmcnk_features:
    flist.append(element[0])
df_pbmcnk.columns = flist
pbmcnk_fileidx.columns = ['filename', 'index']
print('shape of fileidx', pbmcnk_fileidx.shape)
df_pbmcnk['cell_tag'] = 'pbmcnk2018Apr30_' + pbmcnk_fileidx["filename"].map(int).map(str) + '_' + pbmcnk_fileidx[
    "index"].map(int).map(str)
df_pbmcnk['label'] = 'T-cell'
df_pbmcnk['class'] = 3
print(df_pbmcnk.head(5))
df_pbmcnk = df_pbmcnk.sample(frac=1).reset_index(drop=True)
print(df_pbmcnk.shape)

frames = [df_monocyte, df_pbmct, df_pbmcb, df_pbmcnk]
df_all = pd.concat(frames, ignore_index=True)

# EXCLUDE FLUOR FEATURES
if fluor == 0:
    df_all[feat_cols] = (df_all[feat_cols] - df_all[feat_cols].mean()) / df_all[feat_cols].std()
    X_txt = df_all[feat_cols].values
    print('size of data matrix:', X_txt.shape)
# ONLY USE FLUOR FEATURES
if fluor == 1:
    df_all[feat_cols_fluor_only] = (df_all[feat_cols_fluor_only] - df_all[feat_cols_fluor_only].mean()) / df_all[
        feat_cols_fluor_only].std()
    X_txt = df_all[feat_cols_fluor_only].values
    print('size of data matrix:', X_txt.shape)
if fluor == 2:  # all features including fluor
    df_all[feat_cols_includefluor] = (df_all[feat_cols_includefluor] - df_all[feat_cols_includefluor].mean()) / df_all[
        feat_cols_includefluor].std()
    X_txt = df_all[feat_cols_includefluor].values

label_txt = df_all['class'].values
tag_txt = df_all['cell_tag'].values
print(X_txt.size, label_txt.size)

time_start = time.time()
print('starting tsne', time.ctime())
# SCIKIT very slow
# X_embedded = TSNE(n_components=2).fit_transform(X_txt)
# github alexisbcook version:
# X_embedded = tsne(X_txt)
# X_embedded = bhtsnevdm.run_bh_tsne(X_txt, initial_dims=X_txt.shape[1])
tsne = multicore_tsne(n_jobs=8, perplexity=30, verbose=1, n_iter=1000, learning_rate=lr)
params = 'n_jobs=8, perplexity = 30,verbose=1,n_iter=1000,learning_rate =' + str(lr)
print(params)
X_embedded = tsne.fit_transform(X_txt)
print(X_embedded.shape)
print(data_file_name, ' BH done! Time elapsed: {} seconds'.format(time.time() - time_start))

true_label = np.asarray(label_txt)
true_label = np.reshape(true_label, (true_label.shape[0], 1))
print('true label shape:', true_label.shape)
true_label = true_label.astype(int)
tag = np.asarray(tag_txt)
tag = np.reshape(tag, (tag.shape[0], 1))
# true_labels = np.vstack((np.zeros((X0.shape[0],1)),np.ones((X1.shape[0],1))))
# X01 = np.vstack((X0,X1))
#np.savetxt(data_file_name, X_txt, comments='', header=str(n_total) + ' ' + str(int(X_txt.shape[1])), fmt="%f",
           delimiter=" ")
#np.savetxt(embedding_file_name, X_embedded, comments='', header=str(n_total) + ' ' + str(int(X_embedded.shape[1])),
           fmt="%f", delimiter=" ")
#np.savetxt(label_file_name, label_txt, fmt="%i", delimiter="")
#np.savetxt(tag_file_name, tag_txt, fmt="%s", delimiter="")





def func_mode(ll):  # return MODE of list
    # If multiple items are maximal, the function returns the first one encountered.
    return max(set(ll), key=ll.count)


def func_counter(ll):
    c_0 = ll.count(0)
    c_1 = ll.count(1)
    if c_0 > c_1: return 0
    if c_0 < c_1: return 1
    if c_0 == c_1: return 999


def plot_mst(model, true_labels, num_data, data_lasso, color_feature, X_data_array, feature_name):
    # http://nbviewer.jupyter.org/github/jakevdp/mst_clustering/blob/master/MSTClustering.ipynb

    X_dict = {}
    X_dict_dbscan = {}
    Index_dict = {}
    Index_dict_dbscan = {}
    print('in plot')
    X_plot = model.X_fit_
    mst_labels = list(model.labels_)

    m = 999
    for k in range(len(mst_labels)):
        x = X_plot[k, 0]
        y = X_plot[k, 1]
        X_dict.setdefault(mst_labels[k], []).append((x, y))

        Index_dict.setdefault(mst_labels[k], []).append(true_labels[k])
        # Index_dict_dbscan.setdefault(dbscan_labels[k], []).append(true_labels[k])
    # X_dict_dbscan.setdefault(dbscan_labels[k], []).append((x, y))

    sorted_keys = list(sorted(X_dict.keys()))
    print('number of distinct groups:', len(sorted_keys))
    # sorted_keys_dbscan =list(sorted(X_dict_dbscan.keys()))
    print(sorted_keys, ' sorted keys')
    error_count = []
    pbmc_labels = []
    thp1_labels = []
    unknown_labels = []
    fp = 0
    fn = 0
    tp = 0
    tn = 0

    for kk in sorted_keys:
        vals = [t for t in Index_dict[kk]]
        # print('kk ', kk, 'has length ', len(vals))
        majority_val = func_mode(vals)
        len_unknown = 0

        if (kk == -1):
            unknown_labels.append(kk)
            len_unknown = len(vals)
        if (majority_val == 1) and (kk != -1):
            thp1_labels.append(kk)
            # print(majority_val, 'is majority val')
            fp = fp + len([e for e in vals if e != majority_val])
            tp = tp + len([e for e in vals if e == majority_val])
            error_count.append(len([e for e in vals if e != majority_val]))
        if (majority_val == 0) and (kk != -1):
            pbmc_labels.append(kk)
            # print(majority_val, 'is majority val')
            fn = fn + len([e for e in vals if e != majority_val])
            tn = tn + len([e for e in vals if e == majority_val])
            error_count.append(len([e for e in vals if e != majority_val]))
        if majority_val == 999:
            unknown_labels.append(kk)
            # print(kk,' has no majority')
    # print('thp1_labels:', thp1_labels)
    # print('pbmc_labels:', pbmc_labels)
    # print('error count for each group is: ', error_count)
    # print('len unknown', len_unknown)
    error_rate = sum(error_count) / n_total
    print((sum(error_count) + len_unknown) * 100 / n_total, '%')
    print('fp is :', fp * 100 / (n_pbmc), '%', 'and fn is: ', fn * 100 / (n_cancer), '%')
    print('true-ratio is:', ratio, ':1')
    comp_n_cancer = n_cancer + fp - fn
    comp_n_pbmc = n_pbmc - fp + fn
    if comp_n_cancer != 0:
        computed_ratio = comp_n_pbmc / comp_n_cancer
        print('computed-ratio is:', computed_ratio, ':1')
    print('fp:', fp)
    print('fn:', fn)
    print('tp', tp)
    print('tn', tn)
    print('sum of tp, tn,fn,fp', tn + tp + fn + fp)
    print('fp,fn,tp,tn:', fp, fn, tp, tn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = precision * recall * 2 / (precision + recall)
    print('f1-score', f1_score)
    '''
    error_count_dbscan = []
    pbmc_labels_dbscan =[]
    thp1_labels_dbscan =[]
    unknown_labels_dbscan = []
    fp_dbscan=0
    fn_dbscan=0
    for kk in sorted_keys_dbscan:
        vals = [t for t in Index_dict_dbscan[kk]]
        print('kk ', kk, 'has length ', len(vals))
        majority_val = func_mode(vals)
        len_unknown = 0

        if (kk == -1):
            unknown_labels_dbscan.append(kk)
            len_unknown = len(vals)
        if (majority_val ==1) and (kk != -1): 
            thp1_labels_dbscan.append(kk)
            print(majority_val, 'is majority val')
            fp_dbscan = fp_dbscan+ len([e for e in vals if e!=majority_val])
            error_count_dbscan.append(len([e for e in vals if e!=majority_val]))
        if (majority_val ==0) and (kk != -1): 
            pbmc_labels_dbscan.append(kk)
            print(majority_val, 'is majority val')
            fn_dbscan = fn_dbscan+ len([e for e in vals if e!=majority_val])
            error_count_dbscan.append(len([e for e in vals if e!=majority_val]))
        if majority_val ==999 : 
            unknown_labels_dbscan.append(kk)
            print(kk,' has no majority') 
    print('thp1_labels:', thp1_labels_dbscan)
    print('pbmc_labels:', pbmc_labels_dbscan)
    print('error count for each group by dbscan is: ', error_count_dbscan)

    error_rate_dbscan = sum(error_count_dbscan)/N
    print('fp is :', fp_dbscan*100/(n_pbmc), ' and fn is: ', fn_dbscan*100/(n_cancer))
    print(sum(error_count_dbscan)*100/N, '%')
    '''
    print(new_file_name_title)

    fig, ax = plt.subplots(1, 3, figsize=(36, 12), sharex=True, sharey=True)
    segments = model.get_graph_segments(full_graph=True)

    # ax[0].scatter(X[:, 0], X[:, 1], c=true_labels, cmap='nipy_spectral_r', zorder=2, alpha=0.5, s=4) *USE THIS
    idx = np.where(color_feature < 5 * np.std(color_feature))
    print(idx[0].shape)
    c_keep = color_feature[idx[0]]
    print('ckeep shape', c_keep.shape)
    X_keep = X_data_array[idx[0], :]
    print('xkeep shape', X_keep.shape)
    print(c_keep.min(), c_keep.max())
    # s= ax[2].scatter(X_keep[:,0], X_keep[:,1], c =c_keep[:,0], s=4, cmap = 'Reds')
    # cb = plt.colorbar(s)

    lman = LassoManager(ax[0], data_lasso)
    ax[0].text(0.95, 0.01, "blue: pbmc", transform=ax[1].transAxes, verticalalignment='bottom',
               horizontalalignment='right', color='green', fontsize=10)
    # color = model.labels_.reshape(1,-1)
    # colors = plt.cm.jet(np.linspace(0, 1, len(X_dict)))

    colors_pbmc = plt.cm.winter(np.linspace(0, 1, len(pbmc_labels)))
    colors_thp1 = plt.cm.autumn(np.linspace(0, 1, len(thp1_labels)))
    # colors_pbmc_dbscan = plt.cm.winter(np.linspace(0, 1, len(pbmc_labels_dbscan)))
    # colors_thp1_dbscan = plt.cm.autumn(np.linspace(0, 1, len(thp1_labels_dbscan)))
    for color_p, ll_p in zip(colors_pbmc, pbmc_labels):
        x = [t[0] for t in X_dict[ll_p]]
        population = len(x)
        y = [t[1] for t in X_dict[ll_p]]
        ax[1].scatter(x, y, color=color_p, s=2, alpha=1, label='pbmc ' + str(ll_p) + ' Cellcount = ' + str(len(x)))
        ax[1].annotate(str(ll_p), xytext=(np.mean(x), np.mean(y)), xy=(np.mean(x), np.mean(y)), color='black',
                       weight='semibold')
        # ax[1].scatter(np.mean(x), np.mean(y),  color = color_p, s=population, alpha=1)
        ax[2].annotate(str(ll_p), xytext=(np.mean(x), np.mean(y)), xy=(np.mean(x), np.mean(y)), color='black',
                       weight='semibold')
        ax[2].scatter(np.mean(x), np.mean(y), color=color_p, s=np.log(population), alpha=1, zorder=2)
    for color_t, ll_t in zip(colors_thp1, thp1_labels):
        x = [t[0] for t in X_dict[ll_t]]
        population = len(x)
        y = [t[1] for t in X_dict[ll_t]]
        ax[1].scatter(x, y, color=color_t, s=4, alpha=1, label=cancer + ' ' + str(ll_t) + ' Cellcount = ' + str(len(x)))
        ax[1].annotate(str(ll_t), xytext=(np.mean(x), np.mean(y)), xy=(np.mean(x), np.mean(y)), color='black',
                       weight='semibold')
        # ax[1].scatter(np.mean(x), np.mean(y),  color = color_t, s=population, alpha=1)
        ax[2].annotate(str(ll_t), xytext=(np.mean(x), np.mean(y)), xy=(np.mean(x), np.mean(y)), color='black',
                       weight='semibold')
        ax[2].scatter(np.mean(x), np.mean(y), color=color_t, s=np.log(population), alpha=1, zorder=2)
        # xf = [t[0] for t in X_dict_force[ll]]
        # yf = [t[1] for t in X_dict_force[ll]]
        # ax[2].scatter(xf, yf,  color = color, s=4, alpha=1, label = ll)
        # ax[2].annotate(str(ll), xytext=(np.mean(xf), np.mean(yf)), xy = (np.mean(xf), np.mean(yf)),  color= 'black', weight = 'semibold' )
    n_clusters = len(pbmc_labels) + len(thp1_labels)
    # for l in range(n_clusters):
    # ax[2].contour(label == l)
    ax[2].plot(segments[0], segments[1], '-k', zorder=1, linewidth=1)

    print(n_cancer, n_pbmc)
    print(error_count, fp, fn)
    ax[1].text(0.95, 0.01, "error: " + " {:.2f}".format(error_rate * 100) + '%' + " FP: " + " {:.2f}".format(
        fp * 100 / n_pbmc) + "%. FN of " + "{:.2f}".format(fn * 100 / (n_cancer)) + '%', transform=ax[1].transAxes,
               verticalalignment='bottom', horizontalalignment='right', color='green', fontsize=10)
    for uu in unknown_labels:
        x = [t[0] for t in X_dict[uu]]
        y = [t[1] for t in X_dict[uu]]
        ax[1].scatter(x, y, color='gray', s=4, alpha=1, label=uu)

    '''
    for color_p, ll_p in zip(colors_pbmc_dbscan, pbmc_labels_dbscan):
            x = [t[0] for t in X_dict_dbscan[ll_p]]
            y = [t[1] for t in X_dict_dbscan[ll_p]]
            ax[2].scatter(x, y,  color = color_p, s=2, alpha=1, label = 'pbmc '+str(ll_p) +' Cellcount = '+str(len(x)) )
            ax[2].annotate(str(ll_p), xytext=(np.mean(x), np.mean(y)), xy = (np.mean(x), np.mean(y)), color= 'black', weight = 'semibold')
            #xf = [t[0] for t in X_dict_force[ll]]
            #yf = [t[1] for t in X_dict_force[ll]]
            #ax[2].scatter(xf, yf,  color = color, s=4, alpha=1, label = ll)
            #ax[2].annotate(str(ll), xytext=(np.mean(xf), np.mean(yf)), xy = (np.mean(xf), np.mean(yf)),  color= 'black', weight = 'semibold' )
    for color_t, ll_t in zip(colors_thp1_dbscan, thp1_labels_dbscan):
            x = [t[0] for t in X_dict_dbscan[ll_t]]
            y = [t[1] for t in X_dict_dbscan[ll_t]]
            ax[2].scatter(x, y,  color = color_t, s=4, alpha=1, label = cancer+' '+ str(ll_t)+' Cellcount = '+str(len(x)) )
            ax[2].annotate(str(ll_t), xytext=(np.mean(x), np.mean(y)), xy = (np.mean(x), np.mean(y)), color= 'black', weight = 'semibold')
            #xf = [t[0] for t in X_dict_force[ll]]
            #yf = [t[1] for t in X_dict_force[ll]]
            #ax[2].scatter(xf, yf,  color = color, s=4, alpha=1, label = ll)
            #ax[2].annotate(str(ll), xytext=(np.mean(xf), np.mean(yf)), xy = (np.mean(xf), np.mean(yf)),  color= 'black', weight = 'semibold' )
    print(N, n_cancer,n_pbmc)
    print(sum(error_count_dbscan), fp_dbscan, fn_dbscan)
 '''

    ##labels = list(model.labels_.astype(str))#.reshape(1,-1)

    ##print(labels)

    ##segments = model.get_graph_segments(full_graph=False)
    ##ax[1].plot(segments[0], segments[1], '-k',linewidth=0.2 ,zorder=1, lw=1)
    print('plotting MST subplot')

    # ax[1].scatter(X[:, 0], X[:, 1], c=model.labels_, cmap='nipy_spectral_r',label = labels ,zorder=2, alpha=0.5, s=4)

    # ax[2].legend()

    ax[1].axis('tight')
    title_str0 = version + ' with perplexity : ' + str(perp) + ' on ' + new_file_name_title
    # title_str1 = 'MST: cutoff' + str(cutoff_scale) +' min_cluster:' +str(min_cluster_size)
    title_str1 = 'MST: mean + ' + str(sigma_factor) + '-sigma cutoff and min cluster size of: ' + str(
        min_cluster_size) + '\n' + "error: " + " {:.2f}".format(error_rate * 100) + '%' + " FP: " + " {:.2f}".format(
        fp * 100 / n_pbmc) + "%. FN of " + "{:.2f}".format(
        fn * 100 / (n_cancer)) + '%' + '\n' + 'computed ratio:' + "{:.4f}".format(
        computed_ratio) + ' f1-score:' + "{:.4f}".format(f1_score)
    title_str2 = feature_name
    # ax[2].set_title(graph_title_force, size=16)
    ax[1].set_title(title_str1, size=10)
    ax[0].set_title(title_str0, size=10)
    # ax[2].set_title(title_str2, size=12)
    box = ax[1].get_position()
    ax[1].set_position([box.x0, box.y0, box.width * 0.9, box.height])

    # Put a legend to the right of the current axis
    ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=6)
    plt.show()


time_start = time.time()
# n_neighbors = 100
# default n_neighbors is 20 but changing to 50 or 100 does not change the accuracy
model = MSTClustering(cutoff_scale=0.3, approximate=True, min_cluster_size=min_cluster_size, sigma_factor=sigma_factor)
model.fit_predict(X_embedded)
# model_dbscan = DBSCAN(eps = 0.1, min_samples = 5)
# dbscan_labels = model_dbscan.fit_predict(X_data)
print('MST done! Time elapsed: {} seconds'.format(time.time() - time_start))

data_lasso = [Datum(*xy, z, t) for xy, z, t in zip(X_embedded, true_label, tag)]
feature_name = 'Attenuation density'
color_feature = df_all[feature_name].values.reshape((-1, 1))
print(color_feature.max(), color_feature.std())
plot_mst(model, true_label[:, 0], len(true_label), data_lasso, color_feature, X_embedded, feature_name)