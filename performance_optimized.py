'''
Created on 5 May, 2018
   index cell id filename
0      3    0000    venus
1      2   98987     mars
2      3    0000    venus
3      2   98987     mars
4      1    1324     satu
5      0    5656    yaigh

@author: shobi
'''

import os
import sys
import LargeVis
import numpy as np
import time
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from MulticoreTSNE import MulticoreTSNE as multicore_tsne
import scipy.io
import pandas as pd
import matplotlib.pyplot as plt
# from mst_clustering import MSTClustering
from MST_clustering_mergetooclose import MSTClustering
import time
from sklearn import metrics
from sklearn.neighbors import KernelDensity
from scipy import stats
from pandas import ExcelWriter


# 0: no fluor
# 1: only fluor
# 2: all features (fluor + non-fluor)

def get_data(cancer_type, benign_type, n_cancer, ratio, fluor, dataset_number, new_folder_name, method):
    n_pbmc = int(n_cancer * ratio)
    n_total = int(n_pbmc + n_cancer)
    new_file_name = new_file_name_title = 'N' + str(n_total) + '_r{:.2f}'.format(ratio) + cancer_type + '_pbmc_gated_d' + str(
        dataset_number)
    if method == 'bh':
        label_file_name = '/home/shobi/Thesis/BarnesHutMC_data/' + new_folder_name + '/BH_label_' + new_file_name + '.txt'
        tag_file_name = '/home/shobi/Thesis/BarnesHutMC_data/' + new_folder_name + '/BH_tag_' + new_file_name + '.txt'
        data_file_name = '/home/shobi/Thesis/BarnesHutMC_data/' + new_folder_name + '/BH_data_' + new_file_name + '.txt'
    if method == 'lv':
        label_file_name = '/home/shobi/Thesis/LV_data/' + new_folder_name + '/LV_label_' + new_file_name + '.txt'
        tag_file_name = '/home/shobi/Thesis/LV_data/' + new_folder_name + '/LV_tag_' + new_file_name + '.txt'
        data_file_name = '/home/shobi/Thesis/LV_data/' + new_folder_name + '/LV_data_' + new_file_name + '.txt'

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
                         'DMD contrast 4', 'Mean phase arrangement', 'Phase arrangement var',
                         'Phase arrangement skewness', 'Phase orientation var', 'Phase orientation kurtosis',
                         'Fluorescence (Peak)', 'Fluorescence (Area)', 'Fluorescence density',
                         'Fluorescence-Phase correlation']
    # KELVINS PBMC AND THP1 HAVE EMPTY COLUMNS FOR FLUOR WHICH WE WILL DROP LATER. THE FOCUS FACTOR FEATURE IS THE FINAL FEATURE
    featureName = ['File ID', 'Cell ID', 'Area', 'Volume', 'Circularity', 'Attenuation density', 'Amplitude var',
                   'Amplitude skewness', 'Amplitude kurtosis', 'Dry mass', 'Dry mass density', 'Dry mass var',
                   'Dry mass skewness', 'Peak phase', 'Phase var', 'Phase skewness', 'Phase kurtosis', 'DMD contrast 1',
                   'DMD contrast 2', 'DMD contrast 3', 'DMD contrast 4', 'Mean phase arrangement',
                   'Phase arrangement var', 'Phase arrangement skewness', 'Phase orientation var',
                   'Phase orientation kurtosis', 'Fluorescence (Peak)', 'Fluorescence (Area)', 'Fluorescence density',
                   'Fluorescence-Phase correlation', 'Focus factor 1', 'Focus factor 2']
    # ALL FEATURES EXCLUDING FILE AND CELL ID:
    feat_cols = ['Area', 'Volume', 'Circularity', 'Attenuation density', 'Amplitude var', 'Amplitude skewness',
                 'Amplitude kurtosis', 'Dry mass', 'Dry mass density', 'Dry mass var', 'Dry mass skewness',
                 'Peak phase', 'Phase var', 'Phase skewness', 'Phase kurtosis', 'DMD contrast 1', 'DMD contrast 2',
                 'DMD contrast 3', 'DMD contrast 4', 'Mean phase arrangement', 'Phase arrangement var',
                 'Phase arrangement skewness', 'Phase orientation var', 'Phase orientation kurtosis', 'Focus factor 1',
                 'Focus factor 2']
    feat_cols_includefluor = ['Area', 'Volume', 'Circularity', 'Attenuation density', 'Amplitude var',
                              'Amplitude skewness', 'Amplitude kurtosis', 'Dry mass', 'Dry mass density',
                              'Dry mass var', 'Dry mass skewness', 'Peak phase', 'Phase var', 'Phase skewness',
                              'Phase kurtosis', 'DMD contrast 1', 'DMD contrast 2', 'DMD contrast 3', 'DMD contrast 4',
                              'Mean phase arrangement', 'Phase arrangement var', 'Phase arrangement skewness',
                              'Phase orientation var', 'Phase orientation kurtosis', 'Focus factor 1', 'Focus factor 2',
                              'Fluorescence (Peak)', 'Fluorescence (Area)', 'Fluorescence density',
                              'Fluorescence-Phase correlation']
    feat_cols_fluor_only = ['Fluorescence (Peak)', 'Fluorescence (Area)', 'Fluorescence density',
                            'Fluorescence-Phase correlation']
    feat_cols1 = ['Area', 'Volume', 'Circularity', 'Attenuation density', 'Amplitude var']

    print('loaded pbmc')
    # MCF7_Raw = scipy.io.loadmat('/home/shobi/Thesis/Data/MCF7_clean_real.mat') #32 x 306,968

    if benign_type == 'pbmc':
        print('constructing dataframe for ', benign_type)
        PBMC_Raw = scipy.io.loadmat(
            '/home/shobi/Thesis/Data/ShobiGatedData/pbmc2017Nov22_gatedPbmc.mat')  # 28 x 466,266
        pbmc_struct = PBMC_Raw['pbmc2017Nov22_gatedPbmc']
        df_pbmc = pd.DataFrame(pbmc_struct[0, 0]['cellparam'].transpose().real)
        pbmc_fileidx = pbmc_struct[0, 0]['gated_idx'][0].tolist()
        pbmc_features = pbmc_struct[0, 0]['cellparam_label'][0].tolist()
        flist = []
        idxlist = []
        for element in pbmc_features:
            flist.append(element[0])
        df_pbmc.columns = flist
        pbmc_fileidx = pd.DataFrame(pbmc_struct[0, 0]['gated_idx'].transpose())
        pbmc_fileidx.columns = ['filename', 'matlabindex']
        print('shape of fileidx', pbmc_fileidx.shape)
        df_pbmc['cell_filename'] = 'pbmc2017Nov22_' + pbmc_fileidx["filename"].map(int).map(str)
        df_pbmc['cell_idx_inmatfile'] = pbmc_fileidx["matlabindex"].map(int).map(str)
        df_pbmc['cell_tag']='pbmc2017Nov22_' + pbmc_fileidx["filename"].map(int).map(str)+'midx'+pbmc_fileidx["matlabindex"].map(int).map(str)
        df_pbmc['label'] = 'PBMC'
        df_pbmc['class'] = 0
        df_benign = df_pbmc.sample(frac=1).reset_index(drop=False)[0:n_pbmc]
        # print(df_benign.head(5))
        print(df_benign.shape)

    # pbmc_fluor_raw = scipy.io.loadmat('/home/shobi/Thesis/Data/pbmc_fluor_clean_real.mat') #42,308 x 32
    # nsclc_fluor_raw = scipy.io.loadmat('/home/shobi/Thesis/Data/nsclc_fluor_clean_real.mat') #1,031 x 32
    if cancer_type == 'acc220':
        print('constructing dataframe for ', cancer_type)
        acc220_Raw = scipy.io.loadmat(
            '/home/shobi/Thesis/Data/ShobiGatedData/acc2202017Nov22_gatedAcc220.mat')  # 28 x 416,421
        acc220_struct = acc220_Raw['acc2202017Nov22_gatedAcc220']
        df_acc220 = pd.DataFrame(acc220_struct[0, 0]['cellparam'].transpose().real)
        acc220_fileidx = acc220_struct[0, 0]['gated_idx'][0].tolist()
        acc220_features = acc220_struct[0, 0]['cellparam_label'][0].tolist()
        flist = []
        for element in acc220_features:
            flist.append(element[0])
        df_acc220.columns = flist
        acc220_fileidx = pd.DataFrame(acc220_struct[0, 0]['gated_idx'].transpose())
        acc220_fileidx.columns = ['filename', 'matlabindex']
        print('shape of fileidx', acc220_fileidx.shape)
        df_acc220['cell_filename'] = 'acc2202017Nov22_' + acc220_fileidx["filename"].map(int).map(str)
        df_acc220['cell_idx_inmatfile'] = acc220_fileidx["matlabindex"].map(int).map(
            str)  # should be same number as image number within that folder
        df_acc220['cell_tag'] = 'acc2202017Nov22_' + acc220_fileidx["filename"].map(int).map(str) + 'midx' + acc220_fileidx[
            "matlabindex"].map(int).map(str)
        df_acc220['label'] = 'acc220'
        df_acc220['class'] = 1
        df_cancer = df_acc220.sample(frac=1).reset_index(drop=False)[0:n_cancer]
        print(df_cancer.shape)

    if cancer_type == 'k562':
        print('constructing dataframe for ', cancer_type)
        K562_Raw = scipy.io.loadmat('/home/shobi/Thesis/Data/ShobiGatedData/k5622017Nov08_gatedK562.mat')
        k562_struct = K562_Raw['k5622017Nov08_gatedK562']
        df_k562 = pd.DataFrame(k562_struct[0, 0]['cellparam'].transpose().real)
        k562_features = k562_struct[0, 0]['cellparam_label'][0].tolist()
        flist = []
        for element in k562_features:
            flist.append(element[0])
        df_k562.columns = flist
        k562_fileidx = pd.DataFrame(k562_struct[0, 0]['gated_idx'].transpose())
        k562_fileidx.columns = ['filename', 'matlabindex']
        print('shape of fileidx', k562_fileidx.shape)
        df_k562['cell_filename'] = 'k5622017Nov08_' + k562_fileidx["filename"].map(int).map(str)
        df_k562['cell_idx_inmatfile'] =  k562_fileidx["matlabindex"].map(int).map(str) #should be same number as image number within that folder
        df_k562['cell_tag'] = 'k5622017Nov08_' + k562_fileidx["filename"].map(int).map(str) + 'midx' + k562_fileidx[
            "matlabindex"].map(int).map(str)
        df_k562['label'] = 'K562'
        df_k562['class'] = 1
        df_cancer = df_k562.sample(frac=1).reset_index(drop=False)[0:n_cancer]
        print(df_cancer.shape)

    if cancer_type == 'thp1':
        print('constructing dataframe for ', cancer_type)
        THP1_Raw = scipy.io.loadmat(
            '/home/shobi/Thesis/Data/ShobiGatedData/thp12017Nov22_gatedThp1.mat')  # 28 x 307,339
        thp1_struct = THP1_Raw['thp12017Nov22_gatedThp1']
        df_thp1 = pd.DataFrame(thp1_struct[0, 0]['cellparam'].transpose())
        thp1_fileidx = thp1_struct[0, 0]['gated_idx'][0].tolist()
        thp1_features = thp1_struct[0, 0]['cellparam_label'][0].tolist()
        flist = []
        idxlist = []
        for element in thp1_features:
            flist.append(element[0])
        df_thp1.columns = flist

        thp1_fileidx = pd.DataFrame(thp1_struct[0, 0]['gated_idx'].transpose())
        thp1_fileidx.columns = ['filename', 'matlabindex']
        print('shape of fileidx', thp1_fileidx.shape)
        df_thp1['cell_filename'] = 'thp12017Nov22_' + thp1_fileidx["filename"].map(int).map(str)
        df_thp1['cell_idx_inmatfile'] = thp1_fileidx["matlabindex"].map(int).map(
            str)  # should be same number as image number within that folder
        df_thp1['cell_tag'] = 'thp12017Nov022_' + thp1_fileidx["filename"].map(int).map(str) + 'midx' + thp1_fileidx[
            "matlabindex"].map(int).map(str)
        df_thp1['label'] = 'thp1'
        df_thp1['class'] = 1
        df_cancer = df_thp1.sample(frac=1).reset_index(drop=False)[0:n_cancer]
        print(df_cancer.shape)

    '''
    df_pbmc['subtype'] = ['lym' if (x <=110) & (x>=30) else 'u' for x in df_pbmc['Area']]
    df_pbmc['subtype'] = ['mon' if x >=150 else str(s) for x,s in zip(df_pbmc['Area'],df_pbmc['subtype'])]
    print(df_pbmc['subtype'].value_counts() )
    df_pbmc['cell_tag'] = 'P'+df_pbmc['subtype']+'_'+ df_pbmc['cell_tag'].map(str)
    print('subtype check complete')
    '''
    '''
    df_pbmc_fluor = pd.DataFrame(pbmc_fluor_raw['pbmc_fluor_clean_real'])
    df_pbmc_fluor = df_pbmc_fluor.replace('inf', 0)
    df_pbmc_fluor.columns = featureName_fluor
    df_pbmc_fluor['label'] = 'PBMC'
    df_pbmc_fluor['cell_tag'] = df_pbmc_fluor.index
    df_pbmc_fluor['cell_tag'] = 'P'+df_pbmc_fluor['cell_tag'].map(str)+'_'+df_pbmc_fluor["File ID"].map(int).map(str) + '_' + df_pbmc_fluor["Cell ID"].map(int).map(str)
    df_pbmc_fluor['class'] = 0
    df_pbmc_fluor = df_pbmc_fluor.sample(frac=1).reset_index(drop=True)[0:n_pbmc]
    print(df_pbmc_fluor.shape)

    df_nsclc_fluor = pd.DataFrame(nsclc_fluor_raw['nsclc_fluor_clean_real'])
    df_nsclc_fluor = df_nsclc_fluor.replace('inf', 0)
    df_nsclc_fluor.columns = featureName_fluor
    df_nsclc_fluor['label'] = 'nsclc'
    df_nsclc_fluor['cell_tag'] = df_nsclc_fluor.index
    df_nsclc_fluor['cell_tag'] = 'N'+df_nsclc_fluor['cell_tag'].map(str)+'_'+df_nsclc_fluor["File ID"].map(int).map(str) + '_' + df_nsclc_fluor["Cell ID"].map(int).map(str)
    df_nsclc_fluor['class'] = 1
    df_nsclc_fluor = df_nsclc_fluor.sample(frac=1).reset_index(drop=True)[0:n_cancer]
    print(df_pbmc_fluor.shape)
    '''
    # frames = [df_pbmc_fluor,df_nsclc_fluor]
    frames = [df_benign, df_cancer]
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
        df_all[feat_cols_includefluor] = (df_all[feat_cols_includefluor] - df_all[feat_cols_includefluor].mean()) / \
                                         df_all[feat_cols_includefluor].std()
        X_txt = df_all[feat_cols_includefluor].values

    label_txt = df_all['class'].values
    tag_txt = df_all['cell_filename'].values
    print(X_txt.size, label_txt.size)
    true_label = np.asarray(label_txt)
    true_label = np.reshape(true_label, (true_label.shape[0], 1))
    print('true label shape:', true_label.shape)
    true_label = true_label.astype(int)
    tag = np.asarray(tag_txt)
    tag = np.reshape(tag, (tag.shape[0], 1))
    index_list = list(df_all['index'].values)
    # index_list = np.reshape(index_list,(index_list.shape[0],1))
    # print('index list', index_list)
    np.savetxt(data_file_name, X_txt, comments='', header=str(n_total) + ' ' + str(int(X_txt.shape[1])), fmt="%f",
               delimiter=" ")
    np.savetxt(label_file_name, label_txt, fmt="%i", delimiter="")
    np.savetxt(tag_file_name, tag_txt, fmt="%s", delimiter="")
    return true_label, tag, X_txt, new_file_name, df_all, index_list,flist


def simple_accuracy(predicted_labels, true_labels, data_version, index_list, df_all,lr, tsne_version):
    time_start = time.time()
    print('entered accuracy of mode calculation, the time is now', time.ctime())
    # index list: the list of the original index in the original dataframe
    n_tot = true_labels.shape[0]
    n_cancer = list(true_labels).count(1)
    n_pbmc = list(true_labels).count(0)
    predicted_labels = predicted_labels.transpose()
    tn = fn = tp = fp = 0
    precision = 0
    recall = 0
    f1_score = 0
    fn_accum_list = []
    fp_accum_list = []
    fn_index_list = []
    computed_ratio = 0
    unique_version = 'lr'+str(lr)+'tsne'+str(tsne_version)
    column_names_tags = ['celltype_'+unique_version, 'filename_'+unique_version, 'idx_inmatfile_'+unique_version,'File ID_'+unique_version, 'Cell ID_'+unique_version,'df_all idx_'+unique_version]
    df_fptemp = pd.DataFrame(columns = column_names_tags)
    df_fntemp = pd.DataFrame(columns=column_names_tags)

    # print(predicted_labels)
    # print(true_labels)
    for k in range(n_tot):
        if predicted_labels[k] == 0 and true_labels[k] == 0:
            tn = tn + 1
        if (predicted_labels[k] == 0) and (true_labels[k] == 1):
            fn = fn + 1
            fn_tag_list= [df_all.loc[k,'label'],df_all.loc[k,'cell_filename'],df_all.loc[k,'cell_idx_inmatfile'],df_all.loc[k,'File ID'],df_all.loc[k,'Cell ID'],df_all.loc[k, 'index']]
            fn_accum_list.append(fn_tag_list)
            #df_temp = pd.DataFrame([fn_tag_list],columns = column_names_tags)
            #df_fntemp = df_fntemp.append(df_temp)
        if (predicted_labels[k] == 1) and (true_labels[k] == 0):
            fp = fp + 1
            fn_tag_list = [df_all.loc[k, 'label'], df_all.loc[k, 'cell_filename'], df_all.loc[k, 'cell_idx_inmatfile'],
                           df_all.loc[k, 'File ID'], df_all.loc[k, 'Cell ID'], df_all.loc[k, 'index']]
            #fp_tag_list.append([df_all.loc[k, 'index'], df_all.loc[k, 'cell_tag']])
            fp_tag_list = [df_all.loc[k, 'label'], df_all.loc[k, 'cell_filename'], df_all.loc[k, 'cell_idx_inmatfile'],
                           df_all.loc[k, 'File ID'], df_all.loc[k, 'Cell ID'], df_all.loc[k, 'index']]
            fp_accum_list.append(fp_tag_list)
            #df_temp = pd.DataFrame([fp_tag_list], columns=column_names_tags)
            #df_fptemp = df_fptemp.append(df_temp)
        if (predicted_labels[k] == 1) and (true_labels[k] == 1):
            tp = tp + 1
    if len(fp_accum_list)>0: df_fptemp = pd.concat([pd.DataFrame([i], columns = column_names_tags) for i in fp_accum_list], ignore_index=True)
    if len(fn_accum_list) > 0: df_fntemp = pd.concat([pd.DataFrame([i], columns=column_names_tags) for i in fn_accum_list], ignore_index=True)
    error_rate = (fp + fn) / n_tot
    comp_n_cancer = tp + fp
    comp_n_pbmc = fn + tn

    if n_pbmc!=0: tnr = tn / n_pbmc
    if n_cancer != 0: fnr = fn / n_cancer
    if n_cancer != 0: tpr = tp / n_cancer
    if n_pbmc != 0: fpr = fp / n_pbmc
    if comp_n_cancer != 0:
        computed_ratio = comp_n_pbmc / comp_n_cancer
        # print('computed-ratio is:', computed_ratio, ':1' )
    if tp != 0 or fn != 0: recall = tp / (tp + fn)  # ability to find all positives
    if tp != 0 or fp != 0: precision = tp / (tp + fp)  # ability to not misclassify negatives as positives
    if precision != 0 or recall != 0: f1_score = precision * recall * 2 / (precision + recall)
    print(computed_ratio, 'is the computed ratio')
    summary_simple_acc = [data_version,n_cancer,n_pbmc, f1_score, tnr, fnr, tpr, fpr, precision, recall, error_rate, computed_ratio]
    print('completed accuracy of mode calculation, the time is now', time.ctime(), 'time elapsed is: {} seconds'.format(time.time()-time_start))
    return summary_simple_acc, df_fptemp, df_fntemp


def run_lv(version, input_data, perplexity, lr, new_file_name, new_folder_name):
    outdim = 2
    threads = 8
    samples = -1
    prop = -1
    alpha = lr
    trees = -1
    neg = -1
    neigh = -1
    gamma = -1
    perp = perplexity
    fea = 1
    #alpha is the initial learning rate
    time_start = time.time()
    print('starting largevis', time.ctime())
    embedding_plot_title = new_file_name +'_lr'+str(lr)+'_lv'+str(version)
    embedding_filename = '/home/shobi/Thesis/LV_data/' + new_folder_name + '/LV_embedding' + new_file_name + '_lr' + str(
        lr) +'_perplexity'+str(perplexity)+ '_lv' + str(version) + '.txt'
    LV_input_data = input_data.tolist() # will make a list of lists. required for parsing LV input
    LargeVis.loaddata(LV_input_data)
    X_embedded_LV=LargeVis.run(outdim, threads, samples, prop, alpha, trees, neg, neigh, gamma, perp)
    X_embedded = np.array(X_embedded_LV)
    print('X_embedded shape: ',X_embedded.shape)
    time_elapsed = time.time() - time_start
    print(embedding_filename, ' LV done! Time elapsed: {} seconds'.format(time_elapsed))
    np.savetxt(embedding_filename, X_embedded, comments='',
               header=str(int(X_embedded.shape[0])) + ' ' + str(int(X_embedded.shape[1])), fmt="%f", delimiter=" ")
    return X_embedded, embedding_filename,  time_elapsed,embedding_plot_title

def run_mctsne(version, input_data,perplexity, lr, new_file_name, new_folder_name):
    time_start = time.time()
    print('starting tsne', time.ctime())
    embedding_plot_title = new_file_name +'_lr'+str(lr)+'_bh'+str(version)
    embedding_filename = '/home/shobi/Thesis/BarnesHutMC_data/'+new_folder_name+'/BH_embedding' +  new_file_name +'_lr'+str(lr)+'_perplexity'+str(perplexity)+'_bh'+str(version)+ '.txt'
    tsne = multicore_tsne(n_jobs=8, perplexity = perplexity,verbose=1,n_iter=1000,learning_rate = lr)
    params ='n_jobs=8, perplexity = ' + str(perplexity) + "verbose=1,n_iter=1000,learning_rate =" +str(lr)
    print(params)
    X_embedded = tsne.fit_transform(input_data)
    print(X_embedded.shape)
    time_elapsed = time.time() - time_start
    print(embedding_filename,' BH-MC done! Time elapsed: {} seconds'.format(time_elapsed))
    np.savetxt(embedding_filename,X_embedded,comments='', header = str(int(X_embedded.shape[0]))+' '+str(int(X_embedded.shape[1])),  fmt="%f", delimiter=" ")
    return X_embedded, embedding_filename, time_elapsed, embedding_plot_title


def run_dbscan(X_embedded, n_cancer, true_label, data_version, tsne_version, embedding_filename, tsne_runtime):
    list_roc = []
    sigma_list = [1, 0.5] #eps
    if n_cancer > 1000:
        cluster_size_list = [20]
    else:
        cluster_size_list = [15]
    iclus = 0
    for i_sigma in sigma_list:
        for i_cluster_size in cluster_size_list:
            print('Starting DBSCAN', time.ctime())
            model = DBSCAN(eps = i_sigma, min_samples = i_cluster_size).fit(X_embedded)
            time_start = time.time()
            mst_runtime = time.time() - time_start
            print('DBSCAN Done! Time elapsed: {:.2f} seconds'.format(mst_runtime),
                  'tsne version {}'.format(tsne_version), 'data version {}'.format(data_version),
                  'sigma {}'.format(i_sigma), 'and min cluster {}'.format(i_cluster_size))
            vals_roc, predict_class_array = accuracy_mst(model, true_label[:, 0],
                                                         embedding_filename, clustering_algo='dbscan')
            list_roc.append(vals_roc + [tsne_runtime])

            if iclus == 0:
                predict_class_aggregate = np.array(predict_class_array)
            else:
                predict_class_aggregate = np.vstack((predict_class_aggregate, predict_class_array))

            iclus = 1
    df_accuracy = pd.DataFrame(list_roc,
                               columns=['embedding filename', 'eps', 'min cluster size', 'merge-too-close factor','f1-score', 'tnr', 'fnr',
                                        'tpr', 'fpr', 'precision', 'recall', 'num_groups', 'clustering runtime',
                                        'tsne runtime'])

    eps_opt = df_accuracy['eps'][df_accuracy['f1-score'].idxmax()]
    print(eps_opt, 'eps opt')
    min_cluster_size_opt = df_accuracy['min cluster size'][df_accuracy['f1-score'].idxmax()]
    return predict_class_aggregate, df_accuracy, eps_opt, min_cluster_size_opt


def run_mstclustering(X_embedded, n_cancer, true_label, data_version, tsne_version, embedding_filename, tsne_runtime):
    list_roc = []

    sigma_list = [3,2.5, 2]
    tooclosefactor_list = [15,10,0]
    if n_cancer > 1000:
        cluster_size_list = [20]
        min_cluster_size = 20
    else:
        cluster_size_list = [15]
        min_cluster_size = 15
    iclus = 0
    for i_sigma in sigma_list:
        for i_tooclosefactor in tooclosefactor_list:
            model = MSTClustering(cutoff_scale=0.3, approximate=True, min_cluster_size=min_cluster_size,
                                  sigma_factor=i_sigma, tooclosefactor = i_tooclosefactor)
            time_start = time.time()
            print('Starting Clustering', time.ctime())
            model.fit_predict(X_embedded)
            mst_runtime = time.time() - time_start
            print('Clustering Done! Time elapsed: {:.2f} seconds'.format(mst_runtime),
                  'tsne version {}'.format(tsne_version), 'data version {}'.format(data_version),
                  'sigma {}'.format(i_sigma),'tooclosefactor {}'.format(i_tooclosefactor), 'and min cluster {}'.format(min_cluster_size))


            vals_roc, predict_class_array = accuracy_mst(model, true_label[:, 0], embedding_filename, clustering_algo='mst')
            list_roc.append(vals_roc + [tsne_runtime])

            if iclus == 0:
                predict_class_aggregate = np.array(predict_class_array)

            else:
                predict_class_aggregate = np.vstack((predict_class_aggregate, predict_class_array))

            iclus = 1
    df_accuracy = pd.DataFrame(list_roc,
                               columns=['embedding filename', 'sigma', 'min cluster size','merge-too-close factor', 'f1-score', 'tnr', 'fnr',
                                        'tpr', 'fpr', 'precision', 'recall', 'num_groups', 'clustering runtime',
                                        'tsne runtime'])

    sigma_opt = df_accuracy['sigma'][df_accuracy['f1-score'].idxmax()]
    min_cluster_size_opt = df_accuracy['min cluster size'][df_accuracy['f1-score'].idxmax()]
    mergetooclose_opt = df_accuracy['merge-too-close factor'][df_accuracy['f1-score'].idxmax()]
    return predict_class_aggregate, df_accuracy, sigma_opt, min_cluster_size_opt,mergetooclose_opt


def accuracy_mst(model, true_labels, embedding_filename, clustering_algo):
    if clustering_algo =='dbscan':
        sigma = model.eps
        min_cluster_size =model.min_samples
        mergetooclosefactor = None
    if clustering_algo =='mst':
        sigma = model.sigma_factor
        min_cluster_size = model.min_cluster_size
        mergetooclosefactor = model.tooclosefactor

    X_dict = {}
    Index_dict = {}
    X = model.X_fit_
    print(X.shape)
    mst_labels = list(model.labels_)

    N = len(mst_labels)
    n_cancer = list(true_labels).count(1)
    n_pbmc = list(true_labels).count(0)
    m = 999
    for k in range(N):
        x = X[k, 0]
        y = X[k, 1]
        X_dict.setdefault(mst_labels[k], []).append((x, y))
        Index_dict.setdefault(mst_labels[k], []).append(true_labels[k])
        # Index_dict_dbscan.setdefault(dbscan_labels[k], []).append(true_labels[k])
    # X_dict_dbscan.setdefault(dbscan_labels[k], []).append((x, y))
    num_groups = len(Index_dict)
    sorted_keys = list(sorted(X_dict.keys()))
    error_count = []
    pbmc_labels = []
    thp1_labels = []
    unknown_labels = []
    fp = 0
    fn = 0
    tp = 0
    tn = 0
    precision = 0
    recall = 0
    f1_score = 0

    for kk in sorted_keys:
        vals = [t for t in Index_dict[kk]]
        majority_val = func_counter(vals)
        len_unknown = 0
        if (majority_val == 1) and (kk != -1):
            thp1_labels.append(kk)
            fp = fp + len([e for e in vals if e != majority_val])
            tp = tp + len([e for e in vals if e == majority_val])
            error_count.append(len([e for e in vals if e != majority_val]))
        if (majority_val == 0) and (kk != -1):
            pbmc_labels.append(kk)
            fn = fn + len([e for e in vals if e != majority_val])
            tn = tn + len([e for e in vals if e == majority_val])
            error_count.append(len([e for e in vals if e != majority_val]))
        if majority_val == 999:
            thp1_labels.append(kk)
            unknown_labels.append(kk)
            print(kk, ' has no majority, we are adding it to cancer_class')
            fp = fp + len([e for e in vals if e != majority_val])
            tp = tp + len([e for e in vals if e == majority_val])
            error_count.append(len([e for e in vals if e != majority_val]))
    predict_class_array = np.array(mst_labels)
    mst_labels_array = np.array(mst_labels)
    for cancer_class in thp1_labels:
        predict_class_array[mst_labels_array == cancer_class] = 1
    for benign_class in pbmc_labels:
        predict_class_array[mst_labels_array == benign_class] = 0
    predict_class_array.reshape((predict_class_array.shape[0], -1))
    error_rate = sum(error_count) / N
    comp_n_cancer = tp + fp
    comp_n_pbmc = fn + tn
    tnr = tn / n_pbmc
    fnr = fn / n_cancer
    tpr = tp / n_cancer
    fpr = fp / n_pbmc
    if comp_n_cancer != 0:
        computed_ratio = comp_n_pbmc / comp_n_cancer
        # print('computed-ratio is:', computed_ratio, ':1' )
    if tp != 0 or fn != 0: recall = tp / (tp + fn)  # ability to find all positives
    if tp != 0 or fp != 0: precision = tp / (tp + fp)  # ability to not misclassify negatives as positives
    if precision != 0 or recall != 0: f1_score = precision * recall * 2 / (precision + recall)

    print('f1_score', 'fnr ', 'sigma', ' min cluster size', 'mergetooclose factor', f1_score, fnr, sigma, min_cluster_size, mergetooclosefactor)
    mst_runtime = model.clustering_runtime_
    accuracy_val = [embedding_filename, sigma, min_cluster_size, mergetooclosefactor, f1_score, tnr, fnr, tpr, fpr, precision,
                    recall, num_groups, mst_runtime]
    return accuracy_val, predict_class_array


def plot_mst_simple(model, true_labels, embedding_filename, embedding_plot_title, sigma, min_cluster,
                    cancer_type, clustering_method,df_all, feature1_name, feature2_name):
    # http://nbviewer.jupyter.org/github/jakevdp/mst_clustering/blob/master/MSTClustering.ipynb

    X_dict = {}
    Index_dict = {}
    X_plot = model.X_fit_


    mst_labels = list(model.labels_)
    num_groups = len(set(mst_labels))


    N = len(mst_labels)
    n_cancer = list(true_labels).count(1)
    n_pbmc = list(true_labels).count(0)
    m = 999
    for k in range(N):
        x = X_plot[k, 0]
        y = X_plot[k, 1]
        X_dict.setdefault(mst_labels[k], []).append((x, y))

        Index_dict.setdefault(mst_labels[k], []).append(true_labels[k])

    sorted_keys = list(sorted(X_dict.keys()))
    print('in plot: number of distinct groups:', len(sorted_keys))

    error_count = []
    pbmc_labels = []
    thp1_labels = []
    unknown_labels = []
    fp = 0
    fn = 0
    tp = 0
    tn = 0
    f1_score = 0
    precision = 0
    recall = 0
    f1_score = 0
    computed_ratio = 999

    for kk in sorted_keys:
        vals = [t for t in Index_dict[kk]]
        # print('kk ', kk, 'has length ', len(vals))
        majority_val = func_counter(vals)
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
            thp1_labels.append(kk)
            unknown_labels.append(kk)
            print(kk, ' has no majority, we are adding it to cancer_class')
            fp = fp + len([e for e in vals if e != majority_val])
            tp = tp + len([e for e in vals if e == majority_val])
            error_count.append(len([e for e in vals if e != majority_val]))
            # print(kk,' has no majority')
    # print('thp1_labels:', thp1_labels)
    # print('pbmc_labels:', pbmc_labels)
    # print('error count for each group is: ', error_count)
    # print('len unknown', len_unknown)
    error_rate = sum(error_count) / N
    # print((sum(error_count)+len_unknown)*100/N, '%')
    comp_n_cancer = tp + fp
    comp_n_pbmc = fn + tn
    tnr = tn / n_pbmc
    fnr = fn / n_cancer
    print('fnr is', fnr)
    tpr = tp / n_cancer
    fpr = fp / n_pbmc
    if comp_n_cancer != 0:
        computed_ratio = comp_n_pbmc / comp_n_cancer
        # print('computed-ratio is:', computed_ratio, ':1' )
    if tp != 0 or fn != 0: recall = tp / (tp + fn)  # ability to find all positives
    if tp != 0 or fp != 0: precision = tp / (tp + fp)  # ability to not misclassify negatives as positives
    if precision != 0 or recall != 0:
        f1_score = precision * recall * 2 / (precision + recall)
        print('f1-score is', f1_score)

    if clustering_method =='mst':
        fig, ax = plt.subplots(3, 2, figsize=(24, 24))# sharex=True, sharey=True)
        segments = model.get_graph_segments(full_graph=True)
    if clustering_method =='dbscan':
        fig, ax = plt.subplots(3, 2, figsize=(24, 24))# sharex=True, sharey=True)
    important_feature_densityplot(feature1_name, feature2_name, df_all, ax[2][0],ax[2][1])
    tsne_densityplot(X_plot, ax[1][0],ax[1][1],df_all)



    idx_cancer = np.where(np.asarray(true_labels)==1)[0]
    idx_benign = np.where(np.asarray(true_labels) == 0)[0]

    #ax[0][0].scatter(X_plot[idx_cancer, 0], X_plot[idx_cancer, 1], c=true_labels, cmap='nipy_spectral_r', zorder=2, alpha=0.5, s=4)
    ax[0][0].scatter(X_plot[idx_cancer, 0], X_plot[idx_cancer, 1], color='red', zorder=2,alpha=0.5, s=2)
    ax[0][0].scatter(X_plot[idx_benign, 0], X_plot[idx_benign, 1], color='blue', zorder=2, alpha=0.5, s=2)


    # lman = LassoManager(ax[0][0], data_lasso)
    # ax[0][0].text(0.95, 0.01, "blue: pbmc", transform=ax[0][1].transAxes, verticalalignment='bottom', horizontalalignment='right',color='green', fontsize=10)

    colors_pbmc = plt.cm.winter(np.linspace(0, 1, len(pbmc_labels)))
    colors_thp1 = plt.cm.autumn(np.linspace(0, 1, len(thp1_labels)))

    for color_p, ll_p in zip(colors_pbmc, pbmc_labels):
        x = [t[0] for t in X_dict[ll_p]]
        population = len(x)
        y = [t[1] for t in X_dict[ll_p]]
        ax[0][1].scatter(x, y, color=color_p, s=2, alpha=0.5, label='pbmc ' + str(ll_p) + ' Cellcount = ' + str(len(x)))
        ax[0][1].annotate(str(ll_p), xytext=(np.mean(x), np.mean(y)), xy=(np.mean(x), np.mean(y)), color='black',
                       weight='semibold')
        # ax[0][1].scatter(np.mean(x), np.mean(y),  color = color_p, s=population, alpha=1)
        if clustering_method == 'mst':ax[1][0].annotate(str(ll_p) + '_n' + str(len(x)), xytext=(np.mean(x), np.mean(y)), xy=(np.mean(x), np.mean(y)),
        color='black', weight='semibold')
        if clustering_method == 'mst': ax[1][0].scatter(np.mean(x), np.mean(y), color=color_p, s=3 * np.log(population), alpha=1, zorder=3)
    for color_t, ll_t in zip(colors_thp1, thp1_labels):
        x = [t[0] for t in X_dict[ll_t]]
        population = len(x)
        y = [t[1] for t in X_dict[ll_t]]
        ax[0][1].scatter(x, y, color=color_t, s=2, alpha=0.5,
                      label=cancer_type + ' ' + str(ll_t) + ' Cellcount = ' + str(len(x)))
        ax[0][1].annotate(str(ll_t), xytext=(np.mean(x), np.mean(y)), xy=(np.mean(x), np.mean(y)), color='black',
                       weight='semibold')
        if clustering_method =='mst': ax[1][0].annotate(str(ll_t) + '_n' + str(len(x)), xytext=(np.mean(x), np.mean(y)), xy=(np.mean(x), np.mean(y)),
                       color='black', weight='semibold')
        if clustering_method == 'mst':ax[1][0].scatter(np.mean(x), np.mean(y), color=color_t, s=3*np.log(population), alpha=1, zorder=3)

    if clustering_method == 'mst': ax[1][0].plot(segments[0], segments[1], '-k', zorder=2, linewidth=0.3)
    ax[0][1].text(0.95, 0.01, "number of groups: " + " {:.0f}".format(num_groups) + " FP: " + " {:.2f}".format(
        fp * 100 / n_pbmc) + "%. FN of " + "{:.2f}".format(fn * 100 / (n_cancer)) + '%', transform=ax[0][1].transAxes,
               verticalalignment='bottom', horizontalalignment='right', color='green', fontsize=10)
    for uu in unknown_labels:
        x = [t[0] for t in X_dict[uu]]
        y = [t[1] for t in X_dict[uu]]
        ax[0][1].scatter(x, y, color='gray', s=2, alpha=1, label=uu)
    ax[0][1].axis('tight')
    title_str0 = embedding_plot_title
    # title_str1 = 'MST: cutoff' + str(cutoff_scale) +' min_cluster:' +str(min_cluster_size)
    if clustering_method == 'mst':title_str1 = 'MST: mean + ' + str(sigma) + '-sigma cutoff and min cluster size of: ' + str(
        min_cluster) + '\n' + "error: " + " {:.2f}".format(error_rate * 100) + '%' + " FP: " + " {:.2f}".format(
        fp * 100 / n_pbmc) + "%. FN of " + "{:.1f}".format(
        fn * 100 / (n_cancer)) + '%' + '\n' + 'computed ratio:' + "{:.1f}".format(
        computed_ratio) + ' f1-score:' + "{:.2f}".format(f1_score * 100)
    if clustering_method == 'dbscan': title_str1 = 'DBSCAN eps + ' + str(
        sigma) + ' and min cluster size of: ' + str(
        min_cluster) + '\n' + "error: " + " {:.2f}".format(error_rate * 100) + '%' + " FP: " + " {:.1f}".format(
        fp * 100 / n_pbmc) + "%. FN of " + "{:.1f}".format(
        fn * 100 / (n_cancer)) + '%' + '\n' + 'computed ratio:' + "{:.1f}".format(
        computed_ratio) + ' f1-score:' + "{:.2f}".format(f1_score * 100)
    title_str2 = 'Cluster centers and connectivity on density plot of tsne embedding'
    # ax[1][0].set_title(graph_title_force, size=16)
    ax[0][1].set_title(title_str1, size=10)
    ax[0][0].set_title(title_str0, size=10)

    if clustering_method == 'mst': ax[1][0].set_title(title_str2, size=10)

    # Put a legend to the right of the current axis
    ##box = ax[0][1].get_position()
    ##ax[0][1].set_position([box.x0, box.y0, box.width *0.9, box.height])
    ##ax[0][1].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize = 6)
    plt.savefig(embedding_filename[:-4] + '_'+clustering_method+ '.png', bbox_inches='tight')

    mst_runtime = model.clustering_runtime_
    roc_val = [embedding_filename, sigma, min_cluster, f1_score, tnr, fnr, tpr, fpr, precision, recall, num_groups,
               mst_runtime]
    print(roc_val, 'roc_val from plotmstsimple')
    # plt.show()
    return roc_val



def func_mode(ll):  # return MODE of list
    # If multiple items are maximal, the function returns the first one encountered.
    return max(set(ll), key=ll.count)


def func_counter(ll):
    c_0 = ll.count(0)
    c_1 = ll.count(1)
    if c_0 > c_1: return 0
    if c_0 < c_1: return 1
    if c_0 == c_1: return 999


def auc_accuracy(df_roc):
    fpr_list = df_roc['fpr'].values.tolist()
    tpr_list = df_roc['tpr'].values.tolist()
    fnr_list = df_roc['fnr'].values.tolist()
    tnr_list = df_roc['tnr'].values.tolist()
    precision_list = df_roc['precision'].values.tolist()
    recall_list = df_roc['recall'].values.tolist()
    fpr_list.append(1)
    tpr_list.append(1)
    tpr_list.insert(0, 0)
    fpr_list.insert(0, 0)
    fnr_list.append(1)
    tnr_list.append(1)
    fnr_list.insert(0, 0)
    tnr_list.insert(0, 0)
    precision_list.append(0)
    recall_list.append(1)
    precision_list.insert(0, 0)
    recall_list.insert(0, 1)
    auc_pr_val = metrics.auc(precision_list, recall_list, reorder=True)
    auc_roc_val = metrics.auc(fpr_list, tpr_list, reorder=True)
    print('precision-recall AUC:', auc_pr_val)
    print('ROC-AUC (fpr vs. tpr):', auc_roc_val)
    return auc_pr_val, auc_roc_val

def tsne_densityplot(X_embedded,ax1,ax2,df_all):
    # instantiate and fit the KDE model
    xy = X_embedded.T #xy is 2xn matrix
    print('embedded shape', xy.shape)
    d = xy.shape[0]
    n = xy.shape[1]
    print('dxn:', d, n)
    bw = (n * (d + 2) / 4.) ** (-1. / (d + 4))  # silverman
    # bw = n**(-1./(d+4)) # scott
    print('bw: {}'.format(bw))
    kde = KernelDensity(bandwidth=bw, metric='euclidean',
                        kernel='gaussian', algorithm='ball_tree')
    kde.fit(xy.T)
    print('finished kde.fit()')
    xmin = xy[0].min()-1#max along 0th column
    xmax = xy[0].max()+1
    ymin = xy[1].min()-1
    ymax = xy[1].max()+1
    print('tsne x and y min max coordin:', xmin,xmax, ymin,ymax)
    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]

    positions = np.vstack([X.ravel(), Y.ravel()]).T #nx2 matrix whre first column is x-coordin and 2nd is y-coordinates
    Z_tsne = np.reshape(np.exp(kde.score_samples(positions)), X.shape)
    # plot contours of the density
    plot_peaks(X, Y,positions, Z_tsne, ax1, ax2, density_type='tsne')
    c_keep = df_all['class'].values.reshape((-1, 1))
    c_keep = c_keep.astype(str)

    c_keep[c_keep == '0'] = 'blue'
    c_keep[c_keep == '1'] = 'red'
    ax1.scatter(X_embedded[:, 0], X_embedded[:, 1], c=c_keep[:, 0], s=1, alpha=0.2, zorder=2)

    ax1.set_xlabel('tsne feature1')
    ax1.set_ylabel('tsne feature2')
    ax1.set_title('Cluster centers and connectivity on density plot of tsne embedding', size=10)
    ax2.set_xlabel('tsne feature1')
    ax2.set_ylabel('tsne feature2')
    ax2.set_title('Intensity peaks on density plot of tsne embedding', size=10)

    # ax[1][1].imshow(np.rot90(Z), cmap=plt.cm.viridis, extent=[xmin, xmax, ymin, ymax])
def important_features(X_data, true_label,flist,df_all):
    print(true_label[0:10], true_label[-5:])
    from sklearn.ensemble import RandomForestRegressor
    time_start = time.time()
    print('current time is:', time.ctime())
    X = X_data
    Y = true_label[:,0]
    names = flist #list of feature names
    rf = RandomForestRegressor()
    rf.fit(X, Y)
    print('time elapsed: {:.2f} seconds'.format(time.time()-time_start))
    print("Features sorted by their score:")
    sorted_feature_list = sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), names),
           reverse=True)
    print(sorted_feature_list)
    feature1_name = sorted_feature_list[0][1]
    feature2_name = sorted_feature_list[1][1]
    print(feature1_name, feature2_name)
    return feature1_name, feature2_name
def important_feature_densityplot(feature1_name, feature2_name, df_all,ax1,ax2):
    feature1 = df_all[feature1_name].values.reshape((-1,1))
    feature2 = df_all[feature2_name].values.reshape((-1,1))
    X_features = np.concatenate((feature1, feature2), axis=1) #should give nx2 array
    print('min values of feature1 and 2', feature1.min(), feature2.min())
    idx1 = np.where(abs(feature1) < 5* np.std(feature1))
    print('idx1', idx1[0].shape)
    idx2 = np.where(abs(feature2) < 5* np.std(feature2))
    print('idx2', idx2[0].shape)
    idx =  np.intersect1d(idx1[0],idx2[0])
    print('idx intersection length', len(idx))
    # print('ckeep shape', c_keep.shape)
    X_keep = X_features[idx,:]
    print('xkeep shape', X_keep.shape)
    # instantiate and fit the KDE model
    xy = X_keep.T #xy is a 2xn matrix
    d = xy.shape[0]
    n = xy.shape[1]
    print('dxn:', d, n)
    bw = (n * (d + 2) / 4.) ** (-1. / (d + 4))  # silverman
    # bw = n**(-1./(d+4)) # scott
    print('bw: {}'.format(bw))
    kde = KernelDensity(bandwidth=bw, metric='euclidean',
                        kernel='gaussian', algorithm='ball_tree')
    kde.fit(xy.T) #xy.T is nx2 matrix
    print('finished kde.fit()')
    print('xy has shape:', xy.shape)
    xmin = xy[0].min()-1
    xmax = xy[0].max()+1
    ymin = xy[1].min()-1
    ymax = xy[1].max()+1
    print(xmin,xmax,ymin,ymax)

    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]

    print('X shape', X.shape)
    positions = np.vstack([X.ravel(), Y.ravel()]).T
    print(positions.shape)

    Z = np.reshape(np.exp(kde.score_samples(positions)), X.shape)
    print('z shape', Z.shape)

    c_keep = df_all['class'].values.reshape((-1, 1))[idx]
    c_keep = c_keep.astype(str)
    c_keep_list = list(c_keep)

    c_keep[c_keep == '0'] = 'blue'
    c_keep[c_keep == '1'] = 'red'
    # plot contours of the density


    levels = np.linspace(0, Z.max(), 25)

    plot_peaks(X, Y,positions, Z, ax1, ax2, density_type='features')
    ax1.scatter(X_keep[:, 0], X_keep[:, 1], c=c_keep[:, 0], s=1, alpha=0.2,zorder=2)
    print('feature1 is ', feature1_name)
    ax1.set_xlabel(feature1_name)
    ax1.set_ylabel(feature2_name)
    ax1.set_title('Density plot of highest ranked features', size=10)
    ax2.set_xlabel(feature1_name)
    ax2.set_ylabel(feature2_name)
    ax2.set_title('Intensity peaks on density plot of highest ranked features', size=10)
    #plt.show()
    return


def detect_peaks(image, density_type = 'features'):
    from scipy.ndimage.filters import maximum_filter
    from scipy.ndimage.morphology import generate_binary_structure, binary_erosion, binary_dilation
    from skimage.feature import peak_local_max
    """
    Takes an image and detect the peaks usingthe local maximum filter.
    Returns a boolean mask of the peaks (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
    """
    image_max = maximum_filter(image,size=3,mode = 'constant')
    detected_peaks = peak_local_max(image, min_distance = 3)
    average_intensity = np.mean(image)
    std_intensity = np.std(image)
    keep_detected_peaks = []
    print('all detected peak shape', detected_peaks.shape)
    if density_type =='features':
        threshold = average_intensity
        print('threshold is ', threshold)
    else:
        threshold = average_intensity + 3 * std_intensity
        print('threshold is ', threshold)
    for row in detected_peaks:
        xp = row[0]
        yp = row[1]
        if(image[xp,yp] > threshold):
            keep_detected_peaks.append(row)
    print(keep_detected_peaks)
    print('image shape is ', image.shape)
    keep_detected_peaks = np.asarray(keep_detected_peaks)
    print('kept detected peak shape', keep_detected_peaks.shape)
    #a = np.zeros((7,7))
    #a[3,3]=1
    #struct1 = generate_binary_structure(2, 1)
    #neighborhood = binary_dilation(a, structure=struct1).astype(a.dtype)
    # define an 8-connected neighborhood
    #neighborhood = generate_binary_structure(2,2)

    #apply the local maximum filter; all pixel of maximal value
    #in their neighborhood are set to 1
    #local_max = maximum_filter(image, footprint=neighborhood)==image
    #local_max = maximum_filter(image, size = 10) == image
    #print(np.where(local_max==True), 'are local max')
    #local_max is a mask that contains the peaks we are
    #looking for, but also the background.
    #In order to isolate the peaks we must remove the background from the mask.

    #we create the mask of the background
    background = (image==0)

    #a little technicality: we must erode the background in order to
    #successfully subtract it form local_max, otherwise a line will
    #appear along the background border (artifact of the local maximum filter)
    #eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)

    #we obtain the final mask, containing only peaks,
    #by removing the background from the local_max mask (xor operation)
    #detected_peaks = local_max #^ eroded_background

    return keep_detected_peaks

def plot_peaks(X,Y, positions, image,ax1,ax2, density_type):
    #applying the detection and plotting results
    detected_peaks = detect_peaks(image, density_type)
    levels = np.linspace(0, image.max(), 25)
    ax1.contourf(X,Y,image,levels=levels, cmap=plt.cm.Greens, zorder=1)
    s= ax2.contourf(X,Y, image,levels=levels, cmap=plt.cm.Greens, zorder=1)
    #print('x detected peak', detected_peaks[:, 0])
    #print('y detected peak', detected_peaks[:, 1])
    xco = X[detected_peaks[:,0],0]
    yco = Y[0,detected_peaks[:,1]]
    #print(Y)
    #print('xco,yco', xco, yco)
    #ax2.scatter(detected_peaks[:, 1],detected_peaks[:,0], c= 'red')
    if density_type =='features':
        size_peaks = 12
        print('setting peak point size to ', size_peaks)
    else:
        size_peaks = 6
        print('setting peak point size to ', size_peaks)
    ax2.scatter(xco, yco, c='red', s = size_peaks)
    cb = plt.colorbar(s, ax = ax2)

    #ax2.contourf(X,Y,detected_peaks,levels=levels, cmap=plt.cm.Greens, zorder=2)


def main():
    #REMEMBER TO DISABLE DBSCAN IF COUNT IS GREATER THAN 500K
    method = 'bh'
    perplexity = 30
    n_cancer =50000
    n_benign =50000
    ratio = n_benign/n_cancer
    n_total = n_cancer + (ratio * n_cancer)
    '''
    if n_total >= 250000: lr = 2000
    if n_total >=100000 and n_total < 250000: lr= 1500
    if n_total <100000: lr = 1000

    '''
    cancer_type = 'acc220'
    benign_type = 'pbmc'
    fluor = 0
    num_dataset_versions = 3
    num_tsne_versions = 1
    dataset_version_range = range(num_dataset_versions)
    tsne_version_range = range(num_tsne_versions)
    list_accuracy_opt0 = []
    list_accuracy_opt1 = []
    if n_total < 500000:
        list_accuracy_opt_dbscan = []
        pred_list_dbscan = []
    pred_list = []

    if method =='lv':
        lr = 1
        perplexity = 30
        lr_range = [lr]#[0.5, 1, 1.5,2, 2.5]
        perp_range = [10,30,50,100,150]
    if method == 'bh':
        lr = 1000
        perplexity = 30
        lr_range = [lr]#[100,1000,1500,2000,2500]
        perp_range = [10, 30, 50, 100, 150]
    new_folder_name = cancer_type + '_r{:.2f}'.format(ratio) + '_n' + str(n_cancer)
    if method == 'bh': path_tocreate = '/home/shobi/Thesis/BarnesHutMC_data/' + new_folder_name
    if method == 'lv':path_tocreate = '/home/shobi/Thesis/LV_data/' + new_folder_name
    os.mkdir(path_tocreate)

    lr_file = lr_range[0]
    for dataset_version in dataset_version_range:
        if method == 'lv':
            excel_file_name = '/home/shobi/Thesis/LV_data/' + new_folder_name + '/LV_excel_' + cancer_type + '_data' + str(dataset_version) +'_r{:.2f}'.format(ratio) + '_ncancer' + str(n_cancer) + '.xlsx'
        else: excel_file_name = '/home/shobi/Thesis/BarnesHutMC_data/' + new_folder_name + '/BH_excel_' + cancer_type + '_data' + str(dataset_version) +'_r{:.2f}'.format(ratio)+ '_ncancer' + str(n_cancer) + '.xlsx'
        writer = ExcelWriter(excel_file_name)
        true_label, tag, X_data, new_file_name, df_all, index_list,flist = get_data(cancer_type, benign_type, n_cancer, ratio,
                                                                              fluor, dataset_version, new_folder_name, method = method)
        print('flist:', flist[2:])
        feature1_name, feature2_name = important_features(X_data, true_label, flist[2:], df_all)
        for perplexity in perp_range:
            print(lr, 'is lr')
            for tsne_version in tsne_version_range:
                new_folder_name = cancer_type + '_r{:.2f}'.format(ratio)+ '_n' + str(n_cancer)
                print('dataset_version', dataset_version, 'tsne_version is', tsne_version)
                if method == 'lv':
                    X_embedded, embedding_filename, tsne_runtime, embedding_plot_title = run_lv(tsne_version, X_data, perplexity, lr,
                                                                          new_file_name, new_folder_name)
                if method =='bh': X_embedded, embedding_filename, tsne_runtime, embedding_plot_title = run_mctsne(tsne_version, X_data, perplexity, lr,
                                                                          new_file_name, new_folder_name)
                predict_class_aggregate, df_accuracy, sigma_opt, min_cluster_size_opt, tooclosefactor_opt = run_mstclustering(
                    X_embedded, n_cancer, true_label, dataset_version, tsne_version, embedding_filename, tsne_runtime)
                if n_total < 500000:
                    predict_class_aggregate_dbscan, df_accuracy_dbscan, eps_opt, min_cluster_size_opt_dbscan = run_dbscan(X_embedded, n_cancer, true_label, dataset_version, tsne_version, embedding_filename, tsne_runtime)
                    print('predict_class_aggregate_dbscan', predict_class_aggregate_dbscan.shape)
                temp1, temp2 = auc_accuracy(df_accuracy)
                auc_list = [temp1,temp2, tsne_runtime]

                if n_total < 500000:
                    temp1, temp2 = auc_accuracy(df_accuracy_dbscan)
                    auc_list_dbscan = [temp1,temp2, tsne_runtime]

                if tsne_version == 0 :#and lr == lr_range[0]:
                    print(tsne_version, 'agg')
                    predict_class_aggregate_all = predict_class_aggregate
                    print('shape of predict_class_aggregate', predict_class_aggregate_all.shape)

                    if n_total < 500000: predict_class_aggregate_all_dbscan = predict_class_aggregate_dbscan

                # if dataset_version ==0 and tsne_version==0:
                if perplexity == perp_range[0] and tsne_version == 0:
                    df_all_merge = df_accuracy

                    if n_total < 500000: df_all_dbscan = df_accuracy_dbscan
                else:
                    df_all_merge = pd.concat([df_all_merge, df_accuracy], ignore_index=True)

                    if n_total < 500000:df_all_dbscan = pd.concat([df_all_dbscan, df_accuracy_dbscan], ignore_index=True)
                if tsne_version != 0:
                    predict_class_aggregate_all = np.vstack((predict_class_aggregate_all, predict_class_aggregate))

                    if n_total < 500000:
                        predict_class_aggregate_all_dbscan = np.vstack((predict_class_aggregate_all_dbscan, predict_class_aggregate_dbscan))
                        print(dataset_version, tsne_version, 'd and t version')

                model = MSTClustering(cutoff_scale=0.3, approximate=True, min_cluster_size=min_cluster_size_opt,
                                      sigma_factor=sigma_opt, tooclosefactor=tooclosefactor_opt)
                model.fit_predict(X_embedded)
                if n_total < 500000:model_dbscan = DBSCAN(eps_opt, min_cluster_size_opt_dbscan).fit(X_embedded)
                print('auc list', auc_list )
                list_accuracy_opt1.append(
                    plot_mst_simple(model, true_label[:, 0], embedding_filename, embedding_plot_title,sigma_opt,min_cluster_size_opt, cancer_type,clustering_method='mst',df_all = df_all,feature1_name = feature1_name,feature2_name=feature2_name) + auc_list)

                if n_total < 500000:
                    list_accuracy_opt_dbscan.append(plot_mst_simple(model_dbscan, true_label[:, 0], embedding_filename, embedding_plot_title,eps_opt,min_cluster_size_opt_dbscan, cancer_type,clustering_method='dbscan',df_all = df_all, feature1_name=feature1_name,feature2_name=feature2_name) + auc_list_dbscan)
                    print(predict_class_aggregate_all_dbscan.shape, 'predict_class_aggregate_all_dbscan shape')
            print('aggregate predictions has shape ', predict_class_aggregate.shape)
            predict_class_final = stats.mode(predict_class_aggregate_all)[0]
            if n_total < 500000:predict_class_final_dbscan = stats.mode(predict_class_aggregate_all_dbscan)[0]
            print('mode of predictions has shape ', predict_class_final.shape)
            summary_simple_acc, df_fp_temp, df_fn_temp = simple_accuracy(predict_class_final, true_label, dataset_version,
                                                                   index_list, df_all,lr, tsne_version)
            if n_total < 500000:
                print(predict_class_final_dbscan.shape,predict_class_final.shape)
                summary_simple_acc_dbscan, df_fp_temp_dbscan, df_fn_temp_dbscan = simple_accuracy(predict_class_final_dbscan, true_label, dataset_version,index_list, df_all,lr, tsne_version)
            pred_list.append([perplexity, lr] + summary_simple_acc)
            if n_total < 500000: pred_list_dbscan.append([perplexity,lr] + summary_simple_acc_dbscan)
            df_fn_temp.to_excel(writer, 'fn tags'+'perp'+str(perplexity)+'d'+str(dataset_version), index=False)  # per lr (i.e. same frequency as mode)
            df_fp_temp.to_excel(writer, 'fp tags'+'perp'+str(perplexity)+'d'+str(dataset_version), index=False)
            ##fp_tag_list.append(fp_tags)
            #fn_tag_list.append(fn_tags)
        #df_fp_tags = pd.DataFrame(fp_tag_list).transpose()
        #df_fn_tags = pd.DataFrame(fn_tag_list).transpose()
        df_mode = pd.DataFrame(pred_list,
                               columns=['perplexity','learning rate', 'data version', 'n_cancer', 'n_pbmc', 'f1-score', 'tnr', 'fnr', 'tpr', 'fpr',
                                        'precision', 'recall', 'error_rate', 'computed_ratio'])
        if n_total < 500000: df_mode_dbscan = pd.DataFrame(pred_list_dbscan,columns=['perplexity','learning rate', 'data version','n_cancer', 'n_pbmc', 'f1-score', 'tnr', 'fnr', 'tpr', 'fpr',
                                        'precision', 'recall', 'error_rate', 'computed_ratio'])
        df_accuracy_opt1 = pd.DataFrame(list_accuracy_opt1,
                                        columns=['embedding filename', 'sigma', 'min cluster size', 'f1-score', 'tnr',
                                                 'fnr', 'tpr', 'fpr', 'precision', 'recall', 'num_groups',
                                                 'clustering runtime', 'auc prec', 'auc roc', 'tsne runtime'])


        df_all_merge.to_excel(writer, 'All', index=False)
        df_accuracy_opt1.to_excel(writer, 'merged_too_close',index=False) #best tsne run for each dataset (with merging)

        df_mode.to_excel(writer, 'Mode',index=False) #one mode per set of equal param tsne
        if n_total < 500000: df_mode_dbscan.to_excel(writer, 'DBSCAN Mode')  #one mode per set of equal param tsne. so if there are 4 tsnes per Learning rate, then these 4 tsnes yield one mode
        writer.save()
        print('successfully saved excel files')
        #df_fn_tags.to_excel(writer_tags, 'fn tags',index=False) #per mode
        #df_fp_tags.to_excel(writer_tags, 'fp tags',index=False)
        #writer_tags.save()

if __name__ == '__main__':
    main()