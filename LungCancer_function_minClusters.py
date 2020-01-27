'''
Latest version of Lung Cancer classifier using ALPHA and APT.
Can choose between Jan, May and June datasets
'''
from MulticoreTSNE import MulticoreTSNE as multicore_tsne  # https://github.com/DmitryUlyanov/Multicore-TSNE
import Louvain_igraph as ls
import Performance_phenograph as pp
import numpy as np
import time
import scipy.io
import pandas as pd
import matplotlib.pyplot as plt
from MST_minClusters import MSTClustering
from MST_3D_4Oct import MSTClustering3D
from MST_2D import MSTClustering2D
import Plotting_3D as Plotting_3D
from pandas import ExcelWriter
from Louvain_igraph import accuracy_mst
import time
import sklearn.cluster
from sklearn.cluster import DBSCAN, KMeans
import os.path
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn import metrics

print(os.path.abspath(sklearn.cluster.__file__))

def get_data(fluor=0, n_eachsubtype=None, randomseedval=1):
    fluor = fluor

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

    # January, May(H526) and June (HCC827)

    df_h2170 = make_subtype_df('h2170','h21702018Jan23_gatedH2170',0,n_eachsubtype, randomseedval, HTC_filename ="\\\\Desktop-u14r2et\\G\\2018Jan23\\" ) #h2170_Raw, 28 x 302,635
    df_h1975 = make_subtype_df('h1975','h19752018Jan23_gatedH1975',1,n_eachsubtype, randomseedval, HTC_filename= 'F:\\') #60447
    df_h526 = make_subtype_df('h562','H5262018May24_gatedH526',2,n_eachsubtype, randomseedval, HTC_filename = '\\\\Desktop-p9kngca\htc-4(b)\\2018May24_cancer\\')#375889
    df_h520 = make_subtype_df('h560','h5202018Jan03_gatedH520',3,n_eachsubtype, randomseedval, HTC_filename= '\\\\DESKTOP-H5E5CH1\\DVDproc - 2 (G)\\20180103 H520\\')#451208
    df_h358 = make_subtype_df('h358','h3582018Jan03_gatedH358',4,n_eachsubtype, randomseedval, HTC_filename='\\\\DESKTOP-H5E5CH1\\DVDproc - 2\\2018Jan3_h358_520_526\\')#170198
    df_h69 = make_subtype_df('h69','h692018Jan23_gatedH69',5,n_eachsubtype, randomseedval,HTC_filename= 'F:\\') #130075
    df_hcc827 = make_subtype_df('hcc827','hcc8272018Jun05_gatedHcc827',6,n_eachsubtype, randomseedval, HTC_filename='\\\\Desktop-p9kngca\\htc-4(b)\\2018Jun05_lungcancer\\')
    '''
    # May, January (H520), June (HCC827)
    df_h2170 = make_subtype_df('H21702018May24_gatedH2170',0,n_eachsubtype, randomseedval, HTC_filename ='\\\\Desktop-p9kngca\htc-4(b)\\2018May24_cancer\\' ) #h2170_Raw, 28 x 302,635
    df_h1975 = make_subtype_df('H19752018May24_gatedH1975',1,n_eachsubtype, randomseedval, HTC_filename= '\\\\Desktop-p9kngca\htc-4(b)\\2018May24_cancer\\') #60447
    df_h526 = make_subtype_df('H5262018May24_gatedH526',2,n_eachsubtype, randomseedval, HTC_filename = '\\\\Desktop-p9kngca\htc-4(b)\\2018May24_cancer\\')#375889
    df_h520 = make_subtype_df('h5202018Jan03_gatedH520', 3, n_eachsubtype, randomseedval, HTC_filename='\\\\DESKTOP-H5E5CH1\\DVDproc - 2 (G)\\20180103 H520\\')  # 451208
    df_h358 = make_subtype_df('H3582018May24_gatedH358',4,n_eachsubtype, randomseedval, HTC_filename='\\\\Desktop-p9kngca\htc-4(b)\\2018May24_cancer\\')#170198
    df_h69 = make_subtype_df('H692018May24_gatedH69',5,n_eachsubtype, randomseedval,HTC_filename= '\\\\Desktop-p9kngca\htc-4(b)\\2018May24_cancer\\') #130075
    df_hcc827 = make_subtype_df('hcc8272018Jun05_gatedHcc827',6,n_eachsubtype, randomseedval, HTC_filename='\\\\Desktop-p9kngca\\htc-4(b)\\2018Jun05_lungcancer\\')
    '''
    # June, May(H526)
    '''
    df_h2170 = make_subtype_df('h21702018Jun05_gatedH2170',0,n_eachsubtype, randomseedval, HTC_filename ="\\\\Desktop-p9kngca\\htc-4(b)\\2018Jun05_lungcancer\\" ) #h2170_Raw, 28 x 302,635
    df_h1975 = make_subtype_df('h19752018Jun05_gatedH1975',1,n_eachsubtype, randomseedval, HTC_filename= '\\\\Desktop-p9kngca\\htc-4(b)\\2018Jun05_lungcancer\\') #60447
    df_h526 = make_subtype_df('H5262018May24_gatedH526',2,n_eachsubtype, randomseedval, HTC_filename = '\\\\Desktop-p9kngca\htc-4(b)\\2018May24_cancer\\')#375889
    df_h520 = make_subtype_df('h5202018Jun05_gatedH520',3,n_eachsubtype, randomseedval, HTC_filename= '\\\\Desktop-p9kngca\\htc-4(b)\\2018Jun05_lungcancer\\')#451208
    df_h358 = make_subtype_df('h3582018Jun05_gatedH358',4,n_eachsubtype, randomseedval, HTC_filename='\\\\Desktop-p9kngca\\htc-4(b)\\2018Jun05_lungcancer\\')#170198
    df_h69 = make_subtype_df('h692018Jun05_gatedH69',5,n_eachsubtype, randomseedval,HTC_filename= '\\\\Desktop-p9kngca\\htc-4(b)\\2018Jun05_lungcancer\\') #130075
    df_hcc827 = make_subtype_df('hcc8272018Jun05_gatedHcc827', 6, n_eachsubtype, randomseedval, HTC_filename='\\\\Desktop-p9kngca\\htc-4(b)\\2018Jun05_lungcancer\\')
    '''
    frames = [df_h2170, df_h1975, df_h526, df_h520 ,df_h358, df_h69,df_hcc827]
    df_all = pd.concat(frames, ignore_index=True,sort=False)


    # EXCLUDE FLUOR FEATURES
    if fluor == 0:
        #print('raw min', df_all[feat_cols].min())
        #df_all[feat_cols] = (df_all[feat_cols] + df_all[feat_cols].min().abs())+1
        #print('min',df_all[feat_cols].min())
        #df_all[feat_cols] = df_all[feat_cols].apply(np.log)
        #print('mean', df_all[feat_cols].mean())
        #print('std', df_all[feat_cols].std())
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
    true_label = np.asarray(label_txt)
    true_label = true_label.astype(int)
    print('data matrix size', X_txt.size)
    print('true label shape:', true_label.shape)

    return df_all, true_label, X_txt, feat_cols



def make_subtype_df(str_subtypename, subtype_name, class_val,n_eachsubtype=None, randomseedval = 1, HTC_filename='dummy'):
    print('getting ', subtype_name)
    print(randomseedval, ' is the randomseed value')
    subtype_raw = scipy.io.loadmat(
        '/home/shobi/Thesis/Data/ShobiGatedData/LungCancer_ShobiGatedData_cleanup/'+subtype_name+'.mat')
    subtype_struct = subtype_raw[subtype_name]
    df_subtype = pd.DataFrame(subtype_struct[0, 0]['cellparam'].transpose().real)
    subtype_features = subtype_struct[0, 0]['cellparam_label'][0].tolist()
    subtype_fileidx = pd.DataFrame(subtype_struct[0, 0]['gated_idx'].transpose())
    flist = []
    for element in subtype_features:
        flist.append(element[0])
    df_subtype.columns = flist
    subtype_fileidx.columns = ['filename', 'index']
    print('shape of fileidx', subtype_fileidx.shape)
    df_subtype['cell_filename'] = HTC_filename + subtype_name.split('_')[0] + '_' + subtype_fileidx["filename"].map(
            int).map(str)
    if class_val == 3 or class_val==4:
        df_subtype['cell_filename'] = HTC_filename + subtype_name.split('_')[0] + '_' + subtype_fileidx["filename"].map(
            int).map(str)+'mat'
        print('class val is ',class_val)

    df_subtype['cell_idx_inmatfile'] = subtype_fileidx["index"]#.map(int).map( str)  # should be same number as image number within that folder
    df_subtype['cell_tag'] = subtype_name.split('_')[0] + subtype_fileidx["filename"].map(int).map(str) + '_' + \
                              subtype_fileidx["index"].map(int).map(str)
    df_subtype['label'] = class_val
    df_subtype['class'] = class_val
    df_subtype['class_name'] = str_subtypename
    print('shape before drop dups', df_subtype.shape)
    df_subtype = df_subtype.drop_duplicates(subset = flist, keep='first')
    df_subtype.replace([np.inf, -np.inf], np.nan).dropna()
    print('shape after drop dups', df_subtype.shape)
    print(df_subtype.head(5))
    #if class_val ==6:
        #df_subtype = df_subtype.sample(frac=1).reset_index(drop=True)
    if n_eachsubtype ==None:
        df_subtype = df_subtype.sample(frac=1).reset_index(drop=False)
    if n_eachsubtype !=None:
        if n_eachsubtype < df_subtype.shape[0]:
            df_subtype = df_subtype.sample(frac=1, random_state = randomseedval).reset_index(drop=False)[0:n_eachsubtype]
        else: df_subtype = df_subtype.sample(frac=1, random_state = randomseedval).reset_index(drop=False)
    print(df_subtype.shape)
    return df_subtype


def func_mode(ll):  # return MODE of list
    # If multiple items are maximal, the function returns the first one encountered.
    return max(set(ll), key=ll.count)


def func_counter(ll): #for binary classifier
    c_0 = ll.count(0)
    c_1 = ll.count(1)
    if c_0 > c_1: return 0
    if c_0 < c_1: return 1
    if c_0 == c_1: return 999

def plot_all_methods(X_embedded, true_label, embedding_filename, dbscan_labels, mst_labels, louvain_labels, pheno_labels, kmeans_labels, onevsall_mst, onevsall_dbscan,onevsall_louvain,onevsall_pheno, onevsall_kmeans, dimred,sigma_opt, eps_opt, min_cluster_opt,dbscan_min_clustersize, knn_opt):

    fig, ax = plt.subplots(1, 2, figsize=(24, 12), sharex=True, sharey=True)
    # if one_vs_all == 'h2170': onevsall = 0
    # if one_vs_all == 'h1975': onevsall =1
    # if one_vs_all == 'h526': onevsall =2
    # if one_vs_all == 'h520': onevsall = 3
    # if one_vs_all == 'h358': onevsall = 4
    # if one_vs_all == 'h69': onevsall = 5
    # if one_vs_all == 'hcc827': onevsall = 6

    X_dict = {}
    X_dict_true = {}
    X_dict_dbscan = {}
    Index_dict = {}
    Index_dict_dbscan = {}

    X_plot = X_embedded

    N = len(true_label)
    if knn_opt==None:
        for k in range(N):
            x = X_plot[k, 0]
            y = X_plot[k, 1]
            X_dict.setdefault(mst_labels[k], []).append((x, y)) #coordinates of the points by mst groups
            Index_dict.setdefault(mst_labels[k], []).append(true_label[k]) #true label kth data point grouped by mst_group
            X_dict_true.setdefault(true_label[k],[]).append((x,y))
    if knn_opt!=None:
        for k in range(N):
            x = X_plot[k, 0]
            y = X_plot[k, 1]
            X_dict.setdefault(louvain_labels[k], []).append((x, y)) #coordinates of the points by mst groups
            Index_dict.setdefault(louvain_labels[k], []).append(true_label[k]) #true label kth data point grouped by mst_group
            X_dict_true.setdefault(true_label[k],[]).append((x,y))

    for true_group in X_dict_true.keys():

        if true_group ==0:
            true_color = 'gray'
            true_label_str = 'h2170'
        if true_group ==1:
            true_color = 'forestgreen'
            true_label_str = 'h1975'
        if true_group ==2:
            true_color = 'orange'
            true_label_str = 'h526'
        if true_group == 3:
            true_color = 'red'
            true_label_str = 'h520'
        if true_group == 4:
            true_color = 'mediumpurple'
            true_label_str = 'h358'
        if true_group == 5:
            true_color = 'deepskyblue'
            true_label_str = 'h69'
        if true_group==6:
            true_color = 'lightpink'
            true_label_str = 'hcc827'
        print('true group', true_group, true_color, true_label_str)
        x = [t[0] for t in X_dict_true[true_group]]
        y = [t[1] for t in X_dict_true[true_group]]
        population = len(x)
        ax[0].scatter(x, y, color=true_color, s=2, alpha=0.6, label=true_label_str+' Cellcount = ' + str(population))
        ax[0].annotate(true_label_str, xytext=(np.mean(x), np.mean(y)), xy=(np.mean(x), np.mean(y)), color='black',
                       weight='semibold')
    box = ax[0].get_position()
    ax[0].set_position([box.x0, box.y0, box.width *0.9, box.height])
    ax[0].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize = 6)
    title_str0 = 'Multi-Class Lung Cancer Cell Lines:Ground Truth. \n'+'Total Cell count is ' +str(N)  # embedding_filename
    ax[0].set_title(title_str0, size=10)
    if knn_opt==None:
        ax[1]= plot_onemethod(ax[1],X_embedded,mst_labels, true_label,onevsall_mst, 'mst', dimred,sigma_opt, min_cluster_opt, None)
    if knn_opt!=None:
        ax[1] = plot_onemethod(ax[1], X_embedded, louvain_labels, true_label, onevsall_louvain, 'louvain', dimred, None, None, knn_opt)
    box1 = ax[1].get_position()
    ax[1].set_position([box1.x0, box1.y0, box1.width * 0.9, box1.height])
    ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=6)
    #ax[1][0]= plot_onemethod(ax[1][0],X_embedded,dbscan_labels, true_label,onevsall_dbscan, 'dbscan', dimred,eps_opt, dbscan_min_clustersize, None)
    #ax[1][1]= plot_onemethod(ax[1][1],X_embedded,louvain_labels, true_label,onevsall_louvain, 'louvain', dimred,None, None, knn_opt)
    #ax[2][0]= plot_onemethod(ax[2][0],X_embedded,pheno_labels, true_label,onevsall_pheno, 'phenograph', dimred,None, None, 30)
    #ax[2][1]= plot_onemethod(ax[2][1],X_embedded,kmeans_labels, true_label,onevsall_kmeans, 'kmeans', dimred,None, None, None)



    plt.savefig(embedding_filename + '_' + dimred + '.png', bbox_inches='tight')


def plot_onemethod(ax, X_embedded, model_labels, true_labels, onevsall,method, dimred, sigma, min_cluster, knn_opt):

    # if one_vs_all == 'h2170': onevsall = 0
    # if one_vs_all == 'h1975': onevsall =1
    # if one_vs_all == 'h526': onevsall =2
    # if one_vs_all == 'h520': onevsall = 3
    # if one_vs_all == 'h358': onevsall = 4
    # if one_vs_all == 'h69': onevsall = 5
    # if one_vs_all == 'hcc827': onevsall = 6
    if onevsall  == 0: onevsall_opt='h2170'
    if onevsall == 1: onevsall_opt='h1975'
    if onevsall  == 2: onevsall_opt='h526'
    if onevsall == 3:onevsall_opt='h520'
    if onevsall  == 4:onevsall_opt= 'h358'
    if onevsall  == 5:onevsall_opt='h69'
    if onevsall  == 6:onevsall_opt='hcc827'

    X_dict = {}
    X_dict_true = {}

    Index_dict = {}

    X_plot = X_embedded

    mst_labels = model_labels
    num_groups = len(set(mst_labels))


    N = len(mst_labels)
    n_cancer = list(true_labels).count(onevsall)
    n_pbmc = N-n_cancer
    m = 999
    for k in range(N):
        x = X_plot[k, 0]
        y = X_plot[k, 1]
        X_dict.setdefault(mst_labels[k], []).append((x, y)) #coordinates of the points by mst groups
        Index_dict.setdefault(mst_labels[k], []).append(true_labels[k]) #true label kth data point grouped by mst_group
        X_dict_true.setdefault(true_labels[k],[]).append((x,y))
    sorted_keys = list(sorted(X_dict.keys()))
    print('in plot: number of distinct groups:', len(sorted_keys))
    # sorted_keys_dbscan =list(sorted(X_dict_dbscan.keys()))
    # print(sorted_keys, ' sorted keys')
    error_count = []
    h2170_labels = []
    h1975_labels = []
    h526_labels = []
    h520_labels = []
    h358_labels = []
    h69_labels = []
    hcc827_labels = []
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
        majority_val = func_mode(vals)
        if majority_val == onevsall:
            tp = tp + len([e for e in vals if e == onevsall])
            fp = fp + len([e for e in vals if e != onevsall])
            error_count.append(len([e for e in vals if e != majority_val]))
        else:
            fn = fn + len([e for e in vals if e == onevsall])
            tn = tn + len([e for e in vals if e != onevsall])
            error_count.append(len([e for e in vals if e != majority_val]))

        if (majority_val == 0):
            h2170_labels.append(kk)

        if (majority_val == 1):
            h1975_labels.append(kk)

        if majority_val == 2:
            h526_labels.append(kk)

        if majority_val == 3:
            h520_labels.append(kk)

        if (majority_val == 4):
            h358_labels.append(kk)

        if (majority_val == 5):
            h69_labels.append(kk)

        if majority_val == 6:
            hcc827_labels.append(kk)




    total_error_rate = sum(error_count)/N
    error_rate = (fp+fn)/(fp+fn+tn+tp)
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

    colors_h2170 = plt.cm.Greys_r(np.linspace(0.2, 0.6, len(h2170_labels)))
    colors_h1975 = plt.cm.Greens_r(np.linspace(0.2, 0.6, len(h1975_labels)))
    colors_h526 = plt.cm.Wistia_r(np.linspace(0.2, 0.6, len(h526_labels))) #orangey yellows
    colors_h520 = plt.cm.Reds_r(np.linspace(0.2, 0.4, len(h520_labels)))
    colors_h358 = plt.cm.Purples_r(np.linspace(0.2, 0.6, len(h358_labels)))
    colors_h69 = plt.cm.Blues_r(np.linspace(0.2, 0.6, len(h69_labels)))
    colors_hcc827 = plt.cm.spring(np.linspace(0, 0.4, len(hcc827_labels)))

    pair_color_group_list = [(colors_h2170, h2170_labels, ['h2170']*len(h2170_labels)),(colors_h1975,h1975_labels, ['h1975']*len(h1975_labels)),(colors_h526, h526_labels, ['h526']*len(h526_labels)),(colors_h520, h520_labels, ['h520']*len(h520_labels)),(colors_h358, h358_labels, ['h358']*len(h358_labels)),
                             (colors_h69, h69_labels, ['h69'] * len(h69_labels)),(colors_hcc827, hcc827_labels, ['hcc827'] * len(hcc827_labels))]
    for color, group, name in pair_color_group_list:
        for color_m, ll_m, label_m in zip(color, group,name):
            x = [t[0] for t in X_dict[ll_m]]
            y = [t[1] for t in X_dict[ll_m]]
            population = len(x)
            ax.scatter(x, y, color=color_m, s=2, alpha=0.6, label= label_m+'_' + str(ll_m) + ' Cellcount = ' + str(len(x)))
            ax.annotate(str(int(ll_m)), xytext=(np.mean(x), np.mean(y)), xy=(np.mean(x), np.mean(y)), color='black',
                           weight='semibold')

    ax.text(0.95, 0.01, "number of groups: " + " {:.0f}".format(num_groups) + " FP: " + " {:.2f}".format(
        fp * 100 / n_pbmc) + "%. FN of " + "{:.2f}".format(fn * 100 / (n_cancer)) + '%', transform=ax.transAxes,
               verticalalignment='bottom', horizontalalignment='right', color='green', fontsize=10)

    ax.axis('tight')

    if method == 'mst':
        title_str1 = 'MST on '+ dimred +' embedding: mean + ' + str(sigma) + '-sigma cutoff and min cluster size of: ' + str(
        min_cluster) + '\n' +"Total error rate: {:.1f}".format(total_error_rate * 100) + '%\n' + "One-vs-all for " +onevsall_opt+ " FP: " + " {:.1f}".format(
        fp * 100 / n_pbmc) + "%. FN of " + "{:.1f}".format(
        fn * 100 / (n_cancer)) + '%' + '\n' ' F1-score:' + "{:.2f} %".format(f1_score*100)
    if method == 'louvain':
        title_str1 = 'Shobi Louvain on ' +str(knn_opt)+'-NN graph clustering overlaid on. ' +dimred+ ' embedding. \n'+'Total error rate: {:.1f}'.format(total_error_rate * 100) + '% \n' + "One-vs-all for " +onevsall_opt+" FP: " + " {:.1f}".format(
        fp * 100 / n_pbmc) + "%. FN of " + "{:.1f}".format(
        fn * 100 / (n_cancer)) + '%' + ' F1-score:' + "{:.2f} %".format(f1_score*100)
    if method == 'phenograph':
        title_str1 = 'Phenograph on 30-NN graph clustering overlaid on. ' + dimred + ' embedding. \n'+'Total error rate: {:.1f}'.format(total_error_rate * 100) + '% \n'+ ' One-vs-all for ' + onevsall_opt+ ". FP: " + " {:.1f}".format(
            fp * 100 / n_pbmc) + "%. FN of " + "{:.1f}".format(
            fn * 100 / (n_cancer)) + "% F1-score: {:.2f} %".format(f1_score*100)
    if method == 'dbscan': title_str1 = 'DBSCAN on '+ dimred +' embedding .Eps = {:.2f}'.format(sigma) + ' and min cluster size of: ' + str(
        min_cluster) + '\n' + ". Total error rate: " + " {:.2f}".format(total_error_rate * 100) + '% \n' +  "One-vs-all for " +onevsall_opt+ "FP: " + " {:.1f}".format(
        fp * 100 / n_pbmc) + "%. FN of " + "{:.1f}".format(
        fn * 100 / (n_cancer)) + '% F1-score:' + "{:.2f} %".format(f1_score * 100)
    if method == 'kmeans': title_str1 = 'KMEANS on ' + dimred + ' embedding \n.'+ 'Total error rate:  {:.2f}'.format(
        total_error_rate * 100) + '% \n' + "One-vs-all for " + onevsall_opt + "FP: " + " {:.1f}".format(
        fp * 100 / n_pbmc) + "%. FN of " + "{:.1f}".format(
        fn * 100 / (n_cancer)) + '% F1-score:' + "{:.2f} %".format(f1_score * 100)

    ax.set_title(title_str1, size=8)
    return ax
    #plt.show()

def plot_mst_simple(model_labels, true_labels, embedding_filename, sigma, min_cluster, one_vs_all,X_embedded, method,knn_opt=None,dimred='dimred_method'):


    # http://nbviewer.jupyter.org/github/jakevdp/mst_clustering/blob/master/MSTClustering.ipynb

    if one_vs_all == 'h2170': onevsall = 0
    if one_vs_all == 'h1975': onevsall =1
    if one_vs_all == 'h526': onevsall =2
    if one_vs_all == 'h520': onevsall = 3
    if one_vs_all == 'h358': onevsall = 4
    if one_vs_all == 'h69': onevsall = 5
    if one_vs_all == 'hcc827': onevsall = 6

    X_dict = {}
    X_dict_true = {}

    Index_dict = {}

    X_plot = X_embedded

    mst_labels = model_labels
    num_groups = len(set(mst_labels))


    N = len(mst_labels)
    n_cancer = list(true_labels).count(onevsall)
    n_pbmc = N-n_cancer
    m = 999
    for k in range(N):
        x = X_plot[k, 0]
        y = X_plot[k, 1]
        X_dict.setdefault(mst_labels[k], []).append((x, y)) #coordinates of the points by mst groups
        Index_dict.setdefault(mst_labels[k], []).append(true_labels[k]) #true label kth data point grouped by mst_group
        X_dict_true.setdefault(true_labels[k],[]).append((x,y))
    sorted_keys = list(sorted(X_dict.keys()))
    print('in plot: number of distinct groups:', len(sorted_keys))
    # sorted_keys_dbscan =list(sorted(X_dict_dbscan.keys()))
    # print(sorted_keys, ' sorted keys')
    error_count = []
    monocytes_labels = [] #0,1,2,3 are the true labels of the subtypes in that order
    h2170_labels = []
    h1975_labels = []
    h526_labels = []
    h520_labels = []
    h358_labels = []
    h69_labels = []
    hcc827_labels = []
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
        majority_val = func_mode(vals)
        if majority_val == onevsall:
            tp = tp + len([e for e in vals if e == onevsall])
            fp = fp + len([e for e in vals if e != onevsall])
            error_count.append(len([e for e in vals if e != majority_val]))
        else:
            fn = fn + len([e for e in vals if e == onevsall])
            tn = tn + len([e for e in vals if e != onevsall])
            error_count.append(len([e for e in vals if e != majority_val]))

        if (majority_val == 0):
            h2170_labels.append(kk)

        if (majority_val == 1):
            h1975_labels.append(kk)

        if majority_val == 2:
            h526_labels.append(kk)

        if majority_val == 3:
            h520_labels.append(kk)

        if (majority_val == 4):
            h358_labels.append(kk)

        if (majority_val == 5):
            h69_labels.append(kk)

        if majority_val == 6:
            hcc827_labels.append(kk)

    error_rate = (fp+fn)/(fp+fn+tn+tp)
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

    fig, ax = plt.subplots(1, 2, figsize=(36, 12), sharex=True, sharey=True)
    #segments = model.get_graph_segments(full_graph=True)

    #ax[0].scatter(X_plot[:, 0], X_plot[:, 1], c=true_labels, cmap='nipy_spectral_r', zorder=2, alpha=0.5, s=4)
    for true_group in X_dict_true.keys():

        if true_group ==0:
            true_color = 'gray'
            true_label_str = 'h2170'
        if true_group ==1:
            true_color = 'forestgreen'
            true_label_str = 'h1975'
        if true_group ==2:
            true_color = 'orange'
            true_label_str = 'h526'
        if true_group == 3:
            true_color = 'red'
            true_label_str = 'h520'
        if true_group == 4:
            true_color = 'mediumpurple'
            true_label_str = 'h358'
        if true_group == 5:
            true_color = 'deepskyblue'
            true_label_str = 'h69'
        if true_group==6:
            true_color = 'lightpink'
            true_label_str = 'hcc827'
        print('true group', true_group, true_color, true_label_str)
        x = [t[0] for t in X_dict_true[true_group]]
        y = [t[1] for t in X_dict_true[true_group]]
        population = len(x)
        ax[0].scatter(x, y, color=true_color, s=2, alpha=1, label=true_label_str+' Cellcount = ' + str(population))
        ax[0].annotate(true_label_str, xytext=(np.mean(x), np.mean(y)), xy=(np.mean(x), np.mean(y)), color='black',
                       weight='semibold')

    colors_h2170 = plt.cm.Greys_r(np.linspace(0.1, 0.8, len(h2170_labels)))
    colors_h1975 = plt.cm.Greens_r(np.linspace(0.1, 0.8, len(h1975_labels)))
    colors_h526 = plt.cm.Wistia_r(np.linspace(0.1, 0.6, len(h526_labels))) #orangey yellows
    colors_h520 = plt.cm.Reds_r(np.linspace(0.1, 0.4, len(h520_labels)))
    colors_h358 = plt.cm.Purples_r(np.linspace(0.2, 0.8, len(h358_labels)))
    colors_h69 = plt.cm.Blues_r(np.linspace(0.2, 0.8, len(h69_labels)))
    colors_hcc827 = plt.cm.spring(np.linspace(0, 0.4, len(hcc827_labels)))

    pair_color_group_list = [(colors_h2170, h2170_labels, ['h2170']*len(h2170_labels)),(colors_h1975,h1975_labels, ['h1975']*len(h1975_labels)),(colors_h526, h526_labels, ['h526']*len(h526_labels)),(colors_h520, h520_labels, ['h520']*len(h520_labels)),(colors_h358, h358_labels, ['h358']*len(h358_labels)),
                             (colors_h69, h69_labels, ['h69'] * len(h69_labels)),(colors_hcc827, hcc827_labels, ['hcc827'] * len(hcc827_labels))]
    for color, group, name in pair_color_group_list:
        for color_m, ll_m, label_m in zip(color, group,name):
            x = [t[0] for t in X_dict[ll_m]]
            y = [t[1] for t in X_dict[ll_m]]
            population = len(x)
            ax[1].scatter(x, y, color=color_m, s=2, alpha=1, label= label_m+'_' + str(ll_m) + ' Cellcount = ' + str(len(x)))
            ax[1].annotate(str(int(ll_m)), xytext=(np.mean(x), np.mean(y)), xy=(np.mean(x), np.mean(y)), color='black',
                           weight='semibold')
            # ax[1].scatter(np.mean(x), np.mean(y),  color = color_p, s=population, alpha=1)
            #ax[2].annotate(str(ll_m) + '_n' + str(len(x)), xytext=(np.mean(x), np.mean(y)), xy=(np.mean(x), np.mean(y)),
                           #color='black', weight='semibold')
            #ax[2].scatter(np.mean(x), np.mean(y), color=color_m, s=3 * np.log(population), alpha=1, zorder=2)

    #ax[2].plot(segments[0], segments[1], '-k', zorder=1, linewidth=1)
    ax[1].text(0.95, 0.01, "number of groups: " + " {:.0f}".format(num_groups) + " FP: " + " {:.2f}".format(
        fp * 100 / n_pbmc) + "%. FN of " + "{:.2f}".format(fn * 100 / (n_cancer)) + '%', transform=ax[1].transAxes,
               verticalalignment='bottom', horizontalalignment='right', color='green', fontsize=10)

    ax[1].axis('tight')
    title_str0 = 'Multi-Class Lung Cancer Cell Lines" '#embedding_filename
    if method == 'mst':
        title_str1 = 'MST on '+ dimred +' embedding: mean + ' + str(sigma) + '-sigma cutoff and min cluster size of: ' + str(
        min_cluster) + '\n' + "One-vs-all for " +one_vs_all+" has error: " + " {:.1f}".format(error_rate * 100) + '%' + " FP: " + " {:.1f}".format(
        fp * 100 / n_pbmc) + "%. FN of " + "{:.1f}".format(
        fn * 100 / (n_cancer)) + '%' + '\n' + 'computed ratio:' + "{:.1f}".format(
        computed_ratio) + ' f1-score:' + "{:.1f}".format(f1_score)
    if method == 'louvain':
        title_str1 = 'Louvain on ' +str(knn_opt)+'-NN graph clustering overlaid on. ' +dimred+ 'emdbedding. \n'' One-vs-all for ' +one_vs_all+". Has error: " + " {:.1f}".format(error_rate * 100) + '%' + " FP: " + " {:.1f}".format(
        fp * 100 / n_pbmc) + "%. FN of " + "{:.1f}".format(
        fn * 100 / (n_cancer)) + '%' + '\n' + 'computed ratio:' + "{:.1f}".format(
        computed_ratio) + ' f1-score:' + "{:.1f}".format(f1_score)
    title_str2 = 'graph layout with cluster populations'

    ax[1].set_title(title_str1, size=10)
    ax[0].set_title(title_str0, size=10)
    #ax[2].set_title(title_str2, size=12)

    box = ax[0].get_position()
    ax[0].set_position([box.x0, box.y0, box.width *0.9, box.height])
    ax[0].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize = 6)
    plt.savefig(embedding_filename+'_' +method+ '_'+dimred+'.png', bbox_inches='tight')
    #plt.show()
def multiclass_mst_accuracy(X_embedded, true_label,df_all,av_peaks=0, too_big_factor = 0.20): #use 20 for multiclass and 5 for rare
    #fig_density, ax_density = plt.subplots(1, 2, figsize=(36, 12))

    if av_peaks == 0:
        xy_peaks  =pp.tsne_densityplot(X_embedded[:,:2], ax1 = None, ax2= None, df_all=df_all, mode='vals only')
        #plt.show()
        XZ=X_embedded[:, np.ix_([0,2])]
        XZ = XZ[:,0,:]
        YZ =X_embedded[:, np.ix_([1,2])]
        YZ = YZ[:, 0, :]
        print('xz and yz shape',XZ.shape, YZ.shape)
        xz_peaks = pp.tsne_densityplot(XZ, None, None, df_all, mode='only_vals')
        yz_peaks = pp.tsne_densityplot(YZ, None, None, df_all, mode='only_vals')
        av_peaks = round((xy_peaks.shape[0]+yz_peaks.shape[0]+xz_peaks.shape[0])/3) +10 #on sep 13 i added +10
        print('no. peaks,', xy_peaks.shape,xz_peaks.shape,yz_peaks.shape, av_peaks)

    #av_peaks = 15
    min_clustersize = [20, 10]#[50,30,20,10]
    f1_temp = -1
    f1_sum_best = 0
    list_roc = []
    targets = list(set(true_label))
    if len(targets) > 0:
        target_range = targets
    else:
        target_range = [1]
    for i_min_clustersize in min_clustersize:
        model = MSTClustering3D(approximate=True, min_cluster_size=i_min_clustersize, max_labels=av_peaks, true_label=true_label)
        #model = MSTClustering()
        time_start = time.time()
        print('Starting Clustering', time.ctime())
        model.fit_predict(X_embedded)
        clustering_labels = model.labels_
        too_big = False
        for cluster_i in set(clustering_labels):
            cluster_i_loc = np.where(np.asarray(clustering_labels) == cluster_i)[0]
            pop_i = len(cluster_i_loc)
            if pop_i > too_big_factor * len(true_label):
                too_big = True
                cluster_big_loc = cluster_i_loc
        list_pop_too_bigs = [pop_i]
        while too_big == True:

            print('removing too big')
            X_data_big = X_embedded[cluster_big_loc, :]
            pop_too_big_cur = X_data_big.shape[0]
            #knn_struct_big = ls.make_knn_struct(X_data_big)
            print(X_data_big.shape)
            #louvain_labels_big = ls.run_toobig_sublouvain(X_data_big, knn_struct_big, k_nn=50, self_loop=False)
            model_big = MSTClustering3D(min_cluster_size=20, approximate=True, n_neighbors=30,max_labels=7)
            model_big.fit_predict(X_data_big)
            louvain_labels_big = model_big.labels_
            louvain_labels_big = np.asarray(louvain_labels_big)
            print('set of new big labels ', set(list(louvain_labels_big.flatten())))
            louvain_labels_big = louvain_labels_big + 1000  # len(set_louvain_labels)
            print('set of new big labels +1000 ', set(list(louvain_labels_big.flatten())))
            pop_list = []
            for item in set(list(louvain_labels_big.flatten())):
                pop_list.append(list(louvain_labels_big.flatten()).count(item))
            print('pop of big list', pop_list)
            jj = 0
            clustering_labels = np.asarray(clustering_labels)
            for j in cluster_big_loc:
                clustering_labels[j] = louvain_labels_big[jj]
                jj = jj + 1
            dummy, clustering_labels = np.unique(list(clustering_labels.flatten()), return_inverse=True)
            print('new set of labels ', set(list(clustering_labels.flatten())))
            too_big = False
            set_clustering_labels = set(list(clustering_labels.flatten()))
            for cluster_i in set_clustering_labels:
                cluster_i_loc = np.where(clustering_labels == cluster_i)[0]
                pop_i = len(cluster_i_loc)
                not_already_expanded = pop_i not in list_pop_too_bigs
                if pop_i > too_big_factor * len(true_label) and not_already_expanded ==True:
                    too_big = True
                    print('cluster', cluster_i, 'is too big with population', pop_i)
                    cluster_big_loc = cluster_i_loc
                    cluster_big = cluster_i
                    big_pop = pop_i
            if too_big == True:
                list_pop_too_bigs.append(big_pop)
                print('cluster', cluster_big, 'is too big with population', big_pop, 'and will be expanded')
        print('final shape before too_small allocation', set(list(clustering_labels.flatten())))
        clustering_labels = list(clustering_labels.flatten())

        runtime_mst = time.time() - time_start
        onevsall_str = 'placeholder'
        f1_sum = 0
        f1_accumulated=0
        for onevsall_val in target_range:
            '''
            if onevsall_val == 0: onevsall_str = 'h2170'
            if onevsall_val == 1: onevsall_str = 'h1975'
            if onevsall_val == 2: onevsall_str = 'h526'
            if onevsall_val == 3: onevsall_str = 'h520'
            if onevsall_val == 4: onevsall_str = 'h358'
            if onevsall_val == 5: onevsall_str = 'h69'
            if onevsall_val == 6: onevsall_str = 'hcc827'
            '''
            vals_roc, predict_class_array, majority_truth_labels = accuracy_mst(clustering_labels, true_label,
                                                         embedding_filename=None, clustering_algo='multiclass mst',
                                                         onevsall=onevsall_val)

            if vals_roc[1] > f1_temp:
                f1_temp = vals_roc[1]
                onevsall_val_opt = onevsall_val
            f1_sum = f1_sum + vals_roc[1]
            f1_current = vals_roc[1]
            f1_accumulated = f1_accumulated + f1_current * (list(true_label).count(onevsall_val)) / len(true_label)
            list_roc.append([model.sigma_factor, i_min_clustersize, model.tooclosefactor, onevsall_val, onevsall_str] + vals_roc + [runtime_mst])
        print('ARI APT',
              adjusted_rand_score(np.asarray(true_label), np.asarray(clustering_labels)),
              metrics.adjusted_mutual_info_score(true_label, clustering_labels))
        print("f1-score weighted", f1_accumulated)

        if f1_sum > f1_sum_best:
            f1_sum_best = f1_sum
            temp_best_labels = clustering_labels
            sigma_opt = model.sigma_factor
            tooclose_factor_opt = model.tooclosefactor
            onevsall_best = onevsall_val_opt
            min_clustersize_opt = i_min_clustersize
    majority_truth_labels = np.zeros(len(true_label))

    df_accuracy = pd.DataFrame(list_roc,
                           columns=['sigma factor', 'min cluster size','merge-too-close factor', 'onevsall target','cell type','error rate', 'f1-score', 'tnr', 'fnr',
                                    'tpr', 'fpr', 'precision', 'recall', 'num_groups', 'runtime'])

    return df_accuracy, temp_best_labels, sigma_opt, min_clustersize_opt, tooclose_factor_opt, onevsall_best, majority_truth_labels


def run_dbscan(X_embedded, true_label):
    # note: if the eps value goes below 0.3 for larger cell counts (n>100000), then the number of big-groups exceeds 1500 and merging is too slow
    from sklearn.neighbors import kneighbors_graph
    from scipy.sparse.csgraph import minimum_spanning_tree
    G = kneighbors_graph(X_embedded, n_neighbors=20,
                         mode='distance',
                         metric="euclidean")
    mst_tree_ = minimum_spanning_tree(G, overwrite=True)
    d = mst_tree_.data
    mu_edge = np.mean(d)
    std_edge = np.std(d)
    print('mean edge and std are ', mu_edge, std_edge)
    f1_temp = -1
    f1_sum_best = 0
    list_roc = []
    sigma_list = [0.5]  # eps=0.5 is the default, we are setting eps as max of 0.5 vs. mu+2.5*std of edge distance in mst_graph of embedding
    tooclosefactor_list = [0]
    i_cluster_size = 20

    targets = list(set(true_label))
    if len(targets) > 2:
        target_range = targets
    else:
        target_range = [1]
    for i_sigma in sigma_list:
        for i_tooclosefactor in tooclosefactor_list:
            eps_dbscan = max(i_sigma, mu_edge + 3 * std_edge)
            print('Starting DBSCAN with eps of ', eps_dbscan, time.ctime())
            time_start = time.time()
            model = DBSCAN(eps=eps_dbscan, min_samples=i_cluster_size, tooclose_factor=i_tooclosefactor).fit(X_embedded)
            mst_runtime = time.time() - time_start
            f1_sum= 0
            for onevsall_val in target_range:
                vals_roc, predict_class_array = accuracy_mst(model, true_label,
                                                         embedding_filename=None, clustering_algo='dbscan',onevsall=onevsall_val)
                vals_roc = [onevsall_val]+vals_roc
                list_roc.append(vals_roc)
                f1_sum = f1_sum+ vals_roc[5]
                print('f1 score in dbscan',vals_roc[5])
                if vals_roc[5] > f1_temp:
                    f1_temp = vals_roc[5]
                    onevsall_val_opt = onevsall_val
            if f1_sum > f1_sum_best:
                f1_sum_best = f1_sum
                temp_best_labels = list(model.labels_)
                if i_tooclosefactor == 20:
                    temp_best_labels_tooclose20 = list(model.labels_)
                    eps_opt_tooclose20 = eps_dbscan
                    onevsall_opt_tooclose20 = onevsall_val_opt
                eps_opt = eps_dbscan
                tooclose_factor_opt = i_tooclosefactor
                onevsall_best = onevsall_val_opt
    df_accuracy = pd.DataFrame(list_roc,
                               columns=['one-vs-all', 'embedding_filename','eps', 'min cluster size', 'merge-too-close factor','error rate',
                                        'f1-score', 'tnr', 'fnr',
                                        'tpr', 'fpr', 'precision', 'recall', 'num_groups', 'clustering runtime'])
    return df_accuracy, temp_best_labels, eps_opt, i_cluster_size, tooclose_factor_opt, onevsall_best

def run_kmeans(X_embedded, true_label,df_all):

    list_roc = []
    targets = list(set(true_label))
    if len(targets) > 2:
        target_range = targets
    else:
        target_range = [1]
    f1_temp=-1
    init_peaks = pp.tsne_densityplot(X_embedded, None, None, df_all, mode='only_vals')
    n_clusters = init_peaks.shape[0]
    #n_jobs = -3 means all but two CPUs is used for this task

    time_start = time.time()
    model = KMeans(n_clusters=n_clusters, init=init_peaks, n_init=1,max_iter=1, verbose = 1, n_jobs=-3).fit(X_embedded)
    runtime = time.time() - time_start

    for onevsall_val in target_range:
        vals_roc, predict_class_array = accuracy_mst(model, true_label,
                                                     embedding_filename=None, clustering_algo='kmeans',onevsall=onevsall_val)
        vals_roc = [onevsall_val]+vals_roc+[runtime]
        list_roc.append(vals_roc)

        if vals_roc[2] > f1_temp:
            f1_temp = vals_roc[2]
            temp_best_labels = list(model.labels_)
            onevsall_val_opt = onevsall_val

    df_accuracy = pd.DataFrame(list_roc,
                       columns=['target','error rate','f1-score', 'tnr', 'fnr',
                                'tpr', 'fpr', 'precision', 'recall', 'num_groups', 'clustering runtime'])

    return df_accuracy, temp_best_labels,onevsall_val_opt
def get_SampleImageIDs(cluster_labels, df_all, true_labels):
    import random
    cluster_i_tag_list = []
    for cluster_i in set(cluster_labels):
        cluster_i_loc = np.where(cluster_labels == cluster_i)[0]
        population_cluster_i = len(cluster_i_loc)
        majority_truth = func_mode(list(true_labels[cluster_i_loc]))
        random.shuffle(cluster_i_loc)
        cluster_i_loc_20= cluster_i_loc[0:20]
        #if population_cluster_i>200:
        if population_cluster_i > 20:
            for k in cluster_i_loc_20:
                cluster_i_tag_list.append([cluster_i,majority_truth,df_all.loc[k, 'label'], df_all.loc[k, 'cell_filename'], df_all.loc[k, 'cell_idx_inmatfile'],
                       df_all.loc[k, 'File ID'], df_all.loc[k, 'Cell ID'], df_all.loc[k, 'index']])
    column_names_tags = ['cluster','majority truth','celltype', 'filename', 'idx_inmatfile','File ID', 'Cell ID','df_all idx']
    df_sample_imagelist = pd.concat([pd.DataFrame([i], columns=column_names_tags) for i in cluster_i_tag_list], ignore_index=True)
    df_sample_imagelist = df_sample_imagelist.sort_values(['majority truth','cluster'])
    return df_sample_imagelist
def run_main(new_file_name, n_eachsubtype= None, randomseedval=1):
    df_all, true_label, X_txt, feat_cols = get_data(fluor=0, n_eachsubtype=n_eachsubtype, randomseedval=randomseedval)


    n_total = X_txt.shape[0]
    frames = None
    perplexity_range = [50]#,10,30,70,90]
    for perplexity in perplexity_range:
        excel_file_name = '/home/shobi/Thesis/MultiClass_MinCluster/ '+new_file_name+'_N'+str(n_total)+'_perp'+str(perplexity)+'.xlsx'
        excel_file_name_mst = '/home/shobi/Thesis/MultiClass_MinCluster/ ' + new_file_name + '_N' + str(
            n_total) + '_perp' + str(perplexity) + 'mst.xlsx'
        plot_name = '/home/shobi/Thesis/MultiClass_MinCluster/'+new_file_name+'_N'+str(n_total)+'_perp'+str(perplexity)
        print(plot_name)

        writer = ExcelWriter(excel_file_name)
        writer_mst = ExcelWriter(excel_file_name_mst)
        if perplexity == perplexity_range[0]:
            predict_class_aggregate_louvain, df_accuracy_louvain, best_louvain_labels, knn_opt, onevsall_opt_louvain, majority_vals = ls.run_mainlouvain(X_txt, true_label, self_loop=False)
            majority_truth_labels_louvain = np.zeros(len(true_label))
            for cluster_i in set(best_louvain_labels):
                #print('clusteri', cluster_i)
                cluster_i_loc = np.where(best_louvain_labels == cluster_i)[0]
                #print('loc',cluster_i_loc)
                majority_truth = func_mode(list(true_label[cluster_i_loc]))
                #print(majority_truth)
                majority_truth_labels_louvain[cluster_i_loc] = majority_truth


            df_all['majority_vote_class_louvain'] = majority_truth_labels_louvain
            df_all['cluster_louvain'] = best_louvain_labels
            print(best_louvain_labels)
            df_all_heatmap_louvain = df_all.sort_values(['majority_vote_class_louvain', 'cluster_louvain'])
            feat_cols.remove('Dry mass var')
            df_heatmap_louvain =df_all_heatmap_louvain[feat_cols]
            print(feat_cols)
            print(df_heatmap_louvain.max())
            fig, (ax0,ax1) = plt.subplots(1, 2, figsize=(36, 12))
            ax0.pcolor(df_heatmap_louvain,vmin=-4, vmax=4)
            ax0.set_xticks(np.arange(0.5, len(df_heatmap_louvain.columns), 1))
            ax0.set_xticklabels(df_heatmap_louvain.columns)
            #ax0.set_yticks(np.arange(0.5, len(df_heatmap_louvain.index), 1))
            #ax0.set_yticklabels(df_all_heatmap_louvain['cluster_louvain'].values[0::200]
            ylist = df_all_heatmap_louvain['cluster_louvain'].values
            ylist_majority = df_all_heatmap_louvain['majority_vote_class_louvain'].values
            ynewlist = []
            maj = ylist_majority[0]
            if maj == 0: maj_str = 'h2170'
            if maj == 1: maj_str = 'h1975'
            if maj == 2: maj_str = 'h526'
            if maj == 3: maj_str = 'h520'
            if maj == 4: maj_str = 'h358'
            if maj == 5: maj_str = 'h69'
            if maj == 6: maj_str = 'hcc827'
            ynewlist.append('cluster ' + str(int(ylist[0])) + ' ' + maj_str)
            ytickloc =[0]
            for i in range(len(ylist)-1):
                #if ylist[i+1] == ylist[i]: ynewlist.append('')
                if ylist[i+1] != ylist[i]:
                    maj = ylist_majority[i+1]
                    if maj == 0: maj_str = 'h2170'
                    if maj == 1: maj_str = 'h1975'
                    if maj == 2: maj_str = 'h526'
                    if maj == 3: maj_str = 'h520'
                    if maj == 4: maj_str = 'h358'
                    if maj == 5: maj_str = 'h69'
                    if maj == 6: maj_str = 'hcc827'
                    ynewlist.append('cluster '+str(int(ylist[i + 1]))+' '+maj_str)
                    ytickloc.append(int(i + 1))
            ax0.set_yticks(ytickloc)
            ax0.set_yticklabels(ynewlist)
            ax0.grid(axis ='y',color="w", linestyle='-', linewidth=2)
            #ax0.set_yticklabels(np.arange(0.5, len(df_heatmap_louvain.index), 100), df_all_heatmap_louvain['cluster_louvain'].values[0::100])
            #ax0.set_xticklabels(np.arange(0.5, len(df_heatmap_louvain.columns), 1), df_heatmap_louvain.columns)
            ax0.set_title('ALPH Heatmap: cell level')

            ax0.tick_params(axis='x', rotation = 45)
            #[l.set_visible(False) for (i, l) in enumerate(plt.xticks()) if i % nn != 0]
            #plt.locator_params(axis='y', nbins=10)

            df_all_mean_clusters_louvain = df_all_heatmap_louvain.groupby('cluster_louvain',as_index=False)[feat_cols+['majority_vote_class_louvain']].mean()
            df_all_mean_clusters_louvain = df_all_mean_clusters_louvain.sort_values(['majority_vote_class_louvain', 'cluster_louvain'])
            df_mean_clusters_louvain = df_all_mean_clusters_louvain[feat_cols]
            ax1.pcolor(df_mean_clusters_louvain, vmin=-4, vmax=4)
            ax1.set_xticks(np.arange(0.5, len(df_heatmap_louvain.columns), 1))
            ax1.set_xticklabels(df_heatmap_louvain.columns)
            ax1.set_yticks(np.arange(0, len(df_all_mean_clusters_louvain.index), 1))
            ax1.set_yticklabels(ynewlist)
            ax1.grid(axis = 'y', color="w", linestyle='-', linewidth=2)
            #ax1.set_yticklabels(df_all_mean_clusters_louvain['cluster_louvain'])
            #ax1.set_yticklabels(np.arange(0.5, len(df_all_mean_clusters_louvain), 1),df_all_mean_clusters_louvain['cluster_louvain'])
            #ax1.set_xticklabels(np.arange(0.5, len(df_mean_clusters_louvain.columns), 1), df_mean_clusters_louvain.columns)
            ax1.set_title('ALPH Heatmap: cluster level')
            ax1.tick_params(axis='x', rotation=45)
            df_sample_imagelist = get_SampleImageIDs(best_louvain_labels, df_all,true_label)
            df_accuracy_louvain.to_excel(writer, 'louvain', index=False)
            df_sample_imagelist.to_excel(writer, 'louvain_images', index=False)

            plt.savefig(plot_name + 'louvain_heatmap.png')
            writer.save()

        time_start = time.time()

        params_lv = 'lr =1, perp = ' + str(perplexity)
        X_LV_embedded, lv_embedding_filename, lv_runtime, lv_embedding_plot_title = pp.run_lv(version=None,
                                                                                              input_data=X_txt,
                                                                                              perplexity=perplexity,
                                                                                              lr=1,
                                                                                              new_file_name=new_file_name,
                                                                                              new_folder_name=None,outdim=3)
        from scipy import stats
        X_LV_embedded= stats.zscore(X_LV_embedded, axis =0)
        lv_runtime = time.time() - time_start


        #df_accuracy_kmeans_lv, best_labels_kmeans_lv, onevsall_opt_kmeans_lv = run_kmeans(X_LV_embedded[:,:2], true_label,df_all)
        #df_accuracy_kmeans_lv.to_excel(writer, 'kmeans_lv', index=False)
        df_accuracy_mst_lv, best_labels_mst_lv, sigma_opt_lv, min_clustersize_mst_lv, tooclose_factor_opt, onevsall_opt_mst_lv,majority_truth_labels_mst_lv = multiclass_mst_accuracy(
            X_LV_embedded, true_label,df_all)
        df_accuracy_mst_lv.to_excel(writer, 'mst_lv', index=False)
        df_sample_imagelist = get_SampleImageIDs(best_labels_mst_lv, df_all,true_label)
        df_sample_imagelist.to_excel(writer, 'mst_images', index=False)
        df_sample_imagelist.to_excel(writer_mst, 'mst_images', index=False)


        dict_time = {'lv runtime': [lv_runtime],
                     'lv params': [params_lv]}  # , ' bh runtime': [tsne_runtime],  'bh params': [params_tsne]}
        df_time = pd.DataFrame(dict_time)
        df_time.to_excel(writer, 'embedding time', index=False)
        writer.save()
        print('successfully saved excel files')
        if perplexity == perplexity_range[0]:
            fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(36, 12))
            df_all['majority_vote_class_APT_p50'] = majority_truth_labels_mst_lv
            df_all['cluster_APT_p50'] = best_labels_mst_lv
            df_all_heatmap_APT = df_all.sort_values(['majority_vote_class_APT_p50', 'cluster_APT_p50'])
            df_heatmap_APT = df_all_heatmap_APT[feat_cols]
            ax0.pcolor(df_heatmap_APT,vmin=-4, vmax=4)
            ylist = df_all_heatmap_APT['cluster_APT_p50'].values
            ynewlist = []
            maj = ylist_majority[0]
            if maj == 0: maj_str = 'h2170'
            if maj == 1: maj_str = 'h1975'
            if maj == 2: maj_str = 'h526'
            if maj == 3: maj_str = 'h520'
            if maj == 4: maj_str = 'h358'
            if maj == 5: maj_str = 'h69'
            if maj == 6: maj_str = 'hcc827'
            ynewlist.append('cluster ' + str(int(ylist[0])) + ' ' + maj_str)
            ytickloc = [0]
            for i in range(len(ylist) - 1):
                # if ylist[i+1] == ylist[i]: ynewlist.append('')
                if ylist[i + 1] != ylist[i]:
                    maj = ylist_majority[i + 1]
                    if maj == 0: maj_str = 'h2170'
                    elif maj == 1: maj_str = 'h1975'
                    elif maj == 2: maj_str = 'h526'
                    elif maj == 3: maj_str = 'h520'
                    elif maj == 4: maj_str = 'h358'
                    elif maj == 5: maj_str = 'h69'
                    elif maj == 6: maj_str = 'hcc827'
                    else: print('no matching majority val')
                    ynewlist.append('cluster ' + str(int(ylist[i + 1])) + ' ' + maj_str)
                    ytickloc.append(int(i + 1))
            ax0.set_yticks(ytickloc)
            ax0.set_yticklabels(ynewlist)
            #ax0.set_yticks(np.arange(0.5, len(df_all_heatmap_APT.index), 200))
            #ax0.set_yticklabels(df_all_heatmap_APT['cluster_APT_p50'].values[0::200])
            ax0.set_xticklabels(df_heatmap_APT.columns)
            ax0.set_xticks(np.arange(0.5, len(df_heatmap_APT.columns), 1))
            '''
            ax0.set_yticklabels(np.arange(0.5, len(df_all_heatmap_APT.index), 100), df_all_heatmap_APT['cluster_APT_p50'].values[0::100])
            ax0.set_xticklabels(np.arange(0.5, len(df_heatmap_APT.columns), 1), df_heatmap_APT.columns)
            '''
            ax0.set_title('APT Heatmap: cell level')
            ax0.tick_params(axis='x', rotation=45)
            ax0.grid(axis = 'y', color="w", linestyle='-', linewidth=2)
            df_all_mean_clusters_APT = df_all_heatmap_APT.groupby('cluster_APT_p50', as_index=False)[feat_cols+['majority_vote_class_APT_p50']].mean()
            df_all_mean_clusters_APT = df_all_mean_clusters_APT.sort_values(['majority_vote_class_APT_p50', 'cluster_APT_p50'])
            df_mean_clusters_APT = df_all_mean_clusters_APT[feat_cols]

            ax1.pcolor(df_mean_clusters_APT, vmin=-4, vmax=4)

            ax1.set_yticks(np.arange(0, len(df_all_mean_clusters_APT), 1))
            ax1.set_yticklabels(ynewlist)

            ax1.set_xticks(np.arange(0.5, len(df_mean_clusters_APT.columns), 1))
            ax1.set_xticklabels(df_mean_clusters_APT.columns)

            '''
            ax1.set_yticklabels(np.arange(0.5, len(df_all_mean_clusters_APT), 1),df_all_mean_clusters_APT['cluster_APT_p50'])
            ax1.set_xticklabels(np.arange(0.5, len(df_mean_clusters_APT.columns), 1), df_mean_clusters_APT.columns)
            '''
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(axis = 'y', color="w", linestyle='-', linewidth=2)
            ax1.set_title('APT Heatmap: cluster level')

            plt.savefig(plot_name + 'APT_heatmap.png')



        #df_accuracy_dbscan_lv, dbscan_best_labels_lv, eps_opt_lv, dbscan_min_clustersize_lv, tooclose_factor_opt_lv, onevsall_opt_dbscan_lv = run_dbscan(
        #    X_LV_embedded, true_label)
        #df_accuracy_dbscan_lv.to_excel(writer, 'dbscan_lv', index=False)
        '''
        plot_all_methods(X_LV_embedded, true_label, embedding_filename=plot_name+'all', dbscan_labels=dbscan_best_labels_lv,
                         mst_labels=best_labels_mst_lv, louvain_labels=best_louvain_labels,
                         pheno_labels=None, kmeans_labels=best_labels_kmeans_lv, onevsall_mst=onevsall_opt_mst_lv,
                         onevsall_dbscan=onevsall_opt_dbscan_lv, onevsall_louvain=onevsall_opt_louvain,
                         onevsall_pheno=None,
                         onevsall_kmeans=onevsall_opt_kmeans_lv, dimred='lv', sigma_opt=sigma_opt_lv,
                         eps_opt=eps_opt_lv, min_cluster_opt=min_clustersize_mst_lv,
                         dbscan_min_clustersize=dbscan_min_clustersize_lv,
                         knn_opt=knn_opt)
        '''

        plot_all_methods(X_LV_embedded, true_label, embedding_filename=plot_name+'APT', dbscan_labels=None,
                         mst_labels=best_labels_mst_lv, louvain_labels=None,
                         pheno_labels=None, kmeans_labels=None, onevsall_mst=onevsall_opt_mst_lv,
                         onevsall_dbscan=None, onevsall_louvain=None,
                         onevsall_pheno=None,
                         onevsall_kmeans=None, dimred='lv', sigma_opt=sigma_opt_lv,
                         eps_opt=None, min_cluster_opt=min_clustersize_mst_lv,
                         dbscan_min_clustersize=None,
                         knn_opt=None)
        if perplexity == perplexity_range[0]:
            plot_all_methods(X_LV_embedded, true_label, embedding_filename=plot_name + '_ALPH', dbscan_labels=None,
                             mst_labels=None, louvain_labels=best_louvain_labels,
                             pheno_labels=None, kmeans_labels=None, onevsall_mst=onevsall_opt_mst_lv,
                             onevsall_dbscan=None, onevsall_louvain=onevsall_opt_louvain,
                             onevsall_pheno=None,
                             onevsall_kmeans=None, dimred='lv', sigma_opt=None, eps_opt=None, min_cluster_opt=None,
                             dbscan_min_clustersize=None,
                             knn_opt=knn_opt)

        Plotting_3D.save_anim('APT',X_LV_embedded, true_label, embedding_filename=plot_name, dbscan_labels=None,
                         mst_labels=best_labels_mst_lv, louvain_labels=None,
                         pheno_labels=None, kmeans_labels=None, onevsall_mst=onevsall_opt_mst_lv,
                         onevsall_dbscan=None, onevsall_louvain=None,
                         onevsall_pheno=None,
                         onevsall_kmeans=None, dimred='lv', sigma_opt=sigma_opt_lv,
                         eps_opt=None, min_cluster_opt=min_clustersize_mst_lv,
                         dbscan_min_clustersize=None,
                         knn_opt=None)
        if perplexity == perplexity_range[0]:
            Plotting_3D.save_anim('ALPH',X_LV_embedded, true_label, embedding_filename=plot_name+'ALPH', dbscan_labels=None,
                             mst_labels=None, louvain_labels=best_louvain_labels,
                             pheno_labels=None, kmeans_labels=None, onevsall_mst=onevsall_opt_mst_lv,
                             onevsall_dbscan=None, onevsall_louvain=onevsall_opt_louvain,
                             onevsall_pheno=None,
                             onevsall_kmeans=None, dimred='lv', sigma_opt=None, eps_opt=None, min_cluster_opt=None,
                             dbscan_min_clustersize=None,
                             knn_opt=knn_opt)

        '''
        time_start = time.time()
        print('starting tsne', time.ctime())
        learning_rate_bh =2000
        if n_total >500000: learning_rate_bh = 2500
        if n_total > 1000000: learning_rate_bh = 3500
        params_tsne = 'n_jobs=8, perplexity = ' + str(perplexity) + ' ,verbose=1,n_iter=1000,learning_rate =' + str(learning_rate_bh)
        tsne = multicore_tsne(n_jobs=8, perplexity=perplexity, verbose=1, n_iter=1000, learning_rate=learning_rate_bh, angle=0.2)

        X_embedded = tsne.fit_transform(X_txt)
        print(X_embedded.shape)
        tsne_runtime = time.time() - time_start
        print(params_tsne, new_file_name, '\n',' BH done! Time elapsed: {} seconds'.format(tsne_runtime))
        df_accuracy_kmeans, temp_best_labels_kmeans, onevsall_opt_kmeans = run_kmeans(X_embedded, true_label, df_all)

        df_accuracy_mst_bh, temp_best_labels_mst, sigma_opt, min_clustersize, tooclose_factor_opt,onevsall_opt_mst= multiclass_mst_accuracy(X_embedded, true_label)
        df_accuracy_mst_bh.to_excel(writer, 'mst_bh', index=False)
        df_accuracy_dbscan, dbscan_best_labels, eps_opt, dbscan_min_clustersize, tooclose_factor_opt,onevsall_opt_dbscan = run_dbscan(X_embedded, true_label)
        df_accuracy_dbscan.to_excel(writer, 'dbscan_bh', index=False)
        df_accuracy_kmeans.to_excel(writer, 'kmeans_bh', index=False)

        plot_all_methods(X_embedded, true_label, embedding_filename=plot_name, dbscan_labels=dbscan_best_labels, mst_labels=temp_best_labels_mst, louvain_labels=best_louvain_labels,
                         pheno_labels=None, kmeans_labels = temp_best_labels_kmeans, onevsall_mst=onevsall_opt_mst, onevsall_dbscan=onevsall_opt_dbscan,onevsall_louvain=onevsall_opt_louvain,onevsall_pheno= None, onevsall_kmeans = onevsall_opt_kmeans,dimred='bh', sigma_opt= sigma_opt, eps_opt = eps_opt, min_cluster_opt = min_clustersize,dbscan_min_clustersize = dbscan_min_clustersize, knn_opt=knn_opt)
        '''
        '''
        plot_all_methods(X_embedded, true_label, embedding_filename=plot_name+'3sigma_mst', dbscan_labels=dbscan_best_labels_tooclose20,
                         mst_labels=temp_best_labels_mst_s3, louvain_labels=best_louvain_labels,
                         pheno_labels=None, kmeans_labels=temp_best_labels_kmeans,
                         onevsall_mst=onevsall_best_mst_s3, onevsall_dbscan=onevsall_opt_dbscan_tooclose20,
                         onevsall_louvain=onevsall_opt_louvain, onevsall_pheno=None,
                         onevsall_kmeans=onevsall_opt_kmeans, dimred='bh', sigma_opt=sigma_opt_s3, eps_opt=eps_opt_tooclose20,
                         min_cluster_opt=min_clustersize_s3, dbscan_min_clustersize=dbscan_min_clustersize, knn_opt=knn_opt)
        '''
        #predict_class_aggregate_pheno, df_accuracy_pheno, best_pheno_labels, onevsall_opt_pheno= ls.run_phenograph(X_txt,true_label)
        #df_accuracy_pheno.to_excel(writer, 'pheno', index=False)

        '''
        plot_all_methods(X_LV_embedded, true_label, embedding_filename=plot_name + '3sigma_mst',
                         dbscan_labels=dbscan_best_labels_tooclose20_lv,
                         mst_labels=temp_best_labels_mst_lv_s3, louvain_labels=best_louvain_labels,
                         pheno_labels=None, kmeans_labels=best_labels_kmeans_lv, onevsall_mst=onevsall_best_mst_lv_s3,
                         onevsall_dbscan=onevsall_opt_dbscan_tooclose20_lv, onevsall_louvain=onevsall_opt_louvain,
                         onevsall_pheno=None, onevsall_kmeans=onevsall_opt_kmeans_lv, dimred='lv', sigma_opt=sigma_opt_lv_s3,
                         eps_opt=eps_opt_tooclose20_lv, min_cluster_opt=min_clustersize_mst_lv_s3,
                         dbscan_min_clustersize=dbscan_min_clustersize_lv,
                         knn_opt=knn_opt)
        '''





def main():
    import random
    print('time now is', time.ctime())
    randomseedval = 1
    print('file randomseed val: ', randomseedval)
    #run_main('LCJAN_avpeaks_Jul19_4pm_Randomseed'+str(randomseedval)+'_', n_eachsubtype =15000, randomseedval = randomseedval)
    #run_main('LCJAN_avpeaks_Jul19_4pm_Randomseed'+str(randomseedval)+'_', n_eachsubtype =50000, randomseedval = randomseedval)
    #run_main('LCJAN_avpeaks_Jul19_4pm_Randomseed' + str(randomseedval) + '_', n_eachsubtype=None, randomseedval=randomseedval)
    run_main('LCJAN_3D_20Sep'+'_', n_eachsubtype=5000, randomseedval=randomseedval)
    #run_main('LC_Jul17_1am_Randomseed' + str(randomseedval) + '_', n_eachsubtype=150000, randomseedval=randomseedval)

if __name__ == '__main__':
    main()