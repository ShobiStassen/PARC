'''
Latest version of Lung Cancer classifier using ALPHA and APT.
Can choose between Jan, May and June datasets
'''
#will make changes to the max_labels in the too_big() function and min_cluster size in max_pop
from MulticoreTSNE import MulticoreTSNE as multicore_tsne  # https://github.com/DmitryUlyanov/Multicore-TSNE
import copy
import Louvain_igraph_Jac24Sept as ls
from sklearn.neighbors import KNeighborsClassifier
import Performance_phenograph as pp
import numpy as np
import time
import scipy.io
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 350
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
from MST_minClusters import MSTClustering
#from MST_APTSOM import MSTClustering3D
from MST_3D_current import MSTClustering3D
from MST_2D import MSTClustering2D
import Plotting_3D as Plotting_3D
from pandas import ExcelWriter
from Louvain_igraph_Jac24Sept import accuracy_mst
import time
import sklearn.cluster
from sklearn.cluster import DBSCAN, KMeans
import os.path
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn import metrics
import PARC as parc

print(os.path.abspath(sklearn.cluster.__file__))

def write_list_to_file(input_list, filename):
    """Write the list to file."""

    with open(filename, "w") as outfile:
        for entries in input_list:
            outfile.write(str(entries))
            outfile.write("\n")

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
    feat_cols= ['Area', 'Volume', 'Circularity', 'Attenuation density', 'Amplitude var', 'Amplitude skewness',
                 'Amplitude kurtosis', 'Dry mass', 'Dry mass density', 'Dry mass var', 'Dry mass skewness', 'Peak phase',
                 'Phase var', 'Phase skewness', 'Phase kurtosis', 'DMD contrast 1', 'DMD contrast 2', 'DMD contrast 3',
                 'DMD contrast 4', 'Mean phase arrangement', 'Phase arrangement var', 'Phase arrangement skewness',
                 'Phase orientation var', 'Phase orientation kurtosis']#, 'Focus factor 1', 'Focus factor 2']

    feat_cols_VOLONLY= ['Area', 'Volume', 'Circularity'] #feat_cols_vol_only
    # ALL FEATURES EXCLUDING FILE AND CELL ID:
    feat_cols_exVolume = [ 'Attenuation density', 'Amplitude var', 'Amplitude skewness',
                 'Amplitude kurtosis', 'Dry mass', 'Dry mass density', 'Dry mass var', 'Dry mass skewness',
                 'Peak phase','Phase var', 'Phase skewness', 'Phase kurtosis', 'DMD contrast 1', 'DMD contrast 2', 'DMD contrast 3',
                 'DMD contrast 4', 'Mean phase arrangement', 'Phase arrangement var', 'Phase arrangement skewness',
                 'Phase orientation var', 'Phase orientation kurtosis']#, 'Focus factor 1', 'Focus factor 2']
    feat_cols_includefluor = ['Area', 'Volume', 'Circularity', 'Attenuation density', 'Amplitude var', 'Amplitude skewness',
                              'Amplitude kurtosis', 'Dry mass', 'Dry mass density', 'Dry mass var', 'Dry mass skewness',
                              'Peak phase', 'Phase var', 'Phase skewness', 'Phase kurtosis', 'DMD contrast 1',
                              'DMD contrast 2', 'DMD contrast 3', 'DMD contrast 4', 'Mean phase arrangement',
                              'Phase arrangement var', 'Phase arrangement skewness', 'Phase orientation var',
                              'Phase orientation kurtosis', 'Focus factor 1', 'Focus factor 2', 'Fluorescence (Peak)',
                              'Fluorescence (Area)', 'Fluorescence density', 'Fluorescence-Phase correlation']
    feat_cols_fluor_only = ['Fluorescence (Peak)', 'Fluorescence (Area)', 'Fluorescence density',
                            'Fluorescence-Phase correlation']
    print('num features is', len(feat_cols))
    # January, May(H526) and June (HCC827)

    df_h2170 = make_subtype_df('h2170','h21702018Jan23_gatedH2170',0,n_eachsubtype, randomseedval, HTC_filename ="\\\\Desktop-u14r2et\\G\\2018Jan23\\" ) # same path 2170_Raw, 28 x 302,635
    df_h1975 = make_subtype_df('h1975','h19752018Jan23_gatedH1975',1,100, randomseedval, HTC_filename= '\\\\147.8.182.49\g\\') #60447 *same path*
    df_h526 = make_subtype_df('h526','H5262018May24_gatedH526',2,n_eachsubtype, randomseedval, HTC_filename = 'E:\\2018May24_cancer\\')#375889 #WAS 562 until Jan16 \\Desktop-p9kngca\e
    df_h520 = make_subtype_df('h520', 'h5202018Jan03_gatedH520',3, n_eachsubtype, randomseedval,HTC_filename='\\\\DESKTOP-H5E5CH1\\DVDproc - 2 (G)\\20180103 H520\\')  # 451208 *same path*
    df_h358 = make_subtype_df('h358','h3582018Jan03_gatedH358',4,n_eachsubtype, randomseedval, HTC_filename='\\\\DESKTOP-H5E5CH1\\DVDproc - 2\\2018Jan3_h358_520_526\\')#170198 *same path*
    df_h69 = make_subtype_df('h69','h692018Jan23_gatedH69',5,n_eachsubtype, randomseedval,HTC_filename= '\\\\147.8.182.49\g\\') #130075 *same path*
    df_hcc827 = make_subtype_df('hcc827','hcc8272018Jun05_gatedHcc827',6,n_eachsubtype, randomseedval, HTC_filename='E:\\2018Jun05_lungcancer\\') #same path
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
    #frames = [df_h2170,df_h1975,df_h526,df_h520,df_h358]
    df_all = pd.concat(frames, ignore_index=True,sort=False)


    # EXCLUDE FLUOR FEATURES
    if fluor == 0:
        #print('raw min', df_all[feat_cols].min())
        #df_all[feat_cols] = (df_all[feat_cols] + df_all[feat_cols].min().abs())+1
        #print('min',df_all[feat_cols].min())
        #df_all[feat_cols] = df_all[feat_cols].apply(np.log)
        #print('mean', df_all[feat_cols].mean())
        #print('std', df_all[feat_cols].std())
        #NORMALIZATION
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
    #print(randomseedval, ' is the randomseed value')
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
    #print('shape of fileidx', subtype_fileidx.shape)
    df_subtype['cell_filename'] = HTC_filename + subtype_name.split('_')[0] + '_' + subtype_fileidx["filename"].map(
            int).map(str)
    if class_val==4:
        df_subtype['cell_filename'] = HTC_filename + subtype_name.split('_')[0] + '_' + subtype_fileidx["filename"].map(
            int).map(str)+'mat'
        #print('class val is ',class_val)

    df_subtype['cell_idx_inmatfile'] = subtype_fileidx["index"]#.map(int).map( str)  # should be same number as image number within that folder
    df_subtype['cell_tag'] = subtype_name.split('_')[0] + subtype_fileidx["filename"].map(int).map(str) + '_' + \
                              subtype_fileidx["index"].map(int).map(str)
    df_subtype['label'] = class_val
    df_subtype['class'] = class_val
    df_subtype['class_name'] = str_subtypename

    #print('shape before drop dups', df_subtype.shape)
    df_subtype = df_subtype.drop_duplicates(subset = flist, keep='first')
    df_subtype.replace([np.inf, -np.inf], np.nan).dropna()
    #print('shape after drop dups', df_subtype.shape)
    #print(df_subtype.head(5))
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

def plot_all_methods(X_embedded, true_label, embedding_filename, dbscan_labels, mst_labels, louvain_labels, pheno_labels, kmeans_labels, onevsall_mst, onevsall_dbscan,onevsall_louvain,onevsall_pheno, onevsall_kmeans, dimred,sigma_opt, eps_opt, min_cluster_opt,dbscan_min_clustersize, knn_opt=15):

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
        ax[0].scatter(x, y, color=true_color, s=1, alpha=0.6, label=true_label_str+' Cellcount = ' + str(population), edgecolors= 'none')
        ax[0].annotate(true_label_str, xytext=(np.mean(x), np.mean(y)), xy=(np.mean(x), np.mean(y)), color='black',
                       weight='semibold')
    box = ax[0].get_position()
    ax[0].set_position([box.x0, box.y0, box.width *0.9, box.height])
    ax[0].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize = 10, markerscale = 6)
    title_str0 = 'Multi-Class Lung Cancer Cell Lines:Ground Truth. \n'+'Total Cell count is ' +str(N)  # embedding_filename
    ax[0].set_title(title_str0, size=10)
    if knn_opt==None:
        ax[1]= plot_onemethod(ax[1],X_embedded,mst_labels, true_label,onevsall_mst, 'mst', dimred,sigma_opt, min_cluster_opt, None)
    if knn_opt!=None:
        ax[1] = plot_onemethod(ax[1], X_embedded, louvain_labels, true_label, onevsall_louvain, 'louvain', dimred, None, None, knn_opt)
    box1 = ax[1].get_position()
    ax[1].set_position([box1.x0, box1.y0, box1.width * 0.9, box1.height])
    ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize = 6, markerscale = 6)
    #ax[1][0]= plot_onemethod(ax[1][0],X_embedded,dbscan_labels, true_label,onevsall_dbscan, 'dbscan', dimred,eps_opt, dbscan_min_clustersize, None)
    #ax[1][1]= plot_onemethod(ax[1][1],X_embedded,louvain_labels, true_label,onevsall_louvain, 'louvain', dimred,None, None, knn_opt)
    #ax[2][0]= plot_onemethod(ax[2][0],X_embedded,pheno_labels, true_label,onevsall_pheno, 'phenograph', dimred,None, None, 30)
    #ax[2][1]= plot_onemethod(ax[2][1],X_embedded,kmeans_labels, true_label,onevsall_kmeans, 'kmeans', dimred,None, None, None)

    plt.show()

    plt.savefig(embedding_filename + '_' + dimred + '.tif', bbox_inches='tight', dpi=350)


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

    pair_color_group_list = [(colors_h2170, h2170_labels, ['h2170']*len(h2170_labels)),(colors_h526, h526_labels, ['h526']*len(h526_labels)),(colors_h520, h520_labels, ['h520']*len(h520_labels)),(colors_h358, h358_labels, ['h358']*len(h358_labels)),
                             (colors_h69, h69_labels, ['h69'] * len(h69_labels)),(colors_hcc827, hcc827_labels, ['hcc827'] * len(hcc827_labels)),(colors_h1975,h1975_labels, ['h1975']*len(h1975_labels))]
    for color, group, name in pair_color_group_list:
        for color_m, ll_m, label_m in zip(color, group,name):
            x = [t[0] for t in X_dict[ll_m]]
            y = [t[1] for t in X_dict[ll_m]]
            population = len(x)
            ax.scatter(x, y, color=color_m, s=1, alpha=0.6, label= label_m+'_' + str(ll_m) + ' Cellcount = ' + str(len(x)), edgecolors = 'none')
            #ax.annotate(str(int(ll_m)), xytext=(np.mean(x), np.mean(y)), xy=(np.mean(x), np.mean(y)), color='black',  weight='semibold')
    '''
    ax.text(0.95, 0.01, "number of groups: " + " {:.0f}".format(num_groups) + " FP: " + " {:.2f}".format(
        fp * 100 / n_pbmc) + "%. FN of " + "{:.2f}".format(fn * 100 / (n_cancer)) + '%', transform=ax.transAxes,
               verticalalignment='bottom', horizontalalignment='right', color='green', fontsize=10)
    '''
    ax.axis('tight')

    if method == 'mst':
        title_str1 = 'MST on '+ dimred +' embedding: mean + ' + str(sigma) + '-sigma cutoff and min cluster size of: ' + str(
        min_cluster) + '\n' +"Total error rate: {:.1f}".format(total_error_rate * 100) + '%\n' + "One-vs-all for " +onevsall_opt+ " FP: " + " {:.1f}".format(
        fp * 100 / n_pbmc) + "%. FN of " + "{:.1f}".format(
        fn * 100 / (n_cancer)) + '%' + '\n' ' F1-score:' + "{:.2f} %".format(f1_score*100)
    if method == 'louvain':
        title_str1 = 'PARC Clustering'
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
        ax[0].scatter(x, y, color=true_color, s=1, alpha=1, label=true_label_str+' Cellcount = ' + str(population))
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
            ax[1].scatter(x, y, color=color_m, s=1, alpha=1, label= label_m+'_' + str(ll_m) + ' Cellcount = ' + str(len(x)))
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
    ax[0].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize = 10, markerscale = 6)
    plt.savefig(embedding_filename+'_' +method+ '_'+dimred+'.tif', bbox_inches='tight', dpi=350)
    #plt.show()
def multiclass_mst_accuracy(X_embedded, true_label,df_all,av_peaks=15, too_big_factor = 0.20, verbose_print=False,inputGraph = False, original_data = None, min_clustersize = 10,peak_threshhold = 1,final_small_pop = 20): #use 20 for multiclass and 5 for rare
    #fig_density, ax_density = plt.subplots(1, 2, figsize=(36, 12))
    #for CTYof use peak_threshhold =0, else for 10x GENE use peak_threshhold=1

    if av_peaks == 0:
        xy_peaks  =pp.tsne_densityplot(X_embedded[:,:2], ax1 = None, ax2= None, df_all=df_all, mode='vals only',peak_threshhold=peak_threshhold)
        #plt.show()
        XZ=X_embedded[:, np.ix_([0,2])]
        XZ = XZ[:,0,:]
        YZ =X_embedded[:, np.ix_([1,2])]
        YZ = YZ[:, 0, :]
        print('xz and yz shape',XZ.shape, YZ.shape)
        xz_peaks = pp.tsne_densityplot(XZ, None, None, df_all=None, mode='only_vals', peak_threshhold = peak_threshhold)
        yz_peaks = pp.tsne_densityplot(YZ, None, None, df_all=None, mode='only_vals',peak_threshhold=peak_threshhold)
        av_peaks = round((xy_peaks.shape[0]+yz_peaks.shape[0]+xz_peaks.shape[0])/3)#+10 #on 20th Sep we are adding +10 as av_peaks for cytof and 10X seems too low otherwise
        if peak_threshhold ==-2:
            av_peaks = av_peaks+10
            print('adding ten extra peaks for buffer')
        print('no. peaks,', xy_peaks.shape,xz_peaks.shape,yz_peaks.shape, av_peaks)

    #av_peaks = 15
    min_clustersize = [min_clustersize]
    #,10]#[10,20](used 10,20 until OCT 3, we are changing just to experiment with SOMs)#[10]#[50,30,20,10] until sep10 we used 10, most of sep11 we tried 50 and then 20, but now going back to 10. at 430pm on 12th Sep i change it back to 20
    #if too_big_factor == 0.05:
     #   too_big_factor = min(0.05, 20000 / len(true_label))
     #   min_clustersize = [100]
    f1_temp = -1
    f1_sum_best = 0
    list_roc = []
    targets = list(set(true_label))
    if len(targets) > 0:
        target_range = targets
    else:
        target_range = [1]
    for i_min_clustersize in min_clustersize:

        if inputGraph:
            print('input is a graph')
            print('shape of original data', original_data.shape)
            print('min cluster size is', i_min_clustersize)
            model = MSTClustering3D(approximate=True, min_cluster_size=i_min_clustersize, max_labels=av_peaks, true_label=true_label, metric='precomputed', X_fit_original=original_data)
        else: model = MSTClustering3D(approximate=True, min_cluster_size=i_min_clustersize, max_labels=av_peaks, true_label=true_label, X_fit_original=original_data)
        #model = MSTClustering()
        time_start = time.time()
        print('Starting Clustering', time.ctime())
        model.fit_predict(X_embedded)
        clustering_labels = model.labels_
        too_big = False
        #print('length of X_embedded', X_embedded[0])
        for cluster_i in set(clustering_labels):
            cluster_i_loc = np.where(np.asarray(clustering_labels) == cluster_i)[0]
            pop_i = len(cluster_i_loc)
            if inputGraph==True:total_pop = original_data.shape[0]
            else: total_pop = len(true_label)
            if pop_i > too_big_factor * total_pop:
                too_big = True
                cluster_big = pop_i
                cluster_big_loc = cluster_i_loc
        list_pop_too_bigs = [pop_i]
        while too_big == True:
            if inputGraph==True: X_data_big = original_data[cluster_big_loc, :]
            else:X_data_big = X_embedded[cluster_big_loc, :]
            big_pop = X_data_big.shape[0]
            print('removing cluster',cluster_big,' too big population of ', big_pop)
            #knn_struct_big = ls.make_knn_struct(X_data_big)
            print(X_data_big.shape)
            #louvain_labels_big = ls.run_toobig_sublouvain(X_data_big, knn_struct_big, k_nn=50, self_loop=False)
            if inputGraph:

                print('input is a graph')
                print('shape of big data', X_data_big.shape)
                knn_struct = ls.make_knn_struct(X_data_big, ef=50)
                X_data_copy = copy.deepcopy(X_data_big) #i.e. the codebook
                big_neighbor_array, big_distance_array = knn_struct.knn_query(X_data_copy, k=5)
                ##CODE FOR DENSEST NEIGHBOR GRAPH START
                #print('random distance entry', big_distance_array[2,])

                #big_density_array = np.empty([big_neighbor_array.shape[0], 1])
                #big_dense_neighbor_array = np.empty([big_neighbor_array.shape[0], 1])
                #big_dense_distance_array = np.empty([big_neighbor_array.shape[0], 1])
                #for i in range(big_neighbor_array.shape[0]):
                #    big_dense_neighbor_array[i, 0] = big_neighbor_array[i, np.argmax(big_density_array[big_neighbor_array[i,]])]
                #    big_dense_distance_array[i, 0] = big_distance_array[i, np.argmax(big_density_array[big_neighbor_array[i,]])]
                #print('big dense distance array', big_dense_distance_array)
                #X_data_big_graph, dummy = ls.make_csrmatrix_noselfloop(big_dense_neighbor_array, big_dense_distance_array, dist_std=3, keep_all=True)
                ## END CODE FOR DENSEST NEIGHBOR GRAPH
                X_data_big_graph, dummy = ls.make_csrmatrix_noselfloop(big_neighbor_array,
                                                                       big_distance_array, dist_std=3,
                                                                       keep_all=True)

                model_big = MSTClustering3D(approximate=True, min_cluster_size=1, max_labels=7, #min cluster size was 5
                                        true_label=true_label, metric='precomputed', X_fit_original=X_data_big)
                model_big.fit_predict(X_data_big_graph)
            else:
                model_big = MSTClustering3D(min_cluster_size=20, approximate=True, n_neighbors=30,
                                            max_labels=7)  # max = 7 until sep11 a,d min_cluster = 20 n_neigh was 20
                model_big.fit_predict(X_data_big)
            louvain_labels_big = model_big.labels_
            pop_list = []
            for item in set(list(louvain_labels_big.flatten())):
                pop_list.append(list(louvain_labels_big.flatten()).count(item))
            print('pop list if using MST to expand cluster', pop_list)
            not_expanded_enough = False
            '''
            if pop_list[0] >0.95*big_pop: not_expanded_enough = True
            if len(set(louvain_labels_big))==1 or not_expanded_enough ==True:
                X_data_big_louvain = X_data_ndim[cluster_big_loc]
                print('making KNN for louvain to expand LARGE clusters')
                knn_struct_big = ls.make_knn_struct(X_data_big)
                louvain_labels_big=ls.run_toobig_sublouvain(X_data_big_louvain, knn_struct_big, k_nn=25, self_loop=False,keep_all=False) #was knn=50
            else: louvain_labels_big = np.asarray(louvain_labels_big)
            '''
            louvain_labels_big = np.asarray(louvain_labels_big)
            print('set of new big labels ', set(list(louvain_labels_big.flatten())))
            louvain_labels_big = louvain_labels_big + 1000  # len(set_louvain_labels)
            #print('set of new big labels +1000 ', set(list(louvain_labels_big.flatten())))
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

                if pop_i > too_big_factor * total_pop and not_already_expanded ==True:
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
        counts = np.bincount(clustering_labels)
        if inputGraph == True:  to_remove = np.where(counts < 1) #use this for SOMs codebook data
        else: to_remove = np.where(counts < final_small_pop)[0] #was 50 in the morning of sept 12. sept 13 changing it to 10
        clustering_labels = np.asarray(clustering_labels)
        print('number of too small clusters at the end is', len(to_remove))
        if len(to_remove) > 0:
            for i in to_remove:
                clustering_labels[clustering_labels == i] = -1
            dummy, clustering_labels = np.unique(clustering_labels, return_inverse=True)
            clustering_labels -= 1  # keep -1 labels the same
        X_big = X_embedded[clustering_labels != -1]
        # print('shape of X_big ',X_big.shape)
        X_small = X_embedded[clustering_labels == -1]
        # print('x_small shape: ', X_small.shape)
        if X_small.shape[0] > 0:
            labels_big = clustering_labels[clustering_labels != -1]
            neigh = KNeighborsClassifier(n_neighbors=3)
            neigh.fit(X_big, labels_big)
            # print('made KneighborClassifier')
            y_small = neigh.predict(X_small)
            # print('y_small shape:',y_small.shape)
            y_small_ix = np.where(clustering_labels == -1)[0]
            # print('made outlier labels of length', len(y_small_ix))
            ii = 0
            for iy in y_small_ix:
                clustering_labels[iy] = y_small[ii]
                # print(y_small[ii])
                ii = ii + 1
        clustering_labels = list(clustering_labels.flatten())
        runtime_mst = time.time() - time_start
        print('runtime', runtime_mst)
        onevsall_str = 'placeholder'
        f1_sum = 0
        f1_accumulated=0
        f1_acc_noweighting = 0
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
            vals_roc, predict_class_array, majority_truth_labels, numclusters_targetval = accuracy_mst(clustering_labels, true_label,
                                                         embedding_filename=None, clustering_algo='multiclass mst',
                                                         onevsall=onevsall_val,verbose_print=verbose_print)

            if vals_roc[1] > f1_temp:
                f1_temp = vals_roc[1]
                onevsall_val_opt = onevsall_val
            f1_sum = f1_sum + vals_roc[1]
            f1_current = vals_roc[1]
            f1_accumulated = f1_accumulated + f1_current * (list(true_label).count(onevsall_val)) / len(true_label)
            f1_acc_noweighting = f1_acc_noweighting + f1_current
            list_roc.append([model.sigma_factor, i_min_clustersize, model.tooclosefactor, onevsall_val, onevsall_str] + vals_roc + [runtime_mst])
        #print('ARI APT',
        #      adjusted_rand_score(np.asarray(true_label), np.asarray(clustering_labels)),
        #      metrics.adjusted_mutual_info_score(true_label, clustering_labels))
        print("f1-score accumulated (weighted by population) ", f1_accumulated)
        f1_mean = f1_acc_noweighting/len(target_range)
        print("f1-score mean, no weighting ", f1_mean)


        if f1_sum > f1_sum_best:
            f1_sum_best = f1_sum
            temp_best_labels = clustering_labels
            sigma_opt = model.sigma_factor
            tooclose_factor_opt = model.tooclosefactor
            onevsall_best = onevsall_val_opt
            min_clustersize_opt = i_min_clustersize

    print('best min cluster size was', min_clustersize_opt)
    df_accuracy = pd.DataFrame(list_roc,
                           columns=['sigma factor', 'min cluster size','merge-too-close factor', 'onevsall target','cell type','error rate', 'f1-score', 'tnr', 'fnr',
                                    'tpr', 'fpr', 'precision', 'recall', 'num_groups', 'population of target','runtime'])

    return df_accuracy, temp_best_labels, sigma_opt, min_clustersize_opt, tooclose_factor_opt, onevsall_best, majority_truth_labels, f1_accumulated, f1_mean


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
                vals_roc, predict_class_array, maj_truth_labels = accuracy_mst(model, true_label,
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
        vals_roc, predict_class_array, maj_truth_labels  = accuracy_mst(model, true_label,
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

def get_SampleImageIDs_matchClusterMap(cluster_labels, df_all, true_labels,cluster_map_order):
    import random
    cluster_i_tag_list = []
    for cluster_i in cluster_map_order:
        cluster_i_loc = np.where(cluster_labels == np.int64(cluster_i))[0]
        print(len(cluster_i_loc), 'population of cluster', cluster_i)
        population_cluster_i = len(cluster_i_loc)

        majority_truth = func_mode(list(true_labels[cluster_i_loc]))
        random.shuffle(cluster_i_loc)
        cluster_i_loc_20= cluster_i_loc[0:10]

        if population_cluster_i>100:
            for k in cluster_i_loc_20:
                cluster_i_tag_list.append([np.int64(cluster_i),majority_truth,df_all.loc[k, 'label'], df_all.loc[k, 'cell_filename'], df_all.loc[k, 'cell_idx_inmatfile'],
                       df_all.loc[k, 'File ID'], df_all.loc[k, 'Cell ID'], df_all.loc[k, 'index']])
    column_names_tags = ['cluster','majority truth','celltype', 'filename', 'idx_inmatfile','File ID', 'Cell ID','df_all idx']
    df_sample_imagelist = pd.concat([pd.DataFrame([i], columns=column_names_tags) for i in cluster_i_tag_list], ignore_index=True)
    return df_sample_imagelist


def run_main(new_file_name, n_eachsubtype=None, randomseedval=1):
    df_all, true_label, X_txt, feat_cols = get_data(fluor=0, n_eachsubtype=n_eachsubtype, randomseedval=randomseedval)
    # df = pd.DataFrame(X_txt)
    n_total = X_txt.shape[0]
    print('dims of data', X_txt.shape)
    '''
    for n_clusters in[7,8,9,10,12,14,16,18,20]:#,22,24,26,28,30,32,34,36,38,40]:
        model = KMeans(n_clusters=n_clusters, n_init=3,max_iter=30, verbose=0, n_jobs=-3).fit(X_txt)

        f1_mean = 0
        for onevsall_val in list(set(true_label)):
            # print('target is', onevsall_val)
            vals_roc, predict_class_array, maj, numclusters_targetval = ls.accuracy_mst(model.labels_, true_label,
                                                                                        embedding_filename=None,
                                                                                        clustering_algo='louvain',
                                                                                        onevsall=onevsall_val)

            f1_current = vals_roc[1]
            # print('for target', onevsall_val, 'the f1-score is', f1_current)
            #f1_accumulated = f1_accumulated + f1_current * (list(true_label).count(onevsall_val)) / N
            f1_mean = f1_current + f1_mean
            print('f1-score for target', onevsall_val, 'is', f1_current)
        print('kmeans f1 score', f1_mean / len(set(true_label)), 'at num clusters', len(set(model.labels_)))
        print('ari for kemans',n_clusters, 'groups', adjusted_rand_score(np.asarray(true_label), model.labels_))
    '''
    datamatrix_file_name = '/home/shobi/Thesis/MultiClass_MinCluster/April 2019/' + 'datamatrix_Oct25' + new_file_name + '_N' + str(
        n_total) + '.txt'
    True_label_file_name = '/home/shobi/Thesis/MultiClass_MinCluster/April 2019/' + 'true_label_Oct25' + new_file_name + '_N' + str(
        n_total) + '.txt'
    # df.to_csv("/home/shobi/Thesis/Rcode/LungCancerData.txt", header=None, index=None)
    # np.savetxt("/home/shobi/Thesis/Rcode/FLOCK/LungCancerData_RareTarget1975_N"+str(X_txt.shape[0])+"RandInt"+str(randomseedval)+"_May.txt",X_txt,delimiter='\t', fmt = '%f',header  = "Area\tVolume\tCircularity\tAttenuationdensity\tAmplitudevar\tAmplitudeskewness\tAmplitudekurtosis\tFocusfactor1\tFocusfactor2\tDrymass\tDrymassdensity\tDrymassvar\tDrymassskewness\tPeakphase\tPhasevar\tPhaseskewness\tPhasekurtosis\tDMDcontrast1\tDMDcontrast2\tDMDcontrast3\tDMDcontrast\tMeanphasearrangement\tPhasearrangementvar\tPhasearrangementskewness\tPhaseorientationvar\tPhaseorientationkurtosis")
    # np.savetxt("/home/shobi/Thesis/Rcode/FlowPeaks/LungCancerData_RareTarget1975_TrueLabel_N"+str(X_txt.shape[0])+"RandInt"+str(randomseedval)+"_May.txt", true_label, delimiter=',')
    # np.savetxt("/home/shobi/Thesis/Rcode/FLOCK/LungCancerData_RareTarget1975_TrueLabel_N" + str(X_txt.shape[0]) + "RandInt" + str(randomseedval) + "_May.txt", true_label, delimiter=',')
    # np.savetxt("/home/shobi/Thesis/Rcode/FlowSOM/LungCancerData_RareTarget1975_TrueLabel_N" + str(X_txt.shape[0]) + "RandInt" + str(randomseedval) + "_May.txt", true_label, delimiter=',')
    # np.savetxt("/home/shobi/Thesis/Rcode/FlowSOM/LungCancerData_RareTarget1975_N" + str(X_txt.shape[0]) + "RandInt" + str(randomseedval) + "_May.txt", X_txt, delimiter=',', fmt='%f')
    # np.savetxt("/home/shobi/Thesis/Rcode/FlowPeaks/LungCancerData_RareTarget1975_N" + str(X_txt.shape[0]) + "RandInt" + str(randomseedval) + "_May.txt", X_txt, delimiter=',', fmt='%f')
    np.savetxt(datamatrix_file_name, X_txt, delimiter=',', fmt='%f')
    write_list_to_file(true_label, True_label_file_name)

    print('saved data as txt')

    frames = None
    perplexity_range = [30]  # [10, 30, 50,70,90]
    partition_seed = 3
    weighted = False
    keep_all = True
    for small_pop_i in [10,10,10,10]:
        excel_file_name = '/home/shobi/Thesis/MultiClass_MinCluster/April 2019/' + new_file_name + '_N' + str(
            n_total) + '_perp.xlsx'
        mst_label_file_name = '/home/shobi/Thesis/MultiClass_MinCluster/Jan 2019/' + new_file_name + '_N' + str(
            n_total) + '_perp.txt'

        Pheno_label_file_name = '/home/shobi/Thesis/MultiClass_MinCluster/April 2019/' + new_file_name + 'Pheno_N' + str(
            n_total) + '.txt'
        # excel_file_name_mst = '/home/shobi/Thesis/MultiClass_MinCluster/ ' + new_file_name + '_N' + str(
        #    n_total) + '_perp' + str(perplexity) + 'mst.xlsx'
        plot_name = '/home/shobi/Thesis/MultiClass_MinCluster/April 2019/' + new_file_name + '_N' + str(
            n_total) + '_perp' + str(small_pop_i)
        print(plot_name)

        writer = ExcelWriter(excel_file_name)

        jac_std_list = [2]
        dist_std = 2
        true_label = pd.Series(true_label)

        # run alph
        print('start alph at', time.ctime())


        if weighted == True:
            print('weighted edges')
        else:
            print('unweighted edges')
        for jac_std in jac_std_list:
            print('jac', jac_std)
            Alph_label_file_name = '/home/shobi/Thesis/MultiClass_MinCluster/April 2019/' + new_file_name + '_N' + str(
                n_total) + 'partitionSeed' + str(partition_seed) + 'keepall' + str(keep_all) + 'weighted' + str(
                weighted) +'jac_std'+str(jac_std)+ 'smallpop'+str(small_pop_i)+'.txt'
            #print('start phenograph')
            #predict_class_aggregate, df_accuracy, pheno_labels, onevsall_opt, majority_truth_labels, pheno_time, f1_mean_pheno = ls.run_phenograph( X_txt, true_label, knn=5)

            predict_class_aggregate_louvain, df_accuracy_louvain, best_louvain_labels, knn_opt, onevsall_opt_louvain, maj_truth_labels, time_end_knn, time_end_prune, time_end_louvain, time_end_total, f1_accumulated, f1_mean, time_end_knn_query, num_edges = ls.run_mainlouvain(
                X_txt, true_label, self_loop=False, keep_all_dist=keep_all, jac_weighted_edges=weighted,
                jac_std=jac_std, dist_std=dist_std, small_pop=small_pop_i, knn_in=5, partition_seed=partition_seed)
            print('ari for PARC', len(set(best_louvain_labels)), 'groups', adjusted_rand_score(np.asarray(true_label), best_louvain_labels))
            write_list_to_file(best_louvain_labels, Alph_label_file_name)
            write_list_to_file(true_label, True_label_file_name)
            print('saved files')
        # predict_class_aggregate_louvain_5, df_accuracy_louvain_5pop, best_louvain_labels_pop5, knn_opt, onevsall_opt_louvain, maj_truth_labels, time_end_knn, time_end_prune, time_end_louvain, time_end_total, f1_accumulated, f1_mean, time_end_knn_query, num_edges = ls.run_mainlouvain(
        #    X_txt, true_label, self_loop=False, keep_all_dist=True, jac_weighted_edges=weighted, jac_std=jac_std,                dist_std=dist_std, small_pop=5)
        print('end alph at', time.ctime())
        df_accuracy_louvain.to_excel(writer, 'alph_small10', index=False)
        # df_accuracy_louvain_5pop.to_excel(writer, 'alph_small5', index=False)
        writer.save()
def run_main_original(new_file_name, n_eachsubtype= None, randomseedval=1):
    df_all, true_label, X_txt, feat_cols = get_data(fluor=0, n_eachsubtype=n_eachsubtype, randomseedval=randomseedval)

    #df = pd.DataFrame(X_txt)
    n_total = X_txt.shape[0]
    print('dims of data', X_txt.shape)
    datamatrix_file_name = '/home/shobi/Thesis/MultiClass_MinCluster/April 2019/' + 'datamatrix_Oct22' + new_file_name + '_N' + str(
        n_total) + '.txt'
    True_label_file_name = '/home/shobi/Thesis/MultiClass_MinCluster/April 2019/' + 'true_label_Oct22' + new_file_name + '_N' + str(
        n_total) + '.txt'
    #df.to_csv("/home/shobi/Thesis/Rcode/LungCancerData.txt", header=None, index=None)
    #np.savetxt("/home/shobi/Thesis/Rcode/FLOCK/LungCancerData_RareTarget1975_N"+str(X_txt.shape[0])+"RandInt"+str(randomseedval)+"_May.txt",X_txt,delimiter='\t', fmt = '%f',header  = "Area\tVolume\tCircularity\tAttenuationdensity\tAmplitudevar\tAmplitudeskewness\tAmplitudekurtosis\tFocusfactor1\tFocusfactor2\tDrymass\tDrymassdensity\tDrymassvar\tDrymassskewness\tPeakphase\tPhasevar\tPhaseskewness\tPhasekurtosis\tDMDcontrast1\tDMDcontrast2\tDMDcontrast3\tDMDcontrast\tMeanphasearrangement\tPhasearrangementvar\tPhasearrangementskewness\tPhaseorientationvar\tPhaseorientationkurtosis")
    #np.savetxt("/home/shobi/Thesis/Rcode/FlowPeaks/LungCancerData_RareTarget1975_TrueLabel_N"+str(X_txt.shape[0])+"RandInt"+str(randomseedval)+"_May.txt", true_label, delimiter=',')
    #np.savetxt("/home/shobi/Thesis/Rcode/FLOCK/LungCancerData_RareTarget1975_TrueLabel_N" + str(X_txt.shape[0]) + "RandInt" + str(randomseedval) + "_May.txt", true_label, delimiter=',')
    #np.savetxt("/home/shobi/Thesis/Rcode/FlowSOM/LungCancerData_RareTarget1975_TrueLabel_N" + str(X_txt.shape[0]) + "RandInt" + str(randomseedval) + "_May.txt", true_label, delimiter=',')
    #np.savetxt("/home/shobi/Thesis/Rcode/FlowSOM/LungCancerData_RareTarget1975_N" + str(X_txt.shape[0]) + "RandInt" + str(randomseedval) + "_May.txt", X_txt, delimiter=',', fmt='%f')
    #np.savetxt("/home/shobi/Thesis/Rcode/FlowPeaks/LungCancerData_RareTarget1975_N" + str(X_txt.shape[0]) + "RandInt" + str(randomseedval) + "_May.txt", X_txt, delimiter=',', fmt='%f')
    np.savetxt(datamatrix_file_name, X_txt, delimiter=',', fmt='%f')
    write_list_to_file(true_label, True_label_file_name)

    print('saved data as txt')

    frames = None
    perplexity_range = [30]#[10, 30, 50,70,90]
    for perplexity in perplexity_range:
        excel_file_name = '/home/shobi/Thesis/MultiClass_MinCluster/April 2019/'+new_file_name+'_N'+str(n_total)+'_perp'+str(perplexity)+'.xlsx'
        mst_label_file_name = '/home/shobi/Thesis/MultiClass_MinCluster/Jan 2019/' + new_file_name + '_N' + str(
            n_total) + '_perp' + str(perplexity) +'.txt'
        Alph_label_file_name = '/home/shobi/Thesis/MultiClass_MinCluster/April 2019/' + new_file_name + '_N' + str(
            n_total)  + '.txt'


        Pheno_label_file_name = '/home/shobi/Thesis/MultiClass_MinCluster/April 2019/' + new_file_name + 'Pheno_N' + str(
            n_total) + '.txt'
        #excel_file_name_mst = '/home/shobi/Thesis/MultiClass_MinCluster/ ' + new_file_name + '_N' + str(
        #    n_total) + '_perp' + str(perplexity) + 'mst.xlsx'
        plot_name = '/home/shobi/Thesis/MultiClass_MinCluster/April 2019/'+new_file_name+'_N'+str(n_total)+'_perp'+str(perplexity)
        print(plot_name)

        writer = ExcelWriter(excel_file_name)
        #writer_mst = ExcelWriter(excel_file_name_mst)

        if perplexity == perplexity_range[0]:
            time_start_alph = time.time()
            print('start pheno', time.asctime())
            #predict_class_aggregate_louvain, df_accuracy_louvain, best_louvain_labels, onevsall_opt, majority_truth_labels, pheno_time, f1_acc_noweighting = ls.run_phenograph(X_txt, true_label)

            #knn_opt = 30
            #onevsall_opt_louvain = 0

            jac_std_list=['median']

            dist_std =2



            true_label = pd.Series(true_label)
            '''
            #run K-means
            for k_clusters in [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70]:  # ,25,30,35,40,45,50,55,60,65,70]:
                print('k clusters is', k_clusters)
                kmeans = KMeans(n_clusters=k_clusters, max_iter=150, random_state=10).fit(X_txt)
                targets = list(set(true_label))
                if len(targets) >= 2:
                    target_range = targets
                else:
                    target_range = [1]
                N = len(true_label)
                f1_accumulated = 0
                f1_mean = 0
                for onevsall_val in [1]:#0,1,2]:
                    vals_roc, predict_class_array, maj, num_onevsall_vall_clusters = ls.accuracy_mst(
                        list(kmeans.labels_), true_label,
                        embedding_filename=None, clustering_algo='louvain', onevsall=onevsall_val)
                    f1_current = vals_roc[1]
                    print(f1_current, 'is the f1-score of target', onevsall_val)
                    #f1_accumulated = f1_accumulated + f1_current * (list(true_label).count(onevsall_val)) / N
                    #f1_mean = f1_current + f1_mean
                #list_of_lists.append([k_clusters, f1_accumulated, f1_mean / len(target_range), ARI, AMI])
                #print('stats', list_of_lists)
            print('END KMEANS')
            '''
            #run alph
            print('start alph at', time.ctime())
            weighted = True
            if weighted == True:print('weighted edges')
            else: print('unweighted edges')
            for jac_std in jac_std_list:
                print('jac std', jac_std, 'dis std', dist_std)
                #predict_class_aggregate, df_accuracy, pheno_labels, onevsall_opt, majority_truth_labels, pheno_time, f1_mean_pheno = ls.run_phenograph( X_txt, true_label, knn=15)
                predict_class_aggregate_louvain, df_accuracy_louvain, best_louvain_labels, knn_opt, onevsall_opt_louvain, maj_truth_labels, time_end_knn, time_end_prune, time_end_louvain, time_end_total, f1_accumulated, f1_mean, time_end_knn_query, num_edges = ls.run_mainlouvain(X_txt, true_label, self_loop=False, keep_all_dist= True,jac_weighted_edges = weighted, jac_std=jac_std, dist_std=dist_std, small_pop=10, knn_in=5)
                write_list_to_file(best_louvain_labels, Alph_label_file_name)
                write_list_to_file(true_label, True_label_file_name)
                print('saved files')
            #predict_class_aggregate_louvain_5, df_accuracy_louvain_5pop, best_louvain_labels_pop5, knn_opt, onevsall_opt_louvain, maj_truth_labels, time_end_knn, time_end_prune, time_end_louvain, time_end_total, f1_accumulated, f1_mean, time_end_knn_query, num_edges = ls.run_mainlouvain(
            #    X_txt, true_label, self_loop=False, keep_all_dist=True, jac_weighted_edges=weighted, jac_std=jac_std,                dist_std=dist_std, small_pop=5)
            print('end alph at', time.ctime())

            '''
            weighted = not weighted
            print('now weighted is', weighted, 'jac:', jac_std)
            predict_class_aggregate_louvain, df_accuracy_louvain, best_louvain_labels, knn_opt, onevsall_opt_louvain, maj_truth_labels, time_end_knn, time_end_prune, time_end_louvain, time_end_total, f1_accumulated, f1_mean, time_end_knn_query, num_edges = ls.run_mainlouvain(
                X_txt, true_label, self_loop=False, keep_all_dist=False, jac_weighted_edges=weighted, jac_std=jac_std,
                dist_std=dist_std, small_pop=10)
            weighted = not weighted
            jac_std = 'median'
            print('now weighted is', weighted, 'jac:', jac_std)
            predict_class_aggregate_louvain, df_accuracy_louvain, best_louvain_labels, knn_opt, onevsall_opt_louvain, maj_truth_labels, time_end_knn, time_end_prune, time_end_louvain, time_end_total, f1_accumulated, f1_mean, time_end_knn_query, num_edges = ls.run_mainlouvain(
                X_txt, true_label, self_loop=False, keep_all_dist=False, jac_weighted_edges=weighted, jac_std=jac_std,
                dist_std=dist_std, small_pop=10)
            weighted = not weighted
            print('now weighted is', weighted, 'jac:', jac_std)
            predict_class_aggregate_louvain, df_accuracy_louvain, best_louvain_labels, knn_opt, onevsall_opt_louvain, maj_truth_labels, time_end_knn, time_end_prune, time_end_louvain, time_end_total, f1_accumulated, f1_mean, time_end_knn_query, num_edges = ls.run_mainlouvain(
                X_txt, true_label, self_loop=False, keep_all_dist=False, jac_weighted_edges=weighted, jac_std=jac_std,
                dist_std=dist_std, small_pop=10)
            '''
            df_accuracy_louvain.to_excel(writer, 'alph_small10', index=False)
            #df_accuracy_louvain_5pop.to_excel(writer, 'alph_small5', index=False)
            writer.save()
            print('alph time elapsed', time.time() - time_start_alph)



            majority_truth_labels_louvain = maj_truth_labels #np.zeros(len(true_label))
            print('number of clusters:', len(set(best_louvain_labels)),set(best_louvain_labels))
            best_louvain_labels = np.asarray(best_louvain_labels)

            #run Phenograph
            # for knn_k in [30]:
            #     time_start_pheno = time.time()
            #     predict_class_aggregate_pheno, df_accuracy_pheno, pheno_labels, onevsall_opt, majority_truth_labels_pheno, pheno_time, f1_acc_noweighting_pheno = ls.run_phenograph(X_txt, true_label, knn=knn_k)
            #
            #     df_accuracy_pheno.to_excel(writer, 'pheno_knn'+str(knn_k), index=False)
            #     #write_list_to_file(pheno_labels,Pheno_label_file_name)
            #     writer.save()
            #     #print('ari for pheno with', len(set(pheno_labels)), 'groups',
            #     #  adjusted_rand_score(np.asarray(true_label), pheno_labels))
            #     print('pheno time elapsed', time.time() - time_start_pheno)
            #     for onevsall_val in list(set(true_label)):
            #         vals_roc, predict_class_array, maj, num_onevsall_vall_clusters = ls.accuracy_mst(pheno_labels, true_label,embedding_filename=None, clustering_algo='louvain', onevsall=onevsall_val)
            #         f1_current = vals_roc[1]
            #         print(f1_current, 'is the f1-score of target', onevsall_val)
            #
            #     for cluster_i in set(best_louvain_labels):
            #         cluster_i_loc = np.where(best_louvain_labels == cluster_i)[0]
            #         majority_truth = func_mode(list(true_label[cluster_i_loc]))
            #         majority_truth_labels_louvain[cluster_i_loc] = majority_truth

            #'''

            df_all['majority_vote_class_louvain'] = majority_truth_labels_louvain
            print('majority truth', majority_truth_labels_louvain[0:10], majority_truth_labels_louvain[20000:20010])
            df_all['cluster_louvain'] = best_louvain_labels
            #print(best_louvain_labels)
            df_all_heatmap_louvain = df_all.sort_values(['majority_vote_class_louvain', 'cluster_louvain'])
            feat_cols.remove('Dry mass var')
            df_heatmap_louvain =df_all_heatmap_louvain[feat_cols]
            df_sample_imagelist = get_SampleImageIDs(best_louvain_labels, df_all, true_label)
            df_sample_imagelist.to_excel(writer, 'louvain_images', index=False)
            writer.save()
        #     #start COMMENT OUT 1
        #
        #     #print(feat_cols)
        #     #print(df_heatmap_louvain.max())
            fig, (ax0,ax1) = plt.subplots(1, 2, figsize=(36, 12))
            #ax0.pcolor(df_heatmap_louvain,vmin=-4, vmax=4)
            #ax0.set_xticks(np.arange(0.5, len(df_heatmap_louvain.columns), 1))
            #ax0.set_xticklabels(df_heatmap_louvain.columns)
            ##ax0.set_yticks(np.arange(0.5, len(df_heatmap_louvain.index), 1))
            ##ax0.set_yticklabels(df_all_heatmap_louvain['cluster_louvain'].values[0::200]
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
            #ax0.set_yticks(ytickloc)
            #ax0.set_yticklabels(ynewlist)
            #ax0.grid(axis ='y',color="w", linestyle='-', linewidth=2)
            ##ax0.set_yticklabels(np.arange(0.5, len(df_heatmap_louvain.index), 100), df_all_heatmap_louvain['cluster_louvain'].values[0::100])
            ##ax0.set_xticklabels(np.arange(0.5, len(df_heatmap_louvain.columns), 1), df_heatmap_louvain.columns)
            #ax0.set_title('ALPH Heatmap: cell level')

            #ax0.tick_params(axis='x', rotation = 45)
            ##[l.set_visible(False) for (i, l) in enumerate(plt.xticks()) if i % nn != 0]
            ##plt.locator_params(axis='y', nbins=10)

            df_all_mean_clusters_louvain = df_all_heatmap_louvain.groupby('cluster_louvain',as_index=False)[feat_cols+['majority_vote_class_louvain']].mean()
            print('shape of df all mean', df_all_mean_clusters_louvain.shape)
            print(df_all_mean_clusters_louvain['majority_vote_class_louvain'])
            df_all_mean_clusters_louvain = df_all_mean_clusters_louvain.sort_values(['majority_vote_class_louvain', 'cluster_louvain'])
            print(df_all_mean_clusters_louvain)
            df_all_mean_clusters_louvain['population'] = ''
            for cluster_i in set(best_louvain_labels):
                cluster_i_loc = np.where(best_louvain_labels == cluster_i)[0]
                population_cluster_i = len(cluster_i_loc)
                df_all_mean_clusters_louvain.loc[df_all_mean_clusters_louvain.cluster_louvain == cluster_i, 'population'] = population_cluster_i


            # df_all_mean_clusters_louvain=df_all_mean_clusters_louvain.loc[df_all_mean_clusters_louvain['population'] >400]
            # df_all_mean_clusters_louvain = df_all_mean_clusters_louvain.loc[
            #     df_all_mean_clusters_louvain['population'] != 1362]
            # df_all_mean_clusters_louvain = df_all_mean_clusters_louvain.loc[
            #     df_all_mean_clusters_louvain['population'] != 725]
            # print(df_all_mean_clusters_louvain.shape, df_all_mean_clusters_louvain.head())
            # display_columns = ['Area', 'Circularity', 'Attenuation density', 'Amplitude var', 'Amplitude skewness',
            #      'Amplitude kurtosis', 'Dry mass', 'Dry mass density', 'Peak phase',
            #      'Phase skewness', 'Phase kurtosis', 'DMD contrast 1', 'Mean phase arrangement', 'Phase orientation kurtosis', 'Focus factor 1', 'Focus factor 2']
            df_mean_clusters_louvain = df_all_mean_clusters_louvain[feat_cols] #display cols for very concise

            #hmap = ax1.pcolor(df_mean_clusters_louvain, vmin=-1, vmax=1)
            #ax1.set_xticks(np.arange(0.5, len(df_heatmap_louvain.columns), 1))
            #ax1.set_xticklabels(df_heatmap_louvain.columns)
            #ax1.set_yticks(np.arange(0, len(df_all_mean_clusters_louvain.index), 1))
            #ax1.set_yticklabels(ynewlist)
            #ax1.grid(axis = 'y', color="w", linestyle='-', linewidth=2)
            ##ax1.set_yticklabels(df_all_mean_clusters_louvain['cluster_louvain'])
            ##ax1.set_yticklabels(np.arange(0.5, len(df_all_mean_clusters_louvain), 1),df_all_mean_clusters_louvain['cluster_louvain'])
            ##ax1.set_xticklabels(np.arange(0.5, len(df_mean_clusters_louvain.columns), 1), df_mean_clusters_louvain.columns)
            #ax1.set_title('ALPH Heatmap: cluster level')
            #ax1.tick_params(axis='x', rotation=45)
            #plt.colorbar(hmap)
            #plt.savefig(plot_name + 'alph_heatmap.tif', dpi=350)
            df_mean_clusters_louvain = df_mean_clusters_louvain.apply(lambda x: [y if (y <= 1) else 1 for y in x])
            df_mean_clusters_louvain = df_mean_clusters_louvain.apply(lambda x: [y if (y > -1) else -1 for y in x])
            import seaborn as sns
            cmap_div = sns.diverging_palette(240, 10, as_cmap=True)
            g = sns.clustermap(df_mean_clusters_louvain, row_cluster=False, col_cluster=True, cmap=cmap_div)
            plt.savefig(plot_name + 'alph_clustermap_noxlabels.tif', dpi=350)
            new_row_order = [item.get_text() for item in g.ax_heatmap.yaxis.get_majorticklabels()]
            print(new_row_order,'clustermap row order')

            df_sample_new_order_imagelist_louvain = get_SampleImageIDs_matchClusterMap(best_louvain_labels, df_all,
                                                                                       true_label, new_row_order)
            df_sample_new_order_imagelist_louvain.to_excel(writer, 'ALPH clustermap images', index=False)
            writer.save()

            g.ax_row_dendrogram.set_visible(False)
            g.ax_col_dendrogram.set_visible(False)
            g.cax.set_visible(False)

            plt.setp(g.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
            # plt.setp(g.ax_heatmap.xaxis.get_majorticklabels(), rotation=0)
            col = g.ax_col_dendrogram.get_position()
            g.ax_heatmap.set_position([col.x0 * 0.2, col.y0 * 0.5, col.width * 1.2, col.height * 4])

            plt.savefig(plot_name + 'alph_clustermap.tif', dpi=350)
            plt.show()

        #
        #
        #
        #
        from MulticoreTSNE import MulticoreTSNE as multicore_tsne
        import random
        subsample_rate = 1# used 10 for entire LC dataset in PARC

        alph_labels_array = np.asarray(best_louvain_labels)
        true_label_array = np.asarray(true_label)
        # false_v = np.zeros((len(alph_labels), 1), dtype=bool)
        first_pass = True
        for label_i in set(best_louvain_labels):
            v = alph_labels_array == label_i
            v_where = np.where(v)[0].tolist()
            shuffle = random.sample(v_where, round(len(v_where) / subsample_rate))
            # false_v[shuffle] = 1
            if first_pass == True:
                X_final = X_txt[shuffle]
                label_final_array = alph_labels_array[shuffle]
                true_label_final = true_label_array[shuffle]
                first_pass = False
            else:
                X_final = np.concatenate((X_final, X_txt[shuffle]))
                label_final_array = np.concatenate((label_final_array, alph_labels_array[shuffle]))
                true_label_final = np.concatenate((true_label_final,true_label_array[shuffle]))
        print('subsampled dims', X_final.shape, label_final_array.shape)

        counts = np.bincount(label_final_array)
        to_remove = np.where(counts < 10)[0]  # which label values to remove. for parc paper we used 1000 on full dataset
        print(to_remove)
        if len(to_remove) > 0:
            for i in to_remove:
                print(np.sum(label_final_array == i))
                label_final_array[label_final_array == i] = -1
                print('changing labels')
            dummy, label_final_array = np.unique(label_final_array, return_inverse=True)
            label_final_array -= 1  # keep -1 labels the same
        idx_to_keep = np.where(label_final_array != -1)[0]
        print(len(idx_to_keep), 'labels to keep')

        true_labels_tokeep = true_label_final[idx_to_keep].tolist()
        labels_tokeep = label_final_array[idx_to_keep].tolist()
        X_final = X_final[idx_to_keep,:]
        print('dimensions of true, labels, X_data_tokeep', len(true_labels_tokeep), set(true_labels_tokeep), len(labels_tokeep), set(labels_tokeep), X_final.shape)
        time_start_lv = time.time()
        tsne = multicore_tsne(n_jobs=8, perplexity=30, verbose=1, n_iter=500, learning_rate=1000, angle=0.5)
        X_LV_embedded = tsne.fit_transform(X_final)

        #params_lv = 'lr =1, perp = ' + str(perplexity)
        #X_LV_embedded, lv_embedding_filename, lv_runtime, lv_embedding_plot_title = pp.run_lv(version=None, input_data=X_txt, perplexity=perplexity,    lr=1, new_file_name=new_file_name, new_folder_name=None,outdim=3)
        from scipy import stats
        X_LV_embedded= stats.zscore(X_LV_embedded, axis =0)
        # ##df_accuracy_kmeans_lv, best_labels_kmeans_lv, onevsall_opt_kmeans_lv = run_kmeans(X_LV_embedded[:,:2], true_label,df_all)
        # ##df_accuracy_kmeans_lv.to_excel(writer, 'kmeans_lv', index=False)
        # for final_small_pop_i in [20]:
        #     df_accuracy_mst_lv, best_labels_mst_lv, sigma_opt_lv, min_clustersize_mst_lv, tooclose_factor_opt, onevsall_opt_mst_lv,majority_truth_labels_mst_lv,f1_accumulated, f1_mean = multiclass_mst_accuracy(
        #         X_LV_embedded, true_label,df_all,av_peaks=0,peak_threshhold=1, final_small_pop=final_small_pop_i)
        #         X_LV_embedded, true_label,df_all,av_peaks=0,peak_threshhold=1, final_small_pop=final_small_pop_i)
        #     df_accuracy_mst_lv.to_excel(writer, 'mst_lv'+str(final_small_pop_i), index=False)
        #     write_list_to_file(best_labels_mst_lv,mst_label_file_name)
        #
        # ##df_sample_imagelist = get_SampleImageIDs(best_labels_mst_lv, df_all,true_label)
        # ##df_sample_imagelist.to_excel(writer, 'mst_images', index=False)
        # ##df_sample_imagelist.to_excel(writer_mst, 'mst_images', index=False)
        #
        #
        # dict_time = {'lv runtime': [lv_runtime],
        #              'lv params': [params_lv]}  # , ' bh runtime': [tsne_runtime],  'bh params': [params_tsne]}
        # df_time = pd.DataFrame(dict_time)
        # df_time.to_excel(writer, 'embedding time', index=False)
        # writer.save()
        # print('successfully saved excel file')
        # # if perplexity == perplexity_range[0]:
        # #     fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(36, 12))
        # #     df_all['majority_vote_class_APT_p30'] = majority_truth_labels_mst_lv
        # #     df_all['cluster_APT_p30'] = best_labels_mst_lv
        # #     df_all_heatmap_APT = df_all.sort_values(['majority_vote_class_APT_p30', 'cluster_APT_p30'])
        # #     df_heatmap_APT = df_all_heatmap_APT[feat_cols]
        # #     ax0.pcolor(df_heatmap_APT,vmin=-4, vmax=4)
        # #     ylist = df_all_heatmap_APT['cluster_APT_p30'].values
        # #     ynewlist = []
        # #     maj = ylist_majority[0]
        # #     if maj == 0: maj_str = 'h2170'
        # #     if maj == 1: maj_str = 'h1975'
        # #     if maj == 2: maj_str = 'h526'
        # #     if maj == 3: maj_str = 'h520'
        # #     if maj == 4: maj_str = 'h358'
        # #     if maj == 5: maj_str = 'h69'
        # #     if maj == 6: maj_str = 'hcc827'
        # #     ynewlist.append('cluster ' + str(int(ylist[0])) + ' ' + maj_str)
        # #     ytickloc = [0]
        # #     for i in range(len(ylist) - 1):
        # #         # if ylist[i+1] == ylist[i]: ynewlist.append('')
        # #         if ylist[i + 1] != ylist[i]:
        # #             maj = ylist_majority[i + 1]
        # #             if maj == 0: maj_str = 'h2170'
        # #             elif maj == 1: maj_str = 'h1975'
        # #             elif maj == 2: maj_str = 'h526'
        # #             elif maj == 3: maj_str = 'h520'
        # #             elif maj == 4: maj_str = 'h358'
        # #             elif maj == 5: maj_str = 'h69'
        # #             elif maj == 6: maj_str = 'hcc827'
        # #             else: print('no matching majority val')
        # #             ynewlist.append('cluster ' + str(int(ylist[i + 1])) + ' ' + maj_str)
        # #             ytickloc.append(int(i + 1))
        # #     ax0.set_yticks(ytickloc)
        # #     ax0.set_yticklabels(ynewlist)
        # #     #ax0.set_yticks(np.arange(0.5, len(df_all_heatmap_APT.index), 200))
        # #     #ax0.set_yticklabels(df_all_heatmap_APT['cluster_APT_p30'].values[0::200])
        # #     ax0.set_xticklabels(df_heatmap_APT.columns)
        # #     ax0.set_xticks(np.arange(0.5, len(df_heatmap_APT.columns), 1))
        # #     '''
        # #     ax0.set_yticklabels(np.arange(0.5, len(df_all_heatmap_APT.index), 100), df_all_heatmap_APT['cluster_APT_p30'].values[0::100])
        # #     ax0.set_xticklabels(np.arange(0.5, len(df_heatmap_APT.columns), 1), df_heatmap_APT.columns)
        # #     '''
        # #     ax0.set_title('APT Heatmap: cell level')
        # #     ax0.tick_params(axis='x', rotation=45)
        # #     ax0.grid(axis = 'y', color="w", linestyle='-', linewidth=2)
        # #     df_all_mean_clusters_APT = df_all_heatmap_APT.groupby('cluster_APT_p30', as_index=False)[feat_cols+['majority_vote_class_APT_p30']].mean()
        # #     df_all_mean_clusters_APT = df_all_mean_clusters_APT.sort_values(['majority_vote_class_APT_p30', 'cluster_APT_p30'])
        # #     df_mean_clusters_APT = df_all_mean_clusters_APT[feat_cols]
        # #
        # #     ax1.pcolor(df_mean_clusters_APT, vmin=-4, vmax=4)
        # #
        # #     ax1.set_yticks(np.arange(0, len(df_all_mean_clusters_APT), 1))
        # #     ax1.set_yticklabels(ynewlist)
        # #
        # #     ax1.set_xticks(np.arange(0.5, len(df_mean_clusters_APT.columns), 1))
        # #     ax1.set_xticklabels(df_mean_clusters_APT.columns)
        # #
        # #     '''
        # #     ax1.set_yticklabels(np.arange(0.5, len(df_all_mean_clusters_APT), 1),df_all_mean_clusters_APT['cluster_APT_p30'])
        # #     ax1.set_xticklabels(np.arange(0.5, len(df_mean_clusters_APT.columns), 1), df_mean_clusters_APT.columns)
        # #     '''
        # #     ax1.tick_params(axis='x', rotation=45)
        # #     ax1.grid(axis = 'y', color="w", linestyle='-', linewidth=2)
        # #     ax1.set_title('APT Heatmap: cluster level')
        # #
        # #     plt.savefig(plot_name + 'APT_heatmap.png')
        #
        #
        #
        # ##df_accuracy_dbscan_lv, dbscan_best_labels_lv, eps_opt_lv, dbscan_min_clustersize_lv, tooclose_factor_opt_lv, onevsall_opt_dbscan_lv = run_dbscan(
        # ##    X_LV_embedded, true_label)
        # ##df_accuracy_dbscan_lv.to_excel(writer, 'dbscan_lv', index=False)
        #
        # plot_all_methods(X_LV_embedded, true_label, embedding_filename=plot_name+'all', dbscan_labels=dbscan_best_labels_lv,
        #                  mst_labels=best_labels_mst_lv, louvain_labels=best_louvain_labels,
        #                  pheno_labels=None, kmeans_labels=best_labels_kmeans_lv, onevsall_mst=onevsall_opt_mst_lv,
        #                  onevsall_dbscan=onevsall_opt_dbscan_lv, onevsall_louvain=onevsall_opt_louvain,
        #                  onevsall_pheno=None,
        #                  onevsall_kmeans=onevsall_opt_kmeans_lv, dimred='lv', sigma_opt=sigma_opt_lv,
        #                  eps_opt=eps_opt_lv, min_cluster_opt=min_clustersize_mst_lv,
        #                  dbscan_min_clustersize=dbscan_min_clustersize_lv,
        #                  knn_opt=knn_opt)

        #
        # # plot_all_methods(X_LV_embedded, true_label, embedding_filename=plot_name+'APT', dbscan_labels=None,
        # #                  mst_labels=best_labels_mst_lv, louvain_labels=None,
        # #                  pheno_labels=None, kmeans_labels=None, onevsall_mst=onevsall_opt_mst_lv,
        # #                  onevsall_dbscan=None, onevsall_louvain=None,
        # #                  onevsall_pheno=None,
        # #                  onevsall_kmeans=None, dimred='lv', sigma_opt=sigma_opt_lv,
        # #                  eps_opt=None, min_cluster_opt=min_clustersize_mst_lv,
        # #                  dbscan_min_clustersize=None,
        # #                  knn_opt=None)
        if perplexity == perplexity_range[0]:
            plot_all_methods(X_LV_embedded, true_labels_tokeep, embedding_filename=plot_name + '_PLOTALPH', dbscan_labels=None,
                             mst_labels=None, louvain_labels=labels_tokeep,
                             pheno_labels=None, kmeans_labels=None, onevsall_mst=0,
                             onevsall_dbscan=None, onevsall_louvain=0,
                             onevsall_pheno=None,
                             onevsall_kmeans=None, dimred='lv', sigma_opt=None, eps_opt=None, min_cluster_opt=None,
                             dbscan_min_clustersize=None,
                             knn_opt=30)
        #     # plot_all_methods(X_LV_embedded, true_label, embedding_filename=plot_name + '_Pheno', dbscan_labels=None,
        #     #                  mst_labels=None, louvain_labels=pheno_labels, pheno_labels=None, kmeans_labels=None, onevsall_mst=onevsall_opt_mst_lv,
        #     #                  onevsall_dbscan=None, onevsall_louvain=onevsall_opt_louvain,
        #     #                  onevsall_pheno=None,
        #     #                  onevsall_kmeans=None, dimred='lv', sigma_opt=None, eps_opt=None, min_cluster_opt=None,
        #     #                  dbscan_min_clustersize=None,
        #     #                  knn_opt=30)
        # '''
        # Plotting_3D.save_anim('APT',X_LV_embedded, true_label, embedding_filename=plot_name, dbscan_labels=None,
        #                  mst_labels=best_labels_mst_lv, louvain_labels=None,
        #                  pheno_labels=None, kmeans_labels=None, onevsall_mst=onevsall_opt_mst_lv,
        #                  onevsall_dbscan=None, onevsall_louvain=None,
        #                  onevsall_pheno=None,
        #                  onevsall_kmeans=None, dimred='lv', sigma_opt=sigma_opt_lv,
        #                  eps_opt=None, min_cluster_opt=min_clustersize_mst_lv,
        #                  dbscan_min_clustersize=None,
        #                  knn_opt=None)
        #
        # if perplexity == perplexity_range[0]:
        #     Plotting_3D.save_anim('ALPH',X_LV_embedded, true_label, embedding_filename=plot_name+'ALPH', dbscan_labels=None,
        #                      mst_labels=None, louvain_labels=best_louvain_labels,
        #                      pheno_labels=None, kmeans_labels=None, onevsall_mst=0,
        #                      onevsall_dbscan=None, onevsall_louvain=0,
        #                      onevsall_pheno=None,
        #                      onevsall_kmeans=None, dimred='lv', sigma_opt=None, eps_opt=None, min_cluster_opt=None,
        #                      dbscan_min_clustersize=None,
        #                      knn_opt=30)
        # '''
        # '''
        # time_start = time.time()
        # print('starting tsne', time.ctime())
        # learning_rate_bh =2000
        # if n_total >500000: learning_rate_bh = 2500
        # if n_total > 1000000: learning_rate_bh = 3500
        # params_tsne = 'n_jobs=8, perplexity = ' + str(perplexity) + ' ,verbose=1,n_iter=1000,learning_rate =' + str(learning_rate_bh)
        # tsne = multicore_tsne(n_jobs=8, perplexity=perplexity, verbose=1, n_iter=1000, learning_rate=learning_rate_bh, angle=0.2) #the default angle should be 0.5 and is prob adequate
        #
        # X_embedded = tsne.fit_transform(X_txt)
        # print(X_embedded.shape)
        # tsne_runtime = time.time() - time_start
        # print(params_tsne, new_file_name, '\n',' BH done! Time elapsed: {} seconds'.format(tsne_runtime))
        # df_accuracy_kmeans, temp_best_labels_kmeans, onevsall_opt_kmeans = run_kmeans(X_embedded, true_label, df_all)
        #
        # df_accuracy_mst_bh, temp_best_labels_mst, sigma_opt, min_clustersize, tooclose_factor_opt,onevsall_opt_mst= multiclass_mst_accuracy(X_embedded, true_label)
        # df_accuracy_mst_bh.to_excel(writer, 'mst_bh', index=False)
        # df_accuracy_dbscan, dbscan_best_labels, eps_opt, dbscan_min_clustersize, tooclose_factor_opt,onevsall_opt_dbscan = run_dbscan(X_embedded, true_label)
        # df_accuracy_dbscan.to_excel(writer, 'dbscan_bh', index=False)
        # df_accuracy_kmeans.to_excel(writer, 'kmeans_bh', index=False)
        #
        # plot_all_methods(X_embedded, true_label, embedding_filename=plot_name, dbscan_labels=dbscan_best_labels, mst_labels=temp_best_labels_mst, louvain_labels=best_louvain_labels,
        #                  pheno_labels=None, kmeans_labels = temp_best_labels_kmeans, onevsall_mst=onevsall_opt_mst, onevsall_dbscan=onevsall_opt_dbscan,onevsall_louvain=onevsall_opt_louvain,onevsall_pheno= None, onevsall_kmeans = onevsall_opt_kmeans,dimred='bh', sigma_opt= sigma_opt, eps_opt = eps_opt, min_cluster_opt = min_clustersize,dbscan_min_clustersize = dbscan_min_clustersize, knn_opt=knn_opt)
        # '''
        # '''
        # plot_all_methods(X_embedded, true_label, embedding_filename=plot_name+'3sigma_mst', dbscan_labels=dbscan_best_labels_tooclose20,
        #                  mst_labels=temp_best_labels_mst_s3, louvain_labels=best_louvain_labels,
        #                  pheno_labels=None, kmeans_labels=temp_best_labels_kmeans,
        #                  onevsall_mst=onevsall_best_mst_s3, onevsall_dbscan=onevsall_opt_dbscan_tooclose20,
        #                  onevsall_louvain=onevsall_opt_louvain, onevsall_pheno=None,
        #                  onevsall_kmeans=onevsall_opt_kmeans, dimred='bh', sigma_opt=sigma_opt_s3, eps_opt=eps_opt_tooclose20,
        #                  min_cluster_opt=min_clustersize_s3, dbscan_min_clustersize=dbscan_min_clustersize, knn_opt=knn_opt)
        # '''
        # #predict_class_aggregate_pheno, df_accuracy_pheno, best_pheno_labels, onevsall_opt_pheno= ls.run_phenograph(X_txt,true_label)
        # #df_accuracy_pheno.to_excel(writer, 'pheno', index=False)
        #
        # '''
        # plot_all_methods(X_LV_embedded, true_label, embedding_filename=plot_name + '3sigma_mst',
        #                  dbscan_labels=dbscan_best_labels_tooclose20_lv,
        #                  mst_labels=temp_best_labels_mst_lv_s3, louvain_labels=best_louvain_labels,
        #                  pheno_labels=None, kmeans_labels=best_labels_kmeans_lv, onevsall_mst=onevsall_best_mst_lv_s3,
        #                  onevsall_dbscan=onevsall_opt_dbscan_tooclose20_lv, onevsall_louvain=onevsall_opt_louvain,
        #                  onevsall_pheno=None, onevsall_kmeans=onevsall_opt_kmeans_lv, dimred='lv', sigma_opt=sigma_opt_lv_s3,
        #                  eps_opt=eps_opt_tooclose20_lv, min_cluster_opt=min_clustersize_mst_lv_s3,
        #                  dbscan_min_clustersize=dbscan_min_clustersize_lv,
        #                  knn_opt=knn_opt)
        # '''






def main():
    import random
    print('time now is', time.ctime())
    import random
    for n_sample in [50000]:#[6000,13000,18000,19000,23000,35000,50000]:#[1000, 5000, 6000, 7000, 8000, 9000, 10000,11000,12000,13000,14000,15000,16000,17000,18000,19000,20000,21000,22000,23000,24000,25000,30000,35000,40000,45000]:
        for randomseedval in random.sample(range(1, 1000), 1):#random.randint(1,1000)
            randomseedval1 = 209#1234 552 is used in paper when clustering ALL cells
            run_main('PARC_rand209_VolOnly_nov5' + str(randomseedval1),  n_eachsubtype=n_sample, randomseedval=randomseedval1)
            #run_main('RareH1975_LC_PARC_25Oct' + str(randomseedval1) + 'JacMedianSmall20_', n_eachsubtype=70000, randomseedval=randomseedval1)
        #random.randint(1,1000)
        print('file randomseed val: ', randomseedval1, 'all LC cells')
    #run_main('LCJAN_avpeaks_Jul19_4pm_Randomseed'+str(randomseedval)+'_', n_eachsubtype =15000, randomseedval = randomseedval)

    #run_main('LCJAN_avpeaks_Jul19_4pm_Randomseed'+str(randomseedval)+'_', n_eachsubtype =50000, randomseedval = randomseedval)
    #run_main('LCJAN_avpeaks_Jul19_4pm_Randomseed' + str(randomseedval) + '_', n_eachsubtype=None, randomseedval=randomseedval)
    #run_main('LCJAN_3D_Oct15'+'_', n_eachsubtype=5000, randomseedval=randomseedval)
    #run_main('LC_Phenograph_Jan' + '_', n_eachsubtype=1000, randomseedval=randomseedval)

    #run_main('LC_may6_Randomseed_leiden_noprune' + str(randomseedval) + '_', n_eachsubtype=None, randomseedval=randomseedval)

def main1():
    import csv
    true_label=[]
    FlowSom_label = []
    label_volume = []
    label_no_volume = []
    for jac_std in ['median']:#,0.05,0.1,0.15,0.2,0.25,0.3]:
        label_volume=[]
    #with open("/home/shobi/Thesis/MultiClass_MinCluster/Feb 2019/LC_ALPH_Feb19_randint123_N70000.txt", 'rt') as f:
    #with open('/home/shobi/Thesis/Rcode/FLOCK/Flock_results_LungCancerData_RareTarget1975_N30100RandInt711_May.txt', 'rt') as f:
        with open("/home/shobi/Thesis/MultiClass_MinCluster/April 2019/LC_209_rareH1975_N281604_SeuratOct22K10P15.txt", 'rt') as f:
        #with open('/home/shobi/Thesis/MultiClass_MinCluster/April 2019/VolumeFeatures_LC_ALPH_AllCells1234JacMedianSmall20__N1113369.txt', 'rt') as f:
        #with open('/home/shobi/Thesis/Rcode/FlowPeaks/FlowPeaks_semiauto_labels_LC_LungCancerData_RareTarget1975_N30100RandInt888_May.txt','rt') as f:
        #with open('/home/shobi/Thesis/Rcode/my_first_R_project/PBMC_68K/10X_FlowSOM_Autok_20grid_v2.txt', 'rt') as f:
        #with open('/home/shobi/Thesis/Rcode/FlowSOM/LungCancerData_N1113369RandInt789_Feb_FlowSOM_18k_20grid_v1.txt', 'rt') as f:
        #with open("/home/shobi/Thesis/MultiClass_MinCluster/Feb 2019/LC_ALPH_Jan28_randint672_N70000.txt", 'rt') as f:
            next(f)
            for line in f:
                #line = line.strip().replace('\"', '')
                #line = line.strip().replace('\"', '')
                #FlowSom_label.append(int(float(line)))
                label_volume.append(int(line))
        #print('there are',len(set(FlowSom_label)), 'clusters')
        print('there are', len(set(label_volume)), 'vol clusters')
    '''    
    with open('/home/shobi/Thesis/MultiClass_MinCluster/April 2019/NoVolumeFeatures_LC_ALPH_AllCells1234JacMedianSmall20__N1113369.txt',
              'rt') as f:

        for line in f:
            # line = line.strip().replace('\"', '')
            label_no_volume.append(int(float(line)))

    # print('there are',len(set(FlowSom_label)), 'clusters')
    print('there are', len(set(label_no_volume)), 'no-vol clusters')
    '''
    #with open('/home/shobi/Thesis/Rcode/Feb2019/LungCancerData_TrueLabel_N1113369RandInt789_Feb.txt', 'rt') as f:
    #with open('/home/shobi/Thesis/Rcode/my_first_R_project/PBMC_68K/annotations_zhang.txt', 'rt') as f:
    #with open('/home/shobi/Thesis/Rcode/FlowSOM/LungCancerData_RareTarget1975_TrueLabel_N6100RandInt964_May.txt', 'rt') as f:
    #with open('/home/shobi/Thesis/Rcode/FlowSOM/LungCancerData_TrueLabel_N1113369RandInt789_Feb.txt',              'rt') as f:

    with open('/home/shobi/Thesis/MultiClass_MinCluster/April 2019/true_label_Oct22AllFeatures_LC_PARC_22Oct873JacMedianSmall20__N281604.txt','rt') as f:
        #next(f)
        for line in f:
            line = line.strip().replace('\"', '')
            #true_label.append(line) USE FOR PBMC 10x
            true_label.append(int(float(line)))

    print('length of true labels and vol labels',len(true_label),len(label_volume))
    true_label = pd.Series(true_label)
    print('ari for vol and truelabel',  'groups', adjusted_rand_score(np.asarray(true_label), label_volume))
    #print('ari for no-vol and truelabel', 'groups', adjusted_rand_score(np.asarray(true_label), label_no_volume))
    #print('ari for no-vol and vol', 'groups', adjusted_rand_score(np.asarray(label_volume), label_no_volume))
    #print("Adjusted Mutual Information: %0.5f"       % metrics.adjusted_mutual_info_score(true_label, FlowSom_label))
    targets = list(set(true_label))
    if len(targets) >= 2:
        target_range = targets
    else:
        target_range = [1]
    N = len(true_label)
    f1_accumulated = 0
    vol=True
    for label_i in [label_volume]:#, label_no_volume]:
        f1_mean = 0
        for onevsall_val in list(set(true_label)):
            print('here')
            #print('target is', onevsall_val)
            vals_roc, predict_class_array, maj,numclusters_targetval  = ls.accuracy_mst(label_i, true_label,
                                                                 embedding_filename=None, clustering_algo='louvain',
                                                                 onevsall=onevsall_val)
            print('here2')
            f1_current = vals_roc[1]
            #print('for target', onevsall_val, 'the f1-score is', f1_current)
            f1_accumulated = f1_accumulated + f1_current * (list(true_label).count(onevsall_val)) / N
            f1_mean = f1_current+f1_mean
            print('f1-score for target', onevsall_val, 'is', f1_current)
        #print(f1_accumulated, ' f1 accumulated (weighted by population of sub population) for FlowSom')
        if vol ==True: print('f1-mean for Volume-Labels is',f1_mean/len(targets))
        else: print('f1-mean for No-Volume-Labels is',f1_mean/len(targets))
        vol=False
def main1():
    n_eachsubtype = 5500
    randomseedval = 10
    df_all, true_label, X_txt, feat_cols = get_data(fluor=0, n_eachsubtype=n_eachsubtype, randomseedval=randomseedval)
    print('type true label', type(true_label))
    p1 = parc.PARC(X_txt, true_label=true_label)
    p1.run_PARC()
    print(p1.f1_mean, p1.f1_accumulated)
    print(p1.stats_df['f1-score'])
    p1 = parc.PARC(X_txt)
    p1.run_PARC()

    print('----------')
    p1 = parc.PARC(X_txt)
    p1.run_PARC()

    print('----------')
    p1 = parc.PARC(X_txt)
    p1.run_PARC()

    p1 = parc.PARC(X_txt, true_label=true_label)
    p1.run_PARC()




if __name__ == '__main__':
    main()