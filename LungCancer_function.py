from MulticoreTSNE import MulticoreTSNE as multicore_tsne  # https://github.com/DmitryUlyanov/Multicore-TSNE
import Louvain_igraph as ls
import Performance_phenograph as pp
import numpy as np
import time
import scipy.io
import pandas as pd
import matplotlib.pyplot as plt
from MST_clustering_mergetooclose import MSTClustering
from pandas import ExcelWriter
from Louvain_igraph import accuracy_mst
import time
import sklearn.cluster
from sklearn.cluster import DBSCAN, KMeans
import os.path


print(os.path.abspath(sklearn.cluster.__file__))

def get_data(fluor=0, n_eachsubtype=None):
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

    df_h2170 = make_subtype_df('h21702018Jan23_gatedH2170',0,n_eachsubtype) #h2170_Raw, 28 x 302,635
    df_h1975 = make_subtype_df('h19752018Jan23_gatedH1975',1,n_eachsubtype) #60447
    df_h526 = make_subtype_df('h5262018Jan03_gatedH526',2,n_eachsubtype)#375889
    df_h520 = make_subtype_df('h5202018Jan03_gatedH520',3,n_eachsubtype)#451208
    df_h358 = make_subtype_df('h3582018Jan03_gatedH358',4,n_eachsubtype)#170198
    df_h69 = make_subtype_df('h692018Jan23_gatedH69',5,n_eachsubtype) #130075
    df_hcc827 = make_subtype_df('hcc8272018Jan23_gatedHcc827',6,n_eachsubtype)

    frames = [df_h2170, df_h1975, df_h526, df_h520, df_h358, df_h69,df_hcc827]
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
    true_label = np.asarray(label_txt)
    true_label = true_label.astype(int)
    print('data matrix size', X_txt.size)
    print('true label shape:', true_label.shape)

    return df_all, true_label, X_txt



def make_subtype_df(subtype_name, class_val,n_eachsubtype=None):
    print('getting ', subtype_name)
    subtype_raw = scipy.io.loadmat(
        '/home/shobi/Thesis/Data/ShobiGatedData/LungCancer_ShobiGatedData_cleanup/'+subtype_name+'.mat')  # 28 x 302,635
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
    df_subtype['cell_tag'] = subtype_name.split('_')[0] + subtype_fileidx["filename"].map(int).map(str) + '_' + \
                              subtype_fileidx["index"].map(int).map(str)
    df_subtype['label'] = 'Monocyte'
    df_subtype['class'] = class_val
    print(df_subtype.head(5))
    if class_val ==6:
        df_subtype = df_subtype.sample(frac=1).reset_index(drop=True)
    if class_val !=6 and n_eachsubtype ==None:
        df_subtype = df_subtype.sample(frac=1).reset_index(drop=True)
    if class_val !=6 and n_eachsubtype !=None:
        if n_eachsubtype < df_subtype.shape[0]:
            df_subtype = df_subtype.sample(frac=1).reset_index(drop=True)[0:n_eachsubtype]
        else: df_subtype = df_subtype.sample(frac=1).reset_index(drop=True)
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

    fig, ax = plt.subplots(3, 2, figsize=(24, 24), sharex=True, sharey=True)
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
    for k in range(N):
        x = X_plot[k, 0]
        y = X_plot[k, 1]
        X_dict.setdefault(mst_labels[k], []).append((x, y)) #coordinates of the points by mst groups
        Index_dict.setdefault(mst_labels[k], []).append(true_label[k]) #true label kth data point grouped by mst_group
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
        ax[0][0].scatter(x, y, color=true_color, s=2, alpha=0.6, label=true_label_str+' Cellcount = ' + str(population))
        ax[0][0].annotate(true_label_str, xytext=(np.mean(x), np.mean(y)), xy=(np.mean(x), np.mean(y)), color='black',
                       weight='semibold')
    box = ax[0][0].get_position()
    ax[0][0].set_position([box.x0, box.y0, box.width *0.9, box.height])
    ax[0][0].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize = 6)
    title_str0 = 'Multi-Class Lung Cancer Cell Lines:Ground Truth. \n'+'Total Cell count is ' +str(N)  # embedding_filename
    ax[0][0].set_title(title_str0, size=10)
    ax[0][1]= plot_onemethod(ax[0][1],X_embedded,mst_labels, true_label,onevsall_mst, 'mst', dimred,sigma_opt, min_cluster_opt, None)
    ax[1][0]= plot_onemethod(ax[1][0],X_embedded,dbscan_labels, true_label,onevsall_dbscan, 'dbscan', dimred,eps_opt, dbscan_min_clustersize, None)
    ax[1][1]= plot_onemethod(ax[1][1],X_embedded,louvain_labels, true_label,onevsall_louvain, 'louvain', dimred,None, None, knn_opt)
    #ax[2][0]= plot_onemethod(ax[2][0],X_embedded,pheno_labels, true_label,onevsall_pheno, 'phenograph', dimred,None, None, 30)
    ax[2][1]= plot_onemethod(ax[2][1],X_embedded,kmeans_labels, true_label,onevsall_kmeans, 'kmeans', dimred,None, None, None)



    plt.savefig(embedding_filename + '_allmethods_' + dimred + '.png', bbox_inches='tight')


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
def multiclass_mst_accuracy(X_embedded, true_label):
    min_clustersize = 20
    sigma_range = [3,2.5,2]
    mergetooclose_factor_range = [50,40,30,20,10]
    f1_temp = -1
    f1_sum_best = 0
    list_roc = []
    targets = list(set(true_label))
    if len(targets) > 2:
        target_range = targets
    else:
        target_range = [1]
    for i_sigma in sigma_range:
        for i_tooclosefactor in mergetooclose_factor_range:
            model = MSTClustering(approximate=True, min_cluster_size=min_clustersize,
                                  sigma_factor=i_sigma, tooclosefactor=i_tooclosefactor)
            time_start = time.time()
            print('Starting MST Clustering', time.ctime())
            model.fit_predict(X_embedded)
            clustering_labels = model.labels_
            runtime_mst = time.time() - time_start
            if i_sigma ==3 and i_tooclosefactor ==20:
                temp_best_labels_s3 = clustering_labels
                sigma_opt_s3 = i_sigma
                tooclose_factor_opt_s3 = i_tooclosefactor
                min_clustersize_s3 = 20


            f1_sum = 0
            for onevsall_val in target_range:
                vals_roc, predict_class_array = accuracy_mst(clustering_labels, true_label,
                                                             embedding_filename=None, clustering_algo='multiclass mst',
                                                             onevsall=onevsall_val)
                f1_sum = f1_sum + vals_roc[0]
                if vals_roc[0] > f1_temp:
                    f1_temp = vals_roc[0]
                    #temp_best_labels = list(model.labels_)
                    #sigma_opt = i_sigma
                    onevsall_val_opt = onevsall_val
                    if i_sigma ==3 and i_tooclosefactor==20: onevsall_best_s3 = onevsall_val
                    #tooclose_factor_opt = i_tooclosefactor
                list_roc.append([i_sigma, min_clustersize, i_tooclosefactor, onevsall_val] + vals_roc + [runtime_mst])

            if f1_sum > f1_sum_best:
                f1_sum_best = f1_sum
                temp_best_labels = list(model.labels_)
                sigma_opt = i_sigma
                tooclose_factor_opt = i_tooclosefactor
                onevsall_best = onevsall_val_opt

    df_accuracy = pd.DataFrame(list_roc,
                           columns=['sigma factor', 'min cluster size','merge-too-close factor', 'onevsall target','error rate', 'f1-score', 'tnr', 'fnr',
                                    'tpr', 'fpr', 'precision', 'recall', 'num_groups', 'runtime'])
    if onevsall_val_opt  == 0: onevsall_opt='h2170'
    if onevsall_val_opt == 1: onevsall_opt='h1975'
    if onevsall_val_opt  == 2: onevsall_opt='h526'
    if onevsall_val_opt  == 3:onevsall_opt='h520'
    if onevsall_val_opt  == 4:onevsall_opt= 'h358'
    if onevsall_val_opt  == 5:onevsall_opt='h69'
    if onevsall_val_opt  == 6:onevsall_opt='hcc827'

    return df_accuracy, temp_best_labels, sigma_opt, min_clustersize, tooclose_factor_opt, onevsall_best,temp_best_labels_s3, sigma_opt_s3, min_clustersize_s3, tooclose_factor_opt_s3, onevsall_best


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
    tooclosefactor_list = [20,10,5,0]
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
    return df_accuracy, temp_best_labels, eps_opt, i_cluster_size, tooclose_factor_opt, onevsall_best, temp_best_labels_tooclose20, eps_opt_tooclose20,onevsall_opt_tooclose20

def run_kmeans(X_embedded, true_label,df_all):
    init_peaks = pp.tsne_densityplot(X_embedded, None, None, df_all, mode='only_vals')
    list_roc = []
    targets = list(set(true_label))
    if len(targets) > 2:
        target_range = targets
    else:
        target_range = [1]
    f1_temp=-1
    n_clusters = init_peaks.shape[0]
    #n_jobs = -3 means all but two CPUs is used for this task

    time_start = time.time()
    model = KMeans(n_clusters=n_clusters, init=init_peaks, n_init=1, verbose = 1, n_jobs=-3).fit(X_embedded)
    runtime = time.time() - time_start

    for onevsall_val in target_range:
        vals_roc, predict_class_array = accuracy_mst(model, true_label,
                                                     embedding_filename=None, clustering_algo='kmeans',onevsall=onevsall_val)
        vals_roc = [onevsall_val]+vals_roc+[runtime]
        list_roc.append(vals_roc)

        if vals_roc[1] > f1_temp:
            f1_temp = vals_roc[1]
            temp_best_labels = list(model.labels_)
            onevsall_val_opt = onevsall_val

    df_accuracy = pd.DataFrame(list_roc,
                       columns=['target','error rate','f1-score', 'tnr', 'fnr',
                                'tpr', 'fpr', 'precision', 'recall', 'num_groups', 'clustering runtime'])

    return df_accuracy, temp_best_labels,onevsall_val_opt

def run_main(new_file_name, n_eachsubtype= None):

    perplexity = 10
    df_all, true_label, X_txt = get_data(fluor = 0,n_eachsubtype = n_eachsubtype)
    n_total = X_txt.shape[0]
    excel_file_name = '/home/shobi/Thesis/MultiClass/ '+new_file_name+'_N'+str(n_total)+'.xlsx'
    plot_name = '/home/shobi/Thesis/MultiClass/'+new_file_name+'_N'+str(n_total)
    print(plot_name)
    writer = ExcelWriter(excel_file_name)
    predict_class_aggregate_louvain, df_accuracy_louvain, best_louvain_labels, knn_opt,onevsall_opt_louvain= ls.run_mainlouvain(X_txt,true_label, self_loop = False)

    df_accuracy_louvain.to_excel(writer, 'louvain', index=False)

    time_start = time.time()
    print('starting tsne', time.ctime())
    learning_rate_bh = 2000
    if n_total >500000: learning_rate_bh = 2500
    if n_total > 1000000: learning_rate_bh = 3500
    params_tsne = 'n_jobs=8, perplexity = ' + str(perplexity)+  ' ,verbose=1,n_iter=1000,learning_rate =' + str(learning_rate_bh) + 'angle = 0.2'
    tsne = multicore_tsne(n_jobs=8, perplexity=perplexity, verbose=1, n_iter=1000, learning_rate=learning_rate_bh, angle = 0.2)

    X_embedded = tsne.fit_transform(X_txt)
    print(X_embedded.shape)
    tsne_runtime = time.time() - time_start
    print(params_tsne, new_file_name, '\n',' BH done! Time elapsed: {} seconds'.format(tsne_runtime))
    df_accuracy_kmeans, temp_best_labels_kmeans, onevsall_opt_kmeans = run_kmeans(X_embedded, true_label, df_all)

    df_accuracy_mst_bh, temp_best_labels_mst, sigma_opt, min_clustersize, tooclose_factor_opt,onevsall_opt_mst,temp_best_labels_mst_s3, sigma_opt_s3, min_clustersize_s3, tooclose_factor_opt_s3, onevsall_best_mst_s3= multiclass_mst_accuracy(X_embedded, true_label)
    df_accuracy_mst_bh.to_excel(writer, 'mst_bh', index=False)
    df_accuracy_dbscan, dbscan_best_labels, eps_opt, dbscan_min_clustersize, tooclose_factor_opt,onevsall_opt_dbscan,dbscan_best_labels_tooclose20, eps_opt_tooclose20,onevsall_opt_dbscan_tooclose20 = run_dbscan(X_embedded, true_label)
    df_accuracy_dbscan.to_excel(writer, 'dbscan_bh', index=False)
    df_accuracy_kmeans.to_excel(writer, 'kmeans_bh', index=False)

    plot_all_methods(X_embedded, true_label, embedding_filename=plot_name, dbscan_labels=dbscan_best_labels, mst_labels=temp_best_labels_mst, louvain_labels=best_louvain_labels,
                     pheno_labels=None, kmeans_labels = temp_best_labels_kmeans, onevsall_mst=onevsall_opt_mst, onevsall_dbscan=onevsall_opt_dbscan,onevsall_louvain=onevsall_opt_louvain,onevsall_pheno= None, onevsall_kmeans = onevsall_opt_kmeans,dimred='bh', sigma_opt= sigma_opt, eps_opt = eps_opt, min_cluster_opt = min_clustersize,dbscan_min_clustersize = dbscan_min_clustersize, knn_opt=knn_opt)

    plot_all_methods(X_embedded, true_label, embedding_filename=plot_name+'3sigma_mst', dbscan_labels=dbscan_best_labels_tooclose20,
                     mst_labels=temp_best_labels_mst_s3, louvain_labels=best_louvain_labels,
                     pheno_labels=None, kmeans_labels=temp_best_labels_kmeans,
                     onevsall_mst=onevsall_best_mst_s3, onevsall_dbscan=onevsall_opt_dbscan_tooclose20,
                     onevsall_louvain=onevsall_opt_louvain, onevsall_pheno=None,
                     onevsall_kmeans=onevsall_opt_kmeans, dimred='bh', sigma_opt=sigma_opt_s3, eps_opt=eps_opt_tooclose20,
                     min_cluster_opt=min_clustersize_s3, dbscan_min_clustersize=dbscan_min_clustersize, knn_opt=knn_opt)
    #predict_class_aggregate_pheno, df_accuracy_pheno, best_pheno_labels, onevsall_opt_pheno= ls.run_phenograph(X_txt,true_label)
    #df_accuracy_pheno.to_excel(writer, 'pheno', index=False)
    time_start = time.time()
    params_lv = 'lr =1, perp = '+ str(perplexity)
    X_LV_embedded, lv_embedding_filename, lv_runtime, lv_embedding_plot_title = pp.run_lv(version=None, input_data=X_txt,perplexity=perplexity, lr=1,  new_file_name=new_file_name, new_folder_name=None)
    lv_runtime = time.time() - time_start

    df_accuracy_kmeans_lv, best_labels_kmeans_lv, onevsall_opt_kmeans_lv = run_kmeans(X_LV_embedded, true_label, df_all)
    df_accuracy_kmeans_lv.to_excel(writer, 'kmeans_lv', index=False)
    df_accuracy_mst_lv, best_labels_mst_lv, sigma_opt_lv, min_clustersize_mst_lv, tooclose_factor_opt, onevsall_opt_mst_lv, temp_best_labels_mst_lv_s3, sigma_opt_lv_s3, min_clustersize_mst_lv_s3, tooclose_factor_opt_lv_s3, onevsall_best_mst_lv_s3 = multiclass_mst_accuracy(
        X_LV_embedded, true_label)
    df_accuracy_mst_lv.to_excel(writer, 'mst_lv', index=False)
    df_accuracy_dbscan_lv, dbscan_best_labels_lv, eps_opt_lv, dbscan_min_clustersize_lv, tooclose_factor_opt_lv, onevsall_opt_dbscan_lv, dbscan_best_labels_tooclose20_lv, eps_opt_tooclose20_lv, onevsall_opt_dbscan_tooclose20_lv = run_dbscan(
        X_LV_embedded, true_label)
    df_accuracy_dbscan_lv.to_excel(writer, 'dbscan_lv', index=False)

    plot_all_methods(X_LV_embedded, true_label, embedding_filename=plot_name, dbscan_labels=dbscan_best_labels_lv,
                     mst_labels=best_labels_mst_lv, louvain_labels=best_louvain_labels,
                     pheno_labels=None, kmeans_labels=best_labels_kmeans_lv, onevsall_mst=onevsall_opt_mst_lv,
                     onevsall_dbscan=onevsall_opt_dbscan_lv, onevsall_louvain=onevsall_opt_louvain, onevsall_pheno=None,
                     onevsall_kmeans=onevsall_opt_kmeans_lv, dimred='lv', sigma_opt=sigma_opt_lv,
                     eps_opt=eps_opt_lv, min_cluster_opt=min_clustersize_mst_lv,
                     dbscan_min_clustersize=dbscan_min_clustersize_lv,
                     knn_opt=knn_opt)
    plot_all_methods(X_LV_embedded, true_label, embedding_filename=plot_name + '3sigma_mst',
                     dbscan_labels=dbscan_best_labels_tooclose20_lv,
                     mst_labels=temp_best_labels_mst_lv_s3, louvain_labels=best_louvain_labels,
                     pheno_labels=None, kmeans_labels=best_labels_kmeans_lv, onevsall_mst=onevsall_best_mst_lv_s3,
                     onevsall_dbscan=onevsall_opt_dbscan_tooclose20_lv, onevsall_louvain=onevsall_opt_louvain,
                     onevsall_pheno=None, onevsall_kmeans=onevsall_opt_kmeans_lv, dimred='lv', sigma_opt=sigma_opt_lv_s3,
                     eps_opt=eps_opt_tooclose20_lv, min_cluster_opt=min_clustersize_mst_lv_s3,
                     dbscan_min_clustersize=dbscan_min_clustersize_lv,
                     knn_opt=knn_opt)




    dict_time = {'lv runtime': [lv_runtime], 'lv params': [params_lv], ' bh runtime': [tsne_runtime],
                 'bh params': [params_tsne]}

    df_time = pd.DataFrame(dict_time)
    df_time.to_excel(writer, 'embedding time', index=False)

    writer.save()
    print('successfully saved excel files')

def main():
    print('time now is', time.ctime())
    run_main('LC_Jun27_3pm_', n_eachsubtype = 1500)

if __name__ == '__main__':
    main()