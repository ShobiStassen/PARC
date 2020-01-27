'''
Created on 5 May, 2018

@author: shobi
'''
# benchmark speeds on mnist for various tsne implementations: https://github.com/scikit-learn/scikit-learn/issues/10044
# from bhtsnevdm import run_bh_tsne #https://github.com/dominiek/p11ython-bhtsne
from MulticoreTSNE import MulticoreTSNE as multicore_tsne  # https://github.com/DmitryUlyanov/Multicore-TSNE
import Louvain_igraph_Jac24Sept as ls
import LungCancer_function_minClusters_sep10 as LC
import numpy as np
import time
import scipy.io
import pandas as pd
import matplotlib.pyplot as plt
from MST_clustering_mergetooclose import MSTClustering
from pandas import ExcelWriter
import time
import Performance_phenograph as pp
import phenograph
#from MST_3D_4Oct import MSTClustering3D
from MST_3D_current import MSTClustering3D
#from Louvain_igraph import accuracy_mst
from Louvain_igraph_Jac24Sept import accuracy_mst



version = 'Multicore tsne'
lr = 1000
# cancer = 'acc220'
cancer = 'k562_gated'
# cancer = 'fluor_nsclc'
fluor = 0
min_cluster_size = 20
print('min size', min_cluster_size)
sigma_factor = 2

# 0: no fluor
# 1: only fluor
# 2: all features (fluor + non-fluor)
perp = 30

def write_list_to_file(input_list, csv_filename):
    """Write the list to csv file."""

    with open(csv_filename, "w") as outfile:
        for entries in input_list:
            outfile.write(str(entries))
            outfile.write("\n")

def get_data(fluor=0,rnd_int = 0):
    fluor = fluor
    new_file_name = "pbmcsubtypes_trial3"
    new_folder_name = 'PBMC subtypes'
    label_file_name = '/home/shobi/Thesis/BarnesHutMC_data/' + new_folder_name + '/BH_label_' + new_file_name + '.txt'
    tag_file_name = '/home/shobi/Thesis/BarnesHutMC_data/' + new_folder_name + '/BH_tag_' + new_file_name + '.txt'
    data_file_name = '/home/shobi/Thesis/BarnesHutMC_data/' + new_folder_name + '/BH_data_' + new_file_name + '.txt'
    embedding_filename = '/home/shobi/Thesis/MultiClass/BH_embedding_' + new_file_name + '.txt'


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
    #acc220_Raw = scipy.io.loadmat('/home/shobi/Thesis/Data/ShobiGatedData/acc2202017Nov22_gatedAcc220.mat')  # 28 x 416,421
    # print('loaded acc220')
    # pbmc_fluor_raw = scipy.io.loadmat('/home/shobi/Thesis/Data/pbmc_fluor_clean_real.mat') #42,308 x 32
    # nsclc_fluor_raw = scipy.io.loadmat('/home/shobi/Thesis/Data/nsclc_fluor_clean_real.mat') #1,031 x 32
    pbmc_monocyte_Raw = scipy.io.loadmat('/home/shobi/Thesis/Data/ShobiGatedData/monocyte2018Apr30_gatedMonocyte.mat')  # 32*48,831
    pbmct_Raw = scipy.io.loadmat('/home/shobi/Thesis/Data/ShobiGatedData/pbmct2018Apr30_gatedPbmct.mat')  # 32*4,474
    pbmcb_Raw = scipy.io.loadmat('/home/shobi/Thesis/Data/ShobiGatedData/pbmcb2018Apr30_gatedPbmcb.mat')  # 32*890
    pbmcnk_Raw = scipy.io.loadmat('/home/shobi/Thesis/Data/ShobiGatedData/nk2018Apr30_gatedNk.mat')  # 32*7,475

    print('loaded pbmc subtypes')

    print('getting monocyte')
    monocyte_struct = pbmc_monocyte_Raw['monocyte2018Apr30_gatedMonocyte']
    df_monocyte = pd.DataFrame(monocyte_struct[0, 0]['cellparam'].transpose().real)
    monocyte_features = monocyte_struct[0, 0]['cellparam_label'][0].tolist()
    monocyte_fileidx = pd.DataFrame(monocyte_struct[0, 0]['gated_idx'].transpose())
    flist = []
    for element in monocyte_features:
        flist.append(element[0])
    df_monocyte.columns = flist
    monocyte_fileidx.columns = ['filename', 'index']
    print('shape of fileidx', monocyte_fileidx.shape)
    df_monocyte['cell_idx_inmatfile'] = monocyte_fileidx["index"]
    df_monocyte['cell_tag'] = 'monocyte2018Apr30_' + monocyte_fileidx["filename"].map(int).map(str) + '_' + \
                              monocyte_fileidx["index"].map(int).map(str)
    df_monocyte['cell_filename'] = '\\\\Desktop-2v97mic\\E\\2018Apr30pbmcfluo\\monocyte2018Apr30_'+monocyte_fileidx["filename"].map(int).map(str)
    df_monocyte['label'] = 'mono'
    df_monocyte['class'] = 0
    print(df_monocyte.head(5))
    df_monocyte = df_monocyte.sample(frac=1,random_state=rnd_int).reset_index(drop=False)[0:3000]
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
    df_pbmct['cell_filename'] = '\\\\Desktop-2v97mic\\E\\2018Apr30pbmcfluo\\pbmct2018Apr30_'+pbmct_fileidx["filename"].map(int).map(str)
    df_pbmct['cell_idx_inmatfile'] = pbmct_fileidx["index"]
    df_pbmct['cell_tag'] = 'pbmct2018Apr30_' + pbmct_fileidx["filename"].map(int).map(str) + '_' + pbmct_fileidx[
        "index"].map(int).map(str)
    df_pbmct['label'] = 'T-cell'
    df_pbmct['class'] = 1
    print(df_pbmct.head(5))
    df_pbmct = df_pbmct.sample(frac=1,random_state=rnd_int).reset_index(drop=False)
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
    df_pbmcb['cell_filename'] = '\\\\Desktop-2v97mic\\E\\2018Apr30pbmcfluo\\pbmcb2018Apr30_'+ pbmcb_fileidx["filename"].map(int).map(str)
    df_pbmcb['cell_idx_inmatfile'] = pbmcb_fileidx["index"]
    df_pbmcb['cell_tag'] = 'pbmct2018Apr30_' + pbmcb_fileidx["filename"].map(int).map(str) + '_' + pbmcb_fileidx[
        "index"].map(int).map(str)
    df_pbmcb['label'] = 'b-cell'
    df_pbmcb['class'] = 2
    print(df_pbmcb.head(5))
    df_pbmcb = df_pbmcb.sample(frac=1,random_state=rnd_int).reset_index(drop=False)
    print(df_pbmcb.shape)

    print('getting NK-cell')
    pbmcnk_struct = pbmcnk_Raw['nk2018Apr302018_gatedNk']
    df_pbmcnk = pd.DataFrame(pbmcnk_struct[0, 0]['cellparam'].transpose().real)
    pbmcnk_features = pbmcnk_struct[0, 0]['cellparam_label'][0].tolist()
    pbmcnk_fileidx = pd.DataFrame(pbmcnk_struct[0, 0]['gated_idx'].transpose())
    flist = []
    for element in pbmcnk_features:
        flist.append(element[0])
    df_pbmcnk.columns = flist
    pbmcnk_fileidx.columns = ['filename', 'index']
    print('shape of fileidx', pbmcnk_fileidx.shape)
    df_pbmcnk['cell_filename'] = '\\\\Desktop-2v97mic\\E\\2018Apr30pbmcfluo\\nk2018Apr30_'+pbmcnk_fileidx["filename"].map(int).map(str)
    df_pbmcnk['cell_idx_inmatfile'] = pbmcnk_fileidx["index"]
    df_pbmcnk['cell_tag'] = 'nk2018Apr30_' + pbmcnk_fileidx["filename"].map(int).map(str) + '_' + pbmcnk_fileidx[
        "index"].map(int).map(str)
    df_pbmcnk['label'] = 'nk-cell'
    df_pbmcnk['class'] = 3
    print(df_pbmcnk.head(5))
    df_pbmcnk = df_pbmcnk.sample(frac=1,random_state=rnd_int).reset_index(drop=False)[0:6000]
    print(df_pbmcnk.shape)

    frames = [df_monocyte, df_pbmct,df_pbmcb, df_pbmcnk]
    df_all = pd.concat(frames, ignore_index=True,sort=False)

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
    #tag_txt = df_all['cell_filename'].values
    print(X_txt.size, label_txt.size)
    true_label = np.asarray(label_txt)
    #true_label = np.reshape(true_label, (true_label.shape[0], 1))
    print('true label shape:', true_label.shape)
    true_label = true_label.astype(int)
    #tag = np.asarray(tag_txt)
    #tag = np.reshape(tag, (tag.shape[0], 1))
    #index_list = list(df_all['index'].values)
    # index_list = np.reshape(index_list,(index_list.shape[0],1))
    # print('index list', index_list)

    return true_label, X_txt, df_all, feat_cols



def func_mode(ll):  # return MODE of list
    # If multiple items are maximal, the function returns the first one encountered.
    return max(set(ll), key=ll.count)


def func_counter(ll): #for binary classifier
    c_0 = ll.count(0)
    c_1 = ll.count(1)
    if c_0 > c_1: return 0
    if c_0 < c_1: return 1
    if c_0 == c_1: return 999


def plot_mst_simple(model_labels, true_labels, embedding_filename, sigma, min_cluster, onevsall,X_embedded, method,knn_opt=None):
    print(true_labels)
    # http://nbviewer.jupyter.org/github/jakevdp/mst_clustering/blob/master/MSTClustering.ipynb

    X_dict = {}
    X_dict_true = {}
    X_dict_dbscan = {}
    Index_dict = {}
    Index_dict_dbscan = {}

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
    tcells_labels = []
    bcells_labels = []
    nkcells_labels = []
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
            monocytes_labels.append(kk)

        if (majority_val == 1):
            tcells_labels.append(kk)

        if majority_val == 2:
            bcells_labels.append(kk)

        if majority_val == 3:
            nkcells_labels.append(kk)


    error_rate = (fp+fn)/(fp+fn+tn+tp)
    total_error = sum(error_count)/(N)
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
        if true_group ==3:
            true_color = 'darkorange'
            true_label = 'nkcell'
            print('inside group nkcell', true_color)
        elif true_group ==0:
            true_color = 'gray'
            true_label = 'mono'
        elif true_group ==1:
            true_color = 'forestgreen'
            true_label = 'tcell'
        else:
            true_color = 'deepskyblue'
            true_label = 'bcell'
        print('true group', true_group, true_color, true_label)
        x = [t[0] for t in X_dict_true[true_group]]
        y = [t[1] for t in X_dict_true[true_group]]
        population = len(x)
        ax[0].scatter(x, y, color=true_color, s=2, alpha=1, label=true_label+' Cellcount = ' + str(population))
        ax[0].annotate(true_label, xytext=(np.mean(x), np.mean(y)), xy=(np.mean(x), np.mean(y)), color='black',
                       weight='semibold')


        # idx = np.where(color_feature < 5* np.std(color_feature))
        # print(idx[0].shape)
        # print('ckeep shape', c_keep.shape)
        # X_keep = X_data_array[idx[0],:]
        # print('xkeep shape', X_keep.shape)
        # print(c_keep.min(), c_keep.max())
        # s= ax[2].scatter(X_keep[:,0], X_keep[:,1], c =c_keep[:,0], s=4, cmap = 'Reds')
        # cb = plt.colorbar(s)

        # lman = LassoManager(ax[0], data_lasso)
        # ax[0].text(0.95, 0.01, "blue: pbmc", transform=ax[1].transAxes, verticalalignment='bottom', horizontalalignment='right',color='green', fontsize=10)

    colors_monocytes = plt.cm.Greys_r(np.linspace(0.2, 0.6, len(monocytes_labels)))
    colors_tcells = plt.cm.Greens_r(np.linspace(0.2, 0.6, len(tcells_labels)))
    colors_bcells = plt.cm.Blues_r(np.linspace(0.2, 0.6, len(bcells_labels)))
    colors_nkcells = plt.cm.Oranges_r(np.linspace(0.2, 0.6, len(nkcells_labels)))
    pair_color_group_list = [(colors_monocytes, monocytes_labels, ['mono']*len(monocytes_labels)),(colors_tcells, tcells_labels, ['tcells']*len(tcells_labels)),(colors_bcells, bcells_labels, ['bcells']*len(bcells_labels)),(colors_nkcells, nkcells_labels, ['nkcells']*len(nkcells_labels))]
    for color, group, name in pair_color_group_list:
        for color_m, ll_m, label_m in zip(color, group,name):
            x = [t[0] for t in X_dict[ll_m]]
            y = [t[1] for t in X_dict[ll_m]]
            population = len(x)
            ax[1].scatter(x, y, color=color_m, s=2, alpha=1, label= label_m+'_' + str(ll_m) + ' Cellcount = ' + str(len(x)))
            ax[1].annotate(str(ll_m), xytext=(np.mean(x), np.mean(y)), xy=(np.mean(x), np.mean(y)), color='black',
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
    title_str0 = 'Multiclass PBMC subtypes" '#embedding_filename
    if method == 'mst':
        title_str1 = 'MST: mean + ' + str(sigma) + '-sigma cutoff and too_close factor of: ' + str(
        min_cluster) + '\n' + "Error rate is " + " {:.1f}".format(total_error * 100) + '%'
    if method == 'louvain':
        title_str1 = 'Louvain on ' +str(knn_opt)+'-NN graph\n. Error rate is ' + " {:.1f}".format(total_error * 100)
    title_str2 = 'graph layout with cluster populations'

    ax[1].set_title(title_str1, size=10)
    ax[0].set_title(title_str0, size=10)
    #ax[2].set_title(title_str2, size=12)

    box = ax[0].get_position()
    ax[0].set_position([box.x0, box.y0, box.width *0.9, box.height])
    ax[0].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize = 6)
    print('legend 1')
    box1 = ax[1].get_position()
    ax[1].set_position([box1.x0, box1.y0, box1.width * 0.9, box1.height])
    ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=6)
    plt.savefig(embedding_filename+'_' +method+ '.png', bbox_inches='tight')
    #plt.show()
def get_SampleImageIDs(cluster_labels, df_all, true_labels):
    import random
    cluster_i_tag_list = []
    print('set of cluster labels', set(cluster_labels))
    for cluster_i in set(cluster_labels):
        cluster_i_loc = np.where(cluster_labels == cluster_i)[0]
        population_cluster_i = len(cluster_i_loc)
        majority_truth = func_mode(list(true_labels[cluster_i_loc]))
        random.shuffle(cluster_i_loc)
        cluster_i_loc_20= cluster_i_loc[0:20]
        #print('cluster 20',cluster_i_loc_20)
        if population_cluster_i>100:
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
#print(label_txt)
def multiclass_mst_accuracy(X_embedded, true_label, df_all):
    xy_peaks = pp.tsne_densityplot(X_embedded[:, :2], None, None, df_all, mode='only_vals')
    XZ = X_embedded[:, np.ix_([0, 2])]
    XZ = XZ[:, 0, :]
    YZ = X_embedded[:, np.ix_([1, 2])]
    YZ = YZ[:, 0, :]
    print('xz and yz shape', XZ.shape, YZ.shape)
    xz_peaks = pp.tsne_densityplot(XZ, None, None, df_all, mode='only_vals')
    yz_peaks = pp.tsne_densityplot(YZ, None, None, df_all, mode='only_vals')
    av_peaks = round((xy_peaks.shape[0] + yz_peaks.shape[0] + xz_peaks.shape[0]) / 3)
    print('no. peaks,', xy_peaks.shape, xz_peaks.shape, yz_peaks.shape, av_peaks)
    min_clustersize = [30,20]
    f1_temp = -1
    f1_sum_best = 0
    list_roc = []
    targets = list(set(true_label))
    if len(targets) >= 2:
        target_range = targets
    else:
        target_range = [1]
    for i_min_clustersize in min_clustersize:
        model = MSTClustering3D(approximate=True, min_cluster_size=i_min_clustersize, max_labels= av_peaks)
        time_start = time.time()
        print('Starting Clustering', time.ctime())
        model.fit_predict(X_embedded)
        clustering_labels = model.labels_
        runtime_mst = time.time() - time_start

        f1_sum = 0
        for onevsall_val in target_range:
            if onevsall_val ==0:
                onevsall_str = 'mono'
            if onevsall_val ==1:
                onevsall_str = 't-cell'
            if onevsall_val == 2:
                onevsall_str = 'b-cell'
            if onevsall_val == 3:
                onevsall_str = 'nk-cell'

            vals_roc, predict_class_array = accuracy_mst(clustering_labels, true_label,
                                                         embedding_filename=None, clustering_algo='multiclass mst',
                                                         onevsall=onevsall_val)
            print('f1-score, sigma-factor, min_cluster_size, tooclose-factor: ', onevsall_str, model.sigma_factor, i_min_clustersize, model.tooclosefactor)
            f1_sum = f1_sum + vals_roc[1]
            if vals_roc[1] > f1_temp:
                f1_temp = vals_roc[1]
                onevsall_val_opt = onevsall_val

            list_roc.append([model.sigma_factor, i_min_clustersize, model.tooclosefactor, onevsall_val, onevsall_str] + vals_roc + [runtime_mst])

        if f1_sum > f1_sum_best:
            f1_sum_best = f1_sum
            temp_best_labels = list(model.labels_)
            sigma_opt = model.sigma_factor
            tooclose_factor_opt = model.tooclosefactor
            onevsall_best = onevsall_val_opt
            min_clustersize_opt = i_min_clustersize
        majority_truth_labels = np.zeros(len(true_label))

        for cluster_i in set(clustering_labels):
            cluster_i_loc = np.where(clustering_labels == cluster_i)[0]
            population_cluster_i = len(cluster_i_loc)
            majority_truth = func_mode(list(true_label[cluster_i_loc]))
            majority_truth_labels[cluster_i_loc] = majority_truth

    df_accuracy = pd.DataFrame(list_roc,
                           columns=['sigma factor', 'min cluster size','merge-too-close factor', 'onevsall target','celltype','error rate', 'f1-score', 'tnr', 'fnr',
                                    'tpr', 'fpr', 'precision', 'recall', 'num_groups', 'runtime'])


    return df_accuracy, temp_best_labels, sigma_opt, min_clustersize_opt, tooclose_factor_opt, onevsall_best, majority_truth_labels
def run_main(new_file_name, rnd_int):
    excel_file_name = '/home/shobi/Thesis/MultiClass/PBMC/ '+new_file_name+'_rndint'+str(rnd_int)+'.xlsx'
    plot_name = '/home/shobi/Thesis/MultiClass/PBMC/'+new_file_name+'_rndint'+str(rnd_int)
    print('random seedval is', rnd_int)
    writer = ExcelWriter(excel_file_name)
    true_label, X_data, df_all,feat_cols = get_data(fluor = 0,rnd_int=rnd_int)
    '''
    df = pd.DataFrame(X_data)
    print('dims of data', X_data.shape)
    #df.to_csv("/home/shobi/Thesis/Rcode/LungCancerData.txt", header=None, index=None)
    np.savetxt("/home/shobi/Thesis/Rcode/PBMC_5000mono.txt",X_data,delimiter=',')
    np.savetxt("/home/shobi/Thesis/Rcode/PBMC_5000mono_TrueLabel.txt", true_label, delimiter=',')
    print('len true label', len(true_label))
    
    '''
    '''
    print('start phenograph')
    predict_class_aggregate, df_accuracy_pheno, pheno_labels, onevsall_opt_pheno, majority_truth_labels_pheno, pheno_time, f1_acc_noweighting_pheno = ls.run_phenograph(X_data,true_label)
    print('finish phenograph with mean f1-score (unweighted) at', f1_acc_noweighting_pheno, 'and ',len(set(pheno_labels)),'groups')
    df_accuracy_pheno.to_excel(writer, 'Phenograph', index=False)
    '''
    print('start alph')
    jac_std_list = [1]#0.15
    small_pop = 100# 5 as default
    for jac_stdin jac_std_list:
        predict_class_aggregate, df_accuracy_alph, best_alph_labels, knn_opt, onevsall_opt, maj_vals, time_end_knn, time_end_prune, time_end_louvain, time_end_total,f1_accumulated,f1_mean,time_end_knn_query, num_edges= ls.run_mainlouvain(X_data,true_label, self_loop = False,too_big_factor=0.5, keep_all=True, Jac_std=jac_std, small_pop=small_pop)
        print('end alph with ', len(set(best_alph_labels)), 'clusters')
        if jac_std== jac_std_list[0]: df_accuracy = df_accuracy_alph
        else: df_accuracy = df_accuracy.append(df_accuracy_alph, ignore_index=True)
        df_sample_imagelist_louvain = get_SampleImageIDs(best_alph_labels, df_all,true_label)
        df_sample_imagelist_louvain.to_excel(writer,'ALPH images', index=False)

    df_accuracy.to_excel(writer, 'ALPH', index=False)

    targets = list(set(true_label))
    if len(targets) >= 2:
        target_range = targets
    else:
        target_range = [1]
    N = len(true_label)
    f1_accumulated = 0
    f1_mean = 0
    for onevsall_val in target_range:
        print('target is', onevsall_val)
        vals_roc, predict_class_array, maj, numclusters_targetval= ls.accuracy_mst(list(best_alph_labels), true_label,
                                                             embedding_filename=None, clustering_algo='louvain',
                                                             onevsall=onevsall_val)
        f1_current = vals_roc[1]
        f1_accumulated = f1_accumulated + f1_current * (list(true_label).count(onevsall_val)) / N
        f1_mean = f1_current + f1_mean
    print(f1_accumulated, ' f1 accumulated (weighted by population of sub population) for ALPH')
    print('f1-mean is', f1_mean / len(targets))


    writer.save()
    print('successfully saved excel files')
    '''
    majority_truth_labels_louvain = np.zeros(len(true_label))
    for cluster_i in set(best_louvain_labels):
        #print('clusteri', cluster_i)
        cluster_i_loc = np.where(best_louvain_labels == cluster_i)[0]
        #print('loc',cluster_i_loc)
        majority_truth = func_mode(list(true_label[cluster_i_loc]))
        #print(majority_truth)
        majority_truth_labels_louvain[cluster_i_loc] = majority_truth
    '''
    df_all['majority_vote_class_louvain'] = maj_vals
    df_all['cluster_louvain'] = best_alph_labels
    print(best_alph_labels)
    df_all_heatmap_louvain = df_all.sort_values(['majority_vote_class_louvain', 'cluster_louvain'])
    feat_cols.remove('Dry mass var')
    df_heatmap_louvain =df_all_heatmap_louvain[feat_cols]

    df_all_mean_clusters_louvain = df_all_heatmap_louvain.groupby('cluster_louvain',as_index=False)[feat_cols+['majority_vote_class_louvain']].mean()
    df_all_mean_clusters_louvain = df_all_mean_clusters_louvain.sort_values(['majority_vote_class_louvain', 'cluster_louvain'])
    df_mean_clusters_louvain = df_all_mean_clusters_louvain[feat_cols]
    df_mean_clusters_louvain = df_mean_clusters_louvain.apply(lambda x: [y if (y <= 1) else 1 for y in x])
    df_mean_clusters_louvain = df_mean_clusters_louvain.apply(lambda x: [y if (y > -1) else -1 for y in x])
    import seaborn as sns
    #g = sns.clustermap(df_mean_clusters_louvain, cmap='viridis',row_cluster=False)
    #plt.savefig(plot_name + 'alph_clustermap_colsonly.jpg')

    g = sns.clustermap(df_mean_clusters_louvain, cmap='viridis')
    new_row_order=[item.get_text() for item in g.ax_heatmap.yaxis.get_majorticklabels()]
    # plt.setp(g.ax_heatmap.get_xticklabels(), rotation=45)
    df_sample_new_order_imagelist_louvain = get_SampleImageIDs_matchClusterMap(best_alph_labels, df_all, true_label, new_row_order)
    df_sample_new_order_imagelist_louvain.to_excel(writer, 'ALPH clustermap images', index=False)
    g.ax_row_dendrogram.set_visible(False)
    g.ax_col_dendrogram.set_visible(False)
    g.cax.set_visible(False)

    plt.setp(g.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
    #plt.setp(g.ax_heatmap.xaxis.get_majorticklabels(), rotation=0)
    col = g.ax_col_dendrogram.get_position()
    g.ax_heatmap.set_position([col.x0*0.2, col.y0*0.5, col.width*1.2, col.height*4])
    writer.save()
    plt.savefig(plot_name + 'alph_clustermap.jpg')
    plt.show()

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(36, 12))
    ax0.pcolor(df_heatmap_louvain, vmin=-1, vmax=1, cmap='viridis')
    ax0.set_xticks(np.arange(0.5, len(df_heatmap_louvain.columns), 1))
    ax0.set_xticklabels(df_heatmap_louvain.columns)
    # ax0.set_yticks(np.arange(0.5, len(df_heatmap_louvain.index), 1))
    # ax0.set_yticklabels(df_all_heatmap_louvain['cluster_louvain'].values[0::200]
    ylist = df_all_heatmap_louvain['cluster_louvain'].values
    ylist_majority = df_all_heatmap_louvain['majority_vote_class_louvain'].values
    ynewlist = []
    maj = ylist_majority[0]
    if maj == 0: maj_str = 'mono'
    if maj == 1: maj_str = 'T-cell'
    if maj == 2: maj_str = 'B-cell'
    if maj == 3: maj_str = 'NK'

    ynewlist.append('cluster ' + str(int(ylist[0])) + ' ' + maj_str)
    ytickloc = [0]
    for i in range(len(ylist) - 1):
        # if ylist[i+1] == ylist[i]: ynewlist.append('')
        if ylist[i + 1] != ylist[i]:
            maj = ylist_majority[i + 1]
            if maj == 0: maj_str = 'mono'
            if maj == 1: maj_str = 'T-cell'
            if maj == 2: maj_str = 'B-cell'
            if maj == 3: maj_str = 'NK'
            ynewlist.append('cluster ' + str(int(ylist[i + 1])) + ' ' + maj_str)
            ytickloc.append(int(i + 1))
    ax0.set_yticks(ytickloc)
    ax0.set_yticklabels(ynewlist)
    ax0.grid(axis='y', color="w", linestyle='-', linewidth=2)
    # ax0.set_yticklabels(np.arange(0.5, len(df_heatmap_louvain.index), 100), df_all_heatmap_louvain['cluster_louvain'].values[0::100])
    # ax0.set_xticklabels(np.arange(0.5, len(df_heatmap_louvain.columns), 1), df_heatmap_louvain.columns)
    ax0.set_title('ALPH Heatmap: cell level')

    ax0.tick_params(axis='x', rotation=45)
    # [l.set_visible(False) for (i, l) in enumerate(plt.xticks()) if i % nn != 0]
    # plt.locator_params(axis='y', nbins=10)

    ax1.pcolor(df_mean_clusters_louvain, vmin=-1, vmax=1,cmap='viridis') #-4
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
    #df_sample_imagelist = get_SampleImageIDs(best_alph_labels, df_all,true_label)
    #df_accuracy_alph.to_excel(writer, 'ALPH', index=False)
    #df_sample_imagelist.to_excel(writer, 'louvain_images', index=False)
    plt.show()
    plt.savefig(plot_name + 'louvain_heatmap.png')
    time_start = time.time()
    print('starting tsne', time.ctime())

    '''
    tsne = multicore_tsne(n_jobs=8, perplexity=30, verbose=1, n_iter=1000, learning_rate=lr, angle = 0.2)
    params = 'n_jobs=8, perplexity = 30,verbose=1,n_iter=1000,learning_rate =' + str(lr)

    X_embedded = tsne.fit_transform(X_txt)
    print(X_embedded.shape)
    print(params, new_file_name, '\n',' BH done! Time elapsed: {} seconds'.format(time.time() - time_start))
    '''
    import Performance_minClusters_rarecell as pp
    X_embedded, embedding_filename, tsne_runtime, embedding_plot_title = pp.run_lv(0, X_data,
                                                                                perplexity=30, lr=1,#was perp50
                                                                                new_file_name=new_file_name,
                                                                                new_folder_name='', outdim=3)

    import Plotting_3D as Plotting_3D
    Plotting_3D.plotPBMC_3D(best_alph_labels, true_label,
                            plot_name + str(new_file_name) +'ALPH with jac_std: ' + str(jac_std) , sigma=None, min_cluster=None, onevsall=onevsall_opt,
                            X_embedded=X_embedded, method='louvain')
    #Plotting_3D.plotPBMC_3D(, true_label,
    #                        plot_name + str(new_file_name) +' Phenograph', sigma=None, min_cluster=None, onevsall=onevsall_opt,
    #                        X_embedded=X_embedded, method='louvain')
    too_big_factor_i = 0.4
    df_running = pd.DataFrame()
    for av_peaks_i in [0,40,35,30,25,20,15,10]:
        for min_cluster_size_i in [10,20]:
            df_accuracy_mst_lv, best_labels_mst_lv, sigma_opt_lv, min_clustersize_mst_lv, tooclose_factor_opt, onevsall_opt_mst_lv, majority_truth_labels_mst,f1_accumulated, f1_mean = LC.multiclass_mst_accuracy(
                    X_embedded, true_label, df_all,av_peaks =av_peaks_i,peak_threshhold=2 ,min_clustersize=min_cluster_size_i, too_big_factor = too_big_factor_i)
            write_list_to_file(best_labels_mst_lv,
                               '/home/shobi/Thesis/MultiClass/PBMC/APT'+new_file_name + 'toobig' + str(
                                   too_big_factor_i * 100) + 'Perp' + str(perp) + '.txt')
            df_running = pd.concat([df_running, df_accuracy_mst_lv])
            targets = list(set(true_label))
            if len(targets) >= 2:
                target_range = targets
            else:
                target_range = [1]
            N = len(true_label)
            f1_accumulated = 0
            f1_mean = 0
            for onevsall_val in target_range:
                print('target is', onevsall_val)
                vals_roc, predict_class_array, maj, numclusters_targetval_apt = ls.accuracy_mst(list(best_labels_mst_lv), true_label,
                                                                     embedding_filename=None, clustering_algo='louvain',
                                                                     onevsall=onevsall_val)
                f1_current = vals_roc[1]
                f1_accumulated = f1_accumulated + f1_current * (list(true_label).count(onevsall_val)) / N
                f1_mean = f1_current + f1_mean
            print(f1_accumulated, ' f1 accumulated (weighted by population of sub population) for APT')
            print('f1-mean is', f1_mean / len(targets))
            df_running.to_excel(writer,'mst_stats', index=False)
            #df_sample_imagelist = get_SampleImageIDs(best_labels_mst_lv, df_all,true_label)
            #df_sample_imagelist.to_excel(writer, 'mst_images', index=False)

            writer.save()
            print('saving MST labels')
            fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(36, 12))
            df_all['majority_vote_class_APT_p50'] = majority_truth_labels_mst
            df_all['cluster_APT_p50'] = best_labels_mst_lv
            df_all_heatmap_APT = df_all.sort_values(['majority_vote_class_APT_p50', 'cluster_APT_p50'])
            df_heatmap_APT = df_all_heatmap_APT[feat_cols]
            ax0.pcolor(df_heatmap_APT, vmin=-4, vmax=4)
            ylist = df_all_heatmap_APT['cluster_APT_p50'].values
            ynewlist = []
            maj = ylist_majority[0]
            if maj == 0: maj_str = 'mono'
            if maj == 1: maj_str = 'T-cell'
            if maj == 2: maj_str = 'B-cell'
            if maj == 3: maj_str = 'NK'
            ynewlist.append('cluster ' + str(int(ylist[0])) + ' ' + maj_str)
            ytickloc = [0]
            for i in range(len(ylist) - 1):
                # if ylist[i+1] == ylist[i]: ynewlist.append('')
                if ylist[i + 1] != ylist[i]:
                    maj = ylist_majority[i + 1]
                    if maj == 0: maj_str = 'mono'
                    if maj == 1: maj_str = 'T-cell'
                    if maj == 2: maj_str = 'B-cell'
                    if maj == 3: maj_str = 'NK'
                    ynewlist.append('cluster ' + str(int(ylist[i + 1])) + ' ' + maj_str)
                    ytickloc.append(int(i + 1))
            ax0.set_yticks(ytickloc)
            ax0.set_yticklabels(ynewlist)
            # ax0.set_yticks(np.arange(0.5, len(df_all_heatmap_APT.index), 200))
            # ax0.set_yticklabels(df_all_heatmap_APT['cluster_APT_p50'].values[0::200])
            ax0.set_xticklabels(df_heatmap_APT.columns)
            ax0.set_xticks(np.arange(0.5, len(df_heatmap_APT.columns), 1))
            '''
            ax0.set_yticklabels(np.arange(0.5, len(df_all_heatmap_APT.index), 100), df_all_heatmap_APT['cluster_APT_p50'].values[0::100])
            ax0.set_xticklabels(np.arange(0.5, len(df_heatmap_APT.columns), 1), df_heatmap_APT.columns)
            '''
            ax0.set_title('APT Heatmap: cell level')
            ax0.tick_params(axis='x', rotation=45)
            ax0.grid(axis='y', color="w", linestyle='-', linewidth=2)
            df_all_mean_clusters_APT = df_all_heatmap_APT.groupby('cluster_APT_p50', as_index=False)[
                feat_cols + ['majority_vote_class_APT_p50']].mean()
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
            ax1.grid(axis='y', color="w", linestyle='-', linewidth=2)
            ax1.set_title('APT Heatmap: cluster level')

            plt.savefig(plot_name + 'APT_heatmap.png')
            #true_label = true_label.astype(int)
            #tag = np.asarray(tag_txt)
            #tag = np.reshape(tag, (tag.shape[0], 1))
            # true_labels = np.vstack((np.zeros((X0.shape[0],1)),np.ones((X1.shape[0],1))))
            # X01 = np.vstack((X0,X1))
            #np.savetxt(data_file_name, X_txt, comments='', header=str(n_total) + ' ' + str(int(X_txt.shape[1])), fmt="%f",delimiter=" ")
            #np.savetxt(embedding_file_name, X_embedded, comments='', header=str(n_total) + ' ' + str(int(X_embedded.shape[1])),fmt="%f", delimiter=" ")
            #np.savetxt(label_file_name, label_txt, fmt="%i", delimiter="")
            #np.savetxt(tag_file_name, tag_txt, fmt="%s", delimiter="")

            new_folder_name = ''
            time_start = time.time()
            # n_neighbors = 100
            # default n_neighbors is 20 but changing to 50 or 100 does not change the accuracy

            #model = MSTClustering3D(approximate=True, min_cluster_size=15, max_labels=av_peaks)
            #model.fit_predict(X_embedded)

            # model_dbscan = DBSCAN(eps = 0.1, min_samples = 5)
            # dbscan_labels = model_dbscan.fit_predict(X_data)
            print('MST done! Time elapsed: {} seconds'.format(time.time() - time_start))

            if(min_cluster_size_i==10 and av_peaks_i==0): Plotting_3D.plotPBMC_3D(best_alph_labels,true_label, plot_name+'_avpeaks'+str(av_peaks_i)+str(new_file_name)+'_smallpop'+str(min_cluster_size_i), sigma= None, min_cluster =None,  onevsall = onevsall_opt,X_embedded = X_embedded, method = 'louvain')
            Plotting_3D.plotPBMC_3D(best_labels_mst_lv, true_label, plot_name+'_avpeaks'+str(av_peaks_i)+str(new_file_name)+'_smallpop'+str(min_cluster_size_i), sigma_opt_lv, tooclose_factor_opt,onevsall = onevsall_opt_mst_lv, X_embedded = X_embedded, method ='mst')

def main():
    import random
    rnd_int = 672#random.randint(1,1000)
    run_main('jan28_5000Monocytes_6000NK_10Samples',rnd_int=rnd_int)
def main1():
    from sklearn.metrics.cluster import adjusted_rand_score
    from sklearn import metrics
    true_label = []
    FlowSom_label = []
    with open('/home/shobi/Thesis/Rcode/PBMC_5000mono_FlowSOM_35k_20grid_v1.txt', 'rt') as f:
        next(f)
        for line in f:
            line = line.strip().replace('\"', '')
            FlowSom_label.append(int(float(line)))
    with open('/home/shobi/Thesis/Rcode/PBMC_5000mono_TrueLabel.txt', 'rt') as f:
        for line in f:
            line = line.strip().replace('\"', '')
            true_label.append(int(float(line)))

    print(len(true_label), true_label[:10])
    true_label = pd.Series(true_label)
    print('ari for kmeans with 20 ', 'groups', adjusted_rand_score(np.asarray(true_label), FlowSom_label))
    print("Adjusted Mutual Information: %0.5f" % metrics.adjusted_mutual_info_score(true_label, FlowSom_label))
    targets = list(set(true_label))
    if len(targets) >= 2:
        target_range = targets
    else:
        target_range = [1]
    N = len(true_label)
    f1_accumulated = 0
    f1_mean = 0
    for onevsall_val in target_range:
        print('target is', onevsall_val)
        vals_roc, predict_class_array, maj, numclusters_targetval = ls.accuracy_mst(list(FlowSom_label), true_label,
                                                             embedding_filename=None, clustering_algo='louvain',
                                                             onevsall=onevsall_val)
        f1_current = vals_roc[1]
        f1_accumulated = f1_accumulated + f1_current * (list(true_label).count(onevsall_val)) / N
        f1_mean = f1_current+f1_mean
    print(f1_accumulated, ' f1 accumulated (weighted by population of sub population) for FlowSom')
    print('f1-mean is',f1_mean/len(targets))


if __name__ == '__main__':
    main()