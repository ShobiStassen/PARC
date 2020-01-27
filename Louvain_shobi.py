
import phenograph
import os
import sys
import LargeVis
import numpy as np
import time
from sklearn.cluster import DBSCAN, KMeans
from MulticoreTSNE import MulticoreTSNE as multicore_tsne
import scipy.io
import pandas as pd
import hnswlib
from scipy.sparse import csr_matrix
import networkx as nx
import community
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
    if fluor == 2:  # all features including fluor when a dataset has incorporated a fluo marker
        df_all[feat_cols_includefluor] = (df_all[feat_cols_includefluor] - df_all[feat_cols_includefluor].mean()) / \
                                         df_all[feat_cols_includefluor].std()
        X_txt = df_all[feat_cols_includefluor].values

    label_txt = df_all['class'].values
    tag_txt = df_all['cell_filename'].values
    print(X_txt.size, label_txt.size)
    true_label = np.asarray(label_txt)
    #true_label = np.reshape(true_label, (true_label.shape[0], 1))
    print('true label shape:', true_label.shape)
    true_label = true_label.astype(int)
    tag = np.asarray(tag_txt)
    tag = np.reshape(tag, (tag.shape[0], 1))
    index_list = list(df_all['index'].values)
    # index_list = np.reshape(index_list,(index_list.shape[0],1))
    # print('index list', index_list)
    #np.savetxt(data_file_name, X_txt, comments='', header=str(n_total) + ' ' + str(int(X_txt.shape[1])), fmt="%f",  delimiter=" ")
    #np.savetxt(label_file_name, label_txt, fmt="%i", delimiter="")
    #np.savetxt(tag_file_name, tag_txt, fmt="%s", delimiter="")
    return true_label, tag, X_txt, new_file_name, df_all, index_list,flist

def func_counter(ll):
    c_0 = ll.count(0)
    c_1 = ll.count(1)
    if c_0 > c_1: return 0
    if c_0 < c_1: return 1
    if c_0 == c_1: return 999

def make_csrmatrix(neighbor_array, distance_array):
    row_list = []
    col_list = []
    dist_list = []
    neighbor_array = neighbor_array
    distance_array = distance_array
    n_neighbors = neighbor_array.shape[1]
    n_cells =  neighbor_array.shape[0]
    rowi = 0
    for row in neighbor_array:
        # print(row)
        for ik in range(n_neighbors):
            row_list.append(rowi)
            col_list.append(row[ik])
            dist_list.append(np.sqrt(distance_array[rowi, ik]))  # making it the same as the minkowski distance
        rowi = rowi + 1

    csr_graph = csr_matrix((np.array(dist_list), (np.array(row_list), np.array(col_list))), shape=(n_cells, n_cells))
    return csr_graph
def accuracy_mst(model, true_labels, embedding_filename, clustering_algo):
    if clustering_algo =='dbscan':
        sigma = model.eps
        min_cluster_size =model.min_samples
        mergetooclosefactor = model.tooclose_factor
    elif clustering_algo =='mst':
        sigma = model.sigma_factor
        min_cluster_size = model.min_cluster_size
        mergetooclosefactor = model.tooclosefactor
    elif clustering_algo =='kmeans':
        sigma = None
        min_cluster_size = None
        mergetooclosefactor = None
    else:
        sigma = None
        min_cluster_size = None
        mergetooclosefactor = None


    X_dict = {}
    Index_dict = {}
    #if clustering_algo =='phenograph': X = X_embedded
    #else: X = model.X_fit_
    #print(X.shape)
    if clustering_algo=='phenograph': mst_labels = list(model)
    if clustering_algo=='louvain': mst_labels = model
    else: mst_labels = list(model.labels_)

    N = len(mst_labels)
    n_cancer = list(true_labels).count(1)
    n_pbmc = list(true_labels).count(0)

    for k in range(N):
        #x = X[k, 0]
        #y = X[k, 1]
        #X_dict.setdefault(mst_labels[k], []).append((x, y))
        Index_dict.setdefault(mst_labels[k], []).append(true_labels[k])
        # Index_dict_dbscan.setdefault(dbscan_labels[k], []).append(true_labels[k])
    # X_dict_dbscan.setdefault(dbscan_labels[k], []).append((x, y))
    num_groups = len(Index_dict)
    sorted_keys = list(sorted(Index_dict.keys()))
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
        if kk==-1:
            len_unknown = len(vals)
            print('len unknown', len_unknown)
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

    #print('f1_score', 'fnr ', fpr,'sigma', ' min cluster size', 'mergetooclose factor', f1_score, fnr, fpr,sigma, min_cluster_size, mergetooclosefactor)
    print('f1_score', 'fnr ', 'fpr', f1_score, fnr, fpr)
    if clustering_algo=='phenograph' or clustering_algo =='louvain': mst_runtime = None
    else: mst_runtime = model.clustering_runtime_
    accuracy_val = [embedding_filename, sigma, min_cluster_size, mergetooclosefactor, f1_score, tnr, fnr, tpr, fpr, precision,
                    recall, num_groups, mst_runtime]
    return accuracy_val, predict_class_array


def louvain_run_main(cancer_type, n_cancer,n_benign):

    n_cancer = n_cancer
    n_benign = n_benign
    ratio = n_benign / n_cancer
    print('the ratio is {}'.format(ratio))
    n_total = n_cancer + n_benign
    num_nn = 30
    cancer_type = cancer_type #'thp1'
    benign_type = 'pbmc'

    fluor = 0
    new_folder_name = cancer_type + '_r{:.2f}'.format(ratio) + '_n' + str(n_cancer)
    path_tocreate = '/home/shobi/Thesis/Louvain_data/' + new_folder_name
    num_dataset_versions = 1
    dataset_version_range = range(num_dataset_versions)

    for dataset_version in dataset_version_range:

        excel_file_name = '/home/shobi/Thesis/Louvain_data/' + new_folder_name + '/LV_excel_' + cancer_type + '_data' + str(
            dataset_version) + '_r{:.2f}'.format(ratio) + '_ncancer' + str(n_cancer) + '.xlsx'
        true_label, tag, X_data, new_file_name, df_all, index_list, flist = get_data(cancer_type, benign_type, n_cancer,
                                                                                     ratio,
                                                                                     fluor, dataset_version,
                                                                                     new_folder_name, method='louvain')
        num_dims = X_data.shape[1]
        n_elements = X_data.shape[0]
        p = hnswlib.Index(space='l2', dim=num_dims)
        p.init_index(max_elements=n_elements, ef_construction=200, M=16)
        time_start_knn = time.time()
        # Element insertion (can be called several times):
        p.add_items(X_data)
        p.set_ef(50)  # ef should always be > k
        # Query dataset, k - number of closest elements (returns 2 numpy arrays)
        print('starting query')
        k_nn = 10
        print('number of k-nn is', k_nn)
        neighbor_array, distance_array = p.knn_query(X_data, k=k_nn)

        print('time elapsed {} seconds'.format(time.time()-time_start_knn))
        print(neighbor_array.shape, distance_array.shape)
        csr_array = make_csrmatrix(neighbor_array, distance_array)
        time_start_nx = time.time()
        print('making NetworkX graph')
        GX = nx.from_scipy_sparse_matrix(csr_array, parallel_edges=False, create_using=None, edge_attribute='weight') #networkx graph from csr matrix
        print('time elapsed {} seconds'.format(time.time()- time_start_nx))
        # first compute the best partition. A length n_total dictionary where each key is a i'th cell, and the dict[key] value is the group/community.
        time_start_louvain = time.time()
        print('starting Louvain clustering at', time.ctime())
        partition = community.best_partition(GX)


        louvain_labels = list(partition.values())
        print(set(louvain_labels))
        print('time elapsed {} seconds'.format(time.time()- time_start_louvain))
        accuracy_val, predict_class_array= accuracy_mst(louvain_labels, true_label, embedding_filename=None, clustering_algo='louvain')
        print('predict shape', )
        colors = [partition[n] for n in GX.nodes()]
        my_colors = plt.cm.viridis  # you can select other color pallettes here: https://matplotlib.org/users/colormaps.html
        pos = nx.spring_layout(GX)
        plt.figure(1)
        nx.draw(GX, pos=pos, node_color=true_label, cmap=my_colors, edge_color="#D4D5CE")
        plt.figure(2)
        nx.draw(GX, pos=pos, node_color=colors, cmap=my_colors, edge_color="#D4D5CE")
        plt.show()
def main():

    louvain_run_main('thp1', n_cancer=50, n_benign=460000)


if __name__ == '__main__':
    main()