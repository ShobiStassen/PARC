from sklearn.cluster import DBSCAN
import phenograph
import os
import sys
import LargeVis
import copy
import numpy as np

from sklearn.cluster import DBSCAN, KMeans
from MulticoreTSNE import MulticoreTSNE as multicore_tsne
import scipy.io
import pandas as pd
from scipy import stats
from pandas import ExcelWriter
import hnswlib
from scipy.sparse import csr_matrix
import igraph as ig
import louvain #https://github.com/vtraag/louvain-igraph
import matplotlib.pyplot as plt
# from mst_clustering import MSTClustering
from MST_clustering_mergetooclose import MSTClustering
import time
from sklearn import metrics
from sklearn.neighbors import KernelDensity
from scipy import stats
from pandas import ExcelWriter
from itertools import groupby as g
from scipy.stats import boxcox
from collections import Counter

def Most_Common(lst):
    data = Counter(lst)
    return data.most_common(1)[0][0]
#using this as of Mon OCt 29
# 0: no fluor
# 1: only fluor
# 2: all features (fluor + non-fluor)

def get_data(cancer_type, benign_type, n_cancer, ratio, fluor, dataset_number, new_folder_name, method, randomseedval):
    print('randomseed val ', randomseedval)
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

    #print('loaded pbmc')
    # MCF7_Raw = scipy.io.loadmat('/home/shobi/Thesis/Data/MCF7_clean_real.mat') #32 x 306,968

    if benign_type == 'pbmc':
        #print('constructing dataframe for ', benign_type)
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
        #print('shape of fileidx', pbmc_fileidx.shape)
        df_pbmc['cell_filename'] = 'pbmc2017Nov22_' + pbmc_fileidx["filename"].map(int).map(str)
        df_pbmc['cell_idx_inmatfile'] = pbmc_fileidx["matlabindex"].map(int).map(str)
        df_pbmc['cell_tag']='pbmc2017Nov22_' + pbmc_fileidx["filename"].map(int).map(str)+'midx'+pbmc_fileidx["matlabindex"].map(int).map(str)
        df_pbmc['label'] = 'PBMC'
        df_pbmc['class'] = 0
        df_benign = df_pbmc.sample(frac=1,random_state = randomseedval).reset_index(drop=False)[0:n_pbmc]
        # print(df_benign.head(5))
        #print(df_benign.shape)

    # pbmc_fluor_raw = scipy.io.loadmat('/home/shobi/Thesis/Data/pbmc_fluor_clean_real.mat') #42,308 x 32
    # nsclc_fluor_raw = scipy.io.loadmat('/home/shobi/Thesis/Data/nsclc_fluor_clean_real.mat') #1,031 x 32
    if cancer_type == 'acc220':
        #print('constructing dataframe for ', cancer_type)
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
        #print('shape of fileidx', acc220_fileidx.shape)
        df_acc220['cell_filename'] = 'acc2202017Nov22_' + acc220_fileidx["filename"].map(int).map(str)
        df_acc220['cell_idx_inmatfile'] = acc220_fileidx["matlabindex"].map(int).map(
            str)  # should be same number as image number within that folder
        df_acc220['cell_tag'] = 'acc2202017Nov22_' + acc220_fileidx["filename"].map(int).map(str) + 'midx' + acc220_fileidx[
            "matlabindex"].map(int).map(str)
        df_acc220['label'] = 'acc220'
        df_acc220['class'] = 1
        df_cancer = df_acc220.sample(frac=1,random_state = randomseedval).reset_index(drop=False)[0:n_cancer]
        #print(df_cancer.shape)

    if cancer_type == 'k562':
        #print('constructing dataframe for ', cancer_type)
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
        #print('shape of fileidx', k562_fileidx.shape)
        df_k562['cell_filename'] = 'k5622017Nov08_' + k562_fileidx["filename"].map(int).map(str)
        df_k562['cell_idx_inmatfile'] =  k562_fileidx["matlabindex"].map(int).map(str) #should be same number as image number within that folder
        df_k562['cell_tag'] = 'k5622017Nov08_' + k562_fileidx["filename"].map(int).map(str) + 'midx' + k562_fileidx[
            "matlabindex"].map(int).map(str)
        df_k562['label'] = 'K562'
        df_k562['class'] = 1
        df_cancer = df_k562.sample(frac=1,random_state = randomseedval).reset_index(drop=False)[0:n_cancer]
        #print(df_cancer.shape)

    if cancer_type == 'thp1':
        #print('constructing dataframe for ', cancer_type)
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
        #print('shape of fileidx', thp1_fileidx.shape)
        df_thp1['cell_filename'] = 'thp12017Nov22_' + thp1_fileidx["filename"].map(int).map(str)
        df_thp1['cell_idx_inmatfile'] = thp1_fileidx["matlabindex"].map(int).map(
            str)  # should be same number as image number within that folder
        df_thp1['cell_tag'] = 'thp12017Nov022_' + thp1_fileidx["filename"].map(int).map(str) + 'midx' + thp1_fileidx[
            "matlabindex"].map(int).map(str)
        df_thp1['label'] = 'thp1'
        df_thp1['class'] = 1
        df_cancer = df_thp1.sample(frac=1,random_state = randomseedval).reset_index(drop=False)[0:n_cancer]
        #print(df_cancer.shape)


    # frames = [df_pbmc_fluor,df_nsclc_fluor]
    frames = [df_benign, df_cancer]
    df_all = pd.concat(frames, ignore_index=True)

    # EXCLUDE FLUOR FEATURES
    if fluor == 0:
        df_all[feat_cols] = (df_all[feat_cols] - df_all[feat_cols].mean()) / df_all[feat_cols].std()
        X_txt = df_all[feat_cols].values
        #print('size of data matrix:', X_txt.shape)
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

def run_lv(perplexity, lr, graph_array_name):
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
    LargeVis.loadgraph(graph_array_name)
    X_embedded_LV=LargeVis.run(outdim, threads, samples, prop, alpha, trees, neg, neigh, gamma, perp)
    X_embedded = np.array(X_embedded_LV)
    print('X_embedded shape: ',X_embedded.shape)
    time_elapsed = time.time() - time_start
    return X_embedded
def func_mode(ll):  # return MODE of list
    # If multiple items are maximal, the function returns the first one encountered.
    return max(set(ll), key=ll.count)

def func_counter(ll):
    c_0 = ll.count(0)
    c_1 = ll.count(1)
    if c_0 > c_1: return 0
    if c_0 < c_1: return 1
    if c_0 == c_1: return 999

def make_knn_struct(X_data, ef=100):#100
    num_dims = X_data.shape[1]
    n_elements = X_data.shape[0]
    print('input shape', X_data.shape)
    p = hnswlib.Index(space='l2', dim=num_dims)
    p.init_index(max_elements=n_elements, ef_construction=200, M=30)#M=30 sep8
    time_start_knn = time.time()
    # Element insertion (can be called several times):
    p.add_items(X_data)
    p.set_ef(ef)  # ef should always be > k
    #p.save_index('/home/shobi/Thesis/Louvain_data/test_knnindex.txt')
    return p

def make_csrmatrix_noselfloop(neighbor_array, distance_array,dist_std =0, keep_all_dist= False):
    print('keep all is', keep_all_dist)
    row_list = []
    col_list = []
    weight_list = []
    neighbor_array = neighbor_array #not listed in in any order of proximity
    print('size neighbor array', neighbor_array.shape)
    num_neigh = neighbor_array.shape[1]
    distance_array = distance_array
    n_neighbors = neighbor_array.shape[1]
    n_cells =  neighbor_array.shape[0]
    rowi = 0
    count_0dist =0
    discard_count = 0
    print('dist std factor ', dist_std)
    if keep_all_dist== False: #do some pruning based on distance
        for row in neighbor_array:
            distlist =distance_array[rowi, :]
            #print(distlist)


            to_keep = np.where(distlist < np.mean(distlist)+dist_std*np.std(distlist))[0] #0*std
            updated_nn_ind = row[np.ix_(to_keep)]
            updated_nn_weights = distlist[np.ix_(to_keep)]
            #if len(to_keep)<num_neigh: print('done some distance pruning')
            discard_count = discard_count + (num_neigh-len(to_keep))

            for ik in range(len(updated_nn_ind)):
                if rowi != row[ik]:  # remove self-loops
                    row_list.append(rowi)
                    # col_list.append(row[ik])
                    col_list.append(updated_nn_ind[ik])
                    dist = np.sqrt(updated_nn_weights[ik])
                    # dist = np.sqrt(updated_nn_weights[rowi, ik])# making it the same as the minkowski distance
                    if dist == 0:
                        ## print(distlist)
                        ## dist = (np.mean(distlist) - 0*np.std(distlist))*0.01#0.000001
                        dist = 0.2  # OCT3 we try this
                        count_0dist = count_0dist + 1  # print('rowi and row[ik] are modified to', rowi, row[ik], 'and are',dist, 'dist apart')
                    #weight_list.append(dist)
                    weight_list.append(1 / dist)
                    # if rowi%100000==0: print('distances to 1st and last neighbor:', dist)
            rowi = rowi + 1

    if keep_all_dist== True: #dont prune based on distance
        # print('original nn', len(distlist))
        # to_keep = np.where(distlist <= (np.max(distlist)+1))[0]
        # if len(to_keep)!= num_neigh: print('we are keeping' ,len(to_keep), 'out of ', num_neigh,' neighbors')
        row_list.extend(list(np.transpose(np.ones((n_neighbors, n_cells)) * range(0, n_cells)).flatten()))
        col_list = neighbor_array.flatten().tolist()
        weight_list = (1./(distance_array.flatten()+0.1)).tolist()
        #weight_list = distance_array.flatten().tolist()

        #col_list.extend(row)
        #weight_list.extend(distlist)
        #row_list = []
        #for rowi in list(range(0,n_cells)):
        #    row_list.extend((rowi * np.ones(n_neighbors)).tolist())

        # updated_nn_ind = row
        # updated_nn_weights = distlist

        # print('we are keeping all', len(to_keep))
    # to_keep = np.where(distlist < np.max(distlist))[0]
    # print('to_keep', to_keep)

    # print('updated nn', len(updated_nn_ind))

    # print('distlist update', updated_nn_weights)
    # print('nn update', updated_nn_ind)
    print('share of neighbors discarded in Distance pruning', discard_count,discard_count/neighbor_array.size)
    print('number of zero dist', count_0dist)
    #print('average distances of neighbors', np.mean(weight_list))
    #weight_list = [np.mean(weight) if x==1 else x for x in a]
    from scipy import stats
    csr_graph = csr_matrix((np.array(weight_list), (np.array(row_list), np.array(col_list))), shape=(n_cells, n_cells))
    undirected_graph_array = None
    '''
    if n_neighbors >= 10:
        neighbor_array_noself = np.vstack((np.array(row_list),np.array(col_list))).T
        reverse_neighbor_array_noself =np.fliplr(neighbor_array_noself)
        undirected_neighbor_array = np.concatenate((neighbor_array_noself,reverse_neighbor_array_noself))
        print('undirec neigh array', undirected_neighbor_array.shape )
        weight_array = np.array(weight_list)
        weight_array = np.concatenate((weight_array,weight_array))
        weight_array = np.reshape(weight_array,(weight_array.shape[0],1))
        print('weight array shape', weight_array.shape)
        undirected_graph_array = np.hstack((undirected_neighbor_array,weight_array))
    '''

    return csr_graph, undirected_graph_array
def make_csrmatrix_withselfloop(neighbor_array, distance_array):
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
            if rowi == row[ik]:dist = 0.00001
            else: dist = np.sqrt(distance_array[rowi, ik])# making it the same as the minkowski distance
            dist_list.append(dist)
            #dist_list.append(1 / dist)
            #if rowi%100000==0: print('distances:', dist)
            #if rowi ==row[ik]:
                #print('first neighbor is itself with dist:', dist)
             #
            #if dist ==0: dist = 0.001
            #dist_list.append(1/dist)
        rowi = rowi + 1
    #print('average distances of neighbors', np.mean(dist_list))
    csr_graph = csr_matrix((np.array(dist_list), (np.array(row_list), np.array(col_list))), shape=(n_cells, n_cells))
    return csr_graph
def run_toobig_sublouvain(X_data, knn_struct,k_nn,self_loop = False, keep_all=False,jac_std= 0.3,jac_weighted_edges = False): #0.3 was default dist_std
    n_elements = X_data.shape[0]
    time_start_knn = time.time()
    #X_data_copy = copy.deepcopy(X_data)

    print('number of k-nn is', k_nn)
    neighbor_array, distance_array = knn_struct.knn_query(X_data, k=k_nn)
    # print(neighbor_array, distance_array)

    # print('time elapsed {} seconds'.format(time.time() - time_start_knn))
    print('shapes of neigh and dist array',neighbor_array.shape, distance_array.shape)
    undirected_graph_array_forLV = None
    if self_loop == False:
        csr_array, undirected_graph_array_forLV = make_csrmatrix_noselfloop(neighbor_array, distance_array, keep_all=keep_all, dist_std=0)#was dist_std = default (which is 0 )until Oct11
        #print(undirected_graph_array_forLV)

    if self_loop == True:
        csr_array = make_csrmatrix_withselfloop(neighbor_array, distance_array)
    time_start_nx = time.time()

    sources, targets = csr_array.nonzero()
    # print(len(sources),len(targets))
    mask = np.zeros(len(sources), dtype=bool)
    mask |= (csr_array.data > (np.mean(csr_array.data) + np.std(csr_array.data) * 5))
    print('sum of mask', sum(mask))
    csr_array.data[mask] = 0
    csr_array.eliminate_zeros()
    sources, targets = csr_array.nonzero()
    edgelist = list(zip(sources.tolist(), targets.tolist()))
    edgelist_copy = edgelist.copy()
    print('conversion to igraph', time.ctime())
    G = ig.Graph(edgelist, edge_attrs={'weight': csr_array.data.tolist()})
    print('computing Jaccard metric', time.ctime())
    sim_list = G.similarity_jaccard(pairs=edgelist_copy) #list of jaccard weights
    print('pre-lognormed mean and std', np.mean(sim_list), np.std(sim_list))
    #sim_list = G.similarity_inverse_log_weighted(vertices=edgelist_copy)  #list of jaccard weights
    #sim_list = stats.zscore(boxcox(sim_list,0))#log to get normal distribution
    #sim_list = sim_list +np.abs(np.min(sim_list))
    new_edgelist = []
    sim_list_array = np.asarray(sim_list)
    if jac_std=='median': threshold = np.median(sim_list)
    else: threshold = np.mean(sim_list) - jac_std*np.std(sim_list)
    strong_locs = np.where(sim_list_array>threshold)[0] #if it gets smaller than 0.5 then you get too many tiny tiny clusters of size 2
    for ii in strong_locs: new_edgelist.append(edgelist_copy[ii])
    print(len(sim_list), 'length of sim list before pruning')

    print('percentage of edges KEPT', len(strong_locs) / len(sim_list))
    sim_list_new = list(sim_list_array[strong_locs])
    print(len(sim_list_new), 'length of sim list after pruning')
    if jac_weighted_edges == True:
        G_sim = ig.Graph(list(new_edgelist), edge_attrs={'weight': sim_list_new})
    else: G_sim = ig.Graph(list(new_edgelist))#, edge_attrs={'weight': sim_list_new})
    #G_sim = ig.Graph(list(edgelist), edge_attrs={'weight': sim_list})
    print('average degree of graph is ', np.mean(G_sim.degree()))
    G_sim.simplify(combine_edges='sum')
    print('average degree of SIMPLE graph is ', np.mean(G_sim.degree()), G_sim.degree)
    time_start_louvain = time.time()
    print('starting Louvain clustering at', time.ctime())
    if jac_weighted_edges == True: partition = louvain.find_partition(G_sim, louvain.ModularityVertexPartition, weights='weight')
    else: partition = louvain.find_partition(G_sim, louvain.ModularityVertexPartition)
    small_pop_list = []
    small_cluster_list = []
    louvain_labels = np.empty((n_elements, 1))
    for key in range(len(partition)):
        population = len(partition[key])
        if population < 10:  # 10 for cytof
            #print(key, ' has small population of ', population, )
            small_pop_list.append([t for t in partition[key]])
            small_cluster_list.append(key)
        for cell in partition[key]:
            louvain_labels[cell] = key
    print('list of clusters that have low populiation:', small_cluster_list)
    for small_cluster in small_pop_list:
        for single_cell in small_cluster:
            #print('single cell')
            #print('dim neigh array', neighbor_array.shape)
            old_neighbors = neighbor_array[single_cell, :]
            # print(old_neighbors, 'old neighbors')
            group_of_old_neighbors = louvain_labels[old_neighbors]
            group_of_old_neighbors = list(group_of_old_neighbors.flatten())

            # print('single cell', single_cell, 'has this group_of_old_neighbors', group_of_old_neighbors)
            from statistics import mode
            available_neighbours = set(group_of_old_neighbors) - set(small_cluster_list)
            if len(available_neighbours) > 0:
                available_neighbours_list = [value for value in group_of_old_neighbors if
                                             value in list(available_neighbours)]
                best_group = max(available_neighbours_list, key=available_neighbours_list.count)
                louvain_labels[single_cell] = best_group

            #print('best group is', best_group)
    '''
    louvain_labels = np.empty((n_elements, 1))
    for key in range(len(partition)):
        population = len(partition[key])
        for cell in partition[key]:
            louvain_labels[cell] = key
    '''
    return louvain_labels

def run_sublouvain(X_data, true_label, knn_struct,k_nn,self_loop = False, too_big_factor = 0.15, small_pop = 5, dist_std = 0,keep_all_dist= False,jac_std= 0.3, jac_weighted_edges = False):#50
    time_start_sub = time.time()
    n_elements = X_data.shape[0]
    #X_data_copy = copy.deepcopy(X_data)

    print('number of k-nn is', k_nn, too_big_factor, 'small pop is', small_pop)
    knn_query_start = time.time()
    neighbor_array, distance_array = knn_struct.knn_query(X_data, k=k_nn)
    time_knn_query_end = time.time() - knn_query_start
    #print(neighbor_array, distance_array)

    #print('time elapsed {} seconds'.format(time.time() - time_start_knn))
    print('shapes of neigh and dist array', neighbor_array.shape, distance_array.shape)
    undirected_graph_array_forLV = None
    time_start_prune = time.time()
    if self_loop == False:
        csr_array, undirected_graph_array_forLV = make_csrmatrix_noselfloop(neighbor_array, distance_array, dist_std,keep_all_dist= keep_all_dist)
        #print(undirected_graph_array_forLV)

    if self_loop == True:
        csr_array = make_csrmatrix_withselfloop(neighbor_array, distance_array)
    time_start_nx = time.time()

    sources, targets = csr_array.nonzero()
    edgelist = list(zip(sources,targets))
    '''
    #print(len(sources),len(targets))
    mask = np.zeros(len(sources), dtype=bool)
    mask |= (csr_array.data < (np.mean(csr_array.data) - np.std(csr_array.data)*5))
    print('sum of mask', sum(mask))
    csr_array.data[mask] = 0
    csr_array.eliminate_zeros()
    sources, targets = csr_array.nonzero()
    '''
    #print('EDGELIST', edgelist)
    edgelist_copy = edgelist.copy()
    #print('EDGELIST',edgelist)
    #print('making iGraph')
    G = ig.Graph(edgelist, edge_attrs={'weight': csr_array.data.tolist()})
    print('average degree of prejacard graph is ', np.mean(G.degree()))
    #G = G.simplify(combine_edges='sum') #NEW jan17#
    #print('average degree of SIMPLE graph is ', np.mean(G.degree()))
    #print('edgelist_copy:', edgelist_copy) #[(0,20),(0,22)...]
    print('computing Jaccard metric')
    sim_list = G.similarity_jaccard(pairs = edgelist_copy)

    #print('pre-lognormed mean and std', np.mean(sim_list), np.std(sim_list))
    #plt.hist(sim_list)
    #plt.xlabel("Jaccard co-efficient")
    #plt.ylabel("No. edges")
    #plt.title("Histogram of Jaccard co-efficients")

    #plt.show()
    #sim_list = stats.zscore(boxcox(sim_list,0))#log to get normal distribution
    #sim_list = sim_list +np.abs(np.min(sim_list))
    #plt.hist(sim_list)
    #plt.show()
    #sim_list = G.similarity_inverse_log_weighted(vertices=edgelist_copy)
    #print('simlist jaccard weights mean and std:',np.mean(sim_list),np.std(sim_list))
    new_edgelist = []
    sim_list_array = np.asarray(sim_list)
    edge_list_copy_array = np.asarray(edgelist_copy)

    if jac_std=='median': threshold = np.median(sim_list)
    else: threshold = np.mean(sim_list) - jac_std*np.std(sim_list)
    strong_locs = np.where(sim_list_array>threshold)[0] #if it gets smaller than 0.5 then you get too many tiny tiny clusters of size 2
    #for ii in strong_locs: new_edgelist.append(edgelist_copy[ii])
    print('percentage of edges KEPT', len(strong_locs) / len(sim_list))
    new_edgelist = list(edge_list_copy_array[strong_locs])
    sim_list_new = list(sim_list_array[strong_locs])

    G_sim = ig.Graph(list(new_edgelist), edge_attrs={'weight': sim_list_new})
    print('average degree of graph is ', np.mean(G_sim.degree()))
    G_sim.simplify(combine_edges='sum') #"first"
    print('average degree of SIMPLE graph is ', np.mean(G_sim.degree()))
    #G_sim = ig.Graph(list(edgelist), edge_attrs={'weight': sim_list})
    #layout = G.layout("rt",2)
    #ig.plot(G)
    #print('average degree of graph is ', np.mean(G_sim.degree()))
    #****G_sim.simplify(combine_edges='first')
    #print('average degree of SIMPLE graph is ', np.mean(G_sim.degree()), G_sim.degree)
    #print('vertices', G.vcount())
    #print('time elapsed {} seconds'.format(time.time() - time_start_nx))
    # first compute the best partition. A length n_total dictionary where each dictionary key is a group, and the members of dict[key] are the member cells
    time_end_prune = time.time()- time_start_prune
    time_start_louvain = time.time()
    print('starting Louvain clustering at', time.ctime())
    optimiser = louvain.Optimiser()
    if jac_weighted_edges == True: partition = louvain.find_partition(G_sim, louvain.ModularityVertexPartition, weights='weight')
    else: partition = louvain.find_partition(G_sim, louvain.ModularityVertexPartition)

    #partition = partition.aggregate_partition()
    print('Q=',partition.quality())
    improv = 1
    li=2
    while li<2:
        time0 = time.time()
        if jac_weighted_edges == True:
            newpart = louvain.find_partition(G_sim, louvain.ModularityVertexPartition, weights='weight')
        else:
            newpart = louvain.find_partition(G_sim, louvain.ModularityVertexPartition)

        if newpart.quality()> partition.quality():
            partition = newpart
            li=0
        print('Q = ', partition.quality(), 'is the', li,'th iteration, and took ', int(time.time()-time0),
              'sec')
        li=li+1
    louvain_labels = np.empty((n_elements, 1))
    #print(partition,'partition dict')
    print('finished calling louvain at',time.asctime(),len(partition),'clusters')
    small_pop_list = []
    small_cluster_list=[]

    # for key in range(len(partition)):
    #     population = len(partition[key])
    #     if population < small_pop: #10 for cytof
    #         print(key, ' has small population of ', population, )
    #         small_pop_list.append([t for t in partition[key]])
    #         small_cluster_list.append(key)
    #     for cell in partition[key]:
    #         louvain_labels[cell] = key


    for member_list, cluster_name in partition.items():
        louvain_labels[member_list] = cluster_name
    print('made louvain label list from partition dictionary')

    for member_list, cluster_name in partition.items():
        if len(member_list)<small_pop:
            small_pop_list = small_pop_list + member_list
            small_cluster_list.append(cluster_name)

    print('list of clusters that have low population:', small_cluster_list)

    for small_cluster in small_pop_list:
        for single_cell in small_cluster:
            #print('single cell')
            #print('dim neigh array', neighbor_array.shape)
            old_neighbors = neighbor_array[single_cell,:]
            print('true label of single cell in small pop is' , true_label[single_cell])
            #print(old_neighbors, 'old neighbors')
            group_of_old_neighbors = louvain_labels[old_neighbors]
            group_of_old_neighbors = list(group_of_old_neighbors.flatten())

            #print('single cell', single_cell, 'has this group_of_old_neighbors', group_of_old_neighbors)
            from statistics import mode
            available_neighbours = set(group_of_old_neighbors) - set(small_cluster_list)

            if len(available_neighbours) != 0:
                available_neighbours_list = [value for value in group_of_old_neighbors if value in list(available_neighbours)]
                best_group = max(available_neighbours_list, key=available_neighbours_list.count)
                louvain_labels[single_cell] = best_group
            #print('best group is', best_group)
    louvain_labels_0 = louvain_labels# list(louvain_labels.T)[0]
    too_big = False
    set_louvain_labels = set(list(louvain_labels.T)[0])
    print(set_louvain_labels)
    for cluster_i in set_louvain_labels:
        cluster_i_loc = np.where(louvain_labels == cluster_i)[0]
        pop_i = len(cluster_i_loc)
        if pop_i > too_big_factor*n_elements: #0.4
            too_big = True
            cluster_big_loc  = cluster_i_loc
            list_pop_too_bigs = [pop_i]
            list_too_big = [cluster_i]
            cluster_too_big = cluster_i
            print('cluster', cluster_i, 'of population', pop_i)
    if too_big == True: print('removing too big, cluster, ', cluster_too_big, 'with population ', list_pop_too_bigs[0], 'among list of toobig clusters:', list_pop_too_bigs)
    while too_big == True:
        knn_big = 50 #100 for 8th sep #50 for all the ALPH tests done for 3 professor presentation in start OCT #knn = 5 on the 8-10th oct
        #ef_big = knn_big + 10
        X_data_big = X_data[cluster_big_loc,:]
        knn_struct_big = make_knn_struct(X_data_big)
        print(X_data_big.shape)
        louvain_labels_big = run_toobig_sublouvain(X_data_big, knn_struct_big, k_nn=knn_big, self_loop=False,jac_std=jac_std) #knn=200 for 10x
        print('set of new big labels ',set(list(louvain_labels_big.flatten())))
        louvain_labels_big = louvain_labels_big+1000#len(set(louvain_labels.T)[0])
        print('set of new big labels +1000 ', set(list(louvain_labels_big.flatten())))
        pop_list = []
        for item in set(list(louvain_labels_big.flatten())):
            pop_list.append(list(louvain_labels_big.flatten()).count(item))
        print('pop of new big labels', pop_list)
        jj =0
        print('shape louvain_labels', louvain_labels.shape)
        for j in cluster_big_loc:
            louvain_labels[j] = louvain_labels_big[jj]
            jj=jj+1
        dummy, louvain_labels= np.unique(list(louvain_labels.flatten()), return_inverse=True)
        print('new set of labels ', set(louvain_labels))
        too_big = False
        set_louvain_labels =set(louvain_labels)#list(louvain_labels.T)[0])

        louvain_labels = np.asarray(louvain_labels)
        for cluster_ii in set_louvain_labels:
            cluster_ii_loc = np.where(louvain_labels == cluster_ii)[0]
            pop_ii = len(cluster_ii_loc)
            not_yet_expanded = pop_ii not in list_pop_too_bigs
            if pop_ii > too_big_factor * n_elements and not_yet_expanded==True:
                too_big = True
                print('cluster', cluster_ii, 'is too big and has population', pop_ii)
                cluster_big_loc = cluster_ii_loc
                cluster_big = cluster_ii
                big_pop = pop_ii
        if too_big==True:
            list_pop_too_bigs.append(big_pop)
            print('cluster', cluster_big, 'is too big with population', big_pop,'. It will be expanded')
    dummy, louvain_labels= np.unique(list(louvain_labels.flatten()), return_inverse=True)
    print('new set of labels ', set(louvain_labels))
    print('final shape before too_small allocation', set(list(louvain_labels.flatten())))
    small_pop_list= []
    small_cluster_list = []
    small_pop_exist= False
    print(set(list(louvain_labels.flatten())))
    #for cluster in set(list(louvain_labels.flatten())):
    for cluster in set(louvain_labels):
        population = len(np.where(louvain_labels==cluster)[0])
        print(cluster, 'has population', population)
        if population < small_pop:#10
            small_pop_exist= True
            print(cluster, ' has small population of', population, )
            small_pop_list.append(list(np.where(louvain_labels==cluster)[0]))
            small_cluster_list.append(cluster)
    print('list of small clusters Round 2:',small_cluster_list)
    for small_cluster in small_pop_list:
        print('we will now look at elements of the small cluster', small_cluster)
        for single_cell in small_cluster:
            print('single cell')
            print('dim neigh array', neighbor_array.shape)
            old_neighbors = neighbor_array[single_cell,:]
            #print(old_neighbors, 'old neighbors')
            group_of_old_neighbors = louvain_labels[old_neighbors]
            group_of_old_neighbors = list(group_of_old_neighbors.flatten())

            #print('single cell', single_cell, 'has this group_of_old_neighbors', group_of_old_neighbors)
            from statistics import mode
            available_neighbours = set(group_of_old_neighbors) - set(small_cluster_list)
            if len(available_neighbours) > 0:
                available_neighbours_list = [value for value in group_of_old_neighbors if value in list(available_neighbours)]
                best_group = max(available_neighbours_list, key=available_neighbours_list.count)
                louvain_labels[single_cell] = best_group
            #print('best group is', best_group)
    '''
    while small_pop_exist ==True:
        for small_cluster in small_pop_list:
            for single_cell in small_cluster:
                #print('single cell')
                #print('dim neigh array', neighbor_array.shape)
                old_neighbors = neighbor_array[single_cell, :]
                #print(old_neighbors, 'old neighbors')
                group_of_old_neighbors = louvain_labels[old_neighbors]
                group_of_old_neighbors = list(group_of_old_neighbors.flatten())

                #print('single cell', single_cell, 'has this group_of_old_neighbors', group_of_old_neighbors)
                from statistics import mode
                best_group = max(set(group_of_old_neighbors), key=group_of_old_neighbors.count)
                #print(best_group)
                louvain_labels[single_cell] = best_group
        small_pop_exist = False
        for cluster in set(list(louvain_labels.flatten())):
            population = len(np.where(louvain_labels == cluster)[0])
            if population < small_pop:
                small_pop_exist = True
                print(cluster, ' has small population of', population, )
                small_pop_list.append(np.where(louvain_labels == cluster)[0])
    '''
    dummy, louvain_labels = np.unique(list(louvain_labels.flatten()), return_inverse=True)
    louvain_labels=np.asarray(louvain_labels)
    print('final labels allocation', set(list(louvain_labels.flatten())))
    pop_list=[]
    for item in set(list(louvain_labels.flatten())):
        pop_list.append(list(louvain_labels.flatten()).count(item))
    print('pop of big list', pop_list)
    time_end_louvain = time.time() - time_start_louvain
    return louvain_labels,undirected_graph_array_forLV, time_end_louvain, time_end_prune, time_knn_query_end, len(strong_locs)

def run_mainlouvain(X_data,true_label, self_loop = False,LV_graphinput_file_name =None, LV_plot_file_name =None, too_big_factor = 0.4, knn_in = 30,small_pop = 10,dist_std = 1, keep_all_dist= True,jac_std=0.3,jac_weighted_edges = True):
    list_roc = []
    k_nn_range = [knn_in]#[100]#[30, 20, 10,5,3]#,30]#,15,20,30]
    ef = 50
    f1_temp = -1
    temp_best_labels = []
    iclus = 0
    time_start_total = time.time()
    for k_nn in k_nn_range:
        time_start_knn = time.time()
        knn_struct = make_knn_struct(X_data)
        time_end_knn = time.time() - time_start_knn
        # Query dataset, k - number of closest elements (returns 2 numpy arrays)
        time_start_louvain = time.time()
        louvain_labels, undirected_graph_array_forLV,time_end_louvain, time_end_prune, time_end_knn_query, num_edges = run_sublouvain(X_data, true_label, knn_struct, k_nn, self_loop=self_loop, too_big_factor=too_big_factor, small_pop = small_pop, dist_std = dist_std, keep_all_dist= keep_all_dist,jac_std=jac_std, jac_weighted_edges=jac_weighted_edges)
        if k_nn >=10 and LV_graphinput_file_name!=None:
            LV_graphinput_file_name = LV_graphinput_file_name
            np.savetxt(LV_graphinput_file_name,undirected_graph_array_forLV)
            time_start_lv = time.time()
            X_embedded = run_lv(perplexity=30,lr=1, graph_array_name= LV_graphinput_file_name)
            print('lv runtime', time.time()- time_start_lv)
        #print('number of communities is:', len(set(list(louvain_labels))))
        print('time elapsed {:.2f} seconds'.format(time.time() - time_start_louvain))
        run_time =time.time() - time_start_louvain
        time_end_total = time.time() - time_start_total
        print('current time is: ', time.ctime())
        targets = list(set(true_label))
        #if len(targets) >=2: target_range = targets
        #else: target_range = [1]
        N = len(list(true_label))
        f1_accumulated =0
        f1_acc_noweighting = 0
        for onevsall_val in targets:
            print('target is', onevsall_val)
            vals_roc, predict_class_array, majority_truth_labels, numclusters_targetval = accuracy_mst(list(louvain_labels.flatten()), true_label,
                                                             embedding_filename=None, clustering_algo='louvain', onevsall=onevsall_val)
            #list(louvain_labels.T)[0]
            f1_current = vals_roc[1]
            print('target', onevsall_val,'had f1-score of %.2f' % (f1_current))
            f1_accumulated = f1_accumulated+ f1_current*(list(true_label).count(onevsall_val))/N
            f1_acc_noweighting = f1_acc_noweighting + f1_current
            #print(f1_accumulated, f1_current, list(true_label).count(onevsall_val))
            if f1_current > f1_temp:
                f1_temp = f1_current
                temp_best_labels = list(louvain_labels.flatten())
                onevsall_opt = onevsall_val
                knn_opt = k_nn
                predict_class_array_opt = predict_class_array
            list_roc.append([ef, k_nn,jac_std, dist_std, onevsall_val]+vals_roc +[numclusters_targetval]+ [run_time])

            if iclus == 0:
                predict_class_aggregate = np.array(predict_class_array)
                iclus = 1
            else:
                predict_class_aggregate = np.vstack((predict_class_aggregate, predict_class_array))
        f1_mean = f1_acc_noweighting / len(targets)
        print("f1-score mean ", f1_mean)
    if LV_graphinput_file_name != None:
        import Performance_phenograph as pp
        fig, ax = plt.subplots(1, 2, figsize=(20, 10))
        pp.plot_mst_simple(ax[0], ax[1], temp_best_labels, true_label, None, None, None, None,
                       cancer_type='k562', clustering_method='louvain', X_embedded=X_embedded,
                       k_nn=knn_opt)
        plt.savefig(LV_plot_file_name, bbox_inches='tight')
    df_accuracy = pd.DataFrame(list_roc,
                               columns=['ef','knn', 'jac_std','dist_std','onevsall-target','error rate','f1-score', 'tnr', 'fnr',
                                        'tpr', 'fpr', 'precision', 'recall', 'num_groups','population of target' ,'num clusters','clustering runtime'])
    #print('weighted (by population) F1-score: ', f1_accumulated)
    #knn_opt = df_accuracy['knn'][df_accuracy['f1-score'].idxmax()]
    return predict_class_aggregate, df_accuracy, temp_best_labels,knn_opt, onevsall_opt, majority_truth_labels, time_end_knn, time_end_prune, time_end_louvain, time_end_total,f1_accumulated,f1_mean, time_end_knn_query, num_edges

def run_phenograph(X_data, true_label, knn =30):
    list_roc = []
    f1_accumulated = 0
    iclus = 0
    time_pheno = time.time()
    #print('phenograph started at time:', time.ctime())
    communities, graph, Q = phenograph.cluster(X_data, k=knn)
    #print('communities finished at time :', time.ctime())
    pheno_time = time.time() - time_pheno
    print('phenograph took ', pheno_time, 'seconds for samples size', len(true_label))
    f1_temp = -1
    targets = list(set(true_label))
    print('number of original groups:', len(targets))
    print('number of clusters in Phenograph:', len(set(list(communities))))

    #if len(targets) >2: target_range = target
    #else: target_range = [1]

    f1_acc_noweighting = 0
    for onevsall_val in targets:
        vals_roc, predict_class_array, majority_truth_labels, numclusters_targetval = accuracy_mst(list(communities), pd.Series(true_label),
                                                        None, 'phenograph',
                                                        onevsall=onevsall_val)

        f1_current = vals_roc[1]
        f1_accumulated = f1_accumulated + f1_current * (list(true_label).count(onevsall_val)) / len(true_label)
        f1_acc_noweighting = f1_acc_noweighting + f1_current
        list_roc.append([onevsall_val]+vals_roc + [numclusters_targetval]+[pheno_time]+[knn])
        if vals_roc[1] > f1_temp:
            f1_temp = vals_roc[0]
            temp_best_labels = list(communities)
            onevsall_opt = onevsall_val
        if iclus == 0:
            predict_class_aggregate = np.array(predict_class_array)
            iclus = 1
        else:
            predict_class_aggregate = np.vstack((predict_class_aggregate, predict_class_array))



    df_accuracy = pd.DataFrame(list_roc, columns=['onevsall-target','error rate','f1-score', 'tnr', 'fnr',
                                        'tpr', 'fpr', 'precision', 'recall', 'num_groups', 'population of class/group','num clusters for that class','clustering runtime','knn'])
    print('Accumulated F1-Score', f1_accumulated)
    f1_acc_noweighting = f1_acc_noweighting/len(targets)
    print("f1-score mean ", f1_acc_noweighting )
    return predict_class_aggregate, df_accuracy, list(communities), onevsall_opt, majority_truth_labels, pheno_time, f1_acc_noweighting

def accuracy_mst(model, true_labels, embedding_filename, clustering_algo,phenograph_time = None, onevsall=1,verbose_print = False):
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
    elif clustering_algo =='phenograph':
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
    elif clustering_algo=='louvain' or clustering_algo =='multiclass mst':
        mst_labels = model
        #print('louvain labels',model)

    else: mst_labels = list(model.labels_)
    N = len(mst_labels)
    n_cancer = list(true_labels).count(onevsall)
    n_pbmc = N-n_cancer
    m = 999
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
    small_pop_list = []
    for kk in sorted_keys:
        vals = [t for t in Index_dict[kk]]
        population = len(vals)
        #majority_val = func_counter(vals)
        majority_val = func_mode(vals)
        if majority_val == onevsall: print('cluster',kk, ' has majority', onevsall, 'with population', len(vals))
        #Vif verbose_print== True: print(kk, 'is majority ', majority_val, 'with population ', len(vals))
        if kk==-1:
            len_unknown = len(vals)
            print('len unknown', len_unknown)
        if (majority_val == onevsall) and (kk != -1):
            thp1_labels.append(kk)
            fp = fp + len([e for e in vals if e != onevsall])
            tp = tp + len([e for e in vals if e == onevsall])
            list_error = [e for e in vals if e != majority_val]
            #V if len(list_error)>0: print('majority of errors are in class:', func_mode(list_error))
            #Velse: print('there are no errors in the cluster')
            e_count = len(list_error)
            error_count.append(e_count)
            #Vprint('cluster',kk, 'has error rate of %', e_count*100/len(vals))
        elif (majority_val != onevsall) and (kk != -1):
            pbmc_labels.append(kk)
            tn = tn + len([e for e in vals if e != onevsall])
            fn = fn + len([e for e in vals if e == onevsall])
            error_count.append(len([e for e in vals if e != majority_val]))
        '''
        if majority_val == 999:
            thp1_labels.append(kk)
            unknown_labels.append(kk)
            print(kk, ' has no majority, we are adding it to cancer_class')
            fp = fp + len([e for e in vals if e != onevsall])
            tp = tp + len([e for e in vals if e == onevsall])
            error_count.append(len([e for e in vals if e != majority_val]))
        '''
    predict_class_array = np.array(mst_labels)
    mst_labels_array = np.array(mst_labels)
    number_clusters_for_target = len(thp1_labels)
    for cancer_class in thp1_labels:
        predict_class_array[mst_labels_array == cancer_class] = 1
    for benign_class in pbmc_labels:
        predict_class_array[mst_labels_array == benign_class] = 0
    predict_class_array.reshape((predict_class_array.shape[0], -1))
    error_rate = sum(error_count) / N
    n_target = tp+fn
    comp_n_cancer = tp + fp
    comp_n_pbmc = fn + tn
    tnr = tn / n_pbmc
    fnr = fn / n_cancer
    tpr = tp / n_cancer
    fpr = fp / n_pbmc
    if comp_n_cancer != 0:
        computed_ratio = comp_n_pbmc / comp_n_cancer

    if tp != 0 or fn != 0: recall = tp / (tp + fn)  # ability to find all positives
    if tp != 0 or fp != 0: precision = tp / (tp + fp)  # ability to not misclassify negatives as positives
    if precision != 0 or recall != 0:
        f1_score = precision * recall * 2 / (precision + recall)
    majority_truth_labels = np.empty((len(true_labels),1), dtype=object)

    for cluster_i in set(mst_labels):
        cluster_i_loc = np.where(np.asarray(mst_labels) == cluster_i)[0]
        population_cluster_i = len(cluster_i_loc)
        true_labels= np.asarray(true_labels)
        majority_truth = func_mode(list(true_labels[cluster_i_loc]))
        majority_truth_labels[cluster_i_loc] = majority_truth

    majority_truth_labels = list(majority_truth_labels.flatten())


    if clustering_algo=='phenograph': mst_runtime = phenograph_time
    elif clustering_algo =='multiclass mst' or clustering_algo== 'louvain': mst_runtime = None

    else: mst_runtime = model.clustering_runtime_
    if clustering_algo == 'louvain' or clustering_algo=='phenograph' or clustering_algo == 'multiclass mst' or clustering_algo =='kmeans': accuracy_val = [error_rate,f1_score, tnr, fnr, tpr, fpr, precision,
                    recall, num_groups, n_target]
    else: accuracy_val = [embedding_filename, sigma, min_cluster_size, mergetooclosefactor, error_rate, f1_score, tnr, fnr, tpr, fpr, precision,
                    recall, num_groups, mst_runtime]

    return accuracy_val, predict_class_array, majority_truth_labels, number_clusters_for_target

def run_main(cancer_type, n_cancer,n_benign, randomseedval=[1,2,3]):

    n_cancer = n_cancer
    n_benign = n_benign
    ratio = n_benign / n_cancer
    print('the ratio is {}'.format(ratio))
    print('ncancer, nbenign', n_cancer,n_benign)
    print(cancer_type ,' is the type of cancer')
    n_total = n_cancer + n_benign
    num_nn = 30
    cancer_type = cancer_type #'thp1'
    benign_type = 'pbmc'

    fluor = 0
    new_folder_name = cancer_type + '_r{:.2f}'.format(ratio) + '_n' + str(n_cancer)+'_NoJaccPrune_mask5s_101112'
    path_tocreate = '/home/shobi/Thesis/Louvain_data/' + new_folder_name
    os.mkdir(path_tocreate)
    num_dataset_versions = 1
    dataset_version_range = range(num_dataset_versions)

    for dataset_version in dataset_version_range:
        randomseed_singleval = randomseedval[dataset_version]
        excel_file_name = '/home/shobi/Thesis/Louvain_data/' + new_folder_name + '/Louvain_excel_' + cancer_type + '_data' + str(
            dataset_version) + '_r{:.2f}'.format(ratio) + '_ncancer' + str(n_cancer) + '.xlsx'
        LV_graphinput_file_name = '/home/shobi/Thesis/Louvain_data/' + new_folder_name + '/graphinput_' + cancer_type + '_data' + str(
            dataset_version) + '_r{:.2f}'.format(ratio) + '_ncancer' + str(n_cancer)+'.txt'
        LV_plot_file_name = '/home/shobi/Thesis/Louvain_data/' + new_folder_name + '/LV_Plot_' + cancer_type + '_data' + str(
            dataset_version) + '_r{:.2f}'.format(ratio) + '_ncancer' + str(n_cancer)+'.png'
        true_label, tag, X_data, new_file_name, df_all, index_list, flist = get_data(cancer_type, benign_type, n_cancer,
                                                                                     ratio,
                                                                                     fluor, dataset_version,
                                                                                 new_folder_name, method='louvain',randomseedval=randomseed_singleval)

        writer = ExcelWriter(excel_file_name)

        #model_dbscan = DBSCAN(0.02, 10, tooclose_factor=0).fit(X_data)
        #vals_roc, predict_class_array, majority_truth_labels = accuracy_mst(model_dbscan, true_label,
                                                     #embedding_filename=None, clustering_algo='dbscan')
        predict_class_aggregate_louvain, df_accuracy_louvain, best_louvain_labels, knn_opt, onevsall_opt, maj_vals = run_mainlouvain(X_data,
                                                                                                             true_label, self_loop=False,LV_graphinput_file_name = None, LV_plot_file_name = LV_plot_file_name)


        #predict_class_aggregate_louvain, df_accuracy_louvain_selfloop, best_louvain_labels_selfloop, knn_opt_selfloop, onevsall_opt_selfloop = run_mainlouvain(X_data,
         #                                                                                                    true_label, self_loop=True)

        df_accuracy_louvain.to_excel(writer, 'louvain', index=False)
        #df_accuracy_louvain_selfloop.to_excel(writer, 'louvain self loop', index=False)

        writer.save()
        print('successfully saved excel files')
def main():

    #run_main('thp1', n_cancer=100, n_benign=466000)
    print('weighted graph with ef =50')
    run_main('k562', n_cancer=200000, n_benign=300000, randomseedval=[10])#[10,11,12])
    #run_main('acc220', n_cancer=100, n_benign=466000)
    ##run_main('thp1', n_cancer=50, n_benign=466000)
    #run_main('k562', n_cancer=50, n_benign=466000)
    #run_main('acc220', n_cancer=50, n_benign=466)
    #run_main('acc220', n_cancer=100, n_benign=466000)
    #run_main('acc220', n_cancer=200, n_benign=466000)

if __name__ == '__main__':
    main()