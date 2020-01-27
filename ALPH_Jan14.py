from sklearn.cluster import DBSCAN
import phenograph
import os
import LargeVis
import numpy as np
import scipy.io
import pandas as pd
import hnswlib
from scipy.sparse import csr_matrix
import igraph as ig
import louvain #https://github.com/vtraag/louvain-igraph
import matplotlib.pyplot as plt
import time
from pandas import ExcelWriter

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
    print('time elapsed', time.time()-time_start)

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

def make_csrmatrix_noselfloop(neighbor_array, distance_array,dist_std =1, keep_all_dist= False):
    print('keep all is', keep_all)
    row_list = []
    col_list = []
    weight_list = []
    neighbor_array = neighbor_array
    print('size neighbor array', neighbor_array.shape)
    num_neigh = neighbor_array.shape[1]
    distance_array = distance_array
    n_neighbors = neighbor_array.shape[1]
    n_cells =  neighbor_array.shape[0]
    rowi = 0
    count_0dist =0
    discard_count = 0
    print('dist std factor ', dist_std)
    if keep_all_dist== False:
        for row in neighbor_array:
            distlist =distance_array[rowi, :]
            to_keep = np.where(distlist < np.mean(distlist)+dist_std*np.std(distlist))[0] #0*std
            updated_nn_ind = row[np.ix_(to_keep)]
            updated_nn_weights = distlist[np.ix_(to_keep)]
            discard_count = discard_count + (num_neigh-len(to_keep))

            for ik in range(len(updated_nn_ind)):
                if rowi != row[ik]:  # remove self-loops
                    row_list.append(rowi)
                    col_list.append(updated_nn_ind[ik])
                    dist = np.sqrt(updated_nn_weights[ik])
                    if dist == 0:
                        dist = np.mean(updated_nn_weights)*0.1
                        count_0dist = count_0dist + 1  # print('rowi and row[ik] are modified to', rowi, row[ik], 'and are',dist, 'dist apart')
                    weight_list.append(1 / dist)
            rowi = rowi + 1

    if keep_all_dist== True:

        row_list.extend(list(np.transpose(np.ones((n_neighbors, n_cells)) * range(0, n_cells)).flatten()))
        col_list = neighbor_array.flatten().tolist()
        weight_list = distance_array.flatten().tolist()

    print('share of neighbors discarded in Distance pruning', discard_count,discard_count/neighbor_array.size)
    print('number of zero dist', count_0dist)
    csr_array = csr_matrix((np.array(weight_list), (np.array(row_list), np.array(col_list))), shape=(n_cells, n_cells))
    sources, targets = csr_array.nonzero()
    mask = np.zeros(len(sources), dtype=bool)
    mask |= (csr_array.data < (np.mean(csr_array.data) - np.std(csr_array.data) * 5))
    print('sum of mask', sum(mask))
    csr_array.data[mask] = 0
    csr_array.eliminate_zeros()


    return csr_array

def run_toobig_sublouvain(X_data, knn_struct,k_nn,self_loop = False, keep_all=False,jac_std= 0.3): #0.3 was default dist_std
    n_elements = X_data.shape[0]


    print('number of k-nn is', k_nn)
    neighbor_array, distance_array = knn_struct.knn_query(X_data, k=k_nn)

    print('shapes of neigh and dist array',neighbor_array.shape, distance_array.shape)
    undirected_graph_array_forLV = None
    if self_loop == False:
        csr_array = make_csrmatrix_noselfloop(neighbor_array, distance_array, keep_all=keep_all, dist_std=0)#was dist_std = default (which is 0 )until Oct11


    sources, targets = csr_array.nonzero()
    edgelist = list(zip(sources.tolist(), targets.tolist()))
    edgelist_copy = edgelist.copy()
    print('conversion to igraph', time.ctime())
    G = ig.Graph(edgelist, edge_attrs={'weight': csr_array.data.tolist()})
    print('computing Jaccard metric', time.ctime())
    sim_list = G.similarity_jaccard(pairs=edgelist_copy) #list of jaccard weights
    print('pre-lognormed mean and std', np.mean(sim_list), np.std(sim_list))
    new_edgelist = []
    sim_list_array = np.asarray(sim_list)
    strong_locs = np.where(sim_list_array>np.mean(sim_list) - Jac_std*np.std(sim_list))[0] #if it gets smaller than 0.5 then you get too many tiny tiny clusters of size 2
    for ii in strong_locs: new_edgelist.append(edgelist_copy[ii])
    print(len(sim_list), 'length of sim list before pruning')

    print('percentage of edges KEPT', len(strong_locs) / len(sim_list))
    sim_list_new = list(sim_list_array[strong_locs])
    print(len(sim_list_new), 'length of sim list after pruning')

    G_sim = ig.Graph(list(new_edgelist))#, edge_attrs={'weight': sim_list_new})
    print('average degree of graph is ', np.mean(G_sim.degree()))
    G_sim.simplify(combine_edges='first')
    print('average degree of SIMPLE graph is ', np.mean(G_sim.degree()), G_sim.degree)
    print('starting Louvain clustering at', time.ctime())
    partition = louvain.find_partition(G_sim, louvain.ModularityVertexPartition)#, weights='weight')
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
            old_neighbors = neighbor_array[single_cell, :]
            group_of_old_neighbors = louvain_labels[old_neighbors]
            group_of_old_neighbors = list(group_of_old_neighbors.flatten())

            available_neighbours = set(group_of_old_neighbors) - set(small_cluster_list)
            if len(available_neighbours) > 0:
                available_neighbours_list = [value for value in group_of_old_neighbors if
                                             value in list(available_neighbours)]
                best_group = max(available_neighbours_list, key=available_neighbours_list.count)
                louvain_labels[single_cell] = best_group

    return louvain_labels

def run_sublouvain(X_data, knn_struct,k_nn, too_big_factor = 0.15, small_pop = 5, dist_std = 1,keep_all= False,jac_std= 0.3):#50
    n_elements = X_data.shape[0]

    print('number of k-nn is', k_nn, too_big_factor, 'small pop is', small_pop)
    knn_query_start = time.time()
    neighbor_array, distance_array = knn_struct.knn_query(X_data, k=k_nn)
    time_knn_query_end = time.time() - knn_query_start
    print('shapes of neigh and dist array', neighbor_array.shape, distance_array.shape)
    time_start_prune = time.time()
    csr_array = make_csrmatrix_noselfloop(neighbor_array, distance_array, dist_std,keep_all_dist= keep_all)


    sources, targets = csr_array.nonzero()
    edgelist = list(zip(sources,targets))

    edgelist_copy = edgelist.copy()

    G = ig.Graph(edgelist)#, edge_attrs={'weight': csr_array.data.tolist()})

    print('computing Jaccard metric')
    sim_list = G.similarity_jaccard(pairs = edgelist_copy)

    sim_list_array = np.asarray(sim_list)
    edge_list_copy_array = np.asarray(edgelist_copy)


    strong_locs = np.where(sim_list_array>np.mean(sim_list) - Jac_std*np.std(sim_list))[0] #if it gets smaller than 0.5 then you get too many tiny tiny clusters of size 2
    print('percentage of edges KEPT', len(strong_locs) / len(sim_list))
    new_edgelist = list(edge_list_copy_array[strong_locs])

    G_sim = ig.Graph(list(new_edgelist))#, edge_attrs={'weight': sim_list_new})
    print('average degree of graph is ', np.mean(G_sim.degree()))
    G_sim.simplify(combine_edges='first')
    print('average degree of SIMPLE graph is ', np.mean(G_sim.degree()), G_sim.degree)

    # first compute the best partition. A length n_total dictionary where each dictionary key is a group, and the members of dict[key] are the member cells
    time_end_prune = time.time()- time_start_prune
    time_start_louvain = time.time()
    print('starting Louvain clustering at', time.ctime())
    partition = louvain.find_partition(G_sim, louvain.ModularityVertexPartition)#, weights='weight')
    louvain_labels = np.empty((n_elements, 1))
    small_pop_list = []
    small_cluster_list=[]
    for key in range(len(partition)):
        population = len(partition[key])
        if population < small_pop: #10 for cytof
            small_pop_list.append([t for t in partition[key]])
            small_cluster_list.append(key)
        for cell in partition[key]:
            louvain_labels[cell] = key
    print('list of clusters that have low population:', small_cluster_list)
    for small_cluster in small_pop_list:
        for single_cell in small_cluster:
            old_neighbors = neighbor_array[single_cell,:]
            group_of_old_neighbors = louvain_labels[old_neighbors]
            group_of_old_neighbors = list(group_of_old_neighbors.flatten())
            available_neighbours = set(group_of_old_neighbors) - set(small_cluster_list)

            if len(available_neighbours) != 0:
                available_neighbours_list = [value for value in group_of_old_neighbors if value in list(available_neighbours)]
                best_group = max(available_neighbours_list, key=available_neighbours_list.count)
                louvain_labels[single_cell] = best_group
    too_big = False
    set_louvain_labels = set(list(louvain_labels.T)[0])
    print(set_louvain_labels)
    for cluster_i in set_louvain_labels:
        cluster_i_loc = np.where(louvain_labels == cluster_i)[0]
        pop_i = len(cluster_i_loc)
        if pop_i > too_big_factor*n_elements:
            too_big = True
            cluster_big_loc  = cluster_i_loc
            list_pop_too_bigs = [pop_i]
            cluster_too_big = cluster_i
            print('cluster', cluster_i, 'of population', pop_i)
    if too_big == True: print('removing too big, cluster, ', cluster_too_big, 'with population ', list_pop_too_bigs[0], 'among list of toobig clusters:', list_pop_too_bigs)
    while too_big == True:
        knn_big = 30 #50 for thesis tests# 100 for 8th sep #50 for all the ALPH tests done for 3 professor presentation in start OCT #knn = 5 on the 8-10th oct
        X_data_big = X_data[cluster_big_loc,:]
        knn_struct_big = make_knn_struct(X_data_big)
        print(X_data_big.shape)
        louvain_labels_big = run_toobig_sublouvain(X_data_big, knn_struct_big, k_nn=knn_big, self_loop=False,Jac_std=Jac_std) #knn=200 for 10x
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
            group_of_old_neighbors = louvain_labels[old_neighbors]
            group_of_old_neighbors = list(group_of_old_neighbors.flatten())

            available_neighbours = set(group_of_old_neighbors) - set(small_cluster_list)
            if len(available_neighbours) > 0:
                available_neighbours_list = [value for value in group_of_old_neighbors if value in list(available_neighbours)]
                best_group = max(available_neighbours_list, key=available_neighbours_list.count)
                louvain_labels[single_cell] = best_group

    dummy, louvain_labels = np.unique(list(louvain_labels.flatten()), return_inverse=True)
    louvain_labels=np.asarray(louvain_labels)
    print('final labels allocation', set(list(louvain_labels.flatten())))
    pop_list=[]
    for item in set(list(louvain_labels.flatten())):
        pop_list.append(list(louvain_labels.flatten()).count(item))
    print('pop of big list', pop_list)
    time_end_louvain = time.time() - time_start_louvain
    return louvain_labels, time_end_louvain, time_end_prune, time_knn_query_end, len(strong_locs)

def run_mainlouvain(X_data,true_label, self_loop = False,LV_graphinput_file_name =None, LV_plot_file_name =None, too_big_factor = 0.4, knn_in = 30,small_pop = 50,dist_std = 0, keep_all_dist= True,Jac_std=0.3):
    list_roc = []
    k_nn_range = [knn_in]
    ef = 50
    f1_temp = -1
    temp_best_labels = []
    iclus = 0
    time_start_total = time.time()
    for k_nn in k_nn_range:
        time_start_knn = time.time()
        knn_struct = make_knn_struct(X_data)
        time_end_knn = time.time() - time_start_knn
        time_start_louvain = time.time()
        print(time.ctime(), 'knn and toobig factor are', k_nn, too_big_factor)
        louvain_labels, undirected_graph_array_forLV,time_end_louvain, time_end_prune, time_end_knn_query, num_edges = run_sublouvain(X_data, knn_struct, k_nn, self_loop=self_loop, too_big_factor=too_big_factor, small_pop = small_pop, dist_std = dist_std, keep_all_dist= keep_all,Jac_std=Jac_std)

        print('time elapsed {:.2f} seconds'.format(time.time() - time_start_louvain))
        run_time =time.time() - time_start_louvain
        time_end_total = time.time() - time_start_total
        print('current time is: ', time.ctime())
        targets = list(set(true_label))
        N = len(list(true_label))
        f1_accumulated =0
        f1_acc_noweighting = 0
        for onevsall_val in targets:
            print('target is', onevsall_val)
            vals_roc, predict_class_array, majority_truth_labels, numclusters_targetval = accuracy_mst(list(louvain_labels.flatten()), true_label,
                                                             embedding_filename=None, clustering_algo='louvain', onevsall=onevsall_val)

            f1_current = vals_roc[1]
            f1_accumulated = f1_accumulated+ f1_current*(list(true_label).count(onevsall_val))/N
            f1_acc_noweighting = f1_acc_noweighting + f1_current

            if f1_current > f1_temp:
                f1_temp = f1_current
                temp_best_labels = list(louvain_labels.flatten())
                onevsall_opt = onevsall_val
                knn_opt = k_nn

            list_roc.append([ef, k_nn,Jac_std, onevsall_val]+vals_roc +[numclusters_targetval]+ [run_time])

            if iclus == 0:
                predict_class_aggregate = np.array(predict_class_array)
                iclus = 1
            else:
                predict_class_aggregate = np.vstack((predict_class_aggregate, predict_class_array))
        f1_mean = f1_acc_noweighting / len(targets)
        print("f1-score mean ", f1_mean)
    df_accuracy = pd.DataFrame(list_roc,
                               columns=['ef','knn', 'jac_std','onevsall-target','error rate','f1-score', 'tnr', 'fnr',
                                        'tpr', 'fpr', 'precision', 'recall', 'num_groups','num clusters' ,'clustering runtime'])

    return predict_class_aggregate, df_accuracy, temp_best_labels,knn_opt, onevsall_opt, majority_truth_labels, time_end_knn, time_end_prune, time_end_louvain, time_end_total,f1_accumulated,f1_mean, time_end_knn_query, num_edges

def run_phenograph(X_data, true_label):
    list_roc = []
    f1_accumulated = 0
    iclus = 0
    time_pheno = time.time()

    communities, graph, Q = phenograph.cluster(X_data)

    pheno_time = time.time() - time_pheno
    print('phenograph took ', pheno_time, 'seconds for samples size', len(true_label))
    f1_temp = -1
    targets = list(set(true_label))


    f1_acc_noweighting = 0
    for onevsall_val in targets:
        vals_roc, predict_class_array, majority_truth_labels, numclusters_targetval = accuracy_mst(list(communities), pd.Series(true_label),
                                                        None, 'phenograph',
                                                        onevsall=onevsall_val)

        f1_current = vals_roc[1]
        f1_accumulated = f1_accumulated + f1_current * (list(true_label).count(onevsall_val)) / len(true_label)
        f1_acc_noweighting = f1_acc_noweighting + f1_current
        list_roc.append([onevsall_val]+vals_roc + [numclusters_targetval]+[pheno_time])
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
                                        'tpr', 'fpr', 'precision', 'recall', 'num_groups', 'num clusters','clustering runtime'])
    f1_acc_noweighting = f1_acc_noweighting/len(targets)
    print("f1-score mean ", f1_acc_noweighting )
    return predict_class_aggregate, df_accuracy, list(communities), onevsall_opt, majority_truth_labels, pheno_time, f1_acc_noweighting

def accuracy_mst(model, true_labels, embedding_filename, clustering_algo,phenograph_time = None, onevsall=1):
    if clustering_algo =='mst':
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

    Index_dict = {}

    if clustering_algo=='phenograph': mst_labels = list(model)
    elif clustering_algo=='louvain' or clustering_algo =='multiclass mst':
        mst_labels = model

    else: mst_labels = list(model.labels_)
    N = len(mst_labels)
    n_cancer = list(true_labels).count(onevsall)
    n_pbmc = N-n_cancer
    for k in range(N):
        Index_dict.setdefault(mst_labels[k], []).append(true_labels[k])

    num_groups = len(Index_dict)
    sorted_keys = list(sorted(Index_dict.keys()))
    error_count = []
    NonTarget_labels = []
    target_labels = []
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
        majority_val = func_mode(vals)
        if majority_val == onevsall: print('cluster',kk, ' has majority', onevsall, 'with population', len(vals))
        if kk==-1:
            len_unknown = len(vals)
            print('len unknown', len_unknown)
        if (majority_val == onevsall) and (kk != -1):
            target_labels.append(kk)
            fp = fp + len([e for e in vals if e != onevsall])
            tp = tp + len([e for e in vals if e == onevsall])
            list_error = [e for e in vals if e != majority_val]
            e_count = len(list_error)
            error_count.append(e_count)
        elif (majority_val != onevsall) and (kk != -1):
            NonTarget_labels.append(kk)
            tn = tn + len([e for e in vals if e != onevsall])
            fn = fn + len([e for e in vals if e == onevsall])
            error_count.append(len([e for e in vals if e != majority_val]))

    predict_class_array = np.array(mst_labels)
    mst_labels_array = np.array(mst_labels)
    for cancer_class in target_labels:
        predict_class_array[mst_labels_array == cancer_class] = 1
    for benign_class in NonTarget_labels:
        predict_class_array[mst_labels_array == benign_class] = 0
    predict_class_array.reshape((predict_class_array.shape[0], -1))
    error_rate = sum(error_count) / N
    tnr = tn / n_pbmc
    fnr = fn / n_cancer
    tpr = tp / n_cancer
    fpr = fp / n_pbmc

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
                    recall, num_groups]
    else: accuracy_val = [embedding_filename, sigma, min_cluster_size, mergetooclosefactor, error_rate, f1_score, tnr, fnr, tpr, fpr, precision,
                    recall, num_groups, mst_runtime]

    return accuracy_val, predict_class_array, majority_truth_labels, len(target_labels)

def run_main(X_data, true_label,file_name='test', new_folder_name='ALPH_newfolder'):

    path_tocreate = '/home/shobi/Thesis/Louvain_data/' + new_folder_name
    os.mkdir(path_tocreate)
    excel_file_name = '/home/shobi/Thesis/Louvain_data/' + new_folder_name + '/Louvain_excel_' + file_name+'.xlsx'

    writer = ExcelWriter(excel_file_name)

    predict_class_aggregate_louvain, df_accuracy_louvain, best_louvain_labels, knn_opt, onevsall_opt, maj_vals = run_mainlouvain(X_data, true_label,keep_all=True)

    df_accuracy_louvain.to_excel(writer, 'louvain', index=False)
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