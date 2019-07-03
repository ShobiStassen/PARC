import os
import numpy as np

import scipy.io
import pandas as pd
import hnswlib
from scipy.sparse import csr_matrix
import igraph as ig
import leidenalg
import matplotlib.pyplot as plt
import time
from pandas import ExcelWriter
from collections import Counter


def func_mode(ll):  # return MODE of list
    # If multiple items are maximal, the function returns the first one encountered.
    return max(set(ll), key=ll.count)

class ALPH:
    def __init__(self, data, true_label, too_big_factor = 0.4, small_pop = 10,dist_std_local = 2,dist_std_global = 3,jac_std_global=0.15, keep_all_local_dist = False,jac_weighted_edges = True,knn = 30, n_iter_leiden=5):
          #higher dist_std_local means more edges are kept
          #highter jac_std_global means more edges are kept
        self.data = data
        self.true_label= true_label
        self.too_big_factor = too_big_factor
        self.small_pop = small_pop
        self.dist_std_local =dist_std_local
        self.dist_std_global = dist_std_global
        self.jac_std_global=  jac_std_global
        self.keep_all_local_dist = keep_all_local_dist
        self.jac_weighted_edges = jac_weighted_edges
        self.knn = knn
        self.n_iter_leiden = n_iter_leiden

    def make_knn_struct(self):
        ef=100
        num_dims = self.data.shape[1]
        n_elements = self.data.shape[0]
        print('input shape', n_elements, 'x', num_dims)
        p = hnswlib.Index(space='l2', dim=num_dims)
        p.init_index(max_elements=n_elements, ef_construction=200, M=30)
        p.add_items(self.data)
        p.set_ef(ef)  # ef should always be > k
        return p

    def make_csrmatrix_noselfloop(self, neighbor_array, distance_array):
        local_pruning_bool = not(self.keep_all_local_dist)
        print('will do local pruning based on minowski metric', local_pruning_bool)
        row_list = []
        col_list = []
        weight_list = []
        neighbor_array = neighbor_array  # not listed in in any order of proximity
        print('size neighbor array', neighbor_array.shape)
        num_neigh = neighbor_array.shape[1]
        distance_array = distance_array
        n_neighbors = neighbor_array.shape[1]
        n_cells = neighbor_array.shape[0]
        rowi = 0
        count_0dist = 0
        discard_count = 0
        print('local pruning at:',  self.dist_std_local, 'standard deviations above mean')
        if local_pruning_bool == True:  # do some local pruning based on distance
            for row in neighbor_array:
                distlist = distance_array[rowi, :]
                to_keep = np.where(distlist < np.mean(distlist) + self.dist_std_local * np.std(distlist))[0]  # 0*std
                updated_nn_ind = row[np.ix_(to_keep)]
                updated_nn_weights = distlist[np.ix_(to_keep)]
                discard_count = discard_count + (num_neigh - len(to_keep))

                for ik in range(len(updated_nn_ind)):
                    if rowi != row[ik]:  # remove self-loops
                        row_list.append(rowi)
                        col_list.append(updated_nn_ind[ik])
                        dist = np.sqrt(updated_nn_weights[ik])
                        if dist == 0:
                            count_0dist = count_0dist + 1
                        weight_list.append(dist)

                rowi = rowi + 1

        if local_pruning_bool == False:  # dont prune based on distance
            row_list.extend(list(np.transpose(np.ones((n_neighbors, n_cells)) * range(0, n_cells)).flatten()))
            col_list = neighbor_array.flatten().tolist()
            weight_list = (1. / (distance_array.flatten() + 0.1)).tolist()
        print('share of neighbors discarded in local distance pruning', discard_count, round(discard_count / neighbor_array.size))

        csr_graph = csr_matrix((np.array(weight_list), (np.array(row_list), np.array(col_list))),
                               shape=(n_cells, n_cells))
        return csr_graph

    def run_toobig_sublouvain(self, X_data, knn_struct, jac_std_toobig=0.3,
                              jac_weighted_edges=True):
        n_elements = X_data.shape[0]
        hnsw = self.knn_struct(X_data)
        neighbor_array, distance_array = hnsw.knn_query(X_data, k=self.knn)
        print('shapes of neigh and dist array', neighbor_array.shape, distance_array.shape)
        csr_array = self.make_csrmatrix_noselfloop(neighbor_array, distance_array)
        sources, targets = csr_array.nonzero()
        mask = np.zeros(len(sources), dtype=bool)
        mask |= (csr_array.data > (np.mean(csr_array.data) + np.std(csr_array.data) * 5)) #smaller distance means stronger edge
        print('sum of mask', sum(mask))
        csr_array.data[mask] = 0
        csr_array.eliminate_zeros()
        sources, targets = csr_array.nonzero()
        edgelist = list(zip(sources.tolist(), targets.tolist()))
        edgelist_copy = edgelist.copy()
        G = ig.Graph(edgelist, edge_attrs={'weight': csr_array.data.tolist()})
        sim_list = G.similarity_jaccard(pairs=edgelist_copy)  # list of jaccard weights
        new_edgelist = []
        sim_list_array = np.asarray(sim_list)
        if jac_std_toobig == 'median':
            threshold = np.median(sim_list)
        else:
            threshold = np.mean(sim_list) - jac_std_toobig * np.std(sim_list)
        strong_locs = np.where(sim_list_array > threshold)[0]
        for ii in strong_locs: new_edgelist.append(edgelist_copy[ii])
        sim_list_new = list(sim_list_array[strong_locs])

        if jac_weighted_edges == True:
            G_sim = ig.Graph(list(new_edgelist), edge_attrs={'weight': sim_list_new})
        else:
            G_sim = ig.Graph(list(new_edgelist))
        G_sim.simplify(combine_edges='sum')
        if jac_weighted_edges == True:
            partition = leidenalg.find_partition(G_sim, leidenalg.ModularityVertexPartition, weights='weight',
                                                 n_iterations=self.n_iter_leiden)
        else:
            partition = leidenalg.find_partition(G_sim, leidenalg.ModularityVertexPartition,
                                                 n_iterations=self.n_iter_leiden)
        print('Q=', round(partition.quality()))
        louvain_labels = np.asarray(partition.membership)
        louvain_labels = np.reshape(louvain_labels, (n_elements, 1))
        small_pop_list = []
        small_cluster_list = []
        small_pop_exist = False

        for cluster in set(louvain_labels):
            population = len(np.where(louvain_labels == cluster)[0])
            if population < 10:
                small_pop_exist = True
                small_pop_list.append(list(np.where(louvain_labels == cluster)[0]))
                small_cluster_list.append(cluster)

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

        while small_pop_exist == True:
            small_pop_list = []
            small_pop_exist = False
            for cluster in set(list(louvain_labels.flatten())):
                population = len(np.where(louvain_labels == cluster)[0])
                if population < 10:
                    small_pop_exist = True
                    print(cluster, ' has small population of', population, )
                    small_pop_list.append(np.where(louvain_labels == cluster)[0])
            for small_cluster in small_pop_list:
                for single_cell in small_cluster:
                    old_neighbors = neighbor_array[single_cell, :]
                    group_of_old_neighbors = louvain_labels[old_neighbors]
                    group_of_old_neighbors = list(group_of_old_neighbors.flatten())
                    best_group = max(set(group_of_old_neighbors), key=group_of_old_neighbors.count)
                    louvain_labels[single_cell] = best_group

        dummy, louvain_labels = np.unique(list(louvain_labels.flatten()), return_inverse=True)
        self.labels = louvain_labels
        return louvain_labels

    def run_sublouvain(self):
        X_data = self.data
        too_big_factor = self.too_big_factor
        small_pop =self.small_pop
        jac_std_global=self.jac_std_global
        jac_weighted_edges=self.jac_weighted_edges
        knn =  self.knn
        n_elements = X_data.shape[0]

        print('number of k-nn is', knn, too_big_factor, 'small pop is', small_pop)
        knn_query_start = time.time()
        neighbor_array, distance_array = self.knn_struct.knn_query(X_data, k=knn)

        csr_array = self.make_csrmatrix_noselfloop(neighbor_array, distance_array)
        sources, targets = csr_array.nonzero()


        # #global pruning based on distance metric (typically such that no pruning occurs here)
        # mask = np.zeros(len(csr_array.data), dtype=bool)
        # threshold = np.mean(csr_array.data) + (np.std(csr_array.data)*self.dist_std_global)
        # print(threshold)
        # mask |= (csr_array.data < threshold)
        # csr_array.data[mask] = 0
        # csr_array.eliminate_zeros()
        # sources, targets = csr_array.nonzero()
        edgelist = list(zip(sources, targets))

        edgelist_copy = edgelist.copy()

        G = ig.Graph(edgelist, edge_attrs={'weight': csr_array.data.tolist()})
        print('average degree of prejacard graph is ', np.mean(G.degree()))
        print('computing Jaccard metric')
        sim_list = G.similarity_jaccard(pairs=edgelist_copy)

        sim_list_array = np.asarray(sim_list)
        edge_list_copy_array = np.asarray(edgelist_copy)

        if jac_std_global== 'median':
            threshold = np.median(sim_list)
        else:
            threshold = np.mean(sim_list) - jac_std_global* np.std(sim_list)
        strong_locs = np.where(sim_list_array > threshold)[0]
        print('percentage of edges KEPT after Jaccard', len(strong_locs) / len(sim_list))
        new_edgelist = list(edge_list_copy_array[strong_locs])
        sim_list_new = list(sim_list_array[strong_locs])

        G_sim = ig.Graph(list(new_edgelist), edge_attrs={'weight': sim_list_new})
        print('average degree of graph is ', np.mean(G_sim.degree()))
        G_sim.simplify(combine_edges='sum')  # "first"
        print('average degree of SIMPLE graph is ', np.mean(G_sim.degree()))

        print('starting Louvain clustering at', time.ctime())
        if jac_weighted_edges == True:
            start_leiden = time.time()
            print('call leiden weighted for ', self.n_iter_leiden, 'iterations')
            partition = leidenalg.find_partition(G_sim, leidenalg.ModularityVertexPartition, weights='weight',
                                                 n_iterations=self.n_iter_leiden)
            print(time.time() - start_leiden)
        else:
            start_leiden = time.time()
            print('call leiden NOT weighted', self.n_iter_leiden, 'iterations')
            partition = leidenalg.find_partition(G_sim, leidenalg.ModularityVertexPartition, n_iterations=self.n_iter_leiden)
            print(time.time() - start_leiden)
        time_end_louvain = time.time()
        print('Q=', round(partition.quality()))
        louvain_labels = np.asarray(partition.membership)
        louvain_labels = np.reshape(louvain_labels, (n_elements, 1))


        too_big = False
        set_louvain_labels = set(list(louvain_labels.T)[0])
        print('labels found after Leiden', set_louvain_labels)

        cluster_i_loc = np.where(louvain_labels == 0)[0]  # the 0th cluster is the largest one. so if cluster 0 is not too big, then the others wont be too big either
        pop_i = len(cluster_i_loc)
        if pop_i > too_big_factor * n_elements:  # 0.4
            too_big = True
            cluster_big_loc = cluster_i_loc
            list_pop_too_bigs = [pop_i]
            cluster_too_big = 0

        while too_big == True:
            knn_big = 50  # 100 for 8th sep #50 for all the ALPH tests done for 3 professor presentation in start OCT #knn = 5 on the 8-10th oct
            # ef_big = knn_big + 10
            X_data_big = X_data[cluster_big_loc, :]
            knn_struct_big = self.make_knn_struct(X_data_big)
            print(X_data_big.shape)
            louvain_labels_big = self.run_toobig_sublouvain(X_data_big, knn_struct_big, k_nn=knn_big, self_loop=False,
                                                       jac_std=self.jac_std_global)  # knn=200 for 10x
            print('set of new big labels ', set(louvain_labels_big.flatten()))
            louvain_labels_big = louvain_labels_big + 1000  # len(set(louvain_labels.T)[0])
            print('set of new big labels +1000 ', set(list(louvain_labels_big.flatten())))
            pop_list = []
            for item in set(list(louvain_labels_big.flatten())):
                pop_list.append(list(louvain_labels_big.flatten()).count(item))
            print('pop of new big labels', pop_list)
            jj = 0
            print('shape louvain_labels', louvain_labels.shape)
            for j in cluster_big_loc:
                louvain_labels[j] = louvain_labels_big[jj]
                jj = jj + 1
            dummy, louvain_labels = np.unique(list(louvain_labels.flatten()), return_inverse=True)
            print('new set of labels ', set(louvain_labels))
            too_big = False
            set_louvain_labels = set(louvain_labels)  # list(louvain_labels.T)[0])

            louvain_labels = np.asarray(louvain_labels)
            for cluster_ii in set_louvain_labels:
                cluster_ii_loc = np.where(louvain_labels == cluster_ii)[0]
                pop_ii = len(cluster_ii_loc)
                not_yet_expanded = pop_ii not in list_pop_too_bigs
                if pop_ii > too_big_factor * n_elements and not_yet_expanded == True:
                    too_big = True
                    print('cluster', cluster_ii, 'is too big and has population', pop_ii)
                    cluster_big_loc = cluster_ii_loc
                    cluster_big = cluster_ii
                    big_pop = pop_ii
            if too_big == True:
                list_pop_too_bigs.append(big_pop)
                print('cluster', cluster_big, 'is too big with population', big_pop, '. It will be expanded')
        dummy, louvain_labels = np.unique(list(louvain_labels.flatten()), return_inverse=True)
        small_pop_list = []
        small_cluster_list = []
        small_pop_exist = False

        for cluster in set(louvain_labels):
            population = len(np.where(louvain_labels == cluster)[0])

            if population < small_pop:  # 10
                small_pop_exist = True

                small_pop_list.append(list(np.where(louvain_labels == cluster)[0]))
                small_cluster_list.append(cluster)

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

        while small_pop_exist == True:
            small_pop_list = []
            small_pop_exist = False
            for cluster in set(list(louvain_labels.flatten())):
                population = len(np.where(louvain_labels == cluster)[0])
                if population < small_pop:
                    small_pop_exist = True
                    print(cluster, ' has small population of', population, )
                    small_pop_list.append(np.where(louvain_labels == cluster)[0])
            for small_cluster in small_pop_list:
                for single_cell in small_cluster:
                    old_neighbors = neighbor_array[single_cell, :]
                    group_of_old_neighbors = louvain_labels[old_neighbors]
                    group_of_old_neighbors = list(group_of_old_neighbors.flatten())
                    best_group = max(set(group_of_old_neighbors), key=group_of_old_neighbors.count)
                    louvain_labels[single_cell] = best_group

        print('small and big round took ', round(time.time() - time_end_louvain), 'seconds', time.ctime())
        dummy, louvain_labels = np.unique(list(louvain_labels.flatten()), return_inverse=True)
        louvain_labels = list(louvain_labels.flatten())
        print('final labels allocation', set(louvain_labels))
        pop_list = []
        for item in set(louvain_labels):
            pop_list.append(louvain_labels.count(item))
        print('pop of big list is of length and populations', len(pop_list), pop_list)

        self.labels = louvain_labels #list
        return

    def accuracy(self, onevsall=1):

        true_labels = self.true_label
        Index_dict = {}
        alph_labels = self.labels
        N = len(alph_labels)
        n_cancer = list(true_labels).count(onevsall)
        n_pbmc = N - n_cancer

        for k in range(N):
            Index_dict.setdefault(alph_labels[k], []).append(true_labels[k])
        num_groups = len(Index_dict)
        sorted_keys = list(sorted(Index_dict.keys()))
        error_count = []
        pbmc_labels = []
        thp1_labels = []
        fp, fn,tp,tn, precision, recall, f1_score = 0,0,0,0,0,0,0

        for kk in sorted_keys:
            vals = [t for t in Index_dict[kk]]
            majority_val = func_mode(vals)
            if majority_val == onevsall: print('cluster', kk, ' has majority', onevsall, 'with population', len(vals))
            if kk == -1:
                len_unknown = len(vals)
                print('len unknown', len_unknown)
            if (majority_val == onevsall) and (kk != -1):
                thp1_labels.append(kk)
                fp = fp + len([e for e in vals if e != onevsall])
                tp = tp + len([e for e in vals if e == onevsall])
                list_error = [e for e in vals if e != majority_val]
                e_count = len(list_error)
                error_count.append(e_count)
            elif (majority_val != onevsall) and (kk != -1):
                pbmc_labels.append(kk)
                tn = tn + len([e for e in vals if e != onevsall])
                fn = fn + len([e for e in vals if e == onevsall])
                error_count.append(len([e for e in vals if e != majority_val]))

        predict_class_array = np.array(alph_labels)
        alph_labels_array = np.array(alph_labels)
        number_clusters_for_target = len(thp1_labels)
        for cancer_class in thp1_labels:
            predict_class_array[alph_labels_array == cancer_class] = 1
        for benign_class in pbmc_labels:
            predict_class_array[alph_labels_array == benign_class] = 0
        predict_class_array.reshape((predict_class_array.shape[0], -1))
        error_rate = sum(error_count) / N
        n_target = tp + fn
        tnr = tn / n_pbmc
        fnr = fn / n_cancer
        tpr = tp / n_cancer
        fpr = fp / n_pbmc

        if tp != 0 or fn != 0: recall = tp / (tp + fn)  # ability to find all positives
        if tp != 0 or fp != 0: precision = tp / (tp + fp)  # ability to not misclassify negatives as positives
        if precision != 0 or recall != 0:
            f1_score = precision * recall * 2 / (precision + recall)
        majority_truth_labels = np.empty((len(true_labels), 1), dtype=object)

        for cluster_i in set(alph_labels):
            cluster_i_loc = np.where(np.asarray(alph_labels) == cluster_i)[0]
            true_labels = np.asarray(true_labels)
            majority_truth = func_mode(list(true_labels[cluster_i_loc]))
            majority_truth_labels[cluster_i_loc] = majority_truth

        majority_truth_labels = list(majority_truth_labels.flatten())
        accuracy_val = [error_rate, f1_score, tnr, fnr, tpr, fpr, precision,
                            recall, num_groups, n_target]


        return accuracy_val, predict_class_array, majority_truth_labels, number_clusters_for_target

    def run_mainlouvain(self):
        list_roc = []

        time_start_total = time.time()

        time_start_knn = time.time()
        self.knn_struct = self.make_knn_struct()
        time_end_knn_struct = time.time() - time_start_knn
        # Query dataset, k - number of closest elements (returns 2 numpy arrays)
        self.run_sublouvain()
        run_time =time.time() - time_start_total
        print('time elapsed {:.2f} seconds'.format(run_time))

        print('current time is: ', time.ctime())
        targets = list(set(self.true_label))

        N = len(list(self.true_label))
        f1_accumulated =0
        f1_acc_noweighting = 0
        for onevsall_val in targets:
            print('target is', onevsall_val)
            vals_roc, predict_class_array, majority_truth_labels, numclusters_targetval = self.accuracy(onevsall=onevsall_val)
            f1_current = vals_roc[1]
            print('target', onevsall_val,'had f1-score of %.2f' % (f1_current))
            f1_accumulated = f1_accumulated+ f1_current*(list(self.true_label).count(onevsall_val))/N
            f1_acc_noweighting = f1_acc_noweighting + f1_current


            list_roc.append([self.jac_std_global, self.dist_std_local, onevsall_val]+vals_roc +[numclusters_targetval]+ [run_time])

        f1_mean = f1_acc_noweighting / len(targets)
        print("f1-score (unweighted) mean ", f1_mean)
        print('f1-score weighted (by population): ', f1_accumulated)

        df_accuracy = pd.DataFrame(list_roc,
                                   columns=['jac_std_global','dist_std_local','onevsall-target','error rate','f1-score', 'tnr', 'fnr',
                                            'tpr', 'fpr', 'precision', 'recall', 'num_groups','population of target' ,'num clusters','clustering runtime'])


        self.f1_accumulated = f1_accumulated
        self.f1_mean = f1_mean
        self.stats_df = df_accuracy
        self.majority_truth_labels =  majority_truth_labels
        return

def main():
    from sklearn.datasets import make_blobs
    import matplotlib.pyplot as plt
    from sklearn import datasets
    iris = datasets.load_iris()
    X = iris.data
    y=iris.target
    # features, target = make_blobs(n_samples = 5000,
    #               # two feature variables,
    #               n_features = 3,
    #               # four clusters,
    #               centers = 4,
    #               # with .65 cluster standard deviation,
    #               cluster_std = 0.65,
    #               # shuffled,
    #               shuffle = True)
    #
    # # Create a scatterplot of first two features

    p1 = ALPH(X,y)
    p1.run_mainlouvain()
    plt.scatter(X[:,0],X[:,1], c = y)

    # View scatterplot
    plt.show()
    plt.scatter(X[:, 0], X[:, 1], c=p1.labels)
    plt.show()

if __name__ == '__main__':
    main()

