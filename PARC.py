import numpy as np
import pandas as pd
import hnswlib
from scipy.sparse import csr_matrix
import igraph as ig
import leidenalg
import time



class PARC:
    def __init__(self, data, true_label=None, dist_std_local = 2,jac_std_global='median', keep_all_local_dist = 'auto',too_big_factor = 0.4, small_pop = 10, jac_weighted_edges = True,knn = 30, n_iter_leiden=5, random_seed = 42, num_threads=-1, distance = 'l2',time_smallpop = 15):
          #higher dist_std_local means more edges are kept
          #highter jac_std_global means more edges are kept
        if keep_all_local_dist =='auto':
              if data.shape[0]>300000: keep_all_local_dist = True #skips local pruning to increase speed
              else: keep_all_local_dist = False

        self.data = data
        self.true_label= true_label
        self.dist_std_local =dist_std_local
        self.jac_std_global=  jac_std_global ##0.15 is also a recommended value performing empirically similar to 'median'
        self.keep_all_local_dist = keep_all_local_dist
        self.too_big_factor = too_big_factor  ##if a cluster exceeds this share of the entire cell population, then the PARC will be run on the large cluster. at 0.4 it does not come into play
        self.small_pop = small_pop  # smallest cluster population to be considered a community
        self.jac_weighted_edges = jac_weighted_edges
        self.knn = knn
        self.n_iter_leiden = n_iter_leiden
        self.random_seed = random_seed  # enable reproducible Leiden clustering
        self.num_threads = num_threads  # number of threads used in KNN search/construction
        self.distance = distance  # Euclidean distance 'l2' by default; other options 'ip' and 'cosine'
        self.time_smallpop = time_smallpop

    def make_knn_struct(self, too_big=False, big_cluster=None):
        if self.knn >190: print('please provide a lower K_in for KNN graph construction')
        ef_query = max(100, self.knn+1) #ef always should be >K. higher ef, more accuate query
        if too_big == False:
            num_dims = self.data.shape[1]
            n_elements = self.data.shape[0]
            p = hnswlib.Index(space=self.distance, dim=num_dims) # default to Euclidean distance
            p.set_num_threads(self.num_threads)  # allow user to set threads used in KNN construction
            if n_elements<10000:
                ef_param_const = min(n_elements - 10, 500)
                ef_query = ef_param_const
                print('setting ef_construction to', )
            else:ef_param_const=200
            if num_dims>30:
                p.init_index(max_elements=n_elements, ef_construction=ef_param_const, M=48) ## good for scRNA seq where dimensionality is high
            else: p.init_index(max_elements=n_elements, ef_construction=200, M=30,)
            p.add_items(self.data)
        if too_big == True:
            num_dims = big_cluster.shape[1]
            n_elements = big_cluster.shape[0]
            p = hnswlib.Index(space='l2', dim=num_dims)
            p.init_index(max_elements=n_elements, ef_construction=200, M=30)
            p.add_items(big_cluster)
        p.set_ef(ef_query)  # ef should always be > k
        return p

    def make_csrmatrix_noselfloop(self, neighbor_array, distance_array):
        local_pruning_bool = not(self.keep_all_local_dist)
        if local_pruning_bool == True: print('commencing local pruning based on Euclidean distance metric at', self.dist_std_local, 's.dev above mean')
        row_list = []
        col_list = []
        weight_list = []
        neighbor_array = neighbor_array  # not listed in in any order of proximity
        #print('size neighbor array', neighbor_array.shape)
        num_neigh = neighbor_array.shape[1]
        distance_array = distance_array
        n_neighbors = neighbor_array.shape[1]
        n_cells = neighbor_array.shape[0]
        rowi = 0
        count_0dist = 0
        discard_count = 0

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
                            dist = dist+0.0001
                        weight_list.append(dist)

                rowi = rowi + 1

        if local_pruning_bool == False:  # dont prune based on distance
            row_list.extend(list(np.transpose(np.ones((n_neighbors, n_cells)) * range(0, n_cells)).flatten()))
            col_list = neighbor_array.flatten().tolist()
            weight_list = (1. / (distance_array.flatten() + 0.1)).tolist()
        #if local_pruning_bool == True: print('share of neighbors discarded in local distance pruning %.1f' % (discard_count / neighbor_array.size))

        csr_graph = csr_matrix((np.array(weight_list), (np.array(row_list), np.array(col_list))),
                               shape=(n_cells, n_cells))
        return csr_graph

    def func_mode(self, ll):  # return MODE of list
        # If multiple items are maximal, the function returns the first one encountered.
        return max(set(ll), key=ll.count)
    
    def run_toobig_subPARC(self, X_data, jac_std_toobig=0.3,
                              jac_weighted_edges=True):
        n_elements = X_data.shape[0]
        hnsw = self.make_knn_struct(too_big = True, big_cluster = X_data)
        if n_elements <= 10: print('consider increasing the too_big_factor')
        if n_elements> self.knn: knnbig=self.knn
        else: knnbig = int(max(5,0.2*n_elements))

        neighbor_array, distance_array = hnsw.knn_query(X_data, k=knnbig)
        #print('shapes of neigh and dist array', neighbor_array.shape, distance_array.shape)
        csr_array = self.make_csrmatrix_noselfloop(neighbor_array, distance_array)
        sources, targets = csr_array.nonzero()
        mask = np.zeros(len(sources), dtype=bool)

        mask |= (csr_array.data > (np.mean(csr_array.data) + np.std(csr_array.data) * 5)) #smaller distance means stronger edge

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
        print('jac threshold %.3f' % threshold)
        print('jac std %.3f' % np.std(sim_list))
        print('jac mean %.3f' % np.mean(sim_list))
        strong_locs = np.where(sim_list_array > threshold)[0]
        for ii in strong_locs: new_edgelist.append(edgelist_copy[ii])
        sim_list_new = list(sim_list_array[strong_locs])

        if jac_weighted_edges == True:
            G_sim = ig.Graph(n=n_elements, edges=list(new_edgelist), edge_attrs={'weight': sim_list_new})
        else:
            G_sim = ig.Graph(n=n_elements, edges = list(new_edgelist))
        G_sim.simplify(combine_edges='sum')
        if jac_weighted_edges == True:
            partition = leidenalg.find_partition(G_sim, leidenalg.ModularityVertexPartition, weights='weight',
                                                 n_iterations=self.n_iter_leiden, seed=self.random_seed)
        else:
            partition = leidenalg.find_partition(G_sim, leidenalg.ModularityVertexPartition,
                                                 n_iterations=self.n_iter_leiden,seed=self.random_seed)
        #print('Q= %.2f' % partition.quality())
        PARC_labels_leiden = np.asarray(partition.membership)
        PARC_labels_leiden = np.reshape(PARC_labels_leiden, (n_elements, 1))
        small_pop_list = []
        small_cluster_list = []
        small_pop_exist = False
        dummy, PARC_labels_leiden = np.unique(list(PARC_labels_leiden.flatten()), return_inverse=True)
        for cluster in set(PARC_labels_leiden):
            population = len(np.where(PARC_labels_leiden == cluster)[0])
            if population < 10:
                small_pop_exist = True
                small_pop_list.append(list(np.where(PARC_labels_leiden == cluster)[0]))
                small_cluster_list.append(cluster)

        for small_cluster in small_pop_list:
            for single_cell in small_cluster:
                old_neighbors = neighbor_array[single_cell, :]
                group_of_old_neighbors = PARC_labels_leiden[old_neighbors]
                group_of_old_neighbors = list(group_of_old_neighbors.flatten())
                available_neighbours = set(group_of_old_neighbors) - set(small_cluster_list)
                if len(available_neighbours) > 0:
                    available_neighbours_list = [value for value in group_of_old_neighbors if
                                                 value in list(available_neighbours)]
                    best_group = max(available_neighbours_list, key=available_neighbours_list.count)
                    PARC_labels_leiden[single_cell] = best_group

        time_smallpop_start = time.time()
        print('handling fragments' )
        while (small_pop_exist) == True & (time.time() - time_smallpop_start < self.time_smallpop):
            small_pop_list = []
            small_pop_exist = False
            for cluster in set(list(PARC_labels_leiden.flatten())):
                population = len(np.where(PARC_labels_leiden == cluster)[0])
                if population < 10:
                    small_pop_exist = True

                    small_pop_list.append(np.where(PARC_labels_leiden == cluster)[0])
            for small_cluster in small_pop_list:
                for single_cell in small_cluster:
                    old_neighbors = neighbor_array[single_cell, :]
                    group_of_old_neighbors = PARC_labels_leiden[old_neighbors]
                    group_of_old_neighbors = list(group_of_old_neighbors.flatten())
                    best_group = max(set(group_of_old_neighbors), key=group_of_old_neighbors.count)
                    PARC_labels_leiden[single_cell] = best_group

        dummy, PARC_labels_leiden = np.unique(list(PARC_labels_leiden.flatten()), return_inverse=True)
        self.labels = PARC_labels_leiden
        return PARC_labels_leiden

    def run_subPARC(self):
        X_data = self.data
        too_big_factor = self.too_big_factor
        small_pop =self.small_pop
        jac_std_global=self.jac_std_global
        jac_weighted_edges=self.jac_weighted_edges
        knn =  self.knn
        n_elements = X_data.shape[0]

        #print('number of k-nn is', knn, too_big_factor, 'small pop is', small_pop)

        neighbor_array, distance_array = self.knn_struct.knn_query(X_data, k=knn)

        csr_array = self.make_csrmatrix_noselfloop(neighbor_array, distance_array)
        sources, targets = csr_array.nonzero()

        edgelist = list(zip(sources, targets))

        edgelist_copy = edgelist.copy()

        G = ig.Graph(edgelist, edge_attrs={'weight': csr_array.data.tolist()})
        #print('average degree of prejacard graph is %.1f'% (np.mean(G.degree())))
        #print('computing Jaccard metric')
        sim_list = G.similarity_jaccard(pairs=edgelist_copy)

        print('commencing global pruning')

        sim_list_array = np.asarray(sim_list)
        edge_list_copy_array = np.asarray(edgelist_copy)

        if jac_std_global== 'median':
            threshold = np.median(sim_list)
        else:
            threshold = np.mean(sim_list) - jac_std_global* np.std(sim_list)
        strong_locs = np.where(sim_list_array > threshold)[0]
        #print('Share of edges kept after Global Pruning %.2f' % (len(strong_locs) / len(sim_list)), '%')
        new_edgelist = list(edge_list_copy_array[strong_locs])
        sim_list_new = list(sim_list_array[strong_locs])

        G_sim = ig.Graph(n=n_elements, edges = list(new_edgelist), edge_attrs={'weight': sim_list_new})
        #print('average degree of graph is %.1f' % (np.mean(G_sim.degree())))
        G_sim.simplify(combine_edges='sum')  # "first"
        #print('average degree of SIMPLE graph is %.1f' % (np.mean(G_sim.degree())))
        print('commencing community detection')
        if jac_weighted_edges == True:
            start_leiden = time.time()
            #print('call leiden on weighted graph for ', self.n_iter_leiden, 'iterations')
            partition = leidenalg.find_partition(G_sim, leidenalg.ModularityVertexPartition, weights='weight',
                                                 n_iterations=self.n_iter_leiden, seed=self.random_seed)
            print(time.time() - start_leiden)
        else:
            start_leiden = time.time()
            #print('call leiden on unweighted graph', self.n_iter_leiden, 'iterations')
            partition = leidenalg.find_partition(G_sim, leidenalg.ModularityVertexPartition, n_iterations=self.n_iter_leiden, seed=self.random_seed)
            #print(time.time() - start_leiden)
        time_end_PARC = time.time()
        #print('Q= %.1f' % (partition.quality()))
        PARC_labels_leiden = np.asarray(partition.membership)
        PARC_labels_leiden = np.reshape(PARC_labels_leiden, (n_elements, 1))


        too_big = False

        #print('labels found after Leiden', set(list(PARC_labels_leiden.T)[0])) will have some outlier clusters that need to be added to a cluster if a cluster has members that are KNN

        cluster_i_loc = np.where(PARC_labels_leiden == 0)[0]  # the 0th cluster is the largest one. so if cluster 0 is not too big, then the others wont be too big either
        pop_i = len(cluster_i_loc)
        if pop_i > too_big_factor * n_elements:  # 0.4
            too_big = True
            cluster_big_loc = cluster_i_loc
            list_pop_too_bigs = [pop_i]
            cluster_too_big = 0

        while too_big == True:
            knn_big = 50
            X_data_big = X_data[cluster_big_loc, :]
            PARC_labels_leiden_big = self.run_toobig_subPARC(X_data_big)
            #print('set of new big labels ', set(PARC_labels_leiden_big.flatten()))
            PARC_labels_leiden_big = PARC_labels_leiden_big + 1000
            #print('set of new big labels +1000 ', set(list(PARC_labels_leiden_big.flatten())))
            pop_list = []

            for item in set(list(PARC_labels_leiden_big.flatten())):
                pop_list.append([item, list(PARC_labels_leiden_big.flatten()).count(item)])
            print('pop of big clusters', pop_list)
            jj = 0
            print('shape PARC_labels_leiden', PARC_labels_leiden.shape)
            for j in cluster_big_loc:
                PARC_labels_leiden[j] = PARC_labels_leiden_big[jj]
                jj = jj + 1
            dummy, PARC_labels_leiden = np.unique(list(PARC_labels_leiden.flatten()), return_inverse=True)
            print('new set of labels ', set(PARC_labels_leiden))
            too_big = False
            set_PARC_labels_leiden = set(PARC_labels_leiden)

            PARC_labels_leiden = np.asarray(PARC_labels_leiden)
            for cluster_ii in set_PARC_labels_leiden:
                cluster_ii_loc = np.where(PARC_labels_leiden == cluster_ii)[0]
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
        dummy, PARC_labels_leiden = np.unique(list(PARC_labels_leiden.flatten()), return_inverse=True)
        small_pop_list = []
        small_cluster_list = []
        small_pop_exist = False

        for cluster in set(PARC_labels_leiden):
            population = len(np.where(PARC_labels_leiden == cluster)[0])

            if population < small_pop:  # 10
                small_pop_exist = True

                small_pop_list.append(list(np.where(PARC_labels_leiden == cluster)[0]))
                small_cluster_list.append(cluster)

        for small_cluster in small_pop_list:

            for single_cell in small_cluster:
                old_neighbors = neighbor_array[single_cell, :]
                group_of_old_neighbors = PARC_labels_leiden[old_neighbors]
                group_of_old_neighbors = list(group_of_old_neighbors.flatten())
                available_neighbours = set(group_of_old_neighbors) - set(small_cluster_list)
                if len(available_neighbours) > 0:
                    available_neighbours_list = [value for value in group_of_old_neighbors if
                                                 value in list(available_neighbours)]
                    best_group = max(available_neighbours_list, key=available_neighbours_list.count)
                    PARC_labels_leiden[single_cell] = best_group
        time_smallpop_start = time.time()
        while (small_pop_exist == True) & ( (time.time()-time_smallpop_start)<self.time_smallpop):
            small_pop_list = []
            small_pop_exist = False
            for cluster in set(list(PARC_labels_leiden.flatten())):
                population = len(np.where(PARC_labels_leiden == cluster)[0])
                if population < small_pop:
                    small_pop_exist = True
                    print(cluster, ' has small population of', population, )
                    small_pop_list.append(np.where(PARC_labels_leiden == cluster)[0])
            for small_cluster in small_pop_list:
                for single_cell in small_cluster:
                    old_neighbors = neighbor_array[single_cell, :]
                    group_of_old_neighbors = PARC_labels_leiden[old_neighbors]
                    group_of_old_neighbors = list(group_of_old_neighbors.flatten())
                    best_group = max(set(group_of_old_neighbors), key=group_of_old_neighbors.count)
                    PARC_labels_leiden[single_cell] = best_group

        dummy, PARC_labels_leiden = np.unique(list(PARC_labels_leiden.flatten()), return_inverse=True)
        PARC_labels_leiden = list(PARC_labels_leiden.flatten())
        #print('final labels allocation', set(PARC_labels_leiden))
        pop_list = []
        for item in set(PARC_labels_leiden):
            pop_list.append((item, PARC_labels_leiden.count(item)))
        print('list of cluster labels and populations', len(pop_list), pop_list)

        self.labels = PARC_labels_leiden #list
        return

    def accuracy(self, onevsall=1):

        true_labels = self.true_label
        Index_dict = {}
        PARC_labels = self.labels
        N = len(PARC_labels)
        n_cancer = list(true_labels).count(onevsall)
        n_pbmc = N - n_cancer

        for k in range(N):
            Index_dict.setdefault(PARC_labels[k], []).append(true_labels[k])
        num_groups = len(Index_dict)
        sorted_keys = list(sorted(Index_dict.keys()))
        error_count = []
        pbmc_labels = []
        thp1_labels = []
        fp, fn,tp,tn, precision, recall, f1_score = 0,0,0,0,0,0,0

        for kk in sorted_keys:
            vals = [t for t in Index_dict[kk]]
            majority_val = self.func_mode(vals)
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

        predict_class_array = np.array(PARC_labels)
        PARC_labels_array = np.array(PARC_labels)
        number_clusters_for_target = len(thp1_labels)
        for cancer_class in thp1_labels:
            predict_class_array[PARC_labels_array == cancer_class] = 1
        for benign_class in pbmc_labels:
            predict_class_array[PARC_labels_array == benign_class] = 0
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

        for cluster_i in set(PARC_labels):
            cluster_i_loc = np.where(np.asarray(PARC_labels) == cluster_i)[0]
            true_labels = np.asarray(true_labels)
            majority_truth = self.func_mode(list(true_labels[cluster_i_loc]))
            majority_truth_labels[cluster_i_loc] = majority_truth

        majority_truth_labels = list(majority_truth_labels.flatten())
        accuracy_val = [error_rate, f1_score, tnr, fnr, tpr, fpr, precision,
                            recall, num_groups, n_target]


        return accuracy_val, predict_class_array, majority_truth_labels, number_clusters_for_target

    def run_PARC(self):
        print('input data has shape', self.data.shape[0], '(samples) x',self.data.shape[1], '(features)')
        if self.true_label is None:
            self.true_label = [1]*self.data.shape[0]
        list_roc = []

        time_start_total = time.time()

        time_start_knn = time.time()
        self.knn_struct = self.make_knn_struct()
        time_end_knn_struct = time.time() - time_start_knn
        # Query dataset, k - number of closest elements (returns 2 numpy arrays)
        self.run_subPARC()
        run_time =time.time() - time_start_total
        print('time elapsed {:.1f} seconds'.format(run_time))


        targets = list(set(self.true_label))
        N = len(list(self.true_label))
        self.f1_accumulated = 0
        self.f1_mean = 0
        self.stats_df = pd.DataFrame({'jac_std_global':[self.jac_std_global],'dist_std_local':[self.dist_std_local], 'runtime(s)':[run_time]})
        self.majority_truth_labels = []
        if len(targets)>1:
            f1_accumulated =0
            f1_acc_noweighting = 0
            for onevsall_val in targets:
                print('target is', onevsall_val)
                vals_roc, predict_class_array, majority_truth_labels, numclusters_targetval = self.accuracy(onevsall=onevsall_val)
                f1_current = vals_roc[1]
                print('target', onevsall_val,'has f1-score of %.2f' % (f1_current*100))
                f1_accumulated = f1_accumulated+ f1_current*(list(self.true_label).count(onevsall_val))/N
                f1_acc_noweighting = f1_acc_noweighting + f1_current


                list_roc.append([self.jac_std_global, self.dist_std_local, onevsall_val]+vals_roc +[numclusters_targetval]+ [run_time])

            f1_mean = f1_acc_noweighting / len(targets)
            print("f1-score (unweighted) mean %.2f" % (f1_mean*100),'%')
            print('f1-score weighted (by population) %.2f' % (f1_accumulated*100),'%')

            df_accuracy = pd.DataFrame(list_roc,
                                       columns=['jac_std_global','dist_std_local','onevsall-target','error rate','f1-score', 'tnr', 'fnr',
                                                'tpr', 'fpr', 'precision', 'recall', 'num_groups','population of target' ,'num clusters','clustering runtime'])


            self.f1_accumulated = f1_accumulated
            self.f1_mean = f1_mean
            self.stats_df = df_accuracy
            self.majority_truth_labels =  majority_truth_labels
        return

def main():
    #dummy example to check code runs
    import matplotlib.pyplot as plt
    from sklearn import datasets
    iris = datasets.load_iris()
    X = iris.data
    y=iris.target

    p1 = PARC(X, true_label =None, too_big_factor=0.1) #without labels
    p1.run_PARC()
    print(type(p1.labels), p1.stats_df)
    plt.scatter(X[:,0],X[:,1], c = y)

    # View scatterplot
    plt.show()
    plt.scatter(X[:, 0], X[:, 1], c=p1.labels)
    plt.show()

    p1 = PARC(X,true_label=y)  # without labels
    p1.run_PARC()
    plt.scatter(X[:, 0], X[:, 1], c=y)
    print(type(p1.labels), p1.stats_df)
    # View scatterplot
    plt.show()
    plt.scatter(X[:, 0], X[:, 1], c=p1.labels)
    plt.show()

def main1():
    from scipy.io import loadmat
    from scipy import stats
    import matplotlib.pyplot as plt
    annots = loadmat('/home/shobi/Thesis/Data/Kelvin2020/NPcell.mat')
    #annots = loadmat('/home/shobi/Thesis/Data/Kelvin2020/NPcell(26feature).mat')
    print('annots', annots)

    '''
    labels = []
    with open('/home/shobi/Thesis/Data/Kelvin2020/parc_clusters_n86389_d95Feb28.txt', 'r') as filehandle:
        for line in filehandle:
            # remove linebreak which is the last character of the string
            currentPlace = int(line.strip())
            labels.append(currentPlace)
    print('labels', len(labels))
    print('set of labels', set(labels))
    '''
    print('phenotypes shape', annots['Phenotypes'].shape, annots['PhenoLabel'].shape)

    print('cell ID', annots['CellID'])
    print('header', annots['__header__'])
    print('globals', annots['__globals__'])
    print('phenotypes', annots['Phenotypes'])
    print('batch', annots['Batch'])

    col_pheno = []
    col_pheno_nofluo=[]
    for i in range(100):#43
        print(i)
        col_pheno.append(annots["PhenoLabel"][0][i][0])
    print('list col', col_pheno)
    #col_pheno.remove('Cell ID') use for feat26
    true_label = annots['Batch'][0]
    print('length true', len(true_label), len(set(true_label)))

    data_mat = annots['Phenotypes'].T  # (features x cells).T makes it  cellxfeatures
    print('shape data mat', data_mat.shape)
    df = pd.DataFrame(data_mat)
    print(df.head)

    df.columns = col_pheno
    #col_pheno = list(df.columns)

    #col_pheno_nofluo = col_pheno.copy()
    #fluo_list = ['Fluorescence Ch1 (Peak)', 'Fluorescence Ch1 (Area)', 'Fluorescence density (1d) Ch1', 'Fluorescence density (3d) Ch1', 'Fluorescence-Phase correlation Ch1', 'Fluorescence Ch2 (Peak)', 'Fluorescence Ch2 (Area)', 'Fluorescence density (1d) Ch2', 'Fluorescence density (3d) Ch2','Fluorescence-Phase correlation Ch2']
    #for f in fluo_list:
    #    col_pheno_nofluo.remove(f)
    df['Batch']= annots['Batch'][0]
    df['CellID']=annots['CellID'][0]
    df['Unique_ID'] = 'B'+ df['Batch'].astype('str')+ "C"+df['CellID'].astype('str')
    df =df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]
    #ax = df[['Volume','Fluorescence Ch1 (Area)']].plot.hist(bins=100, alpha=0.5)
    #plt.show()


    df[col_pheno]=(df[col_pheno] - df[col_pheno].mean())/df[col_pheno].std()



    #for i in col_pheno:   print('mean of df col before outlier removal', df[i].mean())


    for i in col_pheno:
        p99 = df[i].quantile(.9999)
        p01=df[i].quantile(.0001)
        df = df[(df[i]<=p99) & (df[i]>=p01)]

    #ax=df[['Volume','Fluorescence Ch1 (Area)']].plot.hist(bins=150, alpha=0.5)
    #plt.show()
    #print('shape df', df.shape, df.head)

    #col_pheno = col_pheno_nofluo
    #data_mat=np.array(df[['Volume','Fluorescence Ch1 (Area)']].values, dtype = np.float64)
    data_mat = np.array(df[col_pheno].values[:,0:95], dtype=np.float64)
    print('shape of data mat', data_mat.shape)
    data_mat_vis = np.array(df[col_pheno].values, dtype = np.float64)
    import umap
    print('start umap', time.ctime())
    #X = umap.UMAP().fit_transform(data_mat)
    print('end umap',time.ctime())
    X = np.loadtxt('/home/shobi/Thesis/Data/Kelvin2020/umap_n86389Feb28.txt')
    def plot_hist(X,df,col_pheno):

        col_label = 0
        for set_num in range(7):#10

            fig, ax = plt.subplots(2, 3, sharey=True)#5
            cc = 0
            #for i, j in zip([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]):
            for i, j in zip([0, 0,0, 1, 1,1], [0, 1, 2, 0, 1, 2]):
                ax[i, j].scatter(X[:, 0], X[:, 1], c=df[col_pheno[col_label + cc]].values, cmap='viridis', s=1, alpha=0.5)
                ax[i, j].set_title(col_pheno[col_label + cc])
                cc = cc + 1
            print('saving histogram set num', set_num)
            plt.show()
            # plt.savefig('/home/shobi/Thesis/Data/Kelvin2020/Histograms/set_' + str(set_num) + '.png')
            col_label = col_label + 6 #10

    true_label = df['Batch'].values


    print('clean matrix', data_mat.shape)
    # print('first col',data_mat[3,:])

    #print('mean', np.mean(data_mat, axis=0))
    #data_mat = stats.zscore(data_mat)
    num_feat = data_mat.shape[1]
    num_cells = data_mat.shape[0]
    p1 = PARC(data_mat, true_label=true_label)  # without labels
    p1.run_PARC()
    labels = p1.labels


    from MulticoreTSNE import MulticoreTSNE as TSNE
    import umap
    df['PARC']=labels
    df['umap_x'] = X[:,0]
    df['umap_y'] = X[:, 1]
    df = df.sort_values(by='PARC')
    #X = TSNE().fit_transform(data_mat[:,0:10])
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    #X = pca.fit_transform(data_mat_vis)
    #plt.scatter(X[:, 0], X[:, 1], c=p1.labels, cmap='tab20')
    #plt.title('PARC on input shape:'+str(data_mat.shape[1]))
    #plt.show()

    import seaborn as sns
    from matplotlib import cm
    cmap_div = sns.diverging_palette(240, 10, as_cmap=True)
    cmap = cm.get_cmap('nipy_spectral')
    a = cmap(np.linspace(0, 1, len(set(labels)))).tolist()
    lut = dict(zip(set(labels), [a[i] for i in range(0,len(set(labels)))]))
    row_colors = df['PARC'].map(lut)
    #col_pheno.remove("Dry Mass var")
    #col_pheno.remove("Phase Radial Distribution")
    g = sns.clustermap(df[col_pheno], row_cluster=False, col_cluster=True, cmap=cmap_div, row_colors=row_colors, vmin=-3,vmax=3.5,xticklabels=1)
    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=45, horizontalalignment='right')
    #g.set_xticklabels(g.get_xticklabels(), rotation=45, horizontalalignment='right')
    plt.show()
    df_mean = df.groupby('PARC')[col_pheno].mean()
    df_mean['PARC'] = df_mean.index
    print('mean df', df_mean)
    row_colors_mean = df_mean['PARC'].map(lut)

    g = sns.clustermap(df_mean[col_pheno], row_cluster=True, col_cluster=True, cmap=cmap_div, row_colors=row_colors_mean, vmin=-3, vmax=3.5,xticklabels=1)
    #g.set_xticklabels(g.get_xticklabels(), rotation=45, horizontalalignment='right')
    plt.show()
    new_row_order = [int(item.get_text()) for item in g.ax_heatmap.yaxis.get_majorticklabels()]
    print(new_row_order, 'clustermap row order')
    labels_array= df['PARC'].values#np.asarray(labels).astype(int)
    cell_order = np.asarray(labels) #placeholder
    count_i=0
    for i,ii in enumerate(new_row_order):
        where = np.where(labels_array==ii)[0]
        print('i,ii,where',i, ii, where)
        cell_order[where]=i
    print('cell order', cell_order)
    df['heatmap_order'] = cell_order
    df = df.sort_values(by='heatmap_order')
    print(df.head)
    print('lut', lut)
    df.reset_index()
    row_colors = df['PARC'].map(lut)
    # col_pheno.remove("Dry Mass var")
    # col_pheno.remove("Phase Radial Distribution")
    g = sns.clustermap(df[col_pheno], row_cluster=False, col_cluster=True, cmap=cmap_div, row_colors=row_colors,
                       vmin=-3, vmax=3.5,xticklabels=1)
    #g.set_xticklabels(g.get_xticklabels(), rotation=45, horizontalalignment='right')
    plt.show()

    df_labels = df[['Batch', "CellID", 'Unique_ID', "PARC", "umap_x", "umap_y"]]

    df_labels.to_csv(
        '/home/shobi/Thesis/Data/Kelvin2020/parc_clusters_n' + str(num_cells) + '_d' + str(num_feat) + 'Mar2.csv')
    np.savetxt(     '/home/shobi/Thesis/Data/Kelvin2020/parc_clusters_n' + str(num_cells) + '_d' + str(num_feat) + 'Mar2.txt',     p1.labels)

    print('start umap', time.ctime())

    # X = np.loadtxt('/home/shobi/Thesis/Data/Kelvin2020/umap_d42_n87456Mar2.txt')
    for cluster_i in set(labels):
        loc_i = np.where(np.asarray(labels) ==cluster_i)[0]
        x=X[loc_i, 0]
        y=X[loc_i, 1]
        plt.scatter(x, y, c=lut[cluster_i], s=2, alpha=0.5, label = str(cluster_i))
        plt.annotate(str(cluster_i), xytext=(np.mean(x), np.mean(y)), xy=(np.mean(x), np.mean(y)), color='black',
                       weight='bold')
    plt.title('PARC on input shape:' + str(data_mat.shape[1]))
    plt.legend(markerscale=3)
    plt.show()

    #print('shape of umap', X.shape)

    print('end umap', time.ctime())


    np.savetxt('/home/shobi/Thesis/Data/Kelvin2020/umap_d' +str(num_feat)+'_n'+str(num_cells)+'Mar2.txt', X)

def main1():
    no_fluo =[]
    fluo= []
    with open('/home/shobi/Thesis/Data/Kelvin2020/Analysis/d95/parc_clusters_n86389_d95Mar2.txt', 'r') as filehandle:
        for line in filehandle:
            # remove linebreak which is the last character of the string
            currentPlace = (line.strip())
            no_fluo.append(currentPlace)
    with open('/home/shobi/Thesis/Data/Kelvin2020/Analysis/d100/parc_clusters_n86389_d100Mar2.txt', 'r') as filehandle:
        for line in filehandle:
            # remove linebreak which is the last character of the string
            currentPlace = (line.strip())
            fluo.append(currentPlace)
    from sklearn.metrics.cluster import adjusted_rand_score
    print('ari for clustering', adjusted_rand_score(np.asarray(fluo), np.asarray(no_fluo)))

if __name__ == '__main__':
    main()

