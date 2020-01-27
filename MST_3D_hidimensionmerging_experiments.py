"""
Minimum Spanning Tree Clustering
being used after the 3 prof ppt in oct 1
"""
from __future__ import division
import Louvain_igraph_Jac24Sept as ls
import numpy as np
import sys
import igraph as ig
from scipy import sparse
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components
from scipy.sparse.csgraph._validation import validate_graph
from sklearn.utils import check_array
import time
import networkx
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial import distance
from random import shuffle
import Plotting_3D



class MSTClustering3D(BaseEstimator, ClusterMixin):
    """Minimum Spanning Tree Clustering

    Parameters
    ----------
    cutoff : float, int, optional
        either the number of edges to cut (if cutoff >= 1) or the fraction of
        edges to cut (if 0 < cutoff < 1). See also the ``cutoff_scale``
        parameter.
    cutoff_scale : float, optional
        minimum size of edges. All edges larger than cutoff_scale will be
        removed (see also ``cutoff`` parameter).
    min_cluster_size : int (default: 1)
        minimum number of points per cluster. Points belonging to smaller
        clusters will be assigned to the background.
    approximate : bool, optional (default: True)
        If True, then compute the approximate minimum spanning tree using
        n_neighbors nearest neighbors. If False, then compute the full
        O[N^2] edges (see Notes, below).
    n_neighbors : int, optional (default: 20)
        maximum number of neighbors of each point used for approximate
        Euclidean minimum spanning tree (MST) algorithm.  Referenced only
        if ``approximate`` is False. See Notes below.
    metric : string (default "euclidean")
        Distance metric to use in computing distances. If "precomputed", then
        input is a [n_samples, n_samples] matrix of pairwise distances (either
        sparse, or dense with NaN/inf indicating missing edges)
    metric_params : dict or None (optional)
        dictionary of parameters passed to the metric. See documentation of
        sklearn.neighbors.NearestNeighbors for details.

    Attributes
    ----------
    full_tree_ : sparse array, shape (n_samples, n_samples)
        Full minimum spanning tree over the fit data
    T_trunc_ : sparse array, shape (n_samples, n_samples)
        Non-connected graph over the final clusters
    labels_: array, length n_samples
        Labels of each point

    Notes
    -----
    This routine uses an approximate Euclidean minimum spanning tree (MST)
    to perform hierarchical clustering.  A true Euclidean minimum spanning
    tree naively costs O[N^3].  Graph traversal algorithms only help so much,
    because all N^2 edges must be used as candidates.  In this approximate
    algorithm, we use k << N edges from each point, so that the cost is only
    O[Nk log(Nk)]. For k = N, the approximation is exact; in practice for
    well-behaved data sets, the result is exact for k << N.
    knn=30
    """

    def __init__(self, min_cluster_size=10,
                 approximate=True, n_neighbors=30,
                 metric='euclidean', metric_params=None, sigma_factor=None, tooclosefactor=None,
                 max_labels=None, true_label=None,precomputed = False, X_fit_original = None ):
        # NOTE metric = 'euclidean' is the same as minkowski with p=2

        # self.cutoff_scale = cutoff_scale
        self.min_cluster_size = min_cluster_size
        self.approximate = approximate
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.metric_params = metric_params
        self.sigma_factor = sigma_factor
        self.tooclosefactor = tooclosefactor
        self.max_labels = max_labels
        self.true_label = true_label
        self.X_fit_original = X_fit_original

    def apply_sigma(self, sigma_factor, max_input_label):
        X_dict = {}  # dictionary containing tuple coords in each cluster. keys are clusters
        X_dict_mean = {}
        X_dict_population = {}
        N = len(self.full_tree_.data)
        mask = np.zeros(N, dtype=bool)
        mask |= (self.full_tree_.data > (self.mean_edge_distance + sigma_factor * self.std_edge_distance))
        if self.metric =='precomputed': #for assigning the small clusters to the big cluster
            X= self.X_fit_original
            X_high = X
            print(X.shape, 'the shape of the codebook')
        else:
            X_high = self.X_fit_original#X = self.X_fit_
            print('assign X_high to shape', X_high.shape)
            X = self.X_fit_

        # mask |= (self.full_tree_.data > self.cutoff_scale)

        # Trim the tree
        cluster_graph = self.full_tree_.copy()
        n_components_1, labels_1 = connected_components(cluster_graph,
                                                    directed=False)
        print('num connected components in MST before removing weak edges:', n_components_1)
        # Eliminate zeros from cluster_graph for efficiency.
        # We want to do this:
        #    cluster_graph.data[mask] = 0
        #    cluster_graph.eliminate_zeros()
        # but there could be explicit zeros in our data!
        # So we call eliminate_zeros() with a stand-in data array,
        # then replace the data when we're finished.
        original_data = cluster_graph.data
        cluster_graph.data = np.arange(1, len(cluster_graph.data) + 1)
        cluster_graph.data[mask] = 0
        cluster_graph.eliminate_zeros()
        cluster_graph.data = original_data[cluster_graph.data.astype(int) - 1]

        # find connected components
        n_components, labels = connected_components(cluster_graph,
                                                    directed=False)
        print('after sigma ',sigma_factor,'num connected components after removing weak edges:', n_components)
        # print('labels:', labels )

        # remove clusters with fewer than min_cluster_size
        counts = np.bincount(labels)
        to_remove = np.where(counts < self.min_cluster_size)[0]

        if len(to_remove) > 0:
            for i in to_remove:
                labels[labels == i] = -1
            dummy, labels = np.unique(labels, return_inverse=True)
            labels -= 1  # keep -1 labels the same
            print('number of large clusters is', len(set(labels)))
            if len(set(labels)) > max_input_label: return X_dict_mean, X_dict, labels
            # print('values of the labels after removing small clusters: ',dummy)
            # print('index of the label array (above), which we use as the new labels', labels)
        # update cluster_graph by eliminating non-clusters
        # operationally, this means zeroing-out rows & columns where
        # the label is negative.
        I = sparse.eye(len(labels))
        I.data[0, labels < 0] = 0

        # we could just do this:
        #   cluster_graph = I * cluster_graph * I
        # but we want to be able to eliminate the zeros, so we use
        # the same indexing trick as above
        original_data = cluster_graph.data
        cluster_graph.data = np.arange(1, len(cluster_graph.data) + 1)
        cluster_graph = I * cluster_graph * I
        cluster_graph.eliminate_zeros()
        cluster_graph.data = original_data[cluster_graph.data.astype(int) - 1]

        X_big = X[labels != -1]
        # print('shape of X_big ',X_big.shape)
        X_small = X[labels == -1]
        # print('x_small shape: ', X_small.shape)
        if X_small.shape[0] > 0:
            labels_big = labels[labels != -1]
            neigh = KNeighborsClassifier(n_neighbors=3)
            neigh.fit(X_big, labels_big)
            # print('made KneighborClassifier')
            y_small = neigh.predict(X_small)
            # print('y_small shape:',y_small.shape)
            y_small_ix = np.where(labels == -1)[0]
            # print('made outlier labels of length', len(y_small_ix))
            ii = 0
            for iy in y_small_ix:
                labels[iy] = y_small[ii]
                # print(y_small[ii])
                ii = ii + 1
        # print('completed updating labels of length', len(labels))
        # print('shape of X,' ,X)
        # print(labels)
        labels_knn_outlier = list(labels)

        mean_array = np.empty([n_components, 3])
        Ntot = len(labels)
        # print(X[Ntot-1])
        # print(labels[Ntot-1])
        #if self.metric =='precomputed':
        #mean_array = np.empty([n_components, self.X.shape[1]])
        for dd in range(X_high.shape[0]):
            ll = []
            for i in range(X_high.shape[1]):
                ll.append(X_high[dd, i])
            # print(dd,x,y, labels[dd])
            X_dict.setdefault(labels[dd], []).append(tuple(ll))
        num_big_components = len(X_dict)

        for key in range(num_big_components):
            mean_list = []
            for i in range(X_high.shape[1]):
                # print('key',key)
                i_list= [t[i] for t in X_dict[key]]
                #mean_array[key, i] = np.mean(i_list)
                mean_list.append(np.mean(i_list))
            X_dict_mean.setdefault(key, []).append(tuple(mean_list))
            X_dict_population.setdefault(key, []).append(counts[key])

        '''
        else:
            for dd in range(len(labels)):
                x = X[dd, 0]
                y = X[dd, 1]
                z = X[dd, 2]
                # print(dd,x,y, labels[dd])
                X_dict.setdefault(labels[dd], []).append((x, y, z))

            num_big_components = len(X_dict)
            # print('num_big_components', num_big_components)
            for key in range(num_big_components):
                # print('key',key)
                x= [t[0] for t in X_dict[key]]
                y = [t[1] for t in X_dict[key]]
                z = [t[2] for t in X_dict[key]]
                mean_array[key, 0] = np.mean(x)
                mean_array[key, 1] = np.mean(y)
                mean_array[key, 2] = np.mean(z)
                X_dict_mean.setdefault(key, []).append((np.mean(x), np.mean(y), np.mean(z)))
                X_dict_population.setdefault(key, []).append(counts[key])
        '''
        print('finished this apply sigma at', sigma_factor)
        return X_dict_mean, X_dict, labels

    def too_close(self, X_dict_mean, X_dict_label, labels, mean, cur_tooclosefactor):
        if self.metric == 'precomputed':  # for assigning the small clusters to the big cluster
            X = self.X_fit_original
            #print(X.shape, 'the shape of the codebook')
        else:
            #X = self.X_fit_
            X = self.X_fit_original
        updated = False
        #if self.metric == 'precomputed':
        #print('inside precompute for function too_close with factor', cur_tooclosefactor)
        copy1 = list(X_dict_mean.keys()).copy()
        shuffle(copy1)
        copy2 = list(X_dict_mean.keys()).copy()
        shuffle(copy2)
        for key1 in copy1:
            # print('key1 is ', key1)
            mindist = 999
            for key2 in copy2:
                coords1 = X_dict_mean[key1][0]
                coords2 = X_dict_mean[key2][0]
                coords = [coords1,coords2]
                if len(coords1)!=X.shape[1]: print('short coords', coords1)
                if len(coords2) != X.shape[1]: print('short coords', coords2)
                dist = distance.cdist(coords, coords, 'euclidean')[0, 1]
                if dist < mindist and dist != 0:
                    mindist = dist #find the closest cluster to Key1
                    nn_key = key2
            if (mindist > 0) and (mindist < cur_tooclosefactor * mean):  # 10
                # print('nn_key of ', key1, 'is ', nn_key)
                popkey1 = len(X_dict_label[key1])
                popkeynn = len(X_dict_label[nn_key])
                # print('pop of keynn', popkeynn)
                if popkey1 >= popkeynn: #merge to the larger cluster of the two clusters
                    idx = np.where(labels == nn_key)[0]
                    labels[idx] = key1
                    # print(X_dict_label[key1])
                    # print(X_dict_label[key2])
                    X_dict_label[key1].extend(list(X_dict_label[nn_key]))
                    # print(X_dict_label[key1])
                    X_dict_label.pop(nn_key, None)
                    mean_list= []
                    for i in range(X.shape[1]):
                        i_list = [t[i] for t in X_dict_label[key1]]
                        mean_list.append(np.mean(i_list))
                    X_dict_mean[key1] = [tuple(mean_list)]
                    X_dict_mean.pop(nn_key, None)
                    # print('key',key1, 'was too close to key', key2)
                    updated = True
                else:
                    idx = np.where(labels == key1)[0]
                    labels[idx] = nn_key
                    # print(X_dict_label[key1])
                    # print(X_dict_label[key2])
                    X_dict_label[nn_key].extend(list(X_dict_label[key1]))
                    # print(X_dict_label[key1])
                    X_dict_label.pop(key1, None)
                    mean_list = []
                    for i in range(X.shape[1]):
                        i_list = [t[i] for t in X_dict_label[nn_key]]
                        mean_list.append(np.mean(i_list))

                    X_dict_mean[nn_key] = [tuple(mean_list)]
                    X_dict_mean.pop(key1, None)
                    # print('key',key1, 'was too close to key', key2)
                    updated = True

                return X_dict_mean, X_dict_label, labels, updated
        updated = False
        '''
        else:
            print('inside 3D for function too_close')
            copy1 = list(X_dict_mean.keys()).copy()
            shuffle(copy1)
            copy2 = list(X_dict_mean.keys()).copy()
            shuffle(copy2)

            for key1 in copy1:
                # print('key1 is ', key1)
                mindist = 999
                for key2 in copy2:
                    coords1 = X_dict_mean[key1][0]
                    coords2 = X_dict_mean[key2][0]
                    coords = [coords1,coords2]
                    dist = distance.cdist(coords, coords, 'euclidean')[0, 1]
                    if dist < mindist and dist != 0:
                        mindist = dist #find the closest cluster to Key1
                        nn_key = key2

                # print('mindist is ', mindist, 'for key1 and key2', key1, nn_key)
                if (mindist > 0) and (mindist < cur_tooclosefactor * mean):  # 10
                    # print('nn_key of ', key1, 'is ', nn_key)
                    popkey1 = len(X_dict_label[key1])
                    popkeynn = len(X_dict_label[nn_key])
                    # print('pop of keynn', popkeynn)
                    if popkey1 >= popkeynn: #merge to the larger cluster of the two clusters
                        idx = np.where(labels == nn_key)[0]
                        labels[idx] = key1
                        # print(X_dict_label[key1])
                        # print(X_dict_label[key2])
                        X_dict_label[key1].extend(list(X_dict_label[nn_key]))
                        # print(X_dict_label[key1])
                        X_dict_label.pop(nn_key, None)
                        x = [t[0] for t in X_dict_label[key1]]
                        y = [t[1] for t in X_dict_label[key1]]
                        z = [t[2] for t in X_dict_label[key1]]
                        X_dict_mean[key1] = [(np.mean(x), np.mean(y), np.mean(z))]
                        X_dict_mean.pop(nn_key, None)
                        # print('key',key1, 'was too close to key', key2)
                        updated = True
                    else:
                        idx = np.where(labels == key1)[0]
                        labels[idx] = nn_key
                        # print(X_dict_label[key1])
                        # print(X_dict_label[key2])
                        X_dict_label[nn_key].extend(list(X_dict_label[key1]))
                        # print(X_dict_label[key1])
                        X_dict_label.pop(key1, None)
                        x = [t[0] for t in X_dict_label[nn_key]]
                        y = [t[1] for t in X_dict_label[nn_key]]
                        z = [t[2] for t in X_dict_label[nn_key]]
                        X_dict_mean[nn_key] = [(np.mean(x), np.mean(y), np.mean(z))]
                        X_dict_mean.pop(key1, None)
                        # print('key',key1, 'was too close to key', key2)
                        updated = True
                    return X_dict_mean, X_dict_label, labels, updated
            updated = False
            '''
        return X_dict_mean, X_dict_label, labels, updated

    def fit(self, X, y=None):
        """Fit the clustering model

        Parameters
        ----------
        X : array_like
            the data to be clustered: shape = [n_samples, n_features]
        """
        time_start = time.time()

        # if self.cutoff_scale is None:
        # raise ValueError("Must specify either cutoff or cutoff_frac")

        # Compute the distance-based graph G from the points in X
        if self.metric == 'precomputed':
            # Input is already a graph. Copy if sparse
            # so we can overwrite for efficiency below.
            self.X_fit_ = None
            G = validate_graph(X, directed=True,
                               csr_output=True, dense_output=False,
                               copy_if_sparse=True, null_value_in=np.inf)

            X_dict = {}
            for k in range(self.X_fit_original.shape[0]):
                ll = []
                for i in range(self.X_fit_original.shape[1]):
                    ll.append(self.X_fit_original[k,i])
                X_dict.setdefault(k, np.asarray(ll))
        elif not self.approximate:
            X = check_array(X)
            self.X_fit_ = X
            kwds = self.metric_params or {}
            G = pairwise_distances(X, metric=self.metric, **kwds)
            G = validate_graph(G, directed=True,
                               csr_output=True, dense_output=False,
                               copy_if_sparse=True, null_value_in=np.inf)
        else:
            # generate a sparse graph using n_neighbors of each point
            X = check_array(X)
            self.X_fit_ = X #changing neihbors to 20 from 30 on sep21

            n_neighbors = min(self.n_neighbors, X.shape[0] - 1)
            '''
            knn_struct = ls.make_knn_struct(X, ef=50)
            import copy
            X_data_copy = copy.deepcopy(X)
            neighbor_array, distance_array = knn_struct.knn_query(X_data_copy, k=30) #k is number of neighs was at 10

            G, dummy = ls.make_csrmatrix_noselfloop(neighbor_array, distance_array, dist_std=3, keep_all=True)
            sources, targets = G.nonzero()
            edgelist = list(zip(sources, targets))

            # print('EDGELIST', edgelist)
            edgelist_copy = edgelist.copy()
            # print('EDGELIST',edgelist)
            # print('making iGraph')
            G_ig = ig.Graph(edgelist)#, edge_attrs={'weight': G.data.tolist()})
            # print('edgelist_copy:', edgelist_copy) #[(0,20),(0,22)...]
            print('computing Jaccard metric')
            sim_list = G_ig.similarity_jaccard(pairs=edgelist_copy)
            # print('pre-lognormed mean and std', np.mean(sim_list), np.std(sim_list))
            #import matplotlib.pyplot as plt
            #from scipy import stats
            #plt.hist(sim_list)
            #plt.show()
            #sim_list = stats.zscore(stats.boxcox(sim_list,0))#log to get normal distribution
            # sim_list = sim_list +np.abs(np.min(sim_list))
            # plt.hist(sim_list)
            # plt.show()
            # sim_list = G.similarity_inverse_log_weighted(vertices=edgelist_copy)
            print('simlist jaccard weights mean and std:', np.mean(sim_list), np.std(sim_list))
            new_edgelist = []
            #G.data = 1-np.asarray(sim_list)
            #sim_list_array = np.asarray(sim_list)
            '''
            G = kneighbors_graph(X, n_neighbors=30,
                                mode='distance',
                                 metric=self.metric,
                                 metric_params=self.metric_params) #cannot compute Covariance matrix for large N

            #xn = X[:, 0]
            #yn = X[:, 1]
            #zn = X[:, 2]
            #xyzn = list(zip(xn, yn, zn))

            import matplotlib.pyplot as plt
            #a dictionary with coords of all the datapoints
            #X_dict = {}

            #for k in range(X.shape[0]):
            #    ll = []
            #    for i in range(self.X_fit_.shape[1]):
            #        ll.append(self.X_fit_[k, i])
            #    X_dict.setdefault(k, np.asarray(ll))
            '''    
            if X.shape[0] < 2000:
                sources, targets = G.nonzero()
                edgelist = zip(sources.tolist(), targets.tolist())
                Plotting_3D.save_anim_graph(edgelist, X, self.true_label,
                                            graph_title='5-NN Graph_' + str(len(self.true_label)))
            '''
            # fig_graph = plt.figure(figsize=(24, 24))
            # ax = fig_graph.add_subplot(1, 1, 1, projection='3d')
            # ax.set_title('5-NN Graph')
            # Plotting_3D.plot_graph_3D(fig_graph=fig_graph, ax=ax, edgelist=edgelist, X=X, true_label=self.true_label)
            '''
            Gnx = networkx.from_scipy_sparse_matrix(G,False, edge_attribute=None)
            pos = networkx.kamada_kawai_layout(Gnx,dim=3)
            print('pos', pos)
            print('dct', X_dict)

            cdict1 = {'blue': ((0.0, 0.0, 1.0),
                               (0.5, 0.1, 0.0),
                               (1.0, 0.0, 0.0)),
                      'green': ((0.0, 0.0, 0.0),
                                (1.0, 0.0, 0.0)),
                      'red': ((0.0, 0.0, 0.0),
                              (0.5, 0.0, 0.1),
                              (1.0, 1.0, 1.0))}
            from matplotlib.colors import LinearSegmentedColormap
            blue_red1 = LinearSegmentedColormap('BlueRed1', cdict1)
            clabels = np.concatenate((np.ones(300)*0.2,np.ones(400)*0.8)).tolist()
            networkx.draw(Gnx, pos, node_size = 8, node_color=clabels,cmap =blue_red1 , width=0.5,edge_color = 'gray')

            plt.show()
            '''
        # HACK to keep explicit zeros (minimum spanning tree removes them)
        zero_fillin = G.data[G.data > 0].min() * 1E-8
        G.data[G.data == 0] = zero_fillin

        # Compute the minimum spanning tree of this graph

        if self.metric =='precomputed': self.full_tree_ = G
        else: self.full_tree_ = minimum_spanning_tree(G, overwrite=True)
        '''
        if X.shape[0] < 2000:
            sources, targets = self.full_tree_.nonzero()
            edgelist = zip(sources.tolist(), targets.tolist())
            Plotting_3D.save_anim_graph(edgelist, X, self.true_label,
                                        graph_title='MST Graph_' + str(len(self.true_label)))
        '''
        # fig_graph = plt.figure(figsize=(24, 24))
        # ax = fig_graph.add_subplot(1, 1, 1, projection='3d')
        # Plotting_3D.plot_graph_3D(fig_graph=fig_graph,ax=ax,edgelist=edgelist,X=X, true_label=self.true_label)
        '''
        MSTnx = networkx.from_scipy_sparse_matrix(self.full_tree_, False)
        networkx.draw(MSTnx, X_dict, node_size = 8, node_color=clabels,cmap =blue_red1,width=0.5,edge_color = 'gray')
        plt.show()
        '''
        # undo the hack to bring back explicit zeros
        self.full_tree_[self.full_tree_ == zero_fillin] = 0
        max_labels = self.max_labels
        # Partition the data by the cutoff
        N = G.shape[0] - 1
        '''
        if N <= 250000:
            #max_labels=min(30,max_labels)
            max_labels=min(100,max_labels)
        if N<=100000:
            max_labels = min(100, max_labels)
            #max_labels = min(20,max_labels)
        if N<=50000:
            max_labels = min(15,max_labels)
        if N <= 10000:
            max_labels = min(10, max_labels)
        if N>250000:
            max_labels = min(50, max_labels)
        '''
        d = self.full_tree_.data
        '''

        import scipy.stats as ss
        import matplotlib.pyplot as plt
        skew_d = ss.skew(d)
        print('skew of MST is ',skew_d)
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
        ax1.hist(d, weights=np.zeros_like(d) + 1. / d.size)

        if abs(skew_d)> 0.8:
            if min(d)<=0: d = d+1+abs(min(d))
            unskew_d, lmd =  ss.boxcox(d)
            if min(unskew_d)<0: unskew_d = unskew_d+abs(min(unskew_d))
            self.full_tree_.data = unskew_d
            d=unskew_d
            fig2 = plt.figure()
            ax2 = fig2.add_subplot(111)
            ax2.hist(d, weights=np.zeros_like(d) + 1. / d.size)
        plt.show()
        '''
        # print('max-labels allowed is:', max_labels)
        mu = np.mean(d)
        sigma = np.std(d)
        self.mean_edge_distance = mu
        self.std_edge_distance = sigma
        print('mean edge length: ', mu, 'std edge length: ', sigma)

        if max_labels <  0:#200: #if not doing SOM then set to 0
            cur_sigma_factor = -2#-1.5 #-2.5
            print('cur sigma factor is', cur_sigma_factor)
            X_dict_mean, X_dict, labels = self.apply_sigma(cur_sigma_factor,500)
            print('len mean dict',len(X_dict_mean))
            cur_tooclosefactor = 0.5
            updated = True
            while len(set(labels)) < 1000 and cur_tooclosefactor <= 50: #on Sep13 changing to. was 30
                updated = True
                while updated == True:
                    X_dict_mean, X_dict, labels, updated = self.too_close(X_dict_mean, X_dict, labels, mu,
                                                                          cur_tooclosefactor=cur_tooclosefactor)
                    if len(set(labels)) <= max_labels:
                        dummy, labels = np.unique(labels, return_inverse=True)
                        clustering_runtime = time.time() - time_start
                        self.labels_ = labels  # final merged labels
                        self.clustering_runtime_ = clustering_runtime
                        self.tooclosefactor = cur_tooclosefactor
                        self.sigma_factor = cur_sigma_factor
                        print('updated num labels: ', len(set(labels)), 'at sigma factor ', cur_sigma_factor, 'and current too close', cur_tooclosefactor)
                        return self
                cur_tooclosefactor = cur_tooclosefactor +0.25#0.5 until we started testing Mosmann
        if max_labels  < 0:#200: #if not doing SOM then set to 0
            cur_sigma_factor = -1#-1.5
            print('cur sigma factor is', cur_sigma_factor)
            X_dict_mean, X_dict, labels = self.apply_sigma(cur_sigma_factor, 500)
            cur_tooclosefactor = 0.5
            updated = True
            while len(set(labels)) < 900 and cur_tooclosefactor <= 50: #on Sep13 changing to. was 30
                updated = True
                while updated == True:
                    X_dict_mean, X_dict, labels, updated = self.too_close(X_dict_mean, X_dict, labels, mu,
                                                                          cur_tooclosefactor=cur_tooclosefactor)
                    if len(set(labels)) <= max_labels:
                        dummy, labels = np.unique(labels, return_inverse=True)
                        clustering_runtime = time.time() - time_start
                        self.labels_ = labels  # final merged labels
                        self.clustering_runtime_ = clustering_runtime
                        self.tooclosefactor = cur_tooclosefactor
                        self.sigma_factor = cur_sigma_factor
                        print('updated num labels: ', len(set(labels)), 'at sigma factor ', cur_sigma_factor, 'and current too close', cur_tooclosefactor)
                        return self
                cur_tooclosefactor = cur_tooclosefactor +0.5
        if max_labels  <0:#200: #if not doing SOM then set to 0
            cur_sigma_factor = -0.75
            print('cur sigma factor is', cur_sigma_factor)
            X_dict_mean, X_dict, labels = self.apply_sigma(cur_sigma_factor,500)
            cur_tooclosefactor = 0
            updated = True
            while len(set(labels)) < 900 and cur_tooclosefactor <= 50: #on Sep13 changing to. was 30
                updated = True
                while updated == True:
                    X_dict_mean, X_dict, labels, updated = self.too_close(X_dict_mean, X_dict, labels, mu,
                                                                          cur_tooclosefactor=cur_tooclosefactor)
                    if len(set(labels)) <= max_labels:
                        dummy, labels = np.unique(labels, return_inverse=True)
                        clustering_runtime = time.time() - time_start
                        self.labels_ = labels  # final merged labels
                        self.clustering_runtime_ = clustering_runtime
                        self.tooclosefactor = cur_tooclosefactor
                        self.sigma_factor = cur_sigma_factor
                        print('updated num labels: ', len(set(labels)), 'at sigma factor ', cur_sigma_factor)
                        return self
                cur_tooclosefactor = cur_tooclosefactor + 0.5
        if max_labels  < 0:#200: #if not doing SOM then set to 0
            cur_sigma_factor = -0.5
            print('cur sigma factor is', cur_sigma_factor)
            X_dict_mean, X_dict, labels = self.apply_sigma(cur_sigma_factor,500)
            cur_tooclosefactor = 0
            updated = True
            while len(set(labels)) < 900 and cur_tooclosefactor <= 50: #on Sep13 changing to. was 30
                updated = True
                while updated == True:
                    X_dict_mean, X_dict, labels, updated = self.too_close(X_dict_mean, X_dict, labels, mu,
                                                                          cur_tooclosefactor=cur_tooclosefactor)
                    if len(set(labels)) <= max_labels:
                        dummy, labels = np.unique(labels, return_inverse=True)
                        clustering_runtime = time.time() - time_start
                        self.labels_ = labels  # final merged labels
                        self.clustering_runtime_ = clustering_runtime
                        self.tooclosefactor = cur_tooclosefactor
                        self.sigma_factor = cur_sigma_factor
                        print('updated num labels: ', len(set(labels)), 'at sigma factor ', cur_sigma_factor)
                        return self
                cur_tooclosefactor = cur_tooclosefactor + 0.5
        if max_labels  < 0:#200: #if not doing SOM then set to 0
            cur_sigma_factor = 0
            print('cur sigma factor is', cur_sigma_factor)
            X_dict_mean, X_dict, labels = self.apply_sigma(cur_sigma_factor,500)
            cur_tooclosefactor = 0
            updated = True
            while len(set(labels)) < 500 and cur_tooclosefactor <= 50: #on Sep13 changing to. was 30
                updated = True
                while updated == True:
                    X_dict_mean, X_dict, labels, updated = self.too_close(X_dict_mean, X_dict, labels, mu,
                                                                          cur_tooclosefactor=cur_tooclosefactor)
                    if len(set(labels)) <= max_labels:
                        dummy, labels = np.unique(labels, return_inverse=True)
                        clustering_runtime = time.time() - time_start
                        self.labels_ = labels  # final merged labels
                        self.clustering_runtime_ = clustering_runtime
                        self.tooclosefactor = cur_tooclosefactor
                        self.sigma_factor = cur_sigma_factor
                        print('updated num labels: ', len(set(labels)), 'at sigma factor ', cur_sigma_factor)
                        return self
                cur_tooclosefactor = cur_tooclosefactor + 0.5
        if max_labels < 10:#0:#15: #SOM 100
            cur_sigma_factor = 0.25
            print('cur sigma factor is', cur_sigma_factor)
            max_input_label = 500
            X_dict_mean, X_dict, labels = self.apply_sigma(cur_sigma_factor, max_input_label)
            cur_tooclosefactor = 0
            updated = True
            while len(set(labels)) < max_input_label and cur_tooclosefactor <= 50: #on Sep13 changing to. was 30
                updated = True
                while updated == True:
                    X_dict_mean, X_dict, labels, updated = self.too_close(X_dict_mean, X_dict, labels, mu,
                                                                          cur_tooclosefactor=cur_tooclosefactor)
                    if len(set(labels)) <= max_labels:
                        dummy, labels = np.unique(labels, return_inverse=True)
                        clustering_runtime = time.time() - time_start
                        self.labels_ = labels  # final merged labels
                        self.clustering_runtime_ = clustering_runtime
                        self.tooclosefactor = cur_tooclosefactor
                        self.sigma_factor = cur_sigma_factor
                        print('updated num labels: ', len(set(labels)), 'at sigma factor ', cur_sigma_factor)
                        return self
                cur_tooclosefactor = cur_tooclosefactor + 5
        if max_labels < 10:
            cur_sigma_factor =0.5
            print('cur sigma factor is', cur_sigma_factor)
            max_input_label = 1500
            X_dict_mean, X_dict, labels = self.apply_sigma(cur_sigma_factor, max_input_label)
            cur_tooclosefactor = 0
            updated = True
            while len(set(labels)) < max_input_label and cur_tooclosefactor <= 50:# was 30 until sep 13 ,was 500 until 21sept
                updated = True
                while updated == True:
                    X_dict_mean, X_dict, labels, updated = self.too_close(X_dict_mean, X_dict, labels, mu,
                                                                          cur_tooclosefactor=cur_tooclosefactor)
                    if len(set(labels)) <= max_labels:
                        dummy, labels = np.unique(labels, return_inverse=True)
                        clustering_runtime = time.time() - time_start
                        self.labels_ = labels  # final merged labels
                        self.clustering_runtime_ = clustering_runtime
                        self.tooclosefactor = cur_tooclosefactor
                        self.sigma_factor = cur_sigma_factor
                        print('updated num labels: ', len(set(labels)), 'at sigma factor ', cur_sigma_factor)
                        return self
                cur_tooclosefactor = cur_tooclosefactor + 5

        cur_sigma_factor =0.75 #added sept 22
        max_input_label = 1000
        X_dict_mean, X_dict, labels = self.apply_sigma(cur_sigma_factor, max_input_label)
        cur_tooclosefactor = 0
        updated = True

        while len(set(labels)) < max_input_label and cur_tooclosefactor <= 50:  # was 500 until 21sept
            updated = True
            while updated == True:
                print('merging at sigma',cur_sigma_factor)
                X_dict_mean, X_dict, labels, updated = self.too_close(X_dict_mean, X_dict, labels, mu,
                                                                      cur_tooclosefactor=cur_tooclosefactor)
                if len(set(labels)) <= max_labels:
                    dummy, labels = np.unique(labels, return_inverse=True)
                    clustering_runtime = time.time() - time_start
                    self.labels_ = labels  # final merged labels
                    self.clustering_runtime_ = clustering_runtime
                    self.tooclosefactor = cur_tooclosefactor
                    self.sigma_factor = cur_sigma_factor
                    print('updated num labels: ', len(set(labels)), 'at sigma factor ', cur_sigma_factor,
                          'and too_close factor of', cur_tooclosefactor)
                    return self
            cur_tooclosefactor = cur_tooclosefactor + 5# 0.2 for SOM  # 10

        cur_sigma_factor = 1.0
        max_input_label = 1000
        X_dict_mean, X_dict, labels = self.apply_sigma(cur_sigma_factor, max_input_label)

        cur_tooclosefactor = 0
        updated = True
        while len(set(labels)) < max_input_label and cur_tooclosefactor <= 50: #was 500 until 21sept
            updated = True
            while updated == True:
                X_dict_mean, X_dict, labels, updated = self.too_close(X_dict_mean, X_dict, labels, mu,
                                                                      cur_tooclosefactor=cur_tooclosefactor)
                if len(set(labels)) <= max_labels:
                    dummy, labels = np.unique(labels, return_inverse=True)
                    clustering_runtime = time.time() - time_start
                    self.labels_ = labels  # final merged labels
                    self.clustering_runtime_ = clustering_runtime
                    self.tooclosefactor = cur_tooclosefactor
                    self.sigma_factor = cur_sigma_factor
                    print('updated num labels: ', len(set(labels)), 'at sigma factor ', cur_sigma_factor,'and too_close factor of', cur_tooclosefactor)
                    return self
            cur_tooclosefactor = cur_tooclosefactor+5 #0.2(SOM 0.2) # 5#10
        cur_sigma_factor = 1.5
        max_input_label = 1000
        X_dict_mean, X_dict, labels = self.apply_sigma(cur_sigma_factor, max_input_label)

        cur_tooclosefactor = 0
        updated = True
        while len(set(labels)) < max_input_label and cur_tooclosefactor <= 50:  # was 500 until 21sept
            updated = True
            # print('updated tooclosefactor: ', cur_tooclosefactor)
            while updated == True:
                X_dict_mean, X_dict, labels, updated = self.too_close(X_dict_mean, X_dict, labels, mu,
                                                                      cur_tooclosefactor)

                if len(set(labels)) <= max_labels:
                    dummy, labels = np.unique(labels, return_inverse=True)
                    clustering_runtime = time.time() - time_start
                    self.labels_ = labels  # final merged labels
                    self.tooclosefactor = cur_tooclosefactor
                    self.clustering_runtime_ = clustering_runtime
                    self.sigma_factor = cur_sigma_factor
                    print('updated num labels: ', len(set(labels)), 'at sigma factor ', cur_sigma_factor, 'and too_close factor of', cur_tooclosefactor)
                    return self
            cur_tooclosefactor = cur_tooclosefactor + 5#10

        cur_sigma_factor = 2
        max_input_label = 500
        X_dict_mean, X_dict, labels = self.apply_sigma(cur_sigma_factor, max_input_label)

        cur_tooclosefactor = 0
        updated = True
        while len(set(labels)) < max_input_label and cur_tooclosefactor <= 50:  # was 500 until 21sept
            updated = True
            # print('updated tooclosefactor: ', cur_tooclosefactor)
            while updated == True:
                X_dict_mean, X_dict, labels, updated = self.too_close(X_dict_mean, X_dict, labels, mu,
                                                                      cur_tooclosefactor)

                if len(set(labels)) <= max_labels:
                    dummy, labels = np.unique(labels, return_inverse=True)
                    clustering_runtime = time.time() - time_start
                    self.labels_ = labels  # final merged labels
                    self.tooclosefactor = cur_tooclosefactor
                    self.clustering_runtime_ = clustering_runtime
                    self.sigma_factor = cur_sigma_factor
                    print('updated num labels: ', len(set(labels)), 'at sigma factor ', cur_sigma_factor,'and too_close factor of', cur_tooclosefactor)
                    return self
            cur_tooclosefactor = cur_tooclosefactor + 5# 10

        cur_sigma_factor = 2.5
        max_input_label = 700
        X_dict_mean, X_dict, labels = self.apply_sigma(cur_sigma_factor, max_input_label)

        cur_tooclosefactor = 0
        updated = True
        while len(set(labels)) < max_input_label and cur_tooclosefactor <= 50:  # was 500 until 21sept
            print('cur sigma', cur_sigma_factor,' and cur too close', cur_tooclosefactor)
            updated = True
            # print('updated tooclosefactor: ', cur_tooclosefactor)
            while updated == True:
                X_dict_mean, X_dict, labels, updated = self.too_close(X_dict_mean, X_dict, labels, mu,
                                                                      cur_tooclosefactor)
                if len(set(labels)) <= max_labels:
                    dummy, labels = np.unique(labels, return_inverse=True)
                    clustering_runtime = time.time() - time_start
                    self.labels_ = labels  # final merged labels
                    self.tooclosefactor = cur_tooclosefactor
                    self.clustering_runtime_ = clustering_runtime
                    self.sigma_factor = cur_sigma_factor
                    print('updated num labels: ', len(set(labels)), 'at sigma factor ', cur_sigma_factor,'and too_close factor of', cur_tooclosefactor)
                    return self
            cur_tooclosefactor = cur_tooclosefactor + 5#10

        cur_sigma_factor = 3.0
        max_input_label = 700
        X_dict_mean, X_dict, labels = self.apply_sigma(cur_sigma_factor, max_input_label)

        cur_tooclosefactor = 0
        updated = True
        while len(set(labels)) < max_input_label and cur_tooclosefactor <= 50:  # was 500 until 21sept
            updated = True
            # print('updated tooclosefactor: ', cur_tooclosefactor)
            while updated == True:
                X_dict_mean, X_dict, labels, updated = self.too_close(X_dict_mean, X_dict, labels, mu,
                                                                      cur_tooclosefactor)

                if len(set(labels)) <= max_labels:
                    dummy, labels = np.unique(labels, return_inverse=True)
                    clustering_runtime = time.time() - time_start
                    self.labels_ = labels  # final merged labels
                    self.tooclosefactor = cur_tooclosefactor
                    self.clustering_runtime_ = clustering_runtime
                    self.sigma_factor = cur_sigma_factor
                    print('updated num labels: ', len(set(labels)), 'at sigma factor ', cur_sigma_factor,'and too_close factor of', cur_tooclosefactor)
                    return self
            cur_tooclosefactor = cur_tooclosefactor + 10
        cur_sigma_factor = 3.5

        print('cur sigma factor is ', cur_sigma_factor)
        max_input_label = 700
        X_dict_mean, X_dict, labels = self.apply_sigma(cur_sigma_factor, max_input_label)
        cur_tooclosefactor = 0
        updated = True
        while len(set(labels)) < max_input_label and cur_tooclosefactor <= 50:  # was 500 until 21sept
            updated == True
            print('cur too close factor in sigma = 3.5', cur_tooclosefactor)
            while updated == True:
                X_dict_mean, X_dict, labels, updated = self.too_close(X_dict_mean, X_dict, labels, mu,
                                                                      cur_tooclosefactor=cur_tooclosefactor)

                if len(set(labels)) <= max_labels:
                    dummy, labels = np.unique(labels, return_inverse=True)
                    clustering_runtime = time.time() - time_start
                    self.labels_ = labels  # final merged labels
                    self.tooclosefactor = cur_tooclosefactor
                    self.clustering_runtime_ = clustering_runtime
                    self.sigma_factor = cur_sigma_factor
                    print('updated num labels: ', len(set(labels)), 'at sigma factor ', cur_sigma_factor)
                    return self
            cur_tooclosefactor = cur_tooclosefactor + 5

        cur_sigma_factor = 4.0
        print('cur sigma factor is ', cur_sigma_factor)
        max_input_label = 500
        X_dict_mean, X_dict, labels = self.apply_sigma(cur_sigma_factor, max_input_label)
        cur_tooclosefactor = 0
        updated = True
        while len(set(labels)) < max_input_label and cur_tooclosefactor <= 50:  # was 500 until 21sept
            updated = True
            while updated == True:
                X_dict_mean, X_dict, labels, updated = self.too_close(X_dict_mean, X_dict, labels, mu,
                                                                      cur_tooclosefactor=cur_tooclosefactor)
                if len(set(labels)) <= max_labels:
                    dummy, labels = np.unique(labels, return_inverse=True)
                    clustering_runtime = time.time() - time_start
                    self.labels_ = labels  # final merged labels
                    self.tooclosefactor = cur_tooclosefactor
                    self.clustering_runtime_ = clustering_runtime
                    self.sigma_factor = cur_sigma_factor
                    print('updated num labels: ', len(set(labels)), 'at sigma factor ', cur_sigma_factor)
                    return self
            cur_tooclosefactor = cur_tooclosefactor + 10
        cur_sigma_factor = 10
        print('cur sigma factor is ', cur_sigma_factor)
        max_input_label = 500
        X_dict_mean, X_dict, labels = self.apply_sigma(cur_sigma_factor, max_input_label)
        cur_tooclosefactor = 0
        updated = True
        while len(set(labels)) < max_input_label and cur_tooclosefactor <= 50:  # was 500 until 21sept
            updated = True
            while updated == True:
                X_dict_mean, X_dict, labels, updated = self.too_close(X_dict_mean, X_dict, labels, mu,
                                                                      cur_tooclosefactor=cur_tooclosefactor)
                if len(set(labels)) <= max_labels:
                    dummy, labels = np.unique(labels, return_inverse=True)
                    clustering_runtime = time.time() - time_start
                    self.labels_ = labels  # final merged labels
                    self.tooclosefactor = cur_tooclosefactor
                    self.clustering_runtime_ = clustering_runtime
                    self.sigma_factor = cur_sigma_factor
                    print('updated num labels: ', len(set(labels)), 'at sigma factor ', cur_sigma_factor)
                    return self
            print('number of clusters after too close factor', cur_tooclosefactor, len(set(labels)))
            cur_tooclosefactor = cur_tooclosefactor+5
        cur_sigma_factor = 10
        #while len(set(labels))> 700: #fails if number of connected components exceeds 700
            #cur_sigma_factor = cur_sigma_factor * 1.5
            #X_dict_mean, X_dict, labels = self.apply_sigma(cur_sigma_factor)
            #print('cur sigma factor is ', cur_sigma_factor)
        print('final sigma factor is ', cur_sigma_factor)
        cur_tooclosefactor = 10
        X_dict_mean, X_dict, labels = self.apply_sigma(cur_sigma_factor, max_input_label = 1000000)
        while len(set(labels)) > max_labels:
            updated = True
            while updated == True:
                X_dict_mean, X_dict, labels, updated = self.too_close(X_dict_mean, X_dict, labels, mu,
                                                                      cur_tooclosefactor=cur_tooclosefactor)
                if len(set(labels)) <= max_labels:
                    dummy, labels = np.unique(labels, return_inverse=True)
                    clustering_runtime = time.time() - time_start
                    self.labels_ = labels  # final merged labels
                    self.tooclosefactor = cur_tooclosefactor
                    self.clustering_runtime_ = clustering_runtime
                    self.sigma_factor = cur_sigma_factor
                    print('updated num labels: ', len(set(labels)), 'at sigma factor ', cur_sigma_factor)
                    return self
            print('number of clusters after too close factor', cur_tooclosefactor, len(set(labels)))
            cur_tooclosefactor = cur_tooclosefactor +5

        clustering_runtime = time.time() - time_start
        dummy, labels = np.unique(labels, return_inverse=True)
        self.labels_ = labels  # final merged labels
        self.tooclosefactor = cur_tooclosefactor
        self.clustering_runtime_ = clustering_runtime
        self.sigma_factor = cur_sigma_factor
        print('updated num labels: ', len(set(labels)), 'at sigma factor ', cur_sigma_factor)
        return self

        '''
        dummy, labels = np.unique(labels, return_inverse=True)
        print('number of merge groups', len(set(labels)))
        clustering_runtime = time.time() - time_start
        self.labels_ = labels #final merged labels
        #self.labels_nomerge_ =labels_knn_outlier #before merging
        #self.cluster_graph_ = cluster_graph
        self.clustering_runtime_ = clustering_runtime
        '''
        return self

    def get_graph_segments(self, full_graph=False):
        """Convenience routine to get graph segments. currently set for the final merged labels

        This is useful for visualization of the graph underlying the algorithm.

        Parameters
        ----------
        full_graph : bool (default: False)
            If True, return the full graph of connections. Otherwise return
            the truncated graph representing clusters.

        Returns
        -------
        segments : tuple of ndarrays
            the coordinates representing the graph. The tuple is of length
            n_features, and each array is of size (n_features, n_edges).
            For n_features=2, the graph can be visualized in matplotlib with,
            e.g. ``plt.plot(segments[0], segments[1], '-k')``
        """
        if not hasattr(self, 'X_fit_'):
            raise ValueError("Must call fit() before get_graph_segments()")
        if self.metric == 'precomputed':
            raise ValueError("Cannot use ``get_graph_segments`` "
                             "with precomputed metric.")

        n_samples, n_features = self.X_fit_.shape

        if full_graph:
            G = sparse.coo_matrix(self.full_tree_)
        else:
            G = sparse.coo_matrix(self.cluster_graph_)
        labels = self.labels_
        n_labels = len(set(labels))
        print('nlabels inplot', n_labels)
        X_mean = np.empty([n_features, 2])
        labels_mean = np.empty([n_labels, 2])
        keep_segment_row_x = []
        keep_segment_col_x = []
        keep_segment_row_y = []
        keep_segment_col_y = []
        xpairs = []
        ypairs = []
        xpairs_mean = []
        ypairs_mean = []

        for key in set(labels):
            idx = np.where(labels == key)[0].tolist()
            # print('idx',idx)
            x = self.X_fit_[idx, 0]
            y = self.X_fit_[idx, 1]
            # print('y',y)
            labels_mean[key, 0] = np.mean(x)
            labels_mean[key, 1] = np.mean(y)

        # print('mean labels', labels_mean)
        # for i in n_samples:
        # X_mean[i,0] = label_means[label[i],0]
        # X_mean[i,1] = label_means[label[i],1]

        shortlist = [i for i in range(len(G.row)) if labels[G.row[i]] != labels[G.col[i]]]
        print('shortlist', shortlist)
        for k in shortlist:
            group_pair = (labels[G.row[k]], labels[G.col[k]])
            # print('group pair', group_pair)
            xends = [self.X_fit_[G.row[k]][0], self.X_fit_[G.col[k]][0]]
            yends = [self.X_fit_[G.row[k]][1], self.X_fit_[G.col[k]][1]]
            xpairs.append(xends)
            ypairs.append(yends)
            xends_mean = [labels_mean[labels[G.row[k]]][0], labels_mean[labels[G.col[k]]][0]]
            # print('xends mean', xends_mean)
            yends_mean = [labels_mean[labels[G.row[k]]][1], labels_mean[labels[G.col[k]]][1]]
            # print('yends mean', yends_mean)
            xpairs_mean.append(xends_mean)
            ypairs_mean.append(yends_mean)
            # print('ypairs_mean',ypairs_mean)
        xlist = []
        ylist = []
        xlist_mean = []
        ylist_mean = []
        for xends, yends in zip(xpairs, ypairs):
            xlist.extend(xends)
            xlist.append(None)
            ylist.extend(yends)
            ylist.append(None)
        # print('xpairs', xpairs_mean)
        # print('ypairs', ypairs_mean)
        for xends_mean, yends_mean in zip(xpairs_mean, ypairs_mean):
            xlist_mean.extend(xends_mean)
            xlist_mean.append(None)
            ylist_mean.extend(yends_mean)
            ylist_mean.append(None)
        new = tuple((xlist, ylist))
        new_mean = tuple((xlist_mean, ylist_mean))
        original = tuple(np.vstack(arrs) for arrs in zip(self.X_fit_[G.row].T, self.X_fit_[G.col].T))
        # print('new mean', new_mean)
        return new_mean