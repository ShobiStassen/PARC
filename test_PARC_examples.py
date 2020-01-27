import PARC as parc

def main0():
    import pandas as pd
    from sklearn.datasets import make_blobs
    import matplotlib.pyplot as plt
    from sklearn import datasets
    #iris = datasets.load_iris()
    digits = datasets.load_digits()

    print('finished loading data')
    X = digits.data
    y=digits.target
    print(type(y))
    # features, target = make_blobs(n_samples = 5000,
    #               # two feature variables,
    #               n_features = 3,
    #               # four clusters,
    #               centers = 4,data
    #               # with .65 cluster standard deviation,
    #               cluster_std = 0.65,
    #               # shuffled,
    #               shuffle = True)
    #
    # # Create a scatterplot of first two features
    from MulticoreTSNE import MulticoreTSNE as multicore_tsne
    tsne = multicore_tsne(n_jobs=8, perplexity=30, verbose=1, n_iter=500, learning_rate=1000, angle=0.5)
    X_tsne = tsne.fit_transform(X)
    Parc1 = parc.PARC(X,y, jac_std_global='median')
    Parc1.run_PARC()

    print(X[0:10,0], X[0:10,1])
    plt.scatter(X_tsne[:,0],X_tsne[:,1], c = y, cmap='gist_rainbow')
    plt.imshow(digits.images[1], cmap=plt.cm.gray_r, interpolation='nearest')
    plt.show()
    plt.imshow(digits.images[20], cmap=plt.cm.gray_r, interpolation='nearest')
    plt.show()

    # View scatterplot
    plt.show()
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=Parc1.labels, cmap='gist_rainbow')
    plt.show()
def main():


    import Louvain_igraph_Jac24Sept as ls
    from sklearn.metrics.cluster import adjusted_rand_score
    import csv
    import numpy as np
    true_label = []
    with open("/home/shobi/Thesis/Rcode/my_first_R_project/GoldStandardDuo/Zheng8eqHVG10_truelabels.txt", 'rt') as f:
    #with open("/home/shobi/Thesis/Rcode/my_first_R_project/PBMC_68K/annotations_zhang.txt", 'rt') as f:
    #with open("/home/shobi/Thesis/Data/CyTOF/Samusik_allTrueLabel_Oct22.txt",'rt') as f:
        next(f)
        for line in f:
            line = line.strip().replace('\"', '')
            true_label.append(line)
    print(len(true_label), 'cell. Number of unique cell types',len(set(true_label) ))
    from collections import Counter
    print('true label stats')
    print(len(true_label), Counter(true_label))

    seurat_labels = []
    ##RUN PHENO/KMEANS

    '''
    data_matrix = csv.reader(
        open("/home/shobi/Thesis/Rcode/pca100Seed0_pbmc68k_oct21.txt", 'rt'),  delimiter=",")
    #data_matrix = csv.reader(    open("/home/shobi/Thesis/Rcode/my_first_R_project/GoldStandardDuo/pca100_mat_KumarAllHVG10.txt", 'rt'),  delimiter=",")
    #data_matrix  = csv.reader(open("/home/shobi/Thesis/Rcode/my_first_R_project/GeneMatrix_1000Var_oct24_notLognormed.txt", 'rt'),delimiter = ",")

    x = list(data_matrix)
    x.pop(0)  # [1:] #remove the header
    X_data = np.array(x).astype("float")
    print('shape of pca', X_data.shape)
    for k_clusters in [3,4,6,8,10,12,14,16,18,20,22]:
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=k_clusters, max_iter=150, random_state=100).fit(X_data)
        print("KMEANS COMPLETE")
        f1_acc_noweighting = 0
        targets = list(set(true_label))
        print('ari of kmeans is', adjusted_rand_score(np.asarray(true_label), kmeans.labels_))
        
        for onevsall_val in targets:
            vals_roc, predict_class_array, majority_truth_labels, numclusters_targetval = ls.accuracy_mst(kmeans.labels_,
                                                                                                          true_label,
                                                                                                          embedding_filename=None,
                                                                                                          clustering_algo='louvain',
                                                                                                          onevsall=onevsall_val)
            f1_current = vals_roc[1]
            print('f1 current for target', onevsall_val, 'is', f1_current)
            # f1_accumulated = f1_accumulated + f1_current * (list(true_label).count(onevsall_val)) / N
            f1_acc_noweighting = f1_acc_noweighting + f1_current
            f1_mean_method = f1_acc_noweighting / len(targets)
        print('final f1 mean other method', f1_mean_method, 'adjusted for pbmc', f1_mean_method * 11 / 10, 'n_clus is',
              len(set(kmeans.labels_)))
        
    # print("Phenograph start")
    # import backup_louvain_igraph_Jac24Sept as ls
    # predict_class_aggregate, df_accuracy, phenograph_labels,  onevsall_opt, majority_truth_labels, pheno_time, f1_acc_noweighting = ls.run_phenograph(X_data,
    #                                                                                                             true_label, knn=35)
    # print("Phenograph complete")
    #
    '''

    #with open("/home/shobi/Thesis/Data/CyTOF/Samusik_allTrueLabel_Oct22.txt", 'rt') as f:
    #with open("/home/shobi/Thesis/Rcode/my_first_R_project/GoldStandardDuo/sc3_68K_PBMC_Log2D_k35.txt",'rt') as f:
    with open("/home/shobi/Thesis/Rcode/my_first_R_project/GoldStandardDuo/Seurat_default_K25_Zheng8eqHVG10.txt",'rt') as f:
    #with open("/home/shobi/Thesis/Rcode/FlowPeaks/FlowPeaks_semiauto_labels_LC_10X_pbmc_pca_50.txt",'rt') as f:
        next(f)

        for line in f:
            line = line.strip().replace('\"', '')
            seurat_labels.append(line)
    print(len(seurat_labels), 'number of unique clusters', len(set(seurat_labels)))
    print('ari for Seurat',  adjusted_rand_score(np.asarray(true_label), seurat_labels))

    #seurat_labels = kmeans.labels_#phenograph_labels
    targets = list(set(true_label))
    f1_acc_noweighting = 0
    for onevsall_val in targets:
        vals_roc, predict_class_array, majority_truth_labels, numclusters_targetval = ls.accuracy_mst(seurat_labels,
                                                                                                      true_label,
                                                                                                      embedding_filename=None,
                                                                                                      clustering_algo='louvain',
                                                                                                      onevsall=onevsall_val)
        f1_current = vals_roc[1]
        print('f1 current for target', onevsall_val, 'is', f1_current)
        # f1_accumulated = f1_accumulated + f1_current * (list(true_label).count(onevsall_val)) / N
        f1_acc_noweighting = f1_acc_noweighting + f1_current
        f1_mean_method = f1_acc_noweighting / len(targets)
    print('final f1 mean other method', f1_mean_method, 'not adjusted for pbmc', f1_mean_method, 'n_clus is', len(set(seurat_labels)))
    import pandas as pd


    #data_matrix = csv.reader(open('/home/shobi/Thesis/Rcode/1000MostVarGenes_pbmc68k_lognorm.txt', 'rt'), delimiter=",")
    #data_matrix = csv.reader(open("/home/shobi/Thesis/Data/CyTOF/Runtimes/Umap100_Samusik_all.txt",'rt'), delimiter = ',')
    #data_matrix  = csv.reader(open("/home/shobi/Thesis/Rcode/my_first_R_project/GeneMatrix_1000Var_oct24_notLognormed.txt", 'rt'),delimiter = ",")
    data_matrix = csv.reader(open("/home/shobi/Thesis/Rcode/my_first_R_project/GeneMatrix_1000Var_oct24_notLognormed.txt", 'rt'),  delimiter=",")
    #data_matrix = csv.reader(open("/home/shobi/Thesis/Rcode/my_first_R_project/PBMC_68K/2500genesM3Drop_pbmc68k_nolognorm.txt", 'rt'),delimiter = ",")
    #data_matrix = csv.reader(open("/home/shobi/Thesis/Rcode/my_first_R_project/GoldStandardDuo/umap100_Zheng8eqExpr10.txt", 'rt'))
    #data_matrix = csv.reader(open('/home/shobi/Thesis/Rcode/my_first_R_project/GeneMatrix_1000Var_oct24_notLognormed.txt', 'rt'),delimiter = ",")
    #random_batch = np.random.choice(remaining_samples,  data.x.xhape[0] / configuration[“batch - factor”], replace = False)
    x =list(data_matrix)
    x.pop(0)  # [1:] #remove the header
    print('shape', np.array(x).shape)
    X_data = np.array(x).astype("float")
    # X_data=pd.DataFrame(X_data[:,0:10])
    # X_data.to_csv("/home/shobi/Thesis/Rcode/FLOCK/zheng8HVG10_pca100_flock.txt", header=True, index=False, sep='\t')
    # print("FLOCK INPUT SAVED", X_data.shape)
    from sklearn.decomposition import PCA
    # import time
    # t0_pca = time.time()
    # pca = PCA(n_components=50)
    # X_data = pca.fit_transform(X_data)
    # print('shape of pca',X_data.shape)
    # print('pca complete', time.time()-t0_pca, 'seconds elapsed')
    parc_method = 'leiden'
    knn_in = 30
    kdim =50#X_data.shape[1]
    RS=4
    too_big_factor = 30  # 30
    dist_std = 2
    small_pop =10
    jac_std = 0.15#'median'
    keep_all = True
    if keep_all == False:
        dis_std_str = 'KeepAll'
    else:
        dis_std_str = str(dist_std)
    weighted = True
    if weighted == True:
        weight_str = 'weighted'
    else:
        weight_str = 'unweighted'
    print('jac:', jac_std, 'dist:', dist_std, 'weighted:', weighted)
    print('using local pruning run times')
    for kdim in [1000]:#[60,100,160,200,260,300,360,410,460,510,560,610,660,710,760,810,860,910,960,1000]:# [10,15,20,25,30,40,60,80,100,120]:
        X_data_in=X_data[:, 0:kdim]
        #print('dims of input to PARC', X_data_in.shape)
        predict_class_aggregate, df_accuracy, parc_labels, knn_opt, onevsall_opt, majority_truth_labels_alph, time_end_knn_construct, time_end_prune, time_end_louvain, time_end_total, f1_accumulated, f1_mean, time_end_knn_query, num_edges = ls.run_mainlouvain(
            X_data=X_data_in, true_label=true_label,   too_big_factor=too_big_factor / 100, knn_in=knn_in, small_pop=small_pop, dist_std=dist_std, jac_std=jac_std,             keep_all_dist=keep_all, jac_weighted_edges=weighted, n_iter_leiden=5, parc_method=parc_method,
            partition_seed=RS)
        from CyTOF_tests_main import write_list_to_file
        #write_list_to_file(['label'] + parc_labels, "/home/shobi/Thesis/Data/CyTOF/Runtimes/Umap39_PARC_Samusik_all.txt")
        print('ari for PARC at kdim',kdim,  adjusted_rand_score(np.asarray(true_label), parc_labels), 'num groups', len(set(parc_labels)), 'f1-score',f1_mean, 'total cells', len(parc_labels))
        #print('ari for Seurat',  adjusted_rand_score(np.asarray(true_label), seurat_labels),'num groups', len(set(seurat_labels)), 'f1-score', f1_mean_method, 'total cells', len(seurat_labels))

if __name__ == '__main__':
    main()
