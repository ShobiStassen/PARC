import scipy.io
import numpy as np
import pandas as pd
import csv
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.sparse import csr_matrix, coo_matrix
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn import metrics
from MulticoreTSNE import MulticoreTSNE as multicore_tsne
from pandas import ExcelWriter
import Performance_phenograph as pp
import time
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
import LungCancer_function_minClusters_sep10 as LC
import plot_pbmc_mixture_10x as plot_10x
import matplotlib.pyplot as plt
import Louvain_igraph_Jac24Sept as ls
from scipy import stats
from mpl_toolkits.mplot3d import Axes3D
def write_list_to_file(input_list, csv_filename):
    """Write the list to csv file."""

    with open(csv_filename, "w") as outfile:
        for entries in input_list:
            outfile.write(str(entries))
            outfile.write("\n")

def make_majority_truth(mst_label_filename, true_labels):
    mst_labels = []
    majority_truth_labels = np.empty((len(true_labels),1), dtype=object)
    with open('/home/shobi/Thesis/Rcode/my_first_R_project/'+mst_label_filename+'.txt', 'rt') as f:
        for line in f:
            line = line.strip().replace('\"', '')
            mst_labels.append(line)
    for cluster_i in set(mst_labels):
        cluster_i_loc = np.where(np.asarray(mst_labels) == cluster_i)[0]
        population_cluster_i = len(cluster_i_loc)
        majority_truth = max(set(list(true_labels[cluster_i_loc])), key=list(true_labels[cluster_i_loc]).count)
        majority_truth_labels[cluster_i_loc] = majority_truth
    majority_truth_labels = list(majority_truth_labels.flatten())
    write_list_to_file(majority_truth_labels,'/home/shobi/Thesis/Rcode/my_first_R_project/'+mst_label_filename+'_majtruth.txt')

true_label = []
true_label_noTh2 = []
true_groups = ['CD14+ Monocyte', 'CD19+ B', 'CD34+', 'CD4+/CD25 T Reg' ,'CD4+/CD45RA+/CD25- Naive T','CD4+/CD45RO+ Memory','CD4+ T Helper2', 'CD56+ NK' ,'CD8+/CD45RA+ Naive Cytotoxic', 'CD8+ Cytotoxic T', 'Dendritic']


#with open('/home/shobi/Thesis/Rcode/my_first_R_project/DropClust/dropClust-master/true_cls_id.txt','rt') as f: #the DropClust annotations, which are quite different from Zhang's original ones
#with open('/home/shobi/Thesis/Rcode/my_first_R_project/cls_id.txt', 'rt') as f:

#with open('/home/shobi/Thesis/Rcode/LungCancerData_TrueLabel_N20200RandInt49_Jan.txt', 'rt') as f:
line_i = 0
discard_Th2_indices= []
tt_label =pd.read_csv('/home/shobi/Thesis/Rcode/my_first_R_project/PBMC_68K/annotations_zhang.txt', header = None)[0]
#tt_label = list(tt_label)
print('ttlabel', type(tt_label))
print('ttlabel', tt_label[0:10], tt_label[68578])
with open('/home/shobi/Thesis/Rcode/my_first_R_project/PBMC_68K/annotations_zhang.txt', 'rt') as f:
    next(f)#for the annotations of the PBMC mixture like in ZHANG. Not the pure bead labels
    for line in f:
        line = line.strip().replace('\"', '')
        true_label.append(line)

        if line != 'CD4+ T Helper2':
                true_label_noTh2.append(line)
        else: discard_Th2_indices.append(line_i)
        line_i=line_i+1

print(set(true_label),'set of truelabels')
print(set(true_label_noTh2),'set of truelabels no Th2')
write_list_to_file(true_label_noTh2,'/home/shobi/Thesis/Rcode/my_first_R_project/pbmc_noTh2_truelabels.txt')
alph_label = []

#with open("/home/shobi/Thesis/Rcode/my_first_R_project/PBMC_68K/alph_labels_68kpbmc_withTh2_dim50knn30_toobig30StdKeepAllsmallPop50_Jac0.15unweighted_May27_leiden_v2.txt", 'rt') as f: #slaph labels
with open("/home/shobi/Thesis/Rcode/my_first_R_project/PBMC_68K/Seurat_default_pbmc.txt", 'rt') as f:
    next(f)
    for line in f:
        line = line.strip().replace('\"', '')
        alph_label.append(int(line))
alph_label_np = np.asarray(alph_label)

print('shape of labels without TH2', alph_label_np.shape)
alph_label_noTh2 = np.delete(alph_label,discard_Th2_indices,axis=0)
print('shape of labels without TH2', alph_label_noTh2.shape)

print('ari for clustering method', adjusted_rand_score(np.asarray(true_label_noTh2), alph_label_noTh2))
print(len(alph_label), len(true_label))
targets = list(set(true_label_noTh2))
f1_acc_noweighting=0
for onevsall_val in targets:
    vals_roc, predict_class_array, majority_truth_labels, numclusters_targetval= ls.accuracy_mst(alph_label_noTh2, true_label_noTh2,
                                                             embedding_filename=None, clustering_algo='louvain', onevsall=onevsall_val)
    f1_current = vals_roc[1]
    print('f1 current for target', onevsall_val, 'is', f1_current)
    #f1_accumulated = f1_accumulated + f1_current * (list(true_label).count(onevsall_val)) / N
    f1_acc_noweighting = f1_acc_noweighting + f1_current
print('final f1 mean', f1_acc_noweighting/len(targets))

print(len(true_label), true_label[:10])
true_label = pd.Series(true_label)
from collections import Counter
print(Counter(true_label))

'''
make_majority_truth('APTauto_PBMC_toobig30.0Perp30sep21v7', true_label)
print('saved maj labels')

dropclust = []
with open('/home/shobi/Thesis/Rcode/my_first_R_project/DropClust/dropClust-master/predicted.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    next(csvreader)
    for row in csvreader:
        dropclust.append(int(row[0]))
print(len(dropclust), dropclust[0:10])
print('ari for DropClust: ARI', adjusted_rand_score(np.asarray(true_label), dropclust))
print("Adjusted Mutual Information: %0.5f"
      % metrics.adjusted_mutual_info_score(true_label, dropclust))
targets = list(set(true_label))
if len(targets) >=2: target_range = targets
else: target_range = [1]
N = len(true_label)
f1_accumulated =0
f1_acc_noweighting = 0
my_tab = pd.crosstab(index = pd.Series(dropclust),
                              columns=true_label)
print(my_tab)
#disable majority_truth_labels for DropClust if you want to get dropclust performance 
for onevsall_val in target_range:

    vals_roc, predict_class_array, majority_truth_labels_dropclust = ls.accuracy_mst(dropclust, true_label,
                                                             embedding_filename=None, clustering_algo='louvain', onevsall=onevsall_val)
    f1_current = vals_roc[1]
    f1_accumulated = f1_accumulated + f1_current * (list(true_label).count(onevsall_val)) / N
    f1_acc_noweighting = f1_acc_noweighting + f1_current
    print(f1_current)
print(f1_accumulated, ' f1 accumulated for DropClust ')
print(f1_acc_noweighting/len(target_range), ' f1 unweighted mean for DropClust ')

'''
'''
list_of_lists=[]
randint = 49
excel_file_name = '/home/shobi/Thesis/Rcode/my_first_R_project/PBMC_68K/10X_ALPHStatsjac0.15Oct11.xlsx'
for k_clusters in [10]:#,15,20,25,30,35,40,45,50]:#10,15,20,25,30,35,40,45,50]:
    flow_label = []
    with open('/home/shobi/Thesis/Rcode/my_first_R_project/alph_labels_68kpbmc_dim50knn30_toobig30Std3smallPop10_Jacp0.15Oct11.txt.txt', 'rt') as f:
        next(f)
        for line in f:
            line = line.strip().replace('\"', '')
            flow_label.append(line)
    ARI = adjusted_rand_score(np.asarray(true_label), flow_label )
    AMI = metrics.adjusted_mutual_info_score(true_label, flow_label)
    print('ari for kmeans with ',len(set(flow_label)),'groups', ARI)
    print("Adjusted Mutual Information:", AMI )
    targets = list(set(true_label))
    if len(targets) >=2: target_range = targets
    else: target_range = [1]
    N = len(true_label)
    f1_accumulated =0
    f1_mean = 0
    for onevsall_val in target_range:
        vals_roc, predict_class_array, maj, num_clus_target = ls.accuracy_mst(list(flow_label), true_label,
                                                                 embedding_filename=None, clustering_algo='louvain', onevsall=onevsall_val)
        f1_current = vals_roc[1]
        print(f1_current,'is f1 score for ',onevsall_val, 'target')
        f1_accumulated = f1_accumulated + f1_current * (list(true_label).count(onevsall_val)) / N
        f1_mean = f1_current + f1_mean
    list_of_lists.append([len(set(flow_label)), f1_accumulated,f1_mean/len(target_range), ARI, AMI])
    print('stats', list_of_lists)
writer = ExcelWriter(excel_file_name)
df = pd.DataFrame(list_of_lists, columns=['Num Clusters',  'f1-acc (weighted by pop)','f1-mean','ARI','AMI'])
df.to_excel(writer, 'Stats', index=False)
writer.save()

'''
#
# print("Homogeneity: %0.3f" % metrics.homogeneity_score(true_label, flow_label))
# print("Completeness: %0.3f" % metrics.completeness_score(true_label, flow_label))
# print("V-measure: %0.3f" % metrics.v_measure_score(true_label, flow_label))
# print("Adjusted Rand Index: %0.3f"
#       % metrics.adjusted_rand_score(true_label, flow_label))
# print("Adjusted Mutual Information: %0.3f"
#       % metrics.adjusted_mutual_info_score(true_label, flow_label))
#print("Silhouette Coefficient: %0.3f"
#      % metrics.silhouette_score(X_data, flow_label))

#m1 = scipy.io.mmread('/home/shobi/test.txt')

#m1_sparse=csr_matrix(m1)
#X_data = m1.toarray()
#robjects.r['load']("/home/shobi/pca_mat.RData")

#READING IN THE DATA MATRIX

print('reading data')
#reader = csv.reader(open("/home/shobi/Thesis/Rcode/pca50_pbmc68k.txt", 'rt'),delimiter = ",") #pca 50 dims x68K for the mixture of PBMC
'''
can use this instead of csv.reader() method. noted Aug23 2019
rr_reader = pd.read_csv("/home/shobi/Thesis/Rcode/pca50_pbmc68k.txt")
rr_reader = rr_reader.values
rr_reader = rr_reader.astype('float')
print('rr type', type(rr_reader))
'''
##reader = csv.reader(open("/home/shobi/test_pure.txt", 'rt'),delimiter = ",") #datamatrix of the pure bead PBMC 94655*50 MATRIX
##reader = csv.reader(open("/home/shobi/Thesis/Rcode/pca1000Seed0_pbmc68k.txt", 'rt'),delimiter = ",")#pca 50 dims x68K for the mixture of PBMC
reader = csv.reader(open("/home/shobi/Thesis/Rcode/Transformed1000GenesSeed0_pbmc68k.txt", 'rt'),delimiter = ",")
#reader = csv.reader(open('/home/shobi/Thesis/Rcode/GeneMatrix_1000Var_oct24.txt', 'rt'),delimiter = ",")



x = list(reader)

#print(x[0], 'header?')
x.pop(0)#[1:] #remove the header for pca1000. this is not needed for pca50 file
X_data = np.array(x).astype("float")
print('new pca oct21', X_data.shape)
#plot clusters with count above 150 cells
counts = np.bincount(alph_label)
labels = np.asarray(alph_label)
print('labels',labels)
print('label',np.sum(labels ==0))
to_remove = np.where(counts < 151)[0]  # which label values to remove
print('to remove', to_remove)

if len(to_remove) > 0:
    for i in to_remove:
        print(np.sum(labels==str(i)))
        labels[labels == str(i)] = -1
        print('changing labels')
    dummy, labels = np.unique(labels, return_inverse=True)
    labels -= 1  # keep -1 labels the same
idx_to_keep = np.where(labels != -1)[0]
print(len(idx_to_keep), 'labels to keep')

true_labels_tokeep = np.asarray(true_label)[idx_to_keep].tolist()
labels_tokeep = np.asarray(labels)[idx_to_keep].tolist()

'''
from FItSNE import fast_tsne

X_plot= fast_tsne.fast_tsne(X_data[:,0:25], learning_rate=10000, perplexity=20)

figtest = plt.figure(figsize=(36, 12))
ax1 = figtest.add_subplot(1, 2, 1)#, projection='3d')
ax2 = figtest.add_subplot(1, 2, 2)#, projection='3d')

ax1= plot_10x.plot_onemethod_2D(ax1, X_plot[idx_to_keep,:], true_labels_tokeep, true_labels_tokeep, method='ALPH', onevsall = 'CD56+ NK',GroundTruth = True)
ax2= plot_10x.plot_onemethod_2D(ax2, X_plot[idx_to_keep,:], labels_tokeep, true_labels_tokeep, method='ALPH', onevsall = 'CD56+ NK')
plt.show()
'''

X_data_noTh2 = np.delete(X_data,discard_Th2_indices,axis=0)

# print('dim of X_data_noTh2', X_data_noTh2.shape)
import copy
'''
#saving data in FLOCK input format 
X_copy = copy.deepcopy(X_data)
print(X_data.shape)
header = "PC1   "
for i in range(2,51):
    if i !=50: header = header+"PC"+str(i)+'\t'
    else: header = header +"PC50"
print(header)
np.savetxt("/home/shobi/Thesis/Rcode/FLOCK/data_10X_PBMC_N"+str(X_data.shape[0])+"Feb26.txt",X_data,delimiter='\t', header  = header, fmt='%f' ) #compatible with FLOCK which needs tab separated columns and header
print('saved data 10x')
'''
#true_label_pure=list(np.ones(9232))+list(np.ones(8385)*2)+list(np.ones(10479)*3)+list(np.ones(10263)*4)+list(np.ones(11953)*5)+list(np.ones(10224)*6)+list(np.ones(10209)*7)+list(np.ones(10085)*8)+list(np.ones(11213)*9)+list(np.ones(2612)*10)
#print('number of pure bead samples', len(true_label_pure)) #LIST OF TRUE LABELS FOR THE PURE POPULATIONS if trying to cluster the pure populations
#pca = PCA(n_components=100)
#X_data = pca.fit_transform(X_data)
'''


list_of_lists=[]
excel_file_name = '/home/shobi/Thesis/Rcode/Kmeans/10X_PBMC_kmeans_noTh2_stats.xlsx'
for k_clusters in [10,15,20,25,30,35,40,45,50,55,60,65,70]:#,25,30,35,40,45,50,55,60,65,70]:

    kmeans = KMeans(n_clusters=k_clusters, max_iter=150,random_state=100).fit(X_data_noTh2)
    #kmeans.labels_
    ARI = adjusted_rand_score(np.asarray(true_label_noTh2), kmeans.labels_ )
    AMI = metrics.adjusted_mutual_info_score(true_label_noTh2, kmeans.labels_)
    print('ari for kmeans with ',k_clusters,'groups', ARI)
    print("Adjusted Mutual Information:", AMI )
    targets = list(set(true_label_noTh2))
    if len(targets) >=2: target_range = targets
    else: target_range = [1]
    N = len(true_label_noTh2)
    f1_accumulated =0
    f1_mean = 0
    for onevsall_val in target_range:
        vals_roc, predict_class_array, maj, num_onevsall_vall_clusters  = ls.accuracy_mst(list(kmeans.labels_), true_label_noTh2,
                                                                 embedding_filename=None, clustering_algo='louvain', onevsall=onevsall_val)
        f1_current = vals_roc[1]
        f1_accumulated = f1_accumulated + f1_current * (list(true_label_noTh2).count(onevsall_val)) / N
        f1_mean = f1_current + f1_mean
    list_of_lists.append([k_clusters, f1_accumulated,f1_mean/len(target_range), ARI, AMI])
    print('stats', list_of_lists)
writer = ExcelWriter(excel_file_name)
df = pd.DataFrame(list_of_lists, columns=['Num Clusters',  'f1-acc (weighted by pop)','f1-mean','ARI','AMI'])
df.to_excel(writer, 'Kmeans Stats', index=False)
writer.save()
'''


import random
print('call alph')

parc_method = 'leiden'
knn_in =30
kdim=1000
too_big_factor = 30 #30
dist_std = 2
small_pop =  10
jac_std= 0.15#'median'
print('jac:')
keep_all =True
if keep_all == True: dis_std_str = 'KeepAll'
else: dis_std_str = str(dist_std)
weighted =True
if weighted == True: weight_str = 'weighted'
else: weight_str = 'unweighted'
print('jac:' ,jac_std, 'dist:', dist_std,'weighted:', weighted)
list_len = []
#excel_file_name = '/home/shobi/Thesis/Rcode/ALPH_dimensions_runtime_nov26_jac0.55_toobig90_knn30.xlsx'
DIR_str='/home/shobi/Thesis/Rcode/my_first_R_project/PBMC_68K/'

seed_list =[1,2,3]#,5,6,7,8,9]#[5,10,15,20,25]+list(np.arange(50,1025,25))
jac_std_list=[30]
X_copy = X_data #X_data_noTh
n_elements = X_data.shape[0]
#subsample
print('percentage is 5')
idx_rand= random.sample(range(n_elements), int(n_elements*0.05))
#true_label = np.asarray(true_label)[idx_rand]
#true_label = true_label.tolist()
print('true_label stats', len(true_label), true_label)
X_data = X_copy[:, 0:kdim]
#X_data = X_data[idx_rand, :]
for RS in seed_list:
    for knn_in in jac_std_list:
        filename = 'PARC_68kpbmc_withTh2_dim' + str(kdim) + 'knn' + str(knn_in) + '_toobig' + str(
            too_big_factor) + 'Std' + dis_std_str + 'smallPop' + str(small_pop) + '_Jac' + str(
            jac_std) + weight_str + 'Keepall' + str(keep_all) + parc_method +'knn'+str(knn_in)+ '_oct25'
        excel_file_name = DIR_str + filename + '.xlsx'
        time_start_alph = time.time()
        #X_data_noTh2 = X_copy[:,0:kdim]

        kdim_check = X_data#_noTh2.shape[1]
        print("dim of subsampled X_data", X_data.shape)
        #if kdim != X_data.shape[1]: print('dimensions do not match')
        #random.seed(123)
        print('randomseed set to', RS)
        predict_class_aggregate, df_accuracy, alph_labels,knn_opt, onevsall_opt, majority_truth_labels_alph,time_end_knn_construct, time_end_prune, time_end_louvain, time_end_total, f1_accumulated, f1_mean, time_end_knn_query, num_edges = ls.run_mainlouvain(X_data, true_label,
                                                                                        too_big_factor = too_big_factor/100, knn_in = knn_in,small_pop = small_pop, dist_std=dist_std, jac_std=jac_std, keep_all_dist=keep_all, jac_weighted_edges=weighted,n_iter_leiden=5, parc_method=parc_method, partition_seed=RS)



        print('adjusted mean for knn', knn_in, 'is', f1_mean*11/10)#filename = 'pheno_labels_10X_PBMC_68K_Feb26_k'+str(pheno_knn)
        #print("ARI %0.5f and AMI %0.5f",  adjusted_rand_score(np.asarray(true_label), np.asarray(alph_labels)),metrics.adjusted_mutual_info_score(np.asarray(true_label), np.asarray(alph_labels)))
        write_list_to_file(alph_labels, DIR_str+filename + '.txt')
        #write_list_to_file(pheno_labels, '/home/shobi/Thesis/Rcode/my_first_R_project/'+filename+'.txt')
        write_list_to_file(majority_truth_labels_alph, '/home/shobi/Thesis/Rcode/my_first_R_project/alph_majoritytruth_'+filename+'.txt')
        print(filename)
        pheno_knn = 30
        #print('start phenograph')
        #predict_class_aggregate, df_accuracy_pheno, pheno_labels, onevsall_opt, majority_truth_labels_pheno, pheno_time, f1_acc_noweighting_pheno = ls.run_phenograph(
        #    X_data, true_label, knn=pheno_knn)
        #runtime = time.time() - time_start_alph
        # filename =  'alph_labels_68kpbmc_dim'+str(kdim)+'knn'+str(knn_in)+f '_toobig'+str(too_big_factor)+'Std'+str(dist_std)+'smallPop'+str(small_pop)+'_Jacp'+str(jac_std)+'May7_leiden_weightedMedianv3.txt'
        print('filename of alph labels', filename, 'took', time_end_total,'seconds')
        #print('num clusters', len(set(alph_labels)))
        list_len.append([kdim_check,len(set(alph_labels)),time_end_knn_construct+time_end_knn_query, time_end_prune, time_end_louvain, time_end_total,f1_mean,f1_accumulated, num_edges])

        print('ari for ALPH with', len(set(alph_labels)), 'groups', adjusted_rand_score(np.asarray(true_label), alph_labels))
        print('Knn is', knn_in, 'and f1-mean is', f1_mean)
        print("Adjusted Mutual Information: %0.5f"% metrics.adjusted_mutual_info_score(true_label, alph_labels))

        #list_len.append([kdim, len(set(pheno_labels)), pheno_time, f1_acc_noweighting_pheno, ])
        #print(list_len)
        writer = ExcelWriter(excel_file_name)
        #df_accuracy_pheno.to_excel(writer, 'pheno', index=False)
        df = pd.DataFrame(list_len, columns=['Dimensions', 'Num Clusters', 'KNN time','prune time','louvain time','total time','f1-mean', 'f1-acc (weighted by pop)', 'num edges'])
        #df = pd.DataFrame(list_len,                      columns=['Dimensions', 'Num Clusters', 'total time', 'f1-mean'])
        df.to_excel(writer, 'runtimes', index=False)
        writer.save()

#end call alph

#
# print('finished', list_len)
#
# #my_tab = pd.crosstab(index = pd.Series(alph_labels),                              columns=true_label)
# #print(my_tab)
# print("finished ALPH")
# #predict_class_aggregate, df_accuracy, phenograph_labels, onevsall_opt,majority_truth_labels_pheno =ls.run_phenograph(X_data,true_label)
# #print("ARI %0.5f and AMI %0.5f",  adjusted_rand_score(np.asarray(true_label), np.asarray(phenograph_labels)),metrics.adjusted_mutual_info_score(true_label, np.asarray(phenograph_labels)))
# labels = []
# #good apt run: APTauto_PBMC_toobig30.0Perp30sep21v7.txt'
#
labels = []
with open('/home/shobi/Thesis/Rcode/my_first_R_project/PBMC_68K/alph_labels_68kpbmc_withTh2_dim50knn30_toobig30StdKeepAllsmallPop50_Jac0.15unweighted_May27_leiden_v2.txt','rt') as f:
#with open('/home/shobi/Thesis/Rcode/my_first_R_project/PBMC_68K/Seurat_oct21.txt','rt') as f:
    # next(f)
    for line in f:
        line = line.strip().replace('\"', '')
        labels.append(int(line))

labels = np.asarray(labels)
print(labels.shape)
targets = list(set(true_label))
if len(targets) >=2: target_range = targets
else: target_range = [1]
N = len(true_label)
f1_accumulated =0
f1_mean = 0
counts = np.bincount(labels)
to_remove = np.where(counts < 151)[0]  # which label values to remove
print(to_remove, type(labels))
if len(to_remove) > 0:
    for i in to_remove:
        print(np.sum(labels==i))
        labels[labels == i] = -1
        print('changing labels')
    dummy, labels = np.unique(labels, return_inverse=True)
    labels -= 1  # keep -1 labels the same
idx_to_keep = np.where(labels != -1)[0]
print(len(idx_to_keep), 'labels to keep')

true_labels_tokeep = np.asarray(true_label)[idx_to_keep].tolist()
labels_tokeep = np.asarray(labels)[idx_to_keep].tolist()
print('number of unique clus', len(set(labels_tokeep)))
print(len(labels_tokeep), 'len of labels to keep and true labels to keep')
print(len(true_labels_tokeep))
#
# list_of_lists = []
# excel_file_name = '/home/shobi/Thesis/Rcode/my_first_R_project/PBMC_68K/APT_stats_minpop20v2.xlsx'
# num_clus = []
# dims = [50]#[5,10,20,30,40,60,70,80,90,125,175,1000]#list(np.arange(0,1050,50))
# print('dims are', dims)
# for dim_i in dims:#,15,25,50,75,100,125,150,175,200,225,250,300,350,400,450,500]:
#     time_start = time.time()
#     #sep 13 changed perp from 50 to 30
#     perp = 30
#     X_data_use = X_data[:,0:dim_i]
#     print('shape of input data', X_data_use.shape)
#     X_LV_embedded, lv_embedding_filename, lv_runtime, lv_embedding_plot_title = pp.run_lv(version=None,
#                                                                                           input_data=X_data_use,
#                                                                                           perplexity=perp,
#                                                                                           lr=1,
#                                                                                           new_file_name='/home/shobi/lv_10xpbmc_april',
#                                                                                        new_folder_name=None, outdim=3)
# '''
#
print('dims of X_data to embed', X_data[:,0:50].shape)
tsne = multicore_tsne(n_jobs=8, perplexity=30, verbose=1, n_iter=1000, learning_rate=1000, angle=0.5)

X_LV_embedded = tsne.fit_transform(X_data[:,0:50])


X_LV_embedded = stats.zscore(X_LV_embedded, axis=0)

X_plot = X_LV_embedded[idx_to_keep,:]



figtest = plt.figure(figsize=(36, 12))
ax1 = figtest.add_subplot(1, 1, 1)
#ax2 = figtest.add_subplot(1, 2, 2, projection='3d')
ax1= plot_10x.plot_onemethod_2D(ax1, X_plot, true_labels_tokeep, true_labels_tokeep, method='ALPH', onevsall = 'CD56+ NK',GroundTruth = True)
plt.show()

figtest2 = plt.figure(figsize=(36, 12))
ax2 = figtest2.add_subplot(1, 1, 1)
ax2= plot_10x.plot_onemethod_2D(ax2, X_plot, labels_tokeep, true_labels_tokeep, method='ALPH', onevsall = 'CD56+ NK',GroundTruth = True)
plt.show()
#ax1= plot_10x.plot_onemethod(ax1, X_plot, true_labels_tokeep, true_labels_tokeep, method='ALPH', onevsall = 'CD56+ NK',GroundTruth = True)
#ax2= plot_10x.plot_onemethod(ax2, X_plot, labels_tokeep, true_labels_tokeep, method='ALPH', onevsall = 'CD56+ NK')

#
# #df_accuracy_mst_lv, best_labels_mst_lv, sigma_opt_lv, min_clustersize_mst_lv, tooclose_factor_opt, onevsall_opt_mst_lv, majority_truth_labels_mst_lv = LC.multiclass_mst_accuracy(
#     #X_LV_embedded, true_label, df_all=None,av_peaks=0)
#
# #print('ARI APT for av_peaks=auto', adjusted_rand_score(np.asarray(true_label), np.asarray(best_labels_mst_lv)), metrics.adjusted_mutual_info_score(true_label, best_labels_mst_lv))
# #write_list_to_file(best_labels_mst_lv, '/home/shobi/Thesis/Rcode/my_first_R_project/APT_labels_sep8')
# peak_threshhold_i = -1
# for too_big_factor_i in [0.3,0.2]:#,0.2]:
#     for av_peaks_i in [0,10,15,20,25,30,35,40,45,50]:#[-2,-1,0,0.5,1,1.5,2]: #-1 means keep all peaks, -2 means add 10 clusters for leway
#         start_cluster_time = time.time()
#         print("TOO BIG FACTOR IS, ", too_big_factor_i)
#
#
#         df_accuracy_mst_lv, best_labels_mst_lv, sigma_opt_lv, min_clustersize_mst_lv, tooclose_factor_opt, onevsall_opt_mst_lv, majority_truth_labels_mst_lv, f1_accumulated, f1_mean = LC.multiclass_mst_accuracy(
#             X_LV_embedded, true_label, df_all=None,av_peaks=av_peaks_i, too_big_factor=too_big_factor_i, verbose_print = True,peak_threshhold=peak_threshhold_i,inputGraph=False,min_clustersize=20)
#         cluster_time = time.time()-start_cluster_time
#         ARI = adjusted_rand_score(np.asarray(true_label), np.asarray(best_labels_mst_lv))
#         AMI =metrics.adjusted_mutual_info_score(true_label, best_labels_mst_lv)
#         print('ARI APT for av_peaks=auto, tooBig ',too_big_factor_i, ARI, AMI)
#         write_list_to_file(best_labels_mst_lv, '/home/shobi/Thesis/Rcode/my_first_R_project/APTauto_PBMC_'+'toobig'+str(too_big_factor_i*100)+'Perp'+str(perp)+'Dimensions'+str(dim_i)+'peakstd'+str(peak_threshhold_i)+'_Oct22.txt')
#         write_list_to_file(majority_truth_labels_mst_lv,'/home/shobi/Thesis/Rcode/my_first_R_project/APTauto_majtruth_PBMC_'+'toobig'+str(too_big_factor_i*100)+'Perp'+str(perp)+'Dimensions'+str(dim_i)+'peakstd'+str(peak_threshhold_i)+'_Oct22.txt')
#         '''
#         #fig, ax = plt.subplots(1, 2, figsize=(36, 12), sharex=True, sharey=True)
#         figtest = plt.figure(figsize=(36, 12))
#         ax1 = figtest.add_subplot(1, 2, 1, projection='3d')
#         ax2 = figtest.add_subplot(1, 2, 2, projection='3d')
#         ax1 = plot_10x.plot_onemethod(ax1, X_LV_embedded, true_label, true_label, method='APT',
#                                         onevsall='CD4+/CD45RO+ Memory')
#         ax2 = plot_10x.plot_onemethod(ax2, X_LV_embedded, best_labels_mst_lv, true_label, method='APT',
#                                         onevsall='CD4+/CD45RO+ Memory')
#         plt.savefig('/home/shobi/Thesis/10x_visuals/APT Scalability/APTauto_PBMC_'+'toobig'+str(too_big_factor_i*100)+'Perp'+str(perp)+'Dimensions'+str(dim_i)+'_Oct22.png')#plt.show()
#         '''
#         #list_of_lists.append([dim_i, peak_threshhold_i,lv_runtime, cluster_time, len(set(best_labels_mst_lv)),f1_mean, f1_accumulated])
#         #print('num clus list', num_clus)
#         list_of_lists.append([av_peaks_i,len(set(best_labels_mst_lv)), f1_accumulated, f1_mean, ARI, AMI])
#         print('stats', list_of_lists)
#         writer = ExcelWriter(excel_file_name)
#         df = pd.DataFrame(list_of_lists, columns=['user defined number clusters','actual num clusters', 'f1-acc (weighted by pop)', 'f1-mean', 'ARI', 'AMI'])
#         df.to_excel(writer, 'runtimes', index=False)
#         writer.save()
#     #writer = ExcelWriter(excel_file_name)
#     #df = pd.DataFrame(list_of_lists, columns=['dims','peak threshhold std factor', 'LV Time', 'Cluster Time', 'num clusters','mean F1-score', 'accumulated F1-score'])
#     #df.to_excel(writer,'runtimes',index=False)
#     #writer.save()
#
#
# '''
# df_accuracy_mst_lv, best_labels_mst_lv, sigma_opt_lv, min_clustersize_mst_lv, tooclose_factor_opt, onevsall_opt_mst_lv, majority_truth_labels_mst_lv = LC.multiclass_mst_accuracy(
#     X_LV_embedded, true_label, df_all=None, av_peaks=50, too_big_factor=too_big_factor_i, verbose_print=True,
#     X_data_ndim=X_data)
# print('ARI APT for av_peaks=50, tooBig ', too_big_factor_i,
#       adjusted_rand_score(np.asarray(true_label), np.asarray(best_labels_mst_lv)),
#       metrics.adjusted_mutual_info_score(true_label, best_labels_mst_lv))
# print('maj labels', majority_truth_labels_mst_lv[:10])
# write_list_to_file(best_labels_mst_lv, '/home/shobi/Thesis/Rcode/my_first_R_project/APT50_PBMC_' + 'toobig' + str(
#     too_big_factor_i * 100) + 'Perp' + str(perp) + 'sep21v8.txt')
# write_list_to_file(majority_truth_labels_mst_lv,
#                    '/home/shobi/Thesis/Rcode/my_first_R_project/APT50_majtruth_PBMC_' + 'toobig' + str(
#                        too_big_factor_i * 100) + 'Perp' + str(perp) + 'sep21v8.txt')
# figtest = plt.figure(figsize=(36, 12))
# ax1 = figtest.add_subplot(1, 2, 1, projection='3d')
# ax2 = figtest.add_subplot(1, 2, 2, projection='3d')
# ax1 = plot_10x.plot_onemethod(ax1, X_LV_embedded, true_label, true_label, method='APT',
#                               onevsall='CD4+/CD45RO+ Memory')
# ax2 = plot_10x.plot_onemethod(ax2, X_LV_embedded, best_labels_mst_lv, true_label, method='APT',
#                               onevsall='CD4+/CD45RO+ Memory')
# plt.show()
#
# df_accuracy_mst_lv, best_labels_mst_lv, sigma_opt_lv, min_clustersize_mst_lv, tooclose_factor_opt, onevsall_opt_mst_lv, majority_truth_labels_mst_lv = LC.multiclass_mst_accuracy(
#     X_LV_embedded, true_label, df_all=None,av_peaks=15,too_big_factor=too_big_factor_i,verbose_print = True,X_data_ndim=X_data)
# print('ARI APT for av_peaks=15, tooBig ',too_big_factor_i, adjusted_rand_score(np.asarray(true_label), np.asarray(best_labels_mst_lv)), metrics.adjusted_mutual_info_score(true_label, best_labels_mst_lv))
# write_list_to_file(best_labels_mst_lv, '/home/shobi/Thesis/Rcode/my_first_R_project/APT15_PBMC_'+'toobig'+str(too_big_factor_i*100)+'Perp'+str(perp)+'sep21v8.txt')
# write_list_to_file(majority_truth_labels_mst_lv,'/home/shobi/Thesis/Rcode/my_first_R_project/APT15_majtruth_PBMC_'+'toobig'+str(too_big_factor_i*100)+'Perp'+str(perp)+'sep21v8.txt')
# figtest = plt.figure(figsize=(36, 12))
# ax1 = figtest.add_subplot(1, 2, 1, projection='3d')
# ax2 = figtest.add_subplot(1, 2, 2, projection='3d')
# ax1= plot_10x.plot_onemethod(ax1, X_LV_embedded, true_label, true_label, method='APT',
#                                 onevsall='CD4+/CD45RO+ Memory')
# ax2 = plot_10x.plot_onemethod(ax2, X_LV_embedded, best_labels_mst_lv, true_label, method='APT',
#                                 onevsall='CD4+/CD45RO+ Memory')
# plt.show()
# #df_accuracy_mst_lv, best_labels_mst_lv, sigma_opt_lv, min_clustersize_mst_lv, tooclose_factor_opt, onevsall_opt_mst_lv, majority_truth_labels_mst_lv = LC.multiclass_mst_accuracy(
#     #X_LV_embedded, true_label, df_all=None,av_peaks=20,too_big_factor=too_big_factor_i,verbose_print = True,X_data_ndim=X_data)
# #print('ARI APT for av_peaks=20, tooBig ',too_big_factor_i, adjusted_rand_score(np.asarray(true_label), np.asarray(best_labels_mst_lv) ),metrics.adjusted_mutual_info_score(true_label, best_labels_mst_lv))
# #write_list_to_file(best_labels_mst_lv, '/home/shobi/Thesis/Rcode/my_first_R_project/APT20_labels_sep21v8.txt_pbmc')
# df_accuracy_mst_lv, best_labels_mst_lv, sigma_opt_lv, min_clustersize_mst_lv, tooclose_factor_opt, onevsall_opt_mst_lv, majority_truth_labels_mst_lv = LC.multiclass_mst_accuracy(
#     X_LV_embedded, true_label, df_all=None,av_peaks=25,too_big_factor=too_big_factor_i,verbose_print = True,X_data_ndim=X_data)
# print('ARI APT for av_peaks=25, tooBig ',too_big_factor_i, adjusted_rand_score(np.asarray(true_label), np.asarray(best_labels_mst_lv)), metrics.adjusted_mutual_info_score(true_label, best_labels_mst_lv))
# write_list_to_file(best_labels_mst_lv, '/home/shobi/Thesis/Rcode/my_first_R_project/APT25_PBMC_'+'toobig'+str(too_big_factor_i*100)+'Perp'+str(perp)+'sep21v8.txt')
# write_list_to_file(majority_truth_labels_mst_lv,'/home/shobi/Thesis/Rcode/my_first_R_project/APT25_majtruth_PBMC_'+'toobig'+str(too_big_factor_i*100)+'Perp'+str(perp)+'sep21v8.txt')
# figtest = plt.figure(figsize=(36, 12))
# ax1 = figtest.add_subplot(1, 2, 1, projection='3d')
# ax2 = figtest.add_subplot(1, 2, 2, projection='3d')
# ax1 = plot_10x.plot_onemethod(ax1, X_LV_embedded, true_label, true_label, method='APT',
#                                 onevsall='CD4+/CD45RO+ Memory')
# ax2 = plot_10x.plot_onemethod(ax2, X_LV_embedded, best_labels_mst_lv, true_label, method='APT',
#                                 onevsall='CD4+/CD45RO+ Memory')
# plt.show()
# df_accuracy_mst_lv, best_labels_mst_lv, sigma_opt_lv, min_clustersize_mst_lv, tooclose_factor_opt, onevsall_opt_mst_lv, majority_truth_labels_mst_lv = LC.multiclass_mst_accuracy(
#     X_LV_embedded, true_label, df_all=None,av_peaks=30,too_big_factor=too_big_factor_i,verbose_print = True,X_data_ndim=X_data)
# write_list_to_file(best_labels_mst_lv, '/home/shobi/Thesis/Rcode/my_first_R_project/APT30_PBMC_'+'toobig'+str(too_big_factor_i*100)+'Perp30sep21v8.txt')
# write_list_to_file(majority_truth_labels_mst_lv,'/home/shobi/Thesis/Rcode/my_first_R_project/APT30_majtruth_PBMC_'+'toobig'+str(too_big_factor_i*100)+'Perp'+str(perp)+'sep21v8.txt')
# print('ARI APT for av_peaks=30, tooBig ',too_big_factor_i, adjusted_rand_score(np.asarray(true_label), np.asarray(best_labels_mst_lv)), metrics.adjusted_mutual_info_score(true_label, best_labels_mst_lv))
# figtest = plt.figure(figsize=(36, 12))
# ax1 = figtest.add_subplot(1, 2, 1, projection='3d')
# ax2 = figtest.add_subplot(1, 2, 2, projection='3d')
# ax1 = plot_10x.plot_onemethod(ax1, X_LV_embedded, true_label, true_label, method='APT',
#                                 onevsall='CD4+/CD45RO+ Memory')
# ax2 = plot_10x.plot_onemethod(ax2, X_LV_embedded, best_labels_mst_lv, true_label, method='APT',
#                                 onevsall='CD4+/CD45RO+ Memory')
# plt.show()
#
# #df_accuracy_mst_lv, best_labels_mst_lv, sigma_opt_lv, min_clustersize_mst_lv, tooclose_factor_opt, onevsall_opt_mst_lv, majority_truth_labels_mst_lv = LC.multiclass_mst_accuracy(
#     #X_LV_embedded, true_label, df_all=None,av_peaks=35,too_big_factor=too_big_factor_i,verbose_print = True,X_data_ndim=X_data)
# #print('ARI APT for av_peaks=35, tooBig ',too_big_factor_i, adjusted_rand_score(np.asarray(true_label), np.asarray(best_labels_mst_lv)), metrics.adjusted_mutual_info_score(true_label, best_labels_mst_lv))
# #write_list_to_file(best_labels_mst_lv, '/home/shobi/Thesis/Rcode/my_first_R_project/APT35_labels_sep21v8.txt_pbmc')
# #df_accuracy_mst_lv, best_labels_mst_lv, sigma_opt_lv, min_clustersize_mst_lv, tooclose_factor_opt, onevsall_opt_mst_lv, majority_truth_labels_mst_lv = LC.multiclass_mst_accuracy(
#     #X_LV_embedded, true_label, df_all=None,av_peaks=40,too_big_factor=too_big_factor_i,verbose_print = True,X_data_ndim=X_data)
# #print('ARI APT for av_peaks=40, tooBig ',too_big_factor_i, adjusted_rand_score(np.asarray(true_label), np.asarray(best_labels_mst_lv)), metrics.adjusted_mutual_info_score(true_label, best_labels_mst_lv))
# #write_list_to_file(best_labels_mst_lv, '/home/shobi/Thesis/Rcode/my_first_R_project/APT45_labels_sep21v8.txt_pbmc')
#
#
# print("TOO BIG FACTOR IS, ", too_big_factor_i)
# df_accuracy_mst_lv, best_labels_mst_lv, sigma_opt_lv, min_clustersize_mst_lv, tooclose_factor_opt, onevsall_opt_mst_lv, majority_truth_labels_mst_lv = LC.multiclass_mst_accuracy(
#     X_LV_embedded, true_label, df_all=None, av_peaks=100, too_big_factor=too_big_factor_i, verbose_print=True,
#     X_data_ndim=X_data)
#
# print('ARI APT for av_peaks=100, tooBig ', too_big_factor_i,
#       adjusted_rand_score(np.asarray(true_label), np.asarray(best_labels_mst_lv)),
#       metrics.adjusted_mutual_info_score(true_label, best_labels_mst_lv))
# write_list_to_file(best_labels_mst_lv, '/home/shobi/Thesis/Rcode/my_first_R_project/APT100_PBMC_' + 'toobig' + str(
#     too_big_factor_i * 100) + 'Perp' + str(perp) + 'sep21v8.txt')
# write_list_to_file(majority_truth_labels_mst_lv,
#                    '/home/shobi/Thesis/Rcode/my_first_R_project/APT100_majtruth_PBMC_' + 'toobig' + str(
#                        too_big_factor_i * 100) + 'Perp' + str(perp) + 'sep21v8.txt')
# figtest = plt.figure(figsize=(36, 12))
# ax1 = figtest.add_subplot(1, 2, 1, projection='3d')
# ax2 = figtest.add_subplot(1, 2, 2, projection='3d')
# ax1 = plot_10x.plot_onemethod(ax1, X_LV_embedded, true_label, true_label, method='APT',
#                               onevsall='CD4+/CD45RO+ Memory')
# ax2 = plot_10x.plot_onemethod(ax2, X_LV_embedded, best_labels_mst_lv, true_label, method='APT',
#                               onevsall='CD4+/CD45RO+ Memory')
# plt.show()
# '''