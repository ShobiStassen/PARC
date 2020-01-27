import pandas
import numpy as np
import fcsparser
import Performance_phenograph as pp
from sklearn.cluster import KMeans
import time
import LungCancer_function_minClusters_sep10 as LC
import plot_pbmc_mixture_10x as plot_10x
import matplotlib.pyplot as plt
import Louvain_igraph as ls
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn import metrics
from scipy import stats
import csv
#str_data_set = 'Mosmann_rare'
#str_data_set = 'Samusik_01'
str_data_set = 'Levine_13dim'
def write_list_to_file(input_list, csv_filename):
    """Write the list to csv file."""

    with open(csv_filename, "w") as outfile:
        for entries in input_list:
            outfile.write(str(entries))
            outfile.write("\n")

#path = "/home/shobi/Thesis/Data/CyTOF/Nilsson_rare.fcs"
path = "/home/shobi/Thesis/Data/CyTOF/"+str_data_set+".fcs"

print('data file is', path)
meta, data = fcsparser.parse(path, reformat_meta=True)
data = data.fillna(value = 999)
columns = data.columns
#for col in columns: #names of surface markers in the dataframe (column titles)
 #   print(col)

#print(data['label'].value_counts())
#print(data.head())

true_label  = data['label']
print(set(true_label), type(true_label))
print([[x, list(true_label).count(x)] for x in set(true_label)])
if str_data_set == 'Nilsson_rare': data = data.drop(['Time','label', 'FSC-A','FSC-H', 'FSC-W', 'SSC-A', 'PI'], axis = 1) #Nilsson
if str_data_set == 'Mosmann_rare': data = data.drop(['Time','label', 'FSC-A','FSC-H', 'FSC-W', 'SSC-A', 'SSC-H','SSC-W','Live_Dead'], axis = 1) #Mossmann
if str_data_set == 'Levine_32dim':  data = data.drop(['Time','Cell_length', 'DNA1', 'DNA2', 'Viability', 'file_number', 'event_number','label', 'individual'], axis = 1) #Levine_32dim
if str_data_set == 'Levine_13dim': data = data.drop(['label'], axis=1) #Levine_13dim
if str_data_set == 'Samusik_all': data = data.drop(['Time', 'Cell_length', 'BC1', 'BC2', 'BC3', 'BC4', 'BC5', 'BC6', 'DNA1', 'DNA2', 'Cisplatin', 'beadDist', 'sample', 'event', 'label'],axis =1) #Samsuik
if str_data_set == 'Samusik_01': data = data.drop(['Time', 'Cell_length', 'BC1', 'BC2', 'BC3', 'BC4', 'BC5', 'BC6', 'DNA1', 'DNA2', 'Cisplatin', 'beadDist', 'sample', 'event', 'label'],axis =1) #Samsuik
#print(data.head())

X_data = data.as_matrix()


#X_data= stats.zscore(X_data, axis=0)
'''
clus_labels = []
with open('/home/shobi/Thesis/Data/CyTOF/kmeans_labels_Nilsson_rare.txt', 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    next(csvreader)
    for row in csvreader:
        clus_labels.append(int(row[0]))
print(len(clus_labels))
print('dim of X_data', X_data.shape)
'''
'''
k_clusters = 40
kmeans = KMeans(n_clusters=k_clusters, max_iter=150).fit(X_data)
kmeans.labels_
print('num labels', len(set(kmeans.labels_)))
print('ari for kmeans with ',k_clusters,'groups', adjusted_rand_score(np.asarray(true_label), kmeans.labels_ ))
print("Adjusted Mutual Information: %0.5f"
      % metrics.adjusted_mutual_info_score(true_label, kmeans.labels_))
targets = list(set(true_label))
if len(targets) >=2: target_range = targets
else: target_range = [1]
N = len(true_label)
write_list_to_file(kmeans.labels_, '/home/shobi/Thesis/Data/CyTOF/kmeans_labels_louvain_Nilsson_rare.txt')

f1_accumulated =0
target_range = set(true_label)
for onevsall_val in target_range:
    vals_roc, predict_class_array = ls.accuracy_mst(clus_labels, true_label,
                                                             embedding_filename=None, clustering_algo='louvain', onevsall=onevsall_val)
    f1_current = vals_roc[1]
    f1_accumulated = f1_accumulated + f1_current * (list(true_label).count(onevsall_val)) / len(true_label)
print(f1_accumulated, ' f1 accumulated for KMEANS')
'''
predict_class_aggregate, df_accuracy, phenograph_labels, onevsall_opt, maj_truth_labels=ls.run_phenograph(X_data,true_label)
write_list_to_file(['label']+phenograph_labels, '/home/shobi/Thesis/Data/CyTOF/phenograph_labels_louvain_'+str_data_set+'v2.txt')
print("ARI %0.5f and AMI %0.5f",  adjusted_rand_score(np.asarray(true_label), np.asarray(phenograph_labels)),metrics.adjusted_mutual_info_score(true_label, np.asarray(phenograph_labels)))
'''
knn_in = 25
too_big_factor = 30
dist_std = -0.25
print(time.localtime())

print('call alph')

alph_file_name = 'alph_labels_louvain_'+str_data_set+'_knn'+str(knn_in)+ '_toobig'+str(too_big_factor)+'Std'+str(dist_std)+'_sep11v2.txt'
print('Making labels for ', str_data_set, ' in ', alph_file_name)
predict_class_aggregate, df_accuracy, alph_labels,knn_opt, onevsall_opt,maj_truth_labels = ls.run_mainlouvain(X_data, true_label, too_big_factor = too_big_factor/100, knn_in = knn_in,dist_std = dist_std, small_pop=50)
write_list_to_file(['label']+alph_labels, '/home/shobi/Thesis/Data/CyTOF/'+alph_file_name)
print('SAVED labels for ', str_data_set, ' in ', alph_file_name)
#print(majority_truth_labels_alph)
print(time.localtime())
    #print("ARI %0.5f and AMI %0.5f",  adjusted_rand_score(np.asarray(true_label), np.asarray(alph_labels)),metrics.adjusted_mutual_info_score(true_label, np.asarray(alph_labels)))
'''

time_start = time.time()

X_LV_embedded, lv_embedding_filename, lv_runtime, lv_embedding_plot_title = pp.run_lv(version=None,
                                                                                      input_data=X_data,
                                                                                      perplexity=30,
                                                                                      lr=1,
                                                                                      new_file_name='/home/shobi/CyTOF1',
                                                                                      new_folder_name=None, outdim=3)
X_LV_embedded = stats.zscore(X_LV_embedded, axis=0)
for too_big_factor_i in [0.3,0.2,0.05,0.1]:
#for too_big_factor_i in [0.3,0.2,0.1,0.05]:
    lv_runtime = time.time() - time_start
    print('LV ran for ', lv_runtime, ' seconds')
    df_accuracy_mst_lv, best_labels_mst_lv_20, sigma_opt_lv, min_clustersize_mst_lv, tooclose_factor_opt, onevsall_opt_mst_lv, majority_truth_labels_mst_lv = LC.multiclass_mst_accuracy(
        X_LV_embedded, true_label, df_all=None,av_peaks=20, too_big_factor = too_big_factor_i, X_data_ndim=X_data)
    write_list_to_file(['label']+best_labels_mst_lv_20, '/home/shobi/Thesis/Data/CyTOF/apt20_'+str(too_big_factor_i)+'v2_min5min20maxlabel9_labels_louvain_ndimAll_'+str_data_set+'.txt')
    print('SAVED labels for ', str_data_set)
    df_accuracy_mst_lv, best_labels_mst_lv_30, sigma_opt_lv, min_clustersize_mst_lv, tooclose_factor_opt, onevsall_opt_mst_lv, majority_truth_labels_mst_lv = LC.multiclass_mst_accuracy(
        X_LV_embedded, true_label, df_all=None,av_peaks=30, too_big_factor = too_big_factor_i, X_data_ndim=X_data)
    write_list_to_file(['label']+best_labels_mst_lv_30, '/home/shobi/Thesis/Data/CyTOF/apt30_'+str(too_big_factor_i)+'v2_min5min20maxlabel9_labels_louvainAll_ndimAll_'+str_data_set+'.txt')
    print('SAVED labels for ', str_data_set)
    df_accuracy_mst_lv, best_labels_mst_lv_75, sigma_opt_lv, min_clustersize_mst_lv, tooclose_factor_opt, onevsall_opt_mst_lv, majority_truth_labels_mst_lv = LC.multiclass_mst_accuracy(
        X_LV_embedded, true_label, df_all=None,av_peaks=75, too_big_factor = too_big_factor_i, X_data_ndim=X_data)
    write_list_to_file(['label']+best_labels_mst_lv_75, '/home/shobi/Thesis/Data/CyTOF/apt75_'+str(too_big_factor_i)+'v2_min5min20maxlabel9_labels_louvainAll_ndimAll_'+str_data_set+'.txt')
    print('SAVED labels for ', str_data_set)
    df_accuracy_mst_lv, best_labels_mst_lv_75, sigma_opt_lv, min_clustersize_mst_lv, tooclose_factor_opt, onevsall_opt_mst_lv, majority_truth_labels_mst_lv = LC.multiclass_mst_accuracy(
        X_LV_embedded, true_label, df_all=None, av_peaks=90, too_big_factor=too_big_factor_i, X_data_ndim=X_data)
    write_list_to_file(['label'] + best_labels_mst_lv_75, '/home/shobi/Thesis/Data/CyTOF/apt90_' + str(too_big_factor_i) + 'v2_min5min20maxlabel9_labels_louvainAll_ndimAll_' + str_data_set + '.txt')
    print('SAVED labels for ', str_data_set)
    df_accuracy_mst_lv, best_labels_mst_lv_60, sigma_opt_lv, min_clustersize_mst_lv, tooclose_factor_opt, onevsall_opt_mst_lv, majority_truth_labels_mst_lv = LC.multiclass_mst_accuracy(
        X_LV_embedded, true_label, df_all=None,av_peaks=60, too_big_factor = too_big_factor_i, X_data_ndim=X_data)
    write_list_to_file(['label']+best_labels_mst_lv_60, '/home/shobi/Thesis/Data/CyTOF/apt60_'+str(too_big_factor_i)+'v2_min5min20maxlabel9_labels_louvainAll_ndimAll_'+str_data_set+'.txt')
    print('SAVED labels for ', str_data_set)
    df_accuracy_mst_lv, best_labels_mst_lv_45, sigma_opt_lv, min_clustersize_mst_lv, tooclose_factor_opt, onevsall_opt_mst_lv, majority_truth_labels_mst_lv = LC.multiclass_mst_accuracy(
        X_LV_embedded, true_label, df_all=None,av_peaks=45, too_big_factor = too_big_factor_i, X_data_ndim=X_data)
    write_list_to_file(['label']+best_labels_mst_lv_45, '/home/shobi/Thesis/Data/CyTOF/apt45_'+str(too_big_factor_i)+'v2_min5min20maxlabel9_labels_louvainAll_ndimAll_'+str_data_set+'.txt')
    print('SAVED labels for ', str_data_set)
    df_accuracy_mst_lv, best_labels_mst_lv_auto, sigma_opt_lv, min_clustersize_mst_lv, tooclose_factor_opt, onevsall_opt_mst_lv, majority_truth_labels_mst_lv = LC.multiclass_mst_accuracy(
       X_LV_embedded, true_label, df_all=None,av_peaks=0, too_big_factor = too_big_factor_i)
    write_list_to_file(['label']+best_labels_mst_lv_auto, '/home/shobi/Thesis/Data/CyTOF/aptauto_'+str(too_big_factor_i)+'v2_min5min20maxlabel9_labels_louvainAll_ndimAll_'+str_data_set+'.txt')
    print('SAVED labels for ', str_data_set)
    print(time.time())
