import Louvain_igraph_Jac24Sept as ls
from sklearn.metrics.cluster import adjusted_rand_score
import pandas as pd
import numpy as np

true_label = []
foldername = "/home/shobi/Thesis/Data/DicksonJan2020Spike/"
data = pd.read_csv(foldername+"1_100000_0.csv", header = None)
#delete = ["Dry Mass Radial Distribution", "Dry Mass Centroid Displacement", "Phase Radial Distribution"]
for d in [43,44,51]:
    data = data.drop(data.columns[d], axis=1)
data = (data - data.mean()) / data.std()
data_values = data.values

print(data.head, data_values.shape)


with open(foldername+"1_100000Class.csv", 'rt') as f:
    #next(f)
    for line in f:
        line = line.strip().replace('\"', '')
        true_label.append(line)
print(len(true_label), 'cell. Number of unique cell types', len(set(true_label)))
print("itemized count:", [[x, list(true_label).count(x)] for x in set(true_label)])
parc_method = 'leiden'
knn_in = 10
RS=4
too_big_factor = 30  # 30
dist_std = 2
small_pop =5
jac_std = 1#'median'
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
predict_class_aggregate, df_accuracy, parc_labels, knn_opt, onevsall_opt, majority_truth_labels_alph, time_end_knn_construct, time_end_prune, time_end_louvain, time_end_total, f1_accumulated, f1_mean, time_end_knn_query, num_edges = ls.run_mainlouvain(
            X_data=data_values, true_label=true_label,   too_big_factor=too_big_factor / 100, knn_in=knn_in, small_pop=small_pop, dist_std=dist_std, jac_std=jac_std,             keep_all_dist=keep_all, jac_weighted_edges=weighted, n_iter_leiden=5, parc_method=parc_method,
            partition_seed=RS)