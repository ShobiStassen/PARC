import time
import pandas as pd
from pandas import ExcelWriter
import copy
import numpy as np
import random
import csv
import Louvain_igraph_Jac24Sept as ls
import scanpy as sc


D1 = pd.read_csv("/home/shobi/Thesis/Data/10X_Immune_Wyatt/vdj_v1_hs_aggregated_donor1_binarized_matrix.csv", delimiter=',')
columns = ['barcode'] + [D1.columns[4:62]]
D1_genes = D1[:columns].copy()
#D1_genes = D1.iloc[:,1:62].copy()
print('d1 genes', D1_genes.columns)
print(D1_genes.shape)

h5_file = "/home/shobi/Thesis/Data/10X_Immune_Wyatt/vdj_v1_hs_aggregated_donor1_filtered_feature_bc_matrix.h5"
adata= sc.read_10x_h5(h5_file)

sc.pp.recipe_zheng17(adata, n_top_genes=3000)
df_expr = adata.to_df()
df_expr['barcode'] = df_expr.index
print(df_expr.head())

#df_expr.columns = var_names



df_subset_expr =  df_expr[df_expr["barcode"].isin(D1["barcode"])]


print('df subset expr', df_subset_expr.shape)


df_merged_total = pd.merge(D1_genes,df_subset_expr , on='barcode')

print('df gene+protein', df_merged_total.shape)
df_merged_total.to_csv("/home/shobi/Thesis/Data/10X_Immune_Wyatt/Donor1_ExprProteinCombo.txt", header=True, index=False, sep=',')
print("saved COMBO matrix to file")

#D1_genes.to_csv("/home/shobi/Thesis/Data/10X_Immune_Wyatt/Donor1_Genes_64dim.txt", header=True, index=False, sep=',')
#print("saved gene matrix to file")
D1_binders = D1.iloc[:,68:118].copy()
#print(D1.head(), D1.shape)
n=D1.shape[0]
#print(D1[4:67])
print(list(enumerate(list(D1), 0  )))
col_names = list(D1)
cell_name= list(np.where(D1_binders.values==True)[0])
#print('cellname', len(cell_name), cell_name[20], type(cell_name))
binder_name= list(np.where(D1_binders.values==True)[1])
binder_final_list = ['unknown']*n
D1['celltype']= random.choices([0,1,2],k=n)#
#print('no. unique tcr-aa', D1['cell_clono_cdr3_aa'].nunique())
ii=0
for i in cell_name:

    binder_final_list[i]=binder_name[ii]
    ii=ii+1

D1['binder']=binder_final_list

### Seurat labels start
Seurat_labels = list( pd.read_csv("/home/shobi/Thesis/Data/10X_Immune_Wyatt/Seurat_labels_Donor1_Genes_64dim.txt", header=None)[0])
print("seurat labels", len(Seurat_labels), len(set(Seurat_labels)), Seurat_labels[1:10])

D1['Seurat']= Seurat_labels

from collections import Counter
D1['Seurat']= Seurat_labels
print('no. unique clusters', D1['Seurat'].nunique())
for label_i in list(set(Seurat_labels)):
    list = D1['cell_clono_cdr3_aa'][D1['Seurat']==label_i]
    print(sum(D1['Seurat'] == label_i),(Counter(list)).most_common(10))

### Seurat labels end

df_group = D1.groupby('celltype')
df_tcr = df_group.apply(lambda x: x['cell_clono_cdr3_aa'].unique())
print(df_tcr)


#print(D1.head())
#print('rows 50-55',D1.iloc[50:55,:])

print(D1_binders.iloc[1,:])
binders_list = ['A0201_FLASKIGRLV_Ca2-indepen-Plip-A2_binder','A2402_CYTWNQMNL_WT1-(235-243)236M_Y_binder']
#print(D1[binders_list])
idx = D1.index[(D1['A0201_FLASKIGRLV_Ca2-indepen-Plip-A2_binder']==True) & (D1['A2402_CYTWNQMNL_WT1-(235-243)236M_Y_binder']==True)]
#print('idx',idx)
print(sum(sum((D1_binders==True).values)))

true_label = D1['binder']

X_data = D1_genes.values
print(type(X_data), X_data.shape)
knn_in=30
too_big_factor = 30
small_pop = 50
dist_std = 2
jac_std = 0.15
keep_all =True
if keep_all == True: dis_std_str = 'KeepAll'
else: dis_std_str = str(dist_std)
weighted =True
if weighted == True: weight_str = 'weighted'
else: weight_str = 'unweighted'
random_seed = 42
n = X_data.shape[0]
#true_label = random.choices([0,1,2],k=n)#list(np.zeros((1,n)).flatten())
#print(true_label)

predict_class_aggregate, df_accuracy, parc_labels,knn_opt, onevsall_opt, majority_truth_labels_alph,time_end_knn_construct, time_end_prune, time_end_louvain, time_end_total, f1_accumulated, f1_mean, time_end_knn_query, num_edges = ls.run_mainlouvain(X_data, true_label,
                                                                                    too_big_factor = too_big_factor/100, knn_in = knn_in,small_pop = small_pop, dist_std=dist_std, jac_std=jac_std, keep_all_dist=keep_all, jac_weighted_edges=weighted,n_iter_leiden=5, partition_seed=random_seed,compute_accuracy=False)
print('type of labels' ,type(parc_labels), set(parc_labels))
print(len(parc_labels))
print(set(parc_labels))
set_labels = set(parc_labels)
new_parc_labels = []
for i in set_labels:
    new_parc_labels.append(i)





from collections import Counter
D1['PARC']= parc_labels
#print('no. unique clusters', D1['PARC'].nunique())
for label_i in new_parc_labels:
    list = D1['cell_clono_cdr3_aa'][D1['PARC']==label_i]



for label_i in new_parc_labels:
    list = D1['cell_clono_cdr3_aa'][D1['PARC']==label_i]
    list_binders = D1['binder'][D1['PARC']==label_i]
    print('TCR:', sum(D1['PARC']==label_i),Counter(list).most_common(10))
    print("binders:", Counter(list_binders).most_common(10))