import pandas as pd
import copy
import numpy as np
import fcsparser
from MulticoreTSNE import MulticoreTSNE as multicore_tsne
import Performance_phenograph as pp
from sklearn.cluster import KMeans
import time
import LungCancer_function_minClusters_sep10 as LC
import plot_pbmc_mixture_10x as plot_10x
import matplotlib.pyplot as plt
import Louvain_igraph_Jac24Sept as ls
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn import metrics
from scipy import stats
import csv
import somoclu  # USE CONDA FORGE TO INSTALL!!!! see intstructions on the github somoclu page
from neupy import algorithms, environment
from sompy.sompy import SOMFactory


# str_data_set = 'Mosmann_rare'
# str_data_set = 'Samusik_01'
# str_data_set = 'Levine_13dim'
def write_list_to_file(input_list, csv_filename):
    """Write the list to csv file."""

    with open(csv_filename, "w") as outfile:
        for entries in input_list:
            outfile.write(str(entries))
            outfile.write("\n")


def sigmoid(x):
    return 1 / (1 + np.exp(x))


def softmax(inputs):
    """
    Calculate the softmax for the give inputs (array)
    :param inputs:
    :return:
    """
    return np.exp(-inputs) / float(sum(np.exp(-inputs)))


# path = "/home/shobi/Thesis/Data/CyTOF/Nilsson_rare.fcs"
# path = "/home/shobi/Thesis/Data/CyTOF/"+str_data_set+".fcs"

# print('data file is', path)
# meta, data = fcsparser.parse(path, reformat_meta=True)
# data = data.fillna(value = 999)
# columns = data.columns
# for col in columns: #names of surface markers in the dataframe (column titles)
#   print(col)

# print(data['label'].value_counts())
# print(data.head())
'''
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
'''

# X_data= stats.zscore(X_data, axis=0)
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

'''

print(time.localtime())

print('call alph')
knn_in = 25
too_big_factor = 30
dist_std = -0.25

alph_file_name = 'alph_labels_louvain_'+str_data_set+'_knn'+str(knn_in)+ '_toobig'+str(too_big_factor)+'Std'+str(dist_std)+'_sep11v1.txt'
print('Making labels for ', str_data_set, ' in ', alph_file_name)
predict_class_aggregate, df_accuracy, alph_labels,knn_opt, onevsall_opt,maj_truth_labels = ls.run_mainlouvain(X_data, true_label, too_big_factor = too_big_factor/100, knn_in = knn_in,dist_std = dist_std, small_pop=50)
write_list_to_file(['label']+alph_labels, '/home/shobi/Thesis/Data/CyTOF/'+alph_file_name)
print('SAVED labels for ', str_data_set, ' in ', alph_file_name)
#print(majority_truth_labels_alph)
print(time.localtime())
    #print("ARI %0.5f and AMI %0.5f",  adjusted_rand_score(np.asarray(true_label), np.asarray(alph_labels)),metrics.adjusted_mutual_info_score(true_label, np.asarray(alph_labels)))
'''


def predict_raw(input_data, n_outputs, weight):
    n_samples = input_data.shape[0]
    output = np.zeros((n_samples, n_outputs))

    for i, input_row in enumerate(input_data):
        output[i, :] = pos_euclid_distance(
            input_row.reshape(1, -1), weight)
    return output


from numpy.linalg import norm


def pos_euclid_distance(input_data, weight):
    """
    Negative Euclidian distance between input
    data and weight.
    Parameters
    ----------
    input_data : array-like
        Input dataset.
    weight : array-like
        Neural network's weights.
    Returns
    -------
    array-like
    """
    euclid_dist = norm(input_data.T - weight, axis=0)
    return np.expand_dims(euclid_dist, axis=0)


def get_prelabels(gridlabel, grid_square_size):
    # grid_square_size along one dimension, assumes square
    # griblabels are BestMatchingUnits Nx2 array
    print(gridlabel)
    gridlabel = gridlabel.astype(int)
    x = gridlabel[:, 0].reshape(1, -1)
    y = gridlabel[:, 1].reshape(1, -1)
    prelabels = x + grid_square_size * y
    prelabels = np.asarray(list(prelabels[0, :]))
    print('prelabels', prelabels.shape, prelabels)

    return prelabels


def soft_max(z):
    t = np.exp(z)
    a = np.exp(z) / np.sum(t, axis=1).reshape(-1, 1)
    return a


def run_main(str_data_set, file_str):
    path = "/home/shobi/Thesis/Data/CyTOF/" + str_data_set + ".fcs"

    print('data file is', path)
    meta, data = fcsparser.parse(path, reformat_meta=True)
    data = data.fillna(value=999)

    columns = data.columns
    # for col in columns: #names of surface markers in the dataframe (column titles)
    #   print(col)

    # print(data['label'].value_counts())
    # print(data.head())

    true_label = data['label']
    print(set(true_label), type(true_label))
    print([[x, list(true_label).count(x)] for x in set(true_label)])
    if str_data_set == 'Nilsson_rare': data = data.drop(['Time', 'label', 'FSC-A', 'FSC-H', 'FSC-W', 'SSC-A', 'PI'],
                                                        axis=1)  # Nilsson
    if str_data_set == 'Mosmann_rare': data = data.drop(
        ['Time', 'label', 'FSC-A', 'FSC-H', 'FSC-W', 'SSC-A', 'SSC-H', 'SSC-W', 'Live_Dead'], axis=1)  # Mossmann
    if str_data_set == 'Levine_32dim':  data = data.drop(
        ['Time', 'Cell_length', 'DNA1', 'DNA2', 'Viability', 'file_number', 'event_number', 'label', 'individual'],
        axis=1)  # Levine_32dim
    if str_data_set == 'Levine_13dim': data = data.drop(['label'], axis=1)  # Levine_13dim
    if str_data_set == 'Samusik_all': data = data.drop(
        ['Time', 'Cell_length', 'BC1', 'BC2', 'BC3', 'BC4', 'BC5', 'BC6', 'DNA1', 'DNA2', 'Cisplatin', 'beadDist',
         'sample', 'event', 'label'], axis=1)  # Samsuik
    if str_data_set == 'Samusik_01': data = data.drop(
        ['Time', 'Cell_length', 'BC1', 'BC2', 'BC3', 'BC4', 'BC5', 'BC6', 'DNA1', 'DNA2', 'Cisplatin', 'beadDist',
         'sample', 'event', 'label'], axis=1)  # Samsuik

    # print(data.head())
    def get_xy(nodes_array):
        dm = np.divmod(nodes_array, 20)
        x = dm[1]  # the remainder
        y = dm[0]  # the quotient
        return np.concatenate((x, y)).reshape(2, -1).T

    def get_average_xy(xy_nodes, probabilities_vec):
        # prob is 1xk vector
        # xy nodes is kX2 array
        return np.dot(probabilities_vec, xy_nodes)

    X_data = data.as_matrix()
    print(X_data.shape)
    X_data_original = np.copy(X_data)
    print(X_data_original.shape)



    ## START ALPH CODE
    '''
    print('call alph')
    knn_in = 10
    too_big_factor = 30
    dist_std = 1
    small_pop = 1
    jac_std= 1
    alph_file_name = 'alph_labels_louvain_' + str_data_set + '_knn' + str(knn_in) + '_toobig' + str(
        too_big_factor) + 'Std' + str(dist_std) +'smallPop'+str(small_pop)+'Jac'+str(Jac_std)+file_str+'.txt'
    print('Making labels for ', str_data_set, ' in ', alph_file_name)
    predict_class_aggregate, df_accuracy, alph_labels, knn_opt, onevsall_opt, maj_truth_labels = ls.run_mainlouvain(
        X_data, true_label, too_big_factor=too_big_factor / 100, knn_in=knn_in, dist_std=dist_std, small_pop=small_pop, keep_all=False,Jac_std=Jac_std)
    alph_labels = np.asarray(alph_labels).T
    print('shape of alph labels', alph_labels.shape)
    print('shape of prelabels', prelabels.shape, prelabels)
    final_labels = alph_labels[prelabels]
    print('final labels', final_labels.shape ,final_labels)
    final_labels = list(final_labels)
    #write_list_to_file(['label'] + alph_labels, '/home/shobi/Thesis/Data/CyTOF/' + alph_file_name)
    write_list_to_file(['label'] + final_labels, '/home/shobi/Thesis/Data/CyTOF/' + alph_file_name)
    print('SAVED labels for ', str_data_set, ' in ', alph_file_name)
    '''
    ## END ALPH CODE

    '''
    time_start = time.time()

    X_LV_embedded, lv_embedding_filename, lv_runtime, lv_embedding_plot_title = pp.run_lv(version=None,
                                                                                          input_data=X_data,
                                                                                          perplexity=30,
                                                                                          lr=1,
                                                                                          new_file_name='/home/shobi/CyTOF1',
                                                                                          new_folder_name=None,
                                                                                          outdim=3)

    print('LV embedding ran for ', time.time() - time_start, ' seconds')
    '''
    ## START CODE FOR using APT on a graph made from the codebook

    knn_struct= ls.make_knn_struct(X_data,ef=50)
    X_data_copy = copy.deepcopy(X_data)
    neighbor_array, distance_array = knn_struct.knn_query(X_data_copy, k=5)
    density_array = np.empty([neighbor_array.shape[0],1])
    dense_neighbor_array = np.empty([neighbor_array.shape[0], 1])
    dense_distance_array = np.empty([neighbor_array.shape[0], 1])
    for i in range(neighbor_array.shape[0]):
        dense_neighbor_array[i, 0] = neighbor_array[i, np.argmax(density_array[neighbor_array[i,]])]
        dense_distance_array[i, 0] = distance_array[i, np.argmax(density_array[neighbor_array[i,]])]

    X_LV_embedded, dummy = ls.make_csrmatrix_noselfloop(dense_neighbor_array,dense_distance_array,dist_std=3,keep_all=True)

    ## END  CODE FOR using APT on a graph made from the codebook

    # tsne = multicore_tsne(n_jobs=4, perplexity=30, verbose=1, n_iter=1000, learning_rate=10, angle=0.2)

    # X_LV_embedded = tsne.fit_transform(codebook)
    # plt.scatter(X_LV_embedded[:,1],X_LV_embedded[:,1])
    # plt.show()
    # knn_struct = ls.make_knn_struct(X_LV_embedded, ef=50)
    # X_data_copy = copy.deepcopy(X_LV_embedded)
    # neighbor_array, distance_array = knn_struct.knn_query(X_data_copy, k=10)
    # X_LV_embedded, dummy = ls.make_csrmatrix_noselfloop(neighbor_array, distance_array, dist_std=3, keep_all=True)
    time_start = time.time()
    if str_data_set == 'Nilsson_rare' or str_data_set == "Mosmann_rare":
        big_factor_list = [0.3, 0.2, 0.1, 0.05]  # ,0.2,0.1,0.05]

    else:
        big_factor_list = [0.1, 0.2, 0.3]  # [0.1,0.05]#,0.2,0.05,0.1]
    # zz = np.zeros((bmu.shape[0],1))
    # X_LV_embedded = np.concatenate((all_average_xys,zz),axis=1)#np.concatenate((bmu, zz), axis=1)
    # X_LV_embedded = stats.zscore(X_LV_embedded, axis=0)
    for too_big_factor_i in big_factor_list:
        for num_peaks_i in [100, 90, 80, 70, 60,50, 40, 30, 20]:
            apt_start = time.time()
            print('start clustering with APT peaks = ', num_peaks_i, 'at too big factor ', too_big_factor_i)

            df_accuracy_mst_lv, best_labels_mst_lv, sigma_opt_lv, min_clustersize_mst_lv, tooclose_factor_opt, onevsall_opt_mst_lv, majority_truth_labels_mst_lv = LC.multiclass_mst_accuracy(
                X_LV_embedded, true_label, df_all=None, av_peaks=num_peaks_i, too_big_factor=too_big_factor_i,
                X_data_ndim=X_data, inputGraph=True,
                original_data=X_data)  # X_data_ndim is the nXd_dim matrix, original data is used in merge_too_close (so codebook)

            ##START CODE: RUNNING APT-SOM then need to use labels from APT to re-assign pre-labels
            # apt_labels = np.asarray(best_labels_mst_lv_100).T
            # print('shape of apt 100 labels', apt_labels.shape)
            # print('shape of prelabels', prelabels.shape, prelabels)
            # final_labels = apt_labels[prelabels]
            # print('final labels', final_labels.shape, final_labels)
            # final_labels = list(final_labels)
            ## END CODE: RUNNING APT-SOM then need to use labels from APT to re-assign pre-labels
            write_list_to_file(['label'] + best_labels_mst_lv,
                               '/home/shobi/Thesis/Data/CyTOF/apt' + str(num_peaks_i) + '_' + str(
                                   too_big_factor_i) + file_str + str_data_set + '.txt')
            print('APT Runtime ', time.time() - time_start, ' seconds')
            print('SAVED labels for ', num_peaks_i, ' peaks ', str_data_set)

            print(time.time())


def main():
    #run_main(str_data_set='Nilsson_rare', file_str='_newgraph_nocutting')
    #run_main(str_data_set="Samusik_01", file_str='')
    #run_main(str_data_set="Levine_13dim", file_str='_Oct9_v1_newtooclose')
    # run_main(str_data_set="Levine_32dim",file_str='Oct4_v1_sompy_sigma-1.5_20g_newtooclose' )
    # run_main(str_data_set = 'Mosmann_rare', file_str='_Oct9_v1_newtooclose')
    run_main(str_data_set="Samusik_all",file_str='_Oct4_v1_sompy_sigma-1.5_20g_newtooclose' )


if __name__ == '__main__':
    main()
    # have not run the currently typed filenames min20min20maxlabels10

