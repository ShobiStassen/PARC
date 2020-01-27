import pandas as pd
from pandas import ExcelWriter
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
plt.rcParams["font.family"] = "Times New Roman"
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 350
import Louvain_igraph_Jac24Sept as ls

from sklearn.metrics.cluster import adjusted_rand_score
from sklearn import metrics
from scipy import stats
import csv
#import somoclu #USE CONDA FORGE TO INSTALL!!!! see intstructions on the github somoclu page
#from neupy import algorithms, environment
#from sompy.sompy import SOMFactory

#str_data_set = 'Mosmann_rare'
#str_data_set = 'Samusik_01'
#str_data_set = 'Levine_13dim'
def func_mode(ll, min_pop):  # return MODE of list
    # If multiple items are maximal, the function returns the first one encountered.
    if ll.count(1)>min_pop:
        print('this list has rare pop above 100 cells')
        return 1
    else:
        print('this list has major pop',max(set(ll), key=ll.count))
        return max(set(ll), key=ll.count)
def plot_onemethod_2D(ax, X_embedded, model_labels, true_labels,  onevsall = 'rare', GroundTruth=False, min_pop_func_mode=100):

    print('here i am')
    X_dict = {}
    X_dict_true = {}

    Index_dict = {}

    X_plot = X_embedded

    mst_labels = model_labels
    num_groups = len(set(mst_labels))


    N = len(mst_labels)
    n_cancer = list(true_labels).count(onevsall)
    n_pbmc = N-n_cancer
    m = 999
    for k in range(N):
        x = X_plot[k, 0]
        y = X_plot[k, 1]

        X_dict.setdefault(mst_labels[k], []).append((x, y)) #coordinates of the points by mst groups
        Index_dict.setdefault(mst_labels[k], []).append(true_labels[k]) #true label kth data point grouped by mst_group
        X_dict_true.setdefault(true_labels[k],[]).append((x,y))
    sorted_keys = list(sorted(X_dict.keys()))
    print('in plot: number of distinct groups:', len(sorted_keys))
    # sorted_keys_dbscan =list(sorted(X_dict_dbscan.keys()))
    # print(sorted_keys, ' sorted keys')
    error_count = []
    rare_labels = []
    notrare_labels = []


    fp = 0
    fn = 0
    tp = 0
    tn = 0
    f1_score = 0
    precision = 0
    recall = 0
    f1_score = 0
    computed_ratio = 999

    for kk in sorted_keys:
        vals = [t for t in Index_dict[kk]]
        print('kk ', kk, 'has length ', len(vals))
        majority_val = func_mode(vals,min_pop_func_mode)
        if majority_val == onevsall:
            tp = tp + len([e for e in vals if e == onevsall])
            fp = fp + len([e for e in vals if e != onevsall])
            error_count.append(len([e for e in vals if e != majority_val]))
        else:
            fn = fn + len([e for e in vals if e == onevsall])
            tn = tn + len([e for e in vals if e != onevsall])
            error_count.append(len([e for e in vals if e != majority_val]))

        if (majority_val ==1):
            rare_labels.append(kk)
            print('found rare')

        if (majority_val == 0):
            notrare_labels.append(kk)
            print('found not rare')

    total_error_rate = sum(error_count)/N
    error_rate = (fp+fn)/(fp+fn+tn+tp)
    comp_n_cancer = tp + fp
    comp_n_pbmc = fn + tn
    tnr = tn / n_pbmc
    fnr = fn / n_cancer
    print('fnr is', fnr)
    tpr = tp / n_cancer
    fpr = fp / n_pbmc
    if comp_n_cancer != 0:
        computed_ratio = comp_n_pbmc / comp_n_cancer
        # print('computed-ratio is:', computed_ratio, ':1' )
    if tp != 0 or fn != 0: recall = tp / (tp + fn)  # ability to find all positives
    if tp != 0 or fp != 0: precision = tp / (tp + fp)  # ability to not misclassify negatives as positives
    if precision != 0 or recall != 0:
        f1_score = precision * recall * 2 / (precision + recall)
        print('f1-score is', f1_score)

    colors_Dendritic = plt.cm.Greys_r(np.linspace(0.2, 0.6, len(notrare_labels)))
    colors_B = plt.cm.Reds_r(np.linspace(0.2,0.4, len(rare_labels)))


    pair_color_group_list = [(colors_Dendritic, notrare_labels, ['Majority'] * len(notrare_labels)), (colors_B, rare_labels, ['Rare'] * len(rare_labels))]
    # pair_color_group_list = [(colors_B, B_labels, ['CD19+ B']*len(B_labels)),(colors_CD4CD25,CD4CD25_labels, ['CD4+/CD25 T Reg']*len(CD4CD25_labels)),(colors_Dendritic, Dendritic_labels, ['Dendritic']*len(Dendritic_labels)),(colors_NK, NK_labels, ['CD56+ NK']*len(NK_labels)),(colors_CD4Helper, CD4Helper_labels, ['CD4+ T Helper2']*len(CD4Helper_labels)),
    #                          (colors_CD34, CD34_labels, ['CD34+'] * len(CD34_labels)),(colors_CD4CD45_RA, CD4CD45_RA_labels, ['CD4+/CD45RA+/CD25- Naive T'] * len(CD4CD45_RA_labels)),(colors_Monocyte, Monocyte_labels, ['CD14+ Monocyte'] * len(Monocyte_labels)),(colors_CD4CD45_RO, CD4CD45_RO_labels, ['CD4+/CD45RO+ Memory'] * len(CD4CD45_RO_labels)),(colors_CD8CytoT, CD8CytoT_labels, ['CD8+ Cytotoxic T'] * len(CD8CytoT_labels)),(colors_CD8Naive,CD8Naive_labels, ['CD8+/CD45RA+ Naive Cytotoxic'] * len(CD8Naive_labels))]
    for color, group, name in pair_color_group_list:
        for color_m, ll_m, label_m in zip(color, group,name):
            x = [t[0] for t in X_dict[ll_m]]
            half = int(len(x)/2)
            y = [t[1] for t in X_dict[ll_m]]
            #x = x[0:half]
            #y = y[0:half]
            #print('color of group', ll_m, label_m, color_m, colors.to_hex(color_m))
            population = len(x)
            print(population, 'is the population of cluster', ll_m)
            if label_m =='Majority': alpha_val = 0.4
            else:alpha_val = 0.8
            if GroundTruth==False: ax.scatter(x, y, color=color_m, s=0.8, alpha=alpha_val,label=label_m + ' '+str(ll_m)+' ' + str(population), edgecolors= 'none')
            else: ax.scatter(x, y, color=color_m, s=0.8, alpha=alpha_val,label=label_m +' ' + str(population),edgecolors= 'none')
            #ax.annotate(label_m, xytext=(np.mean(x), np.mean(y)), xy=(np.mean(x), np.mean(y)), color='black',  weight='semibold',fontsize = 4)

            #ax.scatter(x, y, color=color_m, s=2, alpha=0.6, label= label_m+'_' + str(ll_m) + ' Cellcount = ' + str(len(x)))
            #ax.annotate(ll_m, xyztext=(np.mean(x), np.mean(y),np.mean(z)), xy=(np.mean(x), np.mean(y)), color='black',
            #               weight='semibold')

    #ax.text(0.95, 0.01, "number of groups: " + " {:.0f}".format(num_groups) + " FP: " + " {:.2f}".format(
    #    fp * 100 / n_pbmc) + "%. FN of " + "{:.2f}".format(fn * 100 / (n_cancer)) + '%', transform=ax.transAxes,
    #           verticalalignment='bottom', horizontalalignment='right', color='green', fontsize=10)

    ax.axis('tight')
    title_str1 = 'ALPH \n'+"number of groups: " + " {:.0f}".format(num_groups)


    ax.set_title(title_str1, size=8)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
    #ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8,markerscale=10) #markerscale 10
    #
    # #make panes transparent
    # ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # # make the grid lines transparent
    # ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    # ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    # ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    # ax.autoscale(tight=True)
    return ax

def write_list_to_file(input_list, txt_filename):
    """Write the list to csv file."""
    txt_filename = txt_filename +'.txt'
    with open(txt_filename, "w") as outfile:
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

#path = "/home/shobi/Thesis/Data/CyTOF/Nilsson_rare.fcs"
#path = "/home/shobi/Thesis/Data/CyTOF/"+str_data_set+".fcs"

#print('data file is', path)
#meta, data = fcsparser.parse(path, reformat_meta=True)
#data = data.fillna(value = 999)
#columns = data.columns
#for col in columns: #names of surface markers in the dataframe (column titles)
 #   print(col)

#print(data['label'].value_counts())
#print(data.head())
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


def predict_raw(input_data,n_outputs, weight):

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

def get_prelabels(gridlabel,grid_square_size):
    #grid_square_size along one dimension, assumes square
    #griblabels are BestMatchingUnits Nx2 array
    print(gridlabel)
    gridlabel = gridlabel.astype(int)
    x = gridlabel[:,0].reshape(1,-1)
    y = gridlabel[:,1].reshape(1,-1)
    prelabels = x+grid_square_size*y
    prelabels = np.asarray(list(prelabels[0,:]))
    print('prelabels',prelabels.shape, prelabels)

    return prelabels

def soft_max(z):
    t = np.exp(z)
    a = np.exp(z) / np.sum(t, axis=1).reshape(-1,1)
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
    #true_label.astype('int64')
    true_label.to_csv("/home/shobi/Thesis/Data/CyTOF/" + str_data_set + "TrueLabel_Oct22.txt", header=True,
                      index=False, sep=',')




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
        dm = np.divmod(nodes_array,20)
        x = dm[1] #the remainder
        y = dm[0] #the quotient
        return np.concatenate((x,y)).reshape(2,-1).T
    def get_average_xy(xy_nodes, probabilities_vec):
        #prob is 1xk vector
        #xy nodes is kX2 array
        return np.dot(probabilities_vec,xy_nodes)
    print(data.shape, 'shape of data to be written')
    data.to_csv("/home/shobi/Thesis/Data/CyTOF/"+str_data_set+"Oct22_raw.txt", header=True, index=False, sep=',')
    print('SAVED oct22 dataset', str_data_set)
    X_data = data.as_matrix()
    print(X_data.shape)
    X_data_original = np.copy(X_data)
    print(X_data_original.shape)


    #sm = SOMFactory().build(X_data, normalization=False, initialization='pca', mapsize=(20, 20))
    #sm.train(n_job=4,  train_rough_len=1, train_finetune_len=1, train_rough_radiusin=10,train_rough_radiusfin=0,train_finetune_radiusin=0.5,train_finetune_radiusfin=0)
    #print('shape codebook', sm.codebook.matrix.shape)
    nclusters = 400
    #prod = predict_raw(X_data,nclusters, weight= sm.codebook.matrix.T)
    #print(prod[0:2,:])
    #prod = np.dot(sm.codebook.matrix, data.T)
    #y2 = np.einsum('ij,ij->i', sm.codebook.matrix, sm.codebook.matrix)
    #print(y2.shape)
    #prod *= -2
    #prod +=  y2.reshape(prod.shape[0], 1)
    #X_data = prod
    #bmu = np.empty((X_data.shape[0], 2))
    #bmu[:, 0] = np.argmin(prod, axis=0)
    #bmu[:, 1] = np.min(prod, axis=0)
    #xy = sm.bmu_ind_to_xy(bmu[:, 0])*(0.1)
    #print(xy[:, 0:2])
    #X_data = xy #prod.T
    #X_data = sigmoid(X_data)
    #print(X_data.shape)
    #print(X_data[0:5,:])
    #X_data = np.concatenate((X_data, xy[:,0:2]), axis=1)
    #print(X_data.shape)
    #print(sm.find_bmu(data)[0, :])
    #print(sm.codebook.matrix)
    #bmu = np.empty((data.shape[0], 2))
    # print(sm.predict_probability(data))
    #prod = np.dot(sm.codebook.matrix, X_data.T)


    #sofm = algorithms.SOFM(n_inputs=X_data.shape[1], step=0.5, verbose=True,learning_radius=0, features_grid=(10, 10))
    #print('train SOM')
    #sofm.train(data, epochs=20)
    #print('predict SOM')
    #X_data = (sofm.predict_raw(data))

    #print(sofm.predict(data))
    #print(sofm.weight.shape, sofm.weight)
    #print(np.dot(sofm.predict_raw(data), sofm.weight.T))
    #X_data = stats.zscore(X_data, axis=0) We dont normalize the 50 PCs, this degrades the results


    #reader = csv.reader(open("/home/shobi/Thesis/Data/CyTOF/"+str_data_set+"_sq10Bmus.txt", 'rt'),
    #                    delimiter=",")  # pca 50 dims x68K for the mixture of PBMC
    # reader = csv.reader(open("/home/shobi/test_pure.txt", 'rt'),delimiter = ",") #datamatrix of the pure bead PBMC 94655*50 MATRIX
    #bmu = list(reader)  # [1:]
    #bmu = np.array(bmu).astype("int")
    #print('bmu shape',bmu.shape, bmu) #NX2


    ##SOMS CODE START

    # grid_size = 20
    # som = somoclu.Somoclu(n_columns=grid_size, n_rows=grid_size, maptype="planar", compactsupport=False, initialization='pca', std_coeff=0.3, verbose=2)
    # som.train(X_data_original,epochs = 10)
    # codebook = som.codebook
    # codebook = codebook.reshape(codebook.shape[0] * codebook.shape[1], codebook.shape[2])
    # print('dim codebook', codebook.shape)
    # surface_state = som.get_surface_state()
    # print('surface_state', surface_state.shape)
    # bmus = som.get_bmus(surface_state)
    # print('bmus', bmus.shape)
    # prelabels = get_prelabels(bmus,grid_size)
    # #write_list_to_file(['label'] + list(prelabels),
    # #                  '/home/shobi/Thesis/Data/CyTOF/prelabels_' + str_data_set + file_str + '.txt')
    # print('prelabels', prelabels)
    # print(np.asarray(prelabels[0]))
    # X_data = codebook
    # print(X_data.shape, 'shape of Xdata')

    ##SOMS CODE END


    #codes = csv.reader(open("/home/shobi/Thesis/Data/CyTOF/"+str_data_set+"_sq10Codes.txt", 'rt'),
    #                    delimiter=",")
    #codes = list(codes)  # [1:]
    #codes = np.array(codes).astype("float")
    #prelabels = get_prelabels(bmu,grid_square=10)
    #print('codebook shape', codes.shape) #nodes*dimensions
    #distance_matrix = predict_raw(X_data_original, codes.shape[0], codes.T)



    '''
    print('start double argsort')
    x_toobig_ind = np.argsort(np.argsort(distance_matrix, axis=1))
    x_ind = x_toobig_ind > 4  # discard the largest distances. kepp 8 smallest
    print('finished argsort')
    distance_matrix[x_ind] = 100000
    softmax_probabilities = soft_max(-1*distance_matrix)
    xys = get_xy(np.array(np.where(softmax_probabilities[0, :] > 0)[0]))
    probs_locs = np.where(softmax_probabilities[0,:]>0)[0]
    probs = softmax_probabilities[0,probs_locs]
    print('nonzero',probs_locs ,probs,xys, get_average_xy(np.array(xys),probs))
    all_average_xys = np.empty((X_data.shape[0],2))
    print('start computing average xy')
    for row_number, row in enumerate(softmax_probabilities):
        xys = get_xy(np.array(np.where(row > 0)[0]))
        probs_locs = np.where(row>0)[0]
        probs = row[probs_locs]
        average_xy = get_average_xy(np.array(xys),probs)
        all_average_xys[row_number,0] =average_xy[0]
        all_average_xys[row_number, 1] = average_xy[1]
    print('completed computing average xy', all_average_xys,all_average_xys.shape )

    print('shape of softmax probabilities',softmax_probabilities.shape)
    print('sum of softmax probabilities', np.sum(softmax_probabilities,axis=1))
    '''
    #X_data = predict_raw(X_data, weights.shape[1], weight=weights)

    #print('raw',X_data.shape, X_data)
    #x_toobig_ind = np.argsort(X_data,axis=1)
    #x_ind = x_toobig_ind >370 #indices to discard
    #X_data = 1/X_data
    #X_data[x_ind] = 0
    #X_data *= 10
    #X_data = 1- X_data/X_data.sum(axis=1)[:,None]

    #X_data[x_ind] = 0
    #X_data= stats.zscore(X_data, axis=1)
    #X_data[x_ind] = 0

    #print('probability', X_data.shape,X_data)


    #print('start phenograph')
    #knn_pheno=10

    #predict_class_aggregate, df_accuracy, phenograph_labels, onevsall_opt, majority_truth_labels, pheno_time, f1_mean_pheno= ls.run_phenograph(X_data,
    #                                                                                                            true_label,knn=knn_pheno)
    #write_list_to_file(['label'] + phenograph_labels,
    #                   '/home/shobi/Thesis/Data/CyTOF/phenograph_labels_louvain_' + str_data_set+'K'+str(knn_pheno) + 'v3.txt')
    #print("ARI %0.5f and AMI %0.5f", adjusted_rand_score(np.asarray(true_label), np.asarray(phenograph_labels)),
    #      metrics.adjusted_mutual_info_score(true_label, np.asarray(phenograph_labels)))

    #if str_data_set =="Nilsson_rare" or str_data_set =="Mosmann_rare": bmu = np.arcsinh(bmu/150)
    #else: bmu = np.arcsinh(bmu/5)
    #scale_factor = 10
    #print('bmu scaled by', scale_factor)
    #X_data = softmax_probabilities
    #X_data = np.concatenate((X_data, (bmu-np.mean(bmu))/10), axis=1)
    #X_data = np.concatenate((X_data, bmu), axis=1)
    #print('X_data shape',X_data.shape)
    #print(np.sum(X_data>0.1,axis=1))
    #print(X_data[0:2,])
    #X_data = bmu

    ## START ALPH CODE
    print("call ALPH/PHENO")
    knn_in = 5#100
    print('knn is', knn_in)
    too_big_factor = 30#
    dist_std = 2
    small_pop = 10
    jac_std= 3#0.15#'median'
    weighted = True
    keep_all = False

    for run in [0,1,2,3,4]:
        partition_seed = 2#4
        weighted = False
        for small_pop in [10]:
            for partition_seed in [4]:#0.05,0.1,0.15,0.2,'median',0.25,0.3,0.5,1]:
                if weighted == True:
                    w = 'weighted'
                else:
                    w = 'notweighted'
                for keep_all in [True]:
                    if keep_all == True:dist_std_str = 'keepAllTrue'
                    else: dist_std_str = "KeepAllFalse"+str(dist_std)
                    print('keep all is', keep_all)
                    import math
                    from copy import deepcopy
                    #list_k =  [X_data.shape[0]]#[i for i in range(32000,X_data.shape[0],50000)] +[X_data.shape[0]] +[200,1000,2000,4000,8000,16000] #[X_data.shape[0]]#
                    #list_k = [X_data.shape[0]]#+[1000,2000,4000,8000,16000,32000,64000] +[i for i in range(100000,X_data.shape[0]+1,50000)]
                    #list_of_lists = []
                    X_copy = deepcopy(X_data)
                    #for k in list_k:
                    #indices = np.random.randint(0, X_data.shape[0], k)
                    #X_data_k = X_copy[indices]
                    #X_data_k = X_copy[0:k,:]
                    #print(X_data_k.shape, 'shape of sampled data')
                    #print(w)
                    alph_file_name = 'PARC_' + str_data_set + '_knn' + str(knn_in) + '_toobig' + str(
                        too_big_factor) + 'Std' + dist_std_str + 'smallPop' + str(small_pop) + 'Jac' + str(jac_std) + w + 'seed'+str(partition_seed)+'run'+str(run)+ file_str

                    #alph_file_name = 'alph_labels_louvain_' + str_data_set +'_N'+str(k) +'_knn' + str(knn_in) + '_toobig' + str(
                    #    too_big_factor) + 'Std' + dist_std_str +'smallPop'+str(small_pop)+'Jac'+str(jac_std)+w+file_str
                    print('Making labels for ', str_data_set, ' in ', alph_file_name)
                    #true_label_k = list(true_label[indices]) #list(true_label

                    predict_class_aggregate, df_accuracy, alph_labels, knn_opt, maj_truth_labels_alph, onevsall_opt, time_end_knn, time_end_prune, time_end_louvain, time_end_total, f1_accumulated, f1_mean, time_end_knn_query, num_edges = ls.run_mainlouvain(
                        X_data, true_label, too_big_factor=too_big_factor / 100, knn_in=knn_in, dist_std=dist_std,
                        small_pop=small_pop, keep_all_dist=keep_all, jac_std=jac_std,jac_weighted_edges=weighted, partition_seed=partition_seed, n_iter_leiden=5)  # means keep all as per distance pruning
                    write_list_to_file(['label'] + alph_labels, '/home/shobi/Thesis/Data/CyTOF/Runtimes/' + alph_file_name)

    print("ARI %0.5f and AMI %0.5f", adjusted_rand_score(np.asarray(true_label), np.asarray(alph_labels)),
          metrics.adjusted_mutual_info_score(true_label, np.asarray(alph_labels)))


    #list_of_lists.append([k,weighted,jac_std,keep_all,dist_std,time_end_knn + time_end_knn_query, time_end_prune, time_end_louvain, time_end_total,  len(set(alph_labels)), f1_accumulated, f1_mean, num_edges])
    '''
    predict_class_aggregate, df_accuracy, pheno_labels, onevsall_opt, majority_truth_labels, pheno_time, f1_mean_pheno = ls.run_phenograph(
        X_data, true_label, knn=30)
    print("ARI %0.5f and AMI %0.5f", adjusted_rand_score(np.asarray(true_label), np.asarray(pheno_labels)))
    '''
    #make suitable true labels

    true_val = []
    print(true_label)

    for target in set(true_label):
        true_val.append(target)
    print('target',true_val)
    true_val_0 = true_val[0]
    true_val_1 = true_val[1]

    i = 0
    temp = true_label
    true_labels = []
    for val in temp:
        if val == true_val_0:
            true_labels.append(0)
        else:
            true_labels.append(1)

    from FItSNE import fast_tsne

    X_plot = fast_tsne.fast_tsne(X_data, learning_rate=10000, perplexity=20, max_iter=500)

    figtest = plt.figure(figsize=(36, 12))
    ax1 = figtest.add_subplot(1, 3, 1)  # , projection='3d')
    ax2 = figtest.add_subplot(1, 3, 2)  # , projection='3d')
    ax3 = figtest.add_subplot(1, 3, 3)  # , projection='3d')
    print(true_label)
    ax1 = plot_onemethod_2D(ax1, X_plot, true_labels, true_labels,
                                      onevsall=1, GroundTruth=True, min_pop_func_mode=100)
    ax2 = plot_onemethod_2D(ax2, X_plot, alph_labels, true_labels,min_pop_func_mode=100,
                                     onevsall=1)

    ax3 = plot_onemethod_2D(ax3, X_plot, alph_labels_noprune, true_labels,
                                     onevsall=1,min_pop_func_mode=300)
    plt.show()

    write_list_to_file(['label'] + alph_labels, '/home/shobi/Thesis/Data/CyTOF/Runtimes/' + alph_file_name)

    df = pd.DataFrame(list_of_lists,
                      columns=['N', 'weighted', 'jac', 'keep_all','dist', 'knn time', 'pruning time', 'clustering time', 'total time', 'num clusters',
                               'f1_accumulated', 'f1-mean', 'num edges of graph'])
    excel_file_name_alph = '/home/shobi/Thesis/Data/CyTOF/Runtimes/'+alph_file_name+'.xlsx'

    writer = ExcelWriter(excel_file_name_alph)
    df.to_excel(writer, 'alph', index=False)
    writer.save()
    print('save alph stats')

    '''
    #pheno
    pheno_k = 30
    excel_file_name = '/home/shobi/Thesis/Data/CyTOF/Runtimes/Phenotest_' + str_data_set + '_k' + str(
        pheno_k) + 'Feb27.xlsx'
    writer = ExcelWriter(excel_file_name)
    predict_class_aggregate, df_accuracy, pheno_labels, onevsall_opt, majority_truth_labels, pheno_time,f1_mean_pheno = ls.run_phenograph(X_data,true_label, knn =pheno_k)
    list_of_lists_pheno = []
    list_of_lists_pheno.append([k, pheno_time, len(set(pheno_labels)), f1_mean_pheno])
    write_list_to_file(['label'] + pheno_labels, '/home/shobi/Thesis/Data/CyTOF/Runtimes/' + str_data_set+'PhenoK'+str(pheno_k)+'.txt')
    df_pheno = pd.DataFrame(list_of_lists_pheno, columns=['N', 'pheno time','num clusters','mean F1-score'])
    df_pheno.to_excel(writer, 'pheno', index=False)
    writer.save()
    '''


    # alph_labels = np.asarray(alph_labels).T
    # print('shape of alph labels', alph_labels.shape)
    # print('shape of prelabels', prelabels.shape, prelabels)
    # final_labels = alph_labels[prelabels]
    # print('final labels', final_labels.shape ,final_labels)
    # final_labels = list(final_labels)
    #write_list_to_file(['label'] + pheno_labels, '/home/shobi/Thesis/Data/CyTOF/' + alph_file_name)

    ## END ALPH CODE

    ## START APT CODE

    time_start = time.time()

    X_LV_embedded, lv_embedding_filename, lv_runtime, lv_embedding_plot_title = pp.run_lv(version=None,
                                                                                          input_data=X_data,
                                                                                          perplexity=30,
                                                                                          lr=1,
                                                                                          new_file_name='/home/shobi/CyTOF1',
                                                                                          new_folder_name=None, outdim=3)

    print('LV embedding ran for ',time.time() - time_start, ' seconds')
    predict_class_aggregate, df_accuracy, alph_labels, knn_opt, onevsall_opt, maj_truth_labels, time_end_knn, time_end_prune, time_end_louvain, time_end_total = ls.run_mainlouvain(
        X_LV_embedded, true_label, too_big_factor=30 / 100, knn_in=30, dist_std=3,
        small_pop=10, keep_all=True, Jac_std=1)  # means keep all as per distance pruning
    write_list_to_file(['label'] + alph_labels, '/home/shobi/Thesis/Data/CyTOF/ALPHonLV_'+str_data_set+'.txt')


    #from sklearn.decomposition import PCA
    #pca = PCA(n_components=3)
    #pcs = pca.fit_transform(X_data_original)


    ## START CODE FOR using APT on a graph made from the codebook

    #knn_struct= ls.make_knn_struct(codebook,ef=50)
    #X_data_copy = copy.deepcopy(codebook)
    #neighbor_array, distance_array = knn_struct.knn_query(X_data_copy, k=10)
    #X_LV_embedded, dummy = ls.make_csrmatrix_noselfloop(neighbor_array,distance_array,dist_std=3,keep_all=True)

    ## END  CODE FOR using APT on a graph made from the codebook

    #tsne = multicore_tsne(n_jobs=4, perplexity=30, verbose=1, n_iter=1000, learning_rate=10, angle=0.2)

    #X_LV_embedded = tsne.fit_transform(codebook)
    #plt.scatter(X_LV_embedded[:,1],X_LV_embedded[:,1])
    #plt.show()
    #knn_struct = ls.make_knn_struct(X_LV_embedded, ef=50)
    #X_data_copy = copy.deepcopy(X_LV_embedded)
    #neighbor_array, distance_array = knn_struct.knn_query(X_data_copy, k=10)
    #X_LV_embedded, dummy = ls.make_csrmatrix_noselfloop(neighbor_array, distance_array, dist_std=3, keep_all=True)

    time_start = time.time()
    if str_data_set =='Nilsson_rare' or str_data_set =="Mosmann_rare": big_factor_list = [0.3]#,0.2,0.1,0.05]#,0.2,0.1,0.05]

    else: big_factor_list = [0.3]#,0.2,0.1,0.05]#[0.1,0.05]#,0.2,0.05,0.1]
    #zz = np.zeros((bmu.shape[0],1))
    #X_LV_embedded = np.concatenate((all_average_xys,zz),axis=1)#np.concatenate((bmu, zz), axis=1)
    #X_LV_embedded = stats.zscore(X_LV_embedded, axis=0)
    for too_big_factor_i in big_factor_list:
        for num_peaks_i in [50,0]:
            for small_pop_i in [10]:
                apt_start = time.time()
                print('start clustering with APT peaks = ', num_peaks_i, 'at too big factor ', too_big_factor_i)


                df_accuracy_mst_lv, best_labels_mst_lv, sigma_opt_lv, min_clustersize_mst_lv, tooclose_factor_opt, onevsall_opt_mst_lv, majority_truth_labels_mst_lv, f1_accumulated, f1_mean = LC.multiclass_mst_accuracy(
                    X_LV_embedded, true_label, df_all=None, av_peaks=num_peaks_i, too_big_factor=too_big_factor_i,inputGraph=False, original_data=X_data_original, min_clustersize=small_pop_i,peak_threshhold=-1) #X_data_ndim is the nXd_dim matrix, original data is used in merge_too_close and apply_sigma(so codebook)

                ##START CODE: RUNNING APT-SOM then need to use labels from APT to re-assign pre-labels
                # apt_labels = np.asarray(best_labels_mst_lv).T
                # print('shape of apt 100 labels', apt_labels.shape)
                # print('shape of prelabels', prelabels.shape, prelabels)
                # final_labels = apt_labels[prelabels]
                # print('final labels', final_labels.shape, final_labels)
                # final_labels = list(final_labels)
                ## END CODE: RUNNING APT-SOM then need to use labels from APT to re-assign pre-labels
                file_name_save = '/home/shobi/Thesis/Data/CyTOF/apt'+str(num_peaks_i)+'_' + str(
                    too_big_factor_i) +'smallpop'+str(small_pop_i)+ file_str + str_data_set + '.txt'
                write_list_to_file(['label']+ best_labels_mst_lv, file_name_save)
                print('APT Runtime ', time.time() - time_start, ' seconds')
                print('SAVED labels for ' ,num_peaks_i,' peaks ', str_data_set)
                print('saved to', file_name_save)

    '''
            df_accuracy_mst_lv, best_labels_mst_lv_60, sigma_opt_lv, min_clustersize_mst_lv, tooclose_factor_opt, onevsall_opt_mst_lv, majority_truth_labels_mst_lv = LC.multiclass_mst_accuracy(
                X_LV_embedded, true_label, df_all=None, av_peaks=60, too_big_factor=too_big_factor_i, X_data_ndim=X_data, inputGraph=True, original_data = codebook)
            apt_labels = np.asarray(best_labels_mst_lv_60).T
            print('shape of apt 60 labels', apt_labels.shape)
            print('shape of prelabels', prelabels.shape, prelabels)
            final_labels = apt_labels[prelabels]
            print('final labels', final_labels.shape, final_labels)
            final_labels = list(final_labels)
            write_list_to_file(['label'] + final_labels, '/home/shobi/Thesis/Data/CyTOF/apt60_' + str(
                too_big_factor_i) + file_str + str_data_set + '.txt')
            print('SAVED labels for ', str_data_set)
            df_accuracy_mst_lv, best_labels_mst_lv_30, sigma_opt_lv, min_clustersize_mst_lv, tooclose_factor_opt, onevsall_opt_mst_lv, majority_truth_labels_mst_lv = LC.multiclass_mst_accuracy(
                X_LV_embedded, true_label, df_all=None,av_peaks=30, too_big_factor = too_big_factor_i, X_data_ndim=X_data,inputGraph=True, original_data = codebook)
            apt_labels = np.asarray(best_labels_mst_lv_30).T
            print('shape of apt 30 labels', apt_labels.shape)
            print('shape of prelabels', prelabels.shape, prelabels)
            final_labels = apt_labels[prelabels]
            print('final labels', final_labels.shape, final_labels)
            final_labels = list(final_labels)
            write_list_to_file(['label'] + final_labels, '/home/shobi/Thesis/Data/CyTOF/apt30_' + str(
                too_big_factor_i) + file_str + str_data_set + '.txt')
            write_list_to_file(['label']+final_labels, '/home/shobi/Thesis/Data/CyTOF/apt30_'+str(too_big_factor_i)+file_str+str_data_set+'.txt')
            print('SAVED labels for ', str_data_set)
            df_accuracy_mst_lv, best_labels_mst_lv_45, sigma_opt_lv, min_clustersize_mst_lv, tooclose_factor_opt, onevsall_opt_mst_lv, majority_truth_labels_mst_lv = LC.multiclass_mst_accuracy(
                X_LV_embedded, true_label, df_all=None, av_peaks=45, too_big_factor=too_big_factor_i, X_data_ndim=X_data, inputGraph=True, original_data = codebook)
            write_list_to_file(['label'] + best_labels_mst_lv_45, '/home/shobi/Thesis/Data/CyTOF/apt45_' + str(
                too_big_factor_i) + file_str + str_data_set + '.txt')
            print('SAVED labels for ', str_data_set)
            apt_labels = np.asarray(best_labels_mst_lv_45).T
            print('shape of apt 45 labels', apt_labels.shape)
            print('shape of prelabels', prelabels.shape, prelabels)
            final_labels = apt_labels[prelabels]
            print('final labels', final_labels.shape, final_labels)
            final_labels = list(final_labels)
            write_list_to_file(['label'] + final_labels, '/home/shobi/Thesis/Data/CyTOF/apt45_' + str(
                too_big_factor_i) + file_str + str_data_set + '.txt')
            print('SAVED labels for 45', str_data_set)
            df_accuracy_mst_lv, best_labels_mst_lv_75, sigma_opt_lv, min_clustersize_mst_lv, tooclose_factor_opt, onevsall_opt_mst_lv, majority_truth_labels_mst_lv = LC.multiclass_mst_accuracy(
                X_LV_embedded, true_label, df_all=None,av_peaks=75, too_big_factor = too_big_factor_i, X_data_ndim=X_data,inputGraph=True, original_data = codebook)
            #write_list_to_file(['label']+best_labels_mst_lv_75, '/home/shobi/Thesis/Data/CyTOF/apt75_'+str(too_big_factor_i)+file_str+str_data_set+'.txt')
            #print('SAVED labels for ', str_data_set)

            apt_labels = np.asarray(best_labels_mst_lv_75).T
            print('shape of apt 75 labels', apt_labels.shape)
            print('shape of prelabels', prelabels.shape, prelabels)
            final_labels = apt_labels[prelabels]
            print('final labels', final_labels.shape, final_labels)
            final_labels = list(final_labels)
            write_list_to_file(['label'] + final_labels, '/home/shobi/Thesis/Data/CyTOF/apt75_' + str(
                too_big_factor_i) + file_str + str_data_set + '.txt')
            print('SAVED labels for 75', str_data_set)
            df_accuracy_mst_lv, best_labels_mst_lv_20, sigma_opt_lv, min_clustersize_mst_lv, tooclose_factor_opt, onevsall_opt_mst_lv, majority_truth_labels_mst_lv = LC.multiclass_mst_accuracy(
                X_LV_embedded, true_label, df_all=None, av_peaks=20, too_big_factor=too_big_factor_i, X_data_ndim=X_data,inputGraph=True, original_data = codebook)
            apt_labels = np.asarray(best_labels_mst_lv_20).T
            print('shape of apt 20 labels', apt_labels.shape)
            print('shape of prelabels', prelabels.shape, prelabels)
            final_labels = apt_labels[prelabels]
            print('final labels', final_labels.shape, final_labels)
            final_labels = list(final_labels)
            write_list_to_file(['label'] + final_labels, '/home/shobi/Thesis/Data/CyTOF/apt20_' + str(
                too_big_factor_i) + file_str + str_data_set + '.txt')
            print('SAVED labels for 20', str_data_set)

            #write_list_to_file(['label'] + best_labels_mst_lv_20, '/home/shobi/Thesis/Data/CyTOF/apt20_' + str(
            #    too_big_factor_i) + file_str + str_data_set + '.txt')
            #print('SAVED labels for ', str_data_set)
            #df_accuracy_mst_lv, best_labels_mst_lv_auto, sigma_opt_lv, min_clustersize_mst_lv, tooclose_factor_opt, onevsall_opt_mst_lv, majority_truth_labels_mst_lv = LC.multiclass_mst_accuracy(
            #   X_LV_embedded, true_label, df_all=None,av_peaks=0, too_big_factor = too_big_factor_i)
            #write_list_to_file(['label']+best_labels_mst_lv_auto, '/home/shobi/Thesis/Data/CyTOF/aptauto_'+str(too_big_factor_i)+file_str+str_data_set+'.txt')
            #print('SAVED labes for ', str_data_set)
    '''
            #print(time.time())


def main():
    run_main(str_data_set='Mosmann_rare', file_str='nov18')
    #run_main(str_data_set='Nilsson_rare', file_str='nov18')
    #run_main(str_data_set="Samusik_01", file_str='Nov8')
    #run_main(str_data_set="Samusik_all", file_str='')
    #run_main(str_data_set="Levine_13dim", file_str='Nov2')
    #run_main(str_data_set="Levine_32dim",file_str='May20PM_Leidenv2_iter5_fastSmall' )

    #run_main(str_data_set='Mosmann_rare', file_str='oct24')

    #run_main(str_data_set="Mosmann_rare",file_str='jac_figure' )

def main1():

    compute_ARI(str_data_set='Levine_13dim', labelfilename = 'formatted_Levine_13dim.txt')

def compute_ARI(str_data_set, labelfilename):


    path = "/home/shobi/Thesis/Data/CyTOF/" + str_data_set + ".fcs"

    print('data file is', path)
    meta, data = fcsparser.parse(path, reformat_meta=True)
    data = data.fillna(value=999)

    true_label = data['label']
    print(set(true_label), type(true_label))

    cluster_label = []


    with open('/home/shobi/Thesis/Rcode/FLOCK/'+labelfilename,  'rt') as f:

        next(f)
        for line in f:
            line = line.strip().replace('\"', '')
            cluster_label.append(int(float(line)))
    print('there are', len(set(cluster_label)), 'clusters')

    print('ari for clustering', adjusted_rand_score(np.asarray(true_label), cluster_label))
    print("Adjusted Mutual Information: %0.5f"
          % metrics.adjusted_mutual_info_score(true_label, cluster_label))
    targets = list(set(true_label))
    N = len(true_label)
    f1_mean = 0
    target_range = set(true_label)
    for onevsall_val in target_range:
        vals_roc, predict_class_array, maj, numclusters_targetval = ls.accuracy_mst(cluster_label, true_label,
                                                        embedding_filename=None, clustering_algo='louvain',
                                                        onevsall=onevsall_val)
        f1_current = vals_roc[1]
        print('for target', onevsall_val, 'the f1-score is', f1_current)
        f1_mean = f1_current + f1_mean
    print('f1-mean is', f1_mean / len(targets))

if __name__ == '__main__':
    main()
    #have not run the currently typed filenames min20min20maxlabels10
