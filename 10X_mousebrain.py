import scanpy as sc

print(sc.__file__)
import seaborn as sns
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import sys
from matplotlib import colors
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 350
plt.rcParams["font.family"] = "Times New Roman"
#sc.settings.verbosity = 2  # show logging output
#sc.settings.autosave = True  # save figures, do not show them
#sc.settings.set_figure_params(dpi=300)  # set sufficiently high resolution for saving
zeileis_26 = [
    "#023fa5", "#7d87b9", "#bec1d4", "#d6bcc0", "#bb7784", "#8e063b", "#4a6fe3",
    "#8595e1", "#b5bbe3", "#e6afb9", "#e07b91", "#d33f6a", "#11c638", "#8dd593",
    "#c6dec7", "#ead3c6", "#f0b98d", "#ef9708", "#0fcfc0", "#9cded6", "#d5eae7",
    "#f3e1eb", "#f6c4e1", "#f79cd4",
    '#7f7f7f', "#c7c7c7", "#1CE6FF", "#336600"]


# from http://godsnotwheregodsnot.blogspot.de/2012/09/color-distribution-methodology.html
godsnot_64 = [
    # "#000000",  # remove the black, as often, we have black colored annotation
    "#FFFF00", "#1CE6FF", "#FF34FF", "#FF4A46", "#008941", "#006FA6", "#A30059",
    "#FFDBE5", "#7A4900", "#0000A6", "#63FFAC", "#B79762", "#004D43", "#8FB0FF", "#997D87",
    "#5A0007", "#809693", "#FEFFE6", "#1B4400", "#4FC601", "#3B5DFF", "#4A3B53", "#FF2F80",
    "#61615A", "#BA0900", "#6B7900", "#00C2A0", "#FFAA92", "#FF90C9", "#B903AA", "#D16100",
    "#DDEFFF", "#000035", "#7B4F4B", "#A1C299", "#300018", "#0AA6D8", "#013349", "#00846F",
    "#372101", "#FFB500", "#C2FFED", "#A079BF", "#CC0744", "#C0B9B2", "#C2FF99", "#001E09",
    "#00489C", "#6F0062", "#0CBD66", "#EEC3FF", "#456D75", "#B77B68", "#7A87A1", "#788D66",
    "#885578", "#FAD09F", "#FF8A9A", "#D157A0", "#BEC459", "#456648", "#0086ED", "#886F4C",
    "#34362D", "#B4A8BD", "#00A6AA", "#452C2C", "#636375", "#A3C8C9", "#FF913F", "#938A81",
    "#575329", "#00FECF", "#B05B6F", "#8CD0FF", "#3B9700", "#04F757", "#C8A1A1", "#1E6E00",
    "#7900D7", "#A77500", "#6367A9", "#A05837", "#6B002C", "#772600", "#D790FF", "#9B9700",
    "#549E79", "#FFF69F", "#201625", "#72418F", "#BC23FF", "#99ADC0", "#3A2465", "#922329",
    "#5B4534", "#FDE8DC", "#404E55", "#0089A3", "#CB7E98", "#A4E804", "#324E72", "#6A3A4C"]

# https://stackoverflow.com/questions/33295120/how-to-generate-gif-256-colors-palette

godsnot_pink =['#B88183', '#922329', '#5A0007', '#D86A78', '#FF8A9A', '#E20027', '#943A4D', '#B05B6F', '#FEB2C6', '#D83D66', '#FF1A59', '#FFDBE5', '#CC0744', '#CB7E98', '#FF2F80', '#6B002C', '#A74571', '#C6005A', '#FF5DA7', '#FF90C9', '#A30059', '#DA007C', '#D157A0', '#DDB6D0', '#962B75', '#A97399', '#D20096', '#E773CE', '#AA5199', '#E704C4', '#6B3A64', '#FFA0F2', '#6F0062', '#B903AA', '#FF34FF']

godsnot_bluegreen =['#3B5DFF', '#C8D0F6', '#6D80BA', '#0045D2', '#00489C', '#0060CD', '#012C58', '#0086ED', '#0AA3F7', '#006FA6', '#0089A3', '#1CE6FF', '#00A6AA', '#00C6C8', '#006A66', '#518A87', '#66E1D3', '#004D43', '#15A08A', '#00C2A0', '#02684E', '#C2FFED', '#47675D', '#8ADBB4', '#0CBD66', '#549E79', '#6C8F7D', '#63FFAC', '#1BE177', '#B5D6C3', '#3D4F44', '#4B8160', '#66796D', '#71BB8C', '#04F757', '#001E09', '#D2DCD5', '#00B433', '#9FB2A4', '#003109', '#A3F3AB', '#456648']

godsnot_yellow = ['#FFFF00', '#FFF69F', '#F4D749', '#CCAA35', '#513A01', '#FFB500', '#A77500', '#D68E01', '#7A4900', '#372101', '#A45B02', '#E7AB63', '#FAD09F', '#D16100', '#A76F42', '#5B3213', '#CA834E', '#FF913F', '#953F00', '#BE4700', '#772600', '#A05837', '#EA8B66', '#FF6832', '#C86240', '#B77B68', '#FFAA92', '#89412E', '#E83000', '#643127', '#1E0200', '#9C6966', '#BF5650', '#BA0900', '#FF4A46', '#F4ABAA', '#000000', '#452C2C', '#C8A1A1']


DIR = "/home/shobi/Thesis/Paper_writing/10XMouseBrain/"

glutamatergic_genes_global = ['Slc17a7','Slc17a6','Tbr1', 'Eomes']
    # , 'Cux', 'Trh', 'Rapgef3', 'Efr3a', 'Slc17a8', 'Nxph4', 'Mup5', 'Gpr139',
    #                    'Foxp2', 'Pappa2', 'Ctxn3', 'Stac', 'Chrna6', 'Fam84b', 'Car3', 'Osr1', 'Tunar', 'Oprk1',
    #                    'Rxfp2', 'Lemd1',
    #                             'Wnt7b', 'Postn', 'Colq', 'Batf3', 'Stard8', 'Hsd11b1', 'Rspo1', 'Scnn1a', 'Deptor',
    #                    'Rorb', 'Cux2', 'Enpp2', 'Otof', 'Ngb', 'Ptgs2', 'Rorb', 'Ctxn', 'Scnn1a', 'Arf5', 'Hsd11b1','Ucma','Myl4', 'Qrfpr','Bcl6', 'Tph2','Stac','Cdh13','Ddit4l','Col6a1','Mgp','Ly6d','Sla','Car12','Syt17','Prss22','Ctgf']
GABAergic_genes_global = ['Gad1', 'Gad2','Slc32a1']#, 'Ndnf', 'Chat','Htr3a','Nos1']#,'Chodl', 'Nos1','Myh8']#, 'Etv1', 'll1rapl2', 'Myh8', 'Chrna2', 'Tac2']
                   # 'Crhr2', 'Calb2', 'Hpse', 'C1ql3', 'Crh', 'Nts', 'Gabrg1', 'Th', 'Prdm8', 'Calb1', 'Reln',
                   # 'Gpr149', 'Cpne5', 'Vipr2', 'Nkx2.1',
                   # 'Lamp5', 'Pax6', 'Ndnf', 'Egln3', 'Pdlim5', 'Slc35d3', 'Vax1', 'Lhx6', 'Vip', 'Serpinf1', 'Col14a1',
                   # 'Sncg', 'Ptprk', 'Crispld2', 'lgfbp6', 'Gpc3', 'Lmo1', 'Cck', 'Rspo4', 'Cbln4', 'Htr1f',
                   # 'C1ql1', 'ltih5']

non_neuronal_genes_global = ['Olig1', 'Hes1','Aldoc', 'Gja1']#, 'Mog','Opalin','Xdh', 'Myl9', 'Ctss','Pdgfra']# added mog and opalin for 1M ]#'Myl9','Pdgfra','96*Rik','Bgn','Mbp','Mag','Mog', 'Opalin', ]

def write_list_to_file(input_list, filename):
    """Write the list to file."""
    with open(filename, "w") as outfile:
        for entries in input_list:
            outfile.write(str(entries))
            outfile.write("\n")
def basic_analysis2(filename):

    adata= sc.read_h5ad(filename)
    list_genes = adata.var_names
    if 'Slc17a6' in list_genes: print('found Slc17a6')


    #adata = sc.read_10x_h5(filename)
    print('read data', adata.n_vars, adata.n_obs)
    # sc.pp.subsample(adata, fraction=0.9,random_state=10)
    # print('subsample')

    # sc.pp.filter_genes(adata, min_counts=1)
    # print('filtered genes')
    #
    #
    #
    # sc.pp.normalize_per_cell(adata, key_n_counts='n_counts_all') #normalize after filtering
    # print('normalized data', adata.n_vars, adata.n_obs)
    #
    #
    # adata.write('subsampledFilteredNorm_90percent_rand10.h5ad')
    # print('finished writing')


    #sc.pp.filter_genes_dispersion(adata, flavor='cell_ranger', n_top_genes=2000, log=False)
    sc.pp.highly_variable_genes(adata, n_top_genes=1000, flavor='cell_ranger', n_bins=20)
    print('found most var genes')

    most_var_bool = adata.var['highly_variable']
    print('num var genes', sum(most_var_bool))
    most_var_gene_names = adata.var_names[most_var_bool]
    print(type(most_var_gene_names), most_var_gene_names[0:20])
    with open(DIR+"2000_most_var_genes_list_100percent.txt", "w") as output:
        for listitem in most_var_gene_names.tolist():
            output.write('%s\n' % listitem)
    return most_var_gene_names

def make_donut_plot():
    # Make data: I have 3 groups and 7 subgroups
    group_names = ['groupA', 'groupB', 'groupC']
    group_size = [12, 11, 30]
    subgroup_names = ['A.1', 'A.2', 'A.3', 'B.1', 'B.2', 'C.1', 'C.2', 'C.3', 'C.4', 'C.5']
    subgroup_size = [4, 3, 5, 6, 5, 10, 5, 5, 4, 6]

    # Create colors
    a, b, c = [plt.cm.Blues, plt.cm.Reds, plt.cm.Greens]

    # First Ring (outside)
    fig, ax = plt.subplots()
    ax.axis('equal')
    mypie, _ = ax.pie(group_size, radius=1.3, labels=group_names, colors=[a(0.6), b(0.6), c(0.6)])
    plt.setp(mypie, width=0.3, edgecolor='white')

    # Second Ring (Inside)
    mypie2, _ = ax.pie(subgroup_size, radius=1.3 - 0.3, labels=subgroup_names, labeldistance=0.7,
                       colors=[a(0.5), a(0.4), a(0.3), b(0.5), b(0.4), c(0.6), c(0.5), c(0.4), c(0.3), c(0.2)])
    plt.setp(mypie2, width=0.4, edgecolor='white')
    plt.margins(0, 0)

    # show it
    plt.show()


def basic_analysis(filename):

    #sc.logging.print_versions()

    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4985242/ Feb2016 Neuron Taxonomy
    #https: // www.biorxiv.org / content / biorxiv / early / 2017 / 12 / 06 / 229542.full.pdf Dec 2017 Neuron Taxonomy II

    #read in data
    adata = sc.read_10x_h5(filename)
    adata.var_names_make_unique()
    print('original dimensions', adata.n_obs,adata.n_vars)
    #print('subsample')
    #sc.pp.subsample(adata, fraction=0.75)
    #print('dims after subsampling', adata.n_obs, adata.n_vars)


    print('subsample')
    sc.pp.subsample(adata, fraction=0.7,random_state=10)
    print('dims after subsampling',adata.n_obs, adata.n_vars)
    print('filter genes')
    sc.pp.filter_genes(adata, min_counts=1, copy=False, inplace=True)
    print('dims after filtering genes I', adata.n_obs, adata.n_vars)
    # works with 0.8 recipe using around 80% RAM and 50 bins
    print('normalize')
    sc.pp.normalize_per_cell(adata, key_n_counts='n_counts',copy=False)  # normalize with total UMI count per cell

    print('variable')
    sc.pp.highly_variable_genes(adata,n_top_genes=1000, flavor='cell_ranger',n_bins = 20)
    print(adata.n_obs, adata.n_vars)
    print(len(adata.var['highly_variable']))
    most_var_bool = adata.var['highly_variable']
    print(len(most_var_bool),'len of most_var_bool index')
    most_var_gene_names = adata.var_names[most_var_bool]
    print(len(most_var_gene_names), most_var_gene_names[0:10])

    #read in data again
    print('read in data again')
    adata2 = sc.read_10x_h5(filename)
    adata2.var_names_make_unique()

    adata2 = adata2[:, most_var_gene_names]
    print('dims of data2', adata2.n_obs, adata2.n_vars)
    print('normalize')
    sc.pp.normalize_per_cell(  # normalize with total UMI count per cell
        adata2, key_n_counts='n_counts_all', copy=False)
    print(adata2.var_names[0:10])
    sc.pp.normalize_per_cell(adata2)
    sc.pp.log1p(adata2)
    sc.pp.scale(adata2)

    #run pca
    time_pca_start = time.time()
    print('run pca')
    sc.pp.pca(adata2, n_comps = 50, svd_solver = 'auto', random_state=0) #used for paper

    print('pca time seconds elapsed', time.time()-time_pca_start)
    print('dims of X_pca', adata2.obsm['X_pca'].shape)


    #alph
    print('start alph')
    import Louvain_igraph_Jac24Sept as alph
    true_label = ['Glutamatergic']*(adata2.n_obs-1000) +['GABA']*1000#list(df["label"])
    #adata2.obsm['X_pca'] if using 50 PCA only in ALPH

    predict_class_aggregate, df_accuracy, alph_labels, knn_opt, onevsall_opt, majority_truth_labels, time_end_knn, time_end_prune, time_end_louvain, time_end_total, f1_accumulated, f1_mean, time_end_knn_query, num_edges = alph.run_mainlouvain(adata2.obsm['X_pca'], true_label, small_pop=500, dist_std=2, jac_std=0.15 )
    print(len(alph_labels), 'len of alph labels.', adata2.n_obs, type(((alph_labels))))
    adata2.obs["alph"] = pd.Categorical(alph_labels)

    #sc.pp.recipe_zheng17(adata)
    print('completed alph', time.ctime())
    population_dict = {}
    pop_list = []
    for key in set(alph_labels):
        pop_key = alph_labels.count(key)
        population_dict.update({key: pop_key})
        pop_list.append(pop_key)
    print('pop_dict',population_dict)

    axs = sc.pl.rank_genes_groups_matrixplot(adata2, n_genes=3, dendrogram=False)
    from FItSNE import fast_tsne
    Z = fast_tsne.fast_tsne(adata2.obsm['X_pca'], learning_rate=10000, perplexity=20)
    import matplotlib.pyplot as plt
    plt.figure()
    plt.scatter(Z[:, 0], Z[:, 1], c=alph_labels, s=1, cmap='gist_ncar')
    plt.tight_layout()
    plt.show()


def analysis_onVarGenes(list_of_most_var_genes, filename, extension = 'h5ad', extra_genes=[], random_identifier = 1):

    if extension =='h5ad': adata2 = sc.read_h5ad(filename)
    else: adata2 = sc.read_10x_h5(filename)

    print('loaded data',extension,' file with dimensions',  adata2.n_obs,adata2.n_vars)

    key_genes = []
    for item in extra_genes:
        if item not in adata2.var_names: print(item,' not in adata2')
        else:
            key_genes.append(item)
            if item not in list_of_most_var_genes:
                list_of_most_var_genes.append(item)
                print('added ',item, ' to list of most var genes')

    temp_keep = []
    for item in GABAergic_genes_global:
        if item in adata2.var_names:
            temp_keep.append(item)
            if item not in list_of_most_var_genes:
                list_of_most_var_genes.append(item)
                print('added ', item)
        else: print(item,'not in var_names')
    GABAergic_genes = temp_keep



    temp_keep = []
    for item in glutamatergic_genes_global:
        if item in adata2.var_names:
            temp_keep.append(item)
            if item not in list_of_most_var_genes:
                list_of_most_var_genes.append(item)
                print('added ', item)
        else:print(item, 'not in var_names')
    glutamatergic_genes = temp_keep

    temp_keep = []
    for item in non_neuronal_genes_global:
        if item in adata2.var_names:
            temp_keep.append(item)
            if item not in list_of_most_var_genes:
                list_of_most_var_genes.append(item)
                print('added ', item)
        else: print(item, 'not in var_names')
    non_neuronal_genes = temp_keep

    adata2 = adata2[:, list_of_most_var_genes]

    print('dims of data2', adata2.n_obs, adata2.n_vars)
    print('normalize')
    sc.pp.normalize_per_cell(  # normalize with total UMI count per cell (you would have noramlized once before selecting variable genes. now you normalize again
        adata2, key_n_counts='n_counts_all', copy=False)
    sc.pp.log1p(adata2)
    sc.pp.scale(adata2)

    #run pca
    time_start = time.time()
    print('run pca')
    #for fraction_i in [0.05]:
    for n_obs in [1000000,1300000]:
        n_comps = 20#50
        #print('run pca at subsampling of', fraction_i)
        #adata_subsample = sc.pp.subsample(adata2, fraction=fraction_i, n_obs=None, random_state=0, copy=True) #testing
        adata_subsample = sc.pp.subsample(adata2, n_obs=n_obs, random_state=0, copy=True)  # testing
        #sc.pp.pca(adata2, n_comps = n_comps, svd_solver = 'auto', random_state=0) #original

        sc.pp.pca(adata_subsample, n_comps=n_comps, svd_solver='auto', random_state=0)  # testing
        n_samples = int(0.001*adata_subsample.obsm['X_pca'].shape[0])
        np.savetxt(DIR + 'datamatrixN'+str(n_samples)+"K"+str(n_comps)+'PC.txt', adata_subsample.obsm['X_pca'], delimiter=',', fmt='%f')
        time_elapsed = time.time() - time_start
        print('time elapsed PCA',round(time_elapsed),'seconds. dims of X_pca', adata_subsample.obsm['X_pca'].shape)
        #print('time elapsed PCA', round(time_elapsed), 'seconds. dims of X_pca', adata_subsample.obsm['X_pca'].shape)
        #sc.pp.neighbors(adata_subsample) #testing

    #alph

    import Louvain_igraph_Jac24Sept as alph
    true_label = ['Glutamatergic']*(adata2.n_obs-1000) +['GABA']*1000#list(df["label"])
    print('start alph', time.asctime())
    time_alph_start = time.time()
    jac_std = 'median'
    small_pop = 20
    dist_std = 2
    keep_all =True
    weighted = False
    print('jac', jac_std)
    print('small pop',small_pop)
    print('dist_std', dist_std)
    print('keep all dist edges', keep_all)
    print('weighted edges',weighted)
    predict_class_aggregate, df_accuracy, alph_labels, knn_opt, onevsall_opt, majority_truth_labels, time_end_knn, time_end_prune, time_end_louvain, time_end_total, f1_accumulated, f1_mean, time_end_knn_query, num_edges = alph.run_mainlouvain(adata2.obsm['X_pca'], true_label, small_pop=small_pop, dist_std=dist_std, jac_std=jac_std, keep_all_dist=keep_all,jac_weighted_edges=weighted)# clusters in analysis: small_pop = 300, keep_all_dist = True
    print('time elapsed alph seconds', round(time.time() - time_alph_start))
    write_list_to_file(alph_labels, DIR+'alph_labels_N_May28_dis2jacp15'+str(int(round(adata2.n_obs)/1000))+'K.txt')
    print(len(alph_labels), 'len of alph labels.', adata2.n_obs, type(((alph_labels))))

    #
    # alph_labels = []
    # file = 'alph_labels_N_May28_dis2jacp151306K.txt'
    # with open(DIR+ file,'rt') as f:
    #     for line in f:
    #         line = line.strip().replace('\"', '')
    #         alph_labels.append(int(float(line)))
    # print('there are',len(set(alph_labels)), 'clusters')


    alph_labels_str = [str(i) for i in alph_labels]
    adata2.obs["alph"] = pd.Categorical(alph_labels)
    adata2.obs["alph_str"] = pd.Categorical(alph_labels_str)

    print('completed alph', time.asctime())

    df_temp = pd.DataFrame({"alph": alph_labels})
    population_dict = {}
    pop_list = []
    for key in set(alph_labels):
        pop_key = alph_labels.count(key)
        population_dict.update({key: pop_key})
        pop_list.append(pop_key)
    print('pop_dict',population_dict)
    n_clus = len(set(alph_labels))
    top_n = 2  # top 2 genes per cluster


    final_list_plot_genes =  union(union(union(GABAergic_genes, glutamatergic_genes), non_neuronal_genes), key_genes)

    temp_list= []
    for item in final_list_plot_genes:
        if item in adata2.var_names: temp_list.append(item)
    final_list_plot_genes = temp_list
    print(len(final_list_plot_genes), final_list_plot_genes)

    marker_dict = {'useful_genes': final_list_plot_genes}
    marker_dict_sparse = {'useful_genes': union(union(GABAergic_genes,glutamatergic_genes),non_neuronal_genes)}
    print('marker dict', marker_dict)
    df_average_exp = marker_gene_expression(adata2,marker_dict,partition_key='alph')

    df_average_exp['population'] = pop_list
    df_average_exp['mean_non_neuronal_genes'] = df_average_exp[non_neuronal_genes].max(axis=1)
    df_average_exp['mean_neuronal_genes'] = df_average_exp[GABAergic_genes + glutamatergic_genes].max(axis=1)
    df_average_exp['mean_GABAergic_genes'] = df_average_exp[GABAergic_genes].max(axis=1)
    df_average_exp['mean_non_GABAergic_genes'] = df_average_exp[non_neuronal_genes + glutamatergic_genes].max(
        axis=1)
    df_average_exp['mean_glutamatergic_genes'] = df_average_exp[glutamatergic_genes].max(axis=1)
    df_average_exp['mean_non_glutamatergic_genes'] = df_average_exp[non_neuronal_genes + GABAergic_genes].max(
        axis=1)
    conditions = [
        (df_average_exp['mean_GABAergic_genes'] > df_average_exp['mean_non_GABAergic_genes']),
        df_average_exp['mean_glutamatergic_genes'] > df_average_exp['mean_non_glutamatergic_genes']]
    choices = ['GABA', 'GLU']
    df_average_exp['cluster_type'] = np.select(conditions, choices, default='NN')


    df_temp['cluster_type'] = "" #create a column in df_temp which states major cell type of each cell
    df_temp['alph_newname'] = ""# create a column where the label will be GABA+label (or Glu+label...)

    print('cluster type',df_average_exp[['cluster_type','population']])
    print('using .max()', glutamatergic_genes, GABAergic_genes, non_neuronal_genes)
    print('GLUTA pop', df_average_exp.loc[df_average_exp['cluster_type'] == 'GLU', 'population'].sum())
    print('GABA pop', df_average_exp.loc[df_average_exp['cluster_type'] == 'GABA', 'population'].sum())
    print('Non-Neuronal pop', df_average_exp.loc[df_average_exp['cluster_type'] == 'NN', 'population'].sum())
    g=0
    new_palette = []
    for major_type_i in ['GLU','GABA','NN']:
        major_type_bool = np.asarray(df_average_exp[['cluster_type']]) ==major_type_i
        loc_major_type_i= list(np.where(major_type_bool)[0])
        num_clus_thistype = len(loc_major_type_i)
        if major_type_i == 'GLU': colors_temp = godsnot_bluegreen
        if major_type_i == 'GABA': colors_temp = godsnot_pink
        if major_type_i == 'NN': colors_temp = godsnot_yellow
        new_palette = new_palette+list(colors_temp[0:num_clus_thistype])
        print('loc_major_type_i', loc_major_type_i)
        for label_i in loc_major_type_i:
            df_temp.loc[df_temp.alph == label_i,'cluster_type'] = major_type_i
            df_temp.loc[df_temp.alph ==label_i,'alph_newname'] = g
            g=g+1
    print('new palette', new_palette)
    alph_new_labels = list(df_temp['alph_newname'])
    adata2.obs["alph_newname"] = pd.Categorical(alph_new_labels)
    alph_new_labels_str = [str(i) for i in alph_new_labels]
    adata2.obs["alph_new_labels_str"] = pd.Categorical(alph_new_labels_str)

    pop_list = []
    population_dict = {}
    for key in set(alph_labels):
        pop_key = alph_new_labels.count(key)
        population_dict.update({key: pop_key})
        pop_list.append(pop_key)
    print('pop_dict', population_dict)

    df_average_exp1 = marker_gene_expression(adata2,marker_dict,partition_key='alph_newname')
    df_average_exp1['population'] = pop_list
    df_average_exp1['mean_non_neuronal_genes'] = df_average_exp1[non_neuronal_genes].max(axis=1)
    df_average_exp1['mean_neuronal_genes'] = df_average_exp1[GABAergic_genes + glutamatergic_genes].max(axis=1)
    df_average_exp1['mean_GABAergic_genes'] = df_average_exp1[GABAergic_genes].max(axis=1)
    df_average_exp1['mean_non_GABAergic_genes'] = df_average_exp1[non_neuronal_genes + glutamatergic_genes].max(
        axis=1)
    df_average_exp1['mean_glutamatergic_genes'] = df_average_exp1[glutamatergic_genes].max(axis=1)
    df_average_exp1['mean_non_glutamatergic_genes'] = df_average_exp1[non_neuronal_genes + GABAergic_genes].max(
        axis=1)
    conditions = [
        (df_average_exp1['mean_GABAergic_genes'] > df_average_exp1['mean_non_GABAergic_genes']),
        df_average_exp1['mean_glutamatergic_genes'] > df_average_exp1['mean_non_glutamatergic_genes']]
    choices = ['GABA', 'GLU']
    df_average_exp1['cluster_type'] = np.select(conditions, choices, default='NN')



    print('cluster type',df_average_exp1[['cluster_type','population']])
    print('using .max()', glutamatergic_genes, GABAergic_genes, non_neuronal_genes)
    print('GLUTA pop', df_average_exp1.loc[df_average_exp1['cluster_type'] == 'GLU', 'population'].sum())
    print('GABA pop', df_average_exp1.loc[df_average_exp1['cluster_type'] == 'GABA', 'population'].sum())
    print('Non-Neuronal pop', df_average_exp1.loc[df_average_exp1['cluster_type'] == 'NN', 'population'].sum())
    print('columns of df_avg_exp1', df_average_exp1.columns.tolist())

    #print('final',df_average_exp)

    #Rank genes
    # sc.tl.rank_genes_groups(adata2, 'alph', method='wilcoxon')
    # print('finished ranking')
    # rank_genes_list = []
    #
    # for i in range(top_n):
    #     temp = list(adata2.uns['rank_genes_groups']['names'][i]) #ith row corresponds to ith ranked gene for each group. so there are n_clus columns
    #
    #     rank_genes_list = rank_genes_list+temp
    # final_list_plot_genes = union(union(union(union(GABAergic_genes,glutamatergic_genes),non_neuronal_genes), key_genes),rank_genes_list)

    '''
    adata2.uns["alph_colors"] = {}
    for i in range(len(set(alph_labels))):
        adata2.uns["alph_colors"].update({i:zeileis_26[i]})
    print(adata2.uns["alph_colors"])
    '''

    if "alph_colors" not in adata2.uns:
        _set_default_colors_for_categorical_obs(adata2, 'alph_newname',new_palette)


    cmap_new = colors.LinearSegmentedColormap.from_list(
        'cmap_64', new_palette, n_clus) #godsnot[0:64]



    print(time.asctime())
    final_list_plot_genes_custom = ['Stmn2','Snap25','Tbr1','Eomes',
                                    'Slc17a6', 'Slc17a7','Gad2','Gad1', 'Slc32a1', 'Dlx6os1','Htr3a' ,'Hes1', 'Aldoc','Gfap','Gja1','Sst','Olig1',"Calcr",'Calb1'] #'Gfap','Cd24a','Sox11','Igfbpl1','Mt3','Dlx','Sp9','Sst', 'Olig1','Reln'
    temp_list= []
    for item in final_list_plot_genes_custom:
        if item in adata2.var_names: temp_list.append(item)
        else: print(item, 'not in var names')
    final_list_plot_genes_custom = temp_list
    print('custom plot genes list', len(final_list_plot_genes_custom), final_list_plot_genes_custom)

    ##shobi_heatmap(df_average_exp, final_list_plot_genes, alph_labels,godsnot_64[0:n_clus], population_dict, random_identifier)
    ##plt.show()

    shobi_heatmap(df_average_exp1, final_list_plot_genes_custom, alph_new_labels, new_palette, population_dict, random_identifier) #godsnot[0:64]
    #nowplt.show()
    adata_subsample = sc.pp.subsample(adata2, fraction=0.01, n_obs=None, random_state=0, copy=True)
    #adata_subsample.X = np.clip(adata_subsample.X,-1,1)
    #now ax = sc.pl.heatmap(adata_subsample, final_list_plot_genes_custom, groupby='alph_newname', vmin=-.6,vmax=1)
    #now plt.show()
    #now ax = sc.pl.tracksplot(adata_subsample, final_list_plot_genes_custom, groupby='alph_newname')
    #now plt.show()
    #now ax = sc.pl.stacked_violin(adata2, final_list_plot_genes_custom, row_palette=new_palette, groupby='alph_newname', swap_axes=False, save=str(rand_identifier) + '_violin.tif') #godsnot_64[0:n_clus]
    #now plt.show()

    '''
    for cluster_type_i in ["GLU", "GABA", "NN"]:
        violin_list_genes = []
        bool_cluster_type = df_temp['cluster_type'] ==cluster_type_i
        print('bool clus type',df_average_exp1.index[df_average_exp1['cluster_type'] ==cluster_type_i].tolist())
        group_index = df_average_exp1.index[df_average_exp1['cluster_type'] == cluster_type_i].tolist()
        for j in range(top_n):
            for major_celltype_loc in group_index:
                print(j)
                violin_list_genes.append(adata2.uns['rank_genes_groups']['names'][j][major_celltype_loc])
            print('violin list genes',violin_list_genes)
        if cluster_type_i =='GLU': violin_list_genes = union(violin_list_genes,glutamatergic_genes)
        if cluster_type_i == 'GABA': violin_list_genes = union(violin_list_genes, GABAergic_genes)
        if cluster_type_i == 'NN': violin_list_genes = union(violin_list_genes, non_neuronal_genes)
        row_pallete = np.asarray(godsnot_64)[group_index].tolist()
        print('order',[str(i) for i in group_index])
        print(adata2[bool_cluster_type, :].shape)
        #row_palette=row_pallete
        #use 'alph' when using row_palette so that groups are in order
        #use 'alph_str' when swapping the axes as the x-axis labels are strings. if clusters are the y-column, then use integers
        ax=sc.pl.stacked_violin(adata2[bool_cluster_type,:], final_list_plot_genes_custom, row_palette=row_pallete, groupby='alph_newname', swap_axes=False,save = str(rand_identifier)+'_violin.jpg')
        plt.show()
        #ax = sc.pl.stacked_violin(adata2[bool_cluster_type, :], violin_list_genes, groupby='alph_str',swap_axes=True, save=str(rand_identifier) + '_violin.jpg')
        #plt.show()
    '''
    #axs = sc.pl.rank_genes_groups_heatmap(adata2, show_gene_labels=True)#, n_genes=3, vmax=4)

    #axs = sc.pl.rank_genes_groups_matrixplot(adata2, n_genes=3, dendrogram=False)
    print('time before heatmap', time.ctime())


    print(time.ctime())
    print(time.asctime())
    #plt.show()
    X_pca_copy = adata2.obsm['X_pca']
    if adata2.n_obs > 20000:subsample_rate = 10
    if adata2.n_obs == 20000: subsample_rate = 2
    alph_labels_array = np.asarray(alph_new_labels)
    alph_labels_array = np.asarray(alph_new_labels)
    #false_v = np.zeros((len(alph_labels), 1), dtype=bool)
    first_pass = True
    for label_i in set(alph_labels):
        v = alph_labels_array==label_i
        v_where= np.where(v)[0].tolist()
        shuffle = random.sample(v_where, round(len(v_where)/subsample_rate))
        #false_v[shuffle] = 1
        if first_pass == True:
            X_final = X_pca_copy[shuffle]
            label_final_array = alph_labels_array[shuffle]
            first_pass = False
        else:
            X_final = np.concatenate((X_final, X_pca_copy[shuffle]))
            label_final_array = np.concatenate((label_final_array, alph_labels_array[shuffle]))
    print('subsampled dims', X_final.shape, label_final_array.shape )
    '''
    import Performance_phenograph as pp
    Z, lv_embedding_filename, lv_runtime, lv_embedding_plot_title = pp.run_lv(version=None,
                                                                              input_data=X_final,
                                                                              perplexity=30,
                                                                              lr=1,
                                                                              new_file_name='/home/shobi/Thesis/Paper_writing/10XMouseBrain/Figures/LV_20K_v1.txt',
                                                                              new_folder_name=None, outdim=3)

    import Plotting_3D as Plotting_3D


    Plotting_3D.plot10X_mouse(label_final_array.tolist(), majority_labels= df_average_exp['cluster_type'].tolist(), embedding_filename='/home/shobi/Thesis/Paper_writing/10XMouseBrain/Figures/LV_test20k', X_embedded=Z, color_codes = godsnot_64[0:n_clus])



    fig, ax = plt.subplots(1, 1)
    scatter = plt.scatter(Z[:, 0], Z[:, 1], c=label_final_array.tolist(), s=1, cmap=cmap_new)

    color_list = alph_labels
    clset = set(zip(color_list, alph_labels))
    handles = [plt.plot([], color=scatter.get_cmap()(scatter.norm(c)), ls="", marker="o")[0] for c, l in clset]
    labels = [l for c, l in clset]
    ax.legend(handles, labels)
    plt.show()
    #axs = sc.pl.rank_genes_groups_tracksplot(adata2, n_genes=3)
    '''
    from FItSNE import fast_tsne


    fig, ax = plt.subplots(1,1)
    fig1, ax1 = plt.subplots(1, 1)
    if subsample_rate ==1:
        Z = fast_tsne.fast_tsne(adata2.obsm['X_pca'], learning_rate=1000)
        scatter = ax.scatter(Z[:, 0], Z[:, 1], c=alph_new_labels, s=1, cmap=cmap_new,edgecolors= 'none')
        color_list= alph_labels
        clset = set(zip(color_list, alph_labels))
        handles = [plt.plot([], color=scatter.get_cmap()(scatter.norm(c)), ls="", marker="o")[0] for c, l in clset]
        labels = [l for c, l in clset]
        ax.legend(handles, labels)
        plt.show()
    else:
        label_final_list = label_final_array.tolist()
        Z = fast_tsne.fast_tsne(X_final, learning_rate=1000)
        scatter = ax.scatter(Z[:, 0], Z[:, 1], c=label_final_list, s=0.6, cmap=cmap_new,edgecolors= 'none')
        scatter1 = ax1.scatter(Z[:, 0], Z[:, 1], c=label_final_list, s=0.6, cmap=cmap_new, edgecolors='none')
        color_list= label_final_list
        clset = set(zip(color_list, label_final_list))
        handles = [plt.plot([], color=scatter.get_cmap()(scatter.norm(c)), ls="", marker="o")[0] for c, l in clset]
        labels = [l for c, l in clset]
        #ax.legend(handles, labels)
        #handles, labels = ax.get_legend_handles_labels()
        # sort both labels and handles by labels
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))

        ax1.legend(handles, labels,fontsize=4,markerscale=0.5)
        plt.show()

def union(a, b):
    """ return the union of two lists """
    return list(set(a) | set(b))

def _set_default_colors_for_categorical_obs(adata, value_to_plot, new_palette):
    from matplotlib import rcParams
    """
    Sets the adata.uns[value_to_plot + '_colors'] using default color palettes
    Parameters
    ----------
    adata : annData object
    value_to_plot : name of a valid categorical observation
    Returns
    -------
    None
    """


    categories = adata.obs[value_to_plot].cat.categories
    length = len(categories)

    # check if default matplotlib palette has enough colors
    if len(rcParams['axes.prop_cycle'].by_key()['color']) >= length:
        cc = rcParams['axes.prop_cycle']()
        palette = [next(cc)['color'] for _ in range(length)]

    else:
        if length <= 28:
            palette = zeileis_26
        elif length > len(zeileis_26):  # 103 colors
            palette = new_palette
    adata.uns[value_to_plot + '_colors'] = palette[:length]

def marker_gene_expression(anndata, marker_dict, gene_symbol_key=None, partition_key='alph'):
    """A function go get mean z-score expressions of marker genes
    #
    # Inputs:
    #    anndata         - An AnnData object containing the data set and a partition
    #    marker_dict     - A dictionary with cell-type markers. The markers should be stores as anndata.var_names or
    #                      an anndata.var field with the key given by the gene_symbol_key input
    #    gene_symbol_key - The key for the anndata.var field with gene IDs or names that correspond to the marker
    #                      genes
    #    partition_key   - The key for the anndata.obs field where the cluster IDs are stored. The default is
    #                      'louvain_r1' """

    # Test inputs
    if partition_key not in anndata.obs.columns.values:
        print('KeyError: The partition key was not found in the passed AnnData object.')
        print('   Have you done the clustering? If so, please tell pass the cluster IDs with the AnnData object!')


    if (gene_symbol_key != None) and (gene_symbol_key not in anndata.var.columns.values):
        print('KeyError: The provided gene symbol key was not found in the passed AnnData object.')
        print('   Check that your cell type markers are given in a format that your anndata object knows!')


    if gene_symbol_key:
        gene_ids = anndata.var[gene_symbol_key]
    else:
        gene_ids = anndata.var_names

    clusters = anndata.obs[partition_key].cat.categories
    n_clust = len(clusters)
    marker_exp = pd.DataFrame(columns=clusters)
    #marker_exp['marker_type'] = pd.Series({}, dtype='str')
    marker_names = []

    z_scores = sc.pp.scale(anndata, copy=True) #anndata

    i = 0
    for group in marker_dict:
        # Find the corresponding columns and get their mean expression in the cluster
        for gene in marker_dict[group]:
            ens_idx = np.in1d(gene_ids, gene)  # Note there may be multiple mappings

            if np.sum(ens_idx) == 0:
                continue
            else:
                z_scores.obs[ens_idx[0]] = z_scores.X[:, ens_idx].mean(1)  # works for both single and multiple mapping
                ens_idx = ens_idx[0]

            clust_marker_exp = z_scores.obs.groupby(partition_key)[ens_idx].apply(np.mean).tolist()
            #clust_marker_exp.append(group)
            marker_exp.loc[i] = clust_marker_exp
            marker_names.append(gene)
            i += 1

    # Replace the rownames with informative gene symbols
    marker_exp.index = marker_names
    print('marker exp', marker_exp) #THE COLS are the original partition_key and are arrange in increasing order 0,1,2,....
    from sklearn import preprocessing

    marker_exp = marker_exp.T
    x = marker_exp.values  # returns a numpy array
    standard_scaler = preprocessing.RobustScaler()
    x = standard_scaler.fit_transform(x)
    standard_scaler = preprocessing.StandardScaler()
    x_scaled = standard_scaler.fit_transform(x)
    marker_exp = pd.DataFrame(x_scaled, columns = marker_exp.columns) #the index value matches the group cluster number

    return (marker_exp)

def shobi_heatmap(df_org,used_cols,alph_labels,cmap_list, pop_dict,rand_identifier):

    df = df_org.loc[:, used_cols]
    df = df.clip(-1.2, 1.2)
    #used_networks = list(set(alph_labels))
    used_networks = list(df.index.values)
    # Create a custom palette to identify the networks
    network_pal = cmap_list#sns.cubehelix_palette(len(used_networks),
                                        #light=.9, dark=.1, reverse=True,
                                        #start=1, rot=-2)
    print('network pal',network_pal)
    network_lut = dict(zip(map(str, used_networks), network_pal))
    print('network lut',network_lut)

    # Convert the palette to vectors that will be drawn on the side of the matrix
    #network_labels = df.columns.get_level_values("index")
    network_labels =df.index.tolist()
    network_labels=[str(i) for i in network_labels]
    print('network labels', network_labels)
    network_colors = pd.Series(network_labels).map(network_lut)
    print(network_colors)

    # Create a custom colormap for the heatmap values
    cmap_div=sns.diverging_palette(240, 10, as_cmap=True)
    yticklabels = [str(alph_labels_i) +' '+ str(pop_dict[alph_labels_i])+' '+str(df_org.at[int(alph_labels_i),'cluster_type']) for alph_labels_i in used_networks] #in set(alph_labels)
    print("yticklabels",yticklabels)
    # Draw the full plot
    g = sns.clustermap(df,

                      # Turn on/off the clustering
                      row_cluster=True, col_cluster=True,

                      # Add colored class labels
                      row_colors=network_colors,

                      # Make the plot look better when many rows/cols
                      linewidths=0, xticklabels=True, yticklabels=yticklabels, cmap=cmap_div,  robust=True)

    g.ax_row_dendrogram.set_visible(False)
    g.ax_col_dendrogram.set_visible(False)

    # Draw the legend bar for the classes
    #for label in network_labels:
    #    g.ax_col_dendrogram.bar(0, 0, color=network_lut[label],
    #                            label=label, linewidth=0)
    #g.ax_col_dendrogram.legend(loc="center", ncol=5)

    # Adjust the postion of the main colorbar for the heatmap
    g.cax.set_position([.15, .2, .03, .45])
    #plt.subplots_adjust(left=0)

    g.savefig(DIR+"/Figures/clustermap"+str(rand_identifier)+".tif")
if __name__ == '__main__':
    dataset = '1M'
    #basic_analysis2(DIR + '1M_neurons_filtered_gene_bc_matrices_h5.h5')
    #list_of_most_var_genes = basic_analysis2(DIR + 'subsampled20k_70percent_rand10.h5ad')

    #list_of_most_var_genes = basic_analysis2(DIR + 'subsampled20k_70percent_rand10.h5ad')

    #basic_analysis2(DIR + '1M_neurons_neuron20k.h5')
    #list_of_most_var_genes = basic_analysis2(DIR+"subsampledFilteredNorm_90percent_rand10.h5ad")

    #adata = sc.read_10x_h5(DIR+'1M_neurons_neuron20k.h5')
    #list_genes = adata.var_names

    import random
    print('time start loading data', time.asctime())
    rand_identifier = random.randint(1,1000)


    list_of_most_var_genes = []
    if dataset =='1M':
        most_var_genes_file = 'most_var_genes_list_100percent.txt'
        key_genes = ['Calb1', 'Calcr', 'Cck', 'Reln', 'Vip', 'Aqp4', 'Igfbpl1', 'Riiad1', 'Ascl1', 'Top2a', 'Slc17a6',
                     'Slc17a7', 'Tbr1', 'Rorb', 'Lhx6', 'Nr4a2', 'Trp73', 'Mki67', 'Ntf3', 'Syt6', 'Six3', 'Snap25',
                     'Dlx1', 'Sst', 'Pvalb', 'Calb2', 'Nfib', 'Mdk', 'Fabp7', 'Mt3', 'Cd24a', 'Sox11', 'Dlx6os1', 'Sp9',
                     'Gfap', 'Igfbpl1', 'Mt3', 'Aldoc', 'Reln', 'Htr3a', "Stmn2"]
        with open(DIR + most_var_genes_file, 'rt') as f:  # most_var_genes_list_100percent.txt
            for line in f:
                line = line.strip().replace('\"', '')
                list_of_most_var_genes.append(line)
        analysis_onVarGenes(list_of_most_var_genes, DIR + 'FilteredNorm_100percent.h5ad', 'h5ad', extra_genes=key_genes,
                            random_identifier=rand_identifier)
    if dataset == '20K':
        most_var_genes_file = 'most_var_genes_list.txt'
        key_genes = []
        with open(DIR + most_var_genes_file, 'rt') as f:  # most_var_genes_list_100percent.txt
            for line in f:
                line = line.strip().replace('\"', '')
                list_of_most_var_genes.append(line)
        analysis_onVarGenes(list_of_most_var_genes, DIR + '1M_neurons_neuron20k.h5', 'h5', extra_genes=key_genes)

    print('num of most var genes',len(list_of_most_var_genes))


    # for item in key_genes:
    #     if item not in list_of_most_var_genes:
    #         list_of_most_var_genes.append(item)
    #         print(item,'was added to the list')
    #
    #
    # print('num of most var genes',len(list_of_most_var_genes))
    # print(list(set(glutamatergic_genes).intersection(list_of_most_var_genes)),len(set(glutamatergic_genes)))

    #list_of_most_var_genes = basic_analysis2(DIR + 'FilteredNorm_100percent.h5ad')



    #analysis_onVarGenes(list_of_most_var_genes, DIR + '1M_neurons_filtered_gene_bc_matrices_h5.h5')

    #basic_analysis2(DIR + 'FilteredNorm_100percent.h5ad')
    '''
    import Plotting_3D as Plotting_3D
    import random
    for x in range(10):
        print
        random.randint(1, 101)
    rand_X = np.random.rand(1000, 3)
    rand_labels = [random.randint(1,11) for i in range(1000)]
    n_clus = len(set(rand_labels))
    majority_labels = [random.randint(1,4) for i in range(1000)]

    Plotting_3D.plot10X_mouse(rand_labels, majority_labels= majority_labels, embedding_filename='/home/shobi/Thesis/Paper_writing/10XMouseBrain/Figures/LV_testrandom', X_embedded=rand_X, color_codes = godsnot_64[0:n_clus])
    '''