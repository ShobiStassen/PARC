import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as colors
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 350
import matplotlib.pylab as plt
plt.rcParams["font.family"] = "Times New Roman"
from matplotlib.text import Annotation
from mpl_toolkits.mplot3d.proj3d import proj_transform
from mpl_toolkits.mplot3d import Axes3D
class Annotation3D(Annotation):
    '''Annotate the point xyz with text s'''

    def __init__(self, s, xyz, *args, **kwargs):
        Annotation.__init__(self,s, xy=(0,0), *args, **kwargs)
        self._verts3d = xyz

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.xy=(xs,ys)
        Annotation.draw(self, renderer)

def annotate3D(ax, s, *args, **kwargs):
    '''add anotation text s to to Axes3d ax'''

    tag = Annotation3D(s, *args, **kwargs)
    ax.add_artist(tag)

def func_mode(ll):  # return MODE of list
    # If multiple items are maximal, the function returns the first one encountered.
    return max(set(ll), key=ll.count)


def plot_onemethod(ax, X_embedded, model_labels, true_labels, method, onevsall = 'CD4+/CD45RO+ Memory', GroundTruth=False):

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
        z = X_plot[k, 2]
        X_dict.setdefault(mst_labels[k], []).append((x, y,z)) #coordinates of the points by mst groups
        Index_dict.setdefault(mst_labels[k], []).append(true_labels[k]) #true label kth data point grouped by mst_group
        X_dict_true.setdefault(true_labels[k],[]).append((x,y,z))
    sorted_keys = list(sorted(X_dict.keys()))
    print('in plot: number of distinct groups:', len(sorted_keys))
    # sorted_keys_dbscan =list(sorted(X_dict_dbscan.keys()))
    # print(sorted_keys, ' sorted keys')
    error_count = []
    B_labels = []
    Dendritic_labels = []
    CD8CytoT_labels = []
    CD8Naive_labels = []
    CD4Helper_labels = []
    CD34_labels = []
    CD4CD25_labels = []
    CD4CD45_RA_labels = []
    CD4CD45_RO_labels = []
    Monocyte_labels = []
    NK_labels = []

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
        # print('kk ', kk, 'has length ', len(vals))
        majority_val = func_mode(vals)
        if majority_val == onevsall:
            tp = tp + len([e for e in vals if e == onevsall])
            fp = fp + len([e for e in vals if e != onevsall])
            error_count.append(len([e for e in vals if e != majority_val]))
        else:
            fn = fn + len([e for e in vals if e == onevsall])
            tn = tn + len([e for e in vals if e != onevsall])
            error_count.append(len([e for e in vals if e != majority_val]))

        if (majority_val == 'CD19+ B'):
            B_labels.append(kk)

        if (majority_val == 'Dendritic'):
            Dendritic_labels.append(kk)

        if majority_val == 'CD8+ Cytotoxic T':
            CD8CytoT_labels.append(kk)

        if majority_val == 'CD4+ T Helper2':
            CD4Helper_labels.append(kk)

        if (majority_val == 'CD34+'):
            CD34_labels.append(kk)

        if (majority_val == 'CD4+/CD25 T Reg'):
            CD4CD25_labels.append(kk)

        if majority_val == 'CD4+/CD45RA+/CD25- Naive T':
            CD4CD45_RA_labels.append(kk)
        if majority_val == 'CD4+/CD45RO+ Memory':
            CD4CD45_RO_labels.append(kk)

        if majority_val == 'CD8+/CD45RA+ Naive Cytotoxic':
            CD8Naive_labels.append(kk)
        if majority_val == 'CD56+ NK':
            NK_labels.append(kk)
        if majority_val == 'CD14+ Monocyte':
            Monocyte_labels.append(kk)

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

    colors_B = plt.cm.Greys_r(np.linspace(0.2, 0.6, len(B_labels)))
    colors_Dendritic = plt.cm.PiYG(np.linspace(0.2, 0.6, len(Dendritic_labels)))
    colors_CD8CytoT = plt.cm.Greens(np.linspace(0.2, 0.6, len(CD8CytoT_labels)))
    colors_CD4Helper = plt.cm.PiYG(np.linspace(0.2, 0.6, len(CD4Helper_labels)))
    colors_CD34 = plt.cm.PiYG(np.linspace(0.2, 0.6, len(CD34_labels)))
    colors_CD4CD25 = plt.cm.Reds_r(np.linspace(0.2, 0.4, len(CD4CD25_labels)))
    colors_CD4CD45_RA = plt.cm.Wistia(np.linspace(0.2, 0.4, len(CD4CD45_RA_labels))) #orangey yellow
    colors_CD4CD45_RO = plt.cm.Wistia_r(np.linspace(0.2, 0.4, len(CD4CD45_RO_labels)))  # orangey yellow
    colors_NK = plt.cm.Purples_r(np.linspace(0.2, 0.6, len(NK_labels)))
    colors_Monocyte = plt.cm.Blues_r(np.linspace(0.2, 0.6, len(Monocyte_labels)))
    colors_CD8Naive = plt.cm.spring(np.linspace(0, 0.4, len(CD8Naive_labels)))

    pair_color_group_list = [(colors_B, B_labels, ['CD19+ B']*len(B_labels)),(colors_CD4CD25,CD4CD25_labels, ['CD25 T Reg']*len(CD4CD25_labels)),(colors_Dendritic, Dendritic_labels, ['Dendritic']*len(Dendritic_labels)),(colors_NK, NK_labels, ['CD56+ NK']*len(NK_labels)),(colors_CD4Helper, CD4Helper_labels, ['CD4+ T Helper2']*len(CD4Helper_labels)),
                             (colors_CD34, CD34_labels, ['CD34+'] * len(CD34_labels)),(colors_CD4CD45_RA, CD4CD45_RA_labels, ['CD4+/CD45RA+/CD25- Naive T'] * len(CD4CD45_RA_labels)),(colors_Monocyte, Monocyte_labels, ['CD14+ Monocyte'] * len(Monocyte_labels)),(colors_CD4CD45_RO, CD4CD45_RO_labels, ['CD4+/CD45RO+ Memory'] * len(CD4CD45_RO_labels)),(colors_CD8CytoT, CD8CytoT_labels, ['CD8+ Cytotoxic T'] * len(CD8CytoT_labels)),(colors_CD8Naive,CD8Naive_labels, ['CD8+/CD45RA+ Naive Cytotoxic'] * len(CD8Naive_labels))]
    for color, group, name in pair_color_group_list:
        for color_m, ll_m, label_m in zip(color, group,name):
            x = [t[0] for t in X_dict[ll_m]]
            y = [t[1] for t in X_dict[ll_m]]
            z = [t[2] for t in X_dict[ll_m]]
            population = len(x)
            if GroundTruth==False: ax.scatter(x, y, z, color=color_m, s=2, alpha=0.6,label=label_m + ' '+str(ll_m)+' Cellcount = ' + str(population))
            else: ax.scatter(x, y, z, color=color_m, s=2, alpha=0.6,label=label_m + ' Cellcount = ' + str(population))
            annotate3D(ax, s=label_m, xyz=(np.mean(x), np.mean(y), np.mean(z)), fontsize=6, xytext=(-3, 3),   textcoords='offset points', ha='right', va='bottom')
            #ax.scatter(x, y, color=color_m, s=2, alpha=0.6, label= label_m+'_' + str(ll_m) + ' Cellcount = ' + str(len(x)))
            #ax.annotate(ll_m, xyztext=(np.mean(x), np.mean(y),np.mean(z)), xy=(np.mean(x), np.mean(y)), color='black',
            #               weight='semibold')

    #ax.text(0.95, 0.01, "number of groups: " + " {:.0f}".format(num_groups) + " FP: " + " {:.2f}".format(
    #    fp * 100 / n_pbmc) + "%. FN of " + "{:.2f}".format(fn * 100 / (n_cancer)) + '%', transform=ax.transAxes,
    #           verticalalignment='bottom', horizontalalignment='right', color='green', fontsize=10)

    ax.axis('tight')

    if method == 'APT':
        title_str1 = 'APT' +"Total error rate: {:.1f}".format(total_error_rate * 100) + '%\n'+"number of groups: " + " {:.0f}".format(num_groups)
    if method == 'ALPH':
        title_str1 = 'ALPH \n'+'Total error rate: {:.1f}'.format(total_error_rate * 100) + '%\n'+"number of groups: " + " {:.0f}".format(num_groups)
    if method == 'kmeans': title_str1 = 'Total error rate:  {:.2f}'.format(total_error_rate * 100)+ '%\n'+"number of groups: " + " {:.0f}".format(num_groups)

    ax.set_title(title_str1, size=8)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10,markerscale=10)

    #make panes transparent
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # make the grid lines transparent
    ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.autoscale(tight=True)
    return ax
    #plt.show()
def plot_onemethod_2D(ax, X_embedded, model_labels, true_labels, method, onevsall = 'CD4+/CD45RO+ Memory', GroundTruth=False):

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
    B_labels = []
    Dendritic_labels = []
    CD8CytoT_labels = []
    CD8Naive_labels = []
    CD4Helper_labels = []
    CD34_labels = []
    CD4CD25_labels = []
    CD4CD45_RA_labels = []
    CD4CD45_RO_labels = []
    Monocyte_labels = []
    NK_labels = []

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
        # print('kk ', kk, 'has length ', len(vals))
        majority_val = func_mode(vals)
        if majority_val == onevsall:
            tp = tp + len([e for e in vals if e == onevsall])
            fp = fp + len([e for e in vals if e != onevsall])
            error_count.append(len([e for e in vals if e != majority_val]))
        else:
            fn = fn + len([e for e in vals if e == onevsall])
            tn = tn + len([e for e in vals if e != onevsall])
            error_count.append(len([e for e in vals if e != majority_val]))

        if (majority_val == 'CD19+ B'):
            B_labels.append(kk)

        if (majority_val == 'Dendritic'):
            Dendritic_labels.append(kk)

        if majority_val == 'CD8+ Cytotoxic T':
            CD8CytoT_labels.append(kk)

        if majority_val == 'CD4+ T Helper2':
            CD4Helper_labels.append(kk)

        if (majority_val == 'CD34+'):
            CD34_labels.append(kk)

        if (majority_val == 'CD4+/CD25 T Reg'):
            CD4CD25_labels.append(kk)

        if majority_val == 'CD4+/CD45RA+/CD25- Naive T':
            CD4CD45_RA_labels.append(kk)
        if majority_val == 'CD4+/CD45RO+ Memory':
            CD4CD45_RO_labels.append(kk)

        if majority_val == 'CD8+/CD45RA+ Naive Cytotoxic':
            CD8Naive_labels.append(kk)
        if majority_val == 'CD56+ NK':
            NK_labels.append(kk)
        if majority_val == 'CD14+ Monocyte':
            Monocyte_labels.append(kk)

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

    colors_B = plt.cm.Greys_r(np.linspace(0.2, 0.6, len(B_labels)))
    colors_Dendritic = plt.cm.viridis(np.linspace(0.8,0.9, len(Dendritic_labels)))
    colors_CD8CytoT = plt.cm.Greens(np.linspace(0.4, 0.6, len(CD8CytoT_labels)))
    colors_CD4Helper = plt.cm.PiYG(np.linspace(0.2, 0.6, len(CD4Helper_labels)))
    colors_CD34 = plt.cm.PiYG(np.linspace(0.2, 0.6, len(CD34_labels)))
    colors_CD4CD25 = plt.cm.Reds_r(np.linspace(0.2, 0.4, len(CD4CD25_labels)))
    colors_CD4CD45_RA = plt.cm.Wistia(np.linspace(0.2, 0.4, len(CD4CD45_RA_labels))) #orangey yellow
    colors_CD4CD45_RO = plt.cm.Wistia_r(np.linspace(0.2, 0.4, len(CD4CD45_RO_labels)))  # orangey yellow
    colors_NK = plt.cm.Purples_r(np.linspace(0.2, 0.6, len(NK_labels)))
    colors_Monocyte = plt.cm.Blues_r(np.linspace(0.2, 0.6, len(Monocyte_labels)))
    colors_CD8Naive = plt.cm.spring(np.linspace(0, 0.4, len(CD8Naive_labels)))

    pair_color_group_list = [(colors_B, B_labels, ['CD19+ B'] * len(B_labels)),
                             (colors_CD4CD25, CD4CD25_labels, ['CD25 T-Reg'] * len(CD4CD25_labels)),
                             (colors_Dendritic, Dendritic_labels, ['Dendritic'] * len(Dendritic_labels)),
                             (colors_NK, NK_labels, ['CD56+ NK'] * len(NK_labels)),
                             (colors_CD4Helper, CD4Helper_labels, ['T-Helper2'] * len(CD4Helper_labels)),
                             (colors_CD34, CD34_labels, ['CD34+'] * len(CD34_labels)), (
                             colors_CD4CD45_RA, CD4CD45_RA_labels,
                             ['Naive-T'] * len(CD4CD45_RA_labels)),
                             (colors_Monocyte, Monocyte_labels, ['Monocyte'] * len(Monocyte_labels)),
                             (colors_CD4CD45_RO, CD4CD45_RO_labels, ['CD45RO+ Memory'] * len(CD4CD45_RO_labels)),
                             (colors_CD8CytoT, CD8CytoT_labels, ['Cytotoxic T'] * len(CD8CytoT_labels)), (
                             colors_CD8Naive, CD8Naive_labels, ['Naive Cytotoxic'] * len(CD8Naive_labels))]
    # pair_color_group_list = [(colors_B, B_labels, ['CD19+ B']*len(B_labels)),(colors_CD4CD25,CD4CD25_labels, ['CD4+/CD25 T Reg']*len(CD4CD25_labels)),(colors_Dendritic, Dendritic_labels, ['Dendritic']*len(Dendritic_labels)),(colors_NK, NK_labels, ['CD56+ NK']*len(NK_labels)),(colors_CD4Helper, CD4Helper_labels, ['CD4+ T Helper2']*len(CD4Helper_labels)),
    #                          (colors_CD34, CD34_labels, ['CD34+'] * len(CD34_labels)),(colors_CD4CD45_RA, CD4CD45_RA_labels, ['CD4+/CD45RA+/CD25- Naive T'] * len(CD4CD45_RA_labels)),(colors_Monocyte, Monocyte_labels, ['CD14+ Monocyte'] * len(Monocyte_labels)),(colors_CD4CD45_RO, CD4CD45_RO_labels, ['CD4+/CD45RO+ Memory'] * len(CD4CD45_RO_labels)),(colors_CD8CytoT, CD8CytoT_labels, ['CD8+ Cytotoxic T'] * len(CD8CytoT_labels)),(colors_CD8Naive,CD8Naive_labels, ['CD8+/CD45RA+ Naive Cytotoxic'] * len(CD8Naive_labels))]
    for color, group, name in pair_color_group_list:
        for color_m, ll_m, label_m in zip(color, group,name):
            x = [t[0] for t in X_dict[ll_m]]
            half = int(len(x)/2)
            y = [t[1] for t in X_dict[ll_m]]
            #x = x[0:half]
            #y = y[0:half]
            print('color of group', ll_m, label_m, color_m, colors.to_hex(color_m))
            population = len(x)
            if GroundTruth==False: ax.scatter(x, y, color=color_m, s=0.8, alpha=0.6,label=label_m + ' '+str(ll_m)+' ' + str(population), edgecolors= 'none')
            else: ax.scatter(x, y, color=color_m, s=0.8, alpha=0.6,label=label_m +' ' + str(population),edgecolors= 'none')
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
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8,markerscale=10) #markerscale 10
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
    #plt.show()

def main():
    import pandas as pd
    import seaborn as sns
    import matplotlib as mpl
    mpl.rcParams['figure.dpi'] = 350
    plt.rcParams["font.family"] = "Times New Roman"
    df_0 = pd.read_csv("/home/shobi/Thesis/10x_visuals/April2019/test_leiden1") #has top 2 genes

    print(df_0)
    df = df_0.loc[:, df_0.columns != 'Unnamed: 0']
    df = df.loc[:, df.columns != 'PF4.1']
    print(list(df))
    used_networks = ['Dendritic 13   367', 'Dendritic 11   719','CD14+ Monocyte 9   2083', 'CD14+ Monocyte 10   1853', 'CD19+ B 8   3888', 'CD19+ B 17   296', 'CD34+ 12   534', 'CD25 T Reg 6   4316', 'CD45RA+ Naive T 0   16646','CD45RA+ Naive T 15   334', 'CD45RA+ Naive T 14   365','CD45RO+ Memory 1   9690', 'CD45RO+ Memory 16   323', 'CD56+ NK 3   6477', 'CD56+ NK 5   4605', 'Naive Cytotoxic 2   6641', 'Cytotoxic T 7   4066', 'Cytotoxic T 4   4876']#list(df_0['Unnamed: 0'])
    alph_labels = used_networks
    print('alph labels',alph_labels, len(alph_labels))
    new_pal = ['#bddf26','#7ad151','#1764ab','#94c4df','#404040','#b5b5b5','#de77ae','#bc141a','#faed2d','#ffce0a','#ffe015','#fe9900','#ffb100','#61409b','#b6b6d8','#ff00ff','#4bb062','#d3eecd']#5b9bd5','#9cc4e6','#808080','#e6e6e6','#f4a1f5','#d60637','#f7e94d','#faa520','#ffc000','#8539bf','#b284f1','#f53cf2','#70ad47','#c5e0b4','#de8cf3','#e2f0d9','blue','blue']
    print("used networks",used_networks,)
    # Create a custom palette to identify the networks
    network_pal = new_pal#sns.cubehelix_palette(len(used_networks),     light=.9, dark=.1, reverse=True,       start=1, rot=-2)
    print('network pal',network_pal, len(network_pal))
    network_lut = dict(zip(map(str, used_networks), network_pal))
    print('network lut',network_lut)

    # Convert the palette to vectors that will be drawn on the side of the matrix
    #network_labels = df.columns.get_level_values("index")
    #network_labels =df.index.tolist()
    network_labels=[str(i) for i in used_networks]
    print('network labels', network_labels)
    network_colors = pd.Series(network_labels).map(network_lut)
    print(network_colors)
    sns.set(font_scale=0.5,font = "Times New Roman")#1.4
    cmap_div = sns.diverging_palette(240, 10, as_cmap=True)
    #df = (df - df.mean()) / (df.std())
    #print('mean ',df.mean())
    # Draw the full plot
    g = sns.clustermap(df,

                       # Turn on/off the clustering
                       row_cluster=False, col_cluster=True,

                       # Add colored class labels
                       row_colors=network_colors,

                       # Make the plot look better when many rows/cols
                       linewidths=0, xticklabels=True, yticklabels= alph_labels, cmap=cmap_div,
                       robust=True, vmax=2,    cbar_kws = {"orientation": "horizontal"})

    plt.subplots_adjust(left=-0.2, bottom=0.2)#right=2
    g.cax.set_visible(False)

    g.ax_row_dendrogram.set_visible(False)
    g.ax_col_dendrogram.set_visible(False)

    plt.show()

if __name__ == '__main__':
    main()