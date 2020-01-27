
from mpl_toolkits.mplot3d.proj3d import proj_transform
from matplotlib.text import Annotation
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 350
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D





'''
# Create some random data, I took this piece from here:
# http://matplotlib.org/mpl_examples/mplot3d/scatter3d_demo.py
def randrange(n, vmin, vmax):
    return (vmax - vmin) * np.random.rand(n) + vmin
n = 100
xx = randrange(n, 23, 32)
yy = randrange(n, 0, 100)
zz = randrange(n, -50, -25)

# Create a figure and a 3D Axes
fig = plt.figure()
ax = Axes3D(fig)

# Create an init function and the animate functions.
# Both are explained in the tutorial. Since we are changing
# the the elevation and azimuth and no objects are really
# changed on the plot we don't have to return anything from
# the init and animate function. (return value is explained
# in the tutorial.
def init():
    ax.scatter(xx, yy, zz, marker='o', s=20, c="goldenrod", alpha=0.6)
    # make the panes transparent
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # make the grid lines transparent
    ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    return fig,

def animate(i):
    ax.view_init(elev=10., azim=i)
    return fig,

# Animate
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=360, interval=20, blit=True)

# Save
anim.save('/home/shobi/Thesis/presentations/basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
plt.savefig('/home/shobi/Thesis/presentations/basic_animationframe.png', bbox_inches='tight')
'''


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
        #0?
def annotate3D(ax, s, *args, **kwargs):
    '''add anotation text s to to Axes3d ax'''

    tag = Annotation3D(s, *args, **kwargs)
    ax.add_artist(tag)



from mpl_toolkits.mplot3d.art3d import Line3DCollection
def plot_graph_3D(fig_graph, ax, edgelist, X,true_label,graph_title, noedges = False, colors = 'different'):
    # X is the matrix nx3 where row i is the x,y,z coords of vertex i

    xn = X[:, 0]
    yn = X[:, 1]
    zn = X[:, 2]
    if noedges == False:
        xyzn = list(zip(xn, yn, zn))
        segments =[(xyzn[s], xyzn[t]) for s, t in edgelist]
        edge_col = Line3DCollection(segments, lw=0.15)#, colors='#6a9cad')
        ax.add_collection3d(edge_col)
    if colors =='uniform':ax.scatter(xn, yn, zn, alpha=0.5, c='#b5cdd6',s=12) #c5edd5  #, edgecolors='#90b5c1'
    else:  ax.scatter(xn, yn, zn, alpha=0.5, c=true_label, s=15) #s=12

    #make panes transparent
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # make the grid lines transparent
    ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.set_xlim(-3,3)
    ax.set_ylim(-3, 3)
    ax.set_zlim(-3, 3)
    plt.show()
    plt.savefig('/home/shobi/Thesis/presentations/image_green'+graph_title+'.jpg')


def animate_graph(i,fig_graph,ax1):
    ax1.view_init(elev=10., azim=i)
    return fig_graph,

def save_anim_graph(edgelist, X,true_label,graph_title = 'graph'):
    fig_graph = plt.figure(figsize=(24, 24))
    ax = fig_graph.add_subplot(1, 1, 1, projection='3d')
    plot_graph_3D(fig_graph, ax, edgelist, X, true_label, graph_title=graph_title + 'colors', noedges=True)
    plot_graph_3D(fig_graph,ax,edgelist,X,true_label,graph_title = graph_title+'nocolors', colors='uniform')


    #ax.autoscale(tight=True)


    #anim = animation.FuncAnimation(fig_graph, animate_graph,  frames=360, fargs=(fig_graph,ax),interval=20, blit=True)
    #anim.save('/home/shobi/Thesis/presentations/animated_'+graph_title+'.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

def test_init(figtest, ax1,ax2):
    # data: coordinates of nodes and links
    xn = [1.1, 1.9, 0.1, 0.3, 1.6, 0.8, 2.3, 1.2, 1.7, 1.0, -0.7, 0.1, 0.1, -0.9, 0.1, -0.1, 2.1, 2.7, 2.6, 2.0]
    yn = [-1.2, -2.0, -1.2, -0.7, -0.4, -2.2, -1.0, -1.3, -1.5, -2.1, -0.7, -0.3, 0.7, -0.0, -0.3, 0.7, 0.7, 0.3, 0.8, 1.2]
    zn = [-1.6, -1.5, -1.3, -2.0, -2.4, -2.1, -1.8, -2.8, -0.5, -0.8, -0.4, -1.1, -1.8, -1.5, 0.1, -0.6, 0.2, -0.1, -0.8, -0.4]
    group = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 2, 2, 2, 3, 3, 3, 3]
    edges = [(1, 0), (2, 0), (3, 0), (3, 2), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0), (9, 0), (11, 10), (11, 3), (11, 2), (11, 0), (12, 11), (13, 11), (14, 11), (15, 11), (17, 16), (18, 16), (18, 17), (19, 16), (19, 17), (19, 18)]
    xyzn = list(zip(xn, yn, zn))
    segments = [(xyzn[s], xyzn[t]) for s, t in edges]


    '''
    def init():
        # plot vertices
        ax.scatter(xn, yn, zn, marker='o', c=group, s=64)
        # plot edges
        edge_col = Line3DCollection(segments, lw=0.2)
        ax.add_collection3d(edge_col)
        # add vertices annotation.
        for j, xyz_ in enumerate(xyzn):
            annotate3D(ax, s=str(j), xyz=xyz_, fontsize=10, xytext=(-3, 3),
                       textcoords='offset points', ha='right', va='bottom')
        return fig,
    '''
    # plot vertices
    ax1.scatter(xn, yn, zn, marker='o', c=group, s=64)
    ax2.scatter(xn, yn, zn, marker='x', c=group, s=64)
    # plot edges
    edge_col = Line3DCollection(segments, lw=0.2)
    edge_col2 = Line3DCollection(segments, lw=0.2)
    ax1.add_collection3d(edge_col)
    ax2.add_collection3d(edge_col2)
    # add vertices annotation.
    for j, xyz_ in enumerate(xyzn):
        annotate3D(ax1, s=str(j), xyz=xyz_, fontsize=10, xytext=(-3, 3),
                   textcoords='offset points', ha='right', va='bottom')
        annotate3D(ax2, s=str(j), xyz=xyz_, fontsize=10, xytext=(-3, 3),
                   textcoords='offset points', ha='right', va='bottom')
    for ax in [ax1,ax2]:
        # make the panes transparent
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        # make the grid lines transparent
        ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
        ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
        ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    return figtest,
def animate_test(i,figtest,ax1,ax2):
    ax1.view_init(elev=10., azim=i)
    ax2.view_init(elev=10., azim=i)
    return figtest,
def save_anim_test():
    figtest = plt.figure(figsize=(36, 12))
    ax1_test = figtest.add_subplot(1, 2, 1, projection='3d')
    ax2_test = figtest.add_subplot(1, 2, 2, projection='3d')
    test_init(figtest,ax1_test,ax2_test)
    plt.show()
    plt.savefig('/home/shobi/Thesis/presentations/basic_animation.png')
    anim = animation.FuncAnimation(figtest, animate_test,  frames=360, fargs=(figtest,ax1_test,ax2_test),interval=20, blit=True)
    anim.save('/home/shobi/Thesis/presentations/basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
    ax1_test.autoscale(tight=True)
    ax2_test.autoscale(tight=True)
#save_anim_test()


def func_mode(ll):  # return MODE of list
    # If multiple items are maximal, the function returns the first one encountered.
    return max(set(ll), key=ll.count)
def plot_all_methods(figt, ax1,ax2, X_embedded, true_label, embedding_filename, dbscan_labels, mst_labels, louvain_labels, pheno_labels, kmeans_labels, onevsall_mst, onevsall_dbscan,onevsall_louvain,onevsall_pheno, onevsall_kmeans, dimred,sigma_opt, eps_opt, min_cluster_opt,dbscan_min_clustersize, knn_opt):

    # if one_vs_all == 'h2170': onevsall = 0
    # if one_vs_all == 'h1975': onevsall =1
    # if one_vs_all == 'h526': onevsall =2
    # if one_vs_all == 'h520': onevsall = 3
    # if one_vs_all == 'h358': onevsall = 4
    # if one_vs_all == 'h69': onevsall = 5
    # if one_vs_all == 'hcc827': onevsall = 6

    X_dict = {}
    X_dict_true = {}
    X_dict_dbscan = {}
    Index_dict = {}
    Index_dict_dbscan = {}

    X_plot = X_embedded

    N = len(true_label)
    if knn_opt==None:
        for k in range(N):
            x = X_plot[k, 0]
            y = X_plot[k, 1]
            z = X_plot[k, 2]
            X_dict.setdefault(mst_labels[k], []).append((x, y,z)) #coordinates of the points by mst groups
            Index_dict.setdefault(mst_labels[k], []).append(true_label[k]) #true label kth data point grouped by mst_group
            X_dict_true.setdefault(true_label[k],[]).append((x,y,z))
    if knn_opt!=None:
        for k in range(N):
            x = X_plot[k, 0]
            y = X_plot[k, 1]
            z = X_plot[k, 2]
            X_dict.setdefault(louvain_labels[k], []).append((x, y,z)) #coordinates of the points by mst groups
            Index_dict.setdefault(louvain_labels[k], []).append(true_label[k]) #true label kth data point grouped by mst_group
            X_dict_true.setdefault(true_label[k],[]).append((x,y,z))

    for true_group in X_dict_true.keys():

        if true_group ==0:
            true_color = 'gray'
            true_label_str = 'h2170'
        if true_group ==1:
            true_color = 'forestgreen'
            true_label_str = 'h1975'
        if true_group ==2:
            true_color = 'orange'
            true_label_str = 'h526'
        if true_group == 3:
            true_color = 'red'
            true_label_str = 'h520'
        if true_group == 4:
            true_color = 'mediumpurple'
            true_label_str = 'h358'
        if true_group == 5:
            true_color = 'deepskyblue'
            true_label_str = 'h69'
        if true_group==6:
            true_color = 'lightpink'
            true_label_str = 'hcc827'
        print('true group', true_group, true_color, true_label_str)
        x = [t[0] for t in X_dict_true[true_group]]
        y = [t[1] for t in X_dict_true[true_group]]
        z = [t[2] for t in X_dict_true[true_group]]
        population = len(x)
        if population > 5000:
            rp = np.random.RandomState(seed=42).permutation(len(x))[0:5000]
            print('downsampling for 3D truth labels')
            x = np.asarray(x)
            y = np.asarray(y)
            z = np.asarray(z)
            listrp = list(rp)
            x = x[listrp]
            y = y[listrp]
            z = z[listrp]
        ax1.scatter(x, y,z, color=true_color, s=2, alpha=0.6, label=true_label_str+' Cellcount = ' + str(population))
        annotate3D(ax1, s=true_label_str, xyz=(np.mean(x), np.mean(y),np.mean(z)), fontsize=10, xytext=(-3, 3),
                   textcoords='offset points', ha='right', va='bottom')
        #ax1.annotate(true_label_str, xytext=(np.mean(x), np.mean(y)), xy=(np.mean(x), np.mean(y)), color='black', weight='semibold')
    #box = ax1.get_position()
    #ax1.set_position([box.x0, box.y0, box.width *0.9, box.height])
    #ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize = 6)
    title_str0 = 'Multi-Class Lung Cancer Cell Lines:Ground Truth. \n'+'Total Cell count is ' +str(N)  # embedding_filename
    ax1.set_title(title_str0, size=10)
    if knn_opt==None:
        ax2= plot_onemethod(ax2,X_embedded,mst_labels, true_label,onevsall_mst, 'mst', 'LargeVis',sigma_opt, min_cluster_opt, None)
    if knn_opt!=None:
        ax2 = plot_onemethod(ax2, X_embedded, louvain_labels, true_label, onevsall_louvain, 'louvain', 'LargeVis', None, None, knn_opt)
    #box1 = ax2.get_position()
    #ax2.set_position([box1.x0, box1.y0, box1.width * 0.9, box1.height])
    #ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=6)
    #ax2[0]= plot_onemethod(ax2[0],X_embedded,dbscan_labels, true_label,onevsall_dbscan, 'dbscan', dimred,eps_opt, dbscan_min_clustersize, None)
    #ax2[1]= plot_onemethod(ax2[1],X_embedded,louvain_labels, true_label,onevsall_louvain, 'louvain', dimred,None, None, knn_opt)
    #ax[2][0]= plot_onemethod(ax[2][0],X_embedded,pheno_labels, true_label,onevsall_pheno, 'phenograph', dimred,None, None, 30)
    #ax[2][1]= plot_onemethod(ax[2][1],X_embedded,kmeans_labels, true_label,onevsall_kmeans, 'kmeans', dimred,None, None, None)



    plt.savefig(embedding_filename + '_allmethods_' + dimred + '.png', bbox_inches='tight')
    return figt,

def plot_onemethod(ax, X_embedded, model_labels, true_labels, onevsall,method, dimred, sigma, min_cluster, knn_opt):

    # if one_vs_all == 'h2170': onevsall = 0
    # if one_vs_all == 'h1975': onevsall =1
    # if one_vs_all == 'h526': onevsall =2
    # if one_vs_all == 'h520': onevsall = 3
    # if one_vs_all == 'h358': onevsall = 4
    # if one_vs_all == 'h69': onevsall = 5
    # if one_vs_all == 'hcc827': onevsall = 6
    if onevsall  == 0: onevsall_opt='h2170'
    if onevsall == 1: onevsall_opt='h1975'
    if onevsall  == 2: onevsall_opt='h526'
    if onevsall == 3:onevsall_opt='h520'
    if onevsall  == 4:onevsall_opt= 'h358'
    if onevsall  == 5:onevsall_opt='h69'
    if onevsall  == 6:onevsall_opt='hcc827'

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
        z = X_plot[k,2]
        X_dict.setdefault(mst_labels[k], []).append((x, y,z)) #coordinates of the points by mst groups
        Index_dict.setdefault(mst_labels[k], []).append(true_labels[k]) #true label kth data point grouped by mst_group
        X_dict_true.setdefault(true_labels[k],[]).append((x,y,z))
    sorted_keys = list(sorted(X_dict.keys()))
    print('in plot: number of distinct groups:', len(sorted_keys))
    # sorted_keys_dbscan =list(sorted(X_dict_dbscan.keys()))
    # print(sorted_keys, ' sorted keys')
    error_count = []
    h2170_labels = []
    h1975_labels = []
    h526_labels = []
    h520_labels = []
    h358_labels = []
    h69_labels = []
    hcc827_labels = []
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

        if (majority_val == 0):
            h2170_labels.append(kk)

        if (majority_val == 1):
            h1975_labels.append(kk)

        if majority_val == 2:
            h526_labels.append(kk)

        if majority_val == 3:
            h520_labels.append(kk)

        if (majority_val == 4):
            h358_labels.append(kk)

        if (majority_val == 5):
            h69_labels.append(kk)

        if majority_val == 6:
            hcc827_labels.append(kk)




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

    colors_h2170 = plt.cm.Greys_r(np.linspace(0.2, 0.6, len(h2170_labels)))
    colors_h1975 = plt.cm.Greens_r(np.linspace(0.2, 0.6, len(h1975_labels)))
    colors_h526 = plt.cm.Wistia_r(np.linspace(0.2, 0.6, len(h526_labels))) #orangey yellows
    colors_h520 = plt.cm.Reds_r(np.linspace(0.2, 0.4, len(h520_labels)))
    colors_h358 = plt.cm.Purples_r(np.linspace(0.2, 0.6, len(h358_labels)))
    colors_h69 = plt.cm.Blues_r(np.linspace(0.2, 0.6, len(h69_labels)))
    colors_hcc827 = plt.cm.spring(np.linspace(0, 0.4, len(hcc827_labels)))

    pair_color_group_list = [(colors_h2170, h2170_labels, ['h2170']*len(h2170_labels)),(colors_h1975,h1975_labels, ['h1975']*len(h1975_labels)),(colors_h526, h526_labels, ['h526']*len(h526_labels)),(colors_h520, h520_labels, ['h520']*len(h520_labels)),(colors_h358, h358_labels, ['h358']*len(h358_labels)),
                             (colors_h69, h69_labels, ['h69'] * len(h69_labels)),(colors_hcc827, hcc827_labels, ['hcc827'] * len(hcc827_labels))]
    for color, group, name in pair_color_group_list:
        for color_m, ll_m, label_m in zip(color, group,name):
            x = [t[0] for t in X_dict[ll_m]]
            y = [t[1] for t in X_dict[ll_m]]
            z = [t[2] for t in X_dict[ll_m]]
            population = len(x)
            if population>5000:
                print('downsampling for 3D image')
                rp = np.random.RandomState(seed=42).permutation(len(x))[0:5000]
                x= np.asarray(x)
                y = np.asarray(y)
                z = np.asarray(z)
                listrp = list(rp)
                x = x[listrp]
                y = y[listrp]
                z = z[listrp]
            ax.scatter(x, y,z, color=color_m, s=2, alpha=0.6, label= label_m+'_' + str(ll_m) + ' Cellcount = ' + str(population))
            annotate3D(ax, s=str(int(ll_m)), xyz=(np.mean(x), np.mean(y), np.mean(z)), fontsize=10, xytext=(-3, 3),
                       textcoords='offset points', ha='right', va='bottom', weight = 'semibold', color = 'black')


    if method == 'mst':
        title_str1 = 'APT: mean + ' + str(sigma) + '-sigma cutoff and min cluster size of: ' + str(
        min_cluster) + '\n' +"Total error rate: {:.1f}".format(total_error_rate * 100) + '%'
    if method == 'louvain':
        title_str1 = 'ALPH visualized on. ' +dimred+ '. \n'+'Total error rate: {:.1f}'.format(total_error_rate * 100) + '%'
    if method == 'phenograph':
        title_str1 = 'Phenograph on 30-NN graph clustering overlaid on. ' + dimred + ' embedding. \n'+'Total error rate: {:.1f}'.format(total_error_rate * 100) + '%'
    if method == 'dbscan': title_str1 = 'DBSCAN on '+ dimred +' embedding .Eps = {:.2f}'.format(sigma) + ' and min cluster size of: ' + str(
        min_cluster) + '\n' + ". Total error rate: " + " {:.2f}".format(total_error_rate * 100) + '%'
    if method == 'kmeans': title_str1 = 'KMEANS on ' + dimred + ' embedding \n.'+ 'Total error rate:  {:.2f}'.format(
        total_error_rate * 100) + '%'

    ax.set_title(title_str1, size=8)
    return ax
    #plt.show()

def animate(i,figt,ax1,ax2):
    ax1.view_init(elev=10., azim=i)
    ax2.view_init(elev=10., azim=i)
    return figt,

def save_anim(method, X_embedded, true_label, embedding_filename, dbscan_labels, mst_labels, louvain_labels, pheno_labels, kmeans_labels,
              onevsall_mst, onevsall_dbscan,onevsall_louvain,onevsall_pheno, onevsall_kmeans, dimred,sigma_opt, eps_opt, min_cluster_opt,dbscan_min_clustersize, knn_opt):
    figt = plt.figure(figsize=(36, 12))
    ax1 = figt.add_subplot(1, 2, 1, projection='3d')
    ax2 = figt.add_subplot(1, 2, 2, projection='3d')
    for ax  in [ax1,ax2]:
        #make panes transparent
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        # make the grid lines transparent
        ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
        ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
        ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
        ax.autoscale(tight=True)

    plot_all_methods(figt, ax1,ax2, X_embedded, true_label, embedding_filename, dbscan_labels, mst_labels, louvain_labels, pheno_labels, kmeans_labels,
                     onevsall_mst, onevsall_dbscan,onevsall_louvain,onevsall_pheno, onevsall_kmeans, dimred,sigma_opt, eps_opt, min_cluster_opt,dbscan_min_clustersize, knn_opt)
    print('saving 3D figure')
    plt.savefig(embedding_filename +'_'+ method+'.png')
    anim = animation.FuncAnimation(figt, animate,  frames=360, fargs=(figt,ax1,ax2),interval=20, blit=True)
    anim.save(embedding_filename+'_'+ method+'.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

########################### PBMC ###################
def plotPBMC_3D(model_labels, true_labels, embedding_filename, sigma, min_cluster, onevsall,X_embedded, method,knn_opt=None):
    print(true_labels)
    # http://nbviewer.jupyter.org/github/jakevdp/mst_clustering/blob/master/MSTClustering.ipynb

    X_dict = {}
    X_dict_true = {}
    Index_dict = {}
    X_plot = X_embedded

    mst_labels = model_labels

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

    error_count = []
    monocytes_labels = [] #0,1,2,3 are the true labels of the subtypes in that order
    tcells_labels = []
    bcells_labels = []
    nkcells_labels = []
    fp = 0
    fn = 0
    tp = 0
    tn = 0
    precision = 0
    recall = 0


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

        if (majority_val == 0):
            monocytes_labels.append(kk)

        if (majority_val == 1):
            tcells_labels.append(kk)

        if majority_val == 2:
            bcells_labels.append(kk)

        if majority_val == 3:
            nkcells_labels.append(kk)


    error_rate = (fp+fn)/(fp+fn+tn+tp)
    total_error = sum(error_count)/(N)
    comp_n_cancer = tp + fp
    comp_n_pbmc = fn + tn

    if comp_n_cancer != 0:
        computed_ratio = comp_n_pbmc / comp_n_cancer
        # print('computed-ratio is:', computed_ratio, ':1' )
    if tp != 0 or fn != 0: recall = tp / (tp + fn)  # ability to find all positives
    if tp != 0 or fp != 0: precision = tp / (tp + fp)  # ability to not misclassify negatives as positives
    if precision != 0 or recall != 0:
        f1_score = precision * recall * 2 / (precision + recall)
        print('f1-score is', f1_score)

    #fig, ax = plt.subplots(1, 2, figsize=(36, 12), sharex=True, sharey=True)
    fig = plt.figure(figsize=(36, 12))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    #segments = model.get_graph_segments(full_graph=True)

    #ax1.scatter(X_plot[:, 0], X_plot[:, 1], c=true_labels, cmap='nipy_spectral_r', zorder=2, alpha=0.5, s=4)
    for true_group in X_dict_true.keys():
        if true_group ==3:
            true_color = 'darkorange'
            true_label = 'nkcell'
            print('inside group nkcell', true_color)
        elif true_group ==0:
            true_color = 'gray'
            true_label = 'mono'
        elif true_group ==1:
            true_color = 'forestgreen'
            true_label = 'tcell'
        else:
            true_color = 'deepskyblue'
            true_label = 'bcell'
        print('true group', true_group, true_color, true_label)
        x = [t[0] for t in X_dict_true[true_group]]
        y = [t[1] for t in X_dict_true[true_group]]
        z = [t[2] for t in X_dict_true[true_group]]
        population = len(x)
        ax1.scatter(x, y, z, color=true_color, s=2, alpha=0.6, label=true_label+' Cellcount = ' + str(population))
        annotate3D(ax1, s=true_label, xyz=(np.mean(x), np.mean(y), np.mean(z)), fontsize=10, xytext=(-3, 3),
                   textcoords='offset points', ha='right', va='bottom', weight = 'semibold')



        # idx = np.where(color_feature < 5* np.std(color_feature))
        # print(idx[0].shape)
        # print('ckeep shape', c_keep.shape)
        # X_keep = X_data_array[idx[0],:]
        # print('xkeep shape', X_keep.shape)
        # print(c_keep.min(), c_keep.max())
        # s= ax[2].scatter(X_keep[:,0], X_keep[:,1], c =c_keep[:,0], s=4, cmap = 'Reds')
        # cb = plt.colorbar(s)

        # lman = LassoManager(ax1, data_lasso)
        # ax1.text(0.95, 0.01, "blue: pbmc", transform=ax[1].transAxes, verticalalignment='bottom', horizontalalignment='right',color='green', fontsize=10)

    colors_monocytes = plt.cm.Greys_r(np.linspace(0.2, 0.6, len(monocytes_labels)))
    colors_tcells = plt.cm.Greens_r(np.linspace(0.2, 0.6, len(tcells_labels)))
    colors_bcells = plt.cm.Blues_r(np.linspace(0.2, 0.6, len(bcells_labels)))
    colors_nkcells = plt.cm.Oranges_r(np.linspace(0.2, 0.6, len(nkcells_labels)))
    pair_color_group_list = [(colors_monocytes, monocytes_labels, ['mono']*len(monocytes_labels)),(colors_tcells, tcells_labels, ['tcells']*len(tcells_labels)),(colors_bcells, bcells_labels, ['bcells']*len(bcells_labels)),(colors_nkcells, nkcells_labels, ['nkcells']*len(nkcells_labels))]
    for color, group, name in pair_color_group_list:
        for color_m, ll_m, label_m in zip(color, group,name):
            x = [t[0] for t in X_dict[ll_m]]
            y = [t[1] for t in X_dict[ll_m]]
            z = [t[2] for t in X_dict[ll_m]]
            population = len(x)
            ax2.scatter(x, y, z, color=color_m, s=2, alpha=0.6, label=label_m + '_' + str(ll_m) + ' Cellcount = ' + str(population))
            #annotate3D(ax2, s=str(ll_m), xyz=(np.mean(x), np.mean(y), np.mean(z)), fontsize=10, xytext=(-3, 3), textcoords='offset points', ha='right', va='bottom', weight = 'semibold')

    #ax2.text(0.95, 0.01, "number of groups: " + " {:.0f}".format(num_groups) , transform=ax2.transAxes,
    #           verticalalignment='bottom', horizontalalignment='right', color='green', fontsize=10)

    #ax2.axis('tight')
    title_str0 = 'Multiclass PBMC subtypes" '#embedding_filename
    if method == 'mst':
        title_str1 = 'MST: mean + ' + str(sigma) + '-sigma cutoff and too_close factor of: ' + str(
        min_cluster) + '\n' + "Error rate is " + " {:.1f}".format(total_error * 100) + '%'
    if method == 'louvain':
        title_str1 = 'Louvain on ' +str(knn_opt)+'-NN graph\n. Error rate is ' + " {:.1f}".format(total_error * 100)
    title_str2 = 'graph layout with cluster populations'

    ax2.set_title(title_str1, size=10)
    ax1.set_title(title_str0, size=10)
    #ax[2].set_title(title_str2, size=12)

    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width *0.9, box.height])
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize = 10, markerscale = 6)
    box1 = ax2.get_position()
    ax2.set_position([box1.x0, box1.y0, box1.width * 0.9, box1.height])
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10, markerscale =6)
    for ax  in [ax1,ax2]:
        #make panes transparent
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        # make the grid lines transparent
        ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
        ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
        ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
        ax.autoscale(tight=True)
    plt.savefig(embedding_filename+'_' +method+ '.png', bbox_inches='tight')
    #plt.show()

def plot10X_mouse(model_labels, majority_labels, embedding_filename, X_embedded,color_codes):

    # http://nbviewer.jupyter.org/github/jakevdp/mst_clustering/blob/master/MSTClustering.ipynb

    X_dict = {}
    X_dict_true = {}
    Index_dict = {}
    X_plot = X_embedded


    N = len(model_labels)

    for k in range(N):
        x = X_plot[k, 0]
        y = X_plot[k, 1]
        z = X_plot[k, 2]
        X_dict.setdefault(model_labels[k], []).append((x, y,z)) #coordinates of the points by mst groups
        Index_dict.setdefault(model_labels[k], []).append(majority_labels[int(model_labels[k])]) #true label kth data point grouped by mst_group
        X_dict_true.setdefault(majority_labels[int(model_labels[k])],[]).append((x,y,z))
    sorted_keys = list(sorted(X_dict.keys()))
    print('in 3D plot: number of distinct groups:', len(sorted_keys))


    fig, ax = plt.subplots(1, 2, figsize=(36, 12), sharex=True, sharey=True)
    #fig = plt.figure(figsize=(24, 12))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    #segments = model.get_graph_segments(full_graph=True)

    #ax1.scatter(X_plot[:, 0], X_plot[:, 1], c=true_labels, cmap='nipy_spectral_r', zorder=2, alpha=0.5, s=4)
    t = 0
    for true_group in X_dict_true.keys():
        true_color = color_codes[t]
        t=t+1
        true_label = str(true_group)
        print('true group', true_group, true_color, true_label)
        x = [t[0] for t in X_dict_true[true_group]]
        y = [t[1] for t in X_dict_true[true_group]]
        z = [t[2] for t in X_dict_true[true_group]]
        population = len(x)
        ax1.scatter(x, y, z, color=true_color, s=2, alpha=0.6, label=true_label+' Cellcount = ' + str(population))
        annotate3D(ax1, s=true_label, xyz=(np.mean(x), np.mean(y), np.mean(z)), fontsize=10, xytext=(-3, 3),
                   textcoords='offset points', ha='right', va='bottom', weight = 'semibold')


    color_lut = dict(zip(list(set(model_labels)), zip(color_codes,majority_labels)))

    for alph_label in color_lut.keys():
            x = [t[0] for t in X_dict[alph_label]]
            y = [t[1] for t in X_dict[alph_label]]
            z = [t[2] for t in X_dict[alph_label]]
            population = len(x)
            ax2.scatter(x, y, z, color=color_lut[alph_label][0], s=2, alpha=0.6, label=str(alph_label) + '_' + str(color_lut[alph_label][1]) + ' Cellcount = ' + str(population))
            #annotate3D(ax2, s=str(ll_m), xyz=(np.mean(x), np.mean(y), np.mean(z)), fontsize=10, xytext=(-3, 3), textcoords='offset points', ha='right', va='bottom', weight = 'semibold')

    #ax2.text(0.95, 0.01, "number of groups: " + " {:.0f}".format(num_groups) , transform=ax2.transAxes,
    #           verticalalignment='bottom', horizontalalignment='right', color='green', fontsize=10)

    #ax2.axis('tight')
    title_str0 = 'Major groups" '#embedding_filename
    title_str1 = 'ALPH clusters'


    ax2.set_title(title_str1, size=10)
    ax1.set_title(title_str0, size=10)
    #ax[2].set_title(title_str2, size=12)

    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width *0.9, box.height])
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize = 10, markerscale = 6)
    box1 = ax2.get_position()
    ax2.set_position([box1.x0, box1.y0, box1.width * 0.9, box1.height])
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10, markerscale =6)
    for ax  in [ax1,ax2]:
        #make panes transparent
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        # make the grid lines transparent
        ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
        ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
        ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
        ax.autoscale(tight=True)
    plt.savefig(embedding_filename+'_' + '.jpg', bbox_inches='tight')
    plt.show()