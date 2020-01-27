import numpy as np
import matplotlib.pylab as plt
plt.rcParams["font.family"] = "Times New Roman"
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 350
def spike_test_heatmap():
    data_old_louvain_alph_h1975 = np.array([[0.80,	0.86,	0.76,	0.69,	0.80,	0.79,	0.76,	0.68,	0.72,	0.73,	0.75,0.69 ,0.74 ,0.77 ,0.73 ,0.76 ],[0.69	,0.71,	0.67,	0.50,	0.31,	0.28,	0,	0.26,	0.36,	0,	0,0.24 ,0.40 ,0.27 ,0.00 ,0.19 ]])
    data_h358 = np.array([[0.72,0.78,	0.57,0.69,0.54,0,0,0,0,0,0,0,0,0,0,0],[0.72	,0,	0,	0	,0	,0,0,0,0,0,0,0,0,0,0,0]]) #h358 appended zeros to make the same size as h1975

    data_pheno_h1975 = np.array([[18.0 	,24.88 	,57.8,	51.00 	,49.5,	32.64 ,	15.75 ,	32.75 ,	7.83 ,	28.00 ,	35.75 ,	0.00 	,16.25 ,	38.60 ,	32.38 ,	26.43 ,	0,	0	,21.3 ,	15.0 ,	21.5 	,0.0, 	15.3, 	14.3 	,17,	0]])
    #data_alph_leiden_h1975 =np.array([[70.2 ,	65.9 ,	73.4 	,70.0 ,	71.7 ,	72.1 ,	65.3 ,	63.8 	,59.8 ,	57.8 	,61.1 	,59.0 	,62.0 ,	61.5 ,	66.3 ,	67.2 	,60.0 ,	52.7 ,	47.0 ,	56.5 ,	61.5 ,	60.8 	,56.1 	,57.3 	,54.0 ,	51.7 ],[18.0 	,24.88 	,57.8,	51.00 	,49.5,	32.64 ,	15.75 ,	32.75 ,	7.83 ,	28.00 ,	35.75 ,	0.00 	,16.25 ,	38.60 ,	32.38 ,	26.43 ,	0,	0	,21.3 ,	15.0 ,	21.5 	,0.0, 	15.3, 	14.3 	,17,	0]])
    data_alph_leiden_h1975 = np.array([[70.2, 72.1, 62.3 ,68.7,61.5, 60.8 ,56.1 ,57.3 ,54.0, 54.5],
                                       [18.0 ,32.64,  35.75, 26.43,  21.5 ,0.0, 15.3, 14.3 ,17, 0]])
    y_label = ["PARC","Phenograph"]
    import seaborn as sns
    cmap_div = sns.diverging_palette(240, 10, as_cmap=True)
    #x_label = ["5000","6000","7000","8000","9000","10000","11000","12000","13000","14000","15000","16000","17000","18000","19000","20000",]
    x_label = ["5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25","30","35","40","45","50"]
    x_label = ["5", "10",  "15", "20", "25", "30", "35", "40", "45", "50"]
    print(len(x_label), data_alph_leiden_h1975.shape)
    #for h358 x_label = ["3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18"] makes them the same length as h1975 and then you just crop out the zeros in the image version
    fig, ax = plt.subplots()
    im = ax.imshow(data_alph_leiden_h1975, cmap=cmap_div)
    #cbar = ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set_xticks(np.arange(len(x_label)))
    ax.set_yticks(np.arange(len(y_label )))
    # ... and label them with the respective list entries
    ax.set_xticklabels(x_label)
    ax.set_yticklabels(y_label )

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor", size = 12)
    plt.setp(ax.get_yticklabels(), size = 12)
    # Loop over data dimensions and create text annotations.
    for i in range(0,np.shape(data_alph_leiden_h1975)[0]):#range(len(data_alph_leiden_h1975)):
        for j in range(0,np.shape(data_alph_leiden_h1975)[1]):

            text = ax.text(j, i, int(round(data_alph_leiden_h1975[i, j])),
                           ha="center", va="center", color="black",size=12)

    #ax.set_title("F1-score of H1975 (n=100)")
    ax.set_title("F1-score of H1975 (n=100)")
    plt.xlabel('Num. samples from each of the other cell types [000s]')

    fig.tight_layout()
    plt.show()

def make_donutplot():

    # Make data: I have 3 groups and 7 subgroups
    group_names = ['Gluta', 'GABA', 'NN']
    alph_group_size = [65.1,17.9,16.9]
    Scscope_group_size = [62.6,17,20.4]

    splitseq_names = ['Neuronal',' ','NN']
    splitseq_group_size=[83,0,17]

    # Create colors
    a, b, c = [plt.cm.Blues, plt.cm.Reds, plt.cm.Greens]

    # First Ring (outside)
    fig, ax = plt.subplots()
    ax.axis('equal')
    mypie, _ = ax.pie(alph_group_size, radius=1.3, colors=['#0045D2','#d10011','#f4dd07'])
    plt.setp(mypie, width=0.2, edgecolor='white') #labels=group_names,

    # Second Ring (Inside)
    mypie2, _ = ax.pie(Scscope_group_size, radius=1.3 - 0.3,  labeldistance=0.7,
                       colors=['#0045D2','#d10011','#f4dd07']) #labels=group_names,
    plt.setp(mypie2, width=0.2, edgecolor='white')

    # Third Ring (Inside)
    mypie3, _ = ax.pie(splitseq_group_size, radius=1.3 - 0.6,  labeldistance=0.7,
                       colors=['#0045D2','#d10011','#f4dd07']) #labels=splitseq_names,
    plt.setp(mypie3, width=0.2, edgecolor='white')
    plt.margins(0, 0)

    # show it
    plt.show()

def make_scatter():
    plt.scatter(np.random.randn(100), np.random.randn(100), s = 20,  c='#b5cdd6', edgecolors='#90b5c1')
    plt.show()


def make_3Dbarchart():
    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    from matplotlib import cm

    result_all1 = [[47.7,63.4,	67,	0,66.5,0, 62.2, 43.7,0,	80.1,98.8],
              [46.1,56.27,	65.8,	0,63.0,0,50,17.9,	0,	31,98.1],
              [32.8,63.7,	58.7,	0,44.66,0,35.2,39.5,0,	0,85.27],
              [42.4,49.6,	52.0,	0,65.1,0,7.0,27.2,	0,	0,91],
              [37.9,72.7,	63.1,	0,19.49,0,10.2,8.9,	0,	0,66.5],
              [21.5,23.7,	32.25,	0,2.8579,0,0,1.6,	0,	0,35.8],
                   ]

    #Leiden
    result_all = [[47,62.0,	67,	0,73,0, 62, 49.3,0,	70,98.8], #parc
              [46.1,56.27,	65.8,	0,63.0,0,50,17.9,	0,	31,98.1], #pheno
                  [46, 55.6, 62.5, 0, 62.4, 0, 14.5, 49, 0, 0, 35.8],  #have not run seurat on lungALL "35.8" is dummy but not used in actual plot
              [32.8,63.7,	58.7,	0,44.66,0,35.2,39.5,0,	0,85.27], #flowsom
              [42.4,49.6,	52.0,	0,65.1,0,7.0,27.2,	0,	0,91], #kmeans
              [37.9,72.7,	63.1,	0,19.49,0,10.2,8.9,	0,	0,66.5], #flock
              [21.5,23.7,	32.25,	0,2.8579,0,0,1.6,	0,	0,35.8]    ] #flowpeaks


    result_multiple = [[47.7,	63.4,	67,	66.5],
                       [46.1,	56.27,	65.8,	66.515],
                       [32.8,	63.7,	58.7,	44.66],
                       [42.6,	50.3,	53.2,	65.9],
                       [37.9,	72.7,	63.1,	19.49],
                       [21.5,23.7,32.25,2.8579]]


    result_rare = [[	43.7,	62.2,		80.1],
              [	17.9,	50,	31],
              [39.5,	35.2,		0],
              [31.3,	8.3,		0],
              [8.9,	10.2,	0],
              [	1.6,	0,	0]]

    result = np.array(result_all, dtype=np.int).T

    fig = plt.figure(figsize=(5, 5))#, dpi=150)
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.set_alpha(0.5)

    ylabels = np.array(['  Levine_13dim','  Levine_32dim' ,'  Samusik_all', ' ','  PBMC',' ','  Mosmann_rare','  Nilsson_rare', ' ', '  LungCancer_rare'])
    #xlabels = np.array( ['Levine_13','Levine_32', 'Samusik','PBMC_10X'])
    #xlabels = np.array( [ 'Nilsson_rare', 'Mosmann_rare',  'LungCancer_rare'])
    xlabels = np.array(['PARC', 'Phenograph', 'Seurat','FlowSOM','KMeans', 'Flock','FlowPeaks'])
    #ylabels = np.array(['Alph', 'Phenograph', 'FlowSOM','KMeans', 'Flock','FlowPeaks'])
    xpos = np.arange(xlabels.shape[0])

    ypos = np.arange(ylabels.shape[0])

    xposM, yposM = np.meshgrid(xpos, ypos, copy=False)

    zpos = result
    zpos = zpos.ravel()

    dx = 0.5
    dy = 0.5
    dz = zpos

    ax1.w_xaxis.set_ticks(xpos + dx / 2.)
    ax1.w_xaxis.set_ticklabels(xlabels)

    ax1.w_yaxis.set_ticks(ypos + dy / 2.)
    ax1.w_yaxis.set_ticklabels(ylabels)

    # make the panes transparent
    ax1.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax1.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax1.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # make gridlines transparent
    ax1.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax1.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax1.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)

    values = np.linspace(0.2, 1., yposM.ravel().shape[0])
    colors = cm.rainbow(values)
    for color in colors: #set transparency to 0.5
        color[3]=0.5
    #print(colors)
    #values = np.linspace(0.2, 1., result.shape[0])
    print(xposM.ravel().shape[0])
    #cmaps = [cm.Blues, cm.Reds, cm.Greens,cm.Blues, cm.Reds, cm.Greens, cm.Blues]
    #colors = np.hstack([c(values) for c in cmaps]).reshape(-1, 4)

    colors_list = []
    # for i in list(range(xposM.ravel().shape[0])):
    #     if i%7 == 0: colors.append('blue')
    #     elif i % 7 == 1: colors.append('yellow')
    #     elif i % 7 == 2: colors.append('red')
    #     else: colors.append('green')
    #
    #colors = ['blue']*7+['red']*7+['green']*7+['blue']*7+['red']*7+['green']*7
    # colors0 = np.asarray([1.00000000e-01, 5.87785252e-01, 9.51056516e-01, 5.00000000e-01]*3+[1.00000000e-01, 5.87785252e-01, 9.51056516e-01, 0]+[1.00000000e-01, 5.87785252e-01, 9.51056516e-01, 5.00000000e-01]+[1.00000000e-01, 5.87785252e-01, 9.51056516e-01, 0]+[1.00000000e-01, 5.87785252e-01, 9.51056516e-01, 0.5]*2+[1.00000000e-01, 5.87785252e-01, 9.51056516e-01, 0]+[1.00000000e-01, 5.87785252e-01, 9.51056516e-01, 5.00000000e-01])
    colorsParc = np.asarray([1.00000000e-01, 5.87785252e-01, 9.51056516e-01, 5.00000000e-01])
    colorsPheno = np.asarray([0.16862745098039217, 0.7372549019607844, 0.2901960784313726,0.5])
    colorsFlowSOM = np.asarray([0.9568627450980393, 0.5843137254901961, 0.09411764705882353,0.5])
    colorsSeurat = np.asarray( [0.9098039215686274, 0.1450980392156863, 0.24705882352941178, 0.5]) #seurat
    colorsFlock = np.asarray([1.0, 0.8627450980392157, 0.10980392156862745,0.5])
    colorsKMeans =np.asarray([0.9568627450980393, 0.2196078431372549, 0.8823529411764706,0.5])
    colorsFlowPeaks =np.asarray([0.6784313, 0.2588235294, 0.9607843137,0.5])

    #np.asarray( [1.00000000e+00 ,0,6.12323400e-17, 0.5]*10)
    colors_none = np.asarray([1,1,1,0]*7).reshape(-1,4)
    #colors = np.hstack([colors0,colors1,colors2,colors3,colors4,colors5]).reshape(-1,4)

    print(colors.shape)

    print('r3', result[0], xposM, yposM)
    #ax1.bar3d(xposM.ravel(), yposM.ravel(), dz * 0, dx, dy, dz, color=colors, edgecolor = 'black',linewidth=.2)
    for i in range(10):
        if i == 3 or i==5 or i==8:ax1.bar3d(xposM[i], yposM[i], 0, dx, dy, result[i], color=colors_none, edgecolor = 'black',linewidth=0)
        else:ax1.bar3d(xposM[i], yposM[i], 0, dx, dy, result[i], color=np.hstack([colorsParc, colorsPheno, colorsSeurat,colorsFlowSOM, colorsKMeans, colorsFlock,colorsFlowPeaks]).reshape(-1,4), edgecolor = 'black',linewidth=0.05)
        if i ==9: print('LC',result[i])
    #ax1.set_facecolor(colors)
    plt.setp(ax1.get_xticklabels(), fontsize=4)
    plt.setp(ax1.get_yticklabels(), fontsize=4)
    plt.setp(ax1.get_zticklabels(), fontsize=4)

    plt.show()
def error_bar_plot():
    fig, (ax0) = plt.subplots(nrows=1)
    x = np.array([ 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 30, 35, 40, 45, 50])
    len_x = x.shape[0]
    zero_error = [0 for i in range(len_x)]

    y_pheno = np.array([	18.0 	,24.88 	,57.8,	51.00 	,49.5,	32.64 ,	15.75 ,	32.75 ,	7.83 ,	28.00 ,	35.75 ,	0.00 	,16.25 ,	38.60 ,	32.38 ,	26.43 ,	0,	0	,21.3 ,	15.0 ,	21.5 	,0.0, 	15.3, 	14.3 	,17,	0])
    y_alph = np.array([	70.2 ,	65.9 ,	73.4 	,70.0 ,	71.7 ,	72.1 ,	65.3 ,	63.8 	,59.8 ,	57.8 	,61.1 	,59.0 	,62.0 ,	61.5 ,	66.3 ,	67.2 	,60.0 ,	52.7 ,	47.0 ,	56.5 ,	61.5 ,	60.8 	,56.1 	,57.3 	,54.0 ,	51.7 ])
    std_dev_pheno = np.array([36.0, 	34.5 ,	41.1, 	34.0, 	35.0 ,	34.2 	,31.5 ,	37.9, 	19.2, 	32.6, 	28.4, 	0.0, 	32.5 ,	37.7, 	34.9 ,	33.4 	,0.0 ,	0.0, 	39.1 	,0.0, 	30.2 ,	0.0 ,	26.4 ,	24.8, 	29.4, 	0.0 ])
    std_dev_alph = np.array([6.1, 	5.9 ,	3.6 ,	2.8 ,	8.0 ,	6.5 ,	4.0 ,	8.3 ,	6.0 ,	12.4 ,	4.5 	,5.2, 	7.5 	,7.9 ,	6.0, 	4.4 ,	3.2 ,	14.6, 	4.2, 	2.1, 	7.1, 	4.9, 	7.5 ,	6.7 ,	1.4, 	3.8 ])


    #ax0.errorbar(x, y_pheno, zero_error, fmt='o', color='blue', linewidth = 0.5,capsize=2)
    ax0.errorbar(x, y_alph, std_dev_alph, fmt='o', color='green', linewidth = 0.5, capsize = 2)
    plt.xlabel('Num. samples from each of the other cell types [000s]')
    plt.title('ALPH one-vs-all F1-Score of H1975')
    plt.ylim(-20,100)

def heatmap_Kvalues():

    array_vals= np.array([[0.49,0.62,0.55],[0.14,0,0],[0.18,0.66,0.24],[0.18,0.6,0],[0.15,0.01,0],[0.14,0.01,0],[0.21,0.15,0],[0.37,0.46,0],[0.18,0.49,0],[0.19,0.48,0],[0.18,0.50,0],[.184,.6,.58],[.189,.6,.02],[.145,.49,0.01],[.18,.0,0],[.18,0.03,0]])
    #PARC, Pheno, Seurat
    import seaborn as sns
    cmap_div = sns.diverging_palette(240, 10, as_cmap=True)
    g = sns.clustermap(array_vals, cmap=cmap_div, col_cluster=False, row_cluster=False)
    plt.savefig('/home/shobi/Thesis/Paper_writing/Paper_Figures'+ 'kvals_heatmap_withSeurat.png', dpi=350)


    plt.show()
if __name__ == '__main__':
   #error_bar_plot()
    #spike_test_heatmap()
   make_3Dbarchart()
   #heatmap_Kvalues()


