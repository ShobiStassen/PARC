## doesnt visualize as well as box-plot

import numpy as np
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 350
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = '13'
## agg backend is used to create plot as a .png file
#mpl.use('agg')
import matplotlib.pyplot as plt
title = []
type = 'HVG10-100PC'
title_68 = '68K PBMC -50PCs'

PARC = np.array([73.4])
FlowSOM= np.array([	58,62.8,63.4,66.1,66.35,66.36,66.36])
KMeans= np.array([	66,66,69,68,69,69,70])
Seurat = np.array([63])
#SC3 = np.array([55])
data_68 = [PARC, FlowSOM, KMeans, Seurat]#,SC3]

title_z8 = 'Zheng8eq - 100PCs'
PARC = np.array([60])
FlowSOM= np.array([	42,48,49,50,50,50,55])
KMeans= np.array([	38,42,45.5,45.6,46,48,51])
Seurat = np.array([48])
#SC3 = np.array([58])
data_z8 = [PARC, FlowSOM, KMeans, Seurat]#,SC3]

title_z4 = 'Zheng4uneq - 100PCs'
PARC = np.array([91])
FlowSOM= np.array([	73,72.6,72.6,72.6,72.3,71.6])
KMeans= np.array([	63,65.1,66,72.5,72.5,72.4,72.7])
Seurat = np.array([85])
#SC3 = np.array([98.6])
data_z4 = [PARC, FlowSOM, KMeans, Seurat]#,SC3]

title_koh = 'Koh - 100PCs'
PARC = np.array([98])
FlowSOM= np.array([83,83,86,86,97,97])
KMeans= np.array([97,97,98,97,98	])
Seurat = np.array([97])
SC3 = np.array([87])
data_koh = [PARC, FlowSOM, KMeans, Seurat]#,SC3]


title_kum = 'KumarAll - 100PCs'
PARC = np.array([99])
FlowSOM= np.array([	99,98.5,99])
KMeans= np.array([	99,98.5,99])
Seurat = np.array([99])
SC3 = np.array([99])
data_kum = [PARC, FlowSOM, KMeans, Seurat,SC3]

type = 'HVG10 filtered counts'
title_68_c = '68K PBMC - Gene Count Matrix'
PARC = np.array([64])
FlowSOM= np.array([	46,48,48,48.3,52,52.8,52.8])
KMeans= np.array([	48.4,52.4,49.3,48.3,48.3,52.8,52.8,54])
Seurat = np.array([57])
SC3 = np.array([45])
data_68_c = [PARC, FlowSOM, KMeans, Seurat,SC3]

title_z8_c = 'Zheng8eq - Gene Count Matrix'
PARC = np.array([69.6])
FlowSOM= np.array([	44.2,44.2,47.6,54.7,50.5,51.8,58.7])
KMeans= np.array([	45.6,48.8,53.4,50.4,54,59,61])
Seurat = np.array([54.7])
SC3 = np.array([62])
data_z8_c = [PARC, FlowSOM, KMeans, Seurat,SC3]

title_z4_c = 'Zheng4uneq - Gene Count Matrix'
PARC = np.array([97])
FlowSOM= np.array([	44.4,71.2,71.5,71.5,72.1,95.1])
KMeans= np.array([	71,70.7,70.7,71.7,92.1,92.2,90.3])
Seurat = np.array([97])
SC3 = np.array([98])
data_z4_c = [PARC, FlowSOM, KMeans, Seurat,SC3]




# Color codes: https://matplotlib.org/examples/color/named_colors.html
#boxColors = ['deepskyblue', 'mediumaquamarine' ,'coral','blueviolet','gold','hotpink']
boxColors = ['deepskyblue', 'mediumaquamarine' ,'peachpuff','blueviolet','gold','lightpink','tomato']
boxOutlineColors = ['dodgerblue', 'forestgreen' ,'orangered','indigo','goldenrod','deeppink','crimson']


#data_to_plot = np.concatenate(data_to_plot).reshape(())

# Create a figure instance
fig = plt.figure()#1, figsize=(6, 5))
titles = [title_68,title_z8, title_z4, title_68_c,title_z8_c,title_z4_c]
for i in [1,2,3,4,5,6]:
    # Create an axes instance
    ax = fig.add_subplot(2,3,i)#(111)
    # Create the boxplot



    ## add patch_artist=True option to ax.boxplot()
    ## to get fill color
    if i ==1: bp = ax.boxplot(data_68, patch_artist=True)
    if i == 2: bp = ax.boxplot(data_z8, patch_artist=True)
    if i == 3: bp = ax.boxplot(data_z4, patch_artist=True)
    if i == 4: bp = ax.boxplot(data_68_c, patch_artist=True)
    if i == 5: bp = ax.boxplot(data_z8_c, patch_artist=True)
    if i == 6: bp = ax.boxplot(data_z4_c, patch_artist=True)
    #if i == 4: bp = ax.boxplot(data_koh, patch_artist=True)
    #if i == 5: bp = ax.boxplot(data_kum, patch_artist=True)


    ## change color and linewidth of the whiskers
    whisker_color_i =0
    count = 0
    for whisker in bp['whiskers']:
        whisker.set(color=boxOutlineColors[whisker_color_i], linewidth=1)
        print(whisker_color_i)
        if count%2==1:
            whisker_color_i = whisker_color_i + 1
        count = count+1

    print(whisker_color_i,'number of whiskers')
    ## change color and linewidth of the caps
    cap_color_i =0
    count = 0
    for cap in bp['caps']:
        cap.set(color=boxOutlineColors[cap_color_i], linewidth=1)
        if count%2 ==1: cap_color_i = cap_color_i+1
        count =count+1
    print(count, 'number of caps')
    ## change color and linewidth of the medians
    median_color_i = 0
    for median in bp['medians']:
        median.set(color=boxOutlineColors[median_color_i], linewidth=5)
        median_color_i = median_color_i+1
    ## change the style of fliers and their fill
    for flier in bp['fliers']:
        flier.set(marker='o', color='#e7298a', alpha=0.5)

    face_color_i=0
    ## change outline color, fill color and linewidth of the boxes
    for box in bp['boxes']:
        # change outline color
        box.set( color=boxOutlineColors[face_color_i], linewidth=0)
        # change fill color
        box.set( facecolor = boxColors[face_color_i] )
        face_color_i =face_color_i+1
    x_labels_str = ['PARC','Flowsom', "KMeans", "Seurat","SC3"]
    '''
    #add a basic legend https://matplotlib.org/gallery/statistics/boxplot_demo.html
    x0 = 0.15
    y0 = 0.15
    count = 0
    
    for i in range(0,6):
        fig.text(x0, y0+count*0.04, x_labels_str[count], backgroundcolor = 'white', color=boxOutlineColors[count], weight='bold',
             size='small')
        count = count +1
    '''
    ## Custom x-axis labels
    if i>3: ax.set_xticklabels(x_labels_str)
    else: ax.set_xticklabels(['','','','',''])
    if i == 1 or i==4:    ax.set_ylabel('Mean F1-Score (%)')
    ax.set_title(titles[i-1])
    plt.xticks(rotation=20)
plt.show()
#fig.savefig('/home/shobi/Thesis/10x_visuals/PBMC/'+title+'.png', bbox_inches='tight')
fig.savefig('/home/shobi/Thesis/Paper_writing/Paper_Figures/'+title+'duoNovCounts.tif', bbox_inches='tight',dpi = 350,format = 'png')