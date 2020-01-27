## doesnt visualize as well as box-plot

import numpy as np
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 350
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = '13'
## agg backend is used to create plot as a .png file
#mpl.use('agg')
dataset = 'Spike_H1975'
import matplotlib.pyplot as plt
#Lung Cancer Dataset
if dataset == "Lung_Cancer":
    title = 'Lung Cancer - Mean F1-accuracy'
    PARC = np.array([98.8]) #10-70 clusters,
    Pheno = np.array([98.1])
    #FlowSOM= np.array([79.65,87.95,82.02,89.63,	90.81,	90.81,	95.16,	95.16,	95.48,	95.48,	95.48,	93.17,	93.97,	89.99])#79.6559,	82.02, 90.67 Aug2019
    FlowSOM = np.array([66.1,79.65,80.1,80.8,82.02,82.44,90.6,90.6]) #n_clus =9, 10, 12,14,...,24 Nov2019
    FlowPeaks= np.array([35.749])
    FLOCK= np.array([66.54])
    #KMeans= np.array([78.19,91.55,	91.088,	91.66,	93.157,	91.342,	89.065,	93.6125502,93.90458,93.6, 94.23, 94.65,	93.9])#78.1958723,	91.55,Aug 2019
    KMeans = np.array([65.67408, 75.24580437,85.71410,78.506,90.8694,91.6191254, 91.6298,91.241976,92]) #clusters from 10-24 in intervals of 2 Nov2019

if dataset == "Spike_H1975":
    title = 'F1-score of H1975 (0.2% total)'
    PARC = np.array([72])#6,72.2,62.2,67,79.3,70,72,79,72,72]) #10-70 clusters,
    Pheno = np.array([28])#0,70.4,67,0,0,72.4,70,0,0,0,0,0,0,0,0])
    FlowSOM= np.array([	0])
    FlowPeaks= np.array([0])
    FLOCK= np.array([0])
    KMeans= np.array([	0])
    Seurat = np.array([0])
#10X PBMC Gene dataset
if dataset == "10X_PBMC":
    title = "10X_PBMC - Mean F1-accuracy"
    PARC = np.array([72.5])
    Pheno = np.array([66.515])
    FlowSOM= np.array([	57.27,	57.44,	67.24,	67.65,	67.86,	67.46,	68.21267,	67.86,	67.94,	67.95,	67.99,	63.73328231, 46.24,57.42]) #46.24,	57.42#20Grid
    FlowPeaks= np.array([2.8579])
    FLOCK= np.array([19.49])
    KMeans= np.array([	66.24522869,	69.30109553,	68.96895879,	67.98098267,	68.37358721,	68.20055709,	67.00916829,	67.9509312,	68.98352134,	68.33425971,	68.39412102,50.38540195,56.91918195])#50.38540195,56.91918195,
    Seurat = np.array([64])
if dataset == 'Levine_13dim':
    title = 'Levine_13dim- Mean F1-accuracy'
    PARC = np.array([49.0]) #10-70 clusters,
    Pheno = np.array([46.1])
    #FlowSOM= np.array([40.24,38.6,		34.7,	36,	35.46,	36.9,	30.3,	34.7,	30.19,	27.7,	23.76]) #51.22,41.9 Aug2019
    FlowSOM = np.array([34.7, 36, 35.46, 36.9, 30.3, 34.7, 30.19, 27.7])  # 51.22,41.9 Nov2019
    FlowPeaks= np.array([21.49])
    FLOCK= np.array([37.85])
    KMeans= np.array([42.8,	43.5,	42.4,	43.1,	44.1,	43.7,	47.1,	47.7,	43.2,	41.9,	27.7])#43.1,43.2
    Seurat = np.array([46])

if dataset == "Levine_32dim":
    title = 'Levine_32dim - Mean F1-accuracy'
    PARC = np.array([62.4]) #10-70 clusters,
    Pheno = np.array([56.27])
    FlowSOM = np.array([71.2,58.1,56.8,55.65,55.2,53.1,53.2,54.44,53.9]) #ntype-2*ntypes
    #FlowSOM= np.array([	67.7,	78.1,	71.2,	74.2,	55.65,	53.1,	54.44,40.8]) #aug2019 72.12,	73.64,
    FlowPeaks= np.array([23.65])
    FLOCK= np.array([72.68])
    KMeans= np.array([47.2,	45.4,	51.7,	44.8,	48.8,	49.4,	45.8,	50.1]) ##54.5,	54.2,aug2019
    Seurat = np.array([55.6])

if dataset == "Samusik_01":
    title = 'Samusik_01 - Mean F1-accuracy'
    PARC = np.array([64.2]) #10-70 clusters,
    Pheno = np.array([61.3])
    FlowSOM= np.array([66.91,70.97,	66.58,	68.79,	71.08,	69.1,	69.39,	67.1,	57.81,	63.85,	53.1,	44.7,	26.4,	16.9])#66.91,70.97,
    FlowPeaks= np.array([5.78])
    FLOCK= np.array([60.77])
    KMeans= np.array([53.3,60.5,59.1,	57.7, 55.7,	54,	54.6,	55.1,	52.8,	45.7,	42.4,	29.7,	29.7])#53.3,60.5,
    Seurat = np.array([58.17])

if dataset == "Samusik_all":
    title = 'Samusik_all - Mean F1-accuracy'
    PARC = np.array([66.5, 66.5,66.5, 66.5,66.5, 66.5,66.5, 66.5,66.5, 66.5,66.5]) #10-70 clusters,
    Pheno = np.array([65.8,65.8,65.8,65.8,65.8,65.8,65.8,65.8,65.8,65.8,65.8])
    #FlowSOM= np.array([72.68,	69.75,	69.64,	70.85,	65.65,	64.4,	61.86,	58.67,	53.23,	45.25,	32.98])#72.22,	70.1 aug2019
    FlowSOM = np.array( [69.94,69.67, 65.65, 64.4,62.4, 61.86, 58.67, 55.13,54.16,55.6,54.2,53.23])  # k + n to 2n_classes // Nov2019

    FlowPeaks= np.array([32.25,32.25,32.25,32.25,32.25,32.25,32.25,32.25,32.25,32.25,32.25])
    FLOCK= np.array([63.108,63.108,63.108,63.108,63.108,63.108,63.108,63.108,63.108,63.108,63.108,])
    #KMeans= np.array([	60.6,	54,	60,	59.8,	60.1,	61.52,	55.2,	54.3,	40.3])#aug2019
    KMeans = np.array([59.4, 54, 60, 59.8, 60.1, 61.52, 55.2, 54.3, 40.3, 37.2])  # // nov2019
    Seurat = np.array([62])# nov2019 average of default 10pcs and full feature input
if dataset == "Mosmann_rare":
    title = 'Mosmann_rare - Mean F1-accuracy'
    PARC = np.array([61.5]) #10-70 clusters,
    Pheno = np.array([50])
    FlowSOM= np.array([	64.3,	11.1,	65.6,	68.13,	65.3,	69.33,	65.23,	9.25,	10.17,	8.79,	10.4,10.7])#67.96,	10.7
    FlowPeaks= np.array([0])
    FLOCK= np.array([10.2])
    KMeans= np.array([	12.3,	11.9,	11.5,	11,	10.9,	10.4,	5.3,	1,	3,	0,	0,])#18.7,12.5,
    Seurat = np.array([49])
if dataset == "Nilsson_rare":
    title = 'Nilsson_rare - Mean F1-accuracy'
    PARC = np.array([49.5]) #10-70 clusters,
    Pheno = np.array([17.9])
    FlowSOM= np.array([	56.03,	42.96,	38.77,	53.63,	36.77,	55.86,	58.36,	29.13,	59.44,	31.85,	0.073,	0.051,36.36])#36.36,	50.8,
    FlowPeaks= np.array([1.61])
    FLOCK= np.array([8.89])
    KMeans= np.array([	53.1,	53.1,	52.7,	24.63,	24.4,	21.1,	20.9,	20.6,	8.8,	10])#55.3,	53.1,
    Seurat = np.array([18])
# Color codes: https://matplotlib.org/examples/color/named_colors.html
#boxColors = ['deepskyblue', 'mediumaquamarine' ,'coral','blueviolet','gold','hotpink']
boxColors = ['deepskyblue', 'mediumaquamarine' ,'peachpuff','blueviolet','gold','lightpink','tomato']
boxOutlineColors = ['dodgerblue', 'forestgreen' ,'orangered','indigo','goldenrod','deeppink','crimson']


data_to_plot = [PARC, Pheno, FlowSOM, FlowPeaks, FLOCK, KMeans, Seurat]

#data_to_plot = np.concatenate(data_to_plot).reshape(())
print('data to plot', data_to_plot)
# Create a figure instance
fig = plt.figure(1, figsize=(6, 5))

# Create an axes instance
ax = fig.add_subplot(111)
# Create the boxplot



## add patch_artist=True option to ax.boxplot()
## to get fill color
bp = ax.boxplot(data_to_plot, patch_artist=True)


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
x_labels_str = ['PARC',"Phenograph",'Flowsom', "FlowPeaks","Flock", "Kmeans", "Seurat"]
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
ax.set_ylim([-1,75])
ax.set_xticklabels(x_labels_str)
#plt.xticks(rotation=20)
ax.set_ylabel('Mean F1-accuracy (%)')
ax.set_title(title)
plt.show()
#fig.savefig('/home/shobi/Thesis/10x_visuals/PBMC/'+title+'.png', bbox_inches='tight')
fig.savefig('/home/shobi/Thesis/Paper_writing/Paper_Figures/'+title+'Nov.tif', bbox_inches='tight',dpi = 350,format = 'png')