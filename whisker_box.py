## numpy is used for creating fake data
import numpy as np
import matplotlib as mpl

## agg backend is used to create plot as a .png file
mpl.use('agg')

import matplotlib.pyplot as plt
title = 'F1-score of H1975 cells (0.2% total)'
#apt_score = np.array([98.515946,98.65333953,98.43929338,98.74309553,98.33884442,98.8]) #10-70 clusters,
alph_score= np.array([80])
pheno_score = np.array([30])
flowsom_score= np.array([0])
flowpeaks_score = np.array([0])
flock_score = np.array([0])
#drop_clust_score = np.array([0.4963])
kmeans_score = np.array([0])

data_to_plot = [alph_score, pheno_score, flowsom_score, flowpeaks_score, flock_score,kmeans_score]
# Create a figure instance
fig = plt.figure(1, figsize=(6, 5))

# Create an axes instance
ax = fig.add_subplot(111)
# Create the boxplot

bp = ax.boxplot(data_to_plot)

## add patch_artist=True option to ax.boxplot()
## to get fill color
bp = ax.boxplot(data_to_plot, patch_artist=True)

## change outline color, fill color and linewidth of the boxes

for box in bp['boxes']:
    # change outline color
    box.set( color='#7570b3', linewidth=2)
    # change fill color
    box.set( facecolor = '#1b9e77' )

## change color and linewidth of the whiskers
for whisker in bp['whiskers']:
    whisker.set(color='#7570b3', linewidth=2)

## change color and linewidth of the caps
for cap in bp['caps']:
    cap.set(color='#7570b3', linewidth=2)

## change color and linewidth of the medians
for median in bp['medians']:
    median.set(color='#b2df8a', linewidth=5)

## change the style of fliers and their fill
for flier in bp['fliers']:
    flier.set(marker='o', color='#e7298a', alpha=0.5)

## Custom x-axis labels
ax.set_xticklabels(['ALPH',"Phenograph",'FlowSOM','FlowPeaks', "Flock" ,'K-means'])
ax.set_ylabel('Mean F1-accuracy (%)')
ax.set_title(title)
#fig.savefig('/home/shobi/Thesis/10x_visuals/PBMC/'+title+'.png', bbox_inches='tight')
fig.savefig('/home/shobi/Thesis/MultiClass_MinCluster/'+title+'.png', bbox_inches='tight')