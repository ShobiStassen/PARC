## Parameters and Attributes

**Parameter tuning**
For a more detailed explanation of the impact of tuning key parameters please see the Supplementary Analysis in our paper. 
[PARC Supplementary Analysis](https://oup.silverchair-cdn.com/oup/backfile/Content_public/Journal/bioinformatics/PAP/10.1093_bioinformatics_btaa042/1/btaa042_supplementary-data.pdf?Expires=1583098421&Signature=R1gJB7MebQjg7t9Mp-MoaHdubyyke4OcoevEK5817el27onwA7TlU-~u7Ug1nOUFND2C8cTnwBle7uSHikx7BJ~SOAo6xUeniePrCIzQBi96MvtoL674C8Pd47a4SAcHqrA2R1XMLnhkv6M8RV0eWS-4fnTPnp~lnrGWV5~mdrvImwtqKkOyEVeHyt1Iajeb1W8Msuh0I2y6QXlLDU9mhuwBvJyQ5bV8sD9C-NbdlLZugc4LMqngbr5BX7AYNJxvhVZMSKKl4aMnIf4uMv4aWjFBYXTGwlIKCjurM2GcHK~i~yzpi-1BMYreyMYnyuYHi05I9~aLJfHo~Qd3Ux2VVQ__&Key-Pair-Id=APKAIE5G5CRDK6RD3PGA)

| Input Parameter | Description |
| ---------- |----------|
| `data` | (numpy.ndarray) n_samples x n_features |
| `true_label` | (numpy.ndarray) (optional)|
| `dist_std_local` |  (optional, default = 2) local pruning threshold: the number of standard deviations above the mean minkowski distance between neighbors of a given node. the higher the parameter, the more edges are retained|
| `jac_std_global` |  (optional, default = 'median') global level  graph pruning. This threshold can also be set as the number of standard deviations below the network's mean-jaccard-weighted edges. 0.1-1 provide reasonable pruning. higher value means less pruning. e.g. a value of 0.15 means all edges that are above mean(edgeweight)-0.15*std(edge-weights) are retained. We find both 0.15 and 'median' to yield good results resulting in pruning away ~ 50-60% edges |
| `dist_std_local` |  (optional, default = 2) local pruning threshold: the number of standard deviations above the mean minkowski distance between neighbors of a given node. higher value means less pruning|
| `random_seed` |  (optional, default = 42) The random seed to pass to Leiden|
| `resolution_parameter` |  (optional, default = 1) Uses ModuliartyVP and RBConfigurationVertexPartition|
| `jac_weighted_edges` |  (optional, default = True) Uses Jaccard weighted pruned graph as input to community detection. For very large datasets set this to False to observe a speed-up with negligble impact on accuracy |

| Attributes | Description |
| ---------- |----------|
| `labels` | (list) length n_samples of corresponding cluster labels |
| `f1_mean` | (list) f1 score (not weighted by population). For details see supplementary section of [paper](https://doi.org/10.1101/765628) |
| `stats_df` | (DataFrame) stores parameter values and performance metrics |
