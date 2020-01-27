import igraph as ig
import leidenalg
import louvain
G = ig.Graph.Erdos_Renyi(100, 0.1);
part = leidenalg.find_partition(G,leidenalg.ModularityVertexPartition)
part_louvain =louvain.find_partition(G, louvain.ModularityVertexPartition)
print(part)
print(part_louvain)