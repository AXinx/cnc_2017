# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 20:30:07 2016

@author: xinruyue
"""

from network_dw import *
from scipy.misc import imshow,imsave

#随机网
E = nx.random_graphs.erdos_renyi_graph(1000,0.008)
#无标度网
B = nx.random_graphs.barabasi_albert_graph(1000,4)
#小世界网
W = nx.random_graphs.watts_strogatz_graph(1000,14,0.1)

g = W #or W
plt.figure(1)
pos = nx.spring_layout(g)
nx.draw_networkx_nodes(g, pos, node_color='r', node_size=50, alpha=1)
nx.draw_networkx_edges(g, pos, alpha=0.6)
ax= plt.gca()
ax.collections[0].set_edgecolor("#2B2B2B")
plt.title('WS network')
plt.axis('off')
plt.show()
plt.savefig('WS_net')

#2.deepWalk
corp = randomWalk(g,10000)
vec,mod,nod = word_2_vec(corp,20)

#3.pca
pca = PCA(n_components=2)
lowdim = pca.fit_transform(vec)

#4.栅格化
img_data = img_max(lowdim,48)

#plt.figure(2)
#plt.imshow(img_data)
#imshow(img_data)
imsave('WS_img.jpg',img_data)
#plt.savefig('random_48')

#4.plot
plt.figure(3)
par = list(zip(*lowdim))
x = par[0]
y = par[1]
plt.scatter(x,y,s=100,color='w',edgecolors='b')
plt.title('WS pointcloud')
plt.savefig('WS_w2v')

plt.show()
