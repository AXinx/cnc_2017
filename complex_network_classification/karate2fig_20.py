# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 11:23:15 2016

@author: xinruyue
"""
from network_dw import *
from scipy.misc import imsave

# karate data
g = nx.Graph()
g.add_edges_from(
    [(2, 1), (3, 1), (3, 2), (4, 1), (4, 2), (4, 3), (5, 1), (6, 1), (7, 1), (7, 5), (7, 6), (8, 1), (8, 2), (8, 3),
     (8, 4), (9, 1), (9, 3), (10, 3), (11, 1), (11, 5), (11, 6), (12, 1), (13, 1), (13, 4), (14, 1), (14, 2), (14, 3),
     (14, 4), (17, 6), (17, 7), (18, 1), (18, 2), (20, 1), (20, 2), (22, 1), (22, 2), (26, 24), (26, 25), (28, 3),
     (28, 24), (28, 25), (29, 3), (30, 24), (30, 27), (31, 2), (31, 9), (32, 1), (32, 25), (32, 26), (32, 29), (33, 3),
     (33, 9), (33, 15), (33, 16), (33, 19), (33, 21), (33, 23), (33, 24), (33, 30), (33, 31), (33, 32), (34, 9),
     (34, 10), (34, 14), (34, 15), (34, 16), (34, 19), (34, 20), (34, 21), (34, 23), (34, 24), (34, 27), (34, 28),
     (34, 29), (34, 30), (34, 31), (34, 32), (34, 33)])

type_1 = ['5', '6', '7', '11', '17']
type_2 = ['1', '2', '4', '8', '12', '13', '14', '18', '20', '22']
type_3 = ['3', '10', '25', '26', '28', '29', '32']
type_4 = ['9', '15', '16', '19', '21', '23', '24', '27', '30', '31', '33', '34']

type_list = [type_1, type_2, type_3, type_4]
color = ['blue', 'red', 'green', 'purple']

node_colors = []
for each in type_list:
    m = type_list.index(each)
    for i in each:
        i = int(i)
        node_colors.append(color[m])

plt.figure(1)
nx.draw_networkx(g,node_color=node_colors, with_labels=True)
plt.axis('off')
plt.savefig('karate_data')

# deepwalk
corp = randomWalk(g, 10000)
vec, mod, nod = word_2_vec(corp, 20)

# PCA
pca = PCA(n_components=2)
lowdim = pca.fit_transform(vec)

#4.栅格化
img_data = img_max(lowdim,48)

plt.figure(3)
imsave('karate_28.jpg', img_data)


# plot
plt.figure(2)
for each in type_list:
    m = type_list.index(each)
    for i in each:
        nod = list(nod)
        idx = nod.index(i)
        cor = lowdim[idx]
        plt.scatter(cor[0], cor[1], c=color[m], s=100)
plt.savefig('karate_w2v')
plt.show()

