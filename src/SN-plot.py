# -*- coding: utf-8 -*-

## Affichage graphique d'un réseau de capteurs donné par un
## fichier JSON tel que fourni par le programme
## SN-sampling.py

import json
import numpy as np
import matplotlib.pyplot as plt


## Paramètres
snfile="snet.json"              # Fichier du réseau
hfile="snet-divrot.json"        # Fichier de l'harmonique



## Chargement du réseau sous forme JSON
with open(snfile, 'r') as f:
    data = json.load(f)

width, height = data['size']
locations = np.array([complex(*v['location']) for v in data['vertices']])
edges = np.array([e['vertices'] for e in data['edges']])
triangles = np.array([t['edges'] for t in data['triangles']])
triangles_vertices = np.array([t['vertices'] for t in data['triangles']])

nvertices = len(locations)
nedges = len(edges)
ntriangles = len(triangles)



if hfile:
    with open(hfile, 'r') as f:
        data = json.load(f)
        chains = np.zeros((nedges, data['dimension']))
        for i, weights in enumerate(data['chains']):
            for e in weights:
                chains[e['id'], i] = e['weight']
else:
    chains=np.zeros((edges,0))



## Affichage
plt.close('all')

# Pour le réseau
plt.figure(figsize = (8 * width, 8 * height))
ax = plt.axes(frameon=False)
ax.axis([-0.05, width + 0.05, -0.05, height + 0.05])
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

for t in triangles_vertices:
    zs = locations[t]
    ax.fill(zs.real, zs.imag,
            fill=True, facecolor='blue', alpha=0.4)

for i, e in enumerate(edges):
    zs = locations[e]
    ax.plot(zs.real, zs.imag,
            color='green', lw=2)

for i, z in enumerate(locations):
    ax.text(z.real, z.imag, "%2d" % i, fontsize=10, color='red',
            horizontalalignment='center', verticalalignment='center')

plt.savefig('{}-{:03}.pdf'.format(hfile, 0),
            frameon=False, bbox_inches='tight', pad_inches=0.0)
plt.draw()

# Pour les chaines harmoniques
cmap = plt.get_cmap('Reds')

for ichain, chain in enumerate(chains.T):
    chain = np.abs(chain)
    chain = (chain - chain.min()) / (chain.max() - chain.min())
    
    plt.figure(figsize = (8 * width, 8 * height))
    ax = plt.axes(frameon=False)
    ax.axis([-0.05, width + 0.05, -0.05, height + 0.05])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    for t in triangles_vertices:
        zs = locations[t]
        ax.fill(zs.real, zs.imag,
                fill=True, facecolor='blue', alpha=0.4)

    for i, e in enumerate(edges):
        zs = locations[e]
        ax.plot(zs.real, zs.imag,
                color=cmap(chain[i]),
                lw=10 * chain[i])

    for i, z in enumerate(locations):
        ax.text(z.real, z.imag, "%2d" % i, fontsize=10, color='red',
                horizontalalignment='center', verticalalignment='center')

    plt.savefig('{}-{:03}.pdf'.format(hfile, 1+ichain),
                frameon=False, bbox_inches='tight', pad_inches=0.0)
    plt.draw()
        
plt.show(block=False)
