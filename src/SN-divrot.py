# -*- coding: utf-8 -*-


## Calcul d'une base de forme harmonique à partir de
## projections successives pour annuler des rotationnels ou
## une divergence. Le complexe est issu d'un réseau de
## capteurs donné par un fichier JSON tel que fourni par le
## programme SN-sampling.py

import json
import numpy as np
import numpy.linalg as la
import numpy.random as rnd
import scipy.sparse as sp
import scipy.sparse.linalg as lin
import matplotlib.pyplot as plt


## Paramètres
nchains = 5                     # # chaines aléatoires
niterations = 50000             # # d'itérations
prot = 0.7                      # Proba. d'un rotationnel
snfile = "snet.json"            # Fichier du réseau
output = "snet-divrot.json"     # Fichier résultat
rnd.seed(1234)                  # Graine du GNA


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



## Construction des matrices de bord et du Laplacien
coef = np.vstack((-np.ones(nedges),
                  +np.ones(nedges)))
D1 = sp.csc_matrix((coef.T.flat,
                    edges.flat,
                    np.arange(0,2*nedges+1,2)),
                   shape=(nvertices, nedges),
                   dtype=np.float).tocsr()
coef = np.vstack((+np.ones(ntriangles),
                  -np.ones(ntriangles),
                  +np.ones(ntriangles)))
D2 = sp.csc_matrix((coef.T.flat,
                    triangles.flat,
                    np.arange(0,3*ntriangles+1,3)),
                   shape=(nedges, ntriangles),
                   dtype=np.float)

L = D2.dot(D2.T) + D1.T.dot(D1)



## Récupération des chaînes harmoniques
H = rnd.randn(nedges, nchains) # Initialisation aléatoire
H /= la.norm(H, axis=0)        # Normalisation

Ls = np.zeros((niterations, nchains)) # Mesure d'harmonicité
Ns = np.zeros((niterations, nchains)) # Mesure de dégénérescence

D1 = D1.tocsr()
D2 = D2.tocsr()

for it in range(niterations):
    if rnd.rand() < prot:
        t = rnd.choice(ntriangles)
        es, _, a = sp.find(D2[:,t])        
    else:
        v = rnd.choice(nvertices)
        _, es, a = sp.find(D1[v,:])
    a /= la.norm(a)

    X = H[es, :]
    X -= np.dot(a, X) * a[:,np.newaxis]
    norms = la.norm(X, axis=0)
    for j, n in enumerate(norms):
        X[:, j] *= la.norm(H[es,j], axis=0) / n
    H[es,:] = X

    Ls[it,:] = la.norm(L.dot(H), axis=0)
    Ns[it,:] = la.norm(H, axis=0)



## Enregistrement des Harmoniques
with open(output, 'w') as file:
    json.dump({'dimension': nchains,
               'chains': [[{'id': i, 'weight': w}
                            for i, w in enumerate(chain)]
                           for chain in H.T]},
              file,
              separators=(',', ':'),
              indent=2)



## Affichage des résultats
plt.figure(figsize = (8 * width, 8 * height))
plt.plot(Ls)
plt.xlabel('Iterations')
plt.ylabel('Harmonicity')

plt.savefig('{}-harmonicity.pdf'.format(output),
            frameon=False, bbox_inches='tight', pad_inches=0.0)
plt.draw()

plt.figure(figsize = (8 * width, 8 * height))
plt.plot(Ns)
plt.xlabel('Iterations')
plt.ylabel('Norm')

plt.savefig('{}-norm.pdf'.format(output),
            frameon=False, bbox_inches='tight', pad_inches=0.0)
plt.draw()

plt.show(block=False)
