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
nchains = 100                   # # chaines aléatoires
niterations = 50000              # # d'itérations
prot = 0.6                       # Proba. d'un rotationnel
snfile = "snet.json"             # Fichier du réseau
output = "snet-divrot-harmonicity-60"      # Fichier résultat
rnd.seed(1234)                   # Graine du GNA


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
Ls = np.zeros((niterations, nchains)) # Mesure d'harmonicité

# D1 = D1.tocsr()
D2 = D2.tocsr()


for ic in range(nchains):
    print('\r{: 3} / {}'.format(ic, nchains), end='\r', flush=True)
    H = rnd.randn(nedges)       # Initialisation aléatoire
    H /= la.norm(H, axis=0)     # Normalisation
    
    for it in range(niterations):
        if rnd.rand() < prot:
            t = rnd.choice(ntriangles)
            es, _, a = sp.find(D2[:,t])
        else:
            v = rnd.choice(nvertices)
            _, es, a = sp.find(D1[v,:])
        a /= la.norm(a)

        X = H[es]
        X -= np.dot(a, X) * a
        X *= la.norm(H[es]) / la.norm(X)
        H[es] = X

        Ls[it,ic] = la.norm(L.dot(H), axis=0)


## Affichage des résultats
plt.figure(figsize = (4 * width, 4 * height))
Lmean = np.median(Ls, axis=1)
Ltop = np.percentile(Ls, 75, axis=1) 
Lbot = np.percentile(Ls, 25, axis=1)
plt.fill_between(range(niterations), Ltop, Lbot, facecolor="green", alpha=0.5)
plt.plot(Lmean)

plt.xlabel('Iterations')
plt.ylabel('Harmonicity')
plt.ylim(ymin=0)

plt.savefig('{}.pdf'.format(output),
            frameon=False, bbox_inches='tight', pad_inches=0.0)
plt.draw()

plt.show()
