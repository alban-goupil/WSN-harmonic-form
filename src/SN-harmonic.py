# -*- coding: utf-8 -*-


## Calcul d'une base de forme harmonique à partir de la
## decription d'un réseau de capteurs donné par un fichier
## JSON tel que fourni par le programme SN-sampling.py

import json
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as lin


## Paramètres
snfile="snet.json"              # Fichier du réseau
output="snet-harm.json"         # Fichier résultat


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



## Construction des matrices de bord
coef = np.vstack((-np.ones(nedges),
                  +np.ones(nedges)))
D1 = sp.csc_matrix((coef.T.flat,
                    edges.flat,
                    np.arange(0,2*nedges+1,2)),
                   shape=(nvertices, nedges),
                   dtype=np.float)
coef = np.vstack((+np.ones(ntriangles),
                  -np.ones(ntriangles),
                  +np.ones(ntriangles)))
D2 = sp.csc_matrix((coef.T.flat,
                    triangles.flat,
                    np.arange(0,3*ntriangles+1,3)),
                   shape=(nedges, ntriangles),
                   dtype=np.float)

## Construction du Laplacien
L = D2.dot(D2.T) + D1.T.dot(D1)


## Récupération des chaînes harmoniques
lambdas = lin.eigsh(L, which='SM', k=L.shape[0]-1, return_eigenvectors=False)
betti = np.sum(lambdas < 1e-10)
H = lin.eigsh(L, which='SM', k=betti)[1]


## Enregistrement des Harmoniques
with open(output, 'w') as file:
    json.dump({'dimension': H.shape[1],
               'chains': [[{'id': i, 'weight': w}
                            for i, w in enumerate(chain)]
                           for chain in H.T]},
              file,
              separators=(',', ':'),
              indent=2)

