# -*- coding: utf-8 -*-

# Utilisation de l'algorithme de [Bri07] pour échantillonner
# un réseau de capteurs selon un échantillonnage de Poisson
# rapide.

import math
import json
import numpy as np
import numpy.random as rnd


## Paramètres
width, height = 2.0, 1.0        # Terrain de jeu
rmin = 0.1             # Distance min pour l'échantillonnage
k = 30                 # Nombre de point de test
rcon = 0.22            # Rayon de connexion pour les arêtes
rnd.seed(1234)         # Graine du GNA
output="snet.json"     # Fichier de sortie

# Liste de fonction indiquant si le point z est dans la zone
filters = [lambda z: 0 <= z.real < width,
           lambda z: 0 <= z.imag < height,
           lambda z: np.abs(z - 0.5 - 0.5j) > .25,
           lambda z: np.abs(z - 1.5 - 0.5j) > .22]



### Algorithme de [Bri07]

## Mise en place de la grille
nrows = int(np.ceil(height * np.sqrt(2) / rmin))
ncols = int(np.ceil(width * np.sqrt(2) / rmin))

cellwidth = width / ncols
cellheight = height / nrows

grid = np.empty((nrows, ncols), dtype=np.complex)
grid[:] = np.infty * (1+1j)


## Point initial
z =  np.infty
while not all(f(z) for f in filters):
    z =  width * rnd.random() + 1j * height * rnd.random()
    
grid[int(z.imag / cellheight), int(z.real / cellwidth)] = z

locations = [z]
actives = [0]


## Ajout de points tant que possible
while actives:
    # Sélection d'un point actifs
    i = rnd.randint(len(actives))
    actives[i], actives[-1] = actives[-1], actives[i]
    p = locations[actives[-1]]

    # Tirage de k voisins dans la couronne autour de p de rayons [r, 2r]
    qs = rmin*np.sqrt(1.0+3.0*rnd.sample(k))*np.exp(2j*np.pi*rnd.sample(k))
    qs = p + rmin * np.sqrt(rnd.uniform(1.0, 3.0, k)) \
                  * np.exp(1j*rnd.uniform(0.0, 2 * np.pi, k))

    # Recherche d'un prochain point parmi les voisins
    for q in qs:
        if all(f(q) for f in filters):
            qi = int(q.imag / cellheight)
            qj = int(q.real / cellwidth)

            # Les points potentiellement trop proches
            pains = grid[max(qi-2,0):min(qi+3, nrows),
                         max(qj-2,0):min(qj+3, ncols)]
            
            # Si il n'y a pas de gêneurs, on garde q
            if (np.abs(pains - q) > rmin).all():
                actives.append(len(locations))
                locations.append(q)
                grid[qi, qj] = q
                break
    else:
        actives.pop()
locations = np.asarray(locations)
nvertices = len(locations)



## Construction de la relation de voisinage pour une
## distance rcon. Constructions des edges par la même
## occasion
neighbors = {i: [] for i in range(nvertices)}
edges = []
for u in range(nvertices):
    for v in range(1+u, nvertices):
        if abs(locations[u] - locations[v]) < rcon:
            neighbors[u].append(len(edges))
            edges.append((u, v))
nedges = len(edges)

triangles = []
triangles_vertices = []
for u in range(nvertices):
    for uv in neighbors[u]:
        v = edges[uv][1]
        for vw in neighbors[v]:
            w = edges[vw][1]
            for ux in neighbors[u]:
                x = edges[ux][1] 
                if x == w:
                    triangles.append((uv, ux, vw))
                    triangles_vertices.append((u, v, w))
ntriangles = len(triangles)



## Sauvegarde du réseau sous forme JSON
with open(output, 'w') as file:
    json.dump({'size': [width, height],
               'vertices': [{'id': i, 'location': [ z.real, z.imag ]}
                            for i, z in enumerate(locations)],
               'edges': [{'id':i, 'vertices': e}
                         for i, e in enumerate(edges)],
               'triangles': [{'id': i, 'edges': t, 'vertices': triangles_vertices[i]}
                             for i, t in enumerate(triangles)]},
              file,
              separators=(',', ':'),
              indent=2)
