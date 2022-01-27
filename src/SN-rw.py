# -*- coding: utf-8 -*-

# Utilisation d'une marche aléatoire pour la détection des
# trous de couverture via les chaînes harmoniques.


## Paramètres
snfile="snet.json"     # Fichier du réseau
hfile="snet-harm.json" # Fichier de l'harmonique


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


with open(hfile, 'r') as f:
    data = json.load(f)
    chains = np.zeros((nedges, data['dimension']))
    for i, weights in enumerate(data['chains']):
        for e in weights:
            chains[e['id'], i] = e['weight']
