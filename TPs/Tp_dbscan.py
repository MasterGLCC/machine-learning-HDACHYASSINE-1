# Scénario : Détection de zones géographiques urbaines
# Imaginons que tu travailles pour une ville intelligente. 
# Tu disposes de données GPS de 500 points représentant des arrêts de taxis. 
# L'objectif est de regrouper automatiquement ces arrêts en zones (centre-ville, aéroport, gare, périphérie...) 
# sans connaître le nombre de zones à l'avance, et d'identifier les points isolés (arrêts anormaux).



import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Simulation d'arrêts de taxis dans différentes zones de la ville
cluster1 = np.random.randn(150, 2) * 0.5 + [2, 2]   # centre-ville
cluster2 = np.random.randn(100, 2) * 0.4 + [6, 5]   # aéroport
cluster3 = np.random.randn(80, 2) * [1.5, 0.3] + [4, 8]  # gare (forme allongée)
bruit    = np.random.uniform(0, 10, (30, 2))          # arrêts isolés

X = np.vstack([cluster1, cluster2, cluster3, bruit])


# ============================================================
# DBSCAN avec scikit-learn
# ============================================================

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Normalisation : moyenne=0, écart-type=1
# important pour que epsilon soit cohérent dans toutes les dimensions
scaler = StandardScaler()
X_normalise = scaler.fit_transform(X)

# eps    : rayon du voisinage autour de chaque point
# min_samples : nombre min de voisins pour être considéré point noyau
modele = DBSCAN(eps=0.4, min_samples=5, metric='euclidean')

# fit_predict entraîne et retourne les labels en une seule étape
# labels >= 0 → numéro du cluster
# labels == -1 → bruit (outlier)
labels = modele.fit_predict(X_normalise)

# index et coordonnées des points noyaux détectés
indices_noyaux = modele.core_sample_indices_
points_noyaux  = modele.components_


# ============================================================
# Visualisation
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# données brutes avant clustering
axes[0].scatter(X[:, 0], X[:, 1], c='steelblue', s=20, alpha=0.5)
axes[0].set_title('Données brutes (arrêts de taxis)')
axes[0].set_xlabel('Longitude')
axes[0].set_ylabel('Latitude')
axes[0].grid(True, alpha=0.3)

# résultats DBSCAN
couleurs = plt.cm.tab10(np.linspace(0, 1, max(labels) + 2))

for cluster_id in set(labels):
    masque = labels == cluster_id
    if cluster_id == -1:
        axes[1].scatter(X[masque, 0], X[masque, 1],
                        c='black', marker='x', s=60,
                        linewidths=1.5, label='Bruit', zorder=5)
    else:
        axes[1].scatter(X[masque, 0], X[masque, 1],
                        color=couleurs[cluster_id], s=40,
                        alpha=0.8, label=f'Zone {cluster_id}')

# on remet les points noyaux dans l'échelle d'origine pour les afficher
X_noyaux_original = scaler.inverse_transform(points_noyaux)
axes[1].scatter(X_noyaux_original[:, 0], X_noyaux_original[:, 1],
                s=80, facecolors='none', edgecolors='red',
                linewidths=0.8, label='Points noyaux', zorder=6, alpha=0.4)

axes[1].set_title('Résultat DBSCAN')
axes[1].set_xlabel('Longitude')
axes[1].set_ylabel('Latitude')
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

plt.suptitle('DBSCAN — Détection automatique de zones urbaines', fontsize=14)
plt.tight_layout()
plt.show()


# ============================================================
# Résumé
# ============================================================

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_bruit    = np.sum(labels == -1)

print(f"Clusters trouvés : {n_clusters}")
print(f"Points de bruit  : {n_bruit} ({n_bruit/len(X)*100:.1f}%)")
for c in sorted(set(labels) - {-1}):
    print(f"  Zone {c} : {np.sum(labels == c)} arrêts")