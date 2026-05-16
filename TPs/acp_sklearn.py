# ============================================================
#  ACP AVEC SCIKIT-LEARN - Dataset Iris
#  Version simplifiee grace a la librairie
# ============================================================

# --- Importations ---
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches  # Pour creer des legendes personnalisees
from matplotlib.patches import Ellipse

# StandardScaler : standardise les donnees (centrage + reduction)
from sklearn.preprocessing import StandardScaler

# PCA : classe qui implemente l'ACP en interne (meme algorithme qu'au scratch)
from sklearn.decomposition import PCA

# Le dataset Iris integre dans scikit-learn
from sklearn.datasets import load_iris


# ETAPE 0 - Chargement du dataset

iris = load_iris()
X = iris.data          # Donnees brutes : 150 x 4
y = iris.target        # Etiquettes des especes : 0, 1, ou 2
labels = iris.target_names   # ['setosa', 'versicolor', 'virginica']
features = iris.feature_names  # Les 4 noms de variables

print(f"Donnees : {X.shape[0]} observations, {X.shape[1]} variables")
print(f"Variables : {list(features)}\n")


# ETAPE 1 - Standardisation avec StandardScaler

# Identique au centrage/reduction manuel, mais gere automatiquement.
# StandardScaler calcule la moyenne et l'ecart-type sur X,
# puis transforme chaque valeur : (x - mean) / std

scaler = StandardScaler()  # Cree l'objet StandardScaler

# fit_transform(X) fait deux choses en un appel :
#   - fit(X)      : calcule la moyenne et l'ecart-type de chaque colonne
#   - transform(X): applique la standardisation
X_scaled = scaler.fit_transform(X)  # Retourne un array (150, 4) standardise

print("Verification apres StandardScaler :")
print(f"  Moyennes (~= 0) : {np.round(X_scaled.mean(axis=0), 10)}")
print(f"  Ecarts-types (~= 1) : {np.round(X_scaled.std(axis=0), 10)}\n")

# ============================================================
# ETAPE 2 - Application de l'ACP
# ============================================================
# PCA(n_components=2) signifie qu'on veut garder les 2 premieres
# composantes principales (reduction de 4D -> 2D).
#
# En interne, scikit-learn fait exactement ce qu'on a code from scratch :
# matrice de covariance -> decomposition en valeurs propres -> tri -> projection.

pca = PCA(n_components=2)  # Cree l'objet PCA avec 2 composantes

# fit_transform :
#   - fit(X_scaled)      : calcule les composantes principales
#   - transform(X_scaled): projette les donnees dans le nouvel espace
X_pca = pca.fit_transform(X_scaled)  # Retourne un array (150, 2)

print(f"Donnees apres ACP : {X_pca.shape}")

# ============================================================
# ETAPE 3 - Analyse des resultats
# ============================================================

# explained_variance_ratio_ : proportion de variance expliquee par chaque PC
ev = pca.explained_variance_ratio_

print("\nVariance expliquee :")
print(f"  PC1 : {ev[0]*100:.2f}%")
print(f"  PC2 : {ev[1]*100:.2f}%")
print(f"  Total (PC1+PC2) : {(ev[0]+ev[1])*100:.2f}%\n")

# components_ : les vecteurs propres (axes principaux)
# Chaque ligne est un vecteur propre dans l'espace original (4D)
print("Vecteurs propres (loadings) :")
print("(Chaque colonne = contribution des 4 variables originales a une PC)")
loading_matrix = pca.components_.T  # Transposee : lignes = variables, colonnes = PCs
for feat, load in zip(features, loading_matrix):
    print(f"  {feat[:20]:<22}: PC1={load[0]:+.3f}  PC2={load[1]:+.3f}")
print()

# explained_variance_ : les valeurs propres brutes
print(f"Valeurs propres brutes : {np.round(pca.explained_variance_, 4)}\n")

# ============================================================
# ETAPE 4 - Visualisation complete
# ============================================================
colors = ['#E74C3C', '#2ECC71', '#3498DB']
markers = ['o', 's', '^']  # Cercle, carre, triangle par espece

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("ACP avec Scikit-learn - Dataset Iris", fontsize=14, fontweight='bold')

# --- Graphique 1 : Scatter plot des composantes ---
ax1 = axes[0]
for i, (label, color, marker) in enumerate(zip(labels, colors, markers)):
    mask = (y == i)
    ax1.scatter(X_pca[mask, 0],
                X_pca[mask, 1],
                c=color, marker=marker, label=label,
                alpha=0.8, edgecolors='white', s=70)

# Encercler les clusters pour visualiser la separation
for i, color in enumerate(colors):
    mask = (y == i)
    pts = X_pca[mask]
    cx, cy = pts.mean(axis=0)
    sx, sy = pts.std(axis=0)
    ellipse = Ellipse((cx, cy), width=sx*4, height=sy*4,
                      edgecolor=color, facecolor=color, alpha=0.1, linewidth=2)
    ax1.add_patch(ellipse)

ax1.set_xlabel(f"PC1 ({ev[0]*100:.1f}% variance)", fontsize=11)
ax1.set_ylabel(f"PC2 ({ev[1]*100:.1f}% variance)", fontsize=11)
ax1.set_title("Projection 2D des donnees")
ax1.legend(title="Espece")
ax1.grid(True, alpha=0.3)

# --- Graphique 2 : Scree plot ---
ax2 = axes[1]
pca_full = PCA()  # ACP avec toutes les composantes pour le scree plot
pca_full.fit(X_scaled)
ev_full = pca_full.explained_variance_ratio_
cum_ev = np.cumsum(ev_full)
pc_labels = [f"PC{i+1}" for i in range(len(ev_full))]

bars = ax2.bar(pc_labels, ev_full * 100, color='#3498DB', alpha=0.8, zorder=2)
ax2.plot(pc_labels, cum_ev * 100, 'o-', color='#E74C3C',
         linewidth=2.5, markersize=8, label='Variance cumulee', zorder=3)
ax2.axhline(y=95, color='#27AE60', linestyle='--', linewidth=1.5, label='Seuil 95%')
ax2.axhline(y=80, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='Seuil 80%')

for bar, ratio in zip(bars, ev_full):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{ratio*100:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax2.set_xlabel("Composante principale")
ax2.set_ylabel("Variance expliquee (%)")
ax2.set_title("Scree Plot")
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3, axis='y', zorder=1)

# --- Graphique 3 : Biplot (donnees + loadings des variables originales) ---
ax3 = axes[2]

# Affiche les points projetes
for i, (label, color) in enumerate(zip(labels, colors)):
    mask = (y == i)
    ax3.scatter(X_pca[mask, 0], X_pca[mask, 1],
                c=color, alpha=0.3, s=40)

# Affiche les fleches des variables originales (loadings)
# L'amplitude de la fleche indique l'importance de la variable
scale = 3  # Facteur d'echelle pour rendre les fleches visibles
loadings = pca.components_.T  # (4 variables, 2 PCs)

for j, feat_name in enumerate(features):
    dx = loadings[j, 0] * scale  # Composante sur PC1
    dy = loadings[j, 1] * scale  # Composante sur PC2
    ax3.annotate('', xy=(dx, dy), xytext=(0, 0),
                 arrowprops=dict(arrowstyle='->', color='#E74C3C', lw=2))
    short_name = feat_name.replace(' (cm)', '').replace('sepal ', 'sep. ').replace('petal ', 'pet. ')
    ax3.text(dx * 1.15, dy * 1.15, short_name,
             fontsize=8, color='#C0392B', ha='center', fontweight='bold')

ax3.axhline(0, color='gray', linewidth=0.5, alpha=0.5)
ax3.axvline(0, color='gray', linewidth=0.5, alpha=0.5)
ax3.set_xlabel(f"PC1 ({ev[0]*100:.1f}%)")
ax3.set_ylabel(f"PC2 ({ev[1]*100:.1f}%)")
ax3.set_title("Biplot : donnees + variables originales")
ax3.grid(True, alpha=0.2)

patches = [mpatches.Patch(color=c, label=l) for c, l in zip(colors, labels)]
ax3.legend(handles=patches, fontsize=8, loc='lower right')

plt.tight_layout()
plt.savefig('acp_sklearn.png', dpi=150, bbox_inches='tight')
plt.show()
print("Graphique sauvegarde : acp_sklearn.png")


# ETAPE 5 - Reconstruction (inverse transform)

# On peut reconstruire une approximation des donnees originales
# a partir des 2 composantes. Plus on garde de composantes, plus
# la reconstruction est fidele.

X_reconstructed = pca.inverse_transform(X_pca)  # (150, 4) reconstruit depuis (150, 2)
reconstruction_error = np.mean((X_scaled - X_reconstructed) ** 2)
print(f"\nErreur de reconstruction (MSE) avec 2 PC : {reconstruction_error:.4f}")
print("(Erreur nulle = reconstruction parfaite, impossible avec reduction de dim.)")
