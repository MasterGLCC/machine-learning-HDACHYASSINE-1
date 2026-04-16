# =============================================================================
# TP RÉGRESSION COMPLÈTE - PYTHON
# Scénario : Prédiction du prix de maisons et classification de leur type
# Dataset : Données immobilières simulées (surface, chambres, âge, prix...)
#
# Contenu :
#   1. Régression Linéaire Simple    (from scratch + sklearn)
#   2. Régression Linéaire Multiple  (from scratch + sklearn)
#   3. Régression Polynomiale        (from scratch + sklearn)
#   4. Régression Logistique         (from scratch + sklearn)
# =============================================================================

# --- IMPORTATION DES BIBLIOTHÈQUES ---
import numpy as np                        # Pour les calculs numériques (matrices, vecteurs)
import matplotlib.pyplot as plt           # Pour les visualisations / graphiques
import pandas as pd                       # Pour la gestion des données en tableaux
from sklearn.linear_model import LinearRegression, LogisticRegression  # Modèles sklearn
from sklearn.preprocessing import PolynomialFeatures, StandardScaler   # Preprocessing
from sklearn.model_selection import train_test_split                    # Découpage train/test
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix  # Métriques

# --- CONFIGURATION AFFICHAGE ---
np.random.seed(42)       # Fixer le hasard pour avoir des résultats reproductibles
plt.style.use('seaborn-v0_8-whitegrid')  # Style visuel propre pour les graphiques


# =============================================================================
# GÉNÉRATION DU DATASET : DONNÉES IMMOBILIÈRES
# =============================================================================

n = 200  # Nombre de maisons dans notre dataset

# Surface habitable en m² (entre 30 et 200 m²)
surface = np.random.uniform(30, 200, n)

# Nombre de chambres (entre 1 et 5)
chambres = np.random.randint(1, 6, n)

# Âge de la maison en années (entre 0 et 50 ans)
age = np.random.randint(0, 51, n)

# Distance au centre-ville en km (entre 1 et 30 km)
distance = np.random.uniform(1, 30, n)

# --- Calcul du prix RÉEL (avec une formule + bruit) ---
# Le prix dépend de : surface (+fort), chambres (+moyen), âge (-faible), distance (-moyen)
bruit = np.random.normal(0, 15000, n)   # Bruit aléatoire pour simuler la réalité
prix = (1200 * surface) + (8000 * chambres) - (500 * age) - (2000 * distance) + 50000 + bruit

# --- Création du DataFrame (tableau de données) ---
df = pd.DataFrame({
    'surface': surface,
    'chambres': chambres,
    'age': age,
    'distance': distance,
    'prix': prix
})

# Affichage d'un aperçu du dataset
print("=" * 60)
print("APERÇU DU DATASET (5 premières lignes)")
print("=" * 60)
print(df.head())
print(f"\nNombre de maisons : {n}")
print(f"Prix moyen : {prix.mean():.0f} €")
print(f"Prix min / max : {prix.min():.0f} € / {prix.max():.0f} €")


# =============================================================================
# PARTIE 1 : RÉGRESSION LINÉAIRE SIMPLE
# Objectif : Prédire le PRIX uniquement à partir de la SURFACE
# =============================================================================

print("\n" + "=" * 60)
print("1. RÉGRESSION LINÉAIRE SIMPLE")
print("=" * 60)

# --- Préparation des données ---
X_simple = surface        # Variable explicative (entrée)
Y_simple = prix           # Variable cible (sortie)

# Découpage 80% entraînement, 20% test
X_train_s, X_test_s, Y_train_s, Y_test_s = train_test_split(
    X_simple, Y_simple, test_size=0.2, random_state=42
)

# -------------------------------------------------------
# FROM SCRATCH : Moindres carrés ordinaires
# Formule : y = b0 + b1 * x
# b1 = Σ(xi - x̄)(yi - ȳ) / Σ(xi - x̄)²
# b0 = ȳ - b1 * x̄
# -------------------------------------------------------

def regression_lineaire_simple(x_train, y_train, x_test):
    """Calcul des coefficients d'une régression linéaire simple."""
    x_moy = np.mean(x_train)   # Moyenne des x d'entraînement
    y_moy = np.mean(y_train)   # Moyenne des y d'entraînement

    # Calcul du coefficient directeur b1
    numerateur   = np.sum((x_train - x_moy) * (y_train - y_moy))
    denominateur = np.sum((x_train - x_moy) ** 2)
    b1 = numerateur / denominateur   # Pente de la droite

    # Calcul de l'ordonnée à l'origine b0
    b0 = y_moy - b1 * x_moy

    # Prédiction sur les données de test
    y_pred = b0 + b1 * x_test

    return b0, b1, y_pred

# Appel de la fonction
b0_scratch, b1_scratch, y_pred_scratch = regression_lineaire_simple(
    X_train_s, Y_train_s, X_test_s
)

# Calcul des métriques d'évaluation
mse_scratch = mean_squared_error(Y_test_s, y_pred_scratch)   # Erreur quadratique moyenne
r2_scratch  = r2_score(Y_test_s, y_pred_scratch)             # Coefficient de détermination

print(f"\n[FROM SCRATCH]")
print(f"  b0 (intercept) = {b0_scratch:.2f}")
print(f"  b1 (pente)     = {b1_scratch:.2f}")
print(f"  MSE            = {mse_scratch:.2f}")
print(f"  R²             = {r2_scratch:.4f}")

# -------------------------------------------------------
# AVEC SKLEARN
# -------------------------------------------------------

# LinearRegression attend un tableau 2D, donc on reshape (-1, 1) = une colonne
modele_lr = LinearRegression()
modele_lr.fit(X_train_s.reshape(-1, 1), Y_train_s)         # Entraînement
y_pred_lr = modele_lr.predict(X_test_s.reshape(-1, 1))     # Prédiction

mse_lr = mean_squared_error(Y_test_s, y_pred_lr)
r2_lr  = r2_score(Y_test_s, y_pred_lr)

print(f"\n[SKLEARN]")
print(f"  b0 (intercept) = {modele_lr.intercept_:.2f}")
print(f"  b1 (coeff)     = {modele_lr.coef_[0]:.2f}")
print(f"  MSE            = {mse_lr:.2f}")
print(f"  R²             = {r2_lr:.4f}")

# --- Visualisation ---
plt.figure(figsize=(8, 5))
plt.scatter(X_test_s, Y_test_s, alpha=0.5, label='Données réelles', color='steelblue')
x_range = np.linspace(X_test_s.min(), X_test_s.max(), 100)
plt.plot(x_range, b0_scratch + b1_scratch * x_range,
         color='tomato', linewidth=2, label='Droite from scratch')
plt.xlabel('Surface (m²)')
plt.ylabel('Prix (€)')
plt.title('Régression Linéaire Simple : Prix ~ Surface')
plt.legend()
plt.tight_layout()
plt.savefig('reg_lineaire_simple.png', dpi=100)
plt.show()


# =============================================================================
# PARTIE 2 : RÉGRESSION LINÉAIRE MULTIPLE
# Objectif : Prédire le PRIX avec TOUTES les variables (surface, chambres, âge, distance)
# =============================================================================

print("\n" + "=" * 60)
print("2. RÉGRESSION LINÉAIRE MULTIPLE")
print("=" * 60)

# --- Préparation des données ---
# On construit la matrice X avec toutes les variables explicatives
X_multiple = df[['surface', 'chambres', 'age', 'distance']].values  # Matrice (200, 4)
Y_multiple = df['prix'].values                                        # Vecteur cible (200,)

X_train_m, X_test_m, Y_train_m, Y_test_m = train_test_split(
    X_multiple, Y_multiple, test_size=0.2, random_state=42
)

# -------------------------------------------------------
# FROM SCRATCH : Équation normale
# Formule : β = (XᵀX)⁻¹ Xᵀy
# On ajoute une colonne de 1 pour le terme constant (b0)
# -------------------------------------------------------

def regression_lineaire_multiple(X_train, y_train, X_test):
    """Régression linéaire multiple via l'équation normale (forme matricielle)."""
    n_train = X_train.shape[0]   # Nombre d'exemples d'entraînement

    # Ajout d'une colonne de 1 à gauche → terme biais (b0)
    # np.ones crée une colonne de 1, np.hstack colle les colonnes ensemble
    X_b = np.hstack([np.ones((n_train, 1)), X_train])    # Forme (n, 5)

    # Équation normale : β = (XᵀX)⁻¹ Xᵀy
    # np.linalg.inv  → inverse d'une matrice
    # .T             → transposée d'une matrice
    beta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y_train

    # Prédiction sur le jeu de test
    n_test = X_test.shape[0]
    X_test_b = np.hstack([np.ones((n_test, 1)), X_test])   # Ajout colonne de 1 au test
    y_pred = X_test_b @ beta                                # Produit matriciel = prédiction

    return beta, y_pred

# Appel
beta_scratch, y_pred_mult_scratch = regression_lineaire_multiple(X_train_m, Y_train_m, X_test_m)

mse_mult_scratch = mean_squared_error(Y_test_m, y_pred_mult_scratch)
r2_mult_scratch  = r2_score(Y_test_m, y_pred_mult_scratch)

print(f"\n[FROM SCRATCH]")
print(f"  b0 (intercept) = {beta_scratch[0]:.2f}")
print(f"  b1 (surface)   = {beta_scratch[1]:.2f}")
print(f"  b2 (chambres)  = {beta_scratch[2]:.2f}")
print(f"  b3 (age)       = {beta_scratch[3]:.2f}")
print(f"  b4 (distance)  = {beta_scratch[4]:.2f}")
print(f"  MSE = {mse_mult_scratch:.2f}  |  R² = {r2_mult_scratch:.4f}")

# -------------------------------------------------------
# AVEC SKLEARN
# -------------------------------------------------------

modele_lrm = LinearRegression()
modele_lrm.fit(X_train_m, Y_train_m)
y_pred_lrm = modele_lrm.predict(X_test_m)

mse_lrm = mean_squared_error(Y_test_m, y_pred_lrm)
r2_lrm  = r2_score(Y_test_m, y_pred_lrm)

print(f"\n[SKLEARN]")
print(f"  Intercept      = {modele_lrm.intercept_:.2f}")
print(f"  Coefficients   = {modele_lrm.coef_}")
print(f"  MSE = {mse_lrm:.2f}  |  R² = {r2_lrm:.4f}")

# --- Visualisation : Valeurs réelles vs prédites ---
plt.figure(figsize=(7, 5))
plt.scatter(Y_test_m, y_pred_mult_scratch, alpha=0.5, color='mediumseagreen', label='From scratch')
plt.plot([Y_test_m.min(), Y_test_m.max()],
         [Y_test_m.min(), Y_test_m.max()],
         'r--', linewidth=2, label='Prédiction parfaite')
plt.xlabel('Prix réel (€)')
plt.ylabel('Prix prédit (€)')
plt.title('Régression Multiple : Réel vs Prédit')
plt.legend()
plt.tight_layout()
plt.savefig('reg_lineaire_multiple.png', dpi=100)
plt.show()


# =============================================================================
# PARTIE 3 : RÉGRESSION POLYNOMIALE
# Objectif : Capturer une relation NON-LINÉAIRE entre la surface et le prix
# Idée : le prix n'augmente pas de façon strictement linéaire avec la surface
# =============================================================================

print("\n" + "=" * 60)
print("3. RÉGRESSION POLYNOMIALE")
print("=" * 60)

# --- Création d'un dataset avec relation non-linéaire ---
# On génère un signal avec une courbe (quadratique) + bruit
X_poly = np.sort(np.random.uniform(20, 200, 150))   # Surface de 20 à 200 m², triée
Y_poly = 500 * X_poly + 3 * X_poly**2 - 0.01 * X_poly**3 + np.random.normal(0, 20000, 150)

X_train_p, X_test_p, Y_train_p, Y_test_p = train_test_split(
    X_poly, Y_poly, test_size=0.2, random_state=42
)

# -------------------------------------------------------
# FROM SCRATCH : Régression polynomiale degré 2
# On ajoute manuellement les colonnes x et x²
# -------------------------------------------------------

def regression_polynomiale_scratch(x_train, y_train, x_test, degre=2):
    """
    Construit manuellement la matrice des features polynomial [1, x, x², ..., x^d]
    et applique l'équation normale.
    """
    # Construction de la matrice de Vandermonde : colonnes [1, x, x², ...]
    # np.vander crée les puissances de x, increasing=True commence par x^0
    X_train_p = np.vander(x_train, N=degre + 1, increasing=True)  # Forme (n, degre+1)
    X_test_p  = np.vander(x_test,  N=degre + 1, increasing=True)

    # Équation normale : β = (XᵀX)⁻¹ Xᵀy
    beta = np.linalg.inv(X_train_p.T @ X_train_p) @ X_train_p.T @ y_train

    # Prédiction
    y_pred = X_test_p @ beta
    return beta, y_pred

# Test avec différents degrés pour comparer
plt.figure(figsize=(10, 5))
x_plot = np.linspace(X_poly.min(), X_poly.max(), 300)  # 300 points pour la courbe lisse

plt.scatter(X_poly, Y_poly, alpha=0.3, color='steelblue', s=15, label='Données')

couleurs_degres = ['tomato', 'orange', 'purple']  # Couleur par degré

for i, degre in enumerate([1, 2, 3]):
    # FROM SCRATCH
    beta_p, _ = regression_polynomiale_scratch(X_train_p, Y_train_p, X_test_p, degre=degre)
    x_plot_vander = np.vander(x_plot, N=degre + 1, increasing=True)
    y_plot = x_plot_vander @ beta_p

    _, y_pred_test_p = regression_polynomiale_scratch(X_train_p, Y_train_p, X_test_p, degre=degre)
    r2_p = r2_score(Y_test_p, y_pred_test_p)

    plt.plot(x_plot, y_plot, color=couleurs_degres[i],
             linewidth=2, label=f'Degré {degre} (R²={r2_p:.3f})')
    print(f"  Degré {degre} - R² from scratch = {r2_p:.4f}")

# -------------------------------------------------------
# AVEC SKLEARN
# -------------------------------------------------------
poly_features = PolynomialFeatures(degree=2, include_bias=True)  # Crée les colonnes x, x²
X_train_poly_sk = poly_features.fit_transform(X_train_p.reshape(-1, 1))  # Fit + transform
X_test_poly_sk  = poly_features.transform(X_test_p.reshape(-1, 1))       # Transform seul

modele_poly = LinearRegression()
modele_poly.fit(X_train_poly_sk, Y_train_p)
y_pred_poly_sk = modele_poly.predict(X_test_poly_sk)

r2_poly_sk = r2_score(Y_test_p, y_pred_poly_sk)
print(f"\n[SKLEARN] Régression polynomiale degré 2 : R² = {r2_poly_sk:.4f}")

plt.xlabel('Surface (m²)')
plt.ylabel('Prix (€)')
plt.title('Régression Polynomiale : Comparaison des degrés')
plt.legend()
plt.tight_layout()
plt.savefig('reg_polynomiale.png', dpi=100)
plt.show()


# =============================================================================
# PARTIE 4 : RÉGRESSION LOGISTIQUE
# Objectif : CLASSIFIER les maisons en "Chère" (1) ou "Pas chère" (0)
#            selon leur surface et leur distance
# Note : la régression logistique est une CLASSIFICATION, pas une régression de prix
# =============================================================================

print("\n" + "=" * 60)
print("4. RÉGRESSION LOGISTIQUE")
print("=" * 60)

# --- Préparation du problème de classification ---
# On crée une variable binaire : 1 si le prix > médiane, 0 sinon
seuil_prix = np.median(prix)   # Seuil = médiane des prix
Y_class    = (prix > seuil_prix).astype(int)  # 1 = "chère", 0 = "pas chère"
print(f"\nSeuil de classification (médiane) : {seuil_prix:.0f} €")
print(f"Nombre de maisons chères     : {Y_class.sum()}")
print(f"Nombre de maisons pas chères : {(1 - Y_class).sum()}")

# On utilise la surface et la distance comme variables d'entrée
X_log = df[['surface', 'distance']].values    # Matrice (200, 2)

# Normalisation INDISPENSABLE pour la régression logistique
scaler    = StandardScaler()   # Centrage-réduction : (x - μ) / σ
X_log_sc  = scaler.fit_transform(X_log)   # Calcul μ et σ sur X_log, puis transformation

X_train_l, X_test_l, Y_train_l, Y_test_l = train_test_split(
    X_log_sc, Y_class, test_size=0.2, random_state=42
)

# -------------------------------------------------------
# FROM SCRATCH : Descente de gradient pour la régression logistique
# Modèle : P(y=1 | x) = sigmoid(Xβ) = 1 / (1 + e^(-Xβ))
# Optimisation : on minimise la log-vraisemblance négative (cross-entropy)
# -------------------------------------------------------

def sigmoid(z):
    """Fonction sigmoid : transforme un score réel en probabilité entre 0 et 1."""
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))  # clip pour éviter overflow numérique

def regression_logistique_scratch(X_train, y_train, X_test, lr=0.1, n_iter=1000):
    """
    Entraînement par descente de gradient.
    lr     : taux d'apprentissage (learning rate)
    n_iter : nombre d'itérations de la descente
    """
    n_exemples, n_features = X_train.shape   # Dimensions de la matrice d'entraînement

    # Ajout de la colonne de biais (1) → intercept
    X_b       = np.hstack([np.ones((n_exemples, 1)), X_train])    # Forme (n, 3)
    X_test_b  = np.hstack([np.ones((X_test.shape[0], 1)), X_test])

    # Initialisation des poids à zéro
    beta = np.zeros(n_features + 1)   # Vecteur de taille (n_features + 1) = 3

    # ---- Boucle de descente de gradient ----
    for _ in range(n_iter):
        z     = X_b @ beta              # Score brut = produit matriciel
        y_hat = sigmoid(z)              # Probabilité prédite P(y=1)

        # Gradient de la cross-entropy par rapport à beta
        # gradient = Xᵀ (ŷ - y) / n
        gradient = X_b.T @ (y_hat - y_train) / n_exemples

        # Mise à jour des poids : on descend dans la direction du gradient
        beta -= lr * gradient

    # Prédiction sur le test
    proba_pred = sigmoid(X_test_b @ beta)     # Probabilités entre 0 et 1
    y_pred     = (proba_pred >= 0.5).astype(int)  # Seuil 0.5 pour la classe binaire

    return beta, y_pred, proba_pred

# Appel de la fonction
beta_log, y_pred_log_scratch, proba_scratch = regression_logistique_scratch(
    X_train_l, Y_train_l, X_test_l
)

acc_scratch = accuracy_score(Y_test_l, y_pred_log_scratch)
cm_scratch  = confusion_matrix(Y_test_l, y_pred_log_scratch)

print(f"\n[FROM SCRATCH]")
print(f"  Précision (accuracy) = {acc_scratch * 100:.2f}%")
print(f"  Matrice de confusion :\n{cm_scratch}")

# -------------------------------------------------------
# AVEC SKLEARN
# -------------------------------------------------------

modele_logreg = LogisticRegression(max_iter=1000)   # max_iter : convergence garantie
modele_logreg.fit(X_train_l, Y_train_l)
y_pred_logreg = modele_logreg.predict(X_test_l)

acc_sk = accuracy_score(Y_test_l, y_pred_logreg)
cm_sk  = confusion_matrix(Y_test_l, y_pred_logreg)

print(f"\n[SKLEARN]")
print(f"  Précision (accuracy) = {acc_sk * 100:.2f}%")
print(f"  Matrice de confusion :\n{cm_sk}")

# --- Visualisation : Frontière de décision ---
def plot_decision_boundary(X, y, modele_type='scratch', beta=None, modele_sk=None):
    """Trace la frontière de décision et les points colorés par classe."""
    h = 0.05   # Résolution de la grille

    # Grille couvrant l'espace des features
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    grid = np.c_[xx.ravel(), yy.ravel()]   # Tous les points de la grille

    if modele_type == 'scratch':
        grid_b = np.hstack([np.ones((grid.shape[0], 1)), grid])
        Z = (sigmoid(grid_b @ beta) >= 0.5).astype(int)
    else:
        Z = modele_sk.predict(grid)

    Z = Z.reshape(xx.shape)   # Reshape pour correspondre à la grille 2D

    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlGn)   # Zones colorées
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlGn,
                edgecolors='k', s=25, alpha=0.7)
    plt.xlabel('Surface (normalisée)')
    plt.ylabel('Distance (normalisée)')

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

plt.sca(axes[0])   # Sélection du premier sous-graphique
plot_decision_boundary(X_test_l, Y_test_l, modele_type='scratch', beta=beta_log)
axes[0].set_title(f'Logistique - From Scratch (acc={acc_scratch*100:.1f}%)')

plt.sca(axes[1])   # Sélection du deuxième sous-graphique
plot_decision_boundary(X_test_l, Y_test_l, modele_type='sklearn', modele_sk=modele_logreg)
axes[1].set_title(f'Logistique - Sklearn (acc={acc_sk*100:.1f}%)')

plt.tight_layout()
plt.savefig('reg_logistique.png', dpi=100)
plt.show()


# =============================================================================
# RÉCAPITULATIF FINAL DES PERFORMANCES
# =============================================================================

print("\n" + "=" * 60)
print("RÉCAPITULATIF DES PERFORMANCES")
print("=" * 60)
print(f"{'Modèle':<35} {'Métrique':<12} {'From Scratch':>14} {'Sklearn':>12}")
print("-" * 75)
print(f"{'Rég. Linéaire Simple':<35} {'R²':<12} {r2_scratch:>14.4f} {r2_lr:>12.4f}")
print(f"{'Rég. Linéaire Multiple':<35} {'R²':<12} {r2_mult_scratch:>14.4f} {r2_lrm:>12.4f}")
print(f"{'Rég. Polynomiale (deg=2)':<35} {'R²':<12} {'(voir graphe)':>14} {r2_poly_sk:>12.4f}")
print(f"{'Rég. Logistique':<35} {'Accuracy':<12} {acc_scratch*100:>13.2f}% {acc_sk*100:>11.2f}%")
print("=" * 60)
print("\nFin du TP - Tous les graphiques ont été sauvegardés.")