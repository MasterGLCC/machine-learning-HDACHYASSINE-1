"""
============================================================
  TP : KNN et SVM — avec Scikit-Learn
  Scénario : Classification de tumeurs (Bénigne / Maligne)
  Dataset   : Breast Cancer Wisconsin (sklearn.datasets)
============================================================
"""

# ──────────────────────────────────────────────
#  IMPORTATIONS
# ──────────────────────────────────────────────
import numpy as np                                            # Calculs matriciels et vectoriels
from sklearn.datasets import load_breast_cancer              # Dataset médical intégré à sklearn
from sklearn.model_selection import train_test_split         # Diviser les données
from sklearn.preprocessing import StandardScaler             # Normalisation des features
from sklearn.neighbors import KNeighborsClassifier           # KNN bibliothèque
from sklearn.svm import SVC                                  # SVM bibliothèque
from sklearn.metrics import accuracy_score, classification_report  # Métriques d'évaluation


# ══════════════════════════════════════════════
#  ÉTAPE 1 : CHARGEMENT ET PRÉTRAITEMENT DES DONNÉES
# ══════════════════════════════════════════════

# Charger le dataset Breast Cancer
# Il contient 569 échantillons, 30 features (rayon, texture, périmètre...)
# La cible (target) : 0 = malin, 1 = bénin
data = load_breast_cancer()

X = data.data    # Matrice des features (569 x 30)
y = data.target  # Vecteur des étiquettes (569 valeurs : 0 ou 1)

print("=== Informations sur le dataset ===")
print(f"Nombre d'échantillons : {X.shape[0]}")
print(f"Nombre de features    : {X.shape[1]}")
print(f"Classes               : {data.target_names}")

# Diviser les données en ensemble d'entraînement (80%) et de test (20%)
# random_state=42 garantit la reproductibilité des résultats
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Normalisation des données (très important pour KNN et SVM !)
# StandardScaler centre les données (moyenne=0, écart-type=1)
# Sans normalisation, les features avec de grandes valeurs dominent le calcul de distance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # fit = calcule moyenne/écart-type, transform = applique
X_test  = scaler.transform(X_test)       # transform seulement (pas de re-fit pour éviter le data leakage)


# ══════════════════════════════════════════════
#  PARTIE A : KNN AVEC SCIKIT-LEARN
# ══════════════════════════════════════════════

print("\n=== KNN AVEC SCIKIT-LEARN ===")

# Créer le classificateur KNN avec K=5 voisins
# n_neighbors=5 : nombre de voisins à considérer (hyperparamètre)
# metric='euclidean' : utilise la distance euclidienne pour mesurer la proximité
knn_sklearn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')

# Entraîner le modèle : KNN mémorise simplement les données (lazy learning)
# Il ne construit pas de modèle explicite, il stocke X_train et y_train
knn_sklearn.fit(X_train, y_train)

# Prédire les classes pour les données de test
# Pour chaque point de test → calcule distances → garde K voisins → vote majoritaire
y_pred_knn = knn_sklearn.predict(X_test)

# Évaluation : comparer les prédictions avec les vraies étiquettes
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print(f"Accuracy (KNN, K=5) : {accuracy_knn * 100:.2f}%")
print("\nRapport de classification :")
print(classification_report(y_test, y_pred_knn, target_names=data.target_names))


# ══════════════════════════════════════════════
#  PARTIE B : SVM AVEC SCIKIT-LEARN
# ══════════════════════════════════════════════

print("\n=== SVM AVEC SCIKIT-LEARN ===")

# Créer le SVM avec un noyau RBF (Radial Basis Function)
# kernel='rbf' : transforme les données dans un espace de dimension supérieure
#                pour séparer des classes non-linéairement séparables
# C=1.0 : paramètre de régularisation
#          grand C → moins de régularisation, frontière plus complexe
#          petit C → plus de régularisation, frontière plus simple (plus générale)
# random_state=42 : reproductibilité
svm_sklearn = SVC(kernel='rbf', C=1.0, random_state=42)

# Entraîner le modèle : le SVM cherche l'hyperplan à marge maximale
# entre les vecteurs de support des deux classes
svm_sklearn.fit(X_train, y_train)

# Prédire : chaque point est classé selon son côté par rapport à l'hyperplan
y_pred_svm = svm_sklearn.predict(X_test)

# Évaluation
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print(f"Accuracy (SVM, kernel RBF) : {accuracy_svm * 100:.2f}%")
print("\nRapport de classification :")
print(classification_report(y_test, y_pred_svm, target_names=data.target_names))


# ══════════════════════════════════════════════
#  COMPARAISON FINALE
# ══════════════════════════════════════════════

print("\n" + "="*45)
print("  COMPARAISON DES MODÈLES")
print("="*45)
print(f"  KNN  sklearn (K=5)     : {accuracy_knn * 100:.2f}%")
print(f"  SVM  sklearn (RBF)     : {accuracy_svm * 100:.2f}%")
print("="*45)
