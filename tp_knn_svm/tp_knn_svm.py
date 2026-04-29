"""
TP : Classification de tumeurs avec KNN et SVM
Dataset : Breast Cancer Wisconsin
On veut prédire si une tumeur est bénigne ou maligne
à partir de 30 caractéristiques mesurées sur des cellules.
"""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report


# ── Chargement des données 

data = load_breast_cancer()

X = data.data    # 569 patients, chacun décrit par 30 mesures
y = data.target  # 0 = maligne, 1 = bénigne

print("=== Dataset ===")
print(f"Échantillons : {X.shape[0]}")
print(f"Features     : {X.shape[1]}")
print(f"Classes      : {data.target_names}")

# On coupe : 80% pour entraîner, 20% pour tester
# random_state=42 pour avoir les mêmes résultats à chaque exécution
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# KNN et SVM sont très sensibles aux échelles des features.
# Par exemple, une feature qui vaut 1000 va écraser une autre qui vaut 0.01
# dans le calcul de distance. StandardScaler règle ce problème :
# il ramène tout à moyenne=0 et écart-type=1.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # on apprend les stats sur le train...
X_test  = scaler.transform(X_test)       # ...et on les applique au test (sans re-apprendre)


# ── KNN ────────────────────────────────────────────────────────────────────

print("\n=== KNN ===")

# K=5 : pour classer un nouveau patient, on regarde ses 5 voisins les plus proches
# et on vote. La classe majoritaire gagne.
knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')

# KNN ne fait rien ici vraiment — il mémorise juste les données.
# Le vrai travail se passe au moment de predict().
knn.fit(X_train, y_train)

y_pred_knn = knn.predict(X_test)

print(f"Accuracy : {accuracy_score(y_test, y_pred_knn) * 100:.2f}%")
print(classification_report(y_test, y_pred_knn, target_names=data.target_names))


# ── SVM ────────────────────────────────────────────────────────────────────

print("\n=== SVM ===")

# Le SVM cherche la frontière qui sépare les deux classes avec le plus grand écart possible.
# kernel='rbf' lui permet de tracer des frontières courbes (pas juste une ligne droite),
# ce qui est utile quand les données ne sont pas séparables linéairement.
# C=1.0 contrôle le compromis entre bien classer les points d'entraînement
# et garder une frontière simple — une valeur de 1 est un bon point de départ.
svm = SVC(kernel='rbf', C=1.0, random_state=42)

svm.fit(X_train, y_train)

y_pred_svm = svm.predict(X_test)

print(f"Accuracy : {accuracy_score(y_test, y_pred_svm) * 100:.2f}%")
print(classification_report(y_test, y_pred_svm, target_names=data.target_names))


# ── Résumé ─────────────────────────────────────────────────────────────────

print("\n" + "="*40)
print("  RÉSULTATS FINAUX")
print("="*40)
print(f"  KNN (K=5)   : {accuracy_score(y_test, y_pred_knn) * 100:.2f}%")
print(f"  SVM (RBF)   : {accuracy_score(y_test, y_pred_svm) * 100:.2f}%")
print("="*40)
