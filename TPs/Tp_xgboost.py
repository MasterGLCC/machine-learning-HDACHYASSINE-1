# TP XGBoost — Prédiction du diabète
# On utilise le dataset Pima Indians : 768 patientes, 8 features médicales
# Le but : prédire si une patiente est diabétique (1) ou non (0)

import sys
sys.stdout.reconfigure(encoding='utf-8')

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import xgboost as xgb

# ── 1. Chargement des données ──────────────────────────────────

# le dataset Pima Indians : 768 patientes, 8 features médicales
# le but est de prédire si une patiente est diabétique (1) ou non (0)

url = ("https://raw.githubusercontent.com/jbrownlee/Datasets"
       "/master/pima-indians-diabetes.data.csv")
colonnes = ['Grossesses','Glucose','PressionArterielle','EpaisseurPeau',
            'Insuline','IMC','FonctionPedigree','Age','Diabete']

try:
    data = pd.read_csv(url, names=colonnes)
except Exception:
    # au cas où y'a pas internet, je génère des données similaires
    np.random.seed(42)
    n = 500
    data = pd.DataFrame({
        'Grossesses': np.random.randint(0, 15, n),
        'Glucose': np.random.randint(70, 200, n),
        'PressionArterielle': np.random.randint(40, 120, n),
        'EpaisseurPeau': np.random.randint(10, 60, n),
        'Insuline': np.random.randint(0, 300, n),
        'IMC': np.round(np.random.uniform(18, 50, n), 1),
        'FonctionPedigree': np.round(np.random.uniform(0.1, 2.5, n), 3),
        'Age': np.random.randint(20, 80, n),
        'Diabete': np.random.randint(0, 2, n)
    })

print("Dataset chargé :", data.shape)
print(data.head())

# ── 2. Préparation ────────────────────────────────────────────

# je sépare les features de la colonne cible
X = data.drop('Diabete', axis=1).values
y = data['Diabete'].values

# 80% pour entraîner, 20% pour tester
# random_state=42 pour avoir les mêmes résultats à chaque fois
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train: {X_train.shape[0]} exemples | Test: {X_test.shape[0]} exemples")

# ── 3. DMatrix — format interne de XGBoost ───────────────────

# XGBoost a son propre format de données, c'est plus optimisé que numpy directement
# il pré-trie les valeurs pour accélérer la construction des arbres
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest  = xgb.DMatrix(X_test,  label=y_test)

# ── 4. Paramètres du modèle ───────────────────────────────────

params = {
    # classification binaire, la sortie sera une probabilité entre 0 et 1
    'objective':   'binary:logistic',
    'eval_metric': 'logloss',   # on surveille la log-loss pendant l'entraînement

    # profondeur max de chaque arbre — si trop grand ça va overfitter
    'max_depth': 4,

    # le learning rate, plus c'est petit plus l'apprentissage est prudent
    'eta': 0.1,

    # à chaque arbre on prend seulement 80% des données (comme random forest)
    # ça aide à ne pas trop coller aux données d'entraînement
    'subsample': 0.8,

    # pareil mais pour les features, on en prend 80% aléatoirement par arbre
    'colsample_bytree': 0.8,

    # une feuille doit avoir au moins ce poids pour exister
    'min_child_weight': 1,

    # gain minimum pour qu'une division soit acceptée — 0 = pas de contrainte
    'gamma': 0.1,

    # régularisation L2 sur les poids des feuilles, ça évite les valeurs extrêmes
    'reg_lambda': 1.0,

    # reproductibilité + utiliser tous les coeurs dispo
    'seed': 42,
    'nthread': -1,
}

# ── 5. Entraînement ───────────────────────────────────────────

print("\n" + "="*45)
print("  Entraînement XGBoost")
print("="*45)

# je surveille train et validation en même temps pour voir si ça diverge
evals = [(dtrain, 'train'), (dtest, 'val')]

modele = xgb.train(
    params,
    dtrain,
    num_boost_round=200,       # max 200 arbres
    evals=evals,
    early_stopping_rounds=20,  # si pas d'amélioration pendant 20 rounds on arrête
    verbose_eval=20            # afficher toutes les 20 itérations
)

print(f"\nMeilleure itération trouvée : {modele.best_iteration}")

# ── 6. Prédictions ────────────────────────────────────────────

# predict retourne des probabilités, pas directement des classes
proba_test = modele.predict(dtest)

# je mets un seuil à 0.5 pour convertir en 0 ou 1
y_pred = (proba_test >= 0.5).astype(int)

# ── 7. Évaluation ─────────────────────────────────────────────

print("\n--- Résultats sur le jeu de test ---")
print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")

print("\nRapport complet :")
print(classification_report(y_test, y_pred,
      target_names=['Non diabétique', 'Diabétique']))

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
print("Matrice de confusion :")
print(f"  TN={tn}  FP={fp}")
print(f"  FN={fn}  TP={tp}")
# les FN sont les plus dangereux ici : ce sont les diabétiques qu'on a rates

# ── 8. Importance des features ────────────────────────────────

# XGBoost nous dit quelles features il a le plus utilisées dans ses arbres
importance = modele.get_fscore()
importance_triee = sorted(importance.items(), key=lambda x: x[1], reverse=True)

noms = colonnes[:-1]  # sans la colonne Diabete
print("\nFeatures les plus importantes :")
for feat_id, score in importance_triee:
    idx = int(feat_id[1:])
    print(f"  {noms[idx]:25s} : {score}")

# ── 9. Interface scikit-learn (plus simple) ───────────────────

# XGBoost a aussi une interface compatible sklearn, pratique pour GridSearchCV
print("\n--- Avec l'interface sklearn ---")

modele_sk = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss',
    random_state=42
)

modele_sk.fit(X_train, y_train)
y_pred_sk = modele_sk.predict(X_test)
print(f"Accuracy : {accuracy_score(y_test, y_pred_sk):.4f}")

# predict_proba donne [P(classe 0), P(classe 1)] pour chaque exemple
proba_sk = modele_sk.predict_proba(X_test)
print(f"Exemple 0 - non-diabetique: {proba_sk[0,0]:.2f} | diabetique: {proba_sk[0,1]:.2f}")

# ── 10. Recherche des meilleurs hyperparamètres ───────────────

# j'essaie plusieurs combinaisons pour trouver les meilleurs paramètres
# 3x2x2 = 12 combinaisons x 5 folds = 60 entraînements au total
print("\n--- GridSearchCV ---")

grille = {
    'max_depth':     [3, 4, 5],
    'learning_rate': [0.05, 0.1],
    'n_estimators':  [50, 100]
}

recherche = GridSearchCV(
    xgb.XGBClassifier(
        eval_metric='logloss',
        random_state=42
    ),
    grille,
    cv=5,           # validation croisée 5 plis
    scoring='accuracy',
    n_jobs=-1
)
recherche.fit(X_train, y_train)

print(f"Meilleurs parametres : {recherche.best_params_}")
print(f"Meilleure accuracy CV : {recherche.best_score_:.4f}")
print(f"Accuracy sur test     : {accuracy_score(y_test, recherche.best_estimator_.predict(X_test)):.4f}")