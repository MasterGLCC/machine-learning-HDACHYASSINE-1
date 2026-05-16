
# TP Naive Bayes - Avec la bibliothèque scikit-learn

# Scénario :
#   On veut créer un programme qui détecte automatiquement
#   si un email est un SPAM ou un email normal (HAM).
#   Pour ça, on va utiliser les mots présents dans l'email.
#   Par exemple, si un email contient "gratuit", "argent",
#   "gagnez", il y a de fortes chances que ce soit un spam.
#   Le modèle va apprendre ces associations tout seul à partir
#   d'exemples qu'on lui donne pendant l'entraînement.


from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np


# nos emails avec leurs étiquettes (1 = spam, 0 = ham)
textes = [
    "gagnez argent gratuit maintenant",
    "offre spéciale achetez maintenant promotion",
    "gagner prix argent facile",
    "offre gratuite cliquez ici",
    "promotion achetez maintenant",
    "argent rapide gain facile offre",
    "cliquez gagnez récompense gratuite",
    "réunion demain au bureau",
    "compte rendu projet envoyé",
    "déjeuner demain disponible",
    "réunion annulée bureau fermé",
    "projet terminé rapport envoyé",
    "planning semaine prochaine disponible",
    "document partagé pour révision",
]

etiquettes = [1, 1, 1, 1, 1, 1, 1,   # spam
              0, 0, 0, 0, 0, 0, 0]    # ham


# on sépare les données : 70% pour entraîner, 30% pour tester
# random_state=42 permet d'avoir les mêmes résultats à chaque exécution
X_train, X_test, y_train, y_test = train_test_split(
    textes, etiquettes,
    test_size=0.3,
    random_state=42
)

print(f"Emails d'entrainement : {len(X_train)}")
print(f"Emails de test        : {len(X_test)}")


# le modèle ne comprend pas les mots directement, il a besoin de chiffres
# CountVectorizer transforme chaque email en un vecteur qui compte les mots
# ex : "gagnez argent" → [1, 0, 1, 0, ...] selon le vocabulaire appris
vectoriseur = CountVectorizer()

# fit_transform sur le train : il apprend le vocabulaire ET transforme les textes
X_train_vec = vectoriseur.fit_transform(X_train)

# sur le test on fait seulement transform, pas fit
# c'est important : on ne doit pas laisser le modèle voir les données de test avant
X_test_vec = vectoriseur.transform(X_test)

print(f"\nVocabulaire appris : {len(vectoriseur.vocabulary_)} mots")
print(f"Taille matrice train : {X_train_vec.shape[0]} emails x {X_train_vec.shape[1]} mots")


# on crée le modèle Naive Bayes multinomial, adapté aux comptages de mots
# alpha=1.0 c'est le lissage de Laplace, pour éviter les probabilités nulles
modele = MultinomialNB(alpha=1.0)

# fit() calcule P(classe) et P(mot|classe) à partir des données d'entraînement
modele.fit(X_train_vec, y_train)

print("\nModele entraine !")


# predict() choisit la classe avec la probabilité postérieure la plus haute
y_pred = modele.predict(X_test_vec)

# predict_proba() donne les probabilités exactes pour chaque classe
y_proba = modele.predict_proba(X_test_vec)


noms_classes = ["ham", "spam"]

print("\n" + "=" * 50)
print("  Resultats")
print("=" * 50)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy : {accuracy * 100:.1f}%")

# le rapport donne precision, recall et f1-score pour chaque classe
# precision  : sur tous les emails classés spam, combien étaient vraiment spam
# recall     : sur tous les vrais spam, combien on a réussi à trouver
# f1-score   : moyenne entre precision et recall
print("\nRapport de classification :")
print(classification_report(y_test, y_pred, target_names=noms_classes))

# la matrice de confusion montre les vrais/faux positifs et négatifs
cm = confusion_matrix(y_test, y_pred)
print("Matrice de confusion :")
print(f"           Predit ham  Predit spam")
for i, row in enumerate(cm):
    print(f"Reel {noms_classes[i]:4s}   {row[0]:9}  {row[1]:9}")


# test sur des nouveaux emails que le modèle n'a jamais vus
nouveaux_emails = [
    "gagnez prix gratuit argent maintenant",
    "réunion bureau projet semaine prochaine",
    "offre exclusive cliquez maintenant",
    "rapport disponible pour révision",
]

# on utilise le même vectoriseur qu'avant, pas un nouveau
nouveaux_vec = vectoriseur.transform(nouveaux_emails)

predictions  = modele.predict(nouveaux_vec)
probabilites = modele.predict_proba(nouveaux_vec)

print("\nPredictions sur nouveaux emails :")
print("-" * 50)
for email, pred, proba in zip(nouveaux_emails, predictions, probabilites):
    label = noms_classes[pred]
    print(f"\n  Email : '{email}'")
    print(f"    P(ham)  = {proba[0]:.4f}")
    print(f"    P(spam) = {proba[1]:.4f}")
    print(f"  => {label.upper()}")


# on regarde quels mots sont les plus caractéristiques de chaque classe
# feature_log_prob_ contient log P(mot|classe) pour chaque mot du vocabulaire
print("\nTop 5 mots par classe :")
mots = vectoriseur.get_feature_names_out()

for i, classe in enumerate(noms_classes):
    log_probs  = modele.feature_log_prob_[i]
    top_idx    = np.argsort(log_probs)[::-1][:5]  # les 5 indices avec le log-prob le plus haut
    top_mots   = [(mots[j], np.exp(log_probs[j])) for j in top_idx]
    print(f"\n  {classe} :")
    for mot, prob in top_mots:
        print(f"    '{mot}' -> P={prob:.4f}")
