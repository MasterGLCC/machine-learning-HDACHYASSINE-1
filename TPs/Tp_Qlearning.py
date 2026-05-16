# ============================================================
#  TP Q-Learning — Version avec Bibliothèque
#  Scénario : Robot livreur dans un entrepôt (FrozenLake 4x4)
#  Bibliothèques : Gymnasium + Stable-Baselines3 (DQN)
# ============================================================

# ── Installation (à faire une seule fois dans le terminal) ──
# pip install gymnasium stable-baselines3

import gymnasium as gym                                        # la bibliothèque d'environnements RL standard
from stable_baselines3 import DQN                             # Deep Q-Network : version moderne du Q-learning
from stable_baselines3.common.evaluation import evaluate_policy  # pour évaluer la politique apprise

# ─── 1. Créer l'environnement FrozenLake ────────────────────
# FrozenLake = grille 4x4 identique à notre scénario entrepôt
# S=départ | F=case libre | H=trou (-1) | G=cible (+1)
env = gym.make(
    'FrozenLake-v1',          # nom de l'environnement Gymnasium
    is_slippery=False,        # désactive le glissement aléatoire → déterministe
    render_mode='rgb_array'   # nécessaire pour l'évaluation visuelle
)

# ─── 2. Créer le modèle DQN ─────────────────────────────────
# DQN = Q-learning mais avec un réseau de neurones
# au lieu d'une table Q (meilleur pour les grands espaces d'états)
model = DQN(
    policy='MlpPolicy',           # réseau de neurones multi-couches (dense)
    env=env,                      # l'environnement à apprendre
    learning_rate=1e-3,           # α = taux d'apprentissage (équivalent alpha=0.1)
    gamma=0.99,                   # facteur de discount (importance du futur)
    exploration_fraction=0.5,     # 50% des steps consacrés à l'exploration
    exploration_final_eps=0.01,   # epsilon minimum = 1% (toujours un peu d'exploration)
    batch_size=32,                # nb d'expériences rejouées à chaque mise à jour
    buffer_size=10_000,           # taille du replay buffer (mémoire d'expériences)
    learning_starts=1000,         # nb de steps avant de commencer à apprendre
    target_update_interval=500,   # fréquence de mise à jour du réseau cible
    verbose=1                     # affiche les logs d'entraînement dans la console
)

# ─── 3. Entraîner le modèle ──────────────────────────────────
# La bibliothèque gère automatiquement :
#   - la boucle d'épisodes (while not done)
#   - la mise à jour de Bellman : Q(s,a) ← Q(s,a) + α[r + γ·maxQ(s',a') - Q(s,a)]
#   - la décroissance d'epsilon (exploration → exploitation)
#   - le replay buffer (rejouer des expériences passées pour stabiliser l'apprentissage)
model.learn(total_timesteps=50_000)   # 50 000 interactions avec l'environnement

# ─── 4. Sauvegarder le modèle entraîné ──────────────────────
model.save('robot_entrepot_dqn')      # sauvegarde le réseau de neurones sur disque
# Pour recharger plus tard : model = DQN.load('robot_entrepot_dqn', env=env)

# ─── 5. Évaluer les performances ────────────────────────────
mean_reward, std_reward = evaluate_policy(
    model,                    # le modèle entraîné
    env,                      # l'environnement de test
    n_eval_episodes=100,      # on joue 100 épisodes de test
    deterministic=True        # pas d'exploration au test → on utilise la meilleure action
)
print(f'\nRécompense moyenne sur 100 épisodes : {mean_reward:.2f} ± {std_reward:.2f}')
# Résultat attendu après entraînement : proche de 1.0 (le robot atteint la cible)

# ─── 6. Rejouer une partie complète ─────────────────────────
print('\n--- Simulation d\'une partie ---')
obs, _ = env.reset()          # réinitialise l'environnement → état initial = case S (0)
done = False
step_count = 0

while not done:
    action, _ = model.predict(obs, deterministic=True)          # le réseau choisit l'action
    obs, reward, terminated, truncated, info = env.step(action) # on joue l'action
    done = terminated or truncated                              # fin si cible ou trou atteint
    step_count += 1

    action_names = ['← Gauche', '↓ Bas', '→ Droite', '↑ Haut']
    print(f'  Step {step_count} | Action : {action_names[int(action)]} | Récompense : {reward}')

if reward == 1:
    print('\n✓ Le robot a atteint la cible (colis récupéré) !')
else:
    print('\n✗ Le robot est tombé dans un trou.')

env.close()   # ferme l'environnement proprement