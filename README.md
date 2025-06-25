# Clara

Assistant conversationnel pour expliquer vos modèles de machine learning avec LIME et SHAP.

## Description

Imaginez un assistant intelligent, toujours prêt à vous accompagner dans la **compréhension** de vos modèles de machine learning : c’est Clara ! Conçue pour démystifier le « boîte noire », Clara vous offre :

- 🚀 **Chargement aisé** de vos modèles et datasets (Pandas, NumPy, pickle…)
- 🔍 **Explications locales** (LIME et SHAP) pour chaque prédiction individuellement
- 📈 **Analyse globale** de l’importance des variables et de leurs interactions
- 💬 **Dialogue interactif** en langage naturel via la fonction `run_chatbot`
- 🗒️ **Contexte personnalisé** (métier, dataset, cas d’usage) pour des réponses encore plus pertinentes

Que vous soyez **data scientist**, **analyste métier** ou simplement **curieux**, Clara transforme l’interprétation de vos modèles en une expérience ludique et pédagogique.

## Installation

```bash
pip install git+https://github.com/alexpicart855/Chatbot_clara.git
```

> **Colab** : dans une cellule notebook, utilisez `!pip install git+https://github.com/alexpicart855/Chatbot_clara.git`

## Usage

```python
from chatbot_clara import run_chatbot

# Préparez votre modèle et vos données
model = ...         # modèle scikit-learn ou similaire
X_train = ...       # DataFrame/array d'entraînement
X_test = ...        # DataFrame/array de test

# (Optionnel) Contexte métier ou dataset
context = {
    "dataset_name": "iris",
    "business_problem": "classification des espèces",
    "notes": "Exemple pédagogique"
}

# Lancez Clara
run_chatbot(model, X_train, X_test, context)
# ou sans contexte
run_chatbot(model, X_train, X_test)
```

## Structure du projet

```
chatbot_clara/    # code du module
  ├── __init__.py
  └── chatbot.py
setup.py          # installation du package
README.md         # documentation
```

## Licence

Ce projet est publié sous la **licence MIT**.

> **Important** : Le fichier `LICENSE` doit être inclus dans toutes les distributions du package. Sans ce fichier, les utilisateurs ne sont pas légalement autorisés à réutiliser ou redistribuer le code.

