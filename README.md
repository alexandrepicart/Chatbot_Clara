# Clara

Assistant conversationnel pour expliquer vos modèles de machine learning avec LIME et SHAP.

## Description

Imaginez un assistant intelligent, toujours prêt à vous accompagner dans la **compréhension** de vos modèles de machine learning : c’est Clara ! Conçue pour démystifier les « boîtes noires », Clara vous offre :

- 🚀 **Chargement aisé** de vos modèles et datasets (Pandas, NumPy, pickle…)
- 🔍 **Explications locales** (LIME et SHAP) pour chaque prédiction individuellement
- 📈 **Analyse globale** de l’importance des variables et de leurs interactions
- 💬 **Dialogue interactif** en langage naturel via la fonction `run_chatbot`
- 🗒️ **Contexte personnalisé** (métier, dataset, cas d’usage) pour des réponses encore plus pertinentes

Que vous soyez **data scientist**, **analyste métier** ou simplement **curieux**, Clara transforme l’interprétation de vos modèles en une expérience ludique et pédagogique.

## Installation

Avant d’installer Clara, vous devez fournir votre **clé API Google**. Vous pouvez :

- **Dans le shell** (Linux/macOS/Windows) :

  ```bash
  export GOOGLE_API_KEY="votre_clé_ici"
  ```

- **Dans un notebook Python** (ex. Google Colab), avant d’installer ou d’importer Clara :

  ```python
  import os
  os.environ["GOOGLE_API_KEY"] = "votre_clé_ici"
  ```

  *Cette méthode définit la variable pour la session Python en cours.*

Ensuite, installez Clara :

```bash
pip install git+https://github.com/alexandrepicart/Chatbot_clara.git
```

> **Sur Google Colab**, vous pouvez lancer directement dans une cellule :
>
> ```bash
> !pip install git+https://github.com/alexandrepicart/Chatbot_clara.git
> ```

## Usage

L’argument `context` est **optionnel** : il permet à Clara de personnaliser ses explications en fonction de votre **cas d’usage**, du **jeu de données** ou d’éléments métier que vous fournissez.

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

Le dépôt se compose d’une structure **épurée** facilitant la maintenance et l’évolution :

```
chatbot_clara/    # code du module
  ├── __init__.py  # initialisation du package
  └── chatbot.py   # logique de dialogue et explications
setup.py          # script d’installation du package
README.md         # documentation (ce fichier)
```

## Licence

Ce projet est publié sous la **licence MIT**.

