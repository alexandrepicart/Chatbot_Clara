# chatbot_clara

Package Clara avec intégration complète des fonctions d'explication LIME et SHAP et run_chatbot du notebook.

## Installation

```bash
pip install .
```

## Utilisation

```python
from chatbot_clara import run_chatbot

# Préparez votre modèle et vos données
model = ...
X_train = ...
X_test = ...

# Lancez Clara
run_chatbot(model, X_train, X_test)
```
