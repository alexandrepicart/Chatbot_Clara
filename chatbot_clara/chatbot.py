import os
from langchain.agents import Tool, initialize_agent, AgentType
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage
from langchain.memory import ConversationBufferMemory
import pandas as pd
import numpy as np
import re
from lime.lime_tabular import LimeTabularExplainer
from sklearn.compose import ColumnTransformer
from sklearn.base import TransformerMixin
import shap
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import is_classifier, is_regressor

def run_chatbot(
    model,
    X_train,
    X_test,
    raw_context=None,
    num_features: int = 4,
    random_state: int = 42
):
    
    
    def explain_lime(
    model,
    X_train: pd.DataFrame,
    X_instance: pd.DataFrame,
    num_features: int = 10,
    random_state: int = 42
    ):
        
        """
        Explique la prédiction du `model` sur `X_instance` à l'aide de LIME.
        Gère indifféremment :
        - un Pipeline contenant un ColumnTransformer
        - un estimator seul

        Pour les features numériques, détecte automatiquement tout scaler/dispositif implémentant `inverse_transform`
        pour reconvertir les seuils LIME des unités normalisées aux unités d'origine.
        """
        # 1) Détection du pipeline et extraction du préprocesseur/estimateur
        is_pipeline = isinstance(model, Pipeline)
        preproc = None
        estimator = model
        if is_pipeline:
            for name, step in model.named_steps.items():
                if isinstance(step, ColumnTransformer):
                    preproc = step
                    break
            # l'estimateur se trouve après le ColumnTransformer
            if preproc is not None:
                steps = list(model.named_steps.items())
                idx = next((i for i, (_, s) in enumerate(steps) if s is preproc), None)
                if idx is not None and idx + 1 < len(steps):
                    estimator = steps[idx + 1][1]

        # 2) Préparation des noms et données brutes
        raw_feature_names = X_train.columns.tolist()
        raw_training_data = X_train.values

        # 3) Transformation et détection du scaler numérique
        numeric_scaler = None
        numeric_cols = []
        if preproc is not None:
            # transformer NumericScale: rechercher un scaler (StandardScaler, MinMaxScaler, etc.)
            for name, transformer, cols in preproc.transformers_:
                tr = preproc.named_transformers_.get(name)
                # si pipeline, rechercher un scaler à l'intérieur
                if isinstance(tr, Pipeline):
                    for step in tr.named_steps.values():
                        # on suppose que les scalers ont 'inverse_transform' et des attributs typiques
                        if hasattr(step, 'inverse_transform') and (
                            hasattr(step, 'scale_') and hasattr(step, 'mean_')
                        ) or (
                            hasattr(step, 'inverse_transform') and hasattr(step, 'data_min_') and hasattr(step, 'data_max_')
                        ):
                            numeric_scaler = step
                            numeric_cols = cols
                            break
                else:
                    if hasattr(tr, 'inverse_transform') and (
                        hasattr(tr, 'scale_') and hasattr(tr, 'mean_')
                    ) or (
                        hasattr(tr, 'inverse_transform') and hasattr(tr, 'data_min_') and hasattr(tr, 'data_max_')
                    ):
                        numeric_scaler = tr
                        numeric_cols = cols
                if numeric_scaler:
                    break
            # application du préproc s'il existe
            X_train_enc = preproc.transform(X_train)
            feat_names  = preproc.get_feature_names_out()
            X_inst_enc  = preproc.transform(X_instance)
        else:
            X_train_enc = raw_training_data
            feat_names  = raw_feature_names
            X_inst_enc  = X_instance.values

        # 4) Construction de l'explainer LIME
        explainer = LimeTabularExplainer(
            training_data=X_train_enc,
            feature_names=feat_names,
            class_names=(estimator.classes_.tolist() if hasattr(estimator, 'classes_') else None),
            mode=('classification' if hasattr(estimator, 'predict_proba') else 'regression'),
            random_state=random_state
        )

        # 5) Fonction de prédiction pour LIME
        def predict_fn(arr: np.ndarray):
            if preproc is not None:
                return (estimator.predict_proba(arr) if hasattr(estimator, 'predict_proba')
                        else estimator.predict(arr))
            else:
                df = pd.DataFrame(arr, columns=raw_feature_names)
                return (estimator.predict_proba(df) if hasattr(estimator, 'predict_proba')
                        else estimator.predict(df))

        # 6) Génération de l'explication
        exp = explainer.explain_instance(
            X_inst_enc[0],
            predict_fn,
            num_features=num_features
        )

        # 7) Dés-scaling des seuils (intervalles et seuils simples) via inverse_transform
        pattern = re.compile(
            r"([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*<\s*([\w\s\(\)-]+?)\s*<=\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)"
            r"|([\w\s\(\)-]+?)\s*(<=|<|>=|>)\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)"
        )
        explanations = []
        # longueur du vecteur numérique encodé
        n_num = len(numeric_cols)

        for feat, weight in exp.as_list():
            desc = feat
            m = pattern.match(feat) if numeric_scaler is not None else None
            if m:
                # cas intervalle
                if m.group(1) and m.group(2) and m.group(3):
                    low_s, name, high_s = float(m.group(1)), m.group(2), float(m.group(3))
                    if name in numeric_cols:
                        idx = numeric_cols.index(name)
                        # construire vecteur scaled
                        vec_low  = np.zeros((1, n_num)); vec_low[0, idx] = low_s
                        vec_high = np.zeros((1, n_num)); vec_high[0, idx] = high_s
                        low_r  = numeric_scaler.inverse_transform(vec_low)[0, idx]
                        high_r = numeric_scaler.inverse_transform(vec_high)[0, idx]
                        desc = f"{low_r:.2f} < {name} <= {high_r:.2f}"
                # cas seuil simple
                elif m.group(4) and m.group(5) and m.group(6):
                    name, op, thr_s = m.group(4), m.group(5), float(m.group(6))
                    if name in numeric_cols:
                        idx = numeric_cols.index(name)
                        vec = np.zeros((1, n_num)); vec[0, idx] = thr_s
                        thr_r = numeric_scaler.inverse_transform(vec)[0, idx]
                        desc = f"{name} {op} {thr_r:.2f}"
            explanations.append({"feature": desc, "weight": weight})

        # calcul de la contribution en %
        total = sum(abs(item['weight']) for item in explanations)
        for item in explanations:
            item['contrib_pct'] = abs(item['weight']) / total * 100

        # 8) Prédiction finale
        model_info = {"model_class": estimator.__class__.__name__}
        if hasattr(estimator, 'predict_proba'):
            data_for_pred = X_inst_enc if preproc is not None else X_instance
            proba = estimator.predict_proba(data_for_pred)[0]
            pred  = estimator.predict(data_for_pred)[0]
            prediction = {'proba': proba.tolist(), 'pred': int(pred)}
            mode = 'classification'
        else:
            data_for_pred = X_inst_enc if preproc is not None else X_instance
            pred = float(estimator.predict(data_for_pred)[0])
            prediction = {'pred': pred}
            mode = 'regression'

        return {
            'model_info':   model_info,
            'mode':         mode,
            'prediction':   prediction,
            'explanations': explanations
        }
    
    def explain_shap_local(model, X_train, X_instance, top_n: int = 10):
        """
        Explique localement la prédiction d'un modèle (classification ou régression) avec SHAP.
        Gère les pipelines scikit-learn, ColumnTransformer, et évite les warnings de feature names.

        Paramètres:
            model      : Pipeline ou estimator scikit-learn entraîné
            X_train    : pd.DataFrame ou np.ndarray pour le background SHAP
            X_instance : pd.DataFrame, pd.Series ou np.ndarray, instance(s) à expliquer
            top_n      : int, nombre de features à retourner (triées par importance)

        Retourne:
            dict avec :
                - model_info   : {'model_class': str}
                - mode         : 'classification' ou 'regression'
                - prediction   : {'pred': ..., 'proba': [...]} si classification sinon {'pred': ...}
                - base_value   : valeur de base SHAP (probabilité moyenne ou sortie moyenne)
                - explanations : liste de dicts {'feature','shap_value','contrib_pct'}
        """
        # 1) Extraire préprocesseur et estimateur final
        estimator = model
        preproc = None
        if isinstance(model, Pipeline):
            for _, step in model.named_steps.items():
                if isinstance(step, ColumnTransformer):
                    preproc = step
                    break
            if preproc is not None:
                steps = list(model.named_steps.items())
                idx = next((i for i, (_, s) in enumerate(steps) if s is preproc), None)
                if idx is not None and idx + 1 < len(steps):
                    estimator = steps[idx + 1][1]

        # 2) Préparer X_train pour SHAP
        if preproc is not None:
            if isinstance(X_train, pd.DataFrame):
                df_train = X_train
            else:
                if hasattr(preproc, 'feature_names_in_'):
                    df_train = pd.DataFrame(np.array(X_train), columns=preproc.feature_names_in_)
                else:
                    raise ValueError("ColumnTransformer requires input DataFrame with feature_names_in_.")
            X_train_enc = preproc.transform(df_train)
            feature_names = preproc.get_feature_names_out()
        else:
            if isinstance(X_train, pd.DataFrame):
                X_train_enc = X_train.values
                feature_names = X_train.columns.tolist()
            else:
                X_train_enc = np.array(X_train)
                feature_names = [f"X{i}" for i in range(X_train_enc.shape[1])]

        # 3) Préparer X_instance pour transformation
        if isinstance(X_instance, pd.DataFrame):
            df_inst = X_instance
        elif isinstance(X_instance, pd.Series):
            df_inst = X_instance.to_frame().T
        else:
            arr = np.array(X_instance)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            elif arr.ndim != 2:
                raise ValueError(f"X_instance must be 1D or 2D, got ndim={arr.ndim}")
            if preproc is not None:
                df_inst = pd.DataFrame(arr, columns=preproc.feature_names_in_)
            else:
                df_inst = pd.DataFrame(arr, columns=feature_names)
        X_instance_enc = preproc.transform(df_inst) if preproc is not None else df_inst.values
        if X_instance_enc.shape[0] > 1:
            X_instance_enc = X_instance_enc[:1]

        # 4) Détecter classification vs régression
        if is_classifier(estimator):
            mode = 'classification'
        elif is_regressor(estimator):
            mode = 'regression'
        else:
            mode = 'classification' if hasattr(estimator, 'predict_proba') else 'regression'

        # 5) Wrapper adaptatif pour predict_fn
        def make_predict_fn(fn):
            if hasattr(estimator, 'feature_names_in_'):
                def _predict(X):
                    df = pd.DataFrame(X, columns=feature_names)
                    return fn(df)
            else:
                def _predict(X):
                    return fn(X)
            return _predict

        if mode == 'classification':
            predict_fn = make_predict_fn(estimator.predict_proba)
        else:
            predict_fn = make_predict_fn(estimator.predict)

        # 6) Créer explainer et calculer SHAP
        explainer = shap.Explainer(predict_fn, X_train_enc, feature_names=feature_names)
        shap_output = explainer(X_instance_enc)

        # 7) Prédiction sans warnings
        use_df = hasattr(estimator, 'feature_names_in_')
        X_pred = pd.DataFrame(X_instance_enc, columns=feature_names) if use_df else X_instance_enc
        if mode == 'classification':
            pred = estimator.predict(X_pred)[0]
            proba = estimator.predict_proba(X_pred)[0]
            classes = estimator.classes_ if hasattr(estimator, 'classes_') else None
            class_index = list(classes).index(pred) if classes is not None and pred in classes else 0
            raw_vals = shap_output.values[0, :, class_index]
            base_val = shap_output.base_values[0, class_index]
        else:
            pred = estimator.predict(X_pred)[0]
            proba = None
            raw_vals = shap_output.values[0]
            base_val = shap_output.base_values[0]

        # 8) Construire contributions triées
        vals = np.array(raw_vals).flatten()
        contributions = [{'feature': name, 'shap_value': float(v)} for name, v in zip(feature_names, vals)]
        contributions.sort(key=lambda x: abs(x['shap_value']), reverse=True)
        if top_n and len(contributions) > top_n:
            contributions = contributions[:top_n]
        total_abs = sum(abs(c['shap_value']) for c in contributions)
        for c in contributions:
            c['contrib_pct'] = (abs(c['shap_value']) / total_abs * 100) if total_abs else 0.0

        # 9) Préparer la sortie finale
        result = {
            'model_info': {'model_class': estimator.__class__.__name__},
            'mode': mode,
            'prediction': {'pred': int(pred) if mode=='classification' else float(pred)},
            'base_value': float(base_val),
            'explanations': contributions
        }
        if proba is not None:
            result['prediction']['proba'] = [float(p) for p in proba]
        return result

    def explain_shap_global(model, X_train, X_test, top_n: int = 10, bg_thresh: int = 50):
        """
        Calcule les importances globales SHAP (en pourcentage) d'un modèle scikit-learn
        (régression ou classification multiclasse), en gérant pipelines et échantillonnage
        pour accélérer le calcul.
        Retourne uniquement 'feature' et 'contrib_pct'.
        """
        # 1) Extraire l'estimateur final d'un Pipeline
        estimator = model
        preproc = None
        if isinstance(model, Pipeline):
            for _, step in model.named_steps.items():
                if isinstance(step, ColumnTransformer):
                    preproc = step
                    break
            if preproc is not None:
                steps = list(model.named_steps.items())
                idx = next((i for i, (_, s) in enumerate(steps) if s is preproc), None)
                if idx is not None and idx+1 < len(steps):
                    estimator = steps[idx+1][1]

        # 2) Mode (classification vs régression)
        mode = 'classification' if is_classifier(estimator) else 'regression'
        # 3) Construction des DataFrames pour noms de colonnes
        def to_df(X):
            if isinstance(X, pd.DataFrame):
                return X.copy()
            arr = np.asarray(X)
            cols = getattr(estimator, 'feature_names_in_', None)
            if cols is None and preproc is not None:
                cols = getattr(preproc, 'feature_names_in_', None)
            return pd.DataFrame(arr, columns=cols) if cols is not None else pd.DataFrame(arr)
        X_train_df, X_test_df = to_df(X_train), to_df(X_test)

        # 4) Encodage via ColumnTransformer si présent
        if preproc is not None:
            X_train_enc = preproc.transform(X_train_df)
            X_test_enc  = preproc.transform(X_test_df)
            feature_names = preproc.get_feature_names_out()
        else:
            X_train_enc, X_test_enc = X_train_df.values, X_test_df.values
            feature_names = list(X_train_df.columns)

        # 5) Échantillonnage du background
        background = shap.sample(X_train_enc, bg_thresh) if X_train_enc.shape[0] > bg_thresh else X_train_enc

        # 6) Choix de l’explainer et calcul des shap_values
        mc = estimator.__class__.__name__.lower()
        if any(k in mc for k in ['forest', 'tree', 'boost', 'xgb', 'lgbm', 'catboost', 'gradient']):
            try:
                expl = shap.TreeExplainer(estimator, background,
                                        model_output='probability', check_additivity=False)
            except (TypeError, NotImplementedError):
                try:
                    expl = shap.TreeExplainer(estimator, background, model_output='probability')
                except (TypeError, NotImplementedError):
                    expl = shap.TreeExplainer(estimator, background)
            raw = expl.shap_values(X_test_enc, check_additivity=False) \
                if hasattr(expl, 'shap_values') else expl.shap_values(X_test_enc)
            shap_arr = np.stack(raw, axis=0) if isinstance(raw, list) else raw[np.newaxis, ...]
        else:
            pred_fn = (estimator.predict_proba
                    if (mode=='classification' and hasattr(estimator,'predict_proba'))
                    else estimator.predict)
            expl = shap.KernelExplainer(pred_fn, background)
            raw = expl.shap_values(X_test_enc)
            shap_arr = np.stack(raw, axis=0) if isinstance(raw, list) else raw[np.newaxis, ...]

        # 7) Calcul du pourcentage de contribution
        abs_arr = np.abs(shap_arr)
        imp_raw = np.mean(abs_arr, axis=(0, 1))
        if imp_raw.ndim > 1:
            imp_raw = np.mean(imp_raw, axis=1)
        imp_vals = imp_raw.flatten().tolist()
        total = sum(imp_vals)

        # 8) Construction et tri du résultat
        global_importances = []
        for feat, val in zip(feature_names, imp_vals):
            pct = (val / total * 100.0) if total > 0 else 0.0
            global_importances.append({
                'feature': feat,
                'contrib_pct': float(pct)
            })
        global_importances.sort(key=lambda x: x['contrib_pct'], reverse=True)

        if top_n and len(global_importances) > top_n:
            global_importances = global_importances[:top_n]

        return {
            'model_info': {'model_class': estimator.__class__.__name__},
            'mode': mode,
            'global_importances': global_importances
        }
        
    
    """
    Lance Clara avec un résumé ciblé du contexte pour expliquer les prédictions,
    et conserve la conversation en mémoire pour tenir compte du contexte.
    """
    # 1) Initialiser la mémoire tampon pour la conversation
    memory = ConversationBufferMemory(
        memory_key="chat_history",  # clé pour insérer l'historique dans le prompt
        return_messages=True        # conserver les messages sous forme structurée
    )
    # 2) Vérification de la clé API (inchangé)
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError("Veuillez définir la variable d'environnement 'GOOGLE_API_KEY'.")
    # 3) Instanciation des modèles de langage (LLMs) (inchangé)
    llm_clara = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.5)
    llm_summarizer = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.1)
    # 4) Construction de raw_text à partir de raw_context (inchangé)
    if raw_context:
        if isinstance(raw_context, dict):
            parts = []
            for k, v in raw_context.items():
                if isinstance(v, dict):
                    parts.append(f"{k}:")
                    for sk, sv in v.items():
                        parts.append(f"  - {sk}: {sv}")
                else:
                    parts.append(f"{k}: {v}")
            raw_text = "\n".join(parts)
        else:
            raw_text = str(raw_context)
    else:
        raw_text = ""
    # 5) Génération d'un résumé ciblé du contexte (compact_context) (inchangé)
    compact_context = ""
    if raw_text:
        prompt_summary = (
            "Tu es un assistant expert en ML. À partir du texte ci-dessous qui décrit un jeu de données, "
            "produis un résumé structuré en deux parties :\n"
            "1) Contexte général (1 à 2 phrases : objectif, taille, période).\n"
            "2) Variables d’entrée : pour chaque variable, une courte phrase précisant son nom, son type "
            "(numérique ou catégoriel) et sa signification.\n\n"
            f"Texte descriptif :\n{raw_text}"
        )
        messages = [HumanMessage(content=prompt_summary)]
        res = llm_summarizer.generate([messages])
        compact_context = res.generations[0][0].message.content.strip()
    # 6) Définition des outils LIME et SHAP pour l'agent
    def lime_explanation_tool(instance_idx: str) -> dict:
        try:
            idx = int(instance_idx)
        except (ValueError, TypeError):
            raise ValueError("Donnez-moi un indice d'instance valide.")
        X_inst = X_test.iloc[[idx]]
        return explain_lime(
            model=model,
            X_train=X_train,
            X_instance=X_inst,
            num_features=num_features,
            random_state=random_state
        )
    def shap_local_explanation_tool(instance_idx: str) -> dict:
        try:
            idx = int(instance_idx)
        except (ValueError, TypeError):
            raise ValueError("Donnez-moi un indice d'instance valide.")
        X_inst = X_test.iloc[[idx]]
        return explain_shap_local(
            model=model,
            X_train=X_train,
            X_instance=X_inst,
            top_n=num_features
        )
    def shap_global_explanation_tool(instance_idx: str = None) -> dict:
        # L'explication globale ne nécessite pas d'indice d'instance spécifique
        return explain_shap_global(
            model=model,
            X_train=X_train,
            X_test=X_test,
            top_n=num_features
        )
    lime_tool = Tool(
        name="lime_explanation",
        func=lime_explanation_tool,
        description="Renvoie un JSON LIME pour l'instance indiquée."
    )
    shap_local_tool = Tool(
        name="shap_local_explanation",
        func=shap_local_explanation_tool,
        description="Renvoie un JSON d'explication locale SHAP pour l'instance indiquée."
    )
    shap_global_tool = Tool(
        name="shap_global_explanation",
        func=shap_global_explanation_tool,
        description="Renvoie un JSON donnant l'importance globale des variables du modèle (SHAP)."
    )
    # 7) Construction du prompt système pour Clara (enrichi avec les trois outils)
    system_prompt = (
        "Ton nom est Clara, une experte en explicabilité de modèles ML via LIME et SHAP.\n"
        "Tu disposes de trois outils :\n"
        "- `lime_explanation` pour expliquer localement une instance avec LIME.\n"
        "- `shap_local_explanation` pour expliquer localement une instance avec SHAP.\n"
        "- `shap_global_explanation` pour fournir l'importance globale des variables du modèle avec SHAP.\n"
        "\nUtilise l'outil approprié en fonction de la demande de l'utilisateur et INTERPETE les valeurs de SHAP pour un non-technique:\n"
        "- Pour une explication locale sur une instance, utilise `lime_explanation` ou `shap_local_explanation` selon la méthode voulue.\n"
        "- Pour une demande d'importance globale des variables (par ex. les features les plus importantes), utilise `shap_global_explanation`."
        "- ne parle pas de shap value si on te le demande pas et explique toujours quel methode tu utilises (SHAP ou LIME)"

    )
    if compact_context:
        system_prompt += (
            "\n--- Contexte (résumé) :\n"
            f"{compact_context}\n"
            "------------------------------------\n"
            "(Utilise ces informations pour adapter tes explications.)\n"
        )
    # (Optionnel) Affichage du prompt système final pour débogage
    #print("=== [DEBUG] Prompt système injecté à Clara ===")
    #print(system_prompt)
    #print("=== [FIN DEBUG] ===\n")
    # 8) Initialisation de l'agent LangChain avec les outils et la mémoire
    agent = initialize_agent(
        tools=[lime_tool, shap_local_tool, shap_global_tool],
        llm=llm_clara,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=False,
        memory=memory,
        agent_kwargs={
            "prefix": system_prompt,
            "memory_key": "chat_history"
        }
    )
    # 9) Boucle de conversation interactive
    print("=== Clara, experte LIME et SHAP ===")
    print("Tapez 'exit' pour quitter.")
    while True:
        user_q = input("Vous> ")
        if user_q.strip().lower() == "exit":
            print("Au revoir !")
            break
        try:
            resp = agent.run(user_q)
        except Exception as e:
            print(f"Clara> Erreur : {e}")
        else:
            print("Clara> " + resp)
