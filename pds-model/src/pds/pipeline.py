import logging
import pickle
import sys
import numpy as np
import pandas

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn import metrics
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

RANDOM_STATE = 42

logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s - %(levelname)s]: %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def load_dataset(dataset_path) -> pandas.DataFrame:
    """
    Carrega um arquivo CSV com métricas extraídas e rótulos associados.

    Parameters:
        dataset_path (str): Caminho para o arquivo CSV do dataset.

    Returns:
        sdp_dataset (pandas.DataFrame): DataFrame com os dados carregados.
    """
    sdp_dataset = pandas.read_csv(
        dataset_path,
        index_col=None,
        header=0,
        delimiter=';'
    )
    return sdp_dataset


def save_model(model, model_name, data_balance, cv_criteria):
    """
    Salva o modelo fornecido em um arquivo pickle.

    Parameters:
        model (obj): Instância do modelo a ser salva.
        model_name (str): Nome identificador do modelo.
        data_balance (str): Método de balanceamento de dados utilizado.
        cv_criteria (str): Critério de validação cruzada utilizado.
    """
    with open(f"model_{model_name}_{cv_criteria.upper()}_{data_balance}.pkl", "wb") as model_file:
        pickle.dump(model, model_file)


def load_model(file_model_path):
    """
    Carrega um modelo previamente salvo em um arquivo pickle.

    Parameters:
        file_model_path (str): Caminho para o arquivo .pkl contendo o modelo.

    Returns:
        model (obj): O modelo carregado.
    """
    with open(file_model_path, 'rb') as model_file:
        return pickle.load(model_file)


def _predict_with_scores(estimator, X_eval):
    """
    Obtém rótulos discretos, scores contínuos e probabilidades previstas pelo classificador.

    Returns:
        tuple: (y_pred, y_score, y_proba)
    """
    y_pred = estimator.predict(X_eval)
    if hasattr(estimator, "predict_proba"):
        y_proba = estimator.predict_proba(X_eval)[:, 1]
        y_score = y_proba
    elif hasattr(estimator, "decision_function"):
        y_score = estimator.decision_function(X_eval)
        y_proba = 1.0 / (1.0 + np.exp(-np.clip(y_score, -500.0, 500.0)))
    else:
        y_score = y_pred
        y_proba = y_pred
    return y_pred, y_score, y_proba


def extract_model_metrics_scores(y_test, y_pred, y_score=None, y_proba=None) -> dict:
    """
    Extrai métricas de desempenho comparando predições, scores contínuos,
    probabilidades previstas e valores reais.

    Parameters:
        y_test (array-like): Rótulos reais do conjunto de teste.
        y_pred (array-like): Rótulos binários previstos pelo modelo.
        y_score (array-like, optional): Score contínuo do classificador para cálculo de ROC-AUC.
        y_proba (array-like, optional): Probabilidades previstas da classe positiva para Brier Score.

    Returns:
        scores (dict): Dicionário com métricas como acurácia, ROC-AUC, Brier, F1, etc.
    """
    continuous_score = y_score if y_score is not None else (y_proba if y_proba is not None else y_pred)
    prob_score = y_proba if y_proba is not None else y_pred

    scores = {
        "accuracy_score": metrics.accuracy_score(y_test, y_pred),
        "roc_auc_score": metrics.roc_auc_score(y_test, continuous_score),
        "f1_score": metrics.f1_score(y_test, y_pred, zero_division=0),
        "precision_score": metrics.precision_score(y_test, y_pred, zero_division=0),
        "recall_score": metrics.recall_score(y_test, y_pred, zero_division=0),
        "matthews_corrcoef": metrics.matthews_corrcoef(y_test, y_pred),
        "brier_score_loss": metrics.brier_score_loss(y_test, prob_score),
        "confusion_matrix": metrics.confusion_matrix(y_test, y_pred),
        "classification_report": metrics.classification_report(y_test, y_pred, zero_division=0),
    }
    return scores


def _build_estimator_pipeline(estimator, data_balance):
    """
    Constrói a pipeline completa de pré-processamento, balanceamento (restrito ao treino)
    e classificação, evitando vazamento de dados (data leakage) entre treino e validação/teste.
    """
    if data_balance == "SMOTE":
        return ImbPipeline([
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("clf", clone(estimator)),
        ])
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", clone(estimator)),
    ])


def _prepare_leak_free_inner_cv(X_train, y_train, data_balance, n_splits=5):
    """
    Prepara os folds internos da validação cruzada aplicando StandardScaler e SMOTE
    exclusivamente sobre a partição de treino de cada fold interno (sem jamais tocar
    na partição de validação interna), produzindo resultado matematicamente idêntico
    à avaliação de ImbPipeline dentro do GridSearchCV sem recalcular o SMOTE milhares
    de vezes para o mesmo fold interno.
    """
    X_arr = np.asarray(X_train, dtype=float)
    y_arr = np.asarray(y_train)
    inner_skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    chunks_X = []
    chunks_y = []
    custom_cv = []
    offset = 0

    for inner_tr_idx, inner_val_idx in inner_skf.split(X_arr, y_arr):
        scaler = StandardScaler()
        X_inner_tr = scaler.fit_transform(X_arr[inner_tr_idx])
        y_inner_tr = y_arr[inner_tr_idx]

        X_inner_val = scaler.transform(X_arr[inner_val_idx])
        y_inner_val = y_arr[inner_val_idx]

        if data_balance == "SMOTE":
            smote = SMOTE(random_state=RANDOM_STATE)
            X_inner_tr, y_inner_tr = smote.fit_resample(X_inner_tr, y_inner_tr)

        n_tr = len(X_inner_tr)
        n_val = len(X_inner_val)
        tr_slice = np.arange(offset, offset + n_tr)
        val_slice = np.arange(offset + n_tr, offset + n_tr + n_val)

        chunks_X.extend([X_inner_tr, X_inner_val])
        chunks_y.extend([y_inner_tr, y_inner_val])
        custom_cv.append((tr_slice, val_slice))
        offset += n_tr + n_val

    X_pool = np.vstack(chunks_X)
    y_pool = np.concatenate(chunks_y)
    return X_pool, y_pool, custom_cv


def _fit_leak_free_grid_search(estimator, param_grid, X_train, y_train, data_balance, cv_criteria, precomputed_cv=None):
    """
    Executa busca em grade com validação cruzada interna livre de vazamento de SMOTE/escala
    e ajusta a pipeline final vencedora (StandardScaler -> SMOTE -> Classificador) em X_train.
    """
    clean_grid = {
        (k[5:] if k.startswith("clf__") else k): v
        for k, v in (param_grid or {}).items()
    }

    if clean_grid:
        if precomputed_cv is None:
            X_pool, y_pool, custom_cv = _prepare_leak_free_inner_cv(X_train, y_train, data_balance, n_splits=5)
        else:
            X_pool, y_pool, custom_cv = precomputed_cv

        grid_model = GridSearchCV(
            clone(estimator),
            clean_grid,
            cv=custom_cv,
            scoring=cv_criteria,
            refit=False,
            n_jobs=-1,
        )
        grid_model.fit(X_pool, y_pool)
        tuned_estimator = clone(estimator).set_params(**grid_model.best_params_)
    else:
        tuned_estimator = clone(estimator)

    best_pipeline = _build_estimator_pipeline(tuned_estimator, data_balance)
    best_pipeline.fit(X_train, y_train)
    return best_pipeline


def _get_models_and_grids(grid_search: bool):
    """
    Retorna os classificadores configurados com semente fixa (reprodutibilidade)
    e suporte a probabilidades calibradas, além de suas grades de hiperparâmetros.
    """
    train_models = {
        "SVC": SVC(probability=True, random_state=RANDOM_STATE),
        "LRC": LogisticRegression(random_state=RANDOM_STATE),
        "RFC": RandomForestClassifier(random_state=RANDOM_STATE),
        "GBC": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }

    if grid_search:
        grid_params_list = {
            "LRC": {
                "C": [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10],
                "penalty": ["l1", "l2"],
                "max_iter": [10**7],
                "fit_intercept": [True],
                "solver": ["liblinear"],
            },
            "RFC": {
                "n_estimators": [5, 30, 50, 75, 100, 150, 200],
                "max_depth": [4, 5, 6, 7, 8, None],
            },
            "SVC": {
                "C": [0.01, 0.05, 0.1, 0.5, 1, 5, 10],
                "kernel": ["linear", "rbf"],
                "max_iter": [10**7],
            },
            "GBC": {
                "loss": ["log_loss"],
                "learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
                "min_samples_split": np.linspace(0.1, 0.5, 12),
                "min_samples_leaf": np.linspace(0.1, 0.5, 12),
                "max_depth": [3, 5, 8],
                "max_features": ["log2", "sqrt"],
                "criterion": ["friedman_mse"],
                "subsample": [0.5, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0],
                "n_estimators": [10],
            },
        }
    else:
        grid_params_list = {"LRC": {}, "RFC": {}, "SVC": {}, "GBC": {}}

    return train_models, grid_params_list


def run_experiment(dataset, x_features, y_label, data_balance, models, grid_params_list, cv_criteria) -> dict:
    """
    Executa benchmark de diversos modelos usando validação cruzada estratificada,
    com pré-processamento e SMOTE isolados dentro de cada fold interno de validação.

    Parameters:
        dataset (pandas.DataFrame): DataFrame contendo as features e rótulos.
        x_features (list of str): Lista com nomes das colunas das features.
        y_label (str): Nome da coluna do rótulo.
        data_balance (str): Método para balanceamento de dados ('SMOTE' ou outro).
        models (dict): Dicionário nome->instância dos modelos a avaliar.
        grid_params_list (dict): Dicionário nome->parâmetros do GridSearchCV.
        cv_criteria (str): Critério de scoring para o GridSearchCV (ex: 'roc_auc').

    Returns:
        fold_results (dict): Dicionário onde cada chave é o índice do fold e o valor
                             é outro dicionário com informações de cada modelo
                             ('score' e 'best_estimator').
    """
    X = dataset[x_features]
    y = dataset[y_label]

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
    models_info_per_fold = {}

    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        precomputed_cv = _prepare_leak_free_inner_cv(X_train, y_train, data_balance, n_splits=5)

        models_info = {}
        for model in models:
            logger.debug(f"Fold {i} - Avaliando {model}...")
            best_estimator = _fit_leak_free_grid_search(
                estimator=models[model],
                param_grid=grid_params_list[model],
                X_train=X_train,
                y_train=y_train,
                data_balance=data_balance,
                cv_criteria=cv_criteria,
                precomputed_cv=precomputed_cv,
            )
            y_pred, y_score, y_proba = _predict_with_scores(best_estimator, X_test)
            metrics_scores = extract_model_metrics_scores(
                y_test, y_pred, y_score=y_score, y_proba=y_proba
            )
            models_info[model] = {
                "score": metrics_scores,
                "best_estimator": best_estimator,
            }

        models_info_per_fold[i] = models_info

    return models_info_per_fold


def do_benchmark(
    grid_search=False,
    data_balance="SMOTE",
    dataset_path=None,
    cv_criteria="roc_auc",
    selected_models=["LRC", "RFC", "SVC", "GBC"],
) -> dict:
    """
    Orquestra o benchmark de modelos selecionados a partir de um arquivo de dados.

    Parameters:
        grid_search (bool): Se True, utiliza GridSearchCV para otimização de hiperparâmetros.
        data_balance (str): Método de balanceamento de dados ('SMOTE' ou outro).
        dataset_path (str): Caminho para o arquivo CSV do dataset.
        cv_criteria (str): Critério de scoring para validação cruzada (ex: 'roc_auc').
        selected_models (list of str): Lista de chaves dos modelos a serem treinados.

    Returns:
        fold_results (dict): Resultado do benchmark para cada fold.
    """
    x_features = ["LOC", "COM", "BLK", "NOF", "NOC", "APF", "AMC", "NER", "NEH", "CYC", "MAD"]
    y_label = "BUG"

    dataset = load_dataset(dataset_path)
    train_models, grid_params_list = _get_models_and_grids(grid_search=grid_search)
    models = {i: train_models[i] for i in train_models if i in selected_models}

    fold_results = run_experiment(
        dataset=dataset,
        x_features=x_features,
        y_label=y_label,
        data_balance=data_balance,
        models=models,
        grid_params_list=grid_params_list,
        cv_criteria=cv_criteria,
    )

    return fold_results


def build_champion_model(dataset, x_features, y_label, data_balance, model_info, cv_criteria) -> dict:
    """
    Cria o modelo final (champion model) a partir de divisão treino/teste estratificada
    e validação cruzada interna com SMOTE isolado por fold.

    Parameters:
        dataset (pandas.DataFrame): DataFrame com features e rótulos completos.
        x_features (list of str): Nomes das colunas das features.
        y_label (str): Nome da coluna do rótulo.
        data_balance (str): Método de balanceamento de dados ('SMOTE' ou outro).
        model_info (dict): Informações sobre o modelo ('instance', 'grid_params_list', 'name').
        cv_criteria (str): Critério de scoring para validação cruzada (ex: 'roc_auc').

    Returns:
        metrics_scores (dict): Métricas de desempenho do modelo final.
    """
    X = dataset[x_features]
    y = dataset[y_label]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )

    best_estimator = _fit_leak_free_grid_search(
        estimator=model_info.get("instance"),
        param_grid=model_info.get("grid_params_list"),
        X_train=X_train,
        y_train=y_train,
        data_balance=data_balance,
        cv_criteria=cv_criteria,
    )

    y_pred, y_score, y_proba = _predict_with_scores(best_estimator, X_test)
    metrics_scores = extract_model_metrics_scores(
        y_test, y_pred, y_score=y_score, y_proba=y_proba
    )

    save_model(best_estimator, model_info.get("name"), data_balance, cv_criteria)

    return metrics_scores


def make_model(
    grid_search=False,
    data_balance="SMOTE",
    dataset_path=None,
    cv_criteria="roc_auc",
    selected_model=None,
) -> dict:
    """
    Constrói e avalia o modelo selecionado como champion.

    Parameters:
        grid_search (bool): Se True, usa GridSearchCV para otimizar hiperparâmetros.
        data_balance (str): Método de balanceamento de dados ('SMOTE' ou outro).
        dataset_path (str): Caminho para o arquivo CSV do dataset.
        cv_criteria (str): Critério de scoring para validação cruzada (ex: 'roc_auc').
        selected_model (str): Chave do modelo a treinar (e.g., 'LRC', 'RFC', 'SVC', 'GBC').

    Returns:
        metrics_scores (dict): Métricas de desempenho do modelo final.
    """
    x_features = ["LOC", "COM", "BLK", "NOF", "NOC", "APF", "AMC", "NER", "NEH", "CYC", "MAD"]
    y_label = "BUG"

    dataset = load_dataset(dataset_path)
    train_models, grid_params_list = _get_models_and_grids(grid_search=grid_search)

    model_info = {
        "name": selected_model,
        "instance": train_models.get(selected_model),
        "grid_params_list": grid_params_list.get(selected_model),
    }

    metrics_scores = build_champion_model(
        dataset, x_features, y_label, data_balance, model_info, cv_criteria
    )
    return metrics_scores


def select_best_model(fold_results) -> str:
    """
    Seleciona o modelo com maior média de ROC-AUC contínuo a partir dos resultados por fold.

    Parameters:
        fold_results (dict): Resultado do benchmark, dicionário de folds->modelos.

    Returns:
        best_model_name (str): Nome do modelo com maior média de ROC-AUC.
    """
    results = {}
    for fold in fold_results.keys():
        for model_name in fold_results.get(fold).keys():
            roc_auc = fold_results[fold][model_name]["score"]["roc_auc_score"]
            if results.get(model_name) is None:
                results[model_name] = {"roc_auc": [roc_auc]}
            else:
                results[model_name]["roc_auc"].append(roc_auc)

    best_model_score = -1.0
    best_model_name = None
    for key in results.keys():
        avg_roc_auc = sum(results[key]["roc_auc"]) / len(results[key]["roc_auc"])
        logger.debug(f"Modelo {key} - ROC-AUC Médio ({len(results[key]['roc_auc'])} folds): {avg_roc_auc:.4f}")
        if avg_roc_auc > best_model_score:
            best_model_score = avg_roc_auc
            best_model_name = key

    return best_model_name


def start(dataset_path):
    logger.debug("[Step-1] Realizando Benchmark")
    fold_results = do_benchmark(
        grid_search=True,
        dataset_path=dataset_path,
        selected_models=["LRC", "RFC", "SVC", "GBC"],
    )

    for fold, models_info in fold_results.items():
        for model_name, info in models_info.items():
            sc = info["score"]
            logger.debug(
                f"Fold {fold} - Model {model_name}: "
                f"ROC-AUC={sc['roc_auc_score']:.4f}, Brier={sc['brier_score_loss']:.4f}, "
                f"F1={sc['f1_score']:.4f}, Acc={sc['accuracy_score']:.4f}, "
                f"Pre={sc['precision_score']:.4f}, Rec={sc['recall_score']:.4f}"
            )

    logger.debug("\n[Step-2] Selecionando Melhor Modelo")
    best_model_name = select_best_model(fold_results)
    logger.debug(f"\nBest Model {best_model_name}")

    logger.debug("\n[Step-3] Criando o Modelo Final")
    metric_scores = make_model(
        grid_search=True,
        dataset_path=dataset_path,
        selected_model=best_model_name,
    )
    sc = metric_scores
    logger.debug(
        f"Champion {best_model_name}: ROC-AUC={sc['roc_auc_score']:.4f}, "
        f"Brier={sc['brier_score_loss']:.4f}, F1={sc['f1_score']:.4f}, "
        f"Acc={sc['accuracy_score']:.4f}, Pre={sc['precision_score']:.4f}, "
        f"Rec={sc['recall_score']:.4f}"
    )


def main():
    if len(sys.argv) > 1:
        dataset_path = str(sys.argv[1])
        start(dataset_path)
    else:
        print("Você deve prover o caminho para o dataset.")


if __name__ == "__main__":
    main()
