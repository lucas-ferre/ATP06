import sys
import time
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd

from pds.pipeline import do_benchmark, select_best_model, make_model, logger, extract_model_metrics_scores

def sanitize_for_json(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int64, np.int32, np.integer)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.floating)):
        return float(obj)
    elif isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    elif not isinstance(obj, (str, int, float, bool, type(None))):
        return str(obj)
    return obj

def run_full_dataset_experiment(dataset_path: str, dataset_name: str):
    print(f"\n{'='*70}")
    print(f" INICIANDO EXPERIMENTO COMPLETO: {dataset_name} ({dataset_path})")
    print(f"{'='*70}\n")
    logger.info(f"--- INICIO EXPERIMENTO: {dataset_name} ({dataset_path}) ---")

    t_start = time.time()

    # Step 1: Benchmark 10-fold CV
    print(f"[{dataset_name}] [Passo 1/3] Executando do_benchmark (10-fold CV para LRC, RFC, SVC, GBC)...")
    fold_results = do_benchmark(
        grid_search=True,
        dataset_path=dataset_path,
        selected_models=["LRC", "RFC", "SVC", "GBC"]
    )

    # Step 2: Seleção do Melhor Modelo
    for fold_idx, models_dict in fold_results.items():
        for m_name, info in models_dict.items():
            sc = info["score"]
            logger.debug(
                f"[{dataset_name}] Fold {fold_idx} - Model {m_name}: "
                f"ROC-AUC={sc['roc_auc_score']:.4f}, Brier={sc['brier_score_loss']:.4f}, "
                f"F1={sc['f1_score']:.4f}, Acc={sc['accuracy_score']:.4f}, "
                f"Pre={sc['precision_score']:.4f}, Rec={sc['recall_score']:.4f}"
            )

    print(f"[{dataset_name}] [Passo 2/3] Calculando médias e selecionando melhor modelo...")
    best_model_name = select_best_model(fold_results)
    print(f"[{dataset_name}] Melhor Modelo Selecionado: {best_model_name}")
    logger.info(f"[{dataset_name}] Melhor Modelo Selecionado: {best_model_name}")

    # Step 3: Construção do Modelo Campeão Final
    print(f"[{dataset_name}] [Passo 3/3] Treinando Modelo Campeão Final ({best_model_name}) no split 75/25...")
    champion_scores = make_model(
        grid_search=True,
        dataset_path=dataset_path,
        selected_model=best_model_name
    )
    logger.info(
        f"[{dataset_name}] Champion {best_model_name} (Hold-out 25%): "
        f"ROC-AUC={champion_scores['roc_auc_score']:.4f}, "
        f"Brier={champion_scores['brier_score_loss']:.4f}, "
        f"F1={champion_scores['f1_score']:.4f}, "
        f"Acc={champion_scores['accuracy_score']:.4f}, "
        f"Pre={champion_scores['precision_score']:.4f}, "
        f"Rec={champion_scores['recall_score']:.4f}, "
        f"MCC={champion_scores['matthews_corrcoef']:.4f}"
    )

    t_total = time.time() - t_start
    print(f"[{dataset_name}] Concluído em {t_total:.1f}s ({t_total/60:.2f} min).")
    logger.info(f"--- FIM EXPERIMENTO: {dataset_name} ({t_total:.1f}s) ---")

    # Arquivo .pkl gerado
    pkl_src = Path(f"model_{best_model_name}_ROC_AUC_SMOTE.pkl")
    pkl_dst = Path(f"model_{dataset_name}_{best_model_name}_ROC_AUC_SMOTE.pkl")
    if pkl_src.exists():
        shutil.copy(pkl_src, pkl_dst)
        print(f"[{dataset_name}] Modelo salvo e preservado em: {pkl_dst}")

    # Processar métricas por modelo para sumarização
    summary_by_model = {}
    for fold_idx, models_dict in fold_results.items():
        for m_name, info in models_dict.items():
            if m_name not in summary_by_model:
                summary_by_model[m_name] = {
                    "roc_auc": [],
                    "f1": [],
                    "accuracy": [],
                    "precision": [],
                    "recall": [],
                    "mcc": [],
                    "brier": []
                }
            sc = info["score"]
            summary_by_model[m_name]["roc_auc"].append(float(sc["roc_auc_score"]))
            summary_by_model[m_name]["f1"].append(float(sc["f1_score"]))
            summary_by_model[m_name]["accuracy"].append(float(sc["accuracy_score"]))
            summary_by_model[m_name]["precision"].append(float(sc["precision_score"]))
            summary_by_model[m_name]["recall"].append(float(sc["recall_score"]))
            summary_by_model[m_name]["mcc"].append(float(sc["matthews_corrcoef"]))
            summary_by_model[m_name]["brier"].append(float(sc["brier_score_loss"]))

    stats_by_model = {}
    for m_name, metrics_dict in summary_by_model.items():
        stats_by_model[m_name] = {}
        for metric_name, values in metrics_dict.items():
            stats_by_model[m_name][f"{metric_name}_mean"] = float(np.mean(values))
            stats_by_model[m_name][f"{metric_name}_std"] = float(np.std(values))

    # Formatar resultados serializáveis
    serializable_folds = {}
    for fold_idx, models_dict in fold_results.items():
        serializable_folds[str(fold_idx)] = {}
        for m_name, info in models_dict.items():
            best_est = info["best_estimator"]
            clf_est = best_est.named_steps["clf"] if hasattr(best_est, "named_steps") and "clf" in best_est.named_steps else best_est
            serializable_folds[str(fold_idx)][m_name] = {
                "score": sanitize_for_json(info["score"]),
                "best_params": sanitize_for_json(clf_est.get_params())
            }

    result = {
        "dataset_name": dataset_name,
        "dataset_path": dataset_path,
        "runtime_seconds": t_total,
        "best_model_name": best_model_name,
        "champion_scores": sanitize_for_json(champion_scores),
        "model_stats": stats_by_model,
        "fold_results": serializable_folds
    }

    return result

def main():
    datasets = [
        ("v3_0_6_trf_dataset.csv", "ATP05_v3_0_6"),
        ("v2_3_1_trf_dataset.csv", "BASELINE_v2_3_1")
    ]

    all_results = {}
    for path, name in datasets:
        res = run_full_dataset_experiment(path, name)
        all_results[name] = res
        
        # Salvar checkpoint parcial
        with open("results_comparison_checkpoint.json", "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)

    with open("results_comparison.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    Path("results_comparison_checkpoint.json").unlink(missing_ok=True)

    # Restaurar o modelo campeão do dataset vencedor (maior ROC-AUC médio em CV) como modelo padrão
    best_ds_key = max(
        all_results.keys(),
        key=lambda k: all_results[k]["model_stats"][all_results[k]["best_model_name"]]["roc_auc_mean"],
    )
    overall_best_model = all_results[best_ds_key]["best_model_name"]
    overall_pkl_src = Path(f"model_{best_ds_key}_{overall_best_model}_ROC_AUC_SMOTE.pkl")
    overall_pkl_dst = Path(f"model_{overall_best_model}_ROC_AUC_SMOTE.pkl")
    if overall_pkl_src.exists():
        shutil.copy(overall_pkl_src, overall_pkl_dst)

    logger.info(
        f"=== COMPARATIVO FINAL CONCLUIDO === Dataset vencedor: {best_ds_key} | "
        f"Campeao: {overall_best_model} | "
        f"CV ROC-AUC={all_results[best_ds_key]['model_stats'][overall_best_model]['roc_auc_mean']:.4f} | "
        f"Hold-out ROC-AUC={all_results[best_ds_key]['champion_scores']['roc_auc_score']:.4f}"
    )

    print("\n" + "="*70)
    print(" EXPERIMENTOS CONCLUÍDOS COM SUCESSO! Resultados salvos em results_comparison.json")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
