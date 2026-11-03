import os
import tempfile
import unittest
import numpy as np
import pandas as pd

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

from pds.pipeline import (
    RANDOM_STATE,
    _fit_leak_free_grid_search,
    _get_models_and_grids,
    do_benchmark,
    extract_model_metrics_scores,
    load_model,
    make_model,
    select_best_model,
)


class TestModelPipeline(unittest.TestCase):
    def setUp(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.v3_path = os.path.join(base_dir, "v3_0_6_trf_dataset.csv")

    def test_continuous_roc_auc_and_probability_brier(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1])
        y_proba = np.array([0.10, 0.55, 0.75, 0.90])

        scores = extract_model_metrics_scores(
            y_true, y_pred, y_score=y_proba, y_proba=y_proba
        )
        expected_auc = metrics.roc_auc_score(y_true, y_proba)
        label_auc = metrics.roc_auc_score(y_true, y_pred)
        expected_brier = metrics.brier_score_loss(y_true, y_proba)
        label_brier = metrics.brier_score_loss(y_true, y_pred)

        self.assertNotAlmostEqual(expected_auc, label_auc)
        self.assertNotAlmostEqual(expected_brier, label_brier)
        self.assertAlmostEqual(scores["roc_auc_score"], expected_auc, places=12)
        self.assertAlmostEqual(scores["brier_score_loss"], expected_brier, places=12)

    def test_svc_is_sklearn_svc_and_all_models_seeded(self):
        models, grids = _get_models_and_grids(grid_search=True)
        self.assertIsInstance(models["SVC"], SVC)
        self.assertNotIsInstance(models["SVC"], LinearSVC)
        self.assertTrue(models["SVC"].probability)
        for name, clf in models.items():
            self.assertEqual(clf.random_state, RANDOM_STATE, msg=f"{name} missing random_state")
            self.assertGreater(len(grids[name]), 0, msg=f"{name} grid should not be empty")

    def test_leak_free_inner_cv_matches_imbpipeline_gridsearch(self):
        df = pd.read_csv(self.v3_path, sep=";")
        features = ["LOC", "COM", "BLK", "NOF", "NOC", "APF", "AMC", "NER", "NEH", "CYC", "MAD"]
        X = df[features]
        y = df["BUG"]

        clf = SVC(probability=True, random_state=RANDOM_STATE)
        param_grid = {"C": [0.1, 1.0, 5.0], "kernel": ["linear", "rbf"]}

        fast_est = _fit_leak_free_grid_search(
            estimator=clf,
            param_grid=param_grid,
            X_train=X,
            y_train=y,
            data_balance="SMOTE",
            cv_criteria="roc_auc",
        )

        ref_pipe = ImbPipeline([
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("clf", SVC(probability=True, random_state=RANDOM_STATE)),
        ])
        inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        ref_gs = GridSearchCV(
            ref_pipe,
            {f"clf__{k}": v for k, v in param_grid.items()},
            cv=inner_cv,
            scoring="roc_auc",
            n_jobs=-1,
        ).fit(X, y)

        self.assertIsInstance(fast_est, ImbPipeline)
        np.testing.assert_allclose(
            fast_est.predict_proba(X),
            ref_gs.best_estimator_.predict_proba(X),
            atol=1e-12,
        )

    def test_reproducibility_and_serialization(self):
        res1 = do_benchmark(
            grid_search=False,
            dataset_path=self.v3_path,
            selected_models=["LRC", "RFC", "SVC", "GBC"],
        )
        res2 = do_benchmark(
            grid_search=False,
            dataset_path=self.v3_path,
            selected_models=["LRC", "RFC", "SVC", "GBC"],
        )

        for fold in res1:
            for model_name in res1[fold]:
                self.assertAlmostEqual(
                    res1[fold][model_name]["score"]["roc_auc_score"],
                    res2[fold][model_name]["score"]["roc_auc_score"],
                    places=12,
                )
                self.assertAlmostEqual(
                    res1[fold][model_name]["score"]["brier_score_loss"],
                    res2[fold][model_name]["score"]["brier_score_loss"],
                    places=12,
                )

        best = select_best_model(res1)
        self.assertIn(best, ["LRC", "RFC", "SVC", "GBC"])

        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                os.chdir(tmpdir)
                scores = make_model(
                    grid_search=False,
                    dataset_path=self.v3_path,
                    selected_model=best,
                )
                pkl_path = os.path.join(tmpdir, f"model_{best}_ROC_AUC_SMOTE.pkl")
                self.assertTrue(os.path.exists(pkl_path))
                loaded = load_model(pkl_path)
                df = pd.read_csv(self.v3_path, sep=";")
                features = ["LOC", "COM", "BLK", "NOF", "NOC", "APF", "AMC", "NER", "NEH", "CYC", "MAD"]
                proba = loaded.predict_proba(df[features].iloc[:5])
                self.assertEqual(proba.shape, (5, 2))
                self.assertGreater(scores["roc_auc_score"], 0.5)
            finally:
                os.chdir(cwd)


if __name__ == "__main__":
    unittest.main()
