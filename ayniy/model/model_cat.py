import os

import catboost as cb
import numpy as np
import pandas as pd

from ayniy.model.model import Model
from ayniy.utils import Data


class ModelCatClassifier(Model):
    def train(
        self,
        tr_x: pd.DataFrame,
        tr_y: pd.DataFrame,
        va_x: pd.DataFrame = None,
        va_y: pd.DataFrame = None,
        te_x: pd.DataFrame = None,
    ) -> None:

        # ハイパーパラメータの設定
        params = dict(self.params)
        self.model: cb.CatBoostClassifier = cb.CatBoostClassifier(**params)

        self.model.fit(
            tr_x,
            tr_y,
            cat_features=self.categorical_features,
            eval_set=(va_x, va_y),
            verbose=100,
            use_best_model=True,
            plot=False,
        )

    def predict(self, te_x: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(te_x)[:, 1]

    def save_model(self) -> None:
        model_path = os.path.join("../output/model", f"{self.run_fold_name}.model")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        Data.dump(self.model, model_path)

    def load_model(self) -> None:
        model_path = os.path.join("../output/model", f"{self.run_fold_name}.model")
        self.model = Data.load(model_path)


class ModelCatRegressor(Model):
    def train(
        self,
        tr_x: pd.DataFrame,
        tr_y: pd.DataFrame,
        va_x: pd.DataFrame = None,
        va_y: pd.DataFrame = None,
        te_x: pd.DataFrame = None,
    ) -> None:

        # ハイパーパラメータの設定
        params = dict(self.params)
        self.model: cb.CatBoostRegressor = cb.CatBoostRegressor(**params)

        self.model.fit(
            tr_x,
            tr_y,
            cat_features=self.categorical_features,
            eval_set=(va_x, va_y),
            verbose=100,
            use_best_model=True,
            plot=False,
        )

    def predict(self, te_x: pd.DataFrame) -> np.ndarray:
        return self.model.predict(te_x)

    def save_model(self) -> None:
        model_path = os.path.join("../output/model", f"{self.run_fold_name}.model")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        Data.dump(self.model, model_path)

    def load_model(self) -> None:
        model_path = os.path.join("../output/model", f"{self.run_fold_name}.model")
        self.model = Data.load(model_path)
