import os

import numpy as np
import pandas as pd
import xgboost as xgb

from ayniy.model.model import Model
from ayniy.utils import Data


class ModelXGB(Model):
    def train(
        self,
        tr_x: pd.DataFrame,
        tr_y: pd.DataFrame,
        va_x: pd.DataFrame = None,
        va_y: pd.DataFrame = None,
        te_x: pd.DataFrame = None,
    ) -> None:

        # データのセット
        validation = va_x is not None
        dtrain = xgb.DMatrix(tr_x, label=tr_y)
        if validation:
            dvalid = xgb.DMatrix(va_x, label=va_y)

        # ハイパーパラメータの設定
        params = dict(self.params)
        num_round = params.pop("num_round")

        # 学習
        if validation:
            early_stopping_rounds = params.pop("early_stopping_rounds")
            watchlist = [(dtrain, "train"), (dvalid, "eval")]
            self.model = xgb.train(
                params,
                dtrain,
                num_round,
                evals=watchlist,
                early_stopping_rounds=early_stopping_rounds,
            )
        else:
            watchlist = [(dtrain, "train")]
            self.model = xgb.train(params, dtrain, num_round, evals=watchlist)

    def predict(self, te_x: pd.DataFrame) -> np.ndarray:
        dtest = xgb.DMatrix(te_x)
        return self.model.predict(dtest, ntree_limit=self.model.best_ntree_limit)  # type: ignore

    def save_model(self) -> None:
        model_path = os.path.join("../output/model", f"{self.run_fold_name}.model")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        # best_ntree_limitが消えるのを防ぐため、pickleで保存することとした
        Data.dump(self.model, model_path)

    def load_model(self) -> None:
        model_path = os.path.join("../output/model", f"{self.run_fold_name}.model")
        self.model = Data.load(model_path)
