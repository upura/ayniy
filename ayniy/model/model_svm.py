import os

import numpy as np
import pandas as pd
from sklearn.svm import SVC

from ayniy.model.model import Model
from ayniy.utils import Data


class ModelSVM(Model):
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
        self.model: SVC = SVC(**params)
        self.model.fit(tr_x, tr_y)

    def predict(self, te_x: pd.DataFrame) -> np.ndarray:
        return self.model.predict(te_x)

    def save_model(self) -> None:
        model_path = os.path.join("../output/model", f"{self.run_fold_name}.model")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        Data.dump(self.model, model_path)

    def load_model(self) -> None:
        model_path = os.path.join("../output/model", f"{self.run_fold_name}.model")
        self.model = Data.load(model_path)
