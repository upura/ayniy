import os
import catboost as cb

from ayniy.model import Model
from ayniy.utils import Data


class ModelCAT(Model):

    def train(self, tr_x, tr_y, va_x=None, va_y=None):

        # ハイパーパラメータの設定
        params = dict(self.params)
        self.model = cb.CatBoostRegressor(**params)

        self.model.fit(tr_x, tr_y,
                       cat_features=self.categorical_features,
                       eval_set=(va_x, va_y),
                       verbose=100,
                       plot=False)

    def predict(self, te_x):
        return self.model.predict(te_x)

    def save_model(self):
        model_path = os.path.join('../output/model', f'{self.run_fold_name}.model')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        Data.dump(self.model, model_path)

    def load_model(self):
        model_path = os.path.join('../output/model', f'{self.run_fold_name}.model')
        self.model = Data.load(model_path)
