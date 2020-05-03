import os

from ngboost import NGBRegressor, NGBClassifier

from ayniy.model.model import Model
from ayniy.utils import Data


# To do: https://github.com/upura/wids2020/blob/master/experiments/exp_62.ipynb
class ModelNgbClassifier(Model):

    def train(self, tr_x, tr_y, va_x=None, va_y=None, te_x=None):

        # ハイパーパラメータの設定
        params = dict(self.params)
        early_stopping_rounds = params.pop('early_stopping_rounds')

        self.model = NGBClassifier(**params)
        self.model.fit(tr_x.values, tr_y.astype(int).values,
                       va_x.values, va_y.astype(int).values,
                       early_stopping_rounds=early_stopping_rounds)

    def predict(self, te_x):
        return self.model.predict_proba(te_x.values)[:, 1]

    def save_model(self):
        model_path = os.path.join('../output/model', f'{self.run_fold_name}.model')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        Data.dump(self.model, model_path)

    def load_model(self):
        model_path = os.path.join('../output/model', f'{self.run_fold_name}.model')
        self.model = Data.load(model_path)


class ModelNgbRegressor(Model):

    def train(self, tr_x, tr_y, va_x=None, va_y=None, te_x=None):

        # ハイパーパラメータの設定
        params = dict(self.params)
        early_stopping_rounds = params.pop('early_stopping_rounds')

        self.model = NGBRegressor(**params)
        self.model.fit(tr_x.values, tr_y.astype(int).values,
                       va_x.values, va_y.astype(int).values,
                       early_stopping_rounds=early_stopping_rounds)

    def predict(self, te_x):
        return self.model.predict(te_x.values)

    def save_model(self):
        model_path = os.path.join('../output/model', f'{self.run_fold_name}.model')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        Data.dump(self.model, model_path)

    def load_model(self):
        model_path = os.path.join('../output/model', f'{self.run_fold_name}.model')
        self.model = Data.load(model_path)
