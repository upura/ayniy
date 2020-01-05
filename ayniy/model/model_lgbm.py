import os
import pandas as pd
import lightgbm as lgb

from ayniy.model import Model
from ayniy.utils import Data


class ModelLGBM(Model):

    def train(self, tr_x, tr_y, va_x=None, va_y=None):

        # データのセット
        validation = va_x is not None
        lgb_train = lgb.Dataset(tr_x, tr_y, categorical_feature=self.categorical_features)
        if validation:
            lgb_eval = lgb.Dataset(va_x, va_y, reference=lgb_train, categorical_feature=self.categorical_features)

        # ハイパーパラメータの設定
        params = dict(self.params)
        num_round = params.pop('num_boost_round')

        # 学習
        if validation:
            early_stopping_rounds = params.pop('early_stopping_rounds')
            self.model = lgb.train(
                params, lgb_train, num_round,
                valid_sets=[lgb_train, lgb_eval],
                verbose_eval=1000,
                early_stopping_rounds=early_stopping_rounds
            )
        else:
            self.model = lgb.train(
                params, lgb_train, num_round,
                valid_sets=[lgb_train],
                verbose_eval=1000
            )

    def predict(self, te_x):
        return self.model.predict(te_x, num_iteration=self.model.best_iteration)

    def feature_importance(self, te_x):
        fold_importance_df = pd.DataFrame()
        fold_importance_df["Feature"] = te_x.columns.values
        fold_importance_df["importance"] = self.model.feature_importance(importance_type='gain')
        return fold_importance_df

    def save_model(self):
        model_path = os.path.join('../output/model', f'{self.run_fold_name}.model')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        Data.dump(self.model, model_path)

    def load_model(self):
        model_path = os.path.join('../output/model', f'{self.run_fold_name}.model')
        self.model = Data.load(model_path)
