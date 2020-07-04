import os
import json

import lightgbm as lgb
import numpy as np
import optuna.integration.lightgbm as optuna_lgb
import pandas as pd
from scipy.misc import derivative

from ayniy.model.model import Model
from ayniy.utils import Data


class ModelLGBM(Model):

    def train(self, tr_x, tr_y, va_x=None, va_y=None, te_x=None):

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
                verbose_eval=500,
                early_stopping_rounds=early_stopping_rounds
            )
        else:
            self.model = lgb.train(
                params, lgb_train, num_round,
                valid_sets=[lgb_train],
                verbose_eval=500
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


class ModelOptunaLGBM(Model):

    def train(self, tr_x, tr_y, va_x=None, va_y=None, te_x=None):

        # データのセット
        validation = va_x is not None
        lgb_train = optuna_lgb.Dataset(tr_x, tr_y, categorical_feature=self.categorical_features, free_raw_data=False)
        if validation:
            lgb_eval = optuna_lgb.Dataset(va_x, va_y, reference=lgb_train, categorical_feature=self.categorical_features,
                                          free_raw_data=False)

        # ハイパーパラメータの設定
        params = dict(self.params)
        num_round = params.pop('num_boost_round')
        best_params, tuning_history = dict(), list()

        # 学習
        if validation:
            early_stopping_rounds = params.pop('early_stopping_rounds')
            self.model = optuna_lgb.train(
                params, lgb_train, num_round,
                valid_sets=[lgb_train, lgb_eval],
                verbose_eval=1000,
                early_stopping_rounds=early_stopping_rounds,
                best_params=best_params,
                tuning_history=tuning_history
            )
        else:
            self.model = optuna_lgb.train(
                params, lgb_train, num_round,
                valid_sets=[lgb_train],
                verbose_eval=1000,
                best_params=best_params,
                tuning_history=tuning_history
            )
        print('Best Params:', best_params)
        with open(f'../output/model/{self.run_fold_name}_best_params.json', 'w') as f:
            json.dump(best_params, f, indent=4, separators=(',', ': '))

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


def focal_loss_lgb(y_pred, dtrain, alpha, gamma):
    a, g = alpha, gamma
    y_true = dtrain.label

    def fl(x, t):
        p = 1 / (1 + np.exp(-x))
        return -(a * t + (1 - a) * (1 - t)) * ((1 - (t * p + (1 - t) * (1 - p))) ** g) * (t * np.log(p) + (1 - t) * np.log(1 - p))

    def partial_fl(x):
        return fl(x, y_true)

    grad = derivative(partial_fl, y_pred, n=1, dx=1e-6)
    hess = derivative(partial_fl, y_pred, n=2, dx=1e-6)
    return grad, hess


def focal_loss_lgb_eval_error(y_pred, dtrain, alpha, gamma):
    a, g = alpha, gamma
    y_true = dtrain.label
    p = 1 / (1 + np.exp(-y_pred))
    loss = -(a * y_true + (1 - a) * (1 - y_true)) * ((1 - (y_true * p + (1 - y_true) * (1 - p))) ** g) * (y_true * np.log(p) + (1 - y_true) * np.log(1 - p))
    # (eval_name, eval_result, is_higher_better)
    return 'focal_loss', np.mean(loss), False


class ModelFocalLGBM(Model):

    def train(self, tr_x, tr_y, va_x=None, va_y=None, te_x=None):

        def focal_loss(x, y):
            return focal_loss_lgb(x, y, alpha=0.25, gamma=1.)

        def focal_loss_eval(x, y):
            return focal_loss_lgb_eval_error(x, y, alpha=0.25, gamma=1.)

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
                verbose_eval=500,
                early_stopping_rounds=early_stopping_rounds,
                fobj=focal_loss, feval=focal_loss_eval
            )
        else:
            self.model = lgb.train(
                params, lgb_train, num_round,
                valid_sets=[lgb_train],
                verbose_eval=500,
                fobj=focal_loss, feval=focal_loss_eval
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
