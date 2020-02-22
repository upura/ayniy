import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, mean_absolute_error, roc_auc_score
from typing import Callable
from typing import Union
import matplotlib.pyplot as plt
import seaborn as sns

from ayniy.model.model import Model
from ayniy.utils import Logger, Data


logger = Logger()


class Runner:

    def __init__(self, run_name: str, model_cls: Callable[[str, dict], Model],
                 X_train, X_test, y_train, evaluation_metric,
                 params: dict, categorical_features=None, n_fold=5):
        self.run_name = run_name
        self.model_cls = model_cls
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.evaluation_metric = evaluation_metric
        self.params = params
        self.categorical_features = categorical_features
        self.n_fold = n_fold

    def train_fold(self, i_fold: int):
        """クロスバリデーションでのfoldを指定して学習・評価を行う

        他のメソッドから呼び出すほか、単体でも確認やパラメータ調整に用いる

        :param i_fold: foldの番号（すべてのときには'all'とする）
        :return: （モデルのインスタンス、レコードのインデックス、予測値、評価によるスコア）のタプル
        """
        # 学習データの読込
        X_train = self.X_train
        y_train = self.y_train

        # 学習データ・バリデーションデータをセットする
        fold_id = pd.read_csv('../input/fold_id.csv', header=None)
        tr_idx = X_train.loc[(fold_id != i_fold)[0]].index
        va_idx = X_train.loc[(fold_id == i_fold)[0]].index
        X_tr, y_tr = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
        X_val, y_val = X_train.iloc[va_idx], y_train.iloc[va_idx]

        # 学習を行う
        model = self.build_model(i_fold)
        model.train(X_tr, y_tr, X_val, y_val, self.X_test)

        # バリデーションデータへの予測・評価を行う
        pred_val = model.predict(X_val)

        if self.evaluation_metric == 'log_loss':
            score = log_loss(y_val, pred_val, eps=1e-15, normalize=True)
        elif self.evaluation_metric == 'mean_absolute_error':
            score = mean_absolute_error(y_val, pred_val)
        elif self.evaluation_metric == 'auc':
            score = roc_auc_score(y_val, pred_val)

        # モデル、インデックス、予測値、評価を返す
        return model, va_idx, pred_val, score

    def run_train_cv(self) -> None:
        """クロスバリデーションでの学習・評価を行う

        学習・評価とともに、各foldのモデルの保存、スコアのログ出力についても行う
        """
        logger.info(f'{self.run_name} - start training cv')

        scores = []
        va_idxes = []
        preds = []

        # 各foldで学習を行う
        for i_fold in range(self.n_fold):
            # 学習を行う
            logger.info(f'{self.run_name} fold {i_fold} - start training')
            model, va_idx, va_pred, score = self.train_fold(i_fold)
            logger.info(f'{self.run_name} fold {i_fold} - end training - score {score}')

            # モデルを保存する
            model.save_model()

            # 結果を保持する
            va_idxes.append(va_idx)
            scores.append(score)
            preds.append(va_pred)

        # 各foldの結果をまとめる
        va_idxes = np.concatenate(va_idxes)
        order = np.argsort(va_idxes)
        preds = np.concatenate(preds, axis=0)
        preds = preds[order]

        logger.info(f'{self.run_name} - end training cv - score {np.mean(scores)}')

        # 予測結果の保存
        Data.dump(preds, f'../output/pred/{self.run_name}-train.pkl')

        # 評価結果の保存
        logger.result_scores(self.run_name, scores)

    def run_predict_cv(self) -> None:
        """クロスバリデーションで学習した各foldのモデルの平均により、テストデータの予測を行う

        あらかじめrun_train_cvを実行しておく必要がある
        """
        show_feature_importance = ('lgbm' in self.run_name)

        logger.info(f'{self.run_name} - start prediction cv')
        X_test = self.X_test
        preds = []
        if show_feature_importance:
            feature_importances = pd.DataFrame()

        # 各foldのモデルで予測を行う
        for i_fold in range(self.n_fold):
            logger.info(f'{self.run_name} - start prediction fold:{i_fold}')
            model = self.build_model(i_fold)
            model.load_model()
            pred = model.predict(X_test)
            preds.append(pred)
            logger.info(f'{self.run_name} - end prediction fold:{i_fold}')
            if show_feature_importance:
                feature_importances = pd.concat([
                    feature_importances,
                    model.feature_importance(X_test)
                ], axis=0)

        # 予測の平均値を出力する
        pred_avg = np.mean(preds, axis=0)

        # 予測結果の保存
        Data.dump(pred_avg, f'../output/pred/{self.run_name}-test.pkl')

        logger.info(f'{self.run_name} - end prediction cv')

        # 特徴量の重要度
        if show_feature_importance:
            cols = feature_importances.groupby('Feature').mean().sort_values(by="importance", ascending=False)[:200].index
            pd.DataFrame(cols).to_csv(f'../output/importance/{self.run_name}-fi.csv')

            best_features = feature_importances.loc[feature_importances.Feature.isin(cols)]
            plt.figure(figsize=(14, 26))
            sns.barplot(x="importance", y="Feature", data=best_features.sort_values(by="importance", ascending=False))
            plt.title('LightGBM Features (averaged over folds)')
            plt.tight_layout()
            plt.savefig(f'../output/importance/{self.run_name}-fi.png')
            plt.show()

    def build_model(self, i_fold: Union[int, str]) -> Model:
        """クロスバリデーションでのfoldを指定して、モデルの作成を行う

        :param i_fold: foldの番号
        :return: モデルのインスタンス
        """
        # ラン名、fold、モデルのクラスからモデルを作成する
        run_fold_name = f'{self.run_name}-{i_fold}'
        return self.model_cls(run_fold_name, self.params, self.categorical_features)


class ResRunner:

    def __init__(self, run_name: str, model_cls: Callable[[str, dict], Model],
                 X_train, X_test, y_train, evaluation_metric, res_threshold,
                 params: dict, categorical_features=None, n_fold=5):
        self.run_name = run_name
        self.model_cls = model_cls
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.evaluation_metric = evaluation_metric
        self.res_threshold = res_threshold
        self.params = params
        self.categorical_features = categorical_features
        self.n_fold = n_fold

    def train_fold(self, i_fold: int):
        """クロスバリデーションでのfoldを指定して学習・評価を行う

        他のメソッドから呼び出すほか、単体でも確認やパラメータ調整に用いる

        :param i_fold: foldの番号（すべてのときには'all'とする）
        :return: （モデルのインスタンス、レコードのインデックス、予測値、評価によるスコア）のタプル
        """
        # 学習データの読込
        X_train = self.X_train
        y_train = self.y_train

        # 学習データ・バリデーションデータをセットする
        fold_id = pd.read_csv('../input/fold_id.csv', header=None)
        tr_idx = X_train.loc[(fold_id != i_fold)[0]].index
        va_idx = X_train.loc[(fold_id == i_fold)[0]].index
        X_tr, y_tr = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
        X_val, y_val = X_train.iloc[va_idx], y_train.iloc[va_idx]

        # 残差でダウンサンプリング
        X_tr = X_tr.loc[(X_tr['res'] < self.res_threshold).values]
        y_tr = y_tr.loc[(X_tr['res'] < self.res_threshold).values]
        X_tr.drop('res', axis=1, inplace=True)
        X_val.drop('res', axis=1, inplace=True)

        # 学習を行う
        model = self.build_model(i_fold)
        model.train(X_tr, y_tr, X_val, y_val, self.X_test)

        # バリデーションデータへの予測・評価を行う
        pred_val = model.predict(X_val)

        if self.evaluation_metric == 'log_loss':
            score = log_loss(y_val, pred_val, eps=1e-15, normalize=True)
        elif self.evaluation_metric == 'mean_absolute_error':
            score = mean_absolute_error(y_val, pred_val)
        elif self.evaluation_metric == 'auc':
            score = roc_auc_score(y_val, pred_val)

        # モデル、インデックス、予測値、評価を返す
        return model, va_idx, pred_val, score

    def run_train_cv(self) -> None:
        """クロスバリデーションでの学習・評価を行う

        学習・評価とともに、各foldのモデルの保存、スコアのログ出力についても行う
        """
        logger.info(f'{self.run_name} - start training cv')

        scores = []
        va_idxes = []
        preds = []

        # 各foldで学習を行う
        for i_fold in range(self.n_fold):
            # 学習を行う
            logger.info(f'{self.run_name} fold {i_fold} - start training')
            model, va_idx, va_pred, score = self.train_fold(i_fold)
            logger.info(f'{self.run_name} fold {i_fold} - end training - score {score}')

            # モデルを保存する
            model.save_model()

            # 結果を保持する
            va_idxes.append(va_idx)
            scores.append(score)
            preds.append(va_pred)

        # 各foldの結果をまとめる
        va_idxes = np.concatenate(va_idxes)
        order = np.argsort(va_idxes)
        preds = np.concatenate(preds, axis=0)
        preds = preds[order]

        logger.info(f'{self.run_name} - end training cv - score {np.mean(scores)}')

        # 予測結果の保存
        Data.dump(preds, f'../output/pred/{self.run_name}-train.pkl')

        # 評価結果の保存
        logger.result_scores(self.run_name, scores)

    def run_predict_cv(self) -> None:
        """クロスバリデーションで学習した各foldのモデルの平均により、テストデータの予測を行う

        あらかじめrun_train_cvを実行しておく必要がある
        """
        show_feature_importance = ('lgbm' in self.run_name)

        logger.info(f'{self.run_name} - start prediction cv')
        X_test = self.X_test
        preds = []
        if show_feature_importance:
            feature_importances = pd.DataFrame()

        # 各foldのモデルで予測を行う
        for i_fold in range(self.n_fold):
            logger.info(f'{self.run_name} - start prediction fold:{i_fold}')
            model = self.build_model(i_fold)
            model.load_model()
            pred = model.predict(X_test)
            preds.append(pred)
            logger.info(f'{self.run_name} - end prediction fold:{i_fold}')
            if show_feature_importance:
                feature_importances = pd.concat([
                    feature_importances,
                    model.feature_importance(X_test)
                ], axis=0)

        # 予測の平均値を出力する
        pred_avg = np.mean(preds, axis=0)

        # 予測結果の保存
        Data.dump(pred_avg, f'../output/pred/{self.run_name}-test.pkl')

        logger.info(f'{self.run_name} - end prediction cv')

        # 特徴量の重要度
        if show_feature_importance:
            cols = feature_importances.groupby('Feature').mean().sort_values(by="importance", ascending=False)[:200].index
            pd.DataFrame(cols).to_csv(f'../output/importance/{self.run_name}-fi.csv')

            best_features = feature_importances.loc[feature_importances.Feature.isin(cols)]
            plt.figure(figsize=(14, 26))
            sns.barplot(x="importance", y="Feature", data=best_features.sort_values(by="importance", ascending=False))
            plt.title('LightGBM Features (averaged over folds)')
            plt.tight_layout()
            plt.savefig(f'../output/importance/{self.run_name}-fi.png')
            plt.show()

    def build_model(self, i_fold: Union[int, str]) -> Model:
        """クロスバリデーションでのfoldを指定して、モデルの作成を行う

        :param i_fold: foldの番号
        :return: モデルのインスタンス
        """
        # ラン名、fold、モデルのクラスからモデルを作成する
        run_fold_name = f'{self.run_name}-{i_fold}'
        return self.model_cls(run_fold_name, self.params, self.categorical_features)


class PseudoRunner:

    def __init__(self, run_name: str, model_cls: Callable[[str, dict], Model],
                 X_train, X_test, y_train, y_test_pred, evaluation_metric,
                 params: dict, pl_threshold=None, categorical_features=None, n_fold=5):
        self.run_name = run_name
        self.model_cls = model_cls
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test_pred = y_test_pred
        self.evaluation_metric = evaluation_metric
        self.params = params
        self.pl_threshold = pl_threshold
        self.categorical_features = categorical_features
        self.n_fold = n_fold

    def train_fold(self, i_fold: int):
        """クロスバリデーションでのfoldを指定して学習・評価を行う

        他のメソッドから呼び出すほか、単体でも確認やパラメータ調整に用いる

        :param i_fold: foldの番号（すべてのときには'all'とする）
        :return: （モデルのインスタンス、レコードのインデックス、予測値、評価によるスコア）のタプル
        """
        # 学習データの読込
        X_train = self.X_train
        y_train = self.y_train

        # 学習データ・バリデーションデータをセットする
        fold_id = pd.read_csv('../input/fold_id.csv', header=None)
        tr_idx = X_train.loc[(fold_id != i_fold)[0]].index
        va_idx = X_train.loc[(fold_id == i_fold)[0]].index
        X_tr, y_tr = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
        X_val, y_val = X_train.iloc[va_idx], y_train.iloc[va_idx]

        # Pseudo Lebeling
        if self.pl_threshold:
            X_add = self.X_test.loc[(self.y_test_pred < self.pl_threshold) | (self.y_test_pred > 1 - self.pl_threshold)]
            y_add = pd.DataFrame(self.y_test_pred).loc[(self.y_test_pred < self.pl_threshold) | (self.y_test_pred > 1 - self.pl_threshold)]
            y_add = pd.DataFrame(([1 if ya > 0.5 else 0 for ya in y_add[0]]))
        else:
            X_add = self.X_test
            y_add = pd.DataFrame(self.y_test_pred)
        print(f'added X_test: {len(X_add)}')
        X_tr = pd.concat([X_tr, X_add])
        y_tr = pd.concat([y_tr, y_add])

        # 学習を行う
        model = self.build_model(i_fold)
        model.train(X_tr, y_tr, X_val, y_val, self.X_test)

        # バリデーションデータへの予測・評価を行う
        pred_val = model.predict(X_val)

        if self.evaluation_metric == 'log_loss':
            score = log_loss(y_val, pred_val, eps=1e-15, normalize=True)
        elif self.evaluation_metric == 'mean_absolute_error':
            score = mean_absolute_error(y_val, pred_val)
        elif self.evaluation_metric == 'auc':
            score = roc_auc_score(y_val, pred_val)

        # モデル、インデックス、予測値、評価を返す
        return model, va_idx, pred_val, score

    def run_train_cv(self) -> None:
        """クロスバリデーションでの学習・評価を行う

        学習・評価とともに、各foldのモデルの保存、スコアのログ出力についても行う
        """
        logger.info(f'{self.run_name} - start training cv')

        scores = []
        va_idxes = []
        preds = []

        # 各foldで学習を行う
        for i_fold in range(self.n_fold):
            # 学習を行う
            logger.info(f'{self.run_name} fold {i_fold} - start training')
            model, va_idx, va_pred, score = self.train_fold(i_fold)
            logger.info(f'{self.run_name} fold {i_fold} - end training - score {score}')

            # モデルを保存する
            model.save_model()

            # 結果を保持する
            va_idxes.append(va_idx)
            scores.append(score)
            preds.append(va_pred)

        # 各foldの結果をまとめる
        va_idxes = np.concatenate(va_idxes)
        order = np.argsort(va_idxes)
        preds = np.concatenate(preds, axis=0)
        preds = preds[order]

        logger.info(f'{self.run_name} - end training cv - score {np.mean(scores)}')

        # 予測結果の保存
        Data.dump(preds, f'../output/pred/{self.run_name}-train.pkl')

        # 評価結果の保存
        logger.result_scores(self.run_name, scores)

    def run_predict_cv(self) -> None:
        """クロスバリデーションで学習した各foldのモデルの平均により、テストデータの予測を行う

        あらかじめrun_train_cvを実行しておく必要がある
        """
        show_feature_importance = ('lgbm' in self.run_name)

        logger.info(f'{self.run_name} - start prediction cv')
        X_test = self.X_test
        preds = []
        if show_feature_importance:
            feature_importances = pd.DataFrame()

        # 各foldのモデルで予測を行う
        for i_fold in range(self.n_fold):
            logger.info(f'{self.run_name} - start prediction fold:{i_fold}')
            model = self.build_model(i_fold)
            model.load_model()
            pred = model.predict(X_test)
            preds.append(pred)
            logger.info(f'{self.run_name} - end prediction fold:{i_fold}')
            if show_feature_importance:
                feature_importances = pd.concat([
                    feature_importances,
                    model.feature_importance(X_test)
                ], axis=0)

        # 予測の平均値を出力する
        pred_avg = np.mean(preds, axis=0)

        # 予測結果の保存
        Data.dump(pred_avg, f'../output/pred/{self.run_name}-test.pkl')

        logger.info(f'{self.run_name} - end prediction cv')

        # 特徴量の重要度
        if show_feature_importance:
            cols = feature_importances.groupby('Feature').mean().sort_values(by="importance", ascending=False)[:200].index
            pd.DataFrame(cols).to_csv(f'../output/importance/{self.run_name}-fi.csv')

            best_features = feature_importances.loc[feature_importances.Feature.isin(cols)]
            plt.figure(figsize=(14, 26))
            sns.barplot(x="importance", y="Feature", data=best_features.sort_values(by="importance", ascending=False))
            plt.title('LightGBM Features (averaged over folds)')
            plt.tight_layout()
            plt.savefig(f'../output/importance/{self.run_name}-fi.png')
            plt.show()

    def build_model(self, i_fold: Union[int, str]) -> Model:
        """クロスバリデーションでのfoldを指定して、モデルの作成を行う

        :param i_fold: foldの番号
        :return: モデルのインスタンス
        """
        # ラン名、fold、モデルのクラスからモデルを作成する
        run_fold_name = f'{self.run_name}-{i_fold}'
        return self.model_cls(run_fold_name, self.params, self.categorical_features)
