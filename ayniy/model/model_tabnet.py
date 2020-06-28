import os

from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor

from ayniy.model.model import Model
from ayniy.utils import Data


class ModelTabNetClassifier(Model):

    def train(self, tr_x, tr_y, va_x=None, va_y=None, te_x=None):

        categorical_dims = {}
        for col in self.categorical_features:
            tr_x[col] = tr_x[col].fillna("unk")
            va_x[col] = va_x[col].fillna("unk")
            te_x[col] = te_x[col].fillna("unk")
            categorical_dims[col] = len(set(tr_x[col].values) | set(va_x[col].values) | set(te_x[col].values))

        cat_idxs = [i for i, f in enumerate(tr_x.columns) if f in self.categorical_features]
        cat_dims = [categorical_dims[f] for i, f in enumerate(tr_x.columns) if f in self.categorical_features]
        cat_emb_dim = [10 for _ in categorical_dims]

        for col in tr_x.columns:
            tr_x[col] = tr_x[col].fillna(tr_x[col].mean())
            va_x[col] = va_x[col].fillna(tr_x[col].mean())
            te_x[col] = te_x[col].fillna(tr_x[col].mean())

        self.model = TabNetClassifier(cat_dims=cat_dims, cat_emb_dim=cat_emb_dim, cat_idxs=cat_idxs)
        self.model.fit(X_train=tr_x.values, y_train=tr_y.values,
                       X_valid=va_x.values, y_valid=va_y.values,
                       max_epochs=1000,
                       patience=50,
                       batch_size=1024,
                       virtual_batch_size=128)

    def predict(self, te_x):
        return self.model.predict_proba(te_x.values)[:, 1].reshape(-1, )

    def save_model(self):
        model_path = os.path.join('../output/model', f'{self.run_fold_name}.model')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        Data.dump(self.model, model_path)

    def load_model(self):
        model_path = os.path.join('../output/model', f'{self.run_fold_name}.model')
        self.model = Data.load(model_path)


class ModelTabNetRegressor(Model):

    def train(self, tr_x, tr_y, va_x=None, va_y=None, te_x=None):

        categorical_dims = {}
        for col in self.categorical_features:
            tr_x[col] = tr_x[col].fillna("unk")
            va_x[col] = va_x[col].fillna("unk")
            te_x[col] = te_x[col].fillna("unk")
            categorical_dims[col] = len(set(tr_x[col].values) | set(va_x[col].values) | set(te_x[col].values))

        cat_idxs = [i for i, f in enumerate(tr_x.columns) if f in self.categorical_features]
        cat_dims = [categorical_dims[f] for i, f in enumerate(tr_x.columns) if f in self.categorical_features]
        cat_emb_dim = [10 for _ in categorical_dims]

        for col in tr_x.columns:
            tr_x[col] = tr_x[col].fillna(tr_x[col].mean())
            va_x[col] = va_x[col].fillna(tr_x[col].mean())
            te_x[col] = te_x[col].fillna(tr_x[col].mean())

        self.model = TabNetRegressor(cat_dims=cat_dims, cat_emb_dim=cat_emb_dim, cat_idxs=cat_idxs)
        self.model.fit(X_train=tr_x.values, y_train=tr_y.values.reshape(-1, 1),
                       X_valid=va_x.values, y_valid=va_y.values.reshape(-1, 1),
                       max_epochs=1000,
                       patience=50,
                       batch_size=1024,
                       virtual_batch_size=128)

    def predict(self, te_x):
        return self.model.predict(te_x.values).reshape(-1, )

    def save_model(self):
        model_path = os.path.join('../output/model', f'{self.run_fold_name}.model')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        Data.dump(self.model, model_path)

    def load_model(self):
        model_path = os.path.join('../output/model', f'{self.run_fold_name}.model')
        self.model = Data.load(model_path)
