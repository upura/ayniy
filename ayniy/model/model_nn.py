import os
from typing import Any, Dict

from keras import backend as K
from keras.callbacks import EarlyStopping, Callback, ModelCheckpoint
from keras.layers.advanced_activations import PReLU
from keras.layers.core import Dense, Dropout
from keras.layers import Input, Embedding, Flatten, concatenate, Multiply
from keras.layers.normalization import BatchNormalization
from keras.models import load_model
from keras.models import Model as kerasModel
import numpy as np
import pandas as pd
import tensorflow as tf

from ayniy.model.model import Model as oriModel

# tensorflowの警告抑制
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def get_keras_data(
    df: pd.DataFrame, numerical_features: Any, categorical_features: Any
) -> pd.DataFrame:
    X = {"numerical": df[numerical_features].values}
    for c in categorical_features:
        X[c] = df[c]
    return X


class CyclicLR(Callback):
    def __init__(
        self,
        base_lr: float = 0.001,
        max_lr: float = 0.006,
        step_size: float = 2000.0,
        mode: str = "triangular",
        gamma: float = 1.0,
        scale_fn: Any = None,
        scale_mode: str = "cycle",
    ):
        super(CyclicLR, self).__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn is None:
            if self.mode == "triangular":
                self.scale_fn = lambda x: 1.0
                self.scale_mode = "cycle"
            elif self.mode == "triangular2":
                self.scale_fn = lambda x: 1 / (2.0 ** (x - 1))
                self.scale_mode = "cycle"
            elif self.mode == "exp_range":
                self.scale_fn = lambda x: gamma ** (x)
                self.scale_mode = "iterations"
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.0
        self.trn_iterations = 0.0
        self.history: Dict = {}

        self._reset()

    def _reset(
        self,
        new_base_lr: float = None,
        new_max_lr: float = None,
        new_step_size: float = None,
    ) -> None:
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr is not None:
            self.base_lr = new_base_lr
        if new_max_lr is not None:
            self.max_lr = new_max_lr
        if new_step_size is not None:
            self.step_size = new_step_size
        self.clr_iterations = 0.0

    def clr(self) -> float:
        cycle = np.floor(1 + self.clr_iterations / (2 * self.step_size))
        x = np.abs(self.clr_iterations / self.step_size - 2 * cycle + 1)
        if self.scale_mode == "cycle":
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(
                0, (1 - x)
            ) * self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(
                0, (1 - x)
            ) * self.scale_fn(self.clr_iterations)

    def on_train_begin(self, logs: Dict = {}) -> None:
        logs = logs or {}

        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())

    def on_batch_end(self, epoch: int, logs: Dict = None) -> None:
        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1

        self.history.setdefault("lr", []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault("iterations", []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        K.set_value(self.model.optimizer.lr, self.clr())


def se_block(input: Any, channels: int, r: int = 8) -> Any:
    x = Dense(channels // r, activation="relu")(input)
    x = Dense(channels, activation="sigmoid")(x)
    return Multiply()([input, x])


class ModelNN(oriModel):
    def train(
        self,
        tr_x: pd.DataFrame,
        tr_y: pd.DataFrame,
        va_x: pd.DataFrame = None,
        va_y: pd.DataFrame = None,
        te_x: pd.DataFrame = None,
    ) -> None:
        # データのセット・スケーリング
        numerical_features = [
            c for c in tr_x.columns if c not in self.categorical_features   # type: ignore
        ]
        validation = va_x is not None

        # パラメータ
        dropout = self.params["dropout"]
        nb_epoch = self.params["nb_epoch"]
        patience = self.params["patience"]

        # モデルの構築
        inp_cats = []
        embs = []
        data = pd.concat([tr_x, va_x, te_x]).reset_index(drop=True)

        for c in self.categorical_features:  # type: ignore
            inp_cat = Input(shape=[1], name=c)
            inp_cats.append(inp_cat)
            embs.append((Embedding(data[c].max() + 1, 4)(inp_cat)))

        cats = Flatten()(concatenate(embs))
        cats = Dense(4, activation="linear")(cats)
        cats = BatchNormalization()(cats)
        cats = PReLU()(cats)

        inp_numerical = Input(shape=[len(numerical_features)], name="numerical")
        nums = Dense(500, activation="linear")(inp_numerical)
        nums = BatchNormalization()(nums)
        nums = PReLU()(nums)
        x = Dropout(dropout)(nums)

        x = concatenate([nums, cats])
        x = se_block(nums, 32)
        x = BatchNormalization()(x)
        x = Dropout(dropout / 2)(x)
        x = Dense(250, activation="relu")(x)
        x = Dense(125, activation="relu")(x)
        x = Dense(64, activation="relu")(x)
        x = Dense(40, activation="relu")(x)
        out = Dense(31, activation="softmax", name="out1")(x)

        model = kerasModel(inputs=[inp_numerical], outputs=out)
        model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")
        # print(model.summary())
        n_train = len(tr_x)
        batch_size_nn = 256

        tr_x = get_keras_data(tr_x, numerical_features, self.categorical_features)
        va_x = get_keras_data(va_x, numerical_features, self.categorical_features)

        clr_tri = CyclicLR(
            base_lr=1e-5,
            max_lr=1e-2,
            step_size=n_train // batch_size_nn,
            mode="triangular2",
        )
        ckpt = ModelCheckpoint(
            f"../output/model/model_{self.run_fold_name}.hdf5",
            save_best_only=True,
            monitor="val_loss",
            mode="min",
        )
        if validation:
            early_stopping = EarlyStopping(
                monitor="val_loss",
                patience=patience,
                verbose=1,
                restore_best_weights=True,
            )
            model.fit(
                tr_x,
                tr_y,
                epochs=nb_epoch,
                batch_size=batch_size_nn,
                verbose=2,
                validation_data=(va_x, va_y),
                callbacks=[ckpt, clr_tri, early_stopping],
            )
        else:
            model.fit(
                tr_x, tr_y, nb_epoch=nb_epoch, batch_size=batch_size_nn, verbose=2
            )
        model.load_weights(f"../output/model/model_{self.run_fold_name}.hdf5")  # type: ignore

        # モデル・スケーラーの保持
        self.model = model

    def predict(self, te_x: pd.DataFrame) -> np.ndarray:
        numerical_features = [
            c for c in te_x.columns if c not in self.categorical_features   # type: ignore
        ]
        te_x = get_keras_data(te_x, numerical_features, self.categorical_features)
        pred = self.model.predict(te_x)  # type: ignore
        return pred

    def save_model(self) -> None:
        model_path = os.path.join("../output/model", f"{self.run_fold_name}.h5")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.model.save(model_path)  # type: ignore

    def load_model(self) -> None:
        model_path = os.path.join("../output/model", f"{self.run_fold_name}.h5")
        self.model = load_model(model_path)
