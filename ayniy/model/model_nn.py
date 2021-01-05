import os
from typing import Any

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model as kerasModel
import numpy as np
import pandas as pd

from ayniy.model.model import Model as oriModel


def get_keras_data(
    df: pd.DataFrame, numerical_features: Any, categorical_features: Any
) -> pd.DataFrame:
    X = {"numerical": df[numerical_features].values}
    for c in categorical_features:
        X[c] = df[c]
    return X


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
            c for c in tr_x.columns if c not in self.categorical_features  # type: ignore
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
            inp_cat = layers.Input(shape=[1], name=c)
            inp_cats.append(inp_cat)
            embs.append((layers.Embedding(data[c].max() + 1, 4)(inp_cat)))

        cats = layers.Flatten()(layers.concatenate(embs))
        cats = layers.Dense(4, activation="linear")(cats)

        inp_numerical = layers.Input(shape=[len(numerical_features)], name="numerical")
        nums = layers.Dense(500, activation="relu")(inp_numerical)
        nums = layers.BatchNormalization()(nums)
        x = layers.Dropout(dropout)(nums)

        x = layers.concatenate([nums, cats])
        x = layers.Dense(40, activation="relu")(x)
        out = layers.Dense(31, activation="linear", name="out1")(x)

        model = kerasModel(inputs=[inp_numerical], outputs=out)
        model.compile(
            optimizer="adam", loss="mse", metrics=[keras.metrics.RootMeanSquaredError()]
        )
        batch_size = 256

        tr_x = get_keras_data(tr_x, numerical_features, self.categorical_features)
        va_x = get_keras_data(va_x, numerical_features, self.categorical_features)

        if validation:
            early_stopping = keras.callbacks.EarlyStopping(
                patience=patience, min_delta=0.001, restore_best_weights=True,
            )
            model.fit(
                tr_x,
                tr_y,
                validation_data=(va_x, va_y),
                batch_size=batch_size,
                epochs=nb_epoch,
                callbacks=[early_stopping],
            )
        else:
            model.fit(tr_x, tr_y, batch_size=batch_size, nb_epoch=nb_epoch)
        model.load_weights(f"../output/model/model_{self.run_fold_name}.hdf5")  # type: ignore

        # モデル・スケーラーの保持
        self.model = model

    def predict(self, te_x: pd.DataFrame) -> np.ndarray:
        numerical_features = [
            c for c in te_x.columns if c not in self.categorical_features  # type: ignore
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
