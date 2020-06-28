import os

from keras import backend as K
from keras.callbacks import EarlyStopping, Callback, ModelCheckpoint
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.layers.core import Dense, Dropout, Reshape
from keras.layers import (Input, Embedding, Flatten, concatenate, Multiply,
                          Conv1D, GlobalMaxPool1D,
                          Bidirectional, TimeDistributed, SpatialDropout1D, GRU)
from keras.layers.normalization import BatchNormalization
from keras.models import load_model
from keras.models import Model as kerasModel
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
# from sklearn.preprocessing import QuantileTransformer
import tensorflow as tf

from ayniy.model.model import Model as oriModel
from ayniy.preprocessing import standerize, fillna

# tensorflowの警告抑制
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def rmse(y, y_pred):
    return K.sqrt(K.mean(K.square(y - y_pred)))


def prauc(y, y_pred):
    return tf.py_func(average_precision_score, (y, y_pred), tf.double)


def get_keras_data(df, numerical_features, categorical_features):
    X = {"numerical": df[numerical_features].values}
    for c in categorical_features:
        X[c] = df[c]
    return X


class CyclicLR(Callback):
    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular',
                 gamma=1., scale_fn=None, scale_mode='cycle'):
        super(CyclicLR, self).__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn is None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1 / (2. ** (x - 1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma ** (x)
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr is not None:
            self.base_lr = new_base_lr
        if new_max_lr is not None:
            self.max_lr = new_max_lr
        if new_step_size is not None:
            self.step_size = new_step_size
        self.clr_iterations = 0.

    def clr(self):
        cycle = np.floor(1 + self.clr_iterations / (2 * self.step_size))
        x = np.abs(self.clr_iterations / self.step_size - 2 * cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * self.scale_fn(
                self.clr_iterations)

    def on_train_begin(self, logs={}):
        logs = logs or {}

        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())

    def on_batch_end(self, epoch, logs=None):

        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        K.set_value(self.model.optimizer.lr, self.clr())


def se_block(input, channels, r=8):
    x = Dense(channels // r, activation="relu")(input)
    x = Dense(channels, activation="sigmoid")(x)
    return Multiply()([input, x])


class ModelTNNRegressor(oriModel):

    def train(self, tr_x, tr_y, va_x=None, va_y=None, te_x=None):

        # データのセット・スケーリング
        numerical_features = [c for c in tr_x.columns if c not in self.categorical_features]
        validation = va_x is not None

        # パラメータ
        dropout = self.params['dropout']
        nb_epoch = self.params['nb_epoch']
        patience = self.params['patience']

        # モデルの構築
        inp_cats = []
        embs = []
        data = pd.concat([tr_x, va_x, te_x]).reset_index(drop=True)

        for c in self.categorical_features:
            inp_cat = Input(shape=[1], name=c)
            inp_cats.append(inp_cat)
            embs.append((Embedding(data[c].max() + 1, 4)(inp_cat)))

        cats = Flatten()(concatenate(embs))
        cats = Dense(4, activation="linear")(cats)
        cats = BatchNormalization()(cats)
        cats = PReLU()(cats)

        inp_numerical = Input(shape=[len(numerical_features)], name="numerical")
        nums = Dense(32, activation="linear")(inp_numerical)
        nums = BatchNormalization()(nums)
        nums = PReLU()(nums)
        nums = Dropout(dropout)(nums)

        x = concatenate([nums, cats])
        x = se_block(x, 32 + 4)
        x = BatchNormalization()(x)
        x = Dropout(dropout / 2)(x)
        x = Dense(1000, activation="relu")(x)
        x = Dense(800, activation="relu")(x)
        x = Dense(300, activation="relu")(x)
        out = Dense(1, activation="linear", name="out1")(x)

        model = kerasModel(inputs=inp_cats + [inp_numerical], outputs=out)
        # model.compile(loss='mean_absolute_error', optimizer='adam')
        model.compile(loss=rmse, optimizer='adam')
        # print(model.summary())
        n_train = len(tr_x)
        batch_size_nn = 256

        # preprocessing
        tr_x, va_x = standerize(tr_x, va_x, {'encode_col': numerical_features})
        # prep = QuantileTransformer(output_distribution="normal")
        # tr_x.loc[:, numerical_features] = prep.fit_transform(tr_x.loc[:, numerical_features])
        # va_x.loc[:, numerical_features] = prep.transform(va_x.loc[:, numerical_features])
        tr_x, va_x = fillna(tr_x, va_x, {'encode_col': tr_x.columns}, {'how': 'median'})

        tr_x = get_keras_data(tr_x, numerical_features, self.categorical_features)
        va_x = get_keras_data(va_x, numerical_features, self.categorical_features)

        clr_tri = CyclicLR(base_lr=1e-5, max_lr=1e-2, step_size=n_train // batch_size_nn, mode="triangular2")
        ckpt = ModelCheckpoint(f'../output/model/model_{self.run_fold_name}.hdf5', save_best_only=True,
                               monitor='val_loss', mode='min')
        if validation:
            early_stopping = EarlyStopping(monitor='val_loss', patience=patience,
                                           verbose=1, restore_best_weights=True)
            model.fit(tr_x, tr_y, epochs=nb_epoch, batch_size=batch_size_nn, verbose=2,
                      validation_data=(va_x, va_y), callbacks=[ckpt, clr_tri, early_stopping])
        else:
            model.fit(tr_x, tr_y, nb_epoch=nb_epoch, batch_size=batch_size_nn, verbose=2)
        model.load_weights(f'../output/model/model_{self.run_fold_name}.hdf5')

        # モデル・スケーラーの保持
        self.model = model

    def predict(self, te_x):
        numerical_features = [c for c in te_x.columns if c not in self.categorical_features]
        te_x = get_keras_data(te_x, numerical_features, self.categorical_features)
        pred = self.model.predict(te_x).reshape(-1, )
        return pred

    def save_model(self):
        model_path = os.path.join('../output/model', f'{self.run_fold_name}.h5')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.model.save(model_path)

    def load_model(self):
        model_path = os.path.join('../output/model', f'{self.run_fold_name}.h5')
        # self.model = load_model(model_path)
        self.model = load_model(model_path, custom_objects={'rmse': rmse})


class ModelTNNClassifier(oriModel):

    def train(self, tr_x, tr_y, va_x=None, va_y=None, te_x=None):

        # データのセット・スケーリング
        numerical_features = [c for c in tr_x.columns if c not in self.categorical_features]
        validation = va_x is not None

        # パラメータ
        dropout = self.params['dropout']
        nb_epoch = self.params['nb_epoch']
        patience = self.params['patience']

        # モデルの構築
        inp_cats = []
        embs = []
        data = pd.concat([tr_x, va_x, te_x]).reset_index(drop=True)

        for c in self.categorical_features:
            inp_cat = Input(shape=[1], name=c)
            inp_cats.append(inp_cat)
            embs.append((Embedding(data[c].max() + 1, 4)(inp_cat)))
        cats = Flatten()(concatenate(embs))
        cats = Dense(4, activation="linear")(cats)
        cats = BatchNormalization()(cats)
        cats = PReLU()(cats)

        inp_numerical = Input(shape=[len(numerical_features)], name="numerical")
        nums = Dense(32, activation="linear")(inp_numerical)
        nums = BatchNormalization()(nums)
        nums = PReLU()(nums)
        nums = Dropout(dropout)(nums)

        x = concatenate([nums, cats])
        x = se_block(x, 32 + 4)
        x = BatchNormalization()(x)
        x = Dropout(dropout / 2)(x)
        x = Dense(1000, activation="relu")(x)
        x = Dense(800, activation="relu")(x)
        x = Dense(300, activation="relu")(x)
        out = Dense(1, activation="sigmoid", name="out1")(x)

        model = kerasModel(inputs=inp_cats + [inp_numerical], outputs=out)
        model.compile(loss='binary_crossentropy', optimizer='adam')
        # print(model.summary())
        n_train = len(tr_x)
        batch_size_nn = 256

        tr_x = get_keras_data(tr_x, numerical_features, self.categorical_features)
        va_x = get_keras_data(va_x, numerical_features, self.categorical_features)

        clr_tri = CyclicLR(base_lr=1e-5, max_lr=1e-2, step_size=n_train // batch_size_nn, mode="triangular2")
        ckpt = ModelCheckpoint(f'../output/model/model_{self.run_fold_name}.hdf5', save_best_only=True,
                               monitor='val_loss', mode='min')
        if validation:
            early_stopping = EarlyStopping(monitor='val_loss', patience=patience,
                                           verbose=1, restore_best_weights=True)
            model.fit(tr_x, tr_y, epochs=nb_epoch, batch_size=batch_size_nn, verbose=2,
                      validation_data=(va_x, va_y), callbacks=[ckpt, clr_tri, early_stopping])
        else:
            model.fit(tr_x, tr_y, nb_epoch=nb_epoch, batch_size=batch_size_nn, verbose=2)
        model.load_weights(f'../output/model/model_{self.run_fold_name}.hdf5')

        # モデル・スケーラーの保持
        self.model = model

    def predict(self, te_x):
        numerical_features = [c for c in te_x.columns if c not in self.categorical_features]
        te_x = get_keras_data(te_x, numerical_features, self.categorical_features)
        pred = self.model.predict(te_x).reshape(-1, )
        return pred

    def save_model(self):
        model_path = os.path.join('../output/model', f'{self.run_fold_name}.h5')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.model.save(model_path)

    def load_model(self):
        model_path = os.path.join('../output/model', f'{self.run_fold_name}.h5')
        self.model = load_model(model_path)


class ModelCNNClasifier(oriModel):

    def train(self, tr_x, tr_y, va_x=None, va_y=None, te_x=None):
        audio_features = [c for c in tr_x.columns if "spec" in c]

        # データのセット・スケーリング
        numerical_features = [c for c in tr_x.columns if (c not in audio_features)]
        validation = va_x is not None

        # パラメータ
        dropout = self.params['dropout']
        nb_epoch = self.params['nb_epoch']
        patience = self.params['patience']

        # モデルの構築
        inp_numerical = Input(shape=[len(numerical_features)], name="numerical")
        nums = Dense(32, activation="linear")(inp_numerical)
        nums = BatchNormalization()(nums)
        nums = PReLU()(nums)
        nums = Dropout(dropout)(nums)

        # https://www.kaggle.com/yuval6967/3rd-place-cnn
        inp_audio = Input(shape=[512], name="audio")
        audio = Reshape((512, 1))(inp_audio)
        audio = Conv1D(256, 32, padding='same', name='Conv1')(audio)
        audio = BatchNormalization()(audio)
        audio = LeakyReLU(alpha=0.1)(audio)
        audio = Dropout(0.2)(audio)
        audio = Conv1D(256, 24, padding='same', name='Conv2')(audio)
        audio = BatchNormalization()(audio)
        audio = LeakyReLU(alpha=0.1)(audio)
        audio = Dropout(0.2)(audio)
        audio = Conv1D(128, 16, padding='same', name='Conv3')(audio)
        audio = BatchNormalization()(audio)
        audio = LeakyReLU(alpha=0.1)(audio)
        audio = GlobalMaxPool1D()(audio)
        audio = Dropout(dropout)(audio)

        x = concatenate([nums, audio])
        x = BatchNormalization()(x)
        x = Dropout(dropout / 2)(x)
        out = Dense(1, activation="sigmoid", name="out1")(x)

        model = kerasModel(inputs=[inp_numerical] + [inp_audio], outputs=out)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[prauc])

        # print(model.summary())
        n_train = len(tr_x)
        batch_size_nn = 512

        tr_x = get_keras_data(tr_x, numerical_features, audio_features)
        va_x = get_keras_data(va_x, numerical_features, audio_features)

        clr_tri = CyclicLR(base_lr=1e-5, max_lr=1e-2, step_size=n_train // batch_size_nn, mode="triangular2")
        ckpt = ModelCheckpoint(f'../output/model/model_{self.run_fold_name}.hdf5', save_best_only=True,
                               monitor='val_loss', mode='min')
        if validation:
            early_stopping = EarlyStopping(monitor='val_loss', patience=patience,
                                           verbose=1, restore_best_weights=True)
            model.fit(tr_x, tr_y, epochs=nb_epoch, batch_size=batch_size_nn, verbose=2,
                      validation_data=(va_x, va_y), callbacks=[ckpt, clr_tri, early_stopping])
        else:
            model.fit(tr_x, tr_y, nb_epoch=nb_epoch, batch_size=batch_size_nn, verbose=2)
        model.load_weights(f'../output/model/model_{self.run_fold_name}.hdf5')

        # モデル・スケーラーの保持
        self.model = model

    def predict(self, te_x):
        audio_features = [c for c in te_x.columns if "spec" in c]
        numerical_features = [c for c in te_x.columns if (c not in audio_features)]
        te_x = get_keras_data(te_x, numerical_features, audio_features)
        pred = self.model.predict(te_x).reshape(-1, )
        return pred

    def save_model(self):
        model_path = os.path.join('../output/model', f'{self.run_fold_name}.h5')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.model.save(model_path)

    def load_model(self):
        model_path = os.path.join('../output/model', f'{self.run_fold_name}.h5')
        self.model = load_model(model_path, custom_objects={'prauc': prauc})


class ModelRNNClasifier(oriModel):

    def train(self, tr_x, tr_y, va_x=None, va_y=None, te_x=None):
        audio_features = [c for c in tr_x.columns if "spec" in c]

        # データのセット・スケーリング
        numerical_features = [c for c in tr_x.columns if (c not in self.categorical_features) and (c not in audio_features)]
        validation = va_x is not None

        # パラメータ
        dropout = self.params['dropout']
        nb_epoch = self.params['nb_epoch']
        patience = self.params['patience']

        # モデルの構築
        inp_cats = []
        embs = []
        data = pd.concat([tr_x, va_x, te_x]).reset_index(drop=True)

        for c in self.categorical_features:
            inp_cat = Input(shape=[1], name=c)
            inp_cats.append(inp_cat)
            embs.append((Embedding(data[c].max() + 1, 4)(inp_cat)))
        cats = Flatten()(concatenate(embs))
        cats = Dense(10, activation="linear")(cats)
        cats = BatchNormalization()(cats)
        cats = PReLU()(cats)

        inp_numerical = Input(shape=[len(numerical_features)], name="numerical")
        nums = Dense(32, activation="linear")(inp_numerical)
        nums = BatchNormalization()(nums)
        nums = PReLU()(nums)
        nums = Dropout(dropout)(nums)

        # https://www.kaggle.com/zerrxy/plasticc-rnn
        inp_audio = Input(shape=[512], name="audio")
        audio = Reshape((512, 1))(inp_audio)

        audio = TimeDistributed(Dense(40, activation='relu'))(audio)
        audio = Bidirectional(GRU(80, return_sequences=True))(audio)
        audio = SpatialDropout1D(0.2)(audio)

        audio = GlobalMaxPool1D()(audio)
        audio = Dropout(dropout)(audio)

        x = concatenate([nums, cats, audio])
        x = BatchNormalization()(x)
        x = Dropout(dropout / 2)(x)
        out = Dense(1, activation="sigmoid", name="out1")(x)

        model = kerasModel(inputs=inp_cats + [inp_numerical] + [inp_audio], outputs=out)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[prauc])

        # print(model.summary())
        n_train = len(tr_x)
        batch_size_nn = 256

        tr_x = get_keras_data(tr_x, numerical_features, self.categorical_features, audio_features)
        va_x = get_keras_data(va_x, numerical_features, self.categorical_features, audio_features)

        clr_tri = CyclicLR(base_lr=1e-5, max_lr=1e-2, step_size=n_train // batch_size_nn, mode="triangular2")
        ckpt = ModelCheckpoint(f'../output/model/model_{self.run_fold_name}.hdf5', save_best_only=True,
                               monitor='val_loss', mode='min')
        if validation:
            early_stopping = EarlyStopping(monitor='val_loss', patience=patience,
                                           verbose=1, restore_best_weights=True)
            model.fit(tr_x, tr_y, epochs=nb_epoch, batch_size=batch_size_nn, verbose=2,
                      validation_data=(va_x, va_y), callbacks=[ckpt, clr_tri, early_stopping])
        else:
            model.fit(tr_x, tr_y, nb_epoch=nb_epoch, batch_size=batch_size_nn, verbose=2)
        model.load_weights(f'../output/model/model_{self.run_fold_name}.hdf5')

        # モデル・スケーラーの保持
        self.model = model

    def predict(self, te_x):
        audio_features = [c for c in te_x.columns if "spec" in c]
        numerical_features = [c for c in te_x.columns if (c not in self.categorical_features) and (c not in audio_features)]
        te_x = get_keras_data(te_x, numerical_features, self.categorical_features, audio_features)
        pred = self.model.predict(te_x).reshape(-1, )
        return pred

    def save_model(self):
        model_path = os.path.join('../output/model', f'{self.run_fold_name}.h5')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.model.save(model_path)

    def load_model(self):
        model_path = os.path.join('../output/model', f'{self.run_fold_name}.h5')
        self.model = load_model(model_path, custom_objects={'prauc': prauc})
