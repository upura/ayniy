import os
from keras.callbacks import EarlyStopping
from keras.layers.advanced_activations import PReLU
from keras.layers.core import Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential, load_model
from sklearn.preprocessing import StandardScaler

from ayniy.model import Model
from ayniy.utils import Data

# tensorflowの警告抑制
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class ModelNN(Model):

    def train(self, tr_x, tr_y, va_x=None, va_y=None):

        nb_classes = 1
        # データのセット・スケーリング
        validation = va_x is not None
        scaler = StandardScaler()
        scaler.fit(tr_x)
        tr_x = scaler.transform(tr_x)

        if validation:
            va_x = scaler.transform(va_x)

        # パラメータ
        layers = self.params['layers']
        dropout = self.params['dropout']
        units = self.params['units']
        nb_epoch = self.params['nb_epoch']
        patience = self.params['patience']

        # モデルの構築
        model = Sequential()
        model.add(Dense(units, input_shape=(tr_x.shape[1],)))
        model.add(PReLU())
        model.add(BatchNormalization())
        model.add(Dropout(dropout))

        for l in range(layers - 1):
            model.add(Dense(units))
            model.add(PReLU())
            model.add(BatchNormalization())
            model.add(Dropout(dropout))

        model.add(Dense(nb_classes))
        model.compile(loss='mean_squared_error', optimizer='adam')

        if validation:
            early_stopping = EarlyStopping(monitor='val_loss', patience=patience,
                                           verbose=1, restore_best_weights=True)
            model.fit(tr_x, tr_y, epochs=nb_epoch, batch_size=128, verbose=2,
                      validation_data=(va_x, va_y), callbacks=[early_stopping])
        else:
            model.fit(tr_x, tr_y, nb_epoch=nb_epoch, batch_size=128, verbose=2)

        # モデル・スケーラーの保持
        self.model = model
        self.scaler = scaler

    def predict(self, te_x):
        te_x = self.scaler.transform(te_x)
        pred = self.model.predict_proba(te_x)
        return pred

    def save_model(self):
        model_path = os.path.join('../output/model', f'{self.run_fold_name}.h5')
        scaler_path = os.path.join('../output/model', f'{self.run_fold_name}-scaler.pkl')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.model.save(model_path)
        Data.dump(self.scaler, scaler_path)

    def load_model(self):
        model_path = os.path.join('../output/model', f'{self.run_fold_name}.h5')
        scaler_path = os.path.join('../output/model', f'{self.run_fold_name}-scaler.pkl')
        self.model = load_model(model_path)
        self.scaler = Data.load(scaler_path)
