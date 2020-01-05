import os
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping
from keras.layers import Dense
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import StandardScaler

from ayniy.model import Model as oriModel
from ayniy.utils import Data

# tensorflowの警告抑制
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class ModelMNN(oriModel):

    def train(self, tr_x, tr_y, va_x=None, va_y=None):

        # データのセット・スケーリング
        validation = va_x is not None
        scaler = StandardScaler()

        # パラメータ
        nb_epoch = self.params['nb_epoch']
        patience = self.params['patience']

        # モデルの構築
        model = Sequential()
        model.add(Dense(1000, input_dim=len(tr_x.columns), activation='relu'))
        model.add(Dense(800, activation='relu'))
        model.add(Dense(300, activation='relu'))
        model.add(Dense(1))

        model.compile(Adam(lr=0.001), loss="mean_absolute_error")

        ckpt = ModelCheckpoint(f'../output/model/model_{self.run_fold_name}.hdf5', save_best_only=True,
                               monitor='val_loss', mode='min')
        if validation:
            early_stopping = EarlyStopping(monitor='val_loss', patience=patience,
                                           verbose=1, restore_best_weights=True)
            model.fit(tr_x, tr_y, epochs=nb_epoch, batch_size=128, verbose=2,
                      validation_data=(va_x, va_y), callbacks=[ckpt, early_stopping])
        else:
            model.fit(tr_x, tr_y, nb_epoch=nb_epoch, batch_size=128, verbose=2)
        model.load_weights(f'../output/model/model_{self.run_fold_name}.hdf5')

        # モデル・スケーラーの保持
        self.model = model
        self.scaler = scaler

    def predict(self, te_x):
        pred = self.model.predict(te_x)
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
